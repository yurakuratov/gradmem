import logging
import math
import warnings
from types import SimpleNamespace

import torch
from torch import nn
from tqdm.auto import tqdm

from grad_memgpt import GradMemGPT, GradMemGPTConfig, _is_main_process, get_backbone

try:
    from fla.layers.mamba2 import Mamba2 as FlaMamba2
    from fla.models.utils import Cache as FlaCache
    _fla_import_error = None
except (ImportError, RuntimeError) as exc:
    FlaMamba2 = None
    FlaCache = None
    _fla_import_error = exc


logger = logging.getLogger(__name__)


class IdentityEnergyEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = int(input_size)

    def forward(self, hidden, state=None):
        return hidden, state


class FlaMamba2EnergyEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        state_size=128,
        conv_kernel=4,
        expand=2,
        head_dim=64,
        chunk_size=256,
        backend="cuda",
    ):
        super().__init__()
        if FlaMamba2 is None:
            raise ImportError(
                "energy_model_type='mamba2' requires a working flash-linear-attention package (`fla`)"
            ) from _fla_import_error
        self.input_size = int(input_size)
        self.state_size = int(state_size)
        self.conv_kernel = int(conv_kernel)
        self.expand = int(expand)
        self.head_dim = int(head_dim)
        self.num_heads = self.expand * self.input_size // self.head_dim
        self.conv_dim = self.expand * self.input_size + 2 * self.state_size
        self.mamba = FlaMamba2(
            hidden_size=self.input_size,
            state_size=int(state_size),
            conv_kernel=int(conv_kernel),
            expand=int(expand),
            head_dim=int(head_dim),
            chunk_size=int(chunk_size),
            layer_idx=0,
            backend=backend,
        )

    @staticmethod
    def _clone_state(state):
        if state is None:
            return None
        return {name: tensor.clone() for name, tensor in state.items()}

    def forward(self, hidden, state=None):
        legacy_state = None if state is None else [self._clone_state(state)]
        cache = FlaCache.from_legacy_cache(legacy_state)

        # FLA accepts a complete sequence only while prefilling an empty cache.
        # Once recurrent state exists, feed one token at a time through its
        # supported cached-decoding path.
        chunks = (hidden,) if state is None else hidden.split(1, dim=1)
        outputs = []
        for chunk in chunks:
            encoded = self.mamba(
                chunk,
                past_key_values=cache,
                use_cache=True,
            )
            outputs.append(encoded[0] if isinstance(encoded, tuple) else encoded)

        next_layer_state = cache[0]
        next_state = {
            "conv_state": next_layer_state["conv_state"].clone(),
            "recurrent_state": next_layer_state["recurrent_state"].clone(),
        }
        return torch.cat(outputs, dim=1), next_state


class EnergyGradMemConfig(GradMemGPTConfig):
    model_type = "energy_gradmem"

    def __init__(
        self,
        inner_objective="neural",
        energy_hidden_size=None,
        energy_num_layers=2,
        energy_dropout=0.0,
        energy_future_mode="next_token",
        energy_ce_guidance=False,
        energy_ce_guidance_alpha=0.01,
        energy_inner_ce_weight=0.0,
        energy_weight_rms_reg=0.0,
        energy_weight_rms_threshold=0.0,
        energy_delta_reg=0.0,
        energy_delta_max=1.0,
        energy_replay_weight=0.0,
        energy_model_type="lstm",
        energy_segment_state_size=None,
        energy_mamba_state_size=128,
        energy_mamba_conv_kernel=4,
        energy_mamba_expand=2,
        energy_mamba_head_dim=64,
        energy_mamba_chunk_size=256,
        energy_mamba_backend="cuda",
        segment_write_mode="sequential",
        segment_size=None,
        memory_rotation="none",
        memory_rotation_angle=None,
        reading_optimization=False,
        K_read=1,
        read_lr=0.1,
        clip_read_norm=None,
        read_grad_mode="second",
        energy_pretrain_objective="ce",
        energy_pretrain_steps=0,
        energy_pretrain_batch_size=16,
        energy_pretrain_seq_len=32,
        energy_pretrain_lr=1e-3,
        energy_pretrain_seed=0,
        energy_pretrain_l2_reg=0.0,
        return_energy_state=False,
        **kwargs,
    ):
        legacy_energy_l2_reg = kwargs.pop("energy_l2_reg", None)
        super().__init__(**kwargs)
        if legacy_energy_l2_reg is not None:
            if float(energy_weight_rms_reg) != 0.0:
                raise ValueError(
                    "energy_l2_reg and energy_weight_rms_reg cannot both be set; "
                    "use energy_weight_rms_reg"
                )
            warnings.warn(
                "energy_l2_reg is deprecated and now configures energy_weight_rms_reg "
                "with energy_weight_rms_threshold=0",
                DeprecationWarning,
                stacklevel=2,
            )
            energy_weight_rms_reg = legacy_energy_l2_reg
            energy_weight_rms_threshold = 0.0
        if inner_objective == "lstm":
            warnings.warn(
                "inner_objective='lstm' is deprecated; use inner_objective='neural' with energy_model_type='lstm' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            inner_objective = "neural"
        if inner_objective not in ("neural", "cross_entropy", "embedding_l1", "embedding_l2"):
            raise ValueError("inner_objective must be one of: 'neural', 'cross_entropy', 'embedding_l1', 'embedding_l2'")
        if energy_future_mode not in ("none", "next_token"):
            raise ValueError("energy_future_mode must be one of: 'none', 'next_token'")
        if energy_model_type not in ("lstm", "identity", "mamba2", "segment_delta_gru"):
            raise ValueError(
                "energy_model_type must be one of: 'lstm', 'identity', 'mamba2', 'segment_delta_gru'"
            )
        if segment_write_mode not in ("sequential", "parallel"):
            raise ValueError("segment_write_mode must be one of: 'sequential', 'parallel'")
        if segment_size is not None and int(segment_size) <= 0:
            raise ValueError("segment_size must be a positive integer when set")
        if memory_rotation not in ("none", "pairwise"):
            raise ValueError("memory_rotation must be one of: 'none', 'pairwise'")
        if memory_rotation_angle is not None and not math.isfinite(float(memory_rotation_angle)):
            raise ValueError("memory_rotation_angle must be finite when set")
        if memory_rotation == "none" and memory_rotation_angle is not None:
            raise ValueError("memory_rotation_angle requires memory_rotation='pairwise'")
        if memory_rotation == "pairwise":
            if self.memory_backend != "prefix":
                raise ValueError("memory_rotation='pairwise' currently supports memory_backend='prefix' only")
            if segment_write_mode != "sequential":
                raise ValueError("memory_rotation='pairwise' currently supports segment_write_mode='sequential' only")
            if self.use_adam:
                raise ValueError("memory_rotation='pairwise' currently requires use_adam=False")
        if energy_model_type == "segment_delta_gru":
            if inner_objective != "neural":
                raise ValueError("energy_model_type='segment_delta_gru' requires inner_objective='neural'")
            if segment_write_mode != "sequential":
                raise ValueError("energy_model_type='segment_delta_gru' requires segment_write_mode='sequential'")
            if self.memory_backend != "prefix":
                raise ValueError("energy_model_type='segment_delta_gru' requires memory_backend='prefix'")
            if int(energy_pretrain_steps) != 0:
                raise ValueError("energy_model_type='segment_delta_gru' does not support energy pretraining")
        if inner_objective in ("embedding_l1", "embedding_l2") and energy_future_mode != "next_token":
            raise ValueError("embedding_l1/embedding_l2 inner objectives require energy_future_mode='next_token'")
        if inner_objective == "cross_entropy" and energy_ce_guidance:
            raise ValueError("inner_objective='cross_entropy' does not support energy_ce_guidance")
        if int(K_read) < 0:
            raise ValueError("K_read must be a non-negative integer")
        if not math.isfinite(float(read_lr)) or float(read_lr) < 0.0:
            raise ValueError("read_lr must be finite and non-negative")
        if clip_read_norm is not None and (
            not math.isfinite(float(clip_read_norm)) or float(clip_read_norm) <= 0.0
        ):
            raise ValueError("clip_read_norm must be finite and positive when provided")
        if read_grad_mode not in ("first", "second"):
            raise ValueError("read_grad_mode must be one of: first, second")
        if reading_optimization:
            if inner_objective != "neural":
                raise ValueError("reading_optimization requires inner_objective='neural'")
            if energy_future_mode != "none":
                raise ValueError("reading_optimization requires energy_future_mode='none'")
            if int(K_read) < 1:
                raise ValueError("reading_optimization requires K_read >= 1")
            if float(read_lr) <= 0.0:
                raise ValueError("reading_optimization requires read_lr > 0")
        regularization_values = {
            "energy_weight_rms_reg": energy_weight_rms_reg,
            "energy_weight_rms_threshold": energy_weight_rms_threshold,
            "energy_delta_reg": energy_delta_reg,
            "energy_delta_max": energy_delta_max,
            "energy_replay_weight": energy_replay_weight,
        }
        for name, value in regularization_values.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if inner_objective != "neural" and float(energy_weight_rms_reg) != 0.0:
            raise ValueError("energy_weight_rms_reg requires inner_objective='neural'")
        if float(energy_replay_weight) != 0.0 and energy_model_type != "segment_delta_gru":
            raise ValueError("energy_replay_weight requires energy_model_type='segment_delta_gru'")
        if float(energy_delta_reg) != 0.0:
            if segment_write_mode != "sequential":
                raise ValueError("energy_delta_reg requires segment_write_mode='sequential'")
            if self.memory_backend != "prefix":
                raise ValueError("energy_delta_reg requires memory_backend='prefix'")
            if self.grad_mode != "second":
                raise ValueError("energy_delta_reg requires grad_mode='second'")
            if self.use_adam:
                raise ValueError("energy_delta_reg requires use_adam=False")
        if inner_objective != "cross_entropy" and self.use_write_head:
            raise ValueError("EnergyGradMem supports use_write_head=True only with inner_objective='cross_entropy'")
        if inner_objective == "cross_entropy":
            if energy_model_type != "lstm":
                raise ValueError("inner_objective='cross_entropy' does not use energy_model_type; leave energy_model_type='lstm'")
            if float(energy_inner_ce_weight) != 0.0:
                raise ValueError("inner_objective='cross_entropy' does not use energy_inner_ce_weight; leave it at 0.0")
            if int(energy_pretrain_steps) != 0:
                raise ValueError("inner_objective='cross_entropy' does not support energy pretraining")
            if energy_pretrain_objective != "ce":
                raise ValueError("inner_objective='cross_entropy' does not use energy_pretrain_objective; leave it at 'ce'")
            if float(energy_pretrain_l2_reg) != 0.0:
                raise ValueError("inner_objective='cross_entropy' does not use energy_pretrain_l2_reg; leave it at 0.0")
        if segment_write_mode == "parallel":
            if self.memory_backend != "prefix":
                raise ValueError("segment_write_mode='parallel' currently supports memory_backend='prefix' only")
            if inner_objective != "cross_entropy" and energy_future_mode != "none":
                raise ValueError("segment_write_mode='parallel' requires energy_future_mode='none'")
            if energy_ce_guidance:
                raise ValueError("segment_write_mode='parallel' does not support energy_ce_guidance")
        if energy_pretrain_objective not in ("ce", "embedding_mean_abs_diff", "embedding_l2"):
            raise ValueError("energy_pretrain_objective must be one of: 'ce', 'embedding_mean_abs_diff', 'embedding_l2'")
        self.inner_objective = inner_objective
        self.energy_hidden_size = energy_hidden_size
        self.energy_num_layers = energy_num_layers
        self.energy_dropout = energy_dropout
        self.energy_future_mode = energy_future_mode
        self.energy_ce_guidance = energy_ce_guidance
        self.energy_ce_guidance_alpha = energy_ce_guidance_alpha
        self.energy_inner_ce_weight = energy_inner_ce_weight
        self.energy_weight_rms_reg = energy_weight_rms_reg
        self.energy_weight_rms_threshold = energy_weight_rms_threshold
        self.energy_delta_reg = energy_delta_reg
        self.energy_delta_max = energy_delta_max
        self.energy_replay_weight = energy_replay_weight
        self.energy_model_type = energy_model_type
        self.energy_segment_state_size = energy_segment_state_size
        self.energy_mamba_state_size = energy_mamba_state_size
        self.energy_mamba_conv_kernel = energy_mamba_conv_kernel
        self.energy_mamba_expand = energy_mamba_expand
        self.energy_mamba_head_dim = energy_mamba_head_dim
        self.energy_mamba_chunk_size = energy_mamba_chunk_size
        self.energy_mamba_backend = energy_mamba_backend
        self.segment_write_mode = segment_write_mode
        self.segment_size = segment_size
        self.memory_rotation = memory_rotation
        self.memory_rotation_angle = memory_rotation_angle
        self.reading_optimization = reading_optimization
        self.K_read = K_read
        self.read_lr = read_lr
        self.clip_read_norm = clip_read_norm
        self.read_grad_mode = read_grad_mode
        self.energy_pretrain_objective = energy_pretrain_objective
        self.energy_pretrain_steps = energy_pretrain_steps
        self.energy_pretrain_batch_size = energy_pretrain_batch_size
        self.energy_pretrain_seq_len = energy_pretrain_seq_len
        self.energy_pretrain_lr = energy_pretrain_lr
        self.energy_pretrain_seed = energy_pretrain_seed
        self.energy_pretrain_l2_reg = energy_pretrain_l2_reg
        self.return_energy_state = return_energy_state


class EnergyGradMem(GradMemGPT):
    config_class = EnergyGradMemConfig

    def __init__(self, config):
        super().__init__(config)

        model_hidden_size = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", None))
        if model_hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")

        energy_hidden_size = config.energy_hidden_size or model_hidden_size
        energy_num_layers = int(config.energy_num_layers)
        if energy_num_layers < 1:
            raise ValueError("energy_num_layers must be >= 1")

        self.inner_objective = config.inner_objective
        self.energy_hidden_size = int(energy_hidden_size)
        self.energy_num_layers = energy_num_layers
        self.energy_future_mode = config.energy_future_mode
        self.energy_ce_guidance = bool(getattr(config, "energy_ce_guidance", False))
        self.energy_ce_guidance_alpha = float(getattr(config, "energy_ce_guidance_alpha", 0.01))
        self.energy_inner_ce_weight = float(getattr(config, "energy_inner_ce_weight", 0.0))
        self.energy_weight_rms_reg = float(getattr(config, "energy_weight_rms_reg", 0.0))
        self.energy_weight_rms_threshold = float(getattr(config, "energy_weight_rms_threshold", 0.0))
        self.energy_delta_reg = float(getattr(config, "energy_delta_reg", 0.0))
        self.energy_delta_max = float(getattr(config, "energy_delta_max", 1.0))
        self.energy_replay_weight = float(getattr(config, "energy_replay_weight", 0.0))
        self.energy_model_type = getattr(config, "energy_model_type", "lstm")
        energy_segment_state_size = getattr(config, "energy_segment_state_size", None)
        self.energy_segment_state_size = int(
            model_hidden_size if energy_segment_state_size is None else energy_segment_state_size
        )
        if self.energy_segment_state_size < 1:
            raise ValueError("energy_segment_state_size must be a positive integer")
        self.segment_write_mode = getattr(config, "segment_write_mode", "sequential")
        self.segment_size = getattr(config, "segment_size", None)
        if self.segment_size is not None:
            self.segment_size = int(self.segment_size)
        self.memory_rotation = getattr(config, "memory_rotation", "none")
        memory_rotation_angle = getattr(config, "memory_rotation_angle", None)
        self.memory_rotation_angle = None if memory_rotation_angle is None else float(memory_rotation_angle)
        self.reading_optimization = bool(getattr(config, "reading_optimization", False))
        self.K_read = int(getattr(config, "K_read", 1))
        self.read_lr = float(getattr(config, "read_lr", 0.1))
        clip_read_norm = getattr(config, "clip_read_norm", None)
        self.clip_read_norm = None if clip_read_norm is None else float(clip_read_norm)
        self.read_grad_mode = getattr(config, "read_grad_mode", "second")
        self.energy_pretrain_objective = getattr(config, "energy_pretrain_objective", "ce")
        self.energy_pretrain_steps = int(getattr(config, "energy_pretrain_steps", 0))
        self.energy_pretrain_batch_size = int(getattr(config, "energy_pretrain_batch_size", 16))
        self.energy_pretrain_seq_len = int(getattr(config, "energy_pretrain_seq_len", 32))
        self.energy_pretrain_lr = float(getattr(config, "energy_pretrain_lr", 1e-3))
        self.energy_pretrain_seed = int(getattr(config, "energy_pretrain_seed", 0))
        self.energy_pretrain_l2_reg = float(getattr(config, "energy_pretrain_l2_reg", 0.0))
        self.return_energy_state = bool(getattr(config, "return_energy_state", False))
        dropout = float(config.energy_dropout) if energy_num_layers > 1 else 0.0
        energy_input_size = model_hidden_size * 2 if self.energy_future_mode == "next_token" else model_hidden_size
        self.energy_input_size = int(energy_input_size)

        if self.inner_objective == "neural":
            if self.energy_model_type == "lstm":
                self.energy_encoder = nn.LSTM(
                    input_size=energy_input_size,
                    hidden_size=self.energy_hidden_size,
                    num_layers=energy_num_layers,
                    dropout=dropout,
                    batch_first=True,
                )
                energy_head_input_size = self.energy_hidden_size
            elif self.energy_model_type == "identity":
                self.energy_encoder = IdentityEnergyEncoder(energy_input_size)
                energy_head_input_size = energy_input_size
            elif self.energy_model_type == "mamba2":
                mamba_expand = int(getattr(config, "energy_mamba_expand", 2))
                mamba_head_dim = int(getattr(config, "energy_mamba_head_dim", 64))
                if mamba_expand <= 0 or mamba_head_dim <= 0:
                    raise ValueError("energy_mamba_expand and energy_mamba_head_dim must be positive integers")
                if (mamba_expand * self.energy_input_size) % mamba_head_dim != 0:
                    raise ValueError(
                        "Invalid Mamba2 energy dimensions: "
                        "energy_mamba_expand * energy_input_size must be divisible by energy_mamba_head_dim; "
                        f"energy_mamba_expand={mamba_expand}, "
                        f"energy_input_size={self.energy_input_size}, "
                        f"energy_mamba_head_dim={mamba_head_dim}"
                    )
                self.energy_encoder = FlaMamba2EnergyEncoder(
                    energy_input_size,
                    state_size=getattr(config, "energy_mamba_state_size", 128),
                    conv_kernel=getattr(config, "energy_mamba_conv_kernel", 4),
                    expand=mamba_expand,
                    head_dim=mamba_head_dim,
                    chunk_size=getattr(config, "energy_mamba_chunk_size", 256),
                    backend=getattr(config, "energy_mamba_backend", "cuda"),
                )
                energy_head_input_size = energy_input_size
            elif self.energy_model_type == "segment_delta_gru":
                memory_width = self.mem.size(-1)
                self.segment_state_gru = nn.GRUCell(
                    input_size=self.n_mem_tokens * memory_width,
                    hidden_size=self.energy_segment_state_size,
                )
                self.token_energy_mlp = nn.Sequential(
                    nn.Linear(
                        energy_input_size + self.energy_segment_state_size,
                        self.energy_hidden_size,
                    ),
                    nn.SiLU(),
                    nn.Linear(self.energy_hidden_size, 1),
                    nn.Softplus(beta=1, threshold=20),
                )
                with torch.no_grad():
                    self.token_energy_mlp[0].weight[:, energy_input_size:].zero_()
                if self.energy_replay_weight > 0.0:
                    self.replay_energy_mlp = nn.Sequential(
                        nn.Linear(
                            self.n_mem_tokens * memory_width + self.energy_segment_state_size,
                            self.energy_hidden_size,
                        ),
                        nn.SiLU(),
                        nn.Linear(self.energy_hidden_size, 1),
                        nn.Softplus(beta=1, threshold=20),
                    )
                energy_head_input_size = None
            else:
                raise ValueError(f"Unsupported energy_model_type={self.energy_model_type}")
            if energy_head_input_size is not None:
                self.energy_head = nn.Sequential(
                    nn.Linear(energy_head_input_size, 1),
                    nn.Softplus(beta=1, threshold=20),
                )

    def _named_energy_parameters(self):
        named_params = []
        seen = set()
        for module_name in (
            "segment_state_gru",
            "token_energy_mlp",
            "replay_energy_mlp",
            "energy_encoder",
            "energy_head",
        ):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for parameter_name, parameter in module.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                named_params.append((f"{module_name}.{parameter_name}", parameter))
        return named_params

    def _energy_parameters(self):
        return [parameter for _, parameter in self._named_energy_parameters()]

    def _energy_weight_parameters(self):
        return [
            parameter
            for name, parameter in self._named_energy_parameters()
            if "bias" not in name.rsplit(".", 1)[-1].lower()
        ]

    def _energy_dtype(self):
        params = self._energy_parameters()
        if params:
            return params[0].dtype
        return next(self.parameters()).dtype

    def set_energy_trainable(self, trainable=True):
        for param in self._energy_parameters():
            param.requires_grad = trainable

    def _context_segments(self, context_input_ids):
        if isinstance(context_input_ids, torch.Tensor):
            if context_input_ids.ndim != 2:
                raise ValueError(
                    "context_input_ids tensor must have shape [B, S]; "
                    f"got {tuple(context_input_ids.shape)}"
                )
            if self.segment_size is not None:
                if self.segment_size <= 0:
                    raise ValueError("segment_size must be a positive integer when set")
                if context_input_ids.size(1) % self.segment_size != 0:
                    raise ValueError(
                        "context_input_ids length must be divisible by segment_size; "
                        f"length={context_input_ids.size(1)}, segment_size={self.segment_size}"
                    )
                return list(context_input_ids.split(self.segment_size, dim=1))
            return [context_input_ids]
        if isinstance(context_input_ids, (list, tuple)):
            if len(context_input_ids) == 0:
                raise ValueError("context_input_ids segment list must be non-empty")
            for i, segment in enumerate(context_input_ids):
                if not isinstance(segment, torch.Tensor) or segment.ndim != 2:
                    raise ValueError(
                        "each context segment must be a tensor with shape [B, S]; "
                        f"segment={i}, type={type(segment)}, shape={getattr(segment, 'shape', None)}"
                    )
            return list(context_input_ids)
        raise ValueError("context_input_ids must be a tensor [B, S] or a list of tensors [B, S_i]")

    def _write_context_start(self, batch_ctx):
        if self.memory_backend == "prefix":
            return batch_ctx["mem_offset"]
        return 0

    def _energy_values(self, hidden, energy_state=None):
        if self.inner_objective in ("embedding_l1", "embedding_l2"):
            if hidden.size(-1) % 2 != 0:
                raise ValueError(f"{self.inner_objective} requires an even energy input size")
            pred, target = hidden.chunk(2, dim=-1)
            if self.inner_objective == "embedding_l1":
                return (pred - target).abs().mean(dim=-1), energy_state
            return (pred - target).pow(2).mean(dim=-1), energy_state

        if self.energy_model_type == "segment_delta_gru":
            if energy_state is None:
                raise ValueError("segment_delta_gru energy computation requires an initialized segment state")
            state_tokens = energy_state[:, None, :].expand(
                hidden.size(0),
                hidden.size(1),
                self.energy_segment_state_size,
            )
            conditioned = torch.cat((hidden, state_tokens.to(dtype=hidden.dtype)), dim=-1)
            return self.token_energy_mlp(conditioned).squeeze(-1), energy_state

        with torch.backends.cudnn.flags(enabled=False):
            encoded, energy_state = self.energy_encoder(hidden, energy_state)
        energy = self.energy_head(encoded).squeeze(-1)
        return energy, energy_state

    def _energy_loss(self, hidden, mask, energy_state):
        # Eval still needs inner-loop gradients. cuDNN RNN backward rejects eval-mode
        # modules, so use the native autograd path for this small objective model.
        previous_energy_state = energy_state
        energy, energy_state = self._energy_values(hidden, energy_state)

        if self.energy_model_type == "mamba2" and energy_state is not None:
            active_samples = mask.bool().any(dim=1)
            if previous_energy_state is None:
                energy_state = {
                    name: value * active_samples.view(
                        active_samples.size(0),
                        *([1] * (value.ndim - 1)),
                    ).to(value.dtype)
                    for name, value in energy_state.items()
                }
            else:
                energy_state = {
                    name: torch.where(
                        active_samples.view(
                            active_samples.size(0),
                            *([1] * (value.ndim - 1)),
                        ),
                        value,
                        previous_energy_state[name],
                    )
                    for name, value in energy_state.items()
                }

        mask = mask.to(dtype=energy.dtype)
        valid_lengths = mask.sum(dim=1)
        per_sample_energy = (energy * mask).sum(dim=1) / valid_lengths.clamp_min(1.0)
        per_sample_energy = per_sample_energy * (valid_lengths > 0).to(per_sample_energy.dtype)
        return per_sample_energy.sum(), energy_state, energy

    def _prepare_energy_state(self, energy_state, memory):
        batch_size = memory.size(0)
        if self.inner_objective != "neural":
            return energy_state

        if self.energy_model_type == "segment_delta_gru":
            expected_shape = (batch_size, self.energy_segment_state_size)
            if energy_state is None:
                return memory.new_zeros(expected_shape)
            if not isinstance(energy_state, torch.Tensor):
                raise ValueError(
                    "segment_delta_gru energy_state must be a tensor with shape "
                    f"{expected_shape}; got {type(energy_state).__name__}"
                )
            if tuple(energy_state.shape) != expected_shape:
                raise ValueError(
                    "segment_delta_gru energy_state must have shape "
                    f"{expected_shape}; got {tuple(energy_state.shape)}"
                )
            return energy_state.to(device=memory.device, dtype=memory.dtype)

        if self.energy_model_type == "lstm" and energy_state is not None:
            expected_shape = (self.energy_num_layers, batch_size, self.energy_hidden_size)
            if not isinstance(energy_state, tuple) or len(energy_state) != 2:
                raise ValueError(
                    "LSTM energy_state must be an (h, c) tuple with each tensor shaped "
                    f"{expected_shape}"
                )
            h, c = energy_state
            if not isinstance(h, torch.Tensor) or not isinstance(c, torch.Tensor):
                raise ValueError("LSTM energy_state must contain tensors")
            if tuple(h.shape) != expected_shape or tuple(c.shape) != expected_shape:
                raise ValueError(
                    "LSTM energy_state tensors must each have shape "
                    f"{expected_shape}; got h={tuple(h.shape)}, c={tuple(c.shape)}"
                )
            return (
                h.to(device=memory.device, dtype=self._energy_dtype()),
                c.to(device=memory.device, dtype=self._energy_dtype()),
            )

        if self.energy_model_type == "mamba2" and energy_state is not None:
            expected_shapes = {
                "conv_state": (
                    batch_size,
                    self.energy_encoder.conv_dim,
                    self.energy_encoder.conv_kernel,
                ),
                "recurrent_state": (
                    batch_size,
                    self.energy_encoder.num_heads,
                    self.energy_encoder.head_dim,
                    self.energy_encoder.state_size,
                ),
            }
            if not isinstance(energy_state, dict):
                raise ValueError(
                    "Mamba2 energy_state must be a dictionary containing "
                    "conv_state and recurrent_state tensors"
                )
            if set(energy_state) != set(expected_shapes):
                raise ValueError(
                    "Mamba2 energy_state must contain exactly the keys "
                    f"{sorted(expected_shapes)}; got {sorted(energy_state)}"
                )
            for name, expected_shape in expected_shapes.items():
                value = energy_state[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    actual_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                    raise ValueError(
                        f"Mamba2 energy_state[{name!r}] must have shape "
                        f"{expected_shape}; got {actual_shape}"
                    )
            return {
                name: value.to(device=memory.device)
                for name, value in energy_state.items()
            }

        return energy_state

    def _run_write_model(self, write_batch, memory_state):
        write_model_kwargs = write_batch.get("model_kwargs", {})
        with self.memory_backend_impl.activation_context(memory_state):
            if self.use_write_head:
                outs = get_backbone(self.model)(
                    inputs_embeds=write_batch["inputs_embeds"],
                    return_dict=True,
                    **write_model_kwargs,
                )
                hidden = outs.last_hidden_state[:, write_batch["logits_start"]:, :]
                return SimpleNamespace(logits=self.write_head(hidden), hidden_states=None, logits_start_override=0)
            outs = self.model(
                inputs_embeds=write_batch["inputs_embeds"],
                output_hidden_states=True,
                return_dict=True,
                **write_model_kwargs,
            )
        if outs.hidden_states is None:
            raise ValueError("Base model did not return hidden states for energy objective")
        return outs

    def _write_token_ce(self, write_out, write_batch):
        logits_start = getattr(write_out, "logits_start_override", write_batch["logits_start"])
        logits = write_out.logits[:, logits_start:, :]
        logits_loss = logits[:, :-1]
        label_shift = write_batch.get("label_shift", 0)
        labels_loss = write_batch["lm_labels"][:, label_shift:]
        mask_loss = write_batch["mask"][:, label_shift:]
        if logits_loss.size(1) != labels_loss.size(1) or labels_loss.size(1) != mask_loss.size(1):
            raise ValueError(
                "Invalid CE-guidance alignment: "
                f"logits_len={logits_loss.size(1)}, labels_len={labels_loss.size(1)}, mask_len={mask_loss.size(1)}"
            )
        token_ce = nn.functional.cross_entropy(
            logits_loss.reshape(-1, logits.size(-1)),
            labels_loss.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels_loss.size())
        if label_shift == 0:
            # Prefix write logits include the pre-context position. Energy at context
            # token t is conditioned on token t+1, so skip CE for token 0.
            token_ce = token_ce[:, 1:]
            mask_loss = mask_loss[:, 1:]
        return token_ce, mask_loss

    def _write_inner_ce_loss(self, write_out, write_batch):
        token_ce, mask_loss = self._write_inner_ce_tokens(write_out, write_batch)
        return self._masked_token_loss_sum(token_ce, mask_loss)

    def _write_inner_ce_tokens(self, write_out, write_batch):
        logits_start = getattr(write_out, "logits_start_override", write_batch["logits_start"])
        logits = write_out.logits[:, logits_start:, :]
        logits_loss = logits[:, :-1]
        label_shift = write_batch.get("label_shift", 0)
        labels_loss = write_batch["lm_labels"][:, label_shift:]
        mask_loss = write_batch["mask"][:, label_shift:]
        if logits_loss.size(1) != labels_loss.size(1) or labels_loss.size(1) != mask_loss.size(1) or labels_loss.size(1) == 0:
            raise ValueError(
                "Invalid inner CE alignment: "
                f"logits_len={logits_loss.size(1)}, labels_len={labels_loss.size(1)}, mask_len={mask_loss.size(1)}"
            )
        token_ce = nn.functional.cross_entropy(
            logits_loss.reshape(-1, logits.size(-1)),
            labels_loss.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels_loss.size())
        return token_ce, mask_loss

    def _write_parallel_inner_ce_loss(self, write_out, write_batch, batch_size, n_segments):
        token_ce, mask = self._write_inner_ce_tokens(write_out, write_batch)
        seq_len = token_ce.size(1)
        token_ce = token_ce.reshape(batch_size, n_segments * seq_len)
        mask = mask.reshape(batch_size, n_segments * seq_len)
        return self._masked_token_loss_sum(token_ce, mask)

    @staticmethod
    def _energy_ce_guidance_loss(energy, token_ce, mask):
        energy = energy[:, :token_ce.size(1)]
        mask = mask.to(dtype=energy.dtype)
        per_token = (energy - token_ce.detach().to(dtype=energy.dtype)).abs() * mask
        return per_token.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_token_loss_sum(token_loss, mask):
        mask = mask.to(dtype=token_loss.dtype)
        valid_lengths = mask.sum(dim=1)
        per_sample = (token_loss * mask).sum(dim=1) / valid_lengths.clamp_min(1.0)
        per_sample = per_sample * (valid_lengths > 0).to(per_sample.dtype)
        return per_sample.sum()

    def _sample_energy_pretrain_batch(self, batch_size, seq_len, generator, device):
        vocab_size = int(self.model.config.vocab_size)
        pad_id = self.model.config.pad_token_id
        low = 1 if vocab_size > 1 else 0
        tokens = torch.randint(low, vocab_size, (batch_size, seq_len), generator=generator, device=device)
        if pad_id is not None:
            tokens = torch.where(tokens == pad_id, (tokens + 1) % vocab_size, tokens)
        return tokens

    def _embedding_mean_abs_diff_pretrain_loss(self, batch_size, seq_len, generator, device):
        if self.inner_objective != "neural":
            raise ValueError("embedding_mean_abs_diff pretraining requires trainable inner_objective='neural'")
        input_size = self.energy_encoder.input_size
        if input_size % 2 != 0:
            raise ValueError("embedding_mean_abs_diff pretraining requires an even energy encoder input size")
        hidden_size = input_size // 2
        dtype = self._energy_dtype()
        h1 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        h2 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        energy_input = torch.cat([h1, h2], dim=-1)
        target = (h1 - h2).abs().mean(dim=-1)
        energy, _ = self._energy_values(energy_input)
        return (energy - target.detach()).pow(2).mean()

    def _embedding_l2_pretrain_loss(self, batch_size, seq_len, generator, device):
        if self.inner_objective != "neural":
            raise ValueError("embedding_l2 pretraining requires trainable inner_objective='neural'")
        input_size = self.energy_encoder.input_size
        if input_size % 2 != 0:
            raise ValueError("embedding_l2 pretraining requires an even energy encoder input size")
        hidden_size = input_size // 2
        dtype = self._energy_dtype()
        h1 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        h2 = torch.randn(batch_size, seq_len, hidden_size, generator=generator, device=device, dtype=dtype)
        energy_input = torch.cat([h1, h2], dim=-1)
        target = (h1 - h2).pow(2).mean(dim=-1)
        energy, _ = self._energy_values(energy_input)
        return (energy - target.detach()).pow(2).mean()

    def _ce_pretrain_loss(self, backend, pad_id, generator, device):
        context = self._sample_energy_pretrain_batch(
            self.energy_pretrain_batch_size,
            self.energy_pretrain_seq_len,
            generator,
            device,
        )
        memory_state, _ = backend.init_memory_state(context.size(0))
        batch_ctx = backend.prepare_batch(context, context, pad_id)
        write_batch = backend.build_write_inputs(memory_state, batch_ctx)

        with torch.no_grad():
            write_out = self._run_write_model(write_batch, memory_state)
            ctx_hidden = self._extract_context_hidden(write_out.hidden_states[-1], write_batch, batch_ctx)
            energy_input = self._energy_input(ctx_hidden, context, write_batch["mask"]).detach()
            token_ce, token_ce_mask = self._write_token_ce(write_out, write_batch)

        energy, _ = self._energy_values(energy_input)
        return self._energy_ce_guidance_loss(energy, token_ce, token_ce_mask)

    def _energy_parameter_l2_penalty(self):
        params = self._energy_parameters()
        if not params:
            ref = next(self.parameters())
            return ref.new_zeros(())
        penalty = params[0].new_zeros(())
        for param in params:
            penalty = penalty + param.pow(2).sum()
        return penalty

    def _energy_pretrain_l2_regularization(self):
        if self.energy_pretrain_l2_reg <= 0.0:
            ref = next(self.parameters())
            return ref.new_zeros(())
        return self.energy_pretrain_l2_reg * self._energy_parameter_l2_penalty()

    def _energy_weight_rms(self):
        weights = self._energy_weight_parameters()
        if not weights:
            ref = next(self.parameters())
            return ref.new_zeros(())
        flattened_weights = torch.cat([parameter.reshape(-1) for parameter in weights])
        return torch.linalg.vector_norm(flattened_weights) / math.sqrt(flattened_weights.numel())

    def _energy_weight_rms_regularization(self):
        energy_weight_rms = self._energy_weight_rms()
        excess = torch.relu(energy_weight_rms - self.energy_weight_rms_threshold)
        return self.energy_weight_rms_reg * excess.square(), energy_weight_rms

    def pretrain_energy_objective(self):
        if self.energy_pretrain_batch_size < 1 or self.energy_pretrain_seq_len < 2:
            raise ValueError("energy_pretrain_batch_size must be >= 1 and energy_pretrain_seq_len must be >= 2")
        if self.energy_pretrain_steps <= 0:
            return
        if self.inner_objective != "neural":
            raise ValueError("energy pretraining requires trainable inner_objective='neural'")
        if self.energy_model_type == "segment_delta_gru":
            raise ValueError("energy_model_type='segment_delta_gru' does not support energy pretraining")

        was_training = self.training
        device = self.model.get_input_embeddings().weight.device
        generator = torch.Generator(device=device)
        generator.manual_seed(self.energy_pretrain_seed)
        optimizer = torch.optim.AdamW(self._energy_parameters(), lr=self.energy_pretrain_lr, weight_decay=0.0)

        try:
            self.eval()
            self.energy_encoder.train()
            self.energy_head.train()
            backend = self.memory_backend_impl
            pad_id = self.model.config.pad_token_id
            show_progress = _is_main_process()
            log_every = max(1, self.energy_pretrain_steps // 10)
            last_loss = None

            if show_progress:
                logger.info(
                    "Energy pretraining started: objective=%s, steps=%d, batch_size=%d, seq_len=%d, lr=%g, l2_reg=%g",
                    self.energy_pretrain_objective,
                    self.energy_pretrain_steps,
                    self.energy_pretrain_batch_size,
                    self.energy_pretrain_seq_len,
                    self.energy_pretrain_lr,
                    self.energy_pretrain_l2_reg,
                )

            progress = tqdm(
                range(self.energy_pretrain_steps),
                desc="energy pretrain",
                disable=not show_progress,
            )
            for step in progress:
                if self.energy_pretrain_objective == "ce":
                    loss = self._ce_pretrain_loss(backend, pad_id, generator, device)
                elif self.energy_pretrain_objective == "embedding_mean_abs_diff":
                    loss = self._embedding_mean_abs_diff_pretrain_loss(
                        self.energy_pretrain_batch_size,
                        self.energy_pretrain_seq_len,
                        generator,
                        device,
                    )
                elif self.energy_pretrain_objective == "embedding_l2":
                    loss = self._embedding_l2_pretrain_loss(
                        self.energy_pretrain_batch_size,
                        self.energy_pretrain_seq_len,
                        generator,
                        device,
                    )
                else:
                    raise ValueError(f"Unsupported energy_pretrain_objective={self.energy_pretrain_objective}")
                loss = loss + self._energy_pretrain_l2_regularization()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())
                if show_progress:
                    progress.set_postfix(loss=f"{last_loss:.4f}")
                    if (step + 1) % log_every == 0 or step == 0 or (step + 1) == self.energy_pretrain_steps:
                        logger.info(
                            "Energy pretraining step %d/%d: loss=%.6f",
                            step + 1,
                            self.energy_pretrain_steps,
                            last_loss,
                        )

            if show_progress:
                logger.info("Energy pretraining finished: final_loss=%.6f", last_loss)
        finally:
            self.train(was_training)

    def _validate_context_segments(self, context_segments, batch_size):
        for i, segment in enumerate(context_segments):
            if segment.size(0) != batch_size:
                raise ValueError(
                    f"context segment batch size mismatch at segment={i}: "
                    f"segment_B={segment.size(0)}, query_B={batch_size}"
                )

    @staticmethod
    def _init_inner_loop_stats(device):
        return {
            "inner_grad_norm_mean": torch.tensor(0.0, device=device),
            "inner_grad_norm_max": torch.tensor(-1.0, device=device),
            "inner_grad_norm_min": torch.tensor(1e06, device=device),
        }

    def _extract_context_hidden(self, hidden, write_batch, batch_ctx):
        ctx_start = self._write_context_start(batch_ctx)
        ctx_len = write_batch["mask"].size(1)
        ctx_hidden = hidden[:, ctx_start:ctx_start + ctx_len, :]
        if ctx_hidden.size(1) != ctx_len:
            raise ValueError(
                "Invalid energy hidden-state alignment: "
                f"backend={self.memory_backend}, hidden_len={hidden.size(1)}, "
                f"ctx_start={ctx_start}, ctx_len={ctx_len}"
            )
        return ctx_hidden

    def _future_embeddings(self, segment, mask):
        if self.energy_future_mode == "none":
            return None
        if self.energy_future_mode != "next_token":
            raise ValueError(f"Unsupported energy_future_mode={self.energy_future_mode}")

        emb_layer = self.model.get_input_embeddings()
        target_len = mask.size(1)
        emb_dim = emb_layer.embedding_dim
        future = emb_layer.weight.new_zeros(segment.size(0), target_len, emb_dim)
        if segment.size(1) <= 1:
            return future

        segment_len = min(segment.size(1), target_len)
        next_len = max(0, segment_len - 1)
        if next_len == 0:
            return future

        next_valid = mask[:, 1:segment_len].bool()
        next_emb = emb_layer(segment[:, 1:segment_len])
        future[:, :next_len, :] = next_emb * next_valid.unsqueeze(-1).to(next_emb.dtype)
        return future

    def _energy_input(self, ctx_hidden, segment, mask):
        future = self._future_embeddings(segment, mask)
        if future is None:
            return ctx_hidden
        return torch.cat([ctx_hidden, future.to(dtype=ctx_hidden.dtype)], dim=-1)

    def _inner_grad_options(self, global_step, total_steps, keep_energy_graph=False, carries_state_graph=True):
        is_second_order_step = (
            self.grad_mode == "second"
            and global_step >= (total_steps - self.last_K_second_order)
        )
        create_graph = is_second_order_step or self.energy_delta_reg > 0.0
        has_future_energy_use = carries_state_graph and global_step < (total_steps - 1)
        retain_graph = (
            create_graph
            or has_future_energy_use
            or keep_energy_graph
            or (self.add_inner_loss_to_outer and global_step == total_steps - 1)
        )
        return create_graph, retain_graph

    @staticmethod
    def _record_grad_stats(stats, grads, batch_size, device):
        g_sq = torch.zeros(batch_size, device=device)
        for g in grads:
            g_sq = g_sq + g.reshape(batch_size, -1).pow(2).sum(dim=1)
        g_norm = g_sq.sqrt().detach()
        stats["inner_grad_norm_mean"] += g_norm.mean()
        stats["inner_grad_norm_max"] = max(stats["inner_grad_norm_max"], g_norm.max())
        stats["inner_grad_norm_min"] = min(stats["inner_grad_norm_min"], g_norm.min())

    def _updated_inner_params(self, inner_params, grads, opt_state, local_step):
        new_params = []
        for p_idx, (p, g) in enumerate(zip(inner_params, grads)):
            if self.use_adam:
                p_new = self._adam_step(p, g, opt_state.setdefault(str(p_idx), {}), local_step + 1, self.lr)
            else:
                p_new = self._sgd_step(
                    p,
                    g,
                    clip_value=self.inner_clip_value,
                    clip_norm=self.inner_clip_norm,
                )
            new_params.append(p_new)
        return new_params

    @staticmethod
    def _rotate_feature_pairs(memory, angle):
        pair_width = memory.size(-1) - memory.size(-1) % 2
        if pair_width == 0:
            raise ValueError("Pairwise memory rotation requires memory vectors with at least two features")

        pairs = memory[..., :pair_width].reshape(*memory.shape[:-1], -1, 2)
        angle = torch.as_tensor(angle, device=memory.device, dtype=memory.dtype)
        while angle.ndim < memory.ndim:
            angle = angle.unsqueeze(-1)
        cos = angle.cos()
        sin = angle.sin()
        first, second = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (first * cos - second * sin, first * sin + second * cos),
            dim=-1,
        ).flatten(-2)

        if pair_width < memory.size(-1):
            rotated = torch.cat((rotated, memory[..., pair_width:]), dim=-1)
        return rotated

    def _segment_rotation_angle(self, active_segment_counts):
        if self.memory_rotation == "none":
            return None
        if self.memory_rotation != "pairwise":
            raise ValueError(f"Unsupported memory_rotation={self.memory_rotation}")
        if self.memory_rotation_angle is not None:
            return self.memory_rotation_angle
        return math.pi / active_segment_counts.clamp_min(1)

    def _rotate_active_tensor(self, tensor, angle, active_samples):
        if angle is None or not active_samples.any():
            return tensor
        rotated = self._rotate_feature_pairs(tensor, angle)
        active_shape = (active_samples.size(0),) + (1,) * (tensor.ndim - 1)
        return torch.where(active_samples.reshape(active_shape), rotated, tensor)

    def _update_segment_delta_state(self, segment_state, delta_for_state, active_samples):
        candidate_state = self.segment_state_gru(
            delta_for_state.flatten(start_dim=1),
            segment_state,
        )
        return torch.where(active_samples[:, None], candidate_state, segment_state)

    def _replay_energy(self, memory, segment_state, replay_active):
        zero_energy = memory.flatten(start_dim=1).sum(dim=1) * 0.0
        if self.energy_replay_weight <= 0.0 or not replay_active.any():
            return zero_energy
        replay_input = torch.cat(
            (
                memory.flatten(start_dim=1),
                segment_state.to(dtype=memory.dtype),
            ),
            dim=1,
        )
        replay_energy = self.replay_energy_mlp(replay_input).squeeze(-1)
        return torch.where(replay_active, replay_energy, zero_energy)

    @staticmethod
    def _replay_gradient_metrics(
        current_gradient,
        weighted_replay_gradient,
        active_samples,
        complete_non_replay_gradient=None,
    ):
        current = current_gradient.detach().float().flatten(start_dim=1)[active_samples]
        replay = weighted_replay_gradient.detach().float().flatten(start_dim=1)[active_samples]
        current_norm = current.norm(dim=1)
        replay_norm = replay.norm(dim=1)
        eps = torch.finfo(current.dtype).eps

        norm_ratio = replay_norm / current_norm.clamp_min(eps)
        norm_product = current_norm * replay_norm
        cosine = torch.where(
            norm_product > 0,
            (current * replay).sum(dim=1) / norm_product.clamp_min(eps),
            torch.zeros_like(norm_product),
        ).clamp(min=-1.0, max=1.0)
        metrics = {
            "current_norm": torch.nan_to_num(current_norm),
            "replay_norm": torch.nan_to_num(replay_norm),
            "norm_ratio": torch.nan_to_num(norm_ratio),
            "cosine": torch.nan_to_num(cosine),
        }
        if complete_non_replay_gradient is not None:
            complete = (
                complete_non_replay_gradient.detach()
                .float()
                .flatten(start_dim=1)[active_samples]
            )
            complete_norm = complete.norm(dim=1)
            metrics["complete_norm_ratio"] = torch.nan_to_num(
                replay_norm / complete_norm.clamp_min(eps)
            )
        return metrics

    @staticmethod
    def _new_replay_diagnostics(reference):
        zero = reference.new_zeros((), dtype=torch.float32)
        return {
            "force_count": 0,
            "current_norm_sum": zero.clone(),
            "current_norm_max": zero.clone(),
            "replay_norm_sum": zero.clone(),
            "replay_norm_max": zero.clone(),
            "norm_ratio_sum": zero.clone(),
            "norm_ratio_max": zero.clone(),
            "cosine_sum": zero.clone(),
            "cosine_min": None,
            "conflict_count": 0,
            "complete_ratio_count": 0,
            "complete_ratio_sum": zero.clone(),
            "complete_ratio_max": zero.clone(),
            "state_count": 0,
            "state_norm_sum": zero.clone(),
            "state_norm_max": zero.clone(),
            "state_change_norm_sum": zero.clone(),
            "state_change_norm_max": zero.clone(),
            "state_saturated_count": 0,
            "state_element_count": 0,
        }

    def _accumulate_replay_gradient_metrics(
        self,
        diagnostics,
        current_gradient,
        weighted_replay_gradient,
        replay_active,
        complete_non_replay_gradient=None,
    ):
        metrics = self._replay_gradient_metrics(
            current_gradient,
            weighted_replay_gradient,
            replay_active,
            complete_non_replay_gradient,
        )
        count = metrics["current_norm"].numel()
        if not count:
            return
        diagnostics["force_count"] += count
        for metric_name, accumulator_prefix in (
            ("current_norm", "current_norm"),
            ("replay_norm", "replay_norm"),
            ("norm_ratio", "norm_ratio"),
        ):
            values = metrics[metric_name]
            diagnostics[f"{accumulator_prefix}_sum"] += values.sum()
            diagnostics[f"{accumulator_prefix}_max"] = torch.maximum(
                diagnostics[f"{accumulator_prefix}_max"],
                values.max(),
            )
        cosine = metrics["cosine"]
        diagnostics["cosine_sum"] += cosine.sum()
        cosine_min = cosine.min()
        diagnostics["cosine_min"] = (
            cosine_min
            if diagnostics["cosine_min"] is None
            else torch.minimum(diagnostics["cosine_min"], cosine_min)
        )
        diagnostics["conflict_count"] += int((cosine < 0).sum().item())

        complete_ratio = metrics.get("complete_norm_ratio")
        if complete_ratio is not None:
            diagnostics["complete_ratio_count"] += complete_ratio.numel()
            diagnostics["complete_ratio_sum"] += complete_ratio.sum()
            diagnostics["complete_ratio_max"] = torch.maximum(
                diagnostics["complete_ratio_max"],
                complete_ratio.max(),
            )

    @staticmethod
    def _accumulate_replay_state_metrics(diagnostics, state_before, state_after, replay_active):
        if not replay_active.any():
            return
        # Replay for the current segment is conditioned on state_before. Use
        # that same state for norm and saturation diagnostics; state_after is
        # relevant only to the boundary-update magnitude below.
        active_state = state_before.detach().float()[replay_active]
        active_change = (state_after - state_before).detach().float()[replay_active]
        state_norm = active_state.norm(dim=1)
        state_change_norm = active_change.norm(dim=1)
        diagnostics["state_count"] += state_norm.numel()
        diagnostics["state_norm_sum"] += state_norm.sum()
        diagnostics["state_norm_max"] = torch.maximum(
            diagnostics["state_norm_max"],
            state_norm.max(),
        )
        diagnostics["state_change_norm_sum"] += state_change_norm.sum()
        diagnostics["state_change_norm_max"] = torch.maximum(
            diagnostics["state_change_norm_max"],
            state_change_norm.max(),
        )
        diagnostics["state_saturated_count"] += int((active_state.abs() > 0.95).sum().item())
        diagnostics["state_element_count"] += active_state.numel()

    def _finalize_replay_diagnostics(self, stats, diagnostics):
        if not hasattr(self, "replay_energy_mlp"):
            return
        zero = diagnostics["current_norm_sum"].new_zeros(())
        force_count = diagnostics["force_count"]
        stats["current_energy_memory_grad_norm_mean"] = (
            diagnostics["current_norm_sum"] / force_count if force_count else zero
        )
        stats["current_energy_memory_grad_norm_max"] = diagnostics["current_norm_max"]
        stats["weighted_replay_memory_grad_norm_mean"] = (
            diagnostics["replay_norm_sum"] / force_count if force_count else zero
        )
        stats["weighted_replay_memory_grad_norm_max"] = diagnostics["replay_norm_max"]
        stats["replay_current_grad_norm_ratio_mean"] = (
            diagnostics["norm_ratio_sum"] / force_count if force_count else zero
        )
        stats["replay_current_grad_norm_ratio_max"] = diagnostics["norm_ratio_max"]
        stats["replay_current_grad_cosine_mean"] = (
            diagnostics["cosine_sum"] / force_count if force_count else zero
        )
        stats["replay_current_grad_cosine_min"] = (
            diagnostics["cosine_min"] if diagnostics["cosine_min"] is not None else zero
        )
        stats["replay_current_grad_conflict_fraction"] = zero.new_tensor(
            diagnostics["conflict_count"] / force_count if force_count else 0.0
        )

        complete_count = diagnostics["complete_ratio_count"]
        if self.energy_inner_ce_weight != 0.0:
            stats["weighted_replay_non_replay_grad_norm_ratio_mean"] = (
                diagnostics["complete_ratio_sum"] / complete_count if complete_count else zero
            )
            stats["weighted_replay_non_replay_grad_norm_ratio_max"] = diagnostics[
                "complete_ratio_max"
            ]

        state_count = diagnostics["state_count"]
        stats["recurrent_state_norm_mean"] = (
            diagnostics["state_norm_sum"] / state_count if state_count else zero
        )
        stats["recurrent_state_norm_max"] = diagnostics["state_norm_max"]
        stats["recurrent_state_change_norm_mean"] = (
            diagnostics["state_change_norm_sum"] / state_count if state_count else zero
        )
        stats["recurrent_state_change_norm_max"] = diagnostics["state_change_norm_max"]
        state_element_count = diagnostics["state_element_count"]
        stats["recurrent_state_saturation_fraction"] = zero.new_tensor(
            diagnostics["state_saturated_count"] / state_element_count
            if state_element_count
            else 0.0
        )

        replay_input_weight = self.replay_energy_mlp[0].weight.detach().float()
        memory_input_size = replay_input_weight.size(1) - self.energy_segment_state_size
        stats["replay_memory_input_weight_rms"] = (
            replay_input_weight[:, :memory_input_size].square().mean().sqrt()
        )
        stats["replay_state_input_weight_rms"] = (
            replay_input_weight[:, memory_input_size:].square().mean().sqrt()
        )

    @staticmethod
    def _validate_parallel_segments(context_segments):
        segment_len = context_segments[0].size(1)
        for i, segment in enumerate(context_segments):
            if segment.size(1) != segment_len:
                raise ValueError(
                    "segment_write_mode='parallel' requires equal segment lengths; "
                    f"segment=0 len={segment_len}, segment={i} len={segment.size(1)}"
                )

    @staticmethod
    def _repeat_prefix_memory_state(memory_state, n_segments):
        repeated = {"mem_batch": memory_state["mem_batch"].repeat_interleave(n_segments, dim=0)}
        if "W_batch" in memory_state:
            repeated["W_batch"] = memory_state["W_batch"].repeat_interleave(n_segments, dim=0)
            repeated["b_batch"] = memory_state["b_batch"].repeat_interleave(n_segments, dim=0)
        return repeated

    def _write_segments_parallel(self, context_segments, query_input_ids, memory_state, energy_state):
        self._validate_parallel_segments(context_segments)

        backend = self.memory_backend_impl
        pad_id = self.model.config.pad_token_id
        device = query_input_ids.device
        batch_size = query_input_ids.size(0)
        n_segments = len(context_segments)
        segment_len = context_segments[0].size(1)
        opt_state = {}
        inner_loss = torch.tensor(0.0, device=device)
        inner_energy_loss = torch.tensor(0.0, device=device)
        inner_ce_loss = torch.tensor(0.0, device=device)
        guidance_loss = torch.tensor(0.0, device=device)
        write_steps = 0
        stats = self._init_inner_loop_stats(device)

        if not self.K:
            return memory_state, energy_state, inner_loss, guidance_loss, write_steps, stats

        flat_segments = torch.stack(context_segments, dim=1).reshape(batch_size * n_segments, segment_len)
        flat_query_input_ids = query_input_ids.repeat_interleave(n_segments, dim=0)
        batch_ctx = backend.prepare_batch(flat_segments, flat_query_input_ids, pad_id)

        with torch.enable_grad():
            for k in range(self.K):
                write_memory_state = self._repeat_prefix_memory_state(memory_state, n_segments)
                write_batch = backend.build_write_inputs(write_memory_state, batch_ctx)
                write_out = self._run_write_model(write_batch, write_memory_state)
                if self.inner_objective == "cross_entropy":
                    step_ce_loss = self._write_parallel_inner_ce_loss(write_out, write_batch, batch_size, n_segments)
                    energy_loss = torch.zeros_like(step_ce_loss)
                    inner_loss = step_ce_loss
                else:
                    ctx_hidden = self._extract_context_hidden(write_out.hidden_states[-1], write_batch, batch_ctx)
                    ctx_len = write_batch["mask"].size(1)
                    ctx_hidden = ctx_hidden.reshape(batch_size, n_segments * ctx_len, ctx_hidden.size(-1))
                    energy_mask = write_batch["mask"].reshape(batch_size, n_segments * ctx_len)
                    energy_loss, energy_state, _ = self._energy_loss(ctx_hidden, energy_mask, energy_state)
                    step_ce_loss = torch.zeros_like(energy_loss)
                    if self.energy_inner_ce_weight != 0.0:
                        step_ce_loss = self._write_parallel_inner_ce_loss(write_out, write_batch, batch_size, n_segments)
                    inner_loss = energy_loss + self.energy_inner_ce_weight * step_ce_loss
                inner_energy_loss = inner_energy_loss + energy_loss.detach()
                inner_ce_loss = inner_ce_loss + step_ce_loss.detach()
                del write_out

                create_graph, retain_graph = self._inner_grad_options(
                    k,
                    self.K,
                    carries_state_graph=self.inner_objective != "cross_entropy",
                )
                inner_params = backend.inner_params(memory_state)
                grads = torch.autograd.grad(
                    inner_loss,
                    inner_params,
                    create_graph=create_graph,
                    retain_graph=retain_graph,
                )

                self._record_grad_stats(stats, grads, batch_size, device)
                new_params = self._updated_inner_params(inner_params, grads, opt_state, k)
                backend.assign_inner_params(memory_state, new_params)
                backend.maybe_detach_after_step(memory_state)
                write_steps += 1

        if write_steps:
            stats["inner_energy_loss"] = inner_energy_loss / (write_steps * batch_size)
            stats["inner_ce_loss"] = inner_ce_loss / (write_steps * batch_size)
        return memory_state, energy_state, inner_loss, guidance_loss, write_steps, stats

    def _write_segments(self, context_segments, query_input_ids, memory_state, energy_state):
        backend = self.memory_backend_impl
        pad_id = self.model.config.pad_token_id
        device = query_input_ids.device
        batch_size = query_input_ids.size(0)
        opt_state = {}
        inner_loss = torch.tensor(0.0, device=device)
        inner_energy_loss = torch.tensor(0.0, device=device)
        inner_ce_loss = torch.tensor(0.0, device=device)
        guidance_loss = torch.tensor(0.0, device=device)
        write_steps = 0
        stats = self._init_inner_loop_stats(device)
        segment_delta_norm_sum = torch.tensor(0.0, device=device)
        segment_delta_norm_max = torch.tensor(0.0, device=device)
        segment_delta_count = 0
        segment_delta_exceed_count = 0
        segment_delta_penalty_sum = None
        replay_energy_sum = torch.tensor(0.0, device=device)
        replay_energy_count = 0
        replay_diagnostics = self._new_replay_diagnostics(inner_loss)
        collect_replay_diagnostics = hasattr(self, "replay_energy_mlp") and not self.training

        if not self.K and self.memory_rotation == "none":
            if self.inner_objective == "neural" or self.energy_delta_reg > 0.0:
                stats["_energy_delta_reg_loss"] = inner_loss.new_zeros(())
                stats["energy_delta_reg_loss"] = inner_loss.new_zeros(())
                stats["energy_delta_exceed_fraction"] = inner_loss.new_zeros(())
            if self.energy_model_type == "segment_delta_gru":
                stats["segment_delta_norm_mean"] = segment_delta_norm_sum
                stats["segment_delta_norm_max"] = segment_delta_norm_max
                stats["segment_state_norm_mean"] = energy_state.detach().norm(dim=1).mean()
                stats["replay_energy_mean"] = replay_energy_sum
                stats["replay_energy_loss"] = replay_energy_sum
                if collect_replay_diagnostics:
                    self._finalize_replay_diagnostics(stats, replay_diagnostics)
            return memory_state, energy_state, inner_loss, guidance_loss, write_steps, stats

        with torch.enable_grad():
            active_segment_masks = [segment.ne(pad_id).any(dim=1) for segment in context_segments]
            total_steps = self.K * sum(bool(active_samples.any()) for active_samples in active_segment_masks)
            global_step = 0
            active_segment_counts = torch.stack(active_segment_masks).sum(dim=0)
            rotation_angle = self._segment_rotation_angle(active_segment_counts)
            has_previous_segment = None
            if self.energy_model_type == "segment_delta_gru":
                has_previous_segment = energy_state.detach().ne(0).any(dim=1)
            for segment, active_samples in zip(context_segments, active_segment_masks):
                batch_ctx = backend.prepare_batch(segment, query_input_ids, pad_id)
                if not active_samples.any():
                    continue
                replay_active = None
                if has_previous_segment is not None:
                    replay_active = active_samples & has_previous_segment

                memory_before = None
                if self.K and (
                    self.energy_delta_reg > 0.0
                    or self.energy_model_type == "segment_delta_gru"
                ):
                    memory_before = memory_state["mem_batch"].clone()

                for k in range(self.K):
                    write_batch = backend.build_write_inputs(memory_state, batch_ctx)
                    write_out = self._run_write_model(write_batch, memory_state)
                    first_order_replay_grad = None
                    current_energy_memory_grad = None
                    weighted_replay_memory_grad = None
                    complete_non_replay_memory_grad = None
                    if self.inner_objective == "cross_entropy":
                        step_ce_loss = self._write_inner_ce_loss(write_out, write_batch)
                        energy_loss = torch.zeros_like(step_ce_loss)
                        base_inner_loss = step_ce_loss
                        inner_loss = base_inner_loss
                    else:
                        ctx_hidden = self._extract_context_hidden(write_out.hidden_states[-1], write_batch, batch_ctx)
                        energy_input = self._energy_input(ctx_hidden, segment, write_batch["mask"])
                        energy_loss, energy_state, energy = self._energy_loss(
                            energy_input,
                            write_batch["mask"],
                            energy_state,
                        )
                        replay_loss = energy_loss.new_zeros(())
                        if self.energy_replay_weight > 0.0:
                            replay_memory = memory_state["mem_batch"]
                            replay_state = energy_state
                            if self.grad_mode == "first" and replay_active.any():
                                # First-order writes normally detach the inner gradient, which
                                # would leave a learned replay head with no target-loss signal.
                                # Differentiate replay through detached probes instead: its
                                # memory-gradient remains learnable with respect to replay-head
                                # parameters, without introducing Hessian paths through memory
                                # history or the recurrent segment state.
                                replay_memory = replay_memory.detach().requires_grad_(True)
                                replay_state = replay_state.detach()
                            replay_energy = self._replay_energy(
                                replay_memory,
                                replay_state,
                                replay_active,
                            )
                            replay_loss = replay_energy.sum()
                            if self.grad_mode == "first" and replay_active.any():
                                first_order_replay_grad = torch.autograd.grad(
                                    self.energy_replay_weight * replay_loss,
                                    replay_memory,
                                    create_graph=True,
                                    retain_graph=True,
                                )[0]
                                weighted_replay_memory_grad = first_order_replay_grad.detach()
                            replay_energy_sum = replay_energy_sum + replay_energy.detach().sum()
                            replay_energy_count += int(replay_active.sum().item())
                        step_ce_loss = torch.zeros_like(energy_loss)
                        if self.energy_ce_guidance:
                            token_ce, token_ce_mask = self._write_token_ce(write_out, write_batch)
                        if self.energy_inner_ce_weight != 0.0:
                            step_ce_loss = self._write_inner_ce_loss(write_out, write_batch)
                        base_inner_loss = (
                            energy_loss
                            + self.energy_inner_ce_weight * step_ce_loss
                        )
                        inner_loss = base_inner_loss + self.energy_replay_weight * replay_loss
                        if collect_replay_diagnostics and replay_active.any():
                            current_energy_memory_grad = torch.autograd.grad(
                                energy_loss,
                                memory_state["mem_batch"],
                                create_graph=False,
                                retain_graph=True,
                            )[0].detach()
                            if weighted_replay_memory_grad is None:
                                weighted_replay_memory_grad = torch.autograd.grad(
                                    self.energy_replay_weight * replay_loss,
                                    memory_state["mem_batch"],
                                    create_graph=False,
                                    retain_graph=True,
                                )[0].detach()
                            if self.energy_inner_ce_weight != 0.0:
                                complete_non_replay_memory_grad = torch.autograd.grad(
                                    base_inner_loss,
                                    memory_state["mem_batch"],
                                    create_graph=False,
                                    retain_graph=True,
                                )[0].detach()
                            self._accumulate_replay_gradient_metrics(
                                replay_diagnostics,
                                current_energy_memory_grad,
                                weighted_replay_memory_grad,
                                replay_active,
                                complete_non_replay_memory_grad,
                            )
                    inner_energy_loss = inner_energy_loss + energy_loss.detach()
                    inner_ce_loss = inner_ce_loss + step_ce_loss.detach()
                    if self.inner_objective != "cross_entropy" and self.energy_ce_guidance:
                        guidance_loss = guidance_loss + self._energy_ce_guidance_loss(
                            energy,
                            token_ce,
                            token_ce_mask,
                        )
                    del write_out

                    create_graph, retain_graph = self._inner_grad_options(
                        global_step,
                        total_steps,
                        keep_energy_graph=self.energy_ce_guidance,
                        carries_state_graph=self.inner_objective != "cross_entropy",
                    )
                    inner_params = backend.inner_params(memory_state)
                    grad_loss = base_inner_loss if first_order_replay_grad is not None else inner_loss
                    grads = torch.autograd.grad(
                        grad_loss,
                        inner_params,
                        create_graph=create_graph,
                        retain_graph=retain_graph,
                    )
                    if first_order_replay_grad is not None:
                        grads = (grads[0] + first_order_replay_grad, *grads[1:])

                    self._record_grad_stats(stats, grads, batch_size, device)
                    new_params = self._updated_inner_params(inner_params, grads, opt_state, global_step)
                    backend.assign_inner_params(memory_state, new_params)
                    backend.maybe_detach_after_step(memory_state)
                    write_steps += 1
                    global_step += 1

                if memory_before is not None:
                    raw_delta = memory_state["mem_batch"] - memory_before
                    if self.energy_model_type == "segment_delta_gru":
                        delta_for_state = self._rotate_active_tensor(raw_delta, rotation_angle, active_samples)
                        state_before_update = energy_state
                        energy_state = self._update_segment_delta_state(
                            energy_state,
                            delta_for_state,
                            active_samples,
                        )
                        if collect_replay_diagnostics:
                            self._accumulate_replay_state_metrics(
                                replay_diagnostics,
                                state_before_update,
                                energy_state,
                                replay_active,
                            )

                    active_delta_norms = raw_delta.flatten(start_dim=1).norm(dim=1)[active_samples]
                    delta_excess = torch.relu(active_delta_norms - self.energy_delta_max)
                    if self.energy_delta_reg > 0.0:
                        active_penalty_sum = delta_excess.square().sum()
                        segment_delta_penalty_sum = (
                            active_penalty_sum
                            if segment_delta_penalty_sum is None
                            else segment_delta_penalty_sum + active_penalty_sum
                        )
                    segment_delta_norm_sum = segment_delta_norm_sum + active_delta_norms.detach().sum()
                    segment_delta_norm_max = torch.maximum(
                        segment_delta_norm_max,
                        active_delta_norms.detach().max(),
                    )
                    segment_delta_count += active_delta_norms.numel()
                    segment_delta_exceed_count += int(
                        (active_delta_norms.detach() > self.energy_delta_max).sum().item()
                    )

                if self.memory_rotation != "none":
                    memory_state["mem_batch"] = self._rotate_active_tensor(
                        memory_state["mem_batch"],
                        rotation_angle,
                        active_samples,
                    )
                if has_previous_segment is not None and self.K:
                    has_previous_segment = has_previous_segment | active_samples

        if write_steps and self.energy_ce_guidance:
            guidance_loss = guidance_loss / write_steps
        if write_steps:
            stats["inner_energy_loss"] = inner_energy_loss / (write_steps * batch_size)
            stats["inner_ce_loss"] = inner_ce_loss / (write_steps * batch_size)
        if self.inner_objective == "neural" or self.energy_delta_reg > 0.0:
            if segment_delta_count:
                stats["segment_delta_norm_mean"] = segment_delta_norm_sum / segment_delta_count
                stats["segment_delta_norm_max"] = segment_delta_norm_max
                stats["energy_delta_exceed_fraction"] = segment_delta_norm_sum.new_tensor(
                    segment_delta_exceed_count / segment_delta_count
                )
            else:
                stats["segment_delta_norm_mean"] = segment_delta_norm_sum
                stats["segment_delta_norm_max"] = segment_delta_norm_max
                stats["energy_delta_exceed_fraction"] = segment_delta_norm_sum.new_zeros(())
            if segment_delta_penalty_sum is None:
                delta_reg_loss = inner_loss.new_zeros(())
            else:
                delta_reg_loss = (
                    self.energy_delta_reg * segment_delta_penalty_sum / segment_delta_count
                )
            stats["_energy_delta_reg_loss"] = delta_reg_loss
            stats["energy_delta_reg_loss"] = delta_reg_loss.detach()
        if self.energy_model_type == "segment_delta_gru":
            stats["segment_state_norm_mean"] = energy_state.detach().norm(dim=1).mean()
            if replay_energy_count:
                replay_energy_mean = replay_energy_sum / replay_energy_count
            else:
                replay_energy_mean = replay_energy_sum
            stats["replay_energy_mean"] = replay_energy_mean
            stats["replay_energy_loss"] = self.energy_replay_weight * replay_energy_mean
            if collect_replay_diagnostics:
                self._finalize_replay_diagnostics(stats, replay_diagnostics)
        return memory_state, energy_state, inner_loss, guidance_loss, write_steps, stats

    @staticmethod
    def _finalize_inner_stats(stats, inner_loss, write_steps, batch_size):
        if write_steps:
            stats["inner_grad_norm_mean"] = stats["inner_grad_norm_mean"] / write_steps
            stats["inner_loss"] = inner_loss.detach() / batch_size
        else:
            stats["inner_loss"] = inner_loss.detach()

    def _add_memory_stats(self, stats, memory_state, memory_state_initial):
        mem_norm, delta_mem_norm = self.memory_backend_impl.compute_memory_stats(memory_state, memory_state_initial)
        stats["mem_norm_mean"] = mem_norm.mean()
        stats["mem_norm_max"] = mem_norm.max()
        stats["mem_norm_min"] = mem_norm.min()
        stats["delta_mem_norm_mean"] = delta_mem_norm.mean()
        stats["delta_mem_norm_max"] = delta_mem_norm.max()
        stats["delta_mem_norm_min"] = delta_mem_norm.min()

    def _read_optimization_mask(self, query_input_ids, label_mask, read_batch):
        pred_len = int(read_batch["pred_len"])
        label_shift = int(read_batch.get("label_shift", 0))
        mask = torch.zeros(
            query_input_ids.size(0),
            pred_len,
            device=query_input_ids.device,
            dtype=torch.bool,
        )

        if label_mask is not None:
            aligned_mask = label_mask[:, label_shift:]
            if aligned_mask.size(1) != pred_len - 1:
                raise ValueError(
                    "Mismatched read-optimization alignment: "
                    f"pred_len={pred_len}, label_mask_len={aligned_mask.size(1)}, "
                    f"label_shift={label_shift}"
                )
            mask[:, :aligned_mask.size(1)] = aligned_mask.bool()
            return mask

        pad_id = self.model.config.pad_token_id
        valid_query_lengths = query_input_ids.ne(pad_id).sum(dim=1)
        prediction_positions = valid_query_lengths - label_shift
        active_samples = prediction_positions.ge(0) & prediction_positions.lt(pred_len)
        if active_samples.any():
            batch_indices = torch.arange(query_input_ids.size(0), device=query_input_ids.device)
            mask[
                batch_indices[active_samples],
                prediction_positions[active_samples],
            ] = True
        return mask

    def _select_read_energy_state(self, energy_state, sample_indices):
        if energy_state is None:
            return None
        if self.energy_model_type == "segment_delta_gru":
            return energy_state.index_select(0, sample_indices)
        if self.energy_model_type == "lstm":
            h, c = energy_state
            return (
                h.index_select(1, sample_indices),
                c.index_select(1, sample_indices),
            )
        if self.energy_model_type == "mamba2":
            return {
                name: value.index_select(0, sample_indices)
                for name, value in energy_state.items()
            }
        return energy_state

    def _clip_read_gradient(self, hidden_grad):
        if self.clip_read_norm is None:
            return hidden_grad
        grad_norm = hidden_grad.flatten(start_dim=1).norm(dim=1, keepdim=True)
        scale = (self.clip_read_norm / grad_norm.clamp_min(1e-12)).clamp(max=1.0)
        while scale.ndim < hidden_grad.ndim:
            scale = scale.unsqueeze(-1)
        return hidden_grad * scale

    def _optimize_read_hidden(self, read_hidden, read_mask, energy_state, create_graph):
        if not read_mask.any():
            raise ValueError("reading_optimization requires at least one active prediction position")

        initial_hidden = read_hidden
        selected_positions = read_mask.nonzero(as_tuple=False)
        sample_indices = selected_positions[:, 0]
        token_indices = selected_positions[:, 1]
        hidden = read_hidden[sample_indices, token_indices].unsqueeze(1)
        if not hidden.requires_grad:
            hidden = hidden.requires_grad_(True)
        selected_energy_state = self._select_read_energy_state(energy_state, sample_indices)
        selected_mask = torch.ones(
            hidden.size(0),
            1,
            device=hidden.device,
            dtype=torch.bool,
        )
        active_samples = read_mask.any(dim=1)
        energy_initial = None
        grad_norm_sum = hidden.new_zeros(())
        grad_norm_max = hidden.new_zeros(())

        for step in range(self.K_read):
            energy_loss, _, _ = self._energy_loss(
                hidden,
                selected_mask,
                selected_energy_state,
            )
            if energy_initial is None:
                energy_initial = energy_loss.detach()
            hidden_grad = torch.autograd.grad(
                energy_loss,
                hidden,
                create_graph=create_graph,
                retain_graph=create_graph or step < self.K_read - 1,
            )[0]
            grad_norms = hidden_grad.flatten(start_dim=1).norm(dim=1)
            grad_norm_sum = grad_norm_sum + grad_norms.detach().mean()
            grad_norm_max = torch.maximum(
                grad_norm_max,
                grad_norms.detach().max(),
            )
            hidden_grad = self._clip_read_gradient(hidden_grad)
            hidden = hidden - self.read_lr * hidden_grad

        # The energy used for the final update describes the pre-update state.
        # Re-evaluate once so the diagnostic measures the optimized state.
        with torch.no_grad():
            energy_final, _, _ = self._energy_loss(
                hidden,
                selected_mask,
                selected_energy_state,
            )

        selected_delta = hidden.squeeze(1) - initial_hidden[sample_indices, token_indices]
        flat_positions = sample_indices * read_hidden.size(1) + token_indices
        flat_delta = torch.zeros_like(read_hidden).flatten(end_dim=1)
        flat_delta = torch.index_copy(flat_delta, 0, flat_positions, selected_delta)
        optimized_hidden = read_hidden + flat_delta.view_as(read_hidden)
        hidden_delta_norms = (
            (optimized_hidden - initial_hidden)
            .detach()
            .flatten(start_dim=1)
            .norm(dim=1)[active_samples]
        )
        selected_count = hidden.new_tensor(selected_mask.size(0))
        stats = {
            "read_energy_initial": energy_initial / selected_count,
            "read_energy_last_step": energy_final.detach() / selected_count,
            "read_grad_norm_mean": grad_norm_sum / self.K_read,
            "read_grad_norm_max": grad_norm_max,
            "read_hidden_delta_norm_mean": hidden_delta_norms.mean(),
            "read_hidden_delta_norm_max": hidden_delta_norms.max(),
        }
        return optimized_hidden, stats

    def _read_from_memory(self, memory_state, query_input_ids, energy_state, label_mask=None):
        backend = self.memory_backend_impl
        pad_id = self.model.config.pad_token_id
        dummy_context = query_input_ids[:, :1].clone()
        read_ctx = backend.prepare_batch(dummy_context, query_input_ids, pad_id)
        read_batch = backend.build_read_inputs(memory_state, read_ctx)
        read_model_kwargs = read_batch.get("model_kwargs", {})
        log_mem_attn = (self.attn_implementation == "eager") and (self.memory_backend in ("prefix", "kv_cache"))
        if log_mem_attn:
            read_model_kwargs = dict(read_model_kwargs)
            read_model_kwargs["output_attentions"] = True
        if self.reading_optimization:
            read_model_kwargs = dict(read_model_kwargs)
            read_model_kwargs["output_hidden_states"] = True

        with backend.activation_context(memory_state):
            with self._disable_write_lora():
                read_out = self.model(
                    inputs_embeds=read_batch["inputs_embeds"],
                    return_dict=True,
                    **read_model_kwargs,
                )

        read_start = read_batch["logits_start"]
        read_end = read_start + read_batch["pred_len"]
        read_stats = {}
        if self.reading_optimization:
            if read_out.hidden_states is None:
                raise ValueError("Base model did not return hidden states for reading_optimization")
            read_hidden = read_out.hidden_states[-1][:, read_start:read_end, :]
            read_mask = self._read_optimization_mask(query_input_ids, label_mask, read_batch)
            outer_grad_enabled = torch.is_grad_enabled()
            create_graph = self.training and outer_grad_enabled
            with torch.enable_grad():
                backbone_hidden = read_hidden
                if self.read_grad_mode == "first" or not create_graph:
                    read_hidden = backbone_hidden.detach().requires_grad_(True)
                optimized_hidden, read_stats = self._optimize_read_hidden(
                    read_hidden,
                    read_mask,
                    energy_state,
                    create_graph=create_graph,
                )
                if self.read_grad_mode == "first" and create_graph:
                    # Straight-through backbone path: the optimized delta still
                    # trains the energy model, while its Hessian does not flow
                    # into the backbone hidden states.
                    optimized_hidden = backbone_hidden + (
                        optimized_hidden - read_hidden.detach()
                    )
                logits = self.model.get_output_embeddings()(optimized_hidden)
            if not outer_grad_enabled:
                logits = logits.detach()
        else:
            logits = read_out.logits[:, read_start:read_end, :]
        return logits, read_batch, read_out if log_mem_attn else None, read_stats

    def _add_read_attention_stats(self, stats, read_out):
        if read_out is None or read_out.attentions is None:
            return
        if self.memory_backend == "prefix":
            mem_start = self.n_ctrl_tokens
            mem_end = self.n_ctrl_tokens + self.n_mem_tokens
        else:
            mem_start = 0
            mem_end = self.n_mem_tokens

        layer_ratios = []
        for att in read_out.attentions:
            if att is None:
                continue
            k_len = att.size(-1)
            if not (0 <= mem_start < mem_end <= k_len):
                raise ValueError(
                    "EnergyGradMem: Invalid memory attention span on read: "
                    f"backend={self.memory_backend}, mem_start={mem_start}, mem_end={mem_end}, k_len={k_len}"
                )
            layer_ratios.append(att[..., mem_start:mem_end].sum(dim=-1).mean())
        if layer_ratios:
            stats["mem_attn_read"] = torch.stack(layer_ratios).mean().detach()

    @staticmethod
    def _target_loss(predictions, labels, read_batch):
        target_logits = predictions[:, :-1]
        target_label_shift = read_batch.get("label_shift", 0)
        target_labels = labels[:, target_label_shift:]
        if target_logits.size(1) != target_labels.size(1):
            raise ValueError(
                f"Mismatched target lengths after alignment: logits_len={target_logits.size(1)}, "
                f"labels_len={target_labels.size(1)}, label_shift={target_label_shift}"
            )
        return nn.functional.cross_entropy(
            target_logits.reshape(-1, predictions.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
        )

    def forward(self, input_ids, labels=None, return_mem=False, return_energy_state=False, energy_state=None):
        if torch.is_inference_mode_enabled():
            # Both memory writes and read optimization use autograd internally.
            # Disable inference tensors for the complete computation while
            # preserving the caller's no-gradient semantics.
            with torch.inference_mode(False), torch.no_grad():
                return self._forward_impl(
                    input_ids,
                    labels=labels,
                    return_mem=return_mem,
                    return_energy_state=return_energy_state,
                    energy_state=energy_state,
                )
        return self._forward_impl(
            input_ids,
            labels=labels,
            return_mem=return_mem,
            return_energy_state=return_energy_state,
            energy_state=energy_state,
        )

    def _forward_impl(self, input_ids, labels=None, return_mem=False, return_energy_state=False, energy_state=None):
        context_segments = self._context_segments(input_ids["context_input_ids"])
        query_input_ids = input_ids["query_input_ids"]
        if energy_state is None:
            energy_state = input_ids.get("energy_state")
        if self.inner_objective == "cross_entropy":
            energy_state = None

        B = query_input_ids.size(0)
        self._validate_context_segments(context_segments, B)

        backend = self.memory_backend_impl
        memory_state, memory_state_initial = backend.init_memory_state(B)
        memory_reference = backend.inner_params(memory_state)[0]
        energy_state = self._prepare_energy_state(energy_state, memory_reference)

        write_fn = self._write_segments_parallel if self.segment_write_mode == "parallel" else self._write_segments
        memory_state, energy_state, inner_loss, guidance_loss, write_steps, inner_loop_stats = write_fn(
            context_segments,
            query_input_ids,
            memory_state,
            energy_state,
        )
        energy_delta_reg_loss = inner_loop_stats.pop(
            "_energy_delta_reg_loss",
            inner_loss.new_zeros(()),
        )
        energy_weight_reg_loss = inner_loss.new_zeros(())
        if self.inner_objective == "neural":
            energy_weight_reg_loss, energy_weight_rms = self._energy_weight_rms_regularization()
            inner_loop_stats["energy_weight_rms_reg_loss"] = energy_weight_reg_loss.detach()
            inner_loop_stats["energy_weight_rms"] = energy_weight_rms.detach()
            inner_loop_stats.setdefault("energy_delta_reg_loss", energy_delta_reg_loss.detach())
            inner_loop_stats.setdefault("energy_delta_exceed_fraction", inner_loss.new_zeros(()))
        self._finalize_inner_stats(inner_loop_stats, inner_loss, write_steps, B)
        self._add_memory_stats(inner_loop_stats, memory_state, memory_state_initial)

        logits_q, read_batch, read_out, read_stats = self._read_from_memory(
            memory_state,
            query_input_ids,
            energy_state,
            label_mask=None if labels is None else labels.ne(-100),
        )
        inner_loop_stats.update(read_stats)
        self._add_read_attention_stats(inner_loop_stats, read_out)

        output = {"predictions": logits_q, "inner_loop_stats": inner_loop_stats}
        if return_mem:
            backend.attach_return_memory(output, memory_state)
        if return_energy_state or self.return_energy_state:
            output["energy_state"] = energy_state

        if labels is None:
            return output

        target_loss = self._target_loss(output["predictions"], labels, read_batch)
        output["inner_loop_stats"]["target_loss"] = target_loss.detach()
        if self.energy_ce_guidance:
            output["inner_loop_stats"]["energy_ce_guidance_loss"] = guidance_loss.detach()
        if self.add_inner_loss_to_outer:
            combined_loss = target_loss + self.inner_loss_weight * (inner_loss / B)
        else:
            combined_loss = target_loss
        if self.energy_ce_guidance:
            combined_loss = combined_loss + self.energy_ce_guidance_alpha * guidance_loss
        if self.inner_objective == "neural":
            combined_loss = combined_loss + energy_weight_reg_loss
        combined_loss = combined_loss + energy_delta_reg_loss
        output["loss"] = combined_loss
        return output
