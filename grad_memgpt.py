import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache
from contextlib import contextmanager, nullcontext
import math
import logging
import re
try:
    import attn_double_bwd  # noqa: F401  # side-effect: registers attention kernels
except ImportError:
    attn_double_bwd = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - handled via runtime checks
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def get_backbone(m):
    if hasattr(m, "get_base_model"):
        base = m.get_base_model()
        if base is not None and base is not m:
            return get_backbone(base)
    if hasattr(m, "base_model"):
        base = getattr(m, "base_model")
        if base is not None and base is not m:
            return get_backbone(base)
    # most HF CausalLM classes define base_model_prefix, e.g. "transformer" (GPT-2), "model" (LLaMA)
    if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
        return getattr(m, m.base_model_prefix)
    # robust fallback
    for attr in ("model", "transformer", "gpt_neox", "backbone", "decoder", "base_model"):
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError("Could not locate backbone submodule")


logger = logging.getLogger(__name__)


def _is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class GradMemGPTConfig(PretrainedConfig):
    """
    Configuration class for GradMemGPT.
    """
    model_type = "grad_memgpt"

    def __init__(self,
                 pretrained_model=None,
                 base_config=None,
                 memory_backend="prefix",
                 n_mem_tokens=8,
                 K=2,
                 lr=0.01,
                 use_adam=False,
                 grad_mode="second",
                 last_K_second_order=None,
                 n_ctrl_tokens=0,
                 inner_clip_value=None,
                 inner_clip_norm=None,
                 use_mem_proj=False,
                 mem_proj_mode="none",
                 use_write_head=False,
                 use_write_lora=False,
                 write_lora_r=8,
                 write_lora_alpha=16,
                 write_lora_dropout=0.0,
                 write_lora_target_modules=None,
                 lora_mem_placement="between_layers",
                 lora_mem_r=8,
                 lora_mem_alpha=16,
                 lora_mem_dropout=0.0,
                 lora_mem_layers=None,
                 lora_mem_target_modules=None,
                 kv_mem_layers=None,
                 freeze_backbone=False,
                 use_gradient_checkpointing=False,
                 attn_implementation="eager",
                 write_objective="reconstruction",
                 energy_head_hidden_dim=None,
                 use_layerwise_energy=False,
                 write_reconstruction_weight=1.0,
                 write_energy_weight=1.0,
                 energy_rank_weight=0.0,
                 energy_traj_weight=0.0,
                 energy_margin=0.1,
                 energy_traj_margin=0.0,
                 energy_rank_temperature=1.0,
                 energy_mix_alpha=0.75,
                 energy_anchor_weight=0.0,
                 lipschitz_weight=0.0,
                 lipschitz_constraint=1.0,
                 energy_memory_search_weight=0.0,
                 energy_memory_search_num_samples=4,
                 energy_memory_search_radius_scale=0.25,
                 energy_memory_search_use_gain_weighting=False,
                 energy_memory_search_gain_ema_decay=0.99,
                 energy_memory_search_min_relative_target_gain=0.0,
                 energy_memory_search_use_best_for_next_step=False,
                 read_focal_gamma=0.0,
                 add_inner_loss_to_outer=False,
                 inner_loss_weight=None,
                 memory_alignment_weight=0.0,
                 step_alignment_weight=0.0,
                 align_last_step=False,
                 grad_align_norm="none",
                 intermediate_read_weight=0.0,
                 memory_noise_sigma=0.0,
                 orthogonal_loss_weight=0.0,
                 ivan_loss_weight=0.0,
                 **kwargs):
        """
        Args:
            pretrained_model: str, name of pretrained model to load (e.g., 'gpt2')
            base_config: dict or PretrainedConfig, config for base model when creating from scratch
            memory_backend: str, memory implementation ("prefix", "lora", "kv_cache")
            n_mem_tokens: int, number of memory tokens
            K: int, number of inner loop steps
            lr: float, inner loop learning rate, it is a effective learning rate per sample
            use_adam: bool, whether to use Adam optimizer in inner loop
            grad_mode: str, gradient mode ("none", "first", "second")
            last_K_second_order: int, use second order update for last K inner gradient steps only
            n_ctrl_tokens: int, number of control tokens
            inner_clip_value: float, gradient clipping value
            inner_clip_norm: float, gradient clipping norm
            use_mem_proj: bool, whether to use memory projection
            mem_proj_mode: str, memory projection mode ("none", "proj", "per_sample")
            use_write_head: bool, whether to use write head
            use_write_lora: bool, enable LoRA adapters during WRITE phase only
            write_lora_r: int, LoRA rank for WRITE adapters
            write_lora_alpha: int, LoRA alpha for WRITE adapters
            write_lora_dropout: float, LoRA dropout for WRITE adapters
            write_lora_target_modules: list[str]|str|None, target module names (None/"auto" for defaults)
            lora_mem_placement: str, where LoRA memory is injected (currently: "between_layers")
            lora_mem_r: int, rank of LoRA memory adapters
            lora_mem_alpha: int, alpha scaling for LoRA memory adapters
            lora_mem_dropout: float, dropout on hidden states before LoRA memory projection
            lora_mem_layers: str|list[int]|None, transformer layers to inject LoRA memory
                examples: None or "all" (all layers), "last_4" (last 4 layers),
                "0,3,7" (explicit indices), [0, 3, 7] (explicit indices as list),
                "none"/"auto"/"" (treated as all layers)
            lora_mem_target_modules: list[str]|str|None, target module names for
                lora_mem_placement="target_modules"; None/"auto" uses model defaults
            kv_mem_layers: str|list[int]|None, transformer layers to inject KV-cache memory
                examples: None or "all" (all layers), "last_4" (last 4 layers),
                "0,3,7" (explicit indices), [0, 3, 7] (explicit indices as list),
                "none"/"auto"/"" (treated as all layers)
            freeze_backbone: bool, freeze backbone weights (READ+WRITE), except LoRA/write head/mem proj
            use_gradient_checkpointing: bool, turn on gradient checkpointing supported by HF models
            write_objective: str, inner WRITE objective ("reconstruction", "energy", or
                "energy_with_reconstruction")
            energy_head_hidden_dim: int|None, hidden dim for energy MLP; defaults to backbone hidden size
            use_layerwise_energy: bool, use a separate energy head at each transformer layer and sum them
            write_reconstruction_weight: float, reconstruction loss weight for energy_with_reconstruction
            write_energy_weight: float, energy loss weight for energy_with_reconstruction
            energy_rank_weight: float, optional context-memory contrastive ranking loss weight
            energy_traj_weight: float, optional monotonic trajectory loss weight
            energy_margin: float, margin for ranking loss
            energy_traj_margin: float, margin for trajectory monotonicity loss
            energy_rank_temperature: float, softplus temperature for contrastive ranking losses
            energy_mix_alpha: float, positive-memory coefficient for interpolated negatives
            energy_anchor_weight: float, optional energy-magnitude anchoring loss weight
            lipschitz_weight: float, weight for the final-state WRITE-objective gradient-norm constraint
            lipschitz_constraint: float, maximum allowed final-state WRITE-objective gradient norm
            energy_memory_search_weight: float, weight for matching each WRITE state to a locally
                perturbed memory with lower downstream READ loss
            energy_memory_search_num_samples: int, number of perturbed candidates per WRITE state
            energy_memory_search_radius_scale: float, candidate chord radius as a fraction of the
                preceding WRITE-step displacement
            energy_memory_search_use_gain_weighting: bool, weight selected-memory matching by its
                detached relative READ-loss improvement
            energy_memory_search_gain_ema_decay: float, decay for the per-WRITE-depth positive-gain
                means used to normalize memory-search gain weights
            energy_memory_search_min_relative_target_gain: float, minimum detached relative READ-loss
                improvement required for an example to contribute to the memory-search loss
            energy_memory_search_use_best_for_next_step: bool, use a straight-through copy of the
                selected target-guided memory as the starting point for the next WRITE step
            read_focal_gamma: float, focal exponent for token-level outer READ loss weighting
            add_inner_loss_to_outer: bool, outer loss = target_loss + inner_loss_weight * inner_loss_mean
            inner_loss_weight: float, weight of inner loss in combined loss
            memory_alignment_weight: float, weight for aligning the cumulative WRITE direction with the
                stopped READ gradient on the final prefix memory
            step_alignment_weight: float, weight for aligning each inner-loop memory update with the
                stopped READ gradient at the resulting memory state
            align_last_step: bool, also align the post-WRITE gradient at M_K, whose update is not applied
            grad_align_norm: str, step-alignment normalization ("none" or "norm"); "norm" weights
                each per-sample cosine loss by the stopped task-gradient norm
            intermediate_read_weight: float, weight for reverse-harmonic READ supervision on M_1...M_(K-1)
            memory_noise_sigma: float, training-only per-coordinate WRITE-gradient noise scale relative
                to the root-mean-square magnitude of the initial prefix memory
            orthogonal_loss_weight: float, weight for enforcing that the scaled stopped outer gradient
                leaves a residual inner update orthogonal to the outer gradient
            ivan_loss_weight: float, weight for the squared inner product between the outer gradient
                and the difference of the summed WRITE gradients and outer gradient
        """
        super().__init__(**kwargs)

        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
            self.base_config = None
        else:
            self.pretrained_model = None
            self.base_config = base_config

        # GradMemGPT specific parameters
        self.memory_backend = memory_backend
        self.n_mem_tokens = n_mem_tokens
        self.K = K
        self.lr = lr
        self.use_adam = use_adam
        self.grad_mode = grad_mode
        self.n_ctrl_tokens = n_ctrl_tokens
        self.inner_clip_value = inner_clip_value
        self.inner_clip_norm = inner_clip_norm
        self.use_mem_proj = use_mem_proj
        self.mem_proj_mode = mem_proj_mode
        self.use_write_head = use_write_head
        self.use_write_lora = use_write_lora
        self.write_lora_r = write_lora_r
        self.write_lora_alpha = write_lora_alpha
        self.write_lora_dropout = write_lora_dropout
        self.write_lora_target_modules = write_lora_target_modules
        self.lora_mem_placement = lora_mem_placement
        self.lora_mem_r = lora_mem_r
        self.lora_mem_alpha = lora_mem_alpha
        self.lora_mem_dropout = lora_mem_dropout
        self.lora_mem_layers = lora_mem_layers
        self.lora_mem_target_modules = lora_mem_target_modules
        self.kv_mem_layers = kv_mem_layers
        self.freeze_backbone = freeze_backbone
        self.last_K_second_order = K if last_K_second_order is None else last_K_second_order
        if grad_mode != "second":
            self.last_K_second_order = 0
        self.last_K_second_order = max(0, min(self.last_K_second_order, K))
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_implementation = attn_implementation
        self.write_objective = write_objective
        self.energy_head_hidden_dim = energy_head_hidden_dim
        self.use_layerwise_energy = use_layerwise_energy
        self.write_reconstruction_weight = write_reconstruction_weight
        self.write_energy_weight = write_energy_weight
        self.energy_rank_weight = energy_rank_weight
        self.energy_traj_weight = energy_traj_weight
        self.energy_margin = energy_margin
        self.energy_traj_margin = energy_traj_margin
        self.energy_rank_temperature = energy_rank_temperature
        self.energy_mix_alpha = energy_mix_alpha
        self.energy_anchor_weight = energy_anchor_weight
        self.lipschitz_weight = float(lipschitz_weight or 0.0)
        self.lipschitz_constraint = float(lipschitz_constraint)
        self.energy_memory_search_weight = energy_memory_search_weight
        self.energy_memory_search_num_samples = energy_memory_search_num_samples
        self.energy_memory_search_radius_scale = energy_memory_search_radius_scale
        self.energy_memory_search_use_gain_weighting = energy_memory_search_use_gain_weighting
        self.energy_memory_search_gain_ema_decay = energy_memory_search_gain_ema_decay
        self.energy_memory_search_min_relative_target_gain = float(
            energy_memory_search_min_relative_target_gain
        )
        self.energy_memory_search_use_best_for_next_step = energy_memory_search_use_best_for_next_step
        self.read_focal_gamma = float(read_focal_gamma)
        self.add_inner_loss_to_outer = add_inner_loss_to_outer
        self.inner_loss_weight = inner_loss_weight
        self.memory_alignment_weight = memory_alignment_weight
        self.step_alignment_weight = step_alignment_weight
        self.align_last_step = align_last_step
        self.grad_align_norm = grad_align_norm
        self.intermediate_read_weight = intermediate_read_weight
        self.memory_noise_sigma = float(memory_noise_sigma or 0.0)
        self.orthogonal_loss_weight = float(orthogonal_loss_weight or 0.0)
        self.ivan_loss_weight = float(ivan_loss_weight or 0.0)

        # Validate mem_proj_mode settings
        assert mem_proj_mode in ["none", "proj", "per_sample"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"
        assert self.memory_backend in ["prefix", "lora", "kv_cache"], (
            "memory_backend must be one of: prefix, lora, kv_cache"
        )
        assert self.write_objective in ["reconstruction", "energy", "energy_with_reconstruction"], (
            "write_objective must be one of: reconstruction, energy, energy_with_reconstruction"
        )
        if self.write_objective in ("energy", "energy_with_reconstruction") and self.memory_backend != "prefix":
            raise ValueError(
                "write_objective='energy' and 'energy_with_reconstruction' are currently supported only "
                "for memory_backend='prefix'"
            )
        if self.energy_rank_temperature <= 0.0:
            raise ValueError("energy_rank_temperature must be > 0")
        if not 0.0 <= self.energy_mix_alpha <= 1.0:
            raise ValueError("energy_mix_alpha must be within [0, 1]")
        if not math.isfinite(self.lipschitz_weight) or self.lipschitz_weight < 0.0:
            raise ValueError("lipschitz_weight must be finite and >= 0")
        if not math.isfinite(self.lipschitz_constraint) or self.lipschitz_constraint < 0.0:
            raise ValueError("lipschitz_constraint must be finite and >= 0")
        if not math.isfinite(self.read_focal_gamma) or self.read_focal_gamma < 0.0:
            raise ValueError("read_focal_gamma must be finite and >= 0")
        if self.lipschitz_weight > 0.0 and self.memory_backend != "prefix":
            raise ValueError("lipschitz_weight > 0 requires memory_backend='prefix'")
        if not math.isfinite(self.energy_memory_search_weight) or self.energy_memory_search_weight < 0.0:
            raise ValueError("energy_memory_search_weight must be finite and >= 0")
        if self.energy_memory_search_num_samples < 1:
            raise ValueError("energy_memory_search_num_samples must be >= 1")
        if (
            not math.isfinite(self.energy_memory_search_radius_scale)
            or self.energy_memory_search_radius_scale <= 0.0
        ):
            raise ValueError("energy_memory_search_radius_scale must be finite and > 0")
        if (
            not math.isfinite(self.energy_memory_search_gain_ema_decay)
            or not 0.0 <= self.energy_memory_search_gain_ema_decay < 1.0
        ):
            raise ValueError("energy_memory_search_gain_ema_decay must be finite and within [0, 1)")
        if (
            not math.isfinite(self.energy_memory_search_min_relative_target_gain)
            or self.energy_memory_search_min_relative_target_gain < 0.0
        ):
            raise ValueError("energy_memory_search_min_relative_target_gain must be finite and >= 0")
        if self.energy_memory_search_weight > 0.0:
            if self.memory_backend != "prefix":
                raise ValueError("energy_memory_search_weight > 0 requires memory_backend='prefix'")
            if self.K <= 0:
                raise ValueError("energy_memory_search_weight > 0 requires K > 0")
            if self.grad_mode != "second" or self.last_K_second_order != self.K:
                raise ValueError(
                    "energy_memory_search_weight > 0 requires all K WRITE steps to be second order"
                )
            if self.use_adam:
                raise ValueError("energy_memory_search_weight > 0 requires SGD WRITE updates")
        if self.energy_memory_search_use_best_for_next_step and self.energy_memory_search_weight <= 0.0:
            raise ValueError(
                "energy_memory_search_use_best_for_next_step requires energy_memory_search_weight > 0"
            )
        if self.memory_alignment_weight < 0.0:
            raise ValueError("memory_alignment_weight must be >= 0")
        if self.step_alignment_weight < 0.0:
            raise ValueError("step_alignment_weight must be >= 0")
        if self.grad_align_norm not in ("none", "norm"):
            raise ValueError("grad_align_norm must be one of: none, norm")
        if self.intermediate_read_weight < 0.0:
            raise ValueError("intermediate_read_weight must be >= 0")
        if not math.isfinite(self.memory_noise_sigma) or self.memory_noise_sigma < 0.0:
            raise ValueError("memory_noise_sigma must be finite and >= 0")
        if self.memory_noise_sigma > 0.0 and self.memory_backend != "prefix":
            raise ValueError("memory_noise_sigma > 0 requires memory_backend='prefix'")
        if self.orthogonal_loss_weight < 0.0:
            raise ValueError("orthogonal_loss_weight must be >= 0")
        if self.ivan_loss_weight < 0.0:
            raise ValueError("ivan_loss_weight must be >= 0")
        if self.ivan_loss_weight > 0.0:
            if self.grad_mode != "second":
                raise ValueError("ivan_loss_weight > 0 requires grad_mode='second'")
            if self.K <= 0:
                raise ValueError("ivan_loss_weight > 0 requires K > 0")
            if self.last_K_second_order != self.K:
                raise ValueError("ivan_loss_weight > 0 requires all K steps to be second order")
        if self.orthogonal_loss_weight > 0.0:
            if self.memory_backend != "prefix":
                raise ValueError("orthogonal_loss_weight > 0 requires memory_backend='prefix'")
            if self.grad_mode != "second":
                raise ValueError("orthogonal_loss_weight > 0 requires grad_mode='second'")
            if self.K <= 0:
                raise ValueError("orthogonal_loss_weight > 0 requires K > 0")
            if self.last_K_second_order <= 0:
                raise ValueError("orthogonal_loss_weight > 0 requires last_K_second_order > 0")
            if self.lr <= 0.0:
                raise ValueError("orthogonal_loss_weight > 0 requires lr > 0")
        if self.memory_alignment_weight > 0.0:
            if self.memory_backend != "prefix":
                raise ValueError("memory_alignment_weight > 0 requires memory_backend='prefix'")
            if self.grad_mode != "second":
                raise ValueError("memory_alignment_weight > 0 requires grad_mode='second'")
            if self.K <= 0:
                raise ValueError("memory_alignment_weight > 0 requires K > 0")
            if self.last_K_second_order <= 0:
                raise ValueError("memory_alignment_weight > 0 requires last_K_second_order > 0")
        if self.step_alignment_weight > 0.0:
            if self.memory_backend != "prefix":
                raise ValueError("step_alignment_weight > 0 requires memory_backend='prefix'")
            if self.grad_mode != "second":
                raise ValueError("step_alignment_weight > 0 requires grad_mode='second'")
            if self.K <= 0:
                raise ValueError("step_alignment_weight > 0 requires K > 0")
            if self.last_K_second_order <= 0:
                raise ValueError("step_alignment_weight > 0 requires last_K_second_order > 0")
        if self.memory_backend == "lora":
            assert self.lora_mem_placement in ["between_layers", "target_modules"], (
                "lora_mem_placement currently supports: 'between_layers', 'target_modules'"
            )


class MemoryBackend:
    def __init__(self, owner):
        self.owner = owner

    @contextmanager
    def activation_context(self, memory_state):
        _ = memory_state
        yield

    def snapshot_memory_state(self, memory_state):
        def snapshot(value):
            if isinstance(value, dict):
                return {key: snapshot(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(snapshot(item) for item in value)
            return value

        return snapshot(memory_state)


class InputPrefixMemoryBackend(MemoryBackend):
    def init_memory_state(self, batch_size):
        o = self.owner
        mem_batch = o.mem.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if o.grad_mode == "none":
            mem_batch = mem_batch.detach().requires_grad_(True)
        else:
            mem_batch = mem_batch.requires_grad_(True)

        memory_state = {"mem_batch": mem_batch}
        memory_state_initial = {"mem_batch": mem_batch.detach().clone()}

        if o.mem_proj_mode == "per_sample":
            W_batch = o.mem_proj.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
            b_batch = o.mem_proj.bias.unsqueeze(0).expand(batch_size, -1).clone()
            if o.grad_mode == "none":
                W_batch = W_batch.detach().requires_grad_(True)
                b_batch = b_batch.detach().requires_grad_(True)
            else:
                W_batch = W_batch.requires_grad_(True)
                b_batch = b_batch.requires_grad_(True)
            memory_state["W_batch"] = W_batch
            memory_state["b_batch"] = b_batch

        return memory_state, memory_state_initial

    def prepare_batch(self, context_input_ids, query_input_ids, pad_id):
        o = self.owner
        B = context_input_ids.size(0)
        mem_offset = o.n_mem_tokens + o.n_ctrl_tokens * 2

        batch_ctx = {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
            "ctx_emb": o.model.get_input_embeddings()(context_input_ids),
            "qry_emb": o.model.get_input_embeddings()(query_input_ids),
            "mem_offset": mem_offset,
        }

        lm_labels = context_input_ids.clone()
        lm_labels[lm_labels == pad_id] = -100
        mask = (lm_labels != -100)

        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_list = [0, -(batch_ctx["ctx_emb"].size(1) + mem_offset) % 32, 0, 0]
            mask = F.pad(mask, pad_list, "constant", 0)
            lm_labels = F.pad(lm_labels, pad_list, "constant", -100)
            batch_ctx["ctx_emb"] = F.pad(batch_ctx["ctx_emb"], [0, 0] + pad_list, "constant", 0)

        batch_ctx["lm_labels"] = lm_labels
        batch_ctx["mask"] = mask

        if o.n_ctrl_tokens > 0:
            batch_ctx["write_st_batch"] = o.write_st.unsqueeze(0).expand(B, -1, -1)
            batch_ctx["write_end_batch"] = o.write_end.unsqueeze(0).expand(B, -1, -1)
            batch_ctx["read_st_batch"] = o.read_st.unsqueeze(0).expand(B, -1, -1)
            batch_ctx["read_end_batch"] = o.read_end.unsqueeze(0).expand(B, -1, -1)

        return batch_ctx

    def build_write_inputs(self, memory_state, batch_ctx):
        o = self.owner
        mem_batch = memory_state["mem_batch"]
        if o.mem_proj_mode == "none":
            mem_inp = mem_batch
        elif o.mem_proj_mode == "proj":
            mem_inp = o.mem_proj(mem_batch)
        else:
            mem_inp = o._apply_linear(mem_batch, memory_state["W_batch"], memory_state["b_batch"])

        if o.n_ctrl_tokens > 0:
            x_ctx = torch.cat(
                [batch_ctx["write_st_batch"], mem_inp, batch_ctx["write_end_batch"], batch_ctx["ctx_emb"]],
                dim=1,
            )
        else:
            x_ctx = torch.cat([mem_inp, batch_ctx["ctx_emb"]], dim=1)

        return {
            "inputs_embeds": x_ctx,
            "lm_labels": batch_ctx["lm_labels"],
            "mask": batch_ctx["mask"],
            "logits_start": batch_ctx["mem_offset"] - 1,
            "label_shift": 0,
            "context_start": batch_ctx["mem_offset"],
        }

    def build_read_inputs(self, memory_state, batch_ctx):
        o = self.owner
        mem_batch = memory_state["mem_batch"]
        if o.mem_proj_mode == "none":
            mem_inp = mem_batch
        elif o.mem_proj_mode == "proj":
            mem_inp = o.mem_proj(mem_batch)
        else:
            mem_inp = o._apply_linear(mem_batch, memory_state["W_batch"], memory_state["b_batch"])

        if o.n_ctrl_tokens > 0:
            x_qry = torch.cat(
                [batch_ctx["read_st_batch"], mem_inp, batch_ctx["read_end_batch"], batch_ctx["qry_emb"]],
                dim=1,
            )
        else:
            x_qry = torch.cat([mem_inp, batch_ctx["qry_emb"]], dim=1)

        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            x_qry = F.pad(x_qry, [0, 0, 0, -x_qry.size(1) % 32], "constant", 0)

        return {
            "inputs_embeds": x_qry,
            "logits_start": batch_ctx["mem_offset"] - 1,
            "pred_len": batch_ctx["qry_emb"].size(1) + 1,
            "label_shift": 0,
        }

    def inner_params(self, memory_state):
        params = [memory_state["mem_batch"]]
        if self.owner.mem_proj_mode == "per_sample":
            params += [memory_state["W_batch"], memory_state["b_batch"]]
        return params

    def assign_inner_params(self, memory_state, new_params):
        memory_state["mem_batch"] = new_params[0]
        if self.owner.mem_proj_mode == "per_sample":
            memory_state["W_batch"] = new_params[1]
            memory_state["b_batch"] = new_params[2]

    def maybe_detach_after_step(self, memory_state):
        if self.owner.grad_mode != "none":
            return
        memory_state["mem_batch"] = memory_state["mem_batch"].detach().requires_grad_(True)
        if self.owner.mem_proj_mode == "per_sample":
            memory_state["W_batch"] = memory_state["W_batch"].detach().requires_grad_(True)
            memory_state["b_batch"] = memory_state["b_batch"].detach().requires_grad_(True)

    def compute_memory_stats(self, memory_state, memory_state_initial):
        mem_batch = memory_state["mem_batch"]
        mem_init = memory_state_initial["mem_batch"]
        mem_norm = mem_batch.norm(dim=(1, 2)).detach()
        delta_mem_norm = (mem_batch - mem_init).detach().norm(dim=(1, 2))
        return mem_norm, delta_mem_norm

    def attach_return_memory(self, output, memory_state):
        output["mem"] = memory_state["mem_batch"]
        if self.owner.mem_proj_mode == "per_sample":
            output["W"] = memory_state["W_batch"]
            output["b"] = memory_state["b_batch"]


class LoraMemoryBackend(MemoryBackend):
    def init_memory_state(self, batch_size):
        o = self.owner
        device = o.model.get_input_embeddings().weight.device
        memory_state = {"lora_mem": {}}
        memory_state_initial = {"lora_mem": {}}
        for slot_id in o.lora_mem_slot_ids:
            param_key = o._lora_mem_param_key(slot_id)
            A0 = o.lora_mem_A0[param_key].to(device=device)
            B0 = o.lora_mem_B0[param_key].to(device=device)
            A_batch = A0.unsqueeze(0).expand(batch_size, -1, -1).clone()
            B_batch = B0.unsqueeze(0).expand(batch_size, -1, -1).clone()
            if o.grad_mode == "none":
                A_batch = A_batch.detach().requires_grad_(True)
                B_batch = B_batch.detach().requires_grad_(True)
            else:
                A_batch = A_batch.requires_grad_(True)
                B_batch = B_batch.requires_grad_(True)
            memory_state["lora_mem"][slot_id] = (A_batch, B_batch)
            memory_state_initial["lora_mem"][slot_id] = (A_batch.detach().clone(), B_batch.detach().clone())
        return memory_state, memory_state_initial

    @contextmanager
    def activation_context(self, memory_state):
        with self.owner._enable_lora_memory(memory_state["lora_mem"]):
            yield

    def prepare_batch(self, context_input_ids, query_input_ids, pad_id):
        o = self.owner
        emb_layer = o.model.get_input_embeddings()
        ctx_emb = emb_layer(context_input_ids)
        qry_emb = emb_layer(query_input_ids)

        lm_labels = context_input_ids.clone()
        lm_labels[lm_labels == pad_id] = -100
        mask = (lm_labels != -100)

        x_ctx = ctx_emb
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_n = (-x_ctx.size(1)) % 32
            if pad_n:
                x_ctx = F.pad(x_ctx, [0, 0, 0, pad_n], "constant", 0)
                lm_labels = F.pad(lm_labels, [0, pad_n], "constant", -100)
                mask = F.pad(mask, [0, pad_n], "constant", 0)

        return {
            "x_ctx": x_ctx,
            "lm_labels": lm_labels,
            "mask": mask,
            "qry_emb": qry_emb,
        }

    def build_write_inputs(self, memory_state, batch_ctx):
        _ = memory_state
        return {
            "inputs_embeds": batch_ctx["x_ctx"],
            "lm_labels": batch_ctx["lm_labels"],
            "mask": batch_ctx["mask"],
            "logits_start": 0,
            "label_shift": 1,
        }

    def build_read_inputs(self, memory_state, batch_ctx):
        _ = memory_state
        o = self.owner
        x_qry = batch_ctx["qry_emb"]
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            x_qry = F.pad(x_qry, [0, 0, 0, (-x_qry.size(1)) % 32], "constant", 0)
        return {
            "inputs_embeds": x_qry,
            "logits_start": 0,
            "pred_len": batch_ctx["qry_emb"].size(1),
            "label_shift": 1,
        }

    def inner_params(self, memory_state):
        params = []
        for slot_id in self.owner.lora_mem_slot_ids:
            A_batch, B_batch = memory_state["lora_mem"][slot_id]
            params.extend([A_batch, B_batch])
        return params

    def assign_inner_params(self, memory_state, new_params):
        p = 0
        for slot_id in self.owner.lora_mem_slot_ids:
            memory_state["lora_mem"][slot_id] = (new_params[p], new_params[p + 1])
            p += 2

    def maybe_detach_after_step(self, memory_state):
        if self.owner.grad_mode != "none":
            return
        for slot_id in self.owner.lora_mem_slot_ids:
            A_batch, B_batch = memory_state["lora_mem"][slot_id]
            memory_state["lora_mem"][slot_id] = (
                A_batch.detach().requires_grad_(True),
                B_batch.detach().requires_grad_(True),
            )

    def compute_memory_stats(self, memory_state, memory_state_initial):
        mem_device = next(iter(memory_state["lora_mem"].values()))[0].device
        mem_norm_sq = torch.zeros(next(iter(memory_state["lora_mem"].values()))[0].size(0), device=mem_device)
        delta_mem_norm_sq = torch.zeros_like(mem_norm_sq)
        for slot_id in self.owner.lora_mem_slot_ids:
            A_batch, B_batch = memory_state["lora_mem"][slot_id]
            A_init, B_init = memory_state_initial["lora_mem"][slot_id]
            mem_norm_sq = mem_norm_sq + A_batch.detach().pow(2).sum(dim=(1, 2))
            mem_norm_sq = mem_norm_sq + B_batch.detach().pow(2).sum(dim=(1, 2))
            delta_mem_norm_sq = delta_mem_norm_sq + (A_batch.detach() - A_init).pow(2).sum(dim=(1, 2))
            delta_mem_norm_sq = delta_mem_norm_sq + (B_batch.detach() - B_init).pow(2).sum(dim=(1, 2))
        return mem_norm_sq.sqrt(), delta_mem_norm_sq.sqrt()

    def attach_return_memory(self, output, memory_state):
        output["lora_mem"] = memory_state["lora_mem"]


class KVCacheMemoryBackend(MemoryBackend):
    def init_memory_state(self, batch_size):
        o = self.owner
        device = o.model.get_input_embeddings().weight.device
        memory_state = {"kv_mem": {}}
        memory_state_initial = {"kv_mem": {}}
        for layer_idx in o.kv_mem_layer_ids:
            K0 = o.kv_mem_K0[str(layer_idx)].to(device=device)
            V0 = o.kv_mem_V0[str(layer_idx)].to(device=device)
            K_batch = K0.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
            V_batch = V0.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
            if o.grad_mode == "none":
                K_batch = K_batch.detach().requires_grad_(True)
                V_batch = V_batch.detach().requires_grad_(True)
            else:
                K_batch = K_batch.requires_grad_(True)
                V_batch = V_batch.requires_grad_(True)
            memory_state["kv_mem"][layer_idx] = (K_batch, V_batch)
            memory_state_initial["kv_mem"][layer_idx] = (K_batch.detach().clone(), V_batch.detach().clone())
        return memory_state, memory_state_initial

    def prepare_batch(self, context_input_ids, query_input_ids, pad_id):
        o = self.owner
        emb_layer = o.model.get_input_embeddings()
        ctx_emb = emb_layer(context_input_ids)
        qry_emb = emb_layer(query_input_ids)

        lm_labels = context_input_ids.clone()
        lm_labels[lm_labels == pad_id] = -100
        mask = (lm_labels != -100)

        x_ctx = ctx_emb
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_n = (-x_ctx.size(1)) % 32
            if pad_n:
                x_ctx = F.pad(x_ctx, [0, 0, 0, pad_n], "constant", 0)
                lm_labels = F.pad(lm_labels, [0, pad_n], "constant", -100)
                mask = F.pad(mask, [0, pad_n], "constant", 0)

        return {
            "x_ctx": x_ctx,
            "lm_labels": lm_labels,
            "mask": mask,
            "qry_emb": qry_emb,
        }

    def build_write_inputs(self, memory_state, batch_ctx):
        return {
            "inputs_embeds": batch_ctx["x_ctx"],
            "lm_labels": batch_ctx["lm_labels"],
            "mask": batch_ctx["mask"],
            "logits_start": 0,
            "label_shift": 1,
            "model_kwargs": {
                "past_key_values": self.owner._build_dynamic_cache(memory_state["kv_mem"]),
                "use_cache": True,
            },
        }

    def build_read_inputs(self, memory_state, batch_ctx):
        o = self.owner
        x_qry = batch_ctx["qry_emb"]
        if o.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            x_qry = F.pad(x_qry, [0, 0, 0, (-x_qry.size(1)) % 32], "constant", 0)
        return {
            "inputs_embeds": x_qry,
            "logits_start": 0,
            "pred_len": batch_ctx["qry_emb"].size(1),
            "label_shift": 1,
            "model_kwargs": {
                "past_key_values": self.owner._build_dynamic_cache(memory_state["kv_mem"]),
                "use_cache": True,
            },
        }

    def inner_params(self, memory_state):
        params = []
        for layer_idx in self.owner.kv_mem_layer_ids:
            K_batch, V_batch = memory_state["kv_mem"][layer_idx]
            params.extend([K_batch, V_batch])
        return params

    def assign_inner_params(self, memory_state, new_params):
        p = 0
        for layer_idx in self.owner.kv_mem_layer_ids:
            memory_state["kv_mem"][layer_idx] = (new_params[p], new_params[p + 1])
            p += 2

    def maybe_detach_after_step(self, memory_state):
        if self.owner.grad_mode != "none":
            return
        for layer_idx in self.owner.kv_mem_layer_ids:
            K_batch, V_batch = memory_state["kv_mem"][layer_idx]
            memory_state["kv_mem"][layer_idx] = (
                K_batch.detach().requires_grad_(True),
                V_batch.detach().requires_grad_(True),
            )

    def compute_memory_stats(self, memory_state, memory_state_initial):
        mem_device = next(iter(memory_state["kv_mem"].values()))[0].device
        mem_norm_sq = torch.zeros(next(iter(memory_state["kv_mem"].values()))[0].size(0), device=mem_device)
        delta_mem_norm_sq = torch.zeros_like(mem_norm_sq)
        for layer_idx in self.owner.kv_mem_layer_ids:
            K_batch, V_batch = memory_state["kv_mem"][layer_idx]
            K_init, V_init = memory_state_initial["kv_mem"][layer_idx]
            mem_norm_sq = mem_norm_sq + K_batch.detach().pow(2).sum(dim=(1, 2, 3))
            mem_norm_sq = mem_norm_sq + V_batch.detach().pow(2).sum(dim=(1, 2, 3))
            delta_mem_norm_sq = delta_mem_norm_sq + (K_batch.detach() - K_init).pow(2).sum(dim=(1, 2, 3))
            delta_mem_norm_sq = delta_mem_norm_sq + (V_batch.detach() - V_init).pow(2).sum(dim=(1, 2, 3))
        return mem_norm_sq.sqrt(), delta_mem_norm_sq.sqrt()

    def attach_return_memory(self, output, memory_state):
        output["kv_mem"] = memory_state["kv_mem"]


class GradMemGPT(PreTrainedModel):
    """
    Transformer-decoder backbone + writable prefix memory (n_mem_tokens x d).
    """
    config_class = GradMemGPTConfig

    def __init__(self, config):
        """
        grad_mode: "none" | "first" | "second"
        none: stop grad in inner update. Outer optimizer ignores mem pathway.
            Initial params of self.mem are never trained. Per-sample memory is updated in inner loop.
        first: first-order update. Outer grads flow to mem, but ignore Hessian term (Straight-Through / FOMAML).
            Only outer loop gradients update self.mem, inner loop (second-order) gradients are ignored:
            self.mem.grad = mem_batch.grad.sum(0)
        second: second-order update. Full MAML. Outer grads include second-order term via a differentiable inner step.

        mem_proj_mode: "none" | "proj" | "per_sample"
        none: no linear projection of mem
        proj: one shared nn.Linear trained by the outer loop only, acts like a gate/preconditioner/tuned inner lr
            mem* = W x mem + b
            mem = mem - lr x W^T x grad -- inner loop update of per-sample mem
            -> mem* = W x (mem - lr x W^T x grad) + b = (W x mem + b) - lr x (W x W^T) x grad
            so W x W^T is a preconditioner of how to apply grads
            special cases:
                W = sI -- tuned learning rate
                W is diagonal -- per-dimension learning rates
                full-rank W -- mixing + scaling, more complex preconditioner
                todo: add constraints on W, e.g. W is diagonal, W is low-rank, ...
        per_sample: per-sample fast weights W_i,b_i updated in the inner loop;
            their initial values (self.mem_proj.*) are meta-learned by the outer loop

        inner loop: [write_st][mem][write_end][context]
        outer loop: [read_st][mem][read_end][query][target]

        write_st/write_end/read_st/read_end are parameters aka prompts, that can be used by model to control
            the write/read operation.
        n_ctrl_tokens = 1 means that [write_st] is a single token.

        mem is updated in inner loop, write_ctrl/read_ctrl/model_params/init_mem are trained by outer loop
        """
        super().__init__(config)

        if config.pretrained_model is not None and config.base_config is not None:
            raise ValueError("Only one of pretrained_model or base_config should be provided")
        if config.pretrained_model is None and config.base_config is None:
            raise ValueError("Either pretrained_model or base_config must be provided to instantiate GradMemGPT")

        # initialize base model, attention is eager to support backward pass over backward pass
        if config.pretrained_model is not None:
            self.model = AutoModelForCausalLM.from_pretrained(config.pretrained_model,
                                                              attn_implementation=config.attn_implementation)
        else:
            self.model = AutoModelForCausalLM.from_config(config.base_config,
                                                          attn_implementation=config.attn_implementation)
        self.attn_implementation = config.attn_implementation
        if self.attn_implementation in ("jvp_flash", "hvp_semi_manual") and attn_double_bwd is None:
            raise ImportError(
                f"attn_implementation={self.attn_implementation} requires triton/attn_double_bwd. "
                "Install triton or use a standard attention implementation (e.g. eager)."
            )
        if attn_double_bwd is None and _is_main_process():
            logger.info("triton/attn_double_bwd is not available; using standard attention kernels")

        self.memory_backend = getattr(config, "memory_backend", "prefix")
        self.lora_mem_placement = getattr(config, "lora_mem_placement", "between_layers")
        self.lora_mem_r = getattr(config, "lora_mem_r", 8)
        self.lora_mem_alpha = getattr(config, "lora_mem_alpha", 16)
        self.lora_mem_dropout = getattr(config, "lora_mem_dropout", 0.0)
        self.lora_mem_layers = getattr(config, "lora_mem_layers", None)
        self.lora_mem_target_modules = getattr(config, "lora_mem_target_modules", None)
        self.kv_mem_layers = getattr(config, "kv_mem_layers", None)
        self.lora_mem_slot_ids = []
        self.lora_mem_param_keys = {}
        self._active_lora_memory = None
        self._lora_mem_hooks = []

        # write-phase LoRA (applies only during inner loop), additional params to train for WRITE operation
        self.use_write_lora = getattr(config, "use_write_lora", False)
        self.write_lora_r = getattr(config, "write_lora_r", 8)
        self.write_lora_alpha = getattr(config, "write_lora_alpha", 16)
        self.write_lora_dropout = getattr(config, "write_lora_dropout", 0.0)
        self.write_lora_target_modules = getattr(config, "write_lora_target_modules", None)
        self.freeze_backbone = getattr(config, "freeze_backbone", False)
        if self.use_write_lora:
            self._init_write_lora()
        if self.freeze_backbone:
            self._freeze_backbone_params()

        if self.memory_backend == "lora":
            if self.lora_mem_placement == "between_layers":
                self._init_lora_memory_between_layers()
            elif self.lora_mem_placement == "target_modules":
                self._init_lora_memory_target_modules()
            else:
                raise ValueError(
                    f"Unsupported lora_mem_placement={self.lora_mem_placement}. "
                    "Supported: between_layers, target_modules"
                )
        elif self.memory_backend == "kv_cache":
            self._init_kv_cache_memory()

        if self.memory_backend == "prefix":
            self.memory_backend_impl = InputPrefixMemoryBackend(self)
        elif self.memory_backend == "lora":
            self.memory_backend_impl = LoraMemoryBackend(self)
        elif self.memory_backend == "kv_cache":
            self.memory_backend_impl = KVCacheMemoryBackend(self)
        else:
            raise ValueError(f"Unsupported memory_backend={self.memory_backend}. Supported: prefix, lora, kv_cache")

        # store GradMemGPT parameters
        self.n_mem_tokens = config.n_mem_tokens
        self.n_ctrl_tokens = config.n_ctrl_tokens
        self.K = config.K
        self.last_K_second_order = config.last_K_second_order
        self.lr = config.lr
        self.use_adam = config.use_adam
        self.grad_mode = config.grad_mode
        self.inner_clip_value = config.inner_clip_value
        self.inner_clip_norm = config.inner_clip_norm
        self.use_mem_proj = config.use_mem_proj
        self.mem_proj_mode = config.mem_proj_mode
        if self.memory_backend == "lora" and self.mem_proj_mode != "none":
            raise ValueError("mem_proj_mode is not supported for memory_backend='lora'")
        if self.memory_backend == "kv_cache" and self.mem_proj_mode == "per_sample":
            raise ValueError(
                "mem_proj_mode='per_sample' is not supported for memory_backend='kv_cache'; "
                "use mem_proj_mode='none' or 'proj'"
            )
        self.use_write_head = config.use_write_head
        self.write_objective = getattr(config, "write_objective", "reconstruction")
        self.energy_head_hidden_dim = getattr(config, "energy_head_hidden_dim", None)
        self.use_layerwise_energy = bool(getattr(config, "use_layerwise_energy", False))
        self.write_reconstruction_weight = float(getattr(config, "write_reconstruction_weight", 1.0))
        self.write_energy_weight = float(getattr(config, "write_energy_weight", 1.0))
        self.energy_rank_weight = float(getattr(config, "energy_rank_weight", 0.0) or 0.0)
        self.energy_traj_weight = float(getattr(config, "energy_traj_weight", 0.0) or 0.0)
        self.energy_margin = float(getattr(config, "energy_margin", 0.1))
        self.energy_traj_margin = float(getattr(config, "energy_traj_margin", 0.0))
        self.energy_rank_temperature = float(getattr(config, "energy_rank_temperature", 1.0))
        self.energy_mix_alpha = float(getattr(config, "energy_mix_alpha", 0.75))
        self.energy_anchor_weight = float(getattr(config, "energy_anchor_weight", 0.0) or 0.0)
        self.lipschitz_weight = float(getattr(config, "lipschitz_weight", 0.0) or 0.0)
        self.lipschitz_constraint = float(getattr(config, "lipschitz_constraint", 1.0))
        self.energy_memory_search_weight = float(
            getattr(config, "energy_memory_search_weight", 0.0) or 0.0
        )
        self.energy_memory_search_num_samples = int(
            getattr(config, "energy_memory_search_num_samples", 4)
        )
        self.energy_memory_search_radius_scale = float(
            getattr(config, "energy_memory_search_radius_scale", 0.25)
        )
        self.energy_memory_search_use_gain_weighting = bool(
            getattr(config, "energy_memory_search_use_gain_weighting", False)
        )
        self.energy_memory_search_gain_ema_decay = float(
            getattr(config, "energy_memory_search_gain_ema_decay", 0.99)
        )
        self.energy_memory_search_min_relative_target_gain = float(
            getattr(config, "energy_memory_search_min_relative_target_gain", 0.0)
        )
        self.energy_memory_search_use_best_for_next_step = bool(
            getattr(config, "energy_memory_search_use_best_for_next_step", False)
        )
        self.read_focal_gamma = float(getattr(config, "read_focal_gamma", 0.0))
        self.read_loss_alignment = getattr(config, "read_loss_alignment", "causal")
        self.memory_alignment_weight = float(getattr(config, "memory_alignment_weight", 0.0) or 0.0)
        self.step_alignment_weight = float(getattr(config, "step_alignment_weight", 0.0) or 0.0)
        self.align_last_step = bool(getattr(config, "align_last_step", False))
        self.grad_align_norm = getattr(config, "grad_align_norm", "none")
        self.intermediate_read_weight = float(getattr(config, "intermediate_read_weight", 0.0) or 0.0)
        self.memory_noise_sigma = float(getattr(config, "memory_noise_sigma", 0.0) or 0.0)
        self.orthogonal_loss_weight = float(getattr(config, "orthogonal_loss_weight", 0.0) or 0.0)
        self.ivan_loss_weight = float(getattr(config, "ivan_loss_weight", 0.0) or 0.0)
        orthogonal_alpha_trainable = self.memory_backend == "prefix" and self.K > 0 and self.lr > 0.0
        if orthogonal_alpha_trainable:
            alpha_init = self.lr if self.lr > 20.0 else math.log(math.expm1(self.lr))
            self.orthogonal_alpha_raw = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_parameter("orthogonal_alpha_raw", None)
        if self.energy_rank_temperature <= 0.0:
            raise ValueError("energy_rank_temperature must be > 0")
        if not math.isfinite(self.read_focal_gamma) or self.read_focal_gamma < 0.0:
            raise ValueError("read_focal_gamma must be finite and >= 0")
        if self.read_loss_alignment not in ("causal", "query_position"):
            raise ValueError("read_loss_alignment must be one of: causal, query_position")
        if not 0.0 <= self.energy_mix_alpha <= 1.0:
            raise ValueError("energy_mix_alpha must be within [0, 1]")
        if not math.isfinite(self.energy_memory_search_weight) or self.energy_memory_search_weight < 0.0:
            raise ValueError("energy_memory_search_weight must be finite and >= 0")
        if self.energy_memory_search_num_samples < 1:
            raise ValueError("energy_memory_search_num_samples must be >= 1")
        if (
            not math.isfinite(self.energy_memory_search_radius_scale)
            or self.energy_memory_search_radius_scale <= 0.0
        ):
            raise ValueError("energy_memory_search_radius_scale must be finite and > 0")
        if (
            not math.isfinite(self.energy_memory_search_gain_ema_decay)
            or not 0.0 <= self.energy_memory_search_gain_ema_decay < 1.0
        ):
            raise ValueError("energy_memory_search_gain_ema_decay must be finite and within [0, 1)")
        if (
            not math.isfinite(self.energy_memory_search_min_relative_target_gain)
            or self.energy_memory_search_min_relative_target_gain < 0.0
        ):
            raise ValueError("energy_memory_search_min_relative_target_gain must be finite and >= 0")
        if self.energy_memory_search_use_best_for_next_step and self.energy_memory_search_weight <= 0.0:
            raise ValueError(
                "energy_memory_search_use_best_for_next_step requires energy_memory_search_weight > 0"
            )
        self.register_buffer("energy_memory_search_gain_ema", torch.zeros(self.K, dtype=torch.float32))
        self.register_buffer(
            "energy_memory_search_gain_ema_initialized",
            torch.zeros(self.K, dtype=torch.bool),
        )
        if self.write_objective in ("energy", "energy_with_reconstruction") and self.memory_backend != "prefix":
            raise ValueError(
                "write_objective='energy' and 'energy_with_reconstruction' are currently supported only "
                "for memory_backend='prefix'"
            )
        self.add_inner_loss_to_outer = config.add_inner_loss_to_outer
        self.inner_loss_weight = config.inner_loss_weight
        if self.add_inner_loss_to_outer:
            if self.inner_loss_weight is None:
                self.inner_loss_weight = 1.0
        else:
            self.inner_loss_weight = 0.0

        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        if self.memory_backend == "prefix":
            # self.mem are inner loop per-sample params, intial states of mem (self.mem) are meta-learned
            self.mem = nn.Parameter(torch.randn(self.n_mem_tokens, n_embd) * 0.02)

            # optional mem projection linear layer
            if self.mem_proj_mode != "none":
                self.mem_proj = nn.Linear(n_embd, n_embd, bias=True)
                # initialize mem_proj to be identity
                with torch.no_grad():
                    nn.init.eye_(self.mem_proj.weight)
                    self.mem_proj.bias.zero_()
            else:
                self.mem_proj = None
        else:
            self.mem = None
            self.mem_proj = None

        # optional read/write control parameters (shape = n_ctrl_tokens × d)
        if self.n_ctrl_tokens > 0:
            # write ctrl tokens can be trained only by outer loop and only if grads flow through inner loop ("second")
            self.write_st = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
            self.write_end = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
            self.read_st = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
            self.read_end = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)

        if self.use_write_head:
            V = self.model.config.vocab_size
            self.write_head = nn.Linear(n_embd, V, bias=False)

            if hasattr(self.model, 'get_output_embeddings'):
                head_params = self.model.get_output_embeddings().weight
            else:  # fallback to input embeddings
                head_params = self.model.get_input_embeddings().weight
            with torch.no_grad():
                self.write_head.weight.copy_(head_params.detach())

        if self.write_objective in ("energy", "energy_with_reconstruction"):
            energy_hidden = self.energy_head_hidden_dim or n_embd
            if self.use_layerwise_energy:
                n_layers = len(self._get_transformer_blocks())
                self.energy_ln = nn.ModuleList(nn.LayerNorm(n_embd) for _ in range(n_layers))
                self.energy_head = nn.ModuleList(
                    nn.Sequential(
                        nn.Linear(n_embd, energy_hidden),
                        nn.SiLU(),
                        nn.Linear(energy_hidden, 1),
                    )
                    for _ in range(n_layers)
                )
            else:
                self.energy_ln = nn.LayerNorm(n_embd)
                self.energy_head = nn.Sequential(
                    nn.Linear(n_embd, energy_hidden),
                    nn.SiLU(),
                    nn.Linear(energy_hidden, 1),
                )
        else:
            self.energy_ln = None
            self.energy_head = None

        self.tie_weights()
        self.main_input_name = "input_ids"
        self.model.config.use_cache = False
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # turn on gradient checkpointing to save gpu ram
        # currently, gradient checkpointing is not used in inner loop, so it wont save GPU RAM at forward pass.
        # but it's still saves some GPU RAM for training in outer loop
        if getattr(config, "use_gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

        n_memory_params = self._count_memory_parameters()
        if _is_main_process():
            logger.info(f"GradMemGPT memory params (backend={self.memory_backend}): total={n_memory_params}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.use_write_lora:
            # PEFT nests the causal LM and wraps targeted weights in base_layer.
            # Accept both plain causal-LM and unwrapped GradMem checkpoints.
            wrapped_model_prefix = "model.base_model.model."
            consumed_keys = set()
            for relative_target_key in self.state_dict():
                if not relative_target_key.startswith(wrapped_model_prefix):
                    continue
                target_key = prefix + relative_target_key
                if target_key in state_dict:
                    continue
                unwrapped_suffix = relative_target_key[len(wrapped_model_prefix):].replace(
                    ".base_layer.", "."
                )
                source_candidates = (
                    prefix + unwrapped_suffix,
                    prefix + "model." + unwrapped_suffix,
                )
                for source_key in source_candidates:
                    if source_key in state_dict:
                        state_dict[target_key] = state_dict[source_key]
                        consumed_keys.add(source_key)
                        break
            for source_key in consumed_keys:
                state_dict.pop(source_key)

        # Checkpoints created before per-depth gain tracking stored scalar buffers.
        for name in (
            "energy_memory_search_gain_ema",
            "energy_memory_search_gain_ema_initialized",
        ):
            key = prefix + name
            value = state_dict.get(key)
            target = getattr(self, name)
            if value is not None and value.numel() == 1 and value.shape != target.shape:
                state_dict[key] = value.reshape(1).expand_as(target).clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def floating_point_ops(self, inputs):
        # dummy method to satisfy base class and it's invocation by trainer:
        # Trainer supposes that `inputs`` is a tensor, not dict.
        return 0

    def tie_weights(self):
        self.model.tie_weights()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # force {"use_reentrant": False}
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                # fallback for older HF versions
                self.model.gradient_checkpointing_enable()

    def _count_memory_parameters(self):
        if self.memory_backend == "prefix":
            return 0 if self.mem is None else self.mem.numel()
        if self.memory_backend == "lora":
            total = 0
            for p in self.lora_mem_A0.values():
                total += p.numel()
            for p in self.lora_mem_B0.values():
                total += p.numel()
            return total
        if self.memory_backend == "kv_cache":
            total = 0
            for p in self.kv_mem_K0.values():
                total += p.numel()
            for p in self.kv_mem_V0.values():
                total += p.numel()
            return total
        raise ValueError(f"Unsupported memory_backend={self.memory_backend}")

    @staticmethod
    def _parse_lora_targets(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            targets = [str(v).strip() for v in value if str(v).strip()]
            return targets or None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "" or cleaned.lower() in ("none", "auto"):
                return None
            targets = [v.strip() for v in cleaned.split(",") if v.strip()]
            return targets or None
        return None

    def _resolve_write_lora_targets(self):
        parsed = self._parse_lora_targets(self.write_lora_target_modules)
        if parsed:
            return parsed

        model_type = getattr(self.model.config, "model_type", None)
        targets_by_type = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt2": ["c_attn", "c_proj", "c_fc"],
            "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        }
        if model_type in targets_by_type:
            return targets_by_type[model_type]

        raise ValueError(
            "write_lora_target_modules is not set and model_type is unknown. "
            "Please provide explicit target modules."
        )

    def _init_write_lora(self):
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise ImportError("peft is required for write_lora. Please install the peft package.")

        target_modules = self._resolve_write_lora_targets()
        lora_config = LoraConfig(
            r=self.write_lora_r,
            lora_alpha=self.write_lora_alpha,
            lora_dropout=self.write_lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        if not (hasattr(self.model, "disable_adapter_layers") and hasattr(self.model, "enable_adapter_layers")):
            raise RuntimeError("PEFT model does not support adapter toggling; cannot enforce write-only LoRA.")

        # keep base model trainable to preserve previous behavior
        for param in self.model.parameters():
            param.requires_grad = True

    def _freeze_backbone_params(self):
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        if self.use_write_lora:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    @contextmanager
    def _disable_write_lora(self):
        if not self.use_write_lora:
            yield
            return
        self.model.disable_adapter_layers()
        try:
            yield
        finally:
            self.model.enable_adapter_layers()

    def _adam_step(self, p, g, state, step_idx, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip_value=10.0):
        """
        Functional Adam:
        - no in-place math on graph-tracked tensors
        - buffers detached, keep them in state dict
        - bias-correction uses step_idx
        - gradient norm is clipped

        Current impl computes updates under torch.no_grad(), so MAML/second-order paths are cut (by design).
        With grad_mode="second" it will still give first-order behavior.

        WARNING: not tested throughly!
        """
        # ---- 0. init ------------------------------------------------------ #
        if "m" not in state:
            state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        m, v = state["m"], state["v"]

        # ---- 1. clip grad to stabilise very first steps ------------------ #
        if clip_value is not None:
            g = torch.clamp(g, min=-clip_value, max=clip_value)

        # ---- 2. update moments (stay outside autograd graph) ------------- #
        with torch.no_grad():
            m_new = beta1 * m + (1 - beta1) * g
            v_new = beta2 * v + (1 - beta2) * (g * g)

            # bias‑correction with current step
            m_hat = m_new / (1 - beta1 ** step_idx)
            v_hat = v_new / (1 - beta2 ** step_idx)
            v_hat = torch.clamp(v_hat, min=eps)        # avoid √0 / 0

            step = lr * m_hat / (v_hat.sqrt() + eps)

        # ---- 3. write back *detached* buffers ---------------------------- #
        state["m"].copy_(m_new.detach())
        state["v"].copy_(v_new.detach())

        # ---- 4. return new parameter tensor (on graph via g) ------------- #
        return p - step

    def _sgd_step(self, p, g, lr=None, clip_value=None, clip_norm=None):
        """
        Stateless SGD with optional element-wise and per-sample total-norm clipping.
        Works for shapes (B,M,d), (B,d,d), (B,d).

        Args
        ----
        p : torch.Tensor          current parameter tensor
        g : torch.Tensor          gradient wrt p
        clip_value : float|None   element-wise clamp, e.g. 5.0
        clip_norm  : float|None   total-norm clamp, e.g. 1.0
        """

        if clip_value is not None:
            # simple element-wise clamp
            g = torch.clamp(g, -clip_value, clip_value)

        if clip_norm is not None:
            # scale gradient if its 2-norm is too large
            # check grad for each sample separately as we do per-sample optimization
            reduce_dims = tuple(range(1, g.ndim))           # all non-batch dims
            g_norm = g.norm(dim=reduce_dims, keepdim=True)  # (B,1,1)
            scale = clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > clip_norm, g * scale, g)

        if lr is None:
            lr = self.lr

        return p - lr * g

    @staticmethod
    def _apply_linear(mem, W, b):
        """
        Functional linear on a batch of memories:
        mem: (B,M,d), W: (B,d,d) or None, b: (B,d) or None
        """
        if W is None:
            return mem
        return torch.baddbmm(b.unsqueeze(1), mem, W.transpose(1, 2))

    def _compute_write_energy(self, hidden_states, write_batch):
        if self.energy_head is None:
            raise RuntimeError("energy_head is not initialized")

        def compute_layer_energy(layer_hidden, layer_norm, energy_head):
            context_hidden = layer_hidden[:, write_batch['context_start']:, :]
            context_mask = write_batch['mask'].to(device=context_hidden.device, dtype=context_hidden.dtype)
            if context_hidden.size(1) != context_mask.size(1):
                raise ValueError(
                    "Invalid energy context span: "
                    f"context_hidden_len={context_hidden.size(1)}, mask_len={context_mask.size(1)}, "
                    f"context_start={write_batch['context_start']}"
                )
            token_energy = F.softplus(energy_head(layer_norm(context_hidden))).squeeze(-1)
            return (token_energy * context_mask).sum(dim=1) / context_mask.sum(dim=1).clamp_min(1)

        if not self.use_layerwise_energy:
            return compute_layer_energy(hidden_states, self.energy_ln, self.energy_head)
        if len(hidden_states) != len(self.energy_head):
            raise ValueError(
                f"Expected hidden states for {len(self.energy_head)} transformer layers, "
                f"got {len(hidden_states)}"
            )
        layer_energies = [
            compute_layer_energy(layer_hidden, layer_norm, energy_head)
            for layer_hidden, layer_norm, energy_head in zip(hidden_states, self.energy_ln, self.energy_head)
        ]
        return torch.stack(layer_energies, dim=0).sum(dim=0)

    def _energy_hidden_states(self, outs):
        if not self.use_layerwise_energy:
            return outs.last_hidden_state
        if outs.hidden_states is None:
            raise RuntimeError("Layer-wise energy requires model hidden states")
        return outs.hidden_states[1:]

    def _run_energy_write_forward(self, write_batch):
        model_kwargs = dict(write_batch.get("model_kwargs", {}))
        if "attention_mask" in write_batch:
            model_kwargs["attention_mask"] = write_batch["attention_mask"]
        model_kwargs["output_hidden_states"] = self.use_layerwise_energy
        outs = get_backbone(self.model)(
            inputs_embeds=write_batch["inputs_embeds"],
            return_dict=True,
            **model_kwargs,
        )
        return outs, self._compute_write_energy(self._energy_hidden_states(outs), write_batch)

    @staticmethod
    def _fixed_point_free_permutation(batch_size, device):
        if batch_size <= 1:
            return None
        shift = torch.randint(1, batch_size, (1,), device=device)
        return (torch.arange(batch_size, device=device) + shift) % batch_size

    @classmethod
    def _build_energy_negative_memories(cls, positive_mem, initial_mem, mix_alpha):
        """Build detached context-mismatched, interpolated, and radius-matched memories."""
        positive_mem = positive_mem.detach()
        initial_mem = initial_mem.to(device=positive_mem.device, dtype=positive_mem.dtype).detach()
        batch_size = positive_mem.size(0)

        permutation = cls._fixed_point_free_permutation(batch_size, positive_mem.device)
        if permutation is None:
            deranged_mem = None
            interpolated_mem = None
        else:
            deranged_mem = positive_mem.index_select(0, permutation)
            interpolated_mem = mix_alpha * positive_mem + (1.0 - mix_alpha) * deranged_mem

        random_direction = torch.randn_like(positive_mem)
        flat_direction = random_direction.reshape(batch_size, -1)
        direction_norm = flat_direction.norm(dim=1).clamp_min(torch.finfo(positive_mem.dtype).eps)
        write_radius = (positive_mem - initial_mem).reshape(batch_size, -1).norm(dim=1)
        scale_shape = (batch_size,) + (1,) * (positive_mem.ndim - 1)
        random_direction = random_direction / direction_norm.view(scale_shape)
        random_mem = initial_mem + random_direction * write_radius.view(scale_shape)

        return {
            "deranged": deranged_mem,
            "interpolated": interpolated_mem,
            "random": random_mem,
        }, permutation

    def _run_energy_memory_candidate(self, backend, memory_template, batch_ctx, mem_batch):
        candidate_state = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in memory_template.items()
        }
        candidate_state["mem_batch"] = mem_batch.detach()
        write_batch = backend.build_write_inputs(candidate_state, batch_ctx)
        with backend.activation_context(candidate_state):
            outs, energy = self._run_energy_write_forward(write_batch)
        del outs
        return energy

    def _compute_energy_landscape_losses(self, backend, memory_state, memory_state_initial, batch_ctx, *,
                                         compute_rank, compute_anchor,):
        if not compute_rank and not compute_anchor:
            raise ValueError("At least one energy landscape objective must be active")

        positive_mem = memory_state["mem_batch"].detach()
        energy_positive = self._run_energy_memory_candidate(backend, memory_state, batch_ctx, positive_mem)

        negative_energies = {}
        negative_losses = {}
        if compute_rank:
            negative_memories, _ = self._build_energy_negative_memories(
                positive_mem,
                memory_state_initial["mem_batch"],
                self.energy_mix_alpha,
            )
            for name, negative_mem in negative_memories.items():
                if negative_mem is None:
                    continue
                energy_negative = self._run_energy_memory_candidate(
                    backend, memory_state, batch_ctx, negative_mem
                )
                negative_energies[name] = energy_negative
                negative_losses[name] = F.softplus(
                    (energy_positive - energy_negative + self.energy_margin)
                    / self.energy_rank_temperature
                ).mean()

        rank_loss = None
        if compute_rank:
            rank_loss = torch.stack(list(negative_losses.values())).mean()

        anchor_loss = None
        if compute_anchor:
            # Negative energies are included only when ranking already needed
            # those candidates. Anchor-only training regularizes the detached
            # positive state without constructing unused negative memories.
            anchor_energies = [energy_positive] + list(negative_energies.values())
            anchor_loss = torch.stack([energy.pow(2).mean() for energy in anchor_energies]).mean()

        return {
            "rank_loss": rank_loss,
            "anchor_loss": anchor_loss,
            "energy_positive": energy_positive,
            "negative_energies": negative_energies,
            "negative_losses": negative_losses,
        }

    def _compute_lipschitz_constraint_loss(self, backend, memory_state, batch_ctx, *, create_graph):
        """Measure and optionally constrain the WRITE gradient at detached M_K locations."""
        candidate_state = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in memory_state.items()
        }
        candidate_memory = memory_state["mem_batch"].detach().requires_grad_(True)
        candidate_state["mem_batch"] = candidate_memory
        write_batch = backend.build_write_inputs(candidate_state, batch_ctx)
        with backend.activation_context(candidate_state):
            if self.write_objective == "reconstruction":
                outs, reconstruction_loss = self._run_reconstruction_write_forward(write_batch)
                write_objective = reconstruction_loss
            else:
                outs, reconstruction_loss, energy_loss = self._run_energy_reconstruction_write_forward(
                    write_batch
                )
                if self.write_objective == "energy":
                    write_objective = energy_loss
                else:
                    write_objective = (
                        self.write_reconstruction_weight * reconstruction_loss
                        + self.write_energy_weight * energy_loss
                    )
        del outs
        memory_gradient = torch.autograd.grad(
            write_objective.sum(),
            candidate_memory,
            create_graph=create_graph,
        )[0]
        gradient_norm = memory_gradient.float().flatten(1).norm(dim=1)
        loss = F.relu(gradient_norm - self.lipschitz_constraint).square().mean()
        return loss, gradient_norm

    def _compute_write_reconstruction_loss(self, logits, write_batch):
        logits_loss = logits[:, :-1]
        label_shift = write_batch.get('label_shift', 0)
        labels_loss = write_batch['lm_labels'][:, label_shift:]
        mask_loss = write_batch['mask'][:, label_shift:]
        logits_len = logits_loss.size(1)
        labels_len = labels_loss.size(1)
        mask_len = mask_loss.size(1)
        if (logits_len != labels_len) or (labels_len != mask_len) or (labels_len == 0):
            raise ValueError(
                "Invalid inner-loop alignment: "
                f"backend={self.memory_backend}, "
                f"logits_len={logits_len}, labels_len={labels_len}, mask_len={mask_len}, "
                f"logits_start={write_batch['logits_start']}, label_shift={label_shift}, "
                f"mismatch_logits_labels={logits_len != labels_len}, "
                f"mismatch_labels_mask={labels_len != mask_len}, "
                f"empty_training_tokens={labels_len == 0}"
            )

        loss = nn.functional.cross_entropy(
            logits_loss.reshape(-1, logits.size(-1)),
            labels_loss.reshape(-1),
            ignore_index=-100,
            reduction='none',
        ).view(logits.size(0), -1)
        seq_len = mask_loss.sum(dim=1).clamp_min(1)
        loss = (loss * mask_loss).sum(1) / seq_len
        return loss

    def _run_reconstruction_write_forward(self, write_batch):
        model_kwargs = dict(write_batch.get('model_kwargs', {}))
        if 'attention_mask' in write_batch:
            model_kwargs['attention_mask'] = write_batch['attention_mask']
        if self.use_write_head:
            outs = get_backbone(self.model)(inputs_embeds=write_batch['inputs_embeds'],
                                            return_dict=True,
                                            **model_kwargs)
            hidden = outs.last_hidden_state[:, write_batch['logits_start']:, :]
            logits = self.write_head(hidden)
        else:
            # outs = self.model(inputs_embeds=write_batch['inputs_embeds'],
            #                   return_dict=True,
            #                   **model_kwargs)
            # logits = outs.logits[:, write_batch['logits_start']:, :]
            outs = get_backbone(self.model)(inputs_embeds=write_batch['inputs_embeds'],
                                            return_dict=True,
                                            **model_kwargs)
            hidden = outs.last_hidden_state[:, write_batch['logits_start']:, :]
            logits = self.model.lm_head(hidden)

        loss = self._compute_write_reconstruction_loss(logits, write_batch)
        return outs, loss

    def _run_energy_reconstruction_write_forward(self, write_batch):
        model_kwargs = dict(write_batch.get('model_kwargs', {}))
        if 'attention_mask' in write_batch:
            model_kwargs['attention_mask'] = write_batch['attention_mask']
        model_kwargs['output_hidden_states'] = self.use_layerwise_energy
        outs = get_backbone(self.model)(
            inputs_embeds=write_batch['inputs_embeds'],
            return_dict=True,
            **model_kwargs,
        )
        hidden = outs.last_hidden_state[:, write_batch['logits_start']:, :]
        if self.use_write_head:
            logits = self.write_head(hidden)
        else:
            output_embeddings = self.model.get_output_embeddings()
            if output_embeddings is None:
                raise RuntimeError("Model does not expose output embeddings for reconstruction logits")
            logits = output_embeddings(hidden)
        reconstruction_loss = self._compute_write_reconstruction_loss(logits, write_batch)
        energy_loss = self._compute_write_energy(self._energy_hidden_states(outs), write_batch)
        return outs, reconstruction_loss, energy_loss

    def _get_transformer_blocks(self):
        backbone = get_backbone(self.model)
        for attr in ("h", "layers", "block", "blocks"):
            if hasattr(backbone, attr):
                blocks = getattr(backbone, attr)
                if isinstance(blocks, (nn.ModuleList, list, tuple)):
                    return list(blocks)
        raise ValueError("Could not resolve transformer block list for LoRA memory placement")

    @staticmethod
    def _parse_lora_mem_layers(value, n_layers):
        if value is None:
            return list(range(n_layers))
        if isinstance(value, (list, tuple)):
            idx = sorted(set(int(v) for v in value))
            return idx
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in ("", "all", "none", "auto"):
                return list(range(n_layers))
            if cleaned.startswith("last_"):
                k = int(cleaned.split("last_", 1)[1])
                k = max(0, min(k, n_layers))
                return list(range(n_layers - k, n_layers))
            idx = sorted(set(int(v.strip()) for v in value.split(",") if v.strip() != ""))
            return idx
        raise ValueError("lora_mem_layers should be None, list[int], or str")

    def _init_lora_memory_between_layers(self):
        blocks = self._get_transformer_blocks()
        n_layers = len(blocks)
        layer_ids = self._parse_lora_mem_layers(self.lora_mem_layers, n_layers)
        for idx in layer_ids:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"lora_mem_layers has out-of-range index {idx} for {n_layers} layers")
        self.lora_mem_layer_ids = layer_ids
        self.lora_mem_slot_ids = list(layer_ids)

        hidden_size = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", None))
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")
        self.lora_mem_scale = float(self.lora_mem_alpha) / float(max(1, self.lora_mem_r))
        self.lora_mem_A0 = nn.ParameterDict()
        self.lora_mem_B0 = nn.ParameterDict()
        self.lora_mem_param_keys = {}
        for layer_idx in self.lora_mem_slot_ids:
            param_key = self._make_between_layer_param_key(layer_idx)
            a = nn.Parameter(
                torch.zeros(hidden_size, self.lora_mem_r),
                requires_grad=True,
            )
            b = nn.Parameter(
                torch.zeros(self.lora_mem_r, hidden_size),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)
            self.lora_mem_param_keys[str(layer_idx)] = param_key
            self.lora_mem_A0[param_key] = a
            self.lora_mem_B0[param_key] = b

        self._register_lora_mem_layer_hooks(blocks)

    @staticmethod
    def _extract_layer_idx_from_module_name(module_name):
        patterns = [
            r"\.h\.(\d+)\.",
            r"\.layers\.(\d+)\.",
            r"\.block\.(\d+)\.",
            r"\.blocks\.(\d+)\.",
        ]
        dotted = f".{module_name}."
        for p in patterns:
            m = re.search(p, dotted)
            if m is not None:
                return int(m.group(1))
        return None

    @staticmethod
    def _sanitize_param_key(text):
        cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text)).strip("_")
        return cleaned or "module"

    def _make_between_layer_param_key(self, layer_idx):
        return f"layer{int(layer_idx)}"

    def _make_target_module_param_key(self, module_name, used_keys):
        parts = str(module_name).split(".")
        leaf = parts[-1] if len(parts) >= 1 else "module"
        parent = parts[-2] if len(parts) >= 2 else "module"
        layer_idx = self._extract_layer_idx_from_module_name(module_name)
        if layer_idx is None:
            base = f"module_{parent}_{leaf}"
        else:
            base = f"layer{layer_idx}_{parent}_{leaf}"
        base = self._sanitize_param_key(base)

        key = base
        suffix = 2
        while key in used_keys:
            key = f"{base}_{suffix}"
            suffix += 1
        return key

    @staticmethod
    def _infer_linear_lora_dims(module, module_name):
        if isinstance(module, nn.Linear):
            return int(module.in_features), int(module.out_features)

        if module.__class__.__name__ == "Conv1D":
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return int(weight.shape[0]), int(weight.shape[1])

        raise ValueError(
            "lora_mem target_modules supports only linear-like modules (nn.Linear, Conv1D). "
            f"Got module={module_name}, type={module.__class__.__name__}"
        )

    def _resolve_lora_mem_targets(self):
        parsed = self._parse_lora_targets(self.lora_mem_target_modules)
        if parsed:
            return parsed

        model_type = getattr(self.model.config, "model_type", None)
        targets_by_type = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt2": ["c_attn", "c_proj", "c_fc"],
            "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        }
        if model_type in targets_by_type:
            return targets_by_type[model_type]

        raise ValueError(
            "lora_mem_target_modules is not set and model_type is unknown. "
            "Please provide explicit target modules."
        )

    def _init_lora_memory_target_modules(self):
        blocks = self._get_transformer_blocks()
        n_layers = len(blocks)
        layer_ids = set(self._parse_lora_mem_layers(self.lora_mem_layers, n_layers))
        for idx in layer_ids:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"lora_mem_layers has out-of-range index {idx} for {n_layers} layers")

        targets = self._resolve_lora_mem_targets()
        backbone = get_backbone(self.model)
        selected = []
        for module_name, module in backbone.named_modules():
            if module_name == "":
                continue
            if not any(module_name == t or module_name.endswith(f".{t}") for t in targets):
                continue
            layer_idx = self._extract_layer_idx_from_module_name(module_name)
            if layer_idx is not None and layer_idx not in layer_ids:
                continue
            selected.append((module_name, module))

        if len(selected) == 0:
            raise ValueError(
                "lora_mem_placement='target_modules' resolved zero modules. "
                f"targets={targets}, layers={sorted(layer_ids)}"
            )

        self.lora_mem_scale = float(self.lora_mem_alpha) / float(max(1, self.lora_mem_r))
        self.lora_mem_A0 = nn.ParameterDict()
        self.lora_mem_B0 = nn.ParameterDict()
        self.lora_mem_slot_ids = []
        self.lora_mem_param_keys = {}
        used_param_keys = set()
        for module_name, module in selected:
            if module_name in self.lora_mem_slot_ids:
                raise ValueError(f"Duplicate lora memory slot key: {module_name}")
            in_dim, out_dim = self._infer_linear_lora_dims(module, module_name)
            param_key = self._make_target_module_param_key(module_name, used_param_keys)
            used_param_keys.add(param_key)
            a = nn.Parameter(torch.zeros(in_dim, self.lora_mem_r), requires_grad=True)
            b = nn.Parameter(torch.zeros(self.lora_mem_r, out_dim), requires_grad=True)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)
            self.lora_mem_param_keys[str(module_name)] = param_key
            self.lora_mem_A0[param_key] = a
            self.lora_mem_B0[param_key] = b
            self.lora_mem_slot_ids.append(module_name)

        self._register_lora_mem_target_hooks(selected)

    def _init_kv_cache_memory(self):
        blocks = self._get_transformer_blocks()
        n_layers = len(blocks)
        layer_ids = self._parse_lora_mem_layers(self.kv_mem_layers, n_layers)
        for idx in layer_ids:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"kv_mem_layers has out-of-range index {idx} for {n_layers} layers")
        self.kv_mem_layer_ids = layer_ids

        hidden_size = getattr(self.model.config, "n_embd", getattr(self.model.config, "hidden_size", None))
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config")

        num_attn_heads = getattr(
            self.model.config,
            "num_attention_heads",
            getattr(self.model.config, "n_head", None),
        )
        if num_attn_heads is None:
            raise ValueError("Could not infer number of attention heads from model config")

        num_kv_heads = getattr(self.model.config, "num_key_value_heads", num_attn_heads)
        head_dim = getattr(self.model.config, "head_dim", hidden_size // num_attn_heads)

        self.kv_mem_K0 = nn.ParameterDict()
        self.kv_mem_V0 = nn.ParameterDict()
        n_mem_tokens = int(getattr(self.config, "n_mem_tokens"))
        mem_proj_mode = getattr(self.config, "mem_proj_mode", "none")
        for layer_idx in self.kv_mem_layer_ids:
            k = nn.Parameter(
                torch.randn(num_kv_heads, n_mem_tokens, head_dim) * 0.02,
                requires_grad=True,
            )
            v = nn.Parameter(
                torch.randn(num_kv_heads, n_mem_tokens, head_dim) * 0.02,
                requires_grad=True,
            )
            self.kv_mem_K0[str(layer_idx)] = k
            self.kv_mem_V0[str(layer_idx)] = v

        if mem_proj_mode == "proj":
            self.mem_proj_K_l = nn.ModuleDict()
            self.mem_proj_V_l = nn.ModuleDict()
            for layer_idx in self.kv_mem_layer_ids:
                proj_k = nn.Linear(head_dim, head_dim, bias=True)
                proj_v = nn.Linear(head_dim, head_dim, bias=True)
                with torch.no_grad():
                    nn.init.eye_(proj_k.weight)
                    proj_k.bias.zero_()
                    nn.init.eye_(proj_v.weight)
                    proj_v.bias.zero_()
                self.mem_proj_K_l[str(layer_idx)] = proj_k
                self.mem_proj_V_l[str(layer_idx)] = proj_v

    def _build_dynamic_cache(self, kv_mem):
        legacy = []
        for layer_idx in self.kv_mem_layer_ids:
            K_batch, V_batch = kv_mem[layer_idx]
            if self.mem_proj_mode == "proj":
                B, H, M, d = K_batch.shape
                K_batch = self.mem_proj_K_l[str(layer_idx)](K_batch.reshape(B, H * M, d)).reshape(B, H, M, d)
                V_batch = self.mem_proj_V_l[str(layer_idx)](V_batch.reshape(B, H * M, d)).reshape(B, H, M, d)
            legacy.append((K_batch, V_batch))
        return DynamicCache.from_legacy_cache(tuple(legacy))

    def _register_lora_mem_layer_hooks(self, blocks):
        if self._lora_mem_hooks:
            return

        for layer_idx in self.lora_mem_layer_ids:
            module = blocks[layer_idx]

            def _hook_fn(_mod, _inp, output, idx=layer_idx):
                if self._active_lora_memory is None or idx not in self._active_lora_memory:
                    return output

                x = output[0] if isinstance(output, tuple) else output
                A_mem, B_mem = self._active_lora_memory[idx]
                if self.lora_mem_dropout > 0.0:
                    x_in = F.dropout(x, p=self.lora_mem_dropout, training=self.training)
                else:
                    x_in = x
                low_rank = torch.einsum("bsd,bdr->bsr", x_in, A_mem)
                delta = torch.einsum("bsr,brd->bsd", low_rank, B_mem)
                x_new = x + self.lora_mem_scale * delta

                if isinstance(output, tuple):
                    out_list = list(output)
                    out_list[0] = x_new
                    return tuple(out_list)
                return x_new

            self._lora_mem_hooks.append(module.register_forward_hook(_hook_fn))

    def _register_lora_mem_target_hooks(self, modules):
        if self._lora_mem_hooks:
            return

        for module_name, module in modules:
            def _hook_fn(_mod, _inp, output, idx=module_name):
                if self._active_lora_memory is None or idx not in self._active_lora_memory:
                    return output

                if len(_inp) == 0:
                    raise ValueError(f"lora_mem target hook got empty inputs for module={idx}")

                x = _inp[0]
                if not isinstance(x, torch.Tensor) or x.ndim != 3:
                    raise ValueError(
                        "lora_mem target_modules expects tensor input with shape [B,S,D], "
                        f"got type={type(x)}, ndim={getattr(x, 'ndim', None)} for module={idx}"
                    )

                base_out = output[0] if isinstance(output, tuple) else output
                if not isinstance(base_out, torch.Tensor) or base_out.ndim != 3:
                    raise ValueError(
                        "lora_mem target_modules expects tensor output with shape [B,S,D], "
                        f"got type={type(base_out)}, ndim={getattr(base_out, 'ndim', None)} for module={idx}"
                    )

                A_mem, B_mem = self._active_lora_memory[idx]
                if self.lora_mem_dropout > 0.0:
                    x_in = F.dropout(x, p=self.lora_mem_dropout, training=self.training)
                else:
                    x_in = x
                low_rank = torch.einsum("bsi,bir->bsr", x_in, A_mem)
                delta = torch.einsum("bsr,bro->bso", low_rank, B_mem)
                if delta.shape != base_out.shape:
                    raise ValueError(
                        "lora_mem target delta shape mismatch: "
                        f"module={idx}, delta_shape={tuple(delta.shape)}, base_out_shape={tuple(base_out.shape)}"
                    )
                out_new = base_out + self.lora_mem_scale * delta

                if isinstance(output, tuple):
                    out_list = list(output)
                    out_list[0] = out_new
                    return tuple(out_list)
                return out_new

            self._lora_mem_hooks.append(module.register_forward_hook(_hook_fn))

    def _lora_mem_param_key(self, slot_id):
        key = self.lora_mem_param_keys.get(str(slot_id))
        if key is None:
            raise ValueError(f"Missing lora memory parameter key for slot_id={slot_id}")
        return key

    @contextmanager
    def _enable_lora_memory(self, memory_params):
        old = self._active_lora_memory
        self._active_lora_memory = memory_params
        try:
            yield
        finally:
            self._active_lora_memory = old

    @staticmethod
    def _intermediate_read_weights(K, device, dtype):
        if K <= 1:
            return torch.empty(0, device=device, dtype=dtype)
        depths = torch.arange(1, K, device=device, dtype=dtype)
        weights = (K - depths + 1).reciprocal()
        return weights / weights.sum()

    def _compute_read_target_loss(self, predictions, read_batch, labels):
        if self.read_loss_alignment == "query_position":
            # MQAR labels are attached to query positions, so score the logits
            # after each query token rather than the next-token causal shift.
            target_logits = predictions[:, 1:] if self.memory_backend == "prefix" else predictions
            target_labels = labels
            target_label_shift = "query_position"
        else:
            target_logits = predictions[:, :-1]
            target_label_shift = read_batch.get('label_shift', 0)
            # Prefix memory can predict the first target token from memory itself (label_shift=0),
            # while LoRA/KV-cache memory without a prepended seed token cannot (label_shift=1).
            target_labels = labels[:, target_label_shift:]
        if target_logits.size(1) != target_labels.size(1):
            raise ValueError(
                f"Mismatched target lengths after alignment: logits_len={target_logits.size(1)}, "
                f"labels_len={target_labels.size(1)}, label_shift={target_label_shift}"
            )
        token_losses = F.cross_entropy(
            target_logits.reshape(-1, predictions.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(target_labels)
        valid = target_labels.ne(-100)
        focal_weights = (1.0 - token_losses.neg().exp()).pow(self.read_focal_gamma)
        return (token_losses * focal_weights * valid).sum() / valid.sum().clamp_min(1)

    def _compute_per_example_read_target_loss(self, predictions, read_batch, labels):
        if self.read_loss_alignment == "query_position":
            target_logits = predictions[:, 1:] if self.memory_backend == "prefix" else predictions
            target_labels = labels
            target_label_shift = "query_position"
        else:
            target_logits = predictions[:, :-1]
            target_label_shift = read_batch.get('label_shift', 0)
            target_labels = labels[:, target_label_shift:]
        if target_logits.size(1) != target_labels.size(1):
            raise ValueError(
                f"Mismatched target lengths after alignment: logits_len={target_logits.size(1)}, "
                f"labels_len={target_labels.size(1)}, label_shift={target_label_shift}"
            )
        token_losses = F.cross_entropy(
            target_logits.reshape(-1, predictions.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(target_labels)
        valid = target_labels.ne(-100)
        focal_weights = (1.0 - token_losses.neg().exp()).pow(self.read_focal_gamma)
        return (token_losses * focal_weights * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    @contextmanager
    def _temporary_model_eval(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    def _sample_norm_preserving_memory_candidates(self, memory, previous_memory):
        """Sample on the memory norm sphere at a chord radius tied to the last WRITE step."""
        center = memory.detach()
        previous = previous_memory.detach()
        original_shape = center.shape
        center_flat = center.float().flatten(1)
        previous_flat = previous.float().flatten(1)
        batch_size, flat_dim = center_flat.shape
        num_samples = self.energy_memory_search_num_samples
        eps = torch.finfo(center_flat.dtype).eps

        center_norm = center_flat.norm(dim=1, keepdim=True)
        unit_center = center_flat / center_norm.clamp_min(eps)
        step_distance = (center_flat - previous_flat).norm(dim=1, keepdim=True)
        chord_radius = self.energy_memory_search_radius_scale * step_distance
        chord_radius = torch.minimum(chord_radius, 2.0 * center_norm * (1.0 - eps))
        angle = 2.0 * torch.asin((chord_radius / (2.0 * center_norm).clamp_min(eps)).clamp(max=1.0 - eps))

        noise = torch.randn(
            num_samples,
            batch_size,
            flat_dim,
            device=center.device,
            dtype=center_flat.dtype,
        )
        tangent = noise - (noise * unit_center.unsqueeze(0)).sum(dim=-1, keepdim=True) * unit_center.unsqueeze(0)
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        unit_tangent = tangent / tangent_norm.clamp_min(eps)
        candidates = center_norm.unsqueeze(0) * (
            angle.cos().unsqueeze(0) * unit_center.unsqueeze(0)
            + angle.sin().unsqueeze(0) * unit_tangent
        )

        valid = (center_norm.gt(eps) & step_distance.gt(eps)).unsqueeze(0) & tangent_norm.gt(eps)
        candidates = torch.where(valid, candidates, center_flat.unsqueeze(0))
        candidates = candidates.to(dtype=center.dtype)
        typed_norm = candidates.norm(dim=-1, keepdim=True)
        target_norm = center.flatten(1).norm(dim=1, keepdim=True).unsqueeze(0)
        candidates = candidates * (target_norm / typed_norm.clamp_min(torch.finfo(center.dtype).eps))
        candidates = torch.where(valid, candidates, center.flatten(1).unsqueeze(0))
        return candidates.reshape(num_samples, *original_shape)

    def _compute_energy_memory_search_loss(
        self,
        backend,
        memory_state,
        previous_memory,
        batch_ctx,
        labels,
    ):
        student_memory = memory_state["mem_batch"]
        with torch.no_grad():
            perturbed = self._sample_norm_preserving_memory_candidates(student_memory, previous_memory)
            candidate_memories = torch.cat([student_memory.detach().unsqueeze(0), perturbed], dim=0)
            candidate_losses = []
            with self._temporary_model_eval():
                for candidate_memory in candidate_memories:
                    candidate_state = dict(memory_state)
                    candidate_state["mem_batch"] = candidate_memory
                    read_batch = backend.build_read_inputs(candidate_state, batch_ctx)
                    model_kwargs = read_batch.get('model_kwargs', {})
                    with backend.activation_context(candidate_state):
                        with self._disable_write_lora():
                            read_out = self.model(
                                inputs_embeds=read_batch['inputs_embeds'],
                                return_dict=True,
                                **model_kwargs,
                            )
                    predictions = read_out.logits[
                        :,
                        read_batch['logits_start']:read_batch['logits_start'] + read_batch['pred_len'],
                        :,
                    ]
                    candidate_losses.append(
                        self._compute_per_example_read_target_loss(predictions, read_batch, labels)
                    )

            candidate_losses = torch.stack(candidate_losses)
            best_indices = candidate_losses.argmin(dim=0)
            batch_indices = torch.arange(student_memory.size(0), device=student_memory.device)
            teacher_memory = candidate_memories[best_indices, batch_indices]
            baseline_loss = candidate_losses[0]
            best_loss = candidate_losses[best_indices, batch_indices]
            target_gain = (baseline_loss - best_loss).clamp_min(0.0)
            relative_target_gain = target_gain / baseline_loss.clamp_min(
                torch.finfo(baseline_loss.dtype).eps
            )

        per_example_half_squared_norm = 0.5 * (
            student_memory.float() - teacher_memory.float()
        ).square().flatten(1).sum(dim=1)
        selected_distance = (student_memory.detach().float() - teacher_memory.float()).flatten(1).norm(dim=1)
        return {
            "loss": per_example_half_squared_norm.mean(),
            "per_example_loss": per_example_half_squared_norm,
            "relative_target_gain": relative_target_gain,
            "teacher_memory": teacher_memory,
            "improvement_rate": best_indices.ne(0).float().mean(),
            "target_gain": target_gain.mean(),
            "max_target_gain": target_gain.max(),
            "selected_distance": selected_distance.mean(),
        }

    @torch.no_grad()
    def _compute_energy_memory_search_gain_weights(self, relative_target_gains):
        gains = relative_target_gains.detach().float().clamp_min(0.0)
        if gains.ndim != 2 or gains.size(0) != self.K:
            raise ValueError(
                f"relative_target_gains must have shape [K, batch], got {tuple(gains.shape)} for K={self.K}"
            )
        positive = gains.gt(0.0)
        gain_sum = (gains * positive).sum(dim=1)
        positive_count = positive.sum(dim=1).to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gain_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(positive_count, op=dist.ReduceOp.SUM)

        batch_positive_mean = gain_sum / positive_count.clamp_min(1.0)
        decay = self.energy_memory_search_gain_ema_decay
        updated_ema = torch.where(
            self.energy_memory_search_gain_ema_initialized,
            decay * self.energy_memory_search_gain_ema + (1.0 - decay) * batch_positive_mean,
            batch_positive_mean,
        )
        has_positive = positive_count.gt(0.0)
        self.energy_memory_search_gain_ema.copy_(torch.where(
            has_positive,
            updated_ema,
            self.energy_memory_search_gain_ema,
        ))
        self.energy_memory_search_gain_ema_initialized.logical_or_(has_positive)

        # Let the normalizer rise immediately but decay smoothly. Otherwise a
        # stale, near-zero EMA can amplify a sudden positive gain by 1 / (1 - decay).
        normalizer = torch.maximum(
            self.energy_memory_search_gain_ema,
            batch_positive_mean,
        ).clamp_min(torch.finfo(torch.float32).eps)
        return gains / normalizer.unsqueeze(1)

    def _combine_depth_auxiliary_losses(
        self,
        inner_loss,
        rank_loss,
        trajectory_loss,
        anchor_loss,
        memory_alignment_loss,
        orthogonal_loss,
    ):
        weighted_inner_loss = self.inner_loss_weight * inner_loss if self.add_inner_loss_to_outer else 0.0
        weighted_orthogonal_loss = (
            self.orthogonal_loss_weight * orthogonal_loss
            if self.orthogonal_loss_weight > 0.0
            else 0.0
        )
        return (
            weighted_inner_loss
            + self.energy_rank_weight * rank_loss
            + self.energy_traj_weight * trajectory_loss
            + self.energy_anchor_weight * anchor_loss
            + self.memory_alignment_weight * memory_alignment_loss
            + weighted_orthogonal_loss
        )

    def _compute_memory_state_auxiliary_losses(
        self,
        backend,
        memory_state,
        memory_state_initial,
        batch_ctx,
        target_loss,
        initial_memory,
        memory_depth,
        energy_loss_history,
        energy_loss_after_write,
        *,
        memory_alignment_active,
        orthogonal_loss_active,
        energy_objective_active,
    ):
        """Evaluate auxiliary objectives at one WRITE trajectory memory state."""
        zero = target_loss.new_tensor(0.0)
        memory_alignment_loss = zero
        memory_alignment_cosine = None
        orthogonal_loss = zero
        orthogonal_residual_dot = None
        memory_gradient_active = memory_alignment_active or orthogonal_loss_active

        if memory_gradient_active:
            memory = memory_state["mem_batch"]
            outer_grad = torch.autograd.grad(
                target_loss,
                memory,
                create_graph=False,
                retain_graph=True,
            )[0].detach()
            if memory_alignment_active:
                write_direction = (initial_memory - memory).float().flatten(1)
                outer_grad_flat = outer_grad.float().flatten(1)
                memory_alignment_cosine = F.cosine_similarity(write_direction, outer_grad_flat, dim=1)
                memory_alignment_loss = 1 - memory_alignment_cosine.mean()
            if orthogonal_loss_active:
                inner_update = (initial_memory - memory).float()
                outer_grad_float = outer_grad.float()
                if self.orthogonal_loss_weight == 0.0:
                    inner_update = inner_update.detach()
                alpha = (
                    F.softplus(self.orthogonal_alpha_raw.float())
                    if self.orthogonal_alpha_raw is not None
                    else zero.new_tensor(self.lr)
                )
                residual = inner_update - alpha * outer_grad_float
                orthogonal_residual_dot = (residual * outer_grad_float).flatten(1).sum(dim=1)
                orthogonal_loss = orthogonal_residual_dot.square().mean()

        rank_loss = zero
        anchor_loss = zero
        rank_active = energy_objective_active and self.energy_rank_weight > 0.0
        anchor_active = energy_objective_active and self.energy_anchor_weight > 0.0
        landscape = None
        if rank_active or anchor_active:
            landscape = self._compute_energy_landscape_losses(
                backend,
                memory_state,
                memory_state_initial,
                batch_ctx,
                compute_rank=rank_active,
                compute_anchor=anchor_active,
            )
            if rank_active:
                rank_loss = landscape["rank_loss"]
            if anchor_active:
                anchor_loss = landscape["anchor_loss"]

        trajectory_loss = zero
        trajectory_active = (
            energy_objective_active
            and self.energy_traj_weight > 0.0
            and memory_depth > 0
            and energy_loss_after_write is not None
        )
        if trajectory_active:
            energy_states = energy_loss_history + [energy_loss_after_write]
            trajectory_terms = [
                F.relu(e_next - e_prev + self.energy_traj_margin).mean()
                for e_prev, e_next in zip(
                    energy_states[:memory_depth],
                    energy_states[1:memory_depth + 1],
                )
            ]
            trajectory_loss = torch.stack(trajectory_terms).mean()

        return {
            "memory_alignment_loss": memory_alignment_loss,
            "memory_alignment_cosine": memory_alignment_cosine,
            "orthogonal_loss": orthogonal_loss,
            "orthogonal_residual_dot": orthogonal_residual_dot,
            "rank_loss": rank_loss,
            "anchor_loss": anchor_loss,
            "trajectory_loss": trajectory_loss,
            "landscape": landscape,
        }

    def _compute_step_alignment(self, target_loss, memory_state, inner_update):
        outer_grad = torch.autograd.grad(
            target_loss,
            memory_state["mem_batch"],
            create_graph=False,
            retain_graph=True,
        )[0].detach()
        cosine = F.cosine_similarity(
            inner_update.float().flatten(1),
            outer_grad.float().flatten(1),
            dim=1,
        )
        if self.grad_align_norm == "norm":
            task_grad_norm = outer_grad.float().flatten(1).norm(dim=1)
            return (1 - task_grad_norm * cosine).mean(), cosine.mean()
        return 1 - cosine.mean(), cosine.mean()

    def forward(self, input_ids, labels=None, return_mem=False):
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']

        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)
        inner_loss = torch.tensor(0.0, device=device)

        backend = self.memory_backend_impl
        memory_state, memory_state_initial = backend.init_memory_state(B)
        # These are diagnostics whenever their prefix-memory gradients exist; their
        # weights control only whether they contribute to the outer objective.
        memory_alignment_active = labels is not None and self.memory_backend == "prefix" and self.K > 0
        step_alignment_active = memory_alignment_active
        orthogonal_loss_active = memory_alignment_active
        ivan_loss_active = labels is not None and self.ivan_loss_weight > 0.0 and self.K > 0
        outer_memory_gradient_active = memory_alignment_active or orthogonal_loss_active or ivan_loss_active
        initial_memory = memory_state["mem_batch"] if memory_alignment_active else None
        intermediate_read_active = labels is not None and self.K > 1
        intermediate_read_objective_active = (
            intermediate_read_active and self.intermediate_read_weight > 0.0
        )
        intermediate_memory_states = []
        inner_memory_updates = []
        inner_grad_sums = None
        batch_ctx = backend.prepare_batch(context_input_ids, query_input_ids, pad_id)
        opt_state = {}
        inner_loss_history = []
        energy_loss_history = []
        inline_memory_search_results = []
        has_write_context = bool(context_input_ids.ne(pad_id).any().item())
        memory_search_active = (
            self.training
            and labels is not None
            and has_write_context
            and self.energy_memory_search_weight > 0.0
        )
        inline_memory_search_active = (
            memory_search_active and self.energy_memory_search_use_best_for_next_step
        )

        inner_loop_stats = {
            'inner_grad_norm_mean': torch.tensor(0.0, device=device),
            'inner_grad_norm_max': torch.tensor(-1.0, device=device),
            'inner_grad_norm_min': torch.tensor(1e06, device=device),
        }

        if self.K and has_write_context:
            with torch.enable_grad():
                for k in range(self.K):
                    gradient_memory_state = memory_state
                    if self.training and self.memory_noise_sigma > 0.0:
                        gradient_memory_state = dict(memory_state)
                        initial_rms = memory_state_initial["mem_batch"].float().square().mean().sqrt().detach()
                        noise = torch.randn_like(memory_state["mem_batch"]) * (
                            self.memory_noise_sigma * initial_rms
                        )
                        gradient_memory_state["mem_batch"] = memory_state["mem_batch"] + noise
                    write_batch = backend.build_write_inputs(gradient_memory_state, batch_ctx)
                    rec_loss = None
                    energy_loss = None
                    if self.write_objective == "energy":
                        with backend.activation_context(gradient_memory_state):
                            outs, rec_loss, energy_loss = self._run_energy_reconstruction_write_forward(
                                write_batch
                            )
                        inner_loss_per_sample = energy_loss
                        del outs
                    elif self.write_objective == "energy_with_reconstruction":
                        with backend.activation_context(gradient_memory_state):
                            outs, rec_loss, energy_loss = self._run_energy_reconstruction_write_forward(write_batch)
                        inner_loss_per_sample = (
                            self.write_reconstruction_weight * rec_loss
                            + self.write_energy_weight * energy_loss
                        )
                        del outs
                    else:
                        with backend.activation_context(gradient_memory_state):
                            outs, rec_loss = self._run_reconstruction_write_forward(write_batch)
                        inner_loss_per_sample = rec_loss
                        del outs

                    inner_loss_history.append(inner_loss_per_sample)
                    if energy_loss is not None:
                        energy_loss_history.append(energy_loss)
                        inner_loop_stats['inner_energy_loss'] = energy_loss.detach().mean()
                    if rec_loss is not None:
                        inner_loop_stats['inner_reconstruction_loss'] = rec_loss.detach().mean()
                    inner_loss = inner_loss_per_sample.sum()

                    is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
                    create_graph = is_second_order_step
                    retain_graph = create_graph or (
                        (self.add_inner_loss_to_outer and k == self.K - 1)
                        or intermediate_read_objective_active
                    )

                    clean_inner_params = backend.inner_params(memory_state)
                    gradient_inner_params = backend.inner_params(gradient_memory_state)
                    grads = torch.autograd.grad(inner_loss, gradient_inner_params,
                                                create_graph=create_graph, retain_graph=retain_graph)
                    if ivan_loss_active:
                        if inner_grad_sums is None:
                            inner_grad_sums = list(grads)
                        else:
                            inner_grad_sums = [grad_sum + grad for grad_sum, grad in zip(inner_grad_sums, grads)]

                    g_sq = torch.zeros(B, device=device)
                    for g in grads:
                        g_sq = g_sq + g.reshape(B, -1).pow(2).sum(dim=1)
                    g_norm = g_sq.sqrt().detach()
                    inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
                    inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
                    inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())

                    new_params = []
                    for p, g, i in zip(clean_inner_params, grads, range(len(clean_inner_params))):
                        if self.use_adam:
                            p_new = self._adam_step(p, g, opt_state.setdefault(str(i), {}), k + 1, self.lr)
                        else:
                            p_new = self._sgd_step(p, g,
                                                   clip_value=self.inner_clip_value,
                                                   clip_norm=self.inner_clip_norm)
                        new_params.append(p_new)
                    if memory_alignment_active:
                        inner_memory_updates.append(clean_inner_params[0] - new_params[0])
                    backend.assign_inner_params(memory_state, new_params)
                    backend.maybe_detach_after_step(memory_state)
                    if inline_memory_search_active:
                        search_result = self._compute_energy_memory_search_loss(
                            backend,
                            memory_state,
                            clean_inner_params[0],
                            batch_ctx,
                            labels,
                        )
                        inline_memory_search_results.append(search_result)
                        student_memory = memory_state["mem_batch"]
                        teacher_memory = search_result["teacher_memory"]
                        memory_state["mem_batch"] = (
                            student_memory + (teacher_memory - student_memory.detach())
                        )
                    if intermediate_read_active and k < self.K - 1:
                        intermediate_memory_states.append(backend.snapshot_memory_state(memory_state))

        if self.K:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / self.K
            inner_loop_stats['inner_loss'] = inner_loss.detach() / B

        inner_loss_after_write = None
        reconstruction_loss_after_write = None
        energy_loss_after_write = None
        with torch.enable_grad():
            final_write_batch = backend.build_write_inputs(memory_state, batch_ctx)
            with backend.activation_context(memory_state):
                if self.write_objective in ("energy", "energy_with_reconstruction"):
                    final_outs, reconstruction_loss_after_write, energy_loss_after_write = (
                        self._run_energy_reconstruction_write_forward(final_write_batch)
                    )
                else:
                    final_outs, reconstruction_loss_after_write = self._run_reconstruction_write_forward(
                        final_write_batch
                    )

        if self.write_objective == "energy":
            inner_loss_after_write = energy_loss_after_write
        elif self.write_objective == "energy_with_reconstruction":
            inner_loss_after_write = (
                self.write_reconstruction_weight * reconstruction_loss_after_write
                + self.write_energy_weight * energy_loss_after_write
            )
        else:
            inner_loss_after_write = reconstruction_loss_after_write

        inner_loop_stats['inner_reconstruction_loss_after_write'] = (
            reconstruction_loss_after_write.detach().mean()
        )
        if energy_loss_after_write is not None:
            inner_loop_stats['inner_energy_loss_after_write'] = energy_loss_after_write.detach().mean()
        del final_outs
        if inner_loss_after_write is not None:
            inner_loop_stats['inner_loss_after_write'] = inner_loss_after_write.detach().mean()
        if inner_loss_after_write is not None and len(inner_loss_history) > 0:
            inner_loss_initial = inner_loss_history[0]
            inner_loss_write_delta = inner_loss_after_write - inner_loss_initial
            inner_loop_stats['inner_loss_initial'] = inner_loss_initial.detach().mean()
            inner_loop_stats['inner_loss_write_delta'] = inner_loss_write_delta.detach().mean()
        inner_loop_stats['write_reconstruction_weight'] = torch.tensor(
            self.write_reconstruction_weight, device=device
        )
        inner_loop_stats['write_energy_weight'] = torch.tensor(self.write_energy_weight, device=device)

        mem_norm, delta_mem_norm = backend.compute_memory_stats(memory_state, memory_state_initial)
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()
        inner_loop_stats['mem_norm_min'] = mem_norm.min()
        inner_loop_stats['delta_mem_norm_mean'] = delta_mem_norm.mean()
        inner_loop_stats['delta_mem_norm_max'] = delta_mem_norm.max()
        inner_loop_stats['delta_mem_norm_min'] = delta_mem_norm.min()

        read_grad_context = torch.enable_grad() if outer_memory_gradient_active else nullcontext()
        with read_grad_context:
            read_batch = backend.build_read_inputs(memory_state, batch_ctx)
            read_model_kwargs = read_batch.get('model_kwargs', {})
            log_mem_attn_read = (
                self.attn_implementation == "eager" and self.memory_backend in ("prefix", "kv_cache")
            )
            if log_mem_attn_read:
                read_model_kwargs = dict(read_model_kwargs)
                read_model_kwargs["output_attentions"] = True
            with backend.activation_context(memory_state):
                with self._disable_write_lora():
                    read_out = self.model(inputs_embeds=read_batch['inputs_embeds'],
                                          return_dict=True,
                                          **read_model_kwargs)
                    logits_q = read_out.logits

            if log_mem_attn_read and read_out.attentions is not None:
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
                            "GradMem: Invalid memory attention span on read: "
                            f"backend={self.memory_backend}, mem_start={mem_start}, mem_end={mem_end}, k_len={k_len}"
                        )
                    layer_ratios.append(att[..., mem_start:mem_end].sum(dim=-1).mean())
                if layer_ratios:
                    inner_loop_stats['mem_attn_read'] = torch.stack(layer_ratios).mean().detach()

            # LoRA/KV-cache memory backends need one seed token to start autoregressive prediction;
            # they cannot predict the very first token from memory alone when query is empty.
            logits_q = logits_q[
                :, read_batch['logits_start']:read_batch['logits_start'] + read_batch['pred_len'], :
            ]

        output = {'predictions': logits_q, 'inner_loop_stats': inner_loop_stats}
        if return_mem:
            backend.attach_return_memory(output, memory_state)

        if labels is None:
            return output

        target_grad_context = torch.enable_grad() if outer_memory_gradient_active else nullcontext()
        with target_grad_context:
            target_loss = self._compute_read_target_loss(output['predictions'], read_batch, labels)

        zero = target_loss.new_tensor(0.0)
        memory_search_loss = zero
        memory_search_unweighted_loss = zero
        memory_search_improvement_rate = zero
        memory_search_target_gain = zero
        memory_search_max_target_gain = zero
        memory_search_relative_target_gain = zero
        memory_search_max_relative_target_gain = zero
        memory_search_included_rate = zero
        memory_search_selected_distance = zero
        if memory_search_active:
            if inline_memory_search_active:
                search_results = inline_memory_search_results
            else:
                search_states = intermediate_memory_states + [memory_state]
                previous_memory = memory_state_initial["mem_batch"]
                search_results = []
                for search_state in search_states:
                    search_results.append(
                        self._compute_energy_memory_search_loss(
                            backend,
                            search_state,
                            previous_memory,
                            batch_ctx,
                            labels,
                        )
                    )
                    previous_memory = search_state["mem_batch"].detach()
            per_example_search_losses = torch.stack(
                [result["per_example_loss"] for result in search_results]
            )
            relative_target_gains = torch.stack(
                [result["relative_target_gain"] for result in search_results]
            )
            memory_search_unweighted_loss = per_example_search_losses.mean()
            memory_search_relative_target_gain = relative_target_gains.mean()
            memory_search_max_relative_target_gain = relative_target_gains.max()
            included = relative_target_gains.ge(
                self.energy_memory_search_min_relative_target_gain
            )
            included_float = included.to(dtype=per_example_search_losses.dtype)
            included_count = included_float.sum()
            memory_search_included_rate = included_float.mean()
            if self.energy_memory_search_use_gain_weighting:
                thresholded_gains = relative_target_gains * included
                gain_weights = self._compute_energy_memory_search_gain_weights(thresholded_gains)
                weighted_losses = gain_weights * per_example_search_losses
            else:
                weighted_losses = included_float * per_example_search_losses
            memory_search_loss = weighted_losses.sum() / included_count.clamp_min(1.0)
            memory_search_improvement_rate = torch.stack(
                [result["improvement_rate"] for result in search_results]
            ).mean()
            memory_search_target_gain = torch.stack(
                [result["target_gain"] for result in search_results]
            ).mean()
            memory_search_max_target_gain = torch.stack(
                [result["max_target_gain"] for result in search_results]
            ).max()
            memory_search_selected_distance = torch.stack(
                [result["selected_distance"] for result in search_results]
            ).mean()

        ivan_loss = zero
        ivan_residual_dot = zero
        if ivan_loss_active and inner_grad_sums is not None:
            final_inner_params = backend.inner_params(memory_state)
            outer_grads = torch.autograd.grad(
                target_loss,
                final_inner_params,
                create_graph=True,
                retain_graph=True,
            )
            ivan_residual_dot_per_sample = torch.zeros(B, device=device, dtype=torch.float32)
            for inner_grad_sum, outer_grad in zip(inner_grad_sums, outer_grads):
                outer_grad_float = outer_grad.float()
                ivan_residual_dot_per_sample = ivan_residual_dot_per_sample + (
                    (inner_grad_sum.float() - outer_grad_float) * outer_grad_float
                ).reshape(B, -1).sum(dim=1)
            ivan_residual_dot = ivan_residual_dot_per_sample.mean()
            ivan_loss = ivan_residual_dot_per_sample.square().mean()
        intermediate_read_loss = zero
        step_alignment_losses = []
        step_alignment_cosines = []
        if intermediate_memory_states:
            intermediate_depth_losses = []
            energy_objective_active = self.write_objective in ("energy", "energy_with_reconstruction")
            for depth, intermediate_memory_state in enumerate(intermediate_memory_states, start=1):
                intermediate_grad_context = (
                    torch.enable_grad()
                    if step_alignment_active or intermediate_read_objective_active
                    else torch.no_grad()
                )
                with intermediate_grad_context:
                    intermediate_read_batch = backend.build_read_inputs(intermediate_memory_state, batch_ctx)
                    intermediate_model_kwargs = intermediate_read_batch.get('model_kwargs', {})
                    with backend.activation_context(intermediate_memory_state):
                        with self._disable_write_lora():
                            intermediate_read_out = self.model(
                                inputs_embeds=intermediate_read_batch['inputs_embeds'],
                                return_dict=True,
                                **intermediate_model_kwargs,
                            )
                    intermediate_predictions = intermediate_read_out.logits[
                        :,
                        intermediate_read_batch['logits_start']:
                        intermediate_read_batch['logits_start'] + intermediate_read_batch['pred_len'],
                        :,
                    ]
                    depth_target_loss = self._compute_read_target_loss(
                        intermediate_predictions,
                        intermediate_read_batch,
                        labels,
                    )

                if step_alignment_active:
                    depth_step_alignment_loss, depth_step_alignment_cosine = self._compute_step_alignment(
                        depth_target_loss,
                        intermediate_memory_state,
                        inner_memory_updates[depth - 1],
                    )
                    step_alignment_losses.append(depth_step_alignment_loss)
                    step_alignment_cosines.append(depth_step_alignment_cosine)

                if intermediate_read_objective_active:
                    depth_inner_loss = inner_loss_history[depth - 1].sum() / B
                    depth_auxiliary_losses = self._compute_memory_state_auxiliary_losses(
                        backend,
                        intermediate_memory_state,
                        memory_state_initial,
                        batch_ctx,
                        depth_target_loss,
                        initial_memory,
                        depth,
                        energy_loss_history,
                        energy_loss_after_write,
                        memory_alignment_active=memory_alignment_active,
                        orthogonal_loss_active=orthogonal_loss_active,
                        energy_objective_active=energy_objective_active,
                    )
                    depth_aux_loss = self._combine_depth_auxiliary_losses(
                        inner_loss=depth_inner_loss,
                        rank_loss=depth_auxiliary_losses["rank_loss"],
                        trajectory_loss=depth_auxiliary_losses["trajectory_loss"],
                        anchor_loss=depth_auxiliary_losses["anchor_loss"],
                        memory_alignment_loss=depth_auxiliary_losses["memory_alignment_loss"],
                        orthogonal_loss=depth_auxiliary_losses["orthogonal_loss"],
                    )
                    intermediate_depth_losses.append(depth_target_loss + depth_aux_loss)
                else:
                    intermediate_depth_losses.append(depth_target_loss)

            intermediate_depth_losses = torch.stack(intermediate_depth_losses)
            intermediate_weights = self._intermediate_read_weights(
                self.K,
                device=intermediate_depth_losses.device,
                dtype=intermediate_depth_losses.dtype,
            )
            intermediate_read_loss = (intermediate_weights * intermediate_depth_losses).sum()
        inner_loss_for_outer = inner_loss / B
        loss_stats = {}
        landscape_stats = {}
        energy_objective_active = self.write_objective in ("energy", "energy_with_reconstruction")
        lipschitz_active = self.lipschitz_weight > 0.0
        lipschitz_diagnostic_active = self.memory_backend == "prefix"
        rank_active = energy_objective_active and self.energy_rank_weight > 0.0
        anchor_active = energy_objective_active and self.energy_anchor_weight > 0.0
        trajectory_active = (
            energy_objective_active
            and self.energy_traj_weight > 0.0
            and energy_loss_after_write is not None
        )
        final_auxiliary_losses = self._compute_memory_state_auxiliary_losses(
            backend,
            memory_state,
            memory_state_initial,
            batch_ctx,
            target_loss,
            initial_memory,
            self.K,
            energy_loss_history,
            energy_loss_after_write,
            memory_alignment_active=memory_alignment_active,
            orthogonal_loss_active=orthogonal_loss_active,
            energy_objective_active=energy_objective_active,
        )
        lipschitz_loss = zero
        lipschitz_gradient_norm = zero.expand(B)
        if lipschitz_diagnostic_active:
            with torch.enable_grad():
                lipschitz_loss, lipschitz_gradient_norm = self._compute_lipschitz_constraint_loss(
                    backend,
                    memory_state,
                    batch_ctx,
                    create_graph=lipschitz_active,
                )
        if step_alignment_active and inner_memory_updates:
            final_step_alignment_loss, final_step_alignment_cosine = self._compute_step_alignment(
                target_loss,
                memory_state,
                inner_memory_updates[-1],
            )
            step_alignment_losses.append(final_step_alignment_loss)
            step_alignment_cosines.append(final_step_alignment_cosine)

            if self.align_last_step:
                with torch.enable_grad():
                    reconstruction_grad_after_write = torch.autograd.grad(
                        reconstruction_loss_after_write.sum(),
                        memory_state["mem_batch"],
                        create_graph=self.step_alignment_weight > 0.0,
                        retain_graph=True,
                    )[0]
                reconstruction_alignment_loss_after_write, reconstruction_alignment_cosine_after_write = (
                    self._compute_step_alignment(
                        target_loss,
                        memory_state,
                        reconstruction_grad_after_write,
                    )
                )
                step_alignment_losses.append(reconstruction_alignment_loss_after_write)
                step_alignment_cosines.append(reconstruction_alignment_cosine_after_write)
            step_alignment_loss = torch.stack(step_alignment_losses).mean()
            step_alignment_cosine = torch.stack(step_alignment_cosines).mean()
        else:
            step_alignment_loss = zero
            step_alignment_cosine = zero
        memory_alignment_loss = final_auxiliary_losses["memory_alignment_loss"]
        memory_alignment_cosine = final_auxiliary_losses["memory_alignment_cosine"]
        orthogonal_loss = final_auxiliary_losses["orthogonal_loss"]
        orthogonal_residual_dot = final_auxiliary_losses["orthogonal_residual_dot"]
        rank_loss = final_auxiliary_losses["rank_loss"]
        anchor_loss = final_auxiliary_losses["anchor_loss"]
        traj_loss = final_auxiliary_losses["trajectory_loss"]
        landscape = final_auxiliary_losses["landscape"]
        rank_deranged_loss = zero
        rank_interpolated_loss = zero
        rank_random_loss = zero
        if memory_alignment_cosine is None:
            memory_alignment_cosine = zero
        if orthogonal_residual_dot is None:
            orthogonal_residual_dot = zero
        orthogonal_alpha = (
            F.softplus(self.orthogonal_alpha_raw.float())
            if self.orthogonal_alpha_raw is not None
            else zero.new_tensor(self.lr)
        )

        if landscape is not None:
            energy_positive = landscape["energy_positive"]
            landscape_stats["energy_positive_mean"] = energy_positive.mean()

        if rank_active:
            loss_names = {
                "deranged": "energy_rank_deranged_loss",
                "interpolated": "energy_rank_interpolated_loss",
                "random": "energy_rank_random_loss",
            }
            component_losses = {}
            for name, energy_negative in landscape["negative_energies"].items():
                component_loss = landscape["negative_losses"][name]
                component_losses[loss_names[name]] = component_loss
                landscape_stats[f"energy_negative_{name}_mean"] = energy_negative.mean()
                landscape_stats[f"energy_margin_{name}_mean"] = (energy_negative - energy_positive).mean()
                landscape_stats[f"energy_margin_violation_{name}"] = (
                    energy_positive + self.energy_margin > energy_negative
                ).to(dtype=energy_positive.dtype).mean()

            rank_deranged_loss = component_losses.get("energy_rank_deranged_loss", zero)
            rank_interpolated_loss = component_losses.get("energy_rank_interpolated_loss", zero)
            rank_random_loss = component_losses.get("energy_rank_random_loss", zero)

            loss_stats.update({
                "energy_rank_loss": rank_loss,
                "energy_rank_deranged_loss": rank_deranged_loss,
                "energy_rank_interpolated_loss": rank_interpolated_loss,
                "energy_rank_random_loss": rank_random_loss,
            })

        if anchor_active:
            loss_stats["energy_anchor_loss"] = anchor_loss

        if trajectory_active:
            loss_stats["energy_traj_loss"] = traj_loss

        weighted_inner_loss = self.inner_loss_weight * inner_loss_for_outer if self.add_inner_loss_to_outer else zero
        weighted_rank_loss = self.energy_rank_weight * rank_loss if rank_active else zero
        weighted_traj_loss = self.energy_traj_weight * traj_loss if trajectory_active else zero
        weighted_anchor_loss = self.energy_anchor_weight * anchor_loss if anchor_active else zero
        weighted_lipschitz_loss = self.lipschitz_weight * lipschitz_loss if lipschitz_active else zero
        weighted_memory_search_loss = (
            self.energy_memory_search_weight * memory_search_loss if memory_search_active else zero
        )
        energy_aux_loss = (
            weighted_rank_loss
            + weighted_traj_loss
            + weighted_anchor_loss
            + weighted_memory_search_loss
        )
        weighted_memory_alignment_loss = self.memory_alignment_weight * memory_alignment_loss
        weighted_step_alignment_loss = self.step_alignment_weight * step_alignment_loss
        weighted_intermediate_read_loss = self.intermediate_read_weight * intermediate_read_loss
        weighted_orthogonal_loss = self.orthogonal_loss_weight * orthogonal_loss
        weighted_ivan_loss = self.ivan_loss_weight * ivan_loss
        auxiliary_loss = self._combine_depth_auxiliary_losses(
            inner_loss=inner_loss_for_outer,
            rank_loss=rank_loss,
            trajectory_loss=traj_loss,
            anchor_loss=anchor_loss,
            memory_alignment_loss=memory_alignment_loss,
            orthogonal_loss=orthogonal_loss,
        )
        outer_loss = (
            target_loss
            + auxiliary_loss
            + weighted_memory_search_loss
            + weighted_lipschitz_loss
            + weighted_ivan_loss
            + weighted_step_alignment_loss
            + weighted_intermediate_read_loss
        )
        loss_stats.update({
            "outer_loss": outer_loss,
            "target_loss": target_loss,
            "memory_alignment_loss": memory_alignment_loss,
            "memory_alignment_cosine": memory_alignment_cosine.mean(),
            "step_alignment_loss": step_alignment_loss,
            "step_alignment_cosine": step_alignment_cosine,
            "orthogonal_loss": orthogonal_loss,
            "orthogonal_residual_dot": orthogonal_residual_dot.mean(),
            "orthogonal_alpha": orthogonal_alpha,
            "ivan_loss": ivan_loss,
            "ivan_residual_dot": ivan_residual_dot,
            "intermediate_read_loss": intermediate_read_loss,
            "lipschitz_grad_norm_mean": lipschitz_gradient_norm.mean(),
            "lipschitz_grad_norm_max": lipschitz_gradient_norm.max(),
        })
        if lipschitz_active:
            loss_stats["lipschitz_loss"] = lipschitz_loss
        if memory_search_active:
            loss_stats.update({
                "energy_memory_search_loss": memory_search_loss,
                "energy_memory_search_improvement_rate": memory_search_improvement_rate,
                "energy_memory_search_target_gain": memory_search_target_gain,
                "energy_memory_search_max_target_gain": memory_search_max_target_gain,
                "energy_memory_search_mean_relative_target_gain": memory_search_relative_target_gain,
                "energy_memory_search_max_relative_target_gain": memory_search_max_relative_target_gain,
                "energy_memory_search_included_rate": memory_search_included_rate,
                "energy_memory_search_selected_distance": memory_search_selected_distance,
            })
        if rank_active or trajectory_active or anchor_active or memory_search_active:
            loss_stats["energy_aux_loss"] = energy_aux_loss
        for name, value in {**loss_stats, **landscape_stats}.items():
            output["inner_loop_stats"][name] = value.detach()
        output['loss'] = outer_loss
        return output
