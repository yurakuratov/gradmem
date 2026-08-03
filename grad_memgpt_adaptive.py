"""GradMemGPT — adaptive (segmented, gated-recurrence) fork.

Fork of ``grad_memgpt_old.py`` (clean gradient inner-loop base) with two additions:

1. **Segment division** (ported from ``grad_memgpt.py``): the WRITE context is
   split into ``n_segments`` (near-)equal chunks and each segment is compressed
   into the same memory by running ``K`` inner gradient steps on it.

   CRITICAL difference from ``grad_memgpt.py``: memory is *not* reset between
   segments. The first segment starts from ``m_0``; every later segment
   continues from the previous ``m_t``. The inner loop is therefore a genuine
   RNN unrolled over ``n_segments * K`` steps, with the gradient-descent step as
   the input signal ``x_t = -alpha * grad`` and ``m`` as the hidden state.
   This is exactly the regime where a gated / selective recurrence pays off:
   information written in segment 1 must survive segments 2, 3, ... without
   being overwritten.

2. **Adaptive (less-rewriting) memory updates.** The plain SGD rule is a
   degenerate running-sum RNN::

       m_t = m_{t-1} + x_t                   # x_t = -alpha * grad

   which has no retention mechanism at all — a new gradient is added with full
   weight, so it can stomp on the directions earlier facts relied on.
   Catastrophic forgetting here is structural. This file generalises the update
   to a gated recurrence::

       m_t = r_t * m_{t-1} + w_t * x_t

   with three modes selected by ``memory_update_rule``:
     * "sgd"     : r_t = 1, w_t = 1  (exact old pure-sum; the baseline to beat).
     * "convex"  : one gate a_t in (0,1); r_t = 1-a_t, w_t = a_t. Minimal
                   convex-interpolation baseline (Family 3). Inits a_t ~= 0.95
                   so it starts as ~SGD and learns to interpolate.
     * "mamba"   : r_t = exp(-Delta_t) (Mamba-style exponential retention, in
                   (0,1]), Delta_t = softplus(lin_r(phi)); w_t = sigmoid(lin_w(phi)).
                   Input- AND state-dependent per-dim retention (Family 2).
                   ``gate_retention="sigmoid"`` switches retention to a sigmoid
                   (GRU / Family 1). Inits retention ~= 0.993, write ~= 0.95 so
                   the operator starts as ~SGD, then opens the gates to write
                   selectively where new info should land and protect old info
                   elsewhere. Exponential (vs sigmoid) retention stays
                   well-conditioned over long unrollings because retention
                   multiplies across steps: prod_t r_t = exp(-sum_t Delta_t).

   The gate heads are tiny ``nn.Linear``s over per-token features phi(m, g);
   they are meta-learned by the *existing* second-order ``autograd.grad`` path
   (create_graph=True on the last K steps), exactly like ``mem_proj``. No new
   optimisation machinery is introduced.

Only the gradient inner-loop path is kept (no Hopfield / energy / learned-update
/ gated-delta / read-lora / recon-pretrain branches) — this is intentionally the
clean fork for studying adaptive memory updates in isolation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from contextlib import contextmanager
import attn_double_bwd  # noqa: F401  # side-effect: registers attention kernels

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


class GradMemGPTConfig(PretrainedConfig):
    """
    Configuration class for GradMemGPT (adaptive fork).
    """
    model_type = "grad_memgpt"

    def __init__(self,
                 pretrained_model=None,
                 base_config=None,
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
                 freeze_backbone=False,
                 use_gradient_checkpointing=False,
                 attn_implementation="eager",
                 add_inner_loss_to_outer=False,
                 inner_loss_weight=None,
                 # ---- segmentation (Setup: cross-segment memory carry) ---- #
                 n_segments=1,
                 segment_size=None,
                 seg_bptt=None,
                 # ---- adaptive memory update ---- #
                 memory_update_rule="sgd",
                 gate_features="grad_state",
                 gate_granularity="per_dim",
                 gate_retention="exp",
                 convex_bias_init=3.0,
                 mamba_retain_bias_init=-5.0,
                 mamba_write_bias_init=3.0,
                 **kwargs):
        """
        Args (new vs grad_memgpt_old.py):
            n_segments: int, number of equal segments to split the WRITE context
                into (1 = whole context, = old behaviour). Mutually exclusive
                with segment_size.
            segment_size: int|None, fixed tokens-per-segment; n_segments is then
                ceil(S / segment_size). Mutually exclusive with n_segments.
            seg_bptt: int|None, truncated-BPTT window over the segment
                recurrence: keep the autograd graph only for the last seg_bptt
                segments, detach older segment boundaries. None = keep the full
                graph (correct but memory-heavy for long unrollings under
                grad_mode "first"/"second"). No effect under grad_mode "none".
            memory_update_rule: str, inner-loop memory update operator.
                "sgd"     : m = m + (-lr*g)                 (old pure-sum, no new params)
                "convex"  : m = (1-a)*m + a*(-lr*g), a=sigmoid(lin(phi))   (Family 3)
                "mamba"   : m = exp(-D)*m + w*(-lr*g), D=softplus(lin_r(phi)),
                            w=sigmoid(lin_w(phi)); retention set by gate_retention (Family 2/1)
            gate_features: str, input phi to the gate heads.
                "grad"      : phi = g            (in_dim = d)
                "state"     : phi = m            (in_dim = d)
                "grad_state": phi = [g, m]       (in_dim = 2d)  [default, true selectivity]
            gate_granularity: str, gate output width.
                "per_dim"  : gate shape [B, M, d]  (richest; default)
                "per_token": gate shape [B, M, 1]  (broadcast over d; cheapest)
            gate_retention: str, retention form for memory_update_rule="mamba".
                "exp"    : r = exp(-Delta)  (Mamba exponential decay; default)
                "sigmoid": r = sigmoid(lin) (GRU-style gate; Family 1)
                NOTE on init: mamba_retain_bias_init is calibrated for the "exp"
                path (softplus(-5)~=0.007 -> r=exp(-0.007)~=0.993, i.e. starts as
                ~SGD). For "sigmoid" the same bias gives r=sigmoid(-5)~=0.007
                (near-total forgetting) — the signs are opposite. If you switch
                to gate_retention="sigmoid", set mamba_retain_bias_init to a
                POSITIVE value (e.g. +5.0 -> r~=0.993) to preserve the SGD-start.
            convex_bias_init: float, bias init for the convex gate (sigmoid(b) ~= 0.95
                at 3.0 -> starts near SGD).
            mamba_retain_bias_init: float, bias init for Delta head (softplus(b) ~= 0.007
                at -5.0 -> retention ~= 0.993 -> starts near SGD).
            mamba_write_bias_init: float, bias init for write gate (sigmoid(b) ~= 0.95
                at 3.0 -> starts near SGD).

        (All other args are unchanged from grad_memgpt_old.py; see its docstring.)
        """
        super().__init__(**kwargs)

        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
            self.base_config = None
        else:
            self.pretrained_model = None
            self.base_config = base_config

        # GradMemGPT specific parameters (unchanged from old)
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
        self.freeze_backbone = freeze_backbone
        self.last_K_second_order = K if last_K_second_order is None else last_K_second_order
        if grad_mode != "second":
            self.last_K_second_order = 0
        self.last_K_second_order = max(0, min(self.last_K_second_order, K))
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_implementation = attn_implementation
        self.add_inner_loss_to_outer = add_inner_loss_to_outer
        self.inner_loss_weight = inner_loss_weight

        # segmentation
        self.n_segments = n_segments
        self.segment_size = segment_size
        self.seg_bptt = seg_bptt

        # adaptive update
        self.memory_update_rule = memory_update_rule
        self.gate_features = gate_features
        self.gate_granularity = gate_granularity
        self.gate_retention = gate_retention
        self.convex_bias_init = convex_bias_init
        self.mamba_retain_bias_init = mamba_retain_bias_init
        self.mamba_write_bias_init = mamba_write_bias_init

        # Validate mem_proj_mode settings
        assert mem_proj_mode in ["none", "proj", "per_sample"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), \
            "use_mem_proj must be True if mem_proj_mode is set"

        # Validate segmentation settings
        assert n_segments >= 1, f"n_segments must be >= 1, got {n_segments}"
        if segment_size is not None:
            assert n_segments == 1, \
                f"segment_size and n_segments are mutually exclusive, got " \
                f"segment_size={segment_size} and n_segments={n_segments}"
            assert segment_size >= 1, f"segment_size must be >= 1, got {segment_size}"
        if seg_bptt is not None:
            assert seg_bptt >= 1, f"seg_bptt must be >= 1 or None, got {seg_bptt}"

        # Validate adaptive-update settings
        assert memory_update_rule in ("sgd", "convex", "mamba"), \
            f"memory_update_rule must be 'sgd', 'convex', or 'mamba', got '{memory_update_rule}'"
        assert gate_features in ("grad", "state", "grad_state"), \
            f"gate_features must be 'grad', 'state', or 'grad_state', got '{gate_features}'"
        assert gate_granularity in ("per_dim", "per_token"), \
            f"gate_granularity must be 'per_dim' or 'per_token', got '{gate_granularity}'"
        assert gate_retention in ("exp", "sigmoid"), \
            f"gate_retention must be 'exp' or 'sigmoid', got '{gate_retention}'"


class GradMemGPT(PreTrainedModel):
    """
    Transformer-decoder backbone + writable prefix memory (n_mem_tokens x d),
    segmented WRITE context with cross-segment memory carry, and an optional
    gated (Mamba / convex) memory-update recurrence. See module docstring.
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
        per_sample: per-sample fast weights W_i,b_i updated in the inner loop;
            their initial values (self.mem_proj.*) are meta-learned by the outer loop

        Segmentation: the WRITE context is split into n_segments chunks. The
        first segment starts from m_0; later segments CONTINUE from the previous
        m_t (no reset), so the inner loop is an RNN unrolled over n_segments*K.

        inner loop (per segment): [write_st][mem][write_end][segment]
        outer loop:               [read_st][mem][read_end][query][target]
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

        # store GradMemGPT parameters (unchanged from old)
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
        self.use_write_head = config.use_write_head
        self.add_inner_loss_to_outer = config.add_inner_loss_to_outer
        self.inner_loss_weight = config.inner_loss_weight
        if self.add_inner_loss_to_outer:
            if self.inner_loss_weight is None:
                self.inner_loss_weight = 1.0
        else:
            self.inner_loss_weight = 0.0

        # segmentation config
        self.n_segments = config.n_segments
        self.segment_size = config.segment_size
        self.seg_bptt = config.seg_bptt

        # adaptive-update config
        self.memory_update_rule = config.memory_update_rule
        self.gate_features = config.gate_features
        self.gate_granularity = config.gate_granularity
        self.gate_retention = config.gate_retention

        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
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

        # ---- adaptive-update gate heads (built only when rule != "sgd") ---- #
        # All gates are differentiable nn.Linear over per-token features phi;
        # meta-learned by the existing second-order path. Initialised so the
        # operator starts as ~SGD (retention ~= 1, write ~= 1) and learns to
        # deviate. See module docstring for the exact recurrence per mode.
        if self.memory_update_rule != "sgd":
            gate_in_dim = {"grad": n_embd, "state": n_embd, "grad_state": 2 * n_embd}[self.gate_features]
            gate_out_dim = n_embd if self.gate_granularity == "per_dim" else 1

            if self.memory_update_rule == "convex":
                # one gate a in (0,1): m = (1-a)*m + a*(-lr*g)
                self.convex_head = nn.Linear(gate_in_dim, gate_out_dim)
                nn.init.zeros_(self.convex_head.weight)
                nn.init.constant_(self.convex_head.bias, config.convex_bias_init)
            else:  # "mamba"
                # retention gate r in (0,1] and write gate w in (0,1), independent
                # (mirrors gated_delta's separate alpha/beta). retention form set
                # by gate_retention: exp(-softplus(.)) (Mamba) or sigmoid(.) (GRU).
                self.retain_head = nn.Linear(gate_in_dim, gate_out_dim)
                self.write_head_gate = nn.Linear(gate_in_dim, gate_out_dim)
                nn.init.zeros_(self.retain_head.weight)
                nn.init.constant_(self.retain_head.bias, config.mamba_retain_bias_init)
                nn.init.zeros_(self.write_head_gate.weight)
                nn.init.constant_(self.write_head_gate.bias, config.mamba_write_bias_init)

        self.tie_weights()
        self.main_input_name = "input_ids"
        self.model.config.use_cache = False
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # turn on gradient checkpointing to save gpu ram
        if getattr(config, "use_gradient_checkpointing", False):
            self.gradient_checkpointing_enable()

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

        NOTE: Adam does NOT compose with the gated update rules (convex/mamba):
        those replace the SGD step with a gated recurrence and have no notion of
        a momentum/variance buffer. Adam is only used when memory_update_rule="sgd".

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

            # bias-correction with current step
            m_hat = m_new / (1 - beta1 ** step_idx)
            v_hat = v_new / (1 - beta2 ** step_idx)
            v_hat = torch.clamp(v_hat, min=eps)        # avoid sqrt(0)/0

            step = lr * m_hat / (v_hat.sqrt() + eps)

        # ---- 3. write back *detached* buffers ---------------------------- #
        state["m"].copy_(m_new.detach())
        state["v"].copy_(v_new.detach())

        # ---- 4. return new parameter tensor (on graph via g) ------------- #
        return p - step

    def _clip_grad(self, g):
        """Element-wise and per-sample total-norm clipping. Shared by all update rules."""
        if self.inner_clip_value is not None:
            g = torch.clamp(g, -self.inner_clip_value, self.inner_clip_value)
        if self.inner_clip_norm is not None:
            # scale gradient if its 2-norm is too large; per-sample (all non-batch dims)
            reduce_dims = tuple(range(1, g.ndim))
            g_norm = g.norm(dim=reduce_dims, keepdim=True)
            scale = self.inner_clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > self.inner_clip_norm, g * scale, g)
        return g

    def _sgd_step(self, p, g, lr=None):
        """Stateless SGD: m = m - lr*g. r_t = 1, w_t = 1 (degenerate running-sum RNN)."""
        g = self._clip_grad(g)
        if lr is None:
            lr = self.lr
        return p - lr * g

    def _gate_features(self, m, g):
        """Per-token gate input phi. m, g: [B, M, d] -> [B, M, in_dim]."""
        if self.gate_features == "grad":
            return g
        if self.gate_features == "state":
            return m
        return torch.cat([g, m], dim=-1)  # "grad_state"

    def _gated_step(self, m, g):
        """
        Adaptive memory update: m_t = r_t * m_{t-1} + w_t * (-lr * g_t).

        - "convex": r = 1-a, w = a, a = sigmoid(lin(phi))   (single gate, Family 3)
        - "mamba":  r = exp(-Delta) [or sigmoid], Delta = softplus(lin_r(phi));
                    w = sigmoid(lin_w(phi))                 (Family 2 / 1)

        Both modes clip the gradient first (same as _sgd_step) and are fully
        differentiable, so second-order MAML flows grad to the gate heads.
        Returns (m_new, gate_stats) where gate_stats is a dict of detached
        per-sample means for diagnostics (None for unused gates).
        """
        g = self._clip_grad(g)
        x = -self.lr * g                                   # SGD step = input signal
        phi = self._gate_features(m, g)
        stats = {}

        if self.memory_update_rule == "convex":
            a = torch.sigmoid(self.convex_head(phi))       # [B,M,out] (broadcast over d if per_token)
            m_new = (1.0 - a) * m + a * x
            stats["gate_write_mean"] = a.mean().detach()    # = w = a; r = 1-a
            return m_new, stats

        # "mamba"
        if self.gate_retention == "exp":
            delta = F.softplus(self.retain_head(phi))       # >= 0
            retain = torch.exp(-delta)                      # in (0, 1]
            stats["gate_delta_mean"] = delta.mean().detach()
        else:  # "sigmoid" (GRU-style retention, Family 1)
            retain = torch.sigmoid(self.retain_head(phi))   # in (0, 1)
        write = torch.sigmoid(self.write_head_gate(phi))    # in (0, 1)
        m_new = retain * m + write * x
        stats["gate_retain_mean"] = retain.mean().detach()
        stats["gate_write_mean"] = write.mean().detach()
        return m_new, stats

    @staticmethod
    def _apply_linear(mem, W, b):
        """
        Functional linear on a batch of memories:
        mem: (B,M,d), W: (B,d,d) or None, b: (B,d) or None
        """
        if W is None:
            return mem
        return torch.baddbmm(b.unsqueeze(1), mem, W.transpose(1, 2))

    def forward(self, input_ids, labels=None, return_mem=False):
        # context_input_ids : B × S   (segments only, each ends with `|`)
        # query_input_ids   : B × Q   (e.g.  "?!K:V!|") i.e. the last segment
        # labels            : B × Q   (-100 everywhere except the target tokens (V!|))

        """
        All tensors already padded to the same length in the datacollator.
        """
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']

        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)
        inner_loss = torch.tensor(0.0, device=device)

        # actual model inputs starts after mem tokens and ctrl tokens
        mem_offset = self.n_mem_tokens + self.n_ctrl_tokens * 2

        # ctrl tokens
        if self.n_ctrl_tokens > 0:
            write_st_batch = self.write_st.unsqueeze(0).expand(B, -1, -1)
            write_end_batch = self.write_end.unsqueeze(0).expand(B, -1, -1)
            read_st_batch = self.read_st.unsqueeze(0).expand(B, -1, -1)
            read_end_batch = self.read_end.unsqueeze(0).expand(B, -1, -1)
        else:
            write_st_batch = write_end_batch = None
            read_st_batch = read_end_batch = None

        # ---------------------------------------------------------------- #
        # Cross-segment carry: mem_batch is initialised ONCE from self.mem
        # and flows through every segment (no per-segment reset). This is the
        # RNN hidden state. mem_batch_initial is kept for stats / delta norm.
        # ---------------------------------------------------------------- #
        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]
        mem_batch_initial = mem_batch.clone()

        # per-sample params for mem_proj (fast weights; reset per segment, like grad_memgpt.py)
        if self.mem_proj_mode == "per_sample":
            W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
            b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()

        # handling gradients for the carried memory state (meta-params):
        # "none" detaches the recurrence entirely; "first"/"second" keep the graph.
        if self.grad_mode == "none":
            mem_batch = mem_batch.detach().requires_grad_(True)
            if self.mem_proj_mode == "per_sample":
                W_batch = W_batch.detach().requires_grad_(True)
                b_batch = b_batch.detach().requires_grad_(True)
        else:
            mem_batch = mem_batch.requires_grad_(True)
            if self.mem_proj_mode == "per_sample":
                W_batch = W_batch.requires_grad_(True)
                b_batch = b_batch.requires_grad_(True)

        opt_state = {}                       # moments for stateless Adam (reset per segment below)
        inner_loop_stats = {'inner_grad_norm_mean': torch.tensor(0.0, device=device),
                            'inner_grad_norm_max': torch.tensor(-1.0, device=device),
                            'inner_grad_norm_min': torch.tensor(1e06, device=device)}
        if self.memory_update_rule != "sgd":
            inner_loop_stats['gate_retain_mean'] = torch.tensor(0.0, device=device)
            inner_loop_stats['gate_write_mean'] = torch.tensor(0.0, device=device)
            if self.memory_update_rule == "mamba" and self.gate_retention == "exp":
                inner_loop_stats['gate_delta_mean'] = torch.tensor(0.0, device=device)
            inner_loop_stats['_n_gate_steps'] = 0
        total_inner_steps = 0

        # ---------------------------------------------------------------- #
        # 1.  INNER loop on context. WRITE context to mem, segment by segment.
        # ---------------------------------------------------------------- #
        ctx_emb = None
        if self.K and context_input_ids.ne(pad_id).any():
            # re-enable autograd even if outer context is `no_grad`
            with torch.enable_grad():
                # build ctx embedding once, then reuse it with updated mem
                ctx_emb = self.model.get_input_embeddings()(context_input_ids)      # [B,S,d]
                # lm labels: reconstructing the context, last mem/ctrl token predicts the first token of the context
                lm_labels = context_input_ids.clone()
                lm_labels[lm_labels == pad_id] = -100
                # loss mask
                mask = (lm_labels != -100)

                # ---- split context into segments (ceil-padded to equal size) ---- #
                seq_len = ctx_emb.size(1)
                if self.segment_size is not None:
                    n_segments = math.ceil(seq_len / self.segment_size)
                    segment_size = self.segment_size
                else:
                    n_segments = self.n_segments
                    segment_size = (seq_len + n_segments - 1) // n_segments  # ceil division
                pad_len = segment_size * n_segments - seq_len
                if pad_len > 0:
                    # left-pad so real tokens keep their trailing `|` boundary at segment ends
                    ctx_emb = F.pad(ctx_emb, [0, 0, pad_len, 0], "constant", 0)
                    mask = F.pad(mask, [pad_len, 0], "constant", 0)
                    lm_labels = F.pad(lm_labels, [pad_len, 0], "constant", -100)

                for seg_idx in range(n_segments):
                    seg_start = seg_idx * segment_size
                    seg_end = seg_start + segment_size
                    seg_emb = ctx_emb[:, seg_start:seg_end, :]
                    seg_mask = mask[:, seg_start:seg_end]
                    seg_labels = lm_labels[:, seg_start:seg_end]

                    # per-sample: does this segment have any real tokens?
                    seg_has_tokens = seg_mask.any(dim=1)  # [B]
                    if not seg_has_tokens.any():
                        continue

                    seg_seq_len = seg_mask.sum(dim=1).clamp_min(1)  # [B]

                    # ---- truncated BPTT over the segment recurrence ---- #
                    # Detach the carried state at boundaries older than seg_bptt
                    # segments (only matters for graph-carrying grad_modes).
                    # The VALUE is preserved (detach is identity w.r.t. data),
                    # so cross-segment carry is unaffected; only the autograd
                    # graph is cut to bound activation memory over long unrollings.
                    if (self.grad_mode != "none" and self.seg_bptt is not None
                            and (n_segments - seg_idx - 1) >= self.seg_bptt):
                        mem_batch = mem_batch.detach().requires_grad_(True)

                    # reset per-sample mem_proj fast weights and Adam state per segment
                    # (the persistent state is mem_batch itself; proj is a per-segment readout)
                    if self.mem_proj_mode == "per_sample":
                        W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
                        b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()
                        if self.grad_mode == "none":
                            W_batch = W_batch.detach().requires_grad_(True)
                            b_batch = b_batch.detach().requires_grad_(True)
                        else:
                            W_batch = W_batch.requires_grad_(True)
                            b_batch = b_batch.requires_grad_(True)
                    opt_state = {}

                    # build attention mask + position_ids for this segment so that
                    # left-pad positions are neither attended to nor counted in positions
                    cur_mem_attn_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
                    if self.n_ctrl_tokens > 0:
                        cur_ctrl_attn_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
                        cur_attn_mask = torch.cat([cur_ctrl_attn_mask, cur_mem_attn_mask,
                                                   cur_ctrl_attn_mask, seg_mask.long()], dim=1)
                    else:
                        cur_attn_mask = torch.cat([cur_mem_attn_mask, seg_mask.long()], dim=1)
                    cur_position_ids = (cur_attn_mask.cumsum(-1) - 1).clamp(min=0)

                    for k in range(self.K):
                        if self.mem_proj_mode == 'none':
                            mem_inp = mem_batch
                        elif self.mem_proj_mode == 'proj':
                            mem_inp = self.mem_proj(mem_batch)
                        else:  # per_sample
                            mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

                        if self.n_ctrl_tokens > 0:
                            x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch, seg_emb], dim=1)
                        else:
                            x_ctx = torch.cat([mem_inp, seg_emb], dim=1)    # [B,M+seg_size,d]

                        if self.use_write_head:
                            outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=cur_attn_mask,
                                                            position_ids=cur_position_ids, return_dict=True)
                            h = outs.last_hidden_state                     # [B,M+seg_size,V]
                            h = h[:, mem_offset-1:, :]                     # [B,seg_size,V]
                            logits = self.write_head(h)
                            del h
                        else:
                            outs = self.model(inputs_embeds=x_ctx, attention_mask=cur_attn_mask,
                                              position_ids=cur_position_ids, return_dict=True)
                            logits = outs.logits                           # [B,M+seg_size,V]
                            logits = logits[:, mem_offset-1:, :]           # [B,seg_size,V]

                        seg_inner_loss = nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            seg_labels.reshape(-1),
                            ignore_index=-100,
                            reduction='none',
                        ).view(B, -1)
                        # per-sample losses, invariant to batch size B:
                        # g_i = d inner_loss / d mem_i = d inner_loss_i / d mem_i
                        seg_inner_loss = (seg_inner_loss * seg_mask).sum(1) / seg_seq_len
                        inner_loss = seg_inner_loss.sum()
                        del outs, logits

                        is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
                        create_graph = is_second_order_step
                        # only the last segment's inner_loss needs to be retained for add_inner_loss_to_outer
                        is_last_segment = (seg_idx == n_segments - 1)
                        retain_graph = create_graph or (self.add_inner_loss_to_outer
                                                        and (k == self.K - 1) and is_last_segment)

                        # get inner loop gradients
                        if self.mem_proj_mode == 'per_sample':
                            g_mem, g_W, g_b = torch.autograd.grad(inner_loss, [mem_batch, W_batch, b_batch],
                                                                  create_graph=create_graph, retain_graph=retain_graph)
                        else:
                            g_mem = torch.autograd.grad(inner_loss, mem_batch,
                                                        create_graph=create_graph, retain_graph=retain_graph)[0]

                        # track inner grad norm
                        g_norm = g_mem.reshape(B, -1).norm(dim=1).detach()
                        inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
                        inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
                        inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())

                        # ---- apply the (possibly gated) memory update ---- #
                        if self.memory_update_rule == "sgd":
                            if self.use_adam:
                                mem_batch = self._adam_step(mem_batch, g_mem, opt_state.setdefault('mem', {}),
                                                            k + 1, self.lr)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = self._adam_step(W_batch, g_W, opt_state.setdefault('W', {}),
                                                              k + 1, self.lr)
                                    b_batch = self._adam_step(b_batch, g_b, opt_state.setdefault('b', {}),
                                                              k + 1, self.lr)
                                    raise NotImplementedError("Adam is not tested, be careful!")
                            else:
                                mem_batch = self._sgd_step(mem_batch, g_mem)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = self._sgd_step(W_batch, g_W)
                                    b_batch = self._sgd_step(b_batch, g_b)
                        else:
                            # convex / mamba gated recurrence. NOTE: does not
                            # compose with Adam (no moment buffers); always SGD-style
                            # write gate on the clipped gradient.
                            mem_batch, gate_stats = self._gated_step(mem_batch, g_mem)
                            if self.mem_proj_mode == 'per_sample':
                                W_batch, _ = self._gated_step(W_batch, g_W)
                                b_batch, _ = self._gated_step(b_batch, g_b)
                            for kk, vv in gate_stats.items():
                                inner_loop_stats[kk] = inner_loop_stats[kk] + vv
                            inner_loop_stats['_n_gate_steps'] += 1

                        if self.grad_mode in ['none']:
                            mem_batch = mem_batch.detach().requires_grad_(True)
                            if self.mem_proj_mode == 'per_sample':
                                W_batch = W_batch.detach().requires_grad_(True)
                                b_batch = b_batch.detach().requires_grad_(True)
                        elif self.grad_mode in ['first', 'second']:
                            pass  # do nothing, keep gradients flow

                        total_inner_steps += 1

        if total_inner_steps > 0:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / total_inner_steps
            if self.K:
                inner_loop_stats['inner_loss'] = inner_loss.detach() / B
        if self.memory_update_rule != "sgd" and inner_loop_stats.get('_n_gate_steps', 0) > 0:
            n = inner_loop_stats['_n_gate_steps']
            inner_loop_stats['gate_retain_mean'] /= n
            inner_loop_stats['gate_write_mean'] /= n
            if 'gate_delta_mean' in inner_loop_stats:
                inner_loop_stats['gate_delta_mean'] /= n
            del inner_loop_stats['_n_gate_steps']

        # mem_batch: [B,M,d]  (final carried state, encodes all segments)
        mem_norm = mem_batch.norm(dim=(1, 2)).detach()  # B
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()
        inner_loop_stats['mem_norm_min'] = mem_norm.min()
        # log how mem has changed from initial state to state after inner loop
        detla_mem_norm = (mem_batch - mem_batch_initial).detach().norm(dim=(1, 2))
        inner_loop_stats['delta_mem_norm_mean'] = detla_mem_norm.mean()
        inner_loop_stats['delta_mem_norm_max'] = detla_mem_norm.max()
        inner_loop_stats['delta_mem_norm_min'] = detla_mem_norm.min()

        if ctx_emb is not None:
            del ctx_emb, lm_labels

        # ---------------------------------------------------------------- #
        # 2.  READ phase – compute outer loss on target predictions based on query, read from mem
        # ---------------------------------------------------------------- #
        qry_emb = self.model.get_input_embeddings()(query_input_ids)          # [B,Q,d]

        if self.mem_proj_mode == "none":
            mem_inp = mem_batch
        elif self.mem_proj_mode == "proj":
            mem_inp = self.mem_proj(mem_batch)
        else:  # "per_sample"
            mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

        if self.n_ctrl_tokens > 0:
            # add params that can control read operation from mem
            x_qry = torch.cat([read_st_batch, mem_inp, read_end_batch, qry_emb], dim=1)
        else:
            x_qry = torch.cat([mem_inp, qry_emb], dim=1)                      # [B,M+Q,d]

        # pad to multiple of 32 for compatibility with JVP Flash Attention
        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_list = [0, 0, 0, -x_qry.size(1) % 32]
            x_qry = F.pad(x_qry, pad_list, "constant", 0)

        with self._disable_write_lora():
            logits_q = self.model(inputs_embeds=x_qry).logits                 # [B,M+Q,V]
        logits_q = logits_q[:, mem_offset-1:mem_offset+qry_emb.size(1), :]    # [B,Q+1,V]

        output = {'predictions': logits_q, 'inner_loop_stats': inner_loop_stats}
        if return_mem:
            output['mem'] = mem_batch
            if self.mem_proj_mode == "per_sample":
                output['W'] = W_batch
                output['b'] = b_batch

        if labels is None:
            return output

        # logits has prediction for +1 token, so we cut it as we do not have label for it
        # labels are not shifted, as we take prediction for the first token from mem vectors
        target_loss = nn.functional.cross_entropy(
            logits_q[:, :-1].reshape(-1, logits_q.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        output['inner_loop_stats']['target_loss'] = target_loss.detach()
        if self.add_inner_loss_to_outer:
            inner_loss_mean = inner_loss / B
            combined_loss = target_loss + self.inner_loss_weight * inner_loss_mean
        else:
            combined_loss = target_loss
        output['loss'] = combined_loss
        return output
