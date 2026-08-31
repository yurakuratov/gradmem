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
import re

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
                 # ---- VAE-style memory regularisation (default-off) ---- #
                 # See "Recent additions" in AGENTS.md for the full rationale.
                 mem_noise_std=0.0,           # σ: fixed Gaussian noise on mem before READ (0 = off)
                 mem_prior_weight=0.0,        # λ: weight of MSE(m_0, m_final) outer-loop prior (0 = off)
                 mem_prior_anneal_steps=0,    # λ warmup: 0 -> mem_prior_weight over this many steps (0 = constant)
                 # ---- STAR worst-case stability regulariser (default-off) ---- #
                 # Adapted from "STAR: Stability-Inducing Weight Perturbation for
                 # Continual Learning" (ICLR 2025) with the continually-updated
                 # weights replaced by the memory tokens: at each segment boundary
                 # the READ outputs on already-stored KV probes must stay stable
                 # under the worst-case (gradient-ascent) perturbation of the
                 # memory. See _star_boundary_loss for the exact math.
                 star_weight=0.0,             # λ: outer-loss weight (0 = off)
                 star_gamma=0.01,             # γ: relative perturbation size ‖δ_j‖/‖m_j‖ per memory token
                 star_epsilon=1e-2,           # ε: δ0 noise std as a fraction of the per-token mem norm
                 star_ascent_steps=1,         # gradient-ascent steps for the worst-case δ (paper: 1)
                 star_perturbation="grad",    # "grad" (worst case) | "random" (matched-magnitude noise ablation)
                 star_probe_scope="all",      # probes from segments "all" (<= s) | "past" (< s)
                 star_correct_only=True,      # restrict probes to currently-retrievable pairs (STAR's x*)
                 star_boundaries="all",       # which segment boundaries: "all" | "last"
                 star_anneal_steps=0,         # λ warmup: 0 -> star_weight over this many steps (0 = constant)
                 # ---- outer-loss vanilla replay (default-off; STAR ablation) ---- #
                 # Meta-level rehearsal on the SAME probes/boundaries as STAR:
                 # + rho * CE(READ(m_s; probe), answer) -- corrective where STAR
                 # is preventive (non-zero gradient whenever an old pair is
                 # answered imperfectly). See _replay_boundary_loss.
                 replay_weight=0.0,           # rho: outer-loss weight (0 = off)
                 replay_boundaries="all",     # which segment boundaries: "all" | "last"
                 replay_probe_scope="all",    # probes from segments "all" (<= s) | "past" (< s)
                 replay_correct_only=True,    # restrict probes to currently-retrievable pairs
                 replay_anneal_steps=0,       # rho warmup: 0 -> replay_weight over this many steps
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
            mem_noise_std: float, VAE-style sampling. After the WRITE inner loop the
                final (deterministic) memory m is perturbed as m_used = m + sigma*eps,
                eps ~ N(0, I), and READ consumes m_used. Reparameterised, so gradients
                flow to the mean m and through it to the inner-loop params; eps carries
                no grad. Train-only (eval READ stays deterministic). 0.0 = off.
            mem_prior_weight: float, VAE-style prior regularisation weight lambda.
                Adds lambda * MSE(m_0, m_final) to the OUTER loss only, where m_0 are
                the initial (meta-learned) memory tokens. m_0 is DETACHED in the term,
                so the reg pulls m -> m_0 but never m_0 -> m (avoids the "prior drifts
                to satisfy the reg" collapse). Computed on the deterministic endpoint
                (pre-sampling), not the noisy sample. 0.0 = off.
            mem_prior_anneal_steps: int, lambda warmup: linear 0 -> mem_prior_weight
                over this many outer steps, then constant. 0 = constant lambda from
                step 0 (no warmup).
            star_weight: float, STAR stability-regulariser weight lambda (0 = off).
                Adds lambda * mean over segment boundaries of the worst-case KL
                    L_STAR(m_s) = KL( READ(m_s; probes) || READ(m_s + delta; probes) )
                    delta = argmax over per-token-normalised perturbations
                where probes are teacher-forced ?!K:V!| queries of KV pairs already
                written into memory (see star_probe_scope / star_correct_only).
                delta is found with star_ascent_steps noise-initialised gradient-
                ascent steps (paper Eq. 9-11, adapted per memory token instead of
                per layer) and is DETACHED from the meta-graph, as is the reference
                branch -- so the term never exceeds the existing second-order
                autograd depth. Train-only: eval forwards are unaffected.
            star_gamma: float, relative perturbation size gamma: after the ascent
                step, ‖delta_j‖_2 / ‖m_j‖_2 ~= gamma per memory token j (paper
                grid-searches 0.001..0.05).
            star_epsilon: float, delta0 noise std as a fraction of the per-token
                mem norm (delta escapes the all-zero-gradient point at delta=0,
                where the KL and its gradient vanish).
            star_ascent_steps: int, number of normalised gradient-ascent steps for
                delta (paper: 1 is best, >=5 degrades the first-order approximation).
            star_perturbation: "grad" (worst-case direction, the method) or
                "random" (delta = gamma-normalised Gaussian noise, the paper's
                Table-3 ablation -- also the adversarial analogue of mem_noise_std).
            star_probe_scope: "all" probes from segments <= s (buffer-at-end-of-task
                semantics, like STAR's rehearsal buffer after task t) or "past"
                segments < s only (excludes the just-written segment).
            star_correct_only: bool, restrict probes to pairs currently retrieved
                with exact match under m_s (STAR's x*: "you can only forget what
                you currently know"; enforcing stability on not-yet-learned content
                collapses performance in the paper's Table 4).
            star_boundaries: "all" segment boundaries or only the "last" one
                (cheapest; for S=2 they coincide).
            star_anneal_steps: int, lambda warmup: linear 0 -> star_weight over
                this many outer steps, then constant. 0 = constant from step 0.
            replay_weight: float, outer-loss vanilla-replay weight rho (0 = off).
                Adds rho * mean over segment boundaries of the CE of READ(m_s;
                probe) against the answer tokens the teacher-forced probe
                already contains -- the corrective counterpart of STAR's
                preventive KL (non-zero gradient whenever an old pair is
                answered imperfectly). Uses the SAME probe set, boundaries and
                correct-only masking as STAR so the {replay} x {STAR} ablation
                isolates the functional form. One extra probe forward per
                boundary (cheaper than STAR's three).
            replay_boundaries: "all" | "last" -- same semantics as star_boundaries.
            replay_probe_scope: "all" | "past" -- same semantics as star_probe_scope.
            replay_correct_only: bool, same semantics as star_correct_only.
            replay_anneal_steps: int, rho warmup: linear 0 -> replay_weight over
                this many outer steps, then constant. 0 = constant.

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

        # VAE-style memory regularisation
        self.mem_noise_std = mem_noise_std
        self.mem_prior_weight = mem_prior_weight
        self.mem_prior_anneal_steps = mem_prior_anneal_steps

        # STAR worst-case stability regulariser
        self.star_weight = star_weight
        self.star_gamma = star_gamma
        self.star_epsilon = star_epsilon
        self.star_ascent_steps = star_ascent_steps
        self.star_perturbation = star_perturbation
        self.star_probe_scope = star_probe_scope
        self.star_correct_only = star_correct_only
        self.star_boundaries = star_boundaries
        self.star_anneal_steps = star_anneal_steps

        # outer-loss vanilla replay (STAR ablation)
        self.replay_weight = replay_weight
        self.replay_boundaries = replay_boundaries
        self.replay_probe_scope = replay_probe_scope
        self.replay_correct_only = replay_correct_only
        self.replay_anneal_steps = replay_anneal_steps

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

        # Validate VAE-style regularisation settings
        assert mem_noise_std >= 0.0, f"mem_noise_std must be >= 0.0, got {mem_noise_std}"
        assert mem_prior_weight >= 0.0, f"mem_prior_weight must be >= 0.0, got {mem_prior_weight}"
        assert mem_prior_anneal_steps >= 0, \
            f"mem_prior_anneal_steps must be >= 0, got {mem_prior_anneal_steps}"

        # Validate STAR regulariser settings
        assert star_weight >= 0.0, f"star_weight must be >= 0.0, got {star_weight}"
        assert star_gamma >= 0.0, f"star_gamma must be >= 0.0, got {star_gamma}"
        assert star_epsilon >= 0.0, f"star_epsilon must be >= 0.0, got {star_epsilon}"
        assert star_ascent_steps >= 1, \
            f"star_ascent_steps must be >= 1, got {star_ascent_steps}"
        assert star_perturbation in ("grad", "random"), \
            f"star_perturbation must be 'grad' or 'random', got '{star_perturbation}'"
        assert star_probe_scope in ("all", "past"), \
            f"star_probe_scope must be 'all' or 'past', got '{star_probe_scope}'"
        assert isinstance(star_correct_only, bool), "star_correct_only must be a bool"
        assert star_boundaries in ("all", "last"), \
            f"star_boundaries must be 'all' or 'last', got '{star_boundaries}'"
        assert star_anneal_steps >= 0, \
            f"star_anneal_steps must be >= 0, got {star_anneal_steps}"

        # Validate vanilla-replay settings (STAR ablation)
        assert replay_weight >= 0.0, f"replay_weight must be >= 0.0, got {replay_weight}"
        assert replay_boundaries in ("all", "last"), \
            f"replay_boundaries must be 'all' or 'last', got '{replay_boundaries}'"
        assert replay_probe_scope in ("all", "past"), \
            f"replay_probe_scope must be 'all' or 'past', got '{replay_probe_scope}'"
        assert isinstance(replay_correct_only, bool), "replay_correct_only must be a bool"
        assert replay_anneal_steps >= 0, \
            f"replay_anneal_steps must be >= 0, got {replay_anneal_steps}"


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

        # VAE-style memory regularisation
        self.mem_noise_std = config.mem_noise_std
        self.mem_prior_weight = config.mem_prior_weight
        self.mem_prior_anneal_steps = config.mem_prior_anneal_steps

        # STAR worst-case stability regulariser (no new parameters: pure loss term)
        self.star_weight = config.star_weight
        self.star_gamma = config.star_gamma
        self.star_epsilon = config.star_epsilon
        self.star_ascent_steps = config.star_ascent_steps
        self.star_perturbation = config.star_perturbation
        self.star_probe_scope = config.star_probe_scope
        self.star_correct_only = config.star_correct_only
        self.star_boundaries = config.star_boundaries
        self.star_anneal_steps = config.star_anneal_steps

        # outer-loss vanilla replay (STAR ablation)
        self.replay_weight = config.replay_weight
        self.replay_boundaries = config.replay_boundaries
        self.replay_probe_scope = config.replay_probe_scope
        self.replay_correct_only = config.replay_correct_only
        self.replay_anneal_steps = config.replay_anneal_steps

        self.current_train_step = 0   # stamped by CustomTrainer.compute_loss via set_train_step

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

    # ---------------------------------------------------------------- #
    # Train-step / schedule hooks for the VAE-style prior warmup.
    # The adaptive model extends PreTrainedModel directly (it does NOT
    # inherit the base grad_memgpt.GradMemGPT), so these accessors must be
    # defined here. CustomTrainer.compute_loss already calls
    # m.set_train_step(self.state.global_step) guarded by hasattr; without
    # these the call silently no-ops and current_train_step stays at 0.
    # Mirrors the _energy_recon_weight_now shape in grad_memgpt.py:776.
    # ---------------------------------------------------------------- #
    def set_train_step(self, step):
        """Stamped by the trainer so the lambda warmup schedule knows the outer step."""
        self.current_train_step = int(step)

    def _mem_prior_weight_now(self):
        """Lambda schedule: 0 -> mem_prior_weight over mem_prior_anneal_steps, then constant."""
        if self.mem_prior_anneal_steps <= 0:
            return self.mem_prior_weight
        frac = min(self.current_train_step / self.mem_prior_anneal_steps, 1.0)
        return self.mem_prior_weight * frac

    def _star_weight_now(self):
        """STAR lambda schedule: 0 -> star_weight over star_anneal_steps, then constant."""
        if self.star_anneal_steps <= 0:
            return self.star_weight
        frac = min(self.current_train_step / self.star_anneal_steps, 1.0)
        return self.star_weight * frac

    def _replay_weight_now(self):
        """Replay rho schedule: 0 -> replay_weight over replay_anneal_steps, then constant."""
        if self.replay_anneal_steps <= 0:
            return self.replay_weight
        frac = min(self.current_train_step / self.replay_anneal_steps, 1.0)
        return self.replay_weight * frac

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

    # ---------------------------------------------------------------- #
    # STAR worst-case stability regulariser (train-only).
    # ---------------------------------------------------------------- #
    # Adapted from STAR (ICLR 2025, arXiv:2503.01595) with the continually-
    # updated model weights replaced by the carried memory state: at a
    # segment boundary s, the READ output distribution on already-stored KV
    # probes must stay stable under a worst-case perturbation of m_s,
    #
    #     L_STAR(m_s) = KL( READ(m_s; p*) || READ(m_s + delta; p*) )
    #     delta = delta0 + gamma * (||m_j|| / ||g_j||) * g_j     (per token j)
    #     g      = d KL / d (m + delta)  at the perturbed point
    #     delta0 = eps * ||m_j|| * N(0, I)
    #
    # The perturbation plays the role of STAR's "future parameter update":
    # the inner-loop GD steps of LATER segments move the memory in (at the
    # boundary) unknown directions, so robustness is enforced against the
    # worst case. p* are teacher-forced ?!K:V!| probe queries of KV pairs
    # already written into memory, restricted to currently-retrievable ones
    # (exact match under m_s) -- STAR's x*: you can only forget what you
    # currently know. Only the PERTURBED branch carries meta-gradient; the
    # reference branch and delta are detached (the paper's first-order
    # approximation), so the term never exceeds the existing second-order
    # autograd depth. "random" replaces the ascent direction with matched-
    # magnitude Gaussian noise (the paper's Table-3 ablation; the adversarial
    # analogue of mem_noise_std). No new parameters: a pure loss term.

    @staticmethod
    def _star_kl(ref_logits, pert_logits, score, selected):
        """KL(ref || pert) summed over scored positions, meaned over selected probes.

        Slot t of the [.., Q+1, V] READ logits predicts token t (the same
        convention as _probe_exact_match), so slot t is compared against the
        query token at position t. score/selected: [B, n_q, Q] / [B, n_q].
        """
        log_ref = F.log_softmax(ref_logits.float(), dim=-1)
        log_pert = F.log_softmax(pert_logits.float(), dim=-1)
        # per-slot KL: sum_v p_v (log p_v - log q_v), then keep slots 0..Q-1
        kl_pos = (log_ref.exp() * (log_ref - log_pert)).sum(-1)          # [B, n_q, Q+1]
        kl_probe = (kl_pos[:, :, :-1] * score).sum(-1)                   # [B, n_q]
        denom = selected.float().sum().clamp_min(1.0)
        return (kl_probe * selected.float()).sum() / denom

    @staticmethod
    def _probe_exact_match_batch(logits, qids, tmask, ignore_token_ids):
        """Vectorised exact match over a whole probe batch: [B, n_q, Q+1, V].

        Same scoring as _probe_exact_match (slot t predicts token t; scored on
        target_mask minus ignore tokens; nothing-to-score counts as correct)
        but in a handful of tensor ops -- no Python loops and, critically, no
        per-probe device sync (the scalar helper's `if score.sum() == 0:`
        syncs once per probe, which is latency-bound and can dominate the
        STAR/replay boundary cost). Returns [B, n_q] bool.
        """
        preds = logits.argmax(dim=-1)[:, :, :-1]         # [B, n_q, Q]
        score = tmask.clone()
        for ig in ignore_token_ids:
            score &= (qids != ig)
        return ((preds == qids) | ~score).all(dim=-1)

    def _star_boundary_loss(self, mem_batch, kv_queries, boundary_seg,
                            W_batch=None, b_batch=None,
                            read_st_batch=None, read_end_batch=None):
        """Worst-case KL at one segment boundary. Returns None when no probe
        is eligible (empty scope, or nothing currently retrievable), else
        (loss, n_probes, delta_ratio) where loss is a scalar carrying
        meta-gradient through the perturbed branch only.
        """
        qids = kv_queries['query_input_ids']            # [B, n_q, Q]
        tmask = kv_queries['target_mask']               # [B, n_q, Q] bool
        seg_idx = kv_queries['seg_idx']                 # [B, n_q]
        qmask = kv_queries['mask']                      # [B, n_q] bool
        ignore_token_ids = kv_queries.get('ignore_token_ids', [])

        in_scope = qmask & (seg_idx <= boundary_seg if self.star_probe_scope == "all"
                            else seg_idx < boundary_seg)
        if not in_scope.any():
            return None

        mem_det = mem_batch.detach()
        W_det = W_batch.detach() if isinstance(W_batch, torch.Tensor) else W_batch
        b_det = b_batch.detach() if isinstance(b_batch, torch.Tensor) else b_batch

        # ---- reference branch (anchor q_theta_t): detached, no_grad ------- #
        with torch.no_grad():
            ref_logits = self._read_once(mem_det, qids, read_st_batch, read_end_batch,
                                         W_det, b_det).detach()
            if self.star_correct_only:
                # vectorised x* mask (no per-probe device syncs)
                selected = in_scope & self._probe_exact_match_batch(
                    ref_logits, qids, tmask, ignore_token_ids)
            else:
                selected = in_scope

        if not selected.any():
            return None

        # scored (target-content) positions for the KL: target_mask minus the
        # structural ignore tokens (! and |) -- same scoring as EM itself
        score = tmask.clone()
        for ig in ignore_token_ids:
            score &= (qids != ig)

        # ---- worst-case delta (per memory token, relative-normalised) ----- #
        tok_norm = mem_det.norm(dim=-1, keepdim=True)                     # [B, M, 1]
        if self.star_perturbation == "random":
            z = torch.randn_like(mem_det)
            z_norm = z.norm(dim=-1, keepdim=True)
            delta = self.star_gamma * tok_norm * z / (z_norm + 1e-12)
        else:
            delta = self.star_epsilon * tok_norm * torch.randn_like(mem_det)
            for _ in range(self.star_ascent_steps):
                # detached working point: the ascent never touches the meta-graph
                m_pert = (mem_det + delta).requires_grad_(True)
                pert_logits = self._read_once(m_pert, qids, read_st_batch, read_end_batch,
                                              W_det, b_det)
                kl = self._star_kl(ref_logits, pert_logits, score, selected)
                g, = torch.autograd.grad(kl, m_pert)
                g_norm = g.norm(dim=-1, keepdim=True)
                delta = delta + self.star_gamma * tok_norm * g / (g_norm + 1e-12)
        delta = delta.detach()

        # ---- final perturbed branch: the only meta-gradient path ---------- #
        star_logits = self._read_once(mem_batch + delta, qids, read_st_batch, read_end_batch,
                                      W_batch, b_batch)
        loss = self._star_kl(ref_logits, star_logits, score, selected)
        ratio = (delta.norm(dim=-1) / (tok_norm.squeeze(-1) + 1e-12)).mean().detach()
        return loss, selected.sum(), ratio

    def _replay_boundary_loss(self, mem_batch, kv_queries, boundary_seg,
                              W_batch=None, b_batch=None,
                              read_st_batch=None, read_end_batch=None):
        """Outer-loss vanilla replay (meta-level rehearsal) at one segment boundary.

        The corrective counterpart of _star_boundary_loss, kept deliberately
        symmetric so the {replay} x {STAR} ablation isolates the functional
        form on IDENTICAL data: same teacher-forced probes, same boundary
        semantics, same scope / correct-only selection. Where STAR penalises
        the worst-case OUTPUT DRIFT under a memory perturbation (zero gradient
        when retrieval is already robust -- preventive), replay directly
        minimises the CE of READ(m_s; probe) against the answer tokens the
        probe already contains (non-zero gradient whenever an old pair is
        answered imperfectly -- corrective, can recover dropped pairs).

        One grad-carrying probe forward per boundary (STAR needs three).
        Returns None when no probe is eligible, else
        (loss, n_probes, em_rate) where em_rate is the fraction of in-scope
        probes currently retrieved with exact match (a live forgetting
        measurement, also logged when the term itself is masked empty).
        """
        qids = kv_queries['query_input_ids']            # [B, n_q, Q]
        tmask = kv_queries['target_mask']               # [B, n_q, Q] bool
        seg_idx = kv_queries['seg_idx']                 # [B, n_q]
        qmask = kv_queries['mask']                      # [B, n_q] bool
        ignore_token_ids = kv_queries.get('ignore_token_ids', [])

        in_scope = qmask & (seg_idx <= boundary_seg if self.replay_probe_scope == "all"
                            else seg_idx < boundary_seg)
        if not in_scope.any():
            return None

        # one grad-carrying forward; the correct-only mask comes from its own
        # (detached) argmax, so the CE gradient flows only through selected
        # probes' logits
        logits = self._read_once(mem_batch, qids, read_st_batch, read_end_batch,
                                 W_batch, b_batch)
        em = self._probe_exact_match_batch(logits.detach(), qids, tmask, ignore_token_ids)
        em_rate = em[in_scope].float().mean().detach()

        selected = in_scope & em if self.replay_correct_only else in_scope
        if not selected.any():
            return None

        # scored positions (identical convention to _star_kl / EM): target_mask
        # minus the structural ignore tokens; slot t predicts token t
        score = tmask.clone()
        for ig in ignore_token_ids:
            score &= (qids != ig)

        logp = F.log_softmax(logits.float(), dim=-1)                     # [B, n_q, Q+1, V]
        tok_logp = logp[:, :, :-1].gather(-1, qids.unsqueeze(-1)).squeeze(-1)  # [B, n_q, Q]
        ce_probe = -(tok_logp * score).sum(-1)                           # [B, n_q]
        denom = selected.float().sum().clamp_min(1.0)
        loss = (ce_probe * selected.float()).sum() / denom
        return loss, selected.sum(), em_rate

    @staticmethod
    def _apply_linear(mem, W, b):
        """
        Functional linear on a batch of memories:
        mem: (B,M,d), W: (B,d,d) or None, b: (B,d) or None
        """
        if W is None:
            return mem
        return torch.baddbmm(b.unsqueeze(1), mem, W.transpose(1, 2))

    # ---------------------------------------------------------------- #
    # Per-segment forgetting evaluation (eval-only)
    # ---------------------------------------------------------------- #
    # After each model-segment is written to memory, probe retrieval of every
    # KV pair written so far (segments [0..k]) to track how older facts degrade
    # as later segments rewrite the carried memory state. Returns a per-sample
    # lower-triangular correctness matrix [B, n_seg, n_seg] (row = the segment a
    # KV pair lives in, col = the segment-after-write at which it was probed) and
    # a count mask of the same shape (which (s,k) cells were actually probed).
    # The trainer aggregates these across the eval set into the forgetting curve.
    # Eval-only, gradient-safe: the WRITE inner loop still runs under enable_grad
    # (the memory update needs autograd.grad), but every retrieval READ is wrapped
    # in no_grad + .detach() so it neither pollutes the inner-loop graph nor
    # retains activations. See forward_per_segment_eval below.

    @staticmethod
    def _segment_bounds(seq_len, n_segments_cfg, segment_size_cfg):
        """Same ceil-division chunking as forward() uses for the WRITE context.

        Returns (n_segments, segment_size, [(seg_start, seg_end), ...]) over the
        REAL (un-padded) token positions [0, seq_len). KV-pair attribution and
        the forward's segmentation must agree, so this is the single source of
        truth shared by _parse_kv_pairs and forward_per_segment_eval.
        """
        if segment_size_cfg is not None:
            n_segments = math.ceil(seq_len / segment_size_cfg)
            segment_size = segment_size_cfg
        else:
            n_segments = n_segments_cfg
            segment_size = (seq_len + n_segments - 1) // n_segments  # ceil division
        bounds = [(i * segment_size, min((i + 1) * segment_size, seq_len))
                  for i in range(n_segments)]
        return n_segments, segment_size, bounds

    def _parse_kv_pairs(self, context_input_ids, tokenizer, n_segments, segment_size):
        """Recover (query, target) probe pairs and their source model-segment.

        Tokenizer is 1 char = 1 token for the KV alphabet, so char offsets map
        directly to token positions. KV pairs are regex'd as ``!K:V!`` from the
        decoded context; each pair is attributed to the model-segment that
        FULLY contains it (a pair straddling a boundary -> the later segment,
        since it isn't fully "written" until that segment is processed).

        Returns a list (one per batch sample) of lists of
        ``(query_ids[Q], target_ids[T], source_seg_idx)``.
        """
        device = context_input_ids.device
        pad_id = tokenizer.pad_token_id
        results = []
        for b in range(context_input_ids.size(0)):
            ids = context_input_ids[b]
            real = ids[ids != pad_id] if (ids == pad_id).any() else ids
            real_ids = real.tolist()
            # decode without special tokens; KV syntax chars are in-vocab regular tokens
            text = tokenizer.decode(real_ids, skip_special_tokens=True)
            # !K:V!  -- non-greedy value; K/V are runs of the kv alphabet (no '!|')
            spans = []
            for m in re.finditer(r'!([^!|:]+):([^!|]+)!', text):
                k, v = m.group(1), m.group(2)
                # char span of this whole !K:V! == token span (1 char/token)
                tok_start = m.start()
                tok_end = m.end()          # exclusive
                spans.append((k, v, tok_start, tok_end))
            # attribute each KV to the model-segment fully containing it
            sample_pairs = []
            for k, v, ts, te in spans:
                # Attribute to the segment containing the KV's LAST token (closing
                # '!'): a KV is only retrievable once its whole !K:V! span is
                # written, i.e. when the segment containing the span's END is
                # processed. End-position (not start) is correct under straddling.
                src_seg = min((te - 1) // segment_size, n_segments - 1) if segment_size else 0
                query = f'?!{k}:'
                target = f'{v}!|'
                q_ids = tokenizer(query, add_special_tokens=False).input_ids
                t_ids = tokenizer(target, add_special_tokens=False).input_ids
                sample_pairs.append((q_ids, t_ids, src_seg))
            results.append(sample_pairs)
        return results

    def _read_once(self, mem_batch, query_input_ids, read_st_batch, read_end_batch,
                   W_batch=None, b_batch=None):
        """One batched READ against the current memory state.

        Faithful factor of the READ phase (forward lines ~895-917): project mem,
        concat [read_st?, mem, read_end?, query], forward under disabled write-LoRA,
        slice out the per-query-token logits. Batched over the query axis by
        flattening [B, n_q, Q] -> [B*n_q, Q] so this is a single forward pass.

        Args:
            mem_batch: [B, M, d] carried memory state (caller passes .detach()).
            query_input_ids: [B, n_q, Q] probe queries (already padded by collator).
        Returns:
            logits [B, n_q, Q+1, V].
        """
        B, n_q, Q = query_input_ids.shape
        emb_layer = self.model.get_input_embeddings()
        # [B, n_q, Q, d] -> [B*n_q, Q, d]
        qry_emb = emb_layer(query_input_ids.reshape(B * n_q, Q))
        mem_rep = mem_batch.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, *mem_batch.shape[1:])

        if self.mem_proj_mode == "none":
            mem_inp = mem_rep
        elif self.mem_proj_mode == "proj":
            mem_inp = self.mem_proj(mem_rep)
        else:  # "per_sample"
            W_rep = W_batch.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, *W_batch.shape[1:])
            b_rep = b_batch.unsqueeze(1).expand(-1, n_q, -1).reshape(B * n_q, -1)
            mem_inp = self._apply_linear(mem_rep, W_rep, b_rep)

        mem_offset = self.n_mem_tokens + self.n_ctrl_tokens * 2
        if self.n_ctrl_tokens > 0:
            # [B, C, d] -> [B, 1, C, d] -> [B, n_q, C, d] -> [B*n_q, C, d]
            rs = read_st_batch.reshape(B, 1, self.n_ctrl_tokens, -1).expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
            re_ = read_end_batch.reshape(B, 1, self.n_ctrl_tokens, -1).expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
            x_qry = torch.cat([rs, mem_inp, re_, qry_emb], dim=1)
        else:
            x_qry = torch.cat([mem_inp, qry_emb], dim=1)                       # [B*n_q, M+Q, d]

        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_list = [0, 0, 0, -x_qry.size(1) % 32]
            x_qry = F.pad(x_qry, pad_list, "constant", 0)

        with self._disable_write_lora():
            logits = self.model(inputs_embeds=x_qry).logits                   # [B*n_q, M+Q, V]
        logits = logits[:, mem_offset - 1:mem_offset + Q, :]                  # [B*n_q, Q+1, V]
        return logits.reshape(B, n_q, Q + 1, -1)

    @staticmethod
    def _probe_exact_match(logits, query_ids, target_mask, ignore_token_ids):
        """Per-probe exact-match flag, teacher-forced (mirrors compute_metrics_fn).

        The READ query is the teacher-forced sequence ``?!K:V!|`` (query + target
        concatenated, exactly as collate_fn builds it). logits[t] predicts token
        t+1, so the prediction AT a target position predicts the NEXT target token.
        We score only the positions flagged by ``target_mask`` (the ``V!|`` span,
        excluding structural '!'/'|' which are in ignore_token_ids): a probe is
        correct iff argmax matches the actual token id at every scored position.

        Args:
            logits: [n_q, Q+1, V] (the +1 is the next-token slot).
            query_ids: [n_q, Q] the teacher-forced query tokens (?!K:V!|).
            target_mask: [n_q, Q] bool, True at the V!| target positions.
            ignore_token_ids: token ids to skip (e.g. '!','|') -- scored on the
                remaining content tokens only.
        Returns: [n_q] bool.
        """
        preds = logits.argmax(dim=-1)[:, :-1]                       # [n_q, Q]
        n_q = logits.size(0)
        correct = torch.ones(n_q, dtype=torch.bool, device=logits.device)
        for i in range(n_q):
            tmask = target_mask[i]                                   # [Q]
            qids_i = query_ids[i]                                    # [Q]
            # exclude ignored structural tokens from scoring
            score = tmask.clone()
            for ig in ignore_token_ids:
                score &= (qids_i != ig)
            if score.sum() == 0:
                correct[i] = True                                    # nothing to score
                continue
            correct[i] = (preds[i][score] == qids_i[score]).all()
        return correct

    def forward_per_segment_eval(self, input_ids, kv_queries):
        """Eval-only: per-query retrieval results for the forgetting matrix.

        Runs the normal segmented WRITE inner loop (memory identical to training,
        via forward(collect_segment_mems=True)), and after each model-segment's K
        steps, probes retrieval of every KV pair whose source segment <= the
        just-written segment (cumulative forgetting curve).

        Returns FLAT per-probe arrays (not a [n_seg,n_seg] matrix) because a
        segment can hold multiple KV pairs -- a cell would otherwise be
        overwritten by the last query. The trainer bins these by (src, probe)
        to build the matrix.

        Args:
            input_ids: {'context_input_ids': [B,S]} (the WRITE context).
            kv_queries: {'query_input_ids': [B, n_q_max, Q],   # teacher-forced ?!K:V!|
                         'target_mask':    [B, n_q_max, Q],   # True at the V!| positions
                         'seg_idx':        [B, n_q_max],
                         'mask':           [B, n_q_max] bool}
        Returns:
            src_seg:   [n_probes] long  -- source segment of each probed KV
            probe_seg: [n_probes] long  -- segment-after-write at which it was probed
            em:        [n_probes] bool  -- exact-match flag for that probe
        """
        context_input_ids = input_ids['context_input_ids']
        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)

        qids = kv_queries['query_input_ids']            # [B, n_q, Q]  teacher-forced
        tmask = kv_queries['target_mask']               # [B, n_q, Q]
        seg_idx = kv_queries['seg_idx']                 # [B, n_q]
        qmask = kv_queries['mask']                      # [B, n_q]
        n_q_max, Q = qids.shape[1], qids.shape[2]
        ignore_token_ids = kv_queries.get('ignore_token_ids', [])

        # segment bounds (padded length, matching forward)
        seq_len_padded = context_input_ids.size(1)
        n_seg, seg_sz, _ = self._segment_bounds(seq_len_padded, self.n_segments, self.segment_size)

        src_seg_list, probe_seg_list, em_list = [], [], []

        if not self.K or not (context_input_ids != pad_id).any() or qmask.sum() == 0:
            return (torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.bool, device=device))

        # ---- get the EXACT per-segment memory states from the real forward ---- #
        # forward() requires a query_input_ids for its READ phase; pass a minimal
        # single-token placeholder -- its READ output is discarded.
        fwd_input = {'context_input_ids': context_input_ids,
                     'query_input_ids': context_input_ids[:, :1]}
        out = self.forward(fwd_input, labels=None, collect_segment_mems=True)
        segment_mems = out['segment_mems']           # list[n_segments] of [B,M,d] or None
        n_seg_actual = len(segment_mems)             # == n_segments (None placeholders keep alignment)

        if self.n_ctrl_tokens > 0:
            read_st_batch = self.read_st.unsqueeze(0).expand(B, -1, -1)
            read_end_batch = self.read_end.unsqueeze(0).expand(B, -1, -1)
        else:
            read_st_batch = read_end_batch = None

        with torch.no_grad():
            for seg_idx_loop in range(min(n_seg_actual, n_seg)):
                # use the TRUE segment index: segment_mems is indexed by seg_idx,
                # and empty segments are None (memory unchanged from the previous
                # non-empty segment). Using the true index (not a list offset)
                # keeps the probe-seg aligned with the KV's source-seg so the
                # forgetting matrix diagonal is meaningful.
                mem_det = segment_mems[seg_idx_loop]
                if mem_det is None:
                    # empty segment: no WRITE ran, memory unchanged. Still probe
                    # so KV pairs in segments <= seg_idx_loop are tested at this
                    # probe point -- use the last available snapshot.
                    mem_det = next((segment_mems[j] for j in range(seg_idx_loop, -1, -1)
                                    if segment_mems[j] is not None), None)
                    if mem_det is None:
                        continue
                probe_here = (seg_idx <= seg_idx_loop) & qmask             # [B, n_q]
                if not probe_here.any():
                    continue
                logits_q = self._read_once(mem_det, qids, read_st_batch, read_end_batch)
                # logits_q: [B, n_q, Q+1, V]
                for b in range(B):
                    active = probe_here[b]
                    if not active.any():
                        continue
                    ai = active.nonzero(as_tuple=True)[0]
                    for q_i in ai.tolist():
                        s_src = int(seg_idx[b, q_i].item())
                        em = self._probe_exact_match(
                            logits_q[b:b+1, q_i],
                            qids[b:b+1, q_i],
                            tmask[b:b+1, q_i],
                            ignore_token_ids)
                        src_seg_list.append(s_src)
                        probe_seg_list.append(seg_idx_loop)
                        em_list.append(bool(em[0].item()))

        return (torch.tensor(src_seg_list, dtype=torch.long, device=device),
                torch.tensor(probe_seg_list, dtype=torch.long, device=device),
                torch.tensor(em_list, dtype=torch.bool, device=device))

    def forward(self, input_ids, labels=None, return_mem=False, collect_segment_mems=False,
                kv_queries=None):
        # context_input_ids : B × S   (segments only, each ends with `|`)
        # query_input_ids   : B × Q   (e.g.  "?!K:V!|") i.e. the last segment
        # labels            : B × Q   (-100 everywhere except the target tokens (V!|))
        # kv_queries        : optional ?!K:V!| probe set for the STAR boundary
        #                     loss (top-level batch key, NOT inside input_ids —
        #                     the Trainer's eval input-decoding path pads
        #                     input_ids leaf tensors and rejects other types).
        #                     Ignored unless training with star_weight > 0.

        """
        All tensors already padded to the same length in the datacollator.
        """
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']

        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)
        inner_loss = torch.tensor(0.0, device=device)

        # STAR / replay regulariser state (train-only; no-op unless enabled —
        # with both weights at 0 or eval mode nothing below executes, so the
        # forward is bitwise identical to the pre-regulariser one).
        star_active = (self.training and self.star_weight > 0)
        replay_active = (self.training and self.replay_weight > 0)
        kv_queries = kv_queries if (star_active or replay_active) else None
        star_loss_sum = torch.tensor(0.0, device=device)
        star_terms = 0
        star_n_probes = 0
        star_delta_ratio = torch.tensor(0.0, device=device)
        replay_loss_sum = torch.tensor(0.0, device=device)
        replay_terms = 0
        replay_n_probes = 0
        replay_em_rate = torch.tensor(0.0, device=device)

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
                    # right-pad so real tokens keep their trailing `|` boundary at segment ends
                    ctx_emb = F.pad(ctx_emb, [0, 0, 0, pad_len], "constant", 0)
                    mask = F.pad(mask, [0, pad_len], "constant", 0)
                    lm_labels = F.pad(lm_labels, [0, pad_len], "constant", -100)

                # per-segment memory snapshots (eval-only forgetting probe): one
                # detached copy of the carried state after each segment's K steps,
                # indexed by the TRUE segment index. Empty segments (all-pad,
                # possible with left-padding) are skipped by `continue` below, so
                # we must NOT use append-order -- that would misalign snapshot[i]
                # with segment i and corrupt the forgetting matrix diagonal. We
                # store None for skipped segments so indices stay aligned.
                segment_mems = ([None] * n_segments) if collect_segment_mems else None

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
                            outs = get_backbone(self.model)(inputs_embeds=x_ctx, return_dict=True)
                            h = outs.last_hidden_state                     # [B,M+seg_size,V]
                            h = h[:, mem_offset-1:, :]                     # [B,seg_size,V]
                            logits = self.write_head(h)
                            del h
                        else:
                            outs = self.model(inputs_embeds=x_ctx, return_dict=True)
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

                    # snapshot the carried memory after this segment's K steps
                    # (eval-only forgetting probe uses these EXACT states so the
                    # measured retrieval matches what the model actually produces).
                    # Store at the TRUE seg_idx so probe alignment holds even when
                    # earlier empty segments were skipped.
                    if collect_segment_mems:
                        segment_mems[seg_idx] = mem_batch.detach()

                    # ---- STAR worst-case stability term (train-only) ------ #
                    # Fires at the boundary AFTER this segment's K steps: the
                    # probes are KV pairs already stored in the carried memory,
                    # and the worst-case perturbation stands in for the memory
                    # movement that later segments' inner-loop updates cause.
                    # Under seg_bptt, boundaries older than the window have a
                    # detached mem_batch: their meta-gradient then flows only
                    # through the perturbed probe forward (local robustness),
                    # not through the inner loop — correct, just weaker.
                    if star_active and kv_queries is not None:
                        W_arg = W_batch if self.mem_proj_mode == 'per_sample' else None
                        b_arg = b_batch if self.mem_proj_mode == 'per_sample' else None
                        res = self._star_boundary_loss(
                            mem_batch, kv_queries, seg_idx, W_arg, b_arg,
                            read_st_batch, read_end_batch)
                        if res is not None:
                            loss_b, star_n_probes, star_delta_ratio = res
                            if self.star_boundaries == "last":
                                # keep only the LAST processed boundary
                                # (trailing empty segments are skipped above)
                                star_loss_sum = loss_b
                                star_terms = 1
                            else:
                                star_loss_sum = star_loss_sum + loss_b
                                star_terms += 1

                    # ---- vanilla replay (train-only, same boundaries/probes) ---- #
                    # Corrective counterpart of the STAR term: CE of the READ
                    # outputs on the stored probes against their answer tokens.
                    if replay_active and kv_queries is not None:
                        W_arg = W_batch if self.mem_proj_mode == 'per_sample' else None
                        b_arg = b_batch if self.mem_proj_mode == 'per_sample' else None
                        res_r = self._replay_boundary_loss(
                            mem_batch, kv_queries, seg_idx, W_arg, b_arg,
                            read_st_batch, read_end_batch)
                        if res_r is not None:
                            loss_r, replay_n_probes, replay_em_rate = res_r
                            if self.replay_boundaries == "last":
                                replay_loss_sum = loss_r
                                replay_terms = 1
                            else:
                                replay_loss_sum = replay_loss_sum + loss_r
                                replay_terms += 1

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
        # VAE-style regularisation: prior on the deterministic mean, then
        # (optionally) sample a perturbed memory for READ.
        # ---------------------------------------------------------------- #
        # The prior is computed on the DETERMINISTIC inner-loop endpoint m
        # (not the noisy sample), so it constrains where the inner loop lands.
        # m_0 = mem_batch_initial is DETACHED -> the reg pulls m -> m_0 only,
        # never m_0 -> m (avoids the "prior drifts to satisfy the reg" collapse).
        # lambda is scheduled by _mem_prior_weight_now (0 -> mem_prior_weight
        # over mem_prior_anneal_steps). Default mem_prior_weight=0 -> no-op.
        mem_prior_w = self._mem_prior_weight_now()
        if self.mem_prior_weight > 0:
            mem_prior_loss = F.mse_loss(mem_batch, mem_batch_initial.detach())
        else:
            mem_prior_loss = torch.tensor(0.0, device=device)
        inner_loop_stats['mem_prior_loss'] = mem_prior_loss.detach()
        inner_loop_stats['mem_prior_weight_now'] = torch.tensor(float(mem_prior_w), device=device)

        # Reparameterised sampling: m_used = m + sigma*eps, eps ~ N(0, I). eps
        # carries no grad, so gradients flow to the mean m (and through it to
        # the inner-loop params) exactly as in a VAE; the noise is a forward-
        # pass bottleneck forcing READ-time robustness to memory perturbation.
        # Train-only: eval READ stays deterministic (keeps token_accuracy
        # stable; the per-segment forgetting probes snapshot BEFORE this point,
        # so they are unaffected regardless). Default mem_noise_std=0 -> no-op.
        if self.training and self.mem_noise_std > 0:
            noise = torch.randn_like(mem_batch)
            mem_sampled = mem_batch + self.mem_noise_std * noise
            inner_loop_stats['mem_sampled_delta_norm_mean'] = \
                (mem_sampled - mem_batch).detach().norm(dim=(1, 2)).mean()
            mem_batch = mem_sampled

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
        if collect_segment_mems:
            output['segment_mems'] = segment_mems
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
        # Additive chain so target / inner-loss / prior terms compose cleanly.
        # With both weights at their defaults (inner_loss_weight=0 when
        # add_inner_loss_to_outer is False, mem_prior_weight=0) this reduces
        # to exactly `target_loss` -- bitwise-identical to the pre-regulariser
        # forward.
        combined_loss = target_loss
        if self.add_inner_loss_to_outer:
            combined_loss = combined_loss + self.inner_loss_weight * (inner_loss / B)
        if self.mem_prior_weight > 0:
            combined_loss = combined_loss + mem_prior_w * mem_prior_loss
        if star_terms > 0:
            star_w_now = self._star_weight_now()
            star_loss = star_loss_sum / star_terms
            combined_loss = combined_loss + star_w_now * star_loss
            # stats only when the term actually fired: star-off forwards keep
            # the inner_loop_stats dict keys identical (bitwise-off guarantee)
            output['inner_loop_stats']['star_kl'] = star_loss.detach()
            output['inner_loop_stats']['star_weight_now'] = torch.tensor(
                float(star_w_now), device=device)
            output['inner_loop_stats']['star_n_probes'] = star_n_probes.detach().float()
            output['inner_loop_stats']['star_delta_ratio'] = star_delta_ratio
        if replay_terms > 0:
            replay_w_now = self._replay_weight_now()
            replay_loss = replay_loss_sum / replay_terms
            combined_loss = combined_loss + replay_w_now * replay_loss
            output['inner_loop_stats']['replay_ce'] = replay_loss.detach()
            output['inner_loop_stats']['replay_weight_now'] = torch.tensor(
                float(replay_w_now), device=device)
            output['inner_loop_stats']['replay_n_probes'] = replay_n_probes.detach().float()
            output['inner_loop_stats']['replay_em_rate'] = replay_em_rate
        output['loss'] = combined_loss
        return output
