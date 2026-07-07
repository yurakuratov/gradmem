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
    Configuration class for GradMemGPT.
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
                 use_read_lora=False,
                 read_lora_r=8,
                 read_lora_alpha=16,
                 read_lora_dropout=0.0,
                 read_lora_target_modules=None,
                 freeze_backbone=None,
                 use_gradient_checkpointing=False,
                 attn_implementation="eager",
                 add_inner_loss_to_outer=False,
                 inner_loss_weight=None,
                 use_hopfield_memory=False,
                 hopfield_n_segments=1,
                 hopfield_segment_size=None,
                 hopfield_retrieval_mode="softmax",
                 hopfield_beta_init=1.0,
                 use_separate_hopfield_mem=False,
                 hopfield_proj_dim=None,
                 hopfield_direct_query=False,
                 hopfield_value_as_key=False,
                 hopfield_value_proj_dim=None,
                 hopfield_bptt_segments=None,
                 memory_update="gradient",
                 use_mem_residual=False,
                 use_reconstruction_loss=False,
                 reconstruction_loss_weight=1.0,
                 use_energy_inner_loss=False,
                 n_energy_tokens=4,
                 energy_mlp_hidden_dim=None,
                 energy_mlp_n_layers=2,
                 energy_readout="energy_tokens",
                 energy_recon_weight=0.0,
                 energy_recon_weight_end=None,
                 energy_recon_anneal_steps=0,
                 stabilize_energy_head=True,
                 energy_out_scale=1.0,
                 use_gated_delta_memory=False,
                 gated_delta_state_dim=128,
                 gated_delta_alpha_init=0.9,
                 gated_delta_beta_init=0.5,
                 gated_delta_bptt_segments=None,
                 recon_pretrain_steps=0,
                 recon_pretrain_target_weight=0.0,
                 recon_pretrain_recon_weight=1.0,
                 use_learned_inner_update=False,
                 n_learned_update_tokens=4,
                 learned_update_mlp_hidden_dim=None,
                 learned_update_mlp_n_layers=2,
                 learned_update_treat_as_gradient=True,
                 learned_update_final_tanh=False,
                 learned_update_warmup_steps=0,
                 learned_update_normalize=False,
                 **kwargs):
        """
        Args:
            pretrained_model: str, name of pretrained model to load (e.g., 'gpt2')
            base_config: dict or PretrainedConfig, config for base model when creating from scratch
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
            use_read_lora: bool, enable LoRA adapters during READ phase only
            read_lora_r: int, LoRA rank for READ adapters
            read_lora_alpha: int, LoRA alpha for READ adapters
            read_lora_dropout: float, LoRA dropout for READ adapters
            read_lora_target_modules: list[str]|str|None, target module names (None/"auto" for defaults)
            freeze_backbone: bool|None, freeze backbone weights (READ+WRITE), except LoRA/write head/mem proj.
                None (default) auto-freezes when either write_lora or read_lora is enabled; True/False overrides.
            use_gradient_checkpointing: bool, turn on gradient checkpointing supported by HF models
            add_inner_loss_to_outer: bool, outer loss = target_loss + inner_loss_weight * inner_loss_mean
            inner_loss_weight: float, weight of inner loss in combined loss
use_hopfield_memory: bool, enable Hopfield-like external memory
             hopfield_n_segments: int, number of segments to split context into for Hopfield storage (1 = whole context)
             hopfield_segment_size: int|None, fixed segment size (tokens per segment); mutually exclusive with hopfield_n_segments
             hopfield_retrieval_mode: str, retrieval mode ("raw", "softmax", "beta_softmax")
             hopfield_beta_init: float, initial value for learnable beta parameter (only used with "beta_softmax")
             use_separate_hopfield_mem: bool, use separate initial memory tokens for key/query/value roles in Hopfield
             hopfield_proj_dim: int|None, dimension for key/query projection layers (None = no projection)
             hopfield_direct_query: bool, skip forward pass for query key, use self.mem_query directly (requires use_separate_hopfield_mem)
             hopfield_value_as_key: bool, use value pattern as key in Hopfield STORE instead of forward pass
hopfield_value_proj_dim: int|None, dimension for value projection in Hopfield STORE/RETRIEVE (None = no projection)
              hopfield_bptt_segments: int|None, number of last segments to keep in backprop graph for Hopfield (None = all segments)
              memory_update: str, how to update memory in inner loop ("gradient" for SGD/Adam, "forward" for RMT-style forward pass)
             use_mem_residual: bool, use residual connection in forward memory update: mem = mem + LN(mem_out) (requires memory_update="forward")
              use_reconstruction_loss: bool, add reconstruction loss (context LM loss) to outer loss during forward inner loop
              reconstruction_loss_weight: float, weight of reconstruction loss in combined loss
             use_energy_inner_loss: bool, replace reconstruction CE in the gradient inner loop with a learned scalar
                 "energy" read out from dedicated energy tokens at the end of the context (meta-learned via the
                 outer/second-order path). Requires memory_update="gradient", grad_mode="second".
             n_energy_tokens: int, number of learnable energy tokens appended at the end of the WRITE context
             energy_mlp_hidden_dim: int|None, hidden width of the energy MLP (None = n_embd)
             energy_mlp_n_layers: int, number of Linear layers in the energy MLP (incl. output layer; min 1)
             energy_readout: str, source of the energy MLP input - "energy_tokens" (dedicated tokens at the
                 end of the WRITE context; mem stays prefix) or "mem_tokens" (memory tokens themselves placed
                 at the end; no separate energy-token param, MLP input dim = n_mem_tokens*d)
             energy_recon_weight: float, reconstruction-CE weight added to the energy inner loss at step 0
                 (Option A only: gradient + energy_tokens). 0.0 = pure energy.
             energy_recon_weight_end: float|None, final recon weight after linear annealing; None = constant
                 (no schedule, uses energy_recon_weight throughout)
             energy_recon_anneal_steps: int, outer steps to linearly anneal energy_recon_weight ->
                 energy_recon_weight_end; 0 = no schedule
            stabilize_energy_head: bool, build a collapse-proof energy head (LayerNorm on energy_h +
                a fixed-norm linear read-out) instead of a plain MLP. Prevents the energy gradient from
                vanishing (the outer loop can only learn the read-out direction, not shrink its scale).
            energy_out_scale: float, fixed scale of the stabilized energy read-out (controls the energy's
                effective magnitude vs recon); only used when stabilize_energy_head=True
             use_gated_delta_memory: bool, enable Gated DeltaNet-style external memory (alternative to Hopfield)
             gated_delta_state_dim: int, dimension d_k = d_v of the Gated Delta state matrix S (B, d, d)
             gated_delta_alpha_init: float, initial value for the alpha (forget gate) head bias (sigmoid output)
             gated_delta_beta_init: float, initial value for the beta (write strength) head bias (sigmoid output)
             gated_delta_bptt_segments: int|None, number of last segments to keep in backprop graph for Gated Delta (None = all)
             use_learned_inner_update: bool, replace the analytic inner-loop gradient (autograd.grad of the
                 reconstruction CE) with a learned delta predicted by an MLP head from dedicated readout tokens'
                 last-layer embeddings. The head is trained END-TO-END by the outer target_loss backpropagating
                 through the K unrolled inner steps (like an RNN unrolled K times), NOT by imitating the real
                 gradient. The inner step becomes a fully-forward differentiable composition, so the meta-gradient
                 to self.mem flows at ~FOMAML cost with no backward-over-backward. The head IS the memory
                 pathway, so it needs the meta-gradient chain -> requires grad_mode='first' or 'second' (not
                 'none'). Mutually exclusive with use_energy_inner_loss. Requires memory_update="gradient".
             n_learned_update_tokens: int, number of learnable readout tokens appended at the end of the WRITE
                 context (feature width = n_learned_update_tokens * n_embd).
             learned_update_mlp_hidden_dim: int|None, hidden width of the learned-update MLP (None = n_embd).
             learned_update_mlp_n_layers: int, number of Linear layers in the learned-update MLP (min 1).
             learned_update_treat_as_gradient: bool, if True the head output is fed to _sgd_step (reuses self.lr
                 + inner_clip_*, behaves like a learned/preconditioned gradient); if False, mem = mem + delta
                 (raw delta, maximally expressive, its own scale).
             learned_update_final_tanh: bool, apply tanh to the head output to bound its magnitude.
             learned_update_warmup_steps: int, if >0, during the first N outer steps additionally add an MSE
                 imitation loss (delta vs the real reconstruction-CE gradient, both detached/grad-stopped as
                 appropriate) to bootstrap the head near the analytic gradient; after N steps this term vanishes
                 and the head is purely end-to-end. 0 = pure end-to-end from step 0.
             learned_update_normalize: bool, L2-normalize the head's delta to unit norm per sample over the
                 full M*d vector before applying the update. Removes the task-irrelevant magnitude degree of
                 freedom (GPT2's ln_1 normalizes mem-token embeddings before attention, so magnitude barely
                 affects target_loss and would otherwise drift unbounded). Per-sample step magnitude then
                 equals self.lr exactly (under learned_update_treat_as_gradient), and total displacement is
                 bounded by K*lr. When on, the imitation warmup target is also normalized so warmup matches
                 directions. Same unidentifiable-scale rationale as stabilize_energy_head.
         """
        super().__init__(**kwargs)

        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
            self.base_config = None
        else:
            self.pretrained_model = None
            self.base_config = base_config

        # GradMemGPT specific parameters
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
        self.use_read_lora = use_read_lora
        self.read_lora_r = read_lora_r
        self.read_lora_alpha = read_lora_alpha
        self.read_lora_dropout = read_lora_dropout
        self.read_lora_target_modules = read_lora_target_modules
        self.freeze_backbone = freeze_backbone
        self.last_K_second_order = K if last_K_second_order is None else last_K_second_order
        if grad_mode != "second":
            self.last_K_second_order = 0
        self.last_K_second_order = max(0, min(self.last_K_second_order, K))
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_implementation = attn_implementation
        self.add_inner_loss_to_outer = add_inner_loss_to_outer
        self.inner_loss_weight = inner_loss_weight
        self.use_hopfield_memory = use_hopfield_memory
        self.hopfield_n_segments = hopfield_n_segments
        self.hopfield_segment_size = hopfield_segment_size
        self.hopfield_retrieval_mode = hopfield_retrieval_mode
        self.hopfield_beta_init = hopfield_beta_init
        self.use_separate_hopfield_mem = use_separate_hopfield_mem
        self.hopfield_proj_dim = hopfield_proj_dim
        self.hopfield_direct_query = hopfield_direct_query
        self.hopfield_value_as_key = hopfield_value_as_key
        self.hopfield_value_proj_dim = hopfield_value_proj_dim
        self.hopfield_bptt_segments = hopfield_bptt_segments
        self.memory_update = memory_update
        self.use_mem_residual = use_mem_residual
        self.use_reconstruction_loss = use_reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.use_energy_inner_loss = use_energy_inner_loss
        self.n_energy_tokens = n_energy_tokens
        self.energy_mlp_hidden_dim = energy_mlp_hidden_dim
        self.energy_mlp_n_layers = energy_mlp_n_layers
        self.energy_readout = energy_readout
        self.energy_recon_weight = energy_recon_weight
        self.energy_recon_weight_end = energy_recon_weight_end
        self.energy_recon_anneal_steps = energy_recon_anneal_steps
        self.stabilize_energy_head = stabilize_energy_head
        self.energy_out_scale = energy_out_scale

        self.use_gated_delta_memory = use_gated_delta_memory
        self.gated_delta_state_dim = gated_delta_state_dim
        self.gated_delta_alpha_init = gated_delta_alpha_init
        self.gated_delta_beta_init = gated_delta_beta_init
        self.gated_delta_bptt_segments = gated_delta_bptt_segments

        # Reconstruction-only pretrain warmup (default-off). During the first
        # `recon_pretrain_steps` outer steps the target (READ) loss is scaled by
        # recon_pretrain_target_weight (e.g. 0 -> pure-reconstruction pretrain)
        # and the inner-loop reconstruction loss is up-weighted by
        # recon_pretrain_recon_weight. Both weights linearly ramp to their
        # normal values over recon_pretrain_steps (driven by current_train_step,
        # stamped via CustomTrainer.compute_loss -> set_train_step).
        self.recon_pretrain_steps = recon_pretrain_steps
        self.recon_pretrain_target_weight = recon_pretrain_target_weight
        self.recon_pretrain_recon_weight = recon_pretrain_recon_weight

        # Learned inner-update head (default-off). A learned MLP head predicts the
        # memory delta directly from readout-token last-layer embeddings, replacing
        # the analytic inner-loop gradient. Trained end-to-end by the outer loss
        # through the unrolled K steps (no autograd.grad in the inner loop). See
        # the docstring above for details.
        self.use_learned_inner_update = use_learned_inner_update
        self.n_learned_update_tokens = n_learned_update_tokens
        self.learned_update_mlp_hidden_dim = learned_update_mlp_hidden_dim
        self.learned_update_mlp_n_layers = learned_update_mlp_n_layers
        self.learned_update_treat_as_gradient = learned_update_treat_as_gradient
        self.learned_update_final_tanh = learned_update_final_tanh
        self.learned_update_warmup_steps = learned_update_warmup_steps
        self.learned_update_normalize = learned_update_normalize

        # Validate mem_proj_mode settings
        assert mem_proj_mode in ["none", "proj", "per_sample", "proj_rw"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"
        assert hopfield_retrieval_mode in ("raw", "softmax", "beta_softmax", "mean"), \
            f"hopfield_retrieval_mode must be 'raw', 'softmax', 'beta_softmax', or 'mean', got '{hopfield_retrieval_mode}'"
        assert not (use_separate_hopfield_mem and not use_hopfield_memory), \
            "use_separate_hopfield_mem requires use_hopfield_memory=True"
        assert not (hopfield_direct_query and not use_separate_hopfield_mem), \
            "hopfield_direct_query requires use_separate_hopfield_mem=True"
        assert not (hopfield_value_as_key and not use_hopfield_memory), \
            "hopfield_value_as_key requires use_hopfield_memory=True"
        if hopfield_value_proj_dim is not None:
            assert use_hopfield_memory, "hopfield_value_proj_dim requires use_hopfield_memory=True"
            assert hopfield_value_proj_dim > 0, f"hopfield_value_proj_dim must be positive, got {hopfield_value_proj_dim}"
        if hopfield_segment_size is not None:
            assert hopfield_n_segments == 1, \
                f"hopfield_segment_size and hopfield_n_segments are mutually exclusive, got segment_size={hopfield_segment_size} and n_segments={hopfield_n_segments}"
            assert hopfield_segment_size >= 1, \
                f"hopfield_segment_size must be >= 1, got {hopfield_segment_size}"
        if hopfield_proj_dim is not None:
            assert hopfield_proj_dim > 0, f"hopfield_proj_dim must be positive, got {hopfield_proj_dim}"
        if hopfield_bptt_segments is not None:
            assert hopfield_bptt_segments >= 1, f"hopfield_bptt_segments must be >= 1 or None, got {hopfield_bptt_segments}"
            assert use_hopfield_memory, "hopfield_bptt_segments requires use_hopfield_memory=True"
        assert memory_update in ("gradient", "forward", "forward_energy"), \
            f"memory_update must be 'gradient', 'forward', or 'forward_energy', got '{memory_update}'"
        assert not (use_mem_residual and memory_update != "forward"), \
            "use_mem_residual requires memory_update='forward'"
        assert not (use_reconstruction_loss and memory_update != "forward"), \
            "use_reconstruction_loss requires memory_update='forward'"

        # Validate energy-based inner loss settings
        assert energy_readout in ("energy_tokens", "mem_tokens"), \
            f"energy_readout must be 'energy_tokens' or 'mem_tokens', got '{energy_readout}'"
        if use_energy_inner_loss:
            assert memory_update in ("gradient", "forward_energy"), \
                "use_energy_inner_loss requires memory_update='gradient' or 'forward_energy'"
            assert grad_mode == "second", \
                "use_energy_inner_loss requires grad_mode='second' (energy head is meta-learned via the second-order path)"
            assert not add_inner_loss_to_outer, \
                "use_energy_inner_loss is incompatible with add_inner_loss_to_outer"
            assert not use_reconstruction_loss, \
                "use_energy_inner_loss is mutually exclusive with use_reconstruction_loss"
            assert energy_mlp_n_layers >= 1, \
                f"energy_mlp_n_layers must be >= 1, got {energy_mlp_n_layers}"
            if energy_readout == "energy_tokens":
                assert n_energy_tokens >= 1, \
                    f"n_energy_tokens must be >= 1, got {n_energy_tokens}"

        # Validate energy+reconstruction (Option A) settings
        energy_recon_enabled = (energy_recon_weight > 0) or (energy_recon_weight_end not in (None, 0))
        if energy_recon_enabled:
            assert use_energy_inner_loss, \
                "energy_recon_weight requires use_energy_inner_loss=True"
            assert memory_update == "gradient", \
                "energy_recon_weight requires memory_update='gradient' (recon needs mem as a prefix)"
            assert energy_readout == "energy_tokens", \
                "energy_recon_weight requires energy_readout='energy_tokens' (Option A)"
            assert energy_recon_anneal_steps >= 0, \
                f"energy_recon_anneal_steps must be >= 0, got {energy_recon_anneal_steps}"

        # Validate learned inner-update head settings
        if use_learned_inner_update:
            assert memory_update == "gradient", \
                "use_learned_inner_update requires memory_update='gradient'"
            assert grad_mode in ("first", "second"), \
                "use_learned_inner_update requires grad_mode='first' or 'second' (the head is the " \
                "memory pathway, so it needs the meta-gradient chain that grad_mode='none' detaches)"
            assert not use_energy_inner_loss, \
                "use_learned_inner_update is mutually exclusive with use_energy_inner_loss"
            assert not use_reconstruction_loss, \
                "use_learned_inner_update is mutually exclusive with use_reconstruction_loss"
            assert n_learned_update_tokens >= 1, \
                f"n_learned_update_tokens must be >= 1, got {n_learned_update_tokens}"
            assert learned_update_mlp_n_layers >= 1, \
                f"learned_update_mlp_n_layers must be >= 1, got {learned_update_mlp_n_layers}"
            assert learned_update_warmup_steps >= 0, \
                f"learned_update_warmup_steps must be >= 0, got {learned_update_warmup_steps}"

        # Validate stabilized energy head settings.
        # NOTE: stabilize_energy_head is ignored when use_energy_inner_loss=False (the energy head is
        # only built under that flag), so we do NOT assert use_energy_inner_loss here — doing so would
        # break HF's to_diff_dict()/repr(), which instantiate the config with all defaults.
        if stabilize_energy_head:
            assert energy_out_scale > 0, \
                f"energy_out_scale must be > 0, got {energy_out_scale}"

        # Validate Gated Delta settings
        assert not (use_gated_delta_memory and use_hopfield_memory), \
            "use_gated_delta_memory and use_hopfield_memory are mutually exclusive"
        if use_gated_delta_memory:
            assert gated_delta_state_dim > 0, \
                f"gated_delta_state_dim must be positive, got {gated_delta_state_dim}"
            assert 0.0 < gated_delta_alpha_init < 1.0, \
                f"gated_delta_alpha_init must be in (0, 1), got {gated_delta_alpha_init}"
            assert 0.0 < gated_delta_beta_init < 1.0, \
                f"gated_delta_beta_init must be in (0, 1), got {gated_delta_beta_init}"
        if gated_delta_bptt_segments is not None:
            assert gated_delta_bptt_segments >= 1, \
                f"gated_delta_bptt_segments must be >= 1 or None, got {gated_delta_bptt_segments}"
            assert use_gated_delta_memory, "gated_delta_bptt_segments requires use_gated_delta_memory=True"

        # Validate LoRA settings
        if use_write_lora:
            assert write_lora_r > 0, f"write_lora_r must be positive, got {write_lora_r}"
            assert write_lora_alpha > 0, f"write_lora_alpha must be positive, got {write_lora_alpha}"
        if use_read_lora:
            assert read_lora_r > 0, f"read_lora_r must be positive, got {read_lora_r}"
            assert read_lora_alpha > 0, f"read_lora_alpha must be positive, got {read_lora_alpha}"
        assert freeze_backbone in (None, True, False), \
            f"freeze_backbone must be None/True/False, got {freeze_backbone}"


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

        # write-phase LoRA (applies only during inner loop), additional params to train for WRITE operation
        self.use_write_lora = getattr(config, "use_write_lora", False)
        self.write_lora_r = getattr(config, "write_lora_r", 8)
        self.write_lora_alpha = getattr(config, "write_lora_alpha", 16)
        self.write_lora_dropout = getattr(config, "write_lora_dropout", 0.0)
        self.write_lora_target_modules = getattr(config, "write_lora_target_modules", None)
        # read-phase LoRA (applies only during READ phase), decouples READ from WRITE backbone updates
        self.use_read_lora = getattr(config, "use_read_lora", False)
        self.read_lora_r = getattr(config, "read_lora_r", 8)
        self.read_lora_alpha = getattr(config, "read_lora_alpha", 16)
        self.read_lora_dropout = getattr(config, "read_lora_dropout", 0.0)
        self.read_lora_target_modules = getattr(config, "read_lora_target_modules", None)
        # freeze_backbone: None => auto-freeze when either LoRA is enabled
        freeze_backbone_cfg = getattr(config, "freeze_backbone", None)
        if freeze_backbone_cfg is None:
            self.freeze_backbone = bool(self.use_write_lora or self.use_read_lora)
        else:
            self.freeze_backbone = bool(freeze_backbone_cfg)
        # adapter name bookkeeping (filled in by _init_phase_lora)
        self.write_adapter_name = None
        self.read_adapter_name = None
        if self.use_write_lora or self.use_read_lora:
            self._init_phase_lora()
        if self.freeze_backbone:
            self._freeze_backbone_params()

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
        self.use_write_head = config.use_write_head
        self.add_inner_loss_to_outer = config.add_inner_loss_to_outer
        self.inner_loss_weight = config.inner_loss_weight
        if self.add_inner_loss_to_outer:
            if self.inner_loss_weight is None:
                self.inner_loss_weight = 1.0
        else:
            self.inner_loss_weight = 0.0

        self.memory_update = config.memory_update
        self.use_mem_residual = config.use_mem_residual
        self.use_reconstruction_loss = config.use_reconstruction_loss
        self.reconstruction_loss_weight = config.reconstruction_loss_weight

        self.use_energy_inner_loss = config.use_energy_inner_loss
        self.n_energy_tokens = config.n_energy_tokens
        self.energy_mlp_hidden_dim = config.energy_mlp_hidden_dim
        self.energy_mlp_n_layers = config.energy_mlp_n_layers
        self.energy_readout = config.energy_readout
        self.energy_recon_weight = config.energy_recon_weight
        self.energy_recon_weight_end = config.energy_recon_weight_end
        self.energy_recon_anneal_steps = config.energy_recon_anneal_steps
        self.stabilize_energy_head = config.stabilize_energy_head
        self.energy_out_scale = config.energy_out_scale

        # Learned inner-update head config (see GradMemGPTConfig docstring).
        self.use_learned_inner_update = config.use_learned_inner_update
        self.n_learned_update_tokens = config.n_learned_update_tokens
        self.learned_update_mlp_hidden_dim = config.learned_update_mlp_hidden_dim
        self.learned_update_mlp_n_layers = config.learned_update_mlp_n_layers
        self.learned_update_treat_as_gradient = config.learned_update_treat_as_gradient
        self.learned_update_final_tanh = config.learned_update_final_tanh
        self.learned_update_warmup_steps = config.learned_update_warmup_steps
        self.learned_update_normalize = config.learned_update_normalize

        # current outer-loop step, stamped by the trainer before forward (for the recon schedule)
        self.current_train_step = 0

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
            if self.mem_proj_mode == "proj_rw":
                self.read_mem_proj = nn.Linear(n_embd, n_embd, bias=True)
                with torch.no_grad():
                    nn.init.eye_(self.read_mem_proj.weight)
                    self.read_mem_proj.bias.zero_()
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

        if self.use_mem_residual:
            self.mem_residual_ln = nn.LayerNorm(n_embd)

        # Learnable energy-based inner-loop objective (replaces reconstruction CE).
        # Energy read-out source depends on mode/readout:
        #   memory_update="forward_energy": energy read from mem_out (RMT forward output);
        #      no separate energy-token param; MLP input dim = n_mem_tokens * d.
        #   energy_readout="energy_tokens" (gradient mode): dedicated energy tokens appended at the
        #      END of the WRITE context (mem stays as prefix); MLP input dim = n_energy_tokens * d.
        #   energy_readout="mem_tokens" (gradient mode): memory tokens themselves placed at the END;
        #      no separate energy-token param; MLP input dim = n_mem_tokens * d.
        # The energy MLP -> scalar is minimised over mem_batch in the inner loop, and is meta-learned
        # by the outer loop only (second-order path).
        if self.use_energy_inner_loss:
            if self.memory_update == "forward_energy":
                in_dim = self.n_mem_tokens * n_embd
            elif self.energy_readout == "energy_tokens":
                self.energy_tokens = nn.Parameter(torch.randn(self.n_energy_tokens, n_embd) * 0.02)
                in_dim = self.n_energy_tokens * n_embd
            else:  # "mem_tokens"
                in_dim = self.n_mem_tokens * n_embd
            self.energy_in_dim = in_dim
            if self.stabilize_energy_head:
                # Collapse-proof energy head: LayerNorm on energy_h + a fixed-norm linear read-out.
                # No hidden weights -> the outer loop can only learn the read-out direction, not shrink
                # the energy's scale, so the inner-loop gradient cannot vanish.
                self.energy_norm = nn.LayerNorm(in_dim)
                self.energy_dir = nn.Parameter(torch.randn(in_dim) * 0.02)
                self.energy_bias = nn.Parameter(torch.zeros(1))
                self.energy_mlp = None
            else:
                hid = self.energy_mlp_hidden_dim if self.energy_mlp_hidden_dim is not None else n_embd
                energy_layers = []
                for i in range(self.energy_mlp_n_layers):
                    layer_in = in_dim if i == 0 else hid
                    layer_out = 1 if i == self.energy_mlp_n_layers - 1 else hid
                    energy_layers.append(nn.Linear(layer_in, layer_out))
                    if i < self.energy_mlp_n_layers - 1:
                        energy_layers.append(nn.GELU())
                self.energy_mlp = nn.Sequential(*energy_layers)

        # Learned inner-update head: predicts the memory delta directly from
        # readout-token last-layer embeddings, replacing the analytic inner-loop
        # gradient. Layout mirrors energy Option A: [ctrl?, mem, ctrl?, seg, readout].
        # The head is trained end-to-end by the outer loss through the unrolled K
        # steps (no autograd.grad in the inner loop), so the meta-gradient to
        # self.mem flows forward-only at ~FOMAML cost.
        if self.use_learned_inner_update:
            self.learned_update_tokens = nn.Parameter(
                torch.randn(self.n_learned_update_tokens, n_embd) * 0.02)
            in_dim = self.n_learned_update_tokens * n_embd
            self.learned_update_in_dim = in_dim
            out_dim = self.n_mem_tokens * n_embd
            hid = self.learned_update_mlp_hidden_dim if self.learned_update_mlp_hidden_dim is not None else n_embd
            update_layers = []
            for i in range(self.learned_update_mlp_n_layers):
                layer_in = in_dim if i == 0 else hid
                layer_out = out_dim if i == self.learned_update_mlp_n_layers - 1 else hid
                update_layers.append(nn.Linear(layer_in, layer_out))
                if i < self.learned_update_mlp_n_layers - 1:
                    update_layers.append(nn.GELU())
            if self.learned_update_final_tanh:
                update_layers.append(nn.Tanh())
            self.learned_update_head = nn.Sequential(*update_layers)

        # Hopfield-like external memory
        self.use_hopfield_memory = getattr(config, "use_hopfield_memory", False)
        self.hopfield_n_segments = getattr(config, "hopfield_n_segments", 1)
        self.hopfield_segment_size = getattr(config, "hopfield_segment_size", None)
        self.hopfield_retrieval_mode = getattr(config, "hopfield_retrieval_mode", "softmax")
        self.hopfield_beta_init = getattr(config, "hopfield_beta_init", 1.0)
        self.use_separate_hopfield_mem = getattr(config, "use_separate_hopfield_mem", False)
        self.hopfield_proj_dim = getattr(config, "hopfield_proj_dim", None)
        self.hopfield_direct_query = getattr(config, "hopfield_direct_query", False)
        self.hopfield_value_as_key = getattr(config, "hopfield_value_as_key", False)
        self.hopfield_value_proj_dim = getattr(config, "hopfield_value_proj_dim", None)
        self.hopfield_bptt_segments = getattr(config, "hopfield_bptt_segments", None)

        if self.use_hopfield_memory:
            if self.hopfield_n_segments < 1:
                raise ValueError(f"hopfield_n_segments must be >= 1, got {self.hopfield_n_segments}")
            if self.hopfield_retrieval_mode == "beta_softmax":
                self.hopfield_beta = nn.Parameter(torch.tensor(self.hopfield_beta_init))
            if self.use_separate_hopfield_mem:
                self.mem_key = nn.Parameter(torch.randn(self.n_mem_tokens, n_embd) * 0.02)
                self.mem_query = nn.Parameter(torch.randn(self.n_mem_tokens, n_embd) * 0.02)
            if self.hopfield_proj_dim is not None:
                pattern_dim = self.n_mem_tokens * n_embd
                self.hopfield_key_proj = nn.Linear(pattern_dim, self.hopfield_proj_dim, bias=True)
                self.hopfield_query_proj = nn.Linear(pattern_dim, self.hopfield_proj_dim, bias=True)
                if self.hopfield_proj_dim == pattern_dim:
                    with torch.no_grad():
                        nn.init.eye_(self.hopfield_key_proj.weight)
                        self.hopfield_key_proj.bias.zero_()
                        nn.init.eye_(self.hopfield_query_proj.weight)
                        self.hopfield_query_proj.bias.zero_()

            if self.hopfield_value_proj_dim is not None:
                pattern_dim = self.n_mem_tokens * n_embd
                self.hopfield_value_proj = nn.Linear(pattern_dim, self.hopfield_value_proj_dim, bias=True)
                self.hopfield_value_inv_proj = nn.Linear(self.hopfield_value_proj_dim, pattern_dim, bias=True)
                if self.hopfield_value_proj_dim == pattern_dim:
                    with torch.no_grad():
                        nn.init.eye_(self.hopfield_value_proj.weight)
                        self.hopfield_value_proj.bias.zero_()
                        nn.init.eye_(self.hopfield_value_inv_proj.weight)
                        self.hopfield_value_inv_proj.bias.zero_()

        if self.use_hopfield_memory and self.hopfield_segment_size is not None:
            max_pos = getattr(self.model.config, 'n_positions',
                              getattr(self.model.config, 'max_position_embeddings', None))
            max_seq_in_segment = self.hopfield_segment_size + self.n_mem_tokens + self.n_ctrl_tokens * 2
            if max_pos is not None and max_seq_in_segment > max_pos:
                raise ValueError(
                    f"Segment + memory tokens ({max_seq_in_segment}) exceeds model's max position embeddings "
                    f"({max_pos}). Reduce hopfield_segment_size or increase max_position_embeddings."
                )

        # Gated DeltaNet-style external memory (alternative to Hopfield)
        self.use_gated_delta_memory = getattr(config, "use_gated_delta_memory", False)
        self.gated_delta_state_dim = getattr(config, "gated_delta_state_dim", 128)
        self.gated_delta_alpha_init = getattr(config, "gated_delta_alpha_init", 0.9)
        self.gated_delta_beta_init = getattr(config, "gated_delta_beta_init", 0.5)
        self.gated_delta_bptt_segments = getattr(config, "gated_delta_bptt_segments", None)

        # Reconstruction-only pretrain warmup (default-off)
        self.recon_pretrain_steps = getattr(config, "recon_pretrain_steps", 0)
        self.recon_pretrain_target_weight = getattr(config, "recon_pretrain_target_weight", 0.0)
        self.recon_pretrain_recon_weight = getattr(config, "recon_pretrain_recon_weight", 1.0)

        if self.use_gated_delta_memory:
            pattern_dim = self.n_mem_tokens * n_embd  # M*d
            d = self.gated_delta_state_dim
            # Projections between pattern space (M*d) and state space (d)
            self.gd_key_proj = nn.Linear(pattern_dim, d, bias=True)
            self.gd_query_proj = nn.Linear(pattern_dim, d, bias=True)
            self.gd_value_proj = nn.Linear(pattern_dim, d, bias=True)
            self.gd_value_inv_proj = nn.Linear(d, pattern_dim, bias=True)
            # Per-segment alpha/beta heads: features -> scalar in (0, 1)
            self.gd_alpha_head = nn.Linear(pattern_dim, 1, bias=True)
            self.gd_beta_head = nn.Linear(pattern_dim, 1, bias=True)
            # Init biases so initial sigmoid(bias) ~= alpha_init / beta_init
            with torch.no_grad():
                a_init = self.gated_delta_alpha_init
                b_init = self.gated_delta_beta_init
                self.gd_alpha_head.bias.fill_(math.log(a_init / (1.0 - a_init)))
                self.gd_beta_head.bias.fill_(math.log(b_init / (1.0 - b_init)))

        # Segment-size check also applies to Gated Delta (reuses the same segmenting logic)
        if self.use_gated_delta_memory and self.hopfield_segment_size is not None:
            max_pos = getattr(self.model.config, 'n_positions',
                              getattr(self.model.config, 'max_position_embeddings', None))
            max_seq_in_segment = self.hopfield_segment_size + self.n_mem_tokens + self.n_ctrl_tokens * 2
            if max_pos is not None and max_seq_in_segment > max_pos:
                raise ValueError(
                    f"Segment + memory tokens ({max_seq_in_segment}) exceeds model's max position embeddings "
                    f"({max_pos}). Reduce hopfield_segment_size or increase max_position_embeddings."
                )

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

    def floating_point_ops(self, inputs):
        # dummy method to satisfy base class and it's invocation by trainer:
        # Trainer supposes that `inputs`` is a tensor, not dict.
        return 0

    def set_train_step(self, step):
        # Called by the trainer (compute_loss) so the recon-weight schedule knows the outer step.
        self.current_train_step = int(step)

    def _energy_recon_weight_now(self):
        # Linear schedule: energy_recon_weight -> energy_recon_weight_end over energy_recon_anneal_steps.
        start = self.energy_recon_weight
        if self.energy_recon_weight_end is None or self.energy_recon_anneal_steps <= 0:
            return start
        frac = min(self.current_train_step / self.energy_recon_anneal_steps, 1.0)
        return start + (self.energy_recon_weight_end - start) * frac

    def _energy_forward(self, energy_h):
        # energy_h: [B, in_dim] -> energy per sample [B]
        if self.stabilize_energy_head:
            h = self.energy_norm(energy_h)
            dir_n = self.energy_dir / (self.energy_dir.norm() + 1e-8)   # unit-norm read-out direction
            return (h * dir_n).sum(-1) * self.energy_out_scale + self.energy_bias.squeeze(0)
        return self.energy_mlp(energy_h).squeeze(-1)

    def _recon_pretrain_target_weight_now(self):
        # Target-loss weight schedule for the reconstruction-only pretrain warmup.
        # = recon_pretrain_target_weight at step 0, linearly ramping to 1.0 over
        # recon_pretrain_steps, then constant at 1.0. When recon_pretrain_steps
        # <= 0 the schedule is disabled (always returns 1.0).
        if self.recon_pretrain_steps <= 0:
            return 1.0
        frac = min(self.current_train_step / self.recon_pretrain_steps, 1.0)
        return self.recon_pretrain_target_weight + (1.0 - self.recon_pretrain_target_weight) * frac

    def _recon_pretrain_recon_weight_now(self):
        # Reconstruction-loss weight during the pretrain warmup:
        # = recon_pretrain_recon_weight at step 0, linearly ramping to 0.0 over
        # recon_pretrain_steps, then constant at 0.0 (i.e. after pretrain the
        # normal inner-loss term, gated by add_inner_loss_to_outer, takes over).
        if self.recon_pretrain_steps <= 0:
            return 0.0
        frac = min(self.current_train_step / self.recon_pretrain_steps, 1.0)
        return self.recon_pretrain_recon_weight * (1.0 - frac)

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

    def _resolve_lora_targets(self, value, adapter_name):
        parsed = self._parse_lora_targets(value)
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
            f"{adapter_name}_lora_target_modules is not set and model_type is unknown. "
            "Please provide explicit target modules."
        )

    def _init_phase_lora(self):
        """Set up phase-specific LoRA adapters on the backbone.

        WRITE adapter is registered as PEFT's "default" adapter; READ adapter
        (if enabled) is added as a second adapter named "read". The two adapter
        sets are disjoint and toggled via set_adapter / enable/disable_adapter_layers.
        """
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise ImportError("peft is required for write_lora/read_lora. Please install the peft package.")

        write_config = None
        read_config = None
        if self.use_write_lora:
            write_config = LoraConfig(
                r=self.write_lora_r,
                lora_alpha=self.write_lora_alpha,
                lora_dropout=self.write_lora_dropout,
                target_modules=self._resolve_lora_targets(self.write_lora_target_modules, "write"),
                task_type=TaskType.CAUSAL_LM,
            )
        if self.use_read_lora:
            read_config = LoraConfig(
                r=self.read_lora_r,
                lora_alpha=self.read_lora_alpha,
                lora_dropout=self.read_lora_dropout,
                target_modules=self._resolve_lora_targets(self.read_lora_target_modules, "read"),
                task_type=TaskType.CAUSAL_LM,
            )

        # The first adapter registered via get_peft_model becomes "default".
        # Register the WRITE adapter first when both are enabled so that
        # "default" = write (matches the legacy write-only naming convention).
        if write_config is not None:
            self.model = get_peft_model(self.model, write_config)
            self.write_adapter_name = "default"
            if read_config is not None:
                self.model.add_adapter("read", read_config)
                self.read_adapter_name = "read"
        else:
            # read-only: the single adapter is "default"
            self.model = get_peft_model(self.model, read_config)
            self.read_adapter_name = "default"
            self.write_adapter_name = None

        if not (hasattr(self.model, "disable_adapter_layers") and hasattr(self.model, "enable_adapter_layers")):
            raise RuntimeError("PEFT model does not support adapter toggling; cannot enforce phase-only LoRA.")
        if self.use_write_lora and self.use_read_lora and not hasattr(self.model, "set_adapter"):
            raise RuntimeError("PEFT model does not support set_adapter; cannot toggle write/read adapters.")

        # Default active adapter = WRITE (forward()'s inner loop runs WRITE first);
        # for read-only, the single adapter is left active and disabled during WRITE.
        if self.write_adapter_name is not None:
            self.model.set_adapter(self.write_adapter_name)
        else:
            self.model.set_adapter(self.read_adapter_name)

        # Legacy escape hatch: keep base model trainable when not auto-freezing.
        # When self.freeze_backbone is True, _freeze_backbone_params() handles it next.
        if not self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = True

    def _freeze_backbone_params(self):
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        if self.use_write_lora or self.use_read_lora:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def _set_phase_adapter(self, phase):
        """Activate the LoRA adapter for the given phase ('write' or 'read').

        Uses set_adapter (both-adapters) or enable/disable_adapter_layers
        (single-adapter) to select which LoRA set is applied in the forward pass.
        Afterwards, re-enables requires_grad on ALL LoRA params: set_adapter and
        disable_adapter_layers set non-active adapters' params to requires_grad=False,
        which would block second-order MAML gradients during backward (the WRITE
        adapter receives grad through the inner-loop graph, the READ adapter through
        the READ-forward graph — both must stay trainable at backward time).
        """
        if not (self.use_write_lora or self.use_read_lora):
            return
        both = self.use_write_lora and self.use_read_lora
        if both:
            name = self.write_adapter_name if phase == "write" else self.read_adapter_name
            self.model.set_adapter(name)
        elif self.use_write_lora:
            if phase == "write":
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()
        else:  # read-only
            if phase == "read":
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()
        # Restore requires_grad on every LoRA param so backward can populate .grad
        # for both adapter sets through their respective computation graphs.
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    @contextmanager
    def _disable_write_lora(self):
        """Legacy context manager: disable WRITE LoRA during the READ phase.
        Kept for backwards compatibility; new code uses _set_phase_adapter('read')."""
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

    def _compress_segment(self, seg_emb, seg_mask, seg_has_tokens, mem_prefix,
                          write_st_batch, write_end_batch, B, device):
        """Causal forward pass compressing a segment into M mem-token features.

        Places mem_prefix at the END so it can attend to all preceding content.
        Returns last_hidden_state[:, -n_mem_tokens:, :].view(B, -1) -> [B, M*d].

        Shared by Hopfield STORE (key) and Gated Delta WRITE (key + alpha/beta).
        """
        hopf_mem_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
        seg_mask_for_attn = seg_mask.long()
        if not seg_has_tokens.all():
            seg_mask_for_attn = seg_mask_for_attn.clone()
            seg_mask_for_attn[~seg_has_tokens] = 1
        if self.n_ctrl_tokens > 0:
            hopf_ctrl_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
            x_seg_comp = torch.cat([write_st_batch, seg_emb, write_end_batch, mem_prefix], dim=1)
            seg_attn_mask = torch.cat([hopf_ctrl_mask, seg_mask_for_attn, hopf_ctrl_mask, hopf_mem_mask], dim=1)
        else:
            x_seg_comp = torch.cat([seg_emb, mem_prefix], dim=1)
            seg_attn_mask = torch.cat([seg_mask_for_attn, hopf_mem_mask], dim=1)

        position_ids = seg_attn_mask.cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)
        outs_seg = get_backbone(self.model)(inputs_embeds=x_seg_comp, attention_mask=seg_attn_mask,
                                            position_ids=position_ids, return_dict=True)
        features = outs_seg.last_hidden_state[:, -self.n_mem_tokens:, :].view(B, -1)  # [B, M*d]
        del outs_seg
        return features

    def _compress_query(self, qry_emb, query_input_ids, labels, pad_id, mem_prefix,
                        read_st_batch, read_end_batch, B, device):
        """Causal forward pass compressing the query into M mem-token features.

        Shared by Hopfield RETRIEVE (query key) and Gated Delta RETRIEVE (query).
        Returns [B, M*d].
        """
        qry_mask = (query_input_ids != pad_id).to(dtype=torch.long)
        if labels is not None:
            qry_mask[labels >= 0] = 0
        hopf_mem_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
        if self.n_ctrl_tokens > 0:
            hopf_ctrl_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
            x_qry_comp = torch.cat([read_st_batch, qry_emb, read_end_batch, mem_prefix], dim=1)
            qry_attn_mask = torch.cat([hopf_ctrl_mask, qry_mask, hopf_ctrl_mask, hopf_mem_mask], dim=1)
        else:
            x_qry_comp = torch.cat([qry_emb, mem_prefix], dim=1)
            qry_attn_mask = torch.cat([qry_mask, hopf_mem_mask], dim=1)

        position_ids = qry_attn_mask.cumsum(-1) - 1
        outs_q = get_backbone(self.model)(inputs_embeds=x_qry_comp, attention_mask=qry_attn_mask,
                                          position_ids=position_ids, return_dict=True)
        features = outs_q.last_hidden_state[:, -self.n_mem_tokens:, :].view(B, -1)  # [B, M*d]
        del outs_q
        return features

    def forward(self, input_ids, labels=None, return_mem=False):
        # context_input_ids : B × S   (segments only, each ends with `|`)
        # query_input_ids   : B × Q   (e.g.  "?!K:V!|") i.e. the last segment
        # labels            : B × Q   (‑100 everywhere except the target tokens (V!|))

        """
        All tensors already padded to the same length in the datacollator.
        """
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']

        pad_id = self.model.config.pad_token_id
        device = context_input_ids.device
        B = context_input_ids.size(0)

        # actual model inputs starts after mem tokens and ctrl tokens
        mem_offset = self.n_mem_tokens + self.n_ctrl_tokens * 2

        # ctrl tokens
        if self.n_ctrl_tokens > 0:
            write_st_batch = self.write_st.unsqueeze(0).expand(B, -1, -1)
            write_end_batch = self.write_end.unsqueeze(0).expand(B, -1, -1)
            read_st_batch = self.read_st.unsqueeze(0).expand(B, -1, -1)
            read_end_batch = self.read_end.unsqueeze(0).expand(B, -1, -1)
        else:
            write_st_batch = write_end_batch = read_st_batch = read_end_batch = None

        # mem_batch_initial is always self.mem — used for Hopfield keys and as reset point
        mem_batch_initial = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]

        # per-sample params for mem_proj (initialized from outer loop params, reset per segment)
        if self.mem_proj_mode == "per_sample":
            W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
            b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()

        n_segments = self.hopfield_n_segments  # default; overridden inside inner loop if hopfield_segment_size is set

        # Hopfield memory storage (fresh each forward call)
        if self.use_hopfield_memory:
            stored_keys = []
            stored_values = []
            stored_masks = []
            hopfield_stored = torch.zeros(B, dtype=torch.bool, device=device)

        # Gated Delta state matrix (fresh each forward call, S_0 = 0)
        if self.use_gated_delta_memory:
            d = self.gated_delta_state_dim
            # dtype follows the context embeddings; fall back to float32 if no context
            S_dtype = torch.float32
            S = torch.zeros(B, d, d, device=device, dtype=S_dtype)
            gd_written = torch.zeros(B, dtype=torch.bool, device=device)
            gd_alpha_sum = torch.tensor(0.0, device=device)
            gd_beta_sum = torch.tensor(0.0, device=device)
            gd_n_written = 0

        total_inner_loss_detached = torch.tensor(0.0, device=device)
        last_segment_inner_loss = None
        total_inner_steps = 0
        n_segments_with_context = 0

        # Per-segment inner-loss diagnostics: track the batch-mean inner loss of
        # each processed segment to report min/mean/max across segments. A low
        # min (≈ facts-only loss) under noise means some segments reconstruct
        # well -> the bottleneck is retrieval/READ, not the WRITE pathway.
        seg_inner_losses = []  # list of per-segment batch-mean (detached) scalars

        inner_loop_stats = {'inner_grad_norm_mean': torch.tensor(0.0, device=device),
                            'inner_grad_norm_max': torch.tensor(-1.0, device=device),
                            'inner_grad_norm_min': torch.tensor(1e06, device=device)}
        if self.use_learned_inner_update:
            inner_loop_stats['learned_update_delta_norm_mean'] = torch.tensor(0.0, device=device)
            inner_loop_stats['learned_update_delta_norm_max'] = torch.tensor(-1.0, device=device)
            inner_loop_stats['learned_update_delta_norm_min'] = torch.tensor(1e06, device=device)
            inner_loop_stats['_n_learned_update_steps'] = 0
            if self.learned_update_warmup_steps > 0:
                inner_loop_stats['learned_update_target_grad_norm_mean'] = torch.tensor(0.0, device=device)
                inner_loop_stats['learned_update_target_grad_norm_max'] = torch.tensor(-1.0, device=device)
                inner_loop_stats['_n_learned_warmup_steps'] = 0

        seg_nonempty_counts = torch.zeros(B, dtype=torch.long, device=device)
        seg_nonempty_sizes = torch.zeros(B, dtype=torch.long, device=device)

        rec_losses = []
        learned_update_imitation_losses = []  # MSE(delta, g_real) terms during learned-update warmup

        # energy+recon (Option A) accumulators
        energy_recon_detached = torch.tensor(0.0, device=device)
        n_energy_recon_steps = 0
        energy_recon_weight_now = None

        # Track last segment's mem_batch for stats and as fallback for READ phase
        last_mem_batch = mem_batch_initial

        # ---------------------------------------------------------------- #
        # 1.  INNER loop on context. WRITE context to mem, segment by segment.
        # ---------------------------------------------------------------- #
        ctx_emb = None
        if self.K and context_input_ids.ne(pad_id).any():
            # re‑enable autograd even if outer context is `no_grad`
            with torch.enable_grad():
                # activate WRITE-phase LoRA (or disable READ adapter for read-only configs)
                self._set_phase_adapter("write")
                # build ctx embedding once
                ctx_emb = self.model.get_input_embeddings()(context_input_ids)      # [B,S,d]
                # lm labels: reconstructing the context, last mem/ctrl token predicts the first token of the context
                lm_labels = context_input_ids.clone()
                lm_labels[lm_labels == pad_id] = -100
                # loss mask
                mask = (lm_labels != -100)

                # Split context into segments (ceil-padded to equal size)
                seq_len = ctx_emb.size(1)
                if (self.use_hopfield_memory or self.use_gated_delta_memory) and self.hopfield_segment_size is not None:
                    n_segments = math.ceil(seq_len / self.hopfield_segment_size)
                    segment_size = self.hopfield_segment_size
                else:
                    n_segments = self.hopfield_n_segments
                    segment_size = (seq_len + n_segments - 1) // n_segments  # ceil division
                pad_len = segment_size * n_segments - seq_len

                if pad_len > 0:
                    ctx_emb = F.pad(ctx_emb, [0, 0, pad_len, 0], "constant", 0)
                    mask = F.pad(mask, [pad_len, 0], "constant", 0)
                    lm_labels = F.pad(lm_labels, [pad_len, 0], "constant", -100)

                for seg_idx in range(n_segments):
                    seg_start = seg_idx * segment_size
                    seg_end = seg_start + segment_size
                    seg_emb = ctx_emb[:, seg_start:seg_end, :]
                    seg_mask = mask[:, seg_start:seg_end]
                    seg_labels = lm_labels[:, seg_start:seg_end]

                    # Per-sample: does this segment have any real tokens?
                    seg_has_tokens = seg_mask.any(dim=1)  # [B]

                    seg_nonempty_counts += seg_has_tokens.long()
                    seg_nonempty_sizes += (seg_mask.sum(dim=1) * seg_has_tokens.long())

                    if not seg_has_tokens.any():
                        continue

                    seg_seq_len = seg_mask.sum(dim=1).clamp_min(1)  # [B]

                    if self.memory_update == "gradient":
                        # ---- GRADIENT mode inner loop (existing) ----
                        # Pad segment for JVP Flash Attention compatibility
                        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
                            _extra = self.n_energy_tokens if (self.use_energy_inner_loss
                                                              and self.energy_readout == "energy_tokens") else 0
                            if self.use_learned_inner_update:
                                _extra += self.n_learned_update_tokens
                            _fixed_len = mem_offset + _extra
                            seg_pad_len = -(seg_emb.size(1) + _fixed_len) % 32
                            seg_pad_list = [0, seg_pad_len, 0, 0]
                            cur_seg_mask = F.pad(seg_mask, seg_pad_list, "constant", 0)
                            cur_seg_labels = F.pad(seg_labels, seg_pad_list, "constant", -100)
                            cur_seg_emb = F.pad(seg_emb, [0, 0] + seg_pad_list, "constant", 0)
                        else:
                            cur_seg_emb = seg_emb
                            cur_seg_mask = seg_mask
                            cur_seg_labels = seg_labels

                        # Build attention mask: all ones for mem/ctrl, segment mask for context.
                        # With energy_readout="mem_tokens" the memory tokens sit at the END (so they
                        # attend to the segment); otherwise mem is a prefix and (energy Option A)
                        # energy tokens are appended at the end.
                        cur_mem_attn_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
                        if self.n_ctrl_tokens > 0:
                            cur_ctrl_attn_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
                        if self.use_energy_inner_loss and self.energy_readout == "mem_tokens":
                            if self.n_ctrl_tokens > 0:
                                cur_attn_mask = torch.cat([cur_ctrl_attn_mask, cur_seg_mask.long(),
                                                           cur_ctrl_attn_mask, cur_mem_attn_mask], dim=1)
                            else:
                                cur_attn_mask = torch.cat([cur_seg_mask.long(), cur_mem_attn_mask], dim=1)
                        else:
                            if self.n_ctrl_tokens > 0:
                                cur_attn_mask = torch.cat([cur_ctrl_attn_mask, cur_mem_attn_mask,
                                                           cur_ctrl_attn_mask, cur_seg_mask.long()], dim=1)
                            else:
                                cur_attn_mask = torch.cat([cur_mem_attn_mask, cur_seg_mask.long()], dim=1)
                            if self.use_energy_inner_loss:  # Option A: energy tokens at the end
                                cur_energy_attn_mask = torch.ones(B, self.n_energy_tokens, dtype=torch.long, device=device)
                                cur_attn_mask = torch.cat([cur_attn_mask, cur_energy_attn_mask], dim=1)
                            if self.use_learned_inner_update:  # readout tokens at the end
                                cur_readout_attn_mask = torch.ones(B, self.n_learned_update_tokens,
                                                                   dtype=torch.long, device=device)
                                cur_attn_mask = torch.cat([cur_attn_mask, cur_readout_attn_mask], dim=1)
                        # clamp to 0: masked (pad) positions preceding the first real token would
                        # otherwise yield position_id = -1 (cumsum-1), which is out of range for the
                        # position-embedding gather. Real tokens keep their natural positions; clamped
                        # positions are hidden by attention_mask=0, so their position_id is irrelevant.
                        cur_position_ids = (cur_attn_mask.cumsum(-1) - 1).clamp(min=0)

                        # Reset mem_batch to initial for each segment
                        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]

                        # Reset per-sample params for each segment
                        if self.mem_proj_mode == "per_sample":
                            W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
                            b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()

                        # handling gradients for meta-params:
                        if self.grad_mode == "none":
                            mem_batch = mem_batch.detach().requires_grad_(True)
                            if self.mem_proj_mode == "per_sample":
                                W_batch = W_batch.detach().requires_grad_(True)
                                b_batch = b_batch.detach().requires_grad_(True)
                        else:
                            mem_batch = mem_batch.requires_grad_(True)
                            if self.mem_proj_mode == "per_sample":
                                W_batch = W_batch.requires_grad_(True)
                                b_batch = W_batch.requires_grad_(True)

                        # Reset Adam state per segment
                        opt_state = {}

                        for k in range(self.K):
                            if self.mem_proj_mode == 'none':
                                mem_inp = mem_batch
                            elif self.mem_proj_mode in ('proj', 'proj_rw'):
                                mem_inp = self.mem_proj(mem_batch)
                            else:  # per-sample
                                mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

                            if self.use_learned_inner_update:
                                # Learned inner-update branch: an MLP head predicts the memory delta
                                # directly from readout-token last-layer embeddings. No autograd.grad
                                # is used to update mem_batch; the update is a fully-forward
                                # differentiable composition, so the outer loss backpropagates through
                                # the K unrolled steps to self.mem and the head at ~FOMAML cost.
                                # Layout mirrors energy Option A: [ctrl?, mem, ctrl?, seg, readout].
                                readout_tok = self.learned_update_tokens.unsqueeze(0).expand(B, -1, -1)
                                if self.n_ctrl_tokens > 0:
                                    x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch,
                                                       cur_seg_emb, readout_tok], dim=1)
                                else:
                                    x_ctx = torch.cat([mem_inp, cur_seg_emb, readout_tok], dim=1)
                                outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=cur_attn_mask,
                                                                position_ids=cur_position_ids, return_dict=True)
                                h = outs.last_hidden_state
                                # Readout features -> predicted delta (carries graph to mem/head).
                                phi = h[:, -self.n_learned_update_tokens:, :].reshape(B, -1)   # [B, in_dim]
                                delta = self.learned_update_head(phi)                           # [B, M*d]
                                delta = delta.reshape(B, self.n_mem_tokens, -1)                 # [B, M, d]

                                # Remove the task-irrelevant magnitude DOF: per-sample unit-norm delta.
                                # Differentiable: the Jacobian (I - d̂ d̂ᵀ)/‖delta‖ projects the radial
                                # component out of the head's gradient too, forcing the head to learn only
                                # the direction that affects the task (GPT2's ln_1 normalizes mem-token
                                # embeddings before attention, so magnitude barely affects target_loss and
                                # would otherwise drift unbounded). Applied once here so the stat, the
                                # warmup MSE, and the update step all see the same normalized delta. Per-
                                # sample step magnitude then equals self.lr exactly (inner_clip_norm is a
                                # harmless no-op once ‖delta‖=1).
                                if self.learned_update_normalize:
                                    reduce_dims = tuple(range(1, delta.ndim))   # per-sample over M·d
                                    delta = delta / (delta.norm(dim=reduce_dims, keepdim=True) + 1e-8)

                                # Reconstruction CE on the segment (mem is a prefix, so the segment
                                # attends to mem -> recon has a graph to mem_batch). Reused for both
                                # the detached stat inner_loss and the optional imitation warmup.
                                h_seg = h[:, mem_offset - 1:-self.n_learned_update_tokens, :]
                                if self.use_write_head:
                                    rec_logits = self.write_head(h_seg)
                                else:
                                    rec_logits = self.model.get_output_embeddings()(h_seg)
                                recon = nn.functional.cross_entropy(
                                    rec_logits[:, :-1].reshape(-1, rec_logits.size(-1)),
                                    cur_seg_labels.reshape(-1),
                                    ignore_index=-100, reduction='none',
                                ).view(B, -1)
                                recon = (recon * cur_seg_mask).sum(1) / seg_seq_len
                                recon = recon.sum()                  # scalar, graph-connected to mem_batch
                                inner_loss = recon.detach()          # stat only (no grad used here)
                                del outs, h, h_seg, rec_logits

                                # track delta norm (analogous to inner_grad_norm)
                                delta_norm = delta.reshape(B, -1).norm(dim=1).detach()
                                inner_loop_stats['learned_update_delta_norm_mean'] += delta_norm.mean()
                                inner_loop_stats['learned_update_delta_norm_max'] = max(
                                    inner_loop_stats['learned_update_delta_norm_max'], delta_norm.max())
                                inner_loop_stats['learned_update_delta_norm_min'] = min(
                                    inner_loop_stats['learned_update_delta_norm_min'], delta_norm.min())
                                inner_loop_stats['_n_learned_update_steps'] += 1

                                # Optional imitation warmup: regress delta toward the real
                                # reconstruction-CE gradient (detached target). Only during the first
                                # learned_update_warmup_steps outer steps; collected into
                                # learned_update_imitation_losses and added to the outer loss. After
                                # warmup this term vanishes and the head is purely end-to-end.
                                # NOTE: retain_graph=True is required because `recon` and `delta` share
                                # the same backbone forward graph; freeing it here would break the
                                # later loss.backward() through delta.
                                if self.learned_update_warmup_steps > 0 and \
                                        self.current_train_step < self.learned_update_warmup_steps:
                                    g_real = torch.autograd.grad(recon, mem_batch,
                                                                create_graph=False, retain_graph=True)[0].detach()
                                    if self.learned_update_normalize:
                                        # Magnitude is declared task-irrelevant: match directions only.
                                        _rd = tuple(range(1, g_real.ndim))   # per-sample over M·d
                                        g_real = g_real / (g_real.norm(dim=_rd, keepdim=True) + 1e-8)
                                    learned_update_imitation_losses.append(
                                        nn.functional.mse_loss(delta, g_real))
                                    # log the norm of the imitated target gradient, so warmup dynamics
                                    # are diagnosable: if ‖g_real‖ itself -> 0 (without normalization),
                                    # the head is anchored to a vanishing target (Story A), not failing to
                                    # fit one (Story B). Under normalization this reads ~1.0 by construction.
                                    g_real_norm = g_real.reshape(B, -1).norm(dim=1)
                                    inner_loop_stats['learned_update_target_grad_norm_mean'] += g_real_norm.mean()
                                    inner_loop_stats['learned_update_target_grad_norm_max'] = max(
                                        inner_loop_stats['learned_update_target_grad_norm_max'], g_real_norm.max())
                                    inner_loop_stats['_n_learned_warmup_steps'] += 1
                                    del g_real, g_real_norm

                                # apply the learned update (no autograd.grad, forward-connected)
                                if self.learned_update_treat_as_gradient:
                                    mem_batch = self._sgd_step(mem_batch, delta,
                                                               clip_value=self.inner_clip_value,
                                                               clip_norm=self.inner_clip_norm)
                                else:
                                    mem_batch = mem_batch + delta

                                # per-step detach in "none" mode preserves semantics (only head trains)
                                if self.grad_mode in ['none']:
                                    mem_batch = mem_batch.detach().requires_grad_(True)
                                    if self.mem_proj_mode == "per_sample":
                                        W_batch = W_batch.detach().requires_grad_(True)
                                        b_batch = b_batch.detach().requires_grad_(True)

                                total_inner_steps += 1
                                continue  # skip the autograd.grad + SGD step below

                            if self.use_energy_inner_loss:
                                if self.energy_readout == "mem_tokens":
                                    # Option B: memory tokens at the END read out the segment (causal);
                                    # their last-layer hidden states feed the energy MLP. mem is both the
                                    # optimised variable and the read-out (no separate energy tokens).
                                    if self.n_ctrl_tokens > 0:
                                        x_ctx = torch.cat([write_st_batch, cur_seg_emb, write_end_batch,
                                                           mem_inp], dim=1)
                                    else:
                                        x_ctx = torch.cat([cur_seg_emb, mem_inp], dim=1)
                                    outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=cur_attn_mask,
                                                                    position_ids=cur_position_ids, return_dict=True)
                                    energy_h = outs.last_hidden_state[:, -self.n_mem_tokens:, :].reshape(B, -1)
                                    inner_loss = self._energy_forward(energy_h).sum()
                                    del outs
                                else:
                                    # Option A: energy tokens at the END read out mem's effect on the segment.
                                    energy_tok = self.energy_tokens.unsqueeze(0).expand(B, -1, -1)
                                    if self.n_ctrl_tokens > 0:
                                        x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch,
                                                           cur_seg_emb, energy_tok], dim=1)
                                    else:
                                        x_ctx = torch.cat([mem_inp, cur_seg_emb, energy_tok], dim=1)
                                    outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=cur_attn_mask,
                                                                    position_ids=cur_position_ids, return_dict=True)
                                    h = outs.last_hidden_state
                                    energy_h = h[:, -self.n_energy_tokens:, :].reshape(B, -1)
                                    energy = self._energy_forward(energy_h).sum()
                                    recon_w = self._energy_recon_weight_now()
                                    energy_recon_weight_now = recon_w
                                    if recon_w > 0:
                                        # Reconstruction CE on the segment positions (mem is a prefix, so the
                                        # segment attends to mem -> recon has a gradient w.r.t. mem_batch).
                                        # Slice excludes the energy tokens appended at the end.
                                        h_seg = h[:, mem_offset - 1:-self.n_energy_tokens, :]
                                        if self.use_write_head:
                                            rec_logits = self.write_head(h_seg)
                                        else:
                                            rec_logits = self.model.get_output_embeddings()(h_seg)
                                        recon = nn.functional.cross_entropy(
                                            rec_logits[:, :-1].reshape(-1, rec_logits.size(-1)),
                                            cur_seg_labels.reshape(-1),
                                            ignore_index=-100,
                                            reduction='none',
                                        ).view(B, -1)
                                        recon = (recon * cur_seg_mask).sum(1) / seg_seq_len
                                        recon = recon.sum()
                                        inner_loss = energy + recon_w * recon
                                        energy_recon_detached = energy_recon_detached + recon.detach()
                                        n_energy_recon_steps += 1
                                    else:
                                        inner_loss = energy
                                    del outs
                            else:
                                if self.n_ctrl_tokens > 0:
                                    x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch, cur_seg_emb], dim=1)
                                else:
                                    x_ctx = torch.cat([mem_inp, cur_seg_emb], dim=1)    # [B,M+seg_size,d]

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

                                inner_loss = nn.functional.cross_entropy(
                                    logits[:, :-1].reshape(-1, logits.size(-1)),
                                    cur_seg_labels.reshape(-1),
                                    ignore_index=-100,
                                    reduction='none',
                                ).view(B, -1)
                                inner_loss = (inner_loss * cur_seg_mask).sum(1) / seg_seq_len
                                inner_loss = inner_loss.sum()
                                del outs, logits

                            total_inner_steps += 1

                            is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
                            create_graph = is_second_order_step
                            # Only retain graph for add_inner_loss_to_outer on the last K-step;
                            # earlier segments' inner_loss will be detached, so their graphs are waste
                            # but retaining them is safe and simplifies the logic
                            retain_graph = create_graph or (self.add_inner_loss_to_outer and (k == self.K - 1))

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

                            if self.use_adam:
                                mem_batch = self._adam_step(mem_batch, g_mem, opt_state.setdefault('mem', {}), k + 1, self.lr)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = self._adam_step(W_batch, g_W, opt_state.setdefault('W', {}), k + 1, self.lr)
                                    b_batch = self._adam_step(b_batch, g_b, opt_state.setdefault('b', {}), k + 1, self.lr)
                                    raise NotImplementedError("Adam is not tested, be careful!")
                            else:
                                mem_batch = self._sgd_step(mem_batch, g_mem,
                                                           clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = self._sgd_step(W_batch, g_W,
                                                             clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
                                    b_batch = self._sgd_step(b_batch, g_b,
                                                             clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)

                            if self.grad_mode in ['none']:
                                mem_batch = mem_batch.detach().requires_grad_(True)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = W_batch.detach().requires_grad_(True)
                                    b_batch = b_batch.detach().requires_grad_(True)
                            elif self.grad_mode in ['first', 'second']:
                                pass  # do nothing, keep gradients flow

                        # Accumulate inner loss for stats (detached) and for combined loss (graph-connected, last segment only)
                        total_inner_loss_detached = total_inner_loss_detached + inner_loss.detach()
                        last_segment_inner_loss = inner_loss
                        n_segments_with_context += 1
                        # Per-segment diagnostic: batch-mean inner loss (last K-step).
                        seg_inner_losses.append((inner_loss.detach() / B))

                    elif self.memory_update == "forward_energy":
                        # RMT forward write + energy-gradient refinement of the INPUT mem_batch.
                        # The forward pass ([mem,seg,mem]) produces mem_out, which always encodes the
                        # segment (so READ always has useful input -> no deadlock); the energy head
                        # refines the input mem_batch via the second-order path. READ uses mem_out.
                        # Graceful fallback: if the energy refinement is unhelpful the outer loop
                        # flattens the energy landscape -> input delta -> 0 -> pure RMT write.
                        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]
                        mem_batch_init_seg = mem_batch.detach()  # snapshot for input-refinement stat

                        # Reset per-sample params for each segment
                        if self.mem_proj_mode == "per_sample":
                            W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
                            b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()

                        # handling gradients for meta-params:
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

                        opt_state = {}

                        # RMT layout attention mask (mem at both ends) + position ids, built once
                        fe_mem_attn_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
                        if self.n_ctrl_tokens > 0:
                            fe_ctrl_attn_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
                            fe_attn_mask = torch.cat([fe_ctrl_attn_mask, fe_mem_attn_mask, fe_ctrl_attn_mask,
                                                      seg_mask.long(), fe_ctrl_attn_mask, fe_mem_attn_mask], dim=1)
                        else:
                            fe_attn_mask = torch.cat([fe_mem_attn_mask, seg_mask.long(), fe_mem_attn_mask], dim=1)
                        fe_position_ids = (fe_attn_mask.cumsum(-1) - 1).clamp(min=0)

                        for k in range(self.K):
                            if self.mem_proj_mode == 'none':
                                mem_inp = mem_batch
                            elif self.mem_proj_mode in ('proj', 'proj_rw'):
                                mem_inp = self.mem_proj(mem_batch)
                            else:  # per_sample
                                mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

                            if self.n_ctrl_tokens > 0:
                                x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch,
                                                   seg_emb, write_st_batch, mem_inp], dim=1)
                            else:
                                x_ctx = torch.cat([mem_inp, seg_emb, mem_inp], dim=1)
                            outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=fe_attn_mask,
                                                            position_ids=fe_position_ids, return_dict=True)
                            mem_out = outs.last_hidden_state[:, -self.n_mem_tokens:, :]
                            energy_h = mem_out.reshape(B, -1)
                            inner_loss = self._energy_forward(energy_h).sum()
                            del outs

                            total_inner_steps += 1
                            is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
                            create_graph = is_second_order_step
                            retain_graph = create_graph

                            if self.mem_proj_mode == 'per_sample':
                                g_mem, g_W, g_b = torch.autograd.grad(inner_loss, [mem_batch, W_batch, b_batch],
                                                                      create_graph=create_graph, retain_graph=retain_graph)
                            else:
                                g_mem = torch.autograd.grad(inner_loss, mem_batch,
                                                            create_graph=create_graph, retain_graph=retain_graph)[0]

                            g_norm = g_mem.reshape(B, -1).norm(dim=1).detach()
                            inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
                            inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
                            inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())

                            if self.use_adam:
                                mem_batch = self._adam_step(mem_batch, g_mem, opt_state.setdefault('mem', {}), k + 1, self.lr)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = self._adam_step(W_batch, g_W, opt_state.setdefault('W', {}), k + 1, self.lr)
                                    b_batch = self._adam_step(b_batch, g_b, opt_state.setdefault('b', {}), k + 1, self.lr)
                                    raise NotImplementedError("Adam is not tested, be careful!")
                            else:
                                mem_batch = self._sgd_step(mem_batch, g_mem,
                                                           clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = self._sgd_step(W_batch, g_W,
                                                              clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
                                    b_batch = self._sgd_step(b_batch, g_b,
                                                              clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)

                            if self.grad_mode == 'none':
                                mem_batch = mem_batch.detach().requires_grad_(True)
                                if self.mem_proj_mode == 'per_sample':
                                    W_batch = W_batch.detach().requires_grad_(True)
                                    b_batch = b_batch.detach().requires_grad_(True)

                        # Final forward with the refined mem_batch -> mem_out for READ
                        if self.mem_proj_mode == 'none':
                            mem_inp = mem_batch
                        elif self.mem_proj_mode in ('proj', 'proj_rw'):
                            mem_inp = self.mem_proj(mem_batch)
                        else:  # per_sample
                            mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)
                        if self.n_ctrl_tokens > 0:
                            x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch,
                                               seg_emb, write_st_batch, mem_inp], dim=1)
                        else:
                            x_ctx = torch.cat([mem_inp, seg_emb, mem_inp], dim=1)
                        outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=fe_attn_mask,
                                                        position_ids=fe_position_ids, return_dict=True)
                        mem_out_read = outs.last_hidden_state[:, -self.n_mem_tokens:, :]
                        del outs

                        total_inner_loss_detached = total_inner_loss_detached + inner_loss.detach()
                        last_segment_inner_loss = inner_loss
                        seg_inner_losses.append((inner_loss.detach() / B))

                        # input-refinement stat: how far the energy head moved the input
                        # (-> 0 signals "fell back to pure RMT")
                        input_delta = (mem_batch.detach() - mem_batch_init_seg).norm(dim=(1, 2))
                        if 'energy_input_delta_mem_norm_mean' not in inner_loop_stats:
                            inner_loop_stats['energy_input_delta_mem_norm_mean'] = torch.tensor(0.0, device=device)
                            inner_loop_stats['energy_input_delta_mem_norm_max'] = torch.tensor(-1.0, device=device)
                            inner_loop_stats['energy_input_delta_mem_norm_min'] = torch.tensor(1e06, device=device)
                            inner_loop_stats['_n_fe_segs'] = 0
                        inner_loop_stats['energy_input_delta_mem_norm_mean'] += input_delta.mean()
                        inner_loop_stats['energy_input_delta_mem_norm_max'] = max(inner_loop_stats['energy_input_delta_mem_norm_max'], input_delta.max())
                        inner_loop_stats['energy_input_delta_mem_norm_min'] = min(inner_loop_stats['energy_input_delta_mem_norm_min'], input_delta.min())
                        inner_loop_stats['_n_fe_segs'] += 1

                        n_segments_with_context += 1
                        # downstream Hopfield STORE / last_mem_batch must use the READ memory
                        mem_batch = mem_out_read

                    else:  # "forward" – RMT-style forward pass memory update
                        # Reset mem_batch to initial for each segment
                        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]

                        # Reset per-sample params for each segment
                        if self.mem_proj_mode == "per_sample":
                            W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
                            b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()

                        for k in range(self.K):
                            # Apply memory projection for write phase
                            if self.mem_proj_mode == 'none':
                                mem_inp = mem_batch
                            elif self.mem_proj_mode in ('proj', 'proj_rw'):
                                mem_inp = self.mem_proj(mem_batch)
                            else:  # per_sample
                                mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

                            # Build input: [ctrl, mem, ctrl, seg, ctrl, mem] or [mem, seg, mem]
                            # Memory tokens at both start and end; updated memory extracted from end
                            if self.n_ctrl_tokens > 0:
                                x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch,
                                                  seg_emb, write_st_batch, mem_inp], dim=1)
                                ctrl_attn_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
                                mem_attn_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
                                attn_mask = torch.cat([ctrl_attn_mask, mem_attn_mask, ctrl_attn_mask,
                                                      seg_mask.long(), ctrl_attn_mask, mem_attn_mask], dim=1)
                            else:
                                x_ctx = torch.cat([mem_inp, seg_emb, mem_inp], dim=1)
                                mem_attn_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)
                                attn_mask = torch.cat([mem_attn_mask, seg_mask.long(), mem_attn_mask], dim=1)

                            position_ids = attn_mask.cumsum(-1) - 1

                            # Forward pass through backbone (write_lora enabled during WRITE phase)
                            outs = get_backbone(self.model)(inputs_embeds=x_ctx, attention_mask=attn_mask,
                                                            position_ids=position_ids, return_dict=True)
                            h = outs.last_hidden_state

                            # Extract updated memory from last n_mem_tokens
                            mem_out = h[:, -self.n_mem_tokens:]

                            # Optional reconstruction loss (skip first iteration – memory is uninitialised)
                            if self.use_reconstruction_loss and k > 0:
                                logits_st_pos = self.n_mem_tokens + self.n_ctrl_tokens * 2 - 1
                                logits_end_pos = -(self.n_mem_tokens + self.n_ctrl_tokens) if self.n_ctrl_tokens > 0 else -self.n_mem_tokens

                                if self.use_write_head:
                                    rec_logits = self.write_head(h[:, logits_st_pos:logits_end_pos, :])
                                else:
                                    rec_logits = self.model.get_output_embeddings()(h[:, logits_st_pos:logits_end_pos, :])

                                rec_loss = nn.functional.cross_entropy(
                                    rec_logits[:, :-1].reshape(-1, rec_logits.size(-1)),
                                    seg_labels.reshape(-1),
                                    ignore_index=-100,
                                    reduction='mean',
                                )
                                rec_losses.append(rec_loss)

                            # Track per-step delta mem norm (skip first step)
                            if k > 0:
                                step_delta = (mem_out - mem_batch).norm(dim=(1, 2)).detach()
                                if 'step_delta_mem_norm_mean' not in inner_loop_stats:
                                    inner_loop_stats['step_delta_mem_norm_mean'] = torch.tensor(0.0, device=device)
                                    inner_loop_stats['step_delta_mem_norm_max'] = torch.tensor(-1.0, device=device)
                                    inner_loop_stats['step_delta_mem_norm_min'] = torch.tensor(1e06, device=device)
                                    inner_loop_stats['_n_forward_steps'] = 0
                                inner_loop_stats['step_delta_mem_norm_mean'] += step_delta.mean()
                                inner_loop_stats['step_delta_mem_norm_max'] = max(inner_loop_stats['step_delta_mem_norm_max'], step_delta.max())
                                inner_loop_stats['step_delta_mem_norm_min'] = min(inner_loop_stats['step_delta_mem_norm_min'], step_delta.min())
                                inner_loop_stats['_n_forward_steps'] += 1

                            # Apply mem_residual if configured (skip first step)
                            if self.use_mem_residual and k > 0:
                                mem_batch = mem_batch + self.mem_residual_ln(mem_out)
                            else:
                                mem_batch = mem_out

                            # Detach if grad_mode == "none" to prevent gradient flow through inner loop
                            if self.grad_mode == "none":
                                mem_batch = mem_batch.detach()

                            del h, outs

                        total_inner_steps += self.K
                        n_segments_with_context += 1

                    # Per-segment Hopfield STORE
                    if self.use_hopfield_memory and seg_has_tokens.any():
                        # Value: gradient-connected mem_batch after inner loop on this segment
                        value_pattern = mem_batch.view(B, -1)  # [B, M*d]

                        if self.hopfield_value_proj_dim is not None:
                            value_pattern = self.hopfield_value_proj(value_pattern)

                        if self.hopfield_value_as_key:
                            # Use value pattern directly as key (no forward pass)
                            seg_key = value_pattern
                        else:
                            # Key: causal forward-pass to compress segment into mem tokens
                            # mem_key at END so it can attend to all preceding content tokens
                            if self.use_separate_hopfield_mem:
                                mem_key_prefix = self.mem_key.unsqueeze(0).expand(B, -1, -1)
                            else:
                                mem_key_prefix = mem_batch_initial
                            seg_key = self._compress_segment(seg_emb, seg_mask, seg_has_tokens, mem_key_prefix,
                                                              write_st_batch, write_end_batch, B, device)

                        # Apply key projection if configured
                        if self.hopfield_proj_dim is not None:
                            seg_key = self.hopfield_key_proj(seg_key)

                        # L2-normalize keys only (values keep their magnitude)
                        seg_key = F.normalize(seg_key, dim=-1)

                        # Detach keys/values for segments outside the BPTT window
                        # to reduce peak memory by cutting the computation graph
                        if self.hopfield_bptt_segments is not None and (n_segments - seg_idx - 1) >= self.hopfield_bptt_segments:
                            seg_key = seg_key.detach()
                            value_pattern = value_pattern.detach()

                        # Store for attention-based retrieval
                        stored_keys.append(seg_key)
                        stored_values.append(value_pattern)
                        stored_masks.append(seg_has_tokens)
                        hopfield_stored = hopfield_stored | seg_has_tokens

                    # Per-segment Gated Delta WRITE (state recurrence)
                    if self.use_gated_delta_memory and seg_has_tokens.any():
                        # Compress segment into features using mem_batch_initial as prefix
                        seg_features = self._compress_segment(seg_emb, seg_mask, seg_has_tokens, mem_batch_initial,
                                                               write_st_batch, write_end_batch, B, device)

                        # Key, alpha, beta from segment features
                        k_t = F.normalize(self.gd_key_proj(seg_features), dim=-1)            # [B, d]
                        alpha_t = torch.sigmoid(self.gd_alpha_head(seg_features)).squeeze(-1)  # [B]
                        beta_t = torch.sigmoid(self.gd_beta_head(seg_features)).squeeze(-1)   # [B]

                        # Value from gradient-updated mem_batch
                        v_t = self.gd_value_proj(mem_batch.view(B, -1))                     # [B, d]

                        # BPTT detach for segments outside the window
                        if self.gated_delta_bptt_segments is not None and \
                                (n_segments - seg_idx - 1) >= self.gated_delta_bptt_segments:
                            S = S.detach()

                        # Gated delta rule: S = S @ (alpha * (I - beta * k k^T)) + beta * v k^T
                        d = self.gated_delta_state_dim
                        I = torch.eye(d, device=device, dtype=S.dtype)
                        kkt = torch.bmm(k_t.unsqueeze(2), k_t.unsqueeze(1))               # [B, d, d]
                        a = alpha_t.view(B, 1, 1)
                        b = beta_t.view(B, 1, 1)
                        transition = a * (I - b * kkt)                                     # [B, d, d]
                        vk = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))                # [B, d, d]
                        S = torch.bmm(S, transition) + b * vk

                        gd_written = gd_written | seg_has_tokens
                        gd_alpha_sum = gd_alpha_sum + alpha_t[seg_has_tokens].sum().detach()
                        gd_beta_sum = gd_beta_sum + beta_t[seg_has_tokens].sum().detach()
                        gd_n_written += seg_has_tokens.sum().item()

                    # Keep last segment's mem_batch for stats and READ phase fallback
                    last_mem_batch = mem_batch

        if total_inner_steps > 0:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / total_inner_steps
            if n_segments_with_context > 0:
                inner_loop_stats['inner_loss'] = (total_inner_loss_detached / n_segments_with_context) / B
                if self.use_energy_inner_loss:
                    inner_loop_stats['energy'] = inner_loop_stats['inner_loss']
        # Per-segment inner-loss diagnostics (batch-mean per segment).
        # inner_loss_seg_min: the best-reconstructing segment's loss. Under
        # noise, a value near the facts-only loss (~0.6) means the WRITE pathway
        # is fine and the bottleneck is downstream (retrieval/READ); a value
        # near the mean (~4) means even facts fail to reconstruct.
        if seg_inner_losses:
            seg_losses_stack = torch.stack(seg_inner_losses)
            inner_loop_stats['inner_loss_seg_min'] = seg_losses_stack.min().detach()
            inner_loop_stats['inner_loss_seg_max'] = seg_losses_stack.max().detach()
        if n_energy_recon_steps > 0:
            inner_loop_stats['energy_recon'] = (energy_recon_detached / n_energy_recon_steps) / B
        if energy_recon_weight_now is not None:
            inner_loop_stats['energy_recon_weight'] = torch.tensor(float(energy_recon_weight_now), device=device)
        if '_n_forward_steps' in inner_loop_stats and inner_loop_stats['_n_forward_steps'] > 0:
            inner_loop_stats['step_delta_mem_norm_mean'] /= inner_loop_stats['_n_forward_steps']
            del inner_loop_stats['_n_forward_steps']
        if '_n_fe_segs' in inner_loop_stats and inner_loop_stats['_n_fe_segs'] > 0:
            inner_loop_stats['energy_input_delta_mem_norm_mean'] /= inner_loop_stats['_n_fe_segs']
            del inner_loop_stats['_n_fe_segs']
        if '_n_learned_update_steps' in inner_loop_stats and inner_loop_stats['_n_learned_update_steps'] > 0:
            inner_loop_stats['learned_update_delta_norm_mean'] /= inner_loop_stats['_n_learned_update_steps']
            del inner_loop_stats['_n_learned_update_steps']
        if '_n_learned_warmup_steps' in inner_loop_stats and inner_loop_stats['_n_learned_warmup_steps'] > 0:
            inner_loop_stats['learned_update_target_grad_norm_mean'] /= inner_loop_stats['_n_learned_warmup_steps']
            del inner_loop_stats['_n_learned_warmup_steps']
        if learned_update_imitation_losses:
            inner_loop_stats['learned_update_imitation_loss'] = \
                torch.stack(learned_update_imitation_losses).mean().detach()
        if rec_losses:
            inner_loop_stats['rec_loss'] = torch.stack(rec_losses).mean().detach()
        # mem stats from last segment
        mem_norm = last_mem_batch.norm(dim=(1, 2)).detach()  # B
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()
        inner_loop_stats['mem_norm_min'] = mem_norm.min()
        detla_mem_norm = (last_mem_batch - mem_batch_initial).detach().norm(dim=(1, 2))
        inner_loop_stats['delta_mem_norm_mean'] = detla_mem_norm.mean()
        inner_loop_stats['delta_mem_norm_max'] = detla_mem_norm.max()
        inner_loop_stats['delta_mem_norm_min'] = detla_mem_norm.min()

        inner_loop_stats['seg_nonempty_count_mean'] = seg_nonempty_counts.float().mean().detach()
        inner_loop_stats['seg_nonempty_count_max'] = seg_nonempty_counts.max().detach()
        inner_loop_stats['seg_nonempty_count_min'] = seg_nonempty_counts.min().detach()
        has_any_seg = seg_nonempty_counts > 0
        if has_any_seg.any():
            per_sample_avg_seg_size = seg_nonempty_sizes[has_any_seg].float() / seg_nonempty_counts[has_any_seg].float()
            inner_loop_stats['seg_nonempty_size_mean'] = per_sample_avg_seg_size.mean().detach()
            inner_loop_stats['seg_nonempty_size_max'] = per_sample_avg_seg_size.max().detach()
            inner_loop_stats['seg_nonempty_size_min'] = per_sample_avg_seg_size.min().detach()
        else:
            inner_loop_stats['seg_nonempty_size_mean'] = torch.tensor(0.0, device=device)
            inner_loop_stats['seg_nonempty_size_max'] = torch.tensor(0.0, device=device)
            inner_loop_stats['seg_nonempty_size_min'] = torch.tensor(0.0, device=device)

        qry_emb = self.model.get_input_embeddings()(query_input_ids)          # [B,Q,d]
        # ---------------------------------------------------------------- #
        # Hopfield RETRIEVE phase (if enabled)
        # ---------------------------------------------------------------- #
        if self.use_hopfield_memory and self.K and hopfield_stored.any():
            # Stack stored keys and values for attention-based retrieval
            keys = torch.stack(stored_keys, dim=1)    # [B, n_stored, M*d]
            values = torch.stack(stored_values, dim=1)  # [B, n_stored, M*d]
            mask = torch.stack(stored_masks, dim=1)      # [B, n_stored]

            if self.hopfield_retrieval_mode == "mean":
                # Average all stored value patterns with equal weight
                weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                retrieved_pattern = (weights.unsqueeze(2) * values).sum(dim=1)  # [B, M*d]
                probs = weights  # [B, n_stored]
            else:
                if self.hopfield_direct_query:
                    query_key = self.mem_query.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)  # [B, M*d]
                else:
                    if self.use_separate_hopfield_mem:
                        mem_query_prefix = self.mem_query.unsqueeze(0).expand(B, -1, -1)
                    else:
                        mem_query_prefix = mem_batch_initial
                    query_key = self._compress_query(qry_emb, query_input_ids, labels, pad_id,
                                                     mem_query_prefix, read_st_batch, read_end_batch,
                                                     B, device)

                if self.hopfield_proj_dim is not None:
                    query_key = self.hopfield_query_proj(query_key)

                query_key = F.normalize(query_key, dim=-1)

                scores = torch.bmm(query_key.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)  # [B, n_stored]

                if self.hopfield_retrieval_mode == "raw":
                    probs = F.softmax(scores.masked_fill(~mask.bool(), float('-inf')), dim=-1)  # [B, n_stored]
                    scores = scores * mask.float()
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values).squeeze(1)  # [B, M*d]
                elif self.hopfield_retrieval_mode == "softmax":
                    scores = scores.masked_fill(~mask.bool(), float('-inf'))
                    scores = F.softmax(scores, dim=-1)
                    probs = scores  # [B, n_stored]
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values).squeeze(1)  # [B, M*d]
                elif self.hopfield_retrieval_mode == "beta_softmax":
                    scores = scores * self.hopfield_beta
                    scores = scores.masked_fill(~mask.bool(), float('-inf'))
                    scores = F.softmax(scores, dim=-1)
                    probs = scores  # [B, n_stored]
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values).squeeze(1)  # [B, M*d]
                else:
                    raise ValueError(f"Unknown hopfield_retrieval_mode: {self.hopfield_retrieval_mode}")

            # Hopfield retrieval score distribution metrics
            with torch.no_grad():
                probs_v = probs[hopfield_stored]            # [n_valid, n_stored]
                mask_v = mask[hopfield_stored]              # [n_valid, n_stored]
                n_valid_per_sample = mask_v.sum(dim=1)      # [n_valid]
                n_stored = probs_v.size(1)

                entropy = -(probs_v * torch.log(probs_v.clamp_min(1e-12))).sum(dim=1)  # [n_valid]
                denom = torch.log(n_valid_per_sample.clamp_min(1))
                entropy_norm = torch.where(
                    n_valid_per_sample > 1,
                    entropy / denom.clamp_min(1e-12),
                    torch.zeros_like(entropy),
                )

                sorted_probs, _ = torch.sort(probs_v, dim=1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=1)
                n_seg_50 = (cum_probs < 0.5).sum(dim=1) + 1
                n_seg_95 = (cum_probs < 0.95).sum(dim=1) + 1
                n_seg_50 = n_seg_50.clamp(max=n_stored)
                n_seg_95 = n_seg_95.clamp(max=n_stored)

                inner_loop_stats['hopfield_entropy_mean'] = entropy.mean().detach()
                inner_loop_stats['hopfield_entropy_max'] = entropy.max().detach()
                inner_loop_stats['hopfield_entropy_min'] = entropy.min().detach()
                inner_loop_stats['hopfield_entropy_norm_mean'] = entropy_norm.mean().detach()
                inner_loop_stats['hopfield_entropy_norm_max'] = entropy_norm.max().detach()
                inner_loop_stats['hopfield_entropy_norm_min'] = entropy_norm.min().detach()
                inner_loop_stats['hopfield_n_seg_50_mean'] = n_seg_50.float().mean().detach()
                inner_loop_stats['hopfield_n_seg_50_max'] = n_seg_50.float().max().detach()
                inner_loop_stats['hopfield_n_seg_50_min'] = n_seg_50.float().min().detach()
                inner_loop_stats['hopfield_n_seg_95_mean'] = n_seg_95.float().mean().detach()
                inner_loop_stats['hopfield_n_seg_95_max'] = n_seg_95.float().max().detach()
                inner_loop_stats['hopfield_n_seg_95_min'] = n_seg_95.float().min().detach()

            if self.hopfield_value_proj_dim is not None:
                retrieved_pattern = self.hopfield_value_inv_proj(retrieved_pattern)

            # Handle samples with no stored segments: fall back to initial memory
            if not hopfield_stored.all():
                retrieved_pattern = torch.where(
                    hopfield_stored.unsqueeze(1),
                    retrieved_pattern,
                    mem_batch_initial.view(B, -1),
                )

            # Reshape back to individual memory tokens: [B, M, d]
            assoc_memory = retrieved_pattern.view(B, self.n_mem_tokens, -1)
            last_mem_batch = assoc_memory

        # ---------------------------------------------------------------- #
        # Gated Delta RETRIEVE phase (if enabled)
        # ---------------------------------------------------------------- #
        if self.use_gated_delta_memory and self.K and gd_written.any():
            # Cast S to query embedding dtype for the read matmul
            S = S.to(dtype=qry_emb.dtype)
            # Compress query into features using mem_batch_initial as prefix
            query_features = self._compress_query(qry_emb, query_input_ids, labels, pad_id,
                                                   mem_batch_initial, read_st_batch, read_end_batch,
                                                   B, device)
            q_t = F.normalize(self.gd_query_proj(query_features), dim=-1)  # [B, d]

            # Linear read: o = S @ q
            retrieved = torch.bmm(S, q_t.unsqueeze(2)).squeeze(2)          # [B, d]
            retrieved_pattern = self.gd_value_inv_proj(retrieved)           # [B, M*d]

            # Handle samples with no writes: fall back to initial memory
            if not gd_written.all():
                retrieved_pattern = torch.where(
                    gd_written.unsqueeze(1),
                    retrieved_pattern,
                    mem_batch_initial.view(B, -1),
                )

            assoc_memory = retrieved_pattern.view(B, self.n_mem_tokens, -1)  # [B, M, d]
            last_mem_batch = assoc_memory

            # Gated Delta stats
            with torch.no_grad():
                inner_loop_stats['gd_alpha_mean'] = (gd_alpha_sum / max(gd_n_written, 1)).detach()
                inner_loop_stats['gd_beta_mean'] = (gd_beta_sum / max(gd_n_written, 1)).detach()
                S_norm = S.flatten(1).norm(dim=1)
                inner_loop_stats['gd_S_norm_mean'] = S_norm.mean().detach()
                inner_loop_stats['gd_S_norm_max'] = S_norm.max().detach()
                inner_loop_stats['gd_S_norm_min'] = S_norm.min().detach()
                inner_loop_stats['gd_n_written_mean'] = gd_written.float().mean().detach()

        if ctx_emb is not None:
            del lm_labels

        # ---------------------------------------------------------------- #
        # 2.  READ phase – compute outer loss on target predictions based on query, read from mem
        # ---------------------------------------------------------------- #
        mem_batch = last_mem_batch

        if self.mem_proj_mode == "none":
            mem_inp = mem_batch
        elif self.mem_proj_mode == "proj":
            mem_inp = self.mem_proj(mem_batch)
        elif self.mem_proj_mode == "proj_rw":
            mem_inp = self.read_mem_proj(mem_batch)
        else:  # "per_sample"
            W_read = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1)
            b_read = self.mem_proj.bias.unsqueeze(0).expand(B, -1)
            mem_inp = self._apply_linear(mem_batch, W_read, b_read)
        read_mem_offset = mem_offset

        if self.n_ctrl_tokens > 0:
            # add params that can control read operation from mem
            x_qry = torch.cat([read_st_batch, mem_inp, read_end_batch, qry_emb], dim=1)
        else:
            x_qry = torch.cat([mem_inp, qry_emb], dim=1)                      # [B,M+Q,d]

        # pad to multiple of 32 for compatibility with JVP Flash Attention
        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_list = [0, 0, 0, -x_qry.size(1) % 32]
            x_qry = F.pad(x_qry, pad_list, "constant", 0)

        self._set_phase_adapter("read")
        logits_q = self.model(inputs_embeds=x_qry).logits                 # [B,M+Q,V]
        logits_q = logits_q[:, read_mem_offset-1:read_mem_offset+qry_emb.size(1), :]    # [B,Q+1,V]

        output = {'predictions': logits_q, 'inner_loop_stats': inner_loop_stats}
        if return_mem:
            output['mem'] = mem_batch
            if self.mem_proj_mode == "per_sample":
                W_out = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1)
                b_out = self.mem_proj.bias.unsqueeze(0).expand(B, -1)
                output['W'] = W_out
                output['b'] = b_out

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

        # Reconstruction-only pretrain warmup (default-off: both weights are
        # inert when recon_pretrain_steps <= 0 -> tw=1.0, recon_pre_w=0.0).
        # During pretrain the READ (target) loss is down-weighted and the
        # inner-loop reconstruction loss (last-segment, graph-connected) is
        # up-weighted; both ramp linearly to their normal values over
        # recon_pretrain_steps. This reuses the existing last-segment inner-loss
        # graph (see retain_graph logic above) so no extra graphs are retained.
        tw = self._recon_pretrain_target_weight_now()
        recon_pre_w = self._recon_pretrain_recon_weight_now()
        pretrain_recon_term = recon_pre_w * (last_segment_inner_loss / B) \
            if (recon_pre_w > 0 and last_segment_inner_loss is not None) else 0.0
        output['inner_loop_stats']['recon_pretrain_target_weight'] = torch.tensor(float(tw), device=device)
        output['inner_loop_stats']['recon_pretrain_recon_weight'] = torch.tensor(float(recon_pre_w), device=device)

        if self.add_inner_loss_to_outer and last_segment_inner_loss is not None:
            combined_loss = tw * target_loss + \
                (self.inner_loss_weight + recon_pre_w) * last_segment_inner_loss / B
        elif self.use_reconstruction_loss and rec_losses:
            combined_loss = tw * target_loss + \
                (self.reconstruction_loss_weight + recon_pre_w) * torch.stack(rec_losses).mean()
        elif recon_pre_w > 0 and last_segment_inner_loss is not None:
            # Pretrain active but neither add_inner_loss_to_outer nor
            # use_reconstruction_loss is on: still apply the pretrain recon term.
            combined_loss = tw * target_loss + pretrain_recon_term
        else:
            combined_loss = tw * target_loss
        # Learned inner-update imitation warmup (default-off). During the first
        # learned_update_warmup_steps outer steps, add the MSE(delta, g_real)
        # bootstrap term collected in the WRITE loop. After warmup the list is
        # empty and this is a no-op.
        if learned_update_imitation_losses:
            combined_loss = combined_loss + torch.stack(learned_update_imitation_losses).mean()
        output['loss'] = combined_loss
        return output
