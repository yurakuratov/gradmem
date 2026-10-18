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


# ---------------------------------------------------------------- #
# Shared pair-aware segmentation helpers (kv-retrieval).
#
# Module-level so BOTH grad_memgpt.py and grad_memgpt_adaptive.py (and the
# runner's collator) share one implementation: the KV->segment division must
# be identical everywhere or the forgetting-matrix attribution desyncs from
# the model's actual chunking.
# ---------------------------------------------------------------- #

def detect_pair_ends(context_input_ids, pad_id, bang_id, colon_id):
    """Per-sample ``!K:V!`` pair end positions, detected from the token ids.

    Mirrors the collator's regex ``!([^!|:]+):([^!|]+)!`` with a small state
    machine over the delimiter ids: a pair opens at ``!``, must see ``:``
    before the closing ``!`` (bangs in between are ignored, matching the
    regex's char classes for well-formed kv data). Scans only the real
    (non-pad) prefix. The LAST pair's end is extended to the end of the
    sample's real tokens so the trailing ``|`` (and any trailing noise)
    belongs to the final segment instead of being dropped.

    Returns list[B] of list[int] pair-end positions (exclusive, in token
    coords of context_input_ids). Detection runs on CPU (one sync per
    forward); the inner loop's backbone forwards dominate the cost anyway.
    """
    ids_cpu = context_input_ids.detach().to("cpu")
    all_ends = []
    for b in range(ids_cpu.size(0)):
        row = ids_cpu[b].tolist()
        real_len = len(row)
        for p, t in enumerate(row):
            if t == pad_id:
                real_len = p
                break
        ends = []
        in_pair = False
        seen_colon = False
        for p in range(real_len):
            t = row[p]
            if t == bang_id:
                if not in_pair:
                    in_pair, seen_colon = True, False
                elif seen_colon:
                    ends.append(p + 1)        # closing '!' included
                    in_pair = False
            elif t == colon_id and in_pair and not seen_colon:
                seen_colon = True
        if ends:
            ends[-1] = real_len               # trailing '|' / noise -> last segment
        all_ends.append(ends)
    return all_ends


def divide_pairs(n_pairs, pairs_per_segment, n_groups_cfg):
    """Balanced pair->group sizes; the SAME logic must run collator-side.

    pairs_per_segment mode: group g = pairs [g*P, (g+1)*P) -> ceil(n/P)
    groups. Count mode: n_groups_cfg groups; the first n % n_groups_cfg
    groups get one extra pair (8 pairs over 4 groups -> 2/2/2/2; 8 over 3
    -> 3/3/2). Returns the list of group sizes (may contain zeros when
    n_pairs < n_groups_cfg).
    """
    if pairs_per_segment is not None:
        n_groups = (n_pairs + pairs_per_segment - 1) // pairs_per_segment
        return [min(pairs_per_segment, n_pairs - g * pairs_per_segment)
                for g in range(n_groups)]
    base, rem = divmod(n_pairs, n_groups_cfg)
    return [base + (1 if g < rem else 0) for g in range(n_groups_cfg)]


def pair_segment_spans(context_input_ids, pad_id, delim_ids, pairs_per_segment, n_groups_cfg):
    """Pair-aware chunking: per-segment (start, end) token spans per sample.

    Segment g of a sample covers its group-g pairs contiguously
    [end of pair group g-1, end of the group's last pair). The batch-global
    segment count is the max over samples; samples with fewer groups get
    empty (0, 0) spans for the tail indices (the segment loop's existing
    all-pad skip handles them). Spans stay within each sample's real
    tokens, so no whole-context padding is needed in this mode.

    Returns (n_segments, list[n_segments] of (starts[B], ends[B]) int lists).
    """
    bang_id, colon_id = delim_ids
    ends_per_sample = detect_pair_ends(context_input_ids, pad_id, bang_id, colon_id)
    per_sample_groups = []
    n_segments = 0
    for ends in ends_per_sample:
        n_pairs = len(ends)
        sizes = divide_pairs(n_pairs, pairs_per_segment, n_groups_cfg)
        spans = []
        start_tok = 0          # token position where the current group starts
        pair_idx = 0           # index of the group's first pair
        for size in sizes:
            if size > 0:
                end_tok = ends[pair_idx + size - 1]   # token end of the group's last pair
                spans.append((start_tok, end_tok))
                start_tok = end_tok
                pair_idx += size
            else:
                spans.append((0, 0))
        per_sample_groups.append(spans)
        n_segments = max(n_segments, len(spans))
    B = context_input_ids.size(0)
    spans_by_seg = []
    for g in range(n_segments):
        starts, ends_g = [], []
        for spans in per_sample_groups:
            s, e = spans[g] if g < len(spans) else (0, 0)
            starts.append(s)
            ends_g.append(e)
        spans_by_seg.append((starts, ends_g))
    return n_segments, spans_by_seg


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
                 hopfield_value_activation="linear",
                 hopfield_value_topk=None,
                 hopfield_value_l1_weight=0.0,
                 hopfield_value_gate_init=0.5,
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
                 pair_aware_segmentation=False,
                 pairs_per_segment=None,
                 pair_delim_token_ids=None,
                 use_linear_state_read=False,
                 read_state_heads=4,
                 read_state_layers="all",
                 read_state_beta_init=1.0,
                 read_state_gate_init=0.1,
                 read_state_read_sites="all",
                 read_state_segment_emb=False,
                 read_state_exact_softmax=False,
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
             hopfield_value_activation: str, activation applied to the projected value code in Hopfield STORE
                 ("linear" = z = h, the original behavior; "topk" = keep the k largest positive coords;
                 "relu" = z = relu(h); "gated" = z = h * sigmoid(gate(v)) / hopfield_value_gate_init).
                 Any non-linear mode requires use_hopfield_memory=True and a non-null hopfield_value_proj_dim,
                 and is applied once in STORE: the stored tensor is the activated code, and both retrieval
                 paths decode it with the matching helper.
             hopfield_value_topk: int|None, k of the top-k winner code (only used with
                 hopfield_value_activation="topk"; requires 1 <= k <= hopfield_value_proj_dim). The actual
                 nnz may be < k when fewer than k coords are positive. Must be None in other modes.
             hopfield_value_l1_weight: float >= 0, weight of the OUTER L1 sparsity loss mean(abs(z)) on the
                 value codes (positive only supported with "relu"; the term never enters the K-step inner
                 update, and old BPTT-detached segments never re-enter the graph through it). With a
                 positive weight the DECODER is functionally unit-column-normalized
                 (F.normalize(weight, dim=0)) so the model cannot cancel the L1 pressure by shrinking the
                 encoder and growing decoder columns; "relu" with weight 0 and all other modes keep the
                 plain nn.Linear decoder.
             hopfield_value_gate_init: float in (0, 1), initial constant gate value for the "gated" mode
                 (the gate Linear is zero-weight/logit-bias initialized, so the initial activated code is
                 exactly h). Inert (but still validated) in the other modes.
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
             pair_aware_segmentation: bool, chunk the WRITE context at !K:V! pair boundaries instead of
                 fixed-width token windows (kv-retrieval only). Pair boundaries are detected directly
                 from context_input_ids via pair_delim_token_ids, so no extra batch plumbing is needed
                 and eval paths (incl. forward_per_segment_eval) see the same chunking for free.
                 Fixes the straddling pathology where a fixed window boundary falls between a pair's
                 key and its value, making that pair unretrievable from any single Hopfield/GD slot.
             pairs_per_segment: int|None, pair-aware mode: number of KV pairs per segment
                 (n_segments = ceil(n_pairs / pairs_per_segment) per sample). None = divide this
                 sample's pairs as equally as possible over hopfield_n_segments segments (the first
                 n_pairs % hopfield_n_segments groups get the extra pair). Mutually exclusive with
                 hopfield_segment_size.
             pair_delim_token_ids: list[int]|None, [open/close '!' id, ':' id] used to detect pairs in
                 context_input_ids. Auto-filled by the runner from the tokenizer when
                 pair_aware_segmentation is on.
             use_linear_state_read: bool, replace the Hopfield RETRIEVE + memory-prefix READ with
                 per-level linear-attention reads of the stored bank (TOKENWISE_READ.md; requires
                 use_hopfield_memory=True, mutually exclusive with use_gated_delta_memory and with
                 hopfield_value_proj_dim -- values are the raw post-inner-loop memories). The bank
                 is folded once into per-head states S_r = sum k v^T (k = L2-normalized
                 read_state_key_proj of each stored memory TOKEN, v = read_state_value_proj); at
                 every instrumented decoder block a forward hook injects
                 gamma_l * W_o(concat_r beta_r * normalize(Q_r(LN(x))) @ S_r) at the read sites of
                 the block OUTPUT (parallel, Flamingo/Titan-style; the frozen backbone pathway is
                 untouched). The READ sequence drops the memory prefix: [read_st?, read_end?, query]
                 (read_sites="all") or the two-pass interface layout (read_sites="interface").
                 Read-module params are outer-loop only; stored values join the outer graph through
                 the reads (detached per hopfield_bptt_segments exactly as today). Incompatible
                 with use_gradient_checkpointing (block hooks must fire during the READ forward,
                 which checkpoint-recompute would run outside the active read pass).
             read_state_heads: int >= 1, number of read heads H_r; head dim d_h = n_embd / H_r.
                 Wider heads reduce state cross-talk (state cost O(d^2/H_r) either way).
             read_state_layers: "all" or list[int], which decoder blocks get the read hook.
                 Indices refer to backbone blocks (gpt2 backbone.h, llama backbone.layers).
             read_state_beta_init: float, initial retrieval sharpness beta_r^l per head/layer
                 (scales the L2-normalized query; learnable per layer).
             read_state_gate_init: float > 0, initial per-layer gate gamma^l (LayerScale-style
                 small-positive init: 0 could never escape, 1.0 would perturb the frozen reader).
             read_state_read_sites: str, "all" = the bias is injected at every real (non-pad)
                 position of the read sequence; "interface" = only at the M learnable interface
                 tokens (count = n_mem_tokens, two-pass layout: pass 1 conditions [query
                 (target-span blinded), M_0] with the reads on; pass 2 answers
                 [read_st?, interface, read_end?, query] with the pass-1 interface states).
             read_state_segment_emb: stub (must stay False in v1): optional learnable per-segment
                 embedding before kappa/nu. Not implemented -- the per-segment count is dynamic
                 (pair-aware / segment_size modes), so the parameter would need lazy sizing.
             read_state_exact_softmax: bool, ablation arm: replace the superposed state read
                 q @ S_r with exact per-token softmax attention over the unfolded bank
                 softmax(q . K_r^T) @ V_r (O(N) per token, stored-out segments/pads masked).
                 Shares all projections/gates/site masks with the state arm.
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
        self.hopfield_value_activation = hopfield_value_activation
        self.hopfield_value_topk = hopfield_value_topk
        self.hopfield_value_l1_weight = hopfield_value_l1_weight
        self.hopfield_value_gate_init = hopfield_value_gate_init
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

        # Pair-aware segmentation (kv-retrieval): chunk the WRITE context at !K:V!
        # pair boundaries instead of fixed token windows (see docstring above).
        self.pair_aware_segmentation = pair_aware_segmentation
        self.pairs_per_segment = pairs_per_segment
        self.pair_delim_token_ids = pair_delim_token_ids

        # Linear-state per-level read (TOKENWISE_READ.md; see docstring above).
        self.use_linear_state_read = use_linear_state_read
        self.read_state_heads = read_state_heads
        self.read_state_layers = read_state_layers
        self.read_state_beta_init = read_state_beta_init
        self.read_state_gate_init = read_state_gate_init
        self.read_state_read_sites = read_state_read_sites
        self.read_state_segment_emb = read_state_segment_emb
        self.read_state_exact_softmax = read_state_exact_softmax

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
        # Nonlinear/sparse value codes (see the docstring above for semantics).
        assert hopfield_value_activation in ("linear", "topk", "relu", "gated"), \
            f"hopfield_value_activation must be 'linear', 'topk', 'relu', or 'gated', got '{hopfield_value_activation}'"
        if hopfield_value_activation != "linear":
            assert use_hopfield_memory, \
                f"hopfield_value_activation='{hopfield_value_activation}' requires use_hopfield_memory=True"
            assert hopfield_value_proj_dim is not None, \
                f"hopfield_value_activation='{hopfield_value_activation}' requires hopfield_value_proj_dim " \
                f"(the activation applies to the projected value code)"
        assert not (hopfield_value_topk is not None and hopfield_value_activation != "topk"), \
            f"hopfield_value_topk is only used with hopfield_value_activation='topk', got " \
            f"activation='{hopfield_value_activation}' with topk={hopfield_value_topk}"
        if hopfield_value_activation == "topk":
            assert hopfield_value_topk is not None and 1 <= hopfield_value_topk <= hopfield_value_proj_dim, \
                f"topk activation requires 1 <= hopfield_value_topk <= hopfield_value_proj_dim, got " \
                f"k={hopfield_value_topk}, proj_dim={hopfield_value_proj_dim}"
        assert hopfield_value_l1_weight >= 0, \
            f"hopfield_value_l1_weight must be >= 0, got {hopfield_value_l1_weight}"
        assert not (hopfield_value_l1_weight > 0 and hopfield_value_activation != "relu"), \
            f"a positive hopfield_value_l1_weight is only supported with hopfield_value_activation='relu', " \
            f"got activation='{hopfield_value_activation}' with l1_weight={hopfield_value_l1_weight}"
        assert 0 < hopfield_value_gate_init < 1, \
            f"hopfield_value_gate_init must be in (0, 1), got {hopfield_value_gate_init}"
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

        # Validate pair-aware segmentation settings (kv-retrieval)
        if pairs_per_segment is not None:
            assert pair_aware_segmentation, \
                "pairs_per_segment requires pair_aware_segmentation=True"
            assert pairs_per_segment >= 1, \
                f"pairs_per_segment must be >= 1, got {pairs_per_segment}"
        if pair_aware_segmentation:
            assert hopfield_segment_size is None, \
                "pair_aware_segmentation and hopfield_segment_size are mutually exclusive " \
                "(pair boundaries, not a token width, define the segments)"
            assert pair_delim_token_ids is not None and len(pair_delim_token_ids) == 2, \
                "pair_aware_segmentation requires pair_delim_token_ids=[bang_id, colon_id] " \
                "(the runner auto-fills them from the tokenizer)"
            assert all(isinstance(t, int) and t >= 0 for t in pair_delim_token_ids), \
                f"pair_delim_token_ids must be two non-negative token ids, got {pair_delim_token_ids}"
            assert pair_delim_token_ids[0] != pair_delim_token_ids[1], \
                "pair_delim_token_ids entries must be distinct"

        # Validate linear-state read settings (TOKENWISE_READ.md).
        assert read_state_read_sites in ("all", "interface"), \
            f"read_state_read_sites must be 'all' or 'interface', got '{read_state_read_sites}'"
        assert isinstance(read_state_heads, int) and read_state_heads >= 1, \
            f"read_state_heads must be an int >= 1, got {read_state_heads}"
        assert read_state_gate_init > 0, \
            f"read_state_gate_init must be > 0, got {read_state_gate_init}"
        if isinstance(read_state_layers, str):
            assert read_state_layers == "all", \
                f"read_state_layers must be 'all' or a list of block indices, got {read_state_layers!r}"
        else:
            assert isinstance(read_state_layers, (list, tuple)) and len(read_state_layers) > 0, \
                f"read_state_layers must be 'all' or a non-empty list of block indices, got {read_state_layers!r}"
            assert all(isinstance(i, int) and i >= 0 for i in read_state_layers), \
                f"read_state_layers entries must be non-negative ints, got {read_state_layers!r}"
            assert len(set(read_state_layers)) == len(list(read_state_layers)), \
                f"read_state_layers entries must be distinct, got {read_state_layers!r}"
        if read_state_segment_emb:
            # Stub (v1): the per-segment count is dynamic under pair-aware /
            # segment_size chunking, so the embedding would need lazy sizing.
            raise NotImplementedError(
                "read_state_segment_emb is not implemented in v1 (see TOKENWISE_READ.md); keep it False")
        if read_state_exact_softmax:
            assert use_linear_state_read, \
                "read_state_exact_softmax is an ablation arm of use_linear_state_read and requires it"
        if use_linear_state_read:
            assert use_hopfield_memory, \
                "use_linear_state_read requires use_hopfield_memory=True (it reads the Hopfield STORE bank)"
            assert not use_gated_delta_memory, \
                "use_linear_state_read is mutually exclusive with use_gated_delta_memory"
            assert hopfield_value_proj_dim is None, \
                "use_linear_state_read requires hopfield_value_proj_dim=None " \
                "(values are the raw post-inner-loop memory tokens; the encoded-code pathway is for the one-shot mixture)"
            assert not use_gradient_checkpointing, \
                "use_linear_state_read is incompatible with use_gradient_checkpointing " \
                "(the block hooks must fire during the READ forward; checkpoint recompute would re-run the blocks " \
                "outside the active read pass and drop the injected bias)"


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

        # Pair-aware segmentation (kv-retrieval): chunk the WRITE context at !K:V!
        # pair boundaries instead of fixed token windows.
        self.pair_aware_segmentation = getattr(config, "pair_aware_segmentation", False)
        self.pairs_per_segment = getattr(config, "pairs_per_segment", None)
        self.pair_delim_token_ids = getattr(config, "pair_delim_token_ids", None)

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
        # Nonlinear/sparse value codes. getattr defaults keep old serialized
        # configs (without these fields) loading as linear with no aux loss.
        self.hopfield_value_activation = getattr(config, "hopfield_value_activation", "linear")
        self.hopfield_value_topk = getattr(config, "hopfield_value_topk", None)
        self.hopfield_value_l1_weight = getattr(config, "hopfield_value_l1_weight", 0.0)
        self.hopfield_value_gate_init = getattr(config, "hopfield_value_gate_init", 0.5)
        # ReLU L1 sparsity is active (and switches the decoder to unit-normalized
        # columns) only for activation="relu" with a positive weight.
        self.hopfield_value_l1_active = (self.hopfield_value_activation == "relu"
                                         and self.hopfield_value_l1_weight > 0)

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
                # When the activated value code doubles as the STORE key
                # (hopfield_value_as_key with a projected value), the key
                # projection's source is the widened code dim Dv, not P = M*d.
                key_src_dim = self.hopfield_value_proj_dim \
                    if (self.hopfield_value_as_key and self.hopfield_value_proj_dim is not None) \
                    else pattern_dim
                self.hopfield_key_proj = nn.Linear(key_src_dim, self.hopfield_proj_dim, bias=True)
                self.hopfield_query_proj = nn.Linear(pattern_dim, self.hopfield_proj_dim, bias=True)
                if self.hopfield_proj_dim == key_src_dim:
                    with torch.no_grad():
                        nn.init.eye_(self.hopfield_key_proj.weight)
                        self.hopfield_key_proj.bias.zero_()
                if self.hopfield_proj_dim == pattern_dim:
                    with torch.no_grad():
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
                if self.hopfield_value_activation == "gated":
                    # Input-dependent elementwise soft gate. Zero weight +
                    # logit(gate_init) bias => the initial gate is constantly
                    # gate_init, so the division by gate_init below makes the
                    # initial activated code exactly h, while gradients train
                    # the gate immediately.
                    self.hopfield_value_gate = nn.Linear(pattern_dim, self.hopfield_value_proj_dim, bias=True)
                    with torch.no_grad():
                        self.hopfield_value_gate.weight.zero_()
                        g = self.hopfield_value_gate_init
                        self.hopfield_value_gate.bias.fill_(math.log(g / (1.0 - g)))

            # Without a key projection the value-as-key source must already be
            # P-sized; fail here (with the dims spelled out) instead of at bmm.
            if self.hopfield_value_as_key and self.hopfield_value_proj_dim is not None \
                    and self.hopfield_proj_dim is None \
                    and self.hopfield_value_proj_dim != self.n_mem_tokens * n_embd:
                raise ValueError(
                    f"hopfield_value_as_key with hopfield_value_proj_dim={self.hopfield_value_proj_dim} "
                    f"requires hopfield_proj_dim to be set: without a key projection the key source dim "
                    f"({self.hopfield_value_proj_dim}) must equal the memory pattern dim "
                    f"M*d ({self.n_mem_tokens * n_embd}).")

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

        # Linear-state per-level read (TOKENWISE_READ.md): replaces the Hopfield
        # RETRIEVE + memory-prefix READ when enabled. The read modules are
        # OUTER-LOOP ONLY params (like the read LoRA): they never enter the
        # inner loop, whose autograd.grad calls only ever target mem_batch /
        # per-sample proj tensors.
        self.use_linear_state_read = getattr(config, "use_linear_state_read", False)
        self.read_state_heads = getattr(config, "read_state_heads", 4)
        self.read_state_layers_cfg = getattr(config, "read_state_layers", "all")
        self.read_state_beta_init = getattr(config, "read_state_beta_init", 1.0)
        self.read_state_gate_init = getattr(config, "read_state_gate_init", 0.1)
        self.read_state_read_sites = getattr(config, "read_state_read_sites", "all")
        self.read_state_exact_softmax = getattr(config, "read_state_exact_softmax", False)
        # Escape hatch for tests/debugging: when True the registered hooks are
        # a no-op (same read layout, no injected bias).
        self._read_state_hooks_disabled = False
        # Active read-pass context (set around READ-phase forwards, else None):
        # carries the built state, the site mask and the stats accumulators the
        # block hooks read. WRITE-phase / compression forwards see it None.
        self._read_state_ctx = None

        if self.use_linear_state_read:
            if n_embd % self.read_state_heads != 0:
                raise ValueError(
                    f"read_state_heads={self.read_state_heads} must divide n_embd={n_embd}")
            # Layer subset: "all" or explicit block indices; validate against
            # the backbone now (the config only sees the raw list).
            backbone = get_backbone(self.model)
            if hasattr(backbone, "h"):          # GPT-2
                blocks = backbone.h
            elif hasattr(backbone, "layers"):   # Llama
                blocks = backbone.layers
            else:
                raise AttributeError(
                    "Could not locate decoder blocks on the backbone "
                    "(expected backbone.h for GPT-2 or backbone.layers for Llama).")
            n_blocks = len(blocks)
            if self.read_state_layers_cfg == "all":
                layer_ids = list(range(n_blocks))
            else:
                layer_ids = list(self.read_state_layers_cfg)
                bad = [i for i in layer_ids if i >= n_blocks]
                if bad:
                    raise ValueError(
                        f"read_state_layers indices {bad} out of range for backbone with "
                        f"{n_blocks} blocks")
            self.read_state_layer_ids = layer_ids

            d_h = n_embd // self.read_state_heads
            self.read_state_head_dim = d_h
            # Shared-across-layers bank projections kappa_r, nu_r: R^d -> R^{d_h}
            # per head (one Linear d->d, reshaped to [H, d_h, d] — equivalent to
            # per-head linears). The state is built once from these.
            self.read_state_key_proj = nn.Linear(n_embd, n_embd, bias=False)
            self.read_state_value_proj = nn.Linear(n_embd, n_embd, bias=False)
            # No bias on W_o: a zero state (nothing stored) must give exactly a
            # zero injected bias (grad_memgpt contract; a bias term would leak
            # gamma * b into every read even with an empty bank).
            self.read_state_out_proj = nn.Linear(n_embd, n_embd, bias=False)
            # Per-layer Q, beta, gamma, LN. LN matches ln_1 placement on the
            # block INPUT; it is deliberately a NEW LayerNorm (not the
            # backbone's own ln_1) so the query path is independent of it.
            self.read_state_q_proj = nn.ModuleList(
                [nn.Linear(n_embd, n_embd, bias=False) for _ in layer_ids])
            self.read_state_ln = nn.ModuleList(
                [nn.LayerNorm(n_embd) for _ in layer_ids])
            self.read_state_beta = nn.ParameterList(
                [nn.Parameter(torch.full((self.read_state_heads,), float(self.read_state_beta_init)))
                 for _ in layer_ids])
            self.read_state_gamma = nn.ParameterList(
                [nn.Parameter(torch.full((n_embd,), float(self.read_state_gate_init)))
                 for _ in layer_ids])
            # Interface tokens M_0 (read_sites="interface"): count = n_mem_tokens.
            if self.read_state_read_sites == "interface":
                self.read_state_interface_mem = nn.Parameter(
                    torch.randn(self.n_mem_tokens, n_embd) * 0.02)
            # map backbone block index -> position in the per-layer ModuleLists
            self.read_state_layer_pos = {blk: pos for pos, blk in enumerate(layer_ids)}
            # Register the block hooks once; they no-op unless the read-pass
            # context is active (see _read_state_ctx / _read_state_pass).
            for blk_idx in layer_ids:
                blocks[blk_idx].register_forward_hook(
                    self._make_read_state_hook(blk_idx), with_kwargs=True)

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

        # clamp to 0: an all-pad query (possible for INACTIVE probe slots when
        # samples in a batch have different pair counts) would otherwise yield
        # position_id = -1 and crash the position-embedding gather; real READ
        # queries start with a non-pad token, so the clamp is a no-op there.
        position_ids = (qry_attn_mask.cumsum(-1) - 1).clamp(min=0)
        outs_q = get_backbone(self.model)(inputs_embeds=x_qry_comp, attention_mask=qry_attn_mask,
                                          position_ids=position_ids, return_dict=True)
        features = outs_q.last_hidden_state[:, -self.n_mem_tokens:, :].view(B, -1)  # [B, M*d]
        del outs_q
        return features

    # ---------------------------------------------------------------- #
    # Linear-state per-level read (TOKENWISE_READ.md; use_linear_state_read).
    #
    # The stored bank never enters the read sequence: it is folded once into
    # per-head associative states S_r = sum_(s,j) k_(s,j) v_(s,j)^T, and a
    # forward hook on each instrumented decoder block injects
    #   gamma_l * W_o(concat_r beta_r * normalize(Q_r(LN(x))) @ S_r)
    # at the read sites of the block OUTPUT (parallel additive injection; the
    # pretrained attention pathway is bitwise untouched). All read-module
    # params are outer-loop only; the hooks no-op unless a READ pass is
    # active (see _read_state_ctx / _read_state_pass), so WRITE-phase and
    # compression forwards are unaffected.
    # ---------------------------------------------------------------- #

    def _build_linear_state(self, values_stack, masks_stack):
        """Fold the stored bank into per-head linear-attention states.

        values_stack: [B, c, M*d] raw stored memories (the linear-state STORE
            appends the raw post-inner-loop mem_batch per segment);
        masks_stack: [B, c] bool, False where a sample had no tokens in that
            segment (those items contribute nothing -- S = 0 for a sample with
            no stored segments at all, i.e. zero bias, no fallback prefix).

        Returns a dict:
            S:         [B, H, d_h, d_h] per-head states
            K, V:      [B, H, N, d_h]    unfolded bank keys/values (N = c*M);
                       used by the exact-softmax arm and the cross-talk diag
            item_mask: [B, N] bool, False for stored-out items
        """
        B, c, P = values_stack.shape
        d = P // self.n_mem_tokens
        H, d_h = self.read_state_heads, self.read_state_head_dim
        N = c * self.n_mem_tokens
        bank = values_stack.reshape(B, N, d)
        item_mask = masks_stack.unsqueeze(-1).expand(B, c, self.n_mem_tokens).reshape(B, N)
        k = self.read_state_key_proj(bank).view(B, N, H, d_h)
        k = F.normalize(k, dim=-1)
        v = self.read_state_value_proj(bank).view(B, N, H, d_h)
        keep = item_mask.view(B, N, 1, 1).to(k.dtype)
        k, v = k * keep, v * keep
        k = k.permute(0, 2, 1, 3)                       # [B, H, N, d_h]
        v = v.permute(0, 2, 1, 3)
        S = torch.einsum('bhnd,bhne->bhde', k, v)       # [B, H, d_h, d_h]
        return {'S': S, 'K': k, 'V': v, 'item_mask': item_mask}

    @staticmethod
    def _expand_linear_state(state, n_q):
        """Replicate a per-sample read state over the probe axis ([B,*] -> [B*n_q,*])."""
        def rep(t):
            return t.unsqueeze(1).expand(-1, n_q, *([-1] * (t.dim() - 1))).reshape(
                t.size(0) * n_q, *t.shape[1:])
        return {'S': rep(state['S']), 'K': rep(state['K']), 'V': rep(state['V']),
                'item_mask': rep(state['item_mask'])}

    @contextmanager
    def _read_state_pass(self, state, site_mask, collect_stats=False, want_cross_talk=False,
                         stats=None):
        """Activate the block hooks for one READ forward.

        site_mask: [B*, T, 1] float, 1 at the read sites (0 at pads / non-sites).
        `stats` is the accumulator dict the hooks fill (pass a shared dict to
        merge stats across the two interface passes). Nested passes (interface
        two-pass) stack and restore the previous ctx.
        """
        prev = self._read_state_ctx
        ctx = dict(state)
        ctx.update({'active': True,
                    'site_mask': site_mask,
                    'collect_stats': collect_stats,
                    'want_cross_talk': want_cross_talk,
                    'cross_talk_layer_pos': len(self.read_state_layer_ids) - 1,
                    'stats': stats if stats is not None else {}})
        self._read_state_ctx = ctx
        try:
            yield ctx
        finally:
            self._read_state_ctx = prev

    def _make_read_state_hook(self, block_idx):
        """Forward hook for decoder block `block_idx` (registered once in __init__)."""
        layer_pos = self.read_state_layer_pos[block_idx]

        def _hook(module, args, kwargs, output):
            ctx = self._read_state_ctx
            if ctx is None or not ctx.get('active') or self._read_state_hooks_disabled:
                return None
            x = args[0] if args else kwargs.get('hidden_states')
            mu = self._linear_state_read_bias(layer_pos, x, ctx)
            mu = mu * ctx['site_mask'].to(mu.dtype)
            if ctx.get('collect_stats'):
                with torch.no_grad():
                    acc = ctx['stats'].setdefault(layer_pos, {'mu_sum': None, 'x_sum': None, 'count': 0})
                    acc['mu_sum'] = mu.detach().float().norm(dim=-1).sum() if acc['mu_sum'] is None \
                        else acc['mu_sum'] + mu.detach().float().norm(dim=-1).sum()
                    acc['x_sum'] = x.detach().float().norm(dim=-1).sum() if acc['x_sum'] is None \
                        else acc['x_sum'] + x.detach().float().norm(dim=-1).sum()
                    acc['count'] += mu.size(0) * mu.size(1)
            if isinstance(output, tuple):
                return (output[0] + mu.to(output[0].dtype),) + tuple(output[1:])
            return output + mu.to(output.dtype)

        return _hook

    def _linear_state_read_bias(self, layer_pos, x, ctx):
        """Memory bias injected at one instrumented block.

        x: [B*, T, d] block INPUT hidden state; q_r = beta_r * norm(Q_r(LN(x)))
        per head, then mu = q_r @ S_r (state arm) or softmax(q_r K_r^T) V_r
        (exact-softmax arm, stored-out items masked); heads concatenated,
        projected by W_o, gated by gamma_l. Site masking is the hook's job.
        """
        d_h, H = self.read_state_head_dim, self.read_state_heads
        h = self.read_state_ln[layer_pos](x)
        q = self.read_state_q_proj[layer_pos](h)                     # [B*, T, d]
        Bt, T = q.shape[0], q.shape[1]
        q = q.view(Bt, T, H, d_h)
        q_hat = F.normalize(q, dim=-1)
        q_s = q_hat * self.read_state_beta[layer_pos].view(1, 1, H, 1)
        if self.read_state_exact_softmax:
            scores = torch.einsum('bthd,bhnd->bthn', q_s, ctx['K'])  # [B*, T, H, N]
            scores = scores.masked_fill(~ctx['item_mask'][:, None, None, :], float('-inf'))
            # A sample with NO stored items would softmax over all -inf (NaN);
            # give it zero scores and zero the weights -> zero read, like the
            # state arm's S = 0.
            any_item = ctx['item_mask'].any(dim=-1)                  # [B*]
            scores = torch.where(any_item[:, None, None, None], scores,
                                 torch.zeros_like(scores))
            w = F.softmax(scores, dim=-1) * any_item[:, None, None, None].to(scores.dtype)
            mu = torch.einsum('bthn,bhnd->bthd', w, ctx['V'])
        else:
            # mu[...,e] = sum_d q[d] S[d,e]  (q S, not q S^T: the output's head
            # axis must be labelled with S's SECOND matrix index)
            mu = torch.einsum('bthd,bhde->bthe', q_s, ctx['S'])      # [B*, T, H, d_h]
        if ctx.get('want_cross_talk') and layer_pos == ctx['cross_talk_layer_pos']:
            self._accumulate_read_state_cross_talk(q_hat, ctx)
        mu = mu.reshape(Bt, T, -1)
        mu = self.read_state_out_proj(mu)
        return mu * self.read_state_gamma[layer_pos]

    @staticmethod
    def _accumulate_read_state_cross_talk(q_hat, ctx):
        """Eval diagnostic: mean |q.k| over NON-top bank items (interference floor).

        Detached; accumulated on the last instrumented layer only. Valid items
        are filled with -1 before sorting so the per-position top-1 drops out
        (|q.k| in [0, 1] for normalized q, k).
        """
        with torch.no_grad():
            scores = torch.einsum('bthd,bhnd->bthn', q_hat, ctx['K']).abs().detach()
            scores = scores.masked_fill(~ctx['item_mask'][:, None, None, :], -1.0)
            sorted_scores, _ = scores.sort(dim=-1, descending=True)
            valid = sorted_scores > -0.5
            n_valid = valid.sum(dim=-1)
            rest = sorted_scores[..., 1:]
            acc = ctx['stats'].setdefault('cross_talk', {'sum': None, 'count': 0})
            seg_sum = (rest * valid[..., 1:]).sum()
            acc['sum'] = seg_sum if acc['sum'] is None else acc['sum'] + seg_sum
            acc['count'] = acc['count'] + (n_valid - 1).clamp(min=0).sum()

    def _drain_read_state_stats(self, ctx, inner_loop_stats, device):
        """Fold the read-pass accumulators into inner_loop_stats (all detached).

        Enabled-but-idle stats are explicit zeros so 'off' vs 'idle' stays
        distinguishable on the metric stream (see AGENTS.md).
        """
        H = self.read_state_heads
        gammas = torch.stack([g.abs().mean() for g in self.read_state_gamma])
        betas = torch.stack([b.mean() for b in self.read_state_beta])
        inner_loop_stats['read_state_gate_abs_mean'] = gammas.mean().detach()
        inner_loop_stats['read_state_beta_mean'] = betas.mean().detach()
        mu_ratios = []
        for layer_pos, acc in ctx['stats'].items():
            if not isinstance(layer_pos, int):
                continue
            if acc['count'] > 0 and acc['x_sum'] is not None and float(acc['x_sum']) > 0:
                mu_ratios.append((acc['mu_sum'] / acc['x_sum']).detach())
        inner_loop_stats['read_state_mu_over_x_norm_ratio'] = \
            torch.stack(mu_ratios).mean() if mu_ratios else torch.zeros((), device=device)
        ct = ctx['stats'].get('cross_talk')
        if ct is not None:
            inner_loop_stats['read_state_cross_talk'] = \
                (ct['sum'] / ct['count'].clamp(min=1)).detach() if ct['count'] > 0 \
                else torch.zeros((), device=device)

    # ---------------------------------------------------------------- #
    # Per-segment forgetting evaluation (eval-only)
    # ---------------------------------------------------------------- #
    # After each model-segment is written, probe retrieval of every KV pair
    # written so far (segments [0..k]) to track how older facts degrade as
    # later segments are processed. Returns a per-probe lower-triangular
    # correctness pattern (row = the segment a KV pair lives in, col = the
    # segment-after-write at which it was probed); the trainer bins the flat
    # (src, probe, em) triples into the [n_seg, n_seg] forgetting matrix.
    # Eval-only, gradient-safe: the WRITE inner loop still runs under enable_grad
    # (the memory update needs autograd.grad), but every retrieval READ is wrapped
    # in no_grad + .detach() so it neither pollutes the inner-loop graph nor
    # retains activations. See forward_per_segment_eval below.

    @staticmethod
    def _segment_bounds(seq_len, n_segments_cfg, segment_size_cfg):
        """Same ceil-division chunking as forward() uses for the WRITE context.

        NOTE (differs from grad_memgpt_adaptive): forward() honors
        hopfield_segment_size only when use_hopfield_memory or
        use_gated_delta_memory is set; the caller must pass segment_size_cfg=None
        for plain runs to mirror that condition.

        Returns (n_segments, segment_size, [(seg_start, seg_end), ...]) over the
        REAL (un-padded) token positions [0, seq_len). KV-pair attribution and
        the forward's segmentation must agree, so this mirrors the chunking the
        collator's build_kv_probe_queries uses (modulo forward's extra LEFT-pad
        to segment-aligned length, which the collator accounts for itself).
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

    def _detect_pair_ends(self, context_input_ids, pad_id):
        """Thin wrapper over the shared module-level detect_pair_ends."""
        return detect_pair_ends(context_input_ids, pad_id, *self.pair_delim_token_ids)

    @staticmethod
    def _divide_pairs(n_pairs, pairs_per_segment, n_groups_cfg):
        """Thin wrapper over the shared module-level divide_pairs.

        Kept as the canonical entry point: the runner's collator
        (build_kv_probe_queries) and the tests call GradMemGPT._divide_pairs so
        the model-side division and the KV->segment attribution can never
        diverge.
        """
        return divide_pairs(n_pairs, pairs_per_segment, n_groups_cfg)

    def _pair_segment_spans(self, context_input_ids, pad_id):
        """Thin wrapper over the shared module-level pair_segment_spans.

        Count-mode grouping uses self.hopfield_n_segments (the main model's
        segment-count knob).
        """
        return pair_segment_spans(context_input_ids, pad_id, self.pair_delim_token_ids,
                                  self.pairs_per_segment, self.hopfield_n_segments)

    def _read_once(self, mem_batch, query_input_ids, read_st_batch, read_end_batch):
        """One batched READ against the given memory state(s).

        Faithful factor of the READ phase in forward(): project mem with the
        READ-phase semantics ('proj_rw' uses read_mem_proj; 'per_sample'
        re-uses the meta-learned mem_proj init, NOT the inner-loop fast
        weights), concat [read_st?, mem, read_end?, query], pad for JVP Flash,
        forward under the READ-phase LoRA adapter, slice out the per-query-token
        logits. Batched over the query axis by flattening [B, n_q, Q] ->
        [B*n_q, Q] so this is a single forward pass.

        Args:
            mem_batch: [B, M, d] memory shared by all probes (plain modes), or
                [B, n_q, M, d] per-probe memories (Hopfield / Gated Delta, whose
                retrieval is query-dependent).
            query_input_ids: [B, n_q, Q] probe queries (already padded).
        Returns:
            logits [B, n_q, Q+1, V].
        """
        B, n_q, Q = query_input_ids.shape
        emb_layer = self.model.get_input_embeddings()
        qry_emb = emb_layer(query_input_ids.reshape(B * n_q, Q))            # [B*n_q, Q, d]
        if mem_batch.dim() == 3:
            mem_rep = mem_batch.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(
                B * n_q, *mem_batch.shape[1:])
        else:  # [B, n_q, M, d] per-probe retrieved memories
            mem_rep = mem_batch.reshape(B * n_q, *mem_batch.shape[2:])

        if self.mem_proj_mode == "none":
            mem_inp = mem_rep
        elif self.mem_proj_mode == "proj":
            mem_inp = self.mem_proj(mem_rep)
        elif self.mem_proj_mode == "proj_rw":
            mem_inp = self.read_mem_proj(mem_rep)
        else:  # "per_sample": READ re-uses the meta-learned init, like forward()
            W_read = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1)
            b_read = self.mem_proj.bias.unsqueeze(0).expand(B, -1)
            W_rep = W_read.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, *W_read.shape[1:])
            b_rep = b_read.unsqueeze(1).expand(-1, n_q, -1).reshape(B * n_q, -1)
            mem_inp = self._apply_linear(mem_rep, W_rep, b_rep)

        mem_offset = self.n_mem_tokens + self.n_ctrl_tokens * 2
        if self.n_ctrl_tokens > 0:
            # [B, C, d] -> [B, 1, C, d] -> [B, n_q, C, d] -> [B*n_q, C, d]
            rs = read_st_batch.reshape(B, 1, self.n_ctrl_tokens, -1).expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
            re_ = read_end_batch.reshape(B, 1, self.n_ctrl_tokens, -1).expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
            x_qry = torch.cat([rs, mem_inp, re_, qry_emb], dim=1)
        else:
            x_qry = torch.cat([mem_inp, qry_emb], dim=1)                    # [B*n_q, M+Q, d]

        # pad to multiple of 32 for compatibility with JVP Flash Attention
        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_list = [0, 0, 0, -x_qry.size(1) % 32]
            x_qry = F.pad(x_qry, pad_list, "constant", 0)

        self._set_phase_adapter("read")
        logits = self.model(inputs_embeds=x_qry).logits                     # [B*n_q, M+Q, V]
        logits = logits[:, mem_offset - 1:mem_offset + Q, :]                # [B*n_q, Q+1, V]
        return logits.reshape(B, n_q, Q + 1, -1)

    # ---------------------------------------------------------------- #
    # Linear-state READ forwards (use_linear_state_read)
    # ---------------------------------------------------------------- #

    def _linear_read_forward_pass(self, x, attn_mask, site_mask, state,
                                  collect_stats=False, want_cross_talk=False, stats=None,
                                  need_logits=True):
        """One forward with the read hooks active on `site_mask`.

        x: [B*, T, d]; attn_mask: [B*, T] long; site_mask: [B*, T, 1] float.
        Pads to a multiple of 32 on the jvp paths (embeddings AND mask), runs
        the READ-phase adapter. need_logits=True runs the full CausalLM (call
        .logits on the result); False runs the bare backbone (call
        .last_hidden_state -- the interface pass-1 conditioning needs only
        hidden states).
        """
        T = x.size(1)
        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual') and T % 32:
            pad_len = 32 - T % 32
            x = F.pad(x, [0, 0, 0, pad_len], "constant", 0)
            attn_mask = F.pad(attn_mask, [0, pad_len], "constant", 0)
        # cumsum-1 clamp: masked positions (blinded target span / pads) must
        # not produce position_id = -1 (same pattern as _compress_query).
        position_ids = (attn_mask.cumsum(-1) - 1).clamp(min=0)
        runner = self.model if need_logits else get_backbone(self.model)
        with self._read_state_pass(state, site_mask, collect_stats=collect_stats,
                                   want_cross_talk=want_cross_talk, stats=stats):
            self._set_phase_adapter("read")
            return runner(inputs_embeds=x, attention_mask=attn_mask,
                          position_ids=position_ids, return_dict=True)

    @staticmethod
    def _slice_read_logits(logits_full, read_offset, q_len):
        """[B*, T, V] -> [B*, Q+1, V]: row j predicts query token j.

        `read_offset` prefix tokens (ctrl / interface) precede the query, so
        row j = logits at position read_offset-1+j (matching the memory-prefix
        READ's slicing). With NO prefix (read_offset == 0) nothing predicts
        query token 0 -- pad a zero row; every scored layout has label -100 at
        query position 0, so CE never consumes it.
        """
        if read_offset > 0:
            return logits_full[:, read_offset - 1:read_offset + q_len, :]
        head = torch.zeros(logits_full.size(0), 1, logits_full.size(-1),
                           dtype=logits_full.dtype, device=logits_full.device)
        return torch.cat([head, logits_full[:, :q_len, :]], dim=1)

    def _linear_state_read_logits(self, qry_emb, query_input_ids, labels,
                                  read_st_batch, read_end_batch, state,
                                  collect_stats=False, collect_cross_talk=False, stats=None,
                                  return_interface=False):
        """READ-phase logits for the linear-state mode -> [B, Q+1, V].

        qry_emb: [B, Q, d]; query_input_ids: [B, Q]; state: read state already
        expanded to the batching of qry_emb. read_sites="all": one pass over
        [read_st?, read_end?, query], bias at every real position.
        read_sites="interface": pass 1 conditions the learnable interface
        tokens M_0 on the target-span-blinded query (reads active on the M_0
        positions); pass 2 answers [read_st?, interface, read_end?, query]
        with the pass-1 interface states (reads active on the interface
        positions only). The blinding reuses _compress_query's labels >= 0
        pattern so pass 1 cannot peek at the answer. return_interface=True
        additionally returns the pass-1 interface embeddings [B, M, d]
        (diagnostics/tests: the query-sensitivity and blinding checks).
        """
        pad_id = self.model.config.pad_token_id
        device = qry_emb.device
        C = self.n_ctrl_tokens
        M = self.n_mem_tokens
        query_real = (query_input_ids != pad_id).long()                  # [B, Q]
        if C > 0:
            ctrl_ones = torch.ones(query_real.size(0), C, dtype=torch.long, device=device)
        if self.read_state_read_sites == "all":
            if C > 0:
                x = torch.cat([read_st_batch, read_end_batch, qry_emb], dim=1)
                attn = torch.cat([ctrl_ones, query_real], dim=1)
            else:
                x = qry_emb
                attn = query_real
            site_mask = attn.float().unsqueeze(-1)                       # every real position
            out = self._linear_read_forward_pass(x, attn, site_mask, state,
                                                 collect_stats=collect_stats,
                                                 want_cross_talk=collect_cross_talk,
                                                 stats=stats)
            logits = self._slice_read_logits(out.logits, C, query_input_ids.size(1))
            return (logits, None) if return_interface else logits

        # ---- interface mode, pass 1: condition M_0 on the blinded query ----
        qry_mask = query_real.clone()
        if labels is not None:
            qry_mask[labels >= 0] = 0
        m0 = self.read_state_interface_mem.unsqueeze(0).expand(query_real.size(0), -1, -1)
        mem_ones = torch.ones(query_real.size(0), M, dtype=torch.long, device=device)
        if C > 0:
            x1 = torch.cat([read_st_batch, qry_emb, read_end_batch, m0], dim=1)
            a1 = torch.cat([ctrl_ones, qry_mask, ctrl_ones, mem_ones], dim=1)
        else:
            x1 = torch.cat([qry_emb, m0], dim=1)
            a1 = torch.cat([qry_mask, mem_ones], dim=1)
        T1 = x1.size(1)
        site1 = torch.zeros(query_real.size(0), T1, 1, device=device)
        site1[:, -M:] = 1.0
        out1 = self._linear_read_forward_pass(x1, a1, site1, state,
                                              collect_stats=collect_stats, stats=stats,
                                              need_logits=False)
        iface = out1.last_hidden_state[:, T1 - M:T1, :]                  # [B, M, d]

        # ---- pass 2: answer with the conditioned interface embeddings ----
        if C > 0:
            x2 = torch.cat([read_st_batch, iface, read_end_batch, qry_emb], dim=1)
            a2 = torch.cat([ctrl_ones, mem_ones, ctrl_ones, query_real], dim=1)
        else:
            x2 = torch.cat([iface, qry_emb], dim=1)
            a2 = torch.cat([mem_ones, query_real], dim=1)
        T2 = x2.size(1)
        site2 = torch.zeros(query_real.size(0), T2, 1, device=device)
        site2[:, C:C + M] = 1.0
        out2 = self._linear_read_forward_pass(x2, a2, site2, state,
                                              collect_stats=collect_stats,
                                              want_cross_talk=collect_cross_talk,
                                              stats=stats)
        logits = self._slice_read_logits(out2.logits, C + M, query_input_ids.size(1))
        return (logits, iface) if return_interface else logits

    def _read_once_linear(self, state, query_input_ids, read_st_batch, read_end_batch,
                          target_mask=None):
        """Batched linear-state READ for the forgetting probes (mirrors _read_once).

        state: [B, ...] per-sample read state (built from stored_*[:n_stored]);
        query_input_ids: [B, n_q, Q]. Flattens the probe axis, drops the memory
        prefix, and serves the state read (all mode) or the interface two-pass
        (interface mode). target_mask blinds the teacher-forced target span in
        the interface pass-1 conditioning (the _compress_query pattern).
        Returns logits [B, n_q, Q+1, V].
        """
        B, n_q, Q = query_input_ids.shape
        device = query_input_ids.device
        flat_qids = query_input_ids.reshape(B * n_q, -1)
        qry_emb = self.model.get_input_embeddings()(flat_qids)           # [B*n_q, Q, d]
        labels = None
        if target_mask is not None:
            labels = torch.where(target_mask.reshape(B * n_q, -1), flat_qids,
                                 torch.full_like(flat_qids, -100))
        if self.n_ctrl_tokens > 0:
            rs = read_st_batch.reshape(B, 1, self.n_ctrl_tokens, -1) \
                .expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
            re_ = read_end_batch.reshape(B, 1, self.n_ctrl_tokens, -1) \
                .expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
        else:
            rs = re_ = None
        state_exp = self._expand_linear_state(state, n_q)
        logits = self._linear_state_read_logits(qry_emb, flat_qids, labels, rs, re_,
                                                state_exp)
        return logits.reshape(B, n_q, Q + 1, -1)

    @staticmethod
    def _probe_exact_match(logits, query_ids, target_mask, ignore_token_ids):
        """Per-probe exact-match flag, teacher-forced (mirrors compute_metrics_fn).

        The READ query is the teacher-forced sequence ``?!K:V!|`` (query + target
        concatenated, exactly as collate_fn / build_kv_probe_queries build it).
        logits[t] predicts token t+1, so the prediction AT a target position
        predicts the NEXT target token. We score only the positions flagged by
        ``target_mask`` (the ``V!|`` span, excluding structural '!'/'|' which
        are in ignore_token_ids): a probe is correct iff argmax matches the
        actual token id at every scored position.

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

    def _encode_hopfield_value(self, value_pattern):
        """Flattened segment memory [B, P] -> activated value code ([B, Dv] when projected, else [B, P]).

        The single activation point of the Hopfield STORE path: the stored
        tensor is the activated code z (never the preactivation h), and both
        retrieval paths (forward RETRIEVE + _probe_boundary_memory) decode it
        with _decode_hopfield_value. Modes (hopfield_value_activation):
          linear: z = h (the original, exactly backward-compatible behavior)
          topk:   nonnegative k-winner code -- the k largest positive coords of
                  relu(h) per sample, zero elsewhere (nnz < k when fewer coords
                  are positive)
          relu:   z = relu(h)
          gated:  z = h * sigmoid(gate(v)) / hopfield_value_gate_init (zero-init
                  gate weight + logit(gate_init) bias => initially exactly h)
        """
        if self.hopfield_value_proj_dim is None:
            return value_pattern
        h = self.hopfield_value_proj(value_pattern)
        mode = self.hopfield_value_activation
        if mode == "linear":
            return h
        if mode == "topk":
            positive = F.relu(h)
            vals, idx = positive.topk(min(self.hopfield_value_topk, positive.size(-1)), dim=-1)
            z = torch.zeros_like(positive)
            z.scatter_(-1, idx, vals)
            return z
        if mode == "relu":
            return F.relu(h)
        if mode == "gated":
            gate = torch.sigmoid(self.hopfield_value_gate(value_pattern))
            return h * (gate / self.hopfield_value_gate_init)
        raise ValueError(f"Unknown hopfield_value_activation: {mode}")

    def _decode_hopfield_value(self, value_code):
        """Retrieved value code [B, Dv] -> flattened memory pattern [B, P].

        With ReLU L1 sparsity enabled the decoder is applied with functionally
        unit-normalized columns (F.normalize(weight, dim=0)): an unconstrained
        decoder would let the model cancel the L1 pressure by shrinking encoder
        activations and growing decoder columns at a constant decoded result.
        Other modes (and relu with l1_weight == 0) use the plain nn.Linear.
        """
        if self.hopfield_value_proj_dim is None:
            return value_code
        if self.hopfield_value_l1_active:
            weight = F.normalize(self.hopfield_value_inv_proj.weight, dim=0)
            return F.linear(value_code, weight, self.hopfield_value_inv_proj.bias)
        return self.hopfield_value_inv_proj(value_code)

    @staticmethod
    def _hopfield_value_code_active(z, activation, eps=1e-12):
        """Boolean activity mask used by the hopfield_value_active_frac / feature-utilization stats."""
        if activation in ("relu", "topk"):
            return z > 0
        return z.abs() > eps

    def _accumulate_hopfield_value_stats(self, acc, value_code, value_pattern, seg_has_tokens):
        """Per-segment accumulation for the hopfield_value_* diagnostics.

        acc is the per-forward dict built in forward(); everything stays
        detached so BPTT graph lifetimes are unaffected, and only the
        seg_has_tokens rows contribute (padded/empty sample-segments never
        enter any denominator).
        """
        with torch.no_grad():
            if not seg_has_tokens.any():
                return
            zv = value_code.detach()[seg_has_tokens]            # [n_valid, Dv]
            active = self._hopfield_value_code_active(zv, self.hopfield_value_activation)
            acc['abs_sum'] = acc['abs_sum'] + zv.abs().sum()
            acc['norm_sum'] = acc['norm_sum'] + zv.norm(dim=-1).sum()
            acc['active'] = acc['active'] + active.sum()
            acc['count'] += zv.size(0)
            util = active.any(dim=0)                            # [Dv] bool
            acc['util_any'] = util if acc['util_any'] is None else (acc['util_any'] | util)
            if self.hopfield_value_activation == "gated":
                g = torch.sigmoid(self.hopfield_value_gate(value_pattern[seg_has_tokens].detach()))
                acc['gate_sum'] = acc['gate_sum'] + g.sum()
                acc['gate_sat'] = acc['gate_sat'] + ((g < 0.05) | (g > 0.95)).sum()
            if self.hopfield_value_l1_active:
                # detached diagnostic over ALL processed segments (graph-
                # retained or not); the trainable accumulator lives in forward.
                acc['l1_all_sum'] = acc['l1_all_sum'] + zv.abs().mean(dim=-1).sum()
                acc['l1_all_count'] += zv.size(0)

    def _probe_boundary_memory(self, snap, stored_keys, stored_values, stored_masks,
                               query_input_ids, target_mask, mem_batch_initial,
                               read_st_batch, read_end_batch, B, device):
        """Memory state READ would consume at one segment boundary, for probing.

        Mirrors what forward()'s READ phase consumes if it stopped at this
        boundary, per memory mode:
          - plain (no associative store): the segment's carried mem
            (snap['mem']) -> [B, M, d], shared by all probes. NOTE the plain
            grad_memgpt memory is RESET per segment, so probing a KV from an
            older segment against this mem measures reset-forgetting.
          - Hopfield: retrieval over the segments stored so far
            (stored_*[:snap['n_stored']]). Retrieval is QUERY-dependent, so it
            is re-run per probe -> [B, n_q, M, d]. All retrieval modes
            (raw/softmax/beta_softmax/mean), key/query/value projections,
            direct queries and the unwritten-sample fallback mirror forward()'s
            Hopfield RETRIEVE phase.
          - Gated Delta: linear read o = S @ q against snap['S'] (also
            query-dependent -> [B, n_q, M, d]), mirroring forward()'s Gated
            Delta RETRIEVE phase incl. the unwritten-sample fallback.

        The probe query compression is blind to the teacher-forced target span
        (target_mask stands in for the labels >= 0 positions that forward's
        _compress_query masks out), so retrieval cannot peek at the answer.
        Must be called under no_grad (the snapshots are already detached).
        """
        if not (self.use_hopfield_memory or self.use_gated_delta_memory):
            return snap['mem']                                        # [B, M, d]

        qids = query_input_ids                                        # [B, n_q, Q]
        n_q = qids.size(1)
        pad_id = self.model.config.pad_token_id

        # flatten probes to a [B*n_q] batch; _compress_query is batch-agnostic
        flat_qids = qids.reshape(B * n_q, -1)
        # blind the query compression to the teacher-forced 'V!|' span, exactly
        # like labels >= 0 in forward's _compress_query
        flat_labels = torch.where(target_mask, qids,
                                  torch.full_like(qids, -100)).reshape(B * n_q, -1)
        qry_emb = self.model.get_input_embeddings()(flat_qids)         # [B*n_q, Q, d]

        if self.n_ctrl_tokens > 0:
            rs_flat = read_st_batch.reshape(B, 1, self.n_ctrl_tokens, -1).expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
            re_flat = read_end_batch.reshape(B, 1, self.n_ctrl_tokens, -1).expand(-1, n_q, -1, -1).reshape(B * n_q, self.n_ctrl_tokens, -1)
        else:
            rs_flat = re_flat = None

        if self.use_hopfield_memory and snap['n_stored'] > 0:
            c = snap['n_stored']
            keys = torch.stack(stored_keys[:c], dim=1)                 # [B, c, M*d] (Dv when value-as-key)
            values = torch.stack(stored_values[:c], dim=1)             # [B, c, Dv] (activated codes; M*d unprojected)
            mask = torch.stack(stored_masks[:c], dim=1).bool()         # [B, c]
            stored = mask.any(dim=1)                                   # [B]

            if self.hopfield_direct_query:
                query_key = self.mem_query.unsqueeze(0).expand(B * n_q, -1, -1).reshape(B * n_q, -1)
            else:
                if self.use_separate_hopfield_mem:
                    mem_query_prefix = self.mem_query.unsqueeze(0).expand(B, -1, -1)
                else:
                    mem_query_prefix = mem_batch_initial
                mq_flat = mem_query_prefix.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, *mem_query_prefix.shape[1:])
                query_key = self._compress_query(qry_emb, flat_qids, flat_labels, pad_id,
                                                 mq_flat, rs_flat, re_flat, B * n_q, device)
            if self.hopfield_proj_dim is not None:
                query_key = self.hopfield_query_proj(query_key)
            query_key = F.normalize(query_key, dim=-1)                 # [B*n_q, M*d]

            # replicate the per-sample store over the probe axis
            keys_flat = keys.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, c, -1)
            values_flat = values.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, c, -1)
            mask_flat = mask.unsqueeze(1).expand(-1, n_q, -1).reshape(B * n_q, c)

            if self.hopfield_retrieval_mode == "mean":
                weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                w_flat = weights.unsqueeze(1).expand(-1, n_q, -1).reshape(B * n_q, c)
                # [B*n_q, 1, c] @ [B*n_q, c, Dv] -> [B*n_q, Dv] (the row-vector
                # form; unsqueeze(2) here produced mismatched bmm inner dims)
                retrieved_pattern = torch.bmm(w_flat.unsqueeze(1), values_flat).squeeze(1)
            else:
                scores = torch.bmm(query_key.unsqueeze(1), keys_flat.transpose(1, 2)).squeeze(1)  # [B*n_q, c]
                if self.hopfield_retrieval_mode == "raw":
                    scores = scores * mask_flat.float()
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values_flat).squeeze(1)
                elif self.hopfield_retrieval_mode == "softmax":
                    scores = scores.masked_fill(~mask_flat, float('-inf'))
                    scores = F.softmax(scores, dim=-1)
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values_flat).squeeze(1)
                elif self.hopfield_retrieval_mode == "beta_softmax":
                    scores = scores * self.hopfield_beta
                    scores = scores.masked_fill(~mask_flat, float('-inf'))
                    scores = F.softmax(scores, dim=-1)
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values_flat).squeeze(1)
                else:
                    raise ValueError(f"Unknown hopfield_retrieval_mode: {self.hopfield_retrieval_mode}")

            if self.hopfield_value_proj_dim is not None:
                # decode the retrieved activated code (unit-normalized decoder
                # columns when ReLU L1 is active -- same helper as forward)
                retrieved_pattern = self._decode_hopfield_value(retrieved_pattern)

            # samples with nothing stored yet: fall back to the initial memory
            if not stored.all():
                init_flat = mem_batch_initial.view(B, 1, -1).expand(-1, n_q, -1).reshape(B * n_q, -1)
                retrieved_pattern = torch.where(stored.unsqueeze(1).expand(-1, n_q).reshape(B * n_q),
                                                retrieved_pattern, init_flat)
            return retrieved_pattern.view(B, n_q, self.n_mem_tokens, -1)  # [B, n_q, M, d]

        # Gated Delta
        written = snap['written']                                      # [B] bool
        if written is None or not written.any():
            # nothing written for ANY sample: forward's retrieval is skipped
            # and READ consumes the plain carried memory
            return snap['mem']
        S_rep = snap['S'].to(dtype=qry_emb.dtype)
        S_rep = S_rep.unsqueeze(1).expand(-1, n_q, -1, -1).reshape(B * n_q, *S_rep.shape[1:])
        query_features = self._compress_query(qry_emb, flat_qids, flat_labels, pad_id,
                                              mem_batch_initial.unsqueeze(1).expand(-1, n_q, -1, -1)
                                              .reshape(B * n_q, *mem_batch_initial.shape[1:]),
                                              rs_flat, re_flat, B * n_q, device)
        q_t = F.normalize(self.gd_query_proj(query_features), dim=-1)  # [B*n_q, d]
        retrieved = torch.bmm(S_rep, q_t.unsqueeze(2)).squeeze(2)      # [B*n_q, d]
        retrieved_pattern = self.gd_value_inv_proj(retrieved)          # [B*n_q, M*d]

        # samples with no writes: fall back to the initial memory
        if not written.all():
            init_flat = mem_batch_initial.view(B, 1, -1).expand(-1, n_q, -1).reshape(B * n_q, -1)
            retrieved_pattern = torch.where(written.unsqueeze(1).expand(-1, n_q).reshape(B * n_q),
                                            retrieved_pattern, init_flat)
        return retrieved_pattern.view(B, n_q, self.n_mem_tokens, -1)   # [B, n_q, M, d]

    def forward_per_segment_eval(self, input_ids, kv_queries):
        """Eval-only: per-query retrieval results for the forgetting matrix.

        Runs the normal segmented WRITE inner loop (memory identical to
        training, via forward(collect_segment_mems=True)), and after each
        model-segment's K steps probes retrieval of every KV pair whose source
        segment <= the just-written segment (cumulative forgetting curve).

        Boundary-memory semantics differ by memory mode (mirroring what
        forward()'s READ consumes, see _probe_boundary_memory): plain modes
        RESET mem per segment (cell (s,k), s<k, measures reset-forgetting --
        an architecture property); Hopfield / Gated Delta ACCUMULATE the store
        across segments, and their query-dependent retrieval is re-run per
        probe against the store as of boundary k (interference-forgetting
        inside one associative memory -- directly comparable to the adaptive
        fork's forgetting matrix).

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
        ignore_token_ids = kv_queries.get('ignore_token_ids', [])

        # segment count, matching forward's chunking. Pair-aware mode derives
        # it from the detected pairs (same helper forward uses). Otherwise
        # mirror the fixed-width formula (hopfield_segment_size honored only
        # for hopfield/gated-delta runs).
        if self.pair_aware_segmentation:
            n_seg, _ = self._pair_segment_spans(context_input_ids, pad_id)
        else:
            seq_len_padded = context_input_ids.size(1)
            use_seg_size = (self.use_hopfield_memory or self.use_gated_delta_memory) \
                and self.hopfield_segment_size is not None
            n_seg, seg_sz, _ = self._segment_bounds(
                seq_len_padded, self.hopfield_n_segments,
                self.hopfield_segment_size if use_seg_size else None)

        src_seg_list, probe_seg_list, em_list = [], [], []

        if not self.K or not (context_input_ids != pad_id).any() or qmask.sum() == 0:
            return (torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.bool, device=device))

        # ---- get the EXACT per-boundary memory states from the real forward ---- #
        # forward() requires a query_input_ids for its READ phase; pass a minimal
        # single-token placeholder -- its READ output is discarded.
        fwd_input = {'context_input_ids': context_input_ids,
                     'query_input_ids': context_input_ids[:, :1]}
        out = self.forward(fwd_input, labels=None, collect_segment_mems=True)
        segment_mems = out['segment_mems']              # list[n_segments] of snap-dicts or None
        stored_keys = out.get('stored_keys')
        stored_values = out.get('stored_values')
        stored_masks = out.get('stored_masks')
        n_seg_actual = len(segment_mems)                # == n_segments (None placeholders keep alignment)

        mem_batch_initial = self.mem.unsqueeze(0).expand(B, -1, -1)
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
                snap = segment_mems[seg_idx_loop]
                if snap is None:
                    # empty segment: no WRITE ran, memory state unchanged. Still
                    # probe so KV pairs in segments <= seg_idx_loop are tested at
                    # this probe point -- use the last available snapshot.
                    snap = next((segment_mems[j] for j in range(seg_idx_loop, -1, -1)
                                 if segment_mems[j] is not None), None)
                    if snap is None:
                        continue
                probe_here = (seg_idx <= seg_idx_loop) & qmask             # [B, n_q]
                if not probe_here.any():
                    continue
                if self.use_linear_state_read:
                    # Linear-state probe: there is no retrieved pattern to
                    # compress -- the probe read IS the state read (or interface
                    # two-pass) against the bank as of this boundary
                    # (stored_*[:n_stored]), batched over the probe axis.
                    c = snap['n_stored']
                    if c > 0:
                        values_stack = torch.stack(stored_values[:c], dim=1)
                        masks_stack = torch.stack(stored_masks[:c], dim=1).bool()
                    else:
                        values_stack = mem_batch_initial.view(B, 1, -1)
                        masks_stack = torch.zeros(B, 1, dtype=torch.bool, device=device)
                    state = self._build_linear_state(values_stack, masks_stack)
                    logits_q = self._read_once_linear(state, qids, read_st_batch,
                                                      read_end_batch, target_mask=tmask)
                else:
                    mem_det = self._probe_boundary_memory(
                        snap, stored_keys, stored_values, stored_masks,
                        qids, tmask, mem_batch_initial, read_st_batch, read_end_batch,
                        B, device)
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

    def forward(self, input_ids, labels=None, return_mem=False, collect_segment_mems=False):
        # context_input_ids : B × S   (segments only, each ends with `|`)
        # query_input_ids   : B × Q   (e.g.  "?!K:V!|") i.e. the last segment
        # labels            : B × Q   (‑100 everywhere except the target tokens (V!|))
        # collect_segment_mems : eval-only; snapshot the memory state READ would
        #     consume after each segment's WRITE (see forward_per_segment_eval)

        """
        All tensors already padded to the same length in the datacollator.
        """
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']
        # Present only for ordered two-key concatenation evaluation.  They are
        # deliberately nested under input_ids so Trainer preserves them for
        # compute_metrics_fn, while this forward otherwise ignores metric-only
        # metadata such as pair_value_token_index/source_segments.
        single_query_input_ids = input_ids.get('single_query_input_ids')
        single_labels = input_ids.get('single_labels')

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
        # Nonlinear value-code accumulators (fresh each forward call, detached):
        # `hv` feeds the hopfield_value_* diagnostics, the L1 pair feeds the
        # trainable sparse-loss term (graph-retained segments only, see STORE).
        hv = None
        if self.use_hopfield_memory and self.hopfield_value_proj_dim is not None:
            hv = {'abs_sum': torch.zeros((), device=device),
                  'norm_sum': torch.zeros((), device=device),
                  'active': torch.zeros((), device=device),
                  'count': 0,
                  'util_any': None,
                  'gate_sum': torch.zeros((), device=device),
                  'gate_sat': torch.zeros((), device=device),
                  'l1_all_sum': torch.zeros((), device=device),
                  'l1_all_count': 0}
        hopfield_l1_train_sum = None
        hopfield_l1_train_count = 0
        if self.use_hopfield_memory and self.hopfield_value_l1_active:
            hopfield_l1_train_sum = torch.zeros((), device=device)

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
        # per-boundary memory snapshots (eval-only forgetting probe); filled
        # inside the segment loop, stays None when WRITE never runs
        segment_mems = None

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

                # Split context into segments. Pair-aware mode (kv-retrieval):
                # chunk at !K:V! pair boundaries so a pair's key and value are
                # never split across segments -- a fixed window boundary between
                # them makes the pair unretrievable from any single Hopfield/GD
                # slot (its query matches the previous segment's stored key,
                # which lacks the answer). Default: fixed-width windows,
                # ceil-padded (LEFT) to equal size.
                if self.pair_aware_segmentation:
                    n_segments, pair_spans = self._pair_segment_spans(context_input_ids, pad_id)
                    segment_size = None
                else:
                    pair_spans = None
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

                # per-boundary memory snapshots (eval-only forgetting probe):
                # one entry per segment index. Empty segments (all-pad, possible
                # with the LEFT-padding above) are skipped by `continue` below,
                # so we must NOT use append-order -- that would misalign
                # snapshot[i] with segment i and corrupt the forgetting matrix
                # diagonal. We store None for skipped segments so indices stay
                # aligned (the eval falls back to the last non-None snapshot:
                # an empty segment leaves the memory state unchanged).
                segment_mems = ([None] * n_segments) if collect_segment_mems else None

                for seg_idx in range(n_segments):
                    if pair_spans is not None:
                        # pair-aware: per-sample (start, end) token spans, right-
                        # padded within the window to the batch-max group width
                        # (real tokens at the left, so cumsum position ids give
                        # them 0..k-1 exactly; pads are masked out anyway)
                        starts, ends = pair_spans[seg_idx]
                        width = max(e - s for s, e in zip(starts, ends))
                        seg_emb = ctx_emb.new_zeros(B, width, ctx_emb.size(-1))
                        seg_mask = torch.zeros(B, width, dtype=mask.dtype, device=device)
                        seg_labels = torch.full((B, width), -100, dtype=lm_labels.dtype, device=device)
                        for b in range(B):
                            s, e = starts[b], ends[b]
                            if e > s:
                                seg_emb[b, :e - s] = ctx_emb[b, s:e]
                                seg_mask[b, :e - s] = mask[b, s:e]
                                seg_labels[b, :e - s] = lm_labels[b, s:e]
                    else:
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
                        if self.use_linear_state_read:
                            # Raw-token STORE for the linear-state read
                            # (TOKENWISE_READ.md): values are the raw
                            # post-inner-loop memories, no value encoding
                            # (hopfield_value_proj_dim is None by assertion)
                            # and no segment-key forward pass -- the read
                            # builds token-level keys from the bank itself.
                            # hopfield_bptt_segments detaches the raw tensors
                            # exactly as it detaches keys/values today.
                            value_code = mem_batch.view(B, -1)            # [B, M*d]
                            if self.hopfield_bptt_segments is not None and \
                                    (n_segments - seg_idx - 1) >= self.hopfield_bptt_segments:
                                value_code = value_code.detach()
                            stored_keys.append(None)
                            stored_values.append(value_code)
                            stored_masks.append(seg_has_tokens)
                            hopfield_stored = hopfield_stored | seg_has_tokens
                        else:
                            # Value: gradient-connected mem_batch after inner loop on this segment,
                            # encoded+activated into the stored value code z ([B, Dv] when projected).
                            # The STORED tensor is the activated code, never the preactivation.
                            value_pattern = mem_batch.view(B, -1)  # [B, M*d]
                            value_code = self._encode_hopfield_value(value_pattern)

                            if self.hopfield_value_as_key:
                                # Use the activated value code directly as key (the
                                # activation is applied exactly once -- here)
                                seg_key = value_code
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
                                value_code = value_code.detach()

                            # Store for attention-based retrieval
                            stored_keys.append(seg_key)
                            stored_values.append(value_code)
                            stored_masks.append(seg_has_tokens)
                            hopfield_stored = hopfield_stored | seg_has_tokens

                            # Nonlinear value-code diagnostics (detached) + ReLU L1
                            # sparse-loss accumulator. Only graph-retained segments
                            # enter the trainable accumulator: BPTT-detached codes
                            # have requires_grad=False, so the L1 term can never
                            # retain or restore an old computation graph.
                            if hv is not None:
                                self._accumulate_hopfield_value_stats(hv, value_code, value_pattern, seg_has_tokens)
                                if hopfield_l1_train_sum is not None and value_code.requires_grad:
                                    l1_seg = value_code.abs().mean(dim=-1)[seg_has_tokens]
                                    hopfield_l1_train_sum = hopfield_l1_train_sum + l1_seg.sum()
                                    hopfield_l1_train_count += l1_seg.numel()

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

                    # snapshot the memory state READ would consume at this
                    # boundary (eval-only forgetting probe uses these EXACT
                    # states so the measured retrieval matches what the model
                    # actually produces). Plain modes: the segment's final mem.
                    # Hopfield: only the store COUNT is needed (stored_keys/
                    # values/masks are append-only within one forward, so
                    # final_list[:n_stored] == the list as it was here; this
                    # keeps the snapshot O(1) instead of O(n_seg^2)). Gated
                    # Delta: the state matrix S and per-sample written flag.
                    if collect_segment_mems:
                        segment_mems[seg_idx] = {
                            'mem': mem_batch.detach(),
                            'n_stored': len(stored_keys) if self.use_hopfield_memory else 0,
                            'S': S.detach() if self.use_gated_delta_memory else None,
                            'written': gd_written.clone() if self.use_gated_delta_memory else None,
                        }

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

        # Nonlinear Hopfield value-code diagnostics (projected codes only).
        # Enabled-but-idle stats are explicit zeros; all values detached.
        if hv is not None:
            Dv = self.hopfield_value_proj_dim
            if hv['count'] > 0:
                n_elem = hv['count'] * Dv
                inner_loop_stats['hopfield_value_code_abs_mean'] = (hv['abs_sum'] / n_elem).detach()
                inner_loop_stats['hopfield_value_code_norm_mean'] = (hv['norm_sum'] / hv['count']).detach()
                inner_loop_stats['hopfield_value_active_frac'] = (hv['active'] / n_elem).detach()
                inner_loop_stats['hopfield_value_feature_utilization'] = hv['util_any'].float().mean().detach()
            else:
                inner_loop_stats['hopfield_value_code_abs_mean'] = torch.zeros((), device=device)
                inner_loop_stats['hopfield_value_code_norm_mean'] = torch.zeros((), device=device)
                inner_loop_stats['hopfield_value_active_frac'] = torch.zeros((), device=device)
                inner_loop_stats['hopfield_value_feature_utilization'] = torch.zeros((), device=device)
            with torch.no_grad():
                # column norms of the decoder AS USED: unit-normalized columns
                # when ReLU L1 is active (pinned at 1.0), raw nn.Linear otherwise
                dec_w = self.hopfield_value_inv_proj.weight
                if self.hopfield_value_l1_active:
                    dec_w = F.normalize(dec_w, dim=0)
                inner_loop_stats['hopfield_value_decoder_col_norm_mean'] = dec_w.norm(dim=0).mean().detach()
            if self.hopfield_value_l1_active:
                inner_loop_stats['hopfield_value_l1'] = \
                    (hv['l1_all_sum'] / max(hv['l1_all_count'], 1)).detach()
                inner_loop_stats['hopfield_value_l1_weight'] = \
                    torch.tensor(float(self.hopfield_value_l1_weight), device=device)
            if self.hopfield_value_activation == "gated":
                if hv['count'] > 0:
                    n_gate = hv['count'] * Dv
                    inner_loop_stats['hopfield_value_gate_mean'] = (hv['gate_sum'] / n_gate).detach()
                    inner_loop_stats['hopfield_value_gate_saturation'] = (hv['gate_sat'] / n_gate).detach()
                else:
                    inner_loop_stats['hopfield_value_gate_mean'] = torch.zeros((), device=device)
                    inner_loop_stats['hopfield_value_gate_saturation'] = torch.zeros((), device=device)

        qry_emb = self.model.get_input_embeddings()(query_input_ids)          # [B,Q,d]
        # ---------------------------------------------------------------- #
        # Hopfield RETRIEVE phase (if enabled; skipped entirely by the
        # linear-state read -- its one-shot mixture is what that mode replaces)
        # ---------------------------------------------------------------- #
        if self.use_hopfield_memory and self.K and hopfield_stored.any() \
                and not self.use_linear_state_read:
            # Stack stored keys and values for attention-based retrieval
            keys = torch.stack(stored_keys, dim=1)      # [B, n_stored, M*d] (Dv when value-as-key)
            values = torch.stack(stored_values, dim=1)  # [B, n_stored, Dv] (activated codes; M*d unprojected)
            mask = torch.stack(stored_masks, dim=1)      # [B, n_stored]

            if self.hopfield_retrieval_mode == "mean":
                # Average all stored value patterns with equal weight
                weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                retrieved_pattern = (weights.unsqueeze(2) * values).sum(dim=1)  # [B, Dv] value code (M*d unprojected); decoded below
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
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values).squeeze(1)  # [B, Dv] value code (M*d unprojected); decoded below
                elif self.hopfield_retrieval_mode == "softmax":
                    scores = scores.masked_fill(~mask.bool(), float('-inf'))
                    scores = F.softmax(scores, dim=-1)
                    probs = scores  # [B, n_stored]
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values).squeeze(1)  # [B, Dv] value code (M*d unprojected); decoded below
                elif self.hopfield_retrieval_mode == "beta_softmax":
                    scores = scores * self.hopfield_beta
                    scores = scores.masked_fill(~mask.bool(), float('-inf'))
                    scores = F.softmax(scores, dim=-1)
                    probs = scores  # [B, n_stored]
                    retrieved_pattern = torch.bmm(scores.unsqueeze(1), values).squeeze(1)  # [B, Dv] value code (M*d unprojected); decoded below
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
                # decode the retrieved activated code (unit-normalized decoder
                # columns when ReLU L1 is active -- same helper as the
                # forgetting probe path)
                retrieved_pattern = self._decode_hopfield_value(retrieved_pattern)

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

        if self.use_linear_state_read:
            # Linear-state READ (TOKENWISE_READ.md): the bank is folded into
            # per-head states and injected per layer by the block hooks; the
            # read sequence drops the memory prefix. Nothing stored -> zero
            # state -> zero bias (per sample; no fallback prefix in this mode).
            if self.K and hopfield_stored.any():
                values_stack = torch.stack(stored_values, dim=1)           # [B, c, M*d] raw
                masks_stack = torch.stack(stored_masks, dim=1).bool()      # [B, c]
            else:
                values_stack = mem_batch_initial.view(B, 1, -1)
                masks_stack = torch.zeros(B, 1, dtype=torch.bool, device=device)
            linear_state = self._build_linear_state(values_stack, masks_stack)
            stats_acc = {}
            logits_q = self._linear_state_read_logits(
                qry_emb, query_input_ids, labels, read_st_batch, read_end_batch,
                linear_state, collect_stats=True,
                collect_cross_talk=not self.training, stats=stats_acc)
            self._drain_read_state_stats({'stats': stats_acc}, inner_loop_stats, device)
        else:
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
        if collect_segment_mems:
            output['segment_mems'] = segment_mems if segment_mems is not None else []
            if self.use_hopfield_memory:
                # final append-only store; a boundary snapshot is the first
                # n_stored entries (see the snapshot block in the segment loop).
                # Linear-state STORE appends None keys (no segment-key pass).
                output['stored_keys'] = [k.detach() if k is not None else None for k in stored_keys]
                output['stored_values'] = [v.detach() for v in stored_values]
                output['stored_masks'] = [m.detach() for m in stored_masks]
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
        # ReLU L1 sparsity on the value codes: a pure OUTER loss (it never
        # enters the K-step inner update), added after the mutually exclusive
        # target/recon loss assembly above. Only graph-retained segments
        # contribute (see the STORE accumulation -- BPTT-detached codes cannot
        # re-enter the graph); the mean runs over exactly the valid
        # sample-segments that contributed, so padded/empty ones never inflate
        # the denominator. With labels absent, forward returned above: the
        # detached diagnostics (hopfield_value_l1) are still reported, but no
        # standalone loss is invented.
        if hopfield_l1_train_sum is not None and hopfield_l1_train_count > 0:
            l_sparse = hopfield_l1_train_sum / hopfield_l1_train_count
            combined_loss = combined_loss + self.hopfield_value_l1_weight * l_sparse
        output['loss'] = combined_loss

        # Conditional composition metric: evaluate each constituent key on
        # exactly the same written context.  This is evaluation-only and runs
        # two complete no-grad forwards rather than retaining WRITE graphs or
        # changing the train-time objective.  The recursive calls omit these
        # auxiliary tensors, so they cannot recurse again.
        if (not self.training and labels is not None and not collect_segment_mems
                and single_query_input_ids is not None and single_labels is not None):
            single_predictions = []
            with torch.no_grad():
                for query_idx in range(single_query_input_ids.size(1)):
                    single_out = self.forward(
                        {
                            'context_input_ids': context_input_ids,
                            'query_input_ids': single_query_input_ids[:, query_idx, :],
                        },
                        labels=single_labels[:, query_idx, :],
                    )
                    single_predictions.append(single_out['predictions'])
            output['single_predictions'] = torch.stack(single_predictions, dim=1)
        return output
