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
                 freeze_backbone=False,
                 use_gradient_checkpointing=False,
                 attn_implementation="eager",
                 add_inner_loss_to_outer=False,
                 inner_loss_weight=None,
                 use_hopfield_memory=False,
                 hopfield_n_segments=1,
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
            freeze_backbone: bool, freeze backbone weights (READ+WRITE), except LoRA/write head/mem proj
            use_gradient_checkpointing: bool, turn on gradient checkpointing supported by HF models
            add_inner_loss_to_outer: bool, outer loss = target_loss + inner_loss_weight * inner_loss_mean
            inner_loss_weight: float, weight of inner loss in combined loss
            use_hopfield_memory: bool, enable Hopfield-like external memory
            hopfield_n_segments: int, number of segments to split context into for Hopfield storage (1 = whole context)
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

        # Validate mem_proj_mode settings
        assert mem_proj_mode in ["none", "proj", "per_sample"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"


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
        self.freeze_backbone = getattr(config, "freeze_backbone", False)
        if self.use_write_lora:
            self._init_write_lora()
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

        # Hopfield-like external memory
        self.use_hopfield_memory = getattr(config, "use_hopfield_memory", False)
        self.hopfield_n_segments = getattr(config, "hopfield_n_segments", 1)

        if self.use_hopfield_memory:
            if self.hopfield_n_segments < 1:
                raise ValueError(f"hopfield_n_segments must be >= 1, got {self.hopfield_n_segments}")

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

    @contextmanager
    def _full_attention(self):
        """Context manager that enables full (non-causal) attention.

        For models with hardcoded causal bias buffers (e.g. GPT-2 eager attention),
        temporarily overrides those buffers to enable bidirectional attention.
        For SDPA and custom attention kernels (jvp_flash, hvp_manual, hvp_semi_manual),
        a 4D all-zeros attention mask suffices (is_causal=False when attention_mask is not None).
        """
        backbone = get_backbone(self.model)
        saved = {}

        has_eager_causal_bias = (
            self.attn_implementation == 'eager'
            and any(
                hasattr(m, 'bias') and isinstance(getattr(m, 'bias', None), torch.Tensor)
                and getattr(m, 'bias').dim() == 4 and getattr(m, 'bias').dtype == torch.bool
                for m in backbone.modules()
            )
        )

        if has_eager_causal_bias:
            for module in backbone.modules():
                if hasattr(module, 'bias') and isinstance(getattr(module, 'bias', None), torch.Tensor):
                    bias = getattr(module, 'bias')
                    if bias.dim() == 4 and bias.dtype == torch.bool:
                        saved[id(module)] = bias.clone()
                        module.bias.fill_(True)
        try:
            yield
        finally:
            for module in backbone.modules():
                if id(module) in saved and hasattr(module, 'bias'):
                    module.bias.copy_(saved[id(module)])

    def _forward_full_attn(self, inputs_embeds):
        """Forward pass with full (non-causal) attention using the model's standard interface.

        Uses a 4D all-zeros attention mask (0.0 = attend everywhere) to disable causal masking,
        and overrides GPT-2-style hardcoded causal bias buffers when using eager attention.
        Correctly handles JVP flash attention sequence length requirements.
        Architecture-agnostic: works with GPT-2, LLaMA, GPT-NeoX, etc.
        """
        B, S, D = inputs_embeds.shape

        # Pad to multiple of 32 for compatibility with JVP Flash Attention
        if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
            pad_len = -S % 32
            if pad_len > 0:
                # Pad embedding dimension, then sequence dimension
                inputs_embeds = F.pad(inputs_embeds, [0, 0, 0, pad_len], "constant", 0)

        S_padded = inputs_embeds.size(1)

        # 4D all-zeros float mask: 0.0 = attend everywhere (no masking)
        # For padded sequences, mask out padding positions so real tokens don't attend to them
        full_mask = torch.zeros(B, 1, S_padded, S_padded, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if S_padded > S:
            full_mask[:, :, :, S:] = float('-inf')  # don't attend to padding keys
            full_mask[:, :, S:, :] = float('-inf')   # padding queries produce zero outputs

        backbone = get_backbone(self.model)
        with self._full_attention():
            outs = backbone(inputs_embeds=inputs_embeds, attention_mask=full_mask, return_dict=True)

        return outs

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

        # mem_batch_initial is always self.mem — used for Hopfield keys and as reset point
        mem_batch_initial = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]

        # per-sample params for mem_proj (initialized from outer loop params, reset per segment)
        if self.mem_proj_mode == "per_sample":
            W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()
            b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()

        n_segments = self.hopfield_n_segments if self.use_hopfield_memory else 1

        # Per-sample W_hopfield (fresh each forward call = reset per batch)
        if self.use_hopfield_memory:
            n_embd = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
            hopfield_pattern_dim = self.n_mem_tokens * n_embd
            W_hopfield = torch.zeros(B, hopfield_pattern_dim, hopfield_pattern_dim, device=device)
            hopfield_stored = torch.zeros(B, dtype=torch.bool, device=device)

        last_inner_loss = torch.tensor(0.0, device=device)
        total_inner_steps = 0
        n_segments_with_context = 0

        inner_loop_stats = {'inner_grad_norm_mean': torch.tensor(0.0, device=device),
                            'inner_grad_norm_max': torch.tensor(-1.0, device=device),
                            'inner_grad_norm_min': torch.tensor(1e06, device=device)}

        # Track last segment's mem_batch for stats and as fallback for READ phase
        last_mem_batch = mem_batch_initial

        # ---------------------------------------------------------------- #
        # 1.  INNER loop on context. WRITE context to mem, segment by segment.
        # ---------------------------------------------------------------- #
        ctx_emb = None
        if self.K and context_input_ids.ne(pad_id).any():
            # re‑enable autograd even if outer context is `no_grad`
            with torch.enable_grad():
                # build ctx embedding once
                ctx_emb = self.model.get_input_embeddings()(context_input_ids)      # [B,S,d]
                # lm labels: reconstructing the context, last mem/ctrl token predicts the first token of the context
                lm_labels = context_input_ids.clone()
                lm_labels[lm_labels == pad_id] = -100
                # loss mask
                mask = (lm_labels != -100)

                # Split context into segments (ceil-padded to equal size)
                S = ctx_emb.size(1)
                segment_size = (S + n_segments - 1) // n_segments  # ceil division
                pad_len = segment_size * n_segments - S

                if pad_len > 0:
                    ctx_emb = F.pad(ctx_emb, [0, 0, 0, pad_len], "constant", 0)
                    mask = F.pad(mask, [0, pad_len], "constant", 0)
                    lm_labels = F.pad(lm_labels, [0, pad_len], "constant", -100)

                for seg_idx in range(n_segments):
                    seg_start = seg_idx * segment_size
                    seg_end = seg_start + segment_size
                    seg_emb = ctx_emb[:, seg_start:seg_end, :]
                    seg_mask = mask[:, seg_start:seg_end]
                    seg_labels = lm_labels[:, seg_start:seg_end]

                    # Per-sample: does this segment have any real tokens?
                    seg_has_tokens = seg_mask.any(dim=1)  # [B]

                    if not seg_has_tokens.any():
                        continue

                    seg_seq_len = seg_mask.sum(dim=1).clamp_min(1)  # [B]

                    # Pad segment for JVP Flash Attention compatibility
                    if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
                        seg_pad_len = -(seg_emb.size(1) + mem_offset) % 32
                        seg_pad_list = [0, seg_pad_len, 0, 0]
                        cur_seg_mask = F.pad(seg_mask, seg_pad_list, "constant", 0)
                        cur_seg_labels = F.pad(seg_labels, seg_pad_list, "constant", -100)
                        cur_seg_emb = F.pad(seg_emb, [0, 0] + seg_pad_list, "constant", 0)
                    else:
                        cur_seg_emb = seg_emb
                        cur_seg_mask = seg_mask
                        cur_seg_labels = seg_labels

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
                            b_batch = b_batch.requires_grad_(True)

                    # Reset Adam state per segment
                    opt_state = {}

                    for k in range(self.K):
                        if self.mem_proj_mode == 'none':
                            mem_inp = mem_batch
                        elif self.mem_proj_mode == 'proj':
                            mem_inp = self.mem_proj(mem_batch)
                        else:  # per-sample
                            mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

                        if self.n_ctrl_tokens > 0:
                            x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch, cur_seg_emb], dim=1)
                        else:
                            x_ctx = torch.cat([mem_inp, cur_seg_emb], dim=1)    # [B,M+seg_size,d]

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

                    # Accumulate last-step inner loss for this segment
                    last_inner_loss = last_inner_loss + inner_loss
                    n_segments_with_context += 1

                    # Per-segment Hopfield STORE (per-sample Hebbian update)
                    if self.use_hopfield_memory and seg_has_tokens.any():
                        with torch.no_grad():
                            # Key: forward-pass compressed representation of this segment
                            # Uses initial (pre-update) memory + segment with full attention
                            if self.n_ctrl_tokens > 0:
                                x_seg_comp = torch.cat([write_st_batch, mem_batch_initial, write_end_batch, seg_emb], dim=1)
                            else:
                                x_seg_comp = torch.cat([mem_batch_initial, seg_emb], dim=1)

                            with self._disable_write_lora():
                                outs_seg = self._forward_full_attn(x_seg_comp)
                            seg_key = outs_seg.last_hidden_state[:, :self.n_mem_tokens, :].view(B, -1)  # [B, M*d]
                            del outs_seg

                        # Value: gradient-updated mem_batch after inner loop on this segment
                        value_pattern = mem_batch.detach().view(B, -1)  # [B, M*d]
                        seg_key = seg_key.detach()

                        # L2-normalize
                        value_pattern = F.normalize(value_pattern, dim=-1)
                        seg_key = F.normalize(seg_key, dim=-1)

                        # Hebbian update: W_hopfield[b] += value[b] @ key[b].T
                        # Mask out samples where this segment is all-padding
                        update = torch.bmm(value_pattern.unsqueeze(2), seg_key.unsqueeze(1))  # [B, M*d, M*d]
                        update = update * seg_has_tokens.unsqueeze(1).unsqueeze(2).float()
                        W_hopfield = W_hopfield + update
                        hopfield_stored = hopfield_stored | seg_has_tokens

                    # Keep last segment's mem_batch for stats and READ phase fallback
                    last_mem_batch = mem_batch

        if total_inner_steps > 0:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / total_inner_steps
            if n_segments_with_context > 0:
                inner_loop_stats['inner_loss'] = (last_inner_loss / n_segments_with_context).detach() / B
        # mem stats from last segment
        mem_norm = last_mem_batch.norm(dim=(1, 2)).detach()  # B
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()
        inner_loop_stats['mem_norm_min'] = mem_norm.min()
        detla_mem_norm = (last_mem_batch - mem_batch_initial).detach().norm(dim=(1, 2))
        inner_loop_stats['delta_mem_norm_mean'] = detla_mem_norm.mean()
        inner_loop_stats['delta_mem_norm_max'] = detla_mem_norm.max()
        inner_loop_stats['delta_mem_norm_min'] = detla_mem_norm.min()

        qry_emb = self.model.get_input_embeddings()(query_input_ids)          # [B,Q,d]
        # ---------------------------------------------------------------- #
        # Hopfield RETRIEVE phase (if enabled)
        # ---------------------------------------------------------------- #
        if self.use_hopfield_memory and self.K and hopfield_stored.any():
            with torch.no_grad():
                if self.n_ctrl_tokens > 0:
                    x_qry_comp = torch.cat([read_st_batch, mem_batch_initial, read_end_batch, qry_emb], dim=1)
                else:
                    x_qry_comp = torch.cat([mem_batch_initial, qry_emb], dim=1)

                with self._disable_write_lora():
                    outs_q = self._forward_full_attn(x_qry_comp)
                query_key = outs_q.last_hidden_state[:, :self.n_mem_tokens, :].view(B, -1)  # [B, M*d]
                del outs_q

            query_key = F.normalize(query_key, dim=-1)

            # Per-sample retrieval: query_key[b] @ W_hopfield[b]
            retrieved_pattern = torch.bmm(query_key.unsqueeze(1), W_hopfield.detach()).squeeze(1)  # [B, M*d]

            # Reshape back to individual memory tokens: [B, M, d]
            assoc_memory = retrieved_pattern.view(B, self.n_mem_tokens, -1)
            last_mem_batch = assoc_memory

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

        with self._disable_write_lora():
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
        if self.add_inner_loss_to_outer and n_segments_with_context > 0:
            inner_loss_mean = (last_inner_loss / n_segments_with_context) / B
            combined_loss = target_loss + self.inner_loss_weight * inner_loss_mean
        else:
            combined_loss = target_loss
        output['loss'] = combined_loss
        return output
