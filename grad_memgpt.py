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
                 hopfield_dim=0,
                 hopfield_proj_freeze=True,
                 hopfield_reset_interval=None,
                 concat_hopfield_memory=False,
                 hopfield_grad_through_retrieval=False,
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
            hopfield_dim: int, expanded dimension for Hopfield memory (0 = no expansion)
            hopfield_proj_freeze: bool, freeze random projection weights
            hopfield_reset_interval: int, reset Hopfield memory every N steps (None = never)
            concat_hopfield_memory: bool, concatenate Hopfield retrieved memory with original memory
            hopfield_grad_through_retrieval: bool, backprop through retrieval step
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
        self.hopfield_dim = hopfield_dim
        self.hopfield_proj_freeze = hopfield_proj_freeze
        self.hopfield_reset_interval = hopfield_reset_interval
        self.concat_hopfield_memory = concat_hopfield_memory
        self.hopfield_grad_through_retrieval = hopfield_grad_through_retrieval

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
        self.hopfield_dim = getattr(config, "hopfield_dim", 0)
        self.hopfield_proj_freeze = getattr(config, "hopfield_proj_freeze", True)
        self.hopfield_reset_interval = getattr(config, "hopfield_reset_interval", None)
        self.concat_hopfield_memory = getattr(config, "concat_hopfield_memory", False)
        self.hopfield_grad_through_retrieval = getattr(config, "hopfield_grad_through_retrieval", False)

        if self.use_hopfield_memory:
            effective_dim = self.hopfield_dim if self.hopfield_dim > 0 else n_embd

            # Random projection layer for dimension expansion
            if self.hopfield_dim > 0:
                self.rand_proj = nn.Linear(n_embd, self.hopfield_dim, bias=False)
                if self.hopfield_proj_freeze:
                    with torch.no_grad():
                        nn.init.orthogonal_(self.rand_proj.weight)
                    self.rand_proj.weight.requires_grad = False

                # Projection for retrieving back to original dimension
                self.rand_proj_back = nn.Linear(self.hopfield_dim, n_embd, bias=False)
                with torch.no_grad():
                    nn.init.eye_(self.rand_proj_back.weight)
            else:
                self.rand_proj = None
                self.rand_proj_back = None

            # Hopfield weight matrix (Hebbian-style storage)
            self.W_hopfield = nn.Parameter(torch.zeros(effective_dim, effective_dim))

            # Track steps for reset interval
            self._hopfield_step_counter = 0
        else:
            self.rand_proj = None
            self.rand_proj_back = None
            self.W_hopfield = None
            self._hopfield_step_counter = 0

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

    def _forward_full_attn(self, inputs_embeds, output_hidden_states=True):
        """
        Forward pass with full (non-causal) attention.
        Used for query compression in Hopfield memory retrieval.
        """
        # Get the underlying transformer
        hidden_states = inputs_embeds

        # Manually run through transformer layers with full attention
        for layer in self.model.transformer.h:
            # Create causal mask (full attention = no mask)
            # shape: (1, 1, seq_len, seq_len) with all True = attend to all
            seq_len = hidden_states.size(1)
            attn_mask = torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

            # LayerNorm + Attention
            ln_1_out = layer.ln_1(hidden_states)
            attn_output, _ = layer.attn(ln_1_out, attn_mask=attn_mask)
            hidden_states = hidden_states + attn_output

            # LayerNorm + MLP
            ln_2_out = layer.ln_2(hidden_states)
            mlp_output = layer.mlp(ln_2_out)
            hidden_states = hidden_states + mlp_output

        # Final layer norm
        hidden_states = self.model.transformer.ln_f(hidden_states)

        # Project to vocabulary if needed (but we only need hidden states here)
        # Return as a mock output object with last_hidden_state
        class FakeOutput:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states

        return FakeOutput(hidden_states)

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
        inner_loss = torch.tensor(0.0, device=device)

        # actual model inputs starts after mem tokens and ctrl tokens
        mem_offset = self.n_mem_tokens + self.n_ctrl_tokens * 2

        # ctrl tokens
        if self.n_ctrl_tokens > 0:
            write_st_batch = self.write_st.unsqueeze(0).expand(B, -1, -1)
            write_end_batch = self.write_end.unsqueeze(0).expand(B, -1, -1)
            read_st_batch = self.read_st.unsqueeze(0).expand(B, -1, -1)
            read_end_batch = self.read_end.unsqueeze(0).expand(B, -1, -1)

        # make a copy of the memory that we'll update K times, manage gradients:
        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]
        mem_batch_initial = mem_batch.clone()

        # per-sample params for mem_proj:
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

        opt_state = {}                       # moments for stateless Adam
        inner_loop_stats = {'inner_grad_norm_mean': torch.tensor(0.0, device=device),
                            'inner_grad_norm_max': torch.tensor(-1.0, device=device),
                            'inner_grad_norm_min': torch.tensor(1e06, device=device)}
        # ---------------------------------------------------------------- #
        # 1.  INNER loop on context. WRITE context to mem.
        # ---------------------------------------------------------------- #
        if self.K and context_input_ids.ne(pad_id).any():
            # re‑enable autograd even if outer context is `no_grad`
            with torch.enable_grad():
                # build ctx embedding once, then reuse it with updated mem
                ctx_emb = self.model.get_input_embeddings()(context_input_ids)      # [B,S,d]
                # lm labels: reconstructing the context, last mem/ctrl token predicts the first token of the context
                lm_labels = context_input_ids.clone()
                lm_labels[lm_labels == pad_id] = -100
                # loss mask
                mask = (lm_labels != -100)
                # n real tokens per sample
                seq_len = mask.sum(dim=1).clamp_min(1)

                # pad to multiple of 32 for compatibility with JVP Flash Attention
                if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
                    pad_list = [0, -(ctx_emb.size(1) + mem_offset) % 32, 0, 0]
                    mask = F.pad(mask, pad_list, "constant", 0)
                    lm_labels = F.pad(lm_labels, pad_list, "constant", -100)
                    ctx_emb = F.pad(ctx_emb, [0, 0] + pad_list, "constant", 0)

                for k in range(self.K):
                    if self.mem_proj_mode == 'none':
                        mem_inp = mem_batch
                    elif self.mem_proj_mode == 'proj':
                        mem_inp = self.mem_proj(mem_batch)
                    else:  # per-sample
                        mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

                    if self.n_ctrl_tokens > 0:
                        # add params that can control write operation to mem in inner loop
                        x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch, ctx_emb], dim=1)
                    else:
                        x_ctx = torch.cat([mem_inp, ctx_emb], dim=1)    # [B,M+S,d]

                    if self.use_write_head:
                        # we do not need to compute read memory head logits here, we need only last hidden state
                        # to get write head logits
                        outs = get_backbone(self.model)(inputs_embeds=x_ctx, return_dict=True)
                        h = outs.last_hidden_state                     # [B,M+S,V]
                        h = h[:, mem_offset-1:, :]                     # [B,S,V]
                        logits = self.write_head(h)
                        del h
                    else:
                        outs = self.model(inputs_embeds=x_ctx, return_dict=True)
                        logits = outs.logits                           # [B,M+S,V]
                        logits = logits[:, mem_offset-1:, :]           # [B,S,V]

                    inner_loss = nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        lm_labels.reshape(-1),
                        ignore_index=-100,
                        reduction='none',
                    ).view(B, -1)
                    # per_sample losses, make per-sample inner loss invariant to batch size B:
                    # g_i = d inner_loss / d mem_i = d inner_loss_i / d mem_i
                    inner_loss = (inner_loss * mask).sum(1) / seq_len
                    inner_loss = inner_loss.sum()
                    del outs, logits

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

                    # track inner grad norm (todo: move to _opt_step?, currently we compute g_norm twice)
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

        if self.K:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / self.K
            # log average inner loss
            inner_loop_stats['inner_loss'] = inner_loss.detach() / B
        # mem_batch: [B,M,d]
        mem_norm = mem_batch.norm(dim=(1, 2)).detach()  # B
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()
        inner_loop_stats['mem_norm_min'] = mem_norm.min()
        # log how mem has changed from initial state to state after inner loop
        detla_mem_norm = (mem_batch - mem_batch_initial).detach().norm(dim=(1, 2))
        inner_loop_stats['delta_mem_norm_mean'] = detla_mem_norm.mean()
        inner_loop_stats['delta_mem_norm_max'] = detla_mem_norm.max()
        inner_loop_stats['delta_mem_norm_min'] = detla_mem_norm.min()

        del ctx_emb, lm_labels

        qry_emb = self.model.get_input_embeddings()(query_input_ids)          # [B,Q,d]
        # ---------------------------------------------------------------- #
        # Hopfield-like external memory (if enabled)
        # ---------------------------------------------------------------- #
        if self.use_hopfield_memory and self.K and context_input_ids.ne(pad_id).any():
            # Step 2: Store gradmem tokens in Hopfield memory (Hebbian update)
            # gradmem_sum: [B, d]
            gradmem_sum = mem_batch.sum(dim=1)

            # Project to expanded dimension if needed
            if self.rand_proj is not None:
                gradmem_expanded = self.rand_proj(gradmem_sum)  # [B, d_expanded]
            else:
                gradmem_expanded = gradmem_sum

            # Hebbian update: W += outer_product(gradmem, gradmem)
            # For each sample: outer = v_i @ v_i.T, accumulate over batch
            hopfield_update = torch.bmm(gradmem_expanded.unsqueeze(2), gradmem_expanded.unsqueeze(1))  # [B, d_exp, d_exp]
            with torch.no_grad():
                self.W_hopfield.data += hopfield_update.sum(0)

            # Increment step counter for reset interval
            self._hopfield_step_counter += 1
            if self.hopfield_reset_interval is not None and self._hopfield_step_counter >= self.hopfield_reset_interval:
                with torch.no_grad():
                    self.W_hopfield.data.zero_()
                self._hopfield_step_counter = 0

            # Step 3: Query compression - forward pass with initial memory + query (full attention)
            # Build input: [read_st][mem_init][read_end][query]
            if self.n_ctrl_tokens > 0:
                x_qry_comp = torch.cat([read_st_batch, mem_batch_initial, read_end_batch, qry_emb], dim=1)
            else:
                x_qry_comp = torch.cat([mem_batch_initial, qry_emb], dim=1)

            # Forward with full attention
            outs_q = self._forward_full_attn(x_qry_comp)
            # Extract query memory tokens (first n_mem_tokens positions)
            query_memory_tokens = outs_q.last_hidden_state[:, :self.n_mem_tokens, :]  # [B, M, d]

            # Step 4: Hopfield retrieval
            # Project query memory to expanded dimension
            if self.rand_proj is not None:
                query_expanded = self.rand_proj(query_memory_tokens)  # [B, M, d_expanded]
            else:
                query_expanded = query_memory_tokens

            # Retrieve: assoc = query @ W_hopfield
            retrieved_expanded = torch.bmm(
                query_expanded.view(B * self.n_mem_tokens, -1),
                self.W_hopfield.data
            ).view(B, self.n_mem_tokens, -1)  # [B, M, d_expanded]

            # Project back to original dimension
            if self.rand_proj_back is not None:
                assoc_memory = self.rand_proj_back(retrieved_expanded)  # [B, M, d]
            else:
                assoc_memory = retrieved_expanded

            # Optionally concatenate with original memory
            if self.concat_hopfield_memory:
                # mem_inp will be concatenated with assoc_memory later
                hopfield_mem = assoc_memory
            else:
                # Use only Hopfield-retrieved memory
                mem_batch = assoc_memory

        # ---------------------------------------------------------------- #
        # 2.  READ phase – compute outer loss on target predictions based on query, read from mem
        # ---------------------------------------------------------------- #
        # Determine memory input for READ phase
        if self.use_hopfield_memory and self.concat_hopfield_memory and 'hopfield_mem' in dir():
            # Concatenate Hopfield-retrieved memory with original memory
            if self.mem_proj_mode == "none":
                mem_for_concat = mem_batch
            elif self.mem_proj_mode == "proj":
                mem_for_concat = self.mem_proj(mem_batch)
            else:  # "per_sample"
                mem_for_concat = self._apply_linear(mem_batch, W_batch, b_batch)
            mem_inp = torch.cat([hopfield_mem, mem_for_concat], dim=1)
            # Adjust mem_offset for concatenated memory (need to handle both memory sections)
            read_mem_offset = self.n_mem_tokens * 2 + self.n_ctrl_tokens * 2
        else:
            # Original behavior
            if self.mem_proj_mode == "none":
                mem_inp = mem_batch
            elif self.mem_proj_mode == "proj":
                mem_inp = self.mem_proj(mem_batch)
            else:  # "per_sample"
                mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)
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
