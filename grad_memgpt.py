import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig


class GradMemGPT(PreTrainedModel):
    """
    Transformer-decoder backbone + writable prefix memory (n_mem_tokens x d).
    """
    config_class = AutoConfig

    def __init__(self, config, n_mem_tokens=8, K=3, lr=2e-02, use_adam=True, grad_mode="none", n_ctrl_tokens=0,
                 inner_clip_value=None, inner_clip_norm=None, use_mem_proj=False, mem_proj_mode="none",
                 use_write_head=False):
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
        self.model = AutoModelForCausalLM.from_config(config)
        self.n_mem_tokens = n_mem_tokens
        self.n_ctrl_tokens = n_ctrl_tokens
        self.K = K
        self.lr = lr
        self.use_adam = use_adam
        self.grad_mode = grad_mode
        self.inner_clip_value = inner_clip_value
        self.inner_clip_norm = inner_clip_norm
        self.use_mem_proj = use_mem_proj  # defaults to mem_proj_mode == "proj"
        if mem_proj_mode is None:
            mem_proj_mode = "proj" if use_mem_proj else "none"
        self.mem_proj_mode = mem_proj_mode
        self.use_write_head = use_write_head

        # check args
        assert mem_proj_mode in ["none", "proj", "per_sample"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"

        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.config, 'n_embd', self.config.hidden_size)
        # self.mem are inner loop per-sample params, intial states of mem (self.mem) are meta-learned
        self.mem = nn.Parameter(torch.randn(n_mem_tokens, n_embd) * 0.02)

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
        if n_ctrl_tokens > 0:
            # write ctrl tokens can be trained only by outer loop and only if grads flow through inner loop ("second")
            self.write_st = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.write_end = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.read_st = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.read_end = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)

        if self.use_write_head:
            V = self.model.config.vocab_size
            self.write_head = nn.Linear(n_embd, V, bias=False)

            if hasattr(self.model, 'get_output_embeddings'):
                head_params = self.model.get_output_embeddings().weight
            else:  # fallback to input embeddings
                head_params = self.model.get_input_embeddings().weight
            with torch.no_grad():
                self.write_head.weight.copy_(head_params.detach())

        self.tie_weights()
        self.main_input_name = "input_ids"

    def floating_point_ops(self, inputs):
        # dummy method to satisfy base class and it's invocation by trainer:
        # Trainer supposes that `inputs`` is a tensor, not dict.
        return 0

    def tie_weights(self):
        self.model.tie_weights()

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

        # ctrl tokens
        if self.n_ctrl_tokens > 0:
            write_st_batch = self.write_st.unsqueeze(0).expand(B, -1, -1)
            write_end_batch = self.write_end.unsqueeze(0).expand(B, -1, -1)
            read_st_batch = self.read_st.unsqueeze(0).expand(B, -1, -1)
            read_end_batch = self.read_end.unsqueeze(0).expand(B, -1, -1)

        # make a copy of the memory that we'll update K times, manage gradients:
        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]

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
                # shift‑left LM loss, ignore mem tokens + padding
                lm_labels = context_input_ids.clone()
                lm_labels[lm_labels == pad_id] = -100
                for inner_step in range(self.K):
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
                        x_ctx = torch.cat([mem_inp, ctx_emb], dim=1)                    # [B,M+S,d]

                    outs = self.model(inputs_embeds=x_ctx, output_hidden_states=self.use_write_head, return_dict=True)
                    if self.use_write_head:
                        h = outs.hidden_states[-1]                                      # [B,M+S,V]
                        h = h[:, self.n_mem_tokens+self.n_ctrl_tokens*2:, :]            # [B,S,V]
                        logits = self.write_head(h)
                    else:
                        logits = outs.logits                                            # [B,M+S,V]
                        logits = logits[:, self.n_mem_tokens+self.n_ctrl_tokens*2:, :]  # [B,S,V]

                    inner_loss = nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        lm_labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )

                    # get inner loop gradients
                    create_graph = (self.grad_mode == "second")
                    if self.mem_proj_mode == 'per_sample':
                        g_mem, g_W, g_b = torch.autograd.grad(inner_loss, [mem_batch, W_batch, b_batch],
                                                              create_graph=create_graph)
                    else:
                        g_mem = torch.autograd.grad(inner_loss, mem_batch, create_graph=create_graph)[0]

                    # track inner grad norm (todo: move to _opt_step?, currently we compute g_norm twice)
                    g_norm = g_mem.reshape(B, -1).norm(dim=1).detach()
                    inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
                    inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
                    inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())

                    if self.use_adam:
                        mem_batch = self._adam_step(mem_batch, g_mem, opt_state.setdefault('mem', {}),
                                                    inner_step + 1, self.lr)
                        if self.mem_proj_mode == 'per_sample':
                            W_batch = self._adam_step(W_batch, g_W, opt_state.setdefault('W', {}),
                                                      inner_step + 1, self.lr)
                            b_batch = self._adam_step(b_batch, g_b, opt_state.setdefault('b', {}),
                                                      inner_step + 1, self.lr)
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
            inner_loop_stats['inner_loss'] = inner_loss.detach()
        # mem_batch: [B,M,d]
        mem_norm = mem_batch.norm(dim=(1, 2)).detach()  # B
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()

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

        logits_q = self.model(inputs_embeds=x_qry).logits                     # [B,M+Q,V]
        logits_q = logits_q[:, self.n_mem_tokens+self.n_ctrl_tokens*2:, :]    # [B,Q,V]

        output = {'predictions': logits_q, 'inner_loop_stats': inner_loop_stats}
        if return_mem:
            output['mem'] = mem_batch
            if self.mem_proj_mode == "per_sample":
                output['W'] = W_batch
                output['b'] = b_batch

        if labels is None:
            return output

        # lm loss
        loss = nn.functional.cross_entropy(
            logits_q[:, :-1].reshape(-1, logits_q.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        output['loss'] = loss
        return output
