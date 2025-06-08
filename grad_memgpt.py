import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig


class GradMemGPT(PreTrainedModel):
    """
    Transformer-decoder backbone + writable prefix memory (n_mem_tokens x d).
    """
    config_class = AutoConfig

    def __init__(self, config, n_mem_tokens=8, K=3, lr=2e-02, use_adam=True, grad_mode="none", n_ctrl_tokens=0,
                 inner_clip_value=None, inner_clip_norm=None):
        """
        grad_mode: none, first, second
        none: stop grad in inner update. Outer optimizer ignores mem pathway.
            Initial params of self.mem are never trained. Per-sample memory is updated in inner loop.
        first: first-order update. Outer grads flow to mem, but ignore Hessian term (Straight-Through / FOMAML).
            Only outer loop gradients update self.mem, inner loop (second-order) gradients are ignored:
            self.mem.grad = mem_batch.grad.sum(0)
        second: second-order update. Full MAML. Outer grads include second-order term via a differentiable inner step.

        inner loop: [write_ctrl][mem][context]
        outer loop: [read_ctrl][mem][query][target]

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
        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.config, 'n_embd', self.config.hidden_size)
        self.mem = nn.Parameter(torch.randn(n_mem_tokens, n_embd) * 0.02)
        # read/write control parameters (shape = n_ctrl_tokens × d)
        if n_ctrl_tokens > 0:
            self.write_ctrl = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.read_ctrl = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
        self.tie_weights()
        self.main_input_name = "input_ids"

    def floating_point_ops(self, inputs):
        # dummy method to satisfy base class and it's invocation by trainer:
        # Trainer supposes that `inputs`` is a tensor, not dict.
        return 0

    def tie_weights(self):
        self.model.tie_weights()

    # def _adam_step(self, p, g, state):
    #     beta1, beta2, eps = 0.9, 0.999, 1e-8
    #     if not state:
    #         state["m"] = torch.zeros_like(p)
    #         state["v"] = torch.zeros_like(p)
    #     m, v = state["m"], state["v"]
    #     m.mul_(beta1).add_(g, alpha=1-beta1)
    #     v.mul_(beta2).addcmul_(g, g, value=1-beta2)
    #     m_hat = m / (1-beta1)
    #     v_hat = v / (1-beta2)
    #     return p - self.lr * m_hat / (v_hat.sqrt() + eps)

    def _adam_step(self, p, g, state, step_idx, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip_value=10.0):
        """
        Functional Adam:
        - no in-place math on graph-tracked tensors
        - buffers detached, keep them in state dict
        - bias-correction uses step_idx
        - gradient norm is clipped
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

    def _sgd_step(self, p, g, clip_value=None, clip_norm=None):
        """
        Stateless SGD with optional gradient clipping.

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
            g_norm = g.norm(dim=[1, 2], keepdim=True)                 # (B,1,1)
            scale = clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > clip_norm, g * scale, g)

        return p - self.lr * g

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
            write_ctrl_batch = self.write_ctrl.unsqueeze(0).expand(B, -1, -1).requires_grad_(True)
            read_ctrl_batch = self.read_ctrl.unsqueeze(0).expand(B, -1, -1).requires_grad_(True)

        # make a copy of the memory that we'll update K times, manage gradients:
        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B,M,d]
        if self.grad_mode == "none":
            mem_batch = mem_batch.detach().requires_grad_(True)
        else:
            mem_batch = mem_batch.requires_grad_(True)

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
                ctx_emb = self.model.get_input_embeddings()(context_input_ids)    # [B,S,d]
                for inner_step in range(self.K):
                    x_ctx = torch.cat([mem_batch, ctx_emb], dim=1)                # [B,M+S,d]
                    # add params that can control write operation to mem in inner loop
                    if self.n_ctrl_tokens > 0:
                        x_ctx = torch.cat([write_ctrl_batch, x_ctx], dim=1)

                    logits = self.model(inputs_embeds=x_ctx).logits               # [B,M+S,V]
                    logits = logits[:, self.n_mem_tokens+self.n_ctrl_tokens:, :]  # [B,S,V]
                    # shift‑left LM loss, ignore mem tokens + padding
                    lm_labels = context_input_ids.clone()
                    lm_labels[lm_labels == pad_id] = -100
                    inner_loss = nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        lm_labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )

                    # parameter update
                    if self.grad_mode in ['none', 'first']:
                        g = torch.autograd.grad(inner_loss, mem_batch, create_graph=False)[0]
                    elif self.grad_mode == 'second':
                        g = torch.autograd.grad(inner_loss, mem_batch, create_graph=True)[0]

                    # track inner grad norm (todo: move to _opt_step?, currently we compute g_norm twice)
                    g_norm = g.reshape(B, -1).norm(dim=1).detach()
                    inner_loop_stats['inner_grad_norm_mean'] += g_norm.mean()
                    inner_loop_stats['inner_grad_norm_max'] = max(inner_loop_stats['inner_grad_norm_max'], g_norm.max())
                    inner_loop_stats['inner_grad_norm_min'] = min(inner_loop_stats['inner_grad_norm_min'], g_norm.min())

                    if self.use_adam:
                        mem_batch = self._adam_step(mem_batch, g, opt_state, inner_step + 1, self.lr)
                    else:
                        mem_batch = self._sgd_step(mem_batch, g,
                                                   clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)

                    if self.grad_mode in ['none']:
                        mem_batch = mem_batch.detach().requires_grad_(True)
                    elif self.grad_mode in ['first', 'second']:
                        pass  # do nothing, keep gradients flow

        if self.K:
            inner_loop_stats['inner_grad_norm_mean'] = inner_loop_stats['inner_grad_norm_mean'] / self.K
            inner_loop_stats['inner_loss'] = inner_loss
        # mem_batch: [B,M,d]
        mem_norm = mem_batch.norm(dim=[1, 2]).detach()  # B
        inner_loop_stats['mem_norm_mean'] = mem_norm.mean()
        inner_loop_stats['mem_norm_max'] = mem_norm.max()

        # ---------------------------------------------------------------- #
        # 2.  READ phase – compute outer loss on target predictions based on query, read from mem
        # ---------------------------------------------------------------- #
        qry_emb = self.model.get_input_embeddings()(query_input_ids)        # [B,Q,d]
        x_qry = torch.cat([mem_batch, qry_emb], dim=1)                      # [B,M+Q,d]
        # add params that can control read operation from mem
        if self.n_ctrl_tokens > 0:
            x_qry = torch.cat([read_ctrl_batch, x_qry], dim=1)
        logits_q = self.model(inputs_embeds=x_qry).logits                   # [B,M+Q,V]
        logits_q = logits_q[:, self.n_mem_tokens+self.n_ctrl_tokens:, :]    # [B,Q,V]

        output = {'predictions': logits_q, 'inner_loop_stats': inner_loop_stats}
        if return_mem:
            output['mem'] = mem_batch

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
