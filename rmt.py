import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig


class RMT2Segm(PreTrainedModel):
    """
    RMT-like model with 2 segments only: 1. context 2. query -> target.
    """
    config_class = AutoConfig

    def __init__(self, config, n_mem_tokens=8, n_ctrl_tokens=0, use_mem_proj=False, mem_proj_mode="none"):
        """
        mem_proj_mode: "none" | "proj"
        none: no linear projection of mem
        proj: shared nn.Linear, is used only in write operation (todo: check if it is better to use at read as well)

        1st segment: [write_st][mem][write_end][context][write_st][mem]
        2nd segment: [read_st][mem][read_end][query][target]

        write_st/write_end/read_st/read_end are parameters aka prompts, that can be used by model to control
            the write/read operation.
        n_ctrl_tokens = 1 means that [write_st] is a single token.

        mem is an output from 1st segment
        """
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_config(config)
        self.n_mem_tokens = n_mem_tokens
        self.n_ctrl_tokens = n_ctrl_tokens
        self.use_mem_proj = use_mem_proj  # defaults to mem_proj_mode == "proj"
        if mem_proj_mode is None:
            mem_proj_mode = "proj" if use_mem_proj else "none"
        self.mem_proj_mode = mem_proj_mode

        # check args
        assert mem_proj_mode in ["none", "proj"]
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"

        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.config, 'n_embd', self.config.hidden_size)
        # self.mem initial states.
        self.mem = nn.Parameter(torch.randn(n_mem_tokens, n_embd) * 0.02)

        # optional mem projection linear layer
        if self.mem_proj_mode != "none":
            self.mem_proj = nn.Linear(n_embd, n_embd, bias=True)

        # optional read/write control parameters (shape = n_ctrl_tokens × d)
        if n_ctrl_tokens > 0:
            # write ctrl tokens can be trained only by outer loop and only if grads flow through inner loop ("second")
            self.write_st = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.write_end = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.read_st = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)
            self.read_end = nn.Parameter(torch.randn(n_ctrl_tokens, n_embd) * 0.02)

        self.tie_weights()
        self.main_input_name = "input_ids"

    def floating_point_ops(self, inputs):
        # dummy method to satisfy base class and it's invocation by trainer:
        # Trainer supposes that `inputs`` is a tensor, not dict.
        return 0

    def tie_weights(self):
        self.model.tie_weights()

    def _prepare_inputs(self, input_ids):
        pass

    def forward(self, input_ids, labels=None, return_mem=False):
        # context_input_ids : B × S   (segments only, each ends with `|`)
        # query_input_ids   : B × Q   (e.g.  "?!K:V!|") i.e. the last segment
        # labels            : B × Q   (‑100 everywhere except the target tokens (V!|))

        """
        All tensors already padded to the same length in the datacollator.
        """
        context_input_ids = input_ids['context_input_ids']
        query_input_ids = input_ids['query_input_ids']

        B = context_input_ids.size(0)
        device = context_input_ids.device
        pad_id = self.model.config.pad_token_id

        # ctrl tokens
        if self.n_ctrl_tokens > 0:
            write_st_batch = self.write_st.unsqueeze(0).expand(B, -1, -1)
            write_end_batch = self.write_end.unsqueeze(0).expand(B, -1, -1)
            read_st_batch = self.read_st.unsqueeze(0).expand(B, -1, -1)
            read_end_batch = self.read_end.unsqueeze(0).expand(B, -1, -1)

        # prepare memory states for each sample
        mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1)  # [B,M,d]

        # ---------------------------------------------------------------- #
        # 1.  Process 1st segment: context. WRITE context to mem.
        # ---------------------------------------------------------------- #
        ctx_emb = self.model.get_input_embeddings()(context_input_ids)      # [B,S,d]

        # attention masks
        ctx_mask = (context_input_ids != pad_id).to(dtype=torch.long)
        mem_mask = torch.ones(B, self.n_mem_tokens, dtype=torch.long, device=device)

        mem_inp_write = self.mem_proj(mem_batch) if self.mem_proj_mode == "proj" else mem_batch

        if self.n_ctrl_tokens > 0:
            # add params that can control write operation to mem in inner loop
            x_ctx = torch.cat([write_st_batch, mem_inp_write, write_end_batch,
                               ctx_emb,
                               write_st_batch, mem_inp_write], dim=1)
            # attention masks
            ctrl_mask = torch.ones(B, self.n_ctrl_tokens, dtype=torch.long, device=device)
            attn_mask = torch.cat([ctrl_mask, mem_mask, ctrl_mask,
                                   ctx_mask,
                                   ctrl_mask, mem_mask], dim=1)
        else:
            x_ctx = torch.cat([mem_inp_write, ctx_emb, mem_inp_write], dim=1)  # [B,M+S+M,d]
            # attention masks
            attn_mask = torch.cat([mem_mask, ctx_mask, mem_mask], dim=1)

        # position ids, ignore pads, mem has position ids like there is no pad between context and mem
        position_ids = attn_mask.cumsum(-1) - 1

        outs = self.model(inputs_embeds=x_ctx, position_ids=position_ids, attention_mask=attn_mask,
                          output_hidden_states=True, return_dict=True)
        h = outs.hidden_states[-1]             # [B,M+S+M,d]
        # extract mem tokens states
        mem_batch = h[:, -self.n_mem_tokens:]  # [B,M,d]

        stats = {}
        mem_norm = mem_batch.norm(dim=(1, 2)).detach()  # B
        stats['mem_norm_mean'] = mem_norm.mean()
        stats['mem_norm_max'] = mem_norm.max()

        # ---------------------------------------------------------------- #
        # 2.  Process 2nd segment: query -> target. READ phase.
        # ---------------------------------------------------------------- #
        qry_emb = self.model.get_input_embeddings()(query_input_ids)  # [B,Q,d]
        qry_mask = (query_input_ids != pad_id).to(dtype=torch.long)

        # TODO: check if mem_proj needed here, seems its ok to not use it
        # if self.mem_proj_mode == "none":
        #     mem_inp = mem_batch
        # elif self.mem_proj_mode == "proj":
        #     mem_inp = self.mem_proj(mem_batch)
        mem_inp = mem_batch

        if self.n_ctrl_tokens > 0:
            # add params that can control read operation from mem
            x_qry = torch.cat([read_st_batch, mem_inp, read_end_batch, qry_emb], dim=1)
            qry_attn_mask = torch.cat([ctrl_mask, mem_mask, ctrl_mask, qry_mask], dim=1)
        else:
            x_qry = torch.cat([mem_inp, qry_emb], dim=1)            # [B,M+Q,d]
            qry_attn_mask = torch.cat([mem_mask, qry_mask], dim=1)

        logits_q = self.model(inputs_embeds=x_qry, attention_mask=qry_attn_mask).logits  # [B,M+Q,V]
        logits_q = logits_q[:, self.n_mem_tokens+self.n_ctrl_tokens*2:, :]               # [B,Q,V]

        output = {'predictions': logits_q, 'inner_loop_stats': stats}
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
