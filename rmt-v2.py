import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig


def get_backbone(m):
    # most HF CausalLM classes define base_model_prefix, e.g. "transformer" (GPT-2), "model" (LLaMA)
    if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
        return getattr(m, m.base_model_prefix)
    # robust fallback
    for attr in ("model", "transformer", "gpt_neox", "backbone", "decoder"):
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError("Could not locate backbone submodule")


class RMT2SegmConfig(PretrainedConfig):
    """
    Configuration class for a 2-segment Recurrent Memory Transformer.
    """
    model_type = "rmt_2segm"

    def __init__(self,
                 pretrained_model=None,
                 base_config=None,
                 n_mem_tokens=8,
                 n_ctrl_tokens=0,
                 K=1,
                 use_mem_proj=False,
                 mem_proj_mode="none",
                 use_reconstruction_loss=False,
                 use_reconstruction_loss_at_first_step=False,
                 use_reconstruction_loss_all_steps=False,
                 reconstruction_loss_weight=1.0,
                 use_write_head=False,
                 use_gradient_checkpointing=False,
                 attn_implementation='eager',
                 **kwargs):
        super().__init__(**kwargs)

        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
            self.base_config = None
        else:
            self.pretrained_model = None
            self.base_config = base_config

        # RMT params
        self.n_mem_tokens = n_mem_tokens
        self.K = K
        self.n_ctrl_tokens = n_ctrl_tokens
        self.use_mem_proj = use_mem_proj
        self.mem_proj_mode = mem_proj_mode
        self.use_reconstruction_loss = use_reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.use_write_head = use_write_head

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_implementation = attn_implementation

        assert self.mem_proj_mode in ["none", "proj", "proj_rw"], \
            f"mem_proj_mode must be one of ['none','proj','proj_rw'], got {self.mem_proj_mode}"
        assert self.use_mem_proj == (mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"


class RMT2Segm(PreTrainedModel):
    """
    RMT-like model with 2 segments only: 1. context 2. query -> target.
    Repeats write to memory K times (with forward passes only), then read from memory.
    Default 2 segment RMT model with K=1.
    """
    config_class = RMT2SegmConfig

    def __init__(self, config):
        """
        mem_proj_mode: "none" | "proj" | "proj_rw"
        none: no linear projection of mem
        K: how many times to process 1st segment (write to memory)
        proj: shared nn.Linear, is used only in write operation (todo: check if it is better to use at read as well)
        proj_rw: one nn.Linear in read operation, one in write operation

        1st segment: [write_st][mem][write_end][context][write_st][mem]
        2nd segment: [read_st][mem][read_end][query][target]

        write_st/write_end/read_st/read_end are parameters aka prompts, that can be used by model to control
            the write/read operation.
        n_ctrl_tokens = 1 means that [write_st] is a single token.

        mem is an output from 1st segment
        """
        super().__init__(config)

        if config.pretrained_model is not None and config.base_config is not None:
            raise ValueError("Only one of pretrained_model or base_config should be provided")
        if config.pretrained_model is None and config.base_config is None:
            raise ValueError("Either pretrained_model or base_config must be provided to instantiate RMT2Segm")

        # initialize base model
        if config.pretrained_model is not None:
            self.model = AutoModelForCausalLM.from_pretrained(config.pretrained_model,
                                                              attn_implementation=config.attn_implementation)
        else:
            self.model = AutoModelForCausalLM.from_config(config.base_config,
                                                          attn_implementation=config.attn_implementation)

        self.n_mem_tokens = config.n_mem_tokens
        self.n_ctrl_tokens = config.n_ctrl_tokens
        self.K = config.K
        self.use_mem_proj = config.use_mem_proj  # defaults to mem_proj_mode == "proj"
        self.mem_proj_mode = config.mem_proj_mode
        self.use_reconstruction_loss = config.use_reconstruction_loss
        self.reconstruction_loss_weight = config.reconstruction_loss_weight
        self.use_write_head = config.use_write_head

        # check args
        assert self.mem_proj_mode in ["none", "proj", "proj_rw"]
        assert self.use_mem_proj == (self.mem_proj_mode != 'none'), "use_mem_proj must be True if mem_proj_mode is set"

        # memory parameters (shape = n_mem_tokens × d)
        n_embd = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        # self.mem initial states.
        self.mem = nn.Parameter(torch.randn(self.n_mem_tokens, n_embd) * 0.02)

        # optional mem projection linear layer
        if self.mem_proj_mode == "proj":
            self.mem_proj = nn.Linear(n_embd, n_embd, bias=True)
        elif self.mem_proj_mode == "proj_rw":
            self.mem_proj = nn.Linear(n_embd, n_embd, bias=True)
            self.read_mem_proj = nn.Linear(n_embd, n_embd, bias=True)
            with torch.no_grad():
                nn.init.eye_(self.read_mem_proj.weight)
                self.read_mem_proj.bias.zero_()

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

        self.tie_weights()
        self.main_input_name = "input_ids"
        self.model.config.use_cache = False
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

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

        device = context_input_ids.device
        pad_id = self.model.config.pad_token_id
        B = context_input_ids.size(0)

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

        stats = {}
        rec_losses = []
        mem_outs = []
        if self.K > 1:
            stats['step_delta_mem_norm_mean'] = torch.tensor(0.0, device=device)
            stats['step_delta_mem_norm_max'] = torch.tensor(-1.0, device=device)
            stats['step_delta_mem_norm_min'] = torch.tensor(1e06, device=device)

        # K segments write to memory
        # +1 for the last segment that computes reconstruction loss only
        # todo: ignore rec_loss at inference time, use only at training
        for k in range(self.K + 1):
            mem_inp_write = self.mem_proj(mem_batch) if self.mem_proj_mode in ["proj", "proj_rw"] else mem_batch

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
            # position_ids = attn_mask.cumsum(-1) - 1
            position_ids = attn_mask.cumsum(-1)
            position_ids[:, :self.n_mem_tokens] = 0
            position_ids[:, self.n_mem_tokens+self.n_ctrl_tokens*2:] = 0

            outs = get_backbone(self.model)(inputs_embeds=x_ctx, position_ids=position_ids, attention_mask=attn_mask,
                                            output_hidden_states=True, return_dict=True)
            h = outs.hidden_states[-1]           # [B,M+S+M,d]
            # extract mem tokens states
            mem_out = h[:, -self.n_mem_tokens:]  # [B,M,d]
            mem_outs += [mem_out.clone()]

            if self.use_reconstruction_loss:
                # ignore rec_loss at first step, as initial memory is empty, so loss can't be reduced much
                if k != 0:
                    lm_labels = context_input_ids.clone()
                    lm_labels[lm_labels == pad_id] = -100
                    # get logits for reconstruction loss
                    logits_st_pos = self.n_mem_tokens+self.n_ctrl_tokens*2 - 1
                    logits_end_pos = -(self.n_mem_tokens+self.n_ctrl_tokens*1)

                    if self.use_write_head:
                        lm_logits = self.write_head(h[:, logits_st_pos:logits_end_pos, :])
                    else:
                        lm_logits = self.model.get_output_embeddings()(h[:, logits_st_pos:logits_end_pos, :])

                    rec_loss = nn.functional.cross_entropy(
                        lm_logits[:, :-1].reshape(-1, lm_logits.size(-1)),
                        lm_labels.reshape(-1),
                        ignore_index=-100,
                        reduction='mean')
                    rec_losses += [rec_loss]

            if k > 0 and 'step_delta_mem_norm_mean' in stats:
                step_delta_mem_norm = (mem_out - mem_batch).norm(dim=(1, 2)).detach()
                stats['step_delta_mem_norm_mean'] += step_delta_mem_norm.mean()
                stats['step_delta_mem_norm_max'] = max(stats['step_delta_mem_norm_max'], step_delta_mem_norm.max())
                stats['step_delta_mem_norm_min'] = min(stats['step_delta_mem_norm_min'], step_delta_mem_norm.min())

            mem_batch = mem_out
            del h, outs

        mem_out_first = mem_outs[0].clone().detach()
        # ignore mem_out from the last segment that computes reconstruction loss only
        mem_out = mem_outs[-2]
        del mem_outs

        mem_norm = mem_out.norm(dim=(1, 2)).detach()  # B
        stats['mem_norm_mean'] = mem_norm.mean()
        stats['mem_norm_max'] = mem_norm.max()
        stats['mem_norm_min'] = mem_norm.min()
        if self.K > 1:
            delta_mem_norm = (mem_out - mem_out_first).norm(dim=(1, 2)).detach()
            stats['delta_mem_norm_mean'] = delta_mem_norm.mean()
            stats['delta_mem_norm_max'] = delta_mem_norm.max()
            stats['delta_mem_norm_min'] = delta_mem_norm.min()
            stats['step_delta_mem_norm_mean'] = stats['step_delta_mem_norm_mean'] / (self.K - 1)

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
        if self.mem_proj_mode == "proj_rw":
            mem_read_inp = self.read_mem_proj(mem_out)
        else:
            mem_read_inp = mem_out

        if self.n_ctrl_tokens > 0:
            # add params that can control read operation from mem
            x_qry = torch.cat([read_st_batch, mem_read_inp, read_end_batch, qry_emb], dim=1)
            qry_attn_mask = torch.cat([ctrl_mask, mem_mask, ctrl_mask, qry_mask], dim=1)
        else:
            x_qry = torch.cat([mem_read_inp, qry_emb], dim=1)            # [B,M+Q,d]
            qry_attn_mask = torch.cat([mem_mask, qry_mask], dim=1)

        # Set read and write memory position ids to zeros
        qry_position_ids = qry_attn_mask.cumsum(-1)
        qry_position_ids[:, :self.n_mem_tokens] = 0
        qry_position_ids[:, self.n_mem_tokens+self.n_ctrl_tokens*2:] = 0

        logits_q = self.model(inputs_embeds=x_qry, attention_mask=qry_attn_mask,
                              position_ids=qry_position_ids).logits  # [B,M+Q,V]
        logits_q = logits_q[:, self.n_mem_tokens+self.n_ctrl_tokens*2:, :]               # [B,Q,V]

        if self.use_reconstruction_loss:
            rec_loss = torch.stack(rec_losses).mean()
            stats['rec_loss'] = rec_loss.detach()

        output = {'predictions': logits_q, 'inner_loop_stats': stats}
        if return_mem:
            output['mem'] = mem_out

        if labels is None:
            return output

        # target loss
        target_loss = nn.functional.cross_entropy(
            logits_q[:, :-1].reshape(-1, logits_q.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        output['inner_loop_stats']['target_loss'] = target_loss.detach()

        loss = target_loss

        if self.use_reconstruction_loss:
            loss = target_loss + self.reconstruction_loss_weight * rec_loss

        output['loss'] = loss
        return output
