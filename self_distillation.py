"""Full-context, stop-gradient self-distillation for GradMem's final READ.

WRITE and all existing auxiliary objectives are inherited unchanged. Teacher and
student use the same current backbone; the teacher has no fast-memory prefix,
adapter, or KV state. Hidden states use HF's post-layer outputs (excluding the
embedding output; the last output includes the backbone's final normalization).
"""

import math

import torch
import torch.nn.functional as F

from grad_memgpt import GradMemGPT, GradMemGPTConfig


DISTILLATION_DEFAULTS = {
    'read_ce_weight': 1.0,
    'distill_logits_query_weight': 0.0,
    'distill_logits_target_weight': 0.0,
    'distill_hidden_query_weight': 0.0,
    'distill_hidden_target_weight': 0.0,
}


class SelfDistillationConfig(GradMemGPTConfig):
    """Inherited GradMem config plus nonnegative final-READ loss weights."""

    def __init__(self, read_ce_weight=1.0, distill_logits_query_weight=0.0,
                 distill_logits_target_weight=0.0, distill_hidden_query_weight=0.0,
                 distill_hidden_target_weight=0.0, **kwargs):
        weights = dict(
            read_ce_weight=read_ce_weight,
            distill_logits_query_weight=distill_logits_query_weight,
            distill_logits_target_weight=distill_logits_target_weight,
            distill_hidden_query_weight=distill_hidden_query_weight,
            distill_hidden_target_weight=distill_hidden_target_weight,
        )
        super().__init__(**kwargs)
        for name, value in weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            setattr(self, name, float(value))


class SelfDistillationGradMemGPT(GradMemGPT):
    config_class = SelfDistillationConfig

    def __init__(self, config):
        super().__init__(config)
        for name in DISTILLATION_DEFAULTS:
            setattr(self, name, getattr(config, name))

    def _prepare_outer_supervision(self, input_ids, labels):
        if labels is None or not any(getattr(self, name) for name in DISTILLATION_DEFAULTS
                                     if name != 'read_ce_weight'):
            return None
        return self._full_context_teacher(input_ids, need_hidden=bool(
            self.distill_hidden_query_weight or self.distill_hidden_target_weight))

    def _full_context_teacher(self, input_ids, *, need_hidden=False):
        """Current, detached, memory-free teacher shared by READ and WRITE KL."""
        context = input_ids['context_input_ids']
        query = input_ids['query_input_ids']
        pad_id = self.read_model.config.pad_token_id
        # Pack each context independently, so right/left padding cannot create
        # internal gaps or change query positions in the teacher sequence.
        contexts = [row[row.ne(pad_id)] for row in context]
        lengths = torch.tensor([row.numel() for row in contexts], device=query.device)
        full = query.new_full((query.size(0), int(lengths.max()) + query.size(1)), pad_id)
        for i, row in enumerate(contexts):
            full[i, :row.numel()] = row
            full[i, row.numel():row.numel() + query.size(1)] = query[i]
        mask = full.ne(pad_id)
        # Explicitly disable WRITE-only adapters and any fast LoRA memory.
        with torch.no_grad(), self._disable_write_lora(), self._enable_lora_memory(None):
            teacher = self.read_model(
                input_ids=full, attention_mask=mask,
                position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0),
                output_hidden_states=need_hidden, use_cache=False, return_dict=True,
            )
        positions = lengths[:, None] + torch.arange(query.size(1), device=query.device)
        batch = torch.arange(query.size(0), device=query.device)[:, None]
        # Logits at j-1 predict token j, unlike hidden states located at j.
        return {
            'logits': teacher.logits[batch, (positions - 1).clamp_min(0)].detach(),
            'logits_valid': positions.gt(0),
            'hidden_states': tuple(h[batch, positions].detach() for h in teacher.hidden_states[1:])
                             if need_hidden else None,
        }

    @staticmethod
    def _masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1)

    def _outer_supervision_loss(self, supervision, read_out, read_batch, input_ids, labels, target_loss):
        extra = (self.read_ce_weight - 1.0) * target_loss
        stats = {'weighted_target_loss': self.read_ce_weight * target_loss}
        if supervision is None:
            return extra, stats
        if self.read_loss_alignment != 'causal':
            raise ValueError('Self-distillation currently requires causal READ alignment')
        query = input_ids['query_input_ids']
        valid = query.ne(self.read_model.config.pad_token_id)
        masks = {'target': valid & labels.ne(-100), 'query': valid & labels.eq(-100)}
        n = query.size(1)
        # Prefix READ includes one preceding memory/control position; other
        # backends cannot predict token zero, so omit it only from logit KL.
        token_start = read_batch['logits_start'] + (self.memory_backend == 'prefix')
        first = 0 if token_start > 0 else 1
        if self.distill_logits_query_weight or self.distill_logits_target_weight:
            student = read_out.logits[:, token_start + first - 1:token_start + n - 1].float()
            teacher = supervision['logits'][:, first:].detach().float()
            kl = F.kl_div(F.log_softmax(student, -1), F.log_softmax(teacher, -1),
                          log_target=True, reduction='none').sum(-1)
            for region, mask in masks.items():
                name = f'distill_logits_{region}'
                loss = self._masked_mean(kl, mask[:, first:] & supervision['logits_valid'][:, first:])
                stats[name + '_loss'] = loss
                extra = extra + getattr(self, name + '_weight') * loss
        if supervision['hidden_states'] is not None:
            student_layers = read_out.hidden_states[1:]
            teacher_layers = supervision['hidden_states']
            if len(student_layers) != len(teacher_layers):
                raise ValueError('Teacher/student layer counts differ')
            # Mean over hidden dimensions, tokens, and layers (equal layer weights).
            mse = torch.stack([
                (student[:, token_start:token_start + n].float() - teacher.detach().float()).square().mean(-1)
                for student, teacher in zip(student_layers, teacher_layers)
            ]).mean(0)
            for region, mask in masks.items():
                name = f'distill_hidden_{region}'
                loss = self._masked_mean(mse, mask)
                stats[name + '_loss'] = loss
                extra = extra + getattr(self, name + '_weight') * loss
        stats['distillation_loss'] = extra - (self.read_ce_weight - 1.0) * target_loss
        return extra, stats
