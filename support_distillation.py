"""Support-question KL WRITE and held-out READ, with full meta-gradients.

Teacher conventions match self_distillation.py: current backbone, no memory,
stop-gradient, temperature one, teacher-forced answer tokens.
"""
import math
import random

import torch
import torch.nn.functional as F
from datasets import Dataset

from self_distillation import SelfDistillationConfig, SelfDistillationGradMemGPT


def build_episodes(records, n_support, seed):
    if n_support < 1:
        raise ValueError('n_support_questions must be positive')
    groups = {}
    for row in records:
        if row['answers']['text']:
            groups.setdefault(row['context'], {})[row['question']] = row
    rng = random.Random(seed)
    episodes = []
    for context, questions in groups.items():
        rows = list(questions.values())
        if len(rows) <= n_support:
            continue
        for i, heldout in enumerate(rows):
            support = rng.sample(rows[:i] + rows[i + 1:], n_support)
            episodes.append(dict(
                context=context, query=heldout['question'].strip() + ' ',
                target=heldout['answers']['text'][0].strip(),
                support_queries=[r['question'].strip() + ' ' for r in support],
                support_targets=[r['answers']['text'][0].strip() for r in support],
            ))
    if not episodes:
        raise ValueError(f'No contexts have at least {n_support + 1} distinct answered questions')
    return Dataset.from_list(episodes)


def collate_episodes(batch, tokenizer, max_context_length=None):
    def encode_qa(queries, targets):
        encoded = tokenizer([q + a for q, a in zip(queries, targets)], padding=True,
                            return_tensors='pt', return_offsets_mapping=True)
        labels = encoded.input_ids.clone().fill_(-100)
        for i, (q, a) in enumerate(zip(queries, targets)):
            offsets = encoded.offset_mapping[i]
            mask = (offsets[:, 0] < len(q + a)) & (offsets[:, 1] > len(q))
            mask &= encoded.attention_mask[i].bool()
            if not mask.any():
                raise ValueError('Every question must contain at least one answer token')
            labels[i, mask] = encoded.input_ids[i, mask]
        return encoded.input_ids, labels

    query, labels = encode_qa([r['query'] for r in batch], [r['target'] for r in batch])
    support, support_labels = encode_qa(
        [q for r in batch for q in r['support_queries']],
        [a for r in batch for a in r['support_targets']])
    context = tokenizer([r['context'].strip() for r in batch], padding=True,
                        return_tensors='pt', max_length=max_context_length,
                        truncation=max_context_length is not None).input_ids
    return dict(input_ids=dict(context_input_ids=context, query_input_ids=query,
                               support_input_ids=support, support_labels=support_labels),
                labels=labels)


class SupportDistillationConfig(SelfDistillationConfig):
    def __init__(self, n_support_questions=2, inner_distill_query_weight=0.0,
                 inner_distill_target_weight=1.0, inner_reconstruction_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        if n_support_questions < 1:
            raise ValueError('n_support_questions must be positive')
        for value in (inner_distill_query_weight, inner_distill_target_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Inner KL weights must be finite and nonnegative')
        if not (inner_distill_query_weight or inner_distill_target_weight):
            raise ValueError('At least one inner KL weight must be positive')
        if not math.isfinite(inner_reconstruction_weight) or inner_reconstruction_weight < 0:
            raise ValueError('inner_reconstruction_weight must be finite and nonnegative')
        if self.memory_backend != 'prefix' or self.grad_mode != 'second' or self.use_adam:
            raise ValueError('Support distillation requires prefix memory, grad_mode=second, use_adam=false')
        if self.K < 1 or self.last_K_second_order not in (None, self.K):
            raise ValueError('All K>=1 WRITE steps must be second order')
        # Keep the entire existing CLI, but never silently ignore incompatible objectives.
        unsupported = ('K_sampling', 'use_write_head', 'separate_writer',
                       'add_inner_loss_to_outer', 'energy_rank_weight', 'energy_traj_weight',
                       'energy_anchor_weight', 'lipschitz_weight', 'energy_memory_search_weight',
                       'memory_alignment_weight', 'step_alignment_weight',
                       'intermediate_read_weight', 'orthogonal_loss_weight', 'ivan_loss_weight',
                       'memory_noise_sigma')
        for name in unsupported:
            if getattr(self, name, False):
                raise ValueError(f'{name} is not supported by support-question KL WRITE')
        if self.write_objective != 'reconstruction':
            raise ValueError('Leave write_objective=reconstruction; this runner replaces WRITE with support KL')
        self.n_support_questions = n_support_questions
        self.inner_distill_query_weight = inner_distill_query_weight
        self.inner_distill_target_weight = inner_distill_target_weight
        self.inner_reconstruction_weight = float(inner_reconstruction_weight)


class SupportDistillationGradMemGPT(SelfDistillationGradMemGPT):
    config_class = SupportDistillationConfig

    def forward(self, input_ids, labels=None, return_mem=False):
        if labels is None:
            raise ValueError('Held-out labels are required for episodic training/evaluation')
        # HF evaluation uses no_grad; WRITE still needs gradients w.r.t. memory.
        with torch.enable_grad():
            return self._episode_forward(input_ids, labels, return_mem)

    def _episode_forward(self, inputs, labels, return_mem):
        backend = self.memory_backend_impl
        b = inputs['context_input_ids'].size(0)
        n = self.config.n_support_questions
        support = inputs['support_input_ids']
        support_labels = inputs['support_labels']
        if support.size(0) != b * n:
            raise ValueError('Support batch must have B * n_support_questions rows')
        pad = self.read_model.config.pad_token_id
        teacher_inputs = dict(context_input_ids=inputs['context_input_ids'].repeat_interleave(n, 0),
                              query_input_ids=support)
        # Force logit teacher construction even when outer distillation is disabled.
        # Call the same teacher helper using a local flag, restored immediately.
        old = self.distill_logits_target_weight
        try:
            self.distill_logits_target_weight = 1.0
            teacher = self._prepare_outer_supervision(teacher_inputs, support_labels)
        finally:
            self.distill_logits_target_weight = old
        state, initial = backend.init_memory_state(b)
        # Context is never passed into the student's READ path.
        empty = inputs['context_input_ids'][:, :0]
        support_ctx = backend.prepare_batch(empty.repeat_interleave(n, 0), support, pad)
        reconstruction_ctx = (backend.prepare_batch(inputs['context_input_ids'],
                              inputs['query_input_ids'][:, :0], pad)
                              if self.config.inner_reconstruction_weight else None)
        losses, norms, reconstruction_losses, kl_losses = [], [], [], []
        for _ in range(self.K):
            repeated = {k: v.repeat_interleave(n, 0) for k, v in state.items()}
            read_batch = backend.build_read_inputs(repeated, support_ctx)
            x = read_batch['inputs_embeds']
            offset = read_batch['logits_start'] + 1
            attention = torch.cat([support.new_ones((b * n, offset)), support.ne(pad)], 1)
            if reconstruction_ctx is not None:
                # Independent rows share a forward call, not context information.
                write_batch = backend.build_write_inputs(state, reconstruction_ctx)
                write_x = write_batch['inputs_embeds']
                write_attention = torch.cat([
                    support.new_ones((b, write_batch['context_start'])),
                    inputs['context_input_ids'].ne(pad)], 1)
                length = max(x.size(1), write_x.size(1))
                x = torch.cat([F.pad(x, (0, 0, 0, length - x.size(1))),
                               F.pad(write_x, (0, 0, 0, length - write_x.size(1)))], 0)
                attention = torch.cat([F.pad(attention, (0, length - attention.size(1))),
                                       F.pad(write_attention, (0, length - write_attention.size(1)))], 0)
            out = self.read_model(inputs_embeds=x, attention_mask=attention,
                                  use_cache=False, return_dict=True)
            student = out.logits[:b * n, offset - 1:offset + support.size(1) - 1].float()
            kl = F.kl_div(F.log_softmax(student, -1),
                          F.log_softmax(teacher['logits'].float(), -1),
                          log_target=True, reduction='none').sum(-1)
            valid = support.ne(pad) & teacher['logits_valid']
            per_question = kl.new_zeros(b * n)
            for mask, weight in ((support_labels.ne(-100), self.config.inner_distill_target_weight),
                                 (support_labels.eq(-100), self.config.inner_distill_query_weight)):
                mask = mask & valid
                per_question = per_question + weight * (kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            per_context = per_question.reshape(b, n).mean(-1)
            kl_losses.append(per_context.detach())
            if reconstruction_ctx is not None:
                context = inputs['context_input_ids']
                start = write_batch['logits_start']
                logits = out.logits[b * n:, start:start + context.size(1)].float()
                valid_context = context.ne(pad)
                ce = F.cross_entropy(logits.transpose(1, 2),
                                     context.masked_fill(~valid_context, -100),
                                     reduction='none', ignore_index=-100)
                per_context_ce = ce.sum(-1) / valid_context.sum(-1).clamp_min(1)
                reconstruction_losses.append(per_context_ce.detach())
                per_context = per_context + self.config.inner_reconstruction_weight * per_context_ce
            # Sum across independent memories: inner_lr must not depend on outer batch size.
            params = backend.inner_params(state)
            grads = torch.autograd.grad(per_context.sum(), params, create_graph=self.training)
            norms.append(grads[0].detach().flatten(1).norm(dim=1))
            losses.append(per_context.detach())
            backend.assign_inner_params(state, [
                self._sgd_step(p, g, clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
                for p, g in zip(params, grads)])
        heldout_ctx = backend.prepare_batch(empty, inputs['query_input_ids'], pad)
        read_batch = backend.build_read_inputs(state, heldout_ctx)
        offset = read_batch['logits_start'] + 1
        attention = torch.cat([labels.new_ones((b, offset)), inputs['query_input_ids'].ne(pad)], 1)
        with self._disable_write_lora():
            out = self.read_model(inputs_embeds=read_batch['inputs_embeds'], attention_mask=attention,
                                  output_hidden_states=bool(self.distill_hidden_query_weight or
                                                            self.distill_hidden_target_weight),
                                  use_cache=False, return_dict=True)
        predictions = out.logits[:, read_batch['logits_start']:]
        target_loss = self._compute_read_target_loss(predictions, read_batch, labels)
        supervision = self._prepare_outer_supervision(inputs, labels)
        extra, stats = self._outer_supervision_loss(supervision, out, read_batch, inputs, labels, target_loss)
        mem_norm, delta = backend.compute_memory_stats(state, initial)
        stats.update(inner_loss=torch.stack(losses).mean(), target_loss=target_loss.detach())
        stats['inner_support_kl_loss'] = torch.stack(kl_losses).mean()
        stats['inner_reconstruction_loss'] = (torch.stack(reconstruction_losses).mean()
                                              if reconstruction_losses else target_loss.new_zeros(()))
        for name, values in [('inner_grad_norm', torch.stack(norms)),
                             ('mem_norm', mem_norm), ('delta_mem_norm', delta)]:
            stats[name + '_mean'] = values.mean()
            stats[name + '_max'] = values.max()
            stats[name + '_min'] = values.min()
        stats = {k: v.detach().reshape(1) for k, v in stats.items()}
        result = dict(loss=target_loss + extra, predictions=predictions, inner_loop_stats=stats)
        if return_mem:
            backend.attach_return_memory(result, state)
        return result
