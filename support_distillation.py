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
from grad_memgpt import get_backbone


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
        if not math.isfinite(inner_reconstruction_weight) or inner_reconstruction_weight < 0:
            raise ValueError('inner_reconstruction_weight must be finite and nonnegative')
        if not (inner_distill_query_weight or inner_distill_target_weight or inner_reconstruction_weight
                or self.write_objective in ('energy', 'energy_with_reconstruction')):
            raise ValueError('At least one inner KL, reconstruction, or energy signal must be enabled')
        self.n_support_questions = n_support_questions
        self.inner_distill_query_weight = inner_distill_query_weight
        self.inner_distill_target_weight = inner_distill_target_weight
        self.inner_reconstruction_weight = float(inner_reconstruction_weight)


class SupportDistillationGradMemGPT(SelfDistillationGradMemGPT):
    config_class = SupportDistillationConfig

    def forward(self, input_ids, labels=None, return_mem=False):
        # Retain the existing meta-loop, backend activation, sampling, optimizers,
        # intermediate READs, shaping/search, and auxiliary losses unchanged.
        with torch.enable_grad():
            return super().forward(input_ids, labels=labels, return_mem=return_mem)

    def _prepare_write_supervision(self, inputs, labels):
        b = inputs['context_input_ids'].size(0)
        n = self.config.n_support_questions
        support = inputs['support_input_ids']
        support_labels = inputs['support_labels']
        if support.size(0) != b * n or support_labels.shape != support.shape:
            raise ValueError('Support batch must have B * n_support_questions rows')
        pad = self.read_model.config.pad_token_id
        if not inputs['context_input_ids'].ne(pad).any():
            raise ValueError('Support distillation requires a nonempty context')
        teacher_inputs = dict(context_input_ids=inputs['context_input_ids'].repeat_interleave(n, 0),
                              query_input_ids=support)
        empty = inputs['context_input_ids'][:, :0]
        return dict(
            b=b, n=n, tokens=support, labels=support_labels,
            context=inputs['context_input_ids'],
            teacher=self._full_context_teacher(teacher_inputs),
            batch_ctx=self.memory_backend_impl.prepare_batch(empty.repeat_interleave(n, 0), support, pad),
            kl_history=[],
        )

    @staticmethod
    def _map_state(state, fn):
        if isinstance(state, dict):
            return {key: SupportDistillationGradMemGPT._map_state(value, fn) for key, value in state.items()}
        if isinstance(state, tuple):
            return tuple(SupportDistillationGradMemGPT._map_state(value, fn) for value in state)
        return fn(state)

    def _support_kl(self, logits, read_batch, supervision):
        tokens, labels = supervision['tokens'], supervision['labels']
        # Prefix memory predicts token zero; LoRA/KV need a preceding seed token.
        start = read_batch['logits_start'] + (self.memory_backend == 'prefix')
        first = 0 if start else 1
        student = logits[:, start + first - 1:start + tokens.size(1) - 1].float()
        teacher = supervision['teacher']
        kl = F.kl_div(F.log_softmax(student, -1),
                      F.log_softmax(teacher['logits'][:, first:].float(), -1),
                      log_target=True, reduction='none').sum(-1)
        valid = tokens[:, first:].ne(self.read_model.config.pad_token_id) & teacher['logits_valid'][:, first:]
        per_question = kl.new_zeros(tokens.size(0))
        for mask, weight in ((labels.ne(-100), self.config.inner_distill_target_weight),
                             (labels.eq(-100), self.config.inner_distill_query_weight)):
            mask = mask[:, first:] & valid
            per_question = per_question + weight * (kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return per_question.reshape(supervision['b'], supervision['n']).mean(-1)

    def _support_forward(self, backend, state, supervision):
        repeated = self._map_state(state, lambda tensor: tensor.repeat_interleave(supervision['n'], 0))
        read_batch = backend.build_read_inputs(repeated, supervision['batch_ctx'])
        kwargs = dict(read_batch.get('model_kwargs', {}))
        prefix = read_batch['inputs_embeds'].size(1) - supervision['tokens'].size(1)
        mask = torch.cat([supervision['tokens'].new_ones((supervision['tokens'].size(0), prefix)),
                          supervision['tokens'].ne(self.read_model.config.pad_token_id)], 1)
        if self.memory_backend == 'kv_cache':
            mask = torch.cat([mask.new_ones((mask.size(0), self.n_mem_tokens)), mask], 1)
        kwargs['attention_mask'] = mask
        # Support KL is a WRITE signal, so WRITE-only LoRA stays enabled here.
        with backend.activation_context(repeated):
            out = self.read_model(inputs_embeds=read_batch['inputs_embeds'], return_dict=True, **kwargs)
        return self._support_kl(out.logits, read_batch, supervision)

    def _fused_prefix_forward(self, backend, state, batch_ctx, supervision):
        """Context CE/energy and every support KL share a backbone invocation."""
        b, n = supervision['b'], supervision['n']
        repeated = self._map_state(state, lambda tensor: tensor.repeat_interleave(n, 0))
        read_batch = backend.build_read_inputs(repeated, supervision['batch_ctx'])
        write_batch = backend.build_write_inputs(state, batch_ctx)
        read_x, write_x = read_batch['inputs_embeds'], write_batch['inputs_embeds']
        width = max(read_x.size(1), write_x.size(1))
        x = torch.cat([F.pad(read_x, (0, 0, 0, width - read_x.size(1))),
                       F.pad(write_x, (0, 0, 0, width - write_x.size(1)))], 0)
        pad = self.read_model.config.pad_token_id
        read_mask = torch.cat([supervision['tokens'].new_ones((b * n, read_batch['logits_start'] + 1)),
                               supervision['tokens'].ne(pad)], 1)
        write_mask = torch.cat([supervision['context'].new_ones((b, write_batch['context_start'])),
                                supervision['context'].ne(pad)], 1)
        mask = torch.cat([F.pad(read_mask, (0, width - read_mask.size(1))),
                          F.pad(write_mask, (0, width - write_mask.size(1)))], 0)
        out = get_backbone(self.write_model)(inputs_embeds=x, attention_mask=mask,
                  output_hidden_states=self.use_layerwise_energy, return_dict=True)
        hidden = out.last_hidden_state
        support_logits = self.read_model.get_output_embeddings()(hidden[:b * n, :read_x.size(1)])
        kl = self._support_kl(support_logits, read_batch, supervision)
        context_hidden = hidden[b * n:, :write_x.size(1)]
        head = self.write_head if self.use_write_head else self.write_model.get_output_embeddings()
        reconstruction = self._compute_write_reconstruction_loss(
            head(context_hidden[:, write_batch['logits_start']:]), write_batch)
        energy = None
        if self.write_objective in ('energy', 'energy_with_reconstruction'):
            states = (tuple(h[b * n:, :write_x.size(1)] for h in out.hidden_states[1:])
                      if self.use_layerwise_energy else context_hidden)
            energy = self._compute_write_energy(states, write_batch)
        return out, reconstruction, energy, kl

    def _run_write_objective(self, backend, state, batch_ctx, supervision, *, final=False):
        # Separate writers have different hidden spaces; other memory formats
        # require their own adapter/cache activation. Retain their native paths.
        if self.memory_backend == 'prefix' and not self.separate_writer:
            out, reconstruction, energy, kl = self._fused_prefix_forward(backend, state, batch_ctx, supervision)
        else:
            out, reconstruction, energy, _ = super()._run_write_objective(
                backend, state, batch_ctx, supervision, final=final)
            kl = self._support_forward(backend, state, supervision)
        loss = kl + self.config.inner_reconstruction_weight * reconstruction
        if self.write_objective == 'energy':
            loss = loss + energy  # Preserve original energy-only coefficient.
        elif self.write_objective == 'energy_with_reconstruction':
            loss = loss + self.write_energy_weight * energy + self.write_reconstruction_weight * reconstruction
        if final:
            supervision['final_kl'] = kl.detach().mean()
        else:
            supervision['kl_history'].append(kl.detach().mean())
        return out, reconstruction, energy, loss

    def _write_supervision_stats(self, supervision):
        history = supervision['kl_history']
        return dict(inner_support_kl_loss=history[-1] if history else supervision['final_kl'],
                    inner_support_kl_loss_after_write=supervision['final_kl'])
