import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets

import accelerate
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)
import yaml

from grad_memgpt import GradMemGPT, GradMemGPTConfig
# adaptive (segmented, gated-recurrence) fork, imported lazily on demand. Aliased
# to avoid the class-name clash with the base grad_memgpt module.
_AdaptiveGradMemGPT = None
_AdaptiveGradMemGPTConfig = None
def _load_adaptive():
    global _AdaptiveGradMemGPT, _AdaptiveGradMemGPTConfig
    if _AdaptiveGradMemGPT is None:
        from grad_memgpt_adaptive import GradMemGPT as _GM, GradMemGPTConfig as _GC
        _AdaptiveGradMemGPT, _AdaptiveGradMemGPTConfig = _GM, _GC
    return _AdaptiveGradMemGPT, _AdaptiveGradMemGPTConfig


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer, max_context_length=None, hopfield=False):
    context = [item['context'] for item in batch]
    query = [item['query'] + item['target'] for item in batch]

    if hopfield:
        context_input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True,
                                      padding=True, pad_to_multiple_of=8).input_ids
    else:
        context_input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True,
                                      padding=True, pad_to_multiple_of=8, max_length=max_context_length,
                                      truncation=True).input_ids
    query_encoded = tokenizer(query, return_tensors="pt", add_special_tokens=True,
                              padding=True, pad_to_multiple_of=8, return_offsets_mapping=True)
    query_input_ids = query_encoded['input_ids']
    offsets_mapping = query_encoded['offset_mapping']

    # add labels_mask
    # input_seq: 0, target_seq: 1, seq = input_seq + target_seq
    labels_mask = torch.zeros_like(query_input_ids)
    for i, item in enumerate(batch):
        query_seq_len = len(item['query'])
        target_seq_len = len(item['target'])
        target_st, target_end = query_seq_len, query_seq_len + target_seq_len
        # find target tokens
        # since target is closer to the end (context, query, target), search from the end
        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            # if (target_st, target_end) intersects with (st, end), it is a target token
            if st < target_end and end > target_st:
                labels_mask[i, j] = 1
                in_target = True
            elif in_target:
                break

    labels = query_input_ids * labels_mask + (1 - labels_mask) * -100
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
        'labels': labels,
    }


def preprocess_logits_for_metrics(eval_pred, labels):
    logits, inner_loop_stats = eval_pred
    # saves gpu RAM, as HF Trainer accumulates all eval logits on GPU
    return (logits.argmax(dim=-1), inner_loop_stats)


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    preds, inner_loop_stats = predictions
    preds = preds[..., :-1]
    labels = labels[..., :]

    # Create a mask for tokens that are not padding (-100) and ignored tokens (like ! and |)
    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)

    # Calculate token-level accuracy only on content tokens
    masked_predictions = preds[mask]
    masked_labels = labels[mask]

    accuracy = (masked_predictions == masked_labels).mean()

    # get exact_match per-sample accuracy, ignore masked tokens
    # predictions.shape = (batch_size, seq_len)
    exact_match = np.mean([
        np.all(pred[mask[i]] == lab[mask[i]])
        for i, (pred, lab) in enumerate(zip(preds, labels))
        if np.any(mask[i])  # Skip samples that are all masked
    ])

    for pred, label, inp_c, inp_q in zip(preds[:5], labels[:5],
                                         inputs['context_input_ids'][:5], inputs['query_input_ids'][:5]):
        mask = (label != -100)
        pred = pred[mask]
        inp_c[inp_c == -100] = tokenizer.pad_token_id
        inp_q[inp_q == -100] = tokenizer.pad_token_id
        label[label == -100] = tokenizer.pad_token_id
        print('i:', tokenizer.decode(np.concatenate([inp_c, inp_q]), skip_special_tokens=True).strip())
        print('p:', tokenizer.decode(pred, skip_special_tokens=True).strip())
        print('t:', tokenizer.decode(label, skip_special_tokens=True).strip())
        print('-' * 50)

    metrics = {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
        "inner_loss": float(inner_loop_stats['inner_loss'].mean()),
        "inner_grad_norm": float(inner_loop_stats['inner_grad_norm_mean'].mean()),
        "inner_grad_norm_max": float(inner_loop_stats['inner_grad_norm_max'].max()),
        "inner_grad_norm_min": float(inner_loop_stats['inner_grad_norm_min'].min()),
        "mem_norm_mean": float(inner_loop_stats['mem_norm_mean'].mean()),
        "mem_norm_max": float(inner_loop_stats['mem_norm_max'].max()),
        "mem_norm_min": float(inner_loop_stats['mem_norm_min'].min()),
        "delta_mem_norm_mean": float(inner_loop_stats['delta_mem_norm_mean'].mean()),
        "delta_mem_norm_max": float(inner_loop_stats['delta_mem_norm_max'].max()),
        "delta_mem_norm_min": float(inner_loop_stats['delta_mem_norm_min'].min()),
    }
    if 'target_loss' in inner_loop_stats:
        metrics['target_loss'] = float(inner_loop_stats['target_loss'].mean())
    if 'rec_loss' in inner_loop_stats:
        metrics['rec_loss'] = float(inner_loop_stats['rec_loss'].mean())
    if 'energy' in inner_loop_stats:
        metrics['energy'] = float(inner_loop_stats['energy'].mean())
    if 'energy_recon' in inner_loop_stats:
        metrics['energy_recon'] = float(inner_loop_stats['energy_recon'].mean())
    if 'energy_recon_weight' in inner_loop_stats:
        metrics['energy_recon_weight'] = float(inner_loop_stats['energy_recon_weight'].mean())
    if 'inner_loss_seg_min' in inner_loop_stats:
        metrics['inner_loss_seg_min'] = float(inner_loop_stats['inner_loss_seg_min'].mean())
        metrics['inner_loss_seg_max'] = float(inner_loop_stats['inner_loss_seg_max'].mean())
    if 'recon_pretrain_target_weight' in inner_loop_stats:
        metrics['recon_pretrain_target_weight'] = float(inner_loop_stats['recon_pretrain_target_weight'].mean())
    if 'recon_pretrain_recon_weight' in inner_loop_stats:
        metrics['recon_pretrain_recon_weight'] = float(inner_loop_stats['recon_pretrain_recon_weight'].mean())
    if 'energy_input_delta_mem_norm_mean' in inner_loop_stats:
        metrics['energy_input_delta_mem_norm_mean'] = float(inner_loop_stats['energy_input_delta_mem_norm_mean'].mean())
        metrics['energy_input_delta_mem_norm_max'] = float(inner_loop_stats['energy_input_delta_mem_norm_max'].max())
        metrics['energy_input_delta_mem_norm_min'] = float(inner_loop_stats['energy_input_delta_mem_norm_min'].min())
    if 'step_delta_mem_norm_mean' in inner_loop_stats:
        metrics['step_delta_mem_norm_mean'] = float(inner_loop_stats['step_delta_mem_norm_mean'].mean())
        metrics['step_delta_mem_norm_max'] = float(inner_loop_stats['step_delta_mem_norm_max'].max())
        metrics['step_delta_mem_norm_min'] = float(inner_loop_stats['step_delta_mem_norm_min'].min())
    if 'seg_nonempty_count_mean' in inner_loop_stats:
        metrics['seg_nonempty_count_mean'] = float(inner_loop_stats['seg_nonempty_count_mean'].mean())
        metrics['seg_nonempty_count_max'] = float(inner_loop_stats['seg_nonempty_count_max'].max())
        metrics['seg_nonempty_count_min'] = float(inner_loop_stats['seg_nonempty_count_min'].min())
    if 'seg_nonempty_size_mean' in inner_loop_stats:
        metrics['seg_nonempty_size_mean'] = float(inner_loop_stats['seg_nonempty_size_mean'].mean())
        metrics['seg_nonempty_size_max'] = float(inner_loop_stats['seg_nonempty_size_max'].max())
        metrics['seg_nonempty_size_min'] = float(inner_loop_stats['seg_nonempty_size_min'].min())
    if 'hopfield_entropy_mean' in inner_loop_stats:
        metrics['hopfield_entropy_mean'] = float(inner_loop_stats['hopfield_entropy_mean'].mean())
        metrics['hopfield_entropy_max'] = float(inner_loop_stats['hopfield_entropy_max'].max())
        metrics['hopfield_entropy_min'] = float(inner_loop_stats['hopfield_entropy_min'].min())
        metrics['hopfield_entropy_norm_mean'] = float(inner_loop_stats['hopfield_entropy_norm_mean'].mean())
        metrics['hopfield_entropy_norm_max'] = float(inner_loop_stats['hopfield_entropy_norm_max'].max())
        metrics['hopfield_entropy_norm_min'] = float(inner_loop_stats['hopfield_entropy_norm_min'].min())
        metrics['hopfield_n_seg_50_mean'] = float(inner_loop_stats['hopfield_n_seg_50_mean'].mean())
        metrics['hopfield_n_seg_50_max'] = float(inner_loop_stats['hopfield_n_seg_50_max'].max())
        metrics['hopfield_n_seg_50_min'] = float(inner_loop_stats['hopfield_n_seg_50_min'].min())
        metrics['hopfield_n_seg_95_mean'] = float(inner_loop_stats['hopfield_n_seg_95_mean'].mean())
        metrics['hopfield_n_seg_95_max'] = float(inner_loop_stats['hopfield_n_seg_95_max'].max())
        metrics['hopfield_n_seg_95_min'] = float(inner_loop_stats['hopfield_n_seg_95_min'].min())
    if 'gd_alpha_mean' in inner_loop_stats:
        metrics['gd_alpha_mean'] = float(inner_loop_stats['gd_alpha_mean'].mean())
        metrics['gd_beta_mean'] = float(inner_loop_stats['gd_beta_mean'].mean())
        metrics['gd_S_norm_mean'] = float(inner_loop_stats['gd_S_norm_mean'].mean())
        metrics['gd_S_norm_max'] = float(inner_loop_stats['gd_S_norm_max'].max())
        metrics['gd_S_norm_min'] = float(inner_loop_stats['gd_S_norm_min'].min())
        metrics['gd_n_written_mean'] = float(inner_loop_stats['gd_n_written_mean'].mean())
    if 'learned_update_delta_norm_mean' in inner_loop_stats:
        metrics['learned_update_delta_norm_mean'] = float(inner_loop_stats['learned_update_delta_norm_mean'].mean())
        metrics['learned_update_delta_norm_max'] = float(inner_loop_stats['learned_update_delta_norm_max'].max())
        metrics['learned_update_delta_norm_min'] = float(inner_loop_stats['learned_update_delta_norm_min'].min())
    if 'learned_update_target_grad_norm_mean' in inner_loop_stats:
        # ‖g_real‖ during the imitation warmup window (key only exists while warmup is active).
        metrics['learned_update_target_grad_norm_mean'] = float(inner_loop_stats['learned_update_target_grad_norm_mean'].mean())
        metrics['learned_update_target_grad_norm_max'] = float(inner_loop_stats['learned_update_target_grad_norm_max'].max())
    if 'learned_update_imitation_loss' in inner_loop_stats:
        # MSE(delta, g_real); only present during the warmup window.
        metrics['learned_update_imitation_loss'] = float(inner_loop_stats['learned_update_imitation_loss'].mean())
    # ---- adaptive (segmented/gated) update diagnostics ---- #
    # Only present under the adaptive model with memory_update_rule != "sgd".
    # gate_retain_mean / gate_write_mean: mean retention r_t and write w_t of the
    # gated recurrence m_t = r_t*m_{t-1} + w_t*x_t (->1 = pure SGD running-sum).
    # gate_delta_mean: mean Mamba Delta (->0 = full retention). seg stats below
    # are also emitted by the segmented adaptive path.
    for _k in ('gate_retain_mean', 'gate_write_mean', 'gate_delta_mean',
               'seg_nonempty_count_mean', 'seg_nonempty_size_mean',
               'mem_prior_loss', 'mem_prior_weight_now', 'mem_sampled_delta_norm_mean'):
        if _k in inner_loop_stats:
            metrics[_k] = float(inner_loop_stats[_k].mean())
    return metrics


def collate_fn_no_context(batch, tokenizer):
    """Collate for the "without memory" baseline: context is all-pad.

    Builds the same nested-dict shape as `collate_fn` (`context_input_ids`,
    `query_input_ids`, `labels`) and reuses the identical query/target
    tokenization + offset-based target masking, but sets `context_input_ids`
    to a single pad column per sample. With no non-pad context tokens the
    GradMemGPT WRITE branch is skipped (grad_memgpt.py line ~1209,
    `if self.K and context_input_ids.ne(pad_id).any()`) so the READ phase
    runs against the initial (unwritten) memory — exactly the "without memory"
    condition.
    """
    pad_id = tokenizer.pad_token_id
    # context_input_ids: [B, 1] all-pad. Size 1 keeps it cheap; the WRITE
    # guard checks `.ne(pad_id).any()` which is False for all-pad tensors.
    B = len(batch)
    context_input_ids = torch.full((B, 1), pad_id, dtype=torch.long)

    # query+target tokenization + target masking is identical to collate_fn.
    query = [item['query'] + item['target'] for item in batch]
    query_encoded = tokenizer(query, return_tensors="pt", add_special_tokens=True,
                              padding=True, pad_to_multiple_of=8, return_offsets_mapping=True)
    query_input_ids = query_encoded['input_ids']
    offsets_mapping = query_encoded['offset_mapping']

    labels_mask = torch.zeros_like(query_input_ids)
    for i, item in enumerate(batch):
        query_seq_len = len(item['query'])
        target_seq_len = len(item['target'])
        target_st, target_end = query_seq_len, query_seq_len + target_seq_len
        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            if st < target_end and end > target_st:
                labels_mask[i, j] = 1
                in_target = True
            elif in_target:
                break

    labels = query_input_ids * labels_mask + (1 - labels_mask) * -100
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
        'labels': labels,
    }


def compute_metrics_fn_no_context(eval_pred, ignore_token_ids, tokenizer):
    """Metrics for the "without memory" baseline.

    Identical token-accuracy / exact-match logic to `compute_metrics_fn`, but
    every `inner_loop_stats` access is guarded: when the WRITE phase is skipped
    (all-pad context) the stats dict lacks `inner_loss` and the grad/mem norms,
    so we only surface keys that are actually present.
    """
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    preds, inner_loop_stats = predictions
    preds = preds[..., :-1]
    labels = labels[..., :]

    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)

    masked_predictions = preds[mask]
    masked_labels = labels[mask]
    accuracy = (masked_predictions == masked_labels).mean()

    exact_match = np.mean([
        np.all(pred[mask[i]] == lab[mask[i]])
        for i, (pred, lab) in enumerate(zip(preds, labels))
        if np.any(mask[i])
    ])

    metrics = {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }
    # inner_loop_stats is sparsely populated when WRITE is skipped; surface
    # anything that is present (defensively guarded).
    for k, v in inner_loop_stats.items():
        if k.startswith('_'):
            continue
        if hasattr(v, 'mean'):
            metrics[k] = v.mean().item()
        else:
            metrics[k] = float(v)
    return metrics


class StopOnMetricValue(TrainerCallback):
    def __init__(self, metric_name: str, value: float, higher_is_better: bool = True):
        self.metric_name = metric_name
        self.value = value
        self.higher_is_better = higher_is_better

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not self.metric_name.startswith("eval_"):
            metric_to_check = f"eval_{self.metric_name}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            return
        operator = np.greater_equal if self.higher_is_better else np.less_equal
        if operator(metric_value, self.value):
            control.should_training_stop = True
            logger.info(f'metric {self.metric_name}={metric_value:.4f} >= {self.value:.4f}, stopping training..')


class CurriculumCallback(TrainerCallback):
    def __init__(self, metric_name: str = "exact_match", threshold: float = 0.95):
        self.metric_name = metric_name
        self.threshold = threshold
        self.threshold_reached = False

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not self.metric_name.startswith("eval_"):
            metric_to_check = f"eval_{self.metric_name}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            return
        if metric_value >= self.threshold:
            self.threshold_reached = True
            control.should_training_stop = True
            logger.info(f'curriculum: {self.metric_name}={metric_value:.4f} >= {self.threshold:.4f}, threshold reached! advancing to next stage')


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        num_training_steps = int(num_training_steps / 0.9)  # to make final lr not zero, for linear it is lr/10.
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # stamp the current outer step onto the (unwrapped) model so the energy-recon
        # schedule knows where in training it is; covers train + eval (both call compute_loss)
        m = model
        while hasattr(m, "module"):
            m = m.module
        if hasattr(m, "set_train_step"):
            m.set_train_step(self.state.global_step)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


class DualEvalTrainer(CustomTrainer):
    """Trainer that also runs a "without memory" (no-context) eval pass.

    After the normal `evaluate()` call (context written to memory), this runs
    a second pass against `no_context_dataset` with `no_context_data_collator`
    (which feeds all-pad context → WRITE phase skipped → initial memory used)
    and `no_context_compute_metrics`. The token-accuracy and exact-match
    deltas between the two passes are injected as
    `eval_delta_token_accuracy` / `eval_delta_exact_match`, so the metric the
    task actually cares about (how much memory helps) flows to comet_ml and is
    usable as `metric_for_best_model`.
    """

    def __init__(self, *args, no_context_dataset=None, no_context_data_collator=None,
                 no_context_compute_metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_context_dataset = no_context_dataset
        self.no_context_data_collator = no_context_data_collator
        self.no_context_compute_metrics = no_context_compute_metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 1. Normal eval (context → memory).
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix)

        if self.no_context_dataset is None:
            return metrics

        # 2. "Without memory" eval: temporarily swap collator + compute_metrics
        # so the no-context pass uses the all-pad collate and its (guarded)
        # stats handling, then restore them. metric_key_prefix='no_mem' makes
        # HF auto-prefix the no-context metrics as eval_no_mem_*.
        saved_collator = self.data_collator
        saved_compute_metrics = self.compute_metrics
        try:
            self.data_collator = self.no_context_data_collator
            self.compute_metrics = self.no_context_compute_metrics
            no_ctx_metrics = super().evaluate(
                eval_dataset=self.no_context_dataset, ignore_keys=ignore_keys,
                metric_key_prefix="no_mem")
        finally:
            self.data_collator = saved_collator
            self.compute_metrics = saved_compute_metrics

        # 3. Inject deltas into the returned (primary) metrics dict.
        key_acc = f"{metric_key_prefix}_token_accuracy"
        key_em = f"{metric_key_prefix}_exact_match"
        no_ctx_acc = no_ctx_metrics.get("no_mem_token_accuracy")
        no_ctx_em = no_ctx_metrics.get("no_mem_exact_match")
        if key_acc in metrics and no_ctx_acc is not None:
            metrics[f"{metric_key_prefix}_no_mem_token_accuracy"] = no_ctx_acc
            metrics[f"{metric_key_prefix}_delta_token_accuracy"] = \
                float(metrics[key_acc]) - float(no_ctx_acc)
        if key_em in metrics and no_ctx_em is not None:
            metrics[f"{metric_key_prefix}_no_mem_exact_match"] = no_ctx_em
            metrics[f"{metric_key_prefix}_delta_exact_match"] = \
                float(metrics[key_em]) - float(no_ctx_em)

        # Carry the no-context stats through so they're logged too.
        for k, v in no_ctx_metrics.items():
            metrics.setdefault(k, v)
        return metrics


def _isnan(x):
    import math as _m
    try:
        return _m.isnan(x)
    except (TypeError, ValueError):
        return False


def make_collate_fn_per_segment(tokenizer, max_context_length=None, n_segments=None,
                                segment_size=None):
    """Build a collator that emits, per sample, the WRITE context PLUS one probe
    query per KV pair in the context (with the source model-segment of each KV).

    Context tokenization is identical to ``collate_fn`` (right-padded). For each
    sample we regex the decoded context for ``!K:V!`` pairs and emit:
      query_input_ids: [n_kv, Q]   tokenizing ``?!K:`` for each KV
      target_ids:      [n_kv, T]   tokenizing ``V!|`` for each KV
      seg_idx:         [n_kv]      source model-segment (computed with the same
                                   ceil-division as the model's forward)
    These are stacked into per-batch padded tensors so the model can batch-probe.

    The KV pair -> segment attribution matches the model's segmentation exactly
    (the model shares the same _segment_bounds logic). KV pairs straddling a
    segment boundary are attributed to the later segment (not fully written
    until then).
    """
    import re as _re

    def collate_fn_per_segment(batch, tokenizer, max_context_length=None):
        # --- context: identical to collate_fn ---
        context = [item['context'] for item in batch]
        context_input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True,
                                      padding=True, pad_to_multiple_of=8, max_length=max_context_length,
                                      truncation=True).input_ids

        pad_id = tokenizer.pad_token_id
        B = len(batch)

        # --- per-sample KV probe pairs ---
        # Each probe is the TEACHER-FORCED query '?!K:V!|' (query + target
        # concatenated, exactly as collate_fn builds the normal READ query), plus
        # a target_mask flagging the 'V!|' positions. EM is scored only there,
        # mirroring compute_metrics_fn -- so a probe is correct iff the model
        # re-predicts the value tokens given the correct preceding tokens.
        #
        # KV->segment attribution MUST match the model's forward chunking. The
        # model chunks the PADDED context (ctx_emb.size(1)) via
        # ceil(padded_len / n_segments). The collator right-pads, so a KV's real
        # char position ts (1 char/token for the KV alphabet) is already its
        # position in the PADDED sequence (pad tokens sit to the right of the
        # real tokens). This keeps the collator and the model on exactly the
        # same segment boundaries.
        padded_len = context_input_ids.size(1)
        per_sample = []  # list of [(q_ids, target_mask, seg_idx), ...]
        for b_idx, item in enumerate(batch):
            text = item['context']
            pairs = []
            for m in _re.finditer(r'!([^!|:]+):([^!|]+)!', text):
                k, v = m.group(1), m.group(2)
                pairs.append((m.start(), m.end(), k, v))
            # The context is RIGHT-padded (tokenizer default padding_side='right';
            # collate_fn does not override it). Pads append after the trailing '|',
            # so a KV's real char position == its padded token position. The model
            # chunks the padded tensor, so we attribute using these positions to
            # stay aligned with the model's segment boundaries.
            if segment_size is not None:
                n_seg = max(1, math.ceil(padded_len / segment_size))
                seg_sz = segment_size
            else:
                n_seg = n_segments if n_segments else 1
                seg_sz = max(1, math.ceil(padded_len / n_seg))
            sample_pairs = []
            for ts, te, k, v in pairs:
                # right-padding: padded position == real position (pads are past the '|')
                pad_ts = ts
                pad_te = te
                src_seg = None
                for si in range(n_seg):
                    sstart = si * seg_sz
                    send = min((si + 1) * seg_sz, padded_len)
                    if sstart <= pad_ts and (pad_te - 1) < send:
                        src_seg = si
                        break
                if src_seg is None:
                    src_seg = min(pad_ts // seg_sz, n_seg - 1)
                # teacher-forced query: '?!K:' + target 'V!|'
                q_str = f'?!{k}:'
                t_str = f'{v}!|'
                q_ids = tokenizer(q_str, add_special_tokens=False).input_ids
                t_ids = tokenizer(t_str, add_special_tokens=False).input_ids
                full = q_ids + t_ids
                tmask = [False] * len(q_ids) + [True] * len(t_ids)
                sample_pairs.append((full, tmask, src_seg))
            per_sample.append(sample_pairs)

        n_q_max = max((len(p) for p in per_sample), default=1) or 1
        Q = max((len(q) for p in per_sample for (q, _, _) in p), default=1) or 1

        query_input_ids = torch.full((B, n_q_max, Q), pad_id, dtype=torch.long)
        target_mask = torch.zeros(B, n_q_max, Q, dtype=torch.bool)
        seg_idx = torch.zeros(B, n_q_max, dtype=torch.long)
        qmask = torch.zeros(B, n_q_max, dtype=torch.bool)
        for b, pairs in enumerate(per_sample):
            for j, (q, tm, s) in enumerate(pairs):
                query_input_ids[b, j, :len(q)] = torch.tensor(q, dtype=torch.long)
                target_mask[b, j, :len(tm)] = torch.tensor(tm, dtype=torch.bool)
                seg_idx[b, j] = s
                qmask[b, j] = True

        return {
            'input_ids': {
                'context_input_ids': context_input_ids,
                'query_input_ids': None,  # unused by forward_per_segment_eval; kept for shape compat
            },
            'kv_queries': {
                'query_input_ids': query_input_ids,
                'target_mask': target_mask,
                'seg_idx': seg_idx,
                'mask': qmask,
                'ignore_token_ids': [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']],
            },
        }

    return lambda batch: collate_fn_per_segment(batch, tokenizer, max_context_length)


class PerSegmentForgettingTrainer(DualEvalTrainer):
    """DualEvalTrainer + a per-segment forgetting eval pass.

    On cadence (every ``per_segment_eval_steps``), runs a manual no-grad loop
    that calls ``model.forward_per_segment_eval`` to build the [n_seg, n_seg]
    forgetting matrix, writes it to disk as .json/.csv, optionally logs it as a
    comet table, and injects the per-cell matrix values as scalars into the
    metrics dict (forget_seg{s}_at_seg{k}, which CometCallback auto-forwards to
    comet as chartable lines) plus a few summary scalars. The full matrix also
    lives in the local .json/.csv artifact.

    The pass reuses the same eval dataset (the context is what gets written);
    only the collator differs (it emits KV-probe queries). Normal eval and the
    optional no-context pass keep their own cadence (every eval_steps).
    """

    def __init__(self, *args, per_segment_dataset=None, per_segment_collator=None,
                 n_segments=None, segment_size=None, tokenizer=None,
                 per_segment_eval_steps=None, eval_steps=None, exp_path=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.per_segment_dataset = per_segment_dataset
        self.per_segment_collator = per_segment_collator
        self.n_segments = n_segments
        self.segment_size = segment_size
        self.tokenizer = tokenizer
        self.per_segment_eval_steps = per_segment_eval_steps
        self.eval_steps = eval_steps
        self.exp_path = exp_path
        # One-shot flag: when set, the next evaluate() runs a forgetting pass
        # regardless of the step-cadence gate, then clears itself. Used by the
        # curriculum loop to force a matrix at every stage boundary.
        self._force_forgetting = False
        self._force_stage_idx = None
        self._force_stage_label = None

    def force_forgetting_pass(self, stage_idx=None, stage_label=None):
        """Request a forgetting pass on the next evaluate(), bypassing cadence.

        Used by the curriculum loop to guarantee a forgetting matrix snapshot at
        each stage boundary. ``stage_idx`` / ``stage_label`` (when given) tag the
        resulting artifact (stage-labeled filename + JSON fields) so forced
        passes are distinguishable from periodic ones. The flag is one-shot: it
        clears after the next evaluate() consumes it.
        """
        self._force_forgetting = True
        self._force_stage_idx = stage_idx
        self._force_stage_label = stage_label

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix)

        if self.per_segment_dataset is None:
            return metrics

        step = self.state.global_step
        on_cadence = (self.per_segment_eval_steps is not None
                      and self.eval_steps is not None
                      and self.per_segment_eval_steps > 0
                      and step % self.per_segment_eval_steps == 0)
        forced = self._force_forgetting
        # consume the one-shot force BEFORE the pass (so an exception in the pass
        # still clears the flag)
        force_stage_idx = self._force_stage_idx
        force_stage_label = self._force_stage_label
        self._force_forgetting = False
        self._force_stage_idx = None
        self._force_stage_label = None

        if not (on_cadence or forced):
            return metrics

        forget_metrics = self._forgetting_pass(
            step, stage_idx=force_stage_idx if forced else None,
            stage_label=force_stage_label if forced else None,
            forced=forced)
        for k, v in forget_metrics.items():
            metrics[k] = v
        return metrics

    def _forgetting_pass(self, step, stage_idx=None, stage_label=None, forced=False):
        """Run the per-segment forgetting pass and return summary metrics + write table.

        ``stage_idx`` / ``stage_label`` / ``forced`` tag a curriculum stage-boundary
        pass: the artifact gets a stage-labeled filename and the JSON gains
        ``stage_idx`` / ``stage_label`` / ``forced`` fields, so forced passes are
        distinguishable from periodic (cadence) ones. Periodic passes pass these
        as None/False and keep the plain ``forgetting_matrix_step{N}`` name.
        """
        import json as _json
        import csv as _csv
        from pathlib import Path as _Path
        from torch.utils.data import DataLoader

        device = self.args.device if hasattr(self.args, 'device') else \
                 next(self.model.parameters()).device
        model = self.model
        # unwrap DDP/FSDP/DDP-wrapped if present
        while hasattr(model, "module"):
            model = model.module

        saved_collator = self.data_collator
        self.data_collator = self.per_segment_collator
        try:
            dl = DataLoader(self.per_segment_dataset,
                            batch_size=self.args.per_device_eval_batch_size,
                            collate_fn=self.per_segment_collator, num_workers=0)
        finally:
            self.data_collator = saved_collator

        n_seg = self.n_segments
        correct_acc = torch.zeros(n_seg, n_seg, dtype=torch.float32) if n_seg else None
        count_acc = torch.zeros(n_seg, n_seg, dtype=torch.float32) if n_seg else None
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in dl:
                ctx_ids = batch['input_ids']['context_input_ids'].to(device)
                kv_q = {k: (v.to(device) if hasattr(v, 'to') else v)
                        for k, v in batch['kv_queries'].items()}
                # forward_per_segment_eval returns FLAT per-probe arrays
                # (src_seg, probe_seg, em): one entry per (sample, KV, probe-seg).
                # A segment can hold multiple KVs, so we bin by (src,probe) cell
                # and average -- this is why we can't return a [B,n_seg,n_seg]
                # bool matrix (it would be overwritten by the last KV in a cell).
                src_seg, probe_seg, em_flags = model.forward_per_segment_eval(
                    {'context_input_ids': ctx_ids}, kv_q)
                if src_seg.numel() == 0:
                    continue
                src_seg = src_seg.cpu(); probe_seg = probe_seg.cpu(); em_flags = em_flags.cpu()
                mx = int(max(src_seg.max(), probe_seg.max())) + 1
                # grow accumulators to the largest seg index seen (segment_size mode)
                if correct_acc is None:
                    n_seg = mx
                    correct_acc = torch.zeros(mx, mx, dtype=torch.float32)
                    count_acc = torch.zeros(mx, mx, dtype=torch.float32)
                elif mx > n_seg:
                    pad = mx - n_seg
                    correct_acc = F.pad(correct_acc, (0, pad, 0, pad))
                    count_acc = F.pad(count_acc, (0, pad, 0, pad))
                    n_seg = mx
                # scatter-add: each probe contributes to its (src, probe) cell
                for s, p, ok in zip(src_seg.tolist(), probe_seg.tolist(), em_flags.tolist()):
                    correct_acc[s, p] += 1.0 if ok else 0.0
                    count_acc[s, p] += 1.0
        if was_training:
            model.train()

        with torch.no_grad():
            em = torch.where(count_acc > 0, correct_acc / count_acc.clamp_min(1),
                             torch.full_like(correct_acc, float('nan')))
        em_np = em.numpy()

        # ---- write matrix artifact (.json + .csv) ---- #
        # Forced (curriculum stage-boundary) passes get a stage-labeled filename
        # so they don't collide with periodic ones and are easy to find; periodic
        # passes keep the plain forgetting_matrix_step{N} name.
        out_dir = _Path(self.exp_path) if self.exp_path else _Path('.')
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = list(range(n_seg)); cols = list(range(n_seg))
        if forced and stage_label is not None:
            tag = f'_stage{stage_idx}_{stage_label}' if stage_idx is not None else f'_{stage_label}'
            stem = f'forgetting_matrix_step{step}{tag}'
        else:
            stem = f'forgetting_matrix_step{step}'
        artifact = {
            'step': step,
            'n_segments': n_seg,
            'rows_source_seg': rows,
            'cols_probe_seg': cols,
            'matrix_exact_match': [[None if _isnan(em_np[s, k]) else float(em_np[s, k])
                                    for k in cols] for s in rows],
            'count_per_cell': [[int(count_acc[s, k].item()) for k in cols] for s in rows],
        }
        if forced:
            artifact['forced'] = True
            if stage_idx is not None:
                artifact['stage_idx'] = stage_idx
            if stage_label is not None:
                artifact['stage_label'] = stage_label
        with open(out_dir / f'{stem}.json', 'w') as f:
            _json.dump(artifact, f, indent=2)
        with open(out_dir / f'{stem}.csv', 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['source_seg\\probe_seg'] + [f'after_seg{k}' for k in cols])
            for s in rows:
                row = [f'seg{s}'] + ['' if _isnan(em_np[s, k]) else f'{em_np[s, k]:.4f}'
                                     for k in cols]
                w.writerow(row)

        # ---- metrics injected into the returned dict ---- #
        # HF prefixes these with the eval prefix and CometCallback.on_evaluate
        # forwards them to comet as line charts, so the full matrix IS visible on
        # comet (one chartable scalar per cell), in addition to the local
        # .json/.csv artifact. We emit:
        #   forget_seg{s}_at_seg{k}  -- per-cell EM (the matrix), for every probed
        #                               (s,k); NaN cells are skipped (comet can't
        #                               chart NaN, and HF would reject the key).
        #   forget_diag_mean         -- mean of the diagonal (write quality).
        #   forget_seg0_final        -- seg0's EM at the last probe (headline
        #                               forgetting number for the oldest info).
        #   forget_slope             -- mean per-source-seg EM drop (first probe
        #                               -> last probe); positive = forgetting.
        metrics = {}
        for s in rows:
            for k in cols:
                if not _isnan(em_np[s, k]):
                    metrics[f'forget_seg{s}_at_seg{k}'] = float(em_np[s, k])
        diag = [em_np[i, i] for i in range(n_seg) if not _isnan(em_np[i, i])]
        seg0_final = em_np[0, n_seg - 1] if n_seg > 0 and not _isnan(em_np[0, n_seg - 1]) else float('nan')
        drops = []
        for s in range(n_seg):
            pts = [em_np[s, k] for k in range(s, n_seg) if not _isnan(em_np[s, k])]
            if len(pts) >= 2:
                drops.append(pts[0] - pts[-1])
        metrics['forget_diag_mean'] = float(np.mean(diag)) if diag else 0.0
        metrics['forget_seg0_final'] = float(seg0_final) if not _isnan(seg0_final) else 0.0
        metrics['forget_slope'] = float(np.mean(drops)) if drops else 0.0
        return metrics


class CurriculumTrainer(PerSegmentForgettingTrainer):
    """PerSegmentForgettingTrainer + curriculum-stage logging and step offset.

    Inherits the per-segment forgetting eval path so that, under curriculum
    learning, the forgetting matrix can fire both on its periodic cadence AND be
    forced at every stage boundary (see ``force_forgetting_pass``). When the
    ``per_segment_*`` kwargs are absent (non-forgetting configs) the forgetting
    path is a no-op, so this behaves exactly like the plain ``CustomTrainer``.
    """

    def __init__(self, *args, step_offset=0, curriculum_stage=0, curriculum_stage_label="", **kwargs):
        super().__init__(*args, **kwargs)
        self.step_offset = step_offset
        self.curriculum_stage = curriculum_stage
        self.curriculum_stage_label = curriculum_stage_label

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        logs['curriculum_stage'] = self.curriculum_stage
        logs['curriculum_stage_label'] = self.curriculum_stage_label
        original_step = self.state.global_step
        self.state.global_step += self.step_offset
        super().log(logs, start_time=start_time)
        self.state.global_step = original_step


@dataclass
class ExperimentArgs:
    config: Optional[str] = field(default=None)
    exp_path: Optional[str] = field(default=None)
    runs_dir: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    per_device_batch_size: int = field(default=64)
    data_path: str = field(default='./data/N2-K4V4-S4(32-64)_1M')
    tokenizer_path: str = field(default='./tokenizers/kv_alphabet_62/')
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    auto_find_batch_size: Optional[bool] = field(default=False)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    stop_on_em_threshold: Optional[float] = field(
        default=1.0,
        metadata={"help": "Stop training once eval exact_match >= this threshold. "
                           "Default 1.0 reproduces the prior hard stop. Set >1.0 "
                           "(e.g. 1.1) to disable the EM-stop callback."})
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_context_length: Optional[int] = field(default=None)
    # GradMemGPT parameters
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=3)
    last_K_second_order: Optional[int] = field(default=None)
    inner_lr: Optional[float] = field(default=0.01)
    use_adam: Optional[bool] = field(default=True)
    grad_mode: Optional[str] = field(default="none")
    n_ctrl_tokens: Optional[int] = field(default=0)
    inner_clip_value: Optional[float] = field(default=None)
    inner_clip_norm: Optional[float] = field(default=None)
    use_mem_proj: Optional[bool] = field(default=False)
    mem_proj_mode: Optional[str] = field(default="none")
    use_write_head: Optional[bool] = field(default=False)
    use_write_lora: Optional[bool] = field(default=False)
    write_lora_r: Optional[int] = field(default=8)
    write_lora_alpha: Optional[int] = field(default=16)
    write_lora_dropout: Optional[float] = field(default=0.0)
    write_lora_target_modules: Optional[str] = field(default=None)
    use_read_lora: Optional[bool] = field(default=False)
    read_lora_r: Optional[int] = field(default=8)
    read_lora_alpha: Optional[int] = field(default=16)
    read_lora_dropout: Optional[float] = field(default=0.0)
    read_lora_target_modules: Optional[str] = field(default=None)
    freeze_backbone: Optional[bool] = field(default=None)
    use_gradient_checkpointing: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default="eager")
    add_inner_loss_to_outer: Optional[bool] = field(default=False)
    inner_loss_weight: Optional[float] = field(default=None)
    use_hopfield_memory: Optional[bool] = field(default=False)
    hopfield_n_segments: Optional[int] = field(default=1)
    hopfield_segment_size: Optional[int] = field(default=None)
    hopfield_retrieval_mode: Optional[str] = field(default="softmax")
    hopfield_beta_init: Optional[float] = field(default=1.0)
    use_separate_hopfield_mem: Optional[bool] = field(default=False)
    hopfield_proj_dim: Optional[int] = field(default=None)
    hopfield_direct_query: Optional[bool] = field(default=False)
    hopfield_value_as_key: Optional[bool] = field(default=False)
    hopfield_value_proj_dim: Optional[int] = field(default=None)
    hopfield_bptt_segments: Optional[int] = field(default=None)
    use_gated_delta_memory: Optional[bool] = field(default=False)
    gated_delta_state_dim: Optional[int] = field(default=128)
    gated_delta_alpha_init: Optional[float] = field(default=0.9)
    gated_delta_beta_init: Optional[float] = field(default=0.5)
    gated_delta_bptt_segments: Optional[int] = field(default=None)
    memory_update: Optional[str] = field(default="gradient")
    use_mem_residual: Optional[bool] = field(default=False)
    use_reconstruction_loss: Optional[bool] = field(default=False)
    reconstruction_loss_weight: Optional[float] = field(default=1.0)
    use_energy_inner_loss: Optional[bool] = field(default=False)
    n_energy_tokens: Optional[int] = field(default=4)
    energy_mlp_hidden_dim: Optional[int] = field(default=None)
    energy_mlp_n_layers: Optional[int] = field(default=2)
    energy_readout: Optional[str] = field(default="energy_tokens")
    energy_recon_weight: Optional[float] = field(default=0.0)
    energy_recon_weight_end: Optional[float] = field(default=None)
    energy_recon_anneal_steps: Optional[int] = field(default=0)
    stabilize_energy_head: Optional[bool] = field(default=True)
    energy_out_scale: Optional[float] = field(default=1.0)
    # Reconstruction-only pretrain warmup (default-off)
    recon_pretrain_steps: Optional[int] = field(default=0)
    recon_pretrain_target_weight: Optional[float] = field(default=0.0)
    recon_pretrain_recon_weight: Optional[float] = field(default=1.0)
    # Learned inner-update head (default-off)
    use_learned_inner_update: Optional[bool] = field(default=False)
    n_learned_update_tokens: Optional[int] = field(default=4)
    learned_update_mlp_hidden_dim: Optional[int] = field(default=None)
    learned_update_mlp_n_layers: Optional[int] = field(default=2)
    learned_update_treat_as_gradient: Optional[bool] = field(default=True)
    learned_update_final_tanh: Optional[bool] = field(default=False)
    learned_update_warmup_steps: Optional[int] = field(default=0)
    learned_update_normalize: Optional[bool] = field(default=False)
    # Curriculum learning parameters
    curriculum_enabled: Optional[bool] = field(default=False)
    curriculum_threshold: Optional[float] = field(default=0.95)
    curriculum_levels: Optional[str] = field(default="4,8,16,32,64,128")
    curriculum_data_dir: Optional[str] = field(default="./data")
    curriculum_dataset_template: Optional[str] = field(default="N{n_kv}-K2V2-V62_1M")
    curriculum_stage_overrides: Optional[str] = field(default=None)
    # Dual eval: also evaluate a "without memory" (no-context) pass and report
    # the token-accuracy delta (how much writing context to memory helps).
    no_context_eval: Optional[bool] = field(default=False)
    # Path to a dataset whose 'valid_no_context' (or 'valid') split is used for
    # the no-context pass. If None, reuses the main dataset's valid_no_context /
    # valid split.
    no_context_data_path: Optional[str] = field(default=None)
    # ---- per-segment forgetting eval (adaptive model only) ---- #
    # On cadence (per_segment_eval_steps), after each model-segment is written to
    # memory, probe retrieval of every KV pair written so far to build a [n_seg,
    # n_seg] forgetting matrix (row = KV's source segment, col = probe-after-seg).
    # The full matrix is logged as a table artifact + comet table; only 3 summary
    # scalars hit the metric stream (forget_diag_mean / forget_seg0_final /
    # forget_slope). Requires the adaptive model (an `adaptive:` config section).
    per_segment_eval: Optional[bool] = field(default=False)
    per_segment_eval_steps: Optional[int] = field(default=None)  # default 5*eval_steps if None
    # ---- adaptive (segmented, gated-recurrence) fork (grad_memgpt_adaptive) ---- #
    # Selected by an `adaptive:` section in the config. Segmentation splits the
    # WRITE context into chunks written into the SAME memory (cross-segment
    # carry); memory_update_rule selects the gated recurrence. See
    # grad_memgpt_adaptive.py module docstring. Defaults reproduce the SGD path.
    n_segments: Optional[int] = field(default=1)
    segment_size: Optional[int] = field(default=None)
    seg_bptt: Optional[int] = field(default=None)
    memory_update_rule: Optional[str] = field(default="sgd")
    gate_features: Optional[str] = field(default="grad_state")
    gate_granularity: Optional[str] = field(default="per_dim")
    gate_retention: Optional[str] = field(default="exp")
    convex_bias_init: Optional[float] = field(default=3.0)
    mamba_retain_bias_init: Optional[float] = field(default=-5.0)
    mamba_write_bias_init: Optional[float] = field(default=3.0)
    # ---- VAE-style memory regularisation (default-off; adaptive fork only) ---- #
    mem_noise_std: Optional[float] = field(default=0.0)
    mem_prior_weight: Optional[float] = field(default=0.0)
    mem_prior_anneal_steps: Optional[int] = field(default=0)


def main(config_path: Optional[str] = None):
    parser = HfArgumentParser(ExperimentArgs)
    # When called programmatically (config_path provided), skip sys.argv parsing
    # to avoid conflicts with caller's CLI args (e.g., --debug from run_from_config.py)
    if config_path is not None:
        args = parser.parse_args_into_dataclasses(args=[])[0]
        args.config = config_path
    else:
        args = parser.parse_args_into_dataclasses()[0]

    # Load config from YAML if provided
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

# Flatten config to args (YAML values override ExperimentArgs defaults)
        for section in ['model', 'training', 'dataset', 'gradmem', 'hopfield', 'gated_delta', 'curriculum', 'adaptive']:
            if section in cfg:
                for key, value in cfg[section].items():
                    if key == 'stage_overrides':
                        args.curriculum_stage_overrides = json.dumps({str(k): v for k, v in value.items()})
                    elif hasattr(args, key):
                        setattr(args, key, value)

        # Set exp_path from config if not explicitly set
        if 'exp_path' not in vars(args) or args.exp_path is None:
            from generate_run_name import generate_run_name, get_exp_path, get_data_path
            exp_path = get_exp_path(cfg)
            args.exp_path = str(exp_path)

            # Set data_path from config
            dataset = cfg.get('dataset', {})
            if 'data_path' in dataset:
                args.data_path = dataset['data_path']
            elif 'data_name' in dataset:
                args.data_path = get_data_path(cfg)
            if 'tokenizer_path' in dataset:
                args.tokenizer_path = dataset['tokenizer_path']

        # Resolve the run name (manual override else auto-generated) so it can be
        # forwarded to TrainingArguments -> CometCallback sets the experiment name
        # instead of falling back to a random one.
        if args.run_name is None:
            from generate_run_name import generate_run_name
            args.run_name = cfg.get('run_name') or generate_run_name(cfg)

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    # datasets.utils.logging.set_verbosity(logger.log_level)
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"

    output_dir = Path(args.exp_path) if args.exp_path else Path('./runs/hopfield_eval')
    if accel.is_main_process:
        config_dict = {'cli_args': dict(vars(args))}
        logger.info(f'saving experiment configuration to {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(config_dict, open(output_dir / 'config.json', 'w'), indent=4)

    if args.pretrained_model is None:
        # create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        # create base model config
        if args.base_model == 'gpt2':
            config = AutoConfig.from_pretrained('gpt2')
            config.n_layer = args.n_layer
            config.n_head = args.n_head
            config.n_embd = args.n_embd
        elif args.base_model == 'pythia':
            config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.hidden_size = args.n_embd
            config.intermediate_size = config.hidden_size * 4
        elif args.base_model == 'llama':
            config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.num_key_value_heads = args.n_head
            config.hidden_size = args.n_embd
            config.head_dim = config.hidden_size // config.num_attention_heads
            config.intermediate_size = config.hidden_size * 4
        else:
            raise ValueError(f'Unsupported base model: {args.base_model}')

        config.torch_dtype = "float32"  # weights in float32, at training precision is controlled by accelerate
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.use_cache = False
    else:
        config = None
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    gradmem_config = GradMemGPTConfig(
        pretrained_model=args.pretrained_model, base_config=config,
        n_mem_tokens=args.n_mem_tokens, K=args.K,
        last_K_second_order=args.last_K_second_order,
        lr=args.inner_lr, use_adam=args.use_adam, grad_mode=args.grad_mode,
        n_ctrl_tokens=args.n_ctrl_tokens,
        inner_clip_value=args.inner_clip_value, inner_clip_norm=args.inner_clip_norm,
        use_mem_proj=args.use_mem_proj, mem_proj_mode=args.mem_proj_mode,
        use_write_head=args.use_write_head,
        use_write_lora=args.use_write_lora,
        write_lora_r=args.write_lora_r,
        write_lora_alpha=args.write_lora_alpha,
        write_lora_dropout=args.write_lora_dropout,
        write_lora_target_modules=args.write_lora_target_modules,
        use_read_lora=args.use_read_lora,
        read_lora_r=args.read_lora_r,
        read_lora_alpha=args.read_lora_alpha,
        read_lora_dropout=args.read_lora_dropout,
        read_lora_target_modules=args.read_lora_target_modules,
        freeze_backbone=args.freeze_backbone,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        add_inner_loss_to_outer=args.add_inner_loss_to_outer,
        inner_loss_weight=args.inner_loss_weight,
        use_hopfield_memory=args.use_hopfield_memory,
        hopfield_n_segments=args.hopfield_n_segments,
        hopfield_segment_size=args.hopfield_segment_size,
        hopfield_retrieval_mode=args.hopfield_retrieval_mode,
        hopfield_beta_init=args.hopfield_beta_init,
        use_separate_hopfield_mem=args.use_separate_hopfield_mem,
        hopfield_proj_dim=args.hopfield_proj_dim,
        hopfield_direct_query=args.hopfield_direct_query,
        hopfield_value_as_key=args.hopfield_value_as_key,
        hopfield_value_proj_dim=args.hopfield_value_proj_dim,
        hopfield_bptt_segments=args.hopfield_bptt_segments,
        use_gated_delta_memory=args.use_gated_delta_memory,
        gated_delta_state_dim=args.gated_delta_state_dim,
        gated_delta_alpha_init=args.gated_delta_alpha_init,
        gated_delta_beta_init=args.gated_delta_beta_init,
        gated_delta_bptt_segments=args.gated_delta_bptt_segments,
        memory_update=args.memory_update,
        use_mem_residual=args.use_mem_residual,
        use_reconstruction_loss=args.use_reconstruction_loss,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        use_energy_inner_loss=args.use_energy_inner_loss,
        n_energy_tokens=args.n_energy_tokens,
        energy_mlp_hidden_dim=args.energy_mlp_hidden_dim,
        energy_mlp_n_layers=args.energy_mlp_n_layers,
        energy_readout=args.energy_readout,
        energy_recon_weight=args.energy_recon_weight,
        energy_recon_weight_end=args.energy_recon_weight_end,
        energy_recon_anneal_steps=args.energy_recon_anneal_steps,
        stabilize_energy_head=args.stabilize_energy_head,
        energy_out_scale=args.energy_out_scale,
        recon_pretrain_steps=args.recon_pretrain_steps,
        recon_pretrain_target_weight=args.recon_pretrain_target_weight,
        recon_pretrain_recon_weight=args.recon_pretrain_recon_weight,
        use_learned_inner_update=args.use_learned_inner_update,
        n_learned_update_tokens=args.n_learned_update_tokens,
        learned_update_mlp_hidden_dim=args.learned_update_mlp_hidden_dim,
        learned_update_mlp_n_layers=args.learned_update_mlp_n_layers,
        learned_update_treat_as_gradient=args.learned_update_treat_as_gradient,
        learned_update_final_tanh=args.learned_update_final_tanh,
        learned_update_warmup_steps=args.learned_update_warmup_steps,
        learned_update_normalize=args.learned_update_normalize
    )

    # ---- model selection ------------------------------------------------ #
    # An `adaptive:` section in the config selects the segmented/gated fork
    # (grad_memgpt_adaptive.py): WRITE context is split into segments written
    # into the SAME memory (cross-segment carry), and memory_update_rule picks
    # the gated recurrence. That fork only implements the gradient inner-loop
    # path, so it does NOT accept the Hopfield/energy/learned-update kwargs
    # above (which are ignored if present under adaptive). Routing/collator/
    # metrics are shared -- only the config class + constructor differ.
    use_adaptive_model = isinstance(cfg, dict) and 'adaptive' in cfg if args.config is not None else False
    if use_adaptive_model:
        AdaptiveGM, AdaptiveGMConfig = _load_adaptive()
        gradmem_config = AdaptiveGMConfig(
            pretrained_model=args.pretrained_model, base_config=config,
            n_mem_tokens=args.n_mem_tokens, K=args.K,
            last_K_second_order=args.last_K_second_order,
            lr=args.inner_lr, use_adam=args.use_adam, grad_mode=args.grad_mode,
            n_ctrl_tokens=args.n_ctrl_tokens,
            inner_clip_value=args.inner_clip_value, inner_clip_norm=args.inner_clip_norm,
            use_mem_proj=args.use_mem_proj, mem_proj_mode=args.mem_proj_mode,
            use_write_head=args.use_write_head,
            use_write_lora=args.use_write_lora,
            write_lora_r=args.write_lora_r, write_lora_alpha=args.write_lora_alpha,
            write_lora_dropout=args.write_lora_dropout,
            write_lora_target_modules=args.write_lora_target_modules,
            freeze_backbone=args.freeze_backbone,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            attn_implementation=args.attn_implementation,
            add_inner_loss_to_outer=args.add_inner_loss_to_outer,
            inner_loss_weight=args.inner_loss_weight,
            # adaptive-only knobs
            n_segments=args.n_segments,
            segment_size=args.segment_size,
            seg_bptt=args.seg_bptt,
            memory_update_rule=args.memory_update_rule,
            gate_features=args.gate_features,
            gate_granularity=args.gate_granularity,
            gate_retention=args.gate_retention,
            convex_bias_init=args.convex_bias_init,
            mamba_retain_bias_init=args.mamba_retain_bias_init,
            mamba_write_bias_init=args.mamba_write_bias_init,
            mem_noise_std=args.mem_noise_std,
            mem_prior_weight=args.mem_prior_weight,
            mem_prior_anneal_steps=args.mem_prior_anneal_steps,
        )
        # Adaptive segments need the full (un-truncated) context so every
        # segment has real tokens. The shared collator already disables
        # truncation when use_hopfield_memory/use_gated_delta_memory is set, so
        # set the gated_delta flag to reuse that path for segmented configs too.
        if (args.n_segments > 1) or (args.segment_size is not None):
            args.use_gated_delta_memory = True
        model = AdaptiveGM(gradmem_config)
    else:
        model = GradMemGPT(gradmem_config)

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    if accel.mixed_precision == 'bf16':
        model.to(torch.bfloat16)

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')

    dataset = datasets.load_from_disk(args.data_path)

    def data_collator(batch):
        return collate_fn(batch, tokenizer, max_context_length=args.max_context_length,
                          hopfield=args.use_hopfield_memory or args.use_gated_delta_memory)

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    # No-context ("without memory") eval setup. Prefers a dedicated
    # valid_no_context split; falls back to the same valid split with the
    # no-context collator (which ignores the context field and pads).
    no_context_dataset = None
    no_context_data_collator = None
    no_context_compute_metrics = None
    if args.no_context_eval:
        if args.no_context_data_path is not None:
            no_ctx_ds = datasets.load_from_disk(args.no_context_data_path)
        else:
            no_ctx_ds = dataset
        if 'valid_no_context' in no_ctx_ds:
            no_context_dataset = no_ctx_ds['valid_no_context']
        elif 'valid' in no_ctx_ds:
            no_context_dataset = no_ctx_ds['valid']
        else:
            logger.warning('no_context_eval enabled but no valid/valid_no_context split found; skipping.')
            no_context_dataset = None

        def no_context_data_collator_fn(batch):
            return collate_fn_no_context(batch, tokenizer)

        def no_context_compute_metrics_fn(eval_pred):
            return compute_metrics_fn_no_context(eval_pred, ignore_token_ids, tokenizer)

        no_context_data_collator = no_context_data_collator_fn
        no_context_compute_metrics = no_context_compute_metrics_fn

    output_dir = Path(args.exp_path)

    if args.auto_find_batch_size:
        if args.total_batch_size is None:
            raise ValueError("total_batch_size must be specified when auto_find_batch_size is True")
        args.gradient_accumulation_steps = max(1, args.total_batch_size // (args.per_device_batch_size * accel.num_processes))
    elif args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    if args.curriculum_enabled:
        raw_levels = [x.strip() for x in args.curriculum_levels.split(',')]
        curriculum_levels = []
        for level in raw_levels:
            try:
                curriculum_levels.append(int(level))
            except ValueError:
                curriculum_levels.append(level)
        logger.info(f'curriculum learning enabled: levels={curriculum_levels}, threshold={args.curriculum_threshold}')

        stage_overrides = {}
        if args.curriculum_stage_overrides:
            raw = args.curriculum_stage_overrides
            overrides_dict = json.loads(raw) if isinstance(raw, str) else raw
            stage_overrides = {int(k): v for k, v in overrides_dict.items()}
            logger.info(f'curriculum stage overrides: {stage_overrides}')

        MODEL_PARAM_MAP = {
            'inner_lr': ('lr', 'lr'),
            'inner_clip_value': ('inner_clip_value', 'inner_clip_value'),
            'inner_clip_norm': ('inner_clip_norm', 'inner_clip_norm'),
            'hopfield_bptt_segments': ('hopfield_bptt_segments', 'hopfield_bptt_segments'),
            'hopfield_n_segments': ('hopfield_n_segments', 'hopfield_n_segments'),
            'hopfield_segment_size': ('hopfield_segment_size', 'hopfield_segment_size'),
            'hopfield_beta_init': ('hopfield_beta_init', 'hopfield_beta_init'),
            'hopfield_retrieval_mode': ('hopfield_retrieval_mode', 'hopfield_retrieval_mode'),
            'gated_delta_state_dim': ('gated_delta_state_dim', 'gated_delta_state_dim'),
            'gated_delta_alpha_init': ('gated_delta_alpha_init', 'gated_delta_alpha_init'),
            'gated_delta_beta_init': ('gated_delta_beta_init', 'gated_delta_beta_init'),
            'gated_delta_bptt_segments': ('gated_delta_bptt_segments', 'gated_delta_bptt_segments'),
        }
        TRAINING_PARAM_MAP = {
            'learning_rate', 'warmup_steps', 'weight_decay', 'per_device_batch_size',
            'max_steps', 'eval_steps', 'logging_steps', 'early_stopping_patience',
            'total_batch_size',
        }

        data_dir = Path(args.curriculum_data_dir)

        # ---- per-segment forgetting eval setup (adaptive model only) ---- #
        # Under curriculum, the forgetting matrix fires both on its periodic
        # cadence (per_segment_eval_steps) AND is forced once at every stage
        # boundary (see force_forgetting_pass below the stage loop). Setup mirrors
        # the single-stage path so the two branches stay consistent.
        use_per_segment_eval = bool(args.per_segment_eval) and use_adaptive_model
        per_segment_collator = None
        per_seg_n_segments = None
        per_seg_eval_steps = None
        if use_per_segment_eval:
            per_seg_n_segments = args.n_segments if args.segment_size is None else None
            per_segment_collator = make_collate_fn_per_segment(
                tokenizer, max_context_length=args.max_context_length,
                n_segments=args.n_segments, segment_size=args.segment_size)
            per_seg_eval_steps = args.per_segment_eval_steps
            if per_seg_eval_steps is None:
                per_seg_eval_steps = 5 * args.eval_steps
            assert per_seg_eval_steps % max(args.eval_steps, 1) == 0, \
                f"per_segment_eval_steps ({per_seg_eval_steps}) must be a multiple of " \
                f"eval_steps ({args.eval_steps})"
            logger.info(f'curriculum per-segment forgetting eval enabled: '
                        f'n_segments={args.n_segments}, segment_size={args.segment_size}, '
                        f'cadence={per_seg_eval_steps} steps (+ forced at each stage boundary)')

        all_metrics = {}
        cumulative_steps = 0
        for stage_idx, level in enumerate(curriculum_levels):
            # Apply stage overrides
            overrides = stage_overrides.get(stage_idx, {})
            if overrides:
                logger.info(f'curriculum stage {stage_idx} overrides: {overrides}')
                for param, value in overrides.items():
                    if param in MODEL_PARAM_MAP:
                        model_attr, config_attr = MODEL_PARAM_MAP[param]
                        setattr(model, model_attr, value)
                        setattr(model.config, config_attr, value)
                        logger.info(f'  override model.{model_attr} = {value}')
                    elif param in TRAINING_PARAM_MAP:
                        setattr(args, param, value)
                        logger.info(f'  override args.{param} = {value}')
                    else:
                        logger.warning(f'  override param "{param}" is not in MODEL_PARAM_MAP or TRAINING_PARAM_MAP, skipping')

            if isinstance(level, int):
                dataset_name = args.curriculum_dataset_template.format(n_kv=level)
                label = f"N{level}"
            else:
                dataset_name = level
                label = level
            data_path = data_dir / dataset_name
            logger.info(f'curriculum stage {stage_idx}/{len(curriculum_levels)-1}: label={label}, data_path={data_path}')

            stage_dataset = datasets.load_from_disk(str(data_path))

            stage_output_dir = output_dir / f'stage_{stage_idx}_{label}'
            stage_output_dir.mkdir(parents=True, exist_ok=True)

            def stage_data_collator(batch):
                return collate_fn(batch, tokenizer, max_context_length=args.max_context_length,
                                  hopfield=args.use_hopfield_memory or args.use_gated_delta_memory)

            training_args = TrainingArguments(
                output_dir=stage_output_dir,
                logging_dir=stage_output_dir,
                run_name=args.run_name,

                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_batch_size,
                per_device_eval_batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                auto_find_batch_size=args.auto_find_batch_size,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate,
                lr_scheduler_type=args.lr_scheduler_type,
                gradient_checkpointing=args.use_gradient_checkpointing,

                eval_strategy='steps',
                save_strategy='steps',
                save_steps=args.eval_steps,
                eval_steps=args.eval_steps,
                logging_steps=args.logging_steps,
                report_to='comet_ml',
                metric_for_best_model=args.metric_for_best_model,
                load_best_model_at_end=True,
                eval_on_start=True,
                greater_is_better=True,
                remove_unused_columns=False,
                include_num_input_tokens_seen=False,
                include_for_metrics=['inputs'],
                save_total_limit=1,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                seed=args.seed,
            )

            curriculum_cb = CurriculumCallback(
                metric_name='exact_match',
                threshold=args.curriculum_threshold,
            )

            trainer = CurriculumTrainer(
                model=model,
                args=training_args,
                train_dataset=stage_dataset['train'],
                eval_dataset=stage_dataset['valid'],
                data_collator=stage_data_collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                    curriculum_cb,
                ],
                step_offset=cumulative_steps,
                curriculum_stage=stage_idx,
                curriculum_stage_label=label,
                **({'per_segment_dataset': stage_dataset['valid'],
                    'per_segment_collator': per_segment_collator,
                    'n_segments': per_seg_n_segments,
                    'segment_size': args.segment_size,
                    'tokenizer': tokenizer,
                    'per_segment_eval_steps': per_seg_eval_steps,
                    'eval_steps': args.eval_steps,
                    # per-stage folder so each stage's forgetting artifacts land
                    # in stage_<idx>_<label>/ instead of the run root
                    'exp_path': str(stage_output_dir)} if use_per_segment_eval else {}),
            )

            trainer.train()
            cumulative_steps += trainer.state.global_step

            is_last_stage = (stage_idx == len(curriculum_levels) - 1)

            model = trainer.model
            best_model_path = stage_output_dir / 'best_model'
            trainer.save_model(str(best_model_path))

            if curriculum_cb.threshold_reached and not is_last_stage:
                logger.info(f'curriculum stage {stage_idx} ({label}): threshold reached, advancing to next stage')
            elif curriculum_cb.threshold_reached and is_last_stage:
                logger.info(f'curriculum complete! final stage ({label}) threshold reached')
            else:
                logger.info(f'curriculum stage {stage_idx} ({label}): threshold NOT reached, stopping curriculum')

            # Force a forgetting-matrix snapshot at this stage boundary (bypasses
            # the step-cadence gate; no-op when per-segment eval is disabled).
            if use_per_segment_eval:
                trainer.force_forgetting_pass(stage_idx=stage_idx, stage_label=label)
            metrics = trainer.evaluate(stage_dataset['valid'])
            all_metrics[f'stage_{stage_idx}_{label}'] = metrics
            logger.info(f'stage {stage_idx} ({label}) final metrics: {metrics}')

            if not curriculum_cb.threshold_reached:
                break

        logger.info('curriculum training done. running final evaluation on last completed stage...')
        final_level = curriculum_levels[min(stage_idx, len(curriculum_levels) - 1)]
        if isinstance(final_level, int):
            final_dataset_name = args.curriculum_dataset_template.format(n_kv=final_level)
        else:
            final_dataset_name = final_level
        final_dataset = datasets.load_from_disk(str(data_dir / final_dataset_name))
        final_metrics = trainer.evaluate(final_dataset['valid'])
        logger.info(f'final metrics: {final_metrics}')
        trainer.save_metrics(split='all', metrics=final_metrics)
        trainer.state.save_to_json(output_dir / 'trainer_state.json')
        with open(output_dir / 'curriculum_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
    else:
        # Single-stage training (original behavior)
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=output_dir,
            run_name=args.run_name,

            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            auto_find_batch_size=args.auto_find_batch_size,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_checkpointing=args.use_gradient_checkpointing,

            eval_strategy='steps',
            save_strategy='steps',
            save_steps=args.eval_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            report_to='comet_ml',
            metric_for_best_model=args.metric_for_best_model,
            load_best_model_at_end=True,
            eval_on_start=True,
            greater_is_better=True,
            remove_unused_columns=False,
            include_num_input_tokens_seen=False,
            include_for_metrics=['inputs'],
            save_total_limit=1,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            seed=args.seed,
        )

        # ---- per-segment forgetting eval setup (adaptive model only) ---- #
        # On cadence, builds the [n_seg,n_seg] forgetting matrix and writes it to
        # exp_path/forgetting_matrix_step{N}.json + .csv. Per-cell values are also
        # emitted as metrics (forget_seg{s}_at_seg{k}) so they reach comet as charts.
        use_per_segment_eval = bool(args.per_segment_eval) and use_adaptive_model
        per_segment_collator = None
        per_seg_n_segments = None
        per_seg_eval_steps = None
        if use_per_segment_eval:
            per_seg_n_segments = args.n_segments if args.segment_size is None else None
            per_segment_collator = make_collate_fn_per_segment(
                tokenizer, max_context_length=args.max_context_length,
                n_segments=args.n_segments, segment_size=args.segment_size)
            per_seg_eval_steps = args.per_segment_eval_steps
            if per_seg_eval_steps is None:
                per_seg_eval_steps = 5 * args.eval_steps
            assert per_seg_eval_steps % max(args.eval_steps, 1) == 0, \
                f"per_segment_eval_steps ({per_seg_eval_steps}) must be a multiple of " \
                f"eval_steps ({args.eval_steps})"
            logger.info(f'per-segment forgetting eval enabled: n_segments={args.n_segments}, '
                        f'segment_size={args.segment_size}, cadence={per_seg_eval_steps} steps')

        # ---- trainer selection ---- #
        if use_per_segment_eval:
            trainer_cls = PerSegmentForgettingTrainer
        elif args.no_context_eval:
            trainer_cls = DualEvalTrainer
        else:
            trainer_cls = CustomTrainer
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                       StopOnMetricValue(metric_name='exact_match', value=args.stop_on_em_threshold, higher_is_better=True),
                       ],
        )
        if args.no_context_eval and no_context_dataset is not None:
            trainer_kwargs.update(
                no_context_dataset=no_context_dataset,
                no_context_data_collator=no_context_data_collator,
                no_context_compute_metrics=no_context_compute_metrics,
            )
        if use_per_segment_eval:
            trainer_kwargs.update(
                per_segment_dataset=dataset['valid'],
                per_segment_collator=per_segment_collator,
                n_segments=per_seg_n_segments,
                segment_size=args.segment_size,
                tokenizer=tokenizer,
                per_segment_eval_steps=per_seg_eval_steps,
                eval_steps=args.eval_steps,
                exp_path=str(output_dir),
            )
        trainer = trainer_cls(**trainer_kwargs)
        trainer.train()
        logger.info('training done. running final evaluation...')
        metrics = trainer.evaluate(dataset['valid'])
        logger.info(f'{metrics}')
        trainer.save_metrics(split='all', metrics=metrics)
        trainer.state.save_to_json(output_dir / 'trainer_state.json')


if __name__ == '__main__':
    main()