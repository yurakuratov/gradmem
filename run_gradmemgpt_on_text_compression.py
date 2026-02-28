import json
import logging
import os
import random
from pathlib import Path

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets

import accelerate
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)

from grad_memgpt import GradMemGPT, GradMemGPTConfig


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def _sample_chunk(sample, max_context_length=None):
    context = sample['context']
    n_sentences = sample['n_sentences']
    word_count = sample['word_count']
    sentence_boundaries = sample['sentence_boundaries']
    if max_context_length is None:
        max_context_length = word_count * 1.2
    # estimate on token counts per sentence
    token_count_per_sentence = word_count * 1.2 / n_sentences
    # select end position to be enough to cover ~max_context_length
    sentence_start = random.randint(0, max(0, n_sentences - 1 - int(max_context_length / token_count_per_sentence)))
    return context[sentence_boundaries[sentence_start][0]:]


def collate_fn(batch, tokenizer, special_prefix_ids, special_suffix_ids, max_context_length=None):
    if max_context_length is None:
        raise ValueError("max_context_length must be set for text compression collate_fn")

    text_prefix_ids = tokenizer("text: ", add_special_tokens=False)["input_ids"]
    context = []
    for item in batch:
        ctx = item['context']
        if item['split'] == 'train':
            ctx = _sample_chunk(item, max_context_length=max_context_length)
        if max_context_length is not None:
            # chunks are long, so we can cut them for faster tokenization
            ctx = ctx[:max_context_length*15]
        context += [ctx]

    context_ids = tokenizer(context, add_special_tokens=False,
                            max_length=max_context_length, truncation=True)["input_ids"]

    input_ids = []
    labels = []
    non_target_len = len(special_prefix_ids) + len(text_prefix_ids)
    suffix_len = len(special_suffix_ids)

    for ctx_ids in context_ids:
        full_ids = special_prefix_ids + text_prefix_ids + ctx_ids + special_suffix_ids
        labels_ids = list(full_ids)
        labels_ids[:non_target_len] = [-100] * non_target_len
        if suffix_len > 0:
            labels_ids[-suffix_len:] = [-100] * suffix_len
        input_ids += [full_ids]
        labels += [labels_ids]

    seq_lens = {len(x) for x in input_ids}
    if len(seq_lens) != 1:
        raise ValueError(f"All samples in batch must have equal sequence length; got {sorted(seq_lens)}")

    context_input_ids = torch.tensor(input_ids, dtype=torch.long)
    query_input_ids = context_input_ids
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
        'labels': labels,
    }


def preprocess_logits_for_metrics(eval_pred, labels):
    logits, inner_loop_stats = eval_pred
    return (logits.argmax(dim=-1), inner_loop_stats)


def compute_metrics_fn(eval_pred, tokenizer, debug_print_samples=0):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    preds, inner_loop_stats = predictions
    # Prefix memory predicts the first token from memory (pred_len = label_len + 1).
    # LoRA memory without prepended seed token cannot predict the first token from memory alone
    # in this text-compression setup, so metrics here are not directly comparable across backends.
    pred_len = preds.shape[1]
    label_len = labels.shape[1]
    if pred_len == label_len + 1:
        preds = preds[:, :-1]
        labels = labels[:, :]
    elif pred_len == label_len:
        preds = preds[:, :-1]
        labels = labels[:, 1:]
    else:
        raise ValueError(f"Unexpected prediction/label lengths: pred_len={pred_len}, label_len={label_len}")

    mask = (labels != -100)
    masked_predictions = preds[mask]
    masked_labels = labels[mask]
    accuracy = (masked_predictions == masked_labels).mean()

    exact_match = np.mean([
        np.all(pred[mask[i]] == lab[mask[i]])
        for i, (pred, lab) in enumerate(zip(preds, labels))
        if np.any(mask[i])
    ])

    if debug_print_samples > 0:
        for pred, label, inp_c, inp_q in zip(preds[:debug_print_samples], labels[:debug_print_samples],
                                             inputs['context_input_ids'][:debug_print_samples],
                                             inputs['query_input_ids'][:debug_print_samples]):
            mask = (label != -100)
            pred = pred[mask]
            inp_c[inp_c == -100] = tokenizer.pad_token_id
            inp_q[inp_q == -100] = tokenizer.pad_token_id
            label[label == -100] = tokenizer.pad_token_id
            print('i:', tokenizer.decode(inp_c, skip_special_tokens=True).strip())
            print('p:', tokenizer.decode(pred, skip_special_tokens=True).strip())
            print('t:', tokenizer.decode(label, skip_special_tokens=True).strip())
            print('-' * 50)

    metrics = {
        'token_accuracy': accuracy,
        'exact_match': exact_match,
    }
    for k, v in inner_loop_stats.items():
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


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        num_training_steps = int(num_training_steps / 0.9)  # to make final lr not zero, for linear it is lr/10.
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # log early stopping patience
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    dataset_path: str = field()
    train_split: Optional[str] = field(default="train")
    eval_split: Optional[str] = field(default=None)
    max_eval_samples: Optional[int] = field(default=5000)
    max_context_length: Optional[int] = field(default=256)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    pretrained_model: Optional[str] = field(default="EleutherAI/pythia-160m")
    init_checkpoint: Optional[str] = field(default=None)
    # GradMemGPT parameters
    memory_backend: Optional[str] = field(default="prefix")
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=2)
    last_K_second_order: Optional[int] = field(default=None)
    inner_lr: Optional[float] = field(default=0.08)
    use_adam: Optional[bool] = field(default=False)
    grad_mode: Optional[str] = field(default="second")
    n_ctrl_tokens: Optional[int] = field(default=0)
    inner_clip_value: Optional[float] = field(default=None)
    inner_clip_norm: Optional[float] = field(default=None)
    use_mem_proj: Optional[bool] = field(default=True)
    mem_proj_mode: Optional[str] = field(default="proj")
    use_write_head: Optional[bool] = field(default=True)
    use_write_lora: Optional[bool] = field(default=True)
    write_lora_r: Optional[int] = field(default=8)
    write_lora_alpha: Optional[int] = field(default=16)
    write_lora_dropout: Optional[float] = field(default=0.0)
    write_lora_target_modules: Optional[str] = field(default=None)
    lora_mem_placement: Optional[str] = field(default="between_layers")
    lora_mem_r: Optional[int] = field(default=8)
    lora_mem_alpha: Optional[int] = field(default=16)
    lora_mem_dropout: Optional[float] = field(default=0.0)
    lora_mem_layers: Optional[str] = field(default="all")
    lora_mem_target_modules: Optional[str] = field(default=None)
    kv_mem_layers: Optional[str] = field(default="all")
    freeze_backbone: Optional[bool] = field(default=True)
    use_gradient_checkpointing: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default="eager")
    add_inner_loss_to_outer: Optional[bool] = field(default=False)
    inner_loss_weight: Optional[float] = field(default=None)
    debug_print_samples: Optional[int] = field(default=5)


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    if accel.is_main_process:
        config = {
            'cli_args': dict(vars(args)),
        }
        logger.info(f'saving experiment configuration to {args.exp_path}')
        Path(args.exp_path).mkdir(parents=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # infer special tokens that wrap text from add_special_tokens behavior
    probe_raw = tokenizer("Hi", add_special_tokens=False)["input_ids"]
    probe_full = tokenizer("Hi", add_special_tokens=True)["input_ids"]
    start_idx = None
    for i in range(len(probe_full) - len(probe_raw) + 1):
        if probe_full[i:i + len(probe_raw)] == probe_raw:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(
            "Could not infer tokenizer special-token wrapper from add_special_tokens behavior. "
            f"probe_raw={probe_raw}, probe_full={probe_full}"
        )
    special_prefix_ids = probe_full[:start_idx]
    special_suffix_ids = probe_full[start_idx + len(probe_raw):]

    dataset = datasets.load_from_disk(args.dataset_path)
    if not isinstance(dataset, datasets.DatasetDict):
        raise ValueError("Prepared dataset must be a DatasetDict saved with save_to_disk.")

    train_split = args.train_split or "train"
    if train_split not in dataset:
        train_split = list(dataset.keys())[0]
    train_ds = dataset[train_split]

    if args.eval_split and args.eval_split in dataset:
        eval_split = args.eval_split
    elif "validation" in dataset:
        eval_split = "validation"
    elif "valid" in dataset:
        eval_split = "valid"
    elif "test" in dataset:
        eval_split = "test"
    else:
        raise ValueError("No eval split found. Provide --eval_split.")

    eval_ds = dataset[eval_split]
    for f in ("context", ):
        if f not in train_ds.column_names:
            raise ValueError(f"Prepared dataset must include '{f}' column.")

    if args.max_eval_samples is not None and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.select(range(args.max_eval_samples))

    dataset = datasets.DatasetDict(train=train_ds, valid=eval_ds)

    gradmem_config = GradMemGPTConfig(pretrained_model=args.pretrained_model,
                                      memory_backend=args.memory_backend,
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
                                      lora_mem_placement=args.lora_mem_placement,
                                      lora_mem_r=args.lora_mem_r,
                                      lora_mem_alpha=args.lora_mem_alpha,
                                      lora_mem_dropout=args.lora_mem_dropout,
                                      lora_mem_layers=args.lora_mem_layers,
                                      lora_mem_target_modules=args.lora_mem_target_modules,
                                      kv_mem_layers=args.kv_mem_layers,
                                      freeze_backbone=args.freeze_backbone,
                                      use_gradient_checkpointing=args.use_gradient_checkpointing,
                                      attn_implementation=args.attn_implementation,
                                      add_inner_loss_to_outer=args.add_inner_loss_to_outer,
                                      inner_loss_weight=args.inner_loss_weight)

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

    def data_collator(batch):
        return collate_fn(batch, tokenizer,
                          special_prefix_ids=special_prefix_ids,
                          special_suffix_ids=special_suffix_ids,
                          max_context_length=args.max_context_length)

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, tokenizer, debug_print_samples=args.debug_print_samples)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        report_to='tensorboard',
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

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   StopOnMetricValue(metric_name='exact_match', value=1.0, higher_is_better=True),
                   ],
    )

    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')
