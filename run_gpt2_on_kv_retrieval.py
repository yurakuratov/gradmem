import json
import logging
import os
from pathlib import Path

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets

import accelerate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)

from kv_dataset_utils import create_tokenizer


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer):
    seq = [item['context'] + item['query'] + item['target'] for item in batch]
    input_ids = tokenizer(seq, return_tensors="pt", add_special_tokens=False,
                          padding=True, pad_to_multiple_of=8).input_ids
    # add labels_mask
    # input_seq: 0, target_seq: 1, seq = input_seq + target_seq
    labels_mask = torch.zeros_like(input_ids)
    for i, item in enumerate(batch):
        input_seq_len = len(item['context']) + len(item['query'])
        target_seq_len = len(item['target'])
        labels_mask[i, input_seq_len:input_seq_len+target_seq_len] = 1

    labels = input_ids * labels_mask + (1 - labels_mask) * -100
    return {
        'input_ids': input_ids,
        'labels': labels,
    }


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    logits, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    logits = logits[..., :-1, :]
    labels = labels[..., 1:]
    predictions = np.argmax(logits, axis=-1)

    # Create a mask for tokens that are not padding (-100) and ignored tokens (like ! and |)
    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)
    # Calculate token-level accuracy only on content tokens
    masked_predictions = predictions[mask]
    masked_labels = labels[mask]

    accuracy = (masked_predictions == masked_labels).mean()

    # get exact_match per-sample accuracy, ignore masked tokens
    # predictions.shape = (batch_size, seq_len)
    exact_match = np.mean([
        np.all(pred[mask[i]] == lab[mask[i]])
        for i, (pred, lab) in enumerate(zip(predictions, labels))
        if np.any(mask[i])  # Skip samples that are all masked
    ])

    for pred, label, inp in zip(predictions[:5], labels[:5], inputs[:5]):
        mask = (label != -100)
        pred = pred[mask]
        inp[inp == -100] = 0
        label[label == -100] = 0
        print('i:', tokenizer.decode(inp, skip_special_tokens=True).replace(' ', ''))
        print('p:', tokenizer.decode(pred, skip_special_tokens=True).replace(' ', ''))
        print('t:', tokenizer.decode(label, skip_special_tokens=True).replace(' ', ''))
        print('-' * 50)

    return {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }


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
    data_path: str = field(
        default='/home/jovyan/kuratov/data/test_time_gd/data/N2-K4V4-S4(32-64)_1M',
    )
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
    base_model: Optional[str] = field(default='gpt2')
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    # datasets.utils.logging.set_verbosity(logger.log_level)
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    if accel.is_main_process:
        config = {
            'cli_args': dict(vars(args)),
        }
        logger.info('saving experiment configuration..')
        Path(args.exp_path).mkdir(parents=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    # create tokenizer
    tokenizer = create_tokenizer()

    # create model config
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
    config.vocab_size = 128
    config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    config.bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
    config.eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')

    # create model
    model = AutoModelForCausalLM.from_config(config)

    dataset = datasets.load_from_disk(args.data_path)

    def data_collator(batch):
        return collate_fn(batch, tokenizer)

    # Target sequence looks like: "XXXX!|"
    # Let's not count ! and | in the accuracy calculation
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    # Define custom compute metrics function with ignore tokens
    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    # Training arguments
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
        include_num_input_tokens_seen=True,
        include_for_metrics=['inputs'],
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   StopOnMetricValue(metric_name='exact_match', value=1.0, higher_is_better=True),
                   ],
    )
    # Train the model
    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
