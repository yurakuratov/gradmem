import json
import logging
import os
import re
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

from ttt import TTTConfig, TTTForCausalLM


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer):
    seq = [item['context'] + item['query'] + item['target'] for item in batch]
    seq_encoded = tokenizer(seq, return_tensors="pt", add_special_tokens=True,
                            padding=True, pad_to_multiple_of=8, return_offsets_mapping=True)
    input_ids = seq_encoded['input_ids']
    offsets_mapping = seq_encoded['offset_mapping']

    attn_mask = (input_ids != tokenizer.pad_token_id).to(dtype=torch.long)

    labels_mask = torch.zeros_like(input_ids)
    for i, item in enumerate(batch):
        input_seq_len = len(item['context']) + len(item['query'])
        target_seq_len = len(item['target'])
        target_st, target_end = input_seq_len, input_seq_len + target_seq_len

        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            if st < target_end and end > target_st:
                labels_mask[i, j] = 1
                in_target = True
            elif in_target:
                break

    labels = input_ids * labels_mask + (1 - labels_mask) * -100
    return {
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'labels': labels,
    }


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs

    predictions = predictions[..., :-1]
    labels = labels[..., 1:]

    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)

    masked_predictions = predictions[mask]
    masked_labels = labels[mask]

    accuracy = (masked_predictions == masked_labels).mean()

    exact_match = np.mean([
        np.all(pred[mask[i]] == lab[mask[i]])
        for i, (pred, lab) in enumerate(zip(predictions, labels))
        if np.any(mask[i])
    ])

    for pred, label, inp in zip(predictions[:5], labels[:5], inputs[:5]):
        m = (label != -100)
        pred_m = pred[m]
        inp[inp == -100] = tokenizer.pad_token_id
        label[label == -100] = tokenizer.pad_token_id
        print('i:', tokenizer.decode(inp, skip_special_tokens=True).strip())
        print('p:', tokenizer.decode(pred_m, skip_special_tokens=True).strip())
        print('t:', tokenizer.decode(label, skip_special_tokens=True).strip())
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
        num_training_steps = int(num_training_steps / 0.9)
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    data_path: str = field(default='N2-K2V2-V62')
    tokenizer_path: str = field(default='./tokenizers/kv_alphabet_62/')
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=3e-04)
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.999)
    adam_epsilon: Optional[float] = field(default=1e-8)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    init_checkpoint: Optional[str] = field(default=None)
    model_cpt: Optional[str] = field(default=None)
    # TTT architecture parameters (mapped from ARMT defaults)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    # TTT-specific parameters
    ttt_layer_type: Optional[str] = field(default='linear')
    ttt_base_lr: Optional[float] = field(default=1.0)
    mini_batch_size: Optional[int] = field(default=16)
    pre_conv: Optional[bool] = field(default=False)
    conv_kernel: Optional[int] = field(default=4)
    use_gate: Optional[bool] = field(default=False)
    share_qk: Optional[bool] = field(default=False)
    scan_checkpoint_group_size: Optional[int] = field(default=0)


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
        exp_config = {
            'cli_args': dict(vars(args)),
        }
        logger.info(f'saving experiment configuration to {args.exp_path}')
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump(exp_config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def build_ttt_config():
        return TTTConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.n_embd,
            intermediate_size=args.n_embd * 4,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
            use_gate=args.use_gate,
            share_qk=args.share_qk,
            ttt_layer_type=args.ttt_layer_type,
            ttt_base_lr=args.ttt_base_lr,
            mini_batch_size=args.mini_batch_size,
            pre_conv=args.pre_conv,
            conv_kernel=args.conv_kernel,
            scan_checkpoint_group_size=args.scan_checkpoint_group_size,
        )

    if args.model_cpt is not None and args.model_cpt != 'None':
        checkpoint_dir = args.model_cpt
        if os.path.isdir(args.model_cpt):
            checkpoint_dirs = []
            files = os.listdir(args.model_cpt)
            logger.info(f'Found {len(files)} files in {args.model_cpt}: {files}')
            for item in files:
                item_path = os.path.join(args.model_cpt, item)
                if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                    match = re.match(r'checkpoint-(\d+)', item)
                    if match:
                        step_num = int(match.group(1))
                        checkpoint_dirs.append((step_num, item_path))

            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
                latest_step, checkpoint_dir = checkpoint_dirs[0]
                logger.info(f'Found {len(checkpoint_dirs)} checkpoint(s), loading from latest: checkpoint-{latest_step}')
            else:
                checkpoint_dir = args.model_cpt
                logger.info(f'No checkpoint-* directories found in {args.model_cpt}, using directory directly')

        logger.info(f'Loading TTT model from checkpoint: {checkpoint_dir}')
        try:
            model = TTTForCausalLM.from_pretrained(checkpoint_dir)
            logger.info(f'Successfully loaded TTT model from {checkpoint_dir}')
        except Exception as e:
            logger.warning(f'Failed to load as pretrained model: {e}')
            logger.info('Trying to load from state dict files...')
            ttt_config = build_ttt_config()
            model = TTTForCausalLM(ttt_config)
            checkpoint_paths = [
                os.path.join(checkpoint_dir, "model.safetensors"),
                os.path.join(checkpoint_dir, "pytorch_model.bin"),
                checkpoint_dir,
            ]
            loaded = False
            for cpt_path in checkpoint_paths:
                if os.path.exists(cpt_path):
                    logger.info(f'Loading from: {cpt_path}')
                    if cpt_path.endswith('.safetensors'):
                        state_dict = load_file(cpt_path)
                    else:
                        state_dict = torch.load(cpt_path, map_location='cpu')
                        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                    missing_k, unexpected_k = model.load_state_dict(state_dict, strict=False)
                    if len(missing_k) != 0:
                        logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
                    if len(unexpected_k) != 0:
                        logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f'Could not find checkpoint file in {checkpoint_dir}. Tried: {checkpoint_paths}')
    else:
        ttt_config = build_ttt_config()
        model = TTTForCausalLM(ttt_config)

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    device = accel.device
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f'model parameter count: {param_count:,}')
    if param_count == 0:
        raise RuntimeError("Model has no parameters! Check model initialization.")

    if accel.mixed_precision == 'bf16':
        model = model.to(torch.bfloat16)

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')

    if os.path.exists(args.data_path):
        logger.info(f"Loading dataset from disk: {args.data_path}")
        dataset = datasets.load_from_disk(args.data_path)
    else:
        logger.info(f"Loading dataset from HuggingFace Hub: irodkin/kv_retrieval (subset: {args.data_path})")
        dataset = datasets.load_dataset("irodkin/kv_retrieval", name=args.data_path)

    def data_collator(batch):
        return collate_fn(batch, tokenizer)

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    wandb_run_name = os.environ.get('WANDB_NAME', None)
    if wandb_run_name:
        logger.info(f'Using WANDB_NAME from environment: {wandb_run_name}')

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        label_names=['labels'],
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,

        eval_strategy='steps',
        save_strategy='steps',
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to='wandb',
        run_name=wandb_run_name,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
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
