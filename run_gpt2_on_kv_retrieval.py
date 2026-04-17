import json
import logging
import os
from pathlib import Path
import math
import yaml

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets

import accelerate
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, TrainerState,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)

from transformers.trainer_utils import get_last_checkpoint

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer, max_input_length=None):
    seq = [item['context'] + item['query'] + item['target'] for item in batch]
    seq_encoded = tokenizer(seq, return_tensors="pt", add_special_tokens=True,
                            padding=True, pad_to_multiple_of=8, max_length=max_input_length, truncation=True,
                            return_offsets_mapping=True)
    input_ids = seq_encoded['input_ids']
    offsets_mapping = seq_encoded['offset_mapping']

    attn_mask = (input_ids != tokenizer.pad_token_id).to(dtype=torch.long)
    # add labels_mask
    # input_seq: 0, target_seq: 1, seq = input_seq + target_seq
    labels_mask = torch.zeros_like(input_ids)
    for i, item in enumerate(batch):
        input_seq_len = len(item['context']) + len(item['query'])
        target_seq_len = len(item['target'])
        target_st, target_end = input_seq_len, input_seq_len + target_seq_len

        # find target tokens
        # since target is closer to the end, search from the end
        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            # if (target_st, target_end) intersects with (st, end), it is a target token
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
    # saves gpu RAM, as HF Trainer accumulates all eval logits on GPU
    return logits.argmax(dim=-1)


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs

    # shift for lm loss
    predictions = predictions[..., :-1]
    labels = labels[..., 1:]

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
        inp[inp == -100] = tokenizer.pad_token_id
        label[label == -100] = tokenizer.pad_token_id
        print('i:', tokenizer.decode(inp, skip_special_tokens=True).strip())
        print('p:', tokenizer.decode(pred, skip_special_tokens=True).strip())
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
    config: Optional[str] = field(default=None)
    exp_path: Optional[str] = field(default=None)
    per_device_batch_size: int = field(default=64)
    data_path: str = field(
        default='./data/N2-K4V4-S4(32-64)_1M',
    )
    tokenizer_path: str = field(
        default='./tokenizers/kv_alphabet_62/',
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
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.999)
    adam_epsilon: Optional[float] = field(default=1e-8)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_position_embeddings: Optional[int] = field(default=None)
    max_input_length: Optional[int] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)

    # allow writing to existing folder & resume
    overwrite_output_dir: Optional[bool] = field(default=False)
    # skip training; evaluate best checkpoint
    do_eval_only: Optional[bool] = field(default=False)


if __name__ == '__main__':
    main()


def main(config_path: Optional[str] = None):
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    # Load config from YAML if provided
    if config_path is not None:
        args.config = config_path

    # Load config from YAML if provided
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # Flatten config to args (CLI args override YAML)
        for section in ['model', 'training', 'dataset']:
            if section in cfg:
                for key, value in cfg[section].items():
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, value)

        # Set exp_path from config if not explicitly set
        if args.exp_path is None:
            from generate_run_name import generate_run_name, get_exp_path, get_data_path
            run_name = generate_run_name(cfg)
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

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    # datasets.utils.logging.set_verbosity(logger.log_level)
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"

    output_dir = Path(args.exp_path)
    if accel.is_main_process and output_dir.exists() and not args.overwrite_output_dir and not args.do_eval_only:
        raise RuntimeError(f"Output directory already exists: {output_dir}. "
                           f"Pass --overwrite_output_dir to resume/continue here, or choose a new --exp_path.")

    if accel.is_main_process and not args.do_eval_only:
        config = {
            'cli_args': dict(vars(args)),
        }
        logger.info('saving experiment configuration..')
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    if accel.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    elif accel.mixed_precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32
        args.attn_implementation = None

    if args.pretrained_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, torch_dtype=dtype,
                                                     attn_implementation=args.attn_implementation)
    else:
        # create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        # create model config
        if args.base_model == 'gpt2':
            config = AutoConfig.from_pretrained('gpt2')
            config.n_layer = args.n_layer
            config.n_head = args.n_head
            config.n_embd = args.n_embd
            if args.max_position_embeddings is not None:
                config.n_positions = args.max_position_embeddings
        elif args.base_model == 'pythia':
            config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.hidden_size = args.n_embd
            config.intermediate_size = config.hidden_size * 4
            if args.max_position_embeddings is not None:
                config.max_position_embeddings = args.max_position_embeddings
        elif args.base_model == 'llama':
            config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.num_key_value_heads = args.n_head
            config.hidden_size = args.n_embd
            config.head_dim = config.hidden_size // config.num_attention_heads
            config.intermediate_size = config.hidden_size * 4
            if args.max_position_embeddings is not None:
                config.rope_scaling = None
                config.rope_theta = 10000.0
                config.max_position_embeddings = args.max_position_embeddings
        elif args.base_model == 'mamba':
            config = AutoConfig.from_pretrained('state-spaces/mamba-130m-hf')
            config.num_hidden_layers = args.n_layer
            config.n_layer = args.n_layer
            config.hidden_size = args.n_embd
            config.d_model = args.n_embd
            config.expand = 4
            config.intermediate_size = config.expand * config.hidden_size
            config.d_inner = config.expand * config.hidden_size
            config.time_step_rank = math.ceil(config.hidden_size / 16)
            args.attn_implementation = None
        else:
            raise ValueError(f'Unsupported base model: {args.base_model}')

        config.torch_dtype = dtype  # weights in float32, at training precision is controlled by accelerate
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = 'left'  # to truncate context tokens, not query or target
        # create model
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype,
                                                 attn_implementation=args.attn_implementation)

    model.config.use_cache = False

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')
    logger.info(f'attn_implementation: {args.attn_implementation}')

    dataset = datasets.load_from_disk(args.data_path)

    def data_collator(batch):
        return collate_fn(batch, tokenizer, max_input_length=args.max_input_length)

    # Target sequence looks like: "XXXX!|"
    # Let's not count ! and | in the accuracy calculation
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    # Define custom compute metrics function with ignore tokens
    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        overwrite_output_dir=args.overwrite_output_dir,

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
        report_to='comet_ml',
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
        include_for_metrics=['inputs'],
        save_total_limit=1,
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
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   StopOnMetricValue(metric_name='exact_match', value=1.0, higher_is_better=True),
                   ],
    )

    last_ckpt = get_last_checkpoint(output_dir)

    if args.do_eval_only:
        # load best checkpoint and run eval
        try:
            state_path = Path(last_ckpt) / "trainer_state.json"
            trainer.state = TrainerState.load_from_json(state_path)
            trainer._load_best_model()
            logger.info(f'Successfully loaded best model from {trainer.state.best_model_checkpoint}')
        except Exception as e:
            logger.error(f"Failed to load best model from {output_dir}: {e}")
            exit(1)
    else:
        if last_ckpt:
            logger.info(f'Resuming training from last checkpoint: {last_ckpt}')
        # run training
        trainer.train(resume_from_checkpoint=last_ckpt)
        logger.info('training done. running final evaluation...')
    # run final evaluation
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    if not args.do_eval_only:
        trainer.state.save_to_json(output_dir / 'trainer_state.json')
