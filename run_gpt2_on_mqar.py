import json
import logging
import os
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field

import accelerate
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer, TrainerState,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)

from transformers.trainer_utils import get_last_checkpoint
from zoology_mqar_data import (
    ZOOLOGY_MQAR_SOURCE,
    build_mqar_datasets,
    load_saved_mqar_datasets,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        # MQAR examples are fixed-length; token 0 may be a noisy-pair frame.
        'attention_mask': torch.ones_like(input_ids),
        'labels': labels,
    }


def tensor_batch_to_numpy(batch):
    if isinstance(batch, dict):
        return {key: tensor_batch_to_numpy(value) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.cpu().numpy().copy()
    return batch


def collate_fn_numpy(batch):
    return tensor_batch_to_numpy(collate_fn(batch))


def preprocess_logits_for_metrics(logits, labels):
    # saves gpu RAM, as HF Trainer accumulates all eval logits on GPU
    return logits.argmax(dim=-1)


def compute_metrics_fn(eval_pred):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    mask = (labels != -100)
    if not np.all(mask.any(axis=1)):
        raise ValueError('Every MQAR example must contain at least one supervised query.')

    correct = predictions[mask] == labels[mask]
    token_accuracy = float(correct.mean())
    per_example_token_accuracy = [
        (prediction[example_mask] == label[example_mask]).mean()
        for prediction, label, example_mask in zip(predictions, labels, mask)
    ]
    all_queries_exact_match = [
        np.all(prediction[example_mask] == label[example_mask])
        for prediction, label, example_mask in zip(predictions, labels, mask)
    ]

    for pred, label, inp, example_mask in zip(predictions[:5], labels[:5], inputs[:5], mask[:5]):
        print('i:', np.asarray(inp).tolist())
        print('q:', np.asarray(inp[example_mask]).tolist())
        print('p:', np.asarray(pred[example_mask]).tolist())
        print('t:', np.asarray(label[example_mask]).tolist())
        print('-' * 50)

    return {
        # Every MQAR answer is one token, so per-query EM equals token accuracy.
        'token_accuracy': token_accuracy,
        'exact_match': token_accuracy,
        'all_queries_token_accuracy': float(np.mean(per_example_token_accuracy)),
        'all_queries_exact_match': float(np.mean(all_queries_exact_match)),
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

    def _prepare_input(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return super()._prepare_input(data)

    def floating_point_ops(self, inputs):
        main_input_name = getattr(self.model, "main_input_name", "input_ids")
        if isinstance(inputs.get(main_input_name), np.ndarray):
            inputs = dict(inputs)
            inputs[main_input_name] = torch.from_numpy(inputs[main_input_name])
        return super().floating_point_ops(inputs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # log early stopping patience
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


class MQARTrainer(CustomTrainer):
    """Use Zoology's same-position masked loss without a causal label shift."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = dict(inputs)
        labels = model_inputs.pop('labels')
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        return (loss, outputs) if return_outputs else loss


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    vocab_size: Optional[int] = field(default=8192)
    input_seq_len: Optional[int] = field(default=24)
    num_kv_pairs: Optional[int] = field(default=8)
    train_num_examples: Optional[int] = field(default=100_000)
    valid_num_examples: Optional[int] = field(default=3_000)
    power_a: Optional[float] = field(default=0.01)
    random_non_queries: Optional[bool] = field(default=False)
    mqar_noise_lvl: Optional[float] = field(default=0.0)
    mqar_data_path: Optional[str] = field(default=None)
    data_seed: Optional[int] = field(default=123)
    dense_queries: Optional[bool] = field(default=True)
    query_sampling: Optional[str] = field(default='uniform',
                                          metadata={'help': 'Dense query sampling: uniform, power_law, or zoology.'}
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
    max_grad_norm: Optional[float] = field(default=1.0)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    stop_on_metric_value: Optional[float] = field(default=1.0)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_position_embeddings: Optional[int] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    attention_dropout: Optional[float] = field(default=0.0)

    # allow writing to existing folder & resume
    overwrite_output_dir: Optional[bool] = field(default=False)
    # skip training; evaluate best checkpoint
    do_eval_only: Optional[bool] = field(default=False)


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

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"

    output_dir = Path(args.exp_path)
    if accel.is_main_process and output_dir.exists() and not args.overwrite_output_dir and not args.do_eval_only:
        raise RuntimeError(f"Output directory already exists: {output_dir}. "
                           f"Pass --overwrite_output_dir to resume/continue here, or choose a new --exp_path.")

    if args.mqar_data_path is None:
        train_dataset, valid_dataset, train_data_seed, valid_data_seed = build_mqar_datasets(
            vocab_size=args.vocab_size,
            input_seq_len=args.input_seq_len,
            num_kv_pairs=args.num_kv_pairs,
            train_num_examples=args.train_num_examples,
            valid_num_examples=args.valid_num_examples,
            power_a=args.power_a,
            random_non_queries=args.random_non_queries,
            data_seed=args.data_seed,
            dense_queries=args.dense_queries,
            query_sampling=args.query_sampling,
            mqar_noise_lvl=args.mqar_noise_lvl,
        )
        dataset_metadata = train_dataset.slices
    else:
        train_dataset, valid_dataset, dataset_metadata = load_saved_mqar_datasets(args.mqar_data_path)
        for key, expected in (
            ('vocab_size', args.vocab_size),
            ('num_kv_pairs', args.num_kv_pairs),
            ('dense_queries', args.dense_queries),
            ('query_sampling', args.query_sampling),
            ('mqar_noise_lvl', args.mqar_noise_lvl),
        ):
            if key in dataset_metadata and dataset_metadata[key] != expected:
                raise ValueError(
                    f'Saved MQAR metadata mismatch for {key}: '
                    f'saved={dataset_metadata[key]!r}, requested={expected!r}'
                )
        train_data_seed = dataset_metadata.get('train_data_seed')
        valid_data_seed = dataset_metadata.get('valid_data_seed')
        args.train_num_examples = len(train_dataset)
        args.valid_num_examples = len(valid_dataset)
        args.input_seq_len = dataset_metadata['input_seq_len']

    if accel.is_main_process and not args.do_eval_only:
        noise_enabled = dataset_metadata.get('noise_tokens', 0) > 0
        if args.dense_queries:
            query_layout = (
                'dense context with framed KV pairs and interleaved random noise'
                if noise_enabled else
                'dense context and contiguous query keys; no fillers'
            )
        else:
            query_layout = (
                'upstream power-law query placement with framed KV pairs and interleaved random noise'
                if noise_enabled else
                'upstream power-law query placement'
            )
        config = {
            'cli_args': dict(vars(args)),
            'task_source': ZOOLOGY_MQAR_SOURCE,
            'task': 'Zoology MQAR' + (' dense' if args.dense_queries else ' upstream'),
            'query_layout': query_layout,
            'query_sampling': dataset_metadata.get('query_sampling'),
            'data_generator': dataset_metadata.get('generator', 'saved' if args.mqar_data_path else 'upstream'),
            'query_distribution': dataset_metadata.get(
                'query_distribution', 'power_law_placement'
            ),
            'effective_power_a': dataset_metadata.get('effective_power_a', args.power_a),
            'loss_alignment': 'same-position masked cross entropy; no causal label shift',
            'context_size': dataset_metadata.get('context_size'),
            'query_size': dataset_metadata['input_seq_len'] - dataset_metadata.get('context_size'),
            'input_seq_len': dataset_metadata['input_seq_len'],
            'context_noise': {
                'level': args.mqar_noise_lvl,
                'tokens': dataset_metadata.get('noise_tokens', 0),
                'pair_open_token': dataset_metadata.get('pair_open_token'),
                'pair_close_token': dataset_metadata.get('pair_close_token'),
            },
            'train_data_seed': train_data_seed,
            'valid_data_seed': valid_data_seed,
            'metrics': [
                'token_accuracy',
                'exact_match',
                'all_queries_token_accuracy',
                'all_queries_exact_match',
            ],
        }
        logger.info('saving experiment configuration..')
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(config, open(output_dir / 'config.json', 'w'), indent=4)

    if accel.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    elif accel.mixed_precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32
        args.attn_implementation = None

    if args.pretrained_model is not None:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, torch_dtype=dtype,
                                                     attn_implementation=args.attn_implementation)
        if model.config.vocab_size != args.vocab_size:
            logger.info(
                f'resizing pretrained_model vocabulary from {model.config.vocab_size} '
                f'to MQAR vocab_size {args.vocab_size}'
            )
            model.resize_token_embeddings(args.vocab_size)
    else:
        # create model config
        if args.base_model == 'gpt2':
            config = AutoConfig.from_pretrained('gpt2')
            config.n_layer = args.n_layer
            config.n_head = args.n_head
            config.n_embd = args.n_embd
            config.attn_pdrop = args.attention_dropout
            if args.max_position_embeddings is not None:
                config.n_positions = args.max_position_embeddings
        elif args.base_model == 'pythia':
            config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.hidden_size = args.n_embd
            config.intermediate_size = config.hidden_size * 4
            config.attention_dropout = args.attention_dropout
            if args.max_position_embeddings is not None:
                config.max_position_embeddings = args.max_position_embeddings
        elif args.base_model == 'llama':
            config = AutoConfig.from_pretrained('unsloth/Llama-3.2-1B')
            config.num_hidden_layers = args.n_layer
            config.num_attention_heads = args.n_head
            config.num_key_value_heads = args.n_head
            config.hidden_size = args.n_embd
            config.head_dim = config.hidden_size // config.num_attention_heads
            config.intermediate_size = config.hidden_size * 4
            config.attention_dropout = args.attention_dropout
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
        config.vocab_size = args.vocab_size
        config.pad_token_id = None
        config.bos_token_id = None
        config.eos_token_id = None
        # create model
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype,
                                                 attn_implementation=args.attn_implementation)

    model.config.pad_token_id = None
    model.config.bos_token_id = None
    model.config.eos_token_id = None
    model.get_input_embeddings().padding_idx = None
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
    logger.info(f'train examples: {len(train_dataset)}; valid examples: {len(valid_dataset)}')

    # use collate_fn_numpy if no GPU is available, allows running with 'mps' device on Apple M chips
    data_collator = collate_fn if torch.cuda.is_available() else collate_fn_numpy

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir),

        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
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
        report_to='tensorboard',
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        # NumPy batches are converted to tensors later in CustomTrainer._prepare_input.
        include_num_input_tokens_seen=False,
        include_for_metrics=['inputs'],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = MQARTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   StopOnMetricValue(metric_name='all_queries_exact_match', value=args.stop_on_metric_value,
                                     higher_is_better=True),
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
    metrics = trainer.evaluate(valid_dataset)
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    if not args.do_eval_only:
        trainer.state.save_to_json(output_dir / 'trainer_state.json')
