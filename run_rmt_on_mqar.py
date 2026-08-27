import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from rmt import RMT2Segm, RMT2SegmConfig
from zoology_mqar_data import (
    ZOOLOGY_MQAR_SOURCE,
    build_mqar_datasets,
    load_saved_mqar_datasets,
)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


def collate_fn(batch):
    context_input_ids = torch.stack([item['input_ids'][:item['context_size']] for item in batch])
    query_input_ids = torch.stack([item['input_ids'][item['context_size']:] for item in batch])
    labels = torch.stack([item['labels'][item['context_size']:] for item in batch])
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
        'labels': labels,
    }


class MQARDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, context_size=None):
        self.source_dataset = source_dataset
        self.context_size = context_size or source_dataset.slices['context_size']

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index):
        item = self.source_dataset[index]
        return {
            'input_ids': item['input_ids'],
            'labels': item['labels'],
            'context_size': self.context_size,
        }


def tensor_batch_to_numpy(batch):
    if isinstance(batch, dict):
        return {key: tensor_batch_to_numpy(value) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.cpu().numpy().copy()
    return batch


def collate_fn_numpy(batch):
    return tensor_batch_to_numpy(collate_fn(batch))


def preprocess_logits_for_metrics(eval_pred, labels):
    logits, inner_loop_stats = eval_pred
    return (logits.argmax(dim=-1), inner_loop_stats)


def compute_metrics_fn(eval_pred):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    preds, inner_loop_stats = predictions
    mask = labels != -100
    if not np.all(mask.any(axis=1)):
        raise ValueError('Every MQAR example must contain at least one supervised query.')

    correct = preds[mask] == labels[mask]
    per_example_accuracy = [
        (pred[example_mask] == label[example_mask]).mean()
        for pred, label, example_mask in zip(preds, labels, mask)
    ]
    exact_match = [
        np.all(pred[example_mask] == label[example_mask])
        for pred, label, example_mask in zip(preds, labels, mask)
    ]
    for pred, label, inp in zip(preds[:5], labels[:5], inputs['query_input_ids'][:5]):
        query_mask = label != -100
        print('q:', np.asarray(inp)[query_mask].tolist())
        print('p:', np.asarray(pred)[query_mask].tolist())
        print('t:', np.asarray(label)[query_mask].tolist())
        print('-' * 50)

    metrics = {
        'token_accuracy': float(correct.mean()),
        'exact_match': float(correct.mean()),
        'all_queries_token_accuracy': float(np.mean(per_example_accuracy)),
        'all_queries_exact_match': float(np.mean(exact_match)),
    }
    for key in ('mem_norm_mean', 'mem_norm_max', 'mem_norm_min', 'delta_mem_norm_mean'):
        if key in inner_loop_stats:
            metrics[key] = float(np.asarray(inner_loop_stats[key]).mean())
    return metrics


class StopOnMetricValue(TrainerCallback):
    def __init__(self, metric_name: str, value: float):
        self.metric_name = metric_name
        self.value = value

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        metric_value = metrics.get(f'eval_{self.metric_name}')
        if metric_value is not None and metric_value >= self.value:
            control.should_training_stop = True
            logger.info(f'metric {self.metric_name}={metric_value:.4f} >= {self.value:.4f}, stopping training..')


class MQARRMT(RMT2Segm):
    """RMT with same-position MQAR target loss instead of causal text loss."""

    def forward(self, input_ids, labels=None, return_mem=False):
        if labels is None:
            return super().forward(input_ids, labels=None, return_mem=return_mem)

        output = super().forward(input_ids, labels=None, return_mem=return_mem)
        logits = output['predictions']
        target_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        output['inner_loop_stats']['target_loss'] = target_loss.detach()
        loss = target_loss
        if self.use_reconstruction_loss and 'rec_loss' in output['inner_loop_stats']:
            loss = loss + self.reconstruction_loss_weight * output['inner_loop_stats']['rec_loss']
        output['loss'] = loss
        return output


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    vocab_size: Optional[int] = field(default=8192)
    input_seq_len: Optional[int] = field(default=40)
    num_kv_pairs: Optional[int] = field(default=8)
    train_num_examples: Optional[int] = field(default=100_000)
    valid_num_examples: Optional[int] = field(default=3_000)
    power_a: Optional[float] = field(default=0.01)
    random_non_queries: Optional[bool] = field(default=False)
    mqar_noise_lvl: Optional[float] = field(default=0.0)
    mqar_data_path: Optional[str] = field(default=None)
    data_seed: Optional[int] = field(default=123)
    dense_queries: Optional[bool] = field(default=True)
    query_sampling: Optional[str] = field(default='uniform')
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50_000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.999)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    stop_on_metric_value: Optional[float] = field(default=1.0)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default='gpt2')
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_position_embeddings: Optional[int] = field(default=1024)
    attn_implementation: Optional[str] = field(default='eager')
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=1)
    n_ctrl_tokens: Optional[int] = field(default=0)
    use_mem_proj: Optional[bool] = field(default=False)
    mem_proj_mode: Optional[str] = field(default='none')
    use_reconstruction_loss: Optional[bool] = field(default=False)
    reconstruction_loss_weight: Optional[float] = field(default=1.0)
    use_write_head: Optional[bool] = field(default=False)
    use_mem_residual: Optional[bool] = field(default=False)


def build_base_config(args):
    if args.base_model == 'gpt2':
        config = AutoConfig.from_pretrained('gpt2')
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_embd = args.n_embd
        config.n_positions = args.max_position_embeddings
        config.n_ctx = args.max_position_embeddings
    elif args.base_model == 'pythia':
        config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
        config.num_hidden_layers = args.n_layer
        config.num_attention_heads = args.n_head
        config.hidden_size = args.n_embd
        config.intermediate_size = config.hidden_size * 4
        config.max_position_embeddings = args.max_position_embeddings
    elif args.base_model == 'llama':
        config = AutoConfig.from_pretrained('unsloth/Llama-3.2-1B')
        config.num_hidden_layers = args.n_layer
        config.num_attention_heads = args.n_head
        config.num_key_value_heads = args.n_head
        config.hidden_size = args.n_embd
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.intermediate_size = config.hidden_size * 4
        config.max_position_embeddings = args.max_position_embeddings
        config.rope_scaling = None
        config.rope_theta = 10000.0
    else:
        raise ValueError(f'Unsupported base model: {args.base_model}')

    config.torch_dtype = 'float32'
    config.vocab_size = args.vocab_size
    # Token 0 is a visible MQAR pair frame, not padding.
    # MQAR examples are fixed-length and token 0 is valid data. Do not use
    # vocab_size as a padding sentinel: LLaMA requires padding_idx to be a
    # valid embedding row during model construction.
    config.pad_token_id = None
    config.bos_token_id = None
    config.eos_token_id = None
    config.use_cache = False
    return config


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]
    accel = accelerate.Accelerator()
    transformers.utils.logging.set_verbosity(logging.INFO)

    if args.pretrained_model is not None and args.base_model is not None:
        raise ValueError('Only one of pretrained_model or base_model may be set.')

    if args.mqar_data_path is None:
        source_train, source_valid, train_data_seed, valid_data_seed = build_mqar_datasets(
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
        dataset_metadata = source_train.slices
    else:
        source_train, source_valid, dataset_metadata = load_saved_mqar_datasets(args.mqar_data_path)
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
        args.train_num_examples = len(source_train)
        args.valid_num_examples = len(source_valid)
        args.input_seq_len = dataset_metadata['input_seq_len']

    context_size = dataset_metadata['context_size']
    if hasattr(source_valid, 'slices') and source_valid.slices['context_size'] != context_size:
        raise ValueError('MQAR train and validation context sizes do not match.')
    if dataset_metadata['input_seq_len'] != source_valid[0]['input_ids'].shape[0]:
        raise ValueError('MQAR train and validation sequence lengths do not match.')
    train_dataset = MQARDataset(source_train, context_size=context_size)
    valid_dataset = MQARDataset(source_valid, context_size=context_size)

    if args.pretrained_model is None:
        base_config = build_base_config(args)
    else:
        base_config = None
    rmt_config = RMT2SegmConfig(
        pretrained_model=args.pretrained_model,
        base_config=base_config,
        n_mem_tokens=args.n_mem_tokens,
        K=args.K,
        n_ctrl_tokens=args.n_ctrl_tokens,
        use_mem_proj=args.use_mem_proj,
        mem_proj_mode=args.mem_proj_mode,
        use_reconstruction_loss=args.use_reconstruction_loss,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        use_write_head=args.use_write_head,
        use_mem_residual=args.use_mem_residual,
        attn_implementation=args.attn_implementation,
    )
    model = MQARRMT(rmt_config)
    model.model.config.pad_token_id = None
    model.model.get_input_embeddings().padding_idx = None

    if args.init_checkpoint is not None:
        missing, unexpected = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if missing:
            logger.info(f'{missing} were not loaded from checkpoint.')
        if unexpected:
            logger.info(f'{unexpected} were unexpected in the checkpoint.')

    output_dir = Path(args.exp_path)
    if accel.is_main_process:
        config = {
            'cli_args': dict(vars(args)),
            'task_source': ZOOLOGY_MQAR_SOURCE,
            'task': 'Zoology MQAR with RMT write/read split',
            'query_layout': (
                'dense context with framed KV pairs and interleaved random noise'
                if args.dense_queries and dataset_metadata.get('noise_tokens', 0) > 0 else
                'dense context and contiguous query keys; no fillers'
                if args.dense_queries else
                'upstream query placement with framed KV pairs and interleaved random noise'
                if dataset_metadata.get('noise_tokens', 0) > 0 else
                'upstream query placement'
            ),
            'query_sampling': dataset_metadata.get('query_sampling'),
            'data_generator': dataset_metadata.get('generator', 'saved' if args.mqar_data_path else 'upstream'),
            'context_size': context_size,
            'query_size': dataset_metadata['input_seq_len'] - context_size,
            'input_seq_len': dataset_metadata['input_seq_len'],
            'context_noise': {
                'level': args.mqar_noise_lvl,
                'tokens': dataset_metadata.get('noise_tokens', 0),
                'pair_open_token': dataset_metadata.get('pair_open_token', 0),
                'pair_close_token': dataset_metadata.get('pair_close_token', 1),
            },
            'train_data_seed': train_data_seed,
            'valid_data_seed': valid_data_seed,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(config, open(output_dir / 'config.json', 'w'), indent=4)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        assert args.total_batch_size == args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps

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
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
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
        include_num_input_tokens_seen=False,
        include_for_metrics=['inputs'],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn if torch.cuda.is_available() else collate_fn_numpy,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            StopOnMetricValue('all_queries_exact_match', args.stop_on_metric_value),
        ],
    )
    trainer.train()
    metrics = trainer.evaluate(valid_dataset)
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')
