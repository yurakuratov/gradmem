import json
import logging
import os
from pathlib import Path

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field

import accelerate
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)

from grad_memgpt import GradMemGPT, GradMemGPTConfig
from resume_utils import restore_resume_args
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


def initialize_run_seed(seed):
    transformers.set_seed(seed)


LOSS_COMPONENT_KEYS = [
    "outer_loss",
    "target_loss",
    "energy_rank_loss",
    "energy_rank_deranged_loss",
    "energy_rank_interpolated_loss",
    "energy_rank_random_loss",
    "energy_traj_loss",
    "energy_anchor_loss",
    "energy_aux_loss",
]

ENERGY_LANDSCAPE_STAT_KEYS = [
    "energy_positive_mean",
    "energy_negative_deranged_mean",
    "energy_negative_interpolated_mean",
    "energy_negative_random_mean",
    "energy_margin_deranged_mean",
    "energy_margin_interpolated_mean",
    "energy_margin_random_mean",
    "energy_margin_violation_deranged",
    "energy_margin_violation_interpolated",
    "energy_margin_violation_random",
]

TRAIN_COMPONENT_KEYS = LOSS_COMPONENT_KEYS + ENERGY_LANDSCAPE_STAT_KEYS


class GradMemMQARDataset(torch.utils.data.Dataset):
    """Expose an MQAR example through GradMem's write/read split."""

    def __init__(self, dataset, context_size):
        self.source_dataset = dataset
        self.context_size = context_size
        self._uses_tensors = hasattr(dataset, 'inputs')
        if self._uses_tensors:
            if context_size <= 0 or context_size >= dataset.inputs.shape[1]:
                raise ValueError(
                    f'Invalid context_size={context_size} for input length {dataset.inputs.shape[1]}.'
                )
            if torch.any(dataset.labels[:, :context_size] != -100):
                raise ValueError('MQAR labels unexpectedly supervise the KV context.')
            self.context_input_ids = dataset.inputs[:, :context_size]
            self.query_input_ids = dataset.inputs[:, context_size:]
            self.labels = dataset.labels[:, context_size:]

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index):
        if not self._uses_tensors:
            item = self.source_dataset[index]
            return {
                'context_input_ids': item['input_ids'][:self.context_size],
                'query_input_ids': item['input_ids'][self.context_size:],
                'labels': item['labels'][self.context_size:],
            }
        return {
            'context_input_ids': self.context_input_ids[index],
            'query_input_ids': self.query_input_ids[index],
            'labels': self.labels[index],
        }


def collate_fn(batch):
    context_input_ids = torch.stack([item['context_input_ids'] for item in batch])
    query_input_ids = torch.stack([item['query_input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids': query_input_ids,
        },
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


class MQARGradMemGPT(GradMemGPT):
    """GradMemGPT with an exact MQAR vocabulary and query-position loss."""

    def __init__(self, config):
        super().__init__(config)
        expected_vocab_size = config.mqar_vocab_size
        actual_vocab_size = self.model.config.vocab_size
        if actual_vocab_size != expected_vocab_size:
            raise ValueError(
                f'Base model vocabulary size {actual_vocab_size} does not match '
                f'MQAR vocab_size {expected_vocab_size}.'
            )

        # Noisy MQAR uses token 0 as the visible left frame for each KV pair.
        embeddings = self.model.get_input_embeddings()
        if embeddings.padding_idx == 0:
            embeddings.padding_idx = None
            if not config.mqar_dense_queries:
                initializer_range = getattr(self.model.config, 'initializer_range', 0.02)
                with torch.no_grad():
                    embeddings.weight[0].normal_(mean=0.0, std=initializer_range)
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = None
        self.model.config.eos_token_id = None


def preprocess_logits_for_metrics(eval_pred, labels):
    logits, inner_loop_stats = eval_pred
    # saves gpu RAM, as HF Trainer accumulates all eval logits on GPU
    return (logits.argmax(dim=-1), inner_loop_stats)


class ConsoleEvalMetricsCallback(TrainerCallback):
    """Print every evaluation as a durable rank-zero console line."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero and metrics:
            print(
                f'eval step {state.global_step}: '
                f'{json.dumps(metrics, sort_keys=True, default=float)}',
                flush=True,
            )


def compute_metrics_fn(eval_pred):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    preds, inner_loop_stats = predictions
    pred_len = preds.shape[1]
    label_len = labels.shape[1]
    if pred_len == label_len + 1:
        # Prefix memory returns one pre-query logit before logits at query positions.
        preds = preds[:, 1:]
    elif pred_len != label_len:
        raise ValueError(
            f'Unexpected prediction/label lengths: pred_len={pred_len}, label_len={label_len}'
        )

    mask = labels != -100
    if not np.all(mask.any(axis=1)):
        raise ValueError('Every MQAR example must contain at least one supervised query.')

    correct = preds[mask] == labels[mask]
    token_accuracy = float(correct.mean())
    per_example_token_accuracy = [
        (pred[example_mask] == label[example_mask]).mean()
        for pred, label, example_mask in zip(preds, labels, mask)
    ]
    all_queries_exact_match = [
        np.all(pred[example_mask] == label[example_mask])
        for pred, label, example_mask in zip(preds, labels, mask)
    ]

    for pred, label, inp_c, inp_q, example_mask in zip(
        preds[:5], labels[:5], inputs['context_input_ids'][:5],
        inputs['query_input_ids'][:5], mask[:5]
    ):
        print('i:', np.concatenate([inp_c, inp_q]).tolist())
        print('q:', np.asarray(inp_q[example_mask]).tolist())
        print('p:', np.asarray(pred[example_mask]).tolist())
        print('t:', np.asarray(label[example_mask]).tolist())
        print('-' * 50)

    metrics = {
        # Every MQAR answer is one token, so per-query EM equals token accuracy.
        'token_accuracy': token_accuracy,
        'exact_match': token_accuracy,
        'all_queries_token_accuracy': float(np.mean(per_example_token_accuracy)),
        'all_queries_exact_match': float(np.mean(all_queries_exact_match)),
    }
    if 'inner_grad_norm_mean' in inner_loop_stats:
        metrics['inner_grad_norm'] = float(
            np.asarray(inner_loop_stats['inner_grad_norm_mean']).mean()
        )

    mean_stats = [
        'inner_loss',
        'inner_grad_norm',
        'inner_grad_norm_mean',
        'mem_norm_mean',
        'delta_mem_norm_mean',
        'mem_attn_read',
        'inner_loss_after_write',
        'inner_loss_initial',
        'inner_loss_write_delta',
        'inner_reconstruction_loss',
        'inner_energy_loss',
        'inner_reconstruction_loss_after_write',
        'inner_energy_loss_after_write',
        'write_reconstruction_weight',
        'write_energy_weight',
    ] + TRAIN_COMPONENT_KEYS
    max_stats = ['inner_grad_norm_max', 'mem_norm_max', 'delta_mem_norm_max']
    min_stats = ['inner_grad_norm_min', 'mem_norm_min', 'delta_mem_norm_min']
    for key in mean_stats:
        if key in inner_loop_stats:
            metrics[key] = float(np.asarray(inner_loop_stats[key]).mean())
    for key in max_stats:
        if key in inner_loop_stats:
            metrics[key] = float(np.asarray(inner_loop_stats[key]).max())
    for key in min_stats:
        if key in inner_loop_stats:
            metrics[key] = float(np.asarray(inner_loop_stats[key]).min())
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_component_sums = {}
        self._train_component_count = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        batch_size = labels.size(0) if isinstance(labels, torch.Tensor) else 1
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if model.training and isinstance(outputs, dict):
            stats = outputs.get("inner_loop_stats", {})
            count = loss.detach().new_tensor(float(batch_size))
            if self._train_component_count is None:
                self._train_component_count = count
            else:
                self._train_component_count = self._train_component_count + count
            for key in TRAIN_COMPONENT_KEYS:
                value = stats.get(key)
                if value is None:
                    continue
                weighted_value = value.detach().float().mean() * count
                if key in self._train_component_sums:
                    self._train_component_sums[key] = self._train_component_sums[key] + weighted_value
                else:
                    self._train_component_sums[key] = weighted_value

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        num_training_steps = int(num_training_steps / 0.9)  # to make final lr not zero, for linear it is lr/10.
        return super().create_scheduler(num_training_steps, optimizer)

    def _prepare_input(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return super()._prepare_input(data)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if "loss" in logs and self._train_component_count is not None:
            total_count = self._nested_gather(self._train_component_count).sum().item()
            if total_count > 0:
                for key, value_sum in self._train_component_sums.items():
                    total_value = self._nested_gather(value_sum).sum().item()
                    logs[key] = total_value / total_count
            self._train_component_sums = {}
            self._train_component_count = None

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
    query_sampling: Optional[str] = field(
        default='uniform',
        metadata={'help': 'Dense query sampling: uniform, power_law, or zoology.'},
    )
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    save_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'Save a checkpoint every N optimizer steps; defaults to eval_steps.'},
    )
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.999)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    stop_on_metric_value: Optional[float] = field(default=1.0)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_base_checkpoint: Optional[str] = field(default=None, metadata={'help': 'checkpoint to initialize base model'})
    init_checkpoint: Optional[str] = field(default=None, metadata={'help': 'checkpoint to initialize gradmem model'})
    resume_from_checkpoint: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    max_position_embeddings: Optional[int] = field(default=1024)
    # GradMemGPT parameters
    memory_backend: Optional[str] = field(default="prefix")
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
    lora_mem_placement: Optional[str] = field(default="between_layers")
    lora_mem_r: Optional[int] = field(default=8)
    lora_mem_alpha: Optional[int] = field(default=16)
    lora_mem_dropout: Optional[float] = field(default=0.0)
    lora_mem_layers: Optional[str] = field(default="all")
    lora_mem_target_modules: Optional[str] = field(default=None)
    kv_mem_layers: Optional[str] = field(default="all")
    freeze_backbone: Optional[bool] = field(default=False)
    use_gradient_checkpointing: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default="eager")
    write_objective: Optional[str] = field(default="reconstruction")
    energy_head_hidden_dim: Optional[int] = field(default=None)
    use_layerwise_energy: Optional[bool] = field(default=False)
    write_reconstruction_weight: Optional[float] = field(default=1.0)
    write_energy_weight: Optional[float] = field(default=1.0)
    energy_rank_weight: Optional[float] = field(default=0.0)
    energy_traj_weight: Optional[float] = field(default=0.0)
    energy_margin: Optional[float] = field(default=0.1)
    energy_traj_margin: Optional[float] = field(default=0.0)
    energy_rank_temperature: Optional[float] = field(default=1.0)
    energy_mix_alpha: Optional[float] = field(default=0.75)
    energy_anchor_weight: Optional[float] = field(default=0.0)
    lipschitz_weight: Optional[float] = field(default=0.0)
    lipschitz_constraint: Optional[float] = field(default=1.0)
    energy_memory_search_weight: Optional[float] = field(default=0.0)
    energy_memory_search_num_samples: Optional[int] = field(default=4)
    energy_memory_search_radius_scale: Optional[float] = field(default=0.25)
    energy_memory_search_use_gain_weighting: Optional[bool] = field(default=False)
    energy_memory_search_gain_ema_decay: Optional[float] = field(default=0.99)
    energy_memory_search_min_relative_target_gain: Optional[float] = field(default=0.0)
    energy_memory_search_use_best_for_next_step: Optional[bool] = field(default=False)
    read_focal_gamma: Optional[float] = field(default=0.0)
    add_inner_loss_to_outer: Optional[bool] = field(default=False)
    inner_loss_weight: Optional[float] = field(default=None)
    memory_alignment_weight: Optional[float] = field(default=0.0)
    step_alignment_weight: Optional[float] = field(default=0.0)
    align_last_step: Optional[bool] = field(default=False)
    grad_align_norm: Optional[str] = field(default='none')
    intermediate_read_weight: Optional[float] = field(default=0.0)
    memory_noise_sigma: Optional[float] = field(default=0.0)
    orthogonal_loss_weight: Optional[float] = field(default=0.0)
    ivan_loss_weight: Optional[float] = field(default=0.0)


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    restore_resume_args(args, logger)
    if args.init_checkpoint is not None and args.resume_from_checkpoint is not None:
        raise ValueError('--init_checkpoint and --resume_from_checkpoint are mutually exclusive')

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"
    if args.pretrained_model is not None and args.init_base_checkpoint is not None:
        raise ValueError('pretrained_model and init_base_checkpoint are mutually exclusive.')

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
    if source_valid is not None and hasattr(source_valid, 'slices') and source_valid.slices['context_size'] != context_size:
        raise ValueError('MQAR train and validation context sizes do not match.')
    if dataset_metadata.get('input_seq_len') != source_valid[0]['input_ids'].shape[0]:
        raise ValueError('MQAR train and validation sequence lengths do not match.')
    train_dataset = GradMemMQARDataset(source_train, context_size=context_size)
    valid_dataset = GradMemMQARDataset(source_valid, context_size=context_size)
    output_dir = Path(args.exp_path)

    if accel.is_main_process and args.resume_from_checkpoint is None:
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
            'task': 'Zoology MQAR with GradMem write/read split',
            'query_layout': query_layout,
            'query_sampling': dataset_metadata.get('query_sampling'),
            'data_generator': dataset_metadata.get('generator', 'saved' if args.mqar_data_path else 'upstream'),
            'query_distribution': dataset_metadata.get(
                'query_distribution', 'power_law_placement'
            ),
            'effective_power_a': dataset_metadata.get('effective_power_a', args.power_a),
            'loss_alignment': 'query-position read loss; generated labels are unchanged',
            'context_size': context_size,
            'query_size': dataset_metadata['input_seq_len'] - context_size,
            'input_seq_len': dataset_metadata['input_seq_len'],
            'context_noise': {
                'level': args.mqar_noise_lvl,
                'tokens': dataset_metadata.get('noise_tokens', 0),
                'pair_open_token': dataset_metadata.get('pair_open_token'),
                'pair_close_token': dataset_metadata.get('pair_close_token'),
            },
            'train_data_seed': train_data_seed,
            'valid_data_seed': valid_data_seed,
            'checkpoint_source': (
                {'type': 'pretrained_model', 'path': args.pretrained_model}
                if args.pretrained_model is not None else
                {'type': 'init_base_checkpoint', 'path': args.init_base_checkpoint}
                if args.init_base_checkpoint is not None else
                {'type': 'scratch', 'base_model': args.base_model}
            ),
            'init_gradmem_checkpoint': args.init_checkpoint,
            'metrics': [
                'token_accuracy',
                'exact_match',
                'all_queries_token_accuracy',
                'all_queries_exact_match',
            ],
        }
        logger.info(f'saving experiment configuration to {args.exp_path}')
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(config, open(output_dir / 'config.json', 'w'), indent=4)

    if args.pretrained_model is None:
        # create base model config
        if args.init_base_checkpoint is not None:
            checkpoint_dir = Path(args.init_base_checkpoint).resolve().parent
            checkpoint_config = checkpoint_dir / 'config.json'
            if not checkpoint_config.is_file():
                raise FileNotFoundError(f'Base checkpoint config does not exist: {checkpoint_config}')
            config = AutoConfig.from_pretrained(checkpoint_dir)
            if config.vocab_size != args.vocab_size:
                raise ValueError(
                    f'Base checkpoint vocabulary size {config.vocab_size} does not match '
                    f'MQAR vocab_size {args.vocab_size}.'
                )
            logger.info(f'Building the base model from checkpoint config: {checkpoint_config}')
        elif args.base_model == 'gpt2':
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
            config.rope_scaling = None
            config.rope_theta = 10000.0
            config.max_position_embeddings = args.max_position_embeddings
        else:
            raise ValueError(f'Unsupported base model: {args.base_model}')

        config.torch_dtype = "float32"  # weights in float32, at training precision is controlled by accelerate
        config.vocab_size = args.vocab_size
        config.pad_token_id = 0
        config.bos_token_id = None
        config.eos_token_id = None
        config.use_cache = False
    else:
        config = None
        pretrained_config = AutoConfig.from_pretrained(args.pretrained_model)
        if pretrained_config.vocab_size != args.vocab_size:
            raise ValueError(
                f'Pretrained model vocabulary size {pretrained_config.vocab_size} does not match '
                f'MQAR vocab_size {args.vocab_size}.'
            )

    initialize_run_seed(args.seed)
    gradmem_config = GradMemGPTConfig(pretrained_model=args.pretrained_model, base_config=config,
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
                                       write_objective=args.write_objective,
                                       energy_head_hidden_dim=args.energy_head_hidden_dim,
                                       use_layerwise_energy=args.use_layerwise_energy,
                                       write_reconstruction_weight=args.write_reconstruction_weight,
                                      write_energy_weight=args.write_energy_weight,
                                      energy_rank_weight=args.energy_rank_weight,
                                      energy_traj_weight=args.energy_traj_weight,
                                      energy_margin=args.energy_margin,
                                      energy_traj_margin=args.energy_traj_margin,
                                      energy_rank_temperature=args.energy_rank_temperature,
                                       energy_mix_alpha=args.energy_mix_alpha,
                                       energy_anchor_weight=args.energy_anchor_weight,
                                       lipschitz_weight=args.lipschitz_weight,
                                       lipschitz_constraint=args.lipschitz_constraint,
                                       energy_memory_search_weight=args.energy_memory_search_weight,
                                       energy_memory_search_num_samples=args.energy_memory_search_num_samples,
                                       energy_memory_search_radius_scale=args.energy_memory_search_radius_scale,
                                       energy_memory_search_use_gain_weighting=(
                                           args.energy_memory_search_use_gain_weighting
                                       ),
                                       energy_memory_search_gain_ema_decay=(
                                           args.energy_memory_search_gain_ema_decay
                                       ),
                                       energy_memory_search_min_relative_target_gain=(
                                           args.energy_memory_search_min_relative_target_gain
                                       ),
                                       energy_memory_search_use_best_for_next_step=(
                                           args.energy_memory_search_use_best_for_next_step
                                       ),
                                       read_focal_gamma=args.read_focal_gamma,
                                       add_inner_loss_to_outer=args.add_inner_loss_to_outer,
                                       inner_loss_weight=args.inner_loss_weight,
                                       memory_alignment_weight=args.memory_alignment_weight,
                                       step_alignment_weight=args.step_alignment_weight,
                                       align_last_step=args.align_last_step,
                                       grad_align_norm=args.grad_align_norm,
                                       intermediate_read_weight=args.intermediate_read_weight,
                                       memory_noise_sigma=args.memory_noise_sigma,
                                       orthogonal_loss_weight=args.orthogonal_loss_weight,
                                       ivan_loss_weight=args.ivan_loss_weight,
                                       read_loss_alignment='query_position',
                                      mqar_vocab_size=args.vocab_size,
                                      mqar_dense_queries=args.dense_queries)

    # Create gradmemgpt model
    model = MQARGradMemGPT(gradmem_config)

    model_to_init_from_ckpt = None
    state_dict = None
    if args.init_checkpoint is not None:
        model_to_init_from_ckpt = model
        state_dict = load_file(args.init_checkpoint)
        if args.memory_backend == 'prefix' and 'mem' in state_dict and getattr(model, 'mem', None) is not None:
            ckpt_mem = state_dict['mem']
            model_mem = model.mem
            # if n_mem_tokens is different, slice the checkpoint mem to the model mem shape
            if ckpt_mem.shape[0] != model_mem.shape[0]:
                if ckpt_mem.shape[0] > model_mem.shape[0]:
                    logger.info(
                        f'Slicing checkpoint mem from {tuple(ckpt_mem.shape)} to {tuple(model_mem.shape)}.'
                    )
                    state_dict['mem'] = ckpt_mem[:model_mem.shape[0]]
                else:
                    raise ValueError(
                        f'Checkpoint has fewer memory tokens than model expects: '
                        f'ckpt mem shape={tuple(ckpt_mem.shape)}, model mem shape={tuple(model_mem.shape)}.'
                    )
    elif args.init_base_checkpoint is not None:
        model_to_init_from_ckpt = model.model
        state_dict = load_file(args.init_base_checkpoint)
    if model_to_init_from_ckpt is not None:
        missing_k, unexpected_k = model_to_init_from_ckpt.load_state_dict(state_dict, strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    if accel.mixed_precision == 'bf16':
        model.to(torch.bfloat16)

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')
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
        gradient_checkpointing=args.use_gradient_checkpointing,

        eval_strategy='steps',
        save_strategy='steps',
        save_steps=args.save_steps if args.save_steps is not None else args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,  # input_ids is a dict, so HF Trainer cant get number of tokens
        include_for_metrics=['inputs'],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[ConsoleEvalMetricsCallback(),
                   EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   StopOnMetricValue(metric_name='all_queries_exact_match', value=args.stop_on_metric_value,
                                     higher_is_better=True),
                   ],
    )
    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(valid_dataset)
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')
