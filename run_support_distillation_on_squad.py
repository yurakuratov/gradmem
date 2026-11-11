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
from safetensors.torch import load_file
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser
)

from support_distillation import SupportDistillationGradMemGPT, SupportDistillationConfig
from support_distillation import build_episodes, collate_episodes


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer, max_context_length=None):
    context = [item['context'].strip() for item in batch]
    query = [item['query'] + item['target'] for item in batch]

    context_input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True,
                                  padding=True, pad_to_multiple_of=8,
                                  max_length=max_context_length,
                                  truncation=max_context_length is not None).input_ids
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
    if 'mem_attn_read' in inner_loop_stats:
        metrics['mem_attn_read'] = float(inner_loop_stats['mem_attn_read'].mean())
    for key in (
        'inner_support_kl_loss', 'inner_support_kl_loss_after_write',
        'inner_reconstruction_loss', 'inner_energy_loss', 'inner_loss_after_write',
        'inner_reconstruction_loss_after_write', 'inner_energy_loss_after_write',
        'outer_loss', 'target_loss', 'energy_rank_loss', 'energy_traj_loss',
        'energy_anchor_loss', 'energy_aux_loss', 'step_alignment_loss',
        'intermediate_read_loss', 'weighted_target_loss', 'distillation_loss',
        'distill_logits_query_loss', 'distill_logits_target_loss',
        'distill_hidden_query_loss', 'distill_hidden_target_loss',
    ):
        if key in inner_loop_stats:
            metrics[key] = float(np.asarray(inner_loop_stats[key]).mean())
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
    data_path: str = field(
        default='./data/N2-K4V4-S4(32-64)_1M',
    )
    dataset_name: str = field(default='squad')
    max_context_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Truncate contexts to this many tokens, including special tokens. '
                          'Unset means no limit; queries and targets are not truncated.'},
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
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_base_checkpoint: Optional[str] = field(default=None, metadata={'help': 'checkpoint to initialize base model'})
    init_checkpoint: Optional[str] = field(default=None, metadata={'help': 'checkpoint to initialize gradmemgpt model'})
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    # GradMemGPT parameters
    memory_backend: Optional[str] = field(default="prefix")
    n_mem_tokens: Optional[int] = field(default=8)
    K: Optional[int] = field(default=3)
    K_sampling: Optional[bool] = field(default=False)
    K_min: Optional[int] = field(default=None)
    K_max: Optional[int] = field(default=None)
    ilr_min: Optional[float] = field(default=None)
    ilr_max: Optional[float] = field(default=None)
    ilr_sample_unique: Optional[bool] = field(default=False)
    inner_lr: Optional[float] = field(default=0.01)
    use_adam: Optional[bool] = field(default=False)
    grad_mode: Optional[str] = field(default="second")
    n_ctrl_tokens: Optional[int] = field(default=0)
    inner_clip_value: Optional[float] = field(default=None)
    inner_clip_norm: Optional[float] = field(default=None)
    memory_noise_sigma: Optional[float] = field(default=0.0)
    use_mem_proj: Optional[bool] = field(default=False)
    mem_proj_mode: Optional[str] = field(default="none")
    use_write_head: Optional[bool] = field(default=False)
    lora_mem_placement: Optional[str] = field(default="between_layers")
    lora_mem_r: Optional[int] = field(default=8)
    lora_mem_alpha: Optional[int] = field(default=16)
    lora_mem_dropout: Optional[float] = field(default=0.0)
    lora_mem_layers: Optional[str] = field(default="all")
    lora_mem_target_modules: Optional[str] = field(default=None)
    kv_mem_layers: Optional[str] = field(default="all")
    use_gradient_checkpointing: Optional[bool] = field(default=False)
    attn_implementation: Optional[str] = field(default="eager", metadata={
        "help": "Attention backend: eager, hvp_manual, hvp_semi_manual, flashbob (alias bob), or another registered backend."
    })
    read_focal_gamma: Optional[float] = field(default=0.0)
    last_K_second_order: Optional[int] = field(default=None)
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
    add_inner_loss_to_outer: Optional[bool] = field(default=False)
    inner_loss_weight: Optional[float] = field(default=None)
    memory_alignment_weight: Optional[float] = field(default=0.0)
    step_alignment_weight: Optional[float] = field(default=0.0)
    alignment_sampling: Optional[bool] = field(default=False)
    align_last_step: Optional[bool] = field(default=False)
    grad_align_norm: Optional[str] = field(default="none")
    intermediate_read_weight: Optional[float] = field(default=0.0)
    orthogonal_loss_weight: Optional[float] = field(default=0.0)
    ivan_loss_weight: Optional[float] = field(default=0.0)
    freeze_backbone: Optional[bool] = field(default=False)
    use_write_lora: Optional[bool] = field(default=False)
    write_lora_r: Optional[int] = field(default=8)
    write_lora_alpha: Optional[int] = field(default=16)
    write_lora_dropout: Optional[float] = field(default=0.0)
    write_lora_target_modules: Optional[str] = field(default=None)
    separate_writer: Optional[bool] = field(default=False)
    writer_pretrained_model: Optional[str] = field(default=None)
    writer_base_model: Optional[str] = field(default=None)
    writer_n_layer: Optional[int] = field(default=None)
    writer_n_head: Optional[int] = field(default=None)
    writer_n_embd: Optional[int] = field(default=None)
    read_ce_weight: float = field(default=1.0)
    distill_logits_query_weight: float = field(default=0.0)
    distill_logits_target_weight: float = field(default=0.0)
    distill_hidden_query_weight: float = field(default=0.0)
    distill_hidden_target_weight: float = field(default=0.0)
    n_support_questions: int = field(default=2)
    inner_distill_query_weight: float = field(default=0.0)
    inner_distill_target_weight: float = field(default=1.0)
    inner_reconstruction_weight: float = field(default=0.0, metadata={
        'help': 'Context reconstruction CE weight in inner WRITE, batched with support KL.'})
    dataloader_num_workers: int = field(default=0)
    resume_from_checkpoint: Optional[str] = field(default=None)


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

    assert not (args.pretrained_model is not None and args.base_model is not None), "only one of these args must be set"

    if accel.is_main_process:
        config = {
            'cli_args': dict(vars(args)),
        }
        logger.info(f'saving experiment configuration to {args.exp_path}')
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    if args.pretrained_model is None:
        # create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
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

    writer_config = None
    if args.separate_writer:
        if args.writer_pretrained_model:
            writer_check = AutoConfig.from_pretrained(args.writer_pretrained_model)
            if writer_check.vocab_size != len(tokenizer):
                raise ValueError('Writer and SQuAD tokenizer vocabulary sizes must match')
        else:
            writer_type = args.writer_base_model or args.base_model
            templates = {'gpt2': 'gpt2', 'pythia': 'EleutherAI/pythia-160m',
                         'llama': 'meta-llama/Llama-3.2-1B'}
            if writer_type not in templates:
                raise ValueError('Set writer_base_model to gpt2, pythia, or llama')
            writer_config = AutoConfig.from_pretrained(templates[writer_type])
            layers = args.writer_n_layer or args.n_layer
            heads = args.writer_n_head or args.n_head
            hidden = args.writer_n_embd or args.n_embd
            if writer_type == 'gpt2':
                writer_config.n_layer, writer_config.n_head, writer_config.n_embd = layers, heads, hidden
            else:
                writer_config.num_hidden_layers = layers
                writer_config.num_attention_heads = heads
                writer_config.hidden_size = hidden
                writer_config.intermediate_size = hidden * 4
                if writer_type == 'llama':
                    writer_config.num_key_value_heads = heads
                    writer_config.head_dim = hidden // heads
            writer_config.vocab_size = len(tokenizer)
            writer_config.pad_token_id = tokenizer.pad_token_id
            writer_config.bos_token_id = tokenizer.bos_token_id
            writer_config.eos_token_id = tokenizer.eos_token_id
            writer_config.use_cache = False

    transformers.set_seed(args.seed)
    gradmem_config = SupportDistillationConfig(pretrained_model=args.pretrained_model, base_config=config,
                                      n_support_questions=args.n_support_questions,
                                      inner_distill_query_weight=args.inner_distill_query_weight,
                                      inner_distill_target_weight=args.inner_distill_target_weight,
                                      inner_reconstruction_weight=args.inner_reconstruction_weight,
                                      read_ce_weight=args.read_ce_weight,
                                      distill_logits_query_weight=args.distill_logits_query_weight,
                                      distill_logits_target_weight=args.distill_logits_target_weight,
                                      distill_hidden_query_weight=args.distill_hidden_query_weight,
                                      distill_hidden_target_weight=args.distill_hidden_target_weight,
                                      separate_writer=args.separate_writer,
                                      writer_pretrained_model=args.writer_pretrained_model,
                                      writer_base_config=writer_config,
                                      memory_backend=args.memory_backend,
                                      n_mem_tokens=args.n_mem_tokens, K=args.K,
                                      K_sampling=args.K_sampling, K_min=args.K_min, K_max=args.K_max,
                                      ilr_min=args.ilr_min, ilr_max=args.ilr_max,
                                      ilr_sample_unique=args.ilr_sample_unique,
                                      lr=args.inner_lr, use_adam=args.use_adam, grad_mode=args.grad_mode,
                                      n_ctrl_tokens=args.n_ctrl_tokens,
                                      inner_clip_value=args.inner_clip_value, inner_clip_norm=args.inner_clip_norm,
                                      memory_noise_sigma=args.memory_noise_sigma,
                                      use_mem_proj=args.use_mem_proj, mem_proj_mode=args.mem_proj_mode,
                                      use_write_head=args.use_write_head,
                                      lora_mem_placement=args.lora_mem_placement,
                                      lora_mem_r=args.lora_mem_r,
                                      lora_mem_alpha=args.lora_mem_alpha,
                                      lora_mem_dropout=args.lora_mem_dropout,
                                      lora_mem_layers=args.lora_mem_layers,
                                      lora_mem_target_modules=args.lora_mem_target_modules,
                                       kv_mem_layers=args.kv_mem_layers,
                                       use_gradient_checkpointing=args.use_gradient_checkpointing,
                                       attn_implementation=args.attn_implementation,
                                        read_focal_gamma=args.read_focal_gamma,
                                        last_K_second_order=args.last_K_second_order,
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
                                        energy_memory_search_use_gain_weighting=args.energy_memory_search_use_gain_weighting,
                                        energy_memory_search_gain_ema_decay=args.energy_memory_search_gain_ema_decay,
                                        energy_memory_search_min_relative_target_gain=args.energy_memory_search_min_relative_target_gain,
                                        energy_memory_search_use_best_for_next_step=args.energy_memory_search_use_best_for_next_step,
                                        add_inner_loss_to_outer=args.add_inner_loss_to_outer,
                                        inner_loss_weight=args.inner_loss_weight,
                                        memory_alignment_weight=args.memory_alignment_weight,
                                        step_alignment_weight=args.step_alignment_weight,
                                        alignment_sampling=args.alignment_sampling,
                                        align_last_step=args.align_last_step,
                                        grad_align_norm=args.grad_align_norm,
                                        intermediate_read_weight=args.intermediate_read_weight,
                                        orthogonal_loss_weight=args.orthogonal_loss_weight,
                                        ivan_loss_weight=args.ivan_loss_weight,
                                        freeze_backbone=args.freeze_backbone,
                                        use_write_lora=args.use_write_lora,
                                        write_lora_r=args.write_lora_r,
                                        write_lora_alpha=args.write_lora_alpha,
                                        write_lora_dropout=args.write_lora_dropout,
                                        write_lora_target_modules=args.write_lora_target_modules)

    # Create gradmemgpt model
    model = SupportDistillationGradMemGPT(gradmem_config)

    model_to_init_from_ckpt = None
    if args.init_checkpoint is not None:
        model_to_init_from_ckpt = model
        init_ckpt_path = args.init_checkpoint
    elif args.init_base_checkpoint is not None:
        model_to_init_from_ckpt = model.model
        init_ckpt_path = args.init_base_checkpoint
    if model_to_init_from_ckpt is not None:
        missing_k, unexpected_k = model_to_init_from_ckpt.load_state_dict(load_file(init_ckpt_path), strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model_to_init_from_ckpt is not expecting them!')

    if accel.mixed_precision == 'bf16':
        model.to(torch.bfloat16)

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')

    if args.dataset_name != 'squad':
        raise ValueError('This runner requires SQuAD context/question/answers records')
    raw = (datasets.load_from_disk(args.data_path) if os.path.isdir(args.data_path)
           else datasets.load_dataset(args.dataset_name))
    dataset = {
        'train': build_episodes(raw['train'], args.n_support_questions, args.seed),
        'valid': build_episodes(raw['validation'] if 'validation' in raw else raw['valid'],
                                args.n_support_questions, args.seed + 1),
    }
    logger.info(f"episodes: train={len(dataset['train'])}, valid={len(dataset['valid'])}")

    def data_collator(batch):
        return collate_episodes(batch, tokenizer, args.max_context_length)

    # Target sequence looks like: "XXXX!|"
    # Let's not count ! and | in the accuracy calculation
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in []]

    # Define custom compute metrics function with ignored tokens
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
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
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
        include_num_input_tokens_seen=False,  # input_ids is a dict, so HF Trainer cant get number of tokens
        include_for_metrics=['inputs'],
        save_total_limit=1,
        dataloader_num_workers=args.dataloader_num_workers,
        label_names=['labels'],
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
    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')
