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
import yaml

from grad_memgpt import GradMemGPT, GradMemGPTConfig


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer, max_context_length=None):
    context = [item['context'] for item in batch]
    query = [item['query'] + item['target'] for item in batch]

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
    if 'step_delta_mem_norm_mean' in inner_loop_stats:
        metrics['step_delta_mem_norm_mean'] = float(inner_loop_stats['step_delta_mem_norm_mean'].mean())
        metrics['step_delta_mem_norm_max'] = float(inner_loop_stats['step_delta_mem_norm_max'].max())
        metrics['step_delta_mem_norm_min'] = float(inner_loop_stats['step_delta_mem_norm_min'].min())
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

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


class CurriculumTrainer(CustomTrainer):
    def __init__(self, *args, step_offset=0, curriculum_stage=0, curriculum_n_kv=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_offset = step_offset
        self.curriculum_stage = curriculum_stage
        self.curriculum_n_kv = curriculum_n_kv

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        logs['curriculum_stage'] = self.curriculum_stage
        logs['curriculum_n_kv'] = self.curriculum_n_kv
        original_step = self.state.global_step
        self.state.global_step += self.step_offset
        super().log(logs, start_time=start_time)
        self.state.global_step = original_step


@dataclass
class ExperimentArgs:
    config: Optional[str] = field(default=None)
    exp_path: Optional[str] = field(default=None)
    per_device_batch_size: int = field(default=64)
    data_path: str = field(default='./data/N2-K4V4-S4(32-64)_1M')
    tokenizer_path: str = field(default='./tokenizers/kv_alphabet_62/')
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
    freeze_backbone: Optional[bool] = field(default=False)
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
    memory_update: Optional[str] = field(default="gradient")
    use_mem_residual: Optional[bool] = field(default=False)
    use_reconstruction_loss: Optional[bool] = field(default=False)
    reconstruction_loss_weight: Optional[float] = field(default=1.0)
    # Curriculum learning parameters
    curriculum_enabled: Optional[bool] = field(default=False)
    curriculum_threshold: Optional[float] = field(default=0.95)
    curriculum_levels: Optional[str] = field(default="4,8,16,32,64,128")
    curriculum_data_dir: Optional[str] = field(default="./data")
    curriculum_stage_overrides: Optional[str] = field(default=None)


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
        for section in ['model', 'training', 'dataset', 'gradmem', 'hopfield', 'curriculum']:
            if section in cfg:
                for key, value in cfg[section].items():
                    if key == 'stage_overrides':
                        args.curriculum_stage_overrides = json.dumps({str(k): v for k, v in value.items()})
                    elif hasattr(args, key):
                        setattr(args, key, value)

        # Set exp_path from config if not explicitly set
        if 'exp_path' not in vars(args) or args.exp_path is None:
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
        memory_update=args.memory_update,
        use_mem_residual=args.use_mem_residual,
        use_reconstruction_loss=args.use_reconstruction_loss,
        reconstruction_loss_weight=args.reconstruction_loss_weight
    )

    # Create gradmemgpt model
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
        return collate_fn(batch, tokenizer, max_context_length=args.max_context_length)

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    if args.curriculum_enabled:
        curriculum_levels = [int(x.strip()) for x in args.curriculum_levels.split(',')]
        logger.info(f'curriculum learning enabled: levels={curriculum_levels}, threshold={args.curriculum_threshold}')

        stage_overrides = {}
        if args.curriculum_stage_overrides:
            stage_overrides = {int(k): v for k, v in json.loads(args.curriculum_stage_overrides).items()}
            logger.info(f'curriculum stage overrides: {stage_overrides}')

        MODEL_PARAM_MAP = {
            'inner_lr': ('lr', 'lr'),
        }
        TRAINING_PARAM_MAP = {
            'learning_rate', 'warmup_steps', 'weight_decay', 'per_device_batch_size',
            'max_steps', 'eval_steps', 'logging_steps', 'early_stopping_patience',
        }

        data_dir = Path(args.curriculum_data_dir)
        dataset_name_template = "N{n_kv}-K2V2-V62_1M"

        all_metrics = {}
        cumulative_steps = 0
        for stage_idx, n_kv in enumerate(curriculum_levels):
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

            dataset_name = dataset_name_template.format(n_kv=n_kv)
            data_path = data_dir / dataset_name
            logger.info(f'curriculum stage {stage_idx}/{len(curriculum_levels)-1}: N={n_kv}, data_path={data_path}')

            stage_dataset = datasets.load_from_disk(str(data_path))

            stage_output_dir = output_dir / f'stage_{stage_idx}_N{n_kv}'
            stage_output_dir.mkdir(parents=True, exist_ok=True)

            def stage_data_collator(batch):
                return collate_fn(batch, tokenizer, max_context_length=args.max_context_length)

            training_args = TrainingArguments(
                output_dir=stage_output_dir,
                logging_dir=stage_output_dir,

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
                curriculum_n_kv=n_kv,
            )

            trainer.train()
            cumulative_steps += trainer.state.global_step

            is_last_stage = (stage_idx == len(curriculum_levels) - 1)

            model = trainer.model
            best_model_path = stage_output_dir / 'best_model'
            trainer.save_model(str(best_model_path))

            if curriculum_cb.threshold_reached and not is_last_stage:
                logger.info(f'curriculum stage {stage_idx} (N={n_kv}): threshold reached, advancing to next stage')
            elif curriculum_cb.threshold_reached and is_last_stage:
                logger.info(f'curriculum complete! final stage (N={n_kv}) threshold reached')
            else:
                logger.info(f'curriculum stage {stage_idx} (N={n_kv}): threshold NOT reached, stopping curriculum')

            metrics = trainer.evaluate(stage_dataset['valid'])
            all_metrics[f'stage_{stage_idx}_N{n_kv}'] = metrics
            logger.info(f'stage {stage_idx} (N={n_kv}) final metrics: {metrics}')

            if not curriculum_cb.threshold_reached:
                break

        logger.info('curriculum training done. running final evaluation on last completed stage...')
        final_dataset = datasets.load_from_disk(str(data_dir / dataset_name_template.format(
            n_kv=curriculum_levels[min(stage_idx, len(curriculum_levels) - 1)]
        )))
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


if __name__ == '__main__':
    main()