import json
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
import types
from typing import Dict, Optional
from dataclasses import dataclass, field
import datasets
import weave
import wandb
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

# Add the submodule to the Python path
submodule_path = Path(__file__).parent / "associative-recurrent-memory-transformer"
sys.path.insert(0, str(submodule_path))


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer, use_segmented_inputs: bool = False, max_context_length: Optional[int] = None):
    """
    If use_segmented_inputs=True (babi runs), build ARMT segmented inputs:
      - segment 0: context
      - segment 1: query + target
    and return `input_segmented=True` similarly to run_armt_on_squad.py.

    Otherwise keep the legacy KV-retrieval behavior (single concatenated tensor).
    """
    context = [item["context"] for item in batch]
    query = [item["query"] + item["target"] for item in batch]

    if use_segmented_inputs:
        # Need a pad token for dynamic padding.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        context_kwargs = dict(
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            pad_to_multiple_of=8,
        )
        if max_context_length is not None:
            context_kwargs.update(dict(max_length=max_context_length, truncation=True))

        context_encoded = tokenizer(context, **context_kwargs)
        context_input_ids = context_encoded["input_ids"]
        context_attention_mask = context_encoded["attention_mask"]

        query_encoded = tokenizer(
            query,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            pad_to_multiple_of=8,
            return_offsets_mapping=True,
        )
        query_input_ids = query_encoded["input_ids"]
        query_attention_mask = query_encoded["attention_mask"]
        offsets_mapping = query_encoded["offset_mapping"]

        # Build labels mask for target tokens inside (query + target)
        labels_mask = torch.zeros_like(query_input_ids)
        for i, item in enumerate(batch):
            query_seq_len = len(item["query"])
            target_seq_len = len(item["target"])
            target_st, target_end = query_seq_len, query_seq_len + target_seq_len
            in_target = False
            for j in range(len(offsets_mapping[i]) - 1, -1, -1):
                st, end = offsets_mapping[i][j]
                if st < target_end and end > target_st:
                    labels_mask[i, j] = 1
                    in_target = True
                elif in_target:
                    break

        pad_id = tokenizer.pad_token_id
        query_labels = query_input_ids * labels_mask + (1 - labels_mask) * -100
        query_labels = query_labels.masked_fill(query_input_ids == pad_id, -100)

        context_labels = torch.full_like(context_input_ids, -100)
        labels = torch.cat([context_labels, query_labels], dim=1)

        return {
            "input_ids": [context_input_ids, query_input_ids],
            "attention_mask": [context_attention_mask, query_attention_mask],
            "labels": labels,
            "input_segmented": True,
        }

    # Legacy (non-babi): single concatenated tensor, no segmented forward.
    context_input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False, padding=False).input_ids
    query_encoded = tokenizer(
        query, return_tensors="pt", add_special_tokens=False, padding=False, return_offsets_mapping=True
    )
    query_input_ids = query_encoded["input_ids"]
    offsets_mapping = query_encoded["offset_mapping"]

    # input_seq: 0, target_seq: 1, seq = input_seq + target_seq
    labels_mask = torch.zeros_like(query_input_ids)
    for i, item in enumerate(batch):
        query_seq_len = len(item["query"])
        target_seq_len = len(item["target"])
        target_st, target_end = query_seq_len, query_seq_len + target_seq_len
        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            if st < target_end and end > target_st:
                labels_mask[i, j] = 1
                in_target = True
            elif in_target:
                break

    pad_token_id = tokenizer.pad_token_id
    batch_size = context_input_ids.size(0)
    max_context_len = context_input_ids.size(1)
    max_query_len = query_input_ids.size(1)
    device = context_input_ids.device

    input_ids = torch.full(
        (batch_size, max_context_len + max_query_len), pad_token_id, dtype=torch.long, device=device
    )
    input_ids[:, :max_context_len] = context_input_ids
    input_ids[:, max_context_len : max_context_len + max_query_len] = query_input_ids

    labels = torch.full((batch_size, max_context_len + max_query_len), -100, dtype=torch.long, device=device)
    query_labels = query_input_ids * labels_mask + (1 - labels_mask) * -100
    labels[:, max_context_len : max_context_len + max_query_len] = query_labels

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def preprocess_logits_for_metrics(eval_pred, labels):
    # ARMT ThinkingARMTForCausalLM may return a dict or tuple from forward
    # Handle both cases: tuple (logits, ...) or just logits
    if isinstance(eval_pred, tuple):
        # If it's a tuple, extract logits (first element)
        logits = eval_pred[0]
        # If there are additional elements, they might be stats
        if len(eval_pred) > 1:
            inner_loop_stats = eval_pred[1] if isinstance(eval_pred[1], dict) else {}
        else:
            inner_loop_stats = {}
    elif isinstance(eval_pred, dict):
        # If it's a dict, extract logits
        logits = eval_pred.get('logits', eval_pred)
        inner_loop_stats = {k: v for k, v in eval_pred.items() if k != 'logits'}
    else:
        # Assume it's just logits
        logits = eval_pred
        # Get device from logits to ensure stats are on the same device
        device = logits.device if hasattr(logits, 'device') else None
        inner_loop_stats = {
            'mem_norm_mean': torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            'mem_norm_max': torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            'mem_norm_min': torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
        }
        
    
    # saves gpu RAM, as HF Trainer accumulates all eval logits on GPU
    return (logits.argmax(dim=-1), inner_loop_stats)


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    # Handle case where eval_pred might be a tuple instead of EvalPrediction object
    if isinstance(eval_pred, tuple):
        # If eval_pred is a tuple, it's likely (predictions, labels) or (predictions, labels, inputs)
        if len(eval_pred) >= 2:
            predictions = eval_pred[0]
            labels = eval_pred[1]
            inputs = eval_pred[2] if len(eval_pred) > 2 else {}
        else:
            raise ValueError(f"eval_pred tuple must have at least 2 elements, got {len(eval_pred)}")
    else:
        # Standard EvalPrediction object
        predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    
    # Handle both tuple (preds, stats) and just preds
    if isinstance(predictions, tuple):
        preds, inner_loop_stats = predictions
    else:
        preds = predictions
        # Get device from predictions to ensure stats are on the same device
        device = preds.device if hasattr(preds, 'device') else None
        inner_loop_stats = {
            'mem_norm_mean': torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            'mem_norm_max': torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            'mem_norm_min': torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
        }
    
    preds = preds[..., :-1]
    labels = labels[..., 1:]

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

    # For debugging: show first 5 examples (only when `inputs` is a dense tensor/array)
    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else inputs
    if isinstance(input_ids, (np.ndarray, torch.Tensor)) and getattr(input_ids, "ndim", 0) >= 2:
        for i in range(min(5, len(preds))):
            pred = preds[i]
            label = labels[i]
            if isinstance(input_ids, np.ndarray):
                inp = input_ids[i]
            else:
                inp = input_ids[i].detach().cpu().numpy()
            mask_i = label != -100
            pred_masked = pred[mask_i]
            label_masked = label[mask_i]
            inp_masked = inp[mask_i] if len(inp) == len(label) else inp
            print("i:", tokenizer.decode(inp_masked, skip_special_tokens=True).strip())
            print("p:", tokenizer.decode(pred_masked, skip_special_tokens=True).strip())
            print("t:", tokenizer.decode(label_masked, skip_special_tokens=True).strip())
            print("-" * 50)

    # Convert stats to numpy if they're tensors
    if isinstance(inner_loop_stats, dict):
        stats = {}
        for k, v in inner_loop_stats.items():
            if isinstance(v, torch.Tensor):
                stats[k] = v.cpu().numpy() if v.is_cuda else v.numpy()
            else:
                stats[k] = np.array(v)
        inner_loop_stats = stats

    metrics = {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }
    
    # Add memory stats if available
    if 'mem_norm_mean' in inner_loop_stats:
        mem_norm_mean = inner_loop_stats['mem_norm_mean']
        mem_norm_max = inner_loop_stats['mem_norm_max']
        mem_norm_min = inner_loop_stats['mem_norm_min']
        metrics.update({
            "mem_norm_mean": float(np.mean(mem_norm_mean)) if mem_norm_mean.size > 0 else 0.0,
            "mem_norm_max": float(np.max(mem_norm_max)) if mem_norm_max.size > 0 else 0.0,
            "mem_norm_min": float(np.min(mem_norm_min)) if mem_norm_min.size > 0 else 0.0,
        })
    
    if 'step_delta_mem_norm_mean' in inner_loop_stats:
        metrics.update({
            "step_delta_mem_norm_mean": float(np.mean(inner_loop_stats['step_delta_mem_norm_mean'])),
            "step_delta_mem_norm_max": float(np.max(inner_loop_stats['step_delta_mem_norm_max'])),
            "step_delta_mem_norm_min": float(np.min(inner_loop_stats['step_delta_mem_norm_min'])),
            "delta_mem_norm_mean": float(np.mean(inner_loop_stats['delta_mem_norm_mean'])),
            "delta_mem_norm_max": float(np.max(inner_loop_stats['delta_mem_norm_max'])),
            "delta_mem_norm_min": float(np.min(inner_loop_stats['delta_mem_norm_min'])),
        })
    if 'rec_loss' in inner_loop_stats:
        metrics['rec_loss'] = float(np.mean(inner_loop_stats['rec_loss']))
    if 'target_loss' in inner_loop_stats:
        metrics['target_loss'] = float(np.mean(inner_loop_stats['target_loss']))
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
    armt_impl: str = field(default="thinking")  # one of: ["old", "thinking"]
    data_path: str = field(default='N2-K4V4-S4(32-64)_1M')  # Subset name for HuggingFace Hub dataset irodkin/kv_retrieval
    tokenizer_path: str = field(default='./tokenizers/kv_alphabet_62/')
    max_context_length: Optional[int] = field(default=None)
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
    model_cpt: Optional[str] = field(default=None)  # Path to ARMT model checkpoint directory
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    # ARMT parameters
    num_mem_tokens: Optional[int] = field(default=8)
    d_mem: Optional[int] = field(default=512)
    segment_size: Optional[int] = field(default=512)
    segment_alignment: Optional[str] = field(default='left')
    layers_attr: Optional[str] = field(default='model.layers')
    wrap_pos: Optional[bool] = field(default=False)
    correction: Optional[bool] = field(default=True)
    n_heads: Optional[int] = field(default=1)
    use_denom: Optional[bool] = field(default=True)
    reading_depth_multiplier: Optional[int] = field(default=1)
    writing_depth_multiplier: Optional[int] = field(default=1)


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

    if args.armt_impl not in ("thinking", "old"):
        raise ValueError(f"--armt_impl must be one of ['old', 'thinking'], got: {args.armt_impl}")

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

    if args.armt_impl == "thinking":
        from modeling_amt.thinking import ThinkingARMTConfig as ARMTConfigCls
        from modeling_amt.thinking import ThinkingARMTForCausalLM as ARMTModelCls
    else:
        from modeling_amt.model import ARMTConfig as ARMTConfigCls
        from modeling_amt.model import ARMTForCausalLM as ARMTModelCls

    def build_armt_config() -> "ARMTConfigCls":
        base_model_name = args.pretrained_model
        base_model_config = config if args.pretrained_model is None else None

        cfg_kwargs = dict(
            base_model_config=base_model_config,
            num_mem_tokens=args.num_mem_tokens,
            d_mem=args.d_mem,
            segment_size=args.segment_size,
            segment_alignment=args.segment_alignment,
            layers_attr=args.layers_attr,
            wrap_pos=args.wrap_pos,
            correction=args.correction,
            n_heads=args.n_heads,
            use_denom=args.use_denom,
        )
        if base_model_name is not None:
            cfg_kwargs.update(dict(base_model_name=base_model_name))
            cfg_kwargs.pop("base_model_config", None)
        if args.armt_impl == "thinking":
            cfg_kwargs.update(
                dict(
                    reading_depth_multiplier=args.reading_depth_multiplier,
                    writing_depth_multiplier=args.writing_depth_multiplier,
                )
            )
        armt_config = ARMTConfigCls(**cfg_kwargs)

        if base_model_config is not None:
            armt_config.vocab_size = base_model_config.vocab_size
            armt_config.pad_token_id = base_model_config.pad_token_id
            armt_config.bos_token_id = getattr(base_model_config, "bos_token_id", None)
            armt_config.eos_token_id = getattr(base_model_config, "eos_token_id", None)
        return armt_config

    # Load ARMT model from checkpoint or create new one
    if args.model_cpt is not None and args.model_cpt != 'None':
        # Find the latest checkpoint directory if args.model_cpt is a directory with checkpoint-* subdirectories
        checkpoint_dir = args.model_cpt
        if os.path.isdir(args.model_cpt):
            # Look for checkpoint-* directories
            import re
            checkpoint_dirs = []
            files = os.listdir(args.model_cpt)
            logger.info(f'Found {len(files)} files in {args.model_cpt}: {files}')
            for item in files:
                item_path = os.path.join(args.model_cpt, item)
                if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                    # Extract step number from checkpoint-$STEPS
                    match = re.match(r'checkpoint-(\d+)', item)
                    if match:
                        step_num = int(match.group(1))
                        checkpoint_dirs.append((step_num, item_path))
            
            if checkpoint_dirs:
                # Sort by step number and get the latest (highest step number)
                checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
                latest_step, checkpoint_dir = checkpoint_dirs[0]
                logger.info(f'Found {len(checkpoint_dirs)} checkpoint(s), loading from latest: checkpoint-{latest_step}')
            else:
                # No checkpoint-* directories found, use the directory directly
                checkpoint_dir = args.model_cpt
                logger.info(f'No checkpoint-* directories found in {args.model_cpt}, using directory directly')
        
        # Load ARMT model from checkpoint directory
        logger.info(f'Loading ARMT model from checkpoint: {checkpoint_dir}')
        try:
            # Try loading as a saved model directory (from_pretrained)
            model = ARMTModelCls.from_pretrained(checkpoint_dir)
            logger.info(f'Successfully loaded ARMT model from {checkpoint_dir}')
        except Exception as e:
            logger.warning(f'Failed to load as pretrained model: {e}')
            logger.info('Trying to load from state dict files...')
            armt_config = build_armt_config()
            model = ARMTModelCls(armt_config)
            # Try different checkpoint file locations
            checkpoint_paths = [
                os.path.join(checkpoint_dir, "model_best", "model.safetensors"),
                os.path.join(checkpoint_dir, "model_best", "pytorch_model.bin"),
                os.path.join(checkpoint_dir, "model.safetensors"),
                os.path.join(checkpoint_dir, "pytorch_model.bin"),
                checkpoint_dir,  # Direct path to checkpoint file
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
        if hasattr(model, "segment_size"):
            model.segment_size = args.segment_size
        if hasattr(model, "get_layers"):
            for layer in model.get_layers():
                if hasattr(layer, "segment_size"):
                    layer.segment_size = args.segment_size
    else:
        # Create new ARMT model
        armt_config = build_armt_config()
        model = ARMTModelCls(armt_config)

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    # Move model to device first to ensure parameters are initialized
    device = accel.device
    model = model.to(device)
    
    # Verify model has parameters before proceeding
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f'model parameter count: {param_count:,}')
    if param_count == 0:
        raise RuntimeError("Model has no parameters! Check model initialization.")
    
    if accel.mixed_precision == 'bf16':
        model = model.to(torch.bfloat16)

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')
    logger.info(f'model.dtype: {model.dtype}')

    # Load dataset: check if local path exists, otherwise load from HuggingFace Hub
    if os.path.exists(args.data_path):
        logger.info(f"Loading dataset from disk: {args.data_path}")
        dataset = datasets.load_from_disk(args.data_path)
    else:
        logger.info(f"Loading dataset from HuggingFace Hub: irodkin/kv_retrieval (subset: {args.data_path})")
        dataset = datasets.load_dataset("irodkin/kv_retrieval", name=args.data_path)

    use_segmented_inputs = "babi" in str(args.data_path).lower()
    if use_segmented_inputs:
        logger.info("Detected babi run (data_path contains 'babi'): using ARMT segmented inputs (context / query+target)")

    def data_collator(batch):
        return collate_fn(
            batch,
            tokenizer,
            use_segmented_inputs=use_segmented_inputs,
            max_context_length=args.max_context_length,
        )

    # Target sequence looks like: "XXXX!|"
    # Let's not count ! and | in the accuracy calculation
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    # Define custom compute metrics function with ignored tokens
    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    # Get wandb run name from environment variable if set
    wandb_run_name = os.environ.get('WANDB_NAME', None)
    if wandb_run_name:
        logger.info(f'Using WANDB_NAME from environment: {wandb_run_name}')

    # Training arguments
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
        run_name=wandb_run_name,  # Use WANDB_NAME environment variable if set
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

    # HF Trainer FLOPs estimation expects tensor inputs and calls `.numel()` on `inputs[model.main_input_name]`.
    # With `input_segmented=True`, we pass `input_ids` as a list of segment tensors, so we need a custom estimator.
    if use_segmented_inputs:
        def _floating_point_ops_segmented(self, input_dict, exclude_embeddings: bool = True):
            x = input_dict.get(getattr(self, "main_input_name", "input_ids"))
            if isinstance(x, (list, tuple)):
                tokens = 0
                for t in x:
                    if hasattr(t, "numel"):
                        tokens += int(t.numel())
            elif hasattr(x, "numel"):
                tokens = int(x.numel())
            else:
                tokens = 0
            return 6 * tokens * self.num_parameters(exclude_embeddings=exclude_embeddings)

        model.floating_point_ops = types.MethodType(_floating_point_ops_segmented, model)

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
                   StopOnMetricValue(metric_name='exact_match', value=0.99, higher_is_better=True),
                   ],
    )
    # Train the model
    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
    trainer.state.save_to_json(output_dir / 'trainer_state.json')
