import json
import logging
import os
import sys
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
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    HfArgumentParser,
)

# Add the ARMT submodule to the Python path
submodule_path = Path(__file__).parent / "associative-recurrent-memory-transformer"
sys.path.insert(0, str(submodule_path))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger("")

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def collate_fn(batch, tokenizer):
    # Build two parts: context and (query + target), then concatenate in token space.
    context = [item["context"].strip() for item in batch]
    query = [item["query"] + item["target"] for item in batch]

    context_input_ids = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=True,
        padding="max_length",
        max_length=160,
    ).input_ids

    query_encoded = tokenizer(
        query,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        pad_to_multiple_of=8,
        return_offsets_mapping=True,
    )
    query_input_ids = query_encoded["input_ids"]
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

    input_ids = torch.cat([context_input_ids, query_input_ids], dim=1)
    attention_mask = (input_ids != pad_id).to(dtype=torch.long)

    labels = torch.full_like(input_ids, -100)
    labels[:, context_input_ids.size(1) :] = query_labels

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_logits_for_metrics(eval_pred, labels):
    # Model may return tuple/dict; we only need argmax(logits) for metrics.
    if isinstance(eval_pred, tuple):
        # HF Trainer sometimes passes a tuple of multiple tensors (e.g., logits, hidden_states, etc).
        # Prefer the 3D tensor [batch, seq, vocab] as logits; otherwise fall back to the first element.
        logits = None
        for item in eval_pred:
            if hasattr(item, "dim") and callable(item.dim):
                try:
                    if item.dim() == 3:
                        logits = item
                        break
                except Exception:
                    pass
        if logits is None:
            logits = eval_pred[0]
        # Thinking impl may append inner-loop stats dict; old impl typically does not.
        inner_loop_stats = eval_pred[1] if (len(eval_pred) > 1 and isinstance(eval_pred[1], dict)) else {}
    elif isinstance(eval_pred, dict):
        logits = eval_pred.get("logits", eval_pred)
        inner_loop_stats = {k: v for k, v in eval_pred.items() if k != "logits"}
    else:
        logits = eval_pred
        device = logits.device if hasattr(logits, "device") else None
        inner_loop_stats = {
            "mem_norm_mean": torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            "mem_norm_max": torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            "mem_norm_min": torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
        }

    return (logits.argmax(dim=-1), inner_loop_stats)


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    # Handle both tuple and EvalPrediction
    if isinstance(eval_pred, tuple):
        if len(eval_pred) >= 2:
            predictions = eval_pred[0]
            labels = eval_pred[1]
            inputs = eval_pred[2] if len(eval_pred) > 2 else None
        else:
            raise ValueError(f"eval_pred tuple must have at least 2 elements, got {len(eval_pred)}")
    else:
        predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs

    if isinstance(predictions, tuple):
        preds, inner_loop_stats = predictions
    else:
        preds = predictions
        device = preds.device if hasattr(preds, "device") else None
        inner_loop_stats = {
            "mem_norm_mean": torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            "mem_norm_max": torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
            "mem_norm_min": torch.tensor(-1.0, device=device) if device else torch.tensor(-1.0),
        }

    # Some older ARMT code paths can end up with flattened predictions (1D) after preprocess/gather.
    # If that happens, try to reshape back to label shape when possible.
    try:
        if hasattr(preds, "ndim") and hasattr(labels, "ndim") and preds.ndim == 1 and labels.ndim == 2:
            if preds.size == labels.size:
                preds = preds.reshape(labels.shape)
    except Exception:
        pass

    preds = preds[..., :-1]
    labels = labels[..., 1:]

    mask = labels != -100
    for t_id in ignore_token_ids:
        mask &= labels != t_id

    masked_predictions = preds[mask]
    masked_labels = labels[mask]
    accuracy = (masked_predictions == masked_labels).mean() if masked_labels.size else 0.0

    exact_match = np.mean(
        [
            np.all(pred[mask[i]] == lab[mask[i]])
            for i, (pred, lab) in enumerate(zip(preds, labels))
            if np.any(mask[i])
        ]
    )

    # Debug: print first few samples (target-only view)
    if inputs is not None:
        for i in range(min(5, len(preds))):
            pred = preds[i]
            lab = labels[i]
            m = lab != -100
            pred_m = pred[m]
            lab_m = lab[m].copy()
            lab_m[lab_m == -100] = tokenizer.pad_token_id
            print("p:", tokenizer.decode(pred_m, skip_special_tokens=True).strip())
            print("t:", tokenizer.decode(lab_m, skip_special_tokens=True).strip())
            print("-" * 50)

    metrics = {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }

    # Add memory stats if present
    if isinstance(inner_loop_stats, dict) and "mem_norm_mean" in inner_loop_stats:
        try:
            metrics.update(
                {
                    "mem_norm_mean": float(np.mean(inner_loop_stats["mem_norm_mean"])),
                    "mem_norm_max": float(np.max(inner_loop_stats["mem_norm_max"])),
                    "mem_norm_min": float(np.min(inner_loop_stats["mem_norm_min"])),
                }
            )
        except Exception:
            # stats may be missing / wrong dtype; ignore
            pass
    return metrics


class StopOnMetricValue(TrainerCallback):
    def __init__(self, metric_name: str, value: float, higher_is_better: bool = True):
        self.metric_name = metric_name
        self.value = value
        self.higher_is_better = higher_is_better

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = self.metric_name if self.metric_name.startswith("eval_") else f"eval_{self.metric_name}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            return
        operator = np.greater_equal if self.higher_is_better else np.less_equal
        if operator(metric_value, self.value):
            control.should_training_stop = True
            logger.info(
                f"metric {self.metric_name}={metric_value:.4f} >= {self.value:.4f}, stopping training.."
            )


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        num_training_steps = int(num_training_steps / 0.9)  # make final lr not zero for linear schedule
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs["patience"] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    armt_impl: str = field(default="thinking")  # one of: ["old", "thinking"]
    dataset_name: str = field(default="squad")
    tokenizer_path: str = field(default="./tokenizers/kv_alphabet_62/")
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default="token_accuracy")
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-4)
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.999)
    adam_epsilon: Optional[float] = field(default=1e-8)
    lr_scheduler_type: Optional[str] = field(default="constant_with_warmup")
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    # Used only when training from scratch (pretrained_model is None)
    base_model: Optional[str] = field(default=None)
    pretrained_model: Optional[str] = field(default=None)
    init_checkpoint: Optional[str] = field(default=None)
    model_cpt: Optional[str] = field(default=None)  # path to ARMT checkpoint directory (optional)
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)
    # ARMT parameters
    num_mem_tokens: Optional[int] = field(default=8)
    d_mem: Optional[int] = field(default=512)
    segment_size: Optional[int] = field(default=512)
    segment_alignment: Optional[str] = field(default="left")
    layers_attr: Optional[str] = field(default="model.layers")
    wrap_pos: Optional[bool] = field(default=False)
    correction: Optional[bool] = field(default=True)
    n_heads: Optional[int] = field(default=1)
    use_denom: Optional[bool] = field(default=True)
    reading_depth_multiplier: Optional[int] = field(default=1)
    writing_depth_multiplier: Optional[int] = field(default=1)


if __name__ == "__main__":
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger

    logger = get_logger("")
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f"num processes: {accel.num_processes}")
    logger.info(f"mixed precision: {accel.mixed_precision}")
    logger.info(f"accelerator state: {accel.state}")

    if args.armt_impl not in ("thinking", "old"):
        raise ValueError(f"--armt_impl must be one of ['old', 'thinking'], got: {args.armt_impl}")

    assert not (
        args.pretrained_model is not None and args.base_model is not None
    ), "only one of these args must be set"

    if accel.is_main_process:
        config = {"cli_args": dict(vars(args))}
        logger.info(f"saving experiment configuration to {args.exp_path}")
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump(config, open(os.path.join(args.exp_path, "config.json"), "w"), indent=4)

    if args.pretrained_model is None:
        if args.base_model is None:
            raise ValueError("When --pretrained_model is not set, --base_model must be provided.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if args.base_model == "gpt2":
            base_config = AutoConfig.from_pretrained("gpt2")
            base_config.n_layer = args.n_layer
            base_config.n_head = args.n_head
            base_config.n_embd = args.n_embd
        elif args.base_model == "pythia":
            base_config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
            base_config.num_hidden_layers = args.n_layer
            base_config.num_attention_heads = args.n_head
            base_config.hidden_size = args.n_embd
            base_config.intermediate_size = base_config.hidden_size * 4
        elif args.base_model == "llama":
            base_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
            base_config.num_hidden_layers = args.n_layer
            base_config.num_attention_heads = args.n_head
            base_config.num_key_value_heads = args.n_head
            base_config.hidden_size = args.n_embd
            base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads
            base_config.intermediate_size = base_config.hidden_size * 4
        else:
            raise ValueError(f"Unsupported base model: {args.base_model}")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base_config.torch_dtype = "float32"
        base_config.vocab_size = tokenizer.vocab_size
        base_config.pad_token_id = tokenizer.pad_token_id
        base_config.bos_token_id = tokenizer.bos_token_id
        base_config.eos_token_id = tokenizer.eos_token_id
        base_config.use_cache = False
        base_model_name = None
    else:
        base_config = None
        base_model_name = args.pretrained_model
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
        cfg_kwargs = dict(
            base_model_name=base_model_name,
            base_model_config=base_config,
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
        if args.armt_impl == "thinking":
            cfg_kwargs.update(
                dict(
                    reading_depth_multiplier=args.reading_depth_multiplier,
                    writing_depth_multiplier=args.writing_depth_multiplier,
                )
            )
        armt_config = ARMTConfigCls(**cfg_kwargs)
        if base_config is not None:
            armt_config.vocab_size = base_config.vocab_size
            armt_config.pad_token_id = base_config.pad_token_id
            armt_config.bos_token_id = getattr(base_config, "bos_token_id", None)
            armt_config.eos_token_id = getattr(base_config, "eos_token_id", None)
        return armt_config

    # Load ARMT model from checkpoint or create a new one
    if args.model_cpt is not None and args.model_cpt != "None":
        checkpoint_dir = args.model_cpt
        logger.info(f"Loading ARMT model from checkpoint: {checkpoint_dir}")
        try:
            model = ARMTModelCls.from_pretrained(checkpoint_dir)
            logger.info(f"Successfully loaded ARMT model from {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Failed to load as pretrained model: {e}")
            logger.info("Falling back to state-dict loading...")

            armt_config = build_armt_config()
            model = ARMTModelCls(armt_config)

            checkpoint_paths = [
                os.path.join(checkpoint_dir, "model_best", "model.safetensors"),
                os.path.join(checkpoint_dir, "model_best", "pytorch_model.bin"),
                os.path.join(checkpoint_dir, "model.safetensors"),
                os.path.join(checkpoint_dir, "pytorch_model.bin"),
                checkpoint_dir,
            ]
            loaded = False
            for cpt_path in checkpoint_paths:
                if os.path.exists(cpt_path):
                    logger.info(f"Loading from: {cpt_path}")
                    if cpt_path.endswith(".safetensors"):
                        state_dict = load_file(cpt_path)
                    else:
                        state_dict = torch.load(cpt_path, map_location="cpu")
                        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                            state_dict = state_dict["model_state_dict"]
                    missing_k, unexpected_k = model.load_state_dict(state_dict, strict=False)
                    if len(missing_k) != 0:
                        logger.info(f"{missing_k} were not loaded from checkpoint! These params were initialized.")
                    if len(unexpected_k) != 0:
                        logger.info(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f"Could not find a checkpoint file in {checkpoint_dir}. Tried: {checkpoint_paths}")
    else:
        armt_config = build_armt_config()
        model = ARMTModelCls(armt_config)

    if args.init_checkpoint is not None:
        missing_k, unexpected_k = model.load_state_dict(load_file(args.init_checkpoint), strict=False)
        if len(missing_k) != 0:
            logger.info(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
        if len(unexpected_k) != 0:
            logger.info(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")

    if accel.mixed_precision == "bf16":
        model = model.to(torch.bfloat16)

    logger.info(f"model config: {model.config}")
    logger.info(f"model: {model}")
    logger.info(f"model.dtype: {model.dtype}")

    raw_dataset = datasets.load_dataset(args.dataset_name)
    if "squad" in args.dataset_name:
        from squad_utils import preprocess_dataset
    elif "phonebook" in args.dataset_name:
        from phonebook_utils import preprocess_dataset
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    dataset = preprocess_dataset(raw_dataset)

    def data_collator(batch):
        return collate_fn(batch, tokenizer)

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in []]

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer)

    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    wandb_run_name = os.environ.get("WANDB_NAME", None)
    if wandb_run_name:
        logger.info(f"Using WANDB_NAME from environment: {wandb_run_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        label_names=["labels"],
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
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=wandb_run_name,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_for_metrics=["inputs"],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            StopOnMetricValue(metric_name="exact_match", value=1.0, higher_is_better=True),
        ],
    )
    trainer.train()
    logger.info("training done. running final evaluation...")
    metrics = trainer.evaluate(dataset["valid"])
    logger.info(f"{metrics}")
    trainer.save_metrics(split="all", metrics=metrics)
    trainer.state.save_to_json(output_dir / "trainer_state.json")


