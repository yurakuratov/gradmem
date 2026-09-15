#!/usr/bin/env python3
"""Train EnergyGradMem directly on integer-encoded Zoology multihop data."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import accelerate
import datasets
import numpy as np
import torch
import transformers
from safetensors.torch import load_file
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    LlamaConfig,
    TrainingArguments,
)

from energy_gradmem import EnergyGradMem
from run_energy_gradmem_on_kv_retrieval import (
    EnergyGradMemExperimentArgs,
    EnergyFreezeCallback,
    build_model_config,
    reduce_inner_loop_stat,
)
from run_gradmemgpt_on_kv_retrieval import CustomTrainer, StopOnMetricValue
from zoology_multihop_dataset import (
    BOS_ID,
    CONTEXT_END_ID,
    CONTEXT_START_ID,
    EOS_ID,
    KV_RECORD_WIDTH,
    PAD_ID,
    QUERY_RECORD_WIDTH,
    VOCAB_SIZE,
    configuration_name,
    parse_context_records,
    parse_query_keys,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "gradmem")

logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger("")
PROGRESSION_METRICS_FILENAME = "progression_metrics.json"


def warmup_linear_decay_to_ratio_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    final_ratio: float,
) -> float:
    """Return the LR multiplier for exact linear decay to a nonzero floor."""

    if num_training_steps <= num_warmup_steps:
        raise ValueError("num_training_steps must be greater than num_warmup_steps")
    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be non-negative")
    if not math.isfinite(final_ratio) or not 0.0 < final_ratio <= 1.0:
        raise ValueError("final_ratio must be finite and in (0, 1]")

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    decay_steps = num_training_steps - num_warmup_steps
    decay_progress = min(
        max(float(current_step - num_warmup_steps) / float(decay_steps), 0.0),
        1.0,
    )
    return 1.0 - (1.0 - final_ratio) * decay_progress


def create_warmup_linear_decay_to_ratio_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    final_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a scheduler whose final-step LR is exactly ``final_ratio * LR``."""

    def lr_lambda(current_step: int) -> float:
        return warmup_linear_decay_to_ratio_lambda(
            current_step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            final_ratio=final_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ZoologyTrainer(CustomTrainer):
    """Trainer with an exact nonzero-floor linear scheduler for Zoology."""

    def __init__(self, *args, outer_lr_final_ratio: float, **kwargs):
        self.outer_lr_final_ratio = outer_lr_final_ratio
        super().__init__(*args, **kwargs)

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        scheduler_type = getattr(
            self.args.lr_scheduler_type,
            "value",
            self.args.lr_scheduler_type,
        )
        if scheduler_type != "linear":
            return super().create_scheduler(num_training_steps, optimizer)

        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = create_warmup_linear_decay_to_ratio_scheduler(
                optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                final_ratio=self.outer_lr_final_ratio,
            )
        return self.lr_scheduler


@dataclass
class ZoologyExperimentArgs(EnergyGradMemExperimentArgs):
    """EnergyGradMem arguments specialized for integer Zoology examples."""

    hf_dataset: Optional[str] = field(default="irodkin/zoology_multihop")
    n_pairs: Optional[int] = field(default=8)
    hop_length: int = field(default=1)
    kv_pairs_per_segment: int = field(default=8)
    vocab_size: Optional[int] = field(default=VOCAB_SIZE)
    metric_for_best_model: Optional[str] = field(default="query_accuracy")
    stop_all_queries_exact_match_value: float = field(default=1.0)
    n_layer: Optional[int] = field(default=2)
    n_head: Optional[int] = field(default=1)
    n_embd: Optional[int] = field(default=128)
    n_mem_tokens: Optional[int] = field(default=4)
    energy_model_type: Optional[str] = field(default="segment_delta_gru")
    energy_future_mode: Optional[str] = field(default="none")
    energy_replay_weight: Optional[float] = field(default=0.0)
    reading_optimization: Optional[bool] = field(default=False)
    max_position_embeddings: int = field(default=2048)
    attention_dropout: float = field(default=0.1)
    outer_lr_final_ratio: float = field(default=0.2)
    report_to: str = field(default="wandb")
    dataloader_num_workers: int = field(default=4)


def validate_zoology_args(args: ZoologyExperimentArgs) -> None:
    if args.pretrained_model is not None:
        raise ValueError("The Zoology integer runner does not support pretrained_model")
    if args.base_model not in (None, "llama"):
        raise ValueError("The first Zoology experiment supports base_model='llama' only")
    if int(args.vocab_size) != VOCAB_SIZE:
        raise ValueError(f"Zoology requires vocab_size={VOCAB_SIZE}")
    if int(args.n_pairs) < 1:
        raise ValueError("n_pairs must be positive")
    if int(args.hop_length) < 1 or int(args.n_pairs) % int(args.hop_length):
        raise ValueError("hop_length must be positive and divide n_pairs")
    if int(args.kv_pairs_per_segment) < 1:
        raise ValueError("kv_pairs_per_segment must be positive")
    if int(args.max_position_embeddings) < 1:
        raise ValueError("max_position_embeddings must be positive")
    if not 0.0 <= float(args.attention_dropout) < 1.0:
        raise ValueError("attention_dropout must be in [0, 1)")
    if not math.isfinite(float(args.outer_lr_final_ratio)) or not (
        0.0 < float(args.outer_lr_final_ratio) <= 1.0
    ):
        raise ValueError("outer_lr_final_ratio must be finite and in (0, 1]")
    if args.lr_scheduler_type == "linear" and args.warmup_steps >= args.max_steps:
        raise ValueError(
            "linear scheduling requires warmup_steps to be less than max_steps"
        )

    expected_subset = configuration_name(args.n_pairs, args.hop_length)
    if args.hf_subset is None:
        args.hf_subset = expected_subset
    elif args.hf_subset != expected_subset:
        raise ValueError(
            f"hf_subset must match n_pairs and hop_length: expected {expected_subset!r}, "
            f"got {args.hf_subset!r}"
        )


def build_zoology_base_config(args: ZoologyExperimentArgs) -> LlamaConfig:
    """Build the small decoder used by the first Zoology experiments."""

    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=int(args.n_embd),
        intermediate_size=4 * int(args.n_embd),
        num_hidden_layers=int(args.n_layer),
        num_attention_heads=int(args.n_head),
        num_key_value_heads=int(args.n_head),
        attention_dropout=float(args.attention_dropout),
        max_position_embeddings=int(args.max_position_embeddings),
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        use_cache=False,
        torch_dtype="float32",
    )


def segment_context_input_ids(
    context_input_ids: Sequence[int],
    *,
    n_pairs: int,
    kv_pairs_per_segment: int,
) -> list[list[int]]:
    """Split one flat context only at KV-record boundaries."""

    context = [int(token_id) for token_id in context_input_ids]
    records = parse_context_records(context)
    if len(records) != int(n_pairs):
        raise ValueError(f"expected {n_pairs} context records; found {len(records)}")
    if int(kv_pairs_per_segment) < 1:
        raise ValueError("kv_pairs_per_segment must be positive")

    record_tokens = context[2:-2]
    segments = []
    for first_pair in range(0, int(n_pairs), int(kv_pairs_per_segment)):
        pair_count = min(int(kv_pairs_per_segment), int(n_pairs) - first_pair)
        token_start = first_pair * KV_RECORD_WIDTH
        token_stop = token_start + pair_count * KV_RECORD_WIDTH
        segment = record_tokens[token_start:token_stop]
        if first_pair == 0:
            segment = [BOS_ID, CONTEXT_START_ID, *segment]
        if first_pair + pair_count == int(n_pairs):
            segment = [*segment, CONTEXT_END_ID, EOS_ID]
        segments.append(segment)
    return segments


def query_target_label_positions(n_queries: int) -> list[int]:
    """Return label positions predicted immediately after each QUERY_END."""

    return [1 + (query_index + 1) * QUERY_RECORD_WIDTH for query_index in range(n_queries)]


def collate_zoology_batch(
    batch: Sequence[Dict[str, object]],
    *,
    n_pairs: int,
    kv_pairs_per_segment: int,
) -> Dict[str, object]:
    """Collate integer arrays without inserting target tokens into model input."""

    if not batch:
        raise ValueError("cannot collate an empty batch")

    segmented_contexts = [
        segment_context_input_ids(
            item["context_input_ids"],
            n_pairs=n_pairs,
            kv_pairs_per_segment=kv_pairs_per_segment,
        )
        for item in batch
    ]
    segment_count = len(segmented_contexts[0])
    if any(len(segments) != segment_count for segments in segmented_contexts):
        raise ValueError("all examples in a batch must have the same segment count")

    context_segments = []
    for segment_index in range(segment_count):
        lengths = {
            len(sample_segments[segment_index])
            for sample_segments in segmented_contexts
        }
        if len(lengths) != 1:
            raise ValueError("corresponding context segments must have equal lengths")
        context_segments.append(
            torch.tensor(
                [sample_segments[segment_index] for sample_segments in segmented_contexts],
                dtype=torch.long,
            )
        )

    query_rows = []
    label_rows = []
    hop_rows = []
    target_positions = query_target_label_positions(int(n_pairs))
    for item in batch:
        query_input_ids = [int(token_id) for token_id in item["query_input_ids"]]
        targets = [int(token_id) for token_id in item["targets"]]
        hop_distances = [int(distance) for distance in item["hop_distances"]]
        if len(parse_query_keys(query_input_ids)) != int(n_pairs):
            raise ValueError(f"expected {n_pairs} queries")
        if len(targets) != int(n_pairs) or len(hop_distances) != int(n_pairs):
            raise ValueError("targets and hop_distances must align with every query")
        if set(targets).intersection(query_input_ids):
            raise ValueError("target entities must not occur in query_input_ids")

        labels = [-100] * len(query_input_ids)
        for position, target in zip(target_positions, targets):
            labels[position] = target
        query_rows.append(query_input_ids)
        label_rows.append(labels)
        hop_rows.append(hop_distances)

    query_lengths = {len(row) for row in query_rows}
    if len(query_lengths) != 1:
        raise ValueError("all query collections in a batch must have equal lengths")

    return {
        "input_ids": {
            "context_input_ids": context_segments,
            "query_input_ids": torch.tensor(query_rows, dtype=torch.long),
            # Kept inside input_ids so Trainer retains it for grouped metrics;
            # EnergyGradMem ignores this metadata during its forward pass.
            "hop_distances": torch.tensor(hop_rows, dtype=torch.long),
        },
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def preprocess_zoology_logits(logits, labels):
    del labels
    predictions, inner_loop_stats = logits
    return predictions.argmax(dim=-1), inner_loop_stats


def _aligned_predictions(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if predictions.shape[1] == labels.shape[1] + 1:
        return predictions[:, :-1]
    if predictions.shape[1] == labels.shape[1]:
        return predictions[:, :-1]
    raise ValueError(
        "Unexpected prediction/label lengths: "
        f"predictions={predictions.shape[1]}, labels={labels.shape[1]}"
    )


def compute_zoology_metrics(eval_pred) -> Dict[str, float]:
    """Compute per-query, per-sample, and hop-stratified accuracy."""

    predictions, inner_loop_stats = eval_pred.predictions
    labels = np.asarray(eval_pred.label_ids)
    predictions = _aligned_predictions(np.asarray(predictions), labels)
    if predictions.shape != labels.shape:
        labels = labels[:, 1:]
    mask = labels != -100

    hop_distances = np.asarray(eval_pred.inputs["hop_distances"])
    if hop_distances.ndim != 2:
        raise ValueError("hop_distances must have shape [batch, queries]")

    query_correct_rows = []
    for sample_predictions, sample_labels, sample_mask, sample_hops in zip(
        predictions,
        labels,
        mask,
        hop_distances,
    ):
        query_correct = sample_predictions[sample_mask] == sample_labels[sample_mask]
        if query_correct.size != sample_hops.size:
            raise ValueError(
                "each supervised query must have one hop distance; "
                f"found {query_correct.size} predictions and {sample_hops.size} distances"
            )
        query_correct_rows.append(query_correct)

    query_correct = np.stack(query_correct_rows)
    metrics = {
        "query_accuracy": float(query_correct.mean()),
        "all_queries_exact_match": float(query_correct.all(axis=1).mean()),
    }
    for distance in sorted(np.unique(hop_distances)):
        distance_mask = hop_distances == distance
        metrics[f"accuracy_hop_distance_{int(distance)}"] = float(
            query_correct[distance_mask].mean()
        )

    for name, values in inner_loop_stats.items():
        if name.startswith("_"):
            continue
        metrics[name] = reduce_inner_loop_stat(name, np.asarray(values))
    return metrics


def progression_metrics_for_json(metrics: Dict[str, float]) -> Dict[str, object]:
    """Strip Trainer's prefix and replace non-finite values with JSON null."""

    prefix = "progression_"
    serialized = {}
    for name, value in metrics.items():
        output_name = name[len(prefix) :] if name.startswith(prefix) else name
        numeric_value = float(value)
        serialized[output_name] = numeric_value if math.isfinite(numeric_value) else None
    return serialized


def save_progression_metrics(output_dir: Path, metrics: Dict[str, float]) -> Path:
    """Write the validation metrics for the exact model used for progression."""

    metrics_path = output_dir / PROGRESSION_METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(progression_metrics_for_json(metrics), metrics_file, indent=2)
        metrics_file.write("\n")
    return metrics_path


def load_zoology_dataset(args: ZoologyExperimentArgs):
    if args.data_path is not None:
        return datasets.load_from_disk(args.data_path)
    return datasets.load_dataset(args.hf_dataset, args.hf_subset)


def split_dataset(dataset):
    train = dataset["train"]
    if "validation" in dataset:
        validation = dataset["validation"]
    elif "valid" in dataset:
        validation = dataset["valid"]
    else:
        raise ValueError(f"Dataset has no validation split: {list(dataset)}")
    if "test" not in dataset:
        raise ValueError(f"Dataset has no held-out test split: {list(dataset)}")
    return train, validation, dataset["test"]


def main() -> None:
    args = HfArgumentParser(ZoologyExperimentArgs).parse_args_into_dataclasses()[0]
    validate_zoology_args(args)
    transformers.set_seed(args.seed)

    accelerator = accelerate.Accelerator()
    logger.info("num processes: %d", accelerator.num_processes)
    logger.info("mixed precision: %s", accelerator.mixed_precision)
    logger.info("accelerator state: %s", accelerator.state)

    output_dir = Path(args.exp_path)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
            json.dump({"cli_args": dict(vars(args))}, config_file, indent=4)

    base_config = build_zoology_base_config(args)
    model = EnergyGradMem(build_model_config(args, base_config))
    if args.init_checkpoint is not None:
        missing_keys, unexpected_keys = model.load_state_dict(
            load_file(args.init_checkpoint),
            strict=False,
        )
        logger.info("missing checkpoint keys: %s", missing_keys)
        logger.info("unexpected checkpoint keys: %s", unexpected_keys)
    if accelerator.mixed_precision == "bf16":
        model.to(torch.bfloat16)
    model.to(accelerator.device)

    logger.info("model config: %s", model.config)
    dataset = load_zoology_dataset(args)
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)

    def data_collator(batch):
        return collate_zoology_batch(
            batch,
            n_pairs=args.n_pairs,
            kv_pairs_per_segment=args.kv_pairs_per_segment,
        )

    if args.total_batch_size is None:
        args.total_batch_size = (
            args.per_device_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
    else:
        actual_batch_size = (
            args.per_device_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        if args.total_batch_size != actual_batch_size:
            raise ValueError(
                f"total_batch_size={args.total_batch_size} but effective batch size is "
                f"{actual_batch_size}"
            )

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
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=os.environ.get("WANDB_NAME", output_dir.name),
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_for_metrics=["inputs"],
        save_total_limit=1,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=args.seed,
    )
    trainer = ZoologyTrainer(
        model=model,
        args=training_args,
        outer_lr_final_ratio=args.outer_lr_final_ratio,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_zoology_metrics,
        preprocess_logits_for_metrics=preprocess_zoology_logits,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            ),
            StopOnMetricValue(
                metric_name="all_queries_exact_match",
                value=args.stop_all_queries_exact_match_value,
                higher_is_better=True,
            ),
            EnergyFreezeCallback(args.energy_freezed_steps),
        ],
    )
    trainer.train()
    # Selection callbacks apply only to validation evaluations during training.
    # Removing them prevents progression/test metrics from being mistaken for a
    # checkpoint-selection event or altering early-stopping state.
    trainer.remove_callback(EarlyStoppingCallback)
    trainer.remove_callback(StopOnMetricValue)
    # With load_best_model_at_end=True this is the selected best model. Saving a
    # stable stage-level copy also covers the case where eval-on-start triggers
    # early stopping before Trainer creates a numbered checkpoint.
    trainer.save_model(output_dir / "progression_checkpoint")

    logger.info("evaluating the progression checkpoint on validation")
    progression_metrics = trainer.evaluate(
        validation_dataset,
        metric_key_prefix="progression",
    )
    if accelerator.is_main_process:
        metrics_path = save_progression_metrics(output_dir, progression_metrics)
        logger.info("saved progression metrics to %s", metrics_path)

    logger.info("running held-out test evaluation with the progression model")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    logger.info("%s", test_metrics)
    trainer.save_metrics(split="test", metrics=test_metrics)
    trainer.state.save_to_json(output_dir / "trainer_state.json")


if __name__ == "__main__":
    main()
