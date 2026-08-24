#!/usr/bin/env python3
"""Generic frozen-checkpoint evaluation for memory-writing models.

The evaluator deliberately does not use ``Trainer``: it reconstructs each model
from the config saved beside the checkpoint, disables auxiliary outer losses,
and evaluates fixed checkpoints without mutating their parameters. Expensive
measurements are cached by metric family under each checkpoint directory;
comparison outputs only materialize aliases, aggregates, reports, and plots.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from grad_memgpt import GradMemGPT, GradMemGPTConfig
from kv_dataset_utils import query_target_spans


EXPECTED_MISSING_CHECKPOINT_KEYS = {
    "model.lm_head.weight",
    "energy_memory_search_gain_ema",
    "energy_memory_search_gain_ema_initialized",
}
SAFE_ALIAS_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
RUN_DIR_RE = re.compile(r"^run_(\d+)$")
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
EVALUATOR_SCHEMA_VERSION = 2
METRIC_CACHE_SCHEMA_VERSION = 1
METRIC_CACHE_DIRNAME = "energy_shaping_metrics"
METRIC_FAMILY_VERSIONS = {
    # Bump only the affected family when its measurements or stored fields
    # change. Other families will continue to reuse their checkpoint cache.
    "task": 1,
    "matching": 1,
    "interpolation": 1,
    "radial": 1,
    "contours": 1,
}
CACHE_MODEL_LABEL = "checkpoint"


@dataclass(frozen=True)
class ExperimentInput:
    alias: str
    path: Path
    checkpoint_selector: str


@dataclass(frozen=True)
class ResolvedRun:
    alias: str
    source_path: Path
    run_path: Path
    run_id: str
    seed: int
    checkpoint_selector: str
    checkpoint_path: Path
    checkpoint_name: str
    is_recorded_best: bool


@dataclass(frozen=True)
class SkippedRun:
    alias: str
    run_path: Path
    reason: str


@dataclass
class FrozenModel:
    label: str
    run_path: Path
    checkpoint_path: Path
    model: GradMemGPT
    tokenizer: Any
    digest_before: str
    device: torch.device
    objective_type: str
    expected_training_exact_match: float | None


@dataclass(frozen=True)
class ReportCheckpoint:
    label: str
    checkpoint_path: Path


@dataclass
class LandscapeBatch:
    indices: np.ndarray
    context_input_ids: torch.Tensor
    query_input_ids: torch.Tensor
    initial_mem: torch.Tensor
    positive_mem: torch.Tensor
    initial_objective: torch.Tensor
    positive_objective: torch.Tensor


def parse_int_list(values: Sequence[str | int]) -> list[int]:
    result: list[int] = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                result.append(int(piece))
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _coerce_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def sample_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def aggregate_numeric_rows(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    output = []
    metadata_keys = {
        "model", "alias", "run_id", "seed", "checkpoint", "checkpoint_path",
        "reference_model", "candidate_model", "comparison",
    }
    for group, members in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = {key: value for key, value in zip(group_keys, group)}
        seeds = {member.get("seed") for member in members if member.get("seed") is not None}
        result["run_count"] = len(seeds) if seeds else len(members)
        candidate_keys = list(dict.fromkeys(key for member in members for key in member))
        for key in candidate_keys:
            if key in group_keys or key in metadata_keys:
                continue
            values = [member.get(key) for member in members]
            numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if len(numeric) != len(values) or not numeric:
                continue
            result[f"{key}_mean"] = float(np.mean(numeric))
            result[f"{key}_std"] = sample_std(numeric)
        output.append(result)
    return output


def add_run_metadata(rows: Sequence[dict[str, Any]], spec: ResolvedRun) -> None:
    for row in rows:
        row.update({
            "alias": spec.alias,
            "run_id": spec.run_id,
            "seed": spec.seed,
            "checkpoint": spec.checkpoint_name,
            "checkpoint_path": str(spec.checkpoint_path),
        })


def load_run_cli_args(run_path: Path) -> dict[str, Any]:
    config_path = run_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    config = json.loads(config_path.read_text())
    cli_args = config.get("cli_args")
    if not isinstance(cli_args, dict):
        raise ValueError(f"Invalid run config, expected cli_args mapping: {config_path}")
    return cli_args


def resolve_checkpoint(run_path: Path, checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_path / checkpoint
    if checkpoint.is_file():
        if checkpoint.name != "model.safetensors":
            raise ValueError(f"Expected model.safetensors, got: {checkpoint}")
        return checkpoint
    model_path = checkpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {model_path}")
    return model_path


def checkpoint_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_run_sort_key(path: Path) -> tuple[int, str]:
    match = RUN_DIR_RE.fullmatch(path.name)
    return (int(match.group(1)), path.name) if match else (2**31 - 1, path.name)


def discover_run_directories(path: Path) -> tuple[list[Path], bool]:
    path = path.resolve()
    if (path / "config.json").is_file():
        return [path], True
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment path does not exist: {path}")
    runs = sorted(
        [candidate for candidate in path.iterdir() if candidate.is_dir() and RUN_DIR_RE.fullmatch(candidate.name)],
        key=numeric_run_sort_key,
    )
    if not runs:
        raise ValueError(f"Expected a run directory or experiment folder containing run_* directories: {path}")
    return runs, False


def load_trainer_state(run_path: Path) -> dict[str, Any] | None:
    state_path = run_path / "trainer_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    if not isinstance(state, dict):
        raise ValueError(f"Invalid trainer state: {state_path}")
    return state


def recorded_best_checkpoint(run_path: Path) -> Path | None:
    state = load_trainer_state(run_path)
    if state is None or not state.get("best_model_checkpoint"):
        checkpoint_states = []
        for checkpoint_dir in run_path.iterdir():
            match = CHECKPOINT_DIR_RE.fullmatch(checkpoint_dir.name) if checkpoint_dir.is_dir() else None
            state_path = checkpoint_dir / "trainer_state.json"
            if match and state_path.is_file():
                checkpoint_states.append((int(match.group(1)), state_path))
        for _, state_path in sorted(checkpoint_states, reverse=True):
            candidate_state = json.loads(state_path.read_text())
            if candidate_state.get("best_model_checkpoint"):
                state = candidate_state
                break
    if state is None or not state.get("best_model_checkpoint"):
        return None
    checkpoint_name = Path(str(state["best_model_checkpoint"])).name
    candidate = run_path / checkpoint_name / "model.safetensors"
    return candidate.resolve() if candidate.exists() else None


def latest_checkpoint(run_path: Path) -> Path | None:
    candidates = []
    for path in run_path.iterdir():
        match = CHECKPOINT_DIR_RE.fullmatch(path.name) if path.is_dir() else None
        if match and (path / "model.safetensors").exists():
            candidates.append((int(match.group(1)), path / "model.safetensors"))
    return max(candidates, default=(None, None))[1]


def is_completed_run(run_path: Path) -> bool:
    return (
        (run_path / "all_results.json").is_file()
        and load_trainer_state(run_path) is not None
        and recorded_best_checkpoint(run_path) is not None
    )


def resolve_checkpoint_selector(run_path: Path, selector: str) -> tuple[Path, bool]:
    best = recorded_best_checkpoint(run_path)
    if selector == "best":
        if best is None:
            raise FileNotFoundError(f"Run has no recorded best checkpoint: {run_path}")
        return best, True
    if selector == "latest":
        latest = latest_checkpoint(run_path)
        if latest is None:
            raise FileNotFoundError(f"Run has no checkpoint-* directories: {run_path}")
        return latest.resolve(), bool(best is not None and latest.resolve() == best.resolve())
    resolved = resolve_checkpoint(run_path, selector).resolve()
    return resolved, bool(best is not None and resolved == best.resolve())


def resolve_experiment_inputs(
    inputs: Sequence[ExperimentInput],
    *,
    include_incomplete: bool,
) -> tuple[list[ResolvedRun], list[SkippedRun]]:
    resolved: list[ResolvedRun] = []
    skipped: list[SkippedRun] = []
    for experiment in inputs:
        run_paths, is_exact_run = discover_run_directories(experiment.path)
        for run_path in run_paths:
            completed = is_completed_run(run_path)
            explicitly_addressed_checkpoint = experiment.checkpoint_selector not in ("best",)
            has_recorded_best = (
                experiment.checkpoint_selector == "best"
                and recorded_best_checkpoint(run_path) is not None
            )
            if (
                not completed
                and not include_incomplete
                and not (is_exact_run and explicitly_addressed_checkpoint)
                and not has_recorded_best
            ):
                skipped.append(SkippedRun(
                    alias=experiment.alias,
                    run_path=run_path,
                    reason="incomplete run: missing final results, root trainer state, or recorded best checkpoint",
                ))
                continue
            try:
                checkpoint_path, is_best = resolve_checkpoint_selector(
                    run_path, experiment.checkpoint_selector
                )
            except (FileNotFoundError, ValueError) as error:
                raise type(error)(
                    f"Could not resolve selector {experiment.checkpoint_selector!r} "
                    f"for {experiment.alias}/{run_path.name}: {error}"
                ) from error
            cli_args = load_run_cli_args(run_path)
            seed = cli_args.get("seed")
            if seed is None:
                raise ValueError(f"Run does not record a seed: {run_path}")
            resolved.append(ResolvedRun(
                alias=experiment.alias,
                source_path=experiment.path.resolve(),
                run_path=run_path.resolve(),
                run_id=run_path.name,
                seed=int(seed),
                checkpoint_selector=experiment.checkpoint_selector,
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_path.parent.name,
                is_recorded_best=is_best,
            ))

    seen: dict[tuple[str, int], ResolvedRun] = {}
    for item in resolved:
        key = (item.alias, item.seed)
        if key in seen:
            raise ValueError(
                f"Alias {item.alias!r} contains duplicate seed {item.seed}: "
                f"{seen[key].run_path} and {item.run_path}"
            )
        seen[key] = item
    return resolved, skipped


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def reconstruct_serialized_base_config(config: GradMemGPTConfig) -> None:
    if not isinstance(config.base_config, dict):
        return
    serialized_base = dict(config.base_config)
    model_type = serialized_base.pop("model_type", None)
    if not model_type:
        raise ValueError("Serialized base_config has no model_type")
    config.base_config = AutoConfig.for_model(model_type, **serialized_base)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def saved_exact_match(run_path: Path) -> float | None:
    results_path = run_path / "all_results.json"
    if not results_path.exists():
        return None
    results = json.loads(results_path.read_text())
    value = results.get("eval_exact_match")
    return None if value is None else float(value)


def training_reference(run_path: Path) -> tuple[int, int, float] | None:
    expected = saved_exact_match(run_path)
    if expected is None:
        return None
    cli_args = load_run_cli_args(run_path)
    data_path = str(cli_args.get("data_path", ""))
    match = re.search(r"(?:^|[/_\-])N(\d+)(?:[-_/]|$)", data_path)
    inner_steps = cli_args.get("K")
    if match is None or inner_steps is None:
        return None
    return int(match.group(1)), int(inner_steps), expected


def load_frozen_model(
    label: str,
    run_path: Path,
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> FrozenModel:
    run_path = run_path.resolve()
    model_path = resolve_checkpoint(run_path, checkpoint).resolve()
    checkpoint_dir = model_path.parent
    cli_args = load_run_cli_args(run_path)

    tokenizer_path = cli_args.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError(f"Run {run_path} does not define tokenizer_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    config = GradMemGPTConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    reconstruct_serialized_base_config(config)
    if config.memory_backend != "prefix":
        raise ValueError(
            "Standalone shaping probes currently require prefix memory; "
            f"checkpoint uses {config.memory_backend!r}"
        )
    if config.write_objective not in ("reconstruction", "energy", "energy_with_reconstruction"):
        raise ValueError(f"Unsupported write objective: {config.write_objective}")

    # Auxiliary outer losses must not affect task-loss reporting. The energy
    # head itself remains loaded and is probed separately with shared settings.
    config.energy_rank_weight = 0.0
    config.energy_traj_weight = 0.0
    config.energy_anchor_weight = 0.0
    config.lipschitz_weight = 0.0
    config.energy_memory_search_weight = 0.0
    config.ivan_loss_weight = 0.0
    config.add_inner_loss_to_outer = False
    config.step_alignment_weight = 0.0
    config.last_K_second_order = 0

    model = GradMemGPT(config)
    state_dict = load_file(str(model_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_set = set(missing)
    unexpected_set = set(unexpected)
    if unexpected_set:
        raise ValueError(f"Unexpected checkpoint keys in {model_path}: {sorted(unexpected_set)}")
    if not missing_set.issubset(EXPECTED_MISSING_CHECKPOINT_KEYS):
        raise ValueError(f"Unexpected missing checkpoint keys in {model_path}: {sorted(missing_set)}")
    model.tie_weights()
    resolved_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    model.to(device=resolved_device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return FrozenModel(
        label=label,
        run_path=run_path,
        checkpoint_path=model_path,
        model=model,
        tokenizer=tokenizer,
        digest_before=parameter_digest(model),
        device=resolved_device,
        objective_type=(
            "reconstruction_cross_entropy"
            if config.write_objective == "reconstruction"
            else "learned_energy"
        ),
        expected_training_exact_match=saved_exact_match(run_path),
    )


def collate_kv_batch(batch: Sequence[Mapping[str, str]], tokenizer: Any) -> dict[str, Any]:
    contexts = [item["context"] for item in batch]
    query_and_target = [item["query"] + item["target"] for item in batch]

    context_input_ids = tokenizer(
        contexts,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        pad_to_multiple_of=8,
    ).input_ids
    query_encoded = tokenizer(
        query_and_target,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        pad_to_multiple_of=8,
        return_offsets_mapping=True,
    )
    query_input_ids = query_encoded["input_ids"]
    offsets_mapping = query_encoded["offset_mapping"]

    labels_mask = torch.zeros_like(query_input_ids)
    for row, item in enumerate(batch):
        target_spans = query_target_spans(item["query"], item["target"])
        for column in range(len(offsets_mapping[row])):
            start, end = offsets_mapping[row][column]
            if any(start < target_end and end > target_start
                   for target_start, target_end in target_spans):
                labels_mask[row, column] = 1

    labels = query_input_ids * labels_mask + (1 - labels_mask) * -100
    return {
        "input_ids": {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        "labels": labels,
    }


def iter_dataset_batches(
    dataset: Any,
    tokenizer: Any,
    batch_size: int,
    indices: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    if indices is None:
        indices = list(range(len(dataset)))
    for start in range(0, len(indices), batch_size):
        batch_indices = np.asarray(indices[start:start + batch_size], dtype=np.int64)
        examples = [dataset[int(index)] for index in batch_indices]
        yield batch_indices, collate_kv_batch(examples, tokenizer)


def move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    return value


def set_inner_steps(model: GradMemGPT, inner_steps: int) -> None:
    if inner_steps < 0:
        raise ValueError(f"inner_steps must be non-negative, got {inner_steps}")
    model.K = int(inner_steps)
    model.last_K_second_order = 0
    model.config.K = int(inner_steps)
    model.config.last_K_second_order = 0


def initial_memory(model: GradMemGPT, batch_size: int) -> torch.Tensor:
    state, _ = model.memory_backend_impl.init_memory_state(batch_size)
    return state["mem_batch"].detach()


def objective_for_aligned_memories(
    model: GradMemGPT,
    context_input_ids: torch.Tensor,
    query_input_ids: torch.Tensor,
    mem_batch: torch.Tensor,
) -> torch.Tensor:
    if context_input_ids.size(0) != mem_batch.size(0):
        raise ValueError("Context and candidate-memory batch sizes must match")
    device = next(model.parameters()).device
    context_input_ids = context_input_ids.to(device)
    query_input_ids = query_input_ids.to(device)
    mem_batch = mem_batch.to(device)
    backend = model.memory_backend_impl
    pad_id = model.model.config.pad_token_id
    batch_ctx = backend.prepare_batch(context_input_ids, query_input_ids, pad_id)
    template, _ = backend.init_memory_state(context_input_ids.size(0))
    with torch.no_grad():
        if model.write_objective == "reconstruction":
            candidate_state = dict(template)
            candidate_state["mem_batch"] = mem_batch.detach()
            write_batch = backend.build_write_inputs(candidate_state, batch_ctx)
            with backend.activation_context(candidate_state):
                outs, objective = model._run_reconstruction_write_forward(write_batch)
            del outs
            return objective.detach()
        return model._run_energy_memory_candidate(backend, template, batch_ctx, mem_batch).detach()


def objective_for_aligned_memories_batched(
    model: GradMemGPT,
    context_input_ids: torch.Tensor,
    query_input_ids: torch.Tensor,
    mem_batch: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    values = []
    for start in range(0, context_input_ids.size(0), batch_size):
        end = start + batch_size
        values.append(
            objective_for_aligned_memories(
                model,
                context_input_ids[start:end],
                query_input_ids[start:end],
                mem_batch[start:end],
            ).cpu()
        )
    return torch.cat(values, dim=0)


# Compatibility aliases for users of the original two-energy-model evaluator.
energy_for_aligned_memories = objective_for_aligned_memories
energy_for_aligned_memories_batched = objective_for_aligned_memories_batched


def per_example_task_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_token_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    if logits.size(1) != labels.size(1) + 1:
        raise ValueError(
            "Prefix-memory evaluation expects prediction length to be label length + 1; "
            f"got logits={logits.size(1)}, labels={labels.size(1)}"
        )
    aligned_logits = logits[:, :-1]
    predictions = aligned_logits.argmax(dim=-1)

    content_mask = labels.ne(-100)
    for token_id in ignore_token_ids:
        content_mask &= labels.ne(int(token_id))
    correct = predictions.eq(labels) & content_mask
    content_count = content_mask.sum(dim=1)
    if torch.any(content_count == 0):
        raise ValueError("Encountered validation example with no scored target tokens")

    exact = (correct.sum(dim=1) == content_count).to(torch.float32)
    token_correct = correct.sum(dim=1).to(torch.float32)

    flat_loss = F.cross_entropy(
        aligned_logits.reshape(-1, aligned_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    loss_mask = labels.ne(-100)
    loss_count = loss_mask.sum(dim=1).clamp_min(1)
    example_loss = (flat_loss * loss_mask).sum(dim=1) / loss_count

    return {
        "exact_match": exact.detach().cpu().numpy(),
        "token_correct": token_correct.detach().cpu().numpy(),
        "token_count": content_count.detach().cpu().numpy(),
        "target_loss": example_loss.detach().cpu().numpy(),
    }


def evaluate_task_cell(
    frozen: FrozenModel,
    dataset: Any,
    n_value: int,
    inner_steps: int,
    batch_size: int,
    max_examples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = frozen.model
    set_inner_steps(model, inner_steps)
    ignore_ids = [frozen.tokenizer.convert_tokens_to_ids(token) for token in ("!", "|")]
    n_examples = len(dataset) if max_examples is None else min(len(dataset), max_examples)

    rows: list[dict[str, Any]] = []
    weighted_grad_norm = 0.0
    started = time.perf_counter()
    for indices, batch in iter_dataset_batches(
        dataset, frozen.tokenizer, batch_size, indices=list(range(n_examples))
    ):
        batch = move_batch_to_device(batch, frozen.device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        initial_mem = initial_memory(model, len(indices))
        with torch.no_grad():
            output = model(input_ids, labels=labels, return_mem=True)
        logits = output["predictions"].detach()
        final_mem = output["mem"].detach()
        task = per_example_task_metrics(logits, labels, ignore_ids)

        objective_initial = objective_for_aligned_memories(
            model, input_ids["context_input_ids"], input_ids["query_input_ids"], initial_mem
        ).cpu()
        objective_final = objective_for_aligned_memories(
            model, input_ids["context_input_ids"], input_ids["query_input_ids"], final_mem
        ).cpu()
        mem_norm = final_mem.reshape(len(indices), -1).norm(dim=1).cpu()
        mem_displacement = (final_mem - initial_mem).reshape(len(indices), -1).norm(dim=1).cpu()

        stats = output.get("inner_loop_stats", {})
        grad_norm = float(stats.get("inner_grad_norm_mean", torch.tensor(0.0)).detach().cpu())
        if inner_steps == 0:
            grad_norm = 0.0
        weighted_grad_norm += grad_norm * len(indices)

        for offset, example_index in enumerate(indices):
            rows.append({
                "model": frozen.label,
                "N": int(n_value),
                "K": int(inner_steps),
                "example_index": int(example_index),
                "exact_match": float(task["exact_match"][offset]),
                "token_correct": float(task["token_correct"][offset]),
                "token_count": int(task["token_count"][offset]),
                "target_loss": float(task["target_loss"][offset]),
                "objective_type": frozen.objective_type,
                "objective_initial": float(objective_initial[offset]),
                "objective_final": float(objective_final[offset]),
                "objective_decrease": float(objective_initial[offset] - objective_final[offset]),
                "mem_norm": float(mem_norm[offset]),
                "mem_displacement": float(mem_displacement[offset]),
            })

    runtime = time.perf_counter() - started
    exact_values = np.asarray([row["exact_match"] for row in rows])
    token_correct = sum(row["token_correct"] for row in rows)
    token_count = sum(row["token_count"] for row in rows)
    summary = {
        "model": frozen.label,
        "N": int(n_value),
        "K": int(inner_steps),
        "num_examples": len(rows),
        "exact_match": float(exact_values.mean()),
        "token_accuracy": float(token_correct / token_count),
        "target_loss": float(np.mean([row["target_loss"] for row in rows])),
        "objective_type": frozen.objective_type,
        "objective_initial": float(np.mean([row["objective_initial"] for row in rows])),
        "objective_final": float(np.mean([row["objective_final"] for row in rows])),
        "objective_decrease": float(np.mean([row["objective_decrease"] for row in rows])),
        "inner_grad_norm": float(weighted_grad_norm / len(rows)),
        "mem_norm": float(np.mean([row["mem_norm"] for row in rows])),
        "mem_displacement": float(np.mean([row["mem_displacement"] for row in rows])),
        "runtime_seconds": float(runtime),
    }
    return rows, summary


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    n_resamples: int,
    seed: int,
    statistic: str = "mean",
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap_interval expects a non-empty one-dimensional array")
    if n_resamples <= 0:
        point = float(np.mean(array) if statistic == "mean" else np.median(array))
        return point, point
    if statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")

    rng = np.random.default_rng(seed)
    results = np.empty(n_resamples, dtype=np.float64)
    chunk_size = min(256, n_resamples)
    for start in range(0, n_resamples, chunk_size):
        count = min(chunk_size, n_resamples - start)
        indices = rng.integers(0, array.size, size=(count, array.size))
        samples = array[indices]
        if statistic == "mean":
            results[start:start + count] = samples.mean(axis=1)
        else:
            results[start:start + count] = np.median(samples, axis=1)
    low, high = np.quantile(results, [0.025, 0.975])
    return float(low), float(high)


def paired_task_summaries(
    task_rows: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    seed: int,
    comparisons: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        key = (int(row["N"]), int(row["K"]), int(row["example_index"]))
        grouped[key][str(row["model"])] = row

    labels = list(dict.fromkeys(str(row["model"]) for row in task_rows))
    if comparisons is None:
        comparisons = [
            (labels[left], labels[right])
            for left in range(len(labels))
            for right in range(left + 1, len(labels))
        ]

    summaries = []
    cells = sorted({(int(row["N"]), int(row["K"])) for row in task_rows})
    for comparison_index, (reference_label, candidate_label) in enumerate(comparisons):
        for n_value, inner_steps in cells:
            pairs = []
            for (row_n, row_k, _), by_model in grouped.items():
                if row_n != n_value or row_k != inner_steps:
                    continue
                if reference_label not in by_model or candidate_label not in by_model:
                    raise ValueError(
                        f"Missing paired result for {reference_label}->{candidate_label}, "
                        f"N={n_value}, K={inner_steps}"
                    )
                pairs.append((by_model[reference_label], by_model[candidate_label]))
            differences = np.asarray([
                float(candidate["exact_match"]) - float(reference["exact_match"])
                for reference, candidate in pairs
            ])
            low, high = bootstrap_interval(
                differences,
                n_resamples=bootstrap_resamples,
                seed=seed + n_value * 1009 + inner_steps * 9176 + comparison_index * 104729,
            )
            row = {
                "reference_model": reference_label,
                "candidate_model": candidate_label,
                "comparison": f"{candidate_label} - {reference_label}",
                "N": n_value,
                "K": inner_steps,
                "num_examples": len(pairs),
                "reference_exact_match": float(np.mean([float(pair[0]["exact_match"]) for pair in pairs])),
                "candidate_exact_match": float(np.mean([float(pair[1]["exact_match"]) for pair in pairs])),
                "exact_match_difference": float(differences.mean()),
                "exact_match_difference_ci_low": low,
                "exact_match_difference_ci_high": high,
            }
            summaries.append(row)
    return summaries


def paired_task_examples(
    task_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        key = (int(row["N"]), int(row["K"]), int(row["example_index"]))
        grouped[key][str(row["model"])] = row
    labels = list(dict.fromkeys(str(row["model"]) for row in task_rows))
    if comparisons is None:
        comparisons = [
            (labels[left], labels[right])
            for left in range(len(labels))
            for right in range(left + 1, len(labels))
        ]
    output = []
    for (n_value, inner_steps, example_index), by_model in sorted(grouped.items()):
        for reference_label, candidate_label in comparisons:
            if reference_label not in by_model or candidate_label not in by_model:
                raise ValueError(
                    f"Missing paired task row for {reference_label}->{candidate_label}, "
                    f"N={n_value}, K={inner_steps}, index={example_index}"
                )
            reference = by_model[reference_label]
            candidate = by_model[candidate_label]
            reference_token_accuracy = float(reference["token_correct"]) / float(reference["token_count"])
            candidate_token_accuracy = float(candidate["token_correct"]) / float(candidate["token_count"])
            row = {
                "reference_model": reference_label,
                "candidate_model": candidate_label,
                "comparison": f"{candidate_label} - {reference_label}",
                "N": n_value,
                "K": inner_steps,
                "example_index": example_index,
                "reference_exact_match": float(reference["exact_match"]),
                "candidate_exact_match": float(candidate["exact_match"]),
                "exact_match_difference": float(candidate["exact_match"]) - float(reference["exact_match"]),
                "reference_token_accuracy": reference_token_accuracy,
                "candidate_token_accuracy": candidate_token_accuracy,
                "token_accuracy_difference": candidate_token_accuracy - reference_token_accuracy,
            }
            for metric in (
                "target_loss",
                "objective_initial",
                "objective_final",
                "objective_decrease",
                "mem_norm",
                "mem_displacement",
            ):
                row[f"reference_{metric}"] = float(reference[metric])
                row[f"candidate_{metric}"] = float(candidate[metric])
                row[f"{metric}_difference"] = float(candidate[metric]) - float(reference[metric])
            output.append(row)
    return output


def deterministic_subset_indices(dataset_size: int, count: int, seed: int) -> list[int]:
    count = min(dataset_size, count)
    rng = np.random.default_rng(seed)
    return rng.permutation(dataset_size)[:count].astype(np.int64).tolist()


def collect_landscape_batches(
    frozen: FrozenModel,
    dataset: Any,
    indices: Sequence[int],
    batch_size: int,
    inner_steps: int = 2,
) -> list[LandscapeBatch]:
    model = frozen.model
    set_inner_steps(model, inner_steps)
    collected = []
    for batch_indices, batch in iter_dataset_batches(dataset, frozen.tokenizer, batch_size, indices=indices):
        batch = move_batch_to_device(batch, frozen.device)
        input_ids = batch["input_ids"]
        initial_mem = initial_memory(model, len(batch_indices))
        with torch.no_grad():
            output = model(input_ids, labels=None, return_mem=True)
        positive_mem = output["mem"].detach()
        initial_objective = objective_for_aligned_memories(
            model, input_ids["context_input_ids"], input_ids["query_input_ids"], initial_mem
        )
        positive_objective = objective_for_aligned_memories(
            model, input_ids["context_input_ids"], input_ids["query_input_ids"], positive_mem
        )
        collected.append(LandscapeBatch(
            indices=batch_indices,
            context_input_ids=input_ids["context_input_ids"].cpu(),
            query_input_ids=input_ids["query_input_ids"].cpu(),
            initial_mem=initial_mem.cpu(),
            positive_mem=positive_mem.cpu(),
            initial_objective=initial_objective.cpu(),
            positive_objective=positive_objective.cpu(),
        ))
    return collected


def matching_probe(
    frozen: FrozenModel,
    batches: Sequence[LandscapeBatch],
    n_value: int,
    margin: float,
    eval_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    top1_values = []
    reciprocal_ranks = []
    auc_values = []
    all_margins = []
    rows = []

    for bank_index, bank in enumerate(batches):
        bank_size = bank.positive_mem.size(0)
        context = bank.context_input_ids.repeat_interleave(bank_size, dim=0)
        query = bank.query_input_ids.repeat_interleave(bank_size, dim=0)
        candidates = bank.positive_mem.repeat(bank_size, 1, 1)
        objectives = objective_for_aligned_memories_batched(
            frozen.model, context, query, candidates, eval_batch_size
        ).reshape(bank_size, bank_size).numpy()
        diagonal = np.diag(objectives)
        if not np.allclose(diagonal, bank.positive_objective.numpy(), atol=1e-5, rtol=1e-5):
            raise AssertionError("Matching-matrix diagonal does not equal direct positive objective")

        matrix_metrics = matching_metrics_from_energy_matrix(objectives, margin)
        top1_values.extend(matrix_metrics["top1_values"])
        reciprocal_ranks.extend(matrix_metrics["reciprocal_ranks"])
        auc_values.extend(matrix_metrics["auc_values"])
        all_margins.extend(matrix_metrics["margins"].reshape(-1).tolist())
        for row_index, row_metric in enumerate(matrix_metrics["row_metrics"]):
            rows.append({
                "model": frozen.label,
                "N": n_value,
                "bank_index": bank_index,
                "example_index": int(bank.indices[row_index]),
                **row_metric,
            })

    margins_array = np.asarray(all_margins)
    summary = {
        "model": frozen.label,
        "N": n_value,
        "num_examples": len(rows),
        "objective_type": frozen.objective_type,
        "bank_size": batches[0].positive_mem.size(0) if batches else 0,
        "top1_accuracy": float(np.mean(top1_values)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "pairwise_auc": float(np.mean(auc_values)),
        "mean_margin": float(margins_array.mean()),
        "median_margin": float(np.median(margins_array)),
        "margin_p05": float(np.quantile(margins_array, 0.05)),
        "margin_satisfaction": float(np.mean(margins_array >= margin)),
    }
    return summary, rows


def matching_metrics_from_energy_matrix(energies: np.ndarray, margin: float) -> dict[str, Any]:
    objectives = np.asarray(energies, dtype=np.float64)
    if objectives.ndim != 2 or objectives.shape[0] != objectives.shape[1] or objectives.shape[0] <= 1:
        raise ValueError("Matching objective matrix must be square with size greater than one")
    diagonal = np.diag(objectives)
    top1_values = []
    reciprocal_ranks = []
    auc_values = []
    margin_rows = []
    row_metrics = []
    for row_index, positive in enumerate(diagonal):
        negatives = np.delete(objectives[row_index], row_index)
        margins = negatives - positive
        # Lower energy is better. Ties are pessimistically placed after the
        # positive for top-1/rank, while AUC gives ties half credit.
        rank = 1 + int(np.sum(negatives <= positive))
        top1_values.append(float(rank == 1))
        reciprocal_ranks.append(1.0 / rank)
        auc = float(np.mean(negatives > positive) + 0.5 * np.mean(negatives == positive))
        auc_values.append(auc)
        margin_rows.append(margins)
        row_metrics.append({
            "positive_objective": float(positive),
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "pairwise_auc": auc,
            "mean_margin": float(margins.mean()),
            "minimum_margin": float(margins.min()),
            "margin_satisfaction": float(np.mean(margins >= margin)),
        })
    return {
        "top1_values": top1_values,
        "reciprocal_ranks": reciprocal_ranks,
        "auc_values": auc_values,
        "margins": np.stack(margin_rows),
        "row_metrics": row_metrics,
    }


def concatenate_landscape_batches(
    batches: Sequence[LandscapeBatch],
    limit: int,
) -> LandscapeBatch:
    if not batches:
        raise ValueError("No landscape batches to concatenate")
    count = min(limit, sum(len(batch.indices) for batch in batches))

    def concatenate(field: str) -> torch.Tensor:
        return torch.cat([getattr(batch, field) for batch in batches], dim=0)[:count]

    return LandscapeBatch(
        indices=np.concatenate([batch.indices for batch in batches])[:count],
        context_input_ids=concatenate("context_input_ids"),
        query_input_ids=concatenate("query_input_ids"),
        initial_mem=concatenate("initial_mem"),
        positive_mem=concatenate("positive_mem"),
        initial_objective=concatenate("initial_objective"),
        positive_objective=concatenate("positive_objective"),
    )


def interpolation_probe(
    frozen: FrozenModel,
    data: LandscapeBatch,
    n_value: int,
    t_values: Sequence[float],
    eval_batch_size: int,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mismatched = torch.roll(data.positive_mem, shifts=1, dims=0)
    objective_columns = []
    rows = []
    for t_index, t_value in enumerate(t_values):
        candidate = (1.0 - t_value) * data.positive_mem + t_value * mismatched
        objective = objective_for_aligned_memories_batched(
            frozen.model,
            data.context_input_ids,
            data.query_input_ids,
            candidate,
            eval_batch_size,
        ).numpy()
        if t_index == 0 and not np.allclose(objective, data.positive_objective.numpy(), atol=1e-5, rtol=1e-5):
            raise AssertionError("Interpolation t=0 does not equal positive-memory objective")
        if t_index == len(t_values) - 1 and math.isclose(float(t_value), 1.0):
            direct_mismatched = objective_for_aligned_memories_batched(
                frozen.model,
                data.context_input_ids,
                data.query_input_ids,
                mismatched,
                eval_batch_size,
            ).numpy()
            if not np.allclose(objective, direct_mismatched, atol=1e-5, rtol=1e-5):
                raise AssertionError("Interpolation t=1 does not equal mismatched-memory objective")
        delta = objective - data.positive_objective.numpy()
        low, high = bootstrap_interval(
            delta,
            n_resamples=bootstrap_resamples,
            seed=seed + n_value * 1009 + t_index,
            statistic="median",
        )
        objective_columns.append(objective)
        rows.append({
            "model": frozen.label,
            "N": n_value,
            "t": float(t_value),
            "objective_type": frozen.objective_type,
            "mean_delta_objective": float(delta.mean()),
            "median_delta_objective": float(np.median(delta)),
            "median_delta_ci_low": low,
            "median_delta_ci_high": high,
            "fraction_above_positive": float(np.mean(delta > 0.0)),
        })

    objectives = np.stack(objective_columns, axis=1)
    monotonic = np.all(np.diff(objectives, axis=1) >= -1e-7, axis=1)
    summary = {
        "model": frozen.label,
        "N": n_value,
        "objective_type": frozen.objective_type,
        "num_examples": objectives.shape[0],
        "monotonic_rise_fraction": float(monotonic.mean()),
        "endpoint_mean_delta": float((objectives[:, -1] - objectives[:, 0]).mean()),
    }
    return rows, summary


def seeded_orthonormal_directions(
    num_samples: int,
    num_directions: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(num_samples, num_directions, dimension)).astype(np.float64)
    result = np.empty_like(raw)
    for direction_index in range(num_directions):
        vector = raw[:, direction_index]
        for previous_index in range(direction_index):
            previous = result[:, previous_index]
            projection = np.sum(vector * previous, axis=1, keepdims=True)
            vector = vector - projection * previous
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        if np.any(norm <= np.finfo(np.float64).eps):
            raise RuntimeError("Failed to construct deterministic orthogonal directions")
        result[:, direction_index] = vector / norm
    return result.astype(np.float32)


def radial_probe(
    frozen: FrozenModel,
    data: LandscapeBatch,
    n_value: int,
    radii: Sequence[float],
    num_directions: int,
    eval_batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_count = data.positive_mem.size(0)
    flat_dimension = data.positive_mem[0].numel()
    directions = torch.from_numpy(seeded_orthonormal_directions(
        sample_count, num_directions, flat_dimension, seed + n_value * 1009
    )).reshape(sample_count, num_directions, *data.positive_mem.shape[1:])
    write_radius = (data.positive_mem - data.initial_mem).reshape(sample_count, -1).norm(dim=1)
    rows = []

    for radius in radii:
        candidate = (
            data.positive_mem[:, None]
            + float(radius)
            * write_radius[:, None, None, None]
            * directions
        )
        measured_radius = (candidate - data.positive_mem[:, None]).reshape(
            sample_count, num_directions, -1
        ).norm(dim=2)
        expected_radius = float(radius) * write_radius[:, None]
        if not torch.allclose(measured_radius, expected_radius.expand_as(measured_radius), atol=1e-4, rtol=1e-4):
            raise AssertionError("Radial perturbation does not have the requested norm")

        context = data.context_input_ids.repeat_interleave(num_directions, dim=0)
        query = data.query_input_ids.repeat_interleave(num_directions, dim=0)
        objective = objective_for_aligned_memories_batched(
            frozen.model,
            context,
            query,
            candidate.reshape(sample_count * num_directions, *data.positive_mem.shape[1:]),
            eval_batch_size,
        ).reshape(sample_count, num_directions).numpy()
        if not np.isfinite(objective).all():
            raise AssertionError("Radial probe produced non-finite objectives")
        delta = objective - data.positive_objective.numpy()[:, None]
        rows.append({
            "model": frozen.label,
            "N": n_value,
            "radius": float(radius),
            "num_examples": sample_count,
            "num_directions": num_directions,
            "objective_type": frozen.objective_type,
            "mean_delta_objective": float(delta.mean()),
            "median_delta_objective": float(np.median(delta)),
            "fraction_objective_rises": float(np.mean(delta > 0.0)),
        })
    return rows


def contour_probe(
    frozen: FrozenModel,
    data: LandscapeBatch,
    n_value: int,
    x_values: Sequence[float],
    y_values: Sequence[float],
    eval_batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    positive = data.positive_mem
    mismatched = torch.roll(positive, shifts=1, dims=0)
    direction_u = (mismatched - positive).reshape(positive.size(0), -1)
    scale = direction_u.norm(dim=1).clamp_min(torch.finfo(positive.dtype).eps)
    direction_u = direction_u / scale[:, None]

    random_v = torch.from_numpy(seeded_orthonormal_directions(
        positive.size(0), 1, direction_u.size(1), seed
    )[:, 0])
    random_v = random_v - (random_v * direction_u).sum(dim=1, keepdim=True) * direction_u
    random_v = random_v / random_v.norm(dim=1, keepdim=True).clamp_min(torch.finfo(positive.dtype).eps)
    direction_u = direction_u.reshape_as(positive)
    random_v = random_v.reshape_as(positive)

    rows = []
    for x_value in x_values:
        for y_value in y_values:
            candidate = (
                positive
                + scale[:, None, None]
                * (float(x_value) * direction_u + float(y_value) * random_v)
            )
            objective = objective_for_aligned_memories_batched(
                frozen.model,
                data.context_input_ids,
                data.query_input_ids,
                candidate,
                eval_batch_size,
            ).numpy()
            delta = objective - data.positive_objective.numpy()
            rows.append({
                "model": frozen.label,
                "N": int(n_value),
                "x": float(x_value),
                "y": float(y_value),
                "objective_type": frozen.objective_type,
                "mean_delta_objective": float(delta.mean()),
                "median_delta_objective": float(np.median(delta)),
            })
    return rows


def _svg_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def write_line_svg(
    path: Path,
    series: Mapping[str, Sequence[tuple[float, float]]],
    title: str,
    x_label: str,
    y_label: str,
    std_series: Mapping[str, Sequence[tuple[float, float, float | None]]] | None = None,
) -> None:
    width, height = 760, 460
    left, right, top, bottom = 82, 30, 48, 70
    all_points = [point for points in series.values() for point in points]
    if std_series:
        for points in std_series.values():
            for x_value, mean, std in points:
                spread = 0.0 if std is None else float(std)
                all_points.extend([(x_value, mean - spread), (x_value, mean + spread)])
    if not all_points:
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if math.isclose(y_min, y_max):
        y_min, y_max = y_min - 0.5, y_max + 0.5
    y_padding = 0.08 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    colors = ["#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214", "#0571b0"]
    dashes = ["", "7 4", "2 3", "10 3 2 3", "5 2", "1 2"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_svg_text(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#222">{_svg_text(title)}</text>',
    ]
    for tick_index in range(6):
        fraction = tick_index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = y_coord(y_value)
        elements.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        elements.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{y_value:.3g}</text>')
    for tick_index in range(6):
        fraction = tick_index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = x_coord(x_value)
        elements.append(f'<text x="{x:.2f}" y="{height - bottom + 20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#444">{x_value:.3g}</text>')
    elements.extend([
        f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#333"/>',
        f'<text x="{(left + width - right) / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">{_svg_text(x_label)}</text>',
        f'<text x="18" y="{(top + height - bottom) / 2}" text-anchor="middle" transform="rotate(-90 18 {(top + height - bottom) / 2})" font-family="sans-serif" font-size="13" fill="#222">{_svg_text(y_label)}</text>',
    ])
    if std_series:
        for index, (label, points) in enumerate(std_series.items()):
            valid = sorted(
                (x_value, mean, std)
                for x_value, mean, std in points
                if std is not None and math.isfinite(float(std))
            )
            if not valid:
                continue
            upper = [(x_coord(x), y_coord(mean + float(std))) for x, mean, std in valid]
            lower = [(x_coord(x), y_coord(mean - float(std))) for x, mean, std in reversed(valid)]
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
            color = colors[index % len(colors)]
            elements.append(f'<polygon points="{coordinates}" fill="{color}" fill-opacity="0.14" stroke="none"/>')
    for index, (label, points) in enumerate(series.items()):
        ordered = sorted(points)
        coordinates = " ".join(f"{x_coord(x):.2f},{y_coord(y):.2f}" for x, y in ordered)
        color = colors[index % len(colors)]
        dash = dashes[index % len(dashes)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        for x, y in ordered:
            elements.append(f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="3" fill="{color}"/>')
        legend_x = left + 10 + (index % 3) * 210
        legend_y = top + 16 + (index // 3) * 20
        elements.append(f'<line x1="{legend_x}" x2="{legend_x + 22}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        elements.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="11" fill="#222">{_svg_text(label)}</text>')
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n")


def write_heatmap_svg(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    title: str,
    color_limit: float | None = None,
    value_key: str = "mean_delta_energy",
) -> None:
    if not rows:
        return
    x_values = sorted({float(row["x"]) for row in rows})
    y_values = sorted({float(row["y"]) for row in rows})
    lookup = {(float(row["x"]), float(row["y"])): float(row[value_key]) for row in rows}
    values = np.asarray(list(lookup.values()))
    limit = color_limit if color_limit is not None else max(abs(float(values.min())), abs(float(values.max())))
    limit = max(float(limit), 1e-12)
    width, height = 620, 540
    left, right, top, bottom = 72, 90, 48, 62
    cell_width = (width - left - right) / len(x_values)
    cell_height = (height - top - bottom) / len(y_values)

    def color(value: float) -> str:
        normalized = max(-1.0, min(1.0, value / limit))
        if normalized >= 0:
            red, green, blue = 178, 24, 43
            amount = normalized
        else:
            red, green, blue = 33, 102, 172
            amount = -normalized
        channels = [round(255 + amount * (channel - 255)) for channel in (red, green, blue)]
        return "#" + "".join(f"{channel:02x}" for channel in channels)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_svg_text(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#222">{_svg_text(title)}</text>',
    ]
    for y_index, y_value in enumerate(reversed(y_values)):
        for x_index, x_value in enumerate(x_values):
            value = lookup[(x_value, y_value)]
            x = left + x_index * cell_width
            y = top + y_index * cell_height
            elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width + 0.2:.2f}" height="{cell_height + 0.2:.2f}" fill="{color(value)}"/>')
    for tick_index in range(0, len(x_values), max(1, len(x_values) // 5)):
        x = left + (tick_index + 0.5) * cell_width
        elements.append(f'<text x="{x:.2f}" y="{height - bottom + 20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#444">{x_values[tick_index]:.2g}</text>')
    reversed_y = list(reversed(y_values))
    for tick_index in range(0, len(reversed_y), max(1, len(reversed_y) // 5)):
        y = top + (tick_index + 0.5) * cell_height
        elements.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{reversed_y[tick_index]:.2g}</text>')
    elements.extend([
        f'<text x="{(left + width - right) / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">mismatch direction</text>',
        f'<text x="18" y="{(top + height - bottom) / 2}" text-anchor="middle" transform="rotate(-90 18 {(top + height - bottom) / 2})" font-family="sans-serif" font-size="13" fill="#222">orthogonal direction</text>',
        f'<text x="{width - right + 14}" y="{top + 10}" font-family="sans-serif" font-size="11" fill="#444">+{limit:.3g}</text>',
        f'<text x="{width - right + 14}" y="{top + (height - top - bottom) / 2}" font-family="sans-serif" font-size="11" fill="#444">0</text>',
        f'<text x="{width - right + 14}" y="{height - bottom}" font-family="sans-serif" font-size="11" fill="#444">−{limit:.3g}</text>',
    ])
    for color_index in range(100):
        value = limit * (1 - 2 * color_index / 99)
        y = top + color_index * (height - top - bottom) / 100
        elements.append(f'<rect x="{width - right + 2}" y="{y:.2f}" width="10" height="{(height - top - bottom) / 100 + 0.2:.2f}" fill="{color(value)}"/>')
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n")


def generate_plots(
    output_dir: Path,
    task_summary: Sequence[Mapping[str, Any]],
    interpolation_rows: Sequence[Mapping[str, Any]],
    radial_rows: Sequence[Mapping[str, Any]],
    contour_rows: Sequence[Mapping[str, Any]],
    n_values: Sequence[int],
) -> None:
    plots_dir = output_dir / "plots"
    labels = list(dict.fromkeys(str(row["model"]) for row in task_summary))
    for inner_steps in sorted({int(row["K"]) for row in task_summary}):
        series = {}
        for label in labels:
            points = [
                (float(row["N"]), float(row["exact_match"]))
                for row in task_summary
                if row["model"] == label and int(row["K"]) == inner_steps
            ]
            series[label] = points
        write_line_svg(
            plots_dir / f"task-em-vs-n-k{inner_steps}.svg",
            series,
            f"Exact match vs. number of facts (K={inner_steps})",
            "N facts",
            "exact match",
        )

    for n_value in n_values:
        for metric, y_label in (
            ("exact_match", "exact match"),
            ("target_loss", "target loss"),
            ("objective_decrease", "write-objective decrease"),
        ):
            series = {}
            for label in labels:
                series[label] = [
                    (float(row["K"]), float(row[metric]))
                    for row in task_summary
                    if row["model"] == label and int(row["N"]) == n_value
                ]
            write_line_svg(
                plots_dir / f"task-{metric.replace('_', '-')}-vs-k-n{n_value}.svg",
                series,
                f"{y_label.title()} vs. inner steps (N={n_value})",
                "inner steps K",
                y_label,
            )

    for n_value in sorted({int(row["N"]) for row in interpolation_rows}):
        series = {
            label: [
                (float(row["t"]), float(row["median_delta_objective"]))
                for row in interpolation_rows
                if row["model"] == label and int(row["N"]) == n_value
            ]
            for label in labels
        }
        write_line_svg(
            plots_dir / f"interpolation-n{n_value}.svg",
            series,
            f"Positive-to-mismatched interpolation (N={n_value})",
            "interpolation t",
            "median Δ write objective",
        )

    for n_value in sorted({int(row["N"]) for row in radial_rows}):
        series = {
            label: [
                (float(row["radius"]), float(row["median_delta_objective"]))
                for row in radial_rows
                if row["model"] == label and int(row["N"]) == n_value
            ]
            for label in labels
        }
        write_line_svg(
            plots_dir / f"radial-n{n_value}.svg",
            series,
            f"Random radial landscape scan (N={n_value})",
            "radius / write radius",
            "median Δ write objective",
        )

    for label in labels:
        for n_value in sorted({int(row.get("N", 8)) for row in contour_rows if row["model"] == label}):
            rows = [
                row for row in contour_rows
                if row["model"] == label and int(row.get("N", 8)) == n_value
            ]
            write_heatmap_svg(
                plots_dir / f"contour-{label}-n{n_value}.svg",
                rows,
                f"2D write-objective slice: {label} (N={n_value})",
                value_key="mean_delta_objective",
            )


def generate_aggregate_plots(
    output_dir: Path,
    task_rows: Sequence[Mapping[str, Any]],
    interpolation_rows: Sequence[Mapping[str, Any]],
    radial_rows: Sequence[Mapping[str, Any]],
    contour_rows: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
) -> None:
    plots_dir = output_dir / "plots"
    for inner_steps in sorted({int(row["K"]) for row in task_rows}):
        series = {}
        bands = {}
        for alias in aliases:
            selected = [row for row in task_rows if row["alias"] == alias and int(row["K"]) == inner_steps]
            series[alias] = [(float(row["N"]), float(row["exact_match_mean"])) for row in selected]
            bands[alias] = [
                (float(row["N"]), float(row["exact_match_mean"]), row.get("exact_match_std"))
                for row in selected
            ]
        write_line_svg(
            plots_dir / f"task-em-vs-n-k{inner_steps}.svg",
            series,
            f"Mean exact match vs. number of facts (K={inner_steps})",
            "N facts",
            "exact match mean ± run std",
            std_series=bands,
        )

    for n_value in sorted({int(row["N"]) for row in task_rows}):
        for metric, label in (
            ("exact_match", "exact match"),
            ("target_loss", "target loss"),
            ("objective_decrease", "write-objective decrease"),
        ):
            series = {}
            bands = {}
            for alias in aliases:
                selected = [row for row in task_rows if row["alias"] == alias and int(row["N"]) == n_value]
                series[alias] = [
                    (float(row["K"]), float(row[f"{metric}_mean"])) for row in selected
                ]
                bands[alias] = [
                    (float(row["K"]), float(row[f"{metric}_mean"]), row.get(f"{metric}_std"))
                    for row in selected
                ]
            write_line_svg(
                plots_dir / f"task-{metric.replace('_', '-')}-vs-k-n{n_value}.svg",
                series,
                f"Mean {label} vs. inner steps (N={n_value})",
                "inner steps K",
                f"{label} mean ± run std",
                std_series=bands,
            )

    for n_value in sorted({int(row["N"]) for row in interpolation_rows}):
        series = {}
        bands = {}
        for alias in aliases:
            selected = [row for row in interpolation_rows if row["alias"] == alias and int(row["N"]) == n_value]
            series[alias] = [
                (float(row["t"]), float(row["median_delta_objective_mean"])) for row in selected
            ]
            bands[alias] = [
                (float(row["t"]), float(row["median_delta_objective_mean"]), row.get("median_delta_objective_std"))
                for row in selected
            ]
        write_line_svg(
            plots_dir / f"interpolation-n{n_value}.svg",
            series,
            f"Mean positive-to-mismatched interpolation (N={n_value})",
            "interpolation t",
            "median Δ objective mean ± run std",
            std_series=bands,
        )

    for n_value in sorted({int(row["N"]) for row in radial_rows}):
        series = {}
        bands = {}
        for alias in aliases:
            selected = [row for row in radial_rows if row["alias"] == alias and int(row["N"]) == n_value]
            series[alias] = [
                (float(row["radius"]), float(row["median_delta_objective_mean"])) for row in selected
            ]
            bands[alias] = [
                (float(row["radius"]), float(row["median_delta_objective_mean"]), row.get("median_delta_objective_std"))
                for row in selected
            ]
        write_line_svg(
            plots_dir / f"radial-n{n_value}.svg",
            series,
            f"Mean radial landscape scan (N={n_value})",
            "radius / write radius",
            "median Δ objective mean ± run std",
            std_series=bands,
        )

    for alias in aliases:
        for n_value in sorted({int(row["N"]) for row in contour_rows if row["alias"] == alias}):
            selected = [
                row for row in contour_rows
                if row["alias"] == alias and int(row["N"]) == n_value
            ]
            write_heatmap_svg(
                plots_dir / f"contour-mean-{alias}-n{n_value}.svg",
                selected,
                f"Mean 2D write-objective slice: {alias} (N={n_value})",
                value_key="mean_delta_objective_mean",
            )
            if any(row.get("mean_delta_objective_std") is not None for row in selected):
                write_heatmap_svg(
                    plots_dir / f"contour-std-{alias}-n{n_value}.svg",
                    selected,
                    f"Run standard deviation of 2D slice: {alias} (N={n_value})",
                    value_key="mean_delta_objective_std",
                )

def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def generate_report(
    output_dir: Path,
    frozen_models: Sequence[FrozenModel | ReportCheckpoint],
    task_summary: Sequence[Mapping[str, Any]],
    paired_summary: Sequence[Mapping[str, Any]],
    matching_summary: Sequence[Mapping[str, Any]],
    interpolation_summary: Sequence[Mapping[str, Any]],
    radial_summary: Sequence[Mapping[str, Any]],
    reproduction: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> None:
    task_table = format_markdown_table(
        ["Model", "N", "K", "Exact match", "Token accuracy", "Target loss", "Objective decrease", "Runtime (s)"],
        [[
            row["model"],
            row["N"],
            row["K"],
            f'{float(row["exact_match"]):.4f}',
            f'{float(row["token_accuracy"]):.4f}',
            f'{float(row["target_loss"]):.5f}',
            f'{float(row["objective_decrease"]):.5f}',
            f'{float(row["runtime_seconds"]):.1f}',
        ] for row in task_summary],
    )
    matching_table = format_markdown_table(
        ["Model", "Objective", "N", "Top-1", "MRR", "AUC", "Mean margin", "Margin ≥ 0.1"],
        [[
            row["model"], row["objective_type"], row["N"],
            f'{float(row["top1_accuracy"]):.3f}',
            f'{float(row["mrr"]):.3f}',
            f'{float(row["pairwise_auc"]):.3f}',
            f'{float(row["mean_margin"]):.3f}',
            f'{float(row["margin_satisfaction"]):.3f}',
        ] for row in matching_summary],
    ) if matching_summary else "Landscape probes were skipped."

    monotonic_table = format_markdown_table(
        ["Model", "Objective", "N", "Monotonic interpolation", "Endpoint Δ objective"],
        [[
            row["model"], row["objective_type"], row["N"],
            f'{float(row["monotonic_rise_fraction"]):.3f}',
            f'{float(row["endpoint_mean_delta"]):.3f}',
        ] for row in interpolation_summary],
    ) if interpolation_summary else "Landscape probes were skipped."

    radial_table = format_markdown_table(
        ["Model", "Objective", "N", "Radius", "Objective rises", "Median Δ objective"],
        [[
            row["model"], row["objective_type"], row["N"],
            f'{float(row["radius"]):.3g}',
            f'{float(row["fraction_objective_rises"]):.3f}',
            f'{float(row["median_delta_objective"]):.4f}',
        ] for row in radial_summary if float(row["radius"]) > 0.0],
    ) if radial_summary else "Landscape probes were skipped."

    comparison_table = format_markdown_table(
        ["Comparison", "N", "K", "Reference EM", "Candidate EM", "Difference (95% CI)"],
        [[
            row["comparison"], row["N"], row["K"],
            f'{float(row["reference_exact_match"]):.4f}',
            f'{float(row["candidate_exact_match"]):.4f}',
            (
                f'{float(row["exact_match_difference"]):+.4f} '
                f'[{float(row["exact_match_difference_ci_low"]):+.4f}, '
                f'{float(row["exact_match_difference_ci_high"]):+.4f}]'
            ),
        ] for row in paired_summary],
    ) if paired_summary else "No checkpoint comparisons were requested."

    best_rows = []
    for model in frozen_models:
        for n_value in sorted({int(row["N"]) for row in task_summary}):
            candidates = [
                row for row in task_summary
                if row["model"] == model.label and int(row["N"]) == n_value
            ]
            if candidates:
                best = max(candidates, key=lambda row: float(row["exact_match"]))
                best_rows.append([
                    model.label, n_value, best["K"], f'{float(best["exact_match"]):.4f}'
                ])
    best_table = format_markdown_table(
        ["Model", "N", "Best K", "Best EM"], best_rows
    ) if best_rows else ""
    metadata_lines = ""
    if checkpoint_metadata:
        metadata_lines = "\n".join([
            f'- Run: `{checkpoint_metadata.get("run_id", "—")}`',
            f'- Seed: `{checkpoint_metadata.get("seed", "—")}`',
            f'- Selector: `{checkpoint_metadata.get("checkpoint_selector", "—")}`',
            f'- Objective: `{checkpoint_metadata.get("objective_type", "—")}`',
            f'- Checkpoint SHA-256: `{checkpoint_metadata.get("signature", {}).get("checkpoint_sha256", "—")}`',
        ]) + "\n\n"
    artifact_links = [
        "[plots](plots/)",
        "[task summary](task_summary.csv)",
        "[task examples](task_examples.jsonl)",
    ]
    if matching_summary:
        artifact_links.extend([
            "[matching summary](matching_summary.csv)",
            "[interpolation data](interpolation.csv)",
            "[radial data](radial.csv)",
            "[contours](contours.csv)",
        ])
    artifact_links.append("[reproduction details](reproduction.json)")
    artifact_text = ", ".join(artifact_links[:-1]) + f", and {artifact_links[-1]}"

    text = f"""# Frozen-checkpoint evaluation

## Checkpoints

{chr(10).join(f'- **{model.label}:** `{model.checkpoint_path}`' for model in frozen_models)}

{metadata_lines}Checkpoint parameters remain frozen throughout evaluation.

## Reproduction check

- Status: **{reproduction.get('status', 'not run')}**
- Details: `{json.dumps(json_safe(reproduction), sort_keys=True)}`

## Task metrics

{task_table}

## Checkpoint comparisons

{comparison_table}

## Best K by model and N

{best_table}

## Context–memory matching

{matching_table}

## Interpolation geometry

{monotonic_table}

## Radial geometry

{radial_table}

See {artifact_text} for the complete output.
"""
    (output_dir / "report.md").write_text(text)


def verify_reproduction(
    task_summary: Sequence[Mapping[str, Any]],
    max_examples: int | None,
    expected: Mapping[str, float] | None = None,
    reference_n: int | None = None,
    reference_k: int | None = None,
    tolerance_examples: int = 1,
) -> dict[str, Any]:
    if max_examples is not None and max_examples < 5000:
        return {"status": "skipped", "reason": "partial validation split"}
    if not expected or reference_n is None or reference_k is None:
        return {"status": "skipped", "reason": "no applicable recorded-best reference metric"}
    expected = {str(label): float(value) for label, value in expected.items()}
    observed = {
        str(row["model"]): float(row["exact_match"])
        for row in task_summary
        if int(row["N"]) == int(reference_n) and int(row["K"]) == int(reference_k)
    }
    if set(observed) != set(expected):
        return {
            "status": "skipped",
            "reason": f"reference cell N{reference_n}/K{reference_k} was not evaluated",
            "observed": observed,
        }
    evaluated_counts = [
        int(row["num_examples"])
        for row in task_summary
        if int(row["N"]) == int(reference_n)
        and int(row["K"]) == int(reference_k)
        and row.get("num_examples") is not None
    ]
    reference_count = min(evaluated_counts) if evaluated_counts else 5000
    tolerance = tolerance_examples / reference_count
    differences = {label: observed[label] - value for label, value in expected.items()}
    if any(abs(value) > tolerance + 1e-12 for value in differences.values()):
        raise AssertionError(
            f"N{reference_n}/K{reference_k} reproduction failed: "
            f"expected={expected}, observed={observed}, tolerance={tolerance}"
        )
    return {
        "status": "passed",
        "expected": expected,
        "observed": observed,
        "differences": differences,
        "tolerance": tolerance,
        "num_examples": reference_count,
        "reference_N": int(reference_n),
        "reference_K": int(reference_k),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-paths", nargs="+", type=Path)
    parser.add_argument("--aliases", nargs="+")
    parser.add_argument("--checkpoint-selector", default="best")
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUN_PATH", "CHECKPOINT"),
        help="repeatable frozen model specification",
    )
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--shaped-run", type=Path)
    parser.add_argument("--checkpoint", default="checkpoint-18500")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recompute-metrics",
        nargs="*",
        choices=("all", *METRIC_FAMILY_VERSIONS),
        default=(),
        help="ignore selected checkpoint-local metric caches",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="populate checkpoint-local caches without building aggregate comparisons",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--n-values", nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--inner-steps", nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=143)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/checkpoint_eval"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--skip-landscape", action="store_true")
    parser.add_argument("--landscape-n-values", nargs="+", default=[8, 32])
    parser.add_argument("--matching-n-values", nargs="+", default=None)
    parser.add_argument("--matching-examples", type=int, default=1024)
    parser.add_argument("--matching-bank-size", type=int, default=64)
    parser.add_argument("--scan-examples", type=int, default=512)
    parser.add_argument("--contour-examples", type=int, default=64)
    parser.add_argument("--num-radial-directions", type=int, default=8)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    modes = sum(bool(value) for value in (
        args.model,
        args.experiment_paths,
        args.baseline_run or args.shaped_run,
    ))
    if modes != 1:
        raise ValueError(
            "Use exactly one input mode: --experiment-paths/--aliases, repeatable --model, "
            "or the legacy pair flags"
        )
    if bool(args.experiment_paths) != bool(args.aliases):
        raise ValueError("--experiment-paths and --aliases must be provided together")
    if args.experiment_paths and len(args.experiment_paths) != len(args.aliases):
        raise ValueError("--experiment-paths and --aliases must have equal lengths")
    if args.experiment_paths:
        normalized_paths = [str(path.resolve()) for path in args.experiment_paths]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("--experiment-paths must be unique")
    if bool(args.baseline_run) != bool(args.shaped_run):
        raise ValueError("Legacy mode requires both --baseline-run and --shaped-run")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.matching_bank_size <= 1:
        raise ValueError("matching-bank-size must be greater than one")
    if args.matching_examples < args.matching_bank_size:
        raise ValueError("matching-examples must be at least matching-bank-size")
    if args.matching_examples % args.matching_bank_size != 0:
        raise ValueError("matching-examples must be divisible by matching-bank-size")
    if args.contour_examples > args.scan_examples:
        raise ValueError("contour-examples cannot exceed scan-examples")

    aliases = [item.alias for item in experiment_inputs_from_args(args)]
    if len(aliases) != len({alias.casefold() for alias in aliases}):
        raise ValueError(f"Aliases must be unique: {aliases}")
    invalid = [
        alias for alias in aliases
        if alias in (".", "..") or not SAFE_ALIAS_RE.fullmatch(alias)
    ]
    if invalid:
        raise ValueError(f"Aliases may contain only letters, numbers, '.', '_' and '-': {invalid}")


def model_specs_from_args(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    """Compatibility view of parsed inputs as (alias, path, selector) tuples."""
    return [(item.alias, item.path, item.checkpoint_selector) for item in experiment_inputs_from_args(args)]


def experiment_inputs_from_args(args: argparse.Namespace) -> list[ExperimentInput]:
    if args.model:
        specs = [ExperimentInput(str(label), Path(run_path), str(checkpoint)) for label, run_path, checkpoint in args.model]
    elif args.experiment_paths:
        specs = [
            ExperimentInput(str(alias), Path(path), str(args.checkpoint_selector))
            for alias, path in zip(args.aliases, args.experiment_paths)
        ]
    else:
        specs = [
            ExperimentInput("baseline", args.baseline_run, args.checkpoint),
            ExperimentInput("shaped", args.shaped_run, args.checkpoint),
        ]
    return specs


def requested_comparisons(labels: Sequence[str]) -> list[tuple[str, str]]:
    return [
        (labels[left], labels[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    ]


def evaluation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "data_root": str(args.data_root.resolve()),
        "n_values": list(args.n_values),
        "inner_steps": list(args.inner_steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "max_eval_examples": args.max_eval_examples,
        "skip_landscape": bool(args.skip_landscape),
        "landscape_n_values": list(args.landscape_n_values),
        "matching_n_values": list(args.matching_n_values),
        "matching_examples": int(args.matching_examples),
        "matching_bank_size": int(args.matching_bank_size),
        "scan_examples": int(args.scan_examples),
        "contour_examples": int(args.contour_examples),
        "num_radial_directions": int(args.num_radial_directions),
        "landscape_inner_steps": 2,
        "matching_margin": 0.1,
        "interpolation_values": [float(value) for value in np.linspace(0.0, 1.0, 21)],
        "radial_radii": [0.0, 0.125, 0.25, 0.5, 1.0, 1.5],
        "contour_grid": {
            "x_min": -0.5,
            "x_max": 1.5,
            "y_min": -1.0,
            "y_max": 1.0,
            "size": 31,
        },
        "device": str(args.device),
        "dtype": "float32",
    }


def metric_family_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Return computation inputs grouped by independently cacheable family."""
    common = {
        "data_root": str(args.data_root.resolve()),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "dtype": "float32",
    }
    landscape = {
        **common,
        "inner_steps": 2,
    }
    return {
        "task": {
            **common,
            "n_values": list(args.n_values),
            "inner_steps": list(args.inner_steps),
            "max_eval_examples": args.max_eval_examples,
        },
        "matching": {
            **landscape,
            "n_values": [] if args.skip_landscape else list(args.matching_n_values),
            "examples": int(args.matching_examples),
            "bank_size": int(args.matching_bank_size),
            "margin": 0.1,
        },
        "interpolation": {
            **landscape,
            "n_values": [] if args.skip_landscape else list(args.landscape_n_values),
            "examples": int(args.scan_examples),
            "bank_size": int(args.matching_bank_size),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "values": [float(value) for value in np.linspace(0.0, 1.0, 21)],
        },
        "radial": {
            **landscape,
            "n_values": [] if args.skip_landscape else list(args.landscape_n_values),
            "examples": int(args.scan_examples),
            "bank_size": int(args.matching_bank_size),
            "num_directions": int(args.num_radial_directions),
            "radii": [0.0, 0.125, 0.25, 0.5, 1.0, 1.5],
        },
        "contours": {
            **landscape,
            "n_values": [] if args.skip_landscape else list(args.landscape_n_values),
            "examples": int(args.contour_examples),
            "bank_size": int(args.matching_bank_size),
            "x_values": [float(value) for value in np.linspace(-0.5, 1.5, 31)],
            "y_values": [float(value) for value in np.linspace(-1.0, 1.0, 31)],
            "direction_seed_offset": 71,
        },
    }


def tokenizer_signature_digest(tokenizer: Any) -> str:
    def stable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): stable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        return str(value)

    signature = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocabulary": tokenizer.get_vocab(),
        "added_vocabulary": tokenizer.get_added_vocab(),
        "special_tokens": stable(tokenizer.special_tokens_map),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def per_checkpoint_output_dir(output_dir: Path, spec: ResolvedRun) -> Path:
    return output_dir / "per_checkpoint" / spec.alias / spec.run_id / spec.checkpoint_name


def checkpoint_metric_cache_root(spec: ResolvedRun) -> Path:
    return spec.checkpoint_path.parent / METRIC_CACHE_DIRNAME


def metric_cache_signature(
    checkpoint_sha256: str,
    family: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cache_schema_version": METRIC_CACHE_SCHEMA_VERSION,
        "family": family,
        "family_version": METRIC_FAMILY_VERSIONS[family],
        "checkpoint_sha256": checkpoint_sha256,
        "config": json_safe(config),
    }


def metric_cache_path(spec: ResolvedRun, signature: Mapping[str, Any]) -> Path:
    encoded = json.dumps(json_safe(signature), sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:20]
    return checkpoint_metric_cache_root(spec) / str(signature["family"]) / key


def load_metric_cache(
    cache_path: Path,
    signature: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    manifest_path = cache_path / "manifest.json"
    payload_path = cache_path / "metrics.json"
    if not manifest_path.exists() or not payload_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("signature") != json_safe(signature):
        return None
    payload = json.loads(payload_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metric cache payload: {payload_path}")
    return payload, manifest


def write_metric_cache(
    cache_path: Path,
    signature: Mapping[str, Any],
    payload: Mapping[str, Any],
    frozen: FrozenModel,
) -> dict[str, Any]:
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "signature": signature,
        "checkpoint_path": frozen.checkpoint_path,
        "objective_type": frozen.objective_type,
        "tokenizer_signature_sha256": tokenizer_signature_digest(frozen.tokenizer),
        "torch_version": torch.__version__,
        "datasets_version": datasets.__version__,
    }
    write_json(cache_path / "manifest.json", manifest)
    write_json(cache_path / "metrics.json", payload)
    manifest["status"] = "complete"
    write_json(cache_path / "manifest.json", manifest)
    return manifest


def relabel_metric_rows(rows: Sequence[Mapping[str, Any]], spec: ResolvedRun) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    for row in materialized:
        if "model" in row:
            row["model"] = spec.alias
    add_run_metadata(materialized, spec)
    return materialized


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    checkpoint_output: Path,
) -> None:
    if manifest.get("status") != "complete":
        raise ValueError(f"Cannot resume incomplete checkpoint output: {checkpoint_output}")
    if manifest.get("signature") != json_safe(signature):
        raise ValueError(f"Resume signature mismatch for checkpoint output: {checkpoint_output}")


def evaluate_resolved_run(
    spec: ResolvedRun,
    args: argparse.Namespace,
    loaded_datasets: MutableMapping[int, Any],
    device: torch.device,
) -> tuple[Path, str]:
    def dataset_for(n_value: int) -> Any:
        if n_value not in loaded_datasets:
            dataset_path = args.data_root / f"N{n_value}-K2V2-V62_1M"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            loaded_datasets[n_value] = datasets.load_from_disk(str(dataset_path))["valid"]
        return loaded_datasets[n_value]

    checkpoint_digest = checkpoint_file_digest(spec.checkpoint_path)
    family_configs = metric_family_configs(args)
    family_signatures = {
        family: metric_cache_signature(checkpoint_digest, family, config)
        for family, config in family_configs.items()
    }
    family_paths = {
        family: metric_cache_path(spec, signature)
        for family, signature in family_signatures.items()
    }
    signature = {
        "checkpoint_sha256": checkpoint_digest,
        "evaluation": evaluation_config(args),
        "alias": spec.alias,
        "run_id": spec.run_id,
        "seed": spec.seed,
        "checkpoint": spec.checkpoint_name,
    }
    checkpoint_output = per_checkpoint_output_dir(args.output_dir, spec)
    manifest_path = checkpoint_output / "manifest.json"
    if checkpoint_output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Checkpoint output already exists; use --resume or a new output directory: {checkpoint_output}"
            )
        if not manifest_path.exists() and not args.cache_only:
            raise ValueError(f"Cannot resume checkpoint without manifest: {checkpoint_output}")
        if manifest_path.exists() and not args.cache_only:
            existing = json.loads(manifest_path.read_text())
            validate_resume_manifest(existing, signature, checkpoint_output)
    else:
        checkpoint_output.mkdir(parents=True, exist_ok=False)

    recompute = set(args.recompute_metrics)
    cache_results: dict[str, dict[str, Any]] = {}
    cache_manifests: dict[str, dict[str, Any]] = {}
    for family in METRIC_FAMILY_VERSIONS:
        cached = None
        if "all" not in recompute and family not in recompute:
            cached = load_metric_cache(family_paths[family], family_signatures[family])
        if cached is not None:
            cache_results[family], cache_manifests[family] = cached
            print(
                f"[cache hit] alias={spec.alias} run={spec.run_id} "
                f"checkpoint={spec.checkpoint_name} metric={family}",
                flush=True,
            )

    missing_families = set(METRIC_FAMILY_VERSIONS) - set(cache_results)
    frozen: FrozenModel | None = None
    if missing_families:
        frozen = load_frozen_model(
            CACHE_MODEL_LABEL,
            spec.run_path,
            spec.checkpoint_path,
            device=device,
        )

    if "task" in missing_families:
        assert frozen is not None
        task_rows: list[dict[str, Any]] = []
        task_summary: list[dict[str, Any]] = []
        for n_value in args.n_values:
            for inner_steps in args.inner_steps:
                print(
                    f"[task] alias={spec.alias} run={spec.run_id} seed={spec.seed} "
                    f"checkpoint={spec.checkpoint_name} N={n_value} K={inner_steps}",
                    flush=True,
                )
                rows, summary = evaluate_task_cell(
                    frozen,
                    dataset_for(n_value),
                    n_value,
                    inner_steps,
                    args.batch_size,
                    max_examples=args.max_eval_examples,
                )
                task_rows.extend(rows)
                task_summary.append(summary)
        cache_results["task"] = {"rows": task_rows, "summary": task_summary}

    landscape_missing = missing_families - {"task"}
    landscape_payloads: dict[str, dict[str, Any]] = {
        "matching": {"summary": [], "rows": []},
        "interpolation": {"rows": [], "summary": []},
        "radial": {"rows": []},
        "contours": {"rows": []},
    }
    if landscape_missing:
        assert frozen is not None
        requested_by_family = {
            family: set(family_configs[family]["n_values"])
            for family in landscape_missing
        }
        landscape_n_values = sorted(set().union(*requested_by_family.values()))
        for n_value in landscape_n_values:
            required_counts = []
            if n_value in requested_by_family.get("matching", set()):
                required_counts.append(args.matching_examples)
            if n_value in requested_by_family.get("interpolation", set()):
                required_counts.append(args.scan_examples)
            if n_value in requested_by_family.get("radial", set()):
                required_counts.append(args.scan_examples)
            if n_value in requested_by_family.get("contours", set()):
                required_counts.append(args.contour_examples)
            landscape_count = max(required_counts)
            indices = deterministic_subset_indices(
                len(dataset_for(n_value)), landscape_count, args.seed + n_value
            )
            print(
                f"[landscape] alias={spec.alias} run={spec.run_id} seed={spec.seed} N={n_value} "
                f"metrics={','.join(sorted(family for family, values in requested_by_family.items() if n_value in values))}",
                flush=True,
            )
            batches = collect_landscape_batches(
                frozen,
                dataset_for(n_value),
                indices,
                args.matching_bank_size,
                inner_steps=2,
            )
            if n_value in requested_by_family.get("matching", set()):
                matching_batch_count = args.matching_examples // args.matching_bank_size
                summary, rows = matching_probe(
                    frozen,
                    batches[:matching_batch_count],
                    n_value,
                    margin=0.1,
                    eval_batch_size=args.batch_size,
                )
                landscape_payloads["matching"]["summary"].append(summary)
                landscape_payloads["matching"]["rows"].extend(rows)
            if n_value in requested_by_family.get("interpolation", set()):
                scan_data = concatenate_landscape_batches(batches, args.scan_examples)
                rows, summary = interpolation_probe(
                    frozen,
                    scan_data,
                    n_value,
                    t_values=np.linspace(0.0, 1.0, 21),
                    eval_batch_size=args.batch_size,
                    bootstrap_resamples=args.bootstrap_resamples,
                    seed=args.seed,
                )
                landscape_payloads["interpolation"]["rows"].extend(rows)
                landscape_payloads["interpolation"]["summary"].append(summary)
            if n_value in requested_by_family.get("radial", set()):
                scan_data = concatenate_landscape_batches(batches, args.scan_examples)
                landscape_payloads["radial"]["rows"].extend(radial_probe(
                    frozen,
                    scan_data,
                    n_value,
                    radii=[0.0, 0.125, 0.25, 0.5, 1.0, 1.5],
                    num_directions=args.num_radial_directions,
                    eval_batch_size=args.batch_size,
                    seed=args.seed,
                ))
            if n_value in requested_by_family.get("contours", set()):
                contour_data = concatenate_landscape_batches(batches, args.contour_examples)
                landscape_payloads["contours"]["rows"].extend(contour_probe(
                    frozen,
                    contour_data,
                    n_value,
                    x_values=np.linspace(-0.5, 1.5, 31),
                    y_values=np.linspace(-1.0, 1.0, 31),
                    eval_batch_size=args.batch_size,
                    seed=args.seed + 71,
                ))
        for family in landscape_missing:
            cache_results[family] = landscape_payloads[family]

    if missing_families:
        assert frozen is not None
        digest_after = parameter_digest(frozen.model)
        if digest_after != frozen.digest_before:
            raise AssertionError(f"Frozen model parameters changed during evaluation: {spec.alias}/{spec.run_id}")
        for family in sorted(missing_families):
            cache_manifests[family] = write_metric_cache(
                family_paths[family],
                family_signatures[family],
                cache_results[family],
                frozen,
            )
    else:
        digest_after = None

    metadata_manifest = next(iter(cache_manifests.values()))
    tokenizer_digest = str(metadata_manifest["tokenizer_signature_sha256"])
    objective_type = str(metadata_manifest["objective_type"])
    manifest = {
        "status": "running",
        "signature": signature,
        "alias": spec.alias,
        "source_path": spec.source_path,
        "run_path": spec.run_path,
        "run_id": spec.run_id,
        "seed": spec.seed,
        "checkpoint_selector": spec.checkpoint_selector,
        "checkpoint_path": spec.checkpoint_path,
        "checkpoint_name": spec.checkpoint_name,
        "is_recorded_best": spec.is_recorded_best,
        "objective_type": objective_type,
        "parameter_digest_before": frozen.digest_before if frozen is not None else None,
        "tokenizer_signature_sha256": tokenizer_digest,
        "metric_caches": {
            family: {
                "path": family_paths[family],
                "signature": family_signatures[family],
                "reused": family not in missing_families,
            }
            for family in METRIC_FAMILY_VERSIONS
        },
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "datasets_version": datasets.__version__,
    }
    write_json(manifest_path, manifest)

    if args.cache_only:
        manifest.update({
            "status": "complete",
            "parameter_digest_after": digest_after,
        })
        write_json(manifest_path, manifest)
        del frozen
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        return checkpoint_output, tokenizer_digest

    task_rows = relabel_metric_rows(cache_results["task"]["rows"], spec)
    task_summary = relabel_metric_rows(cache_results["task"]["summary"], spec)

    reference = training_reference(spec.run_path) if spec.is_recorded_best else None
    reproduction = verify_reproduction(
        task_summary,
        args.max_eval_examples,
        expected={spec.alias: reference[2]} if reference else None,
        reference_n=reference[0] if reference else None,
        reference_k=reference[1] if reference else None,
    )
    write_jsonl(checkpoint_output / "task_examples.jsonl", task_rows)
    write_csv(checkpoint_output / "task_summary.csv", task_summary)
    write_json(checkpoint_output / "task_summary.json", task_summary)
    write_json(checkpoint_output / "reproduction.json", reproduction)

    matching_summary = relabel_metric_rows(cache_results["matching"]["summary"], spec)
    matching_rows = relabel_metric_rows(cache_results["matching"]["rows"], spec)
    interpolation_rows = relabel_metric_rows(cache_results["interpolation"]["rows"], spec)
    interpolation_summary = relabel_metric_rows(cache_results["interpolation"]["summary"], spec)
    radial_rows = relabel_metric_rows(cache_results["radial"]["rows"], spec)
    contour_rows = relabel_metric_rows(cache_results["contours"]["rows"], spec)
    if not args.skip_landscape:
        write_csv(checkpoint_output / "matching_summary.csv", matching_summary)
        write_jsonl(checkpoint_output / "matching_examples.jsonl", matching_rows)
        write_csv(checkpoint_output / "interpolation.csv", interpolation_rows)
        write_csv(checkpoint_output / "interpolation_summary.csv", interpolation_summary)
        write_csv(checkpoint_output / "radial.csv", radial_rows)
        write_csv(checkpoint_output / "radial_summary.csv", radial_rows)
        write_csv(checkpoint_output / "contours.csv", contour_rows)
        write_json(checkpoint_output / "landscape_summary.json", {
            "matching": matching_summary,
            "interpolation": interpolation_summary,
            "radial": radial_rows,
        })

    generate_plots(
        checkpoint_output,
        task_summary,
        interpolation_rows,
        radial_rows,
        contour_rows,
        args.n_values,
    )
    generate_report(
        checkpoint_output,
        [ReportCheckpoint(spec.alias, spec.checkpoint_path)],
        task_summary,
        [],
        matching_summary,
        interpolation_summary,
        radial_rows,
        reproduction,
        checkpoint_metadata=manifest,
    )
    manifest.update({
        "status": "complete",
        "parameter_digest_after": digest_after,
    })
    write_json(manifest_path, manifest)
    del frozen
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_output, tokenizer_digest


def aggregate_aligned_jsonl(
    input_paths: Sequence[Path],
    output_path: Path,
    group_keys: Sequence[str],
) -> None:
    if not input_paths:
        write_jsonl(output_path, [])
        return
    handles = [path.open() for path in input_paths]
    try:
        def rows() -> Iterator[dict[str, Any]]:
            for line_number, lines in enumerate(zip_longest(*handles), start=1):
                if any(line is None for line in lines):
                    raise ValueError(f"Per-run JSONL files have different lengths near line {line_number}")
                members = [json.loads(line) for line in lines]
                keys = [tuple(member.get(key) for key in group_keys) for member in members]
                if any(key != keys[0] for key in keys[1:]):
                    raise ValueError(f"Per-run JSONL rows are not aligned near line {line_number}: {keys}")
                yield aggregate_numeric_rows(members, group_keys)[0]
        write_jsonl(output_path, rows())
    finally:
        for handle in handles:
            handle.close()


def aggregate_alias_outputs(
    alias: str,
    specs: Sequence[ResolvedRun],
    checkpoint_outputs: Mapping[tuple[str, int], Path],
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    alias_dir = output_dir / "aggregate" / alias
    alias_dir.mkdir(parents=True, exist_ok=True)
    paths = [checkpoint_outputs[(spec.alias, spec.seed)] for spec in specs]

    task_summary_rows = [
        row for path in paths for row in json.loads((path / "task_summary.json").read_text())
    ]
    task_aggregate = aggregate_numeric_rows(
        task_summary_rows, ["alias", "objective_type", "N", "K"]
    )
    write_csv(alias_dir / "task_summary.csv", task_aggregate)
    write_json(alias_dir / "task_summary.json", task_aggregate)
    aggregate_aligned_jsonl(
        [path / "task_examples.jsonl" for path in paths],
        alias_dir / "task_examples.jsonl",
        ["alias", "objective_type", "N", "K", "example_index"],
    )

    matching_rows = [row for path in paths for row in read_csv(path / "matching_summary.csv")]
    matching_aggregate = aggregate_numeric_rows(
        matching_rows, ["alias", "objective_type", "N"]
    ) if matching_rows else []
    interpolation_rows = [row for path in paths for row in read_csv(path / "interpolation.csv")]
    interpolation_aggregate = aggregate_numeric_rows(
        interpolation_rows, ["alias", "objective_type", "N", "t"]
    ) if interpolation_rows else []
    interpolation_summary_rows = [
        row for path in paths for row in read_csv(path / "interpolation_summary.csv")
    ]
    interpolation_summary_aggregate = aggregate_numeric_rows(
        interpolation_summary_rows, ["alias", "objective_type", "N"]
    ) if interpolation_summary_rows else []
    radial_rows = [row for path in paths for row in read_csv(path / "radial.csv")]
    radial_aggregate = aggregate_numeric_rows(
        radial_rows, ["alias", "objective_type", "N", "radius"]
    ) if radial_rows else []
    contour_rows = [row for path in paths for row in read_csv(path / "contours.csv")]
    contour_aggregate = aggregate_numeric_rows(
        contour_rows, ["alias", "objective_type", "N", "x", "y"]
    ) if contour_rows else []

    write_csv(alias_dir / "matching_summary.csv", matching_aggregate)
    write_csv(alias_dir / "interpolation.csv", interpolation_aggregate)
    write_csv(alias_dir / "interpolation_summary.csv", interpolation_summary_aggregate)
    write_csv(alias_dir / "radial.csv", radial_aggregate)
    write_csv(alias_dir / "radial_summary.csv", radial_aggregate)
    write_csv(alias_dir / "contours.csv", contour_aggregate)
    if matching_rows and all((path / "matching_examples.jsonl").exists() for path in paths):
        aggregate_aligned_jsonl(
            [path / "matching_examples.jsonl" for path in paths],
            alias_dir / "matching_examples.jsonl",
            ["alias", "N", "bank_index", "example_index"],
        )
    write_json(alias_dir / "landscape_summary.json", {
        "matching": matching_aggregate,
        "interpolation": interpolation_summary_aggregate,
        "radial": radial_aggregate,
    })
    generate_aggregate_plots(
        alias_dir,
        task_aggregate,
        interpolation_aggregate,
        radial_aggregate,
        contour_aggregate,
        [alias],
    )
    alias_task_table = format_markdown_table(
        ["N", "K", "Runs", "Exact match", "Token accuracy", "Target loss", "Objective decrease"],
        [[
            row["N"], row["K"], row["run_count"],
            format_mean_std(row, "exact_match"),
            format_mean_std(row, "token_accuracy"),
            format_mean_std(row, "target_loss", 5),
            format_mean_std(row, "objective_decrease", 5),
        ] for row in task_aggregate],
    )
    alias_matching_table = format_markdown_table(
        ["N", "Runs", "Top-1", "MRR", "AUC", "Mean margin", "Margin ≥ 0.1"],
        [[
            row["N"], row["run_count"],
            format_mean_std(row, "top1_accuracy"),
            format_mean_std(row, "mrr"),
            format_mean_std(row, "pairwise_auc"),
            format_mean_std(row, "mean_margin"),
            format_mean_std(row, "margin_satisfaction"),
        ] for row in matching_aggregate],
    ) if matching_aggregate else "Landscape probes were not evaluated."
    (alias_dir / "report.md").write_text(
        f"# Aggregate evaluation: {alias}\n\n"
        "Values are arithmetic means across runs ± sample standard deviation. "
        "Standard deviation is shown as `—` for a single run.\n\n"
        "## Selected runs\n\n"
        + "\n".join(
            f"- `{spec.run_id}`: seed {spec.seed}, `{spec.checkpoint_name}`"
            for spec in specs
        )
        + f"\n\n## Task metrics\n\n{alias_task_table}\n\n"
        + f"## Context–memory matching\n\n{alias_matching_table}\n\n"
        + "See [plots](plots/), [task summary](task_summary.csv), "
        + "[task examples](task_examples.jsonl), and the landscape tables for complete artifacts.\n"
    )
    return {
        "task": task_aggregate,
        "matching": matching_aggregate,
        "interpolation": interpolation_aggregate,
        "interpolation_summary": interpolation_summary_aggregate,
        "radial": radial_aggregate,
        "contours": contour_aggregate,
    }


def compare_task_files_for_seed(
    reference: ResolvedRun,
    candidate: ResolvedRun,
    reference_path: Path,
    candidate_path: Path,
    output_dir: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_metrics = (
        "exact_match",
        "token_accuracy",
        "target_loss",
        "objective_initial",
        "objective_final",
        "objective_decrease",
        "mem_norm",
        "mem_displacement",
    )
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reference_handle = (reference_path / "task_examples.jsonl").open()
    candidate_handle = (candidate_path / "task_examples.jsonl").open()
    try:
        def rows() -> Iterator[dict[str, Any]]:
            for line_number, (reference_line, candidate_line) in enumerate(
                zip_longest(reference_handle, candidate_handle), start=1
            ):
                if reference_line is None or candidate_line is None:
                    raise ValueError(f"Paired task files have different lengths near line {line_number}")
                reference_row = json.loads(reference_line)
                candidate_row = json.loads(candidate_line)
                key_fields = ("N", "K", "example_index")
                reference_key = tuple(reference_row[key] for key in key_fields)
                candidate_key = tuple(candidate_row[key] for key in key_fields)
                if reference_key != candidate_key:
                    raise ValueError(
                        f"Paired task files are not aligned near line {line_number}: "
                        f"{reference_key} != {candidate_key}"
                    )
                paired = paired_task_examples(
                    [reference_row, candidate_row],
                    comparisons=[(reference.alias, candidate.alias)],
                )[0]
                paired.update({
                    "seed": reference.seed,
                    "reference_run_id": reference.run_id,
                    "candidate_run_id": candidate.run_id,
                    "reference_checkpoint": reference.checkpoint_name,
                    "candidate_checkpoint": candidate.checkpoint_name,
                })
                cell = grouped[(int(paired["N"]), int(paired["K"]))]
                for metric in comparison_metrics:
                    cell[f"{metric}_difference"].append(float(paired[f"{metric}_difference"]))
                    cell[f"reference_{metric}"].append(float(paired[f"reference_{metric}"]))
                    cell[f"candidate_{metric}"].append(float(paired[f"candidate_{metric}"]))
                yield paired
        write_jsonl(output_dir / "task_pair_examples.jsonl", rows())
    finally:
        reference_handle.close()
        candidate_handle.close()

    summaries = []
    for cell_index, ((n_value, inner_steps), values) in enumerate(sorted(grouped.items())):
        row = {
            "reference_model": reference.alias,
            "candidate_model": candidate.alias,
            "comparison": f"{candidate.alias} - {reference.alias}",
            "seed": reference.seed,
            "reference_run_id": reference.run_id,
            "candidate_run_id": candidate.run_id,
            "reference_checkpoint": reference.checkpoint_name,
            "candidate_checkpoint": candidate.checkpoint_name,
            "N": n_value,
            "K": inner_steps,
            "num_examples": len(values["exact_match_difference"]),
        }
        for metric_index, metric in enumerate(comparison_metrics):
            differences = np.asarray(values[f"{metric}_difference"], dtype=np.float64)
            low, high = bootstrap_interval(
                differences,
                n_resamples=bootstrap_resamples,
                seed=(
                    bootstrap_seed
                    + reference.seed * 104729
                    + cell_index * 1009
                    + metric_index * 9176
                ),
            )
            row.update({
                f"reference_{metric}": float(np.mean(values[f"reference_{metric}"])),
                f"candidate_{metric}": float(np.mean(values[f"candidate_{metric}"])),
                f"{metric}_difference": float(differences.mean()),
                f"{metric}_difference_ci_low": low,
                f"{metric}_difference_ci_high": high,
            })
        summaries.append(row)
    write_csv(output_dir / "task_paired.csv", summaries)
    write_json(output_dir / "task_paired.json", summaries)
    return summaries


def build_alias_comparison(
    reference_alias: str,
    candidate_alias: str,
    specs_by_alias: Mapping[str, Sequence[ResolvedRun]],
    checkpoint_outputs: Mapping[tuple[str, int], Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    comparison_dir = output_dir / "comparisons" / f"{reference_alias}__{candidate_alias}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    reference_by_seed = {spec.seed: spec for spec in specs_by_alias[reference_alias]}
    candidate_by_seed = {spec.seed: spec for spec in specs_by_alias[candidate_alias]}
    matched_seeds = sorted(set(reference_by_seed) & set(candidate_by_seed))
    per_seed_summaries = []
    for seed in matched_seeds:
        reference = reference_by_seed[seed]
        candidate = candidate_by_seed[seed]
        per_seed_summaries.extend(compare_task_files_for_seed(
            reference,
            candidate,
            checkpoint_outputs[(reference.alias, seed)],
            checkpoint_outputs[(candidate.alias, seed)],
            comparison_dir / f"seed_{seed}",
            args.bootstrap_resamples,
            args.seed,
        ))
    aggregate = aggregate_numeric_rows(
        per_seed_summaries,
        ["reference_model", "candidate_model", "comparison", "N", "K"],
    ) if per_seed_summaries else []
    write_csv(comparison_dir / "task_comparison_summary.csv", aggregate)
    write_json(comparison_dir / "task_comparison_summary.json", aggregate)
    manifest = {
        "reference_alias": reference_alias,
        "candidate_alias": candidate_alias,
        "matched_seeds": matched_seeds,
        "reference_only_seeds": sorted(set(reference_by_seed) - set(candidate_by_seed)),
        "candidate_only_seeds": sorted(set(candidate_by_seed) - set(reference_by_seed)),
    }
    write_json(comparison_dir / "manifest.json", manifest)
    return {"manifest": manifest, "summary": aggregate}


def format_mean_std(row: Mapping[str, Any], metric: str, digits: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "—"
    if std is None:
        return f"{float(mean):.{digits}f} ± —"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def generate_root_report(
    output_dir: Path,
    resolved: Sequence[ResolvedRun],
    skipped: Sequence[SkippedRun],
    aggregate_by_alias: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    comparison_results: Mapping[tuple[str, str], Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    checkpoint_details = {}
    for spec in resolved:
        manifest_path = per_checkpoint_output_dir(output_dir, spec) / "manifest.json"
        if manifest_path.exists():
            checkpoint_details[(spec.alias, spec.seed)] = json.loads(manifest_path.read_text())
    checkpoint_table = format_markdown_table(
        ["Alias", "Run", "Seed", "Selector", "Checkpoint", "Objective", "Recorded best", "SHA-256", "Path"],
        [[
            spec.alias,
            spec.run_id,
            spec.seed,
            spec.checkpoint_selector,
            spec.checkpoint_name,
            checkpoint_details.get((spec.alias, spec.seed), {}).get("objective_type", "—"),
            "yes" if spec.is_recorded_best else "no",
            str(checkpoint_details.get((spec.alias, spec.seed), {}).get("signature", {}).get("checkpoint_sha256", "—"))[:12],
            f'`{spec.checkpoint_path}`',
        ] for spec in resolved],
    ) if resolved else "No checkpoints were evaluated."
    skipped_table = format_markdown_table(
        ["Alias", "Run path", "Reason"],
        [[item.alias, f'`{item.run_path}`', item.reason] for item in skipped],
    ) if skipped else "No discovered runs were skipped."

    task_rows = [row for value in aggregate_by_alias.values() for row in value["task"]]
    task_table = format_markdown_table(
        ["Alias", "N", "K", "Runs", "Exact match", "Token accuracy", "Target loss", "Objective decrease"],
        [[
            row["alias"], row["N"], row["K"], row["run_count"],
            format_mean_std(row, "exact_match"),
            format_mean_std(row, "token_accuracy"),
            format_mean_std(row, "target_loss", 5),
            format_mean_std(row, "objective_decrease", 5),
        ] for row in task_rows],
    ) if task_rows else "No task aggregates were produced."

    best_rows = []
    for alias in dict.fromkeys(spec.alias for spec in resolved):
        alias_rows = [row for row in task_rows if row["alias"] == alias]
        for n_value in sorted({int(row["N"]) for row in alias_rows}):
            candidates = [row for row in alias_rows if int(row["N"]) == n_value]
            best = max(candidates, key=lambda row: float(row["exact_match_mean"]))
            best_rows.append([
                alias, n_value, best["K"], best["run_count"], format_mean_std(best, "exact_match")
            ])
    best_table = format_markdown_table(
        ["Alias", "N", "Best K", "Runs", "Exact match"], best_rows
    ) if best_rows else "No best-K rows were produced."

    matching_rows = [row for value in aggregate_by_alias.values() for row in value["matching"]]
    matching_table = format_markdown_table(
        ["Alias", "N", "Runs", "Top-1", "MRR", "AUC", "Mean margin", "Margin ≥ 0.1"],
        [[
            row["alias"], row["N"], row["run_count"],
            format_mean_std(row, "top1_accuracy"),
            format_mean_std(row, "mrr"),
            format_mean_std(row, "pairwise_auc"),
            format_mean_std(row, "mean_margin"),
            format_mean_std(row, "margin_satisfaction"),
        ] for row in matching_rows],
    ) if matching_rows else "Landscape probes were not evaluated."

    comparison_rows = []
    unmatched_rows = []
    for (reference_alias, candidate_alias), result in comparison_results.items():
        manifest = result["manifest"]
        for row in result["summary"]:
            comparison_rows.append([
                row["comparison"], row["N"], row["K"], row["run_count"],
                format_mean_std(row, "exact_match_difference"),
            ])
        unmatched_rows.append([
            f"{candidate_alias} - {reference_alias}",
            ", ".join(map(str, manifest["matched_seeds"])) or "—",
            ", ".join(map(str, manifest["reference_only_seeds"])) or "—",
            ", ".join(map(str, manifest["candidate_only_seeds"])) or "—",
        ])
    comparison_table = format_markdown_table(
        ["Comparison", "N", "K", "Matched runs", "Exact-match difference"], comparison_rows
    ) if comparison_rows else "No seed-paired comparison rows were produced."
    seed_table = format_markdown_table(
        ["Comparison", "Matched seeds", "Reference-only", "Candidate-only"], unmatched_rows
    ) if unmatched_rows else "Only one alias was evaluated."

    aliases = list(dict.fromkeys(spec.alias for spec in resolved))
    text = f"""# Frozen-checkpoint evaluation

## Evaluation settings

- Device: `{device}`
- Dtype: `float32`
- N values: `{args.n_values}`
- Inner steps: `{args.inner_steps}`
- Validation limit: `{args.max_eval_examples if args.max_eval_examples is not None else 'complete split'}`
- Landscape N values: `{[] if args.skip_landscape else args.landscape_n_values}`
- Matching N values: `{[] if args.skip_landscape else args.matching_n_values}`
- Landscape write steps: `2`
- Matching margin: `0.1`
- Interpolation: `0.00…1.00` (21 points)
- Radial radii: `[0, 0.125, 0.25, 0.5, 1, 1.5]` times write radius
- Contours: `31×31`, x `[-0.5, 1.5]`, y `[-1, 1]`
- Bootstrap resamples: `{args.bootstrap_resamples}`

## Evaluated checkpoints

{checkpoint_table}

## Skipped runs

{skipped_table}

## Task aggregates by alias

Values are arithmetic run means ± sample standard deviation. Standard deviation is shown as `—` for a single run.

{task_table}

## Best K by alias and N

{best_table}

## Seed-paired comparisons

{seed_table}

{comparison_table}

## Context–memory matching aggregates

{matching_table}

## Artifacts

- [Per-checkpoint data and plots](per_checkpoint/)
- [Per-alias aggregates](aggregate/)
- [Seed-paired comparisons](comparisons/)
- [Cross-alias aggregate plots](plots/)
- Evaluated aliases: `{aliases}`
"""
    (output_dir / "report.md").write_text(text)


def main() -> int:
    args = build_parser().parse_args()
    args.n_values = parse_int_list(args.n_values)
    args.inner_steps = parse_int_list(args.inner_steps)
    args.landscape_n_values = parse_int_list(args.landscape_n_values)
    args.matching_n_values = (
        args.n_values if args.matching_n_values is None else parse_int_list(args.matching_n_values)
    )
    validate_args(args)
    seed_everything(args.seed)
    inputs = experiment_inputs_from_args(args)
    resolved, skipped = resolve_experiment_inputs(
        inputs, include_incomplete=args.include_incomplete
    )
    if not resolved:
        reasons = "; ".join(f"{item.run_path}: {item.reason}" for item in skipped)
        raise ValueError(f"No checkpoints were selected. {reasons}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty; use --resume or a new directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    root_manifest: dict[str, Any] = {
        "status": "running",
        "evaluation": evaluation_config(args),
        "inputs": [json_safe(item.__dict__) for item in inputs],
        "resolved_runs": [json_safe(item.__dict__) for item in resolved],
        "skipped_runs": [json_safe(item.__dict__) for item in skipped],
        "torch_version": torch.__version__,
        "datasets_version": datasets.__version__,
        "device": str(device),
        "dtype": "float32",
    }
    write_json(args.output_dir / "manifest.json", root_manifest)

    loaded_datasets: dict[int, Any] = {}
    checkpoint_outputs: dict[tuple[str, int], Path] = {}
    tokenizer_digest: str | None = None
    checkpoint_manifests: list[dict[str, Any]] = []
    for spec in resolved:
        checkpoint_output, current_tokenizer_digest = evaluate_resolved_run(
            spec, args, loaded_datasets, device
        )
        if tokenizer_digest is None:
            tokenizer_digest = current_tokenizer_digest
        elif current_tokenizer_digest != tokenizer_digest:
            raise ValueError(
                "Selected checkpoints use incompatible tokenizers: "
                f"{spec.alias}/{spec.run_id} differs from earlier checkpoints"
            )
        checkpoint_outputs[(spec.alias, spec.seed)] = checkpoint_output
        checkpoint_manifests.append(json.loads((checkpoint_output / "manifest.json").read_text()))

    if args.cache_only:
        root_manifest.update({
            "status": "complete",
            "tokenizer_signature_sha256": tokenizer_digest,
            "checkpoints": checkpoint_manifests,
            "comparisons": [],
        })
        write_json(args.output_dir / "manifest.json", root_manifest)
        print(f"Checkpoint metric caches are complete: {args.output_dir}", flush=True)
        return 0

    alias_order = list(dict.fromkeys(item.alias for item in inputs))
    specs_by_alias = {
        alias: [spec for spec in resolved if spec.alias == alias]
        for alias in alias_order
        if any(spec.alias == alias for spec in resolved)
    }
    aggregate_by_alias = {
        alias: aggregate_alias_outputs(
            alias, specs, checkpoint_outputs, args.output_dir
        )
        for alias, specs in specs_by_alias.items()
    }

    comparison_results: dict[tuple[str, str], dict[str, Any]] = {}
    for reference_alias, candidate_alias in requested_comparisons(list(specs_by_alias)):
        comparison_results[(reference_alias, candidate_alias)] = build_alias_comparison(
            reference_alias,
            candidate_alias,
            specs_by_alias,
            checkpoint_outputs,
            args.output_dir,
            args,
        )

    generate_aggregate_plots(
        args.output_dir,
        [row for aggregate in aggregate_by_alias.values() for row in aggregate["task"]],
        [row for aggregate in aggregate_by_alias.values() for row in aggregate["interpolation"]],
        [row for aggregate in aggregate_by_alias.values() for row in aggregate["radial"]],
        [row for aggregate in aggregate_by_alias.values() for row in aggregate["contours"]],
        list(specs_by_alias),
    )
    generate_root_report(
        args.output_dir,
        resolved,
        skipped,
        aggregate_by_alias,
        comparison_results,
        args,
        device,
    )
    root_manifest.update({
        "status": "complete",
        "tokenizer_signature_sha256": tokenizer_digest,
        "checkpoints": checkpoint_manifests,
        "comparisons": [
            {
                "reference_alias": reference,
                "candidate_alias": candidate,
                **result["manifest"],
            }
            for (reference, candidate), result in comparison_results.items()
        ],
    })
    write_json(args.output_dir / "manifest.json", root_manifest)
    print(f"Evaluation complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
