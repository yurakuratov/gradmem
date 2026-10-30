#!/usr/bin/env python3
"""Evaluate KV-retrieval checkpoints and export one validation-metrics row per checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping

import datasets
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from grad_memgpt import GradMemGPT, GradMemGPTConfig
from run_gradmemgpt_on_kv_retrieval import TRAIN_COMPONENT_KEYS, collate_fn


EXPECTED_MISSING_CHECKPOINT_KEYS = {
    "model.lm_head.weight",
    "energy_memory_search_gain_ema",
    "energy_memory_search_gain_ema_initialized",
}
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint directory or weights file.")
    parser.add_argument("--checkpoints-file", type=Path, help="Text file with one checkpoint path per line.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run directory, or an experiment directory containing multiple runs. Selects each run's best checkpoint.",
    )
    parser.add_argument("--runs-file", type=Path, help="Text file with one --run path per line.")
    parser.add_argument("--data-path", type=Path, default=Path("./data/N2-K4V4-S4(32-64)_1M"))
    parser.add_argument("--tokenizer-path", type=Path, help="Override the tokenizer recorded by each run.")
    parser.add_argument("--max-context-length", type=int, help="Override each run's max_context_length.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--estimate-local-lipschitz",
        action="store_true",
        help="Estimate the WRITE objective's local memory-space Lipschitz coefficient at every WRITE step.",
    )
    parser.add_argument(
        "--lipschitz-max-examples",
        type=int,
        help="Use only the first N validation examples for the Lipschitz probe (default: all).",
    )
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true", help="Stop instead of writing an error row.")
    args = parser.parse_args()
    if args.checkpoints_file:
        args.checkpoint.extend(
            line.strip() for line in args.checkpoints_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.runs_file:
        args.run.extend(
            line.strip() for line in args.runs_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not args.checkpoint and not args.run:
        parser.error("provide at least one --checkpoint, --checkpoints-file, --run, or --runs-file")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.lipschitz_max_examples is not None and args.lipschitz_max_examples <= 0:
        parser.error("--lipschitz-max-examples must be positive")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def resolve_weights_path(checkpoint: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint.parent, checkpoint
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weights_path = checkpoint / filename
        if weights_path.exists():
            return checkpoint, weights_path
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint}")


def load_run_args(checkpoint_dir: Path) -> dict[str, Any]:
    for directory in (checkpoint_dir, *checkpoint_dir.parents):
        path = directory / "config.json"
        if not path.exists():
            continue
        content = json.loads(path.read_text())
        if isinstance(content.get("cli_args"), dict):
            return content["cli_args"]
    return {}


def is_run_directory(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        return isinstance(json.loads(config_path.read_text()).get("cli_args"), dict)
    except json.JSONDecodeError:
        return False


def discover_run_directories(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if is_run_directory(path):
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Run path does not exist: {path}")
    runs = sorted(candidate for candidate in path.iterdir() if candidate.is_dir() and is_run_directory(candidate))
    if not runs:
        raise ValueError(f"Expected a run directory or a directory containing runs: {path}")
    return runs


def checkpoint_directories(run_path: Path) -> list[Path]:
    return sorted(
        (path for path in run_path.iterdir() if path.is_dir() and CHECKPOINT_DIR_RE.fullmatch(path.name)),
        key=lambda path: int(CHECKPOINT_DIR_RE.fullmatch(path.name).group(1)),
    )


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def best_checkpoint_for_run(run_path: Path) -> tuple[Path, str, float | None]:
    """Resolve Trainer's recorded best checkpoint, or rank saved evaluation histories."""
    run_args = load_run_args(run_path)
    states = [load_json(run_path / "trainer_state.json")]
    states.extend(load_json(path / "trainer_state.json") for path in reversed(checkpoint_directories(run_path)))
    for state in states:
        if not state or not state.get("best_model_checkpoint"):
            continue
        checkpoint = run_path / Path(str(state["best_model_checkpoint"])).name
        try:
            _, weights_path = resolve_weights_path(checkpoint)
        except FileNotFoundError:
            continue
        return weights_path, str(run_args.get("metric_for_best_model", "token_accuracy")), state.get("best_metric")

    metric_name = str(run_args.get("metric_for_best_model", "token_accuracy"))
    metric_key = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
    higher_is_better = bool(run_args.get("greater_is_better", True))
    candidates: list[tuple[float, Path]] = []
    for checkpoint in checkpoint_directories(run_path):
        state = load_json(checkpoint / "trainer_state.json")
        if not state:
            continue
        values = [entry[metric_key] for entry in state.get("log_history", []) if metric_key in entry]
        if values:
            candidates.append((float(values[-1]), checkpoint))
    if candidates:
        score, checkpoint = (max if higher_is_better else min)(candidates, key=lambda item: item[0])
        _, weights_path = resolve_weights_path(checkpoint)
        return weights_path, metric_name, score
    checkpoints = checkpoint_directories(run_path)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in run {run_path}")
    _, weights_path = resolve_weights_path(checkpoints[-1])
    return weights_path, metric_name, None


def reconstruct_base_config(config: GradMemGPTConfig) -> None:
    if not isinstance(config.base_config, dict):
        return
    serialized = dict(config.base_config)
    model_type = serialized.pop("model_type", None)
    if not model_type:
        raise ValueError("Serialized base_config has no model_type")
    config.base_config = AutoConfig.for_model(model_type, **serialized)


def load_model(checkpoint: Path, device: torch.device) -> tuple[GradMemGPT, Path, dict[str, Any]]:
    checkpoint_dir, weights_path = resolve_weights_path(checkpoint)
    config = GradMemGPTConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    reconstruct_base_config(config)
    config.energy_memory_search_weight = 0.0
    config.lipschitz_weight = 0.0
    model = GradMemGPT(config)
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(str(weights_path), device="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    if not set(missing).issubset(EXPECTED_MISSING_CHECKPOINT_KEYS):
        raise ValueError(f"Unexpected missing checkpoint keys: {sorted(missing)}")
    model.tie_weights()
    model.to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_dir, load_run_args(checkpoint_dir)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def _write_objective_for_state(
    model: GradMemGPT,
    backend: Any,
    memory_state: dict[str, Any],
    batch_ctx: dict[str, Any],
) -> torch.Tensor:
    write_batch = backend.build_write_inputs(memory_state, batch_ctx)
    with backend.activation_context(memory_state):
        if model.write_objective == "reconstruction":
            outputs, reconstruction = model._run_reconstruction_write_forward(write_batch)
            objective = reconstruction
        else:
            outputs, reconstruction, energy = model._run_energy_reconstruction_write_forward(write_batch)
            if model.write_objective == "energy":
                objective = energy
            elif model.write_objective == "energy_with_reconstruction":
                objective = (
                    model.write_reconstruction_weight * reconstruction
                    + model.write_energy_weight * energy
                )
            else:
                raise ValueError(f"Unsupported write objective: {model.write_objective}")
    del outputs
    return objective


def local_lipschitz_trajectory_batch(
    model: GradMemGPT,
    input_ids: Mapping[str, torch.Tensor],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Replay WRITE and return pointwise gradient norms and trajectory secants.

    For scalar WRITE objective f_C(M), ||grad_M f_C(M_k)||_2 is its pointwise
    local Lipschitz coefficient under the Frobenius/L2 memory norm. Secants are
    observed lower bounds on a Lipschitz constant over each traversed segment.
    """
    if model.memory_backend != "prefix":
        raise ValueError("Local Lipschitz estimation currently requires memory_backend='prefix'")

    context_input_ids = input_ids["context_input_ids"]
    query_input_ids = input_ids["query_input_ids"]
    batch_size = context_input_ids.size(0)
    backend = model.memory_backend_impl
    batch_ctx = backend.prepare_batch(
        context_input_ids,
        query_input_ids,
        model.model.config.pad_token_id,
    )
    memory_state, _ = backend.init_memory_state(batch_size)
    opt_state: dict[str, dict[str, torch.Tensor]] = {}
    gradient_norms: list[np.ndarray] = []
    secants: list[np.ndarray] = []
    previous_objective: torch.Tensor | None = None
    previous_memory: torch.Tensor | None = None

    for step in range(model.K + 1):
        # Leaf states keep the diagnostic independent of training-time graph mode.
        for name, value in list(memory_state.items()):
            if isinstance(value, torch.Tensor):
                memory_state[name] = value.detach().requires_grad_(True)

        with torch.enable_grad():
            objective = _write_objective_for_state(model, backend, memory_state, batch_ctx)
            inner_params = backend.inner_params(memory_state)
            grads = torch.autograd.grad(objective.sum(), inner_params)

        memory = memory_state["mem_batch"]
        memory_grad = grads[0]
        gradient_norms.append(memory_grad.detach().float().flatten(1).norm(dim=1).cpu().numpy())
        if previous_objective is not None and previous_memory is not None:
            objective_delta = (objective.detach() - previous_objective).abs().float()
            memory_delta = (memory.detach() - previous_memory).float().flatten(1).norm(dim=1)
            eps = torch.finfo(memory_delta.dtype).eps
            secants.append((objective_delta / memory_delta.clamp_min(eps)).cpu().numpy())

        if step == model.K:
            break
        previous_objective = objective.detach()
        previous_memory = memory.detach()
        new_params = []
        for index, (parameter, gradient) in enumerate(zip(inner_params, grads)):
            if model.use_adam:
                updated = model._adam_step(
                    parameter,
                    gradient,
                    opt_state.setdefault(str(index), {}),
                    step + 1,
                    model.lr,
                )
            else:
                updated = model._sgd_step(
                    parameter,
                    gradient,
                    clip_value=model.inner_clip_value,
                    clip_norm=model.inner_clip_norm,
                )
            new_params.append(updated.detach())
        backend.assign_inner_params(memory_state, new_params)

    return gradient_norms, secants


def summarize_local_lipschitz(
    gradient_norms: list[list[np.ndarray]],
    secants: list[list[np.ndarray]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for step, batches in enumerate(gradient_norms):
        values = np.concatenate(batches).astype(np.float64)
        metrics.update({
            f"lipschitz_grad_k{step}_mean": float(values.mean()),
            f"lipschitz_grad_k{step}_median": float(np.median(values)),
            f"lipschitz_grad_k{step}_p95": float(np.quantile(values, 0.95)),
            f"lipschitz_grad_k{step}_max": float(values.max()),
        })
    for step, batches in enumerate(secants, start=1):
        values = np.concatenate(batches).astype(np.float64)
        metrics[f"lipschitz_secant_k{step}_mean"] = float(values.mean())
        metrics[f"lipschitz_secant_k{step}_max"] = float(values.max())
    return metrics


def estimate_local_lipschitz(
    model: GradMemGPT,
    dataset: Any,
    tokenizer: Any,
    batch_size: int,
    max_context_length: int | None,
    device: torch.device,
    max_examples: int | None,
) -> dict[str, float]:
    if model.memory_backend != "prefix":
        raise ValueError("Local Lipschitz estimation currently requires memory_backend='prefix'")
    count = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    if count == 0:
        raise ValueError("Validation dataset is empty")
    collator = partial(collate_fn, tokenizer=tokenizer, max_context_length=max_context_length)
    loader = DataLoader(dataset.select(range(count)), batch_size=batch_size, collate_fn=collator, shuffle=False)
    gradient_norms: list[list[np.ndarray]] = [[] for _ in range(model.K + 1)]
    secants: list[list[np.ndarray]] = [[] for _ in range(model.K)]
    for batch in loader:
        batch = move_to_device(batch, device)
        batch_gradients, batch_secants = local_lipschitz_trajectory_batch(model, batch["input_ids"])
        for step, values in enumerate(batch_gradients):
            gradient_norms[step].append(values)
        for step, values in enumerate(batch_secants):
            secants[step].append(values)
    return {
        "lipschitz_num_examples": float(count),
        **summarize_local_lipschitz(gradient_norms, secants),
    }


def evaluate_checkpoint(
    model: GradMemGPT,
    dataset: Any,
    tokenizer: Any,
    batch_size: int,
    max_context_length: int | None,
    device: torch.device,
) -> dict[str, float]:
    collator = partial(collate_fn, tokenizer=tokenizer, max_context_length=max_context_length)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("!", "|")]
    correct_tokens = 0
    scored_tokens = 0
    exact_matches = 0
    examples = 0
    loss_sum = 0.0
    batch_metric_values: dict[str, list[float]] = defaultdict(list)

    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.no_grad():
            output = model(batch["input_ids"], labels=batch["labels"])
        logits = output["predictions"]
        labels = batch["labels"]
        if logits.size(1) == labels.size(1) + 1:
            predictions = logits[:, :-1].argmax(dim=-1)
            aligned_labels = labels
        elif logits.size(1) == labels.size(1):
            predictions = logits[:, :-1].argmax(dim=-1)
            aligned_labels = labels[:, 1:]
        else:
            raise ValueError(
                f"Unexpected prediction/label lengths: {logits.size(1)} and {labels.size(1)}"
            )
        mask = aligned_labels.ne(-100)
        for token_id in ignore_token_ids:
            mask &= aligned_labels.ne(token_id)
        correct = predictions.eq(aligned_labels) & mask
        per_example_counts = mask.sum(dim=1)
        if torch.any(per_example_counts == 0):
            raise ValueError("Validation example has no scored target tokens")
        correct_tokens += int(correct.sum())
        scored_tokens += int(mask.sum())
        exact_matches += int((correct.sum(dim=1) == per_example_counts).sum())
        examples += labels.size(0)
        loss_sum += scalar(output["loss"]) * labels.size(0)
        for key, value in output["inner_loop_stats"].items():
            batch_metric_values[key].append(scalar(value))

    if examples == 0 or scored_tokens == 0:
        raise ValueError("Validation dataset is empty or has no scored tokens")
    metrics = {
        "eval_loss": loss_sum / examples,
        "token_accuracy": correct_tokens / scored_tokens,
        "exact_match": exact_matches / examples,
    }
    # Keep this list and the aggregate names aligned with compute_metrics_fn in
    # run_gradmemgpt_on_kv_retrieval.py.
    runner_stat_names = {
        "inner_loss": "inner_loss",
        "inner_grad_norm_mean": "inner_grad_norm",
        "inner_grad_norm_max": "inner_grad_norm_max",
        "inner_grad_norm_min": "inner_grad_norm_min",
        "mem_norm_mean": "mem_norm_mean",
        "mem_norm_max": "mem_norm_max",
        "mem_norm_min": "mem_norm_min",
        "delta_mem_norm_mean": "delta_mem_norm_mean",
        "delta_mem_norm_max": "delta_mem_norm_max",
        "delta_mem_norm_min": "delta_mem_norm_min",
    }
    optional_runner_stats = [
        "inner_loss_after_write",
        "inner_loss_initial",
        "inner_loss_write_delta",
        "inner_reconstruction_loss",
        "inner_energy_loss",
        "inner_reconstruction_loss_after_write",
        "inner_energy_loss_after_write",
        "write_reconstruction_weight",
        "write_energy_weight",
        *TRAIN_COMPONENT_KEYS,
        "mem_attn_read",
    ]
    for source_name, metric_name in runner_stat_names.items():
        if source_name in batch_metric_values:
            metrics[metric_name] = float(np.mean(batch_metric_values[source_name]))
    for key in optional_runner_stats:
        if key in batch_metric_values:
            metrics[key] = float(np.mean(batch_metric_values[key]))
    return metrics


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def xlsx_cell(reference: str, value: Any, style: int = 0) -> str:
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{reference}"{style_attr}/>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f'<c r="{reference}"{style_attr}><v>{numeric:.16g}</v></c>'
    escaped = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<c r="{reference}" t="inlineStr"{style_attr}><is><t>{escaped}</t></is></c>'


def write_xlsx(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        columns = ["status"]
    worksheet_rows = []
    header = {column: column for column in columns}
    for row_number, row in enumerate([header] + rows, start=1):
        cells = "".join(
            xlsx_cell(f"{excel_column_name(column_number)}{row_number}", row.get(column), style=1 if row_number == 1 else 0)
            for column_number, column in enumerate(columns, start=1)
        )
        worksheet_rows.append(f'<row r="{row_number}">{cells}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + "".join(worksheet_rows) + "</sheetData></worksheet>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>""")
        archive.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>""")
        archive.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="metrics" sheetId="1" r:id="rId1"/></sheets></workbook>""")
        archive.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>""")
        archive.writestr("xl/styles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="2"><xf xfId="0"/><xf xfId="0" fontId="1" applyFont="1"/></cellXfs></styleSheet>""")
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = datasets.load_from_disk(str(args.data_path))["valid"]
    evaluation_targets = [
        {"checkpoint": Path(checkpoint), "selection": "explicit", "selection_metric": None,
         "selection_metric_value": None, "run": None}
        for checkpoint in args.checkpoint
    ]
    for run_arg in args.run:
        for run_path in discover_run_directories(Path(run_arg)):
            checkpoint, metric_name, metric_value = best_checkpoint_for_run(run_path)
            evaluation_targets.append({
                "checkpoint": checkpoint,
                "selection": "best",
                "selection_metric": metric_name,
                "selection_metric_value": metric_value,
                "run": str(run_path),
            })
    rows = []
    for target in evaluation_targets:
        checkpoint_arg = target["checkpoint"]
        row: dict[str, Any] = {
            "run": target["run"],
            "checkpoint": str(Path(checkpoint_arg).expanduser()),
            "selection": target["selection"],
            "selection_metric": target["selection_metric"],
            "selection_metric_value": target["selection_metric_value"],
        }
        try:
            model, checkpoint_dir, run_args = load_model(Path(checkpoint_arg), device)
            tokenizer_path = args.tokenizer_path or run_args.get("tokenizer_path")
            if not tokenizer_path:
                raise ValueError("No tokenizer path; pass --tokenizer-path or evaluate a checkpoint within a run")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            max_context_length = args.max_context_length
            if max_context_length is None:
                max_context_length = run_args.get("max_context_length")
            row.update({
                "checkpoint": str(checkpoint_dir),
                "status": "ok",
                "K": model.K,
                "inner_lr": model.lr,
                "write_objective": model.write_objective,
                "memory_backend": model.memory_backend,
            })
            row.update(evaluate_checkpoint(
                model, dataset, tokenizer, args.batch_size, max_context_length, device
            ))
            if args.estimate_local_lipschitz:
                row.update(estimate_local_lipschitz(
                    model,
                    dataset,
                    tokenizer,
                    args.batch_size,
                    max_context_length,
                    device,
                    args.lipschitz_max_examples,
                ))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:
            if args.fail_fast:
                raise
            row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        rows.append(row)
    write_xlsx(args.output_xlsx, rows)
    print(f"Wrote {len(rows)} checkpoint rows to {args.output_xlsx}")


if __name__ == "__main__":
    main()
