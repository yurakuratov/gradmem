#!/usr/bin/env python3
"""Analyze GradMem failure modes and trajectory losses on mixed-orientation KV retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from analyze_mix_write_components import PAIR_RE, collate_diagnostics, token_components
from evaluate_kv_checkpoints import (
    best_checkpoint_for_run,
    load_model,
    load_run_args,
    move_to_device,
    resolve_device,
)
from kv_dataset_utils import BASE_KV_ALPHABET


SUCCESS_EM = 0.90
EVAL_RE = re.compile(r"^eval_(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--success-em", type=float, default=SUCCESS_EM)
    parser.add_argument("--run-pattern", help="Optional regular expression matched against relative run IDs")
    args = parser.parse_args()
    if args.max_examples <= 0 or args.batch_size <= 0:
        parser.error("--max-examples and --batch-size must be positive")
    if not 0.0 <= args.success_em <= 1.0:
        parser.error("--success-em must be in [0, 1]")
    return args


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def trainer_states(run_dir: Path) -> list[dict[str, Any]]:
    paths = [run_dir / "trainer_state.json", *run_dir.glob("checkpoint-*/trainer_state.json")]
    return [value for path in paths if (value := read_json(path)) is not None]


def latest_history(run_dir: Path) -> list[dict[str, Any]]:
    states = trainer_states(run_dir)
    if not states:
        return []
    state = max(states, key=lambda value: int(value.get("global_step", 0)))
    return [row for row in state.get("log_history", []) if isinstance(row, dict)]


def training_precision(args: dict[str, Any], run_dir: Path) -> str:
    if args.get("bf16") or run_dir.name.endswith("_bf16"):
        return "bf16"
    if args.get("fp16") or run_dir.name.endswith("_fp16"):
        return "fp16"
    return "fp32"


def hp_label(args: dict[str, Any], precision: str) -> str:
    head = "whead" if args.get("use_write_head") else "basehead"
    init = "pretrained" if args.get("pretrained_model") or args.get("init_checkpoint") else "scratch"
    return (
        f"D{args.get('n_embd')}_K{args.get('K')}_ilr{args.get('inner_lr')}_"
        f"{head}_align{args.get('step_alignment_weight', 0.0)}_"
        f"iread{args.get('intermediate_read_weight', 0.0)}_"
        f"lip{args.get('lipschitz_weight', 0.0)}_{init}_{precision}"
    )


def discover_runs(runs_root: Path, success_em: float, run_pattern: str | None = None) -> list[dict[str, Any]]:
    records = []
    pattern = re.compile(run_pattern) if run_pattern else None
    for config_path in sorted(runs_root.glob("gradmem*/run*/config.json")):
        run_dir = config_path.parent
        run_id = str(run_dir.relative_to(runs_root))
        if pattern is not None and pattern.search(run_id) is None:
            continue
        config = read_json(config_path) or {}
        args = config.get("cli_args", {})
        if not isinstance(args, dict) or args.get("write_objective", "reconstruction") != "reconstruction":
            continue
        history = latest_history(run_dir)
        precision = training_precision(args, run_dir)
        eval_rows = [row for row in history if "eval_exact_match" in row]
        best_em = max((float(row["eval_exact_match"]) for row in eval_rows), default=float("nan"))
        best_token = max((float(row.get("eval_token_accuracy", float("nan"))) for row in eval_rows), default=float("nan"))
        weights_path, selector, selector_value = best_checkpoint_for_run(run_dir)
        records.append({
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "hp_group": hp_label(args, precision),
            "seed": int(args.get("seed", -1)),
            "outcome": "successful" if best_em >= success_em else "unsuccessful",
            "success_threshold": success_em,
            "best_exact_match": best_em,
            "best_token_accuracy": best_token,
            "max_step": max((int(row.get("step", 0)) for row in history), default=0),
            "checkpoint": str(weights_path.parent),
            "checkpoint_selector": selector,
            "checkpoint_selector_value": selector_value,
            "n_embd": args.get("n_embd"),
            "K": args.get("K"),
            "inner_lr": args.get("inner_lr"),
            "use_write_head": bool(args.get("use_write_head", False)),
            "step_alignment_weight": float(args.get("step_alignment_weight", 0.0) or 0.0),
            "intermediate_read_weight": float(args.get("intermediate_read_weight", 0.0) or 0.0),
            "lipschitz_weight": float(args.get("lipschitz_weight", 0.0) or 0.0),
            "precision": precision,
            "initialization": (
                "pretrained" if args.get("pretrained_model") or args.get("init_checkpoint") else "scratch"
            ),
        })
    if not records:
        raise ValueError(f"No GradMem runs found below {runs_root}")
    return records


def parse_pairs(row: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    direction = row["context"][:1]
    if direction not in {"F", "B"}:
        raise ValueError(f"Expected F/B context marker: {row['context']!r}")
    pairs = []
    for match in PAIR_RE.finditer(row["context"]):
        left, right = match.group(1), match.group(2)
        pairs.append((left, right) if direction == "F" else (right, left))
    if not pairs:
        raise ValueError(f"No key/value pairs in context: {row['context']!r}")
    return direction, pairs


def make_shared_first_value_row(row: dict[str, str]) -> dict[str, str]:
    """Keep keys fixed while making all values share the queried value's first character."""
    direction, pairs = parse_pairs(row)
    query_match = re.search(r"\?!([^:]+):", row["query"])
    if query_match is None:
        raise ValueError(f"Could not parse query: {row['query']!r}")
    query_key = query_match.group(1)
    target_value = row["target"].split("!", 1)[0]
    if len(target_value) != 2:
        raise ValueError(f"Expected a two-character value: {row['target']!r}")
    if query_key not in {key for key, _ in pairs}:
        raise ValueError(f"Queried key {query_key!r} is absent from context")

    common_first, target_second = target_value
    available = [char for char in BASE_KV_ALPHABET if char != target_second]
    replacement_values: dict[str, str] = {}
    distractor_index = 0
    for key, _ in pairs:
        if key == query_key:
            replacement_values[key] = target_value
        else:
            replacement_values[key] = common_first + available[distractor_index]
            distractor_index += 1

    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        key = left if direction == "F" else right
        value = replacement_values[key]
        return f"!{key}:{value}!" if direction == "F" else f"!{value}:{key}!"

    transformed = dict(row)
    transformed["context"] = PAIR_RE.sub(replace, row["context"])
    _, transformed_pairs = parse_pairs(transformed)
    values = [value for _, value in transformed_pairs]
    if len({value[1] for value in values}) != len(values):
        raise AssertionError("Controlled values do not have unique second characters")
    if {value[0] for value in values} != {common_first}:
        raise AssertionError("Controlled values do not share one first character")
    return transformed


def target_prediction(logits: torch.Tensor, labels: torch.Tensor, tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.size(1) == labels.size(1) + 1:
        predictions, aligned_labels = logits[:, :-1].argmax(dim=-1), labels
    elif logits.size(1) == labels.size(1):
        predictions, aligned_labels = logits[:, :-1].argmax(dim=-1), labels[:, 1:]
    else:
        raise ValueError(
            f"Unexpected READ alignment: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    score_mask = aligned_labels.ne(-100)
    for token in ("!", "|"):
        score_mask &= aligned_labels.ne(tokenizer.convert_tokens_to_ids(token))
    scored_predictions = []
    scored_labels = []
    for row in range(labels.size(0)):
        scored_predictions.append(predictions[row][score_mask[row]])
        scored_labels.append(aligned_labels[row][score_mask[row]])
    if any(value.numel() != 2 for value in scored_labels):
        raise ValueError("Failure taxonomy requires exactly two scored value tokens per example")
    return torch.stack(scored_predictions), torch.stack(scored_labels)


def error_category(first_correct: bool, second_correct: bool) -> str:
    if first_correct and second_correct:
        return "exact"
    if first_correct:
        return "first_correct_second_wrong"
    if second_correct:
        return "first_wrong_second_correct"
    return "both_wrong"


def per_example_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def read_at_memory(model: Any, backend: Any, memory_state: dict[str, Any], batch_ctx: dict[str, Any],
                   labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    read_batch = backend.build_read_inputs(memory_state, batch_ctx)
    model_kwargs = read_batch.get("model_kwargs", {})
    with backend.activation_context(memory_state), model._disable_write_lora():
        output = model.model(inputs_embeds=read_batch["inputs_embeds"], return_dict=True, **model_kwargs)
    logits = output.logits[
        :, read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"], :
    ]
    losses = model._compute_per_example_read_target_loss(logits, read_batch, labels)
    outer_gradient = torch.autograd.grad(losses.sum(), memory_state["mem_batch"], retain_graph=True)[0]
    return logits, losses, outer_gradient


def evaluate_condition(model: Any, tokenizer: Any, rows: list[dict[str, str]], condition: str,
                       run: dict[str, Any], batch_size: int, device: torch.device
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions_out: list[dict[str, Any]] = []
    trajectory_out: list[dict[str, Any]] = []
    for batch_start in range(0, len(rows), batch_size):
        row_batch = rows[batch_start:batch_start + batch_size]
        diagnostics = collate_diagnostics(row_batch, tokenizer)
        directions = diagnostics.pop("directions")
        key_mask = diagnostics.pop("key_mask").to(device)
        value_mask = diagnostics.pop("value_mask").to(device)
        batch = move_to_device(diagnostics, device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        backend = model.memory_backend_impl
        batch_ctx = backend.prepare_batch(
            input_ids["context_input_ids"], input_ids["query_input_ids"], model.model.config.pad_token_id
        )
        memory_state, _ = backend.init_memory_state(len(row_batch))
        opt_state: dict[str, dict[str, torch.Tensor]] = {}
        previous_gradient: torch.Tensor | None = None
        transition_cosines: list[torch.Tensor] = []
        final_logits = None
        final_outer_losses = None

        for step in range(model.K + 1):
            memory_state = {
                name: value.detach().requires_grad_(True) if isinstance(value, torch.Tensor) else value
                for name, value in memory_state.items()
            }
            with torch.enable_grad():
                logits, outer_losses, outer_gradient = read_at_memory(
                    model, backend, memory_state, batch_ctx, labels
                )
                if previous_gradient is not None:
                    transition_cosines.append(F.cosine_similarity(
                        previous_gradient.float().flatten(1), outer_gradient.float().flatten(1), dim=1
                    ))
                with backend.activation_context(memory_state):
                    write_batch = backend.build_write_inputs(memory_state, batch_ctx)
                    reconstruction, _, _ = token_components(model, write_batch)
                    inner_losses = per_example_mean(reconstruction, write_batch["mask"])
                    inner_params = backend.inner_params(memory_state)
                    gradients = torch.autograd.grad(inner_losses.sum(), inner_params, retain_graph=True)
                key_losses = per_example_mean(reconstruction, key_mask)
                value_losses = per_example_mean(reconstruction, value_mask)
                grad_norm = gradients[0].detach().float().flatten(1).norm(dim=1)

            for index in range(len(row_batch)):
                trajectory_out.append({
                    "run_id": run["run_id"],
                    "hp_group": run["hp_group"],
                    "outcome": run["outcome"],
                    "seed": run["seed"],
                    "checkpoint": run["checkpoint"],
                    "condition": condition,
                    "example_id": batch_start + index,
                    "direction": directions[index],
                    "step": step,
                    "inner_loss": float(inner_losses[index].detach().cpu()),
                    "key_loss": float(key_losses[index].detach().cpu()),
                    "value_loss": float(value_losses[index].detach().cpu()),
                    "outer_loss": float(outer_losses[index].detach().cpu()),
                    "inner_grad_norm": float(grad_norm[index].cpu()),
                    "incoming_step_alignment_cosine": (
                        float(transition_cosines[-1][index].detach().cpu()) if transition_cosines else float("nan")
                    ),
                })

            if step == model.K:
                final_cosine = F.cosine_similarity(
                    gradients[0].detach().float().flatten(1),
                    outer_gradient.detach().float().flatten(1),
                    dim=1,
                )
                alignment_terms = [*transition_cosines, final_cosine]
                alignment_loss = 1.0 - torch.stack(alignment_terms).mean(dim=0)
                final_logits = logits.detach()
                final_outer_losses = outer_losses.detach()
                final_alignment_loss = alignment_loss.detach()
                break

            updated = []
            for parameter_index, (parameter, gradient) in enumerate(zip(inner_params, gradients)):
                if model.use_adam:
                    value = model._adam_step(
                        parameter, gradient, opt_state.setdefault(str(parameter_index), {}), step + 1, model.lr
                    )
                else:
                    value = model._sgd_step(
                        parameter, gradient, clip_value=model.inner_clip_value, clip_norm=model.inner_clip_norm
                    )
                updated.append(value.detach())
            previous_gradient = gradients[0].detach()
            backend.assign_inner_params(memory_state, updated)

        assert final_logits is not None and final_outer_losses is not None
        predicted_ids, label_ids = target_prediction(final_logits, labels, tokenizer)
        for index, row in enumerate(row_batch):
            first_correct = bool(predicted_ids[index, 0].eq(label_ids[index, 0]))
            second_correct = bool(predicted_ids[index, 1].eq(label_ids[index, 1]))
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[index].tolist())
            label_tokens = tokenizer.convert_ids_to_tokens(label_ids[index].tolist())
            _, pairs = parse_pairs(row)
            query_key = re.search(r"\?!([^:]+):", row["query"]).group(1)
            distractor_seconds = {value[1] for key, value in pairs if key != query_key}
            predictions_out.append({
                "run_id": run["run_id"],
                "hp_group": run["hp_group"],
                "outcome": run["outcome"],
                "seed": run["seed"],
                "checkpoint": run["checkpoint"],
                "condition": condition,
                "example_id": batch_start + index,
                "direction": directions[index],
                "target_first": label_tokens[0],
                "target_second": label_tokens[1],
                "predicted_first": predicted_tokens[0],
                "predicted_second": predicted_tokens[1],
                "first_correct": first_correct,
                "second_correct": second_correct,
                "exact_match": first_correct and second_correct,
                "error_category": error_category(first_correct, second_correct),
                "wrong_second_is_distractor": (
                    not second_correct and predicted_tokens[1] in distractor_seconds
                ),
                "outer_loss": float(final_outer_losses[index].cpu()),
                "step_alignment_loss": float(final_alignment_loss[index].cpu()),
            })
    return predictions_out, trajectory_out


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def grouped_views(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for hp_group, hp_frame in frame.groupby("hp_group", sort=True):
        yield hp_group, "all", hp_frame
        for outcome, outcome_frame in hp_frame.groupby("outcome", sort=True):
            yield hp_group, outcome, outcome_frame


def summarize_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    taxonomy_rows = []
    token_rows = []
    for hp_group, outcome_group, group in grouped_views(frame):
        for (condition, direction), cell in group.groupby(["condition", "direction"], sort=True):
            counts = Counter(cell["error_category"])
            for category in ("exact", "first_correct_second_wrong", "first_wrong_second_correct", "both_wrong"):
                taxonomy_rows.append({
                    "hp_group": hp_group,
                    "outcome_group": outcome_group,
                    "condition": condition,
                    "direction": direction,
                    "error_category": category,
                    "count": counts[category],
                    "fraction": counts[category] / len(cell),
                    "example_evaluations": len(cell),
                    "run_count": cell["run_id"].nunique(),
                })
            token_rows.append({
                "hp_group": hp_group,
                "outcome_group": outcome_group,
                "condition": condition,
                "direction": direction,
                "first_token_accuracy": cell["first_correct"].mean(),
                "second_token_accuracy": cell["second_correct"].mean(),
                "exact_match": cell["exact_match"].mean(),
                "second_accuracy_given_first_correct": cell.loc[cell.first_correct, "second_correct"].mean(),
                "second_accuracy_given_first_wrong": cell.loc[~cell.first_correct, "second_correct"].mean(),
                "wrong_second_distractor_rate": cell["wrong_second_is_distractor"].mean(),
                "example_evaluations": len(cell),
                "run_count": cell["run_id"].nunique(),
            })
    return pd.DataFrame(taxonomy_rows), pd.DataFrame(token_rows)


def summarize_frozen_correlations(predictions: pd.DataFrame, trajectory: pd.DataFrame) -> pd.DataFrame:
    pivot = trajectory.pivot_table(
        index=["run_id", "hp_group", "outcome", "seed", "condition", "example_id", "direction"],
        columns="step",
        values=["inner_loss", "key_loss", "value_loss", "outer_loss"],
    )
    pivot.columns = [f"{metric}_step{step}" for metric, step in pivot.columns]
    pivot = pivot.reset_index()
    final_steps = trajectory.groupby("run_id")["step"].max().to_dict()
    rows = []
    for _, row in pivot.iterrows():
        final = int(final_steps[row["run_id"]])
        row["inner_loss_delta"] = row[f"inner_loss_step{final}"] - row["inner_loss_step0"]
        row["key_loss_delta"] = row[f"key_loss_step{final}"] - row["key_loss_step0"]
        row["value_loss_delta"] = row[f"value_loss_step{final}"] - row["value_loss_step0"]
        row["outer_loss_final"] = row[f"outer_loss_step{final}"]
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide = wide.merge(
        predictions[["run_id", "condition", "example_id", "step_alignment_loss", "exact_match"]],
        on=["run_id", "condition", "example_id"],
        how="left",
    )
    pairs = [
        ("step_alignment_loss", "outer_loss_final"),
        ("step_alignment_loss", "inner_loss_delta"),
        ("step_alignment_loss", "key_loss_delta"),
        ("step_alignment_loss", "value_loss_delta"),
        ("inner_loss_step0", "outer_loss_final"),
        ("inner_loss_delta", "outer_loss_final"),
        ("key_loss_step0", "outer_loss_final"),
        ("key_loss_delta", "outer_loss_final"),
        ("value_loss_step0", "outer_loss_final"),
        ("value_loss_delta", "outer_loss_final"),
    ]
    output = []
    for hp_group, outcome_group, group in grouped_views(wide):
        for condition, cell in group.groupby("condition", sort=True):
            for x_name, y_name in pairs:
                for method in ("pearson", "spearman"):
                    output.append({
                        "hp_group": hp_group,
                        "outcome_group": outcome_group,
                        "condition": condition,
                        "x": x_name,
                        "y": y_name,
                        "method": method,
                        "correlation": correlation(cell[x_name], cell[y_name], method),
                        "example_evaluations": len(cell),
                        "run_count": cell["run_id"].nunique(),
                    })
    return pd.DataFrame(output)


def extract_training_history(runs: list[dict[str, Any]]) -> pd.DataFrame:
    output = []
    for run in runs:
        for row in latest_history(Path(run["run_dir"])):
            if "eval_exact_match" not in row:
                continue
            output.append({
                "run_id": run["run_id"],
                "hp_group": run["hp_group"],
                "outcome": run["outcome"],
                "seed": run["seed"],
                **{key: value for key, value in row.items() if key == "step" or EVAL_RE.match(key)},
            })
    return pd.DataFrame(output)


def summarize_training_correlations(history: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("eval_step_alignment_loss", "eval_inner_loss"),
        ("eval_step_alignment_loss", "eval_outer_loss"),
        ("eval_step_alignment_loss", "eval_target_loss"),
        ("eval_inner_loss", "eval_outer_loss"),
        ("eval_inner_loss", "eval_target_loss"),
    ]
    output = []
    for run_id, group in history.groupby("run_id", sort=True):
        hp_group = group.hp_group.iloc[0]
        outcome_group = group.outcome.iloc[0]
        for x_name, y_name in pairs:
            if x_name not in group or y_name not in group:
                continue
            for method in ("pearson", "spearman"):
                output.append({
                    "scope": "run",
                    "run_id": run_id,
                    "hp_group": hp_group,
                    "outcome_group": outcome_group,
                    "x": x_name,
                    "y": y_name,
                    "method": method,
                    "correlation": correlation(group[x_name], group[y_name], method),
                    "evaluation_points": len(group),
                    "run_count": 1,
                })
    for hp_group, outcome_group, group in grouped_views(history):
        for x_name, y_name in pairs:
            if x_name not in group or y_name not in group:
                continue
            for method in ("pearson", "spearman"):
                output.append({
                    "scope": "group",
                    "run_id": "",
                    "hp_group": hp_group,
                    "outcome_group": outcome_group,
                    "x": x_name,
                    "y": y_name,
                    "method": method,
                    "correlation": correlation(group[x_name], group[y_name], method),
                    "evaluation_points": len(group),
                    "run_count": group["run_id"].nunique(),
                })
    return pd.DataFrame(output)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    runs = discover_runs(args.runs_root.resolve(), args.success_em, args.run_pattern)
    manifest = pd.DataFrame(runs)
    write_frame(manifest, args.output_dir / "checkpoints.csv")

    history = extract_training_history(runs)
    write_frame(history, args.output_dir / "training_history.csv")
    write_frame(summarize_training_correlations(history), args.output_dir / "training_correlations.csv")

    validation = datasets.load_from_disk(str(args.data_path))["valid"]
    clean_rows = [dict(validation[index]) for index in range(min(args.max_examples, len(validation)))]
    controlled_rows = [make_shared_first_value_row(row) for row in clean_rows]
    with (args.output_dir / "controlled_contexts.jsonl").open("w") as output:
        for example_id, (clean, controlled) in enumerate(zip(clean_rows, controlled_rows)):
            output.write(json.dumps({
                "example_id": example_id,
                "clean": clean,
                "shared_first_value": controlled,
            }) + "\n")

    prediction_rows = []
    trajectory_rows = []
    for run_index, run in enumerate(runs, start=1):
        print(f"[{run_index}/{len(runs)}] {run['run_id']} ({run['outcome']})", flush=True)
        checkpoint = Path(run["checkpoint"])
        run_args = load_run_args(checkpoint)
        tokenizer_path = run_args.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(f"No tokenizer_path for {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model, _, _ = load_model(checkpoint, device)
        if model.memory_backend != "prefix" or model.write_objective != "reconstruction":
            raise ValueError(f"Expected prefix reconstruction GradMem: {checkpoint}")
        for condition, rows in (("clean", clean_rows), ("shared_first_value", controlled_rows)):
            predictions, trajectory = evaluate_condition(
                model, tokenizer, rows, condition, run, args.batch_size, device
            )
            prediction_rows.extend(predictions)
            trajectory_rows.extend(trajectory)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.DataFrame(prediction_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    write_frame(predictions, args.output_dir / "per_example_predictions.csv")
    write_frame(trajectory, args.output_dir / "trajectory_losses.csv")
    taxonomy, token_summary = summarize_predictions(predictions)
    write_frame(taxonomy, args.output_dir / "failure_taxonomy.csv")
    write_frame(token_summary, args.output_dir / "token_summary.csv")
    write_frame(
        summarize_frozen_correlations(predictions, trajectory),
        args.output_dir / "frozen_correlations.csv",
    )
    print(f"Wrote GradMem failure analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
