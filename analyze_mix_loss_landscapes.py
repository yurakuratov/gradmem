#!/usr/bin/env python3
"""Probe inner and READ losses in transition-specific memory planes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluate_kv_checkpoints import (
    _write_objective_for_state,
    load_model,
    load_run_args,
    move_to_device,
    resolve_device,
)
from run_gradmemgpt_on_kv_retrieval import collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--alias", action="append", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--coordinate-limit", type=float, default=3.0)
    parser.add_argument("--transition-points", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-grid-csv", type=Path, required=True)
    parser.add_argument("--output-trajectory-csv", type=Path, required=True)
    args = parser.parse_args()
    if len(args.alias) != len(args.checkpoint):
        parser.error("provide exactly one --alias per --checkpoint")
    if args.max_examples <= 0 or args.grid_size < 3 or args.candidate_batch_size <= 0:
        parser.error("invalid positive example/grid/batch setting")
    if args.coordinate_limit <= 0:
        parser.error("--coordinate-limit must be positive")
    if args.transition_points[0] != 0 or any(
        right <= left for left, right in zip(args.transition_points, args.transition_points[1:])
    ):
        parser.error("--transition-points must start at 0 and be strictly increasing")
    return args


def replay_trajectory(model: Any, input_ids: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    backend = model.memory_backend_impl
    context_ids = input_ids["context_input_ids"]
    query_ids = input_ids["query_input_ids"]
    batch_ctx = backend.prepare_batch(context_ids, query_ids, model.model.config.pad_token_id)
    state, _ = backend.init_memory_state(context_ids.size(0))
    optimizer_state: dict[str, dict[str, torch.Tensor]] = {}
    memories: list[torch.Tensor] = []
    objectives: list[torch.Tensor] = []
    gradient_norms: list[torch.Tensor] = []

    for step in range(model.K + 1):
        state = {
            name: value.detach().requires_grad_(True) if isinstance(value, torch.Tensor) else value
            for name, value in state.items()
        }
        with torch.enable_grad():
            objective = _write_objective_for_state(model, backend, state, batch_ctx)
            parameters = backend.inner_params(state)
            gradients = torch.autograd.grad(objective.sum(), parameters)
        memories.append(state["mem_batch"].detach().clone())
        objectives.append(objective.detach())
        gradient_norms.append(gradients[0].detach().float().flatten(1).norm(dim=1))
        if step == model.K:
            break
        updated = []
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            if model.use_adam:
                value = model._adam_step(
                    parameter,
                    gradient,
                    optimizer_state.setdefault(str(index), {}),
                    step + 1,
                    model.lr,
                )
            else:
                value = model._sgd_step(
                    parameter,
                    gradient,
                    clip_value=model.inner_clip_value,
                    clip_norm=model.inner_clip_norm,
                )
            updated.append(value.detach())
        backend.assign_inner_params(state, updated)
    return memories, objectives, gradient_norms


def step_basis(start: torch.Tensor, end: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    basis_x = (end - start).float()
    flat_x = basis_x.flatten(1)
    radius = flat_x.norm(dim=1)
    eps = torch.finfo(flat_x.dtype).eps
    if torch.any(radius <= eps):
        raise ValueError("Encountered a zero WRITE-step displacement")
    unit_x = flat_x / radius[:, None]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_flat = torch.randn(flat_x.shape, generator=generator, dtype=torch.float32).to(flat_x.device)
    random_flat -= (random_flat * unit_x).sum(dim=1, keepdim=True) * unit_x
    random_norm = random_flat.norm(dim=1)
    if torch.any(random_norm <= eps):
        raise ValueError("Encountered a degenerate random orthogonal direction")
    basis_y = (random_flat / random_norm[:, None] * radius[:, None]).reshape_as(basis_x)
    flat_y = basis_y.flatten(1)
    cosine = (flat_x * flat_y).sum(dim=1) / (flat_x.norm(dim=1) * flat_y.norm(dim=1))
    if torch.any(cosine.abs() > 1e-5):
        raise ValueError("Constructed plane axes are not orthogonal")
    return basis_x, basis_y, radius


@torch.no_grad()
def losses_at_candidates(model: Any, context_ids: torch.Tensor, query_ids: torch.Tensor,
                         labels: torch.Tensor, candidates: torch.Tensor,
                         candidate_batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    backend = model.memory_backend_impl
    inner_values = []
    read_values = []
    for start in range(0, candidates.size(0), candidate_batch_size):
        end = min(start + candidate_batch_size, candidates.size(0))
        candidate = candidates[start:end]
        context = context_ids[start:end]
        query = query_ids[start:end]
        batch_labels = labels[start:end]
        batch_ctx = backend.prepare_batch(context, query, model.model.config.pad_token_id)
        template, _ = backend.init_memory_state(candidate.size(0))
        state = dict(template)
        state["mem_batch"] = candidate
        inner = _write_objective_for_state(model, backend, state, batch_ctx)
        read_batch = backend.build_read_inputs(state, batch_ctx)
        with backend.activation_context(state):
            with model._disable_write_lora():
                output = model.model(
                    inputs_embeds=read_batch["inputs_embeds"],
                    return_dict=True,
                    **read_batch.get("model_kwargs", {}),
                )
        predictions = output.logits[
            :, read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"], :
        ]
        read = model._compute_per_example_read_target_loss(predictions, read_batch, batch_labels)
        inner_values.append(inner.detach().float().cpu())
        read_values.append(read.detach().float().cpu())
    return torch.cat(inner_values).numpy(), torch.cat(read_values).numpy()


def repeated_inputs(tensor: torch.Tensor, repetitions: int) -> torch.Tensor:
    return tensor.repeat((repetitions,) + (1,) * (tensor.ndim - 1))


def analyze_checkpoint(model: Any, alias: str, checkpoint_dir: Path, batch: dict[str, Any],
                       grid: np.ndarray, candidate_batch_size: int, seed: int,
                       transition_points: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    memories, objectives, gradient_norms = replay_trajectory(model, input_ids)
    batch_size = labels.size(0)
    coordinate_pairs = [(float(x), float(y)) for y in grid for x in grid]
    repetitions = len(coordinate_pairs)
    repeated_context = repeated_inputs(input_ids["context_input_ids"], repetitions)
    repeated_query = repeated_inputs(input_ids["query_input_ids"], repetitions)
    repeated_labels = repeated_inputs(labels, repetitions)
    grid_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    interval_radii: dict[int, torch.Tensor] = {}
    read_losses_by_step: dict[int, np.ndarray] = {}
    for interval_index, (start_step, end_step) in enumerate(zip(transition_points, transition_points[1:])):
        center = memories[start_step]
        next_memory = memories[end_step]
        basis_x, basis_y, radius = step_basis(center, next_memory, seed + interval_index)
        interval_radii[start_step] = radius
        candidate_blocks = [center + x * basis_x + y * basis_y for x, y in coordinate_pairs]
        candidates = torch.cat(candidate_blocks, dim=0)
        inner, read = losses_at_candidates(
            model,
            repeated_context,
            repeated_query,
            repeated_labels,
            candidates,
            candidate_batch_size,
        )
        inner = inner.reshape(repetitions, batch_size)
        read = read.reshape(repetitions, batch_size)
        center_index = coordinate_pairs.index((0.0, 0.0))
        center_inner = inner[center_index]
        center_read = read[center_index]
        read_losses_by_step[start_step] = center_read
        if not np.allclose(center_inner, objectives[start_step].float().cpu().numpy(), rtol=1e-4, atol=1e-3):
            raise ValueError(f"Grid center does not match trajectory objective at step {start_step}")
        next_step_index = coordinate_pairs.index((1.0, 0.0))
        expected_inner, _ = losses_at_candidates(
            model,
            input_ids["context_input_ids"],
            input_ids["query_input_ids"],
            labels,
            next_memory,
            candidate_batch_size,
        )
        if not np.allclose(inner[next_step_index], expected_inner, rtol=1e-4, atol=1e-3):
            raise ValueError(f"M{start_step}-centered coordinate (1,0) does not match M{end_step}")
        for index, (x, y) in enumerate(coordinate_pairs):
            grid_rows.append({
                "alias": alias,
                "checkpoint": str(checkpoint_dir),
                "step": start_step,
                "next_step": end_step,
                "x": x,
                "y": y,
                "inner_loss_mean": float(inner[index].mean()),
                "inner_loss_delta_mean": float((inner[index] - center_inner).mean()),
                "read_loss_mean": float(read[index].mean()),
                "read_loss_delta_mean": float((read[index] - center_read).mean()),
                "example_count": batch_size,
            })
    final_step = transition_points[-1]
    _, final_read = losses_at_candidates(
        model,
        input_ids["context_input_ids"],
        input_ids["query_input_ids"],
        labels,
        memories[final_step],
        candidate_batch_size,
    )
    read_losses_by_step[final_step] = final_read
    for point_index, step in enumerate(transition_points):
        preceding = torch.zeros(batch_size, device=memories[step].device)
        if point_index > 0:
            preceding = (
                memories[step] - memories[transition_points[point_index - 1]]
            ).float().flatten(1).norm(dim=1)
        trajectory_rows.append({
            "alias": alias,
            "checkpoint": str(checkpoint_dir),
            "step": step,
            "inner_loss_mean": float(objectives[step].float().mean().cpu()),
            "read_loss_mean": float(read_losses_by_step[step].mean()),
            "gradient_norm_mean": float(gradient_norms[step].mean().cpu()),
            "interval_radius_mean": (
                float(interval_radii[step].mean().cpu()) if step in interval_radii else ""
            ),
            "preceding_interval_displacement_mean": float(preceding.mean().cpu()),
            "example_count": batch_size,
        })
    return grid_rows, trajectory_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = datasets.load_from_disk(str(args.data_path))["valid"]
    rows = [dataset[index] for index in range(min(args.max_examples, len(dataset)))]
    grid = np.linspace(-args.coordinate_limit, args.coordinate_limit, args.grid_size)
    if not np.any(np.isclose(grid, 0.0)) or not np.any(np.isclose(grid, 1.0)):
        raise ValueError("Grid must contain both 0 and 1; use an odd size compatible with the coordinate limit")
    all_grid_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []

    for alias, checkpoint in zip(args.alias, args.checkpoint):
        model, checkpoint_dir, run_args = load_model(Path(checkpoint), device)
        if model.memory_backend != "prefix" or model.mem_proj_mode == "per_sample":
            raise ValueError("Landscape analysis requires prefix memory without per-sample projection")
        tokenizer_path = args.tokenizer_path or run_args.get("tokenizer_path")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model.K = max(args.transition_points)
        model.last_K_second_order = 0
        batch = collate_fn(rows, tokenizer)
        batch = move_to_device(batch, device)
        grid_rows, trajectory_rows = analyze_checkpoint(
            model,
            alias,
            checkpoint_dir,
            batch,
            grid,
            args.candidate_batch_size,
            args.seed,
            args.transition_points,
        )
        all_grid_rows.extend(grid_rows)
        all_trajectory_rows.extend(trajectory_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.output_grid_csv, all_grid_rows)
    write_csv(args.output_trajectory_csv, all_trajectory_rows)
    print(f"Wrote {len(all_grid_rows)} grid rows and {len(all_trajectory_rows)} trajectory rows")


if __name__ == "__main__":
    main()
