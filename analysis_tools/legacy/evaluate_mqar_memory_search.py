#!/usr/bin/env python3
"""Evaluate local memory-search quality for N=8 EnergyGradMem checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis_tools.legacy.evaluate_energy_shaping import recorded_best_checkpoint
from analysis_tools.library import load_model
from run_gradmemgpt_on_mqar import GradMemMQARDataset, collate_fn
from zoology_mqar_data import load_saved_mqar_datasets


SERIES = ("base", "q50", "q75", "best")
METRICS = ("loss", "accuracy", "exact_match")


def score_predictions(model, predictions, read_batch, labels):
    losses = model._compute_per_example_read_target_loss(predictions, read_batch, labels)
    logits = predictions[:, 1:]
    if logits.size(1) != labels.size(1):
        raise ValueError(
            f"Prediction/label length mismatch: {logits.size(1)} vs {labels.size(1)}"
        )
    mask = labels.ne(-100)
    correct = logits.argmax(dim=-1).eq(labels) & mask
    accuracy = correct.sum(dim=1) / mask.sum(dim=1)
    exact_match = correct.sum(dim=1).eq(mask.sum(dim=1)).float()
    return losses.detach(), accuracy.detach(), exact_match.detach()


@torch.no_grad()
def score_memories(model, input_ids, labels, memories):
    backend = model.memory_backend_impl
    batch_ctx = backend.prepare_batch(
        input_ids["context_input_ids"],
        input_ids["query_input_ids"],
        model.model.config.pad_token_id,
    )
    losses = []
    accuracies = []
    exact_matches = []
    for memory in memories:
        state, _ = backend.init_memory_state(labels.size(0))
        state["mem_batch"] = memory
        read_batch = backend.build_read_inputs(state, batch_ctx)
        with backend.activation_context(state), model._disable_write_lora():
            read_out = model.model(
                inputs_embeds=read_batch["inputs_embeds"],
                return_dict=True,
                **read_batch.get("model_kwargs", {}),
            )
        predictions = read_out.logits[
            :,
            read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"],
            :,
        ]
        loss, accuracy, exact_match = score_predictions(
            model, predictions, read_batch, labels
        )
        losses.append(loss)
        accuracies.append(accuracy)
        exact_matches.append(exact_match)
    return torch.stack(losses), torch.stack(accuracies), torch.stack(exact_matches)


def beneficial_quantiles(base, candidates, higher_is_better):
    improvement = candidates - base.unsqueeze(0)
    if not higher_is_better:
        improvement = -improvement
    improvement = improvement.clamp_min(0.0)
    gains = {
        "q50": torch.quantile(improvement, 0.50, dim=0),
        "q75": torch.quantile(improvement, 0.75, dim=0),
        "best": improvement.max(dim=0).values,
    }
    direction = 1.0 if higher_is_better else -1.0
    return {name: base + direction * gain for name, gain in gains.items()}


def evaluate_cell(model, dataset, batch_size, max_examples, seed):
    count = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = collate_fn([dataset[index] for index in range(start, end)])
        input_ids = {key: value.to(model.device) for key, value in batch["input_ids"].items()}
        labels = batch["labels"].to(model.device)

        model.K = 1
        model.config.K = 1
        previous_memory = model(input_ids, labels=None, return_mem=True)["mem"].detach()
        model.K = 2
        model.config.K = 2
        base_output = model(input_ids, labels=None, return_mem=True)
        final_memory = base_output["mem"].detach()

        model.energy_memory_search_num_samples = 4
        model.energy_memory_search_radius_scale = 0.25
        candidates = model._sample_norm_preserving_memory_candidates(
            final_memory, previous_memory
        )

        base_loss, base_accuracy, base_exact_match = score_predictions(
            model, base_output["predictions"], {}, labels
        )
        candidate_loss, candidate_accuracy, candidate_exact_match = score_memories(
            model, input_ids, labels, candidates
        )
        loss_outcomes = beneficial_quantiles(base_loss, candidate_loss, higher_is_better=False)
        accuracy_outcomes = beneficial_quantiles(
            base_accuracy, candidate_accuracy, higher_is_better=True
        )
        exact_match_outcomes = beneficial_quantiles(
            base_exact_match, candidate_exact_match, higher_is_better=True
        )

        step_distance = (final_memory - previous_memory).float().flatten(1).norm(dim=1)
        candidate_distance = (
            candidates - final_memory.unsqueeze(0)
        ).float().flatten(2).norm(dim=2)
        expected_distance = 0.25 * step_distance.unsqueeze(0)
        if not torch.allclose(candidate_distance, expected_distance, atol=2e-4, rtol=2e-4):
            raise AssertionError("Sampled candidates do not have the requested chord radius")

        for offset, example_index in enumerate(range(start, end)):
            row = {
                "example_index": example_index,
                "base_loss": float(base_loss[offset]),
                "q50_loss": float(loss_outcomes["q50"][offset]),
                "q75_loss": float(loss_outcomes["q75"][offset]),
                "best_loss": float(loss_outcomes["best"][offset]),
                "base_accuracy": float(base_accuracy[offset]),
                "q50_accuracy": float(accuracy_outcomes["q50"][offset]),
                "q75_accuracy": float(accuracy_outcomes["q75"][offset]),
                "best_accuracy": float(accuracy_outcomes["best"][offset]),
                "base_exact_match": float(base_exact_match[offset]),
                "q50_exact_match": float(exact_match_outcomes["q50"][offset]),
                "q75_exact_match": float(exact_match_outcomes["q75"][offset]),
                "best_exact_match": float(exact_match_outcomes["best"][offset]),
                "write_step_distance": float(step_distance[offset]),
                "candidate_radius": float(candidate_distance[:, offset].mean()),
            }
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_rows):
    summary = []
    metrics = [metric for metric in METRICS if f"base_{metric}" in run_rows[0]]
    for run_id in sorted({row["run"] for row in run_rows}):
        for n_value in sorted({row["N"] for row in run_rows}):
            selected = [
                row for row in run_rows if row["run"] == run_id and row["N"] == n_value
            ]
            for metric in metrics:
                for series in SERIES:
                    values = [row[f"{series}_{metric}"] for row in selected]
                    summary.append({
                        "run": run_id,
                        "N": n_value,
                        "context_tokens": 2 * n_value,
                        "metric": metric,
                        "series": series,
                        "mean": float(np.mean(values)),
                        "num_examples": len(values),
                    })
    return summary


def aggregate(summary):
    result = []
    metrics = [metric for metric in METRICS if any(row["metric"] == metric for row in summary)]
    for n_value in sorted({row["N"] for row in summary}):
        for metric in metrics:
            for series in SERIES:
                values = [
                    row["mean"] for row in summary
                    if row["N"] == n_value
                    and row["metric"] == metric
                    and row["series"] == series
                ]
                result.append({
                    "N": n_value,
                    "context_tokens": 2 * n_value,
                    "metric": metric,
                    "series": series,
                    "mean": float(np.mean(values)),
                    "std_across_runs": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "num_runs": len(values),
                })
    return result


def plot_results(rows, output_dir, series_labels=None):
    colors = {
        "base": "#9AA5B1",
        "q50": "#8EC5E8",
        "q75": "#B8A1D9",
        "best": "#F2A88F",
    }
    labels = series_labels or {
        "base": "Base memory",
        "q50": "q50 improvement",
        "q75": "q75 improvement",
        "best": "Best-of-4",
    }
    n_values = sorted({row["N"] for row in rows})
    positions = np.arange(len(n_values))
    metrics = [metric for metric in METRICS if any(row["metric"] == metric for row in rows)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for series in SERIES:
            selected = sorted(
                (row for row in rows if row["metric"] == metric and row["series"] == series),
                key=lambda row: row["N"],
            )
            values = np.asarray([row["mean"] for row in selected])
            if metric in ("accuracy", "exact_match"):
                values = 100.0 * values
            ax.plot(
                positions,
                values,
                color=colors[series],
                marker="o",
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.5,
                markersize=6,
                label=labels[series],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([str(value) for value in n_values])
        ax.set_xlabel("Context length (KV pairs)")
        ax.grid(color="#D9DEE7", alpha=0.65, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ylabels = {
        "loss": "Outer loss",
        "accuracy": "Token accuracy (%)",
        "exact_match": "All-query exact match (%)",
    }
    for ax, metric in zip(axes, metrics):
        ax.set_ylabel(ylabels[metric])
        if metric != "exact_match":
            ax.set_yscale("log")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        handlelength=3.0,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output_dir / "memory_search_loss_accuracy.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "memory_search_loss_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/mqar_N8_V8192_L24/"
            "energygradmem_llama_L4H4D256_mem8_K2_ilr5.0_icn10.0_"
            "energy_grad_second_bs_64_lr_1e-04_fp32"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--run-path",
        action="append",
        type=Path,
        help="exact run directory to evaluate; repeat for multiple runs",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        help="exact model.safetensors or checkpoint directory; repeat for multiple checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/mqar_N8_memory_search")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-values", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    args.runs_root = (root / args.runs_root).resolve() if not args.runs_root.is_absolute() else args.runs_root
    args.data_root = (root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets = {}
    for n_value in args.n_values:
        _, valid, metadata = load_saved_mqar_datasets(
            args.data_root / f"mqar_N{n_value}_V8192_L{3 * n_value}"
        )
        datasets[n_value] = GradMemMQARDataset(valid, int(metadata["context_size"]))

    all_rows = []
    checkpoints = []
    if args.checkpoint:
        checkpoint_paths = [
            (root / path).resolve() if not path.is_absolute() else path.resolve()
            for path in args.checkpoint
        ]
        checkpoint_paths = [
            path / "model.safetensors" if path.is_dir() else path
            for path in checkpoint_paths
        ]
    else:
        run_paths = args.run_path or [args.runs_root / f"run_{index}" for index in (1, 2, 3)]
        run_paths = [
            (root / path).resolve() if not path.is_absolute() else path.resolve()
            for path in run_paths
        ]
        checkpoint_paths = []
        for run_path in run_paths:
            checkpoint = recorded_best_checkpoint(run_path)
            if checkpoint is None:
                raise FileNotFoundError(f"No recorded best checkpoint in {run_path}")
            checkpoint_paths.append(checkpoint)
    for run_index, checkpoint in enumerate(checkpoint_paths, start=1):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        checkpoints.append(str(checkpoint))
        print(f"Loading run_{run_index}: {checkpoint}", flush=True)
        model = load_model(checkpoint, device)
        for n_value, dataset in datasets.items():
            print(f"  evaluating N={n_value}", flush=True)
            rows = evaluate_cell(
                model,
                dataset,
                args.batch_size,
                args.max_examples,
                args.seed + 100_000 * run_index + n_value,
            )
            for row in rows:
                row["run"] = f"run_{run_index}"
                row["N"] = n_value
            all_rows.extend(rows)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_run = summarize(all_rows)
    aggregate_rows = aggregate(per_run)
    write_csv(args.output_dir / "per_example.csv", all_rows)
    write_csv(args.output_dir / "per_run_summary.csv", per_run)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    manifest = {
        "checkpoints": checkpoints,
        "datasets": args.n_values,
        "K": 2,
        "num_perturbations": 4,
        "radius_scale": 0.25,
        "radius_definition": "norm-preserving chord radius = 0.25 * ||M2 - M1||",
        "quantile_definition": (
            "Per-example q50/q75/best beneficial gain among four perturbations, "
            "clamped at zero against the unperturbed base; loss and accuracy ranked independently."
        ),
        "num_examples": args.max_examples,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_results(aggregate_rows, args.output_dir)
    print(f"Wrote evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
