#!/usr/bin/env python3
"""Evaluate target-loss-selected memory search rolled through both WRITE steps."""

from __future__ import annotations

import argparse
import gc
import json
import types
from pathlib import Path

import torch

from analysis_tools.legacy.evaluate_energy_shaping import recorded_best_checkpoint
from analysis_tools.legacy.evaluate_mqar_memory_search import (
    SERIES,
    aggregate,
    plot_results,
    score_predictions,
    summarize,
    write_csv,
)
from analysis_tools.library import load_model
from run_gradmemgpt_on_mqar import GradMemMQARDataset, collate_fn
from zoology_mqar_data import load_saved_mqar_datasets


POLICY_RANK = {"q50": 1, "q75": 2, "best": 3}


def rollout_search(policy):
    rank = POLICY_RANK[policy]

    def search(self, backend, memory_state, previous_memory, batch_ctx, labels):
        student_memory = memory_state["mem_batch"]
        with torch.no_grad():
            perturbed = self._sample_norm_preserving_memory_candidates(
                student_memory, previous_memory
            )
            candidate_memories = torch.cat(
                [student_memory.detach().unsqueeze(0), perturbed], dim=0
            )
            candidate_losses = []
            with self._temporary_model_eval():
                for candidate_memory in candidate_memories:
                    candidate_state = dict(memory_state)
                    candidate_state["mem_batch"] = candidate_memory
                    read_batch = backend.build_read_inputs(candidate_state, batch_ctx)
                    with backend.activation_context(candidate_state), self._disable_write_lora():
                        read_out = self.model(
                            inputs_embeds=read_batch["inputs_embeds"],
                            return_dict=True,
                            **read_batch.get("model_kwargs", {}),
                        )
                    predictions = read_out.logits[
                        :,
                        read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"],
                        :,
                    ]
                    candidate_losses.append(
                        self._compute_per_example_read_target_loss(
                            predictions, read_batch, labels
                        )
                    )

            candidate_losses = torch.stack(candidate_losses)
            baseline_loss = candidate_losses[0]
            perturbation_gains = (
                baseline_loss.unsqueeze(0) - candidate_losses[1:]
            ).clamp_min(0.0)
            sorted_indices = perturbation_gains.argsort(dim=0)
            selected_perturbation = sorted_indices[rank]
            batch_indices = torch.arange(
                student_memory.size(0), device=student_memory.device
            )
            selected_gain = perturbation_gains[selected_perturbation, batch_indices]
            selected_indices = torch.where(
                selected_gain.gt(0.0),
                selected_perturbation + 1,
                torch.zeros_like(selected_perturbation),
            )
            teacher_memory = candidate_memories[selected_indices, batch_indices]
            selected_loss = candidate_losses[selected_indices, batch_indices]
            target_gain = (baseline_loss - selected_loss).clamp_min(0.0)
            relative_target_gain = target_gain / baseline_loss.clamp_min(
                torch.finfo(baseline_loss.dtype).eps
            )

        per_example_half_squared_norm = 0.5 * (
            student_memory.float() - teacher_memory.float()
        ).square().flatten(1).sum(dim=1)
        selected_distance = (
            student_memory.detach().float() - teacher_memory.float()
        ).flatten(1).norm(dim=1)
        return {
            "loss": per_example_half_squared_norm.mean(),
            "per_example_loss": per_example_half_squared_norm,
            "relative_target_gain": relative_target_gain,
            "teacher_memory": teacher_memory,
            "improvement_rate": selected_indices.ne(0).float().mean(),
            "target_gain": target_gain.mean(),
            "max_target_gain": target_gain.max(),
            "selected_distance": selected_distance.mean(),
        }

    return search


def evaluate_cell(model, dataset, batch_size, max_examples, seed):
    count = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    original_search = model._compute_energy_memory_search_loss
    rows = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = collate_fn([dataset[index] for index in range(start, end)])
        input_ids = {key: value.to(model.device) for key, value in batch["input_ids"].items()}
        labels = batch["labels"].to(model.device)

        model.eval()
        model.K = 2
        model.config.K = 2
        base_output = model(input_ids, labels=None)
        base_loss, base_accuracy, _ = score_predictions(
            model, base_output["predictions"], {}, labels
        )
        outcomes = {
            "base_loss": base_loss,
            "base_accuracy": base_accuracy,
        }

        for policy in ("q50", "q75", "best"):
            torch.manual_seed(seed + start)
            torch.cuda.manual_seed_all(seed + start)
            model._compute_energy_memory_search_loss = types.MethodType(
                rollout_search(policy), model
            )
            model.energy_memory_search_weight = 1.0
            model.energy_memory_search_num_samples = 4
            model.energy_memory_search_radius_scale = 0.25
            model.energy_memory_search_use_best_for_next_step = True
            model.train()
            output = model(input_ids, labels=labels)
            loss, accuracy, _ = score_predictions(
                model, output["predictions"], {}, labels
            )
            outcomes[f"{policy}_loss"] = loss
            outcomes[f"{policy}_accuracy"] = accuracy

        for offset, example_index in enumerate(range(start, end)):
            row = {"example_index": example_index}
            for metric in ("loss", "accuracy"):
                for series in SERIES:
                    row[f"{series}_{metric}"] = float(
                        outcomes[f"{series}_{metric}"][offset]
                    )
            rows.append(row)

    model._compute_energy_memory_search_loss = original_search
    model.energy_memory_search_weight = 0.0
    model.energy_memory_search_use_best_for_next_step = False
    model.eval()
    return rows


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
        "--output-dir",
        type=Path,
        default=Path("reports/mqar_N8_memory_search_rollout"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
    for n_value in (8, 16, 32, 64):
        _, valid, metadata = load_saved_mqar_datasets(
            args.data_root / f"mqar_N{n_value}_V8192_L{3 * n_value}"
        )
        datasets[n_value] = GradMemMQARDataset(valid, int(metadata["context_size"]))

    all_rows = []
    checkpoints = []
    for run_index in (1, 2, 3):
        run_path = args.runs_root / f"run_{run_index}"
        checkpoint = recorded_best_checkpoint(run_path)
        if checkpoint is None:
            raise FileNotFoundError(f"No recorded best checkpoint in {run_path}")
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
        "datasets": [8, 16, 32, 64],
        "K": 2,
        "num_perturbations_per_step": 4,
        "radius_scale": 0.25,
        "rollout": True,
        "rollout_definition": (
            "After each WRITE step, select an actual loss-improving candidate and use it "
            "as the starting memory for the next step; final READ uses the selected M2 candidate."
        ),
        "policy_definition": (
            "Candidates are ranked by per-example outer-loss gain. q50 uses lower-median "
            "rank 2/4, q75 uses rank 3/4, and best uses rank 4/4; all fall back to base "
            "when the selected candidate does not improve loss. Accuracy is measured on "
            "the same loss-selected memory."
        ),
        "num_examples": args.max_examples,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_results(
        aggregate_rows,
        args.output_dir,
        series_labels={
            "base": "Base trajectory",
            "q50": "q50 rollout",
            "q75": "q75 rollout",
            "best": "Best-of-4 rollout",
        },
    )
    print(f"Wrote rollout evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
