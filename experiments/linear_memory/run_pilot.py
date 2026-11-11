"""Run the bounded intermediate-commit linear-memory pilot.

Run from ``test_time_gd``:

    .venv/bin/python -m experiments.linear_memory.run_pilot \
        --config experiments/linear_memory/pilot_config.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import (
    LinearAssociativeWriter,
    WriterTrace,
    conditional_contamination_prediction,
    error_decomposition,
    exact_snapshot_read,
    folded_linear_attention_read,
    linear_attention_read,
    operation_estimates,
    two_fact_expected_errors,
)


@dataclass(frozen=True)
class Arm:
    name: str
    reader: str
    tau: int
    detach_intermediate: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(command: List[str]) -> str:
    result = subprocess.run(
        ["git", *command], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def _unit_random(shape: Tuple[int, ...], generator: torch.Generator) -> Tensor:
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def make_episode_stream(
    *,
    seed: int,
    count: int,
    batch_size: int,
    num_segments: int,
    value_dim: int,
    global_dim: int,
) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = _unit_random(
        (count, batch_size, num_segments, value_dim), generator
    )
    global_keys = _unit_random(
        (count, batch_size, num_segments, global_dim), generator
    )
    return values, global_keys


def predict(
    model: LinearAssociativeWriter,
    values: Tensor,
    global_keys: Tensor,
    arm: Arm,
) -> Tuple[Tensor, WriterTrace, Tensor]:
    trace = model(values, detach_intermediate=arm.detach_intermediate)
    exact = exact_snapshot_read(trace)
    if arm.reader == "exact":
        return exact, trace, exact
    if arm.reader == "la":
        # The experimental prediction path constructs and reads the ideal
        # folded D-by-(v*p) state.  Raw snapshots remain available because
        # training autograd and the diagnostics need them.
        la = folded_linear_attention_read(trace, global_keys)
        return la, trace, exact
    raise ValueError(f"unknown reader: {arm.reader}")


def normalized_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().sum() / target.square().sum()


@torch.no_grad()
def evaluate(
    model: LinearAssociativeWriter,
    values: Tensor,
    global_keys: Tensor,
    arm: Arm,
    global_dim: int,
) -> Dict[str, Any]:
    prediction, trace, exact = predict(model, values, global_keys, arm)
    denom = values.square().sum()
    metrics: Dict[str, Any] = {
        "nmse": float(normalized_mse(prediction, values)),
        "eta": float(model.eta),
        "num_commits": len(trace.schedule.commit_indices),
    }
    per_fact = (prediction - values).square().sum(dim=-1) / values.square().sum(dim=-1)
    for age in sorted(set(trace.schedule.routing_age.tolist())):
        mask = trace.schedule.routing_age == age
        metrics[f"nmse_age_{age}"] = float(per_fact[:, mask].mean())
    if arm.reader == "la":
        parts = error_decomposition(prediction, exact, values)
        metrics.update(
            local_nmse=float(parts["local"].sum() / denom),
            contamination_nmse=float(parts["contamination"].sum() / denom),
            cross_nmse=float(parts["cross"].sum() / denom),
            predicted_contamination_nmse=float(
                conditional_contamination_prediction(trace, global_dim).sum() / denom
            ),
            decomposition_residual=float(
                (
                    parts["total"]
                    - parts["local"]
                    - parts["contamination"]
                    - parts["cross"]
                )
                .abs()
                .max()
            ),
        )
    else:
        metrics.update(
            local_nmse=metrics["nmse"],
            contamination_nmse=0.0,
            cross_nmse=0.0,
            predicted_contamination_nmse=0.0,
            decomposition_residual=0.0,
        )
    return metrics


def _parameter_gradients(
    model: LinearAssociativeWriter,
    values: Tensor,
    global_keys: Tensor,
    arm: Arm,
) -> Dict[str, Tensor]:
    prediction, _, _ = predict(model, values, global_keys, arm)
    loss = normalized_mse(prediction, values)
    grads = torch.autograd.grad(loss, (model.encoder.weight, model.eta_logit))
    return {"encoder": grads[0].detach(), "eta_logit": grads[1].detach().reshape(1)}


def _vector_stats(a: Tensor, b: Tensor) -> Dict[str, float]:
    a = a.flatten()
    b = b.flatten()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float((a @ b) / denom) if float(denom) > 0 else math.nan
    return {
        "full_norm": float(torch.linalg.vector_norm(a)),
        "detached_norm": float(torch.linalg.vector_norm(b)),
        "difference_norm": float(torch.linalg.vector_norm(a - b)),
        "cosine": cosine,
    }


def gradient_probe(
    config: Dict[str, Any], seed: int, values: Tensor, global_keys: Tensor
) -> List[Dict[str, Any]]:
    model_cfg = config["model"]
    torch.manual_seed(seed)
    template = LinearAssociativeWriter(
        num_segments=model_cfg["num_segments"],
        local_dim=model_cfg["local_dim"],
        value_dim=model_cfg["value_dim"],
        h=model_cfg["h"],
        tau=model_cfg["intermediate_tau"],
        inner_steps=model_cfg["inner_steps"],
        eta_init=model_cfg["eta_init"],
        eta_max=model_cfg["eta_max"],
    )
    state = {key: value.clone() for key, value in template.state_dict().items()}
    rows = []
    for reader in ("exact", "la"):
        full = LinearAssociativeWriter(
            model_cfg["num_segments"], model_cfg["local_dim"], model_cfg["value_dim"],
            model_cfg["h"], model_cfg["intermediate_tau"], model_cfg["inner_steps"],
            model_cfg["eta_init"], model_cfg["eta_max"],
        )
        detached = LinearAssociativeWriter(
            model_cfg["num_segments"], model_cfg["local_dim"], model_cfg["value_dim"],
            model_cfg["h"], model_cfg["intermediate_tau"], model_cfg["inner_steps"],
            model_cfg["eta_init"], model_cfg["eta_max"],
        )
        full.load_state_dict(state)
        detached.load_state_dict(state)
        full_arm = Arm("probe_full", reader, model_cfg["intermediate_tau"], False)
        detached_arm = Arm("probe_detached", reader, model_cfg["intermediate_tau"], True)
        full_pred, _, _ = predict(full, values, global_keys, full_arm)
        detached_pred, _, _ = predict(detached, values, global_keys, detached_arm)
        full_grads = _parameter_gradients(full, values, global_keys, full_arm)
        detached_grads = _parameter_gradients(detached, values, global_keys, detached_arm)
        row: Dict[str, Any] = {
            "seed": seed,
            "reader": reader,
            "forward_max_abs_difference": float((full_pred - detached_pred).abs().max()),
        }
        for parameter in ("encoder", "eta_logit"):
            for key, value in _vector_stats(
                full_grads[parameter], detached_grads[parameter]
            ).items():
                row[f"{parameter}_{key}"] = value
        rows.append(row)
    return rows


def run_two_fact_mc(config: Dict[str, Any]) -> List[Dict[str, float]]:
    mc = config["theory_mc"]
    generator = torch.Generator().manual_seed(mc["seed"])
    rows = []
    for rho in mc["rho_values"]:
        rho = float(rho)
        y = torch.randn(mc["samples"], 2, generator=generator)
        y1, y2 = y[:, 0], y[:, 1]
        rho2 = rho * rho
        endpoint1 = (1.0 - rho2) * y1 + rho * y2
        endpoint_error = 0.5 * ((endpoint1 - y1).square().mean())
        for D in mc["global_dims"]:
            keys = _unit_random((mc["samples"], 2, D), generator)
            similarity = (keys[:, 0] * keys[:, 1]).sum(dim=-1)
            W2x1 = endpoint1
            W1x2 = rho * y1
            snapshot_error = 0.5 * (
                (similarity * W2x1).square().mean()
                + (similarity * W1x2).square().mean()
            )
            theory = two_fact_expected_errors(rho, D)
            rows.append(
                {
                    "rho": rho,
                    "global_dim": D,
                    "endpoint_mc": float(endpoint_error),
                    "endpoint_theory": theory["endpoint_nmse"],
                    "snapshot_la_mc": float(snapshot_error),
                    "snapshot_la_theory": theory["snapshot_la_nmse"],
                    "crossover_D_strict": theory["crossover_D_strict"],
                }
            )
    return rows


def run_arm(
    config: Dict[str, Any],
    seed: int,
    arm: Arm,
    train_values: Tensor,
    train_keys: Tensor,
    eval_values: Tensor,
    eval_keys: Tensor,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model_cfg = config["model"]
    training = config["training"]
    torch.manual_seed(seed)
    model = LinearAssociativeWriter(
        num_segments=model_cfg["num_segments"],
        local_dim=model_cfg["local_dim"],
        value_dim=model_cfg["value_dim"],
        h=model_cfg["h"],
        tau=arm.tau,
        inner_steps=model_cfg["inner_steps"],
        eta_init=model_cfg["eta_init"],
        eta_max=model_cfg["eta_max"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    eval_steps = set(range(0, training["steps"] + 1, training["eval_every"]))
    eval_steps.add(training["steps"])
    eval_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    cumulative_train_seconds = 0.0
    reached_step = None
    reached_train_seconds = None

    for step in range(training["steps"] + 1):
        if step in eval_steps:
            metrics = evaluate(
                model, eval_values, eval_keys, arm, model_cfg["global_dim"]
            )
            row = {"seed": seed, "arm": arm.name, "reader": arm.reader, "step": step, **metrics}
            eval_rows.append(row)
            if reached_step is None and metrics["nmse"] <= training["target_nmse"]:
                reached_step = step
                reached_train_seconds = cumulative_train_seconds
        if step == training["steps"]:
            break

        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _ = predict(model, train_values[step], train_keys[step], arm)
        loss = normalized_mse(prediction, train_values[step])
        loss.backward()
        encoder_grad = float(torch.linalg.vector_norm(model.encoder.weight.grad))
        eta_grad = float(model.eta_logit.grad.abs())
        optimizer.step()
        cumulative_train_seconds += time.perf_counter() - start
        train_rows.append(
            {
                "seed": seed,
                "arm": arm.name,
                "reader": arm.reader,
                "step": step + 1,
                "train_nmse": float(loss.detach()),
                "encoder_grad_norm": encoder_grad,
                "eta_logit_grad_abs": eta_grad,
                "eta": float(model.eta.detach()),
                "cumulative_train_seconds_excluding_eval": cumulative_train_seconds,
            }
        )

    schedule = model.schedule
    estimates = operation_estimates(
        model_cfg["num_segments"], len(schedule.commit_indices), model_cfg["local_dim"],
        model_cfg["value_dim"], model_cfg["global_dim"], model_cfg["inner_steps"],
        model_cfg["num_segments"],
    )
    summary = {
        "seed": seed,
        "arm": arm.name,
        "reader": arm.reader,
        "tau": arm.tau,
        "detach_intermediate": arm.detach_intermediate,
        "steps_to_target_nmse": reached_step,
        "train_seconds_to_target_excluding_eval": reached_train_seconds,
        "target_time_resolution": f"first evaluation at {training['eval_every']}-step intervals",
        "target_nmse": training["target_nmse"],
        "train_seconds_excluding_eval": cumulative_train_seconds,
        "final": eval_rows[-1],
        "operation_estimates_per_episode": estimates,
    }
    return eval_rows, train_rows, summary


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    threads = int(config["runtime"]["torch_num_threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(int(config["runtime"]["torch_num_interop_threads"]))
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    model_cfg = config["model"]
    training = config["training"]
    arms = [Arm(**arm) for arm in config["arms"]]
    all_eval_rows: List[Dict[str, Any]] = []
    all_train_rows: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []

    for seed in config["seeds"]:
        train_values, train_keys = make_episode_stream(
            seed=seed + config["data"]["train_seed_offset"],
            count=training["steps"],
            batch_size=training["batch_size"],
            num_segments=model_cfg["num_segments"],
            value_dim=model_cfg["value_dim"],
            global_dim=model_cfg["global_dim"],
        )
        eval_values_stream, eval_keys_stream = make_episode_stream(
            seed=seed + config["data"]["eval_seed_offset"],
            count=1,
            batch_size=training["eval_episodes"],
            num_segments=model_cfg["num_segments"],
            value_dim=model_cfg["value_dim"],
            global_dim=model_cfg["global_dim"],
        )
        eval_values = eval_values_stream[0]
        eval_keys = eval_keys_stream[0]
        probes.extend(
            gradient_probe(
                config,
                seed,
                eval_values[: training["probe_episodes"]],
                eval_keys[: training["probe_episodes"]],
            )
        )
        for arm in arms:
            eval_rows, train_rows, summary = run_arm(
                config, seed, arm, train_values, train_keys,
                eval_values, eval_keys,
            )
            all_eval_rows.extend(eval_rows)
            all_train_rows.extend(train_rows)
            all_summaries.append(summary)
            print(
                f"seed={seed} arm={arm.name} final_nmse={summary['final']['nmse']:.6f} "
                f"steps_to_target={summary['steps_to_target_nmse']} "
                f"train_seconds={summary['train_seconds_excluding_eval']:.3f}",
                flush=True,
            )

    two_fact_rows = run_two_fact_mc(config)
    _write_csv(output_dir / "eval_curves.csv", all_eval_rows)
    _write_csv(output_dir / "train_curves.csv", all_train_rows)
    _write_csv(output_dir / "gradient_probes.csv", probes)
    _write_csv(output_dir / "two_fact_theory_mc.csv", two_fact_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, allow_nan=True) + "\n"
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    code_paths = [Path(__file__).resolve(), Path(__file__).with_name("model.py").resolve()]
    provenance = {
        "experiment_id": config["experiment_id"],
        "started_from_config_sha256": _sha256(config_path),
        "config_path": str(config_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": ".venv/bin/python -m experiments.linear_memory.run_pilot --config experiments/linear_memory/pilot_config.json",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "file_sha256": {str(path): _sha256(path) for path in code_paths},
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
