"""Run EXP-LM-002 in leakage-separated calibration and evaluation stages.

Run from ``test_time_gd``.  The evaluation stage verifies that the frozen
selection inputs still match the hashes recorded before held-out generation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .allocation import (
    AllocationCandidate,
    calibration_episode_statistics,
    candidate_resources,
    candidates_from_config,
    choose_candidate,
    feasible_regret,
    folded_la_episode_errors,
    heldout_episode_statistics,
    make_local_addresses,
    make_values,
    mean_se_ci,
    point_feasibility,
    prediction,
    stable_seed,
    valid_taus,
    write_local_groups,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("allocation_protocol.md")
ALLOCATION_CODE = Path(__file__).with_name("allocation.py")
MODEL_CODE = Path(__file__).with_name("model.py")
RUNNER_CODE = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_text(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure_runtime(config: Mapping[str, Any]) -> None:
    runtime = config["runtime"]
    torch.set_num_threads(int(runtime["torch_num_threads"]))
    try:
        torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))


def input_hashes(config_path: Path) -> Dict[str, str]:
    hashes = {
        "config": sha256(config_path),
        "protocol": sha256(PROTOCOL),
        "allocation_code": sha256(ALLOCATION_CODE),
        "model_code": sha256(MODEL_CODE),
        "runner_code": sha256(RUNNER_CODE),
    }
    config = read_json(config_path)
    if "audit_protocol" in config:
        audit_protocol = (ROOT / str(config["audit_protocol"])).resolve()
        hashes["audit_protocol"] = sha256(audit_protocol)
    return hashes


def calibration_rows(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = int(config["calibration"]["groups_per_candidate"])
    base = int(config["calibration"]["seed_offset"])
    alpha = float(config["geometry"]["correlated_mixing_coefficient"])
    grid = config["grid"]
    value_dim = int(grid["value_dim"])
    for seed in config["seeds"]:
        for geometry in config["geometry"]["regimes"]:
            for p in grid["local_dims"]:
                for h in grid["reset_horizons"]:
                    address_generator = torch.Generator().manual_seed(
                        stable_seed(base, seed, geometry, p, h, "addresses")
                    )
                    value_generator = torch.Generator().manual_seed(
                        stable_seed(base, seed, p, h, "values")
                    )
                    addresses = make_local_addresses(
                        episodes=n,
                        groups=1,
                        h=h,
                        p=p,
                        geometry=geometry,
                        mixing_coefficient=alpha,
                        generator=address_generator,
                    )
                    values = make_values(
                        episodes=n,
                        groups=1,
                        h=h,
                        value_dim=value_dim,
                        generator=value_generator,
                    )
                    for tau in valid_taus(h):
                        trace = write_local_groups(
                            addresses,
                            values,
                            tau=tau,
                            eta=float(grid["eta"]),
                            inner_steps=int(grid["inner_steps"]),
                        )
                        stats = calibration_episode_statistics(trace)
                        row: Dict[str, Any] = {
                            "seed": seed,
                            "geometry": geometry,
                            "p": p,
                            "h": h,
                            "tau": tau,
                            "episodes": n,
                            "commits_per_group": trace.snapshots.shape[2],
                        }
                        for name, samples in stats.items():
                            summary = mean_se_ci(samples)
                            row[f"{name}_mean"] = summary["mean"]
                            row[f"{name}_se"] = summary["se"]
                        rows.append(row)
    return rows


def calibration_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str, int, int, int], Mapping[str, Any]]:
    return {
        (
            int(row["seed"]),
            str(row["geometry"]),
            int(row["p"]),
            int(row["h"]),
            int(row["tau"]),
        ): row
        for row in rows
    }


def build_predictions_and_selections(
    config: Mapping[str, Any], calibration: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = calibration_lookup(calibration)
    candidates = candidates_from_config(config)
    cap = int(config["selection"]["folded_inference_memory_cap_scalars"])
    value_dim = int(config["grid"]["value_dim"])
    predictions: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    for seed in config["seeds"]:
        for geometry in config["geometry"]["regimes"]:
            for J in config["heldout"]["num_segments"]:
                for Q in config["heldout"]["num_queries"]:
                    for epsilon in config["heldout"]["target_errors"]:
                        scenario_rows = []
                        for candidate in candidates:
                            calibration_row = lookup[
                                (seed, geometry, candidate.p, candidate.h, candidate.tau)
                            ]
                            resources = candidate_resources(
                                candidate, J=J, Q=Q, value_dim=value_dim
                            )
                            predicted_error = prediction(
                                float(calibration_row["delta_mean"]),
                                float(calibration_row["A0_mean"]),
                                float(calibration_row["B_mean"]),
                                resources["groups"],
                                candidate.D,
                            )
                            row = {
                                "seed": seed,
                                "geometry": geometry,
                                "J": J,
                                "Q": Q,
                                "epsilon": epsilon,
                                "p": candidate.p,
                                "h": candidate.h,
                                "tau": candidate.tau,
                                "D": candidate.D,
                                "predicted_error": predicted_error,
                                "calibration_delta": float(calibration_row["delta_mean"]),
                                "calibration_A0": float(calibration_row["A0_mean"]),
                                "calibration_B": float(calibration_row["B_mean"]),
                                **resources,
                            }
                            predictions.append(row)
                            scenario_rows.append(row)
                        for policy in config["selection"]["policies"]:
                            chosen = choose_candidate(
                                scenario_rows,
                                policy=policy,
                                epsilon=float(epsilon),
                                memory_cap=cap,
                            )
                            record: Dict[str, Any] = {
                                "seed": seed,
                                "geometry": geometry,
                                "J": J,
                                "Q": Q,
                                "epsilon": epsilon,
                                "policy": policy,
                                "selected": chosen is not None,
                            }
                            if chosen is not None:
                                for key in (
                                    "p",
                                    "h",
                                    "tau",
                                    "D",
                                    "predicted_error",
                                    "operation_proxy",
                                    "folded_inference_memory_scalars",
                                ):
                                    record[key] = chosen[key]
                            selections.append(record)
    return predictions, selections


def run_calibrate_select(config: Mapping[str, Any], config_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "config_snapshot.json"
    if snapshot.exists() and snapshot.read_bytes() != config_path.read_bytes():
        raise RuntimeError("existing config snapshot differs; use a fresh output directory")
    snapshot.write_bytes(config_path.read_bytes())
    hashes = input_hashes(config_path)
    started = time.perf_counter()
    calibration = calibration_rows(config)
    calibration_path = output / "calibration.csv"
    write_csv(calibration_path, calibration)
    predictions, selections = build_predictions_and_selections(config, calibration)
    prediction_path = output / "candidate_predictions.csv"
    write_csv(prediction_path, predictions)
    artifact = {
        "experiment_id": config["experiment_id"],
        "created_at_utc": utc_now(),
        "leakage_barrier": "created before any held-out episode generation or outcome access",
        "input_hashes": hashes,
        "calibration_sha256": sha256(calibration_path),
        "candidate_predictions_sha256": sha256(prediction_path),
        "selection_count": len(selections),
        "selections": selections,
    }
    artifact_path = output / "selections_preregistered.json"
    write_json(artifact_path, artifact)
    selection_hash = sha256(artifact_path)
    (output / "selections_preregistered.sha256").write_text(selection_hash + "\n")
    pre = {
        "stage": "calibrate-select",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "input_hashes": hashes,
        "calibration_sha256": artifact["calibration_sha256"],
        "candidate_predictions_sha256": artifact["candidate_predictions_sha256"],
        "selections_preregistered_sha256": selection_hash,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_head": git_text(["rev-parse", "HEAD"]),
        "git_status_before_heldout": git_text(["status", "--short"]),
    }
    write_json(output / "pre_evaluation_provenance.json", pre)
    print(json.dumps({"stage": "calibrate-select", "rows": len(calibration), "predictions": len(predictions), "selections": len(selections), "selection_sha256": selection_hash}, indent=2))


def verify_barrier(config_path: Path, output: Path) -> Dict[str, Any]:
    artifact_path = output / "selections_preregistered.json"
    if not artifact_path.exists():
        raise RuntimeError("missing preregistered selections; run calibrate-select first")
    artifact = read_json(artifact_path)
    expected_hash = (output / "selections_preregistered.sha256").read_text().strip()
    if sha256(artifact_path) != expected_hash:
        raise RuntimeError("selection artifact hash mismatch")
    current = input_hashes(config_path)
    if current != artifact["input_hashes"]:
        raise RuntimeError(f"frozen inputs changed after selection: {current} != {artifact['input_hashes']}")
    calibration_path = output / "calibration.csv"
    prediction_path = output / "candidate_predictions.csv"
    if sha256(calibration_path) != artifact["calibration_sha256"]:
        raise RuntimeError("calibration changed after selection")
    if sha256(prediction_path) != artifact["candidate_predictions_sha256"]:
        raise RuntimeError("candidate predictions changed after selection")
    return artifact


def _heldout_trace(
    config: Mapping[str, Any], *, seed: int, geometry: str, J: int, p: int, h: int, tau: int
):
    episodes = int(config["heldout"]["episodes"])
    groups = J // h
    value_dim = int(config["grid"]["value_dim"])
    base = int(config["heldout"]["seed_offset"])
    alpha = float(config["geometry"]["correlated_mixing_coefficient"])
    # Values are shared across all p, h, tau and geometry for a seed/J pair.
    value_generator = torch.Generator().manual_seed(stable_seed(base, seed, J, "values"))
    flat_values = make_values(
        episodes=episodes,
        groups=1,
        h=J,
        value_dim=value_dim,
        generator=value_generator,
    )[:, 0]
    values = flat_values.reshape(episodes, groups, h, value_dim)
    # Addresses are shared across tau; grouping and p necessarily alter them.
    address_generator = torch.Generator().manual_seed(
        stable_seed(base, seed, geometry, J, p, h, "addresses")
    )
    addresses = make_local_addresses(
        episodes=episodes,
        groups=groups,
        h=h,
        p=p,
        geometry=geometry,
        mixing_coefficient=alpha,
        generator=address_generator,
    )
    return write_local_groups(
        addresses,
        values,
        tau=tau,
        eta=float(config["grid"]["eta"]),
        inner_steps=int(config["grid"]["inner_steps"]),
    )


def heldout_candidate_rows(
    config: Mapping[str, Any], calibration: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    lookup = calibration_lookup(calibration)
    rows: List[Dict[str, Any]] = []
    z = float(config["acceptance"]["confidence_z"])
    max_D = max(config["grid"]["global_dims"])
    key_base = int(config["heldout"]["global_key_seed_offset"])
    for seed in config["seeds"]:
        for geometry in config["geometry"]["regimes"]:
            for J in config["heldout"]["num_segments"]:
                for p in config["grid"]["local_dims"]:
                    for h in config["grid"]["reset_horizons"]:
                        for tau in valid_taus(h):
                            trace = _heldout_trace(
                                config,
                                seed=seed,
                                geometry=geometry,
                                J=J,
                                p=p,
                                h=h,
                                tau=tau,
                            )
                            diagnostics = heldout_episode_statistics(trace)
                            delta_summary = mean_se_ci(diagnostics["delta"], z=z)
                            A_summary = mean_se_ci(diagnostics["A"], z=z)
                            key_generator = torch.Generator().manual_seed(
                                stable_seed(key_base, seed, geometry, J, p, h, tau, "global")
                            )
                            raw_keys = torch.randn(
                                trace.snapshots.shape[0],
                                trace.snapshots.shape[1],
                                trace.snapshots.shape[2],
                                max_D,
                                generator=key_generator,
                                dtype=trace.snapshots.dtype,
                            )
                            calibration_row = lookup[(seed, geometry, p, h, tau)]
                            for D in config["grid"]["global_dims"]:
                                reference_samples = diagnostics["delta"] + diagnostics["A"] / D
                                reference = mean_se_ci(reference_samples, z=z)
                                keys = F.normalize(raw_keys[..., :D], dim=-1)
                                sampled = mean_se_ci(
                                    folded_la_episode_errors(trace, keys), z=z
                                )
                                groups = J // h
                                predicted = prediction(
                                    float(calibration_row["delta_mean"]),
                                    float(calibration_row["A0_mean"]),
                                    float(calibration_row["B_mean"]),
                                    groups,
                                    D,
                                )
                                rows.append(
                                    {
                                        "seed": seed,
                                        "geometry": geometry,
                                        "J": J,
                                        "p": p,
                                        "h": h,
                                        "tau": tau,
                                        "D": D,
                                        "episodes": trace.snapshots.shape[0],
                                        "num_groups": groups,
                                        "commits_per_group": trace.snapshots.shape[2],
                                        "predicted_error": predicted,
                                        "reference_mean": reference["mean"],
                                        "reference_se": reference["se"],
                                        "reference_ci_low": reference["ci_low"],
                                        "reference_ci_high": reference["ci_high"],
                                        "sampled_la_mean": sampled["mean"],
                                        "sampled_la_se": sampled["se"],
                                        "sampled_la_ci_low": sampled["ci_low"],
                                        "sampled_la_ci_high": sampled["ci_high"],
                                        "heldout_delta_mean": delta_summary["mean"],
                                        "heldout_delta_se": delta_summary["se"],
                                        "heldout_A_mean": A_summary["mean"],
                                        "heldout_A_se": A_summary["se"],
                                        "prediction_abs_error": abs(predicted - reference["mean"]),
                                        "sampled_minus_reference": sampled["mean"] - reference["mean"],
                                    }
                                )
    return rows


def metric_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str, int, int, int, int, int], Mapping[str, Any]]:
    return {
        (
            int(row["seed"]),
            str(row["geometry"]),
            int(row["J"]),
            int(row["p"]),
            int(row["h"]),
            int(row["tau"]),
            int(row["D"]),
        ): row
        for row in rows
    }


def evaluate_selections(
    config: Mapping[str, Any], selections: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    lookup = metric_lookup(metrics)
    cap = int(config["selection"]["folded_inference_memory_cap_scalars"])
    value_dim = int(config["grid"]["value_dim"])
    zrows: List[Dict[str, Any]] = []
    candidates = candidates_from_config(config)
    for selection in selections:
        seed = int(selection["seed"])
        geometry = str(selection["geometry"])
        J = int(selection["J"])
        Q = int(selection["Q"])
        epsilon = float(selection["epsilon"])
        oracle_options = []
        certified_options = []
        uncertain_options = []
        for candidate in candidates:
            metric = lookup[(seed, geometry, J, candidate.p, candidate.h, candidate.tau, candidate.D)]
            resources = candidate_resources(candidate, J=J, Q=Q, value_dim=value_dim)
            if resources["folded_inference_memory_scalars"] > cap:
                continue
            if float(metric["reference_mean"]) <= epsilon:
                oracle_options.append((resources["operation_proxy"], candidate, metric, resources))
            if float(metric["reference_ci_high"]) <= epsilon:
                certified_options.append((resources["operation_proxy"], candidate, metric, resources))
            elif float(metric["reference_ci_low"]) <= epsilon < float(metric["reference_ci_high"]):
                uncertain_options.append((resources["operation_proxy"], candidate, metric, resources))
        oracle = min(
            oracle_options,
            key=lambda item: (
                item[0], item[3]["folded_inference_memory_scalars"], item[1]
            ),
            default=None,
        )
        certified_oracle = min(
            certified_options,
            key=lambda item: (
                item[0], item[3]["folded_inference_memory_scalars"], item[1]
            ),
            default=None,
        )
        cheapest_uncertain = min(
            uncertain_options,
            key=lambda item: (
                item[0], item[3]["folded_inference_memory_scalars"], item[1]
            ),
            default=None,
        )
        row: Dict[str, Any] = {
            **selection,
            "oracle_exists": oracle is not None,
            "oracle_operation_proxy": oracle[0] if oracle else None,
            "oracle_p": oracle[1].p if oracle else None,
            "oracle_h": oracle[1].h if oracle else None,
            "oracle_tau": oracle[1].tau if oracle else None,
            "oracle_D": oracle[1].D if oracle else None,
            "certified_oracle_exists": certified_oracle is not None,
            "certified_oracle_operation_proxy": certified_oracle[0] if certified_oracle else None,
            "certified_oracle_p": certified_oracle[1].p if certified_oracle else None,
            "certified_oracle_h": certified_oracle[1].h if certified_oracle else None,
            "certified_oracle_tau": certified_oracle[1].tau if certified_oracle else None,
            "certified_oracle_D": certified_oracle[1].D if certified_oracle else None,
            "cheapest_uncertain_operation_proxy": cheapest_uncertain[0] if cheapest_uncertain else None,
            "cheapest_uncertain_p": cheapest_uncertain[1].p if cheapest_uncertain else None,
            "cheapest_uncertain_h": cheapest_uncertain[1].h if cheapest_uncertain else None,
            "cheapest_uncertain_tau": cheapest_uncertain[1].tau if cheapest_uncertain else None,
            "cheapest_uncertain_D": cheapest_uncertain[1].D if cheapest_uncertain else None,
            "cheaper_uncertain_than_certified_oracle": (
                cheapest_uncertain is not None
                and certified_oracle is not None
                and cheapest_uncertain[0] < certified_oracle[0]
            ),
        }
        if not bool(selection["selected"]):
            row.update(
                point_reference_feasible=None,
                reference_interval_status="not_selected",
                point_sampled_feasible=None,
                sampled_interval_status="not_selected",
                false_feasible=False,
                false_infeasible=oracle is not None,
                cost_regret=None,
                selected_certified_feasible=None,
                certified_cost_regret=None,
            )
            zrows.append(row)
            continue
        candidate = AllocationCandidate(
            int(selection["p"]), int(selection["h"]), int(selection["tau"]), int(selection["D"])
        )
        metric = lookup[(seed, geometry, J, candidate.p, candidate.h, candidate.tau, candidate.D)]
        reference_point, reference_interval = point_feasibility(
            float(metric["reference_mean"]),
            float(metric["reference_ci_low"]),
            float(metric["reference_ci_high"]),
            epsilon,
        )
        sampled_point, sampled_interval = point_feasibility(
            float(metric["sampled_la_mean"]),
            float(metric["sampled_la_ci_low"]),
            float(metric["sampled_la_ci_high"]),
            epsilon,
        )
        exact_point, exact_interval = point_feasibility(
            float(metric["heldout_delta_mean"]),
            float(metric["heldout_delta_mean"]) - 1.96 * float(metric["heldout_delta_se"]),
            float(metric["heldout_delta_mean"]) + 1.96 * float(metric["heldout_delta_se"]),
            epsilon,
        )
        row.update(
            reference_mean=float(metric["reference_mean"]),
            reference_se=float(metric["reference_se"]),
            reference_ci_low=float(metric["reference_ci_low"]),
            reference_ci_high=float(metric["reference_ci_high"]),
            sampled_la_mean=float(metric["sampled_la_mean"]),
            sampled_la_se=float(metric["sampled_la_se"]),
            sampled_la_ci_low=float(metric["sampled_la_ci_low"]),
            sampled_la_ci_high=float(metric["sampled_la_ci_high"]),
            heldout_delta_mean=float(metric["heldout_delta_mean"]),
            heldout_A_mean=float(metric["heldout_A_mean"]),
            prediction_abs_error=abs(float(selection["predicted_error"]) - float(metric["reference_mean"])),
            point_reference_feasible=reference_point,
            reference_interval_status=reference_interval,
            point_sampled_feasible=sampled_point,
            sampled_interval_status=sampled_interval,
            exact_oracle_point_feasible=exact_point,
            exact_oracle_interval_status=exact_interval,
            false_feasible=not reference_point,
            false_infeasible=False,
            cost_regret=feasible_regret(
                int(selection["operation_proxy"]), reference_point, oracle[0] if oracle else None
            ),
            selected_certified_feasible=reference_interval == "feasible",
            certified_cost_regret=feasible_regret(
                int(selection["operation_proxy"]),
                reference_interval == "feasible",
                certified_oracle[0] if certified_oracle else None,
            ),
        )
        zrows.append(row)
    return zrows


def oracle_scenario_coverage(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> Dict[str, Any]:
    """Abstention-safe coverage over scenarios where a point oracle exists."""

    oracle_rows = [row for row in rows if bool(row.get("oracle_exists"))]
    covered = [
        row
        for row in oracle_rows
        if bool(row.get("selected")) and bool(row.get("point_reference_feasible"))
    ]
    coverage = len(covered) / len(oracle_rows) if oracle_rows else None
    return {
        "oracle_existing_scenarios": len(oracle_rows),
        "oracle_existing_with_feasible_selection": len(covered),
        "oracle_scenario_coverage": coverage,
        "coverage_threshold": threshold,
        "coverage_passed": coverage is not None and coverage >= threshold,
    }


def summarize(
    config: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]], evaluations: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    policy_summaries: Dict[str, Any] = {}
    acceptance = config["acceptance"]
    coverage_threshold = float(
        config.get("post_result_audit", {}).get(
            "oracle_scenario_coverage_min",
            acceptance["selected_reference_feasible_rate_min"],
        )
    )
    for policy in config["selection"]["policies"]:
        rows = [row for row in evaluations if row["policy"] == policy]
        selected = [row for row in rows if bool(row["selected"])]
        feasible = [row for row in selected if bool(row.get("point_reference_feasible"))]
        regrets = [float(row["cost_regret"]) for row in feasible if row.get("cost_regret") is not None]
        certified_regrets = [
            float(row["certified_cost_regret"])
            for row in selected
            if row.get("certified_cost_regret") is not None
        ]
        abs_errors = [float(row["prediction_abs_error"]) for row in selected]
        policy_summaries[policy] = {
            "scenarios": len(rows),
            "selected": len(selected),
            "no_selection": len(rows) - len(selected),
            "point_reference_feasible": len(feasible),
            "point_reference_feasible_rate_among_selected": len(feasible) / len(selected) if selected else None,
            "false_feasible": sum(bool(row["false_feasible"]) for row in rows),
            "false_infeasible": sum(bool(row["false_infeasible"]) for row in rows),
            "reference_interval_feasible": sum(row.get("reference_interval_status") == "feasible" for row in selected),
            "reference_interval_uncertain": sum(row.get("reference_interval_status") == "uncertain" for row in selected),
            "reference_interval_infeasible": sum(row.get("reference_interval_status") == "infeasible" for row in selected),
            "sampled_point_feasible": sum(bool(row.get("point_sampled_feasible")) for row in selected),
            "median_prediction_abs_error": statistics.median(abs_errors) if abs_errors else None,
            "max_prediction_abs_error": max(abs_errors) if abs_errors else None,
            "median_feasible_cost_regret": statistics.median(regrets) if regrets else None,
            "max_feasible_cost_regret": max(regrets) if regrets else None,
            "certified_oracle_scenarios": sum(bool(row.get("certified_oracle_exists")) for row in rows),
            "selected_certified_feasible": sum(bool(row.get("selected_certified_feasible")) for row in selected),
            "median_certified_cost_regret": statistics.median(certified_regrets) if certified_regrets else None,
            "max_certified_cost_regret": max(certified_regrets) if certified_regrets else None,
            "cheaper_interval_uncertain_than_certified_oracle": sum(
                bool(row.get("cheaper_uncertain_than_certified_oracle")) for row in rows
            ),
            **oracle_scenario_coverage(rows, coverage_threshold),
        }
    theory = policy_summaries["theory"]
    transfer_pass = (
        theory["point_reference_feasible_rate_among_selected"] is not None
        and theory["point_reference_feasible_rate_among_selected"] >= acceptance["selected_reference_feasible_rate_min"]
        and theory["median_feasible_cost_regret"] is not None
        and theory["median_feasible_cost_regret"] <= acceptance["median_feasible_cost_regret_max"]
    )
    audit_pass = transfer_pass and bool(theory["coverage_passed"])
    sampled_differences = [float(row["sampled_minus_reference"]) for row in metrics]
    return {
        "experiment_id": config["experiment_id"],
        "completed_at_utc": utc_now(),
        "candidate_metric_rows": len(metrics),
        "selection_evaluation_rows": len(evaluations),
        "policy_summaries": policy_summaries,
        "acceptance": {
            "feasible_rate_threshold": acceptance["selected_reference_feasible_rate_min"],
            "median_regret_threshold": acceptance["median_feasible_cost_regret_max"],
            "passed": transfer_pass,
        },
        "post_result_audit_criterion": {
            "status": "added after EXP-LM-002 outcomes; preregister in future studies",
            "oracle_scenario_coverage_threshold": coverage_threshold,
            "oracle_scenario_coverage": theory["oracle_scenario_coverage"],
            "oracle_existing_scenarios": theory["oracle_existing_scenarios"],
            "coverage_passed": theory["coverage_passed"],
            "original_acceptance_and_audit_coverage_passed": audit_pass,
        },
        "interval_oracle_sensitivity": {
            "status": "post-result sensitivity; point-feasible oracle remains primary",
            "certified_oracle_scenarios": theory["certified_oracle_scenarios"],
            "selected_certified_feasible": theory["selected_certified_feasible"],
            "median_certified_cost_regret": theory["median_certified_cost_regret"],
            "max_certified_cost_regret": theory["max_certified_cost_regret"],
            "cheaper_interval_uncertain_than_certified_oracle": theory[
                "cheaper_interval_uncertain_than_certified_oracle"
            ],
        },
        "sampled_la_consistency": {
            "mean_sampled_minus_reference": statistics.fmean(sampled_differences),
            "median_abs_sampled_minus_reference": statistics.median(abs(x) for x in sampled_differences),
            "max_abs_sampled_minus_reference": max(abs(x) for x in sampled_differences),
        },
        "scope": "Three seeds and two known geometry regimes; rows sharing held-out episodes across Q, target, or policy are dependent reporting views, not independent replications. No writer fitting or optimization claim.",
    }


def run_evaluate(config: Mapping[str, Any], config_path: Path, output: Path) -> None:
    artifact = verify_barrier(config_path, output)
    started = time.perf_counter()
    calibration = read_csv(output / "calibration.csv")
    metrics = heldout_candidate_rows(config, calibration)
    write_csv(output / "candidate_metrics.csv", metrics)
    evaluations = evaluate_selections(config, artifact["selections"], metrics)
    write_csv(output / "selection_evaluation.csv", evaluations)
    summary = summarize(config, metrics, evaluations)
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "summary.json", summary)
    provenance = {
        "experiment_id": config["experiment_id"],
        "completed_at_utc": utc_now(),
        "input_hashes": input_hashes(config_path),
        "config_snapshot_sha256": sha256(output / "config_snapshot.json"),
        "calibration_sha256": sha256(output / "calibration.csv"),
        "candidate_predictions_sha256": sha256(output / "candidate_predictions.csv"),
        "selections_preregistered_sha256": sha256(output / "selections_preregistered.json"),
        "candidate_metrics_sha256": sha256(output / "candidate_metrics.csv"),
        "selection_evaluation_sha256": sha256(output / "selection_evaluation.csv"),
        "summary_sha256": sha256(output / "summary.json"),
        "rng_scheme": "SHA-256-derived seeds over registered seed/split/geometry/candidate coordinates; held-out values shared over p,h,tau,geometry for each seed/J; addresses shared over tau; raw global Gaussian prefixes shared over D.",
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_head": git_text(["rev-parse", "HEAD"]),
        "git_status": git_text(["status", "--short"]),
        "command": " ".join(sys.argv),
    }
    write_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibrate-select", "evaluate"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    configure_runtime(config)
    output = (ROOT / config["output_dir"]).resolve()
    if args.stage == "calibrate-select":
        run_calibrate_select(config, config_path, output)
    else:
        run_evaluate(config, config_path, output)


if __name__ == "__main__":
    main()
