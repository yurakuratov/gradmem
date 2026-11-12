"""Two-stage leakage-safe runner for EXP-LM-003."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .allocation import (
    calibration_episode_statistics,
    folded_la_episode_errors,
    heldout_episode_statistics,
    make_local_addresses,
    make_values,
    mean_se_ci,
    stable_seed,
    valid_taus,
    write_local_groups,
)
from .refined_allocation import (
    IntegerAllocation,
    LocalConfiguration,
    allocation_resources,
    allocation_sort_key,
    cheaper_uncertain_allocation,
    close_leq,
    eligible_global_dims,
    episode_joint_moments,
    iid_h2_endpoint_moments,
    local_configurations,
    maximum_global_dim,
    minimum_feasible_integer_D,
    oracle_allocation,
    oracle_coverage,
    policy_allows,
    prediction_moments,
    reference_stats_from_moments,
    select_calibrated_allocation,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("refined_allocation_protocol.md")
REFINED_CODE = Path(__file__).with_name("refined_allocation.py")
RUNNER_CODE = Path(__file__).resolve()
ALLOCATION_CODE = Path(__file__).with_name("allocation.py")
MODEL_CODE = Path(__file__).with_name("model.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def input_hashes(config_path: Path) -> Dict[str, str]:
    return {
        "config": sha256(config_path),
        "protocol": sha256(PROTOCOL),
        "refined_code": sha256(REFINED_CODE),
        "runner_code": sha256(RUNNER_CODE),
        "allocation_code": sha256(ALLOCATION_CODE),
        "model_code": sha256(MODEL_CODE),
    }


def configure_runtime(config: Mapping[str, Any]) -> None:
    runtime = config["runtime"]
    torch.set_num_threads(int(runtime["torch_num_threads"]))
    try:
        torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))


def git_text(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def calibration_rows(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = int(config["calibration"]["groups_per_local_configuration"])
    base = int(config["calibration"]["seed_offset"])
    alpha = float(config["geometry"]["correlated_mixing_coefficient"])
    v = int(config["grid"]["value_dim"])
    for seed in config["seeds"]:
        for geometry in config["geometry"]["regimes"]:
            for p in config["grid"]["local_dims"]:
                for h in config["grid"]["reset_horizons"]:
                    x_gen = torch.Generator().manual_seed(
                        stable_seed(base, seed, geometry, p, h, "addresses")
                    )
                    y_gen = torch.Generator().manual_seed(
                        stable_seed(base, seed, p, h, "values")
                    )
                    addresses = make_local_addresses(
                        episodes=n,
                        groups=1,
                        h=h,
                        p=p,
                        geometry=geometry,
                        mixing_coefficient=alpha,
                        generator=x_gen,
                    )
                    values = make_values(
                        episodes=n, groups=1, h=h, value_dim=v, generator=y_gen
                    )
                    for tau in valid_taus(h):
                        trace = write_local_groups(addresses, values, tau=tau)
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
                        if h == 1:
                            row.update(
                                selector_delta=0.0,
                                selector_A0=0.0,
                                selector_B=1.0 / p,
                                selector_source="exact_h1_identity",
                            )
                        else:
                            row.update(
                                selector_delta=row["delta_mean"],
                                selector_A0=row["A0_mean"],
                                selector_B=row["B_mean"],
                                selector_source="empirical_calibration",
                            )
                        rows.append(row)
    return rows


def calibration_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str, int, int, int], Mapping[str, Any]]:
    return {
        (int(r["seed"]), str(r["geometry"]), int(r["p"]), int(r["h"]), int(r["tau"])): r
        for r in rows
    }


def build_frontiers_and_selections(
    config: Mapping[str, Any], calibration: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = calibration_lookup(calibration)
    cap = int(config["selection"]["folded_inference_memory_cap_scalars"])
    slack = float(config["selection"]["predicted_error_slack_multiplier"])
    v = int(config["grid"]["value_dim"])
    frontiers: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    locals_ = local_configurations(config)
    for seed in config["seeds"]:
        for geometry in config["geometry"]["regimes"]:
            local_rows = [lookup[(seed, geometry, x.p, x.h, x.tau)] for x in locals_]
            for J in config["heldout"]["num_segments"]:
                for Q in config["heldout"]["num_queries"]:
                    for epsilon in config["heldout"]["target_errors"]:
                        threshold = slack * epsilon
                        for local, calibration_row in zip(locals_, local_rows):
                            delta, A0, B, source = prediction_moments(calibration_row)
                            A = A0 + (J // local.h - 1) * B
                            max_D = maximum_global_dim(local.p, v, cap)
                            D = minimum_feasible_integer_D(delta, A, threshold, max_D)
                            frontier: Dict[str, Any] = {
                                "seed": seed,
                                "geometry": geometry,
                                "J": J,
                                "Q": Q,
                                "epsilon": epsilon,
                                "selector_threshold": threshold,
                                "p": local.p,
                                "h": local.h,
                                "tau": local.tau,
                                "maximum_D": max_D,
                                "frontier_exists": D is not None,
                                "prediction_delta": delta,
                                "prediction_A": A,
                                "prediction_source": source,
                            }
                            if D is not None:
                                allocation = IntegerAllocation(local.p, local.h, local.tau, D)
                                frontier.update(
                                    D=D,
                                    predicted_error=delta + A / D,
                                    **allocation_resources(allocation, J=J, Q=Q, value_dim=v),
                                )
                            frontiers.append(frontier)
                        for policy in config["selection"]["policies"]:
                            chosen = select_calibrated_allocation(
                                local_rows,
                                policy=policy,
                                J=J,
                                Q=Q,
                                epsilon=epsilon,
                                slack_multiplier=slack,
                                value_dim=v,
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
                            if chosen:
                                record.update(chosen)
                            selections.append(record)
    return frontiers, selections


def run_calibrate_select(config: Mapping[str, Any], config_path: Path, output: Path) -> None:
    if output.exists() and any(path.name != ".gitignore" for path in output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing experiment artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.json").write_bytes(config_path.read_bytes())
    started = time.perf_counter()
    calibration = calibration_rows(config)
    calibration_path = output / "calibration_moments.csv"
    write_csv(calibration_path, calibration)
    frontiers, selections = build_frontiers_and_selections(config, calibration)
    frontier_path = output / "prediction_frontiers.csv"
    write_csv(frontier_path, frontiers)
    artifact = {
        "experiment_id": config["experiment_id"],
        "created_at_utc": utc_now(),
        "leakage_barrier": "sealed before held-out episode generation",
        "input_hashes": input_hashes(config_path),
        "calibration_sha256": sha256(calibration_path),
        "prediction_frontiers_sha256": sha256(frontier_path),
        "selections": selections,
        "fixed_checks": config["fixed_checks"],
    }
    artifact_path = output / "selections_preregistered.json"
    write_json(artifact_path, artifact)
    selection_hash = sha256(artifact_path)
    (output / "selections_preregistered.sha256").write_text(selection_hash + "\n")
    write_json(
        output / "pre_evaluation_provenance.json",
        {
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "input_hashes": artifact["input_hashes"],
            "calibration_sha256": artifact["calibration_sha256"],
            "prediction_frontiers_sha256": artifact["prediction_frontiers_sha256"],
            "selections_preregistered_sha256": selection_hash,
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "git_head": git_text(["rev-parse", "HEAD"]),
            "git_status_before_heldout": git_text(["status", "--short"]),
        },
    )
    print(json.dumps({"calibration_rows": len(calibration), "frontier_rows": len(frontiers), "selection_rows": len(selections), "selection_sha256": selection_hash}, indent=2))


def verify_barrier(config_path: Path, output: Path) -> Dict[str, Any]:
    artifact_path = output / "selections_preregistered.json"
    if not artifact_path.exists():
        raise RuntimeError("missing preregistered selections")
    expected = (output / "selections_preregistered.sha256").read_text().strip()
    if sha256(artifact_path) != expected:
        raise RuntimeError("selection artifact hash mismatch")
    artifact = read_json(artifact_path)
    current = input_hashes(config_path)
    if current != artifact["input_hashes"]:
        raise RuntimeError(f"frozen inputs changed after selection: {current} != {artifact['input_hashes']}")
    if sha256(output / "calibration_moments.csv") != artifact["calibration_sha256"]:
        raise RuntimeError("calibration changed after selection")
    if sha256(output / "prediction_frontiers.csv") != artifact["prediction_frontiers_sha256"]:
        raise RuntimeError("prediction frontiers changed after selection")
    return artifact


def required_sampled_candidates(config: Mapping[str, Any], selections: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str, int, int, int, int], Dict[int, set[str]]]:
    required: Dict[Tuple[int, str, int, int, int, int], Dict[int, set[str]]] = {}
    for row in selections:
        if not bool(row["selected"]):
            continue
        key = (int(row["seed"]), str(row["geometry"]), int(row["J"]), int(row["p"]), int(row["h"]), int(row["tau"]))
        required.setdefault(key, {}).setdefault(int(row["D"]), set()).add(f"selected:{row['policy']}")
    for check in config["fixed_checks"]:
        for seed in config["seeds"]:
            key = (seed, str(check["geometry"]), int(check["J"]), int(check["p"]), int(check["h"]), int(check["tau"]))
            required.setdefault(key, {}).setdefault(int(check["D"]), set()).add(f"fixed:{check['name']}")
    return required


def heldout_trace(config: Mapping[str, Any], *, seed: int, geometry: str, J: int, p: int, h: int, tau: int):
    episodes = int(config["heldout"]["episodes_per_local_configuration"])
    groups = J // h
    v = int(config["grid"]["value_dim"])
    base = int(config["heldout"]["seed_offset"])
    y_gen = torch.Generator().manual_seed(stable_seed(base, seed, J, "values"))
    flat_values = make_values(episodes=episodes, groups=1, h=J, value_dim=v, generator=y_gen)[:, 0]
    values = flat_values.reshape(episodes, groups, h, v)
    x_gen = torch.Generator().manual_seed(stable_seed(base, seed, geometry, J, p, h, "addresses"))
    addresses = make_local_addresses(
        episodes=episodes,
        groups=groups,
        h=h,
        p=p,
        geometry=geometry,
        mixing_coefficient=float(config["geometry"]["correlated_mixing_coefficient"]),
        generator=x_gen,
    )
    return write_local_groups(addresses, values, tau=tau)


def heldout_moments_and_samples(
    config: Mapping[str, Any], selections: Sequence[Mapping[str, Any]], started: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    moments_rows: List[Dict[str, Any]] = []
    sampled_rows: List[Dict[str, Any]] = []
    required = required_sampled_candidates(config, selections)
    key_base = int(config["heldout"]["global_key_seed_offset"])
    limit = float(config["runtime"]["scientific_run_time_limit_seconds"])
    for seed in config["seeds"]:
        for geometry in config["geometry"]["regimes"]:
            for J in config["heldout"]["num_segments"]:
                for local in local_configurations(config):
                    if time.perf_counter() - started > limit:
                        raise RuntimeError("scientific execution time limit reached before completion")
                    trace = heldout_trace(
                        config, seed=seed, geometry=geometry, J=J,
                        p=local.p, h=local.h, tau=local.tau,
                    )
                    episode = heldout_episode_statistics(trace)
                    moments = episode_joint_moments(episode["delta"], episode["A"])
                    row = {
                        "seed": seed,
                        "geometry": geometry,
                        "J": J,
                        "p": local.p,
                        "h": local.h,
                        "tau": local.tau,
                        "commits_per_group": trace.snapshots.shape[2],
                        **moments,
                    }
                    moments_rows.append(row)
                    key = (seed, geometry, J, local.p, local.h, local.tau)
                    if key not in required:
                        continue
                    requested = required[key]
                    max_D = max(requested)
                    key_gen = torch.Generator().manual_seed(
                        stable_seed(key_base, seed, geometry, J, local.p, local.h, local.tau, "global")
                    )
                    raw_keys = torch.randn(
                        trace.snapshots.shape[0], trace.snapshots.shape[1],
                        trace.snapshots.shape[2], max_D,
                        generator=key_gen, dtype=trace.snapshots.dtype,
                    )
                    for D, reasons in sorted(requested.items()):
                        keys = F.normalize(raw_keys[..., :D], dim=-1)
                        sampled = mean_se_ci(folded_la_episode_errors(trace, keys))
                        reference = reference_stats_from_moments(moments, D)
                        sampled_rows.append(
                            {
                                "seed": seed,
                                "geometry": geometry,
                                "J": J,
                                "p": local.p,
                                "h": local.h,
                                "tau": local.tau,
                                "D": D,
                                "reasons": ";".join(sorted(reasons)),
                                "reference_mean": reference["mean"],
                                "reference_se": reference["se"],
                                "sampled_la_mean": sampled["mean"],
                                "sampled_la_se": sampled["se"],
                                "sampled_la_ci_low": sampled["ci_low"],
                                "sampled_la_ci_high": sampled["ci_high"],
                                "sampled_minus_reference": sampled["mean"] - reference["mean"],
                            }
                        )
    return moments_rows, sampled_rows


def moment_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str, int, int, int, int], Mapping[str, Any]]:
    return {
        (int(r["seed"]), str(r["geometry"]), int(r["J"]), int(r["p"]), int(r["h"]), int(r["tau"])): r
        for r in rows
    }


def evaluate_policies(
    config: Mapping[str, Any], selections: Sequence[Mapping[str, Any]], moment_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    lookup = moment_lookup(moment_rows)
    cap = int(config["selection"]["folded_inference_memory_cap_scalars"])
    v = int(config["grid"]["value_dim"])
    z = float(config["acceptance"]["confidence_z"])
    slack = float(config["selection"]["predicted_error_slack_multiplier"])
    rows: List[Dict[str, Any]] = []
    for selection in selections:
        seed, geometry, J, Q = int(selection["seed"]), str(selection["geometry"]), int(selection["J"]), int(selection["Q"])
        epsilon, policy = float(selection["epsilon"]), str(selection["policy"])
        local_moments = [r for r in moment_rows if int(r["seed"]) == seed and str(r["geometry"]) == geometry and int(r["J"]) == J]
        point_oracle = oracle_allocation(local_moments, policy=policy, J=J, Q=Q, threshold=epsilon, value_dim=v, memory_cap=cap, certified=False, z=z)
        certified_oracle = oracle_allocation(local_moments, policy=policy, J=J, Q=Q, threshold=epsilon, value_dim=v, memory_cap=cap, certified=True, z=z)
        slack_oracle = oracle_allocation(local_moments, policy=policy, J=J, Q=Q, threshold=slack * epsilon, value_dim=v, memory_cap=cap, certified=False, z=z)
        uncertain = cheaper_uncertain_allocation(
            local_moments,
            certified_oracle_cost=int(certified_oracle["operation_proxy"]) if certified_oracle else None,
            policy=policy, J=J, Q=Q, threshold=epsilon, value_dim=v, memory_cap=cap, z=z,
        )
        row: Dict[str, Any] = {
            **selection,
            "point_oracle_exists": point_oracle is not None,
            "point_oracle_operation_proxy": point_oracle["operation_proxy"] if point_oracle else None,
            "point_oracle_p": point_oracle["p"] if point_oracle else None,
            "point_oracle_h": point_oracle["h"] if point_oracle else None,
            "point_oracle_tau": point_oracle["tau"] if point_oracle else None,
            "point_oracle_D": point_oracle["D"] if point_oracle else None,
            "certified_oracle_exists": certified_oracle is not None,
            "certified_oracle_operation_proxy": certified_oracle["operation_proxy"] if certified_oracle else None,
            "slack_oracle_exists": slack_oracle is not None,
            "slack_oracle_operation_proxy": slack_oracle["operation_proxy"] if slack_oracle else None,
            "cheaper_interval_uncertain_exists": uncertain is not None,
            "cheaper_interval_uncertain_operation_proxy": uncertain["operation_proxy"] if uncertain else None,
        }
        if not bool(selection["selected"]):
            row.update(
                point_feasible=None,
                interval_status="not_selected",
                slack_point_feasible=None,
                false_feasible=False,
                false_infeasible=point_oracle is not None,
                cost_regret=None,
                certified_cost_regret=None,
                slack_cost_regret=None,
            )
            rows.append(row)
            continue
        key = (seed, geometry, J, int(selection["p"]), int(selection["h"]), int(selection["tau"]))
        moments = lookup[key]
        stats = reference_stats_from_moments(moments, int(selection["D"]), z=z)
        point_feasible = close_leq(stats["mean"], epsilon)
        slack_feasible = close_leq(stats["mean"], slack * epsilon)
        if stats["ci_high"] <= epsilon:
            interval_status = "feasible"
        elif stats["ci_low"] > epsilon:
            interval_status = "infeasible"
        else:
            interval_status = "uncertain"
        selected_cost = int(selection["operation_proxy"])
        row.update(
            reference_mean=stats["mean"],
            reference_se=stats["se"],
            reference_ci_low=stats["ci_low"],
            reference_ci_high=stats["ci_high"],
            point_feasible=point_feasible,
            interval_status=interval_status,
            slack_point_feasible=slack_feasible,
            false_feasible=not point_feasible,
            false_infeasible=False,
            cost_regret=(selected_cost / int(point_oracle["operation_proxy"]) - 1.0) if point_feasible and point_oracle else None,
            certified_cost_regret=(selected_cost / int(certified_oracle["operation_proxy"]) - 1.0) if interval_status == "feasible" and certified_oracle else None,
            slack_cost_regret=(selected_cost / int(slack_oracle["operation_proxy"]) - 1.0) if slack_feasible and slack_oracle else None,
        )
        rows.append(row)
    return rows


def sharing_comparisons(evaluations: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, int, int, float], Dict[str, Mapping[str, Any]]] = {}
    for row in evaluations:
        key = (int(row["seed"]), str(row["geometry"]), int(row["J"]), int(row["Q"]), float(row["epsilon"]))
        grouped.setdefault(key, {})[str(row["policy"])] = row
    output: List[Dict[str, Any]] = []
    for key, policies in grouped.items():
        independent, shared = policies["independent"], policies["shared"]
        both_selected_feasible = bool(independent.get("point_feasible")) and bool(shared.get("point_feasible"))
        both_oracles = bool(independent.get("point_oracle_exists")) and bool(shared.get("point_oracle_exists"))
        output.append(
            {
                "seed": key[0], "geometry": key[1], "J": key[2], "Q": key[3], "epsilon": key[4],
                "independent_selected_p": independent.get("p"), "independent_selected_D": independent.get("D"),
                "independent_selected_cost": independent.get("operation_proxy"), "independent_selected_feasible": independent.get("point_feasible"),
                "shared_selected_p": shared.get("p"), "shared_selected_h": shared.get("h"), "shared_selected_tau": shared.get("tau"), "shared_selected_D": shared.get("D"),
                "shared_selected_cost": shared.get("operation_proxy"), "shared_selected_feasible": shared.get("point_feasible"),
                "selected_both_feasible": both_selected_feasible,
                "selected_shared_to_independent_cost_ratio": (int(shared["operation_proxy"]) / int(independent["operation_proxy"])) if both_selected_feasible else None,
                "selected_shared_win": both_selected_feasible and int(shared["operation_proxy"]) < int(independent["operation_proxy"]),
                "independent_oracle_cost": independent.get("point_oracle_operation_proxy"),
                "shared_oracle_cost": shared.get("point_oracle_operation_proxy"),
                "oracle_both_exist": both_oracles,
                "oracle_shared_to_independent_cost_ratio": (int(shared["point_oracle_operation_proxy"]) / int(independent["point_oracle_operation_proxy"])) if both_oracles else None,
                "oracle_shared_win": both_oracles and int(shared["point_oracle_operation_proxy"]) < int(independent["point_oracle_operation_proxy"]),
            }
        )
    return output


def transfer_diagnostics(
    config: Mapping[str, Any], calibration: Sequence[Mapping[str, Any]], moments: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    calibration_map = calibration_lookup(calibration)
    rows: List[Dict[str, Any]] = []
    cap = int(config["selection"]["folded_inference_memory_cap_scalars"])
    v = int(config["grid"]["value_dim"])
    for held in moments:
        key = (int(held["seed"]), str(held["geometry"]), int(held["p"]), int(held["h"]), int(held["tau"]))
        cal = calibration_map[key]
        delta, A0, B, source = prediction_moments(cal)
        predicted_A = A0 + (int(held["J"]) // int(held["h"]) - 1) * B
        errors = []
        for D in eligible_global_dims(int(held["p"]), v, cap):
            predicted = delta + predicted_A / D
            reference = float(held["delta_mean"]) + float(held["A_mean"]) / D
            errors.append(abs(predicted - reference))
        rows.append(
            {
                "seed": held["seed"], "geometry": held["geometry"], "J": held["J"],
                "p": held["p"], "h": held["h"], "tau": held["tau"],
                "prediction_source": source, "eligible_D_count": len(errors),
                "mean_abs_error_over_D": statistics.fmean(errors),
                "max_abs_error_over_D": max(errors),
                "delta_abs_error": abs(delta - float(held["delta_mean"])),
                "A_abs_error": abs(predicted_A - float(held["A_mean"])),
            }
        )
    return rows


def fixed_check_rows(config: Mapping[str, Any], moments: Sequence[Mapping[str, Any]], sampled: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lookup = moment_lookup(moments)
    sample_lookup = {
        (int(r["seed"]), str(r["geometry"]), int(r["J"]), int(r["p"]), int(r["h"]), int(r["tau"]), int(r["D"])): r
        for r in sampled
    }
    rows: List[Dict[str, Any]] = []
    for check in config["fixed_checks"]:
        if not str(check["name"]).startswith("iid_"):
            continue
        for seed in config["seeds"]:
            key = (seed, str(check["geometry"]), int(check["J"]), int(check["p"]), int(check["h"]), int(check["tau"]))
            stats = reference_stats_from_moments(lookup[key], int(check["D"]))
            sample = sample_lookup[key + (int(check["D"]),)]
            rows.append(
                {
                    "name": check["name"], "seed": seed, **{k: check[k] for k in ("geometry", "J", "p", "h", "tau", "D")},
                    "theory_error": check.get("theory_error"),
                    "reference_mean": stats["mean"], "reference_se": stats["se"],
                    "reference_ci_low": stats["ci_low"], "reference_ci_high": stats["ci_high"],
                    "sampled_la_mean": sample["sampled_la_mean"], "sampled_la_se": sample["sampled_la_se"],
                }
            )
    return rows


def summarize(config: Mapping[str, Any], evaluations: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]], sampled: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    threshold = float(config["acceptance"]["oracle_scenario_coverage_min"])
    policy_summaries: Dict[str, Any] = {}
    for policy in config["selection"]["policies"]:
        rows = [r for r in evaluations if r["policy"] == policy]
        point = oracle_coverage(rows, oracle_key="point_oracle_exists", feasible_key="point_feasible", threshold=threshold)
        slack_cov = oracle_coverage(rows, oracle_key="slack_oracle_exists", feasible_key="slack_point_feasible", threshold=threshold)
        regrets = [float(r["cost_regret"]) for r in rows if r.get("cost_regret") not in (None, "")]
        slack_regrets = [float(r["slack_cost_regret"]) for r in rows if r.get("slack_cost_regret") not in (None, "")]
        certified_regrets = [float(r["certified_cost_regret"]) for r in rows if r.get("certified_cost_regret") not in (None, "")]
        policy_summaries[policy] = {
            "scenarios": len(rows), "selected": sum(bool(r["selected"]) for r in rows),
            "false_feasible": sum(bool(r["false_feasible"]) for r in rows),
            "false_infeasible": sum(bool(r["false_infeasible"]) for r in rows),
            "primary_point_oracle": point,
            "median_feasible_cost_regret": statistics.median(regrets) if regrets else None,
            "max_feasible_cost_regret": max(regrets) if regrets else None,
            "certified_oracle_scenarios": sum(bool(r["certified_oracle_exists"]) for r in rows),
            "median_certified_cost_regret": statistics.median(certified_regrets) if certified_regrets else None,
            "cheaper_interval_uncertain": sum(bool(r["cheaper_interval_uncertain_exists"]) for r in rows),
            "slack_matched_diagnostic": {
                **slack_cov,
                "median_cost_regret": statistics.median(slack_regrets) if slack_regrets else None,
                "max_cost_regret": max(slack_regrets) if slack_regrets else None,
            },
        }
    primary = policy_summaries["unrestricted"]
    passed = (
        primary["primary_point_oracle"]["coverage_passed"]
        and primary["median_feasible_cost_regret"] is not None
        and primary["median_feasible_cost_regret"] <= float(config["acceptance"]["median_feasible_cost_regret_max"])
    )
    by_h: Dict[str, Any] = {}
    for h in config["grid"]["reset_horizons"]:
        rows = [r for r in diagnostics if int(r["h"]) == h]
        by_h[str(h)] = {
            "local_configuration_views": len(rows),
            "median_local_mean_abs_error_over_D": statistics.median(float(r["mean_abs_error_over_D"]) for r in rows),
            "max_local_max_abs_error_over_D": max(float(r["max_abs_error_over_D"]) for r in rows),
        }
    differences = [abs(float(r["sampled_minus_reference"])) for r in sampled]
    return {
        "experiment_id": config["experiment_id"], "completed_at_utc": utc_now(),
        "acceptance_passed": passed, "policy_summaries": policy_summaries,
        "sharing_comparison": {
            "reporting_rows": len(comparisons),
            "selected_shared_wins": sum(bool(r["selected_shared_win"]) for r in comparisons),
            "restricted_oracle_shared_wins": sum(bool(r["oracle_shared_win"]) for r in comparisons),
            "dependent_views_note": "rows reuse episode sets across Q and epsilon",
        },
        "sampled_la_consistency": {
            "unique_rows": len(sampled),
            "median_abs_sampled_minus_reference": statistics.median(differences),
            "max_abs_sampled_minus_reference": max(differences),
        },
        "full_grid_transfer_by_h": by_h,
    }


EVALUATION_ARTIFACTS = (
    "heldout_moments.csv",
    "sampled_la_consistency.csv",
    "policy_evaluation.csv",
    "sharing_comparison.csv",
    "transfer_diagnostics.csv",
    "fixed_checks.csv",
    "summary.json",
    "provenance.json",
)


def refuse_evaluation_overwrite(output: Path) -> None:
    existing = [name for name in EVALUATION_ARTIFACTS if (output / name).exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite existing evaluation artifacts: " + ", ".join(existing)
        )


def run_evaluate(config: Mapping[str, Any], config_path: Path, output: Path) -> None:
    refuse_evaluation_overwrite(output)
    artifact = verify_barrier(config_path, output)
    started = time.perf_counter()
    moments, sampled = heldout_moments_and_samples(config, artifact["selections"], started)
    write_csv(output / "heldout_moments.csv", moments)
    write_csv(output / "sampled_la_consistency.csv", sampled)
    evaluations = evaluate_policies(config, artifact["selections"], moments)
    write_csv(output / "policy_evaluation.csv", evaluations)
    comparisons = sharing_comparisons(evaluations)
    write_csv(output / "sharing_comparison.csv", comparisons)
    calibration = read_csv(output / "calibration_moments.csv")
    diagnostics = transfer_diagnostics(config, calibration, moments)
    write_csv(output / "transfer_diagnostics.csv", diagnostics)
    fixed = fixed_check_rows(config, moments, sampled)
    write_csv(output / "fixed_checks.csv", fixed)
    summary = summarize(config, evaluations, comparisons, sampled, diagnostics)
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "summary.json", summary)
    provenance = {
        "completed_at_utc": utc_now(), "command": " ".join(sys.argv),
        "input_hashes": input_hashes(config_path),
        "config_snapshot_sha256": sha256(output / "config_snapshot.json"),
        "calibration_sha256": sha256(output / "calibration_moments.csv"),
        "prediction_frontiers_sha256": sha256(output / "prediction_frontiers.csv"),
        "selections_preregistered_sha256": sha256(output / "selections_preregistered.json"),
        "heldout_moments_sha256": sha256(output / "heldout_moments.csv"),
        "policy_evaluation_sha256": sha256(output / "policy_evaluation.csv"),
        "summary_sha256": sha256(output / "summary.json"),
        "rng_scheme": "SHA-256-derived disjoint calibration/local-test/global-key streams; held-out values shared over p,h,tau,geometry for seed/J; addresses shared over tau; global raw Gaussian prefixes shared over requested D.",
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "git_head": git_text(["rev-parse", "HEAD"]), "git_status": git_text(["status", "--short"]),
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
