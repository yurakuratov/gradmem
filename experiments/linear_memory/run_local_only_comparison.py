"""Leakage-safe two-stage runner for EXP-LM-004."""

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

from .allocation import mean_se_ci, stable_seed
from .local_only_comparison import (
    METHOD_ORDER,
    MethodAllocation,
    coverage,
    endpoint_delta_error_aggregate,
    hierarchy_moments,
    local_additive_error,
    method_frontiers,
    resource_row,
    select_methods,
    simulate_hierarchy,
    simulate_local_additive,
    simulate_local_delta,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("local_only_comparison_protocol.md")
THEORY_CODE = Path(__file__).with_name("local_only_comparison.py")
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
        "theory_code": sha256(THEORY_CODE),
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


def allocation_from_row(row: Mapping[str, Any]) -> MethodAllocation:
    return MethodAllocation(
        str(row["method"]),
        int(row["p"]),
        int(row["h"]) if row.get("h") not in (None, "") else None,
        int(row["tau"]) if row.get("tau") not in (None, "") else None,
        int(row["D"]) if row.get("D") not in (None, "") else None,
    )


def allocation_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    allocation = allocation_from_row(row)
    return (
        allocation.method,
        int(row["J"]),
        allocation.p,
        allocation.h,
        allocation.tau,
        allocation.D,
    )


def theory_error(allocation: MethodAllocation, J: int) -> float:
    if allocation.method == "local_delta":
        return endpoint_delta_error_aggregate(allocation.p, J)
    if allocation.method == "local_additive":
        return local_additive_error(allocation.p, J)
    if allocation.method == "hierarchical":
        assert allocation.h is not None and allocation.tau is not None and allocation.D is not None
        moments = hierarchy_moments(allocation.p, allocation.h, allocation.tau, J)
        return moments["delta"] + moments["A"] / allocation.D
    raise ValueError(f"unknown method {allocation.method}")


def enforce_runtime_limit(started: float, config: Mapping[str, Any], stage: str) -> None:
    limit = float(config["runtime"]["scientific_stage_time_limit_seconds"])
    elapsed = time.perf_counter() - started
    if elapsed > limit:
        raise RuntimeError(f"{stage} exceeded registered {limit:g}s runtime limit ({elapsed:.3f}s)")


def build_analytic_artifacts(
    config: Mapping[str, Any], *, started: float | None = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    started = time.perf_counter() if started is None else started
    task, allocation_config = config["task"], config["allocation"]
    cap, v = int(allocation_config["mutable_state_cap_scalars"]), int(task["value_dim"])
    frontiers: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    for J in task["num_facts"]:
        for Q in task["num_queries"]:
            for epsilon in task["target_errors"]:
                for multiplier in allocation_config["threshold_multipliers"]:
                    enforce_runtime_limit(started, config, "predict-select")
                    threshold = float(multiplier) * float(epsilon)
                    local_frontiers, chosen = select_methods(
                        J=int(J), Q=int(Q), threshold=threshold,
                        value_dim=v, memory_cap=cap,
                    )
                    view = {
                        "J": J, "Q": Q, "epsilon": epsilon,
                        "threshold_multiplier": multiplier, "threshold": threshold,
                    }
                    frontiers.extend({**view, **row} for row in local_frontiers)
                    selections.extend({**view, **row} for row in chosen)
    return frontiers, selections


def build_evaluation_union(config: Mapping[str, Any], selections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cap = int(config["allocation"]["mutable_state_cap_scalars"])
    v = int(config["task"]["value_dim"])
    union: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def add(row: Mapping[str, Any], source: str) -> None:
        record = dict(row)
        record.pop("method_scope", None)
        record["J"] = int(record["J"])
        key = allocation_key(record)
        if key in union:
            sources = set(str(union[key]["sources"]).split(";"))
            sources.add(source)
            union[key]["sources"] = ";".join(sorted(sources))
        else:
            allocation = allocation_from_row(record)
            population = {
                "method": allocation.method, "J": int(record["J"]), "p": allocation.p,
                "h": allocation.h, "tau": allocation.tau, "D": allocation.D,
                "population_error": theory_error(allocation, int(record["J"])),
                "sources": source,
            }
            if allocation.method == "hierarchical":
                assert allocation.h is not None and allocation.tau is not None
                moments = hierarchy_moments(allocation.p, allocation.h, allocation.tau, int(record["J"]))
                population.update(population_delta=moments["delta"], population_A=moments["A"])
            union[key] = population

    for row in selections:
        if row.get("method_scope") == "overall" or not bool(row.get("selected", True)):
            continue
        source = f"selection:m{row['threshold_multiplier']}:Q{row['Q']}:e{row['epsilon']}"
        add(row, source)
        if math.isclose(float(row["threshold_multiplier"]), 1.0):
            allocation = allocation_from_row(row)
            if allocation.method in ("local_delta", "local_additive"):
                maximum_p = cap // v
                for neighbor_p in (allocation.p - 1, allocation.p + 1):
                    if 1 <= neighbor_p <= maximum_p:
                        add({"method": allocation.method, "J": row["J"], "p": neighbor_p}, f"primary_neighbor:p{neighbor_p}")
            else:
                assert allocation.D is not None
                maximum_D = cap // (allocation.p * v) - 1
                for neighbor_D in (allocation.D - 1, allocation.D + 1):
                    if 1 <= neighbor_D <= maximum_D:
                        add(
                            {"method": allocation.method, "J": row["J"], "p": allocation.p,
                             "h": allocation.h, "tau": allocation.tau, "D": neighbor_D},
                            f"primary_neighbor:D{neighbor_D}",
                        )
    for check in config["fixed_checks"]:
        add(check, f"fixed:{check['name']}")
    return sorted(
        union.values(),
        key=lambda row: (
            int(row["J"]), METHOD_ORDER[str(row["method"])], int(row["p"]),
            int(row.get("h") or 0), int(row.get("tau") or 0), int(row.get("D") or 0),
        ),
    )


def run_predict_select(config: Mapping[str, Any], config_path: Path, output: Path) -> None:
    if output.exists() and any(path.name != ".gitignore" for path in output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing experiment artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.json").write_bytes(config_path.read_bytes())
    started = time.perf_counter()
    frontiers, selections = build_analytic_artifacts(config, started=started)
    enforce_runtime_limit(started, config, "predict-select")
    frontier_path = output / "analytic_frontiers.csv"
    write_csv(frontier_path, frontiers)
    union = build_evaluation_union(config, selections)
    artifact = {
        "experiment_id": config["experiment_id"],
        "created_at_utc": utc_now(),
        "leakage_barrier": "exact theory selections sealed before held-out generation",
        "input_hashes": input_hashes(config_path),
        "analytic_frontiers_sha256": sha256(frontier_path),
        "selections": selections,
        "evaluation_union": union,
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
            "analytic_frontiers_sha256": artifact["analytic_frontiers_sha256"],
            "selections_preregistered_sha256": selection_hash,
            "frontier_rows": len(frontiers),
            "selection_rows": len(selections),
            "evaluation_configurations": len(union),
            "command": " ".join(sys.argv),
        },
    )
    print(json.dumps({"selection_sha256": selection_hash, "frontier_rows": len(frontiers), "evaluation_configurations": len(union)}, indent=2))


def verify_barrier(config_path: Path, output: Path) -> Dict[str, Any]:
    artifact_path = output / "selections_preregistered.json"
    artifact = read_json(artifact_path)
    recorded_selection_hash = (output / "selections_preregistered.sha256").read_text().strip()
    if sha256(artifact_path) != recorded_selection_hash:
        raise RuntimeError("sealed selection artifact hash mismatch")
    current_hashes = input_hashes(config_path)
    if current_hashes != artifact["input_hashes"]:
        raise RuntimeError("frozen inputs changed after selection")
    if sha256(output / "analytic_frontiers.csv") != artifact["analytic_frontiers_sha256"]:
        raise RuntimeError("analytic frontier changed after selection")
    if sha256(output / "config_snapshot.json") != sha256(config_path):
        raise RuntimeError("config snapshot mismatch")
    return artifact


EVALUATION_ARTIFACTS = (
    "heldout_evaluation.csv",
    "selection_evaluation.csv",
    "pairwise_comparison.csv",
    "fixed_checks.csv",
    "summary.json",
    "provenance.json",
)


def refuse_evaluation_overwrite(output: Path) -> None:
    existing = [name for name in EVALUATION_ARTIFACTS if (output / name).exists()]
    if existing:
        raise RuntimeError("refusing to overwrite existing evaluation artifacts: " + ", ".join(existing))


def unit_random(shape: Sequence[int], generator: torch.Generator) -> torch.Tensor:
    return F.normalize(torch.randn(*shape, generator=generator, dtype=torch.float64), dim=-1)


def heldout_streams(config: Mapping[str, Any], *, seed: int, J: int, p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    heldout = config["heldout"]
    episodes, v = int(heldout["episodes_per_configuration"]), int(config["task"]["value_dim"])
    x_generator = torch.Generator().manual_seed(
        stable_seed(heldout["local_seed_offset"], seed, J, p, "flat_addresses")
    )
    y_generator = torch.Generator().manual_seed(
        stable_seed(heldout["value_seed_offset"], seed, J, "flat_values")
    )
    return unit_random((episodes, J, p), x_generator), unit_random((episodes, J, v), y_generator)


def global_keys(config: Mapping[str, Any], *, seed: int, allocation: MethodAllocation, J: int) -> torch.Tensor:
    assert allocation.h is not None and allocation.tau is not None and allocation.D is not None
    episodes = int(config["heldout"]["episodes_per_configuration"])
    groups, commits = J // allocation.h, allocation.h // allocation.tau
    generator = torch.Generator().manual_seed(
        stable_seed(
            config["heldout"]["global_key_seed_offset"], seed, J, allocation.p,
            allocation.h, allocation.tau, allocation.D, "global_keys",
        )
    )
    return unit_random((episodes, groups, commits, allocation.D), generator)


def heldout_rows(
    config: Mapping[str, Any], evaluation_union: Sequence[Mapping[str, Any]], *, started: float
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    z = float(config["heldout"]["confidence_z"])
    for seed in config["seeds"]:
        for J in config["task"]["num_facts"]:
            configs_J = [row for row in evaluation_union if int(row["J"]) == int(J)]
            for p in sorted({int(row["p"]) for row in configs_J}):
                enforce_runtime_limit(started, config, "evaluate")
                addresses, values = heldout_streams(config, seed=int(seed), J=int(J), p=p)
                local_samples: Dict[str, torch.Tensor] = {}
                if any(row["method"] == "local_delta" for row in configs_J if int(row["p"]) == p):
                    local_samples["local_delta"] = simulate_local_delta(addresses, values)
                if any(row["method"] == "local_additive" for row in configs_J if int(row["p"]) == p):
                    local_samples["local_additive"] = simulate_local_additive(addresses, values)
                for evaluation in [row for row in configs_J if int(row["p"]) == p]:
                    allocation = allocation_from_row(evaluation)
                    sampled_stats: Dict[str, float] = {}
                    if allocation.method == "hierarchical":
                        keys = global_keys(config, seed=int(seed), allocation=allocation, J=int(J))
                        reference, sampled, local_delta, distractor_A = simulate_hierarchy(
                            addresses, values, h=int(allocation.h), tau=int(allocation.tau),
                            D=int(allocation.D), global_keys=keys,
                        )
                        samples = reference
                        sampled_stats = {f"sampled_la_{key}": value for key, value in mean_se_ci(sampled, z=z).items()}
                        sampled_stats["sampled_minus_reference"] = float(sampled.mean() - reference.mean())
                        delta_stats = mean_se_ci(local_delta, z=z)
                        A_stats = mean_se_ci(distractor_A, z=z)
                        sampled_stats.update(
                            empirical_delta_mean=delta_stats["mean"],
                            empirical_delta_se=delta_stats["se"],
                            empirical_A_mean=A_stats["mean"],
                            empirical_A_se=A_stats["se"],
                        )
                    else:
                        samples = local_samples[allocation.method]
                    stats = mean_se_ci(samples, z=z)
                    rows.append(
                        {
                            "seed": seed, **evaluation,
                            "episodes": int(config["heldout"]["episodes_per_configuration"]),
                            "empirical_mean": stats["mean"], "empirical_se": stats["se"],
                            "empirical_ci_low": stats["ci_low"], "empirical_ci_high": stats["ci_high"],
                            "prediction_residual": stats["mean"] - float(evaluation["population_error"]),
                            **sampled_stats,
                        }
                    )
    return rows


def interval_status(row: Mapping[str, Any], epsilon: float) -> str:
    if float(row["empirical_ci_high"]) <= epsilon:
        return "feasible"
    if float(row["empirical_ci_low"]) > epsilon:
        return "infeasible"
    return "uncertain"


def selection_evaluation_rows(
    config: Mapping[str, Any], artifact: Mapping[str, Any], heldout: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    lookup = {(int(row["seed"]),) + allocation_key(row): row for row in heldout}
    primary_lookup = {
        (int(row["J"]), int(row["Q"]), float(row["epsilon"]), str(row["method"])): row
        for row in artifact["selections"]
        if row.get("method_scope") != "overall"
        and math.isclose(float(row["threshold_multiplier"]), 1.0)
    }
    output: List[Dict[str, Any]] = []
    for selection in artifact["selections"]:
        if selection.get("method_scope") == "overall":
            continue
        primary = primary_lookup[(
            int(selection["J"]), int(selection["Q"]), float(selection["epsilon"]), str(selection["method"])
        )]
        population_oracle_exists = bool(primary.get("selected", True))
        for seed in config["seeds"]:
            if not bool(selection.get("selected", True)):
                output.append(
                    {
                        "seed": seed, **selection,
                        "deployment_epsilon": float(selection["epsilon"]),
                        "population_oracle_exists": population_oracle_exists,
                        "empirical_feasible": False,
                        "direct_sampled_feasible": False,
                        "interval_status": "not_selected",
                        "direct_sampled_interval_status": "not_selected",
                        "reserve_target_empirical_feasible": False,
                        "own_threshold_operation_regret": None,
                        "reserve_overhead_vs_primary_epsilon_optimum": None,
                    }
                )
                continue
            observed = lookup[(int(seed),) + allocation_key(selection)]
            epsilon = float(selection["epsilon"])
            primary_cost = int(primary["operation_proxy"]) if population_oracle_exists else None
            reserve_overhead = (
                int(selection["operation_proxy"]) / primary_cost - 1.0
                if primary_cost is not None
                and math.isclose(float(selection["threshold_multiplier"]), 0.95)
                else 0.0 if primary_cost is not None else None
            )
            if selection["method"] == "hierarchical":
                direct_mean = float(observed["sampled_la_mean"])
                direct_low = float(observed["sampled_la_ci_low"])
                direct_high = float(observed["sampled_la_ci_high"])
                direct_status = (
                    "feasible" if direct_high <= epsilon
                    else "infeasible" if direct_low > epsilon
                    else "uncertain"
                )
            else:
                direct_mean = float(observed["empirical_mean"])
                direct_status = interval_status(observed, epsilon)
            output.append(
                {
                    "seed": seed,
                    **selection,
                    "deployment_epsilon": epsilon,
                    "population_oracle_exists": population_oracle_exists,
                    "empirical_mean": observed["empirical_mean"],
                    "empirical_se": observed["empirical_se"],
                    "empirical_ci_low": observed["empirical_ci_low"],
                    "empirical_ci_high": observed["empirical_ci_high"],
                    "prediction_residual": observed["prediction_residual"],
                    "empirical_feasible": float(observed["empirical_mean"]) <= epsilon,
                    "direct_sampled_mean": direct_mean,
                    "direct_sampled_feasible": direct_mean <= epsilon,
                    "interval_status": interval_status(observed, epsilon),
                    "direct_sampled_interval_status": direct_status,
                    "reserve_target_empirical_feasible": float(observed["empirical_mean"]) <= float(selection["threshold"]),
                    "own_threshold_operation_regret": 0.0,
                    "own_threshold_regret_note": "zero by construction: exact optimum within this method at its registered threshold",
                    "reserve_overhead_vs_primary_epsilon_optimum": reserve_overhead,
                }
            )
    return output


def pairwise_rows(config: Mapping[str, Any], selections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    methods = tuple(METHOD_ORDER)
    restricted = [r for r in selections if r.get("method_scope") != "overall" and bool(r.get("selected", True))]
    for J in config["task"]["num_facts"]:
        for Q in config["task"]["num_queries"]:
            for epsilon in config["task"]["target_errors"]:
                for multiplier in config["allocation"]["threshold_multipliers"]:
                    view = [
                        r for r in restricted
                        if int(r["J"]) == J and int(r["Q"]) == Q
                        and float(r["epsilon"]) == epsilon
                        and float(r["threshold_multiplier"]) == multiplier
                    ]
                    by_method = {str(r["method"]): r for r in view}
                    for left_index, left in enumerate(methods):
                        for right in methods[left_index + 1:]:
                            if left not in by_method or right not in by_method:
                                continue
                            a, b = by_method[left], by_method[right]
                            cost_a, cost_b = int(a["operation_proxy"]), int(b["operation_proxy"])
                            output.append(
                                {
                                    "J": J, "Q": Q, "epsilon": epsilon,
                                    "threshold_multiplier": multiplier,
                                    "left_method": left, "right_method": right,
                                    "left_cost": cost_a, "right_cost": cost_b,
                                    "left_memory": int(a["mutable_state_scalars"]),
                                    "right_memory": int(b["mutable_state_scalars"]),
                                    "winner": left if cost_a < cost_b else right if cost_b < cost_a else "tie",
                                    "right_over_left_cost_ratio": cost_b / cost_a,
                                }
                            )
    return output


def fixed_check_rows(config: Mapping[str, Any], heldout: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in heldout:
        fixed = [source for source in str(row["sources"]).split(";") if source.startswith("fixed:")]
        for source in fixed:
            output.append({"name": source.split(":", 1)[1], **row})
    return output


def summarize(
    config: Mapping[str, Any], heldout: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    minimum = float(config["acceptance"]["reserve_selection_deployment_epsilon_coverage_min"])
    reserve = [row for row in selections if math.isclose(float(row["threshold_multiplier"]), 0.95)]
    primary = [row for row in selections if math.isclose(float(row["threshold_multiplier"]), 1.0)]
    by_method: Dict[str, Any] = {}
    for method in METHOD_ORDER:
        reserve_rows = [row for row in reserve if row["method"] == method]
        primary_rows = [row for row in primary if row["method"] == method]
        direct_rows = [
            {**row, "empirical_feasible": row.get("direct_sampled_feasible", False)}
            for row in reserve_rows
        ]
        overheads = [
            float(row["reserve_overhead_vs_primary_epsilon_optimum"])
            for row in reserve_rows
            if row.get("reserve_overhead_vs_primary_epsilon_optimum") is not None
        ]
        by_method[method] = {
            "reserve_deployment_epsilon_coverage": coverage(reserve_rows, threshold=minimum),
            "reserve_direct_sampled_deployment_epsilon_coverage": coverage(direct_rows, threshold=minimum),
            "reserve_target_empirically_feasible": sum(bool(r["reserve_target_empirical_feasible"]) for r in reserve_rows),
            "reserve_interval_feasible": sum(r["interval_status"] == "feasible" for r in reserve_rows),
            "reserve_cost_overhead_vs_primary": {
                "median": statistics.median(overheads) if overheads else None,
                "maximum": max(overheads) if overheads else None,
            },
            "primary_deployment_point_feasible": sum(bool(r["empirical_feasible"]) for r in primary_rows),
            "primary_views": len(primary_rows),
        }
    residuals = [abs(float(row["prediction_residual"])) for row in heldout]
    la_differences = [
        abs(float(row["sampled_minus_reference"]))
        for row in heldout if row["method"] == "hierarchical"
    ]
    passed = all(by_method[m]["reserve_deployment_epsilon_coverage"]["passed"] for m in METHOD_ORDER)
    return {
        "experiment_id": config["experiment_id"],
        "completed_at_utc": utc_now(),
        "acceptance_passed": passed,
        "method_summaries": by_method,
        "unique_evaluated_configurations": len({allocation_key(row) for row in heldout}),
        "heldout_rows": len(heldout),
        "median_absolute_prediction_residual": statistics.median(residuals),
        "maximum_absolute_prediction_residual": max(residuals),
        "sampled_la_consistency": {
            "rows": len(la_differences),
            "median_absolute_difference": statistics.median(la_differences),
            "maximum_absolute_difference": max(la_differences),
        },
        "pairwise_rows": len(pairs),
        "dependence_note": "Q, epsilon, method, and neighbor views reuse episode streams and are not independent replications",
    }


def run_evaluate(config: Mapping[str, Any], config_path: Path, output: Path) -> None:
    refuse_evaluation_overwrite(output)
    artifact = verify_barrier(config_path, output)
    started = time.perf_counter()
    heldout = heldout_rows(config, artifact["evaluation_union"], started=started)
    enforce_runtime_limit(started, config, "evaluate")
    write_csv(output / "heldout_evaluation.csv", heldout)
    selection_eval = selection_evaluation_rows(config, artifact, heldout)
    write_csv(output / "selection_evaluation.csv", selection_eval)
    pairs = pairwise_rows(config, artifact["selections"])
    write_csv(output / "pairwise_comparison.csv", pairs)
    fixed = fixed_check_rows(config, heldout)
    write_csv(output / "fixed_checks.csv", fixed)
    summary = summarize(config, heldout, selection_eval, pairs)
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "summary.json", summary)
    write_json(
        output / "provenance.json",
        {
            "completed_at_utc": utc_now(), "command": " ".join(sys.argv),
            "input_hashes": input_hashes(config_path),
            "config_snapshot_sha256": sha256(output / "config_snapshot.json"),
            "analytic_frontiers_sha256": sha256(output / "analytic_frontiers.csv"),
            "selections_preregistered_sha256": sha256(output / "selections_preregistered.json"),
            "heldout_evaluation_sha256": sha256(output / "heldout_evaluation.csv"),
            "selection_evaluation_sha256": sha256(output / "selection_evaluation.csv"),
            "summary_sha256": sha256(output / "summary.json"),
            "rng_scheme": "SHA-256-derived p-specific flat IID key streams, J-specific values shared across p/method/h, and separate configuration-specific global-key streams",
            "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
            "git_head": git_text(["rev-parse", "HEAD"]), "git_status": git_text(["status", "--short"]),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("predict-select", "evaluate"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    configure_runtime(config)
    output = (ROOT / config["output_dir"]).resolve()
    if args.stage == "predict-select":
        run_predict_select(config, config_path, output)
    else:
        run_evaluate(config, config_path, output)


if __name__ == "__main__":
    main()
