"""Exact integer allocation and moment utilities for EXP-LM-003."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from .allocation import AllocationCandidate, candidate_resources, valid_taus


@dataclass(frozen=True, order=True)
class LocalConfiguration:
    p: int
    h: int
    tau: int


@dataclass(frozen=True, order=True)
class IntegerAllocation:
    p: int
    h: int
    tau: int
    D: int


def local_configurations(config: Mapping[str, Any]) -> List[LocalConfiguration]:
    grid = config["grid"]
    return [
        LocalConfiguration(p, h, tau)
        for p in grid["local_dims"]
        for h in grid["reset_horizons"]
        for tau in valid_taus(h)
    ]


def maximum_global_dim(p: int, value_dim: int, memory_cap: int) -> int:
    payload = p * value_dim
    return memory_cap // payload - 1


def eligible_global_dims(p: int, value_dim: int, memory_cap: int) -> range:
    maximum = maximum_global_dim(p, value_dim, memory_cap)
    return range(1, max(0, maximum) + 1)


def close_leq(value: float, threshold: float) -> bool:
    """Inclusive comparison allowing only floating representation error."""

    return value <= threshold or math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-15
    )


def validate_nonnegative(value: float, *, name: str, scale: float = 1.0) -> float:
    tolerance = 1e-12 * max(1.0, abs(scale))
    if value >= 0.0:
        return value
    if value >= -tolerance:
        return 0.0
    raise ValueError(f"materially negative {name}: {value}")


def clamp_nonnegative_samples(samples: Tensor, *, name: str) -> Tensor:
    scale = max(1.0, float(samples.detach().abs().max()))
    minimum = float(samples.detach().min())
    validate_nonnegative(minimum, name=name, scale=scale)
    return samples.clamp_min(0.0)


def minimum_feasible_integer_D(
    delta: float,
    A: float,
    threshold: float,
    maximum_D: int,
) -> int | None:
    """Exact bounded integer search with explicit zero-noise boundary rules."""

    A = validate_nonnegative(A, name="A", scale=max(abs(delta), abs(threshold), abs(A)))
    if maximum_D < 1:
        return None
    if A == 0.0:
        return 1 if close_leq(delta, threshold) else None
    if delta >= threshold:
        return None
    estimate = max(1, math.ceil(A / (threshold - delta)))
    # Inspect neighbors to guard ceil errors at exactly representable boundaries.
    for D in range(max(1, estimate - 2), min(maximum_D, estimate + 2) + 1):
        if close_leq(delta + A / D, threshold):
            if D == 1 or not close_leq(delta + A / (D - 1), threshold):
                return D
    # A bounded enumeration is cheap (D<=511 in the registered grid) and is
    # the final authority if the ratio was numerically awkward.
    for D in range(1, maximum_D + 1):
        if close_leq(delta + A / D, threshold):
            return D
    return None


def policy_allows(local: LocalConfiguration, policy: str) -> bool:
    if policy == "unrestricted":
        return True
    if policy == "independent":
        return local.h == 1 and local.tau == 1
    if policy == "shared":
        return local.h > 1
    raise ValueError(f"unknown policy: {policy}")


def prediction_moments(row: Mapping[str, Any]) -> Tuple[float, float, float, str]:
    p, h = int(row["p"]), int(row["h"])
    if h == 1:
        return 0.0, 0.0, 1.0 / p, "exact_h1_identity"
    return (
        float(row["delta_mean"]),
        float(row["A0_mean"]),
        float(row["B_mean"]),
        "empirical_calibration",
    )


def allocation_resources(
    allocation: IntegerAllocation, *, J: int, Q: int, value_dim: int
) -> Dict[str, int]:
    return candidate_resources(
        AllocationCandidate(allocation.p, allocation.h, allocation.tau, allocation.D),
        J=J,
        Q=Q,
        value_dim=value_dim,
    )


def allocation_sort_key(row: Mapping[str, Any]) -> Tuple[int, ...]:
    return (
        int(row["operation_proxy"]),
        int(row["folded_inference_memory_scalars"]),
        int(row["p"]),
        int(row["h"]),
        int(row["tau"]),
        int(row["D"]),
    )


def select_calibrated_allocation(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    J: int,
    Q: int,
    epsilon: float,
    slack_multiplier: float,
    value_dim: int,
    memory_cap: int,
) -> Dict[str, Any] | None:
    threshold = slack_multiplier * epsilon
    candidates: List[Dict[str, Any]] = []
    for row in calibration_rows:
        local = LocalConfiguration(int(row["p"]), int(row["h"]), int(row["tau"]))
        if not policy_allows(local, policy):
            continue
        delta, A0, B, source = prediction_moments(row)
        groups = J // local.h
        A = A0 + (groups - 1) * B
        maximum_D = maximum_global_dim(local.p, value_dim, memory_cap)
        D = minimum_feasible_integer_D(delta, A, threshold, maximum_D)
        if D is None:
            continue
        allocation = IntegerAllocation(local.p, local.h, local.tau, D)
        resources = allocation_resources(allocation, J=J, Q=Q, value_dim=value_dim)
        candidates.append(
            {
                "p": local.p,
                "h": local.h,
                "tau": local.tau,
                "D": D,
                "predicted_error": delta + A / D,
                "selector_threshold": threshold,
                "prediction_delta": delta,
                "prediction_A": A,
                "prediction_source": source,
                **resources,
            }
        )
    return min(candidates, key=allocation_sort_key, default=None)


def brute_force_calibrated_allocation(
    calibration_rows: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any] | None:
    """Test/reference implementation enumerating every integer D."""

    policy = str(kwargs["policy"])
    J, Q = int(kwargs["J"]), int(kwargs["Q"])
    epsilon = float(kwargs["epsilon"])
    slack = float(kwargs["slack_multiplier"])
    value_dim, cap = int(kwargs["value_dim"]), int(kwargs["memory_cap"])
    candidates: List[Dict[str, Any]] = []
    for row in calibration_rows:
        local = LocalConfiguration(int(row["p"]), int(row["h"]), int(row["tau"]))
        if not policy_allows(local, policy):
            continue
        delta, A0, B, source = prediction_moments(row)
        A = A0 + (J // local.h - 1) * B
        for D in eligible_global_dims(local.p, value_dim, cap):
            error = delta + A / D
            if not close_leq(error, slack * epsilon):
                continue
            allocation = IntegerAllocation(local.p, local.h, local.tau, D)
            candidates.append(
                {
                    "p": local.p,
                    "h": local.h,
                    "tau": local.tau,
                    "D": D,
                    "predicted_error": error,
                    "prediction_source": source,
                    **allocation_resources(allocation, J=J, Q=Q, value_dim=value_dim),
                }
            )
    return min(candidates, key=allocation_sort_key, default=None)


def episode_joint_moments(delta: Tensor, A: Tensor) -> Dict[str, float]:
    delta = delta.detach().to(torch.float64).flatten()
    A = clamp_nonnegative_samples(A.detach().to(torch.float64).flatten(), name="episode A")
    if delta.shape != A.shape or delta.numel() < 2:
        raise ValueError("delta and A need matching episode vectors with n>=2")
    centered_delta = delta - delta.mean()
    centered_A = A - A.mean()
    n = delta.numel()
    return {
        "episodes": n,
        "delta_mean": float(delta.mean()),
        "A_mean": float(A.mean()),
        "delta_variance": float(centered_delta.square().sum() / (n - 1)),
        "A_variance": float(centered_A.square().sum() / (n - 1)),
        "delta_A_covariance": float((centered_delta * centered_A).sum() / (n - 1)),
    }


def reference_stats_from_moments(
    moments: Mapping[str, Any], D: int, *, z: float = 1.96
) -> Dict[str, float]:
    if D < 1:
        raise ValueError("D must be positive")
    mean = float(moments["delta_mean"]) + float(moments["A_mean"]) / D
    variance = (
        float(moments["delta_variance"])
        + 2.0 * float(moments["delta_A_covariance"]) / D
        + float(moments["A_variance"]) / (D * D)
    )
    scale = (
        abs(float(moments["delta_variance"]))
        + 2.0 * abs(float(moments["delta_A_covariance"])) / D
        + abs(float(moments["A_variance"])) / (D * D)
    )
    variance = validate_nonnegative(variance, name="reference variance", scale=scale)
    se = math.sqrt(variance / int(moments["episodes"]))
    return {
        "mean": mean,
        "variance": variance,
        "se": se,
        "ci_low": mean - z * se,
        "ci_high": mean + z * se,
    }


def direct_reference_stats(delta: Tensor, A: Tensor, D: int, *, z: float = 1.96) -> Dict[str, float]:
    samples = delta.detach().to(torch.float64).flatten() + clamp_nonnegative_samples(
        A.detach().to(torch.float64).flatten(), name="episode A"
    ) / D
    variance = float(samples.var(unbiased=True))
    se = math.sqrt(variance / samples.numel())
    mean = float(samples.mean())
    return {
        "mean": mean,
        "variance": variance,
        "se": se,
        "ci_low": mean - z * se,
        "ci_high": mean + z * se,
    }


def oracle_allocation(
    moment_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    J: int,
    Q: int,
    threshold: float,
    value_dim: int,
    memory_cap: int,
    certified: bool,
    z: float = 1.96,
) -> Dict[str, Any] | None:
    """Enumerate all integer D; certified mode never assumes CI monotonicity."""

    feasible: List[Dict[str, Any]] = []
    for moments in moment_rows:
        local = LocalConfiguration(
            int(moments["p"]), int(moments["h"]), int(moments["tau"])
        )
        if not policy_allows(local, policy):
            continue
        for D in eligible_global_dims(local.p, value_dim, memory_cap):
            stats = reference_stats_from_moments(moments, D, z=z)
            criterion = stats["ci_high"] if certified else stats["mean"]
            if not close_leq(criterion, threshold):
                continue
            allocation = IntegerAllocation(local.p, local.h, local.tau, D)
            feasible.append(
                {
                    "p": local.p,
                    "h": local.h,
                    "tau": local.tau,
                    "D": D,
                    "reference_mean": stats["mean"],
                    "reference_ci_low": stats["ci_low"],
                    "reference_ci_high": stats["ci_high"],
                    **allocation_resources(allocation, J=J, Q=Q, value_dim=value_dim),
                }
            )
    return min(feasible, key=allocation_sort_key, default=None)


def cheaper_uncertain_allocation(
    moment_rows: Sequence[Mapping[str, Any]],
    *,
    certified_oracle_cost: int | None,
    policy: str,
    J: int,
    Q: int,
    threshold: float,
    value_dim: int,
    memory_cap: int,
    z: float = 1.96,
) -> Dict[str, Any] | None:
    if certified_oracle_cost is None:
        return None
    uncertain: List[Dict[str, Any]] = []
    for moments in moment_rows:
        local = LocalConfiguration(int(moments["p"]), int(moments["h"]), int(moments["tau"]))
        if not policy_allows(local, policy):
            continue
        for D in eligible_global_dims(local.p, value_dim, memory_cap):
            allocation = IntegerAllocation(local.p, local.h, local.tau, D)
            resources = allocation_resources(allocation, J=J, Q=Q, value_dim=value_dim)
            if resources["operation_proxy"] >= certified_oracle_cost:
                continue
            stats = reference_stats_from_moments(moments, D, z=z)
            if stats["ci_low"] <= threshold < stats["ci_high"]:
                uncertain.append(
                    {
                        "p": local.p,
                        "h": local.h,
                        "tau": local.tau,
                        "D": D,
                        **stats,
                        **resources,
                    }
                )
    return min(uncertain, key=allocation_sort_key, default=None)


def oracle_coverage(rows: Sequence[Mapping[str, Any]], *, oracle_key: str, feasible_key: str, threshold: float) -> Dict[str, Any]:
    oracle_rows = [row for row in rows if bool(row.get(oracle_key))]
    covered = [
        row
        for row in oracle_rows
        if bool(row.get("selected")) and bool(row.get(feasible_key))
    ]
    coverage = len(covered) / len(oracle_rows) if oracle_rows else None
    return {
        "oracle_existing_scenarios": len(oracle_rows),
        "oracle_existing_with_feasible_selection": len(covered),
        "coverage": coverage,
        "coverage_threshold": threshold,
        "coverage_passed": coverage is not None and coverage >= threshold,
    }


def iid_h2_endpoint_moments(p: int) -> Dict[str, float]:
    return {
        "delta": 0.5 * (1.0 / p + 3.0 / (p * (p + 2))),
        "A0": 0.0,
        "B": (2.0 - 1.0 / p) / p,
    }
