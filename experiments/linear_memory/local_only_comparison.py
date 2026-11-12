"""Exact population theory and allocation for EXP-LM-004.

The module compares an endpoint delta writer, a direct additive writer, and
the delta-writer plus folded linear-attention hierarchy.  It deliberately
contains no fitted quantities: every selection is made from IID population
moments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from .allocation import (
    folded_la_episode_errors,
    heldout_episode_statistics,
    write_local_groups,
)


METHOD_ORDER = {"local_delta": 0, "local_additive": 1, "hierarchical": 2}


@dataclass(frozen=True, order=True)
class MethodAllocation:
    method: str
    p: int
    h: int | None = None
    tau: int | None = None
    D: int | None = None


def divisors(n: int) -> Tuple[int, ...]:
    if n < 1:
        raise ValueError("n must be positive")
    return tuple(i for i in range(1, n + 1) if n % i == 0)


def validate_nonnegative(value: float, *, name: str, scale: float = 1.0) -> float:
    tolerance = 1e-12 * max(1.0, abs(scale))
    if value >= 0.0:
        return value
    if value >= -tolerance:
        return 0.0
    raise ValueError(f"materially negative {name}: {value}")


def close_leq(value: float, threshold: float) -> bool:
    return value <= threshold or math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-15
    )


def delta_writer_snapshot_moments(p: int, h: int) -> Dict[str, Any]:
    """Return exact IID moments at every fact and snapshot endpoint.

    Arrays use zero-based storage but implement the one-based recurrences in
    the protocol: ``a[c]`` is E||W_c||_F^2 and ``energy[c-1][j-1]`` is
    E||W_c x_j||^2.
    """

    if p < 1 or h < 1:
        raise ValueError("p and h must be positive")
    lam = 1.0 - 1.0 / p
    beta = 1.0 - 2.0 / p + 2.0 / (p * (p + 2.0))
    a = [0.0]
    for _ in range(h):
        a.append(lam * a[-1] + 1.0)

    energy = [[0.0 for _ in range(h)] for _ in range(h)]
    for j0 in range(h):
        j = j0 + 1
        for c0 in range(h):
            c = c0 + 1
            if c < j:
                value = a[c] / p
            elif c == j:
                value = 1.0
            else:
                previous = energy[c0 - 1][j0]
                value = (
                    beta * previous
                    + a[c - 1] / (p * (p + 2.0))
                    + 1.0 / p
                )
            energy[c0][j0] = value
    return {"lambda": lam, "beta": beta, "a": a, "energy": energy}


def endpoint_delta_error_per_fact(p: int, J: int) -> float:
    moments = delta_writer_snapshot_moments(p, J)
    lam = float(moments["lambda"])
    energy = moments["energy"]
    error = sum(
        energy[J - 1][j - 1] - 2.0 * lam ** (J - j) + 1.0
        for j in range(1, J + 1)
    ) / J
    return validate_nonnegative(error, name="endpoint delta error", scale=2.0)


def endpoint_delta_error_aggregate(p: int, J: int) -> float:
    moments = delta_writer_snapshot_moments(p, J)
    beta = float(moments["beta"])
    a = moments["a"]
    R = 0.0
    for t in range(1, J + 1):
        R = beta * R + (t - 1) * (a[t - 1] / (p * (p + 2.0)) + 1.0 / p) + 1.0
    error = (R - 2.0 * a[J] + J) / J
    return validate_nonnegative(error, name="aggregate endpoint delta error", scale=2.0)


def hierarchy_moments(p: int, h: int, tau: int, J: int) -> Dict[str, float]:
    if J % h or h % tau:
        raise ValueError("registered hierarchy requires tau|h and h|J")
    moments = delta_writer_snapshot_moments(p, h)
    lam = float(moments["lambda"])
    a = moments["a"]
    energy = moments["energy"]
    commits = tuple(range(tau, h + 1, tau))
    local_errors: List[float] = []
    distractor_energies: List[float] = []
    for j in range(1, h + 1):
        route = next(c for c in commits if c >= j)
        local_errors.append(energy[route - 1][j - 1] - 2.0 * lam ** (route - j) + 1.0)
        distractor_energies.append(
            sum(energy[c - 1][j - 1] for c in commits if c != route)
        )
    delta = validate_nonnegative(sum(local_errors) / h, name="hierarchy delta")
    A0 = validate_nonnegative(sum(distractor_energies) / h, name="hierarchy A0")
    B = validate_nonnegative(sum(a[c] / p for c in commits), name="hierarchy B")
    A = validate_nonnegative(A0 + (J // h - 1) * B, name="hierarchy A")
    return {"delta": delta, "A0": A0, "B": B, "A": A}


def local_additive_error(p: int, J: int) -> float:
    if p < 1 or J < 1:
        raise ValueError("p and J must be positive")
    return (J - 1.0) / p


def minimum_feasible_D(delta: float, A: float, threshold: float, maximum_D: int) -> int | None:
    A = validate_nonnegative(A, name="A", scale=max(abs(delta), abs(A), abs(threshold)))
    if maximum_D < 1:
        return None
    if A == 0.0:
        return 1 if close_leq(delta, threshold) else None
    if delta >= threshold:
        return None
    estimate = max(1, math.ceil(A / (threshold - delta)))
    for D in range(max(1, estimate - 2), min(maximum_D, estimate + 2) + 1):
        if close_leq(delta + A / D, threshold):
            if D == 1 or not close_leq(delta + A / (D - 1), threshold):
                return D
    for D in range(1, maximum_D + 1):
        if close_leq(delta + A / D, threshold):
            return D
    return None


def resource_row(allocation: MethodAllocation, *, J: int, Q: int, value_dim: int) -> Dict[str, int]:
    p, v = allocation.p, value_dim
    payload = p * v
    if allocation.method == "local_delta":
        write = J * (3 * payload + v)
        read = Q * payload
        return {
            "local_write_ops": write,
            "global_commit_mac": 0,
            "read_mac": read,
            "operation_proxy": write + read,
            "mutable_state_scalars": payload,
            "active_local_scalars": payload,
            "folded_state_scalars": 0,
            "num_commits": 0,
        }
    if allocation.method == "local_additive":
        write = J * payload
        read = Q * payload
        return {
            "local_write_ops": write,
            "global_commit_mac": 0,
            "read_mac": read,
            "operation_proxy": write + read,
            "mutable_state_scalars": payload,
            "active_local_scalars": payload,
            "folded_state_scalars": 0,
            "num_commits": 0,
        }
    if allocation.method != "hierarchical" or allocation.h is None or allocation.tau is None or allocation.D is None:
        raise ValueError(f"invalid allocation: {allocation}")
    if J % allocation.h or allocation.h % allocation.tau:
        raise ValueError("hierarchical resource row requires tau|h and h|J")
    commits = J // allocation.tau
    write = J * (3 * payload + v)
    build = commits * allocation.D * payload
    read = Q * (allocation.D * payload + payload)
    return {
        "local_write_ops": write,
        "global_commit_mac": build,
        "read_mac": read,
        "operation_proxy": write + build + read,
        "mutable_state_scalars": payload * (1 + allocation.D),
        "active_local_scalars": payload,
        "folded_state_scalars": payload * allocation.D,
        "num_commits": commits,
    }


def allocation_sort_key(row: Mapping[str, Any]) -> Tuple[int, ...]:
    method = str(row["method"])
    return (
        int(row["operation_proxy"]),
        int(row["mutable_state_scalars"]),
        METHOD_ORDER[method],
        int(row["p"]),
        int(row.get("h") or 0),
        int(row.get("tau") or 0),
        int(row.get("D") or 0),
    )


def method_frontiers(
    *, J: int, Q: int, threshold: float, value_dim: int, memory_cap: int
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    maximum_p = memory_cap // value_dim
    for p in range(1, maximum_p + 1):
        for method, error in (
            ("local_delta", endpoint_delta_error_aggregate(p, J)),
            ("local_additive", local_additive_error(p, J)),
        ):
            allocation = MethodAllocation(method, p)
            resources = resource_row(allocation, J=J, Q=Q, value_dim=value_dim)
            rows.append({
                "method": method, "p": p, "h": None, "tau": None, "D": None,
                "population_error": error, "feasible": close_leq(error, threshold),
                **resources,
            })
    for h in divisors(J):
        for tau in divisors(h):
            for p in range(1, maximum_p + 1):
                maximum_D = memory_cap // (p * value_dim) - 1
                if maximum_D < 1:
                    break
                moments = hierarchy_moments(p, h, tau, J)
                D = minimum_feasible_D(moments["delta"], moments["A"], threshold, maximum_D)
                row: Dict[str, Any] = {
                    "method": "hierarchical", "p": p, "h": h, "tau": tau,
                    "D": D, "maximum_D": maximum_D, **moments,
                    "feasible": D is not None,
                }
                if D is not None:
                    allocation = MethodAllocation("hierarchical", p, h, tau, D)
                    row.update(
                        population_error=moments["delta"] + moments["A"] / D,
                        **resource_row(allocation, J=J, Q=Q, value_dim=value_dim),
                    )
                rows.append(row)
    return rows


def select_methods(
    *, J: int, Q: int, threshold: float, value_dim: int, memory_cap: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frontiers = method_frontiers(
        J=J, Q=Q, threshold=threshold, value_dim=value_dim, memory_cap=memory_cap
    )
    selected: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        eligible = [r for r in frontiers if r["method"] == method and r["feasible"]]
        if eligible:
            selected.append(dict(min(eligible, key=allocation_sort_key)))
        else:
            selected.append({"method": method, "selected": False})
    for row in selected:
        row.setdefault("selected", True)
    eligible_selected = [r for r in selected if r["selected"]]
    overall = dict(min(eligible_selected, key=allocation_sort_key)) if eligible_selected else {"selected": False}
    overall["method_scope"] = "overall"
    selected.append(overall)
    return frontiers, selected


def simulate_local_delta(addresses: Tensor, values: Tensor) -> Tensor:
    if addresses.shape[:2] != values.shape[:2]:
        raise ValueError("address/value episode and fact axes must match")
    episodes, J, p = addresses.shape
    v = values.shape[-1]
    W = values.new_zeros(episodes, v, p)
    for j in range(J):
        x, y = addresses[:, j], values[:, j]
        prediction = torch.einsum("evp,ep->ev", W, x)
        W = W + (y - prediction).unsqueeze(-1) * x.unsqueeze(-2)
    predictions = torch.einsum("evp,ejp->ejv", W, addresses)
    return (predictions - values).square().sum(dim=-1).mean(dim=-1)


def simulate_local_additive(addresses: Tensor, values: Tensor) -> Tensor:
    W = torch.einsum("ejv,ejp->evp", values, addresses)
    predictions = torch.einsum("evp,ejp->ejv", W, addresses)
    return (predictions - values).square().sum(dim=-1).mean(dim=-1)


def simulate_hierarchy(
    addresses: Tensor, values: Tensor, *, h: int, tau: int, D: int, global_keys: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    episodes, J, p = addresses.shape
    if J % h:
        raise ValueError("h must divide J")
    groups = J // h
    trace = write_local_groups(
        addresses.reshape(episodes, groups, h, p),
        values.reshape(episodes, groups, h, values.shape[-1]),
        tau=tau,
    )
    statistics = heldout_episode_statistics(trace)
    reference = statistics["delta"] + statistics["A"] / D
    sampled = folded_la_episode_errors(trace, global_keys)
    return reference, sampled, statistics["delta"], statistics["A"]


def coverage(rows: Sequence[Mapping[str, Any]], *, threshold: float) -> Dict[str, Any]:
    eligible = [row for row in rows if bool(row.get("population_oracle_exists", True))]
    denominator = len(eligible)
    numerator = sum(
        bool(row.get("selected")) and bool(row.get("empirical_feasible"))
        for row in eligible
    )
    value = numerator / denominator if denominator else None
    return {
        "eligible_views": denominator,
        "empirically_feasible_views": numerator,
        "coverage": value,
        "passed": value is not None and value >= threshold,
    }
