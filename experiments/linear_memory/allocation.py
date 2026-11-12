"""Allocation theory utilities for EXP-LM-002.

This module intentionally uses a fixed delta-rule writer with fresh local
addresses.  It shares the commit-schedule definition with EXP-LM-001 but does
not use its learned position encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import CommitSchedule, build_commit_schedule


@dataclass(frozen=True, order=True)
class AllocationCandidate:
    p: int
    h: int
    tau: int
    D: int


@dataclass
class LocalGroupTrace:
    """Batched independent reset groups.

    Shapes are addresses ``[episode,group,h,p]``, values
    ``[episode,group,h,v]``, and snapshots
    ``[episode,group,commit,v,p]``.
    """

    addresses: Tensor
    values: Tensor
    snapshots: Tensor
    schedule: CommitSchedule


def stable_seed(*parts: object) -> int:
    """Derive a deterministic PyTorch seed without Python hash randomization."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def unit_random(shape: Sequence[int], generator: torch.Generator, *, dtype: torch.dtype) -> Tensor:
    return F.normalize(torch.randn(*shape, generator=generator, dtype=dtype), dim=-1)


def make_local_addresses(
    *,
    episodes: int,
    groups: int,
    h: int,
    p: int,
    geometry: str,
    mixing_coefficient: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Generate fresh marginally isotropic addresses for every group."""

    if min(episodes, groups, h, p) < 1:
        raise ValueError("address dimensions must be positive")
    if geometry == "iid":
        return unit_random((episodes, groups, h, p), generator, dtype=dtype)
    if geometry == "correlated":
        if not 0.0 <= mixing_coefficient <= 1.0:
            raise ValueError("mixing_coefficient must lie in [0,1]")
        shared = torch.randn(episodes, groups, 1, p, generator=generator, dtype=dtype)
        independent = torch.randn(
            episodes, groups, h, p, generator=generator, dtype=dtype
        )
        raw = (
            math.sqrt(mixing_coefficient) * shared
            + math.sqrt(1.0 - mixing_coefficient) * independent
        )
        return F.normalize(raw, dim=-1)
    raise ValueError(f"unknown geometry: {geometry}")


def make_values(
    *,
    episodes: int,
    groups: int,
    h: int,
    value_dim: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    return unit_random((episodes, groups, h, value_dim), generator, dtype=dtype)


def write_local_groups(
    addresses: Tensor,
    values: Tensor,
    *,
    tau: int,
    eta: float = 1.0,
    inner_steps: int = 1,
) -> LocalGroupTrace:
    """Apply the batched delta rule, resetting at every explicit group."""

    if addresses.ndim != 4 or values.ndim != 4:
        raise ValueError("addresses and values must have rank four")
    if addresses.shape[:3] != values.shape[:3]:
        raise ValueError("address/value episode, group, and fact axes must match")
    if inner_steps < 1:
        raise ValueError("inner_steps must be positive")
    episodes, groups, h, p = addresses.shape
    value_dim = values.shape[-1]
    schedule = build_commit_schedule(h, h=h, tau=tau)
    commit_lookup = {
        int(fact): index for index, fact in enumerate(schedule.commit_indices.tolist())
    }
    W = values.new_zeros(episodes, groups, value_dim, p)
    snapshots: List[Tensor] = []
    for fact in range(h):
        x = addresses[:, :, fact]
        y = values[:, :, fact]
        for _ in range(inner_steps):
            prediction = torch.einsum("egvp,egp->egv", W, x)
            W = W + eta * (y - prediction).unsqueeze(-1) * x.unsqueeze(-2)
        if fact in commit_lookup:
            snapshots.append(W.clone())
    return LocalGroupTrace(
        addresses=addresses,
        values=values,
        snapshots=torch.stack(snapshots, dim=2),
        schedule=schedule,
    )


def snapshot_responses(trace: LocalGroupTrace) -> Tensor:
    """Apply every same-group snapshot to every fact address.

    Returns ``[episode,group,fact,commit,value]``.
    """

    return torch.einsum("egcvp,egjp->egjcv", trace.snapshots, trace.addresses)


def exact_local_predictions(trace: LocalGroupTrace) -> Tensor:
    responses = snapshot_responses(trace)
    routes = trace.schedule.fact_to_commit.to(responses.device)
    index = routes.view(1, 1, -1, 1, 1).expand(
        responses.shape[0], responses.shape[1], -1, 1, responses.shape[-1]
    )
    return responses.gather(dim=3, index=index).squeeze(3)


def calibration_episode_statistics(trace: LocalGroupTrace) -> Dict[str, Tensor]:
    """Return one `delta`, `A0`, and `B` observation per episode."""

    responses = snapshot_responses(trace)
    exact = exact_local_predictions(trace)
    delta = (exact - trace.values).square().sum(dim=-1).mean(dim=(1, 2))
    response_energy = responses.square().sum(dim=-1)
    total_energy = response_energy.sum(dim=-1)
    routes = trace.schedule.fact_to_commit.to(responses.device)
    route_index = routes.view(1, 1, -1, 1).expand(
        responses.shape[0], responses.shape[1], -1, 1
    )
    target_energy = response_energy.gather(3, route_index).squeeze(3)
    A0 = (total_energy - target_energy).mean(dim=(1, 2))
    p = trace.addresses.shape[-1]
    B = trace.snapshots.square().sum(dim=(-1, -2, -3)).mean(dim=1) / float(p)
    return {"delta": delta, "A0": A0, "B": B}


def heldout_episode_statistics(trace: LocalGroupTrace) -> Dict[str, Tensor]:
    """Compute local error and all-snapshot distractor energy per episode."""

    exact = exact_local_predictions(trace)
    delta = (exact - trace.values).square().sum(dim=-1).mean(dim=(1, 2))
    # query group/fact, source group/commit, value
    all_responses = torch.einsum(
        "eacvp,egjp->egjacv", trace.snapshots, trace.addresses
    )
    all_energy = all_responses.square().sum(dim=-1).sum(dim=(-1, -2))
    routes = trace.schedule.fact_to_commit.to(trace.addresses.device)
    episodes, groups, h = trace.addresses.shape[:3]
    e_index = torch.arange(episodes, device=trace.addresses.device)[:, None, None]
    g_index = torch.arange(groups, device=trace.addresses.device)[None, :, None]
    j_index = torch.arange(h, device=trace.addresses.device)[None, None, :]
    target_response = all_responses[
        e_index,
        g_index,
        j_index,
        g_index,
        routes.view(1, 1, h),
    ]
    target_energy = target_response.square().sum(dim=-1)
    A = (all_energy - target_energy).mean(dim=(1, 2))
    return {"delta": delta, "A": A}


def folded_la_episode_errors(trace: LocalGroupTrace, global_keys: Tensor) -> Tensor:
    """Return direct folded-LA MSE averaged within each episode."""

    if global_keys.shape[:3] != trace.snapshots.shape[:3]:
        raise ValueError("global keys must align with snapshot episode/group/commit axes")
    state = torch.einsum("egcd,egcvp->edvp", global_keys, trace.snapshots)
    routes = trace.schedule.fact_to_commit.to(global_keys.device)
    index = routes.view(1, 1, -1, 1).expand(
        global_keys.shape[0], global_keys.shape[1], -1, global_keys.shape[-1]
    )
    queries = global_keys.gather(dim=2, index=index)
    retrieved = torch.einsum("egjd,edvp->egjvp", queries, state)
    predictions = torch.einsum("egjvp,egjp->egjv", retrieved, trace.addresses)
    return (predictions - trace.values).square().sum(dim=-1).mean(dim=(1, 2))


def mean_se_ci(samples: Tensor, z: float = 1.96) -> Dict[str, float]:
    samples = samples.detach().to(dtype=torch.float64).flatten()
    mean = float(samples.mean())
    se = float(samples.std(unbiased=True) / math.sqrt(samples.numel())) if samples.numel() > 1 else math.nan
    return {"mean": mean, "se": se, "ci_low": mean - z * se, "ci_high": mean + z * se}


def prediction(delta: float, A0: float, B: float, groups: int, D: int) -> float:
    return float(delta + (A0 + (groups - 1) * B) / D)


def minimum_global_dim(delta: float, A: float, epsilon: float) -> int | None:
    """Smallest positive integer D for expected error `delta + A/D <= epsilon`."""

    if A < 0.0:
        raise ValueError("A must be nonnegative")
    if A == 0.0:
        return 1 if delta <= epsilon else None
    if delta >= epsilon:
        return None
    return max(1, math.ceil(A / (epsilon - delta)))


def valid_taus(h: int) -> Tuple[int, ...]:
    return tuple(sorted({tau for tau in (1, 2, h) if tau <= h}))


def candidates_from_config(config: Mapping[str, Any]) -> List[AllocationCandidate]:
    grid = config["grid"]
    return [
        AllocationCandidate(p=p, h=h, tau=tau, D=D)
        for p in grid["local_dims"]
        for h in grid["reset_horizons"]
        for tau in valid_taus(h)
        for D in grid["global_dims"]
    ]


def candidate_resources(
    candidate: AllocationCandidate,
    *,
    J: int,
    Q: int,
    value_dim: int,
) -> Dict[str, int]:
    if J % candidate.h:
        raise ValueError("J must be divisible by h for the registered cost")
    commits_per_group = len(build_commit_schedule(candidate.h, candidate.h, candidate.tau).commit_indices)
    groups = J // candidate.h
    commits = groups * commits_per_group
    payload = candidate.p * value_dim
    local_write = J * (3 * payload + value_dim)
    state_build = commits * candidate.D * payload
    folded_read = Q * (candidate.D * payload + payload)
    return {
        "groups": groups,
        "commits_per_group": commits_per_group,
        "num_commits": commits,
        "payload_scalars": payload,
        "active_local_scalars": payload,
        "folded_state_scalars": candidate.D * payload,
        "folded_inference_memory_scalars": payload + candidate.D * payload,
        "raw_snapshot_bank_scalars": commits * payload,
        "raw_global_key_bank_scalars": commits * candidate.D,
        "local_write_ops": local_write,
        "state_build_mac": state_build,
        "folded_read_mac": folded_read,
        "operation_proxy": local_write + state_build + folded_read,
    }


def policy_allows(candidate: AllocationCandidate, policy: str) -> bool:
    if policy == "theory":
        return True
    if policy == "independent_p4":
        return candidate.p == 4 and candidate.h == 1 and candidate.tau == 1
    if policy == "endpoint_p4":
        return candidate.p == 4 and candidate.tau == candidate.h
    raise ValueError(f"unknown policy: {policy}")


def choose_candidate(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: str,
    epsilon: float,
    memory_cap: int,
) -> Dict[str, Any] | None:
    """Choose from precomputed prediction/resource rows, with frozen tie-breaks."""

    feasible = []
    for row in rows:
        candidate = AllocationCandidate(
            int(row["p"]), int(row["h"]), int(row["tau"]), int(row["D"])
        )
        if not policy_allows(candidate, policy):
            continue
        if float(row["predicted_error"]) > epsilon:
            continue
        if int(row["folded_inference_memory_scalars"]) > memory_cap:
            continue
        feasible.append(dict(row))
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            int(row["operation_proxy"]),
            int(row["folded_inference_memory_scalars"]),
            int(row["p"]),
            int(row["h"]),
            int(row["tau"]),
            int(row["D"]),
        ),
    )


def point_feasibility(mean: float, ci_low: float, ci_high: float, epsilon: float) -> Tuple[bool, str]:
    point = mean <= epsilon
    if ci_high <= epsilon:
        interval = "feasible"
    elif ci_low > epsilon:
        interval = "infeasible"
    else:
        interval = "uncertain"
    return point, interval


def feasible_regret(selected_cost: int, selected_feasible: bool, oracle_cost: int | None) -> float | None:
    if not selected_feasible or oracle_cost is None:
        return None
    return selected_cost / oracle_cost - 1.0
