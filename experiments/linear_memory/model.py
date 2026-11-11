"""Differentiable two-level linear associative memory.

The local state is an operator ``W in R^(value_dim x local_dim)`` updated by
the delta rule.  Whole-operator snapshots are committed at segment endpoints
and may be read exactly or folded into a random-key linear-attention state.

Global routing is deliberately oracle-controlled: each fact queries the first
commit in its own reset group after that fact was written.  This isolates the
storage and gradient-path questions from learned global routing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CommitSchedule:
    """Commit endpoints and oracle routes, using zero-based segment indices."""

    commit_indices: Tensor
    group_final_mask: Tensor
    fact_to_commit: Tensor
    routing_age: Tensor


@dataclass
class WriterTrace:
    """Immutable snapshots and address metadata returned by the writer."""

    local_addresses: Tensor
    live_snapshots: Tensor
    storage_snapshots: Tensor
    schedule: CommitSchedule


def build_commit_schedule(num_segments: int, h: int, tau: int) -> CommitSchedule:
    """Build reset groups and commits, including non-divisible group tails.

    A group is reset every ``h`` elementary segments.  Inside each group a
    commit occurs every ``tau`` segments and always at the group endpoint.
    Every fact routes to the first commit in its group at or after its write.
    """

    if num_segments < 1:
        raise ValueError("num_segments must be positive")
    if not (1 <= tau <= h):
        raise ValueError("expected 1 <= tau <= h")

    commits = []
    group_finals = []
    routes = [-1] * num_segments
    for group_start in range(0, num_segments, h):
        group_end = min(group_start + h, num_segments)  # exclusive
        endpoints = list(range(group_start + tau - 1, group_end, tau))
        if not endpoints or endpoints[-1] != group_end - 1:
            endpoints.append(group_end - 1)
        first_commit = len(commits)
        commits.extend(endpoints)
        group_finals.extend([False] * (len(endpoints) - 1) + [True])
        for fact in range(group_start, group_end):
            local_route = next(i for i, endpoint in enumerate(endpoints) if endpoint >= fact)
            routes[fact] = first_commit + local_route

    commit_tensor = torch.tensor(commits, dtype=torch.long)
    route_tensor = torch.tensor(routes, dtype=torch.long)
    ages = commit_tensor[route_tensor] - torch.arange(num_segments, dtype=torch.long)
    return CommitSchedule(
        commit_indices=commit_tensor,
        group_final_mask=torch.tensor(group_finals, dtype=torch.bool),
        fact_to_commit=route_tensor,
        routing_age=ages,
    )


class LinearAssociativeWriter(nn.Module):
    """Shared trainable local address encoder and bounded delta-rule step."""

    def __init__(
        self,
        num_segments: int,
        local_dim: int,
        value_dim: int,
        h: int,
        tau: int,
        inner_steps: int = 1,
        eta_init: float = 0.5,
        eta_max: float = 1.25,
    ) -> None:
        super().__init__()
        if local_dim < 1 or value_dim < 1 or inner_steps < 1:
            raise ValueError("dimensions and inner_steps must be positive")
        if not (0.0 < eta_init < eta_max):
            raise ValueError("eta_init must lie strictly between zero and eta_max")
        self.num_segments = num_segments
        self.local_dim = local_dim
        self.value_dim = value_dim
        self.h = h
        self.tau = tau
        self.inner_steps = inner_steps
        self.eta_max = float(eta_max)
        self.encoder = nn.Linear(num_segments, local_dim, bias=False)
        logit = math.log(eta_init / (eta_max - eta_init))
        self.eta_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        self.schedule = build_commit_schedule(num_segments, h, tau)

    @property
    def eta(self) -> Tensor:
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def local_addresses(self) -> Tensor:
        raw = torch.eye(
            self.num_segments,
            device=self.encoder.weight.device,
            dtype=self.encoder.weight.dtype,
        )
        return F.normalize(self.encoder(raw), dim=-1, eps=1e-12)

    def forward(self, values: Tensor, detach_intermediate: bool = False) -> WriterTrace:
        """Write a batch of episodes.

        Args:
            values: ``[batch, num_segments, value_dim]`` targets.
            detach_intermediate: stop gradients only through non-final snapshot
                storage branches.  The live recurrent state remains attached,
                and every reset-group final snapshot remains differentiable.
        """

        if values.ndim != 3 or values.shape[1:] != (
            self.num_segments,
            self.value_dim,
        ):
            raise ValueError(
                f"expected values [batch,{self.num_segments},{self.value_dim}], "
                f"got {tuple(values.shape)}"
            )
        addresses = self.local_addresses()
        schedule = self.schedule
        commit_lookup = {
            int(segment): commit_pos
            for commit_pos, segment in enumerate(schedule.commit_indices.tolist())
        }
        batch = values.shape[0]
        W = values.new_zeros(batch, self.value_dim, self.local_dim)
        live_snapshots = []
        storage_snapshots = []
        for segment in range(self.num_segments):
            if segment % self.h == 0:
                W = values.new_zeros(batch, self.value_dim, self.local_dim)
            x = addresses[segment]
            for _ in range(self.inner_steps):
                prediction = torch.einsum("bvp,p->bv", W, x)
                residual = values[:, segment] - prediction
                W = W + self.eta * residual.unsqueeze(-1) * x.view(1, 1, -1)
            if segment in commit_lookup:
                commit_pos = commit_lookup[segment]
                snapshot = W.clone()
                live_snapshots.append(snapshot)
                is_final = bool(schedule.group_final_mask[commit_pos])
                storage = snapshot if (is_final or not detach_intermediate) else snapshot.detach()
                storage_snapshots.append(storage)

        return WriterTrace(
            local_addresses=addresses,
            live_snapshots=torch.stack(live_snapshots, dim=1),
            storage_snapshots=torch.stack(storage_snapshots, dim=1),
            schedule=schedule,
        )


def exact_snapshot_read(trace: WriterTrace) -> Tensor:
    """Oracle-gather the first available snapshot after every fact write."""

    chosen = trace.storage_snapshots[:, trace.schedule.fact_to_commit]
    return torch.einsum("bjvp,jp->bjv", chosen, trace.local_addresses)


def snapshot_payloads(trace: WriterTrace) -> Tensor:
    """Return every stored operator applied to every fact address.

    Shape is ``[batch, fact, commit, value_dim]``.
    """

    return torch.einsum(
        "bcvp,jp->bjcv", trace.storage_snapshots, trace.local_addresses
    )


def select_commit_keys(global_key_table: Tensor, schedule: CommitSchedule) -> Tensor:
    """Select per-endpoint keys from ``[batch, segment, global_dim]`` table."""

    return global_key_table[:, schedule.commit_indices]


def linear_attention_read(trace: WriterTrace, global_key_table: Tensor) -> Tuple[Tensor, Tensor]:
    """Read the raw snapshot bank with random-unit-key linear attention.

    Queries are the exact key of the oracle target snapshot.  This controls
    routing while retaining superposition from all other committed snapshots.
    """

    stored_keys = select_commit_keys(global_key_table, trace.schedule)
    query_keys = stored_keys[:, trace.schedule.fact_to_commit]
    weights = torch.einsum("bjd,bcd->bjc", query_keys, stored_keys)
    payloads = snapshot_payloads(trace)
    predictions = torch.einsum("bjc,bjcv->bjv", weights, payloads)
    return predictions, weights


def folded_linear_attention_read(
    trace: WriterTrace, global_key_table: Tensor
) -> Tensor:
    """Construct ``S=sum k tensor vec(W)`` and read it explicitly."""

    stored_keys = select_commit_keys(global_key_table, trace.schedule)
    flat_snapshots = trace.storage_snapshots.flatten(start_dim=2)
    state = torch.einsum("bcd,bcz->bdz", stored_keys, flat_snapshots)
    query_keys = stored_keys[:, trace.schedule.fact_to_commit]
    retrieved_flat = torch.einsum("bjd,bdz->bjz", query_keys, state)
    retrieved = retrieved_flat.reshape(
        retrieved_flat.shape[0],
        retrieved_flat.shape[1],
        trace.storage_snapshots.shape[2],
        trace.storage_snapshots.shape[3],
    )
    return torch.einsum("bjvp,jp->bjv", retrieved, trace.local_addresses)


def error_decomposition(predictions: Tensor, exact_predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
    """Pointwise squared-error decomposition for LA retrieval.

    ``total = local + contamination + cross`` exactly, up to floating error.
    Returned quantities are sums over value coordinates for every batch/fact.
    """

    local_error = exact_predictions - targets
    contamination = predictions - exact_predictions
    return {
        "total": (local_error + contamination).square().sum(dim=-1),
        "local": local_error.square().sum(dim=-1),
        "contamination": contamination.square().sum(dim=-1),
        "cross": 2.0 * (local_error * contamination).sum(dim=-1),
    }


def conditional_contamination_prediction(trace: WriterTrace, global_dim: int) -> Tensor:
    """Conditional expected LA contamination energy for every fact.

    For iid isotropic unit global keys and an exact-key query, the prediction
    is ``(1/D) sum_{c != route(j)} ||W_c x_j||^2``.  Snapshot payloads may be
    arbitrarily correlated; independence between payloads is not required.
    """

    payload_energy = snapshot_payloads(trace).square().sum(dim=-1)
    total = payload_energy.sum(dim=-1)
    route = trace.schedule.fact_to_commit.view(1, -1, 1).expand(
        payload_energy.shape[0], -1, 1
    )
    self_energy = payload_energy.gather(dim=-1, index=route).squeeze(-1)
    return (total - self_energy) / float(global_dim)


def two_fact_expected_errors(rho: float, global_dim: int) -> Dict[str, float]:
    """Closed-form two-fact scalar errors for eta=1 and unit local keys."""

    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be a valid unit-vector inner product")
    if global_dim < 1:
        raise ValueError("global_dim must be positive")
    rho2 = rho * rho
    rho4 = rho2 * rho2
    endpoint = 0.5 * (rho2 + rho4)
    snapshot_energy = 0.5 * (1.0 + rho4)
    return {
        "endpoint_nmse": endpoint,
        "snapshot_local_nmse": 0.0,
        "snapshot_payload_energy": snapshot_energy,
        "snapshot_la_nmse": snapshot_energy / global_dim,
        "crossover_D_strict": math.inf if endpoint == 0.0 else snapshot_energy / endpoint,
    }


def operation_estimates(
    num_segments: int,
    num_commits: int,
    local_dim: int,
    value_dim: int,
    global_dim: int,
    inner_steps: int,
    num_queries: int,
) -> Dict[str, int]:
    """Transparent scalar-operation/state estimates, not measured FLOPs.

    A multiply-add is counted as one MAC.  Separate scalar multiply/add terms
    are grouped into the same rough equivalent to keep the ledger readable.
    """

    payload = local_dim * value_dim
    local_write = num_segments * inner_steps * (3 * payload + value_dim)
    exact_read = num_queries * payload
    raw_la_read = num_queries * num_commits * (global_dim + payload + value_dim)
    folded_build = num_commits * global_dim * payload
    folded_read = num_queries * (global_dim * payload + payload)
    return {
        "ideal_folded_state_scalars": global_dim * payload,
        "raw_snapshot_bank_scalars": num_commits * payload,
        "raw_global_key_bank_scalars": num_commits * global_dim,
        "local_write_op_estimate": local_write,
        "exact_read_mac_estimate": exact_read,
        "raw_la_read_op_estimate": raw_la_read,
        "folded_state_build_mac_estimate": folded_build,
        "folded_read_mac_estimate": folded_read,
    }
