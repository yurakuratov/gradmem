from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
import json

from experiments.linear_memory.allocation import (
    AllocationCandidate,
    calibration_episode_statistics,
    candidate_resources,
    choose_candidate,
    feasible_regret,
    heldout_episode_statistics,
    make_local_addresses,
    minimum_global_dim,
    prediction,
    write_local_groups,
)
from experiments.linear_memory.model import LinearAssociativeWriter
import experiments.linear_memory.run_allocation as allocation_runner


def test_batched_fresh_address_writer_matches_original_for_shared_addresses() -> None:
    torch.manual_seed(41)
    episodes, h, p, v, tau = 5, 5, 3, 2, 2
    addresses = F.normalize(torch.randn(h, p, dtype=torch.float64), dim=-1)
    values = F.normalize(torch.randn(episodes, h, v, dtype=torch.float64), dim=-1)
    dynamic = write_local_groups(
        addresses.view(1, 1, h, p).expand(episodes, 1, -1, -1),
        values[:, None],
        tau=tau,
        eta=1.0,
        inner_steps=1,
    )

    original = LinearAssociativeWriter(
        h, p, v, h=h, tau=tau, eta_init=1.0, eta_max=1.25
    ).double()
    with torch.no_grad():
        original.encoder.weight.copy_(addresses.T)
    old_trace = original(values)
    torch.testing.assert_close(dynamic.snapshots[:, 0], old_trace.live_snapshots)


def test_partial_commit_schedule_is_preserved_by_batched_writer() -> None:
    addresses = torch.eye(5, dtype=torch.float64).view(1, 1, 5, 5)
    values = torch.ones(1, 1, 5, 1, dtype=torch.float64)
    trace = write_local_groups(addresses, values, tau=2)
    assert trace.schedule.commit_indices.tolist() == [1, 3, 4]
    assert trace.schedule.fact_to_commit.tolist() == [0, 0, 1, 1, 2]
    assert trace.snapshots.shape == (1, 1, 3, 1, 5)


def test_analytic_B_matches_independent_cross_group_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(77)
    episodes, h, p, v = 9000, 4, 3, 2
    source_x = make_local_addresses(
        episodes=episodes,
        groups=1,
        h=h,
        p=p,
        geometry="correlated",
        mixing_coefficient=0.7,
        generator=generator,
    )
    values = F.normalize(
        torch.randn(episodes, 1, h, v, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    trace = write_local_groups(source_x, values, tau=2)
    B = calibration_episode_statistics(trace)["B"].mean()
    independent_x = F.normalize(
        torch.randn(episodes, p, generator=generator, dtype=torch.float64), dim=-1
    )
    responses = torch.einsum("ecvp,ep->ecv", trace.snapshots[:, 0], independent_x)
    observed = responses.square().sum(dim=(-1, -2)).mean()
    samples = responses.square().sum(dim=(-1, -2))
    se = samples.std(unbiased=True) / math.sqrt(episodes)
    assert abs(observed - B) <= 5.0 * se + 0.01


def test_cross_group_energy_scales_exactly_in_orthogonal_case() -> None:
    # Every group stores an identity operator. The routed local read is exact,
    # A0=0, B=1, and every other group contributes exactly one unit of energy.
    addresses = torch.eye(2, dtype=torch.float64).view(1, 1, 2, 2).expand(1, 3, -1, -1)
    values = torch.eye(2, dtype=torch.float64).view(1, 1, 2, 2).expand(1, 3, -1, -1)
    trace = write_local_groups(addresses, values, tau=2)
    calibration = calibration_episode_statistics(
        write_local_groups(addresses[:, :1], values[:, :1], tau=2)
    )
    heldout = heldout_episode_statistics(trace)
    assert calibration["delta"].item() == pytest.approx(0.0)
    assert calibration["A0"].item() == pytest.approx(0.0)
    assert calibration["B"].item() == pytest.approx(1.0)
    assert heldout["A"].item() == pytest.approx(2.0)
    assert prediction(0.0, 0.0, 1.0, groups=3, D=5) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("delta", "A", "epsilon", "expected"),
    [
        (0.1, 0.0, 0.1, 1),
        (0.11, 0.0, 0.1, None),
        (0.1, 1.0, 0.1, None),
        (0.09, 1.0, 0.1, 100),
        (0.0, 0.01, 0.2, 1),
    ],
)
def test_minimum_global_dim_edges(delta: float, A: float, epsilon: float, expected: int | None) -> None:
    assert minimum_global_dim(delta, A, epsilon) == expected


def test_finite_grid_selection_and_infeasible_regret() -> None:
    rows = [
        {
            "p": 2,
            "h": 2,
            "tau": 2,
            "D": 8,
            "predicted_error": 0.2,
            "operation_proxy": 100,
            "folded_inference_memory_scalars": 36,
        },
        {
            "p": 4,
            "h": 1,
            "tau": 1,
            "D": 4,
            "predicted_error": 0.1,
            "operation_proxy": 120,
            "folded_inference_memory_scalars": 40,
        },
        {
            "p": 4,
            "h": 4,
            "tau": 4,
            "D": 4,
            "predicted_error": 0.15,
            "operation_proxy": 80,
            "folded_inference_memory_scalars": 40,
        },
    ]
    chosen = choose_candidate(rows, policy="theory", epsilon=0.15, memory_cap=100)
    assert (chosen["p"], chosen["h"], chosen["D"]) == (4, 4, 4)
    independent = choose_candidate(
        rows, policy="independent_p4", epsilon=0.15, memory_cap=100
    )
    assert independent is not None and independent["h"] == 1
    assert choose_candidate(rows, policy="theory", epsilon=0.05, memory_cap=100) is None
    assert feasible_regret(120, True, 100) == pytest.approx(0.2)
    assert feasible_regret(80, False, 100) is None
    assert feasible_regret(80, True, None) is None


def test_selection_is_invariant_to_test_only_seed_metadata() -> None:
    candidate = AllocationCandidate(4, 1, 1, 8)
    resources = candidate_resources(candidate, J=16, Q=1, value_dim=2)
    base = {
        "p": candidate.p,
        "h": candidate.h,
        "tau": candidate.tau,
        "D": candidate.D,
        "predicted_error": 0.08,
        **resources,
    }
    rows_a = [{**base, "test_only_seed": 1}]
    rows_b = [{**base, "test_only_seed": 999999}]
    selected_a = choose_candidate(rows_a, policy="theory", epsilon=0.1, memory_cap=2048)
    selected_b = choose_candidate(rows_b, policy="theory", epsilon=0.1, memory_cap=2048)
    for key in ("p", "h", "tau", "D", "operation_proxy"):
        assert selected_a[key] == selected_b[key]


def test_evaluation_barrier_rejects_model_dependency_drift(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    model_dependency = tmp_path / "model.py"
    model_dependency.write_text("VERSION = 1\n")
    monkeypatch.setattr(allocation_runner, "MODEL_CODE", model_dependency)

    output = tmp_path / "output"
    output.mkdir()
    calibration = output / "calibration.csv"
    predictions = output / "candidate_predictions.csv"
    calibration.write_text("value\n1\n")
    predictions.write_text("value\n1\n")
    artifact = {
        "input_hashes": allocation_runner.input_hashes(config_path),
        "calibration_sha256": allocation_runner.sha256(calibration),
        "candidate_predictions_sha256": allocation_runner.sha256(predictions),
        "selections": [],
    }
    artifact_path = output / "selections_preregistered.json"
    artifact_path.write_text(json.dumps(artifact))
    (output / "selections_preregistered.sha256").write_text(
        allocation_runner.sha256(artifact_path) + "\n"
    )
    allocation_runner.verify_barrier(config_path, output)

    model_dependency.write_text("VERSION = 2\n")
    with pytest.raises(RuntimeError, match="frozen inputs changed"):
        allocation_runner.verify_barrier(config_path, output)


def test_audit_coverage_does_not_pass_all_abstention() -> None:
    rows = [
        {
            "oracle_exists": True,
            "selected": False,
            "point_reference_feasible": None,
        }
        for _ in range(10)
    ]
    result = allocation_runner.oracle_scenario_coverage(rows, threshold=0.9)
    assert result["oracle_existing_scenarios"] == 10
    assert result["oracle_existing_with_feasible_selection"] == 0
    assert result["oracle_scenario_coverage"] == 0.0
    assert result["coverage_passed"] is False


def test_audit_coverage_rejects_one_easy_selection_with_many_abstentions() -> None:
    rows = [
        {
            "oracle_exists": True,
            "selected": index == 0,
            "point_reference_feasible": index == 0,
        }
        for index in range(10)
    ]
    # Feasibility among selected is 1/1, but oracle-scenario coverage is 1/10.
    result = allocation_runner.oracle_scenario_coverage(rows, threshold=0.9)
    assert result["oracle_existing_with_feasible_selection"] == 1
    assert result["oracle_scenario_coverage"] == pytest.approx(0.1)
    assert result["coverage_passed"] is False


def test_audit_coverage_never_passes_vacuously_without_an_oracle() -> None:
    result = allocation_runner.oracle_scenario_coverage(
        [{"oracle_exists": False, "selected": False}], threshold=0.9
    )
    assert result["oracle_scenario_coverage"] is None
    assert result["coverage_passed"] is False
