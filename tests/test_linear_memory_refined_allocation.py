from __future__ import annotations

import json
import math

import pytest
import torch

from experiments.linear_memory.refined_allocation import (
    IntegerAllocation,
    allocation_resources,
    brute_force_calibrated_allocation,
    close_leq,
    direct_reference_stats,
    eligible_global_dims,
    episode_joint_moments,
    iid_h2_endpoint_moments,
    minimum_feasible_integer_D,
    oracle_allocation,
    oracle_coverage,
    reference_stats_from_moments,
    select_calibrated_allocation,
)
import experiments.linear_memory.run_refined_allocation as refined_runner


def test_joint_moment_variance_shortcut_matches_episode_vectors() -> None:
    generator = torch.Generator().manual_seed(83)
    delta = torch.rand(513, generator=generator, dtype=torch.float64)
    A = 0.4 * delta + torch.rand(513, generator=generator, dtype=torch.float64)
    moments = episode_joint_moments(delta, A)
    for D in (1, 7, 73):
        shortcut = reference_stats_from_moments(moments, D)
        direct = direct_reference_stats(delta, A, D)
        for key in ("mean", "variance", "se", "ci_low", "ci_high"):
            assert shortcut[key] == pytest.approx(direct[key], rel=1e-12, abs=1e-14)


def test_reference_variance_clamps_only_tiny_negative_cancellation() -> None:
    tiny = {
        "episodes": 100,
        "delta_mean": 0.0,
        "A_mean": 0.0,
        "delta_variance": 1.0,
        "A_variance": 1.0,
        "delta_A_covariance": -1.0 - 1e-14,
    }
    assert reference_stats_from_moments(tiny, 1)["variance"] == 0.0
    material = {**tiny, "delta_A_covariance": -1.1}
    with pytest.raises(ValueError, match="materially negative reference variance"):
        reference_stats_from_moments(material, 1)


def test_exact_integer_selector_matches_tiny_brute_force() -> None:
    rows = [
        {"p": 2, "h": 1, "tau": 1, "delta_mean": 99.0, "A0_mean": 99.0, "B_mean": 99.0},
        {"p": 4, "h": 2, "tau": 2, "delta_mean": 0.02, "A0_mean": 0.0, "B_mean": 0.15},
        {"p": 8, "h": 4, "tau": 2, "delta_mean": 0.04, "A0_mean": 0.2, "B_mean": 0.08},
    ]
    kwargs = dict(
        policy="unrestricted", J=16, Q=3, epsilon=0.2,
        slack_multiplier=0.95, value_dim=2, memory_cap=256,
    )
    exact = select_calibrated_allocation(rows, **kwargs)
    brute = brute_force_calibrated_allocation(rows, **kwargs)
    assert exact is not None and brute is not None
    for key in ("p", "h", "tau", "D", "operation_proxy"):
        assert exact[key] == brute[key]


def test_independent_policy_optimizes_p_and_integer_D() -> None:
    rows = [
        {"p": p, "h": 1, "tau": 1, "delta_mean": 7.0, "A0_mean": 8.0, "B_mean": 9.0}
        for p in (2, 4, 8, 16)
    ]
    selected = select_calibrated_allocation(
        rows,
        policy="independent",
        J=16,
        Q=1,
        epsilon=0.1,
        slack_multiplier=0.95,
        value_dim=2,
        memory_cap=2048,
    )
    assert selected is not None
    assert (selected["p"], selected["h"], selected["tau"], selected["D"]) == (2, 1, 1, 79)
    assert selected["prediction_source"] == "exact_h1_identity"


def test_integer_D_zero_noise_and_exact_boundary_rules() -> None:
    assert minimum_feasible_integer_D(0.1, 0.0, 0.1, 10) == 1
    assert minimum_feasible_integer_D(0.10000000000000002, 0.0, 0.1, 10) == 1
    assert minimum_feasible_integer_D(0.1, 1e-6, 0.1, 10_000_000) is None
    assert minimum_feasible_integer_D(0.0, 1.0, 0.1, 20) == 10
    assert not close_leq(0.1000001, 0.1)


def test_certified_oracle_enumerates_nonmonotone_CI_endpoint() -> None:
    moments = {
        "p": 2,
        "h": 2,
        "tau": 2,
        "episodes": 1000,
        "delta_mean": 0.0845,
        "A_mean": 0.001,
        "delta_variance": 0.0001,
        "A_variance": 0.04,
        "delta_A_covariance": -0.0015,
    }
    uppers = [reference_stats_from_moments(moments, D)["ci_high"] for D in range(1, 200)]
    feasible_D = [D for D, upper in enumerate(uppers, start=1) if upper <= 0.085]
    assert feasible_D
    assert feasible_D[-1] < 199  # upper endpoint rises above target again.
    oracle = oracle_allocation(
        [moments], policy="shared", J=16, Q=1, threshold=0.085,
        value_dim=2, memory_cap=2048, certified=True,
    )
    assert oracle is not None
    manual = min(
        (
            allocation_resources(IntegerAllocation(2, 2, 2, D), J=16, Q=1, value_dim=2)["operation_proxy"],
            D,
        )
        for D in feasible_D
    )
    assert (oracle["operation_proxy"], oracle["D"]) == manual


def test_dependency_barrier_rejects_imported_model_drift(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    model = tmp_path / "model.py"
    model.write_text("VERSION=1\n")
    monkeypatch.setattr(refined_runner, "MODEL_CODE", model)
    output = tmp_path / "out"
    output.mkdir()
    calibration = output / "calibration_moments.csv"
    frontier = output / "prediction_frontiers.csv"
    calibration.write_text("x\n1\n")
    frontier.write_text("x\n1\n")
    artifact = {
        "input_hashes": refined_runner.input_hashes(config),
        "calibration_sha256": refined_runner.sha256(calibration),
        "prediction_frontiers_sha256": refined_runner.sha256(frontier),
        "selections": [],
    }
    artifact_path = output / "selections_preregistered.json"
    artifact_path.write_text(json.dumps(artifact))
    (output / "selections_preregistered.sha256").write_text(
        refined_runner.sha256(artifact_path) + "\n"
    )
    refined_runner.verify_barrier(config, output)
    model.write_text("VERSION=2\n")
    with pytest.raises(RuntimeError, match="frozen inputs changed"):
        refined_runner.verify_barrier(config, output)


def test_evaluation_refuses_to_overwrite_completed_artifacts(tmp_path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    # Pre-evaluation files are expected and do not trigger the guard.
    for name in (
        "config_snapshot.json",
        "calibration_moments.csv",
        "prediction_frontiers.csv",
        "selections_preregistered.json",
        "selections_preregistered.sha256",
        "pre_evaluation_provenance.json",
    ):
        (output / name).write_text("placeholder\n")
    refined_runner.refuse_evaluation_overwrite(output)
    (output / "policy_evaluation.csv").write_text("completed\n")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        refined_runner.refuse_evaluation_overwrite(output)


def test_coverage_is_abstention_safe_and_nonvacuous() -> None:
    abstain = [
        {"oracle": True, "selected": index == 0, "feasible": index == 0}
        for index in range(10)
    ]
    result = oracle_coverage(
        abstain, oracle_key="oracle", feasible_key="feasible", threshold=0.9
    )
    assert result["coverage"] == pytest.approx(0.1)
    assert result["coverage_passed"] is False
    empty = oracle_coverage(
        [{"oracle": False, "selected": False, "feasible": None}],
        oracle_key="oracle", feasible_key="feasible", threshold=0.9,
    )
    assert empty["coverage"] is None
    assert empty["coverage_passed"] is False


def test_registered_iid_witness_formulae_and_cost_lower_bound() -> None:
    p16 = iid_h2_endpoint_moments(16)
    error16 = p16["delta"] + 7 * p16["B"] / 16
    assert error16 == pytest.approx(0.08943684895833333)
    assert allocation_resources(IntegerAllocation(16, 2, 2, 16), J=16, Q=1, value_dim=2)["operation_proxy"] == 6208

    p32 = iid_h2_endpoint_moments(32)
    error32 = p32["delta"] + 15 * p32["B"] / 12
    assert error32 == pytest.approx(0.09390797334558823)
    shared_cost = allocation_resources(IntegerAllocation(32, 2, 2, 12), J=32, Q=1, value_dim=2)["operation_proxy"]
    assert shared_cost == 19328
    independent_relaxed_lower_bound = 20288 + 624
    assert shared_cost / independent_relaxed_lower_bound - 1.0 < -0.075
