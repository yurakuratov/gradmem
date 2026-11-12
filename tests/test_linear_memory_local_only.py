from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from experiments.linear_memory.local_only_comparison import (
    MethodAllocation,
    coverage,
    delta_writer_snapshot_moments,
    divisors,
    endpoint_delta_error_aggregate,
    endpoint_delta_error_per_fact,
    hierarchy_moments,
    local_additive_error,
    method_frontiers,
    resource_row,
    select_methods,
    simulate_local_additive,
    simulate_local_delta,
)
import experiments.linear_memory.run_local_only_comparison as runner


@pytest.mark.parametrize("J", [1, 2, 4, 16, 32])
@pytest.mark.parametrize("p", [1, 2, 3, 16, 64])
def test_aggregate_and_per_fact_endpoint_recurrences_agree(p: int, J: int) -> None:
    assert endpoint_delta_error_aggregate(p, J) == pytest.approx(
        endpoint_delta_error_per_fact(p, J), rel=2e-11, abs=2e-13
    )


def test_known_h1_h2_and_p1_formulae() -> None:
    J, p = 16, 7
    h1 = hierarchy_moments(p, 1, 1, J)
    assert h1 == pytest.approx({"delta": 0.0, "A0": 0.0, "B": 1 / p, "A": (J - 1) / p})

    p = 11
    h2 = hierarchy_moments(p, 2, 2, J)
    expected_delta = 0.5 * (1 / p + 3 / (p * (p + 2)))
    expected_B = (2 - 1 / p) / p
    assert h2["delta"] == pytest.approx(expected_delta)
    assert h2["A0"] == pytest.approx(0.0, abs=1e-15)
    assert h2["B"] == pytest.approx(expected_B)

    assert endpoint_delta_error_aggregate(1, J) == pytest.approx(2 * (J - 1) / J)
    p1_h2 = hierarchy_moments(1, 2, 2, J)
    assert p1_h2["delta"] == pytest.approx(1.0)
    assert p1_h2["B"] == pytest.approx(1.0)


def test_direct_random_writer_matches_population_theory() -> None:
    # This executes the update equations on random vectors; it does not use
    # the analytic moment recurrence to construct the samples.
    episodes, J, p, v = 30000, 4, 3, 2
    generator = torch.Generator().manual_seed(3481)
    x = F.normalize(torch.randn(episodes, J, p, generator=generator, dtype=torch.float64), dim=-1)
    y = F.normalize(torch.randn(episodes, J, v, generator=generator, dtype=torch.float64), dim=-1)
    samples = simulate_local_delta(x, y)
    theory = endpoint_delta_error_aggregate(p, J)
    se = float(samples.std(unbiased=True) / math.sqrt(episodes))
    assert abs(float(samples.mean()) - theory) <= 5 * se


def test_additive_formula_and_orthogonal_direct_write() -> None:
    J = 4
    x = torch.eye(J, dtype=torch.float64).unsqueeze(0)
    y = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]], dtype=torch.float64)
    assert simulate_local_additive(x, y).item() == pytest.approx(0.0)
    assert local_additive_error(4, J) == pytest.approx(0.75)


def test_exact_integer_optimizer_matches_small_full_brute_force() -> None:
    J, Q, v, cap, threshold = 4, 3, 2, 32, 0.6
    frontiers, selected = select_methods(J=J, Q=Q, threshold=threshold, value_dim=v, memory_cap=cap)
    by_method = {row["method"]: row for row in selected if row.get("method_scope") != "overall"}

    brute_hierarchy = []
    for h in divisors(J):
        for tau in divisors(h):
            for p in range(1, cap // v + 1):
                moments = hierarchy_moments(p, h, tau, J)
                for D in range(1, cap // (p * v)):
                    error = moments["delta"] + moments["A"] / D
                    if error <= threshold + 1e-12:
                        allocation = MethodAllocation("hierarchical", p, h, tau, D)
                        brute_hierarchy.append({
                            "method": "hierarchical", "p": p, "h": h, "tau": tau, "D": D,
                            **resource_row(allocation, J=J, Q=Q, value_dim=v),
                        })
    brute = min(
        brute_hierarchy,
        key=lambda row: (row["operation_proxy"], row["mutable_state_scalars"], row["p"], row["h"], row["tau"], row["D"]),
    )
    exact = by_method["hierarchical"]
    assert tuple(exact[key] for key in ("p", "h", "tau", "D", "operation_proxy")) == tuple(
        brute[key] for key in ("p", "h", "tau", "D", "operation_proxy")
    )
    assert any(row["method"] == "local_delta" and row["p"] == cap // v for row in frontiers)


def test_resource_accounting_has_no_fictitious_local_only_la_charge() -> None:
    delta = resource_row(MethodAllocation("local_delta", 5), J=16, Q=3, value_dim=2)
    additive = resource_row(MethodAllocation("local_additive", 5), J=16, Q=3, value_dim=2)
    assert delta["global_commit_mac"] == additive["global_commit_mac"] == 0
    assert delta["folded_state_scalars"] == additive["folded_state_scalars"] == 0
    assert delta["operation_proxy"] == 16 * (3 * 10 + 2) + 3 * 10
    assert additive["operation_proxy"] == (16 + 3) * 10


def test_same_p_flat_stream_is_identical_before_hierarchy_reshape() -> None:
    config = {
        "task": {"value_dim": 2},
        "heldout": {
            "episodes_per_configuration": 7,
            "local_seed_offset": 10,
            "value_seed_offset": 20,
        },
    }
    x1, y1 = runner.heldout_streams(config, seed=77, J=8, p=4)
    x2, y2 = runner.heldout_streams(config, seed=77, J=8, p=4)
    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(x1.reshape(7, 4, 2, 4).reshape_as(x1), x2)
    x_other, y_other = runner.heldout_streams(config, seed=77, J=8, p=3)
    assert x_other.shape[-1] == 3
    torch.testing.assert_close(y_other, y1)  # Values are shared even across p.


def test_dependency_barrier_rejects_imported_model_drift(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    model = tmp_path / "model.py"
    model.write_text("VERSION=1\n")
    monkeypatch.setattr(runner, "MODEL_CODE", model)
    output = tmp_path / "results"
    output.mkdir()
    (output / "config_snapshot.json").write_bytes(config_path.read_bytes())
    frontier = output / "analytic_frontiers.csv"
    frontier.write_text("x\n1\n")
    artifact = {
        "input_hashes": runner.input_hashes(config_path),
        "analytic_frontiers_sha256": runner.sha256(frontier),
        "selections": [], "evaluation_union": [],
    }
    artifact_path = output / "selections_preregistered.json"
    artifact_path.write_text(json.dumps(artifact))
    (output / "selections_preregistered.sha256").write_text(runner.sha256(artifact_path) + "\n")
    runner.verify_barrier(config_path, output)
    model.write_text("VERSION=2\n")
    with pytest.raises(RuntimeError, match="frozen inputs changed"):
        runner.verify_barrier(config_path, output)


def test_evaluation_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    runner.refuse_evaluation_overwrite(output)
    (output / "heldout_evaluation.csv").write_text("completed\n")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.refuse_evaluation_overwrite(output)


def test_coverage_is_abstention_safe() -> None:
    assert coverage([], threshold=0.9) == {
        "eligible_views": 0, "empirically_feasible_views": 0,
        "coverage": None, "passed": False,
    }
    one_of_ten = coverage(
        [{"selected": index == 0, "empirical_feasible": index == 0} for index in range(10)], threshold=0.9
    )
    assert one_of_ten["coverage"] == pytest.approx(0.1)
    assert one_of_ten["passed"] is False
    deployment_oracles = coverage(
        [
            {"population_oracle_exists": True, "selected": False, "empirical_feasible": False},
            {"population_oracle_exists": False, "selected": False, "empirical_feasible": False},
        ],
        threshold=0.9,
    )
    assert deployment_oracles["eligible_views"] == 1
    assert deployment_oracles["coverage"] == 0.0


def test_registered_runtime_limit_is_enforced() -> None:
    config = {"runtime": {"scientific_stage_time_limit_seconds": 0.0}}
    with pytest.raises(RuntimeError, match="exceeded registered"):
        runner.enforce_runtime_limit(0.0, config, "test")


def test_registered_fixed_population_values() -> None:
    witness = hierarchy_moments(32, 2, 2, 32)
    assert witness["delta"] + witness["A"] / 12 == pytest.approx(0.09390797334558823)
    assert local_additive_error(310, 32) == pytest.approx(0.1)
    assert endpoint_delta_error_aggregate(1, 16) == pytest.approx(1.875)
