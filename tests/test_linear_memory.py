from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.linear_memory.model import (
    LinearAssociativeWriter,
    WriterTrace,
    build_commit_schedule,
    conditional_contamination_prediction,
    error_decomposition,
    exact_snapshot_read,
    folded_linear_attention_read,
    linear_attention_read,
    two_fact_expected_errors,
)


def _fixed_trace(snapshots: torch.Tensor, addresses: torch.Tensor, h: int, tau: int) -> WriterTrace:
    schedule = build_commit_schedule(addresses.shape[0], h=h, tau=tau)
    return WriterTrace(addresses, snapshots, snapshots, schedule)


def test_orthogonal_local_keys_eta_one_have_no_forgetting() -> None:
    model = LinearAssociativeWriter(3, 3, 2, h=3, tau=3, eta_init=0.5).double()
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(3, dtype=torch.float64))
        model.eta_logit.copy_(torch.tensor(math.log(4.0), dtype=torch.float64))
    values = torch.tensor(
        [[[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25]]], dtype=torch.float64
    )
    recalled = exact_snapshot_read(model(values))
    torch.testing.assert_close(recalled, values)


def test_scalar_repeated_key_overwrites_previous_value() -> None:
    model = LinearAssociativeWriter(2, 1, 1, h=2, tau=2, eta_init=0.5).double()
    with torch.no_grad():
        model.encoder.weight.fill_(1.0)
        model.eta_logit.copy_(torch.tensor(math.log(4.0), dtype=torch.float64))
    values = torch.tensor([[[2.0], [-3.0]]], dtype=torch.float64)
    recalled = exact_snapshot_read(model(values))
    assert recalled[0, 0, 0].item() == pytest.approx(-3.0)
    assert recalled[0, 1, 0].item() == pytest.approx(-3.0)


def test_linear_transition_gradient_path_identity() -> None:
    # m_t = J_t m_(t-1), L = sum_c b_c^T m_c.
    Js = [
        torch.tensor([[0.8, 0.1], [-0.2, 0.7]], dtype=torch.float64),
        torch.tensor([[0.5, -0.3], [0.4, 0.9]], dtype=torch.float64),
        torch.tensor([[1.1, 0.2], [0.0, 0.6]], dtype=torch.float64),
    ]
    m0 = torch.tensor([0.7, -1.2], dtype=torch.float64, requires_grad=True)
    states = [m0]
    for J in Js:
        state = J @ states[-1]
        state.retain_grad()
        states.append(state)
    commits = {
        1: torch.tensor([0.3, -0.4], dtype=torch.float64),
        2: torch.tensor([-0.2, 0.6], dtype=torch.float64),
        3: torch.tensor([0.5, 0.1], dtype=torch.float64),
    }
    loss = sum((commits[c] * states[c]).sum() for c in commits)
    loss.backward()

    j = 1
    manual = commits[1].clone()
    product = torch.eye(2, dtype=torch.float64)
    for c in range(j + 1, 4):
        product = Js[c - 1] @ product
        manual = manual + product.T @ commits[c]
    torch.testing.assert_close(states[j].grad, manual)


def test_folded_linear_attention_equals_direct_kernel_sum() -> None:
    torch.manual_seed(2)
    batch, segments, value_dim, local_dim, global_dim = 3, 5, 2, 3, 7
    schedule = build_commit_schedule(segments, h=3, tau=2)
    snapshots = torch.randn(
        batch, len(schedule.commit_indices), value_dim, local_dim, dtype=torch.float64
    )
    addresses = F.normalize(torch.randn(segments, local_dim, dtype=torch.float64), dim=-1)
    trace = WriterTrace(addresses, snapshots, snapshots, schedule)
    key_table = F.normalize(
        torch.randn(batch, segments, global_dim, dtype=torch.float64), dim=-1
    )
    direct, _ = linear_attention_read(trace, key_table)
    folded = folded_linear_attention_read(trace, key_table)
    torch.testing.assert_close(folded, direct, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("reader", ["exact", "la"])
def test_detached_and_full_match_forward_but_change_writer_gradients(reader: str) -> None:
    torch.manual_seed(11)
    full = LinearAssociativeWriter(6, 3, 2, h=6, tau=2, eta_init=0.55).double()
    detached = LinearAssociativeWriter(6, 3, 2, h=6, tau=2, eta_init=0.55).double()
    detached.load_state_dict(full.state_dict())
    values = F.normalize(torch.randn(4, 6, 2, dtype=torch.float64), dim=-1)
    keys = F.normalize(torch.randn(4, 6, 9, dtype=torch.float64), dim=-1)

    trace_full = full(values, detach_intermediate=False)
    trace_detached = detached(values, detach_intermediate=True)
    if reader == "exact":
        pred_full = exact_snapshot_read(trace_full)
        pred_detached = exact_snapshot_read(trace_detached)
    else:
        pred_full, _ = linear_attention_read(trace_full, keys)
        pred_detached, _ = linear_attention_read(trace_detached, keys)
    torch.testing.assert_close(pred_full, pred_detached)

    loss_full = (pred_full - values).square().sum(dim=-1).mean()
    loss_detached = (pred_detached - values).square().sum(dim=-1).mean()
    grad_full = torch.autograd.grad(loss_full, tuple(full.parameters()))
    grad_detached = torch.autograd.grad(loss_detached, tuple(detached.parameters()))
    flat_full = torch.cat([g.reshape(-1) for g in grad_full])
    flat_detached = torch.cat([g.reshape(-1) for g in grad_detached])
    assert torch.linalg.vector_norm(flat_full - flat_detached) > 1e-6

    # A final-snapshot-only loss has the same gradient in both modes: detaching
    # storage branches did not truncate the continuing recurrence or endpoint.
    full.zero_grad(set_to_none=True)
    detached.zero_grad(set_to_none=True)
    trace_full = full(values, detach_intermediate=False)
    trace_detached = detached(values, detach_intermediate=True)
    final_mask = trace_full.schedule.group_final_mask
    final_full = trace_full.storage_snapshots[:, final_mask].square().sum()
    final_detached = trace_detached.storage_snapshots[:, final_mask].square().sum()
    endpoint_grad_full = torch.autograd.grad(final_full, tuple(full.parameters()))
    endpoint_grad_detached = torch.autograd.grad(final_detached, tuple(detached.parameters()))
    for got, expected in zip(endpoint_grad_detached, endpoint_grad_full):
        torch.testing.assert_close(got, expected)


def test_reset_and_nondivisible_commit_coverage() -> None:
    schedule = build_commit_schedule(num_segments=12, h=5, tau=2)
    assert schedule.commit_indices.tolist() == [1, 3, 4, 6, 8, 9, 11]
    assert schedule.group_final_mask.tolist() == [False, False, True, False, False, True, True]
    assert schedule.fact_to_commit.tolist() == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6]
    assert schedule.routing_age.tolist() == [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    endpoint = build_commit_schedule(num_segments=12, h=5, tau=5)
    assert endpoint.commit_indices.tolist() == [4, 9, 11]


def test_reset_blocks_later_group_gradient_to_prior_group_values() -> None:
    model = LinearAssociativeWriter(4, 2, 1, h=2, tau=2, eta_init=0.6).double()
    addresses = F.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [1.0, -0.2], [-0.3, 1.0]],
            dtype=torch.float64,
        ),
        dim=-1,
    )
    with torch.no_grad():
        model.encoder.weight.copy_(addresses.T)
    values = torch.tensor(
        [[[0.2], [-0.7], [1.1], [0.4]]], dtype=torch.float64, requires_grad=True
    )
    trace = model(values)
    later_group_loss = trace.live_snapshots[:, 1].square().sum()
    value_grad = torch.autograd.grad(later_group_loss, values)[0]
    torch.testing.assert_close(value_grad[:, :2], torch.zeros_like(value_grad[:, :2]))
    assert torch.linalg.vector_norm(value_grad[:, 2:]) > 0


def test_delta_writer_jacobian_matches_transition_product_with_k_two() -> None:
    inner_steps = 2
    eta = 0.6
    model = LinearAssociativeWriter(
        3, 3, 1, h=3, tau=3, inner_steps=inner_steps, eta_init=eta
    ).double()
    addresses = F.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            dtype=torch.float64,
        ),
        dim=-1,
    )
    with torch.no_grad():
        model.encoder.weight.copy_(addresses.T)
    values = torch.tensor(
        [[[0.3], [-0.8], [1.2]]], dtype=torch.float64, requires_grad=True
    )
    endpoint = model(values).live_snapshots[0, 0, 0]
    actual = torch.stack(
        [
            torch.autograd.grad(endpoint[j], values, retain_graph=True)[0][0, 0, 0]
            for j in range(model.local_dim)
        ]
    )

    actual_eta = float(model.eta.detach())
    identity = torch.eye(model.local_dim, dtype=torch.float64)
    transition_1 = torch.linalg.matrix_power(
        identity - actual_eta * torch.outer(addresses[1], addresses[1]), inner_steps
    )
    transition_2 = torch.linalg.matrix_power(
        identity - actual_eta * torch.outer(addresses[2], addresses[2]), inner_steps
    )
    acquisition = 1.0 - (1.0 - actual_eta) ** inner_steps
    expected = acquisition * addresses[0] @ transition_1 @ transition_2
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_error_decomposition_includes_cross_term() -> None:
    target = torch.tensor([[[1.0, -2.0], [0.5, 0.25]]], dtype=torch.float64)
    exact = torch.tensor([[[0.8, -1.7], [0.2, 0.4]]], dtype=torch.float64)
    pred = torch.tensor([[[1.1, -1.5], [-0.1, 0.8]]], dtype=torch.float64)
    terms = error_decomposition(pred, exact, target)
    torch.testing.assert_close(
        terms["total"], terms["local"] + terms["contamination"] + terms["cross"]
    )
    assert terms["cross"].abs().max() > 0


def test_correlated_payload_contamination_prediction_monte_carlo() -> None:
    # Snapshot payloads are intentionally highly correlated.  Only global
    # keys are resampled; the 1/D conditional prediction must still hold.
    torch.manual_seed(7)
    samples, segments, local_dim, value_dim, D = 12000, 4, 2, 2, 11
    schedule = build_commit_schedule(segments, h=4, tau=1)
    base = torch.tensor(
        [[1.0, -0.5], [0.25, 0.75]], dtype=torch.float64
    )
    offsets = torch.arange(1, 5, dtype=torch.float64).view(1, 4, 1, 1)
    snapshots = base.view(1, 1, 2, 2) * offsets
    addresses = F.normalize(
        torch.tensor(
            [[1.0, 0.2], [0.8, -0.1], [-0.3, 1.0], [0.5, 0.7]],
            dtype=torch.float64,
        ),
        dim=-1,
    )
    trace = WriterTrace(addresses, snapshots, snapshots, schedule)
    predicted = conditional_contamination_prediction(trace, D)[0, 0]

    keys = F.normalize(torch.randn(samples, segments, D, dtype=torch.float64), dim=-1)
    repeated_trace = WriterTrace(
        addresses,
        snapshots.expand(samples, -1, -1, -1),
        snapshots.expand(samples, -1, -1, -1),
        schedule,
    )
    la, _ = linear_attention_read(repeated_trace, keys)
    exact = exact_snapshot_read(repeated_trace)
    observed_samples = (la[:, 0] - exact[:, 0]).square().sum(dim=-1)
    observed = observed_samples.mean()
    standard_error = observed_samples.std(unbiased=True) / math.sqrt(samples)
    assert abs(observed - predicted) <= 5.0 * standard_error + 0.01 * predicted


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.7, 1.0])
def test_two_fact_closed_form(rho: float) -> None:
    D = 13
    theory = two_fact_expected_errors(rho, D)
    values = torch.tensor(
        [[[y1], [y2]] for y1 in (-1.0, 1.0) for y2 in (-1.0, 1.0)],
        dtype=torch.float64,
    )
    x = torch.tensor(
        [[1.0, 0.0], [rho, math.sqrt(max(0.0, 1.0 - rho * rho))]],
        dtype=torch.float64,
    )
    endpoint = LinearAssociativeWriter(2, 2, 1, h=2, tau=2, eta_init=0.5).double()
    snapshots = LinearAssociativeWriter(2, 2, 1, h=2, tau=1, eta_init=0.5).double()
    with torch.no_grad():
        for model in (endpoint, snapshots):
            model.encoder.weight.copy_(x.T)
            model.eta_logit.copy_(torch.tensor(math.log(4.0), dtype=torch.float64))
    endpoint_pred = exact_snapshot_read(endpoint(values))
    endpoint_nmse = float((endpoint_pred - values).square().mean())
    assert endpoint_nmse == pytest.approx(theory["endpoint_nmse"], abs=1e-12)

    snapshot_trace = snapshots(values)
    snapshot_pred = exact_snapshot_read(snapshot_trace)
    torch.testing.assert_close(snapshot_pred, values, atol=1e-12, rtol=1e-12)
    predicted_noise = float(
        conditional_contamination_prediction(snapshot_trace, D).mean()
    )
    assert predicted_noise == pytest.approx(theory["snapshot_la_nmse"], abs=1e-12)
    if rho == 0.0:
        assert math.isinf(theory["crossover_D_strict"])
    else:
        expected = (1.0 + rho**4) / (rho**2 + rho**4)
        assert theory["crossover_D_strict"] == pytest.approx(expected)
