import pytest
import torch
from types import SimpleNamespace
from transformers import GPT2Config

import energy_gradmem as energy_gradmem_module
from energy_gradmem import EnergyGradMem, EnergyGradMemConfig
from grad_memgpt import GradMemGPT, GradMemGPTConfig
from run_energy_gradmem_on_kv_retrieval import EnergyFreezeCallback, strip_trailing_context_separator


def _base_config():
    return GPT2Config(
        vocab_size=101,
        n_embd=48,
        n_layer=2,
        n_head=4,
        n_positions=48,
        n_ctx=48,
        pad_token_id=0,
        eos_token_id=2,
    )


def _model(
    memory_backend="prefix",
    K=2,
    energy_future_mode="next_token",
    inner_objective="neural",
    energy_model_type="lstm",
    energy_segment_state_size=None,
    segment_write_mode="sequential",
    segment_size=None,
    memory_rotation="none",
    memory_rotation_angle=None,
    inner_lr=0.01,
    inner_clip_norm=None,
    use_adam=False,
    use_write_head=False,
    **kwargs,
):
    return EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend=memory_backend,
            n_mem_tokens=4,
            K=K,
            lr=inner_lr,
            use_adam=use_adam,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            use_write_head=use_write_head,
            attn_implementation="eager",
            inner_clip_norm=inner_clip_norm,
            inner_objective=inner_objective,
            energy_hidden_size=32,
            energy_num_layers=2,
            energy_dropout=0.0,
            energy_future_mode=energy_future_mode,
            energy_model_type=energy_model_type,
            energy_segment_state_size=energy_segment_state_size,
            segment_write_mode=segment_write_mode,
            segment_size=segment_size,
            memory_rotation=memory_rotation,
            memory_rotation_angle=memory_rotation_angle,
            **kwargs,
        )
    )


def test_strip_trailing_context_separator_removes_only_final_pipe():
    batch = [
        {"context": "!Tg:ON!!jr:Tk!|", "query": "Tg:", "target": "ON"},
        {"context": "!Tg:ON!!jr:Tk!", "query": "jr:", "target": "Tk"},
    ]

    stripped = strip_trailing_context_separator(batch)

    assert stripped[0]["context"] == "!Tg:ON!!jr:Tk!"
    assert stripped[1]["context"] == "!Tg:ON!!jr:Tk!"
    assert batch[0]["context"] == "!Tg:ON!!jr:Tk!|"


def test_forward_single_segment_prefix():
    torch.manual_seed(0)
    model = _model()
    model.eval()

    B, S, Q = 2, 6, 4
    context = torch.randint(1, 101, (B, S))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
        return_mem=True,
        return_energy_state=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)
    assert "mem" in output
    assert "energy_state" in output
    h, c = output["energy_state"]
    assert h.shape == (2, B, 32)
    assert c.shape == (2, B, 32)


def test_forward_multi_segment_prefix_unequal_lengths():
    torch.manual_seed(0)
    model = _model()
    model.eval()

    B, Q = 2, 4
    segments = [
        torch.randint(1, 101, (B, 5)),
        torch.randint(1, 101, (B, 7)),
        torch.randint(1, 101, (B, 3)),
    ]
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        labels=labels,
        return_energy_state=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)
    assert "inner_loss" in output["inner_loop_stats"]
    assert output["energy_state"][0].shape == (2, B, 32)


def test_forward_parallel_segments_prefix_equal_lengths():
    torch.manual_seed(0)
    model = _model(K=2, energy_future_mode="none", segment_write_mode="parallel")
    model.eval()

    B, Q = 2, 4
    segments = [torch.randint(1, 101, (B, 5)), torch.randint(1, 101, (B, 5))]
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        labels=labels,
        return_mem=True,
        return_energy_state=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)
    assert output["mem"].shape == (B, 4, 48)
    assert output["energy_state"][0].shape == (2, B, 32)
    assert "inner_energy_loss" in output["inner_loop_stats"]


def test_identity_energy_model_uses_transformer_hidden_states_directly():
    model = _model(K=1, energy_future_mode="none", energy_model_type="identity")
    model.eval()

    hidden = torch.randn(2, 5, 48)
    encoded, state = model.energy_encoder(hidden, state="kept")

    assert encoded is hidden
    assert state == "kept"
    assert model.energy_head[0].in_features == 48


def test_forward_identity_energy_model_prefix():
    torch.manual_seed(0)
    model = _model(K=1, energy_future_mode="none", energy_model_type="identity")
    model.eval()

    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))

    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels)

    assert torch.isfinite(output["loss"]).item()


@pytest.mark.parametrize("memory_backend", ["lora", "kv_cache"])
@pytest.mark.parametrize("energy_model_type", ["lstm", "identity"])
def test_neural_energy_forward_without_delta_regularization_supports_nonprefix_backends(
    memory_backend,
    energy_model_type,
):
    torch.manual_seed(0)
    backend_kwargs = {}
    if memory_backend == "lora":
        backend_kwargs = {
            "lora_mem_placement": "between_layers",
            "lora_mem_layers": "all",
            "lora_mem_r": 4,
            "lora_mem_alpha": 8,
        }
    model = _model(
        memory_backend=memory_backend,
        K=1,
        energy_future_mode="none",
        energy_model_type=energy_model_type,
        energy_delta_reg=0.0,
        **backend_kwargs,
    )
    model.eval()
    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))

    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
        return_mem=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert ("lora_mem" if memory_backend == "lora" else "kv_mem") in output
    assert output["inner_loop_stats"]["energy_delta_reg_loss"] == 0


def test_segment_delta_gru_shapes_and_token_energy_reduction():
    model = _model(
        K=1,
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=7,
    )
    hidden = torch.randn(2, 5, 96)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    state = torch.randn(2, 7)

    loss, returned_state, energy = model._energy_loss(hidden, mask, state)

    assert model.segment_state_gru.input_size == 4 * 48
    assert model.token_energy_mlp[0].in_features == 96 + 7
    assert energy.shape == (2, 5)
    assert loss.shape == ()
    assert returned_state is state


def test_inner_gradient_norm_clipping_bounds_update_and_preserves_direction():
    inner_lr = 0.25
    clip_norm = 1.0
    model = _model(
        K=1,
        energy_model_type="identity",
        energy_future_mode="none",
        inner_lr=inner_lr,
        inner_clip_norm=clip_norm,
    )
    param = torch.zeros(2, 2, 3)
    grad = torch.tensor(
        [
            [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
            [[-6.0, 8.0, 0.0], [0.0, 15.0, 0.0]],
        ]
    )

    updated = model._updated_inner_params([param], [grad], {}, local_step=0)[0]
    update = updated - param
    update_norms = update.flatten(start_dim=1).norm(dim=1)
    expected_directions = -grad.flatten(start_dim=1)
    actual_directions = update.flatten(start_dim=1)
    cosine = torch.nn.functional.cosine_similarity(actual_directions, expected_directions)

    assert torch.all(update_norms <= inner_lr * clip_norm + 1e-6)
    assert torch.allclose(update_norms, torch.full_like(update_norms, inner_lr), atol=1e-6)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)


def test_segment_delta_gru_initial_energy_is_independent_of_segment_state():
    torch.manual_seed(0)
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=7,
    )
    hidden = torch.randn(2, 5, 48)
    zero_state = torch.zeros(2, 7)
    random_state = torch.randn(2, 7)

    zero_state_energy, _ = model._energy_values(hidden, zero_state)
    random_state_energy, _ = model._energy_values(hidden, random_state)

    first_linear = model.token_energy_mlp[0]
    assert torch.count_nonzero(first_linear.weight[:, 48:]) == 0
    assert torch.count_nonzero(first_linear.weight[:, :48]) > 0
    assert torch.allclose(zero_state_energy, random_state_energy)


def test_segment_delta_gru_zero_initialized_state_columns_receive_gradients():
    torch.manual_seed(0)
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=7,
    )
    hidden = torch.randn(2, 5, 48)
    state = torch.randn(2, 7)
    mask = torch.ones(2, 5, dtype=torch.bool)

    loss, _, _ = model._energy_loss(hidden, mask, state)
    loss.backward()

    state_column_grad = model.token_energy_mlp[0].weight.grad[:, 48:]
    assert torch.count_nonzero(model.token_energy_mlp[0].weight[:, 48:]) == 0
    assert torch.isfinite(state_column_grad).all()
    assert state_column_grad.norm() > 0


def test_segment_delta_gru_preserves_flattened_memory_slot_order():
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=5,
    )
    captured = []

    class CaptureGRU(torch.nn.Module):
        def forward(self, inputs, state):
            captured.append(inputs.detach().clone())
            return state

    model.segment_state_gru = CaptureGRU()
    delta = torch.arange(4 * 48, dtype=torch.float32).reshape(1, 4, 48)
    state = torch.zeros(1, 5)
    active = torch.ones(1, dtype=torch.bool)

    model._update_segment_delta_state(state, delta, active)
    model._update_segment_delta_state(state, delta[:, [1, 0, 2, 3]], active)

    assert torch.equal(captured[0], delta.flatten(start_dim=1))
    assert torch.equal(captured[1][:, :48], captured[0][:, 48:96])
    assert torch.equal(captured[1][:, 48:96], captured[0][:, :48])
    assert not torch.equal(captured[0], captured[1])


def test_segment_delta_state_updates_once_per_segment_and_is_used_next_segment():
    torch.manual_seed(0)
    model = _model(
        K=3,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=6,
    )
    model.eval()
    conditioned_states = []
    gru_outputs = []

    def capture_conditioned(_module, args):
        conditioned_states.append(args[0][..., -6:].detach().clone())

    def capture_gru_output(_module, _args, output):
        gru_outputs.append(output.detach().clone())

    mlp_hook = model.token_energy_mlp.register_forward_pre_hook(capture_conditioned)
    gru_hook = model.segment_state_gru.register_forward_hook(capture_gru_output)
    segments = [torch.randint(1, 101, (2, 5)), torch.randint(1, 101, (2, 5))]
    query = torch.randint(1, 101, (2, 4))
    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        return_energy_state=True,
    )
    mlp_hook.remove()
    gru_hook.remove()

    assert len(conditioned_states) == 2 * model.K
    assert len(gru_outputs) == 2
    assert all(torch.equal(state, conditioned_states[0]) for state in conditioned_states[:model.K])
    assert torch.count_nonzero(conditioned_states[0]) == 0
    assert all(
        torch.allclose(state[:, 0, :], gru_outputs[0])
        for state in conditioned_states[model.K:]
    )
    assert torch.allclose(output["energy_state"], gru_outputs[1])


def test_segment_delta_gru_uses_memory_before_first_and_after_final_inner_step(monkeypatch):
    model = _model(
        K=2,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=5,
    )
    model.eval()
    captured = []

    class CaptureGRU(torch.nn.Module):
        def forward(self, inputs, state):
            captured.append(inputs.detach().clone())
            return state

    model.segment_state_gru = CaptureGRU()
    increment = torch.arange(4 * 48, dtype=model.mem.dtype).reshape(1, 4, 48) / 1000

    def deterministic_updates(inner_params, grads, opt_state, local_step):
        del grads, opt_state
        return [inner_params[0] + (local_step + 1) * increment]

    monkeypatch.setattr(model, "_updated_inner_params", deterministic_updates)
    context = torch.randint(1, 101, (1, 5))
    query = torch.randint(1, 101, (1, 4))
    model(
        {"context_input_ids": context, "query_input_ids": query},
        return_energy_state=True,
    )

    expected_delta = 3 * increment
    assert len(captured) == 1
    assert torch.allclose(captured[0], expected_delta.flatten(start_dim=1))


def test_segment_delta_gru_padding_updates_only_active_samples():
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=5,
    )
    model.eval()
    calls = 0

    class IncrementStateGRU(torch.nn.Module):
        def forward(self, inputs, state):
            nonlocal calls
            calls += 1
            return state + 1

    model.segment_state_gru = IncrementStateGRU()
    segments = [
        torch.ones(2, 5, dtype=torch.long),
        torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]),
        torch.zeros(2, 5, dtype=torch.long),
    ]
    query = torch.ones(2, 4, dtype=torch.long)
    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        return_energy_state=True,
    )

    assert calls == 2
    assert torch.equal(output["energy_state"][0], torch.ones(5))
    assert torch.equal(output["energy_state"][1], torch.full((5,), 2.0))


def test_segment_delta_gru_k_zero_preserves_state_and_rotation_contract():
    model = _model(
        K=0,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=5,
        memory_rotation="pairwise",
        memory_rotation_angle=torch.pi / 2,
    )
    model.eval()
    with torch.no_grad():
        initial_memory = torch.arange(model.mem.numel(), dtype=model.mem.dtype).reshape_as(model.mem)
        model.mem.copy_(initial_memory)

    external_state = torch.randn(2, 5)
    context = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    query = torch.ones(2, 4, dtype=torch.long)
    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        return_mem=True,
        return_energy_state=True,
        energy_state=external_state,
    )

    expected_rotated = EnergyGradMem._rotate_feature_pairs(initial_memory, torch.pi / 2)
    assert torch.allclose(output["mem"][0], expected_rotated, atol=1e-6)
    assert torch.equal(output["mem"][1], initial_memory)
    assert torch.equal(output["energy_state"], external_state)
    assert output["inner_loop_stats"]["segment_delta_norm_mean"] == 0


def test_segment_delta_gru_rotates_delta_and_memory_once(monkeypatch):
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=5,
        memory_rotation="pairwise",
        memory_rotation_angle=torch.pi / 2,
    )
    model.eval()
    captured = []

    class CaptureGRU(torch.nn.Module):
        def forward(self, inputs, state):
            captured.append(inputs.detach().clone())
            return state

    model.segment_state_gru = CaptureGRU()
    with torch.no_grad():
        model.mem.zero_()
    raw_delta = torch.arange(4 * 48, dtype=model.mem.dtype).reshape(1, 4, 48) / 1000

    def deterministic_updates(inner_params, grads, opt_state, local_step):
        del grads, opt_state, local_step
        return [inner_params[0] + raw_delta]

    monkeypatch.setattr(model, "_updated_inner_params", deterministic_updates)
    context = torch.ones(1, 5, dtype=torch.long)
    query = torch.ones(1, 4, dtype=torch.long)
    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        return_mem=True,
        return_energy_state=True,
    )

    once_rotated = EnergyGradMem._rotate_feature_pairs(raw_delta, torch.pi / 2)
    assert torch.allclose(captured[0], once_rotated.flatten(start_dim=1), atol=1e-6)
    assert torch.allclose(output["mem"], once_rotated, atol=1e-6)


def test_segment_delta_gru_second_order_target_gradients_are_finite():
    torch.manual_seed(0)
    model = _model(
        K=2,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=8,
    )
    model.train()
    segments = [torch.randint(1, 101, (2, 5)), torch.randint(1, 101, (2, 5))]
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))

    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        labels=labels,
    )
    output["loss"].backward()

    assert model.grad_mode == "second"
    for module in (model.segment_state_gru, model.token_energy_mlp):
        assert all(param.grad is not None for param in module.parameters())
        assert all(torch.isfinite(param.grad).all() for param in module.parameters())
    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all()


def test_segment_delta_gru_state_round_trip_and_validation():
    model = _model(
        K=1,
        energy_future_mode="next_token",
        energy_model_type="segment_delta_gru",
        energy_segment_state_size=5,
        energy_ce_guidance=True,
    )
    model.eval()
    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))

    first = model(
        {"context_input_ids": context, "query_input_ids": query},
        return_energy_state=True,
    )
    second = model(
        {"context_input_ids": context, "query_input_ids": query},
        return_energy_state=True,
        energy_state=first["energy_state"],
    )

    assert first["energy_state"].shape == (2, 5)
    assert second["energy_state"].shape == (2, 5)
    assert not torch.equal(first["energy_state"], second["energy_state"])

    with pytest.raises(ValueError, match="must be a tensor"):
        model(
            {"context_input_ids": context, "query_input_ids": query},
            energy_state=(torch.zeros(1), torch.zeros(1)),
        )
    with pytest.raises(ValueError, match=r"must have shape \(2, 5\)"):
        model(
            {"context_input_ids": context, "query_input_ids": query},
            energy_state=torch.zeros(2, 6),
        )


def test_mamba2_energy_model_uses_fla_layer(monkeypatch):
    seen_kwargs = None

    class FakeFlaMamba2(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            nonlocal seen_kwargs
            seen_kwargs = kwargs

        def forward(self, hidden, **kwargs):
            return hidden + 1.0, None, None

    monkeypatch.setattr(energy_gradmem_module, "FlaMamba2", FakeFlaMamba2)
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="mamba2",
        energy_mamba_state_size=32,
        energy_mamba_conv_kernel=3,
        energy_mamba_expand=1,
        energy_mamba_head_dim=16,
        energy_mamba_chunk_size=64,
        energy_mamba_backend="triton",
    )

    hidden = torch.zeros(2, 5, 48)
    encoded, state = model.energy_encoder(hidden, state="ignored")

    assert torch.equal(encoded, torch.ones_like(hidden))
    assert state == "ignored"
    assert seen_kwargs == {
        "hidden_size": 48,
        "state_size": 32,
        "conv_kernel": 3,
        "expand": 1,
        "head_dim": 16,
        "chunk_size": 64,
        "backend": "triton",
    }
    assert model.energy_head[0].in_features == 48


def test_mamba2_energy_model_rejects_invalid_head_dim_default():
    with pytest.raises(ValueError, match=r"energy_mamba_expand \* energy_input_size"):
        _model(K=1, energy_future_mode="none", energy_model_type="mamba2")


def test_mamba2_energy_model_requires_fla(monkeypatch):
    monkeypatch.setattr(energy_gradmem_module, "FlaMamba2", None)

    with pytest.raises(ImportError, match="flash-linear-attention"):
        _model(
            K=1,
            energy_future_mode="none",
            energy_model_type="mamba2",
            energy_mamba_head_dim=16,
        )


def test_forward_mamba2_energy_model_prefix(monkeypatch):
    class FakeFlaMamba2(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.proj = torch.nn.Linear(kwargs["hidden_size"], kwargs["hidden_size"])

        def forward(self, hidden, **kwargs):
            return self.proj(hidden), None, None

    monkeypatch.setattr(energy_gradmem_module, "FlaMamba2", FakeFlaMamba2)
    torch.manual_seed(0)
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="mamba2",
        energy_mamba_expand=1,
        energy_mamba_head_dim=16,
    )
    model.eval()

    B, Q = 2, 4
    context = torch.randint(1, 101, (B, 5))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels)

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)


def test_context_tensor_segment_size_returns_segment_list():
    model = _model(K=1, segment_size=4)
    context = torch.arange(24).reshape(2, 12)

    segments = model._context_segments(context)

    assert isinstance(segments, list)
    assert len(segments) == 3
    assert all(segment.shape == (2, 4) for segment in segments)
    assert torch.equal(torch.cat(segments, dim=1), context)


def test_context_tensor_segment_size_rejects_uneven_lengths():
    model = _model(K=1, segment_size=4)
    context = torch.arange(20).reshape(2, 10)

    with pytest.raises(ValueError, match="divisible by segment_size"):
        model._context_segments(context)


def test_context_segment_list_ignores_segment_size():
    model = _model(K=1, segment_size=4)
    segments = [torch.arange(10).reshape(2, 5), torch.arange(12).reshape(2, 6)]

    returned = model._context_segments(segments)
    assert len(returned) == len(segments)
    assert all(torch.equal(a, b) for a, b in zip(returned, segments))


def test_sequential_tensor_segment_size_matches_segment_list():
    torch.manual_seed(0)
    model_tensor = _model(K=1, energy_future_mode="none", segment_size=5)
    model_list = _model(K=1, energy_future_mode="none")
    model_list.load_state_dict(model_tensor.state_dict())
    model_tensor.eval()
    model_list.eval()

    B, Q = 2, 4
    segments = [torch.randint(1, 101, (B, 5)), torch.randint(1, 101, (B, 5))]
    context = torch.cat(segments, dim=1)
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output_tensor = model_tensor({"context_input_ids": context, "query_input_ids": query}, labels=labels)
    output_list = model_list({"context_input_ids": segments, "query_input_ids": query}, labels=labels)

    assert torch.allclose(output_tensor["predictions"], output_list["predictions"])
    assert torch.allclose(output_tensor["inner_loop_stats"]["inner_loss"], output_list["inner_loop_stats"]["inner_loss"])


def test_pairwise_memory_rotation_preserves_norm_and_odd_feature():
    memory = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

    rotated = EnergyGradMem._rotate_feature_pairs(memory, torch.pi / 2)

    expected = torch.tensor([[[-2.0, 1.0, -4.0, 3.0, 5.0]]])
    assert torch.allclose(rotated, expected, atol=1e-6)
    assert torch.allclose(rotated.norm(dim=-1), memory.norm(dim=-1))


def test_sequential_rotation_uses_pi_over_segments_and_rotates_final_segment():
    torch.manual_seed(0)
    model = _model(K=0, memory_rotation="pairwise")
    model.eval()

    with torch.no_grad():
        initial_memory = torch.arange(model.mem.numel(), dtype=model.mem.dtype).reshape_as(model.mem)
        model.mem.copy_(initial_memory)

    B, Q = 2, 4
    segments = [torch.randint(1, 101, (B, 5)), torch.randint(1, 101, (B, 5))]
    query = torch.randint(1, 101, (B, Q))

    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        return_mem=True,
    )

    expected = -initial_memory.unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(output["mem"], expected, atol=1e-5)


def test_explicit_memory_rotation_angle_overrides_default():
    model = _model(K=0, memory_rotation="pairwise", memory_rotation_angle=torch.pi / 2)
    model.eval()

    with torch.no_grad():
        model.mem.zero_()
        model.mem[..., 0] = 1.0

    context = torch.ones(1, 5, dtype=torch.long)
    query = torch.ones(1, 4, dtype=torch.long)
    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        return_mem=True,
    )

    assert torch.allclose(output["mem"][..., 0], torch.zeros_like(output["mem"][..., 0]), atol=1e-6)
    assert torch.allclose(output["mem"][..., 1], torch.ones_like(output["mem"][..., 1]), atol=1e-6)


def test_forward_with_memory_rotation_preserves_outer_gradients():
    torch.manual_seed(0)
    model = _model(K=1, memory_rotation="pairwise", memory_rotation_angle=0.1)
    model.train()

    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))
    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
    )
    output["loss"].backward()

    assert torch.isfinite(output["loss"]).item()
    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()


def test_padding_only_segment_does_not_change_default_rotation():
    torch.manual_seed(0)
    model_without_padding = _model(K=1, energy_future_mode="none", memory_rotation="pairwise")
    model_with_padding = _model(K=1, energy_future_mode="none", memory_rotation="pairwise")
    model_with_padding.load_state_dict(model_without_padding.state_dict())
    model_without_padding.eval()
    model_with_padding.eval()

    active_segments = [torch.randint(1, 101, (1, 5)), torch.randint(1, 101, (1, 5))]
    padded_segments = active_segments + [torch.zeros(1, 5, dtype=torch.long)]
    query = torch.randint(1, 101, (1, 4))

    output_without_padding = model_without_padding(
        {"context_input_ids": active_segments, "query_input_ids": query},
        return_mem=True,
    )
    output_with_padding = model_with_padding(
        {"context_input_ids": padded_segments, "query_input_ids": query},
        return_mem=True,
    )

    assert torch.allclose(output_with_padding["mem"], output_without_padding["mem"])
    assert torch.allclose(
        output_with_padding["predictions"],
        output_without_padding["predictions"],
        atol=1e-6,
    )


def test_rotation_mask_is_independent_for_each_sample():
    model = _model(K=0, memory_rotation="pairwise", memory_rotation_angle=torch.pi / 2)
    model.eval()

    with torch.no_grad():
        initial_memory = torch.arange(model.mem.numel(), dtype=model.mem.dtype).reshape_as(model.mem)
        model.mem.copy_(initial_memory)

    segments = [
        torch.ones(2, 5, dtype=torch.long),
        torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]),
    ]
    query = torch.ones(2, 4, dtype=torch.long)
    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        return_mem=True,
    )

    once_rotated = EnergyGradMem._rotate_feature_pairs(initial_memory, torch.pi / 2)
    assert torch.allclose(output["mem"][0], once_rotated, atol=1e-6)
    assert torch.allclose(output["mem"][1], -initial_memory, atol=1e-5)


def test_forward_parallel_segments_tensor_segment_size():
    torch.manual_seed(0)
    model = _model(K=2, energy_future_mode="none", segment_write_mode="parallel", segment_size=5)
    model.eval()

    B, Q = 2, 4
    context = torch.randint(1, 101, (B, 10))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
        return_mem=True,
        return_energy_state=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)
    assert output["mem"].shape == (B, 4, 48)
    assert output["energy_state"][0].shape == (2, B, 32)


def test_parallel_segments_rejects_unequal_lengths():
    model = _model(K=1, energy_future_mode="none", segment_write_mode="parallel")
    B, Q = 2, 4
    segments = [torch.randint(1, 101, (B, 5)), torch.randint(1, 101, (B, 6))]
    query = torch.randint(1, 101, (B, Q))

    with pytest.raises(ValueError, match="equal segment lengths"):
        model({"context_input_ids": segments, "query_input_ids": query})


def test_single_batch_train_updates_energy_params():
    torch.manual_seed(0)
    model = _model(K=2)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    B, Q = 2, 4
    segments = [torch.randint(1, 101, (B, 5)), torch.randint(1, 101, (B, 6))]
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    optimizer.zero_grad(set_to_none=True)
    output = model({"context_input_ids": segments, "query_input_ids": query}, labels=labels)
    assert torch.isfinite(output["loss"]).item()
    output["loss"].backward()

    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()
    assert model.mem.grad.detach().norm().item() > 0.0

    energy_grads = [
        p.grad.detach().norm()
        for name, p in model.named_parameters()
        if name.startswith("energy_") and p.grad is not None
    ]
    assert energy_grads
    assert torch.stack(energy_grads).sum().item() > 0.0


def test_energy_loss_uses_masked_average():
    model = _model(K=1, energy_future_mode="none")

    class IdentityEncoder(torch.nn.Module):
        def forward(self, hidden, state=None):
            return hidden, state

    model.energy_encoder = IdentityEncoder()
    model.energy_head = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.energy_head.weight.fill_(1.0)

    hidden = torch.tensor([[[1.0], [3.0], [100.0]], [[2.0], [4.0], [6.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    loss, state, energy = model._energy_loss(hidden, mask, None)

    assert state is None
    assert torch.equal(energy, hidden.squeeze(-1))
    assert torch.allclose(loss, torch.tensor(6.0))  # (1 + 3) / 2 + (2 + 4 + 6) / 3


@pytest.mark.parametrize(
    "inner_objective,expected_energy",
    [
        ("embedding_l1", torch.tensor([[2.0, 5.0, 2.0]])),
        ("embedding_l2", torch.tensor([[5.0, 25.0, 4.0]])),
    ],
)
def test_fixed_embedding_energy_objectives(inner_objective, expected_energy):
    model = _model(K=1, inner_objective=inner_objective)
    hidden = torch.tensor([[[1.0, 3.0, 2.0, 6.0], [4.0, 1.0, 9.0, 6.0], [8.0, 8.0, 6.0, 6.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    loss, state, energy = model._energy_loss(hidden, mask, None)

    assert state is None
    assert torch.allclose(energy, expected_energy)
    assert torch.allclose(loss, expected_energy[:, :2].mean())
    assert not hasattr(model, "energy_encoder")
    assert not hasattr(model, "energy_head")


def test_forward_with_fixed_embedding_energy_is_stateless():
    torch.manual_seed(0)
    model = _model(K=1, inner_objective="embedding_l1")
    B, S, Q = 2, 5, 4
    context = torch.randint(1, 101, (B, S))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
        return_energy_state=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["energy_state"] is None


def test_cross_entropy_inner_objective_has_no_energy_model():
    model = _model(K=1, inner_objective="cross_entropy", energy_future_mode="next_token")

    assert not hasattr(model, "energy_encoder")
    assert not hasattr(model, "energy_head")


def test_forward_cross_entropy_inner_objective_prefix():
    torch.manual_seed(0)
    model = _model(K=1, inner_objective="cross_entropy", energy_future_mode="next_token")
    model.eval()

    B, Q = 2, 4
    context = torch.randint(1, 101, (B, 5))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels, return_energy_state=True)

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)
    assert output["energy_state"] is None
    assert output["inner_loop_stats"]["inner_energy_loss"].item() == 0.0
    assert torch.allclose(
        output["inner_loop_stats"]["inner_loss"],
        output["inner_loop_stats"]["inner_ce_loss"],
        atol=1e-6,
    )


def test_cross_entropy_inner_objective_clears_stale_energy_state():
    model = _model(K=1, inner_objective="cross_entropy", energy_future_mode="next_token")
    stale_state = (
        torch.ones(2, 2, 32),
        torch.ones(2, 2, 32),
    )
    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))

    output = model(
        {"context_input_ids": context, "query_input_ids": query, "energy_state": stale_state},
        return_energy_state=True,
    )

    assert output["energy_state"] is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"energy_model_type": "identity"}, "energy_model_type"),
        ({"energy_inner_ce_weight": 1.0}, "energy_inner_ce_weight"),
        ({"energy_pretrain_steps": 1}, "energy pretraining"),
        ({"energy_pretrain_objective": "embedding_l2"}, "energy_pretrain_objective"),
        ({"energy_pretrain_l2_reg": 0.1}, "energy_pretrain_l2_reg"),
    ],
)
def test_cross_entropy_rejects_irrelevant_energy_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EnergyGradMemConfig(
            base_config=_base_config(),
            inner_objective="cross_entropy",
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            lr=0.01,
            use_adam=False,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            use_write_head=False,
            attn_implementation="eager",
            **kwargs,
        )


def test_cross_entropy_matches_gradmemgpt_default_write_loss():
    torch.manual_seed(0)
    gradmem = GradMemGPT(
        GradMemGPTConfig(
            base_config=_base_config(),
            memory_backend="prefix",
            n_mem_tokens=4,
            K=2,
            lr=0.01,
            use_adam=False,
            grad_mode="first",
            use_mem_proj=False,
            mem_proj_mode="none",
            use_write_head=False,
            attn_implementation="eager",
        )
    )
    energy = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend="prefix",
            n_mem_tokens=4,
            K=2,
            lr=0.01,
            use_adam=False,
            grad_mode="first",
            use_mem_proj=False,
            mem_proj_mode="none",
            use_write_head=False,
            attn_implementation="eager",
            inner_objective="cross_entropy",
        )
    )
    energy.load_state_dict(gradmem.state_dict(), strict=True)
    gradmem.eval()
    energy.eval()

    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))
    inputs = {"context_input_ids": context, "query_input_ids": query}

    gradmem_out = gradmem(inputs, labels=labels, return_mem=True)
    energy_out = energy(inputs, labels=labels, return_mem=True)

    assert torch.allclose(energy_out["predictions"], gradmem_out["predictions"], atol=1e-6)
    assert torch.allclose(energy_out["mem"], gradmem_out["mem"], atol=1e-6)
    assert torch.allclose(
        energy_out["inner_loop_stats"]["inner_loss"],
        gradmem_out["inner_loop_stats"]["inner_loss"],
        atol=1e-6,
    )
    assert torch.allclose(energy_out["loss"], gradmem_out["loss"], atol=1e-6)


def test_cross_entropy_allows_write_head():
    torch.manual_seed(0)
    model = _model(K=1, inner_objective="cross_entropy", use_write_head=True)
    model.eval()

    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))

    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels)

    assert hasattr(model, "write_head")
    assert torch.isfinite(output["loss"]).item()


def test_parallel_cross_entropy_allows_future_mode_because_energy_is_unused():
    torch.manual_seed(0)
    model = _model(
        K=1,
        inner_objective="cross_entropy",
        energy_future_mode="next_token",
        segment_write_mode="parallel",
    )
    model.eval()

    B, Q = 2, 4
    context = torch.randint(1, 101, (B, 10))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels)

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (B, Q + 1, 101)
    assert output["inner_loop_stats"]["inner_energy_loss"].item() == 0.0


def test_energy_ce_guidance_loss_uses_l1_detached_ce_and_mask():
    energy = torch.tensor([[1.0, 2.0, 100.0]], requires_grad=True)
    ce = torch.tensor([[3.0, 4.0, 1000.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    loss = EnergyGradMem._energy_ce_guidance_loss(energy, ce, mask)

    assert torch.allclose(loss, torch.tensor(2.0))
    loss.backward()
    assert energy.grad is not None
    assert ce.grad is None


@pytest.mark.parametrize("label_shift", [0, 1])
def test_write_token_ce_aligns_to_energy_positions(label_shift):
    model = _model(K=1, energy_future_mode="next_token")
    B, S, V = 1, 4, 7
    labels = torch.tensor([[1, 2, 3, -100]])
    mask = labels.ne(-100)

    logits_len = S + 1 if label_shift == 0 else S
    logits = torch.zeros(B, logits_len, V)
    write_out = SimpleNamespace(logits=logits)
    write_batch = {
        "logits_start": 0,
        "label_shift": label_shift,
        "lm_labels": labels,
        "mask": mask,
    }

    token_ce, token_ce_mask = model._write_token_ce(write_out, write_batch)

    if label_shift == 0:
        original_ce = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, S)
        assert token_ce.shape == (B, S - 1)
        assert torch.equal(token_ce_mask, mask[:, 1:])
        assert torch.allclose(token_ce, original_ce[:, 1:])
    else:
        assert token_ce.shape == (B, S - 1)
        assert torch.equal(token_ce_mask, mask[:, 1:])


def test_inner_ce_loss_keeps_prefix_token_zero():
    model = _model(K=1, energy_future_mode="none")
    B, S, V = 1, 4, 7
    labels = torch.tensor([[1, 2, 3, -100]])
    mask = labels.ne(-100)
    logits = torch.zeros(B, S + 1, V)
    write_out = SimpleNamespace(logits=logits)
    write_batch = {
        "logits_start": 0,
        "label_shift": 0,
        "lm_labels": labels,
        "mask": mask,
    }

    inner_ce = model._write_inner_ce_loss(write_out, write_batch)
    token_ce, token_ce_mask = model._write_token_ce(write_out, write_batch)
    original_ce = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, S)
    expected = (original_ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    assert torch.allclose(inner_ce, expected.sum())
    assert token_ce.shape == (B, S - 1)
    assert torch.equal(token_ce_mask, mask[:, 1:])


def test_forward_with_ce_guidance_reports_aux_loss():
    torch.manual_seed(0)
    model = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            lr=0.01,
            use_adam=False,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            attn_implementation="eager",
            inner_objective="neural",
            energy_hidden_size=32,
            energy_num_layers=2,
            energy_ce_guidance=True,
            energy_ce_guidance_alpha=0.01,
        )
    )
    model.train()

    B, S, Q = 2, 5, 4
    context = torch.randint(1, 101, (B, S))
    query = torch.randint(1, 101, (B, Q))
    labels = torch.randint(1, 101, (B, Q))

    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels)

    assert torch.isfinite(output["loss"]).item()
    assert "energy_ce_guidance_loss" in output["inner_loop_stats"]
    assert output["inner_loop_stats"]["energy_ce_guidance_loss"].item() >= 0.0


def test_inner_ce_weight_adds_ce_to_inner_objective():
    torch.manual_seed(0)
    weight = 2.0
    model = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            lr=0.01,
            use_adam=False,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            attn_implementation="eager",
            inner_objective="neural",
            energy_hidden_size=32,
            energy_num_layers=2,
            energy_future_mode="none",
            energy_inner_ce_weight=weight,
        )
    )
    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))

    output = model({"context_input_ids": context, "query_input_ids": query})
    stats = output["inner_loop_stats"]

    assert stats["inner_ce_loss"].item() > 0.0
    assert torch.allclose(
        stats["inner_loss"],
        stats["inner_energy_loss"] + weight * stats["inner_ce_loss"],
        atol=1e-5,
    )


def test_energy_pretrain_changes_only_energy_params():
    torch.manual_seed(0)
    model = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            lr=0.01,
            use_adam=False,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            attn_implementation="eager",
            inner_objective="neural",
            energy_hidden_size=32,
            energy_num_layers=2,
            energy_pretrain_steps=2,
            energy_pretrain_batch_size=2,
            energy_pretrain_seq_len=5,
            energy_pretrain_lr=1e-3,
        )
    )
    assert model.config.energy_pretrain_steps == 2

    energy_before = [p.detach().clone() for p in model.energy_encoder.parameters()]
    energy_before += [p.detach().clone() for p in model.energy_head.parameters()]
    mem_before = model.mem.detach().clone()
    backbone_before = model.model.get_input_embeddings().weight.detach().clone()

    model.pretrain_energy_objective()

    energy_after = [p.detach() for p in model.energy_encoder.parameters()]
    energy_after += [p.detach() for p in model.energy_head.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(energy_before, energy_after))
    assert torch.equal(mem_before, model.mem.detach())
    assert torch.equal(backbone_before, model.model.get_input_embeddings().weight.detach())

    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))
    output = model({"context_input_ids": context, "query_input_ids": query}, labels=labels)

    assert torch.isfinite(output["loss"]).item()


def test_embedding_mean_abs_diff_pretrain_changes_only_energy_params():
    torch.manual_seed(0)
    model = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            lr=0.01,
            use_adam=False,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            attn_implementation="eager",
            inner_objective="neural",
            energy_hidden_size=32,
            energy_num_layers=2,
            energy_pretrain_objective="embedding_mean_abs_diff",
            energy_pretrain_steps=2,
            energy_pretrain_batch_size=2,
            energy_pretrain_seq_len=5,
            energy_pretrain_lr=1e-3,
        )
    )

    energy_before = [p.detach().clone() for p in model.energy_encoder.parameters()]
    energy_before += [p.detach().clone() for p in model.energy_head.parameters()]
    mem_before = model.mem.detach().clone()
    backbone_before = model.model.get_input_embeddings().weight.detach().clone()

    model.pretrain_energy_objective()

    energy_after = [p.detach() for p in model.energy_encoder.parameters()]
    energy_after += [p.detach() for p in model.energy_head.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(energy_before, energy_after))
    assert torch.equal(mem_before, model.mem.detach())
    assert torch.equal(backbone_before, model.model.get_input_embeddings().weight.detach())


def test_embedding_mean_abs_diff_pretrain_loss_uses_random_vectors_not_backbone():
    model = _model(K=1)

    class SumEncoder(torch.nn.Module):
        input_size = 96

        def forward(self, hidden, state=None):
            return hidden.sum(dim=-1, keepdim=True), state

    model.energy_encoder = SumEncoder()
    model.energy_head = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.energy_head.weight.fill_(0.0)

    def run_write_model(*args, **kwargs):
        raise AssertionError("embedding_mean_abs_diff pretraining should not run the backbone")

    model._run_write_model = run_write_model

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    loss = model._embedding_mean_abs_diff_pretrain_loss(
        batch_size=2,
        seq_len=3,
        generator=gen,
        device=torch.device("cpu"),
    )

    expected_gen = torch.Generator(device="cpu")
    expected_gen.manual_seed(0)
    h1 = torch.randn(2, 3, 48, generator=expected_gen)
    h2 = torch.randn(2, 3, 48, generator=expected_gen)
    target = (h1 - h2).abs().mean(dim=-1)
    expected = target.pow(2).mean()

    assert torch.isfinite(loss).item()
    assert torch.allclose(loss, expected)


def test_embedding_l2_pretrain_loss_uses_random_vectors_not_backbone():
    model = _model(K=1)

    class SumEncoder(torch.nn.Module):
        input_size = 96

        def forward(self, hidden, state=None):
            return hidden.sum(dim=-1, keepdim=True), state

    model.energy_encoder = SumEncoder()
    model.energy_head = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.energy_head.weight.fill_(0.0)

    def run_write_model(*args, **kwargs):
        raise AssertionError("embedding_l2 pretraining should not run the backbone")

    model._run_write_model = run_write_model

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    loss = model._embedding_l2_pretrain_loss(
        batch_size=2,
        seq_len=3,
        generator=gen,
        device=torch.device("cpu"),
    )

    expected_gen = torch.Generator(device="cpu")
    expected_gen.manual_seed(0)
    h1 = torch.randn(2, 3, 48, generator=expected_gen)
    h2 = torch.randn(2, 3, 48, generator=expected_gen)
    target = (h1 - h2).pow(2).mean(dim=-1)
    expected = target.pow(2).mean()

    assert torch.isfinite(loss).item()
    assert torch.allclose(loss, expected)


def test_energy_pretrain_l2_regularization_uses_energy_params_only():
    model = _model(K=1)
    model.energy_pretrain_l2_reg = 0.25

    expected = torch.tensor(0.0)
    for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters()):
        expected = expected + param.detach().pow(2).sum()
    expected = expected * model.energy_pretrain_l2_reg

    assert torch.allclose(model._energy_pretrain_l2_regularization(), expected)

    with torch.no_grad():
        model.mem.add_(100.0)
    assert torch.allclose(model._energy_pretrain_l2_regularization(), expected)


@pytest.mark.parametrize(
    "energy_model_type,energy_future_mode",
    [
        ("lstm", "none"),
        ("identity", "none"),
        ("segment_delta_gru", "none"),
    ],
)
def test_outer_energy_weight_rms_uses_all_weights_and_excludes_biases(
    energy_model_type,
    energy_future_mode,
):
    model = _model(
        K=1,
        energy_model_type=energy_model_type,
        energy_future_mode=energy_future_mode,
        energy_weight_rms_reg=0.25,
        energy_weight_rms_threshold=0.5,
    )
    with torch.no_grad():
        for name, parameter in model._named_energy_parameters():
            parameter.fill_(100.0 if "bias" in name.rsplit(".", 1)[-1].lower() else 2.0)

    regularization, rms = model._energy_weight_rms_regularization()
    assert torch.allclose(rms, torch.tensor(2.0))
    assert torch.allclose(regularization, torch.tensor(0.25 * (2.0 - 0.5) ** 2))
    assert sum(parameter.numel() for parameter in model._energy_weight_parameters()) > 0

    with torch.no_grad():
        model.mem.add_(100.0)
    unchanged_regularization, unchanged_rms = model._energy_weight_rms_regularization()
    assert torch.equal(unchanged_rms, rms)
    assert torch.equal(unchanged_regularization, regularization)


def test_outer_energy_weight_rms_penalty_is_exactly_zero_below_threshold():
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="identity",
        energy_weight_rms_reg=3.0,
        energy_weight_rms_threshold=1.1,
    )
    with torch.no_grad():
        for parameter in model._energy_weight_parameters():
            parameter.fill_(1.0)

    regularization, rms = model._energy_weight_rms_regularization()

    assert torch.equal(rms, torch.tensor(1.0))
    assert torch.equal(regularization, torch.tensor(0.0))


def test_outer_energy_weight_rms_regularization_is_added_to_main_objective():
    torch.manual_seed(0)
    model = _model(
        K=0,
        energy_future_mode="none",
        energy_model_type="lstm",
        energy_weight_rms_reg=0.1,
        energy_weight_rms_threshold=0.0,
    )
    model.eval()
    context = torch.randint(1, 101, (2, 5))
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))

    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
    )
    expected_regularization, expected_rms = model._energy_weight_rms_regularization()

    assert torch.allclose(
        output["inner_loop_stats"]["energy_weight_rms_reg_loss"],
        expected_regularization.detach(),
    )
    assert torch.allclose(output["inner_loop_stats"]["energy_weight_rms"], expected_rms.detach())
    assert torch.allclose(
        output["loss"],
        output["inner_loop_stats"]["target_loss"] + expected_regularization,
    )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"energy_weight_rms_reg": -0.1}, "must be finite and non-negative"),
        ({"energy_weight_rms_threshold": float("nan")}, "must be finite and non-negative"),
        ({"energy_delta_reg": float("inf")}, "must be finite and non-negative"),
        ({"energy_delta_max": -0.1}, "must be finite and non-negative"),
        (
            {"inner_objective": "cross_entropy", "energy_weight_rms_reg": 0.1},
            "requires inner_objective='neural'",
        ),
        ({"energy_delta_reg": 0.1, "grad_mode": "first"}, "requires grad_mode='second'"),
        ({"energy_delta_reg": 0.1, "use_adam": True}, "requires use_adam=False"),
        (
            {"energy_delta_reg": 0.1, "segment_write_mode": "parallel", "energy_future_mode": "none"},
            "requires segment_write_mode='sequential'",
        ),
    ],
)
def test_outer_energy_regularization_rejects_invalid_config(kwargs, error):
    with pytest.raises(ValueError, match=error):
        EnergyGradMemConfig(base_config=_base_config(), **kwargs)


@pytest.mark.parametrize("energy_model_type", ["lstm", "identity", "segment_delta_gru"])
def test_energy_delta_penalty_uses_full_segment_delta_and_active_samples(
    monkeypatch,
    energy_model_type,
):
    model = _model(
        K=2,
        energy_future_mode="none",
        energy_model_type=energy_model_type,
        energy_delta_reg=0.5,
        energy_delta_max=0.2,
    )
    model.eval()
    increment = torch.full((1, 4, 48), 0.01, dtype=model.mem.dtype)

    def deterministic_updates(inner_params, grads, opt_state, global_step):
        del grads, opt_state
        return [inner_params[0] + (global_step + 1) * increment]

    monkeypatch.setattr(model, "_updated_inner_params", deterministic_updates)
    segments = [
        torch.ones(2, 5, dtype=torch.long),
        torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]),
        torch.zeros(2, 5, dtype=torch.long),
    ]
    query = torch.ones(2, 4, dtype=torch.long)

    output = model({"context_input_ids": segments, "query_input_ids": query})

    first_delta_norm = torch.linalg.vector_norm(3 * increment)
    second_delta_norm = torch.linalg.vector_norm(7 * increment)
    active_norms = torch.stack([first_delta_norm, first_delta_norm, second_delta_norm])
    expected = 0.5 * torch.relu(active_norms - 0.2).square().mean()
    stats = output["inner_loop_stats"]
    assert torch.allclose(stats["energy_delta_reg_loss"], expected)
    assert torch.allclose(
        stats["energy_delta_exceed_fraction"],
        (active_norms > 0.2).float().mean(),
    )
    assert torch.allclose(stats["segment_delta_norm_mean"], active_norms.mean())
    assert torch.allclose(stats["segment_delta_norm_max"], active_norms.max())


def test_energy_delta_penalty_uses_pre_rotation_delta(monkeypatch):
    common = dict(
        K=1,
        energy_future_mode="none",
        energy_model_type="identity",
        energy_delta_reg=0.5,
        energy_delta_max=0.1,
    )
    plain = _model(**common)
    rotated = _model(
        **common,
        memory_rotation="pairwise",
        memory_rotation_angle=torch.pi / 2,
    )
    rotated.load_state_dict(plain.state_dict())
    increment = torch.arange(4 * 48, dtype=plain.mem.dtype).reshape(1, 4, 48) / 1000

    def deterministic_updates(inner_params, grads, opt_state, global_step):
        del grads, opt_state, global_step
        return [inner_params[0] + increment]

    monkeypatch.setattr(plain, "_updated_inner_params", deterministic_updates)
    monkeypatch.setattr(rotated, "_updated_inner_params", deterministic_updates)
    context = torch.ones(1, 5, dtype=torch.long)
    query = torch.ones(1, 4, dtype=torch.long)

    plain_output = plain(
        {"context_input_ids": context, "query_input_ids": query},
        return_mem=True,
    )
    rotated_output = rotated(
        {"context_input_ids": context, "query_input_ids": query},
        return_mem=True,
    )

    assert torch.allclose(
        plain_output["inner_loop_stats"]["energy_delta_reg_loss"],
        rotated_output["inner_loop_stats"]["energy_delta_reg_loss"],
    )
    expected_rotated = EnergyGradMem._rotate_feature_pairs(plain_output["mem"], torch.pi / 2)
    assert torch.allclose(rotated_output["mem"], expected_rotated, atol=1e-6)


def test_energy_delta_penalty_is_exactly_zero_below_upper_bound():
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="identity",
        energy_delta_reg=2.0,
        energy_delta_max=100.0,
    )
    context = torch.ones(1, 5, dtype=torch.long)
    query = torch.ones(1, 4, dtype=torch.long)

    stats = model({"context_input_ids": context, "query_input_ids": query})["inner_loop_stats"]

    assert torch.equal(stats["energy_delta_reg_loss"], torch.tensor(0.0))
    assert torch.equal(stats["energy_delta_exceed_fraction"], torch.tensor(0.0))


def test_energy_delta_penalty_propagates_gradients_to_energy_model():
    torch.manual_seed(0)
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="identity",
        energy_delta_reg=1.0,
        energy_delta_max=0.0,
    )
    backend = model.memory_backend_impl
    memory_state, _ = backend.init_memory_state(1)
    energy_state = model._prepare_energy_state(None, memory_state["mem_batch"])
    context = [torch.randint(1, 101, (1, 5))]
    query = torch.randint(1, 101, (1, 4))

    _, _, _, _, _, stats = model._write_segments(
        context,
        query,
        memory_state,
        energy_state,
    )
    delta_reg_loss = stats["_energy_delta_reg_loss"]
    gradients = torch.autograd.grad(delta_reg_loss, tuple(model.energy_head.parameters()))

    assert delta_reg_loss > 0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_energy_pretrain_optimizer_has_no_adamw_weight_decay(monkeypatch):
    model = _model(K=1)
    model.energy_pretrain_steps = 1
    seen_weight_decay = None

    class RecordingAdamW(torch.optim.SGD):
        def __init__(self, params, lr, weight_decay=0.01):
            nonlocal seen_weight_decay
            seen_weight_decay = weight_decay
            super().__init__(params, lr=lr, weight_decay=weight_decay)

    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)

    model.pretrain_energy_objective()

    assert seen_weight_decay == 0.0


def test_next_token_future_embeddings_are_appended():
    model = _model(K=1, energy_future_mode="next_token")
    segment = torch.tensor([[3, 4, 2, 0]])
    ctx_hidden = torch.zeros(1, 4, 48)
    mask = segment.ne(model.model.config.pad_token_id)

    energy_input = model._energy_input(ctx_hidden, segment, mask)
    token_emb = model.model.get_input_embeddings()(segment)

    assert energy_input.shape == (1, 4, 96)
    assert torch.allclose(energy_input[:, :, :48], ctx_hidden)
    assert torch.allclose(energy_input[:, 0, 48:], token_emb[:, 1, :])
    assert torch.allclose(energy_input[:, 1, 48:], token_emb[:, 2, :])
    assert torch.equal(energy_input[:, 2:, 48:], torch.zeros_like(energy_input[:, 2:, 48:]))


def test_next_token_future_embeddings_skip_padding():
    model = _model(K=1, energy_future_mode="next_token")
    segment = torch.tensor([[3, 0, 4, 2]])
    ctx_hidden = torch.zeros(1, 4, 48)
    mask = segment.ne(model.model.config.pad_token_id)

    energy_input = model._energy_input(ctx_hidden, segment, mask)
    token_emb = model.model.get_input_embeddings()(segment)

    assert torch.equal(energy_input[:, 0, 48:], torch.zeros_like(energy_input[:, 0, 48:]))
    assert torch.allclose(energy_input[:, 2, 48:], token_emb[:, 3, :])


def test_second_order_steps_are_global():
    model = _model(K=4, energy_future_mode="none")
    model.grad_mode = "second"
    model.last_K_second_order = 1

    create_graph_flags = [model._inner_grad_options(step, total_steps=12)[0] for step in range(12)]

    assert create_graph_flags == [False] * 11 + [True]


def test_trailing_padding_does_not_consume_second_order_steps(monkeypatch):
    torch.manual_seed(0)
    model = _model(
        K=2,
        energy_future_mode="none",
        energy_model_type="identity",
        last_K_second_order=1,
    )
    model.train()
    seen_options = []
    original_inner_grad_options = model._inner_grad_options

    def recording_inner_grad_options(global_step, total_steps, **kwargs):
        options = original_inner_grad_options(global_step, total_steps, **kwargs)
        seen_options.append((global_step, total_steps, options[0]))
        return options

    monkeypatch.setattr(model, "_inner_grad_options", recording_inner_grad_options)
    segments = [
        torch.randint(1, 101, (2, 5)),
        torch.zeros(2, 5, dtype=torch.long),
    ]
    query = torch.randint(1, 101, (2, 4))
    labels = torch.randint(1, 101, (2, 4))

    output = model(
        {"context_input_ids": segments, "query_input_ids": query},
        labels=labels,
    )
    output["loss"].backward()

    assert seen_options == [(0, 2, False), (1, 2, True)]
    assert all(parameter.grad is not None for parameter in model.energy_head.parameters())


def test_sequential_adam_bias_correction_uses_global_write_step(monkeypatch):
    model = _model(
        K=2,
        energy_future_mode="none",
        energy_model_type="identity",
        use_adam=True,
    )
    model.eval()
    seen_step_indices = []

    def recording_adam_step(param, grad, state, step_idx, lr):
        del state
        seen_step_indices.append(step_idx)
        return param - lr * grad

    monkeypatch.setattr(model, "_adam_step", recording_adam_step)
    segments = [
        torch.randint(1, 101, (2, 5)),
        torch.randint(1, 101, (2, 5)),
    ]
    query = torch.randint(1, 101, (2, 4))

    model({"context_input_ids": segments, "query_input_ids": query})

    assert seen_step_indices == [1, 2, 3, 4]


def test_rejects_unknown_inner_objective():
    with pytest.raises(ValueError, match="inner_objective"):
        EnergyGradMemConfig(base_config=_base_config(), inner_objective="other")


def test_lstm_inner_objective_is_deprecated_alias_for_neural():
    with pytest.warns(DeprecationWarning, match="inner_objective='lstm' is deprecated"):
        config = EnergyGradMemConfig(base_config=_base_config(), inner_objective="lstm")

    assert config.inner_objective == "neural"


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            {"inner_objective": "cross_entropy"},
            "requires inner_objective='neural'",
        ),
        (
            {"segment_write_mode": "parallel", "energy_future_mode": "none"},
            "requires segment_write_mode='sequential'",
        ),
        (
            {"memory_backend": "lora"},
            "requires memory_backend='prefix'",
        ),
        (
            {"energy_pretrain_steps": 1},
            "does not support energy pretraining",
        ),
    ],
)
def test_segment_delta_gru_rejects_unsupported_configurations(kwargs, error):
    with pytest.raises(ValueError, match=error):
        EnergyGradMemConfig(
            base_config=_base_config(),
            energy_model_type="segment_delta_gru",
            **kwargs,
        )


def test_segment_delta_gru_defaults_state_size_and_rejects_invalid_size():
    model = _model(
        K=1,
        energy_future_mode="none",
        energy_model_type="segment_delta_gru",
    )
    assert model.energy_segment_state_size == 48

    with pytest.raises(ValueError, match="positive integer"):
        _model(
            K=1,
            energy_future_mode="none",
            energy_model_type="segment_delta_gru",
            energy_segment_state_size=0,
        )


def test_rejects_fixed_embedding_objective_without_next_token_target():
    with pytest.raises(ValueError, match="energy_future_mode='next_token'"):
        EnergyGradMemConfig(
            base_config=_base_config(),
            inner_objective="embedding_l1",
            energy_future_mode="none",
        )


def test_rejects_parallel_segments_without_prefix_backend():
    with pytest.raises(ValueError, match="memory_backend='prefix'"):
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend="lora",
            segment_write_mode="parallel",
            energy_future_mode="none",
        )


def test_rejects_parallel_segments_with_future_embeddings():
    with pytest.raises(ValueError, match="energy_future_mode='none'"):
        EnergyGradMemConfig(
            base_config=_base_config(),
            segment_write_mode="parallel",
            energy_future_mode="next_token",
        )


def test_rejects_parallel_segments_with_energy_ce_guidance():
    with pytest.raises(ValueError, match="energy_ce_guidance"):
        EnergyGradMemConfig(
            base_config=_base_config(),
            segment_write_mode="parallel",
            energy_future_mode="none",
            energy_ce_guidance=True,
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"memory_rotation": "other"}, "memory_rotation"),
        (
            {"memory_rotation": "none", "memory_rotation_angle": 0.1},
            "requires memory_rotation='pairwise'",
        ),
        (
            {"memory_backend": "lora", "memory_rotation": "pairwise"},
            "memory_backend='prefix'",
        ),
        (
            {
                "memory_rotation": "pairwise",
                "segment_write_mode": "parallel",
                "energy_future_mode": "none",
            },
            "segment_write_mode='sequential'",
        ),
        (
            {"memory_rotation": "pairwise", "use_adam": True},
            "use_adam=False",
        ),
    ],
)
def test_rejects_invalid_memory_rotation_config(kwargs, error):
    with pytest.raises(ValueError, match=error):
        EnergyGradMemConfig(base_config=_base_config(), **kwargs)


def test_rejects_pretraining_for_fixed_embedding_objective():
    model = _model(K=1, inner_objective="embedding_l2")
    model.energy_pretrain_steps = 1

    with pytest.raises(ValueError, match="energy pretraining"):
        model.pretrain_energy_objective()


def test_rejects_unknown_future_mode():
    with pytest.raises(ValueError, match="energy_future_mode"):
        EnergyGradMemConfig(base_config=_base_config(), energy_future_mode="suffix")


def test_rejects_unknown_pretrain_objective():
    with pytest.raises(ValueError, match="energy_pretrain_objective"):
        EnergyGradMemConfig(base_config=_base_config(), energy_pretrain_objective="bad")


def test_rejects_write_head_for_energy_objective():
    with pytest.raises(ValueError, match="use_write_head"):
        EnergyGradMemConfig(base_config=_base_config(), use_write_head=True)


def test_set_energy_trainable_toggles_only_energy_params():
    model = _model(K=1)

    model.set_energy_trainable(False)

    assert all(not p.requires_grad for p in model.energy_encoder.parameters())
    assert all(not p.requires_grad for p in model.energy_head.parameters())
    assert model.mem.requires_grad

    model.set_energy_trainable(True)

    assert all(p.requires_grad for p in model.energy_encoder.parameters())
    assert all(p.requires_grad for p in model.energy_head.parameters())
    assert model.mem.requires_grad


def test_energy_freeze_callback_zeroes_grads_until_threshold():
    model = _model(K=1)
    callback = EnergyFreezeCallback(freezed_steps=2)
    args = SimpleNamespace()
    control = SimpleNamespace()
    state = SimpleNamespace(global_step=0)

    callback.on_train_begin(args, state, control, model=model)
    assert all(p.requires_grad for p in model.energy_encoder.parameters())
    assert all(p.requires_grad for p in model.energy_head.parameters())

    for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters()):
        param.grad = torch.ones_like(param)
    callback.on_pre_optimizer_step(args, state, control, model=model)
    assert all(
        torch.count_nonzero(param.grad).item() == 0
        for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters())
    )

    state.global_step = 1
    for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters()):
        param.grad = torch.ones_like(param)
    callback.on_pre_optimizer_step(args, state, control, model=model)
    assert all(
        torch.count_nonzero(param.grad).item() == 0
        for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters())
    )

    state.global_step = 2
    callback.on_step_begin(args, state, control, model=model)
    for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters()):
        param.grad = torch.ones_like(param)
    callback.on_pre_optimizer_step(args, state, control, model=model)
    assert all(
        torch.all(param.grad == 1)
        for param in list(model.energy_encoder.parameters()) + list(model.energy_head.parameters())
    )
