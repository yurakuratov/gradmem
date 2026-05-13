import pytest
import torch
from types import SimpleNamespace
from transformers import GPT2Config

from energy_gradmem import EnergyGradMem, EnergyGradMemConfig


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


def _model(memory_backend="prefix", K=2, energy_future_mode="next_token"):
    return EnergyGradMem(
        EnergyGradMemConfig(
            base_config=_base_config(),
            memory_backend=memory_backend,
            n_mem_tokens=4,
            K=K,
            lr=0.01,
            use_adam=False,
            grad_mode="second",
            use_mem_proj=False,
            mem_proj_mode="none",
            use_write_head=False,
            attn_implementation="eager",
            inner_objective="lstm",
            energy_hidden_size=32,
            energy_num_layers=2,
            energy_dropout=0.0,
            energy_future_mode=energy_future_mode,
        )
    )


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


def test_energy_ce_guidance_loss_uses_detached_ce_and_mask():
    energy = torch.tensor([[1.0, 2.0, 100.0]], requires_grad=True)
    ce = torch.tensor([[3.0, 4.0, 1000.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    loss = EnergyGradMem._energy_ce_guidance_loss(energy, ce, mask)

    assert torch.allclose(loss, torch.tensor(4.0))
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
            inner_objective="lstm",
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


def test_rejects_unknown_inner_objective():
    with pytest.raises(ValueError, match="inner_objective='lstm'"):
        EnergyGradMemConfig(base_config=_base_config(), inner_objective="other")


def test_rejects_unknown_future_mode():
    with pytest.raises(ValueError, match="energy_future_mode"):
        EnergyGradMemConfig(base_config=_base_config(), energy_future_mode="suffix")


def test_rejects_write_head_for_energy_objective():
    with pytest.raises(ValueError, match="use_write_head"):
        EnergyGradMemConfig(base_config=_base_config(), use_write_head=True)
