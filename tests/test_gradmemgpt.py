from contextlib import nullcontext

import pytest
import torch
from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig

from grad_memgpt import GradMemGPT, GradMemGPTConfig, get_backbone


def _build_base_config(model_family: str):
    vocab_size = 101
    if model_family == "gpt2":
        return GPT2Config(
            vocab_size=vocab_size,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_positions=32,
            n_ctx=32,
        )
    if model_family == "gpt_neox":
        return GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=32,
        )
    if model_family == "llama":
        return LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
    raise ValueError(f"Unknown model_family={model_family}")


def _backend_extra_kwargs(memory_backend: str):
    if memory_backend == "lora":
        return {
            "lora_mem_placement": "between_layers",
            "lora_mem_layers": "all",
            "lora_mem_r": 4,
            "lora_mem_alpha": 8,
        }
    return {}


def _assert_return_mem_payload(memory_backend: str, output: dict):
    if memory_backend == "prefix":
        assert "mem" in output
    elif memory_backend == "lora":
        assert "lora_mem" in output
    elif memory_backend == "kv_cache":
        assert "kv_mem" in output
    else:
        raise ValueError(f"Unknown backend={memory_backend}")


def _snapshot_memory_tensors(model: GradMemGPT, memory_backend: str, memory_state: dict):
    if memory_backend == "prefix":
        return [memory_state["mem_batch"].detach().clone()]
    if memory_backend == "lora":
        tensors = []
        for slot_id in model.lora_mem_slot_ids:
            a_batch, b_batch = memory_state["lora_mem"][slot_id]
            tensors.append(a_batch.detach().clone())
            tensors.append(b_batch.detach().clone())
        return tensors
    if memory_backend == "kv_cache":
        tensors = []
        for layer_idx in model.kv_mem_layer_ids:
            k_batch, v_batch = memory_state["kv_mem"][layer_idx]
            tensors.append(k_batch.detach().clone())
            tensors.append(v_batch.detach().clone())
        return tensors
    raise ValueError(f"Unknown backend={memory_backend}")


def _memory_init_params(model: GradMemGPT, memory_backend: str):
    if memory_backend == "prefix":
        return [model.mem]
    if memory_backend == "lora":
        params = []
        for key in sorted(model.lora_mem_A0.keys()):
            params.append(model.lora_mem_A0[key])
            params.append(model.lora_mem_B0[key])
        return params
    if memory_backend == "kv_cache":
        params = []
        for key in sorted(model.kv_mem_K0.keys(), key=int):
            params.append(model.kv_mem_K0[key])
            params.append(model.kv_mem_V0[key])
        return params
    raise ValueError(f"Unknown backend={memory_backend}")


@pytest.mark.forward
@pytest.mark.all
@pytest.mark.parametrize("model_family", ["gpt2", "gpt_neox", "llama"])
@pytest.mark.parametrize("memory_backend", ["prefix", "lora", "kv_cache"])
def test_forward(model_family: str, memory_backend: str):
    torch.manual_seed(0)

    base_config = _build_base_config(model_family)
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend=memory_backend,
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        **_backend_extra_kwargs(memory_backend),
    )

    model = GradMemGPT(model_config)
    model.eval()

    backend = model.memory_backend_impl
    memory_snapshots = []
    original_init_memory_state = backend.init_memory_state
    original_assign_inner_params = backend.assign_inner_params

    def _init_memory_state_with_snapshot(batch_size):
        memory_state, memory_state_initial = original_init_memory_state(batch_size)
        memory_snapshots.append(_snapshot_memory_tensors(model, memory_backend, memory_state))
        return memory_state, memory_state_initial

    def _assign_inner_params_with_snapshot(memory_state, new_params):
        original_assign_inner_params(memory_state, new_params)
        memory_snapshots.append(_snapshot_memory_tensors(model, memory_backend, memory_state))

    backend.init_memory_state = _init_memory_state_with_snapshot
    backend.assign_inner_params = _assign_inner_params_with_snapshot

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
        return_mem=True,
    )

    assert "loss" in output
    assert "predictions" in output
    assert "inner_loop_stats" in output
    assert torch.isfinite(output["loss"]).item()

    predictions = output["predictions"]
    expected_pred_len = qry_len + 1 if memory_backend == "prefix" else qry_len
    assert predictions.shape == (batch_size, expected_pred_len, vocab_size)

    stats = output["inner_loop_stats"]
    assert torch.isfinite(stats["inner_grad_norm_mean"]).item()
    assert torch.isfinite(stats["delta_mem_norm_mean"]).item()
    assert torch.isfinite(stats["inner_loss"]).item()
    assert torch.isfinite(stats["inner_reconstruction_loss"]).item()
    assert torch.isfinite(stats["inner_reconstruction_loss_after_write"]).item()
    assert torch.isfinite(stats["inner_loss_after_write"]).item()
    assert "inner_energy_loss" not in stats
    assert "inner_energy_loss_after_write" not in stats
    assert stats["inner_grad_norm_mean"].item() >= 0
    assert stats["delta_mem_norm_mean"].item() >= 0

    _assert_return_mem_payload(memory_backend, output)

    if memory_backend in ("prefix", "kv_cache"):
        assert "mem_attn_read" in stats
        mem_attn_read = stats["mem_attn_read"].item()
        assert 0.0 <= mem_attn_read <= 1.0
    else:
        assert "mem_attn_read" not in stats

    assert len(memory_snapshots) == model_config.K + 1
    for step_idx in range(1, len(memory_snapshots)):
        prev_step = memory_snapshots[step_idx - 1]
        curr_step = memory_snapshots[step_idx]
        assert len(prev_step) == len(curr_step)
        total_delta = 0.0
        for prev_t, curr_t in zip(prev_step, curr_step):
            total_delta += float((curr_t - prev_t).pow(2).sum().item())
        assert total_delta > 0.0, (
            f"Memory did not change across inner steps for backend={memory_backend}, "
            f"step={step_idx - 1}->{step_idx}"
        )


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("model_family", ["gpt2", "gpt_neox", "llama"])
@pytest.mark.parametrize("memory_backend", ["prefix", "lora", "kv_cache"])
def test_single_batch_train(model_family: str, memory_backend: str):
    torch.manual_seed(0)

    base_config = _build_base_config(model_family)
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend=memory_backend,
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        **_backend_extra_kwargs(memory_backend),
    )

    model = GradMemGPT(model_config)
    model.train()

    mem_init_params = _memory_init_params(model, memory_backend)
    assert len(mem_init_params) > 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    optimizer.zero_grad(set_to_none=True)
    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )
    loss = output["loss"]
    assert torch.isfinite(loss).item()

    loss.backward()

    for p in mem_init_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all().item()
        assert p.grad.detach().norm().item() > 0.0

    grad_norm_sq = torch.tensor(0.0)
    params_with_grad = []
    before_by_param = {}
    for p in trainable_params:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item()
            grad_norm_sq = grad_norm_sq + p.grad.detach().pow(2).sum()
            params_with_grad.append(p)
            before_by_param[id(p)] = p.detach().clone()
    assert grad_norm_sq.sqrt().item() > 0.0

    optimizer.step()
    changed = any(not torch.equal(before_by_param[id(p)], p.detach()) for p in params_with_grad)
    assert changed


@pytest.mark.forward
@pytest.mark.all
def test_fail_fast_on_inner_alignment_mismatch():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="lora",
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        **_backend_extra_kwargs("lora"),
    )

    model = GradMemGPT(model_config)
    model.eval()

    original_build_write_inputs = model.memory_backend_impl.build_write_inputs

    def _broken_build_write_inputs(memory_state, batch_ctx):
        batch = original_build_write_inputs(memory_state, batch_ctx)
        batch["label_shift"] = 0
        return batch

    model.memory_backend_impl.build_write_inputs = _broken_build_write_inputs

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    with pytest.raises(ValueError, match="Invalid inner-loop alignment"):
        _ = model(
            {
                "context_input_ids": context_input_ids,
                "query_input_ids": query_input_ids,
            },
            labels=labels,
        )


@pytest.mark.forward
@pytest.mark.all
@pytest.mark.parametrize("model_family", ["gpt2", "gpt_neox", "llama"])
def test_forward_lora_target_modules_auto(model_family: str):
    torch.manual_seed(0)

    base_config = _build_base_config(model_family)
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="lora",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        lora_mem_placement="target_modules",
        lora_mem_target_modules=None,
        lora_mem_layers="all",
        lora_mem_r=4,
        lora_mem_alpha=8,
    )

    model = GradMemGPT(model_config)
    model.eval()
    assert all(not k.startswith("slot_") for k in model.lora_mem_A0.keys())
    assert all(k.startswith("layer") for k in model.lora_mem_A0.keys())

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
        return_mem=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (batch_size, qry_len, vocab_size)
    assert "lora_mem" in output


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("model_family", ["gpt2", "gpt_neox", "llama"])
def test_single_batch_train_lora_target_modules_auto(model_family: str):
    torch.manual_seed(0)

    base_config = _build_base_config(model_family)
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="lora",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        lora_mem_placement="target_modules",
        lora_mem_target_modules=None,
        lora_mem_layers="all",
        lora_mem_r=4,
        lora_mem_alpha=8,
    )

    model = GradMemGPT(model_config)
    model.train()

    mem_init_params = _memory_init_params(model, "lora")
    assert len(mem_init_params) > 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    optimizer.zero_grad(set_to_none=True)
    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )
    loss = output["loss"]
    assert torch.isfinite(loss).item()

    loss.backward()
    for p in mem_init_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all().item()
        assert p.grad.detach().norm().item() > 0.0


@pytest.mark.forward
@pytest.mark.all
def test_fail_fast_lora_target_modules_no_match():
    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="lora",
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        lora_mem_placement="target_modules",
        lora_mem_target_modules="definitely_not_existing_target",
        lora_mem_layers="all",
        lora_mem_r=4,
        lora_mem_alpha=8,
    )

    with pytest.raises(ValueError, match="resolved zero modules"):
        _ = GradMemGPT(model_config)


@pytest.mark.forward
@pytest.mark.all
def test_fail_fast_lora_target_modules_non_linear_not_supported():
    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="lora",
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        lora_mem_placement="target_modules",
        lora_mem_target_modules="attn",
        lora_mem_layers="all",
        lora_mem_r=4,
        lora_mem_alpha=8,
    )

    with pytest.raises(ValueError, match="supports only linear-like modules"):
        _ = GradMemGPT(model_config)


@pytest.mark.forward
@pytest.mark.all
def test_lora_target_module_delta_matches_linear_formula():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="lora",
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        lora_mem_placement="target_modules",
        lora_mem_target_modules="c_fc",
        lora_mem_layers="all",
        lora_mem_r=4,
        lora_mem_alpha=8,
        lora_mem_dropout=0.0,
    )

    model = GradMemGPT(model_config)
    model.eval()

    assert len(model.lora_mem_slot_ids) > 0
    slot_id = model.lora_mem_slot_ids[0]
    module = dict(get_backbone(model.model).named_modules())[slot_id]

    memory_state, _ = model.memory_backend_impl.init_memory_state(batch_size=1)
    A_mem, B_mem = memory_state["lora_mem"][slot_id]

    x = torch.randn(1, 3, A_mem.size(1))

    with torch.no_grad():
        y_base = module(x)

    with torch.no_grad():
        with model._enable_lora_memory(memory_state["lora_mem"]):
            y_with_mem = module(x)

    low_rank = torch.einsum("bsi,bir->bsr", x, A_mem)
    delta = torch.einsum("bsr,bro->bso", low_rank, B_mem)
    y_expected = y_base + model.lora_mem_scale * delta

    assert torch.allclose(y_with_mem, y_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.forward
@pytest.mark.all
def test_forward_prefix_energy_objective():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy",
        energy_rank_weight=0.1,
        energy_traj_weight=0.01,
    )

    model = GradMemGPT(model_config)
    model.eval()

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
        return_mem=True,
    )

    assert torch.isfinite(output["loss"]).item()
    assert output["predictions"].shape == (batch_size, qry_len + 1, vocab_size)
    stats = output["inner_loop_stats"]
    assert "inner_loss_after_write" in stats
    assert "inner_loss_initial" in stats
    assert "inner_loss_write_delta" in stats
    assert "inner_reconstruction_loss" in stats
    assert "inner_energy_loss" in stats
    assert "inner_reconstruction_loss_after_write" in stats
    assert "inner_energy_loss_after_write" in stats
    assert "energy_rank_loss" in stats
    assert "energy_traj_loss" in stats
    assert torch.isfinite(stats["inner_loss_after_write"]).item()
    assert torch.isfinite(stats["inner_loss_write_delta"]).item()
    assert "mem" in output


@pytest.mark.one_batch_train
@pytest.mark.all
def test_single_batch_train_prefix_energy_objective():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy",
        energy_rank_weight=0.1,
        energy_traj_weight=0.01,
    )

    model = GradMemGPT(model_config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    optimizer.zero_grad(set_to_none=True)
    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )
    output["loss"].backward()

    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()
    energy_grads = [p.grad for p in model.energy_head.parameters()]
    energy_grads = [g for g in energy_grads if g is not None]
    assert len(energy_grads) > 0
    assert all(torch.isfinite(g).all().item() for g in energy_grads)
    assert model.energy_ln.weight.grad is not None
    assert torch.isfinite(model.energy_ln.weight.grad).all().item()


@pytest.mark.forward
@pytest.mark.all
def test_layerwise_energy_sums_transformer_layer_energies():
    torch.manual_seed(0)
    base_config = _build_base_config("gpt2")
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=1,
        grad_mode="none",
        write_objective="energy",
        use_layerwise_energy=True,
    ))

    batch_size = 2
    sequence_length = 7
    context_start = 3
    hidden_states = tuple(
        torch.randn(batch_size, sequence_length, base_config.n_embd)
        for _ in range(base_config.n_layer)
    )
    write_batch = {
        "context_start": context_start,
        "mask": torch.ones(batch_size, sequence_length - context_start),
    }

    energy = model._compute_write_energy(hidden_states, write_batch)
    expected = []
    for layer_hidden, layer_norm, energy_head in zip(hidden_states, model.energy_ln, model.energy_head):
        context_hidden = layer_hidden[:, context_start:, :]
        token_energy = torch.nn.functional.softplus(energy_head(layer_norm(context_hidden))).squeeze(-1)
        expected.append(token_energy.mean(dim=1))

    assert len(model.energy_head) == base_config.n_layer
    assert torch.allclose(energy, torch.stack(expected).sum(dim=0))


@pytest.mark.forward
@pytest.mark.all
def test_forward_prefix_energy_with_reconstruction_objective():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy_with_reconstruction",
        write_reconstruction_weight=1.0,
        write_energy_weight=0.1,
        energy_rank_weight=0.1,
        energy_traj_weight=0.01,
    )

    model = GradMemGPT(model_config)
    model.eval()

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )

    assert torch.isfinite(output["loss"]).item()
    stats = output["inner_loop_stats"]
    for key in [
        "inner_reconstruction_loss",
        "inner_energy_loss",
        "inner_reconstruction_loss_after_write",
        "inner_energy_loss_after_write",
        "inner_loss_after_write",
        "inner_loss_initial",
        "inner_loss_write_delta",
        "write_reconstruction_weight",
        "write_energy_weight",
        "energy_rank_loss",
        "energy_traj_loss",
    ]:
        assert key in stats
        assert torch.isfinite(stats[key]).item()


@pytest.mark.forward
@pytest.mark.all
def test_energy_with_reconstruction_uses_shared_write_forward():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy_with_reconstruction",
        write_reconstruction_weight=1.0,
        write_energy_weight=0.1,
    )

    model = GradMemGPT(model_config)
    model.eval()

    call_counts = {
        "combined": 0,
        "reconstruction": 0,
        "energy": 0,
    }
    original_combined = model._run_energy_reconstruction_write_forward
    original_reconstruction = model._run_reconstruction_write_forward
    original_energy = model._run_energy_write_forward

    def _count_combined(write_batch):
        call_counts["combined"] += 1
        return original_combined(write_batch)

    def _count_reconstruction(write_batch):
        call_counts["reconstruction"] += 1
        return original_reconstruction(write_batch)

    def _count_energy(write_batch):
        call_counts["energy"] += 1
        return original_energy(write_batch)

    model._run_energy_reconstruction_write_forward = _count_combined
    model._run_reconstruction_write_forward = _count_reconstruction
    model._run_energy_write_forward = _count_energy

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )

    assert torch.isfinite(output["loss"]).item()
    assert call_counts["combined"] == model_config.K + 1
    assert call_counts["reconstruction"] == 0
    assert call_counts["energy"] == 0


@pytest.mark.one_batch_train
@pytest.mark.all
def test_single_batch_train_prefix_energy_with_reconstruction_objective():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy_with_reconstruction",
        write_reconstruction_weight=1.0,
        write_energy_weight=0.1,
    )

    model = GradMemGPT(model_config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    batch_size = 2
    ctx_len = 6
    qry_len = 4
    vocab_size = base_config.vocab_size

    context_input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len))
    query_input_ids = torch.randint(0, vocab_size, (batch_size, qry_len))
    labels = torch.randint(0, vocab_size, (batch_size, qry_len))

    optimizer.zero_grad(set_to_none=True)
    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )
    output["loss"].backward()

    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()
    energy_grads = [p.grad for p in model.energy_head.parameters()]
    energy_grads = [g for g in energy_grads if g is not None]
    assert len(energy_grads) > 0
    assert all(torch.isfinite(g).all().item() for g in energy_grads)


@pytest.mark.forward
@pytest.mark.all
def test_prefix_energy_write_handles_padded_contexts():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    base_config.pad_token_id = 0
    base_config.eos_token_id = 0
    model_config = GradMemGPTConfig(
        base_config=base_config,
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
        write_objective="energy",
    )

    model = GradMemGPT(model_config)
    model.eval()

    context_input_ids = torch.tensor([
        [5, 6, 7, 8, 9, 10],
        [11, 12, 13, 0, 0, 0],
    ])
    query_input_ids = torch.randint(1, base_config.vocab_size, (2, 4))
    labels = torch.randint(1, base_config.vocab_size, (2, 4))

    output = model(
        {
            "context_input_ids": context_input_ids,
            "query_input_ids": query_input_ids,
        },
        labels=labels,
    )

    assert torch.isfinite(output["loss"]).item()
    assert torch.isfinite(output["inner_loop_stats"]["inner_loss_after_write"]).item()


@pytest.mark.forward
@pytest.mark.all
def test_energy_objective_rejects_unsupported_backend():
    base_config = _build_base_config("gpt2")

    for write_objective in ["energy", "energy_with_reconstruction"]:
        with pytest.raises(ValueError, match="supported only for memory_backend='prefix'"):
            _ = GradMemGPTConfig(
                base_config=base_config,
                memory_backend="lora",
                n_mem_tokens=4,
                K=1,
                lr=0.01,
                use_adam=False,
                grad_mode="none",
                use_mem_proj=False,
                mem_proj_mode="none",
                use_write_head=False,
                attn_implementation="eager",
                write_objective=write_objective,
                **_backend_extra_kwargs("lora"),
            )


@pytest.mark.forward
@pytest.mark.all
def test_unknown_write_objective_rejects():
    base_config = _build_base_config("gpt2")

    with pytest.raises(AssertionError, match="write_objective must be one of"):
        _ = GradMemGPTConfig(
            base_config=base_config,
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            lr=0.01,
            use_adam=False,
            grad_mode="none",
            use_mem_proj=False,
            mem_proj_mode="none",
            use_write_head=False,
            attn_implementation="eager",
            write_objective="definitely_not_valid",
        )


@pytest.mark.forward
@pytest.mark.all
def test_energy_negative_memory_geometry():
    torch.manual_seed(0)
    positive = torch.randn(5, 4, 8, requires_grad=True)
    initial = torch.randn(5, 4, 8)
    alpha = 0.75

    negatives, permutation = GradMemGPT._build_energy_negative_memories(
        positive, initial, alpha
    )

    assert permutation is not None
    assert torch.all(permutation != torch.arange(positive.size(0)))
    assert torch.allclose(negatives["deranged"], positive.detach()[permutation])
    assert torch.allclose(
        negatives["interpolated"],
        alpha * positive.detach() + (1.0 - alpha) * negatives["deranged"],
    )
    expected_radius = (positive.detach() - initial).reshape(positive.size(0), -1).norm(dim=1)
    random_radius = (negatives["random"] - initial).reshape(positive.size(0), -1).norm(dim=1)
    assert torch.allclose(random_radius, expected_radius, atol=1e-5, rtol=1e-5)
    assert all(not value.requires_grad for value in negatives.values())

    singleton = torch.randn(1, 4, 8, requires_grad=True)
    singleton_negatives, singleton_permutation = GradMemGPT._build_energy_negative_memories(
        singleton, singleton.detach(), alpha
    )
    assert singleton_permutation is None
    assert singleton_negatives["deranged"] is None
    assert singleton_negatives["interpolated"] is None
    assert torch.allclose(singleton_negatives["random"], singleton.detach())
    assert torch.isfinite(singleton_negatives["random"]).all().item()


@pytest.mark.forward
@pytest.mark.all
def test_energy_memory_search_candidate_geometry():
    torch.manual_seed(0)
    model, _, _ = _build_shaped_energy_model(
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_num_samples=5,
        energy_memory_search_radius_scale=0.25,
    )
    memory = torch.randn(3, 4, 8, requires_grad=True)
    previous = memory.detach() + 0.1 * torch.randn_like(memory)

    candidates = model._sample_norm_preserving_memory_candidates(memory, previous)

    assert candidates.shape == (5, 3, 4, 8)
    assert not candidates.requires_grad
    expected_norm = memory.detach().flatten(1).norm(dim=1)
    candidate_norm = candidates.flatten(2).norm(dim=2)
    assert torch.allclose(candidate_norm, expected_norm.unsqueeze(0), atol=1e-5, rtol=1e-5)
    expected_radius = 0.25 * (memory.detach() - previous).flatten(1).norm(dim=1)
    candidate_radius = (candidates - memory.detach().unsqueeze(0)).flatten(2).norm(dim=2)
    assert torch.allclose(candidate_radius, expected_radius.unsqueeze(0), atol=1e-5, rtol=1e-5)

    unchanged = model._sample_norm_preserving_memory_candidates(memory, memory.detach())
    assert torch.allclose(unchanged, memory.detach().unsqueeze(0).expand_as(unchanged))
    zero = torch.zeros_like(memory)
    zero_candidates = model._sample_norm_preserving_memory_candidates(zero, previous)
    assert torch.equal(zero_candidates, zero.unsqueeze(0).expand_as(zero_candidates))
    assert torch.isfinite(zero_candidates).all().item()


@pytest.mark.forward
@pytest.mark.all
def test_energy_memory_search_selects_best_candidate_per_example():
    class FakeBackend:
        @staticmethod
        def build_read_inputs(memory_state, _batch_ctx):
            return {
                "inputs_embeds": memory_state["mem_batch"],
                "logits_start": 0,
                "pred_len": 2,
                "label_shift": 0,
            }

        @staticmethod
        def activation_context(_memory_state):
            return nullcontext()

    class FakeReadModel(torch.nn.Module):
        def forward(self, inputs_embeds, return_dict=True, **_kwargs):
            del return_dict
            logits = inputs_embeds.new_zeros(inputs_embeds.size(0), 2, 2)
            logits[:, 0, 0] = inputs_embeds[:, 0, 0]
            return type("FakeOutput", (), {"logits": logits})()

    model, _, _ = _build_shaped_energy_model(
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_num_samples=2,
    )
    original_read_model = model.model
    model.model = FakeReadModel()
    student = torch.zeros(2, 1, 1, requires_grad=True)
    perturbed = torch.tensor([
        [[[4.0]], [[-4.0]]],
        [[[-4.0]], [[4.0]]],
    ])
    model._sample_norm_preserving_memory_candidates = lambda *_args: perturbed
    try:
        result = model._compute_energy_memory_search_loss(
            FakeBackend(),
            {"mem_batch": student},
            torch.zeros_like(student),
            {},
            torch.zeros(2, 1, dtype=torch.long),
        )
    finally:
        model.model = original_read_model

    assert result["loss"].item() == pytest.approx(8.0)
    assert result["improvement_rate"].item() == pytest.approx(1.0)
    assert result["target_gain"].item() > 0.0
    assert result["max_target_gain"].item() >= result["target_gain"].item()
    assert result["per_example_loss"].shape == (2,)
    assert result["relative_target_gain"].shape == (2,)
    assert torch.all(result["relative_target_gain"] > 0.0)
    result["loss"].backward()
    assert student.grad is not None
    assert torch.all(student.grad < 0.0)


@pytest.mark.forward
@pytest.mark.all
def test_energy_memory_search_gain_weight_ema_uses_positive_relative_gains():
    model, _, _ = _build_shaped_energy_model(
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_use_gain_weighting=True,
        energy_memory_search_gain_ema_decay=0.99,
    )
    first_gains = torch.tensor([[0.0, 0.1], [0.2, 0.3]])
    first_weights = model._compute_energy_memory_search_gain_weights(first_gains)
    assert model.energy_memory_search_gain_ema_initialized.item()
    assert model.energy_memory_search_gain_ema.item() == pytest.approx(0.2)
    assert torch.allclose(first_weights, torch.tensor([[0.0, 0.5], [1.0, 1.5]]))
    assert first_weights[first_gains > 0.0].mean().item() == pytest.approx(1.0)

    second_gains = torch.tensor([[0.4, 0.0]])
    second_weights = model._compute_energy_memory_search_gain_weights(second_gains)
    expected_ema = 0.99 * 0.2 + 0.01 * 0.4
    assert model.energy_memory_search_gain_ema.item() == pytest.approx(expected_ema)
    assert second_weights[0, 0].item() == pytest.approx(0.4 / expected_ema)
    ema_before_zero_batch = model.energy_memory_search_gain_ema.clone()
    zero_weights = model._compute_energy_memory_search_gain_weights(torch.zeros(2, 2))
    assert torch.equal(zero_weights, torch.zeros_like(zero_weights))
    assert torch.equal(model.energy_memory_search_gain_ema, ema_before_zero_batch)


def _build_shaped_energy_model(batch_size=2, **config_overrides):
    base_config = _build_base_config("gpt2")
    config_values = dict(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy",
        energy_rank_weight=0.1,
        energy_traj_weight=0.01,
        energy_anchor_weight=0.001,
        energy_rank_temperature=1.0,
        energy_mix_alpha=0.75,
        add_inner_loss_to_outer=True,
        inner_loss_weight=0.2,
    )
    config_values.update(config_overrides)
    model_config = GradMemGPTConfig(**config_values)
    model = GradMemGPT(model_config)
    context = torch.randint(0, base_config.vocab_size, (batch_size, 6))
    query = torch.randint(0, base_config.vocab_size, (batch_size, 4))
    labels = torch.randint(0, base_config.vocab_size, (batch_size, 4))
    inputs = {"context_input_ids": context, "query_input_ids": query}
    return model, inputs, labels


@pytest.mark.one_batch_train
@pytest.mark.all
def test_energy_shaping_loss_accounting_and_backward():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(batch_size=2)
    model.train()
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    raw_keys = [
        "outer_loss",
        "target_loss",
        "energy_rank_loss",
        "energy_rank_deranged_loss",
        "energy_rank_interpolated_loss",
        "energy_rank_random_loss",
        "energy_traj_loss",
        "energy_anchor_loss",
    ]
    for key in raw_keys + ["energy_aux_loss"]:
        assert key in stats
        assert torch.isfinite(stats[key]).item()

    expected_rank = torch.stack([
        stats["energy_rank_deranged_loss"],
        stats["energy_rank_interpolated_loss"],
        stats["energy_rank_random_loss"],
    ]).mean()
    assert torch.allclose(stats["energy_rank_loss"], expected_rank)

    expected_aux = (
        model.energy_rank_weight * stats["energy_rank_loss"]
        + model.energy_traj_weight * stats["energy_traj_loss"]
        + model.energy_anchor_weight * stats["energy_anchor_loss"]
    )
    expected_outer = (
        stats["target_loss"]
        + model.inner_loss_weight * stats["inner_loss"]
        + expected_aux
    )
    assert torch.allclose(stats["energy_aux_loss"], expected_aux)
    assert torch.allclose(stats["outer_loss"], expected_outer)
    assert torch.allclose(output["loss"].detach(), expected_outer)

    output["loss"].backward()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)


@pytest.mark.one_batch_train
@pytest.mark.all
def test_energy_memory_search_runs_after_each_write_step_and_is_training_only():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_weight=0.2,
        energy_memory_search_num_samples=2,
        energy_memory_search_radius_scale=0.25,
        add_inner_loss_to_outer=False,
        memory_alignment_weight=0.0,
        step_alignment_weight=0.0,
        intermediate_read_weight=0.0,
    )
    calls = []

    def _fixed_search(_backend, memory_state, previous_memory, _batch_ctx, _labels):
        calls.append((memory_state["mem_batch"], previous_memory))
        value = memory_state["mem_batch"].float().square().mean()
        return {
            "loss": value,
            "per_example_loss": value.expand(memory_state["mem_batch"].size(0)),
            "relative_target_gain": value.new_tensor([0.1, 0.2]),
            "improvement_rate": value.new_tensor(0.5),
            "target_gain": value.new_tensor(0.25),
            "max_target_gain": value.new_tensor(0.25 + 0.125 * len(calls)),
            "selected_distance": value.new_tensor(0.125),
        }

    model._compute_energy_memory_search_loss = _fixed_search
    model.train()
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert len(calls) == model.K
    assert calls[0][1].grad_fn is None
    assert torch.equal(calls[1][1], calls[0][0].detach())
    expected_search_loss = torch.stack([
        state[0].float().square().mean() for state in calls
    ]).mean()
    assert torch.allclose(stats["energy_memory_search_loss"], expected_search_loss.detach())
    assert stats["energy_memory_search_improvement_rate"].item() == pytest.approx(0.5)
    assert stats["energy_memory_search_target_gain"].item() == pytest.approx(0.25)
    assert stats["energy_memory_search_max_target_gain"].item() == pytest.approx(0.5)
    assert stats["energy_memory_search_selected_distance"].item() == pytest.approx(0.125)
    assert torch.allclose(
        stats["energy_aux_loss"],
        model.energy_memory_search_weight * stats["energy_memory_search_loss"],
    )
    assert torch.allclose(
        stats["outer_loss"],
        stats["target_loss"] + model.energy_memory_search_weight * stats["energy_memory_search_loss"],
    )
    output["loss"].backward()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)

    def _unexpected_search(*_args, **_kwargs):
        raise AssertionError("memory search must be disabled during evaluation")

    model._compute_energy_memory_search_loss = _unexpected_search
    model.eval()
    evaluation = model(inputs, labels=labels)
    assert "energy_memory_search_loss" not in evaluation["inner_loop_stats"]


@pytest.mark.one_batch_train
@pytest.mark.all
def test_energy_memory_search_applies_gain_weights_across_all_write_states():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_weight=0.2,
        energy_memory_search_use_gain_weighting=True,
        energy_memory_search_gain_ema_decay=0.99,
        add_inner_loss_to_outer=False,
        memory_alignment_weight=0.0,
        step_alignment_weight=0.0,
        intermediate_read_weight=0.0,
    )
    calls = 0

    def _fixed_search(_backend, memory_state, _previous_memory, _batch_ctx, _labels):
        nonlocal calls
        calls += 1
        graph_anchor = memory_state["mem_batch"].float().flatten(1).sum(dim=1) * 0.0
        if calls == 1:
            per_example_loss = graph_anchor + graph_anchor.new_tensor([1.0, 2.0])
            relative_gain = graph_anchor.new_tensor([0.1, 0.2])
        else:
            per_example_loss = graph_anchor + graph_anchor.new_tensor([3.0, 4.0])
            relative_gain = graph_anchor.new_tensor([0.0, 0.3])
        return {
            "loss": per_example_loss.mean(),
            "per_example_loss": per_example_loss,
            "relative_target_gain": relative_gain,
            "improvement_rate": relative_gain.gt(0.0).float().mean(),
            "target_gain": relative_gain.mean(),
            "max_target_gain": relative_gain.max(),
            "selected_distance": per_example_loss.new_tensor(0.125),
        }

    model._compute_energy_memory_search_loss = _fixed_search
    model.train()
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert calls == model.K
    assert model.energy_memory_search_gain_ema.item() == pytest.approx(0.2)
    assert stats["energy_memory_search_unweighted_loss"].item() == pytest.approx(2.5)
    assert stats["energy_memory_search_loss"].item() == pytest.approx(2.125)
    assert stats["energy_memory_search_relative_target_gain"].item() == pytest.approx(0.15)
    assert stats["energy_memory_search_mean_relative_target_gain"].item() == pytest.approx(0.15)
    assert stats["energy_memory_search_max_relative_target_gain"].item() == pytest.approx(0.3)
    assert stats["energy_memory_search_gain_weight_mean"].item() == pytest.approx(1.0)
    assert stats["energy_memory_search_gain_weight_max"].item() == pytest.approx(1.5)
    assert stats["energy_memory_search_gain_ema"].item() == pytest.approx(0.2)
    assert torch.allclose(
        stats["outer_loss"],
        stats["target_loss"] + model.energy_memory_search_weight * stats["energy_memory_search_loss"],
    )


@pytest.mark.one_batch_train
@pytest.mark.all
def test_energy_memory_search_can_roll_best_candidate_into_next_write_step():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_weight=0.2,
        energy_memory_search_use_best_for_next_step=True,
        add_inner_loss_to_outer=False,
        memory_alignment_weight=0.0,
        step_alignment_weight=0.0,
        intermediate_read_weight=0.0,
    )
    backend = model.memory_backend_impl
    original_build_write_inputs = backend.build_write_inputs
    write_memories = []
    search_students = []
    search_previous = []
    search_teachers = []

    def _record_build_write_inputs(memory_state, batch_ctx):
        write_memories.append(memory_state["mem_batch"].detach().clone())
        return original_build_write_inputs(memory_state, batch_ctx)

    def _fixed_search(_backend, memory_state, previous_memory, _batch_ctx, _labels):
        student = memory_state["mem_batch"]
        teacher = student.detach() + 0.01
        search_students.append(student.detach().clone())
        search_previous.append(previous_memory.detach().clone())
        search_teachers.append(teacher.clone())
        per_example_loss = 0.5 * (student.float() - teacher.float()).square().flatten(1).sum(dim=1)
        batch_size = student.size(0)
        return {
            "loss": per_example_loss.mean(),
            "per_example_loss": per_example_loss,
            "relative_target_gain": student.new_full((batch_size,), 0.1),
            "teacher_memory": teacher,
            "improvement_rate": student.new_tensor(1.0),
            "target_gain": student.new_tensor(0.1),
            "max_target_gain": student.new_tensor(0.1),
            "selected_distance": (student.detach() - teacher).flatten(1).norm(dim=1).mean(),
        }

    backend.build_write_inputs = _record_build_write_inputs
    model._compute_energy_memory_search_loss = _fixed_search
    model.train()
    output = model(inputs, labels=labels, return_mem=True)

    assert len(search_students) == model.K
    assert len(write_memories) == model.K + 1  # K inner forwards plus the post-WRITE diagnostic.
    assert torch.allclose(write_memories[1], search_teachers[0])
    assert torch.allclose(search_previous[1], search_teachers[0])
    assert torch.allclose(write_memories[-1], search_teachers[-1])
    assert torch.allclose(output["mem"].detach(), search_teachers[-1])
    assert output["mem"].grad_fn is not None
    output["loss"].backward()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)


@pytest.mark.one_batch_train
@pytest.mark.all
def test_disabled_energy_shaping_preserves_target_only_loss():
    torch.manual_seed(0)
    base_config = _build_base_config("gpt2")
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        attn_implementation="eager",
        write_objective="energy",
    ))
    model.train()
    context = torch.randint(0, base_config.vocab_size, (2, 6))
    query = torch.randint(0, base_config.vocab_size, (2, 4))
    labels = torch.randint(0, base_config.vocab_size, (2, 4))

    def _unexpected_landscape_call(*_args, **_kwargs):
        raise AssertionError("disabled shaping must not evaluate landscape candidates")

    model._compute_energy_landscape_losses = _unexpected_landscape_call
    model._compute_energy_memory_search_loss = _unexpected_landscape_call
    output = model(
        {"context_input_ids": context, "query_input_ids": query},
        labels=labels,
    )
    stats = output["inner_loop_stats"]
    assert torch.allclose(output["loss"].detach(), stats["target_loss"])
    assert torch.allclose(stats["outer_loss"], stats["target_loss"])
    for key in (
        "energy_aux_loss",
        "energy_rank_loss",
        "energy_rank_deranged_loss",
        "energy_rank_interpolated_loss",
        "energy_rank_random_loss",
        "energy_traj_loss",
        "energy_anchor_loss",
        "energy_positive_mean",
        "energy_negative_deranged_mean",
        "energy_margin_random_mean",
        "energy_margin_violation_deranged",
    ):
        assert key not in stats
    output["loss"].backward()


@pytest.mark.one_batch_train
@pytest.mark.all
def test_anchor_only_uses_detached_positive_without_negative_candidates():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.001,
    )
    model.train()
    candidate_calls = 0

    def _unexpected_negative_call(*_args, **_kwargs):
        raise AssertionError("anchor-only mode must not construct negative memories")

    def _fixed_positive_energy(_backend, _memory_template, _batch_ctx, mem_batch):
        nonlocal candidate_calls
        candidate_calls += 1
        assert not mem_batch.requires_grad
        return mem_batch.new_tensor([2.0, -3.0])

    model._build_energy_negative_memories = _unexpected_negative_call
    model._run_energy_memory_candidate = _fixed_positive_energy
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert candidate_calls == 1
    assert stats["energy_positive_mean"].item() == pytest.approx(-0.5)
    assert stats["energy_anchor_loss"].item() == pytest.approx(6.5)
    assert torch.allclose(
        stats["energy_aux_loss"],
        model.energy_anchor_weight * stats["energy_anchor_loss"],
    )
    for key in (
        "energy_rank_loss",
        "energy_rank_deranged_loss",
        "energy_negative_random_mean",
        "energy_margin_deranged_mean",
        "energy_margin_violation_interpolated",
        "energy_traj_loss",
    ):
        assert key not in stats
    output["loss"].backward()


@pytest.mark.one_batch_train
@pytest.mark.all
def test_energy_landscape_candidates_detach_memory_but_train_energy():
    torch.manual_seed(0)
    model, inputs, _ = _build_shaped_energy_model(batch_size=2)
    model.train()
    backend = model.memory_backend_impl
    memory_state, memory_state_initial = backend.init_memory_state(batch_size=2)
    candidate_mem = (
        memory_state["mem_batch"].detach()
        + 0.1 * torch.randn_like(memory_state["mem_batch"])
    )
    candidate_mem.requires_grad_(True)
    memory_state["mem_batch"] = candidate_mem
    batch_ctx = backend.prepare_batch(
        inputs["context_input_ids"],
        inputs["query_input_ids"],
        model.model.config.pad_token_id,
    )

    landscape = model._compute_energy_landscape_losses(
        backend,
        memory_state,
        memory_state_initial,
        batch_ctx,
        compute_rank=True,
        compute_anchor=True,
    )
    expected_anchor = torch.stack([
        landscape["energy_positive"].pow(2).mean(),
        *[
            energy.pow(2).mean()
            for energy in landscape["negative_energies"].values()
        ],
    ]).mean()
    assert torch.allclose(landscape["anchor_loss"], expected_anchor)
    (landscape["rank_loss"] + landscape["anchor_loss"]).backward()

    assert candidate_mem.grad is None
    energy_grads = [parameter.grad for parameter in model.energy_head.parameters()]
    assert any(gradient is not None for gradient in energy_grads)
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_grads if gradient is not None)


@pytest.mark.forward
@pytest.mark.all
def test_energy_shaping_parameter_validation():
    base_config = _build_base_config("gpt2")
    with pytest.raises(ValueError, match="energy_rank_temperature"):
        GradMemGPTConfig(base_config=base_config, energy_rank_temperature=0.0)
    with pytest.raises(ValueError, match="energy_mix_alpha"):
        GradMemGPTConfig(base_config=base_config, energy_mix_alpha=1.1)


@pytest.mark.forward
@pytest.mark.all
def test_energy_memory_search_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    defaults = GradMemGPTConfig(base_config=base_config)
    assert defaults.energy_memory_search_weight == 0.0
    assert defaults.energy_memory_search_num_samples == 4
    assert defaults.energy_memory_search_radius_scale == pytest.approx(0.25)
    assert defaults.energy_memory_search_use_gain_weighting is False
    assert defaults.energy_memory_search_gain_ema_decay == pytest.approx(0.99)
    assert defaults.energy_memory_search_use_best_for_next_step is False

    invalid_overrides = [
        {"energy_memory_search_weight": -0.1},
        {"energy_memory_search_num_samples": 0},
        {"energy_memory_search_radius_scale": 0.0},
        {"energy_memory_search_gain_ema_decay": -0.1},
        {"energy_memory_search_gain_ema_decay": 1.0},
        {"energy_memory_search_use_best_for_next_step": True},
        {"energy_memory_search_weight": 0.1, "write_objective": "reconstruction"},
        {
            "energy_memory_search_weight": 0.1,
            "write_objective": "energy",
            "memory_backend": "prefix",
            "K": 0,
            "grad_mode": "second",
        },
        {
            "energy_memory_search_weight": 0.1,
            "write_objective": "energy",
            "memory_backend": "prefix",
            "K": 2,
            "grad_mode": "first",
        },
        {
            "energy_memory_search_weight": 0.1,
            "write_objective": "energy",
            "memory_backend": "prefix",
            "K": 2,
            "grad_mode": "second",
            "last_K_second_order": 1,
        },
        {
            "energy_memory_search_weight": 0.1,
            "write_objective": "energy",
            "memory_backend": "prefix",
            "K": 2,
            "grad_mode": "second",
            "use_adam": True,
        },
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError, match="energy_memory_search"):
            GradMemGPTConfig(base_config=base_config, **overrides)

    config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=2,
        grad_mode="second",
        use_adam=False,
        write_objective="energy",
        energy_memory_search_weight=0.1,
        energy_memory_search_num_samples=7,
        energy_memory_search_radius_scale=0.4,
        energy_memory_search_use_gain_weighting=True,
        energy_memory_search_gain_ema_decay=0.95,
        energy_memory_search_use_best_for_next_step=True,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.energy_memory_search_weight == pytest.approx(0.1)
    assert restored.energy_memory_search_num_samples == 7
    assert restored.energy_memory_search_radius_scale == pytest.approx(0.4)
    assert restored.energy_memory_search_use_gain_weighting is True
    assert restored.energy_memory_search_gain_ema_decay == pytest.approx(0.95)
    assert restored.energy_memory_search_use_best_for_next_step is True

    model = GradMemGPT(config)
    model.energy_memory_search_gain_ema.fill_(0.3)
    model.energy_memory_search_gain_ema_initialized.fill_(True)
    state_dict = model.state_dict()
    assert state_dict["energy_memory_search_gain_ema"].item() == pytest.approx(0.3)
    assert state_dict["energy_memory_search_gain_ema_initialized"].item()


def _build_memory_alignment_model(**config_overrides):
    base_config = _build_base_config("gpt2")
    config_values = dict(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_mem_proj=True,
        mem_proj_mode="per_sample",
        use_write_head=False,
        attn_implementation="eager",
        write_objective="energy",
        memory_alignment_weight=0.3,
    )
    config_values.update(config_overrides)
    model = GradMemGPT(GradMemGPTConfig(**config_values))
    inputs = {
        "context_input_ids": torch.randint(0, base_config.vocab_size, (2, 6)),
        "query_input_ids": torch.randint(0, base_config.vocab_size, (2, 4)),
    }
    labels = torch.randint(0, base_config.vocab_size, (2, 4))
    return model, inputs, labels


@pytest.mark.one_batch_train
@pytest.mark.all
def test_memory_alignment_loss_formula_and_backward():
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model()
    model.train()

    output = model(inputs, labels=labels, return_mem=True)
    target_loss = torch.nn.functional.cross_entropy(
        output["predictions"][:, :-1].reshape(-1, output["predictions"].size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    final_memory_outer_grad = torch.autograd.grad(
        target_loss,
        output["mem"],
        retain_graph=True,
    )[0].detach()
    initial_memory = model.mem.unsqueeze(0).expand_as(output["mem"])
    write_direction = (initial_memory - output["mem"]).float().flatten(1)
    expected_cosine = torch.nn.functional.cosine_similarity(
        write_direction,
        final_memory_outer_grad.float().flatten(1),
        dim=1,
    )
    expected_alignment_loss = 1 - expected_cosine.mean()

    stats = output["inner_loop_stats"]
    assert torch.allclose(
        stats["memory_alignment_cosine"],
        expected_cosine.detach().mean(),
    )
    assert torch.allclose(stats["memory_alignment_loss"], expected_alignment_loss.detach())
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.memory_alignment_weight * stats["memory_alignment_loss"],
    )

    output["loss"].backward()
    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()
    assert model.mem_proj.weight.grad is not None
    assert torch.isfinite(model.mem_proj.weight.grad).all().item()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)
    assert any(gradient.norm().item() > 0.0 for gradient in energy_gradients)


@pytest.mark.forward
@pytest.mark.all
def test_memory_alignment_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    invalid_overrides = [
        {"memory_alignment_weight": -0.1},
        {"memory_alignment_weight": 0.1, "memory_backend": "lora"},
        {"memory_alignment_weight": 0.1, "grad_mode": "first"},
        {"memory_alignment_weight": 0.1, "K": 0},
        {"memory_alignment_weight": 0.1, "K": 2, "last_K_second_order": 0},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError, match="memory_alignment_weight"):
            GradMemGPTConfig(base_config=base_config, **overrides)

    config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=2,
        grad_mode="second",
        memory_alignment_weight=0.25,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.memory_alignment_weight == pytest.approx(0.25)


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("grad_align_norm", ["none", "norm"])
def test_step_alignment_uses_each_inner_update_and_resulting_memory_gradient(grad_align_norm):
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        memory_alignment_weight=0.0,
        step_alignment_weight=0.4,
        grad_align_norm=grad_align_norm,
        intermediate_read_weight=0.0,
    )
    model.train()
    recorded_losses = []
    recorded_cosines = []
    original_step_alignment = model._compute_step_alignment

    def _record_step_alignment(target_loss, memory_state, inner_update):
        loss, cosine = original_step_alignment(target_loss, memory_state, inner_update)
        outer_grad = torch.autograd.grad(
            target_loss,
            memory_state["mem_batch"],
            retain_graph=True,
        )[0].detach()
        expected_cosine = torch.nn.functional.cosine_similarity(
            inner_update.float().flatten(1),
            outer_grad.float().flatten(1),
            dim=1,
        )
        expected_loss = 1 - expected_cosine.mean()
        if grad_align_norm == "norm":
            expected_loss = (1 - outer_grad.float().flatten(1).norm(dim=1) * expected_cosine).mean()
        assert torch.allclose(cosine, expected_cosine.mean())
        assert torch.allclose(loss, expected_loss)
        recorded_losses.append(loss)
        recorded_cosines.append(cosine)
        return loss, cosine

    model._compute_step_alignment = _record_step_alignment
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert len(recorded_losses) == model.K + 1
    expected_loss = torch.stack(recorded_losses).mean()
    expected_cosine = torch.stack(recorded_cosines).mean()
    assert torch.allclose(stats["step_alignment_loss"], expected_loss.detach())
    assert torch.allclose(stats["step_alignment_cosine"], expected_cosine.detach())
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.step_alignment_weight * stats["step_alignment_loss"],
    )

    output["loss"].backward()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)
    assert any(gradient.norm().item() > 0.0 for gradient in energy_gradients)


@pytest.mark.forward
@pytest.mark.all
def test_step_alignment_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    assert GradMemGPTConfig(base_config=base_config).step_alignment_weight == 0.0
    assert GradMemGPTConfig(base_config=base_config).grad_align_norm == "none"
    invalid_overrides = [
        {"step_alignment_weight": -0.1},
        {"step_alignment_weight": 0.1, "memory_backend": "lora"},
        {"step_alignment_weight": 0.1, "grad_mode": "first"},
        {"step_alignment_weight": 0.1, "K": 0},
        {"step_alignment_weight": 0.1, "K": 2, "last_K_second_order": 0},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError, match="step_alignment_weight"):
            GradMemGPTConfig(base_config=base_config, **overrides)
    with pytest.raises(ValueError, match="grad_align_norm"):
        GradMemGPTConfig(base_config=base_config, grad_align_norm="invalid")

    config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=2,
        grad_mode="second",
        step_alignment_weight=0.25,
        grad_align_norm="norm",
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.step_alignment_weight == pytest.approx(0.25)
    assert restored.grad_align_norm == "norm"


@pytest.mark.one_batch_train
@pytest.mark.all
def test_trainer_supports_memory_alignment_during_train_and_no_grad_eval(tmp_path):
    from transformers import TrainingArguments
    from run_gradmemgpt_on_kv_retrieval import CustomTrainer

    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        use_mem_proj=False,
        mem_proj_mode="none",
    )
    trainer = CustomTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
    )

    model.train()
    train_loss = trainer.compute_loss(model, {"input_ids": inputs, "labels": labels})
    trainer.log({"loss": train_loss.detach().item()})
    logged = trainer.state.log_history[-1]
    for key in ("memory_alignment_loss", "memory_alignment_cosine"):
        assert key in logged
        assert torch.isfinite(torch.tensor(logged[key])).item()

    model.eval()
    with torch.no_grad():
        output = model(inputs, labels=labels)
    assert torch.isfinite(output["loss"]).item()
    for key in ("memory_alignment_loss", "memory_alignment_cosine"):
        assert key in output["inner_loop_stats"]
        assert torch.isfinite(output["inner_loop_stats"][key]).item()


@pytest.mark.one_batch_train
@pytest.mark.all
def test_orthogonal_loss_formula_alpha_initialization_and_backward():
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        memory_alignment_weight=0.0,
        orthogonal_loss_weight=0.7,
    )
    model.train()
    assert torch.allclose(
        torch.nn.functional.softplus(model.orthogonal_alpha_raw.detach()),
        torch.tensor(model.lr),
    )

    output = model(inputs, labels=labels, return_mem=True)
    target_loss = torch.nn.functional.cross_entropy(
        output["predictions"][:, :-1].reshape(-1, output["predictions"].size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    stopped_outer_grad = torch.autograd.grad(
        target_loss,
        output["mem"],
        retain_graph=True,
    )[0].detach().float()
    initial_memory = model.mem.unsqueeze(0).expand_as(output["mem"])
    cumulative_inner_update = (initial_memory - output["mem"]).float()
    alpha = torch.nn.functional.softplus(model.orthogonal_alpha_raw.float())
    residual_dot = (
        (cumulative_inner_update - alpha * stopped_outer_grad) * stopped_outer_grad
    ).flatten(1).sum(dim=1)
    expected_loss = residual_dot.square().mean()

    stats = output["inner_loop_stats"]
    assert torch.allclose(stats["orthogonal_residual_dot"], residual_dot.detach().mean())
    assert torch.allclose(stats["orthogonal_alpha"], alpha.detach())
    assert torch.allclose(stats["orthogonal_loss"], expected_loss.detach())
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.orthogonal_loss_weight * stats["orthogonal_loss"],
    )

    output["loss"].backward()
    assert model.orthogonal_alpha_raw.grad is not None
    assert torch.isfinite(model.orthogonal_alpha_raw.grad).item()
    assert model.orthogonal_alpha_raw.grad.abs().item() > 0.0


@pytest.mark.forward
@pytest.mark.all
def test_orthogonal_loss_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    for disabled_weight in (None, 0.0):
        config = GradMemGPTConfig(base_config=base_config, orthogonal_loss_weight=disabled_weight)
        model = GradMemGPT(config)
        assert config.orthogonal_loss_weight == 0.0
        assert model.orthogonal_alpha_raw is not None
        assert torch.allclose(
            torch.nn.functional.softplus(model.orthogonal_alpha_raw.detach()),
            torch.tensor(model.lr),
        )

    invalid_overrides = [
        {"orthogonal_loss_weight": -0.1},
        {"orthogonal_loss_weight": 0.1, "memory_backend": "lora"},
        {"orthogonal_loss_weight": 0.1, "grad_mode": "first"},
        {"orthogonal_loss_weight": 0.1, "K": 0},
        {"orthogonal_loss_weight": 0.1, "K": 2, "last_K_second_order": 0},
        {"orthogonal_loss_weight": 0.1, "lr": 0.0},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError, match="orthogonal_loss_weight"):
            GradMemGPTConfig(base_config=base_config, **overrides)

    config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=2,
        grad_mode="second",
        orthogonal_loss_weight=0.25,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.orthogonal_loss_weight == pytest.approx(0.25)


@pytest.mark.one_batch_train
@pytest.mark.all
def test_ivan_loss_formula_and_backward():
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        use_mem_proj=False,
        mem_proj_mode="none",
        memory_alignment_weight=0.0,
        ivan_loss_weight=0.7,
    )
    model.train()
    inner_grads = []
    original_sgd_step = model._sgd_step

    def _record_sgd_step(parameter, gradient, **kwargs):
        inner_grads.append(gradient)
        return original_sgd_step(parameter, gradient, **kwargs)

    model._sgd_step = _record_sgd_step
    output = model(inputs, labels=labels, return_mem=True)
    target_loss = torch.nn.functional.cross_entropy(
        output["predictions"][:, :-1].reshape(-1, output["predictions"].size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    outer_grad = torch.autograd.grad(target_loss, output["mem"], retain_graph=True)[0].float()
    inner_grad_sum = torch.stack(inner_grads).sum(dim=0).float()
    residual_dot = ((inner_grad_sum - outer_grad) * outer_grad).flatten(1).sum(dim=1)
    expected_loss = residual_dot.square().mean()

    stats = output["inner_loop_stats"]
    assert len(inner_grads) == model.K
    assert torch.allclose(stats["ivan_residual_dot"], residual_dot.detach().mean())
    assert torch.allclose(stats["ivan_loss"], expected_loss.detach())
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.ivan_loss_weight * stats["ivan_loss"],
    )

    output["loss"].backward()
    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)


@pytest.mark.forward
@pytest.mark.all
def test_ivan_loss_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    for disabled_weight in (None, 0.0):
        config = GradMemGPTConfig(base_config=base_config, ivan_loss_weight=disabled_weight)
        model = GradMemGPT(config)
        assert config.ivan_loss_weight == 0.0
        assert model.ivan_loss_weight == 0.0

    invalid_overrides = [
        {"ivan_loss_weight": -0.1},
        {"ivan_loss_weight": 0.1, "grad_mode": "first"},
        {"ivan_loss_weight": 0.1, "K": 0},
        {"ivan_loss_weight": 0.1, "K": 2, "last_K_second_order": 1},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError, match="ivan_loss_weight"):
            GradMemGPTConfig(base_config=base_config, **overrides)

    config = GradMemGPTConfig(
        base_config=base_config,
        K=2,
        grad_mode="second",
        ivan_loss_weight=0.25,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.ivan_loss_weight == pytest.approx(0.25)


@pytest.mark.one_batch_train
@pytest.mark.all
def test_zero_weight_orthogonal_loss_trains_only_alpha():
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        memory_alignment_weight=0.0,
        orthogonal_loss_weight=0.0,
    )
    model.train()

    output = model(inputs, labels=labels)
    target_loss = torch.nn.functional.cross_entropy(
        output["predictions"][:, :-1].reshape(-1, output["predictions"].size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    target_mem_grad = torch.autograd.grad(target_loss, model.mem, retain_graph=True)[0]
    outer_mem_grad = torch.autograd.grad(output["loss"], model.mem, retain_graph=True)[0]
    alpha_grad = torch.autograd.grad(output["loss"], model.orthogonal_alpha_raw)[0]

    assert torch.allclose(output["loss"].detach(), target_loss.detach())
    assert torch.allclose(outer_mem_grad, target_mem_grad)
    assert torch.isfinite(alpha_grad).item()
    assert alpha_grad.abs().item() > 0.0


@pytest.mark.one_batch_train
@pytest.mark.all
def test_intermediate_read_uses_reverse_harmonic_weights_and_backpropagates():
    torch.manual_seed(0)
    base_config = _build_base_config("gpt2")
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=4,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        attn_implementation="eager",
        intermediate_read_weight=0.4,
    ))
    model.train()
    inputs = {
        "context_input_ids": torch.randint(0, base_config.vocab_size, (2, 6)),
        "query_input_ids": torch.randint(0, base_config.vocab_size, (2, 4)),
    }
    labels = torch.randint(0, base_config.vocab_size, (2, 4))

    recorded_read_losses = []
    original_compute_read_target_loss = model._compute_read_target_loss

    def _record_read_loss(predictions, read_batch, target_labels):
        loss = original_compute_read_target_loss(predictions, read_batch, target_labels)
        recorded_read_losses.append(loss)
        return loss

    model._compute_read_target_loss = _record_read_loss
    output = model(inputs, labels=labels)

    assert len(recorded_read_losses) == model.K
    expected_weights = torch.tensor([1 / 4, 1 / 3, 1 / 2], dtype=recorded_read_losses[0].dtype)
    expected_weights = expected_weights / expected_weights.sum()
    assert torch.allclose(
        model._intermediate_read_weights(model.K, expected_weights.device, expected_weights.dtype),
        expected_weights,
    )
    expected_intermediate_loss = (
        expected_weights * torch.stack(recorded_read_losses[1:])
    ).sum()
    stats = output["inner_loop_stats"]
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.intermediate_read_weight * expected_intermediate_loss.detach(),
    )
    assert "intermediate_read_loss" in stats

    output["loss"].backward()
    assert model.mem.grad is not None
    assert torch.isfinite(model.mem.grad).all().item()
    assert model.mem.grad.norm().item() > 0.0


@pytest.mark.one_batch_train
@pytest.mark.all
def test_intermediate_read_includes_full_depth_auxiliary_objective():
    torch.manual_seed(0)
    base_config = _build_base_config("gpt2")
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=3,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        attn_implementation="eager",
        write_objective="energy",
        energy_rank_weight=0.1,
        energy_traj_weight=0.01,
        energy_anchor_weight=0.001,
        add_inner_loss_to_outer=True,
        inner_loss_weight=0.2,
        memory_alignment_weight=0.3,
        orthogonal_loss_weight=0.4,
        intermediate_read_weight=0.5,
    ))
    model.train()
    inputs = {
        "context_input_ids": torch.randint(0, base_config.vocab_size, (2, 6)),
        "query_input_ids": torch.randint(0, base_config.vocab_size, (2, 4)),
    }
    labels = torch.randint(0, base_config.vocab_size, (2, 4))

    landscape_calls = 0
    original_landscape = model._compute_energy_landscape_losses
    recorded_read_losses = []
    original_read_loss = model._compute_read_target_loss
    recorded_aux_losses = []
    original_combine_aux = model._combine_depth_auxiliary_losses

    def _count_landscape(*args, **kwargs):
        nonlocal landscape_calls
        landscape_calls += 1
        return original_landscape(*args, **kwargs)

    def _record_read_loss(*args, **kwargs):
        loss = original_read_loss(*args, **kwargs)
        recorded_read_losses.append(loss)
        return loss

    def _record_aux_loss(*args, **kwargs):
        loss = original_combine_aux(*args, **kwargs)
        expected = (
            model.inner_loss_weight * kwargs["inner_loss"]
            + model.energy_rank_weight * kwargs["rank_loss"]
            + model.energy_traj_weight * kwargs["trajectory_loss"]
            + model.energy_anchor_weight * kwargs["anchor_loss"]
            + model.memory_alignment_weight * kwargs["memory_alignment_loss"]
            + model.orthogonal_loss_weight * kwargs["orthogonal_loss"]
        )
        assert torch.allclose(loss, expected)
        recorded_aux_losses.append(loss)
        return loss

    model._compute_energy_landscape_losses = _count_landscape
    model._compute_read_target_loss = _record_read_loss
    model._combine_depth_auxiliary_losses = _record_aux_loss
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert landscape_calls == model.K
    assert len(recorded_read_losses) == model.K
    assert len(recorded_aux_losses) == model.K
    intermediate_weights = model._intermediate_read_weights(
        model.K,
        device=recorded_read_losses[0].device,
        dtype=recorded_read_losses[0].dtype,
    )
    expected_intermediate_loss = (
        intermediate_weights
        * (torch.stack(recorded_read_losses[1:]) + torch.stack(recorded_aux_losses[:-1]))
    ).sum()
    expected_outer_loss = (
        recorded_read_losses[0]
        + recorded_aux_losses[-1]
        + model.intermediate_read_weight * expected_intermediate_loss
    )
    assert torch.allclose(output["loss"], expected_outer_loss)
    assert "intermediate_read_loss" in stats

    output["loss"].backward()
    assert model.orthogonal_alpha_raw.grad is not None
    assert torch.isfinite(model.orthogonal_alpha_raw.grad).item()


@pytest.mark.forward
@pytest.mark.all
def test_intermediate_read_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    default_config = GradMemGPTConfig(base_config=base_config)
    assert default_config.intermediate_read_weight == 0.0
    with pytest.raises(ValueError, match="intermediate_read_weight"):
        GradMemGPTConfig(base_config=base_config, intermediate_read_weight=-0.1)

    config = GradMemGPTConfig(base_config=base_config, intermediate_read_weight=0.25)
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.intermediate_read_weight == pytest.approx(0.25)


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("shaping_active", [False, True])
def test_trainer_and_eval_metrics_preserve_active_only_schema(tmp_path, shaping_active):
    from transformers import EvalPrediction, TrainingArguments
    from run_gradmemgpt_on_kv_retrieval import (
        CustomTrainer,
        TRAIN_COMPONENT_KEYS,
        compute_metrics_fn,
    )

    class _Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(_tokens, skip_special_tokens=True):
            return ""

    torch.manual_seed(0)
    overrides = {} if shaping_active else {
        "energy_rank_weight": 0.0,
        "energy_traj_weight": 0.0,
        "energy_anchor_weight": 0.0,
        "add_inner_loss_to_outer": False,
    }
    model, inputs, labels = _build_shaped_energy_model(batch_size=2, **overrides)

    trainer = CustomTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
    )
    model.train()
    train_loss = trainer.compute_loss(model, {"input_ids": inputs, "labels": labels})
    trainer.log({"loss": train_loss.detach().item()})
    logged = trainer.state.log_history[-1]

    model.eval()
    with torch.no_grad():
        output = model(inputs, labels=labels)

    stats = {
        key: value.detach().cpu().numpy()
        for key, value in output["inner_loop_stats"].items()
    }
    eval_prediction = EvalPrediction(
        predictions=(output["predictions"].argmax(dim=-1).cpu().numpy(), stats),
        label_ids=labels.cpu().numpy(),
        inputs={key: value.cpu().numpy() for key, value in inputs.items()},
    )
    metrics = compute_metrics_fn(eval_prediction, [], _Tokenizer())

    assert logged["outer_loss"] == pytest.approx(train_loss.detach().item(), rel=1e-5)
    assert metrics["outer_loss"] == pytest.approx(output["loss"].item(), rel=1e-5)
    for exported in (logged, metrics):
        for key in ("outer_loss", "target_loss"):
            assert key in exported
            assert torch.isfinite(torch.tensor(exported[key])).item()

    diagnostic_loss_keys = {
        "memory_alignment_loss",
        "memory_alignment_cosine",
        "step_alignment_loss",
        "step_alignment_cosine",
        "orthogonal_loss",
        "orthogonal_residual_dot",
        "orthogonal_alpha",
        "ivan_loss",
        "ivan_residual_dot",
        "intermediate_read_loss",
    }
    for exported in (logged, metrics):
        assert diagnostic_loss_keys <= exported.keys()
        assert all(torch.isfinite(torch.tensor(exported[key])).item() for key in diagnostic_loss_keys)
        assert not any(key.startswith("final_") for key in exported)

    active_only_keys = set(TRAIN_COMPONENT_KEYS) - {"outer_loss", "target_loss"} - diagnostic_loss_keys
    if not shaping_active:
        assert active_only_keys.isdisjoint(logged)
        assert active_only_keys.isdisjoint(metrics)
        return

    representative_active_keys = {
        "energy_aux_loss",
        "energy_rank_loss",
        "energy_traj_loss",
        "energy_anchor_loss",
        "energy_positive_mean",
        "energy_negative_deranged_mean",
        "energy_margin_random_mean",
        "energy_margin_violation_interpolated",
    }
    for exported in (logged, metrics):
        assert representative_active_keys <= exported.keys()
        assert all(
            torch.isfinite(torch.tensor(exported[key])).item()
            for key in representative_active_keys
        )
