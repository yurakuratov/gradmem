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
    assert "final_inner_loss_mean" in stats
    assert "energy_rank_loss" in stats
    assert "energy_traj_loss" in stats
    assert torch.isfinite(stats["final_inner_loss_mean"]).item()
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
    assert all(g is not None for g in energy_grads)
    assert all(torch.isfinite(g).all().item() for g in energy_grads)
    assert model.energy_ln.weight.grad is not None
    assert torch.isfinite(model.energy_ln.weight.grad).all().item()


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
    assert torch.isfinite(output["inner_loop_stats"]["final_inner_loss_mean"]).item()


@pytest.mark.forward
@pytest.mark.all
def test_energy_objective_rejects_unsupported_backend():
    base_config = _build_base_config("gpt2")

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
            write_objective="energy",
            **_backend_extra_kwargs("lora"),
        )
