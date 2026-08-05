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
    assert "inner_loss_after_write" in stats
    assert "inner_loss_initial" in stats
    assert "inner_loss_write_delta" in stats
    assert "inner_energy_loss" in stats
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


@pytest.mark.one_batch_train
@pytest.mark.all
def test_memory_alignment_loss_is_added_to_outer_objective():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        memory_alignment_weight=0.3,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        add_inner_loss_to_outer=False,
    )
    model.train()
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert torch.isfinite(stats["memory_alignment_loss"])
    assert torch.isfinite(stats["memory_alignment_cosine"])
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.memory_alignment_weight * stats["memory_alignment_loss"],
    )
    output["loss"].backward()
    assert model.mem.grad is not None


@pytest.mark.forward
@pytest.mark.all
def test_memory_alignment_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    with pytest.raises(ValueError, match="memory_alignment_weight"):
        GradMemGPTConfig(base_config=base_config, memory_alignment_weight=-0.1)
    with pytest.raises(ValueError, match="memory_alignment_weight"):
        GradMemGPTConfig(base_config=base_config, memory_alignment_weight=0.1, grad_mode="first")

    config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=1,
        grad_mode="second",
        memory_alignment_weight=0.25,
    )
    config.save_pretrained(tmp_path)
    assert GradMemGPTConfig.from_pretrained(tmp_path).memory_alignment_weight == pytest.approx(0.25)


@pytest.mark.one_batch_train
@pytest.mark.all
def test_step_alignment_loss_is_added_to_outer_objective():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        memory_alignment_weight=0.0,
        step_alignment_weight=0.4,
        add_inner_loss_to_outer=False,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
    )
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]
    assert torch.isfinite(stats["step_alignment_loss"])
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.step_alignment_weight * stats["step_alignment_loss"],
    )


@pytest.mark.one_batch_train
@pytest.mark.all
def test_intermediate_read_supervises_written_memory_states():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        K=3,
        intermediate_read_weight=0.4,
        add_inner_loss_to_outer=False,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
    )
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]
    assert torch.isfinite(stats["intermediate_read_loss"])
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.intermediate_read_weight * stats["intermediate_read_loss"],
    )


@pytest.mark.one_batch_train
@pytest.mark.all
def test_orthogonal_loss_is_added_to_outer_objective():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        orthogonal_loss_weight=0.2,
        add_inner_loss_to_outer=False,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
    )
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]
    assert torch.isfinite(stats["orthogonal_loss"])
    assert torch.allclose(
        output["loss"].detach(),
        stats["target_loss"] + model.orthogonal_loss_weight * stats["orthogonal_loss"],
    )


@pytest.mark.one_batch_train
@pytest.mark.all
def test_memory_search_matches_best_read_candidate():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        energy_memory_search_weight=0.1,
        energy_memory_search_num_samples=2,
        energy_memory_search_radius_scale=0.25,
        add_inner_loss_to_outer=False,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
    )
    model.train()
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]
    assert torch.isfinite(stats["energy_memory_search_loss"])
    assert torch.isfinite(stats["energy_memory_search_target_gain"])


@pytest.mark.forward
@pytest.mark.all
def test_memory_search_gain_weighting_uses_positive_gain_ema():
    model, _, _ = _build_shaped_energy_model(
        energy_memory_search_use_gain_weighting=True,
        energy_memory_search_gain_ema_decay=0.9,
    )
    weights = model._compute_energy_memory_search_gain_weights(torch.tensor([0.1, 0.3, 0.0]))
    assert model.energy_memory_search_gain_ema.item() == pytest.approx(0.2)
    assert torch.allclose(weights, torch.tensor([0.5, 1.5, 0.0]))


@pytest.mark.forward
@pytest.mark.all
def test_layerwise_energy_sums_layer_scores():
    base_config = _build_base_config("gpt2")
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config, memory_backend="prefix", write_objective="energy", use_layerwise_energy=True
    ))
    hidden = tuple(torch.randn(2, 7, base_config.n_embd) for _ in range(base_config.n_layer))
    energy = model._compute_write_energy(hidden, {"context_start": 3, "mask": torch.ones(2, 4)})
    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


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

    shaping_keys = set(TRAIN_COMPONENT_KEYS) - {"outer_loss", "target_loss"}
    if not shaping_active:
        assert shaping_keys.isdisjoint(logged)
        assert shaping_keys.isdisjoint(metrics)
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
