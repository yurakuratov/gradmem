from contextlib import nullcontext
import math

import pytest
import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPTNeoXConfig, LlamaConfig

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


def test_shared_writer_uses_read_model_instance():
    model = GradMemGPT(GradMemGPTConfig(base_config=_build_base_config("gpt2"), K=0))
    assert model.write_model is model.read_model


def test_write_sampling_config_defaults_and_serialization(tmp_path):
    config = GradMemGPTConfig(K=3, lr=0.4)
    assert config.K_sampling is False
    assert (config.K_min, config.K_max) == (3, 3)
    assert (config.ilr_min, config.ilr_max) == (0.4, 0.4)
    assert config.ilr_sample_unique is False
    config = GradMemGPTConfig(
        K=2, lr=0.4, K_sampling=True, K_min=1, K_max=5,
        ilr_min=0.01, ilr_max=1.0, ilr_sample_unique=True,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    for name in ("K", "lr", "K_sampling", "K_min", "K_max", "ilr_min", "ilr_max", "ilr_sample_unique"):
        assert getattr(restored, name) == getattr(config, name)


@pytest.mark.parametrize("kwargs", [
    {"K_min": 0}, {"K_min": 3, "K_max": 2}, {"K_min": 1.5},
    {"ilr_min": 0.0}, {"ilr_min": -1.0}, {"ilr_max": float("inf")},
    {"ilr_min": 1.0, "ilr_max": 0.1}, {"ilr_min": float("nan")},
])
def test_write_sampling_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError, match="K_sampling requires"):
        GradMemGPTConfig(K_sampling=True, **kwargs)


@pytest.mark.forward
@pytest.mark.all
@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("use_adam", [False, True])
def test_write_sampling_is_training_only_and_rates_are_log_uniform(monkeypatch, unique, use_adam):
    base_config = GPT2Config(
        vocab_size=31, n_embd=16, n_layer=1, n_head=2, n_positions=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config, n_mem_tokens=2, K=1, lr=0.4,
        K_sampling=True, K_min=2, K_max=3, ilr_min=0.01, ilr_max=1.0,
        ilr_sample_unique=unique, write_objective="energy", use_adam=use_adam,
    ))
    inputs = {
        "context_input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
        "query_input_ids": torch.tensor([[4, 5], [4, 5]]),
    }
    randint = torch.randint
    sampled_depths = []

    def record_randint(low, high, size, **kwargs):
        assert (low, high, size) == (2, 4, ())
        value = randint(low, high, size, **kwargs)
        sampled_depths.append(value.item())
        return value

    monkeypatch.setattr(torch, "randint", record_randint)
    rates = []
    if use_adam:
        original_step = model._adam_step

        def record_step(p, g, state, step_idx, lr):
            rates.append(lr.detach().clone())
            return original_step(p, g, state, step_idx, lr)

        monkeypatch.setattr(model, "_adam_step", record_step)
    else:
        original_step = model._sgd_step

        def record_step(p, g, **kwargs):
            rates.append(kwargs["lr"].detach().clone())
            return original_step(p, g, **kwargs)

        monkeypatch.setattr(model, "_sgd_step", record_step)

    torch.manual_seed(42)
    expected_depth = randint(2, 4, ()).item()
    expected_lr = torch.empty((2,) if unique else ()).uniform_(
        math.log(0.01), math.log(1.0),
    ).exp_()
    torch.manual_seed(42)
    output = model(inputs, return_mem=True)
    assert sampled_depths == [expected_depth]
    assert len(rates) == expected_depth
    for lr in rates:
        torch.testing.assert_close(lr.flatten(), expected_lr.flatten())
        assert lr.shape == ((2, 1, 1) if unique else ())
    assert output["inner_loop_stats"]["write_steps"].item() == expected_depth
    if unique:
        assert not torch.allclose(output["mem"][0], output["mem"][1])
    else:
        torch.testing.assert_close(output["mem"][0], output["mem"][1])
    assert (model.K, model.lr) == (1, 0.4)

    rates.clear()
    model.eval()
    output = model(inputs, return_mem=True)
    assert sampled_depths == [expected_depth]
    assert len(rates) == 1
    assert rates[0].item() == pytest.approx(0.4)
    assert output["inner_loop_stats"]["write_steps"].item() == 1
    torch.testing.assert_close(output["mem"][0], output["mem"][1])


@pytest.mark.forward
@pytest.mark.all
def test_write_sampling_preserves_second_order_shaping_and_gain_buffers():
    model = GradMemGPT(GradMemGPTConfig(
        base_config=GPT2Config(vocab_size=31, n_embd=16, n_layer=1, n_head=2, n_positions=32),
        n_mem_tokens=2, K=1, lr=0.4, K_sampling=True, K_min=3, K_max=3,
        ilr_min=0.1, ilr_max=0.1, ilr_sample_unique=True,
        write_objective="energy", energy_rank_weight=0.1, energy_traj_weight=0.1,
        energy_anchor_weight=0.1, intermediate_read_weight=0.1,
        energy_memory_search_weight=0.1, energy_memory_search_num_samples=1,
        energy_memory_search_use_gain_weighting=True,
    ))
    inputs = {
        "context_input_ids": torch.tensor([[1, 2, 3], [5, 6, 7]]),
        "query_input_ids": torch.tensor([[4, 8], [4, 9]]),
    }
    labels = torch.tensor([[-100, 8], [-100, 9]])
    output = model(inputs, labels=labels)
    assert output["inner_loop_stats"]["write_steps"].item() == 3
    assert model.energy_memory_search_gain_ema.shape == (3,)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.mem.grad is not None and torch.isfinite(model.mem.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.energy_head.parameters())
    model._compute_energy_memory_search_gain_weights(torch.ones(1, 2))


@pytest.mark.forward
@pytest.mark.all
def test_separate_writer_supports_different_hidden_size_and_energy_backward():
    writer_config = GPT2Config(
        vocab_size=101, n_embd=32, n_layer=1, n_head=4, n_positions=32, n_ctx=32,
    )
    model = GradMemGPT(GradMemGPTConfig(
        base_config=_build_base_config("gpt2"),
        separate_writer=True,
        writer_base_config=writer_config,
        memory_backend="prefix",
        n_mem_tokens=2,
        K=1,
        lr=0.01,
        grad_mode="second",
        write_objective="energy",
    ))
    assert model.write_model is not model.read_model
    assert model.write_memory_proj.in_features == 64
    assert model.write_memory_proj.out_features == 32

    input_ids = {
        "context_input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "query_input_ids": torch.tensor([[9, 10, 11], [12, 13, 14]]),
    }
    labels = torch.tensor([[-100, 15, -100], [-100, 16, -100]])
    output = model(input_ids, labels=labels, return_mem=True)
    output["loss"].backward()
    assert output["predictions"].shape == (2, 4, 101)
    assert output["mem"].shape == (2, 2, 64)
    assert next(model.read_model.parameters()).grad is not None
    assert next(model.write_model.parameters()).grad is not None


def test_separate_writer_rejects_non_prefix_memory():
    with pytest.raises(ValueError, match="supported only for memory_backend='prefix'"):
        GradMemGPTConfig(
            base_config=_build_base_config("gpt2"),
            separate_writer=True,
            writer_base_config=_build_base_config("gpt2"),
            memory_backend="lora",
        )


@pytest.mark.parametrize("option", ["use_write_lora", "use_write_head"])
def test_separate_writer_rejects_write_only_modules(option):
    with pytest.raises(ValueError, match=f"does not support {option}=True"):
        GradMemGPTConfig(
            base_config=_build_base_config("gpt2"),
            separate_writer=True,
            writer_base_config=_build_base_config("gpt2"),
            **{option: True},
        )


def test_separate_writer_checkpoint_round_trip(tmp_path):
    writer_config = GPT2Config(
        vocab_size=101, n_embd=32, n_layer=1, n_head=4, n_positions=32, n_ctx=32,
    )
    model = GradMemGPT(GradMemGPTConfig(
        base_config=_build_base_config("gpt2"),
        separate_writer=True,
        writer_base_config=writer_config,
        K=0,
    ))
    model.save_pretrained(tmp_path)
    restored = GradMemGPT.from_pretrained(tmp_path)
    assert restored.write_model is not restored.read_model
    assert restored.write_model.config.n_embd == 32
    assert restored.write_memory_proj.weight.shape == (32, 64)


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


@pytest.mark.one_batch_train
@pytest.mark.all
def test_prefix_energy_objective_supports_write_lora():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="second",
        use_write_lora=True,
        write_lora_r=4,
        write_lora_alpha=8,
        freeze_backbone=True,
        write_objective="energy",
        attn_implementation="eager",
    ))
    model.train()

    inputs = {
        "context_input_ids": torch.randint(0, base_config.vocab_size, (2, 6)),
        "query_input_ids": torch.randint(0, base_config.vocab_size, (2, 4)),
    }
    labels = torch.randint(0, base_config.vocab_size, (2, 4))
    output = model(inputs, labels=labels)

    assert torch.isfinite(output["loss"]).item()
    output["loss"].backward()
    write_lora_grads = [
        parameter.grad
        for name, parameter in model.model.named_parameters()
        if "lora_" in name and parameter.requires_grad and parameter.grad is not None
    ]
    assert write_lora_grads
    assert all(torch.isfinite(gradient).all().item() for gradient in write_lora_grads)
    assert any(gradient.norm().item() > 0.0 for gradient in write_lora_grads)


@pytest.mark.forward
@pytest.mark.all
def test_write_lora_loads_plain_backbone_checkpoint():
    torch.manual_seed(0)

    base_config = _build_base_config("gpt2")
    plain_model = AutoModelForCausalLM.from_config(base_config)
    model = GradMemGPT(GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=0,
        use_write_lora=True,
        write_lora_r=4,
        write_lora_alpha=8,
        write_objective="energy",
    ))

    _, unexpected_keys = model.load_state_dict(plain_model.state_dict(), strict=False)

    source_key = "transformer.h.0.attn.c_attn.weight"
    target_key = "model.base_model.model.transformer.h.0.attn.c_attn.base_layer.weight"
    assert unexpected_keys == []
    assert torch.equal(model.state_dict()[target_key], plain_model.state_dict()[source_key])


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
    assert call_counts["combined"] == model_config.K + 2
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
    assert model.energy_memory_search_gain_ema_initialized.all().item()
    assert torch.allclose(model.energy_memory_search_gain_ema, torch.tensor([0.1, 0.25]))
    assert torch.allclose(first_weights, torch.tensor([[0.0, 1.0], [0.8, 1.2]]))
    assert first_weights[first_gains > 0.0].mean().item() == pytest.approx(1.0)

    second_gains = torch.tensor([[0.4, 0.0], [0.0, 0.0]])
    second_weights = model._compute_energy_memory_search_gain_weights(second_gains)
    expected_ema = torch.tensor([0.99 * 0.1 + 0.01 * 0.4, 0.25])
    assert torch.allclose(model.energy_memory_search_gain_ema, expected_ema)
    assert second_weights[0, 0].item() == pytest.approx(1.0)
    ema_before_zero_batch = model.energy_memory_search_gain_ema.clone()
    zero_weights = model._compute_energy_memory_search_gain_weights(torch.zeros(2, 2))
    assert torch.equal(zero_weights, torch.zeros_like(zero_weights))
    assert torch.equal(model.energy_memory_search_gain_ema, ema_before_zero_batch)


@pytest.mark.forward
@pytest.mark.all
def test_energy_memory_search_gain_weight_ema_does_not_amplify_gain_scale_increase():
    model, _, _ = _build_shaped_energy_model(
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_use_gain_weighting=True,
        energy_memory_search_gain_ema_decay=0.99,
    )
    model._compute_energy_memory_search_gain_weights(torch.full((2, 2), 1e-8))

    gains = torch.tensor([[0.0, 0.0], [0.0, 0.4]])
    weights = model._compute_energy_memory_search_gain_weights(gains)

    assert weights[-1, -1].item() == pytest.approx(1.0)
    assert weights.max().item() <= gains.gt(0.0).sum().item()


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
def test_lipschitz_constraint_loss_accounting_and_backward():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        add_inner_loss_to_outer=False,
        lipschitz_weight=0.25,
        lipschitz_constraint=0.0,
    )
    model.train()

    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert stats["lipschitz_grad_norm_mean"].item() > 0.0
    assert stats["lipschitz_loss"].item() > 0.0
    expected_outer = stats["target_loss"] + model.lipschitz_weight * stats["lipschitz_loss"]
    assert torch.allclose(stats["outer_loss"], expected_outer)
    assert torch.allclose(output["loss"].detach(), expected_outer)

    output["loss"].backward()
    energy_gradients = [
        parameter.grad for parameter in model.energy_head.parameters()
        if parameter.grad is not None
    ]
    assert energy_gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)


@pytest.mark.forward
@pytest.mark.all
def test_lipschitz_config_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    defaults = GradMemGPTConfig(base_config=base_config)
    assert defaults.lipschitz_weight == 0.0
    assert defaults.lipschitz_constraint == 1.0

    for values in (
        {"lipschitz_weight": -0.1},
        {"lipschitz_constraint": -0.1},
        {"lipschitz_weight": float("inf")},
    ):
        with pytest.raises(ValueError, match="lipschitz"):
            GradMemGPTConfig(base_config=base_config, **values)
    with pytest.raises(ValueError, match="memory_backend='prefix'"):
        GradMemGPTConfig(base_config=base_config, memory_backend="lora", lipschitz_weight=0.1)

    config = GradMemGPTConfig(
        base_config=base_config,
        lipschitz_weight=0.25,
        lipschitz_constraint=12.0,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.lipschitz_weight == pytest.approx(0.25)
    assert restored.lipschitz_constraint == pytest.approx(12.0)


@pytest.mark.forward
@pytest.mark.all
def test_read_focal_loss_formula_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    defaults = GradMemGPTConfig(base_config=base_config)
    assert defaults.read_focal_gamma == 0.0

    for value in (-0.1, float("inf")):
        with pytest.raises(ValueError, match="read_focal_gamma"):
            GradMemGPTConfig(base_config=base_config, read_focal_gamma=value)

    config = GradMemGPTConfig(base_config=base_config, read_focal_gamma=2.0)
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.read_focal_gamma == pytest.approx(2.0)

    model = GradMemGPT(config)
    predictions = torch.tensor([
        [[4.0, 0.0, -1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
        [[0.5, 1.0, -0.5], [2.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
    ], requires_grad=True)
    labels = torch.tensor([[0, 2], [1, -100]])
    read_batch = {"label_shift": 0}

    token_losses = torch.nn.functional.cross_entropy(
        predictions[:, :-1].reshape(-1, predictions.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    valid = labels.ne(-100)
    expected_tokens = token_losses * (1.0 - token_losses.neg().exp()).square() * valid
    expected = expected_tokens.sum() / valid.sum()
    expected_per_example = expected_tokens.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    loss = model._compute_read_target_loss(predictions, read_batch, labels)
    per_example = model._compute_per_example_read_target_loss(predictions, read_batch, labels)
    assert torch.allclose(loss, expected)
    assert torch.allclose(per_example, expected_per_example)
    loss.backward()
    assert torch.isfinite(predictions.grad).all()

    model.read_focal_gamma = 0.0
    expected_ce = torch.nn.functional.cross_entropy(
        predictions.detach()[:, :-1].reshape(-1, predictions.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(model._compute_read_target_loss(predictions.detach(), read_batch, labels), expected_ce)


@pytest.mark.one_batch_train
@pytest.mark.all
def test_memory_noise_perturbs_training_gradient_inputs_but_not_clean_state():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        K=1,
        lr=0.0,
        memory_noise_sigma=0.5,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        add_inner_loss_to_outer=False,
    )
    backend = model.memory_backend_impl
    original_build_write_inputs = backend.build_write_inputs
    observed_memories = []

    def _record_memory(memory_state, batch_ctx):
        observed_memories.append(memory_state["mem_batch"].detach().clone())
        return original_build_write_inputs(memory_state, batch_ctx)

    backend.build_write_inputs = _record_memory
    initial = model.mem.detach().unsqueeze(0).expand(2, -1, -1)

    model.train()
    train_output = model(inputs, labels=labels, return_mem=True)
    assert not torch.equal(observed_memories[0], initial)
    assert torch.equal(train_output["mem"].detach(), initial)

    observed_memories.clear()
    model.eval()
    model(inputs, labels=labels)
    assert torch.equal(observed_memories[0], initial)


@pytest.mark.forward
@pytest.mark.all
def test_memory_noise_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    defaults = GradMemGPTConfig(base_config=base_config)
    assert defaults.memory_noise_sigma == 0.0
    with pytest.raises(ValueError, match="memory_noise_sigma"):
        GradMemGPTConfig(base_config=base_config, memory_noise_sigma=-0.1)
    with pytest.raises(ValueError, match="memory_backend='prefix'"):
        GradMemGPTConfig(base_config=base_config, memory_backend="lora", memory_noise_sigma=0.1)

    config = GradMemGPTConfig(base_config=base_config, memory_noise_sigma=0.25)
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.memory_noise_sigma == pytest.approx(0.25)


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("write_objective", ["energy", "reconstruction"])
def test_energy_memory_search_runs_after_each_write_step_and_is_training_only(write_objective):
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        write_objective=write_objective,
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
    if write_objective == "energy":
        energy_gradients = [
            parameter.grad for parameter in model.energy_head.parameters()
            if parameter.grad is not None
        ]
        assert energy_gradients
        assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)
    else:
        assert model.energy_head is None

    def _unexpected_search(*_args, **_kwargs):
        raise AssertionError("memory search must be disabled during evaluation")

    model._compute_energy_memory_search_loss = _unexpected_search
    model.eval()
    evaluation = model(inputs, labels=labels)
    assert "energy_memory_search_loss" not in evaluation["inner_loop_stats"]


@pytest.mark.one_batch_train
@pytest.mark.all
def test_reconstruction_memory_search_forward_and_backward():
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        write_objective="reconstruction",
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_weight=0.2,
        energy_memory_search_num_samples=1,
        add_inner_loss_to_outer=False,
        memory_alignment_weight=0.0,
        step_alignment_weight=0.0,
        intermediate_read_weight=0.0,
    )
    model.train()

    output = model(inputs, labels=labels)
    search_loss = output["inner_loop_stats"]["energy_memory_search_loss"]

    assert model.energy_head is None
    assert torch.isfinite(search_loss).item()
    output["loss"].backward()


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
    assert torch.allclose(model.energy_memory_search_gain_ema, torch.tensor([0.15, 0.3]))
    assert stats["energy_memory_search_loss"].item() == pytest.approx(11.0 / 6.0)
    assert stats["energy_memory_search_mean_relative_target_gain"].item() == pytest.approx(0.15)
    assert stats["energy_memory_search_max_relative_target_gain"].item() == pytest.approx(0.3)
    assert "energy_memory_search_unweighted_loss" not in stats
    assert "energy_memory_search_relative_target_gain" not in stats
    assert "energy_memory_search_gain_weight_mean" not in stats
    assert "energy_memory_search_gain_weight_max" not in stats
    assert "energy_memory_search_gain_ema" not in stats
    assert torch.allclose(
        stats["outer_loss"],
        stats["target_loss"] + model.energy_memory_search_weight * stats["energy_memory_search_loss"],
    )


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("use_gain_weighting", [False, True])
def test_energy_memory_search_masks_examples_below_minimum_relative_gain(use_gain_weighting):
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        energy_rank_weight=0.0,
        energy_traj_weight=0.0,
        energy_anchor_weight=0.0,
        energy_memory_search_weight=0.2,
        energy_memory_search_use_gain_weighting=use_gain_weighting,
        energy_memory_search_min_relative_target_gain=0.15,
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
    stats = model(inputs, labels=labels)["inner_loop_stats"]

    assert stats["energy_memory_search_loss"].item() == pytest.approx(3.0)
    assert stats["energy_memory_search_included_rate"].item() == pytest.approx(0.5)


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("write_objective", ["energy", "reconstruction"])
def test_energy_memory_search_can_roll_best_candidate_into_next_write_step(write_objective):
    torch.manual_seed(0)
    model, inputs, labels = _build_shaped_energy_model(
        batch_size=2,
        write_objective=write_objective,
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
    # K inner forwards, post-WRITE loss diagnostic, and always-on Lipschitz diagnostic.
    assert len(write_memories) == model.K + 2
    assert torch.allclose(write_memories[1], search_teachers[0])
    assert torch.allclose(search_previous[1], search_teachers[0])
    assert torch.allclose(write_memories[-1], search_teachers[-1])
    assert torch.allclose(output["mem"].detach(), search_teachers[-1])
    assert output["mem"].grad_fn is not None
    output["loss"].backward()
    if write_objective == "energy":
        energy_gradients = [
            parameter.grad for parameter in model.energy_head.parameters()
            if parameter.grad is not None
        ]
        assert energy_gradients
        assert all(torch.isfinite(gradient).all().item() for gradient in energy_gradients)
    else:
        assert model.energy_head is None


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
    assert defaults.energy_memory_search_min_relative_target_gain == pytest.approx(0.0)
    assert defaults.energy_memory_search_use_best_for_next_step is False

    invalid_overrides = [
        {"energy_memory_search_weight": -0.1},
        {"energy_memory_search_num_samples": 0},
        {"energy_memory_search_radius_scale": 0.0},
        {"energy_memory_search_gain_ema_decay": -0.1},
        {"energy_memory_search_gain_ema_decay": 1.0},
        {"energy_memory_search_min_relative_target_gain": -0.1},
        {"energy_memory_search_use_best_for_next_step": True},
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
        write_objective="reconstruction",
        energy_memory_search_weight=0.1,
        energy_memory_search_num_samples=7,
        energy_memory_search_radius_scale=0.4,
        energy_memory_search_use_gain_weighting=True,
        energy_memory_search_gain_ema_decay=0.95,
        energy_memory_search_min_relative_target_gain=0.05,
        energy_memory_search_use_best_for_next_step=True,
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.energy_memory_search_weight == pytest.approx(0.1)
    assert restored.energy_memory_search_num_samples == 7
    assert restored.energy_memory_search_radius_scale == pytest.approx(0.4)
    assert restored.energy_memory_search_use_gain_weighting is True
    assert restored.energy_memory_search_gain_ema_decay == pytest.approx(0.95)
    assert restored.energy_memory_search_min_relative_target_gain == pytest.approx(0.05)
    assert restored.energy_memory_search_use_best_for_next_step is True

    model = GradMemGPT(config)
    model.energy_memory_search_gain_ema.fill_(0.3)
    model.energy_memory_search_gain_ema_initialized.fill_(True)
    state_dict = model.state_dict()
    assert torch.allclose(state_dict["energy_memory_search_gain_ema"], torch.full((config.K,), 0.3))
    assert state_dict["energy_memory_search_gain_ema_initialized"].all().item()

    legacy_state_dict = model.state_dict()
    legacy_state_dict["energy_memory_search_gain_ema"] = torch.tensor(0.4)
    legacy_state_dict["energy_memory_search_gain_ema_initialized"] = torch.tensor(True)
    restored_model = GradMemGPT(config)
    restored_model.load_state_dict(legacy_state_dict)
    assert torch.allclose(
        restored_model.energy_memory_search_gain_ema,
        torch.full((config.K,), 0.4),
    )
    assert restored_model.energy_memory_search_gain_ema_initialized.all().item()

    larger_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend="prefix",
        K=5,
        grad_mode="second",
        use_adam=False,
        write_objective="reconstruction",
    )
    larger_model = GradMemGPT(larger_config)
    larger_model.load_state_dict(state_dict)
    assert torch.allclose(
        larger_model.energy_memory_search_gain_ema,
        torch.tensor([0.3, 0.3, 0.0, 0.0, 0.0]),
    )
    assert torch.equal(
        larger_model.energy_memory_search_gain_ema_initialized,
        torch.tensor([True, True, False, False, False]),
    )


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
@pytest.mark.parametrize("align_last_step", [False, True])
def test_step_alignment_uses_each_inner_update_and_resulting_memory_gradient(
    grad_align_norm,
    align_last_step,
):
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        memory_alignment_weight=0.0,
        step_alignment_weight=0.4,
        alignment_sampling=True,
        align_last_step=align_last_step,
        grad_align_norm=grad_align_norm,
        intermediate_read_weight=0.0,
    )
    model.train()
    recorded_losses = []
    recorded_cosines = []
    sampled_negative_memories = []
    original_step_alignment = model._compute_step_alignment
    original_build_negatives = model._build_energy_negative_memories

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

    def _record_negative_memories(*args, **kwargs):
        negatives, permutation = original_build_negatives(*args, **kwargs)
        sampled_negative_memories.extend(
            memory for memory in negatives.values() if memory is not None
        )
        return negatives, permutation

    model._compute_step_alignment = _record_step_alignment
    model._build_energy_negative_memories = _record_negative_memories
    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert len(sampled_negative_memories) == 3
    assert all(not memory.requires_grad for memory in sampled_negative_memories)
    assert all(memory.grad_fn is None for memory in sampled_negative_memories)
    assert len(recorded_losses) == model.K + int(align_last_step) + len(sampled_negative_memories)
    expected_loss = torch.stack(recorded_losses).mean()
    expected_cosine = torch.stack(recorded_cosines).mean()
    assert torch.allclose(stats["step_alignment_loss"], expected_loss.detach())
    assert torch.allclose(stats["step_alignment_cosine"], expected_cosine.detach())
    assert torch.allclose(
        stats["negative_step_alignment_loss"],
        torch.stack(recorded_losses[-3:]).mean().detach(),
    )
    assert torch.allclose(
        stats["negative_step_alignment_cosine"],
        torch.stack(recorded_cosines[-3:]).mean().detach(),
    )
    for name in ("deranged", "interpolated", "random"):
        assert f"negative_step_alignment_{name}_loss" in stats
        assert f"negative_step_alignment_{name}_cosine" in stats
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


@pytest.mark.one_batch_train
@pytest.mark.all
def test_step_alignment_sampling_is_disabled_by_default():
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        memory_alignment_weight=0.0,
        step_alignment_weight=0.4,
        intermediate_read_weight=0.0,
    )

    def _unexpected_negative_sampling(*_args, **_kwargs):
        raise AssertionError("regular step alignment must not sample negative memories")

    model._build_energy_negative_memories = _unexpected_negative_sampling
    model.train()
    stats = model(inputs, labels=labels)["inner_loop_stats"]

    assert "negative_step_alignment_loss" not in stats
    assert "negative_step_alignment_cosine" not in stats


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("write_objective", ["reconstruction", "energy_with_reconstruction"])
def test_step_alignment_sampling_supports_all_write_objectives(write_objective):
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        write_objective=write_objective,
        memory_alignment_weight=0.0,
        step_alignment_weight=0.4,
        alignment_sampling=True,
        intermediate_read_weight=0.0,
    )
    model.train()

    output = model(inputs, labels=labels)
    stats = output["inner_loop_stats"]

    assert torch.isfinite(stats["negative_step_alignment_loss"]).item()
    assert torch.isfinite(stats["negative_step_alignment_cosine"]).item()
    output["loss"].backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)


@pytest.mark.forward
@pytest.mark.all
def test_step_alignment_sampling_is_training_only():
    torch.manual_seed(0)
    model, inputs, labels = _build_memory_alignment_model(
        memory_alignment_weight=0.0,
        step_alignment_weight=0.4,
        alignment_sampling=True,
        intermediate_read_weight=0.0,
    )
    model.eval()

    def _unexpected_negative_sampling(*_args, **_kwargs):
        raise AssertionError("evaluation must not sample alignment memories")

    model._build_energy_negative_memories = _unexpected_negative_sampling
    with torch.no_grad():
        output = model(inputs, labels=labels)

    stats = output["inner_loop_stats"]
    assert "negative_step_alignment_loss" not in stats
    assert "negative_step_alignment_cosine" not in stats


@pytest.mark.forward
@pytest.mark.all
def test_training_runners_log_average_alignment_cosines():
    from run_gradmemgpt_on_kv_retrieval import LOSS_COMPONENT_KEYS as kv_keys
    from run_gradmemgpt_on_mqar import LOSS_COMPONENT_KEYS as mqar_keys

    expected_keys = {
        "step_alignment_cosine",
        "negative_step_alignment_cosine",
        "negative_step_alignment_deranged_cosine",
        "negative_step_alignment_interpolated_cosine",
        "negative_step_alignment_random_cosine",
    }
    assert expected_keys <= set(kv_keys)
    assert expected_keys <= set(mqar_keys)


@pytest.mark.forward
@pytest.mark.all
def test_step_alignment_defaults_validation_and_serialization(tmp_path):
    base_config = _build_base_config("gpt2")
    assert GradMemGPTConfig(base_config=base_config).step_alignment_weight == 0.0
    assert GradMemGPTConfig(base_config=base_config).alignment_sampling is False
    assert GradMemGPTConfig(base_config=base_config).align_last_step is False
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
        write_objective="energy",
        step_alignment_weight=0.25,
        alignment_sampling=True,
        align_last_step=True,
        grad_align_norm="norm",
    )
    config.save_pretrained(tmp_path)
    restored = GradMemGPTConfig.from_pretrained(tmp_path)
    assert restored.step_alignment_weight == pytest.approx(0.25)
    assert restored.alignment_sampling is True
    assert restored.align_last_step is True
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
def test_zero_weight_orthogonal_loss_does_not_train_alpha():
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
    alpha_grad = torch.autograd.grad(
        output["loss"],
        model.orthogonal_alpha_raw,
        allow_unused=True,
    )[0]

    assert torch.allclose(output["loss"].detach(), target_loss.detach())
    assert torch.allclose(outer_mem_grad, target_mem_grad)
    assert alpha_grad is None


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


@pytest.mark.forward
@pytest.mark.all
def test_kv_runner_outer_adam_beta_defaults():
    from run_gradmemgpt_on_kv_retrieval import ExperimentArgs

    beta1 = ExperimentArgs.__dataclass_fields__["adam_beta1"].default
    beta2 = ExperimentArgs.__dataclass_fields__["adam_beta2"].default
    assert beta1 == pytest.approx(0.9)
    assert beta2 == pytest.approx(0.999)


@pytest.mark.forward
@pytest.mark.all
def test_kv_runner_seed_controls_parameter_initialization():
    from run_gradmemgpt_on_kv_retrieval import initialize_run_seed

    base_config = _build_base_config("gpt2")

    def initialize(seed):
        initialize_run_seed(seed)
        model = GradMemGPT(GradMemGPTConfig(
            base_config=base_config,
            memory_backend="prefix",
            n_mem_tokens=4,
            K=1,
            write_objective="energy",
        ))
        return {
            "memory": model.mem.detach().clone(),
            "backbone": next(model.model.parameters()).detach().clone(),
            "energy": next(model.energy_head.parameters()).detach().clone(),
        }

    first = initialize(123)
    repeated = initialize(123)
    different = initialize(124)

    assert all(torch.equal(first[name], repeated[name]) for name in first)
    assert all(not torch.equal(first[name], different[name]) for name in first)


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
    overrides = {
        "lipschitz_weight": 0.1,
        "lipschitz_constraint": 0.0,
    } if shaping_active else {
        "energy_rank_weight": 0.0,
        "energy_traj_weight": 0.0,
        "energy_anchor_weight": 0.0,
        "lipschitz_weight": 0.0,
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
        "lipschitz_grad_norm_mean",
        "lipschitz_grad_norm_max",
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
        "lipschitz_loss",
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
