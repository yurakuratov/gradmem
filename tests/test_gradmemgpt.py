import pytest
import torch
from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig

from grad_memgpt import GradMemGPT, GradMemGPTConfig


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
    else:
        raise ValueError(f"Unknown backend={memory_backend}")


@pytest.mark.forward
@pytest.mark.all
@pytest.mark.parametrize("model_family", ["gpt2", "gpt_neox", "llama"])
@pytest.mark.parametrize("memory_backend", ["prefix", "lora"])
def test_forward(model_family: str, memory_backend: str):
    torch.manual_seed(0)

    base_config = _build_base_config(model_family)
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend=memory_backend,
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        **_backend_extra_kwargs(memory_backend),
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
    assert stats["inner_grad_norm_mean"].item() >= 0
    assert stats["delta_mem_norm_mean"].item() >= 0

    _assert_return_mem_payload(memory_backend, output)


@pytest.mark.one_batch_train
@pytest.mark.all
@pytest.mark.parametrize("model_family", ["gpt2", "gpt_neox", "llama"])
@pytest.mark.parametrize("memory_backend", ["prefix", "lora"])
def test_single_batch_train(model_family: str, memory_backend: str):
    torch.manual_seed(0)

    base_config = _build_base_config(model_family)
    model_config = GradMemGPTConfig(
        base_config=base_config,
        memory_backend=memory_backend,
        n_mem_tokens=4,
        K=1,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        use_mem_proj=False,
        mem_proj_mode="none",
        use_write_head=False,
        attn_implementation="eager",
        **_backend_extra_kwargs(memory_backend),
    )

    model = GradMemGPT(model_config)
    model.train()

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
