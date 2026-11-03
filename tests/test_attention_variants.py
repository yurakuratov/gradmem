"""Numerical regressions for the original higher-order attention variants."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from attn_double_bwd.hvp_manual import hvp_manual
from attn_double_bwd.hvp_semi_manual import hvp_semi_manual


def _derivatives(out, inputs):
    first = torch.autograd.grad(out.square().sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), inputs)
    return out, first, second


@pytest.mark.parametrize("backend", ["hvp_manual", "hvp_semi_manual"])
@pytest.mark.parametrize("mask_kind", ["causal", "boolean", "additive", "none"])
def test_attention_variant_derivatives(backend, mask_kind):
    device = "cuda" if backend == "hvp_semi_manual" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for the original semi-manual kernels")
    torch.manual_seed(41)
    inputs = [torch.randn(1, 2, 32, 32, device=device, requires_grad=True) for _ in range(3)]
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    mask = None
    if mask_kind in ("boolean", "additive"):
        mask = torch.ones(1, 2, 32, 32, dtype=torch.bool, device=device)
        mask[..., -3:] = False
        if mask_kind == "additive":
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -float("inf"))
    causal = mask_kind == "causal"
    function = hvp_manual if backend == "hvp_manual" else hvp_semi_manual
    actual, _ = function(SimpleNamespace(is_causal=causal), *inputs, mask, is_causal=causal)
    actual = actual.transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(*reference, attn_mask=mask, is_causal=causal)
    actual = _derivatives(actual, inputs)
    expected = _derivatives(expected, reference)
    for candidate, target in zip(torch.utils._pytree.tree_leaves(actual), torch.utils._pytree.tree_leaves(expected)):
        torch.testing.assert_close(candidate, target, atol=3e-3, rtol=3e-3)
        relative_error = (candidate - target).norm() / target.norm().clamp_min(1e-12)
        assert relative_error < 0.005


@pytest.mark.parametrize("backend", [hvp_manual, hvp_semi_manual], ids=["manual", "semi_manual"])
@pytest.mark.parametrize("dropout", [0.25, 1.0])
def test_attention_dropout_derivatives(backend, dropout):
    torch.manual_seed(42)
    inputs = [torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    torch.manual_seed(12)
    actual, _ = backend(SimpleNamespace(is_causal=False), *inputs, None, dropout=dropout)
    torch.manual_seed(12)
    q, k, v = reference
    p = (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(-1)
    keep = torch.rand_like(p) >= dropout
    p = p * keep * (1 / (1 - dropout) if dropout < 1 else 0)
    expected = p @ v
    for candidate, target in zip(torch.utils._pytree.tree_leaves(_derivatives(actual.transpose(1, 2), inputs)),
                                 torch.utils._pytree.tree_leaves(_derivatives(expected, reference))):
        torch.testing.assert_close(candidate, target, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["hvp_manual", "hvp_semi_manual"])
@pytest.mark.parametrize("objective", ["reconstruction", "energy", "energy_with_reconstruction"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_variant_gradmem_meta_training(backend, objective, dtype):
    from contextlib import nullcontext
    from transformers import LlamaConfig
    from grad_memgpt import GradMemGPT, GradMemGPTConfig

    config = LlamaConfig(vocab_size=41, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         pad_token_id=0, max_position_embeddings=128)
    model = GradMemGPT(GradMemGPTConfig(base_config=config, attn_implementation=backend,
                                       n_mem_tokens=2, K=2, grad_mode="second",
                                       write_objective=objective)).cuda().train()
    inputs = {"context_input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], device="cuda"),
              "query_input_ids": torch.tensor([[8, 9, 10], [11, 12, 13]], device="cuda")}
    labels = torch.tensor([[-100, 14, 15], [-100, 16, -100]], device="cuda")
    with torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext():
        output = model(inputs, labels=labels)
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.mem.grad is not None and torch.isfinite(model.mem.grad).all()
