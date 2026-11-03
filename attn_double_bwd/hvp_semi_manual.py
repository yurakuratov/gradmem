"""ATen FlashAttention forward/first backward, explicit second backward.

The fused operators are private PyTorch APIs, tested with PyTorch 2.9/Triton
3.5 (also the local FlashBoB environment). Masks, dropout, FP32, CPU, and
unsupported shapes use the fully manual implementation without changing their
semantics. No tutorial Triton forward/backward kernels or JVP machinery remain.
"""

import torch
from transformers.utils import logging

from .hvp_manual import SDPA_FullManualBwd, _prepare_attention, hvp_manual


logger = logging.get_logger(__name__)


class _ATenFlashBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, go, q, k, v, out, lse, cu_q, cu_k, seed, offset, max_q, max_k, causal, scale):
        go = go.contiguous()
        ctx.save_for_backward(go, q, k, v, None)
        ctx.settings = None, causal, 1.0, scale
        return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            go, q, k, v, out, lse, cu_q, cu_k, max_q, max_k,
            0.0, causal, seed, offset, scale=scale,
        )[:3]

    @staticmethod
    def backward(ctx, gq, gk, gv):
        # The same explicit adjoint as hvp_manual, not a derivative of ATen's
        # opaque first-backward op (which has no native second backward).
        grads = SDPA_FullManualBwd.backward(ctx, gq, gk, gv)
        return (*grads[:4], *((None,) * 10))


class _ATenFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        out, lse, cu_q, cu_k, max_q, max_k, seed, offset, _ = (
            torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, dropout_p=0.0, is_causal=causal,
                return_debug_mask=False, scale=scale,
            )
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, seed, offset)
        ctx.settings = max_q, max_k, causal, scale
        return out

    @staticmethod
    def backward(ctx, go):
        q, k, v, out, lse, cu_q, cu_k, seed, offset = ctx.saved_tensors
        max_q, max_k, causal, scale = ctx.settings
        grads = _ATenFlashBackward.apply(
            go, q, k, v, out, lse, cu_q, cu_k, seed, offset,
            max_q, max_k, causal, scale,
        )
        return *grads, None, None


def hvp_semi_manual(module, query, key, value, attention_mask, dropout=0.0,
                    scaling=None, is_causal=None, **kwargs):
    query, key, value, attention_mask, scaling, is_causal = _prepare_attention(
        module, query, key, value, attention_mask, scaling, is_causal,
    )
    eligible = (
        query.device.type == "cuda" and query.dtype in (torch.float16, torch.bfloat16)
        and attention_mask is None and not dropout
        and query.shape[-2] == key.shape[-2]
        and query.shape[-1] in (32, 48, 64, 128)
        and query.shape == key.shape == value.shape
        and not kwargs.get("output_attentions", False) and kwargs.get("head_mask") is None
    )
    if not eligible:
        logger.warning_once(
            "hvp_semi_manual is using manual attention: ATen FlashAttention requires "
            "unmasked square CUDA FP16/BF16 calls without dropout (head dim 32/48/64/128)."
        )
        return hvp_manual(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    query, key, value = (x.contiguous() for x in (query, key, value))
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    out = _ATenFlash.apply(query, key, value, bool(is_causal), scale)
    return out.transpose(1, 2).contiguous(), None
