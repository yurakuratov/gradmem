"""Transformers adapter for bundled FlashBoB, with differentiable math fallback.

No separately installed ``bob`` package or sibling checkout is used.
"""

import torch
from torch.nn import functional as F
from transformers.utils import logging


logger = logging.get_logger(__name__)
_sdpa_bob = None


def register_flashbob():
    global _sdpa_bob
    if _sdpa_bob is None:
        try:
            from .flashbob import sdpa_bob
        except ImportError as error:
            raise ImportError(
                "Bundled FlashBoB requires a compatible environment. "
                "Install environment-flashbob.yaml "
                "(PyTorch 2.8/Triton 3.4 or PyTorch 2.9/Triton 3.5)."
            ) from error
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask

        _sdpa_bob = sdpa_bob
        for name in ("flashbob", "bob"):
            AttentionInterface.register(name, flashbob_attention_forward)
            # Never drop padding, cache-position, or sliding-window masks.
            AttentionMaskInterface.register(name, sdpa_mask)


def _fallback_reason(query, key, value, attention_mask, dropout, window, kwargs):
    if attention_mask is not None:
        return "an explicit attention mask"
    if dropout:
        return "attention dropout"
    if kwargs.get("head_mask") is not None or kwargs.get("output_attentions", False):
        return "head masks or requested attention weights"
    if query.shape[-1] != value.shape[-1]:
        return "different key/value head dimensions"
    if query.device.type == "cuda":
        if query.dtype not in (torch.float16, torch.bfloat16):
            return "CUDA attention outside FP16/BF16"
        if query.shape[-1] not in (32, 48, 64, 128):
            return "an unsupported CUDA head dimension"
    elif query.device.type != "cpu" or query.dtype not in (
        torch.float16, torch.bfloat16, torch.float32
    ):
        return "an unsupported device or dtype"
    if query.shape[1] != key.shape[1]:
        if (query.device.type != "cuda" or query.dtype != torch.bfloat16
                or query.shape[-1] not in (32, 64, 128)
                or query.shape[-2] != key.shape[-2]
                or query.shape[1] // key.shape[1] > 32 or window):
            return "unsupported grouped-query geometry/dtype"
    return None


def _math_attention(query, key, value, mask, dropout, scaling, causal, window, kwargs):
    """Math SDPA must be forced: default CUDA SDPA cannot double-backward."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if query.shape[1] != key.shape[1]:
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if causal or window:
        nq, nk = query.shape[-2], key.shape[-2]
        qpos = torch.arange(nq, device=query.device) + max(0, nk - nq)
        kpos = torch.arange(nk, device=query.device)
        allowed = kpos[None, :] <= qpos[:, None]
        if window:
            allowed = allowed & (kpos[None, :] > qpos[:, None] - window)
        if mask is None:
            mask = allowed
        elif mask.dtype == torch.bool:
            mask = mask & allowed
        else:
            mask = mask.masked_fill(~allowed, float("-inf"))
    if kwargs.get("head_mask") is not None or kwargs.get("output_attentions", False):
        # Preserve optional HF features as well as the second-order graph.
        dtype = query.dtype
        q = query.float() if dtype in (torch.float16, torch.bfloat16) else query
        scores = q @ key.to(q.dtype).transpose(-2, -1)
        scores = scores * (scaling if scaling is not None else query.shape[-1] ** -0.5)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
        weights = scores.masked_fill(empty, 0).softmax(dim=-1).masked_fill(empty, 0).to(dtype)
        weights = F.dropout(weights, p=dropout, training=dropout > 0)
        if kwargs.get("head_mask") is not None:
            weights = weights * kwargs["head_mask"].to(weights.dtype)
        return weights @ value, weights
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling,
        )
    return out, None


def flashbob_attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None,
    is_causal=None, **kwargs,
):
    """HF layout in: [B,H,N,D]; layout out: [B,N,H,D]."""
    # Match SDPA autocast semantics. RoPE/upcast norms can promote Q/K back to
    # FP32 even though projections and V are BF16. Casts retain their gradients.
    if torch.is_autocast_enabled(query.device.type) and query.dtype != torch.float64:
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (tensor.to(dtype) for tensor in (query, key, value))
    if is_causal is None:
        is_causal = attention_mask is None and getattr(module, "is_causal", True)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[..., :key.shape[-2]]
    # GPT-2's eager path has optional nonstandard scaling; do not lose it.
    if hasattr(module, "scale_attn_weights") and scaling is None:
        scaling = query.shape[-1] ** -0.5 if module.scale_attn_weights else 1.0
        if getattr(module, "scale_attn_by_inverse_layer_idx", False):
            scaling /= module.layer_idx + 1
    window = max(0, kwargs.get("sliding_window") or 0)
    reason = _fallback_reason(query, key, value, attention_mask, dropout, window, kwargs)
    if reason is not None:
        logger.warning_once(
            f"FlashBoB is using differentiable math attention for {reason}; "
            "semantics are preserved, but this call is not accelerated."
        )
        out, weights = _math_attention(
            query, key, value, attention_mask, dropout, scaling, bool(is_causal), window, kwargs,
        )
    else:
        if _sdpa_bob is None:
            register_flashbob()
        out = _sdpa_bob(
            query, key, value, is_causal=bool(is_causal), scale=scaling, window_size=window,
        )
        weights = None
    return out.transpose(1, 2).contiguous(), weights
