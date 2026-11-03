"""Explicit attention forward, backward, and backward-over-backward."""

import torch


def _probabilities(q, k, mask, causal, scale):
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, -float("inf")) if mask.dtype == torch.bool else scores + mask
    if causal:
        allowed = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, -float("inf"))
    empty = torch.isneginf(scores).all(-1, keepdim=True)
    return scores.masked_fill(empty, 0).softmax(-1).masked_fill(empty, 0)


def _work(tensor):
    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


class SDPA_FullManualBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, q, k, v, attn_mask, is_causal, dropout_mask, dropout_scale, scale):
        ctx.save_for_backward(grad_out, q, k, v, dropout_mask)
        ctx.settings = attn_mask, is_causal, dropout_scale, scale
        with torch.autocast(q.device.type, enabled=False):
            go, qw, kw, vw = map(_work, (grad_out, q, k, v))
            p = _probabilities(qw, kw, attn_mask, is_causal, scale)
            r = dropout_mask.to(p.dtype) * dropout_scale if dropout_mask is not None else 1.0
            dp = (go @ vw.transpose(-2, -1)) * r
            ds = p * (dp - (dp * p).sum(-1, keepdim=True))
            return ((ds @ kw * scale).to(q.dtype),
                    (ds.transpose(-2, -1) @ qw * scale).to(k.dtype),
                    ((p * r).transpose(-2, -1) @ go).to(v.dtype))

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        grad_out, q, k, v, dropout_mask = ctx.saved_tensors
        mask, causal, dropout_scale, scale = ctx.settings
        with torch.autocast(q.device.type, enabled=False):
            go, qw, kw, vw = map(_work, (grad_out, q, k, v))
            gq, gk, gv = [_work(g) if g is not None else torch.zeros_like(_work(x))
                          for g, x in zip((grad_q, grad_k, grad_v), (q, k, v))]
            p = _probabilities(qw, kw, mask, causal, scale)
            r = dropout_mask.to(p.dtype) * dropout_scale if dropout_mask is not None else 1.0
            dp = (go @ vw.transpose(-2, -1)) * r
            center = (dp * p).sum(-1, keepdim=True)
            ds = p * (dp - center)
            bar_ds = (gq @ kw.transpose(-2, -1) + qw @ gk.transpose(-2, -1)) * scale
            beta = (bar_ds * p).sum(-1, keepdim=True)
            bar_dp = p * (bar_ds - beta)
            bar_p = (go @ gv.transpose(-2, -1)) * r + bar_ds * (dp - center) - dp * beta
            bar_scores = p * (bar_p - (bar_p * p).sum(-1, keepdim=True))
            dgo = (p * r) @ gv + (bar_dp * r) @ vw
            dq = (ds @ gk + bar_scores @ kw) * scale
            dk = (ds.transpose(-2, -1) @ gq + bar_scores.transpose(-2, -1) @ qw) * scale
            dv = (bar_dp * r).transpose(-2, -1) @ go
        return dgo.to(grad_out.dtype), dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


class SDPA_FullManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if not 0 <= dropout_p <= 1:
            raise ValueError("dropout_p must be between 0 and 1")
        scale = scale if scale is not None else q.shape[-1] ** -0.5
        with torch.autocast(q.device.type, enabled=False):
            qw, kw, vw = map(_work, (q, k, v))
            p = _probabilities(qw, kw, attn_mask, is_causal, scale)
            dropout_mask = torch.rand_like(p) >= dropout_p if dropout_p else None
            dropout_scale = 1 / (1 - dropout_p) if dropout_p < 1 else 0.0
            if dropout_mask is not None:
                p = p * dropout_mask * dropout_scale
            out = (p @ vw).to(q.dtype)
        ctx.save_for_backward(q, k, v, dropout_mask)
        ctx.settings = attn_mask, is_causal, dropout_scale, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, dropout_mask = ctx.saved_tensors
        mask, causal, dropout_scale, scale = ctx.settings
        grads = SDPA_FullManualBwd.apply(grad_out, q, k, v, mask, causal, dropout_mask, dropout_scale, scale)
        return *grads, None, None, None, None


def _prepare_attention(module, query, key, value, attention_mask, scaling, is_causal):
    if torch.is_autocast_enabled(query.device.type) and query.dtype != torch.float64:
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (x.to(dtype) for x in (query, key, value))
    if query.shape[1] != key.shape[1]:
        groups = query.shape[1] // key.shape[1]
        key, value = (x.repeat_interleave(groups, dim=1) for x in (key, value))
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[..., :key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[-2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
    if scaling is None and hasattr(module, "scale_attn_weights"):
        scaling = query.shape[-1] ** -0.5 if module.scale_attn_weights else 1.0
        if getattr(module, "scale_attn_by_inverse_layer_idx", False):
            scaling /= module.layer_idx + 1
    return query, key, value, attention_mask, scaling, is_causal


def hvp_manual(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    query, key, value, attention_mask, scaling, is_causal = _prepare_attention(
        module, query, key, value, attention_mask, scaling, is_causal,
    )
    out = SDPA_FullManual.apply(query, key, value, attention_mask, dropout, is_causal, scaling)
    return out.transpose(1, 2).contiguous(), None
