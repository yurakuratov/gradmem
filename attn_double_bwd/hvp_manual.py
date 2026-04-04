import torch
import triton
import triton.language as tl

from transformers.integrations.sdpa_attention import repeat_kv

from typing import Optional


@triton.jit
def _philox_rand_kernel(
    out_ptr,
    offs_ptr,
    seed,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    offs = tl.load(offs_ptr + idx, mask=mask, other=0).to(tl.uint32)
    rnd = tl.rand(seed, offs)
    tl.store(out_ptr + idx, rnd, mask=mask)


def _philox_rand_from_offsets(out: torch.Tensor, offsets: torch.Tensor, seed: int) -> None:
    flat_out = out.reshape(-1)
    flat_offs = offsets.reshape(-1).contiguous()
    n = flat_out.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]),)
    _philox_rand_kernel[grid](flat_out, flat_offs, seed, n, BLOCK=1024)


def _next_cuda_philox_state(
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> tuple[int, int]:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device for dropout RNG, got {device}")
    state = torch.randint(
        0,
        2**63 - 1,
        (2,),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    return int(state[0].item()), int(state[1].item())


def _philox_dropout_keep(
    shape,
    strides,
    p: float,
    device: torch.device,
    seed: int,
    offset: int,
) -> Optional[torch.Tensor]:
    if p <= 0.0:
        return None
    bsz, nheads, seqlen_q, seqlen_k = shape
    stride_b, stride_h, stride_m, stride_n = strides
    b = torch.arange(bsz, device=device, dtype=torch.int64)[:, None, None, None] * stride_b
    h = torch.arange(nheads, device=device, dtype=torch.int64)[None, :, None, None] * stride_h
    m = torch.arange(seqlen_q, device=device, dtype=torch.int64)[None, None, :, None] * stride_m
    n = torch.arange(seqlen_k, device=device, dtype=torch.int64)[None, None, None, :] * stride_n
    philox_offs = torch.as_tensor(offset, device=device, dtype=torch.int64) + b + h + m + n
    rand = torch.empty(shape, device=device, dtype=torch.float32)
    _philox_rand_from_offsets(rand, philox_offs, seed)
    return rand > p


class SDPA_FullManualBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, q, k, v, attn_mask, is_causal, dropout_p, philox_seed, philox_offset, scale):
        ctx.save_for_backward(grad_out, q, k, v)
        ctx.attn_mask = attn_mask
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.dropout_p = float(dropout_p)
        ctx.philox_seed = int(philox_seed)
        ctx.philox_offset = int(philox_offset)
        d = q.size(-1)
        scale = scale if (scale is not None) else (1 / d ** 0.5)

        scores = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
            scores = scores + attn_mask
        if is_causal:
            S, L = scores.size(-2), scores.size(-1)
            causal = torch.triu(scores.new_full((S, L), float("-inf")), diagonal=1).to(q.dtype)
            scores = scores + causal
        
        P = torch.softmax(scores, dim=-1)
        if ctx.dropout_p > 0.0:
            keep = _philox_dropout_keep(
                P.shape, P.stride(), ctx.dropout_p, P.device, ctx.philox_seed, ctx.philox_offset
            )
            dropout_scale = 1.0 / (1.0 - ctx.dropout_p)
            P_drop = P * keep.to(P.dtype) * dropout_scale
        else:
            keep = None
            P_drop = P

        dV  = P_drop.transpose(-2, -1) @ grad_out
        dP_drop  = grad_out @ v.transpose(-2, -1)
        if keep is not None:
            dP = dP_drop * keep.to(dP_drop.dtype) * dropout_scale
        else:
            dP = dP_drop
        Ssm = (dP * P).sum(dim=-1, keepdim=True)
        dS  = (dP - Ssm) * P
        dQ  = dS @ k * scale
        dK  = dS.transpose(-2, -1) @ q * scale
        return dQ, dK, dV, None, None, None, None
    
    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v, *_):
        grad_out, q, k, v = ctx.saved_tensors
        attn_mask, is_causal = ctx.attn_mask, ctx.is_causal
        dropout_p, scale = ctx.dropout_p, ctx.scale

        d = q.size(-1)
        scale = scale if (scale is not None) else (1 / d ** 0.5)

        scores = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
            scores = scores + attn_mask
        if is_causal:
            S, L = scores.size(-2), scores.size(-1)
            causal = torch.triu(scores.new_full((S, L), float("-inf")), diagonal=1).to(q.dtype)
            scores = scores + causal

        P = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            keep = _philox_dropout_keep(
                P.shape, P.stride(), dropout_p, P.device, ctx.philox_seed, ctx.philox_offset
            )
            dropout_scale = 1.0 / (1.0 - dropout_p)
            P = P * keep.to(P.dtype) * dropout_scale

        dP  = grad_out @ v.transpose(-2, -1)
        Ssm = (dP * P).sum(dim=-1, keepdim=True)
        dS  = (dP - Ssm) * P

        bar_dS = (grad_q @ k.transpose(-2, -1) + q @ grad_k.transpose(-2, -1)) * scale
        beta = (bar_dS * P).sum(dim=-1, keepdim=True)
        bar_dP = P * (bar_dS - beta)
        bar_P  = grad_out @ grad_v.transpose(-2, -1) + bar_dS * (dP - Ssm) - dP * beta

        tmp = (bar_P * P).sum(dim=-1, keepdim=True)
        bar_scores = (bar_P - tmp) * P

        dgrad_grad_out = P @ grad_v + bar_dP @ v
        dgrad_q        = (dS @ grad_k + bar_scores @ k) * scale
        dgrad_k        = (dS.transpose(-2, -1) @ grad_q + bar_scores.transpose(-2, -1) @ q) * scale
        dgrad_v        = bar_dP.transpose(-2, -1) @ grad_out

        return dgrad_grad_out, dgrad_q, dgrad_k, dgrad_v, None, None, None, None, None, None
    

class SDPA_FullManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        ctx.attn_mask = attn_mask
        ctx.is_causal = bool(is_causal)
        ctx.scale = scale
        ctx.dropout_p = float(dropout_p)

        d = q.size(-1)
        scale = scale if (scale is not None) else (1 / d ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
            scores = scores + attn_mask

        if is_causal:
            S, L = scores.size(-2), scores.size(-1)
            scores = scores + torch.triu(scores.new_full((S, L), float("-inf")), diagonal=1)

        P = torch.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0.0:
            philox_seed, philox_offset = _next_cuda_philox_state(P.device)
            keep = _philox_dropout_keep(
                P.shape, P.stride(), dropout_p, P.device, philox_seed, philox_offset
            )
            dropout_scale = 1.0 / (1.0 - dropout_p)
            P = P * keep.to(P.dtype) * dropout_scale
        else:
            philox_seed = 0
            philox_offset = 0

        ctx.philox_seed = int(philox_seed)
        ctx.philox_offset = int(philox_offset)
        ctx.save_for_backward(q, k, v)
        
        out = P @ v
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        attn_mask, is_causal = ctx.attn_mask, ctx.is_causal
        scale = ctx.scale
        return SDPA_FullManualBwd.apply(
            grad_out,
            q,
            k,
            v,
            attn_mask,
            is_causal,
            ctx.dropout_p,
            ctx.philox_seed,
            ctx.philox_offset,
            scale,
        )


def hvp_manual(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Expected query/key/value to be 4D BHSD tensors, got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    query_bhsd = query
    key_bhsd = key
    value_bhsd = value

    if hasattr(module, "num_key_value_groups"):
        key_bhsd = repeat_kv(key_bhsd, module.num_key_value_groups)
        value_bhsd = repeat_kv(value_bhsd, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key_bhsd.shape[-2]]
        if attention_mask.size(1) < query_bhsd.size(1):
            attention_mask = attention_mask.expand(-1, query_bhsd.size(1), -1, -1)
    
    if is_causal is None:
        is_causal = query_bhsd.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    attn_output = SDPA_FullManual.apply(
        query_bhsd.contiguous(),
        key_bhsd.contiguous(),
        value_bhsd.contiguous(),
        attention_mask,
        dropout,
        is_causal,
        scaling,
    )
    # Match HF SDPA integration expectation: return (B, S, H, D).
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
