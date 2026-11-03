"""Model setup shared by the fused higher-order attention backends."""

import torch


def _default_attention_mask(module, args, kwargs):
    """Avoid HF 4.57's quadratic packed mask for default sequential positions.

    All-valid 2D masks have identical semantics for these otherwise unmasked
    calls, and let the SDPA mask builder omit the causal matrix. Explicit masks
    and position IDs are never replaced, so padding and packed batches retain
    their ordinary semantics.
    """
    if kwargs.get("attention_mask") is not None or kwargs.get("position_ids") is not None:
        return args, kwargs
    inputs = kwargs.get("inputs_embeds")
    if inputs is None:
        inputs = kwargs.get("input_ids")
    if inputs is None and args:
        inputs = args[0]
    if inputs is None:
        return args, kwargs
    cache = kwargs.get("past_key_values")
    if cache is not None and (not hasattr(cache, "get_seq_length") or getattr(cache, "is_compileable", False)):
        return args, kwargs
    past_length = cache.get_seq_length() if cache is not None else 0
    kwargs["attention_mask"] = torch.ones(
        inputs.shape[0], past_length + inputs.shape[1], dtype=torch.bool, device=inputs.device,
    )
    return args, kwargs


def prepare_attention_model(model):
    model.register_forward_pre_hook(_default_attention_mask, with_kwargs=True)
    return model
