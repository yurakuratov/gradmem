"""Checkpoint, MQAR, fixed-memory scoring, and numerical analysis helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"


def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def resolve_checkpoint(value):
    """Require an explicitly provided checkpoint folder containing model weights."""
    path = repo_path(value)
    if path.is_file():
        raise ValueError(f"Expected a checkpoint folder, not a file: {path}")
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint folder must contain model.safetensors: {path}")
    return weights


def load_model(checkpoint, device="cpu", inner_lr=None, dtype=torch.float32):
    # Training-runner imports query CUDA; keep them out of plotting-only startup.
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from grad_memgpt import GradMemGPTConfig
    from run_gradmemgpt_on_mqar import MQARGradMemGPT

    config = GradMemGPTConfig.from_pretrained(checkpoint.parent, local_files_only=True)
    if isinstance(config.base_config, dict):
        base = dict(config.base_config)
        config.base_config = AutoConfig.for_model(base.pop("model_type"), **base)
    for key in ("energy_rank_weight", "energy_traj_weight", "energy_anchor_weight",
                "lipschitz_weight", "energy_memory_search_weight", "ivan_loss_weight",
                "memory_alignment_weight", "step_alignment_weight", "intermediate_read_weight",
                "orthogonal_loss_weight"):
        setattr(config, key, 0.0)
    config.add_inner_loss_to_outer = False
    config.energy_memory_search_use_best_for_next_step = False
    config.grad_mode = "none"
    config.last_K_second_order = 0
    if inner_lr is not None:
        if inner_lr <= 0:
            raise ValueError("inner_lr must be positive")
        config.lr = inner_lr
    model = MQARGradMemGPT(config).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=False)
    allowed = {"model.lm_head.weight", "energy_memory_search_gain_ema", "energy_memory_search_gain_ema_initialized"}
    if unexpected or not set(missing).issubset(allowed):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()
    model.to(device=device).eval()
    model.requires_grad_(False)
    return model


def load_data(path, examples, offset=0):
    from zoology_mqar_data import load_saved_mqar_datasets

    _, valid, metadata = load_saved_mqar_datasets(repo_path(path))
    if examples <= 0 or offset < 0 or offset + examples > len(valid):
        raise ValueError(f"Requested examples [{offset}, {offset + examples}) outside {len(valid)} rows")
    context_size = int(metadata["context_size"])
    items = [valid[i] for i in range(offset, offset + examples)]
    tokens = torch.stack([item["input_ids"] for item in items])
    labels = torch.stack([item["labels"] for item in items])
    if torch.any(labels[:, :context_size] != -100):
        raise ValueError("Dataset supervises context tokens; invalid MQAR WRITE/READ split")
    return {
        "context_input_ids": tokens[:, :context_size],
        "query_input_ids": tokens[:, context_size:],
    }, labels[:, context_size:], metadata


def set_steps(model, steps):
    if steps < 0:
        raise ValueError("WRITE steps must be non-negative")
    model.K = model.config.K = steps
    model.last_K_second_order = model.config.last_K_second_order = 0


def require_prefix(model):
    if model.memory_backend != "prefix" or model.use_mem_proj:
        raise ValueError("Memory analysis requires prefix memory without per-sample projection")


def collect_trajectory(model, inputs, steps):
    """Deterministic eval trajectories; frozen weights, memory gradients still enabled."""
    require_prefix(model)
    state, _ = model.memory_backend_impl.init_memory_state(inputs["context_input_ids"].size(0))
    states = [state["mem_batch"].detach()]
    original = model.K
    try:
        for step in range(1, steps + 1):
            set_steps(model, step)
            states.append(model(inputs, labels=None, return_mem=True)["mem"].detach())
    finally:
        set_steps(model, original)
    return states


def update_singular_values(states):
    """Per-example reduced spectra of consecutive flattened WRITE displacements."""
    if len(states) < 2:
        raise ValueError("Update SVD requires at least one WRITE update")
    dtype = torch.float64 if states[0].dtype == torch.float64 else torch.float32
    updates = torch.stack([(current - previous).detach().to(dtype=dtype).flatten(1)
                           for previous, current in zip(states, states[1:])], dim=1)
    if not torch.isfinite(updates).all():
        raise ValueError("Nonfinite WRITE updates; update SVD aborted")
    values = torch.linalg.svdvals(updates)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite singular values; update SVD aborted")
    return values


def step_projection_ratios(states, proj_steps=2, zero_step_policy="error"):
    """Per-example norm ratio projecting d_k onto its next proj_steps updates."""
    if type(proj_steps) is not int or proj_steps < 1:
        raise ValueError("proj_steps must be a positive integer")
    if zero_step_policy not in ("error", "nan"):
        raise ValueError("zero_step_policy must be error or nan")
    if len(states) < proj_steps + 2:
        raise ValueError(f"Step projection requires at least {proj_steps + 1} WRITE updates")
    updates = torch.stack([(current - previous).detach().double().flatten(1)
                           for previous, current in zip(states, states[1:])], dim=1)
    if not torch.isfinite(updates).all():
        raise ValueError("Nonfinite WRITE updates; step projection aborted")
    norms = updates.norm(dim=-1, keepdim=True)
    zero = norms[:, :-proj_steps, 0] == 0
    if zero_step_policy == "error" and zero.any():
        positions = zero.nonzero()
        raise ValueError("Zero-length current WRITE step has an undefined projection norm ratio; "
                         f"{len(positions)} affected (batch-local example, one-based WRITE step) pairs: "
                         f"{[(int(example), int(step) + 1) for example, step in positions[:20]]}. "
                         "Use a shorter trajectory or explicitly set step_orth_proj.zero_step_policy='nan' "
                         "to report conditional nonzero-step means with coverage counts.")
    unit = updates / torch.where(norms > 0, norms, torch.ones_like(norms))
    # [example, current step, memory dimension, following step].
    basis = unit[:, 1:].unfold(1, proj_steps, 1)
    # Normalize columns so numerical rank does not depend on relative step lengths.
    inverse = torch.linalg.pinv(basis)
    projected = basis @ (inverse @ unit[:, :-proj_steps].unsqueeze(-1))
    ratios = projected.squeeze(-1).norm(dim=-1).clamp(0, 1)
    return ratios.masked_fill(zero, float("nan")) if zero_step_policy == "nan" else ratios


def retained_energy_rank(singular_values, threshold):
    """Smallest one-based rank retaining threshold energy; zero spectrum is undefined."""
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected a finite non-negative singular-value vector")
    if not 0 < threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1]")
    if values.max() == 0:
        return None
    energy = np.cumsum((values / values.max()) ** 2)
    fraction = energy / energy[-1]
    return int(np.searchsorted(fraction, threshold) + 1)


def prediction_metrics(model, predictions, labels, read_batch=None, detach=True):
    losses = model._compute_per_example_read_target_loss(predictions, read_batch or {}, labels)
    # MQAR aligns each answer with its query position, not the preceding token.
    target = predictions[:, 1:] if predictions.size(1) == labels.size(1) + 1 else predictions
    if target.shape[:2] != labels.shape:
        raise ValueError("Prediction/label alignment mismatch")
    mask = labels.ne(-100)
    counts = mask.sum(1)
    if torch.any(counts == 0):
        raise ValueError("An example has no supervised query positions")
    correct = target.argmax(-1).eq(labels) & mask
    return {
        "outer_loss": losses.detach() if detach else losses,
        "token_accuracy": correct.sum(1).float() / counts,
        "exact_match": correct.sum(1).eq(counts).float(),
    }


def score_memory(model, inputs, labels, memory, grad_enabled=False):
    """Per-example inner objective and context-free READ metrics at fixed memory."""
    with torch.set_grad_enabled(grad_enabled):
        require_prefix(model)
        backend = model.memory_backend_impl
        state, _ = backend.init_memory_state(memory.size(0))
        state["mem_batch"] = memory
        batch = backend.prepare_batch(inputs["context_input_ids"], inputs["query_input_ids"], model.model.config.pad_token_id)
        write = backend.build_write_inputs(state, batch)
        with backend.activation_context(state):
            if model.write_objective == "reconstruction":
                output, inner = model._run_reconstruction_write_forward(write)
            elif model.write_objective == "energy":
                output, inner = model._run_energy_write_forward(write)
            else:
                output, ce, energy = model._run_energy_reconstruction_write_forward(write)
                inner = model.write_reconstruction_weight * ce + model.write_energy_weight * energy
        del output
        read = backend.build_read_inputs(state, batch)
        with backend.activation_context(state), model._disable_write_lora():
            output = model.model(inputs_embeds=read["inputs_embeds"], return_dict=True, **read.get("model_kwargs", {}))
        predictions = output.logits[:, read["logits_start"]:read["logits_start"] + read["pred_len"]]
        return dict(inner_loss=inner if grad_enabled else inner.detach(),
                    **prediction_metrics(model, predictions, labels, read, detach=not grad_enabled))


def memory_basis(states, basis_steps):
    a, b = basis_steps
    if not 1 <= a < b < len(states):
        raise ValueError("basis_steps must be two increasing, positive WRITE-step indices")
    center = states[a - 1]
    x = states[a] - center
    y = states[b] - states[b - 1]
    directions = torch.stack((x.double().flatten(1), y.double().flatten(1)), dim=-1)
    finite = torch.stack([torch.isfinite(state).flatten(1).all(1) for state in states]).all(0)
    if not finite.all():
        raise ValueError(f"Nonfinite WRITE trajectory for batch-local examples "
                         f"{(~finite).nonzero().flatten().tolist()}; landscape aborted")
    # Test angular independence, not relative step size. Keep the plotted axes unchanged.
    norms = directions.norm(dim=1)
    unit = directions / torch.where(norms > 0, norms, torch.ones_like(norms)).unsqueeze(1)
    invalid = torch.linalg.matrix_rank(unit) != 2
    if invalid.any():
        raise ValueError(f"Selected WRITE displacements for basis_steps={list(basis_steps)} "
                         f"do not define a two-dimensional plane for batch-local examples "
                         f"{invalid.nonzero().flatten().tolist()}; displacement norms="
                         f"{norms[invalid].tolist()}. Select different basis_steps; "
                         "zero or collinear updates cannot define a 2D landscape.")
    inverse = torch.linalg.pinv(unit)
    coordinates = torch.stack([
        (inverse @ (state.double() - center.double()).flatten(1).unsqueeze(-1)).squeeze(-1) / norms
        for state in states
    ])
    return center, x, y, coordinates.mean(1).cpu().numpy()


def finite_difference_cosine(inner, outer, alpha, beta):
    gradients = [np.stack(np.gradient(values.astype(float), beta, alpha, edge_order=2), axis=-1)
                 for values in (inner, outer)]
    return gradient_cosine(*gradients)


def gradient_cosine(inner, outer):
    denominator = np.linalg.norm(inner, axis=-1) * np.linalg.norm(outer, axis=-1)
    result = np.full(denominator.shape, np.nan)
    valid = denominator > np.finfo(float).eps
    result[valid] = np.clip((inner * outer).sum(-1)[valid] / denominator[valid], -1, 1)
    return result


def gradient_norm(values, alpha, beta, normalize=False):
    values = values.astype(float)
    if normalize:
        span = np.ptp(values)
        values = (values - values.min()) / span if span > 0 else np.zeros_like(values)
    dy, dx = np.gradient(values, beta, alpha, edge_order=2)
    return np.hypot(dx, dy)


def alignment_statistics(cosine):
    values = cosine[np.isfinite(cosine)]
    if not values.size:
        raise ValueError("No valid nonzero-gradient cells for alignment")
    return dict(mean_cosine=float(values.mean()), median_cosine=float(np.median(values)),
                positive_fraction=float((values > 0).mean()),
                negative_fraction=float((values < 0).mean()),
                strong_positive_fraction=float((values >= .5).mean()),
                strong_negative_fraction=float((values <= -.5).mean()),
                valid_cells=int(values.size))


def beneficial_outcomes(base, candidates, quantiles, higher_is_better=False):
    """Metric-wise oracle gains; non-beneficial candidates contribute zero gain."""
    sign = 1 if higher_is_better else -1
    gains = ((candidates - base[None]) * sign).clamp_min(0)
    values = [torch.quantile(gains, q, dim=0) for q in quantiles]
    return torch.stack([base + sign * gain for gain in values])
