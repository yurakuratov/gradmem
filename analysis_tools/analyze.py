"""JSON-driven landscapes, K/lr sweeps, WRITE geometry, and memory-search analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import plots
from .library import (
    REPORTS, alignment_statistics, beneficial_outcomes, collect_trajectory, finite_difference_cosine,
    gradient_cosine, load_data, load_model, memory_basis, prediction_metrics, repo_path,
    require_prefix, resolve_checkpoint, retained_energy_rank, score_memory, set_steps,
    step_projection_ratios, update_singular_values,
)


FONT_SIZES = {
    "subtitle_font_size": ("axes.titlesize", "figure.titlesize"),
    "tick_font_size": ("xtick.labelsize", "ytick.labelsize"),
    "axis_label_font_size": ("axes.labelsize", "figure.labelsize"),
    "legend_font_size": ("legend.fontsize", "legend.title_fontsize"),
}


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def average_runs(arrays, allow_nan=False):
    values = np.stack(arrays)
    if allow_nan:
        counts = np.isfinite(values).sum(0)
        mean = np.divide(np.nansum(values, axis=0), counts,
                         out=np.full(values.shape[1:], np.nan), where=counts > 0)
        squared_error = np.nansum((values - mean) ** 2, axis=0)
        variance = np.divide(squared_error, counts - 1, out=np.zeros_like(mean), where=counts > 1)
        return mean, np.where(counts > 0, np.sqrt(variance), np.nan)
    return values.mean(0), values.std(0, ddof=1 if len(arrays) > 1 else 0)


def landscape_settings(config, setup):
    settings = dict(config.get("landscape", {}))
    settings.update(setup.get("landscape", {}))
    if type(settings.setdefault("real_grads", False)) is not bool:
        raise ValueError("landscape.real_grads must be true or false")
    if type(settings.setdefault("separate_colorbars", False)) is not bool:
        raise ValueError("landscape.separate_colorbars must be true or false")
    if settings["separate_colorbars"] != config.get("landscape", {}).get("separate_colorbars", False):
        raise ValueError("Use shared landscape.separate_colorbars, not setup overrides")
    if type(settings.setdefault("include_cosine_sim", True)) is not bool:
        raise ValueError("landscape.include_cosine_sim must be true or false")
    if settings["include_cosine_sim"] != config.get("landscape", {}).get("include_cosine_sim", True):
        raise ValueError("Use shared landscape.include_cosine_sim, not setup overrides")
    if type(settings.setdefault("report_avg", False)) is not bool:
        raise ValueError("landscape.report_avg must be true or false")
    if settings["report_avg"] != config.get("landscape", {}).get("report_avg", False):
        raise ValueError("Use shared landscape.report_avg, not setup overrides")
    basis = setup.get("basis_steps", settings.get("basis_steps", [1, 2]))
    if (not isinstance(basis, list) or len(basis) != 2
            or any(type(step) is not int for step in basis)
            or not 1 <= basis[0] < basis[1]):
        raise ValueError("basis_steps must be two increasing positive integers")
    settings["basis_steps"] = basis
    # A later basis needs enough states even if an inherited K is smaller.
    steps = max(settings.get("steps", basis[1]), basis[1])
    if type(steps) is not int or steps < basis[1]:
        raise ValueError("Landscape steps must include both selected basis steps")
    settings["steps"] = steps
    return settings


def landscape_run(model, inputs, labels, settings, batch_size):
    real_grads = settings.get("real_grads", False)
    if type(real_grads) is not bool:
        raise ValueError("landscape.real_grads must be true or false")
    basis = settings.get("basis_steps", [1, 2])
    steps = settings.get("steps", max(basis))
    if steps < max(basis):
        raise ValueError("Landscape steps must include both selected basis steps")
    size = settings.get("grid_size", 41)
    if isinstance(size, int):
        size = [size, size]
    if len(size) != 2 or min(size) < 3:
        raise ValueError("grid_size must be >=3 or [alpha_points, beta_points] >=3")
    x_range, y_range = settings.get("alpha_range", [-1, 3]), settings.get("beta_range", [-1, 3])
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError("Landscape axis ranges must be increasing")
    alpha, beta = np.linspace(*x_range, size[0]), np.linspace(*y_range, size[1])
    inner, outer = np.zeros((len(beta), len(alpha))), np.zeros((len(beta), len(alpha)))
    if real_grads:
        inner_grad = np.zeros((*inner.shape, 2))
        outer_grad = np.zeros_like(inner_grad)
    trajectories = []
    for start in range(0, labels.size(0), batch_size):
        batch = {key: value[start:start + batch_size].to(model.device) for key, value in inputs.items()}
        targets = labels[start:start + batch_size].to(model.device)
        states = collect_trajectory(model, batch, steps)
        center, x, y, trajectory = memory_basis(states, basis)
        weight = targets.size(0) / labels.size(0)
        trajectories.append(trajectory * weight)
        for row, b in enumerate(beta):
            for col, a in enumerate(alpha):
                if real_grads:
                    with torch.enable_grad():
                        coordinates = center.new_tensor([a, b], requires_grad=True)
                        memory = center + coordinates[0] * x + coordinates[1] * y
                        metrics = score_memory(model, batch, targets, memory, grad_enabled=True)
                        # Differentiate the example-mean surfaces in the plotted coordinates.
                        inner_grad[row, col] += torch.autograd.grad(
                            metrics["inner_loss"].mean(), coordinates, retain_graph=True)[0].detach().cpu().numpy() * weight
                        outer_grad[row, col] += torch.autograd.grad(
                            metrics["outer_loss"].mean(), coordinates)[0].detach().cpu().numpy() * weight
                else:
                    metrics = score_memory(model, batch, targets, center + float(a) * x + float(b) * y)
                inner[row, col] += float(metrics["inner_loss"].detach().mean()) * weight
                outer[row, col] += float(metrics["outer_loss"].detach().mean()) * weight
    alignment = (gradient_cosine(inner_grad, outer_grad) if real_grads
                 else finite_difference_cosine(inner, outer, alpha, beta))
    return dict(alpha=alpha, beta=beta, inner_loss=inner, outer_loss=outer,
                alignment=alignment,
                trajectory=sum(trajectories))


def extrapolation_run(model, inputs, labels, settings, batch_size):
    steps = settings.get("steps", [0, 1, 2, 4, 8, 16])
    if not steps or any(not isinstance(k, int) or k < 0 for k in steps):
        raise ValueError("Extrapolation steps must be a non-empty list of non-negative integers")
    result = {key: [] for key in ("outer_loss", "token_accuracy", "exact_match")}
    for step in steps:
        set_steps(model, step)
        metrics = {key: [] for key in result}
        for start in range(0, labels.size(0), batch_size):
            batch = {key: value[start:start + batch_size].to(model.device) for key, value in inputs.items()}
            targets = labels[start:start + batch_size].to(model.device)
            output = model(batch, labels=None)
            with torch.no_grad():
                scores = prediction_metrics(model, output["predictions"], targets)
            for key in metrics:
                metrics[key].append(scores[key].cpu())
        for key in result:
            result[key].append(float(torch.cat(metrics[key]).mean()))
    return {key: np.array(values) for key, values in result.items()}


def sweep_settings(config, setup):
    settings = dict(config.get("k_lr_sweep", {}))
    settings.update(setup.get("k_lr_sweep", {}))
    steps, rates = settings.get("steps"), settings.get("inner_lrs")
    if (not isinstance(steps, list) or not steps
            or any(type(k) is not int or k < 0 for k in steps)
            or len(set(steps)) != len(steps)):
        raise ValueError("k_lr_sweep steps must be distinct non-negative integers")
    if (not isinstance(rates, list) or not rates
            or any(type(lr) not in (int, float) or not np.isfinite(lr) or lr <= 0 for lr in rates)
            or len(set(rates)) != len(rates)):
        raise ValueError("k_lr_sweep inner_lrs must be distinct finite positive numbers")
    return settings


def k_lr_sweep_run(model, inputs, labels, settings, batch_size):
    """Independent inference evaluations; matrices have rows K and columns lr."""
    original_lr, original_steps = model.lr, model.K
    results = []
    try:
        for lr in settings["inner_lrs"]:
            model.lr = model.config.lr = lr
            result = extrapolation_run(model, inputs, labels, settings, batch_size)
            if any(not np.isfinite(values).all() for values in result.values()):
                raise ValueError(f"Nonfinite sweep metrics at inner_lr={lr}; sweep aborted")
            results.append(result)
    finally:
        model.lr = model.config.lr = original_lr
        set_steps(model, original_steps)
    return {key: np.stack([result[key] for result in results], axis=1) for key in results[0]}


def update_svd_settings(config):
    settings = dict(config.get("update_svd", {}))
    steps = settings.get("steps")
    threshold = settings.setdefault("energy_threshold", .95)
    if type(steps) is not int or steps < 1:
        raise ValueError("update_svd steps must be an explicit positive integer (total K)")
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("update_svd energy_threshold must be a finite number in (0, 1]")
    return settings


def update_svd_run(model, inputs, labels, settings, batch_size):
    settings = update_svd_settings({"update_svd": settings})
    spectra = []
    for start in range(0, labels.size(0), batch_size):
        batch = {key: value[start:start + batch_size].to(model.device) for key, value in inputs.items()}
        states = collect_trajectory(model, batch, settings["steps"])
        spectra.append(update_singular_values(states).cpu().numpy())
    values = np.concatenate(spectra)
    return dict(singular_values=values.mean(0), singular_values_per_example=values)


def step_orth_proj_settings(config):
    settings = dict(config.get("step_orth_proj", {}))
    if type(settings.get("steps")) is not int or settings["steps"] < 1:
        raise ValueError("step_orth_proj steps must be an explicit positive integer (K plotted points)")
    value = settings.setdefault("proj_steps", 2)
    spans = value if isinstance(value, list) else [value]
    if value is None:
        if not (settings.get("add_step_norm", False) or settings.get("include_cosine_sim", False)):
            raise ValueError("proj_steps=null requires add_step_norm or include_cosine_sim")
    elif (not spans or any(type(span) is not int or span < 1 for span in spans)
            or len(set(spans)) != len(spans)):
        raise ValueError("step_orth_proj proj_steps must be a positive integer or a nonempty list of distinct positive integers")
    if settings.get("zero_step_policy", "error") not in ("error", "nan"):
        raise ValueError("step_orth_proj zero_step_policy must be error or nan")
    if type(settings.get("add_step_norm", False)) is not bool:
        raise ValueError("step_orth_proj add_step_norm must be true or false")
    if type(settings.get("include_cosine_sim", False)) is not bool:
        raise ValueError("step_orth_proj include_cosine_sim must be true or false")
    return settings


def step_orth_proj_run(model, inputs, labels, settings, batch_size):
    settings = step_orth_proj_settings({"step_orth_proj": settings})
    value = settings["proj_steps"]
    spans = [] if value is None else value if isinstance(value, list) else [value]
    ratios, norms, cosines = [], [], []
    for start in range(0, labels.size(0), batch_size):
        batch = {key: value[start:start + batch_size].to(model.device) for key, value in inputs.items()}
        states = collect_trajectory(model, batch, settings["steps"] + max(spans, default=int(settings.get("include_cosine_sim", False))))
        policy = {"zero_step_policy": settings["zero_step_policy"]} if "zero_step_policy" in settings else {}
        projected = [step_projection_ratios(states[:settings["steps"] + span + 1], span, **policy) for span in spans]
        ratios.append((torch.stack(projected, dim=1) if isinstance(value, list) else projected[0]).cpu().numpy()
                      if projected else np.empty((len(batch["context_input_ids"]), 0, settings["steps"])))
        updates = torch.stack([
            (current - previous).detach().double().flatten(1)
            for previous, current in zip(states[:settings["steps"] + 1], states[1:settings["steps"] + 2])
        ], dim=1)
        if not torch.isfinite(updates).all():
            raise ValueError("Nonfinite WRITE updates; step analysis aborted")
        lengths = updates.norm(dim=-1)
        norms.append(lengths[:, :settings["steps"]].cpu().numpy())
        if settings.get("include_cosine_sim", False):
            valid = (lengths[:, :-1] > 0) & (lengths[:, 1:] > 0)
            if not valid.all() and settings.get("zero_step_policy", "error") == "error":
                raise ValueError("Zero-length current or next WRITE step has an undefined cosine similarity; "
                                 "use zero_step_policy='nan' to omit undefined pairs with coverage counts")
            unit = updates / torch.where(lengths > 0, lengths, torch.ones_like(lengths)).unsqueeze(-1)
            cosine = (unit[:, :-1] * unit[:, 1:]).sum(-1).clamp(-1, 1)
            cosines.append(cosine.masked_fill(~valid, float("nan")).cpu().numpy())
    values = np.concatenate(ratios)
    if settings.get("zero_step_policy", "error") == "nan":
        counts = np.isfinite(values).sum(0)
        mean = np.divide(np.nansum(values, axis=0), counts,
                         out=np.full(values.shape[1:], np.nan), where=counts > 0)
        if np.isnan(values).any():
            print(f"Warning: {np.isnan(values).sum()} zero-step projection ratios are undefined; "
                  "reporting conditional nonzero-step means. Valid/total example counts are saved.", flush=True)
        result = dict(projection_ratio=mean, projection_ratio_per_example=values,
                      projection_valid_examples=counts, projection_total_examples=np.full_like(counts, len(values)))
    else:
        result = dict(projection_ratio=values.mean(0), projection_ratio_per_example=values)
    norms = np.concatenate(norms)
    result.update(step_norm=norms.mean(0), step_norm_per_example=norms)
    if settings.get("include_cosine_sim", False):
        values = np.concatenate(cosines)
        counts = np.isfinite(values).sum(0)
        mean = np.divide(np.nansum(values, axis=0), counts,
                         out=np.full(values.shape[1:], np.nan), where=counts > 0)
        if np.isnan(values).any():
            print("Warning: undefined consecutive-step cosines omitted; valid/total pair counts are saved.", flush=True)
        result.update(cosine_sim=mean, cosine_sim_per_example=values,
                      cosine_valid_examples=counts, cosine_total_examples=np.full_like(counts, len(values)))
    return result


def memory_search_run(model, inputs, labels, settings, batch_size):
    require_prefix(model)
    steps = settings.get("steps", 2)
    samples = settings.get("num_samples", 4)
    radius = settings.get("radius_scale", .25)
    quantiles = settings.get("quantiles", [.5, .75, 1.0])
    if steps < 1 or samples < 1 or radius < 0:
        raise ValueError("Memory search needs steps>=1, num_samples>=1, radius_scale>=0")
    if not quantiles or any(q < 0 or q > 1 for q in quantiles) or quantiles != sorted(set(quantiles)):
        raise ValueError("quantiles must be sorted distinct numbers in [0,1]")
    model.energy_memory_search_num_samples = samples
    model.energy_memory_search_radius_scale = radius
    baseline, candidates = {}, {}
    for start in range(0, labels.size(0), batch_size):
        batch = {key: value[start:start + batch_size].to(model.device) for key, value in inputs.items()}
        targets = labels[start:start + batch_size].to(model.device)
        states = collect_trajectory(model, batch, steps)
        memories = model._sample_norm_preserving_memory_candidates(states[-1], states[-2])
        base = score_memory(model, batch, targets, states[-1])
        candidate_scores = [score_memory(model, batch, targets, memory) for memory in memories]
        for key in base:
            baseline.setdefault(key, []).append(base[key].cpu())
            candidates.setdefault(key, []).append(torch.stack([value[key].cpu() for value in candidate_scores]))
    result = {}
    for key in baseline:
        base = torch.cat(baseline[key])
        candidate = torch.cat(candidates[key], dim=1)
        outcomes = beneficial_outcomes(base, candidate, quantiles, key in ("token_accuracy", "exact_match"))
        result[key + "_baseline"] = np.array(float(base.mean()))
        result[key + "_outcomes"] = outcomes.mean(1).numpy()
        result[key + "_baseline_per_example"] = base.numpy()
        result[key + "_candidates"] = candidate.numpy()
    return result


def validate_config(config):
    if config.get("precision", "fp32") not in ("fp32", "fp64"):
        raise ValueError("precision must be fp32 or fp64")
    if config.get("kind") not in ("landscape", "extrapolation", "memory_search", "k_lr_sweep", "update_svd", "step_orth_proj"):
        raise ValueError("kind must be landscape, extrapolation, memory_search, k_lr_sweep, update_svd, or step_orth_proj")
    if config["kind"] == "landscape":
        if type(config.get("landscape", {}).get("report_avg", False)) is not bool:
            raise ValueError("landscape.report_avg must be true or false")
    figsize = config.get("figsize")
    if figsize is not None and (not isinstance(figsize, list) or len(figsize) != 2 or any(
        type(value) not in (int, float) or not np.isfinite(value) or value <= 0 for value in figsize
    )):
        raise ValueError("figsize must be null or [width, height] with finite positive dimensions in inches")
    dpi = config.get("dpi", 220)
    if type(dpi) not in (int, float) or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a finite positive number")
    for key in FONT_SIZES:
        value = config.get(key)
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be null or a finite positive number in points")
    kind = config["kind"]
    if type(config.get(kind, {}).get("show_n_runs", kind not in ("extrapolation", "memory_search"))) is not bool:
        raise ValueError(f"{kind}.show_n_runs must be true or false")
    if not config.get("setups"):
        raise ValueError("At least one setup is required")
    labels = [setup["label"] for setup in config["setups"]]
    if len(set(labels)) != len(labels):
        raise ValueError("Setup labels must be unique")
    if any(not setup.get("runs") for setup in config["setups"]):
        raise ValueError("Every setup requires at least one explicit run")
    if any("precision" in setup or "precision" in setup.get(config["kind"], {}) or any(
        isinstance(run, dict) and "precision" in run for run in setup["runs"]
    ) for setup in config["setups"]) or "precision" in config.get(config["kind"], {}):
        raise ValueError("Use shared top-level precision, not kind/setup/run overrides")
    if any(isinstance(run, dict) and "selector" in run
           for setup in config["setups"] for run in setup["runs"]):
        raise ValueError("selector is not supported; provide the checkpoint folder directly in path")
    if config.get("batch_size", 64) < 1 or config.get("examples", 256) < 1:
        raise ValueError("examples and batch_size must be positive")
    if config["kind"] in ("extrapolation", "step_orth_proj"):
        settings = config.get(config["kind"], {})
        overrides = settings.get("dash_start_by_setup", {})
        if not isinstance(overrides, dict) or any(label not in labels for label in overrides):
            raise ValueError(f"{config['kind']}.dash_start_by_setup must be an object keyed by existing setup labels")
        steps = (settings.get("steps", [0, 1, 2, 4, 8, 16]) if config["kind"] == "extrapolation"
                 else range(1, step_orth_proj_settings(config)["steps"] + 1))
        for start in overrides.values():
            if start is not None and (type(start) is not int or start < 0 or start not in steps):
                raise ValueError(f"{config['kind']}.dash_start_by_setup values must be null or an integer in plotted steps")
        ends = settings.get("dash_end_by_setup", {})
        if not isinstance(ends, dict) or any(label not in labels for label in ends):
            raise ValueError(f"{config['kind']}.dash_end_by_setup must be an object keyed by existing setup labels")
        for label, end in ends.items():
            start = overrides.get(label, settings.get("dash_start"))
            if end is not None and (type(end) is not int or end not in steps
                                    or type(start) is not int or end <= start):
                raise ValueError(f"{config['kind']}.dash_end_by_setup values must be null or a plotted integer step after dash_start")
        if config["kind"] == "extrapolation" and any(start is not None for start in overrides.values()):
            if any(a >= b for a, b in zip(steps, steps[1:])):
                raise ValueError("extrapolation.steps must be strictly increasing when dash_start_by_setup is set")
        if settings.get("uncertainty_style", "errorbar") not in ("errorbar", "band"):
            raise ValueError(f"{config['kind']}.uncertainty_style must be errorbar or band")
        if type(settings.get("show_runs", False)) is not bool:
            raise ValueError(f"{config['kind']}.show_runs must be true or false")
    if config["kind"] == "extrapolation":
        settings = config.get("extrapolation", {})
        reference = settings.get("outer_loss_reference")
        if reference is not None and (type(reference) not in (int, float)
                                     or not np.isfinite(reference) or reference <= 0):
            raise ValueError("extrapolation.outer_loss_reference must be null or a finite positive number")
        dash_start = settings.get("dash_start")
        if dash_start is not None:
            steps = settings.get("steps", [0, 1, 2, 4, 8, 16])
            if type(dash_start) is not int or dash_start < 0 or dash_start not in steps:
                raise ValueError("extrapolation.dash_start must be null or a non-negative integer in steps")
            if any(a >= b for a, b in zip(steps, steps[1:])):
                raise ValueError("extrapolation.steps must be strictly increasing when dash_start is set")
    if config["kind"] in ("step_orth_proj", "update_svd"):
        kind = config["kind"]
        settings = step_orth_proj_settings(config) if kind == "step_orth_proj" else update_svd_settings(config)
        dash_start = settings.get("dash_start")
        if dash_start is not None and (type(dash_start) is not int or not 1 <= dash_start <= settings["steps"]):
            raise ValueError(f"{kind}.dash_start must be null or an integer between 1 and steps")
    if config["kind"] == "step_orth_proj":
        step_orth_proj_settings(config)
        for setup in config["setups"]:
            if setup.get("step_orth_proj", {}) or any(
                isinstance(run, dict) and "step_orth_proj" in run for run in setup["runs"]
            ):
                raise ValueError("Use shared step_orth_proj settings, not setup/run overrides")
    if config["kind"] == "update_svd":
        update_svd_settings(config)
        for setup in config["setups"]:
            if setup.get("update_svd", {}) or any(
                isinstance(run, dict) and "update_svd" in run for run in setup["runs"]
            ):
                raise ValueError("Use shared update_svd settings, not setup/run overrides")
    if config["kind"] == "k_lr_sweep":
        if "inner_lr" in config:
            raise ValueError("Use k_lr_sweep.inner_lrs, not scalar inner_lr overrides")
        for setup in config["setups"]:
            sweep_settings(config, setup)
            if "inner_lr" in setup or any(
                isinstance(run, dict) and "inner_lr" in run for run in setup["runs"]
            ):
                raise ValueError("Use k_lr_sweep.inner_lrs, not scalar inner_lr overrides")
            if any(isinstance(run, dict) and "k_lr_sweep" in run for run in setup["runs"]):
                raise ValueError("Put k_lr_sweep settings on the setup, not inside runs")


def draw_plot(path, config, summaries, settings, alpha=None, beta=None):
    """Common rendering path for fresh analyses and saved-report redraws."""
    kind, dpi = config["kind"], config.get("dpi", 220)
    size = {"figsize": config["figsize"]} if config.get("figsize") is not None else {}
    fonts = {parameter: config[key] for key, parameters in FONT_SIZES.items()
             if config.get(key) is not None for parameter in parameters}
    with plots.plt.rc_context(fonts):
        if kind == "landscape":
            plots.landscapes(path, summaries, alpha, beta, settings.get("basis_steps", [1, 2]),
                             dpi, settings.get("log_scale", True),
                              separate_colorbars=settings.get("separate_colorbars", False),
                              include_cosine_sim=settings.get("include_cosine_sim", True),
                               show_n_runs=settings.get("show_n_runs", True),
                               **({"report_avg": settings["report_avg"]} if "report_avg" in settings else {}), **size)
        elif kind == "extrapolation":
            plots.extrapolation(path, summaries, settings.get("steps", [0, 1, 2, 4, 8, 16]), dpi,
                                dash_start=settings.get("dash_start"), show_n_runs=settings.get("show_n_runs", False),
                                 **{key: settings[key] for key in ("uncertainty_style", "show_runs", "dash_start_by_setup", "dash_end_by_setup", "outer_loss_reference") if key in settings}, **size)
        elif kind == "k_lr_sweep":
            plots.k_lr_sweep(path, summaries, dpi, show_n_runs=settings.get("show_n_runs", True), **size)
        elif kind == "update_svd":
            plots.update_svd(path, summaries, settings["steps"], settings.get("energy_threshold", .95), dpi,
                             dash_start=settings.get("dash_start"), show_n_runs=settings.get("show_n_runs", True), **size)
        elif kind == "step_orth_proj":
            plots.step_orth_proj(path, summaries, settings["steps"], dpi, proj_steps=settings.get("proj_steps", 2),
                                 dash_start=settings.get("dash_start"), show_n_runs=settings.get("show_n_runs", True),
                                  **{key: settings[key] for key in ("uncertainty_style", "show_runs", "zero_step_policy", "dash_start_by_setup", "dash_end_by_setup")
                                    if key in settings}, add_step_norm=settings.get("add_step_norm", False),
                                 **({"include_cosine_sim": settings["include_cosine_sim"]} if "include_cosine_sim" in settings else {}), **size)
        else:
            plots.memory_search(path, summaries, settings.get("quantiles", [.5, .75, 1.0]), dpi,
                                show_n_runs=settings.get("show_n_runs", False), **size)


def run_analysis(config):
    validate_config(config)
    output = repo_path(config["output_dir"])
    if not output.is_relative_to(REPORTS):
        raise ValueError("output_dir must live under reports/")
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}; choose a new output_dir")
    datasets = config.get("datasets", [config.get("dataset")])
    if not datasets or any(not value for value in datasets):
        raise ValueError("Specify dataset or datasets")
    device = torch.device(config.get("device", "cpu"))
    precision = config.get("precision", "fp32")
    dtype = {"fp32": torch.float32, "fp64": torch.float64}[precision]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        # Launch remains explicit: never pick an arbitrary occupied GPU automatically.
        import os
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Set CUDA_VISIBLE_DEVICES after inspecting nvidia-smi")
    resolved = []
    for index, setup in enumerate(config["setups"]):
        entries = []
        for run in setup["runs"]:
            spec = {"path": run} if isinstance(run, str) else run
            if config["kind"] == "landscape" and "basis_steps" in spec:
                raise ValueError("Put basis_steps on the setup, not inside runs")
            checkpoint = resolve_checkpoint(spec["path"])
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            run_config = checkpoint.parent.parent / "config.json"
            training_seed = (json.loads(run_config.read_text()).get("cli_args", {}).get("seed")
                             if run_config.is_file() else None)
            entry = dict(path=str(checkpoint), sha256=digest.hexdigest(), seed=training_seed,
                         inner_lr=spec.get("inner_lr", setup.get("inner_lr")))
            if config["kind"] == "landscape":
                entry["landscape"] = landscape_settings(config, setup)
            elif config["kind"] == "k_lr_sweep":
                entry["k_lr_sweep"] = sweep_settings(config, setup)
            elif config["kind"] == "update_svd":
                entry["update_svd"] = update_svd_settings(config)
            elif config["kind"] == "step_orth_proj":
                entry["step_orth_proj"] = step_orth_proj_settings(config)
            entries.append(entry)
        if len({entry["path"] for entry in entries}) != len(entries):
            raise ValueError(f"Duplicate checkpoint within setup {setup['label']}")
        resolved.append(dict(label=setup["label"], runs=entries, index=index))
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = dict(setups=resolved, precision=precision, examples=config.get("examples", 256),
                      example_offset=config.get("example_offset", 0),
                      example_indices=list(range(config.get("example_offset", 0),
                          config.get("example_offset", 0) + config.get("examples", 256))),
                       alignment_aggregation="per-run cosine maps, then equal run average",
                        memory_search_semantics="metric-wise beneficial oracle; harmful candidates clamped to zero gain")
    if config["kind"] == "step_orth_proj":
        projection_settings = step_orth_proj_settings(config)
        spans = projection_settings["proj_steps"]
        provenance.update(
            step_orth_proj=projection_settings,
            trajectory_steps=projection_settings["steps"] + (int(projection_settings.get("include_cosine_sim", False))
                if spans is None else max(spans) if isinstance(spans, list) else spans),
            projection="norm(Proj_span(d_(k+1),...,d_(k+P)) d_k) / norm(d_k); d_k=M_k-M_(k-1); P=proj_steps",
            aggregation="per-example ratios, then example mean per run, then equal run mean and sample SD",
            degeneracy="following steps use numerical span; zero current step is undefined and rejected",
            projection_rtol="max(memory_dimension, proj_steps) * float64 eps; unit-normalized basis columns",
            step_norm="L2 norm of each stored WRITE displacement, averaged over all examples including zero steps; equal-run mean and sample SD; independent of projection span",
            cosine_sim="When enabled: signed per-example cosine(d_k,d_(k+1)); example mean per run, equal-run mean/sample SD; zero pairs follow zero_step_policy",
            trajectory_collection="replay WRITE with K=1..trajectory_steps; reuse trajectory across requested metrics/spans; quadratic WRITE cost",
            run_counts={setup["label"]: len(setup["runs"]) for setup in resolved})
        if projection_settings.get("zero_step_policy", "error") == "nan":
            provenance.update(
                zero_step_policy="Undefined zero-current-step ratios are NaN; no fabricated zero/one ratios",
                degeneracy="Zero current steps are missing values with explicit counts; nonfinite updates are rejected",
                aggregation="conditional mean over nonzero-current-step examples per run; equal mean/sample SD over valid runs per point",
                coverage="Valid/total examples saved per run/span/step; valid run counts saved per setup/span/step")
        if len({len(setup["runs"]) for setup in resolved}) > 1:
            print("Warning: setups have unequal run coverage; see manifest run_counts", flush=True)
    if config["kind"] == "update_svd":
        provenance.update(
            update_svd=update_svd_settings(config),
            update_matrix="per example: rows flatten(M_k - M_(k-1)), k=1..K; raw reduced SVD",
            aggregation="example-mean spectra per run, then equal run mean and sample SD",
            rank_marker="smallest rank retaining energy_threshold of squared displayed mean spectrum; zero undefined",
            trajectory_collection="replay WRITE with K=1..total K; quadratic WRITE cost",
            run_counts={setup["label"]: len(setup["runs"]) for setup in resolved})
        if len({len(setup["runs"]) for setup in resolved}) > 1:
            print("Warning: setups have unequal run coverage; see manifest run_counts", flush=True)
    if config["kind"] == "k_lr_sweep":
        provenance.update(matrix_axes=["steps", "inner_lrs"],
                          aggregation="equal run mean and sample standard deviation; single-run std is zero",
                          sweep_semantics="fixed-checkpoint inference; absolute learning rates; independent WRITE per cell",
                          run_counts={setup["label"]: len(setup["runs"]) for setup in resolved})
        if len({len(setup["runs"]) for setup in resolved}) > 1:
            print("Warning: setups have unequal run coverage; see manifest run_counts", flush=True)
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    settings = (update_svd_settings(config) if config["kind"] == "update_svd"
                else step_orth_proj_settings(config) if config["kind"] == "step_orth_proj"
                else config.get(config["kind"], {}))
    dataset_names = set()
    for dataset in datasets:
        dataset = {"path": dataset, "label": Path(dataset).name} if isinstance(dataset, str) else dataset
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", dataset["label"])
        if name in dataset_names:
            raise ValueError("Dataset labels must yield distinct report folder names")
        dataset_names.add(name)
        folder = output / name
        folder.mkdir()
        inputs, targets, metadata = load_data(dataset["path"], config.get("examples", 256), config.get("example_offset", 0))
        (folder / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summaries, rows = [], []
        for setup, resolution in zip(config["setups"], resolved):
            effective_settings = dict(settings)
            effective_settings.update(setup.get(config["kind"], {}))
            if config["kind"] not in ("landscape", "k_lr_sweep") and effective_settings != settings:
                raise ValueError("Use shared extrapolation/search settings; vary inner_lr per setup or run")
            if config["kind"] == "landscape" and any(
                effective_settings.get(key) != settings.get(key)
                for key in ("alpha_range", "beta_range", "grid_size")
            ):
                raise ValueError("Compared landscapes must share axis ranges and grid_size")
            runs = []
            trained_settings = []
            for run_index, entry in enumerate(resolution["runs"]):
                run_settings = entry.get(config["kind"], effective_settings)
                print(f"{config['kind']} {name} | {setup['label']} | {entry['path']}", flush=True)
                torch.manual_seed(config.get("seed", 0))
                checkpoint = Path(entry["path"])
                model = load_model(checkpoint, device, entry["inner_lr"], dtype)
                if config["kind"] == "k_lr_sweep":
                    trained_settings.append(dict(K=model.K, inner_lr=model.lr))
                function = {"landscape": landscape_run, "extrapolation": extrapolation_run,
                             "memory_search": memory_search_run,
                             "update_svd": update_svd_run,
                             "step_orth_proj": step_orth_proj_run,
                             "k_lr_sweep": k_lr_sweep_run}[config["kind"]]
                data = function(model, inputs, targets, run_settings, config.get("batch_size", 64))
                run_folder = folder / f"setup_{resolution['index']}"
                run_folder.mkdir(exist_ok=True)
                filename = run_folder / f"run_{run_index}.npz"
                axes = (dict(steps=np.array(run_settings["steps"]), inner_lrs=np.array(run_settings["inner_lrs"]))
                        if config["kind"] == "k_lr_sweep" else {})
                if config["kind"] == "update_svd":
                    axes = dict(ranks=np.arange(1, len(data["singular_values"]) + 1))
                elif config["kind"] == "step_orth_proj":
                    axes = dict(steps=np.arange(1, run_settings["steps"] + 1))
                    if isinstance(run_settings["proj_steps"], list) or run_settings["proj_steps"] is None:
                        axes["proj_steps"] = np.array(run_settings["proj_steps"] or [], dtype=int)
                np.savez_compressed(filename, **data, **axes)
                details = dict(checkpoint=entry["path"], effective_inner_lr=model.lr,
                                precision=precision,
                               seed=entry["seed"],
                               write_objective=model.write_objective,
                               file=str(filename), config=model.config.to_dict(),
                                analysis_settings=run_settings,
                                original_config=json.loads((checkpoint.parent / "config.json").read_text()))
                if config["kind"] == "k_lr_sweep":
                    details["trained_settings"] = trained_settings[-1]
                    details["matrix_axes"] = ["steps", "inner_lrs"]
                (run_folder / f"run_{run_index}.json").write_text(json.dumps(details, indent=2, default=str) + "\n")
                if config["kind"] == "landscape":
                    rows.append(dict(setup=setup["label"], run=run_index, checkpoint=entry["path"],
                                     **alignment_statistics(data["alignment"])))
                elif config["kind"] == "extrapolation":
                    for position, step in enumerate(settings.get("steps", [0, 1, 2, 4, 8, 16])):
                        rows.append(dict(setup=setup["label"], run=run_index, checkpoint=entry["path"], K=step,
                                         **{key: float(value[position]) for key, value in data.items()}))
                elif config["kind"] == "k_lr_sweep":
                    for row, step in enumerate(run_settings["steps"]):
                        for col, lr in enumerate(run_settings["inner_lrs"]):
                            rows.append(dict(setup=setup["label"], run=run_index, checkpoint=entry["path"],
                                             K=step, inner_lr=lr,
                                              **{key: float(value[row, col]) for key, value in data.items()}))
                elif config["kind"] == "update_svd":
                    rank = retained_energy_rank(data["singular_values"], run_settings["energy_threshold"])
                    for index, value in enumerate(data["singular_values"]):
                        rows.append(dict(setup=setup["label"], run=run_index, checkpoint=entry["path"],
                                         K=run_settings["steps"], inner_lr=model.lr, rank=index + 1,
                                         singular_value=float(value), energy_threshold=run_settings["energy_threshold"],
                                          retained_rank=rank))
                elif config["kind"] == "step_orth_proj":
                    spans = run_settings["proj_steps"]
                    spans = spans if isinstance(spans, list) else [spans]
                    curves = (np.full((1, settings["steps"]), np.nan) if run_settings["proj_steps"] is None
                              else np.atleast_2d(data["projection_ratio"]))
                    for panel, (span, values) in enumerate(zip(spans, curves)):
                        for index, value in enumerate(values):
                            coverage = (dict(
                                valid_examples=int(np.atleast_2d(data["projection_valid_examples"])[panel, index]),
                                total_examples=int(np.atleast_2d(data["projection_total_examples"])[panel, index]))
                                if span is not None and "projection_valid_examples" in data else {})
                            rows.append(dict(setup=setup["label"], run=run_index, checkpoint=entry["path"],
                                             step=index + 1, proj_steps=span,
                                              inner_lr=model.lr, **(dict(projection_ratio=float(value)) if span is not None else {}), **coverage,
                                             **(dict(step_norm=float(data["step_norm"][index])) if "step_norm" in data else {}),
                                             **(dict(cosine_sim=float(data["cosine_sim"][index]),
                                                     cosine_valid_examples=int(data["cosine_valid_examples"][index]),
                                                     cosine_total_examples=int(data["cosine_total_examples"][index]))
                                                if "cosine_sim" in data else {})))
                runs.append(data)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            summary = dict(label=setup["label"], runs=len(runs))
            if config["kind"] == "k_lr_sweep":
                summary.update(steps=run_settings["steps"], inner_lrs=run_settings["inner_lrs"],
                               trained_settings=trained_settings)
            if config["kind"] == "landscape":
                summary["basis_steps"] = resolution["runs"][0]["landscape"]["basis_steps"]
                for key in ("inner_loss", "outer_loss"):
                    summary[key], summary[key + "_std"] = average_runs([run[key] for run in runs])
                trajectories = np.full((len(runs), max(len(run["trajectory"]) for run in runs), 2), np.nan)
                for index, run in enumerate(runs):
                    trajectories[index, :len(run["trajectory"])] = run["trajectory"]
                summary["trajectory"] = np.nanmean(trajectories, axis=0)
                summary["trajectory_runs"] = np.isfinite(trajectories[:, :, 0]).sum(axis=0)
                squared_error = np.nansum((trajectories - summary["trajectory"]) ** 2, axis=0)
                counts = summary["trajectory_runs"][:, None]
                summary["trajectory_std"] = np.sqrt(np.divide(
                    squared_error, counts - 1, out=np.zeros_like(squared_error), where=counts > 1))
                # Never compute cosine from averaged loss grids.
                maps = np.stack([run["alignment"] for run in runs])
                valid_count = np.isfinite(maps).sum(0)
                summary["alignment"] = np.divide(np.nansum(maps, axis=0), valid_count,
                                                  out=np.full(maps.shape[1:], np.nan), where=valid_count > 0)
                summary["mean_cosine"] = float(np.mean([row["mean_cosine"] for row in rows
                                                       if row["setup"] == setup["label"]]))
            else:
                if config["kind"] == "update_svd" and len({len(run["singular_values"]) for run in runs}) != 1:
                    raise ValueError("Compared runs within a setup must share reduced spectrum length")
                for key in runs[0]:
                    if not key.endswith(("_candidates", "_per_example")):
                        conditional = (config["kind"] == "step_orth_proj" and key in ("projection_ratio", "cosine_sim")
                                       and settings.get("zero_step_policy", "error") == "nan")
                        mean, std = average_runs([run[key] for run in runs], **({"allow_nan": True} if conditional else {}))
                        summary[key + "_mean"], summary[key + "_std"] = mean, std
                        summary[key] = mean
                if config["kind"] == "step_orth_proj" and settings.get("zero_step_policy", "error") == "nan":
                    summary["projection_valid_runs"] = np.isfinite(np.stack([run["projection_ratio"] for run in runs])).sum(0)
                if config["kind"] == "step_orth_proj" and "cosine_sim" in runs[0]:
                    summary["cosine_valid_runs"] = np.isfinite(np.stack([run["cosine_sim"] for run in runs])).sum(0)
            if config["kind"] == "update_svd":
                summary["retained_rank"] = retained_energy_rank(summary["singular_values"], settings["energy_threshold"])
                summary["ranks"] = axes["ranks"]
            np.savez_compressed(folder / f"setup_{resolution['index']}_mean.npz",
                                **{key: value for key, value in summary.items() if isinstance(value, np.ndarray) and key not in axes},
                                **axes)
            if config["kind"] == "extrapolation" and settings.get("show_runs", False):
                for metric in ("outer_loss", "token_accuracy", "exact_match"):
                    summary[metric + "_runs"] = np.stack([run[metric] for run in runs])
            elif config["kind"] == "step_orth_proj" and settings.get("show_runs", False):
                summary["projection_ratio_runs"] = np.stack([run["projection_ratio"] for run in runs])
                if settings.get("add_step_norm", False):
                    summary["step_norm_runs"] = np.stack([run["step_norm"] for run in runs])
                if settings.get("include_cosine_sim", False):
                    summary["cosine_sim_runs"] = np.stack([run["cosine_sim"] for run in runs])
            summaries.append(summary)
        write_csv(folder / "per_run.csv", rows)
        stem = "landscapes" if config["kind"] == "landscape" else config["kind"]
        draw_plot(folder / f"{stem}.png", config, summaries, settings,
                  runs[0].get("alpha"), runs[0].get("beta"))
        if config["kind"] == "landscape":
            statistics = []
            for setup in summaries:
                selected = [row for row in rows if row["setup"] == setup["label"]]
                statistics.append(dict(setup=setup["label"], runs=setup["runs"], **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in alignment_statistics(runs[0]["alignment"])
                }))
            write_csv(folder / "alignment_summary.csv", statistics)
        elif config["kind"] == "extrapolation":
            write_csv(folder / "summary.csv", [dict(setup=setup["label"], runs=setup["runs"], K=step,
                **{key: float(setup[key][index]) for key in setup if key.endswith(("_mean", "_std"))})
                for setup in summaries for index, step in enumerate(settings.get("steps", [0, 1, 2, 4, 8, 16]))])
        elif config["kind"] == "k_lr_sweep":
            write_csv(folder / "summary.csv", [dict(setup=setup["label"], runs=setup["runs"], K=step,
                inner_lr=lr, **{key: float(setup[key][row, col]) for key in setup
                               if key.endswith(("_mean", "_std"))})
                for setup in summaries for row, step in enumerate(setup["steps"])
                for col, lr in enumerate(setup["inner_lrs"])])
        elif config["kind"] == "update_svd":
            write_csv(folder / "summary.csv", [dict(
                setup=setup["label"], runs=setup["runs"], K=settings["steps"], rank=index + 1,
                singular_value_mean=float(setup["singular_values_mean"][index]),
                singular_value_std=float(setup["singular_values_std"][index]),
                energy_threshold=settings["energy_threshold"], retained_rank=setup["retained_rank"])
                 for setup in summaries for index in range(len(setup["singular_values"]))])
        elif config["kind"] == "step_orth_proj":
            spans = settings["proj_steps"]
            spans = spans if isinstance(spans, list) else [spans]
            write_csv(folder / "summary.csv", [dict(
                setup=setup["label"], runs=setup["runs"], step=index + 1,
                proj_steps=span,
                **(dict(projection_ratio_mean=float(np.atleast_2d(setup["projection_ratio_mean"])[panel, index]),
                        projection_ratio_std=float(np.atleast_2d(setup["projection_ratio_std"])[panel, index]))
                   if span is not None else {}),
                **(dict(step_norm_mean=float(setup["step_norm_mean"][index]),
                        step_norm_std=float(setup["step_norm_std"][index])) if "step_norm_mean" in setup else {}),
                **(dict(cosine_sim_mean=float(setup["cosine_sim_mean"][index]),
                        cosine_sim_std=float(setup["cosine_sim_std"][index]),
                        cosine_valid_runs=int(setup["cosine_valid_runs"][index]),
                        cosine_valid_examples_mean=float(setup["cosine_valid_examples"][index]),
                        cosine_total_examples_mean=float(setup["cosine_total_examples"][index]))
                   if "cosine_sim_mean" in setup else {}),
                **(dict(valid_runs=int(np.atleast_2d(setup["projection_valid_runs"])[panel, index]),
                        valid_examples_mean=float(np.atleast_2d(setup["projection_valid_examples"])[panel, index]),
                        total_examples_mean=float(np.atleast_2d(setup["projection_total_examples"])[panel, index]))
                   if span is not None and "projection_valid_runs" in setup else {}))
                for setup in summaries for panel, span in enumerate(spans) for index in range(settings["steps"])])
        else:
            write_csv(folder / "summary.csv", [dict(setup=setup["label"], metric=key,
                baseline=float(setup[key + "_baseline"]), quantile=q,
                beneficial_outcome=float(setup[key + "_outcomes"][index]))
                for setup in summaries for key in ("inner_loss", "outer_loss", "token_accuracy", "exact_match")
                for index, q in enumerate(settings.get("quantiles", [.5, .75, 1.0]))])
    print(f"Reports saved to {output}", flush=True)
    return output


def redraw_analysis(config):
    """Render saved aggregate results without loading data, checkpoints, or models."""
    output = repo_path(config["output_dir"])
    if not output.is_relative_to(REPORTS):
        raise ValueError("output_dir must live under reports/")
    saved = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    kind = config.get("kind", saved["kind"])
    if kind != saved["kind"]:
        raise ValueError(f"Redraw kind {kind!r} is incompatible with saved kind {saved['kind']!r}")
    options = {
        "landscape": ("log_scale", "separate_colorbars", "include_cosine_sim", "show_n_runs", "report_avg"),
        "extrapolation": ("dash_start", "dash_start_by_setup", "dash_end_by_setup", "outer_loss_reference", "show_n_runs", "uncertainty_style", "show_runs"),
        "step_orth_proj": ("dash_start", "dash_start_by_setup", "dash_end_by_setup", "show_n_runs", "uncertainty_style", "show_runs", "add_step_norm", "include_cosine_sim"),
        "update_svd": ("dash_start", "show_n_runs"),
        "k_lr_sweep": ("show_n_runs",), "memory_search": ("show_n_runs",),
    }
    if kind not in options:
        raise ValueError(f"Unsupported saved plot kind: {kind}")
    for section in options:
        if section != kind and config.get(section):
            raise ValueError(f"Saved report is {kind!r}, but redraw options were supplied under {section!r}; "
                             f"put plot options under {kind!r}")
    effective = dict(saved)
    effective.update({key: config[key] for key in ("dpi", "figsize", *FONT_SIZES) if key in config})
    settings = dict(saved.get(kind, {}))
    settings.update({key: value for key, value in config.get(kind, {}).items() if key in options[kind]})
    if kind == "step_orth_proj" and "proj_steps" in config.get(kind, {}) and config[kind]["proj_steps"] is None:
        settings["proj_steps"] = None
    effective[kind] = settings
    # Only saved setups participate in validation; input setups/datasets are ignored.
    validate_config(effective)
    if kind == "landscape":
        landscape_settings(effective, {})
        if type(settings.get("log_scale", True)) is not bool:
            raise ValueError("landscape.log_scale must be true or false")
    entries = manifest.get("setups", [])
    if (len(entries) != len(saved["setups"]) or
            {entry["index"] for entry in entries} != set(range(len(entries)))):
        raise ValueError("Saved setup manifest is incompatible with config.json")
    setup_labels = config.get("setup_labels")
    if "setup_labels" in config and (
        not isinstance(setup_labels, list) or not setup_labels
        or any(not isinstance(label, str) for label in setup_labels)
        or len(set(setup_labels)) != len(setup_labels)
        or any(label not in {entry["label"] for entry in entries} for label in setup_labels)
    ):
        raise ValueError("setup_labels must be a nonempty list of unique saved setup labels")
    jobs = []
    datasets = saved.get("datasets", [saved.get("dataset")])
    for dataset in datasets:
        label = Path(dataset).name if isinstance(dataset, str) else dataset["label"]
        folder = output / re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        summaries, alpha, beta = [], None, None
        table = []
        if kind in ("update_svd", "memory_search"):
            with (folder / "summary.csv").open() as handle:
                table = list(csv.DictReader(handle))
        if kind == "landscape" and settings.get("report_avg", False) and settings.get("include_cosine_sim", True):
            with (folder / "alignment_summary.csv").open() as handle:
                table = list(csv.DictReader(handle))
        for entry in entries:
            index, label = entry["index"], entry["label"]
            setup = saved["setups"][index]
            if label != setup["label"] or len(entry["runs"]) != len(setup["runs"]):
                raise ValueError("Saved setup labels/run counts disagree with manifest")
            path = folder / f"setup_{index}_mean.npz"
            with np.load(path, allow_pickle=False) as arrays:
                summary = dict(arrays)
            summary.update(label=label, runs=len(entry["runs"]))

            def require(key, shape, finite=True):
                value = summary.get(key)
                if (not isinstance(value, np.ndarray) or value.shape != shape or
                        (finite and not np.isfinite(value).all())):
                    raise ValueError(f"Incompatible saved {kind} data in {path}: expected {key} shape {shape}")
                return value

            if kind == "landscape":
                with np.load(folder / f"setup_{index}" / "run_0.npz", allow_pickle=False) as run:
                    x, y = run["alpha"], run["beta"]
                if (x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 3 or
                        not np.isfinite(x).all() or not np.isfinite(y).all() or
                        np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
                    raise ValueError(f"Invalid saved landscape axes in {folder}")
                if alpha is not None and (not np.array_equal(alpha, x) or not np.array_equal(beta, y)):
                    raise ValueError("Saved landscapes do not share the same axes")
                alpha, beta = x, y
                for key in ("inner_loss", "outer_loss"):
                    require(key, (len(beta), len(alpha)))
                if settings.get("include_cosine_sim", True):
                    require("alignment", (len(beta), len(alpha)), finite=False)
                    if settings.get("report_avg", False):
                        selected = [row for row in table if row["setup"] == label]
                        if len(selected) != 1 or int(selected[0]["runs"]) != summary["runs"]:
                            raise ValueError(f"Missing or incompatible saved mean cosine for {label}")
                        mean = float(selected[0]["mean_cosine"])
                        if not np.isfinite(mean) or not -1 <= mean <= 1:
                            raise ValueError(f"Invalid saved mean cosine for {label}")
                        summary["mean_cosine"] = mean
                trajectory = summary.get("trajectory")
                if trajectory is None or trajectory.ndim != 2 or trajectory.shape[1] != 2:
                    raise ValueError(f"Invalid saved trajectory in {path}")
                summary["basis_steps"] = entry["runs"][0]["landscape"]["basis_steps"]
            elif kind == "extrapolation":
                shape = (len(settings.get("steps", [0, 1, 2, 4, 8, 16])),)
                for metric in ("outer_loss", "token_accuracy", "exact_match"):
                    for suffix in ("_mean", "_std"):
                        require(metric + suffix, shape)
                if settings.get("show_runs", False):
                    for metric in ("outer_loss", "token_accuracy", "exact_match"):
                        values = []
                        for run_index in range(summary["runs"]):
                            with np.load(folder / f"setup_{index}" / f"run_{run_index}.npz", allow_pickle=False) as run:
                                value = run[metric]
                            if value.shape != shape or not np.isfinite(value).all():
                                raise ValueError(f"Invalid saved per-run {metric} for {label}")
                            values.append(value)
                        summary[metric + "_runs"] = np.stack(values)
            elif kind == "k_lr_sweep":
                axes = entry["runs"][0].get(kind) or sweep_settings(saved, setup)
                for key in ("steps", "inner_lrs"):
                    values = require(key, (len(axes[key]),))
                    if not np.array_equal(values, axes[key]):
                        raise ValueError(f"Saved sweep {key} disagree with manifest in {path}")
                    summary[key] = values.tolist()
                shape = (len(summary["steps"]), len(summary["inner_lrs"]))
                for metric in ("outer_loss", "token_accuracy", "exact_match"):
                    for suffix in ("_mean", "_std"):
                        require(metric + suffix, shape)
                summary["trained_settings"] = [json.loads(
                    (folder / f"setup_{index}" / f"run_{run}.json").read_text())["trained_settings"]
                    for run in range(summary["runs"])]
            elif kind == "step_orth_proj":
                spans = saved.get(kind, {}).get("proj_steps", 2)
                shape = ((0, settings["steps"]) if spans is None else
                         (len(spans), settings["steps"]) if isinstance(spans, list) else (settings["steps"],))
                conditional = settings.get("zero_step_policy", "error") == "nan"
                for key in ("projection_ratio_mean", "projection_ratio_std"):
                    values = require(key, shape, finite=not conditional)
                    if np.isinf(values).any():
                        raise ValueError(f"Infinite saved projection statistics in {path}")
                if conditional:
                    for key in ("projection_valid_examples", "projection_total_examples", "projection_valid_runs"):
                        require(key, shape)
                    if (np.any(summary["projection_valid_runs"] < 0) or
                            np.any(summary["projection_valid_runs"] > summary["runs"]) or
                            np.any(summary["projection_valid_runs"] != np.floor(summary["projection_valid_runs"])) or
                            np.any(summary["projection_valid_examples"] < 0) or
                            np.any(summary["projection_valid_examples"] > summary["projection_total_examples"]) or
                            np.any(summary["projection_total_examples"] <= 0)):
                        raise ValueError(f"Invalid saved projection coverage counts in {path}")
                    valid = summary["projection_valid_runs"] > 0
                    if (not np.array_equal(valid, summary["projection_valid_examples"] > 0) or
                            not np.array_equal(valid, np.isfinite(summary["projection_ratio_mean"])) or
                            not np.array_equal(valid, np.isfinite(summary["projection_ratio_std"]))):
                        raise ValueError(f"Saved projection coverage disagrees with missing values in {path}")
                require("steps", (settings["steps"],))
                if not np.array_equal(summary["steps"], np.arange(1, settings["steps"] + 1)):
                    raise ValueError(f"Invalid saved projection step axis in {path}")
                if isinstance(spans, list) or spans is None:
                    require("proj_steps", (len(spans or []),))
                    if not np.array_equal(summary["proj_steps"], spans or []):
                        raise ValueError(f"Saved projection spans disagree with config in {path}")
                if settings.get("add_step_norm", False):
                    for key in ("step_norm_mean", "step_norm_std"):
                        if key not in summary:
                            raise ValueError(f"Saved report has no {key} in {path}; run a new step-projection analysis "
                                             "to record WRITE-step norms. Redraw never recomputes missing data.")
                        if np.any(require(key, (settings["steps"],)) < 0):
                            raise ValueError(f"Negative saved WRITE-step norm statistics in {path}")
                if settings.get("include_cosine_sim", False):
                    cosine_shape = (settings["steps"],)
                    for key in ("cosine_sim_mean", "cosine_sim_std", "cosine_valid_examples",
                                "cosine_total_examples", "cosine_valid_runs"):
                        if key not in summary:
                            raise ValueError(f"Saved report has no {key}; a new analysis with include_cosine_sim=true is required")
                        values = require(key, cosine_shape, finite=not (conditional and key.startswith("cosine_sim")))
                        if np.isinf(values).any():
                            raise ValueError(f"Infinite saved consecutive-step cosine statistics in {path}")
                    count = summary["cosine_valid_runs"]
                    if (np.any(count < 0) or np.any(count > summary["runs"]) or np.any(count != np.floor(count)) or
                            np.any(summary["cosine_valid_examples"] < 0) or
                            np.any(summary["cosine_valid_examples"] > summary["cosine_total_examples"]) or
                            np.any(summary["cosine_total_examples"] <= 0) or
                            not np.array_equal(count > 0, summary["cosine_valid_examples"] > 0) or
                            not np.array_equal(count > 0, np.isfinite(summary["cosine_sim_mean"])) or
                            not np.array_equal(count > 0, np.isfinite(summary["cosine_sim_std"])) or
                            np.any(np.abs(summary["cosine_sim_mean"]) > 1) or np.any(summary["cosine_sim_std"] < 0)):
                        raise ValueError(f"Invalid saved consecutive-step cosine values/coverage in {path}")
                if settings.get("show_runs", False):
                    values, norm_values, cosine_values = [], [], []
                    for run_index in range(summary["runs"]):
                        with np.load(folder / f"setup_{index}" / f"run_{run_index}.npz", allow_pickle=False) as run:
                            value = run["projection_ratio"]
                            if "steps" not in run or not np.array_equal(run["steps"], summary["steps"]):
                                raise ValueError(f"Invalid saved per-run projection step axis for {label}")
                            if ((isinstance(spans, list) or spans is None) and
                                    ("proj_steps" not in run or not np.array_equal(run["proj_steps"], spans or []))):
                                raise ValueError(f"Invalid saved per-run projection span axis for {label}")
                            if settings.get("add_step_norm", False):
                                if "step_norm" not in run:
                                    raise ValueError(f"Saved per-run step_norm missing for {label}; a new analysis is required")
                                norm = run["step_norm"]
                                if norm.shape != (settings["steps"],) or not np.isfinite(norm).all() or np.any(norm < 0):
                                    raise ValueError(f"Invalid saved per-run WRITE-step norms for {label}")
                                norm_values.append(norm)
                            if settings.get("include_cosine_sim", False):
                                if "cosine_sim" not in run:
                                    raise ValueError(f"Saved per-run cosine_sim missing for {label}; a new analysis is required")
                                cosine = run["cosine_sim"]
                                if (cosine.shape != (settings["steps"],) or np.isinf(cosine).any() or
                                        (not conditional and np.isnan(cosine).any()) or np.any(np.abs(cosine) > 1)):
                                    raise ValueError(f"Invalid saved per-run consecutive-step cosines for {label}")
                                cosine_values.append(cosine)
                        if value.shape != shape or np.isinf(value).any() or (not conditional and np.isnan(value).any()):
                            raise ValueError(f"Invalid saved per-run projection ratios for {label}")
                        values.append(value)
                    summary["projection_ratio_runs"] = np.stack(values)
                    if settings.get("add_step_norm", False):
                        summary["step_norm_runs"] = np.stack(norm_values)
                    if settings.get("include_cosine_sim", False):
                        summary["cosine_sim_runs"] = np.stack(cosine_values)
            elif kind == "update_svd":
                values = summary.get("singular_values_mean")
                if values is None or values.ndim != 1 or not 1 <= len(values) <= settings["steps"]:
                    raise ValueError(f"Invalid saved SVD spectrum in {path}")
                for key in ("singular_values_mean", "singular_values_std", "ranks"):
                    require(key, values.shape)
                if not np.array_equal(summary["ranks"], np.arange(1, len(values) + 1)):
                    raise ValueError(f"Invalid saved SVD rank axis in {path}")
                ranks = {row["retained_rank"] for row in table if row["setup"] == label}
                if len(ranks) != 1:
                    raise ValueError(f"Missing or inconsistent retained rank for {label} in summary.csv")
                rank = ranks.pop()
                summary["retained_rank"] = int(rank) if rank else None
                if rank and not 1 <= int(rank) <= len(values):
                    raise ValueError(f"Invalid retained rank for {label}")
            else:
                for metric in ("inner_loss", "outer_loss", "token_accuracy", "exact_match"):
                    require(metric + "_outcomes", (len(settings.get("quantiles", [.5, .75, 1.0])),))
                    baselines = {float(row["baseline"]) for row in table
                                 if row["setup"] == label and row["metric"] == metric}
                    if len(baselines) != 1 or not all(np.isfinite(value) for value in baselines):
                        raise ValueError(f"Missing or inconsistent {metric} baseline for {label} in summary.csv")
                    summary[metric + "_baseline"] = baselines.pop()
            summaries.append(summary)
        if setup_labels is not None:
            summaries = [summary for summary in summaries if summary["label"] in setup_labels]
        jobs.append((folder, summaries, alpha, beta))
    # Validate every dataset before writing. One index covers all figures in this redraw.
    stem = "landscapes" if kind == "landscape" else kind
    indices = [int(match.group(1)) for folder, *_ in jobs for path in folder.glob(f"{stem}_redraw_*")
               if (match := re.match(rf"{re.escape(stem)}_redraw_(\d+)(?:[_.])", path.name))]
    index = max(indices, default=0) + 1
    for folder, summaries, alpha, beta in jobs:
        path = folder / f"{stem}_redraw_{index}.png"
        draw_plot(path, effective, summaries, settings, alpha, beta)
        path.with_suffix(".json").write_text(json.dumps(dict(
            kind=kind, source_config=str(output / "config.json"), source_manifest=str(output / "manifest.json"),
            dpi=effective.get("dpi", 220), figsize=effective.get("figsize"), plot_settings=settings,
            font_sizes={key: effective.get(key) for key in FONT_SIZES},
            request=config,
            setups=[dict(label=summary["label"], runs=summary["runs"]) for summary in summaries],
        ), indent=2) + "\n")
        print(f"Redrawn plot: {path}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON configuration; paths are repository-relative")
    parser.add_argument("--redraw", action="store_true", help="Draw indexed new plots from saved report data only")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.redraw:
        redraw_analysis(config)
    else:
        run_analysis(config)


if __name__ == "__main__":
    main()
