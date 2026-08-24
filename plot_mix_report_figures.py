#!/usr/bin/env python3
"""Generate publication figures for the mix-N8 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "gradmem": "#2C6E9B",
    "energy-success": "#D1495B",
    "energy-failed": "#7A5195",
    "scratch": "#6B7280",
    "plateau": "#E07A5F",
    "key": "#355070",
    "value": "#EAAC8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/mix-N8-K2V2-V62_1M"))
    parser.add_argument("--components-csv", type=Path, required=True)
    parser.add_argument("--k-sweep-csv", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def trainer_history(run: Path) -> list[dict]:
    path = run / "trainer_state.json"
    if not path.exists():
        checkpoints = sorted(
            run.glob("checkpoint-*/trainer_state.json"),
            key=lambda item: int(item.parent.name.split("-")[-1]),
        )
        if not checkpoints:
            return []
        path = checkpoints[-1]
    return json.loads(path.read_text()).get("log_history", [])


def plot_learning_curves(runs_root: Path, output_dir: Path) -> None:
    specs = [
        ("GradMem s143", "gradmem_llama_L4H4D256_mem8_K2_ilr0.4_grad_second_bs_64_lr_1e-04_fp32_init_llama/run_1", COLORS["gradmem"], "-"),
        ("GradMem s144", "gradmem_llama_L4H4D256_mem8_K2_ilr0.4_grad_second_bs_64_lr_1e-04_fp32_init_llama/run_2", COLORS["gradmem"], "--"),
        ("Energy s143", "energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_energy_grad_second_stepalign0.1_iread0.1_bs_64_lr_1e-04_fp32/run_1", COLORS["energy-failed"], "-"),
        ("Energy s144", "energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_energy_grad_second_stepalign0.1_iread0.1_bs_64_lr_1e-04_fp32/run_2", COLORS["energy-success"], "-"),
        ("Energy s145", "energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_energy_grad_second_stepalign0.1_iread0.1_bs_64_lr_1e-04_fp32/run_3", COLORS["energy-failed"], "--"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for label, suffix, color, linestyle in specs:
        points = [entry for entry in trainer_history(runs_root / suffix) if "eval_exact_match" in entry]
        if not points:
            continue
        x = np.asarray([float(point["step"]) / 1000 for point in points])
        y = np.asarray([float(point["eval_exact_match"]) * 100 for point in points])
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.25, alpha=0.92)
    ax.axhline(99, color="#111827", linewidth=0.8, linestyle=":", label="99% target")
    ax.set(xlabel="Outer training step (thousands)", ylabel="Validation exact match (%)", ylim=(-2, 102))
    ax.legend(ncol=2, frameon=False, loc="lower right")
    save(fig, output_dir, "learning_curves")


def plot_k_sweep(path: Path, output_dir: Path) -> None:
    data = pd.read_csv(path)
    labels = {"gradmem": "GradMem", "energy-success": "Energy success", "energy-failed": "Energy failed"}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55), sharey=True)
    for ax, direction, title in zip(axes, ("F", "B"), ("Forward contexts", "Backward contexts")):
        selected = data[data.direction == direction]
        for alias in ("gradmem", "energy-success", "energy-failed"):
            rows = selected[selected.alias == alias].sort_values("K")
            proportion = rows.exact_match.to_numpy()
            count = rows.example_count.to_numpy()
            z = 1.96
            denominator = 1 + z * z / count
            center = (proportion + z * z / (2 * count)) / denominator
            radius = z * np.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
            lower_error = np.maximum(0.0, proportion - center + radius)
            upper_error = np.maximum(0.0, center + radius - proportion)
            ax.errorbar(rows.K, proportion * 100,
                        yerr=np.vstack([lower_error * 100, upper_error * 100]),
                        marker="o", markersize=3.6, linewidth=1.35, capsize=2,
                        label=labels[alias], color=COLORS[alias])
        ax.axvline(2, color="#9CA3AF", linestyle=":", linewidth=0.9)
        ax.set(title=title, xlabel="Evaluation-time WRITE steps K", xticks=sorted(data.K.unique()), ylim=(-3, 103))
    axes[0].set_ylabel("Exact match (%)")
    axes[1].legend(frameon=False, loc="best")
    save(fig, output_dir, "exact_match_vs_k")


def plot_write_diagnostics(path: Path, output_dir: Path) -> None:
    data = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55))
    gradmem = data[(data.alias == "gradmem") & (data.direction == "F")]
    for component, label, color in (("reconstruction_key", "Key CE", COLORS["key"]),
                                    ("reconstruction_value", "Value CE", COLORS["value"])):
        rows = gradmem[gradmem.component == component].sort_values("step")
        axes[0].plot(rows.step, rows["mean"], marker="o", label=label, color=color, linewidth=1.5)
    axes[0].set(title="GradMem: forward WRITE", xlabel="Memory state $M_k$", ylabel="Token reconstruction CE", xticks=[0, 1, 2])
    axes[0].legend(frameon=False)

    energy = data[(data.alias == "energy-success") & (data.direction == "F")]
    for component, label, color in (("energy_key", "Key energy", COLORS["key"]),
                                    ("energy_value", "Value energy", COLORS["value"])):
        rows = energy[energy.component == component].sort_values("step")
        axes[1].plot(rows.step, rows["mean"], marker="o", label=label, color=color, linewidth=1.5)
    axes[1].set(title="Energy success: forward WRITE", xlabel="Memory state $M_k$", ylabel="Mean token energy", xticks=[0, 1, 2])
    axes[1].legend(frameon=False)

    for alias, label, color in (("gradmem", "GradMem", COLORS["gradmem"]),
                                ("energy-success", "Energy success", COLORS["energy-success"]),
                                ("energy-failed", "Energy failed", COLORS["energy-failed"])):
        rows = data[(data.alias == alias) & (data.direction == "F") &
                    (data.component.str.endswith("key"))].sort_values("step")
        axes[2].plot(rows.step, rows.gradient_norm_mean, marker="o", label=label, color=color, linewidth=1.5)
    axes[2].set_yscale("log")
    axes[2].set(title="WRITE gradient scale", xlabel="Memory state $M_k$",
                ylabel=r"Mean $\|\nabla_M\mathcal{L}\|_2$", xticks=[0, 1, 2])
    axes[2].legend(frameon=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, output_dir, "write_diagnostics")


def plot_stability(path: Path, output_dir: Path) -> None:
    data = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.65), sharey=True)
    for ax, initialization, title in zip(axes, ("scratch", "plateau"), ("From scratch", "Successful plateau initialization")):
        selected = data[data.initialization == initialization]
        names = list(dict.fromkeys(selected.intervention))
        x = np.arange(len(names))
        for position, name in enumerate(names):
            values = selected[selected.intervention == name].best_exact_match.to_numpy() * 100
            offsets = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else np.asarray([0.0])
            ax.scatter(np.full(len(values), position) + offsets, values, color=COLORS[initialization],
                       edgecolor="white", linewidth=0.5, s=34, zorder=3)
            if len(values) > 1:
                ax.vlines(position, values.min(), values.max(), color="#111827", linewidth=1.1, zorder=2)
        ax.set(title=title, ylabel="Best validation EM (%)", xticks=x, xticklabels=names, ylim=(0, 102))
        ax.tick_params(axis="x", rotation=24)
    fig.tight_layout(w_pad=1.4)
    save(fig, output_dir, "stability_interventions")


def main() -> None:
    args = parse_args()
    setup_style()
    plot_learning_curves(args.runs_root, args.output_dir)
    plot_k_sweep(args.k_sweep_csv, args.output_dir)
    plot_write_diagnostics(args.components_csv, args.output_dir)
    plot_stability(args.stability_csv, args.output_dir)
    print(f"Wrote report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
