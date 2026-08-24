#!/usr/bin/env python3
"""Plot WRITE-step extrapolation and centered local WRITE/READ loss slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


COLORS = {
    "plateau-init": "#111827",
    "lipschitz-s143": "#4C78A8",
    "lipschitz-s144": "#72A0C1",
    "lip-noise-s143": "#F2A65A",
    "lip-search-s143": "#D1495B",
    "lip-search-s144": "#9C4050",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-sweep-csv", type=Path, required=True)
    parser.add_argument("--grid-csv", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def wilson(proportion: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = 1.96
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return np.maximum(0, proportion - center + radius), np.maximum(0, center + radius - proportion)


def plot_extrapolation(path: Path, output_dir: Path) -> None:
    data = pd.read_csv(path)
    labels = {
        "plateau-init": "Initialization",
        "lipschitz-s143": "Lipschitz s143",
        "lipschitz-s144": "Lipschitz s144",
        "lip-noise-s143": "Lip.+noise s143",
        "lip-search-s143": "Lip.+search s143",
        "lip-search-s144": "Lip.+search s144",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8), sharey=True)
    for ax, direction, title in zip(axes, ("F", "B"), ("Forward contexts", "Backward contexts")):
        for alias in labels:
            rows = data[(data.alias == alias) & (data.direction == direction)].sort_values("K")
            p = rows.exact_match.to_numpy()
            low, high = wilson(p, rows.example_count.to_numpy())
            ax.errorbar(rows.K, p * 100, yerr=np.vstack([low, high]) * 100,
                        marker="o", markersize=3, linewidth=1.1, capsize=1.8,
                        color=COLORS[alias], label=labels[alias])
        ax.axvline(2, color="#9CA3AF", linestyle=":", linewidth=0.8)
        ax.set(title=title, xlabel="Evaluation-time WRITE steps K", xticks=sorted(data.K.unique()), ylim=(-2, 102))
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Exact match (%)")
    axes[1].legend(frameon=False, ncol=2, loc="lower left")
    save(fig, output_dir, "plateau_write_step_extrapolation")


def symmetric_limit(values: np.ndarray, quantile: float = 0.97) -> float:
    limit = float(np.quantile(np.abs(values), quantile))
    return max(limit, np.finfo(float).eps)


def plot_landscape_model(data: pd.DataFrame, trajectory: pd.DataFrame,
                         alias: str, output_dir: Path) -> None:
    selected = data[data.alias == alias]
    transitions = sorted(selected[["step", "next_step"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(len(transitions), 2, figsize=(5.5, 2.35 * len(transitions)), sharex=True, sharey=True)
    inner_limit = symmetric_limit(selected.inner_loss_delta_mean.to_numpy())
    read_limit = symmetric_limit(selected.read_loss_delta_mean.to_numpy())
    inner_levels = np.linspace(-inner_limit, inner_limit, 21)
    read_levels = np.linspace(-read_limit, read_limit, 21)
    contour_handles = []
    for row_index, (step, next_step) in enumerate(transitions):
        rows = selected[(selected.step == step) & (selected.next_step == next_step)]
        radius = float(
            trajectory[(trajectory.alias == alias) & (trajectory.step == step)].interval_radius_mean.iloc[0]
        )
        xs = np.sort(rows.x.unique())
        ys = np.sort(rows.y.unique())
        for column, (field, title, limit) in enumerate((
            ("inner_loss_delta_mean", "Inner energy change", inner_limit),
            ("read_loss_delta_mean", "READ loss change", read_limit),
        )):
            ax = axes[row_index, column]
            matrix = rows.pivot(index="y", columns="x", values=field).loc[ys, xs].to_numpy()
            levels = inner_levels if column == 0 else read_levels
            contour = ax.contourf(xs, ys, matrix, levels=levels, cmap="coolwarm",
                                  norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), extend="both")
            contour_handles.append(contour)
            ax.scatter([0], [0], color="black", marker="x", s=24, linewidth=1.1)
            ax.arrow(0, 0, 1, 0, color="black", width=0.025, head_width=0.16,
                     length_includes_head=True, zorder=4)
            ax.scatter([1], [0], facecolor="white", edgecolor="black", s=24, zorder=5)
            ax.axhline(0, color="black", alpha=0.18, linewidth=0.5)
            ax.axvline(0, color="black", alpha=0.18, linewidth=0.5)
            ax.set_aspect("equal")
            if row_index == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"$M_{step}\\to M_{next_step}$\northogonal offset $y$")
            if row_index == len(transitions) - 1:
                ax.set_xlabel("corresponding interval offset $x$")
            ax.text(0.03, 0.96, f"mean step L2={radius:.3g}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5})
    fig.colorbar(contour_handles[0], ax=axes[:, 0], shrink=0.72, pad=0.03, label="mean change")
    fig.colorbar(contour_handles[1], ax=axes[:, 1], shrink=0.72, pad=0.03, label="mean change")
    fig.suptitle(alias.replace("-", " "), y=0.995)
    save(fig, output_dir, f"landscape_{alias}")


def plot_lipschitz_trajectory(path: Path, output_dir: Path) -> None:
    data = pd.read_csv(path)
    labels = {"plateau-init": "Initialization", "lipschitz-s144": "Lipschitz", "lip-search-s144": "Lip.+search"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    for alias, label in labels.items():
        rows = data[data.alias == alias].sort_values("step")
        color = COLORS[alias]
        axes[0].plot(rows.step, rows.gradient_norm_mean, marker="o", label=label, color=color)
        axes[1].plot(rows.step, rows.inner_loss_mean, marker="o", label=label, color=color)
        axes[2].plot(rows.step, rows.read_loss_mean, marker="o", label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set(title="Local gradient norm", ylabel=r"mean $\|\nabla_M E\|_2$")
    axes[1].set(title="Inner trajectory", ylabel="mean energy")
    axes[2].set(title="Outer trajectory", ylabel="mean READ loss")
    for ax in axes:
        ax.set(xlabel="Memory state $M_k$", xticks=sorted(data.step.unique()))
        ax.grid(alpha=0.18)
    axes[2].legend(frameon=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, output_dir, "lipschitz_trajectory")


def main() -> None:
    args = parse_args()
    setup_style()
    plot_extrapolation(args.k_sweep_csv, args.output_dir)
    grid = pd.read_csv(args.grid_csv)
    trajectory = pd.read_csv(args.trajectory_csv)
    for alias in grid.alias.unique():
        plot_landscape_model(grid, trajectory, alias, args.output_dir)
    plot_lipschitz_trajectory(args.trajectory_csv, args.output_dir)
    print(f"Wrote extrapolation and landscape figures to {args.output_dir}")


if __name__ == "__main__":
    main()
