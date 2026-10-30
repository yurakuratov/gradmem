#!/usr/bin/env python3
"""Plot N32 and N64 training-length memory-search outcomes together."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


SERIES = ("base", "q50", "q75", "best")
COLORS = {
    "base": "#9AA5B1",
    "q50": "#8EC5E8",
    "q75": "#B8A1D9",
    "best": "#F2A88F",
}
LABELS = {
    "base": "Base memory",
    "q50": "q50 improvement",
    "q75": "q75 improvement",
    "best": "Best-of-4",
}


def read_rows(path: Path, n_value: int) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if int(row["N"]) != n_value:
            continue
        result.append({
            "N": n_value,
            "metric": row["metric"],
            "series": row["series"],
            "mean": float(row["mean"]),
        })
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[2]
    report_dir = root / "reports/mqar_training_length_memory_search_report"
    rows = read_rows(
        root / "reports/mqar_N32_init_N16_memory_search_N32_N64/aggregate_summary.csv",
        32,
    )
    rows += read_rows(
        root / "reports/mqar_N64_init_N16_memory_search/aggregate_summary.csv",
        64,
    )

    positions = np.arange(2)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.825))
    metrics = (
        ("loss", "Outer loss"),
        ("accuracy", "Token accuracy (%)"),
        ("exact_match", "All-query exact match (%)"),
    )
    for ax, (metric, ylabel) in zip(axes, metrics):
        baseline_by_n = {
            n_value: next(
                row["mean"] for row in rows
                if row["N"] == n_value
                and row["metric"] == metric
                and row["series"] == "base"
            )
            for n_value in (32, 64)
        }
        for series in SERIES:
            values = []
            for n_value in (32, 64):
                row = next(
                    row for row in rows
                    if row["N"] == n_value
                    and row["metric"] == metric
                    and row["series"] == series
                )
                if metric == "loss":
                    value = 100.0 * row["mean"] / baseline_by_n[n_value]
                else:
                    value = 100.0 * (row["mean"] - baseline_by_n[n_value])
                values.append(value)
            ax.plot(
                positions,
                values,
                color=COLORS[series],
                marker="o",
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.6,
                markersize=7,
                label=LABELS[series],
            )
        ax.set_xticks(positions, ("N32 checkpoint\non N=32", "N64 checkpoint\non N=64"))
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        baseline_level = 100.0 if metric == "loss" else 0.0
        ax.axhline(baseline_level, color="#667085", linewidth=1.1, zorder=0)
        ax.grid(color="#D9DEE7", alpha=0.65, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Outer loss (% baseline)")
    axes[1].set_ylabel("Token-accuracy gain (pp)")
    axes[2].set_ylabel("Exact-match gain (pp)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        handlelength=3.0,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(report_dir / "N32_N64_memory_search.png", dpi=220, bbox_inches="tight")
    fig.savefig(report_dir / "N32_N64_memory_search.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
