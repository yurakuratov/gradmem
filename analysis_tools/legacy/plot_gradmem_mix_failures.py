#!/usr/bin/env python3
"""Create figures and LaTeX tables for the GradMem mix-N8 failure report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "successful": "#147d64",
    "unsuccessful": "#a64132",
    "all": "#425466",
}
CATEGORY_COLORS = {
    "exact": "#147d64",
    "first_correct_second_wrong": "#d28b26",
    "first_wrong_second_correct": "#547aa5",
    "both_wrong": "#9a9a9a",
}
CATEGORY_LABELS = {
    "exact": "both correct",
    "first_correct_second_wrong": "first only",
    "first_wrong_second_correct": "second only",
    "both_wrong": "both wrong",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 160,
    })


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def config_names(checkpoints: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for hp, row in checkpoints.groupby("hp_group", sort=False).first().iterrows():
        if row.n_embd == 128:
            label = "D128, K1"
        elif not row.use_write_head:
            label = "base head"
        elif row.step_alignment_weight == 0:
            label = "write head"
        elif row.lipschitz_weight > 0:
            label = "+ align/read/lip"
        else:
            label = "+ align/read"
        labels[hp] = label
    return labels


def run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(
        ["run_id", "hp_group", "outcome", "seed", "condition", "direction"], sort=True
    ).agg(
        first_accuracy=("first_correct", "mean"),
        second_accuracy=("second_correct", "mean"),
        exact_match=("exact_match", "mean"),
        distractor_rate=("wrong_second_is_distractor", "mean"),
    ).reset_index()


def plot_condition_em(metrics: pd.DataFrame, names: dict[str, str], output_dir: Path) -> None:
    hp_order = list(names)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), sharey=True)
    cells = [
        ("clean", "F", "Clean forward"),
        ("clean", "B", "Clean backward"),
        ("shared_first_value", "F", "Shared-first forward"),
        ("shared_first_value", "B", "Shared-first backward"),
    ]
    for axis, (condition, direction, title) in zip(axes, cells):
        cell = metrics[(metrics.condition == condition) & (metrics.direction == direction)]
        for x, hp in enumerate(hp_order):
            hp_cell = cell[cell.hp_group == hp]
            for offset, (_, row) in enumerate(hp_cell.iterrows()):
                jitter = (offset - (len(hp_cell) - 1) / 2) * 0.08
                axis.scatter(
                    x + jitter,
                    100 * row.exact_match,
                    s=24,
                    color=COLORS[row.outcome],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
        axis.set_title(title)
        axis.set_xticks(range(len(hp_order)), [names[value] for value in hp_order], rotation=38, ha="right")
        axis.set_ylim(-2, 102)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Exact match (%)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in (("successful", COLORS["successful"]), ("unsuccessful", COLORS["unsuccessful"]))
    ]
    axes[-1].legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, output_dir / "figures", "condition_exact_match")


def plot_taxonomy(predictions: pd.DataFrame, names: dict[str, str], output_dir: Path) -> None:
    rows = []
    for hp, hp_frame in predictions.groupby("hp_group", sort=False):
        for outcome, group in [("all", hp_frame), *list(hp_frame.groupby("outcome", sort=True))]:
            if outcome != "all" and group.run_id.nunique() == hp_frame.run_id.nunique():
                continue
            label = names[hp] + (f" ({outcome})" if outcome != "all" else "")
            rows.append((hp, outcome, label, group))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, max(3.5, 0.34 * len(rows) + 1.2)), sharex=True)
    for axis, condition, direction, title in [
        (axes[0, 0], "clean", "F", "Clean forward"),
        (axes[0, 1], "clean", "B", "Clean backward"),
        (axes[1, 0], "shared_first_value", "F", "Shared-first forward"),
        (axes[1, 1], "shared_first_value", "B", "Shared-first backward"),
    ]:
        left = np.zeros(len(rows))
        for category in CATEGORY_LABELS:
            values = []
            for _, _, _, group in rows:
                cell = group[(group.condition == condition) & (group.direction == direction)]
                values.append(100 * cell.error_category.eq(category).mean())
            axis.barh(
                range(len(rows)), values, left=left, color=CATEGORY_COLORS[category],
                label=CATEGORY_LABELS[category], height=0.72,
            )
            left += np.nan_to_num(values)
        axis.set_title(title)
        axis.set_xlim(0, 100)
        axis.grid(axis="x", alpha=0.18)
        axis.set_yticks(range(len(rows)), [row[2] for row in rows])
        axis.invert_yaxis()
    axes[0, 1].legend(frameon=False, loc="lower right")
    axes[1, 0].set_xlabel("Examples (%)")
    axes[1, 1].set_xlabel("Examples (%)")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures", "failure_taxonomy")


def plot_trajectory(trajectory: pd.DataFrame, names: dict[str, str], output_dir: Path) -> None:
    clean = trajectory[trajectory.condition == "clean"]
    groups = []
    for hp, hp_frame in clean.groupby("hp_group", sort=False):
        for outcome, group in hp_frame.groupby("outcome", sort=True):
            groups.append((hp, outcome, group))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35))
    for hp, outcome, group in groups:
        summary = group.groupby("step")[["key_loss", "value_loss", "outer_loss"]].mean()
        label = names[hp] + (" success" if outcome == "successful" else "")
        linestyle = "-" if outcome == "successful" else "--"
        color = plt.cm.tab10(list(names).index(hp) % 10)
        for axis, metric, title in zip(
            axes, ("key_loss", "value_loss", "outer_loss"), ("Key CE", "Value CE", "READ loss")
        ):
            axis.plot(summary.index, summary[metric], marker="o", ms=2.5, lw=1.1,
                      linestyle=linestyle, color=color, label=label)
            axis.set_title(title)
            axis.set_xlabel("WRITE state")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean loss")
    axes[-1].set_yscale("log")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures", "loss_trajectories")


def plot_training_correlations(correlations: pd.DataFrame, names: dict[str, str], output_dir: Path) -> None:
    selected = correlations[
        (correlations.scope == "run")
        & (correlations.method == "spearman")
        & (
            ((correlations.x == "eval_step_alignment_loss") & (correlations.y == "eval_outer_loss"))
            | ((correlations.x == "eval_inner_loss") & (correlations.y == "eval_outer_loss"))
        )
    ].copy()
    selected["pair"] = np.where(
        selected.x.eq("eval_step_alignment_loss"), "alignment vs READ", "inner vs READ"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
    for axis, pair in zip(axes, ("alignment vs READ", "inner vs READ")):
        cell = selected[selected.pair == pair]
        for x, hp in enumerate(names):
            hp_cell = cell[cell.hp_group == hp]
            for offset, (_, row) in enumerate(hp_cell.iterrows()):
                jitter = (offset - (len(hp_cell) - 1) / 2) * 0.08
                axis.scatter(x + jitter, row.correlation, s=24, color=COLORS[row.outcome_group],
                             edgecolor="white", linewidth=0.4)
        axis.axhline(0, color="black", lw=0.7)
        axis.set_title(pair)
        axis.set_xticks(range(len(names)), list(names.values()), rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Within-run Spearman correlation")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures", "training_loss_correlations")


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_tables(checkpoints: pd.DataFrame, metrics: pd.DataFrame, names: dict[str, str], output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    rows = []
    for _, checkpoint in checkpoints.iterrows():
        cell = metrics[(metrics.run_id == checkpoint.run_id)]
        values = {}
        for condition, direction, key in [
            ("clean", "F", "clean_f"), ("clean", "B", "clean_b"),
            ("shared_first_value", "F", "control_f"),
            ("shared_first_value", "B", "control_b"),
        ]:
            values[key] = 100 * cell[(cell.condition == condition) & (cell.direction == direction)].exact_match.iloc[0]
        rows.append(
            f"{latex_escape(names[checkpoint.hp_group])} & {int(checkpoint.seed)} & "
            f"{checkpoint.outcome} & {100 * checkpoint.best_exact_match:.1f} & "
            f"{values['clean_f']:.1f} & {values['clean_b']:.1f} & "
            f"{values['control_f']:.1f} & {values['control_b']:.1f} \\\\"
        )
    (table_dir / "run_results.tex").write_text("\n".join(rows) + "\n")

    trajectory = pd.read_csv(output_dir / "data" / "trajectory_losses.csv")
    clean = trajectory[trajectory.condition == "clean"]
    trajectory_rows = []
    for (hp, outcome), group in clean.groupby(["hp_group", "outcome"], sort=False):
        start = group[group.step == 0][["key_loss", "value_loss", "outer_loss"]].mean()
        final_step = group.step.max()
        final = group[group.step == final_step][["key_loss", "value_loss", "outer_loss"]].mean()
        trajectory_rows.append(
            f"{latex_escape(names[hp])} & {outcome} & "
            f"{start.key_loss:.2f}$\\to${final.key_loss:.2f} & "
            f"{start.value_loss:.2f}$\\to${final.value_loss:.2f} & "
            f"{start.outer_loss:.2f}$\\to${final.outer_loss:.3f} \\\\"
        )
    (table_dir / "trajectory_summary.tex").write_text("\n".join(trajectory_rows) + "\n")


def main() -> None:
    args = parse_args()
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    configure_style()
    checkpoints = pd.read_csv(args.data_dir / "checkpoints.csv")
    predictions = pd.read_csv(args.data_dir / "per_example_predictions.csv")
    trajectory = pd.read_csv(args.data_dir / "trajectory_losses.csv")
    correlations = pd.read_csv(args.data_dir / "training_correlations.csv")
    names = config_names(checkpoints)
    metrics = run_metrics(predictions)
    metrics.to_csv(args.data_dir / "run_metrics.csv", index=False)
    plot_condition_em(metrics, names, args.output_dir)
    plot_taxonomy(predictions, names, args.output_dir)
    plot_trajectory(trajectory, names, args.output_dir)
    plot_training_correlations(correlations, names, args.output_dir)
    write_tables(checkpoints, metrics, names, args.output_dir)
    print(f"Wrote figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
