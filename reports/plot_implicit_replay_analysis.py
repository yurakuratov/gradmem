#!/usr/bin/env python3
"""Plot the read-only implicit-replay checkpoint analysis.

The script consumes the ``*_full.json`` and ``*_subset.json`` files emitted by
``/tmp/analyze_implicit_replay.py``.  It does not load or modify checkpoints.
Each figure is written as both PNG (for quick viewing) and SVG (for papers or
further editing).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


RUN_ORDER = ("state512_run7", "state512_run8", "state128_read")
RUN_COLORS = {
    "state512_run7": "#0072B2",
    "state512_run8": "#D55E00",
    "state128_read": "#009E73",
}
RUN_LABELS = {
    "state512_run7": "State 512 · run 7",
    "state512_run8": "State 512 · run 8",
    "state128_read": "State 128 + read K=2 · run 7",
}
RUN_SHORT_LABELS = {
    "state512_run7": "S512 · R7",
    "state512_run8": "S512 · R8",
    "state128_read": "S128 + read · R7",
}
STACK_COLORS = {
    "input": "#009E73",
    "recurrent": "#0072B2",
    "bias": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run7-full", type=Path, required=True)
    parser.add_argument("--run7-subset", type=Path, required=True)
    parser.add_argument("--run8-full", type=Path, required=True)
    parser.add_argument("--run8-subset", type=Path, required=True)
    parser.add_argument("--state128-read-full", type=Path, required=True)
    parser.add_argument("--state128-read-off-full", type=Path, required=True)
    parser.add_argument("--state128-read-subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
        }
    )


def save_figure(fig: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def sorted_records(mapping: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return sorted((int(key), value) for key, value in mapping.items())


def plot_run_lines(
    ax: Axes,
    values: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    percent: bool = False,
    zero_line: bool = False,
) -> None:
    for run, (x_values, y_values) in values.items():
        ax.plot(
            x_values,
            y_values,
            marker="o",
            color=RUN_COLORS[run],
            label=RUN_SHORT_LABELS[run],
        )
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if zero_line:
        ax.axhline(0.0, color="#666666", linewidth=1.0)


def make_retention_chart(
    full: dict[str, dict[str, Any]],
    subset: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)

    direct_exact: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    direct_token: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    final_exact: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run in RUN_ORDER:
        retention = sorted_records(subset[run]["retention"]["by_later_segments"])
        ages = np.array([age for age, _ in retention])
        direct_exact[run] = (
            ages,
            np.array([row["exact_match"] for _, row in retention]),
        )
        direct_token[run] = (
            ages,
            np.array([row["token_accuracy"] for _, row in retention]),
        )

        source = sorted_records(full[run]["accuracy"]["by_source_segment"])
        final_exact[run] = (
            np.array([7 - source_segment for source_segment, _ in source]),
            np.array([row["exact_match"] for _, row in source]),
        )

    plot_run_lines(axes[0], direct_exact, percent=True)
    axes[0].set_title("Direct retention: exact match")
    axes[0].set_xlabel("Later context segments processed")
    axes[0].set_ylabel("Exact match")
    axes[0].set_ylim(0, 1.02)

    plot_run_lines(axes[1], direct_token, percent=True)
    axes[1].set_title("Direct retention: token accuracy")
    axes[1].set_xlabel("Later context segments processed")
    axes[1].set_ylabel("Token accuracy")
    axes[1].set_ylim(0, 1.02)

    plot_run_lines(axes[2], final_exact, percent=True)
    axes[2].set_title("Full validation after all 8 segments")
    axes[2].set_xlabel("Later context segments after queried key")
    axes[2].set_ylabel("Exact match")
    axes[2].set_ylim(0, 1.02)

    for ax in axes:
        ax.set_xticks(range(8))
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    fig.suptitle(
        "Accuracy falls strongly as newer segments follow the queried value",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Direct curves: 256-example stratified subset; full validation: 5,000 examples.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_dir, "retention-by-query-age")


def make_ablation_chart(
    subset: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    ablation_order = [
        ("normal", "Normal"),
        ("read_optimization_disabled", "Read opt\noff†"),
        ("replay_disabled", "Replay\noff"),
        ("state_shuffled", "State\nshuffled"),
        ("state_zeroed", "State\nzeroed"),
        ("current_state_conditioning_disabled", "Current state\ncols 0*"),
        ("replay_state_conditioning_disabled", "Replay state\ncols 0*"),
    ]
    x_values = np.arange(len(ablation_order))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.0), constrained_layout=True)

    for ax, metric, title in (
        (axes[0], "exact_match", "Exact match"),
        (axes[1], "token_accuracy", "Token accuracy"),
    ):
        for run_index, run in enumerate(RUN_ORDER):
            values = [
                (
                    subset[run]["ablations"][key]["overall"][metric]
                    if key in subset[run]["ablations"]
                    else np.nan
                )
                for key, _ in ablation_order
            ]
            positions = x_values + (run_index - (len(RUN_ORDER) - 1) / 2) * width
            bars = ax.bar(
                positions,
                values,
                width=width,
                color=RUN_COLORS[run],
                label=RUN_SHORT_LABELS[run],
                alpha=0.92,
            )
            ax.bar_label(
                bars,
                labels=[f"{100 * value:.0f}" if np.isfinite(value) else "" for value in values],
                padding=2,
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_xticks(x_values, [label for _, label in ablation_order])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 0.83)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[0].legend(loc="upper right", ncol=3)
    fig.suptitle(
        "State shuffling has no effect; read optimization is also neutral on this subset",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "256-example subset. †Only applicable to state 128. *Zeroing trained input columns is an out-of-distribution diagnostic.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_dir, "state-and-replay-ablations")


def diagnostic_series(
    full: dict[str, dict[str, Any]], metric: str, *, start: int = 0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run in RUN_ORDER:
        rows = sorted_records(full[run]["segment_diagnostics"])
        rows = [(segment, row) for segment, row in rows if segment >= start]
        series[run] = (
            np.array([segment + 1 for segment, _ in rows]),
            np.array([row[metric]["mean"] for _, row in rows]),
        )
    return series


def make_force_chart(full: dict[str, dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    flat_axes = axes.ravel()

    plot_run_lines(flat_axes[0], diagnostic_series(full, "clip_affected"), percent=True)
    flat_axes[0].set_title("Updates affected by clipping")
    flat_axes[0].set_ylabel("Fraction of inner updates")
    flat_axes[0].set_ylim(0, 1.05)

    plot_run_lines(flat_axes[1], diagnostic_series(full, "delta_per_step_lr_norm"))
    flat_axes[1].set_title("Effective write magnitude")
    flat_axes[1].set_ylabel(r"$\|\Delta M\|_2 / (K\,\mathrm{inner\_lr})$")
    flat_axes[1].set_ylim(bottom=0)

    for run in RUN_ORDER:
        current = diagnostic_series(full, "current_grad_norm", start=1)[run]
        replay = diagnostic_series(full, "weighted_replay_grad_norm", start=1)[run]
        flat_axes[2].plot(
            current[0],
            current[1],
            marker="o",
            color=RUN_COLORS[run],
            label=f"{RUN_SHORT_LABELS[run]} current",
        )
        flat_axes[2].plot(
            replay[0],
            replay[1],
            marker="s",
            linestyle="--",
            color=RUN_COLORS[run],
            alpha=0.75,
            label=f"{RUN_SHORT_LABELS[run]} replay",
        )
    flat_axes[2].set_yscale("log")
    flat_axes[2].set_title("Forces before clipping")
    flat_axes[2].set_ylabel("Flattened-memory gradient norm")

    plot_run_lines(
        flat_axes[3], diagnostic_series(full, "replay_current_ratio", start=1)
    )
    flat_axes[3].axhline(1.0, color="#666666", linewidth=1.0)
    flat_axes[3].set_title("Replay/current force ratio")
    flat_axes[3].set_ylabel("Weighted replay norm / current norm")
    flat_axes[3].set_ylim(bottom=0)

    plot_run_lines(
        flat_axes[4],
        diagnostic_series(full, "replay_current_cosine", start=1),
        zero_line=True,
    )
    flat_axes[4].set_title("Force alignment")
    flat_axes[4].set_ylabel("Cosine similarity")
    flat_axes[4].set_ylim(-0.12, 0.18)

    plot_run_lines(
        flat_axes[5],
        diagnostic_series(full, "replay_current_conflict", start=1),
        percent=True,
    )
    flat_axes[5].set_title("Samples with conflicting forces")
    flat_axes[5].set_ylabel("Fraction with cosine < 0")
    flat_axes[5].set_ylim(0, 1.05)

    for ax in flat_axes:
        ax.set_xlabel("Context segment")
        ax.set_xticks(range(1, 9))
    flat_axes[0].legend(loc="lower left", ncol=3, fontsize=8)
    flat_axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "All later checkpoints are clip-limited while replay remains weakly aligned",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "memory-force-diagnostics")


def make_gru_chart(full: dict[str, dict[str, Any]], output_dir: Path) -> None:
    gates = ("reset", "update", "new")
    gate_labels = {"reset": "Reset gate", "update": "Update gate", "new": "Candidate"}
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 10.3), constrained_layout=True)
    x_values = np.arange(1, 9)

    for row_index, run in enumerate(RUN_ORDER):
        diagnostics = full[run]["segment_diagnostics"]
        for column_index, gate in enumerate(gates):
            ax = axes[row_index, column_index]
            bottom = np.zeros(8)
            for component in ("input", "recurrent", "bias"):
                values = np.array(
                    [
                        diagnostics[str(segment)][
                            f"gru_{gate}_{component}_fraction"
                        ]["mean"]
                        for segment in range(8)
                    ]
                )
                ax.bar(
                    x_values,
                    values,
                    bottom=bottom,
                    color=STACK_COLORS[component],
                    label=component.capitalize(),
                    width=0.72,
                )
                bottom += values

            ax.set_title(f"{RUN_LABELS[run]} · {gate_labels[gate]}")
            ax.set_xlabel("Context segment")
            ax.set_xticks(x_values)
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            if column_index == 0:
                ax.set_ylabel("Share of preactivation magnitude")

    axes[0, 0].legend(loc="upper right", ncol=3, fontsize=8)
    fig.suptitle(
        "GRU dynamics are dominated by recurrence and bias, not the memory delta",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Input is the flattened segment delta; fractions are relative absolute preactivation contributions.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output_dir, "gru-preactivation-contributions")


def make_state_chart(full: dict[str, dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    plot_run_lines(axes[0], diagnostic_series(full, "recurrent_state_norm"))
    axes[0].set_title("Recurrent-state norm")
    axes[0].set_xlabel("Context segment")
    axes[0].set_ylabel(r"Mean $\|s_j\|_2$")
    axes[0].set_xticks(range(1, 9))

    for run in RUN_ORDER:
        for metric, label, marker in (
            ("reset_gate_saturation", "reset gate", "o"),
            ("update_gate_saturation", "update gate", "s"),
            ("recurrent_state_saturation", "state", "^"),
        ):
            x_values, y_values = diagnostic_series(full, metric)[run]
            axes[1].plot(
                x_values,
                y_values,
                marker=marker,
                linestyle={"o": "-", "s": "--", "^": ":"}[marker],
                color=RUN_COLORS[run],
                label=f"{RUN_SHORT_LABELS[run]} {label}",
            )
    axes[1].set_title("Gate and state saturation")
    axes[1].set_xlabel("Context segment")
    axes[1].set_ylabel("Saturated fraction")
    axes[1].set_xticks(range(1, 9))
    axes[1].set_ylim(0, 1.0)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(loc="upper left", fontsize=8, ncol=3)

    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    fig.suptitle(
        "State magnitude and gate saturation accumulate with segment count",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "state-and-gate-saturation")


def weighted_metric(
    diagnostics: dict[str, Any], metric: str, *, start: int = 0
) -> float:
    total = 0.0
    count = 0
    for segment, row in sorted_records(diagnostics):
        if segment < start:
            continue
        summary = row[metric]
        total += float(summary["mean"]) * int(summary["count"])
        count += int(summary["count"])
    return total / count


def mean_gru_input_fraction(
    diagnostics: dict[str, Any], *, start: int = 1
) -> float:
    values = []
    for segment, row in sorted_records(diagnostics):
        if segment < start:
            continue
        values.extend(
            row[f"gru_{gate}_input_fraction"]["mean"]
            for gate in ("reset", "update", "new")
        )
    return float(np.mean(values))


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:+.{digits}f} pp"


def copy_input_data(args: argparse.Namespace, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for source in (
        args.run7_full,
        args.run7_subset,
        args.run8_full,
        args.run8_subset,
        args.state128_read_full,
        args.state128_read_off_full,
        args.state128_read_subset,
    ):
        shutil.copy2(source, data_dir / source.name)


def make_html_report(
    full: dict[str, dict[str, Any]],
    subset: dict[str, dict[str, Any]],
    state128_read_off_full: dict[str, Any],
    output_dir: Path,
    *,
    standalone: bool = False,
) -> None:
    overall = {
        run: full[run]["accuracy"]["overall"] for run in RUN_ORDER
    }
    retention = {
        run: subset[run]["retention"]["by_later_segments"]
        for run in RUN_ORDER
    }
    ablations = {
        run: subset[run]["ablations"] for run in RUN_ORDER
    }

    retention_drop = {
        run: retention[run]["7"]["exact_match"]
        - retention[run]["0"]["exact_match"]
        for run in RUN_ORDER
    }
    replay_effect = {
        run: ablations[run]["normal"]["overall"]["exact_match"]
        - ablations[run]["replay_disabled"]["overall"]["exact_match"]
        for run in RUN_ORDER
    }
    shuffle_effect = {
        run: ablations[run]["state_shuffled"]["overall"]["exact_match"]
        - ablations[run]["normal"]["overall"]["exact_match"]
        for run in RUN_ORDER
    }
    clip_fraction = {
        run: weighted_metric(full[run]["segment_diagnostics"], "clip_affected")
        for run in RUN_ORDER
    }
    replay_cosine = {
        run: weighted_metric(
            full[run]["segment_diagnostics"], "replay_current_cosine", start=1
        )
        for run in RUN_ORDER
    }
    replay_conflict = {
        run: weighted_metric(
            full[run]["segment_diagnostics"], "replay_current_conflict", start=1
        )
        for run in RUN_ORDER
    }
    gru_input = {
        run: mean_gru_input_fraction(full[run]["segment_diagnostics"])
        for run in RUN_ORDER
    }
    read_off_overall = state128_read_off_full["accuracy"]["overall"]
    read_effect = (
        overall["state128_read"]["exact_match"]
        - read_off_overall["exact_match"]
    )
    state128_gain = (
        overall["state128_read"]["exact_match"]
        - overall["state512_run8"]["exact_match"]
    )

    checkpoint_rows = "".join(
        f"""
        <tr>
          <td>{RUN_LABELS[run]}</td>
          <td><code>{html.escape(str(full[run]['checkpoint']))}</code></td>
          <td>{pct(overall[run]['exact_match'])}</td>
          <td>{pct(overall[run]['token_accuracy'])}</td>
        </tr>
        """
        for run in RUN_ORDER
    )
    checkpoint_rows += f"""
        <tr>
          <td>State 128 · read disabled*</td>
          <td><code>Same checkpoint; read-only inference ablation</code></td>
          <td>{pct(read_off_overall['exact_match'])}</td>
          <td>{pct(read_off_overall['token_accuracy'])}</td>
        </tr>
    """

    chart_stems = (
        "retention-by-query-age",
        "state-and-replay-ablations",
        "memory-force-diagnostics",
        "gru-preactivation-contributions",
        "state-and-gate-saturation",
    )
    if standalone:
        image_sources = {}
        for stem in chart_stems:
            encoded = base64.b64encode((output_dir / f"{stem}.svg").read_bytes()).decode(
                "ascii"
            )
            image_sources[stem] = f"data:image/svg+xml;base64,{encoded}"
        output_name = "implicit-replay-report-standalone.html"
        footer_note = (
            "All five figures are embedded in this file; no companion files are required."
        )
    else:
        image_sources = {stem: f"{stem}.svg" for stem in chart_stems}
        output_name = "implicit-replay-report.html"
        footer_note = "Raw inputs are bundled under <code>data/</code>."

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Implicit replay checkpoint analysis</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --text: #152033;
      --muted: #59677b;
      --border: #d9e0e8;
      --blue: #0072b2;
      --orange: #d55e00;
      --green: #009e73;
      --good-bg: #edf8f3;
      --warn-bg: #fff6e5;
      --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 36px auto 72px; }}
    header {{
      padding: 34px clamp(22px, 5vw, 54px);
      border-radius: 24px;
      color: #fff;
      background: linear-gradient(135deg, #0f3154 0%, #126a91 62%, #d55e00 150%);
      box-shadow: var(--shadow);
    }}
    header p {{ max-width: 790px; margin: 10px 0 0; color: #e6f4fb; }}
    h1 {{ margin: 0; font-size: clamp(30px, 5vw, 52px); line-height: 1.08; }}
    h2 {{ margin: 0 0 18px; font-size: clamp(22px, 3vw, 31px); }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    section {{ margin-top: 42px; }}
    .lede {{ font-size: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 18px; }}
    .finding {{
      padding: 22px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--surface);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
    }}
    .finding p {{ margin: 0; color: var(--muted); }}
    .metric {{ display: block; margin-bottom: 5px; font-size: 28px; font-weight: 700; }}
    .state512-run7 {{ color: var(--blue); }}
    .state512-run8 {{ color: var(--orange); }}
    .state128-read {{ color: var(--green); }}
    figure {{
      margin: 22px 0 0;
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--surface);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
    }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ margin: 12px 4px 0; color: var(--muted); }}
    .callout {{
      margin-top: 22px;
      padding: 18px 20px;
      border-left: 5px solid var(--orange);
      border-radius: 10px;
      background: var(--warn-bg);
    }}
    .callout.good {{ border-left-color: #008b68; background: var(--good-bg); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
    ul {{ padding-left: 22px; }}
    footer {{ margin-top: 46px; color: var(--muted); font-size: 14px; }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f1520; --surface: #171f2c; --text: #edf3fb; --muted: #aebbd0;
        --border: #2c394b; --good-bg: #122a25; --warn-bg: #332718;
        --shadow: 0 16px 40px rgba(0, 0, 0, .22);
      }}
      figure img {{ background: #fff; border-radius: 8px; }}
    }}
    @media (max-width: 760px) {{
      main {{ width: min(100% - 20px, 1180px); margin-top: 10px; }}
      header {{ border-radius: 16px; }}
      .grid {{ grid-template-columns: 1fr; }}
      figure {{ padding: 8px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Implicit replay: state size and read optimization</h1>
    <p class="lede">Read-only comparison of two state-512 checkpoints with a state-128 checkpoint trained with K=2 read optimization. All three were re-evaluated on the same validation data and corrected diagnostic protocol.</p>
  </header>

  <section>
    <h2>Executive findings</h2>
    <div class="grid">
      <article class="finding">
        <span class="metric"><span class="state128-read">{pct(overall['state128_read']['exact_match'])}</span> ({pp(state128_gain)})</span>
        <h3>Best full-validation exact match</h3>
        <p>The state-128 + read checkpoint exceeds the stronger state-512 checkpoint by 11.9 points and reaches {pct(overall['state128_read']['token_accuracy'], 2)} token accuracy.</p>
      </article>
      <article class="finding">
        <span class="metric"><span class="state512-run7">{pp(retention_drop['state512_run7'])}</span> / <span class="state512-run8">{pp(retention_drop['state512_run8'])}</span> / <span class="state128-read">{pp(retention_drop['state128_read'])}</span></span>
        <h3>Exact-match loss from age 0 to age 7</h3>
        <p>All variants still forget with age, but the state-128 checkpoint loses substantially less accuracy over seven later writes.</p>
      </article>
      <article class="finding">
        <span class="metric"><span class="state512-run7">{pp(shuffle_effect['state512_run7'])}</span> / <span class="state512-run8">{pp(shuffle_effect['state512_run8'])}</span> / <span class="state128-read">{pp(shuffle_effect['state128_read'])}</span></span>
        <h3>Effect of shuffling state across samples</h3>
        <p>Predictions are exactly unchanged for all three checkpoints, so reducing the state size did not make it measurably sample-specific.</p>
      </article>
      <article class="finding">
        <span class="metric"><span class="state512-run7">{pp(replay_effect['state512_run7'])}</span> / <span class="state512-run8">{pp(replay_effect['state512_run8'])}</span> / <span class="state128-read">{pp(replay_effect['state128_read'])}</span></span>
        <h3>Replay contribution to exact match</h3>
        <p>Replay contributes most in state-512 run 8. It provides a smaller 3.9-point benefit to the state-128 checkpoint.</p>
      </article>
      <article class="finding">
        <span class="metric"><span class="state128-read">{pp(read_effect)}</span></span>
        <h3>Read-optimization effect on full validation</h3>
        <p>Turning K=2 read optimization off changes none of the 5,000 exact-match outcomes or 10,000 target-token predictions. The stronger score is not caused by applying read updates at inference.</p>
      </article>
      <article class="finding">
        <span class="metric"><span class="state512-run7">{pct(clip_fraction['state512_run7'])}</span> / <span class="state512-run8">{pct(clip_fraction['state512_run8'])}</span> / <span class="state128-read">{pct(clip_fraction['state128_read'])}</span></span>
        <h3>Inner updates affected by clipping</h3>
        <p>Both later checkpoints are fully clip-limited. Their learned force magnitudes therefore do not translate proportionally into larger writes.</p>
      </article>
    </div>
  </section>

  <section>
    <h2>1. Retention decays with query age</h2>
    <figure>
      <img src="{image_sources['retention-by-query-age']}" alt="Retention and full-validation accuracy decline as later segments follow the queried key.">
      <figcaption>The direct evaluation asks each query immediately after its source segment and again after every later segment. The state-128 checkpoint is strongest after later writes, but still loses 31.6 exact-match points from age 0 to age 7.</figcaption>
    </figure>
    <div class="callout good"><strong>Conclusion:</strong> the 8×8 failure is genuinely age-dependent. It is not explained only by some source segments being intrinsically harder.</div>
  </section>

  <section>
    <h2>2. State matters, but appears non-specific</h2>
    <figure>
      <img src="{image_sources['state-and-replay-ablations']}" alt="Checkpoint ablations compare normal, replay-disabled, shuffled-state, zero-state, and state-conditioning removals.">
      <figcaption>State shuffling produces identical results to the normal model in all three checkpoints, whereas zeroing the state is highly damaging. For state 128, disabling read optimization is exactly neutral on both this subset and the full 5,000-example validation set. Column-zeroing interventions are out of distribution and should be interpreted only as sensitivity probes.</figcaption>
    </figure>
    <div class="callout"><strong>Interpretation:</strong> the model uses the state, but the shuffling result says it contains little actionable sample identity. It behaves more like a segment clock or shared control trajectory.</div>
  </section>

  <section>
    <h2>3. Replay grows, but is not a structured opposing force</h2>
    <figure>
      <img src="{image_sources['memory-force-diagnostics']}" alt="Per-segment clipping, memory delta, current and replay force norms, force ratio, cosine, and conflict fraction.">
      <figcaption>Mean replay/current cosine is {replay_cosine['state512_run7']:+.3f}, {replay_cosine['state512_run8']:+.3f}, and {replay_cosine['state128_read']:+.3f}; conflict occurs for {pct(replay_conflict['state512_run7'])}, {pct(replay_conflict['state512_run8'])}, and {pct(replay_conflict['state128_read'])} of active samples. The state-128 replay force is smaller relative to the current force and remains nearly orthogonal.</figcaption>
    </figure>
    <div class="callout"><strong>Interpretation:</strong> replay helps two checkpoints functionally, but none shows a consistently structured opposing gradient or sample-specific recurrent state.</div>
  </section>

  <section>
    <h2>4. The GRU largely ignores its delta input</h2>
    <figure>
      <img src="{image_sources['gru-preactivation-contributions']}" alt="Stacked GRU preactivation contributions show input, recurrent, and bias fractions by gate and segment.">
      <figcaption>After the first segment, the flattened delta contributes only {pct(gru_input['state512_run7'], 2)}, {pct(gru_input['state512_run8'], 2)}, and {pct(gru_input['state128_read'], 2)} of average gate preactivation magnitude. Recurrent and bias terms dominate every checkpoint.</figcaption>
    </figure>
    <figure>
      <img src="{image_sources['state-and-gate-saturation']}" alt="Recurrent state norm and gate saturation increase over context segments.">
      <figcaption>State norms and saturation accumulate with segment count. The smaller state does not avoid state saturation: 42.6% of its coordinates exceed |0.95| by the eighth segment.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Decision</h2>
    <p class="lede">The state-128 checkpoint is the best of the three and forgets more slowly, so a larger recurrent state is clearly unnecessary here. But disabling inference-time read optimization leaves every one of its 5,000 predictions unchanged. The improvement must therefore come from its training trajectory or another run difference—not the evaluated K=2 read update itself. Its state also remains non-specific under shuffling, its delta contribution is only {pct(gru_input['state128_read'], 2)}, and every write is clipped. The evidence still points first to delta normalization/state conditioning and clipping-scale work.</p>
  </section>

  <section>
    <h2>Evaluated checkpoints</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Run</th><th>Checkpoint source</th><th>Exact match</th><th>Token accuracy</th></tr></thead>
        <tbody>{checkpoint_rows}</tbody>
      </table>
    </div>
    <p>Full validation contains 5,000 examples. Direct retention and ablations use a 256-example subset stratified to 32 examples per source segment. *The read-disabled row is an inference ablation of the state-128 checkpoint, not a separate checkpoint.</p>
  </section>

  <footer>Generated from the corrected read-only analysis JSON. Saved checkpoints were not modified. {footer_note}</footer>
</main>
</body>
</html>
"""
    (output_dir / output_name).write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    full = {
        "state512_run7": load_json(args.run7_full),
        "state512_run8": load_json(args.run8_full),
        "state128_read": load_json(args.state128_read_full),
    }
    subset = {
        "state512_run7": load_json(args.run7_subset),
        "state512_run8": load_json(args.run8_subset),
        "state128_read": load_json(args.state128_read_subset),
    }
    state128_read_off_full = load_json(args.state128_read_off_full)
    expected_source_names = {
        "state512_run7": "run7",
        "state512_run8": "run8",
        "state128_read": "state128_read",
    }
    for run, source_name in expected_source_names.items():
        if (
            full[run].get("run") != source_name
            or subset[run].get("run") != source_name
        ):
            raise ValueError(
                f"Input files for {run} do not contain the expected "
                f"run={source_name!r} data"
            )
    if state128_read_off_full.get("run") != "state128_read":
        raise ValueError(
            "Read-disabled input does not contain the expected "
            "run='state128_read' data"
        )

    configure_style()
    make_retention_chart(full, subset, args.output_dir)
    make_ablation_chart(subset, args.output_dir)
    make_force_chart(full, args.output_dir)
    make_gru_chart(full, args.output_dir)
    make_state_chart(full, args.output_dir)
    copy_input_data(args, args.output_dir)
    make_html_report(full, subset, state128_read_off_full, args.output_dir)
    make_html_report(
        full,
        subset,
        state128_read_off_full,
        args.output_dir,
        standalone=True,
    )


if __name__ == "__main__":
    main()
