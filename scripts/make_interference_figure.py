#!/usr/bin/env python
"""Interference-curve figure for the results report
(reports/figures/interference_em_vs_n.svg).

Left: best eval EM per phase vs N — the pre-registered Δ(N) readout (mean
over seeds; dots = seeds; open markers = phases truncated by the
infrastructure kills, so their value is a snapshot, not a converged best).
Middle: step-0 warm-transfer EM — the phase's first eval after loading the
previous phase's checkpoint, i.e. read-side interference before any
retraining at the larger bank.
Right: read_state_cross_talk tail mean vs N — the interference floor, mean
|q·k| over NON-target bank items on the last instrumented layer (same
diagnostic in both arms; softmax suppresses the floor, the linear read sums it).

Run:  uv run python scripts/make_interference_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "reports", "results", "interference_curve.json")
OUT = os.path.join(HERE, "..", "reports", "figures", "interference_em_vs_n.svg")

STYLE = {
    "interface":        ("interface (superposed state, O(1) read)", "#c77b28", "o"),
    "interface_softmax": ("interface + exact softmax (O(N) read)", "#1e7d43", "s"),
    "adaptive":         ("adaptive carried-memory (sgd, 2 tokens)", "#c0392b", "^"),
}


def phase(summary, arm, n):
    return summary.get(f"{arm}_N{n}")


def main():
    data = json.load(open(DATA))
    S = data["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0))

    # ---- left: best EM vs N ------------------------------------------
    ax = axes[0]
    for arm, (label, color, marker) in STYLE.items():
        xs, ys, trunc = [], [], []
        for n in [8, 16, 32, 64, 96]:
            p = phase(S, arm, n)
            if not p:
                continue
            xs.append(n); ys.append(p["mean_best_em"]); trunc.append(p["any_truncated"])
            ax.scatter([n] * len(p["seeds"]), [s["best_em"] for s in p["seeds"]],
                       color=color, s=14, alpha=0.55, zorder=2,
                       edgecolors="none")
        ax.plot(xs, ys, color=color, lw=2.0, label=label, zorder=3)
        for x, y, t in zip(xs, ys, trunc):
            if t:
                ax.scatter([x], [y], facecolors="white", edgecolors=color, s=64,
                           marker="D", zorder=4)
    ax.axhline(0.99, color="#8a93a5", lw=0.8, ls="--")
    # upper right is empty (no arm is near 0.99 beyond N=32)
    ax.annotate("stop threshold 0.99", (45, 1.003), fontsize=8, color="#8a93a5")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 96])
    ax.set_xticklabels([8, 16, 32, 64, 96])
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("KV pairs in bank N (curriculum phase)")
    ax.set_ylabel("best eval exact match (mean over seeds)")
    ax.set_title("Best EM vs N (dots = seeds)")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()

    # ---- middle: step-0 warm-transfer EM -----------------------------
    ax = axes[1]
    for arm in ["interface", "interface_softmax", "adaptive"]:
        label, color, _ = STYLE[arm]
        xs = [n for n in [16, 32, 64] if phase(S, arm, n)]
        ys = [phase(S, arm, n)["mean_step0_em"] for n in xs]
        ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels([16, 32, 64])
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("new bank size N (warm-started from N/2)")
    ax.set_ylabel("step-0 EM (first eval after checkpoint load)")
    ax.set_title("Warm-transfer EM — read-side interference\nbefore any retraining")
    ax.grid(alpha=0.25)

    # ---- right: cross-talk vs N --------------------------------------
    ax = axes[2]
    for arm in ["interface", "interface_softmax"]:
        label, color, _ = STYLE[arm]
        xs, ys = [], []
        for n in [8, 16, 32, 64]:
            p = phase(S, arm, n)
            vals = [s["cross_talk_tail"] for s in p["seeds"]
                    if s.get("cross_talk_tail") is not None] if p else []
            if vals:
                xs.append(n); ys.append(sum(vals) / len(vals))
        ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64])
    ax.set_xticklabels([8, 16, 32, 64])
    ax.set_xlabel("bank size N")
    ax.set_ylabel("read_state_cross_talk (tail mean)")
    ax.set_title("Interference floor: mean |q·k| over\nnon-target bank items (interface arms)")
    ax.grid(alpha=0.25)

    # one shared legend below the panels: per-panel legends occluded the
    # adaptive arm's low N=32 points
    handles.append(plt.Line2D([], [], ls="none", marker="D", ms=7,
                              markerfacecolor="white", markeredgecolor="#555"))
    labels.append("phase truncated by infra kill (snapshot, not converged)")
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
