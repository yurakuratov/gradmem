#!/usr/bin/env python
"""EM-vs-step figure for the results report (reports/figures/em_curves.svg).

Left: every ladder arm, mean over the 3 seeds (thin lines = seeds).
Right: the conditioning ablation pair — same seeds, same data, the knob is
the only difference.

Run:  uv run python scripts/make_em_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(HERE, "..", "reports", "results", "em_curves.json")
OUT = os.path.join(HERE, "..", "reports", "figures", "em_curves.svg")

STYLE = {
    "hopfield_control":         ("(a) Hopfield one-shot (control)", "#8a93a5", "-"),
    "bank_prefix_control":      ("(b) Bank-prefix control",        "#b08968", "-"),
    "linear_exact_softmax":     ("(c) Exact-softmax heads",        "#1e7d43", "-"),
    "linear_all":               ("(d) Linear-state, all positions", "#6a4fa3", "-"),
    "linear_interface":         ("(d\u2032) Interface, conditioned", "#c77b28", "-"),
    "linear_interface_nocond":  ("(d\u2033) Interface, query-blind", "#c0392b", "-"),
}


def main():
    data = json.load(open(CURVES))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True)

    ax = axes[0]
    for arm, (label, color, _) in STYLE.items():
        seeds = data.get(arm, {})
        for seed, curve in sorted(seeds.items()):
            xs = [s / 1000 for s, _ in curve]
            ys = [v for _, v in curve]
            ax.plot(xs, ys, color=color, alpha=0.25, lw=0.9)
        if seeds:
            steps = sorted({s for c in seeds.values() for s, _ in c})
            import numpy as np
            mean = []
            for st in steps:
                vals = [dict(c).get(st) for c in seeds.values()]
                vals = [v for v in vals if v is not None]
                mean.append((st / 1000, sum(vals) / len(vals) if vals else None))
            mx = [p[0] for p in mean if p[1] is not None]
            my = [p[1] for p in mean if p[1] is not None]
            ax.plot(mx, my, color=color, lw=2.2, label=label)
    ax.set_title("Read-interface ladder — ordered two-key concat (3 seeds each)")
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("two-key both-values exact match")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.9)

    ax = axes[1]
    for arm in ("linear_interface", "linear_interface_nocond"):
        label, color, _ = STYLE[arm]
        for seed, curve in sorted(data[arm].items()):
            xs = [s / 1000 for s, _ in curve]
            ys = [v for _, v in curve]
            ax.plot(xs, ys, color=color, alpha=0.45, lw=1.2,
                    label=f"{label}, seed {seed}")
    ax.set_title("Conditioning ablation — paired seeds (the knob is the only difference)")
    ax.set_xlabel("training step (thousands)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT, format="svg")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
