"""
Plot cramming sweep results from a directory of JSON files.

Reads every JSON matching --pattern inside --results_dir, extracts an x-axis
value from a dotted config/result path (--x_field), and plots:
  - mean information gain (bar + per-sample scatter)
  - mean accuracy (secondary y-axis)
  - number of inner steps (annotated on bars)

Reusable for any hyperparameter sweep (layers, rank, lr, …).
"""

import argparse
import fnmatch
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def resolve(obj, dotted_key):
    """Resolve 'config.layer_idx' or 'results.0.mean_info_gain' style paths."""
    for part in dotted_key.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def main():
    ap = argparse.ArgumentParser(description="Plot cramming sweep results")
    ap.add_argument("--results_dir", default="cramming_results")
    ap.add_argument("--pattern", default="*",
                    help="glob pattern matched against filenames (without .json)")
    ap.add_argument("--x_field", default="config.layer_idx",
                    help="dotted path to the x-axis value inside each JSON")
    ap.add_argument("--x_label", default="Hyperparameter")
    ap.add_argument("--title", default="Cramming sweep")
    ap.add_argument("--output", default=None,
                    help="output image path (shows interactively if omitted)")
    args = ap.parse_args()

    entries = []
    for fname in sorted(os.listdir(args.results_dir)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        if not fnmatch.fnmatch(stem, f"*{args.pattern}*"):
            continue
        with open(os.path.join(args.results_dir, fname)) as f:
            data = json.load(f)
        if not data.get("results"):
            continue
        r = data["results"][-1]
        entries.append(dict(
            x=resolve(data, args.x_field),
            mean_info_gain=r["mean_info_gain"],
            info_gains=r.get("info_gains", []),
            mean_acc=r["mean_acc"],
            accs=r.get("accs", []),
            inner_steps=r.get("inner_steps", None),
        ))

    if not entries:
        print(f"No matching JSON files in {args.results_dir} "
              f"for pattern '*{args.pattern}*'")
        return

    entries.sort(key=lambda e: e["x"])
    xs = [e["x"] for e in entries]
    ig = [e["mean_info_gain"] for e in entries]
    accs = [e["mean_acc"] for e in entries]

    fig, ax1 = plt.subplots(figsize=(max(8, len(xs) * 0.8), 5))

    # bars: information gain
    bars = ax1.bar(range(len(xs)), ig, color="#4c8bf5", alpha=0.85, zorder=2,
                   label="Info gain (mean)")

    # scatter per-sample info gains
    for i, e in enumerate(entries):
        if e["info_gains"]:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(e["info_gains"]))
            ax1.scatter([i + j for j in jitter], e["info_gains"],
                        s=18, color="#1a3a6e", alpha=0.5, zorder=3)

    # annotate inner steps on each bar
    for i, (bar, e) in enumerate(zip(bars, entries)):
        if e["inner_steps"] is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{e["inner_steps"]}', ha="center", va="bottom",
                     fontsize=8, color="#333")

    ax1.set_xticks(range(len(xs)))
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlabel(args.x_label)
    ax1.set_ylabel("Information gain (sum of token losses)")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # secondary axis: accuracy
    ax2 = ax1.twinx()
    ax2.plot(range(len(xs)), accs, "o-", color="#e8433e", markersize=5,
             linewidth=1.5, label="Accuracy (mean)", zorder=4)
    ax2.set_ylabel("Reconstruction accuracy")
    ax2.set_ylim(min(min(accs) - 0.02, 0.9), 1.005)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)

    ax1.set_title(args.title)
    fig.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
