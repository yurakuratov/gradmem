#!/usr/bin/env python
"""Collect the linear-read ladder + interface-conditioning results from Comet.

Usage (needs HTTPS_PROXY if comet is not directly reachable):

    HTTPS_PROXY=http://127.0.0.1:10809 COMET_API_KEY=... \
        uv run python scripts/collect_comet_results.py [--out reports/results/results.json]

Groups the 21 experiments (7 arms x 3 seeds) of project `gradmem-linear-read`
by arm (parsed from the run name) and seed (from the logged `seed` param),
pulls the eval curves that matter for the H3 ladder and the interface
conditioning ablation, and writes a JSON with per-experiment summaries
(best/final values + the step of the best) — the source for the results
report tables. Read-only w.r.t. the project.
"""

from __future__ import annotations

import argparse
import json
import os

# Arm registry: run-name suffix (see configs' run_naming keys) -> ladder arm.
ARMS = {
    "ns4_m2_K2_hopfield_bank": "bank_prefix_control",
    "ns4_m2_K2_hopfield_ds_N8-K2V2-V62-S4-ordered_pair_cross_1M": "hopfield_control",
    "ns4_m2_K2_hopfield_lsr_sitesall_rh4": "linear_all",
    "ns4_m2_K2_hopfield_lsr_sitesall_rh4_ds_N8-K2V2-V62_1M": "linear_single_key_regression",
    "ns4_m2_K2_hopfield_lsr_sitesall_rh4_exact": "linear_exact_softmax",
    "ns4_m2_K2_hopfield_lsr_sitesinterface_rh4": "linear_interface",
    "ns4_m2_K2_hopfield_lsr_sitesinterface_rh4_icondFalse": "linear_interface_nocond",
}

# eval curves to pull (comet prefixes the HF metric name with eval/).
CURVES = [
    "eval/loss",
    "eval/exact_match",
    "eval/target_loss",
    "eval/two_key_both_values_exact_match",
    "eval/two_key_value1_exact_match",
    "eval/two_key_value2_exact_match",
    "eval/two_key_cross_segment_exact_match",
    "eval/two_key_same_segment_exact_match",
    "eval/two_key_both_single_exact_match",
    "eval/two_key_joint_exact_match_given_both_single_correct",
    "eval/two_key_reversal_error_rate",
    "eval/read_state_gate_abs_mean",
    "eval/read_state_mu_over_x_norm_ratio",
    "eval/read_state_beta_mean",
    "eval/read_state_cross_talk",
    "eval/forget_diag_mean",
    "eval/forget_seg0_final",
    "eval/forget_slope",
]


def curve_stats(points):
    """[{'step': s, 'metricValue': v}, ...] -> {'last_step', 'final', 'best', 'best_step', 'n'}.

    'best' is the MAX of the curve (EM-style); loss curves get best=final-side
    minimum under 'min'.
    """
    if not points:
        return None
    steps = [int(p["step"]) for p in points]
    vals = [float(p["metricValue"]) for p in points]
    i_best = max(range(len(vals)), key=lambda i: vals[i])
    i_min = min(range(len(vals)), key=lambda i: vals[i])
    return {
        "n": len(vals),
        "last_step": steps[-1],
        "final": vals[-1],
        "best": vals[i_best],
        "best_step": steps[i_best],
        "min": vals[i_min],
        "min_step": steps[i_min],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/results/results.json")
    ap.add_argument("--workspace", default="zhekahauska")
    ap.add_argument("--project", default="gradmem-linear-read")
    args = ap.parse_args()

    from comet_ml import API

    api = API(api_key=os.environ["COMET_API_KEY"])
    exps = api.get_experiments(workspace=args.workspace, project_name=args.project)
    print(f"{len(exps)} experiments in {args.workspace}/{args.project}")

    out = []
    for e in sorted(exps, key=lambda x: x.name):
        arm = ARMS.get(e.name, "UNKNOWN:" + e.name)
        params = {p["name"]: p.get("valueCurrent") for p in e.get_parameters_summary()}
        md = e.get_metadata()
        rec = {
            "arm": arm,
            "run_name": e.name,
            "experiment_key": e.id,
            "seed": int(params.get("seed", -1)),
            "exp_path": params.get("exp_path"),
            "data_path": params.get("data_path"),
            "duration_ms": md.get("durationMillis"),
            "running": md.get("running"),
            "has_crashed": md.get("hasCrashed"),
            "error": md.get("error"),
            "curves": {},
        }
        available = {m["name"] for m in e.get_metrics_summary()}
        for name in CURVES:
            if name not in available:
                continue
            try:
                pts = e._get_metric(name)
            except Exception as exc:  # keep going; record the miss
                rec["curves"][name] = {"error": str(exc)}
                continue
            rec["curves"][name] = curve_stats(pts)
        out.append(rec)
        last = rec["curves"].get("eval/loss", {})
        tk = rec["curves"].get("eval/two_key_both_values_exact_match", {})
        print(f"{arm:32s} seed={rec['seed']:6d} last_step={last.get('last_step', '?'):>7} "
              f"two_key_best={tk.get('best', float('nan')):.4f}@{tk.get('best_step', '?')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out} ({len(out)} experiments)")


if __name__ == "__main__":
    main()
