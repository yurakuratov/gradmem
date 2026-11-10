#!/usr/bin/env python
"""Collect the interference-curve results from Comet (project
`gradmem-linear-read-interference`) into
`reports/results/interference_curve.json` — the data source for
`reports/INTERFERENCE_CURVE_RESULTS.md` and `make_interference_figure.py`.

Usage (needs HTTPS_PROXY if comet is not directly reachable):

    HTTPS_PROXY=http://127.0.0.1:10809 COMET_API_KEY=... \
        uv run python scripts/collect_interference_results.py

Pulls every experiment of the project (run names `interf_<arm>_N<N>`, the
checkpoint-chained curriculum of tasks/INTERFERENCE_CURVE_TASK.md revisions
2-3: arms `interface` / `interface_softmax` from
manifests/kv_interference_curve.yaml, arm `adaptive` from
manifests/kv_interference_adaptive.yaml), the eval curves that matter for the
pre-registered readouts (best EM per phase, warm-transfer step-0 EM,
read-state cross-talk), run metadata (timestamps, crash, error), and emits a
per-phase summary table. Read-only w.r.t. the project.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

SEEDS = [4378, 43894, 2343]
NS = [8, 16, 32, 64, 96]
ARMS = ["interface", "interface_softmax", "adaptive"]

CURVES = [
    "eval/exact_match", "eval/loss", "eval/target_loss", "eval/token_accuracy",
    "eval/read_state_cross_talk", "eval/read_state_beta_mean",
    "eval/read_state_gate_abs_mean", "eval/read_state_mu_over_x_norm_ratio",
    "eval/inner_loss",
]
PARAMS = ["seed", "exp_path", "data_path", "max_steps", "stop_on_em_threshold",
          "early_stopping_patience", "warmup_steps", "per_device_batch_size",
          "total_batch_size", "learning_rate", "eval_steps", "init_checkpoint",
          "n_segments", "per_segment_eval"]


def retry(fn, n=4):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as exc:
            last = exc
            time.sleep(3 * (i + 1))
    raise last


def curve(rec, name):
    c = rec["curves"].get(name, [])
    return c if isinstance(c, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/results/interference_curve.json")
    ap.add_argument("--workspace", default="zhekahauska")
    ap.add_argument("--project", default="gradmem-linear-read-interference")
    args = ap.parse_args()

    from comet_ml import API

    api = API(api_key=os.environ["COMET_API_KEY"])
    exps = api.get_experiments(workspace=args.workspace, project_name=args.project)
    print(f"{len(exps)} experiments in {args.workspace}/{args.project}")

    raw = []
    for e in sorted(exps, key=lambda x: (x.name, x.id)):
        rec = {"run_name": e.name, "experiment_key": e.id}
        md = retry(lambda: e.get_metadata())
        rec["start_ms"] = md.get("startTimeMillis")
        rec["end_ms"] = md.get("endTimeMillis")
        rec["duration_ms"] = md.get("durationMillis")
        rec["has_crashed"] = md.get("hasCrashed")
        rec["error"] = md.get("error")
        params = retry(lambda: {p["name"]: p.get("valueCurrent")
                                for p in e.get_parameters_summary()})
        rec["params"] = {k: params.get(k) for k in PARAMS}
        avail = {m["name"] for m in retry(lambda: e.get_metrics_summary())}
        rec["curves"] = {}
        for name in CURVES:
            if name not in avail:
                continue
            pts = retry(lambda n=name: e._get_metric(n))
            rec["curves"][name] = [{"step": int(p["step"]), "v": float(p["metricValue"])}
                                   for p in pts]
        raw.append(rec)
        em = curve(rec, "eval/exact_match")
        print(f"{rec['run_name']:30s} seed={params.get('seed')} crashed={rec['has_crashed']} "
              f"n_evals={len(em)}", flush=True)

    # per-phase summary: best EM (the pre-registered readout), warm-transfer
    # step-0 EM, threshold-crossing step, cross-talk tail mean, truncation.
    summary = {}
    for arm in ARMS:
        for n in NS:
            rows = []
            for s in SEEDS:
                rs = [r for r in raw if r["run_name"] == f"interf_{arm}_N{n}"
                      and str(r["params"].get("seed")) == str(s)]
                if not rs:
                    continue
                r = rs[0]
                em = curve(r, "eval/exact_match")
                ct = curve(r, "eval/read_state_cross_talk")
                if not em:
                    continue
                i = max(range(len(em)), key=lambda i: em[i]["v"])
                first99 = next((p["step"] for p in em if p["v"] >= 0.99), None)
                max_steps = int(r["params"].get("max_steps") or 50000)
                rows.append({
                    "seed": s,
                    "best_em": em[i]["v"], "best_step": em[i]["step"],
                    "final_em": em[-1]["v"], "last_step": em[-1]["step"],
                    "step0_em": em[0]["v"], "first_em_ge_099_step": first99,
                    "cross_talk_tail": (st.mean(p["v"] for p in ct[-5:])
                                        if ct else None),
                    "has_crashed": r["has_crashed"], "error": r["error"],
                    # truncated = stopped before the cap AND before the 0.99
                    # threshold (killed by the infrastructure events)
                    "truncated": em[-1]["step"] < max_steps and first99 is None,
                })
            if rows:
                summary[f"{arm}_N{n}"] = {
                    "mean_best_em": st.mean(x["best_em"] for x in rows),
                    "mean_step0_em": st.mean(x["step0_em"] for x in rows),
                    "any_truncated": any(x["truncated"] for x in rows),
                    "seeds": rows,
                }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"raw": raw, "summary": summary}, f, indent=1)
    print(f"wrote {args.out} ({len(raw)} experiments, {len(summary)} phases)")

    print(f"\n{'arm':18s} {'N':>3} {'bestEM':>7} {'step0':>6} {'trunc':>5}")
    for key, s in summary.items():
        arm, n = key.rsplit("_N", 1)
        print(f"{arm:18s} {n:>3} {s['mean_best_em']:7.4f} {s['mean_step0_em']:6.3f} "
              f"{str(s['any_truncated']):>5}")


if __name__ == "__main__":
    main()
