# Interference curve vs N (single-key KV, checkpoint-chained curriculum) — results

> **Date:** 2026-11-09 · **Project:** `zhekahauska/gradmem-linear-read-interference` (Comet, 30 experiments).
> **Provenance:** metrics collected with
> [`scripts/collect_interference_results.py`](../scripts/collect_interference_results.py)
> (raw curves + per-phase summary: [`results/interference_curve.json`](results/interference_curve.json));
> figure by [`scripts/make_interference_figure.py`](../scripts/make_interference_figure.py).
> **Brief & pre-registration:** [`tasks/INTERFERENCE_CURVE_TASK.md`](../tasks/INTERFERENCE_CURVE_TASK.md)
> revisions 2–3; predictions verbatim in
> [`manifests/kv_interference_curve.yaml`](../manifests/kv_interference_curve.yaml) (P1–P3) and
> [`manifests/kv_interference_adaptive.yaml`](../manifests/kv_interference_adaptive.yaml) (P-A1/P-A2).
> **Completeness: PARTIAL.** Both cluster jobs died on 2026-11-03 (an external interrupt, then a
> full disk). 15 of 45 planned runs never started and 9 of the 30 that exist were cut short, so
> P2 is decided but P1 is only tested through N=16, and no arm reached N=96.

## Bottom line

1. **Superposition binds at N=32** under the pre-registered rule. Δ(32) = 0.052 > 0.05, against
   Δ(8) = 0.0005 and Δ(16) = 0.0025. The 0.052 is a **lower bound**. The exact-softmax control
   was killed after ≤2.5k of its 50k steps, while the superposed arm used its full 50k. The seed
   ranges at N=32 do not overlap: the control's worst seed (0.975) beats the superposed arm's best
   (0.948).
2. **The superposed read collapses at N=64.** EM plateaus at 0.37–0.47, with both EM and eval loss
   flat from about 10k to 23k steps. The phase was killed at 45% of budget, and no N=64 control
   exists, so Δ(64) cannot be computed.
3. **Warm transfer gives the cleanest signature (not pre-registered; it comes free with the chain
   design).** Doubling the bank costs the exact read almost nothing: first-eval EM is 0.93 at both
   N=16 and N=32. For the superposed read it costs 0.865 → 0.688 → 0.254 at N=16/32/64.
4. **The binding point matches the record's own interference estimate.** Each read head has key
   width d_h = n_embd/H_r = 128/4 = 32. The 2026-11-03 STATUS estimate puts random-key
   noise/signal at √((N−1)/d_h): 0.47, 0.68, **0.98**, 1.40 at N = 8/16/32/64. Binding happens
   where that ratio reaches 1. The measured interference floor at N ≥ 32 (0.143) also equals the
   random-unit-vector value for 32 dims (E|cos| ≈ 0.142). This is consistent with the estimate,
   not proof of it; a d_h sweep would test it directly (see "Next").
5. **Carried memory fails at every N, as pre-registered (P-A1).** Best EM is 0.82 → 0.48 → 0.15 at
   N=8/16/32. The gap to the separate-token interface arm grows from 0.17 to 0.51 to 0.78. At
   every tested N, writing per-segment slots matters far more than the superposition penalty.

![Interference curve](figures/interference_em_vs_n.svg)

## Pre-registered predictions — verdicts

| prediction | verdict | evidence |
|---|---|---|
| **P1** `interface_softmax` reaches EM ≥ 0.99 at every phase through N=96 | **held through N=16; untested beyond** | ≥0.99 at N=8 (17–19k steps) and N=16 (5.5–9.5k steps). N=32 got only 1–2.5k steps before the disk-full kill (best 0.975–0.982; same trajectory shape as N=16 at that point). N=64 and N=96 never ran. |
| **P2** superposition binds at the first N with Δ(N) > 0.05; holds if Δ ≤ 0.02 through N=96 | **binds at N=32** | Δ = +0.0005 / +0.0025 / **+0.052** at N=8/16/32. Per-seed Δ(32): 0.058, 0.069, 0.030 (seeds 4378/43894/2343). Each softmax value is a lower bound. Cross-talk rises with N as predicted (0.127 → 0.135 → 0.143 → 0.143), but only modestly. |
| **P3** kernel difference small at N=8; divergence only where superposition binds | **held** | Δ ≤ 0.0025 at N=8 and N=16; the kernels diverge at N=32, the binding N. |
| **P-A1** carried arm does NOT reach 0.99 even at small N; forgetting gaps widen with phase | **held (EM part); forgetting-matrix part not checkable here** | Never ≥ 0.99. Best 0.78–0.88 at N=8 with the full 50k budget. The forgetting matrices were not logged to Comet (no per-segment metrics or assets); they exist only in the remote run dirs. |
| **P-A2** per-phase gap to `interface` = effect size of separate memory tokens | **measured** | 0.173 (N=8), 0.514 (N=16), 0.781 (N=32; carried arm truncated but flat from step 0). Budget asymmetry by design: 2 carried tokens vs N stored. |

## Results by phase

Best eval EM per seed (seed order 4378 / 43894 / 2343). Column "ended": `thr` = early stop at
EM ≥ 0.99, `cap` = 50k-step cap, `KILLED` = cut short by the infrastructure events (value is a
snapshot).

| arm | N | best EM per seed | mean | ended | step-0 EM (warm transfer) | interference floor |
|---|---|---|---|---|---|---|
| interface | 8 | 0.9922 / 0.9904 / 0.9914 | **0.9913** | thr @ 22.5k / 19.5k / 23.5k | — (scratch) | 0.127 |
| interface | 16 | 0.9886 / 0.9870 / 0.9926 | **0.9894** | cap / cap / thr @ 47.5k | 0.865 | 0.135 |
| interface | 32 | 0.9178 / 0.9134 / 0.9476 | **0.9263** | cap ×3 | 0.688 | 0.143 |
| interface | 64 | 0.3932 / 0.3692 / 0.4652 | **0.409** | KILLED @ 22.5k / 22.5k / 23k | 0.254 | 0.143 |
| interface_softmax | 8 | 0.9916 / 0.9930 / 0.9908 | **0.9918** | thr @ 18k / 19k / 17k | — (scratch) | 0.163 |
| interface_softmax | 16 | 0.9912 / 0.9922 / 0.9924 | **0.9919** | thr @ 5.5k / 9.5k / 6k | 0.927 | 0.170 |
| interface_softmax | 32 | 0.9754 / 0.9822 / 0.9772 | **0.978** | KILLED @ 2.5k / 1k / 2.5k | 0.929 | 0.176 |
| adaptive | 8 | 0.7950 / 0.8790 / 0.7812 | **0.818** | cap ×3 | — (scratch) | n/a |
| adaptive | 16 | 0.4876 / 0.4974 / 0.4404 | **0.475** | cap ×3 | 0.354 | n/a |
| adaptive | 32 | 0.1494 / 0.1628 / 0.1236 | **0.145** | KILLED @ 2.5k / 2.5k / 3k | 0.139 | n/a |

Not run: `interface` N=96; `interface_softmax` N=64 and N=96; `adaptive` N=64 and N=96 (3 seeds each).

## Readouts

### P2 — Δ(N), and why truncation does not undercut the verdict

Best-so-far EM never decreases as more evals arrive. So a truncated run's best is a lower bound on
what the full phase would have reached. At N=32 the superposed arm finished its full 50k steps,
so its best is final; the control's best can only go up. Δ(32) ≥ 0.052 therefore holds under the
full protocol too.

A caveat on what "binds" means here: it holds **under the 50k budget**. The superposed N=32 curve
was still rising slowly at the cap (seed 2343: 0.935 → 0.948 over the last 6k steps), and at N=16
it fell just short of 0.99 for two seeds (0.987/0.989, both still rising at 50k). Part of the N=32
gap is therefore sample efficiency. N=64 looks different: EM and loss have been flat since about
10k steps (seed 2343 at 0.455 → 0.454 across 10k–23k, eval loss within 0.55–0.56 over the last
5k steps). That is a plateau,
not a run cut off mid-climb.

### Warm transfer — a cleaner read-side probe (not pre-registered)

Each phase's first eval runs the previous phase's checkpoint on a bank twice as large, before any
retraining. The write path is per-segment and identical (constant load per segment), so this
measures how much the **read** is hurt by the larger bank:

- `interface_softmax`: 0.927 (N=16), 0.929 (N=32). Exact softmax over a larger unfolded bank
  barely notices the extra items.
- `interface`: 0.865 (N=16), 0.688 (N=32), 0.254 (N=64). The superposed state loses accuracy
  steeply as items pile into a fixed 32×32-per-head state.
- `adaptive`: 0.354 (N=16), 0.139 (N=32). The carried memory loses even more.

### Interference floor (`read_state_cross_talk`)

The metric is the mean |q̂·k̂| over the **non-target** bank items, measured on the last
instrumented layer with q and k normalized ([`grad_memgpt.py:1732`](../grad_memgpt.py)). Both arms
compute it the same way. Softmax normalization suppresses that floor; the linear read sums it.

- `interface`: 0.127 → 0.135 → 0.143 → 0.143. At N=8 it sits below the random-vector value; from
  N=32 on it saturates at **0.142, the expected |cos| between random unit vectors in d_h = 32
  dims**. At N=8 the model keeps non-target keys somewhat apart; from N ≥ d_h they behave like
  random vectors.
- `interface_softmax`: 0.163 → 0.170 → 0.176, above random. The exact read tolerates a higher
  floor without losing EM.
- The floor's rise is small (+13%). The useful quantity is what the linear read accumulates, about
  √(N−1) × floor relative to a unit target signal: ≈ 0.34 / 0.52 / 0.79 / 1.14 at N = 8/16/32/64.
  That matches the record's analytic √((N−1)/d_h) (STATUS 2026-11-03) and crosses 1 between N=32
  and N=64. On this reading the binding N is set by the key width, N* ≈ d_h = 32, not by training.
  This is a **hypothesis consistent with all four points**, not yet a test.

### Carried memory (P-A1 / P-A2)

The adaptive sgd arm cannot reach 0.99 even with 4 segments (N=8). Its best is 0.78–0.88 after the
full 50k steps, and each doubling roughly halves it again: 0.82 → 0.48 → 0.15. N=32 was cut at
2.5–3k steps, but that curve was flat from step 0 (mean step-0 EM 0.139, mean best 0.145), so the
truncation hides nothing upward. The forgetting matrices, which the brief makes a **primary** readout for this arm, are
**not on Comet**: the runs logged only aggregate eval metrics. N=8 and N=16 are complete, so their
matrices should exist in the remote `runs/N{8,16}-K2V2-V62_1M/interf_adaptive_N*/run_*/` dirs.

### Sample efficiency

Steps to EM ≥ 0.99, from a warm start, are 5.5–9.5k for the exact read at N=16. The superposed read
needed 47.5k for one seed and did not get there within 50k for the other two. From scratch at N=8
the two arms are close (17–19k vs 19.5–23.5k). The superposition cost grows with N well before it
shows up as an accuracy gap.

## What happened to the runs (timeline, local time)

- **Curve job** (`ccde14ed`; arms in sequence, the 3 seeds of each arm in parallel). The interface
  chains started **10-29 17:51**: N8 took 2.5–3.1 h, N16 13.7–14.9 h, N32 37.2–37.4 h. N64 started
  11-01 00:04–00:46. **All three N64 phases died 11-03 10:51–11:29** with
  `InterruptedExperiment('zhekahauska')` at 22.5–23k steps, about 59 h into the phase.
- **Adaptive job** (`7c2ed873`, separate run spec, started **11-02 17:43**): N8 took 4.5–4.8 h, N16 10.5–11.0 h.
  N32 started 08:43–09:30. **All three ended 11-03 11:08**, the same window as the interface kill,
  with no error recorded.
- **Softmax chains** started 11-03 11:29, right after the last interface phase died, so the curve
  job's scheduler survived or was resumed. N8 took 2.2–2.5 h, N16 1.8–2.8 h, and N32 started
  15:38–16:49. **All three ended 11-03 18:15.** Two record `OSError(28, 'No space left on device')`;
  the third ended the same minute with no error recorded (plausibly the same cause, with nothing
  left to log to).
- **Chain consequence** ([`run_interference_chain.py`](../scripts/run_interference_chain.py)): a
  failed phase stops its chain, and only `done` phases are skipped on rerun. A rerun would
  therefore restart each chain's killed phase from the previous phase's checkpoint, with weights
  only and step 0, rather than resuming mid-phase.
- Two jobs stopping within the same ~40 minutes points to a cluster-level event or a manual stop.
  Comet cannot tell which. The disk-full cause is certain for the softmax N32 runs; what filled
  the disk (most likely accumulated checkpoints) has to be checked on the remote.

### Cost reality

Measured per-step wall time, including evals:

| arm | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| interface | 0.48 s | 1.05 s | 2.7 s | 9.3 s |

A full 50k-step N=64 phase works out to about 5.4 days. N=96 runs at B=16 × 4 accumulation; if it
is 2× slower again (an extrapolation), the interface chain alone needs about 2.5 weeks at full
budget, against the pre-registered "~3–5 days wall" for the whole job. When killed, the interface
chains had already used 4.7 days with N64 45% done. Threshold stops shorten phases a lot for arms
that converge (softmax N16 finished in about 2 h).

## Caveats

- **Truncated phases:** softmax N32 (2–5% of its budget), interface N64 (45%), adaptive N32 (5–6%).
  The direction of each bias is stated above. None of them flips a verdict, but P1 at N ≥ 32 and
  Δ(64) are genuinely missing.
- **The curriculum confound is intended** (pre-registered fairness note). Arms are compared under
  identical curricula; absolute N readings mix in the warm start.
- **Single-key retrieval only.** The ordered-concat ladder task was not run at larger N.
- **3 seeds.** The N=32 seed ranges do not overlap, but per-seed Δ(32) ranges from 0.030 to 0.069.

## Next (options for the PI and user, not run)

1. **Infrastructure before any rerun:** check `df` and `du runs/` on the remote, prune non-best
   checkpoints of completed phases, and check that the `runs/interference_curve/*.chain.json` state
   files survived the disk-full event (a write may have been cut off mid-file).
2. **Cheapest decision-relevant rerun: the softmax chains only**, resuming into N32 and then N64
   (the wrapper skips the `done` N8 and N16 phases). This closes P1 at N=32/64 and gives Δ(64).
   The softmax arm stops at the threshold quickly, so this should take hours to a few days.
   Interface N64 is already informative (a plateau at 0.41); its N96 phase would cost 10+ days and
   is unlikely to change the conclusion.
3. **Direct test of the capacity hypothesis:** the planned `kv_interface_softmax.yaml` arm2 (key
   width 64 via `read_state_heads: 2`, config-only) on this curriculum at N=32 and N=64. If
   N* ≈ d_h, the binding point moves to about 64.
4. **Mitigation the record already plans:** "If superposition binds: coarse top-k segment routing"
   (PROGRESS missing-experiment #1). That condition is now met.
5. **Adaptive forgetting matrices:** pull the N8/N16 per-segment eval artifacts from the remote run
   dirs to close P-A1's forgetting-matrix part.
