# Interference curve vs N (single-key KV, checkpoint-chained curriculum) — results

> **Date:** 2026-11-09, updated 2026-11-12 with the resume job's results ·
> **Project:** `zhekahauska/gradmem-linear-read-interference` (Comet, 39 experiments).
> **Provenance:** metrics collected with
> [`scripts/collect_interference_results.py`](../scripts/collect_interference_results.py)
> (raw curves + per-phase summary: [`results/interference_curve.json`](results/interference_curve.json);
> the collector disambiguates rerun generations by newest end time);
> figure by [`scripts/make_interference_figure.py`](../scripts/make_interference_figure.py).
> **Brief & pre-registration:** [`tasks/INTERFERENCE_CURVE_TASK.md`](../tasks/INTERFERENCE_CURVE_TASK.md)
> revisions 2–3 (P1–P3, P-A1/P-A2) and [`tasks/SOFTMAX_RESUME_TASK.md`](../tasks/SOFTMAX_RESUME_TASK.md)
> (P-S1–P-S3); predictions verbatim in the manifest headers.
> **Completeness:** the exact-softmax control is COMPLETE through N=96 (resumed after the
> 2026-11-03 infrastructure failures; every phase a clean threshold stop). The `interface` arm is
> complete through N=32 and plateaued-but-truncated at N=64; it never ran N=96. The `adaptive` arm
> is complete through N=16 and truncated at N=32. P1 and P2 are now decided without caveats;
> Δ(96) is undefined (one arm missing).

## Bottom line

1. **P1 confirmed in full: the exact-softmax control reaches EM ≥ 0.99 at every phase through
   N=96**, every phase a threshold stop (N=96 crossed 0.99 within 0.5–2.5k steps of its warm
   start). The write path and protocol are exonerated at every bank size — whatever fails at
   larger N is read-side.
2. **Superposition binds at N=32** under the pre-registered rule. Δ(32) = **0.065** > 0.05 with
   both sides now converged (Δ(8) = 0.0005, Δ(16) = 0.0025). Per-seed Δ(32) = 0.073 / 0.080 /
   0.044 — two of three seeds above the line, mean well above; the seed ranges do not overlap.
3. **At N=64 the superposed read collapses.** Δ(64) = **0.58** (control 0.991, interface 0.409).
   The interface N64 phases were killed at 45% of budget, but their EM and eval loss were flat
   from ~10k to 23k steps — a plateau, not a mid-climb cut. The control at the same protocol
   crossed 0.99 in 4.5–8.5k steps.
4. **Warm transfer gives the cleanest signature** (step-0 EM after loading the previous phase's
   checkpoint, before any retraining): the exact read is essentially N-invariant — 0.927 / 0.929 /
   0.940 / 0.965 at N=16/32/64/96 (rising, because later warm starts come from better-trained
   checkpoints). The superposed read falls off a cliff: 0.865 / 0.688 / 0.254 at N=16/32/64.
5. **The binding point matches the record's own interference estimate.** Each read head has key
   width d_h = n_embd/H_r = 128/4 = 32. The 2026-11-03 STATUS estimate puts random-key
   noise/signal at √((N−1)/d_h): 0.47 / 0.68 / **0.98** / 1.40 at N = 8/16/32/64. Binding at
   N ≈ d_h, total collapse at N = 2·d_h. The measured interference floor at N ≥ 32 (0.143) also
   equals the random-unit-vector value for 32 dims (E|cos| ≈ 0.142). Consistent with the estimate,
   not proof of it; a d_h sweep would test it directly (see "Next").
6. **Carried memory fails at every N, as pre-registered (P-A1).** Best EM is 0.82 → 0.48 → 0.15 at
   N=8/16/32. The gap to the separate-token interface arm grows from 0.17 to 0.51 to 0.78. At
   every tested N, writing per-segment slots matters far more than the superposition penalty.

![Interference curve](figures/interference_em_vs_n.svg)

## Pre-registered predictions — verdicts

| prediction | verdict | evidence |
|---|---|---|
| **P1** `interface_softmax` reaches EM ≥ 0.99 at every phase through N=96 | **held at every phase** | N=8: 17–19k steps; N=16: 5.5–9.5k; N=32: 8.5–11k (resume rerun, clean threshold stops); N=64: 4.5–8.5k; N=96: **0.5–2.5k**. Best EM 0.9900–0.9934 at every phase. |
| **P2** superposition binds at the first N with Δ(N) > 0.05; holds if Δ ≤ 0.02 through N=96 | **binds at N=32** | Δ = +0.0005 / +0.0025 / **+0.065** / **+0.582** at N=8/16/32/64. Per-seed Δ(32): 0.073 / 0.080 / 0.044. Cross-talk rises with N as predicted (0.127 → 0.135 → 0.143 → 0.143), modestly. Δ(96) undefined (interface N96 never ran). |
| **P3** kernel difference small at N=8; divergence only where superposition binds | **held** | Δ ≤ 0.0025 at N=8 and N=16; the kernels diverge at N=32, the binding N. |
| **P-A1** carried arm does NOT reach 0.99 even at small N; forgetting gaps widen with phase | **held (EM part); forgetting-matrix part not checkable here** | Never ≥ 0.99. Best 0.78–0.88 at N=8 with the full 50k budget. The forgetting matrices were not logged to Comet; they exist in the remote run dirs (archived at `~/archives/interference_curve_runs_20261103.tar.gz`). |
| **P-A2** per-phase gap to `interface` = effect size of separate memory tokens | **measured** | 0.173 (N=8), 0.514 (N=16), 0.781 (N=32; carried arm truncated but flat from step 0). Budget asymmetry by design: 2 carried tokens vs N stored. |
| **P-S1** resumed N32 phases reach EM ≥ 0.99 (resume-job pre-registration) | **held** | 0.9908 / 0.9930 / 0.9912 at 11k / 8.5k / 10.5k steps — within the predicted ≤10–15k band. |
| **P-S2** softmax N64 reaches EM ≥ 0.99 — the write path is NOT the N=64 bottleneck | **held decisively** | 0.9906 / 0.9918 / 0.9912 at 8.5k / 4.5k / 6k steps. The plateau outcome (which would have re-indicted protocol/write) did not occur. |
| **P-S3** if N64 holds, N96 runs; no directional bet | **held** | N96: 0.9900 / 0.9902 / 0.9934 at 1k / 0.5k / 2.5k steps. Step-0 EM 0.96–0.97 — the exact read transfers almost perfectly to a 48-segment bank. |

## Results by phase

Best eval EM per seed (seed order 4378 / 43894 / 2343). Column "ended": `thr` = early stop at
EM ≥ 0.99, `cap` = 50k-step cap, `KILLED` = cut short by the infrastructure events (value is a
snapshot). Step-0 EM = first eval after the warm-start checkpoint load.

| arm | N | best EM per seed | mean | ended | step-0 EM | interference floor |
|---|---|---|---|---|---|---|
| interface | 8 | 0.9922 / 0.9904 / 0.9914 | **0.9913** | thr @ 22.5k / 19.5k / 23.5k | — (scratch) | 0.127 |
| interface | 16 | 0.9886 / 0.9870 / 0.9926 | **0.9894** | cap / cap / thr @ 47.5k | 0.865 | 0.135 |
| interface | 32 | 0.9178 / 0.9134 / 0.9476 | **0.9263** | cap ×3 | 0.688 | 0.143 |
| interface | 64 | 0.3932 / 0.3692 / 0.4652 | **0.409** | KILLED @ 22.5k / 22.5k / 23k | 0.254 | 0.143 |
| interface_softmax | 8 | 0.9916 / 0.9930 / 0.9908 | **0.9918** | thr @ 18k / 19k / 17k | — (scratch) | 0.163 |
| interface_softmax | 16 | 0.9912 / 0.9922 / 0.9924 | **0.9919** | thr @ 5.5k / 9.5k / 6k | 0.927 | 0.170 |
| interface_softmax | 32 | 0.9908 / 0.9930 / 0.9912 | **0.9917** | thr @ 11k / 8.5k / 10.5k | 0.929 | 0.173 |
| interface_softmax | 64 | 0.9906 / 0.9918 / 0.9912 | **0.9912** | thr @ 8.5k / 4.5k / 6k | 0.940 | 0.176 |
| interface_softmax | 96 | 0.9900 / 0.9902 / 0.9934 | **0.9912** | thr @ 1k / 0.5k / 2.5k | 0.965 | 0.177 |
| adaptive | 8 | 0.7950 / 0.8790 / 0.7812 | **0.818** | cap ×3 | — (scratch) | n/a |
| adaptive | 16 | 0.4876 / 0.4974 / 0.4404 | **0.475** | cap ×3 | 0.354 | n/a |
| adaptive | 32 | 0.1494 / 0.1628 / 0.1236 | **0.145** | KILLED @ 2.5k / 2.5k / 3k | 0.139 | n/a |

Never run: `interface` N=96; `adaptive` N=64 and N=96 (3 seeds each). The softmax N32 rows are
the resume-job reruns (11-10/11); the 11-03 disk-full kills of the first N32 generation are part
of the timeline below, and their truncated values (0.975–0.982, rising) are superseded.

## Readouts

### P2 — Δ(N), both sides converged

| N | Δ(N) = softmax − interface | note |
|---|---|---|
| 8 | +0.0005 | parity (as at convergence in the ladder) |
| 16 | +0.0025 | parity, though interface needs ~5× more steps |
| 32 | **+0.0654** | **binds**; per-seed 0.073 / 0.080 / 0.044; both arms converged (interface cap ×3, control threshold ×3) |
| 64 | **+0.5820** | per-seed 0.597 / 0.527 / 0.622; control converged, interface plateaued (see caveat below) |
| 96 | undefined | interface N96 never ran |

The N=64 caveat, once, plainly: the interface phases used 22.5–23k of their 50k steps. Best EM
can only rise with more steps. But EM and eval loss were flat across the last ~13k steps (seed
2343: EM 0.455 → 0.454, loss within 0.55–0.56), and the same-arm trend (0.99 → 0.99 → 0.93 →
0.41) plus the control's flat 0.99 make recovery to anywhere near 0.99 implausible. Δ(64) ≈ 0.58
is the honest reading; a pedantic reading would call it a lower bound.

A residual caveat on "binds": it holds **under the 50k budget**. The interface N=32 curve was
still rising slowly at its cap (seed 2343: 0.935 → 0.948 over the last 6k steps), and at N=16 it
fell just short of 0.99 for two seeds (0.987/0.989, both still rising at 50k). Part of the N=32
gap is therefore sample efficiency. N=64 is different in kind: a plateau.

### Warm transfer — a clean read-side probe (not pre-registered)

Each phase's first eval runs the previous phase's checkpoint on a bank twice as large, before any
retraining. The write path is per-segment and identical (constant load per segment), so this
measures how much the **read** is hurt by the larger bank:

- `interface_softmax`: 0.927 (N=16), 0.929 (N=32), 0.940 (N=64), 0.965 (N=96). Exact softmax
  over a larger unfolded bank barely notices the extra items — the mild rise reflects
  better-trained warm-start checkpoints, not an N effect.
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
- `interface_softmax`: 0.163 → 0.170 → 0.173 → 0.176 → 0.177, above random and drifting up
  slightly. The exact read tolerates a higher floor without losing EM — it never needs to shape
  the superposed state to stay readable.
- The floor's rise is small (+13%). The useful quantity is what the linear read accumulates, about
  √(N−1) × floor relative to a unit target signal: ≈ 0.34 / 0.52 / 0.79 / 1.14 at N = 8/16/32/64.
  That matches the record's analytic √((N−1)/d_h) (STATUS 2026-11-03) and crosses 1 between N=32
  and N=64. On this reading the binding N is set by the key width, N* ≈ d_h = 32, not by training.
  This is a **hypothesis consistent with all the data**, not yet a test.

### Carried memory (P-A1 / P-A2)

The adaptive sgd arm cannot reach 0.99 even with 4 segments (N=8). Its best is 0.78–0.88 after the
full 50k steps, and each doubling roughly halves it again: 0.82 → 0.48 → 0.15. N=32 was cut at
2.5–3k steps, but that curve was flat from step 0 (mean step-0 EM 0.139, mean best 0.145), so the
truncation hides nothing upward. The forgetting matrices, a **primary** readout for this arm, are
**not on Comet**: only aggregate eval metrics were logged. The N=8/N=16 artifacts are in the
remote archive (`~/archives/interference_curve_runs_20261103.tar.gz`, login node).

### Sample efficiency

Steps to EM ≥ 0.99 (all arms, warm-started phases):

| arm | N=8 (scratch) | N=16 | N=32 | N=64 | N=96 |
|---|---|---|---|---|---|
| interface_softmax | 17–19k | 5.5–9.5k | 8.5–11k | 4.5–8.5k | **0.5–2.5k** |
| interface | 19.5–23.5k | 47.5k (1 of 3; others capped) | never | never | — |

The exact read gets *cheaper* as the bank grows (warm starts improve and the read transfers);
the superposed read gets drastically more expensive and then fails outright. Superposition's
cost is visible in sample efficiency at N=16 — well before the accuracy gap opens at N=32.

## What happened to the runs (timeline, local time)

### The original jobs (2026-10-29 → 11-03)

- **Curve job** (`ccde14ed`; arms in sequence, the 3 seeds of each arm in parallel). The interface
  chains started **10-29 17:51**: N8 took 2.5–3.1 h, N16 13.7–14.9 h, N32 37.2–37.4 h. N64 started
  11-01 00:04–00:46. **All three N64 phases died 11-03 10:51–11:29** with
  `InterruptedExperiment('zhekahauska')` at 22.5–23k steps, about 59 h into the phase.
- **Adaptive job** (`7c2ed873`, separate run spec, started **11-02 17:43**): N8 took 4.5–4.8 h,
  N16 10.5–11.0 h. N32 started 08:43–09:30. **All three ended 11-03 11:08**, the same window as
  the interface kill, with no error recorded.
- **Softmax chains** (still inside the curve job) started 11-03 11:29, right after the last
  interface phase died: N8 2.2–2.5 h, N16 1.8–2.8 h, then N32 started 15:38–16:49. **All three
  died 11-03 18:15** — two with `OSError(28, 'No space left on device')`, the third the same
  minute with nothing left to log with.
- Two jobs stopping within the same ~40 minutes points to a cluster-level event or a manual stop;
  Comet cannot tell which. The disk-full cause is certain for the softmax N32 deaths.

### The resume job (submitted 11-10, user-directed)

`lm-mpi-job-960c43c4-…`, manifest
[`kv_interference_softmax_resume.yaml`](../manifests/kv_interference_softmax_resume.yaml),
`interface_softmax` chains only — the wrapper skipped the `done` N8/N16 phases via the chain
state files staged from the dead job's run copy (disk remediation first: two dead-job `.venv`
leftovers, 5.3G + 98M, deleted; the dead `runs/` tree archived). Queued ~7 h in a congested
queue, then ran clean **11-10 23:16 → 11-12 08:07** (~33 h): N32 reruns (3–6.4 h each, threshold
stops), N64 (9.8–18.6 h), N96 (6.3–12.6 h). No crashes, no disk incidents. The full submission
evidence (state-file verification, local resume smoke with step-0 EM 0.918 matching the killed
run exactly, fan-out audit) is in STATUS.md 2026-11-12.

### Cost reality

Measured per-step wall time under 3-chain contention on one A100 (threshold-stop phases are much
shorter than the caps, so phase wall ≈ steps × rate):

| arm | N=8 | N=16 | N=32 | N=64 | N=96 |
|---|---|---|---|---|---|
| interface (s/step) | 0.48 | 1.05 | 2.7 | 9.3 | — (never ran) |
| interface_softmax (s/step) | ~0.5 | ~1 | 2–3.7 | ~8 | ~18–23 (one window implies ~60 under full 3-chain contention) |

The pre-registered "~3–5 days wall" for the whole 30-run curve was wrong mainly for the interface
arm (a full N64 phase alone is ~5.4 days; its N96 would be ~10+ days). The control is far
cheaper: the entire softmax resume (N32→N96, 3 chains) took ~33 h. Any follow-up that only needs
the control side is cheap; anything that runs the superposed read at N ≥ 64 to its cap is not.

## Caveats

- **Interface N64 is truncated** (45% of budget) — see the Δ(64) reading above. **Adaptive N32 is
  truncated** (flat from step 0; hides nothing upward). Everything else converged.
- **Interface N96 never ran** — Δ(96) is undefined. Given the N64 plateau and the control's flat
  0.99, the missing point is very unlikely to change any conclusion; running it would cost ~10+
  days.
- **The curriculum confound is intended** (pre-registered fairness note). Arms are compared under
  identical curricula; absolute N readings mix in the warm start.
- **Single-key retrieval only.** The ordered-concat ladder task was not run at larger N.
- **3 seeds.** At N=32, two of three per-seed Δ values exceed the 0.05 line (0.073, 0.080, 0.044).

## Next (options for the PI and user, not run)

1. **Direct test of the capacity hypothesis:** the planned `kv_interface_softmax.yaml` arm2 (key
   width 64 via `read_state_heads: 2`, config-only) on this curriculum at N=32 and N=64. If
   N* ≈ d_h, the binding point moves from 32 to ~64. This is now the sharpest open question the
   curve poses: is the collapse set by key width (architecture) or by item count vs training
   (optimization)?
2. **Mitigation the record already plans:** "If superposition binds: coarse top-k segment routing"
   (PROGRESS missing-experiment #1) — the condition is met, with Δ(64) = 0.58 as the headroom.
3. **Adaptive forgetting matrices:** pull the N8/N16 per-segment eval artifacts from the remote
   archive to close P-A1's forgetting-matrix part.
4. **Optional completeness:** interface N96 (~10+ days) and adaptive N64/N96 — low information
   value per the readings above; skip unless a reviewer wants the full grid.
5. **N=128 extension** (dataset exists) — for the control only it is cheap; for the superposed
   read it would only extend the plateau. Interesting only in combination with (1) or (2).
