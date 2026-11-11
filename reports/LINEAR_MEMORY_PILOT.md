# EXP-LM-001 — intermediate commits in linear associative memory

**Task ID:** EXP-LM-001  
**Status:** completed  
**Date:** 2026-11-10

## Objective

Implement the first executable contribution-roadmap milestone and answer two
controlled questions:

1. Do intermediate snapshots create additional writer-gradient routes when
   detached and differentiable arms have identical forward storage?
2. Do random-key linear-attention retrieval errors agree with explicit
   analytical predictions, including when snapshot payloads are correlated?

A positive optimization-speed result was not an acceptance condition.

## Methods

The local memory is `W in R^(v x p)` with update

`W <- W + eta (y - W x) x^T`.

The pilot used `J=12`, `p=4`, `v=3`, `h=6`, `K=1`, and a bounded shared
step size. A learned linear position codebook maps twelve fixed one-hot fact
identities to normalized local addresses. Values are fresh iid Gaussian unit
vectors in each episode. This is a shared position-codebook experiment, not a
content-key encoder experiment.

Whole-operator snapshots are committed either only at group endpoints
(`tau=6`, two commits) or every two segments including endpoints (`tau=2`, six
commits). Every fact uses oracle routing to its first subsequent commit within
the same reset group. The readers are exact snapshot gather and a folded
`D=16` random-unit-key LA state. Global keys are exogenous and independent of
the payloads.

For intermediate commits, the detached arm stops gradients only through
non-final snapshot storage branches. The live recurrence and every group-final
snapshot remain differentiable. Because the learned local address also serves
as read query `x`, query-side encoder gradients remain in the detached arm.
The intervention isolates intermediate memory-output routes, not every route
to the shared encoder.

Each arm used the same fixed training/evaluation episode streams for three
seeds (101, 202, 303), 120 Adam updates, batch size 32, and 128 held-out
episodes. The NMSE target 0.20, evaluation interval 20 steps, and all
hyperparameters were fixed before execution. Config SHA-256:
`190b0d0fb46c4d253ba1951af45c0e6ad2a994d25d7549ab9fb6a26a014a8fa9`.

## Analytical result

For a target snapshot `r`, fix arbitrary payload responses `z_c = W_c x` and
let all global keys be iid zero-mean isotropic unit vectors, independent of
the payloads and selected target. With `q=k_r`, contamination is

`n = sum_(c != r) (k_r^T k_c) z_c`.

Conditioned on `k_r` and all payloads, distinct coefficients have zero cross
moment and each has second moment `1/D`. Therefore

`E[||n||^2 | {z_c}] = (1/D) sum_(c != r) ||z_c||^2`.

Payload independence is unnecessary: the snapshots may be arbitrarily
correlated. Since `E[n | {z_c}]=0`, the expected local-error/noise cross term
also vanishes. Thus expected total squared error is

`delta + A/D`,

where `delta` is local snapshot error and `A` is non-target payload-response
energy. For an expected-error requirement `epsilon`, the minimal integer
address width under these assumptions is

`D_min = max(1, ceil(A / (epsilon - delta)))`,

when `A>0` and `delta<epsilon`. If `A=0`, `D=1` is feasible whenever
`delta<=epsilon`, including equality. If `A>0` and `delta>=epsilon`, or if
`delta>epsilon` for any `A`, no finite `D` suffices. This is an expected-error
statement, not a worst-case bound.

For the two-fact scalar case with unit local keys of inner product `rho`,
`K=1`, and `eta=1`, endpoint-only storage has

`delta_endpoint = (rho^2 + rho^4)/2`.

Per-fact snapshots have zero oracle local error and LA error

`E_snapshot = (1 + rho^4)/(2D)`.

Snapshots improve expected error precisely when

`D > (1 + rho^4)/(rho^2 + rho^4)` for `rho != 0`.

At `rho=0`, endpoint storage is already exact and finite-D LA snapshots make
the result worse. This gives an explicit success/failure crossover rather
than a universal hierarchy advantage.

## Results

All 15 analytical and implementation tests passed. They exercise the actual
writer for orthogonal no-forgetting and repeated-key overwrite; check the
delta-rule Jacobian product through `K=2` writes, the generic gradient-path
identity, folded/direct LA equivalence, detached/full matched forwards,
preserved endpoint gradients and live recurrence, reset-gradient isolation,
tail commits, error decomposition, and correlated-payload contamination
prediction. The focused command completed as `15 passed in 2.37s`.

Held-out results are mean ± sample standard deviation over three seeds:

| Arm | Final NMSE at step 120 | Steps to NMSE <= 0.20 | CPU train seconds to target |
|---|---:|---:|---:|
| Endpoint, exact | 0.382127 ± 0.009776 | not reached (3/3) | — |
| Intermediate detached, exact | 0.000195 ± 0.000146 | 40, 40, 40 | 0.311 ± 0.028 |
| Intermediate full, exact | 0.000532 ± 0.000014 | 40, 40, 40 | 0.334 ± 0.052 |
| Endpoint, LA | 0.415504 ± 0.007008 | not reached (3/3) | — |
| Intermediate detached, LA | 0.115257 ± 0.003145 | 60, 60, 60 | 0.499 ± 0.019 |
| Intermediate full, LA | 0.114972 ± 0.002925 | 60, 40, 40 | 0.386 ± 0.073 |

The detached and full intermediate arms had exactly identical predictions at
the fixed-parameter probes (maximum absolute difference 0 for every seed and
reader), but their parameter gradients differed. Averaged across seeds, the
full-minus-detached gradient difference norms were 0.139 (encoder) and 0.193
(step-size logit) for exact reading, and 0.121 and 0.164 for LA. This directly
verifies additional writer-gradient routes.

The exact-reader detached/full arms reached the target at the same 20-step
resolution. For LA, full differentiation reached the target at steps
`[60,40,40]`, versus `[60,60,60]` when detached. The mean LA curves separated
most at steps 60 and 80 (full: 0.151 and 0.123; detached: 0.183 and 0.150), then
converged to the same approximately 0.115 interference floor. This is a small,
reader-specific pilot signal, not evidence of a general convergence speedup.

At the final LA evaluations, measured versus analytically predicted
contamination NMSE was 0.0308 versus 0.0291 for endpoint storage. For the two
intermediate arms it was 0.1060 versus 0.1091 (detached) and 0.1049 versus
0.1079 (full). Observed cross terms were small (absolute means below 0.001).
The independent two-fact Monte Carlo grid matched theory with maximum relative
error 0.96% for nonzero endpoint errors and 3.33% for LA errors.

## Cost ledger

Counts below are explicit scalar-operation/MAC estimates per episode; they are
not measured FLOPs. Training wall times are single-process CPU observations
and exclude evaluation.

| Quantity | Endpoint (`C=2`) | Intermediate (`C=6`) |
|---|---:|---:|
| Ideal folded state `D*p*v` | 192 scalars | 192 scalars |
| Raw snapshot bank `C*p*v` | 24 | 72 |
| Raw global-key bank `C*D` | 32 | 96 |
| Active local operator `p*v` | 12 | 12 |
| Local writes | 468 estimated ops | 468 estimated ops |
| Folded-state build | 384 MAC | 1,152 MAC |
| Folded read, 12 queries | 2,448 MAC | 2,448 MAC |
| Raw-bank LA alternative, 12 queries | 744 estimated ops | 2,232 estimated ops |

The table is a component ledger, not full measured memory or full model
compute. In both columns the learned position codebook adds `p*J=48` trainable
scalars and the bounded step size adds one. The implementation evaluates the
dense codebook on all `J` one-hot identities, approximately `J^2*p=576` MAC,
plus normalization; an embedding lookup implementation could avoid that dense
cost. Each episode also supplies the full exogenous `J*D=192`-scalar global-key
table and carries `J` oracle routes plus commit metadata. Training retains raw
snapshots, both live/storage stacks, optimizer state, and autograd activations;
none is included in the ideal `D*p*v` inference-state row.

At this small commit count, the raw bank is smaller than the folded state and
its estimated direct read is cheaper. EXP-LM-001 therefore establishes no
small-scale computational gain from folding.

## Evidence

- Protocol and frozen configuration: [`experiments/linear_memory/protocol.md`](../experiments/linear_memory/protocol.md)
- Implementation: [`experiments/linear_memory/model.py`](../experiments/linear_memory/model.py) and [`run_pilot.py`](../experiments/linear_memory/run_pilot.py)
- Tests: [`tests/test_linear_memory.py`](../tests/test_linear_memory.py)
- Raw evaluation curves: [`eval_curves.csv`](results/linear_memory/EXP-LM-001/eval_curves.csv)
- Raw training curves: [`train_curves.csv`](results/linear_memory/EXP-LM-001/train_curves.csv)
- Fixed-parameter gradient probes: [`gradient_probes.csv`](results/linear_memory/EXP-LM-001/gradient_probes.csv)
- Two-fact Monte Carlo: [`two_fact_theory_mc.csv`](results/linear_memory/EXP-LM-001/two_fact_theory_mc.csv)
- Run summaries and operation estimates: [`summary.json`](results/linear_memory/EXP-LM-001/summary.json)
- Environment, command, git state, and hashes: [`provenance.json`](results/linear_memory/EXP-LM-001/provenance.json)

Reproduction commands:

```bash
.venv/bin/python -m pytest -q tests/test_linear_memory.py
.venv/bin/python -m experiments.linear_memory.run_pilot \
  --config experiments/linear_memory/pilot_config.json
```

## Limitations

- The dominant endpoint/intermediate difference is forward historical access
  and increased snapshot count. Only the matched intermediate detached/full
  comparison isolates gradient routing.
- The learned local encoder is a fixed position codebook. Results do not show
  learned content addressing or transfer to unseen fact identities.
- Global routing is oracle and exogenous. Learned global routing may change
  both the error law and optimization behavior.
- Three seeds, one `(p,D,h,tau,K)` setting, and 20-step evaluation resolution
  cannot establish a convergence-speed claim. CPU timings are too small to be
  treated as robust systems benchmarks.
- The contamination identity assumes fresh iid zero-mean isotropic global
  keys independent of payloads, local queries, and selected commits. Learned,
  reused, or correlated global addresses need covariance terms.
- No transformer, LoRA, deep-prefix, or one-level capacity-matched baseline
  was tested at this milestone.

## Conclusion

Intermediate commits do create additional differentiable writer routes while
preserving identical forward storage in the matched detached/full comparison.
In this pilot those routes gave no target-step improvement with exact reads,
and a modest but unconfirmed earlier-target signal with LA. Intermediate
snapshots primarily improved representational retention here; LA then imposed
a predictable interference floor.

The random-key error law remains valid for correlated snapshots because it
conditions on their payloads. The two-fact result supplies the first explicit
allocation crossover: snapshots are useful only when global address width is
large enough to offset their added interference, and they lose when local
keys are already orthogonal.

## Recommended follow-up

Proceed to a small pre-registered `(rho,D,tau)` or `(p,D,h,tau)` falsification
grid chosen from `delta + A/D`, retaining detached/full and exact/LA controls.
The next milestone should test whether the predicted `D_min` selects feasible
configurations out of sample. A transformer integration or larger sweep is
premature until the allocation rule survives that held-out linear test.
