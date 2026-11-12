# EXP-LM-002 — held-out allocation from short-group calibration

**Task ID:** EXP-LM-002  
**Status:** completed  
**Date:** 2026-11-11

## Objective

Test whether short, single-group measurements can select local/global memory
allocations for unseen longer contexts without fitting on those contexts. The
registered model predicts

`E_hat = delta + [A0 + (G-1) B] / D`,

where `delta` is routed local-read error, `A0` is same-group non-target
snapshot energy, `B` is the expected energy contributed by one independent
group, `G=J/h`, and `D` is global address width.

The predeclared acceptance test required at least 90% point-feasible
unrestricted selections and at most 10% median operation regret among feasible
selections. A miss would still count as a completed negative experiment.

## Methods

The fixed local writer was `W <- W + (y-Wx)x^T`, with `eta=1`, one update per
fact, unit-norm two-dimensional values, and a reset every `h` facts. Local
addresses were fresh in every episode and reset group. We tested iid unit
Gaussian addresses and normalized correlated mixtures
`sqrt(0.7) u_group + sqrt(0.3) e_fact`. The number 0.7 is the mixture weight,
not the realized address cosine.

The grid contained `p in {2,4,8}`, `h in {1,2,4,8}`, unique valid commit
intervals from `{1,2,h}`, and `D in {4,8,16,32,64,128}`. In particular,
`h=tau=1` represented independent local writes. Each seed/geometry/local
configuration was calibrated on 512 isolated groups. Allocation was then
frozen for held-out `J in {16,32}`, `Q in {1,32}`, and expected-error targets
`epsilon in {0.10,0.25}`.

The operation objective was

`J(3pv+v) + C D p v + Q(Dpv+pv)`,

a registered mixed primitive-operation proxy rather than measured FLOPs.
Folded inference memory `pv+Dpv` was capped at 2048 scalars. Address
generation, oracle routing, and route metadata were excluded. The study used
three seeds (1101, 2202, 3303), 256 complete held-out episodes per candidate,
and episode-level standard errors.

The run enforced a leakage barrier. Calibration, all 7,776 candidate
predictions, and all 144 policy selections were written before held-out
episodes were generated. The sealed selection artifact has SHA-256
`98bbe4e64d6eff40545028655569096cd2793e5c0b70242ac34ca54ecb40be1d`.

The held-out reference analytically integrates global random keys conditional
on each held-out local episode. It is a low-variance application of the same
proved identity. Every one of the 1,944 candidate configurations was also
evaluated with an explicit sampled folded-LA state as a finite-sample
implementation and distribution consistency check, not independent
validation.

## Results

The registered acceptance test passed.

| Policy | Feasible selections | Median feasible regret | Maximum regret | Median prediction error |
|---|---:|---:|---:|---:|
| Unrestricted theory | 48/48 | 0.0000 | 0.0000 | 0.000169 |
| Independent writes, `p=4` | 48/48 | 0.0261 | 0.0815 | 0.000332 |
| Endpoint-only, `p=4` | 48/48 | 0.0261 | 0.0815 | 0.000332 |

All unrestricted selections were feasible by the entire 95% reference
interval, and all explicit sampled-LA point estimates met their targets. There
were no false-feasible, false-infeasible, or no-selection cases. Maximum
absolute prediction error among selected unrestricted configurations was
0.00165. The correlated regime was slightly noisier than iid: selected median
absolute error was 0.000236 versus 0.000161, with respective maxima 0.00165
and 0.000912.

The 48 rows per policy are reporting scenarios, not independent experimental
replications. They reuse 12 held-out episode sets (three seeds, two geometries,
and two context lengths) across query counts and error targets. Query count
changes cost but not retrieval error.

The selected configuration depended on context length and error target but
not on geometry, seed, or query count:

| `J` | `epsilon` | Selected `(p,h,tau,D)` | Mean predicted | Mean reference | Mean sampled LA | Memory scalars |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.10 | `(2,1,1,128)` | 0.058594 | 0.058586 | 0.058482 | 516 |
| 16 | 0.25 | `(2,1,1,32)` | 0.234375 | 0.234345 | 0.236985 | 132 |
| 32 | 0.10 | `(4,1,1,128)` | 0.060547 | 0.060463 | 0.060128 | 1,032 |
| 32 | 0.25 | `(2,1,1,64)` | 0.242188 | 0.242191 | 0.243351 | 260 |

For `h=tau=1`, local writes are exact, `A0=0`, and `B=1/p`, so the predictor
reduces to `(J-1)/(pD)`. The observed means closely match this exact special
case. The unrestricted choices also matched the cheapest point-feasible grid
configuration in every scenario, producing zero measured operation regret.

The endpoint-only `p=4` heuristic selected `h=1` in all scenarios and
therefore collapsed to the independent-write heuristic. This is a substantive
result of the registered comparison: no `h>1` shared local trajectory was
cost-optimal under the current writer, grid, and deployment proxy. The
experiment supports calibrated allocation transfer while providing no
evidence that shared local writing is advantageous in this regime.

Across the full grid, not only selected points, median absolute prediction
error was 0.00512 and the maximum was 0.1095. Errors were larger for longer
shared groups, with median values 0.00058, 0.00392, 0.00891, and 0.00655 for
`h=1,2,4,8`. Thus the strong selected-point result should not be read as
uniformly precise prediction of every candidate from 512 calibration groups.

The sampled folded-LA 95% interval contained the conditional reference mean
for 1,904/1,944 candidates (97.9%). For unrestricted selected points it did so
for 48/48 cases. Across the whole grid, median absolute sampled-minus-reference
difference was 0.00474 and the maximum was 0.215; the latter occurred among
high-noise non-selected candidates. This consistency check supports the LA
implementation and sampling distribution but does not create an independent
test of the local transfer law.

## Post-result audit and interval sensitivity

An audit after the original outcomes were visible found two instrumentation
and reporting gaps: `model.py`, which supplies the commit schedule, was not in
the frozen dependency hashes; and feasibility among selected configurations
could reward selective abstention. The original files and sealed selection
artifact were left unchanged. The originally executed source bundle was
archived with SHA-256
`733614f757d533488e0dcd30ed4860a437f187e9d2e84ddca55e2bba4fc00104`.

The audit rerun used the identical scientific grid, RNG streams, sample counts,
and selector in a separate output directory. Its selection artifact hash is
`9cd0defd968af5996d5400961008985847f2be6e704a32cf8c8d7a403589c6a6`.
Calibration, candidate predictions, and held-out candidate metrics are
byte-identical to the original; all selections and all 40 common selection-
evaluation fields are equal. The audit input hashes now include `model.py`.

The post-result audit criterion defines coverage as oracle-existing scenarios
with a feasible selection divided by all oracle-existing scenarios. It uses a
90% threshold, cannot pass with an empty denominator, and must be
preregistered in future experiments. Audit coverage was 48/48 (100%), so the
original result does not depend on abstention.

For sensitivity, a candidate was called certified feasible only when its full
episode-level 95% reference interval lay below the target. A certified oracle
existed in all 48 unrestricted reporting scenarios; all 48 selected points
were certified feasible; median and maximum regret versus the cheapest
certified oracle were zero. No scenario contained a cheaper interval-uncertain
candidate than the certified oracle. The original point-feasible oracle
remains the primary registered comparator; this interval analysis is explicitly
post-result.

## Evidence

- Frozen protocol and configuration:
  [`allocation_protocol.md`](../experiments/linear_memory/allocation_protocol.md),
  [`allocation_config.json`](../experiments/linear_memory/allocation_config.json)
- Implementation:
  [`allocation.py`](../experiments/linear_memory/allocation.py),
  [`run_allocation.py`](../experiments/linear_memory/run_allocation.py)
- Tests: [`test_linear_memory_allocation.py`](../tests/test_linear_memory_allocation.py)
- Sealed selections:
  [`selections_preregistered.json`](results/linear_memory/EXP-LM-002/selections_preregistered.json)
- Calibration and candidate predictions:
  [`calibration.csv`](results/linear_memory/EXP-LM-002/calibration.csv),
  [`candidate_predictions.csv`](results/linear_memory/EXP-LM-002/candidate_predictions.csv)
- Held-out candidate and selection results:
  [`candidate_metrics.csv`](results/linear_memory/EXP-LM-002/candidate_metrics.csv),
  [`selection_evaluation.csv`](results/linear_memory/EXP-LM-002/selection_evaluation.csv)
- Summary and provenance:
  [`summary.json`](results/linear_memory/EXP-LM-002/summary.json),
  [`provenance.json`](results/linear_memory/EXP-LM-002/provenance.json)
- Audit protocol and reproduction comparison:
  [`allocation_audit_protocol.md`](../experiments/linear_memory/allocation_audit_protocol.md),
  [`reproduction_comparison.json`](results/linear_memory/EXP-LM-002-AUDIT/reproduction_comparison.json)
- Audit summary and certified-oracle rows:
  [`summary.json`](results/linear_memory/EXP-LM-002-AUDIT/summary.json),
  [`selection_evaluation.csv`](results/linear_memory/EXP-LM-002-AUDIT/selection_evaluation.csv)
- Original executed-source archive:
  [`executed_sources_before_audit.tar.gz`](results/linear_memory/EXP-LM-002/executed_sources_before_audit.tar.gz)

Verification and reproduction commands, run from `test_time_gd`:

```bash
.venv/bin/python -m pytest -q \
  tests/test_linear_memory_allocation.py tests/test_linear_memory.py
.venv/bin/python -m experiments.linear_memory.run_allocation \
  --config experiments/linear_memory/allocation_audit_config.json \
  --stage calibrate-select
.venv/bin/python -m experiments.linear_memory.run_allocation \
  --config experiments/linear_memory/allocation_audit_config.json \
  --stage evaluate
```

The final focused test run passed all 30 tests. The evaluation produced 1,944
candidate rows and 144 policy-evaluation rows.

## Limitations

- The held-out lengths are new, but the two geometry families and all local
  hyperparameter values were known during calibration.
- Global routing is oracle-supplied. Address generation and routing cost are
  outside the registered operation proxy.
- The conditional reference and predictor share the same global-key identity;
  explicit sampled LA is a consistency check, while transfer of local moments
  is tested through independent held-out local episodes.
- The cost is an analytical mixed-operation proxy. It is not wall-clock
  performance, hardware efficiency, or a complete training-memory ledger.
- Only three seeds were used. Facts within an episode were correctly treated
  as correlated, but normal confidence intervals remain approximate. Rows
  sharing an episode set across query count, target, or policy are dependent.
- The fixed `eta=1,K=1` writer strongly favors single-fact groups. Results do
  not establish that independent writes dominate learned shared memories,
  GradMem tokens, LoRA states, or other local update rules.
- This milestone contains no detached/full gradient intervention and provides
  no new evidence about optimization speed.

## Conclusion

In the registered grid, the allocation rule selected feasible, minimum-cost
configurations for both held-out context lengths. Every selected point was the
exact `h=tau=1` special case, where `delta=A0=0`, `B=1/p`, and the error is
known as `(J-1)/(pD)`. Its success therefore required essentially no empirical
calibration learning. It verifies the selection and cost machinery but is weak
evidence for transfer of estimated moments.

The relevant transfer evidence comes from the full grid, especially `h>1`:
median prediction error was 0.00512, but the maximum was 0.1095 and error grew
for several longer shared groups. This is promising but imperfect calibration
transfer. The scientific conclusion is consequently narrow: no `h>1` writer
was optimal among the registered `p<=8` and power-of-two `D` candidates under
this analytical cost proxy. It does not establish that independent writes are
optimal for the fixed writer outside that grid, much less for learned memory.

## Recommended follow-up

Broaden both local width and global-address resolution before modifying the
cost model. A post-result calculation illustrates why. For iid addresses with
`eta=1` and `h=tau=2`,

`delta = 0.5[1/p + 3/(p(p+2))]`, `A0=0`, and `B=(2-1/p)/p`.

The out-of-grid point `p=16,D=16,J=16` predicts error 0.08944, memory 544,
and operation costs 6,208 (`Q=1`) or 23,072 (`Q=32`). It beats the originally
selected in-grid `p=2,h=1,D=128` costs 8,932 and 24,928, but it is not a global
optimum: the also-out-of-grid independent point `p=2,D=75` reaches error 0.10
with cost 5,328 at `Q=1`. This post-result calculation must stay outside the
original oracle. It shows that both the `p<=8` truncation and coarse
power-of-two `D` grid can determine the apparent crossover.

A future preregistered study should refine both axes, use measured cost or a
justified compression model, and test whether the predicted sharing crossover
survives finer allocation choices. Transformer integration should wait for
that nontrivial linear-memory result.
