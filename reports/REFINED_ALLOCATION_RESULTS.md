# EXP-LM-003 — refined local/global allocation

**Task ID:** EXP-LM-003  
**Status:** completed  
**Date:** 2026-11-11

## Objective

Determine whether shared local delta-rule writing can reduce the registered
deployment-operation proxy relative to independently written memories after
jointly refining local width `p` and global address width `D`. The comparison
uses the same expected-error law, folded-memory cap, writer, and cost model as
EXP-LM-002. It makes no wall-clock, learned-routing, transformer, LoRA, or
optimized-local-only claim.

The primary acceptance criterion was at least 90% feasible-selection coverage
among scenarios with a feasible point oracle and at most 10% median feasible
cost regret. An empty oracle denominator could not pass.

## Methods

The local state `W in R^(v x p)` used one update per fact,

`W <- W + (y-Wx)x^T`,

with `eta=1`, `K=1`, `v=2`, reset horizon `h`, and commit interval `tau`.
Every query used oracle routing to the first same-group snapshot after its
fact. Local widths were `p in {2,4,8,16,32,64}`; reset horizons were
`h in {1,2,4,8}`; valid commit intervals were unique values from `{1,2,h}`.
Every positive integer `D` satisfying `pv(1+D)<=2048` was eligible.

The policies were unrestricted, independent (`h=tau=1`, optimizing both
`p,D`), and shared (`h>1`). All selectors required predicted error at most
`0.95 epsilon`. This 5% reserve was fixed before data and was a heuristic, not
a confidence guarantee. Actual feasibility and the primary oracle used
`epsilon`. Ties were resolved by operation cost, memory, then
`(p,h,tau,D)`.

Calibration used 4,096 isolated groups for each local configuration, seed, and
geometry. For `h>1` it estimated

`E_hat(D) = delta + [A0+(J/h-1)B]/D`.

For every `h=1` candidate the selector used the exact identity
`delta=A0=0, B=1/p`; empirical `h=1` measurements were diagnostics only.

Held-out evaluation used 1,024 complete episodes for each local
configuration, three new seeds (4404, 5505, 6606), context lengths 16 and 32,
query counts 1 and 32, and targets 0.10 and 0.25. The iid geometry used fresh
normalized Gaussian addresses. The correlated geometry used normalized
`sqrt(0.7)u_group+sqrt(0.3)e_fact` addresses.

The cost remained

`J(3pv+v) + C Dpv + Q(Dpv+pv)`.

This is a mixed primitive-operation/MAC proxy rather than measured FLOPs.
Address generation, oracle routing, and metadata are excluded. Folded memory
is `pv(1+D)` scalars. Raw snapshot and global-key banks are reported
separately.

The two-stage run sealed all dependencies, 324 calibration rows, 2,592
integer-frontier rows, and 144 selections before held-out generation. The
selection artifact SHA-256 is
`d97868b12e8ed1a6082f905c76bc373d5933e092b124c6ae143671a3aefec500`.

## Primary allocation result

The registered primary criterion passed.

| Policy | Point-feasible coverage | Median point-oracle regret | Maximum regret | Selected interval-feasible |
|---|---:|---:|---:|---:|
| Unrestricted | 48/48 | 4.81% | 10.38% | 48/48 |
| Independent | 48/48 | 4.81% | 6.28% | 48/48 |
| Shared | 48/48 | 5.27% | 10.38% | 48/48 |

The 48 rows per policy are dependent reporting views. They reuse 12 held-out
episode sets—three seeds, two geometries, and two context lengths—across query
counts and error targets. They are not 48 independent replications.

The unrestricted calibrated selector chose independent writing in 39 rows and
shared writing in nine. Those nine were exactly three iid, `Q=1` scenario
families repeated over three seeds:

| `J` | `epsilon` | Independent selection | Shared selection | Independent cost | Shared cost | Shared saving |
|---:|---:|---|---|---:|---:|---:|
| 16 | 0.25 | `(p=2,h=1,D=32)` | `(p=8,h=2,tau=2,D=11)` | 2,404 | 2,400 | 0.17% |
| 32 | 0.10 | `(p=2,h=1,D=164)` | `(p=32,h=4,tau=4,D=20 or 21)` | 22,100 | 17,792 or 18,368 | 16.9–19.5% |
| 32 | 0.25 | `(p=2,h=1,D=66)` | `(p=16,h=4,tau=4,D=13)` | 9,164 | 6,912 | 24.6% |

All compared configurations were point-feasible and interval-feasible. The
held-out policy-restricted point oracles also favored sharing in exactly these
nine rows. Oracle shared/independent cost ratios ranged from 0.755 to 0.995.

The `J=16` difference is four proxy operations and is negligible and
boundary-sensitive. The meaningful result is the `J=32,Q=1` iid regime:
shared local writing reduces the registered compute proxy by roughly 17–25%
at the same target and under the same memory cap.

This is not memory Pareto dominance. At `J=32,epsilon=0.10`, independent
folded-plus-active memory is 660 scalars, while the shared selections use
1,344 or 1,408. At `epsilon=0.25`, the comparison is 268 versus 448 scalars.
All remain below the 2,048-scalar cap. Sharing trades additional folded state
for fewer global commits and lower proxy compute.

For `epsilon=0.10`, raw independent snapshot/key banks contain 128/5,248
scalars, versus 512 and 160–168 for the shared choices. For `epsilon=0.25`,
they contain 128/2,112 versus 256/104. These banks are diagnostic alternative
representations and are not included in folded inference memory.

No shared win occurred for `Q=32` or for the correlated geometry. More reads
increase the cost of the wider shared payload, erasing its write-side gain.

## Registered analytical checks

For iid `h=tau=2`, the registered formulas were

`delta=(1/p+3/[p(p+2)])/2`, `A0=0`, `B=(2-1/p)/p`.

| Witness | Theory error | Held-out reference mean across seeds | Average per-seed SE | Sampled-LA mean |
|---|---:|---:|---:|---:|
| `J16,p16,h2,D16` | 0.089437 | 0.089624 | 0.000738 | 0.089847 |
| Independent boundary `J16,p2,D75` | 0.100000 | 0.099965 | 0.000202 | 0.099064 |
| `J32,p32,h2,D12` | 0.093908 | 0.093843 | 0.000269 | 0.093824 |

The `J32` shared witness costs `18,496+832Q`. Any independent `h=1` memory
with `p>=2` meeting population error 0.10 has `pD>=310`; even relaxing integer
and memory constraints gives cost at least `20,288+624Q`. The witness therefore has a
registered analytical advantage for integer `Q<=8`. Its new held-out result
agrees with the prediction: all three seed-level reference intervals contained
the theoretical error and lay below 0.10. This is separate from the selected `h=4` result,
which supplied the lowest calibrated cost at `Q=1`.

The independent boundary witness has population error exactly 0.10 and was
excluded by selector slack. Its finite-sample mean being slightly below 0.10
is not evidence against the population equality.

## Prediction quality, slack, and uncertainty

The unrestricted primary selection was point-feasible and interval-feasible
in all 48 rows. Explicit sampled folded-LA point means were also feasible in
48/48 unrestricted, 48/48 independent, and 48/48 shared rows. Of 72 unique
sampled candidates and fixed checks, the median absolute sampled-minus-
conditional-reference difference was 0.000797 and the maximum was 0.00506.
This is a finite-sample implementation/distribution consistency check using
the same global-key identity, not independent proof of that identity.

The slack-matched diagnostic was weaker: only 40/48 unrestricted and
independent selections had held-out conditional mean below `0.95 epsilon`;
shared achieved 47/48. All remained below the actual target. Thus the reserve
protected primary feasibility here but did not guarantee its nominal
slack-level target. Median slack-oracle regret was zero where the selection was
slack-feasible.

An interval-certified oracle existed in all rows. Unrestricted certified-
oracle median and maximum regrets were 4.35% and 6.92%. Nevertheless, 41/48
unrestricted rows contained a candidate cheaper than the certified oracle
whose interval crossed the target. The certified optimum is therefore
conservative and boundary-sensitive; the data do not establish robust
optimality among close candidates.

Full-grid transfer was summarized with equal weight per local configuration,
not per integer `D`:

| `h` | Median local mean absolute error over eligible `D` | Maximum local maximum error |
|---:|---:|---:|
| 1 | 0.000132 | 0.02696 |
| 2 | 0.000637 | 0.11780 |
| 4 | 0.001219 | 0.18856 |
| 8 | 0.002015 | 0.32999 |

Typical transfer error is small, but worst cases grow substantially with local
horizon. The exact `h=1` population identity still has finite held-out
conditional-moment sampling error.

## Evidence

- Frozen design:
  [`refined_allocation_protocol.md`](../experiments/linear_memory/refined_allocation_protocol.md),
  [`refined_allocation_config.json`](../experiments/linear_memory/refined_allocation_config.json)
- Implementation:
  [`refined_allocation.py`](../experiments/linear_memory/refined_allocation.py),
  [`run_refined_allocation.py`](../experiments/linear_memory/run_refined_allocation.py)
- Tests:
  [`test_linear_memory_refined_allocation.py`](../tests/test_linear_memory_refined_allocation.py)
- Sealed calibration and selections:
  [`calibration_moments.csv`](results/linear_memory/EXP-LM-003/calibration_moments.csv),
  [`prediction_frontiers.csv`](results/linear_memory/EXP-LM-003/prediction_frontiers.csv),
  [`selections_preregistered.json`](results/linear_memory/EXP-LM-003/selections_preregistered.json)
- Held-out evidence:
  [`heldout_moments.csv`](results/linear_memory/EXP-LM-003/heldout_moments.csv),
  [`policy_evaluation.csv`](results/linear_memory/EXP-LM-003/policy_evaluation.csv),
  [`sharing_comparison.csv`](results/linear_memory/EXP-LM-003/sharing_comparison.csv),
  [`fixed_checks.csv`](results/linear_memory/EXP-LM-003/fixed_checks.csv),
  [`sampled_la_consistency.csv`](results/linear_memory/EXP-LM-003/sampled_la_consistency.csv)
- Summary and provenance:
  [`summary.json`](results/linear_memory/EXP-LM-003/summary.json),
  [`provenance.json`](results/linear_memory/EXP-LM-003/provenance.json)

Commands run from `test_time_gd`:

```bash
.venv/bin/python -m pytest -q \
  tests/test_linear_memory_refined_allocation.py \
  tests/test_linear_memory_allocation.py tests/test_linear_memory.py
.venv/bin/python -m experiments.linear_memory.run_refined_allocation \
  --config experiments/linear_memory/refined_allocation_config.json \
  --stage calibrate-select
.venv/bin/python -m experiments.linear_memory.run_refined_allocation \
  --config experiments/linear_memory/refined_allocation_config.json \
  --stage evaluate
```

Forty focused and regression tests passed. The instrumented calibration stage
took 5.43 seconds (about seven seconds of CLI wall time); held-out evaluation
took 82.8 seconds, below the ten-minute limit.

## Limitations

- The result concerns iid LA-backed memories under an analytical proxy. It is
  not a wall-clock benchmark and does not compare optimized local-only memory,
  deep writable tokens, LoRA, learned routing, or transformers.
- Global routing is oracle-supplied. Address generation and metadata costs are
  omitted.
- The shared winners use more folded memory than the independent baseline and
  are advantageous only under a cap, not at equal actual memory.
- Scenario rows reuse episode sets across query count, target, and policy.
  Three seeds do not support broad frequency claims.
- Confidence intervals are pointwise normal intervals. Forty-one unrestricted
  rows have cheaper uncertain alternatives, so certified-optimum claims are
  fragile near the boundary.
- The correlated generator draws one latent vector per reset group. Changing
  `h` changes correlation-block boundaries and therefore the address
  distribution. Correlated results are comparisons within a reset-conditioned
  distribution family, not causal interventions on one fixed correlated fact
  stream. A fixed exogenous correlation-block study would require a transfer
  model that accounts for dependence across resets.
- Moment predictions are least reliable in some long-horizon, low-`D` regions,
  as the full-grid maxima show.

## Conclusion

Shared local writing can reduce the registered compute proxy relative to an
independent LA-backed baseline after both `p` and integer `D` are optimized.
The strongest controlled evidence is iid, `J=32,Q=1`: calibrated shared
configurations were feasible and used roughly 17–25% fewer proxy operations,
while consuming more folded memory but remaining under the common cap. The
registered iid `h=2` witness independently matched its analytical expected
error and lower-bound comparison.

The advantage is conditional. It disappears at `Q=32`, was absent in the
reset-conditioned correlated family, is not memory Pareto dominance, and does
not yet imply a transformer or wall-clock benefit.

## Recommended follow-up

First add a fair enlarged local-only baseline to answer the original one-level
versus hierarchical question under matched state, parameter, and retrieval-
error budgets. This baseline must optimize its local width rather than inherit
the earlier small-memory restriction. Then calibrate the analytical operation
proxy against measured write/read time and memory. The current low-query iid
shared/independent pair remains a useful secondary systems check once that
one-level comparison is in place. A later learned-address experiment should
fix exogenous correlation blocks independently of reset horizon.
