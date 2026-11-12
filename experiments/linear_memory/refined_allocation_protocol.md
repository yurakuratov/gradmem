# EXP-LM-003 protocol: refined integer allocation

## Status and hypothesis

This protocol is frozen before calibration or held-out outcomes are generated.
The internal execution gate requires focused tests and PI design-audit release
before either scientific stage may run.

The question is whether shared local delta-rule writing can be cheaper than
independent one-fact writes after jointly refining local width `p` and global
address width `D`, under the unchanged EXP-LM-002 error law, memory cap, and
operation proxy. A positive hierarchical result is not required.

Primary acceptance requires at least 90% feasible-selection coverage among
scenarios with a feasible point oracle and no more than 10% median cost regret
among feasible selections. An empty oracle denominator cannot pass.

## Writer, distributions, and moments

Each reset group starts from `W=0` and applies one update per fact,

`W <- W + (y-Wx)x^T`,

with `eta=1`, `K=1`, unit local keys, unit two-dimensional values, reset
horizon `h`, and snapshot interval `tau`. A query routes to the first snapshot
after its fact in the same group.

Local keys are fresh per episode and group. The iid regime uses normalized
Gaussian keys. The correlated regime normalizes
`sqrt(0.7)u_group + sqrt(0.3)e_fact`; 0.7 is a mixture coefficient rather than
the realized cosine. Groups and values are independent.

Isolated-group calibration estimates

- `delta = E mean_j ||W_r(j)x_j-y_j||^2`,
- `A0 = E mean_j sum_(c != r(j)) ||W_c x_j||^2`, and
- `B = (1/p) E sum_c ||W_c||_F^2`.

For `G=J/h` groups the predictor is

`E_hat(D) = delta + [A0+(G-1)B]/D`.

For every `h=1` candidate, all policies use the exact identity
`delta=A0=0`, `B=1/p`; empirical calibration is retained only as a diagnostic.
For `h>1`, policies use calibration estimates.

## Frozen domain, cost, and policies

The domain is `p in {2,4,8,16,32,64}`, `h in {1,2,4,8}`, unique valid `tau`
from `{1,2,h}`, and every positive integer `D` satisfying
`pv(1+D)<=2048`. Thus the independent baseline optimizes both `p` and `D`
over the same domain as the unrestricted policy.

The deployment proxy remains

`cost = J(3pv+v) + C D p v + Q(Dpv+pv)`,

where `C` is total snapshots. It mixes scalar operations and MACs and is not a
wall-clock claim. Folded inference memory is `pv(1+D)` scalars. Raw snapshot
and address-bank sizes are reported separately; no commit overhead is added.

All primary selectors require `E_hat <= 0.95 epsilon`. The fixed 5% relative
slack is a heuristic reserve, not a statistical guarantee. Actual feasibility
and the primary oracles use `epsilon`, without slack. A registered diagnostic
also reports coverage and regret against a held-out point oracle at
`0.95 epsilon`; this separates model-transfer error from the cost of the slack
but does not replace primary acceptance. Policies are:

- `unrestricted`: all registered `(p,h,tau,D)`;
- `independent`: require `h=tau=1`, optimizing `p,D`;
- `shared`: require `h>1`, optimizing `p,h,tau,D`.

Ties resolve by cost, folded memory, then `(p,h,tau,D)`. The implementation
selects the exact least-cost eligible integer `D`; tests compare it with brute
force. Boundary comparisons tolerate only floating representation error.
Zero-noise `delta=epsilon` is feasible, while `A>0,delta>=epsilon` has no
finite solution.

## Sampling and leakage barrier

New seeds are 4404, 5505, and 6606. Calibration uses 4,096 isolated groups
per seed, geometry, and local configuration with seed offset 410000. Held-out
evaluation uses 1,024 complete episodes per seed, geometry, local
configuration, and `J in {16,32}`, with separate local and global-key offsets
420000 and 430000. Query counts are 1 and 32; targets are 0.10 and 0.25.

Episodes are reused across `D`, query count, error target, and policy. These
scenario rows are dependent reporting views, not independent replications.

Execution is split:

1. freeze and hash the config, this protocol, new refined source/runner, and
   imported `allocation.py` and `model.py`;
2. calibrate and write all moment estimates, integer-frontier predictions, and
   policy selections;
3. seal the artifact and its hash before held-out generation;
4. verify every frozen dependency, then generate held-out outcomes.

The runner refuses to overwrite an existing experiment directory and aborts
evaluation on any dependency, calibration, prediction, or selection drift.

## Held-out reference, oracles, and comparisons

For each local trace, held-out episode vectors `delta_i,A_i` are computed once.
For any integer `D`, conditional-reference variance is obtained from

`Var(delta+A/D) = Var(delta) + 2 Cov(delta,A)/D + Var(A)/D^2`.

Tests compare this shortcut with direct episode vectors. Point means are
monotone because `A>=0`; point oracle search uses exact integer eligibility.
The normal-95% upper endpoint need not be monotone, so certified-oracle search
enumerates every eligible integer `D`. These are pointwise intervals, not
simultaneous guarantees.

Every policy reports selection, abstention, false feasibility, point and
interval feasibility, point-oracle regret, certified-oracle regret, and
oracle-scenario coverage. Regret is undefined when the selected candidate is
infeasible. Cheaper interval-uncertain alternatives are reported.
Slack-matched point-oracle existence, coverage, and regret are reported as
diagnostics with the same abstention-safe denominator.

Shared versus independent comparisons include calibrated selections and
held-out policy-restricted point oracles. A win is labelled only when both
compared configurations are feasible. Cost ratios, selected `(p,h,tau,D)`,
and distance to the target are retained.

Full-grid transfer is summarized with equal weight per local configuration:
for each `(p,h,tau)`, compute mean and maximum absolute prediction error over
all its eligible integer `D`, then aggregate those local-configuration
summaries by `h`. This prevents configurations with more eligible widths from
creating artificial replication.

## Sampled-LA consistency and fixed checks

Explicit folded LA is evaluated only for deduplicated selected candidates,
the three fixed iid witnesses, and sentinel `(p=4,h=4,tau=1,D=32,J=16)` in both
geometries and all seeds. It is a finite-sample consistency check against a
conditional reference using the same identity, not independent proof.

Registered analytical checks are:

- iid `h=tau=2`:
  `delta=(1/p+3/[p(p+2)])/2`, `A0=0`, `B=(2-1/p)/p`;
- shared witness `J=16,p=16,h=tau=2,D=16`:
  expected error 0.0894368489583 and Q=1 cost 6,208;
- independent boundary witness `J=16,p=2,h=tau=1,D=75`:
  expected error 0.10 and Q=1 cost 5,328.
- low-query shared witness `J=32,p=32,h=tau=2,D=12`:
  expected error 0.0939079733456, memory 832, and cost `18,496+832Q`.
  Any independent `h=1` candidate satisfying true error 0.10 has
  `pD>=310`; even relaxing integrality and the memory bound gives cost at least
  `20,288+624Q`. Therefore this shared witness is cheaper for integer `Q<=8`
  (7.57% at `Q=1`) under the registered proxy. This analytic comparison was
  fixed before EXP-LM-003 data and is limited to LA-backed memories under this
  proxy.

All witnesses are evaluated with new iid held-out episodes. The boundary
witness is evaluated even though selector slack excludes it. Its finite-sample
position relative to the target is not treated as proof or disproof of the
population equality.

## Commands after PI release

Run from `test_time_gd`:

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
