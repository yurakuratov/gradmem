# EXP-LM-002 protocol: held-out allocation from short-group calibration

## Objective and frozen hypothesis

Calibrate local error and snapshot-response energy on isolated reset groups,
then use those statistics to choose `(p,h,tau,D)` before observing longer
contexts. For `G=J/h` independent groups, the registered predictor is

`E_hat(G,D) = delta + [A0 + (G-1) B] / D`.

The primary test asks whether selections made from this predictor meet unseen
`J in {16,32}` expected-error targets at low operation regret. The registered
acceptance thresholds are at least 90% point-feasible unrestricted selections
and at most 10% median cost regret among feasible unrestricted selections.
Failure is a valid negative result and does not trigger threshold or grid
tuning.

This is a frozen-writer distribution study. It contains no learned writer,
gradient-routing intervention, or optimization-speed claim. EXP-LM-001
already established matched forward values for detached and differentiable
snapshot branches.

## Local writer and calibration quantities

Every group starts with `W=0`, where `W in R^(v x p)`, and contains `h`
facts. One delta-rule update is applied per fact:

`W <- W + (y - W x) x^T`.

Here `eta=1`, `K=1`, `v=2`, and every `x` and `y` has unit Euclidean norm.
Within a group, snapshots are committed every `tau` facts and at the endpoint.
Each query routes to the first commit after its fact. Let `r(j)` denote that
commit and `W_c` the operator at commit `c`. Calibration uses 512 independent
single-group episodes and estimates

- `delta = E mean_j ||W_r(j) x_j - y_j||^2`,
- `A0 = E mean_j sum_(c != r(j)) ||W_c x_j||^2`, and
- `B = (1/p) E sum_c ||W_c||_F^2`.

For a query address from another independent marginally isotropic group,
`E_x ||W_c x||^2 = ||W_c||_F^2/p`. Therefore every additional group adds
expected distractor energy `B`. Iid zero-mean isotropic unit global keys give
the additional factor `1/D`.

Local addresses are fresh for every episode and group. In the `iid` regime,
they are normalized iid Gaussian vectors. In the `correlated` regime,

`x_t = normalize(sqrt(a) u_group + sqrt(1-a) e_t)`, with `a=0.7`,

where `u_group` and `e_t` are independent isotropic Gaussians. The number
`a` is a mixing coefficient, not the actual address inner product. Groups are
independent and both regimes are marginally isotropic. Values are independent
of all addresses.

## Frozen grid, policies, and cost

The grid is `p in {2,4,8}`, `h in {1,2,4,8}`, unique valid `tau` values from
`{1,2,h}`, and `D in {4,8,16,32,64,128}`. The `h=tau=1` cases are the
independent-write baseline. Held-out context lengths are divisible by all
registered horizons.

For each seed, geometry, `J`, query count `Q`, and error target `epsilon`, the
unrestricted policy selects the least-cost predicted-feasible grid point under
the memory cap. Two fixed-policy comparators use exactly the same calibration:

- `independent_p4`: fix `p=4,h=tau=1` and choose `D`;
- `endpoint_p4`: fix `p=4,tau=h` and choose `h,D`.

Ties are resolved deterministically by operation proxy, memory, then
`(p,h,tau,D)`. No policy can inspect held-out outcomes.

For `C=(J/h)*(h/tau)` commits, the registered deployment proxy is

`cost = J(3pv+v) + C D p v + Q(Dpv+pv)`.

It mixes scalar local-update operations and multiply-accumulate counts, so it
is a ranking proxy rather than measured FLOPs. Folded inference memory is
`M=pv+Dpv` scalars and must not exceed 2048. Address generation, oracle route
discovery, and route metadata are excluded. Raw snapshot/key bank metadata are
reported separately. The comparator is limited to this reader and grid; it is
not a universal one-level-memory comparison.

## Leakage barrier and evaluation

Execution has two explicit stages:

1. hash and snapshot this configuration and protocol; calibrate on `J=h`;
2. write predictions and selections for every registered scenario;
3. only then generate disjoint held-out episodes and evaluate the full grid.

The pre-evaluation selection artifact records its creation time plus config,
protocol, calibration, and code hashes. Evaluation aborts if those inputs do
not match. Held-out values and local addresses are reused across global widths
where possible; global Gaussian draws are shared by prefix before
dimension-specific normalization. Random streams are derived from the
registered seed, split, geometry, and candidate coordinates and recorded in
provenance.

The low-variance held-out reference analytically integrates iid global keys
conditional on each held-out local episode. It is the same proven
`delta+A/D` identity, not an independent validation. Explicit folded-LA draws
are also evaluated with episode-level mean, standard error, and 95% normal
intervals. Facts within an episode are never treated as independent samples.
Held-out local `delta` and `A` are diagnostics only.

Reported outcomes include absolute prediction error; point and interval
feasibility; false-feasible, false-infeasible, and no-selection counts;
operation regret against the cheapest point-feasible full-grid reference under
the same memory cap; and exact-oracle local-read feasibility. Regret is
undefined for an infeasible selection. Borderline confidence intervals are
labelled uncertain while point decisions remain visible.

## Commands

Run from `test_time_gd`:

```bash
.venv/bin/python -m pytest -q tests/test_linear_memory_allocation.py
.venv/bin/python -m experiments.linear_memory.run_allocation \
  --config experiments/linear_memory/allocation_config.json --stage calibrate-select
.venv/bin/python -m experiments.linear_memory.run_allocation \
  --config experiments/linear_memory/allocation_config.json --stage evaluate
```
