# EXP-LM-004 protocol: local-only versus hierarchical linear memory

## Question and status

This calibration-free study asks when a hierarchical delta-writer plus folded
linear-attention state is cheaper than either an enlarged endpoint delta
writer or a direct additive writer at a common IID population-error target and
mutable-state cap. The primary result is the exact population allocation; new
held-out draws only check the registered selections and boundaries.

The protocol, configuration, analytic tables, selections, new source, runner,
and imported execution dependencies are sealed before held-out generation.
Scientific evaluation requires focused tests and PI design-audit release. No
setting may change in response to held-out results.

## Common task

Each episode contains `J in {16,32}` IID unit-spherical local keys and
independent unit two-dimensional values. Query count is `Q in {1,32}`, target
error is `epsilon in {0.10,0.25}`, and mutable state is capped at 2,048
scalars. Delta writers use one unit-rate update per fact. Supplied address
generation, oracle-route discovery, route/key metadata, and temporary
arithmetic buffers are excluded consistently from persistent state. There are
no learned encoder parameters.

All methods acting at the same `p,J,seed` receive the identical flat key and
value stream. A hierarchical method reshapes this flat stream into reset
groups only after generation. Global keys use a separate stream. This keeps
the IID fact distribution fixed as `h` changes.

Changing `p` changes the dimension of the random address representation rather
than embedding one fixed raw input with a charged encoder. The hierarchy also
receives an oracle global route. Input encoding and query construction are
outside the proxy, so the comparison is not a universal theorem for methods
fed the same raw inputs.

## Methods, resource proxy, and allocation

`local_delta` keeps one `W in R^(v x p)`, resets it only before the full
stream, applies `W <- W+(y-Wx)x^T`, and reads every fact from final `W`.
Every integer `p=1,...,1024` is eligible. Its state is `pv`, and its proxy is

`J(3pv+v)+Qpv`.

`local_additive` directly applies `W <- W+yx^T` and reads from final `W` over
the same integer widths. Its state is `pv`, exact error is `(J-1)/p`, and its
proxy is `(J+Q)pv` under the same outer-product MAC convention used for LA
state construction.

`hierarchical` resets the delta writer every `h` facts and commits every
`tau` facts, where every divisor `h|J` and every divisor `tau|h` is eligible.
Every integer `p,D>=1` satisfying `pv(1+D)<=2048` is eligible. Oracle routing
selects the first same-group commit at or after a fact. Its proxy is

`J(3pv+v)+(J/tau)Dpv+Q(Dpv+pv)`.

The hierarchy retains the registered cost even at degenerate `h=1` or `h=J`.
No alias receives a discount. Costs are mixed primitive-operation/MAC proxies,
not measured FLOPs or wall-clock speed.

For each method and scenario, select the least-cost allocation satisfying the
exact expected-error target. Repeat at `0.95 epsilon` as a conservative
finite-sample sensitivity. Ties use cost, state, fixed method order
`local_delta,local_additive,hierarchical`, then `(p,h,tau,D)`. The overall
winner is the cheapest of the three method-restricted optima. Pairwise costs
are compared only when both methods meet the same target and cap; actual
achieved errors and state sizes need not be equal.

## Exact IID theory

For a `p`-dimensional delta writer define

`lambda=1-1/p`, `beta=1-2/p+2/[p(p+2)]`, `a_0=0`,
`a_t=lambda*a_(t-1)+1`.

Here `a_t=E||W_t||_F^2`. For fact `j` and state after fact `c`, response energy
`e_cj=E||W_c x_j||^2` obeys

- `e_cj=a_c/p` for `c<j`;
- `e_jj=1`;
- `e_tj=beta*e_(t-1,j)+a_(t-1)/[p(p+2)]+1/p` for `t>j`.

For `c>=j`, target alignment is `lambda^(c-j)`. Endpoint delta error is

`mean_j [e_Jj-2 lambda^(J-j)+1]`.

The implementation also evaluates the audited aggregate recurrence

`R_0=0`,
`R_t=beta R_(t-1)+(t-1)[a_(t-1)/(p(p+2))+1/p]+1`,
`delta_J=(R_J-2a_J+J)/J`,

and tests equality with the per-fact expression, including `p=1`.

For hierarchy, commits are `c=tau,2tau,...,h`, and `r(j)` is the first
commit at or after fact `j`:

`delta=mean_j[e_r(j),j-2lambda^(r(j)-j)+1]`,
`A0=mean_j sum_(c != r(j)) e_cj`,
`B=sum_c a_c/p`,
`E=delta+[A0+(J/h-1)B]/D`.

Only negligible floating cancellation may be clamped. For each `(p,h,tau)`,
the smallest feasible integer `D` is found with explicit neighbor checks and
bounded enumeration fallback. Tests compare the optimizer with brute force.

## Frozen analytical and empirical stages

The predict-select stage writes the complete local-method width frontiers,
the minimum-`D` hierarchy frontier for every `(p,h,tau)`, method-restricted
selections, overall selections, and the exact evaluated-configuration union.
It seals hashes of config, protocol, new module/runner, and imported
`allocation.py`/`model.py`. Evaluation refuses dependency drift or overwrite.

The union contains all selections at both multipliers, primary local-only
width neighbors `p-1,p+1` when eligible, primary hierarchy `D-1,D+1`
neighbors when eligible, and fixed checks. Neighbors are diagnostics and
never alter selections. Evaluation uses seeds 7707, 8808, 9909 and 1,024
complete episodes per unique configuration.

Endpoint delta and additive candidates are read directly from their final
state. Hierarchy reports the conditional reference `delta_i+A_i/D` and an
explicit sampled folded-LA result. Episode-mean error is the sampling unit;
normal 95% intervals use between-episode SE. Reused `Q`, epsilon, method, and
neighbor views are dependent and are not independent replications.

The registered fixed checks are hierarchy `(J32,p32,h=tau2,D12)` with error
0.09390797334558823; delta endpoints `(J16,p16)` and `(J32,p32)`; additive
boundary `(J32,p310)` with error 0.10; and delta `(J16,p1)` with error 1.875.
Boundary sample placement is not evidence against an exact population
equality.

## Acceptance and reporting

For each method, reserve selections at `0.95 epsilon` must achieve at least
90% empirical point-feasibility coverage at deployment `epsilon` whenever the
population selection exists. Empty denominators fail. Primary epsilon choices
are reported separately because they may lie exactly on a boundary.

The report gives exact population optima, all pairwise feasible cost and state
comparisons, prediction residuals and intervals, direct LA consistency, and
failures. Exact population regret is available analytically; no held-out
full-grid oracle is claimed. Conclusions are limited to this IID linear model
and analytical proxy, without transformer, LoRA, learned-routing, or
wall-clock claims.

## Commands after design release

From `test_time_gd`:

```bash
.venv/bin/python -m pytest -q tests/test_linear_memory_local_only.py \
  tests/test_linear_memory_refined_allocation.py \
  tests/test_linear_memory_allocation.py tests/test_linear_memory.py
.venv/bin/python -m experiments.linear_memory.run_local_only_comparison \
  --config experiments/linear_memory/local_only_comparison_config.json \
  --stage predict-select
.venv/bin/python -m experiments.linear_memory.run_local_only_comparison \
  --config experiments/linear_memory/local_only_comparison_config.json \
  --stage evaluate
```
