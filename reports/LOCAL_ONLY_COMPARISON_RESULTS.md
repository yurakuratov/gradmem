# EXP-LM-004: local-only versus hierarchical linear memory

**Task ID:** EXP-LM-004  
**Status:** completed; registered acceptance passed  
**Question:** At the same IID expected-error target and mutable-state cap, when
is delta-writer plus additive-LA hierarchy cheaper than an enlarged local-only
memory?

## Methods

The comparison used the exact audited IID moment recurrences, not fitted
calibration. It optimized every integer local width `p=1,...,1024`. The
hierarchy additionally optimized every divisor `h|J`, every divisor `tau|h`,
and every affordable integer global width `D>=1` under 2,048 mutable scalars.
The comparators were an endpoint delta writer and a direct additive writer.

The registered grid was `J={16,32}`, `Q={1,32}`, and
`epsilon={0.10,0.25}`. Primary choices met `epsilon`; reserve choices met
`0.95 epsilon`. Evaluation used fresh seeds 7707, 8808, and 9909 with 1,024
episodes per unique configuration. All methods with the same `p,J,seed`
received the same flat IID key/value stream. The hierarchy reshaped it only
after generation; its global keys used a separate stream.

The cost is a mixed primitive-operation/MAC proxy. State excludes supplied
addresses, oracle routing and metadata, input encoding, query construction,
and temporary arithmetic. It is not measured FLOPs or wall-clock time.

## Primary exact population results

The optimum depends on workload. Direct additive local memory wins five of
eight scenarios, endpoint delta wins one, and genuine sharing (`h=4`) wins
the two `J=32,Q=1` scenarios. Hierarchy's other optima are degenerate
`h=1` LA stores and do not beat the strongest local-only method.

| J | Q | eps | overall winner | allocation | expected error | state | cost |
|---:|---:|---:|---|---|---:|---:|---:|
| 16 | 1 | 0.10 | local additive | p=150 | 0.100000 | 300 | 5,100 |
| 16 | 1 | 0.25 | local additive | p=60 | 0.250000 | 120 | 2,040 |
| 16 | 32 | 0.10 | local delta | p=85 | 0.098731 | 170 | 13,632 |
| 16 | 32 | 0.25 | local additive | p=60 | 0.250000 | 120 | 5,760 |
| 32 | 1 | 0.10 | hierarchical | p=39,h=4,tau=4,D=12 | 0.099902 | 1,014 | 16,054 |
| 32 | 1 | 0.25 | hierarchical | p=16,h=4,tau=4,D=12 | 0.246540 | 416 | 6,624 |
| 32 | 32 | 0.10 | local additive | p=310 | 0.100000 | 620 | 39,680 |
| 32 | 32 | 0.25 | local additive | p=124 | 0.250000 | 248 | 15,872 |

At `J=32,Q=1,epsilon=.10`, sharing costs 16,054 versus 20,460 for
additive local memory, a 21.5% reduction, and versus 33,432 for endpoint
delta. At `epsilon=.25`, it costs 6,624 versus 8,184, a 19.1% reduction.
Sharing uses more state in both cases: 1,014 versus 620 scalars, and 416
versus 248. These are compute savings under a common memory cap, not Pareto
dominance in compute and memory.

The full method-restricted ledger is:

| J/Q/eps | method | allocation | E | state | write | commit | read | total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 16/1/.10 | local delta | p=85 | .098731 | 170 | 8,192 | 0 | 170 | 8,362 |
| 16/1/.10 | local additive | p=150 | .100000 | 300 | 4,800 | 0 | 300 | 5,100 |
| 16/1/.10 | hierarchy | p=1,h=1,tau=1,D=150 | .100000 | 302 | 128 | 4,800 | 302 | 5,230 |
| 16/1/.25 | local delta | p=37 | .245406 | 74 | 3,584 | 0 | 74 | 3,658 |
| 16/1/.25 | local additive | p=60 | .250000 | 120 | 1,920 | 0 | 120 | 2,040 |
| 16/1/.25 | hierarchy | p=1,h=1,tau=1,D=60 | .250000 | 122 | 128 | 1,920 | 122 | 2,170 |
| 16/32/.10 | local delta | p=85 | .098731 | 170 | 8,192 | 0 | 5,440 | 13,632 |
| 16/32/.10 | local additive | p=150 | .100000 | 300 | 4,800 | 0 | 9,600 | 14,400 |
| 16/32/.10 | hierarchy | p=1,h=1,tau=1,D=150 | .100000 | 302 | 128 | 4,800 | 9,664 | 14,592 |
| 16/32/.25 | local delta | p=37 | .245406 | 74 | 3,584 | 0 | 2,368 | 5,952 |
| 16/32/.25 | local additive | p=60 | .250000 | 120 | 1,920 | 0 | 3,840 | 5,760 |
| 16/32/.25 | hierarchy | p=1,h=1,tau=1,D=60 | .250000 | 122 | 128 | 1,920 | 3,904 | 5,952 |
| 32/1/.10 | local delta | p=172 | .099955 | 344 | 33,088 | 0 | 344 | 33,432 |
| 32/1/.10 | local additive | p=310 | .100000 | 620 | 19,840 | 0 | 620 | 20,460 |
| 32/1/.10 | hierarchy | p=39,h=4,tau=4,D=12 | .099902 | 1,014 | 7,552 | 7,488 | 1,014 | 16,054 |
| 32/1/.25 | local delta | p=74 | .249932 | 148 | 14,272 | 0 | 148 | 14,420 |
| 32/1/.25 | local additive | p=124 | .250000 | 248 | 7,936 | 0 | 248 | 8,184 |
| 32/1/.25 | hierarchy | p=16,h=4,tau=4,D=12 | .246540 | 416 | 3,136 | 3,072 | 416 | 6,624 |
| 32/32/.10 | local delta | p=172 | .099955 | 344 | 33,088 | 0 | 11,008 | 44,096 |
| 32/32/.10 | local additive | p=310 | .100000 | 620 | 19,840 | 0 | 19,840 | 39,680 |
| 32/32/.10 | hierarchy | p=1,h=1,tau=1,D=310 | .100000 | 622 | 256 | 19,840 | 19,904 | 40,000 |
| 32/32/.25 | local delta | p=74 | .249932 | 148 | 14,272 | 0 | 4,736 | 19,008 |
| 32/32/.25 | local additive | p=124 | .250000 | 248 | 7,936 | 0 | 7,936 | 15,872 |
| 32/32/.25 | hierarchy | p=1,h=1,tau=1,D=124 | .250000 | 250 | 256 | 7,936 | 8,000 | 16,192 |

For intuition only, the fixed-`J`, large-`p` expansion is

`delta_J(p)=(J-1)/(2p)+(J-1)(2J+5)/(6p^2)+O(p^-3)`.

Thus delta writing needs about half the additive width at small error, but its
write proxy is about three times larger. The leading crossover is near
`Q=J`; finite-width corrections and integer allocation determine the exact
rankings above.

## Reserve sensitivity and held-out verification

The 5% reserve increased cost by 4.35%--6.67%, depending on scenario and
method:

| J/Q/eps | delta overhead | additive overhead | hierarchy overhead |
|---|---:|---:|---:|
| 16/1/.10 | 4.69% | 5.33% | 5.20% |
| 16/1/.25 | 5.36% | 6.67% | 6.27% |
| 16/32/.10 | 4.69% | 5.33% | 5.26% |
| 16/32/.25 | 5.38% | 6.67% | 6.45% |
| 32/1/.10 | 5.22% | 5.48% | 5.11% |
| 32/1/.25 | 5.38% | 5.65% | 4.35% |
| 32/32/.10 | 5.22% | 5.48% | 5.44% |
| 32/32/.25 | 5.39% | 5.65% | 5.53% |

Registered acceptance used empirical point feasibility of reserve selections
at deployment epsilon. It passed for every method: 24/24 dependent seed views
for local delta, local additive, and hierarchy's conditional reference. Direct
sampled folded LA also passed 24/24 hierarchy views. All reserve intervals
were fully below deployment epsilon for both the conditional reference and
direct sampled results. At the stricter reserve target itself, point means
were feasible in 24/24 delta, 16/24 additive, and 23/24 hierarchy views; that
stricter check was diagnostic rather than the acceptance criterion.

Primary choices often sit at the population boundary. Conditional-reference
point feasibility was 14/24 delta, 12/24 additive, and 13/24 hierarchy views.
Their intervals were mostly boundary-uncertain; this does not revise the
exact population rankings. Direct hierarchy outcomes were 12/24 point
feasible at primary epsilon.

Across 59 configurations and three seeds (177 rows), median absolute
population-prediction residual was 0.000333 and maximum was 0.00772. Raw
95% interval inclusion was 137/177. Thirty-six exclusions were deterministic
`p=1,h=1` conditional-reference rows differing from theory only by about
`1e-17`; a `1e-15` numerical tolerance gives 173/177. The remaining four
pointwise misses are compatible with inspecting 177 non-independent 95%
intervals; no simultaneous coverage claim is made. Sampled LA minus the
conditional reference had median absolute difference 0.000845 and maximum
0.00468 over 72 rows. This is implementation/distribution consistency using
the same identity, not independent validation of the theory.

All 15 fixed-check seed rows contained their theory values in their reference
95% intervals. The registered shared witness
`J=32,p=32,h=tau=2,D=12` had theory 0.093908; reference means were 0.094018,
0.093792, and 0.093868, while direct sampled means were 0.094166, 0.093654,
and 0.092901. The additive `p=310` population boundary and delta `p=1`
identity also behaved as registered.

## Limitations

- This is an IID linear associative-memory result under a supplied random
  address representation. Changing `p` changes that representation; encoder
  construction is uncharged.
- The hierarchy receives an extra oracle global route. Query construction,
  routing metadata, and address banks are excluded.
- The cost is analytical. Kernel efficiency, bandwidth, batching, and
  wall-clock behavior may change the practical crossover.
- Local-only LoRA, learned routing, transformer-layer writes, and fixed raw
  input encoders were not tested. The result is not a universal one-level
  versus hierarchical-memory theorem.
- The 24 views per method reuse seeds across `Q` and epsilon and are not 24
  independent experiments.

## Conclusion and recommended follow-up

Hierarchy has a real but workload-specific population advantage against both
optimized local comparators: at `J=32,Q=1`, it saves 19.1%--21.5% proxy
compute at the same error target and cap, while using more actual state. The
direct additive baseline wins most other registered scenarios, and endpoint
delta becomes competitive when reads are numerous. This supports an
allocation view: local width, reset/commit schedule, global width, and query
volume must be optimized jointly.

The next experiment should keep the exact population comparison but replace
the abstract proxy with measured kernels and charge realistic encoding and
route construction. A transformer/LoRA comparison should come only after a
matched state, parameter, and retrieval-error budget is specified.

## Reproduction and evidence

Commands, run from `test_time_gd`:

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

Tests: 76 passed in 2.04 seconds. Predict-select took 18.15 instrumented
seconds and evaluation 26.42 seconds, both below the 600-second cap. The
sealed selection SHA-256 is
`d69b256e487807cffe3f9326262bc025ed5e40fef0b0e8212bee211b7d6029f3`.
The frontier contains 180,224 rows, and the evaluation contains 59 unique
configurations and 177 seed-configuration rows.

Canonical artifacts:

- protocol/config: `experiments/linear_memory/local_only_comparison_protocol.md`,
  `local_only_comparison_config.json`;
- implementation/tests: `local_only_comparison.py`,
  `run_local_only_comparison.py`, `tests/test_linear_memory_local_only.py`;
- raw results and hashes: `reports/results/linear_memory/EXP-LM-004/`.
