# EXP-LM-001 protocol: intermediate commits in linear associative memory

## Objective

Test whether intermediate whole-operator snapshots add writer-gradient routes
at identical forward storage, and verify explicit retrieval-error predictions
for random-key linear attention. A positive optimization result is not required.

## Pre-registered model and data

The local state is `W in R^(v x p)` and each fact performs one update
`W <- W + eta (y - W x) x^T`. A shared linear encoder maps twelve fixed
one-hot fact identities to normalized local addresses (`p=4`). This encoder is
a learned fixed-position codebook shared across episodes, not a content-key
encoder; the same identities and initialization are used across arms. The shared step
size is `1.25 sigmoid(a)`, initialized to `0.4`. Targets are fresh iid Gaussian
vectors normalized to unit length (`v=3`) in every episode. Reset horizon is
`h=6`. Endpoint arms use `tau=6`; intermediate arms use `tau=2`. Random global
keys are iid normalized Gaussian vectors (`D=16`) indexed by segment endpoint.

Global routing is oracle-controlled: every query uses the key/index of the
first commit in its own group after its write. It is therefore a snapshot-
address control, not a learned/content-routing result. Exact gather and
random-key linear attention read the same snapshots. LA predictions construct
and query the ideal folded state; the raw snapshot bank is still retained for
autograd and diagnostics and is accounted for separately.

## Arms and intervention

For each reader, compare endpoint-only, intermediate snapshots detached only
on non-final storage branches, and fully differentiable intermediate
snapshots. Every group-final commit remains differentiable. The live recurrence
is never detached. The local encoder also supplies the read query `x`, so the
detached arm retains query-side encoder gradients while removing the
intermediate memory-output route. Fixed-parameter probes report encoder and
step-size gradients separately.

## Training and evaluation

Seeds are 101, 202, and 303. Each arm sees the same pre-generated 120 training
batches per seed (32 episodes each) and the same held-out 128 episodes. Adam
uses learning rate 0.03. Evaluate at step 0 and every 20 updates. The
predeclared normalized-MSE target is 0.20; non-reaching arms are reported as
censored. All-fact query loss is used, so group-final paths contribute.
Training wall time excludes evaluation. CPU execution uses one PyTorch intra-
op and one inter-op thread with deterministic algorithms.

Metrics: held-out normalized MSE, MSE by routing age, LA local error,
contamination, cross term, conditional analytical contamination prediction,
step-size and encoder gradient norms, fixed-parameter detached/full gradient
probe, time and steps to target. Operation counts are formula estimates, not
measured FLOPs. The ideal folded `D*p*v` state is reported separately from raw
snapshot/key banks and autograd tensors.

## Analytical checks

Tests cover orthogonal no-forgetting, repeated-key overwrite, the linear
transition gradient identity, folded/direct LA equivalence, matched detached/
full forward values, preserved live recurrence and final gradients, reset/tail
coverage, and exact error decomposition.

For iid isotropic unit global keys independent of fixed payloads and query
`q=k_r`, conditional contamination is

`E ||sum_(c!=r) (k_r^T k_c) z_c||^2 = (1/D) sum_(c!=r) ||z_c||^2`.

The payloads may be correlated. A reproducible Monte Carlo test uses correlated
snapshots. The two-fact closed-form crossover is also checked for the frozen
`rho` and `D` grid in `pilot_config.json`.

## Commands

```bash
.venv/bin/python -m pytest -q tests/test_linear_memory.py
.venv/bin/python -m experiments.linear_memory.run_pilot \
  --config experiments/linear_memory/pilot_config.json
```

The run writes raw CSV/JSON plus hashes and environment provenance under
`reports/results/linear_memory/EXP-LM-001/`.
