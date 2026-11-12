# EXP-LM-003: refined local/global allocation

PI brief, 2026-11-11. Cycle 1 of the user's next-milestone task.

## Question and definition of done

Does shared local writing remain cheaper than independently written memories
when both local width and global address resolution are improved, at the same
expected retrieval error and folded-memory budget? Produce a reproducible
positive or negative result under the unchanged EXP-LM-002 operation proxy.
No claim about wall-clock acceleration, learned routing, or transformers.

The milestone requires a frozen protocol/config and pre-evaluation selection
artifact; meaningful tests; held-out evaluation; comparison with an independent
baseline optimized over the same p and D domain; error/coverage/regret tables;
and a report stating when sharing wins, loses, or is unresolved. Prediction
acceptance: at least 90% feasible-selection coverage among scenarios with a
feasible point oracle, and median feasible cost regret at most 10%. Empty
denominators cannot pass. A failed prediction is a valid completed experiment.

## Ownership and constraints

Research scientist owns new implementation/config/protocol/tests and evidence
files for EXP-LM-003. You are not alone in the codebase; preserve others' edits,
especially notebooks. Do not modify the PI records, transformer models, prior
allocation/model/pilot source files, or EXP-LM-001/002/AUDIT artifacts. Import
existing utilities; implement new functionality in new refined-allocation
modules. Scoped ignore exceptions are allowed to preserve reproducible files.
No remote services, downloads, training, commits, or expanded grids.

Implement and test first. Send PI the finalized protocol/config and proposed
execution commands; wait for the PI's design-audit release BEFORE calibration
or held-out scientific runs. This is an internal review, not user permission.

## Frozen scientific settings

- Same unit-key/unit-value delta writer, eta=1, K=1, v=2, reset every h facts,
  first subsequent snapshot oracle routing, tau independent of h.
- p in {2,4,8,16,32,64}; h in {1,2,4,8}; unique valid tau from {1,2,h}.
- Every positive integer D satisfying pv(1+D)<=2048 is eligible. No additional
  power-of-two restriction. Enumerating the scalar moment formula is cheap;
  do not materialize a sampled LA state for every integer D.
- Same iid and correlated (.7 Gaussian mixing weight) distributions as EXP002.
- New seeds 4404,5505,6606, with documented separate calibration/test streams.
- Calibration: 4096 isolated J=h groups per local configuration/seed/geometry.
- Held-out: 1024 episodes per local configuration/seed/geometry/J; J={16,32},
  Q={1,32}, target epsilon={.10,.25}. Reuse episodes across D,Q,epsilon and
  policies and explicitly state this dependence.
- Cost remains J(3pv+v)+C Dpv+Q(Dpv+pv), folded memory pv(1+D)<=2048.
  Report raw bank/address costs separately; no invented commit overhead.
- All primary selectors require predicted error <=.95*epsilon. This fixed
  5% relative slack is declared BEFORE data and applied to every policy.
  Evaluate actual feasibility at epsilon, and regret against the epsilon
  oracle. Slack is a heuristic reserve, not a statistical guarantee.

## Policies and theory

Primary policies: unrestricted; independent (h=tau=1, optimize p,D); shared
only (h>1, optimize p,h,tau,D). An endpoint-only policy is optional if trivial
to support, but must be frozen before running. Deterministic tie breaking:
cost, memory, then p,h,tau,D. Never restrict independent baseline to p4.

Use calibration prediction delta+[A0+(J/h-1)B]/D. Use exact h1 identity
delta=A0=0, B=1/p for ALL policies' h1 candidates, and report the empirical
calibration values as diagnostics. For h>1 use estimated moments.

Register these already-derived checks separately from policy selection:
IID h=tau=2 gives delta=(1/p+3/[p(p+2)])/2, A0=0, B=(2-1/p)/p.
At J16,p16,D16 expected error=.08943684896; cost6208 at Q1.
Independent J16,p2,D75 expected error=.10; cost5328 at Q1.
Evaluate both fixed witnesses with new held-out episodes, even if the second
fails the selector's slack criterion. Report theoretical and observed errors
with SE/CI, without counting boundary point feasibility as a proof/disproof.
Additional fixed witness, independently audited before EXP003 data:
IID J32,p32,h=tau2,D12 gives delta=.01700367647, B=.0615234375,
A=.9228515625, E=.0939079733456 < .095. Memory832; cost18496+832Q.
Every independent h1 memory with p>=2 meeting true epsilon=.1 has pD>=310,
and therefore cost>=20288+624Q, even relaxing integer D and the upper width/
memory restrictions. At Q1 the shared witness costs19328 versus lowerbound
20912, at least7.57% less; the inequality holds for integer Q<=8. This is an
under-proxy theoretical witness, not wall-clock or universal-memory dominance.
Evaluate it at J32 in all three iid seeds, including explicit sampled LA.

## Efficient, honest evaluation

Compute held-out delta_i,A_i once per local trace, then all integer-D
conditional-reference means and episode-level variances using the joint
moments: Var(delta+A/D)=Var(delta)+2 Cov(delta,A)/D+Var(A)/D^2.
Check this shortcut against direct episode-vector computation in tests.
Do not assume the normal-CI upper endpoint is monotone in D; enumerate D
for interval feasibility. Means are monotone because A>=0. Guard floating
ceil and exact-boundary cases. A zero-noise candidate at delta=epsilon is
feasible; delta>=epsilon with A>0 is not.

Evaluate explicit sampled folded LA for every UNIQUE selected candidate
across policies and scenarios, both fixed IID witnesses at J16, and fixed
sentinel (p4,h4,tau1,D32,J16) in both geometries/all seeds. Use fresh global
keys, reuse traces when appropriate, and deduplicate repeated evaluations.
This is a consistency check against the same-identity conditional reference,
not independent proof of the law. No need to explicitly sample every D.

Report point oracle and interval-certified oracle (pointwise normal95%
intervals, not simultaneous guarantees), cheaper uncertain alternatives,
false feasibility, abstention, feasible-scenario coverage, undefined regret
for infeasible selections, all policies' feasibility and cost. For sharing
versus independent compare both calibrated selections and held-out restricted
oracles; only label a win when both compared configurations are feasible.
Include cost ratio, selected h,tau,p,D and boundary sensitivity. Do not pool
48 scenario rows as independent replications. Include full-grid prediction
error by h, but avoid letting thousands of D copies imply new evidence.
Also report slack-matched (.95*epsilon) point-oracle coverage/regret diagnostics
to distinguish deliberate reserve cost from selection error. These do not
replace the primary epsilon-oracle criterion. Validate A nonnegative, allowing
only tiny floating-point roundoff to be clamped. If using an optional endpoint
comparator, restrict it to h>1 to keep it distinct from independent writes.

## Reproducibility and verification

Freeze config, protocol, all imported local execution dependencies (including
model.py and allocation.py), calibration, predictions, and selections before
evaluation. Abort on drift. Refuse to overwrite prior experiment artifacts.
Store raw moment summaries, candidate/policy tables, config and source hashes,
commands, environment, and a concise evidence report. Suggested names:
experiments/linear_memory/refined_allocation*.py/json/md;
tests/test_linear_memory_refined_allocation.py;
reports/REFINED_ALLOCATION_RESULTS.md;
reports/results/linear_memory/EXP-LM-003/;
research/experiments/EXP-LM-003/.

Focused tests must include moment variance shortcut, all-integer selection
versus tiny brute force, optimized independent baseline, zero-noise/boundary
cases, dependency barrier, and abstention-safe coverage. Reuse earlier guards
where applicable. Run prior two linear test files as regression checks.
Keep CPU runs bounded; batch episodes if needed without changing the sample
count. Report a blocker before exceeding 10 minutes for scientific execution.
