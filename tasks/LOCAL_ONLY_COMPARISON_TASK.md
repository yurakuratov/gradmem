# EXP-LM-004: local-only versus hierarchical linear memory

PI brief, 2026-11-11. Cycle 1 of the next user-authorized milestone.

## Question and completion criterion

At a common expected-error target and mutable-state budget, when is a
hierarchical delta-writer/additive-LA store cheaper than an enlarged local-only
memory? Give the local-only writer every affordable positive integer width.
Include a direct additive local-only memory as a second strong comparator.
Positive, negative, or mixed results are valid. This is an IID linear-model
comparison under the existing operation proxy, not a systems or transformer
claim.

Completion requires independently audited error theory, exact bounded integer
allocation, frozen pre-evaluation choices, focused tests, fresh held-out
checks of chosen configurations, and an evidence report including failures
and storage/compute tradeoffs. Predeclare empirical feasibility coverage >=90%
at deployment epsilon for each method when its population oracle is feasible.
Empty denominators cannot pass. Report theory/measurement discrepancies and
intervals rather than retuning after outcomes.

## Ownership and execution gate

Research scientist owns NEW local-only-comparison source/config/protocol/tests
and evidence artifacts. You are not alone in the codebase. Preserve existing
notebooks, PI records, earlier executed source, and all EXP-LM-001/002/003
artifacts. No transformer changes, remote services, downloads, training, or
commits. The PI owns PROGRESS/STATUS/IDEAS. Scoped ignore exceptions are OK.

Implement and run focused tests first. Send finalized protocol/config and
commands for PI design-audit release BEFORE generating scientific held-out
data. Exact analytic optimization and synthetic unit tests are permitted
before release. No settings may be changed based on held-out outcomes.

## Common task and budgets

- IID unit spherical local addresses, independent zero-mean unit values, v=2.
  Do not run the reset-dependent correlated generator in this milestone.
- J={16,32}, Q={1,32}, epsilon={.10,.25}, mutable-state cap M=2048 scalars.
- PRIMARY theoretical comparison uses exact population optima at epsilon.
  Also freeze reserve selections at t=.95*epsilon for all methods, and apply
  the >=90% empirical deployment-epsilon feasibility check to these reserve
  selections. The exact theory has no calibration uncertainty; the reserve
  is a separate finite-sample sensitivity, not the primary frontier.
- Fresh seeds7707,8808,9909; 1024 held-out episodes per evaluated configuration.
- eta=1,K=1 for all delta writers. Inputs/addresses, routing and temporary
  arithmetic buffers are excluded consistently from persistent-state memory;
  report those exclusions. No learned encoder parameters.
- Use the unchanged mixed primitive-operation/MAC proxy. Report separate
  local write, global commit, and read components; no wall-clock speedup.

## Three methods and one allocation decision

1. Local delta: one W(v,p), reset only before the full J-fact stream; retrieve
   all facts from final W. All integer p=1..floor(M/v) are allowed.
   Memory=pv. Cost=J(3pv+v)+Qpv. No LA or fictitious commit/routing overhead.
2. Local additive: W+=yx^T, retrieve from final W. Same p domain and memory.
   Error=(J-1)/p. Cost=(J+Q)pv using the same outer-product MAC convention as
   global LA writes. This is a different local writing rule, clearly labelled.
3. Hierarchical delta+LA: W resets every h facts and commits every tau; h
   ranges over ALL positive divisors of J and tau over ALL divisors of h.
   Every integer p,D>=1 with pv(1+D)<=M is allowed. Oracle route is first
   same-group commit at/after the queried fact. Cost remains
   J(3pv+v)+(J/tau)Dpv+Q(Dpv+pv). Memory=pv(1+D). This includes degenerate
   h=1 and h=J; do not give those aliases an unregistered cost discount.

Select each method's least-cost expected-feasible allocation at epsilon and t. Also select
the best overall method, with ties by cost, memory, fixed method order
(local_delta, local_additive, hierarchical), then p,h,tau,D. Report method
choices, parameters, actual expected errors and state use, and all pairwise
cost comparisons only when both compared methods satisfy the error target.
Comparisons are under equal requirements and caps, not identical achieved
errors or equal actual storage. A positive result against one comparator
does not imply dominance against the other.

## Exact IID moment theory (independently audited before execution)

For a p-dimensional local delta writer, define

    lambda=1-1/p, beta=1-2/p+2/[p(p+2)], a_0=0,
    a_t=lambda*a_(t-1)+1.

Here a_t=E||W_t||_F^2. For fact j and snapshot c, define response energy e_cj:

    e_cj=a_c/p, c<j;
    e_jj=1;
    e_tj=beta*e_(t-1,j)+a_(t-1)/[p(p+2)]+1/p, t>j.

For c>=j the target alignment is E[y_j^T W_c x_j]=lambda^(c-j). Therefore
endpoint local-delta error is mean_j[e_Jj-2*lambda^(J-j)+1]. An independent
aggregate recurrence gives the same result:

    R_0=0,
    R_t=beta*R_(t-1)+(t-1)*(a_(t-1)/[p(p+2)]+1/p)+1,
    delta_J=(R_J-2*a_J+J)/J.

For hierarchy, let commits c=tau,2tau,...,h and r(j) be first c>=j:

    delta=mean_j[e_r(j),j-2*lambda^(r(j)-j)+1],
    A0=mean_j sum_(c != r(j)) e_cj,
    B=sum_c a_c/p,
    E=delta+[A0+(J/h-1)B]/D.

The independent audit confirmed both recurrences and the snapshot extension.
Use these exact moments for selection; no empirical
calibration is necessary. Verify h1, h2 endpoint and p1 edge cases. Enumerate
all affordable p; do not assume monotonicity of local error without proof.
For each hierarchical p,h,tau, cost rises with D, so use the smallest feasible
integer D with explicit neighbor/boundary checks. Validate against a tiny
brute-force domain. Guard only negligible roundoff negatives; never hide
materially invalid moments. Use special handling for A=0 and delta=t.

## Frozen predictions and held-out verification

Stages are predict-select (theory only) and evaluate. Seal config, protocol,
all local imported execution dependencies, analytic candidate/frontier tables,
and selections before held-out data. Refuse overwrite and dependency drift.
Do not present the analytic oracle as independent empirical validation.

Evaluate ONLY the union of unique chosen allocations at both epsilon and
.95*epsilon for each method/J/Q/target, plus fixed checks below, all seeds.
For the primary epsilon local-only selections, also evaluate width p-1/p+1
when inside their budget. For primary hierarchical selections evaluate D-1/
D+1 at fixed p,h,tau when eligible. These predeclared immediate neighbors
check the capacity boundary; do not use their outcomes to revise selections.
This includes boundary cases for diagnosis without selecting on their data.
No exhaustive simulation at all1024 widths. All comparisons use an identical
IID population. Pair values across methods; for the SAME p and J, pair the
whole flat local-key stream across methods/reset horizons before reshaping
into groups. Different p may have separate declared RNG streams. This avoids
algorithm-dependent data distributions. Global keys use a separate stream.

For each delta-local-only candidate, obtain empirical error from final W.
For additive-local-only, implement its direct write and read. For hierarchy,
compute empirical local delta_i and non-target A_i, then conditional-reference
error delta_i+A_i/D, plus explicit sampled folded LA. Average over facts
within an episode before computing SE and normal95% intervals. Do not treat
Q/target/method views reusing data as independent replications.

Fixed checks: (a) IID J32,p32,h=tau2,D12 hierarchy (known expected .09390797),
(b) local-delta J16,p16 and J32,p32 (may be infeasible; verify theory), and
(c) local-additive J32,p310 (population error .10 boundary), and (d) local-delta
J16,p1 (population error 2*(J-1)/J). Report per-seed
theory, empirical means, SE/CI, and residuals; boundary sample feasibility is
not proof/disproof of population equality.

Primary verification: per-method deployment-epsilon feasibility coverage of
the reserve selections (the primary epsilon choices may sit on the boundary),
mean prediction discrepancies, intervals, and explicit LA consistency. Exact
population operation regret is known from the analytic domain, so label it
as such; no held-out full-grid oracle is observed in this study. Report
reserve-target empirical feasibility separately. No rerunning/tuning to
obtain preferred rankings or interval outcomes.

## Evidence, tests, and runtime

Suggested new paths: experiments/linear_memory/local_only_comparison*.py/json/md
and run_local_only_comparison.py; tests/test_linear_memory_local_only.py;
reports/LOCAL_ONLY_COMPARISON_RESULTS.md; reports/results/linear_memory/EXP-LM-004/;
research/experiments/EXP-LM-004/.

Tests: aggregate/per-fact recurrences agree; known h1,h2,p1 formulas; synthetic
direct-writer check that is not merely a copy of the recurrence; additive
formula/write behavior; exact integer optimizer vs small brute force;
resource accounting without artificial local-only LA charges; same-p flat
address/value pairing across methods/h; dependency barrier/overwrite refusal;
abstention-safe coverage. Run previous three linear-memory test files too.
Do not alter them or their execution dependencies.

Keep scientific stages under10minutes CPU. Batch episodes if needed without
changing sample counts/distributions. Do not benchmark these Python study
loops as inference performance. Deliver concise method/cost/memory tables,
complete reproduction evidence, and a candid result. No figure required.
