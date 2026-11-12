# PROGRESS — multi-segment GradMem: current status, facts, hypotheses, and what is missing

Living summary; update at every milestone. Journal of changes: `STATUS.md`.
Pool of future directions: `IDEAS.md`. Cluster workflow: `docs/REMOTE.md`.

**One-line state:** independently written per-segment memories + segment-wise
linear-attention retrieval (the Hopfield mode) solve catastrophic forgetting on
single-key KV retrieval, but the one-shot read interface fails any task that
must **integrate content from two segments** (ordered pair concatenation).
**The ladder has now RUN** (results:
[`reports/READ_LADDER_RESULTS.md`](reports/READ_LADDER_RESULTS.md)): every
per-token/per-level linear-state read fixes concat (exact-softmax 1.00,
all-positions 0.997, interface two-pass 0.999) while the one-shot control
(0.25), the bank-prefix control (0.02) and the query-blind interface (0.01)
fail — and the conditioning ablation shows **pass-1 query conditioning is
load-bearing** (prediction held, Δ > 0.98 per seed). The deployable read is
(d′): interface-only reads + two-pass conditioning. **The interference curve
vs N is decided** (2026-11-12,
[`reports/INTERFERENCE_CURVE_RESULTS.md`](reports/INTERFERENCE_CURVE_RESULTS.md)).
After the 2026-11-03 double infrastructure failure, the exact-softmax control
was resumed and completed the full curriculum — EM ≥ 0.99 at every phase
through N=96 (all threshold stops; N96 within 0.5–2.5k steps). On single-key
with the deployable read, **superposition binds at N=32**: Δ against exact
softmax at the same interface sites is 0.0005 / 0.0025 / **0.065** /
**0.58** at N=8/16/32/64, and the interface read plateaus at EM 0.41 at N=64
(vs control 0.99 at the identical protocol). This matches the random-key
estimate √((N−1)/d_h) with key width d_h = 32 (a hypothesis; d_h sweep
pending). The carried-memory baseline (same curriculum, every segment
writing the SAME M=2 tokens;
[`manifests/kv_interference_adaptive.yaml`](manifests/kv_interference_adaptive.yaml))
fails at every N (0.82 / 0.48 / 0.15 at N=8/16/32). Per-segment slots are
the first-order effect; read superposition is second-order and appears only
once N ≈ d_h.

## The challenge

Write contexts that arrive over time (segments) into a tiny memory (M tokens)
such that the model can still (a) recall any individual fact later — no
catastrophic forgetting — and (b) **combine facts from different segments in
one read** (multi-key queries, joins). Plain carried-memory gradmem fails (a);
the current Hopfield read fails (b).

## Established facts

1. **Carried-memory multi-segment gradmem forgets.** The inner loop over
   segments is a running sum of gradient steps with no protection of stored
   directions; forgetting matrices show old pairs degrading as later segments
   are written (2026-09-10; see also [`STAR.md`](docs/STAR.md) and
   [`docs/ADAPTIVE_MEMORY.md`](docs/ADAPTIVE_MEMORY.md) for the analysis).
2. **Independent per-segment writes + attention retrieval mitigate
   forgetting.** Hopfield mode (per-segment K-step gradient write into M
   tokens; store key = segment compression, value = post-inner-loop memory;
   retrieve = one query-conditioned mixture): with **2 mem tokens, 8 KV pairs,
   4 pair-aligned segments** all pairs are stored perfectly; the forgetting
   matrix stays clean also at key dim 64 (2026-09-10).
3. **STAR and learned gates are second-order effects on the carried memory.**
   STAR ≈ +10% under perturbation; learned gated SGD updates make no
   difference (2026-09-10).
4. **Segmentation must be pair-aware.** Fixed token windows straddle `!K:V!`
   pairs, making them unretrievable from any single slot; pair-aware
   segmentation fixes it (commit `0f0478b`; shared helpers in
   [`grad_memgpt.py`](grad_memgpt.py)).
5. **Cross-segment integration fails under the one-shot linear-attention
   read.** Ordered pair concatenation (`?!K1:?!K2:` → `V1!V2!|`, keys in
   different segments; dataset in [`kv_dataset_utils.py`](kv_dataset_utils.py)
   `query_mode='ordered_pair_concat'`, commit `d5b6f18`): vanilla linear
   attention severely fails (2026-10-15).
6. **Non-linear sparse value codes do not improve it.** top-k / ReLU+L1 /
   gated value projections (commit `d02209d`,
   [`configs/gradmemgpt/kv_retrieval/hopfield_value_activation.yaml`](
   configs/gradmemgpt/kv_retrieval/hopfield_value_activation.yaml)) leave the
   concat result unchanged.
7. **Why (read-interface analysis).** Three coupled failures in the current
   READ: **routing** — the whole query is compressed to one vector, one score
   per stored segment, so a two-key query is a compromise direction; **payload**
   — softmax weights are convex, so retrieval is a superposition `Σaᵢvᵢ`
   inside the fixed M·d space that the reader must demix; **no iterative
   retrieval** — one pre-read mixture per forward, so pointer-chasing is
   structurally out of reach. A *linear* value projection is algebraically
   eliminable (2026-10-05 note: `ΣaᵢEvᵢ = E(Σaᵢvᵢ)`), the stored-code
   nonlinearity sits before the mixing while the post-mixing decode is linear,
   and LayerNorm makes raw-vs-softmax magnitude differences moot — no setting
   of the one-shot mixture (raw/softmax/β-softmax/mean) can deliver both
   payloads separately.
8. **The ladder ran (2026-10-22): every per-token/per-level read fixes
   ordered-concat; both one-shot-style interfaces fail.** Best two-key EM
   (mean of seeds 4378/43894/2343, ordered-concat): exact-softmax 1.000,
   linear-state all-positions 0.997, interface two-pass 0.999 — vs hopfield
   one-shot 0.248, bank-prefix control 0.024. The hopfield failure is exactly
   the compromise-query signature: each value individually ≈ 0.5, joint ≈
   0.5·0.5 = 0.25. The bank-prefix failure (with the SAME stored tokens that
   give (c)/(d)/(d′) 1.00) indicts the prefix interface, not write quality.
   Single-key regression: 0.997+ EM. Report:
   [`reports/READ_LADDER_RESULTS.md`](reports/READ_LADDER_RESULTS.md).
9. **Interface conditioning is load-bearing (ablation, 2026-10-22).** Removing
   the interface pass-1 (`read_state_interface_conditioning=false`, the
   query-blind "left-prefix causal" cell) collapses the interface arm from
   0.999 to 0.012 (Δ > 0.98 on every seed) while its reads stay active and
   calibrated (gate, ‖μ‖/‖x‖, β ≈ the conditioned arm) — the fixed M·d
   channel carries query-agnostic content, exactly the pre-registered
   prediction. The causality trap is demonstrated empirically: per-level
   state reads fix concat ONLY when their queries can see the task query.

## Theoretical scope clarification (2026-11-02; no new experiments)

Independent segment writes protect earlier segment encodings from subsequent
inner-loop gradient rewriting, conditional on fixed outer parameters. This
is narrower than a guarantee against retrieval forgetting. Folding those
encodings into `S = sum k v^T` can still produce key-collision cross-talk;
the query-conditioned interface adds a separate extraction constraint.
Carried SGD has local transition Jacobian `I - eta H` (unclipped plain SGD),
whereas independent additive state writes have identity transition Jacobian.
Both mechanisms still require a successful decoder. The current linear mode
retains a bank and builds S at READ, rather than incrementally maintaining
only S. At d=128/H=4 its 4096-coordinate matrix state is also larger than
M=2 carried tokens (256 coordinates), so equal token count is not a matched
state-budget comparison. Existing facts support mitigation in the measured
regime; they do not establish unlimited retention or isolate write-rule
effects from state size and reader changes. See the 2026-11-02 STATUS entry.

**Capacity/compute follow-up (2026-11-02; analytic, not benchmarked):** a
linear two-stage model with local address width p, global width D, T segments
and b arbitrary vector targets per segment requires b<=p and T*b<=D for
exact recall with fixed linear query features. Orthogonal segment addressing
can attain that bound; random segment frames instead incur relative
cross-talk MSE `(T-1)*b/D` under orthonormal local keys and iid centered
values. This does not set the nonlinear reader's empirical KV capacity.
Actual interface arithmetic currently computes all-position reads and then
masks their outputs (`grad_memgpt.py:1679-1680`); 2*M*L is an active-injection
count, not the number of computed reads. At N=8,H=4,d_h=32 the 4096-scalar
state is larger than the 2048-scalar projected K/V bank. Folded reads gain
asymptotically with growing N and reuse, but no small-N runtime advantage
has been established here. See the follow-up STATUS entry for assumptions
and operation-count crossover.

**Fixed-error comparison (2026-11-03; theoretical):** shallow-token count is
an insufficient comparator for linear-state complexity. Independent deep
tokens, per-layer KV prefixes, and fast LoRA can expand writable state with
no proportional increase in sequence length. Compare minimum total write
plus query-read cost at matched retrieval error and memory budget. For a
linear target map, the query-weighted singular-value tail determines the
minimum factor rank at a given error; a factored LA state and a LoRA-style
map can implement the same read. Online acquisition remains a separate
constraint: useful new writable directions are measured by J_new projected
onto the old-query Jacobian's nullspace, not parameter count alone. No
universal LA advantage follows; cheap commits must offset encoding,
interference-driven state size, and read cost against the best competing
write schedule. No new empirical result; see the 2026-11-03 STATUS entry.

**Complementary hierarchy (2026-11-09; analytic):** LA can index whole
gradient-written local memories rather than individual tokens. A local
p-by-v matrix stored as one payload needs a D-by-(p*v) outer state, with
tensor-product query (global q, local x). Exact orthogonal capacity is D*p,
paid by D*p*v scalars; previous D-only bounds refer to flat total address
width. At fixed N, local b-fact payload size P, and random-key error epsilon,
outer storage scales as N*P/(b*epsilon), dense commits as
N^2*P/(b^2*epsilon). Lossless grouping P=b*v preserves leading storage but
reduces dense commits; true compression can also reduce storage, while
overparameterized payloads may increase it. Evaluate local fitting error,
decoder sensitivity, routing, and payload format before applying these
ideal laws to prefixes/LoRA. This whole-payload design differs from the
implemented token-wise linear state. See 2026-11-09 STATUS; no new runs.

**Contribution/evidence audit (2026-11-09):** the current adaptive gap is a
whole-architecture comparison, not an isolated LA/optimization effect.
Exact-softmax independent writes converge faster than LA in the matched
interface arms; outer training steps do not measure inner WRITE convergence.
Short interrupted phases are censored, and width-dependent capacity remains
a hypothesis. The cross-talk diagnostic is non-top-1 absolute similarity
before site masking, not target-aware/value-weighted noise. Proposed paper
direction (draft only): predict and validate local/global capacity allocation
at fixed error, supported by a capacity/read-matched write-isolation study.
See STATUS 2026-11-09(c) and IDEAS; no new experiments or report edits.

**Required baseline extension (2026-11-09):** include sequential shared
gradient writes INSIDE each local group, followed by one outer commit and
a local reset. Local carry horizon h interpolates independent elementary
writes (h=1) and fully carried WRITE dynamics (h=J). Treat h as an allocation
variable with p,D,K: larger groups reduce outer entry count but may worsen
local retention and credit assignment. Total inner steps stay J*K at fixed
per-segment K; maximum local write dependency is h*K before graph truncation.
Compare against joint writing of the same group and report real compute.
Specified in IDEAS; no implementation or experiment performed.

## Contribution roadmap (2026-11-09; implementation started 2026-11-10)

**Reading entry point:** [Introduction to local gradient memory and global
associative storage](docs/HIERARCHICAL_MEMORY_INTRODUCTION.md) defines the
notation, derives the basic results, and explains the proposed mechanisms
without requiring code inspection.

**Execution update (2026-11-11):** the standalone linear writer, analytical/
gradient checks, bounded snapshot pilot, and first held-out allocation study
are implemented. See the reviewed results below,
[`LINEAR_MEMORY_PILOT.md`](reports/LINEAR_MEMORY_PILOT.md), and
[`ALLOCATION_TRANSFER_RESULTS.md`](reports/ALLOCATION_TRANSFER_RESULTS.md).
EXP-LM-002 selected independent local writes throughout its limited grid.
The subsequent EXP-LM-003 refinement found shared-writing savings for IID
addresses at low query count under the same operation proxy; see
[`REFINED_ALLOCATION_RESULTS.md`](reports/REFINED_ALLOCATION_RESULTS.md).
EXP-LM-004 now compares against enlarged local delta and direct additive
memories over every affordable integer width. Hierarchy wins two of eight
registered population scenarios; strong local-only methods win the other six.
See [`LOCAL_ONLY_COMPARISON_RESULTS.md`](reports/LOCAL_ONLY_COMPARISON_RESULTS.md).
The general phase diagram, measured systems costs, deep-prefix/LoRA comparisons,
and transformer integration remain pending.

**Working thesis:** local gradient writing, global associative storage, and
intermediate commits offer distinct ways to spend memory and compute.
Predict when each is worthwhile at a fixed retrieval error, including
regimes where a strong one-level memory wins. Intermediate commits may
improve outer credit assignment while retaining a persistent local writer;
this is a falsifiable supporting mechanism, not an established speedup.

**Target deliverables:** an explicit linear associative-memory theory;
a controlled gradient-routing study; and a validated allocation rule for
local size p, global address D, reset horizon h, commit interval tau, and
inner-step count K. No universal dominance or novel-attention-shortcut claim.
Execution status is recorded separately; this plan does not supersede the
historical results above.

### Stage 0 — freeze the comparison and cost model

- Define N facts, J elementary segments, Q queries per written context,
  a retrieval-error target epsilon, an outer-training budget, and an inference
  memory budget. Keep local address size p separate from actual payload
  size P(p), retained-fact count b, and architectural parameter count.
- Specify local reset h and commits tau independently, including final
  partial groups; record optimizer-buffer resets and derivative truncation.
  Distinguish whole-payload theory from implemented token-wise K/V storage.
- Primary deployment objective: minimize C_local + C_commit + Q*C_read
  subject to held-out error <= epsilon and the memory/training budgets.
  Also report training FLOPs/time to target error as a separate frontier.
  Count temporary banks, K/V tensors, autograd activations, projection costs,
  and actual read sites; DP is only the ideal folded whole-payload state.
- Separate frozen-outer-weight memory acquisition/retention from end-to-end
  outer optimization. Use normalized MSE in the linear model and explicit
  KV error/EM targets in the transformer; no assumed universal conversion.
- Audit prior work on gradient-written memory, recurrent attention shortcuts,
  and associative memory to identify the precise gap before claiming novelty.

**Exit artifact:** one comparison specification, baseline table, cost ledger,
and falsifiable claims, with error targets and evaluation protocol fixed
before the corresponding experiments.

### Stage 1 — derive a minimal linear theory

Use $W_j=W_{(j-1)}+\eta*(v_j-W_{(j-1)}x_j)*x_j^T$ and explicitly specified local
key/value distributions. Encode whole $W$ snapshots with global keys in $D$
dimensions; begin with fixed/oracle keys, then test learned routing.

1. Derive acquisition and forgetting versus p,K,h and local-key geometry;
   include orthogonal-key cases with no overwrite interference.
2. Derive retrieval error with multiple correlated snapshots. Retain payload
   and address covariance, duplicated facts, bias, and cross terms rather
   than automatically substituting (number_of_commits-1)/D. Separate an
   exact oracle snapshot read from implementable query-dependent routing.
3. Prove the gradient-path decomposition in IDEAS. Under restricted fixed
   routing/linear-quadratic assumptions, analyze conditioning or convergence
   if tractable. Otherwise state only a path-length theorem; stronger or
   shorter gradients alone do not prove faster convergence.
4. Combine the error calculation with measured/derived operation counts to
   obtain crossover conditions against specified one-level algorithms.
   Include finite-rank/information constraints and cases where hierarchy
   loses; factorized or sparsely addressed one-level solutions can be cheap.

**Exit artifact:** propositions with explicit assumptions and a predicted
phase diagram over local capacity, address size, and commit frequency. A
bound that only says 'larger memory helps' is insufficient: it must select
an allocation or predict a crossover that can be tested out of sample.

### Stage 2 — test the mechanism in a controlled linear system

First verify numerical predictions for the fixed writer. To test OUTER
optimization, define shared trainable writer/encoder parameters and train
them across associative-recall episodes; a parameter-free delta rule alone
cannot establish an outer-training speedup.

Compare endpoint-only commits, intermediate commits with snapshot-branch
stop-gradient, and fully differentiable intermediate commits. Keep final
commits and the live recurrence differentiable in all applicable arms.
The latter two must have equal forward values at identical parameters and
different intended writer gradients. Run an exact-reader control on the
same snapshots to isolate superposition. Keep gates, normalization, keys,
loss scale, and training protocol matched; permit equally budgeted tuning.

Sweep h and tau at fixed p,D,K first; then vary p,D and key correlations
around predicted crossovers. Measure error by fact age, target-aware
value-weighted read contamination, gradient contributions/alignment, and
training FLOPs/time to target. Report memory and non-reaching/censored runs.

**Gate:** does differentiable snapshot access improve training compute
beyond identical forward snapshot storage, reproducibly across seeds?
If only snapshot availability helps, retain a retention/capacity claim and
drop the gradient-speedup claim. If exact attention benefits but LA does
not, diagnose routing/interference before escalating model or context size.

### Stage 3 — falsify the allocation rule on held-out linear problems

Estimate local distortion, payload cost, and noise sensitivity on a limited
calibration set. Choose p,D,h,tau,K using the theory before evaluating new
N, query counts Q, and key-correlation regimes. Compare with an exhaustive
small-grid oracle and simple fixed-allocation heuristics. Report feasibility
(meeting epsilon), deployment/training cost, and cost regret relative to
the best feasible grid configuration. State the grid's limits; it is not a
global optimum. Preregister acceptable regret and uncertainty using pilot
precision, without tuning them on the held-out results.

**Gate:** predicts useful choices beyond the calibration grid and identifies
both favorable and unfavorable regimes. If coefficients require retuning
at every N, narrow the theory's claim to descriptive scaling.

### Stage 4 — validate in GradMem against strong baselines

Begin with controlled N16/N32 KV tasks, identical initialization/data order
where architectures permit, and matched tuning budgets. Existing curriculum
curves motivate this experiment but are not a causal convergence comparison.
Use repeated seeds, common checkpoints for frozen-writer interventions, and
confidence intervals; do not equate fewer outer steps with lower compute.

Required baseline families:
- One-level shared memory with increased local capacity, including deep
  prefixes/tokens and a competitive fast-LoRA configuration, so sequence
  length is not artificially tied to memory capacity.
- Independent writes plus global storage (h=tau=1).
- Shared segmented local writes with endpoint-only commits (tau=h).
- Shared local writes with intermediate commits (tau<h), both branch-
  detached and differentiable.
- Exact attention over the same snapshots as a routing/reader control.
- Sequential subsegment versus joint encoding of the SAME local group,
  with actual compute matched/reported rather than equal nominal K.

Stage these comparisons; do not run the full Cartesian product. First use
one local parameterization to isolate h/tau and the gradient intervention.
Then compare the best configurations on resource frontiers against stronger
one-level and alternative local parameterizations. Expand context lengths
only after the mechanism and cost accounting are credible.

**Gate:** a reproducible improvement on a specified error/compute/memory
frontier against competitive baselines, with the mechanism's scope stated.
For broader claims, test a second local parameterization. Avoid averaging
LoRA factors as if they were a linear operator; define payload encoding and
include its cost and distortion. Defer that hierarchical variant if needed,
but retain a strong one-level LoRA competitor.

### Stage 5 — determine the paper's scope

Minimum coherent result: linear propositions and failure regimes, held-out
allocation predictions, a causal snapshot-gradient ablation (including a
null result if found), and transformer resource frontiers with strong local
baselines. Add ordered-pair/multi-fact retrieval before claiming integration
benefits; add a longer realistic task only after the controlled results.

Planned main figures: (1) local/global/cadence phase diagram with predictions
and measurements, (2) matched-forward snapshot gradient intervention,
(3) fixed-error compute/state frontiers, (4) held-out allocation regret.
The evidence chooses whether the paper leads with resource allocation,
optimization, or retention. Faster inner fitting, universal superiority,
and constant-memory unlimited capacity remain outside the claims.

**Immediate next milestone (after EXP-LM-004):** connect the separate local
write, global commit, and read costs to an actual implementation. Freeze the
payload/address mapping and a measured-cost protocol before running a matched
transformer comparison. Include enlarged/deep local storage and fast LoRA
without treating their capacity as additional sequence length. The linear
comparison establishes a restricted compute/storage tradeoff with oracle
routing; it does not establish a transformer or wall-clock advantage. Broader
convergence theory remains open.

## Linear milestone specification and derivations (2026-11-10)

**Scope fixed before pilot:** one associative pair per elementary segment;
local W in R^(v x p); K delta-rule updates per segment; reset every h
segments; commit every tau segments within each group plus its final state.
Partial groups receive an endpoint commit. The query is explicitly given
the address of the first commit at or after its fact within its own group.
This oracle separates storage/gradient mechanisms from learned routing; it
is not an end-to-end content-addressed solution. Global keys are independent
zero-mean isotropic unit vectors, independent of the local episodes/writer,
local queries, and selected commit indices. The pilot
must include trainable shared writer parameters to study outer learning.

Primary intervention is full versus branch-detached INTERMEDIATE commits,
with all group endpoints and the continuing recurrence differentiable.
Same seeds/data/initial parameters, same forward computation, same query
loss and normalization. Endpoint-only storage and exact oracle snapshot
reads are controls. Predeclare normalized-MSE target, step budget, seeds,
and evaluation episodes in the runnable configuration before training.
Record nonreaches rather than adjusting the target after seeing results.
If the learned local encoder also produces the local query x, its query-
side gradient remains live in both arms: the intervention removes only
snapshot-storage routes. Global query addresses remain exogenous. State
Jacobian paths terminate at resets and sum only over later commits in the
SAME group; shared writer parameters can receive gradients from all groups.

**Conditional error identity (new derivation):** for a query with local key
x, value y and selected commit r, define z_c=W_c*x, local residual
e=z_r-y, and LA contamination n=sum_(c!=r)(k_r^T*k_c)*z_c. Then samplewise
||prediction-y||^2=||e||^2+||n||^2+2*e^T*n. Conditional on all local
snapshots and x,y,r, random global addresses give E[n]=0 and
E[||n||^2]=(1/D)*sum_(c!=r)||z_c||^2. Cross terms between different
commits vanish because their global keys are independent and zero-mean;
the snapshots themselves need not be independent. Isotropy gives the
variance 1/D. Thus payload correlation alone does not invalidate this law.

Averaging over facts/episodes with one common, positive target-energy
normalizer gives E[NMSE]=delta+A/D, where delta is exact selected-snapshot
NMSE and A is normalized non-target response energy. A counts redundancy
through its actual response energies, not an assumed unit payload norm.
Content-dependent/learned global keys, key-dependent commit selection, or
read normalization require a different calculation with bias/cross terms.
Frozen learned local weights are allowed if fresh test addresses remain
independent. This is an expected-error identity, not a tail guarantee.

For A>0 and delta<epsilon, the minimal integer address size under this
model is D_min=max(1,ceil(A/(epsilon-delta))). If delta>=epsilon and A>0,
no finite D suffices. If A=0, feasibility is simply delta<=epsilon.
This rule holds for a fixed local writer and commit policy; optimizing D
jointly with a retrained writer needs recalibration/predictive modeling.

**An explicit crossover:** take two unit local keys with inner product rho,
two independent zero-mean unit-variance scalar values, eta=1,K=1,h=2, and
zero initial W. Query both facts equally. Endpoint-only tau=2 gives
W2*x1=(1-rho^2)*y1+rho*y2 and W2*x2=y2. Its expected MSE is therefore
(rho^2+rho^4)/2, with zero global interference because there is one commit.
With tau=1 and the first-available-commit oracle, local error is zero;
the two non-target response energies average to (1+rho^4)/2. Hence LA MSE
is (1+rho^4)/(2D). Intermediate commits improve error exactly when
D>(1+rho^4)/(rho^2+rho^4), for rho!=0. Orthogonal local keys (rho=0)
make endpoint-only storage perfect while intermediate LA commits add
noise at any finite D. This demonstrates both favorable and unfavorable
regimes without claiming a universal benefit or better storage efficiency.
The oracle supplies a version address; routing cost/learnability is excluded.

**Cost ledger for this toy model:** P=p*v, C=sum_groups ceil(h_group/tau).
Dense local update work is Theta(J*K*p*v); global commits Theta(C*D*P);
full-payload retrieval for Q queries Theta(Q*(D*P+p*v)). These counts
exclude a learned local encoder, which must be added separately. Folded
state uses D*P scalars plus local P; a bank implementation also stores C*P
payloads and C*D addresses, and training keeps additional graph tensors.
Alternative contraction orders/sparse readers may be cheaper. Count the
actual pilot implementation separately from this streaming ideal.

**Prior-art boundary checked for the milestone:**
[Sparse Attentive Backtracking](https://arxiv.org/abs/1809.03702) already
uses attention shortcuts to past states for temporal credit assignment;
[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
already connects additive associative writes, capacity limits, and delta
updates. [ARMT](https://arxiv.org/abs/2407.04841) and
[GradMem](https://arxiv.org/abs/2603.13875) are required architectural
comparators. This focused source check is not an exhaustive novelty audit.
The intended contribution remains predictive allocation and a controlled
mechanism study, not invention of gradient shortcuts or delta-rule memory.

## Linear milestone results: EXP-LM-001 (2026-11-10; reviewed)

Implementation: [`experiments/linear_memory/`](experiments/linear_memory/).
Report: [`LINEAR_MEMORY_PILOT.md`](reports/LINEAR_MEMORY_PILOT.md).
The isolated harness supports local reset h, commit interval tau, repeated
inner writes, partial groups, exact/oracle and folded-LA reads, and snapshot-
branch detachment. Fifteen focused tests pass, including actual delta-rule
Jacobians and zero cross-reset value gradients.

The predeclared pilot used J12,p4,v3,D16,h6,K1, tau6 or tau2, three seeds,
120 Adam updates, and target NMSE<=0.20 evaluated every20 updates. A learned
position codebook supplies local keys; global keys and routing are exogenous.
Fresh values/global keys vary by episode. This is not learned content routing.

| Reader / storage | Final NMSE, mean over 3 seeds | First observed target step, per seed |
|---|---:|---|
| Exact / endpoint | 0.38213 | none / none / none |
| Exact / intermediate detached | 0.000195 | 40 / 40 / 40 |
| Exact / intermediate full | 0.000532 | 40 / 40 / 40 |
| LA / endpoint | 0.41550 | none / none / none |
| LA / intermediate detached | 0.11526 | 60 / 60 / 60 |
| LA / intermediate full | 0.11497 | 60 / 40 / 40 |

Fixed-parameter detached/full forward differences are exactly zero for all
seeds/readers, while encoder and step-size gradients differ. The intervention
works. The small LA early-target signal is not a general speedup result; exact
reads show no threshold-step benefit at this resolution, and final LA errors
coincide closely. Snapshot availability is the dominant gain over endpoints.
The conditional noise law agrees with measurements, and the two-fact Monte
Carlo crossover check reproduces both success and failure regimes.

At C=2 or6, the raw banks are smaller/cheaper than the D*p*v folded state;
no folding efficiency benefit is established. CPU times are single short
measurements, operation counts are estimates, and training retains raw
banks/autograd graphs. Full cost exclusions are listed in the report.

**Decision:** first executable milestone accepted; cycle1 ends here. Next
test whether delta+A/D selects feasible D and useful p,h,tau allocations on
held-out workloads. Retain branch-detach and exact-reader controls, and do
not infer content-routing or transformer performance from this pilot.

## Held-out allocation milestone specification (2026-11-11)

**Cycle 1 question:** can statistics measured on short isolated reset groups
select useful (p,h,tau,D) allocations for unseen longer contexts at a fixed
expected retrieval error and a fixed folded-state budget? Implementation is
authorized for a bounded local CPU study, EXP-LM-002. A negative predictive
or comparative result completes the milestone if the protocol and evidence
are reproducible; thresholds must not be adjusted after evaluation.

**Transfer law:** calibrate delta (selected-snapshot local error), A0 (sum of
same-group non-target response energies), and B (all responses from one
independent other group) on J=h only. With unit-value normalization,
B=(1/p)*E sum_c ||W_c||_F^2, because a local query from another group is
independent and marginally isotropic. For G=J/h independent equal-size groups,
the predictor is E_hat=delta+[A0+(G-1)*B]/D. The within-group coefficient A0
can include correlated snapshots. The covariance between different global
keys still vanishes under the prior milestone's random-address assumptions.
This law assumes the local writer and group distribution are unchanged;
it does not assert transfer to new geometries or a retrained writer.

**Frozen design:** p={2,4,8}, h={1,2,4,8}, valid unique tau from {1,2,h},
D={4,8,16,32,64,128}, v2,eta1,K1. Local addresses and unit random values are
fresh for each episode. Regimes: independent unit Gaussian directions, or
normalized sqrt(a)*u_group+sqrt(1-a)*e_fact with independent isotropic Gaussian
u,e and a=.7. This a is a mixing coefficient, not the realized dot product.
Both regimes have isotropic marginals and independent reset groups; each is
calibrated separately. This frozen writer replaces the first pilot's learned
position codebook and makes no new gradient-routing or training-speed claim.

Calibration uses 512 isolated groups per candidate and seed. Held-out sets
contain 256 complete episodes at J16/J32; costs are evaluated for Q1/Q32
queries and error targets .10/.25. Seeds 1101/2202/3303. The candidate grid
contains the h=tau=1 independent-write baseline and endpoint/intermediate
commit policies. The primary selector jointly chooses p,h,tau,D. Fixed-policy
comparators constrain p4,h=tau1 or p4,tau=h and select remaining dimensions
from the SAME calibration, without seeing test outcomes.

The selection proxy is J*(3pv+v)+C*D*pv+Q*(Dpv+pv), with folded-state budget
pv+Dpv<=2048 scalars. This is a declared primitive-operation/MAC proxy, not
measured FLOPs. Address generation and oracle discovery costs are excluded;
bank and address metadata are reported separately. All regret comparisons
use the same grid, reader, budget, and cost definition.

**Leakage barrier:** freeze protocol/config, calibrate short groups, then
write and hash all selections/predictions BEFORE generating/evaluating long
test episodes. Test-derived delta/A remain diagnostics, never selector inputs.
Evaluate explicit folded LA with fresh global keys and an additional
conditional-expected-error reference integrating global keys analytically.
The latter has lower variance but uses the same proven identity and is not
an independent test of that identity. Confidence intervals use independent
episodes, not individual correlated queries within an episode.

**Predeclared decision criteria:** at least90% point-feasible unrestricted
selections, median feasible cost regret at most10% against the cheapest
feasible grid reference under the same budget. Report uncertain intervals,
false feasibility decisions, no-selection cases, and infeasible-selection
regret as undefined. These are study criteria, not high-probability guarantees
from the theory, and scenario counts sharing datasets are not independent
replications.

**Potential limitation identified before evaluation:** independent writes
have delta=A0=0 and B=1/p exactly with unit keys/values and eta1. Their error
is (J-1)/(pD). They may dominate this frozen-writer cost grid. If so, successful
selection alone will not demonstrate a benefit from nontrivial h/tau
allocation; retain this baseline and report the collapse instead of changing
the grid. Prediction of full-grid errors remains a separate useful test.

Protocol/config: `experiments/linear_memory/allocation_protocol.md` and
`allocation_config.json`. This specification precedes evaluation; outcomes
and post-result qualifications follow separately.

## Held-out allocation results: EXP-LM-002 (2026-11-11)

The sealed calibration stage produced 7,776 predictions and 144 selections
before evaluation of 1,944 candidate cases. The unrestricted selector met
the target in all 48 scenario rows, with zero cost regret against the cheapest
point-feasible configuration in the registered grid. All 48 also met the
target by the full episode-level 95% conditional-reference interval and by
the sampled-LA point estimate. Selected median/max absolute prediction
errors were .000169/.00165. The rows reuse data across Q and error targets;
they are not 48 independent replications.

Every selected allocation had h=tau=1. The four (J,epsilon)->(p,D) choices
were (16,.10)->(2,128), (16,.25)->(2,32), (32,.10)->(4,128), and
(32,.25)->(2,64). Both p4 restricted policies also chose h1 and became the
same baseline; their median/max cost regret was .0261/.0815. These are
finite-grid comparisons, not global optimality or evidence that hierarchy
dominates a strong one-level memory.

The selected predictions reduce to the exact special case (J-1)/(pD), so
this success provides limited evidence about nontrivial h/tau allocation.
Across the complete grid, median/max absolute prediction error was
.00512/.1095. Explicit sampled LA was consistent with the conditional
reference (1,904/1,944 candidate intervals contained it), but that reference
uses the same global-key identity as the predictor. Transfer of local
moments is assessed on disjoint held-out local episodes; the identity is
not independently validated by comparing it to itself.

**Post-result audit, kept separate from preregistration:** the initial hash
manifest omitted imported `model.py`; the repaired guard includes it. The
original acceptance criterion conditioned on making a selection, which can
reward abstention. The additional audit criterion counts feasible selections
over all scenarios with a feasible grid oracle and rejects empty coverage;
the 90% threshold is unchanged. An interval-certified oracle sensitivity
analysis complements the original point oracle. Original artifacts and the
executed source archive are preserved; the instrumentation rerun uses a
separate `EXP-LM-002-AUDIT` directory and unchanged scientific settings.
These reporting corrections are post-result, not retrospectively registered.

**Audit outcome and verification:** all 30 focused/prior linear-memory tests
passed. Calibration, candidate predictions, and candidate metrics were
byte-identical in the isolated reproduction; selections and all 40 common
evaluation fields were unchanged. Oracle-scenario coverage was 48/48.
All 48 selected configurations were interval-feasible, certified-oracle
median/max regret remained zero, and no cheaper interval-uncertain candidate
was found. These are pointwise approximate intervals, not a simultaneous
high-probability optimality guarantee. See the reproduction comparison in
`reports/results/linear_memory/EXP-LM-002-AUDIT/`.

**Decision:** cycle 1 of this task is complete and the declared held-out
allocation milestone is reached. Stop here under AGENTS.md. Accept the
restricted allocation result and its limitations; the refined-grid study
below is the next proposed milestone, not part of this completed run.

### Post-result analytic prediction and grid limitation

For IID spherical local addresses, independent zero-mean unit values, eta1,
K1, and h=tau=2, write rho=x1^T x2. The first fact's endpoint residual is
-rho^2*y1+rho*y2, while the second fact is exact. Consequently,

    delta = (1/p + 3/[p(p+2)])/2,
    A0 = 0,
    B = (2-1/p)/p,
    E = delta + (J/2-1)*B/D.

This independently audited derivation predicts E=.08943685 for
J16,p16,h=tau2,D16, below the .10 target. Memory is 544 scalars and cost is
1568+4096+544Q: 6208 at Q1 and 23072 at Q32. It beats the original selected
p2,h=tau1,D128 configuration (8932/24928) under the unchanged proxy. Thus
the p<=8 cap can hide candidates that beat the reported grid optimum.

However, the D grid is also restrictive: independent p2,D75 gives exact
expected E=.10, memory 304, and cost 5328 at J16,Q1, cheaper than that grouped
candidate. This is an expectation calculation at the boundary, not a
finite-sample certified result. Neither new candidate was run or included
in EXP-LM-002's oracle/regret. These calculations motivate refining BOTH
p and D before claiming a robust sharing advantage; they do not justify
adding an arbitrary commit penalty to obtain a preferred outcome.

## Refined allocation milestone specification: EXP-LM-003 (2026-11-11)

**Cycle 1 question:** does a shared local writer offer a cost advantage once
both local width and global address resolution are improved? The authorized
bounded CPU study follows `tasks/REFINED_ALLOCATION_TASK.md`. A reproducible
positive or negative answer completes this milestone; transformer integration
and measured acceleration remain outside it.

Keep the unit-key/unit-value eta1,K1,v2 writer, oracle routing, operation proxy,
and 2048-scalar folded-state cap unchanged. Expand p to {2,4,8,16,32,64}; retain
h={1,2,4,8} and unique valid tau from {1,2,h}. Allow EVERY positive integer D
under the memory cap. The independent baseline now optimizes p and D over the
same domain. Compare unrestricted, independent-only, and shared-only h>1
selectors, with deterministic ties and identical resources.

Fresh seeds are 4404,5505,6606. Calibrate each known iid/correlated geometry on
4096 isolated groups; evaluate 1024 held-out episodes per local configuration
at J16/J32, Q1/Q32, epsilon .10/.25. h1 candidates use their exact theoretical
moments; estimated h1 moments remain diagnostics. h>1 predictions use the
short-group empirical moments. Every selector targets .95*epsilon, a fixed
5% reserve chosen before data to reduce boundary sensitivity. Evaluation and
regret use epsilon. The reserve is not a confidence guarantee.

Freeze protocol, config, imported execution dependencies, calibration,
predictions, and selections before held-out evaluation. Conditional-reference
errors for all integer D follow from per-episode delta,A moments. Enumerate
the interval oracle without assuming normal-CI endpoints are monotone.
Explicit sampled LA is limited to unique selected configurations and fixed
witness/sentinel cases, and remains a consistency check. Scenario rows reuse
episodes and do not multiply the number of independent replications.

Prediction acceptance is >=90% feasible-selection coverage among scenarios
with a feasible point oracle, and <=10% median feasible cost regret. No
selection or an empty oracle denominator cannot produce a vacuous pass.
Report interval sensitivity and cheaper uncertain alternatives. Report both
calibrated policies and restricted held-out oracles when comparing sharing
to independence; neither infeasible solutions nor noisy boundary cases
establish a cost advantage. The earlier J16,p16,h2,D16 and independent
J16,p2,D75 analytic examples are fixed checks, separate from selection.

**Pre-execution analytic witness (independently audited):** with IID local
addresses, J32,p32,h=tau2,D12 gives delta=.01700367647, A=.9228515625,
E=.09390797335<.095, and memory832. Its proxy cost is18496+832Q. For ANY
independent h1 memory with p>=2 meeting true epsilon=.1, the identity
(J-1)/(pD)<=epsilon implies pD>=310. Consequently,

    cost = Jv + pv(3J+Q) + (J+Q)Dpv
         >= 20288 + 624Q.

This bound even relaxes integer D and upper limits on p/memory. At Q1 the
shared witness costs19328 versus independent lowerbound20912: at least7.57%
less under the proxy. It beats that bound for Q<1792/208=8.615, including
integer Q<=8. This sufficient condition establishes a restricted theoretical
sharing advantage without relying on a coarse D grid. It does not prove the
shared candidate is optimal or that sharing loses above that Q. The candidate
is a fixed held-out/sampled-LA check in EXP003, selected before new data.

Pre-execution audit also adds slack-matched oracle/coverage diagnostics to
separate the reserve's cost from prediction error. Primary criteria still
use the original epsilon. Tiny negative A from roundoff must be guarded;
integer boundary choices must be checked explicitly.

**Pre-execution release:** 40 focused and prior linear-memory tests passed.
Independent theory and implementation audits found no scientific blocker;
the PI released the frozen calibration/selection and held-out stages. This
specification and the fixed witness precede EXP-LM-003 outcomes. The runner
guards dependency drift and refuses to overwrite completed evaluations.

## Refined allocation results: EXP-LM-003 (2026-11-11)

The predeclared primary criteria passed: all 48 unrestricted scenario rows
were feasible at epsilon, with median cost regret .04808 and maximum .10385
against the point-feasible held-out oracle. The criterion concerned median,
not maximum regret. Independent-only and shared-only policies also had 48/48
feasible selections. The original frozen selection artifact has SHA-256
`d97868b12e8ed1a6082f905c76bc373d5933e092b124c6ae143671a3aefec500`.
Calibration took about 5.4s and evaluation 82.8s on CPU; these are experiment
execution times, not benchmarks of deployed memory architectures.

Sharing won in 9 of 48 dependent reporting views, identically categorized
by the calibrated-policy and restricted-oracle comparisons. The substantive
gains occur for IID addresses, J32,Q1, in all three seeds:

| Target error | Independent (p,h,tau,D) | Shared (p,h,tau,D) | Independent/shared cost | Independent/shared memory scalars |
|---|---|---|---|---|
| .10 | (2,1,1,164) | (32,4,4,20 or21) | 22100 / 17792 or18368 | 660 / 1344 or1408 |
| .25 | (2,1,1,66) | (16,4,4,13) | 9164 / 6912 | 268 / 448 |

The reductions are 16.9–19.5% and 24.6%, respectively. These are proxy-compute
gains under a common memory CAP, with more actual memory used by the shared
writer; they are not Pareto improvements in both compute and storage. The
remaining three nominal wins are J16,Q1,epsilon.25 with cost 2400 versus 2404,
only .17% and unsuitable as a practical headline. Independent writing wins
every Q32 comparison and all comparisons in the correlated family.

The fixed, analytically audited two-fact witness at J32,p32,h=tau2,D12 was
confirmed on fresh samples. Its three conditional-reference errors were
.0936774,.0936513,.0941990, versus prediction .0939080. All three pointwise
95% intervals contain the prediction and lie below .10. Sampled-LA means
were .0933891,.0933201,.0947618. The witness's proof of cost19328 below the
continuous independent lowerbound20912 is separate from the empirically
selected h4 configurations, and does not rely on a discrete-D oracle.

**Uncertainty and scope:** finer D puts many feasible-oracle choices near
the boundary: 41/48 unrestricted views have a cheaper interval-uncertain
alternative to the interval-feasible oracle. Thus exact rankings are not
statistically certified. At the stricter .95*epsilon diagnostic target,
unrestricted coverage is 40/48; at the actual epsilon it is 48/48. This
diagnostic boundary sensitivity must not be hidden by the primary pass.
Full-grid moment-transfer error grows with h, and sampled LA remains a
consistency check using the same conditional-reference identity.

The correlated generator's latent blocks coincide with reset groups, so
changing h also changes the joint address distribution. Those results
describe a reset-conditioned family, not an intervention on one fixed
correlated fact stream. IID comparisons share the same population and
support the principal gain. Exogenous fixed correlation blocks would require
a revised transfer analysis when dependence crosses reset boundaries.

This study does not compare against enlarged local-only memory, layerwise
tokens, LoRA, or learned content routing, and establishes no convergence or
wall-clock advantage. Those remain necessary for the broader contribution.

**Verification and decision:** all 40 focused/prior linear tests passed.
The run produced 324 calibration rows, 2592 predicted frontiers, 648 held-out
local moment rows, 144 policy views, and 72 unique sampled-LA checks. All
selected conditional-reference intervals and sampled-LA point means were
feasible at the original target. The final dependency hashes match those
sealed before evaluation. Independent audits covered theory, implementation,
and interpretation. Cycle 1 is complete and this milestone is reached; stop
here under AGENTS.md. Accept the restricted IID compute/storage crossover
and retain the enlarged local-only comparison as the next proposed milestone.

## Local-only comparison specification: EXP-LM-004 (2026-11-11)

**Cycle 1 question:** when does hierarchy reduce the operation proxy relative
to an enlarged local-only associative memory under the same expected-error
requirement and persistent-state cap? The scientist brief is
`tasks/LOCAL_ONLY_COMPARISON_TASK.md`. Compare both carried delta-rule memory
and direct additive memory; the second control separates a hierarchy benefit
from the cost of a particular local update rule. A negative or mixed outcome
is a valid completed milestone.

### Exact IID theory, independently audited before execution

For a v-by-p delta memory with eta=K=1, unit spherical IID keys and independent
zero-mean unit values, let a_n=E||W_n||_F^2 and let R_n be the sum, over the n
stored facts, of E||W_n*x_j||^2. Define lambda=1-1/p and
beta=1-2/p+2/[p(p+2)]. Then

    a_0=R_0=0,
    a_n=lambda*a_(n-1)+1,
    R_n=beta*R_(n-1)+(n-1)*(a_(n-1)/[p(p+2)]+1/p)+1,
    delta_n(p)=(R_n-2*a_n+n)/n.

The first recurrence follows because the retained and newly written matrix
components have orthogonal right factors. The second follows from the
spherical fourth moment; the new fact is read exactly. Summed target-response
alignment obeys the same recurrence as a_n, giving the final error formula.
At p1, delta_n=2(n-1)/n (complete replacement); at n2 it reduces to the
previous .5*(1/p+3/[p(p+2)]) result.

For a snapshot at time c and a fact j, response energy e_cj is a_c/p before
the fact arrives, equals1 at c=j, and subsequently obeys

    e_cj=beta*e_(c-1,j)+a_(c-1)/[p(p+2)]+1/p.

Its target alignment after arrival is lambda^(c-j). Consequently, delta,
same-group distractor energy A0, and cross-group energy B for every h,tau
schedule can all be computed exactly using the existing routing rule. This
extends delta+[A0+(J/h-1)B]/D beyond the earlier h1/h2 special cases. No
empirical calibration is required in this IID study.

An additional independently audited fixed-J, large-p expansion is

    delta_J(p)=(J-1)/(2p)+(J-1)(2J+5)/(6p^2)+O(p^-3).

Consequently, as epsilon tends to zero with continuous widths and no active
cap, p_delta=(J-1)/(2epsilon)+(2J+5)/3+O(epsilon), versus
p_additive=(J-1)/epsilon. Delta writing corrects the current fact exactly;
to leading order only subsequent writes interfere with that fact, whereas
additive writing leaves interference from both earlier and later facts.
This explains the leading factor-of-two capacity advantage of local delta
memory. Its more expensive writes counterbalance that smaller state: under
the proxy the leading local-delta/additive cost crossover is Q=J. Finite-error
width corrections shift the continuous crossover upward; integer allocation
and a binding cap need the exact calculation. This asymptotic is explanatory
theory, not an extra evaluated workload or a universal memory-capacity claim.

### Frozen comparison and validation design

All methods use v2, J={16,32}, Q={1,32}, epsilon={.10,.25}, and persistent
state cap2048 scalars. The primary frontier is the exact population optimum
at epsilon. An additional .95*epsilon selection for every method is a shared
robustness sensitivity, not protection against calibration error.

| Method | Eligible dimensions | Persistent state | Operation proxy |
|---|---|---|---|
| Carried local delta | every integer p1..1024 | pv | J(3pv+v)+Qpv |
| Local additive W+=yx^T | every integer p1..1024 | pv | (J+Q)pv |
| Delta plus global LA | every integer p,D>=1 with pv(1+D)<=2048; all divisor h of J and divisor tau of h | pv(1+D) | J(3pv+v)+(J/tau)Dpv+Q(Dpv+pv) |

The additive error is exactly (J-1)/p. All comparisons use the same existing
mixed-operation/MAC convention, report its components, and assign no fictitious
LA overhead to local-only methods. No result is claimed at equal actual state
use merely because the cap is shared. The hierarchy receives an oracle global
route in addition to its local address; address construction and routing
metadata costs remain excluded. Varying p changes the random representation,
so the comparison is for an IID address family, not arbitrary fixed inputs.

Seal exact candidates, optima and reserve selections, config, protocol, and
imported code before held-out generation. Fresh seeds7707,8808,9909 use1024
episodes. Evaluate unique selected allocations at both targets, immediate
local-width or hierarchical-D boundary neighbors, and fixed known/p1 checks.
For the same p,J, pair the flat IID key/value stream across all methods and
reset horizons before grouping. Different widths may use separate recorded
random streams. Omit the reset-conditioned correlated family.

Acceptance for empirical deployment feasibility is >=90% coverage at epsilon
for each method's reserve selections when its population oracle is feasible;
empty denominators cannot pass. Primary epsilon choices may lie on a true
population boundary, so sample means above that boundary are not automatically
model failures. Report theory/measurement residuals, pointwise intervals,
reserve sensitivity, and direct sampled-LA consistency without using held-out
data to replace the analytic oracle or retune choices. Only the evaluated
subset receives empirical validation; no full-grid simulation is claimed.

**Status:** completed after independent theory/core/runner audits and PI
release. The frozen design above was not changed after held-out generation.

## Local-only comparison results: EXP-LM-004 (2026-11-11)

The exact bounded integer comparison has a mixed result. Costs below are the
registered operation proxy, optimized separately at the same population-error
target and 2048-scalar cap. The winner is the overall minimum;
actual storage and achieved errors differ across methods.

| J | Q | epsilon | Local delta cost | Local additive cost | Hierarchical cost | Winner |
|---|---|---|---|---|---|---|
| 16 | 1 | .10 | 8362 | 5100 | 5230 | Additive |
| 16 | 1 | .25 | 3658 | 2040 | 2170 | Additive |
| 16 | 32 | .10 | 13632 | 14400 | 14592 | Delta |
| 16 | 32 | .25 | 5952 | 5760 | 5952 | Additive |
| 32 | 1 | .10 | 33432 | 20460 | 16054 | Hierarchy |
| 32 | 1 | .25 | 14420 | 8184 | 6624 | Hierarchy |
| 32 | 32 | .10 | 44096 | 39680 | 40000 | Additive |
| 32 | 32 | .25 | 19008 | 15872 | 16192 | Additive |

At J32,Q1, hierarchy uses h=tau=4,D=12, with p39 at epsilon .10
and p16 at epsilon .25. Its population errors are .09990219 and .24653972.
Against the stronger local-only comparator (additive), proxy costs fall by
21.53% and 19.06%, while state grows from 620 to 1014 scalars and from 248
to 416. Local delta uses even less state (344 and 148) but costs more. This
is a compute/storage tradeoff within a common cap, not Pareto dominance.

All other hierarchical optima use p1,h=tau=1 and are more expensive than
direct additive memory. Local delta wins J16,Q32,epsilon .10: its p85
state uses 170 scalars and costs 13632, versus additive p150,300 scalars,
cost14400. The asymptotic capacity advantage of delta writing therefore
does not by itself imply a compute advantage: query count, finite-error
corrections, and integer widths matter. No selected hierarchy uses an
intermediate commit tau<h; this comparison adds no evidence of an
optimization benefit from intermediate anchors.

**Verification:** 180224 analytic frontier rows yield 64 selection rows;
59 unique allocations (including frozen neighbors/fixed checks) were evaluated
on three fresh seeds with 1024 episodes each. The 177 seed/configuration rows
have median absolute prediction residual .00033276 and maximum .00772436.
All three methods achieve 24/24 deployment-epsilon feasible reserve views;
hierarchy also achieves 24/24 using direct sampled LA. All 24 reference
reserve intervals per method lie below deployment epsilon. These are dependent
views, not 24 independent replications. The 5% reserve costs a median
5.29%/5.56%/5.35% extra for delta/additive/hierarchy, respectively.

Primary choices at the exact epsilon boundary are only point-feasible in
14/24 delta, 12/24 additive, and 13/24 hierarchical views. This is compatible
with finite-sample fluctuation around a population boundary, and is reported
separately from reserve acceptance. No outcomes were used to revise selections.
The hierarchy reference averages over global keys conditionally; direct
sampled LA is an additional check, with median/max absolute differences
.00084533/.00468345 over 72 rows. Pointwise intervals are not simultaneous
guarantees. Only the registered subset, not the full analytic frontier, was
simulated.

The final independent evidence audit confirmed all table and coverage
aggregates. Raw theory-in-interval coverage is 137/177; 36 exclusions are
deterministic p1,h1 reference roundoff at about 1e-17. With an explicitly
reported 1e-15 comparison tolerance, coverage is 173/177. The four material
pointwise misses and the numerical diagnostic are retained in the report;
none changes the frozen population selections or acceptance rule.

**Reproducibility:** 76 focused/prior tests pass. Prediction/selection took
18.15s internally and evaluation 26.42s on CPU. These are study execution
times, not method performance benchmarks. The selection artifact hash is
`d69b256e487807cffe3f9326262bc025ed5e40fef0b0e8212bee211b7d6029f3`.
Source/config/protocol/imported dependencies were sealed before evaluation;
previous experiment sources and artifacts were preserved. Report:
`reports/LOCAL_ONLY_COMPARISON_RESULTS.md`; artifacts:
`reports/results/linear_memory/EXP-LM-004/`.

**Scope and decision:** accept the restricted IID low-query hierarchy win
against both enlarged one-level comparators, along with its larger state and
the six local-only wins. Changing p changes the supplied random address
representation; hierarchy additionally receives oracle global routing.
Address construction, metadata, temporary buffers, and learned encoders are
outside the cost/state proxy. The result therefore makes no universal,
transformer, LoRA, or wall-clock claim. Cycle 1/5 reaches the declared
milestone; stop here. Measured costs and competitive transformer baselines
are the next proposed work, not runs launched by this study.

## Hypotheses ledger

| hypothesis | status | strongest support | strongest doubt / missing |
|---|---|---|---|
| **H1** independently written per-segment memories + attention retrieval beat carried memory on forgetting | **supported** (KV single-key) | fact 2: all pairs stored with 2 tokens / 8 pairs / 4 segments; clean forgetting matrices | not yet shown at large segment counts (N ≥ 64 is confounded by read superposition); carried-memory control now tabulated on the identical curriculum (2026-11-09): 0.82/0.48/0.15 vs interface 0.99/0.99/0.93 at N=8/16/32 (budget unmatched by design: 2 carried tokens vs N stored); its forgetting matrices are still on the remote only |
| **H2** the concat failure is a **read-interface** bottleneck, not a write-quality one | **supported** (ladder, 2026-10-22) | facts 5–7; the ran control: bank-prefix fails at 0.024 WHILE (c)/(d)/(d′) reach 1.00/0.997/0.999 on the SAME stored tokens — the prefix interface, not write quality, is indicted (fact 8) | prefix-interface failure is established only at this training scale (4L/128d from scratch); a pretrained/larger frozen reader might yet make the prefix work |
| **H3** per-token, per-level linear-attention reads from a shared state (`S = Σkᵢvᵢᵀ`), added as a bias to softmax attention over the query context, fix routing + payload + iteration at O(1) read cost | **supported for routing+payload** (ladder 2026-10-22: 0.995–1.00 vs 0.25/0.02 controls; single-key 0.997+); implementation STATUS.md 2026-10-18 (b)/(c) | facts 8–9; interface variant (d′) matches all-position reads at M·L sites — cheaper architecture suffices given pass-1 conditioning | the **iteration** claim (pointer chasing, depth-bounded multi-hop) is untested; **O(1)-read capacity is bounded**: at d_h=32, superposition binds at N=32 on single-key (Δ(32)=0.065 converged; Δ(64)=0.58 — EM 0.41 vs control 0.99 at identical protocol; control ≥ 0.99 through N=96; [report](reports/INTERFERENCE_CURVE_RESULTS.md)). O(1) reads hold only while N ≲ d_h per head (capacity hypothesis, d_h sweep pending) |

## Missing experiments (decision-critical, in priority order)

1. **Interference curve vs N** — **DECIDED (2026-11-12,
   [`reports/INTERFERENCE_CURVE_RESULTS.md`](reports/INTERFERENCE_CURVE_RESULTS.md)):
   superposition binds at N=32; total collapse at N=64.** After the
   2026-11-03 double infrastructure failure, the softmax control was resumed
   (job `960c43c4`, manifest
   [`kv_interference_softmax_resume.yaml`](manifests/kv_interference_softmax_resume.yaml))
   and completed the full curriculum: EM ≥ 0.99 at every phase through N=96.
   Δ(N) = 0.0005 / 0.0025 / 0.065 / 0.58 at N=8/16/32/64. Still missing and
   judged low-information: interface N96 (~10+ days, Δ(96) stays undefined),
   adaptive N64/N96. The adaptive arm's forgetting matrices are in the remote
   archive only. Protocol per user: single-key KV, checkpoint-chained
   curriculum N ∈ {8,16,32,64,96}, 50k steps/phase, 0.99-EM early stop; arms
   `interface` + `interface_softmax` (user-selected). `read_state_cross_talk`
   rose 0.127 → 0.143 and saturated at the random-key |cos| for d_h=32.
   Superposition binds, so the planned mitigation applies: coarse top-k
   segment routing (reuse the compressed-query scores) before building the
   state. The direct capacity test is d_h = 64 (#2 arm2) on this curriculum.
2. **Interface read: kernel × key-geometry cells** —
   [`manifests/kv_interface_softmax.yaml`](manifests/kv_interface_softmax.yaml):
   arm1 softmax kernel (config-only), arm2 key width 64 = the old hopfield
   dim (`read_state_heads: 2`, config-only), arm3 **chunk-granularity keys**
   (one key per stored memory chunk, `W_seg·M_grad`, no per-token addressing
   — the "chunk as whole with a key" storage model; needs the small task in
   [`tasks/SEGMENT_KEY_TASK.md`](tasks/SEGMENT_KEY_TASK.md)). Chain of
   single-factor deltas vs (d′): arm1 isolates the kernel, arm2−arm1 the key
   width, arm3−arm2 the chunk-vs-token granularity. Note: the linear mode
   already skips `_compress_segment` — keys already derive from `M_grad`;
   arm3 changes only granularity. Pre-registered predictions in the manifest
   header; arm3's outcome is open (chunk addressing ⇒ simpler storage;
   token-level routing might be load-bearing).
3. **Pointer chasing / associative join** (`!K1:K2! … !K2:V!`) — the
   depth-bounded-iteration claim is H3's remaining untested leg (fact 7,
   failure 3); dataset noted 2026-09-10 (`irodkin/ar_multihop`). Compare
   (d)/(d′) vs (a).
4. **Writer×read factorial** — forward writer (RMT-style, `memory_update=
   "forward"`, the IDEAS.md baseline) behind the winning read (d′): does a
   decoder-informed forward writer + linear-state read beat the gradient
   writer? After the read settles.
5. **babilong sanity** of (d′) (reference point: ARMT worked on babilong).
6. **Concat-format forgetting probe** — the existing per-segment probes use
   single-key `?!K:V!|` queries, which concat-trained arms never see (so the
   current matrices measure format transfer, not forgetting — report caveat);
   add two-key probes restricted to earlier segments.
7. **Retrieval-mode ladder on concat** (`mean`/`softmax`/`beta_softmax`/`raw`):
   config-only documentation that no one-shot mixture setting solves it
   (low priority now that fact 8 establishes the mechanism directly).
8. **Scale calibration at larger models/data** — everything above ran on
   4L/128d from scratch; before any deployment claim, replicate (d′) at a
   larger width/depth and on curriculum N.

## Map

- Working files (repo root): `README.md` (repo & harness overview) · this file
  · `STATUS.md` · `IDEAS.md` · `AGENTS.md` (agent conventions).
- `docs/` — guides, derivations, designs only:
  [`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md) (the H3 read-interface
  spec) · [`docs/ADAPTIVE_MEMORY.md`](docs/ADAPTIVE_MEMORY.md) (gated
  carried-memory updates) · [`docs/STAR.md`](docs/STAR.md) (worst-case
  stability regularizer) · [`docs/CURRICULUM.md`](docs/CURRICULUM.md)
  (stage-wise N curriculum) · [`docs/REMOTE.md`](docs/REMOTE.md) (cluster
  workflow).
- `tasks/` — executor briefs: [`tasks/TOKENWISE_READ_TASK.md`](
  tasks/TOKENWISE_READ_TASK.md) (H3 implementation) ·
  [`tasks/INTERFACE_CONDITIONING_TASK.md`](tasks/INTERFACE_CONDITIONING_TASK.md)
  (query-blind baseline + submission) · [`tasks/SEGMENT_KEY_TASK.md`](
  tasks/SEGMENT_KEY_TASK.md) (chunk-granularity keys, pending).
- `reports/` — results & evidence: [`reports/READ_LADDER_RESULTS.md`](
  reports/READ_LADDER_RESULTS.md) · [`reports/INTERFERENCE_CURVE_RESULTS.md`](
  reports/INTERFERENCE_CURVE_RESULTS.md) (+ `reports/figures/`,
  `reports/results/`).
- Models: [`grad_memgpt.py`](grad_memgpt.py) (base gradmem + Hopfield +
  Gated-Delta externals; pair-aware segmentation helpers) ·
  [`grad_memgpt_adaptive.py`](grad_memgpt_adaptive.py) (carried memory, gated
  updates, STAR) · [`rmt.py`](rmt.py).
- Data: [`kv_dataset_utils.py`](kv_dataset_utils.py) (`generate_sequence`,
  `query_mode='ordered_pair_concat'`) · `notebooks/dump_dataset.ipynb` ·
  `scripts/download_kv_retrieval.sh`, `scripts/download_babi.sh`.
- Configs: [`configs/gradmemgpt/kv_retrieval/`](configs/gradmemgpt/kv_retrieval)
  (`hopfield.yaml`, `hopfield_value_activation.yaml`, `gated_delta.yaml`,
  `adaptive_*.yaml`, …).
- Manifests (scheduler grids): [`manifests/`](manifests) (`kv_hopfield.yaml`,
  `stage0_capacity_vs_interference.yaml`, …; schema in
  `run_scheduler.py` docstring).
- Run artifacts: `runs/`; cluster submission workflow in
  [`docs/REMOTE.md`](docs/REMOTE.md).
