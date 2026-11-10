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
vs N has partial results** (2026-11-09,
[`reports/INTERFERENCE_CURVE_RESULTS.md`](reports/INTERFERENCE_CURVE_RESULTS.md)).
Both jobs died on 2026-11-03 (an interrupt, then a full disk), so no arm
reached N=96 and the control is truncated from N=32 on. On single-key with
the deployable read, **superposition binds at N=32**: Δ against exact
softmax at the same interface sites is 0.0005 / 0.0025 / 0.052 at
N=8/16/32, and the interface read plateaus at EM 0.41 at N=64. This matches
the random-key estimate √((N−1)/d_h) with key width d_h = 32 (a hypothesis;
d_h sweep pending). The carried-memory baseline (same curriculum, every
segment writing the SAME M=2 tokens;
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

## Contribution roadmap (2026-11-09; research plan, execution not started)

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
This roadmap does not launch jobs or supersede the historical results above.

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

Use W_j=W_(j-1)+eta*(v_j-W_(j-1)x_j)*x_j^T and explicitly specified local
key/value distributions. Encode whole W snapshots with global keys in D
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

**Immediate next milestone:** Stage 0's specification plus Stage 1's linear
error/gradient model and a bounded Stage 2 experiment brief. No large sweep
is needed to decide whether to proceed. Implementation and runs follow the
PI/research-scientist workflow in AGENTS.md when execution is undertaken.

## Hypotheses ledger

| hypothesis | status | strongest support | strongest doubt / missing |
|---|---|---|---|
| **H1** independently written per-segment memories + attention retrieval beat carried memory on forgetting | **supported** (KV single-key) | fact 2: all pairs stored with 2 tokens / 8 pairs / 4 segments; clean forgetting matrices | not yet shown at large segment counts (N ≥ 64 is confounded by read superposition); carried-memory control now tabulated on the identical curriculum (2026-11-09): 0.82/0.48/0.15 vs interface 0.99/0.99/0.93 at N=8/16/32 (budget unmatched by design: 2 carried tokens vs N stored); its forgetting matrices are still on the remote only |
| **H2** the concat failure is a **read-interface** bottleneck, not a write-quality one | **supported** (ladder, 2026-10-22) | facts 5–7; the ran control: bank-prefix fails at 0.024 WHILE (c)/(d)/(d′) reach 1.00/0.997/0.999 on the SAME stored tokens — the prefix interface, not write quality, is indicted (fact 8) | prefix-interface failure is established only at this training scale (4L/128d from scratch); a pretrained/larger frozen reader might yet make the prefix work |
| **H3** per-token, per-level linear-attention reads from a shared state (`S = Σkᵢvᵢᵀ`), added as a bias to softmax attention over the query context, fix routing + payload + iteration at O(1) read cost | **supported for routing+payload** (ladder 2026-10-22: 0.995–1.00 vs 0.25/0.02 controls; single-key 0.997+); implementation STATUS.md 2026-10-18 (b)/(c) | facts 8–9; interface variant (d′) matches all-position reads at M·L sites — cheaper architecture suffices given pass-1 conditioning | the **iteration** claim (pointer chasing, depth-bounded multi-hop) is untested; **O(1)-read capacity is bounded**: at d_h=32, superposition binds at N=32 on single-key (Δ(32)=0.052; EM 0.41 at N=64; [report](reports/INTERFERENCE_CURVE_RESULTS.md)). O(1) reads hold only while N ≲ d_h per head (capacity hypothesis, d_h sweep pending) |

## Missing experiments (decision-critical, in priority order)

1. **Interference curve vs N** — **PARTIAL RESULTS (2026-11-09,
   [`reports/INTERFERENCE_CURVE_RESULTS.md`](reports/INTERFERENCE_CURVE_RESULTS.md)):
   superposition binds at N=32.** Both jobs (`ccde14ed`, `7c2ed873`) died on
   2026-11-03 (an interrupt, then a full disk). Still missing: the softmax
   control at N ≥ 32 (P1, Δ(64)), N96 for all arms, and the adaptive arm's
   forgetting matrices (remote only). Manifest:
   [`kv_interference_curve.yaml`](manifests/kv_interference_curve.yaml).
   Protocol per user: single-key KV, checkpoint-chained curriculum
   N ∈ {8,16,32,64,96}, 50k steps/phase, 0.99-EM early stop; arms
   `interface` + `interface_softmax` (user-selected: hopfield dropped,
   linear-all deferred — definitions kept in the chain script for a later
   manifest). Revision-2 predictions P1–P3 in the manifest header.
   `read_state_cross_talk` rose 0.127 → 0.143 and saturated at the
   random-key |cos| for d_h=32. Superposition binds, so the planned
   mitigation applies: coarse top-k segment routing (reuse the
   compressed-query scores) before building the state. The direct capacity
   test is d_h = 64 (#2 arm2) on this curriculum.
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
