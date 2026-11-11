# IDEAS — pool of candidate directions and draft hypotheses

Not a plan; nothing here is committed or pre-registered. When an idea graduates
to "will run", it moves into `PROGRESS.md` (missing experiments) and gets a
config/manifest; decisions get journaled in `STATUS.md`. Status tags:
**draft** (rough), **specified** (geometry/goal concrete), **gated** (waiting
on another result).

## Candidate scientific contribution (2026-11-09; draft, not authorized runs)

The user-requested staged roadmap is now in
[`PROGRESS.md`](PROGRESS.md#contribution-roadmap-2026-11-09-implementation-started-2026-11-10).
The hypotheses below remain unproven; the roadmap specifies decision gates.
The first linear implementation/pilot milestone is now reviewed in
[`LINEAR_MEMORY_PILOT.md`](reports/LINEAR_MEMORY_PILOT.md); its limited
gradient-routing signal does not establish the broader contribution.

- **Optimize locally, store globally: fixed-error memory allocation.** Main
  question: given N facts, an error tolerance, and a query workload, when
  should compute/state be spent on the gradient writer's local capacity p
  versus the global address D? Predict the preferred allocation from local
  retained-fact count b, payload size P, finite-step distortion, measured
  payload-noise sensitivity, writer cost, and routing/read costs. Validate
  predictions on held-out context lengths and at least two local writable
  parameterizations. Include failure regimes and a competitive one-level
  deep-prefix/fast-LoRA baseline. *draft.*
- **Mechanistic supporting claim: independent writes shorten memory's
  optimization dependency path.** Measure acquisition/retention with frozen
  outer weights and outer training compute to target error. The existing
  report motivates this but does not identify it causally; exact bank reads
  train faster than folded LA, and the tiny carried baseline changes both
  capacity and reader. Start with a same-budget three-arm shared+exact,
  disjoint+exact, disjoint+LA comparison; fourth shared+LA arm if needed.
  This tests independence separately from read superposition. *draft.*

## Segmented shared writes inside local storage (2026-11-09; baseline requirement)

The local writer need not encode its entire input jointly or reset after
every smallest segment. Add a local carry horizon h: divide J elementary
segments into T=ceil(J/h) groups, initialize a fresh local memory for each
group, sequentially apply K gradient steps per elementary segment to the
SAME local state, and commit only the group's final state to the outer
store. Intermediate local states are not separately stored in this baseline.
The final group may contain fewer than h segments.

For group s, m_(s,0)=m0 and
m_(s,j)=F_K(x_(s,j);m_(s,j-1)), followed by outer commit G(m_(s,h_s)).
Local state may be input tokens, deep prefixes, or fast adapter parameters.
Reset the local state at the next group boundary; optimizer-buffer carry
and graph truncation must be specified separately from numerical state carry.

- h=1: independent elementary-segment writes.
- 1<h<J: shared sequential writes within local groups, independent groups
  accumulated globally.
- h=J: fully carried WRITE dynamics; one final group is committed. The READ
  can still differ from the original adaptive prefix baseline.

This makes the reset/commit interval a fourth allocation variable alongside
p,D,K. For whole-local-memory payloads, the working small-noise model is
E approximately delta_local(p,K,h)+Gamma(p,K,h)*(ceil(J/h)-1)/D.
Here delta_local includes acquisition AND within-group forgetting, and Gamma
is an MSE amplification coefficient. Neither is assumed monotone in h.
For token/component-wise commits, account for e(p) entries per group rather
than applying the one-payload formula unchanged.

The longest local numerical write dependency is h*K inner steps, rather
than J*K; derivative dependencies agree only when not truncated/detached.
At fixed J and K per elementary segment, there are still J*K inner steps
overall. Increasing h reduces the NUMBER of global commits, not this inner
step count. A longer h trades reduced outer interference/commit work against
local interference, harder credit assignment, and potentially larger graph
memory. It can be useful even without increasing local state size p.

Required comparisons: (i) sweep h at fixed elementary segmentation and
local parameterization; (ii) compare sequential subsegment writing against
joint writing of the SAME group's concatenated context, with the same local
state and outer reader. Report actual FLOPs/time and state/activation memory:
K steps per subsegment and K steps over an entire group are not compute-
matched. Use matched-budget fronts and target-format local retention probes;
also optimize D when studying capacity allocation at fixed retrieval error.
No implementation or run is authorized by this baseline note.

## Intermediate commits of shared memory (2026-11-09; draft)

Decouple the local reset/carry horizon h from a commit interval tau. Within
each group, continue the same local trajectory m_j=F_j(m_(j-1)), but commit
selected intermediate states as well as the final state. Endpoint-only
storage above remains a required baseline (tau=h for full groups). With
tau=1 and h=J, every segment endpoint is stored while the writer carries
its local state through the entire sequence. Intermediate inner-step
iterates are another possible granularity; start with segment endpoints.

Two distinct mechanisms: historical states can retain information lost
later in the local trajectory; differentiable commits provide shorter
paths from the final answer loss to earlier writes. Neither mechanism
guarantees improved error or convergence. A stored snapshot does not constrain
later local updates, so "anchor" means a readout/credit-assignment access
point, not a stability regularizer. This targets OUTER training; the inner
objective is unchanged unless global storage is explicitly fed back to it.

For fixed commit coefficients a_c, let S=sum_c a_c A(m_c), with A the
complete key/value encoding into a common associative state, L=ell(R(S,q)),
J_t=dF_t/dm_(t-1), and b_c=a_c DA(m_c)^T grad_S L (vectorizing S).
Then, in the untruncated graph with no other routes from m_j to the loss,
grad_(m_j) L=sum_(c>=j) (J_c ... J_(j+1))^T b_c. The c=j term has an
identity empty product and bypasses all subsequent local-update Jacobians.
State-dependent gates belong inside A and must be differentiated too.
This is a path decomposition, not a lower bound on gradient magnitude:
read routing may suppress a branch and contributions can cancel.

Linear associative example: W_j=W_(j-1)+eta*(v_j-W_(j-1)x_j)x_j^T.
The homogeneous perturbation obeys dW_j=dW_(j-1)*(I-eta*x_j*x_j^T).
A read of a committed W_j can send gradient to it without traversing the
later right-multiplications. A whole-operator commit is
S=sum_c k_c tensor vec(W_c), with query retrieving sum_c (q^T k_c)W_c.
This assumes the whole-payload architecture, not the current token-wise
implementation; key/value encoding derivatives apply in the latter too.

If tau divides h and J, there are J/tau commits and any segment endpoint
has a commit at most tau segments ahead. A shortest available readout path
therefore traverses at most tau*K subsequent inner updates; the longest
local dependency remains h*K. This does not automatically reduce autograd
activation memory or backpropagation cost. At fixed outer dimensions D,P,
the folded state is still D*P scalars, while dense commit work grows as
(J/tau)*D*P. Retained raw banks and training graphs need separate accounting.
Smaller tau increases correlated contributions, duplication, and possible
global interference; independent-item MSE=(T-1)/D cannot be applied without
checking its assumptions. At fixed retrieval error D may need to increase.

Mechanism control: compare (1) endpoint-only commits, (2) intermediate
commits A(stopgrad(m_c)) but a differentiable final commit, and (3) the
identical intermediate commits A(m_c). In (2), detach ONLY the snapshot
branch, leaving the live carried recurrence unchanged and K/V projections
trainable. At identical parameters (2)/(3) have identical forward values
and storage; their difference isolates the extra gradient routes into the
writer. Their training trajectories will naturally diverge. Compare LA
against exact attention over the same snapshots to separate routing from
superposition. Report target-error training FLOPs/time, activation memory,
retention by age, and gradient contributions/alignment by commit age.

Potential contribution: predict the preferred commit interval jointly with
p,D,K,h, balancing credit-assignment distance against global interference
and compute. Attention-based shortcuts to past recurrent states are already
established (Sparse Attentive Backtracking, NeurIPS 2018:
https://arxiv.org/abs/1809.03702); novelty would require the gradient-written
local state, compressed global storage, and a validated allocation theory,
not the shortcut idea alone. Hypothesis only; no implementation or runs.

## Read architectures (the active front)

- **Linear-state per-level read (H3, the current proposal)** — per-head states
  `S_h = Σᵢ kᵢₕ vᵢₕᵀ` built once from the stored memory tokens; per-layer
  L2-normalized queries with learnable β; Flamingo-style per-layer gate;
  parallel additive injection `h = x + attn(x) + g·qS`; no memory tokens in the
  read. Precedent: Titan's long-term-memory read, Based's
  linear+softmax hybrid. **Full spec with design decisions, cost accounting,
  config surface, and the experimental ladder:
  [`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md).** *specified; module not
  written.*
  - Scale-calibration sub-decisions pinned in discussion: normalized queries
    (cosine semantics — same lesson as `stabilize_energy_head`), gates
    small-but-nonzero init (zero-init may never escape), per-layer query
    projections but shared K/V (state built once).
- **Interface-token reads (the middle rung)** — the same linear state, but
  only M learnable interface tokens (separate M_0, two-pass layout) read it
  per layer; everything else reaches memory via softmax attention to those
  tokens. Resampler pattern (Flamingo/Q-Former). Decides whether
  cross-segment integration needs per-token reads or only per-layer reads
  through a small interface; the placement/causality trap (query-blind
  left-prefix reads) and its two-pass resolution are specified in the spec's
  Variant section. *specified ([docs/TOKENWISE_READ.md](docs/TOKENWISE_READ.md)
  § Variant); module not written.*
- **Bank-prefix read (diagnostic control)** — prepend all stored per-segment
  memory tokens at READ, let the backbone's own attention retrieve. Recall
  upper bound; context grows linearly in n_stored·M, so never the deployable
  design, but the cleanest H2 test and the reference point the linear state
  must approach. *specified.*
  - Minimal variant: retrieve **top-2** segments by the existing scores and
    *concatenate* (not mix) their memories as a `[2M, d]` prefix — isolates
    mixture-vs-separate-tokens directly.
- **Coarse-to-fine routing (scaling path)** — top-k segment selection with the
  existing compressed-query scores, then token-wise read / state construction
  from the selected segments only. Bounds read cost and state interference
  when pairs scale. *gated on the interference curve (PROGRESS #5).*
- **Iterative / multi-round retrieval** — retrieve per key in the query and
  concatenate results; needed for pointer chasing if reads stay outside the
  backbone. Mostly superseded by per-level reads (depth = iteration for free),
  but remains the fallback if the additive-bias read underdelivers. *draft.*
- **Multi-head segment attention** — keep one-shot retrieval but H heads ×
  per-head scores instead of one M·d query. Weaker middle ground; useful only
  as an ablation arm to quantify how much of the gap routing alone explains.
  *draft.*
- **Gated-delta state under per-level reads** — build S by the existing gated
  delta recurrence instead of plain summation (adds per-segment write-strength
  control) once the per-level read exists. Plain sum is the cleaner first
  experiment. *gated on H3 landing.*

## Diagnostics & measurement

- **Retrieval-mode ladder on concat** — mean / softmax / beta_softmax / raw on
  the ordered-concat task; documents that no one-shot mixture setting solves
  it (supports H2). Config-only. *specified; cheap.*
- **β sweep as diagnostic** — high β → clean single-pattern retrieval; low β →
  blend; if neither extreme helps concat, the convexity/uniformity story is
  confirmed from the existing knobs. *draft.*
- **Per-position retrieval probing** — with the bank-prefix control, log which
  stored segments each query position attends to (attention rollout); tells
  whether routing or payload is the residual failure if the control also
  fails. *draft.*
- **State-interference curve** — exact-match vs n_stored for bank-prefix read
  vs linear-state read at matched everything; the gap is pure state
  superposition cross-talk (the MQAR/Zoology-style ceiling). *specified.*
- **Write-read gradient balance** — under per-level reads the read module
  becomes the only outer-loss path shaping stored values (until BPTT
  detachment); watch inner-loss vs outer-loss drift. *draft.*

## Benchmarks & tasks

- **Hex addition / value synthesis across segments** — two keys in different
  segments, answer = function of both values (not concatenation). Tests
  integration, not just ordered recall. *draft; generator extension of
  `query_mode`.*
- **Associative join / pointer chasing** — `!K1:K2! … !K2:V!`, query `?!K1:` →
  `V`; requires iterative retrieval. HF dataset `irodkin/ar_multihop` noted
  2026-09-10. The strongest test of depth-wise retrieval. *specified; needs
  the new read.*
- **babilong** — sanity benchmark for the new read (reference: ARMT worked on
  babilong; past runs in `runs/babilong_*`). *specified.*
- **Curriculum N4→128 with cross-segment queries** — stretch the existing
  curriculum harness (`CURRICULUM.md`) to the concat query mode so scaling is
  measured on the integration task too. *gated on H2/H3 settling.*

## Write side (guarded — do not disturb H1)

- **Keep independent per-segment writes** — the anti-forgetting property
  (PROGRESS fact 2) depends on it; any cross-segment write carry reintroduces
  the forgetting trade-off. Constraint, not an idea.
- **Learned write selection / importance** — decide per segment how strongly
  to store (analog of H2 in the RL proposal). Only after the read interface
  settles, or write-quality and read-quality confound each other. *gated.*
- **Per-segment write budget (K schedule)** — more inner steps for denser
  segments; interacts with the outer/inner loss balance. *draft.*

## Baselines & comparisons

- **ARMT-style forward writer behind the same read interface** — if the new
  read lifts forward-written memories as much as gradient-written ones, the
  read was the bottleneck all along; if gradmem still leads, H1 strengthens.
  The decisive writer×read factorial. *specified.*
- **RMT / plain replay at matched read** — existing `rmt.py` + vanilla replay
  runs, re-scored on the concat benchmark. *draft.*
- **TTT writer** (arXiv:2407.04620) — inner-model weights as memory; isolates
  "gradient writes per se" from "gradient writes into reader-visible token
  space". *draft.*
