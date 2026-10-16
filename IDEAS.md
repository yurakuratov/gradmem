# IDEAS — pool of candidate directions and draft hypotheses

Not a plan; nothing here is committed or pre-registered. When an idea graduates
to "will run", it moves into `PROGRESS.md` (missing experiments) and gets a
config/manifest; decisions get journaled in `STATUS.md`. Status tags:
**draft** (rough), **specified** (geometry/goal concrete), **gated** (waiting
on another result).

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
