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
vs N is RUNNING** (resubmitted 2026-10-29, job `ccde14ed`, early gate
passed) with user-selected arms: `interface` (linear kernel, the deployable
d′) + `interface_softmax` (exact softmax at the same interface sites — the
no-superposition control); hopfield dropped, linear-all variants deferred.
Single-key checkpoint curriculum N ∈ {8,16,32,64,96}, 3 seeds,
50k-steps/0.99-EM protocol — revision-2 predictions in
[`manifests/kv_interference_curve.yaml`](manifests/kv_interference_curve.yaml).

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

## Hypotheses ledger

| hypothesis | status | strongest support | strongest doubt / missing |
|---|---|---|---|
| **H1** independently written per-segment memories + attention retrieval beat carried memory on forgetting | **supported** (KV single-key) | fact 2: all pairs stored with 2 tokens / 8 pairs / 4 segments; clean forgetting matrices | not yet shown at large segment counts; carried-memory controls with matched budget not tabulated here |
| **H2** the concat failure is a **read-interface** bottleneck, not a write-quality one | **supported** (ladder, 2026-10-22) | facts 5–7; the ran control: bank-prefix fails at 0.024 WHILE (c)/(d)/(d′) reach 1.00/0.997/0.999 on the SAME stored tokens — the prefix interface, not write quality, is indicted (fact 8) | prefix-interface failure is established only at this training scale (4L/128d from scratch); a pretrained/larger frozen reader might yet make the prefix work |
| **H3** per-token, per-level linear-attention reads from a shared state (`S = Σkᵢvᵢᵀ`), added as a bias to softmax attention over the query context, fix routing + payload + iteration at O(1) read cost | **supported for routing+payload** (ladder 2026-10-22: 0.995–1.00 vs 0.25/0.02 controls; single-key 0.997+); implementation STATUS.md 2026-10-18 (b)/(c) | facts 8–9; interface variant (d′) matches all-position reads at M·L sites — cheaper architecture suffices given pass-1 conditioning | the **iteration** claim (pointer chasing, depth-bounded multi-hop) is untested; interference-vs-N RUNNING (curve submitted 2026-10-26) |

## Missing experiments (decision-critical, in priority order)

1. **Interference curve vs N** — **RUNNING (resubmitted 2026-10-29, job
   `ccde14ed`, early gate passed; manifest
   [`kv_interference_curve.yaml`](manifests/kv_interference_curve.yaml))**.
   Protocol per user: single-key KV, checkpoint-chained curriculum
   N ∈ {8,16,32,64,96}, 50k steps/phase, 0.99-EM early stop; arms
   `interface` + `interface_softmax` (user-selected: hopfield dropped,
   linear-all deferred — definitions kept in the chain script for a later
   manifest). Revision-2 predictions P1–P3 in the manifest header;
   `read_state_cross_talk` trajectory is part of the readout (already
   rising in the live run). If superposition binds: coarse top-k segment
   routing (reuse the compressed-query scores) before building the state.
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
  reports/READ_LADDER_RESULTS.md) (+ `reports/figures/`,
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
