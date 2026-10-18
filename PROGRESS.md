# PROGRESS — multi-segment GradMem: current status, facts, hypotheses, and what is missing

Living summary; update at every milestone. Journal of changes: `STATUS.md`.
Pool of future directions: `IDEAS.md`. Cluster workflow: `docs/REMOTE.md`.

**One-line state:** independently written per-segment memories + segment-wise
linear-attention retrieval (the Hopfield mode) solve catastrophic forgetting on
single-key KV retrieval, but the one-shot read interface fails any task that
must **integrate content from two segments** (ordered pair concatenation), and
non-linear/sparse value codes do not help — the read architecture itself is the
bottleneck. The per-token, per-level linear-state read (H3, Titan-style) is now
**implemented** (config-gated, write side untouched); what is unverified is
whether it actually fixes concat — run the ladder in `manifests/kv_linear_read.yaml`.

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
   are written (2026-09-10; see also [`STAR.md`](STAR.md) and
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

## Hypotheses ledger

| hypothesis | status | strongest support | strongest doubt / missing |
|---|---|---|---|
| **H1** independently written per-segment memories + attention retrieval beat carried memory on forgetting | **supported** (KV single-key) | fact 2: all pairs stored with 2 tokens / 8 pairs / 4 segments; clean forgetting matrices | not yet shown at large segment counts; carried-memory controls with matched budget not tabulated here |
| **H2** the concat failure is a **read-interface** bottleneck, not a write-quality one | **working hypothesis** | facts 5–7: fails across all retrieval modes and all value codes; the mixture/superposition argument | decisive control missing: bank-prefix read (all stored tokens as prefix) — if that also fails, write side is implicated |
| **H3** per-token, per-level linear-attention reads from a shared state (`S = Σkᵢvᵢᵀ`), added as a bias to softmax attention over the query context, fix routing + payload + iteration at O(1) read cost | **implemented + reviewed** ([docs/TOKENWISE_READ.md](docs/TOKENWISE_READ.md) + `grad_memgpt.py` `use_linear_state_read`; STATUS.md 2026-10-18 (b) impl, (c) review: contract verified, one latent jvp site-mask padding bug fixed) | per-token/per-head queries remove the compromise query; unnormalized weights are not convex; depth gives iterative retrieval; S=0 is a graceful no-store fallback; locked by 14 unit tests (state==explicit attention, gate-zero inertness, site masks incl. jvp padding, blinding, boundary-k==truncated forward, write-side purity) | accuracy unverified: the concat ladder (all / interface / exact-softmax vs the Hopfield control) + single-key regression guard are queued in `manifests/kv_linear_read.yaml`; scale calibration (LayerNorm + gates) remains the practical risk; state interference grows with n_stored |

## Missing experiments (decision-critical, in priority order)

1. **Run the H3 read ladder** — the linear-state per-level read is IMPLEMENTED
   and **reviewed** (`use_linear_state_read` in [`grad_memgpt.py`](
   grad_memgpt.py); brief: [`docs/TOKENWISE_READ_TASK.md`](
   docs/TOKENWISE_READ_TASK.md); journal: STATUS.md 2026-10-18 (b) impl, (c)
   review + one latent jvp-padding fix). Missing is only the **evidence**:
   run `manifests/kv_linear_read.yaml` (Hopfield control / linear all /
   linear interface / exact-softmax × seeds 4378/43894/2343 on the
   ordered-concat data, plus the single-key regression guard with forgetting
   matrices from `per_segment_eval`). (c) vs (d) separates prefix-vs-head
   from exact-vs-superposed; (d′) vs (d) decides whether integration needs
   per-token reads or only per-layer reads through a small interface.
   Watchlist while it runs: `read_state_gate_abs_mean` (collapse → 0),
   `read_state_mu_over_x_norm_ratio` (calibration), `read_state_beta_mean`,
   eval-only `read_state_cross_talk`, and `inner_loss_seg_*` (write-read
   objective tension).
2. **Bank-prefix diagnostic control (ladder arm (b))** — NOT implemented
   (out of the implementation task's scope). Prepend all stored per-segment
   memory tokens at READ (upper bound; not scalable by design). It is the
   H2 decider but only decisive if (d)/(d′) fail — implement while the
   15-run manifest executes. Cheap variant that isolates the failure mode:
   retrieve **top-2** segments and *concatenate* their memories as a
   `[2M, d]` prefix.
3. **Retrieval-mode ladder on concat** (`mean` / `softmax` / `beta_softmax` /
   `raw`): config-only; documents that no one-shot mixture setting solves it.
4. **Scale to more pairs**: state-interference curve as n_stored grows; if
   cross-talk binds, coarse top-k segment routing (reuse the compressed-query
   scores) before building the state.
5. **Cross-segment benchmark suite**: hex addition (multi-key synthesis);
   associative join / pointer chasing (`!K1:K2! … !K2:V!`, needs iterative
   retrieval; HF dataset `irodkin/ar_multihop` noted 2026-09-10).
6. **babilong sanity** of the new read (reference point: ARMT worked on
   babilong).
7. **Writer×read factorial** (forward writer behind the winning read; IDEAS
   baseline) — after the read settles.

## Map

- Working files (repo root): `README.md` (repo & harness overview) · this file
  · `STATUS.md` · `IDEAS.md` · `AGENTS.md` (agent conventions).
- Topic docs: [`docs/ADAPTIVE_MEMORY.md`](docs/ADAPTIVE_MEMORY.md) (gated
  carried-memory updates) · [`docs/CURRICULUM.md`](docs/CURRICULUM.md)
  (stage-wise N curriculum) · [`STAR.md`](STAR.md) (worst-case stability
  regularizer, still at root).
- Design specs & workflow (docs/): [`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md)
  (the H3 read-interface spec) · [`docs/REMOTE.md`](docs/REMOTE.md)
  (cluster workflow).
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
