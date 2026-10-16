# STATUS — journal of additions, decisions, and pending work

Newest entry first. Each entry: date, what changed, decisions taken (with
rationale), verification status. Current state and hypotheses live in
`PROGRESS.md`; uncommitted directions in `IDEAS.md`.

**Maintenance rule:** every merged change updates this journal and, when it
changes the evidence picture, `PROGRESS.md` in the same commit.

## 2026-10-15 (d) — interface-token read variant folded into the H3 spec

**Added** — the *Variant* section in
[`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md): the same linear state,
read only by M learnable interface tokens (dedicated M_0; per-layer state
reads masked to those positions; all other positions reach memory via
softmax attention to them — the Flamingo-resampler/Q-Former pattern). The
three read designs now form a read-site ladder: one-shot (current) → M·L
interface → T·L·H all positions.

**Decisions** — the placement/causality trap is documented and resolved:
right-appended causal = memory unreachable at answer positions; left-prefix
causal = query-blind reads (silent regression to today's architecture).
Pinned resolution: the **two-pass layout** (pass 1 = `_compress_query`
layout with target-span blinding, for query-conditioning; pass 2 = the
current READ layout with M_0 + per-level state reads replacing the one-shot
mixture). New knob `read_state_read_sites: all | interface`; ladder arm (d′)
with decision criterion "(d′) vs (d) decides whether integration needs
per-token reads or only per-layer reads through a small interface"; new
diagnostics (reader attention mass on interface tokens by answer/non-answer
positions; query-sensitivity check of interface reads).

## 2026-10-15 (c) — H3 spec written: token-wise reads via linear attention

**Added** — [`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md): the formal
spec for the linear-state per-level read. Mechanism framed as *extra
attention heads on every reader block whose keys/values are the stored memory
tokens, folded once into a per-head associative state* — per-layer normalized
queries with learnable β, unnormalized (non-convex) linear-attention
weights, gated parallel additive injection, no memory tokens at READ, S = 0
graceful fallback. Includes the pinned design-decision table (kernel, norms,
gates, projection sharing, query source, values, write-side invariance), cost
accounting (~20 % FLOP overhead per instrumented layer, O(d²/H_r) state),
proposed `read_linear:` config surface, diagnostics, the four-arm
experimental ladder (one-shot / bank-prefix / exact-softmax heads /
linear-state) with pre-agreed decision criteria, risks, open questions, and
prior art (Titan, Based, Flamingo, DeltaNet, ARMT).

**Decisions** — unnormalized kernel (normalization would reintroduce per-token
convexity — the failure being removed); shared κ/ν with per-layer Q (state
built once, layer distributions differ); values are the raw post-inner-loop
memory tokens (projected value codes were for the one-shot mixture's
benefit); write side untouched. PROGRESS H3 → *specified*; missing-experiment
#3 links the spec.

## 2026-10-15 (b) — project organization around the multi-segment challenge

**Added** — `PROGRESS.md` (living summary: facts, hypotheses ledger, missing
experiments), this journal, `IDEAS.md` (directions pool), and
`docs/REMOTE.md` (cluster workflow, adapted from the gradmem_rl copy —
project path `/home/jovyan/dzhivelikian/test_time_gd`, this repo's
`run.yaml`/manifest specifics). Layout mirrors gradmem_rl: root keeps the
working files; topic docs (`STAR.md`, `ADAPTIVE_MEMORY.md`, `CURRICULUM.md`)
stay at root for now.

**Decision** — the multi-segment challenge (forgetting + cross-segment
integration) is the organizing frame of the repo; single-segment gradmem work
is background. Evidence so far was tracked in the obsidian daily notes — the
facts and the decision trail are now consolidated here; the daily notes remain
the scratch space.

## 2026-10-15 (a) — concat failure concluded architectural; read-interface hypothesis adopted

**Result** — ordered pair concatenation (`?!K1:?!K2:` → `V1!V2!|`, keys in
different segments): the segment-wise-query linear-attention read fails to
aggregate information from two segments. Non-linear/sparse value codes
(`hopfield_value_activation` topk/relu/gated, added 2026-10-06) do not improve
it.

**Analysis** (see PROGRESS fact 7) — one compressed query → one score per
segment (routing); convex softmax mixture → superposition in fixed M·d
(payload); single pre-read → no iterative retrieval. Linear value projections
are algebraically eliminable; the code nonlinearity sits before mixing while
the decode after mixing is linear; LayerNorm neutralizes raw-vs-softmax
magnitude differences.

**Decision** — next architecture: **token-wise reads of the Hopfield state**
(ARMT-style), refined in discussion to the **linear-state per-level read**
(Titan-style: per-token, per-layer queries into a shared `S = Σkᵢvᵢᵀ`, added
as a bias to softmax attention over the query context, no memory tokens in the
read). Extending the read context with the full memory bank is kept **only as
a diagnostic control** (upper bound), not as the scalable design.

## 2026-10-06 — non-linear / sparse value codes (commit `d02209d`)

**Added** — `hopfield_value_activation` ∈ {linear, topk, relu, gated} with
`hopfield_value_topk`, `hopfield_value_l1_weight` (unit-column-normalized
decoder under L1), `hopfield_value_gate_init`; example config
`configs/gradmemgpt/kv_retrieval/hopfield_value_activation.yaml`; value-code
diagnostics (`hopfield_value_*` stats).

**Rationale** (2026-10-05 note) — a linear value projection is eliminable:
`ΣaᵢE(vᵢ) = E(Σaᵢvᵢ)`, so widening only re-parameterizes a P×P adapter;
non-negative sparse codes (top-k after ReLU recommended) prevent cancellation
between latent coordinates. Outcome: did **not** fix concat (2026-10-15a).

## 2026-09-29 — ordered pair concatenation dataset (commit `d5b6f18`)

**Added** — `query_mode='ordered_pair_concat'` in
`kv_dataset_utils.generate_sequence` with `query_pair_segment_relation`
(`different` = cross-segment, the probing case; `any`). The benchmark that
exposed the read-interface failure.

## 2026-09-10 — forgetting matrices + first Hopfield results

**Results** (2 mem tokens, 8 KV pairs, 4 pair-aligned segments):

- Hopfield (independent per-segment writes + query-conditioned retrieval):
  **all pairs stored perfectly**; clean forgetting matrices also at key dim 64.
- Carried memory + STAR: slightly better (~10% under perturbation); learned
  gated SGD updates: no difference; carried-memory gradmem: forgets.
- Next steps recorded then: attention-entropy metrics (since implemented:
  `hopfield_entropy_*`, `hopfield_n_seg_50/95`), multi-segment combination
  task, truncated BPTT for Hopfield (`hopfield_bptt_segments`), scale to more
  pairs, several retrievals, babilong.

## 2026-09-02/03 — forgetting matrix, unified metrics, pair-aware segmentation (commits `a063f10`, `aa9f6dc`, `0f0478b`)

Per-segment forgetting evaluation (probe every stored pair after each segment
write → `[n_seg, n_seg]` matrix, `forward_per_segment_eval` +
`_probe_boundary_memory`); metrics unified across runners; segmentation moved
to `!K:V!` pair boundaries (fixed windows straddled pairs — the straddling
pathology), with the collator sharing the model's division via
`GradMemGPT._divide_pairs`.

## Earlier

- **2026-08-31** — STAR-like regularizer for gradmem (`04841cc`); vanilla
  replay control (`a253c33`); tokenization/logging fixes (`fd7a7b9`,
  `37b34a4`).
- Stage-0 capacity-vs-interference grid
  (`manifests/stage0_capacity_vs_interference.yaml`) and the m×n_segments
  decision protocol in its header.

## Pending tasks

- [ ] Bank-prefix diagnostic control (+ top-2-concat minimal variant) —
      decides H2 (read-interface vs write-quality).
- [ ] Retrieval-mode ladder on concat (mean/softmax/beta_softmax/raw).
- [ ] Implement the linear-state per-level read (H3): per-head states,
      per-layer normalized queries + learnable β, gated additive injection;
      config-gated next to the current retrieval for A/B.
- [ ] Regression guards for the new read: forgetting matrix + single-key KV.
- [ ] Scale-to-more-pairs interference curve; top-k coarse routing if needed.
- [ ] Cross-segment benchmark suite: hex addition; pointer chasing
      (`irodkin/ar_multihop`).
- [ ] babilong sanity for the new read.
