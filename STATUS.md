# STATUS — journal of additions, decisions, and pending work

Newest entry first. Each entry: date, what changed, decisions taken (with
rationale), verification status. Current state and hypotheses live in
`PROGRESS.md`; uncommitted directions in `IDEAS.md`.

**Maintenance rule:** every merged change updates this journal and, when it
changes the evidence picture, `PROGRESS.md` in the same commit.

## 2026-10-18 (c) — implementation review of 8b69397: verdict + one fix

**Reviewed** — the H3 implementation against
[`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md) /
[TOKENWISE_READ_TASK.md](docs/TOKENWISE_READ_TASK.md). Verified in code:
config validation (all pinned asserts, plus a sensible extra — incompatible
with `use_gradient_checkpointing`, with rationale), module construction
(shared bias-free κ/ν, per-layer Q/β/γ/LN, bias-free `W_o` with the S=0
rationale), state construction (per-head normalized keys, stored-out item
masking, S=0 for empty banks), hook injection (parallel from the block
input, tuple/non-tuple outputs, site masks, detached stats, ctx-gated so
WRITE/compression forwards are untouched), READ rewiring (no-prefix "all"
mode; interface two-pass with the `labels ≥ 0` blinding; zero-row slicing
for the unpredictable query position 0), the flag-gated raw-token STORE
branch (BPTT detach semantics preserved, `stored_keys` None entries
guarded), eval mirrors (`_read_once_linear`, boundary probes → forgetting
matrix in all arms), runner/`run_from_config`/run-naming plumbing, configs,
and the 15-run manifest. Suite rerun independently: 13 new + 61 existing
green.

**Fixed** — one latent bug: `_linear_read_forward_pass` padded `x` and
`attn_mask` for the jvp %32 alignment but **not** `site_mask`, so the hooks'
`mu * site_mask` would shape-mismatch on
`attn_implementation ∈ {jvp_flash, hvp_semi_manual}` whenever `T % 32 ≠ 0`.
No config uses those paths today (everything is eager), so nothing was
broken at runtime — it would have bitten the first second-order jvp run.
One-line fix (zero-pad `site_mask` alongside) + regression test
`test_jvp_padding_site_mask_aligned`, verified to fail without the fix and
pass with it (file now 14 tests).

**Gap for the experiment plan** — ladder arm (b) bank-prefix control is not
implemented (out of the task's scope by design). It is the H2 decider but
is only decisive if (d)/(d′) fail; implement it while the 15-run manifest
executes. PROGRESS missing-experiments rewritten into the run plan.

## 2026-10-18 (b) — H3 implemented: the linear-state per-level read

**Added** — `use_linear_state_read` in
[`grad_memgpt.py`](grad_memgpt.py): a config-gated READ mode replacing the
Hopfield RETRIEVE + memory-prefix READ per
[`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md). The stored per-segment
memories are folded once into per-head linear-attention states
(`S_r = Σ k v^T`, k = L2-normalized `read_state_key_proj` of each stored
memory TOKEN, v = `read_state_value_proj`); a forward hook on each
instrumented decoder block injects `γ^ℓ ⊙ W_o(concat_r β_r ·
normalize(Q_r(LN(x))) @ S_r)` at the block OUTPUT (parallel additive; the
frozen backbone attention path is bitwise untouched). Per-layer Q/β/γ/LN,
shared κ/ν, `read_state_layers` subset support, read-site masks excluding
pads (incl. the jvp %32 padding), and the S=0 no-store fallback (zero bias,
no fallback prefix). Three arms on the same modules: `read_sites="all"`
(every real position), `read_sites="interface"` (the two-pass Variant:
pass 1 conditions learnable M_0 on the target-span-blinded query, pass 2
answers `[read_st?, interface, read_end?, query]` with the conditioned
embeddings), and `read_state_exact_softmax` (per-token softmax over the
unfolded bank). Eval mirrors: `_read_once_linear` +
`forward_per_segment_eval` serve the state read against
`stored_*[:n_stored]`, so the forgetting matrix works in all arms.
Diagnostics (detached, train+eval): `read_state_gate_abs_mean`,
`read_state_beta_mean`, `read_state_mu_over_x_norm_ratio` (‖μ‖/‖x‖ — see
deviations), `read_state_cross_talk` (eval-only, mean |q·k| over non-top
bank items). Configs `linear_state_read.yaml` /
`linear_state_read_interface.yaml` (ordered-concat data), manifest
`kv_linear_read.yaml` (control / all / interface / exact-softmax × 3 seeds +
single-key regression guard). `read_linear:` yaml section plumbed through
`run_from_config.py` (flattening + overrides), the runner argparse
(`read_state_layers` parsed from a comma-separated string),
`compute_metrics_fn` (explicit `read_state_*` block) and
`generate_run_name._KNOWN_SECTIONS`.

**STORE change (linear mode only)** — STORE skips the segment-key
`_compress_segment` forward pass and value encoding (raw `mem_batch` is
appended; `stored_keys` gets `None` entries, guarded where detached); with
the flag off the STORE code path is byte-identical to before.

**Decisions** — (1) `W_o` is bias-free: a bias term would leak `γ·b` into
every read site even with an empty bank, breaking the pinned "S = 0 ⇒ zero
bias" contract. (2) Linear-mode READ forwards pass an explicit
attention_mask + cumsum-1-clamped position ids (needed for interface
blinding; with no prefix and C=0 the [Q+1, V] logit layout is preserved by a
zero row for the unpredictable query position 0, whose label is always
-100). (3) `use_linear_state_read` asserts `use_gradient_checkpointing=False`
— checkpoint recompute re-runs blocks outside the active read pass, so the
hooks would silently drop the bias during backward. (4)
`read_state_segment_emb` is a stubbed `NotImplementedError`: the per-segment
count is dynamic under pair-aware / segment_size chunking, so the embedding
would need lazy sizing (spec-sanctioned deferral). (5) The exact-softmax arm
shares κ/ν/Q/β/γ/W_o and the site masks with the state arm — only the read
kernel differs.

**Deviations from the spec** — `read_state_mu_norm_ratio` is logged as
`read_state_mu_over_x_norm_ratio` = ‖μ‖/‖x‖ (mean over sites, averaged over
layers): the attention sub-output norm would need a second hook per block on
the attention submodule; the block-input norm is the documented fallback
from the task brief. The per-position retrieval probe (argmax segment per
query position) is reduced to the `read_state_cross_talk` scalar for now;
the full probe is left for the interference-curve task.

**Verification** — `test_grad_memgpt_linear_state_read.py` (13 tests): state
read == explicit `Σ (q·k) v` in float64 (this caught a real einsum bug: the
output subscript labelled the head axis with S's FIRST matrix index, i.e.
the read computed `q Sᵀ` — fixed); gate-zero bitwise inert; no-stored ⇒ zero
bias (bitwise vs hooks-disabled); site masks exact at the injecting block
(all + interface, both passes); exact arm == manual softmax (and ≠ state
arm); interface pass-1 blinding (target-span perturbation leaves the
interface embeddings bitwise unchanged) + query sensitivity (non-target
perturbation changes them); boundary-k probe == truncated forward (logits
allclose, EM flags identical); WRITE side bitwise untouched by the hooks +
outer grads reach every read module under `grad_mode="second"`;
`collect_segment_mems` purity; layer-subset + config guards. Full suite:
61 existing + 13 new green under `uv run --extra dev pytest`. Smoke runs
(2 training steps, batch 8, GPU): all four arms complete, log
`train_/eval_read_state_*`, and write `forgetting_matrix_step*.json/csv`
(artifacts under `runs/default/…`, logs under `/tmp/kv_smoke_logs/`).
`run_from_config.py --dry-run` clean for both configs;
`run_scheduler.py manifests/kv_linear_read.yaml --dry-run` clean (15 runs).
NOT yet verified: training-scale behavior (gate/β drift, stability) and the
actual concat accuracy — that is the ladder in `kv_linear_read.yaml`, to run
on the cluster.

## 2026-10-18 — executor brief for the H3 implementation

**Added** — [`docs/TOKENWISE_READ_TASK.md`](docs/TOKENWISE_READ_TASK.md):
the implementation task for the linear-state read spec, written for an
executor agent with no conversation context. Grounds the spec in the
codebase (config flow yaml → `run_from_config.py` section flattening →
runner argparse → `GradMemGPTConfig`; stats via `inner_loop_stats` →
`compute_metrics`; block access for gpt2 `h[i]` / llama `layers[i]`), fixes
the v1 contract (raw stored tokens only — asserts
`hopfield_value_proj_dim=None`; STORE skips key/encoding work in this mode;
hooks parallel-inject at block outputs with read-site masks incl. the
jvp_flash %32 padding; interface two-pass with `labels ≥ 0` blinding), and
sets an 8-phase plan (recon → config/state → default mode → exact-softmax →
interface → eval mirrors → diagnostics → configs/manifest/smoke → docs)
with named unit tests as per-phase exit conditions and six final acceptance
criteria (flag-off bitwise inertness; write side untouched; forgetting
matrix works in the new mode; all four arms smoke-run). Bank-prefix control,
top-k routing, and the pointer-chasing dataset are explicitly out of scope.

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
