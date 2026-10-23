# TASK — implement the linear-state per-level read ([TOKENWISE_READ.md](../docs/TOKENWISE_READ.md))

Executor brief. You are implementing the read architecture specified in
[TOKENWISE_READ.md](../docs/TOKENWISE_READ.md) — **that document is authoritative for
all design decisions** (kernel choice, normalization, gating, two-pass
layout, naming). This file adds the codebase grounding, the phase plan, and
the acceptance criteria. Work through the phases in order; each has a
checkable exit condition.

## Context (read first, in this order)

1. [TOKENWISE_READ.md](../docs/TOKENWISE_READ.md) — the spec. § The mechanism, §
   Design decisions, and the Variant section are binding.
2. [PROGRESS.md](../PROGRESS.md) — why: facts 5–7 (the one-shot read fails
   cross-segment integration), hypothesis H3. The regression guards
   (single-key KV, forgetting matrix) are part of acceptance.
3. [AGENTS.md](../AGENTS.md) — repo conventions (uv, run_from_config.py,
   output-folder naming).
4. Code: [`grad_memgpt.py`](../grad_memgpt.py) is the only model file to
   touch. Line references below are approximate as of 2026-10-18 and will
   drift — locate by name.

## Ground rules (invariants)

- **Write side untouched.** Per-segment inner loops, STORE semantics, and
  `use_gated_delta_memory`/`memory_update` paths must not change. The read
  module is outer-loop only and never enters an `autograd.grad` inner step.
- **Flag-off is bitwise inert.** With `read_linear.use_linear_state_read:
  false` (default), every existing mode must produce identical outputs to
  the current code. The full existing test suite stays green.
- Existing modes keep working alongside the new one (the new read replaces
  the Hopfield RETRIEVE only when enabled).
- Follow the repo's docstring/comment style (see the Hopfield code for
  density and tone). Every new config knob gets a docstring entry in
  `GradMemGPTConfig` and a line in the config yaml comments.
- **Maintenance rule:** on completion, add a `STATUS.md` journal entry and
  update `PROGRESS.md` (H3 status: specified → implemented) in the same
  change.

## Scope

**In scope:** the `read_linear:` config surface (all three arms: `all`
positions, `interface` tokens, `exact_softmax` ablation), state construction
from stored memory tokens, per-layer injection via forward hooks, the
two-pass interface layout, eval/probe mirrors so the forgetting matrix and
`forward_per_segment_eval` work in the new mode, diagnostics, unit tests,
example configs, a scheduler manifest, and a local smoke run.

**Out of scope (do not build):** bank-prefix control arm, top-k segment
routing, pointer-chasing/hex datasets, any writer change, ARMT/forward-writer
baseline. These are separate tasks in [IDEAS.md](../IDEAS.md).

## Codebase map (where things plug in)

| What | Where in `grad_memgpt.py` |
|---|---|
| Config params + validation | `GradMemGPTConfig.__init__` (~L150–627) — mirror the hopfield knobs' assert style |
| Module construction | `GradMemGPT.__init__` — see the `use_hopfield_memory` block (~L894–962) |
| WRITE: per-segment STORE | segment loop, "Per-segment Hopfield STORE" (~L2629–2680) |
| READ: Hopfield RETRIEVE (to be replaced when enabled) | ~L2836–2935 |
| READ: sequence composition `x_qry = [read_st?, mem_inp, read_end?, qry_emb]` | ~L2974–3000 |
| Query compression w/ target blinding (`labels >= 0`) | `_compress_query` (~L1336) |
| Batched read for eval probes | `_read_once` (~L1429) |
| Boundary-state mirror for the forgetting matrix | `_probe_boundary_memory` (~L1611) |
| Per-segment eval entry | `forward_per_segment_eval` (~L1742) |
| forward() | ~L1871; returns `{'predictions', 'inner_loop_stats', ...}` — stats flow to the trainer/`compute_metrics` |
| Backbone access | `get_backbone()` (~L18). Decoder blocks: GPT-2 `backbone.h[i]`, Llama `backbone.layers[i]` — both must work (default config is gpt2; manifests set llama) |

Config plumbing: `run_from_config.py` flattens yaml sections to CLI flags
(section list at ~L102; the `hopfield:` loop at ~L224 is the pattern to copy
for a new `read_linear:` section). The runner
`run_gradmemgpt_on_kv_retrieval.py` needs matching argparse args (find the
hopfield ones) and `compute_metrics` entries for the new stats (the
`hopfield_entropy_*` block at ~L413 is the pattern).

## The contract (load-bearing details)

**State construction** (once per READ, after the segment loop): stack the
stored per-segment memories `[B, n_stored, M, d]` + boolean stored mask
`[B, n_stored]` (reuse `stored_masks`; samples/segments with no tokens are
masked out). Per head $r$: keys `κ_r(m)/‖·‖`, values `ν_r(m)`, state
`S_r = Σ k v^T` → keep as `[B, H_r, d_h, d_h]`. Per-sample: no stored
segments ⇒ `S = 0` ⇒ zero bias (no fallback prefix — the current
`hopfield_stored`-where branch is not needed in this mode).

**Raw tokens only (v1 constraint):** `use_linear_state_read=True` must
assert `hopfield_value_proj_dim is None` (values are the raw post-inner-loop
memories; the encoded-code pathway is for the one-shot mixture). In this
mode STORE skips `_compress_segment` key computation and value encoding —
append raw `mem_batch` (`[B, M, d]`) and the mask; `hopfield_bptt_segments`
detaches those raw tensors exactly as it does keys/values today.

**Injection** (parallel, per block, via forward hooks on the decoder
blocks): from the block *input* hidden state `x`, compute
`μ = W_o · concat_r( β_r^ℓ · normalize(Q_r^ℓ(LN(x))) @ S_r )` and add
`γ^ℓ ⊙ μ` **only at read-site positions** (see masks below) to the block
*output* hidden state. Block hooks must handle tuple outputs (GPT-2 and
Llama blocks both return tuples) and must be a no-op when the module is not
in an active read pass (guard with a flag set around the READ forwards, so
WRITE-phase and compression forwards are untouched). LN: reuse
`nn.LayerNorm(n_embd)` per layer (matches ln_1 placement; do NOT call the
backbone's own ln_1 — the query path must be independent of it).

**Read-site masks.** `read_state_read_sites="all"`: every real (non-pad)
position of the read sequence. `"interface"`: only the M interface-token
positions. With `attn_implementation in ('jvp_flash', 'hvp_semi_manual')`
the read sequence is zero-padded to a multiple of 32 — the site mask must
exclude pad positions (μ is zeroed there). The site mask, the state, and the
"active" flag should travel together (e.g. a small context object the hooks
read; set before the READ forward, clear after).

**Default mode layout (read_sites="all"):** read sequence is
`[read_st?, read_end?, query]` — no mem tokens. Position ids via the
existing `cumsum-1 clamp` pattern.

**Interface mode (two-pass, per the spec's Variant):**
- Pass 1 — conditioning: `[query (target-span blinded), M_0 tokens]`, hooks
  active only at the interface positions. Blinding reuses the
  `_compress_query` pattern (`labels >= 0` positions masked out of the
  query's attention mask). Take the final hidden states at the interface
  positions `[B, M, d]` as the conditioned interface embeddings.
- Pass 2 — answering: `[read_st?, interface, read_end?, query]` with those
  embeddings as the interface positions' inputs, hooks still active only
  there. `M_0` is a fresh `nn.Parameter` (count = `n_mem_tokens`); the
  pass-1 layout, masks and position ids mirror `_compress_query`.
- `mem_proj` is not applied to interface tokens (they are not memories);
  ctrl tokens carry over unchanged.

**Exact-softmax arm (`read_state_exact_softmax=True`):** replace `q S_r`
with per-token softmax over the bank: `softmax(q·K_r^T) @ V_r` computed from
the stacked bank (mask stored-out segments and pad positions). Same
injection, same sites. This is the ablation arm — keep the code path small
and shared (projections, gates, site masks identical).

**Validation asserts to add** (Config): `use_linear_state_read` requires
`use_hopfield_memory=True`; mutually exclusive with
`use_gated_delta_memory`; requires `hopfield_value_proj_dim is None`;
`read_state_read_sites ∈ {"all", "interface"}`;
`read_state_exact_softmax` requires `use_linear_state_read`;
`read_state_layers` is `"all"` or a list of valid block indices;
`read_state_heads ≥ 1`; `read_state_gate_init > 0`.

**Gradient semantics:** read-module params get outer-loss gradients only.
Stored values participate in the outer graph through the reads unless
detached by `hopfield_bptt_segments` (keep that semantics). Verify the
inner loop is untouched: read params must not appear in any
`autograd.grad` inner call (they won't, by construction — add a test).

## Phase plan

### P0 — Reconnaissance (no code)
Run the existing suite (`uv run pytest test_grad_memgpt_*.py
test_kv_dataset_utils.py test_two_key_metrics.py`); map the config flow
end-to-end for one hopfield knob (yaml → CLI → argparse → `GradMemGPTConfig`
→ model); confirm block access on both gpt2 (from-config) and llama.
**Exit:** you can name the exact files/lines where a `read_linear:` knob
must be registered, and you have a 2-minute local forward smoke of the
hopfield config.

### P1 — Config + modules + state
Config params (spec § Proposed config surface: `use_linear_state_read`,
`read_state_heads`, `read_state_layers`, `read_state_beta_init`,
`read_state_gate_init`, `read_state_read_sites`,
`read_state_segment_emb` — the last may be a no-op stub with an assert-off
if it proves entangled, note it in STATUS), validation asserts, module
construction (κ, ν shared; per-layer Q, β, γ, LN; `W_o`; interface `M_0`),
state construction from the stored bank.
**Exit:** unit test `test_state_read_equals_explicit_attention` (μ from the
state == explicit `Σ (q·k) v` over the bank, float64, all heads).

### P2 — Default mode (read_sites="all")
Hooks, site masks, padding, `S = 0` fallback, READ-phase rewiring (skip the
Hopfield RETRIEVE block; drop the mem prefix).
**Exit:** `test_gate_zero_is_bitwise_inert`,
`test_no_stored_segments_no_bias`, `test_read_sites_masking` (bias present
only at real read sites; pads unaffected), plus the flag-off suite green.

### P3 — Exact-softmax arm
**Exit:** `test_exact_softmax_arm` (matches a manual per-token softmax over
the bank).

### P4 — Interface mode (two-pass)
**Exit:** `test_interface_blinding` (perturbing target-span tokens in the
pass-1 input leaves the interface embeddings unchanged),
`test_interface_query_sensitivity` (perturbing non-target query tokens
changes interface μ), `test_interface_read_sites` (bias only at interface
positions in both passes).

### P5 — Eval mirrors
`_read_once`, `_probe_boundary_memory`, and `forward_per_segment_eval` must
serve the new read: boundary snapshots already carry `n_stored`; the probe
read becomes the state read (or interface two-pass) against
`stored_*[:n_stored]`, batched over the probe axis like today. The
forgetting matrix must be producible in `read_sites="all"` and `"interface"`.
**Exit:** `test_forward_per_segment_eval_linear_read` — for a fixed input,
probing at boundary k matches a forward that stops after k segments (logits
agree within tolerance; exact-match flags identical).

### P6 — Diagnostics + metrics plumbing
Log into `inner_loop_stats` (runner `compute_metrics` mirrors them; follow
the `hopfield_entropy_*` pattern): `read_state_gate_abs_mean`,
`read_state_mu_norm_ratio` (‖μ‖/‖Attn-out‖ per layer — needs the attention
sub-output norm; if awkward, log ‖μ‖/‖x‖ and name it accordingly),
`read_state_beta_mean`, `read_state_cross_talk` (mean |q·k| over non-top
bank items, eval-time or detached). All detached.
**Exit:** metrics visible in a smoke run's logs.

### P7 — Configs, manifest, smoke
- `configs/gradmemgpt/kv_retrieval/linear_state_read.yaml` (default arm,
  `read_sites: all`, on the ordered-concat dataset
  `data/N8-K2V2-V62-S4-ordered_pair_cross_1M` — already generated; regenerate
  via `notebooks/dump_dataset.ipynb` with `query_mode='ordered_pair_concat'`,
  `query_pair_segment_relation='different'` if missing).
- `configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml`.
- `manifests/kv_linear_read.yaml`: experiments = hopfield baseline
  (existing config, control), linear all, linear interface, exact-softmax
  (override); grid: seeds [4378, 43894, 2343]; `max_parallel: 6`; plus a
  single-key regression experiment on `N8-K2V2-V62_1M` (same shape as
  `manifests/kv_hopfield.yaml`).
- `run_naming` keys for the new knobs in the new configs.
**Exit:** `run_from_config.py --dry-run` clean for both configs;
`run_scheduler.py manifests/kv_linear_read.yaml --dry-run` clean; one short
local training run per arm (reduced steps/batch is fine) completes, logs the
new metrics, and writes the forgetting matrix.

### P8 — Docs + journal
STATUS.md entry (what/decisions/verification), PROGRESS.md H3 →
implemented (with what is still unverified), config-knob docs in the yamls.
**Exit:** link check over the touched markdown files.

## Final acceptance criteria

1. Full test suite (existing + new) green under `uv run pytest`.
2. Flag-off bitwise inertness demonstrated by the untouched existing suite
   plus `test_gate_zero_is_bitwise_inert`.
3. The four ladder-relevant arms runnable from configs: hopfield baseline,
   `all`, `interface`, `exact_softmax` — smoke-run locally, metrics logged.
4. Forgetting matrix + `forward_per_segment_eval` work in the new mode
   (P5).
5. Write-side untouched: `test_grad_memgpt_adaptive_*`, hopfield and
   pair-aware tests unchanged and green.
6. STATUS.md + PROGRESS.md updated per the maintenance rule.

## Reporting

End with: files touched, tests added (names), smoke-run artifacts paths,
any spec deviation (with reason) recorded in STATUS.md, and known gaps left
for follow-up (e.g. `read_state_segment_emb` stubbed, `mu_norm_ratio`
denominator substituted).
