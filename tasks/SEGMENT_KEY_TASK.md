# TASK — chunk-granularity keys for the interface read (`read_state_segment_items`)

Executor brief. Small, config-gated extension of the linear-state read
([TOKENWISE_READ.md](../docs/TOKENWISE_READ.md); implementation journal: STATUS.md
2026-10-18 (b)/(c)). Adds ladder arm3 of
[`manifests/kv_interface_softmax.yaml`](../manifests/kv_interface_softmax.yaml):
**one key per stored memory chunk (segment), derived directly from `M_grad`** —
the "store each memory chunk as whole with a key to it" model — instead of the
current per-token keys.

## Context

1. [READ_LADDER_RESULTS.md](../reports/READ_LADDER_RESULTS.md) — the ladder results; the
   interface arm (d′) is the deployable configuration (EM 0.999).
2. [TOKENWISE_READ.md](../docs/TOKENWISE_READ.md) § The mechanism — the current state
   construction: per-**token** items, keys = per-head slices of one shared κ.
3. This brief's design discussion (STATUS.md 2026-10-22 (b)): the granularity
   question — chunk vs token addressing — is the last untested axis of the
   interface read's key geometry.

**Fact to internalize first:** the linear mode already skips
`_compress_segment` (STORE is raw; keys already derive from `M_grad`). This
task changes ONLY the item granularity: per-segment chunks instead of
per-token entries. There is no segment-compression forward to remove.

## Design (pinned)

- Items = segments: `N = c` (not `c·M`). Source tensor: the same
  `values_stack [B, c, M·d]` — unflattened per chunk.
- **Keys:** one new shared `nn.Linear(M·d, d, bias=False)`
  (`read_state_seg_key_proj`), applied to each chunk pattern, reshaped to
  per-head slices `[B, c, H, d_h]`, L2-normalized. Per-head slices of one
  matrix, exactly like κ today; the *chunk* keeps a single address per head,
  shared across layers (state built once). "The same hopfield dim": with
  `d=128, read_state_heads=2`, `d_h = 64` — the run uses that.
- **Values:** one new shared `nn.Linear(M·d, d, bias=False)`
  (`read_state_seg_value_proj`), per-head slices `[B, c, H, d_h]`.
  Rationale (load-bearing): the additive bias `μ_t` must be `d`-dimensional
  per position — the one-shot read could reshape a retrieved `M·d` chunk into
  M prefix tokens, the bias read cannot, so the chunk's payload is compressed
  per head. Do NOT change this to raw `M·d` values.
- `_build_linear_state` gains the branch; downstream hooks/einsums are
  dimension-agnostic to `N` and stay untouched; `item_mask` becomes
  `[B, c]` (stored-out segments), the exact-softmax arm and cross-talk
  diagnostic inherit the new item set for free. Eval mirrors
  (`_read_once_linear`, boundary probes) call the same builder — no separate
  work expected; verify with the boundary-k test in chunk mode.
- **Config:** `read_state_segment_items: bool = False` in `read_linear:`
  (GradMemGPTConfig param + docstring + model getattr; runner argparse field
  + pass-through). Validation: requires `use_linear_state_read=True`; inert
  (ignored) outside it is fine, but assert it is not combined with anything
  that assumes token items (nothing does today — state that in the docstring).
- Default `False` ⇒ all existing behavior bitwise unchanged.

## Manifest arm (add when implemented — the file above reserves its slot)

```yaml
  # arm3: chunk-granularity keys (W_seg·M_grad per segment, d_h = 64)
  - name: interface_softmax_segkey64
    command: "python run_from_config.py --config"
    base_config: "configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml"
    overrides:
      read_linear.read_state_exact_softmax: true
      read_linear.read_state_heads: 2
      read_linear.read_state_segment_items: true
      model.base_model: llama
      dataset.data_name: N8-K2V2-V62-S4-ordered_pair_cross_1M
      dataset.data_path: "/home/jovyan/dzhivelikian/test_time_gd/data/N8-K2V2-V62-S4-ordered_pair_cross_1M"
    grid:
      training.seed: [4378, 43894, 2343]
```

## Tests (extend `test_grad_memgpt_linear_state_read.py`)

1. `test_segment_items_state_equals_explicit` — chunk-mode μ == explicit
   `Σ_s (q·k_s) v_s` over segments (float64, stored-out chunks masked).
2. `test_segment_items_exact_softmax` — matches manual per-token softmax over
   the c chunk keys.
3. `test_segment_items_default_unchanged` — default False: logits bitwise
   equal to the pre-change interface arm on a fixed input.
4. `test_segment_items_boundary_probe` — boundary-k probe == truncated
   forward in chunk mode (EM flags identical).
5. Config-guard test for the new validation.

## Phase plan & acceptance

- **P1 implement** → tests 1–5 green, whole file green.
- **P2 verify** — full suite (`uv run --extra dev pytest`); manifest dry-run
  now shows 9 runs (6 existing + 3 new); dry-run of the interface config with
  the new override clean.
- **P3 docs** — knob into TOKENWISE_READ.md § config surface (one line);
  STATUS entry (implementation + manifest arm added); PROGRESS updated.
- **P4 submit** — same discipline as INTERFACE_CONDITIONING_TASK.md Phase 4
  (interactive shell, flock guard, fan-out audit; one submission at a time;
  early gate on `read_state_*` metrics). Arm3 rides with arms 1–2 in
  `manifests/kv_interface_softmax.yaml`.
- **Done when:** suite green; 9-run dry-run clean; job submitted past the
  early gate; journal updated. Pre-registered reading for arm3 (record at
  submission): open outcome — chunk addressing removes per-token routing;
  if arm3 ≈ arm2, per-token keys are unnecessary (simpler storage, one
  address per chunk); if arm3 < arm2, token-level addressing is load-bearing.

## Out of scope

Bank-prefix work, top-k routing, `read_state_segment_emb`, writer changes,
the full-context sanity control (separate item in PROGRESS).
