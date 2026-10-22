# TASK — query-blind interface baseline (`read_state_interface_conditioning`) + remote submission

Executor brief. Small, well-scoped change on top of the implemented
linear-state read, plus one cluster job. Work the phases in order; the
submission (Phase 4) must not disturb the ladder job that is currently
running. No conversation context is assumed.

## Context (read first, in this order)

1. [TOKENWISE_READ.md](TOKENWISE_READ.md) — the read architecture spec,
   especially the **Variant** section (the interface-token two-pass design)
   and its "placement / causality trap" discussion.
2. [STATUS.md](../STATUS.md) entries 2026-10-18 (b) implementation and (c)
   review — the code you are modifying and its verification bar.
3. [PROGRESS.md](../PROGRESS.md) — the running read ladder (missing
   experiment #1) and where this ablation fits (the missing-experiments
   item on the interface conditioning ablation).
4. [REMOTE.md](REMOTE.md) — the cluster workflow you will use in Phase 4.
   Read it INCLUDING the submission gotchas: they were earned through two
   postmortems (STATUS.md 2026-10-19, 2026-10-19 (b)) — interactive-shell
   env, `*.checkpoint.json` tar-exclude, API-timeout fan-out, flock guard.

**What and why.** The interface arm (`read_sites: "interface"`) currently
conditions its M learnable interface tokens on the query in a first pass
(target-span blinded), then answers in a second pass. This task adds the
**query-blind cell of the conditioning ablation**: a mode that skips pass 1
entirely and uses the raw learnable `M_0` as the pass-2 interface embeddings.
Without pass 1 the interface tokens sit at the left of the answering pass
and can never see the query, so their per-level state reads fetch
query-agnostic content — the "left-prefix causal" configuration the spec's
causality trap section describes.

**Pre-registered prediction (record it in STATUS.md at submission time,
before results exist):** the query-blind arm underperforms the conditioned
interface arm on BOTH the two-key concat exact-match and single-key
exact-match, because retrieval degrades to query-agnostic compression
through the fixed M·d channel. Either outcome is informative — prediction
holds ⇒ conditioning is load-bearing and the two-pass machinery is
justified (an empirical demonstration of the causality trap); prediction
fails ⇒ pass 1 can be dropped, halving the interface arm's read compute.

**Interim ladder context (2026-10-20, unverified, results not yet
collected):** the linear-state arms reportedly do very well on
ordered-concat while the bank-prefix control fails — so the read-head
designs are the live story and understanding *why* the interface arm works
(conditioning or not) is on the critical path. This ablation is the
cheapest way to answer it.

## Ground rules

- Same invariants as the implementation task: write side untouched,
  flag-off/default behavior bitwise identical (default
  `read_state_interface_conditioning: true`), full existing suite green
  (`uv run --extra dev pytest`).
- Follow the repo style (see the existing linear-state read code and its
  tests). New config knob ⇒ docstring in `GradMemGPTConfig` + config-yaml
  comment.
- Maintenance rule: journal the change + the submission in `STATUS.md` and
  update `PROGRESS.md` in the same change.
- **Do not touch the running ladder job** (`9a6c630c` or its successor —
  no `cr kill`, no edits under the remote `.cryri/` copies, no resubmit of
  `manifests/kv_linear_read.yaml`). The new job is a separate submission.
- **Submission discipline** (from the two postmortems): submit ONLY from an
  interactive shell carrying the env; never blind-resubmit after a submit
  timeout — audit `cr jobs` for fan-out duplicates first (the flock guard
  in `run.yaml` makes duplicates self-exit, but they must be identified and
  the wedged ones stopped via the UI); never `cr kill` in the shared
  namespace without checking the job description.

## Scope

**In:** the knob + skip-pass-1 implementation, tests, a manifest arm, local
verification (dry-runs + tests; short GPU smoke if a GPU is available),
spec/PROGRESS/STATUS doc updates, and the remote submission of the new arm.

**Out:** anything about pass-1 internals (conditioning-only vs
conditioning+reads is a *different*, later cell), bank-prefix arm, top-k
routing, writer changes, touching `manifests/kv_linear_read.yaml` (the
running job's manifest — leave it and its `.checkpoint.json` alone).

## Implementation contract

**Config.** `read_state_interface_conditioning: bool = True`:
- `GradMemGPTConfig.__init__` param + docstring + `GradMemGPT.__init__`
  `getattr` (default True — old configs must load unchanged).
- Validation: `read_state_interface_conditioning=False` requires
  `use_linear_state_read=True` and `read_state_read_sites == "interface"`.

**Code** — `grad_memgpt.py`, `_linear_state_read_logits`, interface branch:
when conditioning is off, skip the pass-1 forward entirely and set
`iface = m0` (the expanded `M_0`); pass 2 is unchanged (same layout, site
mask, slicing). `return_interface=True` must return the raw `M_0` broadcast
in this mode (tests use it). `_read_once_linear` already threads
`labels`/`target_mask` for pass-1 blinding — in this mode they are simply
unused; keep the signature. Stats: only pass 2 accumulates now (pass 1
contributed before); no other stats changes. Hooks/ctx machinery untouched.

**Plumbing.** `run_from_config.py` flattens every `read_linear:` key
automatically (no change expected — verify). The runner
(`run_gradmemgpt_on_kv_retrieval.py`) needs the new argparse field +
pass-through to the config (mirror `read_state_read_sites`). Check
`generate_run_name.py` only if you add the knob to a new config's
`run_naming` keys (recommended: `icond` prefix, so run dirs are
distinguishable: `...icondFalse...`).

**Tests** — extend `test_grad_memgpt_linear_state_read.py`:
1. `test_interface_no_conditioning_uses_raw_m0` — with the knob off,
   `return_interface` gives exactly `M_0` (broadcast), and a spy on the
   first instrumented block confirms the pass-2 interface positions' inputs
   are `M_0`; also: exactly ONE read forward fires (no pass-1 forward —
   count block-0 invocations; with conditioning on it is two).
2. `test_interface_no_conditioning_query_blind` — the inversion of the
   existing `test_interface_query_sensitivity`: perturbing non-target
   query tokens leaves the interface-position hidden states (and logits)
   unchanged.
3. `test_interface_conditioning_default_unchanged` — default True keeps
   current behavior (the existing interface tests already pin this; add a
   config-default guard).
4. Config-guard test: the new validation asserts fire.

**Manifest.** New file `manifests/kv_interface_conditioning.yaml`: ONE
experiment (`linear_interface_nocond`), base config
`configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml`,
override `read_linear.read_state_interface_conditioning: false`, same
model/dataset overrides as the ladder's `linear_interface` arm (llama,
ordered-concat dataset path), grid seeds `[4378, 43894, 2343]`,
`max_parallel: 3`. Analysis pairs these runs against the ladder's
`linear_interface` runs (same seeds, same data) — do NOT duplicate the
conditioned arm in this manifest.

## Phase plan

### P1 — Implement (config, code, plumbing)
**Exit:** the four tests above pass; full file green.

### P2 — Verify locally
`uv run --extra dev pytest` (whole suite green);
`run_from_config.py --config configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml read_linear.read_state_interface_conditioning=false --dry-run` clean;
`run_scheduler.py manifests/kv_interface_conditioning.yaml --dry-run` clean (3 runs).
If a local GPU is free: a 2-step smoke of the nocond arm (batch 8) — must
train, log `read_state_*` metrics, and write a forgetting matrix.

### P3 — Docs
Spec: add the knob to the § Proposed config surface table and a one-line
note in the Variant section that the conditioning ablation's query-blind
cell is implemented (link the manifest). PROGRESS: fold into missing
experiment #2 (conditioning ablation) with the task link. STATUS: entry
covering the implementation + the pre-registered prediction (exact wording
from the Context section above) — written BEFORE submission.

### P4 — Submit the remote job (follow [REMOTE.md](REMOTE.md), incl. its gotchas)
1. Check the ladder job's state (`bash .cr_env.sh status 9a6c630c` or the
   current job id from STATUS.md). **Safe default: wait until it finishes**
   (the ablation is 3 runs and the ladder has priority); submitting
   concurrently is acceptable only if you have verified the flock-guarded
   `run.yaml` is intact remotely and you audit for fan-out afterwards.
2. Tar-sync the repo to `/home/jovyan/dzhivelikian/test_time_gd` with the
   REMOTE.md recipe — it now excludes `*.checkpoint.json` (a local dry-run
   checkpoint tar-synced into the project dir once aborted the scheduler
   on a manifest-hash mismatch). Never sync a `.venv` or `.venv.*`.
3. Edit `run.yaml`'s command to
   `... run_scheduler.py manifests/kv_interface_conditioning.yaml ...`
   (it currently points at the ladder manifest — this is the documented
   manifest-selection mechanism). Do NOT remove the `flock` guard prefix of
   the command.
4. Submit from an interactive shell with the env (the documented recipe):
   `ssh -o RemoteCommand=none cloud-lolora 'bash -i -c "source
   /home/jovyan/dzhivelikian/.dzhivelikian.bashrc; cd
   /home/jovyan/dzhivelikian/test_time_gd; bash
   /home/jovyan/dzhivelikian/.cr_env.sh submit run.yaml -y"'`.
   If the client prints a submit TIMEOUT/error: do NOT resubmit — audit
   `cr jobs` (duplicates may spawn minutes late; identified by
   description/creation time), confirm which job holds the manifest, kill
   only confirmed duplicates/wedged spawns via the UI, and only then decide.
5. **Early gate** (first eval, ~step 500–1000, from the run copy's logs):
   `read_state_gate_abs_mean` not collapsing to 0,
   `read_state_mu_over_x_norm_ratio` in a sane band (~0.05–0.5), loss
   decreasing. If it fails the gate: kill THIS job only (verify it is not
   the ladder job first), journal the failure, and stop for review — do
   not debug-tune on the cluster.
6. Update STATUS: job hash + early-gate verdict + expected finish.

**Exit:** job submitted and past the early gate; STATUS carries hash,
prediction, and gate verdict.

## Definition of done

1. Whole suite green (existing + the 4 new tests).
2. Dry-runs clean; manifest dry-run shows exactly 3 runs.
3. Default (`true`) behavior bitwise unchanged (existing interface tests
   prove it).
4. Job submitted, past the early gate, hash journaled; pre-registered
   prediction recorded before any result could exist.
5. Spec/PROGRESS/STATUS updated; link check over touched markdown.

## Reporting

Files touched; test names; dry-run outputs; job hash + early-gate metrics;
any deviation from this brief (with reason) journaled in STATUS. Do not
analyze the finished results — collection and the paired comparison against
the ladder's `linear_interface` arm are a separate task once the runs
complete.
