# TASK — resume the interference curve's `interface_softmax` chains only (N32 → N64 → N96)

Executor brief. User directive 2026-11-10: after the 2026-11-03 double
infrastructure failure, resume ONLY the `interface_softmax` chains of the
interference curve. The `interface` and `adaptive` chains stay as they are
(partial results already analysed). Context and numbers:
[`reports/INTERFERENCE_CURVE_RESULTS.md`](../reports/INTERFERENCE_CURVE_RESULTS.md).

## What died and what "resume" means here

Both jobs were killed on 2026-11-03: an external interrupt (~11:00–11:30,
`InterruptedExperiment`) hit the interface N64 phases and the whole adaptive
job; the softmax chains — which ran AFTER the interface chains (scheduler
`schedule: sequential`) — then completed N8 and N16 cleanly (threshold stops)
and their N32 phases died at 18:15 with `OSError(28, 'No space left on
device')` at ≤2.5k of 50k steps.

`scripts/run_interference_chain.py` is idempotent: phases recorded `done` in
`runs/interference_curve/<arm>_<seed>.chain.json` are skipped on rerun; a
killed phase is rerun from step 0, warm-started (weights-only) from the
previous phase's best checkpoint (re-globbed fresh via
`resolve_prev_run_dir`, so stale absolute paths inside `done` records are
harmless). So a rerun of the wrapper with `chain.arm=interface_softmax`
executes exactly: skip N8, skip N16, run N32, N64, N96.

**The trap:** `load_state` silently falls back to `{'phases': {}}` on a
corrupt/unreadable state file — the chain would then RETRAIN FROM SCRATCH
(N8 onward) while looking like a resume. Verifying the state files
explicitly (step 2) is mandatory, not optional.

## Step 1 — remote disk diagnosis and remediation (the disk killed the last run)

On `ssh cloud-lolora` (see [`docs/REMOTE.md`](../docs/REMOTE.md) for env
discipline):

1. `df -h` the home filesystem; record free space.
2. `du -sh /home/jovyan/dzhivelikian/.cryri/run_* | sort -h` — dead jobs
   leave their `.venv` behind (the `rm -r .venv` cleanup never runs on a
   killed job); REMOTE.md explicitly sanctions deleting those. Expect
   several ~10–20 GB leftovers from the killed 10-29 and 11-02 jobs (plus
   any older ones). Delete ONLY `.venv` dirs inside `.cryri/run_*` copies of
   jobs that are certainly dead (verify via `cr status <hash>` or creation
   time; never touch running jobs). Record what was deleted and the freed
   space.
3. Quantify what filled the disk in the first place (checkpoints are tiny —
   the model is ~4L/128d; suspect .venv copies, caches, artifact syncs, or
   other users on the shared FS). Ensure the free space comfortably covers
   the new job: its own `.venv` + small checkpoints + sync-backs for a run
   of up to ~8 days. Record the before/after numbers.

## Step 2 — locate and verify the dead job's chain state (resume depends on it)

1. Find the curve job's run copy: `ls -td
   /home/jovyan/dzhivelikian/.cryri/run_* | head` — expected
   `run_2026_10_29_1748_*` (it contains `runs/interference_curve/`; the
   adaptive job's copy is `run_2026_11_02_1458_*`).
2. For each of `interface_softmax_{4378,43894,2343}.chain.json`:
   `json.load` it (a parse failure = STOP and repair, see below), and check
   phases `N8` and `N16` have `status: done`. Record their `best_em` —
   expected ≈ 0.991–0.993 (matches the Comet numbers).
3. For each seed, verify the N16 run dir exists under the copy's
   `runs/N16-K2V2-V62_1M/interf_interface_softmax_N16/run_<seed>_*/` and
   contains at least one `checkpoint-*/model.safetensors` (this is N32's
   warm-start). Record sizes.
4. **Corruption case:** if any state file fails to parse or lacks `done`
   N8/N16, do NOT submit (silent from-scratch retrain). Repair instead:
   rebuild the record from the run dir's `trainer_state.json`
   (`read_phase_metrics` shows the schema) and say so in the report.

## Step 3 — stage the resume artifacts into the remote project dir

The job copy (made at submit from the project dir) includes `runs/` (it is
NOT in `exclude_from_copy`), while the laptop tar-sync excludes it — so
artifacts are copied WITHIN the remote, from the dead run copy into
`/home/jovyan/dzhivelikian/test_time_gd/runs/`:

- `runs/interference_curve/interface_softmax_{4378,43894,2343}.chain.json`
  (the 3 state files — exact paths);
- the softmax `N8` and `N16` run dirs
  (`runs/N{8,16}-K2V2-V62_1M/interf_interface_softmax_N{8,16}/run_<seed>_*/`,
  whole dirs). Do NOT copy the killed N32 partials (the rerun creates fresh
  dirs; `resolve_prev_run_dir` takes newest-mtime per seed, so stale
  partials are harmless but useless).
- all 9 chain state files (tiny) so the interface/adaptive resumes stay
  possible.
- **Archive the dead copy's whole `runs/` tree** as a tarball OUTSIDE the
  project dir (e.g. `~/archives/interference_curve_runs_20261103.tar.gz`) —
  future interface-N64/adaptive resumes would need their N32 best
  checkpoints, and the `.cryri` copies are not safe long-term. Keep the
  project dir's `runs/` lean (the job copy carries it): record `du -sh` of
  what a new submit will copy.

## Step 4 — new manifest + new run spec (new files only; nothing existing is modified)

1. **`manifests/kv_interference_softmax_resume.yaml`** — model it on
   [`manifests/kv_interference_curve.yaml`](../manifests/kv_interference_curve.yaml):
   ONE experiment (`name: interf_interface_softmax`, same `command:
   python scripts/run_interference_chain.py`, `base_config` placeholder
   pointing at the new manifest itself, `overrides: chain.arm:
   interface_softmax`), `grid: chain.seed: [4378, 43894, 2343]`,
   `max_parallel: 3`, `schedule: sequential`. Header comments: resume
   semantics (what this job runs and skips, the state-file dependency, the
   staging done in step 3), a pointer to the disk remediation evidence, and
   the pre-registered predictions below VERBATIM (house discipline:
   predictions land in the header BEFORE submission).
2. **`run_softmax_resume.yaml`** — copy of `run_adaptive.yaml` with only:
   command → `manifests/kv_interference_softmax_resume.yaml`; description →
   `interference-curve softmax chains resume (N32-N96)`. Same comet project
   `gradmem-linear-read-interference`, same flock guard (untouched), same
   `run_from_copy`/`exclude_from_copy`.

## Step 5 — verification before submission

1. `uv run python run_scheduler.py manifests/kv_interference_softmax_resume.yaml --dry-run`:
   exactly 3 chain commands, all `chain.arm=interface_softmax`, correct seeds.
2. **Local resume smoke (the load-bearing check; RTX 3050 is enough).** In
   the local repo (local `runs/` mutations are harmless — the tar-sync
   excludes `runs/`), place the real state files and the real softmax N8/N16
   run dirs pulled from the remote under the same paths, then run the
   wrapper for ONE seed with miniature knobs, e.g.
   `chain.phases=3 chain.max_steps=2 chain.per_device_batch_size=8`. Assert:
   (a) it prints `phase 1/2 … already done … skipping` for N8 and N16;
   (b) the printed N32 child command contains
   `init_checkpoint=<the real N16 run dir>`;
   (c) the 2-step N32 run completes and its step-0 eval EM is ≥ 0.85 —
   expected ≈ 0.92–0.95, matching the killed N32 runs' step-0
   (0.918/0.925/0.945): direct proof the right weights loaded and the read
   works. (Note: `base_model=llama` is the tiny from-scratch Llama-arch
   model; if the local HF cache lacks the config, mind REMOTE.md's 401
   postmortem — HF_TOKEN must be in the env.)
3. Full pytest suite green (`uv run --extra dev pytest`) — no code changes
   are expected, this is the house rule.

## Step 6 — submission (REMOTE.md discipline, unchanged)

Remote data check (`data/N{32,64,96}-K2V2-V62_1M` exist in the project dir);
delete local dry-run `*.checkpoint.json` first; tar-sync (keep every
exclude, esp. `*.checkpoint.json` and `runs`); submit ONE job from
`run_softmax_resume.yaml` through `bash -i -c` with the env sourced; fan-out
audit (submit-timeout spawns duplicates — the flock guard makes them
harmless, but audit `cr jobs` anyway); record the job id. Then the **early
gate** if it starts in time (~1–2k steps of the first N32 phases): step-0 EM
≈ 0.92–0.95, eval_loss falling, zero tracebacks, fresh
`runs/N32-.../interf_interface_softmax_N32/run_<seed>_<uid>` dirs with the
right seeds, `read_state_*` metrics logging, `df` headroom stable.

If the shell cannot get the cluster env (HF_TOKEN/COMET — REMOTE.md's 401
postmortem), STOP after step 5, leave everything submission-ready, and
report the exact submit command — acceptable cycle outcome; do not retry
submissions.

## Pre-registered predictions (paste VERBATIM into the manifest header, before submission)

- **P-S1 (N32 completes):** warm-started from the N16 checkpoints (best EM
  0.991–0.992), the rerun N32 phases reach EM ≥ 0.99 within the 50k budget
  and stop on the threshold. Evidence: the killed N32 runs were at
  0.975–0.982 after ≤2.5k steps and rising; N16 crossed 0.99 at 5.5–9.5k
  steps from a comparable level. If N32 instead plateaus below 0.99, the
  truncated-Δ(32) reading must be revisited against the converged control.
- **P-S2 (N64 control holds — the write path is not the N=64 bottleneck):**
  warm-started from N32 ≥ 0.99, softmax N64 reaches EM ≥ 0.99 under the
  budget. Basis: per-segment write load is constant by design; the exact
  read's step-0 transfer was 0.93 at both N16 and N32. If softmax N64 also
  plateaus ≪ 0.99 (the interface arm plateaued at 0.41), the N=64 failure
  is protocol/write-bound, not superposition — Δ(64) becomes
  uninterpretable and the curve report must be revised.
- **P-S3 (N96, open):** if N64 holds, N96 runs; no directional bet — the
  control has no superposition mechanism; a failure there would indict
  write capacity or the 48-segment curriculum.
- **Δ(64) expectation given P-S2:** interface N64 plateaued at best-EM mean
  0.409 (EM and eval loss flat from ~10k to 23k steps); if softmax N64
  reaches ≥ 0.99, Δ(64) ≥ 0.58 — at N = 2·d_h the superposition cost is
  total, not gradual.
- **Fairness note:** identical protocol/curriculum/seeds as the original
  curve; each rerun N32 phase restarts from step 0 (weights-only warm start
  from the N16 best) — the killed partials are discarded, so all three
  seeds get a clean, identical N32 phase. The interface arm's N32/N64
  numbers it is compared against are unchanged.
- **Cost estimate:** N32 measured 2–3.7 s/step on the killed runs (threshold
  stop hoped ≤ ~10–15k steps → ≤ ~0.5 day); N64 at B=32 ≈ 9–10 s/step
  (interface measured 9.3), ≤ ~2.5 days if it stops by ~20k; N96 at B=16 is
  unmeasured — extrapolated ~1.5–2.5× N64 per step, up to ~5 days if it
  needs many steps. The 3 chains share one A100 (as the original job did);
  wall = slowest chain. Expected total ~3–8 days if N96 runs, ~2–4 days if
  it ends at N64. Known risk: the 2026-11-03 interrupt cause was never
  identified — if the cluster kills ~5-day jobs, N96 may be cut.

## Constraints

- New files only: the manifest and the run spec. Do NOT modify
  `kv_interference_curve.yaml`, `kv_interference_adaptive.yaml`, `run.yaml`,
  `run_adaptive.yaml`, the wrapper, the scheduler, or any source/config. If
  a runner fix turns out to be unavoidable, keep it minimal,
  default-preserving, unit-tested, and call it out in the report.
- Do not touch dead-job artifacts beyond the sanctioned `.venv` deletions,
  the staged copies, and the archive tarball. Never `cr kill` without
  checking the description (shared namespace). ONE submission.
- The `interface` and `adaptive` chains are NOT resumed (user decision;
  interface N96 would cost 10+ days at ~19 s/step extrapolated and its N64
  plateau already answers the question).

## Verification / definition of done

1. Disk remediation evidence (df/du before+after, what was deleted).
2. State-file verification output (3 files parse, N8/N16 done, best_em
   recorded) + staging evidence (paths copied, archive tarball path+size,
   `du` of the job-copy payload).
3. Dry-run output (exactly 3 chains) + local resume smoke evidence (skip
   lines, N32 command with init_checkpoint, step-0 EM) + pytest green.
4. Job id + submission audit + early-gate evidence if it started; expected
   finish date.
5. Report back with all of the above; also note for the LATER collection
   cycle: the Comet project will then contain TWO generations of
   `interf_interface_softmax_N32` runs (killed + rerun) — the collector must
   disambiguate by newest `end_ms` per (run_name, seed).

## Explicitly out of scope

Resuming the interface/adaptive chains, any analysis or collection (next
cycle), N=128, the (d′)-vs-(d) kernel comparison at N ≥ 64, top-k routing
(the planned mitigation, separate design cycle), any write-side change.
