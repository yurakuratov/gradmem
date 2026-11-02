# TASK — interference curve vs N on single-key KV, via checkpoint-chained curriculum

Executor brief. Implements PROGRESS.md missing-experiment #1 (interference
curve vs N) under the protocol fixed by the user on 2026-10-26:
**vanilla single-key KV-retrieval** (NOT ordered-concat), curriculum over
`N ∈ {8, 16, 32, 64, 96}` pairs with **explicit checkpoint chaining** (train
N=8, warm-start N=16 from the N=8 checkpoint, and so on — the "safer"
variant of the embed curriculum mechanics; it lets each phase carry its own
batch/OOM settings), and a **fair stopping protocol: max 50k steps per phase,
early stop at eval exact_match ≥ 0.99**.

## Question (pre-registered)

Does the superposed linear-attention state (`S = Σ k vᵀ`) lose accuracy as
the memory bank grows, relative to exact softmax over the unfolded bank? At
N=8 the two are equal at convergence (ladder, EM 0.995–1.00;
[READ_LADDER_RESULTS.md](../reports/READ_LADDER_RESULTS.md)); superposition
cost sample efficiency only. This curve finds where (if anywhere) that
changes.

Design invariant (why this isolates interference): **per-segment load is
constant** — 2 pairs per segment into M=2 memory tokens, pair-aware
segmentation, so segments = N/2 and the bank = N stored tokens. Only
cross-talk between stored items grows.

## Arms (3; same write path everywhere)

| arm | config | deltas vs ladder |
|---|---|---|
| `exact_softmax` (c) | `configs/gradmemgpt/kv_retrieval/linear_state_read.yaml` + `read_linear.read_state_exact_softmax: true` | none — the no-superposition control (O(N) read) |
| `linear_all` (d) | `configs/gradmemgpt/kv_retrieval/linear_state_read.yaml` | none — the superposed state (O(1) read) |
| `hopfield` (a) | `configs/gradmemgpt/kv_retrieval/hopfield.yaml` | the ladder `hopfield_control` overrides: `gradmem.use_mem_proj false`, `gradmem.mem_proj_mode none`, `gradmem.add_inner_loss_to_outer false`, `gradmem.n_mem_tokens 2` |

The interface arm (d′) is deliberately NOT run in cycle 1: on single-key the
superposition mechanism is shared between (d) and (d′), and (d) is the
proven config on this task (regression arm, EM 0.997+). (d′) can be added at
the binding N later. Seeds: **4378, 43894, 2343** (house standard).

## Phase table (per chain = one arm × one seed)

| phase | dataset (`data/`, exists locally) | `segmentation.n_segments` | init |
|---|---|---|---|
| 1 | `N8-K2V2-V62_1M` | 4 | scratch |
| 2 | `N16-K2V2-V62_1M` | 8 | phase 1 |
| 3 | `N32-K2V2-V62_1M` | 16 | phase 2 |
| 4 | `N64-K2V2-V62_1M` | 32 | phase 3 |
| 5 | `N96-K2V2-V62_1M` | 48 | phase 4 |

Common per-phase overrides: `model.base_model llama`,
`training.max_steps 50000`, `training.stop_on_em_threshold 0.99` (runner arg,
default 1.0 — config-only), `training.early_stopping_patience 50` (flat-run
backstop: 50 evals × 500 steps = 25k steps without improvement; note the
config's default 250 could never fire under a 50k cap),
`training.warmup_steps 10000` for phase 1 **and 1000 for phases ≥ 2**
(warm-starting a converged model needs little warmup; 10k would burn the
budget), `training.eval_steps 500` (config default), explicit top-level
`run_name` = `interf_<arm>_N<N>` (deterministic per phase; avoids auto-name
collisions with the ladder's single-key regression arm on the same datasets —
its runs live under `runs/N8-K2V2-V62_1M/` too).

Chain semantics: phase k+1 sets `init_checkpoint` to the previous phase's run
dir `runs/<data_name>/interf_<arm>_N<N>/`, resolved by the runner through
`run_<seed>_<uid8>/` → best/newest `checkpoint-*/model.safetensors` (the
figure10 mechanism — `manifests/figure10_memory_vs_K_vs_kv_pairs.yaml`
header documents it; **verify the resolution picks the current seed's run**
when the dir holds three seeds, and prefers the newest uid on retry/rerun).
Weights-only warm start (no optimizer state) — pin and verify. M=2 is
constant, so no mem slicing occurs. If a phase ends without reaching 0.99,
CONTINUE the chain anyway (the curve records the shortfall; "stop the chain
for all arms" is an analysis-time rule, not a runtime one). If a phase
process crashes, stop that chain and let the others proceed.

## Delivery mechanism: chain wrapper + manifest (the scheduler has NO dependency support)

`run_scheduler.py` cannot express "run B after run A finishes" (sequential
dequeues in order but does not wait). Therefore:

1. **New `scripts/run_interference_chain.py`** — takes the scheduler's
   positional `key=val` overrides (`chain.arm`, `chain.seed`) and executes
   the 5 phases of that (arm, seed) chain by calling `run_from_config.py`
   per phase with the overrides above. It resolves the previous phase's run
   dir itself after each phase (glob `runs/<data>/interf_<arm>_N<N>/run_<seed>_*`,
   newest mtime). It writes a chain state file
   `runs/interference_curve/<arm>_<seed>.chain.json` (per phase: status,
   best EM, steps, wall time, run dir) and **skips already-completed phases
   on rerun** (idempotent resume — cheap insurance given the submission-
   storm history). Per-phase per-device batch may be adjusted for OOM (next
   section) — encode the chosen sizes in the phase table inside the script.
2. **New `manifests/kv_interference_curve.yaml`** — 3 experiments (one per
   arm), `command: "python scripts/run_interference_chain.py"`, `grid:
   chain.seed: [4378, 43894, 2343]`, `schedule: sequential`,
   `max_parallel: 3` (three seeds of one arm in parallel; ladder ran 6
   concurrent runs at N=8, and phases spend most time at small N).
   No `base_config` (the wrapper owns it).

## OOM care (the reason the user chose explicit checkpoints)

Context length and the per-step inner loop (segments × K grad steps) grow
~linearly with N; phase 5 has 12× the N=8 context and 12× the segments.
Before submitting: run a **local GPU memory probe** of each arm's N=96
config (a few steps, gpt2 backbone is fine for the probe, batch 64, and
also at batch 32/16). If N=96 at batch 64 does not fit, set
`per_device_batch_size` smaller at the large-N phases and keep
`total_batch_size: 64` via gradient accumulation (verify the runner derives
accumulation from total/per-device). Record the probe numbers in the report
and the chosen per-phase batch table in the chain script.

## Pre-registered predictions (paste VERBATIM into the manifest header, before submission)

- **P1 (control):** `exact_softmax` reaches EM ≥ 0.99 at every phase through
  N=96 under the 50k budget. If it fails at some N, that phase's reading is
  protocol/write-bound, not superposition — recorded as such; the other arms'
  results at that N are read against the degraded control, not against 1.0.
- **P2 (the question):** with Δ(N) = (exact_softmax EM − linear_all EM),
  mean over seeds at each phase's best EM: superposition **binds** at the
  first N with Δ(N) > 0.05; if Δ(N) ≤ 0.02 through N=96, superposition
  **holds** at this scale (follow-up: extend to N=128, dataset exists).
  `read_state_cross_talk` should rise with N in `linear_all`; its trajectory
  is part of the readout.
- **P3 (context, direction open):** `hopfield` (convex one-shot mixture) may
  hold (⇒ single-key-at-scale does not discriminate one-shot from per-level
  reads; the discriminating follow-up is concat at the binding N) or degrade
  earlier than `linear_all` (⇒ per-level reads also win the scaling axis).
  Either outcome is informative; no directional bet.
- **Fairness note:** curriculum means later phases start warm. Arm
  comparisons at each N remain fair (identical curriculum per arm); absolute
  N-readings mix curriculum with interference — by design (the
  deployment-relevant notion is growing the bank with a trained model).
- **Cost estimate:** 45 runs total (3 arms × 5 phases × 3 seeds), 9 chains;
  phase 1 from scratch, later phases warm-started; bounded by the 0.99 stop,
  the 50k cap, and patience 50. Expected ~4–7 days wall on the A100.

## Constraints

- **No write-path changes; no default-behavior changes.** Everything is
  config/overrides plus the two new files (chain script, manifest). If a
  runner fix turns out to be unavoidable (e.g. `init_checkpoint` per-seed
  resolution), keep it minimal, default-preserving, unit-tested, and call it
  out in the report.
- Do not modify other manifests (`kv_interface_softmax.yaml` etc.) or the
  scheduler core.
- Comet project: set `COMET_PROJECT_NAME: gradmem-linear-read-interference`
  in run.yaml — a NEW project for this experiment family (user directive
  2026-10-26; keeps the ladder's `gradmem-linear-read` collection clean).
  Later collection passes the project explicitly to
  `scripts/collect_comet_results.py`.
- Predictions land in the manifest header BEFORE submission (house
  pre-registration discipline).

## Verification / definition of done

1. `python run_scheduler.py manifests/kv_interference_curve.yaml --dry-run`:
   exactly 9 chain commands, correct arms/seeds.
2. **Chain smoke (local GPU):** a 2-phase miniature (N8→N16, gpt2 backbone,
   2 steps per phase, tiny batch) proving (a) phase 2 actually loads phase
   1's weights (assert via a param checksum before/after init), (b) the
   state file marks phase 1 done and skips it on an immediate rerun,
   (c) `stop_on_em_threshold`/`max_steps` overrides arrive in the child
   command. Full pytest suite green (`uv run --extra dev pytest`).
3. **Memory probe** results recorded (section above).
4. **Remote:** check `data/N{16,32,64,96}-K2V2-V62_1M` exist on the remote
   (`ssh cloud-lolora`; fetch with `scripts/download_kv_retrieval.sh` there
   if missing — data/ is excluded from both the tar sync and the job copy);
   tar-sync per `docs/REMOTE.md` (keep `--exclude='*.checkpoint.json'`,
   delete any local dry-run `*.checkpoint.json` first); retarget `run.yaml`
   command to `manifests/kv_interference_curve.yaml`, description to
   `interference-curve single-key curriculum`, and `COMET_PROJECT_NAME` to
   `gradmem-linear-read-interference` (it still points at the finished
   conditioning manifest); flock guard untouched; ONE submission
   from a shell with the env; fan-out audit; then the **early gate** at
   ~1–2k steps: eval_loss falling, zero tracebacks, correct
   `interf_<arm>_N8` dirs with the right seeds, `read_state_*` logging in
   the linear arms.
5. Report back: job id + copy name, manifest path, dry-run output, smoke and
   probe evidence, remote data check, early-gate evidence, expected finish.

If the shell cannot get the cluster env (HF_TOKEN/COMET — see REMOTE.md's
401 postmortem), STOP after step 3, leave everything submission-ready, and
report the exact submit command instead — that is an acceptable cycle
outcome; do not burn time retrying submissions.

## Explicitly out of scope

Collecting results (days later; separate cycle), the (d′) interface arm,
ordered-concat at larger N, N=128, any write-side change, top-k routing.

---

## Revision 2 (2026-10-26, user-directed): re-arm to interface reads; resubmit

The user overrode the cycle-1 arm set after submission (the pending job is
cancelled and replaced). **Arm selection is the user's call; the PI proposes
options and asks.**

- **CANCEL the pending cycle-1 job** (identify ours by description
  "interference-curve single-key curriculum" in `cr jobs`/the platform UI;
  never touch jobs not positively ours). It has not started training
  (pending in queue); nothing of value is lost.
- **New arm set (user decision):**
  - `interface` — base `configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml`
    (linear kernel, per-token keys, d_h=32; the deployable (d′), the
    ladder's fastest most seed-consistent solver)
  - `interface_softmax` — the same base + `read_linear.read_state_exact_softmax: true`
    (exact softmax over the unfolded bank at the SAME interface sites and
    two-pass conditioning — the no-superposition control; this is arm1 of
    `manifests/kv_interface_softmax.yaml` riding the curriculum)
  - `hopfield` dropped (fails concat — uninteresting baseline). `linear_all`
    and the all-positions exact-softmax variants are DEFERRED to a later
    cycle, not cancelled — keep their definitions in the chain script's arm
    table so a follow-up manifest can run them without code changes.
- Everything else is unchanged: phase table, protocol (50k / 0.99 /
  patience 50 / warmup 10k→1k), chaining mechanics, run names
  `interf_<arm>_N<N>`, comet project `gradmem-linear-read-interference`,
  run.yaml target/description.
- **New required verifications** (the interface arms have NEVER run on
  single-key data — cycle 1 smoked only `linear_all`):
  1. interface-arm chain smoke (2-phase miniature, gpt2, 2 steps × 2
     phases): two-pass conditioning works on single-key labels; warm-start
     and resume asserted as before.
  2. memory probe for BOTH new arms at N=96 (the interface read is
     two-pass — re-measure; do not reuse the cycle-1 table) → set the
     per-phase batch table from the new numbers.
  3. Replace the manifest header's predictions with the Revision-2 text
     below, verbatim, before resubmission; dry-run = exactly 6 chains
     (2 arms × 3 seeds).
- **Resubmit ONE job** (same REMOTE.md discipline, flock guard, fan-out
  audit). Report the new job id; if it sits pending in the queue at report
  time, that is acceptable — the early gate then happens when it starts.

### Revision-2 pre-registered predictions (verbatim into the manifest header)

- **P1 (control):** `interface_softmax` (exact softmax over the unfolded
  bank at the interface sites) reaches EM ≥ 0.99 at every phase through
  N=96 under the 50k budget. If it fails at some N, that phase's reading is
  protocol/write-bound, not superposition — recorded as such.
- **P2 (the question):** with Δ(N) = (interface_softmax EM − interface EM),
  mean over seeds at each phase's best EM: superposition **binds** at the
  first N with Δ(N) > 0.05; if Δ(N) ≤ 0.02 through N=96, superposition
  **holds** at this scale (follow-up: extend to N=128, dataset exists).
  `read_state_cross_talk` should rise with N in `interface`; its trajectory
  is part of the readout.
- **P3 (kernel, soft expectation — no strong bet):** at N=8 the kernel
  difference should be small (convexity does not bite at M specialized
  interface sites — the `kv_interface_softmax.yaml` arm1 pre-registration);
  any divergence between the kernels is expected to appear only where
  superposition begins to bind.
- **Fairness note:** unchanged from revision 1 (identical curriculum per
  arm; warm starts by design).
- **Cost estimate:** 30 runs (2 arms × 5 phases × 3 seeds), 6 chains;
  expected ~3–5 days wall on the A100 once started.

---

## Revision 3 (2026-11-02, user-directed): carried-memory baseline arm as a SEPARATE job

The main question the curve must answer is **whether separate memory tokens
per segment mitigate catastrophic forgetting** — so the carried-memory
baseline (every segment writes into the SAME M tokens) runs on the identical
curriculum, as its own cluster job alongside the running interference job.

- **Arm `adaptive`** (user-selected, from
  [`grad_memgpt_adaptive.py`](../grad_memgpt_adaptive.py)): the adaptive
  fork's native cross-segment memory carry, **baseline configuration —
  `memory_update_rule: sgd`** (the plain gradient inner loop; NO mamba/convex
  gated recurrence, NO STAR, NO replay), memory prefix read as-is. Write
  hyperparameters matched to the curve's other arms where the config accepts
  them: `n_mem_tokens 2`, `K 2`, `inner_lr 0.04`, `grad_mode second`,
  `use_write_head true`. Segmentation identical: pair-aware, 2 pairs per
  segment, `n_segments = N/2` per phase (the carry across those N/2 segments
  into the same 2 tokens is the mechanism under test).
- **Budget asymmetry is the design, record it:** 2 carried tokens total vs
  N stored tokens for the interface arms — the comparison measures the
  architecture axis (same-token carry vs per-segment slots), not matched
  capacity.
- **Checkpoint chaining, NOT the embedded `curriculum:` section** (the user
  called the checkpoint variant safer; also uniform with the other arms).
  No `curriculum:` block anywhere in the arm's config.
- **Protocol identical:** phase table N ∈ {8,16,32,64,96}, 50k steps/phase,
  `stop_on_em_threshold 0.99`, patience 50, warmup 10k (phase 1) → 1k
  (phases ≥2), seeds 4378/43894/2343, `per_segment_eval: true` — **the
  forgetting matrix is a PRIMARY readout for this arm** (it directly
  visualizes carry-overwriting as N grows; 48×48 at N=96 is fine).
- **Delivery: separate manifest + separate run spec.** New
  `manifests/kv_interference_adaptive.yaml` (ONE experiment `interf_adaptive`,
  grid seed ×3 = 3 chains via the same wrapper — add the arm to the wrapper's
  ARMS table) and new `run_adaptive.yaml` (copy of run.yaml: same env, same
  comet project `gradmem-linear-read-interference`, same flock guard — the
  `.job_lock` is per-run-copy so the two jobs cannot collide; command targets
  the new manifest; description "interference-curve carried-memory baseline
  (adaptive sgd)"). Do NOT modify `run.yaml` or the running job's manifest.
  `max_parallel: 3`.
- **Verification:** adaptive chain smoke (2-phase miniature, gpt2, 2 steps:
  runs on single-key, warm-start checksum, resume-skip, forgetting-matrix
  artifact written, `stop_on_em_threshold` honored); memory probe at N=96 for
  the adaptive arm (carried inner loop over 48 segments — new numbers, set
  the batch table); full suite; dry-run = exactly 3 chains. Then tar-sync
  (checkpoint-json excludes), submit ONE job from `run_adaptive.yaml` per
  REMOTE.md; fan-out audit; do NOT touch the running interference job
  (`ccde14ed`). Early gate if it starts in time.
- **Base config derivation:** start from the `adaptive:` section structure of
  `configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml` (segmented carry
  setup) with the mamba/gated bits removed (`memory_update_rule: sgd`) — or
  `adaptive_stage0.yaml` if that is closer to the vanilla setup; pick
  whichever is the cleaner baseline and say so in the report. No `curriculum:`
  block, no STAR keys.

### Revision-3 pre-registered predictions (verbatim into the new manifest header)

- **P-A1 (the expected failure):** the carried arm does NOT reach EM ≥ 0.99
  even at small N under the 50k budget (fact 1: the running-sum inner loop
  overwrites earlier segments; at 4 segments the 2026-09-10 forgetting
  matrices already showed degradation, while the per-segment-write arms sit
  at 0.99+). Forgetting matrices should show diagonal ≫ last-column gaps
  that widen with each phase.
- **P-A2 (the readout):** the per-phase gap to the `interface` arms is the
  effect size of "separate memory tokens mitigate forgetting". If the
  carried arm unexpectedly HOLDS (EM ≥ 0.99 through N=96), fact 1 is
  contradicted at matched protocol — a major surprise requiring
  investigation before any H1 claim (protocol differences vs 2026-09-10
  would be the first suspect, not H1's failure).
- **Fairness note:** 2 carried tokens vs N stored tokens is by design
  (architecture axis, not matched capacity); identical curriculum, protocol,
  and seeds as the interface arms.
- **Cost estimate:** 15 runs (3 seeds × 5 phases), 3 chains; if P-A1 holds,
  phases end by patience (~25–30k steps) rather than the cap — expect
  ~1–2 days wall on its own GPU.
