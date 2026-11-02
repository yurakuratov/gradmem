# STATUS — journal of additions, decisions, and pending work

Newest entry first. Each entry: date, what changed, decisions taken (with
rationale), verification status. Current state and hypotheses live in
`PROGRESS.md`; uncommitted directions in `IDEAS.md`.

**Maintenance rule:** every merged change updates this journal and, when it
changes the evidence picture, `PROGRESS.md` in the same commit.

## 2026-11-02 — cycle 1: theoretical comparison of carried writes and linear-state interface reads

**Milestone reached (analysis only).** Research-scientist code inspection
confirmed that the linear interface baseline still uses gradient writes:
each segment starts from the same learned initial tokens, writes independently,
and appends its result to a bank. READ constructs `S_h = sum k_i v_i^T` once
from that bank; incremental additive state writes are an algebraically
equivalent interpretation, not the current execution strategy.

**Framework:** separate (1) modification of earlier encodings by subsequent
writes, (2) interference when reading superposed encodings, and (3) the
query-conditioned interface's ability to extract the required information.
For fixed model parameters, unclipped carried SGD has local state Jacobian
`I - eta H`; independent additive writes have identity state Jacobian.
The latter removes cross-segment gradient rewriting of earlier encodings,
but does not guarantee old-answer retention: at a fixed head query the new
read contribution is `sum_new (q dot k_i) v_i`, and later-layer queries can
also change. The random-isotropic-key noise estimate grows as
`sqrt((N-1)/d_h)` only under explicit independence/centering assumptions.
Gradient descent itself is not inherently destructive: a linear regression
delta update protects old orthogonal key directions exactly.

**Comparison constraints:** at d=128, H=4 the matrix state has 4096 scalar
coordinates versus 256 for M=2 carried tokens (other adaptive M choices
change this ratio). Current code also retains the growing bank and projected
K/V; fixed-size state does not imply constant total forward memory. Both
interface passes read the state. Match state budgets, segment boundaries,
write steps, and read interfaces before attributing an accuracy gap to the
write operator. Proposed diagnostics: old-probe loss changes versus write
gradient alignment, segment-order sensitivity, key-overlap versus retrieval
error, and interference scaling with N and d_h. No experiments launched,
tests run, source/config changes, or new performance claims. Existing run
status was not checked. Stop after cycle 1; no further cycle needed.

## 2026-10-29 — cycle 2: interference curve re-armed to interface reads; resubmitted, EARLY GATE PASSED

**User-directed re-arm (protocol note: arm selection is the user's call — the
PI proposes options and asks).** The cycle-1 pending job was stopped by the
user personally. New submitted arms: `interface` (linear kernel, per-token
keys, d_h=32 — the deployable (d′) and the ladder's fastest solver) +
`interface_softmax` (exact softmax over the unfolded bank at the SAME
interface sites and two-pass conditioning — the no-superposition control;
arm1 of [`kv_interface_softmax.yaml`](manifests/kv_interface_softmax.yaml)
riding the curriculum). `hopfield` dropped (fails concat — uninteresting
baseline); `linear_all`/all-positions `exact_softmax` deferred, definitions
kept in the wrapper's ARMS table. Revision-2 predictions P1–P3 verbatim in
the manifest header (Δ(N) = interface_softmax − interface EM; superposition
binds at first Δ > 0.05, holds if Δ ≤ 0.02 through N=96; cross-talk
trajectory part of the readout). Everything else unchanged from cycle 1
(phases, 50k/0.99/patience-50/warmup 10k→1k, chaining, comet project
`gradmem-linear-read-interference`).

**Verification — first-ever interface runs on single-key.** Chain smoke
(`interface`, 2 phases N8→N16, gpt2, 2 steps): two-pass conditioning clean
on single-key labels; phase-2 warm-start PROVEN by checksum (sha256 of
model-at-init restricted to checkpoint keys == sha256 of phase-1's
resolved `model.safetensors`; 78/78 shared tensors bitwise-equal; runner
log shows the load); idempotent resume skips completed phases.
`interface_softmax` phase-1 smoke clean. Suite **82 passed**. Memory probe
re-measured for the two-pass arms (N96/B16 ≈ 3.6 GiB, ~0.235 GB/sample;
batch table 64/64/64/32/16 with total_batch 64 via accumulation; A100
projection < 20 GB for 3 concurrent chains). Remote data N8–N96 verified
present. Stale cycle-1 scheduler checkpoint deleted before sync.

**Submitted and RUNNING — job `ccde14ed`
(`lm-mpi-job-ccde14ed-f57a-409a-8b5c-a90c188e7e02`, copy
`run_2026_10_29_1748_358129`), early gate PASSED at step ~2.7k/50k of
phase 1:** eval_loss 2.969 → 2.767 (steps 2k → 2.5k), zero tracebacks,
`runs/N8-K2V2-V62_1M/interf_interface_N8/run_{4378,43894,2343}_*` with the
right seeds, `read_state_*` logging live — **`read_state_cross_talk` 0.117
→ 0.169 and rising (the P2 readout trajectory is already recording)**.
`interface_softmax` chains start after arm 1 finishes (sequential
schedule); its single-key kernel path is smoke-covered locally. One job at
submit time (fan-out audit clean); expected ~3–5 days wall once running.

**Adjacent problems noticed, NOT fixed (house habits cover them):**
(1) `run_from_config.py --debug` silently drops the `segmentation:` YAML
section — the runner's programmatic flatten list lacks it; debug-mode-only
quirk, the production CLI path is unaffected. (2) Local dry-runs silently
write `manifests/<name>.checkpoint.json` (the known tar-exclude gotcha) —
delete before any sync, as done here.

## 2026-10-26 — cycle 1: interference curve vs N submitted (single-key, checkpoint curriculum)

**Protocol (user-fixed)** — vanilla single-key KV-retrieval (NOT concat),
curriculum `N ∈ {8,16,32,64,96}` via **explicit checkpoint chaining** (phase
k+1 warm-starts from phase k's best checkpoint — the user-preferred variant
of the embed curriculum mechanics, allowing per-phase batch/OOM control),
fair stopping = **50k steps/phase max, early stop at eval EM ≥ 0.99**
(`stop_on_em_threshold` runner arg, config-only), patience 50 as flat
backstop, warmup 10k (phase 1) → 1k (phases ≥ 2). Arms: (c) exact-softmax,
(d) linear-state all-positions, (a) hopfield one-shot control; seeds
4378/43894/2343; 45 runs in 9 chains. Design invariant: 2 pairs/segment into
M=2 tokens (segments = N/2, bank = N tokens) — only stored-item cross-talk
grows. Brief: [`tasks/INTERFERENCE_CURVE_TASK.md`](tasks/INTERFERENCE_CURVE_TASK.md);
pre-registered predictions P1–P3 + fairness note + cost estimate pasted
VERBATIM into the manifest header BEFORE submission.

**Decision — chain wrapper, not scheduler deps** — `run_scheduler.py` has no
dependency support (sequential dequeues in order but does not wait), so each
(arm, seed) chain is ONE scheduler run of the new
[`scripts/run_interference_chain.py`](scripts/run_interference_chain.py),
which sequences the 5 phases itself (per-seed run-dir resolution by newest
mtime, weights-only warm start, idempotent resume via
`runs/interference_curve/<arm>_<seed>.chain.json`; a sub-0.99 phase still
chains — the curve records the shortfall). Manifest
[`manifests/kv_interference_curve.yaml`](manifests/kv_interference_curve.yaml)
(9 chain commands; `base_config` is a documented placeholder the wrapper
ignores). Local smoke chain (gpt2, 2 steps × 2 phases) verified warm-start +
resume (`runs/interference_curve/smoke_linear_all_4378.chain.json`); memory
probe on real N96 batches → per-phase batch 64/64/64/32/16 with
total_batch 64 via grad accumulation (A100 projection < 15 GB for 3
concurrent chains; probe table in the manifest header).

**Decision — (d′) interface arm deferred from the curve** — rationale: (d)
and (d′) share the superposed state (the mechanism under test) and differ
only in read sites; (d) is the single-key-proven config (ladder regression
arm EM 0.997+) while (d′) never ran on single-key, so an untested arm
warm-chaining through phases risks confounding arm-specific warm-up trouble
with interference. Counterweight recorded: (d′) is the deployable read and
concentrates state extraction on M·L taps, so it could bind earlier than
(d); its chain is one wrapper entry + one manifest experiment (zero new
machinery) and is the natural follow-up — at the binding N per P2, or as a
full chain before any deployment claim.

**Comet** — new project `gradmem-linear-read-interference` in
[`run.yaml`](run.yaml) (user directive; keeps the ladder's
`gradmem-linear-read` collection clean; later collection passes the project
explicitly to `scripts/collect_comet_results.py`).

**Submitted — PENDING in queue** (submission confirmed by the user
2026-10-26; long queue ahead; job id to be recorded at first monitoring
pass). run.yaml → `manifests/kv_interference_curve.yaml`, description
"interference-curve single-key curriculum", flock guard intact. The early
gate (eval_loss falling, zero tracebacks, `interf_<arm>_N8` dirs with the
right seeds, `read_state_*` logging) is due when the job STARTS — not yet
checked. **Caveat:** the executor's report was lost when its agent run was
cancelled after submission; the verification above is from the artifacts
themselves (manifest header, chain script, dry-run = exactly 9 chains, smoke
state file, run.yaml retarget) — consistent with the brief's definition of
done except the pytest-suite run and the remote data presence check, which
are unconfirmed. Expected ~4–7 days of GPU time once started.

## 2026-10-25 — research workflow: PI ⇄ research-scientist hierarchy

**Changed** — two-level research workflow installed. New subagent
`.zcode/agents/research-scientist.md` (GLM-5.3-Flash) owns all coding,
testing, and run execution; `AGENTS.md` gained a leading "Research
Workflow" section defining the PI (main chat, GLM-5.3) role, the 5-step
workflow cycle (formulate → brief → delegate → review → record), and hard
stop conditions (5 completed cycles max, or the cycle-1 milestone reached,
whichever comes first; no autonomous 6th cycle). A project-agnostic twin
of the agent definition at `gradmem/.zcode/agents/research-scientist.md`
(added together with the gradmem_rl copy; the brief names the target repo)
keeps the subagent dispatchable when ZCode is opened at the gradmem root
instead of inside `test_time_gd`.

**Decision** — the PI never edits source or runs experiments directly; it
owns direction, briefs, evidence review, and the research record
(STATUS/PROGRESS/IDEAS/AGENTS). Delegation goes through the Agent tool
(`research-scientist`), one self-contained brief per dispatch; long briefs
follow the `tasks/` convention. Rationale: spend expensive-model reasoning
on direction and cheap-model tokens on execution, and bound autonomous
runs by an explicit cycle budget so control returns to the user at a
milestone or after 5 cycles.

## 2026-10-22 (c) — documentation reorganization: docs / tasks / reports

**Changed** — markdown layout per the three-folder convention:
`docs/` = guides, derivations, designs (TOKENWISE_READ, ADAPTIVE_MEMORY,
STAR — moved from root, CURRICULUM, REMOTE); `tasks/` = executor briefs
(TOKENWISE_READ_TASK, INTERFACE_CONDITIONING_TASK, SEGMENT_KEY_TASK — moved
from docs/); `reports/` = results & evidence (READ_LADDER_RESULTS + its
`figures/` and `results/` asset dirs — moved from docs/). All moves via
`git mv` (rename-tracked; `docs/results/` was untracked, plain mv).
Cross-references updated everywhere: root working files, docs/tasks/reports
cross-links, manifest-header path mentions (incl.
`manifests/kv_interface_conditioning.yaml`), AGENTS.md's STAR link, and the
asset default paths in `scripts/collect_comet_results.py` /
`make_em_figure.py` / `make_read_schematics.py` (`docs/results|figures` →
`reports/...`). Historical STATUS path mentions updated to the new
locations (text unchanged). Link check green across all md files.

**Decision** — root keeps only the working files (README, PROGRESS, STATUS,
IDEAS, AGENTS); new results land in `reports/`, new executor briefs in
`tasks/`, designs and guides continue in `docs/`. Matches the gradmem_rl
two-folder convention plus the explicit tasks/ split.

## 2026-10-22 (b) — interface read: kernel × key-geometry cells (manifest + chunk-key task)

**Architecture fact (from the design discussion, worth recording):** the
linear-state read stores per-**token** items (N = segments × M) with per-head
slices of one shared κ, shared across layers — NOT one key per memory chunk
(the original Hopfield model). The linear mode also already skips
`_compress_segment`: keys already derive from `M_grad`. The remaining
untested geometry axis is chunk-vs-token granularity.

**Added** — [`manifests/kv_interface_softmax.yaml`](manifests/kv_interface_softmax.yaml)
(6 runs, config-only): arm1 `interface_softmax` (exact-softmax kernel at
interface sites — completes the kernel × site matrix) and arm2
`interface_softmax_h64` (`read_state_heads: 2` ⇒ key dim 64 = the old
hopfield key dim). Pre-registered predictions in the manifest header: arm1 ≥
(d′) at (c)-like convergence (convexity should not bite at M specialized
interface sites); arm2 within noise of arm1. And
[`tasks/SEGMENT_KEY_TASK.md`](tasks/SEGMENT_KEY_TASK.md) — the small task for
arm3 `interface_softmax_segkey64`: chunk-granularity items (N = segments),
keys `normalize(W_seg·M_grad)` per head (d_h 64 via heads=2), values
per-head `ν_seg: M·d → d` (pinned rationale: the additive bias μ must be
d-dimensional — a retrieved `M·d` chunk cannot be reshaped into prefix
tokens in the bias read); knob `read_state_segment_items`, 5 named tests,
manifest entry reserved in the manifest header, submission rides arms 1–2.
Arm3's reading is open: chunk addressing ⇒ simpler storage (one address per
chunk); if it loses to arm2, token-level addressing is load-bearing.

## 2026-10-22 — LADDER + CONDITIONING-ABLATION RESULTS: every per-token read solves concat; query-blindness kills the interface read (pre-registered prediction HOLDS)

**Collected** — all 21 experiments (7 arms × 3 seeds) of
`zhekahauska/gradmem-linear-read` via the new
[`scripts/collect_comet_results.py`](scripts/collect_comet_results.py)
(summaries: `reports/results/results.json`, curves: `reports/results/
em_curves.json`). Stopping protocol (user): runs reaching EM 1.0 ended at
their threshold (exact-softmax, 47–76k steps); every other run manually
stopped after >80k steps — the failing arms are flat with no trend at their
stop points. **Report (with schematic read-method diagrams + EM-curve
figure):** [`reports/READ_LADDER_RESULTS.md`](reports/READ_LADDER_RESULTS.md);
figures regenerable via `scripts/make_read_schematics.py` /
`scripts/make_em_figure.py`.

**Best two-key both-values EM, ordered-concat (mean over seeds 4378/43894/
2343):** hopfield one-shot 0.248; bank-prefix control 0.024; exact-softmax
1.000 (all seeds, by 19.5–29k steps); linear-state all-positions 0.997;
interface conditioned **0.999**; interface query-blind **0.012**.
Single-key regression arm (linear-state config on single-key data): EM
0.997–0.9986.

**Pre-registered prediction verdict — HOLDS.** The query-blind arm
underperforms the conditioned interface on every seed (Δ > 0.98: 0.0026 vs
0.9988, 0.0202 vs 0.9998, 0.0126 vs 0.9986). Mechanism as predicted: the
blind arm's reads stay active and calibrated (gate 0.18–0.25, ‖μ‖/‖x‖ and
β like the conditioned arm) but the fixed M·d channel carries
query-agnostic content — per-value EM ≈ 0.04–0.14, joint ≈ v1·v2 ≈ 0.01,
eval loss never fits (0.67–1.18 vs 0.0003–0.0011). The causality trap is
now empirically demonstrated; **pass 1 conditioning is load-bearing and the
two-pass machinery is justified.** Honest gap: the prediction's single-key
clause is untested directly (the ablation manifest has no single-key-data
arm by design); the nearest proxies (probe-format forgetting matrices,
`two_key_both_single_exact_match` 0.013–0.279 → 0.0004–0.0012) both collapse
for the blind arm.

**Ladder decisions (spec § Experimental plan):** (b) fails harder than (a)
while (c)/(d)/(d′) hit 1.00/0.997/0.999 on the SAME stored tokens ⇒ the
bank-prefix failure indicts the prefix interface, not write quality
(confirming the 2026-10-20 interim reading; H2 stands). (d) vs (b): the
folded O(1) state solves what the O(N) prefix cannot. (c) vs (d): exact
attention converges ~1.5× faster; superposition costs sample efficiency, not
accuracy at N = 8 bank items (interference-vs-N curve decides at scale).
(d′) vs (d): M·L query-conditioned interface reads match all-position reads
(most seed-consistent solver: 30.5–31.5k steps to 0.99) — per-token reads
are NOT needed, given pass 1.

**Caveat recorded** — the per-segment forgetting matrices probe with
single-key `?!K:V!|` queries, which concat-trained arms never see in
training: for these arms the matrices measure format transfer, not
forgetting (exact-softmax seed spread 0.34–1.00 diag is variance, not loss;
no arm degrades within training). A concat-format forgetting probe is the
honest instrument going forward (added to PROGRESS missing experiments).

## 2026-10-20 (b) — query-blind interface baseline implemented (`read_state_interface_conditioning`)

**Added** — `read_state_interface_conditioning: bool = True` in
[`grad_memgpt.py`](grad_memgpt.py) per
[`tasks/INTERFACE_CONDITIONING_TASK.md`](tasks/INTERFACE_CONDITIONING_TASK.md):
in the interface read mode, `false` skips the pass-1 conditioning forward
ENTIRELY and answers pass 2 with the raw learnable `M_0` (the query-blind
"left-prefix causal" cell of the conditioning ablation — the spec's
causality-trap configuration, run deliberately). `_linear_state_read_logits`
hoists `m0`/`mem_ones` and branches: conditioning on = the unchanged two-pass;
off = `iface = m0`, one read forward instead of two, `return_interface`
returns the raw `M_0` broadcast, labels/target_mask unused (signature kept),
stats accumulate from pass 2 only. Validation: `false` requires
`use_linear_state_read` AND `read_state_read_sites == "interface"`. Plumbing:
runner argparse field + pass-through (mirrors `read_state_read_sites`);
`run_from_config.py` flattening needed no change (verified). Manifest
`manifests/kv_interface_conditioning.yaml`: ONE experiment
`linear_interface_nocond` (interface config, override
`read_linear.read_state_interface_conditioning: false`, llama + the ladder's
ordered-concat dataset path, seeds 4378/43894/2343, `max_parallel: 3`),
paired at analysis time with the ladder's `linear_interface` runs — the
conditioned arm is NOT duplicated. Write side untouched; flag-off/default
behavior unchanged.

**Fix (latent, found by the GPU smoke)** — `run_from_config.py` forwards a
top-level `run_name_suffix` as a CLI flag (line ~193) but the runner's
`ExperimentArgs` never declared it, so the FIRST config/manifest to use the
suffix would have died in HfArgumentParser ("Some specified arguments are not
used"). No existing config used it — the new manifest is the first. Added the
optional `run_name_suffix: Optional[str] = None` field (the runner reads the
suffix from the merged yaml via `generate_run_name`; the field only makes the
forwarded arg parseable).

**Run-naming deviation** — the brief suggested a `run_naming` `icond` key so
dirs show `...icondFalse...`; `generate_run_name._format_value` skips
False-valued keys by pinned design ("negative flags don't add information"),
so that route would have left the blind arm UNNAMED while renaming any future
conditioned rerun (`icond` for True). Used a manifest-level
`run_name_suffix: icondFalse` override instead: exactly the recommended
`...icondFalse...` dir suffix, zero naming drift for every other arm
(dry-run-verified: `ns4_m2_K2_hopfield_lsr_sitesinterface_rh4_icondFalse`).

**Tests** — `test_grad_memgpt_linear_state_read.py` +4 (21 total):
`test_interface_no_conditioning_uses_raw_m0` (exactly ONE read forward via a
block-0 invocation count — flipping the knob back on gives two; `return_interface`
IS `M_0`; pass-2 block-0 inputs at the interface positions are `M_0` + wpe),
`test_interface_no_conditioning_query_blind` (inversion of the query-sensitivity
test: non-target query perturbation leaves the interface embeddings AND the
interface-position logit row bitwise unchanged — with teeth: answering rows
change), `test_interface_conditioning_default_unchanged` (config default True;
an old config WITHOUT the attribute loads via the getattr default; pass 1
transforms M_0), `test_interface_conditioning_config_guards` (both asserts
fire; valid cell constructs). Full suite: 82 passed
(`uv run --extra dev pytest`). Dry-runs clean: `run_from_config.py` with the
override (flag flattened, name suffixed), `run_scheduler.py
manifests/kv_interface_conditioning.yaml --dry-run` = exactly 3 runs. GPU
smoke (2 steps, batch 8, gpt2 smoke dataset): the nocond arm trains, logs
`train_/eval_read_state_*`, and writes `forgetting_matrix_step*.json/csv`
(artifacts under `runs/default/..._icondFalse_smoke`, log
`/tmp/kv_smoke_logs/interface_nocond.log`).

**Pre-registered prediction (recorded BEFORE submission, no results exist
yet):** the query-blind arm underperforms the conditioned interface arm on
BOTH the two-key concat exact-match and single-key exact-match, because
retrieval degrades to query-agnostic compression through the fixed M·d
channel. Either outcome is informative — prediction holds ⇒ conditioning is
load-bearing and the two-pass machinery is justified (an empirical
demonstration of the causality trap); prediction fails ⇒ pass 1 can be
dropped, halving the interface arm's read compute.

**Submitted — job `9fa451fc`** (`lm-mpi-job-9fa451fc-f9db-47b5-8852-ee7cb256fcd4`,
submitted 2026-10-20 17:51 from an interactive shell with the env, clean
single-job response; run copy `.cryri/run_2026_10_20_1751_e1bc5e`; the ladder
job `9a6c630c` was left untouched and Running — concurrent submission
approved by the ladder's owner). Pre-submission remote checks: tar-synced
with the REMOTE.md recipe (incl. `*.checkpoint.json` exclude; a LOCAL
dry-run checkpoint the new scheduler had written was deleted, not synced);
remote `run.yaml` verified (flock guard intact, command →
`manifests/kv_interface_conditioning.yaml`, description
"interface-conditioning-ablation nocond", md5 parity on
grad_memgpt.py / runner / run_from_config.py / manifest, no `.venv`, no
stale checkpoints). Fan-out audit: exactly one job created at submit time;
three later jobs (#192–194, 18:00–18:12) could NOT be positively identified
as ours via the CLI (no description exposed) and were left alone per the
don't-stop-unidentified rule — if any are our late spawns, the flock guard
self-exits them before `uv sync`; ours holds the lock and is Running.

**Early gate (step ~1000–1400, all 3 seeds `linear_interface_nocond__{0,1,2}`,
from the run copy's logs): PASS.** `read_state_gate_abs_mean` ≈ 0.1002 (init
0.1 — no collapse); eval_loss falling on every seed (≈4.97 → ≈4.39 / 4.41 /
4.52); zero tracebacks; the nocond flag confirmed live in the run commands
and the exp dirs carry the `icondFalse` marker.
`read_state_mu_over_x_norm_ratio` ≈ 0.002 — below the ~0.05–0.5 watchlist
band as an ABSOLUTE value, but calibrated against the ladder's own
`linear_interface` arm this is its normal early trajectory (that arm read
0.00066 at the same age and only ~0.022 after ~24 h, while reportedly solving
concat; the band evidently describes the mature regime) — the nocond arm sits
~2–3× HIGHER than its conditioned sibling at the same age and is rising.
Not a gate failure; collected results remain the decider. Expected finish:
~3–4 days at ~1 step/s if run to 300k max steps; earlier per-run if
early_stopping_patience (250 evals) fires. **`run.yaml` currently points at
`manifests/kv_interface_conditioning.yaml`** — flip it back to
`manifests/kv_linear_read.yaml` before any ladder resubmission.

## 2026-10-20 — query-blind interface baseline: executor brief (+ interim ladder note)

**Interim observation (single-source, results not yet collected/verified):**
in the running ladder (job `9a6c630c`), the linear-state arms reportedly do
very well on ordered-concat while the **bank-prefix control fails
completely**. Two consequences if it holds across seeds: (1) linear read
beating the hopfield control with the identical write path is a clean
causal isolation of the read interface (H2's mechanism); (2) bank-prefix
failing is *against* its pre-agreed role as the recall upper bound — and
since (d) succeeds with the SAME stored tokens, its failure indicts the
prefix interface (frozen reader + raw-token prefix at training scale), not
write quality. The decision tree's step-1 inference ("(b) fails ⇒ H2
rejected") is therefore unsound as stated; treat (b) as measuring the
prefix interface, not write quality. Collect before concluding.

**Added** — [`tasks/INTERFACE_CONDITIONING_TASK.md`](
tasks/INTERFACE_CONDITIONING_TASK.md): executor brief for the third cell of
the interface conditioning ablation — `read_state_interface_conditioning:
false` skips the pass-1 conditioning and answers with raw `M_0` (the
query-blind "left-prefix causal" configuration from the spec's
causality-trap analysis). Contract: config knob + validation, skip-pass-1
in `_linear_state_read_logits` (`return_interface` returns `M_0`), 4 named
tests (raw-M₀ use incl. a one-read-forward count, inverted
query-sensitivity, default-unchanged guard, config guards), new manifest
`manifests/kv_interface_conditioning.yaml` (3 seeds, paired with the
ladder's `linear_interface` runs), and a phased remote submission aligned
with the postmortem discipline (interactive-shell env, `*.checkpoint.json`
tar-exclude, flock guard preserved, fan-out audit after any submit
timeout, ladder job untouched). **Pre-registered prediction (to be
recorded at submission, before results):** the query-blind arm underperforms
the conditioned interface on both two-key and single-key EM; either outcome
is informative.

## 2026-10-19 (b) — submission postmortem #2: API-timeout fan-out; flock guard; ladder RUNNING

**What happened** — resubmitting per the (a) postmortem, every `cr submit`
13:36–14:05 printed "Failed to submit job: API error 0: Request failed after
4 attempts: Read timed out (read timeout=30)" yet the gateway had created a
job PER attempt server-side (4–8 observed; late spawns even minutes after the
client exited — the 13:48 pair, a 14:14 job), ALL pinned to the single
`.cryri/run_*` copy made before the client's retry loop. The concurrent
containers raced `uv sync` on the shared `.venv` (log evidence: `failed to
remove .venv/lib: Directory not empty`) and the scheduler then burned all 18
runs in ~40 s, each dying with `accelerate: command not found` /
`ModuleNotFoundError: accelerate.commands.launch`. Two storms (13:36, 13:46)
plus the 13:48 pair were stopped and their poisoned copies deleted; the
June-era `stage0` checkpoint leftover was also removed from the remote
project dir and the code re-synced + md5-verified (run.yaml, manifest,
grad_memgpt.py, run_scheduler.py, run_from_config.py identical
local↔remote).

**Fix — flock guard in `run.yaml`** (no retry/timeout knob exists in `cr
submit`): the job command now begins `exec 9>.job_lock && { flock -n 9 ||
{ echo DUPLICATE_SPAWN_LOCK_HELD_EXITING; exit 0; }; } && …`. First job to
start holds the lock; duplicate spawns exit 0 at ~40 s instead of racing the
venv; real failures still propagate their exit codes (only "lock held" maps
to 0). Verified live on the 14:05 submit: 5 jobs fanned out → 3 exited
cleanly within a minute, exactly 1 is running the ladder, 1 (`f23ce671`)
is wedged in worker boot and can only ever self-exit by the guard (stop it
manually via the UI if it lingers).

**Current state — the ladder is RUNNING** (first healthy submission): job
`9a6c630c` on copy `run_2026_10_19_1405_fe03b2` since ~14:07; env correct
(HF_TOKEN expanded — submitted from an interactive shell with the env, per
REMOTE.md); scheduler checkpoint shows 6 runs `running` (all arms, seed 0 —
run_scheduler parallelizes arms on the one GPU) + 12 `pending`; per-run logs
under `manifests/kv_linear_read.yaml.logs/` advancing with no tracebacks
(`bank_prefix_control__0` past its step-0 eval, eval_loss 6.12). A 14:14
Pending job of unknown provenance may be another late spawn — the guard
covers it if ours. Monitor: `bash .cr_env.sh status 9a6c630c`; do NOT
resubmit on a submit-timeout error without auditing `cr jobs` + the
scheduler checkpoint first. Also: never `cr kill` in the shared namespace
without checking the description — ours are identifiable in the platform UI.

## 2026-10-19 — E2: bank-prefix control arm (b) implemented; E1 ladder submission postmortem

**Added (E2)** — `use_bank_prefix_read` in [`grad_memgpt.py`](grad_memgpt.py):
ladder arm (b), the recall UPPER BOUND that decides H2. READ prepends ALL
stored per-segment memory tokens as an ordinary causal prefix
(`[read_st?, bank, read_end?, query]`; stored-out segments masked from
attention, cumsum position ids compress over masked slots) — no state, no
hooks, no mem_proj. Shares the linear-state mode's raw-token STORE
(`_raw_bank_store`); the read is `_bank_prefix_read_logits` /
`_read_once_bank_prefix`; `forward_per_segment_eval` serves it, so the
forgetting matrix works in this arm too. Config guards: mutually exclusive
with `use_linear_state_read` (and gated delta), requires
`hopfield_value_proj_dim=None`. Config `bank_prefix_read.yaml`; manifest
entries: `bank_prefix_control` in `kv_linear_read.yaml` + standalone
`kv_bank_prefix.yaml` (3 seeds, ordered-concat). Tests +4 (17 in
`test_grad_memgpt_linear_state_read.py`): predictions equal a manual causal
forward over `[bank, query]`; the outer gradient reaches `self.mem` through
plain attention (the arm's write-shaping signal; needs `grad_mode='first'` —
'none' detaches mem from the outer graph by design); `forward_per_segment_eval`
boundary-k == truncated forward (EM flags identical); config guards. Local
GPU smoke green (eval_loss 5.70 @ 2 steps, below the linear arms' 5.79 — the
prefix workspace helps immediately, as an upper bound should); full suite
78 passed.

**E1 submission postmortem** — the ladder (`run.yaml` →
`manifests/kv_linear_read.yaml`) took five failed submissions before the
current clean, queued state; every failure was submission-workflow, not
model code (the code behaved exactly as in the local smokes): (1) literal
`"$HF_TOKEN"` in the job env — `.dzhivelikian.bashrc` has an interactivity
guard (`case $-`), so sourcing it from non-interactive ssh sets nothing →
401 GatedRepoError on the gated Llama-3.2-1B config in all 15 runs (fix:
submit from a shell that actually has the env — interactive shell or a
dumped-env file); (2) a June-era broken `.venv` in the remote project dir
was partially copied into run copies → `uv sync` aborts with "not a valid
Python environment" (fix: moved out of the project dir; NOTE
`exclude_from_copy` matches `.venv` only, so any `.venv.*` sibling is NOT
excluded); (3) a LOCAL dry-run's `kv_linear_read.yaml.checkpoint.json`
tar-synced into the project dir → scheduler aborts on manifest-hash mismatch
after the manifest edit (fix: deleted remotely, `--exclude='*.checkpoint.json'`
added to the tar recipe in REMOTE.md); (4) duplicate near-simultaneous
submissions from hung ssh sessions raced one run copy (two jobs on one
directory: one's `rm -r .venv` cleanup breaking the other's `uv sync`) and
the shared-UV_CACHE race produced a venv missing safetensors (18/18 runs
dead in ~50 s each). Cache validated clean afterwards via a throwaway
`UV_PROJECT_ENVIRONMENT` sync (safetensors 0.5.3 + transformers 4.53.0
import fine). All gotchas are now in [`docs/REMOTE.md`](docs/REMOTE.md).
**Current state:** the remote project dir is submission-ready (no `.venv`,
no stale checkpoints, 18-run manifest incl. `bank_prefix_control`); the
clean 12:09 submission was cancelled in the queue during the manual cleanup;
no ladder job is running — resubmit (`bash .cr_env.sh submit run.yaml -y`
from a shell with the env) when the queue allows, ONE at a time, and verify
the job record before retrying.

## 2026-10-18 (c) — implementation review of 8b69397: verdict + one fix

**Reviewed** — the H3 implementation against
[`docs/TOKENWISE_READ.md`](docs/TOKENWISE_READ.md) /
[TOKENWISE_READ_TASK.md](tasks/TOKENWISE_READ_TASK.md). Verified in code:
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

**Added** — [`tasks/TOKENWISE_READ_TASK.md`](tasks/TOKENWISE_READ_TASK.md):
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
