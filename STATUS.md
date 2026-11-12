# STATUS — journal of additions, decisions, and pending work

Newest entry first. Each entry: date, what changed, decisions taken (with
rationale), verification status. Current state and hypotheses live in
`PROGRESS.md`; uncommitted directions in `IDEAS.md`.

**Maintenance rule:** every merged change updates this journal and, when it
changes the evidence picture, `PROGRESS.md` in the same commit.

## 2026-11-12 — interference curve: softmax chains resumed and collected; P1/P2 decided without caveats

**(a) Resume submission (2026-11-10, user-directed, cycle 1 of the resume
task; brief [`tasks/SOFTMAX_RESUME_TASK.md`](tasks/SOFTMAX_RESUME_TASK.md)).**
Only the `interface_softmax` chains were resumed. Remote disk remediation:
two dead-job `.venv` leftovers deleted (5.3G + 98M; the 12T share sat at
96%/559G free — biggest own-side consumer found: `.cryri/gradmem_rl` at 45G
and growing, separate live job, untouched). All 3 softmax chain state files
verified (parse; N8/N16 `done`, best EM matching Comet; N32 `running` — no
corruption, so no silent-retrain risk). Staged into the remote project dir:
9 chain state files + softmax N8/N16 run dirs (84M); dead job's whole
`runs/` tree archived at
`~/archives/interference_curve_runs_20261103.tar.gz` (343M). New files only:
[`manifests/kv_interference_softmax_resume.yaml`](manifests/kv_interference_softmax_resume.yaml)
(P-S1–P-S3 pre-registered verbatim before submission) +
[`run_softmax_resume.yaml`](run_softmax_resume.yaml). Verified: dry-run =
3 chains; local resume smoke on the RTX 3050 skipped N8/N16, warm-started
N32 from the real N16 checkpoint (step-0 EM 0.918 = the killed seed-4378
run's step-0 exactly; local gated-HF-config fallback used and disclosed —
no HF_TOKEN on the laptop, cache primed from the ungated mirror, model_type
verified, remote job unaffected); pytest 97 passed. ONE submission:
`lm-mpi-job-960c43c4-d04e-4c28-aac3-e1a6001d867c` (11-10 16:20), fan-out
audit clean. Incident during sync: a cwd reset made the first tar-sync
archive the parent `gradmem/` dir; caught, litter (~26M, all new paths)
removed, re-synced, byte-verified on the remote (no pre-existing file
overwritten).

**(b) Results collected (2026-11-12).** Job queued ~7 h, then ran clean
11-10 23:16 → 11-12 08:07 (~33 h): N32 reruns, N64, N96 — all threshold
stops, zero crashes. Collected with
[`scripts/collect_interference_results.py`](scripts/collect_interference_results.py)
(now 39 experiments; the collector disambiguates the two N32 generations by
newest end time). Report updated:
[`reports/INTERFERENCE_CURVE_RESULTS.md`](reports/INTERFERENCE_CURVE_RESULTS.md).

**Verdicts (pre-registered):** **P1 confirmed in full** — the exact-softmax
control reaches EM ≥ 0.99 at EVERY phase through N=96 (N96 crossed within
0.5–2.5k steps; best EM 0.9900–0.9934 everywhere). **P2: superposition
binds at N=32** with both sides converged: Δ = +0.0005 / +0.0025 /
**+0.065** / **+0.58** at N=8/16/32/64 (per-seed Δ(32) 0.073/0.080/0.044).
**P-S1/P-S2/P-S3 all held** (N32 8.5–11k steps; N64 4.5–8.5k; N96 0.5–2.5k).
Interface N64 stays truncated-but-plateaued (EM and loss flat 10k→23k);
interface N96 and adaptive N64/N96 never ran — Δ(96) undefined, judged
low-information. Warm-transfer step-0 EM: softmax 0.927/0.929/0.940/0.965 at
N=16/32/64/96 vs interface 0.865/0.688/0.254 — the exact read is
N-invariant, the superposed read collapses. Cross-talk floor saturates at
0.143 ≈ E|cos| for d_h=32 random unit vectors; the record's √((N−1)/d_h)
estimate crosses 1 exactly at the binding N.

**Open (recorded in the report's Next):** d_h=64 key-width arm on the
curriculum (the direct capacity test), top-k segment routing (binding
condition met), adaptive forgetting matrices from the archive. No decisions
taken on those.

## 2026-11-11 (d) — enlarged local-only comparison: EXP-LM-004

Cycle 1/5 completed the exact IID comparison against both a carried local
delta writer and a direct additive memory. All affordable integer local
widths are eligible; hierarchy also optimizes every divisor reset/commit
schedule and affordable integer global address size. The three methods share
error targets and a 2048-scalar state cap. Independent theory, implementation,
and runner audits preceded the one registered scientific execution.

**Theory:** exact spherical-moment recurrences predict endpoint delta error
and every eligible hierarchical snapshot schedule without calibration.
For fixed J and large p, delta error is (J-1)/(2p)+O(p^-2), versus additive
error (J-1)/p. Delta's smaller state is offset by more expensive writes;
the leading continuous small-error cost crossover is Q=J, with finite-error
and integer corrections. No universal memory-capacity claim follows.

**Evidence:** hierarchy wins 2/8 primary population scenarios, both J32,Q1.
At epsilon .10/.25, h=tau=4,D12,p39/16 costs 16054/6624, compared with
20460/8184 for the best local-only comparator (additive): 21.53%/19.06%
savings. State rises from 620/248 to 1014/416 scalars. Additive wins five
scenarios; local delta wins J16,Q32,epsilon .10. All other hierarchical
optima collapse to independent writes. No selected schedule uses tau<h.

**Validation:** 59 unique frozen configurations, three fresh seeds, and 1024
episodes/configuration yield 177 rows. Every method's reserve selections meet
deployment epsilon in 24/24 dependent views, including hierarchy's direct
sampled LA. Median/max absolute prediction residual is .00033276/.00772436.
Primary boundary selections are point-feasible in 14/24 delta, 12/24 additive,
13/24 hierarchy views; boundary sampling variation is reported separately.
Reserve median proxy overhead is 5.29%/5.56%/5.35%, respectively. No selection
was revised using held-out outcomes; exact optima are analytic, not empirical
full-grid oracles.

**Reproduction:** 76 focused/prior tests pass. Sealed prediction/selection
took 18.15s internally; evaluation took 26.42s. Selection SHA-256:
`d69b256e487807cffe3f9326262bc025ed5e40fef0b0e8212bee211b7d6029f3`.
Report: `reports/LOCAL_ONLY_COMPARISON_RESULTS.md`; data:
`reports/results/linear_memory/EXP-LM-004/`. Prior experiment files and user
notebook edits were preserved.

**Decision:** milestone reached; stop cycle 1 per AGENTS.md. The contribution
now includes a restricted hierarchy win against strong enlarged one-level
linear memories and explicit losing regimes. It is a compute/storage tradeoff
under an operation proxy with supplied IID addresses and oracle global routing.
Measured systems costs, deep memory/LoRA baselines, learned routing, and
transformer integration remain open. No next experiment was launched.

## 2026-11-11 (c) — refined allocation milestone: EXP-LM-003

Cycle 1/5 implemented and executed the preregistered refinement through the
research-scientist workflow, with independent theory, code, and interpretation
audits. Local width spans p={2,4,8,16,32,64}; all affordable integer D are
eligible. The independent baseline optimizes the same p,D domain. Writer,
2048-scalar folded-state cap, and operation proxy are unchanged. Three fresh
seeds use 4096 calibration groups and 1024 held-out episodes per local case.
Every selector uses the same predeclared 5% error reserve; primary feasibility
is evaluated at the original epsilon, with separate reserve diagnostics.

**Evidence:** unrestricted selection is feasible in 48/48 dependent scenario
views, with 4.81% median and 10.38% maximum regret versus the point oracle.
The registered coverage/median-regret criteria passed. All three policies
made feasible selections in all 48 views. Sharing wins 9 views; the substantive
results are IID,J32,Q1 in all three seeds: h4 endpoint commits cost 16.9–19.5%
less at epsilon.10 and 24.6% less at epsilon.25. Shared memory uses 1344/1408
versus 660 scalars, or 448 versus 268, respectively. This is a compute/storage
tradeoff within a common cap, not a Pareto improvement. J16,Q1,epsilon.25 has
only a nominal .17% gain. Independent writes win every Q32 comparison.

**Theory:** a fixed pre-run IID h2 witness at J32,p32,D12 has exact expected
error .09390797 and cost 19328. Every independent h1 writer with p>=2 meeting
epsilon.10 has continuous-relaxation cost at least 20912 at Q1. This proves
a 7.57% proxy-compute advantage without a D-grid artifact. All three fresh
reference 95% intervals contain the predicted error and lie below .10.
The empirical selector found still-cheaper h4 policies. Neither result is a
wall-clock, learned-routing, or universal one-level-memory claim.

**Uncertainty:** 41/48 unrestricted views contain a cheaper interval-uncertain
alternative to the interval-feasible oracle; exact rankings remain unresolved.
Only 40/48 meet the stricter .95*epsilon target empirically, while all 48 meet
epsilon. The reserve is not a confidence guarantee. Scenario views reuse
episodes; address realizations differ across h even in the common IID
population. The correlated generator defines latent blocks by reset horizon,
so changing h changes the workload distribution. Its losses characterize a
reset-conditioned stress-test family, not one fixed correlated stream.

**Verification:** 40 focused/prior tests passed, including moment variance,
integer selection, optimized baseline, boundaries, dependency drift, abstention,
and overwrite refusal. Calibration and evaluation completed in 5.4s and 82.8s
on CPU. The sealed selection hash is
`d97868b12e8ed1a6082f905c76bc373d5933e092b124c6ae143671a3aefec500`.
Execution dependency hashes match the final files. Prior experiment sources
and artifacts were preserved. Code lives in new refined-allocation modules;
report: `reports/REFINED_ALLOCATION_RESULTS.md`; data:
`reports/results/linear_memory/EXP-LM-003/`.

**Decision:** accept the restricted IID low-query crossover and stop cycle 1
at the declared milestone per AGENTS.md. Next compare against enlarged
local-only memory with fair width/budget access, then address measured cost
and transformer/LoRA baselines. No next study or transformer run was launched.

## 2026-11-11 (b) — held-out allocation milestone: EXP-LM-002

Implemented a standalone calibration/selection/evaluation pipeline through
the research-scientist workflow. The fixed delta-rule writer uses fresh
local addresses, independent reset groups, separately variable reset horizon
h and commit interval tau, and oracle global routing. The transfer law is
E_hat=delta+[A0+(J/h-1)B]/D, calibrated on short J=h groups before any
held-out J16/J32 episodes are evaluated. The registered cost is a mixed
operation proxy with a 2048-scalar folded-memory cap, not measured runtime.

**Evidence:** 1,944 candidate cases, 144 policy scenario rows, three seeds,
and two known geometry families. All 48 unrestricted selections met the
error targets, with zero finite-grid cost regret and median/max absolute
prediction error .000169/.00165. Every selected policy used h=tau=1, whose
expected error is exactly (J-1)/(pD). Both p4 comparator policies collapsed
to the same independent-write baseline. The successful selections therefore
do not establish a benefit from shared local trajectories. Scenario rows
reuse episodes across Q and error targets and are not independent replicates.

**Audit:** original sealed selections, outputs, and an archive of executed
sources were preserved. The original selection SHA-256 remains
`98bbe4e64d6eff40545028655569096cd2793e5c0b70242ac34ca54ecb40be1d`.
Post-result corrections add the imported model dependency to the hash guard,
oracle-feasible scenario coverage to prevent success through abstention,
and a sensitivity comparison against an interval-certified oracle. Direct
sampled LA is described as a consistency check: the conditional reference
uses the same global-key identity as the predictor. Corrections and the
unchanged-settings reproduction are separated into EXP-LM-002-AUDIT.

**Verification:** 30 focused/prior linear-memory tests passed. Audit and
original calibration, candidate predictions, and candidate metrics are
byte-identical; selections and all 40 common evaluation fields are equal.
Audit coverage is 48/48. The interval-certified oracle gives zero median/max
regret, 48/48 selected interval-feasible configurations, and no cheaper
interval-uncertain alternatives. The original point-oracle result remains
primary; these are approximate pointwise intervals, not a simultaneous
guarantee. Prior pilot source/config hashes remain unchanged.

**Direction:** a post-result two-fact derivation predicts that p16,h=tau2,D16
beats the original selected p2,h=tau1,D128 at J16,epsilon.10,Q1/Q32. However,
independent p2,D75 is cheaper still at Q1 and has expected error exactly .10.
Both are outside the original grid and neither enters its oracle or regret.
Next preregister refinement of BOTH local width and global address resolution
under the unchanged proxy; do not add a commit penalty just to favor sharing.
The report is `reports/ALLOCATION_TRANSFER_RESULTS.md`; derivation and
qualification are in PROGRESS. Transformer and convergence claims remain open.

**Decision:** cycle 1/5 complete; declared milestone reached. Stop this task
per AGENTS.md, retaining the refined-grid comparison as the next proposed
milestone. No transformer integration or expanded study was launched.

## 2026-11-11 — entry-level theory introduction

Added `docs/HIERARCHICAL_MEMORY_INTRODUCTION.md` as a self-contained reading
entry point. It defines the local delta-rule memory, separate local/global
addresses, reset and commit intervals, whole-snapshot additive writes,
oracle retrieval, error decomposition, expected-error capacity requirement,
and computation/storage costs. A numerical two-fact example shows both
benefit and harm from intermediate commits. The gradient-routing explanation
distinguishes inner writing from outer training and historical access from
differentiation through snapshots. A notation guide and transformer/LoRA
mapping remove the need to read implementation code.

Equations, edge cases, and pilot statements were checked against the reviewed
linear milestone. Assumptions and pending claims remain explicit; no new
experiments or source changes. Linked the introduction from PROGRESS.

## 2026-11-10 (a) — cycle 1 complete: linear model and snapshot pilot

Implemented the first executable roadmap milestone through the research-
scientist workflow; independently audited the derivations and completed one
remedial review. New standalone code/protocol/config lives in
`experiments/linear_memory/`, with tests in `tests/test_linear_memory.py` and
the evidence report in `reports/LINEAR_MEMORY_PILOT.md`. Existing transformer
code, training jobs, and user notebook edits were not changed.

**Theory:** conditional expected error is delta+A/D under fresh independent,
zero-mean isotropic unit GLOBAL keys and oracle target addresses, even when
snapshots are mutually correlated. A is actual non-target response energy.
Derived a two-fact local-key-overlap crossover including the orthogonal
failure regime. D_min is expectation-only, with the zero-noise edge case
handled explicitly. Local codebook queries may remain differentiable in
both intervention arms; the removed routes are snapshot-storage routes.

**Execution:** EXP-LM-001, 6 arms x 3 seeds x 120 CPU updates; J12,p4,v3,D16,
h6,K1; endpoint tau6 versus intermediate tau2. The local encoder is a learned
position codebook, and global snapshot routing is oracle-controlled. Frozen
config hash, raw curves, summaries, and code/environment provenance are in
`reports/results/linear_memory/EXP-LM-001/`. Scoped ignore exceptions preserve
the JSON configuration/provenance artifacts for normal version control.

**Evidence:** detached/full probes have zero forward difference and nonzero
writer-gradient differences. Exact-reader arms both reach NMSE 0.20 at
step40 in all seeds. LA full reaches at 60/40/40 versus detached 60/60/60,
observed only every20 updates; final means are 0.11497 and 0.11526. Endpoint
arms fail the target in all seeds. This is a limited LA-specific early-target
signal, not a general convergence or systems speedup. Correlated-payload
noise agrees with the conditional energy law; two-fact Monte Carlo relative
errors are <=3.33% for LA and <=0.96% for nonzero endpoint errors.

**Verification:** `.venv/bin/python -m pytest -q tests/test_linear_memory.py`
reports 15 passed in 2.37s, including actual-writer reset independence and
K=2 transition Jacobian, orthogonal retention, overwrite, commit tails,
folded/direct reads, live endpoint gradients, and error decomposition.
The CPU run completed all18 arms. Source implementation was reviewed; memory
and operation tables explicitly exclude/identify codebook, oracle metadata,
raw banks, and training tensors. At this small C the raw bank is cheaper and
smaller than folding; no LA storage/compute advantage is established.

**Decision:** milestone reached; stop cycle1 here per AGENTS.md. Next
milestone is held-out validation of the fixed-error allocation rule, keeping
oracle/position-codebook limitations explicit, before transformer integration.
The rest of the roadmap remains pending. No new large sweep was launched.

## 2026-11-09 (f) — contribution roadmap formulated

Added the requested staged roadmap to PROGRESS: freeze the cost/comparison
model; derive linear acquisition, correlated-snapshot retrieval, and gradient
paths; test the matched-forward gradient intervention; validate allocation
predictions out of sample; and establish transformer resource frontiers
against strong local-only and hierarchical baselines. Commit interval tau
and reset horizon h remain separate throughout.

Primary proposed contribution is predictive fixed-error resource allocation;
outer-optimization improvements are a conditional supporting claim. Gates
explicitly allow a null gradient result, an exact-reader-only benefit, or a
failure to predict held-out allocations. Required controls include deep
local memory, fast LoRA, local segmentation, joint local writing, endpoint
commits, branch-detached snapshots, and exact reads. Training versus
deployment cost and whole-payload theory versus token-wise implementation
are accounted separately. Next milestone is a specification and linear
model/experiment brief, not a broad run grid. Research records only changed;
no source/config modifications, experiments, or new empirical claims.

## 2026-11-09 (e) — intermediate commits as gradient access points

Recorded the user's proposed intermediate shared-memory commits in IDEAS.
Separate commit interval tau from reset/carry horizon h; endpoint-only
storage remains a baseline. Derived the final-loss gradient as a sum of
paths through each later commit, including a direct term at a committed
state. For linear delta-rule writes, that term bypasses subsequent
I-eta*x*x^T factors. This establishes shorter available routes, not a
convergence guarantee or a modification of the inner optimization objective.

Proposed control: identical snapshots with versus without stop-gradient on
the intermediate snapshot branches, retaining the differentiable live
recurrence and final commit. Forward values match at identical parameters;
the comparison tests additional writer-gradient routes. Separate historical
availability, gradient routing, and LA superposition using endpoint-only
and exact-reader controls. Account for correlated snapshots, extra commits,
and training graph memory; fixed-size folded state does not imply fixed
retrieval error. Prior recurrent attention shortcuts limit novelty claims.
Research notes only; no source/config edits, experiments, or new evidence.

## 2026-11-09 (d) — include segmented shared writes inside the local writer

User-requested baseline/theory addition recorded in IDEAS: carry the same
local memory through h elementary segments, run K inner steps on each,
commit only the final group state to global LA, and reset locally between
groups. This applies to tokens, deep prefixes, and adapter codes. Endpoints
h=1 and h=J interpolate independent segment writing and fully carried WRITE
dynamics; intermediate h retains the hierarchy while bounding local carry.

Theory now includes h in local distortion, noise sensitivity, retained local
capacity, and compute. For whole-memory commits and T=ceil(J/h), the working
error model is delta_local(p,K,h)+Gamma(p,K,h)*(T-1)/D. Within-group forgetting
belongs in delta_local. Local dependency depth is up to h*K, but total inner
step count remains J*K when K is held per elementary segment. Graph detach
and optimizer-buffer policy must be distinguished from forward carry.

Add both the h sweep and joint-versus-sequential writing of identical group
content to baseline design. Match/report actual compute, since K steps per
subsegment and K steps on a concatenated group are different budgets. The
reset/commit interval becomes an explicit theoretical optimization variable
alongside p,D,K. Only research notes changed; no source/config edits or runs.

## 2026-11-09 (c) — cycle 1: scientific contribution assessment and evidence audit

**Scope:** evaluate proposed contributions (optimization/convergence from
global LA; capacity/complexity theory for local-gradient/global-associative
memory). Recommendation, not an experiment authorization: use fixed-error
local/global resource allocation as the main contribution, with a controlled
write-isolation mechanism study as supporting evidence.

**Evidence boundary:** the report establishes that the complete independent
write + global interface architecture beats the two-token carried-prefix
baseline at N8/N16 under the stated training budget. It does not isolate LA:
state capacity, WRITE allocation, READ interface, and gradient-path depth
change together. In the matched interface architecture, exact softmax trains
faster than LA: N16 reaches 0.99 in 5.5-9.5k outer steps, versus 47.5k for
one LA seed and two non-reaches by 50k. These are OUTER training steps,
not inner WRITE convergence. N32's completed LA mean best 0.926 versus
truncated exact mean best 0.978 supports the finite-budget kernel gap; the
short adaptive N32 run and LA N64 plateau do not establish asymptotic ceilings.
Curriculum arms inherit different checkpoints; step-0 comparisons are not
a common-frozen-writer intervention.

**Diagnostic correction identified, report left unchanged:** the code's
cross-talk metric excludes the largest absolute-score item, not the known
target (`grad_memgpt.py:1732`). It is computed before interface site masking
and measures unweighted absolute similarity, not actual signed/value-weighted
retrieval noise. Agreement with the random-key estimate is suggestive rather
than a verified causal capacity law.

**Proposed decisive experiment:** a capacity/read-matched three-arm study
with shared sequential writes + exact read, disjoint local writes + exact
read, and disjoint local writes + LA. A fourth shared+LA arm tests interaction.
Use identical total fast-state budgets and common reader geometry; add a
competitive deep-prefix/fast-LoRA one-level baseline for the resource frontier.
Primary causal runs at fixed N, with training FLOPs/time to a fixed error and
right-censoring for non-reaches. Probe old facts during frozen-weight inner
writes and measure real value-weighted interference at active read sites.
Start near N16/N32; this does not require finishing the expensive N96 grid.

**Theory target:** independent writers remove products of later segment
Jacobians from an early segment's path into the stored state; this does NOT
by itself prove improved outer optimization. Combine that mechanism with
local payload size P, capacity b, finite-step distortion delta, MSE-noise
amplification coefficient Gamma, and local write cost. Under explicit random
outer-key/decoder assumptions, epsilon approximately delta+Gamma*(T-1)/D,
T=N/b; hence D_required approximately Gamma*(T-1)/(epsilon-delta).
Budget DP state, T*W_local+T*DP writes, query reads, routing, and actual
runtime storage. Seek a predicted optimal p,D,K allocation and specified
dominance/no-benefit regimes, not a universal dominance theorem: unrestricted
one-level memory can encode and emulate the hierarchical state.

**Novelty check:** ARMT already combines memory tokens and associative
storage (arXiv:2407.04841); GradMem already studies compute and input/per-layer
memory (arXiv:2603.13875, appendices D/J); ATLAS studies local-context memory
optimization and capacity (arXiv:2505.23735). Tensor capacity Dp and the mere
combination are insufficient novelty. The proposed distinctive claim is a
predictive, validated fixed-error allocation rule for gradient-written codes
and global storage. This is a targeted literature check, not a priority proof.

Research-scientist evidence audit completed; milestone reached. No source,
report, config, or manifest changes; no tests/jobs/remote requests. Only PI
research records updated. Stop after cycle 1; all proposed runs remain drafts.

## 2026-11-09 (b) — interference curve collected and analysed (partial; P2 decided)

**What changed:** collected all 30 experiments of Comet project
`gradmem-linear-read-interference` (curve job `ccde14ed`: `interface`,
`interface_softmax`; adaptive job `7c2ed873`: `adaptive`). Report
[`reports/INTERFERENCE_CURVE_RESULTS.md`](reports/INTERFERENCE_CURVE_RESULTS.md),
data `reports/results/interference_curve.json`, figure
`reports/figures/interference_em_vs_n.svg`. New read-only analysis scripts
`scripts/collect_interference_results.py` and
`scripts/make_interference_figure.py` were written in the main chat. That
deviates from the AGENTS.md role split (PI does not write code); flagged to
the user. No source, config or manifest changes.

**Coverage:** both jobs died on 2026-11-03. Interface N64 ×3 and adaptive
N32 ×3 were interrupted together at about 11:00–11:30
(`InterruptedExperiment`). The softmax chains then ran N8→N16→N32 and died at
18:15 with `OSError(28, 'No space left on device')`. 15 of 45 planned runs
never started (interface N96; softmax N64/N96; adaptive N64/N96) and 9 were
truncated (softmax N32 at ≤2.5k steps, interface N64 at 22.5–23k, adaptive
N32 at 2.5–3k).

**Verdicts:** **P2 — superposition binds at N=32.** Δ(N) = +0.0005 / +0.0025
/ **+0.052** at N=8/16/32 (best EM, mean over seeds). Δ(32) is a lower bound:
the control was truncated while the interface arm used its full 50k. Seed
ranges do not overlap (control worst 0.975 > interface best 0.948).
Interface N64 plateaus at EM 0.41 (EM and loss flat from ~10k to 23k steps);
there is no N64 control. **P1** held at N=8/16 (threshold stops at 17–19k
and 5.5–9.5k steps) and is untested at N ≥ 32. **P3** held (kernels
indistinguishable until the binding N). **P-A1** held for EM: the carried arm
never reaches 0.99 (best 0.82 / 0.48 / 0.15 at N=8/16/32). Its forgetting
matrices are not on Comet, so that part is still open. **P-A2** effect size
(interface − adaptive): 0.17 / 0.51 / 0.78.

**Interpretation (hypothesis, consistent with all points; not yet
tested):** the binding N matches the 2026-11-03 random-key estimate
noise/signal ≈ √((N−1)/d_h), with d_h = 128/4 = 32. That gives 0.47 / 0.68 /
0.98 / 1.40 at N=8/16/32/64. The measured interference floor at N ≥ 32
(0.143) equals the random-unit-vector |cos| in 32 dims (0.142). Warm-transfer
step-0 EM (not pre-registered) separates the reads cleanly: exact softmax
0.93 at N16 and N32; superposed 0.865 / 0.688 / 0.254 at N16/32/64.

**Decisions:** none taken; options are listed in the report's "Next" for the
user. The cheapest decision-relevant follow-up is resuming only the softmax
chains (N32→N64) after fixing the remote disk. The direct capacity test is
the planned `kv_interface_softmax.yaml` arm2 (d_h = 64) on the curriculum.
Cost note: N64 runs at 9.3 s/step, so a full 50k phase is about 5.4 days;
the pre-registered 3–5-day estimate for the whole job was far too low.

**Verification:** numbers re-derived end-to-end by the collector from
Comet (matching an independent ad-hoc pull); figure passed the visual gate.

## 2026-11-09 — cycle 1: complementary local/global memory and payload accounting

**User reframing adopted:** compare spending capacity on local gradient
memory versus outer associative addressing within the SAME hierarchical
family. LA can store gradient-written prefixes, tokens, or adapter codes;
it need not compete with those writer parameterizations. The local payload
size and number of global entries are essential additional variables.

**Milestone reached (analytic audit only):** a local linear memory A_s of
shape p-by-v stored under one segment key k_s in R^D gives the tensor
S=sum_s k_s tensor A_s, shape D-by-p-by-v. A query (q,x) reads
sum_s(q dot k_s)*A_s^T*x. Orthogonal segment keys and p independent local
addresses permit T<=D segments and N=T*p<=D*p arbitrary associations,
paid through D*p*v state scalars. Earlier N<=D bounds used D for the TOTAL
address width; here D is only the outer segment address and total feature
width is D*p. These are different architectures, not contradictory bounds.

For random independent unit segment keys, full local load, and independent
isotropic payload values, relative retrieval MSE is (T-1)/D. Slot-wise LA
with a separate random key per local row instead has (T*p-1)/D_slot and
state D_slot*v. At matched small error epsilon and many segments,
D_chunk approximately T/epsilon and D_slot approximately T*p/epsilon:
both states require order T*p*v/epsilon scalars. Dense chunk commits cost
order T^2*p*v/epsilon versus T^2*p^2*v/epsilon for slot commits. At fixed
N=T*p, chunk storage is order N*v/epsilon and commits N^2*v/(p*epsilon).
These are standard dense-algorithm counts, not lower bounds; sparse
coordinate-addressed writes can remove the commit difference.

**General payload law:** if local size p stores b(p) facts into P(p) payload
coordinates, T=N/b(p), ideal random-key outer storage is order
N*P(p)/(b(p)*epsilon), and dense commits order
N^2*P(p)/(b(p)^2*epsilon). Thus payload per retained fact P/b controls
storage, while P/b^2 controls commits. These conclusions require larger p
to support larger groups; at fixed segmentation, chunk storage grows with P.
Nonlinear local decoding adds noise sensitivity and local fitting error:
under small unbiased read perturbations, an approximate error model is
delta_local + gamma(p)*(T-1)/D. Budget D using the remaining error margin;
parameter count alone does not determine gamma or b.

**Qualifications:** full-payload reads cost D*P, but one-hot local queries
can select a tensor slice before reading (D*v); multiple local queries can
amortize one payload retrieval. Chunk routing must be supplied or learned
and charged. Raw LoRA factor averaging does not equal operator averaging;
use shared-basis coefficients, a shared factor, effective operators, or
explicitly analyze bilinear decoding errors. Actual model currently uses
token-wise keys, not this whole-payload tensor construction. Local writer,
projection, optimizer, and retained-bank costs remain additional.

Research-scientist audit confirmed capacity, interference, storage, compute,
and the factor-mixing qualification. Tensor-product binding is established
background (Smolensky, 1990); above cost/error derivations are this analysis.
No code/config edits, tests, training, or remote checks. Stop after cycle 1.

## 2026-11-03 — cycle 1: fixed-error complexity frontier, deep memory and fast LoRA

**User correction incorporated:** increasing writable state does not require
proportionally increasing sequence length. Independent per-layer token
parameters expose approximately L*m*d coordinates at the same m-token prefix
length as shallow memory; KV-only prefixes expose about 2*L*m*d coordinates
without memory-token queries/FFNs; fast LoRA exposes about 2*L_a*r*d factor
coordinates without extra sequence positions. Extra attention/adapter work
and backward passes still count. This invalidates a general computational
advantage inferred only from linear-state size versus shallow-token count.

**Milestone reached (theoretical framework, no experiments):** compare each
architecture's minimum total write + Q-query read cost subject to the SAME
retrieval-error requirement and a stated peak-memory budget. Optimize memory
size, write steps, retention method/replay, and compression within each
family. Count projection, backbone, commit, and runtime-memory costs; do not
equate parameter count with usable capacity or gradient-update FLOPs.

**Linear fixed-error result:** for target memory map F and query covariance
C, let sigma_j be singular values of F*C^(1/2). Best rank-r approximation
has relative expected squared error sum_{j>r}(sigma_j^2)/sum_j(sigma_j^2).
Thus r_epsilon defines the oracle factorized-memory requirement. Dense
linear-state reads cost D*d_v, while factorized reads cost r*(D+d_v), before
shared projections; dense is arithmetically preferable only above the
corresponding rank crossover. A factored linear-attention state is itself a
LoRA-like map. This is a representation/read-cost result, not proof that an
online optimizer can attain the target cheaply. Flat spectra and steeply
decaying spectra imply different winners.

**Write geometry:** linearize old/new retrieval outputs with Jacobians J_o
and J_n with respect to writable coordinates z. With Pi projecting onto
ker(J_o), new residual e is writable without first-order old-output change
iff e lies in range(J_n Pi). Minimum-norm update is
Pi*J_n^T*(J_n*Pi*J_n^T)^dagger*e, squared norm
e^T*(J_n*Pi*J_n^T)^dagger*e. Useful additional memory enlarges or improves
the conditioning of these directions. Constructing this projector is not
assumed cheap; it is a geometric diagnostic. A fixed-J oracle batch-GD
analysis is distinct from actual segment-only sequential optimization.

**No universal winner:** gradient updates to disjoint one-hot-addressed
columns preserve all previous facts exactly, while random-key additive
memory needs D about N/epsilon for normalized MSE epsilon. Orthogonal
addressing can help either writer. LA's potential advantage is cheap
independent commits when alternative writes require expensive optimization
or replay, provided the required global width/read cost remains competitive.
The existing gradient encoder is still charged to LA; analytic commits do
not remove its inner backpropagation.

Research-scientist audit confirmed the nullspace formula, counterexample,
deep-prefix/LoRA cost accounting, and fixed-error comparison. Primary papers
consulted: Prefix-Tuning (ACL 2021), LoRA (arXiv:2106.09685), TTT
(arXiv:2407.04620). No source/config edits, tests, training, or remote status
checks. Stop after cycle 1; current experiment results unchanged.

## 2026-11-02 — follow-up cycle 1: two-stage linear memory capacity and compute

**Milestone reached (analytic audit; no experiments).** Model local gradient
encoding as a p-by-v associative matrix per segment, committed additively to
a D-by-v global state through a D-by-p embedding. For T segments with b
arbitrary vector targets each, fixed linear encoding/query features imply
`N = T*b <= min(T*p, D)` as a necessary exact-recall condition, with b<=p
locally. Orthogonal segment subspaces achieve full-load capacity T*p when
T*p<=D; they require the query to identify the appropriate segment subspace.
Full-rank keys alone do not make same-key additive reads exact: their Gram
matrix must be identity for arbitrary values, or the reader must use dual
queries. These are linear arbitrary-target bounds, not measured capacity
limits for the nonlinear neural reader.

For orthonormal local keys and independent random orthonormal segment frames,
iid centered target values give relative cross-talk MSE `(T-1)*b/D`. With
zero-start K-step local GD and constant step eta, local scaling is
`a = 1-(1-eta)^K`, yielding total relative MSE
`(1-a)^2 + a^2*(T-1)*b/D`. Increasing local optimization accuracy does not
remove global superposition interference.

**Compute qualification (new code observation):** the hook computes
`_linear_state_read_bias` on ALL sequence positions before multiplying by
`site_mask` (`grad_memgpt.py:1679-1680`). Thus 2*M*L counts active interface
injection sites, not actual computed reads; current read arithmetic spans
L*(T1+T2) positions. No optimization was made. The implementation also
retains the growing bank and projected K/V.

Ignoring projection/backbone/diagnostic costs, an unfolded N-item bank costs
O(R*N*(D+v)) for R reads; folding costs O(N*D*v+R*D*v). With D=v=h, the
simple operation-count crossover is R*(2*N-h)>N*h. For H=4,h=32,N=8 the
state uses 4096 coordinates versus 2048 for projected K/V, and a state read
uses twice their leading multiply-accumulate count before folding cost.
This is an analytic cost comparison, not a runtime benchmark.

Local iterative optimization can save work relative to K repeated global
gradient passes, but a direct rank-one global delta update already costs
O(D*v) per fact. Savings require specifying that comparator and accounting
for the commit/projections. The hierarchy relocates iterative work to small
scratch memory and amortizes retrieval; it does not multiply capacity beyond
the global state budget. Research scientist independently checked the
equations; source/configs unchanged. Stop after this follow-up cycle 1.

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

## 2026-11-02 (b) — cycle 3: carried-memory baseline arm submitted as a separate job (the write-side leg of the curve)

**User-directed arm.** The curve's organizing question per the user: whether
**separate memory tokens per segment mitigate catastrophic forgetting** — so
the carried-memory baseline ([`grad_memgpt_adaptive.py`](grad_memgpt_adaptive.py),
every segment writes into the SAME M=2 tokens) runs the IDENTICAL curriculum
as a separate cluster job alongside the running interference job. Config
pinned: the adaptive fork's native cross-segment carry, baseline update
`memory_update_rule: sgd` (plain gradient inner loop; NO mamba/convex
gating, NO STAR, NO replay, NO `curriculum:` block — checkpoint chaining
like every other arm), memory-prefix read as-is; write hyperparameters
matched to the interface arms (M=2, K=2, inner_lr 0.04, grad_mode second,
use_write_head), pair-aware `n_segments = N/2` per phase. Budget asymmetry
by design: 2 carried tokens vs N stored tokens — the architecture axis, not
matched capacity (cf. the 2026-11-02 theory entry's budget-matching
constraint; also note the read state sizes there: the interface arms' bank
is N tokens but the read taps a d²/H state). `per_segment_eval` on — **the
forgetting matrix is a primary readout for this arm**. Revision-3
predictions P-A1/P-A2 verbatim in the new manifest header: P-A1 — the
carried arm fails to reach EM 0.99 even at small N (fact-1 mechanism:
running-sum overwrites), forgetting-matrix diagonal ≫ last-column gaps
widening per phase; P-A2 — the per-phase gap to the interface arms is the
"separate tokens mitigate forgetting" effect size; if the carried arm
unexpectedly HOLDS, fact 1 is contradicted at matched protocol —
investigate before any H1 claim.

**Delivery: separate manifest + run spec.**
[`manifests/kv_interference_adaptive.yaml`](manifests/kv_interference_adaptive.yaml)
(one experiment, seeds 4378/43894/2343 = 3 chains through the same wrapper —
`adaptive` arm added to its ARMS table) +
[`run_adaptive.yaml`](run_adaptive.yaml) (same env, same comet project
`gradmem-linear-read-interference`; the flock guard is per-run-copy so the
two jobs cannot collide). `run.yaml` and the RUNNING interference job
`ccde14ed` untouched.

**Verification.** 2-phase smoke: warm-start PROVEN by checksum (54/54
tensors bitwise-equal to phase-1's resolved `model.safetensors`), rerun
skips both phases, forgetting matrices written (4×4 and 8×8). N=96 probe
(included the 48-segment forgetting pass — this arm's peak must be
measured, not extrapolated): B=16 fits at 2.99 GiB, ~0.185 GiB/sample —
**~14% LIGHTER per sample than the interface arms** (the "carry retains one
big autograd graph ⇒ heavier" prior was wrong; wrapper comment corrected);
batch ladder unified with the interface arms 64/64/64/32/16. Suite
**82 passed**; dry-run = exactly 3 chains. User committed the work
mid-cycle as `083d5a5`.

**Submitted — job `7c2ed873` (`lm-mpi-job-7c2ed873-521c-4de3-8bc4-b6d54cd7ad11`),
PENDING** (queue busy at submit: 3 running / 4 pending ahead). Early gate
due at start: eval_loss trajectory, zero tracebacks,
`runs/N8-K2V2-V62_1M/interf_adaptive_N8/run_{4378,43894,2343}_*`.
Expected ~1–2 days on its own GPU once started (15 runs; plateaued phases
end by patience per P-A1). Executor-note for the record: the interface
job's N=96 phases could likely have used B>16 per the per-sample probe
numbers — left as-is (uniform protocol; already running).

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
