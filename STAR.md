# STAR — worst-case stability regularisation for multi-segment memory writes

> Implementation: [`grad_memgpt_adaptive.py`](./grad_memgpt_adaptive.py) (adaptive fork only)
> Config: [`configs/gradmemgpt/kv_retrieval/adaptive_star.yaml`](./configs/gradmemgpt/kv_retrieval/adaptive_star.yaml)
> Tests: [`test_grad_memgpt_adaptive_star.py`](./test_grad_memgpt_adaptive_star.py)
> Stage-0 manifest: [`manifests/stage0_capacity_vs_interference.yaml`](./manifests/stage0_capacity_vs_interference.yaml)
> Source paper: *STAR: Stability-Inducing Weight Perturbation for Continual Learning*, ICLR 2025, arXiv:2503.01595

## The problem

Multi-segment gradmem writes a long context into one memory state, segment by segment: segment 1 starts from the meta-learned $m_0$, every later segment continues from the previous $m_t$ via $K$ inner-loop gradient steps each. The plain-SGD rule

$$m_t = m_{t-1} - \alpha\,\nabla L(m_{t-1})$$

has no notion of what is already stored — a later segment's gradient is added with full weight and can stomp on the directions earlier facts relied on. The research-notes open question was: *to avoid rewriting information, we need to know which axes are important* — but hand-designing that importance (gates, sparsity) is hard.

STAR's answer: don't learn what is important — **optimise for the worst case**. Make the READ outputs on already-stored content invariant to any perturbation of the memory of a controlled size, and whatever future write steps do (within that size), old facts survive.

## Where it comes from

In the paper's continual-learning (CL) setting, model weights $\theta_t$ are updated on a task stream and forgetting happens because future updates drift the output distribution on old data. STAR adds a regulariser: outputs on **correctly classified buffer samples** must be stable under the worst-case weight perturbation in a local ball. The transplant to gradmem is one level down, and almost mechanical:

| STAR (continual learning) | multi-segment gradmem |
|---|---|
| continually-updated weights $\theta_t$ | carried memory state $m$, updated by inner-loop SGD |
| task stream $X_t$ | segment stream $\mathrm{seg}_1 \dots \mathrm{seg}_S$ |
| future updates $\theta_{t+s}$ (unknown) | later segments' inner-loop steps (unknown at the boundary) |
| rehearsal buffer $M_t$ | earlier segments — already in context, rehearsal data is free |
| buffer sample correct now ($x^*$) | KV pair retrieved with exact match under $m_s$ |
| output dist. $q_\theta(x)$ | READ outputs conditioned on memory: $f([m;\text{probe}])$ |

Two existing pieces of the codebase are the *weak* versions of STAR, which is a useful way to place it:

- `mem_noise_std` (VAE-style noise on memory before READ) = STAR's **random-perturbation ablation**. The paper's Table 3: matched-magnitude random noise gains ~1 point, the **gradient** (worst-case) direction gains ~5.5 — the entire contribution of the paper is that upgrade.
- `mem_prior_weight` (MSE to $m_0$) = a **parameter-space anchor**, the regulariser class (sSGD/L2-to-init) that output-space consistency beats in the paper's comparisons.

## The loss

At every segment boundary $s$ (after segment $s$'s $K$ inner steps), with $m_s$ the carried memory:

$$\mathcal{L}_{\text{STAR}}(m_s) \;=\; \mathrm{KL}\big(\,\mathrm{READ}(m_s;\,p^*)\;\big\|\;\mathrm{READ}(m_s+\delta;\,p^*)\,\big)$$

$$\delta \;=\; \delta_0 + \gamma\cdot\frac{\lVert m_j\rVert_2}{\lVert g_j\rVert_2}\, g_j \quad\text{(per memory token } j\text{)}, \qquad \lVert\delta_0\rVert_2 = \varepsilon\,\lVert m_j\rVert_2$$

($\delta_0$ is **vector-normalised** noise — a bare $\varepsilon\lVert m_j\rVert\cdot\mathcal{N}(0,I_d)$ has vector norm $\varepsilon\sqrt{d}\,\lVert m_j\rVert$, $\sim$11× too large at $d{=}128$, which drowns the $\gamma$-scaled ascent direction; a test locks $\lVert\delta_j\rVert/\lVert m_j\rVert \approx \sqrt{\gamma^2{+}\varepsilon^2}$.)

$$g \;=\; \nabla_{m+\delta_0}\,\mathrm{KL}\big(\mathrm{READ}(m_s) \,\big\|\, \mathrm{READ}(m+\delta_0)\big) \quad\text{(one gradient-}\textbf{ascent}\text{ step)}$$

The total outer loss gains $\lambda \cdot \overline{\mathcal{L}_{\text{STAR}}}$, where the bar is the mean over the fired boundaries (each boundary's term is itself the mean over selected probes across the batch), with $\lambda$ warmed linearly $0 \to \lambda$ over `star_anneal_steps` if set.

**Probes $p^*$** are the teacher-forced `?!K:V!|` queries of KV pairs already attributed to written segments (`star_probe_scope`: `all` = segments $\le s$, the buffer-at-end-of-task semantics; `past` = $<s$ only), restricted — unless `star_correct_only=false` — to pairs **currently retrieved with exact match** under $m_s$ (STAR's $x^*$: *you can only forget what you currently know*). The paper's Table 4 shows why: enforcing stability on not-yet-learned content collapses performance.

**Per-token normalisation** replaces the paper's per-layer normalisation: memory tokens play the role of layers (each has its own numeric scale). After the ascent step, $\lVert\delta_j\rVert_2/\lVert m_j\rVert_2 \approx \sqrt{\gamma^2+\varepsilon^2} \approx \gamma$.

**KL convention.** Slot $t$ of the READ logits predicts token $t$ (same convention as `_probe_exact_match`). The KL is computed only at the **scored positions** — the value-token positions of the probe, i.e. `target_mask` minus the structural `!`/`|` — exactly where EM is judged. Per probe it is summed over scored positions; the loss is the mean over selected probes across the batch.

**Why worst-case beats average.** With the reference fixed at $m_s$, $\mathrm{KL}(m_s \| m_s+\delta) \approx \tfrac12 \delta^\top F \delta$ where $F$ is the output Fisher w.r.t. memory. One noise-initialised ascent step computes $\delta \propto F\,\delta_0$ — a single **power iteration** on $F$. So STAR ≈ penalising the *top curvature direction* of the retrieval loss in memory space: the directions where perturbing memory most damages old content. A random perturbation only regularises the average curvature, not the max. (The `attn_double_bwd` HVP kernels could compute the exact top eigenvector if the 1-step approximation ever becomes the bottleneck.)

**Why KL-consistency, not replay.** Re-including old pairs in the inner loss with CE (the "sample random KV pairs as subtasks" idea) makes old and new writes compete for the same capacity and can be destructive once memory has drifted. The KL term has **zero gradient at $\delta=0$** — a pure curvature penalty that never pulls memory back toward old content, it only makes the retrieval map locally flat. (The outer-loss *vanilla replay* ablation below makes this an empirical comparison rather than an argument.)

## Vanilla replay — the corrective ablation

Because STAR shares its **data mechanics** with rehearsal (buffer = written segments, EM=1 selection, cumulative re-inclusion at every boundary) while differing in its **gradient signal**, the natural control is vanilla replay on identical data. It is implemented as a sibling of the STAR term, deliberately symmetric so the comparison isolates *only the functional form*:

$$\mathcal{L}_{\text{replay}}(m_s) \;=\; \mathrm{CE}\big(\mathrm{READ}(m_s;\,p^*)\big) \;=\; -\tfrac{1}{|p^*|}\sum_{j \in p^*}\; \sum_{t \in \text{scored}(j)} \log \mathrm{READ}(m_s;\text{probe}_j)_t[\text{answer}_t]$$

same teacher-forced probes (which already contain their answers, so the CE target is free), same boundaries, same `probe_scope`, same correct-only mask. The outer loss gains $\rho \cdot \overline{\mathcal{L}_{\text{replay}}}$ (mean over fired boundaries, $\rho$ warmup via `replay_anneal_steps`). One grad-carrying forward per boundary — cheaper than STAR's three.

| | STAR (consistency) | replay (refitting) |
|---|---|---|
| gradient when retrieval is already perfect | **zero** (silent) | non-zero (keeps pulling) |
| effect | **preventive**: makes what works hard to break | **corrective**: can recover dropped pairs |
| on a forgotten pair | abandons it (EM mask is a ratchet) | pulls it back |
| analogy | flatness / SAM-style robustness | experience replay (ER) |

The STAR paper's own headline configuration is replay **plus** worst-case consistency (their ER+STAR), so the interesting 2×2 is {replay} × {STAR} — replay repairs what STAR failed to prevent, STAR keeps replay's repairs stable. Prediction from the paper's small-buffer results: STAR-only ≥ replay-only, combo best; on kv4 with $S{=}2$ the buffer is everything (no eviction), the regime that favours STAR most.

## Gradient flow & safety properties

- **Only the perturbed branch carries meta-gradient.** The reference logits are computed under `no_grad` and detached; $\delta$ is built from detached tensors (`mem_det + δ` working point for the ascent, `create_graph=False` in `autograd.grad`). This is the paper's first-order SAM-style approximation, and it guarantees the term **never exceeds the existing second-order autograd depth** — no third-order derivatives even under `grad_mode="second"`.
- **The WRITE trajectory is untouched.** STAR is an outer-loss-only term: predictions and the carried memory are *bitwise identical* with STAR on vs off (locked by tests). What changes is only where the meta-optimiser steers $\theta$, $m_0$, gates, projections — toward memory states whose READ behaviour is locally flat.
- **Train-only.** The term is gated on `self.training`; eval forwards (including the per-segment forgetting eval) are unchanged. `star_weight=0` — or simply no `kv_queries` in the batch — is a bitwise no-op with identical `inner_loop_stats` keys.
- **Works under every `grad_mode`.** Under `"second"` the STAR gradient reaches `self.mem` through the unrolled inner loop; under `"none"` (detached recurrence) it still reaches the backbone through the perturbed probe forward — useful for cheap first experiments.
- **`seg_bptt` interaction.** Boundaries older than the BPTT window have a detached `mem_batch`; their STAR meta-gradient then flows only through the perturbed probe forward ("local robustness"), not through the inner loop. Correct, just weaker — expected when truncating.
- **Empty guards.** If no probe is in scope, or nothing is currently retrievable, the boundary silently contributes nothing (term exactly zero, no stats keys added).

## Configurability

All knobs live under the `adaptive:` config section (flattened to `--star-*` CLI flags automatically; `run_from_config.py` needs no edit).

| knob | default | meaning |
|---|---|---|
| `star_weight` | `0.0` | $\lambda$, the outer-loss weight. **0 = off** (bitwise no-op). Paper grid: 0.005–0.1 |
| `star_gamma` | `0.01` | $\gamma$, relative perturbation size per memory token. Paper grid: 0.001–0.05 |
| `star_epsilon` | `0.01` | $\varepsilon$, $\delta_0$ noise std as a fraction of the per-token norm (escapes the all-zero-gradient point at $\delta=0$, where KL and its gradient vanish) |
| `star_ascent_steps` | `1` | ascent steps for $\delta$. Paper: 1 is best, 3 sometimes slightly better, $\ge$5 degrades (first-order approximation breaks down) |
| `star_perturbation` | `"grad"` | `"grad"` = worst-case direction (the method); `"random"` = matched-magnitude Gaussian noise (the Table-3 ablation) |
| `star_probe_scope` | `"all"` | probes from segments $\le s$ (`"past"`: $<s$, excludes the just-written segment) |
| `star_correct_only` | `true` | restrict probes to currently-retrievable pairs (STAR's $x^*$) |
| `star_boundaries` | `"all"` | fire at every segment boundary; `"last"` = only the final (processed) boundary, cheapest |
| `star_anneal_steps` | `0` | $\lambda$ warmup $0 \to \lambda$ over this many outer steps (driven by `set_train_step`, like the mem-prior) |
| `replay_weight` | `0.0` | $\rho$, the vanilla-replay weight (the corrective ablation above; 0 = off) |
| `replay_boundaries` | `"all"` | same semantics as `star_boundaries`, for the replay term |
| `replay_probe_scope` | `"all"` | same semantics as `star_probe_scope`, for the replay term |
| `replay_correct_only` | `true` | same semantics as `star_correct_only`, for the replay term |
| `replay_anneal_steps` | `0` | $\rho$ warmup, mirroring `star_anneal_steps` |

## How it fits the codebase

| what | where |
|---|---|
| config params + validation + docstring | `GradMemGPTConfig.__init__` (`grad_memgpt_adaptive.py:148`) |
| $\lambda$ / $\rho$ schedules | `_star_weight_now` / `_replay_weight_now` (`grad_memgpt_adaptive.py:755`/`:762`) |
| position-summed KL | `_star_kl` (`grad_memgpt_adaptive.py:848`) |
| vectorised EM mask (shared by both boundary terms) | `_probe_exact_match_batch` (`grad_memgpt_adaptive.py:864`) |
| the whole boundary term | `_star_boundary_loss` (`grad_memgpt_adaptive.py:880`) |
| vanilla-replay boundary term | `_replay_boundary_loss` (`grad_memgpt_adaptive.py:955`) |
| forward integration (train-only gate, boundary loop, loss assembly + idle-zeros) | `forward`: `star_active`/`replay_active` at `:1314`, boundary calls at `:1568`/`:1588`, loss assembly at `:1720`+ |
| train-time metric surfacing (generic: ALL `inner_loop_stats` tensor keys → `train_*` in the HF logs → comet; no trainer edit needed for new metrics) | `CustomTrainer.compute_loss` / `.log` (`run_gradmemgpt_on_kv_retrieval.py:366`) |
| probe construction (shared with the forgetting eval) | `build_kv_probe_queries` (`run_gradmemgpt_on_kv_retrieval.py:459`) |
| train collator attaches probes when `star_weight>0` **or** `replay_weight>0` | `use_probes` at `run_gradmemgpt_on_kv_retrieval.py:1361` (single-stage + curriculum stages) |
| CLI fields → config | `ExperimentArgs` at `:983`+, pass-through in the `AdaptiveGMConfig(...)` call |

`_star_boundary_loss` reuses the READ machinery verbatim — `_read_once` for the batched probe forwards and `_probe_exact_match` for the correct-only mask — so the stability term measures *exactly* the quantity the forgetting matrix measures. Per boundary it costs one no-grad reference forward, `star_ascent_steps` × (forward+backward) for the ascent, and one meta-gradient-carrying forward — small probe sequences ($M$ + ~7 tokens), typically ~1.5–2× training time, matching the paper's reported overhead.

**The `kv_queries` batch key is top-level, on purpose.** The model takes it as a named `forward` argument (`kv_queries=...`), not inside `input_ids`: the HF Trainer's eval input-decoding path pads `input_ids` leaves and **raises on non-tensor types** (the list of ignore-token ids crashes `pad_across_processes`). Eval-mode forwards receive and ignore the probes, so eval behaviour is unchanged.

## Segment geometry on kv4 (read before changing segmentation)

kv4 contexts are BOS + 4×7 = 29 tokens, right-padded to a multiple of 8 → **32**. The model chunks the *padded* length:

- `n_segments: 2` → boundary at token 16: pairs {1,2} fully in segment 0, pair 3 straddles (attributed to segment 1 — the later segment, per the collator convention), pair 4 in segment 1. **This is the pilot setup.**
- `segment_size: 14` would give $\lceil 32/14 \rceil = 3$ segments — a trap: 14 = 2 KV pairs only holds pre-padding.

## Run

### Stage 0 first (STAR off) — decide the pilot's memory size

```bash
python run_scheduler.py manifests/stage0_capacity_vs_interference.yaml --dry-run  # inspect
python run_scheduler.py manifests/stage0_capacity_vs_interference.yaml            # 5 runs
```

$m\in\{3,4\}$ × `n_segments`$\in\{1,2\}$ + a same-seed repeat (seed-repro check). Decision rule:

- $m{=}3$ **fails one-segment too** → the drop is **capacity** → pilot uses $m{=}4$ (STAR cannot add capacity).
- $m{=}3$ **passes one-segment, fails two-segment** → **interference** confirmed → pilot uses $m{=}3$ (sharpest STAR signal).

### The pilot ladder — a 2×2 plus a perturbation control

```bash
CFG=configs/gradmemgpt/kv_retrieval/adaptive_star.yaml

# 1. baseline: plain-SGD multi-segment write (no regulariser)
python run_from_config.py --config $CFG adaptive.star_weight=0.0

# 2. random-perturbation ablation (the mem_noise_std-style control)
python run_from_config.py --config $CFG adaptive.star_perturbation=random

# 3. full STAR (worst-case consistency), then the small grid
python run_from_config.py --config $CFG                                        # λ=0.1, γ=0.01
python run_from_config.py --config $CFG adaptive.star_weight=0.05
python run_from_config.py --config $CFG adaptive.star_gamma=0.05

# 4. vanilla replay (the corrective control; one probe forward per boundary)
python run_from_config.py --config $CFG adaptive.star_weight=0.0 adaptive.replay_weight=0.1

# 5. replay + STAR (the paper's headline configuration, ER+STAR)
python run_from_config.py --config $CFG adaptive.replay_weight=0.1
```

Runs 1/3/4/5 form the {replay} × {STAR} 2×2 on identical probes and boundaries; run 2 splits STAR into its random vs worst-case halves (the paper's Table 3).

**Acceptance criteria** (what "STAR works" means here):

1. **Forgetting gap** $G = \mathrm{EM}_{1\text{seg}} - \mathrm{EM}_{2\text{seg}}$ at fixed $m$ shrinks toward 0 — multi-segment writing becomes as good as one-shot writing of the same content.
2. **Ladder ordering** on retention: baseline $\le$ random $\le$ grad.
3. No regression on one-segment tasks (stability must not kill plasticity).
4. Retention matrices (`forgetting_matrix_step*.json/csv` per run dir) show the off-diagonal decay flattening.

## Performance (what it should cost, and what to do if it doesn't)

Measured on the kv4 pilot config (B=64, `grad_mode="second"`, S=2, n_q=4 probes) on a laptop RTX 3050: **~20% overhead** (33 → 39 ms/step; each `_star_boundary_loss` ≈ 5 ms, a probe forward ≈ 0.9 ms), unchanged under the `accelerate` wrapper and bf16. Replay adds one probe forward per boundary — less. If a STAR run is *much* slower than that, it is not the method's compute — it is one of the latency-bound paths, all of which are now optimised:

- **Correct-only mask** — was a per-sample Python loop whose `if score.sum() == 0:` device-syncs once per probe (~250 syncs/step; latency-bound, amplified by deep GPU queues on faster cards and by multi-device setups). Now `_probe_exact_match_batch`, a vectorised one-shot (locked equal to the scalar helper by a test). The eval-only forgetting pass still uses the scalar helper (eval cadence, not hot).
- **Probe collator** — was 2 tokenizer calls *per KV pair* (~512 per batch; per-call overhead dominates on fast tokenizers). Now one batched encode per side (2 calls total); outputs verified bitwise-identical to the old builder. If the dataloader runs with `num_workers=0`, this was inline in the training loop — the most plausible cause of a ~30× slowdown observed on one remote setup.
- No `.item()` syncs remain in the boundary losses (stats are tensors).

If slowness persists after syncing these files, bisect on the remote: `adaptive.star_correct_only=false` (mask path), `adaptive.star_boundaries=last` (half the terms), and `adaptive.star_weight=0 adaptive.replay_weight=0.1` (collator only, no STAR compute). A `py-spy dump --pid <trainer_pid>` (or `py-spy top`) on the slow run pinpoints the stuck stack immediately.

## Diagnostics

| metric | meaning |
|---|---|
| `train_star_kl` | mean worst-case KL over the fired boundaries (the term being minimised; should trend down across training). **0.0 = enabled but idle** (x\* empty — nothing currently retrievable) |
| `train_star_delta_ratio` | achieved $\lVert\delta_j\rVert/\lVert m_j\rVert$ — expect $\approx\sqrt{\gamma^2+\varepsilon^2}$ (exactly $\gamma$ for `"random"`). Far from that = a bug, not a result |
| `train_star_n_probes` | number of selected probes at the **last** fired boundary |
| `train_star_weight_now` | the scheduled $\lambda$ (sanity for the warmup) |
| `train_replay_ce` | mean replay CE over the fired boundaries (the corrective term; should fall toward 0 on retrievable pairs) |
| `train_replay_em_rate` | fraction of in-scope probes currently retrieved with exact match at the last boundary — a **live forgetting measurement** during training, reported even when the CE term is masked empty |
| `train_replay_weight_now` | the scheduled $\rho$ |

(Eval-side metrics never include these keys: the terms are train-only by design.)

**Metric surfacing (train-time).** The terms are train-only and the standard `inner_loop_stats` → eval → comet path only carries eval forwards — so `CustomTrainer.compute_loss` captures the STAR/replay stats of every training step and `CustomTrainer.log` injects them into the training logs as **`train_star_kl`**, **`train_star_delta_ratio`**, **`train_star_n_probes`**, **`train_star_weight_now`** (and the `train_replay_*` equivalents) — these are the curves to watch on comet. When a term is *enabled but idle* (no eligible probe: empty scope, or nothing currently retrievable under the x\* mask — the normal state early in training), the metrics are reported as **zeros** rather than going silent, so "off" and "idle" are distinguishable. `train_replay_em_rate` is reported whenever probes were seen, even when the CE term itself is masked empty.

## Tests

```bash
python test_grad_memgpt_adaptive_star.py
```

locks in: (A) bitwise-off guarantee (weight 0 / eval mode / no probes); (B) WRITE-path invariance + positive $\lambda\cdot$KL contribution; (C) meta-gradient flow to `self.mem` under `"second"` and to the backbone even under `"none"`; (D) empty-scope silent no-op + `"last"`-boundary single term; (E) random-$\delta$ hits $\gamma$ exactly and a tiny $\gamma$ collapses the loss back to the star-less value; (F) replay: outer-only invariance, positive CE, $\rho\cdot$CE contribution, grad flow, `replay_em_rate` $\in [0,1]$, and exact additivity $\Delta\mathrm{loss} = \lambda\cdot\mathrm{kl} + \rho\cdot\mathrm{ce}$ when both terms are on.

## Footguns

- `star_epsilon=0` with `star_perturbation="grad"` is a silent no-op: $\delta_0=0 \Rightarrow$ KL $=0 \Rightarrow g=0 \Rightarrow \delta=0$.
- `"random"` ignores `star_epsilon`/`star_ascent_steps` by design (matched-magnitude noise only).
- Setting `star_weight>0` makes the train collator emit probes — expect a small dataloader cost (regex per context) and the ~1.5–2× step-time overhead.
- Under curriculum, `star_anneal_steps` restarts every stage (`global_step` resets per stage — same caveat as the mem-prior and energy schedules; see CURRICULUM.md).
- On the offline laptop: `COMET_START_ONLINE=0` + `CUDA_VISIBLE_DEVICES=0` (`.envrc` points at a GPU that only exists on the remote machine).

## Extensions (not yet implemented)

- **Lookahead consistency** — gradmem-specific sharpening of STAR: at the boundary the *actual* next write step $\delta_{\text{next}} = -\alpha\nabla L_{\text{recon}}(m_s; \mathrm{seg}_{s+1})$ is computable (the whole context is available at WRITE time). Regularise $\mathrm{KL}(\mathrm{READ}(m_s) \| \mathrm{READ}(m_s+\delta_{\text{next}}))$ — penalises the exact forgetting mechanism, no ascent needed. Fits as `star_perturbation="next_step"`.
- **Inner-loop replay** — the faithful ER mapping ($\rho \cdot L_{\text{recon}}(m;\,\text{old pairs})$ folded into `seg_inner_loss`; the "sample random KV pairs as subtasks" idea). Changes the **test-time trajectory**, unlike the outer-loss replay above, so it muddies the same-data comparison — defer until the outer 2×2 shows signal.
- **STAR gradient as gate feature** — feed $\nabla_m \mathcal{L}_{\text{STAR}}$ into `gate_features` of the gated rules, or project updates orthogonal to it (OGD-style): the importance signal the gates currently lack.
- **Exact worst case** — replace the 1-step power iteration with an HVP-computed top eigenvector (`attn_double_bwd`).
- **Boundary sampling** — `star_boundaries="sample"` for one random boundary per step when $S$ grows large.
