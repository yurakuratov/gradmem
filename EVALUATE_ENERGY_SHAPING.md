# Frozen-checkpoint evaluator

[`evaluate_energy_shaping.py`](evaluate_energy_shaping.py) compares frozen GradMem and EnergyGradMem checkpoints. It evaluates task generalization, the memory-writing landscape, and—when requested—whether the inner write objective is aligned with downstream read quality.

The evaluator does not train the model. It reconstructs each model from its run configuration, loads a checkpoint strictly, evaluates in FP32 on the selected device, and checks that the parameter hash is unchanged afterward.

## Quick start

Evaluate every completed `run_*` in one experiment folder at its recorded best checkpoint:

```bash
python evaluate_energy_shaping.py \
  --experiment-paths runs/N8-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K2_ilr0.4_grad_second_bs_64_lr_1e-04_fp32 \
  --aliases baseline \
  --checkpoint-selector best \
  --device mps \
  --data-root ./data \
  --n-values 8 16 32 64 \
  --inner-steps 0 1 2 4 8 16 32 \
  --output-dir runs/shaping_eval/baseline
```

Compare several experiment folders:

```bash
python evaluate_energy_shaping.py \
  --experiment-paths <gradmem-folder> <energy-folder> <shaped-folder> \
  --aliases baseline energy energy_shaped \
  --checkpoint-selector best \
  --device mps \
  --data-root ./data \
  --n-values 8 16 32 64 \
  --inner-steps 0 1 2 4 8 16 32 \
  --quality-diagnostics \
  --output-dir runs/shaping_eval/three_way
```

Select individual runs or use different checkpoint selectors:

```bash
python evaluate_energy_shaping.py \
  --model baseline <baseline-run> checkpoint-14500 \
  --model energy <energy-run> best \
  --model energy_shaped <shaped-run> latest \
  --device mps \
  --output-dir runs/shaping_eval/exact_checkpoints
```

`best` reads `trainer_state.json.best_model_checkpoint`. `latest` selects the numerically greatest checkpoint directory. An exact selector such as `checkpoint-14500` selects that checkpoint in every resolved run.

An input path containing `config.json` is treated as one run. Otherwise the evaluator expands numerically ordered `run_*` directories and ignores `.lock` directories. Incomplete runs are skipped by default and recorded in the report; `--include-incomplete` allows them when the selector can be resolved.

### Limiting the amount of data

The task grid uses the complete validation set unless `--max-eval-examples` is set:

```bash
--max-eval-examples 512
```

Landscape and quality diagnostics have separate limits:

- `--matching-examples`: examples used by context–memory matching.
- `--scan-examples`: examples used by the landscape interpolation and radial scans.
- `--contour-examples`: examples averaged in the 2D contour slice.
- `--quality-examples`: examples used by objective–quality diagnostics.
- `--matching-bank-size`: number of memories in each matching bank.
- `--num-radial-directions`: random orthogonal directions per example.

### Adding quality diagnostics to an existing evaluation

Quality diagnostics are additive and can be computed for an existing completed output:

```bash
python evaluate_energy_shaping.py \
  --output-dir runs/shaping_eval/three_way \
  --resume \
  --quality-diagnostics
```

When base arguments are omitted during resume, they are recovered from the root manifest. Completed compatible artifacts are reused. Plot presentation is regenerated from saved data; model inference is performed only for missing diagnostics. A checkpoint directory is reused only when its checkpoint hash and evaluation signature match.

## Terminology and objective conventions

- **N**: number of facts in a KV-retrieval context.
- **K**: number of inner memory-writing updates used at inference.
- **Training K**: the K stored in the checkpoint's run configuration.
- **Initial memory**: the learned memory initialization, denoted $M_0$.
- **Written memory**: memory after K inner updates, denoted $M_K$.
- **Target NLL / target loss**: per-example cross-entropy of the downstream read prediction.
- **Inner write objective**, denoted $J(C,M)$: the objective minimized when writing memory.

The inner objective depends on the model:

- GradMem with `write_objective=reconstruction`: per-example context reconstruction cross-entropy.
- EnergyGradMem: learned scalar energy.

Lower is better for both objective types, but their numerical scales are not comparable. Do not compare absolute reconstruction loss with learned energy. Do not rely on absolute learned-energy scale across separately trained checkpoints either. Within-model changes, ranks, AUC, monotonicity, and downstream quality are safer comparison signals.

During task evaluation, auxiliary shaping losses and outer inner-loss contributions are disabled. The inference-time K is overridden for each grid cell and `last_K_second_order=0`. This changes neither the inner update values nor model parameters; it avoids unnecessary higher-order graphs.

## Output layout

```text
output/
  report.md
  manifest.json
  per_checkpoint/<alias>/<run_id>/<checkpoint>/
    report.md
    manifest.json
    task_examples.jsonl
    task_summary.csv
    matching_summary.csv
    interpolation.csv
    radial.csv
    contours.csv
    quality_alignment/                 # only with --quality-diagnostics
    plots/
  aggregate/<alias>/
    report.md
    task_summary.csv
    quality_alignment/
    plots/
  comparisons/<alias_a>__<alias_b>/
    seed_<seed>/
    task_comparison_summary.csv
  plots/                               # alias-level comparison plots
```

There are three plot levels:

1. `per_checkpoint/.../plots/` shows one checkpoint without run-to-run bands.
2. `aggregate/<alias>/plots/` averages repeated runs of one experiment alias. Lines are arithmetic run means; bands are sample standard deviations across runs when present.
3. `plots/` at the root compares alias aggregates. Inner-objective plots that cannot share a meaningful scale remain separate per alias.

For one run, sample standard deviation is undefined and stored as `null`; reports render it as `—`. Across-run standard deviation measures seed/checkpoint variability. Bootstrap intervals measure finite-validation-sample uncertainty conditional on a checkpoint. Neither substitutes for the other.

## Task-generalization plots

### `task-em-vs-n-k<K>.svg`

**What is plotted**

- X axis: N, the number of context facts.
- Y axis: exact-match rate in `[0,1]`.
- K is fixed to the value in the filename.
- Each line is a checkpoint or experiment alias.
- Aggregate/root bands, when available, are sample standard deviations across runs.

Exact match is one only when every scored target token is correct. Formatting separator tokens `!` and `|` are excluded from exact-match and token-accuracy scoring.

**What it shows**

This is the main context-length generalization plot. A curve that remains high as N grows generalizes beyond the training context size. Compare aliases vertically only when they use the same data and read metric.

### `task-exact-match-vs-k-n<N>.svg`

**What is plotted**

- X axis: inference-time inner steps K.
- Y axis: exact-match rate.
- N is fixed to the value in the filename.

**How to interpret it**

- Rising then flat: more writing helps and then saturates.
- Peak followed by decline: excess inner optimization damages retrieval.
- Flat from K=0: the writable memory or write updates add little for that condition.
- Improvement only at large K: the learned update may be too small or poorly conditioned for the intended inference budget.

### `task-target-loss-vs-k-n<N>.svg`

**What is plotted**

- X axis: inference-time K.
- Y axis: mean downstream target cross-entropy.
- Lower is better.

Target loss retains confidence information that exact match discards. If exact match is saturated but target loss still falls, the model is becoming more confident without changing decoded answers. If exact match falls while target loss rises, extra writing is plainly harmful. If the two disagree, inspect per-example results and token accuracy.

### `task-objective-decrease-vs-k-n<N>.svg`

For each example, the evaluator computes

$$
\Delta J_K = J(C,M_0)-J(C,M_K).
$$

**What is plotted**

- X axis: inference-time K.
- Y axis: mean write-objective decrease.
- Positive values mean the inner loop lowered its own objective.

**How to interpret it**

This verifies optimization behavior, not usefulness. A model can show increasing objective decrease while target loss or exact match gets worse. That is precisely the failure the quality-alignment diagnostics are intended to detect. Do not compare the magnitude of this curve across incompatible objective types.

## Context–memory matching metrics

Context–memory matching is currently reported in `matching_summary.csv`, `matching_examples.jsonl`, and Markdown tables rather than as a dedicated SVG.

For a bank of B examples, the evaluator writes K=2 memories and constructs the complete matrix

$$
J_{ij}=J(C_i,M_j).
$$

The diagonal is the correct context–memory pairing. Off-diagonal elements are mismatches. Lower objective is better.

- **Top-1 accuracy**: fraction of rows where the diagonal is the unique lowest objective. Ties are treated pessimistically.
- **MRR**: mean reciprocal rank of the diagonal memory.
- **Pairwise AUC**: probability that a mismatch has higher objective than the positive; ties get half credit.
- **Margin**: $J(C_i,M_j)-J(C_i,M_i)$ for $j\ne i$.
- **Margin satisfaction**: fraction of positive/mismatch pairs whose margin is at least `0.1`.

Top-1, MRR, and AUC are scale-independent and are the preferred cross-checkpoint geometry comparisons. Values near random-bank behavior mean the objective does not identify which written memory belongs to a context. Strong matching does not by itself prove that the selected memory produces a correct downstream read.

## Standalone landscape plots

The standalone landscape probes use K=2-written memories for every model. They ask whether the written memory looks like a favorable point under that model's own inner objective. They do not directly measure read quality.

### `interpolation-n<N>.svg`

Each positive written memory $M_i^+$ is paired deterministically with a different example's memory $M_i^-$. Candidates follow

$$
M_i(t)=(1-t)M_i^+ + tM_i^-,\qquad t\in[0,1].
$$

**What is plotted**

- X axis: interpolation coefficient t.
- `t=0`: positive memory for the current context.
- `t=1`: deranged/mismatched memory.
- Y axis: median over examples of
  $J(C_i,M_i(t))-J(C_i,M_i^+)$.

The base landscape scan uses 21 evenly spaced t values. Per-checkpoint bootstrap intervals and monotonic-rise frequency are stored in the CSV/JSON summaries; aggregate plot bands are run-to-run standard deviation.

**How to interpret it**

- A smooth positive rise toward `t=1` means moving toward a mismatch is penalized.
- A flat curve means weak discrimination along this direction.
- Negative regions mean some interpolated memories receive a better objective than the positive memory.
- A high endpoint alone is weaker evidence than a consistently rising path.

### `radial-n<N>.svg`

For each positive memory, the evaluator creates seeded orthonormal/random directions $v_d$. The write radius is

$$
r_i=\lVert M_i^+-M_{0,i}\rVert_2.
$$

Candidates are

$$
M_{i,d}(a)=M_i^+ + a\,r_i v_{i,d},
$$

for normalized radii `a = [0, .125, .25, .5, 1, 1.5]`.

**What is plotted**

- X axis: perturbation radius divided by the write radius.
- Y axis: median objective change from the positive memory over examples and directions.

**How to interpret it**

- Positive growth in both small and large radii suggests a local basin around the positive memory.
- Negative values near zero mean the K=2 endpoint is not a local minimum along many sampled directions.
- A rise only at large radius indicates weak local curvature.

The fraction of directions whose objective rises is stored in `radial.csv`. Because only finitely many directions are sampled, this is a diagnostic slice, not a proof of a local minimum.

### `contour-<alias>-n<N>.svg`

Per-checkpoint contours average a two-dimensional slice over examples. Let

$$
u_i=\frac{M_i^- - M_i^+}{\lVert M_i^- - M_i^+\rVert_2}
$$

be the positive-to-mismatched direction, let $v_i$ be a seeded orthogonal direction, and let $s_i=\lVert M_i^- - M_i^+\rVert_2$. The plotted memory is

$$
M_i(x,y)=M_i^+ + s_i(xu_i+yv_i).
$$

**Axes and color**

- X axis: normalized mismatch direction.
- Y axis: normalized seeded orthogonal direction.
- `(x=0,y=0)`: the positive memory.
- `(x=1,y=0)`: the mismatched memory.
- Color: mean objective change from the positive memory.

The grid is `31×31`, with x in `[-0.5,1.5]` and y in `[-1,1]` by default.

**How to interpret it**

A favorable slice has a low region near the origin and rises toward the mismatch and orthogonal perturbations. A displaced minimum suggests the positive endpoint is not centered in the local basin. Ridges, flat valleys, or lower regions toward the mismatch reveal anisotropy or mis-ranking. Because the plot averages examples, consult per-example probes when a mean contour looks deceptively smooth.

At aggregate/root level:

- `contour-mean-<alias>-n<N>.svg` is the across-run mean slice.
- `contour-std-<alias>-n<N>.svg` is sample standard deviation across runs at every grid coordinate.

The standard-deviation heatmap shows reproducibility, not landscape quality: low variability can describe a consistently good or consistently bad surface.

## Objective–quality diagnostics

Enable these with `--quality-diagnostics`. They directly test whether lower inner objective corresponds to better downstream reads.

For each candidate memory, the evaluator records:

- inner write objective;
- target NLL;
- exact match and token accuracy;
- distance to the training-K memory;
- distance to a trajectory oracle memory.

The candidate families are:

- **Trajectory**: memories after each configured `--quality-inner-steps` K value.
- **Interpolation**: 11 points from the training-K memory to a deranged memory.
- **Radial**: the configured random directions and normalized radii around the training-K memory.

The **trajectory oracle** is not a ground-truth memory. For each example, it is the sampled trajectory state with the lowest target NLL:

$$
K_i^*=\arg\min_{K\in\mathcal K} \operatorname{NLL}(M_{i,K}).
$$

Distance-to-oracle is $\lVert M-M_{i,K_i^*}\rVert_2$. Distance-to-training-K is distance to the checkpoint's configured training-K state.

### `objective-vs-target-nll[-<alias>]-n<N>.svg`

This scatter uses trajectory candidates only.

**Axes and marks**

- X axis: objective change from the training-K state.
- Y axis: target-NLL change from the training-K state.
- Each faint point: one example at one sampled K.
- Point color: K, shown in the legend.
- Large labeled points: medians at each K.
- The connecting line follows those K medians; it is not an optimization path for a single example.

Both axes use “lower is better.” Therefore:

- lower-left: objective and read quality both improve;
- upper-right: both worsen;
- upper-left: objective improves but read quality worsens—the dangerous conflict quadrant;
- lower-right: read quality improves even though the objective worsens.

The axes are clipped to readable symmetric limits based on the 99th percentile, so extreme outliers remain in machine-readable data but can be clipped visually.

At aggregate/root level the plotted candidate values are aligned per-example run means. Separate alias plots are used because inner-objective scales can differ.

### `objective-vs-token-accuracy[-<alias>]-n<N>.svg`

This is the higher-is-better counterpart of the target-NLL scatter.

- X axis: objective change from the training-K state; lower is better.
- Y axis: token-accuracy change from the training-K state; higher is better.
- Faint points: individual examples at sampled K values.
- Large connected K points: mean objective and mean token-accuracy changes at each K.

The favorable alignment direction is now upper-left: objective decreases while token accuracy increases. The shaded conflict quadrant is lower-left: the objective decreases while token accuracy also decreases. Token accuracy is more informative than EM when only some target tokens change.

### `objective-vs-exact-match[-<alias>]-n<N>.svg`

- X axis: objective change from the training-K state; lower is better.
- Y axis: exact-match change from the training-K state; higher is better.
- Faint points have discrete Y values `-1`, `0`, or `+1` because per-example exact match is binary.
- Large connected K points use means, so their Y coordinate equals the change in the population EM rate.

As with token accuracy, upper-left is beneficial and the shaded lower-left quadrant is a conflict. Most individual points may overlap at zero when predictions do not change; the connected mean points expose smaller aggregate EM differences that a median would hide. The Y axis is fixed to `[-1,1]` for consistent interpretation.

### `correlation-matrix-n<N>.svg`

For each example and candidate family, the evaluator computes Spearman rank correlation between objective and each error/distance metric. For trajectory, the ranks are computed across the sampled K states shown compactly by their minimum–maximum range, for example `trajectory K=0–32`.

For example, the trajectory target-NLL cell starts from

$$
\rho_i=\operatorname{Spearman}
\left(
\{J(C_i,M_{i,K})\}_{K\in\mathcal K},
\{\operatorname{NLL}(M_{i,K})\}_{K\in\mathcal K}
\right).
$$

The per-checkpoint cell is the median $\rho_i$ across valid examples. Aggregate/root cells are arithmetic run means of those per-run medians.

Columns are:

- **target NLL**;
- **token error**, equal to `1 - token accuracy`;
- **exact error**, equal to `1 - exact match`;
- **distance to oracle**;
- **distance to train K**.

**How to interpret the sign**

- `+1`: lower objective consistently ranks lower error or smaller distance as better.
- `0`: no consistent rank relationship.
- `-1`: lower objective consistently ranks worse quality or greater distance as better.
- `—`: correlation is undefined, usually because the quality value is constant for that example/family. This is common for binary exact error when every candidate is either correct or incorrect.

Positive correlation is desirable because both objective and the plotted error/distance metrics are lower-is-better. Always check valid-example coverage in `candidate_family_summary.csv`; a strong median based on a small nonconstant subset can be misleading.

The matrix also has machine-readable companion metrics:

- **Ordering concordance**: fraction of comparable candidate pairs ordered in the same direction by objective and target NLL.
- **Objective-selected NLL regret**: target NLL of the minimum-objective candidate minus the minimum target NLL available in that candidate family. Zero is ideal.

### `trajectory-deltas-vs-k-n<N>.svg`

**What is plotted**

- X axis: sampled K.
- Y axis: median change from the training-K state.
- Two series: inner-objective change and target-NLL change.
- The zero tick/reference is the training-K baseline.

This compactly shows whether objective descent and read quality move together as K changes. If the objective continues downward while target NLL turns upward, the inner objective is misaligned along the write trajectory. The two series have different semantic units, so compare direction and turning points rather than vertical magnitude.

At root level the analogous plots are split into:

- `trajectory-inner-objective-<alias>-n<N>.svg`: inner-objective delta for one alias;
- `target-nll-delta-vs-k-n<N>.svg`: target-NLL delta across aliases, with run-standard-deviation bands.

This split avoids putting incompatible inner objectives on one axis.

## Reading the plots together

Use the following sequence:

1. **Task curves**: Does EM or target NLL improve with K, especially at larger N?
2. **Objective-decrease curve**: Is the inner loop actually minimizing its objective?
3. **Trajectory scatter and correlation matrix**: Does lower objective rank better reads across K and perturbation families?
4. **Matching/interpolation/radial/contours**: Does the objective have discriminative and locally favorable geometry around written memories?

Typical patterns:

- Objective falls, quality improves, and correlations are positive: the writer and objective are aligned.
- Objective falls while quality worsens: more replay will likely amplify an objective-alignment problem.
- Matching/AUC and local geometry improve without task gains: shaping changes the landscape, but the read model does not exploit it.
- Task gains without better standalone geometry: the useful effect may be specific to the trajectory or read path rather than global landscape structure.

These are diagnostics, not pass/fail rules. With one checkpoint per pipeline, paired bootstrap intervals quantify validation-example uncertainty but not training-seed uncertainty. Use repeated runs before attributing differences to shaping.

## Machine-readable results

Important files include:

- `task_examples.jsonl`: per-example task metrics for every N/K cell.
- `task_summary.csv` and `.json`: aggregate task metrics.
- `matching_examples.jsonl` and `matching_summary.csv`: context–memory ranking results.
- `interpolation.csv` and `interpolation_summary.csv`: landscape interpolation curve and monotonicity.
- `radial.csv`: radial objective changes and rise fractions.
- `contours.csv`: values underlying contour heatmaps.
- `quality_alignment/candidates.jsonl`: every trajectory, interpolation, and radial candidate.
- `quality_alignment/correlations.csv`: per-example rank correlations.
- `quality_alignment/candidate_family_summary.csv`: checkpoint or run-aggregate correlation summaries.
- `quality_alignment/trajectory_summary.csv`: task/objective trajectory summaries by K.
- `comparisons/<a>__<b>/seed_<seed>/task_pair_examples.jsonl`: paired example differences for matched seeds.
- `comparisons/<a>__<b>/task_comparison_summary.csv`: mean and sample standard deviation of run-level paired differences.

The root `manifest.json` records selected and skipped checkpoints, hashes, seeds, resolved device, evaluation arguments, and quality-diagnostic settings. Consult it before comparing outputs generated by different commands.
