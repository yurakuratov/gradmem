# Analysis Tools

All maintained analysis code lives here. All generated artifacts live in
`reports/`. Run commands from the repository root in the `rmt` environment.

## Unified MQAR CLI

```bash
conda run -n rmt python -m analysis_tools analysis_tools/examples/landscape.json
conda run -n rmt python -m analysis_tools analysis_tools/examples/extrapolation.json
conda run -n rmt python -m analysis_tools analysis_tools/examples/k_lr_sweep.json
conda run -n rmt python -m analysis_tools analysis_tools/examples/update_svd.json
conda run -n rmt python -m analysis_tools analysis_tools/examples/step_orth_proj.json
conda run -n rmt python -m analysis_tools analysis_tools/examples/memory_search.json
```

The examples are editable JSON templates, not hard-coded experiment launchers.
The default device is CPU. For GPU analysis, inspect `nvidia-smi`, set `device`
to `cuda` in the JSON, and explicitly select a free GPU:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n rmt python -m analysis_tools experiment.json
```

Analysis runs never overwrite an existing report directory. Use a new `output_dir`
for each experiment. Redraw mode adds indexed plots to an existing report without
replacing its plots or data. Config paths in JSON are repository-relative or absolute,
independent of the JSON file location. Outputs must be under `reports/`.

## Redrawing

Redraw plots directly from an existing report, without loading checkpoints or
datasets, running WRITE/READ, or recomputing aggregates, gradients, or SVD:

```bash
conda run -n rmt python -m analysis_tools redraw.json --redraw
```

Redraw is CPU-only: startup does not import model/training modules, query CUDA
availability/device counts, or initialize CUDA. No GPU selection is needed,
even when the original report was evaluated on CUDA.

For example, a plotting-only JSON can contain:

```json
{
  "kind": "landscape",
  "output_dir": "reports/existing_landscape_report",
  "dpi": 220,
  "figsize": [18, 10],
  "landscape": {
    "separate_colorbars": true,
    "include_cosine_sim": false,
    "log_scale": true
  }
}
```

Use the existing report root as `output_dir`. `kind` is optional in redraw mode;
when supplied, it must match the saved report. You may also reuse a full analysis
JSON: input `setups`, datasets, device, and numerical analysis settings are ignored.
Saved setup labels, run counts, axes, bases, and results are authoritative. This
works even if the original checkpoints and datasets are no longer available.

To show only selected saved setups, add top-level `"setup_labels": ["GradMem",
"EnergyGradMem", "EnergyGradMem (sampling)"]` to the redraw JSON. Labels must
exactly match the saved report and be unique; omitted labels are hidden, with
the retained setups staying in their original order. This redraw-only filter
does not change any saved arrays or statistics. The sidecar records the selected
setups and request. Omit it to retain all setups.

Only `dpi`, `figsize`, the global font-size options below, and these per-kind plot
options override saved settings:

Put per-kind options under the matching report-kind key: a landscape report uses
`"landscape": {...}`, not `"extrapolation": {...}`. Options supplied in another
plot-kind section raise an error instead of being silently ignored.

| Kind | Plot Options |
| --- | --- |
| `landscape` | `log_scale`, `separate_colorbars`, `include_cosine_sim`, `show_n_runs`, `report_avg` |
| `extrapolation` | `dash_start`, `dash_start_by_setup`, `dash_end_by_setup`, `outer_loss_reference`, `show_n_runs`, `uncertainty_style`, `show_runs` |
| `step_orth_proj` | `dash_start`, `dash_start_by_setup`, `dash_end_by_setup`, `show_n_runs`, `uncertainty_style`, `show_runs`, `add_step_norm`, `include_cosine_sim` |
| `update_svd` | `dash_start`, `show_n_runs` |
| `k_lr_sweep`, `memory_search` | `show_n_runs` |

Omitted options retain the original config's values or the usual defaults.
For `step_orth_proj`, `"proj_steps": null` is also accepted in redraw requests
to hide all projection panels and keep only enabled norm/cosine panels. This
does not change or recompute any saved statistics. Non-null `proj_steps` changes
are still numerical settings and are not applied by redraw.
Changing K, learning rates, landscape basis/grid, projection span, SVD energy
threshold, or search quantiles requires a new analysis; redraw never applies
these changes or substitutes/recomputes missing results.

All saved datasets are checked for compatible aggregate arrays and required
metadata before any plots are written. Missing or incomplete report artifacts
raise an error. Landscape alignment comes from the saved mean of per-run cosine
maps, never from gradients of averaged loss grids. Existing warnings and
provenance remain intact.

Outputs use names such as `landscapes_redraw_1.png`, then
`landscapes_redraw_2.png`, with a matching JSON sidecar recording the source report
and effective plotting settings. One index is selected across all dataset folders.
K/lr sweeps also generate indexed metric plots such as
`k_lr_sweep_redraw_1_outer_loss.png`. Original config, manifest, NPZ, CSV, and plot
files are left untouched. See `examples/redraw.json` for a minimal template.

## Configuration

- `kind`: `landscape`, `extrapolation`, `k_lr_sweep`, `update_svd`, `step_orth_proj`, or `memory_search`.
- `output_dir`: required report directory.
- `dataset`: `{ "label": "N16", "path": "data/mqar_N16_V8192_L48" }`.
- `datasets`: optionally a list of dataset objects instead of `dataset`.
- `examples`: exact example count (default 256); out-of-range requests fail.
- `example_offset`: first validation row (default 0).
- `batch_size`: evaluation microbatch size (default 64).
- `precision`: shared model/WRITE/READ precision, `"fp32"` (default) or `"fp64"`,
  for every analysis kind. Set `"precision": "fp64"` at the top level to recompute
  trajectories and scores in double precision; update SVD and memory-search
  sampling also preserve it. The effective precision is saved in the manifest
  and per-run JSON. FP64 uses more memory and can be much slower on GPUs; use an
  attention implementation supporting double precision (for example, eager).
  Third-party kernels may use lower-precision internal arithmetic. This may
  resolve rounding-induced degeneracy, but genuinely collinear/zero WRITE
  updates still cannot define a 2D landscape. Use a new report directory;
  `--redraw` only uses saved results and cannot change their numerical precision.
- `seed`: perturbation RNG seed (default 0).
- `dpi`: image resolution in dots per inch (default 220).
- `figsize`: optional `[width, height]` in inches, using finite positive numbers.
  Applies to every plot kind in normal and redraw modes; omitted or `null` keeps
  the automatic size. For K/lr sweeps, the same explicit size applies to the combined
  figure and all three individual metric figures. Pixel dimensions also depend on `dpi`.
- `subtitle_font_size`: subplot title font size in points, for all plot types.
- `tick_font_size`: x/y tick label font size in points, including colorbar ticks.
- `axis_label_font_size`: x/y axis and colorbar label font size in points.
- `legend_font_size`: font size in points for any plot with a legend.
- `setups`: any number of objects, each with a unique `label` and explicit `runs`.

`runs` is a list of string paths, each pointing directly to a checkpoint folder
containing `model.safetensors`:

```json
"runs": ["/REPLACE_WITH_CHECKPOINT_1", "/REPLACE_WITH_CHECKPOINT_2"]
```

Font sizes are independent top-level options and must be finite positive numbers.
Omit them or set them to `null` to use the plot's existing defaults. They apply
to every plot type in normal and `--redraw` modes, without changing global
Matplotlib defaults for subsequent figures. For example:

```json
"subtitle_font_size": 18,
"tick_font_size": 12,
"axis_label_font_size": 15,
"legend_font_size": 12
```

Run folders and weights file paths are not accepted; there is no automatic
checkpoint selection. `inner_lr` can be set at setup
level; otherwise the checkpoint's learning rate is retained. For `k_lr_sweep`,
scalar `inner_lr` overrides are prohibited (see below). Duplicate
checkpoint paths within one setup are rejected. There are no inferred seed
folders or substitutions between similarly named checkpoint families.

## Landscapes

`landscape` options: `basis_steps`, `steps`, `alpha_range`, `beta_range`,
`grid_size`, `log_scale`, `real_grads`, `separate_colorbars`, `include_cosine_sim`, `show_n_runs`, and `report_avg`.

All six plot types support `show_n_runs` in their shared per-kind settings object,
in normal and redraw modes. For example, `"landscape": {"show_n_runs": false}`
hides the parenthesized run counts in setup titles. K/lr sweeps use the same title
convention, memory search uses setup x-axis labels, and line plots use legends.
The default is `true` for landscapes, K/lr sweeps, step projection, and update SVD;
it is `false` for extrapolation and memory search, preserving existing appearances.
Counts remain in saved results and provenance regardless of this display option.

Set global `landscape.include_cosine_sim` to `false` to omit the cosine-alignment
row and its colorbars, leaving only inner WRITE and outer READ loss panels.
The default is `true`. This controls plotting only: alignment computation and
saved alignment grids/statistics are unchanged. It works with either colorbar
layout and cannot differ between setups.

Set global `landscape.report_avg` to `true` (default `false`) to add
`Avg. cosine: ...` beneath each column's cosine panel. It is ignored when
`include_cosine_sim` is `false`. The scalar is the equal-run average of each
run's mean over its valid cosine cells, not the cosine of averaged loss grids
or the mean of a partially covered aggregate cosine map. This is 2D
coordinate-space gradient alignment, not full-memory gradient alignment.

This also works on existing reports without rebuilding landscapes:

```json
{
  "output_dir": "reports/n64_shaping_v2_fp64",
  "landscape": {"include_cosine_sim": true, "report_avg": true}
}
```

Run with `conda run -n rmt python -m analysis_tools redraw.json --redraw`.
Redraw reads the saved scalar means from `alignment_summary.csv`; missing or
invalid statistics raise an error rather than silently substituting a different
aggregation. Original results and plots remain unchanged.

Set global `landscape.separate_colorbars` to `true` to give every panel its own
colorbar. Inner and outer loss panels then use their own value ranges and
respect `log_scale`; alignment panels retain the fixed cosine range `[-1,1]`.
The default, `false`, uses one shared colorbar and scale per row, allowing direct
color comparisons across setups. With separate colorbars, compare loss values
using each panel's colorbar rather than colors alone. This is a shared layout
option and cannot differ between setups.

Landscape colorbars show the lower bound, midpoint, and upper bound, without
minor ticks. The midpoint is geometric for logarithmic loss scales and arithmetic
for linear scales; cosine alignment uses `-1, 0, 1`. Only the rightmost colorbar
in each row carries the vertical metric label.

Set `"real_grads": true` in the global or setup-level `landscape` object to
compute alignment using autograd derivatives with respect to `(alpha, beta)`
at each grid point. The default, `false`, uses finite differences of the sampled
loss grids. Autograd requires backward passes and additional activation memory;
reduce `batch_size` if needed. Both modes measure 2D coordinate-space alignment,
not full-memory gradient alignment, and preserve the same loss surfaces.

`grid_size` controls sampling density, NOT image DPI. It accepts a single
integer or `[alpha_points, beta_points]`, each at least 3.

With `basis_steps: [2,3]`, the plane is anchored at `M1`, with directions
`M2-M1` and `M3-M2`: `M1=(0,0)`, `M2=(1,0)`, `M3=(1,1)`.
Other increasing step pairs are allowed; intermediate trajectory points are
projected into the plane. Rank validation and projection use float64,
unit-normalized displacement columns, so unequal step lengths do not cause a
false rank failure; the plotted axes remain the original, unnormalized updates.
Zero or collinear displacement pairs and nonfinite trajectories fail with
batch-local example indices rather than fabricating coordinates. Extremely small
updates may reflect float32 rounding rather than a meaningful optimization direction;
passing the rank check alone does not establish that a landscape is informative.
Memory is per-example, and READ never sees context.
Each setup may override `basis_steps` and `steps` in its own `landscape` object;
axis ranges and grid sizes remain shared across compared columns.

Set `basis_steps` directly on each setup; all its runs use that basis:

```json
{
  "landscape": {"basis_steps": [1, 2]},
  "setups": [{
    "label": "My setup",
    "basis_steps": [2, 3],
    "runs": [
      "/REPLACE_WITH_CHECKPOINT_1",
      "/REPLACE_WITH_CHECKPOINT_2"
    ]
  }]
}
```

Basis precedence is **setup `basis_steps` > setup `landscape.basis_steps` > global
`landscape.basis_steps` > `[1,2]`**. Both runs in this example use `[2,3]`.
The inherited WRITE-step count is extended when needed to reach the chosen basis.
Memory-state labels start at the plane's origin: `[2,3]` displays `M1`, `M2`, `M3`,
not `M0`, `M1`, `M2`. Basis overrides inside run objects are rejected.
Effective settings are saved in each run's JSON and in the checkpoint manifest.
Loss/alignment maps are averaged in each run's own normalized plane. Plotted
trajectory coordinates are mean projections; for states not reached by every
run, only the available runs contribute (`trajectory_runs` records the counts).

The plot has arbitrary setup columns and three rows: configured inner WRITE
objective, outer READ target loss, and gradient cosine alignment. Reconstruction
and mixed energy/reconstruction checkpoints use their actual weighted WRITE
objective, not an incorrectly labeled energy-only proxy. Each loss row has a
shared, data-bounded scale. Palettes are viridis/magma/coolwarm; axes show only
outer integer ticks. Log colorbars use in-range `1x10^n`/`3x10^n` ticks and two
scientific fallback ticks for narrow ranges.
Loss rows retain white contour lines; alignment includes a black zero-cosine contour.

**Aggregation:** example-averaged loss grids are computed separately per run.
Cosine maps (finite-difference or autograd) are computed separately per run, THEN averaged
equally across runs. The scalar summary is the arithmetic mean of per-run
cosine means. Cosine of averaged loss grids is never used. These are 2D
coordinate-space diagnostics, not full-memory Euclidean gradient alignment.
In autograd mode, coordinate derivatives are averaged over examples before
computing each run's cosine map, not example-wise cosines averaged afterward.

## Extrapolation

Set `extrapolation.uncertainty_style` to `"band"` for transparent arithmetic
mean +/-1 sample-SD bands instead of error bars (default `"errorbar"`). Percentage
bands are displayed within `[0,100]`. Log-loss bands whose lower endpoint is
nonpositive are truncated at the visible log-axis floor and marked with downward
triangles. Neither the SD nor saved statistics
are changed. The automatically chosen floor is half the smallest positive mean,
visible per-run loss, or positive lower SD endpoint across setups.
Set `extrapolation.show_runs` to `true` (default `false`) to overlay thin, faint
individual-run curves in the matching setup color. These follow `dash_start`
without adding legend entries. Both options work in redraw mode; `show_runs`
requires the saved per-run NPZ files but never re-evaluates the model.

`extrapolation.steps` is any list of non-negative K values. Every explicit run
is evaluated on the identical example indices. Plots show three panels: token
accuracy (%), outer READ loss, and exact match (%), with equal-run averages and
sample SD. Token accuracy and exact match use linear y-axes; outer READ loss
always uses a logarithmic y-axis. This also applies to redraws. All three metrics
are included in CSV/NPZ results. Single-run SD is zero, not an estimate of seed variability.
WRITE uses inference-time query-independent updates, with training auxiliaries
disabled. Model weights remain frozen; memory differentiation stays enabled.

Optional shared plotting settings go in the `extrapolation` object:

```json
"extrapolation": {
  "steps": [0, 1, 2, 4, 8, 16],
  "dash_start": 2,
  "show_n_runs": true
}
```

`dash_start` defaults to `null` (all curves solid). Set it to an evaluated K to
draw a vertical dashed line in all three panels and make every curve dashed from
that K onward, keeping its color and a single legend entry. With this option,
`steps` must be strictly increasing. K ticks remain equally spaced, including
when the evaluated K values are nonuniform.

To choose a different boundary for each setup's line, add `dash_start_by_setup`
inside `extrapolation` (keys must exactly match setup labels):

```json
"extrapolation": {
  "steps": [0, 1, 2, 4, 8, 16],
  "dash_start": 2,
  "dash_start_by_setup": {
    "GradMem": 4,
    "EnergyGradMem": 8,
    "Always solid": null
  }
}
```

Unlisted setups inherit `dash_start`; an explicit `null` keeps that setup solid.
Each override must be an evaluated K. It applies to the mean and optional thin
individual-run curves in every panel, preserving colors and single legend entries.
Only shared `dash_start` draws a gray vertical reference line; overrides change
line styles without adding reference lines. This is also a redraw-only option,
so no numerical recomputation is needed. Redraw retains the saved mapping when
omitted; a supplied mapping replaces it (use `{}` to clear all overrides).

Optional `dash_end_by_setup` sets the last dashed point for each listed setup,
then resumes solid lines from that point to the next and all subsequent points.
For example, with `"dash_start_by_setup": {"GradMem": 7}` and
`"dash_end_by_setup": {"GradMem": 11}`, GradMem is solid through K=7,
dashed from K=7 to K=11, and solid again from K=11 onward. The end must be an
evaluated step strictly after that setup's effective start. Omitted or `null`
ends retain the dashed style through the final point. This applies to every
panel and individual-run overlay, and also works under `step_orth_proj` and
in redraw mode. These are manually chosen styling boundaries, not automatic
per-run loss-threshold classification.

Set `extrapolation.outer_loss_reference` to `1.0` to highlight that loss with
a horizontal dark-gray dashed line in the outer READ loss panel only. The
default is `null` (no reference); other finite positive loss levels are allowed.
This plotting option also works with `--redraw`.

`show_n_runs` defaults to `false`; `true` appends the evaluated run count to each
setup's legend label, e.g. `GradMem (3 runs)` or `GradMem (1 run)`.

## K/LR Sweeps

`kind: "k_lr_sweep"` evaluates fixed MQAR checkpoints over inference-time WRITE
step counts and inner learning rates. There is no retraining or checkpoint
selection based on sweep outcomes. WRITE remains query-independent, READ never
sees the original context, and training auxiliaries are disabled. Model weights
remain frozen while memory differentiation stays enabled.

Set the global grid in `k_lr_sweep`; a setup's `k_lr_sweep` object can override
either axis for all of its runs:

```json
{
  "k_lr_sweep": {"steps": [0, 1, 2, 4, 8], "inner_lrs": [0.1, 0.2, 0.4, 0.8]},
  "setups": [{
    "label": "My setup",
    "k_lr_sweep": {"steps": [0, 2, 4, 8, 16]},
    "runs": ["/REPLACE_WITH_CHECKPOINT"]
  }]
}
```

Grid precedence is **setup `k_lr_sweep` > global `k_lr_sweep`**, independently
for each axis; the example inherits the global `inner_lrs`. Supply both axes
after merging. Both must be nonempty lists with distinct values. `steps` must
contain strict non-negative integers, not booleans or fractional numbers;
`inner_lrs` must contain finite positive numbers, not booleans. Axis order is
preserved. Scalar `inner_lr` overrides at global, setup, or run level are
prohibited for sweeps to avoid ambiguity; use the `inner_lrs` grid instead.
Grid overrides belong on setups, not inside run objects.
Nonfinite metrics abort the sweep rather than silently dropping failed cells.

Every grid cell uses identical validation example indices. Metrics are outer
READ target loss, token accuracy, and exact match, averaged equally across runs
with sample SD (`ddof=1`); single-run SD is zero, not evidence of seed stability.
The combined plot has three metric rows and one column per setup, with K rows
and learning-rate columns in each annotated heatmap. Each metric has a shared
color scale across setups; accuracy means and SDs are shown in percentage points.
White cell borders and contrast-aware labels show mean +/- sample SD. A star
marks the checkpoint-trained `(K, inner_lr)` only if all runs in that setup have
the same recorded training setting and that exact pair is present on its grid.
Missing, differing, or off-grid training settings are not marked.

Each dataset report includes `per_run.csv` (one row per run and grid cell),
`summary.csv` (equal-run mean and sample SD per cell), per-run metrics NPZ files,
and aggregate metrics NPZ files with explicit `steps` and `inner_lrs` axes.
Metric matrices have shape `[K, inner_lr]`. The input `config.json`, resolved
checkpoint `manifest.json`, and per-run JSON configs record effective grids,
checkpoint provenance, and original training settings separately from inference
settings. Plots include `k_lr_sweep.png`, `k_lr_sweep_outer_loss.png`,
`k_lr_sweep_token_accuracy.png`, and `k_lr_sweep_exact_match.png`.

`examples/k_lr_sweep.json` defaults to CPU, 256 examples, and batch size 64.
Its `/REPLACE_WITH_*_CHECKPOINT_*` paths are deliberately nonexistent placeholders:
replace every path with an actual MQAR checkpoint folder containing
`model.safetensors` before invoking it. There is no automatic checkpoint selection,
inferred seed coverage, or substitution of similarly named runs. Choose a new
directory under `reports/` for every sweep; existing reports are never overwritten.

## Step Projection

`kind: "step_orth_proj"` plots how much of each WRITE displacement lies in the
span of the next `proj_steps` displacements in full flattened memory space:

```text
d_k = M_k - M_(k-1)
P = proj_steps
ratio_k = ||Proj_span(d_(k+1), ..., d_(k+P))(d_k)||_2 / ||d_k||_2
```

Set `"step_orth_proj": {"steps": 16, "proj_steps": 2}` for K=16 plotted points,
k=1,...,16, each projected onto the next two steps. `proj_steps` accepts either
a positive integer (default 2) or a nonempty list of distinct positive integers:
1 projects onto a line, 2 onto a plane, and larger values onto the following
steps' span, whose dimension may be smaller if dependent. A list creates one
side-by-side panel per value, in the supplied order, with shared x/y axes,
consistent setup colors, and one legend. For example:

```json
"step_orth_proj": {
  "steps": 16,
  "proj_steps": [1, 2, 4],
  "uncertainty_style": "band",
  "show_runs": true,
  "add_step_norm": true,
  "include_cosine_sim": true,
  "show_n_runs": true,
  "dash_start": 2
}
```

Set `"proj_steps": null` to skip projection-ratio computation and panels entirely.
At least one of `add_step_norm` or `include_cosine_sim` must be enabled. For just
the two auxiliary panels:

```json
"step_orth_proj": {
  "steps": 16,
  "proj_steps": null,
  "add_step_norm": true,
  "include_cosine_sim": true,
  "uncertainty_style": "band",
  "show_runs": true
}
```

This collects K updates for norms only, or K+1 when cosines are enabled. Norms
stay leftmost and cosine stays rightmost; either can also be shown alone.
CSV rows have blank `proj_steps` and no projection-ratio columns in this mode.
NPZs retain an empty projection axis. Omitting `proj_steps` still defaults to 2;
it is not the same as explicitly setting `null`. Redraw can hide projection
panels this way if the requested auxiliary data was already recorded.

The analysis collects K+max(P) WRITE updates once per batch and reuses that
trajectory for all requested spans, so every plotted point has its requested
P following steps. Smaller-span panels use only their own following-step window.
This is a step-index curve from that trajectory, not separate evaluations of a
final-memory metric at different K. Settings are shared across setups and runs;
checkpoint paths and optional `inner_lr` overrides follow the usual conventions.
Only prefix memory without per-sample projection is supported. WRITE remains
query-independent with frozen model weights and training auxiliaries disabled.

Ratios are computed per example, then averaged over examples within each run.
The line plot shows equal-run means with sample SD across runs (`ddof=1`);
single-run SD is zero. Use `uncertainty_style: "band"` for bold mean curves with
transparent +/-1 sample-SD shading, displayed within `[0,1]`, or `"errorbar"`
(default) for SD whiskers. `show_runs: true` adds faint individual-run curves;
the default is `false`. SDs are computed across run-level example means, not
pooled across examples, and remain unchanged in saved data. Both options work
with single/multiple panels and redraw, as do font sizes, `figsize`,
`show_n_runs`, and `dash_start`. A ratio of 1 means the step lies in the following span,
and 0 means it is perpendicular. This is a norm ratio, not a squared norm ratio
or an orthogonal-residual ratio. Collinear following steps use their line;
all-zero following steps give projection zero. A zero current step is undefined
and raises an error by default rather than silently dropping an example. Projection uses
a pseudoinverse of unit-normalized columns in float64, with numerical rank
tolerance `max(memory_dimension, proj_steps) * float64 eps`.

### Consecutive-Step Cosine

Set `step_orth_proj.include_cosine_sim` to `true` (default `false`) to add one
panel on the far right showing the signed cosine between `d_k = M_k-M_(k-1)`
and `d_(k+1)`, for k=1,...,K. Cosines are computed per example in full flattened
memory space, averaged over examples within each run, then averaged equally
over runs with sample SD. This is not cosine between averaged updates, and it
is independent of `proj_steps`. The existing trajectory already includes the
extra update needed for the final point; no additional WRITE calls are made.

The panel shares x-axis, setup colors, styling, and legend with the other panels,
but uses its own linear y-axis spanning `[-1,1]`. Negative values mean opposing
directions. SD bands are displayed within `[-1,1]`; saved SDs are unchanged.
Both consecutive updates must be nonzero: `zero_step_policy: "error"` rejects
undefined cosines, while `"nan"` omits them with explicit valid/total pair and
run counts, gaps at all-undefined points, and red rings on affected finite means.
These counts may differ from projection-ratio coverage because a zero following
update makes its cosine undefined even when the projection ratio is defined.

Enabled analyses save `cosine_sim`/`cosine_sim_per_example` per run, aggregate
mean/SD and coverage arrays, and cosine columns in CSVs. `--redraw` can show/hide
the panel from those saved results. Older reports without recorded cosines
require a new analysis with this option enabled; redraw never computes missing
cosines. With norms enabled, panel order is norms, projection spans, cosine.

### WRITE-Step Norms

Set `step_orth_proj.add_step_norm` to `true` (default `false`) to prepend one
leftmost line-plot panel of `||M_k - M_(k-1)||_2`, k=1,...,K, for every setup.
It shares the WRITE-step x-axis and setup colors/legend with the projection
panels, but has its own logarithmic y-axis. Projection panels remain linear.
The norm panel is independent of the
requested projection spans, so multiple `proj_steps` values still add only one
norm panel. `figsize`, font controls, `dash_start`, `show_n_runs`,
`uncertainty_style`, and `show_runs` apply to it as well. Norm SD bands/errorbars
with nonpositive lower bounds are truncated at the log-axis floor, marked with
downward triangles. Zero norm values remain zero
in saved statistics; the log plot leaves curve gaps and marks zero means at the
floor rather than inventing positive updates. The floor is half the smallest
positive visible mean, per-run norm, or lower SD endpoint, or `1e-12` if there
are no positive norms. Norm bands are not capped at one.

Each norm is computed per example from the stored memory displacement, then
averaged over **all examples**, including exact zero updates. The setup curve
is the equal-run mean of those run-level means, with sample SD across runs.
Unlike conditional projection ratios under `zero_step_policy: "nan"`, norm means
do not exclude stalled examples. These are mean norms, not the norm of a mean
displacement. No extra WRITE trajectory is collected.

New analyses always save `step_norm` (`[K]`) and `step_norm_per_example`
(`[examples,K]`) per run, plus aggregate `step_norm_mean`/`step_norm_std`, even
when the panel is disabled. CSV rows also include step-norm statistics (repeated
for each projection span). Therefore a later `--redraw` can enable the norm
panel without recomputation. Reports produced before norms were recorded cannot
recover them from projection ratios alone; enabling this option raises a clear
error and requires a new analysis rather than silently recomputing WRITE.

### Zero WRITE Steps

`step_orth_proj.zero_step_policy` defaults to `"error"`. An exactly zero
displacement has no direction, so its projection ratio is `0/0`, not zero.
This can occur even with a finite, nonzero gradient if an SGD update is too small
to change stored float32 memory. It is not evidence of NaNs in the model and does
not by itself establish convergence.

Explicitly set `"zero_step_policy": "nan"` to continue while retaining undefined
ratios as missing values. Each run's curve becomes the conditional mean over
examples with a nonzero current WRITE step. Setup means and sample SDs are then
computed over valid run means at each point, with equal weight per valid run.
A single valid run has SD zero; no valid runs gives a gap, not a fabricated
zero ratio. Following zero displacements still contribute their true numerical
span. Nonfinite updates remain errors under either policy.

This changes the averaging cohort as K increases. Interpret the conditional
curves with their coverage, not as fixed-cohort results. Warnings report missing
ratios. Undefined values are omitted
from plotted curves. A small hollow red circle surrounds each finite setup-mean
point affected by omitted examples or runs, using the saved coverage counts
(even when individual-run curves are hidden). All-undefined points remain gaps,
since there is no mean point to mark. The norm panel is unaffected: a zero update
has a defined norm even when its projection ratio is undefined.
Per-run NPZs save the full
per-example NaN mask via the ratios, plus `projection_valid_examples` and
`projection_total_examples` arrays. Aggregate NPZs save those mean counts and
`projection_valid_runs`. Per-run CSV rows include `valid_examples`/`total_examples`;
summary CSV rows include `valid_runs`, `valid_examples_mean`, and
`total_examples_mean`. Redraw preserves the missing values and validates the
counts, without recomputing projections. This policy must be chosen for a new
analysis; changing it in a redraw request does not recover missing numerical data.

Each dataset report contains `step_orth_proj.png`, `per_run.csv`, `summary.csv`,
per-run NPZ files (including `[examples,K]` ratios for scalar `proj_steps`), and
setup-aggregate NPZ files with means, SDs, and explicit step axes. For a list of
P projection spans, per-example ratios have shape `[examples,P,K]` and per-run
and aggregate mean/SD curves have shape `[P,K]`. The `proj_steps` axis is saved
explicitly in list-mode NPZ files, and CSVs contain one row per span and step.
Even a one-element list retains its panel axis; scalar inputs keep the existing
scalar report format. Input JSON, checkpoint manifest,
effective configs, and example indices are saved as for other analyses.
Unequal run coverage is reported. The shared trajectory collector currently
replays WRITE for each length 1,...,K+max(P), incurring quadratic WRITE cost.
Redraw uses the report's saved projection spans and never computes additional
ones. Request a new analysis in a new output directory to add/change spans;
redraw-only `proj_steps` changes are ignored, like other numerical settings.
See `examples/step_orth_proj.json` for an example configuration; use a new output
directory under `reports/`.

Optional `step_orth_proj.dash_start` marks a plotted WRITE step with a vertical
dashed line; all curves become dashed from that point onward without extra
legend entries. It must be an integer from 1 through `steps`, or `null` (default,
all curves solid). Set `step_orth_proj.show_n_runs` to `false` to hide run counts
in legend labels; its default is `true`.

`step_orth_proj.dash_start_by_setup` provides the same per-label overrides as
extrapolation, including shared fallback, explicit `null` for solid curves, and
redraw support. Values must be plotted step indices from 1 through `steps`.
For example, `"dash_start_by_setup": {"GradMem": 2, "EnergyGradMem": 4}`
sets different boundaries for the two lines in every projection-span, norm,
and cosine panel, including optional individual-run overlays. Only shared
`dash_start` controls the gray vertical reference line.

## Update SVD

`kind: "update_svd"` measures the singular-value spectrum of a K-step WRITE
update trajectory. Set shared options in the global `update_svd` object:

```json
"update_svd": {"steps": 8, "energy_threshold": 0.95}
```

Optional `update_svd.dash_start` marks a singular-value **rank**, not a WRITE
step, with a vertical dashed line and makes all spectra dashed from that rank
onward, without extra legend entries. It must be an integer from 1 through
`steps`, or `null` (default). If the reduced spectrum has fewer ranks than the
chosen boundary, its curve remains solid. Set `update_svd.show_n_runs` to `false`
to hide run counts in legends; its default is `true`. Retained-energy rank
markers and annotations are unchanged.

`steps` is K and must be a positive integer, not a boolean or fractional number.
`energy_threshold` must be in `(0,1]`. These settings are shared across all
setups and runs; use explicit setup labels and run paths as in the other kinds.
Only prefix memory without per-sample projection is supported. WRITE remains
inference-time and query-independent, with training auxiliaries disabled.

For each example, flatten each displacement `M_k-M_(k-1)` into a row of a
`[K,D]` matrix, where D is the flattened memory dimension and `k=1,...,K`.
Compute reduced `svdvals`, yielding `min(K,D)` raw singular values in descending
order. Values are not normalized. Average spectra across examples within each
run first, then average those per-run spectra equally within each setup.
Error bars show sample SD across runs (`ddof=1`); single-run SD is zero, not an
estimate of seed variability.

The rank marker is the smallest rank r whose cumulative fraction of squared
**displayed setup-mean singular values** reaches `energy_threshold`:
`sum(mean_singular_values[:r]**2) / sum(mean_singular_values**2) >= energy_threshold`.
It is not the average of individual examples' or runs' retained ranks, nor a
threshold applied to averaged squared singular values. A zero spectrum has
undefined retained energy and receives no rank marker.

Although the diagnostic describes a K-step trajectory, the existing
`collect_trajectory` replays WRITE separately for `K=1,...,K` rather than
collecting all states in one pass. It therefore executes `K*(K+1)/2` WRITE
updates per example, a quadratic WRITE cost in K.

Each dataset report includes `update_svd.png`, `per_run.csv`, `summary.csv`,
and NPZ files preserving individual example spectra, per-run spectra, setup
means and sample SDs, and explicit rank axes. The input `config.json`, resolved
checkpoint `manifest.json`, and per-run effective configs record provenance
and analysis settings; example indices are saved as for the other kinds.

`examples/update_svd.json` defaults to CPU, 256 examples, batch size 64, K=8,
and an energy threshold of 0.95. Replace every `/REPLACE_WITH_*_CHECKPOINT_*` path
with an actual MQAR checkpoint folder containing `model.safetensors` before
invoking it, without inferred seed coverage or substitutions. Choose a new
directory under `reports/`; existing reports are never overwritten.

## Memory Search

`memory_search` options: `steps`, `num_samples`, `radius_scale`, and sorted
`quantiles` in `[0,1]`. Perturbations use the model's norm-preserving candidate
sampler around final `MK`; radius is relative to `MK-M(K-1)`.

Reports contain baseline, raw candidate outcomes, and beneficial percentiles
for accuracy, exact match, inner loss, and outer loss. Harmful candidates
contribute zero gain, as requested. These are metric-wise oracle outcomes,
not one common candidate selected for all metrics and not inference-time
query-guided writing. Stacked bars use incremental percentile improvements;
loss improvements overlay downward from the outlined baseline. Historical
target-guided rollout analysis remains under `legacy/`.

## Files And Reports

- `analyze.py`: unified JSON execution, aggregation, serialization.
- `library.py`: checkpoint resolution/loading, data, memory scoring, finite differences.
- `plots.py`: arbitrary setup-count plots and beneficial percentile bars.
- `diffusion/`: preserved D-only, K-only, K-D sweeps and error-placement analysis.
- `legacy/`: retained KV, mixed-objective, memory-search rollout and cache diagnostics.
- `examples/`: JSON inputs for maintained workflows.

Reports save the input `config.json`, resolved `manifest.json`, dataset metadata,
per-run NPZ grids/metrics and effective model configs, averaged NPZ data, CSV
summaries, and plots. Existing historical outputs were moved intact from
`analysis/` into `reports/`; see `reports/migration.md`. Historical bad results
remain explicitly marked invalid; migration does not certify their correctness.

## Diffusion And Legacy Commands

```bash
conda run -n rmt python -m analysis_tools.diffusion.analyze_diffusion_gradmem_kd_sweep --help
conda run -n rmt python -m analysis_tools.diffusion.analyze_diffusion_pair_errors --help
conda run -n rmt python -m analysis_tools.legacy.evaluate_energy_shaping --help
```

Existing CLI flags remain supported for these maintained specialist tools.
Do not create root-level one-off plotters or experiment-specific evaluators.
