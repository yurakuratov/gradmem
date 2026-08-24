# Mix-N8 Report

`report.tex` is a NeurIPS 2025 preprint-style report of the current mix-N8 experiments.

Compile from this directory with:

```bash
latexmk -pdf report.tex
```

The diagnostic data in `mix_write_components.csv` and `mix_write_components.json` were generated from 256 frozen validation examples with `analyze_mix_write_components.py`. `mix_k_sweep.csv` was generated with `evaluate_mix_k_sweep.py` for evaluation-time `K` in `{0,1,2,4,8}`.

Regenerate all PDF and SVG figures with:

```bash
conda run -n rmt python plot_mix_report_figures.py \
  --runs-root runs/mix-N8-K2V2-V62_1M \
  --components-csv reports/mix_n8/mix_write_components.csv \
  --k-sweep-csv reports/mix_n8/mix_k_sweep.csv \
  --stability-csv reports/mix_n8/stability_runs.csv \
  --output-dir reports/mix_n8/figures
```

Run the figure command from the repository root.

The frozen K-sweep used batch size 32 and the first 256 examples from `data/mix-N8-K2V2-V62_1M/valid`. Exact checkpoint suffixes are listed in `report.tex`; pass them to `evaluate_mix_k_sweep.py` with aliases `gradmem`, `energy-success`, and `energy-failed` to reproduce `mix_k_sweep.csv`.

The plateau-descendant WRITE-step sweep is stored in `plateau_k_extrapolation.csv`. The local loss analysis in `loss_landscape_grid.csv` and `loss_landscape_trajectory.csv` used `analyze_mix_loss_landscapes.py` with 12 examples, a 13-by-13 grid over `[-3,3]^2`, transition points `0 1 2 4 8`, and random-direction seed `20260813`. Each row uses its own transition displacement (`0→1`, `1→2`, `2→4`, or `4→8`) as the x axis and a separately seeded equal-norm orthogonal y direction.

```bash
conda run -n rmt python plot_mix_landscapes.py \
  --k-sweep-csv reports/mix_n8/plateau_k_extrapolation.csv \
  --grid-csv reports/mix_n8/loss_landscape_grid.csv \
  --trajectory-csv reports/mix_n8/loss_landscape_trajectory.csv \
  --output-dir reports/mix_n8/figures
```

Landscape coordinates are per-example and per-interval local units: `x=1` is exactly the corresponding interval endpoint. Cross-row and cross-checkpoint contour widths are therefore not directly comparable in physical memory distance.
