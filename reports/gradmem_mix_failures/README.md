# GradMem Mix-N8 Failure Report

Generate the frozen-checkpoint analysis from the repository root:

```bash
conda run -n rmt python analyze_gradmem_mix_failures.py \
  --runs-root runs/mix-N8-K2V2-V62_1M \
  --data-path data/mix-N8-K2V2-V62_1M \
  --output-dir reports/gradmem_mix_failures/data \
  --max-examples 512 \
  --batch-size 32 \
  --device cuda
```

Generate figures and tables:

```bash
conda run -n rmt python plot_gradmem_mix_failures.py \
  --data-dir reports/gradmem_mix_failures/data \
  --output-dir reports/gradmem_mix_failures
```

Compile the report from this directory with:

```bash
latexmk -pdf report.tex
```
