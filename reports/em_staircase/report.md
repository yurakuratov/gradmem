# Analysis of the Exact-Match Staircase

## Scope and run alignment

This report compares:

- **Original run:** `runs/N8-K2V2-V62_1M/energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_energy_grad_second_stepalign0.1_iread0.1_bs_64_lr_1e-04_fp32/run_1`
- **Repeat with checkpoints:** `runs/N8-K2V2-V62_1M/energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_energy_grad_second_stepalign0.1_iread0.1_bs_64_lr_1e-04_fp32_extra_cpt/run_1`

The recorded substantive hyperparameters are matched, and both configurations record seed 143, but the effective initialization seeds differed because of the seed-setting problem. The two trajectories therefore cannot be compared checkpoint-by-checkpoint at equal outer step. I align them by their validation regimes instead: the repeat's intermediate checkpoints explain the token-error structure of plateaus also visible in the original metric history.

The original metric history extends to outer step 859,500. Its only retained checkpoint is step 609,500, which has 97.10% exact match (EM). The repeat currently has checkpoints every 2,500 steps through 127,500 and metrics through 128,000; it remains on the approximately 12% EM plateau.

## Main result

The staircase is primarily a **target-token-order phenomenon**:

1. The repeat first learns the **second value token** while the first value token remains poor.
2. At the repeat's stable 12% plateau, final-WRITE (`M2`) accuracy is 94-95% for value token 2 but only about 14-15% for value token 1. Approximately 82% of all examples have *only* value token 2 correct.
3. The original run has the same aggregate error signature on its 12% plateau. Its subsequent approximately 11-12 point EM jumps mostly convert one-token-correct examples into both-token-correct examples. This indicates that the later staircase is predominantly improvement of value token 1 after value token 2 has already been learned.
4. The original step-609,500 checkpoint confirms the interpretation directly: value-1 accuracy is 97.18%, value-2 accuracy is 99.90%, and 140 of its 145 non-EM examples have only value token 2 correct, meaning the remaining error is at value token 1.

This is not a staircase in ordinary context reconstruction quality. Sharp READ improvements can occur while context reconstruction of the corresponding token changes little or gets worse.

## Inferring errors from the original metrics

Only two content tokens, `V1` and `V2`, are scored. Therefore, from EM and token accuracy (`TA`) alone:

```text
P(both correct)    = EM
P(exactly one)     = 2 * (TA - EM)
P(neither correct) = 1 - EM - P(exactly one)
```

The original trajectory has the following representative regimes:

| Outer step | EM | Token accuracy | Exactly one correct | Neither correct |
|---:|---:|---:|---:|---:|
| 20,000 | 5.08% | 25.63% | 41.10% | 53.82% |
| 40,000 | 4.88% | 25.18% | 40.60% | 54.52% |
| 45,000 | 10.46% | 46.40% | 71.88% | 17.66% |
| 75,000 | 11.94% | 53.23% | 82.58% | 5.48% |
| 80,000 | 22.84% | 59.24% | 72.80% | 4.36% |
| 95,000 | 34.92% | 65.79% | 61.74% | 3.34% |
| 110,000 | 47.54% | 72.94% | 50.80% | 1.66% |
| 135,000 | 59.18% | 79.09% | 39.82% | 1.00% |
| 145,000 | 70.56% | 84.97% | 28.82% | 0.62% |
| 160,000 | 83.68% | 91.65% | 15.94% | 0.38% |
| 170,000 | 94.20% | 96.99% | 5.58% | 0.22% |
| 609,500 checkpoint | 97.10% | 98.54% | 2.88% | 0.02% |

The first transition, from approximately 5% to 12% EM, reduces the large neither-correct population. After the 12% plateau, the neither-correct population is already close to zero. Each later transition instead moves roughly 11-12% of examples from exactly-one-correct to both-correct. The repeat checkpoints identify the already-correct token in the exactly-one population as almost always `V2`.

The original run's sharp transition windows are approximately:

| Window | EM before | EM after | Main aggregate conversion |
|---:|---:|---:|---|
| 42,500-45,000 | 5.26% | 10.46% | Neither correct -> one correct |
| 79,000-81,500 | 12.00% | 23.58% | One correct -> both correct |
| 90,500-93,000 | 23.34% | 34.98% | One correct -> both correct |
| 104,000-106,500 | 34.92% | 46.76% | One correct -> both correct |
| 134,000-136,500 | 49.20% | 60.12% | One correct -> both correct |
| 141,500-144,000 | 60.16% | 72.24% | One correct -> both correct |
| 154,500-157,000 | 72.34% | 84.50% | One correct -> both correct |
| 166,000-168,500 | 83.58% | 94.62% | One correct -> both correct |

The near-regular jump size is striking, but it should not by itself be interpreted as one of the eight KV positions being acquired at each jump. The repeat's position-conditioned results below show a more complicated transient and a position-uniform 12% plateau.

## Direct token errors in the repeat

The table reports final-WRITE (`M2`) READ behavior on all 5,000 validation examples.

| Checkpoint | EM | V1 accuracy | V2 accuracy | Only V1 correct | Only V2 correct | Neither correct |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.98% | 10.48% | 10.58% | 9.50% | 9.60% | 79.92% |
| 12,500 | 2.22% | 14.78% | 15.40% | 12.56% | 13.18% | 72.04% |
| 20,000 | 3.62% | 14.34% | 25.60% | 10.72% | 21.98% | 63.68% |
| 25,000 | 5.66% | 15.20% | 36.52% | 9.54% | 30.86% | 53.94% |
| 32,500 | 5.46% | 14.22% | 37.66% | 8.76% | 32.20% | 53.58% |
| 35,000 | 7.02% | 15.06% | 49.00% | 8.04% | 41.98% | 42.96% |
| 40,000 | 7.30% | 14.84% | 49.74% | 7.54% | 42.44% | 42.72% |
| 42,500 | 11.50% | 13.76% | 91.60% | 2.26% | 80.10% | 6.14% |
| 50,000 | 13.00% | 14.74% | 93.78% | 1.74% | 80.78% | 4.48% |
| 125,000 | 12.38% | 14.76% | 94.60% | 2.38% | 82.22% | 3.02% |

The early rises are almost entirely in `V2`; `V1` remains near 14-15% from step 12,500 onward. The largest observed transition, 40,000 -> 42,500, is a jump in `V2` from 49.74% to 91.60%, while `V1` slightly declines from 14.84% to 13.76%.

The errors are not predominantly punctuation or another special token after WRITE. At the stable 12% plateau, wrong `V1` predictions are broadly distributed over alphabet characters. The model is not failing because it emits `!` or `|` in the first value position.

## Pair-position and character controls

The repeat's intermediate `V2` learning is position-biased:

- At step 20,000, queried pair positions 1 and 2 have 49-54% `V2` accuracy, while positions 3-8 have only 15-19%.
- At step 25,000, position 5 is at 96%, positions 1-2 are at 48%, and most other positions remain at 18-23%.
- At step 40,000, position 5 is at 95%; positions 1, 2, 6, 7, and 8 are near 48-51%; positions 3-4 are near 27-28%.
- At step 42,500, the position bias abruptly collapses: all eight positions have 88-93% `V2` accuracy. `V1` remains only 11-16% at every position.
- At step 125,000, all positions have 93-97% `V2` accuracy and 12-18% `V1` accuracy.

Thus, early optimization proceeds through position-specific partial solutions, but the stable 12% plateau is not caused by solving only one pair position. The 40,000 -> 42,500 event is best described as **globalization of the second-token solution across all pair positions**.

It is also not simply a subset of easy alphabet symbols. At step 42,500, every one of the 62 possible `V2` symbols has at least 80% accuracy, while no `V1` symbol reaches 80%. Earlier checkpoints show broad partial accuracy across many symbols rather than a clean solved/unsolved character partition.

## Context reconstruction does not explain the jumps

The following compares final READ accuracy with autoregressive context-reconstruction accuracy at the same final memory `M2`.

| Checkpoint | READ V1 acc. | READ V2 acc. | Context V1 rec. acc. | Context V2 rec. acc. |
|---:|---:|---:|---:|---:|
| 10,000 | 10.48% | 10.58% | 6.41% | 5.39% |
| 12,500 | 14.78% | 15.40% | 5.15% | 3.34% |
| 20,000 | 14.34% | 25.60% | 1.74% | 7.56% |
| 25,000 | 15.20% | 36.52% | 1.20% | 10.10% |
| 32,500 | 14.22% | 37.66% | 1.74% | 9.75% |
| 35,000 | 15.06% | 49.00% | 1.81% | 6.45% |
| 40,000 | 14.84% | 49.74% | 1.69% | 11.41% |
| 42,500 | 13.76% | 91.60% | 1.77% | 11.92% |
| 50,000 | 14.74% | 93.78% | 1.76% | 12.43% |
| 125,000 | 14.76% | 94.60% | 1.85% | 25.91% |

The decisive 40,000 -> 42,500 READ jump adds 41.86 points of `V2` accuracy, but context `V2` reconstruction adds only 0.51 points. The previous 32,500 -> 35,000 READ jump adds 11.34 points while context `V2` reconstruction *falls* by 3.30 points. Context reconstruction CE tells the same story: around the largest jump, `V2` CE changes only from 6.40 to 6.03.

The original metric history is consistent with this decoupling. Across its eight major transitions, final context-reconstruction loss sometimes improves, sometimes is nearly unchanged, and at 154,500 -> 157,000 worsens from 8.89 to 10.62 despite EM rising from 72.34% to 84.50%. There is no monotone reconstruction threshold associated with entering a new EM plateau.

At the original step-609,500 checkpoint, the difference remains large:

- READ: `V1` 97.18%, `V2` 99.90%.
- Context reconstruction at `M2`: `V1` 37.29%, `V2` 49.93%.
- Context key reconstruction at `M2`: `K1` 0.00%, `K2` 0.15%.
- Syntax reconstruction at `M2`: 52.00%.

The learned energy WRITE therefore produces memory highly useful for queried retrieval without making the context generally reconstructible. This is expected to be possible for an energy objective, but the magnitude of the separation is notable.

## Other changes at transitions

### Most retrieval ability appears after the first WRITE step

At every probed repeat checkpoint, `M0` READ is essentially at zero EM. Most useful behavior appears after the first energy-gradient update (`M1`); `M2` is usually a smaller refinement.

- Repeat step 42,500: EM is 0.00% at `M0`, 10.56% at `M1`, and 11.50% at `M2`. `V2` rises from 84.18% at `M1` to 91.60% at `M2`.
- Repeat step 125,000: EM is 0.00%, 12.46%, and 12.38% at `M0`, `M1`, and `M2`; the second step slightly hurts EM even though `V2` improves from 93.84% to 94.60%.
- Original step 609,500: EM is 0.00%, 96.22%, and 97.10%; `V1` is 96.44% after one step and 97.18% after two.

The staircase is thus mainly a change in what the first WRITE update computes, not the sudden appearance of usefulness only after the second update.

### Update scale grows over training but is not a consistent transition trigger

Memory and gradient norms trend upward strongly over the run, but major transitions do not require an upward jump in update scale.

- In the repeat, final memory norm grows from 6.63 at step 10,000 to 70.07 at step 125,000; the initial energy-gradient norm grows from 16.13 to 173.08.
- Yet at the largest repeat transition, 40,000 -> 42,500, final memory norm **falls** from 29.47 to 23.40 and the initial gradient norm **falls** from 72.45 to 57.36.
- In the original, some transitions coincide with norm growth, but the 104,000 -> 106,500 and 134,000 -> 136,500 transitions occur with almost flat or slightly lower memory norms.

Large norms are a property of the trained solution and continue growing long after high EM is reached; they are not by themselves an explanation of the discrete jumps.

### Energy scale and auxiliary diagnostics do not show a universal threshold

Absolute learned energy is not comparable in the same way as a fixed loss because the energy head itself changes and its scale drifts. Across original transitions, inner energy sometimes rises sharply and sometimes falls. The repeat's large 40,000 -> 42,500 transition occurs with almost unchanged pre-final-step inner energy (56.71 -> 56.11) and a higher after-WRITE energy (26.43 -> 31.01).

Other logged diagnostics are similarly non-diagnostic:

- Intermediate READ loss drops at transitions, as expected, and closely tracks final target loss. It confirms that `M1` improves together with `M2` rather than identifying a separate cause.
- Memory attention during READ stays in a narrow range around 0.69-0.74 through the relevant repeat and original transitions.
- Step-alignment cosine remains small (roughly 0.005-0.07) and has no consistent jump direction.

## Interpretation

The evidence supports the following stage-level account:

1. Training first builds a weak, position-biased ability to retrieve `V2`.
2. A sharp transition globalizes `V2` retrieval across all eight context positions and all 62 symbols, producing the stable approximately 12% EM plateau. EM remains low because `V1` is still near 15%.
3. In the successful original seed, later transitions progressively solve `V1`. Since `V2` is already almost always right and neither-correct examples are nearly gone, every gain in `V1` appears almost one-for-one as an EM gain.
4. The final residual error remains overwhelmingly at `V1`.

What is not established by the available checkpoints is *why* `V1` improves in approximately 11-12 point increments in the original run. The equal-step comparison is invalid across the seed mismatch, and the original intermediate weights were not retained. The repeat's early position-conditioned behavior suggests that transient internal specializations can precede a global solution, but its 12% plateau rules out the simplest claim that each visible plateau corresponds directly to one fixed KV slot or one fixed subset of alphabet symbols.

## Reproducibility artifacts

- Diagnostic implementation: `analyze_kv_em_staircase.py`
- Full selected repeat checkpoint diagnostics: `reports/em_staircase/checkpoint_diagnostics.json`
- Repeat diagnostics with pair-position and target-symbol conditioning: `reports/em_staircase/checkpoint_diagnostics_by_token.json`
- Original step-609,500 checkpoint diagnostics: `reports/em_staircase/original_checkpoint_diagnostics.json`

All checkpoint diagnostics use the complete fixed validation set of 5,000 examples, batch size 64, float32 evaluation, the recorded tokenizer, `K=2`, and inner learning rate 0.4.
