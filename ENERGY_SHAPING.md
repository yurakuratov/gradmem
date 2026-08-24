# Energy shaping in GradMemGPT

This note describes the motivation, equations, and exact implementation of
energy-landscape shaping in this repository. It deliberately separates what is
implemented from broader ideas that motivated the design.

The central idea is to train a conditional energy function so that a memory
written for a context has lower energy with that context than incompatible or
perturbed memories:

$$
E(C_i, M_i) < E(C_i, M_j), \qquad i \ne j.
$$

Here, lower energy is better, and $M_i$ is produced by the model's normal inner
write dynamics for context $C_i$. The shaping labels are therefore
self-supervised: no target memory representation is required.

## Base energy-memory system

For the prefix-memory backend used in the experiments, every example starts
from the same learned initial memory $M_0 \in \mathbb{R}^{n_m \times d}$. A
per-example copy is inserted before the context tokens:

$$
[M; C].
$$

The transformer processes this sequence. At each non-padding context position
$t$, a LayerNorm and a two-layer SiLU MLP produce a scalar token energy. The
context-memory energy is their masked mean:

$$
E_\theta(C,M)
=
\frac{1}{|C|}
\sum_{t \in C}
\operatorname{MLP}_{\theta,E}
\left(\operatorname{LN}(h_t([M;C]))\right).
$$

Thus, the energy is conditional on both the context and memory. It is not just
an MLP applied directly to $M$: memory affects the contextual hidden states
through transformer attention. The implementation is in
[`_compute_write_energy`](grad_memgpt.py#L1039) and the energy-head definition
is in [`GradMemGPT.__init__`](grad_memgpt.py#L821).

Writing performs $K$ gradient-descent steps on the per-example memory:

$$
M_{k+1} = M_k - \eta \nabla_{M_k} E_\theta(C,M_k).
$$

The runs discussed here use $K=2$, SGD, and inner learning rate
$\eta=0.4$. With `grad_mode=second` and `last_K_second_order=2`, both steps are
differentiable: the downstream read loss can train the parameters through the
inner optimizer, including the required second-order terms. The inner loop is
implemented in [`GradMemGPT.forward`](grad_memgpt.py#L1574).

After writing, the final memory is placed before the query, and the normal
language-model head predicts the target. The primary outer objective is target
cross-entropy.

## Why the downstream task loss is not enough

The downstream target loss only constrains energy where the finite write
trajectory happens to travel. Many very different energy surfaces can produce
similar two-step endpoints on the training distribution. They can behave very
differently when:

- more inner steps are taken at inference time;
- contexts are longer than those seen during training;
- the memory starts or drifts away from the usual trajectory;
- a memory written for another context is presented.

Energy shaping adds direct constraints on compatibility and nearby geometry.
The main signal holds the context fixed while varying memory, making a purely
context-dependent energy an ineffective shortcut. This follows the negative
sampling logic of [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748),
although this implementation uses pairwise softplus ranking rather than an
InfoNCE denominator over the complete batch.

## Implemented final-memory shaping

Let

$$
M_i^+ = \operatorname{stopgrad}(\operatorname{Write}_\theta(C_i, M_0))
$$

be the final written memory. The implementation constructs three negative
memory candidates. All positive and negative candidate memories are detached
before the auxiliary energy evaluations.

### 1. Deranged in-batch memory

For batch size $B>1$, the code samples one cyclic shift
$s \in \{1,\ldots,B-1\}$ and pairs each context with

$$
M_{i,\mathrm{der}}^- = M_{(i+s)\bmod B}^+.
$$

This is a fixed-point-free derangement, so no context keeps its own memory. It
is one negative per example, not a comparison against every other memory in the
batch.

### 2. Interpolated hard negative

The second negative lies between the positive and deranged memories:

$$
M_{i,\mathrm{mix}}^-
=
\alpha M_i^+ + (1-\alpha)M_{i,\mathrm{der}}^-.
$$

The reported shaped pipeline uses $\alpha=0.75$, making this candidate close to
the positive while retaining mismatched-memory content. Such near-distribution
negatives should be harder than unconstrained Gaussian states.

`energy_mix_alpha` controls this interpolation coordinate. It is **not** a
separate loss coefficient: the three negative losses are averaged equally.

### 3. Write-radius-matched random memory

For each example, the code samples a Gaussian direction $u_i$, normalizes it,
and places a random candidate on a sphere around the initial memory:

$$
r_i = \lVert M_i^+ - M_0 \rVert_2,
\qquad
M_{i,\mathrm{rand}}^-
=
M_0 + r_i\frac{u_i}{\lVert u_i\rVert_2}.
$$

The random state therefore has the same displacement from $M_0$ as the actual
write, rather than the much larger norm of an unscaled $\mathcal N(0,I)$
sample. This is radius matching, not semantic matching.

Negative construction is implemented in
[`_build_energy_negative_memories`](grad_memgpt.py#L1072). Unit tests check the
derangement, interpolation identity, requested random radius, and detachment in
[`test_energy_negative_memory_geometry`](tests/test_gradmemgpt.py#L952).

## Pairwise ranking objective

For each available negative type $q$, the code computes

$$
\mathcal L_q
=
\frac{1}{B}\sum_i
\operatorname{softplus}\left(
\frac{E_\theta(C_i,M_i^+) - E_\theta(C_i,M_{i,q}^-) + m}{\tau}
\right).
$$

The configured margin and temperature are $m=0.1$ and $\tau=1.0$. The final
ranking loss is the unweighted mean over the available negative types:

$$
\mathcal L_{\mathrm{rank}}
=
\frac{1}{|Q|}\sum_{q\in Q}\mathcal L_q,
\qquad
Q=\{\mathrm{der},\mathrm{mix},\mathrm{rand}\}.
$$

Unlike a hinge, softplus continues to provide a smooth, diminishing gradient
after the desired ordering is reached. The code is in
[`_compute_energy_landscape_losses`](grad_memgpt.py#L1111).

For batch size one, deranged and interpolated negatives cannot be formed; only
the radius-matched random negative is used.

## Energy-scale anchoring

Ranking constrains energy differences but does not determine a global offset.
Because memory updates use energy gradients, uncontrolled scale can also alter
the effective inner-step size. When ranking is active, the implementation
weakly penalizes the squared energies of the detached positive and every
negative already evaluated for ranking:

$$
\mathcal L_{\mathrm{anchor}}
=
\frac{1}{1+|Q|}
\left[
\mathbb E_i E_\theta(C_i,M_i^+)^2
+
\sum_{q\in Q}\mathbb E_i E_\theta(C_i,M_{i,q}^-)^2
\right].
$$

This does not force the positive to zero and the negatives to a chosen positive
target; it pulls all evaluated energies weakly toward zero while ranking
determines their relative order. Weak energy regularization has also been used
to stabilize energy-based models; see
[Du & Mordatch, *Implicit Generation and Modeling with Energy Based Models*](https://proceedings.neurips.cc/paper_files/paper/2019/file/378a063b8fdb1db941e34f4bde584c7d-Paper.pdf).

When anchoring is enabled without ranking, no negative memories are constructed.
The anchor is computed only on the detached positive memory:

$$
\mathcal L_{\mathrm{anchor-only}}
=
\mathbb E_i E_\theta(C_i,M_i^+)^2.
$$

This preserves the stop-gradient shaping semantics while avoiding three unused
negative-memory evaluations.

## Implemented trajectory loss

The repository also implements an optional hinge loss on the exact write
iterates:

$$
\mathcal L_{\mathrm{traj}}
=
\frac{1}{K}
\sum_{k=0}^{K-1}
\max\left(0, E(C,M_{k+1})-E(C,M_k)+m_{\mathrm{traj}}\right).
$$

The final pair uses the separately evaluated post-update energy $E(C,M_K)$.
This encourages monotonically decreasing energy along the observed trajectory.
It does not shape lateral directions or a tube around that trajectory. The
implementation is in [`GradMemGPT.forward`](grad_memgpt.py#L1806).

**The reported shaped checkpoints set `energy_traj_weight=0.0`, so this term was
not active.** It is implemented and tested, but is not part of the empirical
shaped-versus-baseline comparison.

The broader motivation for shaping finite-step inference is related to
end-to-end training of structured prediction energy networks:
[Belanger, Yang, and McCallum, *End-to-End Learning for Structured Prediction Energy Networks*](https://proceedings.mlr.press/v70/belanger17a.html).

## Complete outer training loss

The code returns

$$
\mathcal L_{\mathrm{outer}}
=
\mathcal L_{\mathrm{target}}
+ \mathbf 1_{\mathrm{inner}}\lambda_{\mathrm{inner}}\mathcal L_{\mathrm{write}}
+ \lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+ \lambda_{\mathrm{traj}}\mathcal L_{\mathrm{traj}}
+ \lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}.
$$

For the reported runs, `add_inner_loss_to_outer=false`, so the direct
$\mathcal L_{\mathrm{write}}$ term is absent. The target loss still
backpropagates through the differentiable two-step write procedure.

For the shaped pipeline:

$$
\mathcal L_{\mathrm{outer}}
=
\mathcal L_{\mathrm{target}}
+ 0.01\mathcal L_{\mathrm{rank}}
+ 0.001\mathcal L_{\mathrm{anchor}}.
$$

For the baseline pipeline, both auxiliary coefficients are zero, leaving only
the target loss. Loss accounting and logging are implemented near
[`outer_loss`](grad_memgpt.py#L1864) and tested in
[`test_energy_landscape_loss_components_and_accounting`](tests/test_gradmemgpt.py#L991).

## Gradient paths: what detachment does and does not mean

The final written memories and all negative candidates are detached inside the
auxiliary landscape loss. Therefore, the auxiliary ranking and anchor losses do
not train the writer by moving $M_i^+$ toward an easily separable code, and do
not backpropagate through the inner write trajectory.

They do, however, train the network that assigns energies to the fixed
candidates. In the reported runs `freeze_backbone=false`, so this includes the
energy head, LayerNorm, shared transformer backbone, and relevant embeddings—not
only the small energy MLP. A dedicated test confirms that candidate-memory
gradients are absent while energy-model gradients remain finite:
[`test_energy_landscape_candidates_detach_memory_but_train_energy`](tests/test_gradmemgpt.py#L1120).

Separately, the target read loss does backpropagate through the two inner write
steps because second-order mode is enabled. This is how the energy descent
direction remains tied to useful downstream memory rather than compatibility
ranking alone.

## Exact recipe used by the evaluated checkpoints

Both pipelines contain two training stages. Consequently, the frozen-pair
evaluation compares complete pipelines; it cannot isolate the reconstruction
stage from the continuation stage.

| Setting | Baseline reconstruction | Shaped reconstruction | Baseline continuation | Shaped continuation |
| --- | ---: | ---: | ---: | ---: |
| Initialization | random model init | random model init | baseline reconstruction step 15,000 | shaped reconstruction step 15,500 |
| Write objective | energy + reconstruction | energy + reconstruction | energy only | energy only |
| Inner steps / inner LR | 2 / 0.4 | 2 / 0.4 | 2 / 0.4 | 2 / 0.4 |
| Second-order inner steps | 2 | 2 | 2 | 2 |
| Rank weight | 0 | 0.01 | 0 | 0.01 |
| Anchor weight | 0 | 0.001 | 0 | 0.001 |
| Trajectory weight | 0 | 0 | 0 | 0 |
| Rank margin / temperature | inactive | 0.1 / 1.0 | inactive | 0.1 / 1.0 |
| Interpolation $\alpha$ | inactive | 0.75 | inactive | 0.75 |
| Outer learning rate | $10^{-4}$ | $10^{-4}$ | $10^{-4}$ | $10^{-4}$ |
| Batch size | 64 | 64 | 64 | 64 |
| Training data | N8 | N8 | N8 | N8 |
| Evaluated checkpoint | stage initializer | stage initializer | step 18,500 | step 18,500 |

In the reconstruction stage, `write_objective=energy_with_reconstruction` means
that each memory update follows the gradient of

$$
\mathcal L_{\mathrm{write}}
=
\mathcal L_{\mathrm{reconstruction}} + E(C,M),
$$

with both inner weights equal to one. In the continuation stage,
`write_objective=energy`, so memory updates follow energy alone. The landscape
auxiliaries remain outer losses in both shaped stages.

The wrapper defining the calibrated shaping values is
[`scripts/run_energygradmem_shaping_on_kv_retrieval.sh`](scripts/run_energygradmem_shaping_on_kv_retrieval.sh),
and the common launcher is
[`scripts/run_energygradmem_on_kv_retrieval.sh`](scripts/run_energygradmem_on_kv_retrieval.sh).

## What is not implemented

The methodology that motivated this work also proposed several extensions.
They should not be attributed to the present checkpoints:

- no multi-radius shell-ordering loss;
- no forward/backward or lateral tube probes around the write trajectory;
- no same-context multi-start consistency;
- no denoising or score-matching objective;
- no local-convexity or contraction regularization;
- no persistent optimized hard negatives;
- no spectral normalization of the energy head;
- no memory-token shuffle, partial token replacement, or corrupted-context
  negatives;
- no margin that adapts with interpolation $\alpha$;
- no separate per-negative or interpolation-loss weights;
- no all-pairs in-batch InfoNCE objective.

Multi-radius perturbations and denoising score matching remain plausible ways
to constrain off-trajectory regions in high-dimensional memory space; see
[Song and Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://arxiv.org/abs/1910.07762).
However, Euclidean distance in memory space is not guaranteed to represent
semantic incorrectness, so such losses would need careful inner-radius and
negative-quality controls.

## Unimplemented but promising ideas from Energy-Based Transformers

The following ideas are motivated by
[Gladstone et al., *Energy-Based Transformers are Scalable Learners and Thinkers*](https://arxiv.org/abs/2507.02092)
and its [released EBT code](https://github.com/alexiglad/EBT). None is currently
implemented or evaluated in GradMemGPT:

- **Randomized training step count:** vary $K$ during training instead of always
  using $K=2$, so the downstream read loss supervises later trajectory states.
- **Randomized inner step size:** sample $\eta$ per example, or possibly per
  memory token, from a range around the nominal inner learning rate.
- **Longer trajectories with truncated second order:** train on longer writes
  while retaining higher-order differentiation through only the last one or two
  updates.
- **Persistent positive-trajectory replay:** store context and optimized-memory
  pairs, then resume their optimization in later batches. EBT uses this to train
  near and beyond previous endpoints; it is not a hard-negative replay buffer.
- **Perturbed initial memories:** sometimes initialize near $M_0$ rather than
  exactly at $M_0$, teaching the energy field to recover from off-trajectory
  states.
- **Langevin-style path noise:** inject scaled Gaussian noise before memory
  updates to explore a tube around the usual descent trajectory. The noise
  should be calibrated to the memory write radius rather than copied numerically
  from EBT's token-logit setting.
- **Multi-start inference with self-verification:** optimize several perturbed
  initial memories and select the lowest-energy result, provided energy is first
  shown to rank downstream read quality reliably.
- **Normalized or trust-region updates:** constrain gradient or update norms to
  prevent memory drift during long inference trajectories.

The most direct first experiment is randomized $K$ together with randomized
$\eta$. Langevin noise, replay, and multi-start selection are more relevant when
exploration and best-of-$N$ inference are desired.

## Interpretation and limitations

The contrastive signal says that the memory written for $C_i$ should be more
compatible with $C_i$ than sampled alternatives. It does not prove that the
energy is globally calibrated, convex, or density-like. Important limitations
are:

- positives are generated by the current writer rather than supplied by an
  independent oracle;
- in-batch negatives can be false negatives if two contexts admit equivalent
  memory encodings, although this is unlikely for the synthetic KV task;
- one cyclic derangement explores only a small part of the in-batch negative
  set per update;
- the random negative is norm-controlled but can still be semantically trivial;
- anchoring controls evaluated energy magnitudes but does not directly bound
  $\lVert\nabla_M E\rVert$ or the inner-update Jacobian;
- the auxiliary loss updates the shared backbone, so observed gains need not
  arise solely from a more interpretable energy head;
- a single shaped/unshaped checkpoint pair does not measure training-seed
  uncertainty.

## References

1. A. van den Oord, Y. Li, and O. Vinyals,
   [*Representation Learning with Contrastive Predictive Coding*](https://arxiv.org/abs/1807.03748),
   2018.
2. Y. Song and S. Ermon,
   [*Generative Modeling by Estimating Gradients of the Data Distribution*](https://arxiv.org/abs/1910.07762),
   NeurIPS 2019.
3. D. Belanger, B. Yang, and A. McCallum,
   [*End-to-End Learning for Structured Prediction Energy Networks*](https://proceedings.mlr.press/v70/belanger17a.html),
   ICML 2017.
4. Y. Du and I. Mordatch,
   [*Implicit Generation and Modeling with Energy Based Models*](https://proceedings.neurips.cc/paper_files/paper/2019/file/378a063b8fdb1db941e34f4bde584c7d-Paper.pdf),
   NeurIPS 2019.
5. A. Gladstone et al.,
   [*Energy-Based Transformers are Scalable Learners and Thinkers*](https://arxiv.org/abs/2507.02092),
   2025. [Code](https://github.com/alexiglad/EBT).
