# Adaptive Memory Updates — Mamba-style gated recurrence

> Implementation: [`grad_memgpt_adaptive.py`](./grad_memgpt_adaptive.py)
> Config: [`configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml`](./configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml)

## The problem

The baseline inner-loop memory update is a running sum:

$$m_t = m_{t-1} + x_t, \qquad x_t \triangleq -\alpha\,\nabla L(m_{t-1})$$

This is the most degenerate RNN possible. There is **no gate, no decay, no notion of what is already stored** — every gradient step is added with full weight. With segmented WRITE context and **cross-segment memory carry** (segment 1 starts from $m_0$, later segments continue from $m_t$), the inner loop becomes a recurrence unrolled over `n_segments × K` steps. Information written in segment 1 must survive segments 2, 3, … without being overwritten. Under the pure-sum rule a new gradient can stomp on the directions earlier facts relied on — catastrophic forgetting here is **structural**, not a tuning problem.

## The update

Generalize the running sum to a gated recurrence with the gradient step as the input signal:

$$m_t = r_t \odot m_{t-1} \;+\; w_t \odot x_t, \qquad x_t = -\alpha\,\nabla L(m_{t-1})$$

`memory_update_rule="mamba"` uses a Mamba-style **input- and state-dependent per-dimension retention**:

$$\Delta_t = \operatorname{softplus}\!\big(W_r\,\phi(m_{t-1}, g_t)\big), \qquad r_t = \exp(-\Delta_t) \in (0,1]$$
$$w_t = \sigma\!\big(W_w\,\phi(m_{t-1}, g_t)\big) \in (0,1)$$

where $g_t = \nabla L(m_{t-1})$ and $\phi$ is a per-token feature. `gate_retention="sigmoid"` switches retention to $r_t=\sigma(\cdot)$ (GRU-style, Family 1); the exponential form is the default.

### Why this fights forgetting

- **Per-dimension, data-dependent retention.** Where $\Delta_t\!\to\!0$, $r_t\!\to\!1$ → perfect retention, the new gradient is ignored there. The model *learns* where to open up to a write and where to protect stored info.
- **Exponential gating stays conditioned over long unrollings.** Retention multiplies: $\prod_t r_t = \exp(-\sum_t \Delta_t)$. Over the cross-segment recurrence this avoids the saturation/quenching that sigmoid gates suffer at long range — the key reason Mamba/GLA beat LSTM-style gates.
- **Starts as ≈SGD.** Bias init gives $\Delta\!\approx\!0.007 \Rightarrow r\!\approx\!0.993$, $w\!\approx\!0.95$. The operator initializes as the baseline running sum and *learns to deviate*.

## Configurability

| knob | values | meaning |
|---|---|---|
| `memory_update_rule` | `sgd` \| `convex` \| `mamba` | `sgd` = exact baseline (no gate params); `convex` = single gate $a$, $r\!=\!1\!-\!a, w\!=\!a$; `mamba` = above |
| `gate_features` $\phi$ | `grad` \| `state` \| `grad_state` | input to the gate heads; default `grad_state` (true selectivity — both input and state) |
| `gate_granularity` | `per_dim` \| `per_token` | gate width `[B,M,d]` vs `[B,M,1]` (broadcast over $d$) |
| `gate_retention` | `exp` \| `sigmoid` | Mamba exponential decay vs GRU gate |
| `seg_bptt` | int \| null | truncated-BPTT window over the segment recurrence (bounds activation memory over long unrollings) |

## How it fits the codebase

- **No new optimization machinery.** The gate heads are tiny `nn.Linear`s; they are meta-learned by the **existing second-order** `autograd.grad` path (`create_graph=True` on the last `K` steps), exactly like `mem_proj`. All gates are differentiable, so MAML gradients flow to the heads through the unrolled recurrence.
- **`use_gated_delta_memory` is the conceptual neighbor.** That implements the *associative-matrix* (DeltaNet) cousin $S = S(\alpha(I-\beta kk^\top)) + \beta vk^\top$ at the **segment level**. This applies the same gated-SSM philosophy one level down, at the **per-step memory-token** recurrence.

## Footgun

`gate_retention="sigmoid"` needs a **positive** `mamba_retain_bias_init`. The `-5` default is calibrated for the `exp` path (opposite signs): $\operatorname{softplus}(-5)\!\approx\!0.007$ gives $r\!\approx\!0.993$, but $\sigma(-5)\!\approx\!0.007$ gives near-total forgetting. Set `mamba_retain_bias_init: +5.0` to preserve the SGD-start under sigmoid retention.

## Run

```bash
# dry-run
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml --dry-run

# train
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml

# swap to the convex baseline (no new config file)
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml memory_update_rule=convex
```

## Diagnostics logged

| metric | meaning |
|---|---|
| `gate_retain_mean` | mean $r_t$ (→1 = full retention = SGD-like) |
| `gate_write_mean` | mean $w_t$ (→1 = full write) |
| `gate_delta_mean` | mean Mamba $\Delta_t$ (→0 = full retention) |
| `seg_nonempty_count_mean` | mean real-token segments per sample |
| `seg_nonempty_size_mean` | mean tokens per real segment |
