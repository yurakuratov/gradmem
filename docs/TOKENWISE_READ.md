# Token-wise memory reads via linear attention — the linear-state per-level read

> Status: **specified, not implemented** (2026-10-15). Target implementation:
> `grad_memgpt.py` (a config-gated READ mode next to the Hopfield retrieval).
> Evidence context: [PROGRESS.md](../PROGRESS.md) facts 5–7, hypothesis H3;
> journal: [STATUS.md](../STATUS.md) 2026-10-15.

## The problem

The Hopfield READ compresses the whole query into one vector, scores each
stored segment once, and hands the reader a single pre-mixed memory
(`grad_memgpt.py`, Hopfield RETRIEVE + READ phase). Formally, with stored
per-segment patterns $v_i$ (flattened $M{\cdot}d$) and segment keys $k_i$:

$$
o \;=\; \sum_i a_i\, v_i, \qquad a \;=\; \operatorname{softmax}\!\big(\beta\, q{\cdot}K\big),
\qquad q \;=\; \operatorname{compress}(\text{query}) \in \mathbb{R}^{Md},
$$

and the reader consumes $o$ reshaped to $[M,d]$ as a prefix. Three coupled
failures on any task that must **integrate content from different segments**
(ordered pair concatenation `?!K1:?!K2:` → `V1!V2!|`):

1. **Routing** — one query vector must be simultaneously close to $k_1$ and
   $k_2$; a two-key query is a compromise direction.
2. **Payload** — softmax weights are convex ($a_1{+}a_2{=}1$), so retrieval is
   a superposition $a_1v_1{+}a_2v_2$ inside the fixed $M{\cdot}d$ space. The
   reader must *demix* it through a frozen GPT-2 — the capacity regime the
   memory tokens were meant to escape, rebuilt at read time.
3. **No iteration** — one pre-read mixture per forward; pointer chasing
   (retrieve $V_1{=}K_2$, then retrieve $V_2$) is structurally out of reach.

No setting of the one-shot mixture escapes this: `raw`/`softmax`/`mean`
differ only in weights that LayerNorm mostly rescales away, and value-code
nonlinearities sit *before* mixing while the post-mixing decode is linear
(2026-10-05 eliminability note). Non-linear/sparse codes accordingly did not
help (2026-10-15).

## The mechanism

Framing: **an extra set of attention heads on every reader block, whose
keys/values are the stored memory tokens and whose kernel is linear
attention.** The bank never enters the read sequence — it is folded once into
a per-head associative state; each token at each layer reads it with its own
query, and the result is added to the block output as a gated bias.

### Notation

- Segments $s = 1..S$, each written independently by the unchanged inner loop
  into memory tokens $m_{s,j} \in \mathbb{R}^{d}$, $j = 1..M$ (the current
  Hopfield STORE values, pre-flattening). Bank size $N = S \cdot M$.
- Reader: causal transformer, layers $\ell = 1..L$, hidden states $h_t^{\ell}
  \in \mathbb{R}^{d}$, heads $H_r$, head dim $d_h = d / H_r$. Read sequence is
  `[read_ctrl?, query]` — **no memory tokens**.

### 1. State construction (once per READ, query-independent)

Per head $r$, with shared-across-layers projections
$\kappa_r, \nu_r : \mathbb{R}^{d} \to \mathbb{R}^{d_h}$:

$$
k_{s,j}^{(r)} = \frac{\kappa_r(m_{s,j})}{\lVert \kappa_r(m_{s,j}) \rVert},
\qquad
v_{s,j}^{(r)} = \nu_r(m_{s,j}),
\qquad
S_r \;=\; \sum_{s \in \text{stored}} \sum_{j=1}^{M} k_{s,j}^{(r)} \, v_{s,j}^{(r)\top}
\;\in\; \mathbb{R}^{d_h \times d_h}.
$$

Cost $O(N d^2)$ once; state memory $O(d^2 / H_r)$ per sample — independent of
$S$. Unwritten samples get $S_r = 0$ → zero bias → **the no-store fallback
path disappears** (today's `hopfield_stored` where-case).

### 2. Per-level, per-token query

At each layer $\ell$ and read position $t$, from the block input (after the
block's LayerNorm, matching $\text{ln}_1$ placement):

$$
q_{t,r}^{\ell} \;=\; \beta_r^{\ell} \cdot
\frac{Q_r^{\ell}\!\big(\operatorname{LN}(h_t^{\ell-1})\big)}
     {\lVert Q_r^{\ell}\!\big(\operatorname{LN}(h_t^{\ell-1})\big) \rVert}
\;\in\; \mathbb{R}^{d_h},
$$

with per-layer projections $Q_r^{\ell}$ (layer hidden states live in
different distributions) but shared $\kappa_r, \nu_r$ (the state is built
once). The L2 normalization gives cosine-semantics; the learnable
$\beta_r^{\ell}$ is the retrieval sharpness (the Hopfield $\beta$, absorbed
into the query scale).

### 3. Read + gated additive injection

$$
\mu_{t,r}^{\ell} = q_{t,r}^{\ell\top} S_r, \qquad
\mu_t^{\ell} = W_o \, \mathrm{concat}_r(\mu_{t,r}^{\ell}), \qquad
h_t^{\ell} = h_t^{\ell-1} + \mathrm{Attn}_t^{\ell} + \gamma^{\ell} \odot \mu_t^{\ell}.
$$

$\gamma^{\ell} \in \mathbb{R}^{d}$ is a per-layer (per-head granularity
optional) learnable gate, initialized small-but-positive. Injection is
**parallel** to the attention sublayer (Flamingo/Titan style): the memory
branch sees the block input, never blocks the pretrained pathway, and the
frozen backbone is untouched (forward hooks on each block suffice).

### Equivalences

- Unrolled, the read is linear attention over the bank:
  $\mu_{t,r}^{\ell} = \sum_{s,j} \big(q_{t,r}^{\ell} \cdot k_{s,j}^{(r)}\big)
  v_{s,j}^{(r)}$ — one-step Hopfield retrieval (modern-Hopfield
  $\operatorname{softmax}(\beta K q)$ at first order in $\beta$, partition
  function dropped), i.e. exactly the current retrieval formula but with
  **per-token, per-head, per-layer queries, unnormalized weights, additive
  output** instead of one compressed query, convex weights, prefix output.
- With the Gated-Delta mode the same read already exists at one granularity:
  `retrieved = S @ q` (`grad_memgpt.py`, Gated Delta RETRIEVE) — one
  compressed query, one read, reshaped into the prefix. This proposal
  generalizes that line to token-wise, level-wise, additive.

The [Variant](#variant-interface-token-reads--the-same-state-read-only-by-m-special-tokens)
below restricts the same read to $M$ learnable interface tokens.

## Why this fixes the three failures

1. **Routing** — the query distribution over the bank is computed per position
   $t$ and per head: the position that must emit $V_1$ forms its query from a
   hidden state that has (causally, through earlier layers) seen $K_1$ in the
   query text; the $V_2$ positions likewise. $H_r \cdot T \cdot L$ routing
   decisions per read instead of $1$; no compromise vector.
2. **Payload** — weights $q{\cdot}k$ are unnormalized: a query can have high
   affinity to segment 1's keys *and* segment 2's keys simultaneously, so both
   values arrive at full strength. Values are never summed into a shared
   $M{\cdot}d$ slot; whatever superposition occurs is spread across
   $d_h$-dimensional head outputs and is **immediately nonlinearly processed**
   (the block's MLP and all later layers) — the nonlinearity moves to the
   correct side of the mixing, which is precisely where the failed value-code
   nonlinearity was missing.
3. **Iteration** — $\mu^{\ell}$ enters $h^{\ell}$, which forms $q^{\ell+1}$:
   layer $\ell{+}1$ re-reads the state with queries informed by layer $\ell$'s
   retrieval. Multi-hop (retrieve $K_2$, then $V_2$) becomes a depth-bounded
   induction — free iterative retrieval, the property the one-shot design
   cannot have and pointer chasing needs.

### Capacity / interference (the honest bound)

The state superposes all $N = S M$ items. For a query aligned with some
$k_{s^*,j^*}$:

$$
\mu = v_{s^*,j^*} \;+\; \sum_{(s,j) \neq (s^*,j^*)} \big(q \cdot k_{s,j}\big)\, v_{s,j},
\qquad
\text{cross-talk} \sim O\!\big(\sqrt{N / d_h}\big)
$$

for near-orthogonal keys — the standard linear-attention recall ceiling
(MQAR-style results). At the current scale ($S \le 8$, $M{=}2$, $d_h \ge 64$)
this is comfortable; the interference **curve vs $N$** is a first-class
measurement, and the pinned mitigation when it binds is coarse top-k segment
routing: select segments with the existing compressed-query scores, build
$S_r$ from the selected subset only. A second lever: fewer, wider memory
heads ($d_h \uparrow$) since state cost is $O(d^2/H_r)$ either way.

## Variant: interface-token reads — the same state, read only by M special tokens

Restrict the additive bias above to **$M$ learnable interface tokens** (a
dedicated $M_0$, cf. the existing `mem_query` parameter): those tokens read
the state at each layer — same $S_r$, same normalized queries, same gated
additive injection, masked to the $M$ interface positions — and every other
position reaches memory only through ordinary softmax attention to the
interface tokens. This is the **resampler pattern** (Flamingo's gated
cross-attention / Q-Former) applied to the Hopfield state, and it sits on the
same read-site ladder: one compressed query, one shot (current) →
$M{\cdot}L$ interface reads (this variant) → $T{\cdot}L{\cdot}H_r$ reads
(the default design above).

### The placement / causality trap (decides the layout)

- **Appended right, causal**: the interface tokens see the query, but answer
  positions (earlier) can never attend to them — memory unreachable exactly
  where the answer is emitted. Broken as stated.
- **Prefix left, causal** (the current READ layout): answer positions see the
  interface tokens, but the interface tokens never see the query — their
  per-layer reads are **query-blind**. Retrieval degenerates to query-agnostic
  compression: a *silent* regression to today's architecture with extra
  steps. This is the failure mode to guard against (query-sensitivity
  diagnostic below).

Resolutions: **(1) two-pass — recommended**; (2) single pass with a relaxed
attention mask (query positions may attend leftward-only into right-appended
interface tokens; interface tokens must be blinded to the target span or the
answer leaks through them) — feasible on the eager-attention path but needs a
4D mask, footgun-heavy; (3) read the answer out at interface positions —
changes the output interface, rejected.

**Two-pass layout (the spec):**

- **Pass 1 — conditioning**: exactly `_compress_query`'s layout
  $[\text{query (target-span blinded)},\, M_0]$ — the interface tokens attend
  to the query *and* read the state at each layer, arriving query-conditioned.
  The `labels ≥ 0` blinding machinery already exists in `_compress_query`.
- **Pass 2 — answering**: exactly the current READ layout
  $[\text{read}_{st}?,\, \text{interface},\, \text{read}_{end}?,\,
  \text{query}]$, with the pass-1 interface states as the interface positions'
  embeddings, still reading the state per layer. I.e. the current read phase
  with two swaps: learnable $M_0$ + per-level state reads replace the one-shot
  retrieved mixture. `mem_proj`, ctrl tokens, and reader training all carry
  over — the "closer architecturally" rationale in its strongest form.

### Gains / risks vs the read-everywhere design

**Gains.** Read sites $M{\cdot}L$ instead of $T{\cdot}L{\cdot}H_r$ — compute
back to ≈ the current read cost (state construction unchanged), and far less
outer-gradient pressure on the stored values (the stability risk in
*Training & gradient flow*). The interface tokens **restore the prefix
workspace** the no-prefix design gave up: multi-hop can happen inside them
across layers (retrieve $K_2$'s segment at layer 3, re-query with it at
layer 5). And the reader side is not a convex mixture: answer positions
softmax over $M$ *separate* interface tokens, so per-position selection
survives — at the reader level.

**Risks.** The $M$ tokens are again a fixed $M{\cdot}d$ channel between state
and answer. For concat, the optimistic picture is specialization (token A ↔
segment 1, token B ↔ segment 2, answer positions pick accordingly; diversified
$M_0$ inits plus reader-side competition make it plausible), but with $M{=}2$
over many segments each token holds mixtures again, and nothing guarantees
specialization. Honest prediction: between the one-shot read and full
per-token reads, with **under-specialized interface tokens** as the named
failure mode.

### Why run it

It turns "does cross-segment integration need per-token reads, or only
per-layer reads through a small interface?" into one knob
(`read_state_read_sites: interface | all`). Either outcome is a claim: the
cheaper architecture suffices (read-everywhere is over-engineering), or the
read-site count is the load-bearing factor.

## Design decisions (pinned 2026-10-15, with rationale)

| decision | choice | rationale |
|---|---|---|
| kernel | unnormalized linear (no partition function) | normalization reintroduces convexity per token — the exact failure being removed; $\beta$ absorbs sharpness |
| query norm | L2-normalize $q$, learnable $\beta_r^\ell$ | residual norms grow with depth; cosine semantics keeps $\lVert\mu\rVert$ controllable (same lesson as `stabilize_energy_head`) |
| gate | per-layer $\gamma^\ell$, init small-positive | zero-init may never escape; full-init perturbs the frozen reader; LayerScale-style middle |
| projections | shared $\kappa_r,\nu_r$; per-layer $Q_r^\ell$ | state built once (cost), layer distributions differ (correctness) |
| query source | $\operatorname{LN}(h^{\ell-1})$ (block input) | matches $\text{ln}_1$ placement; parallel injection needs the pre-block signal |
| injection | parallel additive at attention sub-output | leaves the pretrained attention path bitwise untouched |
| values | stored post-inner-loop memory tokens | preserves "decoder-informed writes" (H1); no re-compression |
| write side | **unchanged** — independent per-segment inner loops | the anti-forgetting property depends on it |
| read prefix | none (no memory tokens at READ) | fixed read cost; prefix workspace given up — measured against the bank-prefix control |
| BPTT | stored values detach per `hopfield_bptt_segments` as today | read sites multiply the graph through the values; the existing knob bounds it |

## Training & gradient flow

New modules — $\kappa_r, \nu_r, W_o, \{Q_r^\ell\}, \{\beta_r^\ell\},
\{\gamma^\ell\}$ — are **outer-loop only**, like the read LoRA: they never
enter the inner loop, so the second-order path (and `use_energy_inner_loss`,
learned updates, etc.) is untouched. The outer loss now reaches the stored
values through $L \times T \times H_r$ read sites instead of one retrieval —
a much richer signal shaping what the writer stores — while old segments can
still be detached by `hopfield_bptt_segments`. Watch for the inner
(reconstruction) and outer (read-mediated) objectives fighting over memory
content; the existing `inner_loss_seg_*` stats make this visible.

## Cost accounting

Per read token per layer: query projections $d^2$, read $H_r d_h^2 = d^2/H_r$,
output projection $d^2$ → $\approx (2 + 1/H_r)\,d^2$, vs $\approx 12 d^2$ for
a full transformer block — roughly a 20 % FLOP overhead at every instrumented
layer; restrict to a layer subset (`read_state_layers`) if that matters. State
construction is $O(N d^2)$ once; state storage $O(d^2/H_r)$ floats per sample
(49 K at $d{=}768$, $H_r{=}12$ — trivial). No term grows with $S$ after the
state is built.

## Proposed config surface (repo conventions)

A `read_linear:` section (mutually exclusive with the memory-prefix READ;
requires the Hopfield STORE):

| knob | default | meaning |
|---|---|---|
| `use_linear_state_read` | `false` | replace Hopfield RETRIEVE + memory prefix with per-level additive reads |
| `read_state_heads` | 4 | $H_r$ (wider heads than the backbone's are fine and reduce cross-talk) |
| `read_state_layers` | `"all"` | which blocks instrument; list of indices to restrict |
| `read_state_read_sites` | `"all"` | `all` = bias at every position; `interface` = only on the $M$ learnable interface tokens (Variant above; count = `n_mem_tokens`, two-pass layout) |
| `read_state_beta_init` | 1.0 | initial retrieval sharpness per layer |
| `read_state_gate_init` | 0.1 | initial $\gamma^\ell$ |
| `read_state_segment_emb` | `false` | optional learnable per-segment embedding added before $\kappa/\nu$ |
| `read_state_exact_softmax` | `false` | ablation: exact softmax over the bank per token ($O(N)$) instead of the summed state |

## Diagnostics to log

- `read_state_gate_abs_mean` per layer (collapse watch: gate → 0 means the
  read is being switched off);
- `read_state_mu_norm_ratio` = $\lVert\mu\rVert / \lVert\mathrm{Attn}\rVert$
  per layer (calibration watch);
- `read_state_beta` per layer (effective sharpness);
- `read_state_cross_talk` — mean off-target overlap
  $\overline{|q \cdot k|}$ over non-retrieved bank items (interference
  floor; also the x-axis of the interference curve);
- per-position retrieval probe — reuse `_probe_boundary_memory` plumbing:
  at eval, log which stored segment(s) each query position's read most
  aligns with ($\arg\max_s \sum_j q \cdot k_{s,j}$).

Interface mode (`read_state_read_sites: interface`) additionally:

- `read_state_reader_attn_interface` — eval-only reader attention mass on
  the interface tokens, split by answer vs non-answer positions
  (specialization/routing watch; flat mass = under-specialization);
- **query-sensitivity check** — perturb the query text, measure the change
  in the interface tokens' $\mu$; a left-prefix single-pass implementation
  (query-blind reads) fails this check by construction.

## Experimental plan (decision criteria pre-agreed)

Ladder that isolates *where* the loss comes from:

| arm | routing | payload | cost | role |
|---|---|---|---|---|
| (a) current one-shot retrieval | 1 query | convex mix | $O(1)$ | baseline (known to fail concat) |
| (b) bank-prefix control | backbone attn | separate tokens | $O(N)$ context | recall **upper bound**; decides H2 |
| (c) exact-softmax memory heads | per token | separate tokens, normalized | $O(N)$ compute | isolates prefix-vs-head |
| (d) **linear-state read (this doc)** | per token | superposed state, unnormalized | $O(1)$ | the deployable design |
| (d′) interface-token linear-state read | $M{\cdot}L$ (query-conditioned via pass 1) | separate interface tokens, mixtures within | ≈ current read cost | read-site ablation |

1. (b) vs (a) on ordered-concat: if (b) also fails, **write quality** is
   implicated and H2 is rejected — stop, revisit the writer.
2. (d) vs (b): the gap is pure state superposition; measure vs $N$.
3. Regression guards: single-key KV exact match and the forgetting matrix
   must not degrade (token-wise reads should make single-key *easier*).
4. (c) vs (d) tells whether the $O(1)$ state costs accuracy that exact
   per-token attention would recover.
5. (d′) vs (d) tells whether integration needs per-token reads at all, or
   only per-layer reads through a small interface; if (d′) solves concat at
   (d)-level accuracy, the read-site count was the load-bearing factor and
   the cheaper architecture wins.
6. Then the writer×read factorial with a forward writer (IDEAS.md baseline)
   and the pointer-chasing benchmark (depth-bounded iteration is the claim).

## Risks & mitigations

- **Scale calibration** (the practical one) — pinned normals + gates + the
  `mu_norm_ratio` diagnostic; if early training is unstable, drop
  `read_state_gate_init` before touching anything else.
- **Interference growth with $N$** — top-k segment routing (state from
  selected segments only); fewer/wider heads.
- **Gate collapse to zero** — the model learns to ignore memory; monitor
  per-layer gates; small-positive init and $\beta$ learnable per layer give
  the outer loop a cheap knob to turn reads back on.
- **Loss of the prefix workspace** — memory tokens at READ also served as
  scratch space; the bank-prefix arm bounds what this costs.
- **Outer-gradient pressure on writes** — many read sites now shape stored
  values; keep `hopfield_bptt_segments` bounding, watch inner-loss drift.

## Open questions

- Per-head states with head-specialized $\kappa_r$ (learned feature
  partitioning) vs a single shared-state read with wide heads.
- Should $\nu_r$ reuse the Hopfield value-code pathway
  (`hopfield_value_proj`) or read the raw stored tokens? (Raw is the cleaner
  first experiment; the projected code exists for the one-shot mixture's
  benefit.)
- Are `read_st`/`read_end` control tokens still useful without a memory
  prefix, or pure noise now?
- Does layer-0 (embedding-level) reading ever help, or should reads start at
  $\ell \ge 2$ where hidden states already aggregate query context?
- Segment-identity signal: is a learnable segment embedding needed for
  ordered tasks, or does query-position causality suffice?
- Interface mode: what makes the interface tokens specialize (M sizing, $M_0$
  init diversity, an auxiliary diversity loss on their reads)?
- Interface mode: should pass-1 tokens read the state (bidirectional
  conditioning: query shapes tokens, state content shapes them back), or is
  conditioning-only cleaner for the first experiment?

## Prior art

- **Titan** (Behrouz et al., 2024) — neural long-term memory as a matrix
  state, per-token per-layer reads combined additively with attention output
  under a gate; the closest precedent for the read. Differences: our state is
  written by per-segment gradient inner loops (Hopfield STORE), not online
  test-time updates, and the writer comparison is the research question.
- **Based** (Arora et al., 2024) — global linear attention summed with local
  softmax attention per layer; the additive-combination precedent.
- **Flamingo** — gated additive cross-attention into a frozen model; the
  gating pattern — and, for the interface-token variant, the resampler
  pattern (a small learnable token set cross-attending to memory, everything
  else reading them via attention; cf. Q-Former).
- **Fast weights / DeltaNet / GDN** — associative state read $o = Sq$; the
  in-repo Gated-Delta RETRIEVE is this at compressed-query granularity.
- **Linear attention** (Katharopoulos et al., 2020) and **modern Hopfield
  networks** — the kernel and the one-step-retrieval semantics; MQAR-style
  recall limits quantify the interference ceiling.
- **ARMT** — token-wise memory interface through a prefix; the bank-prefix
  control arm is its read.
