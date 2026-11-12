# Introduction to local gradient memory and global associative storage

We want a model to read a long sequence, store what it has learned, and answer
questions after the original sequence is no longer available. The proposed
architecture gives it two kinds of memory: a small working memory updated by
gradient descent, and a global associative memory that stores selected
versions of that working memory.

The central research question is how to divide resources between them. Should
we make the working memory larger, make the global memory larger, or store
versions more often? We want to answer that question for a specified retrieval
error and a specified amount of computation.

The theory starts with a deliberately simple model: a matrix that learns
key–value associations. It makes the effects of overwriting, storage, and
gradient routing explicit. Extending these results to transformer memory
tokens, prefixes, and LoRA is part of the research programme.

## Mathematical notation

A **vector** is a list of numbers. We use vectors to represent both a fact's
address and its answer. A **matrix** is a rectangular array of numbers that
transforms one vector into another through multiplication.

| Notation | Meaning |
|---|---|
| $x\in\mathbb R^p$ | The vector $x$ contains $p$ real numbers. |
| $W\in\mathbb R^{v\times p}$ | The matrix $W$ has $v$ rows and $p$ columns. |
| $x^\top$ | The transpose: a column vector written as a row. |
| $x^\top u$ | The dot product, a number measuring alignment between vectors of the same size. |
| $\lVert x\rVert^2$ | The sum of the squares of the entries of $x$. |
| $\widehat y$ | A predicted or retrieved version of $y$. |
| $\mathbb E[\cdot]$ | An average over the random quantities specified in the model. |
| $\lceil a\rceil$ | The smallest integer at least as large as $a$. |

An address has **unit length** when $\|x\|=1$. Two addresses are
**orthogonal** when their dot product is zero. For unit vectors, a dot product
near one means strong alignment; a value near zero means little overlap.

The superscript $\top$ means transpose, not a time index. Subscripts such
as $j$ usually identify a position in the input sequence.

## A local memory that stores associations

Suppose a fact says “the code for this item is 7.” Its representation has two
parts:

- A **local address** $x\in\mathbb R^p$, identifying the item.
- A **value** $y\in\mathbb R^v$, representing the answer.

Here $p$ is the local address dimension and $v$ is the value dimension.
The letter $v$ denotes a dimension; the value itself is $y$.

Our local memory is a matrix

$
W\in\mathbb R^{v\times p}.
$

To retrieve an answer, multiply the matrix by the address:

$
\widehat y=Wx.
$

For example, with two orthogonal addresses,

$
x_1=\begin{bmatrix}1\\0\end{bmatrix},\qquad
x_2=\begin{bmatrix}0\\1\end{bmatrix},\qquad
W=\begin{bmatrix}3&7\end{bmatrix},
$

the memory returns $Wx_1=3$ and $Wx_2=7$. Here $p=2$ and $v=1$.
The two columns store answers along independent address directions.

For fixed addresses and arbitrary answers, a single linear memory cannot
generally represent more independent associations than its address space
allows. In particular, there can be at most $p$ mutually orthogonal local
addresses. Structured or redundant answers can require fewer independent
directions, so this is not a claim that every dataset has exactly the same
capacity.

## Writing by gradient descent

When a new pair $(x,y)$ arrives, we compare the desired answer with the
current answer. Define the local write loss

$
\ell(W;x,y)=\frac12\|y-Wx\|^2.
$

This is half the squared prediction error. A **gradient** records how the
loss responds to a small change in each memory entry. Gradient descent uses
those sensitivities to choose a correction:

$
\boxed{W\leftarrow W+\eta(y-Wx)x^\top.}
$

The positive number $\eta$ is the **inner learning rate**, or write step
size. The vector $y-Wx$ is the current error. Multiplying it by the row
vector $x^\top$ produces a matrix with the same shape as $W$.
This update is called the **delta rule**: it writes a correction to the
current answer.

With a unit-length address and $\eta=1$, one step makes the current pair's
answer exact. But consider another address $u$. Its answer changes by

$
\Delta(Wu)=\eta(y-Wx)(x^\top u).
$

Here $\Delta(Wu)$ means the new answer minus the old answer. The expression
shows the source of interference directly. An orthogonal old address is
unaffected. An overlapping address can change when we write the new fact.
Repeated interference can destroy previously useful answers: this is the
form of forgetting that the model captures.

Gradient writing therefore does not inevitably cause forgetting. Address
geometry matters. Increasing local capacity can help by allowing facts to use
more distinct directions, provided the writer learns to use them.

## Segments and the local trajectory

Let $J$ be the number of input segments. The simplest model puts one fact
in each segment, so the number of facts $N$ equals $J$. In a transformer,
a segment may contain several facts, so these quantities need not coincide.

Let $K$ be the number of inner gradient steps per segment. Write $W_j$
for the local memory after finishing segment $j$. If we keep using the
same local memory, its trajectory is

$
W_0\longrightarrow W_1\longrightarrow W_2
\longrightarrow\cdots\longrightarrow W_J.
$

The initial state $W_0$ is zero in our linear example. A transformer can
instead start from a learned initialization. Each arrow represents $K$
updates using the next segment.

Two scheduling choices matter:

| Symbol | Meaning |
|---|---|
| $h$ | **Reset horizon:** how many segments share one local memory before it is reset. |
| $\tau$ | **Commit interval:** how often a snapshot of that local memory is sent to global storage. |

We assume $1\leq\tau\leq h$, count commits from the start of each reset
group, and always commit the group endpoint. A short final group also gets
an endpoint commit.

For example, with $h=6$ and $\tau=2$, we commit $W_2,W_4,W_6$, then
reset before segment 7. With $h=6$ and $\tau=6$, we store only $W_6$
from that group. With $h=J$ and $\tau=1$, the local memory persists
through the whole sequence and every segment endpoint is stored.

**Committing and resetting are separate actions.** A commit copies a version
into global storage; it need not interrupt the continuing local trajectory.

## Storing snapshots in a global linear memory

Let $C$ be the total number of commits. We index snapshots by
$c=1,\ldots,C$, and write $W^{(c)}$ for snapshot $c$. This distinguishes
the snapshot number from the segment index: the first snapshot might be
$W_2$, rather than $W_1$.

A snapshot contains

$
P=pv
$

numbers. The symbol $P$ denotes **payload size**. Flatten the snapshot
matrix into a vector $w_c\in\mathbb R^P$. Flattening only rearranges its
entries; it does not compress them.

Give the snapshot a **global address** $k_c\in\mathbb R^D$, where $D$
is the global address dimension. The global memory is

$
\boxed{S=\sum_{c=1}^{C}k_cw_c^\top\in\mathbb R^{D\times P}.}
$

The sum symbol means “add the contribution of every snapshot.” A single
commit performs the additive write

$
S\leftarrow S+k_cw_c^\top.
$

The local and global write rules now have distinct roles. The local delta
rule corrects an answer inside a working memory. The global additive rule
stores a representation of a selected version of that working memory.

To retrieve a snapshot, supply a global query $q\in\mathbb R^D$:

$
\widehat w=S^\top q
=\sum_{c=1}^{C}(q^\top k_c)w_c.
$

Reshape $\widehat w$ into a matrix $\widehat W$, then ask the local
question $x$:

$
\boxed{\widehat y=\widehat W x
=\sum_{c=1}^{C}(q^\top k_c)W^{(c)}x.}
$

There are consequently two addresses: $q$ selects a stored version;
$x$ selects an association within that version. These are raw linear
attention weights. They can be negative and need not sum to one; there is
no softmax or division by the sum of weights in this model. The read is
called linear because, for a fixed stored matrix $S$, its output is a
linear function of the global query $q$.

Adding a new global contribution does not rewrite the earlier contributions
algebraically. Nevertheless, their retrieved answers can change because the
new contribution participates in the same sum. Global additive storage
replaces one kind of interference with another; it does not remove every
source of forgetting or retrieval error.

## Separating local error from global interference

Suppose snapshot $r$ is the version we want to use. Our initial theory
assumes the query knows its address exactly:

$
q=k_r.
$

This is **oracle routing**: the experiment supplies the correct snapshot
address. In the pilot, it chooses the first commit after the fact was written,
within the same reset group. Learning this selection from a real question is
a separate problem.

Assume the global addresses have unit length. The target snapshot then has
weight $k_r^\top k_r=1$. Define

$
z_c=W^{(c)}x,\qquad
e=z_r-y,\qquad
n=\sum_{c\ne r}(k_r^\top k_c)z_c.
$

The vector $z_c$ is snapshot $c$'s answer to the local question. The
vector $e$ is the error of the desired snapshot itself. The vector $n$
is the unwanted contribution from the other snapshots. Therefore

$
\widehat y-y=e+n,
\qquad
\|\widehat y-y\|^2=\|e\|^2+\|n\|^2+2e^\top n.
$

The last term matters: for an individual query, interference can reinforce
or partly cancel the local error.

We can obtain an exact average-error formula under these assumptions:

1. Global addresses are sampled independently, have zero mean and unit
   length, and are equally spread across directions. The technical name for
   this directional symmetry is **isotropy**.
2. The local snapshots, local query, desired value, and selected snapshot
   index are independent of these global addresses.
3. The relevant squared quantities have finite averages.

The average unwanted weight is then zero, and its average square is
$1/D$. Products of weights for different unwanted snapshots average to
zero. Conditional on the local snapshots and question, this gives

$
\mathbb E\|n\|^2=\frac1D\sum_{c\ne r}\|z_c\|^2,
\qquad
\mathbb E[e^\top n]=0.
$

**The snapshots themselves may be correlated.** Independence is needed for
the global addresses in this argument, not for the payloads. Repeated
information still affects the size of their responses and can affect the
distribution of unusually large errors.

## The error law and a global capacity requirement

To compare different answer scales, define the positive normalizer
$Z=\mathbb E\|y\|^2$, the average squared size of the desired answer.
Define normalized mean squared error, abbreviated **NMSE**, as average
squared prediction error divided by $Z$. This is a ratio of averages;
it is not an average of separate per-query ratios.

Use three quantities:

$
\mathcal E=\frac{\mathbb E\|\widehat y-y\|^2}{Z},\qquad
\delta=\frac{\mathbb E\|W^{(r)}x-y\|^2}{Z},\qquad
A=\frac{\mathbb E\sum_{c\ne r}\|W^{(c)}x\|^2}{Z}.
$

Here $\mathcal E$ is total retrieval error, $\delta$ is local snapshot
error, and $A$ measures how strongly all the other snapshots respond to
the query. The expectations now also average over the workload of facts and
questions. Our result is

$
\boxed{\mathcal E=\delta+\frac{A}{D}.}
$

This equation explains the allocation problem. Improving the local writer
can reduce $\delta$. Increasing global address size $D$ reduces the
average interference term. Changing the commit interval can change both
$\delta$ and $A$.

The quantity $A$ is not simply the number of other snapshots. It becomes
approximately that count only if each contributes approximately one unit of
normalized response energy. This distinction matters for overlapping and
partly redundant snapshots.

Let $\varepsilon$ be the maximum acceptable NMSE. For a fixed local
writer and commit policy, if $A>0$ and $\delta<\varepsilon$, we need

$
\boxed{D_{\min}=\max\!\left(1,
\left\lceil\frac{A}{\varepsilon-\delta}\right\rceil\right).}
$

For example, if $\delta=0.10$, $A=0.80$, and the target is
$\varepsilon=0.20$, the predicted minimum is $D=8$. If the local error
already exceeds the target, enlarging global storage cannot fix that local
error in this model. If $A>0$ and $\delta=\varepsilon$, no finite
$D$ meets the target. If $A=0$, only $\delta\leq\varepsilon$ is
required, and $D=1$ suffices.

This is a requirement on **expected error**. It does not guarantee that every
question is answered accurately. Learned global addresses or address-dependent
snapshot selection also require a new error calculation. Retraining the local
writer when changing $D$ can change $\delta$ and $A$, so they cannot
automatically be treated as fixed across such experiments.

## A worked example where extra commits help or hurt

Consider two facts in one local group. Set $h=2$, $K=1$, $\eta=1$,
and start from zero memory. The values are independent scalar random variables
with zero mean and variance one: their average is zero and their average
square is one. Query the two facts equally often.

Let the local addresses have unit length and overlap

$
\rho=x_1^\top x_2.
$

The Greek letter $\rho$, pronounced “rho,” measures the alignment between
the two addresses. If we store only the endpoint, the second write leaves

$
W_2x_1=(1-\rho^2)y_1+\rho y_2,\qquad W_2x_2=y_2.
$

The first answer is partly changed; the second is exact. Averaging over both
questions and both random values gives endpoint error

$
\mathcal E_{\mathrm{end}}=\frac{\rho^2+\rho^4}{2}.
$

There is only one stored snapshot, so oracle global retrieval adds no
interference. If instead we commit after each fact, each question has an
exact local snapshot available, but the two global writes interfere:

$
\mathcal E_{\mathrm{snap}}=\frac{1+\rho^4}{2D}.
$

The subscripts “end” and “snap” label the two storage policies. For
$\rho\ne0$, snapshots are better precisely when

$
D>\frac{1+\rho^4}{\rho^2+\rho^4}.
$

For $\rho=0.5$, the threshold is $D>3.4$:

| Policy | Expected NMSE |
|---|---:|
| Endpoint only | 0.15625 |
| Two snapshots with $D=2$ | 0.265625 |
| Two snapshots with $D=4$ | 0.1328125 |

At $\rho=0$, endpoint storage is already perfect. Extra snapshots produce
positive interference at every finite random-address dimension. Thus the
theory predicts both a benefit and a failure regime. These comparisons
concern retrieval error; the policies also consume different resources.

## What local size and global size cost

Increasing $p$ makes the local address space larger. It also enlarges each
snapshot: $P=pv$. Consequently, in the whole-snapshot architecture,

$
\text{local state size}=pv,\qquad
\text{global state size}=DP=Dpv.
$

These are numbers of stored scalars. Multiply by the bytes per scalar to
obtain byte counts. Global address size and local payload size interact:
doubling $p$ doubles the global state at fixed $D,v$.

The useful exact-capacity construction is to allocate up to $p$ orthogonal
local addresses per group and up to $D$ orthogonal global addresses to
groups. It stores up to $Dp$ independent associations exactly. This uses
$Dpv$ state entries, so the additional capacity is paid for in storage.
For fixed query addresses and a linear reader, the combined address features
also have at most $Dp$ independent directions; arbitrary independent
answers cannot generally exceed that rank. Here **rank** counts independent
directions available to the linear mapping. Random addresses at a finite
error target have a different usable capacity, described by the error law.

Let $Q$ be the number of questions answered after writing the context.
For dense operations that reconstruct a full local matrix on each read, the
leading operation counts are:

| Operation | Cost grows approximately as |
|---|---:|
| Local writing over all segments | $JKpv$ |
| Committing all snapshots | $CDpv$ |
| Retrieving a full snapshot and answering one question | $Dpv+pv$ |
| Answering $Q$ questions this way | $Q(Dpv+pv)$ |

The table describes scaling, not exact runtime or measured processor FLOPs.
It excludes the address encoders and the mechanism that finds the desired
snapshot. Keeping a retrieved matrix for several local questions can amortize
retrieval. Sparse addresses and alternative contraction orders can change
the costs as well.

When $J$, $h$, and $\tau$ divide evenly, $C=J/\tau$. More
generally, if $h_g$ is the length of reset group $g$, then
$C=\sum_g\lceil h_g/\tau\rceil$. Smaller $\tau$ means more commits.
It does not reduce the total $JK$ local update steps.

An explicit bank of snapshots instead stores $CP$ payload numbers and,
when needed, $CD$ address numbers. A folded LA state has fixed shape for
fixed $D,P$, but it can be larger than a small bank. Keeping the error
fixed as the context grows may require increasing $D$.

Training memory is another resource: differentiating through the local writes
can require retaining intermediate computations and snapshots. A compact
inference state does not imply equally compact training.

## Why intermediate commits change gradient routing

There are two learning processes to distinguish. **Inner optimization** writes
the current episode into $W$. **Outer training** learns shared parameters
$\theta$, such as the address encoder or write step size, from many
episodes. Let $\mathcal L$ denote the final answer loss used for this outer
training. The symbol $\ell$ earlier denoted the local write loss.

With endpoint-only storage, an early write influences $\mathcal L$
through all the later local updates before the endpoint. With an intermediate
commit, another path is available:

$
\text{early write}\longrightarrow\text{stored snapshot}
\longrightarrow S\longrightarrow\text{answer loss}.
$

**Backpropagation** calculates how an earlier computation affects the final
loss by working backward through the computation. It sends gradients in the
reverse direction along these paths.
The extra route can supervise the early write without passing through the
subsequent local updates.

For the linear writer, this can be seen directly. Hold addresses and step
size fixed, and let $\Delta W$ be a small change in a memory state. One
later delta-rule update propagates that change as

$
\Delta W_{\mathrm{after}}
=\Delta W_{\mathrm{before}}(I_p-\eta xx^\top).
$

The matrix $I_p$ is the $p$-dimensional identity: multiplying by it
leaves a vector unchanged. Many subsequent writes multiply by many such
factors. A direct read of the earlier snapshot avoids those later factors.

From a completed segment to the next commit, at most $(\tau-1)K$ further
inner updates intervene. The local trajectory can still extend over $hK$
updates. Thus $\tau$ controls how frequently the trajectory becomes
accessible to the final loss, while $h$ controls how long it persists.
Resetting ends the state dependency between groups, though shared parameters
can still learn from all groups.

A shorter path is not a guarantee of faster optimization. Its gradient may
be small, point in an unhelpful direction, or conflict with other paths.
Committing a snapshot also does not change the inner write loss by itself;
the proposed convergence benefit concerns outer training.

To isolate this mechanism, compare two systems with identical snapshots and
identical forward predictions. In one, stop the gradient only on the
intermediate snapshot storage branches. In the other, allow it through.
Keep the live recurrence and final commits differentiable in both. This is
what **detached** versus **fully differentiable** snapshots means. Any shared
encoder gradient coming directly through the local read query remains in
both systems.

The comparison separates extra gradient routes from the forward benefit of
having historical versions available. A snapshot can preserve an old answer
even if no training gradient is allowed through that snapshot.

## Relation to transformer memory tokens and LoRA

In a transformer, the local writable object could be input memory tokens,
prefixes at several layers, or LoRA parameters. The matrix $W$ is a
tractable model of that object; it is not a claim that these objects all
behave as linear maps.

Memory tokens are writable vectors inserted into the model's input.
Per-layer prefixes supply additional writable vectors to attention at
different depths. **LoRA**, short for low-rank adaptation, represents a
weight change through the product of two smaller matrices. In our proposed
use, those smaller matrices could be updated to store the current context.

The distinction between local address dimension $p$ and payload size
$P$ is especially important there. For the linear model, $P=pv$.
For a transformer, the relationship between writable parameter count,
retained information, and useful address directions must be measured or
derived. More layers or a higher LoRA rank can increase local capacity
without lengthening the input prefix.

There is also a storage-design choice. The theory above stores an entire
local state under one global address. The existing token-wise LA interface
assigns keys and values to individual memory tokens. Those are different
payload representations, with different entry counts and costs. The
whole-snapshot formula cannot be transferred to token-wise storage merely
by renaming the dimensions.

For LoRA, a further issue is that averaging two sets of low-rank factors
does not generally average their resulting weight updates. A global memory
must specify whether it stores factors, an effective operator, or a code
interpreted by a decoder; the corresponding error law depends on that choice.

## What is established and what remains a hypothesis

Under the stated random-address assumptions, the error identity
$\mathcal E=\delta+A/D$ is an analytical result. The two-fact crossover
and the existence of shorter gradient routes are also derived results. They
do not imply universal superiority over a larger one-level memory.

The first controlled pilot verified identical forward values and different
writer gradients for detached/full snapshots. With exact snapshot reads,
both reached the chosen target at the same observed training step. With LA,
full differentiation reached it earlier in two of three seeds; their final
errors were essentially equal. The experiment used oracle global routing
and a learned position codebook: a list of address vectors selected by a
fact's position, rather than by its content. It supports further investigation
of the optimization mechanism, rather than a general convergence claim.

The intended scientific contribution is a predictive allocation rule: given
a workload, error target, and resource budget, select local capacity, global
address size, and reset/commit schedules before running a large search.
Testing that prediction on unseen workloads, and comparing against strong
deep-prefix and LoRA baselines, are the remaining central steps.

## Reference guide

| Symbol | Definition |
|---|---|
| $N,J,Q$ | Number of facts, input segments, and later questions; $N=J$ only in the one-fact-per-segment model. |
| $x,y$ | Local address and desired answer. |
| $p,v$ | Number of coordinates in a local address and an answer. |
| $W,W_j$ | Local memory matrix; its state after segment $j$. |
| $K,\eta$ | Inner updates per segment and their learning rate. |
| $h,\tau$ | Reset horizon and commit interval, measured in segments. |
| $C,c,r$ | Number of snapshots, a snapshot index, and the selected snapshot index. |
| $W^{(c)},w_c,P$ | Snapshot matrix, flattened snapshot, and its number of entries; $P=pv$ here. |
| $k_c,q,D$ | Snapshot global address, global query, and global address dimension. |
| $S$ | The global associative state, with $D\times P$ entries. |
| $z_c,e,n$ | Snapshot response to the local query, local residual, and global interference. |
| $Z$ | Average squared answer size used to normalize error. |
| $\mathcal E,\delta,A$ | Total NMSE, local NMSE, and normalized non-target response energy. |
| $\varepsilon,D_{\min}$ | Allowed expected NMSE and predicted minimum global dimension. |
| $\rho$ | Dot product of the two unit local addresses in the worked example. |
| $\theta,\ell,\mathcal L$ | Shared learned parameters, local write loss, and outer answer loss. |

The [pilot report](../reports/LINEAR_MEMORY_PILOT.md) contains the numerical
evidence and reproduction details. The [research roadmap](../PROGRESS.md#contribution-roadmap-2026-11-09-implementation-started-2026-11-10)
states the planned theory and experiments.
