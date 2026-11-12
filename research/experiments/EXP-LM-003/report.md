# EXP-LM-003 report

**Task ID:** EXP-LM-003  
**Status:** completed

**Objective:** Test whether shared local delta writing can beat independently
written LA-backed memory when both local width and every feasible integer
global width are optimized under the same error target, operation proxy, and
folded-memory cap.

**Methods:** Three policies, three new seeds, iid and reset-conditioned
correlated addresses, 4,096 calibration groups, and 1,024 held-out episodes
per local configuration. The run sealed all moments and selections before
held-out generation.

**Results:** Primary unrestricted coverage was 48/48 and median point-oracle
regret was 4.81%, passing the registered 90%/10% criterion. Shared selections
beat optimized independent selections in nine dependent reporting rows, all
iid and `Q=1`. The meaningful gains were at `J=32`: 16.9–19.5% fewer proxy
operations for `epsilon=.10` and 24.6% for `.25`, with more folded memory but
within the same cap. No shared win appeared at `Q=32` or in the correlated
family. All selected conditional-reference intervals and sampled-LA point
means were feasible.

**Evidence:** [Full report](../../../reports/REFINED_ALLOCATION_RESULTS.md) and
[raw results](../../../reports/results/linear_memory/EXP-LM-003/). Forty tests
passed; scientific execution took about 90 seconds total.

**Limitations:** The cost is analytical, routing is oracle, shared winners use
more memory, scenarios reuse episode sets, and changing reset horizon changes
the correlated generator's block distribution. No local-only, LoRA,
transformer, learned-routing, or wall-clock claim is supported.

**Conclusion:** Shared writing has a real low-query iid compute region in this
linear LA-backed model, conditional on a common memory cap. It is not
memory-Pareto-dominant and does not generalize automatically beyond this model.

**Recommended follow-up:** First add a width-optimized local-only baseline
under matched state, parameter, and retrieval-error budgets. Then calibrate the
operation proxy with measured runtime and memory; the current low-query pair is
a useful secondary systems check. Separately study learned addresses with
correlation blocks fixed independently of reset horizon.
