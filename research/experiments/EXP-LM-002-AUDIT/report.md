# EXP-LM-002-AUDIT report

**Task ID:** EXP-LM-002-AUDIT  
**Status:** completed

**Objective:** Correct EXP-LM-002 reproducibility and reporting gaps without
changing its scientific design.

**Methods:** Preserved the original executed sources, added the imported
commit-schedule module to frozen hashes, tested dependency-drift rejection and
abstention counterexamples, then repeated the identical grid and random streams
in a separate results directory.

**Results:** Calibration, predictions, held-out candidate metrics, selections,
and 40 common selection-evaluation fields are unchanged. Oracle-scenario
coverage was 48/48; the criterion cannot pass vacuously. A 95%-interval-
certified oracle existed in all 48 unrestricted reporting scenarios; every
selection was certified feasible; certified median and maximum regret were
zero; no cheaper interval-uncertain candidate could alter the certified
optimum. These rows reuse 12 held-out episode sets.

**Evidence:** [Reproduction comparison](../../../reports/results/linear_memory/EXP-LM-002-AUDIT/reproduction_comparison.json),
[audit summary](../../../reports/results/linear_memory/EXP-LM-002-AUDIT/summary.json),
and [canonical report](../../../reports/ALLOCATION_TRANSFER_RESULTS.md). The
final focused suite passed 30 tests.

**Limitations:** Coverage and certified-oracle analysis were introduced after
the original results. They are sensitivity checks, not preregistered primary
outcomes. The conditional reference and sampled LA share the same identity.

**Conclusion:** The original result survives the audit, but its selected-point
success is the exact independent-write special case. Evidence for learned
moment transfer lies in full-grid error, and no hierarchical advantage was
demonstrated outside the registered grid.

**Recommended follow-up:** Preregister a finer grid over both local width and
global address dimension, with realistic measured cost or justified
compression assumptions.
