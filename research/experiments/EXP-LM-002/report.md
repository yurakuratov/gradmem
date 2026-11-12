# EXP-LM-002 report

**Task ID:** EXP-LM-002  
**Status:** completed

**Objective:** Select fixed-writer local/global allocations for unseen context
lengths using only isolated-group calibration.

**Methods:** Three seeds, two known local-address geometries, 512 calibration
groups per local configuration, and 256 held-out complete episodes per
candidate. The full frozen design and leakage barrier are in
[`protocol.md`](protocol.md).

**Results:** All 48 unrestricted reporting scenarios were point-feasible and
feasible by their full 95% conditional-reference intervals. They reuse 12
held-out episode sets and are not independent replications. Median and maximum
operation regret were zero. All unrestricted choices used `h=tau=1`; no
shared-local point was optimal in the registered grid. All sampled-LA selected
means also met target as a finite-sample consistency check.

**Evidence:** The full report is
[`ALLOCATION_TRANSFER_RESULTS.md`](../../../reports/ALLOCATION_TRANSFER_RESULTS.md).
Raw calibration, sealed selections, held-out metrics, summary, and provenance
are in
[`reports/results/linear_memory/EXP-LM-002`](../../../reports/results/linear_memory/EXP-LM-002/).
All 26 focused and prior linear-memory tests passed.

**Limitations:** The geometries and hyperparameter values were known at
calibration, routing was oracle-supplied, the cost is an analytical proxy, and
the fixed writer strongly favors single-fact groups. There is no new
optimization-speed evidence.

**Conclusion:** Selected-point success used the exact `h=1` special case and
required no meaningful empirical calibration. Full-grid moment errors provide
the transfer evidence: median absolute error was 0.00512, with maximum 0.1095.
No `h>1` point was optimal within the registered `p<=8`, power-of-two-`D`
grid; this is not a claim outside that grid.

**Recommended follow-up:** Refine both local width and global-address
resolution, then test a preregistered sharing crossover using measured costs or
a justified compression model before transformer integration.
