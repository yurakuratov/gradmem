# EXP-LM-002 post-result audit protocol

This protocol was written after EXP-LM-002 outcomes were available. It is an
instrumentation and reporting correction, not a new preregistered scientific
hypothesis. The audit rerun keeps the original grid, seeds, random-stream
derivation, sample counts, selector, memory cap, operation proxy, error
targets, and address geometries exactly unchanged. Its output is isolated in
`EXP-LM-002-AUDIT`; the original sealed artifacts remain unchanged.

The corrections are:

1. Hash the imported `model.py`, which supplies `build_commit_schedule`, as a
   frozen execution dependency.
2. Retain the original feasibility-among-selected and point-oracle regret
   metrics, while adding oracle-scenario coverage:

   `coverage = feasible selections in oracle-existing scenarios /
               all oracle-existing scenarios`.

   An empty denominator cannot pass. The audit threshold is 90%, matching the
   original numerical feasibility threshold. This coverage rule was not in
   the original preregistration and must be preregistered in future studies.
3. Call direct global-key draws a sampled-LA consistency check. They test the
   finite-sample implementation/distribution against a conditional reference
   derived from the same identity; they are not independent validation.
4. Add a sensitivity analysis using an interval-certified oracle. A candidate
   is certified feasible when its full episode-level 95% reference interval
   lies below the target. Report selected certified feasibility, regret versus
   the cheapest certified candidate, and whether a cheaper interval-uncertain
   candidate could alter the certified optimum.

The original point-feasible oracle remains the primary comparator. Scenario
rows that share held-out episodes across `Q`, error target, or policy are
dependent reporting views, not independent experimental replications.
