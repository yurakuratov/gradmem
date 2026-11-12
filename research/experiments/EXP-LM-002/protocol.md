# EXP-LM-002 protocol

The frozen executable protocol is
[`allocation_protocol.md`](../../../experiments/linear_memory/allocation_protocol.md).

It preregisters short-group calibration of `delta`, `A0`, and `B`; prediction
`delta + [A0+(G-1)B]/D`; the `(p,h,tau,D)` grid; two known address geometries;
held-out context lengths 16 and 32; the deployment-operation proxy; a
2048-scalar inference-memory cap; three allocation policies; a two-stage
leakage barrier; and fixed acceptance thresholds.

The selection artifact was sealed before held-out generation with SHA-256
`98bbe4e64d6eff40545028655569096cd2793e5c0b70242ac34ca54ecb40be1d`.
