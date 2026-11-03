# Bundled FlashBoB

This is a workspace-local source snapshot of FlashBoB's `src/bob` package,
copied on 2026-11-04 from the sibling checkout. It is **not** a symlink, Git
submodule, editable installation, or runtime dependency on that checkout.
The public import here is:

```python
from attn_double_bwd.flashbob import sdpa_bob
```

The BSD-3-Clause license and original copyright are retained in `LICENSE`.
See the project root README for the Transformers/GradMem integration and
`environment-flashbob.yaml` for compatible software dependencies. CUDA private
ATen signatures are tested with PyTorch 2.9.1 and Triton 3.5.1.

## Provenance

Source checkout HEAD: `f2bd4a38a9b65bae455ef4336d7b54f0acd3caca`.
The source working tree was **not clean**: `attention.py`, `sdpa.py`,
`kernels/rectangular.py`, and `kernels/square.py` contained local changes.
This snapshot deliberately preserves those source files as found; it is not
claimed to be an exact copy of that Git commit. Only package imports in
`__init__.py`, `attention.py`, and `sdpa.py` were changed to relative imports.

SHA-256 of the original source files **before** the import changes:

```text
b8266f62022be2d6037ffe373cf7c26b59762e2f670178222721d5416aaf8ebd  __init__.py
a94c8d4497a27d9f55b9fc6e1c0e043ab0ce87ae549232fe6a3da21f338eaba6  attention.py
4049bd4f63ba470004b0cc516137da8ca16665503b151c5d5ff56847f8fbd590  sdpa.py
132c8637abd5e3c9d9d5f71d6e851e965bab9f5c83863aae6c5880323aa21b96  kernels/common.py
8dcf4eeffe283522fd8076ee5c5836eeb9984502031bce893fc41028570ea784  kernels/gqa.py
e3be68a2de00d3d04208ad69527a66fc9dca2eb679582f73ef9e15cf2517eb32  kernels/__init__.py
c4d5271b7f06c80225b85fdd8e64e9a290c6d78bb0594363dcb1157237ed4a51  kernels/rectangular.py
cc51ca1b4b50275e3b0630b44c5d23d6c88d454ba69040f352c6ccd1becd0151  kernels/square.py
```

## Support

- Reverse-mode differentiation through second order.
- CUDA MHA: FP16/BF16, head dimensions 32, 48, 64, 128.
- CUDA square GQA/MQA: BF16, dimensions 32, 64, 128; grouping factor <=32.
- CPU MHA: differentiable PyTorch reference.
- Rectangular, offset, and causal sliding-window routes are retained.
- No native dropout or user-provided bias; the HF adapter uses a warned math
  fallback for those cases instead of discarding their semantics.
- The optional `flash-attn` dependency only accelerates eligible rectangular
  causal calls; those routes retain their upstream fallback when it is absent.
