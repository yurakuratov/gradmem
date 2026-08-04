"""Numerical-equivalence regression tests: adaptive fork vs the original.

With the default path ``memory_update_rule="sgd", n_segments=1,
segment_size=None`` the adaptive fork (``grad_memgpt_adaptive.py``) must be
bit-identical to the original (``grad_memgpt_old.py``). These tests lock that
in across the full ``grad_mode x mem_proj_mode`` matrix and check three things:

  A. forward outputs (predictions / mem / loss)
  B. per-step WRITE gradients (``g_mem`` from ``autograd.grad(inner_loss, mem_batch)``
     at each of the K inner steps; for ``per_sample`` also ``g_W, g_b``)
  C. second-order backward (outer grads on ``self.mem`` / ``self.mem_proj``)
  D. sensitivity guard: a non-sgd rule (``convex``) must *differ* from the old
     model, proving the harness is genuinely sensitive.

Run two ways:
    python test_grad_memgpt_adaptive_equivalence.py        # self-contained runner
    pytest test_grad_memgpt_adaptive_equivalence.py        # if pytest is installed
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Module loading: both files define top-level `GradMemGPT` / `GradMemGPTConfig`,
# so load them under distinct module names and import both pairs.
# --------------------------------------------------------------------------- #
def _load_module(name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gmm_old = _load_module("gmm_old", "grad_memgpt_old.py")
_gmm_adp = _load_module("gmm_adp", "grad_memgpt_adaptive.py")

OldModel, OldConfig = _gmm_old.GradMemGPT, _gmm_old.GradMemGPTConfig
AdpModel, AdpConfig = _gmm_adp.GradMemGPT, _gmm_adp.GradMemGPTConfig

# kv_alphabet_62 token ids used below (char-level WordLevel, 70 tokens).
PAD_ID, BOS_ID = 0, 1

# Equivalence tolerance. Observed maxdiff is 0.0 after the left-padding
# rollback; 1e-5 absorbs platform/float noise without masking a real regression.
ATOL = 1e-5


def _base_config():
    """Tiny from-scratch GPT2 config (AGENTS.md smoke-test recipe)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.vocab_size = 70
    cfg.pad_token_id = PAD_ID
    cfg.bos_token_id = BOS_ID
    cfg.eos_token_id = 2
    cfg.use_cache = False
    # disable dropout so the two forwards are deterministic given identical weights
    cfg.resid_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    return cfg


def _make_pair(grad_mode, mem_proj_mode, rule="sgd", K=3):
    """Build an (old, adaptive) instance pair with identical weights.

    The sgd path creates no gate heads, so the state-dict keys match exactly
    and a strict copy reproduces the old model's parameters in the adaptive one.
    """
    base = _base_config()
    torch.manual_seed(0)
    old_cfg = OldConfig(
        base_config=base, n_mem_tokens=4, K=K, lr=0.05, grad_mode=grad_mode,
        use_mem_proj=(mem_proj_mode != "none"), mem_proj_mode=mem_proj_mode,
    )
    adp_cfg = AdpConfig(
        base_config=base, n_mem_tokens=4, K=K, lr=0.05, grad_mode=grad_mode,
        use_mem_proj=(mem_proj_mode != "none"), mem_proj_mode=mem_proj_mode,
        n_segments=1, segment_size=None, memory_update_rule=rule,
    )
    old_model = OldModel(old_cfg)
    adp_model = AdpModel(adp_cfg)
    # sgd path: state dicts have identical keys -> strict copy. A gated rule
    # (convex/mamba) adds gate-head params the old model lacks -> strict=False
    # copies only the shared keys (backbone/mem/mem_proj); the gate heads keep
    # their __init__ values.
    adp_model.load_state_dict(old_model.state_dict(), strict=(rule == "sgd"))
    old_model.eval()
    adp_model.eval()
    return old_model, adp_model


def _inputs():
    """Right-padded context (2 samples, unequal real lengths) + query + labels."""
    # ids drawn from kv_alphabet_62: BOS, then '!K:V!'-ish chars, then pad.
    context = torch.tensor([
        [BOS_ID, 66, 40, 68, 51, 66, 0, 0],   # 6 real tokens
        [BOS_ID, 66, 40, 68, 0, 0, 0, 0],     # 4 real tokens  -> exercises padding
    ])
    query = torch.tensor([[BOS_ID, 66, 40, 68, 69],
                          [BOS_ID, 66, 40, 68, 69]])
    labels = torch.full((2, 5), -100, dtype=torch.long)
    labels[:, 3:5] = query[:, 3:5]            # score the last two tokens
    return {"context_input_ids": context, "query_input_ids": query}, labels


def _max_diff(a, b):
    return (a - b).abs().max().item()


class _GradCapture:
    """Scope object that records every torch.autograd.grad call within it.

    The WRITE phase computes g_mem = autograd.grad(inner_loss, ...) inside
    forward() and never exposes it. This wraps the call (the model looks up
    `torch.autograd.grad` by attribute each time, so the patch is picked up)
    and appends the returned gradient tuple to `recorded`, returning it
    unchanged -- so forward() behaves exactly as normal.
    """

    def __init__(self):
        self.recorded = []
        self._orig = None

    def __enter__(self):
        self._orig = torch.autograd.grad

        def wrapper(outputs, inputs, *args, **kwargs):
            res = self._orig(outputs, inputs, *args, **kwargs)
            self.recorded.append(tuple(t.detach().clone() for t in res))
            return res

        torch.autograd.grad = wrapper
        return self

    def __exit__(self, *exc):
        torch.autograd.grad = self._orig


GRAD_MODES = ("none", "first", "second")
MEM_PROJ_MODES = ("none", "proj", "per_sample")


# --------------------------------------------------------------------------- #
# A. forward equivalence (predictions / mem / loss)
# --------------------------------------------------------------------------- #
def test_forward_equivalence():
    inputs, labels = _inputs()
    for grad_mode in GRAD_MODES:
        for mpm in MEM_PROJ_MODES:
            old_model, adp_model = _make_pair(grad_mode, mpm)
            torch.manual_seed(123)
            out_old = old_model(inputs, labels=labels, return_mem=True)
            torch.manual_seed(123)
            out_adp = adp_model(inputs, labels=labels, return_mem=True)
            checks = {
                "predictions": _max_diff(out_old["predictions"], out_adp["predictions"]),
                "mem": _max_diff(out_old["mem"], out_adp["mem"]),
                "loss": _max_diff(out_old["loss"], out_adp["loss"]),
            }
            worst = max(checks.values())
            assert worst < ATOL, (
                f"forward mismatch grad_mode={grad_mode} mem_proj_mode={mpm}: "
                f"{checks}"
            )
            print(f"  [forward] grad_mode={grad_mode:6s} mem_proj_mode={mpm:10s} "
                  f"maxdiff={worst:.2e} OK")


# --------------------------------------------------------------------------- #
# B. per-step WRITE gradient equivalence
# --------------------------------------------------------------------------- #
def test_write_gradient_equivalence():
    inputs, labels = _inputs()
    for grad_mode in GRAD_MODES:
        for mpm in MEM_PROJ_MODES:
            old_model, adp_model = _make_pair(grad_mode, mpm)

            with _GradCapture() as cap_old:
                torch.manual_seed(123)
                old_model(inputs, labels=labels)
            with _GradCapture() as cap_adp:
                torch.manual_seed(123)
                adp_model(inputs, labels=labels)

            assert len(cap_old.recorded) == len(cap_adp.recorded), (
                f"different number of autograd.grad calls grad_mode={grad_mode} "
                f"mem_proj_mode={mpm}: old={len(cap_old.recorded)} adp={len(cap_adp.recorded)}"
            )
            worst = 0.0
            for step_idx, (g_old, g_adp) in enumerate(zip(cap_old.recorded, cap_adp.recorded)):
                assert len(g_old) == len(g_adp), (
                    f"grad-tuple arity mismatch step={step_idx} grad_mode={grad_mode} "
                    f"mem_proj_mode={mpm}: old={len(g_old)} adp={len(g_adp)}"
                )
                for go, ga in zip(g_old, g_adp):
                    worst = max(worst, _max_diff(go, ga))
            assert worst < ATOL, (
                f"write-grad mismatch grad_mode={grad_mode} mem_proj_mode={mpm}: "
                f"maxdiff={worst:.2e} (over {len(cap_old.recorded)} inner steps)"
            )
            n_tensors = len(cap_old.recorded[0]) if cap_old.recorded else 0
            print(f"  [wgrad]  grad_mode={grad_mode:6s} mem_proj_mode={mpm:10s} "
                  f"steps={len(cap_old.recorded)} tensors/step={n_tensors} "
                  f"maxdiff={worst:.2e} OK")


# --------------------------------------------------------------------------- #
# C. second-order backward equivalence (outer grads to meta-params)
# --------------------------------------------------------------------------- #
def test_backward_equivalence_second_order():
    inputs, labels = _inputs()
    grad_mode, mpm = "second", "proj"   # the user's default config
    old_model, adp_model = _make_pair(grad_mode, mpm)

    torch.manual_seed(123)
    old_model(inputs, labels=labels)["loss"].backward()
    torch.manual_seed(123)
    adp_model(inputs, labels=labels)["loss"].backward()

    checks = {
        "mem.grad": _max_diff(old_model.mem.grad, adp_model.mem.grad),
        "mem_proj.weight.grad": _max_diff(old_model.mem_proj.weight.grad,
                                          adp_model.mem_proj.weight.grad),
        "mem_proj.bias.grad": _max_diff(old_model.mem_proj.bias.grad,
                                        adp_model.mem_proj.bias.grad),
    }
    worst = max(checks.values())
    assert worst < ATOL, f"backward mismatch ({grad_mode}/{mpm}): {checks}"
    print(f"  [bwd]    grad_mode={grad_mode:6s} mem_proj_mode={mpm:10s} "
          f"maxdiff={worst:.2e} OK")


# --------------------------------------------------------------------------- #
# D. sensitivity guard: a non-sgd rule must differ from the old model
# --------------------------------------------------------------------------- #
def test_sensitivity_guard_convex_differs():
    inputs, labels = _inputs()
    grad_mode, mpm = "second", "proj"
    # _make_pair(rule="convex") copies the shared keys (backbone/mem/mem_proj)
    # non-strictly; convex_head keeps its __init__ values. Any difference in the
    # carried memory then comes purely from the convex update rule.
    old_model, adp_model = _make_pair(grad_mode, mpm, rule="convex")

    torch.manual_seed(123)
    out_old = old_model(inputs, labels=labels, return_mem=True)
    torch.manual_seed(123)
    out_adp = adp_model(inputs, labels=labels, return_mem=True)

    mem_diff = _max_diff(out_old["mem"], out_adp["mem"])
    # convex gate changes the WRITE update, so the carried memory must differ.
    assert mem_diff > ATOL, (
        f"sensitivity guard failed: convex rule produced identical mem "
        f"(diff={mem_diff:.2e}); harness is not sensitive."
    )
    print(f"  [guard]  convex rule differs from sgd: mem maxdiff={mem_diff:.2e} OK")


TESTS = [
    test_forward_equivalence,
    test_write_gradient_equivalence,
    test_backward_equivalence_second_order,
    test_sensitivity_guard_convex_differs,
]


def _main():
    print("=" * 70)
    print("grad_memgpt_adaptive vs grad_memgpt_old equivalence (sgd / n_segments=1)")
    print("=" * 70)
    n_fail = 0
    for test in TESTS:
        print(f"\n[{test.__name__}]")
        try:
            test()
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL: {e}")
    print("\n" + "=" * 70)
    if n_fail:
        print(f"RESULT: {n_fail}/{len(TESTS)} test(s) FAILED")
        sys.exit(1)
    print(f"RESULT: all {len(TESTS)} tests passed")


if __name__ == "__main__":
    _main()
