"""STAR worst-case stability regulariser tests (grad_memgpt_adaptive.py).

The STAR term (adapted from ICLR 2025, arXiv:2503.01595) is an outer-loss-only
addition: at each segment boundary it penalises the worst-case KL between READ
outputs on stored KV probes at m_s vs m_s+delta. These tests lock in:

  A. bitwise-off guarantee: star_weight=0 (or eval mode, or star on but no
     kv_queries) must reproduce the star-less forward EXACTLY -- predictions,
     mem and loss -- and leave the inner_loop_stats keys unchanged.
  B. WRITE-path invariance: with STAR on (train mode) the WRITE trajectory is
     untouched, so predictions / mem stay bitwise identical; only the combined
     loss grows by lambda * star_kl > 0.
  C. meta-gradient flow: under grad_mode="second" (and even "none"), the STAR
     term delivers nonzero gradient to the meta-params (self.mem / backbone).
  D. guards: no eligible probes -> term silently off (loss == target-only);
     star_boundaries/probe_scope scoping works.
  E. perturbation calibration: the 'random' perturbation hits exactly
     ||delta_j||/||m_j|| == gamma per token; a near-zero gamma makes the
     combined loss collapse back to the target-only loss.

Run two ways:
    python test_grad_memgpt_adaptive_star.py        # self-contained runner
    pytest test_grad_memgpt_adaptive_star.py        # if pytest is installed
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gmm_adp = _load_module("gmm_adp_star", "grad_memgpt_adaptive.py")
AdpModel, AdpConfig = _gmm_adp.GradMemGPT, _gmm_adp.GradMemGPTConfig

PAD_ID, BOS_ID = 0, 1
ATOL = 1e-6


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
    cfg.resid_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    return cfg


def _make_model(grad_mode="second", K=2, n_mem_tokens=4, **star_kwargs):
    torch.manual_seed(0)
    cfg = AdpConfig(
        base_config=_base_config(), n_mem_tokens=n_mem_tokens, K=K, lr=0.05,
        grad_mode=grad_mode, segment_size=4,          # context len 8 -> 2 segments
        **star_kwargs,
    )
    model = AdpModel(cfg)
    model.eval()
    return model


def _inputs():
    """Right-padded context (len 8 -> 2 segments of 4) + query + labels."""
    context = torch.tensor([
        [BOS_ID, 66, 40, 68, 51, 66, 62, 64],   # 8 real tokens -> both segments full
        [BOS_ID, 66, 40, 68, 51, 66, 0, 0],     # 6 real tokens -> seg 1 has pads
    ])
    query = torch.tensor([[BOS_ID, 66, 40, 68, 69],
                          [BOS_ID, 66, 40, 68, 69]])
    labels = torch.full((2, 5), -100, dtype=torch.long)
    labels[:, 3:5] = query[:, 3:5]
    return {"context_input_ids": context, "query_input_ids": query}, labels


def _kv_queries(n_q_per_seg=1):
    """Teacher-forced probes: one KV pair per segment per sample (seg_idx 0/1).

    Conventions follow build_kv_probe_queries in run_gradmemgpt_on_kv_retrieval:
    the probe is '?!K:V!|' = [?, !, K, :, V, !, |] (no BOS), target_mask flags
    the 'V!|' span and ignore_token_ids holds the structural '!' (68) / '|'
    (69) -- so, exactly as in compute_metrics_fn / _probe_exact_match, only the
    VALUE position survives scoring.
    """
    B, Q = 2, 7
    probe = [66, 68, 40, 62, 51, 68, 69]           # ? ! K : V ! |
    qids = torch.full((B, 2 * n_q_per_seg, Q), PAD_ID, dtype=torch.long)
    tmask = torch.zeros(B, 2 * n_q_per_seg, Q, dtype=torch.bool)
    seg_idx = torch.zeros(B, 2 * n_q_per_seg, dtype=torch.long)
    qmask = torch.zeros(B, 2 * n_q_per_seg, dtype=torch.bool)
    for b in range(B):
        for j in range(2 * n_q_per_seg):
            qids[b, j] = torch.tensor(probe)
            tmask[b, j, 4:7] = True                # the 'V!|' span; '!'/'|' ignored
            seg_idx[b, j] = j // n_q_per_seg       # first probes -> seg 0, rest -> seg 1
            qmask[b, j] = True
    return {'query_input_ids': qids, 'target_mask': tmask, 'seg_idx': seg_idx,
            'mask': qmask, 'ignore_token_ids': [68, 69]}


def _max_diff(a, b):
    return (a - b).abs().max().item()


def _star_on(**overrides):
    kw = dict(star_weight=0.1, star_gamma=0.05, star_epsilon=1e-2,
              star_correct_only=False)   # select all in-scope probes regardless of EM
    kw.update(overrides)
    return kw


def _replay_on(**overrides):
    kw = dict(replay_weight=0.1, replay_correct_only=False)
    kw.update(overrides)
    return kw


# --------------------------------------------------------------------------- #
# A. bitwise-off guarantee
# --------------------------------------------------------------------------- #
def test_star_off_bitwise_identical():
    inputs, labels = _inputs()
    kv = _kv_queries()
    model = _make_model()                          # star_weight=0 default
    inputs_with_kw = dict(kv_queries=kv)

    torch.manual_seed(123)
    out_plain = model(inputs, labels=labels, return_mem=True)
    torch.manual_seed(123)
    out_kv = model(inputs, labels=labels, return_mem=True, **inputs_with_kw)
    assert _max_diff(out_plain["loss"], out_kv["loss"]) == 0.0
    assert _max_diff(out_plain["predictions"], out_kv["predictions"]) == 0.0
    assert _max_diff(out_plain["mem"], out_kv["mem"]) == 0.0
    assert set(out_plain["inner_loop_stats"]) == set(out_kv["inner_loop_stats"])

    # star on but EVAL mode: train-only gate keeps the forward identical
    model_star = _make_model(**_star_on())
    model_star.eval()
    torch.manual_seed(123)
    out_eval = model_star(inputs, labels=labels, return_mem=True, **inputs_with_kw)
    assert _max_diff(out_plain["loss"], out_eval["loss"]) == 0.0
    assert set(out_plain["inner_loop_stats"]) == set(out_eval["inner_loop_stats"])
    print("  [off]     star-off / eval-mode forwards bitwise identical OK")


# --------------------------------------------------------------------------- #
# B. WRITE-path invariance + positive KL contribution
# --------------------------------------------------------------------------- #
def test_star_write_path_invariant_positive_kl():
    inputs, labels = _inputs()
    kv = _kv_queries()
    # large gamma so the contribution clears float32 resolution on the loss
    model = _make_model(**_star_on(star_gamma=1.0))

    model.eval()                                   # star off (train-only)
    torch.manual_seed(123)
    out_off = model(inputs, kv_queries=kv, labels=labels, return_mem=True)

    model.train()                                  # star on
    torch.manual_seed(123)
    out_on = model(inputs, kv_queries=kv, labels=labels, return_mem=True)

    # WRITE trajectory untouched: predictions and carried mem bitwise identical
    assert _max_diff(out_off["predictions"], out_on["predictions"]) == 0.0
    assert _max_diff(out_off["mem"], out_on["mem"]) == 0.0
    # the KL itself is strictly positive and probes were selected
    star_kl = out_on["inner_loop_stats"]["star_kl"].item()
    assert star_kl > 0.0, f"star_kl must be positive, got {star_kl}"
    assert out_on["inner_loop_stats"]["star_n_probes"].item() > 0
    # loss grew by lambda * star_kl (tolerance: float32 addition on the loss)
    diff = out_on["loss"].item() - out_off["loss"].item()
    expected = star_kl * model.star_weight
    assert diff > 0.0, f"loss must grow with STAR on (diff={diff:.3e})"
    assert abs(diff - expected) < 1e-5, f"diff={diff:.3e} != lambda*star_kl={expected:.3e}"
    print(f"  [write]   predictions/mem invariant; lambda*star_kl={expected:.3e} OK")


# --------------------------------------------------------------------------- #
# C. meta-gradient flow (second-order and even with a detached inner loop)
# --------------------------------------------------------------------------- #
def test_star_grad_flow():
    inputs, labels = _inputs()
    kv = _kv_queries()
    for grad_mode in ("second", "none"):
        model = _make_model(grad_mode=grad_mode, **_star_on())
        model.train()
        torch.manual_seed(123)
        model(inputs, kv_queries=kv, labels=labels)["loss"].backward()
        backbone_grad = sum(
            p.grad.abs().sum().item() for n, p in model.model.named_parameters()
            if p.grad is not None)
        assert backbone_grad > 0, \
            f"no STAR gradient reached the backbone (grad_mode={grad_mode})"
        if grad_mode == "second":
            assert model.mem.grad is not None and model.mem.grad.abs().sum() > 0, \
                "no STAR gradient reached self.mem (grad_mode=second)"
        else:
            # by design, grad_mode="none" detaches the recurrence entirely:
            # self.mem is NEVER trained in that mode; the STAR term must still
            # reach the backbone through the perturbed probe forward.
            assert model.mem.grad is None or model.mem.grad.abs().sum() == 0
        print(f"  [grad]    grad_mode={grad_mode:6s} mem/backbone grads as expected OK")


# --------------------------------------------------------------------------- #
# D. guards and scoping
# --------------------------------------------------------------------------- #
def test_star_no_probes_is_noop():
    inputs, labels = _inputs()
    # all probes attributed to segment 1 -> scope 'past' (< boundary_seg) is
    # empty at every boundary: the term contributes NOTHING to the loss, and
    # the idle-zeros stats mark it "enabled but idle" (star_n_probes == 0)
    model = _make_model(**_star_on(star_probe_scope="past"))
    model.train()
    kv = _kv_queries()
    kv['seg_idx'][:] = 1
    torch.manual_seed(123)
    out = model(inputs, kv_queries=kv, labels=labels)
    assert out["inner_loop_stats"]["star_kl"].item() == 0.0
    assert out["inner_loop_stats"]["star_n_probes"].item() == 0.0
    # and with probes present but 'last' boundary selection, exactly one term
    model2 = _make_model(**_star_on(star_boundaries="last"))
    model2.train()
    torch.manual_seed(123)
    out2 = model2(inputs, kv_queries=_kv_queries(), labels=labels)
    assert out2["inner_loop_stats"]["star_kl"].item() > 0
    print("  [guards]  empty scope -> idle zeros (no loss term); 'last' -> single term OK")


# --------------------------------------------------------------------------- #
# E. perturbation calibration
# --------------------------------------------------------------------------- #
def test_star_random_perturbation_norm_and_tiny_gamma():
    inputs, labels = _inputs()
    kv = _kv_queries()

    # 'random': ||delta_j|| / ||m_j|| == gamma exactly, per memory token
    model = _make_model(**_star_on(star_perturbation="random", star_gamma=0.05))
    model.train()
    torch.manual_seed(123)
    out = model(inputs, kv_queries=kv, labels=labels)
    ratio = out["inner_loop_stats"]["star_delta_ratio"].item()
    assert abs(ratio - 0.05) < 1e-4, f"random delta ratio {ratio} != gamma"

    # near-zero perturbation: the combined loss collapses to the star-less loss
    model_tiny = _make_model(**_star_on(star_perturbation="random", star_gamma=1e-7,
                                        star_epsilon=1e-7))
    model_tiny.eval()
    torch.manual_seed(123)
    out_off = model_tiny(dict(inputs), labels=labels)
    model_tiny.train()
    torch.manual_seed(123)
    out_tiny = model_tiny(inputs, kv_queries=kv, labels=labels)
    assert _max_diff(out_off["loss"], out_tiny["loss"]) < 1e-5
    print("  [calib]   random delta ratio == gamma; tiny delta -> loss collapses OK")


# --------------------------------------------------------------------------- #
# F. vanilla replay (outer-loss corrective ablation)
# --------------------------------------------------------------------------- #
def test_replay_outer_term():
    inputs, labels = _inputs()
    kv = _kv_queries()
    model = _make_model(**_replay_on())

    model.eval()                                   # replay off (train-only)
    torch.manual_seed(123)
    out_off = model(inputs, kv_queries=kv, labels=labels, return_mem=True)

    model.train()                                  # replay on
    torch.manual_seed(123)
    out_on = model(inputs, kv_queries=kv, labels=labels, return_mem=True)

    # outer-loss-only: WRITE trajectory and READ outputs bitwise identical
    assert _max_diff(out_off["predictions"], out_on["predictions"]) == 0.0
    assert _max_diff(out_off["mem"], out_on["mem"]) == 0.0
    ce = out_on["inner_loop_stats"]["replay_ce"].item()
    assert ce > 0.0, f"replay_ce must be positive, got {ce}"
    assert out_on["inner_loop_stats"]["replay_n_probes"].item() > 0
    em_rate = out_on["inner_loop_stats"]["replay_em_rate"].item()
    assert 0.0 <= em_rate <= 1.0
    # loss grew by rho * replay_ce
    diff = out_on["loss"].item() - out_off["loss"].item()
    expected = ce * model.replay_weight
    assert diff > 0.0 and abs(diff - expected) < 1e-5, \
        f"diff={diff:.3e} != rho*replay_ce={expected:.3e}"

    # meta-gradient flow: self.mem under "second", backbone even under "none"
    for grad_mode in ("second", "none"):
        m2 = _make_model(grad_mode=grad_mode, **_replay_on())
        m2.train()
        torch.manual_seed(123)
        m2(inputs, kv_queries=kv, labels=labels)["loss"].backward()
        backbone_grad = sum(p.grad.abs().sum().item()
                            for _, p in m2.model.named_parameters() if p.grad is not None)
        assert backbone_grad > 0, f"no replay gradient reached backbone ({grad_mode})"
        if grad_mode == "second":
            assert m2.mem.grad is not None and m2.mem.grad.abs().sum() > 0
    print(f"  [replay]  outer-only invariance; rho*ce={expected:.3e}; em_rate={em_rate:.2f} OK")


def test_star_replay_compose():
    inputs, labels = _inputs()
    kv = _kv_queries()
    model = _make_model(**_star_on(star_gamma=1.0), **_replay_on())
    model.eval()
    torch.manual_seed(123)
    out_off = model(inputs, kv_queries=kv, labels=labels, return_mem=True)
    model.train()
    torch.manual_seed(123)
    out_on = model(inputs, kv_queries=kv, labels=labels, return_mem=True)

    assert _max_diff(out_off["predictions"], out_on["predictions"]) == 0.0
    star_kl = out_on["inner_loop_stats"]["star_kl"].item()
    ce = out_on["inner_loop_stats"]["replay_ce"].item()
    diff = out_on["loss"].item() - out_off["loss"].item()
    expected = star_kl * model.star_weight + ce * model.replay_weight
    assert abs(diff - expected) < 1e-4, \
        f"diff={diff:.3e} != lambda*kl + rho*ce={expected:.3e}"
    print(f"  [compose] lambda*kl + rho*ce = {expected:.3e} additive OK")


# --------------------------------------------------------------------------- #
# G. vectorised EM mask == scalar helper (used by the STAR/replay x* mask)
# --------------------------------------------------------------------------- #
def test_probe_em_batch_matches_scalar():
    torch.manual_seed(7)
    logits = torch.randn(3, 5, 8, 70)                    # [B, n_q, Q+1, V]
    qids = torch.randint(2, 68, (3, 5, 7))               # [B, n_q, Q]
    tmask = torch.zeros(3, 5, 7, dtype=torch.bool)
    tmask[..., 4:] = True
    ig = [68, 69]
    scalar = torch.zeros(3, 5, dtype=torch.bool)
    for b in range(3):
        scalar[b] = AdpModel._probe_exact_match(logits[b], qids[b], tmask[b], ig)
    batch = AdpModel._probe_exact_match_batch(logits, qids, tmask, ig)
    assert torch.equal(scalar, batch), "vectorised EM mask diverged from scalar helper"
    print("  [em-batch] vectorised EM mask == scalar helper OK")


# near-zero perturbation collapses the loss is covered above; here we also
# lock the 'grad' delta calibration: ||delta_j||/||m_j|| ~= sqrt(gamma^2+eps^2)
# (delta0 vector-normalised per token + gamma-normalised ascent step; the two
# are near-orthogonal at random init). A bare eps*||m||*randn delta0 would
# give eps*sqrt(d) instead -- at d=128 that is ~11x the intended scale.
def test_star_grad_delta_calibration():
    inputs, labels = _inputs()
    gamma, eps = 0.05, 0.01
    model = _make_model(**_star_on(star_gamma=gamma, star_epsilon=eps))
    model.train()
    torch.manual_seed(123)
    out = model(inputs, kv_queries=_kv_queries(), labels=labels)
    ratio = out["inner_loop_stats"]["star_delta_ratio"].item()
    expected = (gamma ** 2 + eps ** 2) ** 0.5
    assert abs(ratio - expected) < 0.01, \
        f"grad delta ratio {ratio:.4f} != sqrt(g^2+e^2)={expected:.4f} " \
        f"(delta0 not vector-normalised?)"
    print(f"  [calib-g] grad delta ratio {ratio:.4f} ~= sqrt(g^2+e^2)={expected:.4f} OK")


TESTS = [
    test_star_off_bitwise_identical,
    test_star_write_path_invariant_positive_kl,
    test_star_grad_flow,
    test_star_no_probes_is_noop,
    test_star_random_perturbation_norm_and_tiny_gamma,
    test_replay_outer_term,
    test_star_replay_compose,
    test_probe_em_batch_matches_scalar,
    test_star_grad_delta_calibration,
]


def _main():
    print("=" * 70)
    print("grad_memgpt_adaptive STAR regulariser + vanilla-replay ablation")
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
