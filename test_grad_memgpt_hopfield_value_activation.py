"""Nonlinear/sparse Hopfield value-code tests (grad_memgpt.py).

Locks in the hopfield_value_activation feature:

  1.  linear encode/decode helpers are EXACTLY the direct projections.
  2.  Default (linear) mode leaves existing Hopfield behavior and output
      shapes unchanged (and emits no value-code stats when unprojected).
  3.  topk codes are nonnegative with <= k nonzero entries per sample
      (nnz < k when fewer coords are positive).
  4.  topk backpropagates through selected values only; discarded coords
      receive no activation gradient (checked against the analytic grad).
  5.  relu codes are nonnegative.
  6.  With identical weights/inputs, enabling the ReLU L1 changes the
      combined loss by exactly weight * mean(abs(z)) over the valid
      sample-segments.
  7.  The unit-column-normalized decoder is used in BOTH retrieval paths
      (normal RETRIEVE and the forgetting probe) when ReLU L1 is active.
  8.  Gated mode initially produces exactly the ungated preactivation h.
  9.  Gate parameters receive finite, nonzero gradients.
  10. Every mode supports a widened Dv > P forward+backward with finite
      loss/gradients, plus the required hopfield_value_* diagnostics.
  11. collect_segment_mems stays prediction/loss-pure with an activated
      value mode.
  12. L1 does not defeat hopfield_bptt_segments: old stored codes are
      detached, the trainable sparse term covers only graph-retained
      segments, and backward succeeds.
  13. hopfield_value_as_key works with a widened activated code when a
      compatible hopfield_proj_dim is supplied; invalid unprojected dims
      fail with a clear error at model construction.
  14. Invalid activation/top-k/L1/gate configurations raise useful errors.
  15. Config to_dict/from_dict preserves the new fields; old configs
      without them load as linear with no auxiliary loss.

Run two ways:
    python test_grad_memgpt_hopfield_value_activation.py   # self-contained
    pytest test_grad_memgpt_hopfield_value_activation.py   # if pytest installed
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load_module(name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gmm = _load_module("gmm_hopfield_value", "grad_memgpt.py")
Model, ModelConfig = _gmm.GradMemGPT, _gmm.GradMemGPTConfig

PAD_ID, BOS_ID = 0, 1

# P = n_mem_tokens * n_embd of the tiny base config; Dv is the widened latent.
N_MEM_TOKENS = 4
P_DIM = N_MEM_TOKENS * 32          # 128
DV_WIDE = 2 * P_DIM                # 256 (> P)
TOPK_K = 32
L1_W = 0.05


class _expect_raises:
    """Minimal assertRaises stand-in so the file also runs without pytest."""

    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is not None, f"expected {self.exc_type.__name__} but nothing was raised"
        assert issubclass(exc_type, self.exc_type), \
            f"expected {self.exc_type.__name__}, got {exc_type.__name__}: {exc}"
        return True


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


def _make_model(n_mem_tokens=N_MEM_TOKENS, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        base_config=_base_config(), n_mem_tokens=n_mem_tokens, K=2, lr=0.05,
        grad_mode="none",                                # cheapest for structural tests
        **kw,
    )
    model = Model(cfg)
    model.eval()
    return model


def _hopfield_kw(**extra):
    kw = dict(use_hopfield_memory=True, hopfield_n_segments=2)
    kw.update(extra)
    return kw


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


def _collect(model, fwd_inputs):
    """forward(collect_segment_mems=True) without labels -> store + snapshots."""
    fwd = {"context_input_ids": fwd_inputs["context_input_ids"],
           "query_input_ids": fwd_inputs["context_input_ids"][:, :1]}
    out = model.forward(fwd, labels=None, collect_segment_mems=True)
    return out


def _normalize_decoder(model):
    """Make the decoder columns unit-normalized IN THE STORED WEIGHTS.

    The plain nn.Linear decode of an already-normalized weight equals the
    L1-path's F.normalize decode, so identical state dicts give identical
    base losses with ReLU L1 on vs off (isolating the sparse term).
    """
    with torch.no_grad():
        w = torch.nn.functional.normalize(model.hopfield_value_inv_proj.weight, dim=0)
        model.hopfield_value_inv_proj.weight.copy_(w)
    return model


# ---------------------------------------------------------------- #
# 1: linear helpers == direct projections
# ---------------------------------------------------------------- #

def test_linear_helpers_match_direct_projections():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE))
    v = torch.randn(3, P_DIM)
    z = m._encode_hopfield_value(v)
    h = m.hopfield_value_proj(v)
    assert torch.equal(z, h)
    d = m._decode_hopfield_value(z)
    assert torch.equal(d, m.hopfield_value_inv_proj(z))
    # unprojected Hopfield values pass through unchanged
    m0 = _make_model(seed=1, **_hopfield_kw())
    v0 = torch.randn(2, P_DIM)
    assert m0._encode_hopfield_value(v0) is v0
    assert m0._decode_hopfield_value(v0) is v0
    print("  [1] linear encode/decode == direct projections (identity when unprojected)")


# ---------------------------------------------------------------- #
# 2: default/linear mode unchanged
# ---------------------------------------------------------------- #

def test_default_linear_mode_unchanged():
    # config defaults for the new fields (old configs behave identically)
    cfg = ModelConfig(base_config=_base_config(), n_mem_tokens=N_MEM_TOKENS)
    assert cfg.hopfield_value_activation == "linear"
    assert cfg.hopfield_value_topk is None
    assert cfg.hopfield_value_l1_weight == 0.0
    assert cfg.hopfield_value_gate_init == 0.5

    # unprojected hopfield run: original shapes, no value-code stats
    m = _make_model(**_hopfield_kw())
    inputs, labels = _inputs()
    out = m.forward(inputs, labels=labels)
    assert out['predictions'].shape == (2, 6, 70)
    assert torch.isfinite(out['loss'])
    assert not any(k.startswith('hopfield_value_') for k in out['inner_loop_stats'])

    # projected linear run: stored values are [B, n_stored, Dv]
    mp = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE))
    out = _collect(mp, inputs)
    assert len(out['stored_values']) == 2
    for v in out['stored_values']:
        assert v.shape == (2, DV_WIDE)
        assert not v.requires_grad          # snapshots are detached
    print("  [2] default/linear mode: unchanged behavior, shapes, no unprojected stats")


# ---------------------------------------------------------------- #
# 3: topk code structure
# ---------------------------------------------------------------- #

def test_topk_codes_nonnegative_and_sparse():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="topk",
                                   hopfield_value_topk=TOPK_K))
    torch.manual_seed(7)
    v = torch.randn(5, P_DIM)
    z = m._encode_hopfield_value(v)
    assert (z >= 0).all()
    nnz = (z != 0).sum(dim=-1)
    assert (nnz <= TOPK_K).all(), f"nnz must be <= k, got max {int(nnz.max())}"
    assert (nnz > 0).any()                  # generic input keeps some winners

    # fewer than k positive coords -> nnz < k. Dv == P gets the identity
    # projection init, so h == v exactly: craft inputs with 3 and 0 positives.
    m2 = _make_model(**_hopfield_kw(hopfield_value_proj_dim=P_DIM,
                                    hopfield_value_activation="topk",
                                    hopfield_value_topk=8))
    v_sparse = torch.zeros(2, P_DIM)
    v_sparse[0, :3] = 1.0                   # 3 positive coords < k=8
    v_sparse[1, :5] = -1.0                  # no positive coords
    z_sparse = m2._encode_hopfield_value(v_sparse)
    assert torch.equal((z_sparse != 0).sum(dim=-1), torch.tensor([3, 0]))
    print("  [3] topk codes nonnegative, <= k winners (fewer when positives are scarce)")


# ---------------------------------------------------------------- #
# 4: topk gradients only through selected values
# ---------------------------------------------------------------- #

def test_topk_backprop_selected_coords_only():
    m = _make_model(seed=3, **_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                           hopfield_value_activation="topk",
                                           hopfield_value_topk=TOPK_K))
    torch.manual_seed(11)
    v0 = torch.randn(1, P_DIM)                      # deterministic, non-tied input
    with torch.no_grad():
        h = m.hopfield_value_proj(v0)
        vals, idx = h.topk(TOPK_K, dim=-1)
        assert (vals[:, :-1] > vals[:, 1:] + 1e-6).all(), "test needs distinct top values"

    v = v0.clone().requires_grad_(True)
    z = m._encode_hopfield_value(v)
    assert z.requires_grad
    coef = torch.randn_like(z).detach()
    (z * coef).sum().backward()

    # analytic gradient: only selected coords with h > 0 contribute
    W = m.hopfield_value_proj.weight                # [Dv, P]
    with torch.no_grad():
        assert (h[0, idx[0]] > 0).any(), "test needs at least one selected positive coord"
    expected = torch.zeros(P_DIM)
    with torch.no_grad():
        for i in idx[0].tolist():
            if h[0, i] > 0:
                expected += coef[0, i] * W[i]
    assert torch.allclose(v.grad[0], expected, atol=1e-6), \
        "v.grad must equal the sum over selected positive coords only"
    print("  [4] topk backprop: gradient flows through selected values only")


# ---------------------------------------------------------------- #
# 5: relu codes nonnegative
# ---------------------------------------------------------------- #

def test_relu_codes_nonnegative():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="relu"))
    z = m._encode_hopfield_value(torch.randn(4, P_DIM) * 3)
    assert (z >= 0).all()
    print("  [5] relu codes nonnegative")


# ---------------------------------------------------------------- #
# 6: L1 changes the combined loss by the exact weighted sparse term
# ---------------------------------------------------------------- #

def test_relu_l1_loss_delta():
    m_on = _normalize_decoder(_make_model(
        **_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                       hopfield_value_activation="relu",
                       hopfield_value_l1_weight=L1_W)))
    m_off = _make_model(seed=0, **_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                               hopfield_value_activation="relu",
                                               hopfield_value_l1_weight=0.0))
    m_off.load_state_dict(m_on.state_dict())

    inputs, labels = _inputs()
    out_on = m_on.forward(inputs, labels=labels, collect_segment_mems=True)
    out_off = m_off.forward(inputs, labels=labels, collect_segment_mems=True)
    assert torch.allclose(out_on['predictions'], out_off['predictions'], atol=1e-5)

    # expected sparse term over ALL valid sample-segments (no BPTT truncation)
    per_seg = [z[mask].abs().mean(dim=-1)
               for z, mask in zip(out_on['stored_values'], out_on['stored_masks'])]
    expected_l1 = torch.cat(per_seg).mean()
    delta = (out_on['loss'] - out_off['loss']).detach()
    # rtol covers float32 noise of subtracting two O(10) losses; any real
    # semantic error (wrong denominator, summed dims, included pads) is >1%
    assert torch.allclose(delta, L1_W * expected_l1, rtol=1e-4), \
        f"L1 loss delta {delta.item()} != {L1_W} * {expected_l1.item()}"
    print("  [6] enabling ReLU L1 shifts the combined loss by exactly weight * mean|z|")


# ---------------------------------------------------------------- #
# 7: normalized decoder in both retrieval paths when ReLU L1 is active
# ---------------------------------------------------------------- #

def _probe_expected_mean_retrieval(model, out, snap):
    """Reference decode for the 'mean' retrieval mode (query-independent)."""
    c = snap['n_stored']
    values = torch.stack([v.detach() for v in out['stored_values'][:c]], dim=1)   # [B, c, Dv]
    mask = torch.stack([m.detach() for m in out['stored_masks'][:c]], dim=1)
    weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    retrieved_code = (weights.unsqueeze(2) * values).sum(dim=1)                  # [B, Dv]
    with torch.no_grad():
        if model.hopfield_value_l1_active:
            w = torch.nn.functional.normalize(model.hopfield_value_inv_proj.weight, dim=0)
        else:
            w = model.hopfield_value_inv_proj.weight
        decoded = torch.nn.functional.linear(retrieved_code, w, model.hopfield_value_inv_proj.bias)
    return decoded


def test_decoder_normalization_both_retrieval_paths():
    common = _hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                          hopfield_value_activation="relu",
                          hopfield_value_l1_weight=L1_W,
                          hopfield_retrieval_mode="mean")
    m_probe = _make_model(seed=5, **common)
    inputs, _ = _inputs()
    out = _collect(m_probe, inputs)
    snap = out['segment_mems'][1]
    assert snap['n_stored'] == 2

    # probe path: _probe_boundary_memory must decode with normalized columns
    B = 2
    qids = inputs["query_input_ids"].unsqueeze(1)                     # [B, 1, Q]
    tmask = torch.zeros(B, 1, qids.size(2), dtype=torch.bool)
    mem_init = m_probe.mem.unsqueeze(0).expand(B, -1, -1)
    mem_det = m_probe._probe_boundary_memory(snap, out['stored_keys'], out['stored_values'],
                                             out['stored_masks'], qids, tmask, mem_init,
                                             None, None, B, inputs["context_input_ids"].device)
    expected = _probe_expected_mean_retrieval(m_probe, out, snap)
    assert torch.allclose(mem_det[:, 0], expected.view(B, N_MEM_TOKENS, -1), atol=1e-5), \
        "probe retrieval must use the unit-column-normalized decoder"

    # normal RETRIEVE path: l1>0 model with raw weights must match a twin whose
    # stored weight is pre-normalized with l1 off (identical effective decode)
    m_under = _make_model(seed=5, **common)
    m_twin = _make_model(seed=5, **_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                                hopfield_value_activation="relu",
                                                hopfield_value_l1_weight=0.0,
                                                hopfield_retrieval_mode="mean"))
    m_twin.load_state_dict(m_under.state_dict())   # same seed already; explicit sync
    _normalize_decoder(m_twin)
    with torch.no_grad():
        col_norms = m_under.hopfield_value_inv_proj.weight.norm(dim=0)
        assert not torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-3), \
            "test needs non-unit decoder columns in the raw model"

    fwd_inputs, labels = _inputs()
    p_under = m_under.forward(fwd_inputs, labels=labels)['predictions']
    p_twin = m_twin.forward(fwd_inputs, labels=labels)['predictions']
    assert torch.allclose(p_under, p_twin, atol=1e-4), \
        "normal retrieval must decode with normalized columns when ReLU L1 is active"

    # and the check has teeth: a plain-decode model with the raw weights differs
    m_raw = _make_model(seed=5, **_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                               hopfield_value_activation="relu",
                                               hopfield_value_l1_weight=0.0,
                                               hopfield_retrieval_mode="mean"))
    p_raw = m_raw.forward(fwd_inputs, labels=labels)['predictions']
    assert not torch.allclose(p_under, p_raw, atol=1e-4), \
        "sanity: normalization must actually change the decoded result here"
    print("  [7] normalized decoder used in normal RETRIEVE and probe paths (ReLU L1)")


# ---------------------------------------------------------------- #
# 8/9: gated mode init + gate gradients
# ---------------------------------------------------------------- #

def test_gated_initial_code_equals_h():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="gated",
                                   hopfield_value_gate_init=0.3))
    # zero weight + logit(0.3) bias -> constant initial gate
    assert (m.hopfield_value_gate.weight == 0).all()
    assert torch.allclose(m.hopfield_value_gate.bias,
                          torch.logit(torch.tensor(0.3)).expand_as(m.hopfield_value_gate.bias))
    v = torch.randn(3, P_DIM)
    z = m._encode_hopfield_value(v)
    h = m.hopfield_value_proj(v)
    assert torch.allclose(z, h, atol=1e-6), "initial gated code must equal h exactly"
    print("  [8] gated initial activated code == ungated preactivation h")


def test_gate_parameters_receive_gradients():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="gated"))
    inputs, labels = _inputs()
    out = m.forward(inputs, labels=labels)
    out['loss'].backward()
    gW = m.hopfield_value_gate.weight.grad
    gb = m.hopfield_value_gate.bias.grad
    assert gW is not None and gb is not None
    assert torch.isfinite(gW).all() and torch.isfinite(gb).all()
    assert gW.abs().sum() > 0 and gb.abs().sum() > 0
    print("  [9] gate weight/bias receive finite, nonzero gradients")


# ---------------------------------------------------------------- #
# 10: widened Dv > P forward/backward per mode + diagnostics
# ---------------------------------------------------------------- #

def test_all_modes_widened_forward_backward_and_stats():
    V = 70
    for mode in ("linear", "topk", "relu", "gated"):
        kw = _hopfield_kw(hopfield_value_proj_dim=DV_WIDE)
        if mode == "topk":
            kw.update(hopfield_value_activation="topk", hopfield_value_topk=TOPK_K)
        elif mode == "relu":
            kw.update(hopfield_value_activation="relu", hopfield_value_l1_weight=L1_W)
        elif mode == "gated":
            kw.update(hopfield_value_activation="gated")
        m = _make_model(**kw)
        inputs, labels = _inputs()
        out = m.forward(inputs, labels=labels)
        loss = out['loss']
        assert torch.isfinite(loss), f"{mode}: non-finite loss"
        loss.backward()
        # widened encoder exists and got a finite grad
        g = m.hopfield_value_proj.weight.grad
        assert g is not None and torch.isfinite(g).all(), f"{mode}: bad encoder grad"
        gdec = m.hopfield_value_inv_proj.weight.grad
        assert gdec is not None and torch.isfinite(gdec).all(), f"{mode}: bad decoder grad"
        for n, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{mode}: non-finite grad in {n}"
        assert out['predictions'].shape == (2, 6, V)

        stats = out['inner_loop_stats']
        for key in ("hopfield_value_code_abs_mean", "hopfield_value_code_norm_mean",
                    "hopfield_value_active_frac", "hopfield_value_feature_utilization",
                    "hopfield_value_decoder_col_norm_mean"):
            assert key in stats, f"{mode}: missing stat {key}"
            assert torch.isfinite(stats[key]), f"{mode}: non-finite {key}"
        af = float(stats['hopfield_value_active_frac'])
        assert 0.0 <= af <= 1.0
        if mode == "topk":
            assert af <= TOPK_K / DV_WIDE + 1e-6
        if mode == "relu":
            assert abs(float(stats['hopfield_value_l1_weight']) - L1_W) < 1e-6
            assert float(stats['hopfield_value_l1']) > 0
            # normalized decoder columns are pinned at 1.0
            assert abs(float(stats['hopfield_value_decoder_col_norm_mean']) - 1.0) < 1e-6
        if mode == "gated":
            # zero-init gate weight -> constant gate_init initially
            assert abs(float(stats['hopfield_value_gate_mean']) - 0.5) < 1e-6
            assert abs(float(stats['hopfield_value_gate_saturation'])) < 1e-6
    print("  [10] widened Dv>P forward/backward finite for all modes + diagnostics present")


# ---------------------------------------------------------------- #
# 11: collect_segment_mems purity with an activated mode
# ---------------------------------------------------------------- #

def test_collect_segment_mems_purity_activated_mode():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="topk",
                                   hopfield_value_topk=TOPK_K))
    inputs, labels = _inputs()
    out_plain = m.forward(inputs, labels=labels)
    out_coll = m.forward(inputs, labels=labels, collect_segment_mems=True)
    assert torch.equal(out_plain['predictions'], out_coll['predictions'])
    assert torch.equal(out_plain['loss'], out_coll['loss'])
    print("  [11] collect_segment_mems is prediction/loss-pure with topk codes")


# ---------------------------------------------------------------- #
# 12: L1 respects hopfield_bptt_segments
# ---------------------------------------------------------------- #

def test_relu_l1_respects_bptt_segments():
    # large-ish weight so the sparse term dominates the float32 noise of
    # subtracting two O(10) losses
    w = 0.5
    kw = _hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                      hopfield_value_activation="relu",
                      hopfield_value_l1_weight=w,
                      hopfield_bptt_segments=1)          # only the LAST segment stays in the graph
    m_on = _normalize_decoder(_make_model(seed=9, **kw))
    m_off = _make_model(seed=9, **_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                               hopfield_value_activation="relu",
                                               hopfield_value_l1_weight=0.0,
                                               hopfield_bptt_segments=1))
    m_off.load_state_dict(m_on.state_dict())

    inputs, labels = _inputs()
    out_on = m_on.forward(inputs, labels=labels, collect_segment_mems=True)
    out_off = m_off.forward(inputs, labels=labels, collect_segment_mems=True)

    # seg 0 is outside the BPTT window -> its code was detached at STORE time
    # (trainable accumulator saw requires_grad=False), seg 1 stayed connected.
    z_all = out_on['stored_values']
    masks = out_on['stored_masks']
    per_seg = [z[mask].abs().mean(dim=-1) for z, mask in zip(z_all, masks)]
    only_last = per_seg[-1]
    all_seg = torch.cat(per_seg).mean()
    delta = (out_on['loss'] - out_off['loss']).detach()
    assert torch.allclose(delta, w * only_last.mean(), rtol=1e-3), \
        "the trainable L1 term must cover ONLY the graph-retained segments"
    assert not torch.allclose(delta, w * all_seg, rtol=1e-3), \
        "detached (old) segments must NOT contribute to the trainable L1 term"
    # ... while the detached diagnostic still averages over ALL processed segments
    assert torch.allclose(out_on['inner_loop_stats']['hopfield_value_l1'], all_seg, rtol=1e-5), \
        "hopfield_value_l1 diagnostic must average over all processed segments"

    # backward succeeds: the L1 term never references the freed old graphs
    out_on['loss'].backward()
    print("  [12] L1 under hopfield_bptt_segments: old codes detached, loss graph-safe")


# ---------------------------------------------------------------- #
# 13: value-as-key dimension rules
# ---------------------------------------------------------------- #

def test_value_as_key_dimension_rules():
    # widened activated code as key + compatible key projection -> works
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="topk",
                                   hopfield_value_topk=TOPK_K,
                                   hopfield_value_as_key=True,
                                   hopfield_proj_dim=64))
    assert m.hopfield_key_proj.weight.shape == (64, DV_WIDE)   # key proj: Dv -> proj
    assert m.hopfield_query_proj.weight.shape == (64, P_DIM)   # query proj: P -> proj
    inputs, labels = _inputs()
    out = m.forward(inputs, labels=labels)
    assert torch.isfinite(out['loss'])

    # unprojected value-as-key (source == P, the original behavior) still works
    m0 = _make_model(**_hopfield_kw(hopfield_value_as_key=True))
    out0 = m0.forward(inputs, labels=labels)
    assert torch.isfinite(out0['loss'])

    # widened code as key WITHOUT a key projection: clear error at construction
    try:
        _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_as_key=True))
        raise AssertionError("expected ValueError for widened value-as-key without hopfield_proj_dim")
    except ValueError as e:
        assert "hopfield_value_as_key" in str(e) and "hopfield_proj_dim" in str(e)
    print("  [13] value-as-key: works with compatible key projection, fails clearly otherwise")


# ---------------------------------------------------------------- #
# 14: invalid configurations raise
# ---------------------------------------------------------------- #

def test_invalid_configurations_raise():
    base = dict(base_config=_base_config(), n_mem_tokens=N_MEM_TOKENS)

    def bad(**kw):
        with _expect_raises(AssertionError):
            ModelConfig(**base, **kw)

    bad(hopfield_value_activation="bogus")
    bad(hopfield_value_activation="relu")                       # no hopfield memory
    bad(use_hopfield_memory=True, hopfield_value_activation="relu")   # no proj dim
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="topk")                       # k missing
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="topk", hopfield_value_topk=0)
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="topk", hopfield_value_topk=DV_WIDE + 1)
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="relu", hopfield_value_topk=8)  # topk set in relu mode
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="linear", hopfield_value_topk=8)  # topk set in linear mode
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="relu", hopfield_value_l1_weight=-0.1)
    bad(use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
        hopfield_value_activation="gated", hopfield_value_l1_weight=0.1)  # L1 only for relu
    bad(hopfield_value_gate_init=0.0)
    bad(hopfield_value_gate_init=1.0)
    print("  [14] invalid activation/topk/L1/gate configurations raise useful errors")


# ---------------------------------------------------------------- #
# 15: config save/load
# ---------------------------------------------------------------- #

def test_config_save_load():
    cfg = ModelConfig(base_config=_base_config(), n_mem_tokens=N_MEM_TOKENS,
                      use_hopfield_memory=True, hopfield_value_proj_dim=DV_WIDE,
                      hopfield_value_activation="relu", hopfield_value_topk=None,
                      hopfield_value_l1_weight=0.01, hopfield_value_gate_init=0.7)
    d = cfg.to_dict()
    cfg2 = ModelConfig.from_dict(dict(d))
    assert cfg2.hopfield_value_activation == "relu"
    assert cfg2.hopfield_value_topk is None
    assert cfg2.hopfield_value_l1_weight == 0.01
    assert cfg2.hopfield_value_gate_init == 0.7
    cfg_topk = ModelConfig.from_dict(dict(d, hopfield_value_activation="topk",
                                          hopfield_value_topk=16, hopfield_value_l1_weight=0.0))
    assert cfg_topk.hopfield_value_activation == "topk" and cfg_topk.hopfield_value_topk == 16

    # old serialized configs lack the new fields -> linear, no aux loss
    old = {k: v for k, v in d.items()
           if k not in ("hopfield_value_activation", "hopfield_value_topk",
                        "hopfield_value_l1_weight", "hopfield_value_gate_init")}
    cfg3 = ModelConfig.from_dict(old)
    assert cfg3.hopfield_value_activation == "linear"
    assert cfg3.hopfield_value_topk is None
    assert cfg3.hopfield_value_l1_weight == 0.0
    assert cfg3.hopfield_value_gate_init == 0.5
    print("  [15] config save/load preserves new fields; old configs default to linear")


# ---------------------------------------------------------------- #
# extra: enabled-but-idle stats are explicit zeros
# ---------------------------------------------------------------- #

def test_idle_stats_explicit_zeros():
    m = _make_model(**_hopfield_kw(hopfield_value_proj_dim=DV_WIDE,
                                   hopfield_value_activation="gated"))
    pad_inputs = {"context_input_ids": torch.full((2, 8), PAD_ID, dtype=torch.long),
                  "query_input_ids": _inputs()[0]["query_input_ids"]}
    out = m.forward(pad_inputs, labels=None, collect_segment_mems=True)
    stats = out['inner_loop_stats']
    for key in ("hopfield_value_code_abs_mean", "hopfield_value_code_norm_mean",
                "hopfield_value_active_frac", "hopfield_value_feature_utilization",
                "hopfield_value_gate_mean", "hopfield_value_gate_saturation"):
        assert key in stats, f"missing idle stat {key}"
        assert float(stats[key]) == 0.0, f"{key} must be an explicit zero when idle"
    print("  [extra] enabled-but-idle value-code stats are explicit zeros")


if __name__ == "__main__":
    test_linear_helpers_match_direct_projections()
    test_default_linear_mode_unchanged()
    test_topk_codes_nonnegative_and_sparse()
    test_topk_backprop_selected_coords_only()
    test_relu_codes_nonnegative()
    test_relu_l1_loss_delta()
    test_decoder_normalization_both_retrieval_paths()
    test_gated_initial_code_equals_h()
    test_gate_parameters_receive_gradients()
    test_all_modes_widened_forward_backward_and_stats()
    test_collect_segment_mems_purity_activated_mode()
    test_relu_l1_respects_bptt_segments()
    test_value_as_key_dimension_rules()
    test_invalid_configurations_raise()
    test_config_save_load()
    test_idle_stats_explicit_zeros()
    print("ALL hopfield value-activation tests passed")
