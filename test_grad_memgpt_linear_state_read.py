"""Linear-state per-level read tests (grad_memgpt.py).

Locks in the use_linear_state_read feature (docs/TOKENWISE_READ.md):

  1.  The state read equals explicit attention over the unfolded bank:
      mu = q S_r == sum_(s,j) (q . k_(s,j)) v_(s,j), in float64, all heads,
      with stored-out items contributing exactly zero.
  2.  Zero gates are bitwise inert inside the new read layout (the injected
      bias is exactly 0; outputs equal a hooks-disabled pass).
  3.  No stored segments -> S == 0 -> zero bias (no fallback prefix).
  4.  The bias lands ONLY at real read sites (pads unaffected at the
      injecting block); interface mode: only at the interface positions, in
      BOTH passes.
  5.  The exact-softmax arm matches a manual per-token softmax over the bank.
  6.  Interface pass-1 conditioning is blinded to the target span (perturbing
      target tokens leaves the interface embeddings unchanged) but sensitive
      to non-target query tokens (the query-blind-prefix failure guard).
  7.  forward_per_segment_eval in the new mode: the boundary-k probe equals a
      forward that wrote only the first k segments (logits agree; exact-match
      flags identical).
  8.  Write side untouched: stored values are bitwise identical with the read
      hooks enabled vs disabled, and (grad_mode="second") the read params get
      outer-loss gradients while the inner loop succeeds unchanged.
  9.  collect_segment_mems purity in the new mode; None keys.
  10. Config guards: mutual exclusions and required fields.

Run two ways:
    python test_grad_memgpt_linear_state_read.py   # self-contained
    pytest test_grad_memgpt_linear_state_read.py   # if pytest installed
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


_gmm = _load_module("gmm_linear_state_read", "grad_memgpt.py")
Model, ModelConfig = _gmm.GradMemGPT, _gmm.GradMemGPTConfig

PAD_ID, BOS_ID = 0, 1


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


def _make_model(n_mem_tokens=2, seed=0, read_sites="all", exact=False, grad_mode="none",
                n_segments=2, n_ctrl_tokens=0, **kw):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        base_config=_base_config(), n_mem_tokens=n_mem_tokens, K=2, lr=0.05,
        grad_mode=grad_mode, n_ctrl_tokens=n_ctrl_tokens,
        use_hopfield_memory=True, hopfield_n_segments=n_segments,
        use_linear_state_read=True, read_state_heads=2,
        read_state_read_sites=read_sites, read_state_exact_softmax=exact,
        **kw,
    )
    model = Model(cfg)
    model.eval()
    return model


def _inputs():
    """Context (8 real tokens -> 2 segments of 4) + query + labels."""
    ctx = torch.tensor([[BOS_ID, 66, 40, 68, 51, 66, 62, 64]])
    qry = torch.tensor([[BOS_ID, 66, 40, 68, 69]])
    labels = torch.full((1, 5), -100, dtype=torch.long)
    labels[:, 3:5] = qry[:, 3:5]
    return {"context_input_ids": ctx, "query_input_ids": qry}, labels


def _random_state(model, B=2, c=2, mask_one=False, seed=1, dtype=torch.float32):
    """A read state from a random bank; with mask_one, the tail segment is stored-out."""
    g = torch.Generator().manual_seed(seed)
    P = model.n_mem_tokens * model.model.config.n_embd
    values = torch.randn(B, c, P, generator=g, dtype=dtype)
    masks = torch.ones(B, c, dtype=torch.bool)
    if mask_one:
        masks[:, -1] = False                      # stored-out tail segment
    return model._build_linear_state(values, masks)


# --------------------------------------------------------------------- #
# P1 — the state read IS the explicit bank attention
# --------------------------------------------------------------------- #

def test_state_read_equals_explicit_attention():
    """mu = q S_r == sum_(q.k) v over the bank (float64, all heads, masked-out items zero)."""
    model = _make_model()
    model = model.double()                        # exactness: run everything in float64
    P = model.n_mem_tokens * model.model.config.n_embd
    state = _random_state(model, B=2, c=2, mask_one=True, seed=3, dtype=torch.float64)

    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 5, model.model.config.n_embd, generator=g, dtype=torch.float64)
    mu = model._linear_state_read_bias(0, x, dict(state))

    H, d_h = model.read_state_heads, model.read_state_head_dim
    # replicate the query path exactly: LN -> Q -> normalize -> beta
    h = model.read_state_ln[0](x)
    q = model.read_state_q_proj[0](h).view(2, 5, H, d_h)
    q_hat = torch.nn.functional.normalize(q, dim=-1)
    beta = model.read_state_beta[0].view(1, 1, H, 1)
    q_s = q_hat * beta

    K, V, item_mask = state['K'], state['V'], state['item_mask']
    # explicit per-item accumulation: mu[b,t,h,:] = sum_n (q.k_n) v_n (stored items only),
    # then the same concat -> W_o -> gamma tail the bias function applies.
    mu_read = torch.zeros(2, 5, H, d_h, dtype=torch.float64)
    for n in range(K.size(2)):
        dot = (q_s * K[:, :, n, :].unsqueeze(1)).sum(-1)          # [B, T, H]
        contrib = dot.unsqueeze(-1) * V[:, :, n, :].unsqueeze(1)  # [B, T, H, d_h]
        contrib = contrib * item_mask[:, n].view(2, 1, 1, 1).to(torch.float64)
        mu_read = mu_read + contrib
    mu_ref = model.read_state_out_proj(mu_read.reshape(2, 5, -1)) \
        * model.read_state_gamma[0]

    assert torch.allclose(mu, mu_ref, atol=1e-10), (mu - mu_ref).abs().max()
    assert (~item_mask).sum() > 0                          # the mask was exercised
    # a sample with NO stored segments folds to S == 0
    empty = model._build_linear_state(torch.randn(2, 2, P, dtype=torch.float64),
                                      torch.zeros(2, 2, dtype=torch.bool))
    assert float(empty['S'].abs().max()) == 0.0


# --------------------------------------------------------------------- #
# P2 — default mode (read_sites="all")
# --------------------------------------------------------------------- #

def test_gate_zero_is_bitwise_inert():
    """gamma == 0 -> the injected bias is exactly 0: outputs equal a hooks-disabled pass."""
    model = _make_model()
    inputs, _ = _inputs()
    with torch.no_grad():
        out_on = model.forward(inputs)
        for gamma in model.read_state_gamma:
            gamma.zero_()
        out_gated = model.forward(inputs)
        model._read_state_hooks_disabled = True
        out_off = model.forward(inputs)
        model._read_state_hooks_disabled = False
    assert torch.equal(out_gated['predictions'], out_off['predictions'])
    assert not torch.equal(out_on['predictions'], out_gated['predictions'])  # gates were nonzero


def test_no_stored_segments_no_bias():
    """All-pad context: nothing stored -> S == 0 -> READ identical to a hooks-disabled pass.
    Both arms (the exact-softmax arm must not NaN on an all-masked bank)."""
    for exact in (False, True):
        model = _make_model(exact=exact)
        ctx = torch.full((1, 8), PAD_ID, dtype=torch.long)
        qry = torch.tensor([[BOS_ID, 66, 40, 68, 69]])
        inputs = {"context_input_ids": ctx, "query_input_ids": qry}
        with torch.no_grad():
            out = model.forward(inputs)
            model._read_state_hooks_disabled = True
            out_off = model.forward(inputs)
            model._read_state_hooks_disabled = False
        assert torch.equal(out['predictions'], out_off['predictions']), f"exact={exact}"
        assert not out['predictions'].isnan().any(), f"exact={exact}"
        # the state itself is exactly zero for a stored-out bank
        P = model.n_mem_tokens * model.model.config.n_embd
        state = model._build_linear_state(torch.randn(1, 1, P),
                                          torch.zeros(1, 1, dtype=torch.bool))
        assert float(state['S'].abs().max()) == 0.0
        assert state['item_mask'].sum() == 0
        # ...and the read kernel gives exactly zero on it (both arms)
        x = torch.randn(1, 4, model.model.config.n_embd)
        mu = model._linear_state_read_bias(0, x, dict(state))
        assert float(mu.abs().max()) == 0.0, f"exact={exact}"


def test_read_sites_masking():
    """Bias present only at real read sites; pad positions of the read sequence unaffected."""
    model = _make_model()
    # query with trailing pad tokens: last two positions are pads
    ctx = torch.tensor([[BOS_ID, 66, 40, 68, 51, 66, 62, 64]])
    qry = torch.tensor([[BOS_ID, 66, 40, 68, 0]])          # position 4 is pad
    inputs = {"context_input_ids": ctx, "query_input_ids": qry}
    pad_pos = [4]
    real_pos = [0, 1, 2, 3]

    captures_on, captures_off = [], []
    block0 = _gmm.get_backbone(model.model).h[0]

    def spy(store):
        def _spy(module, args, kwargs, output):
            store.append(output[0].detach().clone())
            return None
        return _spy

    h_on = block0.register_forward_hook(spy(captures_on), with_kwargs=True)
    with torch.no_grad():
        model.forward(inputs)
    h_on.remove()

    model._read_state_hooks_disabled = True
    h_off = block0.register_forward_hook(spy(captures_off), with_kwargs=True)
    with torch.no_grad():
        model.forward(inputs)
    h_off.remove()
    model._read_state_hooks_disabled = False

    # the spy sees every block-0 forward: K=2 WRITE passes + 2 STORE key
    # compressions first (all identical in both runs -- hooks are read-pass
    # only), then the READ pass.
    assert len(captures_on) == len(captures_off) == 5
    diff = (captures_on[-1] - captures_off[-1]).abs()      # == the injected bias * site mask
    assert float(diff[:, pad_pos, :].abs().max()) == 0.0, "bias leaked into pad positions"
    assert float(diff[:, real_pos, :].abs().max()) > 0.0, "no bias at real read sites"


def test_jvp_padding_site_mask_aligned():
    """The %32 jvp read padding must extend site_mask with the sequence, and
    the zeroed pad sites must leave the real-position logits unchanged
    (before the fix the hooks' mu * site_mask shape-mismatched at the first
    instrumented block as soon as T % 32 != 0)."""
    model = _make_model()
    state = _random_state(model, B=1, c=2, seed=5)
    batch, labels = _inputs()
    qry = batch["query_input_ids"]
    qry_emb = model.model.get_input_embeddings()(qry)     # T = 5 -> pad branch fires

    with torch.no_grad():
        logits_eager = model._linear_state_read_logits(qry_emb, qry, labels, None, None, state)
        model.attn_implementation = "jvp_flash"           # simulate the padding branch only
        logits_jvp = model._linear_state_read_logits(qry_emb, qry, labels, None, None, state)
    assert logits_jvp.shape == logits_eager.shape
    assert torch.allclose(logits_eager, logits_jvp, atol=1e-5), \
        (logits_eager - logits_jvp).abs().max()


# --------------------------------------------------------------------- #
# P3 — exact-softmax ablation arm
# --------------------------------------------------------------------- #

def test_exact_softmax_arm():
    """exact=True: mu == manual per-token softmax over the unfolded bank (stored items only)."""
    model = _make_model(exact=True)
    state = _random_state(model, B=2, c=2, mask_one=True, seed=11)
    g = torch.Generator().manual_seed(13)
    x = torch.randn(2, 4, model.model.config.n_embd, generator=g)
    mu = model._linear_state_read_bias(0, x, dict(state))

    H, d_h = model.read_state_heads, model.read_state_head_dim
    h = model.read_state_ln[0](x)
    q = model.read_state_q_proj[0](h).view(2, 4, H, d_h)
    q_hat = torch.nn.functional.normalize(q, dim=-1)
    q_s = q_hat * model.read_state_beta[0].view(1, 1, H, 1)
    scores = torch.einsum('bthd,bhnd->bthn', q_s, state['K'])
    scores = scores.masked_fill(~state['item_mask'][:, None, None, :], float('-inf'))
    w = torch.softmax(scores, dim=-1)
    mu_read = torch.einsum('bthn,bhnd->bthd', w, state['V'])
    mu_ref = model.read_state_out_proj(mu_read.reshape(2, 4, -1)) * model.read_state_gamma[0]

    assert torch.allclose(mu, mu_ref, atol=1e-5)
    # softmax weights are convex per token: they sum to 1 over stored items
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, H), atol=1e-5)

    # sanity: the superposed state arm is a DIFFERENT read (not the same numbers)
    model_state_arm = _make_model(exact=False)
    model_state_arm.load_state_dict(model.state_dict())   # identical read params
    state_arm = _random_state(model_state_arm, B=2, c=2, mask_one=True, seed=11)
    mu_state = model_state_arm._linear_state_read_bias(0, x, dict(state_arm))
    assert not torch.allclose(mu, mu_state, atol=1e-3), "exact arm collapsed into the state arm"


# --------------------------------------------------------------------- #
# P4 — interface mode (two-pass)
# --------------------------------------------------------------------- #

def _iface_state(model):
    return _random_state(model, B=1, c=2, mask_one=False, seed=5)


def _iface_call(model, qry, labels, perturb=None):
    """_linear_state_read_logits on a single query; returns (logits, iface embeddings)."""
    qry_emb = model.model.get_input_embeddings()(qry)
    state = _iface_state(model)
    rs = re_ = None                                       # n_ctrl_tokens = 0 in these tests
    logits, iface = model._linear_state_read_logits(
        qry_emb, qry, labels, rs, re_, state, return_interface=True)
    return logits, iface


def test_interface_blinding():
    """Perturbing target-span tokens in the pass-1 input leaves the interface embeddings
    unchanged (labels >= 0 positions are masked out of the conditioning attention)."""
    model = _make_model(read_sites="interface")
    qry = torch.tensor([[BOS_ID, 66, 40, 68, 69]])
    labels = torch.full((1, 5), -100, dtype=torch.long)
    labels[:, 3:5] = qry[:, 3:5]                          # target span = positions 3, 4
    _, iface_a = _iface_call(model, qry, labels)

    qry_pert = qry.clone()
    qry_pert[0, 3] = 12                                   # different token INSIDE the span
    qry_pert[0, 4] = 13
    with torch.no_grad():
        _, iface_b = _iface_call(model, qry_pert, labels)
    assert torch.equal(iface_a, iface_b), "pass-1 interface embeddings saw the blinded target span"


def test_interface_query_sensitivity():
    """Perturbing NON-target query tokens changes the interface embeddings — the guard
    against a query-blind (left-prefix) implementation of the interface reads."""
    model = _make_model(read_sites="interface")
    qry = torch.tensor([[BOS_ID, 66, 40, 68, 69]])
    labels = torch.full((1, 5), -100, dtype=torch.long)
    labels[:, 3:5] = qry[:, 3:5]
    with torch.no_grad():
        _, iface_a = _iface_call(model, qry, labels)
        qry_pert = qry.clone()
        qry_pert[0, 1] = 55                               # different token OUTSIDE the span
        _, iface_b = _iface_call(model, qry_pert, labels)
    assert (iface_a - iface_b).abs().max() > 1e-4, "interface reads are query-blind"


def test_interface_read_sites():
    """Bias only at the interface positions in BOTH passes (spy-diff on the injecting block)."""
    model = _make_model(read_sites="interface", n_ctrl_tokens=1)
    inputs, _ = _inputs()
    M = model.n_mem_tokens
    C = model.n_ctrl_tokens

    captures_on, captures_off = [], []
    block0 = _gmm.get_backbone(model.model).h[0]

    def spy(store):
        def _spy(module, args, kwargs, output):
            store.append(output[0].detach().clone())
            return None
        return _spy

    h = block0.register_forward_hook(spy(captures_on), with_kwargs=True)
    with torch.no_grad():
        model.forward(inputs)
    h.remove()
    model._read_state_hooks_disabled = True
    h = block0.register_forward_hook(spy(captures_off), with_kwargs=True)
    with torch.no_grad():
        model.forward(inputs)
    h.remove()
    model._read_state_hooks_disabled = False

    assert len(captures_on) == len(captures_off) == 6     # 2 WRITE + 2 STORE + 2 read passes
    # pass 1 = second-to-last capture, pass 2 = last (WRITE passes come first)
    t1 = captures_on[-2].size(1)                          # pass-1 length (unpadded, eager attn)
    d1 = (captures_on[-2] - captures_off[-2]).abs()
    d2 = (captures_on[-1] - captures_off[-1]).abs()
    # pass 1: sites are the LAST M positions (M_0); everything before untouched
    assert float(d1[:, :t1 - M, :].abs().max()) == 0.0
    assert float(d1[:, t1 - M:, :].abs().max()) > 0.0
    # pass 2: sites are exactly [C, C+M). Positions BEFORE the interface tokens
    # are exactly untouched (causal: they cannot see the sites); positions
    # AFTER them change legitimately (they attend to the biased interface
    # states) -- that is the answering path, not a leak.
    t2 = captures_on[-1].size(1)
    assert float(d2[:, :C, :].abs().max()) == 0.0
    assert float(d2[:, C:C + M, :].abs().max()) > 0.0


# --------------------------------------------------------------------- #
# P5 — eval mirrors: forward_per_segment_eval in the new mode
# --------------------------------------------------------------------- #

def test_forward_per_segment_eval_linear_read():
    """Probing at boundary k == a forward that wrote only the first k segments.

    segment_size=16 (hopfield_n_segments=1 + segment_size -> ceil-division
    chunking, so the truncated context re-segments identically): the
    per-segment WRITE inputs are the same for the full (64-token, 4 segments)
    and truncated (32-token, 2 segments) contexts, so the stored bank as of
    boundary k must match -- and with it the probe-read logits and the
    exact-match flags.
    """
    model = _make_model(n_segments=1, hopfield_segment_size=16)
    B = 1
    ctx = (torch.arange(1, 65).view(B, 64) % 67) + 2      # 64 real tokens, no pads
    # two probes: one pair "written" in segment 0, one in segment 2
    q0 = torch.tensor([[BOS_ID, 10, 11, 20, 21]])         # ?!K:V!| shape, ids arbitrary
    q2 = torch.tensor([[BOS_ID, 30, 31, 40, 41]])
    qids = torch.stack([q0, q2], dim=1)                   # [B, 2, 5]
    tmask = torch.zeros(B, 2, 5, dtype=torch.bool)
    tmask[:, :, 3:5] = True
    seg_idx = torch.tensor([[0, 2]])
    qmask = torch.ones(B, 2, dtype=torch.bool)
    kv_queries = {'query_input_ids': qids, 'target_mask': tmask,
                  'seg_idx': seg_idx, 'mask': qmask, 'ignore_token_ids': []}

    # ---- stage-level check: bank + probe logits at boundary 2 (k=2 segments) ----
    fwd_full = {"context_input_ids": ctx, "query_input_ids": ctx[:, :1]}
    out_full = model.forward(fwd_full, labels=None, collect_segment_mems=True)
    k = 2
    ctx_trunc = ctx[:, :k * 16]
    fwd_trunc = {"context_input_ids": ctx_trunc, "query_input_ids": ctx_trunc[:, :1]}
    out_trunc = model.forward(fwd_trunc, labels=None, collect_segment_mems=True)

    vals_full = torch.stack(out_full['stored_values'][:k], dim=1)
    vals_trunc = torch.stack(out_trunc['stored_values'][:k], dim=1)
    masks_full = torch.stack(out_full['stored_masks'][:k], dim=1)
    masks_trunc = torch.stack(out_trunc['stored_masks'][:k], dim=1)
    assert torch.allclose(vals_full, vals_trunc, atol=1e-6), \
        (vals_full - vals_trunc).abs().max()
    assert torch.equal(masks_full, masks_trunc)

    state_full = model._build_linear_state(vals_full, masks_full.bool())
    state_trunc = model._build_linear_state(vals_trunc, masks_trunc.bool())
    rs = re_ = None
    logits_full = model._read_once_linear(state_full, qids, rs, re_, target_mask=tmask)
    logits_trunc = model._read_once_linear(state_trunc, qids, rs, re_, target_mask=tmask)
    assert torch.allclose(logits_full, logits_trunc, atol=1e-4), \
        (logits_full - logits_trunc).abs().max()
    em_full = model._probe_exact_match(logits_full[0, 0:1], qids[0, 0:1], tmask[0, 0:1], [])
    em_trunc = model._probe_exact_match(logits_trunc[0, 0:1], qids[0, 0:1], tmask[0, 0:1], [])
    assert bool(em_full[0]) == bool(em_trunc[0])

    # ---- end-to-end: forward_per_segment_eval flags match the truncated run ----
    src_f, probe_f, em_f = model.forward_per_segment_eval(
        {"context_input_ids": ctx}, kv_queries)
    src_t, probe_t, em_t = model.forward_per_segment_eval(
        {"context_input_ids": ctx_trunc}, kv_queries)

    def triples(src, probe, em, wanted_probe):
        sel = probe == wanted_probe
        return {int(s): bool(e) for s, e in zip(src[sel].tolist(), em[sel].tolist())}

    for boundary in range(4):
        tf = triples(src_f, probe_f, em_f, boundary)
        tt = triples(src_t, probe_t, em_t, boundary)
        if boundary < 2:
            # boundaries the truncated run also reached must agree exactly
            assert tf.keys() == tt.keys(), (boundary, tf, tt)
            for s in tf:
                assert tf[s] == tt[s], (boundary, s, tf[s], tt[s])
        else:
            assert tt == {}, "truncated run probed segments it never wrote"


# --------------------------------------------------------------------- #
# Write-side untouched / gradient semantics
# --------------------------------------------------------------------- #

def test_write_side_untouched_and_read_grads_flow():
    """Stored values bitwise identical with hooks on vs off; read params get
    outer-loss gradients only (grad_mode='second' inner loop succeeds unchanged)."""
    model = _make_model(grad_mode="none")
    inputs, _ = _inputs()
    fwd = {"context_input_ids": inputs["context_input_ids"],
           "query_input_ids": inputs["context_input_ids"][:, :1]}
    with torch.no_grad():
        out_hooks_on = model.forward(fwd, labels=None, collect_segment_mems=True)
        model._read_state_hooks_disabled = True
        out_hooks_off = model.forward(fwd, labels=None, collect_segment_mems=True)
        model._read_state_hooks_disabled = False
    for a, b in zip(out_hooks_on['stored_values'], out_hooks_off['stored_values']):
        assert torch.equal(a, b), "the READ hooks leaked into the WRITE path"
    for a, b in zip(out_hooks_on['stored_masks'], out_hooks_off['stored_masks']):
        assert torch.equal(a, b)
    assert all(k is None for k in out_hooks_on['stored_keys'])

    # outer loss reaches every read module; the second-order inner loop succeeds unchanged
    model2 = _make_model(grad_mode="second", seed=3)
    _, labels = _inputs()
    out = model2.forward(inputs, labels=labels)
    out['loss'].backward()
    for name, p in [("gamma", model2.read_state_gamma[0]), ("beta", model2.read_state_beta[0]),
                    ("q_proj", model2.read_state_q_proj[0].weight),
                    ("ln", model2.read_state_ln[0].weight),
                    ("out_proj", model2.read_state_out_proj.weight),
                    ("key_proj", model2.read_state_key_proj.weight),
                    ("value_proj", model2.read_state_value_proj.weight)]:
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, f"no outer grad on {name}"
    # ...and the inner loop never targets a read param (by construction its
    # autograd.grad only takes mem_batch / per-sample proj tensors)
    assert float(out['inner_loop_stats']['mem_norm_mean']) > 0


def test_collect_segment_mems_purity_linear_mode():
    """collect_segment_mems=True leaves predictions bitwise unchanged (linear mode);
    snapshots stay aligned (None for empty segments)."""
    model = _make_model()
    inputs, _ = _inputs()
    with torch.no_grad():
        plain = model.forward(inputs)
        coll = model.forward(inputs, collect_segment_mems=True)
    assert torch.equal(plain['predictions'], coll['predictions'])
    snaps = coll['segment_mems']
    assert len(snaps) == 2
    assert all(s is not None for s in snaps)
    assert snaps[-1]['n_stored'] == 2 and snaps[0]['n_stored'] == 1


def test_read_state_layers_subset():
    """read_state_layers=[1] instruments only block 1: one per-layer set, grads flow to it."""
    model = _make_model(read_state_layers=[1])
    assert len(model.read_state_gamma) == 1
    block0, block1 = _gmm.get_backbone(model.model).h[0], _gmm.get_backbone(model.model).h[1]
    assert not block0._forward_hooks_with_kwargs and block1._forward_hooks_with_kwargs
    inputs, labels = _inputs()
    out = model.forward(inputs, labels=labels)
    assert out['loss'].isfinite()
    out['loss'].backward()
    assert float(model.read_state_gamma[0].grad.abs().sum()) > 0


# --------------------------------------------------------------------- #
# Bank-prefix control arm (ladder arm (b), use_bank_prefix_read)
# --------------------------------------------------------------------- #

def _make_bank_prefix_model(seed=0, **kw):
    kw.setdefault("hopfield_n_segments", 2)
    torch.manual_seed(seed)
    cfg = ModelConfig(
        base_config=_base_config(), n_mem_tokens=2, K=2, lr=0.05,
        grad_mode="none",
        use_hopfield_memory=True,
        use_bank_prefix_read=True,
        **kw,
    )
    model = Model(cfg)
    model.eval()
    return model


def test_bank_prefix_read_matches_manual_prefix_forward():
    """The bank-prefix READ is an ordinary causal forward over
    [bank, query]: predictions equal that manual construction."""
    model = _make_bank_prefix_model()
    inputs, _ = _inputs()
    with torch.no_grad():
        out = model.forward(inputs)

    # manual reconstruction: run the WRITE exactly as forward does, then a
    # plain read forward over [bank, query]
    fwd = {"context_input_ids": inputs["context_input_ids"],
           "query_input_ids": inputs["context_input_ids"][:, :1]}
    with torch.no_grad():
        full = model.forward(fwd, labels=None, collect_segment_mems=True)
    values = torch.stack(full['stored_values'], dim=1)            # [1, 2, M*d]
    masks = torch.stack(full['stored_masks'], dim=1).bool()
    P = model.n_mem_tokens * model.model.config.n_embd
    bank = values.view(1, 2, model.n_mem_tokens, P // model.n_mem_tokens).reshape(1, -1, P // model.n_mem_tokens)
    pmask = masks.unsqueeze(-1).expand(1, 2, model.n_mem_tokens).reshape(1, -1).long()
    qry = inputs["query_input_ids"]
    qry_emb = model.model.get_input_embeddings()(qry)
    attn = torch.cat([pmask, (qry != PAD_ID).long()], dim=1)
    x = torch.cat([bank, qry_emb], dim=1)
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = model.model(inputs_embeds=x, attention_mask=attn,
                             position_ids=position_ids, return_dict=True).logits
    manual = Model._slice_read_logits(logits, model.n_mem_tokens * 2, qry.size(1))
    assert torch.allclose(out['predictions'], manual, atol=1e-5), \
        (out['predictions'] - manual).abs().max()
    assert all(k is None for k in full['stored_keys'])


def test_bank_prefix_outer_grad_reaches_stored_values():
    """The control arm's point: the outer loss backpropagates through plain
    attention INTO the stored values (the write-shaping signal the one-shot
    mode delivers only through its single mixture). grad_mode='first' keeps
    the inner graph connected to self.mem ('none' detaches it by design)."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        base_config=_base_config(), n_mem_tokens=2, K=1, lr=0.05,
        grad_mode="first", use_hopfield_memory=True, hopfield_n_segments=2,
        use_bank_prefix_read=True)
    model = Model(cfg)
    model.train()
    inputs, labels = _inputs()
    out = model.forward(inputs, labels=labels)
    out['loss'].backward()
    assert model.mem.grad is not None and float(model.mem.grad.abs().sum()) > 0, \
        "no outer gradient reached self.mem through the bank prefix"


def test_bank_prefix_per_segment_eval():
    """forward_per_segment_eval works in the bank-prefix mode: the boundary-k
    probe equals a forward that wrote only the first k segments (EM flags)."""
    model = _make_bank_prefix_model(hopfield_segment_size=16, hopfield_n_segments=1)
    B = 1
    ctx = (torch.arange(1, 65).view(B, 64) % 67) + 2
    q0 = torch.tensor([[BOS_ID, 10, 11, 20, 21]])
    q2 = torch.tensor([[BOS_ID, 30, 31, 40, 41]])
    qids = torch.stack([q0, q2], dim=1)
    tmask = torch.zeros(B, 2, 5, dtype=torch.bool)
    tmask[:, :, 3:5] = True
    kv_queries = {'query_input_ids': qids, 'target_mask': tmask,
                  'seg_idx': torch.tensor([[0, 2]]),
                  'mask': torch.ones(B, 2, dtype=torch.bool),
                  'ignore_token_ids': []}
    src_f, probe_f, em_f = model.forward_per_segment_eval({"context_input_ids": ctx}, kv_queries)
    ctx_trunc = ctx[:, :32]
    src_t, probe_t, em_t = model.forward_per_segment_eval({"context_input_ids": ctx_trunc}, kv_queries)

    def triples(src, probe, em, boundary):
        sel = probe == boundary
        return {int(s): bool(e) for s, e in zip(src[sel].tolist(), em[sel].tolist())}

    assert probe_f.max() == 3 and probe_t.max() == 1
    for boundary in (0, 1):
        tf, tt = triples(src_f, probe_f, em_f, boundary), triples(src_t, probe_t, em_t, boundary)
        assert tf.keys() == tt.keys(), (boundary, tf, tt)
        for s in tf:
            assert tf[s] == tt[s], (boundary, s, tf[s], tt[s])


# --------------------------------------------------------------------- #
# Config guards
# --------------------------------------------------------------------- #

def test_config_guards():
    base = _base_config()
    common = dict(base_config=base, n_mem_tokens=2, K=1, grad_mode="none")

    with _expect_raises(AssertionError):
        ModelConfig(**common, use_linear_state_read=True)  # requires use_hopfield_memory
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_gated_delta_memory=True,
                    use_linear_state_read=True)            # mutually exclusive with gated delta
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    hopfield_value_proj_dim=64)            # raw tokens only (v1)
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    read_state_read_sites="bogus")
    with _expect_raises(AssertionError):
        ModelConfig(**common, read_state_exact_softmax=True)  # requires use_linear_state_read
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    read_state_heads=0)
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    read_state_gate_init=0.0)
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    read_state_layers="bad")
    with _expect_raises(NotImplementedError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    read_state_segment_emb=True)
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    use_gradient_checkpointing=True)
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_bank_prefix_read=True)       # requires use_hopfield_memory
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_linear_state_read=True,
                    use_bank_prefix_read=True)                 # mutually exclusive read modes
    with _expect_raises(AssertionError):
        ModelConfig(**common, use_hopfield_memory=True, use_bank_prefix_read=True,
                    hopfield_value_proj_dim=64)                # raw prefix tokens only
    # out-of-range layer index fails at model construction (needs the backbone)
    with _expect_raises(ValueError):
        _make_model(read_state_layers=[5])
    # valid subset constructs
    _make_model(read_state_layers=[0, 1])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"{fn.__name__}: OK")
    print(f"\n{len(fns)} tests passed.")
