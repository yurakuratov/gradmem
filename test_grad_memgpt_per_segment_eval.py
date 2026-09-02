"""Per-segment forgetting-eval tests (grad_memgpt.py).

Ports the adaptive fork's forgetting matrix (forward_per_segment_eval +
collect_segment_mems) to the main grad_memgpt model. These tests lock in:

  A. purity: collect_segment_mems only snapshots detached state -- eval-mode
     predictions and train-mode loss are bitwise identical with the flag on
     vs off.
  B. snapshot structure: segment_mems is indexed by the TRUE segment index
     (None placeholders keep empty segments aligned); Hopfield snapshots
     carry a monotonically growing store count; Gated Delta snapshots carry
     S and the per-sample written flag.
  C. forward_per_segment_eval returns the flat (src_seg, probe_seg, em)
     triple with the same invariants as the adaptive fork: src <= probe, each
     probe fires at every boundary >= its source segment, dtypes long/long/
     bool. Exercised for all three memory modes: plain (per-segment RESET),
     Hopfield (accumulating store, query-dependent retrieval) and Gated
     Delta (accumulating state matrix).
  D. guards: K=0 / all-pad context / empty probe mask -> empty triple.
  E. collator attribution: build_kv_probe_queries with seg_pad_side='left'
     shifts the KV -> segment attribution by grad_memgpt.py's extra LEFT-pad
     (seg_sz*n_seg - padded_len); 'right' (adaptive) keeps it unshifted.

Run two ways:
    python test_grad_memgpt_per_segment_eval.py     # self-contained runner
    pytest test_grad_memgpt_per_segment_eval.py     # if pytest is installed
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


_gmm = _load_module("gmm_per_seg", "grad_memgpt.py")
Model, ModelConfig = _gmm.GradMemGPT, _gmm.GradMemGPTConfig

_runner = _load_module("runner_per_seg", "run_gradmemgpt_on_kv_retrieval.py") \
    if os.path.exists(os.path.join(_HERE, "run_gradmemgpt_on_kv_retrieval.py")) else None

PAD_ID, BOS_ID = 0, 1


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


def _make_model(K=2, n_mem_tokens=4, **kw):
    torch.manual_seed(0)
    cfg = ModelConfig(
        base_config=_base_config(), n_mem_tokens=n_mem_tokens, K=K, lr=0.05,
        grad_mode="none",                                # cheapest for structural tests
        **kw,
    )
    model = Model(cfg)
    model.eval()
    return model


def _inputs(context=None):
    """Right-padded context (len 8 -> 2 segments of 4) + query + labels."""
    if context is None:
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


def _check_triples(model, inputs, kv_q, n_seg_expected):
    """Structural invariants shared with the adaptive fork's eval."""
    src, probe, em = model.forward_per_segment_eval(inputs, kv_q)
    assert src.numel() == probe.numel() == em.numel() > 0
    assert src.dtype == torch.long and probe.dtype == torch.long and em.dtype == torch.bool
    assert bool((src <= probe).all()), "KV must only be probed at boundaries >= its segment"
    assert int(probe.max()) == n_seg_expected - 1
    assert int(src.min()) >= 0
    # each probe fires exactly once per boundary >= its source segment
    for s in range(n_seg_expected):
        n_probes_s = int((kv_q['seg_idx'][kv_q['mask']] == s).sum())
        n_fired = int((src == s).sum())
        assert n_fired == n_probes_s * (n_seg_expected - s), \
            f"src seg {s}: fired {n_fired}, expected {n_probes_s * (n_seg_expected - s)}"
    return src, probe, em


# ---------------------------------------------------------------- #
# A + B: purity and snapshot structure
# ---------------------------------------------------------------- #

def test_collect_segment_mems_purity_plain():
    """Plain (per-segment RESET) model: snapshots must not perturb the forward."""
    model = _make_model(hopfield_n_segments=2)
    inputs, labels = _inputs()
    with torch.no_grad():
        out_plain = model.forward(inputs)
        out_coll = model.forward(inputs, collect_segment_mems=True)
    assert _max_diff(out_plain['predictions'], out_coll['predictions']) == 0.0

    snaps = out_coll['segment_mems']
    assert len(snaps) == 2
    B, M, d = 2, model.n_mem_tokens, model.model.config.n_embd
    for i, snap in enumerate(snaps):
        assert snap is not None, "both segments are non-empty here"
        assert snap['mem'].shape == (B, M, d)
        assert snap['n_stored'] == 0 and snap['S'] is None and snap['written'] is None
    # plain mode RESETS mem per segment -> the two snapshots differ
    assert _max_diff(snaps[0]['mem'], snaps[1]['mem']) > 0.0
    # train-mode loss is bitwise unchanged too
    model.train()
    torch.manual_seed(1)
    loss_a = model.forward(inputs, labels=labels)['loss']
    torch.manual_seed(1)
    loss_b = model.forward(inputs, labels=labels, collect_segment_mems=True)['loss']
    assert _max_diff(loss_a, loss_b) == 0.0
    print("  [purity]  plain: predictions/loss bitwise identical, 2 snapshots")


def test_collect_segment_mems_purity_hopfield():
    """Hopfield model: purity + store-count snapshots + final store lists."""
    model = _make_model(use_hopfield_memory=True, hopfield_n_segments=2,
                        hopfield_retrieval_mode="softmax")
    inputs, labels = _inputs()
    with torch.no_grad():
        out_plain = model.forward(inputs)
        out_coll = model.forward(inputs, collect_segment_mems=True)
    assert _max_diff(out_plain['predictions'], out_coll['predictions']) == 0.0

    snaps = out_coll['segment_mems']
    assert len(snaps) == 2
    counts = [snap['n_stored'] for snap in snaps]
    assert counts == [1, 2], f"store count must grow per boundary, got {counts}"
    assert len(out_coll['stored_keys']) == 2 and len(out_coll['stored_values']) == 2 \
        and len(out_coll['stored_masks']) == 2
    B = 2
    assert out_coll['stored_keys'][0].shape == (B, model.n_mem_tokens * model.model.config.n_embd)
    assert out_coll['stored_masks'][0].shape == (B,)
    print("  [purity]  hopfield: predictions bitwise identical, n_stored=[1, 2]")


def test_collect_segment_mems_empty_segment_alignment():
    """A segment that is empty for ALL samples stays None (index alignment).

    Both samples have 6 real tokens (+2 right pads); with hopfield_n_segments=4
    the chunk size is 2, so segment 3 is all-pad -> None placeholder, and the
    eval falls back to the last non-None snapshot there.
    """
    context = torch.tensor([
        [BOS_ID, 66, 40, 68, 51, 66, 0, 0],
        [BOS_ID, 62, 64, 66, 40, 68, 0, 0],
    ])
    model = _make_model(use_hopfield_memory=True, hopfield_n_segments=4,
                        hopfield_retrieval_mode="softmax")
    inputs, _ = _inputs(context=context)
    with torch.no_grad():
        out = model.forward(inputs, collect_segment_mems=True)
    snaps = out['segment_mems']
    assert len(snaps) == 4
    assert snaps[3] is None, "segment 3 is all-pad for every sample"
    assert all(s is not None for s in snaps[:3])
    # eval falls back to snapshot 2 at boundary 3 (state unchanged) -> the
    # seg-1 KVs are probed at 3 boundaries, seg-0 KVs at 4
    kv_q = _kv_queries()
    kv_q['seg_idx'][:, 1] = 1          # second probe lives in segment 1
    src, probe, em = model.forward_per_segment_eval(inputs, kv_q)
    assert int(probe.max()) == 3
    assert int((probe[src == 1] == 1).sum() + (probe[src == 1] == 2).sum()
               + (probe[src == 1] == 3).sum()) == int((src == 1).sum())
    print("  [align ]  empty segment -> None placeholder, fallback probes at boundary 3")


def test_collect_segment_mems_gated_delta():
    model = _make_model(use_gated_delta_memory=True, hopfield_segment_size=4,
                        gated_delta_state_dim=16)
    inputs, _ = _inputs()
    with torch.no_grad():
        out_plain = model.forward(inputs)
        out_coll = model.forward(inputs, collect_segment_mems=True)
    assert _max_diff(out_plain['predictions'], out_coll['predictions']) == 0.0
    snaps = out_coll['segment_mems']
    assert len(snaps) == 2
    B, d = 2, model.gated_delta_state_dim
    for snap in snaps:
        assert snap['S'].shape == (B, d, d)
        assert snap['written'].dtype == torch.bool and snap['written'].shape == (B,)
    assert bool(snap['written'].all())
    print("  [gd    ]  gated delta: S snapshots [B,d,d], written flags set")


# ---------------------------------------------------------------- #
# C: forward_per_segment_eval structural invariants, all memory modes
# ---------------------------------------------------------------- #

def test_forward_per_segment_eval_plain():
    model = _make_model(hopfield_n_segments=2)
    inputs, _ = _inputs()
    _check_triples(model, inputs, _kv_queries(), n_seg_expected=2)
    print("  [eval  ]  plain mode: triple invariants hold (2 segments)")


def test_forward_per_segment_eval_hopfield():
    model = _make_model(use_hopfield_memory=True, hopfield_n_segments=2,
                        hopfield_retrieval_mode="softmax")
    inputs, _ = _inputs()
    _check_triples(model, inputs, _kv_queries(), n_seg_expected=2)
    print("  [eval  ]  hopfield mode: triple invariants hold (2 segments)")


def test_forward_per_segment_eval_hopfield_segment_size():
    """hopfield_segment_size path: n_seg = ceil(padded_len / segment_size)."""
    model = _make_model(use_hopfield_memory=True, hopfield_segment_size=3,
                        hopfield_retrieval_mode="softmax")
    inputs, _ = _inputs()
    # padded len 8, segment_size 3 -> ceil(8/3) = 3 segments (extra left-pad 1)
    _check_triples(model, inputs, _kv_queries(), n_seg_expected=3)
    print("  [eval  ]  hopfield segment_size: 3 segments (left-padded)")


def test_forward_per_segment_eval_gated_delta():
    model = _make_model(use_gated_delta_memory=True, hopfield_segment_size=4,
                        gated_delta_state_dim=16)
    inputs, _ = _inputs()
    _check_triples(model, inputs, _kv_queries(), n_seg_expected=2)
    print("  [eval  ]  gated delta mode: triple invariants hold (2 segments)")


def test_forward_per_segment_eval_ctrl_tokens():
    """n_ctrl_tokens > 0 exercises the read_st/read_end plumbing."""
    model = _make_model(n_ctrl_tokens=1, use_hopfield_memory=True,
                        hopfield_n_segments=2, hopfield_retrieval_mode="softmax")
    inputs, _ = _inputs()
    _check_triples(model, inputs, _kv_queries(), n_seg_expected=2)
    print("  [eval  ]  ctrl tokens: triple invariants hold")


# ---------------------------------------------------------------- #
# D: guards
# ---------------------------------------------------------------- #

def test_eval_guards():
    model = _make_model(use_hopfield_memory=True, hopfield_n_segments=2)

    # K = 0 -> no WRITE runs
    inputs, _ = _inputs()
    model.K = 0
    src, probe, em = model.forward_per_segment_eval(inputs, _kv_queries())
    assert src.numel() == probe.numel() == em.numel() == 0
    model.K = 2

    # all-pad context -> WRITE skipped
    pad_inputs = {"context_input_ids": torch.full((2, 8), PAD_ID, dtype=torch.long),
                  "query_input_ids": inputs["query_input_ids"]}
    src, probe, em = model.forward_per_segment_eval(pad_inputs, _kv_queries())
    assert src.numel() == 0

    # empty probe mask
    kv_q = _kv_queries()
    kv_q['mask'][:] = False
    src, probe, em = model.forward_per_segment_eval(inputs, kv_q)
    assert src.numel() == 0
    print("  [guard ]  K=0 / all-pad context / empty qmask -> empty triples")


# ---------------------------------------------------------------- #
# E: collator attribution (seg_pad_side)
# ---------------------------------------------------------------- #

def test_build_kv_probe_queries_pad_side():
    if _runner is None:
        print("  [skip  ]  runner module not importable; skipping collator test")
        return
    tok_path = os.path.join(_HERE, 'tokenizers', 'kv_alphabet_62')
    if not os.path.isdir(tok_path):
        print("  [skip  ]  kv_alphabet_62 tokenizer not found; skipping collator test")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)

    batch = [{'context': '!a:b!|'}]           # 6 chars; +BOS = 7 -> padded to 8
    # segment_size=6 -> n_seg = ceil(8/6) = 2, seg_sz = 6.
    # KV closing '!' at char 4 (te=5):
    #   right (adaptive): 4 // 6 = 0
    #   left  (grad_memgpt.py): extra left-pad = 6*2 - 8 = 4 -> (4+4)//6 = 1
    kv_r = _runner.build_kv_probe_queries(batch, tok, padded_len=8,
                                          segment_size=6, seg_pad_side='right')
    assert int(kv_r['seg_idx'][0, 0]) == 0
    kv_l = _runner.build_kv_probe_queries(batch, tok, padded_len=8,
                                          segment_size=6, seg_pad_side='left')
    assert int(kv_l['seg_idx'][0, 0]) == 1
    # default stays 'right' (adaptive behaviour unchanged)
    kv_d = _runner.build_kv_probe_queries(batch, tok, padded_len=8, segment_size=6)
    assert int(kv_d['seg_idx'][0, 0]) == 0
    print("  [collat]  seg_pad_side shifts attribution by the left-pad amount")


if __name__ == "__main__":
    test_collect_segment_mems_purity_plain()
    test_collect_segment_mems_purity_hopfield()
    test_collect_segment_mems_empty_segment_alignment()
    test_collect_segment_mems_gated_delta()
    test_forward_per_segment_eval_plain()
    test_forward_per_segment_eval_hopfield()
    test_forward_per_segment_eval_hopfield_segment_size()
    test_forward_per_segment_eval_gated_delta()
    test_forward_per_segment_eval_ctrl_tokens()
    test_eval_guards()
    test_build_kv_probe_queries_pad_side()
    print("ALL per-segment eval tests passed")
