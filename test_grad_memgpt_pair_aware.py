"""Pair-aware segmentation tests (grad_memgpt.py).

The pair-aware mode chunks the WRITE context at !K:V! pair boundaries instead
of fixed-width token windows, so a pair's key and value can never be split
across segments (the straddling pathology measured on N8-K2V2: with 4x16-token
windows, pair p4's key lands in segment 1 and its value in segment 2, making
it unretrievable from any single Hopfield slot -> the always-worst forgetting
matrix row 2). These tests lock in:

  A. pair detection from token ids (_detect_pair_ends), incl. pad handling and
     the trailing-'|' extension to the last segment.
  B. the balanced division (_divide_pairs) shared verbatim with the collator.
  C. span construction (_pair_segment_spans) incl. variable pair counts in one
     batch (per-sample equal division, empty tail groups).
  D. THE core invariant: the collator's seg_idx (build_kv_probe_queries with
     pair-aware params) attributes every pair to a model segment that contains
     the pair's ENTIRE token span (no straddling, by construction).
  E. forward + forward_per_segment_eval run under the mode: bitwise purity of
     collect_segment_mems, forgetting-matrix triple invariants, empty-segment
     None-fallback for short samples.
  F. config validation guards.

Run two ways:
    python test_grad_memgpt_pair_aware.py      # self-contained runner
    pytest test_grad_memgpt_pair_aware.py      # if pytest is installed
"""

import importlib.util
import os
import re
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


_gmm = _load_module("gmm_pair_aware", "grad_memgpt.py")
Model, ModelConfig = _gmm.GradMemGPT, _gmm.GradMemGPTConfig

_runner = _load_module("runner_pair_aware", "run_gradmemgpt_on_kv_retrieval.py")

PAIR_RE = re.compile(r'!([^!|:]+):([^!|]+)!')
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _kv_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(_HERE, 'tokenizers', 'kv_alphabet_62'))


def _ctx(n_pairs, offset=0):
    """Dataset-format context: n_pairs * '!K:V!' + trailing '|' (7 tokens/pair)."""
    parts = []
    for i in range(n_pairs):
        k = ALPHABET[(offset + i) % 62] + ALPHABET[(offset + i + 17) % 62]
        v = ALPHABET[(offset + i + 31) % 62] + ALPHABET[(offset + i + 47) % 62]
        parts.append(f'!{k}:{v}!')
    return ''.join(parts) + '|'


def _base_config(vocab=70):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.vocab_size = vocab
    cfg.pad_token_id = 0
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg.use_cache = False
    cfg.resid_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    return cfg


def _delims(tok):
    return [tok.convert_tokens_to_ids('!'), tok.convert_tokens_to_ids(':')]


def _make_model(tok, K=2, n_mem_tokens=4, pair_aware=True, pairs_per_segment=None,
                hopfield_n_segments=2, **kw):
    torch.manual_seed(0)
    cfg = ModelConfig(
        base_config=_base_config(), n_mem_tokens=n_mem_tokens, K=K, lr=0.05,
        grad_mode="none", hopfield_n_segments=hopfield_n_segments,
        pair_aware_segmentation=pair_aware,
        pairs_per_segment=pairs_per_segment,
        pair_delim_token_ids=_delims(tok) if pair_aware else None,
        **kw,
    )
    model = Model(cfg)
    model.eval()
    return model


def _ids_for(tok, contexts):
    """Collator-style right-padded context ids (pad to multiple of 8)."""
    enc = tok(contexts, add_special_tokens=True, padding=True, pad_to_multiple_of=8)
    return torch.tensor(enc['input_ids'], dtype=torch.long)


def _check_triples(model, ctx_ids, kv_q, n_seg_expected):
    src, probe, em = model.forward_per_segment_eval({'context_input_ids': ctx_ids}, kv_q)
    assert src.numel() == probe.numel() == em.numel() > 0
    assert src.dtype == torch.long and probe.dtype == torch.long and em.dtype == torch.bool
    assert bool((src <= probe).all())
    assert int(probe.max()) == n_seg_expected - 1
    for s in range(n_seg_expected):
        n_probes_s = int((kv_q['seg_idx'][kv_q['mask']] == s).sum())
        assert int((src == s).sum()) == n_probes_s * (n_seg_expected - s)
    return src, probe, em


# ---------------------------------------------------------------- #
# A: pair detection
# ---------------------------------------------------------------- #

def test_detect_pair_ends():
    tok = _kv_tokenizer()
    model = _make_model(tok, pairs_per_segment=2)
    ids = _ids_for(tok, [_ctx(4)])          # 4 pairs = 28 chars + '|' = 29 -> pad 32
    assert ids.shape[1] == 32
    ends = model._detect_pair_ends(ids, tok.pad_token_id)[0]
    # pairs end at 7/14/21; the last is extended through '|' to the real end
    assert ends == [7, 14, 21, 29], ends
    # no pairs at all (all pad) -> empty
    empty = torch.full((1, 32), tok.pad_token_id, dtype=torch.long)
    assert model._detect_pair_ends(empty, tok.pad_token_id) == [[]]
    print("  [detect]  pair ends [7,14,21,29]; trailing '|' joins the last segment")


# ---------------------------------------------------------------- #
# B: balanced division
# ---------------------------------------------------------------- #

def test_divide_pairs():
    D = Model._divide_pairs
    assert D(8, 2, 1) == [2, 2, 2, 2]       # pairs mode
    assert D(8, 3, 1) == [3, 3, 2]
    assert D(2, 3, 1) == [2]
    assert D(8, None, 4) == [2, 2, 2, 2]    # count mode: first n%g groups +1
    assert D(8, None, 3) == [3, 3, 2]
    assert D(3, None, 4) == [1, 1, 1, 0]
    assert D(0, 2, 1) == []
    assert D(0, None, 4) == [0, 0, 0, 0]
    print("  [divide]  8/4 -> 2/2/2/2; 8/3 -> 3/3/2; 3/4 -> 1/1/1/0")


# ---------------------------------------------------------------- #
# C: span construction
# ---------------------------------------------------------------- #

def test_pair_segment_spans():
    tok = _kv_tokenizer()
    model = _make_model(tok, pairs_per_segment=2)
    ids = _ids_for(tok, [_ctx(4), _ctx(2)])   # 29 and 15 real tokens
    n_seg, spans = model._pair_segment_spans(ids, tok.pad_token_id)
    assert n_seg == 2                          # max(ceil(4/2), ceil(2/2)=1)
    (s0, e0), (s1, e1) = spans
    # pairs mode groups by pair COUNT: sample0 = 2 groups of 2; sample1 = 1 group of 2
    assert s0 == [0, 0] and e0 == [14, 15]
    assert s1 == [14, 0] and e1 == [29, 0]     # sample1 has no second group -> empty
    # count mode: 1-pair sample over 2 groups -> [1, 0] -> empty second group
    model_c = _make_model(tok, hopfield_n_segments=2)
    ids_c = _ids_for(tok, [_ctx(4), _ctx(1)])
    n_seg_c, spans_c = model_c._pair_segment_spans(ids_c, tok.pad_token_id)
    assert n_seg_c == 2
    assert spans_c[0][1] == [14, 8]            # sample1's single pair + '|' in group 0
    assert spans_c[1][0][1] == 0 and spans_c[1][1][1] == 0   # sample1: empty seg
    print("  [spans ]  per-sample equal division; short sample -> empty tail group")


# ---------------------------------------------------------------- #
# D: collator attribution == model chunking, no straddling
# ---------------------------------------------------------------- #

def _assert_no_straddle(tok, model, batch, **pair_kwargs):
    ids = _ids_for(tok, [b['context'] for b in batch])
    kv_q = _runner.build_kv_probe_queries(batch, tok, ids.size(1), **pair_kwargs)
    n_seg, spans = model._pair_segment_spans(ids, tok.pad_token_id)
    n_checked = 0
    for b, item in enumerate(batch):
        for j, m in enumerate(PAIR_RE.finditer(item['context'])):
            src = int(kv_q['seg_idx'][b, j])
            s0, e0 = spans[src][0][b], spans[src][1][b]
            assert s0 <= m.start() and m.end() <= e0, \
                f"sample {b} pair {j} ({m.group(0)!r}) spans [{m.start()},{m.end()}) " \
                f"but its attributed segment covers [{s0},{e0})"
            n_checked += 1
    return kv_q, n_seg, n_checked


def test_collator_matches_model_no_straddle():
    tok = _kv_tokenizer()
    batch = [{'context': _ctx(8, 0)}, {'context': _ctx(5, 11)}, {'context': _ctx(3, 23)}]

    # count mode: 8 -> 2/2/2/2, 5 -> 2/1/1/1, 3 -> 1/1/1(+empty)
    model = _make_model(tok, hopfield_n_segments=4)
    kv_q, n_seg, n = _assert_no_straddle(tok, model, batch, pair_aware_n_segments=4)
    assert n_seg == 4 and n == 16
    assert kv_q['seg_idx'][0].tolist()[:8] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert kv_q['seg_idx'][1].tolist()[:5] == [0, 0, 1, 2, 3]
    assert kv_q['seg_idx'][2].tolist()[:3] == [0, 1, 2]

    # pairs mode: 8 -> 4 groups of 2, 5 -> 2/2/1, 3 -> 2/1
    model_p = _make_model(tok, pairs_per_segment=2)
    kv_p, n_seg_p, n_p = _assert_no_straddle(tok, model_p, batch, pairs_per_segment=2)
    assert n_seg_p == 4 and n_p == 16
    assert kv_p['seg_idx'][1].tolist()[:5] == [0, 0, 1, 1, 2]
    print("  [align ]  collator seg_idx == model spans; every pair wholly inside its segment")


# ---------------------------------------------------------------- #
# E: forward + forgetting eval under pair-aware chunking
# ---------------------------------------------------------------- #

def test_forward_pair_aware_hopfield():
    tok = _kv_tokenizer()
    model = _make_model(tok, use_hopfield_memory=True, hopfield_retrieval_mode="softmax",
                        pairs_per_segment=2)
    ids = _ids_for(tok, [_ctx(4), _ctx(4)])
    query = torch.tensor(tok(['?!' + 'ab:'[:0] + 'x'], add_special_tokens=False).input_ids[:1] * 2)
    inputs = {'context_input_ids': ids, 'query_input_ids': ids[:, :1]}
    with torch.no_grad():
        out_plain = model.forward(inputs)
        out_coll = model.forward(inputs, collect_segment_mems=True)
    assert (out_plain['predictions'] - out_coll['predictions']).abs().max().item() == 0.0
    snaps = out_coll['segment_mems']
    assert len(snaps) == 2                       # 4 pairs / 2 per segment
    assert [s['n_stored'] for s in snaps] == [1, 2]
    # every pair's tokens lie inside one segment: reconstruction coverage check
    # (each real token belongs to exactly one segment span)
    n_seg, spans = model._pair_segment_spans(ids, tok.pad_token_id)
    for b in range(2):
        cover = sum(e - s for (s, e) in [(spans[g][0][b], spans[g][1][b]) for g in range(n_seg)])
        real = int((ids[b] != tok.pad_token_id).sum())
        assert cover == real, "segments must tile the real tokens exactly once"
    print("  [fwd   ]  purity + snapshots OK; spans tile the real tokens exactly")


def test_forward_per_segment_eval_pair_aware():
    tok = _kv_tokenizer()
    model = _make_model(tok, use_hopfield_memory=True, hopfield_retrieval_mode="softmax",
                        hopfield_n_segments=2)
    batch = [{'context': _ctx(4, 0)}, {'context': _ctx(4, 11)}]
    coll = _runner.make_collate_fn_per_segment(tok, pair_aware_n_segments=2)
    cb = coll(batch)
    src, probe, em = _check_triples(model, cb['input_ids']['context_input_ids'],
                                    cb['kv_queries'], n_seg_expected=2)
    # 2 pairs per segment per sample: counts 4 probes/row over 2 samples
    assert int((src == 0).sum()) == 8 and int((src == 1).sum()) == 4
    print("  [eval  ]  forgetting triples hold under pair-aware chunking")


def test_forward_per_segment_eval_variable_pairs():
    """Short sample (1 pair over 2 groups) -> empty segment -> None fallback."""
    tok = _kv_tokenizer()
    model = _make_model(tok, use_hopfield_memory=True, hopfield_retrieval_mode="softmax",
                        hopfield_n_segments=2)
    batch = [{'context': _ctx(4, 0)}, {'context': _ctx(1, 11)}]
    coll = _runner.make_collate_fn_per_segment(tok, pair_aware_n_segments=2)
    cb = coll(batch)
    src, probe, em = _check_triples(model, cb['input_ids']['context_input_ids'],
                                    cb['kv_queries'], n_seg_expected=2)
    print("  [eval  ]  variable pair counts: empty group handled via None fallback")


# ---------------------------------------------------------------- #
# F: config validation
# ---------------------------------------------------------------- #

def test_config_validation():
    tok = _kv_tokenizer()
    base = dict(base_config=_base_config(), n_mem_tokens=4, K=2, lr=0.05, grad_mode="none")

    def expect_error(kwargs, frag):
        try:
            ModelConfig(**base, **kwargs)
        except AssertionError as e:
            assert frag in str(e), f"wrong error: {e}"
            return
        raise RuntimeError(f"expected AssertionError({frag!r}) for {kwargs}")

    expect_error(dict(pair_aware_segmentation=True),
                 "pair_delim_token_ids")
    expect_error(dict(pair_aware_segmentation=True, pair_delim_token_ids=_delims(tok),
                      hopfield_segment_size=8),
                 "mutually exclusive")
    expect_error(dict(pairs_per_segment=2), "requires pair_aware_segmentation")
    expect_error(dict(pair_aware_segmentation=True, pair_delim_token_ids=[5, 5]),
                 "distinct")
    # valid minimal config constructs
    ModelConfig(**base, pair_aware_segmentation=True, pair_delim_token_ids=_delims(tok))
    print("  [guard ]  config validation guards fire")


# ---------------------------------------------------------------- #
# Adaptive fork (grad_memgpt_adaptive.py) under the unified surface
# ---------------------------------------------------------------- #

_adp = _load_module("gmm_adp_pair_aware", "grad_memgpt_adaptive.py")
AdpModel, AdpConfig = _adp.GradMemGPT, _adp.GradMemGPTConfig


def _make_adaptive(tok, pairs_per_segment=2, n_segments=2, **kw):
    torch.manual_seed(0)
    cfg = AdpConfig(
        base_config=_base_config(), n_mem_tokens=4, K=2, lr=0.05,
        grad_mode="none", n_segments=n_segments,
        pair_aware_segmentation=True,
        pairs_per_segment=pairs_per_segment,
        pair_delim_token_ids=_delims(tok),
        **kw,
    )
    model = AdpModel(cfg)
    model.eval()
    return model


def test_adaptive_pair_aware_forward_and_eval():
    tok = _kv_tokenizer()
    model = _make_adaptive(tok, pairs_per_segment=2)
    batch = [{'context': _ctx(4, 0)}, {'context': _ctx(4, 11)}]
    ids = _ids_for(tok, [b['context'] for b in batch])
    inputs = {'context_input_ids': ids, 'query_input_ids': ids[:, :1]}
    with torch.no_grad():
        out_plain = model.forward(inputs)
        out_coll = model.forward(inputs, collect_segment_mems=True)
    assert (out_plain['predictions'] - out_coll['predictions']).abs().max().item() == 0.0
    snaps = out_coll['segment_mems']
    assert len(snaps) == 2                          # 4 pairs / 2 per segment
    # carried memory: segment 1 starts where segment 0 ended (RNN carry intact
    # under pair-aware chunking) -- the two snapshots must differ (K steps ran)
    assert (snaps[0] - snaps[1]).abs().max().item() > 0.0

    coll = _runner.make_collate_fn_per_segment(tok, pairs_per_segment=2)
    cb = coll(batch)
    kvq = cb['kv_queries']
    # no-straddle: every pair wholly inside its attributed segment (shared spans)
    n_seg, spans = _adp.pair_segment_spans(ids, tok.pad_token_id, _delims(tok), 2, 2)
    for b, item in enumerate(batch):
        for j, m in enumerate(PAIR_RE.finditer(item['context'])):
            src = int(kvq['seg_idx'][b, j])
            assert spans[src][0][b] <= m.start() and m.end() <= spans[src][1][b]
    src, probe, em = _check_triples(model, ids, kvq, n_seg_expected=2)
    assert int((src == 0).sum()) == 8 and int((src == 1).sum()) == 4
    print("  [adp   ]  adaptive fork: purity, carry, no-straddle, eval triples OK")


def test_adaptive_pair_aware_config_validation():
    tok = _kv_tokenizer()
    try:
        AdpConfig(base_config=_base_config(), n_mem_tokens=4, K=2, lr=0.05,
                  grad_mode="none", segment_size=8, pair_aware_segmentation=True,
                  pair_delim_token_ids=_delims(tok))
    except AssertionError as e:
        assert "mutually exclusive" in str(e)
    else:
        raise RuntimeError("expected AssertionError for segment_size + pair_aware")
    print("  [adp   ]  adaptive config guards fire")


if __name__ == "__main__":
    test_detect_pair_ends()
    test_divide_pairs()
    test_pair_segment_spans()
    test_collator_matches_model_no_straddle()
    test_forward_pair_aware_hopfield()
    test_forward_per_segment_eval_pair_aware()
    test_forward_per_segment_eval_variable_pairs()
    test_config_validation()
    test_adaptive_pair_aware_forward_and_eval()
    test_adaptive_pair_aware_config_validation()
    print("ALL pair-aware segmentation tests passed")
