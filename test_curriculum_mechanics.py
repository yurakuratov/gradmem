"""Curriculum-mechanics tests for the adaptive fork (``grad_memgpt_adaptive.py``).

The curriculum machinery lives in ``run_gradmemgpt_on_kv_retrieval.py``
(``CurriculumCallback``, ``CurriculumTrainer``, and the stage loop in ``main()``),
and it is the run path that drives the **adaptive** model stage-by-stage on a
growing context. These tests lock in that the mechanics behave correctly against
a real (tiny) ``GradMemGPT`` (adaptive) instance:

  1. ``CurriculumCallback`` threshold gate (no model) -- below / at / above /
     missing-key / ``eval_`` prefix lookup.
  2. ``CurriculumTrainer`` stage-metadata injection + ``step_offset`` accounting
     (``global_step`` offset applied for logging, then restored).
  3. End-to-end multi-stage loop: advances when the threshold is reached, stops
     when it is not, carries one model across stages, and accumulates
     ``cumulative_steps``.
  4. Segmentation of the growing WRITE context: later curriculum stages have
     longer contexts and the adaptive fork's cross-segment memory carry handles
     them (memory is *not* reset between segments -- the fork's raison d'être).

The integration tests build a tiny from-scratch GPT-2 (AGENTS.md smoke-test
recipe) and run on CPU, so they finish in a few seconds without a GPU.

Run two ways:
    python test_curriculum_mechanics.py        # self-contained runner
    pytest test_curriculum_mechanics.py        # if pytest is installed
"""

import importlib.util
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import partial

import torch

# `no_cuda=True` (used by the CPU-only trainers below) is deprecated in favour of
# `use_cpu` in newer transformers; silence the one-shot FutureWarning so it does
# not drown the test's own output. Both kwargs coexist in current versions.
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*`no_cuda` is deprecated.*")

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Module loading: load the run script (which defines the curriculum classes)
# and the adaptive model module under distinct names. The run script imports
# `grad_memgpt` at top level, so loading it pulls in the full base model too --
# that is expected and matches production.
# --------------------------------------------------------------------------- #
def _load_module(name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_RUN = None
_ADP = None
_TOKENIZER = None


def _run_module():
    global _RUN
    if _RUN is None:
        _RUN = _load_module("rgm_curriculum", "run_gradmemgpt_on_kv_retrieval.py")
    return _RUN


def _adaptive_module():
    global _ADP
    if _ADP is None:
        _ADP = _load_module("gmm_adp_curriculum", "grad_memgpt_adaptive.py")
    return _ADP


def _tokenizer():
    """kv_alphabet_62 char-level WordLevel tokenizer (70 tokens); cached."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            os.path.join(_HERE, "tokenizers", "kv_alphabet_62"))
    return _TOKENIZER


# --------------------------------------------------------------------------- #
# Tiny model + synthetic KV-retrieval data (AGENTS.md smoke-test recipe).
# --------------------------------------------------------------------------- #
PAD_ID, BOS_ID = 0, 1


def _base_gpt2_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.vocab_size = _tokenizer().vocab_size
    cfg.pad_token_id = PAD_ID
    cfg.bos_token_id = BOS_ID
    cfg.eos_token_id = 2
    cfg.use_cache = False
    # disable dropout so a run is deterministic
    cfg.resid_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    return cfg


def _make_adaptive_model(n_segments=1, segment_size=None, rule="sgd", K=2):
    """Tiny adaptive GradMemGPT. Defaults reproduce the plain (SGD) path."""
    Adp = _adaptive_module()
    cfg = _base_gpt2_config()
    mcfg = Adp.GradMemGPTConfig(
        base_config=cfg, n_mem_tokens=4, K=K, lr=0.05, grad_mode="second",
        n_segments=n_segments, segment_size=segment_size, memory_update_rule=rule,
    )
    torch.manual_seed(0)
    return Adp.GradMemGPT(mcfg)


def _kv_row(n_kv, seed):
    """One synthetic KV-retrieval row: ``!K:V!`` facts + a probe for the first.

    Returns the dict schema collate_fn expects: {context, query, target}.
    """
    import random
    r = random.Random(seed)
    alphabet = "abcdefgh"  # 1 char = 1 token for this tokenizer
    parts = []
    for _ in range(n_kv):
        parts.append(f"!{r.choice(alphabet)}:{r.choice(alphabet)}!")
    context = "".join(parts)
    k, v = context[1], context[3]          # the first KV pair
    return {"context": context, "query": f"?!{k}:", "target": f"{v}!|"}


def _stage_dataset(n_kv, n_rows=16, seed=0):
    import datasets
    return datasets.Dataset.from_list([_kv_row(n_kv, seed * 1000 + i) for i in range(n_rows)])


@contextmanager
def _quiet():
    """Swallow noise from the HF Trainer + run module during an integration run.

    The run module's ``compute_metrics_fn`` prints 5 decoded samples per eval
    (bare ``print`` -> ``sys.stdout``) and the HF Trainer logs to the root
    logger (-> ``sys.stderr``). Both are pure debug chatter with no bearing on
    the assertions, so we silence the root logger and swap ``sys.stdout`` /
    ``sys.stderr`` for devnull for the duration of the block. We redirect at the
    Python-object level (not just the OS fd) because ``print`` writes to
    ``sys.stdout`` directly, and also at the OS fd level so C-extension output
    (tqdm, etc.) is caught too.
    """
    import logging
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    saved_fds = (os.dup(1), os.dup(2))
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    devnull_py = open(os.devnull, "w")
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.CRITICAL)
    sys.stdout, sys.stderr = devnull_py, devnull_py
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(devnull_fd)
        devnull_py.close()
        root.setLevel(prev_level)


def _make_trainer(model, tokenizer, train_ds, eval_ds, *, max_steps, threshold,
                  stage, label, step_offset, eval_steps=1):
    """Build a CurriculumTrainer + CurriculumCallback exactly like the run loop."""
    rgm = _run_module()
    from transformers import TrainingArguments

    collate = partial(rgm.collate_fn, tokenizer=tokenizer)
    ignore = [tokenizer.convert_tokens_to_ids(c) for c in "!|"]
    compute_metrics = partial(rgm.compute_metrics_fn, ignore_token_ids=ignore,
                              tokenizer=tokenizer)

    tmp = tempfile.mkdtemp(prefix=f"curr_stage{stage}_")
    args = TrainingArguments(
        output_dir=tmp,
        report_to=[],
        max_steps=max_steps,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=1,
        eval_strategy="steps",
        save_strategy="no",
        eval_steps=eval_steps,
        load_best_model_at_end=False,
        eval_on_start=True,                 # so the callback can fire before train
        greater_is_better=True,
        remove_unused_columns=False,
        include_for_metrics=["inputs"],
        no_cuda=True,                       # CPU: deterministic + no GPU needed
        disable_tqdm=True,
        seed=0,
    )
    cb = rgm.CurriculumCallback(metric_name="exact_match", threshold=threshold)
    trainer = rgm.CurriculumTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=rgm.preprocess_logits_for_metrics,
        callbacks=[cb],
        step_offset=step_offset,
        curriculum_stage=stage,
        curriculum_stage_label=label,
    )
    return trainer, cb


def _run_curriculum(stages, threshold, *, max_steps=3):
    """Compact mirror of the curriculum loop in main() (lines ~1222-1335).

    Each entry of ``stages`` is ``(label, n_kv)`` -- e.g. ``("N8", 8)`` -- mirroring
    the production loop where the dataset name/label and its size advance together.
    Runs stage-by-stage, carrying one model and accumulating ``cumulative_steps``;
    breaks early when a stage does not reach the threshold (the same advancement
    gate as production). Returns a dict with the per-stage results and the carried model.
    """
    tokenizer = _tokenizer()
    model = _make_adaptive_model()
    init_mem = model.mem.detach().clone()

    cumulative = 0
    results = []
    for stage, (label, n_kv) in enumerate(stages):
        ds = _stage_dataset(n_kv, seed=stage)
        trainer, cb = _make_trainer(
            model, tokenizer, ds, ds, max_steps=max_steps, threshold=threshold,
            stage=stage, label=label, step_offset=cumulative)
        with _quiet():
            trainer.train()
        train_logs = [l for l in trainer.state.log_history if "loss" in l]
        results.append({
            "stage": stage,
            "label": label,
            "n_kv": n_kv,
            "global_step": trainer.state.global_step,
            "threshold_reached": cb.threshold_reached,
            "first_log": train_logs[0] if train_logs else {},
        })
        cumulative += trainer.state.global_step
        model = trainer.model              # carry the (possibly updated) model
        if not cb.threshold_reached:
            break                          # advancement gate
    return {
        "stages": results,
        "n_completed": len(results),
        "cumulative_steps": cumulative,
        "model": model,
        "init_mem": init_mem,
    }


# --------------------------------------------------------------------------- #
# 1. CurriculumCallback threshold gate (pure unit, no model)
# --------------------------------------------------------------------------- #
def test_curriculum_callback_threshold_gate():
    from transformers import TrainerControl

    rgm = _run_module()

    def _fresh_control():
        c = TrainerControl()
        c.should_training_stop = False
        return c

    # (a) below threshold -> no effect
    cb = rgm.CurriculumCallback(metric_name="exact_match", threshold=0.95)
    ctrl = _fresh_control()
    cb.on_evaluate(None, None, ctrl, metrics={"eval_exact_match": 0.5})
    assert cb.threshold_reached is False, "below threshold must not set reached"
    assert ctrl.should_training_stop is False, "below threshold must not stop"

    # (b) at threshold (boundary is >=)
    cb = rgm.CurriculumCallback(metric_name="exact_match", threshold=0.95)
    ctrl = _fresh_control()
    cb.on_evaluate(None, None, ctrl, metrics={"eval_exact_match": 0.95})
    assert cb.threshold_reached is True, "boundary value must trigger (>=)"
    assert ctrl.should_training_stop is True, "threshold reached must request stop"

    # (c) above threshold
    cb = rgm.CurriculumCallback(metric_name="exact_match", threshold=0.95)
    ctrl = _fresh_control()
    cb.on_evaluate(None, None, ctrl, metrics={"eval_exact_match": 0.99})
    assert cb.threshold_reached is True
    assert ctrl.should_training_stop is True

    # (d) metric key absent -> graceful no-op (production evals may omit it)
    cb = rgm.CurriculumCallback(metric_name="exact_match", threshold=0.95)
    ctrl = _fresh_control()
    cb.on_evaluate(None, None, ctrl, metrics={"eval_loss": 4.2})  # no eval_exact_match
    assert cb.threshold_reached is False, "missing metric must be a no-op"
    assert ctrl.should_training_stop is False

    # (e) the bare metric_name is auto-prefixed with eval_ for the lookup
    cb = rgm.CurriculumCallback(metric_name="exact_match", threshold=0.5)
    ctrl = _fresh_control()
    cb.on_evaluate(None, None, ctrl, metrics={"exact_match": 0.9})  # bare, no eval_
    assert cb.threshold_reached is False, "only eval_-prefixed key is consulted"

    print("  [cb]     threshold gate: below/at/above/missing/prefix OK")


# --------------------------------------------------------------------------- #
# 2. CurriculumTrainer: stage metadata + step_offset accounting (unit)
# --------------------------------------------------------------------------- #
def test_curriculum_trainer_injects_stage_metadata_and_offset():
    rgm = _run_module()
    from transformers import TrainingArguments

    tokenizer = _tokenizer()
    model = _make_adaptive_model()
    ds = _stage_dataset(4)
    collate = partial(rgm.collate_fn, tokenizer=tokenizer)
    ignore = [tokenizer.convert_tokens_to_ids(c) for c in "!|"]
    compute_metrics = partial(rgm.compute_metrics_fn, ignore_token_ids=ignore,
                              tokenizer=tokenizer)

    tmp = tempfile.mkdtemp(prefix="curr_meta_")
    args = TrainingArguments(
        output_dir=tmp, report_to=[], max_steps=1, no_cuda=True,
        disable_tqdm=True, remove_unused_columns=False, eval_strategy="no",
        save_strategy="no", per_device_train_batch_size=8, logging_steps=1)
    trainer = rgm.CurriculumTrainer(
        model=model, args=args, train_dataset=ds, eval_dataset=ds,
        data_collator=collate, compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=rgm.preprocess_logits_for_metrics,
        step_offset=100, curriculum_stage=2, curriculum_stage_label="N16")

    # Pretend the optimizer just finished step 5 of this stage. The log must
    # report step = 5 + step_offset(100) = 105, then leave global_step at 5.
    # (HF's ProgressCallback echoes the log to stderr; hush it for clean output.)
    trainer.state.global_step = 5
    with _quiet():
        trainer.log({"loss": 0.5})
    assert trainer.state.global_step == 5, "step_offset must not corrupt global_step"
    entry = trainer.state.log_history[-1]
    assert entry["curriculum_stage"] == 2, entry
    assert entry["curriculum_stage_label"] == "N16", entry
    assert entry["step"] == 105, f"logged step must include step_offset: {entry}"

    # A second log at global_step=6 must report 106 -- offset is re-applied, not sticky.
    trainer.state.global_step = 6
    with _quiet():
        trainer.log({"loss": 0.4})
    assert trainer.state.global_step == 6
    assert trainer.state.log_history[-1]["step"] == 106
    print("  [meta]   stage metadata + step_offset: stage=2 label=N16 step=105 OK")


# --------------------------------------------------------------------------- #
# 3. End-to-end: curriculum advances through stages when threshold is reached
# --------------------------------------------------------------------------- #
def test_curriculum_advances_when_threshold_reached():
    # threshold 0.0 -> exact_match (==0 at init) always satisfies >=, so every
    # stage advances. Three stages of growing context.
    res = _run_curriculum([("N4", 4), ("N8", 8), ("N16", 16)], threshold=0.0)

    assert res["n_completed"] == 3, f"all 3 stages must run, got {res['n_completed']}"
    assert [s["label"] for s in res["stages"]] == ["N4", "N8", "N16"]
    assert all(s["threshold_reached"] for s in res["stages"]), \
        "with threshold 0.0 every stage must reach it"

    # cumulative_steps == sum of per-stage global_steps
    per_stage_steps = [s["global_step"] for s in res["stages"]]
    assert res["cumulative_steps"] == sum(per_stage_steps), \
        f"cumulative {res['cumulative_steps']} != sum {sum(per_stage_steps)}"
    assert res["cumulative_steps"] > 0, "stages must actually train"

    # the carried model was updated by training (meta-learned mem moved)
    final_mem = res["model"].mem.detach()
    assert not torch.allclose(final_mem, res["init_mem"]), \
        "shared model must be trained across stages"

    # stage labels + offset propagate into each stage's log history
    for s in res["stages"]:
        fl = s["first_log"]
        assert fl.get("curriculum_stage") == s["stage"], fl
        assert fl.get("curriculum_stage_label") == s["label"], fl

    print(f"  [adv]    3 stages advanced: steps={per_stage_steps} "
          f"cumulative={res['cumulative_steps']} mem-trained=True OK")


# --------------------------------------------------------------------------- #
# 4. End-to-end: curriculum stops when a stage does NOT reach the threshold
# --------------------------------------------------------------------------- #
def test_curriculum_stops_when_threshold_not_reached():
    # threshold 2.0 is unreachable (exact_match in [0,1]) -> stage 0 never reaches
    # and the advancement gate breaks the loop immediately.
    res = _run_curriculum([("N4", 4), ("N8", 8), ("N16", 16)], threshold=2.0)

    assert res["n_completed"] == 1, \
        f"must stop after the first (unreached) stage, got {res['n_completed']}"
    only = res["stages"][0]
    assert only["label"] == "N4"
    assert only["threshold_reached"] is False, "stage 0 must NOT reach threshold 2.0"
    print(f"  [stop]   stopped at stage 0 (reached=False), "
          f"global_step={only['global_step']} OK")


# --------------------------------------------------------------------------- #
# 5. Segmentation handles the growing WRITE context (adaptive-fork specific)
#    The fork's whole point: memory is carried ACROSS segments without reset.
#    Curriculum stages grow the context; a segment_size config splits each stage's
#    context into >=1 segments and the model must train through all of them.
# --------------------------------------------------------------------------- #
def test_curriculum_segmentation_handles_growing_context():
    Adp = _adaptive_module()
    tokenizer = _tokenizer()
    # fixed segment_size => #segments grows with context length across stages.
    # n_segments=1 + segment_size set => n_segments = ceil(S / segment_size).
    model = _make_adaptive_model(n_segments=1, segment_size=8)

    # stage 0 context (4 KV pairs) fits in ~1 segment; stage 1 (12 pairs) spans
    # several segments, exercising the cross-segment memory carry.
    stages = [("N4", 4), ("N12", 12)]
    cumulative = 0
    carried = model
    per_stage_segs = []
    for stage, (label, n_kv) in enumerate(stages):
        ds = _stage_dataset(n_kv, seed=stage)
        # measure how many segments this stage's WRITE context splits into,
        # using the SAME ceil-chunking the forward uses (single source of truth).
        sample_ctx = tokenizer(ds[0]["context"], add_special_tokens=True).input_ids
        n_seg = Adp.GradMemGPT._segment_bounds(
            len(sample_ctx), model.n_segments, model.segment_size)[0]
        per_stage_segs.append((label, n_seg))

        trainer, cb = _make_trainer(
            carried, tokenizer, ds, ds, max_steps=2, threshold=0.0,
            stage=stage, label=label, step_offset=cumulative)
        with _quiet():
            trainer.train()
        assert cb.threshold_reached, f"stage {label} must advance"
        cumulative += trainer.state.global_step
        carried = trainer.model

    # later stage has strictly more segments than the earlier one
    assert per_stage_segs[1][1] > per_stage_segs[0][1] >= 1, \
        f"context must split into more segments at later stage: {per_stage_segs}"
    # the carried (adaptive) model trained through the multi-segment contexts.
    # `model` is the same object as `carried` (shared through the loop), so
    # compare the final memory against a freshly-seeded identical config.
    baseline = _make_adaptive_model(n_segments=1, segment_size=8)
    assert not torch.allclose(carried.mem.detach(), baseline.mem.detach()), \
        "adaptive model must train through segmented contexts"
    print(f"  [seg]    segmentation grows across stages {per_stage_segs}, "
          f"cumulative={cumulative} OK")


# --------------------------------------------------------------------------- #
# 6. Forced forgetting matrix at stage boundary (adaptive + per_segment_eval)
#    The curriculum loop calls trainer.force_forgetting_pass(stage_idx, label)
#    before the end-of-stage eval. This must (a) bypass the step-cadence gate,
#    (b) write a stage-labeled forgetting_matrix_*_stage{idx}_{label}.* artifact,
#    and (c) clear the one-shot flag so a subsequent (un-forced) eval does NOT
#    re-fire off-cadence.
# --------------------------------------------------------------------------- #
def _make_per_segment_trainer(model, tokenizer, ds, *, stage, label, exp_path,
                              per_segment_eval_steps=1_000_000):
    """Like _make_trainer but wires the per-segment forgetting path.

    ``per_segment_eval_steps`` defaults huge so the periodic gate never trips --
    only an explicit force_forgetting_pass() triggers a pass. This isolates the
    forced-trigger behavior from cadence.
    """
    rgm = _run_module()
    from transformers import TrainingArguments

    collate = partial(rgm.collate_fn, tokenizer=tokenizer)
    ignore = [tokenizer.convert_tokens_to_ids(c) for c in "!|"]
    compute_metrics = partial(rgm.compute_metrics_fn, ignore_token_ids=ignore,
                              tokenizer=tokenizer)
    per_segment_collator = rgm.make_collate_fn_per_segment(
        tokenizer, n_segments=model.n_segments, segment_size=model.segment_size)

    args = TrainingArguments(
        output_dir=exp_path, report_to=[], max_steps=1, no_cuda=True,
        disable_tqdm=True, remove_unused_columns=False, include_for_metrics=["inputs"],
        eval_strategy="no", save_strategy="no", per_device_train_batch_size=8,
        per_device_eval_batch_size=8, logging_steps=1)
    trainer = rgm.CurriculumTrainer(
        model=model, args=args, train_dataset=ds, eval_dataset=ds,
        data_collator=collate, compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=rgm.preprocess_logits_for_metrics,
        callbacks=[rgm.CurriculumCallback(metric_name="exact_match", threshold=2.0)],
        step_offset=0, curriculum_stage=stage, curriculum_stage_label=label,
        per_segment_dataset=ds, per_segment_collator=per_segment_collator,
        n_segments=(model.n_segments if model.segment_size is None else None),
        segment_size=model.segment_size, tokenizer=tokenizer,
        per_segment_eval_steps=per_segment_eval_steps, eval_steps=1,
        exp_path=exp_path)
    return trainer


def test_forced_forgetting_pass_fires_at_stage_boundary():
    import glob
    import json as _json

    tokenizer = _tokenizer()
    # segmented adaptive model so forward_per_segment_eval has >=1 segment
    model = _make_adaptive_model(n_segments=1, segment_size=8)
    ds = _stage_dataset(4, seed=0)
    exp_path = tempfile.mkdtemp(prefix="curr_force_")

    trainer = _make_per_segment_trainer(
        model, tokenizer, ds, stage=0, label="N4", exp_path=exp_path)

    # No forced pass yet -> a plain evaluate() must NOT write any artifact.
    # Step must be > 0 and off-cadence: at step 0 the modulo gate is trivially
    # true (0 % N == 0), so move to step 1 with the huge cadence to avoid that
    # edge case and isolate the forced-trigger behavior.
    trainer.state.global_step = 1
    with _quiet():
        trainer.evaluate(ds)
    artifacts = glob.glob(os.path.join(exp_path, "forgetting_matrix_*"))
    assert artifacts == [], f"no artifact expected before force, got {artifacts}"

    # Force a stage-boundary pass, then evaluate -> stage-labeled artifact appears.
    trainer.force_forgetting_pass(stage_idx=0, stage_label="N4")
    with _quiet():
        trainer.evaluate(ds)
    forced_json = glob.glob(os.path.join(exp_path, "forgetting_matrix_*_stage0_N4.json"))
    forced_csv = glob.glob(os.path.join(exp_path, "forgetting_matrix_*_stage0_N4.csv"))
    assert len(forced_json) == 1, f"expected 1 forced .json, got {forced_json}"
    assert len(forced_csv) == 1, f"expected 1 forced .csv, got {forced_csv}"

    # the artifact JSON carries the stage-tagging fields
    with open(forced_json[0]) as f:
        art = _json.load(f)
    assert art.get("forced") is True, art
    assert art.get("stage_idx") == 0, art
    assert art.get("stage_label") == "N4", art
    assert "matrix_exact_match" in art, art

    # the one-shot flag cleared: another (un-forced) evaluate() writes nothing new
    n_before = len(glob.glob(os.path.join(exp_path, "forgetting_matrix_*")))
    with _quiet():
        trainer.evaluate(ds)
    n_after = len(glob.glob(os.path.join(exp_path, "forgetting_matrix_*")))
    assert n_after == n_before, \
        f"force flag must be one-shot; artifact count grew {n_before} -> {n_after}"

    # the forced metrics also flow into the evaluate() return dict
    trainer.force_forgetting_pass(stage_idx=0, stage_label="N4")
    with _quiet():
        m = trainer.evaluate(ds)
    assert "forget_diag_mean" in m, f"forced metrics missing from eval return: {sorted(m)}"
    print("  [force]  forced forgetting pass: stage0_N4 artifact written, "
          "JSON tagged, one-shot reset, metrics returned OK")


TESTS = [
    test_curriculum_callback_threshold_gate,
    test_curriculum_trainer_injects_stage_metadata_and_offset,
    test_curriculum_advances_when_threshold_reached,
    test_curriculum_stops_when_threshold_not_reached,
    test_curriculum_segmentation_handles_growing_context,
    test_forced_forgetting_pass_fires_at_stage_boundary,
]


def _main():
    print("=" * 70)
    print("curriculum mechanics vs grad_memgpt_adaptive (CurriculumCallback / Trainer)")
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
