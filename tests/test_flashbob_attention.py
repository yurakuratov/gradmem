"""FlashBoB integration: operator routes, meta-gradients, and launcher wiring."""

import ast
from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig
from transformers import HfArgumentParser

from attn_double_bwd import flashbob_attention as adapter
from grad_memgpt import GradMemGPT, GradMemGPTConfig


ROOT = Path(__file__).resolve().parents[1]


def _base(family):
    common = dict(vocab_size=41, max_position_embeddings=64, pad_token_id=0)
    if family == "gpt2":
        return GPT2Config(vocab_size=41, n_embd=64, n_head=2, n_layer=1,
                          n_positions=64, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
                          pad_token_id=0)
    if family == "gpt_neox":
        return GPTNeoXConfig(**common, hidden_size=64, num_attention_heads=2,
                             num_hidden_layers=1, intermediate_size=128)
    return LlamaConfig(**common, hidden_size=64, num_attention_heads=2,
                       num_key_value_heads=2, num_hidden_layers=1, intermediate_size=128)


def _model(family, attention, objective="reconstruction", backend="prefix", **kwargs):
    return GradMemGPT(GradMemGPTConfig(
        base_config=_base(family), attn_implementation=attention,
        write_objective=objective, memory_backend=backend,
        n_mem_tokens=2, K=2, lr=0.1, grad_mode="second", **kwargs,
    )).eval()


def _batch(device="cpu"):
    # Right-padded contexts and ignored query targets mimic variable-length NLP,
    # as well as synthetic KV/MQAR batches. READ never receives the context.
    return ({"context_input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]], device=device),
             "query_input_ids": torch.tensor([[9, 10, 11], [12, 13, 14]], device=device)},
            torch.tensor([[-100, 15, 16], [-100, 17, -100]], device=device))


def _require_bob():
    pytest.importorskip("attn_double_bwd.flashbob")
    adapter.register_flashbob()


def test_flashbob_is_bundled_without_external_imports(monkeypatch):
    import builtins
    import inspect
    real_import = builtins.__import__
    def without_external_bob(name, *args, **kwargs):
        if name == "bob" or name.startswith("bob."):
            raise AssertionError("external FlashBoB package was imported")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_external_bob)
    monkeypatch.setattr(adapter, "_sdpa_bob", None)
    adapter.register_flashbob()
    source = Path(inspect.getfile(adapter._sdpa_bob)).resolve()
    assert source.is_relative_to(ROOT / "attn_double_bwd" / "flashbob")
    assert not source.is_symlink()
    model = _model("gpt2", "flashbob")
    inputs, labels = _batch()
    model(inputs, labels=labels)["loss"].backward()
    assert model.mem.grad is not None


@pytest.mark.parametrize("family", ["gpt2", "gpt_neox", "llama"])
@pytest.mark.parametrize("objective", ["reconstruction", "energy", "energy_with_reconstruction"])
def test_flashbob_matches_eager_meta_gradients(family, objective):
    _require_bob()
    torch.manual_seed(12)
    ref = _model(family, "eager", objective)
    model = _model(family, "flashbob", objective)
    model.load_state_dict(ref.state_dict())
    batch, labels = _batch()
    expected = ref(batch, labels=labels, return_mem=True)
    actual = model(batch, labels=labels, return_mem=True)
    torch.testing.assert_close(actual["loss"], expected["loss"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual["mem"], expected["mem"], atol=2e-6, rtol=2e-5)
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, param), (_, reference) in zip(model.named_parameters(), ref.named_parameters()):
        assert (param.grad is None) == (reference.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, reference.grad, atol=5e-6, rtol=2e-4, msg=name)
    assert model.mem.grad is not None and model.mem.grad.norm() > 0


@pytest.mark.parametrize("backend", ["lora", "kv_cache"])
@pytest.mark.parametrize("family", ["gpt2", "gpt_neox", "llama"])
def test_flashbob_other_memory_backends(family, backend):
    _require_bob()
    model = _model(family, "flashbob", backend=backend)
    batch, labels = _batch()
    result = model(batch, labels=labels)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())


@pytest.mark.parametrize("attention", ["flashbob", "bob"])
def test_flashbob_checkpoint_and_separate_writer(tmp_path, attention):
    _require_bob()
    model = _model("gpt2", attention, "energy", separate_writer=True,
                   writer_base_config=_base("llama"))
    model.save_pretrained(tmp_path)
    restored = GradMemGPT.from_pretrained(tmp_path).eval()
    assert restored.config.attn_implementation == attention
    assert restored.write_model.config._attn_implementation == attention
    assert restored.read_model.config._attn_implementation == attention
    batch, labels = _batch()
    restored(batch, labels=labels)["loss"].backward()
    assert restored.mem.grad is not None


def test_flashbob_pretrained_backbone(tmp_path):
    _require_bob()
    GradMemGPT._build_model(None, _base("gpt2"), "eager").save_pretrained(tmp_path)
    model = GradMemGPT(GradMemGPTConfig(pretrained_model=str(tmp_path),
                                       attn_implementation="flashbob", n_mem_tokens=2, K=2))
    batch, labels = _batch()
    model(batch, labels=labels)["loss"].backward()
    assert model.mem.grad is not None


@pytest.mark.parametrize("mode", ["padding", "packed"])
def test_flashbob_preserves_explicit_model_masks_and_positions(mode):
    _require_bob()
    ref = GradMemGPT._build_model(None, _base("llama"), "eager").eval()
    model = GradMemGPT._build_model(None, _base("llama"), "flashbob").eval()
    model.load_state_dict(ref.state_dict())
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    kwargs = {"use_cache": False}
    if mode == "padding":
        kwargs["attention_mask"] = torch.tensor([[1, 1, 1, 0, 0]])
    else:
        kwargs["position_ids"] = torch.tensor([[0, 1, 2, 0, 1]])
    torch.testing.assert_close(model(tokens, **kwargs).logits, ref(tokens, **kwargs).logits,
                               atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", ["mask", "dropout", "float64", "gqa", "head_mask", "window"])
def test_math_fallback_preserves_second_derivatives(mode, monkeypatch):
    def no_bob(*args, **kwargs):
        raise AssertionError("unsupported call reached FlashBoB")
    monkeypatch.setattr(adapter, "_sdpa_bob", no_bob)
    dtype = torch.float64 if mode == "float64" else torch.float32
    q = torch.randn(2, 2, 3, 32, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 1 if mode == "gqa" else 2, 5, 32, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = torch.ones(2, 1, 3, 5, dtype=torch.bool) if mode == "mask" else None
    if mask is not None:
        mask[..., -1] = False
    kwargs = {"head_mask": torch.tensor([1., 0.]).view(1, 2, 1, 1)} if mode == "head_mask" else {}
    if mode == "window":
        # CPU float64 ensures a fallback, including window/offset semantics.
        q, k, v = [x.detach().double().requires_grad_() for x in (q, k, v)]
        kwargs["sliding_window"] = 2
    out, _ = adapter.flashbob_attention_forward(
        SimpleNamespace(is_causal=True), q, k, v, mask,
        dropout=0.2 if mode == "dropout" else 0, **kwargs,
    )
    first = torch.autograd.grad(out.square().sum(), (q, k, v), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (q, k, v))
    assert all(torch.isfinite(g).all() for g in second)
    if mode == "mask":
        # A masked key must not affect any output, including through gradients.
        assert torch.count_nonzero(first[1][..., -1, :]) == 0
    if mode == "head_mask":
        assert torch.count_nonzero(out[:, :, 1]) == 0


def test_flashbob_missing_dependency_is_actionable(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def import_without_bob(name, *args, **kwargs):
        if name == "flashbob":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(adapter, "_sdpa_bob", None)
    monkeypatch.setattr(builtins, "__import__", import_without_bob)
    with pytest.raises(ImportError, match="environment-flashbob.yaml"):
        _model("gpt2", "flashbob")
    # Merely importing or using eager GradMem must not require FlashBoB.
    _model("gpt2", "eager")


def test_flashbob_gpt2_scaling_and_dropout():
    _require_bob()
    config = _base("gpt2")
    config.scale_attn_weights = False
    config.scale_attn_by_inverse_layer_idx = True
    config.attn_pdrop = 0.2
    ref = GradMemGPT._build_model(None, config.to_dict(), "eager").train()
    model = GradMemGPT._build_model(None, config.to_dict(), "flashbob").train()
    model.load_state_dict(ref.state_dict())
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.manual_seed(99)
    expected = ref(tokens).logits
    torch.manual_seed(99)
    actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("masked", [False, True])
def test_flashbob_rectangular_causal_alignment(masked):
    _require_bob()
    q = torch.zeros(1, 1, 2, 32)
    k = torch.zeros(1, 1, 4, 32)
    v = torch.arange(4.).view(1, 1, 4, 1).expand_as(k)
    mask = torch.tensor([True, True, True, False]).view(1, 1, 1, 4) if masked else None
    out, _ = adapter.flashbob_attention_forward(
        SimpleNamespace(is_causal=True), q, k, v, mask, is_causal=True,
    )
    # Cached queries occupy absolute positions 2 and 3, not positions 0 and 1.
    expected = torch.tensor([1., 1. if masked else 1.5])
    torch.testing.assert_close(out[0, :, 0, 0], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("family", ["gpt2", "gpt_neox", "llama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("objective", ["reconstruction", "energy", "energy_with_reconstruction"])
def test_flashbob_cuda_meta_training(family, dtype, objective, monkeypatch):
    _require_bob()
    calls = []
    real_bob = adapter._sdpa_bob
    def counted(*args, **kwargs):
        calls.append(args[0].dtype)
        return real_bob(*args, **kwargs)
    monkeypatch.setattr(adapter, "_sdpa_bob", counted)
    model = _model(family, "flashbob", objective).cuda().train()
    reference = _model(family, "eager", objective).cuda().train()
    reference.load_state_dict(model.state_dict())
    batch, labels = _batch("cuda")
    context = torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    with context:
        result = model(batch, labels=labels, return_mem=True)
    result["loss"].backward()
    with torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext():
        expected = reference(batch, labels=labels)
    expected["loss"].backward()
    torch.testing.assert_close(result["loss"], expected["loss"], atol=0.02, rtol=0.01)
    error = (model.mem.grad - reference.mem.grad).norm() / reference.mem.grad.norm().clamp_min(1e-8)
    assert error < (0.05 if dtype == torch.bfloat16 else 0.01), error.item()
    assert torch.isfinite(result["loss"])
    assert model.mem.grad is not None and torch.isfinite(model.mem.grad).all()
    if objective != "reconstruction":
        assert any(p.grad is not None and p.grad.norm() > 0 for n, p in model.named_parameters() if "energy" in n)
    assert bool(calls) == (dtype != torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_flashbob_energy_shaping_backward(device):
    _require_bob()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _model("llama", "flashbob", "energy", use_layerwise_energy=True,
                   energy_rank_weight=0.01, energy_traj_weight=0.01,
                   energy_anchor_weight=0.001, energy_memory_search_weight=0.1,
                   energy_memory_search_num_samples=2).to(device).train()
    batch, labels = _batch(device)
    with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        output = model(batch, labels=labels)
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.mem.grad is not None and torch.isfinite(model.mem.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashbob_cuda_gqa_second_order(monkeypatch):
    _require_bob()
    model = _base("llama")
    model.num_key_value_heads = 1
    model = GradMemGPT(GradMemGPTConfig(base_config=model, attn_implementation="flashbob",
                                       n_mem_tokens=2, K=2, grad_mode="second")).cuda()
    real_bob = adapter._sdpa_bob
    heads = []
    def counted(q, k, v, **kwargs):
        heads.append((q.shape[1], k.shape[1]))
        return real_bob(q, k, v, **kwargs)
    monkeypatch.setattr(adapter, "_sdpa_bob", counted)
    batch, labels = _batch("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch, labels=labels)
    output["loss"].backward()
    assert (2, 1) in heads
    assert model.mem.grad is not None and torch.isfinite(model.mem.grad).all()


LAUNCHERS = sorted(ROOT.glob("scripts/**/run_*gradmem*.sh"))


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
@pytest.mark.parametrize("attention", ["eager", "hvp_manual", "hvp_semi_manual", "flashbob"])
def test_bash_attention_override_reaches_command(script, tmp_path, attention):
    # Execute real launcher control flow, but never launch training or touch runs/.
    scripts = tmp_path / "scripts"
    shutil.copytree(ROOT / "scripts", scripts)
    (scripts / "collect_env_state.sh").write_text("""
prepare_locked_run() { RUN_LOCK_LOG="$CAPTURE.log"; RUN_LOCK_DIR="$CAPTURE.lock"; return 0; }
find_free_port() { echo 29599; }
print_run_header() { :; }
finalize_locked_run() { :; }
start_run_timer() { :; }
finish_run_timer() { :; }
""")
    binary = tmp_path / "bin"
    binary.mkdir()
    accelerate = binary / "accelerate"
    accelerate.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\n')
    accelerate.chmod(0o755)
    capture = tmp_path / "args"
    env = dict(os.environ, ATTN_IMPL=attention, NP="1", CAPTURE=str(capture),
               PATH=f"{binary}:{os.environ['PATH']}")
    result = subprocess.run(["bash", str(scripts / script.relative_to(ROOT / "scripts"))],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index("--attn_implementation") + 1] == attention
    assert ("_attn_" in args[args.index("--exp_path") + 1]) == (attention != "eager")


@pytest.mark.parametrize("runner", sorted(ROOT.glob("run_gradmemgpt_on_*.py")), ids=lambda p: p.name)
@pytest.mark.parametrize("attention", ["hvp_manual", "hvp_semi_manual", "flashbob"])
def test_python_runners_forward_attention(runner, attention):
    module = importlib.import_module(runner.stem)
    cli = [
        "--exp_path", "unused", "--per_device_batch_size", "2",
        "--attn_implementation", attention,
    ]
    if runner.stem.endswith("text_compression"):
        cli += ["--dataset_path", "unused"]
    args, = HfArgumentParser(module.ExperimentArgs).parse_args_into_dataclasses(cli)
    assert args.attn_implementation == attention
    tree = ast.parse(runner.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "GradMemGPTConfig"]
    assert calls
    for call in calls:
        value = next(kw.value for kw in call.keywords if kw.arg == "attn_implementation")
        assert ast.unparse(value) == "args.attn_implementation"


@pytest.mark.parametrize("runner", sorted(ROOT.glob("run_gradmemgpt_on_*.py")), ids=lambda p: p.name)
@pytest.mark.parametrize("objective", ["reconstruction", "energy"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_dataset_collators_with_flashbob(runner, objective, device):
    _require_bob()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    module = importlib.import_module(runner.stem)
    if runner.stem.endswith("mqar"):
        inputs, labels = _batch()
        batch = module.collate_fn([
            {**{key: value[i] for key, value in inputs.items()}, "labels": labels[i]}
            for i in range(2)
        ])
    else:
        tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "red": 2,
                                        "blue": 3, "where": 4, "Answer": 5,
                                        ":": 6, "text": 7}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                            pad_token="[PAD]", unk_token="[UNK]")
        examples = [{"context": "red blue red", "query": "where ", "target": "blue", "split": "valid"},
                    {"context": "blue", "query": "where ", "target": "red", "split": "valid"}]
        if runner.stem.endswith("text_compression"):
            batch = module.collate_fn(examples, tokenizer, [], [], max_context_length=8)
        else:
            batch = module.collate_fn(examples, tokenizer)
    assert (batch["labels"] != -100).any()
    model = _model("gpt2", "flashbob", objective).to(device)
    batch["input_ids"] = {key: value.to(device) for key, value in batch["input_ids"].items()}
    batch["labels"] = batch["labels"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        loss = model(**batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.mem.grad is not None and torch.isfinite(model.mem.grad).all()
