import pytest
import torch
from transformers import LlamaConfig

from support_distillation import (
    SupportDistillationConfig, SupportDistillationGradMemGPT, build_episodes,
)


def config(**kwargs):
    base = LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=2, pad_token_id=0, use_cache=False)
    return SupportDistillationConfig(base_config=base, n_mem_tokens=2, K=2,
                                     grad_mode='second', attn_implementation='eager', **kwargs)


def batch():
    inputs = dict(context_input_ids=torch.tensor([[1, 2, 3]]),
                  query_input_ids=torch.tensor([[4, 5, 6]]),
                  support_input_ids=torch.tensor([[7, 8, 9], [10, 11, 12]]),
                  support_labels=torch.tensor([[-100, -100, 9], [-100, -100, 12]]))
    return inputs, torch.tensor([[-100, -100, 6]])


def test_grouping_no_leakage_and_reproducibility():
    rows = [dict(context='context', question=f'q{i}', answers={'text': [f'a{i}']})
            for i in range(4)]
    episodes = build_episodes(rows, 2, 42)
    assert episodes.to_list() == build_episodes(rows, 2, 42).to_list()
    assert len(episodes) == 4
    for episode in episodes:
        assert episode['query'] not in episode['support_queries']
        assert len(set(episode['support_queries'])) == 2
    with pytest.raises(ValueError):
        build_episodes(rows, 4, 42)


def test_meta_gradients_and_heldout_independence():
    torch.manual_seed(42)
    model = SupportDistillationGradMemGPT(config())
    inputs, labels = batch()
    output = model(inputs, labels=labels, return_mem=True)
    assert torch.isfinite(output['loss'])
    # A nonzero mixed derivative proves that the outer loss retains the KL WRITE path.
    support_embedding = model.read_model.get_input_embeddings().weight
    grad = torch.autograd.grad(output['loss'], support_embedding, retain_graph=True)[0]
    assert grad[7:13].abs().sum() > 0
    output['loss'].backward()
    assert model.mem.grad is not None and model.mem.grad.abs().sum() > 0
    changed = dict(inputs, query_input_ids=torch.tensor([[13, 14, 15]]))
    altered = model(changed, labels=torch.tensor([[-100, -100, 15]]), return_mem=True)
    torch.testing.assert_close(output['mem'], altered['mem'])
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(inputs, labels=labels)['loss'])


def test_batch_size_independent_inner_updates():
    model = SupportDistillationGradMemGPT(config()).eval()
    inputs, labels = batch()
    single = model(inputs, labels=labels, return_mem=True)['mem']
    doubled = {k: v.repeat(2, 1) for k, v in inputs.items()}
    result = model(doubled, labels=labels.repeat(2, 1), return_mem=True)['mem']
    torch.testing.assert_close(result, single.expand_as(result))


def test_validation_and_serialization(tmp_path):
    cfg = config(inner_distill_target_weight=0.7, inner_reconstruction_weight=0.4)
    cfg.save_pretrained(tmp_path)
    restored = SupportDistillationConfig.from_pretrained(tmp_path)
    assert restored.inner_distill_target_weight == 0.7
    assert restored.inner_reconstruction_weight == 0.4
    assert restored.n_support_questions == 2
    for kwargs in [dict(use_adam=True), dict(energy_rank_weight=0.1),
                   dict(n_support_questions=0), dict(inner_distill_target_weight=-1),
                   dict(inner_reconstruction_weight=-1), dict(inner_reconstruction_weight=float('nan'))]:
        with pytest.raises(ValueError):
            config(**kwargs)


def test_fused_context_ce_and_support_kl():
    model = SupportDistillationGradMemGPT(config(inner_reconstruction_weight=0.4))
    inputs, labels = batch()
    inputs['context_input_ids'] = torch.tensor([[1, 2, 3, 13, 0]])
    calls = []
    def capture(module, args, kwargs, output):
        calls.append((kwargs, output.logits))
    hook = model.read_model.register_forward_hook(capture, with_kwargs=True)
    output = model(inputs, labels=labels, return_mem=True)
    hook.remove()
    # Teacher, K fused forwards, held-out READ; no extra CE forward.
    assert len(calls) == model.K + 2
    ce_losses = []
    for kwargs, logits in calls[1:-1]:
        assert kwargs['inputs_embeds'].size(0) == 3
        offset = model.n_mem_tokens + 2 * model.n_ctrl_tokens
        context = inputs['context_input_ids']
        ce_losses.append(torch.nn.functional.cross_entropy(
            logits[2:, offset - 1:offset - 1 + context.size(1)].transpose(1, 2),
            context.masked_fill(context.eq(0), -100)))
    stats = output['inner_loop_stats']
    torch.testing.assert_close(stats['inner_reconstruction_loss'].squeeze(), torch.stack(ce_losses).mean())
    torch.testing.assert_close(stats['inner_loss'], stats['inner_support_kl_loss'] +
                               0.4 * stats['inner_reconstruction_loss'])
    output['loss'].backward()
    assert torch.isfinite(model.mem.grad).all()
    assert model.mem.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(inputs, labels=labels)['loss'])
