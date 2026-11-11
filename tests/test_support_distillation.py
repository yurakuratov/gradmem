import pytest
import torch
from transformers import LlamaConfig

from support_distillation import (
    SupportDistillationConfig, SupportDistillationGradMemGPT, build_episodes,
)
from grad_memgpt import get_backbone


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
    for kwargs in [dict(n_support_questions=0), dict(inner_distill_target_weight=-1),
                   dict(inner_reconstruction_weight=-1), dict(inner_reconstruction_weight=float('nan'))]:
        with pytest.raises(ValueError):
            config(**kwargs)


def test_fused_context_ce_and_support_kl():
    model = SupportDistillationGradMemGPT(config(inner_reconstruction_weight=0.4))
    inputs, labels = batch()
    inputs['context_input_ids'] = torch.tensor([[1, 2, 3, 13, 0]])
    calls = []
    def capture(module, args, kwargs, output):
        if kwargs.get('inputs_embeds') is not None and kwargs['inputs_embeds'].size(0) == 3:
            calls.append((kwargs, model.read_model.get_output_embeddings()(output.last_hidden_state)))
    hook = get_backbone(model.read_model).register_forward_hook(capture, with_kwargs=True)
    output = model(inputs, labels=labels, return_mem=True)
    hook.remove()
    # K fused updates and one final diagnostic WRITE. No extra CE forward.
    assert len(calls) == model.K + 1
    ce_losses = []
    for kwargs, logits in calls:
        assert kwargs['inputs_embeds'].size(0) == 3
        offset = model.n_mem_tokens + 2 * model.n_ctrl_tokens
        context = inputs['context_input_ids']
        ce_losses.append(torch.nn.functional.cross_entropy(
            logits[2:, offset - 1:offset - 1 + context.size(1)].transpose(1, 2),
            context.masked_fill(context.eq(0), -100)))
    stats = output['inner_loop_stats']
    torch.testing.assert_close(stats['inner_reconstruction_loss'].squeeze(), ce_losses[-2])
    torch.testing.assert_close(stats['inner_loss'], stats['inner_support_kl_loss'] +
                               0.4 * stats['inner_reconstruction_loss'])
    output['loss'].backward()
    assert torch.isfinite(model.mem.grad).all()
    assert model.mem.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(inputs, labels=labels)['loss'])


def test_all_memory_backends_second_order_and_batch_independence():
    for backend in ('prefix', 'lora', 'kv_cache'):
        torch.manual_seed(7)
        model = SupportDistillationGradMemGPT(config(memory_backend=backend, inner_reconstruction_weight=0.3))
        inputs, labels = batch()
        output = model(inputs, labels=labels, return_mem=True)
        # Test higher-order WRITE contribution, not just the direct READ gradient.
        embeddings = model.read_model.get_input_embeddings().weight
        support_grad = torch.autograd.grad(output['loss'], embeddings, retain_graph=True)[0]
        assert support_grad[7:13].abs().sum() > 0, backend
        output['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        single = model(inputs, labels=labels, return_mem=True)
        doubled = model({k: v.repeat(2, 1) for k, v in inputs.items()},
                        labels=labels.repeat(2, 1), return_mem=True)
        key = {'prefix': 'mem', 'lora': 'lora_mem', 'kv_cache': 'kv_mem'}[backend]
        def check(a, b):
            if isinstance(a, dict):
                for k in a:
                    check(a[k], b[k])
            elif isinstance(a, tuple):
                for x, y in zip(a, b):
                    check(x, y)
            else:
                torch.testing.assert_close(b, a.expand_as(b), atol=1e-6, rtol=1e-4)
        check(single[key], doubled[key])


def test_energy_shaping_and_layerwise_energy():
    for objective in ('energy', 'energy_with_reconstruction'):
        model = SupportDistillationGradMemGPT(config(
            write_objective=objective, use_layerwise_energy=True,
            energy_rank_weight=0.1, energy_traj_weight=0.1, energy_anchor_weight=0.01,
            inner_reconstruction_weight=0.2))
        inputs, labels = batch()
        output = model(inputs, labels=labels)
        stats = output['inner_loop_stats']
        for key in ('inner_support_kl_loss', 'inner_energy_loss', 'energy_rank_loss',
                    'energy_traj_loss', 'energy_anchor_loss'):
            assert key in stats and torch.isfinite(stats[key]), key
        expected = stats['inner_support_kl_loss'] + 0.2 * stats['inner_reconstruction_loss']
        expected = expected + stats['inner_energy_loss']
        if objective == 'energy_with_reconstruction':
            expected = expected + stats['inner_reconstruction_loss']
        torch.testing.assert_close(stats['inner_loss'], expected)
        output['loss'].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if 'energy_head' in n)


def test_native_auxiliaries_sampling_and_gradient_modes():
    variants = [
        dict(write_objective='energy', energy_memory_search_weight=0.1,
             energy_memory_search_num_samples=2, energy_memory_search_use_best_for_next_step=True),
        dict(write_objective='energy', energy_memory_search_weight=0.1,
             energy_memory_search_num_samples=2, energy_memory_search_use_gain_weighting=True),
        dict(write_objective='energy', lipschitz_weight=0.1),
        dict(write_objective='energy', intermediate_read_weight=0.2, step_alignment_weight=0.1,
             memory_alignment_weight=0.1, orthogonal_loss_weight=0.1, ivan_loss_weight=0.1,
             add_inner_loss_to_outer=True, inner_loss_weight=0.1, memory_noise_sigma=0.01),
        dict(K_sampling=True, K_min=1, K_max=3, ilr_min=0.01, ilr_max=0.1, ilr_sample_unique=True),
        dict(use_write_head=True),
        dict(use_mem_proj=True, mem_proj_mode='per_sample'),
        dict(last_K_second_order=1),
        dict(grad_mode='first'), dict(grad_mode='none'), dict(use_adam=True),
    ]
    for variant in variants:
        cfg = config(**{k: v for k, v in variant.items() if k != 'grad_mode'})
        if 'grad_mode' in variant:
            cfg.grad_mode = variant['grad_mode']
            cfg.last_K_second_order = 0
        model = SupportDistillationGradMemGPT(cfg)
        inputs, labels = batch()
        output = model(inputs, labels=labels)
        assert torch.isfinite(output['loss']), variant
        output['loss'].backward()
        assert any(p.grad is not None for p in model.parameters()), variant


def test_disabled_kl_matches_native_gradmem():
    from grad_memgpt import GradMemGPT
    variants = [(backend, objective) for backend in ('prefix', 'lora', 'kv_cache')
                for objective in ('reconstruction', 'energy', 'energy_with_reconstruction')]
    for backend, objective in variants:
        cfg = config(memory_backend=backend, write_objective=objective,
                     inner_distill_target_weight=0,
                     inner_reconstruction_weight=1 if objective == 'reconstruction' else 0)
        baseline = GradMemGPT(cfg).eval()
        model = SupportDistillationGradMemGPT(cfg).eval()
        model.load_state_dict(baseline.state_dict())
        inputs, labels = batch()
        expected = baseline(inputs, labels=labels, return_mem=True)
        actual = model(inputs, labels=labels, return_mem=True)
        torch.testing.assert_close(expected['loss'], actual['loss'])
        torch.testing.assert_close(expected['predictions'], actual['predictions'])


def test_separate_writer_and_backend_variants():
    writer = LlamaConfig(vocab_size=23, hidden_size=24, intermediate_size=48,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, pad_token_id=0, use_cache=False)
    variants = [dict(separate_writer=True, writer_base_config=writer),
                dict(separate_writer=True, writer_base_config=writer, write_objective='energy',
                     energy_traj_weight=0.1),
                dict(memory_backend='lora', lora_mem_placement='target_modules',
                     lora_mem_target_modules='q_proj,v_proj'),
                dict(memory_backend='kv_cache', distill_logits_target_weight=0.2,
                     distill_hidden_target_weight=0.1)]
    for variant in variants:
        model = SupportDistillationGradMemGPT(config(inner_reconstruction_weight=0.2, **variant))
        inputs, labels = batch()
        output = model(inputs, labels=labels)
        output['loss'].backward()
        assert torch.isfinite(output['loss'])


def test_write_lora_with_frozen_backbone():
    pytest.importorskip('peft')
    model = SupportDistillationGradMemGPT(config(use_write_lora=True, freeze_backbone=True))
    inputs, labels = batch()
    output = model(inputs, labels=labels)
    output['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if 'lora_' in name)
