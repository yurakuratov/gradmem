import pytest
import torch
from transformers import LlamaConfig

from grad_memgpt import GradMemGPT, GradMemGPTConfig
from self_distillation import (
    DISTILLATION_DEFAULTS, SelfDistillationConfig, SelfDistillationGradMemGPT,
)


def config(cls=SelfDistillationConfig, **kwargs):
    base = LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2,
                       num_key_value_heads=2, pad_token_id=0, use_cache=False,
                       max_position_embeddings=64)
    return cls(base_config=base, n_mem_tokens=2, K=1, lr=0.1,
               grad_mode='second', attn_implementation='eager', **kwargs)


def batch():
    inputs = {'context_input_ids': torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
              'query_input_ids': torch.tensor([[6, 7, 8, 0], [9, 10, 11, 12]])}
    labels = torch.tensor([[-100, -100, 8, -100], [-100, -100, 11, 12]])
    return inputs, labels


def test_config_validation_and_serialization(tmp_path):
    cfg = config(distill_hidden_target_weight=0.7, energy_rank_weight=0.1)
    cfg.save_pretrained(tmp_path)
    restored = SelfDistillationConfig.from_pretrained(tmp_path)
    assert restored.distill_hidden_target_weight == 0.7
    assert restored.energy_rank_weight == 0.1
    for name in DISTILLATION_DEFAULTS:
        for value in (-1, float('nan'), float('inf')):
            with pytest.raises(ValueError, match=name):
                config(**{name: value})


@pytest.mark.parametrize('backend', ['prefix', 'lora', 'kv_cache'])
@pytest.mark.parametrize('objective', ['reconstruction', 'energy', 'energy_with_reconstruction'])
def test_teacher_detached_and_second_order_gradients(backend, objective):
    if backend != 'prefix' and objective != 'reconstruction':
        pytest.skip('Existing energy objectives require prefix memory')
    torch.manual_seed(7)
    model = SelfDistillationGradMemGPT(config(
        memory_backend=backend, write_objective=objective,
        read_ce_weight=0.3, distill_logits_query_weight=0.2,
        distill_logits_target_weight=0.4, distill_hidden_query_weight=0.6,
        distill_hidden_target_weight=0.8,
        energy_rank_weight=0.1 if objective != 'reconstruction' else 0.0,
    ))
    inputs, labels = batch()
    teacher = model._prepare_outer_supervision(inputs, labels)
    assert not teacher['logits'].requires_grad
    assert all(not h.requires_grad and h.grad_fn is None for h in teacher['hidden_states'])
    assert len(teacher['hidden_states']) == 2
    output = model(inputs, labels=labels, return_mem=True)
    stats = output['inner_loop_stats']
    for name in DISTILLATION_DEFAULTS:
        if name != 'read_ce_weight':
            assert torch.isfinite(stats[name.removesuffix('_weight') + '_loss'])
    output['loss'].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    if backend == 'prefix':
        assert model.mem.grad is not None and model.mem.grad.abs().sum() > 0
        if objective != 'reconstruction':
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.energy_head.parameters())


def test_disabled_matches_original_and_ce_weight_is_additive():
    torch.manual_seed(2)
    original = GradMemGPT(config(GradMemGPTConfig))
    model = SelfDistillationGradMemGPT(config())
    model.load_state_dict(original.state_dict())
    inputs, labels = batch()
    expected = original(inputs, labels=labels)
    actual = model(inputs, labels=labels)
    torch.testing.assert_close(actual['loss'], expected['loss'])
    torch.testing.assert_close(actual['predictions'], expected['predictions'])
    model.read_ce_weight = 0.25
    weighted = model(inputs, labels=labels)
    torch.testing.assert_close(weighted['loss'], actual['loss'] - 0.75 * actual['inner_loop_stats']['target_loss'])
    assert model._prepare_outer_supervision(inputs, None) is None


def test_teacher_padding_and_causal_alignment():
    model = SelfDistillationGradMemGPT(config(distill_logits_target_weight=1,
                                             distill_hidden_target_weight=1)).eval()
    inputs, labels = batch()
    teacher = model._prepare_outer_supervision(inputs, labels)
    for i, context in enumerate(inputs['context_input_ids']):
        context = context[context.ne(0)]
        query = inputs['query_input_ids'][i]
        full = torch.cat([context, query])[None]
        with torch.no_grad():
            direct = model.read_model(input_ids=full, attention_mask=full.ne(0),
                                      output_hidden_states=True, return_dict=True)
        start = context.numel()
        torch.testing.assert_close(teacher['logits'][i], direct.logits[0, start - 1:start + len(query) - 1])
        for hidden, expected in zip(teacher['hidden_states'], direct.hidden_states[1:]):
            torch.testing.assert_close(hidden[i, query.ne(0)], expected[0, start:start + len(query)][query.ne(0)])


def test_masked_mean_and_equal_layer_mse():
    model = SelfDistillationGradMemGPT(config(distill_hidden_target_weight=1))
    from types import SimpleNamespace
    inputs, labels = batch()
    # Prefix has two memory positions, then four query/target positions.
    hidden = [torch.zeros(2, 6, 16), torch.ones(2, 6, 16), torch.full((2, 6, 16), 3.)]
    read = SimpleNamespace(hidden_states=hidden)
    supervision = {'hidden_states': (torch.zeros(2, 4, 16), torch.zeros(2, 4, 16))}
    loss, stats = model._outer_supervision_loss(
        supervision, read, {'logits_start': 1}, inputs, labels, torch.tensor(2.))
    torch.testing.assert_close(loss, torch.tensor(5.))  # (1**2 + 3**2) / 2
    torch.testing.assert_close(stats['distill_hidden_target_loss'], torch.tensor(5.))


@pytest.mark.parametrize('backend', ['prefix', 'lora', 'kv_cache'])
def test_logit_kl_alignment_masks_and_teacher_stop_gradient(backend):
    from types import SimpleNamespace
    import torch.nn.functional as F
    model = SelfDistillationGradMemGPT(config(memory_backend=backend,
                                             distill_logits_query_weight=0.2,
                                             distill_logits_target_weight=0.7))
    inputs, labels = batch()
    start = 2 if backend == 'prefix' else 0
    first = 0 if start else 1
    logits = torch.randn(2, 4 + start, 23, requires_grad=True)
    teacher = torch.randn(2, 4, 23, requires_grad=True)
    supervision = {'logits': teacher, 'logits_valid': torch.ones(2, 4, dtype=torch.bool),
                   'hidden_states': None}
    extra, stats = model._outer_supervision_loss(
        supervision, SimpleNamespace(logits=logits), {'logits_start': max(0, start - 1)},
        inputs, labels, torch.tensor(1.))
    token_kl = F.kl_div(logits[:, start + first - 1:start + 3].log_softmax(-1),
                        teacher[:, first:].detach().softmax(-1), reduction='none').sum(-1)
    expected = 0
    for region, weight in [('query', 0.2), ('target', 0.7)]:
        mask = inputs['query_input_ids'].ne(0) & (labels.eq(-100) if region == 'query' else labels.ne(-100))
        value = token_kl[mask[:, first:]].mean()
        torch.testing.assert_close(stats[f'distill_logits_{region}_loss'], value)
        expected = expected + weight * value
    torch.testing.assert_close(extra, expected)
    extra.backward()
    assert logits.grad is not None and teacher.grad is None


def test_checkpoint_roundtrip_and_inference_without_teacher(tmp_path):
    model = SelfDistillationGradMemGPT(config(distill_hidden_target_weight=0.3)).eval()
    model.save_pretrained(tmp_path)
    restored = SelfDistillationGradMemGPT.from_pretrained(tmp_path).eval()
    assert restored.distill_hidden_target_weight == 0.3
    inputs, labels = batch()
    assert restored._prepare_outer_supervision(inputs, None) is None
    torch.testing.assert_close(model(inputs, labels=labels)['loss'],
                               restored(inputs, labels=labels)['loss'])


def test_runner_inherits_and_parses_all_gradmem_options():
    from dataclasses import fields
    from transformers import HfArgumentParser
    from run_gradmemgpt_on_kv_retrieval import ExperimentArgs
    from run_self_distillation_on_kv_retrieval import SelfDistillationArgs
    assert {f.name for f in fields(ExperimentArgs)} <= {f.name for f in fields(SelfDistillationArgs)}
    args, = HfArgumentParser(SelfDistillationArgs).parse_args_into_dataclasses([
        '--exp_path', 'runs/test', '--per_device_batch_size', '2',
        '--energy_rank_weight', '0.25', '--memory_backend', 'lora',
        '--write_objective', 'reconstruction', '--distill_logits_target_weight', '0.7',
        '--distill_hidden_query_weight', '0.2', '--read_ce_weight', '0.9',
    ])
    assert args.energy_rank_weight == 0.25
    assert args.memory_backend == 'lora'
    assert args.distill_logits_target_weight == 0.7
    assert args.distill_hidden_query_weight == 0.2
    assert args.read_ce_weight == 0.9
