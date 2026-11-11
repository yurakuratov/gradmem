"""Energy objectives and shaping across every fast-memory representation."""
import unittest

import torch
from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig

from grad_memgpt import GradMemGPT, GradMemGPTConfig
from support_distillation import SupportDistillationConfig, SupportDistillationGradMemGPT


def backbone(family):
    shared = dict(vocab_size=31, pad_token_id=0, use_cache=False)
    if family == 'gpt2':
        return GPT2Config(n_embd=16, n_layer=2, n_head=2, n_positions=64,
                          resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, **shared)
    if family == 'gpt_neox':
        return GPTNeoXConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, max_position_embeddings=64, **shared)
    return LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=2, **shared)


def batch():
    inputs = dict(context_input_ids=torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
                  query_input_ids=torch.tensor([[8, 9, 10], [11, 12, 13]]),
                  support_input_ids=torch.tensor([[14, 15, 16], [17, 18, 19],
                                                  [20, 21, 22], [23, 24, 25]]),
                  support_labels=torch.tensor([[-100, -100, 16], [-100, -100, 19],
                                                [-100, -100, 22], [-100, -100, 25]]))
    return inputs, torch.tensor([[-100, -100, 10], [-100, -100, 13]])


class EnergyBackendTests(unittest.TestCase):
    def test_energy_head_receives_meta_gradients_without_shaping(self):
        # With shaping disabled, head gradients can only arrive through WRITE.
        for backend in ('prefix', 'lora', 'kv_cache'):
            with self.subTest(backend=backend):
                model = GradMemGPT(GradMemGPTConfig(base_config=backbone('llama'),
                    memory_backend=backend, n_mem_tokens=2, K=2, lr=0.2,
                    grad_mode='second', attn_implementation='eager', write_objective='energy'))
                inputs, labels = batch()
                output = model(inputs, labels=labels)
                output['loss'].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                    for name, p in model.named_parameters() if 'energy_head' in name))

    def test_objectives_shaping_and_second_order_matrix(self):
        for family in ('gpt2', 'gpt_neox', 'llama'):
            for backend, placement in (('prefix', 'between_layers'), ('lora', 'between_layers'),
                                       ('lora', 'target_modules'), ('kv_cache', 'between_layers')):
                for objective in ('energy', 'energy_with_reconstruction'):
                    for support in (False, True):
                        with self.subTest(family=family, backend=backend, placement=placement,
                                          objective=objective, support=support):
                            cfg_cls = SupportDistillationConfig if support else GradMemGPTConfig
                            model_cls = SupportDistillationGradMemGPT if support else GradMemGPT
                            cfg = cfg_cls(base_config=backbone(family), memory_backend=backend,
                                lora_mem_placement=placement, lora_mem_r=2, lora_mem_alpha=4,
                                n_mem_tokens=2, K=2, lr=0.1, grad_mode='second',
                                attn_implementation='eager', write_objective=objective,
                                use_layerwise_energy=objective == 'energy_with_reconstruction',
                                energy_rank_weight=0.1, energy_anchor_weight=0.01,
                                energy_traj_weight=0.1)
                            model = model_cls(cfg)
                            inputs, labels = batch()
                            output = model(inputs, labels=labels, return_mem=True)
                            self.assertTrue(torch.isfinite(output['loss']))
                            output['loss'].backward()
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for name, p in model.named_parameters() if 'energy_head' in name))
                            memory_names = {'prefix': ('mem',), 'lora': ('lora_mem_A0', 'lora_mem_B0'),
                                            'kv_cache': ('kv_mem_K0', 'kv_mem_V0')}[backend]
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for name, p in model.named_parameters()
                                if name == 'mem' or any(name.startswith(key) for key in memory_names)))
                            self.assertTrue(all(torch.isfinite(p.grad).all()
                                for p in model.parameters() if p.grad is not None))

    def test_detached_shaping_candidates_and_padding(self):
        for backend in ('prefix', 'lora', 'kv_cache'):
            with self.subTest(backend=backend):
                model = GradMemGPT(GradMemGPTConfig(base_config=backbone('llama'),
                    memory_backend=backend, n_mem_tokens=2, K=1, grad_mode='second',
                    attn_implementation='eager', write_objective='energy'))
                inputs, _ = batch()
                impl = model.memory_backend_impl
                state, initial = impl.init_memory_state(2)
                ctx = impl.prepare_batch(inputs['context_input_ids'], inputs['query_input_ids'], 0)
                losses = model._compute_energy_landscape_losses(
                    impl, state, initial, ctx, compute_rank=True, compute_anchor=True)
                memories = impl.inner_params(state)
                grads = torch.autograd.grad(losses['rank_loss'] + losses['anchor_loss'],
                                            memories, allow_unused=True, retain_graph=True)
                self.assertTrue(all(grad is None for grad in grads))
                write = impl.build_write_inputs(state, ctx)
                hidden = torch.randn(2, write['context_start'] + 5, 16)
                expected = model._compute_write_energy(hidden, write)
                changed = hidden.clone()
                context = changed[:, write['context_start']:]
                context[~write['mask']] = 10000
                torch.testing.assert_close(expected, model._compute_write_energy(changed, write))

    def test_lipschitz_penalty_for_all_backends(self):
        for backend in ('prefix', 'lora', 'kv_cache'):
            with self.subTest(backend=backend):
                model = GradMemGPT(GradMemGPTConfig(base_config=backbone('llama'),
                    memory_backend=backend, n_mem_tokens=2, K=1, grad_mode='second',
                    attn_implementation='eager', write_objective='energy',
                    lipschitz_weight=0.1, lipschitz_constraint=0))
                inputs, labels = batch()
                output = model(inputs, labels=labels)
                self.assertTrue(torch.isfinite(output['loss']))
                output['loss'].backward()


if __name__ == '__main__':
    unittest.main()
