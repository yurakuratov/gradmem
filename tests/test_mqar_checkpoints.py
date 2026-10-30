import json
from pathlib import Path
import pickle
import random

import numpy as np
import pytest
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from run_gradmemgpt_on_mqar import CustomTrainer, StopOnMetricValue


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None):
        logits = input_ids.float() * self.weight
        return {'loss': logits.square().mean(), 'logits': logits}


@pytest.mark.parametrize('partial_numpy_core', [False, True])
def test_safe_numpy_rng_resume(tmp_path, monkeypatch, partial_numpy_core):
    rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'cpu': torch.random.get_rng_state(),
    }
    rng_file = tmp_path / 'rng_state.pth'
    torch.save(rng_state, rng_file)
    safe_globals_before = torch.serialization.get_safe_globals().copy()
    with pytest.raises(pickle.UnpicklingError, match='Unsupported global'):
        torch.load(rng_file, weights_only=True)

    # Transformers 4.51 loads RNG state without its own NumPy allowlist.
    def load_rng_state(self, checkpoint):
        state = torch.load(Path(checkpoint) / 'rng_state.pth', weights_only=True)
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.random.set_rng_state(state['cpu'])

    monkeypatch.setattr(Trainer, '_load_rng_state', load_rng_state)
    trainer = CustomTrainer(
        model=TinyModel(),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    if partial_numpy_core:
        from types import SimpleNamespace

        monkeypatch.setattr(np, '_core', SimpleNamespace(), raising=False)
    trainer._load_rng_state(str(tmp_path))

    assert random.getstate() == rng_state['python']
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == rng_state['numpy'][0]
    np.testing.assert_array_equal(restored_numpy[1], rng_state['numpy'][1])
    assert restored_numpy[2:] == rng_state['numpy'][2:]
    assert torch.equal(torch.random.get_rng_state(), rng_state['cpu'])
    assert torch.serialization.get_safe_globals() == safe_globals_before
    with pytest.raises(pickle.UnpicklingError, match='Unsupported global'):
        torch.load(rng_file, weights_only=True)


def test_last_checkpoint_with_older_trainer_state(tmp_path, monkeypatch):
    trainer = CustomTrainer(
        model=TinyModel(),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
        save_last_checkpoint=True,
    )
    # Transformers 4.51 has no best_global_step field.
    from types import SimpleNamespace

    trainer.state = SimpleNamespace(best_model_checkpoint='checkpoint-1', best_metric=0.9)

    def save_checkpoint(self):
        assert self.state.best_model_checkpoint is None
        assert self.state.best_metric == 0.9

    monkeypatch.setattr(Trainer, '_save_checkpoint', save_checkpoint)
    trainer._save_checkpoint()


@pytest.mark.parametrize('save_last', [False, True])
@pytest.mark.parametrize('stop_mode', ['max_steps', 'metric', 'patience'])
def test_checkpoint_retention_and_final_state(tmp_path, save_last, stop_mode):
    # Evaluation peaks at step 2, before the off-cadence stop at step 3.
    scores = iter([0.75, 0.8, 0.9, 0.7])
    callbacks = []
    saved_steps = []

    class CheckRetention(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            saved_steps.append(state.global_step)
            if save_last:
                assert [path.name for path in tmp_path.glob('checkpoint-*')] == [
                    f'checkpoint-{state.global_step}'
                ]

    callbacks.append(CheckRetention())
    if stop_mode == 'metric':
        callbacks.append(StopOnMetricValue('token_accuracy', 0.7, higher_is_better=False))
    elif stop_mode == 'patience':
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=3 if stop_mode == 'max_steps' else 8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy='steps',
        eval_steps=1,
        eval_on_start=True,
        save_strategy='steps',
        save_steps=2,
        save_total_limit=1,
        metric_for_best_model='token_accuracy',
        greater_is_better=True,
        load_best_model_at_end=not save_last,
        report_to=[],
        disable_tqdm=True,
    )
    data = [{'input_ids': torch.tensor([1.0]), 'labels': torch.tensor([0.0])}] * 8
    trainer = CustomTrainer(
        model=TinyModel(),
        args=args,
        save_last_checkpoint=save_last,
        train_dataset=data,
        eval_dataset=data,
        compute_metrics=lambda _: {'token_accuracy': next(scores)},
        callbacks=callbacks,
    )
    trainer.train()

    assert trainer.state.global_step == 3
    expected_step = 3 if save_last else 2
    assert [path.name for path in tmp_path.glob('checkpoint-*')] == [f'checkpoint-{expected_step}']
    checkpoint = tmp_path / f'checkpoint-{expected_step}'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == expected_step
    assert (checkpoint / 'optimizer.pt').exists()
    assert (checkpoint / 'scheduler.pt').exists()
    assert trainer.state.best_metric is not None
    if save_last:
        assert saved_steps == [2, 3]
        from safetensors.torch import load_file

        saved = load_file(str(checkpoint / 'model.safetensors'))
        assert torch.equal(saved['weight'], trainer.model.weight.detach().cpu())


@pytest.mark.parametrize('resume_latest', [False, True])
@pytest.mark.parametrize('stop_mode', ['max_steps', 'metric', 'patience'])
def test_resume_preserves_source_and_saves_final_state(tmp_path, resume_latest, stop_mode):
    data = [{'input_ids': torch.tensor([1.0]), 'labels': torch.tensor([0.0])}] * 8
    initial_args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=2,
        per_device_train_batch_size=1,
        eval_strategy='steps',
        eval_steps=1,
        save_steps=2,
        save_total_limit=1,
        metric_for_best_model='token_accuracy',
        greater_is_better=True,
        report_to=[],
        disable_tqdm=True,
    )
    initial_scores = iter([0.4, 0.5])
    CustomTrainer(
        model=TinyModel(), args=initial_args, train_dataset=data,
        eval_dataset=data, compute_metrics=lambda _: {'token_accuracy': next(initial_scores)},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if stop_mode == 'patience' else [],
    ).train()
    source = tmp_path / 'checkpoint-2'
    assert Path(json.loads((source / 'trainer_state.json').read_text())['best_model_checkpoint']) == source
    original_files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    scores = iter([0.9, 0.8, 0.7])
    saved_steps = []

    class CheckRetention(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            saved_steps.append(state.global_step)
            assert {path.name for path in tmp_path.glob('checkpoint-*')} == {
                'checkpoint-2', f'checkpoint-{state.global_step}',
            }
            assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == original_files

    callbacks = [CheckRetention()]
    if stop_mode == 'metric':
        callbacks.append(StopOnMetricValue('token_accuracy', 0.7, higher_is_better=False))
    elif stop_mode == 'patience':
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    resumed_args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=5 if stop_mode == 'max_steps' else 8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy='steps',
        eval_steps=1,
        save_steps=2,
        save_total_limit=1,
        metric_for_best_model='token_accuracy',
        greater_is_better=True,
        load_best_model_at_end=False,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = CustomTrainer(
        model=TinyModel(),
        args=resumed_args,
        save_last_checkpoint=True,
        train_dataset=data,
        eval_dataset=data,
        compute_metrics=lambda _: {'token_accuracy': next(scores)},
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=True if resume_latest else str(source))

    assert trainer.state.global_step == 5
    assert saved_steps == [4, 5]
    assert {path.name for path in tmp_path.glob('checkpoint-*')} == {'checkpoint-2', 'checkpoint-5'}
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == original_files
    final = tmp_path / 'checkpoint-5'
    assert json.loads((final / 'trainer_state.json').read_text())['global_step'] == 5
    assert (final / 'scheduler.pt').exists()
    optimizer_state = torch.load(final / 'optimizer.pt', weights_only=True)
    assert next(iter(optimizer_state['state'].values()))['step'].item() == 5

    from safetensors.torch import load_file

    assert torch.equal(load_file(str(final / 'model.safetensors'))['weight'], trainer.model.weight.detach().cpu())
