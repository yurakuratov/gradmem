import json
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

from analysis_tools import analyze, library, plots
from grad_memgpt import GradMemGPTConfig
from run_gradmemgpt_on_mqar import MQARGradMemGPT


def fake_checkpoint(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text("{}")
    checkpoint = directory / "model.safetensors"
    checkpoint.touch()
    return checkpoint


def test_checkpoint_resolution_requires_explicit_folder(tmp_path):
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    with pytest.raises(FileNotFoundError, match="Checkpoint folder"):
        library.resolve_checkpoint(tmp_path)
    assert library.resolve_checkpoint(checkpoint) == checkpoint / "model.safetensors"
    (checkpoint / "trainer_state.json").write_text(json.dumps({"best_model_checkpoint": "/old/path/checkpoint-5"}))
    with pytest.raises(FileNotFoundError, match="Checkpoint folder"):
        library.resolve_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="not a file"):
        library.resolve_checkpoint(checkpoint / "model.safetensors")


def test_selector_is_rejected():
    config = dict(kind="update_svd", update_svd={"steps": 2}, setups=[
        dict(label="test", runs=[{"path": "checkpoint-5", "selector": "best"}])])
    with pytest.raises(ValueError, match="checkpoint folder directly"):
        analyze.validate_config(config)


def test_basis_steps_two_three_anchor_at_m1():
    m0 = torch.zeros(2, 1, 3)
    m1 = m0 + torch.tensor([[[1., 0., 0.]]])
    m2 = m1 + torch.tensor([[[0., 1., 0.]]])
    m3 = m2 + torch.tensor([[[0., 0., 1.]]])
    center, x, y, trajectory = library.memory_basis([m0, m1, m2, m3], [2, 3])
    assert torch.equal(center, m1)
    assert torch.equal(x, m2 - m1)
    assert torch.equal(y, m3 - m2)
    np.testing.assert_allclose(trajectory[1:], [[0, 0], [1, 0], [1, 1]])


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_memory_basis_is_independent_of_relative_step_lengths(scale):
    m0 = torch.zeros(2, 1, 2048)
    x, y = m0.clone(), m0.clone()
    x[:, 0, 0] = 1
    y[:, 0, 1] = scale
    center, actual_x, actual_y, trajectory = library.memory_basis([m0, x, x + y], [1, 2])
    assert torch.equal(center, m0)
    assert torch.equal(actual_x, x)
    assert torch.equal(actual_y, x + y - x)
    np.testing.assert_allclose(trajectory, [[0, 0], [1, 0], [1, 1]], atol=1e-7)


@pytest.mark.parametrize("kind", ["zero", "collinear", "nonfinite"])
def test_memory_basis_rejects_invalid_example_with_diagnostics(kind):
    m0 = torch.zeros(2, 1, 3)
    x = m0 + torch.tensor([[[1., 0., 0.]]])
    y = m0 + torch.tensor([[[0., 1., 0.]]])
    y[1] = {"zero": 0., "collinear": x[1], "nonfinite": float("nan")}[kind]
    message = "Nonfinite WRITE trajectory" if kind == "nonfinite" else "two-dimensional plane"
    with pytest.raises(ValueError, match=message) as error:
        library.memory_basis([m0, x, x + y], [1, 2])
    assert "batch-local examples [1]" in str(error.value)


def test_cosine_average_is_not_cosine_of_average():
    alpha = beta = np.linspace(-1, 1, 5)
    x, y = np.meshgrid(alpha, beta)
    first = library.finite_difference_cosine(x, x, alpha, beta)
    second = library.finite_difference_cosine(10*y, -y, alpha, beta)
    np.testing.assert_allclose((first + second) / 2, 0)
    averaged = library.finite_difference_cosine((x+10*y)/2, (x-y)/2, alpha, beta)
    assert not np.allclose(averaged, 0)


def test_landscape_real_grads_settings():
    assert analyze.landscape_settings({}, {})["real_grads"] is False
    assert analyze.landscape_settings({"landscape": {"real_grads": True}}, {})["real_grads"] is True
    assert analyze.landscape_settings({"landscape": {"real_grads": True}},
                                      {"landscape": {"real_grads": False}})["real_grads"] is False
    for value in ("true", 1, None):
        with pytest.raises(ValueError, match="real_grads"):
            analyze.landscape_settings({"landscape": {"real_grads": value}}, {})


def test_landscape_separate_colorbars_settings():
    assert analyze.landscape_settings({}, {})["separate_colorbars"] is False
    assert analyze.landscape_settings({"landscape": {"separate_colorbars": True}}, {})["separate_colorbars"] is True
    for value in ("true", 1, None):
        with pytest.raises(ValueError, match="separate_colorbars"):
            analyze.landscape_settings({"landscape": {"separate_colorbars": value}}, {})
    with pytest.raises(ValueError, match="shared landscape.separate_colorbars"):
        analyze.landscape_settings({}, {"landscape": {"separate_colorbars": True}})


def test_landscape_include_cosine_sim_settings():
    assert analyze.landscape_settings({}, {})["include_cosine_sim"] is True
    assert analyze.landscape_settings({"landscape": {"include_cosine_sim": False}}, {})["include_cosine_sim"] is False
    for value in ("false", 0, None):
        with pytest.raises(ValueError, match="include_cosine_sim"):
            analyze.landscape_settings({"landscape": {"include_cosine_sim": value}}, {})
    with pytest.raises(ValueError, match="shared landscape.include_cosine_sim"):
        analyze.landscape_settings({}, {"landscape": {"include_cosine_sim": False}})


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_landscape_autograd_uses_example_mean_coordinate_derivatives(monkeypatch, batch_size):
    model = type("Model", (), {"device": "cpu"})()
    inputs = {"context_input_ids": torch.tensor([[1], [2], [4]])}
    labels = torch.zeros(3, 1)
    def trajectory(model, batch, steps):
        m0 = torch.zeros(len(batch["context_input_ids"]), 1, 2)
        return [m0, m0 + torch.tensor([[[2., 0.]]]), m0 + torch.tensor([[[2., 3.]]])]
    def score(model, batch, targets, memory, grad_enabled=False):
        a, b = memory[:, 0, 0] / 2, memory[:, 0, 1] / 3
        scale = batch["context_input_ids"][:, 0]
        return {"inner_loss": scale * a**3 + b, "outer_loss": a - scale * b}
    monkeypatch.setattr(analyze, "collect_trajectory", trajectory)
    monkeypatch.setattr(analyze, "score_memory", score)
    settings = dict(grid_size=3, alpha_range=[-1, 1], beta_range=[-1, 1])
    result = analyze.landscape_run(model, inputs, labels, dict(settings, real_grads=True), batch_size)
    a, b = np.meshgrid(result["alpha"], result["beta"])
    scale = 7 / 3
    inner = np.stack([3 * scale * a**2, np.ones_like(b)], axis=-1)
    outer = np.stack([np.ones_like(a), np.full_like(b, -scale)], axis=-1)
    np.testing.assert_allclose(result["alignment"], library.gradient_cosine(inner, outer), atol=1e-6)
    finite = analyze.landscape_run(model, inputs, labels, settings, batch_size)
    np.testing.assert_allclose(result["inner_loss"], finite["inner_loss"])
    np.testing.assert_allclose(result["outer_loss"], finite["outer_loss"])
    assert not np.allclose(result["alignment"], finite["alignment"])
    assert np.isnan(library.gradient_cosine(np.zeros((1, 2)), np.ones((1, 2)))).all()


def test_beneficial_outcomes_keep_harmful_candidates_as_zero_gain():
    base = torch.tensor([5., 4.])
    candidates = torch.tensor([[7., 1.], [3., 5.]])
    loss = library.beneficial_outcomes(base, candidates, [0, 1])
    torch.testing.assert_close(loss, torch.tensor([[5., 4.], [3., 1.]]))
    accuracy = library.beneficial_outcomes(base, candidates, [0, 1], True)
    torch.testing.assert_close(accuracy, torch.tensor([[5., 4.], [7., 5.]]))


def test_log_scale_matches_values_and_narrow_fallback_is_sparse():
    norm, ticks, formatter = plots.loss_scale([np.array([7.07, 14.10])])
    assert norm.vmin == 7.07 and norm.vmax == 14.10
    assert len(ticks) == 2
    assert formatter(ticks[0])


@pytest.mark.parametrize("objective", ["reconstruction", "energy", "energy_with_reconstruction"])
def test_fixed_memory_inner_and_context_free_read(objective):
    torch.manual_seed(2)
    base = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                       max_position_embeddings=32)
    config = GradMemGPTConfig(base_config=base, n_mem_tokens=2, K=2, lr=.2,
        grad_mode="none", attn_implementation="eager", write_objective=objective,
        write_reconstruction_weight=.3, write_energy_weight=2.,
        read_loss_alignment="query_position", mqar_vocab_size=32, mqar_dense_queries=True)
    model = MQARGradMemGPT(config).eval().requires_grad_(False)
    inputs = {"context_input_ids": torch.tensor([[2, 17, 3, 18], [4, 19, 5, 20]]),
              "query_input_ids": torch.tensor([[2, 3], [4, 5]])}
    labels = torch.tensor([[17, 18], [19, 20]])
    states = library.collect_trajectory(model, inputs, 2)
    metrics = library.score_memory(model, inputs, labels, states[-1])
    assert all(value.shape == (2,) and torch.isfinite(value).all() for value in metrics.values())
    memory = states[-1].clone().requires_grad_(True)
    differentiable = library.score_memory(model, inputs, labels, memory, grad_enabled=True)
    direction = torch.randn_like(memory)
    epsilon = .0005
    plus = library.score_memory(model, inputs, labels, memory + epsilon * direction)
    minus = library.score_memory(model, inputs, labels, memory - epsilon * direction)
    for key in ("inner_loss", "outer_loss"):
        torch.testing.assert_close(differentiable[key], metrics[key])
        grad = torch.autograd.grad(differentiable[key].mean(), memory, retain_graph=True)[0]
        expected_derivative = (plus[key].mean() - minus[key].mean()) / (2 * epsilon)
        torch.testing.assert_close((grad * direction).sum(), expected_derivative, atol=2e-4, rtol=.02)
    assert all(parameter.grad is None for parameter in model.parameters())
    landscape = analyze.landscape_run(model, inputs, labels,
        {"grid_size": 3, "real_grads": True, "alpha_range": [0, 1], "beta_range": [0, 1]}, 2)
    assert landscape["alignment"].shape == (3, 3)
    assert np.isfinite(landscape["alignment"]).all()
    np.testing.assert_allclose(landscape["outer_loss"][-1, -1], float(metrics["outer_loss"].mean()), atol=1e-6)
    backend = model.memory_backend_impl
    state, _ = backend.init_memory_state(2)
    state["mem_batch"] = states[-1]
    ctx = backend.prepare_batch(inputs["context_input_ids"], inputs["query_input_ids"], 0)
    with torch.no_grad():
        write = backend.build_write_inputs(state, ctx)
        if objective == "energy_with_reconstruction":
            _, ce, energy = model._run_energy_reconstruction_write_forward(write)
            torch.testing.assert_close(metrics["inner_loss"], .3 * ce + 2. * energy)
        elif objective == "energy":
            _, energy = model._run_energy_write_forward(write)
            torch.testing.assert_close(metrics["inner_loss"], energy)
        predictions = model(inputs, labels=None)["predictions"]
        expected = library.prediction_metrics(model, predictions, labels)
        torch.testing.assert_close(metrics["outer_loss"], expected["outer_loss"])
        changed = dict(inputs, context_input_ids=torch.ones_like(inputs["context_input_ids"]))
        changed_metrics = library.score_memory(model, changed, labels, states[-1])
        torch.testing.assert_close(metrics["outer_loss"], changed_metrics["outer_loss"])
    sweep = analyze.k_lr_sweep_run(model, inputs, labels,
                                   {"steps": [0, 2], "inner_lrs": [.1, .2]}, 1)
    assert all(value.shape == (2, 2) and np.isfinite(value).all() for value in sweep.values())
    for key in expected:
        np.testing.assert_allclose(sweep[key][1, 1], float(expected[key].mean()), atol=1e-6)
    assert model.K == model.config.K == 2
    assert model.lr == model.config.lr == .2
    svd = analyze.update_svd_run(model, inputs, labels, {"steps": 3}, 1)
    assert svd["singular_values_per_example"].shape == (2, 3)
    assert np.isfinite(svd["singular_values"]).all()
    assert model.K == model.config.K == 2
    projection = analyze.step_orth_proj_run(model, inputs, labels, {"steps": 2}, 1)
    assert projection["projection_ratio_per_example"].shape == (2, 2)
    assert np.isfinite(projection["projection_ratio"]).all()
    assert np.all((projection["projection_ratio"] >= 0) & (projection["projection_ratio"] <= 1))
    assert model.K == model.config.K == 2


def test_update_svd_uses_displacements_and_separate_examples():
    updates = torch.diag(torch.tensor([3., 2., 1.])).reshape(3, 1, 1, 3)
    updates = torch.cat([updates, -updates], dim=1)
    states = [torch.full((2, 1, 3), 10.)]
    for update in updates:
        states.append(states[-1] + update)
    values = library.update_singular_values(states)
    torch.testing.assert_close(values, torch.tensor([[3., 2., 1.], [3., 2., 1.]]))
    torch.testing.assert_close(library.update_singular_values([state + 100 for state in states]), values)
    assert library.retained_energy_rank([3, 2, 1], .95) == 3
    assert library.retained_energy_rank([3, 2, 1], .9) == 2
    assert library.retained_energy_rank([3, 0, 0], 1.) == 1
    assert library.retained_energy_rank([0, 0], .95) is None


def test_update_svd_reduced_shape_and_nonfinite():
    states = [torch.zeros(1, 1, 2)]
    for _ in range(3):
        states.append(states[-1] + 1)
    assert library.update_singular_values(states).shape == (1, 2)
    torch.testing.assert_close(library.update_singular_values([states[0], states[0]]), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="at least one"):
        library.update_singular_values(states[:1])
    with pytest.raises(ValueError, match="Nonfinite"):
        library.update_singular_values([states[0], states[0] + float("inf")])


def test_step_projection_ratios_full_memory_and_degenerate_spans():
    # Current directions: in-plane, perpendicular, partly in-plane, and opposite.
    current = torch.tensor([[1., 1., 0.], [0., 0., 1.], [.6, 0., .8], [-1., 0., 0.]],
                           dtype=torch.float64).reshape(4, 1, 3)
    first = torch.tensor([[[1., 0., 0.]]], dtype=torch.float64).expand_as(current)
    second = torch.tensor([[[0., 1., 0.]]], dtype=torch.float64).expand_as(current)
    m0 = torch.zeros_like(current)
    states = [m0, current, current + first, current + first + second]
    expected = torch.tensor([[1.], [0.], [.6], [1.]], dtype=torch.float64)
    torch.testing.assert_close(library.step_projection_ratios(states), expected)
    scaled = [m0, current * 7, current * 7 + first * 1e-6,
              current * 7 + first * 1e-6 + second * 1e6]
    torch.testing.assert_close(library.step_projection_ratios(scaled), expected)
    shifted = [state + 10 for state in states]
    torch.testing.assert_close(library.step_projection_ratios(shifted), expected)
    # A line and a span with one zero direction must not invent a second axis.
    for last in (first * 2, torch.zeros_like(first)):
        result = library.step_projection_ratios([m0, current, current + first, current + first + last])
        torch.testing.assert_close(result[:, 0], torch.tensor([2**-.5, 0., .6, 1.], dtype=torch.float64))
    torch.testing.assert_close(library.step_projection_ratios([m0, current, current, current]),
                               torch.zeros(4, 1, dtype=torch.float64))


@pytest.mark.parametrize("proj_steps", [1, 2, 3, 7])
def test_step_projection_ratios_multiple_steps_and_invalid_updates(proj_steps):
    states = [torch.zeros(2, 2, 3)]
    torch.manual_seed(12)
    for _ in range(4 + proj_steps):
        states.append(states[-1] + torch.randn_like(states[-1]))
    ratios = library.step_projection_ratios(states, proj_steps).numpy()
    assert ratios.shape == (2, 4)
    updates = np.stack([(b - a).double().flatten(1).numpy() for a, b in zip(states, states[1:])], axis=1)
    for sample in range(2):
        for step in range(4):
            basis = updates[sample, step + 1:step + 1 + proj_steps].T
            current = updates[sample, step]
            projection = basis @ np.linalg.lstsq(basis, current, rcond=None)[0]
            np.testing.assert_allclose(ratios[sample, step], np.linalg.norm(projection) / np.linalg.norm(current))
    with pytest.raises(ValueError, match="WRITE updates"):
        library.step_projection_ratios(states[:3])
    with pytest.raises(ValueError, match="Zero-length"):
        library.step_projection_ratios([states[0], states[0], states[1], states[2]])
    with pytest.raises(ValueError, match="Nonfinite"):
        library.step_projection_ratios([*states[:3], states[3] * float("nan")])


@pytest.mark.parametrize("steps", [None, 0, -1, 2.5, True, "3"])
def test_step_orth_proj_invalid_settings(steps):
    with pytest.raises(ValueError, match="step_orth_proj steps"):
        analyze.step_orth_proj_settings({"step_orth_proj": {"steps": steps}})


@pytest.mark.parametrize("proj_steps", [None, 0, -1, 2.5, True, "3"])
def test_step_orth_proj_invalid_proj_steps(proj_steps):
    with pytest.raises(ValueError, match="proj_steps"):
        analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 4, "proj_steps": proj_steps}})
    with pytest.raises(ValueError, match="proj_steps"):
        library.step_projection_ratios([], proj_steps)


@pytest.mark.parametrize("proj_steps", [1, 2, 4])
def test_step_orth_proj_run_microbatch_invariance_and_extra_updates(monkeypatch, proj_steps):
    model = type("Model", (), {"device": "cpu"})()
    inputs = {"context_input_ids": torch.arange(1, 6).reshape(5, 1)}
    labels = torch.zeros(5, 1)
    collected = []
    def trajectory(model, batch, steps):
        collected.append(steps)
        scale = batch["context_input_ids"].double().reshape(-1, 1, 1)
        states = [torch.zeros(len(scale), 1, 3, dtype=torch.float64)]
        for k in range(steps):
            update = torch.cat([scale, scale**2 * (k + 1), scale * (k + 1)**2], dim=-1)
            states.append(states[-1] + update)
        return states
    monkeypatch.setattr(analyze, "collect_trajectory", trajectory)
    settings = {"steps": 4, "proj_steps": proj_steps}
    results = [analyze.step_orth_proj_run(model, inputs, labels, settings, size) for size in (1, 2, 5)]
    assert set(collected) == {4 + proj_steps}
    for result in results:
        assert result["projection_ratio_per_example"].shape == (5, 4)
        np.testing.assert_allclose(result["projection_ratio"], result["projection_ratio_per_example"].mean(0))
        for key in result:
            np.testing.assert_allclose(result[key], results[0][key])


@pytest.mark.parametrize("proj_steps", [None, 1, 3])
def test_step_orth_proj_driver_aggregation_plot_and_provenance(tmp_path, monkeypatch, capsys, proj_steps):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    class Model:
        lr = .2
        write_objective = "energy"
        config = type("Config", (), {"to_dict": lambda self: {}})()
    monkeypatch.setattr(analyze, "load_model", lambda *args: Model())
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    values = iter([np.array([[.2, .4], [.4, .6]]), np.array([[.7, .8], [.9, 1.]]),
                   np.array([[.5, .6], [.5, .6]])])
    effective_settings = {"steps": 2, "proj_steps": 2 if proj_steps is None else proj_steps}
    def run(model, inputs, labels, settings, batch_size):
        assert settings == effective_settings
        ratios = next(values)
        return dict(projection_ratio=ratios.mean(0), projection_ratio_per_example=ratios)
    monkeypatch.setattr(analyze, "step_orth_proj_run", run)
    config = dict(kind="step_orth_proj", output_dir=str(reports / "projection"), examples=2, dpi=40,
                  dataset={"label": "test", "path": "fake"}, step_orth_proj={"steps": 2},
                  setups=[dict(label="first", runs=["a", "b"]), dict(label="second", runs=["c"])])
    if proj_steps is not None:
        config["step_orth_proj"]["proj_steps"] = proj_steps
    output = analyze.run_analysis(config)
    assert "unequal run coverage" in capsys.readouterr().out
    with np.load(output / "test/setup_0_mean.npz") as data:
        np.testing.assert_allclose(data["projection_ratio_mean"], [.55, .7])
        np.testing.assert_allclose(data["projection_ratio_std"], np.array([.5, .4]) / np.sqrt(2))
        np.testing.assert_array_equal(data["steps"], [1, 2])
    with np.load(output / "test/setup_1_mean.npz") as data:
        np.testing.assert_array_equal(data["projection_ratio_std"], [0, 0])
    with np.load(output / "test/setup_0/run_0.npz") as data:
        assert data["projection_ratio_per_example"].shape == (2, 2)
        np.testing.assert_array_equal(data["steps"], [1, 2])
    assert (output / "test/step_orth_proj.png").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["trajectory_steps"] == 2 + effective_settings["proj_steps"]
    assert manifest["step_orth_proj"] == effective_settings
    assert manifest["setups"][0]["runs"][0]["step_orth_proj"] == effective_settings
    assert manifest["run_counts"] == {"first": 2, "second": 1}
    saved = json.loads((output / "test/setup_0/run_0.json").read_text())
    assert saved["analysis_settings"] == effective_settings
    with (output / "test/summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert [row["step"] for row in rows] == ["1", "2", "1", "2"]
    assert all(int(row["proj_steps"]) == effective_settings["proj_steps"] for row in rows)
    with (output / "test/per_run.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 6
    config["setups"][0]["step_orth_proj"] = {"steps": 3}
    with pytest.raises(ValueError, match="shared step_orth_proj"):
        analyze.validate_config(config)
    del config["setups"][0]["step_orth_proj"]
    config["setups"][0]["runs"][0] = {"path": "a", "step_orth_proj": {"steps": 3}}
    with pytest.raises(ValueError, match="shared step_orth_proj"):
        analyze.validate_config(config)


@pytest.mark.parametrize("key,value", [
    ("steps", None), ("steps", 0), ("steps", -1), ("steps", 2.5), ("steps", True),
    ("energy_threshold", 0), ("energy_threshold", 1.1), ("energy_threshold", True),
    ("energy_threshold", float("nan")), ("energy_threshold", float("inf")),
])
def test_update_svd_settings_validation(key, value):
    with pytest.raises(ValueError, match="update_svd"):
        analyze.update_svd_settings({"update_svd": {"steps": 3, key: value}})


@pytest.mark.parametrize("kind,options", [
    ("step_orth_proj", {}), ("step_orth_proj", {"dash_start": 2, "show_n_runs": True}),
    ("update_svd", {"dash_start": 1, "show_n_runs": False}),
    ("update_svd", {"dash_start": 3, "show_n_runs": False}),
])
def test_geometry_plot_styles_and_legends(tmp_path, monkeypatch, kind, options):
    setups = [dict(label="First", runs=1), dict(label="Second", runs=3)]
    for setup in setups:
        setup.update(projection_ratio_mean=np.array([.2, .4, .6]),
                     projection_ratio_std=np.array([.01, .02, .03]),
                     singular_values_mean=np.array([3., 2., 1.]),
                     singular_values_std=np.array([.1, .2, .3]),
                     ranks=np.arange(1, 4), retained_rank=2)
    figures = []
    close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: figures.append(figure))
    getattr(plots, kind)(tmp_path / "plot.png", setups, 3, dpi=30, **options)
    figure = figures[-1]
    try:
        ax = figure.axes[0]
        labels = ["First (1 run)", "Second (3 runs)"] if options.get("show_n_runs", True) else ["First", "Second"]
        assert ax.get_legend_handles_labels()[1] == labels
        assert [text.get_text() for text in ax.get_legend().get_texts()] == labels
        curves = [container.lines[0] for container in ax.containers]
        dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
        boundary = options.get("dash_start")
        if boundary is None:
            assert not dashed
            for curve in curves:
                np.testing.assert_array_equal(curve.get_xdata(), [1, 2, 3])
        else:
            assert len(dashed) == 3
            np.testing.assert_array_equal(dashed[-1].get_xdata(), [boundary, boundary])
            for curve, continuation in zip(curves, dashed[:-1]):
                assert curve.get_color() == continuation.get_color()
                np.testing.assert_array_equal(curve.get_xdata(), range(1, boundary + 1))
                np.testing.assert_array_equal(continuation.get_xdata(), range(boundary, 4))
                assert curve.get_ydata()[-1] == continuation.get_ydata()[0]
        if kind == "update_svd":
            assert [text.get_text() for text in ax.texts] == ["First: r=2 (95%)", "Second: r=2 (95%)"]
    finally:
        close(figure)


@pytest.mark.parametrize("kind", ["step_orth_proj", "update_svd"])
@pytest.mark.parametrize("options, message", [
    ({"show_n_runs": "false"}, "show_n_runs"),
    ({"show_n_runs": 1}, "show_n_runs"),
    ({"dash_start": True}, "dash_start"),
    ({"dash_start": 0}, "dash_start"),
    ({"dash_start": 4}, "dash_start"),
    ({"dash_start": 2.}, "dash_start"),
])
def test_geometry_plot_options_validation(kind, options, message):
    config = dict(kind=kind, setups=[dict(label="test", runs=["checkpoint"])])
    config[kind] = dict(steps=3, **options)
    with pytest.raises(ValueError, match=message):
        analyze.validate_config(config)


@pytest.mark.parametrize("kind", ["step_orth_proj", "update_svd"])
def test_geometry_driver_forwards_plot_options(tmp_path, monkeypatch, kind):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    model = type("Model", (), {"lr": .2, "write_objective": "energy",
                              "config": type("Config", (), {"to_dict": lambda self: {}})()})()
    monkeypatch.setattr(analyze, "load_model", lambda *args: model)
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    key = "projection_ratio" if kind == "step_orth_proj" else "singular_values"
    monkeypatch.setattr(analyze, kind + "_run", lambda *args: {
        key: np.array([.2, .4]), key + "_per_example": np.array([[.2, .4], [.2, .4]])})
    received = []
    monkeypatch.setattr(plots, kind, lambda *args, **kwargs: received.append(kwargs))
    options = dict(dash_start=1, show_n_runs=False)
    config = dict(kind=kind, output_dir=str(reports / "experiment"), examples=2,
                  dataset={"label": "test", "path": "fake"},
                  setups=[dict(label="test", runs=["checkpoint"])])
    config[kind] = dict(steps=2, **options)
    analyze.run_analysis(config)
    assert len(received) == 1
    assert all(received[0][key] == value for key, value in options.items())


def test_update_svd_microbatch_invariance(monkeypatch):
    model = type("Model", (), {"device": "cpu"})()
    inputs = {"context_input_ids": torch.arange(1, 6).reshape(5, 1)}
    labels = torch.zeros(5, 1)
    def trajectory(model, batch, steps):
        scale = batch["context_input_ids"].float().reshape(-1, 1, 1)
        return [scale * torch.tensor([[[k, 0.]]]) for k in range(steps + 1)]
    monkeypatch.setattr(analyze, "collect_trajectory", trajectory)
    results = [analyze.update_svd_run(model, inputs, labels, {"steps": 3}, size) for size in (1, 2, 5)]
    for result in results[1:]:
        for key in result:
            np.testing.assert_allclose(result[key], results[0][key])


def test_update_svd_driver_groups_runs_and_marks_mean_spectrum(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    class Model:
        lr = .2
        write_objective = "energy"
        config = type("Config", (), {"to_dict": lambda self: {}})()
    monkeypatch.setattr(analyze, "load_model", lambda *args: Model())
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    spectra = iter([[3., 1.], [1., 1.], [0., 0.]])
    def run(*args):
        values = np.array(next(spectra))
        return dict(singular_values=values, singular_values_per_example=np.tile(values, (2, 1)))
    monkeypatch.setattr(analyze, "update_svd_run", run)
    annotations = []
    annotate = Axes.annotate
    def record(self, text, *args, **kwargs):
        annotations.append(text)
        return annotate(self, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "annotate", record)
    config = dict(kind="update_svd", output_dir=str(reports / "svd"), examples=2, dpi=40,
                  dataset={"label": "test", "path": "fake"}, update_svd={"steps": 2},
                  setups=[dict(label="first", runs=["a", "b"]), dict(label="zero", runs=["c"])])
    output = analyze.run_analysis(config)
    with np.load(output / "test/setup_0_mean.npz") as data:
        np.testing.assert_allclose(data["singular_values_mean"], [2., 1.])
        np.testing.assert_allclose(data["singular_values_std"], [np.sqrt(2), 0])
        np.testing.assert_array_equal(data["ranks"], [1, 2])
    with np.load(output / "test/setup_1_mean.npz") as data:
        np.testing.assert_array_equal(data["singular_values_std"], [0, 0])
    assert annotations == ["first: r=2 (95%)"]
    assert (output / "test/update_svd.png").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_counts"] == {"first": 2, "zero": 1}
    assert manifest["update_svd"] == {"steps": 2, "energy_threshold": .95}
    with (output / "test/summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["retained_rank"] for row in rows] == ["2", "2", "", ""]
    saved = json.loads((output / "test/setup_0/run_0.json").read_text())
    assert saved["analysis_settings"]["steps"] == 2
    config["setups"][0]["update_svd"] = {"steps": 4}
    with pytest.raises(ValueError, match="shared update_svd"):
        analyze.validate_config(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analysis_loader_disables_target_guided_training(tmp_path, monkeypatch, dtype):
    from safetensors.torch import save_file
    base = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    config = GradMemGPTConfig(base_config=base, n_mem_tokens=2, K=2, lr=.2,
        grad_mode="second", attn_implementation="eager", write_objective="energy",
        energy_memory_search_weight=1., energy_memory_search_use_best_for_next_step=True,
        read_loss_alignment="query_position", mqar_vocab_size=32, mqar_dense_queries=True)
    model = MQARGradMemGPT(config)
    config.save_pretrained(tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    save_file({key: value.detach().clone() for key, value in model.state_dict().items()}, str(checkpoint))
    loaded = library.load_model(checkpoint, dtype=dtype)
    assert not loaded.energy_memory_search_use_best_for_next_step
    assert loaded.energy_memory_search_weight == 0
    assert loaded.grad_mode == "none"
    assert not loaded.training and not any(parameter.requires_grad for parameter in loaded.parameters())
    assert all(parameter.dtype == dtype for parameter in loaded.parameters())
    inputs = {"context_input_ids": torch.tensor([[1, 2, 3, 4]]),
              "query_input_ids": torch.tensor([[5, 6]])}
    labels = torch.tensor([[7, 8]])
    states = library.collect_trajectory(loaded, inputs, 2)
    assert all(state.dtype == dtype for state in states)
    scores = library.score_memory(loaded, inputs, labels, states[-1])
    assert scores["inner_loss"].dtype == scores["outer_loss"].dtype == dtype
    assert all(torch.isfinite(value).all() for value in scores.values())
    assert library.update_singular_values(states).dtype == dtype
    for function, settings in [
        (analyze.landscape_run, {"steps": 2, "grid_size": 3, "real_grads": True}),
        (analyze.extrapolation_run, {"steps": [0, 1]}),
        (analyze.k_lr_sweep_run, {"steps": [0, 1], "inner_lrs": [.1]}),
        (analyze.memory_search_run, {"steps": 1, "num_samples": 2}),
        (analyze.update_svd_run, {"steps": 2}),
        (analyze.step_orth_proj_run, {"steps": 1, "proj_steps": 1}),
    ]:
        result = function(loaded, inputs, labels, settings, 1)
        assert all(np.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize("kind", ["landscape", "extrapolation", "k_lr_sweep", "memory_search",
                                  "update_svd", "step_orth_proj"])
@pytest.mark.parametrize("precision", [None, "fp32", "fp64"])
def test_analysis_precision_validation(kind, precision):
    config = dict(kind=kind, setups=[dict(label="test", runs=["checkpoint"])],
                  k_lr_sweep={"steps": [1], "inner_lrs": [.1]},
                  update_svd={"steps": 2}, step_orth_proj={"steps": 2})
    if precision is not None:
        config["precision"] = precision
    analyze.validate_config(config)
    for invalid in [None, True, 64, "float64", "fp16", {}, []]:
        config["precision"] = invalid
        with pytest.raises(ValueError, match="precision must be"):
            analyze.validate_config(config)
    config.pop("precision")
    config[kind] = dict(config.get(kind, {}), precision="fp64")
    with pytest.raises(ValueError, match="top-level precision"):
        analyze.validate_config(config)


def test_fp64_memory_search_preserves_small_displacements():
    from grad_memgpt import GradMemGPT
    model = type("Model", (), {"energy_memory_search_num_samples": 4,
                               "energy_memory_search_radius_scale": .5})()
    center = torch.tensor([[[1., 1., 1.]]], dtype=torch.float64)
    previous = center + 1e-10
    candidates = GradMemGPT._sample_norm_preserving_memory_candidates(model, center, previous)
    assert candidates.dtype == torch.float64
    distances = (candidates - center).flatten(2).norm(dim=-1)
    expected = .5 * (center - previous).flatten(1).norm(dim=-1)
    torch.testing.assert_close(distances, expected.expand_as(distances), rtol=1e-5, atol=1e-15)


def test_fp64_update_svd_preserves_tiny_singular_values():
    states = [torch.zeros(1, 1, 2, dtype=torch.float64)]
    states.append(torch.tensor([[[1., 0.]]], dtype=torch.float64))
    states.append(torch.tensor([[[1., 1e-50]]], dtype=torch.float64))
    values = library.update_singular_values(states)
    assert values.dtype == torch.float64
    assert values[0, 1].item() == pytest.approx(1e-50, rel=1e-10, abs=0)


def test_json_driver_averages_cosines_per_run_and_saves_provenance(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    alpha = beta = np.linspace(-1, 1, 5)
    x, y = np.meshgrid(alpha, beta)
    grids = [(x, x), (10*y, -y)]
    class Model:
        lr = .2
        write_objective = "energy"
        config = type("Config", (), {"to_dict": lambda self: {}, "seed": None})()
    monkeypatch.setattr(analyze, "load_model", lambda *args: Model())
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    def run(*args):
        inner, outer = grids.pop(0)
        return dict(alpha=alpha, beta=beta, inner_loss=inner, outer_loss=outer,
                    alignment=library.finite_difference_cosine(inner, outer, alpha, beta),
                    trajectory=np.array([[0, 0], [1, 0], [1, 1]]))
    monkeypatch.setattr(analyze, "landscape_run", run)
    plot_options = []
    plot_summaries = []
    def record_plot(path, summaries, *args, **kwargs):
        plot_options.append(kwargs)
        plot_summaries.extend(summaries)
    monkeypatch.setattr(plots, "landscapes", record_plot)
    config = dict(kind="landscape", output_dir=str(reports / "experiment"), examples=2,
                   dataset={"label": "test", "path": "fake"},
                   landscape={"separate_colorbars": True, "include_cosine_sim": False},
                  setups=[dict(label="test", runs=["first", "second"])])
    output = analyze.run_analysis(config)
    assert (output / "config.json").is_file()
    assert (output / "manifest.json").is_file()
    assert plot_options == [{"separate_colorbars": True, "include_cosine_sim": False, "show_n_runs": True}]
    assert plot_summaries[0]["mean_cosine"] == pytest.approx(0)
    with np.load(output / "test/setup_0_mean.npz") as result:
        np.testing.assert_allclose(result["alignment"], 0)
    with pytest.raises(FileExistsError):
        analyze.run_analysis(config)


@pytest.mark.parametrize("precision", [None, "fp32", "fp64"])
def test_json_driver_extrapolation_evaluates_every_run(tmp_path, monkeypatch, precision):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    class Model:
        lr = 5.
        write_objective = "energy"
        config = type("Config", (), {"to_dict": lambda self: {}, "seed": 143})()
    calls = []
    def load(checkpoint, device, inner_lr, dtype):
        assert dtype == (torch.float64 if precision == "fp64" else torch.float32)
        calls.append(str(checkpoint))
        return Model()
    monkeypatch.setattr(analyze, "load_model", load)
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    monkeypatch.setattr(analyze, "extrapolation_run", lambda *args: {
        key: np.array([.2, .4]) for key in ("outer_loss", "token_accuracy", "exact_match")})
    plot_options = []
    monkeypatch.setattr(plots, "extrapolation", lambda *args, **kwargs: plot_options.append(kwargs))
    config = dict(kind="extrapolation", output_dir=str(reports / "experiment"), examples=2,
        dataset={"label": "test", "path": "fake"},
        extrapolation={"steps": [1, 2], "dash_start": 1, "show_n_runs": True},
        setups=[dict(label="first", runs=["a", "b"]), dict(label="second", runs=["c"])])
    if precision is not None:
        config["precision"] = precision
    output = analyze.run_analysis(config)
    assert json.loads((output / "manifest.json").read_text())["precision"] == (precision or "fp32")
    assert json.loads((output / "test/setup_0/run_0.json").read_text())["precision"] == (precision or "fp32")
    assert [Path(path).parent.name for path in calls] == ["a", "b", "c"]
    assert (output / "test/summary.csv").is_file()
    assert plot_options == [{"dash_start": 1, "show_n_runs": True}]


@pytest.mark.parametrize("dash_start,show_n_runs", [
    (None, True), (0, False), (2, True), (16, False),
])
def test_extrapolation_plot_styles_and_legends(tmp_path, monkeypatch, dash_start, show_n_runs):
    steps = [0, 1, 2, 4, 8, 16]
    setups = [dict(label="First", runs=1), dict(label="Second", runs=3)]
    for setup in setups:
        for key in ("token_accuracy", "outer_loss", "exact_match"):
            setup[key + "_mean"] = np.linspace(.1, .9, len(steps))
            setup[key + "_std"] = np.full(len(steps), .02)
    figures = []
    close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: figures.append(figure))
    plots.extrapolation(tmp_path / "plot.png", setups, steps, dpi=30,
                        dash_start=dash_start, show_n_runs=show_n_runs)
    figure = figures[-1]
    try:
        expected_labels = ["First (1 run)", "Second (3 runs)"] if show_n_runs else ["First", "Second"]
        assert figure.axes[0].get_yscale() == "linear"
        assert figure.axes[1].get_yscale() == "log"
        assert len(figure.axes) == 3
        assert figure.axes[2].get_yscale() == "linear"
        assert figure.axes[2].get_ylabel() == "Exact match (%)"
        exact_curve = figure.axes[2].containers[0].lines[0]
        np.testing.assert_allclose(exact_curve.get_ydata(),
                                   setups[0]["exact_match_mean"][:len(exact_curve.get_ydata())] * 100)
        assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == expected_labels
        for ax in figure.axes:
            assert ax.get_legend_handles_labels()[1] == expected_labels
            curves = [container.lines[0] for container in ax.containers]
            dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
            if dash_start is None:
                assert not dashed
                for curve in curves:
                    np.testing.assert_array_equal(curve.get_xdata(), range(len(steps)))
            else:
                split = steps.index(dash_start)
                assert len(dashed) == len(setups) + 1
                np.testing.assert_array_equal(dashed[-1].get_xdata(), [split, split])
                for curve, continuation in zip(curves, dashed[:-1]):
                    assert curve.get_color() == continuation.get_color()
                    np.testing.assert_array_equal(curve.get_xdata(), range(split + 1))
                    np.testing.assert_array_equal(continuation.get_xdata(), range(split, len(steps)))
                    assert curve.get_ydata()[-1] == continuation.get_ydata()[0]
    finally:
        close(figure)


@pytest.mark.parametrize("settings, message", [
    ({"show_n_runs": "false"}, "show_n_runs"),
    ({"show_n_runs": 1}, "show_n_runs"),
    ({"dash_start": True}, "dash_start"),
    ({"dash_start": -1}, "dash_start"),
    ({"dash_start": 3}, "dash_start"),
    ({"dash_start": 2.}, "dash_start"),
    ({"steps": [2, 1], "dash_start": 2}, "strictly increasing"),
    ({"steps": [1, 2, 2], "dash_start": 2}, "strictly increasing"),
])
def test_extrapolation_plot_options_validation(settings, message):
    config = dict(kind="extrapolation", extrapolation=settings,
                  setups=[dict(label="test", runs=["checkpoint"])])
    with pytest.raises(ValueError, match=message):
        analyze.validate_config(config)


@pytest.mark.parametrize("config,setup,expected", [
    ({}, {}, [1, 2]),
    ({"landscape": {"basis_steps": [2, 3]}}, {}, [2, 3]),
    ({"landscape": {"basis_steps": [2, 3]}}, {"landscape": {"basis_steps": [3, 4]}}, [3, 4]),
    ({"landscape": {"basis_steps": [2, 3]}}, {"landscape": {"basis_steps": [3, 4]}, "basis_steps": [1, 2]}, [1, 2]),
])
def test_landscape_basis_precedence(config, setup, expected):
    settings = analyze.landscape_settings(config, setup)
    assert settings["basis_steps"] == expected
    assert settings["steps"] == max(expected)


def test_setup_basis_extends_inherited_steps():
    settings = analyze.landscape_settings({"landscape": {"steps": 2}}, {"basis_steps": [2, 3]})
    assert settings["steps"] == 3


@pytest.mark.parametrize("basis", [[0, 1], [2, 1], [1, 1], [1], [1., 2], [True, 2]])
def test_invalid_setup_basis(basis):
    with pytest.raises(ValueError, match="basis_steps"):
        analyze.landscape_settings({}, {"basis_steps": basis})


def test_setup_basis_settings_apply_to_all_runs(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    class Model:
        lr = .2
        write_objective = "energy"
        config = type("Config", (), {"to_dict": lambda self: {}})()
    monkeypatch.setattr(analyze, "load_model", lambda *args: Model())
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    received = []
    def run(model, inputs, labels, settings, batch_size):
        received.append(dict(settings))
        axis = np.arange(3)
        x, y = np.meshgrid(axis, axis)
        return dict(alpha=axis, beta=axis, inner_loss=x, outer_loss=x,
                    alignment=np.ones((3, 3)),
                    trajectory=np.ones((settings["steps"] + 1, 2)))
    monkeypatch.setattr(analyze, "landscape_run", run)
    monkeypatch.setattr(plots, "landscapes", lambda *args, **kwargs: None)
    config = dict(kind="landscape", output_dir=str(reports / "experiment"), examples=2,
                  dataset={"label": "test", "path": "fake"},
                  landscape={"basis_steps": [1, 2], "steps": 2, "real_grads": True},
                  setups=[dict(label="test", basis_steps=[2, 3], runs=[
                      {"path": "first"}, {"path": "second"}])])
    output = analyze.run_analysis(config)
    assert [item["basis_steps"] for item in received] == [[2, 3], [2, 3]]
    assert [item["steps"] for item in received] == [3, 3]
    assert all(item["real_grads"] is True for item in received)
    with np.load(output / "test/setup_0_mean.npz") as result:
        np.testing.assert_array_equal(result["trajectory_runs"], [2, 2, 2, 2])
        assert np.isfinite(result["trajectory_std"]).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["setups"][0]["runs"][0]["landscape"]["basis_steps"] == [2, 3]
    saved = json.loads((output / "test/setup_0/run_0.json").read_text())
    assert saved["analysis_settings"]["basis_steps"] == [2, 3]
    assert saved["analysis_settings"]["real_grads"] is True
    assert manifest["setups"][0]["runs"][0]["landscape"]["real_grads"] is True
    config["output_dir"] = str(reports / "invalid")
    config["setups"][0]["runs"][0]["basis_steps"] = [1, 2]
    with pytest.raises(ValueError, match="on the setup"):
        analyze.run_analysis(config)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("include_cosine_sim", [False, True])
def test_landscape_colorbar_layout_and_scales(tmp_path, monkeypatch, separate, log, include_cosine_sim):
    from matplotlib.figure import Figure
    from matplotlib.colors import LogNorm
    axis = np.linspace(-1, 1, 3)
    x, y = np.meshgrid(axis, axis)
    setups = [dict(label=str(index), runs=1, inner_loss=scale * (x + y + 3),
                   outer_loss=scale * (x - y + 3), alignment=(x + y) / 2,
                   trajectory=np.array([[0, 0], [1, 0], [1, 1]]))
              for index, scale in enumerate([1, 10])]
    bars = []
    colorbar = Figure.colorbar
    def record(self, image, **kwargs):
        result = colorbar(self, image, **kwargs)
        bars.append((image.norm, kwargs["cax"].get_subplotspec(), result))
        return result
    monkeypatch.setattr(Figure, "colorbar", record)
    if not include_cosine_sim:
        for setup in setups:
            del setup["alignment"]
    plots.landscapes(tmp_path / "landscape.png", setups, axis, axis, [1, 2], dpi=30,
                     log=log, separate_colorbars=separate, include_cosine_sim=include_cosine_sim)
    rows = 3 if include_cosine_sim else 2
    assert len(bars) == rows * (2 if separate else 1)
    for row in range(rows):
        selected = bars[row * 2:row * 2 + 2] if separate else [bars[row]]
        for col, (norm, position, bar) in enumerate(selected):
            assert position.rowspan.start == row
            assert position.colspan.start == (2 * col + 1 if separate else 2)
            expected = (1 * (10 ** col), 5 * (10 ** col)) if separate else (1, 50)
            if row == 2:
                expected = (-1, 1)
            assert (norm.vmin, norm.vmax) == expected
            assert isinstance(norm, LogNorm) == (log and row < 2)
            midpoint = np.sqrt(expected[0] * expected[1]) if log and row < 2 else sum(expected) / 2
            np.testing.assert_allclose(bar.get_ticks(), [expected[0], midpoint, expected[1]])
            assert len(bar.ax.yaxis.get_minorticklocs()) == 0
            assert bool(bar.ax.get_ylabel()) == (not separate or col == len(setups) - 1)
    labeled_axes = [bar.ax for _, _, bar in bars if bar.ax.get_ylabel()]
    labeled_axes[0].figure.canvas.draw()
    label_x = [ax.yaxis.label.get_transform().transform(ax.yaxis.label.get_position())[0]
               for ax in labeled_axes]
    np.testing.assert_allclose(label_x, label_x[0], rtol=0, atol=0.01)


def test_landscape_loss_contours_are_present(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    axis = np.linspace(-1, 3, 5)
    x, y = np.meshgrid(axis, axis)
    setup = dict(label="Test", runs=1, inner_loss=np.exp(x+y), outer_loss=np.exp(x-y),
                 alignment=np.ones_like(x), trajectory=np.array([[0, 0], [1, 0], [1, 1]]))
    calls = []
    contour = Axes.contour
    def record(self, *args, **kwargs):
        calls.append(kwargs)
        return contour(self, *args, **kwargs)
    monkeypatch.setattr(Axes, "contour", record)
    plots.landscapes(tmp_path / "landscape.png", [setup], axis, axis, [1, 2], dpi=40)
    assert len(calls) == 2
    assert all(call["colors"] == "white" and len(call["levels"]) == 24 for call in calls)


def test_shifted_basis_labels_actual_memory_states(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    axis = np.linspace(-1, 3, 5)
    x, y = np.meshgrid(axis, axis)
    setup = dict(label="Test", runs=1, basis_steps=[2, 3],
                 inner_loss=np.exp(x+y), outer_loss=np.exp(x-y), alignment=np.ones_like(x),
                 trajectory=np.array([[-.5, -.5], [0, 0], [1, 0], [1, 1]]))
    labels = []
    annotate = Axes.annotate
    def record(self, text, *args, **kwargs):
        labels.append(text)
        return annotate(self, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "annotate", record)
    plots.landscapes(tmp_path / "shifted.png", [setup], axis, axis, [1, 2], dpi=40)
    assert labels == ["M1", "M2", "M3"] * 3


@pytest.mark.parametrize("axis,values", [
    ("steps", []), ("steps", [1, 1]), ("steps", [-1]), ("steps", [1.]),
    ("steps", [True]), ("inner_lrs", []), ("inner_lrs", [0]),
    ("inner_lrs", [-.1]), ("inner_lrs", [.2, .2]), ("inner_lrs", [True]),
    ("inner_lrs", [float("nan")]), ("inner_lrs", [float("inf")]),
])
def test_sweep_invalid_axes(axis, values):
    settings = dict(steps=[0, 1], inner_lrs=[.1, .2])
    settings[axis] = values
    with pytest.raises(ValueError, match=axis):
        analyze.sweep_settings({"k_lr_sweep": settings}, {})


def test_sweep_grid_precedence_and_ambiguous_overrides():
    config = dict(kind="k_lr_sweep", k_lr_sweep=dict(steps=[0, 2], inner_lrs=[.1, .2]),
                  setups=[dict(label="test", runs=["a"])])
    assert analyze.sweep_settings(config, {"k_lr_sweep": {"steps": [4, 1]}}) == {
        "steps": [4, 1], "inner_lrs": [.1, .2]}
    for location in (config, config["setups"][0]):
        location["inner_lr"] = .5
        with pytest.raises(ValueError, match="scalar inner_lr"):
            analyze.validate_config(config)
        del location["inner_lr"]
    config["setups"][0]["runs"] = [{"path": "a", "inner_lr": .5}]
    with pytest.raises(ValueError, match="scalar inner_lr"):
        analyze.validate_config(config)
    config["setups"][0]["runs"] = [{"path": "a", "k_lr_sweep": {}}]
    with pytest.raises(ValueError, match="on the setup"):
        analyze.validate_config(config)


@pytest.mark.parametrize("failure", [None, "exception", "nonfinite"])
def test_sweep_evaluation_orientation_and_restoration(monkeypatch, failure):
    class Model:
        lr, K, last_K_second_order = .5, 3, 0
        config = type("Config", (), {"lr": .5, "K": 3, "last_K_second_order": 0})()
    model = Model()
    calls = []
    def evaluate(model, inputs, labels, settings, batch_size):
        assert model.config.lr == model.lr
        calls.append(model.lr)
        library.set_steps(model, settings["steps"][-1])
        if failure == "exception":
            raise RuntimeError("failed cell")
        values = np.array(settings["steps"]) + model.lr
        if failure == "nonfinite":
            values[0] = np.nan
        return {key: values for key in ("outer_loss", "token_accuracy", "exact_match")}
    monkeypatch.setattr(analyze, "extrapolation_run", evaluate)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            analyze.k_lr_sweep_run(model, {}, None, {"steps": [0, 2], "inner_lrs": [.1, .2]}, 1)
    else:
        data = analyze.k_lr_sweep_run(model, {}, None, {"steps": [0, 2], "inner_lrs": [.1, .2]}, 1)
        assert calls == [.1, .2]
        np.testing.assert_allclose(data["outer_loss"], [[.1, .2], [2.1, 2.2]])
    assert model.lr == model.config.lr == .5
    assert model.K == model.config.K == 3


def test_json_driver_k_lr_sweep_every_run_and_setup_grid(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    class Model:
        lr, K = .5, 3
        write_objective = "energy"
        config = type("Config", (), {"to_dict": lambda self: {"K": self.K, "lr": self.lr},
                                      "K": 3, "lr": .5})()
    loads, calls, rendered = [], [], []
    def load(checkpoint, device, inner_lr, dtype):
        assert inner_lr is None
        model = Model()
        model.index = len(loads)
        loads.append(checkpoint)
        return model
    def evaluate(model, inputs, labels, settings, batch_size):
        calls.append((model.index, model.lr, list(settings["steps"])))
        library.set_steps(model, settings["steps"][-1])
        return {key: np.array(settings["steps"]) + model.lr + model.index
                for key in ("outer_loss", "token_accuracy", "exact_match")}
    monkeypatch.setattr(analyze, "load_model", load)
    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: fake_checkpoint(tmp_path, path))
    monkeypatch.setattr(analyze, "load_data", lambda *args: ({}, torch.ones(2, 1), {}))
    monkeypatch.setattr(analyze, "extrapolation_run", evaluate)
    monkeypatch.setattr(plots, "k_lr_sweep", lambda path, summaries, dpi, **kwargs: rendered.extend(summaries))
    config = dict(kind="k_lr_sweep", output_dir=str(reports / "sweep"), examples=2,
                  dataset={"label": "test", "path": "fake"},
                  k_lr_sweep={"steps": [0, 2], "inner_lrs": [.1, .2]},
                  setups=[dict(label="first", runs=["a", "b"]),
                          dict(label="second", runs=["c"], k_lr_sweep={"steps": [1], "inner_lrs": [.3]})])
    output = analyze.run_analysis(config)
    assert len(loads) == 3
    assert calls == [(0, .1, [0, 2]), (0, .2, [0, 2]), (1, .1, [0, 2]),
                     (1, .2, [0, 2]), (2, .3, [1])]
    with np.load(output / "test/setup_0_mean.npz") as result:
        np.testing.assert_array_equal(result["steps"], [0, 2])
        np.testing.assert_array_equal(result["inner_lrs"], [.1, .2])
        np.testing.assert_allclose(result["outer_loss_mean"], [[.6, .7], [2.6, 2.7]])
        np.testing.assert_allclose(result["outer_loss_std"], np.sqrt(.5))
    with np.load(output / "test/setup_1_mean.npz") as result:
        np.testing.assert_allclose(result["outer_loss_std"], 0)
    details = json.loads((output / "test/setup_0/run_0.json").read_text())
    assert details["trained_settings"] == {"K": 3, "inner_lr": .5}
    assert details["config"] == {"K": 3, "lr": .5}
    assert details["analysis_settings"] == config["k_lr_sweep"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["setups"][1]["runs"][0]["k_lr_sweep"] == {"steps": [1], "inner_lrs": [.3]}
    assert manifest["run_counts"] == {"first": 2, "second": 1}
    with (output / "test/per_run.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert rows[-1]["K"] == "1" and rows[-1]["inner_lr"] == "0.3"
    with (output / "test/summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 5
    assert rendered[1]["steps"] == [1]


def test_sweep_heatmaps_share_scales_and_guard_training_markers(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    setup = dict(label="First", runs=1, steps=[0, 2], inner_lrs=[.1, .2],
                 trained_settings=[{"K": 2, "inner_lr": .2}])
    for key in ("outer_loss", "token_accuracy", "exact_match"):
        setup[key + "_mean"] = np.array([[.1, .2], [.3, .4]])
        setup[key + "_std"] = np.zeros((2, 2))
    second = dict(setup, label="Second", trained_settings=[{"K": 10, "inner_lr": .2}])
    second["outer_loss_mean"] = setup["outer_loss_mean"] + 1
    images, markers, texts = [], [], []
    imshow, plot, text = Axes.imshow, Axes.plot, Axes.text
    def record_image(self, *args, **kwargs):
        images.append(kwargs["norm"])
        return imshow(self, *args, **kwargs)
    def record_plot(self, *args, **kwargs):
        if kwargs.get("marker") == "*":
            markers.append(args)
        return plot(self, *args, **kwargs)
    def record_text(self, x, y, label, **kwargs):
        texts.append(label)
        return text(self, x, y, label, **kwargs)
    monkeypatch.setattr(Axes, "imshow", record_image)
    monkeypatch.setattr(Axes, "plot", record_plot)
    monkeypatch.setattr(Axes, "text", record_text)
    plots.k_lr_sweep(tmp_path / "sweep.png", [setup, second], dpi=30)
    assert len(markers) == 6
    assert images[0] is images[1] and images[0].vmax == 1.4
    assert "10.0\n+/- 0.0" in texts
    assert len(list(tmp_path.glob("sweep*.png"))) == 4


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("include_cosine", [False, True])
@pytest.mark.parametrize("report_avg", [False, True])
def test_landscape_average_annotations(tmp_path, monkeypatch, separate, include_cosine, report_avg):
    axis = np.linspace(-1, 1, 3)
    x, y = np.meshgrid(axis, axis)
    setups = [dict(label=str(index), runs=2, inner_loss=x + y + 3, outer_loss=x - y + 3,
                   alignment=np.zeros_like(x), mean_cosine=mean,
                   trajectory=np.array([[0, 0], [1, 0], [1, 1]]))
              for index, mean in enumerate([.125, -.625])]
    figures = []
    close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: figures.append(figure))
    plots.landscapes(tmp_path / "landscape.png", setups, axis, axis, [1, 2], dpi=30,
                     separate_colorbars=separate, include_cosine_sim=include_cosine, report_avg=report_avg)
    figure = figures[-1]
    try:
        labels = [ax.get_xlabel() for ax in figure.axes if ax.get_xlabel()]
        assert labels == (["Avg. cosine: 0.125", "Avg. cosine: -0.625"]
                          if report_avg and include_cosine else [])
    finally:
        close(figure)


@pytest.mark.parametrize("value", [None, 1, "true", []])
def test_landscape_report_avg_validation(value):
    config = dict(kind="landscape", landscape={"report_avg": value},
                  setups=[dict(label="test", runs=["checkpoint"])])
    with pytest.raises(ValueError, match="report_avg must be"):
        analyze.validate_config(config)
    with pytest.raises(ValueError, match="report_avg must be"):
        analyze.landscape_settings(config, {})
