"""Projection-panel contracts using fake CPU trajectories and real plot artists."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from analysis_tools import analyze, library, plots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def trajectory(model, batch, steps):
    """Non-collinear updates vary by example, seed, and step, not microbatch."""
    example = batch["context_input_ids"].double().reshape(-1, 1, 1)
    coordinate = torch.arange(1, 7, dtype=torch.float64).reshape(1, 1, -1)
    states = [torch.zeros(len(example), 1, 6, dtype=torch.float64)]
    for step in range(1, steps + 1):
        update = torch.sin(coordinate * (step + example / 7 + model.seed / 3))
        update += .2 * torch.cos(coordinate * example * step / 5)
        states.append(states[-1] + update)
    return states


@pytest.mark.parametrize("value", [1, 3, [1], [3, 1], [1, 2, 4]])
def test_settings_preserve_scalar_or_ordered_list(value):
    config = {"step_orth_proj": {"steps": 4, "proj_steps": value}}
    before = copy.deepcopy(config)
    assert analyze.step_orth_proj_settings(config) == config["step_orth_proj"]
    assert config == before
    assert analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 4}}) == {
        "steps": 4, "proj_steps": 2}


@pytest.mark.parametrize("value", [
    True, False, 0, -1, 1.0, 1.5, "1", None, (), (1, 2), {}, [],
    [1, 1], [3, 1, 3], [True], [False], [0], [-1], [1.0], ["1"],
    [None], [[1]], [1, [2]], [1, {}], [1, []], [1, False],
])
def test_settings_reject_invalid_projection_spans(value):
    with pytest.raises(ValueError, match="proj_steps"):
        analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 4, "proj_steps": value}})


@pytest.mark.parametrize("key,value", [
    ("uncertainty_style", "ribbon"), ("uncertainty_style", None),
    ("uncertainty_style", True), ("show_runs", 1), ("show_runs", "false"),
    ("show_runs", None),
])
def test_plot_settings_reject_invalid_values(key, value):
    config = dict(kind="step_orth_proj", step_orth_proj={"steps": 3, key: value},
                  setups=[dict(label="test", runs=["fake"])])
    with pytest.raises(ValueError, match=key):
        analyze.validate_config(config)


@pytest.mark.parametrize("spans,batch_size", [
    ([1], 1), ([3, 1], 2), ([1, 2, 4], 2), ([1, 2, 4], 5),
])
def test_list_reuses_one_trajectory_and_matches_scalar_and_microbatches(monkeypatch, spans, batch_size):
    model = SimpleNamespace(device="cpu", seed=2)
    inputs = {"context_input_ids": torch.arange(1, 6).reshape(5, 1)}
    labels = torch.zeros(5, 1)
    collected, projected = [], []

    def collect(model, batch, steps):
        states = trajectory(model, batch, steps)
        collected.append((steps, states))
        return states

    def project(states, span):
        projected.append((span, states))
        return library.step_projection_ratios(states, span)

    monkeypatch.setattr(analyze, "collect_trajectory", collect)
    monkeypatch.setattr(analyze, "step_projection_ratios", project)
    result = analyze.step_orth_proj_run(model, inputs, labels,
                                      {"steps": 4, "proj_steps": spans}, batch_size)
    assert len(collected) == (5 + batch_size - 1) // batch_size
    assert len(projected) == len(collected) * len(spans)
    for batch_index, (steps, states) in enumerate(collected):
        assert steps == 4 + max(spans)
        for panel, span in enumerate(spans):
            actual_span, sliced = projected[batch_index * len(spans) + panel]
            assert actual_span == span
            assert len(sliced) == 4 + span + 1
            assert all(a is b for a, b in zip(sliced, states))
    assert result["projection_ratio"].shape == (len(spans), 4)
    assert result["projection_ratio_per_example"].shape == (5, len(spans), 4)
    np.testing.assert_allclose(result["projection_ratio"], result["projection_ratio_per_example"].mean(0))
    for panel, span in enumerate(spans):
        scalar = analyze.step_orth_proj_run(model, inputs, labels, {"steps": 4, "proj_steps": span}, 5)
        assert scalar["projection_ratio"].shape == (4,)
        assert scalar["projection_ratio_per_example"].shape == (5, 4)
        np.testing.assert_allclose(result["projection_ratio"][panel], scalar["projection_ratio"], atol=1e-12)
        np.testing.assert_allclose(result["projection_ratio_per_example"][:, panel],
                                   scalar["projection_ratio_per_example"], atol=1e-12)
    assert np.ptp(result["projection_ratio_per_example"][:, 0, 0]) > .01


@pytest.fixture
def report_factory(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    inputs = {"context_input_ids": torch.arange(1, 6).reshape(5, 1)}
    labels = torch.zeros(5, 1)
    checkpoints = []
    for seed in (11, 23, 37):
        run = tmp_path / f"seed_{seed}"
        checkpoint = run / "checkpoint-7"
        checkpoint.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"cli_args": {"seed": seed}}))
        (checkpoint / "config.json").write_text(json.dumps({"seed": seed}))
        (checkpoint / "model.safetensors").touch()
        checkpoints.append(checkpoint)

    def load(checkpoint, device, inner_lr, dtype):
        seed = json.loads((checkpoint.parent / "config.json").read_text())["seed"]
        return SimpleNamespace(seed=seed, device=device, lr=.2, write_objective="energy",
                               config=SimpleNamespace(to_dict=lambda: {"seed": seed, "K": 2}))

    monkeypatch.setattr(analyze, "resolve_checkpoint", lambda path: Path(path) / "model.safetensors")
    monkeypatch.setattr(analyze, "load_model", load)
    monkeypatch.setattr(analyze, "load_data", lambda *args: (inputs, labels, {"examples": 5}))
    monkeypatch.setattr(analyze, "collect_trajectory", trajectory)

    def make(spans=[3, 1], style="errorbar", show_runs=False, datasets=1, include_cosine=False):
        settings = {"steps": 3}
        if spans is not None:
            settings["proj_steps"] = spans
        if style is not None:
            settings["uncertainty_style"] = style
        if show_runs:
            settings["show_runs"] = True
        if include_cosine:
            settings["include_cosine_sim"] = True
        config = dict(kind="step_orth_proj", output_dir=str(reports / f"report_{len(list(reports.iterdir()))}"),
                      step_orth_proj=settings, device="cpu", batch_size=2, examples=5,
                      example_offset=7, dpi=25, figsize=[9, 3],
                      datasets=[dict(path="fake", label=f"data_{i}") for i in range(datasets)],
                      setups=[dict(label="two seeds", runs=[str(p) for p in checkpoints[:2]]),
                              dict(label="one seed", runs=[str(checkpoints[2])])])
        output = analyze.run_analysis(config)
        return SimpleNamespace(output=output, folder=output / "data_0", config=config,
                               inputs=inputs, labels=labels)

    return make


def test_consecutive_cosines_are_signed_and_reuse_trajectory(monkeypatch):
    calls = []
    def collect(model, batch, steps):
        calls.append(steps)
        updates = torch.tensor([[1., 0.], [-2., 0.], [0., 3.], [0., 4.], [2., 0.]], dtype=torch.float64)
        state = torch.zeros(len(batch["context_input_ids"]), 1, 2, dtype=torch.float64)
        states = [state]
        for update in updates[:steps]:
            states.append(states[-1] + update)
        return states
    monkeypatch.setattr(analyze, "collect_trajectory", collect)
    inputs = {"context_input_ids": torch.ones(2, 1, dtype=torch.long)}
    model = SimpleNamespace(device="cpu")
    result = analyze.step_orth_proj_run(model, inputs, torch.zeros(2, 1),
                                       dict(steps=3, proj_steps=[1, 2], include_cosine_sim=True), 2)
    assert calls == [5]
    np.testing.assert_allclose(result["cosine_sim"], [-1, 0, 1])
    np.testing.assert_allclose(result["cosine_sim_per_example"], [[-1, 0, 1], [-1, 0, 1]])
    np.testing.assert_array_equal(result["cosine_valid_examples"], [2, 2, 2])


def test_cosine_requires_both_steps_nonzero(monkeypatch):
    start = torch.zeros(2, 1, 2)
    first = start + torch.tensor([1., 0.])
    last = first + torch.tensor([[[0., 0.]], [[2., 0.]]])
    monkeypatch.setattr(analyze, "collect_trajectory", lambda *args: [start, first, last])
    inputs = {"context_input_ids": torch.ones(2, 1, dtype=torch.long)}
    settings = dict(steps=1, proj_steps=1, include_cosine_sim=True)
    with pytest.raises(ValueError, match="undefined cosine"):
        analyze.step_orth_proj_run(SimpleNamespace(device="cpu"), inputs, torch.zeros(2, 1), settings, 2)
    result = analyze.step_orth_proj_run(SimpleNamespace(device="cpu"), inputs, torch.zeros(2, 1),
                                       dict(settings, zero_step_policy="nan"), 2)
    np.testing.assert_allclose(result["projection_ratio_per_example"], [[0], [1]])
    np.testing.assert_allclose(result["cosine_sim_per_example"], [[np.nan], [1]], equal_nan=True)
    np.testing.assert_allclose(result["cosine_sim"], [1])
    np.testing.assert_array_equal(result["cosine_valid_examples"], [1])


def test_cosine_option_is_boolean():
    assert analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 3}}).get("include_cosine_sim", False) is False
    for value in (None, "true", 1):
        with pytest.raises(ValueError, match="include_cosine_sim"):
            analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 3, "include_cosine_sim": value}})


def test_null_projection_skips_projection_work_and_collects_only_needed_steps(monkeypatch):
    calls = []
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    def collect(model, batch, steps):
        calls.append(steps)
        state = torch.zeros(len(batch["context_input_ids"]), 1, 2, dtype=torch.float64)
        states = [state]
        for update in torch.tensor([[1., 0.], [0., 2.], [-3., 0.], [0., -4.]])[:steps]:
            states.append(states[-1] + update)
        return states
    def forbidden(*args, **kwargs):
        pytest.fail("proj_steps=null must not compute projection ratios")
    monkeypatch.setattr(analyze, "collect_trajectory", collect)
    monkeypatch.setattr(analyze, "step_projection_ratios", forbidden)
    for norms, cosine, length in ((True, False, 3), (False, True, 4), (True, True, 4)):
        result = analyze.step_orth_proj_run(SimpleNamespace(device="cpu"),
            {"context_input_ids": torch.ones(2, 1)}, torch.zeros(2, 1),
            dict(steps=3, proj_steps=None, add_step_norm=norms, include_cosine_sim=cosine), 2)
        assert calls[-1] == length
        assert result["projection_ratio"].shape == (0, 3)
        assert result["projection_ratio_per_example"].shape == (2, 0, 3)
        np.testing.assert_allclose(result["step_norm"], [1, 2, 3])
        if cosine:
            np.testing.assert_allclose(result["cosine_sim"], [0, 0, 0])
        setup = dict(label="test", runs=1, projection_ratio_mean=result["projection_ratio"],
                     projection_ratio_std=np.zeros((0, 3)), step_norm_mean=result["step_norm"],
                     step_norm_std=np.zeros(3), step_norm_runs=result["step_norm"][None])
        if cosine:
            setup.update(cosine_sim_mean=result["cosine_sim"], cosine_sim_std=np.zeros(3),
                         cosine_sim_runs=result["cosine_sim"][None])
        plots.step_orth_proj(Path("unused.png"), [setup], 3, proj_steps=None,
                            add_step_norm=norms, include_cosine_sim=cosine,
                            uncertainty_style="band", show_runs=True)
        assert len(figures[-1].axes) == int(norms) + int(cosine)
        if norms:
            assert figures[-1].axes[0].get_yscale() == "log"
        if cosine:
            assert figures[-1].axes[-1].get_ylabel() == "Mean cosine similarity"


def test_null_projection_reports_and_redraw_hiding(report_factory, monkeypatch):
    figures = []
    original = Figure.savefig
    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    full = report_factory([1, 2], "band", True, include_cosine=True)
    config = copy.deepcopy(full.config)
    config["output_dir"] = str(full.output.parent / "norms_and_cosines")
    config["step_orth_proj"].update(proj_steps=None, add_step_norm=True)
    output = analyze.run_analysis(config)
    folder = output / "data_0"
    with np.load(folder / "setup_0_mean.npz") as data:
        assert data["projection_ratio_mean"].shape == (0, 3)
        assert data["proj_steps"].shape == (0,)
    rows = read_csv(folder / "summary.csv")
    assert len(rows) == 6
    assert all(row["proj_steps"] == "" for row in rows)
    assert "projection_ratio_mean" not in rows[0]
    assert "step_norm_mean" in rows[0] and "cosine_sim_mean" in rows[0]
    assert json.loads((output / "manifest.json").read_text())["trajectory_steps"] == 4
    def forbidden(*args, **kwargs):
        pytest.fail("Redraw must not recompute")
    for name in ("load_model", "load_data", "collect_trajectory", "step_projection_ratios", "average_runs"):
        monkeypatch.setattr(analyze, name, forbidden)
    analyze.redraw_analysis(dict(output_dir=str(output)))
    analyze.redraw_analysis(dict(output_dir=str(full.output),
        step_orth_proj=dict(proj_steps=None, add_step_norm=True, include_cosine_sim=True)))
    for figure in figures[1:]:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_yscale() == "log"
        assert figure.axes[1].get_ylabel() == "Mean cosine similarity"


def test_cosine_saved_aggregation_and_far_right_redraw(report_factory, monkeypatch):
    figures = []
    savefig = Figure.savefig
    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    report = report_factory([2, 1], "band", True, include_cosine=True)
    with np.load(report.folder / "setup_0_mean.npz") as saved:
        mean, std = saved["cosine_sim_mean"], saved["cosine_sim_std"]
    runs = []
    for index in range(2):
        with np.load(report.folder / f"setup_0/run_{index}.npz") as saved:
            runs.append(saved["cosine_sim"])
            assert saved["cosine_sim_per_example"].shape == (5, 3)
    np.testing.assert_allclose(mean, np.mean(runs, axis=0))
    np.testing.assert_allclose(std, np.std(runs, axis=0, ddof=1))
    assert "cosine_sim_mean" in read_csv(report.folder / "summary.csv")[0]
    def forbidden(*args, **kwargs):
        pytest.fail("Redraw must not evaluate or recompute cosines")
    for name in ("load_model", "load_data", "collect_trajectory", "step_projection_ratios", "average_runs"):
        monkeypatch.setattr(analyze, name, forbidden)
    analyze.redraw_analysis(dict(output_dir=str(report.output),
                                step_orth_proj=dict(include_cosine_sim=True, add_step_norm=True)))
    figure = figures[-1]
    assert len(figure.axes) == 4
    assert figure.axes[0].get_yscale() == "log"
    cosine_ax = figure.axes[-1]
    assert cosine_ax.get_ylabel() == "Mean cosine similarity"
    assert cosine_ax.get_yscale() == "linear"
    assert cosine_ax.get_ylim() == (-1.05, 1.05)
    assert not cosine_ax.get_shared_y_axes().joined(cosine_ax, figure.axes[1])
    assert cosine_ax.get_shared_x_axes().joined(cosine_ax, figure.axes[0])
    np.testing.assert_allclose(cosine_ax.lines[0].get_ydata(), mean)
    assert len([line for line in cosine_ax.lines if line.get_alpha() == .3]) == 3
    path = report.folder / "setup_0_mean.npz"
    with np.load(path) as saved:
        arrays = dict(saved)
    del arrays["cosine_sim_mean"]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="new analysis with include_cosine_sim"):
        analyze.redraw_analysis(dict(output_dir=str(report.output)))


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("spans,style,show_runs", [
    ([3, 1], "band", True), ([1], "errorbar", False),
    (1, "band", True), (None, None, False),
])
def test_normal_analysis_arrays_csv_manifest_and_legacy_defaults(report_factory, capsys, spans, style, show_runs):
    report = report_factory(spans, style, show_runs)
    assert "unequal run coverage" in capsys.readouterr().out
    value = 2 if spans is None else spans
    panels = value if isinstance(value, list) else [value]
    shape = (len(panels), 3) if isinstance(value, list) else (3,)
    manifest = json.loads((report.output / "manifest.json").read_text())
    assert manifest["trajectory_steps"] == 3 + max(panels)
    assert manifest["example_indices"] == list(range(7, 12))
    assert manifest["run_counts"] == {"two seeds": 2, "one seed": 1}
    assert manifest["step_orth_proj"]["proj_steps"] == value
    assert "sample SD" in manifest["aggregation"]
    assert json.loads((report.output / "config.json").read_text()) == report.config
    assert (report.folder / "step_orth_proj.png").read_bytes().startswith(b"\x89PNG")
    per_run_rows = read_csv(report.folder / "per_run.csv")
    summary_rows = read_csv(report.folder / "summary.csv")
    assert len(per_run_rows) == 3 * len(panels) * 3
    assert len(summary_rows) == 2 * len(panels) * 3
    for setup_index, seeds in enumerate(((11, 23), (37,))):
        values = []
        for run_index, seed in enumerate(seeds):
            model = SimpleNamespace(seed=seed, device="cpu")
            expected = analyze.step_orth_proj_run(model, report.inputs, report.labels,
                                                  {"steps": 3, "proj_steps": value}, 5)
            with np.load(report.folder / f"setup_{setup_index}/run_{run_index}.npz") as saved:
                assert saved["projection_ratio"].shape == shape
                assert saved["projection_ratio_per_example"].shape == (5, *shape)
                for key in expected:
                    np.testing.assert_allclose(saved[key], expected[key], atol=1e-12)
                np.testing.assert_array_equal(saved["steps"], [1, 2, 3])
                if isinstance(value, list):
                    np.testing.assert_array_equal(saved["proj_steps"], value)
                else:
                    assert "proj_steps" not in saved
                values.append(saved["projection_ratio"])
            details = json.loads((report.folder / f"setup_{setup_index}/run_{run_index}.json").read_text())
            assert details["seed"] == seed
            assert details["analysis_settings"]["proj_steps"] == value
            assert details["config"]["seed"] == seed
            assert manifest["setups"][setup_index]["runs"][run_index]["seed"] == seed
            rows = [r for r in per_run_rows if r["setup"] == report.config["setups"][setup_index]["label"]
                    and int(r["run"]) == run_index]
            assert [(int(r["proj_steps"]), int(r["step"])) for r in rows] == [
                (p, k) for p in panels for k in (1, 2, 3)]
            np.testing.assert_allclose([float(r["projection_ratio"]) for r in rows], expected["projection_ratio"].ravel())
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1 if len(values) > 1 else 0)
        with np.load(report.folder / f"setup_{setup_index}_mean.npz") as saved:
            np.testing.assert_allclose(saved["projection_ratio_mean"], mean)
            np.testing.assert_allclose(saved["projection_ratio_std"], std)
            np.testing.assert_array_equal(saved["steps"], [1, 2, 3])
            assert "projection_ratio_runs" not in saved
            if isinstance(value, list):
                np.testing.assert_array_equal(saved["proj_steps"], value)
        rows = [r for r in summary_rows if r["setup"] == report.config["setups"][setup_index]["label"]]
        assert [(int(r["proj_steps"]), int(r["step"])) for r in rows] == [
            (p, k) for p in panels for k in (1, 2, 3)]
        np.testing.assert_allclose([float(r["projection_ratio_mean"]) for r in rows], mean.ravel())
        np.testing.assert_allclose([float(r["projection_ratio_std"]) for r in rows], std.ravel())
        if len(values) == 1:
            np.testing.assert_array_equal(std, np.zeros(shape))
        else:
            assert np.any(std > 0)


def forbid_computation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Redraw must use saved arrays, not loaders or numerical analysis")

    for module in (analyze, library):
        for name in ("load_model", "load_data", "resolve_checkpoint", "collect_trajectory",
                     "step_projection_ratios", "step_orth_proj_run", "average_runs", "score_memory"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("spans", [[3, 1], [1], None])
def test_redraw_saved_arrays_style_overlays_and_ignored_numeric_options(report_factory, monkeypatch, spans):
    report = report_factory(spans=spans, style=None)
    originals = {p: p.read_bytes() for p in report.output.rglob("*") if p.is_file()}
    expected_runs = []
    for setup_index, count in ((0, 2), (1, 1)):
        values = []
        for i in range(count):
            with np.load(report.folder / f"setup_{setup_index}/run_{i}.npz") as arrays:
                values.append(arrays["projection_ratio"])
        expected_runs.append(np.stack(values))
    forbid_computation(monkeypatch)
    calls = []
    draw = analyze.draw_plot

    def capture(path, config, summaries, settings, *args):
        calls.append((path, copy.deepcopy(settings), summaries))
        draw(path, config, summaries, settings, *args)

    monkeypatch.setattr(analyze, "draw_plot", capture)
    request = dict(output_dir=str(report.output), figsize=[10, 4], dpi=30,
                   step_orth_proj=dict(steps=999, proj_steps=[9, 8], uncertainty_style="band",
                                       show_runs=True, dash_start=2))
    analyze.redraw_analysis(request)
    path, settings, summaries = calls[0]
    assert settings["steps"] == 3
    assert settings.get("proj_steps", 2) == (2 if spans is None else spans)
    assert settings["uncertainty_style"] == "band"
    assert settings["show_runs"] is True
    for summary, expected in zip(summaries, expected_runs):
        np.testing.assert_array_equal(summary["projection_ratio_runs"], expected)
    assert path.name == "step_orth_proj_redraw_1.png"
    assert path.read_bytes().startswith(b"\x89PNG")
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["plot_settings"] == settings
    assert metadata["request"]["step_orth_proj"]["proj_steps"] == [9, 8]
    assert metadata["figsize"] == [10, 4]
    analyze.redraw_analysis(dict(output_dir=str(report.output), step_orth_proj={"uncertainty_style": "errorbar"}))
    assert calls[1][0].name == "step_orth_proj_redraw_2.png"
    assert calls[1][1]["uncertainty_style"] == "errorbar"
    assert "projection_ratio_runs" not in calls[1][2][0]
    for path, contents in originals.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("damage", [
    "aggregate_spans", "aggregate_missing_spans", "aggregate_steps", "aggregate_missing_steps",
    "aggregate_shape", "run_spans", "run_missing_spans", "run_shape", "run_nonfinite",
    "missing_runfile", "run_steps", "run_missing_steps",
])
def test_redraw_rejects_damaged_axes_or_runs_before_any_render(report_factory, monkeypatch, damage):
    report = report_factory(datasets=2)
    # Damage only the last dataset/setup: earlier jobs must not render either.
    folder = report.output / "data_1"
    per_run = damage.startswith("run_") or damage == "missing_runfile"
    path = folder / ("setup_1/run_0.npz" if per_run else "setup_1_mean.npz")
    with np.load(path) as saved:
        arrays = dict(saved)
    if damage == "missing_runfile":
        path.unlink()
    else:
        if damage.endswith("missing_spans"):
            arrays.pop("proj_steps")
        elif damage.endswith("missing_steps"):
            arrays.pop("steps")
        elif damage.endswith("spans"):
            arrays["proj_steps"] = np.array([1, 3])
        elif damage.endswith("steps"):
            arrays["steps"] = np.array([1, 2, 4])
        elif damage.endswith("shape"):
            key = "projection_ratio" if per_run else "projection_ratio_mean"
            arrays[key] = arrays[key][:1]
        else:
            arrays["projection_ratio"][0, 0] = np.nan
        np.savez_compressed(path, **arrays)
    forbid_computation(monkeypatch)
    render = Mock(side_effect=AssertionError("Rendering must follow validation of every dataset"))
    monkeypatch.setattr(analyze, "draw_plot", render)
    with pytest.raises((ValueError, FileNotFoundError)):
        analyze.redraw_analysis(dict(output_dir=str(report.output), step_orth_proj={"show_runs": True}))
    render.assert_not_called()
    assert not list(report.output.rglob("*redraw*"))


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []
    save = Figure.savefig

    def capture(figure, path, *args, **kwargs):
        figures.append(figure)
        return save(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return figures


@pytest.mark.parametrize("style,spans", [
    (None, 1), ("errorbar", [1]), ("errorbar", [3, 1]), ("band", [3, 1]),
])
def test_actual_artists_panels_clipping_overlays_dash_colors_fonts(tmp_path, captured_figures, style, spans):
    panels = spans if isinstance(spans, list) else [spans]
    setups = []
    for index, count in enumerate((2, 1)):
        mean = np.stack([np.array([.1, .6, .9]) - index * .05 - p * .02 for p in range(len(panels))])
        std = np.full_like(mean, .25)
        runs = np.stack([mean + (r * 2 - 1) * .07 for r in range(count)])
        if not isinstance(spans, list):
            mean, std, runs = mean[0], std[0], runs[:, 0]
        setups.append(dict(label=f"setup {index}", runs=count, projection_ratio_mean=mean,
                           projection_ratio_std=std, projection_ratio_runs=runs))
    settings = dict(steps=3, proj_steps=spans, show_runs=True, dash_start=2)
    if style is not None:
        settings["uncertainty_style"] = style
    config = dict(kind="step_orth_proj", figsize=[11, 4], dpi=30, subtitle_font_size=17,
                  tick_font_size=9, axis_label_font_size=13, legend_font_size=8)
    old_title_size = plt.rcParams["axes.titlesize"]
    path = tmp_path / "projection.png"
    analyze.draw_plot(path, config, setups, settings)
    assert path.read_bytes().startswith(b"\x89PNG")
    figure = captured_figures[-1]
    np.testing.assert_array_equal(figure.get_size_inches(), [11, 4])
    assert plt.rcParams["axes.titlesize"] == old_title_size
    assert len(figure.axes) == len(panels)
    colors = []
    for panel, (ax, span) in enumerate(zip(figure.axes, panels)):
        assert ax.get_title() == f"Projection onto the span of the next {span} WRITE step{'s' if span != 1 else ''}"
        assert ax.title.get_fontsize() == 17
        assert ax.xaxis.label.get_fontsize() == 13
        assert ax.get_xticklabels()[0].get_fontsize() == 9
        if len(panels) > 1:
            assert ax.get_shared_x_axes().joined(figure.axes[0], ax)
            assert ax.get_shared_y_axes().joined(figure.axes[0], ax)
        assert ax.get_ylim()[0] <= 0 and ax.get_ylim()[1] >= 1
        assert len(ax.get_legend_handles_labels()[1]) == 2
        assert ax.get_legend_handles_labels()[1] == ["setup 0 (2 runs)", "setup 1 (1 run)"]
        means = ([line for line in ax.lines if line.get_zorder() == 3] if style == "band"
                 else [container.lines[0] for container in ax.containers if isinstance(container, ErrorbarContainer)])
        # Dashed continuations also have zorder 3 in band mode.
        if style == "band":
            means = [line for line in means if line.get_linestyle() != "--"]
        assert len(means) == 2
        panel_colors = [line.get_color() for line in means]
        colors.append(panel_colors)
        assert len(set(panel_colors)) == 2
        for index, (setup, line) in enumerate(zip(setups, means)):
            values = np.atleast_2d(setup["projection_ratio_mean"])[panel]
            np.testing.assert_array_equal(line.get_xdata(), [1, 2])
            np.testing.assert_array_equal(line.get_ydata(), values[:2])
            dashed = [candidate for candidate in ax.lines if candidate.get_color() == line.get_color()
                      and candidate.get_linestyle() == "--" and candidate.get_alpha() != .3]
            assert len(dashed) == 1
            np.testing.assert_array_equal(dashed[0].get_xdata(), [2, 3])
            np.testing.assert_array_equal(dashed[0].get_ydata(), values[1:])
            overlays = [candidate for candidate in ax.lines if candidate.get_alpha() == .3
                        and candidate.get_color() == line.get_color()]
            assert len(overlays) == 2 * setup["runs"]
            expected = setup["projection_ratio_runs"]
            if isinstance(spans, list):
                expected = expected[:, panel]
            for run, solid, tail in zip(expected, overlays[::2], overlays[1::2]):
                np.testing.assert_array_equal(solid.get_xdata(), [1, 2])
                np.testing.assert_array_equal(solid.get_ydata(), run[:2])
                np.testing.assert_array_equal(tail.get_xdata(), [2, 3])
                np.testing.assert_array_equal(tail.get_ydata(), run[1:])
                assert tail.get_linestyle() == "--"
        boundary = [line for line in ax.lines if line.get_color() == "0.4"]
        assert len(boundary) == 1 and boundary[0].get_linestyle() == "--"
        np.testing.assert_array_equal(boundary[0].get_xdata(), [2, 2])
        bands = [collection for collection in ax.collections if isinstance(collection, PolyCollection)]
        if style == "band":
            assert len(bands) == 2
            for band, setup, color in zip(bands, setups, panel_colors):
                vertices = band.get_paths()[0].vertices
                assert np.all((vertices[:, 1] >= 0) & (vertices[:, 1] <= 1))
                mean = np.atleast_2d(setup["projection_ratio_mean"])[panel]
                std = np.atleast_2d(setup["projection_ratio_std"])[panel]
                for k in range(3):
                    ys = vertices[vertices[:, 0] == k + 1, 1]
                    np.testing.assert_allclose([ys.min(), ys.max()], np.clip([mean[k] - std[k], mean[k] + std[k]], 0, 1))
                np.testing.assert_allclose(band.get_facecolor()[0], to_rgba(color, .12))
        else:
            assert not bands
            assert len(ax.containers) == 2
            for container, setup in zip(ax.containers, setups):
                mean = np.atleast_2d(setup["projection_ratio_mean"])[panel]
                std = np.atleast_2d(setup["projection_ratio_std"])[panel]
                segments = container.lines[2][0].get_segments()
                assert len(segments) == 3
                for k, segment in enumerate(segments):
                    np.testing.assert_allclose(segment, [[k + 1, mean[k] - std[k]],
                                                         [k + 1, mean[k] + std[k]]])
    assert all(panel == colors[0] for panel in colors)
    legend = figure.axes[0].get_legend()
    assert len(legend.get_texts()) == 2
    assert all(text.get_fontsize() == 8 for text in legend.get_texts())


@pytest.mark.parametrize("spans", [None, [1], [3, 1]])
def test_plot_defaults_errorbars_without_run_arrays(tmp_path, captured_figures, spans):
    mean = np.array([.2, .5, .7])
    if spans is not None:
        mean = np.stack([mean] * len(spans))
    summary = dict(label="legacy", runs=1, projection_ratio_mean=mean,
                   projection_ratio_std=np.zeros_like(mean))
    options = {} if spans is None else {"proj_steps": spans}
    plots.step_orth_proj(tmp_path / "defaults.png", [summary], 3, dpi=25, **options)
    axes = captured_figures[-1].axes
    assert len(axes) == (1 if spans is None else len(spans))
    for ax, span in zip(axes, [2] if spans is None else spans):
        assert len(ax.containers) == 1 and isinstance(ax.containers[0], ErrorbarContainer)
        assert not any(line.get_alpha() == .3 for line in ax.lines)
        assert f"next {span} WRITE step" in ax.get_title()
