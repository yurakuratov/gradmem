"""WRITE-step norm collection, report persistence, and CPU-only rendering contracts."""

import copy
import csv
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
import numpy as np
import pytest
import torch

from analysis_tools import analyze, library, plots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def example_norms(seed):
    # All examples have a zero second update; seed 1 also has zero first updates.
    return np.array([[0., 0., 10.], [10., 0., 20.], [20., 0., 30.]]) * (seed + 1) * np.array(
        [1 if seed == 0 else 0, 1, 1])


def synthetic_trajectory(model, batch, steps):
    ids = batch["context_input_ids"][:, 0].cpu().numpy()
    updates = torch.zeros(3, 5, 2, dtype=torch.float64)
    updates[:, :3, 0] = torch.from_numpy(example_norms(model.seed))
    updates[:, 3, 1] = 1000
    updates[:, 4, 0] = 2000
    states = [torch.zeros(len(ids), 1, 2, dtype=torch.float64)]
    for update in updates[ids, :steps].unbind(1):
        states.append(states[-1] + update[:, None, :])
    return states


@pytest.mark.parametrize("flag", [False, True])
def test_step_norm_boolean_default_and_no_config_mutation(flag):
    config = {"step_orth_proj": {"steps": 3, "add_step_norm": flag}}
    before = copy.deepcopy(config)
    assert analyze.step_orth_proj_settings(config)["add_step_norm"] is flag
    assert config == before
    assert analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 3}}).get(
        "add_step_norm", False) is False


@pytest.mark.parametrize("flag", [None, 0, 1, "true", "false", [], {}, np.bool_(True)])
def test_step_norm_rejects_nonboolean(flag):
    with pytest.raises(ValueError, match="add_step_norm.*true or false"):
        analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 3, "add_step_norm": flag}})


@pytest.mark.parametrize("spans,flag,batch_size", [
    (1, None, 1), (2, False, 2), ([1], True, 3),
    ([2, 1], None, 2), ([2, 1], False, 8), ([2, 1], True, 1),
])
def test_norm_shapes_all_examples_and_single_reused_trajectory(monkeypatch, spans, flag, batch_size):
    calls = []

    def collect(model, batch, steps):
        calls.append((batch["context_input_ids"][:, 0].tolist(), steps))
        return synthetic_trajectory(model, batch, steps)

    def forbidden(*args, **kwargs):
        pytest.fail("Norm collection must not evaluate or write additional memories")

    monkeypatch.setattr(analyze, "collect_trajectory", collect)
    monkeypatch.setattr(analyze, "score_memory", forbidden)
    model = SimpleNamespace(device="cpu", seed=0, write=forbidden, forward=forbidden)
    settings = dict(steps=3, proj_steps=spans, zero_step_policy="nan")
    if flag is not None:
        settings["add_step_norm"] = flag
    inputs = {"context_input_ids": torch.arange(3).reshape(3, 1)}
    result = analyze.step_orth_proj_run(model, inputs, torch.zeros(3, 1), settings, batch_size)
    span_list = spans if isinstance(spans, list) else [spans]
    assert calls == [(list(range(start, min(start + batch_size, 3))), 3 + max(span_list))
                     for start in range(0, 3, batch_size)]
    ratio_shape = (len(spans), 3) if isinstance(spans, list) else (3,)
    assert result["projection_ratio"].shape == ratio_shape
    assert result["projection_ratio_per_example"].shape == (3, *ratio_shape)
    assert result["step_norm"].shape == (3,)
    assert result["step_norm_per_example"].shape == (3, 3)
    np.testing.assert_array_equal(result["step_norm_per_example"], example_norms(0))
    np.testing.assert_array_equal(result["step_norm"], [10, 0, 20])
    assert np.isnan(result["projection_ratio_per_example"][0]).any()
    assert np.isnan(result["projection_ratio"][..., 1]).all()
    assert np.isfinite(result["step_norm_per_example"]).all()


@pytest.mark.parametrize("policy", ["error", "nan"])
def test_norm_is_flattened_l2_not_mean_token_norm_or_rms(monkeypatch, policy):
    calls = []

    def collect(model, batch, steps):
        calls.append(steps)
        update = torch.tensor([[[3., 0.], [0., -4.]], [[0., 12.], [-5., 0.]]], dtype=torch.float64)
        state = torch.zeros_like(update)
        states = [state]
        for multiplier in [1., 2., 100.]:
            states.append(states[-1] + multiplier * update)
        return states

    monkeypatch.setattr(analyze, "collect_trajectory", collect)
    result = analyze.step_orth_proj_run(
        SimpleNamespace(device="cpu"), {"context_input_ids": torch.zeros(2, 1)}, torch.zeros(2, 1),
        dict(steps=2, proj_steps=1, zero_step_policy=policy), 2)
    assert calls == [3]
    np.testing.assert_allclose(result["step_norm_per_example"], [[5, 10], [13, 26]])
    np.testing.assert_allclose(result["step_norm"], [9, 18])


@pytest.fixture
def report_factory(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    checkpoints = []
    for seed in range(2):
        run = tmp_path / f"seed_{seed}"
        checkpoint = run / "checkpoint-7"
        checkpoint.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"cli_args": {"seed": seed}}))
        (checkpoint / "config.json").write_text(json.dumps({"seed": seed}))
        (checkpoint / "model.safetensors").touch()
        checkpoints.append(str(checkpoint))

    def load(checkpoint, device, inner_lr, dtype):
        seed = json.loads((checkpoint.parent / "config.json").read_text())["seed"]
        return SimpleNamespace(seed=seed, device=device, lr=.2, write_objective="energy",
                               config=SimpleNamespace(to_dict=lambda: {"seed": seed}))

    monkeypatch.setattr(analyze, "load_model", load)
    monkeypatch.setattr(analyze, "load_data", lambda *args: (
        {"context_input_ids": torch.arange(3).reshape(3, 1)}, torch.zeros(3, 1), {"examples": 3}))
    monkeypatch.setattr(analyze, "collect_trajectory", synthetic_trajectory)
    rendered = []
    real_draw_plot = analyze.draw_plot

    def capture(path, config, summaries, settings, *args):
        rendered.append(copy.deepcopy(summaries))
        path.touch()

    monkeypatch.setattr(analyze, "draw_plot", capture)

    def make(spans=2, flag=False, run_count=2, legacy=False):
        if legacy:
            original = analyze.step_orth_proj_run

            def without_norm(*args, **kwargs):
                return {key: value for key, value in original(*args, **kwargs).items()
                        if not key.startswith("step_norm")}

            monkeypatch.setattr(analyze, "step_orth_proj_run", without_norm)
        config = dict(kind="step_orth_proj", output_dir=str(reports / "norms"), device="cpu",
                      examples=3, batch_size=2, dpi=25,
                      dataset={"path": "fake", "label": "synthetic"},
                      step_orth_proj=dict(steps=3, proj_steps=spans, zero_step_policy="nan",
                                          add_step_norm=flag, show_runs=True),
                      setups=[dict(label="Energy", runs=checkpoints[:run_count])])
        output = analyze.run_analysis(config)
        return SimpleNamespace(output=output, folder=output / "synthetic", config=config,
                               rendered=rendered, real_draw_plot=real_draw_plot)

    return make


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("spans,flag,run_count", [
    (2, False, 1), ([1], True, 1), ([2, 1], False, 2), ([2, 1], True, 2),
])
def test_normal_driver_saves_norms_csv_and_equal_run_sample_sd(report_factory, spans, flag, run_count):
    saved = report_factory(spans, flag, run_count)
    run_norms = np.stack([example_norms(seed).mean(0) for seed in range(run_count)])
    mean = run_norms.mean(0)
    std = run_norms.std(0, ddof=1) if run_count > 1 else np.zeros(3)
    for seed in range(run_count):
        with np.load(saved.folder / f"setup_0/run_{seed}.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["step_norm"], run_norms[seed])
            np.testing.assert_array_equal(arrays["step_norm_per_example"], example_norms(seed))
        details = json.loads((saved.folder / f"setup_0/run_{seed}.json").read_text())
        assert details["analysis_settings"]["add_step_norm"] is flag
    with np.load(saved.folder / "setup_0_mean.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["step_norm_mean"], mean)
        np.testing.assert_allclose(arrays["step_norm_std"], std)
        assert arrays["step_norm_mean"].shape == arrays["step_norm_std"].shape == (3,)
        assert "step_norm_per_example" not in arrays
        assert "step_norm_runs" not in arrays
        assert "projection_ratio_runs" not in arrays
        assert np.isnan(arrays["projection_ratio_mean"][..., 1]).all()
    rendered = saved.rendered[0][0]
    if flag:
        assert rendered["step_norm_runs"].shape == (run_count, 3)
        np.testing.assert_array_equal(rendered["step_norm_runs"], run_norms)
    else:
        assert "step_norm_runs" not in rendered
    span_list = spans if isinstance(spans, list) else [spans]
    per_run = read_csv(saved.folder / "per_run.csv")
    summary = read_csv(saved.folder / "summary.csv")
    assert len(per_run) == run_count * len(span_list) * 3
    assert len(summary) == len(span_list) * 3
    assert {int(row["proj_steps"]) for row in summary} == set(span_list)
    for row in per_run:
        assert float(row["step_norm"]) == run_norms[int(row["run"]), int(row["step"]) - 1]
    for row in summary:
        index = int(row["step"]) - 1
        assert float(row["step_norm_mean"]) == pytest.approx(mean[index])
        assert float(row["step_norm_std"]) == pytest.approx(std[index])


def test_driver_csv_norm_columns_are_optional_for_older_result_data(report_factory):
    saved = report_factory([2, 1], legacy=True)
    assert "step_norm" not in read_csv(saved.folder / "per_run.csv")[0]
    assert "step_norm_mean" not in read_csv(saved.folder / "summary.csv")[0]
    assert "step_norm_std" not in read_csv(saved.folder / "summary.csv")[0]


def forbid_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Redraw must use saved norms without evaluation or recomputation")

    for module in (analyze, library):
        for name in ("load_model", "load_data", "resolve_checkpoint", "collect_trajectory", "score_memory",
                     "step_orth_proj_run", "step_projection_ratios", "average_runs"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def snapshot(directory):
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("spans", [2, [2, 1]])
@pytest.mark.parametrize("overlay", [False, True])
def test_redraw_enables_norm_from_flag_false_without_touching_old_data(report_factory, monkeypatch, spans, overlay):
    saved = report_factory(spans, flag=False)
    before = snapshot(saved.output)
    forbid_evaluation(monkeypatch)
    analyze.redraw_analysis(dict(output_dir=str(saved.output),
                                step_orth_proj=dict(add_step_norm=True, show_runs=overlay)))
    summary = saved.rendered[-1][0]
    np.testing.assert_array_equal(summary["step_norm_mean"], [5, 0, 30])
    np.testing.assert_allclose(summary["step_norm_std"], [np.sqrt(50), 0, np.sqrt(200)])
    if overlay:
        np.testing.assert_array_equal(summary["step_norm_runs"], [[10, 0, 20], [0, 0, 40]])
    else:
        assert "step_norm_runs" not in summary
    for path, contents in before.items():
        original = saved.output / path
        assert (original.read_bytes(), original.stat().st_mtime_ns) == contents
    metadata = json.loads((saved.folder / "step_orth_proj_redraw_1.json").read_text())
    assert metadata["plot_settings"]["add_step_norm"] is True
    assert json.loads((saved.output / "config.json").read_text())["step_orth_proj"]["add_step_norm"] is False


@pytest.mark.parametrize("key,bad", [
    ("step_norm_mean", "missing"), ("step_norm_std", "missing"), ("step_norm", "missing"),
    ("step_norm_mean", "shape"), ("step_norm_std", "shape"), ("step_norm", "shape"),
    ("step_norm_mean", "infinite"), ("step_norm_std", "negative"), ("step_norm", "nan"),
])
def test_redraw_rejects_missing_or_invalid_saved_norms(report_factory, monkeypatch, key, bad):
    saved = report_factory()
    path = saved.folder / ("setup_0/run_0.npz" if key == "step_norm" else "setup_0_mean.npz")
    with np.load(path, allow_pickle=False) as arrays:
        data = dict(arrays)
    if bad == "missing":
        del data[key]
    elif bad == "shape":
        data[key] = np.zeros((1, 3))
    else:
        data[key] = data[key].copy()
        data[key][0] = {"infinite": np.inf, "negative": -1, "nan": np.nan}[bad]
    np.savez_compressed(path, **data)
    before = snapshot(saved.output)
    forbid_evaluation(monkeypatch)
    with pytest.raises(ValueError, match="(?i)(step_norm|WRITE-step norm)") as error:
        analyze.redraw_analysis(dict(output_dir=str(saved.output),
                                    step_orth_proj=dict(add_step_norm=True, show_runs=True)))
    if bad == "missing":
        assert "new" in str(error.value).lower()
    assert snapshot(saved.output) == before
    assert len(saved.rendered) == 1


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return figures


@pytest.mark.parametrize("style", ["band", "errorbar"])
def test_real_redraw_keeps_zero_norm_when_all_projection_ratios_missing(
        report_factory, monkeypatch, captured_figures, style):
    saved = report_factory([2, 1], flag=False)
    before = snapshot(saved.output)
    forbid_evaluation(monkeypatch)
    monkeypatch.setattr(analyze, "draw_plot", saved.real_draw_plot)
    analyze.redraw_analysis(dict(output_dir=str(saved.output), step_orth_proj=dict(
        add_step_norm=True, show_runs=True, uncertainty_style=style)))
    figure = captured_figures[-1]
    assert len(figure.axes) == 3
    norm_ax = figure.axes[0]
    if style == "band":
        mean = next(line for line in norm_ax.lines if line.get_label() == "Energy (2 runs)")
    else:
        mean = next(container.lines[0] for container in norm_ax.containers
                    if isinstance(container, ErrorbarContainer))
    np.testing.assert_array_equal(mean.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(mean.get_ydata(), [5, np.nan, 30], equal_nan=True)
    assert norm_ax.get_yscale() == "log"
    assert norm_ax.get_ylim()[0] > 0
    markers = [collection for collection in norm_ax.collections if not isinstance(collection, PolyCollection)]
    if style == "errorbar":
        markers = [collection for collection in markers if hasattr(collection, "get_sizes")]
    assert any(np.any(collection.get_offsets()[:, 0] == 2) for collection in markers)
    assert len([line for line in norm_ax.lines if line.get_alpha() == .3]) == 2
    assert (saved.folder / "step_orth_proj_redraw_1.png").stat().st_size > 0
    for name, contents in before.items():
        path = saved.output / name
        assert (path.read_bytes(), path.stat().st_mtime_ns) == contents


@pytest.mark.parametrize("spans,style,add_norm", [
    (2, "band", True), (2, "errorbar", False), ([1], "errorbar", True),
    ([1], "band", False), ([2, 1], "band", True), ([2, 1], "errorbar", True),
])
def test_real_figure_norm_panel_axes_colors_uncertainty_and_fonts(
        tmp_path, captured_figures, spans, style, add_norm):
    shape = (len(spans), 3) if isinstance(spans, list) else (3,)
    setups = []
    for index, label in enumerate(("CE", "Energy")):
        values = np.broadcast_to([.2, .5, .8], shape).copy()
        setups.append(dict(label=label, runs=2, projection_ratio_mean=values,
                           projection_ratio_std=np.full(shape, .3),
                           projection_ratio_runs=np.stack([values - .1, values + .1]),
                           step_norm_mean=np.array([10., 15., 20.]) + index,
                           step_norm_std=np.array([12., 2., 3.]),
                           step_norm_runs=np.array([[9., 14., 19.], [11., 16., 21.]]) + index))
    settings = dict(steps=3, proj_steps=spans, add_step_norm=add_norm, uncertainty_style=style,
                    show_runs=True, dash_start=2, show_n_runs=True, zero_step_policy="nan")
    config = dict(kind="step_orth_proj", dpi=25, subtitle_font_size=17, tick_font_size=11,
                  axis_label_font_size=13, legend_font_size=9)
    destination = tmp_path / "norms.png"
    analyze.draw_plot(destination, config, setups, settings)
    assert destination.stat().st_size > 0
    figure = captured_figures[-1]
    panels = len(spans) if isinstance(spans, list) else 1
    assert len(figure.axes) == panels + int(add_norm)
    first = figure.axes[0]
    for ax in figure.axes:
        if ax is not first:
            assert first.get_shared_x_axes().joined(first, ax)
        assert ax.get_xlabel() == "WRITE step k"
        assert ax.xaxis.label.get_fontsize() == 13
        assert ax.title.get_fontsize() == 17
        assert all(tick.get_fontsize() == 11 for tick in ax.get_xticklabels())
        boundaries = [line for line in ax.lines if line.get_linestyle() == "--"
                      and np.array_equal(line.get_xdata(), [2, 2])]
        assert len(boundaries) == 1
        for setup_index in range(2):
            if style == "band":
                mean_lines = [line for line in ax.lines if line.get_label() in ("CE (2 runs)", "Energy (2 runs)")]
            else:
                mean_lines = [container.lines[0] for container in ax.containers
                              if isinstance(container, ErrorbarContainer)]
            assert len(mean_lines) == 2
            color = mean_lines[setup_index].get_color()
            assert to_rgba(color) == to_rgba(plt.rcParams["axes.prop_cycle"].by_key()["color"][setup_index])
            tails = [line for line in ax.lines if line.get_linestyle() == "--"
                     and line.get_color() == color and np.array_equal(line.get_xdata(), [2, 3])]
            assert len(tails) == 3  # One mean tail and two same-color individual-run tails.
            assert all(line.get_label() == "_nolegend_" for line in tails)
    projection_axes = figure.axes[int(add_norm):]
    for ax in projection_axes:
        assert ax.get_yscale() == "linear"
    for ax in projection_axes[1:]:
        assert projection_axes[0].get_shared_y_axes().joined(projection_axes[0], ax)
    legend = first.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["CE (2 runs)", "Energy (2 runs)"]
    assert all(text.get_fontsize() == 9 for text in legend.get_texts())
    assert all(ax.get_legend() is None for ax in figure.axes[1:])
    if not add_norm:
        return
    norm_ax = figure.axes[0]
    assert not norm_ax.get_shared_y_axes().joined(norm_ax, projection_axes[0])
    assert norm_ax.get_yscale() == "log"
    assert norm_ax.get_ylim()[0] > 0
    assert norm_ax.get_ylim()[1] > 23
    assert "norm" in norm_ax.get_title().lower()
    assert "L2" in norm_ax.get_ylabel()
    if style == "band":
        bands = [artist for artist in norm_ax.collections if isinstance(artist, PolyCollection)]
        assert len(bands) == 2
        for index, band in enumerate(bands):
            vertices = np.concatenate([path.vertices for path in band.get_paths()])
            assert vertices[:, 1].min() == norm_ax.get_ylim()[0]
            assert vertices[:, 1].max() == 23 + index
        for ax in projection_axes:
            for band in ax.collections:
                if isinstance(band, PolyCollection):
                    y = np.concatenate([path.vertices[:, 1] for path in band.get_paths()])
                    assert y.min() >= 0 and y.max() <= 1
    else:
        containers = [container for container in norm_ax.containers if isinstance(container, ErrorbarContainer)]
        assert len(containers) == 2
        for index, container in enumerate(containers):
            segments = container.lines[2][0].get_segments()
            np.testing.assert_array_equal(segments[-1], [[3, 17 + index], [3, 23 + index]])
    assert figure._supxlabel is None


@pytest.mark.parametrize("style", ["band", "errorbar"])
def test_all_zero_norms_have_explicit_log_floor_markers(tmp_path, captured_figures, style):
    setup = dict(label="Stopped", runs=1, projection_ratio_mean=np.full(3, np.nan),
                 projection_ratio_std=np.full(3, np.nan), projection_ratio_runs=np.full((1, 3), np.nan),
                 step_norm_mean=np.zeros(3), step_norm_std=np.zeros(3), step_norm_runs=np.zeros((1, 3)))
    original = setup["step_norm_mean"].copy()
    plots.step_orth_proj(tmp_path / "zero_norms.png", [setup], 3, dpi=25, add_step_norm=True,
                        uncertainty_style=style, show_runs=True, zero_step_policy="nan")
    figure = captured_figures[-1]
    ax = figure.axes[0]
    assert ax.get_yscale() == "log"
    assert 0 < ax.get_ylim()[0] < ax.get_ylim()[1]
    markers = [collection for collection in ax.collections
               if hasattr(collection, "get_sizes") and not isinstance(collection, PolyCollection)]
    assert len(markers) == 1
    np.testing.assert_array_equal(markers[0].get_offsets()[:, 0], [1, 2, 3])
    np.testing.assert_allclose(markers[0].get_offsets()[:, 1], ax.get_ylim()[0])
    assert figure._supxlabel is None
    np.testing.assert_array_equal(setup["step_norm_mean"], original)
