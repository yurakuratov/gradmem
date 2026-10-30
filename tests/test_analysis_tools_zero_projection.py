"""Zero-current-WRITE projection contracts, using CPU-only synthetic states."""

import copy
import csv
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.path import Path as ArtistPath
import numpy as np
import pytest
import torch

from analysis_tools import analyze, library, plots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def states_from_updates(updates):
    updates = torch.as_tensor(updates, dtype=torch.float64)
    states = [torch.zeros(updates.shape[0], 1, updates.shape[-1], dtype=torch.float64)]
    for update in updates.unbind(1):
        states.append(states[-1] + update[:, None, :])
    return states


@pytest.mark.parametrize("policy", ["error", "nan"])
def test_zero_following_columns_are_valid_not_missing(policy):
    states = states_from_updates([[[3, 4], [0, 0], [1, 0]]])
    torch.testing.assert_close(library.step_projection_ratios(states, 1, "nan"),
                               torch.tensor([[0., float("nan")]], dtype=torch.float64), equal_nan=True)
    torch.testing.assert_close(library.step_projection_ratios(states, 2, policy),
                               torch.tensor([[.6]], dtype=torch.float64))
    all_zero_span = states_from_updates([[[3, 4], [0, 0], [0, 0]]])
    torch.testing.assert_close(library.step_projection_ratios(all_zero_span, 2, policy),
                               torch.zeros(1, 1, dtype=torch.float64))


def test_nan_masks_only_current_zero_and_handles_rank_deficient_span():
    states = states_from_updates([
        [[0, 0], [1, 0], [0, 0], [2, 0], [0, 0]],
        [[3, 4], [1, 0], [2, 0], [0, 0], [0, 1]],
    ])
    expected = torch.tensor([[float("nan"), 1., float("nan")], [.6, 1., 0.]], dtype=torch.float64)
    actual = library.step_projection_ratios(states, zero_step_policy="nan")
    torch.testing.assert_close(actual, expected, equal_nan=True)
    assert actual.device.type == "cpu"


@pytest.mark.parametrize("explicit", [False, True])
def test_strict_policy_reports_batch_local_examples_and_one_based_steps(explicit):
    states = states_from_updates([
        [[1, 0], [0, 0], [1, 0], [0, 0]],
        [[0, 0], [1, 0], [0, 0], [1, 0]],
    ])
    kwargs = {"zero_step_policy": "error"} if explicit else {}
    with pytest.raises(ValueError, match="Zero-length current WRITE step") as error:
        library.step_projection_ratios(states, 1, **kwargs)
    message = str(error.value)
    assert "3 affected" in message
    assert "batch-local example, one-based WRITE step" in message
    assert "[(0, 2), (1, 1), (1, 3)]" in message


@pytest.mark.parametrize("policy,bad,position", [
    ("error", float("nan"), 0), ("error", float("inf"), 2), ("error", -float("inf"), 0),
    ("nan", float("nan"), 2), ("nan", float("inf"), 0), ("nan", -float("inf"), 2),
])
def test_nonfinite_current_or_following_update_still_fails(policy, bad, position):
    updates = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]], dtype=torch.float64)
    updates[0, position, 0] = bad
    with pytest.raises(ValueError, match="Nonfinite WRITE updates"):
        library.step_projection_ratios(states_from_updates(updates), 2, policy)


@pytest.mark.parametrize("policy", [None, "NaN", "ignore", "", True, 0, [], {}])
def test_invalid_zero_policies_rejected_by_library_and_settings(policy):
    with pytest.raises(ValueError, match="zero_step_policy"):
        library.step_projection_ratios(states_from_updates([[[1, 0], [0, 1]]]), 1, policy)
    with pytest.raises(ValueError, match="zero_step_policy"):
        analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 2, "zero_step_policy": policy}})


@pytest.mark.parametrize("spans", [1, 2, [1], [2, 1]])
@pytest.mark.parametrize("policy", ["error", "nan"])
def test_settings_preserve_shape_order_and_do_not_mutate_config(spans, policy):
    config = {"step_orth_proj": {"steps": 4, "proj_steps": spans, "zero_step_policy": policy}}
    before = copy.deepcopy(config)
    assert analyze.step_orth_proj_settings(config) == before["step_orth_proj"]
    assert config == before
    assert analyze.step_orth_proj_settings({"step_orth_proj": {"steps": 4}}) == {
        "steps": 4, "proj_steps": 2}


def synthetic_trajectory(model, batch, steps):
    x, y, z = [1, 0], [0, 1], [0, 0]
    updates = [
        [[x, z, y, z, x, y], [x, z, x, z, z, y], [z, z, y, z, x, y]],
        [[y, z, x, x, y, x], [z, z, y, y, x, y], [z, z, z, z, x, y]],
    ]
    indices = batch["context_input_ids"][:, 0]
    return states_from_updates(torch.tensor(updates[model.seed])[indices, :steps])


def expected_run(seed, spans):
    means = np.array([[0, np.nan, 0, np.nan], [.5, np.nan, 0, np.nan]]) if seed == 0 else np.array([
        [0, np.nan, 1, 0], [0, np.nan, 1, 1]])
    counts = np.array([2, 0, 3, 0] if seed == 0 else [1, 0, 2, 2])
    if isinstance(spans, list):
        return means[np.array(spans) - 1], np.broadcast_to(counts, (len(spans), 4))
    return means[spans - 1], counts


@pytest.mark.parametrize("batch_size,spans,seed", [
    (1, 1, 0), (2, 2, 1), (3, [1], 0), (8, [1], 1), (2, [2, 1], 0), (1, [2, 1], 1),
])
def test_conditional_run_counts_batch_invariance_and_scalar_list_parity(monkeypatch, batch_size, spans, seed):
    calls = []

    def collect(model, batch, steps):
        calls.append(steps)
        return synthetic_trajectory(model, batch, steps)

    monkeypatch.setattr(analyze, "collect_trajectory", collect)
    model = SimpleNamespace(device="cpu", seed=seed)
    inputs = {"context_input_ids": torch.arange(3).reshape(3, 1)}
    labels = torch.zeros(3, 1)
    settings = dict(steps=4, proj_steps=spans, zero_step_policy="nan")
    result = analyze.step_orth_proj_run(model, inputs, labels, settings, batch_size)
    mean, counts = expected_run(seed, spans)
    assert calls == [4 + max(spans if isinstance(spans, list) else [spans])] * ((3 + batch_size - 1) // batch_size)
    assert result["projection_ratio"].shape == mean.shape
    assert result["projection_ratio_per_example"].shape == (3, *mean.shape)
    np.testing.assert_allclose(result["projection_ratio"], mean, equal_nan=True, atol=1e-12)
    np.testing.assert_array_equal(result["projection_valid_examples"], counts)
    np.testing.assert_array_equal(result["projection_total_examples"], np.full(mean.shape, 3))
    np.testing.assert_array_equal(np.isfinite(result["projection_ratio_per_example"]).sum(0), counts)
    np.testing.assert_array_equal(np.isnan(result["projection_ratio"]), counts == 0)
    reference = analyze.step_orth_proj_run(model, inputs, labels, settings, 3)
    for key in result:
        np.testing.assert_allclose(result[key], reference[key], equal_nan=True)
    if isinstance(spans, list):
        for panel, span in enumerate(spans):
            scalar = analyze.step_orth_proj_run(model, inputs, labels, dict(settings, proj_steps=span), 2)
            for key in result:
                # Step norms have no projection-span axis.
                if key.startswith("step_norm"):
                    actual = result[key]
                else:
                    actual = result[key][:, panel] if key.endswith("_per_example") else result[key][panel]
                np.testing.assert_allclose(actual, scalar[key], equal_nan=True)


def test_run_default_remains_strict(monkeypatch):
    monkeypatch.setattr(analyze, "collect_trajectory", synthetic_trajectory)
    with pytest.raises(ValueError, match="Zero-length current WRITE step"):
        analyze.step_orth_proj_run(SimpleNamespace(device="cpu", seed=0),
                                  {"context_input_ids": torch.arange(3).reshape(3, 1)},
                                  torch.zeros(3, 1), {"steps": 4}, 2)


def test_average_runs_different_missing_patterns_sample_sd_and_all_missing():
    runs = [np.array([[1, np.nan, 2, np.nan], [0, 2, np.nan, 4]]),
            np.array([[3, 7, np.nan, np.nan], [1, np.nan, np.nan, 8]]),
            np.array([[5, np.nan, 6, np.nan], [np.nan, 6, np.nan, 6]])]
    mean, std = analyze.average_runs(runs, allow_nan=True)
    np.testing.assert_allclose(mean, [[3, 7, 4, np.nan], [.5, 4, np.nan, 6]], equal_nan=True)
    np.testing.assert_allclose(std, [[2, 0, np.sqrt(8), np.nan],
                                     [np.sqrt(.5), np.sqrt(8), np.nan, 2]], equal_nan=True)
    one_mean, one_std = analyze.average_runs([runs[0]], allow_nan=True)
    np.testing.assert_allclose(one_mean, runs[0], equal_nan=True)
    np.testing.assert_allclose(one_std, np.where(np.isfinite(runs[0]), 0, np.nan), equal_nan=True)
    strict_mean, _ = analyze.average_runs(runs)
    assert np.isnan(strict_mean[0, 1])


@pytest.fixture
def report(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)
    checkpoints = []
    for seed in (0, 1):
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
                               config=SimpleNamespace(to_dict=lambda: {"seed": seed, "K": 4}))

    monkeypatch.setattr(analyze, "load_model", load)
    monkeypatch.setattr(analyze, "load_data", lambda *args: (
        {"context_input_ids": torch.arange(3).reshape(3, 1)}, torch.zeros(3, 1), {"examples": 3}))
    monkeypatch.setattr(analyze, "collect_trajectory", synthetic_trajectory)

    def make(spans):
        config = dict(kind="step_orth_proj", output_dir=str(reports / "zero_projection"),
                      device="cpu", examples=3, example_offset=5, batch_size=2, dpi=25,
                      dataset={"path": "fake", "label": "synthetic"},
                      step_orth_proj=dict(steps=4, proj_steps=spans, zero_step_policy="nan",
                                          uncertainty_style="band", show_runs=True),
                      setups=[dict(label="two seeds", runs=checkpoints)])
        output = analyze.run_analysis(config)
        return SimpleNamespace(output=output, folder=output / "synthetic", config=config)

    return make


def forbid_computation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Redraw must not load models/data or recompute numerical results")

    for module in (analyze, library):
        for name in ("load_model", "load_data", "resolve_checkpoint", "collect_trajectory", "score_memory",
                     "step_projection_ratios", "step_orth_proj_run", "average_runs"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("spans", [1, [2, 1]])
def test_normal_run_saves_conditional_coverage_then_redraws_without_computation(report, monkeypatch, spans):
    figures = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    saved = report(spans)
    manifest = json.loads((saved.output / "manifest.json").read_text())
    assert "conditional" in manifest["aggregation"]
    assert "valid runs" in manifest["aggregation"]
    assert "sample SD" in manifest["aggregation"]
    assert "NaN" in manifest["zero_step_policy"]
    assert "counts" in manifest["coverage"]
    assert manifest["example_indices"] == [5, 6, 7]
    assert json.loads((saved.output / "config.json").read_text()) == saved.config
    run_values = []
    for seed in (0, 1):
        mean, counts = expected_run(seed, spans)
        with np.load(saved.folder / f"setup_0/run_{seed}.npz") as arrays:
            np.testing.assert_allclose(arrays["projection_ratio"], mean, equal_nan=True)
            np.testing.assert_array_equal(arrays["projection_valid_examples"], counts)
            np.testing.assert_array_equal(arrays["projection_total_examples"], np.full(mean.shape, 3))
            np.testing.assert_array_equal(np.isfinite(arrays["projection_ratio_per_example"]).sum(0), counts)
            np.testing.assert_array_equal(arrays["steps"], [1, 2, 3, 4])
            run_values.append(arrays["projection_ratio"])
        details = json.loads((saved.folder / f"setup_0/run_{seed}.json").read_text())
        assert details["analysis_settings"]["zero_step_policy"] == "nan"
        assert details["seed"] == seed
    mean = np.array([[0, np.nan, .5, 0], [.25, np.nan, .5, 1]])
    std = np.array([[0, np.nan, np.sqrt(.5), 0], [np.sqrt(.125), np.nan, np.sqrt(.5), 0]])
    selection = np.array(spans) - 1 if isinstance(spans, list) else spans - 1
    mean, std = mean[selection], std[selection]
    valid_runs = np.broadcast_to([2, 0, 2, 1], mean.shape)
    valid_examples = np.broadcast_to([1.5, 0, 2.5, 1], mean.shape)
    with np.load(saved.folder / "setup_0_mean.npz") as arrays:
        np.testing.assert_allclose(arrays["projection_ratio_mean"], mean, equal_nan=True)
        np.testing.assert_allclose(arrays["projection_ratio_std"], std, equal_nan=True)
        np.testing.assert_array_equal(arrays["projection_valid_runs"], valid_runs)
        np.testing.assert_allclose(arrays["projection_valid_examples"], valid_examples)
        np.testing.assert_array_equal(arrays["projection_total_examples"], np.full(mean.shape, 3))
    with (saved.folder / "per_run.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    panels = spans if isinstance(spans, list) else [spans]
    assert len(rows) == 2 * len(panels) * 4
    for row in rows:
        values, counts = expected_run(int(row["run"]), int(row["proj_steps"]))
        index = int(row["step"]) - 1
        np.testing.assert_allclose(float(row["projection_ratio"]), values[index], equal_nan=True)
        assert int(row["valid_examples"]) == counts[index]
        assert int(row["total_examples"]) == 3
    with (saved.folder / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(panels) * 4
    for key, expected in (("projection_ratio_mean", mean), ("projection_ratio_std", std),
                          ("valid_runs", valid_runs), ("valid_examples_mean", valid_examples),
                          ("total_examples_mean", np.full(mean.shape, 3))):
        np.testing.assert_allclose([float(row[key]) for row in rows], expected.ravel(), equal_nan=True)
    originals = {p: p.read_bytes() for p in saved.output.rglob("*") if p.is_file()}
    forbid_computation(monkeypatch)
    analyze.redraw_analysis(dict(output_dir=str(saved.output), step_orth_proj={"show_runs": True}))
    assert (saved.folder / "step_orth_proj_redraw_1.png").read_bytes().startswith(b"\x89PNG")
    for path, content in originals.items():
        assert path.read_bytes() == content
    for figure in figures:
        assert figure._supxlabel is None
        for panel, ax in enumerate(figure.axes):
            np.testing.assert_allclose(ax.lines[0].get_ydata(), np.atleast_2d(mean)[panel], equal_nan=True)
            # A finite-zero, NaN, finite sequence must restart rather than bridge the gap.
            commands = [code for _, code in ax.lines[0].get_path().iter_segments()]
            assert commands.count(ArtistPath.MOVETO) == 2
            rings = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
            assert len(rings) == 1
            np.testing.assert_allclose(rings[0].get_offsets(),
                                       np.column_stack(([1, 3, 4], np.atleast_2d(mean)[panel, [0, 2, 3]])))
            np.testing.assert_array_equal(rings[0].get_edgecolors(), [[1, 0, 0, 1]])
            assert len(rings[0].get_facecolors()) == 0
            overlays = [line for line in ax.lines if line.get_alpha() == .3]
            assert len(overlays) == 2
            for seed, line in enumerate(overlays):
                expected = np.atleast_2d(run_values[seed])[panel]
                np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)


@pytest.mark.parametrize("style,show_runs,spans", [
    ("band", False, 1), ("errorbar", True, 1),
    ("band", True, [2, 1]), ("errorbar", False, [2, 1]),
])
def test_only_incomplete_finite_means_get_red_rings(tmp_path, monkeypatch, style, show_runs, spans):
    shape = (len(spans), 4) if isinstance(spans, list) else (4,)
    def expanded(values):
        return np.broadcast_to(values, shape).copy()
    setups = []
    for label, missing in (("Complete", False), ("Incomplete", True)):
        mean = expanded([.2, .4, np.nan if missing else .6, .8])
        run_values = np.stack([mean, mean])
        if missing:
            run_values[1, ..., -1] = np.nan
        setups.append(dict(label=label, runs=2, projection_ratio_mean=mean,
                           projection_ratio_std=np.where(np.isfinite(mean), .05, np.nan),
                           projection_valid_examples=expanded([4, 3, 0, 2] if missing else [4, 4, 4, 4]),
                           projection_total_examples=expanded([4, 4, 4, 4]),
                           projection_valid_runs=expanded([2, 2, 0, 1] if missing else [2, 2, 2, 2]),
                           step_norm_mean=np.array([2., 1., .5, .25]), step_norm_std=np.full(4, .01),
                           step_norm_runs=np.array([[2., 1., .5, .25], [2., 1., .5, .25]])))
        if show_runs:
            setups[-1]["projection_ratio_runs"] = run_values
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    plots.step_orth_proj(tmp_path / "rings.png", setups, 4, proj_steps=spans,
                        uncertainty_style=style, show_runs=show_runs, zero_step_policy="nan",
                        add_step_norm=True, dash_start=2, show_n_runs=False)
    figure = figures[0]
    assert not any(isinstance(collection, PathCollection) and collection.get_zorder() == 5
                   for collection in figure.axes[0].collections)
    for ax in figure.axes[1:]:
        rings = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
        assert len(rings) == 1
        np.testing.assert_allclose(rings[0].get_offsets(), [[2, .4], [4, .8]])
        np.testing.assert_array_equal(rings[0].get_edgecolors(), [[1, 0, 0, 1]])
        assert len(rings[0].get_facecolors()) == 0
        assert ax.get_legend_handles_labels()[1] == ["Complete", "Incomplete"]
    assert figure._supxlabel is None


@pytest.mark.parametrize("damage", [
    "mean_inf", "std_inf", "missing_counts", "count_shape", "negative_valid", "excess_valid",
    "zero_total", "negative_runs", "excess_runs", "count_nan", "finite_without_runs",
    "nan_with_runs", "zero_examples_with_runs", "examples_without_runs", "partial_missing_stats",
    "fractional_runs", "aggregate_steps", "aggregate_spans", "run_steps", "run_spans", "run_inf",
])
def test_redraw_rejects_invalid_nanpolicy_statistics_coverage_and_axes(report, monkeypatch, damage):
    saved = report([2, 1])
    run_damage = damage.startswith("run_")
    path = saved.folder / ("setup_0/run_0.npz" if run_damage else "setup_0_mean.npz")
    with np.load(path) as arrays:
        values = {key: value.copy() for key, value in arrays.items()}
    if damage in ("mean_inf", "std_inf"):
        values["projection_ratio_" + damage.split("_")[0]][0, 0] = np.inf
    elif damage == "missing_counts":
        del values["projection_valid_examples"]
    elif damage == "count_shape":
        values["projection_valid_runs"] = np.zeros(4)
    elif damage in ("negative_valid", "excess_valid", "count_nan", "zero_examples_with_runs"):
        values["projection_valid_examples"][0, 0] = {
            "negative_valid": -1, "excess_valid": 4, "count_nan": np.nan, "zero_examples_with_runs": 0}[damage]
    elif damage == "zero_total":
        values["projection_total_examples"][0, 0] = 0
    elif damage in ("negative_runs", "excess_runs", "finite_without_runs", "nan_with_runs", "fractional_runs"):
        values["projection_valid_runs"] = values["projection_valid_runs"].astype(float)
        point = (0, 1) if damage == "nan_with_runs" else (0, 0)
        values["projection_valid_runs"][point] = {
            "negative_runs": -1, "excess_runs": 3, "finite_without_runs": 0,
            "nan_with_runs": 1, "fractional_runs": .5}[damage]
    elif damage == "examples_without_runs":
        values["projection_valid_examples"][0, 1] = 1
    elif damage == "partial_missing_stats":
        values["projection_ratio_std"][0, 1] = 0
    elif damage in ("aggregate_steps", "run_steps"):
        values["steps"][0] = 0
    elif damage in ("aggregate_spans", "run_spans"):
        values["proj_steps"] = values["proj_steps"][::-1]
    elif damage == "run_inf":
        values["projection_ratio"][0, 0] = np.inf
    else:
        raise AssertionError(damage)
    np.savez_compressed(path, **values)
    forbid_computation(monkeypatch)
    with pytest.raises(ValueError, match="[Pp]rojection|step_orth_proj"):
        analyze.redraw_analysis(dict(output_dir=str(saved.output), step_orth_proj={"show_runs": True}))
    assert not list(saved.folder.glob("step_orth_proj_redraw_*"))
