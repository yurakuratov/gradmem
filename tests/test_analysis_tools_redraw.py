"""Saved-report redraw contracts, using synthetic artifacts and real CPU plots."""

import copy
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest

from analysis_tools import analyze, library, plots


KINDS = (
    "landscape", "extrapolation", "k_lr_sweep", "update_svd",
    "step_orth_proj", "memory_search",
)
METRICS = ("inner_loss", "outer_loss", "token_accuracy", "exact_match")
SETTINGS = {
    "landscape": {"basis_steps": [1, 2], "steps": 3, "grid_size": 3},
    "extrapolation": {"steps": [0, 1, 3]},
    "k_lr_sweep": {"steps": [0, 2, 3], "inner_lrs": [.1, .4]},
    "update_svd": {"steps": 3, "energy_threshold": .95},
    "step_orth_proj": {"steps": 3, "proj_steps": 2},
    "memory_search": {"steps": 2, "num_samples": 4,
                      "radius_scale": .25, "quantiles": [.5, .75, 1.]},
}


def snapshot(directory):
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*") if path.is_file()
    }


def assert_untouched(directory, before):
    for name, contents in before.items():
        path = directory / name
        assert path.is_file(), name
        assert (path.read_bytes(), path.stat().st_mtime_ns) == contents, name


def write_table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(autouse=True)
def no_computation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Redraw must not compute results or load models/checkpoints/data")

    names = (
        "run_analysis", "resolve_checkpoint", "load_model", "load_data",
        "collect_trajectory", "score_memory", "prediction_metrics", "set_steps",
        "memory_basis", "finite_difference_cosine", "gradient_cosine",
        "alignment_statistics", "retained_energy_rank", "update_singular_values",
        "step_projection_ratios", "beneficial_outcomes", "average_runs",
        "landscape_run", "extrapolation_run", "k_lr_sweep_run", "update_svd_run",
        "step_orth_proj_run", "memory_search_run",
    )
    for module in (analyze, library):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    yield
    plt.close("all")


@pytest.fixture
def report_factory(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(analyze, "REPORTS", reports)

    def make(kind, multiple_datasets=True):
        output = reports / kind
        output.mkdir()
        datasets = ["unavailable/data/mqar_test", {"label": "Long context / eval",
                                                "path": "unavailable/other_data"}]
        if not multiple_datasets:
            datasets = datasets[:1]
        setups = [dict(label=label, runs=[{"path": f"missing/{index}/checkpoint-10"},
                                         f"missing/{index}/checkpoint-20"])
                  for index, label in enumerate(("CE baseline", "Energy shaped"))]
        config = dict(kind=kind, output_dir=str(output), setups=setups,
                      examples=8, batch_size=2, device="cpu", seed=42,
                      dpi=35, figsize=[10, 8], **{kind: copy.deepcopy(SETTINGS[kind])})
        if multiple_datasets:
            config["datasets"] = datasets
        else:
            config["dataset"] = datasets[0]
        manifest = {"setups": []}
        for index, setup in enumerate(setups):
            runs = [dict(path=f"missing/{index}/checkpoint-{10 * (run + 1)}/model.safetensors",
                         seed=run, sha256="0" * 64, inner_lr=None) for run in range(2)]
            for run in runs:
                if kind == "landscape":
                    # Deliberately differ from the global config's basis.
                    run[kind] = dict(SETTINGS[kind], basis_steps=[2, 3])
                elif kind == "k_lr_sweep":
                    run[kind] = copy.deepcopy(SETTINGS[kind])
            manifest["setups"].append(dict(index=index, label=setup["label"], runs=runs))
        (output / "config.json").write_text(json.dumps(config, indent=2))
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        alpha, beta = np.array([-1., 0., 2.]), np.array([-2., 0., 1., 3.])
        folders, expected = [], []
        stem = "landscapes" if kind == "landscape" else kind
        for dataset in datasets:
            label = Path(dataset).name if isinstance(dataset, str) else dataset["label"]
            folder = output / re.sub(r"[^A-Za-z0-9_.-]", "_", label)
            folder.mkdir()
            folders.append(folder)
            summaries, rows = [], []
            for index, setup in enumerate(setups):
                arrays = {}
                extra = {}
                if kind == "landscape":
                    x, y = np.meshgrid(alpha, beta)
                    arrays = dict(inner_loss=3 + x * .1 + y * .2 + index,
                                  outer_loss=5 + x * .2 - y * .1 + index,
                                  alignment=np.linspace(-.8, .8, x.size).reshape(x.shape),
                                  trajectory=np.array([[-.5, -.5], [0., 0.], [1., 0.], [1., 1.]]),
                                  trajectory_std=np.zeros((4, 2)), trajectory_runs=np.array([2, 2, 2, 1]))
                    extra["basis_steps"] = [2, 3]
                elif kind in ("extrapolation", "k_lr_sweep"):
                    shape = (3, 2) if kind == "k_lr_sweep" else (3,)
                    for metric in METRICS[1:]:
                        mean = np.arange(np.prod(shape)).reshape(shape) * .02 + .2 + .1 * index
                        if metric == "outer_loss":
                            mean += 2
                        arrays.update({metric: mean, metric + "_mean": mean,
                                       metric + "_std": np.full(shape, .01)})
                    arrays["steps"] = np.array(SETTINGS[kind]["steps"])
                    if kind == "k_lr_sweep":
                        arrays["inner_lrs"] = np.array(SETTINGS[kind]["inner_lrs"])
                        extra["trained_settings"] = [dict(K=2, inner_lr=.4)] * 2
                elif kind == "step_orth_proj":
                    values = np.array([.3, .5, .7]) + .02 * index
                    arrays = dict(projection_ratio=values, projection_ratio_mean=values,
                                  projection_ratio_std=np.full(3, .01), steps=np.arange(1, 4))
                elif kind == "update_svd":
                    values = np.array([3., 2., 1.]) + index
                    arrays = dict(singular_values=values, singular_values_mean=values,
                                  singular_values_std=np.full(3, .1), ranks=np.arange(1, 4))
                    # Use the recorded value, not a recomputation from the spectrum.
                    extra["retained_rank"] = 1 + index
                    rows.extend(dict(setup=setup["label"], runs=2, K=3, rank=rank,
                                     singular_value_mean=float(value), singular_value_std=.1,
                                     energy_threshold=.95, retained_rank=extra["retained_rank"])
                                for rank, value in enumerate(values, 1))
                else:
                    for metric in METRICS:
                        baseline = .3 if metric in ("token_accuracy", "exact_match") else 3.
                        baseline += .05 * index
                        sign = 1 if metric in ("token_accuracy", "exact_match") else -1
                        outcomes = baseline + sign * np.array([.01, .03, .05])
                        arrays[metric + "_outcomes"] = outcomes
                        arrays[metric + "_outcomes_mean"] = outcomes
                        arrays[metric + "_outcomes_std"] = np.full(3, .01)
                        # A decoy proves baselines are reloaded from summary.csv.
                        arrays[metric + "_baseline"] = np.array(99.)
                        extra[metric + "_baseline"] = baseline
                        rows.extend(dict(setup=setup["label"], metric=metric, baseline=baseline,
                                         quantile=q, beneficial_outcome=float(value))
                                    for q, value in zip(SETTINGS[kind]["quantiles"], outcomes))
                np.savez_compressed(folder / f"setup_{index}_mean.npz", **arrays)
                run_folder = folder / f"setup_{index}"
                run_folder.mkdir()
                for run in range(2):
                    # Only landscape axes and sweep metadata may be read from per-run files.
                    np.savez_compressed(run_folder / f"run_{run}.npz", alpha=alpha, beta=beta)
                    details = dict(checkpoint=manifest["setups"][index]["runs"][run]["path"],
                                   example_indices=list(range(8)), effective_config={"K": 2, "lr": .4},
                                   analysis_settings=SETTINGS[kind], original_config={})
                    if kind == "k_lr_sweep":
                        details["trained_settings"] = extra["trained_settings"][run]
                    (run_folder / f"run_{run}.json").write_text(json.dumps(details))
                summaries.append(dict(arrays, label=setup["label"], runs=2, **extra))
            if rows:
                write_table(folder / "summary.csv", rows)
            else:
                (folder / "summary.csv").write_text("setup,runs\nCE baseline,2\nEnergy shaped,2\n")
            (folder / "per_run.csv").write_text("saved per-run results\n")
            (folder / "example_indices.json").write_text(json.dumps(list(range(8))))
            (folder / f"{stem}.png").write_bytes(b"original plot must remain byte-for-byte unchanged")
            if kind == "k_lr_sweep":
                for metric in METRICS[1:]:
                    (folder / f"{stem}_{metric}.png").write_bytes(b"original metric plot")
            expected.append(summaries)
        return dict(kind=kind, output=output, config=config, manifest=manifest,
                    folders=folders, expected=expected, stem=stem, alpha=alpha, beta=beta)

    return make


@pytest.fixture(params=KINDS)
def saved_report(request, report_factory):
    return report_factory(request.param)


@pytest.fixture
def saved_figures(monkeypatch):
    recorded = []
    savefig = Figure.savefig

    def record(figure, path, *args, **kwargs):
        recorded.append(dict(path=Path(path), size=figure.get_size_inches().copy(),
                             dpi=kwargs.get("dpi")))
        return savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    return recorded


def plot_overrides(kind):
    if kind == "landscape":
        return dict(log_scale=False, separate_colorbars=True, include_cosine_sim=False)
    if kind in ("extrapolation", "update_svd", "step_orth_proj"):
        return dict(dash_start=1, show_n_runs=True)
    return {}


@pytest.mark.parametrize("kind", ["extrapolation", "step_orth_proj"])
def test_redraw_per_setup_dash_starts(report_factory, monkeypatch, kind):
    report = report_factory(kind, multiple_datasets=False)
    output = report["output"]
    saved = report["config"]
    saved[kind]["dash_start"] = 1
    saved[kind]["dash_start_by_setup"] = {"CE baseline": 3}
    (output / "config.json").write_text(json.dumps(saved))
    before = snapshot(output)
    calls = []
    draw = analyze.draw_plot

    def record(path, config, summaries, settings, *args):
        calls.append(copy.deepcopy(settings))
        draw(path, config, summaries, settings, *args)

    monkeypatch.setattr(analyze, "draw_plot", record)
    request = dict(output_dir=str(output))
    analyze.redraw_analysis(request)
    assert calls[-1]["dash_start_by_setup"] == {"CE baseline": 3}
    overrides = {"CE baseline": None, "Energy shaped": 3}
    request[kind] = {"dash_start_by_setup": overrides, "dash_end_by_setup": {"Energy shaped": None}}
    analyze.redraw_analysis(request)
    assert calls[-1]["dash_start_by_setup"] == overrides
    sidecar = json.loads((report["folders"][0] / f"{kind}_redraw_2.json").read_text())
    assert sidecar["plot_settings"]["dash_start_by_setup"] == overrides
    request[kind] = {"dash_start_by_setup": {"CE baseline": 1}, "dash_end_by_setup": {"CE baseline": 3}}
    if kind == "extrapolation":
        request[kind]["outer_loss_reference"] = 1.0
    analyze.redraw_analysis(request)
    assert calls[-1]["dash_end_by_setup"] == {"CE baseline": 3}
    if kind == "extrapolation":
        assert calls[-1]["outer_loss_reference"] == 1.0
    request[kind] = {"dash_start_by_setup": {}}
    analyze.redraw_analysis(request)
    assert calls[-1]["dash_start_by_setup"] == {}
    assert_untouched(output, before)
    request[kind] = {"dash_start_by_setup": {"Unknown": 1}}
    with pytest.raises(ValueError, match="existing setup labels"):
        analyze.redraw_analysis(request)
    assert len(calls) == 4


def assert_summary(actual, expected):
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("kind", KINDS)
def test_redraw_filters_saved_setups(report_factory, monkeypatch, kind):
    report = report_factory(kind, multiple_datasets=False)
    before = snapshot(report["output"])
    calls = []
    draw = analyze.draw_plot
    def record(path, config, summaries, *args):
        calls.append([summary["label"] for summary in summaries])
        draw(path, config, summaries, *args)
    monkeypatch.setattr(analyze, "draw_plot", record)
    analyze.redraw_analysis(dict(output_dir=str(report["output"]), setup_labels=["Energy shaped"]))
    assert calls == [["Energy shaped"]]
    sidecar = json.loads((report["folders"][0] / f"{report['stem']}_redraw_1.json").read_text())
    assert sidecar["setups"] == [dict(label="Energy shaped", runs=2)]
    assert_untouched(report["output"], before)


@pytest.mark.parametrize("labels", [None, [], "CE baseline", ["Unknown"], ["CE baseline"] * 2, [1]])
def test_redraw_rejects_invalid_setup_selection(report_factory, labels):
    report = report_factory("extrapolation", multiple_datasets=False)
    before = snapshot(report["output"])
    with pytest.raises(ValueError, match="setup_labels"):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]), setup_labels=labels))
    assert snapshot(report["output"]) == before


@pytest.mark.parametrize("include_cosine", [False, True])
def test_redraw_landscape_reports_saved_average(report_factory, monkeypatch, include_cosine):
    report = report_factory("landscape", multiple_datasets=False)
    folder, output = report["folders"][0], report["output"]
    write_table(folder / "alignment_summary.csv", [
        dict(setup="CE baseline", runs=2, mean_cosine=.125),
        dict(setup="Energy shaped", runs=2, mean_cosine=-.625),
    ])
    if not include_cosine:
        (folder / "alignment_summary.csv").unlink()
    before = snapshot(output)
    labels = []
    savefig = Figure.savefig
    def record(figure, *args, **kwargs):
        labels.extend(ax.get_xlabel() for ax in figure.axes if ax.get_xlabel())
        return savefig(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", record)
    analyze.redraw_analysis(dict(output_dir=str(output), landscape={
        "report_avg": True, "include_cosine_sim": include_cosine}))
    assert labels == (["Avg. cosine: 0.125", "Avg. cosine: -0.625"] if include_cosine else [])
    sidecar = json.loads((folder / "landscapes_redraw_1.json").read_text())
    assert sidecar["plot_settings"]["report_avg"] is True
    assert_untouched(output, before)


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "count", "nan", "range"])
def test_redraw_landscape_rejects_invalid_saved_average(report_factory, invalid):
    report = report_factory("landscape", multiple_datasets=False)
    rows = [dict(setup="CE baseline", runs=2, mean_cosine=.25),
            dict(setup="Energy shaped", runs=2, mean_cosine=.5)]
    if invalid == "missing":
        rows.pop()
    elif invalid == "duplicate":
        rows.append(rows[0])
    elif invalid == "count":
        rows[0]["runs"] = 3
    else:
        rows[0]["mean_cosine"] = float("nan") if invalid == "nan" else 1.5
    write_table(report["folders"][0] / "alignment_summary.csv", rows)
    before = snapshot(report["output"])
    with pytest.raises(ValueError, match="saved mean cosine"):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]), landscape={"report_avg": True}))
    assert snapshot(report["output"]) == before


def test_redraw_real_plots_twice_uses_only_saved_results(saved_report, monkeypatch, saved_figures):
    report = saved_report
    kind, output, stem = report["kind"], report["output"], report["stem"]
    before = snapshot(output)
    calls = []
    draw_plot = analyze.draw_plot

    def record(path, config, summaries, settings, alpha=None, beta=None):
        calls.append((copy.deepcopy(config), copy.deepcopy(summaries), copy.deepcopy(settings)))
        if kind == "landscape":
            np.testing.assert_array_equal(alpha, report["alpha"])
            np.testing.assert_array_equal(beta, report["beta"])
        return draw_plot(path, config, summaries, settings, alpha, beta)

    monkeypatch.setattr(analyze, "draw_plot", record)
    overrides = plot_overrides(kind)
    # These stale inputs would fail ordinary analysis validation or trigger loading.
    request = dict(output_dir=str(output), dpi=40, figsize=[11, 9],
                   setups=[{"label": "obsolete", "runs": []}], dataset="missing/new_dataset",
                   datasets=[{"invalid": True}], examples=-1, batch_size=-1,
                   inner_lr=-99, device="cuda", **{kind: dict(overrides, steps=-9,
                                                            basis_steps=[90, 91],
                                                            inner_lrs=[-1], grid_size=101)})
    for iteration in (1, 2):
        if iteration == 2:
            request["kind"] = kind
        assert analyze.redraw_analysis(request) == output
        for folder in report["folders"]:
            path = folder / f"{stem}_redraw_{iteration}.png"
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            metadata = json.loads(path.with_suffix(".json").read_text())
            assert metadata["kind"] == kind
            assert metadata["source_config"] == str(output / "config.json")
            assert metadata["source_manifest"] == str(output / "manifest.json")
            assert metadata["dpi"] == 40
            assert metadata["figsize"] == [11, 9]
            assert metadata["plot_settings"] == dict(SETTINGS[kind], **overrides)
            assert metadata["setups"] == [dict(label=s["label"], runs=2)
                                          for s in report["config"]["setups"]]
            if kind == "k_lr_sweep":
                for metric in METRICS[1:]:
                    metric_path = folder / f"{stem}_redraw_{iteration}_{metric}.png"
                    assert metric_path.read_bytes().startswith(b"\x89PNG")
        assert_untouched(output, before)
        before = snapshot(output)
    assert len(calls) == 4
    for position, (effective, summaries, settings) in enumerate(calls):
        assert effective["setups"] == report["config"]["setups"]
        assert effective["datasets"] == report["config"]["datasets"]
        for key in ("examples", "batch_size", "device"):
            assert effective[key] == report["config"][key]
        assert "inner_lr" not in effective
        assert settings == dict(SETTINGS[kind], **overrides)
        for actual, expected in zip(summaries, report["expected"][position % 2]):
            # Sweep axis lists are restored as Python lists for plotting.
            assert_summary(actual, expected)
    assert len(saved_figures) == (16 if kind == "k_lr_sweep" else 4)
    for figure in saved_figures:
        np.testing.assert_allclose(figure["size"], [11, 9])
        assert figure["dpi"] == 40
    assert not plt.get_fignums()


def test_draw_plot_forwards_top_level_figsize_for_every_figure(saved_report, saved_figures):
    report = saved_report
    config = dict(report["config"], dpi=30, figsize=[12, 10])
    settings = dict(SETTINGS[report["kind"]], **plot_overrides(report["kind"]))
    summaries = copy.deepcopy(report["expected"][0])
    if report["kind"] == "k_lr_sweep":
        for summary in summaries:
            for key in ("steps", "inner_lrs"):
                summary[key] = summary[key].tolist()
    analyze.draw_plot(report["output"] / "figsize.png", config, summaries, settings,
                      report["alpha"], report["beta"])
    assert len(saved_figures) == (4 if report["kind"] == "k_lr_sweep" else 1)
    for figure in saved_figures:
        np.testing.assert_allclose(figure["size"], [12, 10])
        assert figure["dpi"] == 30


def test_redraw_global_index_uses_every_dataset_and_metric_suffix(saved_report, monkeypatch):
    report = saved_report
    stem = report["stem"]
    (report["folders"][0] / f"{stem}_redraw_3.png").write_bytes(b"earlier redraw")
    (report["folders"][1] / f"{stem}_redraw_7.json").write_text("earlier metadata")
    (report["folders"][1] / f"{stem}_redraw_11_outer_loss.png").write_bytes(b"metric only")
    before = snapshot(report["output"])
    paths = []

    def draw(path, *args, **kwargs):
        assert not path.exists()
        paths.append(path)

    monkeypatch.setattr(analyze, "draw_plot", draw)
    analyze.redraw_analysis({"output_dir": str(report["output"])})
    assert paths == [folder / f"{stem}_redraw_12.png" for folder in report["folders"]]
    for path in paths:
        assert path.with_suffix(".json").is_file()
    assert_untouched(report["output"], before)


def test_redraw_single_saved_string_dataset(report_factory, saved_figures):
    report = report_factory("extrapolation", multiple_datasets=False)
    analyze.redraw_analysis({"output_dir": str(report["output"])})
    assert len(saved_figures) == 1
    np.testing.assert_allclose(saved_figures[0]["size"], [10, 8])
    assert saved_figures[0]["dpi"] == 35
    assert (report["folders"][0] / "extrapolation_redraw_1.json").is_file()


def test_redraw_cli_never_queries_or_initializes_cuda(report_factory):
    report = report_factory("extrapolation", multiple_datasets=False)
    request = report["output"] / "redraw_request.json"
    request.write_text(json.dumps(dict(output_dir=str(report["output"]))))
    script = """
import runpy
import sys
from pathlib import Path
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("Redraw must not query or initialize CUDA")

for name in ("device_count", "is_available", "init", "_lazy_init", "set_device", "current_device"):
    setattr(torch.cuda, name, forbidden)

from analysis_tools import analyze
analyze.REPORTS = Path(sys.argv[1]).parent.parent
sys.argv = ["analysis_tools", sys.argv[1], "--redraw"]
runpy.run_module("analysis_tools", run_name="__main__")
assert not torch.cuda.is_initialized()
assert "run_gradmemgpt_on_mqar" not in sys.modules
assert "grad_memgpt" not in sys.modules
assert "transformers" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, str(request)],
                            cwd=library.REPO, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA DEVICE COUNT" not in result.stdout + result.stderr
    assert (report["folders"][0] / "extrapolation_redraw_1.png").is_file()


@pytest.mark.parametrize("dash_start", [None, 1])
def test_extrapolation_bands_clip_display_only_and_show_runs(tmp_path, monkeypatch, dash_start):
    run_losses = np.array([[.001, .01, .1], [.002, .02, .2], [.5, .4, .3]])
    setup = dict(label="Test", runs=3)
    for metric in ("outer_loss", "token_accuracy", "exact_match"):
        values = run_losses if metric == "outer_loss" else np.array([[0., .9, 1.], [.01, 1., .9], [.1, .9, 1.]])
        setup[metric + "_runs"] = values
        setup[metric + "_mean"] = values.mean(0)
        setup[metric + "_std"] = values.std(0, ddof=1)
    before = copy.deepcopy(setup)
    inspected = []
    def inspect(figure, *args, **kwargs):
        ax = figure.axes[1]
        assert ax.get_yscale() == "log"
        assert ax.get_ylim()[0] > 0
        assert not ax.containers  # No error-bar containers in band mode.
        assert ax.get_legend_handles_labels()[1] == ["Test"]
        # A band and downward truncation markers are present.
        assert len(ax.collections) == 2
        markers = ax.collections[0]
        clipped = setup["outer_loss_mean"] - setup["outer_loss_std"] <= 0
        assert len(markers.get_offsets()) == clipped.sum()
        np.testing.assert_allclose(markers.get_offsets()[:, 1], ax.get_ylim()[0])
        paths = ax.collections[1].get_paths()
        assert min(path.vertices[:, 1].min() for path in paths) >= ax.get_ylim()[0]
        assert len(ax.lines) == (9 if dash_start is not None else 4)
        for percent_ax in (figure.axes[0], figure.axes[2]):
            band = percent_ax.collections[0].get_paths()[0].vertices[:, 1]
            assert band.min() >= 0 and band.max() <= 100
        assert figure._supxlabel is None
        inspected.append(figure)
    monkeypatch.setattr(Figure, "savefig", inspect)
    plots.extrapolation(tmp_path / "bands.png", [setup], [0, 1, 2],
                        uncertainty_style="band", show_runs=True, dash_start=dash_start)
    assert len(inspected) == 1
    assert_summary(setup, before)


def test_redraw_bands_load_saved_individual_runs(report_factory, monkeypatch):
    report = report_factory("extrapolation", multiple_datasets=False)
    for index, summary in enumerate(report["expected"][0]):
        for run_index in range(summary["runs"]):
            np.savez_compressed(report["folders"][0] / f"setup_{index}" / f"run_{run_index}.npz",
                                **{metric: summary[metric + "_mean"] for metric in METRICS[1:]})
    received = []
    def record(path, config, summaries, settings, *args):
        assert settings["uncertainty_style"] == "band"
        assert settings["show_runs"] is True
        for summary in summaries:
            for metric in METRICS[1:]:
                assert summary[metric + "_runs"].shape == (2, 3)
        received.append(summaries)
    monkeypatch.setattr(analyze, "draw_plot", record)
    analyze.redraw_analysis(dict(output_dir=str(report["output"]),
                                extrapolation=dict(uncertainty_style="band", show_runs=True)))
    assert len(received) == 1


@pytest.mark.parametrize("options", [dict(uncertainty_style="sem"), dict(show_runs="true")])
def test_extrapolation_uncertainty_options_validation(options):
    with pytest.raises(ValueError, match="uncertainty_style|show_runs"):
        analyze.validate_config(dict(kind="extrapolation", extrapolation=options,
                                     setups=[dict(label="test", runs=["missing"])]))


def test_redraw_global_fonts_style_every_plot_and_restore_defaults(saved_report, monkeypatch):
    report = saved_report
    sizes = dict(subtitle_font_size=19, tick_font_size=13, axis_label_font_size=17, legend_font_size=11)
    previous = {key: plt.rcParams[key] for parameters in analyze.FONT_SIZES.values() for key in parameters}
    inspected = []
    def inspect(figure, *args, **kwargs):
        for ax in figure.axes:
            if ax.get_title():
                assert ax.title.get_fontsize() == sizes["subtitle_font_size"]
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                assert text.get_fontsize() == sizes["tick_font_size"]
            for text in (ax.xaxis.label, ax.yaxis.label):
                if text.get_text():
                    assert text.get_fontsize() == sizes["axis_label_font_size"]
            legend = ax.get_legend()
            if legend is not None:
                assert all(text.get_fontsize() == sizes["legend_font_size"] for text in legend.get_texts())
        inspected.append(figure)
    monkeypatch.setattr(Figure, "savefig", inspect)
    analyze.redraw_analysis(dict(output_dir=str(report["output"]), **sizes))
    assert len(inspected) == (8 if report["kind"] == "k_lr_sweep" else 2)
    assert {key: plt.rcParams[key] for key in previous} == previous
    metadata = json.loads((report["folders"][0] / f"{report['stem']}_redraw_1.json").read_text())
    assert metadata["font_sizes"] == sizes


@pytest.mark.parametrize("show_n_runs", [False, True])
def test_show_n_runs_for_every_plot_type(saved_report, monkeypatch, show_n_runs):
    report = saved_report
    inspected = []
    def inspect(figure, *args, **kwargs):
        kind = report["kind"]
        labels = [setup["label"] for setup in report["config"]["setups"]]
        separator = "\n" if kind in ("landscape", "k_lr_sweep", "memory_search") else " "
        expected = [label + (separator + "(2 runs)" if show_n_runs else "") for label in labels]
        if kind in ("landscape", "k_lr_sweep"):
            assert [ax.get_title() for ax in figure.axes if ax.get_title()] == expected
        elif kind == "memory_search":
            for ax in figure.axes:
                assert [text.get_text() for text in ax.get_xticklabels()] == expected
        else:
            legend = figure.axes[0].get_legend()
            assert [text.get_text() for text in legend.get_texts()] == expected
        inspected.append(figure)
    monkeypatch.setattr(Figure, "savefig", inspect)
    analyze.redraw_analysis(dict(output_dir=str(report["output"]),
                                **{report["kind"]: {"show_n_runs": show_n_runs}}))
    assert len(inspected) == (8 if report["kind"] == "k_lr_sweep" else 2)
    metadata = json.loads((report["folders"][0] / f"{report['stem']}_redraw_1.json").read_text())
    assert metadata["plot_settings"]["show_n_runs"] is show_n_runs


@pytest.mark.parametrize("kind,value", [
    ("landscape", "false"), ("extrapolation", "false"), ("k_lr_sweep", "false"),
    ("update_svd", "false"), ("step_orth_proj", "false"), ("memory_search", "false"),
    ("extrapolation", 1), ("extrapolation", None),
])
def test_show_n_runs_requires_bool_for_every_plot_type(kind, value):
    config = dict(kind=kind, setups=[dict(label="test", runs=["missing"])],
                  **{kind: dict(SETTINGS[kind], show_n_runs=value)})
    with pytest.raises(ValueError, match="show_n_runs"):
        analyze.validate_config(config)


@pytest.mark.parametrize("key,value", [
    ("subtitle_font_size", True), ("subtitle_font_size", "large"),
    ("subtitle_font_size", 0), ("subtitle_font_size", -1),
    ("subtitle_font_size", float("nan")), ("subtitle_font_size", float("inf")),
    ("tick_font_size", 0), ("axis_label_font_size", 0), ("legend_font_size", 0),
])
def test_global_font_size_validation(key, value):
    config = dict(kind="extrapolation", setups=[dict(label="test", runs=["missing"])], **{key: value})
    with pytest.raises(ValueError, match=key):
        analyze.validate_config(config)


def test_redraw_font_options_inherit_and_reset(report_factory, monkeypatch):
    report = report_factory("extrapolation", multiple_datasets=False)
    report["config"].update(subtitle_font_size=20, tick_font_size=18, axis_label_font_size=16, legend_font_size=14)
    (report["output"] / "config.json").write_text(json.dumps(report["config"]))
    received = []
    monkeypatch.setattr(analyze, "draw_plot", lambda path, config, *args: received.append(config))
    analyze.redraw_analysis(dict(output_dir=str(report["output"]), tick_font_size=12, legend_font_size=None))
    assert received[0]["subtitle_font_size"] == 20
    assert received[0]["tick_font_size"] == 12
    assert received[0]["axis_label_font_size"] == 16
    assert received[0]["legend_font_size"] is None


def test_redraw_kind_mismatch_never_writes(saved_report):
    report = saved_report
    before = snapshot(report["output"])
    other = "memory_search" if report["kind"] != "memory_search" else "landscape"
    with pytest.raises(ValueError, match="kind.*incompatible"):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]), kind=other))
    assert snapshot(report["output"]) == before


def test_redraw_rejects_plot_options_in_wrong_section(report_factory):
    report = report_factory("landscape", multiple_datasets=False)
    before = snapshot(report["output"])
    with pytest.raises(ValueError, match="Saved report is 'landscape'.*under 'extrapolation'"):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]),
                                    extrapolation={"include_cosine_sim": False}))
    assert snapshot(report["output"]) == before


REQUIRED_KEY = {
    "landscape": "inner_loss", "extrapolation": "outer_loss_mean",
    "k_lr_sweep": "token_accuracy_std", "update_svd": "singular_values_mean",
    "step_orth_proj": "projection_ratio_mean", "memory_search": "outer_loss_outcomes",
}


@pytest.mark.parametrize("damage", ["absent", "missing_key", "wrong_shape", "nonfinite"])
def test_redraw_rejects_invalid_aggregates_before_any_plot(saved_report, monkeypatch, damage):
    report = saved_report
    # Break the last dataset/setup to check that earlier datasets are not written first.
    path = report["folders"][-1] / "setup_1_mean.npz"
    key = REQUIRED_KEY[report["kind"]]
    if damage == "absent":
        path.unlink()
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        if damage == "missing_key":
            del arrays[key]
        elif damage == "wrong_shape":
            arrays[key] = np.zeros((1, 1, 1))
        else:
            arrays[key] = np.full(arrays[key].shape, np.nan)
        np.savez_compressed(path, **arrays)
    before = snapshot(report["output"])

    def forbidden(*args, **kwargs):
        pytest.fail("All saved datasets must validate before any plotting")

    monkeypatch.setattr(analyze, "draw_plot", forbidden)
    error = FileNotFoundError if damage == "absent" else ValueError
    with pytest.raises(error):
        analyze.redraw_analysis({"output_dir": str(report["output"])})
    assert snapshot(report["output"]) == before


@pytest.mark.parametrize("size", ["10,8", 5, [], [8], [8, 6, 4], [0, 6], [-1, 6],
                                 [True, 6], ["8", 6], [float("nan"), 6],
                                 [8, float("inf")], {"width": 8, "height": 6}])
def test_redraw_rejects_invalid_figsize(report_factory, size):
    report = report_factory("extrapolation", multiple_datasets=False)
    before = snapshot(report["output"])
    with pytest.raises(ValueError, match="figsize"):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]), figsize=size))
    assert snapshot(report["output"]) == before


@pytest.mark.parametrize("size", [None, [8, 6], [8.5, 6.25]])
def test_validate_config_accepts_optional_figsize(report_factory, size):
    report = report_factory("extrapolation", multiple_datasets=False)
    analyze.validate_config(dict(report["config"], figsize=size))


@pytest.mark.parametrize("kind,option,value", [
    ("landscape", "log_scale", "false"),
    ("landscape", "separate_colorbars", 1),
    ("landscape", "include_cosine_sim", None),
    ("extrapolation", "show_n_runs", "true"),
    ("extrapolation", "dash_start", True),
    ("update_svd", "show_n_runs", 1),
    ("update_svd", "dash_start", "1"),
    ("step_orth_proj", "show_n_runs", None),
    ("step_orth_proj", "dash_start", 1.5),
])
def test_redraw_rejects_unsupported_plot_option_types(report_factory, kind, option, value):
    report = report_factory(kind, multiple_datasets=False)
    before = snapshot(report["output"])
    with pytest.raises(ValueError, match=option):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]), **{kind: {option: value}}))
    assert snapshot(report["output"]) == before


@pytest.mark.parametrize("dpi", [0, -1, True, "100", None, float("nan"), float("inf")])
def test_redraw_rejects_invalid_dpi(report_factory, dpi):
    report = report_factory("extrapolation", multiple_datasets=False)
    before = snapshot(report["output"])
    with pytest.raises(ValueError, match="dpi"):
        analyze.redraw_analysis(dict(output_dir=str(report["output"]), dpi=dpi))
    assert snapshot(report["output"]) == before


def test_redraw_requires_report_under_reports(tmp_path):
    with pytest.raises(ValueError, match="under reports"):
        analyze.redraw_analysis({"output_dir": str(tmp_path / "elsewhere")})


def test_cli_redraw_dispatch_reads_json_without_running_analysis(tmp_path, monkeypatch):
    config = dict(output_dir="reports/saved", figsize=[9, 7],
                  landscape={"include_cosine_sim": False})
    path = tmp_path / "redraw.json"
    path.write_text(json.dumps(config))
    received = []
    monkeypatch.setattr(analyze, "redraw_analysis", lambda value: received.append(value))
    monkeypatch.setattr(sys, "argv", ["analysis_tools", str(path), "--redraw"])
    analyze.main()
    assert received == [config]
