"""Independent setup dash boundaries for shared analysis plots."""

import numpy as np
import pytest

from analysis_tools import analyze, plots


@pytest.mark.parametrize("kind", ["extrapolation", "step_orth_proj"])
@pytest.mark.parametrize("style", ["errorbar", "band"])
@pytest.mark.parametrize("shared", [None, 2])
@pytest.mark.parametrize("bounded", [False, True])
def test_per_setup_dash_boundaries(tmp_path, monkeypatch, kind, style, shared, bounded):
    steps = [0, 1, 2, 4] if kind == "extrapolation" else [1, 2, 3, 4]
    setups = []
    for label in ["Early", "Late", "Solid", "Inherited"]:
        setup = dict(label=label, runs=2)
        for metric in ["token_accuracy", "outer_loss", "exact_match", "projection_ratio", "step_norm", "cosine_sim"]:
            setup[metric + "_mean"] = np.linspace(.3, .6, 4)
            setup[metric + "_std"] = np.full(4, .02)
            setup[metric + "_runs"] = np.array([setup[metric + "_mean"] - .02,
                                               setup[metric + "_mean"] + .02])
        setups.append(setup)
    overrides = {"Early": steps[0], "Late": 4, "Solid": None}
    figures = []
    close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: figures.append(figure))
    settings = dict(dash_start=shared, dash_start_by_setup=overrides, show_n_runs=False,
                    show_runs=True, uncertainty_style=style)
    if bounded:
        settings["dash_end_by_setup"] = {"Early": 2}
    if kind == "extrapolation":
        settings["steps"] = steps
        if bounded:
            settings["outer_loss_reference"] = 1.0
    else:
        settings.update(steps=4, proj_steps=[1, 2], add_step_norm=True, include_cosine_sim=True)
        for setup in setups:
            for suffix in ["_mean", "_std"]:
                setup["projection_ratio" + suffix] = np.tile(setup["projection_ratio" + suffix], (2, 1))
            setup["projection_ratio_runs"] = np.stack([setup["projection_ratio_runs"]] * 2, axis=1)
    config = dict(kind=kind, dpi=30, setups=[dict(label=setup["label"], runs=["a", "b"]) for setup in setups],
                  **{kind: settings})
    analyze.validate_config(config)
    analyze.draw_plot(tmp_path / "plot.png", config, setups, settings)
    figure = figures[-1]
    try:
        assert figure._supxlabel is None
        assert len(figure.axes) == (3 if kind == "extrapolation" else 4)
        for ax in figure.axes:
            curves = ([container.lines[0] for container in ax.containers] if style == "errorbar"
                      else [line for line in ax.lines if line.get_label() in overrides or line.get_label() == "Inherited"])
            assert len(curves) == 4
            for setup, curve in zip(setups, curves):
                start = overrides.get(setup["label"], shared)
                split = steps.index(start) if start is not None else 3
                stop = steps.index(2) + 1 if bounded and setup["label"] == "Early" else 4
                x = np.arange(4) if kind == "extrapolation" else np.array(steps)
                np.testing.assert_array_equal(curve.get_xdata(), x[:split + 1])
                dashed = [line for line in ax.lines if line.get_color() == curve.get_color()
                          and line.get_linestyle() == "--"]
                assert len(dashed) == (3 if start is not None else 0)
                for line in dashed:
                    np.testing.assert_array_equal(line.get_xdata(), x[split:stop])
                overlays = [line for line in ax.lines if line.get_color() == curve.get_color()
                            and line.get_alpha() == .3 and line.get_linestyle() == "-"]
                assert len(overlays) == (4 if stop < 4 else 2)
                for line in overlays[:2] if stop == 4 else overlays[::2]:
                    np.testing.assert_array_equal(line.get_xdata(), x[:split + 1])
                if stop < 4:
                    for line in overlays[1::2]:
                        np.testing.assert_array_equal(line.get_xdata(), x[stop - 1:])
                    continuations = [line for line in ax.lines if line.get_color() == curve.get_color()
                                     and line.get_linestyle() == "-" and line.get_marker() == "o"
                                     and line is not curve]
                    assert len(continuations) == 1
                    np.testing.assert_array_equal(continuations[0].get_xdata(), x[stop - 1:])
            references = [line for line in ax.lines if line.get_color() == "0.4"]
            assert len(references) == int(shared is not None)
            assert ax.get_legend_handles_labels()[1] == [setup["label"] for setup in setups]
        references = [(index, line) for index, ax in enumerate(figure.axes) for line in ax.lines
                      if line.get_color() == "0.25"]
        assert len(references) == int(bounded and kind == "extrapolation")
        if references:
            assert references[0][0] == 1
            assert references[0][1].get_linestyle() == "--"
            np.testing.assert_array_equal(references[0][1].get_ydata(), [1., 1.])
    finally:
        close(figure)


@pytest.mark.parametrize("kind", ["extrapolation", "step_orth_proj"])
@pytest.mark.parametrize("overrides", [None, [], 2, {"Unknown": 2}, {"A": True}, {"A": 2.0},
                                        {"A": -1}, {"A": 5}, {"A": "2"}])
def test_invalid_dash_start_maps(kind, overrides):
    settings = dict(steps=[0, 1, 2, 4] if kind == "extrapolation" else 4,
                    dash_start_by_setup=overrides)
    config = dict(kind=kind, setups=[dict(label="A", runs=["checkpoint"])], **{kind: settings})
    with pytest.raises(ValueError, match="dash_start_by_setup"):
        analyze.validate_config(config)


def test_per_setup_dash_requires_increasing_extrapolation_steps():
    config = dict(kind="extrapolation", setups=[dict(label="A", runs=["checkpoint"])],
                  extrapolation=dict(steps=[2, 1], dash_start_by_setup={"A": 1}))
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze.validate_config(config)


@pytest.mark.parametrize("kind", ["extrapolation", "step_orth_proj"])
@pytest.mark.parametrize("ends", [None, [], {"Unknown": 3}, {"A": True}, {"A": 2.5},
                                  {"A": 1}, {"A": 2}, {"A": 5}])
def test_invalid_dash_end_maps(kind, ends):
    settings = dict(steps=[0, 1, 2, 4] if kind == "extrapolation" else 4,
                    dash_start=2, dash_end_by_setup=ends)
    config = dict(kind=kind, setups=[dict(label="A", runs=["checkpoint"])], **{kind: settings})
    with pytest.raises(ValueError, match="dash_end_by_setup"):
        analyze.validate_config(config)


def test_dash_end_requires_enabled_start():
    config = dict(kind="extrapolation", setups=[dict(label="A", runs=["checkpoint"])],
                  extrapolation=dict(steps=[0, 1, 2, 4], dash_end_by_setup={"A": 4}))
    with pytest.raises(ValueError, match="dash_end_by_setup"):
        analyze.validate_config(config)


@pytest.mark.parametrize("reference", [True, 0, -1, "1", float("nan"), float("inf")])
def test_outer_loss_reference_validation(reference):
    config = dict(kind="extrapolation", setups=[dict(label="A", runs=["checkpoint"])],
                  extrapolation={"outer_loss_reference": reference})
    with pytest.raises(ValueError, match="outer_loss_reference"):
        analyze.validate_config(config)
