"""Shared plotting for arbitrary setup counts; no experiment-specific code."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


def loss_scale(arrays, log=True):
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    low, high = float(values.min()), float(values.max())
    if low == high:
        high = low + max(abs(low) * 1e-6, 1e-6)
    if not log or low <= 0:
        return Normalize(low, high), MaxNLocator(nbins=4), None
    exponents = np.arange(np.floor(np.log10(low)), np.ceil(np.log10(high)) + 1)
    ticks = np.sort(np.concatenate((10.0 ** exponents, 3 * 10.0 ** exponents)))
    ticks = ticks[(ticks >= low) & (ticks <= high)]
    if len(ticks) < 2:
        ticks = np.geomspace(low, high, 2)
    def label(value, _):
        exponent = int(np.floor(np.log10(value)))
        return rf"${value / 10**exponent:.2g}\times10^{{{exponent}}}$"
    return LogNorm(low, high), ticks, FuncFormatter(label)


def landscapes(path, setups, alpha, beta, basis_steps, dpi=220, log=True, font_size=15,
               separate_colorbars=False, include_cosine_sim=True, figsize=None, show_n_runs=True, report_avg=False):
    count = len(setups)
    rows = 3 if include_cosine_sim else 2
    with plt.rc_context({"font.size": font_size}):
        figure = plt.figure(figsize=figsize or (7 * count if separate_colorbars else 5.5 * count + 1.5, 5 * rows),
                            layout="constrained")
        widths = [1, .055] * count if separate_colorbars else [1] * count + [.055]
        grid = figure.add_gridspec(rows, len(widths), width_ratios=widths)
        axes = [[figure.add_subplot(grid[row, 2 * col if separate_colorbars else col])
                 for col in range(count)] for row in range(rows)]
        labeled_colorbar_axes = []
        for row, (key, title, cmap) in enumerate((
            ("inner_loss", "Inner WRITE objective", "viridis"),
            ("outer_loss", "Outer READ target loss", "magma"),
            ("alignment", "Mean per-run gradient cosine", "coolwarm"),
        )[:rows]):
            for col, setup in enumerate(setups):
                if row < 2:
                    arrays = [setup[key]] if separate_colorbars else [item[key] for item in setups]
                    norm, _, formatter = loss_scale(arrays, log)
                    levels = (np.geomspace(norm.vmin, norm.vmax, 128) if isinstance(norm, LogNorm)
                              else np.linspace(norm.vmin, norm.vmax, 128))
                else:
                    norm, formatter = Normalize(-1, 1), None
                    levels = np.linspace(-1, 1, 101)
                ax = axes[row][col]
                image = ax.contourf(alpha, beta, setup[key], levels=levels, norm=norm, cmap=cmap)
                if row < 2 and np.nanmin(setup[key]) < np.nanmax(setup[key]):
                    contour_levels = (np.geomspace(norm.vmin, norm.vmax, 24) if isinstance(norm, LogNorm)
                                      else np.linspace(norm.vmin, norm.vmax, 24))
                    ax.contour(alpha, beta, setup[key], levels=contour_levels,
                               norm=norm, colors="white", linewidths=.35, alpha=.45)
                if row == 2 and np.nanmin(setup[key]) < 0 < np.nanmax(setup[key]):
                    ax.contour(alpha, beta, setup[key], levels=[0], colors="black", linewidths=.7)
                trajectory = setup["trajectory"]
                ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", color="#00e5ff", lw=2)
                first_state = setup.get("basis_steps", basis_steps)[0] - 1
                for step in range(first_state, len(trajectory)):
                    x, y = trajectory[step]
                    if alpha[0] <= x <= alpha[-1] and beta[0] <= y <= beta[-1]:
                        ax.annotate(f"M{step}", (x, y), xytext=(0, 7), textcoords="offset points",
                                    ha="center", va="bottom", color="white")
                ax.set_xlim(alpha[[0, -1]])
                ax.set_ylim(beta[[0, -1]])
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.tick_params(axis="x", bottom=row == rows - 1, labelbottom=row == rows - 1)
                ax.tick_params(axis="y", left=col == 0, labelleft=col == 0)
                if row == 2 and report_avg:
                    ax.set_xlabel(f"Avg. cosine: {setup['mean_cosine']:.3f}", fontsize="small")
                if row == 0:
                    title_label = setup["label"]
                    if show_n_runs:
                        title_label += f"\n({setup['runs']} run{'s' if setup['runs'] != 1 else ''})"
                    ax.set_title(title_label)
                if separate_colorbars or col == count - 1:
                    colorbar_col = 2 * col + 1 if separate_colorbars else count
                    midpoint = (np.exp((np.log(norm.vmin) + np.log(norm.vmax)) / 2)
                                if isinstance(norm, LogNorm) else norm.vmin / 2 + norm.vmax / 2)
                    colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[row, colorbar_col]),
                                              ticks=[norm.vmin, midpoint, norm.vmax])
                    colorbar.minorticks_off()
                    if formatter:
                        colorbar.ax.yaxis.set_major_formatter(formatter)
                    if col == count - 1:
                        colorbar.set_label(title)
                        labeled_colorbar_axes.append(colorbar.ax)
        figure.align_ylabels(labeled_colorbar_axes)
        figure.savefig(path, dpi=dpi)
        plt.close(figure)


def extrapolation(path, setups, steps, dpi=220, dash_start=None, show_n_runs=False, figsize=None,
                  uncertainty_style="errorbar", show_runs=False, dash_start_by_setup=None,
                  dash_end_by_setup=None, outer_loss_reference=None):
    x = np.arange(len(steps))
    floor = None
    if uncertainty_style == "band":
        candidates = [np.asarray(setup[key]).ravel() for setup in setups
                      for key in ("outer_loss_mean", "outer_loss_runs") if key in setup]
        candidates.extend(setup["outer_loss_mean"] - setup["outer_loss_std"] for setup in setups)
        positive = np.concatenate(candidates)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if not positive.size:
            raise ValueError("Log loss bands require at least one positive loss value")
        floor = positive.min() / 2
    figure, axes = plt.subplots(1, 3, figsize=figsize or (19.5, 5), layout="constrained")
    axes[1].set_yscale("log")
    for setup in setups:
        start = (dash_start_by_setup or {}).get(setup["label"], dash_start)
        split = steps.index(start) if start is not None else None
        end = (dash_end_by_setup or {}).get(setup["label"])
        stop = steps.index(end) + 1 if end is not None else len(steps)
        legend_label = setup["label"]
        if show_n_runs:
            legend_label += f" ({setup['runs']} run{'s' if setup['runs'] != 1 else ''})"
        for ax, key, label in zip(axes, ("token_accuracy", "outer_loss", "exact_match"),
                                  ("Token accuracy (%)", "Outer READ target loss", "Exact match (%)")):
            multiplier = 1 if key == "outer_loss" else 100
            values = setup[key + "_mean"] * multiplier
            std = setup[key + "_std"] * multiplier
            if uncertainty_style == "band":
                line, = ax.plot(x, values, "o-", linewidth=2.3, markersize=4, label=legend_label, zorder=3)
                lower, upper = values - std, values + std
                if key == "outer_loss":
                    clipped = lower <= 0
                    lower = np.maximum(lower, floor)
                    ax.scatter(x[clipped], np.full(clipped.sum(), floor), marker="v", s=25,
                               color=line.get_color(), clip_on=False, zorder=4, label="_nolegend_")
                else:
                    lower, upper = np.clip(lower, 0, 100), np.clip(upper, 0, 100)
                ax.fill_between(x, lower, upper, color=line.get_color(), alpha=.12, linewidth=0, zorder=0)
            else:
                curve = ax.errorbar(x, values, yerr=std, marker="o", capsize=3, label=legend_label)
                line = curve.lines[0]
            if show_runs:
                for run in setup[key + "_runs"] * multiplier:
                    ax.plot(x if split is None else x[:split + 1], run if split is None else run[:split + 1],
                            color=line.get_color(), linewidth=.9, alpha=.3, zorder=1, label="_nolegend_")
                    if split is not None:
                        ax.plot(x[split:stop], run[split:stop], "--", color=line.get_color(),
                                 linewidth=.9, alpha=.3, zorder=1, label="_nolegend_")
                        if end is not None:
                            ax.plot(x[stop - 1:], run[stop - 1:], color=line.get_color(),
                                    linewidth=.9, alpha=.3, zorder=1, label="_nolegend_")
            if split is not None:
                line.set_data(range(split + 1), values[:split + 1])
                ax.plot(x[split:stop], values[split:stop], "o--",
                        color=line.get_color(), linewidth=line.get_linewidth(), zorder=line.get_zorder(),
                        label="_nolegend_")
                if end is not None:
                    ax.plot(x[stop - 1:], values[stop - 1:], "o-", color=line.get_color(),
                            linewidth=line.get_linewidth(), zorder=line.get_zorder(), label="_nolegend_")
            ax.set_xticks(range(len(steps)), steps)
            ax.set_xlabel("WRITE steps (K)")
            ax.set_ylabel(label)
            ax.grid(alpha=.3)
    if uncertainty_style == "band":
        axes[0].set_ylim(-1, 101)
        axes[2].set_ylim(-1, 101)
        axes[1].set_ylim(bottom=floor)
    if dash_start is not None:
        for ax in axes:
            ax.axvline(steps.index(dash_start), color="0.4", linestyle="--", linewidth=1, label="_nolegend_")
    if outer_loss_reference is not None:
        axes[1].axhline(outer_loss_reference, color="0.25", linestyle="--", linewidth=1.5,
                        zorder=2, label="_nolegend_")
    axes[0].legend(frameon=False)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def step_orth_proj(path, setups, steps, dpi=220, proj_steps=2, dash_start=None, show_n_runs=True, figsize=None,
                   uncertainty_style="errorbar", show_runs=False, zero_step_policy="error", add_step_norm=False,
                   include_cosine_sim=False, dash_start_by_setup=None, dash_end_by_setup=None):
    spans = [] if proj_steps is None else proj_steps if isinstance(proj_steps, list) else [proj_steps]
    panels = [None, *spans] if add_step_norm else spans
    if include_cosine_sim:
        panels = [*panels, "cosine"]
    if not panels:
        raise ValueError("proj_steps=null requires add_step_norm or include_cosine_sim")
    norm_floor = None
    if add_step_norm:
        candidates = [np.asarray(setup[key]).ravel() for setup in setups
                      for key in ("step_norm_mean", "step_norm_runs") if key in setup and (key != "step_norm_runs" or show_runs)]
        candidates.extend(setup["step_norm_mean"] - setup["step_norm_std"] for setup in setups)
        positive = np.concatenate(candidates)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        norm_floor = positive.min() / 2 if positive.size else 1e-12
    figure, axes = plt.subplots(1, len(panels), figsize=figsize or (9 if len(panels) == 1 else 7 * len(panels), 5),
                               sharex=True, squeeze=False, layout="constrained")
    indices = np.arange(1, steps + 1)
    projection_ax = axes[0, int(add_step_norm)] if spans else None
    for panel, (ax, span) in enumerate(zip(axes[0], panels)):
        is_norm = span is None
        is_cosine = span == "cosine"
        projection_index = panel - int(add_step_norm)
        if projection_index > 0 and not is_norm and not is_cosine:
            ax.sharey(projection_ax)
            ax.tick_params(labelleft=False)
        if is_norm:
            ax.set_yscale("log", nonpositive="mask")
        for setup in setups:
            start = (dash_start_by_setup or {}).get(setup["label"], dash_start)
            end = (dash_end_by_setup or {}).get(setup["label"], steps)
            stop = end if end is not None else steps
            label = setup["label"]
            if show_n_runs:
                label += f" ({setup['runs']} run{'s' if setup['runs'] != 1 else ''})"
            metric = "step_norm" if is_norm else "cosine_sim" if is_cosine else "projection_ratio"
            values = setup[metric + "_mean"] if is_norm or is_cosine else np.atleast_2d(setup[metric + "_mean"])[projection_index]
            std = setup[metric + "_std"] if is_norm or is_cosine else np.atleast_2d(setup[metric + "_std"])[projection_index]
            displayed = np.where(values > 0, values, np.nan) if is_norm else values
            if uncertainty_style == "band":
                line, = ax.plot(indices, displayed, "o-", linewidth=2.3, markersize=4, label=label, zorder=3)
                lower = np.maximum(values - std, norm_floor) if is_norm else np.clip(values - std, -1 if is_cosine else 0, 1)
                upper = np.maximum(values + std, norm_floor) if is_norm else np.clip(values + std, -1 if is_cosine else 0, 1)
                ax.fill_between(indices, lower, upper,
                                color=line.get_color(), alpha=.12, linewidth=0, zorder=0)
            else:
                error = np.stack((np.minimum(np.maximum(values - norm_floor, 0), std), std)) if is_norm else std
                curve = ax.errorbar(indices, displayed, yerr=error, marker="o", capsize=3, label=label)
                line = curve.lines[0]
            if is_norm:
                clipped = values - std <= 0
                ax.scatter(indices[clipped], np.full(clipped.sum(), norm_floor), marker="v", s=25,
                           color=line.get_color(), clip_on=False, zorder=4, label="_nolegend_")
            else:
                omitted = np.zeros(steps, dtype=bool)
                prefix = "cosine" if is_cosine else "projection"
                coverage_index = 0 if is_cosine else projection_index
                if prefix + "_valid_examples" in setup:
                    omitted |= (np.atleast_2d(setup[prefix + "_valid_examples"])[coverage_index]
                                < np.atleast_2d(setup[prefix + "_total_examples"])[coverage_index])
                if prefix + "_valid_runs" in setup:
                    omitted |= np.atleast_2d(setup[prefix + "_valid_runs"])[coverage_index] < setup["runs"]
                if metric + "_runs" in setup:
                    runs = setup[metric + "_runs"]
                    if isinstance(proj_steps, list) and not is_cosine:
                        runs = runs[:, projection_index]
                    omitted |= np.isnan(runs).any(axis=0)
                omitted &= np.isfinite(values)
                if omitted.any():
                    ax.scatter(indices[omitted], values[omitted], s=60, facecolors="none", edgecolors="red",
                               linewidths=1.1, zorder=5, label="_nolegend_")
            if show_runs:
                runs = setup[metric + "_runs"]
                if isinstance(proj_steps, list) and not is_norm and not is_cosine:
                    runs = runs[:, projection_index]
                for run in runs:
                    if is_norm:
                        run = np.where(run > 0, run, np.nan)
                    solid_end = start if start is not None else steps
                    ax.plot(indices[:solid_end], run[:solid_end], color=line.get_color(), linewidth=.9,
                            alpha=.3, zorder=1, label="_nolegend_")
                    if start is not None:
                        ax.plot(indices[start - 1:stop], run[start - 1:stop], "--",
                                 color=line.get_color(), linewidth=.9, alpha=.3, zorder=1, label="_nolegend_")
                        if stop < steps:
                            ax.plot(indices[stop - 1:], run[stop - 1:], color=line.get_color(),
                                    linewidth=.9, alpha=.3, zorder=1, label="_nolegend_")
            if start is not None:
                line.set_data(indices[:start], displayed[:start])
                ax.plot(indices[start - 1:stop], displayed[start - 1:stop], "o--", color=line.get_color(),
                         linewidth=line.get_linewidth(), zorder=line.get_zorder(), label="_nolegend_")
                if stop < steps:
                    ax.plot(indices[stop - 1:], displayed[stop - 1:], "o-", color=line.get_color(),
                            linewidth=line.get_linewidth(), zorder=line.get_zorder(), label="_nolegend_")
        if dash_start is not None:
            ax.axvline(dash_start, color="0.4", linestyle="--", linewidth=1, label="_nolegend_")
        ax.set_xlabel("WRITE step k")
        if projection_index == 0 and not is_cosine:
            ax.set_ylabel("Projected step norm / original step norm")
        if is_norm:
            ax.set_title("Mean WRITE-step norm")
            ax.set_ylabel("WRITE displacement L2 norm")
            ax.set_ylim(bottom=norm_floor)
        elif is_cosine:
            ax.set_title("Consecutive WRITE-step cosine")
            ax.set_ylabel("Mean cosine similarity")
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_title(f"Projection onto the span of the next {span} WRITE step{'s' if span != 1 else ''}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.3)
    if projection_ax is not None:
        low, high = projection_ax.get_ylim()
        projection_ax.set_ylim(min(-.05, low), max(1.05, high))
    axes[0, 0].legend(frameon=False)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def update_svd(path, setups, steps, energy_threshold=.95, dpi=220, dash_start=None, show_n_runs=True, figsize=None):
    """Raw reduced spectra grouped by setup; marker uses the displayed mean spectrum."""
    figure, ax = plt.subplots(figsize=figsize or (9, 5), layout="constrained")
    for index, setup in enumerate(setups):
        values = setup["singular_values_mean"]
        ranks = setup["ranks"]
        rank = setup["retained_rank"]
        label = setup["label"]
        if show_n_runs:
            label += f" ({setup['runs']} run{'s' if setup['runs'] != 1 else ''})"
        if rank is None:
            label += "; zero spectrum, energy undefined"
        line = ax.errorbar(ranks, values, yerr=setup["singular_values_std"],
                          marker="o", capsize=3, label=label)
        if dash_start is not None:
            curve = line.lines[0]
            curve.set_data(ranks[:dash_start], values[:dash_start])
            ax.plot(ranks[dash_start - 1:], values[dash_start - 1:], "o--",
                    color=curve.get_color(), label="_nolegend_")
        if rank is not None:
            color = line.lines[0].get_color()
            ax.scatter([rank], [values[rank - 1]], s=120, facecolors="none", edgecolors=color, linewidths=2)
            ax.annotate(f"{setup['label']}: r={rank} ({energy_threshold * 100:g}%)",
                        (rank, values[rank - 1]), xytext=(12, 15 + 18 * (index % 3)),
                        textcoords="offset points", color=color,
                         arrowprops=dict(arrowstyle="-", color=color))
    if dash_start is not None:
        ax.axvline(dash_start, color="0.4", linestyle="--", linewidth=1, label="_nolegend_")
    ax.set_xlabel("Singular-value index (rank)")
    ax.set_ylabel("WRITE-update singular value")
    ax.set_title(f"Reduced SVD of {steps} flattened WRITE updates")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def k_lr_sweep(path, setups, dpi=220, figsize=None, show_n_runs=True):
    """Plot equal-run means and sample SD on each setup's inference K/lr grid."""
    if not setups:
        raise ValueError("k_lr_sweep requires at least one setup")
    metrics = (
        ("outer_loss", "Outer READ target loss", "magma", 1),
        ("token_accuracy", "Token accuracy (%)", "viridis", 100),
        ("exact_match", "Exact match (%)", "viridis", 100),
    )
    norms = {
        key: loss_scale([np.asarray(setup[key + "_mean"]) * multiplier for setup in setups],
                        log=False)[0]
        for key, _, _, multiplier in metrics
    }
    count = len(setups)
    column_width = max(4.5, max(len(setup["inner_lrs"]) for setup in setups) * 1.2)
    row_height = max(3.5, max(len(setup["steps"]) for setup in setups) * .65 + 1.5)
    outputs = [(path, metrics)] + [
        (path.with_name(path.stem + "_" + metric[0] + path.suffix), (metric,))
        for metric in metrics
    ]
    for destination, rows in outputs:
        figure = plt.figure(figsize=figsize or (column_width * count + 1, row_height * len(rows) + .6),
                            layout="constrained")
        grid = figure.add_gridspec(len(rows), count + 1, width_ratios=[1] * count + [.055])
        for row, (key, title, cmap_name, multiplier) in enumerate(rows):
            norm, cmap = norms[key], plt.get_cmap(cmap_name)
            for col, setup in enumerate(setups):
                ax = figure.add_subplot(grid[row, col])
                means = np.asarray(setup[key + "_mean"]) * multiplier
                stds = np.asarray(setup[key + "_std"]) * multiplier
                steps, lrs = setup["steps"], setup["inner_lrs"]
                image = ax.imshow(means, norm=norm, cmap=cmap, aspect="auto")
                for (k_index, lr_index), mean in np.ndenumerate(means):
                    red, green, blue, _ = cmap(norm(mean))
                    luminance = .2126 * red + .7152 * green + .0722 * blue
                    text_color = "white" if luminance < .5 else "black"
                    std = stds[k_index, lr_index]
                    text = (f"{mean:.1f}\n+/- {std:.1f}" if multiplier == 100
                            else f"{mean:.3g}\n+/- {std:.3g}")
                    ax.text(lr_index, k_index, text, ha="center", va="center",
                            color=text_color, fontsize=9)
                trained = setup.get("trained_settings", [])
                if (len(trained) == setup["runs"] and trained
                        and all(setting == trained[0] for setting in trained)):
                    setting = trained[0]
                    if (setting.get("K") in steps and setting.get("inner_lr") in lrs):
                        ax.plot(lrs.index(setting["inner_lr"]) + .32,
                                steps.index(setting["K"]) - .32, marker="*", markersize=12,
                                markerfacecolor="white", markeredgecolor="black", linestyle="none")
                ax.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
                ax.set_yticks(range(len(steps)), steps)
                ax.set_xticks(np.arange(-.5, len(lrs), 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(steps), 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=1.2, alpha=.7)
                ax.tick_params(which="minor", bottom=False, left=False)
                ax.set_xlabel("Inner learning rate")
                ax.set_ylabel("WRITE steps (K)")
                if row == 0:
                    title_label = setup["label"]
                    if show_n_runs:
                        title_label += f"\n({setup['runs']} run{'s' if setup['runs'] != 1 else ''})"
                    ax.set_title(title_label)
            colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[row, count]))
            colorbar.set_label(title)
        figure.supxlabel("Cells: mean +/- sample SD across runs.\n"
                         "*: shared trained (K, inner lr); all runs agree and setting is on grid.",
                         fontsize=9)
        figure.savefig(destination, dpi=dpi)
        plt.close(figure)


def memory_search(path, setups, quantiles, dpi=220, figsize=None, show_n_runs=False):
    """Stack incremental beneficial gains, not cumulative percentile values."""
    figure, axes = plt.subplots(2, 2, figsize=figsize or (13, 9), layout="constrained")
    metrics = ("token_accuracy", "exact_match", "outer_loss", "inner_loss")
    x = np.arange(len(setups))
    colors = plt.get_cmap("viridis")(np.linspace(.25, .9, len(quantiles)))
    for ax, metric in zip(axes.flat, metrics):
        baseline = np.array([setup[metric + "_baseline"] for setup in setups])
        ax.bar(x, baseline, color="#aeb8c2", label="Baseline")
        previous = baseline.copy()
        for index, quantile in enumerate(quantiles):
            outcomes = np.array([setup[metric + "_outcomes"][index] for setup in setups])
            bottom = previous if metric in ("token_accuracy", "exact_match") else outcomes
            ax.bar(x, np.abs(outcomes - previous), bottom=bottom, color=colors[index],
                   label=f"Beneficial q{quantile:g}")
            previous = outcomes
        # Loss improvements are downward overlays; baseline remains a visible outline.
        ax.bar(x, baseline, facecolor="none", edgecolor="#485460", linewidth=1)
        labels = [setup["label"] + (f"\n({setup['runs']} run{'s' if setup['runs'] != 1 else ''})"
                                   if show_n_runs else "") for setup in setups]
        ax.set_xticks(x, labels)
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(frameon=False)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
