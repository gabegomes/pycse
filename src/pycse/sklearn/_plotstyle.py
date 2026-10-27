"""Shared chart styling for the sklearn-compatible models' ``plot`` methods.

One place for the color tokens and the recessive-chrome axes used by every
model, so a KAN, DPOSE or NNBR figure reads as one system. Series colors come
from a validated categorical palette; text and data points use ink tokens.
"""

import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"  # primary text
INK_2 = "#52514e"  # secondary text, data points
MUTED = "#898781"  # tick marks, reference lines
GRID = "#e1e0d9"  # hairline gridlines
BASELINE = "#c3c2b7"  # axis lines
SURFACE = "#fcfcfb"  # chart surface, marker rings
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots, fixed order
PX = 0.75  # one CSS pixel in points


def style_axes(ax, grid=True):
    """Hairline, recessive axes: no top/right spines, solid hairline grid."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(PX)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(False)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=PX, linestyle="-")
        ax.set_axisbelow(True)


def finish_axes(ax, title, xlabel="X", ylabel="y"):
    """Ink-colored labels, left-aligned bold title, frameless legend."""
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, labelcolor=INK_2)


def plot_prediction(
    ax,
    X,
    y,
    mean,
    std=None,
    *,
    label="prediction",
    title="",
    members=None,
    inner_band=False,
    color=SERIES[0],
):
    """Draw a model's prediction against data in the shared style.

    One input column: the sorted prediction curve, a wash for ``mean ± 2 std``
    (and ``± 1 std`` when ``inner_band``), optional faint ensemble ``members``
    (one column per member), and the data as neutral dots. Several input
    columns: a parity plot (predicted vs observed) with ``2 std`` error bars.

    Returns the axes.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    mean = np.asarray(mean).ravel()
    std = None if std is None else np.asarray(std).ravel()
    style_axes(ax)

    if X.ndim == 2 and X.shape[1] > 1:
        lo, hi = float(min(y.min(), mean.min())), float(max(y.max(), mean.max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=PX, zorder=1, label="y = x")
        if std is not None and np.any(std > 0):
            ax.errorbar(
                y, mean, yerr=2 * std, fmt="none", ecolor=INK_2, elinewidth=PX,
                capsize=2, alpha=0.6, zorder=2, label="±2σ (95% CI)",
            )  # fmt: skip
        ax.scatter(
            y, mean, s=(8 * PX) ** 2, color=color, edgecolors=SURFACE, linewidths=PX,
            alpha=0.8, label=label, zorder=3,
        )  # fmt: skip
        finish_axes(ax, title, xlabel="observed y", ylabel="predicted y")
        return ax

    x = X.ravel()
    order = np.argsort(x)
    xs, ms = x[order], mean[order]
    if std is not None:
        ss = std[order]
        ax.fill_between(
            xs, ms - 2 * ss, ms + 2 * ss, alpha=0.12, color=color, linewidth=0,
            label="±2σ (95% CI)", zorder=1,
        )  # fmt: skip
        if inner_band:
            ax.fill_between(
                xs, ms - ss, ms + ss, alpha=0.15, color=color, linewidth=0,
                label="±1σ (68% CI)", zorder=1,
            )  # fmt: skip
    if members is not None:
        P = np.asarray(members)[order, :]
        ax.plot(xs, P, color=MUTED, alpha=0.1, linewidth=PX, zorder=2)
    ax.plot(xs, ms, color=color, linewidth=2 * PX, label=label, zorder=3)
    ax.scatter(
        x, y, s=(8 * PX) ** 2, color=INK_2, edgecolors=SURFACE, linewidths=PX,
        alpha=0.8, label="data", zorder=4,
    )  # fmt: skip
    finish_axes(ax, title)
    return ax


def current_figure():
    return plt.gcf()
