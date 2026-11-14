"""Reliability diagrams. Requires matplotlib (``pip install calibkit[plot]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy
from .metrics import debiased_calibration_error, reliability_table

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def reliability_diagram(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
    confidence: float = 0.95,
    title: str | None = None,
    extra: dict[str, ArrayLike] | None = None,
) -> Figure:
    """Draw observed frequency against mean confidence, with forecast counts below.

    Each non-empty bin is plotted at its mean confidence with a Wilson interval
    on its observed frequency. Both axes always show the full [0, 1] range, so the
    picture cannot flatter a model by zooming in. The lower panel shows how
    many forecasts each bin holds, because a large gap in a nearly empty bin
    means little.

    ``extra`` maps labels to further forecasts for the same outcomes, for example
    a recalibrated version, and draws them on the same axes with distinct markers
    and line styles so the series remain distinguishable in greyscale.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "reliability_diagram needs matplotlib: pip install calibkit[plot]"
        ) from exc

    series: dict[str, ArrayLike] = {"forecast": y_prob}
    if extra:
        series.update(extra)
    styles = [
        ("o", "-", "#1f4e79"),
        ("s", "--", "#c55a11"),
        ("^", ":", "#548235"),
        ("D", "-.", "#7f6000"),
    ]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]}
    )
    top.plot(
        [0, 1],
        [0, 1],
        color="0.55",
        linewidth=1,
        linestyle=(0, (4, 3)),
        label="perfect calibration",
    )
    width = 1.0 / n_bins
    for i, (label, probs) in enumerate(series.items()):
        marker, line, colour = styles[i % len(styles)]
        t = reliability_table(y_true, probs, n_bins, strategy)
        m = t.count > 0
        lo, hi = t.frequency_interval(confidence)
        conf, freq = t.mean_confidence[m], t.observed_frequency[m]
        err = np.vstack([freq - lo[m], hi[m] - freq])
        ce = debiased_calibration_error(y_true, probs, n_bins, strategy)
        top.errorbar(
            conf,
            freq,
            yerr=err,
            marker=marker,
            linestyle=line,
            color=colour,
            markersize=5,
            capsize=2.5,
            linewidth=1.4,
            label=f"{label} (debiased CE {ce:.3f})",
        )
        offset = (i - (len(series) - 1) / 2) * width / max(len(series), 1)
        centres = 0.5 * (t.edges[:-1] + t.edges[1:])
        bottom.bar(
            centres + offset,
            t.count,
            width=width / max(len(series), 1) * 0.9,
            color=colour,
            alpha=0.85,
            hatch=["", "///", "...", "xx"][i % 4],
            edgecolor="white",
            linewidth=0.4,
        )

    top.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), ylabel="observed frequency")
    top.set_aspect("equal", adjustable="box")
    top.grid(True, color="0.92")
    top.legend(loc="upper left", fontsize=8, frameon=False)
    if title:
        top.set_title(title)
    bottom.set(xlim=(-0.02, 1.02), xlabel="mean predicted probability", ylabel="forecasts")
    bottom.grid(True, axis="y", color="0.92")
    fig.text(
        0.99,
        0.005,
        f"{n_bins} {strategy} bins; bars are {confidence:.0%} Wilson intervals",
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout()
    return fig
