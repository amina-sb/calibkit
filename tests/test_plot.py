import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from calibkit.plot import reliability_diagram


def _data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(0)
    p = rng.random(800)
    y = (rng.random(800) < p**1.5).astype(np.float64)
    return y, p


def test_diagram_axes_span_unit_square_and_show_every_series() -> None:
    y, p = _data()
    fig = reliability_diagram(y, p, n_bins=10, extra={"sharpened": p**1.5})
    top, bottom = fig.axes
    lo, hi = top.get_xlim()
    assert lo <= 0.0 and hi >= 1.0
    lo, hi = top.get_ylim()
    assert lo <= 0.0 and hi >= 1.0
    legend = top.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels[0] == "perfect calibration"
    assert labels[1].startswith("forecast (debiased CE")
    assert labels[2].startswith("sharpened (debiased CE")
    # One bar per bin per series.
    assert len(bottom.patches) == 20
    heights = [int(b.get_height()) for b in bottom.patches if isinstance(b, Rectangle)]
    assert sum(heights) == 1600
    matplotlib.pyplot.close(fig)


def test_diagram_series_use_distinct_markers_and_line_styles() -> None:
    y, p = _data()
    fig = reliability_diagram(y, p, extra={"a": p**1.2, "b": p**0.8})
    # Error-bar caps are drawn as marker-only lines; keep the series lines.
    lines = [
        ln
        for ln in fig.axes[0].get_lines()
        if ln.get_linestyle() != "None" and ln.get_marker() not in ("None", "", None)
    ]
    markers = {ln.get_marker() for ln in lines}
    styles = {ln.get_linestyle() for ln in lines}
    assert len(markers) == 3 and len(styles) == 3
    matplotlib.pyplot.close(fig)


def test_diagram_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        reliability_diagram([0, 1], [0.2, 1.4])
