"""Binning schemes for forecast probabilities.

Edge convention
---------------
Bins are right-closed: bin ``k`` holds forecasts in ``(e[k], e[k+1]]``, except the
first bin, which also includes its left edge, so every value in [0, 1] lands in
exactly one bin. This is the convention ``sklearn.calibration.calibration_curve``
uses (it calls ``np.searchsorted(edges[1:-1], p)`` with the default
``side="left"``). Matching it matters more than it looks: classifiers routinely
emit round numbers such as 0.5 or 0.1, and those sit exactly on equal-width
edges, so two libraries that disagree on the convention report different ECEs for
the same forecasts.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_probabilities

Strategy = Literal["uniform", "quantile"]


def bin_edges(
    y_prob: ArrayLike, n_bins: int, strategy: Strategy = "uniform"
) -> NDArray[np.float64]:
    """Return monotone bin edges spanning [0, 1].

    Parameters
    ----------
    y_prob:
        Forecast probabilities. Only used by ``strategy="quantile"``.
    n_bins:
        Requested number of bins.
    strategy:
        ``"uniform"`` gives equal-width bins. ``"quantile"`` gives equal-mass bins
        from the empirical quantiles of ``y_prob``. Tied forecasts can make two
        quantile edges coincide; duplicates are removed, so fewer than ``n_bins``
        bins may be returned rather than bins that can never be populated.
    """
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, int(n_bins) + 1)
    if strategy == "quantile":
        p = check_probabilities(y_prob)
        inner = np.quantile(p, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
        edges = np.concatenate(([0.0], inner, [1.0]))
        return np.unique(edges)
    raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")


def assign_bins(y_prob: ArrayLike, edges: ArrayLike) -> NDArray[np.intp]:
    """Return the bin index of each forecast under the right-closed convention."""
    p = check_probabilities(y_prob)
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be a 1-D array with at least two entries")
    if np.any(np.diff(e) <= 0):
        raise ValueError("edges must be strictly increasing")
    return np.searchsorted(e[1:-1], p, side="left")
