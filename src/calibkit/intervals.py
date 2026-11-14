"""Bootstrap intervals for calibration metrics.

A percentile bootstrap is fine for a nearly unbiased statistic such as the Brier
score. It fails for calibration error. Inside the bootstrap world the "true"
binned error is the original sample's *plug-in* value, so bootstrap replicates of
any calibration-error estimator centre on that, above the estimate. On perfectly
calibrated forecasts the percentile interval for the debiased estimator then
covers the truth far less often than its nominal level (0.69 in a 200-dataset
simulation at nominal 0.95), and the percentile interval for the plug-in ECE
often excludes the plug-in estimate itself.

:func:`calibration_error_interval` fixes this by removing that shift, which is
known exactly inside the bootstrap world, before reading off the percentiles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_pair
from .binning import Strategy
from .metrics import debiased_calibration_error, expected_calibration_error

FloatArray = NDArray[np.float64]


def _check(resamples: int, confidence: float) -> None:
    if not isinstance(resamples, int) or resamples < 2:
        raise ValueError(f"resamples must be an integer >= 2, got {resamples!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")


def percentile_interval(
    stat: Callable[[FloatArray, FloatArray], float],
    y_true: ArrayLike,
    y_prob: ArrayLike,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``stat(y, p)``, resampling rows.

    Appropriate for nearly unbiased statistics such as :func:`brier_score`. Use
    :func:`calibration_error_interval` for calibration error.
    """
    _check(resamples, confidence)
    y, p = check_pair(y_true, y_prob)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, y.size, y.size)
        values[b] = stat(y[idx], p[idx])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def calibration_error_interval(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap interval for the binned L2 calibration error.

    With ``theta`` the debiased squared estimate on the data, ``t`` the plug-in
    squared error on the data (the bootstrap world's true value), and ``q_lo``,
    ``q_hi`` the bootstrap quantiles of the debiased estimate, the interval for
    the squared error is ``[q_lo - (t - theta), q_hi - (t - theta)]``: the
    percentile interval with the bootstrap world's offset between its target
    and the estimate removed. Both ends are then clipped at zero and
    square-rooted, so the result is on the scale of
    :func:`debiased_calibration_error`.

    A basic (reflected) interval, ``[theta - (q_hi - t), theta - (q_lo - t)]``,
    removes the same offset but flips the bootstrap distribution's skew. Near
    zero the distribution of a squared error is strongly right-skewed, and the
    flip can produce an interval collapsed onto zero that misses a small but
    real error. Shifting keeps the skew's orientation.

    Its coverage, against the percentile interval's, is measured in the README's
    coverage table.
    """
    _check(resamples, confidence)
    y, p = check_pair(y_true, y_prob)
    theta = debiased_calibration_error(y, p, n_bins, strategy, squared=True)
    target = expected_calibration_error(y, p, n_bins, strategy, norm="l2") ** 2
    values = debiased_replicates(y, p, n_bins, strategy, resamples, seed)
    alpha = (1.0 - confidence) / 2.0
    q_lo, q_hi = np.quantile(values, [alpha, 1.0 - alpha])
    shift = target - theta
    lo_sq = q_lo - shift
    hi_sq = q_hi - shift
    return float(np.sqrt(max(lo_sq, 0.0))), float(np.sqrt(max(hi_sq, 0.0)))


def debiased_replicates(
    y: FloatArray, p: FloatArray, n_bins: int, strategy: Strategy, resamples: int, seed: int
) -> FloatArray:
    """Bootstrap replicates of the squared debiased estimate (rows resampled)."""
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, y.size, y.size)
        values[b] = debiased_calibration_error(y[idx], p[idx], n_bins, strategy, squared=True)
    return values
