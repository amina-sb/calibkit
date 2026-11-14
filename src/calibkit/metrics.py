"""Calibration metrics for binary forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_pair
from .binning import Strategy, assign_bins, bin_edges

Norm = Literal["l1", "l2"]


@dataclass(frozen=True)
class ReliabilityTable:
    """Per-bin summary of a set of forecasts.

    Empty bins have ``count == 0`` and NaN for both ``mean_confidence`` and
    ``observed_frequency``; they carry no weight in any metric.
    """

    edges: NDArray[np.float64]
    count: NDArray[np.int64]
    mean_confidence: NDArray[np.float64]
    observed_frequency: NDArray[np.float64]

    @property
    def n_bins(self) -> int:
        return int(self.count.size)

    @property
    def weight(self) -> NDArray[np.float64]:
        """Fraction of all forecasts that fall in each bin."""
        return np.asarray(self.count / self.count.sum(), dtype=np.float64)

    @property
    def gap(self) -> NDArray[np.float64]:
        """``mean_confidence - observed_frequency``; positive means overconfident."""
        return self.mean_confidence - self.observed_frequency


def _bin_sums(
    idx: NDArray[np.intp], values: NDArray[np.float64], n_bins: int
) -> NDArray[np.float64]:
    return np.bincount(idx, weights=values, minlength=n_bins).astype(np.float64)


def reliability_table(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
) -> ReliabilityTable:
    """Summarise forecasts by bin: count, mean confidence and observed frequency.

    The non-empty rows reproduce ``sklearn.calibration.calibration_curve``
    (``prob_pred`` and ``prob_true``) for the same bins.
    """
    y, p = check_pair(y_true, y_prob)
    edges = bin_edges(p, n_bins, strategy)
    k = edges.size - 1
    idx = assign_bins(p, edges)
    count = np.bincount(idx, minlength=k).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = _bin_sums(idx, p, k) / count
        freq = _bin_sums(idx, y, k) / count
    return ReliabilityTable(edges, count, conf, freq)


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
    norm: Norm = "l1",
) -> float:
    """Plug-in expected calibration error.

    ``norm="l1"`` is the usual ECE, ``sum_k w_k |conf_k - freq_k|``.
    ``norm="l2"`` is ``sqrt(sum_k w_k (conf_k - freq_k)^2)``, which is never
    smaller than the L1 value.

    This estimator is biased upwards: with a finite sample, ``freq_k`` is noisy,
    and ``|noise|`` has positive mean even when the forecasts are perfectly
    calibrated. The bias grows with the number of bins and shrinks with the
    sample size.
    """
    t = reliability_table(y_true, y_prob, n_bins, strategy)
    mask = t.count > 0
    w = t.weight[mask]
    d = np.abs(t.gap[mask])
    if norm == "l1":
        return float(np.dot(w, d))
    if norm == "l2":
        return float(np.sqrt(np.dot(w, d * d)))
    raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")


def maximum_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
) -> float:
    """Largest absolute gap between confidence and frequency over non-empty bins."""
    t = reliability_table(y_true, y_prob, n_bins, strategy)
    return float(np.max(np.abs(t.gap[t.count > 0])))


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared difference between forecast and outcome."""
    y, p = check_pair(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class BrierDecomposition:
    """Five-term decomposition of the Brier score over a binning.

    ``brier == reliability - resolution + uncertainty
    + within_bin_variance - 2 * within_bin_covariance`` holds exactly, up to
    floating-point rounding.

    The familiar three-term version (Murphy, 1973) drops the last two terms. That
    is exact only when every forecast in a bin takes the same value; for binned
    continuous forecasts the two extra terms are generally non-zero (Stephenson,
    Coelho and Jolliffe, 2008, *Weather and Forecasting* 23(4), 752-757).
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float

    @property
    def three_term(self) -> float:
        """``reliability - resolution + uncertainty``, the Murphy approximation."""
        return self.reliability - self.resolution + self.uncertainty

    @property
    def reconstructed(self) -> float:
        """Sum of all five terms; equals ``brier``."""
        return self.three_term + self.within_bin_variance - 2.0 * self.within_bin_covariance


def brier_decomposition(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
) -> BrierDecomposition:
    """Decompose the Brier score into reliability, resolution, uncertainty and two
    within-bin terms.

    With ``p_bar_k`` and ``o_bar_k`` the mean forecast and mean outcome in bin
    ``k`` and ``o_bar`` the overall base rate:

    * reliability = ``sum_k n_k (p_bar_k - o_bar_k)^2 / n``
    * resolution = ``sum_k n_k (o_bar_k - o_bar)^2 / n``
    * uncertainty = ``o_bar (1 - o_bar)``
    * within-bin variance = ``sum_i (p_i - p_bar_k(i))^2 / n``
    * within-bin covariance = ``sum_i (p_i - p_bar_k(i)) (y_i - o_bar_k(i)) / n``
    """
    y, p = check_pair(y_true, y_prob)
    n = y.size
    t = reliability_table(y, p, n_bins, strategy)
    idx = assign_bins(p, t.edges)
    mask = t.count > 0
    w = t.weight[mask]
    conf = t.mean_confidence[mask]
    freq = t.observed_frequency[mask]
    base = float(np.mean(y))

    dp = p - t.mean_confidence[idx]
    dy = y - t.observed_frequency[idx]
    return BrierDecomposition(
        brier=float(np.mean((p - y) ** 2)),
        reliability=float(np.dot(w, (conf - freq) ** 2)),
        resolution=float(np.dot(w, (freq - base) ** 2)),
        uncertainty=base * (1.0 - base),
        within_bin_variance=float(np.dot(dp, dp) / n),
        within_bin_covariance=float(np.dot(dp, dy) / n),
    )


def debiased_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    strategy: Strategy = "uniform",
    squared: bool = False,
) -> float:
    """L2 calibration error with the finite-sample noise term removed.

    Conditional on the forecasts, a bin's observed frequency ``f_k`` is the mean
    of ``n_k`` Bernoulli outcomes with mean ``mu_k``, so

        E[(c_k - f_k)^2] = (c_k - mu_k)^2 + mu_k (1 - mu_k) / n_k.

    The plug-in estimator therefore overstates the squared error by the second
    term, and ``f_k (1 - f_k) / (n_k - 1)`` is an unbiased estimate of it. Removing
    it per bin gives an unbiased estimate of the binned squared calibration error
    ``sum_k w_k (c_k - mu_k)^2``. This is the correction used by Kumar, Liang and
    Ma (2019), *Verified Uncertainty Calibration*, NeurIPS.

    Parameters
    ----------
    squared:
        Return the squared-error estimate itself, which is unbiased and can be
        negative. Otherwise return ``sqrt(max(estimate, 0))``, which is on the same
        scale as ``expected_calibration_error(norm="l2")``. The square root is not
        unbiased, but it is much less biased than the plug-in value.

    The correction is exactly unbiased when the outcomes in a bin share one mean.
    When their true probabilities ``mu_i`` differ within a bin, it removes slightly
    too much, and the squared estimate is low by ``Var_i(mu_i) / (n_k - 1)`` in that
    bin, where ``Var_i`` is the population variance of the ``mu_i``. That is small
    when bins are narrow, and the test suite checks both cases by exact
    enumeration.

    Bins holding a single forecast have no variance estimate and contribute their
    uncorrected term.
    """
    t = reliability_table(y_true, y_prob, n_bins, strategy)
    mask = t.count > 0
    n_k = t.count[mask].astype(np.float64)
    f = t.observed_frequency[mask]
    sq = t.gap[mask] ** 2
    correction = np.zeros_like(sq)
    many = n_k > 1
    correction[many] = f[many] * (1.0 - f[many]) / (n_k[many] - 1.0)
    estimate = float(np.dot(t.weight[mask], sq - correction))
    if squared:
        return estimate
    return float(np.sqrt(max(estimate, 0.0)))
