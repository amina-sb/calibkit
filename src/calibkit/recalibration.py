"""Post-hoc recalibrators.

Every recalibrator has the same two-method interface::

    cal = IsotonicCalibrator().fit(p_calibration, y_calibration)
    p_calibrated = cal.predict(p_test)

Fit on data the classifier was *not* trained on, and evaluate on data the
recalibrator was not fitted on. A recalibrator scored on its own fitting set looks
far better than it is. Histogram binning, for example, scores an in-sample ECE of
exactly zero on its own bins whatever the data.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_pair, check_probabilities
from .binning import Strategy, assign_bins, bin_edges

FloatArray = NDArray[np.float64]

#: Probabilities are clipped to ``[EPS, 1 - EPS]`` before taking a logit.
EPS = 1e-12


R = TypeVar("R", bound="Recalibrator")


class NotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``."""


class ConvergenceWarning(UserWarning):
    """Raised when an iterative fit stops before converging."""


def logit(p: FloatArray) -> FloatArray:
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.asarray(np.log(q) - np.log1p(-q), dtype=np.float64)


def expit(z: FloatArray) -> FloatArray:
    # Two branches so neither exp() overflows.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class Recalibrator(ABC):
    """Base class: validates inputs and guards against predicting before fitting."""

    _fitted: bool = False

    def fit(self: R, y_prob: ArrayLike, y_true: ArrayLike) -> R:
        y, p = check_pair(y_true, y_prob)
        self._fit(p, y)
        self._fitted = True
        return self

    def predict(self, y_prob: ArrayLike) -> FloatArray:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before predict")
        p = check_probabilities(y_prob)
        return np.clip(self._predict(p), 0.0, 1.0)

    @abstractmethod
    def _fit(self, p: FloatArray, y: FloatArray) -> None: ...

    @abstractmethod
    def _predict(self, p: FloatArray) -> FloatArray: ...


class PlattCalibrator(Recalibrator):
    """Logistic recalibration on the logit: ``q = expit(a * logit(p) + b)``.

    Parameters
    ----------
    smooth_targets:
        Replace the 0/1 targets with ``1 / (n_neg + 2)`` and
        ``(n_pos + 1) / (n_pos + 2)`` (Platt, 1999, section 2.2). This is a mild
        prior that keeps the fit finite when the classes are separable. With
        smoothing off, the fit is plain maximum likelihood and matches an
        unpenalised logistic regression.
    monotone:
        Constrain the slope to ``a >= 0``. A negative maximum-likelihood slope means
        the scores rank the calibration data worse than chance, and following it
        would reverse the classifier's ordering, changing its decisions rather
        than its probabilities. Because the log-likelihood is concave, the
        constrained optimum then lies at ``a = 0``, so every input maps to the
        (smoothed) base rate. A :class:`ConvergenceWarning` is not raised for
        this; a ``UserWarning`` is.
    max_iter, tol:
        Newton iterations and the convergence tolerance on the step size.
    """

    def __init__(
        self,
        smooth_targets: bool = True,
        monotone: bool = True,
        max_iter: int = 100,
        tol: float = 1e-10,
    ):
        self.smooth_targets = smooth_targets
        self.monotone = monotone
        self.max_iter = max_iter
        self.tol = tol
        self.a_ = 1.0
        self.b_ = 0.0

    def _fit(self, p: FloatArray, y: FloatArray) -> None:
        x = logit(p)
        t = y.copy()
        if self.smooth_targets:
            n_pos = float(y.sum())
            n_neg = float(y.size - n_pos)
            t = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        theta = np.array([1.0, 0.0])
        design = np.column_stack([x, np.ones_like(x)])

        def nll(th: FloatArray) -> float:
            z = design @ th
            # log(1 + e^z) - t z, computed stably.
            return float(np.sum(np.logaddexp(0.0, z) - t * z))

        current = nll(theta)
        converged = False
        for _ in range(self.max_iter):
            q = expit(design @ theta)
            grad = design.T @ (q - t)
            w = q * (1.0 - q)
            hess = design.T @ (design * w[:, None]) + 1e-12 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            # Backtracking keeps Newton from overshooting on badly scaled inputs.
            lr = 1.0
            while lr > 1e-10:
                candidate = theta - lr * step
                value = nll(candidate)
                if value <= current:
                    break
                lr *= 0.5
            theta, current = candidate, value
            if np.max(np.abs(lr * step)) < self.tol:
                converged = True
                break
        if not converged:
            warnings.warn(
                "Platt scaling did not converge; the classes may be separable. "
                "Use smooth_targets=True.",
                ConvergenceWarning,
                stacklevel=3,
            )
        a, b = float(theta[0]), float(theta[1])
        if self.monotone and a < 0.0:
            warnings.warn(
                f"Platt slope was negative ({a:.3g}); the scores are anti-correlated "
                "with the labels. Mapping every input to the base rate instead.",
                UserWarning,
                stacklevel=3,
            )
            a, b = 0.0, float(logit(np.array([np.mean(t)]))[0])
        self.a_, self.b_ = a, b

    def _predict(self, p: FloatArray) -> FloatArray:
        return expit(self.a_ * logit(p) + self.b_)


def pool_adjacent_violators(y: FloatArray, w: FloatArray) -> FloatArray:
    """Weighted least-squares isotonic (non-decreasing) fit to ``y`` in given order.

    Standard stack-based pool-adjacent-violators, O(n).
    """
    n = y.size
    means = np.empty(n)
    weights = np.empty(n)
    sizes = np.empty(n, dtype=np.intp)
    top = -1
    for i in range(n):
        top += 1
        means[top], weights[top], sizes[top] = y[i], w[i], 1
        while top > 0 and means[top - 1] > means[top]:
            total = weights[top - 1] + weights[top]
            means[top - 1] = (means[top - 1] * weights[top - 1] + means[top] * weights[top]) / total
            weights[top - 1] = total
            sizes[top - 1] += sizes[top]
            top -= 1
    return np.repeat(means[: top + 1], sizes[: top + 1])


class IsotonicCalibrator(Recalibrator):
    """Non-decreasing step map fitted by pool-adjacent-violators.

    Tied inputs are first pooled to their mean outcome, weighted by count. At
    predict time the map is linearly interpolated between the fitted points and
    held constant outside the fitted range, as
    ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`` does.
    """

    def __init__(self) -> None:
        self.x_: FloatArray = np.empty(0)
        self.y_: FloatArray = np.empty(0)

    def _fit(self, p: FloatArray, y: FloatArray) -> None:
        x, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=y)
        fitted = pool_adjacent_violators(sums / counts, counts.astype(np.float64))
        self.x_, self.y_ = x, fitted

    def _predict(self, p: FloatArray) -> FloatArray:
        return np.asarray(np.interp(p, self.x_, self.y_), dtype=np.float64)


class HistogramCalibrator(Recalibrator):
    """Replace each forecast by the observed frequency of its bin.

    Parameters
    ----------
    n_bins, strategy:
        Bins, as in :func:`calibkit.bin_edges`. For ``"quantile"`` the edges are
        learned from the fitting data.
    prior_strength:
        Pseudo-count pulling each bin's frequency toward the bin's midpoint, i.e.
        ``(positives + s * midpoint) / (count + s)``. Zero, the default, is plain
        histogram binning. An empty bin always maps to its midpoint.

    Note that the output is a monotone map only if the observed frequencies happen
    to increase across bins; histogram binning does not enforce that.
    """

    def __init__(
        self, n_bins: int = 15, strategy: Strategy = "uniform", prior_strength: float = 0.0
    ):
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        self.n_bins = n_bins
        self.strategy = strategy
        self.prior_strength = prior_strength
        self.edges_: FloatArray = np.empty(0)
        self.values_: FloatArray = np.empty(0)

    def _fit(self, p: FloatArray, y: FloatArray) -> None:
        edges = bin_edges(p, self.n_bins, self.strategy)
        k = edges.size - 1
        idx = assign_bins(p, edges)
        count = np.bincount(idx, minlength=k).astype(np.float64)
        pos = np.bincount(idx, weights=y, minlength=k)
        mid = 0.5 * (edges[:-1] + edges[1:])
        s = self.prior_strength
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (pos + s * mid) / (count + s)
        self.edges_ = edges
        self.values_ = np.where(count + s > 0, values, mid)

    def _predict(self, p: FloatArray) -> FloatArray:
        return self.values_[assign_bins(p, self.edges_)]


def cross_calibrate(
    make: Callable[[], Recalibrator],
    y_prob: ArrayLike,
    y_true: ArrayLike,
    n_splits: int = 5,
    seed: int = 0,
) -> FloatArray:
    """Out-of-fold recalibrated probabilities.

    The data are split into ``n_splits`` stratified folds. Each fold is
    recalibrated by a fresh recalibrator from ``make()`` fitted on the other
    folds only, so no forecast is ever adjusted using its own label. Score the
    result with any metric to get an honest estimate of what the recalibrator
    does on new data, without holding out a separate set.
    """
    y, p = check_pair(y_true, y_prob)
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}")
    n_pos = int(y.sum())
    if min(n_pos, y.size - n_pos) < n_splits:
        raise ValueError("each class needs at least n_splits examples for stratified folds")
    rng = np.random.default_rng(seed)
    fold = np.empty(y.size, dtype=np.intp)
    for cls in (0.0, 1.0):
        members = rng.permutation(np.flatnonzero(y == cls))
        fold[members] = np.arange(members.size) % n_splits
    out = np.empty_like(p)
    for k in range(n_splits):
        test = fold == k
        cal = make().fit(p[~test], y[~test])
        out[test] = cal.predict(p[test])
    return out
