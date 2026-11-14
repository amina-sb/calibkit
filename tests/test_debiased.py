"""Exact checks of the debiased estimator by enumerating every outcome vector."""

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from calibkit.metrics import debiased_calibration_error, expected_calibration_error


def expectation(p: list[float], mu: list[float], n_bins: int) -> tuple[float, float]:
    """Exact expectation of the plug-in and debiased squared errors."""
    plug = deb = 0.0
    for ys in itertools.product([0, 1], repeat=len(p)):
        prob = float(np.prod([m if y else 1 - m for y, m in zip(ys, mu, strict=True)]))
        plug += prob * expected_calibration_error(ys, p, n_bins, norm="l2") ** 2
        deb += prob * debiased_calibration_error(ys, p, n_bins, squared=True)
    return plug, deb


def binned_truth(p: list[float], mu: list[float], edges: NDArray[np.float64]) -> float:
    pa, ma = np.array(p), np.array(mu)
    idx = np.searchsorted(edges[1:-1], pa)
    total = 0.0
    for k in np.unique(idx):
        m = idx == k
        total += m.mean() * (pa[m].mean() - ma[m].mean()) ** 2
    return float(total)


def test_unbiased_when_outcomes_share_a_mean_within_each_bin() -> None:
    p = [0.1, 0.15, 0.2, 0.7, 0.75, 0.8]
    mu = [0.3, 0.3, 0.3, 0.6, 0.6, 0.6]
    plug, deb = expectation(p, mu, n_bins=2)
    truth = binned_truth(p, mu, np.linspace(0.0, 1.0, 3, dtype=np.float64))
    assert deb == pytest.approx(truth, abs=1e-14)
    # The plug-in overstates by sum_k w_k mu_k (1 - mu_k) / n_k.
    assert plug - truth == pytest.approx(0.5 * 0.21 / 3 + 0.5 * 0.24 / 3, abs=1e-14)


def test_unbiased_for_perfectly_calibrated_bin_constant_forecasts() -> None:
    p = [0.25] * 4 + [0.6] * 4
    plug, deb = expectation(p, p, n_bins=5)
    assert deb == pytest.approx(0.0, abs=1e-15)
    assert plug > 0.04


def test_low_by_within_bin_variance_when_means_differ() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    mu = [0.05, 0.15, 0.35, 0.45]
    _, deb = expectation(p, mu, n_bins=1)
    truth = binned_truth(p, mu, np.array([0.0, 1.0]))
    expected_shortfall = float(np.var(mu)) / (len(mu) - 1)
    assert truth - deb == pytest.approx(expected_shortfall, abs=1e-14)


def test_square_root_form_is_clipped_at_zero() -> None:
    # Two positives in a bin of forecasts at 0.5: plug-in 0.25, correction 0.
    # One positive and one negative: plug-in 0, correction 0.25 -> negative.
    assert debiased_calibration_error([0, 1], [0.5, 0.5], 1, squared=True) == pytest.approx(-0.25)
    assert debiased_calibration_error([0, 1], [0.5, 0.5], 1) == 0.0


def test_singleton_bins_keep_uncorrected_term() -> None:
    value = debiased_calibration_error([1], [0.3], 10, squared=True)
    assert value == pytest.approx(0.49)
