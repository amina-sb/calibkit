import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from calibkit.metrics import (
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_table,
    wilson_interval,
)


@st.composite
def forecasts(draw: st.DrawFn) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = draw(st.integers(1, 300))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    if draw(st.booleans()):
        # Round to a coarse grid so many forecasts sit exactly on bin edges.
        p = np.round(p, 1)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def _sample(n: int = 2000, seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 3, size=n)
    y = (rng.uniform(size=n) < p**1.3).astype(int)
    return y, p


# ---- reference implementations ------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
@pytest.mark.parametrize("rounded", [False, True])
def test_reliability_table_matches_sklearn(strategy: str, rounded: bool) -> None:
    y, p = _sample()
    if rounded:
        p = np.round(p, 1)
    t = reliability_table(y, p, 10, strategy)  # type: ignore[arg-type]
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy=strategy)
    m = t.count > 0
    np.testing.assert_allclose(t.observed_frequency[m], prob_true, rtol=1e-12)
    np.testing.assert_allclose(t.mean_confidence[m], prob_pred, rtol=1e-12)


@given(forecasts())
def test_brier_matches_sklearn(data: tuple[NDArray[np.float64], NDArray[np.float64]]) -> None:
    y, p = data
    assert brier_score(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-15)


def test_ece_worked_example() -> None:
    # Two bins of width 0.5. Bin 0: forecasts 0.2, 0.4 (mean 0.3), outcomes 0, 1
    # (freq 0.5), gap 0.2. Bin 1: forecasts 0.6, 0.8, 0.9, 0.9 (mean 0.8),
    # outcomes 1, 1, 1, 1 (freq 1.0), gap 0.2. ECE = 2/6 * 0.2 + 4/6 * 0.2 = 0.2.
    y = [0, 1, 1, 1, 1, 1]
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.9]
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.2)
    assert maximum_calibration_error(y, p, n_bins=2) == pytest.approx(0.2)


# ---- properties ---------------------------------------------------------------


@given(forecasts(), st.integers(1, 25))
def test_metric_bounds_and_ordering(
    data: tuple[NDArray[np.float64], NDArray[np.float64]], n_bins: int
) -> None:
    y, p = data
    l1 = expected_calibration_error(y, p, n_bins)
    l2 = expected_calibration_error(y, p, n_bins, norm="l2")
    mce = maximum_calibration_error(y, p, n_bins)
    assert 0.0 <= l1 <= l2 + 1e-12
    assert l2 <= mce + 1e-12
    assert mce <= 1.0


@given(forecasts(), st.integers(0, 2**32 - 1))
def test_ece_invariant_to_row_order(
    data: tuple[NDArray[np.float64], NDArray[np.float64]], seed: int
) -> None:
    y, p = data
    perm = np.random.default_rng(seed).permutation(y.size)
    a = expected_calibration_error(y, p, 10)
    b = expected_calibration_error(y[perm], p[perm], 10)
    assert a == pytest.approx(b, abs=1e-12)


def test_ece_is_zero_when_frequencies_match_exactly() -> None:
    # Ten forecasts of 0.3 with three positives, ten of 0.8 with eight positives.
    p = np.repeat([0.3, 0.8], 10)
    y = np.array([1] * 3 + [0] * 7 + [1] * 8 + [0] * 2)
    assert expected_calibration_error(y, p, 10) == pytest.approx(0.0, abs=1e-15)


@settings(max_examples=200)
@given(forecasts(), st.integers(1, 25), st.sampled_from(["uniform", "quantile"]))
def test_five_term_decomposition_is_exact(
    data: tuple[NDArray[np.float64], NDArray[np.float64]], n_bins: int, strategy: str
) -> None:
    y, p = data
    d = brier_decomposition(y, p, n_bins, strategy)  # type: ignore[arg-type]
    assert d.reconstructed == pytest.approx(d.brier, abs=1e-12)
    assert d.within_bin_variance >= 0.0


def test_three_term_identity_exact_for_bin_constant_forecasts() -> None:
    rng = np.random.default_rng(3)
    p = rng.choice([0.05, 0.35, 0.65, 0.95], size=5000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    d = brier_decomposition(y, p, n_bins=10)
    # A bin mean of identical floats is not bit-identical to them, so the extra
    # terms are rounding noise rather than exactly zero.
    assert d.within_bin_variance == pytest.approx(0.0, abs=1e-20)
    assert d.within_bin_covariance == pytest.approx(0.0, abs=1e-20)
    assert d.three_term == pytest.approx(d.brier, abs=1e-14)


def _continuous(signal_within_bins: bool) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(4)
    p = rng.beta(2, 3, size=20_000)
    if signal_within_bins:
        target = p
    else:
        # Outcomes depend only on which of five bins the forecast is in.
        target = (np.floor(p * 5) + 0.5) / 5
    y = (rng.uniform(size=p.size) < target).astype(int)
    return y, p


def test_three_term_identity_overstates_when_within_bin_variation_is_signal() -> None:
    # The naive expectation is that REL - RES + UNC equals the Brier score. The gap
    # is WBV - 2 * WBC. When outcomes track the forecast inside a bin, WBC ~ WBV
    # and the three-term value is too high by about WBV.
    d = brier_decomposition(*_continuous(signal_within_bins=True), n_bins=5)
    gap = d.brier - d.three_term
    assert gap < 0
    assert abs(gap) / d.brier > 0.01
    assert gap == pytest.approx(-d.within_bin_variance, rel=0.25)
    assert d.reconstructed == pytest.approx(d.brier, abs=1e-12)


def test_three_term_identity_understates_when_within_bin_variation_is_noise() -> None:
    # When within-bin spread carries no information, WBC ~ 0 and the three-term
    # value is too low by about WBV.
    d = brier_decomposition(*_continuous(signal_within_bins=False), n_bins=5)
    gap = d.brier - d.three_term
    assert gap > 0
    assert abs(gap) / d.brier > 0.01
    assert gap == pytest.approx(d.within_bin_variance, rel=0.25)


def test_empty_bins_are_nan_and_ignored() -> None:
    t = reliability_table([0, 1, 1], [0.1, 0.9, 0.95], n_bins=10)
    assert t.count.sum() == 3
    assert np.isnan(t.mean_confidence[5])
    assert t.n_bins == 10
    assert np.isfinite(expected_calibration_error([0, 1, 1], [0.1, 0.9, 0.95], 10))


def test_rejects_unknown_norm() -> None:
    with pytest.raises(ValueError, match="norm"):
        expected_calibration_error([0, 1], [0.2, 0.8], norm="l3")  # type: ignore[arg-type]


# ---- intervals ----------------------------------------------------------------


@given(st.integers(0, 400), st.integers(1, 400), st.sampled_from([0.8, 0.9, 0.95, 0.99]))
def test_wilson_matches_scipy(k: int, n: int, confidence: float) -> None:
    from scipy.stats import binomtest

    k = min(k, n)
    lo, hi = wilson_interval(k, n, confidence)
    ref = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    assert float(lo) == pytest.approx(ref.low, abs=1e-12)
    assert float(hi) == pytest.approx(ref.high, abs=1e-12)


def test_wilson_nonzero_width_at_extremes_and_nan_when_empty() -> None:
    lo, hi = wilson_interval([0, 10, 0], [10, 10, 0])
    assert lo[0] == 0.0 and hi[0] > 0.2
    assert hi[1] == 1.0 and lo[1] < 0.8
    assert np.isnan(lo[2]) and np.isnan(hi[2])


def test_wilson_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(1, 2, 1.0)
    with pytest.raises(ValueError, match="between"):
        wilson_interval(3, 2)


def test_table_intervals_cover_observed_frequency() -> None:
    y, p = _sample()
    t = reliability_table(y, p, 10)
    lo, hi = t.frequency_interval()
    m = t.count > 0
    assert np.all(lo[m] <= t.observed_frequency[m])
    assert np.all(t.observed_frequency[m] <= hi[m])


def test_wilson_coverage_by_simulation() -> None:
    # 95% Wilson intervals should cover the true rate about 95% of the time.
    rng = np.random.default_rng(8)
    n, mu, reps = 60, 0.2, 20_000
    k = (rng.random((reps, n)) < mu).sum(axis=1)
    lo, hi = wilson_interval(k, np.full(reps, n))
    coverage = float(np.mean((lo <= mu) & (mu <= hi)))
    assert 0.93 < coverage < 0.97
