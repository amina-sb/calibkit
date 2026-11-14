import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from calibkit.metrics import brier_score, expected_calibration_error
from calibkit.recalibration import (
    ConvergenceWarning,
    HistogramCalibrator,
    IsotonicCalibrator,
    NotFittedError,
    PlattCalibrator,
    Recalibrator,
    expit,
    logit,
    pool_adjacent_violators,
)

Arr = NDArray[np.float64]


def overconfident(n: int, seed: int) -> tuple[Arr, Arr]:
    """Scores that are too extreme: the true probability is a shrunk logit."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 0.6, size=n)
    truth = expit(0.5 * logit(p) + 0.3)
    y = (rng.uniform(size=n) < truth).astype(np.float64)
    return p, y


ALL = [
    PlattCalibrator,
    IsotonicCalibrator,
    lambda: HistogramCalibrator(n_bins=10),
]


@st.composite
def fit_data(draw: st.DrawFn) -> tuple[Arr, Arr]:
    n = draw(st.integers(2, 200))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    p = rng.uniform(size=n)
    if draw(st.booleans()):
        p = np.round(p, 2)
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return p, y


# ---- reference implementations ------------------------------------------------


def test_platt_matches_unpenalised_logistic_regression() -> None:
    p, y = overconfident(3000, seed=0)
    cal = PlattCalibrator(smooth_targets=False).fit(p, y)
    ref = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(logit(p)[:, None], y)
    assert cal.a_ == pytest.approx(ref.coef_[0, 0], abs=1e-6)
    assert cal.b_ == pytest.approx(ref.intercept_[0], abs=1e-6)
    np.testing.assert_allclose(
        cal.predict(p), ref.predict_proba(logit(p)[:, None])[:, 1], atol=1e-7
    )


def test_platt_recovers_generating_coefficients() -> None:
    p, y = overconfident(200_000, seed=1)
    cal = PlattCalibrator().fit(p, y)
    assert cal.a_ == pytest.approx(0.5, abs=0.02)
    assert cal.b_ == pytest.approx(0.3, abs=0.02)


def test_platt_never_reverses_the_ranking() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([1, 1, 0, 1, 0, 0], dtype=np.float64)
    with pytest.warns(UserWarning, match="anti-correlated"):
        cal = PlattCalibrator().fit(p, y)
    assert cal.a_ == 0.0
    t = (3 + 1) / (3 + 2), 1 / (3 + 2)
    assert cal.predict([0.5])[0] == pytest.approx((3 * t[0] + 3 * t[1]) / 6)
    unconstrained = PlattCalibrator(monotone=False).fit(p, y)
    assert unconstrained.a_ < 0


def test_platt_smoothing_keeps_separable_data_finite() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal = PlattCalibrator(smooth_targets=True).fit(p, y)
    assert np.isfinite(cal.a_) and 0 < cal.a_ < 20
    with pytest.warns(ConvergenceWarning):
        PlattCalibrator(smooth_targets=False, max_iter=25).fit(p, y)


@settings(max_examples=100)
@given(fit_data())
def test_isotonic_matches_sklearn(data: tuple[Arr, Arr]) -> None:
    p, y = data
    ours = IsotonicCalibrator().fit(p, y)
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    grid = np.linspace(0, 1, 101)
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), atol=1e-12)
    np.testing.assert_allclose(ours.predict(p), ref.predict(p), atol=1e-12)


def test_pav_worked_example() -> None:
    # 1, 3, 2, 4 -> the violating pair (3, 2) pools to 2.5.
    out = pool_adjacent_violators(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4))
    np.testing.assert_allclose(out, [1.0, 2.5, 2.5, 4.0])
    # A weighted pool: (4 with weight 3, 1 with weight 1) -> 13/4.
    out = pool_adjacent_violators(np.array([4.0, 1.0]), np.array([3.0, 1.0]))
    np.testing.assert_allclose(out, [3.25, 3.25])


# ---- properties ---------------------------------------------------------------


@pytest.mark.parametrize("make", ALL)
@settings(max_examples=60, deadline=None)
@given(data=fit_data())
def test_outputs_bounded_and_monotone(make: type[Recalibrator], data: tuple[Arr, Arr]) -> None:
    p, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = make().fit(p, y)
    grid = np.linspace(0, 1, 201)
    q = cal.predict(grid)
    assert np.all((q >= 0) & (q <= 1))
    if not isinstance(cal, HistogramCalibrator):
        assert np.all(np.diff(q) >= -1e-12)


@given(fit_data())
def test_isotonic_preserves_mean_of_labels(data: tuple[Arr, Arr]) -> None:
    p, y = data
    cal = IsotonicCalibrator().fit(p, y)
    assert float(np.mean(cal.predict(p))) == pytest.approx(float(np.mean(y)), abs=1e-12)


# ---- does it help -------------------------------------------------------------


@pytest.mark.parametrize("make", ALL)
def test_recalibration_improves_held_out_ece_without_hurting_brier(
    make: type[Recalibrator],
) -> None:
    p_fit, y_fit = overconfident(5000, seed=10)
    p_test, y_test = overconfident(5000, seed=11)
    q = make().fit(p_fit, y_fit).predict(p_test)
    before = expected_calibration_error(y_test, p_test, 15)
    after = expected_calibration_error(y_test, q, 15)
    assert before > 0.08
    assert after < before / 3
    assert brier_score(y_test, q) < brier_score(y_test, p_test)


def test_in_sample_histogram_ece_is_exactly_zero() -> None:
    # Evaluating a recalibrator on its own fitting data with its own bins reports
    # perfect calibration whatever the data. The held-out number is the real one.
    p_fit, y_fit = overconfident(2000, seed=20)
    p_test, y_test = overconfident(2000, seed=21)
    cal = HistogramCalibrator(n_bins=10).fit(p_fit, y_fit)
    in_sample = expected_calibration_error(y_fit, cal.predict(p_fit), 10)
    held_out = expected_calibration_error(y_test, cal.predict(p_test), 10)
    assert in_sample == pytest.approx(0.0, abs=1e-12)
    assert held_out > 0.005


def test_histogram_prior_shrinks_toward_midpoint() -> None:
    p = np.array([0.05, 0.06])
    y = np.array([1.0, 1.0])
    raw = HistogramCalibrator(n_bins=10).fit(p, y)
    shrunk = HistogramCalibrator(n_bins=10, prior_strength=2.0).fit(p, y)
    assert raw.predict([0.05])[0] == 1.0
    assert shrunk.predict([0.05])[0] == pytest.approx((2 + 2 * 0.05) / 4)
    assert raw.predict([0.55])[0] == pytest.approx(0.55)  # empty bin -> midpoint


# ---- interface ----------------------------------------------------------------


@pytest.mark.parametrize("make", ALL)
def test_predict_before_fit_raises(make: type[Recalibrator]) -> None:
    with pytest.raises(NotFittedError):
        make().predict([0.5])


@pytest.mark.parametrize("make", ALL)
def test_fit_validates_inputs(make: type[Recalibrator]) -> None:
    with pytest.raises(ValueError):
        make().fit([0.5, 1.5], [0, 1])
    with pytest.raises(ValueError):
        make().fit([0.5, 0.6], [0, 2])


def test_expit_is_stable_at_extremes() -> None:
    z = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
    out = expit(z)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[-1] == 1.0 and out[2] == 0.5


def test_negative_prior_rejected() -> None:
    with pytest.raises(ValueError):
        HistogramCalibrator(prior_strength=-1.0)
