import numpy as np
import pytest
from numpy.typing import NDArray

from calibkit.intervals import calibration_error_interval, percentile_interval
from calibkit.metrics import brier_score, debiased_calibration_error, expected_calibration_error
from calibkit.recalibration import BaseRateCalibrator, cross_calibrate
from calibkit.simulation import (
    calibrated,
    overconfident,
    true_binned_sq_error,
    true_l2_error,
)

Arr = NDArray[np.float64]


def _draw(g: object, n: int, seed: int) -> tuple[Arr, Arr]:
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < g(p)).astype(np.float64)  # type: ignore[operator]
    return y, p


def test_percentile_interval_for_brier_brackets_the_estimate() -> None:
    y, p = _draw(overconfident, 2000, 0)
    lo, hi = percentile_interval(brier_score, y, p, resamples=400, seed=1)
    assert lo < brier_score(y, p) < hi
    assert hi - lo < 0.03


def test_percentile_interval_for_plug_in_ece_misses_its_own_estimate() -> None:
    # The naive expectation is that a bootstrap interval contains the point
    # estimate. For the plug-in ECE on calibrated data it does not: resampling
    # adds the estimator's upward bias a second time.
    y, p = _draw(calibrated, 3000, 3)
    est = expected_calibration_error(y, p, 15)
    lo, _ = percentile_interval(lambda a, b: expected_calibration_error(a, b, 15), y, p, 400, 0)
    assert lo > est


def test_calibration_error_interval_brackets_large_error() -> None:
    y, p = _draw(overconfident, 2000, 4)
    lo, hi = calibration_error_interval(y, p, 15, resamples=400, seed=0)
    est = debiased_calibration_error(y, p, 15)
    truth = true_binned_sq_error(overconfident, 15) ** 0.5
    assert lo < est < hi
    assert lo < truth < hi


def test_calibration_error_interval_does_not_collapse_on_a_baseline() -> None:
    # A single-bin forecast built from the labels: a reflected interval
    # collapses to (0, 0) here; the shift-corrected one must keep some width.
    y, p = _draw(calibrated, 3000, 42)
    base = cross_calibrate(BaseRateCalibrator, p, y)
    lo, hi = calibration_error_interval(y, base, 15, resamples=400, seed=0)
    assert lo == 0.0
    assert hi > 0.005


def test_binned_truth_never_exceeds_unbinned_truth() -> None:
    # Averaging within bins can only cancel miscalibration (Jensen).
    for b in (1, 5, 15, 100):
        assert true_binned_sq_error(overconfident, b) <= true_l2_error(overconfident) ** 2
    assert true_binned_sq_error(calibrated, 15) == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize("kwargs", [{"resamples": 1}, {"resamples": 2.5}, {"confidence": 1.0}])
def test_interval_arguments_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        calibration_error_interval([0, 1], [0.2, 0.8], **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        percentile_interval(brier_score, [0, 1], [0.2, 0.8], **kwargs)  # type: ignore[arg-type]


def test_intervals_are_reproducible() -> None:
    y, p = _draw(overconfident, 500, 9)
    a = calibration_error_interval(y, p, 10, resamples=100, seed=5)
    b = calibration_error_interval(y, p, 10, resamples=100, seed=5)
    assert a == b
