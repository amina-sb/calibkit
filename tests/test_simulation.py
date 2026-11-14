from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad

from calibkit.simulation import bias_study, calibrated, overconfident, true_l2_error

BINS = (5, 15, 50, 100)
SEEDS = (11, 12, 13, 14, 15)


def test_true_error_matches_adaptive_quadrature() -> None:
    def integrand(x: float) -> float:
        p = np.array([x])
        return float((p - overconfident(p))[0] ** 2)

    ref, _ = quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-12, limit=200)
    assert true_l2_error(overconfident) == pytest.approx(np.sqrt(ref), rel=1e-6)
    assert true_l2_error(calibrated) == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_plug_in_bias_grows_with_bins_on_calibrated_forecasts(seed: int) -> None:
    # The naive expectation is that finer bins measure calibration more
    # precisely. For the plug-in estimator the opposite holds: on perfectly
    # calibrated forecasts its reading rises steadily with the bin count.
    rows = bias_study(calibrated, 1000, BINS, reps=150, seed=seed)
    means = [r.plug_in_l2.mean for r in rows]
    assert all(a < b for a, b in pairwise(means))
    assert rows[1].plug_in_l1.mean > 0.03  # 15 bins, n = 1000, true error 0
    assert rows[-1].plug_in_l2.mean > 4 * rows[0].plug_in_l2.mean


@pytest.mark.parametrize("seed", SEEDS)
def test_debiased_squared_estimate_centres_on_zero(seed: int) -> None:
    rows = bias_study(calibrated, 1000, BINS, reps=150, seed=seed)
    for r in rows:
        assert abs(r.debiased_sq.mean) < 4 * r.debiased_sq_se
        assert r.debiased_l2.mean < r.plug_in_l2.mean / 2.5


@pytest.mark.parametrize("seed", SEEDS)
def test_debiased_accuracy_is_insensitive_to_bin_count(seed: int) -> None:
    rows = bias_study(overconfident, 1000, BINS, reps=150, seed=seed)
    plug = [r.plug_in_l2.rmse for r in rows]
    deb = [r.debiased_l2.rmse for r in rows]
    plug_spread, deb_spread = max(plug) / min(plug), max(deb) / min(deb)
    assert plug_spread > 3.0
    assert deb_spread < plug_spread / 2.5
    # Debiasing removes bias, not variance: with about ten forecasts per bin at
    # 100 bins, the debiased estimate is noisier than at 5 bins.
    assert deb[-1] > deb[0]
    # At its best bin count the plug-in is competitive; the gain is robustness.
    assert min(plug) < 1.3 * min(deb)


def test_rows_report_their_configuration() -> None:
    rows = bias_study(overconfident, 300, (4, 8), reps=5, seed=0)
    assert [r.n_bins for r in rows] == [4, 8]
    assert all(r.n == 300 and r.truth == pytest.approx(0.1277, abs=1e-4) for r in rows)
    assert all(r.plug_in_l2.lo <= r.plug_in_l2.mean <= r.plug_in_l2.hi for r in rows)
