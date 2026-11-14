"""Simulation of calibration-error estimators against a known truth.

Forecasts are uniform on [0, 1], and outcomes are Bernoulli draws from a known
true-probability map ``g``. The true L2 calibration error is then

    CE = sqrt( integral_0^1 (p - g(p))^2 dp ),

computed here by quadrature, so every estimator can be scored against an exact
target rather than against another estimator.

All randomness comes from ``Generator.random``, whose output stream is stable
across numpy releases, so the published tables reproduce on any supported
version.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .metrics import debiased_calibration_error, expected_calibration_error
from .recalibration import expit, logit

FloatArray = NDArray[np.float64]
TruthMap = Callable[[FloatArray], FloatArray]


def calibrated(p: FloatArray) -> FloatArray:
    """Perfect calibration: the outcome probability equals the forecast."""
    return p


def overconfident(p: FloatArray) -> FloatArray:
    """True probability ``expit(0.5 * logit(p) + 0.3)``: forecasts too extreme."""
    return expit(0.5 * logit(p) + 0.3)


def true_l2_error(g: TruthMap, grid_size: int = 200_001) -> float:
    """Unbinned L2 calibration error of uniform forecasts under ``g``."""
    p = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    return float(np.sqrt(np.mean((p - g(p)) ** 2)))


@dataclass(frozen=True)
class EstimatorSummary:
    """Distribution of one estimator over replications."""

    mean: float
    lo: float
    hi: float
    rmse: float


@dataclass(frozen=True)
class StudyRow:
    n: int
    n_bins: int
    truth: float
    plug_in_l1: EstimatorSummary
    plug_in_l2: EstimatorSummary
    debiased_l2: EstimatorSummary
    #: The unclipped squared estimate, which is the unbiased one, against truth**2.
    debiased_sq: EstimatorSummary
    #: Standard error of ``debiased_sq.mean`` across replications.
    debiased_sq_se: float


def _summarise(values: FloatArray, truth: float) -> EstimatorSummary:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return EstimatorSummary(
        mean=float(np.mean(values)),
        lo=float(lo),
        hi=float(hi),
        rmse=float(np.sqrt(np.mean((values - truth) ** 2))),
    )


def bias_study(
    g: TruthMap,
    n: int,
    bin_counts: Sequence[int],
    reps: int,
    seed: int,
) -> list[StudyRow]:
    """Replicate each estimator ``reps`` times at every bin count.

    Each replication draws one dataset and scores it at every bin count, so the
    rows of a study share their datasets and differences between rows are not
    sampling noise between datasets.
    """
    rng = np.random.default_rng(seed)
    truth = true_l2_error(g)
    results = np.empty((4, len(bin_counts), reps))
    for r in range(reps):
        p = rng.random(n)
        y = (rng.random(n) < g(p)).astype(np.float64)
        for j, b in enumerate(bin_counts):
            results[0, j, r] = expected_calibration_error(y, p, b, norm="l1")
            results[1, j, r] = expected_calibration_error(y, p, b, norm="l2")
            sq = debiased_calibration_error(y, p, b, squared=True)
            results[2, j, r] = np.sqrt(max(sq, 0.0))
            results[3, j, r] = sq
    return [
        StudyRow(
            n=n,
            n_bins=b,
            truth=truth,
            plug_in_l1=_summarise(results[0, j], truth),
            plug_in_l2=_summarise(results[1, j], truth),
            debiased_l2=_summarise(results[2, j], truth),
            debiased_sq=_summarise(results[3, j], truth**2),
            debiased_sq_se=float(np.std(results[3, j], ddof=1) / np.sqrt(reps)),
        )
        for j, b in enumerate(bin_counts)
    ]
