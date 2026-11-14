"""Probability calibration for binary classifiers."""

from .binning import assign_bins, bin_edges
from .intervals import calibration_error_interval, percentile_interval
from .metrics import (
    BrierDecomposition,
    ReliabilityTable,
    brier_decomposition,
    brier_score,
    debiased_calibration_error,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_table,
    wilson_interval,
)
from .recalibration import (
    BaseRateCalibrator,
    ConvergenceWarning,
    HistogramCalibrator,
    IsotonicCalibrator,
    NotFittedError,
    PlattCalibrator,
    Recalibrator,
    cross_calibrate,
)

__version__ = "0.1.0"

__all__ = [
    "BaseRateCalibrator",
    "BrierDecomposition",
    "ConvergenceWarning",
    "HistogramCalibrator",
    "IsotonicCalibrator",
    "NotFittedError",
    "PlattCalibrator",
    "Recalibrator",
    "ReliabilityTable",
    "__version__",
    "assign_bins",
    "bin_edges",
    "brier_decomposition",
    "brier_score",
    "calibration_error_interval",
    "cross_calibrate",
    "debiased_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "percentile_interval",
    "reliability_table",
    "wilson_interval",
]
