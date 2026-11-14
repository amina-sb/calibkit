"""Probability calibration for binary classifiers."""

from .binning import assign_bins, bin_edges
from .metrics import (
    BrierDecomposition,
    ReliabilityTable,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_table,
)
from .recalibration import (
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
    "cross_calibrate",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_table",
]
