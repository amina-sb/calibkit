"""Input validation shared by every public function."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_probabilities(y_prob: ArrayLike, name: str = "y_prob") -> NDArray[np.float64]:
    """Return ``y_prob`` as a 1-D float array, rejecting anything outside [0, 1]."""
    p = np.asarray(y_prob, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {p.shape}")
    if p.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.min(p) < 0.0 or np.max(p) > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got range [{np.min(p)}, {np.max(p)}]")
    return p


def check_labels(y_true: ArrayLike, name: str = "y_true") -> NDArray[np.float64]:
    """Return binary labels as a float array of 0.0 and 1.0.

    Booleans and the integers 0/1 are accepted. Anything else is an error rather
    than being coerced, because a label vector of {1, 2} silently treated as
    binary is a common way to produce a meaningless calibration score.
    """
    raw = np.asarray(y_true)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    y = raw.astype(np.float64)
    if not np.all((y == 0.0) | (y == 1.0)):
        bad = np.unique(raw[(y != 0.0) & (y != 1.0)])[:5]
        raise ValueError(f"{name} must contain only 0 and 1; found {bad.tolist()}")
    return y


def check_pair(
    y_true: ArrayLike, y_prob: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate labels and probabilities together and check their lengths agree."""
    y = check_labels(y_true)
    p = check_probabilities(y_prob)
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob differ in length: {y.size} vs {p.size}")
    return y, p
