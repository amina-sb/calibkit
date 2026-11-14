import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from calibkit.binning import assign_bins, bin_edges

probs = arrays(
    np.float64,
    st.integers(1, 200),
    elements=st.floats(0.0, 1.0, allow_nan=False, allow_subnormal=False),
)


def test_uniform_edges() -> None:
    np.testing.assert_allclose(bin_edges([0.3], 4), [0.0, 0.25, 0.5, 0.75, 1.0])


def test_values_on_edges_go_to_the_lower_bin() -> None:
    edges = bin_edges([0.0], 4)
    idx = assign_bins([0.0, 0.25, 0.2500001, 0.5, 1.0], edges)
    assert idx.tolist() == [0, 0, 1, 1, 3]


def test_matches_sklearn_searchsorted_convention() -> None:
    # calibration_curve assigns with np.searchsorted(bins[1:-1], y_prob); mirror it.
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=500), np.linspace(0, 1, 11)])
    edges = bin_edges(p, 10)
    expected = np.searchsorted(np.linspace(0, 1, 11)[1:-1], p)
    np.testing.assert_array_equal(assign_bins(p, edges), expected)


def test_quantile_bins_have_roughly_equal_mass() -> None:
    p = np.random.default_rng(1).beta(2, 5, size=10_000)
    counts = np.bincount(assign_bins(p, bin_edges(p, 10, "quantile")))
    assert counts.size == 10
    assert np.min(counts) >= 990 and np.max(counts) <= 1010


def test_quantile_bins_collapse_on_ties() -> None:
    p = np.array([0.2] * 50 + [0.9] * 50)
    edges = bin_edges(p, 10, "quantile")
    assert np.all(np.diff(edges) > 0)
    assert edges.size < 11


@given(probs, st.integers(1, 30), st.sampled_from(["uniform", "quantile"]))
def test_every_forecast_lands_in_a_valid_bin(p: NDArray[np.float64], n: int, strategy: str) -> None:
    edges = bin_edges(p, n, strategy)  # type: ignore[arg-type]
    idx = assign_bins(p, edges)
    assert np.min(idx) >= 0 and np.max(idx) <= edges.size - 2
    lo, hi = edges[idx], edges[idx + 1]
    assert np.all(p <= hi)
    assert np.all((p > lo) | (idx == 0))


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_rejects_bad_bin_counts(bad: object) -> None:
    with pytest.raises(ValueError):
        bin_edges([0.5], bad)  # type: ignore[arg-type]


def test_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        bin_edges([0.5], 5, "kmeans")  # type: ignore[arg-type]


def test_rejects_unsorted_edges() -> None:
    with pytest.raises(ValueError, match="increasing"):
        assign_bins([0.5], [0.0, 0.6, 0.4, 1.0])
