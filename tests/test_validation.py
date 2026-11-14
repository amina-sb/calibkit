import numpy as np
import pytest

from calibkit._validation import check_labels, check_pair, check_probabilities


def test_accepts_bool_and_int_labels() -> None:
    np.testing.assert_array_equal(check_labels([True, False, True]), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(check_labels(np.array([0, 1])), [0.0, 1.0])


def test_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match=r"only 0 and 1.*\[2\]"):
        check_labels([1, 2, 1])


@pytest.mark.parametrize(
    "p, msg",
    [
        ([0.5, 1.2], "lie in"),
        ([-0.1], "lie in"),
        ([np.nan], "NaN"),
        ([], "empty"),
        ([[0.5]], "1-D"),
    ],
)
def test_rejects_bad_probabilities(p: list[float], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        check_probabilities(p)


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        check_pair([0, 1], [0.5])
