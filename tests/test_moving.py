import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from fast_borf.core.moving import (
    weighted_moving_average_textbook,
    weighted_moving_standard_deviation_welford,
)

LENGTH = 200


def naive_weighted_moving_average(arr, weights, window):
    out = np.full(arr.shape, np.nan)
    for i in range(window - 1, len(arr)):
        a, w = arr[i - window + 1 : i + 1], weights[i - window + 1 : i + 1]
        out[i] = np.sum(a * w) / np.sum(w)
    return out


def naive_weighted_moving_std(arr, weights, window):
    out = np.full(arr.shape, np.nan)
    for i in range(window - 1, len(arr)):
        a, w = arr[i - window + 1 : i + 1], weights[i - window + 1 : i + 1]
        mean = np.sum(a * w) / np.sum(w)
        out[i] = np.sqrt(np.sum(w * (a - mean) ** 2) / np.sum(w))
    return out


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return rng.standard_normal(LENGTH), rng.exponential(1.0, LENGTH)


@pytest.mark.parametrize("window", [1, 2, 7, 50, LENGTH])
def test_moving_average_matches_naive(series, window):
    arr, weights = series
    np.testing.assert_allclose(
        weighted_moving_average_textbook(arr, weights, window),
        naive_weighted_moving_average(arr, weights, window),
        equal_nan=True,
    )


@pytest.mark.parametrize("window", [2, 7, 50, LENGTH])
def test_moving_std_matches_naive(series, window):
    arr, weights = series
    np.testing.assert_allclose(
        weighted_moving_standard_deviation_welford(arr, weights, window),
        naive_weighted_moving_std(arr, weights, window),
        rtol=1e-6,
        equal_nan=True,
    )


@pytest.mark.parametrize("window", [2, 7, 50])
def test_uniform_weights_give_plain_moving_statistics(series, window):
    arr, _ = series
    weights = np.ones(LENGTH)
    windows = sliding_window_view(arr, window)
    np.testing.assert_allclose(
        weighted_moving_average_textbook(arr, weights, window)[window - 1 :],
        windows.mean(axis=1),
    )
    np.testing.assert_allclose(
        weighted_moving_standard_deviation_welford(arr, weights, window)[window - 1 :],
        windows.std(axis=1),
        rtol=1e-6,
    )
