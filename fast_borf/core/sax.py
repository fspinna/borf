"""SAX for one signal: segment means (the PAA step), then one symbol per segment.

Every window of window_size observations (taken every dilation-th point, one
window every stride points) is split into word_length segments. Each segment
mean is z-normalized with the window's mean and standard deviation and turned
into a symbol with the Gaussian breakpoints of the alphabet. Observations are
weighted by the time elapsed since the previous one, so irregular sampling is
handled; with evenly spaced timestamps this is the usual unweighted SAX.

Indices in these functions refer to the observed (non-NaN) points of a signal.
"""

import math

import numba as nb
import numpy as np
from numba import vectorize

from fast_borf.core.moving import (
    FASTMATH,
    weighted_moving_average_textbook,
    weighted_moving_standard_deviation_welford,
)


@nb.njit(fastmath=True, cache=True)
def erfinv(x: float) -> float:
    w = -math.log((1 - x) * (1 + x))
    if w < 5:
        w = w - 2.5
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * w
        p = -3.5233877e-06 + p * w
        p = -4.39150654e-06 + p * w
        p = 0.00021858087 + p * w
        p = -0.00125372503 + p * w
        p = -0.00417768164 + p * w
        p = 0.246640727 + p * w
        p = 1.50140941 + p * w
    else:
        w = math.sqrt(w) - 3
        p = -0.000200214257
        p = 0.000100950558 + p * w
        p = 0.00134934322 + p * w
        p = -0.00367342844 + p * w
        p = 0.00573950773 + p * w
        p = -0.0076224613 + p * w
        p = 0.00943887047 + p * w
        p = 1.00167406 + p * w
        p = 2.83297682 + p * w
    return p * x


@vectorize(cache=True)
def ppf(x: np.ndarray, mu=0, std=1) -> np.ndarray:
    return mu + math.sqrt(2) * erfinv(2 * x - 1) * std


@nb.njit(cache=True)
def breakpoints(alphabet_size: int) -> np.ndarray:
    """Quantiles of the standard normal that split it into equally likely symbols."""
    return ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1], 0, 1)


@nb.njit(fastmath=True, cache=True)
def get_n_windows(sequence_size, window_size, dilation=1, stride=1, padding=0):
    return 1 + math.floor(
        (sequence_size + 2 * padding - window_size - (dilation - 1) * (window_size - 1))
        / stride
    )


@nb.njit(fastmath=True, cache=True)
def window_fits(window_size, dilation, signal_length):
    """Whether a dilated window fits in a signal of this length."""
    return window_size + (window_size - 1) * (dilation - 1) <= signal_length


@nb.njit(fastmath=FASTMATH, cache=True)
def zscore(a: float, mu: float, sigma: float) -> float:
    if sigma == 0:
        return 0
    return (a - mu) / sigma


@nb.njit(fastmath=FASTMATH, cache=True)
def zscore_threshold(
    a: float, mu: float, sigma: float, sigma_global: float, sigma_threshold: float
) -> float:
    if sigma_global == 0:
        return 0
    if sigma / sigma_global < sigma_threshold:
        return 0
    return zscore(a=a, mu=mu, sigma=sigma)


@nb.njit(cache=True)
def segment_means(
    signal,
    timestamps,
    window_size,
    word_length,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    """Z-normalized mean of every segment of every window (the PAA step).

    Returns an array of shape (n_windows, word_length). Windows whose standard
    deviation is below min_window_to_signal_std_ratio times the signal's get 0.
    """
    n_windows = get_n_windows(
        sequence_size=signal.size,
        window_size=window_size,
        dilation=dilation,
        stride=stride,
    )
    n_windows_moving = get_n_windows(
        sequence_size=signal.size, window_size=window_size, dilation=dilation
    )
    global_std = np.std(signal)
    if global_std == 0:
        return np.zeros((n_windows, word_length))
    # Each observation is weighted by the time since the previous one.
    delta_t = np.zeros_like(timestamps)
    delta_t[1:] = np.diff(timestamps)
    delta_t[0] = delta_t[1]  # this is somewhat arbitrary
    seg_size = window_size // word_length
    n_segments = get_n_windows(
        sequence_size=signal.size, window_size=seg_size, dilation=dilation
    )
    means = np.full(n_segments, np.nan)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = weighted_moving_average_textbook(
            signal[d::dilation], delta_t[d::dilation], window_size
        )[window_size - 1 :]
        window_stds[d::dilation] = weighted_moving_standard_deviation_welford(
            signal[d::dilation], delta_t[d::dilation], window_size
        )[window_size - 1 :]
        means[d::dilation] = weighted_moving_average_textbook(
            signal[d::dilation], delta_t[d::dilation], seg_size
        )[seg_size - 1 :]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = zscore_threshold(
                a=means[(i * stride) + (j * seg_size * dilation)],
                mu=window_means[i * stride],
                sigma=window_stds[i * stride],
                sigma_global=global_std,
                sigma_threshold=min_window_to_signal_std_ratio,
            )
    return out


@nb.njit(cache=True)
def discretize(values, bins):
    """Symbols 0 .. len(bins) for values, given breakpoints bins."""
    return np.digitize(values, bins).astype(np.uint8)


@nb.njit(cache=True)
def sax(
    signal,
    timestamps,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    """SAX symbols of every window, shape (n_windows, word_length).

    A constant signal gets symbol 0 everywhere, unlike a flat window inside a
    varying signal, whose z-score of 0 maps to the middle symbol.
    """
    if np.std(signal) == 0:
        n_windows = get_n_windows(signal.size, window_size, dilation, stride)
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    means = segment_means(
        signal,
        timestamps,
        window_size,
        word_length,
        stride,
        dilation,
        min_window_to_signal_std_ratio,
    )
    return discretize(means, bins)


@nb.njit(cache=True)
def window_positions(n_windows, window_size, word_length, stride=1, dilation=1):
    """Points covered by each segment of each window.

    Returns an array of shape (n_windows, word_length, window_size //
    word_length): out[i, j] are the (observed-point) indices averaged in
    segment j of window i.
    """
    seg_size = window_size // word_length
    out = np.empty((n_windows, word_length, seg_size), dtype=np.int64)
    for i in range(n_windows):
        for j in range(word_length):
            for k in range(seg_size):
                out[i, j, k] = i * stride + (j * seg_size + k) * dilation
    return out
