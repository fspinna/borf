import numba as nb
import numpy as np

from fast_borf.core.moving import (
    weighted_moving_average_textbook,
    weighted_moving_standard_deviation_welford,
)
from fast_borf.utils import get_n_windows
from fast_borf.zscore import zscore_threshold


@nb.njit(cache=True)
def sax(
    arr,
    timestamps,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    n_windows = get_n_windows(
        sequence_size=arr.size,
        window_size=window_size,
        dilation=dilation,
        stride=stride,
    )
    n_windows_moving = get_n_windows(
        sequence_size=arr.size, window_size=window_size, dilation=dilation
    )
    global_std = np.std(arr)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    delta_t = np.zeros_like(timestamps)
    delta_t[1:] = np.diff(timestamps)
    delta_t[0] = delta_t[1]  # this is somewhat arbitrary
    # arr_mean = (arr[1:] + arr[:-1]) / 2
    seg_size = window_size // word_length
    n_windows = get_n_windows(
        sequence_size=arr.size,
        window_size=window_size,
        dilation=dilation,
        stride=stride,
    )
    n_segments = get_n_windows(
        sequence_size=arr.size, window_size=seg_size, dilation=dilation
    )
    segment_means = np.full(n_segments, np.nan)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = weighted_moving_average_textbook(
            arr[d::dilation], delta_t[d::dilation], window_size
        )[window_size - 1 :]
        window_stds[d::dilation] = weighted_moving_standard_deviation_welford(
            arr[d::dilation], delta_t[d::dilation], window_size
        )[window_size - 1 :]
        segment_means[d::dilation] = weighted_moving_average_textbook(
            arr[d::dilation], delta_t[d::dilation], seg_size
        )[seg_size - 1 :]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = zscore_threshold(
                a=segment_means[(i * stride) + (j * seg_size * dilation)],
                mu=window_means[i * stride],
                sigma=window_stds[i * stride],
                sigma_global=global_std,
                sigma_threshold=min_window_to_signal_std_ratio,
            )
    return np.digitize(out, bins).astype(np.uint8)
