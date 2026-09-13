import numba as nb
import numpy as np

from fast_borf.bop_utils import ndindex_2d_array, sax_words_to_int
from fast_borf.core.sax import sax
from fast_borf.hash_unique import unique
from fast_borf.utils import (
    are_window_size_and_dilation_compatible_with_signal_length,
    get_norm_bins,
)


@nb.njit(cache=True)
def new_transform_single(
    a: np.ndarray,
    timestamps: np.ndarray,
    window_size,
    word_length,
    alphabet_size,
    bins,
    dilation,
    stride=1,
    min_window_to_signal_std_ratio=0.0,
):
    sax_words = sax(
        arr=a,
        timestamps=timestamps,
        window_size=window_size,
        word_length=word_length,
        bins=bins,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        dilation=dilation,
        stride=stride,
    )
    sax_words = sax_words_to_int(sax_words, alphabet_size)
    return unique(sax_words)


@nb.njit(cache=True)
def new_transform_single_conf(
    a: np.ndarray,
    timestamps: np.ndarray,
    ts_idx,
    signal_idx,
    window_size,
    word_length,
    alphabet_size,
    bins,
    dilation,
    stride=1,
    min_window_to_signal_std_ratio=0.0,
):
    words, counts = new_transform_single(
        a=a,
        timestamps=timestamps,
        window_size=window_size,
        word_length=word_length,
        alphabet_size=alphabet_size,
        bins=bins,
        dilation=dilation,
        stride=stride,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
    )
    ts_idxs = np.full(len(words), ts_idx)
    signal_idxs = np.full(len(words), signal_idx)
    return np.column_stack((ts_idxs, signal_idxs, words, counts))


@nb.njit(cache=True)
def get_signal(panel, panel_timestamps, ts_idx, signal_idx):
    """Return one signal and its timestamps, without missing values.

    When panel_timestamps is None, observations are evenly spaced.
    """
    signal = np.asarray(panel[ts_idx][signal_idx])
    if panel_timestamps is None:
        timestamps = np.arange(signal.size, dtype=np.float64)
    else:
        timestamps = np.asarray(panel_timestamps[ts_idx][0])
    is_observed = ~np.isnan(signal)
    return signal[is_observed], timestamps[is_observed]


@nb.njit(parallel=True, nogil=True, cache=True)
def transform_sax_patterns(
    panel,  # shape (n_ts, n_signals, n_obs)
    panel_timestamps,  # shape (n_ts, 1, n_obs), or None for evenly spaced data
    window_size,
    word_length,
    alphabet_size,
    stride,
    dilation,
    min_window_to_signal_std_ratio=0.0,
):
    bins = get_norm_bins(alphabet_size=alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals
    counts = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        if not are_window_size_and_dilation_compatible_with_signal_length(
            window_size, dilation, signal.size
        ):
            continue
        counts[i + 1] = len(
            new_transform_single_conf(
                a=signal,
                timestamps=signal_timestamps,
                ts_idx=ts_idx,
                signal_idx=signal_idx,
                window_size=window_size,
                word_length=word_length,
                alphabet_size=alphabet_size,
                bins=bins,
                dilation=dilation,
                stride=stride,
                min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            )
        )
    cum_counts = np.cumsum(counts)
    n_rows = np.sum(counts)
    shape = (n_rows, 4)
    out = np.empty(shape, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        if not are_window_size_and_dilation_compatible_with_signal_length(
            window_size, dilation, signal.size
        ):
            continue
        out_ = new_transform_single_conf(
            a=signal,
            timestamps=signal_timestamps,
            ts_idx=ts_idx,
            signal_idx=signal_idx,
            window_size=window_size,
            word_length=word_length,
            alphabet_size=alphabet_size,
            bins=bins,
            dilation=dilation,
            stride=stride,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )
        out[cum_counts[i] : cum_counts[i + 1], :] = out_
    return out
