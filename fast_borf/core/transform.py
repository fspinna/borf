"""SAX words and word counts for a panel of series, in parallel over signals.

A panel is indexed as panel[series][signal] and gives a 1D array per signal,
with NaN for missing values (a NaN-padded 3D array or a ragged awkward array).
Timestamps are indexed as panel_timestamps[series][0], or are None for evenly
spaced observations.
"""

import numba as nb
import numpy as np

from fast_borf.core.sax import breakpoints, get_n_windows, sax, window_fits
from fast_borf.core.unique import unique
from fast_borf.core.words import encode_words


@nb.njit(cache=True)
def ndindex_2d_array(idx, dim2_shape):
    row_idx = idx // dim2_shape
    col_idx = idx % dim2_shape
    return row_idx, col_idx


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


@nb.njit(cache=True)
def signal_words(
    signal,
    timestamps,
    window_size,
    word_length,
    alphabet_size,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    """The SAX word of every window of one signal, as integers."""
    symbols = sax(
        signal=signal,
        timestamps=timestamps,
        window_size=window_size,
        word_length=word_length,
        bins=bins,
        stride=stride,
        dilation=dilation,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
    )
    return encode_words(symbols, alphabet_size)


@nb.njit(cache=True)
def count_signal_words(
    signal,
    timestamps,
    ts_idx,
    signal_idx,
    window_size,
    word_length,
    alphabet_size,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    """Rows of (series index, signal index, word, count) for one signal."""
    words, counts = unique(
        signal_words(
            signal,
            timestamps,
            window_size,
            word_length,
            alphabet_size,
            bins,
            stride,
            dilation,
            min_window_to_signal_std_ratio,
        )
    )
    ts_idxs = np.full(len(words), ts_idx)
    signal_idxs = np.full(len(words), signal_idx)
    return np.column_stack((ts_idxs, signal_idxs, words, counts))


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
    """Count the SAX words of every signal.

    Returns rows of (series index, signal index, word, count), one per word
    occurring in a signal.
    """
    bins = breakpoints(alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals
    # First pass counts the rows of each signal, second pass fills them in.
    counts = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        if not window_fits(window_size, dilation, signal.size):
            continue
        counts[i + 1] = len(
            count_signal_words(
                signal,
                signal_timestamps,
                ts_idx,
                signal_idx,
                window_size,
                word_length,
                alphabet_size,
                bins,
                stride,
                dilation,
                min_window_to_signal_std_ratio,
            )
        )
    cum_counts = np.cumsum(counts)
    out = np.empty((cum_counts[-1], 4), dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        if not window_fits(window_size, dilation, signal.size):
            continue
        out[cum_counts[i] : cum_counts[i + 1], :] = count_signal_words(
            signal,
            signal_timestamps,
            ts_idx,
            signal_idx,
            window_size,
            word_length,
            alphabet_size,
            bins,
            stride,
            dilation,
            min_window_to_signal_std_ratio,
        )
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def panel_words(
    panel,
    panel_timestamps,
    window_size,
    word_length,
    alphabet_size,
    stride,
    dilation,
    min_window_to_signal_std_ratio=0.0,
):
    """The SAX word of every window of every signal.

    Returns (words, word_offsets, observed, observed_offsets). Signals are
    numbered g = series_index * n_signals + signal_index. Signal g's window
    words are words[word_offsets[g] : word_offsets[g + 1]], and
    observed[observed_offsets[g] : observed_offsets[g + 1]] are the positions
    of its non-NaN points in the series, which map the indices of
    core.sax.window_positions back to the series.
    """
    bins = breakpoints(alphabet_size)
    n_signals = len(panel[0])
    iterations = len(panel) * n_signals
    n_windows = np.zeros(iterations + 1, dtype=np.int64)
    n_observed = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        size = np.sum(~np.isnan(np.asarray(panel[ts_idx][signal_idx])))
        n_observed[i + 1] = size
        if window_fits(window_size, dilation, size):
            n_windows[i + 1] = get_n_windows(size, window_size, dilation, stride)
    word_offsets = np.cumsum(n_windows)
    observed_offsets = np.cumsum(n_observed)
    words = np.empty(word_offsets[-1], dtype=np.int64)
    observed = np.empty(observed_offsets[-1], dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        raw = np.asarray(panel[ts_idx][signal_idx])
        observed[observed_offsets[i] : observed_offsets[i + 1]] = np.nonzero(
            ~np.isnan(raw)
        )[0]
        if n_windows[i + 1] == 0:
            continue
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        words[word_offsets[i] : word_offsets[i + 1]] = signal_words(
            signal,
            signal_timestamps,
            window_size,
            word_length,
            alphabet_size,
            bins,
            stride,
            dilation,
            min_window_to_signal_std_ratio,
        )
    return words, word_offsets, observed, observed_offsets
