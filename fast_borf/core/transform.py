"""SAX words and word counts for a panel of series, in parallel over signals.

A panel is indexed as panel[series][signal] and gives a 1D array per signal,
with NaN for missing values (a NaN-padded 3D array or a ragged awkward array).
Timestamps are indexed as panel_timestamps[series][0], or are None for evenly
spaced observations.
"""

import numba as nb
import numpy as np

from fast_borf.core.sax import breakpoints, get_n_windows, sax_words, window_fits
from fast_borf.core.unique import unique


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


@nb.njit(parallel=True, nogil=True, cache=True)
def count_windows(panel, window_size, stride, dilation):
    """Number of windows of every signal, and of its observed points.

    Signals are numbered series_index * n_signals + signal_index.
    """
    n_signals = len(panel[0])
    iterations = len(panel) * n_signals
    n_windows = np.zeros(iterations, dtype=np.int64)
    n_observed = np.zeros(iterations, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        size = np.sum(~np.isnan(np.asarray(panel[ts_idx][signal_idx])))
        n_observed[i] = size
        if window_fits(window_size, dilation, size):
            n_windows[i] = get_n_windows(size, window_size, dilation, stride)
    return n_windows, n_observed


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
    occurring in a signal, ordered by series, signal and word.
    """
    bins = breakpoints(alphabet_size)
    n_signals = len(panel[0])
    iterations = len(panel) * n_signals
    # A signal has at most as many distinct words as windows, so its unique
    # words and counts fit in buffers of that size; they are compacted after.
    n_windows, _ = count_windows(panel, window_size, stride, dilation)
    bound = np.zeros(iterations + 1, dtype=np.int64)
    bound[1:] = np.cumsum(n_windows)
    word_buffer = np.empty(bound[-1], dtype=np.int64)
    count_buffer = np.empty(bound[-1], dtype=np.int64)
    n_unique = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        if n_windows[i] == 0:
            continue
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        words, counts = unique(
            sax_words(
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
        )
        order = np.argsort(words)
        n_unique[i + 1] = len(words)
        word_buffer[bound[i] : bound[i] + len(words)] = words[order]
        count_buffer[bound[i] : bound[i] + len(words)] = counts[order]
    row_offsets = np.cumsum(n_unique)
    out = np.empty((row_offsets[-1], 4), dtype=np.int64)
    for i in nb.prange(iterations):
        start, stop = row_offsets[i], row_offsets[i + 1]
        if start == stop:
            continue
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        out[start:stop, 0] = ts_idx
        out[start:stop, 1] = signal_idx
        out[start:stop, 2] = word_buffer[bound[i] : bound[i] + stop - start]
        out[start:stop, 3] = count_buffer[bound[i] : bound[i] + stop - start]
    return out


@nb.njit(parallel=True, cache=True)
def sum_over_channel_groups(rows, n_series, n_signals, members, member_offsets):
    """Sum word counts over the channels of each group.

    rows are (series index, signal index, word, count), ordered by series,
    signal and word, as returned by transform_sax_patterns. Group k's
    channels are members[member_offsets[k] : member_offsets[k + 1]]. Returns
    rows of (series index, group index, word, summed count), ordered by
    series, group and word.
    """
    n_groups = len(member_offsets) - 1
    # starts[b] is the first row of signal b = series * n_signals + signal.
    starts = np.searchsorted(
        rows[:, 0] * n_signals + rows[:, 1], np.arange(n_series * n_signals + 1)
    )
    n_cells = n_series * n_groups
    # A group has at most as many distinct words as its channels together.
    bound = np.zeros(n_cells + 1, dtype=np.int64)
    for cell in nb.prange(n_cells):
        i, k = ndindex_2d_array(cell, n_groups)
        for m in range(member_offsets[k], member_offsets[k + 1]):
            b = i * n_signals + members[m]
            bound[cell + 1] += starts[b + 1] - starts[b]
    bound = np.cumsum(bound)
    word_buffer = np.empty(bound[-1], dtype=np.int64)
    count_buffer = np.empty(bound[-1], dtype=np.int64)
    n_unique = np.zeros(n_cells + 1, dtype=np.int64)
    for cell in nb.prange(n_cells):
        size = bound[cell + 1] - bound[cell]
        if size == 0:
            continue
        i, k = ndindex_2d_array(cell, n_groups)
        words = np.empty(size, dtype=np.int64)
        counts = np.empty(size, dtype=np.int64)
        filled = 0
        for m in range(member_offsets[k], member_offsets[k + 1]):
            b = i * n_signals + members[m]
            n = starts[b + 1] - starts[b]
            words[filled : filled + n] = rows[starts[b] : starts[b + 1], 2]
            counts[filled : filled + n] = rows[starts[b] : starts[b + 1], 3]
            filled += n
        order = np.argsort(words, kind="mergesort")
        start = bound[cell]
        n_out = 0
        for q in order:
            if n_out > 0 and word_buffer[start + n_out - 1] == words[q]:
                count_buffer[start + n_out - 1] += counts[q]
            else:
                word_buffer[start + n_out] = words[q]
                count_buffer[start + n_out] = counts[q]
                n_out += 1
        n_unique[cell + 1] = n_out
    row_offsets = np.cumsum(n_unique)
    out = np.empty((row_offsets[-1], 4), dtype=np.int64)
    for cell in nb.prange(n_cells):
        start, stop = row_offsets[cell], row_offsets[cell + 1]
        if start == stop:
            continue
        i, k = ndindex_2d_array(cell, n_groups)
        out[start:stop, 0] = i
        out[start:stop, 1] = k
        out[start:stop, 2] = word_buffer[bound[cell] : bound[cell] + stop - start]
        out[start:stop, 3] = count_buffer[bound[cell] : bound[cell] + stop - start]
    return out


@nb.njit(cache=True)
def entries_to_csr(rows, cols, values, n_rows):
    """CSR arrays (indptr, indices, data) of entries (rows[k], cols[k], values[k]).

    Entries keep their input order within each row, so entries whose columns
    increase within each row give sorted indices. There must be no duplicates.
    """
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    for row in rows:
        indptr[row + 1] += 1
    indptr = np.cumsum(indptr)
    next_position = indptr[:-1].copy()
    indices = np.empty(len(rows), dtype=cols.dtype)
    data = np.empty(len(rows), dtype=values.dtype)
    for k in range(len(rows)):
        position = next_position[rows[k]]
        indices[position] = cols[k]
        data[position] = values[k]
        next_position[rows[k]] += 1
    return indptr, indices, data


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
    n_windows, n_observed = count_windows(panel, window_size, stride, dilation)
    word_offsets = np.zeros(iterations + 1, dtype=np.int64)
    word_offsets[1:] = np.cumsum(n_windows)
    observed_offsets = np.zeros(iterations + 1, dtype=np.int64)
    observed_offsets[1:] = np.cumsum(n_observed)
    words = np.empty(word_offsets[-1], dtype=np.int64)
    observed = np.empty(observed_offsets[-1], dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        raw = np.asarray(panel[ts_idx][signal_idx])
        observed[observed_offsets[i] : observed_offsets[i + 1]] = np.nonzero(
            ~np.isnan(raw)
        )[0]
        if n_windows[i] == 0:
            continue
        signal, signal_timestamps = get_signal(
            panel, panel_timestamps, ts_idx, signal_idx
        )
        words[word_offsets[i] : word_offsets[i + 1]] = sax_words(
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
