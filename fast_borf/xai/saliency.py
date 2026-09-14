import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def add_window_importance(
    S,
    F,
    columns,
    word_offsets,
    observed,
    observed_offsets,
    positions,
    n_signals,
    count_overlapping,
):
    """Add each window's feature importance to the points the window covers.

    S has shape (n_series, n_signals, n_timestamps) and F (n_series,
    n_features). columns[w] is the feature of window w's word, or -1 when the
    word is not a feature. Windows, offsets and observed positions are laid out
    as returned by core.transform.panel_words, and positions[k] are the
    (observed-point) indices covered by the k-th window of a signal. With
    count_overlapping=False, a point gets each feature's importance once,
    however many of that feature's windows cover it.
    """
    for g in nb.prange(len(word_offsets) - 1):
        i = g // n_signals
        s = g % n_signals
        cols = columns[word_offsets[g] : word_offsets[g + 1]]
        obs = observed[observed_offsets[g] : observed_offsets[g + 1]]
        if count_overlapping:
            order = np.arange(cols.size)
        else:
            # Windows of the same feature are adjacent, so last[p] == col
            # means the point already got this feature's importance.
            order = np.argsort(cols, kind="mergesort")
        last = np.full(obs.size, -1, dtype=np.int64)
        for k in order:
            col = cols[k]
            if col < 0:
                continue
            importance = F[i, col]
            for j in range(positions.shape[1]):
                for q in range(positions.shape[2]):
                    p = positions[k, j, q]
                    if count_overlapping or last[p] != col:
                        last[p] = col
                        S[i, s, obs[p]] += importance
    return S


@nb.njit(parallel=True, cache=True)
def feature_coverage(
    columns, word_offsets, observed_offsets, positions, n_signals, count_overlapping
):
    """Number of points each feature covers in each series.

    Returns (series, feature, coverage) rows for the features whose word
    occurs, with inputs laid out as in add_window_importance. With
    count_overlapping=True a point counts once per window covering it, so the
    coverage is the number of windows times the points per window; otherwise
    each covered point counts once.
    """
    n_groups = len(word_offsets) - 1
    points_per_window = positions.shape[1] * positions.shape[2]
    # A signal has at most as many distinct features as windows.
    bound = np.zeros(n_groups + 1, dtype=np.int64)
    bound[1:] = np.cumsum(word_offsets[1:] - word_offsets[:-1])
    feature_buffer = np.empty(bound[-1], dtype=np.int64)
    coverage_buffer = np.empty(bound[-1], dtype=np.int64)
    n_features = np.zeros(n_groups + 1, dtype=np.int64)
    for g in nb.prange(n_groups):
        cols = columns[word_offsets[g] : word_offsets[g + 1]]
        seen = np.full(
            observed_offsets[g + 1] - observed_offsets[g], -1, dtype=np.int64
        )
        n = 0
        current = -1
        for k in np.argsort(cols, kind="mergesort"):
            col = cols[k]
            if col < 0:
                continue
            if col != current:
                current = col
                feature_buffer[bound[g] + n] = col
                coverage_buffer[bound[g] + n] = 0
                n += 1
            if count_overlapping:
                coverage_buffer[bound[g] + n - 1] += points_per_window
            else:
                for j in range(positions.shape[1]):
                    for q in range(positions.shape[2]):
                        p = positions[k, j, q]
                        if seen[p] != col:
                            seen[p] = col
                            coverage_buffer[bound[g] + n - 1] += 1
        n_features[g + 1] = n
    offsets = np.cumsum(n_features)
    series = np.empty(offsets[-1], dtype=np.int64)
    features = np.empty(offsets[-1], dtype=np.int64)
    coverage = np.empty(offsets[-1], dtype=np.int64)
    for g in nb.prange(n_groups):
        n = offsets[g + 1] - offsets[g]
        series[offsets[g] : offsets[g + 1]] = g // n_signals
        features[offsets[g] : offsets[g + 1]] = feature_buffer[bound[g] : bound[g] + n]
        coverage[offsets[g] : offsets[g + 1]] = coverage_buffer[bound[g] : bound[g] + n]
    return series, features, coverage
