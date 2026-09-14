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
