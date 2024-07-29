import numba as nb
import numpy as np
from fast_borf.piecewise_aggregate_approximation.piecewise_aggregate_approximation import paa, _paa_gu


@nb.njit
def sax(
    a,
    window_size,
    word_length,
    bins,
    min_window_to_signal_std_ratio=0.0,
):
    return np.digitize(
        paa(
            a=a,
            window_size=window_size,
            word_length=word_length,
            min_std_ratio=min_window_to_signal_std_ratio,
        ),
        bins,
    ).astype(np.uint8)


def sax_gu(a, window_size, word_length, bins, min_std_ratio=0):
    dum_ws = np.zeros(len(a) - window_size + 1)
    dum_wl = np.zeros(word_length)
    out = np.empty((dum_ws.size, dum_wl.size))
    return np.digitize(_paa_gu(a, window_size, word_length, min_std_ratio, dum_ws, dum_wl, out), bins).astype(np.uint8)

