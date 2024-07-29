import numba as nb
import numpy as np
from fast_borf.piecewise_aggregate_approximation.piecewise_aggregate_approximation_dilated import paa


@nb.njit
def sax(
    a,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    return np.digitize(
        paa(
            a=a,
            window_size=window_size,
            word_length=word_length,
            min_std_ratio=min_window_to_signal_std_ratio,
            stride=stride,
            dilation=dilation,
        ),
        bins,
    ).astype(np.uint8)
