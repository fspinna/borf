import numpy as np
import numba as nb


@nb.njit
def align_window_to_segments(window_idx, word_length, seg_size):
    return np.repeat(window_idx, word_length), window_idx + seg_size * np.arange(
        word_length
    )


@nb.njit
def align_window_to_segments_dilated(window_idx, word_length, seg_size, dilation):
    return (
        np.repeat(window_idx, word_length),
        window_idx + seg_size * np.arange(word_length) * dilation,
    )