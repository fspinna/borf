import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def is_empty(a: NDArray) -> bool:
    return a.size == 0


@njit
def is_window_std_negligible(
    sequence_std: float, window_std: float, min_std_ratio: float = 0
) -> bool:
    if sequence_std == 0:
        return True
    if window_std == 0:
        return True
    if np.isnan(sequence_std):
        return False
    if np.isnan(window_std):
        return False
    else:
        ratio = window_std / sequence_std
        if ratio < min_std_ratio:
            return True
        else:
            return False


@njit
def is_valid_segmentation(window_size: int, word_length: int) -> bool:
    if window_size >= word_length:
        return True
    else:
        return False


@njit
def is_valid_windowing(sequence_size: int, window_size: int, dilation: int) -> bool:
    if (
        sequence_size < window_size * dilation
    ):  # if window_size * dilation exceeds the length of the sequence
        return False
    else:
        return True
