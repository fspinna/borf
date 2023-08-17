import numpy as np
from numba import njit
from numpy._typing import NDArray
from ebop.utils.transform_utils import segment
from ebop.utils.condition_utils import is_valid_segmentation


@njit(fastmath=True)
def paa(sequence: NDArray, word_length: int) -> NDArray:
    if not is_valid_segmentation(window_size=sequence.size, word_length=word_length):
        return np.empty(0, dtype=np.float_)
    start, end = segment(window_size=sequence.size, word_length=word_length)
    return np.array(
        [np.nanmean(sequence[start[j] : end[j]]) for j in range(len(start))]
    )
