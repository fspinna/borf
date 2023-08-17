import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple

from ebop.algorithms.least_squares_nan import linear_regression
from ebop.algorithms.symbolic_aggregate_approximation import digitize
from ebop.utils.condition_utils import is_valid_segmentation
from ebop.utils.transform_utils import segment


@njit
def oned_nan_sax(
    sequence: NDArray,
    word_length: int,
    bins_mean: NDArray,
    bins_slope: NDArray,
    scale_factor: float = 0.03,
) -> Tuple[NDArray, NDArray]:
    if not is_valid_segmentation(window_size=sequence.size, word_length=word_length):
        return np.empty(0, dtype=np.int_), np.empty(0, dtype=np.int_)
    if sequence.size == word_length:
        ms = np.zeros(sequence.size)
        cs = sequence
        ms = digitize(sequence=ms, bins=bins_slope * np.sqrt(scale_factor))
        cs = digitize(sequence=cs, bins=bins_mean)
        return ms.astype(np.int_), cs.astype(np.int_)
    start, end = segment(window_size=sequence.size, word_length=word_length)
    msd = np.empty(shape=start.size)
    cs = np.empty(shape=start.size)
    for j in range(len(start)):
        seg = sequence[start[j] : end[j]]
        m, c = linear_regression(seg)
        length = seg.size
        msd[j] = digitize(
            sequence=np.array([m]).astype(np.float_),
            bins=bins_slope * np.sqrt(scale_factor / length),
        ).item()
        cs[j] = c
    return msd.astype(np.int_), digitize(sequence=cs, bins=bins_mean).astype(np.int_)
