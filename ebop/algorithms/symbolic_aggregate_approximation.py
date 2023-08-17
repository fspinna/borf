import numpy as np
from numba import njit
from numpy.typing import NDArray
from ebop.algorithms.piecewise_aggregate_approximation import paa


@njit
def paa_sax(sequence: NDArray, bins: NDArray, word_length: int) -> NDArray:
    if sequence.size == word_length:
        return digitize(sequence=sequence, bins=bins)
    else:
        return digitize(paa(sequence=sequence, word_length=word_length), bins=bins)


@njit
def digitize(sequence: NDArray, bins: NDArray) -> NDArray:
    bins_nan = np.append(bins, np.nan)
    digitized_sequence = np.digitize(sequence, bins_nan)
    digitized_sequence[
        digitized_sequence == bins_nan.size
    ] = -1  # set values of the last bin to nan
    return digitized_sequence.astype(np.float_)
