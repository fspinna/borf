import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple


@njit(fastmath=True)
def window_sum(x: NDArray, w: int) -> Tuple[NDArray, float]:
    c = np.cumsum(x)
    s = c[w - 1 :]
    s[1:] -= c[:-w]
    return s, c[-1] / c.size


@njit(fastmath=True)
def linear_regression(segment: NDArray) -> Tuple[float, float]:
    if np.isnan(segment).all():
        return np.nan, np.nan
    not_nans = ~np.isnan(segment)
    if not_nans.size == 1:
        return 0, segment[not_nans].item()
    x = np.arange(segment.size)[not_nans]
    w = x.size
    y = segment[not_nans]
    sx, _ = window_sum(x, w)
    sy, avg = window_sum(y, w)
    sx2, _ = window_sum(x**2, w)
    sxy, _ = window_sum(x * y, w)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    return slope.item(), avg
