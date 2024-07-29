import numba as nb
import numpy as np
from typing import Tuple, Optional


# @nb.njit
def is_valid_segmentation(window_size: int, word_length: int) -> bool:
    if window_size >= word_length:
        return True
    else:
        return False


# @nb.njit
def segment(window_size: int, word_length: int) -> Tuple[np.ndarray, np.ndarray]:
    assert is_valid_segmentation(window_size=window_size, word_length=word_length)
    bounds = np.linspace(0, window_size, word_length + 1).astype(np.int64)
    return bounds[:-1], bounds[1:]


# @nb.njit(fastmath=True)
def zscore(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(array)
    return (array - mean) / std


# @nb.njit(fastmath=True)
def zscore_inverse(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (array * std) + mean


# @nb.njit
def zscore_transform(
    transformed_array: np.ndarray, transforming_array: Optional[np.array] = None
) -> np.ndarray:
    if transforming_array is None:
        transforming_array = transformed_array
    if np.all(transforming_array == transformed_array[0]):  # if all elements are equal
        return zscore(
            transformed_array,
            mean=transformed_array[0],
            std=0,
        )
    return zscore(
        transformed_array,
        mean=np.nanmean(transforming_array),
        std=np.nanstd(transforming_array),
    )


# @nb.njit
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


# @nb.njit(fastmath=True)
def _paa_single(sequence: np.ndarray, word_length: int) -> np.ndarray:
    if not is_valid_segmentation(window_size=sequence.size, word_length=word_length):
        return np.empty(0, dtype=np.float_)
    start, end = segment(window_size=sequence.size, word_length=word_length)
    return np.array(
        [np.nanmean(sequence[start[j] : end[j]]) for j in range(len(start))]
    )


# @nb.njit
def normalize(
    window,
    signal_std: float,
    min_window_to_signal_std_ratio: float = 0,
):
    window_std = np.nanstd(window)
    if is_window_std_negligible(
        sequence_std=signal_std,
        window_std=window_std,
        min_std_ratio=min_window_to_signal_std_ratio,
    ):
        window = np.repeat(0.0, window.size)
    else:
        window = zscore_transform(transformed_array=window, transforming_array=None)
    return window



def _paa(a, window_size, word_length, dilation, stride, min_window_to_signal_std_ratio=0.0):
    step = (window_size - 1) * dilation + 1
    signal_std = np.nanstd(a)
    for i in np.arange(
        start=0,
        stop=a.size
             - window_size
             - ((window_size - 1) * (dilation - 1))
             + 1,
        step=stride):
        start = i
        end = start + step
        window_idx = np.arange(
            start=start, stop=end, step=dilation, dtype=np.int_
        )
        window = a[window_idx]
        window = normalize(
            window=window,
            signal_std=signal_std,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )
        yield _paa_single(sequence=window, word_length=word_length)


def paa(a, window_size, word_length, dilation, stride, min_std_ratio=0.0):
    return np.array(list(_paa(a, window_size, word_length, dilation, stride, min_std_ratio)))


def digitize(sequence: np.ndarray, bins: np.ndarray) -> np.ndarray:
    bins_nan = np.append(bins, np.nan)
    digitized_sequence = np.digitize(sequence, bins_nan)
    digitized_sequence[
        digitized_sequence == bins_nan.size
    ] = -1  # set values of the last bin to nan
    return digitized_sequence.astype(np.float_)


def _sax(a, window_size, word_length, dilation, stride, bins, min_window_to_signal_std_ratio=0.0):
    step = (window_size - 1) * dilation + 1
    signal_std = np.nanstd(a)
    for i in np.arange(
        start=0,
        stop=a.size
             - window_size
             - ((window_size - 1) * (dilation - 1))
             + 1,
        step=stride):
        start = i
        end = start + step
        window_idx = np.arange(
            start=start, stop=end, step=dilation, dtype=np.int_
        )
        window = a[window_idx]
        window = normalize(
            window=window,
            signal_std=signal_std,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )
        yield digitize(_paa_single(sequence=window, word_length=word_length), bins)


def sax(a, window_size, word_length, dilation, stride, bins, min_std_ratio=0.0):
    return np.array(list(_sax(a, window_size, word_length, dilation, stride, bins, min_std_ratio)))


if __name__ == "__main__":
    a = np.random.randn(1000)
    a = np.arange(18)
    # out = paa(a, 100, 10)
    # out2 = paa(a, 100, 10, dilation=2)
    out3 = paa(a, 6, 3, stride=2, dilation=3)
    for i in out3:
        print(i)

    # b = paa_gu(a, 100, 10)
        