import math
from typing import Optional, Tuple
import numpy as np
from numba import njit, vectorize
from numpy.typing import NDArray
from borf.utils.condition_utils import is_empty, is_valid_segmentation


@njit
def offset(array, mean):
    return array - mean


@njit
def offset_transform(
    transformed_array: NDArray, transforming_array: Optional[NDArray] = None
) -> NDArray:
    if transforming_array is None:
        transforming_array = transformed_array
    return offset(array=transformed_array, mean=np.nanmean(transforming_array))


@njit(fastmath=True)
def zscore(array: NDArray, mean: float, std: float) -> NDArray:
    if std == 0:
        return np.zeros_like(array)
    return (array - mean) / std


@njit(fastmath=True)
def zscore_inverse(array: NDArray, mean: float, std: float) -> NDArray:
    return (array * std) + mean


@njit
def zscore_transform(
    transformed_array: NDArray, transforming_array: Optional[np.array] = None
) -> NDArray:
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


@njit
def zscore_inverse_transform(
    transformed_array: NDArray, transforming_array: NDArray
) -> NDArray:
    return zscore_inverse(
        array=transformed_array,
        mean=np.nanmean(transforming_array),
        std=np.nanstd(transforming_array),
    )


@njit
def array_to_string(array: NDArray, sep=",") -> str:
    if is_empty(array):
        return ""
    str_array = str(array[0])
    for i in range(1, array.size):
        str_array += sep + str(array[i])
    return str_array


@njit
def arrays_to_string(
    a: NDArray, b: NDArray, sep_idx: str = ",", sep_value: str = "."
) -> str:
    if is_empty(a):
        return ""
    str_array = str(a[0]) + sep_value + str(b[0])
    for i in range(1, a.size):
        str_array += sep_idx + str(a[i]) + sep_value + str(b[i])
    return str_array


@njit(fastmath=True)
def erfinv(x: float) -> float:
    w = -math.log((1 - x) * (1 + x))
    if w < 5:
        w = w - 2.5
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * w
        p = -3.5233877e-06 + p * w
        p = -4.39150654e-06 + p * w
        p = 0.00021858087 + p * w
        p = -0.00125372503 + p * w
        p = -0.00417768164 + p * w
        p = 0.246640727 + p * w
        p = 1.50140941 + p * w
    else:
        w = math.sqrt(w) - 3
        p = -0.000200214257
        p = 0.000100950558 + p * w
        p = 0.00134934322 + p * w
        p = -0.00367342844 + p * w
        p = 0.00573950773 + p * w
        p = -0.0076224613 + p * w
        p = 0.00943887047 + p * w
        p = 1.00167406 + p * w
        p = 2.83297682 + p * w
    return p * x


@vectorize
def ppf(x: NDArray, mu=0, std=1) -> NDArray:
    return mu + math.sqrt(2) * erfinv(2 * x - 1) * std


@njit
def get_norm_bins(alphabet_size: int, mu=0, std=1) -> NDArray:
    return ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1], mu, std)


@njit
def pad(array: NDArray, left_pad: int, right_pad: int, mode: str = "edge") -> NDArray:
    if mode == "edge":
        return np.concatenate(
            (np.repeat(array[0], left_pad), array, np.repeat(array[-1], right_pad))
        )
    elif mode == "nans":
        return np.concatenate(
            (np.repeat(np.nan, left_pad), array, np.repeat(np.nan, right_pad))
        )
    else:
        raise NotImplementedError


@njit
def segment(window_size: int, word_length: int) -> Tuple[NDArray, NDArray]:
    assert is_valid_segmentation(window_size=window_size, word_length=word_length)
    bounds = np.linspace(0, window_size, word_length + 1).astype(np.int64)
    return bounds[:-1], bounds[1:]


@njit
def attach_metadata_to_key(
    key: str,
    prefix: str = "",
    signal_separator: str = ";",
    signal_idx: int = -1,
    use_signal_idx: bool = True,
) -> str:
    if use_signal_idx:
        return f"{prefix}{signal_separator}{str(signal_idx)}{signal_separator}{key}"
    else:
        return f"{prefix}{signal_separator}{'-1'}{signal_separator}{key}"


def average_groups(row_idxs, column_idxs, values):
    # Create a 2D array of [row_idxs, column_idxs] pairs
    pairs = np.empty((row_idxs.size, 2))
    pairs[:, 0] = row_idxs
    pairs[:, 1] = column_idxs

    # Sort the array by pairs
    idxs = np.lexsort((column_idxs, row_idxs))
    sorted_pairs = pairs[idxs]
    sorted_values = values[idxs]

    # Find the indices where the pair changes
    change_idxs = np.where(np.any(np.diff(sorted_pairs, axis=0), axis=1))[0]
    start_idxs = np.concatenate(([0], change_idxs + 1))
    end_idxs = np.concatenate((change_idxs + 1, [sorted_pairs.shape[0]]))

    # Compute the average for each group
    averages = np.array(
        [sorted_values[start:end].mean() for start, end in zip(start_idxs, end_idxs)]
    )

    # Get the unique row_idxs and column_idxs
    unique_row_idxs = sorted_pairs[start_idxs, 0]
    unique_column_idxs = sorted_pairs[start_idxs, 1]

    return unique_row_idxs, unique_column_idxs, averages


@njit
def create_dict(items):
    return {k: v for k, v in items}


def is_within_interval(number, start, end):
    return start <= number <= end


@njit
def extract_parameters_from_args(parameter):
    window_size = parameter[0]
    stride = parameter[1]
    dilation = parameter[2]
    word_length = parameter[3]
    alphabet_mean = parameter[4]
    alphabet_slope = parameter[5]
    return alphabet_mean, alphabet_slope, dilation, stride, window_size, word_length
