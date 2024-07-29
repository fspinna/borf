import numba as nb
import numpy as np
from numba import njit, vectorize, set_num_threads
import math
import awkward as ak
from typing import Union, Iterable
import psutil
from fast_borf.constants import HASHMAP_2_SYMBOLS, NORM_BINS_DICT


@njit(fastmath=True, cache=True)
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


@vectorize(cache=True)
def ppf(x: np.ndarray, mu=0, std=1) -> np.ndarray:
    return mu + math.sqrt(2) * erfinv(2 * x - 1) * std


@njit(cache=True)
def get_norm_bins(alphabet_size: int, mu=0, std=1) -> np.ndarray:
    return ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1], mu, std)


@njit
def get_cached_norm_bins(alphabet_size: int, bins=NORM_BINS_DICT) -> np.ndarray:
    return bins[alphabet_size - 2]


@njit(cache=True)
def is_empty(a: np.ndarray) -> bool:
    return a.size == 0


@njit(cache=True)
def create_dict(items):
    return {k: v for k, v in items}


@njit(cache=True)
def is_alphabet_size_valid(alphabet_size):
    if alphabet_size < 2 or alphabet_size > 9:
        return False
    else:
        return True


@njit(cache=True)
def are_alphabet_sizes_valid(alphabet_size_mean, alphabet_size_slope):
    return (
        is_alphabet_size_valid(alphabet_size_mean)
        and is_alphabet_size_valid(alphabet_size_slope)
        and alphabet_size_mean * alphabet_size_slope <= 9
    )


@njit(cache=True)
def is_window_size_divisible_by_word_length(window_size, word_length):
    return window_size % word_length == 0


@njit(cache=True)
def is_window_size_less_than_word_length(window_size, word_length):
    return window_size < word_length


@njit(fastmath=True, cache=True)
def are_window_size_and_dilation_compatible_with_signal_length(window_size, dilation, signal_length):
    if window_size + (window_size - 1) * (dilation - 1) <= signal_length:
        return True
    else:
        return False


@njit(cache=True)
def is_window_size_less_or_equal_than_signal_length(window_size, signal_length):
    return window_size <= signal_length


def check_window_size_word_length(window_size, word_length):
    if window_size % word_length != 0:
        raise ValueError(
            f"window_size ({window_size}) must be a multiple of word_length ({word_length})"
        )
    if window_size < word_length:
        raise ValueError(
            f"window_size ({window_size}) must be greater than word_length ({word_length})"
        )


def check_alphabet_size(alphabet_size):
    if not is_alphabet_size_valid(alphabet_size):
        raise ValueError(f"alphabet_size ({alphabet_size}) must be greater than 1")


def check_alphabet_sizes(alphabet_size_mean, alphabet_size_slope):
    if not are_alphabet_sizes_valid(alphabet_size_mean, alphabet_size_slope):
        raise ValueError(
            f"alphabet_size_mean ({alphabet_size_mean}) * alphabet_size_slope ({alphabet_size_slope}) must be less than or equal to 9"
        )


def check_X(X: Iterable):
    # X is now expected to be an iterable, like a list or tuple
    original_type = type(X).__name__

    # Try to convert X to a numpy array
    try:
        X = np.array(X)
    except Exception as e:
        # If numpy conversion fails, try converting to an awkward array
        try:
            X = ak.Array(X)
        except Exception as e:
            # If both conversions fail, raise an error
            raise ValueError(
                f"X must be convertible to a numpy or awkward array. Found {original_type} instead."
            )

    # # Perform additional checks or processing here if needed
    # # For example, you can check the dimensions of X if it's a numpy array
    # if isinstance(X, np.ndarray):
    #     if X.ndim != 3:
    #         raise ValueError(f"X must be a 3D array. Found {X.ndim} dimensions instead.")

    return X


@njit(cache=True)
def is_valid_windowing(sequence_size: int, window_size: int, dilation: int) -> bool:
    if (
        sequence_size < window_size * dilation
    ):  # if window_size * dilation exceeds the length of the sequence
        return False
    else:
        return True


def drop_nans(X: ak.Array):
    return ak.drop_none(ak.nan_to_none(X))


def set_n_jobs_numba(n_jobs):
    if n_jobs == -1:
        # set_num_threads(psutil.cpu_count(logical=False))
        set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
    else:
        set_num_threads(n_jobs)


@njit(cache=True)
def encode_integers(a, b):
    return HASHMAP_2_SYMBOLS[a, b]


def generate_index(panel):
    return ak.Array([[np.arange(len(x)) for x in X] for X in panel])


@njit(fastmath=True, cache=True)
def get_n_windows(sequence_size, window_size, dilation=1, stride=1, padding=0):
    return 1 + math.floor((sequence_size + 2 * padding - window_size - (dilation - 1) * (window_size - 1)) / stride)


@nb.njit(cache=True)
def log2(x):
    return math.log(x) / math.log(2)


@nb.njit(cache=True)
def halve_symbols(a):
    return a // 2


@nb.njit(fastmath=True, cache=True)
def halve_word(a):
    return (a[::2] + a[1::2]) // 2


@nb.njit(fastmath=True, cache=True)
def convert_to_base_10(number, base):
    result = 0
    multiplier = 1

    while number > 0:
        digit = number % 10
        result += digit * multiplier
        multiplier *= base
        number //= 10

    return result


@nb.njit(fastmath=True, cache=True)
def convert_to_base_l_minus_one(number, base, word_length):
    result = 0
    multiplier = 1

    while number > 0:
        digit = number % (word_length - 1)
        result += digit * multiplier
        multiplier *= base
        number //= (word_length - 1)

    return result

@nb.njit(fastmath=True, cache=True)
def count_digits(number):
    return math.floor(math.log10(number)) + 1
