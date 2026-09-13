import math

import numba as nb
import numpy as np
from numba import njit, set_num_threads, vectorize


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


@njit(cache=True)
def is_empty(a: np.ndarray) -> bool:
    return a.size == 0


@njit(fastmath=True, cache=True)
def are_window_size_and_dilation_compatible_with_signal_length(
    window_size, dilation, signal_length
):
    if window_size + (window_size - 1) * (dilation - 1) <= signal_length:
        return True
    else:
        return False


@njit(cache=True)
def is_valid_windowing(sequence_size: int, window_size: int, dilation: int) -> bool:
    if (
        sequence_size < window_size * dilation
    ):  # if window_size * dilation exceeds the length of the sequence
        return False
    else:
        return True


def set_n_jobs_numba(n_jobs):
    if n_jobs == -1:
        # set_num_threads(psutil.cpu_count(logical=False))
        set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
    else:
        set_num_threads(n_jobs)


@njit(fastmath=True, cache=True)
def get_n_windows(sequence_size, window_size, dilation=1, stride=1, padding=0):
    return 1 + math.floor(
        (sequence_size + 2 * padding - window_size - (dilation - 1) * (window_size - 1))
        / stride
    )


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
