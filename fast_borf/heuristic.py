"""Choose the SAX configurations used by BORF from the length of the series."""

import itertools
from typing import Literal, Optional, Sequence

import numpy as np

Complexity = Literal["quadratic", "linear", "linear_logarithmic"]


def powers_of_two(start, stop):
    """start, 2 * start, 4 * start, ... up to stop (inclusive)."""
    values = []
    value = start
    while value <= stop:
        values.append(value)
        value *= 2
    return values


def get_window_sizes(min_length, max_length, min_window_size=4, max_window_size=None):
    """Powers of two from min_window_size to max_window_size (default max_length).

    Smaller powers of two, except 2, are added back when the shortest series is
    at most twice the largest of them.
    """
    if max_window_size is None:
        max_window_size = max_length
    sizes = powers_of_two(2, max_window_size)
    small = [size for size in sizes if size < min_window_size][1:]
    windows = [size for size in sizes if size >= min_window_size]
    if small and min_length <= 2 * max(small):
        windows = small + windows
    return windows


def get_dilations(max_length, min_dilation=1, max_dilation=None):
    """Powers of two from min_dilation to max_dilation (default log2(max_length))."""
    if max_dilation is None:
        max_dilation = np.log2(max_length)
    return powers_of_two(min_dilation, max_dilation)


def get_logarithmic_stride(m, w, d):
    """Stride that makes the number of windows grow logarithmically with m.

    Returns None when the configuration is not valid for series of length m.
    """
    A = (-w + d * w + m * w - d * w**2) / (-1 + m * np.log(m))
    B = (1 - w + d * w + m * w - d * w**2) / m
    cond_1 = (w == 1) and (m > 1) and (d == 1) and (np.log(m) > 1)
    cond_2 = (w == 1) and (m > 1) and (d > 1) and (np.log(m) > 1)
    cond_3 = (w >= 2) and (m > w) and (1 <= d < ((m - 1) / w - 1)) and (np.log(m) >= B)
    cond_4 = (w >= 2) and (m > w) and (1 <= d < ((m - 1) / w - 1)) and (np.log(m) < B)
    if cond_1 or cond_2 or cond_3:
        return 1
    if cond_4:
        return int(np.ceil(A))
    return None


def generate_configs(
    min_length: int,
    max_length: int,
    min_window_size: int = 4,
    max_window_size: Optional[int] = None,
    max_word_length: int = 8,
    alphabet_sizes: Sequence[int] = (3,),
    min_dilation: int = 1,
    max_dilation: Optional[int] = None,
    complexity: Complexity = "quadratic",
):
    """List the SAX configurations for series of lengths in [min_length, max_length].

    Every combination of window size, dilation, word length and alphabet size is
    kept if the word fits in the window and the dilated window fits in the longest
    series. The stride depends on complexity: 1 for "quadratic", the word length
    for "linear", and a length-dependent stride for "linear_logarithmic" (which
    can drop configurations that do not suit the shortest series).
    """
    window_sizes = get_window_sizes(
        min_length, max_length, min_window_size, max_window_size
    )
    dilations = get_dilations(max_length, min_dilation, max_dilation)
    word_lengths = powers_of_two(2, max_word_length)

    configs = []
    for window_size, dilation, word_length, alphabet_size in itertools.product(
        window_sizes, dilations, word_lengths, alphabet_sizes
    ):
        if word_length > window_size or window_size * dilation > max_length:
            continue
        if complexity == "quadratic":
            stride = 1
        elif complexity == "linear":
            stride = word_length
        elif complexity == "linear_logarithmic":
            # FIXME: this creates problems when min_length != max_length. The only
            #  fix I see is to divide time series signals by size. This is a problem
            #  only when the size differs a lot.
            stride = get_logarithmic_stride(min_length, window_size, dilation)
            if stride is None:
                continue
        else:
            raise ValueError(f"Unknown complexity: {complexity!r}")
        configs.append(
            dict(
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                word_length=word_length,
                alphabet_size=alphabet_size,
            )
        )
    return configs
