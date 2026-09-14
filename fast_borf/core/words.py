"""SAX words as integers: symbol j of a word of length L has weight base**(L - 1 - j)."""

import numba as nb
import numpy as np


@nb.njit(fastmath=True, cache=True)
def encode_word(symbols, base):
    word_length = symbols.shape[0]
    result = 0
    for i in range(0, word_length, 1):
        result += symbols[i] * base ** (word_length - i - 1)
    return result


@nb.njit(fastmath=True, cache=True)
def encode_words(symbols, base):
    """One integer per row of symbols."""
    out = np.empty(symbols.shape[0], dtype=np.int64)
    for i in range(symbols.shape[0]):
        out[i] = encode_word(symbols[i], base)
    return out


@nb.njit(fastmath=True, cache=True)
def decode_word(number, base, word_length):
    symbols = np.zeros(word_length, dtype=np.int32)
    for i in range(word_length):
        power = word_length - i - 1
        symbols[i] = number // (base**power)
        number %= base**power
    return symbols


@nb.njit(fastmath=True, cache=True)
def decode_words(numbers, base, word_length):
    """Symbols of each integer word, shape (len(numbers), word_length)."""
    out = np.empty((numbers.shape[0], word_length), dtype=np.int64)
    for i in range(numbers.shape[0]):
        out[i] = decode_word(numbers[i], base, word_length)
    return out
