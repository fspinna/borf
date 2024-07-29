import numba as nb
import numpy as np


@nb.njit(fastmath=True, cache=True)
def int_to_array_new_base(number, base, word_length):
    array = np.zeros(word_length, dtype=np.int32)
    for i in range(word_length):
        power = word_length - i - 1
        array[i] = number // (base ** power)
        number %= base ** power
    return array


@nb.njit(fastmath=True, cache=True)
def int_to_sax_words(numbers, base, word_length):
    out = np.empty((numbers.shape[0], word_length), dtype=np.int64)
    for i in range(numbers.shape[0]):
        out[i] = int_to_array_new_base(numbers[i], base, word_length)
    return out


@nb.njit(fastmath=True, cache=True)
def array_to_int_new_base(array, base):
    word_length = array.shape[0]
    result = 0
    for i in range(0, word_length, 1):
        result += array[i] * base ** (word_length - i - 1)
    return result


@nb.njit(fastmath=True, cache=True)
def sax_words_to_int(arrays, base):
    out = np.empty(arrays.shape[0], dtype=np.int64)
    for i in range(arrays.shape[0]):
        out[i] = array_to_int_new_base(arrays[i], base)
    return out
