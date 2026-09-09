import numba as nb
import numpy as np

from fast_borf.utils import convert_to_base_10, encode_integers


@nb.njit(cache=True)
def array_to_int(arr):
    result = 0
    for i in range(len(arr)):
        result = result * 10 + arr[i]
    return result


@nb.njit(cache=True)
def arrays_to_int(a, b):
    result = 0
    for i in range(len(a)):
        result = result * 10 + encode_integers(a[i], b[i])
    return result


@nb.njit(cache=True)
def array_to_bytes_str(x):
    return "".join([chr(x[i]) for i in range(len(x))])


@nb.njit(cache=True)
def arrays_to_bytes_str_objmode(x):
    with nb.objmode(result="unicode_type"):
        result = x.tobytes().decode("utf-8")
    return result


@nb.njit(cache=True)
def array_to_str(arr):
    result = ""
    for i in range(len(arr)):
        result += str(arr[i])
    return result


@nb.njit(cache=True)
def add_prepended_number(prepend_num, converted_array, num_digits):
    scaled_prepend = prepend_num * (10**num_digits)
    return scaled_prepend + converted_array


@nb.njit(cache=True)
def ndindex_2d_array(idx, dim2_shape):
    row_idx = idx // dim2_shape
    col_idx = idx % dim2_shape
    return row_idx, col_idx


@nb.njit(cache=True)
def ndindex_3d_array(idx, dim2_shape, dim3_shape):
    plane_size = dim2_shape * dim3_shape
    row_idx = idx // plane_size
    remainder = idx % plane_size
    col_idx = remainder // dim3_shape
    depth_idx = remainder % dim3_shape
    return row_idx, col_idx, depth_idx


@nb.njit(cache=True)
def inverse_nindex_2d_array(dim1_idx, dim2_idx, dim2_shape):
    return dim1_idx * dim2_shape + dim2_idx


@nb.njit(cache=True)
def get_hash_table_size(word_length, alphabet_size):
    max_base_a = array_to_int(np.full(word_length, alphabet_size - 1))
    return convert_to_base_10(max_base_a, alphabet_size)


def separate_timestamps_from_panel(X, contains_time_idx):
    if contains_time_idx:
        timestamps = X[:, -1:, :]
        X = X[:, :-1, :]
    else:
        timestamps = np.repeat(np.arange(X.shape[2])[None, None, :], len(X), axis=0)
    return X, timestamps


@nb.njit(fastmath=True, cache=True)
def array_to_int_new_base(array, base):
    word_length = array.shape[0]
    result = 0
    for i in range(0, word_length, 1):
        result += array[i] * base ** (word_length - i - 1)
    return result


@nb.njit(fastmath=True, cache=True)
def int_to_array_new_base(number, base, word_length):
    array = np.zeros(word_length, dtype=np.int32)
    for i in range(word_length):
        power = word_length - i - 1
        array[i] = number // (base**power)
        number %= base**power
    return array


@nb.njit(fastmath=True, cache=True)
def sax_words_to_int(arrays, base):
    out = np.empty(arrays.shape[0], dtype=np.int64)
    for i in range(arrays.shape[0]):
        out[i] = array_to_int_new_base(arrays[i], base)
    return out


@nb.njit(fastmath=True, cache=True)
def int_to_sax_words(numbers, base, word_length):
    out = np.empty((numbers.shape[0], word_length), dtype=np.int64)
    for i in range(numbers.shape[0]):
        out[i] = int_to_array_new_base(numbers[i], base, word_length)
    return out
