import numba as nb
import numpy as np

from fast_borf.utils import encode_integers, convert_to_base_10


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
