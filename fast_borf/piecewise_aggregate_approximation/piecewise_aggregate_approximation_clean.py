import numba as nb
import numpy as np
from fast_borf.utils import get_n_windows
from fast_borf.moving import move_mean



@nb.njit
def paa_naive(a, window_size, word_length, stride=1, dilation=1):
    seg_size = window_size // word_length
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out_i_j = 0
            for k in range(seg_size):
                out_i_j += a[(i * stride) + (j * seg_size * dilation) + (k * dilation)]
            out[i, j] = out_i_j / seg_size
    return out


@nb.njit
def paa_optimized(a, window_size, word_length, stride=1, dilation=1):
    seg_size = window_size // word_length
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    segment_means = np.full(n_segments, np.nan)
    for d in range(dilation):
        segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1:]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = segment_means[(i * stride) + (j * seg_size * dilation)]
    return out, segment_means


@nb.njit
def paa(a, window_size, word_length, stride=1, dilation=1):
    seg_size = window_size // word_length
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    segment_means = np.full(n_segments, np.nan)
    for d in range(dilation):
        segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1:]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = segment_means[(i * stride) + (j * seg_size * dilation)]
    return out



if __name__ == "__main__":
    # from fast_borf.piecewise_aggregate_approximation.piecewise_aggregate_approximation_dilated import paa
    a = np.random.randn(1000)
    a = np.arange(0, 100, 1.0)
    # a = np.arange(20)
    # real = paa(a, 100, 10)
    out = paa_naive(a, 32, 8, dilation=2, stride=3)
    out2, means = paa_optimized.py_func(a, 32, 8, dilation=2, stride=3)
    print(np.allclose(out, out2))
    # out3 = paa_naive(a, 10, 3, stride=1, dilation=2)
    # print(out3)
    # b = paa_gu(a, 100, 10)