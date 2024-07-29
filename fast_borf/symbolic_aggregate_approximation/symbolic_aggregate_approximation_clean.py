import numba as nb
import numpy as np
from fast_borf.utils import get_n_windows
from fast_borf.moving import move_mean, move_std
from fast_borf.zscore import zscore_threshold

# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')


@nb.njit(fastmath=True)
def is_better_naive(m, w, l):
    if w == 1:
        if m >= 1 and l == 1:
            return True
    elif w == 2:
        if m >= 2 and (l == 1 or l == 2):
            return True
    elif w >= 3:
        upper_m = (1 - 2 * w + w**2) / (-2 + w)
        if w <= m <= upper_m and 1 <= l <= w:
            return True
        elif m > upper_m:
            lower_l = (-m + w + m * w - w**2) / (1 + m - w)
            if lower_l <= l <= w:
                return True
    return False


@nb.njit
def fast_digitize(value, bins):
    for i in range(len(bins)):
        if value < bins[i]:
            return i
    return len(bins)




@nb.njit
def sax_opt(
    a,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_windows_moving = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation)
    global_std = np.std(a)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    seg_size = window_size // word_length
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    out = np.zeros((n_windows, word_length))
    if is_better_naive(m=n_windows, w=word_length, l=word_length):
        for d in range(dilation):
            window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1:]
            window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1:]
        for i in range(n_windows):
            for j in range(word_length):
                out_i_j = 0
                for k in range(seg_size):
                    out_i_j += a[(i * stride) + (j * seg_size * dilation) + (k * dilation)]
                out[i, j] = zscore_threshold(
                    a=out_i_j / seg_size,
                    mu=window_means[i * stride],
                    sigma=window_stds[i * stride],
                    sigma_global=global_std,
                    sigma_threshold=min_window_to_signal_std_ratio
                )
    else:
        segment_means = np.full(n_segments, np.nan)
        for d in range(dilation):
            window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1:]
            window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1:]
            segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1:]
        out = np.zeros((n_windows, word_length))
        for i in range(n_windows):
            for j in range(word_length):
                out[i, j] = zscore_threshold(
                    a=segment_means[(i * stride) + (j * seg_size * dilation)],
                    mu=window_means[i * stride],
                    sigma=window_stds[i * stride],
                    sigma_global=global_std,
                    sigma_threshold=min_window_to_signal_std_ratio
                )
    return np.digitize(out, bins).astype(np.uint8)



@nb.njit
def sax_opt_simple(
    a,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_windows_moving = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation)
    global_std = np.std(a)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    seg_size = window_size // word_length
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    out = np.zeros((n_windows, word_length))
    if seg_size == 1:
        for d in range(dilation):
            window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1:]
            window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1:]
        for i in range(n_windows):
            for j in range(word_length):
                out[i, j] = zscore_threshold(
                    a=a[(i * stride) + (j * seg_size * dilation)],
                    mu=window_means[i * stride],
                    sigma=window_stds[i * stride],
                    sigma_global=global_std,
                    sigma_threshold=min_window_to_signal_std_ratio
                )
    else:
        segment_means = np.full(n_segments, np.nan)
        for d in range(dilation):
            window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1:]
            window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1:]
            segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1:]
        out = np.zeros((n_windows, word_length))
        for i in range(n_windows):
            for j in range(word_length):
                out[i, j] = zscore_threshold(
                    a=segment_means[(i * stride) + (j * seg_size * dilation)],
                    mu=window_means[i * stride],
                    sigma=window_stds[i * stride],
                    sigma_global=global_std,
                    sigma_threshold=min_window_to_signal_std_ratio
                )
    return np.digitize(out, bins).astype(np.uint8)


@nb.njit(cache=True)
def sax(
    a,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,

):
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_windows_moving = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation)
    global_std = np.std(a)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    seg_size = window_size // word_length
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    segment_means = np.full(n_segments, np.nan)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1:]
        window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1:]
        segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1:]
    out = np.zeros((n_windows, word_length))
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = zscore_threshold(
                a=segment_means[(i * stride) + (j * seg_size * dilation)],
                mu=window_means[i * stride],
                sigma=window_stds[i * stride],
                sigma_global=global_std,
                sigma_threshold=min_window_to_signal_std_ratio
            )
    return np.digitize(out, bins).astype(np.uint8)


@nb.njit
def sax_fast_digitize(
    a,
    window_size,
    word_length,
    bins,
    stride=1,
    dilation=1,
    min_window_to_signal_std_ratio=0.0,
):
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_windows_moving = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation)
    global_std = np.std(a)
    if global_std == 0:
        return np.zeros((n_windows, word_length), dtype=np.uint8)
    seg_size = window_size // word_length
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    segment_means = np.full(n_segments, np.nan)
    window_means = np.full(n_windows_moving, np.nan)
    window_stds = np.full(n_windows_moving, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1:]
        window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1:]
        segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1:]
    out = np.zeros((n_windows, word_length), dtype=np.uint8)
    for i in range(n_windows):
        for j in range(word_length):
            out[i, j] = fast_digitize(zscore_threshold(
                a=segment_means[(i * stride) + (j * seg_size * dilation)],
                mu=window_means[i * stride],
                sigma=window_stds[i * stride],
                sigma_global=global_std,
                sigma_threshold=min_window_to_signal_std_ratio
            ), bins)
    return out


if __name__ == "__main__":
    pass
    # from fast_borf.symbolic_aggregate_approximation.symbolic_aggregate_approximation_dilated import sax as sax2
    a = np.random.randn(1000)
    window_size = 32
    word_length = 8
    alphabet_size = 2
    stride = 1
    dilation = 1
    bins = np.zeros(1).astype(np.float64)
    # out = sax(a, window_size, word_length, bins, stride, dilation)
    out2 = sax(a, window_size, word_length, bins, stride, dilation)
    # out3 = sax3(a, window_size, word_length, bins, stride, dilation)
    # print(np.allclose(out3, out2))
