from fast_borf.align import align_window_to_segments_dilated
from fast_borf.moving import move_mean, move_std
from fast_borf.zscore import zscore_threshold
import numpy as np
import numba as nb

from fast_borf.utils import get_n_windows
from fast_borf.constants import FASTMATH

@nb.njit(fastmath=FASTMATH)
def paa(a, window_size, word_length, min_std_ratio=0, stride=1, dilation=1):
    seg_size = window_size // word_length
    global_std = np.std(a)
    real_n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation, stride=stride)
    n_windows = get_n_windows(sequence_size=a.size, window_size=window_size, dilation=dilation)
    n_segments = get_n_windows(sequence_size=a.size, window_size=seg_size, dilation=dilation)
    window_means = np.full(n_windows, np.nan)
    window_stds = np.full(n_windows, np.nan)
    segment_means = np.full(n_segments, np.nan)
    for d in range(dilation):
        window_means[d::dilation] = move_mean(a[d::dilation], window_size)[window_size - 1 :]
        window_stds[d::dilation] = move_std(a[d::dilation], window_size)[window_size - 1 :]
        segment_means[d::dilation] = move_mean(a[d::dilation], seg_size)[seg_size - 1 :]
    out = np.empty((real_n_windows, word_length))
    for window_idx in range(0, n_windows, stride):
        win_align, seg_align = align_window_to_segments_dilated(
            window_idx=window_idx,
            word_length=word_length,
            seg_size=seg_size,
            dilation=dilation,
        )
        for i in range(len(win_align)):
            out[window_idx // stride, i] = zscore_threshold(
                a=segment_means[seg_align[i]],
                mu=window_means[win_align[i]],
                sigma=window_stds[win_align[i]],
                sigma_global=global_std,
                sigma_threshold=min_std_ratio,
            )
    return out





if __name__ == "__main__":
    a = np.random.randn(1000)
    a = np.arange(20)
    # out = paa(a, 100, 10)
    # out2 = paa(a, 100, 10, dilation=2)
    out3 = paa(a, 10, 3, stride=1, dilation=2)
    print(out3)
    # b = paa_gu(a, 100, 10)
