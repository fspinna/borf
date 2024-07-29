from fast_borf.align import align_window_to_segments
from fast_borf.moving import move_mean, move_std
from fast_borf.zscore import zscore_threshold
import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def paa(a, window_size, word_length, min_std_ratio=0, stride=1):
    seg_size = window_size // word_length
    global_std = np.std(a)
    window_means = move_mean(a, window_size)[window_size - 1 :]
    window_stds = move_std(a, window_size)[window_size - 1 :]
    segment_means = move_mean(a, seg_size)[seg_size - 1 :]
    out = np.empty((len(window_means), word_length))
    for window_idx in range(0, len(window_means), stride):
        win_align, seg_align = align_window_to_segments(
            window_idx=window_idx,
            word_length=word_length,
            seg_size=seg_size,
        )

        z = np.empty(word_length)
        for i in range(len(win_align)):
            z[i] = zscore_threshold(
                a=segment_means[seg_align[i]],
                mu=window_means[win_align[i]],
                sigma=window_stds[win_align[i]],
                sigma_global=global_std,
                sigma_threshold=min_std_ratio,
            )

        out[window_idx] = z
    return out


def paa_gu(a, window_size, word_length, min_std_ratio=0):
    dum_ws = np.zeros(len(a) - window_size + 1)
    dum_wl = np.zeros(word_length)
    out = np.empty((dum_ws.size, dum_wl.size))
    return _paa_gu(a, window_size, word_length, min_std_ratio, dum_ws, dum_wl, out)


@nb.guvectorize(
    "float64[:], int64, int64, float64, float64[:], float64[:], float64[:, :]",
    "(n),(),(),(),(m),(l)->(m,l)",
    nopython=True, fastmath=True,
)
def _paa_gu(a, window_size, word_length, min_std_ratio, dum_ws, dum_wl, out):
    out_ = paa(a, window_size, word_length, min_std_ratio)
    for i in range(out_.shape[0]):
        for j in range(out_.shape[1]):
            out[i, j] = out_[i, j]


if __name__ == "__main__":
    a = np.random.randn(1000)
    b = paa_gu(a, 100, 10)
