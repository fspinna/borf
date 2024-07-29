import numba as nb

import awkward as ak
import numpy as np

from fast_borf.symbolic_aggregate_approximation.symbolic_aggregate_approximation_clean import sax
from fast_borf.utils import (
    get_norm_bins,
    are_window_size_and_dilation_compatible_with_signal_length,
    convert_to_base_10,
)
from fast_borf.bag_of_patterns.utils import (
    array_to_int,
    ndindex_2d_array,
)

import pandas as pd
from fast_borf.hash_unique import unique


@nb.njit(cache=True)
def sax_words_to_int(sax_words, alphabet_size):
    return [convert_to_base_10(array_to_int(sax_word), alphabet_size) for sax_word in sax_words]


@nb.njit(cache=True)
def new_transform_single(
    a: np.ndarray,
    window_size,
    word_length,
    alphabet_size,
    bins,
    dilation,
    stride=1,
    min_window_to_signal_std_ratio=0.0,
):
    sax_words = sax(
        a=a,
        window_size=window_size,
        word_length=word_length,
        bins=bins,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        dilation=dilation,
        stride=stride,
    )
    sax_words = np.asarray(sax_words_to_int(sax_words, alphabet_size))
    return unique(sax_words)


@nb.njit(cache=True)
def new_transform_single_conf(
    a: np.ndarray,
    ts_idx,
    signal_idx,
    window_size,
    word_length,
    alphabet_size,
    bins,
    dilation,
    stride=1,
    min_window_to_signal_std_ratio=0.0,
):
    words, counts = new_transform_single(
        a=a,
        window_size=window_size,
        word_length=word_length,
        alphabet_size=alphabet_size,
        bins=bins,
        dilation=dilation,
        stride=stride,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
    )
    ts_idxs = np.full(len(words), ts_idx)
    signal_idxs = np.full(len(words), signal_idx)
    return np.column_stack((ts_idxs, signal_idxs, words, counts))


@nb.njit(parallel=True, nogil=True, cache=True)
def transform_sax_patterns_ts(
        ts: ak.Array,
        window_size,
        word_length,
        alphabet_size,
        stride,
        dilation,
        signal_idx,
        min_window_to_signal_std_ratio=0.0,

):
    bins = get_norm_bins(alphabet_size=alphabet_size)
    counts = np.empty(len(ts)+1, dtype=np.int64)
    for i in nb.prange(len(ts)):
        counts[i+1] = len(new_transform_single_conf(
                a=np.asarray(ts[i]),
                ts_idx=i,
                signal_idx=signal_idx,
                window_size=window_size,
                word_length=word_length,
                alphabet_size=alphabet_size,
                bins=bins,
                dilation=dilation,
                stride=stride,
                min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,))
    cum_counts = np.cumsum(counts)
    n_rows = np.sum(counts)
    shape = (n_rows, 4)
    out = np.empty(shape, dtype=np.int64)
    for j in nb.prange(len(ts)):
        out_ = new_transform_single_conf(
            a=np.asarray(ts[j]),
            ts_idx=j,
            signal_idx=signal_idx,
            window_size=window_size,
            word_length=word_length,
            alphabet_size=alphabet_size,
            bins=bins,
            dilation=dilation,
            stride=stride,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )
        out[cum_counts[j]:cum_counts[j+1]] = out_
    return out



@nb.njit(parallel=True, nogil=True, cache=True)
def transform_sax_patterns(
        panel: ak.Array,
        window_size,
        word_length,
        alphabet_size,
        stride,
        dilation,
        min_window_to_signal_std_ratio=0.0,

):
    bins = get_norm_bins(alphabet_size=alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals
    counts = np.zeros(iterations + 1, dtype=np.int64)
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx])
        signal = signal[~np.isnan(signal)]
        if not are_window_size_and_dilation_compatible_with_signal_length(
                window_size, dilation, signal.size
        ):
            continue
        counts[i+1] = len(new_transform_single_conf(
                a=signal,
                ts_idx=ts_idx,
                signal_idx=signal_idx,
                window_size=window_size,
                word_length=word_length,
                alphabet_size=alphabet_size,
                bins=bins,
                dilation=dilation,
                stride=stride,
                min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,))
    cum_counts = np.cumsum(counts)
    n_rows = np.sum(counts)
    shape = (n_rows, 4)
    out = np.empty(shape, dtype=np.int64)
    # return out, counts, cum_counts
    for i in nb.prange(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx])
        signal = signal[~np.isnan(signal)]
        if not are_window_size_and_dilation_compatible_with_signal_length(
                window_size, dilation, signal.size
        ):
            continue
        out_ = new_transform_single_conf(
            a=signal,
            ts_idx=ts_idx,
            signal_idx=signal_idx,
            window_size=window_size,
            word_length=word_length,
            alphabet_size=alphabet_size,
            bins=bins,
            dilation=dilation,
            stride=stride,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )
        out[cum_counts[i]:cum_counts[i+1], :] = out_
    return out


def transform_sax_patterns_nonumba(
        panel: ak.Array,
        window_size,
        word_length,
        alphabet_size,
        stride,
        dilation,
        min_window_to_signal_std_ratio=0.0,

):
    bins = get_norm_bins(alphabet_size=alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals
    out = list()
    for i in range(iterations):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx])
        signal = signal[~np.isnan(signal)]
        out.append(new_transform_single_conf(
            a=signal,
            ts_idx=ts_idx,
            signal_idx=signal_idx,
            window_size=window_size,
            word_length=word_length,
            alphabet_size=alphabet_size,
            bins=bins,
            dilation=dilation,
            stride=stride,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        ))
    return np.vstack(out)


from joblib import Parallel, delayed


def transform_sax_patterns_nonumba_par(
        panel: ak.Array,
        window_size,
        word_length,
        alphabet_size,
        stride,
        dilation,
        min_window_to_signal_std_ratio=0.0,
):
    bins = get_norm_bins(alphabet_size=alphabet_size)
    n_signals = len(panel[0])
    n_ts = len(panel)
    iterations = n_ts * n_signals

    # Define a helper function for parallel execution
    def process_iteration(i):
        ts_idx, signal_idx = ndindex_2d_array(i, n_signals)
        signal = np.asarray(panel[ts_idx][signal_idx])
        signal = signal[~np.isnan(signal)]
        return new_transform_single_conf(
            a=signal,
            ts_idx=ts_idx,
            signal_idx=signal_idx,
            window_size=window_size,
            word_length=word_length,
            alphabet_size=alphabet_size,
            bins=bins,
            dilation=dilation,
            stride=stride,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        )

    # Use joblib to parallelize the loop
    out = Parallel(n_jobs=-1)(delayed(process_iteration)(i) for i in range(iterations))

    return np.vstack(out)


if __name__ == "__main__":


    np.random.seed(0)
    # X = np.random.randn(1000, 2, 100)
    #SMALL_PANEL = np.random.randn(1, 2, 1_000)
    X = np.random.randn(1_000, 1, 1_000)

    out = transform_sax_patterns_nonumba_par(
        panel=X,
        window_size=32,
        word_length=8,
        alphabet_size=2,
        stride=1,
        dilation=1,
        min_window_to_signal_std_ratio=0.0,
    )

    # x = X[0]
    #
    # out = transform_sax_patterns_ts(
    #     ts=x,
    #     window_size=32,
    #     word_length=8,
    #     alphabet_size=2,
    #     stride=1,
    #     dilation=1,
    #     signal_idx=0,
    # )



    # x = X[0][0]
    #
    # out = new_transform_single_conf(
    #     a=x,
    #     ts_idx=0,
    #     signal_idx=0,
    #     window_size=32,
    #     word_length=8,
    #     alphabet_size=2,
    #     bins=np.zeros(1),
    #     dilation=1,
    # )


    # words, counts = new_transform_single(
    #     a=x,
    #     window_size=32,
    #     word_length=8,
    #     alphabet_size=2,
    #     bins=np.zeros(1),
    #     dilation=1,
    # )
    #
    # # test = concat_test()
    # # X = np.random.randn(1, 1, 50)
    # configs = nb.typed.List([
    #     nb.typed.List([32, 1]),
    # ])
    # out, config = transform_sax_patterns(
    #     X, configurations=configs, alphabet_size=2, word_length=8, stride=1
    # )



    # from classes.utils import list_of_int_dicts_to_coo

    # shape = (1, 1, convert_to_base_10(11111111, 2))
    # out = list_of_int_dicts_to_coo(out, shape)