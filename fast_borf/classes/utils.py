from scipy.sparse import coo_array
from fast_borf.utils import (
    check_window_size_word_length,
    check_alphabet_size,
    check_alphabet_sizes,
    count_digits,
    get_n_windows
)
import numpy as np
import sparse
from joblib import Parallel, delayed
import pandas as pd
import math


def dicts_to_coo(dicts, shape):
    rows = []
    cols = []
    data = []

    for d in dicts:
        if d:
            for key, val in d.items():
                i, j = map(int, key.split(";"))
                rows.append(i)
                cols.append(j)
                data.append(val)

    if rows and cols and data:
        return coo_array((data, (rows, cols)), shape=shape)
    else:
        return coo_array(shape)


def lists_of_lists_to_coo(outer_list, shape):
    outer_list = np.hstack([np.array(list(inner_list)) for inner_list in outer_list])
    return sparse.COO(
        coords=outer_list[:3, :].reshape(3, -1),
        data=outer_list[3, :],
        shape=shape,
        fill_value=0,
    )


def lists_of_lists_to_dok(outer_list, shape):
    dok = sparse.DOK(shape, dtype=np.int64, fill_value=0)
    for inner_list in outer_list:
        for i in range(len(inner_list[0])):
            dok[inner_list[0][i], inner_list[1][i], inner_list[2][i]] = inner_list[3][i]
    return dok


def list_of_int_dicts_to_coo(list_of_dicts, shape):
    tss = []
    signals = []
    words = []
    data = []

    n_digits_word = count_digits(shape[2])
    n_digits_signal = count_digits(shape[1])

    for d in list_of_dicts:
        if d:
            for key, val in d.items():
                key = str(key)
                word = int(key[-n_digits_word:])
                signal = (
                    int(key[-(n_digits_word + n_digits_signal) : -(n_digits_word)]) - 1
                )
                ts = int(key[: -n_digits_word - n_digits_signal]) - 1
                tss.append(ts)
                signals.append(signal)
                words.append(word)
                data.append(val)
    return sparse.COO(
        coords=np.array([tss, signals, words]), data=data, shape=shape, fill_value=0
    )


def lists_of_words_configs_dicts_to_coo(list_of_dicts, list_of_configs, shape):
    tss = []
    signals = []
    words = []
    data = []

    for d_words, d_configs in zip(list_of_dicts, list_of_configs):
        if d_words:
            for key, val in d_words.items():
                tss.append(d_configs["ts_idx"])
                signals.append(d_configs["signal_idx"])
                words.append(key)
                data.append(val)
    return sparse.COO(
        coords=np.array([tss, signals, words]), data=data, shape=shape, fill_value=0
    )


def lists_of_words_configs_lists_to_coo(list_of_dicts, list_of_configs, shape):
    tss = []
    signals = []
    words = []
    data = []

    for d_words, d_configs in zip(list_of_dicts, list_of_configs):
        if d_words:
            for key, val in d_words.items():
                tss.append(d_configs[0])
                signals.append(d_configs[1])
                words.append(key)
                data.append(val)
    return sparse.COO(
        coords=np.array([tss, signals, words]), data=data, shape=shape, fill_value=0
    )


def process_single_config(d_words, d_configs, shape):
    if d_words:
        tss, signals, words, data = [], [], [], []
        for key, val in d_words.items():
            tss.append(d_configs[0])
            signals.append(d_configs[1])
            words.append(key)
            data.append(val)
        return sparse.COO(
            coords=np.array([tss, signals, words, np.zeros(len(tss), dtype=np.int_)]),
            data=data,
            shape=(shape[0], shape[1], shape[2], 1),
            fill_value=0,
        )
    else:
        return sparse.COO(
            coords=np.empty((4, 0)),
            data=[],
            shape=(shape[0], shape[1], shape[2], 1),
            fill_value=0,
        )


def lists_of_words_configs_lists_to_coo_parallel(list_of_dicts, list_of_configs, shape):
    list_of_dicts = [dict(d) for d in list_of_dicts]
    list_of_configs = [list(d) for d in list_of_configs]
    results = Parallel(n_jobs=-1)(
        delayed(process_single_config)(d_words, d_configs, shape)
        for d_words, d_configs in zip(list_of_dicts, list_of_configs)
    )
    return sparse.concatenate(results, axis=-1).sum(axis=-1)


def lists_of_words_configs_lists_to_coo_parallel2(
    list_of_dicts, list_of_configs, shape
):
    results = Parallel(n_jobs=-1)(
        delayed(process_single_config)(d_words, d_configs, shape)
        for d_words, d_configs in zip(
            [dict(d) for d in list_of_dicts], [list(d) for d in list_of_configs]
        )
    )
    return sparse.concatenate(results, axis=-1).sum(axis=-1)


def lists_of_words_configs_lists_to_coo_parallel3(
    list_of_dicts, list_of_configs, shape, n_jobs=-1, backend="loky", return_as="list"
):
    list_of_dicts = [[list(d[0]), list(d[1])] for d in list_of_dicts]
    list_of_configs = [list(d) for d in list_of_configs]
    results = Parallel(n_jobs=n_jobs, backend=backend, return_as=return_as)(
        delayed(process_single_config3)(d_words, d_configs, shape)
        for d_words, d_configs in zip(list_of_dicts, list_of_configs)
    )
    return sparse.concatenate(list(results), axis=-1).sum(axis=-1)


def lists_of_words_configs_lists_to_coo_parallel4(
    list_of_dicts, list_of_configs, shape, n_jobs=-1, backend="loky", return_as="list"
):
    results = Parallel(n_jobs=n_jobs, backend=backend, return_as=return_as)(
        delayed(process_single_config3)(d_words, d_configs, shape)
        for d_words, d_configs in zip(list_of_dicts, list_of_configs)
    )
    return sparse.concatenate(results, axis=-1).sum(axis=-1)


def process_single_config3(d_words, d_configs, shape):
    if d_words:
        return sparse.COO(
            coords=np.array(
                [
                    np.repeat(d_configs[0], len(d_words[0])),
                    np.repeat(d_configs[1], len(d_words[0])),
                    d_words[0],
                    np.zeros(len(d_words[0]), dtype=np.int_),
                ]
            ),
            data=d_words[1],
            shape=(shape[0], shape[1], shape[2], 1),
            fill_value=0,
        )
    else:
        return sparse.COO(
            coords=np.empty((4, 0)),
            data=[],
            shape=(shape[0], shape[1], shape[2], 1),
            fill_value=0,
        )


def lists_to_coo(
    list_of_dicts, list_of_configs, shape, n_jobs=1, normalize=False
):
    results = Parallel(n_jobs=n_jobs)(
        delayed(lists_to_coo_single)(d_words, d_configs, shape, normalize)
        for d_words, d_configs in zip(list_of_dicts, list_of_configs)
    )
    return sparse.concatenate(results, axis=-1).sum(axis=-1)


def lists_to_coo_single(d_words, d_configs, shape, normalize=False):
    shape_ = list(shape)
    shape_.append(1)
    if d_words:
        ts_idx = d_configs[0]
        signal_idx = d_configs[1]
        window_size = d_configs[2]
        word_length = d_configs[3]
        alphabet_size = d_configs[4]
        dilation = d_configs[5]
        stride = d_configs[6]
        signal_size = d_configs[7]
        words = d_words[0]
        counts = d_words[1]
        n_words = len(words)
        if normalize:
            n_windows = get_n_windows(
                sequence_size=signal_size, window_size=window_size, stride=stride, dilation=dilation)
            counts = np.asarray(counts) / n_windows
        return sparse.COO(
            coords=np.asarray(
                [
                    np.repeat(ts_idx, n_words),
                    np.repeat(signal_idx, n_words),
                    np.repeat(window_size, n_words),
                    np.repeat(word_length, n_words),
                    np.repeat(alphabet_size, n_words),
                    np.repeat(dilation, n_words),
                    np.repeat(stride, n_words),
                    words,
                    np.zeros(n_words, dtype=np.int_),
                ]
            ),
            data=counts,
            shape=shape_,
            fill_value=0,
        )
    else:
        return sparse.COO(
            coords=np.empty((9, 0)),
            data=[],
            shape=shape_,
            fill_value=0,
        )


def lists_to_coo_1dsax(
    list_of_dicts, list_of_configs, shape, n_jobs=1, normalize=False
):
    results = Parallel(n_jobs=n_jobs)(
        delayed(lists_to_coo_single_1dsax)(d_words, d_configs, shape, normalize)
        for d_words, d_configs in zip(list_of_dicts, list_of_configs)
    )
    return sparse.concatenate(results, axis=-1).sum(axis=-1)



def lists_to_coo_single_1dsax(d_words, d_configs, shape, normalize=False):
    shape_ = list(shape)
    shape_.append(1)
    if d_words:
        ts_idx = d_configs[0]
        signal_idx = d_configs[1]
        window_size = d_configs[2]
        # word_length = d_configs[3]
        # alphabet_size_mean = d_configs[4]
        # alphabet_size_slope = d_configs[5]
        dilation = d_configs[6]
        stride = d_configs[7]
        signal_size = d_configs[8]
        words = d_words[0]
        counts = d_words[1]
        n_words = len(words)
        if normalize:
            n_windows = get_n_windows(
                sequence_size=signal_size, window_size=window_size, stride=stride, dilation=dilation)
            counts = np.asarray(counts) / n_windows
        dilation = int(np.log2(dilation))
        window_size = int(np.log2(window_size))
        # print(d_configs)
        return sparse.COO(
            coords=np.asarray(
                [
                    np.repeat(ts_idx, n_words),
                    np.repeat(signal_idx, n_words),
                    np.repeat(window_size, n_words),
                    np.repeat(dilation, n_words),
                    words,
                    np.zeros(n_words, dtype=np.int_),
                ]
            ),
            data=counts,
            shape=shape_,
            fill_value=0,
        )
    else:
        return sparse.COO(
            coords=np.empty((6, 0)),
            data=[],
            shape=shape_,
            fill_value=0,
        )


def lists_to_coo_sax(
    list_of_dicts, list_of_configs, shape, n_jobs=1, normalize=False
):
    results = Parallel(n_jobs=n_jobs)(
        delayed(lists_to_coo_single_sax)(d_words, d_configs, shape, normalize)
        for d_words, d_configs in zip(list_of_dicts, list_of_configs)
    )
    return sparse.concatenate(results, axis=-1).sum(axis=-1)



def lists_to_coo_single_sax(d_words, d_configs, shape, normalize=False):
    shape_ = list(shape)
    shape_.append(1)
    if d_configs:
        ts_idx = d_configs[0]
        signal_idx = d_configs[1]
        window_size = d_configs[2]
        # word_length = d_configs[3]
        # alphabet_size_mean = d_configs[4]
        dilation = d_configs[5]
        stride = d_configs[6]
        signal_size = d_configs[7]
        words = d_words[0]
        counts = d_words[1]
        n_words = len(words)
        if normalize:
            n_windows = get_n_windows(
                sequence_size=signal_size, window_size=window_size, stride=stride, dilation=dilation)
            counts = np.asarray(counts) / n_windows
        dilation = int(math.log2(dilation))
        window_size = int(math.log2(window_size))
        return sparse.COO(
            coords=np.asarray(
                [
                    np.repeat(ts_idx, n_words),
                    np.repeat(signal_idx, n_words),
                    np.repeat(window_size, n_words),
                    np.repeat(dilation, n_words),
                    words,
                    np.zeros(n_words, dtype=np.int_),
                ]
            ),
            data=counts,
            shape=shape_,
            fill_value=0,
        )
    else:
        return sparse.COO(
            coords=np.empty((6, 0)),
            data=[],
            shape=shape_,
            fill_value=0,
        )


def lists_of_words_configs_dicts_to_dok(list_of_dicts, list_of_configs, shape):
    out = sparse.DOK(shape, dtype=np.uint64, fill_value=0)
    for d_words, d_configs in zip(list_of_dicts, list_of_configs):
        if d_words:
            for key, val in d_words.items():
                out[d_configs["ts_idx"], d_configs["signal_idx"], key] = val
    return out


def dicts_to_set(out):
    return set([k for d in out for k in d.keys()])


def set_to_dict(out):
    return {integer: counter for counter, integer in enumerate(out)}


def check_sax_parameters(window_size: int, word_length: int, alphabet_size: int):
    check_window_size_word_length(window_size=window_size, word_length=word_length)
    check_alphabet_size(alphabet_size=alphabet_size)


def check_1dsax_parameters(
    window_size: int,
    word_length: int,
    alphabet_size_mean: int,
    alphabet_size_slope: int,
):
    check_window_size_word_length(window_size=window_size, word_length=word_length)
    check_alphabet_sizes(
        alphabet_size_mean=alphabet_size_mean, alphabet_size_slope=alphabet_size_slope
    )


def convert_configs_to_arrays(configs):
    df = pd.DataFrame(configs, columns=["alphabet_size", "window_size", "word_length", "dilation", "stride"])
    grouped = df.groupby(["alphabet_size", "word_length"])
    grouped_dfs = []
    for (_, _), group_df in grouped:
        grouped_dfs.append(group_df.values)
    return grouped_dfs


def convert_configs_to_arrays_1dsax(configs):
    df = pd.DataFrame(configs, columns=["alphabet_size_mean", "alphabet_size_slope", "window_size", "word_length", "dilation", "stride"])
    grouped = df.groupby(["alphabet_size_mean", "alphabet_size_slope", "word_length"])
    grouped_dfs = []
    for (_, _, _), group_df in grouped:
        grouped_dfs.append(group_df.values)
    return grouped_dfs


def convert_configs_to_arrays_sax(configs):
    df = pd.DataFrame(configs, columns=["alphabet_size", "window_size", "word_length", "dilation", "stride"])
    grouped = df.groupby(["alphabet_size", "word_length", "stride"])
    grouped_dfs = []
    for (_, _, _), group_df in grouped:
        grouped_dfs.append(group_df.values)
    return grouped_dfs