import awkward as ak
import numba as nb
from numba import njit
from numpy.typing import NDArray
from ebop.algorithms.oned_nan_symbolic_aggregate_approximation import oned_nan_sax
from ebop.algorithms.symbolic_aggregate_approximation import paa_sax
from ebop.utils.condition_utils import (
    is_empty,
    is_valid_windowing,
    is_window_std_negligible,
)
from ebop.utils.transform_utils import (
    offset_transform,
    zscore_transform,
    get_norm_bins,
    pad,
    array_to_string,
    arrays_to_string,
    attach_metadata_to_key,
)
import numpy as np


@njit
def normalize(
    window: NDArray,
    signal_std: float,
    min_window_to_signal_std_ratio: float = 0,
    flatten_to_zero: bool = False,
) -> NDArray:
    window_std = np.nanstd(window)
    if is_window_std_negligible(
        sequence_std=signal_std,
        window_std=window_std,
        min_std_ratio=min_window_to_signal_std_ratio,
    ):
        if flatten_to_zero:
            window = np.repeat(0.0, window.size)
        else:
            window = offset_transform(transformed_array=window, transforming_array=None)
    else:
        window = zscore_transform(transformed_array=window, transforming_array=None)
    return window


@njit
def extract_sax_words_single_configuration(
    X: ak.Array,
    window_size: int,
    word_length: int,
    alphabet_size_mean: int,
    alphabet_size_slope: int = 0,
    stride: int = 1,
    dilation: int = 1,
    use_signal_id: bool = True,
    window_start_idx: int = 0,
    min_window_to_signal_std_ratio: float = 0,
    prefix: str = "",
    signal_separator: str = ";",
    dict_key_type=nb.types.string,
    dict_value_type=nb.types.int64,
    padding: str = "valid",
    normalize_flatten_to_zero: bool = False,
    feature_counter: int = 0,
):
    words_dict = nb.typed.Dict.empty(key_type=dict_key_type, value_type=dict_value_type)
    bins_mean = get_norm_bins(alphabet_size_mean, 0, 1)
    if alphabet_size_slope != 0:
        bins_slope = get_norm_bins(alphabet_size_slope, 0, 1)
    else:
        bins_slope = np.empty(0)
    n_ts = len(X)
    n_signals = len(X[0])
    step = (window_size - 1) * dilation + 1
    for ts_idx in range(n_ts):
        for signal_idx in range(n_signals):
            signal = np.asarray(X[ts_idx][signal_idx])
            signal_std = np.nanstd(signal)
            if not is_valid_windowing(
                sequence_size=signal.size,
                window_size=window_size,
                dilation=dilation,
            ):
                continue
            if word_length > window_size:  # FIXME
                continue
            if padding != "valid":
                signal = pad(
                    array=signal,
                    left_pad=int((window_size / word_length) / 2),
                    right_pad=int((window_size / word_length) / 2),
                    mode=padding,
                )
            for j in np.arange(
                start=window_start_idx,
                stop=signal.size
                - window_size
                - ((window_size - 1) * (dilation - 1))
                + 1,
                step=stride,
            ):
                start = j
                end = start + step
                window_idx = np.arange(
                    start=start, stop=end, step=dilation, dtype=np.int_
                )
                window = signal[window_idx]
                window = normalize(
                    window=window,
                    signal_std=signal_std,
                    min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
                    flatten_to_zero=normalize_flatten_to_zero,
                )
                sax_word = apply_sax(
                    window=window,
                    word_length=word_length,
                    alphabet_size_slope=alphabet_size_slope,
                    bins_mean=bins_mean,
                    bins_slope=bins_slope,
                    prefix=prefix,
                    signal_idx=signal_idx,
                    signal_separator=signal_separator,
                    use_signal_id=use_signal_id,
                )
                if sax_word not in words_dict:
                    words_dict[sax_word] = feature_counter
                    feature_counter += 1
    return words_dict


@njit
def apply_sax(
    window,
    word_length,
    alphabet_size_slope,
    bins_mean,
    bins_slope,
    prefix,
    signal_idx,
    signal_separator,
    use_signal_id,
):
    if alphabet_size_slope == 0:
        # SAX
        sax_word = paa_sax(
            sequence=window, bins=bins_mean, word_length=word_length
        ).astype(np.int_)
        sax_word = attach_metadata_to_key(
            array_to_string(sax_word),
            prefix=prefix,
            signal_separator=signal_separator,
            signal_idx=signal_idx,
            use_signal_idx=use_signal_id,
        )
    else:
        # SAX-1D
        slopes, means = oned_nan_sax(
            sequence=window,
            word_length=word_length,
            bins_mean=bins_mean,
            bins_slope=bins_slope,
        )
        sax_word = attach_metadata_to_key(
            arrays_to_string(slopes, means),
            prefix=prefix,
            signal_separator=signal_separator,
            signal_idx=signal_idx,
            use_signal_idx=use_signal_id,
        )
    return sax_word


@njit
def compute_word_position(window_idx, strategy="average"):
    if strategy == "average":
        return window_idx.mean()
    elif strategy == "first":
        return window_idx[0]
    elif strategy == "last":
        return window_idx[-1]
    else:
        raise NotImplementedError


@njit
def transform_sax_words_single_configuration(
    X: ak.Array,
    sax_words: nb.typed.Dict,
    row_idxs: nb.typed.List,
    column_idxs: nb.typed.List,
    values: nb.typed.List,
    positions: nb.typed.List,
    window_size: int,
    word_length: int,
    alphabet_size_mean: int,
    alphabet_size_slope: int = 0,
    stride: int = 1,
    dilation: int = 1,
    use_signal_id: bool = True,
    window_start_idx: int = 0,
    store_word_position: bool = False,
    min_window_to_signal_std_ratio: float = 0,
    prefix: str = "",
    signal_separator: str = ";",
    padding: str = "valid",
    normalize_flatten_to_zero: bool = False,
):
    bins_mean = get_norm_bins(alphabet_size_mean, 0, 1)
    if alphabet_size_slope != 0:
        bins_slope = get_norm_bins(alphabet_size_slope, 0, 1)
    else:
        bins_slope = np.empty(0)
    n_ts = len(X)
    n_signals = len(X[0])
    step = (window_size - 1) * dilation + 1
    for ts_idx in range(n_ts):
        for signal_idx in range(n_signals):
            previous_sax_word = ""  # reset the previous sax word
            current_word_reps_counter = 0  # reset the counter
            current_word_pos_counter = 0  # reset the position counter
            signal = np.asarray(X[ts_idx][signal_idx])
            signal_std = np.nanstd(signal)
            if not is_valid_windowing(
                sequence_size=signal.size,
                window_size=window_size,
                dilation=dilation,
            ):
                continue
            if word_length > window_size:  # FIXME
                continue
            if padding != "valid":
                signal = pad(
                    array=signal,
                    left_pad=int((window_size / word_length) / 2),
                    right_pad=int((window_size / word_length) / 2),
                    mode=padding,
                )
            for j in np.arange(
                start=window_start_idx,
                stop=signal.size
                - window_size
                - ((window_size - 1) * (dilation - 1))
                + 1,
                step=stride,
            ):
                start = j
                end = start + step
                window_idx = np.arange(
                    start=start, stop=end, step=dilation, dtype=np.int_
                )
                if is_empty(window_idx):
                    continue
                window = np.asarray(X[ts_idx][signal_idx])[window_idx]
                window = normalize(
                    window=window,
                    signal_std=signal_std,
                    min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
                    flatten_to_zero=normalize_flatten_to_zero,
                )
                sax_word = apply_sax(
                    window=window,
                    word_length=word_length,
                    alphabet_size_slope=alphabet_size_slope,
                    bins_mean=bins_mean,
                    bins_slope=bins_slope,
                    prefix=prefix,
                    signal_idx=signal_idx,
                    signal_separator=signal_separator,
                    use_signal_id=use_signal_id,
                )
                if previous_sax_word == "":
                    current_word_reps_counter += 1
                    current_word_pos_counter += (window_idx[-1] + window_idx[0]) / 2
                elif sax_word == previous_sax_word:
                    current_word_reps_counter += 1
                else:
                    if previous_sax_word in sax_words:
                        update_lists(
                            word_reps_count=current_word_reps_counter,
                            word_pos_count=current_word_pos_counter,
                            row_idxs=row_idxs,
                            values=values,
                            positions=positions,
                            column_idxs=column_idxs,
                            sax_word=previous_sax_word,
                            sax_words=sax_words,
                            ts_idx=ts_idx,
                            store_word_position=store_word_position,
                        )
                    current_word_reps_counter = 1
                    current_word_pos_counter = (window_idx[-1] + window_idx[0]) / 2
                previous_sax_word = sax_word
            if previous_sax_word in sax_words:
                update_lists(
                    word_reps_count=current_word_reps_counter,
                    word_pos_count=current_word_pos_counter,
                    row_idxs=row_idxs,
                    values=values,
                    positions=positions,
                    column_idxs=column_idxs,
                    sax_word=previous_sax_word,
                    sax_words=sax_words,
                    ts_idx=ts_idx,
                    store_word_position=store_word_position,
                )
    return


@njit
def fit_transform_sax_words_single_configuration(
    X: ak.Array,
    row_idxs: nb.typed.List,
    column_idxs: nb.typed.List,
    values: nb.typed.List,
    positions: nb.typed.List,
    window_size: int,
    word_length: int,
    alphabet_size_mean: int,
    alphabet_size_slope: int = 0,
    stride: int = 1,
    dilation: int = 1,
    use_signal_id: bool = True,
    window_start_idx: int = 0,
    store_word_position: bool = False,
    min_window_to_signal_std_ratio: float = 0,
    prefix: str = "",
    signal_separator: str = ";",
    padding: str = "valid",
    normalize_flatten_to_zero: bool = False,
    dict_key_type=nb.types.string,
    dict_value_type=nb.types.int64,
    feature_counter: int = 0,
):
    bins_mean = get_norm_bins(alphabet_size_mean, 0, 1)
    if alphabet_size_slope != 0:
        bins_slope = get_norm_bins(alphabet_size_slope, 0, 1)
    else:
        bins_slope = np.empty(0)
    n_ts = len(X)
    n_signals = len(X[0])
    step = (window_size - 1) * dilation + 1
    words_dict = nb.typed.Dict.empty(key_type=dict_key_type, value_type=dict_value_type)
    for ts_idx in range(n_ts):
        for signal_idx in range(n_signals):
            previous_sax_word = ""  # reset the previous sax word
            current_word_reps_counter = 0  # reset the counter
            current_word_pos_counter = 0  # reset the position counter
            signal = np.asarray(X[ts_idx][signal_idx])
            signal_std = np.nanstd(signal)
            if not is_valid_windowing(
                sequence_size=signal.size,
                window_size=window_size,
                dilation=dilation,
            ):
                continue
            if word_length > window_size:  # FIXME
                continue
            if padding != "valid":
                signal = pad(
                    array=signal,
                    left_pad=int((window_size / word_length) / 2),
                    right_pad=int((window_size / word_length) / 2),
                    mode=padding,
                )
            for j in np.arange(
                start=window_start_idx,
                stop=signal.size
                - window_size
                - ((window_size - 1) * (dilation - 1))
                + 1,
                step=stride,
            ):
                start = j
                end = start + step
                window_idx = np.arange(
                    start=start, stop=end, step=dilation, dtype=np.int_
                )
                if is_empty(window_idx):
                    continue
                window = np.asarray(X[ts_idx][signal_idx])[window_idx]
                window = normalize(
                    window=window,
                    signal_std=signal_std,
                    min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
                    flatten_to_zero=normalize_flatten_to_zero,
                )
                sax_word = apply_sax(
                    window=window,
                    word_length=word_length,
                    alphabet_size_slope=alphabet_size_slope,
                    bins_mean=bins_mean,
                    bins_slope=bins_slope,
                    prefix=prefix,
                    signal_idx=signal_idx,
                    signal_separator=signal_separator,
                    use_signal_id=use_signal_id,
                )
                if sax_word not in words_dict:
                    words_dict[sax_word] = feature_counter
                    feature_counter += 1
                if previous_sax_word == "":
                    current_word_reps_counter += 1
                    current_word_pos_counter += (window_idx[-1] + window_idx[0]) / 2
                elif sax_word == previous_sax_word:
                    current_word_reps_counter += 1
                else:
                    if previous_sax_word in words_dict:
                        update_lists(
                            word_reps_count=current_word_reps_counter,
                            word_pos_count=current_word_pos_counter,
                            row_idxs=row_idxs,
                            values=values,
                            positions=positions,
                            column_idxs=column_idxs,
                            sax_word=previous_sax_word,
                            sax_words=words_dict,
                            ts_idx=ts_idx,
                            store_word_position=store_word_position,
                        )
                    current_word_reps_counter = 1
                    current_word_pos_counter = (window_idx[-1] + window_idx[0]) / 2
                previous_sax_word = sax_word
            if previous_sax_word in words_dict:
                update_lists(
                    word_reps_count=current_word_reps_counter,
                    word_pos_count=current_word_pos_counter,
                    row_idxs=row_idxs,
                    values=values,
                    positions=positions,
                    column_idxs=column_idxs,
                    sax_word=previous_sax_word,
                    sax_words=words_dict,
                    ts_idx=ts_idx,
                    store_word_position=store_word_position,
                )
    return words_dict


@njit
def update_lists(
    word_reps_count,
    word_pos_count,
    row_idxs,
    column_idxs,
    values,
    positions,
    sax_word,
    sax_words,
    ts_idx,
    store_word_position,
):
    row_idxs.append(ts_idx)
    column_idxs.append(sax_words[sax_word])
    values.append(word_reps_count)
    if store_word_position:
        positions.append(word_pos_count / word_reps_count)
