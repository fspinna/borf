import numpy as np
from borf.algorithms.borf_transform import normalize, apply_sax
from borf.utils.condition_utils import is_valid_windowing, is_empty
from borf.utils.transform_utils import get_norm_bins, pad


def get_signal_alignment_idxs_from_word(
    signal,
    sax_query,
    window_size,
    dilation,
    stride,
    word_length,
    alphabet_size_slope,
    alphabet_size_mean,
    padding,
    min_window_to_signal_std_ratio,
    normalize_flatten_to_zero,
    signal_idx,
    prefix,
    signal_separator,
    use_signal_id,
    window_start_idx=0,
    **kwargs
):
    idxs = list()
    bins_mean = get_norm_bins(alphabet_size_mean, 0, 1)
    if alphabet_size_slope != 0:
        bins_slope = get_norm_bins(alphabet_size_slope, 0, 1)
    else:
        bins_slope = np.empty(0)
    step = (window_size - 1) * dilation + 1
    signal_std = np.nanstd(signal)
    if not is_valid_windowing(
        sequence_size=signal.size,
        window_size=window_size,
        dilation=dilation,
    ):
        return list()
        # FIXME: there are some problems here probably. The problem is when you have a different-sized time series,
        #  so the windowing can be valid for some signals and not for others. I think this fix it but I am not 100%
        #  sure.
    if padding != "valid":
        signal = pad(
            array=signal,
            left_pad=int((window_size / word_length) / 2),
            right_pad=int((window_size / word_length) / 2),
            mode=padding,
        )
    for j in np.arange(
        start=window_start_idx,
        stop=signal.size - window_size - ((window_size - 1) * (dilation - 1)) + 1,
        step=stride,
    ):
        start = j
        end = start + step
        window_idx = np.arange(start=start, stop=end, step=dilation, dtype=np.int_)
        if is_empty(window_idx):
            continue
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
        if sax_word == sax_query:
            idxs.append(window_idx)
    return idxs


def get_ts_alignment_idxs_from_word(
    X,
    sax_query,
    window_size,
    dilation,
    stride,
    word_length,
    alphabet_size_slope,
    alphabet_size_mean,
    padding,
    min_window_to_signal_std_ratio,
    normalize_flatten_to_zero,
    prefix,
    signal_separator,
    use_signal_id,
    window_start_idx=0,
    **kwargs
):
    _, signal_idx, _ = sax_query.split(signal_separator)
    signal_idx = int(signal_idx)
    signal = np.asarray(X[0][signal_idx])
    return signal_idx, get_signal_alignment_idxs_from_word(
        signal=signal,
        sax_query=sax_query,
        window_size=window_size,
        dilation=dilation,
        stride=stride,
        word_length=word_length,
        alphabet_size_slope=alphabet_size_slope,
        alphabet_size_mean=alphabet_size_mean,
        padding=padding,
        window_start_idx=window_start_idx,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        normalize_flatten_to_zero=normalize_flatten_to_zero,
        signal_idx=signal_idx,
        prefix=prefix,
        signal_separator=signal_separator,
        use_signal_id=use_signal_id,
    )
