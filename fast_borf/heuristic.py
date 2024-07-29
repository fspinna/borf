import itertools
import numpy as np
from fast_borf.utils import is_valid_windowing, is_empty
from typing import Literal


def get_borf_params(
    time_series_min_length,
    time_series_max_length,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    window_size_power=2,
    word_lengths_n_word_lengths=4,
    strides_n_strides=1,
    alphabets_mean_min_symbols=2,
    alphabets_mean_max_symbols=3,
    alphabets_mean_step=1,
    alphabets_slope_min_symbols=3,
    alphabets_slope_max_symbols=4,
    alphabets_slope_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
):
    params = {}
    params["window_sizes"] = get_window_sizes(
        m_min=time_series_min_length,
        m_max=time_series_max_length,
        min_window_size=window_size_min_window_size,
        max_window_size=window_size_max_window_size,
        power=window_size_power,
    ).tolist()

    params["word_lengths"] = get_word_lengths(
        n_word_lengths=word_lengths_n_word_lengths,
        start=0,
    ).tolist()

    params["dilations"] = get_dilations(
        max_length=time_series_max_length,
        min_dilation=dilations_min_dilation,
        max_dilation=dilations_max_dilation,
    ).tolist()
    params["strides"] = get_strides(n_strides=strides_n_strides).tolist()
    params["alphabet_sizes_slope"] = get_alphabet_sizes(
        min_symbols=alphabets_slope_min_symbols,
        max_symbols=alphabets_slope_max_symbols,
        step=alphabets_slope_step,
    ).tolist()
    params["alphabet_sizes_mean"] = get_alphabet_sizes(
        min_symbols=alphabets_mean_min_symbols,
        max_symbols=alphabets_mean_max_symbols,
        step=alphabets_mean_step,
    ).tolist()
    return params


def get_window_sizes(m_min, m_max, min_window_size=4, max_window_size=None, power=2):
    if max_window_size is None:
        max_window_size = m_max
    m = 2
    windows = list()
    windows_min = list()
    while m <= max_window_size:
        if m < min_window_size:
            windows_min.append(m)
        else:
            windows.append(m)
        m = int(m * power)
    windows = np.array(windows)
    windows_min = np.array(windows_min[1:])
    if not is_empty(windows_min):
        if m_min <= windows_min.max() * power:
            windows = np.concatenate([windows_min, windows])
    return windows.astype(int)


def get_word_lengths(n_word_lengths=4, start=0):
    return np.array([2**i for i in range(start, n_word_lengths + start)])


def get_dilations(max_length, min_dilation=1, max_dilation=None):
    dilations = list()
    max_length_log2 = np.log2(max_length)
    if max_dilation is None:
        max_dilation = max_length_log2
    start = min_dilation
    while start <= max_dilation:
        dilations.append(start)
        start *= 2
    return np.array(dilations)


def get_strides(n_strides=1):
    return np.arange(1, n_strides + 1)


def get_alphabet_sizes(min_symbols=3, max_symbols=4, step=1):
    return np.arange(min_symbols, max_symbols, step)


def get_stride_logarithmic(m, w, d):
    A = (-w + d * w + m * w - d * w**2) / (-1 + m * np.log(m))
    B = (1 - w + d * w + m * w - d * w**2) / m
    cond_1 = (w == 1) and (m > 1) and (d == 1) and (np.log(m) > 1)
    cond_2 = (w == 1) and (m > 1) and (d > 1) and (np.log(m) > 1)
    cond_3 = (w >= 2) and (m > w) and (1 <= d < ((m - 1) / w - 1)) and (np.log(m) >= B)
    cond_4 = (w >= 2) and (m > w) and (1 <= d < ((m - 1) / w - 1)) and (np.log(m) < B)
    if cond_1 or cond_2 or cond_3:
        return 1
    elif cond_4:
        return np.ceil(A)
    else:
        raise ValueError("Invalid parameters")


def generate_sax_parameters_configurations(
    window_sizes,
    strides,
    dilations,
    word_lengths,
    alphabet_sizes_mean,
    alphabet_sizes_slope,
):
    parameters = list(
        itertools.product(
            *[
                window_sizes,
                strides,
                dilations,
                word_lengths,
                alphabet_sizes_mean,
                alphabet_sizes_slope,
            ]
        )
    )
    cleaned_parameters = list()
    for parameter in parameters:
        (
            window_size,
            stride,
            dilation,
            word_length,
            alphabet_mean,
            alphabet_slope,
        ) = extract_parameters_from_args(parameter)
        if word_length > window_size:  # word_length cannot be greater than window_size
            continue
        if (
            alphabet_slope <= 1 and word_length == 1
        ):  # if alphabet_slope <= 1, word_length=1 is useless
            continue
        cleaned_parameters.append(
            dict(
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                word_length=word_length,
                alphabet_size_mean=alphabet_mean,
                alphabet_size_slope=alphabet_slope,
            )
        )
    return cleaned_parameters


def clean_sax_parameters_configurations(parameters, max_length):
    cleaned_parameters = list()
    for parameter in parameters:
        window_size = parameter["window_size"]
        dilation = parameter["dilation"]
        word_length = parameter["word_length"]
        alphabet_mean = parameter["alphabet_size_mean"]
        alphabet_slope = parameter["alphabet_size_slope"]
        stride = parameter["stride"]
        if not is_valid_windowing(
            window_size=window_size, sequence_size=max_length, dilation=dilation
        ):
            continue
        # if window_size == word_length:
        #     continue  # FIXME: should I do this?
        cleaned_parameters.append(
            dict(
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                word_length=word_length,
                alphabet_size_mean=alphabet_mean,
                alphabet_size_slope=alphabet_slope,
            )
        )
    return cleaned_parameters


def sax_parameters_configurations_log_strides(parameters, min_length):
    new_parameters = list()
    for parameter in parameters:
        window_size = parameter["window_size"]
        dilation = parameter["dilation"]
        word_length = parameter["word_length"]
        alphabet_mean = parameter["alphabet_size_mean"]
        alphabet_slope = parameter["alphabet_size_slope"]
        try:
            stride = get_stride_logarithmic(m=min_length, w=window_size, d=dilation)
            parameter = dict(
                window_size=window_size,
                stride=stride,
                dilation=dilation,
                word_length=word_length,
                alphabet_size_mean=alphabet_mean,
                alphabet_size_slope=alphabet_slope,
            )
        except ValueError:
            continue
        new_parameters.append(parameter)
    return new_parameters


def sax_parameters_configurations_linear_strides(parameters):
    new_parameters = list()
    for parameter in parameters:
        window_size = parameter["window_size"]
        dilation = parameter["dilation"]
        word_length = parameter["word_length"]
        alphabet_mean = parameter["alphabet_size_mean"]
        alphabet_slope = parameter["alphabet_size_slope"]
        parameter = dict(
            window_size=window_size,
            stride=word_length,
            dilation=dilation,
            word_length=word_length,
            alphabet_size_mean=alphabet_mean,
            alphabet_size_slope=alphabet_slope,
        )
        new_parameters.append(parameter)
    return new_parameters


def extract_parameters_from_args(parameter):
    window_size = parameter[0]
    stride = parameter[1]
    dilation = parameter[2]
    word_length = parameter[3]
    alphabet_mean = parameter[4]
    alphabet_slope = parameter[5]
    return window_size, stride, dilation, word_length, alphabet_mean, alphabet_slope


def heuristic_function(
    time_series_min_length,
    time_series_max_length,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    window_size_power=2,
    word_lengths_n_word_lengths=3,
    strides_n_strides=1,
    alphabets_mean_min_symbols=2,
    alphabets_mean_max_symbols=3,
    alphabets_mean_step=1,
    alphabets_slope_min_symbols=2,
    alphabets_slope_max_symbols=3,
    alphabets_slope_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
    complexity: Literal["quadratic", "linear_logarithmic", "linear"] = "quadratic",
):
    params = get_borf_params(
        time_series_min_length=time_series_min_length,
        time_series_max_length=time_series_max_length,
        window_size_min_window_size=window_size_min_window_size,
        window_size_max_window_size=window_size_max_window_size,
        window_size_power=window_size_power,
        word_lengths_n_word_lengths=word_lengths_n_word_lengths,
        strides_n_strides=strides_n_strides,
        alphabets_slope_min_symbols=alphabets_slope_min_symbols,
        alphabets_slope_max_symbols=alphabets_slope_max_symbols,
        alphabets_slope_step=alphabets_slope_step,
        alphabets_mean_min_symbols=alphabets_mean_min_symbols,
        alphabets_mean_max_symbols=alphabets_mean_max_symbols,
        alphabets_mean_step=alphabets_mean_step,
        dilations_min_dilation=dilations_min_dilation,
        dilations_max_dilation=dilations_max_dilation,
    )

    params_list = generate_sax_parameters_configurations(
        window_sizes=params["window_sizes"],
        strides=params["strides"],
        dilations=params["dilations"],
        word_lengths=params["word_lengths"],
        alphabet_sizes_mean=params["alphabet_sizes_mean"],
        alphabet_sizes_slope=params["alphabet_sizes_slope"],
    )

    cleaned_params_list = clean_sax_parameters_configurations(
        parameters=params_list, max_length=time_series_max_length
    )

    if complexity == "linear_logarithmic":
        # FIXME: this creates problems when time_series_min_length != time_series_max_length. The only fix I see is
        #  to divide time series signals by size. This is a problem only when the size differs a lot.
        cleaned_params_list = sax_parameters_configurations_log_strides(
            parameters=cleaned_params_list, min_length=time_series_min_length
        )
    elif complexity == "linear":
        cleaned_params_list = sax_parameters_configurations_linear_strides(
            parameters=cleaned_params_list
        )

    return cleaned_params_list


def heuristic_function_sax(
    time_series_min_length,
    time_series_max_length,
    window_size_min_window_size=4,
    window_size_max_window_size=None,
    window_size_power=2,
    word_lengths_n_word_lengths=4,
    strides_n_strides=1,
    alphabets_min_symbols=2,
    alphabets_max_symbols=3,
    alphabets_step=1,
    dilations_min_dilation=1,
    dilations_max_dilation=None,
    complexity: Literal["quadratic", "linear_logarithmic", "linear"] = "quadratic",
):
    configs = heuristic_function(
        time_series_min_length=time_series_min_length,
        time_series_max_length=time_series_max_length,
        window_size_min_window_size=window_size_min_window_size,
        window_size_max_window_size=window_size_max_window_size,
        window_size_power=window_size_power,
        word_lengths_n_word_lengths=word_lengths_n_word_lengths,
        strides_n_strides=strides_n_strides,
        alphabets_slope_min_symbols=0,
        alphabets_slope_max_symbols=1,
        alphabets_slope_step=1,
        alphabets_mean_min_symbols=alphabets_min_symbols,
        alphabets_mean_max_symbols=alphabets_max_symbols,
        alphabets_mean_step=alphabets_step,
        dilations_min_dilation=dilations_min_dilation,
        dilations_max_dilation=dilations_max_dilation,
        complexity=complexity,
    )
    for config in configs:
        config.pop("alphabet_size_slope")
        config["alphabet_size"] = config.pop("alphabet_size_mean")
    return configs





if __name__ == "__main__":
    configs = heuristic_function(20, 100, complexity="quadratic")
    configs_sax = heuristic_function_sax(20, 100, complexity="quadratic")
    configs_sax_linear = heuristic_function_sax(20, 100, complexity="linear")
