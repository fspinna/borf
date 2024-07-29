import numpy as np
import numba as nb
from fast_borf.symbolic_aggregate_approximation.symbolic_aggregate_approximation_clean import sax
from fast_borf.utils import get_norm_bins, are_window_size_and_dilation_compatible_with_signal_length
from fast_borf.xai.utils import int_to_sax_words, sax_words_to_int


@nb.njit
def wsax_matrix_position_to_indices(i, j, dilation, stride, segment_size):
    out = np.full(segment_size, -1, dtype=np.int_)
    for k in range(segment_size):
        out[k] = (i * stride) + (j * dilation * segment_size) + (k * dilation)
    return out


@nb.njit
def wsax_matrix_row_position_to_indices(i, dilation, stride, word_length, segment_size):
    out = np.full((word_length, segment_size), -1, dtype=np.int_)
    for j in range(word_length):
        out[j] = wsax_matrix_position_to_indices(i, j, dilation, stride, segment_size)
    return out


@nb.njit(cache=True)
def wsax_signal_alignment_conversion(
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
    sax_words_int = sax_words_to_int(sax_words, alphabet_size)
    sax_conversion = nb.typed.Dict.empty(
        key_type=nb.types.uint64,
        value_type=nb.types.int64[:, :, :],
    )
    for i in range(len(sax_words)):
        word_int = sax_words_int[i]
        if word_int not in sax_conversion:
            sax_conversion[word_int] = wsax_matrix_row_position_to_indices(
                i=i,
                dilation=dilation,
                stride=stride,
                word_length=word_length,
                segment_size=window_size // word_length
            )[np.newaxis, :, :]
        else:
            stack = np.vstack((
                sax_conversion[word_int],
                wsax_matrix_row_position_to_indices(
                    i=i,
                    dilation=dilation,
                    stride=stride,
                    word_length=word_length,
                    segment_size=window_size // word_length
                )[np.newaxis, :, :]
            ))
            sax_conversion[word_int] = stack
    return sax_conversion


# @nb.njit
def wsax_panel_alignment_conversion(
        panel: np.ndarray,
        window_size,
        word_length,
        alphabet_size,
        dilation,
        stride=1,
        min_window_to_signal_std_ratio=0.0,
        **kwargs
):
    panel_conversion = list()
    bins = get_norm_bins(alphabet_size=alphabet_size)
    for j in range(len(panel)):
        sax_conversion = list()
        for i in range(len(panel[j])):
            signal = np.asarray(panel[j][i])
            signal = signal[~np.isnan(signal)]
            if not are_window_size_and_dilation_compatible_with_signal_length(
                    window_size, dilation, signal.size
            ):
                # sax_conversion.append(
                #     nb.typed.Dict.empty(
                #         key_type=nb.types.uint64,
                #         value_type=nb.types.int64[:, :, :],
                #     )
                # )
                sax_conversion.append(
                    dict()
                )
                continue
            sax_conversion_ = wsax_signal_alignment_conversion(
                a=signal,
                window_size=window_size,
                word_length=word_length,
                bins=bins,
                alphabet_size=alphabet_size,
                dilation=dilation,
                stride=stride,
                min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            )
            # sax_conversion.append(sax_conversion_)
            sax_conversion.append(dict(sax_conversion_))
        panel_conversion.append(sax_conversion)
    return panel_conversion


def wsax_configurations_alignment_conversion(
        panel: np.ndarray,
        configurations: list[dict],
):
    configurations_conversion = list()
    for i in range(len(configurations)):
        configurations_conversion.append(
            wsax_panel_alignment_conversion(
                panel=panel,
                **configurations[i]
            )
        )
    # shape: (n_conf, n_ts, n_signals) each signal is a dict with keys: word_int, value: np.ndarray
    return configurations_conversion



@nb.njit
def align_sax_word_to_sax_converted_signal(
        sax_signal: np.ndarray,
        sax_word: np.ndarray,
        dilation: int,
        stride: int,
        segment_size: int,
):
    word_length = len(sax_word)
    matches = list()
    for i in range(len(sax_signal)):
        word = sax_signal[i]
        if np.array_equal(word, sax_word):
            matches.append(i)
    out = np.full((len(matches), word_length, segment_size), -1, dtype=np.int_)
    for i in range(len(matches)):
        out[i] = wsax_matrix_row_position_to_indices(matches[i], dilation, stride, word_length, segment_size)
    return out


# @nb.njit
# def align_sax_words_to_sax_converted_signal(
#         sax_signal: np.ndarray,
#         sax_words: np.ndarray,
#         dilation: int,
#         stride: int,
#         segment_size: int,
# ):
#     out = list()
#     for sax_word in sax_words:
#         out.append(align_sax_word_to_sax_converted_signal(sax_signal, sax_word, dilation, stride, segment_size))
#     return out
#
#
# @nb.njit
# def align_sax_words_to_raw_signal(
#         signal: np.ndarray,
#         sax_words: np.ndarray,
#         dilation: int,
#         stride: int,
#         segment_size: int,
#         window_size: int,
#         bins: np.ndarray,
#         min_window_to_signal_std_ratio=0.0,
# ):
#     sax_converted_signal = sax(
#         a=signal,
#         window_size=window_size,
#         word_length=sax_words.shape[1],
#         bins=bins,
#         min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
#         dilation=dilation,
#         stride=stride,
#     )
#     return align_sax_words_to_sax_converted_signal(sax_converted_signal, sax_words, dilation, stride, segment_size)


@nb.njit
def align_sax_words_to_raw_ts(
        ts: np.ndarray,
        sax_words: np.ndarray,  # shape: (n_words,) each word is an integer
        signal_idxs: np.ndarray,  # shape: (n_words,) each signal_idx is an integer that points to a signal in ts
        dilation: int,
        stride: int,
        word_length: int,
        window_size: int,
        alphabet_size: int,
        min_window_to_signal_std_ratio=0.0,
    ):
    sax_words_ = int_to_sax_words(sax_words, alphabet_size, word_length)
    segment_size = window_size // word_length
    sax_converted_ts = sax_ts(ts, window_size, word_length, alphabet_size, dilation, stride, min_window_to_signal_std_ratio)
    out = list()
    for i in range(len(sax_words_)):
        signal_idx = signal_idxs[i]
        sax_word = sax_words_[i]
        out.append(align_sax_word_to_sax_converted_signal(sax_converted_ts[signal_idx], sax_word, dilation, stride, segment_size))
    return out, sax_words_, signal_idxs


@nb.njit
def sax_ts(
        ts: np.ndarray,
        window_size: int,
        word_length: int,
        alphabet_size: int,
        dilation: int,
        stride: int,
        min_window_to_signal_std_ratio=0.0,
):
    bins = get_norm_bins(alphabet_size=alphabet_size)
    out = list()
    for i in range(len(ts)):
        signal = np.asarray(ts[i])
        signal = signal[~np.isnan(signal)]
        if not are_window_size_and_dilation_compatible_with_signal_length(
                window_size, dilation, signal.size
        ):
            out.append(np.full((0, word_length), -1, dtype=np.uint8))
        out.append(sax(
            a=signal,
            window_size=window_size,
            word_length=word_length,
            bins=bins,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            dilation=dilation,
            stride=stride,
        ))
    return out




def dict_test():
    sax_conversion = nb.typed.Dict.empty(
        key_type=nb.types.uint64,
        value_type=nb.types.Array(nb.types.uint64, 3, 'C'),
    )
    sax_conversion[0] = np.random.randint(0, 2, (3, 3, 3), dtype=np.uint64)
    return sax_conversion


@nb.njit
def dict_test2():
    sax_conversion = dict()
    sax_conversion[0] = np.full((3,), 1, dtype=np.uint64)
    a = np.append(sax_conversion[0].ravel(), sax_conversion[0].ravel())
    sax_conversion[0] = a
    return sax_conversion


def dict_test3():
    sax_conversion = dict()
    sax_conversion[0] = np.full((1, 3, 3), 1, dtype=np.uint64)
    a = np.vstack([sax_conversion[0], sax_conversion[0]])
    sax_conversion[0] = a
    return sax_conversion



if __name__ == "__main__":

    out = dict_test3()

    #
    # pass
    # sax_words = np.array([1, 10, 11, 0])
    # signal_idxs = np.array([0, 0, 1, 2])
    # ts = np.random.randn(10, 100)
    # panel = [ts, ts]
    # signal = ts[0]
    #
    # word_length = 4
    # alphabet_size = 3
    # window_size = 8
    # dilation = 1
    # stride = 1
    #
    # out = wsax_signal_alignment_conversion(
    #     a=signal,
    #     window_size=window_size,
    #     word_length=word_length,
    #     alphabet_size=alphabet_size,
    #     bins=np.zeros(1),
    #     dilation=dilation,
    #     stride=stride,
    # )
    #
    # out2 = wsax_panel_alignment_conversion(
    #     panel=panel,
    #     window_size=window_size,
    #     word_length=word_length,
    #     alphabet_size=alphabet_size,
    #     bins=np.zeros(1),
    #     dilation=dilation,
    #     stride=stride,
    # )

    #
    # out = align_sax_words_to_raw_ts(
    #     ts, sax_words, signal_idxs, dilation, stride, word_length, window_size, alphabet_size
    # )

    # out = sax_ts(ts, window_size, word_length, alphabet_size, dilation, stride)
    # signal = np.array([
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [13, 14, 15, 16],
    #     [1, 2, 3, 4],
    # ])
    # sax_word = np.array([1, 2, 3, 4])
    # sax_words = np.array([
    #     np.array([1, 2, 3, 4]),
    #     np.array([5, 6, 7, 8]),
    #     np.array([9, 10, 11, 12]),
    #     np.array([13, 14, 15, 16]),
    # ])
    # word_length = len(sax_word)
    # dilation = 1
    # stride = 1
    # segment_size = 2
    # out = align_sax_word_to_sax_converted_signal(signal, sax_word, dilation, stride, segment_size)
    #
    # out2 = align_sax_words_to_sax_converted_signal(signal, sax_words, dilation, stride, segment_size)

    # out = wsax_matrix_position_to_indices(0, 0, dilation, stride, segment_size)
    #
    #
    # out2 = wsax_matrix_row_position_to_indices(0, dilation, stride, word_length, segment_size)
    # print(out2)
