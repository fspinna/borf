import numpy as np

from fast_borf.bop_utils import int_to_array_new_base, separate_timestamps_from_panel
from fast_borf.xai.sax_mapping import wsax_panel_alignment_conversion

# from fast_borf.xai.utils import int_to_array_new_base


class ReceptiveField:
    def __init__(
        self,
        compressed_word_int,
        signal_idx,
        word_length,
        window_size,
        dilation,
        stride,
        alphabet_size,
        min_window_to_signal_std_ratio,
        conf_idx=None,
        feature_idx=None,
        feature_values=None,
        alignments=None,
        mappings=None,
        alignments_indices=None,
        feature_importance=None,
        class_labels=None,
        signal_labels=None,
        contains_time_idx=None,
        **kwargs,
    ):
        self.compressed_word_int = compressed_word_int
        self.signal_idx = signal_idx
        self.word_length = word_length
        self.window_size = window_size
        self.dilation = dilation
        self.stride = stride
        self.alphabet_size = alphabet_size
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.conf_idx = conf_idx
        self.feature_idx = feature_idx
        self.feature_values = feature_values
        self.feature_importance = feature_importance
        self.feature_importance_norm = None
        self.alignments = alignments
        self.mappings = mappings
        self.alignments_indices = alignments_indices
        self.class_labels = class_labels
        self.signal_labels = signal_labels
        self.contains_time_idx = contains_time_idx

        self.plot_idx = np.arange(self.window_size * self.dilation, step=self.dilation)
        self.word_array = int_to_array_new_base(
            self.compressed_word_int, self.alphabet_size, self.word_length
        )

    def __str__(self):
        return (
            f"{self.word_array} - (signal:{self.signal_idx}, window_size:{self.window_size}, word_length:"
            f"{self.word_length}, dilation:{self.dilation}, stride:{self.stride})"
        )

    def align(self, X):
        X, timestamps = separate_timestamps_from_panel(
            X=X, contains_time_idx=self.contains_time_idx
        )
        alignments_indices = wsax_panel_alignment_conversion(
            panel=X,
            panel_timestamps=timestamps,
            window_size=self.window_size,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            dilation=self.dilation,
            stride=self.stride,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
        )
        for i in range(len(alignments_indices)):
            signal = X[i][self.signal_idx]
            is_nan = np.isnan(signal)
            if self.compressed_word_int in alignments_indices[i][self.signal_idx]:
                alignments_indices[i] = alignments_indices[i][self.signal_idx][
                    self.compressed_word_int
                ]
                alignments_indices[i] = np.where(~is_nan)[0][alignments_indices[i]]
            else:
                alignments_indices[i] = np.empty(
                    (0, self.word_length, self.window_size // self.word_length),
                    dtype=np.int_,
                )
        return alignments_indices
