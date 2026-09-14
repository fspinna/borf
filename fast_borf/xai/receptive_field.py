import numpy as np

from fast_borf.core.words import decode_word


class ReceptiveField:
    """Where one BORF feature (a SAX word of one signal, in one configuration) occurs.

    For series i, alignments_indices[i] has shape (n_occurrences,
    word_length, window_size // word_length): for each window where the word
    occurs, the positions in the series of the points averaged in each
    segment. alignments[i] and mappings[i] are the timestamps and values at
    those positions. feature_values, feature_importance and
    feature_importance_norm hold the feature's value and importances for each
    series.
    """

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
        feature_importance_norm=None,
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
        self.feature_importance_norm = feature_importance_norm
        self.alignments = alignments
        self.mappings = mappings
        self.alignments_indices = alignments_indices
        self.class_labels = class_labels
        self.signal_labels = signal_labels
        self.contains_time_idx = contains_time_idx

        # Offsets of a window's points from its first point.
        self.plot_idx = np.arange(self.window_size * self.dilation, step=self.dilation)
        self.word_array = decode_word(
            self.compressed_word_int, self.alphabet_size, self.word_length
        )

    def __str__(self):
        return (
            f"{self.word_array} - (signal:{self.signal_idx}, "
            f"window_size:{self.window_size}, word_length:{self.word_length}, "
            f"dilation:{self.dilation}, stride:{self.stride})"
        )
