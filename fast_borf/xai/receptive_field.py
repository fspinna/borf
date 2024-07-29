from fast_borf.xai.utils import int_to_array_new_base
import numpy as np

class ReceptiveField:
    def __init__(
            self,
            compressed_word_int,
            signal_idx, word_length, window_size, dilation, stride, alphabet_size,
            min_window_to_signal_std_ratio, conf_idx=None, feature_idx=None, feature_values=None, alignments=None,
            mappings=None, feature_importance=None, class_labels=None, signal_labels=None, **kwargs
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
        self.class_labels = class_labels
        self.signal_labels = signal_labels

        self.plot_idx = np.arange(self.window_size * self.dilation, step=self.dilation)
        self.word_array = int_to_array_new_base(self.compressed_word_int, self.alphabet_size, self.word_length)




