import numpy as np
import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from fast_borf.bop_utils import (
    array_to_int,
    int_to_array_new_base,
    int_to_sax_words,
    sax_words_to_int,
    separate_timestamps_from_panel,
)
from fast_borf.utils import convert_to_base_10, set_n_jobs_numba
from fast_borf.weighted.borf_weighted import transform_sax_patterns


class BorfSaxSingleTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size=4,
        dilation=1,
        alphabet_size: int = 3,
        word_length: int = 2,
        stride: int = 1,
        min_window_to_signal_std_ratio: float = 0.0,
        n_jobs: int = 1,
        prefix="",
        contains_time_idx=True,
    ):
        self.window_size = window_size
        self.dilation = dilation
        self.word_length = word_length
        self.stride = stride
        self.alphabet_size = alphabet_size
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.prefix = prefix
        self.n_jobs = n_jobs
        self.n_words = convert_to_base_10(
            array_to_int(np.full(self.word_length, self.alphabet_size - 1)) + 1,
            base=self.alphabet_size,
        )
        self.contains_time_idx = contains_time_idx
        set_n_jobs_numba(n_jobs=self.n_jobs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X, timestamps = separate_timestamps_from_panel(
            X, contains_time_idx=self.contains_time_idx
        )

        shape_ = (len(X), len(X[0]), self.n_words)
        out = transform_sax_patterns(
            panel=X,
            panel_timestamps=timestamps,
            window_size=self.window_size,
            dilation=self.dilation,
            alphabet_size=self.alphabet_size,
            word_length=self.word_length,
            stride=self.stride,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
        )
        # ts_idx, signal_idx, words, count
        return sparse.COO(coords=out[:, :3].T, data=out[:, -1].T, shape=shape_)

    def out_to_in_map(self, features):
        signal_idxs = np.array([key[0] for key in features])
        words = np.array([key[1] for key in features])
        sax_words = int_to_sax_words(
            numbers=words, word_length=self.word_length, base=self.alphabet_size
        )
        in_features = list()
        for signal_idx, word_array in zip(signal_idxs, sax_words):
            in_features.append((signal_idx, tuple(word_array.tolist())))
        return in_features

    def in_to_out_map(self, features):
        signal_idxs = np.array([key[0] for key in features])
        sax_words = np.array([key[1] for key in features])
        words = sax_words_to_int(arrays=sax_words, base=self.alphabet_size)
        out_features = list()
        for signal_idx, word in zip(signal_idxs, words):
            out_features.append((signal_idx, word))
        return out_features
