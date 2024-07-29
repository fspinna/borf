from sklearn.base import TransformerMixin, BaseEstimator

import numpy as np
from fast_borf.bag_of_patterns.borf_sax import (
    transform_sax_patterns,  # change to the old one if needed
    array_to_int, convert_to_base_10
)
from fast_borf.utils import set_n_jobs_numba
import sparse


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
    ):
        self.window_size = window_size
        self.dilation = dilation
        self.word_length = word_length
        self.stride = stride
        self.alphabet_size = alphabet_size
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.prefix = prefix
        self.n_jobs = n_jobs
        self.n_words = convert_to_base_10(array_to_int(np.full(self.word_length, self.alphabet_size - 1)) + 1,
                                          base=self.alphabet_size)
        set_n_jobs_numba(n_jobs=self.n_jobs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        shape_ = (
            len(X),
            len(X[0]),
            self.n_words
        )
        out = transform_sax_patterns(
            panel=X,
            window_size=self.window_size,
            dilation=self.dilation,
            alphabet_size=self.alphabet_size,
            word_length=self.word_length,
            stride=self.stride,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
        )
        # ts_idx, signal_idx, words, count
        return sparse.COO(
            coords=out[:, :3].T,
            data=out[:, -1].T,
            shape=shape_
        )
