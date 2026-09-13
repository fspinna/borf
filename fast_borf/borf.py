from contextlib import contextmanager
from typing import Optional, Sequence

import awkward as ak
import numba as nb
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fast_borf.core.transform import transform_sax_patterns
from fast_borf.heuristic import Complexity, generate_configs

MAX_KEY = np.iinfo(np.int64).max


class BORF(TransformerMixin, BaseEstimator):
    """Bag-Of-Receptive-Fields transformer for regular and irregular time series.

    Every configuration (window size, word length, alphabet size, dilation and
    stride) turns each signal into a bag of SAX words. The output has one column
    per (configuration, signal, word) seen during fit, holding the number of
    times the word occurs.

    Parameters
    ----------
    min_window_size, max_window_size : int
        Window sizes are powers of two in this range. max_window_size defaults
        to the length of the longest series.
    max_word_length : int, default=8
        Word lengths are the powers of two from 2 up to max_word_length.
    alphabet_sizes : sequence of int, default=(3,)
        Alphabet sizes to use.
    min_dilation, max_dilation : int
        Dilations are powers of two in this range. max_dilation defaults to
        log2 of the length of the longest series.
    complexity : {"quadratic", "linear", "linear_logarithmic"}, default="quadratic"
        How the stride is chosen, see fast_borf.heuristic.generate_configs.
    configs : list of dict, optional
        Explicit configurations with keys window_size, word_length,
        alphabet_size, dilation and stride. When given, the parameters above
        are ignored.
    min_window_to_signal_std_ratio : float, default=0.0
        Windows whose standard deviation is below this fraction of the
        signal's standard deviation are treated as flat (z-score 0).
    time_channel : bool, default=False
        If True, the last channel of X holds the timestamps of each series and
        the other channels are its signals. If False, observations are evenly
        spaced.
    n_jobs : int, default=1
        Number of numba threads. -1 uses numba's default (all logical cores).

    Attributes
    ----------
    configs_ : list of dict
        Configurations used, in the order of the output columns.
    feature_index_ : ndarray of shape (n_features, 3)
        (configuration index, signal index, word) of every output column. The
        word is the SAX word encoded as an integer in base alphabet_size.
    vocabularies_ : list of ndarray
        For each configuration, the sorted (signal index, word) pairs seen
        during fit, encoded as signal_index * alphabet_size**word_length + word.
    n_signals_ : int
        Number of signals per series seen during fit.

    Notes
    -----
    X is either a float array of shape (n_series, n_channels, n_timestamps),
    with NaN for missing values or padding, or a ragged awkward array with the
    same nesting.
    """

    def __init__(
        self,
        min_window_size: int = 4,
        max_window_size: Optional[int] = None,
        max_word_length: int = 8,
        alphabet_sizes: Sequence[int] = (3,),
        min_dilation: int = 1,
        max_dilation: Optional[int] = None,
        complexity: Complexity = "quadratic",
        configs: Optional[Sequence[dict]] = None,
        min_window_to_signal_std_ratio: float = 0.0,
        time_channel: bool = False,
        n_jobs: int = 1,
    ):
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.max_word_length = max_word_length
        self.alphabet_sizes = alphabet_sizes
        self.min_dilation = min_dilation
        self.max_dilation = max_dilation
        self.complexity = complexity
        self.configs = configs
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.time_channel = time_channel
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        signals, timestamps = self._split_time_channel(X)
        self.n_signals_ = len(signals[0])
        if self.configs is not None:
            self.configs_ = [dict(config) for config in self.configs]
        else:
            min_length, max_length = series_lengths(signals)
            self.configs_ = generate_configs(
                min_length=min_length,
                max_length=max_length,
                min_window_size=self.min_window_size,
                max_window_size=self.max_window_size,
                max_word_length=self.max_word_length,
                alphabet_sizes=self.alphabet_sizes,
                min_dilation=self.min_dilation,
                max_dilation=self.max_dilation,
                complexity=self.complexity,
            )
        if not self.configs_:
            raise ValueError("No configuration fits series of this length")
        for config in self.configs_:
            if self.n_signals_ * n_words(config) > MAX_KEY:
                raise ValueError(f"Too many possible words for configuration {config}")

        words = self._transform_words(signals, timestamps)
        # A column is kept if its (signal, word) occurs at least once during fit.
        self.vocabularies_ = [
            np.unique(word_keys(rows, config))
            for rows, config in zip(words, self.configs_)
        ]
        self.feature_index_ = np.vstack(
            [
                np.column_stack(
                    [
                        np.full(len(keys), i),
                        keys // n_words(config),
                        keys % n_words(config),
                    ]
                )
                for i, (keys, config) in enumerate(
                    zip(self.vocabularies_, self.configs_)
                )
            ]
        ).astype(np.int64)
        return self._to_matrix(words, len(signals))

    def transform(self, X):
        check_is_fitted(self)
        signals, timestamps = self._split_time_channel(X)
        if len(signals[0]) != self.n_signals_:
            raise ValueError(
                f"X has {len(signals[0])} signals per series, BORF was fitted "
                f"with {self.n_signals_}"
            )
        return self._to_matrix(self._transform_words(signals, timestamps), len(signals))

    def _split_time_channel(self, X):
        if not isinstance(X, ak.Array):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim != 3:
                raise ValueError(
                    "X must have shape (n_series, n_channels, n_timestamps), "
                    f"got {X.shape}"
                )
        if self.time_channel:
            return X[:, :-1], X[:, -1:]
        return X, None

    def _transform_words(self, signals, timestamps):
        """Rows of (series index, signal index, word, count) for each configuration."""
        with numba_threads(self.n_jobs):
            return [
                transform_sax_patterns(
                    panel=signals,
                    panel_timestamps=timestamps,
                    min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
                    **config,
                )
                for config in self.configs_
            ]

    def _to_matrix(self, words, n_series):
        rows, cols, counts = [], [], []
        offset = 0
        for config_words, vocabulary, config in zip(
            words, self.vocabularies_, self.configs_
        ):
            if len(vocabulary):
                keys = word_keys(config_words, config)
                position = np.searchsorted(vocabulary, keys)
                in_vocabulary = (
                    vocabulary[np.minimum(position, len(vocabulary) - 1)] == keys
                )
                rows.append(config_words[in_vocabulary, 0])
                cols.append(position[in_vocabulary] + offset)
                counts.append(config_words[in_vocabulary, 3])
            offset += len(vocabulary)
        if not rows:
            return sp.csr_matrix((n_series, offset), dtype=np.int64)
        return sp.csr_matrix(
            (np.concatenate(counts), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_series, offset),
        )


def n_words(config):
    return config["alphabet_size"] ** config["word_length"]


def word_keys(rows, config):
    """Encode (signal index, word) as one integer, ordered by signal then word."""
    return rows[:, 1] * n_words(config) + rows[:, 2]


def series_lengths(signals):
    """Shortest and longest signal length, counting NaN padding."""
    if isinstance(signals, np.ndarray):
        return signals.shape[2], signals.shape[2]
    counts = ak.ravel(ak.count(signals, axis=2))
    return int(ak.min(counts)), int(ak.max(counts))


@contextmanager
def numba_threads(n_jobs):
    previous = nb.get_num_threads()
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS if n_jobs == -1 else n_jobs)
    try:
        yield
    finally:
        nb.set_num_threads(previous)
