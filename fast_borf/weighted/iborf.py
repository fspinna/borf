from fast_borf.pipeline.reshaper import ReshapeTo2D
from fast_borf.pipeline.to_scipy import ToScipySparse
from fast_borf.pipeline.zero_columns_remover import ZeroColumnsRemover
from fast_borf.weighted.borf_multi import build_pipeline_auto
from sklearn.base import BaseEstimator, TransformerMixin


class IBORF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size_min_window_size=4,
        window_size_max_window_size=None,
        word_lengths_n_word_lengths=4,
        alphabets_min_symbols=3,
        alphabets_max_symbols=4,
        alphabets_step=1,
        dilations_min_dilation=1,
        dilations_max_dilation=None,
        min_window_to_signal_std_ratio: float = 0.0,
        n_jobs=1,
        n_jobs_numba=1,
        transformer_weights=None,
        contains_time_idx=True,
):
        self.window_size_min_window_size = window_size_min_window_size
        self.window_size_max_window_size = window_size_max_window_size
        self.word_lengths_n_word_lengths = word_lengths_n_word_lengths
        self.alphabets_min_symbols = alphabets_min_symbols
        self.alphabets_max_symbols = alphabets_max_symbols
        self.alphabets_step = alphabets_step
        self.dilations_min_dilation = dilations_min_dilation
        self.dilations_max_dilation = dilations_max_dilation
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.n_jobs = n_jobs
        self.n_jobs_numba = n_jobs_numba
        self.transformer_weights = transformer_weights
        self.contains_time_idx = contains_time_idx

        self.time_series_min_length_ = None
        self.time_series_max_length_ = None
        self.configs_ = None

    def fit(self, X, y=None):
        return self._fit(X, y)

    def _fit(self, X, y=None):
        time_series_length = X.shape[2]

        pipeline_objects = [
            (ReshapeTo2D, dict()),
            (ZeroColumnsRemover, dict()),
            (ToScipySparse, dict()),
        ]

        self.pipe_, self.configs_ = build_pipeline_auto(
            time_series_min_length=time_series_length,
            time_series_max_length=time_series_length,
            window_size_min_window_size=self.window_size_min_window_size,
            window_size_max_window_size=self.window_size_max_window_size,
            word_lengths_n_word_lengths=self.word_lengths_n_word_lengths,
            alphabets_min_symbols=self.alphabets_min_symbols,
            alphabets_max_symbols=self.alphabets_max_symbols,
            alphabets_step=self.alphabets_step,
            dilations_min_dilation=self.dilations_min_dilation,
            dilations_max_dilation=self.dilations_max_dilation,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
            n_jobs=self.n_jobs,
            n_jobs_numba=self.n_jobs_numba,
            transformer_weights=self.transformer_weights,
            pipeline_objects=pipeline_objects,
            contains_time_idx=self.contains_time_idx,
        )
        self.pipe_.fit(X)
        return self

    def _transform(self, X, y=None):
        return self.pipe_.transform(X)

    def transform(self, X, y=None):
        return self._transform(X, y)

