from borf.classes.borf_multi import BorfMultiTransformer
from borf.algorithms.borf_heuristic import heuristic_function
from typing import Literal, Optional
import awkward as ak


class BorfAutoTransformer(BorfMultiTransformer):
    def __init__(
        self,
        time_series_min_length: Optional[int] = None,
        time_series_max_length: Optional[int] = None,
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
        complexity: Literal["quadratic", "linear_logarithmic"] = "quadratic",
        strategies: tuple = ("repetitions",),
        use_signal_id=True,
        ignore_equal_contiguous_words=False,
        signal_separator=";",
        parameters_separator="_",
        padding: str = "valid",
        min_window_to_signal_std_ratio: float = 0.15,
        word_position_strategy: str = "average",
        normalize_flatten_to_zero: bool = True,
        prefix: str = "",
        n_jobs=1,
    ):
        self.time_series_min_length = time_series_min_length
        self.time_series_max_length = time_series_max_length
        self.window_size_min_window_size = window_size_min_window_size
        self.window_size_max_window_size = window_size_max_window_size
        self.window_size_power = window_size_power
        self.word_lengths_n_word_lengths = word_lengths_n_word_lengths
        self.strides_n_strides = strides_n_strides
        self.alphabets_mean_min_symbols = alphabets_mean_min_symbols
        self.alphabets_mean_max_symbols = alphabets_mean_max_symbols
        self.alphabets_mean_step = alphabets_mean_step
        self.alphabets_slope_min_symbols = alphabets_slope_min_symbols
        self.alphabets_slope_max_symbols = alphabets_slope_max_symbols
        self.alphabets_slope_step = alphabets_slope_step
        self.dilations_min_dilation = dilations_min_dilation
        self.dilations_max_dilation = dilations_max_dilation
        self.complexity = complexity
        self.configs = None
        if time_series_min_length is not None and time_series_max_length is not None:
            self.configs = heuristic_function(
                time_series_min_length=self.time_series_min_length,
                time_series_max_length=self.time_series_max_length,
                window_size_min_window_size=self.window_size_min_window_size,
                window_size_max_window_size=self.window_size_max_window_size,
                window_size_power=self.window_size_power,
                word_lengths_n_word_lengths=self.word_lengths_n_word_lengths,
                strides_n_strides=self.strides_n_strides,
                alphabets_mean_min_symbols=self.alphabets_mean_min_symbols,
                alphabets_mean_max_symbols=self.alphabets_mean_max_symbols,
                alphabets_mean_step=self.alphabets_mean_step,
                alphabets_slope_min_symbols=self.alphabets_slope_min_symbols,
                alphabets_slope_max_symbols=self.alphabets_slope_max_symbols,
                alphabets_slope_step=self.alphabets_slope_step,
                dilations_min_dilation=self.dilations_min_dilation,
                dilations_max_dilation=self.dilations_max_dilation,
                complexity=self.complexity,
            )
        self.strategies = strategies
        self.use_signal_id = use_signal_id
        self.ignore_equal_contiguous_words = ignore_equal_contiguous_words
        self.signal_separator = signal_separator
        self.parameters_separator = parameters_separator
        self.padding = padding
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.word_position_strategy = word_position_strategy
        self.normalize_flatten_to_zero = normalize_flatten_to_zero
        self.prefix = prefix
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.time_series_max_length = ak.max(ak.ravel(ak.count(X, axis=2)))
        self.time_series_min_length = ak.min(ak.ravel(ak.count(X, axis=2)))
        if self.configs is None:
            self.configs = heuristic_function(
                time_series_min_length=self.time_series_min_length,
                time_series_max_length=self.time_series_max_length,
                window_size_min_window_size=self.window_size_min_window_size,
                window_size_max_window_size=self.window_size_max_window_size,
                window_size_power=self.window_size_power,
                word_lengths_n_word_lengths=self.word_lengths_n_word_lengths,
                strides_n_strides=self.strides_n_strides,
                alphabets_mean_min_symbols=self.alphabets_mean_min_symbols,
                alphabets_mean_max_symbols=self.alphabets_mean_max_symbols,
                alphabets_mean_step=self.alphabets_mean_step,
                alphabets_slope_min_symbols=self.alphabets_slope_min_symbols,
                alphabets_slope_max_symbols=self.alphabets_slope_max_symbols,
                alphabets_slope_step=self.alphabets_slope_step,
                dilations_min_dilation=self.dilations_min_dilation,
                dilations_max_dilation=self.dilations_max_dilation,
                complexity=self.complexity,
            )
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        self.time_series_max_length = ak.max(ak.ravel(ak.count(X, axis=2)))
        self.time_series_min_length = ak.min(ak.ravel(ak.count(X, axis=2)))
        if self.configs is None:
            self.configs = heuristic_function(
                time_series_min_length=self.time_series_min_length,
                time_series_max_length=self.time_series_max_length,
                window_size_min_window_size=self.window_size_min_window_size,
                window_size_max_window_size=self.window_size_max_window_size,
                window_size_power=self.window_size_power,
                word_lengths_n_word_lengths=self.word_lengths_n_word_lengths,
                strides_n_strides=self.strides_n_strides,
                alphabets_mean_min_symbols=self.alphabets_mean_min_symbols,
                alphabets_mean_max_symbols=self.alphabets_mean_max_symbols,
                alphabets_mean_step=self.alphabets_mean_step,
                alphabets_slope_min_symbols=self.alphabets_slope_min_symbols,
                alphabets_slope_max_symbols=self.alphabets_slope_max_symbols,
                alphabets_slope_step=self.alphabets_slope_step,
                dilations_min_dilation=self.dilations_min_dilation,
                dilations_max_dilation=self.dilations_max_dilation,
                complexity=self.complexity,
            )
        return super().fit_transform(X, y)
