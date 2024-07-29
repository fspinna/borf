from sklearn.pipeline import FeatureUnion, make_pipeline
from typing import Sequence, Dict, Optional, Tuple, Literal
from fast_borf.heuristic import heuristic_function_sax
from fast_borf.classes.bag_of_receptive_fields_sax.borf_single import BorfSaxSingleTransformer
import awkward as ak


class BorfPipelineBuilder:
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
        pipeline_objects: Optional[Sequence[Tuple]] = None,
        complexity: Literal['quadratic', "linear"] = 'quadratic',
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
        self.pipeline_objects = pipeline_objects
        self.complexity = complexity

        self.time_series_min_length_ = None
        self.time_series_max_length_ = None
        self.configs_ = None

    def build(self, X):
        self.time_series_max_length_ = ak.max(ak.ravel(ak.count(X, axis=2)))
        self.time_series_min_length_ = ak.min(ak.ravel(ak.count(X, axis=2)))
        pipe, self.configs_ = build_pipeline_auto(
            time_series_min_length=self.time_series_min_length_,
            time_series_max_length=self.time_series_max_length_,
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
            pipeline_objects=self.pipeline_objects,
            complexity=self.complexity,
        )
        return pipe


def build_pipeline(
    configs,
    min_window_to_signal_std_ratio: float = 0.0,
    n_jobs_numba=1,
    n_jobs=1,
    transformer_weights=None,
    pipeline_objects: Optional[Sequence[Tuple]] = None,
):
    transformers = list()
    if pipeline_objects is None:
        pipeline_objects = list()
    for config in configs:
        # alphabet_size, window_size, word_length, dilation, stride
        borf = BorfSaxSingleTransformer(
            **config,
            min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
            n_jobs=n_jobs_numba
        )
        transformer = make_pipeline(borf, *[obj(**kwargs) for obj, kwargs in pipeline_objects])
        transformers.append(transformer)
    union = FeatureUnion(
        transformer_list=[(str(i), transformers[i]) for i in range(len(transformers))],
        n_jobs=n_jobs,
        transformer_weights=transformer_weights,
    )
    return union


def build_pipeline_auto(
        time_series_min_length: int,
        time_series_max_length: int,
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
        pipeline_objects: Optional[Sequence[Tuple]] = None,
        complexity: Literal['quadratic', "linear"] = 'quadratic',
):
    configs = heuristic_function_sax(
        time_series_min_length=time_series_min_length,
        time_series_max_length=time_series_max_length,
        window_size_min_window_size=window_size_min_window_size,
        window_size_max_window_size=window_size_max_window_size,
        word_lengths_n_word_lengths=word_lengths_n_word_lengths,
        alphabets_min_symbols=alphabets_min_symbols,
        alphabets_max_symbols=alphabets_max_symbols,
        alphabets_step=alphabets_step,
        dilations_min_dilation=dilations_min_dilation,
        dilations_max_dilation=dilations_max_dilation,
        complexity=complexity,
    )

    return build_pipeline(
        configs=configs,
        min_window_to_signal_std_ratio=min_window_to_signal_std_ratio,
        n_jobs=n_jobs,
        n_jobs_numba=n_jobs_numba,
        transformer_weights=transformer_weights,
        pipeline_objects=pipeline_objects,
    ), configs


