from sklearn.base import BaseEstimator, TransformerMixin
from borf.classes.borf_single import BorfSingleTransformer
from borf.utils.transform_utils import is_within_interval
from typing import Sequence, Dict
import awkward as ak
from scipy.sparse import hstack
from joblib import Parallel, delayed


class BorfMultiTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        configs: Sequence[Dict],
        strategies: tuple = ("repetitions",),
        use_signal_id=True,
        ignore_equal_contiguous_words=False,
        signal_separator=";",
        parameters_separator="_",
        padding: str = "valid",
        min_window_to_signal_std_ratio: float = 0,
        word_position_strategy: str = "average",
        normalize_flatten_to_zero: bool = False,
        prefix: str = "",
        n_jobs=1,
    ):

        self.configs = configs
        self.strategies = strategies
        self.use_signal_id = use_signal_id
        self.ignore_equal_contiguous_words = ignore_equal_contiguous_words
        self.signal_separator = signal_separator
        self.parameters_separator = parameters_separator
        self.padding = padding
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.word_position_strategy = word_position_strategy
        self.normalize_flatten_to_zero = normalize_flatten_to_zero
        self.n_jobs = n_jobs
        self.prefix = prefix

        self.transformers_ = None
        self.feature_indices_ = None

    def fit(self, X: ak.Array, y=None):
        self.transformers_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(config, X) for config in self.configs
        )
        self.get_feature_indices()
        return self

    def _fit(self, config, X):
        parameter = [
            config["alphabet_size_mean"],
            config["alphabet_size_slope"],
            config["dilation"],
            config["stride"],
            config["window_size"],
            config["word_length"],
        ]
        metadata = self.parameters_separator.join([str(par) for par in parameter])
        transformer = BorfSingleTransformer(
            **config,
            strategies=self.strategies,
            use_signal_id=self.use_signal_id,
            ignore_equal_contiguous_words=self.ignore_equal_contiguous_words,
            signal_separator=self.signal_separator,
            padding=self.padding,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
            word_position_strategy=self.word_position_strategy,
            normalize_flatten_to_zero=self.normalize_flatten_to_zero,
            prefix=self.prefix + metadata
        )
        transformer.fit(X=X)
        return transformer

    def transform(self, X: ak.Array, y=None):
        transformed_data = Parallel(n_jobs=self.n_jobs)(
            delayed(transformer.transform)(X) for transformer in self.transformers_
        )
        return hstack(transformed_data)

    def fit_transform(self, X, y=None, **fit_params):
        Xs_tr = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_transform_)(config, X) for config in self.configs
        )
        self.transformers_, Xs_tr = zip(*Xs_tr)
        self.get_feature_indices()
        return hstack(Xs_tr)

    def _fit_transform_(self, config, X):
        parameter = [
            config["alphabet_size_mean"],
            config["alphabet_size_slope"],
            config["dilation"],
            config["stride"],
            config["window_size"],
            config["word_length"],
        ]
        metadata = self.parameters_separator.join([str(par) for par in parameter])
        transformer = BorfSingleTransformer(
            **config,
            strategies=self.strategies,
            use_signal_id=self.use_signal_id,
            ignore_equal_contiguous_words=self.ignore_equal_contiguous_words,
            signal_separator=self.signal_separator,
            padding=self.padding,
            min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
            word_position_strategy=self.word_position_strategy,
            normalize_flatten_to_zero=self.normalize_flatten_to_zero,
            prefix=self.prefix + metadata
        )
        return transformer, transformer.fit_transform(X=X)

    def get_feature_indices(self):
        self.feature_indices_ = []
        start_idx = 0
        for transformer in self.transformers_:
            n_features = transformer.n_features_
            end_idx = start_idx + n_features - 1
            self.feature_indices_.append((start_idx, end_idx))
            start_idx += n_features
        return self.feature_indices_

    def idx_to_word(self, idx: int):
        for i, (start, end) in enumerate(self.feature_indices_):
            if is_within_interval(idx, start, end):
                return self.transformers_[i].idx_to_word(idx - start)

    def word_to_idx(self, word: str):
        for i, transformer in enumerate(self.transformers_):
            idx = transformer.word_to_idx(word)
            if idx is not None:
                return idx + self.feature_indices_[i][0]

    def get_idx_alignment(self, X, idx):
        for i, (start, end) in enumerate(self.feature_indices_):
            if is_within_interval(idx, start, end):
                return self.transformers_[i].get_idx_alignment(X, idx - start)

    def init_receptive_field(self, idx):
        for i, (start, end) in enumerate(self.feature_indices_):
            if is_within_interval(idx, start, end):
                return self.transformers_[i].init_receptive_field_from_idx(idx - start)

    def init_receptive_fields(self, idxs):
        receptive_fields = []
        for idx in idxs:
            receptive_fields.append(self.init_receptive_field(idx))
        return receptive_fields

    def get_idxs_alignments(self, X, idxs):
        alignments_list = list()
        for i in range(len(X)):
            alignments = list()
            for idx in idxs:
                alignments.append(self.get_idx_alignment(X[i : i + 1], idx))
            alignments_list.append(alignments)
        return alignments_list

    def get_feature_names(self):
        feature_names = []
        for transformer in self.transformers_:
            feature_names += transformer.get_feature_names()
        return feature_names
