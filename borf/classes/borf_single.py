import awkward as ak
import numba as nb
import numpy as np
from scipy.sparse import csc_array, hstack
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Union

from borf.algorithms.borf_transform import (
    extract_sax_words_single_configuration,
    transform_sax_words_single_configuration,
    fit_transform_sax_words_single_configuration,
)
from borf.algorithms.borf_alignment import get_ts_alignment_idxs_from_word
from borf.utils.transform_utils import create_dict
from borf.classes.receptive_field import ReceptiveField
from borf.utils.condition_utils import is_empty
from borf.utils.transform_utils import average_groups


class BorfSingleTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size: int = 4,
        stride: int = 1,
        dilation: int = 1,
        word_length: int = 2,
        alphabet_size_mean: int = 3,
        alphabet_size_slope: int = 0,
        strategies: tuple = ("repetitions",),
        use_signal_id=True,
        ignore_equal_contiguous_words=False,
        signal_separator=";",
        padding: str = "valid",
        min_window_to_signal_std_ratio: float = 0,
        word_position_strategy: str = "average",
        normalize_flatten_to_zero: bool = False,
        prefix: str = "",
    ):
        self.window_size = window_size
        self.stride = stride
        self.dilation = dilation
        self.word_length = word_length
        self.alphabet_size_mean = alphabet_size_mean
        self.alphabet_size_slope = alphabet_size_slope
        self.use_signal_id = use_signal_id
        self.ignore_equal_contiguous_words = ignore_equal_contiguous_words
        self.signal_separator = signal_separator
        self.padding = padding
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.store_word_position = True if "positions" in strategies else False
        self.word_position_strategy = word_position_strategy
        self.normalize_flatten_to_zero = normalize_flatten_to_zero
        self.strategies = strategies
        self.prefix = prefix

        self.words_idx_dict_ = None
        self.idx_words_dict_ = None
        self.n_features_ = None

    def fit(self, X: ak.Array, y=None):
        self.words_idx_dict_ = dict(
            extract_sax_words_single_configuration(
                X=X,
                window_size=self.window_size,
                stride=self.stride,
                dilation=self.dilation,
                word_length=self.word_length,
                alphabet_size_mean=self.alphabet_size_mean,
                alphabet_size_slope=self.alphabet_size_slope,
                use_signal_id=self.use_signal_id,
                signal_separator=self.signal_separator,
                padding=self.padding,
                min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
                normalize_flatten_to_zero=self.normalize_flatten_to_zero,
                prefix=self.prefix,
            )
        )
        self.n_features_ = len(self.words_idx_dict_)
        return self

    def transform(
        self,
        X: ak.Array,
    ):
        rows = nb.typed.List.empty_list(nb.types.int64)
        cols = nb.typed.List.empty_list(nb.types.int64)
        values = nb.typed.List.empty_list(nb.types.int64)
        positions = nb.typed.List.empty_list(nb.types.float64)
        if len(self.words_idx_dict_) > 0:
            transform_sax_words_single_configuration(
                X=X,
                sax_words=create_dict(
                    nb.typed.List(self.words_idx_dict_.items())
                ),  # FIXME
                row_idxs=rows,
                column_idxs=cols,
                values=values,
                positions=positions,
                window_size=self.window_size,
                stride=self.stride,
                dilation=self.dilation,
                word_length=self.word_length,
                alphabet_size_mean=self.alphabet_size_mean,
                alphabet_size_slope=self.alphabet_size_slope,
                use_signal_id=self.use_signal_id,
                signal_separator=self.signal_separator,
                padding=self.padding,
                min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
                store_word_position=self.store_word_position,
                normalize_flatten_to_zero=self.normalize_flatten_to_zero,
                prefix=self.prefix,
            )
        return build_sparse_arrays(
            rows=np.asarray(rows),
            cols=np.asarray(cols),
            values=np.asarray(values),
            positions=np.asarray(positions),
            strategies=self.strategies,
            n_features=self.n_features_,
            n_instances=len(X),
            ignore_equal_contiguous_words=self.ignore_equal_contiguous_words,
            word_position_strategy=self.word_position_strategy,
        )

    def fit_transform(self, X: ak.Array, y=None, **fit_params):
        rows = nb.typed.List.empty_list(nb.types.int64)
        cols = nb.typed.List.empty_list(nb.types.int64)
        values = nb.typed.List.empty_list(nb.types.int64)
        positions = nb.typed.List.empty_list(nb.types.float64)
        self.words_idx_dict_ = dict(
            fit_transform_sax_words_single_configuration(
                X=X,
                row_idxs=rows,
                column_idxs=cols,
                values=values,
                positions=positions,
                window_size=self.window_size,
                stride=self.stride,
                dilation=self.dilation,
                word_length=self.word_length,
                alphabet_size_mean=self.alphabet_size_mean,
                alphabet_size_slope=self.alphabet_size_slope,
                use_signal_id=self.use_signal_id,
                signal_separator=self.signal_separator,
                padding=self.padding,
                min_window_to_signal_std_ratio=self.min_window_to_signal_std_ratio,
                store_word_position=self.store_word_position,
                normalize_flatten_to_zero=self.normalize_flatten_to_zero,
                prefix=self.prefix,
            )
        )
        self.n_features_ = len(self.words_idx_dict_)
        return build_sparse_arrays(
            rows=np.asarray(rows),
            cols=np.asarray(cols),
            values=np.asarray(values),
            positions=np.asarray(positions),
            strategies=self.strategies,
            n_features=self.n_features_,
            n_instances=len(X),
            ignore_equal_contiguous_words=self.ignore_equal_contiguous_words,
            word_position_strategy=self.word_position_strategy,
        )

    def word_to_idx(self, word: str) -> Union[int, None]:
        if len(self.strategies) > 1:
            raise ValueError(
                "This method is only available when using a single strategy."
            )
        if len(self.words_idx_dict_) == 0:
            return None
        return self.words_idx_dict_[word]

    def idx_to_word(self, idx: int) -> Union[str, None]:
        if len(self.strategies) > 1:
            raise ValueError(
                "This method is only available when using a single strategy."
            )
        if len(self.words_idx_dict_) == 0:
            self.idx_words_dict_ = dict()
            return None
        if self.idx_words_dict_ is None:
            self.idx_words_dict_ = {v: k for k, v in self.words_idx_dict_.items()}
        return self.idx_words_dict_[idx]

    def get_word_alignment(self, X, query):
        return get_ts_alignment_idxs_from_word(
            X=X, sax_query=query, **self.get_params()
        )

    def get_idx_alignment(self, X, idx):
        return get_ts_alignment_idxs_from_word(
            X=X, sax_query=self.idx_to_word(idx), **self.get_params()
        )

    def get_idxs_alignments(self, X, idxs):
        alignments_list = list()
        for i in range(len(X)):
            alignments = list()
            for idx in idxs:
                alignments.append(self.get_idx_alignment(X[i : i + 1], idx))
            alignments_list.append(alignments)
        return alignments_list

    def get_feature_names(self):
        return list(self.words_idx_dict_.keys())

    def init_receptive_field(self, query, tabular_idx=None):
        receptive_field = ReceptiveField(
            word=query,
            tabular_idx=tabular_idx,
            signal_separator=self.signal_separator,
            sax_params=self.get_params(),
        )
        return receptive_field

    def init_receptive_field_from_idx(self, idx):
        return self.init_receptive_field(query=self.idx_to_word(idx), tabular_idx=idx)

    def init_receptive_fields(self, queries, tabular_idxs=None):
        return (
            [self.init_receptive_field(query) for query in queries]
            if tabular_idxs is None
            else [
                self.init_receptive_field(query, idx)
                for query, idx in zip(queries, tabular_idxs)
            ]
        )

    def init_receptive_fields_from_idxs(self, idxs):
        return self.init_receptive_fields(
            queries=[self.idx_to_word(idx) for idx in idxs], tabular_idxs=idxs
        )


def build_sparse_array(
    rows,
    cols,
    values,
    positions,
    strategy,
    n_features,
    n_instances,
    ignore_equal_contiguous_words: bool = False,
    word_position_strategy="average",
    **kwargs,
):
    if strategy == "repetitions":
        if ignore_equal_contiguous_words:
            return csc_array(
                (np.ones_like(rows), (rows, cols)),
                shape=(n_instances, n_features),
                dtype=np.float32,
            )
        else:
            return csc_array(
                (values, (rows, cols)),
                shape=(n_instances, n_features),
                dtype=np.float32,
            )
    elif strategy == "positions":
        if is_empty(positions):
            raise ValueError
        else:
            rows, cols, positions = average_groups(
                row_idxs=rows, column_idxs=cols, values=positions
            )
            return csc_array(
                (positions, (rows, cols)),
                shape=(n_instances, n_features),
                dtype=np.float32,
            )
    elif strategy == "binary":
        return (
            csc_array(
                (np.ones_like(rows), (rows, cols)),
                shape=(n_instances, n_features),
                dtype=np.float32,
            )
            >= 1
        ) * 1


def build_sparse_arrays(
    rows,
    cols,
    values,
    positions,
    strategies,
    n_features,
    n_instances,
    ignore_equal_contiguous_words: bool = False,
    word_position_strategy="average",
):
    final_array = build_sparse_array(
        rows=rows,
        cols=cols,
        values=values,
        positions=positions,
        strategy=strategies[0],
        n_features=n_features,
        n_instances=n_instances,
        ignore_equal_contiguous_words=ignore_equal_contiguous_words,
        word_position_strategy=word_position_strategy,
    )
    for strategy in strategies[1:]:
        final_array = hstack(
            [
                final_array,
                build_sparse_array(
                    rows=rows,
                    cols=cols,
                    values=values,
                    positions=positions,
                    strategy=strategy,
                    n_features=n_features,
                    n_instances=n_instances,
                    ignore_equal_contiguous_words=ignore_equal_contiguous_words,
                    word_position_strategy=word_position_strategy,
                ),
            ]
        )
    return final_array
