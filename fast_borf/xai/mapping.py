from collections import OrderedDict
from collections.abc import Mapping
from typing import Literal, Optional

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

from fast_borf.borf import BORF, n_words, to_padded_array
from fast_borf.core.sax import window_positions
from fast_borf.core.transform import panel_words
from fast_borf.xai.receptive_field import ReceptiveField
from fast_borf.xai.saliency import add_window_importance

CACHED_CONFIGS = 16


class BagOfReceptiveFields:
    """Map the feature importances of a model on BORF features back onto the series.

    Workflow::

        explainer = BagOfReceptiveFields(borf).build(X, y_true, y_pred, task)
        explainer.add_feature_importance(F)  # any importances, e.g. SHAP values
        explainer.map_contained_feature_importance_to_saliency()  # sets S_
        explainer.map_notcontained_feature_importance()  # sets F_norm_
        explainer.receptive_fields_[j]  # where feature j's word occurs

    Parameters
    ----------
    borf : BORF
        Fitted BORF whose output columns the importances refer to. Columns
        created by its block_transformer cannot be explained.

    Attributes
    ----------
    mapping : ndarray of shape (n_features, 3)
        (configuration index, signal index, word) of each feature.
    configs : list of dict
        BORF's configurations, with min_window_to_signal_std_ratio and
        contains_time_idx added.
    X_, timestamps_ : ndarray
        The signals and timestamps of the series given to build, NaN-padded.
    X_transformed_ : sparse matrix
        BORF's features of those series.
    F_ : ndarray of shape (n_series, n_features)
        Importances, for the predicted class in classification.
    F_argsort_, F_rank_ : ndarray
        Per series, features sorted by, and ranked by, absolute importance.
    F_sum_, F_sum_argsort_ : ndarray
        Absolute importance summed over series, and features sorted by it.
    F_avg_rank_, F_avg_rank_argsort_ : ndarray
        Rank averaged over series, and features sorted by it. The rank
        attributes are computed on first access, as they are the slowest.
    S_ : ndarray of shape (n_series, n_signals, n_timestamps)
        Saliency: the importances of the words occurring in each series,
        spread over the points their windows cover.
    F_norm_ : ndarray of shape (n_series, n_features)
        Importances of the words absent from each series, rescaled so that
        their sum weighted by window size equals their plain sum; NaN for
        words that occur.
    receptive_fields_ : mapping from feature index to ReceptiveField
        Computed when first accessed.
    """

    def __init__(self, borf: BORF):
        if borf.channel_groups_ is not None:
            raise NotImplementedError(
                "Explanations of BORF(channel_groups=...) are not supported yet"
            )
        if np.any(borf.feature_index_[:, 1] < 0):
            raise ValueError(
                "Some columns were created by block_transformer and do not map "
                "to a single word, so they cannot be explained"
            )
        self.borf = borf
        self.mapping = borf.feature_index_  # (conf_idx, signal_idx, word_idx)
        self.contains_time_idx = borf.time_channel
        self.configs = [
            dict(
                config,
                min_window_to_signal_std_ratio=borf.min_window_to_signal_std_ratio,
                contains_time_idx=borf.time_channel,
            )
            for config in borf.configs_
        ]

        self.X_ = None
        self.timestamps_ = None
        self.y_true_ = None
        self.y_pred_ = None
        self.X_transformed_ = None
        self.F_ = None
        self._ranks = {}
        self.F_sum_ = None
        self.F_sum_argsort_ = None
        self.F_norm_ = None
        self.S_ = None
        self.receptive_fields_ = None
        self.task_ = None

    def __getitem__(self, key):
        return self.receptive_fields_[key]

    def build(
        self,
        X,
        y_true=None,
        y_pred=None,
        task: Optional[Literal["classification", "regression"]] = None,
    ):
        self.task_ = task
        X = to_padded_array(X)
        self.X_transformed_ = self.borf.transform(X)
        if self.contains_time_idx:
            self.X_, self._panel_timestamps = X[:, :-1], X[:, -1:]
            self.timestamps_ = self._panel_timestamps
        else:
            self.X_, self._panel_timestamps = X, None
            self.timestamps_ = np.repeat(
                np.arange(X.shape[2])[None, None, :], len(X), axis=0
            )
        self.y_true_ = y_true
        self.y_pred_ = y_pred
        self._words = OrderedDict()
        self._counts = self._count_words()
        self.receptive_fields_ = ReceptiveFields(self)
        return self

    def add_feature_importance(self, F):
        F = np.array(F)
        if self.task_ == "classification":
            if F.ndim == 2:  # this happens in binary classification
                assert F.shape[0] == len(self.X_)
                assert F.shape[1] == len(self.mapping)
                F = np.concatenate([-F[np.newaxis, ...], F[np.newaxis, ...]], axis=0)
            elif F.ndim == 3:  # this happens in multiclass classification
                assert F[0].shape[0] == len(self.X_)
                assert F[0].shape[1] == len(self.mapping)
            else:
                raise ValueError("F should be 2D or 3D array")
            self.F_ = F[
                self.y_pred_, np.arange(F.shape[1]), :
            ]  # only the importance toward predicted class
        elif self.task_ == "regression":
            assert F.shape[0] == len(self.X_)
            assert F.shape[1] == len(self.mapping)
            self.F_ = F
        else:
            raise ValueError(
                "task should be either 'classification' or 'regression', "
                f"got {self.task_} instead"
            )

        self.F_sum_ = np.abs(self.F_).sum(
            axis=0
        )  # sum of abs importance (global importance across instances)
        self.F_sum_argsort_ = np.argsort(
            -self.F_sum_
        )  # feature idxs sorted by global abs sum
        self._ranks = {}  # the rank attributes below are computed on first access
        self.receptive_fields_.clear()
        return self

    def _rank(self, name, compute):
        if self.F_ is None:
            return None
        if name not in self._ranks:
            self._ranks[name] = compute()
        return self._ranks[name]

    @property
    def F_argsort_(self):
        """For each series, features sorted by absolute importance."""
        return self._rank("argsort", lambda: np.argsort(-np.abs(self.F_), axis=1))

    @property
    def F_rank_(self):
        """For each series, the rank of each feature by absolute importance."""
        return self._rank("rank", lambda: rankdata(-np.abs(self.F_), axis=1))

    @property
    def F_avg_rank_(self):
        """Rank averaged over series (global importance)."""
        return self._rank("avg_rank", lambda: np.mean(self.F_rank_, axis=0))

    @property
    def F_avg_rank_argsort_(self):
        """Features sorted by average rank, most important first."""
        return self._rank("avg_rank_argsort", lambda: np.argsort(self.F_avg_rank_))

    def map_contained_feature_importance_to_saliency(
        self, count_overlapping=True, normalize=True
    ):
        """Spread the importance of each word occurring in a series over its points.

        Every window adds its word's importance to the points it covers. With
        count_overlapping=False, a point gets a word's importance once however
        many of its windows cover it. With normalize=True, each series'
        saliency is rescaled to sum to the importances of its words.
        """
        F = np.ascontiguousarray(self.F_, dtype=np.float64)
        S = np.zeros(self.X_.shape)
        for config_idx in range(len(self.configs)):
            words = self._config_words(config_idx)
            add_window_importance(
                S,
                F,
                words["columns"],
                words["word_offsets"],
                words["observed"],
                words["observed_offsets"],
                words["positions"],
                self.X_.shape[1],
                count_overlapping,
            )
        has_importance = F.sum(axis=1) != 0  # if all feature importance are zero
        S[~has_importance] = 0
        if normalize:
            contained = self._counts > 0
            contained_sum = np.asarray(contained.multiply(F).sum(axis=1)).ravel()
            for i in np.flatnonzero(has_importance):
                S[i] = S[i] / (S[i].sum() / contained_sum[i])
        self.S_ = S
        return self

    def map_notcontained_feature_importance(self):
        """Rescale the importances of the words absent from each series (F_norm_)."""
        F = np.asarray(self.F_, dtype=np.float64)
        absent = ~(self._counts > 0).toarray()
        window_sizes = np.array([c["window_size"] for c in self.configs])[
            self.mapping[:, 0]
        ]
        F_absent = np.where(absent, F, 0.0)
        null_features_sum = F_absent.sum(axis=1, keepdims=True)
        F_sum = (F_absent * window_sizes).sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            F_norm = np.where(absent, F * null_features_sum / F_sum, np.nan)
        self.F_norm_ = F_norm
        self.receptive_fields_.clear()
        return self

    def _config_words(self, config_idx):
        """Words, positions and feature of every window, for one configuration."""
        if config_idx in self._words:
            self._words.move_to_end(config_idx)
            return self._words[config_idx]
        config = self.configs[config_idx]
        words, word_offsets, observed, observed_offsets = panel_words(
            self.X_,
            self._panel_timestamps,
            config["window_size"],
            config["word_length"],
            config["alphabet_size"],
            config["stride"],
            config["dilation"],
            config["min_window_to_signal_std_ratio"],
        )
        n_signals = self.X_.shape[1]
        windows_per_signal = np.diff(word_offsets)
        signal_of_window = np.repeat(
            np.arange(len(windows_per_signal)) % n_signals, windows_per_signal
        )
        result = dict(
            words=words,
            word_offsets=word_offsets,
            observed=observed,
            observed_offsets=observed_offsets,
            positions=window_positions(
                int(windows_per_signal.max(initial=0)),
                config["window_size"],
                config["word_length"],
                config["stride"],
                config["dilation"],
            ),
            columns=self._word_columns(config_idx, signal_of_window, words),
        )
        self._words[config_idx] = result
        if len(self._words) > CACHED_CONFIGS:
            self._words.popitem(last=False)
        return result

    def _word_columns(self, config_idx, signals, words):
        """Feature of each (signal, word), or -1 when it is not a feature."""
        config_slice = self.borf.config_slices_[config_idx]
        rows = self.mapping[config_slice]
        columns = np.full(len(words), -1, dtype=np.int64)
        if len(rows) == 0:
            return columns
        size = n_words(self.configs[config_idx])
        keys = rows[:, 1] * size + rows[:, 2]
        order = np.argsort(keys)
        keys = keys[order]
        window_keys = signals * size + words
        position = np.minimum(np.searchsorted(keys, window_keys), len(keys) - 1)
        found = keys[position] == window_keys
        columns[found] = config_slice.start + order[position[found]]
        return columns

    def _count_words(self):
        """Occurrences of each feature's word in each series, as a sparse matrix."""
        n_signals = self.X_.shape[1]
        rows, cols = [], []
        for config_idx in range(len(self.configs)):
            words = self._config_words(config_idx)
            series_of_window = np.repeat(
                np.arange(len(words["word_offsets"]) - 1) // n_signals,
                np.diff(words["word_offsets"]),
            )
            is_feature = words["columns"] >= 0
            rows.append(series_of_window[is_feature])
            cols.append(words["columns"][is_feature])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int64), (rows, cols)),
            shape=(len(self.X_), len(self.mapping)),
        )

    def _receptive_field(self, feature):
        config_idx, signal_idx, word = (int(v) for v in self.mapping[feature])
        config = self.configs[config_idx]
        words = self._config_words(config_idx)
        n_signals = self.X_.shape[1]
        segment_size = config["window_size"] // config["word_length"]
        alignments_indices, alignments, mappings = [], [], []
        for i in range(len(self.X_)):
            g = i * n_signals + signal_idx
            start, stop = words["word_offsets"][g], words["word_offsets"][g + 1]
            occurrences = np.flatnonzero(words["words"][start:stop] == word)
            observed = words["observed"][
                words["observed_offsets"][g] : words["observed_offsets"][g + 1]
            ]
            if len(occurrences):
                indices = observed[words["positions"][occurrences]]
            else:
                indices = np.empty(
                    (0, config["word_length"], segment_size), dtype=np.int64
                )
            alignments_indices.append(indices)
            alignments.append(self.timestamps_[i, 0][indices])
            mappings.append(self.X_[i, signal_idx][indices])
        return ReceptiveField(
            compressed_word_int=word,
            signal_idx=signal_idx,
            conf_idx=config_idx,
            feature_idx=feature,
            feature_values=self.X_transformed_[:, feature].toarray().ravel(),
            feature_importance=None if self.F_ is None else self.F_[:, feature],
            feature_importance_norm=(
                None if self.F_norm_ is None else self.F_norm_[:, feature]
            ),
            alignments=alignments,
            mappings=mappings,
            alignments_indices=alignments_indices,
            **config,
        )


class ReceptiveFields(Mapping):
    """Receptive fields by feature index, computed and cached on first access."""

    def __init__(self, explainer):
        self._explainer = explainer
        self._cache = {}

    def __getitem__(self, feature):
        if not 0 <= feature < len(self):
            raise KeyError(feature)
        if feature not in self._cache:
            self._cache[feature] = self._explainer._receptive_field(feature)
        return self._cache[feature]

    def __iter__(self):
        return iter(range(len(self)))

    def __len__(self):
        return len(self._explainer.mapping)

    def clear(self):
        """Forget computed fields, e.g. after the importances change."""
        self._cache.clear()
