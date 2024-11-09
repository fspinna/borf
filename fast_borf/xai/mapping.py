import numpy as np
from sklearn.pipeline import FeatureUnion
from fast_borf.xai.pipeline_mapping import map_borf_to_conf
from fast_borf.xai.sax_mapping import align_sax_words_to_raw_ts
from fast_borf.xai.sax_mapping import wsax_configurations_alignment_conversion
from fast_borf.xai.receptive_field import ReceptiveField
import pandas as pd
from scipy.stats import rankdata
from typing import Optional, Literal


class BagOfReceptiveFields:
    def __init__(
        self,
        borf: FeatureUnion,
        borf_position=0,
        reshaper_position=1,
        zero_columns_remover_position=2,
    ):
        self.borf = borf
        self.borf_position = borf_position
        self.reshaper_position = reshaper_position
        self.zero_columns_remover_position = zero_columns_remover_position
        self.mapping = map_borf_to_conf(
            borf=self.borf,
            reshaper_position=reshaper_position,
            zero_columns_remover_position=zero_columns_remover_position,
        )
        self.configs = [
            transformer[self.borf_position].get_params()
            for _, transformer in self.borf.transformer_list
        ]

        self.X_ = None
        self.y_true_ = None
        self.y_pred_ = None
        self.X_transformed_ = None
        self.X_sax_ = None
        self.F_ = None
        self.F_argsort_ = None
        self.F_rank_ = None
        self.F_avg_rank_ = None
        self.F_avg_rank_argsort_ = None
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
        features = np.arange(len(self.mapping))
        self.X_ = X
        self.y_true_ = y_true
        self.y_pred_ = y_pred
        self.X_transformed_ = self.borf.transform(X)
        self.receptive_fields_, self.X_sax_ = build_receptive_fields(
            X=X, X_transformed=self.X_transformed_, features=features, configs=self.configs, mapping=self.mapping
        )
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
        self.F_argsort_ = np.argsort(
            -np.abs(self.F_), axis=1
        )  # for each instance, feature idxs sorted by abs imp
        self.F_sum_ = np.abs(self.F_).sum(
            axis=0
        )  # sum of abs importance (global importance across instances)
        self.F_sum_argsort_ = np.argsort(
            -self.F_sum_
        )  # feature idxs sorted by global abs sum
        self.F_rank_ = rankdata(
            -np.abs(self.F_), axis=1
        )  # for each instance, feature ranks by abs imp
        self.F_avg_rank_ = np.mean(
            self.F_rank_, axis=0
        )  # average rank across instances (global importance)
        self.F_avg_rank_argsort_ = np.argsort(
            self.F_avg_rank_
        )  # feature idxs sorted by global avg rank importance
        for feature, receptive_field in self.receptive_fields_.items():
            receptive_field.feature_importance = self.F_[:, feature]
        return self

    def _add_normalized_feature_importance(self, F_norm):
        assert F_norm.shape[0] == len(self.X_)
        assert F_norm.shape[1] == len(self.mapping)
        self.F_norm_ = F_norm
        for feature, receptive_field in self.receptive_fields_.items():
            receptive_field.feature_importance_norm = F_norm[:, feature]
        return self

    def map_contained_single_feature_importance_to_saliency(
        self, idx, count_overlapping=True, normalize=True
    ):
        X_single = self.X_[idx : idx + 1]
        F_single = self.F_[idx : idx + 1]
        ts_transformed = self.X_transformed_[idx : idx + 1]
        positive_features = np.argwhere(ts_transformed > 0)[:, 1]
        S_single = np.zeros_like(X_single)
        if F_single.sum() == 0:  # if all feature importance are zero
            return S_single
        for receptive_field_idx in positive_features:
            receptive_field = self.receptive_fields_[receptive_field_idx]
            if count_overlapping:
                alignments, counts = np.unique(
                    receptive_field.alignments[idx], return_counts=True
                )
            else:
                alignments = np.unique(receptive_field.alignments[idx])
                counts = np.ones_like(alignments)
            S_single[0, receptive_field.signal_idx, alignments] += (
                receptive_field.feature_importance[idx] * counts
            )
        if normalize:
            S_single = S_single / (
                S_single.sum() / np.sum(F_single[:, positive_features])
            )
        return S_single

    def map_contained_feature_importance_to_saliency(
        self, count_overlapping=True, normalize=True
    ):
        S = list()
        for i in range(len(self.X_)):
            S_single = self.map_contained_single_feature_importance_to_saliency(
                i, count_overlapping, normalize
            )
            S.append(S_single)
        self.S_ = np.vstack(S)
        return self

    # def map_single_notcontained_feature_importance(self, idx):
    #     F_single = self.F_[idx:idx + 1]
    #     ts_transformed = self.X_transformed_[idx:idx + 1]
    #     null_features = np.argwhere(ts_transformed == 0)[:, 1]
    #     null_features_sum = np.sum(F_single[:, null_features])
    #     F_sum = 0
    #     word_lengths = list()
    #     for null_feature in null_features:
    #         feature_importance = self.receptive_fields_[null_feature].feature_importance[idx]
    #         word_length = self.receptive_fields_[null_feature].word_length
    #         word_lengths.append(word_length)
    #         F_sum += feature_importance * word_length
    #     F_single_norm = np.full_like(F_single, np.nan)
    #     F_single_norm[:, null_features] = F_single[:, null_features] * null_features_sum / F_sum
    #     # np.allclose(F_single[:, null_features].sum(),
    #     #             np.nansum(np.array(word_lengths) * F_single_norm[:, null_features].ravel()))
    #     return F_single_norm

    def map_single_notcontained_feature_importance(self, idx):
        F_single = self.F_[idx : idx + 1]
        ts_transformed = self.X_transformed_[idx : idx + 1]
        null_features = np.argwhere(ts_transformed == 0)[:, 1]
        null_features_sum = np.sum(F_single[:, null_features])
        F_sum = 0
        window_sizes = list()
        for null_feature in null_features:
            feature_importance = self.receptive_fields_[
                null_feature
            ].feature_importance[idx]
            window_size = self.receptive_fields_[null_feature].window_size
            window_sizes.append(window_size)
            F_sum += feature_importance * window_size
        F_single_norm = np.full_like(F_single, np.nan)
        F_single_norm[:, null_features] = (
            F_single[:, null_features] * null_features_sum / F_sum
        )
        # np.allclose(F_single[:, null_features].sum(),
        #             np.nansum(np.array(word_lengths) * F_single_norm[:, null_features].ravel()))
        return F_single_norm

    def map_notcontained_feature_importance(self):
        F_norm = list()
        for i in range(len(self.X_)):
            F_single_norm = self.map_single_notcontained_feature_importance(i)
            F_norm.append(F_single_norm)
        self._add_normalized_feature_importance(np.vstack(F_norm))
        return self

    def get_mapping_with_feature_importance(self, idxs=None):
        if idxs is None:
            idxs = np.arange(len(self.X_))
        absolute_importance = np.abs(self.F_[idxs]).sum(axis=0)
        mapping_df = pd.DataFrame(
            self.mapping, columns=["conf_idx", "signal_idx", "word_idx"]
        )
        mapping_df["feature_importance"] = absolute_importance
        return mapping_df

    def get_most_important_not_contained_patterns_by_signal(self, i):
        x_tr = self.X_transformed_[i].toarray()
        f_norm = self.F_norm_[i]
        signal_idxs = self.mapping[:, 1]
        zero_indices = np.argwhere(x_tr == 0)[:, 1]
        num_signals = np.max(signal_idxs) + 1
        sorted_indices_by_signal = []

        # Step 4: Loop through each unique signal index
        for signal in range(num_signals):
            # Identify indices within the current signal
            signal_indices = zero_indices[signal_idxs[zero_indices] == signal]

            if len(signal_indices) > 0:
                filtered_importance = f_norm[signal_indices]
                abs_importance = np.abs(filtered_importance)
                sorted_indices = np.argsort(-abs_importance)
                sorted_original_indices = signal_indices[sorted_indices]
                sorted_indices_by_signal.append(sorted_original_indices)
            else:
                sorted_indices_by_signal.append(np.array([]))
        return sorted_indices_by_signal

    def get_most_important_contained_patterns_by_signal(self, i):
        x_tr = self.X_transformed_[i].toarray()
        f_norm = self.F_norm_[i]
        signal_idxs = self.mapping[:, 1]
        not_zero_indices = np.argwhere(x_tr > 0)[:, 1]
        num_signals = np.max(signal_idxs) + 1
        sorted_indices_by_signal = []

        # Step 4: Loop through each unique signal index
        for signal in range(num_signals):
            # Identify indices within the current signal
            signal_indices = not_zero_indices[signal_idxs[not_zero_indices] == signal]

            if len(signal_indices) > 0:
                filtered_importance = f_norm[signal_indices]
                abs_importance = np.abs(filtered_importance)
                sorted_indices = np.argsort(-abs_importance)
                sorted_original_indices = signal_indices[sorted_indices]
                sorted_indices_by_signal.append(sorted_original_indices)
            else:
                sorted_indices_by_signal.append(np.array([]))
        return sorted_indices_by_signal

    def get_most_important_patterns_by_signal(self, i):
        x_tr = self.X_transformed_[i].toarray()
        f_norm = self.F_norm_[i]
        signal_idxs = self.mapping[:, 1]
        indices = np.arange(len(x_tr[0]))
        num_signals = np.max(signal_idxs) + 1
        sorted_indices_by_signal = []

        # Step 4: Loop through each unique signal index
        for signal in range(num_signals):
            # Identify indices within the current signal
            signal_indices = indices[signal_idxs[indices] == signal]

            if len(signal_indices) > 0:
                filtered_importance = f_norm[signal_indices]
                abs_importance = np.abs(filtered_importance)
                sorted_indices = np.argsort(-abs_importance)
                sorted_original_indices = signal_indices[sorted_indices]
                sorted_indices_by_signal.append(sorted_original_indices)
            else:
                sorted_indices_by_signal.append(np.array([]))
        return sorted_indices_by_signal


def build_receptive_fields(
    X, X_transformed, features, configs, mapping, feature_importance=None
):
    sax_converted_X = wsax_configurations_alignment_conversion(X, configs)
    # X_transformed = self.borf.transform(X)
    X_receptive_fields = dict()
    for feature in features:
        conf_idx, signal_idx, word_idx = mapping[feature]
        config = configs[conf_idx]
        word_length = config["word_length"]
        window_size = config["window_size"]
        ts_receptive_fields_alignments = list()
        ts_receptive_fields_mappings = list()
        for i in range(len(X)):
            if word_idx in sax_converted_X[conf_idx][i][signal_idx]:
                align = sax_converted_X[conf_idx][i][signal_idx][word_idx]
                ts_receptive_fields_alignments.append(align)
                ts_receptive_fields_mappings.append(np.array(X[i, signal_idx][align]))
            else:
                ts_receptive_fields_alignments.append(
                    np.empty(
                        (0, word_length, window_size // word_length), dtype=np.int_
                    )
                )
                ts_receptive_fields_mappings.append(
                    np.empty(
                        (0, word_length, window_size // word_length), dtype=np.int_
                    )
                )
        receptive_field = ReceptiveField(
            compressed_word_int=word_idx,
            signal_idx=signal_idx,
            conf_idx=conf_idx,
            feature_idx=feature,
            feature_values=X_transformed[:, feature].toarray().ravel(),
            feature_importance=feature_importance[:, feature]
            if feature_importance is not None
            else None,
            alignments=ts_receptive_fields_alignments,
            mappings=ts_receptive_fields_mappings,
            **config
        )
        X_receptive_fields[feature] = receptive_field
    return X_receptive_fields, sax_converted_X
