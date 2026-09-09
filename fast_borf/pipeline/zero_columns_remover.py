import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fast_borf.pipeline.utils import apply_map, bidirectional_index_map_from_indices


class ZeroColumnsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, axis=0, map_features=False):
        self.axis = axis
        self.map_features = map_features

        self.n_original_columns_ = None
        self.columns_to_keep_ = None
        self.feature_map_ = None

        self.in_to_out_index_map_ = None
        self.out_to_in_index_map_ = None

    def fit(self, X, y=None):
        # assert X.shape.ndim == 2
        self.n_original_columns_ = X.shape[1]
        self.columns_to_keep_ = np.argwhere(X.any(axis=self.axis)).ravel()
        if self.map_features:
            self.in_to_out_index_map_, self.out_to_in_index_map_ = (
                bidirectional_index_map_from_indices(self.columns_to_keep_)
            )
        return self

    def transform(self, X):
        # assert X.shape.ndim == 2
        return X[..., self.columns_to_keep_]

    def in_to_out_map(self, features):
        return apply_map(features, self.in_to_out_index_map_)

    def out_to_in_map(self, features):
        return apply_map(features, self.out_to_in_index_map_)

    # def fit_feature_mapper(self):
    #     self.feature_map_ = {i: column for i, column in enumerate(self.columns_to_keep_)}
    #     return self
    #
    # def apply_feature_mapper(self, features):
    #     return np.array([self.feature_map_[feature] for feature in features])

    # def inverse_transform(self, X):
    #     shape = (X.shape[0], self.n_original_columns_)
    #     X_inv = sparse.DOK(shape=shape, fill_value=0)
    #     X_inv[..., self.columns_to_keep_] = X
    #     return sparse.COO(X_inv)


# todo: make a transformer that removes columns that are all zeros but adds another columns, that is the sum of the
#  removed columns for the test set (just all zeros for the training set)
