import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fast_borf.pipeline.utils import apply_map, bidirectional_index_map_from_indices


class ReshapeTo2D(BaseEstimator, TransformerMixin):
    def __init__(self, keep_unraveled_index=False, map_features=False):
        self.keep_unraveled_index = keep_unraveled_index
        self.map_features = map_features

        self.unraveled_index_ = None  # shape: (n_flattened_features, 2) -> flattened index -> (dimension, word)
        self.original_shape_ = None

    def fit(self, X, y=None):
        self.original_shape_ = X.shape
        if self.keep_unraveled_index:
            self.unraveled_index_ = np.hstack(
                [np.unravel_index(np.arange(np.prod(X.shape[1:])), X.shape[1:])]
            ).T
            if self.map_features:
                unraveled_index_tuple = [tuple(row) for row in self.unraveled_index_]
                self.in_to_out_index_map_, self.out_to_in_index_map_ = (
                    bidirectional_index_map_from_indices(unraveled_index_tuple)
                )
        return self

    def transform(self, X):
        return X.reshape((X.shape[0], -1))

    def in_to_out_map(self, features):
        return apply_map(features, self.in_to_out_index_map_)

    def out_to_in_map(self, features):
        return apply_map(features, self.out_to_in_index_map_)
