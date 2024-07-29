from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ReshapeTo2D(BaseEstimator, TransformerMixin):
    def __init__(self, keep_unraveled_index=False):
        self.keep_unraveled_index = keep_unraveled_index

        self.unraveled_index_ = None  # shape: (n_flattened_features, 2) -> flattened index -> (dimension, word)
        self.original_shape_ = None

    def fit(self, X, y=None):
        self.original_shape_ = X.shape
        if self.keep_unraveled_index:
            self.unraveled_index_ = np.hstack([np.unravel_index(np.arange(np.prod(X.shape[1:])), X.shape[1:])]).T
        return self

    def transform(self, X):
        return X.reshape((X.shape[0], -1))

