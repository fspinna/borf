from sklearn.base import BaseEstimator, TransformerMixin
import sparse
import numpy as np


class ToScipySparse(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_scipy_sparse()

    def inverse_transform(self, X):
        return sparse.COO.from_scipy_sparse(X)

