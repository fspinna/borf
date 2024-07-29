from sklearn.base import BaseEstimator, TransformerMixin
import sparse


class AggregateAxis(BaseEstimator, TransformerMixin):
    def __init__(self, axis):
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.sum(axis=self.axis)


class AggregateAxisGroups(BaseEstimator, TransformerMixin):
    def __init__(self, axis, groups):
        self.axis = axis
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.concatenate(
            [X[:, self.groups[i]].sum(axis=self.axis, keepdims=True) for i in range(len(self.groups))],
            axis=self.axis
        )
