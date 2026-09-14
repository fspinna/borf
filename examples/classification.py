"""Classify regularly sampled time series with BORF and a linear model.

Each series is a random walk with a short bump (class 0) or dip (class 1) at a
random position. The same model is also trained with every configuration's
block of features normalized separately.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from fast_borf import BORF


def make_dataset(n_series=300, length=150, pattern_length=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_series)
    X = 0.3 * rng.standard_normal((n_series, 1, length)).cumsum(axis=2)
    bump = 3 * np.sin(np.linspace(0, np.pi, pattern_length))
    for i in range(n_series):
        start = rng.integers(0, length - pattern_length)
        X[i, 0, start : start + pattern_length] += bump if y[i] == 0 else -bump
    return X, y


def main():
    X, y = make_dataset()  # X has shape (n_series, n_signals, n_timestamps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = make_pipeline(BORF(n_jobs=-1), RidgeClassifierCV())
    model.fit(X_train, y_train)
    print(f"accuracy: {model.score(X_test, y_test):.3f}")

    borf = model[0]
    print(f"{len(borf.configs_)} configurations, {len(borf.feature_index_)} features")
    print("first configuration:", borf.configs_[0])

    blockwise = make_pipeline(
        BORF(block_transformer=Normalizer(), n_jobs=-1), RidgeClassifierCV()
    )
    blockwise.fit(X_train, y_train)
    print(
        f"accuracy with blockwise normalization: {blockwise.score(X_test, y_test):.3f}"
    )


if __name__ == "__main__":
    main()
