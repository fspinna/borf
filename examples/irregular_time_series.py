"""Classify irregularly sampled, variable-length time series with BORF.

Each series is a noisy sine observed at random times; the class sets its
frequency. Series have different numbers of observations, so they are stored
either NaN-padded in a NumPy array or as a ragged awkward array. The timestamps
go in the last channel and BORF(time_channel=True) uses them; within each
series they must be strictly increasing.
"""

import awkward as ak
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from fast_borf import BORF


def make_dataset(n_series=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_series)
    series = []
    for label in y:
        n_observations = rng.integers(60, 121)
        timestamps = np.sort(rng.uniform(0, 10, n_observations))
        frequency = 1.0 if label == 0 else 1.5
        values = np.sin(2 * np.pi * frequency * timestamps / 3)
        values += 0.2 * rng.standard_normal(n_observations)
        series.append((values, timestamps))
    return series, y


def to_padded_array(series):
    """Shape (n_series, 2, max_length): the signal, then its timestamps."""
    max_length = max(len(values) for values, _ in series)
    X = np.full((len(series), 2, max_length), np.nan)
    for i, (values, timestamps) in enumerate(series):
        X[i, 0, : len(values)] = values
        X[i, 1, : len(values)] = timestamps
    return X


def to_ragged_array(series):
    return ak.Array(
        [[values.tolist(), timestamps.tolist()] for values, timestamps in series]
    )


def main():
    series, y = make_dataset()
    train, test = train_test_split(np.arange(len(y)), random_state=0)

    X = to_padded_array(series)
    model = make_pipeline(BORF(time_channel=True, n_jobs=-1), RidgeClassifierCV())
    model.fit(X[train], y[train])
    print(f"NaN-padded array, accuracy: {model.score(X[test], y[test]):.3f}")

    X_ragged = to_ragged_array(series)
    model.fit(X_ragged[train], y[train])
    print(f"ragged awkward array, accuracy: {model.score(X_ragged[test], y[test]):.3f}")


if __name__ == "__main__":
    main()
