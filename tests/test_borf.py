import awkward as ak
import numba as nb
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from fast_borf import BORF
from fast_borf.core.transform import transform_sax_patterns


def assert_same_matrix(actual, expected):
    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 2, 60)).cumsum(axis=2)


@pytest.fixture
def X_padded(X):
    X = X.copy()
    lengths = np.random.default_rng(1).integers(25, 61, size=len(X))
    for i, length in enumerate(lengths):
        X[i, :, length:] = np.nan
    return X


def to_ragged(X):
    return ak.Array([[s[~np.isnan(s)].tolist() for s in series] for series in X])


def test_ragged_input_matches_nan_padding(X_padded):
    borf = BORF().fit(X_padded)
    borf_ragged = BORF(configs=borf.configs_).fit(to_ragged(X_padded))
    assert_same_matrix(
        borf_ragged.transform(to_ragged(X_padded)), borf.transform(X_padded)
    )


def test_ragged_input_with_time_channel(X_padded):
    timestamps = np.cumsum(
        np.random.default_rng(2).exponential(1.0, X_padded[:, :1].shape), axis=2
    )
    timestamps[np.isnan(X_padded[:, :1])] = np.nan
    X = np.concatenate([X_padded, timestamps], axis=1)
    borf = BORF(time_channel=True).fit(X)
    borf_ragged = BORF(configs=borf.configs_, time_channel=True).fit(to_ragged(X))
    assert_same_matrix(borf_ragged.transform(to_ragged(X)), borf.transform(X))


def test_evenly_spaced_time_channel_matches_no_time_channel(X):
    timestamps = np.tile(np.arange(X.shape[2], dtype=float), (len(X), 1, 1))
    with_time = BORF(time_channel=True).fit_transform(
        np.concatenate([X, timestamps], 1)
    )
    assert_same_matrix(with_time, BORF().fit_transform(X))


def test_columns_are_grouped_by_config(X):
    borf = BORF().fit(X)
    X_all = borf.transform(X)
    config_of_column = borf.feature_index_[:, 0]
    assert np.all(np.diff(config_of_column) >= 0)
    for i, config in enumerate(borf.configs_):
        block = X_all[:, config_of_column == i]
        assert_same_matrix(BORF(configs=[config]).fit_transform(X), block)


def test_feature_index_describes_each_column(X):
    borf = BORF().fit(X)
    X_all = borf.transform(X).toarray()
    assert len(borf.feature_index_) == X_all.shape[1]
    for i, config in enumerate(borf.configs_[:5]):
        rows = transform_sax_patterns(X, None, **config)
        expected = {(s, w): None for _, s, w, _ in rows}
        columns = np.flatnonzero(borf.feature_index_[:, 0] == i)
        assert {tuple(borf.feature_index_[c, 1:]) for c in columns} == set(expected)
        for series, signal, word, count in rows:
            column = columns[
                (borf.feature_index_[columns, 1] == signal)
                & (borf.feature_index_[columns, 2] == word)
            ]
            assert X_all[series, column[0]] == count


def test_words_unseen_during_fit_are_dropped(X):
    borf = BORF().fit(X[:5])
    assert borf.transform(X).shape[1] == borf.fit_transform(X[:5]).shape[1]


def test_linear_complexity_uses_word_length_as_stride(X):
    borf = BORF(complexity="linear").fit(X)
    assert all(c["stride"] == c["word_length"] for c in borf.configs_)


def test_alphabet_sizes(X):
    borf = BORF(alphabet_sizes=(3, 4)).fit(X)
    assert {c["alphabet_size"] for c in borf.configs_} == {3, 4}


@pytest.mark.filterwarnings("error:This Pipeline instance is not fitted")
def test_works_in_sklearn_pipeline(X):
    y = np.arange(len(X)) % 2
    pipe = make_pipeline(BORF(), RidgeClassifier())
    scores = cross_val_score(pipe, X, y, cv=2)
    assert len(scores) == 2


def test_clone_keeps_parameters():
    borf = BORF(alphabet_sizes=(3, 4), time_channel=True, n_jobs=2)
    assert clone(borf).get_params() == borf.get_params()


def test_transform_before_fit_raises(X):
    with pytest.raises(NotFittedError):
        BORF().transform(X)


def test_wrong_number_of_signals_raises(X):
    borf = BORF().fit(X)
    with pytest.raises(ValueError, match="signals"):
        borf.transform(X[:, :1])


def test_2d_input_raises(X):
    with pytest.raises(ValueError, match="shape"):
        BORF().fit(X[:, 0])


def test_series_too_short_raises():
    with pytest.raises(ValueError, match="No configuration"):
        BORF().fit(np.zeros((3, 1, 3)))


def test_threads_do_not_change_output_and_are_restored(X):
    before = nb.get_num_threads()
    assert_same_matrix(BORF(n_jobs=-1).fit_transform(X), BORF().fit_transform(X))
    assert nb.get_num_threads() == before
