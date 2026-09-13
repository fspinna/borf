import awkward as ak
import numba as nb
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, Normalizer

from fast_borf import BORF
from fast_borf.core.transform import transform_sax_patterns
from fast_borf.xai.mapping import BagOfReceptiveFields


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


def with_timestamps(X, timestamps):
    return np.concatenate([X, np.broadcast_to(timestamps, X[:, :1].shape)], axis=1)


@pytest.mark.parametrize(
    "position, value, message",
    [
        (20, 19.0, "strictly increasing"),  # repeats the previous timestamp
        (1, 0.0, "strictly increasing"),
        (20, 10.0, "strictly increasing"),  # goes back in time
        (20, np.nan, "missing"),  # the signals have a value there
    ],
)
def test_invalid_timestamps_raise(X, position, value, message):
    timestamps = np.arange(X.shape[2], dtype=float)
    timestamps[position] = value
    with pytest.raises(ValueError, match=message):
        BORF(time_channel=True).fit(with_timestamps(X, timestamps))


def test_invalid_timestamps_raise_for_ragged_input(X_padded):
    X = with_timestamps(X_padded, np.arange(X_padded.shape[2], dtype=float))
    X[0, -1, 10] = X[0, -1, 9]
    with pytest.raises(ValueError, match="strictly increasing"):
        BORF(time_channel=True).fit(to_ragged(X))


def test_nan_timestamps_are_allowed_where_signals_are_missing(X):
    X = with_timestamps(X, np.arange(X.shape[2], dtype=float))
    X[:, :, 30] = np.nan  # a gap in every channel, timestamps included
    X[:, :, 50:] = np.nan  # padding
    BORF(time_channel=True).fit(X)


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


def test_invalid_vocabulary_raises(X):
    with pytest.raises(ValueError, match="vocabulary"):
        BORF(vocabulary="seen").fit(X)


def test_full_vocabulary_has_every_word(X):
    borf = BORF(vocabulary="full").fit(X[:5])
    for i, (config, block) in enumerate(zip(borf.configs_, borf.config_slices_)):
        n_words = config["alphabet_size"] ** config["word_length"]
        columns = np.arange(borf.n_signals_ * n_words)
        expected = np.column_stack(
            [np.full(len(columns), i), columns // n_words, columns % n_words]
        )
        np.testing.assert_array_equal(borf.feature_index_[block], expected)


def test_full_vocabulary_does_not_depend_on_training_data(X):
    configs = BORF().fit(X).configs_
    first = BORF(vocabulary="full", configs=configs).fit(X[:5])
    second = BORF(vocabulary="full", configs=configs).fit(X[5:])
    np.testing.assert_array_equal(first.feature_index_, second.feature_index_)
    assert_same_matrix(first.transform(X), second.transform(X))


def test_fit_vocabulary_is_full_vocabulary_without_unseen_words(X):
    full = BORF(vocabulary="full").fit(X[:5])
    fit = BORF(vocabulary="fit").fit(X[:5])
    position = {tuple(row): j for j, row in enumerate(full.feature_index_)}
    kept = [position[tuple(row)] for row in fit.feature_index_]
    X_full = full.transform(X)
    assert_same_matrix(fit.transform(X), X_full[:, kept])
    # Words first seen after fit only exist in the full vocabulary.
    assert X_full.sum() > X_full[:, kept].sum()


@pytest.mark.parametrize("vocabulary", ["fit", "full"])
def test_config_slices_match_feature_index(X, vocabulary):
    borf = BORF(vocabulary=vocabulary).fit(X)
    assert borf.config_slices_[0].start == 0
    assert borf.config_slices_[-1].stop == borf.transform(X).shape[1]
    for i, block in enumerate(borf.config_slices_):
        assert np.all(borf.feature_index_[block, 0] == i)


def blocks(borf, X_counts):
    return [X_counts[:, block] for block in borf.config_slices_]


def test_block_transformer_normalizes_each_config(X):
    plain = BORF().fit(X)
    borf = BORF(block_transformer=Normalizer()).fit(X)
    expected = [
        Normalizer().fit_transform(b) for b in blocks(plain, plain.transform(X))
    ]
    np.testing.assert_allclose(
        borf.transform(X).toarray(), np.hstack([b.toarray() for b in expected])
    )
    np.testing.assert_array_equal(borf.feature_index_, plain.feature_index_)
    assert [s.stop - s.start for s in borf.config_slices_] == [
        s.stop - s.start for s in plain.config_slices_
    ]


@pytest.mark.parametrize(
    "selector",
    [SelectKBest(chi2, k=2), make_pipeline(MaxAbsScaler(), SelectKBest(chi2, k=2))],
    ids=["selector", "pipeline"],
)
def test_block_selection_keeps_word_labels(X, selector):
    y = np.arange(len(X)) % 2
    plain = BORF().fit(X)
    borf = BORF(block_transformer=selector).fit(X, y)
    X_counts = plain.transform(X)
    expected_index = []
    for block in plain.config_slices_:
        fitted = clone(selector).fit(X_counts[:, block], y)
        support = (fitted[-1] if hasattr(fitted, "steps") else fitted).get_support()
        expected_index.append(plain.feature_index_[block][support])
    np.testing.assert_array_equal(borf.feature_index_, np.vstack(expected_index))
    assert borf.transform(X).shape[1] == len(borf.feature_index_)


def test_block_transformer_new_columns_have_no_word(X):
    total = FunctionTransformer(lambda b: np.asarray(b.sum(axis=1)))
    borf = BORF(block_transformer=total).fit(X)
    X_t = borf.transform(X)
    assert X_t.shape[1] == len(borf.configs_)
    np.testing.assert_array_equal(borf.feature_index_[:, 0], range(len(borf.configs_)))
    widths = [
        s.stop - s.start for s in BORF(configs=borf.configs_).fit(X).config_slices_
    ]
    for row, width in zip(borf.feature_index_, widths):
        if width > 1:
            assert row[1] == row[2] == -1


def test_block_transformer_fit_transform_matches_transform(X):
    borf = BORF(block_transformer=Normalizer())
    np.testing.assert_allclose(
        borf.fit_transform(X).toarray(), borf.transform(X).toarray()
    )


@pytest.mark.filterwarnings("error:This Pipeline instance is not fitted")
def test_block_selector_gets_y_in_sklearn_pipeline(X):
    y = np.arange(len(X)) % 2
    borf = BORF(block_transformer=SelectKBest(chi2, k=2))
    scores = cross_val_score(make_pipeline(borf, RidgeClassifier()), X, y, cv=2)
    assert len(scores) == 2


def test_clone_keeps_block_transformer():
    borf = BORF(block_transformer=SelectKBest(chi2, k=3))
    assert clone(borf).get_params()["block_transformer__k"] == 3


def test_explanations_need_fit_vocabulary(X):
    with pytest.raises(NotImplementedError, match="vocabulary"):
        BagOfReceptiveFields(BORF(vocabulary="full").fit(X))


def test_explanations_need_word_columns(X):
    total = FunctionTransformer(lambda b: np.asarray(b.sum(axis=1)))
    with pytest.raises(ValueError, match="block_transformer"):
        BagOfReceptiveFields(BORF(block_transformer=total).fit(X))
