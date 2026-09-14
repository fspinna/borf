import awkward as ak
import numpy as np
import pytest
from sklearn.base import clone

from fast_borf import BORF
from fast_borf.xai import BagOfReceptiveFields

N_CHANNELS = 3


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, N_CHANNELS, 40)).cumsum(axis=2)


@pytest.fixture(scope="module")
def configs(X):
    return BORF().fit(X).configs_


def summed_over_groups(X, configs, groups):
    """Default full-vocabulary counts, summed over each group's channels by hand."""
    borf = BORF(configs=configs, vocabulary="full").fit(X)
    counts = borf.transform(X).toarray()
    blocks = []
    for config, block in zip(borf.configs_, borf.config_slices_):
        n_words = config["alphabet_size"] ** config["word_length"]
        per_channel = counts[:, block].reshape(len(X), N_CHANNELS, n_words)
        grouped = np.stack([per_channel[:, g].sum(axis=1) for g in groups], axis=1)
        blocks.append(grouped.reshape(len(X), -1))
    return np.hstack(blocks)


@pytest.mark.parametrize(
    "channel_groups, groups",
    [
        ("all", [[0, 1, 2]]),
        ([[0, 2], [1]], [[0, 2], [1]]),
        ([[0, 1], [1, 2]], [[0, 1], [1, 2]]),  # overlapping groups
        ([[2]], [[2]]),  # other channels ignored
    ],
)
def test_groups_sum_counts_over_their_channels(X, configs, channel_groups, groups):
    borf = BORF(configs=configs, vocabulary="full", channel_groups=channel_groups)
    np.testing.assert_array_equal(
        borf.fit(X).transform(X).toarray(), summed_over_groups(X, configs, groups)
    )
    assert borf.channel_groups_ == groups
    assert borf.feature_index_[:, 1].max() == len(groups) - 1


@pytest.mark.parametrize("channel_groups", ["all", [[0, 2], [1]]])
def test_fit_vocabulary_keeps_the_words_seen_in_each_group(X, configs, channel_groups):
    full = BORF(configs=configs, vocabulary="full", channel_groups=channel_groups)
    fit = BORF(configs=configs, channel_groups=channel_groups)
    X_full = full.fit(X[:5]).transform(X)
    X_fit = fit.fit(X[:5]).transform(X)
    position = {tuple(row): j for j, row in enumerate(full.feature_index_)}
    kept = [position[tuple(row)] for row in fit.feature_index_]
    np.testing.assert_array_equal(X_fit.toarray(), X_full[:, kept].toarray())
    assert np.all(X_full[:5].toarray().sum(axis=0)[kept] > 0)


def test_single_channel_groups_match_default(X):
    default = BORF().fit(X)
    grouped = BORF(channel_groups=[[0], [1], [2]]).fit(X)
    np.testing.assert_array_equal(grouped.feature_index_, default.feature_index_)
    assert (grouped.transform(X) != default.transform(X)).nnz == 0


def test_groups_with_ragged_input_and_time_channel(X):
    X = X.copy()
    X[:3, :, 30:] = np.nan
    timestamps = np.cumsum(np.random.default_rng(1).exponential(1.0, X[:, :1].shape), 2)
    timestamps[np.isnan(X[:, :1])] = np.nan
    X = np.concatenate([X, timestamps], axis=1)
    ragged = ak.Array([[c[~np.isnan(c)].tolist() for c in series] for series in X])
    padded = BORF(channel_groups="all", time_channel=True).fit(X)
    from_ragged = BORF(configs=padded.configs_, channel_groups="all", time_channel=True)
    assert (from_ragged.fit(ragged).transform(ragged) != padded.transform(X)).nnz == 0


@pytest.mark.parametrize(
    "channel_groups, message",
    [
        ("some", "all"),
        ([], "empty"),
        ([[0], []], "empty group"),
        ([[0, 0]], "twice"),
        ([[0, 3]], "outside"),
    ],
)
def test_invalid_groups_raise(X, channel_groups, message):
    with pytest.raises(ValueError, match=message):
        BORF(channel_groups=channel_groups).fit(X)


def test_clone_keeps_channel_groups():
    borf = BORF(channel_groups=[[0, 1], [2]])
    assert clone(borf).get_params()["channel_groups"] == [[0, 1], [2]]


def test_explanations_of_groups_are_not_supported_yet(X):
    with pytest.raises(NotImplementedError, match="channel_groups"):
        BagOfReceptiveFields(BORF(channel_groups="all").fit(X))
