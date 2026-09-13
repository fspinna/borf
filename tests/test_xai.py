"""Tests for the explanations.

The files in tests/reference/xai/ were generated with the explanation code
before it was rewritten (commit in provenance.json), for a binary problem with
irregular timestamps and a 3-class problem with 3 regularly sampled signals.
"""

import json
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import FunctionTransformer

from fast_borf import BORF
from fast_borf.core import breakpoints
from fast_borf.xai import BagOfReceptiveFields

REFERENCE_DIR = Path(__file__).parent / "reference"
XAI_CASES = json.loads((REFERENCE_DIR / "xai" / "provenance.json").read_text())["cases"]


def explain(borf, X, F, y_pred, task="classification", **saliency):
    explainer = BagOfReceptiveFields(borf).build(X, y_pred, y_pred, task=task)
    explainer.add_feature_importance(F)
    explainer.map_contained_feature_importance_to_saliency(**saliency)
    explainer.map_notcontained_feature_importance()
    return explainer


@pytest.fixture(scope="module", params=sorted(XAI_CASES))
def reference(request):
    case = request.param
    time_channel = XAI_CASES[case]["time_channel"]
    data = np.load(REFERENCE_DIR / f"{case}.npz")
    X = data["X"]
    if time_channel:
        X = np.concatenate([X, data["T"]], axis=1)
    with np.load(REFERENCE_DIR / "xai" / f"{case}.npz") as expected:
        expected = dict(expected)
    borf = BORF(time_channel=time_channel).fit(X)
    return X, borf, expected


@pytest.mark.parametrize("count_overlapping", [True, False])
def test_explanation_matches_reference(reference, count_overlapping):
    X, borf, expected = reference
    F = expected["F"].astype(np.float64)
    explainer = explain(
        borf, X, F, expected["y_pred"], count_overlapping=count_overlapping
    )
    np.testing.assert_array_equal(explainer.mapping, expected["mapping"])
    np.testing.assert_array_equal(explainer.F_, expected["F_selected"])
    np.testing.assert_allclose(
        explainer.S_, expected[f"S/{count_overlapping}"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        explainer.F_norm_, expected["F_norm"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(
        explainer.F_avg_rank_argsort_, expected["F_avg_rank_argsort"]
    )


def test_receptive_fields_match_reference(reference):
    X, borf, expected = reference
    explainer = explain(borf, X, expected["F"].astype(np.float64), expected["y_pred"])
    for j in expected["fields"]:
        field = explainer.receptive_fields_[j]
        counts = [len(indices) for indices in field.alignments_indices]
        np.testing.assert_array_equal(counts, expected[f"rf/{j}/counts"])
        np.testing.assert_array_equal(
            np.concatenate(field.alignments_indices), expected[f"rf/{j}/indices"]
        )
        for i, indices in enumerate(field.alignments_indices):
            np.testing.assert_array_equal(
                field.mappings[i], explainer.X_[i, field.signal_idx][indices]
            )
            np.testing.assert_array_equal(
                field.alignments[i], explainer.timestamps_[i, 0][indices]
            )


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((12, 2, 64)).cumsum(axis=2)


@pytest.fixture
def F(X):
    borf = BORF().fit(X)
    return np.random.default_rng(1).standard_normal((len(X), len(borf.feature_index_)))


def test_segments_reproduce_the_word(X):
    # Recompute every occurrence's word from the points of its segments.
    borf = BORF().fit(X)
    explainer = BagOfReceptiveFields(borf).build(X, task="regression")
    for j in range(0, len(borf.feature_index_), 97):
        field = explainer.receptive_fields_[j]
        bins = breakpoints(field.alphabet_size)
        for i, occurrences in enumerate(field.alignments_indices):
            signal = X[i, field.signal_idx]
            for segments in occurrences:
                window = signal[segments.ravel()]
                means = signal[segments].mean(axis=1)
                symbols = np.digitize((means - window.mean()) / window.std(), bins)
                np.testing.assert_array_equal(symbols, field.word_array)


def test_saliency_sums_to_importance_of_contained_words(X, F):
    explainer = explain(BORF().fit(X), X, F, None, task="regression")
    contained = explainer.X_transformed_.toarray() > 0
    np.testing.assert_allclose(
        explainer.S_.sum(axis=(1, 2)), (F * contained).sum(axis=1)
    )


def test_unnormalized_saliency_counts_covering_windows(X):
    borf = BORF().fit(X)
    ones = np.ones((len(X), len(borf.feature_index_)))
    explainer = explain(borf, X, ones, None, task="regression", normalize=False)
    # Every point is covered by at least one window of the smallest configuration.
    assert np.all(explainer.S_ >= 1)
    once = explain(
        borf, X, ones, None, task="regression", normalize=False, count_overlapping=False
    )
    assert np.all(once.S_ <= explainer.S_)


def test_ragged_input_matches_padded(X):
    X = X.copy()
    X[:4, :, 40:] = np.nan
    ragged = ak.Array([[s[~np.isnan(s)].tolist() for s in series] for series in X])
    borf = BORF().fit(X)
    F = np.random.default_rng(1).standard_normal((len(X), len(borf.feature_index_)))
    padded = explain(borf, X, F, None, "regression")
    from_ragged = explain(borf, ragged, F, None, "regression")
    np.testing.assert_allclose(from_ragged.S_, padded.S_)
    for j in (0, 50, 500):
        for a, b in zip(
            from_ragged.receptive_fields_[j].alignments_indices,
            padded.receptive_fields_[j].alignments_indices,
        ):
            np.testing.assert_array_equal(a, b)


def test_full_vocabulary_matches_fit_vocabulary(X, F):
    fit = BORF().fit(X)
    full = BORF(vocabulary="full").fit(X)
    position = {tuple(row): j for j, row in enumerate(full.feature_index_)}
    F_full = np.zeros((len(X), len(full.feature_index_)))
    F_full[:, [position[tuple(row)] for row in fit.feature_index_]] = F
    explainer_fit = explain(fit, X, F, None, "regression")
    explainer_full = explain(full, X, F_full, None, "regression")
    np.testing.assert_allclose(explainer_full.S_, explainer_fit.S_)


def test_selected_features_keep_their_receptive_fields(X):
    y = np.arange(len(X)) % 2
    plain = BORF().fit(X)
    selected = BORF(block_transformer=SelectKBest(chi2, k=3)).fit(X, y)
    plain_explainer = BagOfReceptiveFields(plain).build(X, task="regression")
    explainer = BagOfReceptiveFields(selected).build(X, task="regression")
    position = {tuple(row): j for j, row in enumerate(plain.feature_index_)}
    for j in range(0, len(selected.feature_index_), 7):
        field = explainer.receptive_fields_[j]
        same = plain_explainer.receptive_fields_[
            position[tuple(selected.feature_index_[j])]
        ]
        for a, b in zip(field.alignments_indices, same.alignments_indices):
            np.testing.assert_array_equal(a, b)


def test_receptive_fields_are_a_lazy_mapping(X, F):
    explainer = BagOfReceptiveFields(BORF().fit(X)).build(X, task="regression")
    assert len(explainer.receptive_fields_) == F.shape[1]
    assert explainer[3] is explainer.receptive_fields_[3]
    assert explainer[3].feature_importance is None
    explainer.add_feature_importance(F)
    np.testing.assert_array_equal(explainer[3].feature_importance, F[:, 3])
    with pytest.raises(KeyError):
        explainer.receptive_fields_[F.shape[1]]


def test_columns_created_by_block_transformer_cannot_be_explained(X):
    total = FunctionTransformer(lambda b: np.asarray(b.sum(axis=1)))
    with pytest.raises(ValueError, match="block_transformer"):
        BagOfReceptiveFields(BORF(block_transformer=total).fit(X))
