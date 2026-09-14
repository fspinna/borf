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
from sklearn.linear_model import RidgeClassifierCV
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


def feature_saliency(explainer, F, count_overlapping):
    explainer.add_feature_importance(F)
    return explainer.map_contained_feature_importance_to_saliency(
        count_overlapping=count_overlapping, normalize="feature"
    ).S_.copy()


@pytest.mark.parametrize("count_overlapping", [True, False])
def test_feature_normalization_gives_each_word_its_importance(X, F, count_overlapping):
    explainer = BagOfReceptiveFields(BORF().fit(X)).build(X, task="regression")
    contained = explainer.X_transformed_.toarray() > 0
    S = feature_saliency(explainer, F, count_overlapping)
    np.testing.assert_allclose(S.sum(axis=(1, 2)), (F * contained).sum(axis=1))
    # Linear in the importances, so every word contributes exactly its own.
    G = np.random.default_rng(2).standard_normal(F.shape)
    np.testing.assert_allclose(
        feature_saliency(explainer, F + G, count_overlapping),
        S + feature_saliency(explainer, G, count_overlapping),
        atol=1e-12,
    )
    for j in (0, 100, 1000):
        single = np.zeros_like(F)
        single[:, j] = F[:, j]
        np.testing.assert_allclose(
            feature_saliency(explainer, single, count_overlapping).sum(axis=(1, 2)),
            F[:, j] * contained[:, j],
        )


@pytest.mark.parametrize("count_overlapping", [True, False])
def test_feature_normalization_spreads_over_the_receptive_field(X, count_overlapping):
    explainer = BagOfReceptiveFields(BORF().fit(X)).build(X, task="regression")
    j = 500
    single = np.zeros((len(X), len(explainer.mapping)))
    single[:, j] = 1.0
    S = feature_saliency(explainer, single, count_overlapping)
    field = explainer.receptive_fields_[j]
    for i, indices in enumerate(field.alignments_indices):
        expected = np.zeros(X.shape[2])
        if len(indices):
            if count_overlapping:  # points weighted by the windows covering them
                expected = np.bincount(indices.ravel(), minlength=X.shape[2])
            else:  # each covered point once
                expected[np.unique(indices)] = 1
            expected = expected / expected.sum()
        np.testing.assert_allclose(S[i, field.signal_idx], expected, atol=1e-12)


def test_feature_normalization_of_absent_words(X, F):
    explainer = BagOfReceptiveFields(BORF().fit(X)).build(X, task="regression")
    explainer.add_feature_importance(F).map_notcontained_feature_importance(
        normalize="feature"
    )
    absent = explainer.X_transformed_.toarray() == 0
    window = np.array([c["window_size"] for c in explainer.configs])[
        explainer.mapping[:, 0]
    ]
    np.testing.assert_allclose(explainer.F_norm_[absent], (F / window)[absent])
    assert np.isnan(explainer.F_norm_[~absent]).all()


def test_invalid_normalization_raises(X, F):
    explainer = BagOfReceptiveFields(BORF().fit(X)).build(X, task="regression")
    explainer.add_feature_importance(F)
    with pytest.raises(ValueError, match="normalize"):
        explainer.map_contained_feature_importance_to_saliency(normalize="series")
    with pytest.raises(ValueError, match="normalize"):
        explainer.map_notcontained_feature_importance(normalize=True)


def test_feature_normalization_finds_a_planted_pattern():
    # Random walks with a bump (class 0) or a dip (class 1) at a random place:
    # the saliency toward the predicted class should peak on the pattern.
    def make(n_series, seed, length=150, width=20):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n_series)
        X = 0.3 * rng.standard_normal((n_series, 1, length)).cumsum(axis=2)
        starts = rng.integers(0, length - width, n_series)
        bump = 3 * np.sin(np.linspace(0, np.pi, width))
        for i in range(n_series):
            X[i, 0, starts[i] : starts[i] + width] += bump if y[i] == 0 else -bump
        return X, y, starts

    X_train, y_train, _ = make(300, seed=0)
    X_test, y_test, starts = make(100, seed=1)
    borf = BORF().fit(X_train)
    Z_train = np.arcsinh(borf.transform(X_train).toarray())
    Z_test = np.arcsinh(borf.transform(X_test).toarray())
    model = RidgeClassifierCV().fit(Z_train, y_train)
    # Linear contributions toward class 1, as SHAP's LinearExplainer computes
    # with the whole training set as background.
    F = (Z_test - Z_train.mean(axis=0)) * np.ravel(model.coef_)
    explainer = BagOfReceptiveFields(borf).build(
        X_test, y_test, model.predict(Z_test), task="classification"
    )
    explainer.add_feature_importance(F)

    def peaks_on_pattern(normalize):
        S = explainer.map_contained_feature_importance_to_saliency(
            normalize=normalize
        ).S_
        peak = S[:, 0].argmax(axis=1)
        return np.mean((starts <= peak) & (peak < starts + 20))

    # The pattern covers 13% of each series; about 62% of the peaks land on it
    # with normalize="feature", 29% with normalize="map".
    assert peaks_on_pattern("feature") > 0.5
    assert peaks_on_pattern("feature") > peaks_on_pattern("map") + 0.2
