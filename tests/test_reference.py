"""Compare the current code against outputs saved from the pre-refactor code.

The files in tests/reference/ were generated at the commit recorded in
provenance.json, before any refactoring. They pin the behaviour the cleanup
must preserve: the SAX word counts, the feature matrices produced by the
pipeline, and the configurations chosen by the heuristic.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from fast_borf.core.transform import transform_sax_patterns
from fast_borf.heuristic import heuristic_function_sax
from fast_borf.pipeline.borf_multi import BorfPipelineBuilder
from fast_borf.pipeline.reshaper import ReshapeTo2D
from fast_borf.pipeline.to_scipy import ToScipySparse
from fast_borf.pipeline.zero_columns_remover import ZeroColumnsRemover

REFERENCE_DIR = Path(__file__).parent / "reference"
PROVENANCE = json.loads((REFERENCE_DIR / "provenance.json").read_text())
CASES = sorted(path.stem for path in REFERENCE_DIR.glob("*.npz"))
CORE_CONFIGS = PROVENANCE["core_configs"]
CORE_STD_RATIOS = PROVENANCE["core_std_ratios"]
TRAIN_FRACTION = 0.7


@pytest.fixture(scope="module", params=CASES)
def case(request):
    with np.load(REFERENCE_DIR / f"{request.param}.npz") as data:
        return request.param, dict(data)


def sort_rows(a):
    return a[np.lexsort(a.T[::-1])] if len(a) else a


def transform_core(X, T, config, ratio):
    rows = transform_sax_patterns(
        panel=X,
        panel_timestamps=T,
        min_window_to_signal_std_ratio=ratio,
        **config,
    )
    # Rows are (ts_idx, signal_idx, word, count); their order is not part of
    # the contract.
    return sort_rows(rows)


@pytest.mark.parametrize("ratio", CORE_STD_RATIOS)
@pytest.mark.parametrize("config_idx", range(len(CORE_CONFIGS)))
def test_core_transform_matches_reference(case, config_idx, ratio):
    _, data = case
    rows = transform_core(data["X"], data["T"], CORE_CONFIGS[config_idx], ratio)
    np.testing.assert_array_equal(rows, data[f"core/{config_idx}/{ratio}"])


@pytest.mark.parametrize("ratio", CORE_STD_RATIOS)
@pytest.mark.parametrize("config_idx", range(len(CORE_CONFIGS)))
def test_evenly_spaced_timestamps_match_unweighted_sax(case, config_idx, ratio):
    # The unweighted reference comes from the original (unweighted) SAX. The
    # time unit and origin must not matter, so the timestamps are rescaled by a
    # power of two (exact in floating point) and shifted.
    name, data = case
    key = f"unweighted/{config_idx}/{ratio}"
    if key not in data:
        pytest.skip(f"{name} has irregular timestamps")
    X = data["X"]
    T = np.tile(0.25 * np.arange(X.shape[2]) + 100.0, (len(X), 1, 1))
    rows = transform_core(X, T, CORE_CONFIGS[config_idx], ratio)
    np.testing.assert_array_equal(rows, data[key])


@pytest.mark.parametrize("contains_time_idx", [True, False])
def test_pipeline_matches_reference(case, contains_time_idx):
    name, data = case
    X = data["X"]
    if contains_time_idx:
        X = np.concatenate([X, data["T"]], axis=1)
    n_train = int(len(X) * TRAIN_FRACTION)
    builder = BorfPipelineBuilder(
        pipeline_objects=[
            (ReshapeTo2D, {}),
            (ZeroColumnsRemover, {}),
            (ToScipySparse, {}),
        ],
        contains_time_idx=contains_time_idx,
    )
    pipe = builder.build(X[:n_train])
    pipe.fit(X[:n_train])

    tag = "with_time" if contains_time_idx else "no_time"
    assert builder.configs_ == PROVENANCE["pipeline_configs"][name][tag]
    for split, part in (("train", X[:n_train]), ("test", X[n_train:])):
        prefix = f"pipeline/{tag}/{split}"
        expected = sp.csr_matrix(
            (
                data[f"{prefix}/data"],
                data[f"{prefix}/indices"],
                data[f"{prefix}/indptr"],
            ),
            shape=tuple(data[f"{prefix}/shape"]),
        )
        actual = sp.csr_matrix(pipe.transform(part))
        assert actual.shape == expected.shape
        assert (actual != expected).nnz == 0


@pytest.mark.parametrize("length", sorted(PROVENANCE["heuristic"], key=int))
def test_heuristic_matches_reference(length):
    configs = heuristic_function_sax(int(length), int(length))
    assert configs == PROVENANCE["heuristic"][length]


def test_heuristic_with_variable_lengths_matches_reference():
    assert heuristic_function_sax(20, 100) == PROVENANCE["heuristic_min_max"]
