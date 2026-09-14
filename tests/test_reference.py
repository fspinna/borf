"""Compare the current code against outputs saved from the pre-refactor code.

The files in tests/reference/ were generated at the commit recorded in
provenance.json, before any refactoring. They pin the behaviour the cleanup
must preserve: the SAX word counts, the feature matrices produced by the
original pipeline builder, and the configurations chosen by the heuristic.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from fast_borf import BORF
from fast_borf.core.transform import transform_sax_patterns
from fast_borf.heuristic import generate_configs

REFERENCE_DIR = Path(__file__).parent / "reference"
PROVENANCE = json.loads((REFERENCE_DIR / "provenance.json").read_text())
CASES = sorted(path.stem for path in REFERENCE_DIR.glob("*.npz"))
CORE_CONFIGS = PROVENANCE["core_configs"]
CORE_STD_RATIOS = PROVENANCE["core_std_ratios"]
TRAIN_FRACTION = 0.7
# The reference heuristic outputs were generated with a single alphabet of size 2.
HEURISTIC_ALPHABET_SIZES = (2,)


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


def reference_matrix(data, prefix):
    return sp.csr_matrix(
        (data[f"{prefix}/data"], data[f"{prefix}/indices"], data[f"{prefix}/indptr"]),
        shape=tuple(data[f"{prefix}/shape"]),
    )


def assert_same_matrix(actual, expected):
    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0


@pytest.mark.parametrize("ratio", CORE_STD_RATIOS)
@pytest.mark.parametrize("config_idx", range(len(CORE_CONFIGS)))
def test_core_transform_matches_reference(case, config_idx, ratio):
    _, data = case
    rows = transform_core(data["X"], data["T"], CORE_CONFIGS[config_idx], ratio)
    np.testing.assert_array_equal(rows, data[f"core/{config_idx}/{ratio}"])


@pytest.mark.parametrize("timestamps", ["none", "rescaled"])
@pytest.mark.parametrize("ratio", CORE_STD_RATIOS)
@pytest.mark.parametrize("config_idx", range(len(CORE_CONFIGS)))
def test_evenly_spaced_timestamps_match_unweighted_sax(
    case, config_idx, ratio, timestamps
):
    # The unweighted reference comes from the original (unweighted) SAX. The
    # time unit and origin must not matter, so the timestamps are either
    # omitted or rescaled by a power of two (exact in floating point) and
    # shifted.
    name, data = case
    key = f"unweighted/{config_idx}/{ratio}"
    if key not in data:
        pytest.skip(f"{name} has irregular timestamps")
    X = data["X"]
    T = None
    if timestamps == "rescaled":
        T = np.tile(0.25 * np.arange(X.shape[2]) + 100.0, (len(X), 1, 1))
    rows = transform_core(X, T, CORE_CONFIGS[config_idx], ratio)
    np.testing.assert_array_equal(rows, data[key])


@pytest.mark.parametrize("time_channel", [True, False])
def test_borf_matches_reference(case, time_channel):
    name, data = case
    X = data["X"]
    if time_channel:
        X = np.concatenate([X, data["T"]], axis=1)
    n_train = int(len(X) * TRAIN_FRACTION)
    borf = BORF(time_channel=time_channel)
    X_train = borf.fit_transform(X[:n_train])

    tag = "with_time" if time_channel else "no_time"
    assert borf.configs_ == PROVENANCE["pipeline_configs"][name][tag]
    assert_same_matrix(X_train, reference_matrix(data, f"pipeline/{tag}/train"))
    assert_same_matrix(
        borf.transform(X[:n_train]), reference_matrix(data, f"pipeline/{tag}/train")
    )
    assert_same_matrix(
        borf.transform(X[n_train:]), reference_matrix(data, f"pipeline/{tag}/test")
    )


@pytest.mark.parametrize("length", sorted(PROVENANCE["heuristic"], key=int))
def test_heuristic_matches_reference(length):
    configs = generate_configs(
        int(length), int(length), alphabet_sizes=HEURISTIC_ALPHABET_SIZES
    )
    assert configs == PROVENANCE["heuristic"][length]


def test_heuristic_with_variable_lengths_matches_reference():
    configs = generate_configs(20, 100, alphabet_sizes=HEURISTIC_ALPHABET_SIZES)
    assert configs == PROVENANCE["heuristic_min_max"]
