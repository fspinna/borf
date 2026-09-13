import numpy as np
import pytest

from fast_borf.core import (
    breakpoints,
    decode_words,
    discretize,
    encode_words,
    panel_words,
    sax,
    segment_means,
    transform_sax_patterns,
    window_positions,
)

CONFIG = dict(window_size=16, word_length=4, stride=2, dilation=2)


@pytest.fixture
def signal():
    return np.random.default_rng(0).standard_normal(80).cumsum()


def test_sax_is_discretized_segment_means(signal):
    timestamps = np.cumsum(np.random.default_rng(1).exponential(1.0, signal.size))
    bins = breakpoints(4)
    means = segment_means(signal, timestamps, **CONFIG)
    np.testing.assert_array_equal(
        sax(signal, timestamps, bins=bins, **CONFIG), discretize(means, bins)
    )


def test_segment_means_on_evenly_spaced_signal(signal):
    # With even spacing, each segment mean is the plain mean of its points,
    # z-normalized with the plain mean and std of the window's points.
    means = segment_means(signal, np.arange(signal.size, dtype=float), **CONFIG)
    positions = window_positions(len(means), **CONFIG)
    for window_means, segments in zip(means, positions):
        window = signal[segments.ravel()]
        expected = (signal[segments].mean(axis=1) - window.mean()) / window.std()
        np.testing.assert_allclose(window_means, expected, atol=1e-10)


def test_constant_signal_gets_first_symbol():
    signal = np.full(40, 2.5)
    symbols = sax(
        signal, np.arange(40.0), window_size=8, word_length=4, bins=breakpoints(3)
    )
    assert symbols.shape == (33, 4)
    assert np.all(symbols == 0)


def test_breakpoints_split_the_normal_into_equal_parts():
    np.testing.assert_allclose(breakpoints(4), [-0.6745, 0.0, 0.6745], atol=1e-3)


def test_encode_decode_round_trip():
    symbols = np.random.default_rng(0).integers(0, 5, (20, 6))
    words = encode_words(symbols, 5)
    np.testing.assert_array_equal(decode_words(words, 5, 6), symbols)
    assert encode_words(np.array([[1, 0, 2]]), 3)[0] == 1 * 9 + 0 * 3 + 2


def test_window_positions():
    positions = window_positions(3, window_size=8, word_length=2, stride=3, dilation=2)
    np.testing.assert_array_equal(positions[0], [[0, 2, 4, 6], [8, 10, 12, 14]])
    np.testing.assert_array_equal(positions[2], positions[0] + 6)


def test_panel_words_counts_match_transform():
    rng = np.random.default_rng(2)
    panel = rng.standard_normal((5, 2, 80)).cumsum(axis=2)
    panel[1, 0, 60:] = np.nan
    panel[3, 1, [5, 17, 30]] = np.nan
    config = dict(CONFIG, window_size=8, alphabet_size=3)
    words, word_offsets, observed, observed_offsets = panel_words(panel, None, **config)
    rows = []
    for g in range(len(word_offsets) - 1):
        values, counts = np.unique(
            words[word_offsets[g] : word_offsets[g + 1]], return_counts=True
        )
        rows += [(g // 2, g % 2, v, c) for v, c in zip(values, counts)]
    counted = transform_sax_patterns(panel, None, **config)
    assert sorted(rows) == sorted(map(tuple, counted.tolist()))
    # observed maps back to the non-NaN points of each signal.
    g = 3 * 2 + 1
    np.testing.assert_array_equal(
        observed[observed_offsets[g] : observed_offsets[g + 1]],
        np.flatnonzero(~np.isnan(panel[3, 1])),
    )
