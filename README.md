# fast-borf

Bag-Of-Receptive-Fields (BORF) turns time series into sparse bag-of-words
features. Each signal is cut into sliding windows at many window sizes, word
lengths and dilations, every window becomes a SAX word, and the output counts
how often each word occurs. The features are meant for linear models and can
be mapped back to the parts of the series they come from.

BORF handles univariate and multivariate series, variable lengths, missing
values and irregular sampling, and is a scikit-learn transformer.

BORF is now available in the aeon library!

https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.dictionary_based.BORF.html

For a more customizable estimator and for XAI, continue below.

## Installation

```bash
pip install "git+https://github.com/fspinna/borf"
```

The original implementation for regularly sampled series is available as
release [v0.1.0](https://github.com/fspinna/borf/releases/tag/v0.1.0).

## Quick start

```python
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from fast_borf import BORF

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 1, 200)).cumsum(axis=2)  # (n_series, n_signals, n_timestamps)
y = rng.integers(0, 2, 100)

model = make_pipeline(BORF(n_jobs=-1), RidgeClassifierCV())
model.fit(X, y)
```

See [`examples/`](examples) for complete scripts.

## Input formats

- **NumPy array** of shape `(n_series, n_signals, n_timestamps)`. Use NaN for
  missing values and to pad shorter series.
- **Ragged [awkward](https://awkward-array.org) array** with the same nesting,
  for series of different lengths.
- **Irregular sampling**: put the timestamps of each series in an extra last
  channel and use `BORF(time_channel=True)`. Timestamps must be strictly
  increasing within each series. Without a time channel, observations are
  treated as evenly spaced.

## Main parameters

| parameter | default | meaning |
|---|---|---|
| `min_window_size`, `max_window_size` | 4, longest series | window sizes, powers of two in this range |
| `max_word_length` | 8 | word lengths are the powers of two from 2 up to this |
| `alphabet_sizes` | `(3,)` | SAX alphabet sizes |
| `min_dilation`, `max_dilation` | 1, log2(longest series) | dilations, powers of two in this range |
| `complexity` | `"quadratic"` | stride choice: `"quadratic"` (stride 1), `"linear"` (stride = word length) or `"linear_logarithmic"` |
| `configs` | `None` | explicit list of configurations, instead of the parameters above |
| `min_window_to_signal_std_ratio` | 0.0 | windows flatter than this fraction of the signal's standard deviation count as flat |
| `channel_groups` | `None` | `"all"` or e.g. `[[0, 1, 2], [3, 4, 5]]`: count each word summed over the channels of each group (not yet supported by the explanations) |
| `vocabulary` | `"fit"` | `"fit"`: a column per word seen during fit; `"full"`: every possible word, so the feature space does not depend on the data |
| `block_transformer` | `None` | scikit-learn transformer applied separately to each configuration's columns |
| `time_channel` | `False` | the last channel holds timestamps |
| `n_jobs` | 1 | numba threads, -1 for all cores |

## Inspecting the features

After fitting:

- `configs_`: the configurations (window size, word length, alphabet size,
  dilation, stride), in the order of the output columns.
- `config_slices_[i]`: the output columns of configuration `i`.
- `feature_index_`: for every output column, its configuration index, signal
  index (group index with `channel_groups`) and SAX word (as an integer in
  base `alphabet_size`).

## Blockwise processing

`block_transformer` fits a copy of any scikit-learn transformer or pipeline on
each configuration's columns, for example to normalize or select features per
configuration:

```python
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer

BORF(block_transformer=Normalizer())
BORF(block_transformer=SelectKBest(chi2, k=10))  # 10 words per configuration
```

`feature_index_` and `config_slices_` follow columns that the transformer keeps
or selects.

## Explanations

`fast_borf.xai.BagOfReceptiveFields` maps per-series feature importances back
onto the series. The importances can come from anything: SHAP values, or for a
linear model simply coefficient times feature value.

```python
from fast_borf.xai import BagOfReceptiveFields

explainer = BagOfReceptiveFields(borf).build(X, y_true, y_pred, task="classification")
explainer.add_feature_importance(F)  # (n_series, n_features), or (n_classes, n_series, n_features)
explainer.map_contained_feature_importance_to_saliency()  # S_: importance per point
explainer.map_notcontained_feature_importance()  # F_norm_: importance of absent words

field = explainer.receptive_fields_[explainer.F_avg_rank_argsort_[0]]
field.alignments_indices[i]  # (occurrences, word_length, segment_size) points of each segment
field.mappings[i], field.alignments[i]  # their values and timestamps
```

Receptive fields are computed when first accessed. `fast_borf.core` exposes the
underlying steps (`segment_means`, `discretize`, `window_positions`,
`panel_words`) for custom analyses and plots. See
[`examples/explanation.py`](examples/explanation.py).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

The tests in `tests/test_reference.py` compare against outputs saved from the
original implementation, so refactoring cannot silently change the features.
