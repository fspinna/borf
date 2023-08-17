# ebop

The `ebop` repository contains code for the Extended-Bag-Of-Patterns (EBOP)
## Installation

You can install the package from source with:

```bash
pip install -e ebop
```

To install extra dependencies for the notebooks and explainability, use:

```bash
pip install -e ebop[notebooks,xai]
```

---
## Quick Start

The `EBOP` transformer is designed to integrate seamlessly with scikit-learn's pipeline mechanism, allowing it to be used just like any other transformer. Here's a quick example to get you started:

### Importing the Necessary Libraries

```python
from ebop import EBOP
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
```

### Creating a Generic Pipeline with EBOP

Create a pipeline using `EBOP` as a transformer and `LinearSVC` (or any other classifier/regressor) as the final estimator:

```python
pipe = make_pipeline(EBOP(), LinearSVC())
```

### Training and Scoring

You can now use this pipeline to fit your training data, predict new samples, and score the model, just like any standard scikit-learn pipeline:

```python
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

The `EBOP` transformer's integration with the standard pipeline ensures a consistent and easy-to-use interface, making it a flexible tool for various machine learning tasks.

For more specific examples on classification and regression, please refer to the detailed sections below.


## Usage


### Importing the Required Modules

```python
from ebop import EBOP
from downtime import load_dataset
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.pipeline import make_pipeline
```

### Classification

Load a dataset for classification (needs to be an awkward array):

```python
d = load_dataset('CharacterTrajectories')
X_train, y_train, X_test, y_test = d()
print(d)
```

Create and fit the pipeline with EBOP and LinearSVC:

```python
pipe = make_pipeline(EBOP(n_jobs=-1), LinearSVC())
pipe.fit(X_train, y_train)
```

Score the model on the test set:

```python
pipe.score(X_test, y_test)
```

### Regression

Load a dataset for regression:

```python
d = load_dataset('AppliancesEnergy')
X_train, y_train, X_test, y_test = d()
print(d)
```

Create and fit the pipeline with EBOP and LinearSVR:

```python
pipe = make_pipeline(EBOP(n_jobs=-1), LinearSVR())
pipe.fit(X_train, y_train)
```

Score the model on the test set:

```python
pipe.score(X_test, y_test)
```

## Performance Note

The first run may be slower due to JIT compilation.

## License

Please refer to the license file in the repository for more information.

## Support

For any additional support or information, please refer to the repository's documentation or raise an issue on the GitHub repository.
