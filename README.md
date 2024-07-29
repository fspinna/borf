# BORF
## Installation
Install the package using pip by navigating to the directory containing `setup.py` and running:
```bash
pip install .
```

## Basic Usage
Below is an example that demonstrates how to create and use a machine learning pipeline with BORF:

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier
import numpy as np
from fast_borf import BorfBuilder
from fast_borf.pipeline.zero_columns_remover import ZeroColumnsRemover
from fast_borf.pipeline.reshaper import ReshapeTo2D
from fast_borf.pipeline.to_scipy import ToScipySparse


# Create a dummy dataset
X = np.random.rand(10, 1, 100)
y = np.random.randint(0, 2, 10)

# Setup the BORF builder
builder = BorfBuilder(
    pipeline_objects=[
        (ZeroColumnsRemover, {}),  # Remove columns with all zeros
        (ReshapeTo2D, {}),  # Reshape the data to 2D
        (ToScipySparse, {}),  # Convert the sparse tensor to a scipy sparse matrix
    ],
)
borf = builder.build(X)
pipe = make_pipeline(borf, RidgeClassifier())

# Transform and train
X_transformed = borf.fit_transform(X)
pipe.fit(X, y)
score = pipe.score(X, y)
```

## Explanation
An example of to get an explanation with BORF can be found in the notebook `xai_example.ipynb`.