"""Explain a linear model on BORF features, without SHAP.

Same data as classification.py: random walks with a bump (class 0) or a dip
(class 1). For a linear model, coefficient times feature value is each
feature's contribution to the decision, which serves as the importance. Any
other per-series importances (SHAP values, for instance) work the same way.
"""

import numpy as np
from classification import make_dataset
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from fast_borf import BORF
from fast_borf.xai import BagOfReceptiveFields


def main():
    X, y = make_dataset()
    model = make_pipeline(BORF(n_jobs=-1), RidgeClassifierCV()).fit(X, y)
    borf, classifier = model[0], model[-1]

    X_test, y_test = make_dataset(n_series=20, seed=1)
    y_pred = model.predict(X_test)
    # Contributions toward class 1; for 2 classes the explainer uses their
    # negation for series predicted as class 0.
    F = borf.transform(X_test).multiply(classifier.coef_[0]).toarray()

    explainer = BagOfReceptiveFields(borf).build(
        X_test, y_test, y_pred, task="classification"
    )
    explainer.add_feature_importance(F)
    explainer.map_contained_feature_importance_to_saliency(count_overlapping=False)

    # S_ has one value per point: (n_series, n_signals, n_timestamps).
    print("saliency of the first series:", np.round(explainer.S_[0, 0, :8], 4), "...")

    top = explainer.F_avg_rank_argsort_[0]
    field = explainer.receptive_fields_[top]
    print("most important pattern:", field)
    i = next(i for i, a in enumerate(field.alignments_indices) if len(a))
    occurrence = field.alignments_indices[i][0]
    print(f"first occurrence in series {i}, points of each segment:\n{occurrence}")
    print(f"values of those points:\n{np.round(field.mappings[i][0], 2)}")


if __name__ == "__main__":
    main()
