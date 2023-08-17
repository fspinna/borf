from abc import ABC, abstractmethod

from ebop.xai.ebop_explanation import (
    EbopFeatureExplanation,
    EbopFeatureSaliencyExplanation,
    EbopFeaturesShapeSaliencyExplanation,
)
from ebop.classes.ebop_multi import EbopMultiTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union, Optional, Literal
import awkward as ak
import shap
import numpy as np


class Explainer(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def explain(self, *args, **kwargs):
        pass


class EbopShapExplainer(Explainer):
    def __init__(
        self,
        transformer: EbopMultiTransformer,
        model: Optional[Union[ClassifierMixin, RegressorMixin]] = None,
        pipeline: Optional[Pipeline] = None,
        shap_kwargs: Optional[dict] = None,
        X_shapes: Optional[ak.Array] = None,
        X_shapes_ratio: Optional[float] = None,
        labels: Optional[list] = None,
        **kwargs
    ):
        if pipeline is None:
            pipeline = make_pipeline(transformer, model)
        self.pipeline = pipeline
        self.transformer = transformer

        self.shap_kwargs = dict() if shap_kwargs is None else shap_kwargs
        self.X_shapes = X_shapes
        self.X_shapes_ratio = X_shapes_ratio
        self.feature_names = transformer.get_feature_names()
        self.n_features = len(self.feature_names)
        self.labels = labels

        self.shap_explainer_ = None
        self.X_ = None

    def fit(self, X: ak.Array):
        self.shap_explainer_ = shap.Explainer(
            self.pipeline[-1],
            self.pipeline[:-1].transform(X).toarray(),
            **self.shap_kwargs
        )
        self.X_ = X
        if self.X_shapes is None and self.X_shapes_ratio is None:
            self.X_shapes = self.X_
        elif self.X_shapes is None and self.X_shapes_ratio is not None:
            self.X_shapes = self.X_[:: int(len(self.X_) * (1 - self.X_shapes_ratio))]
        return self

    def _shap_explain(self, X, **kwargs):
        explanation_ = self.shap_explainer_(
            self.pipeline[:-1].transform(X).toarray(), **kwargs
        )
        shap_values = explanation_.values
        base_values = self.shap_explainer_.expected_value
        if (
            len(shap_values.shape) == 2
        ):  # if the model is a binary classifier or a regressor
            if isinstance(self.pipeline[-1], ClassifierMixin):
                shap_values = np.concatenate(
                    [-shap_values[:, :, np.newaxis], shap_values[:, :, np.newaxis]],
                    axis=2,
                )
                base_values = np.array([base_values, base_values])
            elif isinstance(self.pipeline[-1], RegressorMixin):
                shap_values = shap_values[:, :, np.newaxis]
                base_values = np.array([base_values])
        n_features = shap_values.shape[1]
        n_labels = shap_values.shape[2]
        return shap_values, base_values, n_features, n_labels

    def _map_saliency_to_receptive_fields(
        self, x, x_transformed, n_labels, shap_values, alignments
    ):
        saliency_list = [
            np.zeros(shape=(np.array(x[0][i]).ravel().size, n_labels))
            for i in range(len(x[0]))
        ]
        # shape: (n_signals, n_timesteps, n_labels)
        for label in range(n_labels):
            for shap_value, alignment in zip(shap_values[0, :, label], alignments):
                signal_idx = alignment[0]
                if len(alignment[1]) == 0:
                    continue
                flat_alignment = np.sort(np.unique(np.ravel(alignment[1])))
                saliency_list[signal_idx][
                    flat_alignment, label
                ] += shap_value  # / len(flat_alignment)
                # for i in alignment[1]:
                #     saliency_list[signal_idx][i, label] += shap_value # / len(i)
        s = ak.Array([saliency_list])
        # shape: (1, n_signals, n_timesteps, n_labels)
        saliency_list = list()
        for label in range(n_labels):
            s_scaled = s[..., label : label + 1]
            s_scaled = s_scaled * (
                shap_values[..., label].ravel()[x_transformed.ravel() > 0].sum()
                / ak.sum(s_scaled)
            )
            saliency_list.append(s_scaled)
        return ak.concatenate(saliency_list, axis=-1)

    def _map_saliencies_to_receptive_fields(
        self, X, X_transformed, n_labels, shap_values, alignments_list
    ):
        return ak.concatenate(
            [
                self._map_saliency_to_receptive_fields(
                    x=X[i : i + 1],
                    x_transformed=X_transformed[i : i + 1],
                    n_labels=n_labels,
                    shap_values=shap_values[i : i + 1],
                    alignments=alignments_list[i],
                )
                for i in range(len(X))
            ]
        )

    def _explain_saliency(self, X, X_transformed, shap_values, n_features, n_labels):
        alignments_list = self.transformer.get_idxs_alignments(
            X, list(range(n_features))
        )
        S = self._map_saliencies_to_receptive_fields(
            X=X,
            X_transformed=X_transformed,
            shap_values=shap_values,
            n_labels=n_labels,
            alignments_list=alignments_list,
        )
        return S, alignments_list

    def _explain_feature_importance(self, n_features):
        receptive_fields = self.transformer.init_receptive_fields(
            list(range(n_features))
        )
        for receptive_field in receptive_fields:
            receptive_field.get_general_shape(
                X=self.X_shapes, normalize=True, cache=True
            )
        return receptive_fields

    def explain(
        self,
        X,
        explanation_type: Literal[
            "features",
            "features_saliency",
            "feature_shapes_saliency",
        ] = "features_shapes_saliency",
        **kwargs
    ):
        shap_values, base_values, n_features, n_labels = self._shap_explain(X, **kwargs)
        y = self.pipeline.predict(X)
        X_transformed = self.pipeline[:-1].transform(X).toarray()

        if explanation_type == "features":
            return [
                EbopFeatureExplanation(
                    x_transformed=X_transformed[i : i + 1],
                    x_pred=y[i],
                    shap_values=shap_values[i : i + 1],
                    base_values=base_values,
                    n_features=n_features,
                    feature_names=self.feature_names,
                    n_labels=n_labels,
                    labels=self.labels,
                )
                for i in range(len(X))
            ]
        elif explanation_type == "features_saliency":
            S, alignments_list = self._explain_saliency(
                X=X,
                X_transformed=X_transformed,
                shap_values=shap_values,
                n_features=n_features,
                n_labels=n_labels,
            )
            return [
                EbopFeatureSaliencyExplanation(
                    x_transformed=X_transformed[i : i + 1],
                    x_pred=y[i],
                    shap_values=shap_values[i : i + 1],
                    base_values=base_values,
                    n_features=n_features,
                    feature_names=self.feature_names,
                    n_labels=n_labels,
                    labels=self.labels,
                    s=S[i : i + 1],
                    alignments=alignments_list[i],
                    x=X[i : i + 1],
                )
                for i in range(len(X))
            ]
        elif explanation_type == "features_shapes_saliency":
            S, alignments_list = self._explain_saliency(
                X=X,
                X_transformed=X_transformed,
                shap_values=shap_values,
                n_features=n_features,
                n_labels=n_labels,
            )
            receptive_fields = self._explain_feature_importance(
                n_features=n_features,
            )
            return [
                EbopFeaturesShapeSaliencyExplanation(
                    x_transformed=X_transformed[i : i + 1],
                    x_pred=y[i],
                    shap_values=shap_values[i : i + 1],
                    base_values=base_values,
                    n_features=n_features,
                    feature_names=self.feature_names,
                    n_labels=n_labels,
                    labels=self.labels,
                    receptive_fields=receptive_fields,
                    s=S[i : i + 1],
                    alignments=alignments_list[i],
                    x=X[i : i + 1],
                )
                for i in range(len(X))
            ]
        else:
            raise NotImplementedError
