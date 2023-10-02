from abc import ABC

import awkward as ak
import numpy as np
from matplotlib.colors import TwoSlopeNorm, CenteredNorm
from matplotlib.cm import coolwarm, ScalarMappable
from plottime.plot import plot_saliency
import matplotlib.pyplot as plt
import xarray as xr


class Explanation(ABC):
    pass


class BorfFeatureExplanation(Explanation):
    def __init__(
        self,
        x_transformed,
        x_pred,
        shap_values,
        base_values,
        n_features,
        feature_names,
        n_labels,
        labels=None,
    ):
        self.x_transformed = x_transformed
        self.x_pred = x_pred
        self.shap_values = shap_values
        self.abs_shap_values = np.abs(shap_values)
        self.base_values = base_values
        self.n_features = n_features
        self.feature_names = feature_names
        self.n_labels = n_labels
        if labels is None:
            if n_labels == 1:
                labels = [str(x_pred)]
            else:
                labels = [str(i) for i in range(n_labels)]
        self.labels = labels

        self.x_transformed_window_sizes = np.array(
            [
                int(feature_names[i].split(";")[0].split("_")[-2])
                for i in range(len(feature_names))
            ]
        )
        self.signal_idxs = np.array(
            [int(feature_names[i].split(";")[1]) for i in range(len(feature_names))]
        )
        self.shap_values_length_ratio = (
            self.shap_values / self.x_transformed_window_sizes[:, np.newaxis]
        )
        self.abs_shap_values_length_ratio = np.abs(self.shap_values_length_ratio)

        dfs = np.concatenate(
            [
                np.array(
                    [
                        np.arange(self.n_features).astype(int),
                        self.signal_idxs.astype(int),
                        self.shap_values[..., label].ravel(),
                        self.abs_shap_values[..., label].ravel(),
                        self.shap_values_length_ratio[..., label].ravel(),
                        self.abs_shap_values_length_ratio[..., label].ravel(),
                        self.x_transformed.ravel(),
                    ]
                )[..., np.newaxis]
                for label in range(self.n_labels)
            ],
            axis=-1,
        )
        # shape: (3, n_features, n_labels)
        # self.dfs = dfs
        self.word_importance = xr.DataArray(
            dfs,
            dims=["rows", "words", "labels"],
            coords={
                "labels": labels,
                "words": feature_names,
                "rows": [
                    "feature_idxs",
                    "signal_idxs",
                    "shap_values",
                    "abs_shap_values",
                    "shap_values_length_ratio",
                    "abs_shap_values_length_ratio",
                    "counts",
                ],
            },
            attrs={"base_values": base_values, "x_pred": x_pred},
        )


class BorfFeatureSaliencyExplanation(BorfFeatureExplanation):
    def __init__(
        self,
        x,
        x_transformed,
        x_pred,
        s,
        alignments,
        shap_values,
        base_values,
        n_features,
        feature_names,
        n_labels,
        labels=None,
    ):
        super().__init__(
            x_transformed=x_transformed,
            x_pred=x_pred,
            shap_values=shap_values,
            base_values=base_values,
            n_features=n_features,
            feature_names=feature_names,
            n_labels=n_labels,
            labels=labels,
        )
        self.x = x
        self.n_signals = len(x[0])
        self.s = s
        self.alignments = alignments

        shap_min = np.concatenate(
            [
                np.array(ak.ravel(s)),
                self.shap_values_length_ratio[
                    :, self.x_transformed.ravel() > 0, :
                ].ravel(),
            ]
        ).min()
        shap_max = np.concatenate(
            [
                np.array(ak.ravel(s)),
                self.shap_values_length_ratio[
                    :, self.x_transformed.ravel() > 0, :
                ].ravel(),
            ]
        ).max()
        self.vmax = max(abs(shap_min), abs(shap_max))
        self.vmin = -self.vmax

    def _plot_single_saliency(self, x, s, label, norm, cmap="coolwarm", **kwargs):
        return plot_saliency(Y=x, S=s[..., label], cmap=cmap, norm=norm, **kwargs)

    def _plot_multi_saliency(
        self, x, s, n_labels, cmap="coolwarm", norm=TwoSlopeNorm, vcenter=0, **kwargs
    ):
        if vcenter is not None:
            norm = norm(vmin=self.vmin, vmax=self.vmax, vcenter=vcenter)
        else:
            norm = norm(vmin=self.vmin, vmax=self.vmax)
        return [
            self._plot_single_saliency(
                x=x, s=s, label=i, cmap=cmap, norm=norm, **kwargs
            )
            for i in range(n_labels)
        ]

    def plot_saliency(self, **kwargs):
        return self._plot_multi_saliency(
            x=self.x, s=self.s, n_labels=self.n_labels, **kwargs
        )


class BorfFeaturesShapeSaliencyExplanation(BorfFeatureSaliencyExplanation):
    def __init__(
        self,
        x,
        x_transformed,
        x_pred,
        s,
        alignments,
        shap_values,
        base_values,
        n_features,
        feature_names,
        n_labels,
        receptive_fields,
        labels=None,
    ):
        super().__init__(
            x=x,
            x_transformed=x_transformed,
            x_pred=x_pred,
            s=s,
            alignments=alignments,
            shap_values=shap_values,
            base_values=base_values,
            n_features=n_features,
            feature_names=feature_names,
            n_labels=n_labels,
            labels=labels,
        )
        self.receptive_fields = receptive_fields

    def plot_words(
        self,
        n_words=None,
        subplot_kwargs=None,
        not_contained_only=False,
        norm=TwoSlopeNorm,
        vcenter=0,
        **kwargs,
    ):
        if n_words is None:
            n_words = self.n_features
        if subplot_kwargs is None:
            subplot_kwargs = dict()
        if vcenter is not None:
            norm = norm(vcenter=vcenter, vmin=self.vmin, vmax=self.vmax)
        else:
            norm = norm(vmin=self.vmin, vmax=self.vmax)
        axs = []
        for label_idx in range(self.n_labels):
            word_importance = (
                self.word_importance[:, :, label_idx]
                .to_pandas()
                .T.sort_values(by="abs_shap_values", ascending=False)
            )
            if not_contained_only:
                word_importance = word_importance[word_importance.counts == 0]
            idxs = word_importance.feature_idxs[:n_words]
            shap_values = word_importance.shap_values[:n_words]
            shap_values_length_ratio = word_importance.shap_values_length_ratio[
                :n_words
            ]
            x_transformed = word_importance.counts[:n_words]
            for idx, shap_value, shap_value_length_ratio, value in zip(
                idxs, shap_values, shap_values_length_ratio, x_transformed
            ):
                fig, ax = plt.subplots(1, 1, **subplot_kwargs)
                s = self.receptive_fields[int(idx)]
                dilation = s.sax_params["dilation"]
                x_idxs = np.repeat(
                    np.arange(0, dilation * s.shape_.shape[1], dilation),
                    s.shape_.shape[0],
                ).reshape(-1, s.shape_.shape[0])

                ax.set_title(
                    f"word:{s.word}, signal:{s.signal_idx}, label: {label_idx} - {self.labels[label_idx]}\n, "
                    f"shap_value:"
                    f" {np.round(shap_value, 2)}, "
                    f"value: {value}"
                )
                ax.plot(x_idxs, s.shape_.T, c="gray", alpha=0.4, **kwargs)
                ax.plot(
                    x_idxs[:, 0],
                    s.centroid_.ravel(),
                    c=coolwarm(norm(shap_value_length_ratio)),
                    **kwargs,
                )
                axs.append(ax)
        return axs

    def plot(
        self,
        figsize=(8, 5),
        dpi=300,
        sharex=True,
        norm=TwoSlopeNorm,
        vcenter=0,
        subplot_kwargs=None,
        **kwargs,
    ):
        if vcenter is not None:
            norm = norm(vcenter=vcenter, vmin=self.vmin, vmax=self.vmax)
        else:
            norm = norm(vmin=self.vmin, vmax=self.vmax)
        if subplot_kwargs is None:
            subplot_kwargs = dict()
        fig, axs = plt.subplots(
            self.n_signals,  # n_signals
            2,
            squeeze=False,
            sharex=sharex,
            figsize=figsize,
            dpi=dpi,
            **subplot_kwargs,
        )

        s = self.s[..., self.x_pred] if self.n_labels > 1 else self.s[..., 0]
        plot_saliency(
            Y=self.x, S=s, cmap=coolwarm, norm=norm, axs=axs[:, 0:1], **kwargs
        )

        word_importance = (
            self.word_importance[..., self.x_pred]
            if self.n_labels > 1
            else self.word_importance[..., 0]
        )
        word_importance = word_importance.to_pandas().T.sort_values(
            by="abs_shap_values", ascending=False
        )
        word_importance = word_importance[word_importance.counts == 0]

        best_r = dict()
        for signal_idx in range(self.n_signals):
            best_r[signal_idx] = int(
                word_importance[word_importance.signal_idxs == signal_idx].feature_idxs[
                    0
                ]
            )

        for signal_idx, r_idx in best_r.items():
            r = self.receptive_fields[r_idx]
            dilation = r.sax_params["dilation"]
            x_idxs = np.arange(0, dilation * r.shape_.shape[1], dilation)
            axs[signal_idx, 1].plot(
                x_idxs,
                r.centroid_.ravel(),
                c=coolwarm(
                    norm(
                        word_importance[
                            word_importance.feature_idxs == r_idx
                        ].shap_values_length_ratio.values[0]
                    )
                ),
                **kwargs,
            )

        axs[0, 0].set_title(
            "Time Series - " + self.labels[self.x_pred]
        ) if self.n_labels > 1 else axs[0, 0].set_title("Time Series - " + r"$\hat{y}=$" + str(np.round(self.x_pred,
                                                                                                        4)))
        axs[0, 1].set_title("Not-contained Patterns")

        # Remove plot borders
        for ax in axs.ravel():
            ax.spines["top"].set_visible(False)  # Remove top border
            ax.spines["bottom"].set_visible(False)  # Remove bottom border
            ax.spines["left"].set_visible(False)  # Remove left border
            ax.spines["right"].set_visible(False)  # Remove right border
            ax.set_yticklabels([]) # Remove y-axis ticks

        for i in range(self.n_signals):
            if i == self.n_signals - 1:
                axs[i, 0].tick_params(
                    axis="y",
                    which="both",
                    length=0,
                    labelsize=0,
                    top=False,
                    bottom=False,
                    left=False,
                    right=False,
                )
                axs[i, 1].tick_params(
                    axis="both",
                    which="both",
                    length=0,
                    labelsize=0,
                    top=False,
                    bottom=False,
                    left=False,
                    right=False,
                )
            else:
                axs[i, 0].tick_params(
                    axis="both",
                    which="both",
                    length=0,
                    labelsize=0,
                    top=False,
                    bottom=False,
                    left=False,
                    right=False,
                )
                axs[i, 1].tick_params(
                    axis="both",
                    which="both",
                    length=0,
                    labelsize=0,
                    top=False,
                    bottom=False,
                    left=False,
                    right=False,
                )

        axs[self.n_signals - 1, 1].tick_params(labelbottom=False)

        # Create a ScalarMappable object using the 'coolwarm' colormap and the range of values
        sm = ScalarMappable(cmap=coolwarm, norm=norm)
        cbar = fig.colorbar(sm, ax=axs)
        cbar.set_label("Shap Value")

        return axs
