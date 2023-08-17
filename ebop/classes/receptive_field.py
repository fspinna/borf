import awkward as ak
import numpy as np
from ebop.algorithms.ebop_alignment import get_ts_alignment_idxs_from_word
from ebop.utils.transform_utils import zscore_transform
from ebop.utils.condition_utils import is_empty


class ReceptiveField:
    def __init__(
        self,
        word: str,
        sax_params: dict,
        tabular_idx: int = None,
        signal_separator: str = ";",
    ):
        self.word = word
        self.tabular_idx = tabular_idx
        self.signal_idx = int(word.split(signal_separator)[1])
        self.sax_params = sax_params
        self.shape_ = None
        self.centroid_ = None

    def __call__(self, *args, **kwargs):
        return self.word

    def get_multiple_alignment_indexes(self, X: ak.Array):
        alignments_list = list()
        for i in range(len(X)):
            _, alignments = get_ts_alignment_idxs_from_word(
                X=X[i : i + 1], sax_query=self.word, **self.sax_params
            )
            if len(alignments) == 0:
                alignments_list.append([])
            else:
                alignments_list.append(np.vstack(alignments))
        return ak.Array(alignments_list)

    def get_flat_alignment_indexes(self, X: ak.Array):
        return ak.Array(
            [
                np.sort(np.unique(ak.ravel(self.get_multiple_alignment_indexes(X)[i])))
                for i in range(len(X))
            ]
        )

    def get_general_shape(self, X: ak.Array, normalize=True, cache=True):
        alignments = self.get_multiple_alignment_indexes(X)
        shape = list()
        for i in range(len(alignments)):
            if not is_empty(np.asarray(alignments[i])):
                for j in range(len(alignments[i])):
                    if normalize:
                        shape.append(
                            zscore_transform(
                                np.asarray(
                                    X[i][self.signal_idx][np.asarray(alignments[i][j])]
                                )
                            )
                        )
                    else:
                        shape.append(
                            np.asarray(X[i][self.signal_idx][alignments[i][j]])
                        )
        self.shape_ = np.array(shape)
        self.centroid_ = np.mean(self.shape_, axis=0)
        return self.shape_

    def get_general_shape_flat(self, X: ak.Array, normalize=True):
        alignments = self.get_flat_alignment_indexes(X)
        shape = list()
        for i in range(len(alignments)):
            if not is_empty(np.asarray(alignments[i])):
                if normalize:
                    shape.append(
                        zscore_transform(
                            np.asarray(X[i][self.signal_idx][np.asarray(alignments[i])])
                        )
                    )
                else:
                    shape.append(np.asarray(X[i][self.signal_idx][alignments[i]]))
        return ak.Array(shape)
