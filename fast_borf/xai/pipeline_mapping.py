import numpy as np
from sklearn.pipeline import FeatureUnion


# class TransformerPipelineFeatureMapper:
#     def __init__(self, borf: FeatureUnion, reshaper_position=1, zero_columns_remover_position=2):
#         self.borf = borf
#         self.reshaper_position = reshaper_position
#         self.zero_columns_remover_position = zero_columns_remover_position
#
#         self.mapping = map_features_to_conf(
#             transformer_list=self.borf.transformer_list,
#             reshaper_position=self.reshaper_position,
#             zero_columns_remover_position=self.zero_columns_remover_position
#         )
#
#     def map_features_to_conf(self):
#         return


def map_borf_to_conf(borf: FeatureUnion, reshaper_position=1, zero_columns_remover_position=2):
    return map_features_to_conf(
        transformer_list=borf.transformer_list,
        reshaper_position=reshaper_position,
        zero_columns_remover_position=zero_columns_remover_position
    )


def map_features_to_conf(transformer_list, reshaper_position=1, zero_columns_remover_position=2):
    # assuming something like this:
    # transformer_list =
    # [...
    #  ('i',
    #   Pipeline(steps=[('borfsaxsingletransformer', BorfSaxSingleTransformer()),  0
    #                   ('reshapeto2d', ReshapeTo2D(keep_unraveled_index=True)),  1
    #                   ('zerocolumnsremover', ZeroColumnsRemover()),  2
    #                   ('toscipysparse', ToScipySparse())])),  3
    #  ('i+1', ...),
    #  ...]
    mapping = np.empty((0, 3), dtype=np.int_)
    for i, transformer in enumerate(transformer_list):
        conf_index = i
        mapping_ = map_single_conf_features_to_words(
            transformer[1][reshaper_position].unraveled_index_,
            transformer[1][zero_columns_remover_position].columns_to_keep_
        )
        mapping_ = np.hstack([np.full((len(mapping_), 1), conf_index), mapping_])
        #  (conf_index, signal_idx, word_idx)
        mapping = np.vstack([mapping, mapping_])
    return mapping


def map_single_conf_features_to_words(unraveled_index, columns_kept):
    return unraveled_index[columns_kept]