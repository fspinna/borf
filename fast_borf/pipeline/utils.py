def bidirectional_index_map_from_indices(indices):
    out_to_in = {i: column for i, column in enumerate(indices)}
    in_to_out = {column: i for i, column in enumerate(indices)}
    return in_to_out, out_to_in


def apply_map(features, feature_map):
    return [feature_map[feature] for feature in features]
