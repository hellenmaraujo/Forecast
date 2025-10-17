from __future__ import annotations


def test_feature_shapes(prepared_features, config):
    train_features, test_features, metadata = prepared_features
    assert not train_features.empty
    assert not test_features.empty
    target = metadata["target_column"]
    assert target in train_features.columns
    assert target in test_features.columns
    # Ensure chronological order preserved
    timestamp = metadata["timestamp_column"]
    assert train_features[timestamp].is_monotonic_increasing
