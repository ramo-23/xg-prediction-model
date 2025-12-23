import pytest
from src import feature_engineering


def test_add_basic_features_empty():
    df = feature_engineering.add_basic_features(None)
    assert df.empty
