"""Tests for data preprocessing utilities."""

import pandas as pd
import pytest

from rollo_oct.preprocessing.helpers import preprocess_dataframes, make_data_binary


class TestPreprocessDataframes:
    def test_moves_target_first(self):
        train = pd.DataFrame({"a": [1, 2], "y": [0, 1], "b": [3, 4]})
        test = pd.DataFrame({"a": [5, 6], "y": [1, 0], "b": [7, 8]})

        result_train, result_test = preprocess_dataframes(
            train, test, target_label="y", features=["a", "b"]
        )

        assert result_train.columns[0] == "y"
        assert result_test.columns[0] == "y"

    def test_renames_features(self):
        train = pd.DataFrame({"feat_a": [1, 0], "feat_b": [0, 1], "y": [1, 2]})
        test = pd.DataFrame({"feat_a": [0, 1], "feat_b": [1, 0], "y": [2, 1]})

        result_train, _ = preprocess_dataframes(
            train, test, target_label="y", features=["feat_a", "feat_b"]
        )

        assert "1" in result_train.columns
        assert "2" in result_train.columns
        assert "feat_a" not in result_train.columns

    def test_preserves_data_values(self):
        train = pd.DataFrame({"x": [10, 20], "y": [1, 2]})
        test = pd.DataFrame({"x": [30, 40], "y": [2, 1]})

        result_train, result_test = preprocess_dataframes(
            train, test, target_label="y", features=["x"]
        )

        assert list(result_train["y"]) == [1, 2]
        assert list(result_test["y"]) == [2, 1]


class TestMakeDataBinary:
    def test_binary_columns_preserved(self):
        data = pd.DataFrame({"y": [1, 2, 1, 2], "a": [0, 1, 0, 1]})
        result = make_data_binary(data)
        assert result.shape[1] >= 2  # at least y + 1 feature

    def test_categorical_one_hot_encoded(self):
        data = pd.DataFrame({
            "y": [1, 2, 1],
            "color": ["red", "green", "blue"],
        })
        result = make_data_binary(data)
        # 'color' should be one-hot encoded into 3 columns
        assert result.shape[1] >= 4  # y + 3 binary features

    def test_y_is_first_column(self):
        data = pd.DataFrame({"a": [0, 1], "y": [1, 2], "b": [1, 0]})
        result = make_data_binary(data)
        assert result.columns[0] == "y"

    def test_missing_values_filled(self):
        data = pd.DataFrame({
            "y": [1, 2, 1],
            "a": [0, None, 1],
        })
        result = make_data_binary(data)
        assert not result.isnull().any().any()
