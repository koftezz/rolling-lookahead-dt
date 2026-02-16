"""Shared test fixtures."""

import os
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def tiny_binary_dataset():
    """4 samples, 3 binary features, 2 classes. Perfectly separable by feature 1."""
    data = pd.DataFrame({
        "y": [1, 1, 2, 2],
        1: [1, 1, 0, 0],
        2: [0, 1, 0, 1],
        3: [1, 0, 1, 0],
    })
    return data


@pytest.fixture
def small_binary_dataset():
    """20 samples, 5 binary features, 2 classes."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 2, size=(20, 5))
    y = ((X[:, 0] == 1) & (X[:, 1] == 1)).astype(int) + 1
    data = pd.DataFrame(X, columns=[1, 2, 3, 4, 5])
    data.insert(0, "y", y)
    return data


@pytest.fixture
def multiclass_dataset():
    """15 samples, 4 binary features, 3 classes."""
    rng = np.random.RandomState(123)
    X = rng.randint(0, 2, size=(15, 4))
    y = np.array([1] * 5 + [2] * 5 + [3] * 5)
    data = pd.DataFrame(X, columns=[1, 2, 3, 4])
    data.insert(0, "y", y)
    return data


@pytest.fixture
def single_class_dataset():
    """All samples belong to one class."""
    data = pd.DataFrame({
        "y": [1, 1, 1, 1],
        1: [0, 1, 0, 1],
        2: [0, 0, 1, 1],
    })
    return data


@pytest.fixture
def wine_train_data():
    """Stratified sample of the bundled Wine training dataset."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "rollotree", "data")
    path = os.path.join(data_dir, "train.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Stratified sample so all classes are represented
        return df.groupby("y", group_keys=False).apply(
            lambda x: x.head(20)
        ).reset_index(drop=True)
    pytest.skip("Wine train.csv not found")


@pytest.fixture
def wine_test_data():
    """First 30 rows of the bundled Wine test dataset."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "rollotree", "data")
    path = os.path.join(data_dir, "test.csv")
    if os.path.exists(path):
        return pd.read_csv(path).head(30)
    pytest.skip("Wine test.csv not found")
