"""Data preprocessing utilities."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_label: str,
    features: list,
) -> tuple:
    """
    Rearrange DataFrames so the target label is the first column,
    and feature names are converted to ordinal numbers.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        target_label: Name of the target column.
        features: List of feature column names.

    Returns:
        (processed_train_df, processed_test_df)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Move target label to first column
    for df in [train_df, test_df]:
        if target_label in df.columns:
            cols = [target_label] + [c for c in df.columns if c != target_label]
            df_reordered = df[cols]
            if df is train_df:
                train_df = df_reordered
            else:
                test_df = df_reordered

    # Rename features to ordinal numbers
    rename_map = {feat: str(i) for i, feat in enumerate(features, start=1)}
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    return train_df, test_df


def make_data_binary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical features to binary via one-hot encoding.

    Handles missing values with mode imputation. Ensures y values
    are numerical and placed as the first column.

    Args:
        data: Input DataFrame with 'y' as target column.

    Returns:
        Binarized DataFrame.
    """
    data = data.copy()

    # Fill missing values with mode
    cols_with_missing = [col for col in data.columns if data[col].isnull().any()]
    if cols_with_missing:
        for col in cols_with_missing:
            data[col] = data[col].fillna(data[col].mode()[0])
        logger.info(f"Filled missing values in columns: {cols_with_missing}")

    # Identify binary columns
    binary_cols = [
        c for c in data.columns if data[c].nunique() == 2 and c != "y"
    ]

    for col in binary_cols:
        if data[col].dtype not in ["int8", "int16", "int32", "int64"]:
            data[col] = data[col].astype("category").cat.codes

    if binary_cols:
        for col in binary_cols:
            if sum(data[col].unique()) != 1:
                uniques = data[col].unique()
                replace_map = {uniques[0]: 0, uniques[1]: 1}
                data[col] = data[col].map(replace_map)

    # One-hot encode non-binary, non-target columns
    for col in data.columns:
        if col not in binary_cols and col != "y":
            dummy_col = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummy_col], axis=1)
            data = data.drop(col, axis=1)

    # Ensure y is numeric
    if data["y"].dtype == "O":
        data["y"] = data["y"].astype("category").cat.codes

    if data["y"].min() < 1:
        data["y"] = data["y"] + 1

    # Ensure y is first column
    if data.columns[0] != "y":
        cols = ["y"] + [c for c in data.columns if c != "y"]
        data = data[cols]

    # Rename feature columns to ordinal numbers
    feature_cols = [c for c in data.columns if c != "y"]
    rename_map = {old: i for i, old in enumerate(feature_cols, start=1)}
    data = data.rename(columns=rename_map)

    return data
