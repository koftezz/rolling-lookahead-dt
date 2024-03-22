import numpy as np
import pandas as pd
import logging


def train_validation_test_split(data: pd.DataFrame,
                                train_percent: float = .5,
                                validation_percent: float = .25,
                                seed: float = None) -> pd.DataFrame:
    np.random.seed(seed)
    perm = np.random.permutation(data.index)
    m = len(data.index)
    train_end = int(train_percent * m)
    validate_end = int(validation_percent * m) + train_end
    train_df = data.iloc[perm[:train_end]]
    validation_df = data.iloc[perm[train_end:validate_end]]
    test_df = data.iloc[perm[validate_end:]]
    return train_df, validation_df, test_df


def make_data_binary(data: pd.DataFrame) -> pd.DataFrame:
    """

    :param data: input data
    :return: data with binary columns
    """
    cols_with_missing = [col for col in data.columns
                         if data[col].isnull().any()]
    if cols_with_missing:
        for col in cols_with_missing:
            data[col].fillna(data[col].mode()[0], inplace=True)
        logging.info("""There are columns with missing
            values.\nColumns are: {0}\n Replacing with mode. 
            """.format(cols_with_missing))
    binary_cols = [cname for cname in data.columns if
                   data[cname].nunique() == 2 and cname != 'y']

    for col in binary_cols:
        if data[col].dtype not in ['int8', 'int16']:
            logging.info(f"Column {col} is not int type. Transforming it into "
                         f"integer.")
            data[col] = data[col].astype('category').cat.codes

    if binary_cols:
        for col in binary_cols:
            # if sum of unique entries is not equal to 1
            if sum(data[col].unique()) != 1:
                replace = {data[col].unique()[0]: 0,
                           data[col].unique()[1]: 1}
                data[col] = [replace[item] for item in data[col]]
        logging.info("There are {0} binary columns. \nColumns are: {1}".format(
            len(binary_cols), binary_cols))
    else:
        logging.info("No binary columns.")

    total_col = 0
    for col in data.columns:
        if col != "y":
            logging.info("Column: {0} - Unique Values: {1}".format(
                col, data[col].nunique()))
            if data[col].nunique() != 2:
                total_col += data[col].nunique()
    total_col += len(binary_cols) + 1

    for col in data.columns:
        if col not in binary_cols and col != "y":
            dummy_col = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummy_col], axis=1)
            data = data.drop(col, axis=1)

    if total_col != data.shape[1]:
        logging.error("# of expected column is not equal to actual.")
        return

    if data.y.dtype == "O":
        logging.info("Converting y values into numerical.")
        data['y'] = data['y'].astype('category')
        data['y'] = data['y'].cat.codes

    if data.y.min() < 1:
        data['y'] = data['y'] + 1

    if data.columns[-1] != "y":
        logging.info("Reordering y column at the beginning of data.")
        cols_ = list(data.columns)
        cols_.remove('y')
        cols_.insert(0, "y")
        data = data[cols_]

    logging.info("Renaming columns..")
    column_indices = [i for i in range(1, len(data.columns))]
    new_names = column_indices
    old_names = data.columns[column_indices]
    data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    logging.info("Data is binarized.")
    return data


def preprocess_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame, target_label: str, features: list):
    """
    Rearranges the DataFrames such that the target label becomes the first column,
    and feature names are converted into ordinal numbers.

    Args:
    - train_df: pandas DataFrame containing the training data
    - test_df: pandas DataFrame containing the test data
    - target_label: string representing the target label
    - features: list of strings representing feature names

    Returns:
    - pd.DataFrame: preprocessed training DataFrame
    - pd.DataFrame: preprocessed test DataFrame
    """

    # Move target label to the first column for both train and test DataFrames
    if target_label in train_df.columns:
        train_target_idx = train_df.columns.get_loc(target_label)
        train_df_columns = list(train_df.columns)
        train_df_columns = [train_df_columns[train_target_idx]] + train_df_columns[:train_target_idx] + train_df_columns[train_target_idx + 1:]
        train_df = train_df[train_df_columns]

    if target_label in test_df.columns:
        test_target_idx = test_df.columns.get_loc(target_label)
        test_df_columns = list(test_df.columns)
        test_df_columns = [test_df_columns[test_target_idx]] + test_df_columns[:test_target_idx] + test_df_columns[test_target_idx + 1:]
        test_df = test_df[test_df_columns]

    # Rename features to ordinal numbers for both train and test DataFrames
    train_df.rename(columns={feature: str(i) for i, feature in enumerate(features, start=1)}, inplace=True)
    test_df.rename(columns={feature: str(i) for i, feature in enumerate(features, start=1)}, inplace=True)

    return train_df, test_df
