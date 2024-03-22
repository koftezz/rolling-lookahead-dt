from rollo_oct.model.rolling_optimize import _rolling_optimize
from rollo_oct.oct.tree import *
from rollo_oct.oct.optimal_tree import *
from rollo_oct.utils.helpers import preprocess_dataframes

import pandas as pd
import time


def run(train: pd.DataFrame,
        test: pd.DataFrame,
        target_label: str = "y",
        features: List = None,
        depth: int = 2,
        criterion: str = "gini",
        time_limit: int = 1800,
        big_m: int = 99):
    """

    :param target_label:
    :param features:
    :param train:
    :param test:
    :param depth:
    :param criterion:
    :param time_limit:
    :param big_m:
    :return:
    """
    # Example Usage:
    # Assuming df is your pandas DataFrame
    train, test = preprocess_dataframes(
        train_df=train,
        test_df=test,
        target_label=target_label,
        features=features)

    df = pd.concat([train, test])
    P = [int(i) for i in
         list(train.loc[:, train.columns != 'y'].columns)]
    train.columns = ["y", *P]
    test.columns = ["y", *P]
    K = sorted(list(set(df.y)))
    result_dict = {}
    main_model_time = time.time()
    # generate model
    main_model = generate_model(P=P, K=K, data=train, y_idx=0, big_m=big_m)
    # train model
    main_model = train_model(model_dict=main_model, data=train, P=P)
    # predict model
    result = predict_model(data=train, model_dict=main_model, P=P)

    misclassified_leafs = find_misclassification(df=result)
    result_ = predict_model(data=test, model_dict=main_model, P=P)
    train_acc = len(result.loc[result["prediction"] == result["y"]]) / \
                len(result["y"])

    test_acc = len(result_.loc[result_["prediction"] == result_["y"]]) / \
               len(result_["y"])
    del result_

    train = train.drop(["prediction", "leaf"], axis=1)
    test = test.drop(["prediction", "leaf"], axis=1)
    if depth > 2:
        result_dict = _rolling_optimize(predefined_model=main_model,
                                        train_data=train,
                                        test_data=test,
                                        main_depth=2,
                                        target_depth=depth,
                                        features=P,
                                        time_limit=time_limit,
                                        to_go_deep_nodes=misclassified_leafs,
                                        criterion=criterion)

    # add main model
    result_dict[2] = {
        "training_accuracy": train_acc,
        "test_accuracy": test_acc,
        "time": time.time() - main_model_time
    }
    return result_dict
