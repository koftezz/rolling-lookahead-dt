"""
Tree Related Functions
"""
import copy
import logging
import pandas as pd
import numpy as np
from typing import Union, List


def get_child(current_depth: int, target_depth: int, child_node: int) -> int:
    """

    :param current_depth:
    :param target_depth:
    :param child_node:
    :return:
    """
    counter = 0
    if target_depth > current_depth:
        while counter < (target_depth - current_depth):
            if child_node % 2 == 0:
                child_node = child_node * 2
            else:
                child_node = (child_node * 2) + 1
            counter += 1
        return child_node


def generate_nodes(depth: int) -> list:
    """

    :param depth:
    :return:
    """
    nodes = list(range(1, int(round(2 ** (depth + 1)))))
    parent_nodes = nodes[0: 2 ** (depth + 1) - 2 ** depth - 1]
    leaf_nodes = nodes[-2 ** depth:]
    return parent_nodes, leaf_nodes


def leaf_pattern(sub_leaf: int, depth: int, leaf: int) -> int:
    """

    :param sub_leaf: leaf node indexes for sub tree
    :param depth: depth of expansion tree
    :param leaf: leaf node index for main tree to branch on
    for sub tree
    :return: new index for expanded tree
    """
    return (sub_leaf - 2 ** depth) + 2 ** depth * leaf


def parent_pattern(sub_leaf: int, leaf_node: int) -> int:
    """
    Extends parent indexes for main tree
    :param sub_leaf:
    :param leaf_node:
    :return:
    """
    if sub_leaf == 1:
        return leaf_node
    elif sub_leaf < 4:
        return (2 * leaf_node) + (sub_leaf % 2)
    elif sub_leaf < 8:
        return (4 * leaf_node) + (sub_leaf % 4)
    else:
        logging.error("Error in parent pattern!")


def list_nodes_to_branch_on(model_dict: dict, K: list) -> list:
    """
    Returns only leaf nodes with misclassification in it, which is max of
    Nkt is not equal to Nt
    :param K:
    :param model_dict:
    :return:
    """
    return [i for i in
            model_dict["params"]["var_Nt"].keys() if
            model_dict["params"]["var_Nt"][
                i] > 0 and max(
                model_dict["params"]["var_Nkt"][
                    (i, k)]
                for k in K) !=
            model_dict["params"]["var_Nt"][i]]


def parents_of_nodes_to_branch_on(go_deep_nodes: [list, dict]) -> dict:
    """

    :param go_deep_nodes:
    :return:
    """
    parents_to_optimize = dict()
    for node_ in go_deep_nodes:
        if int(node_ / 2) not in parents_to_optimize.keys():
            parents_to_optimize[int(node_ / 2)] = [node_]
        else:
            parents_to_optimize[int(node_ / 2)].append(node_)
    return parents_to_optimize


def get_data_from_nodes(cols: list,
                        nodes: List[Union[dict, list]],
                        var_z_dict: dict,
                        input_data: pd.DataFrame,
                        parent: bool = True) -> dict:
    """

    :param cols:
    :param nodes:
    :param var_z_dict:
    :param input_data:
    :param parent:
    :return:
    """
    if parent:
        train_data_dict = {}
        for p in nodes:
            df = pd.DataFrame([], columns=cols)
            for i in var_z_dict["var_z"].keys():
                # search for both of leaf nodes since
                # it is parents optimizing
                if i[1] in [p * 2, p * 2 + 1]:
                    if var_z_dict["var_z"][i] == 1:
                        df = pd.concat(
                            [df,
                             pd.DataFrame(
                                 [input_data.iloc[i[
                                     0]]])],
                            axis=0)
            train_data_dict[p] = df.reset_index(

            ).drop('index', axis=1)
            del df
    else:
        train_data_dict = {}
        for p in nodes:
            df = pd.DataFrame([], columns=cols)
            for i in var_z_dict["var_z"].keys():
                # search for both of leaf nodes since
                # it is parents optimizing
                if i[1] == p:
                    if var_z_dict["var_z"][i] == 1:
                        df = pd.concat(
                            [df,
                             pd.DataFrame(
                                 [input_data.iloc[i[
                                     0]]])],
                            axis=0)
            train_data_dict[p] = df.reset_index(

            ).drop('index', axis=1)
            del df
    return train_data_dict


def union_model(depth: int,
                main_model: dict,
                union_nodes: List[Union[dict, list]],
                result_dict: dict,
                K: list,
                P: list
                ) -> dict:
    """

    :param depth:
    :param main_model:
    :param union_nodes:
    :param result_dict:
    :param K:
    :param P:
    :return:
    """
    union_model = dict()
    parent_nodes, leaf_nodes = generate_nodes(
        depth=depth)
    nodes = {
        'leaf_nodes': leaf_nodes,
        'parent_nodes': parent_nodes
    }
    union_model["nodes"] = nodes
    union_model["params"] = copy.deepcopy(
        main_model["params"])
    union_model["params"]["var_c"] = {}
    union_model["params"]["var_Nkt"] = {}
    union_model["params"]["var_Lt"] = {}
    union_model["params"]["var_Nt"] = {}
    union_model["params"]["var_z"] = {}
    for node_ in union_nodes:
        union_model["params"]["var_d"].update(
            result_dict[node_]["model"]["params"][
                "var_d"])
        union_model["params"]["var_a"].update(
            result_dict[node_]["model"]["params"][
                "var_a"])
        union_model["params"]["var_c"].update(
            result_dict[node_]["model"]["params"][
                "var_c"])
        union_model["params"]["var_Nkt"].update(
            result_dict[node_]["model"]["params"][
                "var_Nkt"])
        union_model["params"]["var_Lt"].update(
            result_dict[node_]["model"]["params"][
                "var_Lt"])
        union_model["params"]["var_Nt"].update(
            result_dict[node_]["model"]["params"][
                "var_Nt"])
        union_model["params"]["var_z"].update(
            result_dict[node_]["model"]["params"][
                "var_z"])

    for t in leaf_nodes:
        for k in K:
            if (t, k) not in \
                    union_model["params"][
                        "var_c"].keys():
                union_model["params"]["var_c"][
                    t, k] = 0
            if (t, k) not in \
                    union_model["params"][
                        "var_Nkt"].keys():
                union_model["params"]["var_Nkt"][
                    t, k] = 0
        if t not in \
                union_model["params"]["var_Nt"].keys():
            union_model["params"]["var_Nt"][t] = 0
    for t in parent_nodes:
        for j in P:
            if (t, j) not in \
                    union_model["params"][
                        "var_a"].keys():
                union_model["params"]["var_a"][
                    t, j] = 0
    for t in parent_nodes:
        if t not in union_model["params"][
            "var_d"].keys():
            union_model["params"]["var_d"][t] = 0
    return union_model


def gini_index(arr: np.array,
               instance_size: int,
               K: list,
               y_idx: int = 0,
               weighted: bool = True):
    """

    :param arr:
    :param instance_size:
    :param K:
    :param y_idx:
    :param weighted:
    :return:
    """
    sum_ = 0
    for k in K:
        sum_ += np.power(len(arr[np.where(arr[:, y_idx] == k)]) / len(arr), 2)
    sum_ = 1 - sum_
    if weighted:
        sum_ = (len(arr) / instance_size) * sum_
    return sum_


def calculate_gini(data: pd.DataFrame,
                   P: list,
                   K: list,
                   nodes: dict) -> dict:
    """

    :param data:
    :param P:
    :param K:
    :param nodes:
    :return:
    """
    df_arr = np.array(data)
    n = len(data)
    gini_dict = dict()
    for leaf_ in nodes["leaf_nodes"]:
        temp = dict()
        first_var = nodes["leaf_nodes_path"][leaf_][0]
        second_var = nodes["leaf_nodes_path"][leaf_][1]
        for feature_i in P:
            arr = df_arr[np.where((df_arr[:, feature_i] == first_var))]

            for feature_j in P:
                arr_2 = arr[np.where(arr[:, feature_j] == second_var)]
                if len(arr_2) > 0:
                    temp[feature_i, feature_j] = gini_index(arr=arr_2,
                                                            instance_size=n,
                                                            K=K,
                                                            weighted=True)
        gini_dict[leaf_] = copy.deepcopy(temp)
        del temp
        del arr

    return gini_dict


def find_misclassification(df: pd.DataFrame) -> list:
    """

    :param df:
    :return:
    """
    return list(df.loc[
                    df.groupby("leaf")["y"].transform(lambda x: (x.nunique() >
                                                                 1)),
                    'leaf'].unique())


def error_index(arr):
    """

    :param arr:
    :return:
    """
    values, counts = np.unique(arr[:, 0],
                               return_counts=True)
    assigned_class = values[np.argmax(counts)]
    return arr[np.where(arr[:, 0] != assigned_class)].shape[0]


def calculate_misclassification(data: pd.DataFrame,
                                P: list,
                                nodes: dict) -> dict:
    """

    :param data:
    :param P:
    :param nodes:
    :return:
    """
    df_arr = np.array(data)
    misclass_dict = dict()
    for leaf_ in nodes["leaf_nodes"]:
        temp = dict()
        first_var = nodes["leaf_nodes_path"][leaf_][0]
        second_var = nodes["leaf_nodes_path"][leaf_][1]
        for feature_i in P:
            arr = df_arr[np.where((df_arr[:, feature_i] == first_var))]

            for feature_j in P:
                arr_2 = arr[np.where(arr[:, feature_j] == second_var)]
                if len(arr_2) > 0:
                    temp[feature_i, feature_j] = error_index(arr=arr_2)

        misclass_dict[leaf_] = copy.deepcopy(temp)
        del temp
        del arr
        del arr_2
    return misclass_dict
