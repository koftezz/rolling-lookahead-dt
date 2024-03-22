import pandas as pd
import numpy as np
from gurobipy import *

from rollo_oct.oct.tree import generate_nodes, calculate_gini, \
    calculate_misclassification


def generate_model(
        P: list,
        K: list,
        data: pd.DataFrame,
        y_idx: int = 0,
        time_limit: float = 1800,
        gap_limit: float = None,
        log_to_console: bool = False,
        big_m: int = 99,
        criterion: str = "gini",
):
    """

    :param criterion:
    :param big_m:
    :param depth:
    :param P:
    :param K:
    :param data:
    :param leaf_nodes_path:
    :param y_idx:
    :param time_limit:
    :param gap_limit:
    :param log_to_console:
    :return:
    """

    # Create parent & leaf nodes

    leaf_nodes_path = {4: [1, 1],
                       5: [1, 0],
                       6: [0, 1],
                       7: [0, 0]}
    depth = 2
    parent_nodes, leaf_nodes = generate_nodes(depth)
    nodes = dict()
    nodes["leaf_nodes"] = leaf_nodes
    nodes["leaf_nodes_path"] = leaf_nodes_path
    if criterion == "gini":
        logging.info("Calculating gini..")
        coef_dict = calculate_gini(data=data,
                                   P=P,
                                   K=K,
                                   nodes=nodes)
    elif criterion == "misclassification":
        logging.info("Calculating misclassification..")
        coef_dict = calculate_misclassification(data=data,
                                                P=P,
                                                nodes=nodes)
    # init model
    m = Model("RollOCT")

    # Define Variables
    x = dict()
    for i in P:
        for j in P:
            x[i, j] = m.addVar(vtype=GRB.BINARY,
                               name=f'x[{i}, {j}]')

    y = dict()
    for i in P:
        for k in P:
            y[i, k] = m.addVar(vtype=GRB.BINARY,
                               name=f'y[{i}, {k}]')
    # Constraint 1
    m.addConstr(quicksum(x[i, j] for j in P for i in P) == 1)
    # Constraint 2
    m.addConstr(quicksum(y[i, k] for k in P for i in P) == 1)
    # Constraint 3
    for i in P:
        m.addConstr(
            quicksum(x[i, j] for j in P) == quicksum(y[i, k] for k in P))

    # Constraint 4
    # add big m

    obj = quicksum(
        (coef_dict[4].get((i, j), big_m) + coef_dict[5].get((i, j), big_m)) *
        x[i, j]
        for i in P
        for j in P) + \
          quicksum((coef_dict[6].get((i, k), big_m) + coef_dict[7].get((i,
                                                                        k),
                                                                       big_m)) *
                   y[i, k]
                   for i in P
                   for k in P)

    m.setObjective(obj, GRB.MINIMIZE)

    if time_limit:
        m.setParam("TimeLimit", time_limit)
        logging.info(f'Setting Time Limit as {time_limit}')
    if gap_limit:
        m.setParam("MipGap", gap_limit)
        logging.info(f'Setting Optimality Gap as {gap_limit}')
    m.setParam("LogToConsole", int(log_to_console))
    logging.info(f'Setting LogToConsole as {log_to_console}')
    m.update()

    model_dict = {
        'model': m,
        'params': {
            'var_x': x,
            'var_y': y,
            'y_idx': y_idx
        },
        'nodes': {
            'leaf_nodes': leaf_nodes,
            'parent_nodes': parent_nodes,
            "leaf_nodes_path": leaf_nodes_path,
        },
        'depth': depth,
        "P": P,
        "K": K
    }
    logging.info('Model generation is done.')

    return model_dict


def train_model(data: pd.DataFrame,
                model_dict: dict,
                P: list) -> dict:
    """

    :param data:
    :param model_dict:
    :param P:
    :return:
    """
    logging.info("Training..")
    nodes = model_dict['nodes']
    params = model_dict['params']
    m = model_dict['model']
    m.optimize()

    if m.status == GRB.Status.OPTIMAL or \
            (m.status == 9 and m.objVal != -math.inf):
        logging.info(f'Optimal objective: {m.objVal}')
        status = m.status
    elif m.status == GRB.Status.INFEASIBLE:
        logging.info("Model is infeasible")
        return
    elif m.status == GRB.Status.UNBOUNDED:
        logging.info("Model is unbounded")
        return
    elif m.status == 9 and m.objVal != -math.inf:
        logging.info('Time Limit termination')
        return
    else:
        logging.info(f'Optimization status {m.status}')
        return

    logging.info(f"Objective Value: {m.objVal}")
    # Arrange decision variables
    x = {
        (i, j): params['var_x'][i, j].X
        for i in P for j in P
    }
    y = {
        (i, j): params['var_y'][i, j].X
        for i in P for j in P
    }

    for i in P:
        for j in P:
            if x[i, j] > 0:
                first_level = i
                left_second_level = j
                logging.info(
                    f"First Level Feature: {i} & Second Level Left Feature: "
                    f"{j}, {x[i, j]}={x[i, j]}")
            if y[i, j] > 0:
                right_second_level = j
                logging.info(
                    f"First Level Feature: {i} & Second Level Right Feature:"
                    f" {j}, {y[i, j]}={y[i, j]}")

    logging.info("Extracting solution..")
    df_arr = np.array(data)
    target_class = dict()
    for leaf_ in nodes["leaf_nodes"]:
        first_var = nodes["leaf_nodes_path"][leaf_][0]
        second_var = nodes["leaf_nodes_path"][leaf_][1]
        arr = df_arr[np.where((df_arr[:, first_level] == first_var))]
        if leaf_ in [4, 5]:  # left
            arr_2 = arr[np.where(arr[:, left_second_level] == second_var)]
        elif leaf_ in [6, 7]:  # right
            arr_2 = arr[np.where(arr[:, right_second_level] == second_var)]
        else:
            pass
        values, counts = np.unique(arr_2[:, params["y_idx"]],
                                   return_counts=True)
        if len(counts) > 0:
            target_class[leaf_] = values[np.argmax(counts)]

    var_a = {1: [0 if i != first_level else 1 for i in P],
             2: [0 if i != left_second_level else 1 for i in P],
             3: [0 if i != right_second_level else 1 for i in P]
             }
    del df_arr
    logging.info(f'Training done. Loss: {m.objVal}\n'
                 f'Optimization status: {status}\n')
    # model statistics
    details = {
        'run_time': m.Runtime,
        'mip_gap': m.MIPGap,
        'objective': m.objVal,
        'status': status,
        'target_class': target_class,
        "var_a": var_a,
        "selected_features": {1: first_level,
                              2: left_second_level,
                              3: right_second_level
                              }
    }
    model_dict["details"] = details
    logging.info("Training is done.")
    return model_dict


def predict_model(data: pd.DataFrame,
                  P: list,
                  model_dict: dict,
                  pruned_nodes: list = []) -> pd.DataFrame:
    """

    :param model_dict:
    :param pruned_nodes:
    :param data:
    :param P:
    :return:
    """

    prediction = []
    leaf_ = []
    depth = model_dict["depth"]
    var_a = model_dict["details"]["var_a"]
    target_class = model_dict["details"]["target_class"]
    for idx, i in data.iterrows():
        x = np.array(i[P])
        t = 1
        d = 0
        while d < depth:
            at = np.array(var_a[t])
            if at.dot(x) == 1:
                t = t * 2
            else:
                t = t * 2 + 1
            d = d + 1
            if t in pruned_nodes:
                break
        prediction.append(target_class[t])
        leaf_.append(t)
    data["prediction"] = prediction
    data["leaf"] = leaf_
    logging.info("Prediction is done.")
    return data
