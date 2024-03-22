from typing import Optional
import time

from rollo_oct.oct.optimal_tree import predict_model, generate_model, \
    train_model
from rollo_oct.oct.tree import *


def _rolling_optimize(predefined_model: Optional[dict],
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      main_depth: int,
                      target_depth: int,
                      features: list,
                      time_limit: float,
                      to_go_deep_nodes: list,
                      criterion: str = "gini"
                      ) -> pd.DataFrame:
    """

    :param predefined_model:
    :param train_data:
    :param test_data:
    :param main_depth:
    :param target_depth:
    :param features:
    :param time_limit:
    :return:
    """

    model_name = f"rolling_optimizing_gini_{main_depth}"
    logging.info(f"Running {model_name} with main depth {main_depth} by "
                 f"diving in "
                 f"2 level with target depth {target_depth}")

    df_arr = np.array(train_data)
    leaf_nodes_path = predefined_model["nodes"]["leaf_nodes_path"]
    y_idx = 0
    predefined_model.pop('model')
    predefined_model.pop('params')
    final_model = copy.deepcopy(predefined_model)

    selected_features_temp = copy.deepcopy(predefined_model["details"][
                                               "selected_features"])
    selected_features = {1: selected_features_temp}
    to_go_deep_nodes = {i: i for i in to_go_deep_nodes}
    pruned_nodes = [i for i in \
                    leaf_nodes_path if i not in to_go_deep_nodes]
    new_train_data_dict = {}
    result_dict = {}
    for level in range(1, target_depth - main_depth + 1):
        logging.info(f"Extending tree from Depth {main_depth} to "
                     f"{main_depth + level}")
        iter_time = time.time()
        to_go_deep_nodes_ = {}
        sub_K = {}
        parents_to_optimize = parents_of_nodes_to_branch_on(
            go_deep_nodes=to_go_deep_nodes)
        for node_ in parents_to_optimize:

            leafs_ = parents_to_optimize[node_]
            sub_features = selected_features[
                int(node_ / 2)]

            if level == 1:
                first_var = leaf_nodes_path[leafs_[0]][0]
                arr = df_arr[np.where((df_arr[:, sub_features[1]] ==
                                       first_var))]

                sub_K[node_] = list(np.unique(arr[:, y_idx]))
                cols = features.copy()
                cols.insert(y_idx, 'y')
                new_train_data_dict[node_] = pd.DataFrame(
                    arr, columns=cols,
                    index=None)
            else:
                arr = new_train_data_dict[int(node_ / 2)]
                arr = np.array(arr)
                first_var = leaf_nodes_path[to_go_deep_nodes[leafs_[0]]][0]
                arr = arr[np.where((arr[:, sub_features[1]] ==
                                    first_var))]
                sub_K[node_] = list(np.unique(arr[:, y_idx]))
                cols = features.copy()
                cols.insert(y_idx, 'y')
                new_train_data_dict[node_] = pd.DataFrame(
                    arr, columns=cols,
                    index=None)

            logging.info(
                f"Processing for Parent Node: {node_} of Leaf "
                f"Node(s): {leafs_} at Level: {level}")
            main_model = generate_model(P=features,
                                        K=sub_K[node_],
                                        data=new_train_data_dict[node_],
                                        y_idx=y_idx,
                                        time_limit=time_limit,
                                        log_to_console=False,
                                        criterion=criterion)
            main_model = train_model(model_dict=main_model,
                                     data=new_train_data_dict[node_],
                                     P=features)
            result_ = predict_model(
                data=new_train_data_dict[node_],
                model_dict=main_model,
                P=features)
            misclassified_leafs = find_misclassification(df=result_)
            del result_
            temp_dict = {}
            for t in predefined_model["nodes"]["parent_nodes"]:
                new_idx = parent_pattern(sub_leaf=t,
                                         leaf_node=node_)
                temp_dict[new_idx] = main_model[
                    "details"]["var_a"][t]
            main_model["details"]["var_a"] = temp_dict
            # update extended tree's target class
            temp_target = {}
            for t in predefined_model["nodes"]["leaf_nodes"]:
                new_idx = leaf_pattern(sub_leaf=t, depth=main_depth,
                                       leaf=node_)
                if t in main_model["details"]["target_class"]:
                    # less than 3 data points results with error
                    temp_target[new_idx] = \
                        main_model["details"]["target_class"][t]
                    if t in misclassified_leafs:
                        to_go_deep_nodes_[new_idx] = t
                    else:
                        pruned_nodes.append(new_idx)
            # reorganize pruned nodes - if one of the leaf has
            # misclassified, then its parent will reoptimize.
            temp_class_assign = copy.deepcopy(pruned_nodes)
            for i in temp_class_assign:
                for parent in parents_to_optimize:
                    if not (i != parent * 2 and i != (
                            parent * 2 +
                            1)):
                        pruned_nodes.remove(i)
                    if not (i != get_child(1, 2, parent) * 2 or i != (
                            get_child(1, 2, parent) * 2 +
                            1)):
                        pruned_nodes.remove(i)
            new_train_data_dict[node_] = new_train_data_dict[node_].drop(
                ["prediction",
                 "leaf"],
                axis=1)
            # eliminate branched on node from target class
            main_model["details"]["target_class"] = temp_target
            selected_features[node_] = copy.deepcopy(
                main_model["details"][
                    "selected_features"])
            final_model["details"]["var_a"].update(
                main_model["details"]["var_a"])
            for k in [node_ * 2, node_ * 2 + 1]:
                if k in final_model["details"]["target_class"]:
                    final_model["details"]["target_class"].pop(k)
            final_model["details"]["target_class"].update(
                main_model["details"]["target_class"])

        final_model["depth"] = main_depth + level
        result_ = predict_model(data=train_data,
                                model_dict=final_model,
                                P=features,
                                pruned_nodes=pruned_nodes)
        train_acc = len(
            result_.loc[result_["prediction"]
                        == result_["y"]]) / \
                    len(result_["y"])
        del result_
        result = predict_model(data=test_data,
                               model_dict=final_model,
                               P=features,
                               pruned_nodes=pruned_nodes)
        prediction_acc = len(
            result.loc[result["prediction"]
                       == result["y"]]) / \
                         len(result["y"])
        logging.info(
            f"Test Accuracy: {prediction_acc}. Training Accuracy: "
            f"{train_acc}. Iteration is over for level"
            f" {level}. Final "
            f"depth is {main_depth + level}")
        to_go_deep_nodes = to_go_deep_nodes_

        current_depth = main_depth + level
        result_dict[current_depth] = {
            "training_accuracy": train_acc,
            "test_accuracy": prediction_acc,
            "time": time.time() - iter_time

        }
    return result_dict
