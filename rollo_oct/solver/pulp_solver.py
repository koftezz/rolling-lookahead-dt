"""PuLP-based OCT-2 solver supporting HiGHS and Gurobi backends."""

import logging
import numpy as np
import pandas as pd
from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    lpSum,
    value,
    LpStatus,
)

from rollo_oct.solver.base import OCT2Solution, SolverConfig, SolverStatus
from rollo_oct.tree.impurity import ImpurityCriterion
from rollo_oct.tree.utils import generate_nodes, get_leaf_paths_depth2

logger = logging.getLogger(__name__)


class PuLPOCT2Solver:
    """
    Solves the OCT-2 (depth-2 optimal classification tree) MIP formulation
    using PuLP with a configurable backend solver (HiGHS or Gurobi).

    The formulation has binary variables x[i,j] for the left subtree
    and y[i,k] for the right subtree, with constraints ensuring exactly
    one split pair is selected per subtree, and the root feature matches.
    """

    def __init__(self, config: SolverConfig, criterion: ImpurityCriterion):
        self.config = config
        self.criterion = criterion

    def solve(
        self,
        data: pd.DataFrame,
        features: list,
        classes: list,
        y_idx: int = 0,
    ) -> OCT2Solution:
        """
        Build and solve the OCT-2 formulation.

        Args:
            data: DataFrame with target in column y_idx and binary features.
            features: List of feature column indices (P).
            classes: List of unique class labels (K).
            y_idx: Column index of target variable.

        Returns:
            OCT2Solution with extracted tree structure.
        """
        P = features
        K = classes
        leaf_paths = get_leaf_paths_depth2()
        _parent_nodes, leaf_nodes = generate_nodes(2)

        # Compute impurity coefficients
        data_arr = np.array(data)
        coef_dict = self.criterion.compute_leaf_coefficients(
            data=data_arr,
            features=P,
            leaf_nodes=leaf_nodes,
            leaf_paths=leaf_paths,
            classes=K,
            y_idx=y_idx,
        )

        # Determine which (i,j) pairs to keep based on min_samples_leaf
        min_leaf = self.config.min_samples_leaf
        valid_x_pairs = set()
        valid_y_pairs = set()

        for i in P:
            left_mask = data_arr[:, i] == 1
            right_mask = ~left_mask
            for j in P:
                # Left subtree leaves: 4 (j=1) and 5 (j=0)
                n_leaf4 = int(np.sum(left_mask & (data_arr[:, j] == 1)))
                n_leaf5 = int(np.sum(left_mask & (data_arr[:, j] == 0)))
                if n_leaf4 >= min_leaf and n_leaf5 >= min_leaf:
                    valid_x_pairs.add((i, j))

                # Right subtree leaves: 6 (j=1) and 7 (j=0)
                n_leaf6 = int(np.sum(right_mask & (data_arr[:, j] == 1)))
                n_leaf7 = int(np.sum(right_mask & (data_arr[:, j] == 0)))
                if n_leaf6 >= min_leaf and n_leaf7 >= min_leaf:
                    valid_y_pairs.add((i, j))

        # Fall back to all pairs if elimination removes everything
        if not valid_x_pairs:
            valid_x_pairs = {(i, j) for i in P for j in P}
        if not valid_y_pairs:
            valid_y_pairs = {(i, j) for i in P for j in P}

        logger.info(
            f"Variable elimination: {len(valid_x_pairs)}/{len(P)**2} x-pairs, "
            f"{len(valid_y_pairs)}/{len(P)**2} y-pairs"
        )

        # Build PuLP model
        prob = LpProblem("OCT2", LpMinimize)

        x = {
            (i, j): LpVariable(f"x_{i}_{j}", cat="Binary")
            for (i, j) in valid_x_pairs
        }
        y = {
            (i, k): LpVariable(f"y_{i}_{k}", cat="Binary")
            for (i, k) in valid_y_pairs
        }

        # Constraint 1: exactly one (i,j) pair for left subtree
        prob += lpSum(x[pair] for pair in valid_x_pairs) == 1, "one_left_pair"

        # Constraint 2: exactly one (i,k) pair for right subtree
        prob += lpSum(y[pair] for pair in valid_y_pairs) == 1, "one_right_pair"

        # Constraint 3: same root feature for both subtrees
        for i in P:
            x_sum = lpSum(x[i, j] for j in P if (i, j) in valid_x_pairs)
            y_sum = lpSum(y[i, k] for k in P if (i, k) in valid_y_pairs)
            prob += (x_sum == y_sum, f"same_root_{i}")

        # Objective: minimize total impurity across all 4 leaves
        big_m = self.config.big_m
        obj = lpSum(
            (coef_dict[4].get((i, j), big_m) + coef_dict[5].get((i, j), big_m))
            * x[i, j]
            for (i, j) in valid_x_pairs
        ) + lpSum(
            (coef_dict[6].get((i, k), big_m) + coef_dict[7].get((i, k), big_m))
            * y[i, k]
            for (i, k) in valid_y_pairs
        )
        prob += obj

        # Select and configure solver backend
        solver = self._get_solver()

        logger.info("Solving OCT-2 formulation...")
        prob.solve(solver)
        status_str = LpStatus.get(prob.status, "Undefined")
        logger.info(f"Solver status: {status_str}")

        if status_str == "Infeasible":
            return OCT2Solution(status=SolverStatus.INFEASIBLE)
        elif status_str == "Unbounded":
            return OCT2Solution(status=SolverStatus.UNBOUNDED)
        elif status_str != "Optimal":
            return OCT2Solution(status=SolverStatus.ERROR)

        # Extract solution
        root_feature = None
        left_feature = None
        right_feature = None

        for (i, j) in valid_x_pairs:
            xval = x[i, j].varValue
            if xval is not None and xval > 0.5:
                root_feature = i
                left_feature = j
                logger.info(
                    f"Left split: root feature={i}, second feature={j}"
                )
                break
        for (i, k) in valid_y_pairs:
            yval = y[i, k].varValue
            if yval is not None and yval > 0.5:
                root_feature = i
                right_feature = k
                logger.info(
                    f"Right split: root feature={i}, second feature={k}"
                )
                break

        # Determine leaf classes by majority vote
        leaf_classes = {}
        for leaf in leaf_nodes:
            first_val, second_val = leaf_paths[leaf]
            arr = data_arr[np.where(data_arr[:, root_feature] == first_val)]
            subtree_feature = left_feature if first_val == 1 else right_feature
            arr2 = arr[np.where(arr[:, subtree_feature] == second_val)]
            if len(arr2) > 0:
                values, counts = np.unique(arr2[:, y_idx], return_counts=True)
                leaf_classes[leaf] = values[np.argmax(counts)]

        obj_val = value(prob.objective)
        runtime = getattr(prob, "solutionTime", None)

        logger.info(f"OCT-2 solved. Objective={obj_val}, Runtime={runtime}")

        return OCT2Solution(
            status=SolverStatus.OPTIMAL,
            root_feature=root_feature,
            left_feature=left_feature,
            right_feature=right_feature,
            leaf_classes=leaf_classes,
            objective_value=obj_val,
            runtime=runtime,
            mip_gap=None,
        )

    def _get_solver(self):
        """Instantiate the PuLP solver backend based on config."""
        if self.config.solver_name == "gurobi":
            from pulp import GUROBI_CMD

            options = [("TimeLimit", self.config.time_limit)]
            if self.config.mip_gap is not None:
                options.append(("MIPGap", self.config.mip_gap))
            return GUROBI_CMD(
                msg=int(self.config.log_to_console),
                options=options,
            )
        elif self.config.solver_name == "cbc":
            from pulp import PULP_CBC_CMD

            kwargs = {
                "timeLimit": self.config.time_limit,
                "msg": int(self.config.log_to_console),
            }
            if self.config.mip_gap is not None:
                kwargs["gapRel"] = self.config.mip_gap
            return PULP_CBC_CMD(**kwargs)
        else:
            # Default: HiGHS (native Python API via highspy)
            try:
                from pulp import HiGHS_CMD

                solver = HiGHS_CMD(
                    timeLimit=self.config.time_limit,
                    msg=int(self.config.log_to_console),
                    gapRel=self.config.mip_gap,
                )
                if solver.available():
                    return solver
            except Exception:
                pass

            # Fall back to native HiGHS Python API
            from pulp import getSolver

            return getSolver(
                "HiGHS",
                timeLimit=self.config.time_limit,
                msg=int(self.config.log_to_console),
                gapRel=self.config.mip_gap,
            )
