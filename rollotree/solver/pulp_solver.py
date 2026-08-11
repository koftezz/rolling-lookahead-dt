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
    LpSolutionIntegerFeasible,
    LpSolutionOptimal,
)

from rollotree.solver.base import OCT2Solution, SolverConfig, SolverStatus
from rollotree.tree.impurity import ImpurityCriterion
from rollotree.tree.utils import generate_nodes, get_leaf_paths_depth2

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
        # Uses matrix multiply to compute all co-occurrence counts at once
        min_leaf = self.config.min_samples_leaf
        valid_x_pairs, valid_y_pairs = self._compute_valid_pairs(
            data_arr, P, min_leaf
        )

        # Both sides must admit a pair and share at least one root feature.
        # Falling back to all pairs silently violated min_samples_leaf.
        valid_left_roots = {i for i, _j in valid_x_pairs}
        valid_right_roots = {i for i, _j in valid_y_pairs}
        common_roots = valid_left_roots & valid_right_roots
        if not valid_x_pairs or not valid_y_pairs or not common_roots:
            return OCT2Solution(status=SolverStatus.INFEASIBLE)

        valid_x_pairs = sorted(
            pair for pair in valid_x_pairs if pair[0] in common_roots
        )
        valid_y_pairs = sorted(
            pair for pair in valid_y_pairs if pair[0] in common_roots
        )

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

        # Extract solution
        left_root_feature = None
        right_root_feature = None
        left_feature = None
        right_feature = None

        for (i, j) in valid_x_pairs:
            xval = x[i, j].varValue
            if xval is not None and xval > 0.5:
                left_root_feature = i
                left_feature = j
                logger.info(
                    f"Left split: root feature={i}, second feature={j}"
                )
                break
        for (i, k) in valid_y_pairs:
            yval = y[i, k].varValue
            if yval is not None and yval > 0.5:
                right_root_feature = i
                right_feature = k
                logger.info(
                    f"Right split: root feature={i}, second feature={k}"
                )
                break

        complete_incumbent = (
            left_root_feature is not None
            and right_root_feature is not None
            and left_root_feature == right_root_feature
            and left_feature is not None
            and right_feature is not None
        )
        if not complete_incumbent:
            if status_str == "Optimal":
                logger.error("Optimal solve did not contain a complete tree")
            return OCT2Solution(status=SolverStatus.ERROR)

        root_feature = left_root_feature

        # Native HiGHS and CBC can expose a time-limited incumbent with an
        # ``Optimal`` problem status while retaining the real distinction in
        # ``sol_status``.  Inspect the latter first so a feasible incumbent is
        # never promoted to a false optimality certificate.
        solution_status_code = getattr(prob, "sol_status", None)
        if solution_status_code == LpSolutionIntegerFeasible:
            solution_status = SolverStatus.TIME_LIMIT
        elif (
            status_str == "Optimal"
            and solution_status_code == LpSolutionOptimal
        ):
            solution_status = SolverStatus.OPTIMAL
        else:
            return OCT2Solution(status=SolverStatus.ERROR)

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
            status=solution_status,
            root_feature=root_feature,
            left_feature=left_feature,
            right_feature=right_feature,
            leaf_classes=leaf_classes,
            objective_value=obj_val,
            runtime=runtime,
            mip_gap=None,
        )

    @staticmethod
    def _compute_valid_pairs(data_arr, features, min_leaf):
        """Vectorized variable elimination using matrix operations.

        Replaces O(|P|^2 * n_samples) Python loops with a single matrix
        multiply to compute all co-occurrence counts simultaneously.
        """
        # Extract binary feature columns as a float matrix for matmul
        F = data_arr[:, features].astype(np.float64)  # (n_samples, n_features)
        n = F.shape[0]

        # F.T @ F gives count(fi==1 AND fj==1) for all (i,j) pairs
        FT_F = F.T @ F  # (n_features, n_features)
        col_sums = F.sum(axis=0)  # count(fi==1) for each feature

        # Derive all four leaf counts from the co-occurrence matrix:
        # Left subtree (fi==1):
        #   leaf4: fi==1 AND fj==1 = FT_F[i,j]
        #   leaf5: fi==1 AND fj==0 = col_sums[i] - FT_F[i,j]
        left_both_1 = FT_F
        left_fi1_fj0 = col_sums[:, None] - FT_F

        # Right subtree (fi==0):
        #   leaf6: fi==0 AND fj==1 = col_sums[j] - FT_F[i,j]
        #   leaf7: fi==0 AND fj==0 = n - col_sums[i] - col_sums[j] + FT_F[i,j]
        right_fi0_fj1 = col_sums[None, :] - FT_F
        right_both_0 = n - col_sums[:, None] - col_sums[None, :] + FT_F

        # Valid pairs: both leaves in the subtree meet min_leaf threshold
        valid_x_mask = (left_both_1 >= min_leaf) & (left_fi1_fj0 >= min_leaf)
        valid_y_mask = (right_fi0_fj1 >= min_leaf) & (right_both_0 >= min_leaf)

        # Convert boolean matrices to sets of (feature_i, feature_j)
        x_rows, x_cols = np.where(valid_x_mask)
        y_rows, y_cols = np.where(valid_y_mask)

        valid_x_pairs = {(features[i], features[j]) for i, j in zip(x_rows, x_cols)}
        valid_y_pairs = {(features[i], features[j]) for i, j in zip(y_rows, y_cols)}

        return valid_x_pairs, valid_y_pairs

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
