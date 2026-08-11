"""Parallel execution support for rolling OCT-2 subproblems.

Provides picklable data classes and a top-level worker function that
can be dispatched via ProcessPoolExecutor.  Each worker creates a
fresh PuLPOCT2Solver (PuLP objects are not picklable) and returns
the solution plus misclassified-leaf information.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from rollotree.solver.base import OCT2Solution, SolverConfig, SolverStatus
from rollotree.tree.impurity import ImpurityCriterion
from rollotree.tree.nodes import DecisionTree

logger = logging.getLogger(__name__)


@dataclass
class SubproblemInput:
    """Everything a worker process needs to solve one parent's OCT-2 subproblem."""

    parent_node: int
    leaf_ids: list
    parent_data: pd.DataFrame
    features: list
    sub_K: list
    y_idx: int
    solver_config: SolverConfig
    criterion: ImpurityCriterion
    deadline: Optional[float] = None


@dataclass
class SubproblemResult:
    """Result returned by a worker after solving one OCT-2 subproblem."""

    parent_node: int
    leaf_ids: list
    sub_solution: Optional[OCT2Solution]
    sub_misclassified: list
    skipped: bool
    n_samples: int
    parent_data: pd.DataFrame
    elapsed_time: float = 0.0
    timed_out: bool = False


def _solve_subproblem(inp: SubproblemInput) -> SubproblemResult:
    """Solve a single parent's OCT-2 subproblem (runs in a worker process).

    Creates a fresh PuLPOCT2Solver to avoid pickling PuLP internals,
    solves the MIP, builds a temporary sub-tree, and identifies
    misclassified leaves.
    """
    n_samples = len(inp.parent_data)
    started_at = time.perf_counter()

    if n_samples < inp.solver_config.min_samples_split:
        return SubproblemResult(
            parent_node=inp.parent_node,
            leaf_ids=inp.leaf_ids,
            sub_solution=None,
            sub_misclassified=[],
            skipped=True,
            n_samples=n_samples,
            parent_data=inp.parent_data,
            elapsed_time=time.perf_counter() - started_at,
        )

    solver_config = inp.solver_config
    if inp.deadline is not None:
        remaining = inp.deadline - time.perf_counter()
        if remaining <= 0:
            return SubproblemResult(
                parent_node=inp.parent_node,
                leaf_ids=inp.leaf_ids,
                sub_solution=None,
                sub_misclassified=[],
                skipped=False,
                n_samples=n_samples,
                parent_data=inp.parent_data,
                elapsed_time=time.perf_counter() - started_at,
                timed_out=True,
            )
        solver_config = solver_config.copy_with(
            time_limit=min(solver_config.time_limit, max(0.001, remaining))
        )

    # Fresh solver per process — PuLP solver objects are not picklable
    from rollotree.solver.pulp_solver import PuLPOCT2Solver

    solver = PuLPOCT2Solver(config=solver_config, criterion=inp.criterion)
    sub_solution = solver.solve(
        data=inp.parent_data,
        features=inp.features,
        classes=inp.sub_K,
        y_idx=inp.y_idx,
    )

    if sub_solution.status not in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT):
        return SubproblemResult(
            parent_node=inp.parent_node,
            leaf_ids=inp.leaf_ids,
            sub_solution=sub_solution,
            sub_misclassified=[],
            skipped=False,
            n_samples=n_samples,
            parent_data=inp.parent_data,
            elapsed_time=time.perf_counter() - started_at,
        )

    # Build temporary sub-tree to identify misclassified leaves
    sub_tree = DecisionTree(depth=2, features=inp.features)
    sub_tree.set_branch_feature(1, sub_solution.root_feature)
    sub_tree.set_branch_feature(2, sub_solution.left_feature)
    sub_tree.set_branch_feature(3, sub_solution.right_feature)
    for lid, cls in sub_solution.leaf_classes.items():
        sub_tree.set_leaf_class(lid, cls)

    sub_X = np.array(inp.parent_data[inp.features])
    sub_y = np.array(inp.parent_data.iloc[:, inp.y_idx])
    sub_misclassified = sub_tree.get_misclassified_leaves(sub_X, sub_y)

    return SubproblemResult(
        parent_node=inp.parent_node,
        leaf_ids=inp.leaf_ids,
        sub_solution=sub_solution,
        sub_misclassified=sub_misclassified,
        skipped=False,
        n_samples=n_samples,
        parent_data=inp.parent_data,
        elapsed_time=time.perf_counter() - started_at,
    )


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve *n_jobs* to an actual worker count (sklearn convention).

    * ``1``  → sequential (no pool)
    * ``-1`` → all CPU cores
    * ``-2`` → all cores minus one, etc.
    """
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)
