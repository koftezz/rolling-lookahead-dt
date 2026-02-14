"""RollOCT: Rolling Optimal Classification Trees.

Implements the "Rolling Lookahead Learning for Optimal Classification Trees"
algorithm using PuLP for solver-agnostic optimization (HiGHS or Gurobi).
"""

__version__ = "0.1.0"

from rollo_oct.classifier import RollingOCT
from rollo_oct.solver.base import SolverConfig, SolverStatus
from rollo_oct.tree.nodes import DecisionTree
from rollo_oct.tree.impurity import GiniCriterion, MisclassificationCriterion

__all__ = [
    "RollingOCT",
    "SolverConfig",
    "SolverStatus",
    "DecisionTree",
    "GiniCriterion",
    "MisclassificationCriterion",
]
