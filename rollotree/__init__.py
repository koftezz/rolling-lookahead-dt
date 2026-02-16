"""RolloTree: Rolling Optimal Classification Trees.

Implements the "Rolling Lookahead Learning for Optimal Classification Trees"
algorithm using PuLP for solver-agnostic optimization (HiGHS or Gurobi).
"""

__version__ = "2.0.0"

from rollotree.classifier import RollingOCT
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.tree.nodes import DecisionTree
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion
from rollotree.tree.export import export_text, export_graphviz

__all__ = [
    "RollingOCT",
    "SolverConfig",
    "SolverStatus",
    "DecisionTree",
    "GiniCriterion",
    "MisclassificationCriterion",
    "export_text",
    "export_graphviz",
]
