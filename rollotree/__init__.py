"""RolloTree: Rolling Optimal Classification Trees.

Implements the "Rolling Lookahead Learning for Optimal Classification Trees"
algorithm using PuLP for solver-agnostic optimization (HiGHS or Gurobi).
"""

from rollotree._version import __version__

from rollotree.classifier import RollingOCT
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.depth3 import (
    ExactDepth3Solver,
    OCT3CandidateDiagnostic,
    OCT3Solution,
)
from rollotree.rolling.optimizer import DepthResult, SubproblemDiagnostic
from rollotree.tree.nodes import DecisionTree
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion
from rollotree.tree.export import export_text, export_graphviz

__all__ = [
    "RollingOCT",
    "SolverConfig",
    "SolverStatus",
    "ExactDepth3Solver",
    "OCT3CandidateDiagnostic",
    "OCT3Solution",
    "DepthResult",
    "SubproblemDiagnostic",
    "DecisionTree",
    "GiniCriterion",
    "MisclassificationCriterion",
    "export_text",
    "export_graphviz",
]
