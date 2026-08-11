"""Solver abstraction layer: status codes, result containers, and configuration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SolverStatus(Enum):
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"


@dataclass
class OCT2Solution:
    """Result of solving a single depth-2 OCT problem."""

    status: SolverStatus
    root_feature: Optional[int] = None
    left_feature: Optional[int] = None
    right_feature: Optional[int] = None
    leaf_classes: Optional[dict] = field(default_factory=dict)
    objective_value: Optional[float] = None
    runtime: Optional[float] = None
    mip_gap: Optional[float] = None


class SolverConfig:
    """Configuration for the optimization solver.

    Args:
        solver_name: "highs" (open-source, bundled with PuLP) or "gurobi" (commercial).
        time_limit: Maximum solve time per subproblem in seconds.
        mip_gap: Relative MIP optimality gap tolerance (e.g., 0.01 for 1%).
        log_to_console: Whether to print solver output.
        big_m: Penalty for empty-leaf feature pairs in the objective.
        min_samples_split: Minimum number of datapoints required to solve a
            subproblem during rolling expansion. Nodes with fewer samples are
            pruned instead of expanded.
        min_samples_leaf: Minimum number of datapoints required at each leaf
            node. Feature pairs that would produce a leaf with fewer samples
            are eliminated from the OCT-2 formulation.
    """

    def __init__(
        self,
        solver_name: str = "highs",
        time_limit: float = 1800,
        mip_gap: Optional[float] = None,
        log_to_console: bool = False,
        big_m: float = 99,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.solver_name = solver_name.lower()
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.log_to_console = log_to_console
        self.big_m = big_m
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        if self.solver_name not in ("highs", "gurobi", "cbc"):
            raise ValueError(
                f"Unknown solver '{solver_name}'. Choose 'highs', 'gurobi', or 'cbc'."
            )

    def copy_with(self, **changes) -> "SolverConfig":
        """Return a copy with selected values replaced."""
        values = {
            "solver_name": self.solver_name,
            "time_limit": self.time_limit,
            "mip_gap": self.mip_gap,
            "log_to_console": self.log_to_console,
            "big_m": self.big_m,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }
        values.update(changes)
        return SolverConfig(**values)
