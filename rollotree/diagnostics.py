"""Expensive offline depth-3 commitment diagnostics; never a free selector."""
import time
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.solver.depth3 import ExactDepth3Solver


def commitment_regret(data, features, classes, criterion, config=None, y_idx=0, remaining_depth=3):
    """Compare exact fixed/free depth-3 trees over identical candidates/support.

    Nonoptimal or infeasible comparisons return regret=None, never a difference
    between arbitrary incumbents disguised as exact regret.
    """
    config = config or SolverConfig(solver_name="direct")
    start = time.perf_counter()
    shallow = PuLPOCT2Solver(config, criterion).solve(data, features, classes, y_idx)
    record = dict(n_samples=len(data), remaining_depth=remaining_depth,
                  shallow_status=shallow.status.value, shallow_root=shallow.root_feature,
                  shallow_objective=shallow.objective_value, commitment_regret=None,
                  exact=False, root_gap=None, roots_differ=None)
    costs = sorted(shallow.root_costs.values())
    if len(costs) > 1:
        record["root_gap"] = float(costs[1]-costs[0])
    if shallow.status != SolverStatus.OPTIMAL or remaining_depth < 3:
        record.update(reason="shallow not exact or insufficient remaining depth", seconds=time.perf_counter()-start)
        return record
    deeper_start = time.perf_counter()
    solver = ExactDepth3Solver(config.copy_with(mip_gap=0), criterion, deadline=config.deadline)
    fixed = solver.solve(data, features, classes, y_idx, fixed_root=shallow.root_feature)
    free = solver.solve(data, features, classes, y_idx)
    record.update(fixed_status=fixed.status.value, free_status=free.status.value,
                  fixed_objective=fixed.objective_value, free_objective=free.objective_value,
                  deeper_root=free.branch_features.get(1), additional_seconds=time.perf_counter()-deeper_start)
    if fixed.status == free.status == SolverStatus.OPTIMAL:
        regret = fixed.objective_value - free.objective_value
        if regret < -1e-10:
            raise RuntimeError("Nested exact objectives violate nonnegative regret")
        record.update(commitment_regret=max(0.0, float(regret)), exact=True,
                      roots_differ=shallow.root_feature != free.branch_features[1])
    else:
        record["reason"] = "no comparable exact fixed/free solutions"
    record["seconds"] = time.perf_counter()-start
    return record
