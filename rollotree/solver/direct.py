"""Exact separable OCT-2 minimization over precomputed pair costs."""
import time
import numpy as np
from rollotree.solver.base import OCT2Solution, SolverStatus
from rollotree.tree.utils import get_leaf_paths_depth2


class DirectOCT2Solver:
    def __init__(self, config, criterion):
        self.config, self.criterion = config, criterion

    def solve(self, data, features, classes, y_idx=0):
        if self.config.direct_block_size is not None:
            from rollotree.solver.blocked import solve_blocked
            return solve_blocked(self.config, self.criterion, data, features, classes, y_idx)
        from rollotree.solver.pulp_solver import PuLPOCT2Solver
        started = time.perf_counter()
        def expired():
            return (
                self.config.deadline is not None
                and time.time() >= self.config.deadline
            ) or (
                self.config.time_limit is not None
                and time.perf_counter() - started >= self.config.time_limit
            )

        # Coefficient construction is cooperative work too. An already
        # expired request must not allocate the quadratic pair matrices.
        if expired():
            return OCT2Solution(status=SolverStatus.TIME_LIMIT,
                                coefficient_time=0.0, runtime=0.0)
        arr = np.asarray(data)
        paths = get_leaf_paths_depth2()
        costs = self.criterion.compute_leaf_coefficients(
            arr, features, list(paths), paths, classes, y_idx
        )
        if expired():
            return OCT2Solution(status=SolverStatus.TIME_LIMIT,
                                coefficient_time=time.perf_counter() - started,
                                runtime=0.0)
        left, right = PuLPOCT2Solver._compute_valid_pairs(
            arr, features, self.config.min_samples_leaf
        )
        coefficient_time = time.perf_counter() - started
        scan_started = time.perf_counter()
        best = None
        root_costs = {}
        interrupted = False
        for root in features:
            if expired():
                interrupted = True
                break
            ls = [(costs[4][root, child] + costs[5][root, child], pos, child)
                  for pos, child in enumerate(features) if (root, child) in left]
            rs = [(costs[6][root, child] + costs[7][root, child], pos, child)
                  for pos, child in enumerate(features) if (root, child) in right]
            if not ls or not rs:
                continue
            lc, _, lf = min(ls)
            rc, _, rf = min(rs)
            total = float(lc + rc)
            root_costs[root] = total
            if best is None or total < best[0]:
                best = (total, root, lf, rf)
        runtime = time.perf_counter() - scan_started
        status = SolverStatus.TIME_LIMIT if interrupted else SolverStatus.OPTIMAL
        if best is None:
            return OCT2Solution(status=status if interrupted else SolverStatus.INFEASIBLE,
                                coefficient_time=coefficient_time, runtime=runtime)
        total, root, lf, rf = best
        predictions = {}
        for leaf, (first, second) in paths.items():
            labels = arr[(arr[:, root] == first) &
                         (arr[:, lf if first else rf] == second), y_idx]
            values, counts = np.unique(labels, return_counts=True)
            predictions[leaf] = values[np.argmax(counts)]
        return OCT2Solution(
            status=status, root_feature=root, left_feature=lf,
            right_feature=rf, leaf_classes=predictions, objective_value=total,
            coefficient_time=coefficient_time, assembly_time=0.0,
            runtime=runtime, mip_gap=0.0 if status == SolverStatus.OPTIMAL else None,
            root_costs=root_costs,
        )
