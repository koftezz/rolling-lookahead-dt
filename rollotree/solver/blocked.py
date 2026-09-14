"""Root-block coefficient calculation; no full pair matrices."""
import time
import numpy as np
from rollotree.solver.base import OCT2Solution, SolverStatus
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion


def solve_blocked(config, criterion, data, features, classes, y_idx=0):
    if not isinstance(criterion, (GiniCriterion, MisclassificationCriterion)):
        raise TypeError("Blocked direct supports built-in additive criteria")
    arr = np.asarray(data)
    F = np.asarray(arr[:, features], dtype=np.float64)
    labels = arr[:, y_idx]
    n, p = F.shape
    best = None
    roots = {}
    coefficient_time = 0.0
    scan_time = 0.0
    interrupted = False
    for start in range(0, p, config.direct_block_size):
        if config.deadline is not None and time.time() >= config.deadline:
            interrupted = True
            break
        tick = time.perf_counter()
        stop = min(p, start + config.direct_block_size)
        block = F[:, start:stop]
        def paths(matrix, root_block):
            both = root_block.T @ matrix
            rsum, csum = root_block.sum(axis=0), matrix.sum(axis=0)
            return [both, rsum[:, None] - both, csum[None, :] - both,
                    len(matrix) - rsum[:, None] - csum[None, :] + both]
        totals = paths(F, block)
        accum = [np.zeros_like(t) for t in totals]
        for label in classes:
            mask = labels == label
            counts = paths(F[mask], block[mask])
            for a, c in zip(accum, counts):
                if isinstance(criterion, GiniCriterion):
                    a += c * c
                else:
                    np.maximum(a, c, out=a)
        costs = []
        for total, a in zip(totals, accum):
            if isinstance(criterion, GiniCriterion):
                cost = (total - np.divide(a, total, out=np.zeros_like(a), where=total > 0)) / max(1, n)
            else:
                cost = total - a
            cost[total < config.min_samples_leaf] = np.inf
            costs.append(cost)
        left, right = costs[0] + costs[1], costs[2] + costs[3]
        coefficient_time += time.perf_counter() - tick
        tick = time.perf_counter()
        for row, pos in enumerate(range(start, stop)):
            li, ri = int(np.argmin(left[row])), int(np.argmin(right[row]))
            total = float(left[row, li] + right[row, ri])
            if not np.isfinite(total):
                continue
            roots[features[pos]] = total
            if best is None or total < best[0]:
                best = (total, features[pos], features[li], features[ri])
        scan_time += time.perf_counter() - tick
    status = SolverStatus.TIME_LIMIT if interrupted else SolverStatus.OPTIMAL
    if best is None:
        return OCT2Solution(status=status if interrupted else SolverStatus.INFEASIBLE,
                            coefficient_time=coefficient_time, runtime=scan_time)
    total, root, lf, rf = best
    predictions = {}
    for leaf, first, second in [(4,1,1), (5,1,0), (6,0,1), (7,0,0)]:
        values, counts = np.unique(labels[(arr[:, root] == first) &
                                          (arr[:, lf if first else rf] == second)], return_counts=True)
        predictions[leaf] = values[np.argmax(counts)]
    return OCT2Solution(status=status, root_feature=root, left_feature=lf,
                        right_feature=rf, leaf_classes=predictions, objective_value=total,
                        coefficient_time=coefficient_time, assembly_time=0.0,
                        runtime=scan_time, root_costs=roots,
                        mip_gap=0.0 if status == SolverStatus.OPTIMAL else None)
