"""Optional Numba-accelerated tree operations.

When numba is installed, these functions are JIT-compiled and run at near-C
speed. Without numba, they fall back to equivalent pure-Python implementations.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _predict_batch_numba(X, branch_feature_positions, branch_max_id,
                         pruned_ids, leaf_ids, leaf_classes, depth):
    """Route all samples through the tree in a single compiled pass.

    Args:
        X: Feature array (n_samples, n_features).
        branch_feature_positions: Array where index t holds the feature
            position for branch node t (-1 if not a branch node).
        branch_max_id: Maximum branch node ID.
        pruned_ids: Sorted array of pruned node IDs.
        leaf_ids: Array of leaf node IDs.
        leaf_classes: Array of predicted classes, aligned with leaf_ids.
        depth: Maximum tree depth.

    Returns:
        Array of predicted classes (n_samples,).
    """
    n = X.shape[0]
    result_leaf_ids = np.empty(n, dtype=np.int64)

    for i in range(n):
        t = 1
        for _d in range(depth):
            if t > branch_max_id:
                break
            feat_pos = branch_feature_positions[t]
            if feat_pos < 0:
                break
            if X[i, feat_pos] == 1:
                t = t * 2
            else:
                t = t * 2 + 1
            # Check if pruned (binary search on sorted array)
            if _in_sorted(pruned_ids, t):
                break
        result_leaf_ids[i] = t

    return result_leaf_ids


@njit(cache=True)
def _in_sorted(arr, val):
    """Binary search for val in sorted array."""
    lo = 0
    hi = len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == val:
            return True
        elif arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid - 1
    return False
