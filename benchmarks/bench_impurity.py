"""Benchmark vectorized depth-2 impurity coefficient construction.

Usage:
    python benchmarks/bench_impurity.py
"""

from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollotree.tree.impurity import GiniCriterion
from rollotree.tree.utils import get_leaf_paths_depth2


def reference_gini(data, features, leaf_nodes, leaf_paths, classes):
    result = {leaf: {} for leaf in leaf_nodes}
    n_samples = len(data)
    for leaf in leaf_nodes:
        first, second = leaf_paths[leaf]
        for i, j in product(features, repeat=2):
            subset = data[(data[:, i] == first) & (data[:, j] == second)]
            if len(subset):
                counts = np.array([np.sum(subset[:, 0] == c) for c in classes])
                result[leaf][(i, j)] = (
                    len(subset) - np.sum(counts.astype(float) ** 2) / len(subset)
                ) / n_samples
    return result


def run_case(n_samples, n_features, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, n_features), dtype=np.int64)
    y = rng.integers(0, 3, size=n_samples, dtype=np.int64)
    data = np.column_stack([y, X])
    features = list(range(1, n_features + 1))
    leaves = [4, 5, 6, 7]
    paths = get_leaf_paths_depth2()
    classes = [0, 1, 2]

    started = time.perf_counter()
    expected = reference_gini(data, features, leaves, paths, classes)
    reference_seconds = time.perf_counter() - started

    started = time.perf_counter()
    actual = GiniCriterion().compute_leaf_coefficients(
        data, features, leaves, paths, classes
    )
    vectorized_seconds = time.perf_counter() - started

    for leaf in leaves:
        assert expected[leaf].keys() == actual[leaf].keys()
        np.testing.assert_allclose(
            list(expected[leaf].values()), list(actual[leaf].values())
        )
    return reference_seconds, vectorized_seconds


if __name__ == "__main__":
    print("samples features reference_s vectorized_s speedup")
    for feature_count in (32, 64, 128):
        reference, vectorized = run_case(512, feature_count)
        print(
            f"{512:7d} {feature_count:8d} {reference:11.4f} "
            f"{vectorized:12.4f} {reference / vectorized:7.1f}x"
        )
