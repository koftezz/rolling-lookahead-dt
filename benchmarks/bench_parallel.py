"""Benchmark: wall-clock time comparison for n_jobs=1 vs n_jobs=-1.

Usage:
    python benchmarks/bench_parallel.py
"""

import os
import sys
import time

import pandas as pd

# Ensure the package is importable even when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rollotree import RollingOCT


def load_wine():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "rollotree", "data")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train = train.drop("y", axis=1)
    y_train = train["y"]
    X_test = test.drop("y", axis=1)
    y_test = test["y"]
    return X_train, y_train, X_test, y_test


def benchmark(depth, n_jobs, X_train, y_train, X_test, y_test):
    model = RollingOCT(depth=depth, solver="highs", n_jobs=n_jobs, time_limit=300)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    score = model.score(X_test, y_test)
    return elapsed, score


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_wine()
    print(
        f"Dataset: {len(X_train)} train / {len(X_test)} test, "
        f"{X_train.shape[1]} features, {len(y_train.unique())} classes"
    )
    print(f"CPU count: {os.cpu_count()}")
    print()

    for depth in [3, 4, 5]:
        print(f"--- Depth {depth} ---")
        for n_jobs in [1, 2, -1]:
            elapsed, score = benchmark(
                depth, n_jobs, X_train, y_train, X_test, y_test
            )
            label = f"n_jobs={n_jobs}" if n_jobs > 0 else "n_jobs=-1 (all cores)"
            print(f"  {label:30s}  time={elapsed:7.2f}s  test_acc={score:.4f}")
        print()
