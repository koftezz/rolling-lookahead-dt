"""Compare rolling depth-2 and exact depth-3 seeds end to end.

The benchmark uses paired stratified folds and an identical randomly selected
feature set for both strategies. This isolates seed-tree quality from feature
selection luck while measuring requested depths 3, 4, and 5.

Usage:
    python benchmarks/bench_quality.py
    python benchmarks/bench_quality.py --folds 5 --seeds 0 1 2 --features 10
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollotree import RollingOCT


def weighted_gini(model, X, y):
    """Return the fitted tree's sample-weighted leaf Gini objective."""
    leaf_ids = model.apply(X)
    objective = 0.0
    for leaf_id in np.unique(leaf_ids):
        labels = np.asarray(y)[leaf_ids == leaf_id]
        _classes, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        objective += (len(labels) / len(y)) * (
            1.0 - float(np.sum(probabilities**2))
        )
    return objective


def load_wine():
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "rollotree" / "data" / "train.csv")
    test = pd.read_csv(root / "rollotree" / "data" / "test.csv")
    data = pd.concat([train, test], ignore_index=True)
    return data.drop(columns="y"), data["y"]


def benchmark(folds, seeds, feature_count, depths, time_limit, total_time_limit):
    X, y = load_wine()
    if feature_count > X.shape[1]:
        raise ValueError(
            f"features={feature_count} exceeds the dataset width {X.shape[1]}"
        )

    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2026)
    splits = list(splitter.split(X, y))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        selected = sorted(
            rng.choice(X.columns, size=feature_count, replace=False).tolist()
        )
        selected_X = X[selected]

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train = selected_X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = selected_X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            for requested_depth in depths:
                for strategy, initial_depth in (
                    ("rolling_depth2", 2),
                    ("exact_depth3", 3),
                ):
                    started = time.perf_counter()
                    try:
                        model = RollingOCT(
                            depth=requested_depth,
                            initial_depth=initial_depth,
                            criterion="gini",
                            solver="highs",
                            time_limit=time_limit,
                            total_time_limit=total_time_limit,
                            random_state=seed,
                            n_jobs=1,
                        ).fit(X_train, y_train)
                        row = {
                            "seed": seed,
                            "fold": fold,
                            "strategy": strategy,
                            "requested_depth": requested_depth,
                            "actual_depth": model.actual_depth_,
                            "leaves": model.get_n_leaves(),
                            "train_accuracy": model.score(X_train, y_train),
                            "test_accuracy": model.score(X_test, y_test),
                            "train_gini": weighted_gini(
                                model, X_train, y_train
                            ),
                            "fit_seconds": time.perf_counter() - started,
                            "fit_status": model.fit_status_,
                            "error": "",
                        }
                    except Exception as exc:
                        row = {
                            "seed": seed,
                            "fold": fold,
                            "strategy": strategy,
                            "requested_depth": requested_depth,
                            "actual_depth": np.nan,
                            "leaves": np.nan,
                            "train_accuracy": np.nan,
                            "test_accuracy": np.nan,
                            "train_gini": np.nan,
                            "fit_seconds": time.perf_counter() - started,
                            "fit_status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(row)
                    print(
                        f"seed={seed} fold={fold} depth={requested_depth} "
                        f"strategy={strategy} status={row['fit_status']} "
                        f"test={row['test_accuracy']:.4f} "
                        f"gini={row['train_gini']:.4f} "
                        f"seconds={row['fit_seconds']:.3f}",
                        flush=True,
                    )
    return pd.DataFrame(rows)


def print_summary(results):
    successful = results[results["fit_status"] != "error"].copy()
    summary = (
        successful.groupby(["requested_depth", "strategy"])
        .agg(
            runs=("test_accuracy", "size"),
            test_accuracy=("test_accuracy", "mean"),
            test_std=("test_accuracy", "std"),
            train_accuracy=("train_accuracy", "mean"),
            train_gini=("train_gini", "mean"),
            actual_depth=("actual_depth", "mean"),
            leaves=("leaves", "mean"),
            fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    print("\nAggregate results")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    paired = successful.pivot_table(
        index=["seed", "fold", "requested_depth"],
        columns="strategy",
        values=["test_accuracy", "train_accuracy", "train_gini", "fit_seconds"],
    ).dropna()
    deltas = pd.DataFrame(
        {
            "test_accuracy_delta": (
                paired["test_accuracy"]["exact_depth3"]
                - paired["test_accuracy"]["rolling_depth2"]
            ),
            "train_accuracy_delta": (
                paired["train_accuracy"]["exact_depth3"]
                - paired["train_accuracy"]["rolling_depth2"]
            ),
            "gini_reduction": (
                paired["train_gini"]["rolling_depth2"]
                - paired["train_gini"]["exact_depth3"]
            ),
            "runtime_ratio": (
                paired["fit_seconds"]["exact_depth3"]
                / paired["fit_seconds"]["rolling_depth2"]
            ),
        }
    )
    print("\nPaired exact-depth3 minus rolling-depth2 comparison")
    print(
        deltas.groupby("requested_depth")
        .mean()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )

    errors = results[results["fit_status"] == "error"]
    if len(errors):
        print("\nErrors")
        print(errors[["seed", "fold", "strategy", "requested_depth", "error"]])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--total-time-limit", type=float, default=120)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = benchmark(
        folds=args.folds,
        seeds=args.seeds,
        feature_count=args.features,
        depths=args.depths,
        time_limit=args.time_limit,
        total_time_limit=args.total_time_limit,
    )
    print_summary(results)
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nWrote raw results to {args.output}")
