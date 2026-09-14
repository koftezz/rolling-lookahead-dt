"""Paired pilot, not a tuned efficacy study. Training-only median thresholds.

Run: python benchmarks/evaluate_adaptive.py --output /tmp/adaptive_results.json
"""
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from rollotree import RollingOCT
from rollotree.tree.impurity import get_criterion
from rollotree.tree.scoring import score_tree


def dataset(name, seed):
    loaders = dict(iris=load_iris, wine=load_wine, breast_cancer=load_breast_cancer)
    if name in loaders:
        d = loaders[name]()
        return d.data, d.target
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (256, 6))
    if name == "xor3":
        y = X[:,0] ^ X[:,1] ^ X[:,2]
    elif name == "easy":
        y = X[:,0]
    elif name == "noisy":
        y = (X[:,0] ^ X[:,1]) ^ (rng.random(len(X)) < .3)
    else:
        raise ValueError(name)
    return X, y


def worker(name, seed, method, budget, policy="objective"):
    start = time.perf_counter()
    X, y = dataset(name, seed)
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=.3, stratify=y, random_state=seed)
    thresholds = np.median(X[train], axis=0)
    # Synthetic inputs already binary: preserve their candidates.
    binary = X if name in ("xor3", "easy", "noisy") else (X > thresholds).astype(int)
    prep = time.perf_counter()-start
    params = dict(depth=4, solver="direct", direct_block_size=8, criterion="gini",
                  acceptance_policy=policy, total_time_limit=budget, random_state=seed,
                  audit_budget=2, audit_min_samples=32, trace=True)
    if method == "highs":
        params.update(solver="highs", direct_block_size=None)
    elif method == "initial3":
        params.update(initial_depth=3)
    elif method == "exact3":
        params.update(initial_depth=3, depth=3)
    elif method in ("gap", "random", "impurity", "samples", "fixed"):
        params.update(adaptive_lookahead=method)
        if method == "samples":
            params.update(audit_min_samples=64)
        elif method == "impurity":
            params.update(audit_threshold=0.1)
    row = dict(dataset=name, seed=seed, method=method, budget=budget, n_train=len(train),
               n_test=len(test), n_features=binary.shape[1], n_classes=len(np.unique(y)),
               class_counts=np.bincount(y[train]).tolist(), preprocessing_seconds=prep,
               split_sha256=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest(),
               measured=True, params=params if method != "cart" else dict(max_depth=4, random_state=seed))
    fit_start = time.perf_counter()
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if method == "cart":
                model = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(binary[train], y[train])
            else:
                model = RollingOCT(**params).fit(binary[train], y[train])
        fit_seconds = time.perf_counter()-fit_start
        train_pred, test_pred = model.predict(binary[train]), model.predict(binary[test])
        if method == "cart":
            objective = sum(model.tree_.weighted_n_node_samples[i] / len(train) * model.tree_.impurity[i]
                            for i in range(model.tree_.node_count) if model.tree_.children_left[i] == -1)
            status, audits, audit_time, trace, tree = "completed", 0, 0.0, [], None
        else:
            objective = score_tree(model.tree_, binary[train], y[train], get_criterion("gini"))
            status, audits, audit_time, trace = model.fit_status_, model.audit_count_, model.audit_time_, model.search_trace_
            tree = dict(features=model.features_, depth=model.get_depth(),
                        branches={str(n): b.feature_index for n,b in model.tree_.branch_nodes.items() if b.feature_index is not None},
                        leaves={str(n): int(l.predicted_class) for n,l in model.tree_.leaf_nodes.items() if l.predicted_class is not None})
        row.update(fit_seconds=fit_seconds, total_seconds=prep+fit_seconds,
                   training_objective=float(objective), train_accuracy=float(accuracy_score(y[train],train_pred)),
                   test_accuracy=float(accuracy_score(y[test],test_pred)), balanced_accuracy=float(balanced_accuracy_score(y[test],test_pred)),
                   depth=int(model.get_depth()), leaves=int(model.get_n_leaves()), status=status, audit_count=audits,
                   audit_seconds=audit_time, trace=trace, tree=tree,
                   revised_roots=sum(e.get("adopted_as_candidate", False) and e.get("deeper_root") != e.get("baseline_root") for e in trace),
                   warning_categories=sorted(set(type(w.message).__name__ for w in caught)))
    except (ValueError, RuntimeError, TimeoutError) as exc:
        row.update(status="failed", error=str(exc), fit_seconds=time.perf_counter()-fit_start)
    row["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return row


def isolated_worker(connection, name, seed, method, budget):
    try:
        connection.send(worker(name, seed, method, budget))
    except Exception as exc:
        connection.send(dict(dataset=name, seed=seed, method=method, budget=budget, status="failed", error=repr(exc)))
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=4)
    parser.add_argument("--output", default="benchmarks/adaptive_results.json")
    args = parser.parse_args()
    if args.worker:
        name, seed, method, budget = args.worker
        print(json.dumps(worker(name, int(seed), method, float(budget))))
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    results = []
    for seed in [0, 1, 2]:
        for name in ["iris", "wine", "breast_cancer", "xor3", "easy", "noisy"]:
            for budget in [.05, .25]:
                methods = ["cart", "highs", "direct", "initial3", "exact3", "gap", "random", "impurity", "samples", "fixed"]
                # Seeded backend order; each fit gets a fresh process and identical split.
                np.random.default_rng(seed).shuffle(methods)
                for method in methods:
                    if "fork" in multiprocessing.get_all_start_methods():
                        context = multiprocessing.get_context("fork")
                        receive, send = context.Pipe(duplex=False)
                        process = context.Process(target=isolated_worker, args=(send, name, seed, method, budget))
                        with threadpool_limits(limits=1):
                            process.start()
                            send.close()
                            if not receive.poll(60):
                                process.terminate()
                                process.join()
                                raise RuntimeError("Experiment worker failed to return within 60 seconds")
                            results.append(receive.recv())
                            process.join()
                        receive.close()
                    else:
                        output = subprocess.check_output([sys.executable, __file__, "--worker", name, str(seed), method, str(budget)], text=True, env=env)
                        results.append(json.loads(output))
                print(f"Finished {name} seed={seed} budget={budget}", flush=True)
    payload = dict(schema_version=1, revision=revision, platform=platform.platform(), processor=platform.processor(),
                   python=sys.version, packages={p: importlib.metadata.version(p) for p in ["numpy","pandas","pulp","highspy","scikit-learn"]},
                   threads=1, warmup="no fit warmup; fresh sequential fork per fit with imports preloaded (subprocess fallback without fork)", criterion="weighted_gini",
                   timing="fit includes coefficients, assembly and audits; preprocessing separately recorded; imports excluded",
                   budget_note="equal cooperative upper bounds, not equal actual expenditure; CART has no enforced fit budget; exact3 has a smaller maximum depth",
                   results=results)
    Path(args.output).write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
