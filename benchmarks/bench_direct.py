"""Fresh-process direct/HiGHS benchmark. Run from repository root.

python benchmarks/bench_direct.py --output benchmarks/direct_results.json
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def worker(p, backend, block):
    import numpy as np
    import pandas as pd
    from rollotree.solver.base import SolverConfig
    from rollotree.solver.pulp_solver import PuLPOCT2Solver
    from rollotree.tree.impurity import GiniCriterion
    rng = np.random.default_rng(42)
    data = pd.DataFrame(np.column_stack([rng.integers(0, 3, 160), rng.integers(0, 2, (160, p))]))
    solver = PuLPOCT2Solver(SolverConfig(solver_name=backend, direct_block_size=block), GiniCriterion())
    start = time.perf_counter()
    sol = solver.solve(data, list(range(1, p+1)), [0, 1, 2])
    return dict(p=p, backend=backend, block=block, seconds=time.perf_counter()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                coefficient_time=sol.coefficient_time, assembly_time=sol.assembly_time,
                solve_time=sol.runtime, objective=sol.objective_value, status=sol.status.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--output", default="benchmarks/direct_results.json")
    args = parser.parse_args()
    if args.worker:
        p, backend, block = args.worker
        print(json.dumps(worker(int(p), backend, None if block == "none" else int(block))))
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    results = []
    for repeat in range(3):
        for p in [20, 80, 130, 400]:
            methods = [("direct", "none"), ("direct", "8"), ("direct", "32")]
            if p <= 130:
                methods.append(("highs", "none"))
            # Reverse order on alternating repetitions; every observation is a fresh process.
            for backend, block in methods[::(-1 if repeat % 2 else 1)]:
                out = subprocess.check_output([sys.executable, __file__, "--worker", str(p), backend, block], env=env, text=True)
                result = json.loads(out)
                result["repeat"] = repeat
                results.append(result)
    import importlib.metadata
    payload = dict(dataset="synthetic binary n=160, three balanced-in-expectation classes", seed=42,
                   revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                   platform=platform.platform(), machine=platform.machine(), python=sys.version,
                   packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "pulp", "highspy"]},
                   threads=1, warmup="none; fresh process per observation", results=results)
    Path(args.output).write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
