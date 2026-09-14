"""Exact synthetic diagnostic pilot with controlled root perturbations."""
import argparse
import json
import subprocess
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from rollotree.diagnostics import commitment_regret
from rollotree.solver.base import SolverConfig
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import get_criterion
from rollotree.tree.scoring import score_tree
from rollotree import RollingOCT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/commitment_results.json")
    args=parser.parse_args()
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    records=[]
    X=np.array(list(product((0,1),repeat=4))*4)
    criterion=get_criterion("gini")
    config=SolverConfig(solver_name="direct",direct_block_size=2)
    with threadpool_limits(limits=1):
        for seed in range(60):
            y=np.tile(np.random.default_rng(seed).integers(0,2,16),4)
            data=pd.DataFrame(np.column_stack([y,X]))
            record=commitment_regret(data,[1,2,3,4],[0,1],criterion,config,remaining_depth=4)
            record["seed"]=seed
            roots=[]
            rng=np.random.default_rng(seed)
            for _ in range(3):
                perturbed=data.copy()
                chosen=rng.choice(len(data),size=3,replace=False)
                perturbed.iloc[chosen,0]=1-perturbed.iloc[chosen,0]
                sol=PuLPOCT2Solver(config,criterion).solve(perturbed,[1,2,3,4],[0,1])
                roots.append(sol.root_feature)
            record["perturbation_root_change_rate"]=sum(r!=record["shallow_root"] for r in roots)/3
            baseline=RollingOCT(solver="direct",depth=4,acceptance_policy="objective").fit(X,y)
            revised=RollingOCT(solver="direct",depth=4,acceptance_policy="objective",adaptive_lookahead="fixed",audit_budget=1).fit(X,y)
            record["final_objective_gain"]=score_tree(baseline.tree_,X,y,criterion)-score_tree(revised.tree_,X,y,criterion)
            records.append(record)
    Path(args.output).write_text(json.dumps(dict(revision=revision,dataset="all 16 binary four-feature patterns, repeated four times; seeded truth table", criterion="weighted_gini", perturbation="three fixed-seed flips of three labels; extra shallow solves are offline cost", records=records),indent=2)+"\n")


if __name__=="__main__":
    main()
