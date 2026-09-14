"""Reconstruct saved trees and independently verify displayed measurements."""
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rollotree.tree.nodes import DecisionTree
from rollotree.tree.scoring import score_tree
from rollotree.tree.impurity import get_criterion
from evaluate_adaptive import dataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument("path",nargs="?",default="benchmarks/adaptive_results.json");args=parser.parse_args()
    payload=json.load(open(args.path));checked=0
    for r in payload["results"]:
        if not r.get("tree"):
            continue
        X,y=dataset(r["dataset"],r["seed"])
        train,test=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=r["seed"])
        binary=X if r["dataset"] in ("xor3","easy","noisy") else (X>np.median(X[train],axis=0)).astype(int)
        t=r["tree"];tree=DecisionTree(t["depth"],t["features"])
        tree.branch_nodes.clear();tree.leaf_nodes.clear()
        for n,f in t["branches"].items():tree.set_branch_feature(int(n),f)
        for n,label in t["leaves"].items():tree.set_leaf_class(int(n),label)
        assert abs(score_tree(tree,binary[train],y[train],get_criterion("gini"))-r["training_objective"])<1e-10
        assert abs(accuracy_score(y[test],tree.predict(binary[test]))-r["test_accuracy"])<1e-10
        assert tree.get_depth()==r["depth"]
        assert tree.get_n_leaves()==r["leaves"]
        assert r["trace"][-1]["event"]=="stop"
        checked+=1
    print(f"Verified {checked} saved fitted trees, objectives, held-out accuracies and terminal traces.")


if __name__=="__main__":main()
