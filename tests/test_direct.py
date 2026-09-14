"""Raw-row exhaustive oracle, independent of production pair coefficients."""
from itertools import product
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from rollotree import RollingOCT
from rollotree.solver.base import SolverConfig, SolverStatus
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import get_criterion


def raw_cost(data, triple, criterion, support=1):
    root, left, right = triple
    total = 0.0
    for first, second in product((1, 0), repeat=2):
        subset = data[(data[:, root] == first) &
                      (data[:, left if first else right] == second), 0]
        if len(subset) < support:
            return float("inf")
        _, counts = np.unique(subset, return_counts=True)
        total += (len(subset) - max(counts) if criterion == "misclassification"
                  else len(subset) / len(data) * (1 - sum((counts / len(subset)) ** 2)))
    return total


@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
@pytest.mark.parametrize("backend", ["direct", "cbc", "highs"])
@pytest.mark.parametrize("seed", range(5))
def test_raw_oracle(criterion, backend, seed):
    rng = np.random.default_rng(seed)
    arr = np.column_stack([rng.integers(0, 3, 32), rng.integers(0, 2, (32, 4))])
    features = [1, 2, 3, 4]
    expected = min(raw_cost(arr, triple, criterion, 2)
                   for triple in product(features, repeat=3))
    sol = PuLPOCT2Solver(SolverConfig(solver_name=backend, min_samples_leaf=2),
                        get_criterion(criterion)).solve(pd.DataFrame(arr), features, [0, 1, 2])
    assert sol.status == SolverStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(expected, abs=1e-10)
    assert raw_cost(arr, (sol.root_feature, sol.left_feature, sol.right_feature),
                    criterion, 2) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
@pytest.mark.parametrize("backend", ["cbc", "highs", "direct"])
def test_zero_and_pure_depth3(criterion, backend):
    X = np.array(list(product((0, 1), repeat=3)) * 2)
    model = clone(RollingOCT(solver=backend, criterion=criterion, depth=3, initial_depth=3))
    model.fit(X, X[:, 0])
    assert model.score(X, X[:, 0]) == 1
    assert all(d.objective_value == 0.0 for d in model.subproblem_diagnostics_)


@pytest.mark.parametrize("backend", ["cbc", "highs", "direct"])
def test_infeasible(backend):
    arr = np.array([[0, 0, 0], [1, 1, 1]])
    sol = PuLPOCT2Solver(SolverConfig(solver_name=backend), get_criterion("gini")).solve(
        pd.DataFrame(arr), [1, 2], [0, 1])
    assert sol.status == SolverStatus.INFEASIBLE
    assert sol.objective_value is None


@pytest.mark.parametrize("bad", [None, 0.5, 1.0])
def test_invalid_assignment(monkeypatch, bad):
    import pulp
    def fake(problem, solver):
        for variable in problem.variables():
            variable.varValue = bad
        problem.status = pulp.LpStatusOptimal
        problem.sol_status = pulp.LpSolutionOptimal
    monkeypatch.setattr(pulp.LpProblem, "solve", fake)
    X = np.array(list(product((0, 1), repeat=3)))
    arr = np.column_stack([X[:, 0], X])
    sol = PuLPOCT2Solver(SolverConfig(), get_criterion("gini")).solve(pd.DataFrame(arr), [1,2,3], [0,1])
    assert sol.status == SolverStatus.ERROR
