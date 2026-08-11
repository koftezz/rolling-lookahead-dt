"""Tests for the PuLP-based OCT-2 solver."""

import numpy as np
import pandas as pd
import pytest

from rollotree.solver.base import SolverConfig, SolverStatus, OCT2Solution
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion


class TestSolverConfig:
    def test_defaults(self):
        config = SolverConfig()
        assert config.solver_name == "highs"
        assert config.time_limit == 1800
        assert config.mip_gap is None
        assert config.log_to_console is False
        assert config.big_m == 99

    def test_gurobi_config(self):
        config = SolverConfig(solver_name="gurobi", time_limit=60)
        assert config.solver_name == "gurobi"
        assert config.time_limit == 60

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            SolverConfig(solver_name="cplex")

    def test_case_insensitive(self):
        config = SolverConfig(solver_name="HiGHS")
        assert config.solver_name == "highs"


class TestPuLPOCT2Solver:
    def _make_separable_data(self):
        """Create a simple perfectly-separable dataset."""
        data = pd.DataFrame({
            "y": [1, 1, 2, 2, 1, 1, 2, 2],
            1: [1, 1, 0, 0, 1, 1, 0, 0],
            2: [1, 0, 1, 0, 1, 0, 1, 0],
        })
        return data, [1, 2], [1, 2]

    def test_solve_separable_problem(self):
        data, features, classes = self._make_separable_data()
        config = SolverConfig(solver_name="highs", time_limit=60)
        solver = PuLPOCT2Solver(config=config, criterion=GiniCriterion())

        solution = solver.solve(data=data, features=features, classes=classes)

        assert solution.status == SolverStatus.OPTIMAL
        assert solution.root_feature is not None
        assert solution.left_feature is not None
        assert solution.right_feature is not None

    def test_solve_returns_leaf_classes(self):
        data, features, classes = self._make_separable_data()
        config = SolverConfig(solver_name="highs", time_limit=60)
        solver = PuLPOCT2Solver(config=config, criterion=GiniCriterion())

        solution = solver.solve(data=data, features=features, classes=classes)

        assert solution.leaf_classes is not None
        assert len(solution.leaf_classes) > 0
        for leaf_id, cls in solution.leaf_classes.items():
            assert cls in classes

    def test_solve_objective_nonnegative(self):
        data, features, classes = self._make_separable_data()
        config = SolverConfig(solver_name="highs", time_limit=60)
        solver = PuLPOCT2Solver(config=config, criterion=GiniCriterion())

        solution = solver.solve(data=data, features=features, classes=classes)

        assert solution.objective_value is not None
        assert solution.objective_value >= 0

    def test_solve_with_misclassification_criterion(self):
        data, features, classes = self._make_separable_data()
        config = SolverConfig(solver_name="highs", time_limit=60)
        solver = PuLPOCT2Solver(
            config=config, criterion=MisclassificationCriterion()
        )

        solution = solver.solve(data=data, features=features, classes=classes)

        assert solution.status == SolverStatus.OPTIMAL

    def test_solve_larger_dataset(self, small_binary_dataset):
        data = small_binary_dataset
        features = [1, 2, 3, 4, 5]
        classes = sorted(data["y"].unique().tolist())

        config = SolverConfig(solver_name="highs", time_limit=120)
        solver = PuLPOCT2Solver(config=config, criterion=GiniCriterion())

        solution = solver.solve(data=data, features=features, classes=classes)

        assert solution.status == SolverStatus.OPTIMAL
        assert solution.root_feature in features
        assert solution.left_feature in features
        assert solution.right_feature in features

    def test_feasible_nonoptimal_incumbent_maps_to_time_limit(self, monkeypatch):
        """A backend time limit with a complete integer tree is usable."""
        import pulp

        def fake_solve(problem, _solver):
            for variable in problem.variables():
                variable.varValue = 0
            variables = {variable.name: variable for variable in problem.variables()}
            variables["x_1_2"].varValue = 1
            variables["y_1_2"].varValue = 1
            problem.status = pulp.LpStatusNotSolved
            problem.sol_status = pulp.LpSolutionIntegerFeasible
            problem.solutionTime = 0.01

        monkeypatch.setattr(pulp.LpProblem, "solve", fake_solve)
        data, features, classes = self._make_separable_data()
        solution = PuLPOCT2Solver(
            SolverConfig(time_limit=0.01), GiniCriterion()
        ).solve(data, features, classes)

        assert solution.status == SolverStatus.TIME_LIMIT
        assert solution.root_feature == 1
        assert set(solution.leaf_classes) == {4, 5, 6, 7}

    def test_time_limit_without_complete_incumbent_is_error(self, monkeypatch):
        """A backend stop without a complete assignment must not leak a tree."""
        import pulp

        def fake_solve(problem, _solver):
            problem.status = pulp.LpStatusNotSolved
            problem.sol_status = pulp.LpSolutionNoSolutionFound

        monkeypatch.setattr(pulp.LpProblem, "solve", fake_solve)
        data, features, classes = self._make_separable_data()
        solution = PuLPOCT2Solver(
            SolverConfig(time_limit=0.01), GiniCriterion()
        ).solve(data, features, classes)

        assert solution.status == SolverStatus.ERROR
        assert solution.root_feature is None


class TestSolverBackends:
    """Tests that compare HiGHS and Gurobi results (Gurobi tests are skipped if not installed)."""

    @staticmethod
    def _gurobi_available():
        try:
            from pulp import GUROBI_CMD
            return GUROBI_CMD().available()
        except Exception:
            return False

    @pytest.mark.skipif(
        not _gurobi_available.__func__(),
        reason="Gurobi not installed",
    )
    def test_gurobi_solver_produces_result(self):
        data = pd.DataFrame({
            "y": [1, 1, 2, 2],
            1: [1, 1, 0, 0],
            2: [0, 1, 0, 1],
        })
        config = SolverConfig(solver_name="gurobi", time_limit=60)
        solver = PuLPOCT2Solver(config=config, criterion=GiniCriterion())
        solution = solver.solve(data=data, features=[1, 2], classes=[1, 2])
        assert solution.status == SolverStatus.OPTIMAL

    @pytest.mark.skipif(
        not _gurobi_available.__func__(),
        reason="Gurobi not installed",
    )
    def test_both_solvers_same_features(self):
        data = pd.DataFrame({
            "y": [1, 1, 2, 2, 1, 1, 2, 2],
            1: [1, 1, 0, 0, 1, 1, 0, 0],
            2: [1, 0, 1, 0, 1, 0, 1, 0],
        })
        features = [1, 2]
        classes = [1, 2]

        highs_config = SolverConfig(solver_name="highs", time_limit=60)
        gurobi_config = SolverConfig(solver_name="gurobi", time_limit=60)

        highs_solver = PuLPOCT2Solver(
            config=highs_config, criterion=GiniCriterion()
        )
        gurobi_solver = PuLPOCT2Solver(
            config=gurobi_config, criterion=GiniCriterion()
        )

        h_sol = highs_solver.solve(data=data, features=features, classes=classes)
        g_sol = gurobi_solver.solve(data=data, features=features, classes=classes)

        assert h_sol.root_feature == g_sol.root_feature
        assert h_sol.left_feature == g_sol.left_feature
        assert h_sol.right_feature == g_sol.right_feature
