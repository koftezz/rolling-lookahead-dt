"""Policy choices cannot bypass independently checked numerical boundaries."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from rollotree import AuditDecision, RollingOCT
from rollotree.rolling.adaptive import maybe_audit, solution_tree
from rollotree.rolling.optimizer import RollingOptimizer
from rollotree.rolling.parallel import SubproblemResult
from rollotree.solver.base import OCT2Solution, SolverConfig, SolverStatus
from rollotree.solver.depth3 import ExactDepth3Solver, OCT3Solution
from rollotree.solver.pulp_solver import PuLPOCT2Solver
from rollotree.tree.impurity import get_criterion
from rollotree.tree.nodes import DecisionTree
from rollotree.tree.scoring import score_tree


class RecordingSelector:
    def __init__(self, action="audit"):
        self.action = action
        self.states = []

    def select(self, state, *, timeout):
        self.states.append(state)
        return AuditDecision(self.action, confidence=1.0)


def _parity_data(repeats=4):
    X = np.array(list(product((0, 1), repeat=3)) * repeats)
    y = X.sum(axis=1) % 2
    return pd.DataFrame(np.column_stack([y, X]))


def _optimizer(criterion="gini", acceptance="objective", selector=None):
    return RollingOptimizer(
        SolverConfig(solver_name="direct"), get_criterion(criterion),
        adaptive_lookahead="policy", audit_selector=selector or RecordingSelector(),
        audit_min_samples=1, audit_budget=2, policy_time_limit=10,
        acceptance_policy=acceptance, trace=True,
    )


def _shallow(data, criterion):
    features = list(data.columns[1:])
    shallow = PuLPOCT2Solver(SolverConfig(solver_name="direct"), criterion).solve(
        data, features, [0, 1], y_idx=0,
    )
    return shallow, solution_tree(shallow, features)


def _audit(opt, baseline, shallow, data):
    return maybe_audit(opt, baseline, shallow, data, baseline.features, 1, 3, 0)


@pytest.mark.parametrize("acceptance", ["accuracy", "objective"])
@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
def test_policy_cannot_override_solver_objective_mismatch(monkeypatch, acceptance, criterion):
    data = _parity_data()
    opt = _optimizer(criterion, acceptance)
    shallow, baseline = _shallow(data, opt.criterion)
    deeper = ExactDepth3Solver(opt.solver_config, opt.criterion).solve(
        data, baseline.features, [0, 1], y_idx=0,
    )
    assert deeper.objective_value == 0
    # The routed candidate is perfect, but disagrees with the solver report.
    deeper.objective_value = 1.0
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: deeper)

    assert _audit(opt, baseline, shallow, data) is baseline
    event = opt.search_trace_[-1]
    assert opt.policy_call_count_ == opt.audit_count_ == 1
    assert event["proposed_accuracy"] == 1.0
    assert event["adopted_as_candidate"] is False
    assert event["outcome"] == "objective_mismatch"
    assert event["final_commit"] is False


@pytest.mark.parametrize("acceptance", ["accuracy", "objective"])
@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
def test_policy_cannot_override_worsening_audit(monkeypatch, acceptance, criterion):
    X = np.array(list(product((0, 1), repeat=4)) * 2)
    data = pd.DataFrame(np.column_stack([X[:, 0], X]))
    opt = _optimizer(criterion, acceptance)
    shallow, baseline = _shallow(data, opt.criterion)
    # A feasible complete tree ignores the informative first feature.
    candidate = DecisionTree(3, baseline.features)
    branches = {1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4, 7: 4}
    leaves = {leaf: 0 for leaf in range(8, 16)}
    for node, feature in branches.items():
        candidate.set_branch_feature(node, feature)
    for node, label in leaves.items():
        candidate.set_leaf_class(node, label)
    deeper = OCT3Solution(
        SolverStatus.OPTIMAL, branch_features=branches, leaf_classes=leaves,
        objective_value=score_tree(candidate, X, X[:, 0], opt.criterion),
        n_complete_candidates=1,
    )
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: deeper)

    assert _audit(opt, baseline, shallow, data) is baseline
    event = opt.search_trace_[-1]
    assert event["proposed_objective"] > event["existing_objective"]
    assert event["proposed_accuracy"] < event["existing_accuracy"]
    assert event["outcome"] == "acceptance_policy_rejected"
    assert event["adopted_as_candidate"] is event["final_commit"] is False


@pytest.mark.parametrize("status", [SolverStatus.INFEASIBLE, SolverStatus.TIME_LIMIT])
def test_requested_audit_without_usable_incumbent_preserves_baseline(monkeypatch, status):
    data = _parity_data()
    opt = _optimizer()
    shallow, baseline = _shallow(data, opt.criterion)
    monkeypatch.setattr(
        ExactDepth3Solver, "solve", lambda *a, **k: OCT3Solution(status=status),
    )

    assert _audit(opt, baseline, shallow, data) is baseline
    event = opt.search_trace_[-1]
    assert opt.audit_count_ == 1
    assert event["outcome"] == "no_usable_deeper_incumbent"
    assert event["adopted_as_candidate"] is event["final_commit"] is False


def test_initial_adopted_audit_is_committed():
    data = _parity_data()
    X, y = np.asarray(data.iloc[:, 1:]), np.asarray(data.iloc[:, 0])
    model = RollingOCT(
        solver="direct", depth=3, adaptive_lookahead="policy",
        audit_selector=RecordingSelector(), audit_budget=1,
        audit_min_samples=1, policy_time_limit=10,
        acceptance_policy="objective", trace=True,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    audit = next(event for event in model.search_trace_ if event["event"] == "audit")
    assert audit["adopted_as_candidate"] is audit["final_commit"] is True
    assert audit["final_commit_reason"] == "initial_tree_committed"


@pytest.mark.parametrize("malformed", [
    "missing_objective", "nonfinite_objective", "branch", "leaf", "feature", "class",
])
def test_declared_complete_but_malformed_incumbent_preserves_baseline(monkeypatch, malformed):
    data = _parity_data()
    opt = _optimizer()
    shallow, baseline = _shallow(data, opt.criterion)
    deeper = ExactDepth3Solver(opt.solver_config, opt.criterion).solve(
        data, baseline.features, [0, 1], y_idx=0,
    )
    assert deeper.n_complete_candidates > 0
    if malformed == "missing_objective":
        deeper.objective_value = None
    elif malformed == "nonfinite_objective":
        deeper.objective_value = float("nan")
    elif malformed == "branch":
        deeper.branch_features.pop(4)
    elif malformed == "leaf":
        deeper.leaf_classes.pop(8)
    elif malformed == "feature":
        deeper.branch_features[1] = 4  # Outside the candidate feature set.
    else:
        deeper.leaf_classes[8] = 2  # Outside the observed class set.
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: deeper)

    assert _audit(opt, baseline, shallow, data) is baseline
    event = opt.search_trace_[-1]
    assert opt.audit_count_ == 1
    assert event["outcome"] == "no_usable_deeper_incumbent"
    assert event["adopted_as_candidate"] is event["final_commit"] is False


def test_perfect_audit_with_insufficient_leaf_support_is_rejected(monkeypatch):
    data = _parity_data()
    opt = _optimizer()
    shallow, baseline = _shallow(data, opt.criterion)
    deeper = ExactDepth3Solver(opt.solver_config, opt.criterion).solve(
        data, baseline.features, [0, 1], y_idx=0,
    )
    assert deeper.objective_value == 0
    # The shallow leaves contain 8 rows; the deeper leaves contain only 4.
    opt.solver_config.min_samples_leaf = 5
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: deeper)

    assert _audit(opt, baseline, shallow, data) is baseline
    event = opt.search_trace_[-1]
    assert event["outcome"] == "sample_support_rejected"
    assert event["adopted_as_candidate"] is event["final_commit"] is False


@pytest.mark.parametrize("unusable", ["status", "root", "child", "leaf", "objective"])
def test_unusable_shallow_baseline_never_calls_policy(monkeypatch, unusable):
    data = _parity_data()
    selector = RecordingSelector()
    opt = _optimizer(selector=selector)
    shallow, baseline = _shallow(data, opt.criterion)
    if unusable == "status":
        shallow.status = SolverStatus.INFEASIBLE
    elif unusable == "root":
        shallow.root_feature = None
    elif unusable == "child":
        shallow.left_feature = None
    elif unusable == "leaf":
        shallow.leaf_classes.pop(4)
    else:
        shallow.objective_value = float("nan")
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: pytest.fail("audit ran"))

    assert _audit(opt, baseline, shallow, data) is baseline
    assert selector.states == []
    assert opt.policy_call_count_ == opt.audit_count_ == 0
    assert opt.search_trace_[-1]["reason"] == "unusable_baseline"


def test_shallow_baseline_below_minimum_leaf_support_never_calls_policy(monkeypatch):
    data = _parity_data()
    selector = RecordingSelector()
    opt = _optimizer(selector=selector)
    shallow, baseline = _shallow(data, opt.criterion)
    opt.solver_config.min_samples_leaf = 9  # Each routed shallow leaf has 8 rows.
    monkeypatch.setattr(ExactDepth3Solver, "solve", lambda *a, **k: pytest.fail("audit ran"))

    assert _audit(opt, baseline, shallow, data) is baseline
    assert selector.states == []
    assert opt.policy_call_count_ == opt.audit_count_ == 0
    assert opt.search_trace_[-1]["reason"] == "unusable_baseline"


@pytest.mark.parametrize("criterion, residual, gap", [
    ("gini", .28125, .0625),
    ("misclassification", .1875, .125),
])
def test_policy_residual_and_root_gap_are_normalized_per_node(criterion, residual, gap):
    states = []
    for repeats in [4, 8]:
        X = np.array(list(product((0, 1), repeat=4)) * repeats)
        y = np.tile(np.random.default_rng(5).integers(0, 2, 16), repeats)
        data = pd.DataFrame(np.column_stack([y, X]))
        selector = RecordingSelector("skip")
        opt = _optimizer(criterion, selector=selector)
        shallow, baseline = _shallow(data, opt.criterion)

        _audit(opt, baseline, shallow, data)
        state = selector.states[0]
        assert state.residual == pytest.approx(residual)
        assert state.root_gap == pytest.approx(gap)
        assert state.accuracy == .8125
        assert state.root_costs_complete
        states.append(state)
    assert states[0].residual == states[1].residual
    assert states[0].root_gap == states[1].root_gap


def _rollback_data():
    rows = []
    for values in product((0, 1), repeat=5):
        # Feature 2 is strong, and feature 3 is weak. Every original leaf
        # remains mixed, so rolling optimization visits both root children.
        n_ones = 3 * values[1] + values[2]
        rows.extend([[int(i < n_ones), *values] for i in range(4)])
    return pd.DataFrame(rows)


def _reported_tree(data, branch_features, criterion):
    depth = max(branch_features).bit_length()
    features = list(data.columns[1:])
    tree = DecisionTree(depth, features)
    for node, feature in branch_features.items():
        tree.set_branch_feature(node, feature)
    X = np.asarray(data[features])
    y = np.asarray(data.iloc[:, 0])
    routed = tree.apply(X)
    for leaf in tree.leaf_nodes:
        labels, counts = np.unique(y[routed == leaf], return_counts=True)
        tree.set_leaf_class(leaf, labels[np.argmax(counts)] if len(labels) else 0)
    return tree, score_tree(tree, X, y, criterion)


@pytest.mark.parametrize("acceptance, reason", [
    ("objective", "objective_regression"),
    ("accuracy", "level_accuracy_regression"),
])
def test_locally_adopted_audit_rejected_later_has_no_final_commit(monkeypatch, acceptance, reason):
    data = _rollback_data()
    criterion = get_criterion("gini")
    initial, initial_obj = _reported_tree(data, {1: 1, 2: 2, 3: 2}, criterion)

    def make_oct2(frame, branches):
        tree, objective = _reported_tree(frame, branches, criterion)
        return OCT2Solution(
            SolverStatus.OPTIMAL, root_feature=branches[1],
            left_feature=branches[2], right_feature=branches[3],
            leaf_classes=tree.get_target_class_dict(), objective_value=objective,
            root_costs={branches[1]: objective},
        )

    monkeypatch.setattr(
        PuLPOCT2Solver, "solve",
        lambda *a, **k: make_oct2(data, {1: 1, 2: 2, 3: 2}),
    )

    def solve_inputs(self, inputs):
        return [SubproblemResult(
            inp.parent_node, inp.leaf_ids,
            make_oct2(inp.parent_data, {1: 4, 2: 5, 3: 5}),
            [], False, len(inp.parent_data), inp.parent_data,
        ) for inp in inputs]

    monkeypatch.setattr(RollingOptimizer, "_solve_inputs", solve_inputs)

    def solve_audit(self, frame, *args, **kwargs):
        branches = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 5, 7: 5}
        tree, objective = _reported_tree(frame, branches, criterion)
        return OCT3Solution(
            SolverStatus.OPTIMAL, branch_features=branches,
            leaf_classes=tree.get_target_class_dict(), objective_value=objective,
            n_complete_candidates=1,
        )

    monkeypatch.setattr(ExactDepth3Solver, "solve", solve_audit)

    class AuditRollingNodes(RecordingSelector):
        def select(self, state, *, timeout):
            self.states.append(state)
            return AuditDecision("skip" if state.node == 1 else "audit")

    X, y = np.asarray(data.iloc[:, 1:]), np.asarray(data.iloc[:, 0])
    model = RollingOCT(
        solver="direct", depth=4, criterion="gini", acceptance_policy=acceptance,
        adaptive_lookahead="policy", audit_selector=AuditRollingNodes(),
        audit_budget=2, audit_min_samples=1, policy_time_limit=10, trace=True,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), initial.predict(X))
    assert score_tree(model.tree_, X, y, criterion) == pytest.approx(initial_obj)
    adopted = [event for event in model.search_trace_
               if event["event"] == "audit" and event["adopted_as_candidate"]]
    assert len(adopted) == 2
    for event in adopted:
        assert event["proposed_objective"] < event["existing_objective"]
        assert event["proposed_accuracy"] > event["existing_accuracy"]
        assert event["final_commit"] is False
        assert event["final_commit_reason"] == reason
