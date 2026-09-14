import json
import numpy as np
import pytest
from rollotree import RollingOCT
from rollotree.tree.scoring import score_tree
from rollotree.tree.impurity import get_criterion


@pytest.mark.parametrize("policy", ["accuracy", "objective"])
@pytest.mark.parametrize("criterion", ["gini", "misclassification"])
def test_trace_is_observational_and_acceptance(policy, criterion):
    rng = np.random.default_rng(6)
    X, y = rng.integers(0, 2, (80, 5)), rng.integers(0, 3, 80)
    config = dict(depth=4, solver="direct", criterion=criterion, acceptance_policy=policy)
    plain = RollingOCT(**config).fit(X, y)
    traced = RollingOCT(**config, trace=True).fit(X, y)
    assert np.array_equal(plain.predict(X), traced.predict(X))
    assert plain.search_trace_ == []
    json.dumps(traced.search_trace_)
    assert traced.search_trace_[-1]["event"] == "stop"
    if policy == "objective":
        for event in traced.search_trace_:
            if event["event"] == "update" and event["accepted"]:
                assert event["proposed_objective"] <= event["existing_objective"] + 1e-10
        shallow = RollingOCT(solver="direct", criterion=criterion).fit(X, y)
        assert score_tree(traced.tree_, X, y, get_criterion(criterion)) <= score_tree(shallow.tree_, X, y, get_criterion(criterion)) + 1e-10
    else:
        values = [v.training_accuracy for v in traced.depth_results_.values()]
        assert values == sorted(values)


def test_invalid_policy():
    with pytest.raises(ValueError, match="acceptance_policy"):
        RollingOCT(acceptance_policy="bad").fit([[0], [1]], [0, 1])


def test_gini_can_improve_while_accuracy_worsens():
    rng = np.random.default_rng(142)
    X, y = rng.integers(0, 2, (150, 7)), rng.integers(0, 2, 150)
    model = RollingOCT(depth=4, solver="direct", acceptance_policy="objective", trace=True).fit(X, y)
    assert any(e["event"] == "update" and e["accepted"]
               and e["proposed_accuracy"] < e["existing_accuracy"]
               and e["proposed_objective"] < e["existing_objective"]
               for e in model.search_trace_)
