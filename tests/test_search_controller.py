"""Frontier allocation, provenance and budget safety across real tree fits."""
from dataclasses import replace
from itertools import product
import json
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone

from rollotree import AuditDecision, AuditPlan, AuditExample, JevSearchController, RollingOCT
from rollotree.rolling.adaptive import _leaf_evidence
from rollotree.tree.nodes import DecisionTree
from tests.test_jev_adapter import fake_sdk, state


def cube(parity=3):
    X = np.tile(np.array(list(product((0, 1), repeat=6))), (4, 1))
    y = np.bitwise_xor.reduce(X[:, -parity:], axis=1)
    return X, y


class PreferRight:
    def __init__(self, root_action="skip"):
        self.root_action = root_action
        self.seen = []

    def select(self, state, *, timeout):
        return AuditDecision(self.root_action)

    def plan(self, states, *, history, timeout):
        self.seen.append((states, history))
        return AuditPlan({s.node: AuditDecision("audit", priority=float(s.node == max(
            row.node for row in states))) for s in states}, input_tokens=17)


def fit(controller, **kwargs):
    X, y = cube()
    params = dict(depth=5, solver="direct", adaptive_lookahead="policy",
                  audit_selector=controller, audit_budget=1, audit_min_samples=8,
                  policy_max_calls=2, policy_time_limit=10, total_time_limit=10,
                  random_state=42, acceptance_policy="objective", trace=True)
    params.update(kwargs)
    return RollingOCT(**params).fit(X, y)


def test_frontier_order_controls_scarce_audit_and_counts_request_once():
    controller = PreferRight()
    model = fit(controller)
    states, history = controller.seen[0]
    assert {s.node for s in states} == {2, 3}
    assert history == ()
    assert all(s.schema_version == "3" and s.training_fraction == .5 for s in states)
    assert all(s.current_region_residual == .5 and len(s.leaf_evidence) == 4 for s in states)
    executed = [e for e in model.search_trace_ if e["event"] == "audit" and e["selected"]]
    assert [e["node"] for e in executed] == [3]
    assert model.audit_count_ == 1
    assert model.policy_call_count_ == 2  # root plus one batch, despite two frontier nodes
    assert model.policy_input_tokens_ == 17
    plan = next(e for e in model.search_trace_ if e["event"] == "policy_plan")
    assert plan["execution_order"] == [3, 2]
    assert plan["plan_call"] == 2
    assert model.get_depth() <= 5


@pytest.mark.parametrize("bad", ["missing", "unknown", "invalid", "duplicate_usage", "raise"])
def test_bad_plan_fails_closed_for_whole_frontier_without_secret_text(bad):
    class Broken(PreferRight):
        def plan(self, states, **kwargs):
            if bad == "raise":
                raise RuntimeError("secret-provider-body")
            choices = {s.node: AuditDecision("audit") for s in states}
            if bad == "missing":
                choices.pop(next(iter(choices)))
            elif bad == "unknown":
                choices[999] = AuditDecision("audit")
            elif bad == "invalid":
                choices[next(iter(choices))] = AuditDecision("audit", priority=float("nan"))
            else:
                choices[next(iter(choices))] = AuditDecision("audit", input_tokens=12)
            return AuditPlan(choices)
    with pytest.warns(RuntimeWarning, match="Audit planner failed") as caught:
        model = fit(Broken())
    assert len(caught) == 1
    assert model.audit_count_ == model.policy_input_tokens_ == 0
    assert model.policy_call_count_ == 2
    assert model.policy_fallback_count_ == 2
    assert "secret-provider-body" not in json.dumps(model.search_trace_)
    assert "secret-provider-body" not in str(caught[0].message)
    X, y = cube()
    baseline = RollingOCT(depth=5, solver="direct", random_state=42,
                          acceptance_policy="objective").fit(X, y)
    np.testing.assert_array_equal(model.predict(X), baseline.predict(X))


def test_late_plan_cannot_spend_audit_after_policy_budget():
    class Late(PreferRight):
        def plan(self, states, **kwargs):
            time.sleep(.2)
            return super().plan(states, **kwargs)
    controller = Late()
    model = fit(controller, policy_time_limit=.1)
    assert controller.seen
    assert model.policy_time_ >= .2 and model.audit_count_ == 0
    assert any(e.get("fallback_reason") == "policy_budget" for e in model.search_trace_)


def test_no_frontier_request_after_call_limit():
    controller = PreferRight()
    model = fit(controller, policy_max_calls=1)
    assert controller.seen == []
    assert model.policy_call_count_ == 1 and model.audit_count_ == 0


@pytest.mark.parametrize("trace", [False, True])
def test_history_reports_committed_observations_without_leaking_across_fits(trace):
    class Observe(PreferRight):
        def plan(self, states, **kwargs):
            self.seen.append((states, kwargs["history"]))
            return AuditPlan({s.node: AuditDecision("skip") for s in states})
    controller = Observe(root_action="audit")
    X, y = cube(parity=4)
    model = RollingOCT(depth=6, solver="direct", adaptive_lookahead="policy",
                       audit_selector=controller, audit_budget=2, audit_min_samples=8,
                       policy_max_calls=2, policy_time_limit=10,
                       acceptance_policy="objective", random_state=42, trace=trace)
    for _ in range(2):
        controller.seen.clear()
        model.fit(X, y)
        assert len(controller.seen) == 1
        history = controller.seen[0][1]
        assert len(history) == 1 and history[0]["node"] == 1
        assert history[0]["final_commit"] is True
        assert history[0]["final_commit_reason"] == "initial_tree_committed"
        assert history[0]["normalized_proposed_objective"] == history[0]["residual"] == .5
        assert history[0]["audit_seconds"] > 0
        if not trace:
            assert model.search_trace_ == []


def response_for(nodes):
    choices = {}
    for node in nodes:
        choices["action_" + str(node)] = SimpleNamespace(
            choice="audit", confidence=.7,
            probabilities={"audit": .8, "skip": .1, "insufficient_evidence": .1})
        choices["priority_" + str(node)] = SimpleNamespace(
            choice="high" if node == 3 else "low", confidence=.7,
            probabilities={"high": .8, "medium": .1, "low": .1} if node == 3
            else {"high": .1, "medium": .1, "low": .8})
    return SimpleNamespace(choices=choices, model="jev-1.13.0",
                           usage=SimpleNamespace(input_tokens=321))


def test_sdk_batches_atomic_questions_and_keeps_usage_and_credentials_out_of_decisions(monkeypatch):
    calls, _ = fake_sdk(monkeypatch, response=response_for([2, 3]))
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-test-key")
    example = AuditExample(state(), final_objective_gain=.02, audit_seconds=.4)
    controller = JevSearchController(audit_seconds_estimate=.4, reference_cases=(example,))
    states = [replace(state(), node=n, training_fraction=.5) for n in [2, 3]]
    result = controller.plan(states, history=({"audit_seconds": .3},), timeout=.5)
    assert result.input_tokens == 321
    assert all(d.input_tokens is None and d.action == "audit" for d in result.decisions.values())
    assert result.decisions[3].priority > result.decisions[2].priority
    requests = [value for name, value in calls if name == "request"]
    assert len(requests) == 1
    request = requests[0]
    assert set(request["questions"]) == {"action_2", "priority_2", "action_3", "priority_3"}
    assert request["state"]["recent_audits"] == [{"audit_seconds": .3}]
    assert request["state"]["estimated_audit_seconds"] == .4
    assert request["state"]["development_examples"][0]["material_gain"] is True
    assert request["timeout"] == .5
    assert b"private-test-key" not in pickle.dumps(controller)
    assert clone(controller).get_params() == controller.get_params()


def test_nonfinite_development_evidence_fails_before_sdk(monkeypatch):
    calls, _ = fake_sdk(monkeypatch)
    controller = JevSearchController(reference_cases=(
        AuditExample(state(), final_objective_gain=float("nan"), audit_seconds=1),))
    with pytest.raises(ValueError, match="finite"):
        controller.select(state(), timeout=.5)
    assert calls == []


def test_sdk_context_cap_and_optional_confidence_gate(monkeypatch):
    calls, _ = fake_sdk(monkeypatch, response=response_for([3]))
    states = [replace(state(), node=2, training_fraction=.1),
              replace(state(), node=3, training_fraction=.9)]
    result = JevSearchController(max_frontier_nodes=1, min_confidence=.8).plan(
        states, history=(), timeout=.5)
    assert result.decisions[2].fallback_reason == "frontier_limit"
    assert result.decisions[3].fallback_reason == "low_confidence"
    assert result.decisions[3].proposed_action == "audit"
    assert result.input_tokens == 321
    request = next(value for name, value in calls if name == "request")
    assert set(request["state"]["nodes"]) == {"3"}


def test_real_sdk_multi_question_contract_uses_one_request(monkeypatch):
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        answers = {}
        for name, answer in response_for([2, 3]).choices.items():
            answers[name] = dict(type="choice", choice=answer.choice,
                                 probabilities=answer.probabilities, confidence=answer.confidence)
        return httpx.Response(200, json={"model": "jev-1.13.0",
                              "usage": {"input_tokens": 321}, "answers": answers})
    original = sdk.TypeSafeClient
    monkeypatch.setattr(sdk, "TypeSafeClient", lambda **kwargs: original(
        api_key="fake-test-key", transport=httpx.MockTransport(handle), **kwargs))
    result = JevSearchController().plan([replace(state(), node=n) for n in [2, 3]],
                                        history=(), timeout=.5)
    assert len(requests) == 1 and len(requests[0]["questions"]) == 4
    assert result.input_tokens == 321 and result.decisions[3].priority == 1.0


def test_leaf_split_evidence_obeys_support_and_contains_no_labels():
    X = np.array(list(product((0, 1), repeat=3)))
    y = np.where(X[:, 2], "private-positive", "private-negative")
    tree = DecisionTree(depth=2, features=[0, 1, 2])
    for node, feature in [(1, 0), (2, 1), (3, 1)]:
        tree.set_branch_feature(node, feature)
    for node in range(4, 8):
        tree.set_leaf_class(node, "private-positive")
    allowed = _leaf_evidence(tree, X, y, [0, 1, 2], min_support=1)
    blocked = _leaf_evidence(tree, X, y, [0, 1, 2], min_support=2)
    assert all(row["best_one_split_gini_gain"] == .5 for row in allowed)
    assert all(row["best_one_split_gini_gain"] == 0 for row in blocked)
    assert "private-" not in json.dumps(allowed)
