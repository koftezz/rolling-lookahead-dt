import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

from rollotree import AuditDecision, AuditState, JevAuditSelector, RollingOCT
from rollotree.rolling.adaptive import _select_policy
from rollotree.rolling.optimizer import RollingOptimizer
from rollotree.solver.base import OCT2Solution, SolverConfig, SolverStatus
from rollotree.tree.impurity import get_criterion
from tests.test_adaptive import fixture


class FixedSelector(BaseEstimator):
    def __init__(self, action="audit"):
        self.action = action

    def select(self, state, *, timeout):
        assert timeout > 0
        self.state_ = state
        return AuditDecision(self.action)


def fit(selector, **kwargs):
    X, y, _ = fixture()
    params = dict(solver="direct", depth=4, adaptive_lookahead="policy",
                  audit_selector=selector, trace=True, random_state=42,
                  policy_time_limit=10, audit_budget=1,
                  acceptance_policy="objective")
    params.update(kwargs)
    return RollingOCT(**params).fit(X, y)


def test_audit_matches_fixed_and_skip_matches_baseline():
    X, y, _ = fixture()
    for action, mode in [("audit", "fixed"), ("skip", None), ("insufficient_evidence", None)]:
        model = fit(FixedSelector(action))
        baseline = RollingOCT(solver="direct", depth=4, adaptive_lookahead=mode,
                              audit_budget=1, random_state=42,
                              acceptance_policy="objective").fit(X, y)
        np.testing.assert_array_equal(model.predict(X), baseline.predict(X))
        assert model.audit_count_ == (1 if action == "audit" else 0)
        assert model.policy_call_count_ > 0
        json.dumps(model.search_trace_, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"depth": 2}, {"audit_budget": 0},
                                    {"audit_min_samples": 1000}, {"policy_max_calls": 0}])
def test_ineligible_nodes_never_call_selector(kwargs):
    model = fit(FixedSelector(), **kwargs)
    assert model.policy_call_count_ == model.audit_count_ == 0


def test_ineligible_policy_avoids_summary_scoring(monkeypatch):
    from rollotree.rolling import adaptive

    monkeypatch.setattr(adaptive, "score_tree",
                        lambda *a: pytest.fail("ineligible policy prepared a summary"))
    model = fit(FixedSelector(), depth=2)
    assert model.policy_time_ == 0


def test_call_limit_and_reset_on_refit():
    model = fit(FixedSelector("skip"), policy_max_calls=1)
    assert model.policy_call_count_ == 1
    X, y, _ = fixture()
    model.fit(X, y)
    assert model.policy_call_count_ == 1


def test_errors_fallback_and_do_not_expose_exception_text():
    class Broken:
        def select(self, state, *, timeout):
            raise RuntimeError("sensitive-token-do-not-log")
    with pytest.warns(RuntimeWarning, match="Audit selector failed") as caught:
        model = fit(Broken())
    assert len(caught) == 1
    assert model.audit_count_ == 0
    assert model.policy_fallback_count_ == model.policy_call_count_
    assert "sensitive-token" not in json.dumps(model.search_trace_)
    assert "sensitive-token" not in str(caught[0].message)


def test_custom_selector_error_keeps_trace_metadata_serializable():
    class Broken:
        model = object()
        PROMPT_VERSION = {"private": object()}

        def select(self, state, *, timeout):
            raise RuntimeError("private provider response")

    with pytest.warns(RuntimeWarning, match="Audit selector failed"):
        model = fit(Broken(), policy_max_calls=1)
    json.dumps(model.search_trace_, allow_nan=False)
    event = next(e for e in model.search_trace_ if e["event"] == "policy")
    assert event["requested_model"] is event["prompt_version"] is None
    assert event["fallback_reason"] == "policy_error:RuntimeError"


@pytest.mark.parametrize("decision", [None, AuditDecision("invalid"),
    AuditDecision("audit", confidence=float("nan")),
    AuditDecision("audit", probabilities={"audit": 1.0}),
    AuditDecision("audit", input_tokens=-1)])
def test_bad_policy_result_never_triggers_audit(decision):
    selector = FixedSelector()
    selector.select = lambda *a, **k: decision
    with pytest.warns(RuntimeWarning):
        model = fit(selector)
    assert model.audit_count_ == 0


def optimizer(selector):
    return RollingOptimizer(SolverConfig(solver_name="direct"), get_criterion("gini"),
                            adaptive_lookahead="policy", audit_selector=selector,
                            trace=True, policy_time_limit=1)


def select(opt, status=SolverStatus.OPTIMAL):
    shallow = OCT2Solution(status=status, root_costs={1: .2, 2: .3})
    return _select_policy(opt, shallow, np.array([0, 0, 1, 1]), [1, 2], 1, 3, .2, .5, .1)


def test_partial_root_statistics_are_not_complete():
    selector = FixedSelector()
    select(optimizer(selector), SolverStatus.TIME_LIMIT)
    assert not selector.state_.root_costs_complete
    assert selector.state_.root_gap is None
    assert selector.state_.root_tie_count is None
    assert selector.state_.root_cost_spread is None
    assert selector.state_.shallow_status == "time_limit"


def test_predecision_support_and_solver_statistics():
    selector = FixedSelector("skip")
    model = fit(selector, policy_max_calls=1)
    summary = selector.state_
    assert summary.shallow_status == "optimal"
    assert summary.shallow_seconds >= 0
    assert summary.root_tie_count >= 1
    assert summary.root_cost_spread >= summary.root_gap
    assert sum(summary.leaf_support_proportions) == pytest.approx(1)
    assert len(summary.leaf_support_proportions) == 4
    assert summary.min_leaf_support == model.min_samples_leaf
    assert summary.remaining_policy_seconds > 0
    assert summary.schema_version == "2"


def test_custom_skip_reason_is_not_counted_as_fallback():
    selector = FixedSelector("skip")
    selector.select = lambda *a, **k: AuditDecision("skip", reason="low_expected_gain")
    model = fit(selector)
    assert model.policy_call_count_ > 0
    assert model.policy_fallback_count_ == 0
    assert all(e["fallback_reason"] is None for e in model.search_trace_
               if e["event"] == "policy")


def test_expired_fit_budget_after_policy_prevents_audit():
    selector = FixedSelector()
    opt = optimizer(selector)
    def expire(*args, **kwargs):
        opt._deadline = 0
        return AuditDecision("audit")
    selector.select = expire
    assert select(opt) == (False, "fit_budget")
    assert opt.search_trace_[-1]["action"] == "audit"
    assert opt.search_trace_[-1]["effective_action"] == "skip"


def test_policy_time_includes_state_and_rejects_late_answer(monkeypatch):
    from rollotree.rolling import adaptive
    ticks = iter([0, .2, 1.2])
    monkeypatch.setattr(adaptive, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    opt = optimizer(FixedSelector())
    assert select(opt) == (False, "policy_budget")
    assert opt.policy_time_ == 1.2


def test_clone_save_load_predict_without_sdk_or_key(monkeypatch, tmp_path):
    selector = JevAuditSelector()
    model = clone(RollingOCT(solver="direct", adaptive_lookahead="policy", audit_selector=selector))
    assert model.audit_selector is not selector
    model.set_params(audit_selector__min_confidence=.7)
    monkeypatch.setattr(JevAuditSelector, "select", lambda *a, **k: AuditDecision("skip"))
    model = fit(model.audit_selector)
    X, _, _ = fixture()
    expected = model.predict(X)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.setattr(JevAuditSelector, "select", lambda *a, **k: pytest.fail("prediction made a call"))
    model.save(tmp_path / "model.joblib")
    restored = RollingOCT.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(restored.predict(X), expected)
    assert not any("key" in k or "client" in k for k in vars(restored.audit_selector))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(X), expected)


@pytest.mark.parametrize("params", [
    {"adaptive_lookahead": "policy"},
    {"audit_selector": FixedSelector()},
    {"policy_time_limit": 0}, {"policy_max_calls": -1},
    {"adaptive_lookahead": "policy", "audit_selector": JevAuditSelector(timeout=0)},
    {"adaptive_lookahead": "policy", "audit_selector": JevAuditSelector(min_confidence=2)},
])
def test_invalid_parameters(params):
    X, y, _ = fixture()
    with pytest.raises(ValueError):
        RollingOCT(solver="direct", **params).fit(X, y)


def state():
    return AuditState(1, 0, 3, 64, 4, (.5, .5), "gini", .3, .7, .01, True, 4, 5., 2)


def fake_sdk(monkeypatch, choice="audit", confidence=.95):
    observed = {}
    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            observed["closed"] = True
        def system_one(self, **kwargs):
            observed["request"] = kwargs
            return SimpleNamespace(
                choices={"audit": SimpleNamespace(choice=choice, confidence=confidence,
                    probabilities={"audit": .96, "skip": .02, "insufficient_evidence": .02})},
                model="jev-1.13.0", usage=SimpleNamespace(input_tokens=123))
    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(
        TypeSafeClient=Client, Choice=lambda **k: k, RetryPolicy=lambda **k: k))
    return observed


def test_sdk_adapter_request_and_provenance(monkeypatch):
    observed = fake_sdk(monkeypatch)
    decision = JevAuditSelector().select(state(), timeout=.2)
    assert decision.action == "audit"
    assert decision.input_tokens == 123
    assert observed["retry"] == {"max_retries": 0}
    assert observed["timeout"] == observed["request"]["timeout"] == .2
    assert observed["request"]["state"]["class_proportions"] == [.5, .5]
    assert observed["closed"]
    assert "api_key" not in observed


def test_low_confidence_preserves_proposed_action(monkeypatch):
    fake_sdk(monkeypatch, confidence=.2)
    result = JevAuditSelector().select(state(), timeout=1)
    assert (result.action, result.proposed_action, result.reason) == ("skip", "audit", "low_confidence")


def test_missing_sdk_falls_back_without_network(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.warns(RuntimeWarning, match="ModuleNotFoundError"):
        model = fit(JevAuditSelector(), policy_max_calls=1)
    assert model.audit_count_ == 0 and model.policy_fallback_count_ == 1


def test_missing_key_falls_back(monkeypatch):
    pytest.importorskip("typesafe_sdk")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.warns(RuntimeWarning, match="Audit selector failed"):
        model = fit(JevAuditSelector(), policy_max_calls=1)
    assert model.audit_count_ == 0 and model.policy_fallback_count_ == 1


@pytest.mark.parametrize("status", [200, 401, 429])
def test_real_sdk_with_mock_http_transport(monkeypatch, status):
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "mock provider failure"})
        return httpx.Response(200, json={
            "model": "jev-1.13.0", "usage": {"input_tokens": 123},
            "answers": {"audit": {"type": "choice", "choice": "audit", "confidence": .95,
                "probabilities": {"audit": .96, "skip": .02, "insufficient_evidence": .02}}},
        })
    real_client = sdk.TypeSafeClient
    monkeypatch.setattr(sdk, "TypeSafeClient", lambda **kw: real_client(
        api_key="fake-test-key", transport=httpx.MockTransport(handle), **kw))
    if status == 200:
        decision = JevAuditSelector().select(state(), timeout=.5)
        assert decision.action == "audit" and decision.input_tokens == 123
    else:
        with pytest.warns(RuntimeWarning, match="Audit selector failed"):
            model = fit(JevAuditSelector(), policy_max_calls=1)
        assert model.audit_count_ == 0 and model.policy_fallback_count_ == 1
    assert len(requests) == 1  # HTTP errors must not trigger hidden SDK retries.
    assert requests[0]["model"] == "jev-1.13.0"
    assert requests[0]["questions"]["audit"]["type"] == "choice"
