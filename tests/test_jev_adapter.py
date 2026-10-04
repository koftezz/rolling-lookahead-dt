"""Optional SDK contract and request safety, using local fake transports only."""
from dataclasses import asdict, replace
import json
import pickle
import sys
from types import SimpleNamespace

import pytest
from sklearn.base import clone

from rollotree import AuditState, JevAuditSelector


def state():
    return AuditState(1, 0, 3, 64, 4, (.5, .5), "gini", .3, .7,
                      .01, True, 4, 5., 2)


def fake_sdk(monkeypatch, response=None, error=None):
    if response is None:
        response = SimpleNamespace(
            choices={"audit": SimpleNamespace(
                choice="audit", confidence=.95,
                probabilities={"audit": .96, "skip": .02,
                               "insufficient_evidence": .02})},
            model="jev-1.13.0", usage=SimpleNamespace(input_tokens=123))
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append(("closed", None))

        def system_one(self, **kwargs):
            calls.append(("request", kwargs))
            if error is not None:
                raise error
            return response

    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(
        TypeSafeClient=Client, Choice=lambda **kw: kw,
        RetryPolicy=lambda **kw: kw))
    return calls, response


def test_model_alias_retains_requested_and_resolved_versions(monkeypatch):
    calls, _ = fake_sdk(monkeypatch)
    selector = JevAuditSelector(model="jev-latest", timeout=.4)
    result = selector.select(state(), timeout=.1)
    assert result.requested_model == "jev-latest"
    assert result.model == "jev-1.13.0"
    assert result.cache_status == "disabled"
    assert result.fallback_reason is None
    assert calls[0][1] == {
        "model": "jev-latest", "timeout": .1, "retry": {"max_retries": 0}}
    assert calls[1][1]["timeout"] == .1
    assert calls[-1][0] == "closed"
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("timeout", [0, -1])
def test_expired_allowance_does_not_load_optional_sdk(monkeypatch, timeout):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    decision = JevAuditSelector().select(state(), timeout=timeout)
    assert decision.action == "skip"
    assert decision.fallback_reason == "policy_budget"
    assert decision.requested_model == "jev-1.13.0"


@pytest.mark.parametrize("timeout", [True, None, float("nan"), float("inf"), "1"])
def test_invalid_allowance_never_constructs_client(monkeypatch, timeout):
    calls, _ = fake_sdk(monkeypatch)
    with pytest.raises(ValueError, match="timeout"):
        JevAuditSelector().select(state(), timeout=timeout)
    assert calls == []


def test_nonfinite_state_is_rejected_before_credentials(monkeypatch):
    calls, _ = fake_sdk(monkeypatch)
    with pytest.raises(ValueError):
        JevAuditSelector().select(replace(state(), residual=float("nan")), timeout=.1)
    assert calls == []


@pytest.mark.parametrize("model", [None, "", " ", 7])
def test_missing_returned_model_cannot_claim_provenance(monkeypatch, model):
    calls, response = fake_sdk(monkeypatch)
    response.model = model
    with pytest.raises(ValueError, match="model ID"):
        JevAuditSelector().select(state(), timeout=.1)
    assert calls[-1][0] == "closed"


@pytest.mark.parametrize("field,value", [
    ("choice", "unknown"),
    ("confidence", None),
    ("confidence", float("nan")),
    ("confidence", True),
    ("probabilities", {"audit": 1.0}),
    ("probabilities", {"audit": .9, "skip": .9, "insufficient_evidence": .9}),
    ("probabilities", {"audit": True, "skip": 0., "insufficient_evidence": 0.}),
])
def test_malformed_choice_is_unusable(monkeypatch, field, value):
    calls, response = fake_sdk(monkeypatch)
    setattr(response.choices["audit"], field, value)
    with pytest.raises(ValueError):
        JevAuditSelector().select(state(), timeout=.1)
    assert calls[-1][0] == "closed"


def test_low_confidence_preserves_distribution_and_usage(monkeypatch):
    _, response = fake_sdk(monkeypatch)
    response.choices["audit"].confidence = .1
    decision = JevAuditSelector().select(state(), timeout=.1)
    assert decision.action == "skip" and decision.proposed_action == "audit"
    assert decision.reason == decision.fallback_reason == "low_confidence"
    assert decision.input_tokens == 123
    assert decision.probabilities["audit"] == .96
    assert decision.requested_model == decision.model == "jev-1.13.0"


def test_request_failure_closes_client_without_persisting_runtime(monkeypatch):
    calls, _ = fake_sdk(monkeypatch, error=TimeoutError("secret-placeholder"))
    selector = JevAuditSelector()
    before = dict(vars(selector))
    with pytest.raises(TimeoutError):
        selector.select(state(), timeout=.1)
    assert calls[-1][0] == "closed"
    assert vars(selector) == before


def test_completed_request_leaves_clone_and_pickle_network_free(monkeypatch):
    fake_sdk(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", "do-not-persist-test-secret")
    selector = JevAuditSelector()
    selector.select(state(), timeout=.1)
    serialized = pickle.dumps(selector)
    assert b"do-not-persist-test-secret" not in serialized
    for saved in (clone(selector), pickle.loads(serialized)):
        assert saved.get_params() == selector.get_params()
        assert set(vars(saved)) == {"model", "timeout", "min_confidence", "fallback"}


@pytest.mark.parametrize("failure", ["timeout", "malformed"])
def test_actual_sdk_failure_uses_one_http_attempt(monkeypatch, failure):
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if failure == "timeout":
            raise httpx.ReadTimeout("mock timeout", request=request)
        return httpx.Response(200, json={
            "model": "jev-1.13.0", "usage": {"input_tokens": 1},
            "answers": {"audit": {"type": "choice", "choice": "audit",
                                   "confidence": .9,
                                   "probabilities": {"audit": 1.0}}}})

    original = sdk.TypeSafeClient
    monkeypatch.setattr(sdk, "TypeSafeClient", lambda **kwargs: original(
        api_key="fake-test-key", transport=httpx.MockTransport(handle), **kwargs))
    expected = sdk.TypeSafeAPITimeoutError if failure == "timeout" else ValueError
    with pytest.raises(expected):
        JevAuditSelector().select(state(), timeout=.1)
    assert len(requests) == 1
    assert requests[0]["model"] == "jev-1.13.0"
