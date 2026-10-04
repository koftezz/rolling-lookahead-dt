"""Optional audit policies. No policy can bypass numerical acceptance."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Optional, Tuple

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class AuditState:
    """Pre-audit aggregate statistics only; no observations or class names."""

    node: int
    depth: int
    remaining_depth: int
    n_samples: int
    n_features: int
    class_proportions: Tuple[float, ...]
    criterion: str
    residual: float
    accuracy: float
    root_gap: Optional[float]
    root_costs_complete: bool
    n_feasible_roots_observed: int
    remaining_seconds: Optional[float]
    remaining_audits: int
    schema_version: str = "2"
    shallow_status: Optional[str] = None
    shallow_seconds: Optional[float] = None
    root_tie_count: Optional[int] = None
    root_cost_spread: Optional[float] = None
    leaf_support_proportions: Optional[Tuple[float, ...]] = None
    min_leaf_support: Optional[int] = None
    near_min_support: Optional[bool] = None
    remaining_policy_seconds: Optional[float] = None

    def to_dict(self):
        result = asdict(self)
        result["class_proportions"] = list(self.class_proportions)
        if self.leaf_support_proportions is not None:
            result["leaf_support_proportions"] = list(self.leaf_support_proportions)
        return result

    def fingerprint(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class AuditDecision:
    """Result of ``select(state, *, timeout)``; abstention retains the baseline.

    Custom selectors must honor the supplied cooperative timeout, avoid raw
    data/client objects in results, and return one of the three actions below.
    """

    action: str
    reason: str = "policy"
    proposed_action: Optional[str] = None
    probabilities: Optional[dict] = None
    confidence: Optional[float] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    prompt_version: Optional[str] = None
    requested_model: Optional[str] = None
    cache_status: str = "disabled"
    fallback_reason: Optional[str] = None

    def validate(self):
        actions = {"audit", "skip", "insufficient_evidence"}
        if self.action not in actions or (
            self.proposed_action is not None and self.proposed_action not in actions
        ):
            raise ValueError("Invalid audit policy action")
        if self.confidence is not None:
            _probability(self.confidence)
        if self.probabilities is not None:
            if set(self.probabilities) != actions:
                raise ValueError("Policy probabilities must cover all three actions")
            for value in self.probabilities.values():
                _probability(value)
            if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-5):
                raise ValueError("Policy probabilities must sum to one")
        if self.input_tokens is not None and (
            type(self.input_tokens) is not int or self.input_tokens < 0
        ):
            raise ValueError("Invalid input token count")
        for value in (self.reason, self.model, self.requested_model,
                      self.prompt_version, self.cache_status, self.fallback_reason):
            if value is not None and not isinstance(value, str):
                raise ValueError("Policy metadata must be strings")
        json.dumps(asdict(self), allow_nan=False)
        return self


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite probability between zero and one")


class JevAuditSelector(BaseEstimator):
    """Experimental, stateless TypeSafe selector with lazy SDK loading.

    The API key is read from TYPESAFE_API_KEY by the SDK at request time.
    Neither the key nor a client is stored on this sklearn-cloneable object.
    ``min_confidence`` is a routing threshold, not an accuracy guarantee.
    The only supported fallback is ``skip``. HTTP timeouts are cooperative,
    per-operation limits; they are not hard end-to-end deadlines.
    """

    PROMPT_VERSION = "rollotree-audit-v1"

    def __init__(self, model="jev-1.13.0", timeout=0.5,
                 min_confidence=0.8, fallback="skip"):
        self.model = model
        self.timeout = timeout
        self.min_confidence = min_confidence
        self.fallback = fallback

    def validate(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, Real) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("selector timeout must be positive and finite")
        _probability(self.min_confidence)
        if self.fallback != "skip":
            raise ValueError("Only fallback='skip' is supported")

    def select(self, state, *, timeout):
        self.validate()
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout):
            raise ValueError("request timeout must be finite")
        limit = min(float(self.timeout), float(timeout))
        if limit <= 0:
            return AuditDecision("skip", reason="policy_budget",
                                 requested_model=self.model,
                                 prompt_version=self.PROMPT_VERSION,
                                 fallback_reason="policy_budget")
        if not isinstance(state, AuditState):
            raise TypeError("Jev requires an AuditState")
        payload = state.to_dict()
        # Fail locally before loading credentials or constructing an HTTP client.
        json.dumps(payload, allow_nan=False)
        # Runtime-only import keeps the base install and saved predictors local.
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient
        question = Choice(
            instructions=(
                "Choose whether to spend one depth-3 audit on this binary "
                "classification-tree node. The goal is to improve the FINAL "
                "rolling tree's training objective within the remaining fit "
                "budget. A depth-2 baseline already exists. Local root regret "
                "or a small root gap does not prove a final improvement. "
                "These are precomputed aggregate statistics; do not infer "
                "unobserved feature interactions or invent audit costs. "
                "Choose insufficient_evidence when the summary cannot "
                "support a useful forecast. Your answer does not certify "
                "optimality or held-out accuracy."
            ),
            criteria={
                "audit": "Evidence supports spending compute on deeper search here.",
                "skip": "Evidence supports keeping the shallow baseline and saving compute.",
                "insufficient_evidence": "The summary is insufficient to forecast audit benefit.",
            },
        )
        with TypeSafeClient(model=self.model, timeout=limit,
                            retry=RetryPolicy(max_retries=0)) as client:
            response = client.system_one(
                state=payload, questions={"audit": question}, timeout=limit,
            )
        answer = response.choices["audit"]
        if not isinstance(response.model, str) or not response.model.strip():
            raise ValueError("Jev response lacks a model ID")
        decision = AuditDecision(
            action=answer.choice, proposed_action=answer.choice,
            probabilities=dict(answer.probabilities), confidence=answer.confidence,
            model=response.model, prompt_version=self.PROMPT_VERSION,
            requested_model=self.model,
            input_tokens=getattr(getattr(response, "usage", None), "input_tokens", None),
        ).validate()
        if decision.confidence is None:
            raise ValueError("Jev response lacks confidence")
        if decision.confidence < self.min_confidence:
            return AuditDecision(**{
                **asdict(decision), "action": "skip", "reason": "low_confidence",
                "fallback_reason": "low_confidence",
            })
        return decision
