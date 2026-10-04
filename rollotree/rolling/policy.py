"""Optional audit policies. No policy can bypass numerical acceptance."""
from dataclasses import asdict, dataclass, replace
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
    training_fraction: Optional[float] = None
    current_region_residual: Optional[float] = None
    leaf_evidence: Optional[Tuple[dict, ...]] = None

    def to_dict(self):
        result = asdict(self)
        result["class_proportions"] = list(self.class_proportions)
        if self.leaf_support_proportions is not None:
            result["leaf_support_proportions"] = list(self.leaf_support_proportions)
        for name in ("training_fraction", "current_region_residual", "leaf_evidence"):
            if result[name] is None:
                del result[name]
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
    priority: Optional[float] = None

    def validate(self):
        actions = {"audit", "skip", "insufficient_evidence"}
        if self.action not in actions or (
            self.proposed_action is not None and self.proposed_action not in actions
        ):
            raise ValueError("Invalid audit policy action")
        if self.confidence is not None:
            _probability(self.confidence)
        if self.priority is not None:
            _probability(self.priority)
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


@dataclass(frozen=True)
class AuditPlan:
    """One batched invocation, with one decision for every supplied node.

    Token usage belongs to the plan, not each decision. Priorities are ordinal
    routing scores, never predicted numerical gains or correctness guarantees.
    """

    decisions: dict
    input_tokens: Optional[int] = None

    def validate(self, nodes):
        if (not isinstance(self.decisions, dict)
                or any(type(node) is not int for node in self.decisions)
                or set(self.decisions) != set(nodes)):
            raise ValueError("Audit plan must cover exactly the supplied nodes")
        for decision in self.decisions.values():
            if not isinstance(decision, AuditDecision):
                raise TypeError("Audit plan requires AuditDecision values")
            decision.validate()
            if decision.input_tokens is not None:
                raise ValueError("Batched token usage belongs to the plan")
        if self.input_tokens is not None and (
                type(self.input_tokens) is not int or self.input_tokens < 0):
            raise ValueError("Invalid plan input token count")
        return self


@dataclass(frozen=True)
class AuditExample:
    """Offline paired-continuation evidence, frozen before evaluation cases.

    The gain is the completed whole-tree normalized training objective without
    this audit minus the objective with it; local incumbent regret is not a
    substitute. No test labels or rows belong in an example.
    """

    state: AuditState
    final_objective_gain: float
    audit_seconds: float

    def to_dict(self):
        if not isinstance(self.state, AuditState):
            raise TypeError("AuditExample requires an AuditState")
        for value in (self.final_objective_gain, self.audit_seconds):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Audit example measurements must be finite numbers")
        if not -1 <= self.final_objective_gain <= 1 or self.audit_seconds <= 0:
            raise ValueError("Invalid normalized gain or audit duration")
        result = dict(state=self.state.to_dict(), final_objective_gain=self.final_objective_gain,
                      audit_seconds=self.audit_seconds)
        json.dumps(result, allow_nan=False)
        return result


class JevSearchController(JevAuditSelector):
    """Experimental comparison of competing audits in a rolling frontier.

    ``plan`` evaluates benefit and relative priority as separate typed
    questions in one request. State and history are supplied by the optimizer;
    no client, credentials, or cross-fit history live on this estimator.
    ``audit_seconds_estimate`` is an optional measured deployment estimate.
    With no estimate, recorded audit durations are supplied as observations,
    not extrapolated into a fabricated cost prediction.

    Confidence gating is disabled by default for this experimental controller.
    Calibrate ``min_confidence`` on independent development cases before use.
    """

    PROMPT_VERSION = "rollotree-search-v1"

    def __init__(self, model="jev-1.13.0", timeout=2.0,
                 min_confidence=0.0, fallback="skip",
                 audit_seconds_estimate=None, max_frontier_nodes=16,
                 reference_cases=None, min_objective_gain=.01):
        super().__init__(model, timeout, min_confidence, fallback)
        self.audit_seconds_estimate = audit_seconds_estimate
        self.max_frontier_nodes = max_frontier_nodes
        self.reference_cases = reference_cases
        self.min_objective_gain = min_objective_gain

    def validate(self):
        super().validate()
        estimate = self.audit_seconds_estimate
        if estimate is not None and (isinstance(estimate, bool)
                or not isinstance(estimate, Real) or not math.isfinite(estimate)
                or estimate <= 0):
            raise ValueError("audit_seconds_estimate must be positive and finite")
        if type(self.max_frontier_nodes) is not int or not 1 <= self.max_frontier_nodes <= 16:
            raise ValueError("max_frontier_nodes must be an integer from 1 to 16")
        _probability(self.min_objective_gain)
        if self.reference_cases is not None:
            if not isinstance(self.reference_cases, (tuple, list)) or len(self.reference_cases) > 16:
                raise ValueError("reference_cases must contain at most 16 AuditExample records")
            for example in self.reference_cases:
                if not isinstance(example, AuditExample):
                    raise TypeError("reference_cases requires AuditExample records")
                example.to_dict()

    def select(self, state, *, timeout):
        plan = self.plan((state,), history=(), timeout=timeout)
        return replace(plan.decisions[state.node], input_tokens=plan.input_tokens)

    def plan(self, states, *, history, timeout):
        self.validate()
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout):
            raise ValueError("request timeout must be finite")
        states = tuple(states)
        if not states or any(not isinstance(state, AuditState) for state in states):
            raise TypeError("Search controller requires nonempty AuditState input")
        if len({state.node for state in states}) != len(states):
            raise ValueError("Duplicate frontier node")
        limit = min(float(self.timeout), float(timeout))
        if limit <= 0:
            return AuditPlan({state.node: AuditDecision(
                "skip", reason="policy_budget", fallback_reason="policy_budget",
                requested_model=self.model, prompt_version=self.PROMPT_VERSION,
            ) for state in states})
        # Bound payload size before importing the SDK. Stable local ordering is
        # also the documented comparator for nodes omitted from this request.
        considered = sorted(states, key=lambda s: (
            -(s.training_fraction if s.training_fraction is not None else 1) * s.residual,
            s.node,
        ))[:self.max_frontier_nodes]
        payload = {
            "nodes": {str(state.node): state.to_dict() for state in considered},
            "recent_audits": list(history)[-16:],
            "estimated_audit_seconds": self.audit_seconds_estimate,
            "material_final_gain_floor": self.min_objective_gain,
            "development_examples": [dict(
                **example.to_dict(), material_gain=example.final_objective_gain >= self.min_objective_gain,
            ) for example in (self.reference_cases or ())],
            "objective": "final whole-training-set objective reduction under the fit budget",
        }
        for state in considered:
            payload["nodes"][str(state.node)]["computed_context"] = {
                "residual_band": "high" if state.residual >= .2 else "moderate" if state.residual >= .05 else "low",
                "root_gap_band": ("unknown" if state.root_gap is None else
                                  "small" if state.root_gap <= .02 else "large"),
                "affordable_with_estimate": (None if self.audit_seconds_estimate is None or state.remaining_seconds is None
                                             else self.audit_seconds_estimate <= .5 * state.remaining_seconds),
            }
        json.dumps(payload, allow_nan=False)
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient
        questions = {}
        for state in considered:
            key = str(state.node)
            questions["action_" + key] = Choice(
                instructions=(
                    "Assess whether node " + key + " merits a depth-3 audit. "
                    "The target is at least material_final_gain_floor reduction in the final "
                    "whole-tree normalized training objective. Use comparable development examples "
                    "as empirical evidence when provided. Material gain examples can justify an "
                    "affordable audit despite uncertainty, but do not guarantee transfer. "
                    "Use its shallow baseline, leaf evidence, current committed-region loss, "
                    "training fraction, remaining depth and budgets, and recent audit observations. "
                    "A leaf split gain only measures a one-split training improvement; it does not "
                    "prove a useful deeper interaction. History reports local gains and whether the "
                    "region is still committed, not a causal final-tree gain. A small root gap alone "
                    "is not evidence of improvement. Do not invent costs or hidden interactions. "
                    "The solver independently constructs and validates every tree."
                ),
                criteria={
                    "audit": "Supplied evidence supports a useful deeper search within the remaining budget.",
                    "skip": "Little remaining benefit or poor use of the available compute.",
                    "insufficient_evidence": "The evidence cannot distinguish useful search from wasted compute.",
                },
            )
            questions["priority_" + key] = Choice(
                instructions=(
                    "Compare node " + key + " with the other supplied nodes for allocation of "
                    "the limited remaining audits. Judge relative expected contribution to the "
                    "whole training objective, accounting for node weight, remaining improvement, "
                    "support and available cost evidence. This is ordinal priority, not a numerical "
                    "gain estimate. When evidence does not distinguish nodes, use medium."
                ),
                criteria={"high": "Prioritize this node over most alternatives.",
                          "medium": "Comparable to alternatives or relative value is uncertain.",
                          "low": "Other nodes offer stronger evidence of useful improvement."},
            )
        with TypeSafeClient(model=self.model, timeout=limit,
                            retry=RetryPolicy(max_retries=0)) as client:
            response = client.system_one(state=payload, questions=questions, timeout=limit)
        if not isinstance(response.model, str) or not response.model.strip():
            raise ValueError("Jev response lacks a model ID")
        decisions = {}
        for state in states:
            if state not in considered:
                decisions[state.node] = AuditDecision(
                    "skip", reason="frontier_limit", fallback_reason="frontier_limit",
                    requested_model=self.model, prompt_version=self.PROMPT_VERSION)
                continue
            answer = response.choices["action_" + str(state.node)]
            priority = response.choices["priority_" + str(state.node)]
            if set(priority.probabilities) != {"high", "medium", "low"}:
                raise ValueError("Invalid priority distribution")
            for value in priority.probabilities.values():
                _probability(value)
            if not math.isclose(sum(priority.probabilities.values()), 1, abs_tol=1e-5):
                raise ValueError("Priority probabilities must sum to one")
            _probability(priority.confidence)
            decision = AuditDecision(
                action=answer.choice, proposed_action=answer.choice,
                probabilities=dict(answer.probabilities), confidence=answer.confidence,
                priority={"high": 1.0, "medium": .5, "low": 0.0}[priority.choice],
                model=response.model, requested_model=self.model,
                prompt_version=self.PROMPT_VERSION,
            ).validate()
            if decision.confidence is None:
                raise ValueError("Jev response lacks confidence")
            if decision.confidence < self.min_confidence:
                decision = replace(decision, action="skip", reason="low_confidence",
                                   fallback_reason="low_confidence")
            decisions[state.node] = decision
        return AuditPlan(decisions, getattr(getattr(response, "usage", None), "input_tokens", None)).validate(
            [state.node for state in states])
