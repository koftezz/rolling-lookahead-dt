"""Optional frontier planning; the optimizer retains numerical authority."""
import time
import warnings
from dataclasses import asdict

import numpy as np

from rollotree.rolling.adaptive import _audit_state, _usable_shallow, solution_tree
from rollotree.rolling.policy import AuditDecision, AuditPlan
from rollotree.tree.impurity import MisclassificationCriterion
from rollotree.tree.scoring import score_tree


def plan_frontier(optimizer, results, inputs, tree, features, target_depth, y_idx):
    """Prepare disjoint regions together and return their audit-priority order.

    One plan consumes one policy call. Each decision is consumed once by the
    ordinary audit hook, which rechecks budgets and independently scores trees.
    """
    optimizer._planned_audits = {}
    if (optimizer.adaptive_lookahead != "policy"
            or not callable(getattr(optimizer.audit_selector, "plan", None))
            or optimizer.audit_count_ >= optimizer.audit_budget
            or optimizer.policy_call_count_ >= optimizer.policy_max_calls
            or optimizer.policy_time_ >= optimizer.policy_time_limit
            or optimizer._config_for_solve() is None):
        return results
    started = time.perf_counter()
    input_by_parent = {item.parent_node: item for item in inputs}
    states = []
    for result in results:
        remaining_depth = target_depth - (result.parent_node.bit_length() - 1)
        inp = input_by_parent[result.parent_node]
        if (remaining_depth < 3 or len(inp.parent_data) < optimizer.audit_min_samples
                or result.timed_out or result.skipped or not _usable_shallow(result.sub_solution)):
            continue
        allowance = optimizer.policy_time_limit - optimizer.policy_time_ - (time.perf_counter() - started)
        if allowance <= 0 or optimizer._config_for_solve() is None:
            break
        baseline = solution_tree(result.sub_solution, features)
        X = np.asarray(inp.parent_data[features])
        y = np.asarray(inp.parent_data.iloc[:, y_idx])
        _, support = np.unique(baseline.apply(X), return_counts=True)
        if len(support) != 4 or support.min() < optimizer.solver_config.min_samples_leaf:
            continue
        scale = len(y) if isinstance(optimizer.criterion, MisclassificationCriterion) else 1
        costs = sorted(result.sub_solution.root_costs.values())
        state = _audit_state(
            optimizer, result.sub_solution, y, inp.features, result.parent_node,
            remaining_depth, score_tree(baseline, X, y, optimizer.criterion) / scale,
            float(np.mean(baseline.predict(X) == y)),
            (costs[1] - costs[0]) / scale if len(costs) > 1 else None,
            baseline=baseline, X=X, allowance=allowance,
            current_region_residual=float(score_tree(tree, X, y, optimizer.criterion) / scale),
        )
        state.fingerprint()
        states.append(state)
    if not states:
        optimizer.policy_time_ += time.perf_counter() - started
        return results
    history = tuple({key: event.get(key) for key in (
        "node", "depth", "n_samples", "n_features", "residual",
        "normalized_proposed_objective", "audit_seconds", "outcome",
        "final_commit", "final_commit_reason",
    )} for event in optimizer._audit_history[-16:])
    allowance = optimizer.policy_time_limit - optimizer.policy_time_ - (time.perf_counter() - started)
    remaining = optimizer._remaining_time()
    if remaining is not None:
        allowance = min(allowance, remaining)
    attempted = allowance > 0
    if attempted:
        optimizer.policy_call_count_ += 1
        try:
            plan = optimizer.audit_selector.plan(tuple(states), history=history, timeout=allowance)
            if not isinstance(plan, AuditPlan):
                raise TypeError("plan() must return AuditPlan")
            plan.validate([state.node for state in states])
        except Exception as exc:
            # Exception bodies may include request content or credentials.
            reason = "policy_error:" + type(exc).__name__
            metadata = getattr(optimizer.audit_selector, "__dict__", {})
            model = metadata.get("model")
            version = getattr(type(optimizer.audit_selector), "PROMPT_VERSION", None)
            plan = AuditPlan({state.node: AuditDecision(
                "skip", reason=reason, fallback_reason=reason,
                requested_model=model if isinstance(model, str) else None,
                prompt_version=version if isinstance(version, str) else None,
            ) for state in states})
            if not optimizer._policy_warned:
                warnings.warn("Audit planner failed (" + type(exc).__name__ + "). Skipping audits.",
                              RuntimeWarning, stacklevel=3)
                optimizer._policy_warned = True
    else:
        plan = AuditPlan({state.node: AuditDecision(
            "skip", reason="policy_budget", fallback_reason="policy_budget",
        ) for state in states})
    call = optimizer.policy_call_count_ if attempted else None
    optimizer._planned_audits = {
        state.node: (state, plan.decisions[state.node], call) for state in states
    }

    def order(item):
        index, result = item
        pending = optimizer._planned_audits.get(result.parent_node)
        if pending is None or pending[1].action != "audit":
            return (1, 0, 0, index)
        state, decision, _ = pending
        return (0, -(decision.priority if decision.priority is not None else .5),
                -(state.training_fraction * state.residual), index)

    ordered = [result for _, result in sorted(enumerate(results), key=order)]
    state_hashes = {state.node: state.fingerprint() for state in states}
    decisions = {str(node): asdict(decision) for node, decision in plan.decisions.items()}
    elapsed = time.perf_counter() - started
    optimizer.policy_time_ += elapsed
    optimizer.policy_input_tokens_ += plan.input_tokens or 0
    optimizer._record(
        event="policy_plan", plan_call=call, attempted=attempted,
        nodes=[state.node for state in states],
        execution_order=[result.parent_node for result in ordered],
        decisions=decisions,
        state_hashes=state_hashes,
        policy_seconds=elapsed, input_tokens=plan.input_tokens,
    )
    return ordered
