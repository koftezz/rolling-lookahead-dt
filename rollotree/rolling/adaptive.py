"""Transparent, opt-in deeper-audit heuristics. No generalization certificate."""
import time
import warnings
from dataclasses import asdict
from numbers import Real
import numpy as np
from rollotree.solver.base import SolverStatus
from rollotree.solver.depth3 import ExactDepth3Solver
from rollotree.tree.nodes import DecisionTree
from rollotree.tree.scoring import score_tree
from rollotree.tree.impurity import MisclassificationCriterion
from rollotree.rolling.policy import AuditState, AuditDecision


def _select_policy(optimizer, shallow, y, candidates, node, remaining_depth,
                   residual, accuracy, gap, *, baseline=None, X=None, started=None):
    """Bound policy work and recheck time before allowing an audit."""
    selection_started = time.perf_counter()
    started = selection_started if started is None else started
    remaining = optimizer._remaining_time()
    allowance = (optimizer.policy_time_limit - optimizer.policy_time_
                 - (selection_started - started))
    if (allowance <= 0 or (remaining is not None and remaining <= 0)
            or optimizer.policy_call_count_ >= optimizer.policy_max_calls):
        optimizer.policy_time_ += time.perf_counter() - started
        return False, "fit_budget" if remaining is not None and remaining <= 0 else "policy_budget"
    complete = shallow.status == SolverStatus.OPTIMAL
    _, counts = np.unique(y, return_counts=True)
    scale = len(y) if isinstance(optimizer.criterion, MisclassificationCriterion) else 1
    costs = sorted(shallow.root_costs.values())
    supports = None
    if baseline is not None:
        _, supports = np.unique(baseline.apply(X), return_counts=True)
    shallow_times = (shallow.coefficient_time, shallow.assembly_time, shallow.runtime)
    state = AuditState(
        node=int(node), depth=node.bit_length()-1, remaining_depth=int(remaining_depth),
        n_samples=len(y), n_features=len(candidates),
        class_proportions=tuple(sorted((counts / len(y)).tolist(), reverse=True)),
        criterion=("misclassification" if isinstance(optimizer.criterion, MisclassificationCriterion) else "gini"),
        residual=float(residual), accuracy=float(accuracy),
        root_gap=float(gap) if gap is not None and complete else None,
        root_costs_complete=complete,
        n_feasible_roots_observed=len(shallow.root_costs),
        remaining_seconds=remaining,
        remaining_audits=int(optimizer.audit_budget-optimizer.audit_count_),
        shallow_status=shallow.status.value,
        shallow_seconds=(sum(t for t in shallow_times if t is not None)
                         if any(t is not None for t in shallow_times) else None),
        root_tie_count=(int(sum(abs(cost-costs[0]) <= 1e-10 * scale for cost in costs))
                        if complete and costs else None),
        root_cost_spread=(float((costs[-1]-costs[0])/scale)
                          if complete and costs else None),
        leaf_support_proportions=(tuple(sorted((supports / len(y)).tolist()))
                                  if supports is not None else None),
        min_leaf_support=optimizer.solver_config.min_samples_leaf,
        near_min_support=(bool(supports.min() <= 2 * optimizer.solver_config.min_samples_leaf)
                          if supports is not None else None),
        remaining_policy_seconds=allowance,
    )
    state_hash = state.fingerprint()
    # State extraction is part of the policy's allowance too.
    allowance -= time.perf_counter() - selection_started
    remaining = optimizer._remaining_time()
    if remaining is not None:
        allowance = min(allowance, remaining)
    if allowance <= 0:
        optimizer.policy_time_ += time.perf_counter() - started
        return False, "fit_budget" if optimizer._config_for_solve() is None else "policy_budget"
    optimizer.policy_call_count_ += 1
    try:
        decision = optimizer.audit_selector.select(state, timeout=allowance)
        if not isinstance(decision, AuditDecision):
            raise TypeError("select() must return AuditDecision")
        decision.validate()
    except Exception as exc:
        # Never put exception bodies into traces: provider errors may contain
        # request headers, credentials, or complete state payloads.
        reason = "policy_error:" + type(exc).__name__
        metadata = getattr(optimizer.audit_selector, "__dict__", {})
        requested_model = metadata.get("model")
        prompt_version = getattr(type(optimizer.audit_selector), "PROMPT_VERSION", None)
        decision = AuditDecision(
            "skip", reason=reason, fallback_reason=reason,
            requested_model=requested_model if isinstance(requested_model, str) else None,
            prompt_version=prompt_version if isinstance(prompt_version, str) else None,
        )
        if not optimizer._policy_warned:
            warnings.warn(
                "Audit selector failed (" + type(exc).__name__ + "). Skipping audit; "
                "check the optional SDK installation, API key, and connectivity. "
                "See policy trace events for subsequent decisions.",
                RuntimeWarning, stacklevel=3,
            )
            optimizer._policy_warned = True
    elapsed = time.perf_counter() - started
    optimizer.policy_time_ += elapsed
    optimizer.policy_input_tokens_ += decision.input_tokens or 0
    selected = decision.action == "audit"
    reason = decision.reason
    fallback_reason = decision.fallback_reason
    if decision.action == "insufficient_evidence":
        reason = "insufficient_evidence"
        fallback_reason = reason
    if optimizer._config_for_solve() is None:
        selected, reason = False, "fit_budget"
        fallback_reason = reason
    elif optimizer.policy_time_ >= optimizer.policy_time_limit:
        selected, reason = False, "policy_budget"
        fallback_reason = reason
    if fallback_reason is not None:
        optimizer.policy_fallback_count_ += 1
    optimizer._record(
        event="policy", node=node, depth=node.bit_length()-1,
        state_hash=state_hash, state_schema_version=state.schema_version,
        policy_seconds=elapsed, selected=selected, effective_action="audit" if selected else "skip",
        **{**asdict(decision), "reason": reason, "fallback_reason": fallback_reason},
    )
    return selected, reason if reason != "policy" else "policy_" + decision.action


def solution_tree(solution, features):
    tree = DecisionTree(depth=2, features=features)
    for node, feature in [(1, solution.root_feature), (2, solution.left_feature), (3, solution.right_feature)]:
        tree.set_branch_feature(node, feature)
    for node, label in solution.leaf_classes.items():
        tree.set_leaf_class(node, label)
    return tree


def replace_region(tree, local, root):
    """Replace all descendants, including unequal-depth prior subtrees."""
    def descendant(node):
        while node > root:
            node //= 2
        return node == root
    for mapping in (tree.branch_nodes, tree.leaf_nodes):
        for node in list(mapping):
            if descendant(node):
                del mapping[node]
    tree._pruned_node_ids = {n for n in tree._pruned_node_ids if not descendant(n)}
    def mapped(node):
        width = 1 << (node.bit_length()-1)
        return root * width + node - width
    for node, branch in local.branch_nodes.items():
        if branch.feature_index is not None:
            tree.set_branch_feature(mapped(node), branch.feature_index)
    for node, leaf in local.leaf_nodes.items():
        if leaf.predicted_class is not None:
            tree.set_leaf_class(mapped(node), leaf.predicted_class)
    tree.depth = max(n.bit_length()-1 for n in tree.leaf_nodes)
    tree._routing_cache = None


def _record_skipped_audit(optimizer, node, reason):
    optimizer._record(event="audit", node=node, depth=node.bit_length()-1,
                      selected=False, reason=reason, adopted_as_candidate=False,
                      final_commit=False, final_commit_reason=reason)


def maybe_audit(optimizer, baseline, shallow, data, candidates, node, remaining_depth, y_idx):
    """Keep baseline through unsuccessful audits and rescore any deeper candidate."""
    mode = optimizer.adaptive_lookahead
    if mode is None:
        return baseline
    if mode == "policy":
        reason = None
        if remaining_depth < 3:
            reason = "remaining_depth"
        elif optimizer.audit_count_ >= optimizer.audit_budget:
            reason = "audit_budget"
        elif len(data) < optimizer.audit_min_samples:
            reason = "sample_floor"
        elif optimizer._config_for_solve() is None:
            reason = "fit_budget"
        elif (optimizer.policy_call_count_ >= optimizer.policy_max_calls
              or optimizer.policy_time_ >= optimizer.policy_time_limit):
            reason = "policy_budget"
        if reason is not None:
            _record_skipped_audit(optimizer, node, reason)
            return baseline
    policy_started = time.perf_counter() if mode == "policy" else None
    if (shallow.status not in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT)
            or any(feature is None for feature in (
                shallow.root_feature, shallow.left_feature, shallow.right_feature))
            or not isinstance(shallow.leaf_classes, dict)
            or any(shallow.leaf_classes.get(leaf) is None for leaf in (4, 5, 6, 7))
            or not isinstance(shallow.objective_value, Real)
            or not np.isfinite(shallow.objective_value)):
        if policy_started is not None:
            optimizer.policy_time_ += time.perf_counter() - policy_started
        _record_skipped_audit(optimizer, node, "unusable_baseline")
        return baseline
    X = np.asarray(data[baseline.features])
    y = np.asarray(data.iloc[:, y_idx])
    classes = np.unique(y).tolist()
    _, baseline_support = np.unique(baseline.apply(X), return_counts=True)
    if len(baseline_support) != 4 or baseline_support.min() < optimizer.solver_config.min_samples_leaf:
        if policy_started is not None:
            optimizer.policy_time_ += time.perf_counter() - policy_started
        _record_skipped_audit(optimizer, node, "unusable_baseline")
        return baseline
    old_obj = score_tree(baseline, X, y, optimizer.criterion)
    old_acc = float(np.mean(baseline.predict(X) == y))
    scale = len(data) if isinstance(optimizer.criterion, MisclassificationCriterion) else 1
    costs = sorted(shallow.root_costs.values())
    gap = None if len(costs) < 2 else (costs[1]-costs[0])/scale
    reason = mode
    selected = True
    if remaining_depth < 3:
        selected, reason = False, "remaining_depth"
    elif optimizer.audit_count_ >= optimizer.audit_budget:
        selected, reason = False, "audit_budget"
    elif len(data) < optimizer.audit_min_samples:
        selected, reason = False, "sample_floor"
    elif optimizer._config_for_solve() is None:
        selected, reason = False, "fit_budget"
    elif mode == "gap" and (gap is None or gap > optimizer.audit_threshold):
        selected, reason = False, "root_gap_above_threshold_or_unavailable"
    elif mode == "impurity" and old_obj/scale < optimizer.audit_threshold:
        selected, reason = False, "residual_below_threshold"
    elif mode == "random":
        rng = np.random.default_rng(np.random.SeedSequence([int(optimizer._base_seed), node, remaining_depth]))
        selected = bool(rng.random() < 0.5)
        if not selected:
            reason = "random_not_selected"
    elif mode == "policy":
        selected, reason = _select_policy(
            optimizer, shallow, y, candidates, node, remaining_depth,
            old_obj/scale, old_acc, gap, baseline=baseline, X=X, started=policy_started,
        )
    event = dict(event="audit", node=node, depth=node.bit_length()-1,
                 n_samples=len(data), n_features=len(candidates), selected=selected,
                 reason=reason, root_gap=gap, residual=old_obj/scale,
                 existing_objective=old_obj, existing_accuracy=old_acc,
                 baseline_root=shallow.root_feature, adopted_as_candidate=False,
                 final_commit=False, final_commit_reason="not_adopted")
    if not selected:
        optimizer._record(**event)
        return baseline
    config = optimizer._config_for_solve()
    if config is None:
        event.update(selected=False, reason="fit_budget", final_commit_reason="fit_budget")
        optimizer._record(**event)
        return baseline
    optimizer.audit_count_ += 1
    started = time.perf_counter()
    solution = ExactDepth3Solver(config.copy_with(mip_gap=0),
                                optimizer.criterion, n_jobs=1, deadline=optimizer._deadline).solve(
        data, candidates, classes, y_idx)
    elapsed = time.perf_counter()-started
    optimizer.audit_time_ += elapsed
    event.update(status=solution.status.value, audit_seconds=elapsed,
                 proposed_objective=None, proposed_accuracy=None)
    usable = (solution.status in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT)
              and solution.n_complete_candidates
              and isinstance(solution.objective_value, Real)
              and np.isfinite(solution.objective_value)
              and isinstance(solution.branch_features, dict)
              and set(solution.branch_features) == set(range(1, 8))
              and all(feature in candidates for feature in solution.branch_features.values())
              and isinstance(solution.leaf_classes, dict)
              and set(solution.leaf_classes) == set(range(8, 16))
              and all(label in classes for label in solution.leaf_classes.values()))
    if usable:
        candidate = DecisionTree(depth=3, features=baseline.features)
        for n, f in solution.branch_features.items():
            candidate.set_branch_feature(n, f)
        for n, label in solution.leaf_classes.items():
            candidate.set_leaf_class(n, label)
        _, supports = np.unique(candidate.apply(X), return_counts=True)
        if len(supports) != 8 or supports.min() < config.min_samples_leaf:
            event.update(outcome="sample_support_rejected",
                         final_commit_reason="sample_support_rejected")
            optimizer._record(**event)
            return baseline
        new_obj = score_tree(candidate, X, y, optimizer.criterion)
        new_acc = float(np.mean(candidate.predict(X) == y))
        # Independently scored objective must agree with the solver before adoption.
        agrees = abs(new_obj-solution.objective_value) <= 1e-10 * max(1, abs(new_obj))
        adopt = agrees and (new_obj <= old_obj+1e-10 if optimizer.acceptance_policy == "objective" else new_acc+1e-10 >= old_acc)
        event.update(proposed_objective=new_obj, proposed_accuracy=new_acc,
                     deeper_root=solution.branch_features[1], adopted_as_candidate=bool(adopt),
                     outcome=("accepted" if adopt else "objective_mismatch" if not agrees
                              else "acceptance_policy_rejected"))
        if adopt:
            baseline = candidate
    else:
        event["outcome"] = "no_usable_deeper_incumbent"
    event["final_commit_reason"] = "pending_optimizer_acceptance" if event["adopted_as_candidate"] else event["outcome"]
    optimizer._record(**event)
    return baseline
