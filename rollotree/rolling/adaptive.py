"""Transparent, opt-in deeper-audit heuristics. No generalization certificate."""
import time
import numpy as np
from rollotree.solver.base import SolverStatus
from rollotree.solver.depth3 import ExactDepth3Solver
from rollotree.tree.nodes import DecisionTree
from rollotree.tree.scoring import score_tree
from rollotree.tree.impurity import MisclassificationCriterion


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


def maybe_audit(optimizer, baseline, shallow, data, candidates, node, remaining_depth, y_idx):
    """Keep baseline through unsuccessful audits and rescore any deeper candidate."""
    mode = optimizer.adaptive_lookahead
    if mode is None:
        return baseline
    X = np.asarray(data[baseline.features])
    y = np.asarray(data.iloc[:, y_idx])
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
    event = dict(event="audit", node=node, depth=node.bit_length()-1,
                 n_samples=len(data), n_features=len(candidates), selected=selected,
                 reason=reason, root_gap=gap, residual=old_obj/scale,
                 existing_objective=old_obj, existing_accuracy=old_acc,
                 baseline_root=shallow.root_feature, adopted_as_candidate=False)
    if not selected:
        optimizer._record(**event)
        return baseline
    config = optimizer._config_for_solve()
    if config is None:
        event.update(selected=False, reason="fit_budget")
        optimizer._record(**event)
        return baseline
    optimizer.audit_count_ += 1
    started = time.perf_counter()
    solution = ExactDepth3Solver(config.copy_with(mip_gap=0),
                                optimizer.criterion, n_jobs=1, deadline=optimizer._deadline).solve(
        data, candidates, np.unique(y).tolist(), y_idx)
    elapsed = time.perf_counter()-started
    optimizer.audit_time_ += elapsed
    event.update(status=solution.status.value, audit_seconds=elapsed,
                 proposed_objective=None, proposed_accuracy=None)
    if solution.status in (SolverStatus.OPTIMAL, SolverStatus.TIME_LIMIT) and solution.n_complete_candidates:
        candidate = DecisionTree(depth=3, features=baseline.features)
        for n, f in solution.branch_features.items():
            candidate.set_branch_feature(n, f)
        for n, label in solution.leaf_classes.items():
            candidate.set_leaf_class(n, label)
        new_obj = score_tree(candidate, X, y, optimizer.criterion)
        new_acc = float(np.mean(candidate.predict(X) == y))
        # Independently scored objective must agree with the solver before adoption.
        agrees = abs(new_obj-solution.objective_value) <= 1e-10 * max(1, abs(new_obj))
        adopt = agrees and (new_obj <= old_obj+1e-10 if optimizer.acceptance_policy == "objective" else new_acc+1e-10 >= old_acc)
        event.update(proposed_objective=new_obj, proposed_accuracy=new_acc,
                     deeper_root=solution.branch_features[1], adopted_as_candidate=bool(adopt),
                     outcome="accepted" if adopt else "acceptance_policy_rejected")
        if adopt:
            baseline = candidate
    else:
        event["outcome"] = "no_usable_deeper_incumbent"
    optimizer._record(**event)
    return baseline
