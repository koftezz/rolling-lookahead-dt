"""Tree export utilities: text and graphviz representations."""

from typing import Optional, List


def export_text(tree, feature_names: Optional[List[str]] = None) -> str:
    """Export tree structure as a human-readable text representation.

    Parameters
    ----------
    tree : DecisionTree
        A fitted decision tree.
    feature_names : list of str, optional
        Names for each feature (length must match tree.features).
        If None, uses "feature_1", "feature_2", etc.

    Returns
    -------
    str : Multi-line text representation of the tree.
    """
    if feature_names is None:
        feature_names = [f"feature_{f}" for f in tree.features]
    else:
        if len(feature_names) != len(tree.features):
            raise ValueError(
                f"feature_names has {len(feature_names)} entries, "
                f"but tree has {len(tree.features)} features."
            )
    feat_map = dict(zip(tree.features, feature_names))
    lines = []
    _export_text_recurse(tree, 1, 0, feat_map, lines)
    return "\n".join(lines)


def _export_text_recurse(tree, node_id, depth, feat_map, lines):
    """Recursively build text lines for the tree."""
    indent = "|   " * depth

    # Leaf node
    if node_id in tree.leaf_nodes:
        leaf = tree.leaf_nodes[node_id]
        if leaf.predicted_class is not None:
            dist_str = ""
            if leaf.class_distribution:
                dist_str = " " + str(dict(leaf.class_distribution))
            lines.append(f"{indent}class: {leaf.predicted_class}{dist_str}")
        else:
            lines.append(f"{indent}(empty leaf)")
        return

    # Branch node
    node = tree.branch_nodes.get(node_id)
    if node is None or node.feature_index is None:
        lines.append(f"{indent}(unpopulated node)")
        return

    fname = feat_map.get(node.feature_index, f"feature_{node.feature_index}")
    left_child = node_id * 2
    right_child = node_id * 2 + 1

    # Left branch: feature == 1
    lines.append(f"{indent}|--- {fname} == 1")
    if left_child in tree._pruned_node_ids:
        leaf = tree.leaf_nodes.get(left_child)
        cls = leaf.predicted_class if leaf else "?"
        lines.append(f"{indent}|   class: {cls} (pruned)")
    else:
        _export_text_recurse(tree, left_child, depth + 1, feat_map, lines)

    # Right branch: feature == 0
    lines.append(f"{indent}|--- {fname} == 0")
    if right_child in tree._pruned_node_ids:
        leaf = tree.leaf_nodes.get(right_child)
        cls = leaf.predicted_class if leaf else "?"
        lines.append(f"{indent}|   class: {cls} (pruned)")
    else:
        _export_text_recurse(tree, right_child, depth + 1, feat_map, lines)


def export_graphviz(
    tree,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> str:
    """Export tree as a Graphviz DOT string.

    Parameters
    ----------
    tree : DecisionTree
        A fitted decision tree.
    feature_names : list of str, optional
        Names for each feature. If None, uses "feature_1", etc.
    class_names : list of str, optional
        Names for each class label. If None, uses the numeric labels.

    Returns
    -------
    str : DOT format string (can be rendered with graphviz).
    """
    if feature_names is None:
        feature_names = [f"feature_{f}" for f in tree.features]
    feat_map = dict(zip(tree.features, feature_names))

    class_map = {}
    if class_names is not None:
        classes = sorted(
            set(
                leaf.predicted_class
                for leaf in tree.leaf_nodes.values()
                if leaf.predicted_class is not None
            )
        )
        if len(class_names) == len(classes):
            class_map = dict(zip(classes, class_names))

    lines = ["digraph Tree {"]
    lines.append('    node [shape=box, style="filled, rounded"];')
    _graphviz_recurse(tree, 1, feat_map, class_map, lines)
    lines.append("}")
    return "\n".join(lines)


def _graphviz_recurse(tree, node_id, feat_map, class_map, lines):
    """Recursively add nodes and edges to the DOT output."""
    # Leaf node
    if node_id in tree.leaf_nodes:
        leaf = tree.leaf_nodes[node_id]
        if leaf.predicted_class is not None:
            cls = class_map.get(leaf.predicted_class, str(leaf.predicted_class))
            dist_label = ""
            if leaf.class_distribution:
                counts = list(leaf.class_distribution.values())
                dist_label = f"\\nsamples: {sum(counts)}\\nvalue: {counts}"
            label = f"class: {cls}{dist_label}"
            lines.append(
                f'    {node_id} [label="{label}", fillcolor="#aed6f1"];'
            )
        return

    # Branch node
    node = tree.branch_nodes.get(node_id)
    if node is None or node.feature_index is None:
        return

    fname = feat_map.get(node.feature_index, f"feature_{node.feature_index}")
    lines.append(
        f'    {node_id} [label="{fname} == ?", fillcolor="#f9e79f"];'
    )

    left_child = node_id * 2
    right_child = node_id * 2 + 1

    # Left child (feature == 1)
    if left_child in tree._pruned_node_ids:
        leaf = tree.leaf_nodes.get(left_child)
        if leaf and leaf.predicted_class is not None:
            cls = class_map.get(leaf.predicted_class, str(leaf.predicted_class))
            lines.append(
                f'    {left_child} [label="class: {cls}\\n(pruned)", '
                f'fillcolor="#d5dbdb"];'
            )
            lines.append(f'    {node_id} -> {left_child} [label="= 1"];')
    else:
        _graphviz_recurse(tree, left_child, feat_map, class_map, lines)
        lines.append(f'    {node_id} -> {left_child} [label="= 1"];')

    # Right child (feature == 0)
    if right_child in tree._pruned_node_ids:
        leaf = tree.leaf_nodes.get(right_child)
        if leaf and leaf.predicted_class is not None:
            cls = class_map.get(leaf.predicted_class, str(leaf.predicted_class))
            lines.append(
                f'    {right_child} [label="class: {cls}\\n(pruned)", '
                f'fillcolor="#d5dbdb"];'
            )
            lines.append(f'    {node_id} -> {right_child} [label="= 0"];')
    else:
        _graphviz_recurse(tree, right_child, feat_map, class_map, lines)
        lines.append(f'    {node_id} -> {right_child} [label="= 0"];')
