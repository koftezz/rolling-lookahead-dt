"""Tree utility functions for node generation and index mapping."""


def generate_nodes(depth: int) -> tuple:
    """
    Generate parent and leaf node IDs for a complete binary tree.

    Uses standard binary tree numbering: root=1, left child of i is 2i,
    right child of i is 2i+1.

    Returns:
        (parent_node_ids, leaf_node_ids)
    """
    nodes = list(range(1, 2 ** (depth + 1)))
    parent_nodes = nodes[: 2 ** depth - 1]
    leaf_nodes = nodes[2 ** depth - 1 :]
    return parent_nodes, leaf_nodes


def get_leaf_paths_depth2() -> dict:
    """
    Return the leaf paths for a depth-2 binary tree.

    Maps leaf_id -> [root_split_value, second_split_value] where
    1 means "feature == 1" (go left) and 0 means "feature == 0" (go right).

    Leaf 4: left-left  [1, 1]
    Leaf 5: left-right [1, 0]
    Leaf 6: right-left [0, 1]
    Leaf 7: right-right[0, 0]
    """
    return {4: [1, 1], 5: [1, 0], 6: [0, 1], 7: [0, 0]}


def leaf_pattern(sub_leaf: int, depth: int, leaf: int) -> int:
    """
    Map a subtree's leaf index to the global tree's leaf index.

    Args:
        sub_leaf: Leaf node index in the depth-2 subtree (4, 5, 6, or 7).
        depth: Depth of the subtree (always 2 for OCT-2).
        leaf: The parent node in the main tree being expanded.
    """
    return (sub_leaf - 2**depth) + 2**depth * leaf


def parent_pattern(sub_leaf: int, leaf_node: int) -> int:
    """
    Map a subtree's parent index to the global tree's parent index.

    Args:
        sub_leaf: Parent node index in the depth-2 subtree (1, 2, or 3).
        leaf_node: The node in the main tree being expanded.
    """
    if sub_leaf == 1:
        return leaf_node
    elif sub_leaf < 4:
        return (2 * leaf_node) + (sub_leaf % 2)
    elif sub_leaf < 8:
        return (4 * leaf_node) + (sub_leaf % 4)
    else:
        raise ValueError(f"sub_leaf {sub_leaf} out of range for parent_pattern")


def get_child(current_depth: int, target_depth: int, child_node: int) -> int:
    """
    Get the descendant node at target_depth by following the same-parity path.

    For even nodes: always goes left (multiply by 2).
    For odd nodes: always goes right (multiply by 2 + 1).
    """
    for _ in range(target_depth - current_depth):
        if child_node % 2 == 0:
            child_node = child_node * 2
        else:
            child_node = child_node * 2 + 1
    return child_node
