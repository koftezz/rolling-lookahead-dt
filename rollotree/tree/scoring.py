"""Score routed observations without using solver coefficient tables."""
import numpy as np
from rollotree.tree.impurity import GiniCriterion, MisclassificationCriterion


def score_tree(tree, X, y, criterion):
    """Subset-normalized weighted Gini or misclassification count."""
    if isinstance(criterion, MisclassificationCriterion):
        return float(np.sum(tree.predict(X) != y))
    if not isinstance(criterion, GiniCriterion):
        raise TypeError("Tree scoring supports the two built-in criteria")
    leaves = tree.apply(X)
    objective = 0.0
    for leaf in np.unique(leaves):
        labels = y[leaves == leaf]
        _, counts = np.unique(labels, return_counts=True)
        objective += len(labels) / len(y) * (1 - np.sum((counts / len(labels)) ** 2))
    return float(objective)
