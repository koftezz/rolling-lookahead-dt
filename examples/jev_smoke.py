"""One opt-in live Jev request. Requires TYPESAFE_API_KEY and incurs API usage."""
import json
import os
from itertools import product

import numpy as np

from rollotree import JevAuditSelector, RollingOCT


def main():
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise SystemExit("Set TYPESAFE_API_KEY before running this live smoke test.")
    X = np.array(list(product((0, 1), repeat=4)) * 8)
    y = X[:, 0] ^ X[:, 1] ^ X[:, 2]
    common = dict(depth=4, solver="direct", random_state=42,
                  acceptance_policy="objective")
    baseline = RollingOCT(**common).fit(X, y)
    model = RollingOCT(
        **common, adaptive_lookahead="policy",
        audit_selector=JevAuditSelector(timeout=5),
        audit_budget=1, policy_max_calls=1, policy_time_limit=10,
        total_time_limit=30, trace=True,
    ).fit(X, y)
    decisions = [e for e in model.search_trace_ if e["event"] == "policy"]
    print(json.dumps({
        "baseline_training_accuracy": baseline.score(X, y),
        "jev_training_accuracy": model.score(X, y),
        "baseline_fit_seconds": baseline.fit_time_,
        "jev_fit_seconds": model.fit_time_,
        "policy_attempts": model.policy_call_count_,
        "policy_seconds": model.policy_time_,
        "audits": model.audit_count_,
        "decisions": decisions,
    }, indent=2, allow_nan=False))
    if not any(e.get("model") and e.get("probabilities") for e in decisions):
        raise SystemExit("No usable Jev response; inspect the warning and trace.")


if __name__ == "__main__":
    main()
