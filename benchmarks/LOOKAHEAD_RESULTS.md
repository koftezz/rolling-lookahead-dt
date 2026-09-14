# Reproducible lookahead pilot

This pilot supports computational improvements for the same local OCT-2 problem.
It does **not** establish that small-gap adaptive allocation is better than
simpler deeper-lookahead policies. Defaults remain unchanged and adaptation is
experimental. No novelty claim is made.

## Environment and controls

Measured on Apple M4 Pro (12 CPU cores), 24 GiB RAM, macOS ARM64, Python 3.12.12.
The JSON artifacts record OS and package versions and the full source revision.
Library thread pools were limited to one. Fits were sequential, in fresh isolated
processes; the paired study preloaded imports before forking, with no fit warmup.
The direct benchmark used fresh interpreter processes. Peak RSS includes inputs,
libraries, and interpreter memory. These different process setups must not be
used to compare absolute RSS across the two studies.

Three fixed stratified 70/30 splits were paired across methods. Real datasets are
scikit-learn Iris, Wine, and Breast Cancer. Synthetic datasets cover three-way XOR,
a single-feature easy target, and noisy two-way XOR (30% label flips). Real-data
binary thresholds are medians fitted on training rows only. No parameters were
selected on test data. Dataset dimensions, imbalance, candidate counts, split
fingerprints, complete configurations, warnings and status are in the raw rows.

The study uses weighted Gini, depth 4, objective acceptance, and cooperative
0.05/0.25-second fit allowances. All online audit methods get at most two audits.
Gap uses threshold 0.02, impurity uses residual threshold 0.1, sample selection
requires 64 rows, and other methods use a 32-row floor. These are transparent,
prespecified pilot settings, not tuned estimators. Random uses a seeded 50%
selection decision and may spend fewer audits. CART has no enforced time limit;
exact3 is a complete depth-3 reference, not a same-depth-4 oracle. Fixed deeper
initialization and fixed auditing are reported separately.

Equal allowances are **not equal actual compute**: see recorded fit/audit times.
The experiment does not establish a matched-actual-runtime advantage. Timing
includes all fitting work; training-only preprocessing is separately recorded
and also included in total_seconds. Imports and process startup are excluded.
Warnings were captured during fits. No external benchmark timing is reused.

## Direct efficiency

160 synthetic rows, three classes, seed 42, three repetitions per configuration.
All observed objectives agree within 1e-10. Median wall time and peak process RSS:

| Features | Backend | Block | Seconds | RSS MiB |
|---:|---|---:|---:|---:|
| 20 | highs | full | 0.0349 | 150.5 |
| 20 | direct | full | 0.0020 | 147.7 |
| 20 | direct | 8 | 0.0013 | 147.2 |
| 20 | direct | 32 | 0.0011 | 147.4 |
| 80 | highs | full | 0.5744 | 172.4 |
| 80 | direct | full | 0.0189 | 151.9 |
| 80 | direct | 8 | 0.0028 | 147.6 |
| 80 | direct | 32 | 0.0019 | 147.9 |
| 130 | highs | full | 1.5095 | 209.1 |
| 130 | direct | full | 0.0480 | 161.7 |
| 130 | direct | 8 | 0.0042 | 147.5 |
| 130 | direct | 32 | 0.0029 | 148.4 |
| 400 | direct | full | 0.5302 | 299.9 |
| 400 | direct | 8 | 0.0160 | 149.6 |
| 400 | direct | 32 | 0.0098 | 149.9 |

At 400 features, blocking approximately halves whole-process RSS; the reduction
in pair-table memory is larger than the whole-process ratio because interpreter
and inputs remain. This is one machine and synthetic workload, not a universal
speedup claim. Raw repetitions and separate coefficient/assembly/solve timings
are in [direct_results.json](direct_results.json). The repaired membership
operation uses O(p²) dictionary lookups; this says nothing about total fitting
complexity. Coefficients and rolling expansion have additional costs.

## Commitment diagnostics

60 seeded truth tables over all 16 four-feature binary patterns, repeated four
times. All fixed/free depth-3 solves are exact. Positive regret occurs in 11/60.
A gap <=0.02 selects 45 cases, catches 7 of the 11 positive-regret cases, and
misses 4. Of 45 selected cases, 38 have zero regret. This fails to support the
small-gap heuristic as a reliable audit selector.

A single fixed initial audit produces **no final depth-4 objective gain in any
of these 60 cases**, including those with positive local depth-3 regret: rolling
expansion catches up. This separates local diagnostic regret from final quality.
Controlled label perturbations (three additional shallow solves per case) are
recorded offline, not treated as free online signals. Sample count is constant
in this diagnostic family, so its predictive usefulness cannot be inferred here.
See [commitment_results.json](commitment_results.json).

## Paired quality results

Held-out accuracy, mean ± sample standard deviation across the three split seeds
at the 0.25-second allowance (percentage points). These small samples are
exploratory; they are not statistical proof of superiority.

| Dataset | Direct | Gap | Random | Impurity | Samples | Initial depth 3 |
|---|---:|---:|---:|---:|---:|---:|
| iris | 82.2 ± 8.9 | 81.5 ± 9.0 | 81.5 ± 9.0 | 81.5 ± 9.0 | 81.5 ± 9.0 | 81.5 ± 9.0 |
| wine | 90.7 ± 1.9 | 92.6 ± 1.9 | 92.6 ± 1.9 | 93.2 ± 2.1 | 93.2 ± 2.1 | 93.2 ± 2.1 |
| breast_cancer | 91.8 ± 3.0 | 91.0 ± 2.2 | 91.0 ± 2.2 | 91.0 ± 2.2 | 91.0 ± 2.2 | 91.0 ± 2.2 |
| xor3 | 93.1 ± 12.0 | 100.0 ± 0.0 | 93.1 ± 12.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| easy | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| noisy | 67.1 ± 8.4 | 67.5 ± 7.8 | 67.5 ± 7.8 | 67.5 ± 7.8 | 67.5 ± 7.8 | 67.5 ± 7.8 |

Deeper lookahead helps XOR and Wine in this pilot, slightly hurts Iris and Breast
Cancer, and adds nothing on the easy case. Gap does not consistently improve on
impurity/sample selection or fixed initialization. Lower training Gini does not
ensure better held-out performance: Iris improves training Gini while losing
held-out accuracy. Training-objective monotonicity applies within a fit, not
between final trees produced by different initialization/search policies.

All 360 runs returned fitted models; status counts and average cost across both
allowances (36 observations per method) follow. These aggregate means mix datasets
and should not be interpreted as population performance estimates.

| Method | Mean fit seconds | Mean audits | Time-limit status count |
|---|---:|---:|---:|
| cart | 0.0032 | 0.00 | 0 |
| highs | 0.0577 | 0.00 | 9 |
| direct | 0.0148 | 0.00 | 0 |
| initial3 | 0.0229 | 0.00 | 3 |
| exact3 | 0.0188 | 0.00 | 0 |
| gap | 0.0247 | 1.06 | 3 |
| random | 0.0223 | 0.67 | 2 |
| impurity | 0.0237 | 0.83 | 3 |
| samples | 0.0247 | 1.00 | 3 |
| fixed | 0.0242 | 1.00 | 3 |

28/360 fits exceeded their nominal allowance, consistent with documented
cooperative semantics. No hard deadline is claimed. Balanced accuracy, training
objective, tree depth, leaf count, actual audit expenditure, revised-root counts,
RSS, warnings, and every search trace are in [adaptive_results.json](adaptive_results.json).
The exact3 comparator is a strict complete depth-3 reference only; installing
and benchmarking general exact depth-4 methods is outside this pilot.

## Reproduction

```bash
python benchmarks/bench_direct.py --output /tmp/direct_results.json
python benchmarks/diagnose_commitment.py --output /tmp/commitment_results.json
python benchmarks/evaluate_adaptive.py --output /tmp/adaptive_results.json
python benchmarks/verify_traces.py /tmp/adaptive_results.json
python benchmarks/capture_viewer_traces.py --output /tmp/viewer_traces.json
python benchmarks/verify_traces.py /tmp/viewer_traces.json
python examples/build_trace_viewer.py --input /tmp/viewer_traces.json --output /tmp/trace_viewer.html
```

Use the revision in each artifact to reproduce that experiment's exact source.
Dependencies are declared in pyproject.toml; exact installed versions are recorded
in the main and direct artifacts. The diagnostic source uses the same environment.
Timing and timeout behavior will vary across machines and system load. A clean
source revision and paired splits do not make short wall-clock budgets bitwise
reproducible across hardware.

## Trace viewer and delivery

[The self-contained viewer](../examples/trace_viewer.html) reads saved Python
runs; it never executes training in JavaScript. Its three modes use the established
**accuracy acceptance policy**, unlike the objective-policy research comparison
above. Viewer traces are captured separately with identical splits and candidates.
Every result exposes dataset, seed, configuration, split fingerprint, package
versions and source revision. Trees are final fitted trees; the event inspector
shows decision records without falsely presenting intermediate tree animation.
The verifier reconstructs all saved trees and checks objective, held-out accuracy,
depth and leaf count. No raw training observations are embedded.

The user stopped chart/Site work before deployment. The repository viewer and
its saved traces are retained, but no deployed Site update is claimed. Source
retrieval had been blocked by automatic approval review; that step is no longer
part of the requested wrap-up.

Bounded revision for changing data remains a separate future direction. It is
not implemented here; revision constraints may invalidate the direct decomposition.
