# Seed-tree quality benchmark

This benchmark compares the default rolling depth-2 seed with the optional
globally exact depth-3 seed at requested depths 3, 4, and 5.

## Method

- Dataset: bundled binarized Wine data, combining the packaged train/test rows.
- Evaluation: 5 stratified folds and 3 random feature subsets, for 15 paired
  comparisons at each requested depth and 90 fits overall.
- Features: 10 identical randomly selected columns per paired comparison. The
  models receive the same columns with `max_features=None`, isolating the
  seed-tree strategy from feature-selection luck.
- Criterion: Gini; solver: HiGHS; `n_jobs=1`.
- Command: `python benchmarks/bench_quality.py`.

## Results

| Requested depth | Strategy | Test accuracy | Train accuracy | Train Gini | Mean leaves | Mean fit seconds |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Rolling depth-2 | 0.4683 | 0.5440 | 0.5077 | 6.53 | 0.0236 |
| 3 | Exact depth-3 | 0.4606 | 0.5511 | 0.4996 | 8.00 | 0.0519 |
| 4 | Rolling depth-2 | 0.5097 | 0.5880 | 0.4732 | 10.53 | 0.0198 |
| 4 | Exact depth-3 | 0.5042 | 0.5871 | 0.4670 | 11.87 | 0.0583 |
| 5 | Rolling depth-2 | 0.5077 | 0.6142 | 0.4477 | 14.13 | 0.0252 |
| 5 | Exact depth-3 | 0.5002 | 0.6180 | 0.4422 | 15.07 | 0.0639 |

Paired exact-depth-3 differences:

| Requested depth | Test-accuracy delta | 95% CI | Gini reduction | 95% CI | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 3 | -0.0077 | [-0.0228, 0.0075] | 0.0081 | [0.0046, 0.0116] | 3.67x |
| 4 | -0.0056 | [-0.0280, 0.0169] | 0.0062 | [0.0033, 0.0090] | 2.95x |
| 5 | -0.0076 | [-0.0169, 0.0018] | 0.0055 | [0.0015, 0.0095] | 2.52x |

At requested depth 3, exact initialization improved training Gini in 13 of
15 pairs and tied in 2; it never lost on its optimized objective. Held-out
accuracy tied in 12 of 15 pairs, improved once, and declined twice. The
confidence intervals for held-out accuracy include zero at every depth.

## Interpretation

Exact depth-3 initialization provides the intended optimization-quality
benefit: a lower training Gini objective and a complete eight-leaf depth-3
seed with an optimality certificate when every candidate solve is optimal.
On this small benchmark it did not demonstrate a predictive-accuracy gain and
cost roughly 2.5-3.7 times more runtime. The rolling depth-2 seed should remain
the default; exact depth 3 is most useful when objective quality or a depth-3
certificate matters. Results from one small dataset should not be generalized
without running the benchmark on the target workload.
