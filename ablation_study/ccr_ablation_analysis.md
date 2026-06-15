# CCR-Judge Ablation Analysis

This report interprets component removals under the paper_repair fixed-candidate judge-only benchmark.

## main / mmlu / progressive_refinement / shuffle

- Comparative grouping: removing comparative grouping changes Acc by `-0.0065` and JCR by `+0.0387` while Reuse changes by `+0.0392`. This isolates the component as a judge-side decision factor rather than a compute-reuse factor.
- Support scoring: removing support-based deterministic ranking changes Acc by `+0.0131` and JCR by `+0.0972` while Reuse changes by `+0.0719`. This isolates the component as a judge-side decision factor rather than a compute-reuse factor.
- Shortlist compression: removing shortlist compression changes Acc by `+0.0131` and JCR by `+0.0866` while Reuse changes by `+0.0784`. This isolates the component as a judge-side decision factor rather than a compute-reuse factor.
- Random shortlist ccr_random_shortlist_seed42: result missing, so this component cannot be interpreted from this run.
- Comparative payload statistics: optional variant was not run in this matrix, so payload-vs-shortlist attribution remains pending.

## stability / mmlu / parallel_exploration / shuffle

- Comparative grouping: result missing, so this component cannot be interpreted from this run.
- Support scoring: result missing, so this component cannot be interpreted from this run.
- Shortlist compression: result missing, so this component cannot be interpreted from this run.
- Comparative payload statistics: optional variant was not run in this matrix, so payload-vs-shortlist attribution remains pending.

## Core Questions

1. CCR-Judge's core gain should be attributed to the components whose removal most reduces Acc/JCR while leaving Reuse approximately fixed. Use the deltas above to distinguish comparative grouping, support scoring, shortlist compression, and optional payload statistics.
2. The gain surface is identified by the metrics that move: Acc measures answer selection quality, JCR measures decision stability relative to dense, shortlist coverage measures compression quality, and agreement with Full measures behavioral drift.
