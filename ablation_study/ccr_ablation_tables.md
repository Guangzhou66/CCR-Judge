# CCR-Judge Ablation Tables

## main / mmlu / progressive_refinement / shuffle

| Ablation Variant | Acc | JCR | Reuse | Dense winner in shortlist rate | Shortlist answer-group coverage | Avg judge input length | Selection gap | Agreement with Full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCR-Judge (Full) | 0.4902 | 0.1905 | 0.7255 | 0.7059 | 1.0000 | 257.1373 | 0.0458 | 1.0000 |
| CCR-Judge w/o Comparative Grouping | 0.4837 | 0.2292 | 0.7647 | 0.7516 | 1.0000 | 160.0196 | 0.0523 | 0.4184 |
| CCR-Judge w/o Support Scoring | 0.5033 | 0.2877 | 0.7974 | 0.7516 | 1.0000 | 257.1373 | 0.0327 | 0.4828 |
| CCR-Judge w/o Shortlist Compression | 0.5033 | 0.2770 | 0.8039 | 1.0000 | 1.0000 | 267.1373 | 0.0327 | 0.4514 |
| CCR-Judge with Random Shortlist (seed=42) | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

## stability / mmlu / parallel_exploration / shuffle

| Ablation Variant | Acc | JCR | Reuse | Dense winner in shortlist rate | Shortlist answer-group coverage | Avg judge input length | Selection gap | Agreement with Full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCR-Judge (Full) | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CCR-Judge w/o Comparative Grouping | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CCR-Judge w/o Support Scoring | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CCR-Judge w/o Shortlist Compression | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
