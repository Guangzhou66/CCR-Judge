# CCR-Judge Component Ablation Study

This directory contains standalone ablation runners for CCR-Judge. The runners
write to timestamped result roots and do not modify official scorers or overwrite
existing result directories.

## Paper-Ready Ablation Suite

Use this launcher for the current paper ablation package. It covers:

- Layer 1: controlled fixed-slate `paper_repair` component ablation.
- Layer 2: small online full-connected MMLU validation with compare-dense.
- Layer 3: variant-upgrade check against Dense Prefill, Naive Reuse, KVCOMM,
  and PAL-KV when available.

Default settings:

- model: `/pychen/Test/model/Qwen2.5-7B-Instruct`
- fixed-slate datasets: `mmlu humaneval`
- regimes: `parallel_exploration progressive_refinement`
- orderings: `noshuffle shuffle`
- variants: Full, w/o Comparative Grouping, w/o Support Scoring,
  w/o Shortlist Compression, w/o Comparative Payload Statistics, Stats Only,
  Shortlist Only, and Random Shortlist seeds `42/43/44`
- output root: `ablation_study/results/paper_ablation_qwen_<timestamp>/`

```bash
cd /path/to/CCR_JUDGE

bash ablation_study/run_ablation_experiment_suite.sh
```

Add GSM8K when resources allow:

```bash
bash ablation_study/run_ablation_experiment_suite.sh \
  --datasets "mmlu gsm8k humaneval"
```

Run a tiny end-to-end smoke check:

```bash
bash ablation_study/run_ablation_experiment_suite.sh --smoke
```

Important generated files:

- `ablation_fixed_slate_average.csv`
- `ablation_fixed_slate_average.md`
- `ablation_fixed_slate_per_setting.csv`
- `ablation_fixed_slate_per_setting.md`
- `ablation_fixed_slate_random_shortlist_seeds.csv`
- `ablation_fixed_slate_diagnostics.json`
- `ablation_fixed_slate_summary.md`
- `online_mmlu/ablation_online_mmlu_average.csv`
- `online_mmlu/ablation_online_mmlu_average.md`
- `online_mmlu/ablation_online_mmlu_details.jsonl`
- `online_mmlu/ablation_online_mmlu_summary.md`
- `variant_upgrade_check.csv`
- `variant_upgrade_check.md`
- `variant_upgrade_decision_report.md`
- `ablation_experiment_report.md`

The suite creates frozen candidate packs under its own timestamped
`candidate_cache/` and reuses those packs across all fixed-slate variants, so
the component ablation shares the same slate, order, and dense reference.

## Legacy Fixed-Slate Runner

## Default Experiment

The default command runs:

- Layer 1 main testbed: `mmlu / progressive_refinement / shuffle`
- Layer 2 stability testbed: `mmlu / parallel_exploration / shuffle`
- Protocol: `fixed_candidate_judge_only`
- Setting: `mainline_controlled`
- Candidate packs: the frozen paper_repair packs under `result/.../candidate_cache`
- Backbone: `/pychen/Test/model/Llama-3.2-3B-Instruct`

```bash
cd /path/to/CCR_JUDGE

bash ablation_study/run_ccr_ablation.sh
```

## Main Variants

- `CCR-Judge (Full)`
- `CCR-Judge w/o Comparative Grouping`
- `CCR-Judge w/o Support Scoring`
- `CCR-Judge w/o Shortlist Compression`
- `CCR-Judge with Random Shortlist`

Layer 2 stability intentionally uses the reduced matrix:

- `Full`
- `w/o Comparative Grouping`
- `w/o Support Scoring`
- `w/o Shortlist Compression`

The optional payload-statistics variant is implemented but not enabled by default:

```bash
bash ablation_study/run_ccr_ablation.sh \
  --include-optional-payload-statistics-ablation
```

## Random Shortlist Seeds

Run three random shortlist seeds:

```bash
bash ablation_study/run_ccr_ablation.sh \
  --random-shortlist-seeds "42 43 44"
```

## Smoke Test

Use two cases to verify the pipeline before a full GPU run:

```bash
bash ablation_study/run_ccr_ablation.sh \
  --layers main \
  --limit-cases 2
```

## Outputs

Each run creates a timestamped result root:

```text
ablation_study/results/ccr_ablation_<timestamp>/
```

The latest report files are also written to `ablation_study/`:

- `ccr_ablation_run_manifest.md`
- `ccr_ablation_tables.md`
- `ccr_ablation_analysis.md`
- `ccr_ablation_blockers.md`

## Support-Variant Upgrade Check

Use this runner when the question is specifically whether
`CCR-Judge w/o Support Scoring` should replace the current Full CCR-Judge
as the revised main variant. It keeps the experiment in `paper_repair` under
the `fixed_candidate_judge_only` protocol and compares:

- `CCR-Judge (Full)`
- `CCR-Judge w/o Support Scoring`
- `CCR-Judge w/o Shortlist Compression`
- optional `CCR-Judge with Random Shortlist`

The same support-variant check table also includes the existing baselines:

- `Naive Reuse`
- `KVCOMM`
- `PAL-KV`

Default settings:

- `mmlu / progressive_refinement / shuffle`
- `mmlu / parallel_exploration / shuffle`
- `humaneval / progressive_refinement / shuffle`
- `humaneval / parallel_exploration / shuffle`

```bash
cd /path/to/CCR_JUDGE

bash ablation_study/run_support_variant_check.sh
```

This shell wrapper activates the `Judge` conda environment by default because
the base Python in this container is CPU-only. Set
`SUPPORT_VARIANT_SKIP_CONDA=1` only if you intentionally manage CUDA-visible
Python outside the script.

Add the random-shortlist sanity baseline:

```bash
bash ablation_study/run_support_variant_check.sh \
  --include-random \
  --random-seeds "42"
```

Run only the representative MMLU setting:

```bash
bash ablation_study/run_support_variant_check.sh \
  --settings main
```

Run both MMLU settings only:

```bash
bash ablation_study/run_support_variant_check.sh \
  --settings mmlu
```

Run both HumanEval settings only:

```bash
bash ablation_study/run_support_variant_check.sh \
  --settings humaneval
```

Run a small path check without changing the default output location:

```bash
bash ablation_study/run_support_variant_check.sh \
  --settings main \
  --limit-cases 3
```

To chunk HumanEval into smaller subprocesses:

```bash
bash ablation_study/run_support_variant_check.sh \
  --settings humaneval \
  --humaneval-chunk-size 64
```

Support-variant reports are written to `ablation_study/`:

- `ccr_support_variant_check_run_manifest.md`
- `ccr_support_variant_tables.md`
- `ccr_support_variant_analysis.md`
- `ccr_support_variant_upgrade_recommendation.md`

## MMLU Ori-Protocol CCR Ablation

Use this runner when the ablation should be executed under the online
Ori-style protocol instead of the fixed-candidate `paper_repair` protocol.
It runs MMLU with online candidate generation and final selection in the same
run, using compare-dense to compute JCR.

Default settings:

- dataset: `mmlu / val`
- subset: `experiments/mmlu_153_seed888_indices.json`
- model: `/pychen/Test/model/Qwen2.5-7B-Instruct`
- topology: `FullConnected`
- execution mode: `allow_kv_reuse`
- orderings: `noshuffle shuffle`
- variants: `Full`, `w/o Comparative Grouping`, `w/o Support Scoring`,
  `w/o Shortlist Compression`, and random shortlist seeds `42/43/44`

```bash
cd /path/to/CCR_JUDGE

bash ablation_study/run_mmlu_ori_ablation.sh
```

Run a one-line smoke test without launching model inference:

```bash
bash ablation_study/run_mmlu_ori_ablation.sh \
  --smoke \
  --variants "full no_support" \
  --judge-shuffles "noshuffle" \
  --dry-run
```

Run a small real smoke test:

```bash
bash ablation_study/run_mmlu_ori_ablation.sh \
  --question-indices "91" \
  --variants "full" \
  --judge-shuffles "noshuffle"
```

Run a focused formal subset:

```bash
bash ablation_study/run_mmlu_ori_ablation.sh \
  --variants "full no_support no_shortlist" \
  --judge-shuffles "noshuffle shuffle"
```

Outputs are written under a timestamped result root:

- `ori_mmlu_ablation_manifest.md`
- `ori_mmlu_ablation_summary.json`
- `ori_mmlu_ablation_table.md`
