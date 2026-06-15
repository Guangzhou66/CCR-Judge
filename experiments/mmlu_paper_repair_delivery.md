# MMLU Paper Repair Delivery

## Current Protocols

This workspace now keeps two protocol paths in parallel:

- `paper_repair`
  fixed candidates + judge-only benchmark + strict paper JCR
- `ori_style_online_protocol`
  online full-graph execution + optional compare-dense within the same run

## CCR-Judge

CCR-Judge is a judge-side, interaction-aware decision-context repair method applied at the final selection stage.

It estimates structured candidate-slate interactions, constructs a comparative decision-context payload, serializes that payload into judge-consumable context, and then performs final candidate selection.

This method is not a static prompt tweak:

- it is instance-specific and slate-conditioned
- text is only the serialization interface of a structured comparative context
- it operates at the final selection stage and does not modify candidate generation

The current method identifiers are:

- internal method name: `ccr_judge`
- display label: `CCR-Judge`
- internal mode label: `comparative_context_restoration`

Canonical method implementation:

- [`paper_repair/methods/ccr_judge.py`](../paper_repair/methods/ccr_judge.py)

Canonical CCR modules for comparative context estimation, decision-context payload construction, context serialization, and final selection:

- [`paper_repair/repair/ccr_types.py`](../paper_repair/repair/ccr_types.py)
- [`paper_repair/repair/ccr_state.py`](../paper_repair/repair/ccr_state.py)
- [`paper_repair/repair/ccr_shortlist.py`](../paper_repair/repair/ccr_shortlist.py)
- [`paper_repair/repair/ccr_context.py`](../paper_repair/repair/ccr_context.py)
- [`paper_repair/repair/ccr_fallback.py`](../paper_repair/repair/ccr_fallback.py)

## Paper-Repair Entry Points

- [`experiments/build_mmlu_paper_frozen_candidates.py`](build_mmlu_paper_frozen_candidates.py)
- [`experiments/run_mmlu_paper_repair.py`](run_mmlu_paper_repair.py)
- [`experiments/run_mmlu_paper_repair_153.py`](run_mmlu_paper_repair_153.py)
- [`run_all_paper_repair_formal.sh`](../run_all_paper_repair_formal.sh)

Paper protocol / benchmark glue:

- [`paper_repair/protocol/frozen_pack.py`](../paper_repair/protocol/frozen_pack.py)
- [`paper_repair/protocol/method_specs.py`](../paper_repair/protocol/method_specs.py)
- [`paper_repair/protocol/runner.py`](../paper_repair/protocol/runner.py)

Paper evaluation:

- [`paper_repair/eval/parsing.py`](../paper_repair/eval/parsing.py)
- [`paper_repair/eval/jcr.py`](../paper_repair/eval/jcr.py)
- [`paper_repair/eval/details.py`](../paper_repair/eval/details.py)
- [`paper_repair/eval/summary.py`](../paper_repair/eval/summary.py)

## Ori-Style Entry Points

- [`experiments/run_mmlu_ori_protocol.py`](run_mmlu_ori_protocol.py)
- [`experiments/run_mmlu_ori_protocol_153.py`](run_mmlu_ori_protocol_153.py)
- [`run_all_mmlu_ori_protocol.sh`](../run_all_mmlu_ori_protocol.sh)
- [`experiments/evaluate_mmlu_ori_protocol.py`](evaluate_mmlu_ori_protocol.py)

Protocol bridge agents:

- [`KVCOMM/agents/ori_protocol_final_decision.py`](../KVCOMM/agents/ori_protocol_final_decision.py)
- [`KVCOMM/agents/ori_protocol_repair_final_decision.py`](../KVCOMM/agents/ori_protocol_repair_final_decision.py)

Legacy online entry points retained as compatibility wrappers:

- [`experiments/run_mmlu.py`](run_mmlu.py)
- [`experiments/evaluate_mmlu.py`](evaluate_mmlu.py)

## Commands

Paper-repair full 153:

```bash
bash run_all_paper_repair_formal.sh \
  --llm-name /pychen/Test/model/Qwen2.5-7B-Instruct \
  --result-root result/mmlu_paper_repair_formal \
  --cuda-visible-devices 0
```

Ori-style full 153:

```bash
bash run_all_mmlu_ori_protocol.sh \
  --llm-name /pychen/Test/model/Qwen2.5-7B-Instruct \
  --result-root result/mmlu_ori_protocol_153 \
  --cuda-visible-devices 0 \
  --execution-modes "default allow_kv_reuse" \
  --judge-shuffles "noshuffle shuffle" \
  --judge-compare-dense
```

## Notes

- `test3_repair_method` is still accepted as a compatibility alias and resolves to `ccr_judge`.
- The paper protocol keeps strict JCR semantics on fixed candidate packs.
- The Ori-style protocol reports online intra-run compare-dense consistency, not fixed-candidate strict paper JCR.
