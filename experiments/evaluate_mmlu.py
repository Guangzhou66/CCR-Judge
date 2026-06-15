from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from experiments.evaluate_mmlu_ori_protocol import evaluate_ori_protocol


DEPRECATION_NOTE = (
    "Compatibility wrapper: experiments/evaluate_mmlu.py now forwards to "
    "experiments/evaluate_mmlu_ori_protocol.py. "
    "Use evaluate_ori_protocol(...) for the canonical online protocol path."
)


async def evaluate(
    graph,
    dataset,
    limit_questions: Optional[int] = None,
    eval_batch_size: int = 1,
    *,
    mode: str = "default",
    question_indices: Optional[Sequence[int]] = None,
    **kwargs,
) -> float:
    judge_shuffle = os.getenv("KVCOMM_JUDGE_SHUFFLE", "0").lower() in {"1", "true", "yes", "y"}
    passthrough = dict(kwargs)
    results_path = passthrough.pop("results_path", None)
    summary_path = passthrough.pop("summary_path", None)
    main_table_path = passthrough.pop("main_table_path", None)
    judge_compare_dense = bool(passthrough.get("judge_compare_dense"))
    decision_method = passthrough.get("decision_method", "OriProtocolFinalSelectBest")
    topology_mode = passthrough.get("topology_mode", "FullConnected")
    summary = await evaluate_ori_protocol(
        graph=graph,
        dataset=dataset,
        limit_questions=limit_questions,
        question_indices=question_indices,
        eval_batch_size=eval_batch_size,
        mode=mode,
        execution_mode=mode,
        judge_compare_dense_enabled=judge_compare_dense,
        judge_shuffle=judge_shuffle,
        topology_mode=topology_mode,
        decision_method=decision_method,
        results_path=results_path,
        summary_path=summary_path,
        main_table_path=main_table_path,
        **passthrough,
    )
    return float(summary.get("acc") or 0.0)


__all__ = [
    "DEPRECATION_NOTE",
    "evaluate",
]
