from __future__ import annotations

from pathlib import Path

from paper_repair.methods.base import PaperProtocolFinalSelectBest, create_protocol_judge, run_standard_judge_case
from paper_repair.repair.ccr_types import FrozenCandidateRecord, JudgeMethodResult


def create_naive_reuse_judge(llm_name: str, *, domain: str = "mmlu") -> PaperProtocolFinalSelectBest:
    return create_protocol_judge(
        judge_cls=PaperProtocolFinalSelectBest,
        judge_id="paper_judge_naive_reuse",
        llm_name=llm_name,
        domain=domain,
    )


async def run_naive_reuse_judge_case(
    *,
    judge,
    record: FrozenCandidateRecord,
    output_dir: Path,
) -> JudgeMethodResult:
    return await run_standard_judge_case(
        judge=judge,
        method_name="naive_reuse",
        record=record,
        output_dir=output_dir,
        mode="allow_kv_reuse",
    )
