from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

from paper_repair.methods.ccr_judge import create_ccr_judge, run_ccr_judge_case
from paper_repair.methods.dense_judge import create_dense_reference_judge, run_dense_judge_case
from paper_repair.methods.kvcomm_judge import create_kvcomm_judge, run_kvcomm_judge_case
from paper_repair.methods.naive_reuse_judge import create_naive_reuse_judge, run_naive_reuse_judge_case
from paper_repair.methods.pal_kv_judge import create_pal_kv_judge, run_pal_kv_judge_case
from paper_repair.repair.ccr_types import FrozenCandidateRecord, JudgeMethodResult


RunCaseFn = Callable[[Any, FrozenCandidateRecord, Path], Awaitable[JudgeMethodResult]]
JudgeFactoryFn = Callable[[str], Any]


@dataclass(frozen=True)
class JudgeMethodSpec:
    name: str
    label: str
    reuse: str
    mode: str
    judge_factory: JudgeFactoryFn
    run_case: RunCaseFn
    env: Dict[str, str] = field(default_factory=dict)


DEFAULT_METHODS = ["dense", "naive_reuse", "kvcomm", "pal_kv", "ccr_judge"]


async def _run_dense_case(judge: Any, record: FrozenCandidateRecord, output_dir: Path) -> JudgeMethodResult:
    return await run_dense_judge_case(judge=judge, record=record, output_dir=output_dir)


async def _run_naive_case(judge: Any, record: FrozenCandidateRecord, output_dir: Path) -> JudgeMethodResult:
    return await run_naive_reuse_judge_case(judge=judge, record=record, output_dir=output_dir)


async def _run_kvcomm_case(judge: Any, record: FrozenCandidateRecord, output_dir: Path) -> JudgeMethodResult:
    return await run_kvcomm_judge_case(judge=judge, record=record, output_dir=output_dir)


async def _run_pal_kv_case(judge: Any, record: FrozenCandidateRecord, output_dir: Path) -> JudgeMethodResult:
    return await run_pal_kv_judge_case(judge=judge, record=record, output_dir=output_dir)


async def _run_ccr_case(judge: Any, record: FrozenCandidateRecord, output_dir: Path) -> JudgeMethodResult:
    return await run_ccr_judge_case(judge=judge, record=record, output_dir=output_dir)


METHOD_SPECS: Dict[str, JudgeMethodSpec] = {
    "dense": JudgeMethodSpec(
        name="dense",
        label="Dense Prefill",
        reuse="dense_prefill",
        mode="dense_prefill",
        judge_factory=create_dense_reference_judge,
        run_case=_run_dense_case,
    ),
    "naive_reuse": JudgeMethodSpec(
        name="naive_reuse",
        label="Naive Reuse",
        reuse="judge_kv_reuse",
        mode="allow_kv_reuse",
        judge_factory=create_naive_reuse_judge,
        run_case=_run_naive_case,
        env={"KVCOMM_NAIVE_PLACEHOLDERS": "user_question,agent_"},
    ),
    "kvcomm": JudgeMethodSpec(
        name="kvcomm",
        label="KVCOMM",
        reuse="judge_kv_reuse",
        mode="allow_kv_reuse",
        judge_factory=create_kvcomm_judge,
        run_case=_run_kvcomm_case,
    ),
    "pal_kv": JudgeMethodSpec(
        name="pal_kv",
        label="PAL-KV",
        reuse="judge_kv_reuse",
        mode="allow_kv_reuse",
        judge_factory=create_pal_kv_judge,
        run_case=_run_pal_kv_case,
        env={
            "KVCOMM_GROUP_JUDGE_ANCHORS": "1",
            "KVCOMM_GROUP_JUDGE_ANCHORS_BY_POSITION": "0",
        },
    ),
    "ccr_judge": JudgeMethodSpec(
        name="ccr_judge",
        label="CCR-Judge",
        reuse="judge_kv_reuse",
        mode="allow_kv_reuse",
        judge_factory=create_ccr_judge,
        run_case=_run_ccr_case,
    ),
}

METHOD_ALIASES = {
    "test3_repair_method": "ccr_judge",
    "ours": "pal_kv",
}


def validate_requested_methods(methods: list[str] | tuple[str, ...] | None) -> list[str]:
    requested_methods = list(methods or DEFAULT_METHODS)
    canonicalized: list[str] = []
    for method in requested_methods:
        canonical = METHOD_ALIASES.get(method, method)
        canonicalized.append(canonical)
    unsupported = [method for method in canonicalized if method not in METHOD_SPECS]
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")
    seen = set()
    result: list[str] = []
    for method in canonicalized:
        if method in seen:
            continue
        seen.add(method)
        result.append(method)
    return result
