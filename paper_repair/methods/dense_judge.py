from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Sequence

from KVCOMM.utils.metrics import GenerationResult
from dataset_adapters import create_dataset_adapter
from paper_repair.eval.parsing import (
    parse_dense_reference_output,
)
from paper_repair.methods.base import (
    PaperProtocolFinalSelectBest,
    build_spatial_info,
    create_protocol_judge,
)
from paper_repair.repair.ccr_types import FrozenCandidateRecord, JudgeMethodResult


class PaperDenseReferenceSelectBest(PaperProtocolFinalSelectBest):
    pass


def create_dense_reference_judge(llm_name: str, *, domain: str = "mmlu") -> PaperDenseReferenceSelectBest:
    return create_protocol_judge(
        judge_cls=PaperDenseReferenceSelectBest,
        judge_id="paper_judge_dense",
        llm_name=llm_name,
        domain=domain,
    )


def _dense_timeout_seconds() -> float | None:
    raw_value = str(os.getenv("DENSE_REFERENCE_TIMEOUT_SEC", "0") or "0").strip()
    try:
        timeout_seconds = float(raw_value)
    except Exception:
        return None
    return timeout_seconds if timeout_seconds > 0 else None


def _dense_candidate_blocks(
    candidate_agent_ids: Sequence[str],
    spatial_info: Dict[str, Any],
    *,
    dataset_name: str,
    dataset_config: str = "",
) -> str:
    adapter = create_dataset_adapter(
        dataset_name,
        config_name=dataset_config,
    )
    rows = []
    for agent_id in candidate_agent_ids:
        raw_output = str(spatial_info[str(agent_id)]["output"] or "").strip()
        parsed = adapter.parse_candidate_output(
            raw_output,
            dict(spatial_info[str(agent_id)].get("metadata") or {}),
        )
        rows.append(
            adapter.render_dense_candidate_block(
                agent_id=str(agent_id),
                parsed_candidate=parsed,
                raw_output=raw_output,
            )
        )
    return "\n".join(rows)


def _dense_messages(
    *,
    question: str,
    candidate_agent_ids: Sequence[str],
    spatial_info: Dict[str, Any],
    dataset_name: str,
    dataset_config: str = "",
) -> list[Dict[str, str]]:
    adapter = create_dataset_adapter(
        dataset_name,
        config_name=dataset_config,
    )
    return adapter.build_dense_compare_messages(
        question_text=question,
        candidate_agent_ids=candidate_agent_ids,
        candidate_blocks=_dense_candidate_blocks(
            candidate_agent_ids,
            spatial_info,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        ),
    )


def _dense_retry_messages(
    *,
    question: str,
    candidate_agent_ids: Sequence[str],
    spatial_info: Dict[str, Any],
    previous_output: str,
    dataset_name: str,
    dataset_config: str = "",
) -> list[Dict[str, str]]:
    adapter = create_dataset_adapter(
        dataset_name,
        config_name=dataset_config,
    )
    return adapter.build_dense_compare_messages(
        question_text=question,
        candidate_agent_ids=candidate_agent_ids,
        candidate_blocks=_dense_candidate_blocks(
            candidate_agent_ids,
            spatial_info,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        ),
        retry_previous_output=previous_output,
    )


async def _run_with_optional_timeout(coro):
    timeout_seconds = _dense_timeout_seconds()
    if timeout_seconds is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


async def _run_dense_prefill_once(
    *,
    judge: Any,
    record: FrozenCandidateRecord,
    output_dir: Path,
) -> tuple[GenerationResult, str]:
    spatial_info = build_spatial_info(record)
    candidate_agent_ids = list(record.permutation)
    input_dict = {"task": record.question_text, "_paper_candidate_order": candidate_agent_ids}
    use_dense_engine = os.getenv("KVCOMM_ENABLE_DENSE_ENGINE", "0").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    if use_dense_engine and hasattr(judge, "_process_inputs") and hasattr(judge.llm, "generate_for_agent"):
        request_uid = uuid.uuid4().hex[:8]
        input_dict["_request_uid"] = request_uid
        has_prefix_initialized = False
        try:
            has_prefix_initialized = bool(judge.llm.has_prefix_initialized(judge.id))
        except Exception:
            has_prefix_initialized = False
        if not has_prefix_initialized:
            await judge._process_inputs(
                input_dict,
                spatial_info,
                {},
                mode="allow_kv_reuse",
                output_dir=str(output_dir),
            )
        await judge._process_inputs(
            input_dict,
            spatial_info,
            {},
            mode="allow_kv_reuse",
            output_dir=str(output_dir),
        )
        result = await _run_with_optional_timeout(
            judge.llm.generate_for_agent(
                request_uid=request_uid,
                message=input_dict["task"],
                preferred_mode="dense_prefill",
                max_tokens=12,
                output_dir=str(output_dir),
                agent_id=judge.id,
                agent_name=judge.agent_name,
                agent_role=judge.role,
            )
        )
        try:
            judge.llm.finalize_request(request_uid)
        except Exception:
            pass
        return result, "generate_for_agent_dense_prefill"

    result = await _run_with_optional_timeout(
        judge.llm.agen(
            _dense_messages(
                question=record.question_text,
                candidate_agent_ids=candidate_agent_ids,
                spatial_info=spatial_info,
                dataset_name=record.dataset_name,
                dataset_config=record.dataset_config,
            ),
            max_tokens=12,
            agent_id=judge.id,
            agent_name=judge.agent_name,
            agent_role=judge.role,
        )
    )
    return result, "direct_dense_prompt_prefill"


async def _retry_dense_parse_once(
    *,
    judge: Any,
    record: FrozenCandidateRecord,
    output_dir: Path,
    previous_output: str,
) -> GenerationResult:
    spatial_info = build_spatial_info(record)
    return await _run_with_optional_timeout(
        judge.llm.agen(
            _dense_retry_messages(
                question=record.question_text,
                candidate_agent_ids=record.permutation,
                spatial_info=spatial_info,
                previous_output=previous_output,
                dataset_name=record.dataset_name,
                dataset_config=record.dataset_config,
            ),
            max_tokens=12,
            agent_id=judge.id,
            agent_name=judge.agent_name,
            agent_role=judge.role,
        )
    )


def _failure_probe(
    *,
    failure_reason: str,
    previous_output: str,
) -> Dict[str, Any]:
    return {
        "raw_text": str(previous_output or ""),
        "sanitized_text": str(previous_output or "").strip(),
        "normalized_text": str(previous_output or "").strip(),
        "inspected_lines": [],
        "selection_like_lines": [],
        "selected_agent_id": None,
        "selected_answer": None,
        "found_candidate_ids": [],
        "invalid_candidate_ids": [],
        "parse_success": False,
        "failure_reason": failure_reason,
    }


def _selected_answer_for_record(record: FrozenCandidateRecord, selected_agent_id: str | None) -> str | None:
    adapter = create_dataset_adapter(
        record.dataset_name,
        config_name=record.dataset_config,
    )
    return adapter.selected_answer_for_agent(record.candidates, selected_agent_id)


def _canonical_dense_result_text(
    *,
    record: FrozenCandidateRecord,
    selected_agent_id: str | None,
    selected_answer: str | None,
    raw_text: Any,
) -> str:
    if selected_agent_id is None or selected_answer is None:
        return str(raw_text or "")
    adapter = create_dataset_adapter(
        record.dataset_name,
        config_name=record.dataset_config,
    )
    try:
        return str(
            adapter.format_selected_result_text(
                selected_agent_id=str(selected_agent_id),
                selected_answer=str(selected_answer),
            )
        )
    except Exception:
        return str(raw_text or "")


async def _execute_dense_attempt(
    *,
    judge: Any,
    record: FrozenCandidateRecord,
    output_dir: Path,
    retry: bool = False,
    previous_output: str = "",
) -> Dict[str, Any]:
    try:
        if retry:
            generation = await _retry_dense_parse_once(
                judge=judge,
                record=record,
                output_dir=output_dir,
                previous_output=previous_output,
            )
            generation_path = "repair_retry_prompt"
        else:
            generation, generation_path = await _run_dense_prefill_once(
                judge=judge,
                record=record,
                output_dir=output_dir,
            )
        parse_probe = parse_dense_reference_output(
            generation.text,
            allowed_ids=record.permutation,
        )
        selected_agent_id = parse_probe.get("selected_agent_id")
        selected_answer = _selected_answer_for_record(record, selected_agent_id)
        parse_success = bool(selected_agent_id is not None)
        return {
            "status": "ok",
            "generation_path": generation_path,
            "generation_result": generation,
            "parse_probe": {
                **parse_probe,
                "selected_answer": selected_answer if selected_answer is not None else parse_probe.get("selected_answer"),
                "parse_success": parse_success,
                "failure_reason": None if parse_success else parse_probe.get("failure_reason"),
            },
        }
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "generation_path": "repair_retry_prompt" if retry else "direct_dense_prompt_prefill",
            "generation_result": None,
            "parse_probe": _failure_probe(
                failure_reason="timeout",
                previous_output=previous_output,
            ),
        }
    except Exception as exc:
        return {
            "status": "runtime_exception",
            "generation_path": "repair_retry_prompt" if retry else "direct_dense_prompt_prefill",
            "generation_result": None,
            "parse_probe": {
                **_failure_probe(
                    failure_reason="runtime_exception",
                    previous_output=previous_output,
                ),
                "runtime_error_type": type(exc).__name__,
                "runtime_error_message": str(exc),
            },
        }


async def probe_dense_reference_case(
    *,
    judge: Any,
    record: FrozenCandidateRecord,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_pass = await _execute_dense_attempt(
        judge=judge,
        record=record,
        output_dir=output_dir,
        retry=False,
    )

    retry_used = False
    retry_pass = None
    final_probe = dict(first_pass["parse_probe"])
    final_generation = first_pass.get("generation_result")
    if not bool(final_probe.get("parse_success")) and final_probe.get("failure_reason") not in {
        "timeout",
        "runtime_exception",
    }:
        retry_used = True
        retry_pass = await _execute_dense_attempt(
            judge=judge,
            record=record,
            output_dir=output_dir,
            retry=True,
            previous_output=str((first_pass.get("generation_result").text if first_pass.get("generation_result") else "") or ""),
        )
        if bool(retry_pass["parse_probe"].get("parse_success")):
            final_probe = dict(retry_pass["parse_probe"])
            final_generation = retry_pass.get("generation_result")

    retry_success = bool(retry_pass and retry_pass["parse_probe"].get("parse_success"))
    final_failure_reason = final_probe.get("failure_reason")
    if retry_used and not retry_success and final_failure_reason not in {None, "timeout", "runtime_exception"}:
        final_failure_reason = "retry_failed"
    if retry_used and not retry_success:
        final_probe = {
            **final_probe,
            "failure_reason": final_failure_reason,
            "parse_success": False,
        }

    return {
        "record_index": record.record_index,
        "generation_regime": record.generation_regime,
        "permutation": list(record.permutation),
        "first_pass": first_pass,
        "retry_used": retry_used,
        "retry_pass": retry_pass,
        "retry_success": retry_success,
        "final_probe": final_probe,
        "final_generation": final_generation,
    }


async def run_dense_judge_case(
    *,
    judge: Any,
    record: FrozenCandidateRecord,
    output_dir: Path,
) -> JudgeMethodResult:
    probe = await probe_dense_reference_case(
        judge=judge,
        record=record,
        output_dir=output_dir,
    )
    final_probe = dict(probe["final_probe"])
    final_generation = probe.get("final_generation")
    first_pass = probe["first_pass"]
    retry_pass = probe.get("retry_pass")

    selected_agent_id = final_probe.get("selected_agent_id")
    selected_answer = _selected_answer_for_record(record, selected_agent_id)
    parse_success = bool(selected_agent_id is not None)
    failure_reason = None if parse_success else final_probe.get("failure_reason")
    raw_text = ""
    if final_generation is not None:
        raw_text = str(final_generation.text or "")
    elif retry_pass and retry_pass.get("generation_result") is not None:
        raw_text = str(retry_pass["generation_result"].text or "")
    elif first_pass.get("generation_result") is not None:
        raw_text = str(first_pass["generation_result"].text or "")
    raw_text_before_canonicalization = raw_text
    raw_text = _canonical_dense_result_text(
        record=record,
        selected_agent_id=selected_agent_id,
        selected_answer=selected_answer,
        raw_text=raw_text,
    )

    fallback_reason = None
    if probe["retry_used"] and probe["retry_success"]:
        fallback_reason = "dense_parse_retry_success"
    elif probe["retry_used"] and not probe["retry_success"]:
        fallback_reason = "dense_parse_failure_after_retry"
    elif failure_reason in {"timeout", "runtime_exception"}:
        fallback_reason = failure_reason

    raw_generation_metadata = {}
    if final_generation is not None:
        raw_generation_metadata = dict(final_generation.metadata or {})

    return JudgeMethodResult(
        method_name="dense",
        selected_agent_id=selected_agent_id,
        selected_answer=selected_answer,
        parse_success=parse_success,
        fallback_used=bool(probe["retry_used"]),
        fallback_reason=fallback_reason,
        raw_judge_text=raw_text,
        reuse_mode="dense_prefill",
        ttft=(final_generation.ttft if final_generation is not None else None),
        metadata={
            "dense_reference_mode": "dense_prefill",
            "raw_judge_text_before_canonicalization": raw_text_before_canonicalization,
            "dense_generation_path": first_pass.get("generation_path"),
            "dense_parse_retry_used": bool(probe["retry_used"]),
            "dense_parse_retry_success": bool(probe["retry_success"]),
            "dense_parse_failure_reason": failure_reason,
            "dense_first_pass_failure_reason": first_pass["parse_probe"].get("failure_reason"),
            "dense_retry_failure_reason": (
                retry_pass["parse_probe"].get("failure_reason") if retry_pass is not None else None
            ),
            "dense_raw_response_before_normalization": (
                first_pass["generation_result"].text if first_pass.get("generation_result") is not None else ""
            ),
            "dense_raw_response_before_retry": (
                first_pass["generation_result"].text if first_pass.get("generation_result") is not None else ""
            ),
            "dense_retry_raw_response": (
                retry_pass["generation_result"].text
                if retry_pass is not None and retry_pass.get("generation_result") is not None
                else None
            ),
            "dense_normalized_text": final_probe.get("normalized_text"),
            "dense_sanitized_text": final_probe.get("sanitized_text"),
            "dense_first_pass_normalized_text": first_pass["parse_probe"].get("normalized_text"),
            "dense_first_pass_sanitized_text": first_pass["parse_probe"].get("sanitized_text"),
            "dense_retry_normalized_text": (
                retry_pass["parse_probe"].get("normalized_text") if retry_pass is not None else None
            ),
            "dense_retry_sanitized_text": (
                retry_pass["parse_probe"].get("sanitized_text") if retry_pass is not None else None
            ),
            "dense_parser_result": final_probe,
            "dense_first_pass_parser_result": first_pass["parse_probe"],
            "dense_retry_parser_result": (
                retry_pass["parse_probe"] if retry_pass is not None else None
            ),
            "dense_debug_probe": {
                "retry_used": probe["retry_used"],
                "retry_success": probe["retry_success"],
                "first_pass_status": first_pass.get("status"),
                "retry_status": retry_pass.get("status") if retry_pass is not None else None,
            },
            "raw_generation_metadata": raw_generation_metadata,
        },
    )
