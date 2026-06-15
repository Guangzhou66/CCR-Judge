from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Sequence

from KVCOMM.utils.metrics import GenerationResult
from dataset_adapters import create_dataset_adapter
from paper_repair.eval.parsing import (
    normalize_selected_agent_id,
    parse_candidate_output,
    parse_dense_reference_output,
    parse_dense_retry_response,
)


_LEGACY_DENSE_COMPARE_KEYS = {
    "dense_selected_agent_id",
    "agreement_with_dense",
}


def _dense_timeout_seconds() -> float | None:
    raw_value = str(os.getenv("DENSE_REFERENCE_TIMEOUT_SEC", "0") or "0").strip()
    try:
        timeout_seconds = float(raw_value)
    except Exception:
        return None
    return timeout_seconds if timeout_seconds > 0 else None


def _candidate_blocks(
    candidate_agent_ids: Sequence[str],
    spatial_info: Dict[str, Any],
    *,
    dataset_name: str,
) -> str:
    adapter = create_dataset_adapter(dataset_name)
    rows: List[str] = []
    for agent_id in candidate_agent_ids:
        info = spatial_info.get(str(agent_id)) or {}
        raw_output = str(info.get("output") or "").strip()
        parsed = adapter.parse_candidate_output(raw_output, dict(info.get("metadata") or {}))
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
) -> list[Dict[str, str]]:
    adapter = create_dataset_adapter(dataset_name)
    return adapter.build_dense_compare_messages(
        question_text=question,
        candidate_agent_ids=candidate_agent_ids,
        candidate_blocks=_candidate_blocks(candidate_agent_ids, spatial_info, dataset_name=dataset_name),
    )


def _dense_retry_messages(
    *,
    question: str,
    candidate_agent_ids: Sequence[str],
    spatial_info: Dict[str, Any],
    previous_output: str,
    dataset_name: str,
) -> list[Dict[str, str]]:
    adapter = create_dataset_adapter(dataset_name)
    return adapter.build_dense_compare_messages(
        question_text=question,
        candidate_agent_ids=candidate_agent_ids,
        candidate_blocks=_candidate_blocks(candidate_agent_ids, spatial_info, dataset_name=dataset_name),
        retry_previous_output=previous_output,
    )


async def _run_with_optional_timeout(coro):
    timeout_seconds = _dense_timeout_seconds()
    if timeout_seconds is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


def _finalize_retry_probe(
    retry_raw_text: str,
    retry_retry_probe: Dict[str, Any],
    *,
    allowed_ids: Sequence[str],
) -> Dict[str, Any]:
    retry_parse_probe = parse_dense_reference_output(
        retry_raw_text,
        allowed_ids=allowed_ids,
    )
    selected_agent_id = normalize_selected_agent_id(
        retry_retry_probe.get("selected_agent_id"),
        allowed_ids=allowed_ids,
    )
    if selected_agent_id is None:
        return retry_parse_probe
    return {
        **retry_parse_probe,
        "selected_agent_id": selected_agent_id,
        "selected_answer": retry_retry_probe.get("selected_answer") or retry_parse_probe.get("selected_answer"),
        "parse_success": True,
        "failure_reason": None,
    }


def _empty_dense_compare_result(
    *,
    failure_reason: str | None,
    raw_text: str = "",
    retry_used: bool = False,
    retry_success: bool = False,
    retry_failure_reason: str | None = None,
    first_pass_failure_reason: str | None = None,
    first_pass_raw_text: str | None = None,
    retry_raw_text: str | None = None,
    parser_result: Dict[str, Any] | None = None,
    first_pass_parser_result: Dict[str, Any] | None = None,
    retry_parser_result: Dict[str, Any] | None = None,
    runtime_error_type: str | None = None,
    runtime_error_message: str | None = None,
) -> Dict[str, Any]:
    return {
        "dense_selected_agent_id": None,
        "dense_parse_success": False,
        "dense_parse_failure_reason": failure_reason,
        "dense_raw_judge_text": str(raw_text or ""),
        "dense_retry_used": bool(retry_used),
        "dense_retry_success": bool(retry_success),
        "dense_retry_failure_reason": retry_failure_reason,
        "dense_first_pass_failure_reason": first_pass_failure_reason,
        "dense_raw_response_before_retry": first_pass_raw_text,
        "dense_retry_raw_response": retry_raw_text,
        "dense_parser_result": parser_result,
        "dense_first_pass_parser_result": first_pass_parser_result,
        "dense_retry_parser_result": retry_parser_result,
        "dense_compare_source": "ori_structured_compare_dense",
        "dense_runtime_error_type": runtime_error_type,
        "dense_runtime_error_message": runtime_error_message,
    }


async def run_ori_structured_dense_compare(
    *,
    judge: Any,
    question: str,
    candidate_agent_ids: Sequence[str],
    spatial_info: Dict[str, Any],
) -> Dict[str, Any]:
    allowed_ids = [str(item) for item in candidate_agent_ids]
    dataset_name = str(getattr(judge, "domain", "mmlu") or "mmlu")
    if not allowed_ids or not spatial_info:
        return _empty_dense_compare_result(
            failure_reason="missing_selected_agent_header",
        )

    try:
        first_generation = await _run_with_optional_timeout(
            judge.llm.agen(
                _dense_messages(
                    question=question,
                    candidate_agent_ids=allowed_ids,
                    spatial_info=spatial_info,
                    dataset_name=dataset_name,
                ),
                max_tokens=12,
                agent_id=judge.id,
                agent_name=judge.agent_name,
                agent_role=judge.role,
            )
        )
    except asyncio.TimeoutError:
        return _empty_dense_compare_result(
            failure_reason="timeout",
        )
    except Exception as exc:
        return _empty_dense_compare_result(
            failure_reason="runtime_exception",
            runtime_error_type=type(exc).__name__,
            runtime_error_message=str(exc),
        )

    first_probe = parse_dense_reference_output(
        first_generation.text,
        allowed_ids=allowed_ids,
    )
    first_selected_agent_id = normalize_selected_agent_id(
        first_probe.get("selected_agent_id"),
        allowed_ids=allowed_ids,
    )
    if first_selected_agent_id is not None:
        return {
            "dense_selected_agent_id": first_selected_agent_id,
            "dense_parse_success": True,
            "dense_parse_failure_reason": None,
            "dense_raw_judge_text": first_generation.text,
            "dense_retry_used": False,
            "dense_retry_success": False,
            "dense_retry_failure_reason": None,
            "dense_first_pass_failure_reason": None,
            "dense_raw_response_before_retry": first_generation.text,
            "dense_retry_raw_response": None,
            "dense_parser_result": {
                **first_probe,
                "selected_agent_id": first_selected_agent_id,
                "parse_success": True,
                "failure_reason": None,
            },
            "dense_first_pass_parser_result": first_probe,
            "dense_retry_parser_result": None,
            "dense_compare_source": "ori_structured_compare_dense",
            "dense_runtime_error_type": None,
            "dense_runtime_error_message": None,
        }

    try:
        retry_generation = await _run_with_optional_timeout(
            judge.llm.agen(
                _dense_retry_messages(
                    question=question,
                    candidate_agent_ids=allowed_ids,
                    spatial_info=spatial_info,
                    previous_output=first_generation.text,
                    dataset_name=dataset_name,
                ),
                max_tokens=12,
                agent_id=judge.id,
                agent_name=judge.agent_name,
                agent_role=judge.role,
            )
        )
        retry_retry_probe = parse_dense_retry_response(
            retry_generation.text,
            allowed_ids=allowed_ids,
        )
        retry_selected_agent_id = normalize_selected_agent_id(
            retry_retry_probe.get("selected_agent_id"),
            allowed_ids=allowed_ids,
        )
        retry_parser_result = _finalize_retry_probe(
            retry_generation.text,
            retry_retry_probe,
            allowed_ids=allowed_ids,
        )
        if retry_selected_agent_id is not None:
            return {
                "dense_selected_agent_id": retry_selected_agent_id,
                "dense_parse_success": True,
                "dense_parse_failure_reason": None,
                "dense_raw_judge_text": retry_generation.text,
                "dense_retry_used": True,
                "dense_retry_success": True,
                "dense_retry_failure_reason": None,
                "dense_first_pass_failure_reason": first_probe.get("failure_reason"),
                "dense_raw_response_before_retry": first_generation.text,
                "dense_retry_raw_response": retry_generation.text,
                "dense_parser_result": retry_parser_result,
                "dense_first_pass_parser_result": first_probe,
                "dense_retry_parser_result": retry_parser_result,
                "dense_compare_source": "ori_structured_compare_dense",
                "dense_runtime_error_type": None,
                "dense_runtime_error_message": None,
            }
        return _empty_dense_compare_result(
            failure_reason=retry_retry_probe.get("failure_reason") or "retry_failed",
            raw_text=retry_generation.text,
            retry_used=True,
            retry_success=False,
            retry_failure_reason=retry_retry_probe.get("failure_reason") or "retry_failed",
            first_pass_failure_reason=first_probe.get("failure_reason"),
            first_pass_raw_text=first_generation.text,
            retry_raw_text=retry_generation.text,
            parser_result=retry_parser_result,
            first_pass_parser_result=first_probe,
            retry_parser_result=retry_parser_result,
        )
    except asyncio.TimeoutError:
        return _empty_dense_compare_result(
            failure_reason="timeout",
            raw_text=first_generation.text,
            retry_used=True,
            retry_success=False,
            retry_failure_reason="timeout",
            first_pass_failure_reason=first_probe.get("failure_reason"),
            first_pass_raw_text=first_generation.text,
            retry_raw_text=None,
            parser_result=first_probe,
            first_pass_parser_result=first_probe,
            retry_parser_result=None,
        )
    except Exception as exc:
        return _empty_dense_compare_result(
            failure_reason="runtime_exception",
            raw_text=first_generation.text,
            retry_used=True,
            retry_success=False,
            retry_failure_reason="runtime_exception",
            first_pass_failure_reason=first_probe.get("failure_reason"),
            first_pass_raw_text=first_generation.text,
            retry_raw_text=None,
            parser_result=first_probe,
            first_pass_parser_result=first_probe,
            retry_parser_result=None,
            runtime_error_type=type(exc).__name__,
            runtime_error_message=str(exc),
        )


def _strip_legacy_dense_header(text: str) -> str:
    lines = str(text or "").splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        if re.match(r"^\s*Selected agent id\s*\(\s*dense\s*\)\s*:\s*.*$", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def selected_choice_from_spatial_info(
    *,
    spatial_info: Dict[str, Any],
    selected_agent_id: str | None,
    dataset_name: str = "mmlu",
) -> str | None:
    selected = normalize_selected_agent_id(selected_agent_id)
    if selected is None:
        return None
    info = spatial_info.get(str(selected)) or {}
    adapter = create_dataset_adapter(str(dataset_name or "mmlu"))
    parsed = adapter.parse_candidate_output(
        info.get("output") if isinstance(info, dict) else "",
        dict(info.get("metadata") or {}) if isinstance(info, dict) else {},
    )
    value = parsed.get("final_answer", parsed.get("normalized_answer"))
    return None if value is None else str(value).strip()


def canonicalize_selected_result_text(
    result: Any,
    *,
    selected_agent_id: str | None,
    selected_choice: str | None,
    dataset_name: str = "mmlu",
) -> Any:
    if isinstance(result, GenerationResult):
        if selected_agent_id is None or selected_choice is None:
            clean_text = _strip_legacy_dense_header(result.text)
        else:
            adapter = create_dataset_adapter(str(dataset_name or "mmlu"))
            clean_text = adapter.format_selected_result_text(
                selected_agent_id=str(selected_agent_id),
                selected_answer=str(selected_choice),
            )
        return GenerationResult(
            text=clean_text,
            mode=result.mode,
            ttft=result.ttft,
            raw_output=result.raw_output,
            metadata=dict(result.metadata or {}),
        )
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], GenerationResult):
        message, generation = result
        return message, canonicalize_selected_result_text(
            generation,
            selected_agent_id=selected_agent_id,
            selected_choice=selected_choice,
            dataset_name=dataset_name,
        )
    return result


def attach_ori_protocol_metadata(
    result: Any,
    *,
    candidate_agent_ids: Sequence[str],
    dense_compare_result: Dict[str, Any] | None = None,
) -> Any:
    if isinstance(result, GenerationResult):
        metadata = dict(result.metadata or {})
        for key in _LEGACY_DENSE_COMPARE_KEYS:
            metadata.pop(key, None)
        metadata["candidate_agent_ids"] = [str(item) for item in candidate_agent_ids]
        if dense_compare_result is not None:
            metadata["ori_compare_dense_result"] = dict(dense_compare_result)
        return GenerationResult(
            text=_strip_legacy_dense_header(result.text),
            mode=result.mode,
            ttft=result.ttft,
            raw_output=result.raw_output,
            metadata=metadata,
        )
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], GenerationResult):
        message, generation = result
        return message, attach_ori_protocol_metadata(
            generation,
            candidate_agent_ids=candidate_agent_ids,
            dense_compare_result=dense_compare_result,
        )
    return result
