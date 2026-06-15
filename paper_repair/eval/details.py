from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from dataset_adapters import create_dataset_adapter
from dataset_adapters.gsm8k_adapter import _extract_numeric_answer, _extract_ori_parity_number_sync
from paper_repair.eval.jcr import compute_jcr_fields
from paper_repair.eval.parsing import (
    extract_mmlu_choice_letter,
    normalize_choice,
    strip_selected_agent_header,
)
from paper_repair.repair.ccr_types import CaseEvaluationRow, FrozenCandidateRecord, JudgeMethodResult


PROTECTED_AUTHORITATIVE_FIELDS = {
    "record_index",
    "task_id",
    "question_text",
    "gold_answer",
    "generation_regime",
    "candidate_count",
    "candidate_agent_ids",
    "permutation",
    "selected_agent_id",
    "selected_answer",
    "final_answer",
    "parse_success",
    "correct",
    "raw_judge_text",
    "reuse_mode",
    "ttft",
    "method",
    "dense_selected_agent_id",
    "jcr_match",
    "fallback_used",
    "fallback_reason",
}


def _inverse_permutation(permutation: list[str]) -> Dict[str, int]:
    return {str(agent_id): int(position) for position, agent_id in enumerate(permutation)}


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except Exception:
        return False


def _selected_candidate_index(record: FrozenCandidateRecord, agent_id: str | None) -> int | None:
    if agent_id is None:
        return None
    for idx, candidate in enumerate(record.candidates):
        if str(candidate.agent_id) == str(agent_id):
            return idx
    return None


def _adapter_for_record(record: FrozenCandidateRecord):
    adapter_kwargs: Dict[str, Any] = {}
    if record.dataset_config:
        adapter_kwargs["config_name"] = record.dataset_config
    if isinstance(record.target_payload, dict) and isinstance(record.target_payload.get("dataset_json"), str):
        adapter_kwargs["dataset_json"] = record.target_payload["dataset_json"]
    return create_dataset_adapter(record.dataset_name, **adapter_kwargs)


def _target_for_record(record: FrozenCandidateRecord) -> Any:
    if record.dataset_name == "humaneval":
        return dict(record.target_payload or {})
    return record.gold_answer


def _candidate_execution_result(
    *,
    record: FrozenCandidateRecord,
    candidate_index: int,
) -> Dict[str, Any]:
    adapter = _adapter_for_record(record)
    candidate = record.candidates[candidate_index]
    parsed = adapter.parse_candidate_output(candidate.text, dict(candidate.metadata or {}))
    final_answer = parsed.get("final_answer", parsed.get("normalized_answer"))
    evaluation = adapter.evaluate_prediction(final_answer, _target_for_record(record))
    normalized_answer = evaluation.get("normalized_answer")
    if normalized_answer is None:
        normalized_answer = adapter.normalize_prediction_answer(final_answer)
    return {
        **dict(evaluation),
        "agent_id": candidate.agent_id,
        "final_answer": final_answer,
        "normalized_answer": normalized_answer,
        "candidate_correct": bool(evaluation.get("passed")),
    }


def _candidate_diagnostics(record: FrozenCandidateRecord) -> Dict[str, Any]:
    candidate_texts = [candidate.text for candidate in record.candidates]
    candidate_execution_results = [
        _candidate_execution_result(record=record, candidate_index=idx)
        for idx, _candidate in enumerate(record.candidates)
    ]
    candidate_passed_list = [
        bool(result.get("passed"))
        for result in candidate_execution_results
    ]
    num_correct_candidates = sum(int(flag) for flag in candidate_passed_list)
    return {
        "candidate_texts": candidate_texts,
        "candidate_execution_results": candidate_execution_results,
        "candidate_passed_list": candidate_passed_list,
        "num_correct_candidates": num_correct_candidates,
        "any_candidate_upper_bound": bool(num_correct_candidates > 0),
        "all_wrong_flag": bool(num_correct_candidates == 0),
    }


def _official_scoring_from_text(
    *,
    record: FrozenCandidateRecord,
    response_text: str,
    selected_execution_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    adapter = _adapter_for_record(record)
    final_answer_text = strip_selected_agent_header(response_text).strip()
    selected_candidate_passed = (
        None
        if not isinstance(selected_execution_result, dict)
        else bool(selected_execution_result.get("passed"))
    )

    if record.dataset_name == "mmlu":
        target_choice = normalize_choice(record.gold_answer) or "A"
        predicted_choice = extract_mmlu_choice_letter(response_text)
        normalized_predicted_choice = normalize_choice(predicted_choice)
        if isinstance(selected_execution_result, dict):
            official_is_correct = bool(selected_execution_result.get("passed"))
            scoring_branch = "selected_candidate_passed"
            fallback_used = False
            fallback_correctness_result = None
        else:
            official_is_correct = normalized_predicted_choice == target_choice
            scoring_branch = "fallback_final_answer"
            fallback_used = True
            fallback_correctness_result = adapter.evaluate_prediction(final_answer_text, _target_for_record(record))
        return {
            "scorer_mode": adapter.official_scorer_mode(),
            "scoring_branch": scoring_branch,
            "fallback_used": fallback_used,
            "selected_candidate_passed": selected_candidate_passed,
            "fallback_correctness_result": fallback_correctness_result,
            "official_is_correct": bool(official_is_correct),
            "target_choice": target_choice,
            "predicted_choice": predicted_choice,
            "normalized_predicted_choice": normalized_predicted_choice,
            "final_answer_text": final_answer_text,
        }

    if record.dataset_name == "gsm8k":
        official_number = _extract_ori_parity_number_sync(final_answer_text)
        diagnostic_number = _extract_numeric_answer(final_answer_text)
        target_number = str(record.gold_answer).strip() or "0"
        official_is_correct = _float_equal(official_number, target_number)
        return {
            "scorer_mode": "ori_parity",
            "scoring_branch": "ori_parity_final_text",
            "fallback_used": False,
            "selected_candidate_passed": selected_candidate_passed,
            "fallback_correctness_result": None,
            "official_is_correct": bool(official_is_correct),
            "official_gsm8k_number": official_number,
            "extracted_final_number": official_number,
            "diagnostic_gsm8k_number": diagnostic_number,
            "normalized_final_answer": diagnostic_number,
            "final_answer_text": final_answer_text,
        }

    if record.dataset_name == "humaneval":
        evaluation = adapter.evaluate_prediction(response_text, dict(record.target_payload or {}))
        return {
            "scorer_mode": "ori_pyexecutor",
            "scoring_branch": "ori_pyexecutor_final_code",
            "fallback_used": False,
            "selected_candidate_passed": selected_candidate_passed,
            "fallback_correctness_result": None,
            "official_is_correct": bool(evaluation.get("official_humaneval_passed")),
            "official_humaneval_passed": bool(evaluation.get("official_humaneval_passed")),
            "subprocess_humaneval_passed": bool(evaluation.get("subprocess_humaneval_passed")),
            "executor_mode": "ori_pyexecutor",
            "extracted_code": evaluation.get("extracted_code"),
            "execution_error_type": evaluation.get("execution_error_type"),
            "final_answer_text": evaluation.get("extracted_code"),
        }

    if isinstance(selected_execution_result, dict):
        official_is_correct = bool(selected_execution_result.get("passed"))
        scoring_branch = "selected_candidate_passed"
        fallback_used = False
        fallback_correctness_result = None
    else:
        fallback_eval = adapter.evaluate_prediction(final_answer_text, _target_for_record(record))
        official_is_correct = bool(fallback_eval.get("passed"))
        scoring_branch = "fallback_final_answer"
        fallback_used = True
        fallback_correctness_result = fallback_eval
    return {
        "scorer_mode": f"{record.dataset_name}_final_text",
        "scoring_branch": scoring_branch,
        "fallback_used": fallback_used,
        "selected_candidate_passed": selected_candidate_passed,
        "fallback_correctness_result": fallback_correctness_result,
        "official_is_correct": bool(official_is_correct),
        "final_answer_text": final_answer_text,
    }


def merge_metadata_without_overwriting_authoritative_fields(
    *,
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
    protected_fields: set[str] | None = None,
) -> Dict[str, Any]:
    protected = protected_fields or PROTECTED_AUTHORITATIVE_FIELDS
    merged = dict(payload)
    dropped: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key in protected:
            dropped[key] = value
            continue
        merged[key] = value
    if dropped:
        merged["protected_metadata_fields_ignored"] = sorted(dropped.keys())
    return merged


def build_case_evaluation_row(
    *,
    record: FrozenCandidateRecord,
    method_result: JudgeMethodResult,
    dense_reference_result: JudgeMethodResult | None = None,
    judge_shuffle: bool = False,
) -> CaseEvaluationRow:
    dense_selected_agent_id = (
        dense_reference_result.selected_agent_id
        if dense_reference_result is not None and dense_reference_result.parse_success
        else (method_result.selected_agent_id if method_result.parse_success else None)
    )
    jcr_payload = compute_jcr_fields(
        method_result.selected_agent_id,
        dense_selected_agent_id,
    )
    candidate_diag = _candidate_diagnostics(record)
    adapter = _adapter_for_record(record)
    dense_selected_answer = adapter.selected_answer_for_agent(record.candidates, dense_selected_agent_id)
    answer_consistent_disagreement_flag = bool(
        method_result.selected_agent_id is not None
        and dense_selected_agent_id is not None
        and str(method_result.selected_agent_id) != str(dense_selected_agent_id)
        and method_result.selected_answer is not None
        and dense_selected_answer is not None
        and adapter.answer_consistency_match(method_result.selected_answer, dense_selected_answer)
    )
    selected_candidate_index = _selected_candidate_index(record, method_result.selected_agent_id)
    selected_execution_result = (
        candidate_diag["candidate_execution_results"][selected_candidate_index]
        if selected_candidate_index is not None
        else None
    )
    official_scoring = _official_scoring_from_text(
        record=record,
        response_text=method_result.raw_judge_text,
        selected_execution_result=selected_execution_result,
    )
    correct = bool(official_scoring.get("official_is_correct"))
    selected_candidate_passed = (
        candidate_diag["candidate_passed_list"][selected_candidate_index]
        if selected_candidate_index is not None
        else None
    )
    ccr_specific_metadata = {
        key: value
        for key, value in (method_result.metadata or {}).items()
        if key.startswith("ccr_")
        or key.startswith("repair_")
        or key.startswith("interaction_repair_")
    }
    return CaseEvaluationRow(
        record_index=record.record_index,
        method_result=method_result,
        dense_selected_agent_id=dense_selected_agent_id,
        jcr_match=jcr_payload["judge_agree"],
        correct=correct,
        permutation=list(record.permutation),
        task_id=record.task_id,
        question_text=record.question_text,
        gold_answer=record.gold_answer,
        generation_regime=record.generation_regime,
        candidate_count=record.candidate_count,
        candidate_agent_ids=record.candidate_agent_ids(),
        dataset_name=record.dataset_name,
        official_scoring=official_scoring,
        inverse_permutation=_inverse_permutation(list(record.permutation)),
        shuffle_flag=bool(judge_shuffle),
        selected_candidate_passed=selected_candidate_passed,
        candidate_texts=list(candidate_diag["candidate_texts"]),
        candidate_passed_list=list(candidate_diag["candidate_passed_list"]),
        num_correct_candidates=int(candidate_diag["num_correct_candidates"]),
        any_candidate_upper_bound=bool(candidate_diag["any_candidate_upper_bound"]),
        all_wrong_flag=bool(candidate_diag["all_wrong_flag"]),
        answer_consistent_disagreement_flag=answer_consistent_disagreement_flag,
        ccr_specific_metadata=ccr_specific_metadata,
    )


def case_evaluation_row_to_dict(
    row: CaseEvaluationRow,
    *,
    pack_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    method_result = row.method_result
    metadata = dict(method_result.metadata or {})
    raw_generation_metadata = dict(metadata.pop("raw_generation_metadata", {}) or {})
    official_scoring = dict(row.official_scoring or {})
    final_answer_text = official_scoring.get("final_answer_text")
    selected_agent_id_raw = metadata.get("selected_agent_id")
    dense_selected_agent_id_raw = (
        row.dense_selected_agent_id
        if row.dense_selected_agent_id is not None
        else None
    )
    candidate_text_hash = hashlib.sha256(
        json.dumps(list(row.candidate_texts), ensure_ascii=False, sort_keys=False).encode("utf-8")
    ).hexdigest()
    dense_recompute_candidate_block_count = int(row.candidate_count) if method_result.method_name == "dense" else 0
    reuse_candidate_block_count = int(row.candidate_count) if method_result.method_name != "dense" else 0
    answer_jcr_if_available = None
    payload = {
        "dataset": row.dataset_name,
        "sample_id": row.task_id or f"{row.dataset_name}:{int(row.record_index)}",
        "question_id": row.task_id or f"{row.dataset_name}:{int(row.record_index)}",
        "record_index": int(row.record_index),
        "task_id": row.task_id,
        "question_text": row.question_text,
        "gold_answer": row.gold_answer,
        "generation_regime": row.generation_regime,
        "regime": row.generation_regime,
        "shuffle_flag": bool(row.shuffle_flag),
        "candidate_count": int(row.candidate_count),
        "candidate_agent_ids": list(row.candidate_agent_ids),
        "permutation": list(row.permutation),
        "inverse_permutation": dict(row.inverse_permutation),
        "selected_agent_id_raw": selected_agent_id_raw if selected_agent_id_raw is not None else method_result.selected_agent_id,
        "selected_agent_id": method_result.selected_agent_id,
        "selected_agent_id_original_space": method_result.selected_agent_id,
        "selected_agent_parse_success": bool(method_result.parse_success),
        "dense_selected_agent_id_raw": dense_selected_agent_id_raw,
        "selected_answer": method_result.selected_answer,
        "final_answer": final_answer_text if final_answer_text is not None else method_result.selected_answer,
        "final_result_text": method_result.raw_judge_text,
        "final_answer_text": final_answer_text if final_answer_text is not None else method_result.selected_answer,
        "parse_success": bool(method_result.parse_success),
        "selected_agent_id_parse_count": int(bool(method_result.parse_success)),
        "parse_coverage": 1.0 if method_result.parse_success else 0.0,
        "correct": bool(row.correct),
        "raw_judge_text": method_result.raw_judge_text,
        "reuse_metadata": raw_generation_metadata,
        "reuse_mode": method_result.reuse_mode,
        "reuse_rate": 1.0 if method_result.reuse_mode == "kv_reuse" else 0.0,
        "reuse_candidate_block_count": reuse_candidate_block_count,
        "dense_recompute_candidate_block_count": dense_recompute_candidate_block_count,
        "ttft": method_result.ttft,
        "method": method_result.method_name,
        "dense_selected_agent_id": row.dense_selected_agent_id,
        "dense_selected_agent_id_original_space": row.dense_selected_agent_id,
        "dense_selected_agent_parse_success": bool(row.dense_selected_agent_id is not None),
        "jcr_match": row.jcr_match,
        "fallback_used": bool(method_result.fallback_used),
        "fallback_reason": method_result.fallback_reason,
        "selected_candidate_passed": row.selected_candidate_passed,
        "candidate_texts": list(row.candidate_texts),
        "candidate_text_hash": candidate_text_hash,
        "candidate_passed_list": list(row.candidate_passed_list),
        "num_correct_candidates": int(row.num_correct_candidates),
        "any_candidate_upper_bound": bool(row.any_candidate_upper_bound),
        "all_wrong_flag": bool(row.all_wrong_flag),
        "selection_gap": int(bool(row.any_candidate_upper_bound)) - int(bool(row.correct)),
        "answer_consistent_disagreement_flag": bool(row.answer_consistent_disagreement_flag),
        "answer_consistent_disagreement_rate": 1.0 if row.answer_consistent_disagreement_flag else 0.0,
        "answer_jcr_if_available": answer_jcr_if_available,
        "ccr_specific_metadata": dict(row.ccr_specific_metadata),
    }
    payload.update(official_scoring)
    payload["official_scoring_policy"] = official_scoring.get("scorer_mode")
    payload["official_scoring_fallback_used"] = bool(official_scoring.get("fallback_used"))
    payload["fallback_used"] = bool(method_result.fallback_used)
    payload["kvcomm_reuse_event"] = bool(method_result.method_name == "kvcomm" and payload["reuse_rate"] > 0.0)
    payload["pal_kv_reuse_event"] = bool(method_result.method_name == "pal_kv" and payload["reuse_rate"] > 0.0)
    if method_result.method_name == "dense":
        payload["anchor_pool_scope"] = "none"
        payload["gating_decision"] = "dense_prefill_no_reuse"
    elif method_result.method_name == "pal_kv":
        payload["anchor_pool_scope"] = "group_judge_anchors"
        payload["gating_decision"] = "group_anchor_enabled"
    elif method_result.method_name == "kvcomm":
        payload["anchor_pool_scope"] = "kvcomm_default"
        payload["gating_decision"] = "judge_kv_reuse_enabled"
    elif method_result.method_name == "naive_reuse":
        payload["anchor_pool_scope"] = "naive_placeholder_reuse"
        payload["gating_decision"] = "judge_kv_reuse_enabled"
    else:
        payload["anchor_pool_scope"] = "kvcomm_default_ccr"
        payload["gating_decision"] = "judge_kv_reuse_enabled_ccr"
    if pack_metadata:
        payload.update(
            {
                "candidate_pack_hash": pack_metadata.get("candidate_pack_hash"),
                "candidate_build_mode": pack_metadata.get("candidate_build_mode"),
                "execution_side_reuse": pack_metadata.get("execution_side_reuse"),
            }
        )
    return merge_metadata_without_overwriting_authoritative_fields(
        payload=payload,
        metadata=metadata,
    )
