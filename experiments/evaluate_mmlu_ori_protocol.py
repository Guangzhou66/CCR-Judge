from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

from KVCOMM.graph.graph import Graph
from KVCOMM.utils.log import logger
from experiments.mmlu_jcr_utils import compute_jcr_fields
from dataset_adapters import create_dataset_adapter
from paper_repair.eval.parsing import (
    failure_reason_counts,
    normalize_choice,
    normalize_selected_agent_id,
)


ORI_PROTOCOL_CONSISTENCY_NOTE = (
    "Ori-style online protocol: candidate generation and final decision happen in the same online run; "
    "judge consistency compares reuse judge vs extra dense judge within the same run, using the same "
    "candidate ordering and online-generated candidate outputs."
)

ORI_ABLATION_METHOD_LABELS = {
    "OriProtocolCCRJudgeAblationFullFinalSelect": "CCR-Judge (Full)",
    "OriProtocolCCRJudgeNoComparativeGroupingFinalSelect": "CCR-Judge w/o Comparative Grouping",
    "OriProtocolCCRJudgeNoSupportScoringFinalSelect": "CCR-Judge w/o Support Scoring",
    "OriProtocolCCRJudgeNoShortlistCompressionFinalSelect": "CCR-Judge w/o Shortlist Compression",
    "OriProtocolCCRJudgeStatsOnlyFinalSelect": "CCR-Judge Stats Only",
    "OriProtocolCCRJudgeShortlistOnlyFinalSelect": "CCR-Judge Shortlist Only",
    "OriProtocolCCRJudgeNoComparativePayloadStatsFinalSelect": "CCR-Judge w/o Comparative Payload Statistics",
    "OriProtocolCCRJudgeRandomShortlistSeed42FinalSelect": "CCR-Judge with Random Shortlist (seed=42)",
    "OriProtocolCCRJudgeRandomShortlistSeed43FinalSelect": "CCR-Judge with Random Shortlist (seed=43)",
    "OriProtocolCCRJudgeRandomShortlistSeed44FinalSelect": "CCR-Judge with Random Shortlist (seed=44)",
}


def _method_name_from_decision_method(decision_method: str) -> str:
    if decision_method in ORI_ABLATION_METHOD_LABELS:
        return decision_method.replace("OriProtocolCCRJudge", "ori_ccr_").replace("FinalSelect", "").lower()
    if decision_method in {"OriProtocolTest3RepairFinalSelectBest", "OriProtocolCCRJudgeFinalSelect"}:
        return "ccr_judge"
    if decision_method == "OriProtocolFinalSelectBest":
        return "ori_protocol_baseline"
    return decision_method


def _method_label_from_decision_method(decision_method: str) -> str:
    if decision_method in ORI_ABLATION_METHOD_LABELS:
        return ORI_ABLATION_METHOD_LABELS[decision_method]
    if decision_method in {"OriProtocolTest3RepairFinalSelectBest", "OriProtocolCCRJudgeFinalSelect"}:
        return "CCR-Judge"
    if decision_method == "OriProtocolFinalSelectBest":
        return "Ori Protocol Baseline"
    return decision_method


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary_path_from_results(results_path: Optional[str]) -> Optional[str]:
    if not results_path:
        return None
    if results_path.endswith("_details.json"):
        return results_path[:-13] + "_summary.json"
    if results_path.endswith(".json"):
        return results_path[:-5] + "_summary.json"
    return results_path + "_summary.json"


def _main_table_path_from_results(results_path: Optional[str]) -> Optional[str]:
    if not results_path:
        return None
    return str(Path(results_path).with_name("main_table.md"))


def _normalize_metadata_blob(metadata: Any) -> Any:
    if isinstance(metadata, dict):
        return {str(key): _normalize_metadata_blob(value) for key, value in metadata.items()}
    if isinstance(metadata, list):
        return [_normalize_metadata_blob(value) for value in metadata]
    if isinstance(metadata, tuple):
        return [_normalize_metadata_blob(value) for value in metadata]
    if isinstance(metadata, (str, int, float, bool)) or metadata is None:
        return metadata
    return str(metadata)


def _candidate_payload(
    *,
    agent_id: str,
    role: str,
    output_text: str,
    metadata: Dict[str, Any],
    adapter: Any,
) -> Dict[str, Any]:
    parsed = adapter.parse_candidate_output(output_text, metadata)
    return {
        "agent_id": str(agent_id),
        "role": role,
        "text": output_text,
        "final_answer": parsed.get("final_answer", parsed.get("normalized_answer")),
        "normalized_answer": parsed.get("normalized_answer"),
        "metadata": _normalize_metadata_blob(metadata),
    }


def _node_output_for_task(node, task_key: str, mode: str) -> Tuple[str | None, Dict[str, Any]]:
    if mode == "allow_kv_reuse":
        outputs = getattr(node, "outputs", {}).get(task_key, []) if isinstance(getattr(node, "outputs", None), dict) else []
        outputs_meta = getattr(node, "outputs_meta", {}).get(task_key, []) if isinstance(getattr(node, "outputs_meta", None), dict) else []
        text = outputs[-1] if outputs else None
        meta = outputs_meta[-1] if outputs_meta else {}
        return text, dict(meta or {})
    outputs = getattr(node, "outputs", []) if isinstance(getattr(node, "outputs", None), list) else []
    outputs_meta = getattr(node, "outputs_meta", []) if isinstance(getattr(node, "outputs_meta", None), list) else []
    text = outputs[-1] if outputs else None
    meta = outputs_meta[-1] if outputs_meta else {}
    return text, dict(meta or {})


def _collect_candidate_outputs(
    *,
    realized_graph: Graph,
    task_key: str,
    mode: str,
    adapter: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node_id, node in realized_graph.nodes.items():
        output_text, output_meta = _node_output_for_task(node, task_key, mode)
        if output_text is None:
            continue
        rows.append(
            _candidate_payload(
                agent_id=str(node_id),
                role=str(getattr(node, "role", "") or ""),
                output_text=str(output_text),
                metadata=output_meta,
                adapter=adapter,
            )
        )
    rows.sort(key=lambda item: str(item["agent_id"]))
    return rows


def _main_table_row(summary_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": summary_payload.get("method"),
        "method_label": summary_payload.get("method_label"),
        "execution_mode": summary_payload.get("execution_mode"),
        "judge_shuffle": summary_payload.get("judge_shuffle"),
        "judge_compare_dense_enabled": summary_payload.get("judge_compare_dense_enabled"),
        "Acc": summary_payload.get("acc"),
        "JCR": summary_payload.get("JCR"),
        "parse_coverage": summary_payload.get("parse_coverage"),
        "total_cases": summary_payload.get("total_cases"),
    }


ORI_MAIN_TABLE_COLUMNS = [
    ("method_label", "Method"),
    ("execution_mode", "Execution Mode"),
    ("judge_shuffle", "Shuffle"),
    ("judge_compare_dense_enabled", "Compare Dense"),
    ("Acc", "Acc"),
    ("JCR", "JCR"),
    ("parse_coverage", "Parse Coverage"),
    ("total_cases", "total_cases"),
]


def _candidate_ordering_from_decision_meta(
    *,
    decision_meta: Any,
    candidate_outputs: Sequence[Dict[str, Any]],
) -> List[str]:
    candidate_ids = [str(item["agent_id"]) for item in candidate_outputs]
    if isinstance(decision_meta, list):
        for entry in decision_meta:
            if not isinstance(entry, dict):
                continue
            ordering = entry.get("candidate_agent_ids")
            if isinstance(ordering, list) and ordering:
                normalized = [str(item) for item in ordering if str(item) in candidate_ids]
                if normalized:
                    return normalized
    return candidate_ids


def _latest_decision_meta_entry(decision_meta: Any) -> Dict[str, Any]:
    if isinstance(decision_meta, list):
        for entry in reversed(decision_meta):
            if isinstance(entry, dict):
                return entry
        return {}
    if isinstance(decision_meta, dict):
        return decision_meta
    return {}


def _resolve_method_selected_agent_id(
    *,
    raw_text: Any,
    decision_meta: Any,
    candidate_outputs: Sequence[Dict[str, Any]],
    adapter: Any,
) -> str | None:
    metadata_entry = _latest_decision_meta_entry(decision_meta)
    metadata_selected_agent_id = metadata_entry.get("selected_agent_id")
    selected_agent_id = adapter.resolve_selected_agent_id(
        judge_text=raw_text,
        candidates=candidate_outputs,
        metadata_selected_agent_id=metadata_selected_agent_id,
    )
    return normalize_selected_agent_id(
        selected_agent_id,
        allowed_ids=[str(item["agent_id"]) for item in candidate_outputs],
    )


def _structured_dense_compare_result(
    *,
    decision_meta: Any,
    candidate_outputs: Sequence[Dict[str, Any]],
    judge_compare_dense_enabled: bool,
) -> Dict[str, Any]:
    empty_payload = {
        "dense_selected_agent_id": None,
        "dense_parse_success": False,
        "dense_parse_failure_reason": None,
        "dense_raw_judge_text": None,
        "dense_retry_used": False,
        "dense_retry_success": False,
        "dense_retry_failure_reason": None,
        "dense_first_pass_failure_reason": None,
        "dense_raw_response_before_retry": None,
        "dense_retry_raw_response": None,
        "dense_parser_result": None,
        "dense_first_pass_parser_result": None,
        "dense_retry_parser_result": None,
    }
    if not judge_compare_dense_enabled:
        return empty_payload

    metadata_entry = _latest_decision_meta_entry(decision_meta)
    structured = metadata_entry.get("ori_compare_dense_result")
    if not isinstance(structured, dict):
        return {
            **empty_payload,
            "dense_parse_failure_reason": "missing_structured_dense_compare_result",
        }

    candidate_ids = [str(item["agent_id"]) for item in candidate_outputs]
    dense_selected_agent_id = normalize_selected_agent_id(
        structured.get("dense_selected_agent_id"),
        allowed_ids=candidate_ids,
    )
    dense_parse_success = bool(dense_selected_agent_id is not None)
    return {
        **empty_payload,
        **dict(structured),
        "dense_selected_agent_id": dense_selected_agent_id,
        "dense_parse_success": dense_parse_success,
        "dense_parse_failure_reason": (
            None if dense_parse_success else structured.get("dense_parse_failure_reason")
        ),
    }


def _answer_consistent_disagreement(
    *,
    selected_agent_id: str | None,
    dense_selected_agent_id: str | None,
    final_answer: Any,
    dense_final_answer: Any,
    adapter: Any,
) -> bool:
    if selected_agent_id is None or dense_selected_agent_id is None:
        return False
    if str(selected_agent_id) == str(dense_selected_agent_id):
        return False
    return adapter.answer_consistency_match(final_answer, dense_final_answer)


def _annotate_candidate_upper_bound(
    *,
    candidate_outputs: Sequence[Dict[str, Any]],
    gold_answer: Any,
    adapter: Any,
) -> Dict[str, Any]:
    annotated_outputs: List[Dict[str, Any]] = []
    candidate_correct_agent_ids: List[str] = []
    normalized_answers: List[str] = []

    for candidate in candidate_outputs:
        candidate_final_answer = candidate.get("final_answer", candidate.get("normalized_answer"))
        evaluation = adapter.evaluate_prediction(candidate_final_answer, gold_answer)
        normalized_answer = evaluation.get("normalized_answer")
        candidate_correct = bool(evaluation.get("passed"))
        if normalized_answer is not None:
            normalized_answers.append(normalized_answer)
        if candidate_correct:
            candidate_correct_agent_ids.append(str(candidate.get("agent_id")))
        annotated_outputs.append(
            {
                **candidate,
                "final_answer": candidate_final_answer,
                "normalized_answer": normalized_answer,
                "candidate_correct": candidate_correct,
                "candidate_execution_result": dict(evaluation),
            }
        )

    candidate_answer_set_size = len(set(normalized_answers))
    num_candidates = len(candidate_outputs)
    num_correct_candidates = len(candidate_correct_agent_ids)
    return {
        "candidate_outputs": annotated_outputs,
        "num_correct_candidates": num_correct_candidates,
        "has_any_correct_candidate": bool(num_correct_candidates > 0),
        "all_candidates_wrong": bool(num_correct_candidates == 0),
        "all_candidates_same_answer": bool(
            num_candidates > 0
            and len(normalized_answers) == num_candidates
            and candidate_answer_set_size == 1
        ),
        "candidate_answer_set_size": candidate_answer_set_size,
        "candidate_correct_agent_ids": candidate_correct_agent_ids,
    }


def _candidate_execution_result_for_agent(
    *,
    candidate_outputs: Sequence[Dict[str, Any]],
    agent_id: str | None,
) -> Dict[str, Any] | None:
    if agent_id is None:
        return None
    normalized = str(agent_id)
    for candidate in candidate_outputs:
        if str(candidate.get("agent_id")) != normalized:
            continue
        execution_result = candidate.get("candidate_execution_result")
        if isinstance(execution_result, dict):
            return execution_result
    return None


def _markdown_table(columns: Sequence[tuple[str, str]], rows: Iterable[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    row_list = list(rows)
    if not row_list:
        lines.append("| " + " | ".join("-" for _ in columns) + " |")
        return "\n".join(lines)
    for row in row_list:
        rendered = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                rendered.append("-")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _iter_selected_records(
    dataset,
    *,
    limit_questions: Optional[int],
    question_indices: Optional[Sequence[int]] = None,
) -> List[Tuple[int, Any]]:
    wanted = {int(item) for item in question_indices} if question_indices else None
    selected: List[Tuple[int, Any]] = []
    for i_record, record in enumerate(dataset):
        if wanted is not None:
            if i_record not in wanted:
                continue
        selected.append((i_record, record))
        if limit_questions is not None and len(selected) >= limit_questions:
            break
    return selected


def _batched(items: Sequence[Tuple[int, Any]], batch_size: int) -> Iterator[List[Tuple[int, Any]]]:
    batch: List[Tuple[int, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def evaluate_ori_protocol(
    graph: Graph,
    dataset,
    *,
    limit_questions: Optional[int] = None,
    question_indices: Optional[Sequence[int]] = None,
    eval_batch_size: int = 1,
    mode: str = "default",
    execution_mode: str = "default",
    judge_compare_dense_enabled: bool = False,
    judge_shuffle: bool = False,
    topology_mode: str = "FullConnected",
    decision_method: str = "OriProtocolFinalSelectBest",
    results_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    main_table_path: Optional[str] = None,
    dataset_name: str = "mmlu",
    dataset_adapter: Any | None = None,
    **kwargs,
) -> Dict[str, Any]:
    logger.info(
        "Evaluating Ori-style protocol on {} split {}",
        dataset.__class__.__name__,
        dataset.split,
    )

    adapter = dataset_adapter or create_dataset_adapter(dataset_name)
    num_correct = 0
    per_question: List[Dict[str, Any]] = []
    selected_records = _iter_selected_records(
        dataset,
        limit_questions=limit_questions,
        question_indices=question_indices,
    )
    num_batches = int(math.ceil(len(selected_records) / eval_batch_size)) if selected_records else 0

    judge_consistency_num = 0
    judge_consistency_den = 0
    selected_parse_count = 0
    dense_parse_count = 0

    async def _run_record(record_index: int, record: Any, batch_index: int) -> Tuple[int, Any, Dict[str, Any], Graph]:
        realized_graph = copy.deepcopy(graph)
        realized_graph.spatial_logits = graph.spatial_logits
        realized_graph.temporal_logits = graph.temporal_logits
        input_dict = dataset.record_to_input(record)
        input_dict["_batch_index"] = batch_index
        mode_kwargs: Dict[str, Any] = {}
        if mode == "allow_kv_reuse":
            mode_kwargs = dict(kwargs)
        if "judge_mask_prev_agent_attn" in kwargs:
            mode_kwargs["judge_mask_prev_agent_attn"] = kwargs["judge_mask_prev_agent_attn"]
        run_result = await realized_graph.arun(
            input_dict,
            1,
            mode=mode,
            **mode_kwargs,
        )
        return record_index, record, run_result, realized_graph

    for i_batch, record_batch in tqdm(
        enumerate(_batched(selected_records, eval_batch_size)),
        total=num_batches,
    ):
        logger.info("{}", "-" * 80)
        start_ts = time.time()
        tasks: List[asyncio.Task[Tuple[int, Any, Dict[str, Any], Graph]]] = []

        for record_index, record in record_batch:
            tasks.append(asyncio.create_task(_run_record(record_index, record, i_batch)))

        batch_results = await asyncio.gather(*tasks)
        logger.opt(colors=True).info(
            f"<blue>[BATCH TIME]</blue> {time.time() - start_ts:.3f}s"
        )

        for record_index, record, run_result, realized_graph in batch_results:
            task_key = dataset.record_to_input(record).get("task")
            raw_answer = run_result.get("answers", [])
            decision_meta = run_result.get("decision_metadata")
            raw_text = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer
            candidate_outputs = _collect_candidate_outputs(
                realized_graph=realized_graph,
                task_key=task_key,
                mode=mode,
                adapter=adapter,
            )
            candidate_ordering = _candidate_ordering_from_decision_meta(
                decision_meta=decision_meta,
                candidate_outputs=candidate_outputs,
            )
            selected_agent_id = _resolve_method_selected_agent_id(
                raw_text=raw_text,
                decision_meta=decision_meta,
                candidate_outputs=candidate_outputs,
                adapter=adapter,
            )
            dense_compare_result = _structured_dense_compare_result(
                decision_meta=decision_meta,
                candidate_outputs=candidate_outputs,
                judge_compare_dense_enabled=bool(judge_compare_dense_enabled),
            )
            dense_selected_agent_id = dense_compare_result.get("dense_selected_agent_id")
            judge_fields = compute_jcr_fields(selected_agent_id, dense_selected_agent_id)
            parse_success = bool(judge_fields["selected_agent_id"] is not None)
            dense_parse_success = bool(judge_fields["dense_selected_agent_id"] is not None)
            selected_parse_count += int(parse_success)
            dense_parse_count += int(dense_parse_success)
            if judge_fields["jcr_den"]:
                judge_consistency_den += 1
                judge_consistency_num += int(judge_fields["jcr_num"])

            answer = await dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            candidate_audit = _annotate_candidate_upper_bound(
                candidate_outputs=candidate_outputs,
                gold_answer=correct_answer,
                adapter=adapter,
            )
            candidate_outputs = candidate_audit["candidate_outputs"]
            selected_execution_result = _candidate_execution_result_for_agent(
                candidate_outputs=candidate_outputs,
                agent_id=judge_fields["selected_agent_id"],
            )
            dense_execution_result = _candidate_execution_result_for_agent(
                candidate_outputs=candidate_outputs,
                agent_id=judge_fields["dense_selected_agent_id"],
            )
            official_scoring = await adapter.score_official(
                final_answer=answer,
                target=correct_answer,
                selected_execution_result=selected_execution_result,
            )
            is_correct = bool(official_scoring.get("official_is_correct"))
            num_correct += int(is_correct)
            dense_final_answer = adapter.selected_answer_for_agent(candidate_outputs, dense_selected_agent_id)
            answer_consistent_disagreement = _answer_consistent_disagreement(
                selected_agent_id=judge_fields["selected_agent_id"],
                dense_selected_agent_id=judge_fields["dense_selected_agent_id"],
                final_answer=answer,
                dense_final_answer=dense_final_answer,
                adapter=adapter,
            )

            per_question.append(
                {
                    "question_id": f"{dataset.split}:{record_index}",
                    "record_index": int(record_index),
                    "task": task_key,
                    "execution_mode": execution_mode,
                    "judge_compare_dense_enabled": bool(judge_compare_dense_enabled),
                    "judge_shuffle": bool(judge_shuffle),
                    "topology_mode": topology_mode,
                    "decision_method": decision_method,
                    "candidate_outputs": candidate_outputs,
                    "candidate_ordering": list(candidate_ordering),
                    "num_correct_candidates": candidate_audit["num_correct_candidates"],
                    "has_any_correct_candidate": candidate_audit["has_any_correct_candidate"],
                    "all_candidates_wrong": candidate_audit["all_candidates_wrong"],
                    "all_candidates_same_answer": candidate_audit["all_candidates_same_answer"],
                    "candidate_answer_set_size": candidate_audit["candidate_answer_set_size"],
                    "candidate_correct_agent_ids": candidate_audit["candidate_correct_agent_ids"],
                    "selected_agent_id": judge_fields["selected_agent_id"],
                    "dense_selected_agent_id": judge_fields["dense_selected_agent_id"],
                    "selected_agent_parse_success": parse_success,
                    "dense_selected_agent_parse_success": dense_parse_success,
                    "final_result_text": raw_text,
                    "final_answer": answer,
                    "final_answer_text": answer,
                    "dense_final_answer": dense_final_answer,
                    "correct_answer": correct_answer,
                    "official_is_correct": is_correct,
                    "correct": is_correct,
                    "scoring_branch": official_scoring.get("scoring_branch"),
                    "fallback_used": bool(official_scoring.get("fallback_used")),
                    "selected_candidate_passed": official_scoring.get("selected_candidate_passed"),
                    "fallback_correctness_result": official_scoring.get("fallback_correctness_result"),
                    "selected_execution_result": selected_execution_result,
                    "dense_execution_result": dense_execution_result,
                    "parse_success": parse_success,
                    "dense_parse_success": dense_parse_success,
                    "dense_parse_failure_reason": dense_compare_result.get("dense_parse_failure_reason"),
                    "dense_raw_judge_text": dense_compare_result.get("dense_raw_judge_text"),
                    "dense_retry_used": bool(dense_compare_result.get("dense_retry_used")),
                    "dense_retry_success": bool(dense_compare_result.get("dense_retry_success")),
                    "dense_retry_failure_reason": dense_compare_result.get("dense_retry_failure_reason"),
                    "dense_first_pass_failure_reason": dense_compare_result.get("dense_first_pass_failure_reason"),
                    "dense_raw_response_before_retry": dense_compare_result.get("dense_raw_response_before_retry"),
                    "dense_retry_raw_response": dense_compare_result.get("dense_retry_raw_response"),
                    "dense_parser_result": dense_compare_result.get("dense_parser_result"),
                    "dense_first_pass_parser_result": dense_compare_result.get("dense_first_pass_parser_result"),
                    "dense_retry_parser_result": dense_compare_result.get("dense_retry_parser_result"),
                    "jcr_match": judge_fields["judge_agree"],
                    "judge_consistency_match": judge_fields["judge_agree"],
                    "answer_consistent_disagreement": answer_consistent_disagreement,
                    "selection_gap": int(bool(candidate_audit["has_any_correct_candidate"])) - int(bool(is_correct)),
                    **{
                        key: value
                        for key, value in official_scoring.items()
                        if key not in {
                            "official_is_correct",
                            "scoring_branch",
                            "fallback_used",
                            "selected_candidate_passed",
                            "fallback_correctness_result",
                        }
                    },
                    "metadata": _normalize_metadata_blob(decision_meta),
                }
            )

        if per_question:
            logger.opt(colors=True).info(
                "<blue>[ACCURACY]</blue> {:.6%} ({}/{})",
                num_correct / len(per_question),
                num_correct,
                len(per_question),
            )

    judge_consistency_rate = (
        (judge_consistency_num / judge_consistency_den)
        if judge_consistency_den
        else None
    )
    total_cases = len(per_question)
    parse_coverage = (selected_parse_count / total_cases) if total_cases else None
    dense_parse_coverage = (dense_parse_count / total_cases) if total_cases else None
    compare_dense_parse_coverage = (judge_consistency_den / total_cases) if total_cases else None
    any_correct_candidate_count = sum(int(bool(row.get("has_any_correct_candidate"))) for row in per_question)
    all_candidates_wrong_count = sum(int(bool(row.get("all_candidates_wrong"))) for row in per_question)
    all_candidates_same_answer_count = sum(int(bool(row.get("all_candidates_same_answer"))) for row in per_question)
    avg_num_correct_candidates = (
        sum(int(row.get("num_correct_candidates") or 0) for row in per_question) / total_cases
        if total_cases
        else None
    )

    summary_payload = {
        "protocol_name": "ori_style_online_protocol",
        "protocol_note": ORI_PROTOCOL_CONSISTENCY_NOTE,
        "method": _method_name_from_decision_method(decision_method),
        "method_label": _method_label_from_decision_method(decision_method),
        "execution_mode": execution_mode,
        "judge_compare_dense_enabled": bool(judge_compare_dense_enabled),
        "judge_shuffle": bool(judge_shuffle),
        "topology_mode": topology_mode,
        "decision_method": decision_method,
        "total_cases": total_cases,
        "acc": (num_correct / total_cases) if total_cases else 0.0,
        "parse_coverage": parse_coverage,
        "selected_agent_id_parse_count": selected_parse_count,
        "dense_parse_coverage": dense_parse_coverage,
        "dense_selected_agent_id_parse_count": dense_parse_count,
        "jcr_num": judge_consistency_num,
        "jcr_den": judge_consistency_den,
        "JCR": judge_consistency_rate,
        "judge_consistency_num": judge_consistency_num,
        "judge_consistency_den": judge_consistency_den,
        "judge_consistency_rate": judge_consistency_rate,
        "compare_dense_parse_coverage": compare_dense_parse_coverage,
        "judge_consistency_semantics": (
            "ori_style_intra_run_compare_dense_consistency;"
            " compares reuse judge selection vs extra dense judge selection within the same run;"
            " denominator keeps only cases where both selected_agent_id and dense_selected_agent_id parse"
        ),
        "dense_parse_retry_used_count": sum(int(bool(row.get("dense_retry_used"))) for row in per_question),
        "dense_parse_retry_success_count": sum(int(bool(row.get("dense_retry_success"))) for row in per_question),
        "dense_parse_failure_reason_counts": failure_reason_counts(
            per_question,
            key="dense_parse_failure_reason",
        ),
        "dense_retry_failure_reason_counts": failure_reason_counts(
            per_question,
            key="dense_retry_failure_reason",
        ),
        "answer_consistent_disagreement_num": sum(
            int(bool(row.get("answer_consistent_disagreement"))) for row in per_question
        ),
        "answer_consistent_disagreement_rate": (
            sum(int(bool(row.get("answer_consistent_disagreement"))) for row in per_question) / judge_consistency_den
            if judge_consistency_den
            else None
        ),
        "fallback_rate": (
            sum(int(bool(row.get("fallback_used"))) for row in per_question) / total_cases if total_cases else None
        ),
        "any_correct_candidate_count": any_correct_candidate_count,
        "any_correct_candidate_rate": (
            any_correct_candidate_count / total_cases if total_cases else None
        ),
        "all_candidates_wrong_count": all_candidates_wrong_count,
        "all_candidates_wrong_rate": (
            all_candidates_wrong_count / total_cases if total_cases else None
        ),
        "all_candidates_same_answer_count": all_candidates_same_answer_count,
        "all_candidates_same_answer_rate": (
            all_candidates_same_answer_count / total_cases if total_cases else None
        ),
        "avg_num_correct_candidates": avg_num_correct_candidates,
        "online_candidate_upper_bound_acc": (
            any_correct_candidate_count / total_cases if total_cases else None
        ),
        "selection_gap": (
            ((any_correct_candidate_count / total_cases) - (num_correct / total_cases)) if total_cases else None
        ),
        "online_candidate_upper_bound_semantics": (
            "upper bound over the online candidate slate; "
            "counts a case as reachable when at least one candidate answer matches gold after dataset normalization"
        ),
    }

    if results_path is not None:
        _json_dump(Path(results_path), per_question)
        logger.opt(colors=True).info("<blue>[RESULTS SAVED]</blue> {}", results_path)
    summary_target = summary_path or _summary_path_from_results(results_path)
    if summary_target is not None:
        _json_dump(Path(summary_target), summary_payload)
        logger.opt(colors=True).info("<blue>[SUMMARY SAVED]</blue> {}", summary_target)
    table_target = main_table_path or _main_table_path_from_results(results_path)
    if table_target is not None:
        main_table = "# Ori-Style MMLU Protocol Main Table\n\n" + _markdown_table(
            ORI_MAIN_TABLE_COLUMNS,
            [_main_table_row(summary_payload)],
        )
        Path(table_target).write_text(main_table.rstrip() + "\n", encoding="utf-8")
        logger.opt(colors=True).info("<blue>[MAIN TABLE SAVED]</blue> {}", table_target)

    logger.info("Ori-style evaluation complete")
    return summary_payload
