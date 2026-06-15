from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from paper_repair.eval.parsing import normalize_selected_agent_id


def compute_jcr_fields(
    selected_agent_id: Any,
    dense_selected_agent_id: Any,
) -> Dict[str, Any]:
    selected = normalize_selected_agent_id(selected_agent_id)
    dense = normalize_selected_agent_id(dense_selected_agent_id)
    judge_agree = None
    jcr_num = 0
    jcr_den = 0
    if selected is not None and dense is not None:
        jcr_den = 1
        judge_agree = bool(selected == dense)
        jcr_num = int(judge_agree)
    return {
        "selected_agent_id": selected,
        "dense_selected_agent_id": dense,
        "judge_agree": judge_agree,
        "jcr_num": jcr_num,
        "jcr_den": jcr_den,
    }


def compute_observed_jcr_overlap(
    details: Sequence[Dict[str, Any]],
    *,
    selected_key: str = "selected_agent_id",
    dense_key: str = "dense_selected_agent_id",
) -> Dict[str, Any]:
    selected_parse_count = 0
    dense_parse_count = 0
    overlap_num = 0
    overlap_den = 0

    for row in details:
        selected = normalize_selected_agent_id(row.get(selected_key))
        dense = normalize_selected_agent_id(row.get(dense_key))
        selected_parse_count += int(selected is not None)
        dense_parse_count += int(dense is not None)
        if selected is None or dense is None:
            continue
        overlap_den += 1
        overlap_num += int(selected == dense)

    total = len(details)
    return {
        "selected_agent_id_parse_count": selected_parse_count,
        "dense_selected_agent_id_parse_count": dense_parse_count,
        "observed_jcr_overlap_num": overlap_num,
        "observed_jcr_overlap_den": overlap_den,
        "observed_jcr_overlap": (overlap_num / overlap_den) if overlap_den else None,
        "parse_coverage": (overlap_den / total) if total else None,
    }


def compute_strict_paper_jcr(
    details: Sequence[Dict[str, Any]],
    *,
    selected_key: str = "selected_agent_id",
    dense_key: str = "dense_selected_agent_id",
) -> Dict[str, Any]:
    payload = compute_observed_jcr_overlap(
        details,
        selected_key=selected_key,
        dense_key=dense_key,
    )
    return {
        "jcr_num": int(payload["observed_jcr_overlap_num"]),
        "jcr_den": int(payload["observed_jcr_overlap_den"]),
        "jcr": payload["observed_jcr_overlap"],
        "parse_coverage": payload["parse_coverage"],
        "selected_agent_id_parse_count": int(payload["selected_agent_id_parse_count"]),
        "dense_selected_agent_id_parse_count": int(payload["dense_selected_agent_id_parse_count"]),
        "jcr_semantics": (
            "strict_paper_candidate_selection_agreement;"
            " numerator compares selected original candidate ids;"
            " denominator keeps only cases where both ids parse successfully"
        ),
    }


def cumulative_judge_agree_rate(
    details: Iterable[Dict[str, Any]],
    *,
    selected_key: str = "selected_agent_id",
    dense_key: str = "dense_selected_agent_id",
) -> float | None:
    payload = compute_observed_jcr_overlap(
        list(details),
        selected_key=selected_key,
        dense_key=dense_key,
    )
    return payload["observed_jcr_overlap"]

