from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from paper_repair.eval.jcr import compute_strict_paper_jcr
from paper_repair.eval.parsing import failure_reason_counts


MAIN_TABLE_COLUMNS = [
    ("method_label", "Method"),
    ("generation_regime", "Regime"),
    ("judge_shuffle", "Shuffle"),
    ("Acc", "Acc"),
    ("JCR", "JCR"),
    ("reuse", "Reuse"),
    ("total_cases", "total_cases"),
]
JCR_DEFINITION = "JCR = matched selected original candidate ids / cases where both method and dense ids parse"


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(columns: Sequence[tuple[str, str]], rows: Iterable[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    seen = False
    for row in rows:
        seen = True
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
    if not seen:
        lines.append("| " + " | ".join("-" for _ in columns) + " |")
    return "\n".join(lines)


def method_summary(
    *,
    spec: Any,
    rows: Sequence[Dict[str, Any]],
    generation_regime: str,
    judge_shuffle: bool,
) -> Dict[str, Any]:
    total_cases = len(rows)
    acc = (sum(1 for row in rows if row.get("official_is_correct", row.get("correct"))) / total_cases) if total_cases else 0.0
    reuse_rate = (sum(float(row.get("reuse_rate", 0.0)) for row in rows) / total_cases) if total_cases else 0.0
    fallback_rate = (sum(int(bool(row.get("fallback_used"))) for row in rows) / total_cases) if total_cases else 0.0
    upper_bound = (sum(int(bool(row.get("any_candidate_upper_bound"))) for row in rows) / total_cases) if total_cases else 0.0
    all_wrong_rate = (sum(int(bool(row.get("all_wrong_flag"))) for row in rows) / total_cases) if total_cases else 0.0
    selection_gap = (sum(float(row.get("selection_gap", 0.0)) for row in rows) / total_cases) if total_cases else 0.0
    answer_consistent_disagreement_rate = (
        sum(float(row.get("answer_consistent_disagreement_rate", 0.0)) for row in rows) / total_cases
        if total_cases else 0.0
    )

    if spec.name == "dense":
        parse_count = sum(int(bool(row.get("parse_success"))) for row in rows)
        summary = {
            "method": spec.name,
            "method_label": spec.label,
            "generation_regime": generation_regime,
            "judge_shuffle": bool(judge_shuffle),
            "reuse_label": spec.reuse,
            "reuse": reuse_rate,
            "total_cases": total_cases,
            "Acc": acc,
            "JCR": None,
            "jcr_num": 0,
            "jcr_den": 0,
            "parse_coverage": (parse_count / total_cases) if total_cases else None,
            "selected_agent_id_parse_count": parse_count,
            "dense_selected_agent_id_parse_count": parse_count,
            "jcr_semantics": JCR_DEFINITION,
            "fallback_rate": fallback_rate,
            "any_candidate_upper_bound": upper_bound,
            "all_wrong_rate": all_wrong_rate,
            "selection_gap": selection_gap,
            "answer_consistent_disagreement_rate": answer_consistent_disagreement_rate,
        }
    else:
        strict_jcr = compute_strict_paper_jcr(rows)
        summary = {
            "method": spec.name,
            "method_label": spec.label,
            "generation_regime": generation_regime,
            "judge_shuffle": bool(judge_shuffle),
            "reuse_label": spec.reuse,
            "reuse": reuse_rate,
            "total_cases": total_cases,
            "Acc": acc,
            "JCR": strict_jcr["jcr"],
            "jcr_num": strict_jcr["jcr_num"],
            "jcr_den": strict_jcr["jcr_den"],
            "parse_coverage": strict_jcr["parse_coverage"],
            "selected_agent_id_parse_count": strict_jcr["selected_agent_id_parse_count"],
            "dense_selected_agent_id_parse_count": strict_jcr["dense_selected_agent_id_parse_count"],
            "jcr_semantics": strict_jcr["jcr_semantics"],
            "fallback_rate": fallback_rate,
            "any_candidate_upper_bound": upper_bound,
            "all_wrong_rate": all_wrong_rate,
            "selection_gap": selection_gap,
            "answer_consistent_disagreement_rate": answer_consistent_disagreement_rate,
        }

    repair_rows = [
        row for row in rows
        if row.get("ccr_mode") is not None or row.get("repair_mode") is not None
    ]
    if repair_rows:
        shortlist_sizes = [
            len(row.get("ccr_shortlist_candidate_ids") or row.get("shortlisted_candidate_ids") or [])
            for row in repair_rows
        ]
        extra_calls = [
            int(row.get("ccr_extra_judge_calls") or row.get("interaction_repair_extra_judge_calls") or 0)
            for row in repair_rows
        ]
        fallback_count = sum(
            int(bool(row.get("ccr_fallback_used")))
            if row.get("ccr_fallback_used") is not None
            else int(bool(row.get("interaction_repair_fallback_used")))
            for row in repair_rows
        )
        single_pass_count = sum(
            int(bool(row.get("ccr_single_pass_success")))
            if row.get("ccr_single_pass_success") is not None
            else int(bool(row.get("interaction_repair_single_pass_success")))
            for row in repair_rows
        )
        summary.update(
            {
                "ccr_mode": repair_rows[0].get("ccr_mode") or repair_rows[0].get("repair_mode"),
                "ccr_single_pass_rate": (single_pass_count / len(repair_rows)) if repair_rows else None,
                "ccr_fallback_rate": (fallback_count / len(repair_rows)) if repair_rows else None,
                "avg_ccr_shortlist_size": (
                    sum(shortlist_sizes) / len(shortlist_sizes) if shortlist_sizes else None
                ),
                "avg_ccr_extra_judge_calls_per_case": (
                    sum(extra_calls) / len(extra_calls) if extra_calls else None
                ),
            }
        )
    if spec.name == "dense":
        summary.update(
            {
                "dense_parse_retry_used_count": sum(int(bool(row.get("dense_parse_retry_used"))) for row in rows),
                "dense_parse_retry_success_count": sum(int(bool(row.get("dense_parse_retry_success"))) for row in rows),
                "dense_parse_failure_reason_counts": failure_reason_counts(
                    rows,
                    key="dense_parse_failure_reason",
                ),
                "dense_first_pass_failure_reason_counts": failure_reason_counts(
                    rows,
                    key="dense_first_pass_failure_reason",
                ),
                "dense_retry_failure_reason_counts": failure_reason_counts(
                    rows,
                    key="dense_retry_failure_reason",
                ),
            }
        )
    return summary


def dump_method_outputs(
    *,
    output_root: Path,
    spec: Any,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    json_dump(output_root / f"{spec.name}_details.json", list(rows))
    json_dump(output_root / f"{spec.name}_summary.json", summary)


def write_benchmark_outputs(
    *,
    output_root: Path,
    benchmark_name: str,
    model_name: str,
    generation_regime: str,
    judge_shuffle: bool,
    shuffle_seed: int,
    frozen_pack: Dict[str, Any],
    pack_metadata: Dict[str, Any] | None,
    summary_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    dataset_label = str(frozen_pack.get("dataset") or benchmark_name).upper()
    main_table = f"# {dataset_label} Paper Repair Main Table\n\n" + markdown_table(MAIN_TABLE_COLUMNS, summary_rows)
    (output_root / "main_table.md").write_text(main_table.rstrip() + "\n", encoding="utf-8")
    benchmark_summary = {
        "benchmark_name": benchmark_name,
        "dataset": frozen_pack.get("dataset"),
        "llm_name": model_name,
        "generation_regime": generation_regime,
        "judge_shuffle": bool(judge_shuffle),
        "shuffle_seed": int(shuffle_seed),
        "frozen_pack_candidate_count": int(frozen_pack.get("candidate_count") or 0),
        "candidate_pack_hash": (pack_metadata or {}).get("candidate_pack_hash"),
        "candidate_build_mode": (pack_metadata or {}).get("candidate_build_mode"),
        "execution_side_reuse": bool(frozen_pack.get("execution_side_reuse")),
        "methods": list(summary_rows),
        "pal_kv_available": True,
        "jcr_definition": JCR_DEFINITION,
    }
    json_dump(output_root / "benchmark_summary.json", benchmark_summary)
    return benchmark_summary
