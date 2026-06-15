from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FULL_VARIANT = "CCR-Judge (Full)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MMLU Ori-protocol CCR ablation runs.")
    parser.add_argument("--result-root", required=True, help="Root created by run_mmlu_ori_ablation.sh.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Summary JSON path. Defaults to <result-root>/ori_mmlu_ablation_summary.json.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Summary markdown path. Defaults to <result-root>/ori_mmlu_ablation_table.md.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, list):
        for item in reversed(metadata):
            if isinstance(item, dict):
                return item
    if isinstance(metadata, dict):
        return metadata
    return {}


def _shortlist_ids(row: Dict[str, Any]) -> List[str]:
    metadata = _latest_metadata(row)
    values = metadata.get("ccr_shortlist_candidate_ids")
    if not values and isinstance(metadata.get("ccr_context_summary"), dict):
        values = metadata["ccr_context_summary"].get("shortlisted_candidate_ids")
    if not values:
        return []
    return [str(item) for item in values]


def _candidate_answer_map(row: Dict[str, Any]) -> Dict[str, str]:
    answer_by_agent: Dict[str, str] = {}
    for candidate in row.get("candidate_outputs") or []:
        if not isinstance(candidate, dict):
            continue
        agent_id = candidate.get("agent_id")
        if agent_id is None:
            continue
        answer = candidate.get("normalized_answer")
        if answer is None:
            answer = candidate.get("final_answer")
        if answer is None:
            continue
        answer_by_agent[str(agent_id)] = str(answer)
    return answer_by_agent


def _shortlist_answer_group_coverage(row: Dict[str, Any]) -> Optional[float]:
    metadata = _latest_metadata(row)
    value = metadata.get("ccr_ablation_shortlist_answer_group_coverage")
    if isinstance(value, (int, float)):
        return float(value)

    answer_by_agent = _candidate_answer_map(row)
    if not answer_by_agent:
        return None
    all_groups = set(answer_by_agent.values())
    shortlist_groups = {
        answer_by_agent[agent_id]
        for agent_id in _shortlist_ids(row)
        if agent_id in answer_by_agent
    }
    if not all_groups:
        return None
    return len(shortlist_groups) / len(all_groups)


def _metadata_number(row: Dict[str, Any], key: str) -> Optional[float]:
    value = _latest_metadata(row).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metadata_text_length(row: Dict[str, Any], key: str) -> Optional[float]:
    value = _latest_metadata(row).get(key)
    if isinstance(value, str):
        return float(len(value))
    return None


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _fmt_pct(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "-"
    if signed:
        return f"{value * 100:+.2f}"
    return f"{value * 100:.2f}"


def _fmt_float(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _csv_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _row_identity(row: Dict[str, Any]) -> str:
    if row.get("question_id") is not None:
        return str(row["question_id"])
    if row.get("record_index") is not None:
        return str(row["record_index"])
    return str(row.get("task") or "")


def _variant_seed(value: Any) -> Optional[int]:
    text = str(value or "")
    marker = "seed="
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    return int("".join(digits)) if digits else None


def _slate_hash(row: Dict[str, Any]) -> str:
    payload = row.get("candidate_outputs") or []
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _agent_slot(row: Dict[str, Any], agent_id: Any) -> Optional[int]:
    if agent_id is None:
        return None
    text = str(agent_id)
    for index, candidate in enumerate(row.get("candidate_outputs") or []):
        if isinstance(candidate, dict) and str(candidate.get("agent_id")) == text:
            return index
    return None


def _selected_map(details: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {
        _row_identity(row): str(row.get("selected_agent_id"))
        for row in details
        if row.get("selected_agent_id") is not None
    }


def _summarize_run(benchmark_path: Path, result_root: Path) -> Dict[str, Any]:
    payload = _load_json(benchmark_path)
    summary = payload.get("summary", {}) or {}
    details_path = Path(payload.get("details_path") or "")
    details = _load_json(details_path) if details_path.exists() else []
    rel = benchmark_path.relative_to(result_root)
    parts = rel.parts
    shuffle = parts[2] if len(parts) >= 5 and parts[0] == "ori_protocol" and parts[1] == "ablation" else None
    variant = parts[3] if len(parts) >= 5 and parts[0] == "ori_protocol" and parts[1] == "ablation" else None
    method_label = summary.get("method_label") or payload.get("method_label") or variant

    dense_den = 0
    dense_num = 0
    for row in details:
        dense_selected = row.get("dense_selected_agent_id")
        shortlist = _shortlist_ids(row)
        if dense_selected is None or not shortlist:
            continue
        dense_den += 1
        dense_num += int(str(dense_selected) in set(shortlist))

    judge_chars = _avg(
        _metadata_number(row, "ccr_ablation_judge_context_chars")
        if _metadata_number(row, "ccr_ablation_judge_context_chars") is not None
        else _metadata_text_length(row, "judge_context_text")
        for row in details
    )
    payload_chars = _avg(
        _metadata_number(row, "ccr_ablation_context_payload_chars")
        if _metadata_number(row, "ccr_ablation_context_payload_chars") is not None
        else _metadata_text_length(row, "ccr_context_text")
        for row in details
    )
    payload_tokens = _avg(
        _metadata_number(row, "ccr_ablation_context_payload_tokens")
        for row in details
    )
    coverage = _avg(_shortlist_answer_group_coverage(row) for row in details)
    total_cases = int(summary.get("total_cases") or payload.get("total_cases") or len(details) or 0)
    acc = summary.get("acc", payload.get("acc"))
    jcr = summary.get("JCR", payload.get("JCR"))
    return {
        "protocol_name": summary.get("protocol_name") or payload.get("protocol_name") or "ori_style_online_protocol",
        "model": (summary.get("runner_config") or {}).get("model_path")
        if isinstance(summary.get("runner_config"), dict)
        else None,
        "shuffle": shuffle or ("shuffle" if payload.get("judge_shuffle") else "noshuffle"),
        "variant": str(method_label),
        "path_variant": variant,
        "decision_method": payload.get("decision_method") or summary.get("decision_method"),
        "total_cases": total_cases,
        "Acc": None if acc is None else float(acc),
        "JCR": None if jcr is None else float(jcr),
        "parse_coverage": summary.get("parse_coverage", payload.get("parse_coverage")),
        "selected_id_parse_success": summary.get("parse_coverage", payload.get("parse_coverage")),
        "dense_id_parse_success": summary.get("dense_parse_coverage", payload.get("dense_parse_coverage")),
        "jcr_den": summary.get("jcr_den", payload.get("jcr_den")),
        "fallback_rate": summary.get("fallback_rate", payload.get("fallback_rate")),
        "online_candidate_upper_bound_acc": summary.get(
            "online_candidate_upper_bound_acc",
            payload.get("online_candidate_upper_bound_acc"),
        ),
        "selection_gap": summary.get("selection_gap", payload.get("selection_gap")),
        "dense_winner_in_shortlist_rate": _rate(dense_num, dense_den),
        "dense_winner_in_shortlist_num": dense_num,
        "dense_winner_in_shortlist_den": dense_den,
        "shortlist_answer_group_coverage": coverage,
        "avg_judge_input_chars": judge_chars,
        "avg_payload_chars": payload_chars,
        "avg_payload_tokens": payload_tokens,
        "benchmark_summary": str(benchmark_path),
        "details_path": str(details_path),
        "_details": details,
    }


def _weighted_average(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = row.get(key)
        total = int(row.get("total_cases") or 0)
        if value is None or total <= 0:
            continue
        numerator += float(value) * total
        denominator += total
    if denominator <= 0:
        return None
    return numerator / denominator


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser() if args.output_json else result_root / "ori_mmlu_ablation_summary.json"
    output_markdown = (
        Path(args.output_markdown).expanduser()
        if args.output_markdown
        else result_root / "ori_mmlu_ablation_table.md"
    )

    run_rows = [
        _summarize_run(path, result_root)
        for path in sorted(result_root.glob("ori_protocol/ablation/*/*/benchmark_summary.json"))
    ]
    full_by_shuffle = {
        row["shuffle"]: row
        for row in run_rows
        if row.get("variant") == FULL_VARIANT
    }

    for row in run_rows:
        full = full_by_shuffle.get(row["shuffle"])
        if full is None:
            row["delta_acc_vs_full"] = None
            row["delta_jcr_vs_full"] = None
            row["agreement_with_full"] = None
            continue
        row["delta_acc_vs_full"] = (
            None if row.get("Acc") is None or full.get("Acc") is None else row["Acc"] - full["Acc"]
        )
        row["delta_jcr_vs_full"] = (
            None if row.get("JCR") is None or full.get("JCR") is None else row["JCR"] - full["JCR"]
        )
        if row["variant"] == FULL_VARIANT:
            row["agreement_with_full"] = 1.0
        else:
            full_selected = _selected_map(full.pop("_details", []) if False else full["_details"])
            selected = _selected_map(row["_details"])
            shared = sorted(set(full_selected) & set(selected))
            row["agreement_with_full"] = (
                None
                if not shared
                else sum(int(full_selected[key] == selected[key]) for key in shared) / len(shared)
            )

    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in run_rows:
        by_variant.setdefault(row["variant"], []).append(row)

    full_avg_rows = by_variant.get(FULL_VARIANT, [])
    full_avg_acc = _weighted_average(full_avg_rows, "Acc")
    full_avg_jcr = _weighted_average(full_avg_rows, "JCR")
    average_rows = []
    for variant, rows in by_variant.items():
        avg_acc = _weighted_average(rows, "Acc")
        avg_jcr = _weighted_average(rows, "JCR")
        average_rows.append(
            {
                "variant": variant,
                "total_cases": sum(int(row.get("total_cases") or 0) for row in rows),
                "Avg Acc": avg_acc,
                "Delta Acc vs Full": None if avg_acc is None or full_avg_acc is None else avg_acc - full_avg_acc,
                "Avg JCR": avg_jcr,
                "Delta JCR vs Full": None if avg_jcr is None or full_avg_jcr is None else avg_jcr - full_avg_jcr,
                "Selected ID Parse Success": _weighted_average(rows, "selected_id_parse_success"),
                "Dense ID Parse Success": _weighted_average(rows, "dense_id_parse_success"),
                "JCR denominator": sum(int(row.get("jcr_den") or 0) for row in rows),
                "Fallback Rate": _weighted_average(rows, "fallback_rate"),
                "Average Payload Tokens": _weighted_average(rows, "avg_payload_tokens"),
                "Dense Retain": _avg(row.get("dense_winner_in_shortlist_rate") for row in rows),
                "Shortlist Coverage": _avg(row.get("shortlist_answer_group_coverage") for row in rows),
                "Selection Gap": _weighted_average(rows, "selection_gap"),
                "Agreement with Full": _avg(row.get("agreement_with_full") for row in rows),
            }
        )
    random_rows = [
        row
        for variant, rows in by_variant.items()
        if str(variant).startswith("CCR-Judge with Random Shortlist")
        for row in rows
    ]
    if random_rows:
        avg_acc = _weighted_average(random_rows, "Acc")
        avg_jcr = _weighted_average(random_rows, "JCR")
        average_rows.append(
            {
                "variant": "CCR-Judge Random Shortlist (mean over seeds)",
                "total_cases": sum(int(row.get("total_cases") or 0) for row in random_rows),
                "Avg Acc": avg_acc,
                "Delta Acc vs Full": None if avg_acc is None or full_avg_acc is None else avg_acc - full_avg_acc,
                "Avg JCR": avg_jcr,
                "Delta JCR vs Full": None if avg_jcr is None or full_avg_jcr is None else avg_jcr - full_avg_jcr,
                "Selected ID Parse Success": _weighted_average(random_rows, "selected_id_parse_success"),
                "Dense ID Parse Success": _weighted_average(random_rows, "dense_id_parse_success"),
                "JCR denominator": sum(int(row.get("jcr_den") or 0) for row in random_rows),
                "Fallback Rate": _weighted_average(random_rows, "fallback_rate"),
                "Average Payload Tokens": _weighted_average(random_rows, "avg_payload_tokens"),
                "Dense Retain": _avg(row.get("dense_winner_in_shortlist_rate") for row in random_rows),
                "Shortlist Coverage": _avg(row.get("shortlist_answer_group_coverage") for row in random_rows),
                "Selection Gap": _weighted_average(random_rows, "selection_gap"),
                "Agreement with Full": _avg(row.get("agreement_with_full") for row in random_rows),
            }
        )

    public_run_rows = []
    for row in run_rows:
        public = {key: value for key, value in row.items() if key != "_details"}
        public_run_rows.append(public)

    output_payload = {
        "result_root": str(result_root),
        "per_run": public_run_rows,
        "averages": average_rows,
    }
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    online_average_csv = result_root / "ablation_online_mmlu_average.csv"
    online_average_md = result_root / "ablation_online_mmlu_average.md"
    online_details_jsonl = result_root / "ablation_online_mmlu_details.jsonl"
    online_summary_md = result_root / "ablation_online_mmlu_summary.md"

    average_headers = [
        "variant",
        "total_cases",
        "Avg Acc",
        "Delta Acc vs Full",
        "Avg JCR",
        "Delta JCR vs Full",
        "Selected ID Parse Success",
        "Dense ID Parse Success",
        "JCR denominator",
        "Fallback Rate",
        "Average Payload Tokens",
        "Dense Retain",
        "Shortlist Coverage",
        "Selection Gap",
        "Agreement with Full",
    ]
    _write_csv(
        online_average_csv,
        average_headers,
        [{key: _csv_float(row.get(key)) for key in average_headers} for row in average_rows],
    )

    with online_details_jsonl.open("w", encoding="utf-8") as handle:
        for run_row in run_rows:
            for detail in run_row.get("_details") or []:
                payload_count = None
                metadata = detail.get("metadata")
                if isinstance(metadata, list):
                    metadata = next((item for item in reversed(metadata) if isinstance(item, dict)), {})
                if isinstance(metadata, dict):
                    payload_count = metadata.get("ccr_ablation_context_payload_tokens")
                    if payload_count is None:
                        payload_count = metadata.get("ccr_ablation_context_payload_chars")
                record = {
                    "sample_id": detail.get("question_id") or detail.get("record_index"),
                    "dataset": "mmlu",
                    "model": run_row.get("model"),
                    "protocol": run_row.get("protocol_name"),
                    "regime": "online_full_connected",
                    "shuffle": run_row.get("shuffle"),
                    "method": run_row.get("decision_method"),
                    "variant": run_row.get("variant"),
                    "random_seed": _variant_seed(run_row.get("variant")),
                    "candidate_slate_id": _slate_hash(detail),
                    "slate_hash": _slate_hash(detail),
                    "generated_slate": detail.get("candidate_outputs"),
                    "selected_candidate_id": detail.get("selected_agent_id"),
                    "dense_selected_candidate_id": detail.get("dense_selected_agent_id"),
                    "selected_slot": _agent_slot(detail, detail.get("selected_agent_id")),
                    "dense_selected_slot": _agent_slot(detail, detail.get("dense_selected_agent_id")),
                    "parse_success": detail.get("parse_success"),
                    "dense_parse_success": detail.get("dense_parse_success"),
                    "jcr_match": detail.get("jcr_match"),
                    "correct": detail.get("correct"),
                    "payload_token_count": payload_count,
                    "shortlist_size": len(_shortlist_ids(detail)),
                    "fallback_used": detail.get("fallback_used"),
                    "final_answer": detail.get("final_answer") or detail.get("final_answer_text"),
                    "gold_answer": detail.get("correct_answer"),
                    "error_message": detail.get("error_message"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_table_rows = [
        [
            row["variant"],
            _fmt_pct(row.get("Avg Acc")),
            _fmt_pct(row.get("Delta Acc vs Full"), signed=True),
            _fmt_pct(row.get("Avg JCR")),
            _fmt_pct(row.get("Delta JCR vs Full"), signed=True),
            _fmt_pct(row.get("Selected ID Parse Success")),
            _fmt_pct(row.get("Dense ID Parse Success")),
            str(row.get("JCR denominator") or 0),
            _fmt_pct(row.get("Fallback Rate")),
            _fmt_float(row.get("Average Payload Tokens")),
            _fmt_pct(row.get("Dense Retain")),
            _fmt_pct(row.get("Shortlist Coverage")),
            _fmt_pct(row.get("Selection Gap")),
            _fmt_pct(row.get("Agreement with Full")),
        ]
        for row in average_rows
    ]
    per_run_table_rows = [
        [
            str(row["shuffle"]),
            row["variant"],
            _fmt_pct(row.get("Acc")),
            _fmt_pct(row.get("delta_acc_vs_full"), signed=True),
            _fmt_pct(row.get("JCR")),
            _fmt_pct(row.get("delta_jcr_vs_full"), signed=True),
            _fmt_pct(row.get("selected_id_parse_success")),
            _fmt_pct(row.get("dense_id_parse_success")),
            str(row.get("jcr_den") or 0),
            _fmt_pct(row.get("fallback_rate")),
            _fmt_float(row.get("avg_payload_tokens")),
            _fmt_pct(row.get("dense_winner_in_shortlist_rate")),
            _fmt_pct(row.get("shortlist_answer_group_coverage")),
            _fmt_float(row.get("avg_judge_input_chars")),
            _fmt_float(row.get("avg_payload_chars")),
            _fmt_pct(row.get("selection_gap")),
            _fmt_pct(row.get("agreement_with_full")),
        ]
        for row in run_rows
    ]

    markdown = "\n\n".join(
        [
            "# MMLU Ori-Protocol CCR Ablation",
            "All percentage columns are rendered as percentages. JCR is reuse-vs-dense selected-candidate consistency within the same online run.",
            "## Average Across Shuffle Conditions",
            _markdown_table(
                [
                    "Variant",
                    "Avg Acc",
                    "Delta Acc vs Full",
                    "Avg JCR",
                    "Delta JCR vs Full",
                    "Selected Parse",
                    "Dense Parse",
                    "JCR Den.",
                    "Fallback",
                    "Payload Tokens",
                    "Dense Retain",
                    "Shortlist Coverage",
                    "Selection Gap",
                    "Agreement w/ Full",
                ],
                avg_table_rows,
            ),
            "## Per-Ordering Results",
            _markdown_table(
                [
                    "Shuffle",
                    "Variant",
                    "Acc",
                    "Delta Acc vs Full",
                    "JCR",
                    "Delta JCR vs Full",
                    "Selected Parse",
                    "Dense Parse",
                    "JCR Den.",
                    "Fallback",
                    "Payload Tokens",
                    "Dense Retain",
                    "Shortlist Coverage",
                    "Judge Chars",
                    "Payload Chars",
                    "Selection Gap",
                    "Agreement w/ Full",
                ],
                per_run_table_rows,
            ),
        ]
    )
    output_markdown.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    online_average_md.write_text(
        "# Online MMLU Ablation Average\n\n"
        + _markdown_table(
            [
                "Variant",
                "Avg Acc",
                "Delta Acc vs Full",
                "Avg JCR",
                "Delta JCR vs Full",
                "Selected Parse",
                "Dense Parse",
                "JCR Den.",
                "Fallback",
                "Payload Tokens",
                "Dense Retain",
                "Shortlist Coverage",
                "Selection Gap",
                "Agreement w/ Full",
            ],
            avg_table_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    online_summary_md.write_text(
        "# Online MMLU Ablation Summary\n\n"
        "Protocol: online full-connected Ori protocol. JCR is within-run reuse-vs-dense selected-candidate consistency on the same ordered slate.\n\n"
        f"- Runs summarized: `{len(run_rows)}`\n"
        f"- Average rows: `{len(average_rows)}`\n"
        f"- Details JSONL: `{online_details_jsonl}`\n\n"
        + online_average_md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"[SUMMARY SAVED] {output_json}")
    print(f"[TABLE SAVED] {output_markdown}")
    print(f"[ONLINE AVERAGE SAVED] {online_average_csv}")
    print(f"[ONLINE DETAILS SAVED] {online_details_jsonl}")


if __name__ == "__main__":
    main()
