from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _markdown_table(columns: List[Tuple[str, str]], rows: List[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    if not rows:
        lines.append("| " + " | ".join("-" for _ in columns) + " |")
        return "\n".join(lines)
    for row in rows:
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


def _load_benchmark_rows(result_root: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]]:
    main_rows: List[Dict[str, Any]] = []
    detail_rows: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for benchmark_path in sorted(result_root.glob("formal/**/benchmark_summary.json")):
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset") or payload.get("benchmark_name", "").replace("_paper_repair", ""))
        regime = str(payload.get("generation_regime") or "")
        shuffle = "shuffle" if payload.get("judge_shuffle") else "noshuffle"
        benchmark_dir = benchmark_path.parent
        for method_row in payload.get("methods", []):
            method = str(method_row.get("method") or "")
            main_rows.append(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "shuffle": shuffle,
                    "method": method,
                    "method_label": method_row.get("method_label"),
                    "Acc": method_row.get("Acc"),
                    "JCR": method_row.get("JCR"),
                    "Reuse": method_row.get("reuse"),
                    "parse_coverage": method_row.get("parse_coverage"),
                    "fallback_rate": method_row.get("fallback_rate"),
                    "any_candidate_upper_bound": method_row.get("any_candidate_upper_bound"),
                    "all_wrong_rate": method_row.get("all_wrong_rate"),
                    "selection_gap": method_row.get("selection_gap"),
                    "answer_consistent_disagreement_rate": method_row.get("answer_consistent_disagreement_rate"),
                    "total_cases": method_row.get("total_cases"),
                }
            )
            details_path = benchmark_dir / f"{method}_details.json"
            if details_path.exists():
                detail_rows[(dataset, regime, shuffle, method)] = json.loads(details_path.read_text(encoding="utf-8"))
    return main_rows, detail_rows


def _delta_rows(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    for row in main_rows:
        grouped.setdefault((row["dataset"], row["regime"], row["shuffle"]), {})[row["method"]] = row
    rows: List[Dict[str, Any]] = []
    for (dataset, regime, shuffle), methods in sorted(grouped.items()):
        kv = methods.get("kvcomm")
        ccr = methods.get("ccr_judge")
        if not kv or not ccr:
            continue
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "shuffle": shuffle,
                "acc_delta_ccr_minus_kvcomm": (ccr.get("Acc") or 0.0) - (kv.get("Acc") or 0.0),
                "jcr_delta_ccr_minus_kvcomm": ((ccr.get("JCR") or 0.0) - (kv.get("JCR") or 0.0)),
                "parse_delta_ccr_minus_kvcomm": (ccr.get("parse_coverage") or 0.0) - (kv.get("parse_coverage") or 0.0),
                "upper_bound_delta_ccr_minus_kvcomm": (ccr.get("any_candidate_upper_bound") or 0.0) - (kv.get("any_candidate_upper_bound") or 0.0),
                "selection_gap_delta_ccr_minus_kvcomm": (ccr.get("selection_gap") or 0.0) - (kv.get("selection_gap") or 0.0),
            }
        )
    return rows


def _taxonomy_rows(detail_rows: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    grouped_keys = sorted({key[:3] for key in detail_rows})
    rows: List[Dict[str, Any]] = []
    for dataset, regime, shuffle in grouped_keys:
        ccr_rows = {
            str(row.get("question_id")): row
            for row in detail_rows.get((dataset, regime, shuffle, "ccr_judge"), [])
        }
        kv_rows = {
            str(row.get("question_id")): row
            for row in detail_rows.get((dataset, regime, shuffle, "kvcomm"), [])
        }
        if not ccr_rows:
            continue
        shared_ids = sorted(set(ccr_rows) | set(kv_rows))
        counts = Counter()
        for qid in shared_ids:
            ccr = ccr_rows.get(qid, {})
            kv = kv_rows.get(qid, {})
            if bool(ccr.get("all_wrong_flag")):
                counts["upper-bound-limited"] += 1
            if bool(ccr.get("any_candidate_upper_bound")) and not bool(ccr.get("official_is_correct")):
                counts["selection-limited"] += 1
            if not bool(ccr.get("selected_agent_parse_success")):
                counts["parse-failure-dominated"] += 1
            if bool(ccr.get("official_is_correct")) and not bool(kv.get("official_is_correct")):
                counts["CCR-helpful"] += 1
            if bool(kv.get("official_is_correct")) and not bool(ccr.get("official_is_correct")):
                counts["CCR-harmful"] += 1
            if bool(ccr.get("jcr_match")) and not bool(kv.get("jcr_match")) and not bool(ccr.get("official_is_correct")) and bool(kv.get("official_is_correct")):
                counts["JCR-gain but Acc-loss"] += 1
            if bool(ccr.get("official_is_correct")) and not bool(kv.get("official_is_correct")) and bool(ccr.get("jcr_match")) == bool(kv.get("jcr_match")):
                counts["Acc-gain but JCR-flat"] += 1
        row = {"dataset": dataset, "regime": regime, "shuffle": shuffle}
        row.update(counts)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper_repair formal outputs.")
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--output_markdown", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser()
    main_rows, detail_rows = _load_benchmark_rows(result_root)
    delta_rows = _delta_rows(main_rows)
    taxonomy_rows = _taxonomy_rows(detail_rows)

    main_columns = [
        ("dataset", "Dataset"),
        ("regime", "Regime"),
        ("shuffle", "Shuffle"),
        ("method_label", "Method"),
        ("Acc", "Acc"),
        ("JCR", "JCR"),
        ("Reuse", "Reuse"),
    ]
    diagnostic_columns = [
        ("dataset", "Dataset"),
        ("regime", "Regime"),
        ("shuffle", "Shuffle"),
        ("method_label", "Method"),
        ("parse_coverage", "Parse Coverage"),
        ("fallback_rate", "Fallback Rate"),
        ("any_candidate_upper_bound", "Any-Candidate Upper Bound"),
        ("all_wrong_rate", "All Wrong Rate"),
        ("selection_gap", "Selection Gap"),
        ("answer_consistent_disagreement_rate", "Answer-JCR"),
    ]
    delta_columns = [
        ("dataset", "Dataset"),
        ("regime", "Regime"),
        ("shuffle", "Shuffle"),
        ("acc_delta_ccr_minus_kvcomm", "Acc delta"),
        ("jcr_delta_ccr_minus_kvcomm", "JCR delta"),
        ("parse_delta_ccr_minus_kvcomm", "Parse delta"),
        ("upper_bound_delta_ccr_minus_kvcomm", "Upper Bound delta"),
        ("selection_gap_delta_ccr_minus_kvcomm", "Selection Gap delta"),
    ]
    taxonomy_columns = [
        ("dataset", "Dataset"),
        ("regime", "Regime"),
        ("shuffle", "Shuffle"),
        ("upper-bound-limited", "upper-bound-limited"),
        ("selection-limited", "selection-limited"),
        ("parse-failure-dominated", "parse-failure-dominated"),
        ("CCR-helpful", "CCR-helpful"),
        ("CCR-harmful", "CCR-harmful"),
        ("JCR-gain but Acc-loss", "JCR-gain but Acc-loss"),
        ("Acc-gain but JCR-flat", "Acc-gain but JCR-flat"),
    ]

    markdown = "\n\n".join(
        [
            "# Paper Repair Formal Results",
            "## Main Table\n" + _markdown_table(main_columns, main_rows),
            "## Diagnostic Table\n" + _markdown_table(diagnostic_columns, main_rows),
            "## Method Delta Table\n" + _markdown_table(delta_columns, delta_rows),
            "## Taxonomy Table\n" + _markdown_table(taxonomy_columns, taxonomy_rows),
        ]
    ) + "\n"
    Path(args.output_markdown).write_text(markdown, encoding="utf-8")
    _json_dump(
        Path(args.output_json),
        {
            "main_rows": main_rows,
            "diagnostic_rows": main_rows,
            "delta_rows": delta_rows,
            "taxonomy_rows": taxonomy_rows,
        },
    )


if __name__ == "__main__":
    main()
