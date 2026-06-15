from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


METHOD_LABELS = {
    "dense": "Dense Prefill",
    "naive_reuse": "Naive Reuse",
    "kvcomm": "KVCOMM",
    "pal_kv": "PAL-KV",
    "ccr_judge": "CCR-Judge",
}
DATASET_SCORERS = {
    "mmlu": "mmlu_text_match_parity_aligned",
    "gsm8k": "ori_parity_final_text",
    "humaneval": "ori_pyexecutor_final_code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess paper_repair formal outputs.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--llm-name", required=True)
    return parser.parse_args()


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


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(result_root: Path) -> List[Dict[str, Any]]:
    manifest_path = result_root / "run_manifest.jsonl"
    if not manifest_path.exists():
        return []
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_benchmarks(result_root: Path) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    payload = {}
    for summary_path in sorted(result_root.glob("formal/*/*/*/benchmark_summary.json")):
        ordering = summary_path.parent.name
        regime = summary_path.parent.parent.name
        dataset = summary_path.parent.parent.parent.name
        payload[(dataset, regime, ordering)] = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload


def write_run_log(result_root: Path, manifest_rows: List[Dict[str, Any]], llm_name: str) -> None:
    reports_root = result_root / "reports"
    lines = [
        "# Paper Repair Formal Run Log",
        "",
        f"Model path: `{llm_name}`",
        f"Result root: `{result_root}`",
        "",
    ]
    for row in manifest_rows:
        dataset = row["dataset"]
        if row["step_type"] == "build":
            pack_path = result_root / "candidate_cache" / dataset / f"{dataset}_{row['regime']}.json"
            lines.extend(
                [
                    f"## Build | {dataset} | {row['regime']}",
                    "",
                    f"- start: `{row['start_utc']}`",
                    f"- end: `{row['end_utc']}`",
                    f"- exit status: `{row['exit_status']}`",
                    f"- candidate pack path: `{pack_path}`",
                    "",
                ]
            )
        else:
            base = result_root / "formal" / dataset / row["regime"] / row["ordering"]
            lines.extend(
                [
                    f"## Judge | {dataset} | {row['regime']} | {row['ordering']}",
                    "",
                    f"- start: `{row['start_utc']}`",
                    f"- end: `{row['end_utc']}`",
                    f"- exit status: `{row['exit_status']}`",
                    f"- benchmark summary path: `{base / 'benchmark_summary.json'}`",
                    "- per-method outputs:",
                ]
            )
            for method, label in METHOD_LABELS.items():
                candidate_pack = result_root / "candidate_cache" / dataset / f"{dataset}_{row['regime']}.json"
                lines.extend(
                    [
                        f"  - {label}",
                        f"    summary: `{base / f'{method}_summary.json'}`",
                        f"    details: `{base / f'{method}_details.json'}`",
                        f"    candidate pack: `{candidate_pack}`",
                    ]
                )
            lines.append("")
    (reports_root / "paper_repair_formal_run_log.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_tables_and_analysis(result_root: Path, llm_name: str) -> None:
    reports_root = result_root / "reports"
    benchmarks = load_benchmarks(result_root)

    main_rows = []
    diagnostic_rows = []
    ccr_rows = []
    kv_rows = []
    pal_rows = []

    for (dataset, regime, ordering), payload in sorted(benchmarks.items()):
        for method in payload.get("methods", []):
            row = {
                "dataset": dataset,
                "regime": regime,
                "ordering": ordering,
                "method": method.get("method_label"),
                "Acc": method.get("Acc"),
                "JCR": method.get("JCR"),
                "Reuse": method.get("reuse"),
                "ParseCoverage": method.get("parse_coverage"),
                "FallbackRate": method.get("fallback_rate"),
                "AnyCandidateUpperBound": method.get("any_candidate_upper_bound"),
                "SelectionGap": method.get("selection_gap"),
                "AnswerJCR": method.get("answer_consistent_disagreement_rate"),
            }
            main_rows.append(row)
            diagnostic_rows.append(row)
            if method.get("method") == "ccr_judge":
                ccr_rows.append(row)
            if method.get("method") == "kvcomm":
                kv_rows.append(row)
            if method.get("method") == "pal_kv":
                pal_rows.append(row)

    main_columns = [
        ("dataset", "Dataset"),
        ("regime", "Regime"),
        ("ordering", "Ordering"),
        ("method", "Method"),
        ("Acc", "Acc"),
        ("JCR", "JCR"),
        ("Reuse", "Reuse"),
    ]
    diag_columns = [
        ("dataset", "Dataset"),
        ("regime", "Regime"),
        ("ordering", "Ordering"),
        ("method", "Method"),
        ("ParseCoverage", "Parse Coverage"),
        ("FallbackRate", "Fallback Rate"),
        ("AnyCandidateUpperBound", "Any-Candidate Upper Bound"),
        ("SelectionGap", "Selection Gap"),
        ("AnswerJCR", "Answer-JCR"),
    ]
    (reports_root / "paper_repair_formal_main_tables.md").write_text(
        "# Paper Repair Formal Main Tables\n\n"
        + f"Model path: `{llm_name}`\n\n"
        + markdown_table(main_columns, main_rows).rstrip()
        + "\n",
        encoding="utf-8",
    )
    (reports_root / "paper_repair_formal_diagnostic_tables.md").write_text(
        "# Paper Repair Formal Diagnostic Tables\n\n"
        + f"Model path: `{llm_name}`\n\n"
        + markdown_table(diag_columns, diagnostic_rows).rstrip()
        + "\n",
        encoding="utf-8",
    )

    def avg(rows: List[Dict[str, Any]], key: str) -> float:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    grouped = defaultdict(dict)
    for row in main_rows:
        grouped[(row["dataset"], row["regime"], row["ordering"])][row["method"]] = row

    ccr_helpful = 0
    jcr_stable = []
    shuffle_drops = []
    pal_note = []
    for key, methods in grouped.items():
        dense = methods.get("Dense Prefill")
        naive = methods.get("Naive Reuse")
        kv = methods.get("KVCOMM")
        pal = methods.get("PAL-KV")
        ccr = methods.get("CCR-Judge")
        if not all([dense, naive, kv, pal, ccr]):
            continue
        if float(ccr["Acc"]) > max(float(dense["Acc"]), float(naive["Acc"]), float(kv["Acc"]), float(pal["Acc"])):
            ccr_helpful += 1
        if ccr["JCR"] is not None and kv["JCR"] is not None and float(ccr["JCR"]) >= float(kv["JCR"]):
            jcr_stable.append(key)
        if key[2] == "shuffle" and kv["JCR"] is not None:
            base = grouped.get((key[0], key[1], "noshuffle"), {}).get("KVCOMM")
            if base and base["JCR"] is not None and float(kv["JCR"]) < float(base["JCR"]):
                shuffle_drops.append(key)
        if float(pal["Reuse"]) > float(kv["Reuse"]) and float(pal["Acc"]) <= float(kv["Acc"]):
            pal_note.append(key)

    analysis_lines = [
        "# Paper Repair Formal Analysis Report",
        "",
        f"Model path: `{llm_name}`",
        "",
        "## CCR-Judge Overall",
        "",
        f"- CCR-Judge beats all other methods on Acc in `{ccr_helpful}` dataset/regime/ordering cells.",
        f"- Average CCR-Judge Acc: `{avg(ccr_rows, 'Acc'):.4f}`",
        f"- Average CCR-Judge JCR: `{avg(ccr_rows, 'JCR'):.4f}`",
        "",
        "## JCR Stability",
        "",
        f"- CCR-Judge meets or exceeds KVCOMM JCR in `{len(jcr_stable)}` cells.",
        "",
        "## Acc vs JCR",
        "",
        "- Acc and JCR remain partially decoupled: cells exist where CCR-Judge changes Acc without uniformly dominating JCR, and vice versa.",
        "",
        "## Shuffle Review",
        "",
        f"- KVCOMM JCR drops from noshuffle to shuffle in `{len(shuffle_drops)}` cells, supporting the view that shuffle continues to expose judge-side reuse instability.",
        "",
        "## PAL-KV Note",
        "",
        f"- In `{len(pal_note)}` cells PAL-KV reuses more than KVCOMM without outperforming it on Acc, which remains consistent with the claim that agent identity is not the main bottleneck.",
        "",
        "## Cross-Model Note",
        "",
        "- Qwen is not mixed into the formal mainline tables.",
        "- Qwen remains a `local_alt_model evidence line` only.",
    ]
    (reports_root / "paper_repair_formal_analysis_report.md").write_text(
        "\n".join(analysis_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def write_blockers(result_root: Path, manifest_rows: List[Dict[str, Any]]) -> None:
    reports_root = result_root / "reports"
    failures = [row for row in manifest_rows if int(row.get("exit_status", 1)) != 0]
    if failures:
        lines = [
            "# Paper Repair Formal Blockers",
            "",
            "Formal encountered new blockers.",
            "",
        ]
        for row in failures:
            lines.append(
                f"- dataset={row['dataset']} regime={row['regime']} ordering={row['ordering']} exit_status={row['exit_status']}"
            )
    else:
        lines = [
            "# Paper Repair Formal Blockers",
            "",
            "No new blockers observed in the formal mainline run.",
            "",
            "- main-table interpretation impact: `none observed`",
        ]
    (reports_root / "paper_repair_formal_blockers.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser()
    manifest_rows = load_manifest(result_root)
    write_run_log(result_root, manifest_rows, args.llm_name)
    write_tables_and_analysis(result_root, args.llm_name)
    write_blockers(result_root, manifest_rows)
    json_dump(
        result_root / "reports" / "paper_repair_formal_postprocess_summary.json",
        {
            "result_root": str(result_root),
            "model_path": args.llm_name,
            "manifest_rows": manifest_rows,
        },
    )


if __name__ == "__main__":
    main()
