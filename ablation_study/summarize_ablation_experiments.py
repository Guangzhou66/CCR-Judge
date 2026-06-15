from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


FULL_SLUG = "ccr_full"
FULL_LABEL = "CCR-Judge (Full)"
RANDOM_PREFIX = "ccr_random_shortlist_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready CCR-Judge ablation reports.")
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--fixed-root", default=None)
    parser.add_argument("--baseline-root", default=None)
    parser.add_argument("--online-root", default=None)
    parser.add_argument("--model", default="/pychen/Test/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--expected-datasets", default="mmlu humaneval")
    parser.add_argument("--expected-regimes", default="parallel_exploration progressive_refinement")
    parser.add_argument("--expected-orderings", default="noshuffle shuffle")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if _float(v) is not None]
    return sum(nums) / len(nums) if nums else None


def _std(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if _float(v) is not None]
    if len(nums) < 2:
        return 0.0 if nums else None
    return statistics.stdev(nums)


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number * 100:+.2f}" if signed else f"{number * 100:.2f}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    if not rows:
        lines.append("| " + " | ".join("-" for _ in headers) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _method_details_path(block_dir: Path, slug: str) -> Path:
    return block_dir / f"{slug}_details.json"


def _method_summary_path(block_dir: Path, slug: str) -> Path:
    return block_dir / f"{slug}_summary.json"


def _load_rows(block_dir: Path, slug: str) -> List[Dict[str, Any]]:
    path = _method_details_path(block_dir, slug)
    if not path.exists():
        return []
    payload = _read_json(path)
    return payload if isinstance(payload, list) else []


def _load_summary(block_dir: Path, slug: str) -> Optional[Dict[str, Any]]:
    path = _method_summary_path(block_dir, slug)
    if not path.exists():
        return None
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else None


def _shortlist_ids(row: Dict[str, Any]) -> List[str]:
    values = row.get("ccr_shortlist_candidate_ids")
    if values is None:
        payload = row.get("ccr_shortlist_payload")
        if isinstance(payload, dict):
            values = payload.get("shortlisted_candidate_ids")
    if not isinstance(values, list):
        return []
    return [str(item) for item in values]


def _payload_tokens(row: Dict[str, Any]) -> Optional[float]:
    for key in (
        "ccr_ablation_context_payload_tokens",
        "ccr_context_payload_tokens",
        "ccr_ablation_context_payload_chars",
        "ccr_context_payload_chars",
    ):
        number = _float(row.get(key))
        if number is not None:
            return number
    text = row.get("ccr_context_text")
    if isinstance(text, str):
        return float(len(text))
    return None


def _dense_in_shortlist(row: Dict[str, Any]) -> Optional[bool]:
    dense_id = row.get("dense_selected_agent_id")
    shortlist = set(_shortlist_ids(row))
    if dense_id is None or not shortlist:
        return None
    return str(dense_id) in shortlist


def _setting_name(config: Dict[str, Any]) -> str:
    testbed = config.get("testbed") or {}
    return "/".join(
        str(testbed.get(key) or "")
        for key in ("dataset", "regime", "ordering")
    )


def _setting_parts(config: Dict[str, Any]) -> Dict[str, str]:
    testbed = config.get("testbed") or {}
    return {
        "dataset": str(testbed.get("dataset") or ""),
        "regime": str(testbed.get("regime") or ""),
        "shuffle": str(testbed.get("ordering") or ""),
        "layer": str(testbed.get("layer") or ""),
    }


def _parse_success(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        number = _float(value)
        if number is not None:
            return number
    return None


def _agent_slot(row: Dict[str, Any], agent_id: Any) -> Optional[int]:
    if agent_id is None:
        return None
    values = row.get("candidate_agent_ids")
    if isinstance(values, list):
        text = str(agent_id)
        for index, value in enumerate(values):
            if str(value) == text:
                return index
    return None


def _fixed_detail_record(
    *,
    config: Dict[str, Any],
    parts: Dict[str, str],
    variant: Dict[str, Any],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    selected_id = row.get("selected_agent_id")
    dense_id = row.get("dense_selected_agent_id")
    return {
        "sample_id": row.get("sample_id") or row.get("question_id") or row.get("record_index"),
        "dataset": parts.get("dataset"),
        "model": config.get("llm_name"),
        "protocol": config.get("protocol_type") or "fixed_candidate_judge_only",
        "regime": parts.get("regime"),
        "shuffle": parts.get("shuffle"),
        "method": row.get("method") or variant.get("label") or variant.get("slug"),
        "variant": variant.get("label") or variant.get("slug"),
        "variant_slug": variant.get("slug"),
        "random_seed": variant.get("random_seed") if variant.get("random_seed") is not None else row.get("ccr_ablation_random_seed"),
        "candidate_slate_id": row.get("candidate_pack_hash") or config.get("candidate_pack_hash"),
        "slate_hash": row.get("candidate_pack_hash") or config.get("candidate_pack_hash"),
        "selected_candidate_id": selected_id,
        "dense_selected_candidate_id": dense_id,
        "selected_slot": row.get("selected_index") if row.get("selected_index") is not None else _agent_slot(row, selected_id),
        "dense_selected_slot": _agent_slot(row, dense_id),
        "parse_success": row.get("parse_success"),
        "dense_parse_success": row.get("dense_parse_success")
        if row.get("dense_parse_success") is not None
        else row.get("dense_selected_agent_parse_success"),
        "jcr_match": row.get("jcr_match"),
        "correct": row.get("correct") if row.get("correct") is not None else row.get("official_is_correct"),
        "fallback_used": row.get("fallback_used") if row.get("fallback_used") is not None else row.get("ccr_fallback_used"),
        "payload_token_count": _payload_tokens(row),
        "shortlist_size": row.get("ccr_ablation_shortlist_size") if row.get("ccr_ablation_shortlist_size") is not None else len(_shortlist_ids(row)),
        "final_answer": row.get("final_answer") or row.get("final_answer_text"),
        "gold_answer": row.get("gold_answer") or row.get("correct_answer") or row.get("target_choice"),
        "error_message": row.get("error_message"),
    }


def collect_fixed_rows(fixed_root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    detail_records: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    for config_path in sorted(fixed_root.glob("*/*/*/*/ccr_ablation_config.json")):
        block_dir = config_path.parent
        config = _read_json(config_path)
        parts = _setting_parts(config)
        setting = _setting_name(config)
        variants = list(config.get("variants") or [])
        full_summary = _load_summary(block_dir, FULL_SLUG)
        full_acc = _float((full_summary or {}).get("Acc"))
        full_jcr = _float((full_summary or {}).get("JCR"))
        for variant in variants:
            slug = str(variant.get("slug"))
            label = str(variant.get("label") or slug)
            summary = _load_summary(block_dir, slug)
            detail_rows = _load_rows(block_dir, slug)
            if summary is None:
                incomplete.append({**parts, "setting": setting, "variant": label, "reason": "missing_summary"})
                continue
            acc = _float(summary.get("Acc"))
            jcr = _float(summary.get("JCR"))
            for detail_row in detail_rows:
                detail_records.append(
                    _fixed_detail_record(
                        config=config,
                        parts=parts,
                        variant=variant,
                        row=detail_row,
                    )
                )
            rows.append(
                {
                    **parts,
                    "setting": setting,
                    "variant_slug": slug,
                    "variant": label,
                    "Acc": acc,
                    "Delta Acc vs Full": None if acc is None or full_acc is None else acc - full_acc,
                    "JCR": jcr,
                    "Delta JCR vs Full": None if jcr is None or full_jcr is None else jcr - full_jcr,
                    "Parse Success Rate": _float(summary.get("parse_coverage")),
                    "Selected ID Parse Success": _mean(
                        _parse_success(detail_row, "selected_agent_parse_success", "parse_success")
                        for detail_row in detail_rows
                    ),
                    "Dense ID Parse Success": _mean(
                        _parse_success(detail_row, "dense_selected_agent_parse_success", "dense_parse_success")
                        for detail_row in detail_rows
                    ),
                    "Fallback Rate": _float(summary.get("fallback_rate")),
                    "Average Payload Tokens": _mean(_payload_tokens(row) for row in detail_rows),
                    "Average Shortlist Size": _mean(len(_shortlist_ids(row)) for row in detail_rows),
                    "Dense Retain": _mean(
                        1.0 if flag else 0.0
                        for flag in (_dense_in_shortlist(row) for row in detail_rows)
                        if flag is not None
                    ),
                    "JCR denominator": summary.get("jcr_den"),
                    "total_cases": summary.get("total_cases"),
                    "summary_path": str(_method_summary_path(block_dir, slug)),
                    "details_path": str(_method_details_path(block_dir, slug)),
                    "is_random_shortlist": slug.startswith(RANDOM_PREFIX),
                    "random_seed": (
                        int(slug.replace(RANDOM_PREFIX, ""))
                        if slug.startswith(RANDOM_PREFIX) and slug.replace(RANDOM_PREFIX, "").isdigit()
                        else None
                    ),
                }
            )
    return rows, incomplete, detail_records


def aggregate_fixed(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    full_by_setting = {
        row["setting"]: row
        for row in rows
        if row.get("variant_slug") == FULL_SLUG
    }
    aggregates: List[Dict[str, Any]] = []
    for variant, items in by_variant.items():
        wins = 0
        ties = 0
        for item in items:
            full = full_by_setting.get(item["setting"])
            if full is None or item.get("Acc") is None or full.get("Acc") is None:
                continue
            if math.isclose(float(item["Acc"]), float(full["Acc"]), abs_tol=1e-12):
                ties += 1
            elif float(item["Acc"]) > float(full["Acc"]):
                wins += 1
        aggregates.append(
            {
                "Variant": variant,
                "Avg Acc": _mean(item.get("Acc") for item in items),
                "Delta Acc vs Full": _mean(item.get("Delta Acc vs Full") for item in items),
                "Avg JCR": _mean(item.get("JCR") for item in items),
                "Delta JCR vs Full": _mean(item.get("Delta JCR vs Full") for item in items),
                "Parse Success Rate": _mean(item.get("Parse Success Rate") for item in items),
                "Selected ID Parse Success": _mean(item.get("Selected ID Parse Success") for item in items),
                "Dense ID Parse Success": _mean(item.get("Dense ID Parse Success") for item in items),
                "Fallback Rate": _mean(item.get("Fallback Rate") for item in items),
                "Average Payload Tokens": _mean(item.get("Average Payload Tokens") for item in items),
                "Average Shortlist Size": _mean(item.get("Average Shortlist Size") for item in items),
                "Dense Retain": _mean(item.get("Dense Retain") for item in items),
                "JCR denominator": sum(int(item.get("JCR denominator") or 0) for item in items),
                "Acc wins/ties vs Full": f"{wins}+{ties}/{len(items)}",
                "settings_completed": len(items),
            }
        )
    preferred = [
        FULL_LABEL,
        "CCR-Judge w/o Comparative Grouping",
        "CCR-Judge w/o Support Scoring",
        "CCR-Judge w/o Shortlist Compression",
        "CCR-Judge w/o Comparative Payload Statistics",
        "CCR-Judge Stats Only",
        "CCR-Judge Shortlist Only",
    ]
    order = {name: idx for idx, name in enumerate(preferred)}
    return sorted(aggregates, key=lambda row: (order.get(str(row["Variant"]), 100), str(row["Variant"])))


def random_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detail = [
        {
            "setting": row["setting"],
            "dataset": row["dataset"],
            "regime": row["regime"],
            "shuffle": row["shuffle"],
            "seed": row.get("random_seed"),
            "Acc": row.get("Acc"),
            "JCR": row.get("JCR"),
            "Delta Acc vs Full": row.get("Delta Acc vs Full"),
            "Delta JCR vs Full": row.get("Delta JCR vs Full"),
            "Average Payload Tokens": row.get("Average Payload Tokens"),
        }
        for row in rows
        if row.get("is_random_shortlist")
    ]
    by_setting: Dict[str, List[Dict[str, Any]]] = {}
    for row in detail:
        by_setting.setdefault(str(row["setting"]), []).append(row)
    summary: List[Dict[str, Any]] = []
    for setting, items in by_setting.items():
        summary.append(
            {
                "setting": setting,
                "seed": "mean/std",
                "Acc": _mean(item.get("Acc") for item in items),
                "Acc std": _std(item.get("Acc") for item in items),
                "JCR": _mean(item.get("JCR") for item in items),
                "JCR std": _std(item.get("JCR") for item in items),
                "Delta Acc vs Full": _mean(item.get("Delta Acc vs Full") for item in items),
                "Delta JCR vs Full": _mean(item.get("Delta JCR vs Full") for item in items),
                "Average Payload Tokens": _mean(item.get("Average Payload Tokens") for item in items),
            }
        )
    return detail + summary


def _csv_float(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    number = _float(value)
    if number is None:
        return value
    return f"{number:.6f}"


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_fixed_outputs(
    suite_root: Path,
    rows: List[Dict[str, Any]],
    incomplete: List[Dict[str, Any]],
    detail_records: Sequence[Dict[str, Any]],
) -> None:
    average = aggregate_fixed(rows)
    avg_headers = [
        "Variant",
        "Avg Acc",
        "Delta Acc vs Full",
        "Avg JCR",
        "Delta JCR vs Full",
        "Parse Success Rate",
        "Selected ID Parse Success",
        "Dense ID Parse Success",
        "Fallback Rate",
        "Average Payload Tokens",
        "Average Shortlist Size",
        "JCR denominator",
        "Acc wins/ties vs Full",
    ]
    per_headers = [
        "dataset",
        "regime",
        "shuffle",
        "variant",
        "Acc",
        "Delta Acc vs Full",
        "JCR",
        "Delta JCR vs Full",
        "Parse Success Rate",
        "Selected ID Parse Success",
        "Dense ID Parse Success",
        "Fallback Rate",
        "Average Payload Tokens",
        "Average Shortlist Size",
        "JCR denominator",
    ]
    random_rows = random_seed_rows(rows)
    random_headers = [
        "setting",
        "dataset",
        "regime",
        "shuffle",
        "seed",
        "Acc",
        "Acc std",
        "JCR",
        "JCR std",
        "Delta Acc vs Full",
        "Delta JCR vs Full",
        "Average Payload Tokens",
    ]

    _write_csv(
        suite_root / "ablation_fixed_slate_average.csv",
        avg_headers,
        [{key: _csv_float(row.get(key)) for key in avg_headers} for row in average],
    )
    _write_csv(
        suite_root / "ablation_fixed_slate_per_setting.csv",
        per_headers,
        [{key: _csv_float(row.get(key)) for key in per_headers} for row in rows],
    )
    _write_csv(
        suite_root / "ablation_fixed_slate_random_shortlist_seeds.csv",
        random_headers,
        [{key: _csv_float(row.get(key)) for key in random_headers} for row in random_rows],
    )
    _write_jsonl(suite_root / "ablation_fixed_slate_details.jsonl", detail_records)

    avg_md_rows = [
        [
            row["Variant"],
            _fmt_pct(row.get("Avg Acc")),
            _fmt_pct(row.get("Delta Acc vs Full"), signed=True),
            _fmt_pct(row.get("Avg JCR")),
            _fmt_pct(row.get("Delta JCR vs Full"), signed=True),
            _fmt_num(row.get("Average Payload Tokens"), 1),
        ]
        for row in average
    ]
    (suite_root / "ablation_fixed_slate_average.md").write_text(
        "# Fixed-Slate Average Ablation\n\n"
        + _markdown_table(
            ["Variant", "Avg Acc", "Delta Acc", "Avg JCR", "Delta JCR", "Payload Len."],
            avg_md_rows,
        )
        + "\n",
        encoding="utf-8",
    )

    per_md_rows = [
        [
            row["dataset"],
            row["regime"],
            row["shuffle"],
            row["variant"],
            _fmt_pct(row.get("Acc")),
            _fmt_pct(row.get("JCR")),
            _fmt_pct(row.get("Delta Acc vs Full"), signed=True),
            _fmt_pct(row.get("Delta JCR vs Full"), signed=True),
            _fmt_num(row.get("Average Payload Tokens"), 1),
        ]
        for row in rows
    ]
    (suite_root / "ablation_fixed_slate_per_setting.md").write_text(
        "# Fixed-Slate Per-Setting Ablation\n\n"
        + _markdown_table(
            ["Dataset", "Regime", "Shuffle", "Variant", "Acc", "JCR", "Delta Acc", "Delta JCR", "Payload Len."],
            per_md_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics = {
        "fixed_slate_completed_rows": len(rows),
        "incomplete": incomplete,
        "average": average,
        "per_setting": rows,
        "random_shortlist": random_rows,
        "detail_record_count": len(detail_records),
    }
    _write_json(suite_root / "ablation_fixed_slate_diagnostics.json", diagnostics)
    write_smoke_outputs(suite_root, average, detail_records, incomplete)


def write_smoke_outputs(
    suite_root: Path,
    average_rows: Sequence[Dict[str, Any]],
    detail_records: Sequence[Dict[str, Any]],
    incomplete: Sequence[Dict[str, Any]],
) -> None:
    _write_jsonl(suite_root / "smoke_details.jsonl", detail_records)
    table_rows = [
        [
            row.get("Variant"),
            _fmt_pct(row.get("Avg Acc")),
            _fmt_pct(row.get("Avg JCR")),
            _fmt_pct(row.get("Selected ID Parse Success")),
            _fmt_pct(row.get("Dense ID Parse Success")),
            _fmt_pct(row.get("Fallback Rate")),
            _fmt_num(row.get("Average Payload Tokens"), 1),
            _fmt_num(row.get("Average Shortlist Size"), 1),
            str(row.get("JCR denominator") or 0),
        ]
        for row in average_rows
    ]
    metric_table = _markdown_table(
        [
            "Variant",
            "Acc",
            "JCR",
            "Selected Parse",
            "Dense Parse",
            "Fallback",
            "Payload Tokens",
            "Shortlist Size",
            "JCR Den.",
        ],
        table_rows,
    )
    (suite_root / "smoke_metric_table.md").write_text(
        "# Smoke Metric Table\n\n" + metric_table + "\n",
        encoding="utf-8",
    )
    required_fields = [
        "selected_candidate_id",
        "dense_selected_candidate_id",
        "parse_success",
        "dense_parse_success",
        "jcr_match",
        "correct",
        "fallback_used",
        "payload_token_count",
        "shortlist_size",
    ]
    missing_counts = {
        field: sum(1 for record in detail_records if record.get(field) is None)
        for field in required_fields
    }
    random_records = [record for record in detail_records if record.get("random_seed") is not None]
    status = "PASS" if detail_records and not incomplete else "CHECK"
    if any(count == len(detail_records) for count in missing_counts.values()) and detail_records:
        status = "CHECK"
    summary = [
        "# Smoke Summary",
        "",
        f"- Status: `{status}`",
        f"- Detail records: `{len(detail_records)}`",
        f"- Incomplete entries: `{len(incomplete)}`",
        f"- Random-seed records: `{len(random_records)}`",
        "",
        "## Missing Field Counts",
        "",
        _markdown_table(["Field", "Missing"], [[field, str(count)] for field, count in missing_counts.items()]),
        "",
        "## Metrics",
        "",
        metric_table,
        "",
    ]
    (suite_root / "smoke_summary.md").write_text("\n".join(summary), encoding="utf-8")


def collect_baseline_rows(baseline_root: Optional[Path]) -> List[Dict[str, Any]]:
    if baseline_root is None or not baseline_root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(baseline_root.glob("*/*/*/*_summary.json")):
        block = summary_path.parent
        parts = block.relative_to(baseline_root).parts
        if len(parts) < 3:
            continue
        dataset, regime, shuffle = parts[:3]
        summary = _read_json(summary_path)
        rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "shuffle": shuffle,
                "variant": summary.get("method_label") or summary.get("method"),
                "Acc": summary.get("Acc"),
                "JCR": summary.get("JCR"),
                "Parse Success Rate": summary.get("parse_coverage"),
                "Fallback Rate": summary.get("fallback_rate"),
                "Average Payload Tokens": None,
                "JCR denominator": summary.get("jcr_den"),
            }
        )
    return rows


def collect_fixed_dense_rows(fixed_root: Optional[Path]) -> List[Dict[str, Any]]:
    if fixed_root is None or not fixed_root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for config_path in sorted(fixed_root.glob("*/*/*/*/ccr_ablation_config.json")):
        config = _read_json(config_path)
        parts = _setting_parts(config)
        summary = _load_summary(config_path.parent, "dense")
        if summary is None:
            continue
        rows.append(
            {
                "dataset": parts["dataset"],
                "regime": parts["regime"],
                "shuffle": parts["shuffle"],
                "variant": "Dense Prefill",
                "Acc": summary.get("Acc"),
                "JCR": None,
                "Parse Success Rate": summary.get("parse_coverage"),
                "Fallback Rate": summary.get("fallback_rate"),
                "Average Payload Tokens": None,
                "JCR denominator": None,
            }
        )
    return rows


def write_variant_upgrade_outputs(
    suite_root: Path,
    fixed_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
) -> None:
    selected = [
        "Dense Prefill",
        "Naive Reuse",
        "KVCOMM",
        "PAL-KV",
        FULL_LABEL,
        "CCR-Judge w/o Support Scoring",
        "CCR-Judge w/o Shortlist Compression",
        "CCR-Judge w/o Comparative Payload Statistics",
        "CCR-Judge Stats Only",
    ]
    upgrade_rows = [
        row for row in baseline_rows if str(row.get("variant")) in selected
    ] + [
        row for row in fixed_rows if str(row.get("variant")) in selected or str(row.get("variant_slug", "")).startswith(RANDOM_PREFIX)
    ]
    deduped_upgrade_rows: List[Dict[str, Any]] = []
    seen_upgrade_keys = set()
    for row in upgrade_rows:
        key = (
            row.get("dataset"),
            row.get("regime"),
            row.get("shuffle"),
            row.get("variant"),
        )
        if key in seen_upgrade_keys:
            continue
        seen_upgrade_keys.add(key)
        deduped_upgrade_rows.append(row)
    upgrade_rows = deduped_upgrade_rows
    headers = [
        "dataset",
        "regime",
        "shuffle",
        "variant",
        "Acc",
        "JCR",
        "Parse Success Rate",
        "Fallback Rate",
        "Average Payload Tokens",
        "JCR denominator",
    ]
    _write_csv(
        suite_root / "variant_upgrade_check.csv",
        headers,
        [{key: _csv_float(row.get(key)) for key in headers} for row in upgrade_rows],
    )
    md_rows = [
        [
            row.get("dataset"),
            row.get("regime"),
            row.get("shuffle"),
            row.get("variant"),
            _fmt_pct(row.get("Acc")),
            _fmt_pct(row.get("JCR")),
            _fmt_num(row.get("Average Payload Tokens"), 1),
        ]
        for row in upgrade_rows
    ]
    (suite_root / "variant_upgrade_check.md").write_text(
        "# Variant Upgrade Check\n\n"
        + _markdown_table(["Dataset", "Regime", "Shuffle", "Variant", "Acc", "JCR", "Payload Len."], md_rows)
        + "\n",
        encoding="utf-8",
    )

    aggregates = aggregate_fixed(list(fixed_rows))
    full = next((row for row in aggregates if row["Variant"] == FULL_LABEL), None)
    no_support = next((row for row in aggregates if row["Variant"] == "CCR-Judge w/o Support Scoring"), None)
    no_shortlist = next((row for row in aggregates if row["Variant"] == "CCR-Judge w/o Shortlist Compression"), None)

    bullets = []
    recommendation = "Keep CCR-Judge Full as the default unless a variant dominates on Acc/JCR without a payload-cost increase."
    if full and no_shortlist:
        jcr_gain = _float(no_shortlist.get("Delta JCR vs Full"))
        payload_gain = None
        if _float(full.get("Average Payload Tokens")) is not None and _float(no_shortlist.get("Average Payload Tokens")) is not None:
            payload_gain = float(no_shortlist["Average Payload Tokens"]) - float(full["Average Payload Tokens"])
        if jcr_gain is not None and jcr_gain > 0 and payload_gain is not None and payload_gain > 0:
            bullets.append(
                "w/o Shortlist Compression improves JCR but increases payload length; treat it as a no-compression fidelity upper-bound diagnostic rather than a compact default."
            )
        elif jcr_gain is not None and jcr_gain > 0:
            bullets.append(
                "w/o Shortlist Compression improves JCR without a clear payload penalty in completed settings; consider a follow-up default-upgrade check."
            )
    if full and no_support:
        acc_delta = _float(no_support.get("Delta Acc vs Full"))
        jcr_delta = _float(no_support.get("Delta JCR vs Full"))
        bullets.append(
            f"w/o Support Scoring average deltas: Acc {_fmt_pct(acc_delta, signed=True)}, JCR {_fmt_pct(jcr_delta, signed=True)}."
        )
    if not bullets:
        bullets.append("Insufficient completed evidence for an upgrade decision.")
    (suite_root / "variant_upgrade_decision_report.md").write_text(
        "# Variant Upgrade Decision Report\n\n"
        + "\n".join(f"- {item}" for item in bullets)
        + f"\n\nRecommendation: {recommendation}\n",
        encoding="utf-8",
    )


def write_summary_and_report(
    suite_root: Path,
    fixed_rows: Sequence[Dict[str, Any]],
    incomplete: Sequence[Dict[str, Any]],
    model: str,
    online_root: Optional[Path],
    expected_datasets: Sequence[str],
    expected_regimes: Sequence[str],
    expected_orderings: Sequence[str],
) -> None:
    aggregates = aggregate_fixed(list(fixed_rows))
    completed_settings = {
        (str(row.get("dataset")), str(row.get("regime")), str(row.get("shuffle")))
        for row in fixed_rows
    }
    expected_settings = [
        (dataset, regime, ordering)
        for dataset in expected_datasets
        for regime in expected_regimes
        for ordering in expected_orderings
    ]
    missing_settings = [
        "/".join(setting)
        for setting in expected_settings
        if setting not in completed_settings
    ]
    completed_setting_text = ", ".join("/".join(setting) for setting in sorted(completed_settings)) or "none"
    missing_setting_text = ", ".join(missing_settings) if missing_settings else "none"
    fixed_md = suite_root / "ablation_fixed_slate_summary.md"
    fixed_md.write_text(
        "# Fixed-Slate Ablation Summary\n\n"
        f"- Model: `{model}`\n"
        f"- Expected datasets: `{' '.join(expected_datasets)}`\n"
        f"- Expected regimes: `{' '.join(expected_regimes)}`\n"
        f"- Expected orderings: `{' '.join(expected_orderings)}`\n"
        f"- Completed settings: `{completed_setting_text}`\n"
        f"- Missing expected settings: `{missing_setting_text}`\n"
        f"- Completed variant-setting rows: `{len(fixed_rows)}`\n"
        f"- Incomplete entries: `{len(incomplete)}`\n\n"
        "## Main Table\n\n"
        + (suite_root / "ablation_fixed_slate_average.md").read_text(encoding="utf-8").split("\n\n", 1)[-1]
        + "\n",
        encoding="utf-8",
    )

    online_note = "Online validation not completed."
    if online_root and (online_root / "ablation_online_mmlu_average.md").exists():
        online_note = (online_root / "ablation_online_mmlu_average.md").read_text(encoding="utf-8")

    random = random_seed_rows(fixed_rows)
    random_summary_count = sum(1 for row in random if row.get("seed") == "mean/std")
    payload_tradeoff = next(
        (row for row in aggregates if row["Variant"] == "CCR-Judge w/o Shortlist Compression"),
        None,
    )
    payload_line = (
        "No w/o Shortlist Compression results completed."
        if payload_tradeoff is None
        else (
            "w/o Shortlist Compression: "
            f"Delta JCR {_fmt_pct(payload_tradeoff.get('Delta JCR vs Full'), signed=True)}, "
            f"payload len {_fmt_num(payload_tradeoff.get('Average Payload Tokens'), 1)}."
        )
    )
    report = f"""# CCR-Judge Ablation Experiment Report

## 1. Experiment Scope

- Model: `{model}`
- Expected fixed-slate datasets: `{' '.join(expected_datasets)}`
- Expected fixed-slate regimes: `{' '.join(expected_regimes)}`
- Expected fixed-slate orderings: `{' '.join(expected_orderings)}`
- Completed fixed-slate settings: `{completed_setting_text}`
- Missing expected fixed-slate settings: `{missing_setting_text}`
- Fixed-slate completed rows: `{len(fixed_rows)}`
- Incomplete entries: `{len(incomplete)}`
- Result root: `{suite_root}`

## 2. Fixed-Slate Average Ablation

See `ablation_fixed_slate_average.md`. The paper main table should use:

| Variant | Avg Acc | Delta Acc | Avg JCR | Delta JCR | Payload Len. |
| --- | --- | --- | --- | --- | --- |
"""
    for row in aggregates:
        report += (
            f"| {row['Variant']} | {_fmt_pct(row.get('Avg Acc'))} | "
            f"{_fmt_pct(row.get('Delta Acc vs Full'), signed=True)} | "
            f"{_fmt_pct(row.get('Avg JCR'))} | "
            f"{_fmt_pct(row.get('Delta JCR vs Full'), signed=True)} | "
            f"{_fmt_num(row.get('Average Payload Tokens'), 1)} |\n"
        )
    report += f"""
## 3. Per-Setting Ablation

See `ablation_fixed_slate_per_setting.md` for the appendix table. Incomplete settings are listed in `ablation_fixed_slate_diagnostics.json`.

## 4. Random Shortlist Analysis

Random shortlist detail and mean/std rows are in `ablation_fixed_slate_random_shortlist_seeds.csv`.
Completed random mean/std setting rows: `{random_summary_count}`.

## 5. Payload Compression Trade-off

{payload_line}

If no-compression improves JCR while increasing payload length, interpret it as a fidelity upper-bound diagnostic rather than the compact default.

## 6. Payload Statistics Effect

Compare `CCR-Judge (Full)`, `CCR-Judge w/o Comparative Payload Statistics`, and `CCR-Judge Stats Only` in the average and per-setting tables. This isolates whether explicit slate-level statistics carry selected-candidate consistency.

## 7. Online Validation

{online_note}

## 8. Upgrade Recommendation

See `variant_upgrade_decision_report.md`.

## 9. Paper-Ready Summary

CCR-Judge's ablation results should be interpreted as evidence about comparative decision-context repair rather than as a monotonic accuracy-improving heuristic. The controlled fixed-slate diagnostic isolates judge-side final selection and measures how candidate-level consistency changes when comparative grouping, support scoring, payload statistics, and shortlist compression are removed.

The expected paper framing is that structured comparative state matters, but the components expose a compactness--fidelity trade-off. In particular, a no-compression variant can improve selected-candidate fidelity by retaining more candidate information, but this should be treated as a diagnostic upper bound unless it preserves compactness and stability across settings.
"""
    (suite_root / "ablation_experiment_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).expanduser().resolve()
    fixed_root = Path(args.fixed_root).expanduser().resolve() if args.fixed_root else suite_root / "fixed_slate"
    baseline_root = Path(args.baseline_root).expanduser().resolve() if args.baseline_root else suite_root / "variant_upgrade_baselines"
    online_root = Path(args.online_root).expanduser().resolve() if args.online_root else suite_root / "online_mmlu"
    suite_root.mkdir(parents=True, exist_ok=True)

    fixed_rows, incomplete, fixed_detail_records = collect_fixed_rows(fixed_root)
    write_fixed_outputs(suite_root, fixed_rows, incomplete, fixed_detail_records)
    baseline_rows = collect_fixed_dense_rows(fixed_root) + collect_baseline_rows(baseline_root)
    write_variant_upgrade_outputs(suite_root, fixed_rows, baseline_rows)
    write_summary_and_report(
        suite_root=suite_root,
        fixed_rows=fixed_rows,
        incomplete=incomplete,
        model=args.model,
        online_root=online_root if online_root.exists() else None,
        expected_datasets=[part for part in args.expected_datasets.split() if part],
        expected_regimes=[part for part in args.expected_regimes.split() if part],
        expected_orderings=[part for part in args.expected_orderings.split() if part],
    )
    print(f"[REPORT SAVED] {suite_root / 'ablation_experiment_report.md'}")


if __name__ == "__main__":
    main()
