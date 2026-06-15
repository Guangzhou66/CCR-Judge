from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRITY_TARGETS = [
    PROJECT_ROOT / "experiments" / "run_mmlu_ori_protocol.py",
    PROJECT_ROOT / "experiments" / "run_gsm8k_ori_protocol.py",
    PROJECT_ROOT / "experiments" / "run_humaneval_ori_protocol.py",
    PROJECT_ROOT / "run_all_online_ori_protocol_extension_formal.sh",
    PROJECT_ROOT / "result" / "online_ori_protocol_extension_20260420" / "configs" / "frozen_extension_config.yaml",
    PROJECT_ROOT / "result" / "online_ori_protocol_extension_20260420" / "manifests" / "updated_run_manifest.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight check for paper_repair formal mainline launch.")
    parser.add_argument("--llm-name", required=True)
    parser.add_argument("--result-root", required=True)
    return parser.parse_args()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_integrity() -> Dict[str, str]:
    return {str(path): sha256(path) for path in INTEGRITY_TARGETS}


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser()
    reports_root = result_root / "reports"
    candidate_cache_root = result_root / "candidate_cache"
    formal_root = result_root / "formal"

    config_path = PROJECT_ROOT / "result" / "paper_repair_mainline_20260420" / "configs" / "paper_repair_frozen_config.yaml"
    manifest_path = PROJECT_ROOT / "result" / "paper_repair_mainline_20260420" / "manifests" / "paper_repair_run_manifest.md"
    result_table_template = PROJECT_ROOT / "result" / "paper_repair_mainline_20260420" / "reports" / "paper_repair_result_tables.md"
    code_changes_summary = PROJECT_ROOT / "result" / "paper_repair_mainline_20260420" / "reports" / "code_changes_summary.md"

    blockers: List[str] = []

    for path in [config_path, manifest_path, result_table_template, code_changes_summary]:
        if not path.exists():
            blockers.append(f"Missing required artifact: {path}")

    model_path = Path(args.llm_name).expanduser()
    if not model_path.exists():
        blockers.append(f"Formal mainline model path does not exist: {model_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_modes = {
        "mmlu": config["datasets"]["mmlu"]["scorer_mode"],
        "gsm8k": config["datasets"]["gsm8k"]["scorer_mode"],
        "humaneval": config["datasets"]["humaneval"]["scorer_mode"],
    }
    expected_modes = {
        "mmlu": {"mmlu_text_match_parity_aligned", "mmlu_final_text_choice"},
        "gsm8k": {"ori_parity"},
        "humaneval": {"ori_pyexecutor"},
    }
    for dataset, expected in expected_modes.items():
        if dataset_modes.get(dataset) not in expected:
            blockers.append(
                f"Unexpected scorer_mode in config for {dataset}: expected one of {sorted(expected)}, "
                f"got {dataset_modes.get(dataset)}"
            )

    if config["runtime"]["execution_side_reuse"] is not False:
        blockers.append("Frozen config no longer records execution_side_reuse=false.")

    integrity_payload = snapshot_integrity()

    reports_root.mkdir(parents=True, exist_ok=True)
    candidate_cache_root.mkdir(parents=True, exist_ok=True)
    formal_root.mkdir(parents=True, exist_ok=True)
    json_dump(reports_root / "_ori_protocol_integrity_preflight.json", integrity_payload)

    status = "pass" if not blockers else "fail"

    preflight_lines = [
        "# Paper Repair Formal Preflight",
        "",
        f"Formal mainline model pin: `{args.llm_name}`",
        f"Formal result root: `{result_root}`",
        f"Candidate cache root: `{candidate_cache_root}`",
        f"Formal root: `{formal_root}`",
        f"Reports root: `{reports_root}`",
        "",
        "## Checked Inputs",
        "",
        f"- config: `{config_path}`",
        f"- manifest: `{manifest_path}`",
        f"- result table template: `{result_table_template}`",
        f"- code changes summary: `{code_changes_summary}`",
        "",
        "## Model Pin",
        "",
        f"- formal mainline model = `{args.llm_name}`",
        "- local_alt_model evidence line remains `/pychen/Test/model/Qwen2.5-7B-Instruct`",
        "",
        "## Scorer Modes",
        "",
        f"- MMLU: `{dataset_modes['mmlu']}`",
        f"- GSM8K: `{dataset_modes['gsm8k']}`",
        f"- HumanEval: `{dataset_modes['humaneval']}`",
        "",
        "## Ori-Protocol Integrity Baseline",
        "",
        "- current hashes were snapshotted before launch",
        f"- snapshot path: `{reports_root / '_ori_protocol_integrity_preflight.json'}`",
        "",
        "## Result",
        "",
        f"- preflight status: `{status}`",
    ]
    if blockers:
        preflight_lines.extend(["", "## Blockers", ""])
        for item in blockers:
            preflight_lines.append(f"- {item}")
    else:
        preflight_lines.extend(
            [
                "",
                "## Ready",
                "",
                "- config / manifest / path surface is readable",
                "- scorer modes match expected paper_repair semantics",
                "- formal mainline model pin is explicit",
                "- ori_protocol baseline is preserved",
            ]
        )

    (reports_root / "paper_repair_formal_preflight.md").write_text(
        "\n".join(preflight_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    json_dump(
        reports_root / "paper_repair_formal_preflight.json",
        {
            "status": status,
            "model_path": args.llm_name,
            "result_root": str(result_root),
            "candidate_cache_root": str(candidate_cache_root),
            "formal_root": str(formal_root),
            "reports_root": str(reports_root),
            "blockers": blockers,
            "scorer_modes": dataset_modes,
            "ori_protocol_integrity": integrity_payload,
        },
    )
    if blockers:
        blocker_lines = [
            "# Paper Repair Formal Blockers",
            "",
            "Preflight failed. Formal was not launched.",
            "",
        ]
        for item in blockers:
            blocker_lines.append(f"- {item}")
        (reports_root / "paper_repair_formal_blockers.md").write_text(
            "\n".join(blocker_lines).rstrip() + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
