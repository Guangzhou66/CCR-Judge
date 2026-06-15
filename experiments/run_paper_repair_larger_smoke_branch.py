from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging  # noqa: E402
from paper_repair.protocol import build_frozen_candidate_pack, load_question_indices, run_judge_benchmark  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GSM8K_DATASET_ROOT = str(PROJECT_ROOT / "data" / "gsm8k")
DEFAULT_HUMANEVAL_JSON = str(PROJECT_ROOT / "my_datasets" / "humaneval" / "humaneval-py.jsonl")
DEFAULT_METHODS = ["dense", "naive_reuse", "kvcomm", "pal_kv", "ccr_judge"]
DEFAULT_REGIMES = ["parallel_exploration", "progressive_refinement"]
DEFAULT_ORDERINGS = ["noshuffle", "shuffle"]
DEFAULT_MMLU_INDEX_FILE = PROJECT_ROOT / "experiments" / "mmlu_153_seed888_indices.json"
INTEGRITY_TARGETS = [
    PROJECT_ROOT / "experiments" / "run_mmlu_ori_protocol.py",
    PROJECT_ROOT / "experiments" / "run_gsm8k_ori_protocol.py",
    PROJECT_ROOT / "experiments" / "run_humaneval_ori_protocol.py",
    PROJECT_ROOT / "run_all_online_ori_protocol_extension_formal.sh",
    PROJECT_ROOT / "result" / "online_ori_protocol_extension_20260420" / "configs" / "frozen_extension_config.yaml",
    PROJECT_ROOT / "result" / "online_ori_protocol_extension_20260420" / "manifests" / "updated_run_manifest.md",
]
DATASET_SPECS = {
    "mmlu": {
        "split": "val",
        "scorer_mode": "mmlu_text_match_parity_aligned",
        "scoring_branch": "selected_candidate_passed",
        "allowed_scoring_branches": {"selected_candidate_passed", "fallback_final_answer"},
        "official_policy": "mmlu_text_match_parity_aligned",
        "sample_key": "question_id",
        "builder_kwargs": {
            "agent_name": "AnalyzeAgent",
            "agent_role": "MMLU Solver",
            "agent_temperature": 0.2,
            "agent_count": 4,
            "seed": 888,
        },
    },
    "gsm8k": {
        "split": "test",
        "scorer_mode": "ori_parity",
        "scoring_branch": "ori_parity_final_text",
        "official_policy": "ori_parity",
        "sample_key": "question_id",
        "builder_kwargs": {
            "agent_name": "MathSolver",
            "agent_role": "Math Solver",
            "agent_temperature": 0.2,
            "agent_count": 4,
            "seed": 42,
            "adapter_kwargs": {
                "dataset_root": DEFAULT_GSM8K_DATASET_ROOT,
                "config_name": "main",
            },
        },
    },
    "humaneval": {
        "split": "test",
        "scorer_mode": "ori_pyexecutor",
        "scoring_branch": "ori_pyexecutor_final_code",
        "official_policy": "ori_pyexecutor",
        "sample_key": "task_id",
        "builder_kwargs": {
            "agent_name": "CodeWriting",
            "agent_role": "Programming Expert",
            "agent_temperature": 0.2,
            "agent_count": 4,
            "seed": 42,
            "adapter_kwargs": {
                "dataset_json": DEFAULT_HUMANEVAL_JSON,
            },
        },
    },
}
COMMON_REQUIRED_FIELDS = [
    "dataset",
    "sample_id",
    "regime",
    "shuffle_flag",
    "method",
    "final_result_text",
    "final_answer_text",
    "candidate_texts",
    "candidate_text_hash",
    "candidate_pack_hash",
    "candidate_build_mode",
    "execution_side_reuse",
    "selected_agent_id_raw",
    "selected_agent_id_original_space",
    "dense_selected_agent_id_raw",
    "dense_selected_agent_id_original_space",
    "selected_agent_parse_success",
    "permutation",
    "inverse_permutation",
    "official_scoring_policy",
    "official_is_correct",
    "scoring_branch",
    "selected_candidate_passed",
    "fallback_used",
    "reuse_rate",
    "reuse_candidate_block_count",
    "dense_recompute_candidate_block_count",
    "kvcomm_reuse_event",
    "pal_kv_reuse_event",
    "anchor_pool_scope",
    "gating_decision",
    "parse_coverage",
    "any_candidate_upper_bound",
    "selection_gap",
    "answer_jcr_if_available",
]
DATASET_REQUIRED_FIELDS = {
    "mmlu": COMMON_REQUIRED_FIELDS + ["question_id"],
    "gsm8k": COMMON_REQUIRED_FIELDS + [
        "question_id",
        "official_gsm8k_number",
        "extracted_final_number",
        "diagnostic_gsm8k_number",
        "normalized_final_answer",
    ],
    "humaneval": COMMON_REQUIRED_FIELDS + [
        "task_id",
        "extracted_code",
        "official_humaneval_passed",
        "subprocess_humaneval_passed",
        "executor_mode",
        "execution_error_type",
    ],
}
METHOD_LABELS = {
    "dense": "Dense Prefill",
    "naive_reuse": "Naive Reuse",
    "kvcomm": "KVCOMM",
    "pal_kv": "PAL-KV",
    "ccr_judge": "CCR-Judge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a branch-isolated larger smoke for paper_repair.")
    parser.add_argument("--llm-name", required=True)
    parser.add_argument("--model-branch", required=True, choices=["local_alt_model", "paper_default_model"])
    parser.add_argument("--report-prefix", required=True, choices=["qwen", "llama"])
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--mmlu-count", type=int, default=10)
    parser.add_argument("--gsm8k-count", type=int, default=10)
    parser.add_argument("--humaneval-count", type=int, default=10)
    parser.add_argument("--mmlu-index-file", type=str, default=str(DEFAULT_MMLU_INDEX_FILE))
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_integrity(path: Path) -> None:
    payload = {str(item): sha256(item) for item in INTEGRITY_TARGETS}
    json_dump(path, payload)


def write_manifest_row(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def selected_mmlu_indices(index_file: str, count: int) -> List[int]:
    return load_question_indices(index_file=index_file, limit_questions=count)


def build_sample_map(args: argparse.Namespace) -> Dict[str, List[int]]:
    return {
        "mmlu": selected_mmlu_indices(args.mmlu_index_file, args.mmlu_count),
        "gsm8k": list(range(args.gsm8k_count)),
        "humaneval": list(range(args.humaneval_count)),
    }


def safe_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def run_step(
    *,
    manifest_path: Path,
    payload: Dict[str, Any],
    coro: Any,
) -> Any:
    start_utc = utc_now()
    exit_status = 0
    error_message = None
    result = None
    try:
        result = await coro
    except Exception as exc:  # pragma: no cover - runtime orchestration path
        exit_status = 1
        error_message = f"{type(exc).__name__}: {exc}"
    end_utc = utc_now()
    row = dict(payload)
    row.update(
        {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "exit_status": exit_status,
        }
    )
    if error_message is not None:
        row["error_message"] = error_message
    write_manifest_row(manifest_path, row)
    if exit_status != 0:
        raise RuntimeError(error_message or "unknown step failure")
    safe_gc()
    return result


def branch_metadata(
    *,
    args: argparse.Namespace,
    result_root: Path,
    sample_map: Dict[str, List[int]],
) -> Dict[str, Any]:
    return {
        "model_path": args.llm_name,
        "model_branch": args.model_branch,
        "report_prefix": args.report_prefix,
        "result_root": str(result_root),
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kvcomm_force_bfloat16": os.environ.get("KVCOMM_FORCE_BFLOAT16"),
        "datasets": {
            "mmlu": {
                "sample_ids": sample_map["mmlu"],
                "scorer_mode": DATASET_SPECS["mmlu"]["scorer_mode"],
            },
            "gsm8k": {
                "sample_ids": sample_map["gsm8k"],
                "scorer_mode": DATASET_SPECS["gsm8k"]["scorer_mode"],
            },
            "humaneval": {
                "sample_ids": sample_map["humaneval"],
                "scorer_mode": DATASET_SPECS["humaneval"]["scorer_mode"],
            },
        },
        "methods": list(DEFAULT_METHODS),
        "regimes": list(DEFAULT_REGIMES),
        "orderings": list(DEFAULT_ORDERINGS),
        "purpose": "larger_smoke",
    }


def load_pack_hashes(result_root: Path) -> Dict[str, str]:
    hashes = {}
    for dataset in DATASET_SPECS:
        for regime in DEFAULT_REGIMES:
            pack_path = result_root / "packs" / f"{dataset}_{regime}_pack.json"
            if pack_path.exists():
                payload = json.loads(pack_path.read_text(encoding="utf-8"))
                hashes[f"{dataset}:{regime}"] = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
    return hashes


def validation_summary(result_root: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    benchmark_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    manifest_rows = [
        json.loads(line)
        for line in (result_root / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_ok = all(int(row.get("exit_status", 1)) == 0 for row in manifest_rows)

    for dataset in DATASET_SPECS:
        for regime in DEFAULT_REGIMES:
            for ordering in DEFAULT_ORDERINGS:
                summary_path = result_root / "eval" / dataset / regime / ordering / "benchmark_summary.json"
                if summary_path.exists():
                    benchmark_rows[dataset].append(json.loads(summary_path.read_text(encoding="utf-8")))
                for method in DEFAULT_METHODS:
                    details_path = result_root / "eval" / dataset / regime / ordering / f"{method}_details.json"
                    if details_path.exists():
                        rows_by_dataset[dataset].extend(json.loads(details_path.read_text(encoding="utf-8")))

    before = json.loads(
        (result_root / "reports" / "_ori_protocol_integrity_before_larger_smoke.json").read_text(encoding="utf-8")
    )
    after = json.loads(
        (result_root / "reports" / "_ori_protocol_integrity_after_larger_smoke.json").read_text(encoding="utf-8")
    )

    pack_hashes = load_pack_hashes(result_root)
    dataset_results: Dict[str, Any] = {}
    hard_blocker = False
    soft_blockers: List[str] = []
    confidence_risks: List[str] = []

    for dataset, rows in rows_by_dataset.items():
        field_missing = defaultdict(list)
        for row in rows:
            for field in DATASET_REQUIRED_FIELDS[dataset]:
                if field not in row:
                    field_missing[field].append(row.get("sample_id"))

        sharing_fail = []
        by_key: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_key[(row["regime"], str(row["sample_id"]))].append(row)
        for key, group in by_key.items():
            pack_hash_set = {item["candidate_pack_hash"] for item in group}
            candidate_hash_set = {item["candidate_text_hash"] for item in group}
            text_set = {json.dumps(item["candidate_texts"], ensure_ascii=False) for item in group}
            if len(pack_hash_set) != 1 or len(candidate_hash_set) != 1 or len(text_set) != 1:
                sharing_fail.append({"key": key, "pack_hashes": sorted(pack_hash_set)})

        expected_scoring_branch = DATASET_SPECS[dataset]["scoring_branch"]
        allowed_scoring_branches = set(DATASET_SPECS[dataset].get("allowed_scoring_branches") or {expected_scoring_branch})
        expected_policy = DATASET_SPECS[dataset]["official_policy"]
        scoring_fail = []
        selected_diff = 0
        for row in rows:
            if row.get("official_is_correct") != row.get("selected_candidate_passed"):
                selected_diff += 1
            if row.get("scoring_branch") not in allowed_scoring_branches:
                scoring_fail.append((row.get("sample_id"), row.get("method"), row.get("scoring_branch")))
            if row.get("official_scoring_policy") != expected_policy:
                scoring_fail.append((row.get("sample_id"), row.get("method"), row.get("official_scoring_policy")))
            if dataset == "humaneval":
                if row.get("executor_mode") != "ori_pyexecutor":
                    scoring_fail.append((row.get("sample_id"), row.get("method"), row.get("executor_mode")))
                if bool(row.get("official_humaneval_passed")) != bool(row.get("official_is_correct")):
                    scoring_fail.append((row.get("sample_id"), row.get("method"), "official_humaneval_mismatch"))

        perm_fail = []
        shuffle_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("shuffle_flag"):
                shuffle_groups[(row["regime"], str(row["sample_id"]))].append(row)
        for key, group in shuffle_groups.items():
            perms = {json.dumps(item["permutation"], sort_keys=True) for item in group}
            invs = {json.dumps(item["inverse_permutation"], sort_keys=True) for item in group}
            if len(perms) != 1 or len(invs) != 1:
                perm_fail.append(key)

        jcr_fail = []
        for row in rows:
            both_parse = bool(row.get("selected_agent_parse_success")) and bool(row.get("dense_selected_agent_parse_success"))
            if both_parse:
                expected = str(row.get("selected_agent_id_original_space")) == str(row.get("dense_selected_agent_id_original_space"))
                if bool(row.get("jcr_match")) != expected:
                    jcr_fail.append((row.get("sample_id"), row.get("method")))

        reuse_fail = []
        for row in rows:
            if row.get("method") == "dense":
                if float(row.get("reuse_rate", 0.0)) != 0.0:
                    reuse_fail.append((row.get("sample_id"), "dense_reuse_rate"))
                if int(row.get("reuse_candidate_block_count", 0)) != 0:
                    reuse_fail.append((row.get("sample_id"), "dense_reuse_block_count"))
            else:
                if int(row.get("reuse_candidate_block_count", 0)) != int(row.get("candidate_count", 0)):
                    reuse_fail.append((row.get("sample_id"), row.get("method"), "reuse_candidate_block_count"))
                if int(row.get("dense_recompute_candidate_block_count", 0)) != 0:
                    reuse_fail.append((row.get("sample_id"), row.get("method"), "dense_recompute_candidate_block_count"))

        pal_fail = []
        method_presence = {(item["regime"], bool(item["shuffle_flag"]), item["method"]) for item in rows}
        for regime in DEFAULT_REGIMES:
            for shuffle_flag in [False, True]:
                for method in DEFAULT_METHODS:
                    if (regime, shuffle_flag, method) not in method_presence:
                        pal_fail.append((regime, shuffle_flag, method, "missing_method"))
        for row in rows:
            if row.get("method") == "pal_kv":
                if row.get("anchor_pool_scope") != "group_judge_anchors" or row.get("gating_decision") != "group_anchor_enabled":
                    pal_fail.append((row.get("sample_id"), "pal_kv", row.get("anchor_pool_scope"), row.get("gating_decision")))
            if row.get("method") == "kvcomm":
                if row.get("anchor_pool_scope") != "kvcomm_default" or row.get("gating_decision") != "judge_kv_reuse_enabled":
                    pal_fail.append((row.get("sample_id"), "kvcomm", row.get("anchor_pool_scope"), row.get("gating_decision")))

        parse_values = [
            float(item.get("parse_coverage", 0.0))
            for item in rows
            if item.get("method") == "dense"
        ]
        fallback_values = [
            float(bool(item.get("fallback_used")))
            for item in rows
            if item.get("method") == "dense"
        ]

        dataset_status = {
            "fixed_candidate_sharing": not sharing_fail,
            "execution_side_reuse_disabled": all(
                int(item["candidate_count"]) == 4 and item.get("execution_side_reuse") is False and item.get("candidate_build_mode") == "dense_prefill_only"
                for item in rows
            ),
            "official_scoring_boundary": not scoring_fail,
            "shuffle_identical_permutations": not perm_fail,
            "jcr_original_ids": not jcr_fail,
            "reuse_definition_correct": not reuse_fail,
            "pal_kv_baseline_separation": not pal_fail,
            "ori_protocol_isolation": before == after,
        }
        dataset_pass = all(dataset_status.values()) and not field_missing
        if not dataset_pass:
            hard_blocker = True
        if dataset == "humaneval":
            all_acc_zero = all(
                abs(float(method_row.get("Acc", 0.0))) < 1e-12
                for bench in benchmark_rows[dataset]
                for method_row in bench.get("methods", [])
            )
            if all_acc_zero:
                confidence_risks.append("HumanEval larger smoke still shows all-zero Pass@1 / Acc across the method matrix.")
            if any(value < 1.0 for value in parse_values):
                confidence_risks.append("HumanEval Dense Prefill parse coverage is not perfectly flat in larger smoke.")
            if any(value > 0.0 for value in fallback_values):
                confidence_risks.append("HumanEval Dense Prefill still uses fallback on part of the larger smoke slice.")

        dataset_results[dataset] = {
            "sample_ids": meta["datasets"][dataset]["sample_ids"],
            "scorer_mode": meta["datasets"][dataset]["scorer_mode"],
            "pack_hashes": {
                regime: pack_hashes.get(f"{dataset}:{regime}")
                for regime in DEFAULT_REGIMES
            },
            "selected_candidate_passed_diff_count": selected_diff,
            "status": dataset_status,
            "pass": dataset_pass,
            "field_missing_counts": {key: len(value) for key, value in field_missing.items()},
            "sharing_failures": sharing_fail[:5],
            "scoring_failures": scoring_fail[:5],
            "permutation_failures": perm_fail[:5],
            "jcr_failures": jcr_fail[:5],
            "reuse_failures": reuse_fail[:5],
            "pal_failures": pal_fail[:5],
        }

    if not manifest_ok:
        hard_blocker = True
        soft_blockers.append("At least one run_manifest step exited non-zero.")
    if not before == after:
        hard_blocker = True
        soft_blockers.append("ori_protocol integrity hash changed during larger smoke.")

    overall_pass = manifest_ok and all(item["pass"] for item in dataset_results.values()) and not hard_blocker
    return {
        "model_path": meta["model_path"],
        "model_branch": meta["model_branch"],
        "report_prefix": meta["report_prefix"],
        "result_root": str(result_root),
        "manifest_ok": manifest_ok,
        "overall_pass": overall_pass,
        "hard_blocker": hard_blocker,
        "soft_blockers": soft_blockers,
        "confidence_risks": confidence_risks,
        "dataset_results": dataset_results,
        "ori_protocol_isolation": before == after,
        "ori_protocol_diff_keys": sorted(set(before.keys()) ^ set(after.keys())),
    }


def branch_run_log_text(result_root: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> str:
    manifest_rows = [
        json.loads(line)
        for line in (result_root / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines = [
        f"# {meta['report_prefix'].upper()} Larger Smoke Run Log",
        "",
        f"Date: `{datetime.now(timezone.utc).date()}`",
        "",
        f"Model path: `{meta['model_path']}`",
        f"Model branch: `{meta['model_branch']}`",
        f"Result root: `{meta['result_root']}`",
        f"Python: `{meta['python']}`",
        "",
        "Fixed sample ids used across all methods / orderings / regimes in this branch:",
    ]
    for dataset in DATASET_SPECS:
        lines.append(f"- {dataset}: `{meta['datasets'][dataset]['sample_ids']}`")
    for dataset in DATASET_SPECS:
        lines.append("")
        lines.append(f"## Dataset | {dataset}")
        for regime in DEFAULT_REGIMES:
            pack_path = result_root / "packs" / f"{dataset}_{regime}_pack.json"
            pack_hash = summary["dataset_results"].get(dataset, {}).get("pack_hashes", {}).get(regime)
            matching_build = [
                row for row in manifest_rows
                if row.get("dataset") == dataset and row.get("step_type") == "build" and row.get("regime") == regime
            ]
            if matching_build:
                row = matching_build[0]
                lines.extend(
                    [
                        "",
                        f"### Build | {regime}",
                        "",
                        f"- start: `{row['start_utc']}`",
                        f"- end: `{row['end_utc']}`",
                        f"- exit status: `{row['exit_status']}`",
                        f"- pack path: `{pack_path}`",
                        f"- candidate_pack_hash: `{pack_hash}`",
                        "- candidate_build_mode: `dense_prefill_only`",
                    ]
                )
            for ordering in DEFAULT_ORDERINGS:
                matching_judge = [
                    row for row in manifest_rows
                    if row.get("dataset") == dataset
                    and row.get("step_type") == "judge"
                    and row.get("regime") == regime
                    and row.get("ordering") == ordering
                ]
                if not matching_judge:
                    continue
                row = matching_judge[0]
                base = result_root / "eval" / dataset / regime / ordering
                lines.extend(
                    [
                        "",
                        f"### Judge | {regime} | {ordering}",
                        "",
                        f"- start: `{row['start_utc']}`",
                        f"- end: `{row['end_utc']}`",
                        f"- exit status: `{row['exit_status']}`",
                        f"- summary path: `{base / 'benchmark_summary.json'}`",
                        "- details paths:",
                    ]
                )
                for method in DEFAULT_METHODS:
                    lines.append(f"  - `{base / f'{method}_details.json'}`")
    return "\n".join(lines).rstrip() + "\n"


def branch_validation_text(result_root: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> str:
    prefix_upper = meta["report_prefix"].upper()
    larger_pass_text = "true" if summary["overall_pass"] else "false"
    branch_role_text = (
        "local_alt_model evidence line"
        if meta["model_branch"] == "local_alt_model"
        else "paper_default_model candidate line"
    )
    lines = [
        f"# {prefix_upper} Larger Smoke Validation",
        "",
        "## Overall Decision",
        "",
        f"- model_path: `{meta['model_path']}`",
        f"- model_branch: `{meta['model_branch']}`",
        f"- result root: `{meta['result_root']}`",
        f"- larger-smoke passed: `{larger_pass_text}`",
        f"- worth retaining as {branch_role_text}: `{'yes' if summary['overall_pass'] else 'no'}`",
        f"- can continue to cross-model readiness review: `{'yes' if summary['overall_pass'] else 'no'}`",
        "- can start formal now: `no`",
        "",
        "## Dataset Protocol Review",
    ]
    for dataset in DATASET_SPECS:
        result = summary["dataset_results"][dataset]
        lines.extend(
            [
                "",
                f"### {dataset}",
                "",
                f"- model_path: `{meta['model_path']}`",
                f"- model_branch: `{meta['model_branch']}`",
                f"- scorer_mode: `{result['scorer_mode']}`",
                f"- sample_ids: `{result['sample_ids']}`",
                f"- fixed candidate sharing: `{'pass' if result['status']['fixed_candidate_sharing'] else 'fail'}`",
                f"- execution_side_reuse = false: `{'pass' if result['status']['execution_side_reuse_disabled'] else 'fail'}`",
                f"- official scoring boundary: `{'pass' if result['status']['official_scoring_boundary'] else 'fail'}`",
                f"- shuffle identical permutations: `{'pass' if result['status']['shuffle_identical_permutations'] else 'fail'}`",
                f"- original-id JCR: `{'pass' if result['status']['jcr_original_ids'] else 'fail'}`",
                f"- reuse definition: `{'pass' if result['status']['reuse_definition_correct'] else 'fail'}`",
                f"- PAL-KV baseline separation: `{'pass' if result['status']['pal_kv_baseline_separation'] else 'fail'}`",
                f"- ori_protocol isolation: `{'pass' if result['status']['ori_protocol_isolation'] else 'fail'}`",
                f"- selected_candidate_passed != official_is_correct rows: `{result['selected_candidate_passed_diff_count']}`",
                "- pack hashes:",
            ]
        )
        for regime in DEFAULT_REGIMES:
            lines.append(f"  - {regime}: `{result['pack_hashes'][regime]}`")
        if result["field_missing_counts"]:
            lines.append(f"- field_missing_counts: `{result['field_missing_counts']}`")
        if result["sharing_failures"]:
            lines.append(f"- sharing_failures: `{result['sharing_failures']}`")
        if result["scoring_failures"]:
            lines.append(f"- scoring_failures: `{result['scoring_failures']}`")
        if result["permutation_failures"]:
            lines.append(f"- permutation_failures: `{result['permutation_failures']}`")
        if result["jcr_failures"]:
            lines.append(f"- jcr_failures: `{result['jcr_failures']}`")
        if result["reuse_failures"]:
            lines.append(f"- reuse_failures: `{result['reuse_failures']}`")
        if result["pal_failures"]:
            lines.append(f"- pal_failures: `{result['pal_failures']}`")
    lines.extend(
        [
            "",
            "## Branch Verdict",
            "",
            (
                f"`{prefix_upper} larger smoke passed`."
                if summary["overall_pass"]
                else f"`{prefix_upper} larger smoke failed`."
            ),
        ]
    )
    if summary["confidence_risks"]:
        lines.append("")
        lines.append("Confidence / risk notes:")
        for item in summary["confidence_risks"]:
            lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def branch_tables_text(result_root: Path, meta: Dict[str, Any]) -> str:
    rows = []
    for dataset in DATASET_SPECS:
        scorer_mode = DATASET_SPECS[dataset]["scorer_mode"]
        for regime in DEFAULT_REGIMES:
            for ordering in DEFAULT_ORDERINGS:
                summary_path = result_root / "eval" / dataset / regime / ordering / "benchmark_summary.json"
                if not summary_path.exists():
                    continue
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                for method_row in payload.get("methods", []):
                    rows.append(
                        {
                            "model_branch": meta["model_branch"],
                            "model_path": meta["model_path"],
                            "result_root": meta["result_root"],
                            "dataset": dataset,
                            "scorer_mode": scorer_mode,
                            "regime": regime,
                            "ordering": ordering,
                            "method": method_row.get("method_label"),
                            "acc": method_row.get("Acc"),
                            "jcr": method_row.get("JCR"),
                            "reuse": method_row.get("reuse"),
                            "parse_coverage": method_row.get("parse_coverage"),
                            "fallback_rate": method_row.get("fallback_rate"),
                            "any_candidate_upper_bound": method_row.get("any_candidate_upper_bound"),
                            "selection_gap": method_row.get("selection_gap"),
                        }
                    )
    columns = [
        ("model_branch", "Model Branch"),
        ("model_path", "Model Path"),
        ("result_root", "Result Root"),
        ("dataset", "Dataset"),
        ("scorer_mode", "Scorer Mode"),
        ("regime", "Regime"),
        ("ordering", "Ordering"),
        ("method", "Method"),
        ("acc", "Pass@1 / Acc"),
        ("jcr", "JCR"),
        ("reuse", "Reuse"),
        ("parse_coverage", "Parse Coverage"),
        ("fallback_rate", "Fallback Rate"),
        ("any_candidate_upper_bound", "Any-Candidate Upper Bound"),
        ("selection_gap", "Selection Gap"),
    ]
    title = f"# {meta['report_prefix'].upper()} Larger Smoke Tables\n\n"
    title += f"Model path: `{meta['model_path']}`\n\n"
    title += f"Model branch: `{meta['model_branch']}`\n\n"
    title += f"Result root: `{meta['result_root']}`\n\n"
    return title + markdown_table(columns, rows).rstrip() + "\n"


def branch_blockers_text(meta: Dict[str, Any], summary: Dict[str, Any]) -> str:
    title = f"# {meta['report_prefix'].upper()} Larger Smoke Blockers\n\n"
    title += f"Model path: `{meta['model_path']}`\n\n"
    title += f"Model branch: `{meta['model_branch']}`\n\n"
    title += f"Result root: `{meta['result_root']}`\n\n"
    if summary["overall_pass"]:
        title += "No blockers observed in this larger-smoke branch.\n\n"
        if meta["model_branch"] == "local_alt_model":
            title += "- qwen line larger-smoke passed: `yes`\n"
            title += "- worth retaining as local_alt_model evidence: `yes`\n"
        else:
            title += "- llama line larger-smoke passed: `yes`\n"
            title += "- worth retaining as paper_default_model evidence: `yes`\n"
        title += "- can continue to updated readiness review: `yes`\n"
        title += "- can start formal now: `no`\n"
        return title

    title += "Blockers were observed in this larger-smoke branch.\n\n"
    title += f"- manifest_ok: `{summary['manifest_ok']}`\n"
    title += f"- ori_protocol_isolation: `{summary['ori_protocol_isolation']}`\n"
    if summary["soft_blockers"]:
        title += "- soft blockers:\n"
        for item in summary["soft_blockers"]:
            title += f"  - {item}\n"
    for dataset, result in summary["dataset_results"].items():
        if not result["pass"]:
            title += f"- dataset blocker: `{dataset}`\n"
    title += "- can continue to updated readiness review: `yes`\n"
    title += "- can start formal now: `no`\n"
    return title


def write_branch_reports(result_root: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    reports_root = result_root / "reports"
    summary = validation_summary(result_root, meta)
    prefix = meta["report_prefix"]
    json_dump(reports_root / "branch_validation_summary.json", summary)
    (reports_root / f"{prefix}_larger_smoke_run_log.md").write_text(
        branch_run_log_text(result_root, meta, summary),
        encoding="utf-8",
    )
    (reports_root / f"{prefix}_larger_smoke_validation.md").write_text(
        branch_validation_text(result_root, meta, summary),
        encoding="utf-8",
    )
    (reports_root / f"{prefix}_larger_smoke_tables.md").write_text(
        branch_tables_text(result_root, meta),
        encoding="utf-8",
    )
    (reports_root / f"{prefix}_larger_smoke_blockers.md").write_text(
        branch_blockers_text(meta, summary),
        encoding="utf-8",
    )
    return summary


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    result_root = Path(args.result_root).expanduser()
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "packs").mkdir(parents=True, exist_ok=True)
    (result_root / "eval").mkdir(parents=True, exist_ok=True)
    (result_root / "reports").mkdir(parents=True, exist_ok=True)
    (result_root / "logs").mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=result_root / "logs" / "run_paper_repair_larger_smoke_branch.log")
    os.environ["PAPER_REPAIR_DEBUG_LOG"] = str(result_root / "logs" / "runtime_debug.log")

    sample_map = build_sample_map(args)
    meta = branch_metadata(args=args, result_root=result_root, sample_map=sample_map)
    json_dump(result_root / "sample_ids.json", sample_map)
    json_dump(result_root / "branch_metadata.json", meta)

    manifest_path = result_root / "run_manifest.jsonl"
    snapshot_integrity(result_root / "reports" / "_ori_protocol_integrity_before_larger_smoke.json")

    failure: str | None = None
    try:
        for dataset, sample_ids in sample_map.items():
            spec = DATASET_SPECS[dataset]
            for regime in DEFAULT_REGIMES:
                pack_path = result_root / "packs" / f"{dataset}_{regime}_pack.json"
                build_kwargs = dict(spec["builder_kwargs"])
                adapter_kwargs = build_kwargs.pop("adapter_kwargs", None)
                await run_step(
                    manifest_path=manifest_path,
                    payload={
                        "dataset": dataset,
                        "step_type": "build",
                        "regime": regime,
                        "ordering": "n/a",
                        "method_group": "candidate_pack",
                    },
                    coro=build_frozen_candidate_pack(
                        llm_name=args.llm_name,
                        generation_regime=regime,
                        question_indices=sample_ids,
                        dataset_name=dataset,
                        split=spec["split"],
                        adapter_kwargs=adapter_kwargs,
                        output_path=pack_path,
                        **build_kwargs,
                    ),
                )
            for regime in DEFAULT_REGIMES:
                pack_path = result_root / "packs" / f"{dataset}_{regime}_pack.json"
                for ordering in DEFAULT_ORDERINGS:
                    await run_step(
                        manifest_path=manifest_path,
                        payload={
                            "dataset": dataset,
                            "step_type": "judge",
                            "regime": regime,
                            "ordering": ordering,
                            "method_group": "all_methods",
                        },
                        coro=run_judge_benchmark(
                            frozen_pack=pack_path,
                            output_dir=result_root / "eval" / dataset / regime / ordering,
                            methods=DEFAULT_METHODS,
                            llm_name=args.llm_name,
                            judge_shuffle=(ordering == "shuffle"),
                            shuffle_seed=args.shuffle_seed,
                        ),
                    )
    except Exception as exc:  # pragma: no cover - runtime orchestration path
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        snapshot_integrity(result_root / "reports" / "_ori_protocol_integrity_after_larger_smoke.json")
        summary = write_branch_reports(result_root, meta)
        if failure is not None:
            json_dump(result_root / "reports" / "branch_failure.json", {"error": failure, "summary": summary})
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
