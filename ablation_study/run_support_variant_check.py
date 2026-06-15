from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = Path(__file__).resolve().parent
UNIT_SCRIPT = ABLATION_ROOT / "run_support_variant_unit.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_repair.eval.summary import json_dump, markdown_table, method_summary, write_benchmark_outputs  # noqa: E402
from paper_repair.protocol.method_specs import METHOD_SPECS  # noqa: E402

from ablation_study.ccr_ablation_variants import (  # noqa: E402
    MAIN_VARIANTS,
    AblationVariant,
    random_shortlist_variant,
)


DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "paper_repair_formal_llama32_3b_20260421_164715"
DEFAULT_LLM_NAME = "/pychen/Test/model/Llama-3.2-3B-Instruct"
SETTING_TYPE = "mainline_controlled"
PROTOCOL_TYPE = "fixed_candidate_judge_only"
SUPPORT_BASELINE_METHODS = ("naive_reuse", "kvcomm", "pal_kv")


@dataclass(frozen=True)
class SupportCheckSetting:
    layer: str
    name: str
    dataset: str
    regime: str
    ordering: str
    default_limit_cases: int | None
    default_chunk_size: int
    role: str
    description: str

    @property
    def judge_shuffle(self) -> bool:
        return self.ordering == "shuffle"


DEFAULT_SETTINGS = (
    SupportCheckSetting(
        layer="main",
        name="main_mmlu_progressive_shuffle",
        dataset="mmlu",
        regime="progressive_refinement",
        ordering="shuffle",
        default_limit_cases=None,
        default_chunk_size=0,
        role="Layer 1 representative testbed",
        description="Primary paper_repair fixed-candidate judge-only setting for the upgrade decision.",
    ),
    SupportCheckSetting(
        layer="stability",
        name="stability_mmlu_parallel_shuffle",
        dataset="mmlu",
        regime="parallel_exploration",
        ordering="shuffle",
        default_limit_cases=None,
        default_chunk_size=0,
        role="Layer 2 same-dataset stability check",
        description="Checks whether the trend transfers from progressive refinement to parallel exploration.",
    ),
    SupportCheckSetting(
        layer="stability",
        name="stability_humaneval_progressive_shuffle",
        dataset="humaneval",
        regime="progressive_refinement",
        ordering="shuffle",
        default_limit_cases=None,
        default_chunk_size=0,
        role="Layer 2 cross-dataset stability check",
        description="HumanEval progressive-refinement transfer check under the same paper_repair controlled protocol.",
    ),
    SupportCheckSetting(
        layer="stability",
        name="stability_humaneval_parallel_shuffle",
        dataset="humaneval",
        regime="parallel_exploration",
        ordering="shuffle",
        default_limit_cases=None,
        default_chunk_size=0,
        role="Layer 2 cross-dataset stability check",
        description="HumanEval parallel-exploration transfer check under the same paper_repair controlled protocol.",
    ),
)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        number = _float(value)
        if number is not None:
            nums.append(number)
    return (sum(nums) / len(nums)) if nums else None


def _rate(flags: Iterable[bool | None]) -> float | None:
    usable = [flag for flag in flags if flag is not None]
    if not usable:
        return None
    return sum(1 for flag in usable if flag) / len(usable)


def _candidate_pack_path(result_root: Path, setting: SupportCheckSetting) -> Path:
    return result_root / "candidate_cache" / setting.dataset / f"{setting.dataset}_{setting.regime}.json"


def _pack_hash(pack: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(pack, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _case_id(case: Dict[str, Any], ordinal: int) -> str:
    for key in ("record_index", "sample_id", "question_id", "task_id", "id"):
        value = case.get(key)
        if value is not None:
            return str(value)
    return str(ordinal)


def _subset_pack(
    pack: Dict[str, Any],
    *,
    limit_cases: int | None,
    setting_name: str,
) -> Dict[str, Any]:
    if limit_cases is None or int(limit_cases) <= 0:
        subset = dict(pack)
        subset["support_variant_setting_name"] = setting_name
        subset["support_variant_limit_cases"] = None
        return subset
    limit = int(limit_cases)
    subset = dict(pack)
    subset["cases"] = list(pack.get("cases") or [])[:limit]
    question_indices = pack.get("question_indices")
    if isinstance(question_indices, list):
        subset["question_indices"] = question_indices[:limit]
    subset["support_variant_setting_name"] = setting_name
    subset["support_variant_limit_cases"] = limit
    return subset


def _chunk_pack(
    pack: Dict[str, Any],
    *,
    start: int,
    end: int,
    chunk_index: int,
    setting_name: str,
) -> Dict[str, Any]:
    chunk = dict(pack)
    chunk["cases"] = list(pack.get("cases") or [])[start:end]
    question_indices = pack.get("question_indices")
    if isinstance(question_indices, list):
        chunk["question_indices"] = question_indices[start:end]
    chunk["support_variant_setting_name"] = setting_name
    chunk["support_variant_chunk_index"] = chunk_index
    chunk["support_variant_chunk_start"] = start
    chunk["support_variant_chunk_end"] = end
    return chunk


def _pack_case_ids(pack: Dict[str, Any]) -> list[str]:
    return [_case_id(case, index) for index, case in enumerate(pack.get("cases") or [])]


def _parse_seed_list(raw: str) -> list[int]:
    seeds = []
    for part in str(raw or "").replace(",", " ").split():
        if part:
            seeds.append(int(part))
    return seeds or [42]


def _base_support_variants(*, include_random: bool, random_seeds: Sequence[int]) -> list[AblationVariant]:
    keep = {"full", "no_support_scoring", "no_shortlist_compression"}
    variants = [variant for variant in MAIN_VARIANTS if variant.family in keep]
    if include_random:
        variants.extend(random_shortlist_variant(seed) for seed in random_seeds)
    return variants


def _variant_label(slug: str, variants: Sequence[AblationVariant]) -> str:
    if slug == "dense":
        return "Dense"
    if slug in METHOD_SPECS:
        return str(METHOD_SPECS[slug].label)
    for variant in variants:
        if variant.slug == slug:
            return variant.label
    return slug


def _summary_spec(method: str, variants: Sequence[AblationVariant]) -> Any:
    if method in METHOD_SPECS:
        return METHOD_SPECS[method]
    return SimpleNamespace(
        name=method,
        label=_variant_label(method, variants),
        reuse="judge_kv_reuse",
    )


def _load_rows(block_dir: Path, method: str) -> list[dict[str, Any]]:
    path = block_dir / f"{method}_details.json"
    if not path.exists():
        return []
    payload = _json_load(path)
    return payload if isinstance(payload, list) else []


def _load_summary(block_dir: Path, method: str) -> dict[str, Any] | None:
    path = block_dir / f"{method}_summary.json"
    if not path.exists():
        return None
    payload = _json_load(path)
    return payload if isinstance(payload, dict) else None


def _shortlist_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("ccr_shortlist_candidate_ids")
    if ids is None:
        payload = row.get("ccr_shortlist_payload") or {}
        if isinstance(payload, dict):
            ids = payload.get("shortlisted_candidate_ids")
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids]


def _extra_metrics(block_dir: Path, method: str, full_rows_by_record: Dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = _load_rows(block_dir, method)
    dense_in_shortlist: list[bool | None] = []
    full_agreement: list[bool | None] = []
    for row in rows:
        shortlist = set(_shortlist_ids(row))
        dense_id = row.get("dense_selected_agent_id")
        dense_in_shortlist.append(None if dense_id is None or not shortlist else str(dense_id) in shortlist)
        full_row = full_rows_by_record.get(int(row.get("record_index", -1)))
        if full_row is not None:
            left = row.get("selected_agent_id")
            right = full_row.get("selected_agent_id")
            full_agreement.append(None if left is None or right is None else str(left) == str(right))

    return {
        "dense_winner_in_shortlist_rate": _rate(dense_in_shortlist),
        "shortlist_answer_group_coverage": _mean(
            row.get("ccr_ablation_shortlist_answer_group_coverage") for row in rows
        ),
        "avg_judge_input_length": _mean(
            (
                row.get("ccr_ablation_judge_context_tokens")
                if row.get("ccr_ablation_judge_context_tokens") is not None
                else row.get("ccr_ablation_judge_context_chars")
            )
            for row in rows
        ),
        "avg_payload_length": _mean(
            (
                row.get("ccr_ablation_context_payload_tokens")
                if row.get("ccr_ablation_context_payload_tokens") is not None
                else row.get("ccr_ablation_context_payload_chars")
            )
            for row in rows
        ),
        "selected_agent_agreement_with_full": (
            1.0 if method == "ccr_full" and rows else _rate(full_agreement)
        ),
    }


def _method_metrics(block_dir: Path, method: str, full_rows_by_record: Dict[int, dict[str, Any]]) -> dict[str, Any]:
    summary = _load_summary(block_dir, method) or {}
    extras = _extra_metrics(block_dir, method, full_rows_by_record) if method != "dense" else {}
    return {
        "method": method,
        "label": summary.get("method_label") or method,
        "total_cases": summary.get("total_cases"),
        "Acc": summary.get("Acc"),
        "JCR": summary.get("JCR"),
        "reuse": summary.get("reuse"),
        "parse_coverage": summary.get("parse_coverage"),
        "fallback_rate": summary.get("fallback_rate"),
        "any_candidate_upper_bound": summary.get("any_candidate_upper_bound"),
        "selection_gap": summary.get("selection_gap"),
        "dense_winner_in_shortlist_rate": extras.get("dense_winner_in_shortlist_rate"),
        "shortlist_answer_group_coverage": extras.get("shortlist_answer_group_coverage"),
        "avg_judge_input_length": extras.get("avg_judge_input_length"),
        "avg_payload_length": extras.get("avg_payload_length"),
        "selected_agent_agreement_with_full": extras.get("selected_agent_agreement_with_full"),
    }


def _setting_metrics(block_dir: Path, variants: Sequence[AblationVariant]) -> dict[str, dict[str, Any]]:
    full_rows = _load_rows(block_dir, "ccr_full")
    full_rows_by_record = {
        int(row["record_index"]): row
        for row in full_rows
        if isinstance(row, dict) and row.get("record_index") is not None
    }
    methods = ["dense", *SUPPORT_BASELINE_METHODS] + [variant.slug for variant in variants]
    return {method: _method_metrics(block_dir, method, full_rows_by_record) for method in methods}


def _run_subprocess(cmd: Sequence[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        print("$ " + " ".join(cmd), flush=True)
        process = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(cmd))


def _cooldown_after_unit(label: str) -> None:
    raw_seconds = str(os.getenv("SUPPORT_METHOD_COOLDOWN_SEC", "0") or "0").strip()
    try:
        seconds = float(raw_seconds)
    except Exception:
        seconds = 0.0
    if seconds <= 0:
        return
    print(f"[method-cleanup] {label}: child Python exited; cooling GPU for {seconds:g}s.", flush=True)
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    time.sleep(seconds)


def _unit_command(
    *,
    pack_path: Path,
    output_dir: Path,
    llm_name: str,
    method: str,
    dense_details: Path | None,
    judge_shuffle: bool,
    shuffle_seed: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(UNIT_SCRIPT),
        "--frozen-pack",
        str(pack_path),
        "--output-dir",
        str(output_dir),
        "--llm-name",
        llm_name,
        "--variant",
        method,
        "--shuffle-seed",
        str(int(shuffle_seed)),
    ]
    if dense_details is not None:
        cmd.extend(["--dense-details", str(dense_details)])
    if judge_shuffle:
        cmd.append("--judge-shuffle")
    return cmd


def _merge_method_outputs(
    *,
    block_dir: Path,
    unit_dirs: Sequence[Path],
    method: str,
    variants: Sequence[AblationVariant],
    generation_regime: str,
    judge_shuffle: bool,
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for unit_dir in unit_dirs:
        details_path = unit_dir / f"{method}_details.json"
        if not details_path.exists():
            continue
        payload = _json_load(details_path)
        if isinstance(payload, list):
            rows.extend(payload)
    if not rows:
        return None
    rows.sort(key=lambda row: int(row.get("record_index", 0)))
    spec = _summary_spec(method, variants)
    summary = method_summary(
        spec=spec,
        rows=rows,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
    )
    json_dump(block_dir / f"{method}_details.json", rows)
    json_dump(block_dir / f"{method}_summary.json", summary)
    return summary


def _selected_settings(args: argparse.Namespace) -> list[SupportCheckSetting]:
    if args.testbed:
        selected: list[SupportCheckSetting] = []
        for index, raw in enumerate(args.testbed):
            parts = raw.split(":")
            if len(parts) not in {3, 4}:
                raise SystemExit("--testbed must use dataset:regime:ordering or dataset:regime:ordering:layer")
            dataset, regime, ordering = parts[:3]
            layer = parts[3] if len(parts) == 4 else "stability"
            selected.append(
                SupportCheckSetting(
                    layer=layer,
                    name=f"custom_{index}_{dataset}_{regime}_{ordering}",
                    dataset=dataset,
                    regime=regime,
                    ordering=ordering,
                    default_limit_cases=args.limit_cases,
                    default_chunk_size=max(0, int(args.chunk_size or 0)),
                    role="User-specified support-variant check",
                    description="User-specified paper_repair controlled setting.",
                )
            )
        return selected

    if args.settings == "main":
        return [DEFAULT_SETTINGS[0]]
    if args.settings == "mmlu":
        return [setting for setting in DEFAULT_SETTINGS if setting.dataset == "mmlu"]
    if args.settings == "humaneval":
        return [setting for setting in DEFAULT_SETTINGS if setting.dataset == "humaneval"]
    if args.settings == "stability":
        return [setting for setting in DEFAULT_SETTINGS if setting.layer == "stability"]
    return list(DEFAULT_SETTINGS)


def _effective_limit(setting: SupportCheckSetting, args: argparse.Namespace) -> int | None:
    if args.limit_cases is not None:
        return int(args.limit_cases) if int(args.limit_cases) > 0 else None
    if setting.name == "main_mmlu_progressive_shuffle" and args.main_limit_cases is not None:
        return int(args.main_limit_cases) if int(args.main_limit_cases) > 0 else None
    if setting.name == "stability_mmlu_parallel_shuffle" and args.mmlu_parallel_limit_cases is not None:
        return int(args.mmlu_parallel_limit_cases) if int(args.mmlu_parallel_limit_cases) > 0 else None
    if setting.dataset == "humaneval" and args.humaneval_limit_cases is not None:
        return int(args.humaneval_limit_cases) if int(args.humaneval_limit_cases) > 0 else None
    return setting.default_limit_cases


def _effective_chunk_size(setting: SupportCheckSetting, args: argparse.Namespace) -> int:
    if args.chunk_size is not None:
        return max(0, int(args.chunk_size))
    if setting.dataset == "humaneval" and args.humaneval_chunk_size is not None:
        return max(0, int(args.humaneval_chunk_size))
    return max(0, int(setting.default_chunk_size))


def _prepare_setting_packs(
    *,
    original_pack_path: Path,
    setting: SupportCheckSetting,
    output_root: Path,
    limit_cases: int | None,
    chunk_size: int,
) -> tuple[Dict[str, Any], Path, list[Path]]:
    original_pack = _json_load(original_pack_path)
    subset = _subset_pack(original_pack, limit_cases=limit_cases, setting_name=setting.name)
    pack_dir = output_root / "candidate_packs" / setting.name
    pack_dir.mkdir(parents=True, exist_ok=True)
    subset_path = pack_dir / f"{setting.name}_fixed_pack.json"
    _json_write(subset_path, subset)

    cases = list(subset.get("cases") or [])
    if not cases:
        raise ValueError(f"No cases available after subsetting: {original_pack_path}")
    if chunk_size <= 0 or chunk_size >= len(cases):
        return subset, subset_path, [subset_path]

    chunk_paths: list[Path] = []
    for chunk_index, start in enumerate(range(0, len(cases), chunk_size)):
        end = min(start + chunk_size, len(cases))
        chunk = _chunk_pack(
            subset,
            start=start,
            end=end,
            chunk_index=chunk_index,
            setting_name=setting.name,
        )
        chunk_path = pack_dir / f"{setting.name}_chunk{chunk_index:03d}_{start:05d}_{end:05d}.json"
        _json_write(chunk_path, chunk)
        chunk_paths.append(chunk_path)
    return subset, subset_path, chunk_paths


def run_setting(
    *,
    setting: SupportCheckSetting,
    variants: Sequence[AblationVariant],
    result_root: Path,
    output_root: Path,
    llm_name: str,
    shuffle_seed: int,
    limit_cases: int | None,
    chunk_size: int,
    force: bool,
) -> dict[str, Any]:
    original_pack_path = _candidate_pack_path(result_root, setting)
    if not original_pack_path.exists():
        raise FileNotFoundError(f"Candidate pack not found: {original_pack_path}")

    subset_pack, subset_pack_path, chunk_paths = _prepare_setting_packs(
        original_pack_path=original_pack_path,
        setting=setting,
        output_root=output_root,
        limit_cases=limit_cases,
        chunk_size=chunk_size,
    )
    block_dir = output_root / setting.layer / setting.dataset / setting.regime / setting.ordering
    block_dir.mkdir(parents=True, exist_ok=True)
    case_ids = _pack_case_ids(subset_pack)
    block_config = {
        "setting_type": SETTING_TYPE,
        "protocol_type": PROTOCOL_TYPE,
        "setting": asdict(setting),
        "candidate_pack_path": str(original_pack_path),
        "fixed_candidate_pack_path": str(subset_pack_path),
        "candidate_pack_hash": _pack_hash(subset_pack),
        "llm_name": llm_name,
        "shuffle_seed": int(shuffle_seed),
        "limit_cases": limit_cases,
        "chunk_size": chunk_size,
        "chunk_count": len(chunk_paths),
        "case_count": len(case_ids),
        "candidate_count": int(subset_pack.get("candidate_count") or 0),
        "sample_ids_path": str(block_dir / "sample_ids.json"),
        "sample_ids_preview": case_ids[:20],
        "variants": [asdict(variant) for variant in variants],
    }
    _json_write(block_dir / "ccr_support_variant_config.json", block_config)
    _json_write(block_dir / "sample_ids.json", case_ids)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("KVCOMM_FORCE_BFLOAT16", "1")
    env.setdefault("KVCOMM_PROGRESS_ONLY", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    methods = ["dense", *SUPPORT_BASELINE_METHODS] + [variant.slug for variant in variants]
    blockers: list[dict[str, Any]] = []
    method_unit_dirs: dict[str, list[Path]] = {method: [] for method in methods}

    for chunk_index, chunk_path in enumerate(chunk_paths):
        chunk_dir = block_dir / "chunks" / f"chunk{chunk_index:03d}"
        dense_unit_dir = chunk_dir / "dense"
        dense_summary_path = dense_unit_dir / "dense_summary.json"
        try:
            if force or not dense_summary_path.exists():
                _run_subprocess(
                    _unit_command(
                        pack_path=chunk_path,
                        output_dir=dense_unit_dir,
                        llm_name=llm_name,
                        method="dense",
                        dense_details=None,
                        judge_shuffle=setting.judge_shuffle,
                        shuffle_seed=shuffle_seed,
                    ),
                    cwd=PROJECT_ROOT,
                    env=env,
                    log_path=chunk_dir / "logs" / "dense.log",
                )
                _cooldown_after_unit(f"{setting.name}/chunk{chunk_index:03d}/dense")
            method_unit_dirs["dense"].append(dense_unit_dir)
        except Exception as exc:
            blockers.append(
                {
                    "setting": setting.name,
                    "chunk_index": chunk_index,
                    "variant": "dense",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "log_path": str(chunk_dir / "logs" / "dense.log"),
                }
            )
            continue

        dense_details = dense_unit_dir / "dense_details.json"
        for method in methods:
            if method == "dense":
                continue
            unit_dir = chunk_dir / method
            summary_path = unit_dir / f"{method}_summary.json"
            try:
                if force or not summary_path.exists():
                    _run_subprocess(
                        _unit_command(
                            pack_path=chunk_path,
                            output_dir=unit_dir,
                            llm_name=llm_name,
                            method=method,
                            dense_details=dense_details,
                            judge_shuffle=setting.judge_shuffle,
                            shuffle_seed=shuffle_seed,
                        ),
                        cwd=PROJECT_ROOT,
                        env=env,
                        log_path=chunk_dir / "logs" / f"{method}.log",
                    )
                    _cooldown_after_unit(f"{setting.name}/chunk{chunk_index:03d}/{method}")
                method_unit_dirs[method].append(unit_dir)
            except Exception as exc:
                blockers.append(
                    {
                        "setting": setting.name,
                        "chunk_index": chunk_index,
                        "variant": method,
                        "label": _variant_label(method, variants),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "log_path": str(chunk_dir / "logs" / f"{method}.log"),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        summary = _merge_method_outputs(
            block_dir=block_dir,
            unit_dirs=method_unit_dirs[method],
            method=method,
            variants=variants,
            generation_regime=setting.regime,
            judge_shuffle=setting.judge_shuffle,
        )
        if summary is not None:
            summary_rows.append(summary)

    benchmark_summary = write_benchmark_outputs(
        output_root=block_dir,
        benchmark_name=f"{setting.dataset}_paper_repair_support_variant_check",
        model_name=llm_name,
        generation_regime=setting.regime,
        judge_shuffle=setting.judge_shuffle,
        shuffle_seed=shuffle_seed,
        frozen_pack=subset_pack,
        pack_metadata={
            "candidate_pack_hash": _pack_hash(subset_pack),
            "candidate_build_mode": subset_pack.get("candidate_build_mode", "dense_prefill_only"),
        },
        summary_rows=summary_rows,
    )
    _json_write(block_dir / "ccr_support_variant_blockers.json", blockers)
    _json_write(block_dir / "ccr_support_variant_metrics.json", _setting_metrics(block_dir, variants))
    return {
        "block_dir": str(block_dir),
        "setting": asdict(setting),
        "candidate_pack_path": str(original_pack_path),
        "fixed_candidate_pack_path": str(subset_pack_path),
        "candidate_pack_hash": _pack_hash(subset_pack),
        "case_count": len(case_ids),
        "candidate_count": int(subset_pack.get("candidate_count") or 0),
        "sample_ids_path": str(block_dir / "sample_ids.json"),
        "sample_ids_preview": case_ids[:20],
        "variants": [asdict(variant) for variant in variants],
        "blockers": blockers,
        "benchmark_summary_path": str(block_dir / "benchmark_summary.json"),
        "benchmark_summary": benchmark_summary,
        "metrics": _setting_metrics(block_dir, variants),
    }


def _collect_configs(output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    configs = []
    for config_path in sorted(output_root.glob("*/*/*/*/ccr_support_variant_config.json")):
        payload = _json_load(config_path)
        if isinstance(payload, dict):
            configs.append((config_path, payload))
    return configs


def _table_row(setting_label: str, method: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "Setting": setting_label,
        "Variant": metrics.get("label") or method,
        "Cases": metrics.get("total_cases"),
        "Acc": _format_float(metrics.get("Acc")),
        "JCR": _format_float(metrics.get("JCR")),
        "Reuse": _format_float(metrics.get("reuse")),
        "Dense winner in shortlist rate": _format_float(metrics.get("dense_winner_in_shortlist_rate")),
        "Any-candidate upper bound": _format_float(metrics.get("any_candidate_upper_bound")),
        "Selection gap": _format_float(metrics.get("selection_gap")),
        "Avg judge input length": _format_float(metrics.get("avg_judge_input_length")),
        "Avg payload length": _format_float(metrics.get("avg_payload_length")),
        "Agreement with Full": _format_float(metrics.get("selected_agent_agreement_with_full")),
    }


def _delta(metrics: dict[str, dict[str, Any]], variant: str, metric: str) -> float | None:
    full = _float((metrics.get("ccr_full") or {}).get(metric))
    other = _float((metrics.get(variant) or {}).get(metric))
    if full is None or other is None:
        return None
    return other - full


def _support_vs_full_flags(metrics: dict[str, dict[str, Any]], *, min_jcr_delta: float) -> dict[str, bool | None]:
    full = metrics.get("ccr_full") or {}
    support = metrics.get("ccr_wo_support_scoring") or {}
    acc_full = _float(full.get("Acc"))
    acc_support = _float(support.get("Acc"))
    jcr_full = _float(full.get("JCR"))
    jcr_support = _float(support.get("JCR"))
    dense_full = _float(full.get("dense_winner_in_shortlist_rate"))
    dense_support = _float(support.get("dense_winner_in_shortlist_rate"))
    gap_full = _float(full.get("selection_gap"))
    gap_support = _float(support.get("selection_gap"))
    return {
        "jcr_improves": None if jcr_full is None or jcr_support is None else jcr_support >= jcr_full + min_jcr_delta,
        "acc_not_lower": None if acc_full is None or acc_support is None else acc_support >= acc_full,
        "dense_retention_not_lower": (
            None if dense_full is None or dense_support is None else dense_support >= dense_full
        ),
        "selection_gap_not_worse": None if gap_full is None or gap_support is None else gap_support <= gap_full,
        "no_acc_jcr_reversal": (
            None
            if None in (acc_full, acc_support, jcr_full, jcr_support)
            else not (acc_support < acc_full and jcr_support < jcr_full)
        ),
    }


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "not available"
    return "yes" if value else "no"


def _decision(
    *,
    configs: Sequence[tuple[Path, dict[str, Any]]],
    min_jcr_delta: float,
    include_random: bool,
) -> dict[str, Any]:
    setting_results = []
    for config_path, config in configs:
        block_dir = config_path.parent
        metrics_path = block_dir / "ccr_support_variant_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _json_load(metrics_path)
        flags = _support_vs_full_flags(metrics, min_jcr_delta=min_jcr_delta)
        setting = config.get("setting") or {}
        is_main = setting.get("layer") == "main"
        random_ok = None
        random_methods = [method for method in metrics if method.startswith("ccr_random_shortlist_seed")]
        if include_random and random_methods:
            support = metrics.get("ccr_wo_support_scoring") or {}
            random_better_flags = []
            for method in random_methods:
                random_metrics = metrics.get(method) or {}
                support_acc = _float(support.get("Acc"))
                support_jcr = _float(support.get("JCR"))
                random_acc = _float(random_metrics.get("Acc"))
                random_jcr = _float(random_metrics.get("JCR"))
                if None not in (support_acc, support_jcr, random_acc, random_jcr):
                    random_better_flags.append(support_acc >= random_acc and support_jcr >= random_jcr)
            random_ok = all(random_better_flags) if random_better_flags else None
        setting_results.append(
            {
                "setting_name": setting.get("name"),
                "layer": setting.get("layer"),
                "flags": flags,
                "random_ok": random_ok,
                "metrics": metrics,
            }
        )

    main_results = [item for item in setting_results if item["layer"] == "main"]
    extra_results = [item for item in setting_results if item["layer"] != "main"]
    main = main_results[0] if main_results else None
    main_flags = (main or {}).get("flags") or {}
    main_pass_count = sum(
        1
        for key in ("jcr_improves", "acc_not_lower", "dense_retention_not_lower", "selection_gap_not_worse")
        if main_flags.get(key) is True
    )
    main_available_count = sum(
        1
        for key in ("jcr_improves", "acc_not_lower", "dense_retention_not_lower", "selection_gap_not_worse")
        if main_flags.get(key) is not None
    )
    extras_no_reversal = [
        item for item in extra_results
        if (item.get("flags") or {}).get("no_acc_jcr_reversal") is True
    ]
    extras_reversal = [
        item for item in extra_results
        if (item.get("flags") or {}).get("no_acc_jcr_reversal") is False
    ]
    random_checks = [
        item.get("random_ok") for item in setting_results
        if item.get("random_ok") is not None
    ]
    random_pass = all(random_checks) if random_checks else (not include_random)

    main_passes = main_pass_count >= 3 and main_available_count >= 4
    extras_pass = len(extra_results) >= 2 and len(extras_no_reversal) >= 2 and not extras_reversal
    if main_passes and extras_pass and random_pass:
        verdict = "upgrade w/o Support to revised main variant"
    elif main_passes or len(extras_no_reversal) > 0:
        verdict = "w/o Support is promising but needs more evidence"
    else:
        verdict = "retain Full as main variant, keep w/o Support as diagnostic ablation"

    return {
        "verdict": verdict,
        "main_pass_count": main_pass_count,
        "main_available_count": main_available_count,
        "extras_no_reversal_count": len(extras_no_reversal),
        "extra_setting_count": len(extra_results),
        "extras_reversal_count": len(extras_reversal),
        "random_pass": random_pass,
        "setting_results": setting_results,
        "min_jcr_delta": min_jcr_delta,
    }


def generate_reports(
    *,
    output_root: Path,
    reports_dir: Path,
    include_random: bool,
    min_jcr_delta: float,
) -> None:
    configs = _collect_configs(output_root)
    manifest_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for config_path, config in configs:
        block_dir = config_path.parent
        setting = config.get("setting") or {}
        setting_label = f"{setting.get('dataset')}/{setting.get('regime')}/{setting.get('ordering')}"
        block_blockers_path = block_dir / "ccr_support_variant_blockers.json"
        if block_blockers_path.exists():
            block_blockers = _json_load(block_blockers_path)
            if isinstance(block_blockers, list):
                blockers.extend(block_blockers)

        manifest_rows.append(
            {
                "Layer": setting.get("layer"),
                "Setting": setting_label,
                "Role": setting.get("role"),
                "Cases": config.get("case_count"),
                "Branches": config.get("candidate_count"),
                "Chunks": config.get("chunk_count"),
                "Fixed candidate pack": config.get("fixed_candidate_pack_path"),
                "Sample ids": config.get("sample_ids_path"),
                "Result root": str(block_dir),
            }
        )
        metrics_path = block_dir / "ccr_support_variant_metrics.json"
        if not metrics_path.exists():
            continue
        metrics_by_method = _json_load(metrics_path)
        for method in [
            "dense",
            *SUPPORT_BASELINE_METHODS,
            "ccr_full",
            "ccr_wo_support_scoring",
            "ccr_wo_shortlist_compression",
        ]:
            if method in metrics_by_method:
                table_rows.append(_table_row(setting_label, method, metrics_by_method[method]))
        for method in sorted(metrics_by_method):
            if method.startswith("ccr_random_shortlist_seed"):
                table_rows.append(_table_row(setting_label, method, metrics_by_method[method]))

    decision = _decision(
        configs=configs,
        min_jcr_delta=min_jcr_delta,
        include_random=include_random,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        "# CCR Support-Variant Check Run Manifest",
        "",
        f"- Setting type: `{SETTING_TYPE}`",
        f"- Protocol type: `{PROTOCOL_TYPE}`",
        "- Main comparison: `CCR-Judge (Full)` vs `CCR-Judge w/o Support Scoring` vs `CCR-Judge w/o Shortlist Compression`.",
        "- Baseline comparison included on the same fixed-candidate judge-only surface: `Naive Reuse`, `KVCOMM`, and `PAL-KV`.",
        "- Candidate packs, scorer, model, judge temperature, and parsing surface are fixed within each setting.",
        f"- Result root: `{output_root}`",
        "",
        markdown_table(
            [
                ("Layer", "Layer"),
                ("Setting", "Setting"),
                ("Role", "Role"),
                ("Cases", "Cases"),
                ("Branches", "Branches"),
                ("Chunks", "Chunks"),
                ("Fixed candidate pack", "Fixed candidate pack"),
                ("Sample ids", "Sample ids"),
                ("Result root", "Result root"),
            ],
            manifest_rows,
        ),
        "",
        "## Decision Rule",
        "",
        (
            "Upgrade is recommended only if the main testbed satisfies at least three of four criteria "
            f"(JCR improves by at least `{min_jcr_delta:.4f}`, Acc is not lower, dense-winner retention is not lower, "
            "selection gap is not worse), and at least two additional paper_repair settings show no simultaneous "
            "Acc/JCR reversal. If random shortlist is included, w/o Support must not underperform it on both Acc and JCR."
        ),
    ]
    (reports_dir / "ccr_support_variant_check_run_manifest.md").write_text(
        "\n".join(manifest).rstrip() + "\n",
        encoding="utf-8",
    )

    tables = [
        "# CCR Support-Variant Check Tables",
        "",
        markdown_table(
            [
                ("Setting", "Setting"),
                ("Variant", "Variant"),
                ("Cases", "Cases"),
                ("Acc", "Acc"),
                ("JCR", "JCR"),
                ("Reuse", "Reuse"),
                ("Dense winner in shortlist rate", "Dense winner in shortlist rate"),
                ("Any-candidate upper bound", "Any-candidate upper bound"),
                ("Selection gap", "Selection gap"),
                ("Avg judge input length", "Avg judge input length"),
                ("Avg payload length", "Avg payload length"),
                ("Agreement with Full", "Agreement with Full"),
            ],
            table_rows,
        ),
    ]
    (reports_dir / "ccr_support_variant_tables.md").write_text(
        "\n".join(tables).rstrip() + "\n",
        encoding="utf-8",
    )

    analysis_lines = [
        "# CCR Support-Variant Check Analysis",
        "",
        "This analysis treats `paper_repair` as the primary fixed-candidate judge-only benchmark. The goal is not to retune CCR-Judge, but to determine whether removing support scoring is a stable improvement over the current Full variant.",
        "",
        "## Per-Setting Evidence",
        "",
    ]
    for item in decision["setting_results"]:
        flags = item["flags"]
        metrics = item["metrics"]
        setting_name = item["setting_name"]
        acc_delta = _delta(metrics, "ccr_wo_support_scoring", "Acc")
        jcr_delta = _delta(metrics, "ccr_wo_support_scoring", "JCR")
        gap_delta = _delta(metrics, "ccr_wo_support_scoring", "selection_gap")
        dense_delta = _delta(metrics, "ccr_wo_support_scoring", "dense_winner_in_shortlist_rate")
        shortlist_acc_delta = _delta(metrics, "ccr_wo_shortlist_compression", "Acc")
        shortlist_jcr_delta = _delta(metrics, "ccr_wo_shortlist_compression", "JCR")
        analysis_lines.extend(
            [
                f"### {setting_name}",
                "",
                (
                    f"Removing support scoring changes Acc by `{(acc_delta if acc_delta is not None else 0.0):+.4f}` "
                    f"and JCR by `{(jcr_delta if jcr_delta is not None else 0.0):+.4f}` relative to Full. "
                    f"The selection-gap delta is `{(gap_delta if gap_delta is not None else 0.0):+.4f}`, "
                    f"and the dense-winner retention delta is `{(dense_delta if dense_delta is not None else 0.0):+.4f}`."
                ),
                (
                    f"The main decision flags are: JCR improves = {_bool_text(flags.get('jcr_improves'))}; "
                    f"Acc not lower = {_bool_text(flags.get('acc_not_lower'))}; "
                    f"dense-winner retention not lower = {_bool_text(flags.get('dense_retention_not_lower'))}; "
                    f"selection gap not worse = {_bool_text(flags.get('selection_gap_not_worse'))}."
                ),
                (
                    f"Removing shortlist compression changes Acc by `{(shortlist_acc_delta if shortlist_acc_delta is not None else 0.0):+.4f}` "
                    f"and JCR by `{(shortlist_jcr_delta if shortlist_jcr_delta is not None else 0.0):+.4f}` relative to Full. "
                    "This separates the effect of support-based ranking from the effect of candidate compression."
                ),
                "",
            ]
        )
    analysis_lines.extend(
        [
            "## Research Questions",
            "",
            (
                "Q1. Stability over Full is accepted only when w/o Support improves or preserves the relevant metrics "
                "across the representative testbed and does not show a simultaneous Acc/JCR reversal in at least two additional settings."
            ),
            (
                "Q2. If w/o Support improves while w/o Shortlist behaves similarly, the evidence indicates that the current Full variant's "
                "support-based ranking is a more plausible bottleneck than shortlist compression itself. If w/o Shortlist is stronger, "
                "then the shortlist budget or compression boundary remains an independent risk."
            ),
            (
                "Q3. Higher JCR and higher dense-winner-in-shortlist rate indicate that the improvement is closer to Dense-aligned retention "
                "rather than arbitrary decision drift. Low selected-agent agreement with Full is expected when the variant changes the decision path, "
                "but it must be accompanied by non-worse Acc/JCR to be interpretable as an improvement."
            ),
            (
                "Q4. The upgrade recommendation follows the pre-declared decision rule in the manifest, not a single-setting win."
            ),
        ]
    )
    (reports_dir / "ccr_support_variant_analysis.md").write_text(
        "\n".join(analysis_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    recommendation_lines = [
        "# CCR Support-Variant Upgrade Recommendation",
        "",
        f"Verdict: `{decision['verdict']}`",
        "",
        "## Basis",
        "",
        f"- Main-testbed satisfied criteria: `{decision['main_pass_count']}/{decision['main_available_count']}`.",
        f"- Additional settings with no simultaneous Acc/JCR reversal: `{decision['extras_no_reversal_count']}/{decision['extra_setting_count']}`.",
        f"- Additional settings with simultaneous Acc/JCR reversal: `{decision['extras_reversal_count']}`.",
        f"- Random-shortlist requirement passed: `{_bool_text(decision['random_pass'])}`.",
        "",
        "The recommendation is intentionally conservative. A single representative setting is insufficient to promote w/o Support to the revised main variant; the variant must also avoid reversal in secondary paper_repair settings under the same fixed-candidate protocol.",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        recommendation_lines.append(
            markdown_table(
                [
                    ("setting", "Setting"),
                    ("chunk_index", "Chunk"),
                    ("variant", "Variant"),
                    ("error_type", "Error type"),
                    ("error_message", "Error message"),
                    ("log_path", "Log"),
                ],
                blockers,
            )
        )
    else:
        recommendation_lines.append("No run blockers were recorded.")
    (reports_dir / "ccr_support_variant_upgrade_recommendation.md").write_text(
        "\n".join(recommendation_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    _json_write(
        reports_dir / "ccr_support_variant_decision.json",
        {
            **decision,
            "result_root": str(output_root),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper_repair support-variant validation for CCR-Judge.")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default=str(ABLATION_ROOT))
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--settings", choices=["main", "mmlu", "humaneval", "stability", "all"], default="all")
    parser.add_argument("--testbed", action="append", default=None)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--random-seeds", default="42")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--main-limit-cases", type=int, default=None)
    parser.add_argument("--mmlu-parallel-limit-cases", type=int, default=None)
    parser.add_argument("--humaneval-limit-cases", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--humaneval-chunk-size", type=int, default=None)
    parser.add_argument("--min-jcr-delta", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only execution. By default the runner fails fast when CUDA is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    result_root = Path(args.result_root).expanduser().resolve()
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = ABLATION_ROOT / "results" / f"ccr_support_variant_check_{timestamp}"
    reports_dir = Path(args.reports_dir).expanduser().resolve()

    if not args.reports_only and not args.allow_cpu:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if not cuda_available:
            raise SystemExit(
                "CUDA is not available in this process. Refusing to run the 3B judge on CPU. "
                "Run from a CUDA-visible shell, or pass --allow-cpu for a deliberately slow CPU smoke."
            )

    variants = _base_support_variants(
        include_random=bool(args.include_random),
        random_seeds=_parse_seed_list(args.random_seeds),
    )
    if not args.reports_only:
        for setting in _selected_settings(args):
            limit_cases = _effective_limit(setting, args)
            chunk_size = _effective_chunk_size(setting, args)
            run_setting(
                setting=setting,
                variants=variants,
                result_root=result_root,
                output_root=output_root,
                llm_name=str(args.llm_name),
                shuffle_seed=int(args.shuffle_seed),
                limit_cases=limit_cases,
                chunk_size=chunk_size,
                force=bool(args.force),
            )

    generate_reports(
        output_root=output_root,
        reports_dir=reports_dir,
        include_random=bool(args.include_random),
        min_jcr_delta=float(args.min_jcr_delta),
    )
    print(f"Support-variant check outputs: {output_root}")
    print(f"Support-variant check reports: {reports_dir}")


if __name__ == "__main__":
    main()
