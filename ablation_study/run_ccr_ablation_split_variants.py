from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = Path(__file__).resolve().parent
UNIT_SCRIPT = ABLATION_ROOT / "run_support_variant_unit.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_repair.eval.summary import json_dump, method_summary, write_benchmark_outputs  # noqa: E402
from paper_repair.protocol.method_specs import METHOD_SPECS  # noqa: E402

from ablation_study.ccr_ablation_variants import AblationVariant, variants_for_layer  # noqa: E402
from ablation_study.run_ccr_ablation import (  # noqa: E402
    DEFAULT_LLM_NAME,
    DEFAULT_RESULT_ROOT,
    PROTOCOL_TYPE,
    SETTING_TYPE,
    _candidate_pack_path,
    _json_load,
    _json_write,
    _pack_hash,
    _parse_seed_list,
    _selected_testbeds,
    _subset_pack,
    generate_reports,
)


def _variant_label(slug: str, variants: Sequence[AblationVariant]) -> str:
    if slug == "dense":
        return "Dense Prefill"
    for variant in variants:
        if variant.slug == slug:
            return variant.label
    return slug


def _summary_spec(method: str, variants: Sequence[AblationVariant]) -> Any:
    if method in METHOD_SPECS:
        return METHOD_SPECS[method]
    from types import SimpleNamespace

    return SimpleNamespace(
        name=method,
        label=_variant_label(method, variants),
        reuse="judge_kv_reuse",
    )


def _pack_metadata(pack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_pack_hash": _pack_hash(pack),
        "candidate_build_mode": pack.get("candidate_build_mode", "dense_prefill_only"),
        "execution_side_reuse": bool(pack.get("execution_side_reuse")),
    }


def _chunk_pack(
    pack: Dict[str, Any],
    *,
    start: int,
    end: int,
    chunk_index: int,
) -> Dict[str, Any]:
    chunk = dict(pack)
    chunk["cases"] = list(pack.get("cases") or [])[start:end]
    question_indices = pack.get("question_indices")
    if isinstance(question_indices, list):
        chunk["question_indices"] = question_indices[start:end]
    chunk["ablation_chunk_index"] = int(chunk_index)
    chunk["ablation_chunk_start"] = int(start)
    chunk["ablation_chunk_end"] = int(end)
    return chunk


def _write_pack_chunks(
    *,
    pack: Dict[str, Any],
    pack_dir: Path,
    setting_name: str,
    chunk_size: int,
) -> tuple[Path, list[Path]]:
    pack_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = pack_dir / f"{setting_name}_fixed_pack.json"
    _json_write(fixed_path, pack)

    cases = list(pack.get("cases") or [])
    if chunk_size <= 0 or chunk_size >= len(cases):
        return fixed_path, [fixed_path]

    chunk_paths: list[Path] = []
    for chunk_index, start in enumerate(range(0, len(cases), chunk_size)):
        end = min(start + chunk_size, len(cases))
        chunk_path = pack_dir / f"{setting_name}_chunk{chunk_index:03d}_{start:05d}_{end:05d}.json"
        _json_write(
            chunk_path,
            _chunk_pack(pack, start=start, end=end, chunk_index=chunk_index),
        )
        chunk_paths.append(chunk_path)
    return fixed_path, chunk_paths


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


def _cooldown(label: str) -> None:
    raw_seconds = str(os.getenv("ABLATION_VARIANT_COOLDOWN_SEC", "0") or "0").strip()
    try:
        seconds = float(raw_seconds)
    except Exception:
        seconds = 0.0
    if seconds <= 0:
        return
    print(f"[variant-cleanup] {label}: child Python exited; cooling GPU for {seconds:g}s.", flush=True)
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


def _load_rows(unit_dir: Path, method: str) -> list[dict[str, Any]]:
    path = unit_dir / f"{method}_details.json"
    if not path.exists():
        return []
    payload = _json_load(path)
    return payload if isinstance(payload, list) else []


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
        rows.extend(_load_rows(unit_dir, method))
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


def run_testbed_split(
    *,
    testbed: Any,
    variants: Sequence[AblationVariant],
    result_root: Path,
    output_root: Path,
    llm_name: str,
    shuffle_seed: int,
    limit_cases: int | None,
    chunk_size: int,
    force: bool,
) -> dict[str, Any]:
    pack_path = _candidate_pack_path(result_root, testbed)
    if not pack_path.exists():
        raise FileNotFoundError(f"Candidate pack not found: {pack_path}")
    frozen_pack = _subset_pack(_json_load(pack_path), limit_cases)
    cases = list(frozen_pack.get("cases") or [])
    if not cases:
        raise ValueError(f"No cases found in candidate pack: {pack_path}")

    block_dir = output_root / testbed.layer / testbed.dataset / testbed.regime / testbed.ordering
    pack_dir = output_root / "candidate_packs" / testbed.name
    fixed_pack_path, chunk_paths = _write_pack_chunks(
        pack=frozen_pack,
        pack_dir=pack_dir,
        setting_name=testbed.name,
        chunk_size=max(0, int(chunk_size)),
    )
    block_dir.mkdir(parents=True, exist_ok=True)

    block_config = {
        "setting_type": SETTING_TYPE,
        "protocol_type": PROTOCOL_TYPE,
        "testbed": {
            "layer": testbed.layer,
            "name": testbed.name,
            "dataset": testbed.dataset,
            "regime": testbed.regime,
            "ordering": testbed.ordering,
            "description": testbed.description,
        },
        "candidate_pack_path": str(pack_path),
        "fixed_candidate_pack_path": str(fixed_pack_path),
        "candidate_pack_hash": _pack_hash(frozen_pack),
        "llm_name": llm_name,
        "shuffle_seed": int(shuffle_seed),
        "limit_cases": limit_cases,
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunk_paths),
        "case_count": len(cases),
        "candidate_count": int(frozen_pack.get("candidate_count") or 0),
        "variants": [asdict(variant) for variant in variants],
    }
    _json_write(block_dir / "ccr_ablation_config.json", block_config)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("KVCOMM_FORCE_BFLOAT16", "1")
    env.setdefault("KVCOMM_PROGRESS_ONLY", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    methods = ["dense"] + [variant.slug for variant in variants]
    method_unit_dirs: dict[str, list[Path]] = {method: [] for method in methods}
    blockers: list[dict[str, Any]] = []

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
                        judge_shuffle=testbed.judge_shuffle,
                        shuffle_seed=shuffle_seed,
                    ),
                    cwd=PROJECT_ROOT,
                    env=env,
                    log_path=chunk_dir / "logs" / "dense.log",
                )
                _cooldown(f"{testbed.name}/chunk{chunk_index:03d}/dense")
            method_unit_dirs["dense"].append(dense_unit_dir)
        except Exception as exc:
            blockers.append(
                {
                    "testbed": testbed.name,
                    "chunk_index": chunk_index,
                    "variant": "dense",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "log_path": str(chunk_dir / "logs" / "dense.log"),
                }
            )
            continue

        dense_details = dense_unit_dir / "dense_details.json"
        for variant in variants:
            method = variant.slug
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
                            judge_shuffle=testbed.judge_shuffle,
                            shuffle_seed=shuffle_seed,
                        ),
                        cwd=PROJECT_ROOT,
                        env=env,
                        log_path=chunk_dir / "logs" / f"{method}.log",
                    )
                    _cooldown(f"{testbed.name}/chunk{chunk_index:03d}/{method}")
                method_unit_dirs[method].append(unit_dir)
            except Exception as exc:
                blockers.append(
                    {
                        "testbed": testbed.name,
                        "chunk_index": chunk_index,
                        "variant": method,
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
            generation_regime=testbed.regime,
            judge_shuffle=testbed.judge_shuffle,
        )
        if summary is not None:
            summary_rows.append(summary)

    benchmark_summary = write_benchmark_outputs(
        output_root=block_dir,
        benchmark_name=f"{testbed.dataset}_paper_repair_ccr_ablation_split_variants",
        model_name=llm_name,
        generation_regime=testbed.regime,
        judge_shuffle=testbed.judge_shuffle,
        shuffle_seed=int(shuffle_seed),
        frozen_pack=frozen_pack,
        pack_metadata=_pack_metadata(frozen_pack),
        summary_rows=summary_rows,
    )
    _json_write(block_dir / "ccr_ablation_blockers.json", blockers)
    return {
        "block_dir": str(block_dir),
        "testbed": block_config["testbed"],
        "candidate_pack_path": str(pack_path),
        "fixed_candidate_pack_path": str(fixed_pack_path),
        "candidate_pack_hash": _pack_hash(frozen_pack),
        "case_count": len(cases),
        "candidate_count": int(frozen_pack.get("candidate_count") or 0),
        "variants": [asdict(variant) for variant in variants],
        "blockers": blockers,
        "benchmark_summary_path": str(block_dir / "benchmark_summary.json"),
        "benchmark_summary": benchmark_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CCR-Judge component ablations with one Python child per variant."
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default=str(ABLATION_ROOT))
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--random-shortlist-seeds", default="42")
    parser.add_argument("--layers", choices=["main", "stability", "all"], default="all")
    parser.add_argument(
        "--testbed",
        action="append",
        help="Custom testbed as dataset:regime:ordering or dataset:regime:ordering:layer.",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--include-optional-payload-statistics-ablation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else ABLATION_ROOT / "results" / f"ccr_ablation_split_variants_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    reports_dir = Path(args.reports_dir).expanduser()
    result_root = Path(args.result_root).expanduser()
    seeds = _parse_seed_list(args.random_shortlist_seeds)
    testbeds = _selected_testbeds(args)

    run_plan = {
        "setting_type": SETTING_TYPE,
        "protocol_type": PROTOCOL_TYPE,
        "execution_isolation": "one_child_python_per_dense_or_variant_unit",
        "result_root": str(result_root),
        "output_root": str(output_root),
        "reports_dir": str(reports_dir),
        "llm_name": args.llm_name,
        "shuffle_seed": int(args.shuffle_seed),
        "random_shortlist_seeds": seeds,
        "limit_cases": args.limit_cases,
        "chunk_size": int(args.chunk_size),
        "include_optional_payload_statistics_ablation": bool(
            args.include_optional_payload_statistics_ablation
        ),
        "testbeds": [asdict(testbed) for testbed in testbeds],
    }
    _json_write(output_root / "ccr_ablation_run_plan.json", run_plan)

    if not args.report_only:
        run_records = []
        for testbed in testbeds:
            variants = variants_for_layer(
                layer=testbed.layer,
                random_seeds=seeds,
                include_optional=bool(args.include_optional_payload_statistics_ablation),
            )
            record = run_testbed_split(
                testbed=testbed,
                variants=variants,
                result_root=result_root,
                output_root=output_root,
                llm_name=str(args.llm_name),
                shuffle_seed=int(args.shuffle_seed),
                limit_cases=args.limit_cases,
                chunk_size=int(args.chunk_size),
                force=bool(args.force),
            )
            run_records.append(record)
        _json_write(output_root / "ccr_ablation_run_records.json", run_records)

    generate_reports(output_root=output_root, reports_dir=reports_dir)
    print(f"CCR split-variant ablation outputs: {output_root}")
    print(f"CCR split-variant ablation reports: {reports_dir}")


if __name__ == "__main__":
    main()
