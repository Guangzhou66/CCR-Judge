from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper_repair formal mainline with an explicit model pin.")
    parser.add_argument("--llm-name", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_integrity(path: Path) -> None:
    json_dump(path, {str(item): sha256(item) for item in INTEGRITY_TARGETS})


def write_manifest_row(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def run_step(manifest_path: Path, payload: Dict[str, Any], coro: Any) -> Any:
    start_utc = utc_now()
    exit_status = 0
    error_message = None
    result = None
    try:
        result = await coro
    except Exception as exc:  # pragma: no cover - long-running orchestration path
        exit_status = 1
        error_message = f"{type(exc).__name__}: {exc}"
    end_utc = utc_now()
    row = dict(payload)
    row.update({"start_utc": start_utc, "end_utc": end_utc, "exit_status": exit_status})
    if error_message:
        row["error_message"] = error_message
    write_manifest_row(manifest_path, row)
    if exit_status != 0:
        raise RuntimeError(error_message or "unknown run failure")
    safe_gc()
    return result


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    result_root = Path(args.result_root).expanduser()
    reports_root = result_root / "reports"
    formal_root = result_root / "formal"
    candidate_cache_root = result_root / "candidate_cache"
    logs_root = result_root / "logs"
    for path in [reports_root, formal_root, candidate_cache_root, logs_root]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["KVCOMM_PROGRESS_ONLY"] = os.environ.get("KVCOMM_PROGRESS_ONLY", "1")
    os.environ["PYTHONWARNINGS"] = os.environ.get("PYTHONWARNINGS", "ignore")
    os.environ["TRANSFORMERS_VERBOSITY"] = os.environ.get("TRANSFORMERS_VERBOSITY", "error")
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

    configure_logging(log_path=logs_root / "run_paper_repair_formal_mainline.log")
    os.environ["PAPER_REPAIR_DEBUG_LOG"] = str(logs_root / "runtime_debug.log")

    sample_ids = {
        "mmlu": load_question_indices(index_file=str(DEFAULT_MMLU_INDEX_FILE), limit_questions=153),
        "gsm8k": list(range(1319)),
        "humaneval": list(range(161)),
    }
    json_dump(result_root / "sample_ids.json", sample_ids)
    json_dump(
        result_root / "formal_branch_metadata.json",
        {
            "model_path": args.llm_name,
            "model_branch": "paper_default_model",
            "result_root": str(result_root),
            "candidate_cache_root": str(candidate_cache_root),
            "formal_root": str(formal_root),
            "reports_root": str(reports_root),
            "shuffle_seed": args.shuffle_seed,
        },
    )
    (reports_root / "paper_repair_formal_model_pinning_note.md").write_text(
        "# Paper Repair Formal Model Pinning Note\n\n"
        f"formal mainline model = `{args.llm_name}`\n\n"
        "Qwen remains a local_alt_model evidence line and is not used as the formal mainline.\n",
        encoding="utf-8",
    )
    (reports_root / "paper_repair_formal_run_log.md").write_text(
        "# Paper Repair Formal Run Log\n\n"
        f"Model path: `{args.llm_name}`\n\n"
        f"Result root: `{result_root}`\n\n"
        f"Launch status: `running`\n",
        encoding="utf-8",
    )

    manifest_path = result_root / "run_manifest.jsonl"
    snapshot_integrity(reports_root / "_ori_protocol_integrity_before_formal.json")

    try:
        for regime in DEFAULT_REGIMES:
            await run_step(
                manifest_path,
                {"dataset": "mmlu", "step_type": "build", "regime": regime, "ordering": "n/a", "method_group": "candidate_pack"},
                build_frozen_candidate_pack(
                    llm_name=args.llm_name,
                    generation_regime=regime,
                    question_indices=sample_ids["mmlu"],
                    dataset_name="mmlu",
                    split="val",
                    agent_name="AnalyzeAgent",
                    agent_role="MMLU Solver",
                    agent_temperature=0.2,
                    agent_count=4,
                    seed=888,
                    output_path=candidate_cache_root / "mmlu" / f"mmlu_{regime}.json",
                ),
            )
        for dataset, split, seed, adapter_kwargs, agent_name, agent_role, count in [
            ("gsm8k", "test", 42, {"dataset_root": DEFAULT_GSM8K_DATASET_ROOT, "config_name": "main"}, "MathSolver", "Math Solver", 1319),
            ("humaneval", "test", 42, {"dataset_json": DEFAULT_HUMANEVAL_JSON}, "CodeWriting", "Programming Expert", 161),
        ]:
            for regime in DEFAULT_REGIMES:
                await run_step(
                    manifest_path,
                    {"dataset": dataset, "step_type": "build", "regime": regime, "ordering": "n/a", "method_group": "candidate_pack"},
                    build_frozen_candidate_pack(
                        llm_name=args.llm_name,
                        generation_regime=regime,
                        question_indices=sample_ids[dataset],
                        dataset_name=dataset,
                        split=split,
                        agent_name=agent_name,
                        agent_role=agent_role,
                        agent_temperature=0.2,
                        agent_count=4,
                        seed=seed,
                        adapter_kwargs=adapter_kwargs,
                        output_path=candidate_cache_root / dataset / f"{dataset}_{regime}.json",
                    ),
                )

        for dataset in ["mmlu", "gsm8k", "humaneval"]:
            for regime in DEFAULT_REGIMES:
                pack_path = candidate_cache_root / dataset / f"{dataset}_{regime}.json"
                for ordering in DEFAULT_ORDERINGS:
                    await run_step(
                        manifest_path,
                        {
                            "dataset": dataset,
                            "step_type": "judge",
                            "regime": regime,
                            "ordering": ordering,
                            "method_group": "all_methods",
                        },
                        run_judge_benchmark(
                            frozen_pack=pack_path,
                            output_dir=formal_root / dataset / regime / ordering,
                            methods=DEFAULT_METHODS,
                            llm_name=args.llm_name,
                            judge_shuffle=(ordering == "shuffle"),
                            shuffle_seed=args.shuffle_seed,
                        ),
                    )
    finally:
        snapshot_integrity(reports_root / "_ori_protocol_integrity_after_formal.json")
        from experiments.postprocess_paper_repair_formal import main as postprocess_main  # noqa: E402

        argv_backup = sys.argv[:]
        sys.argv = [
            "postprocess_paper_repair_formal.py",
            "--result-root",
            str(result_root),
            "--llm-name",
            args.llm_name,
        ]
        try:
            postprocess_main()
        finally:
            sys.argv = argv_backup


if __name__ == "__main__":
    asyncio.run(main())
