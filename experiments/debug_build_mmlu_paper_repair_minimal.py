from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_FILE = str(PROJECT_ROOT / "experiments" / "mmlu_153_seed888_indices.json")


def _emit(message: str, debug_log: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with debug_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal debug harness for paper_repair MMLU builder.")
    parser.add_argument("--mode", choices=["builder", "oldpath"], default="builder")
    parser.add_argument("--llm_name", type=str, default="/pychen/Test/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation_regime", choices=["parallel_exploration", "progressive_refinement"], default="parallel_exploration")
    parser.add_argument("--index_file", type=str, default=str(DEFAULT_INDEX_FILE))
    parser.add_argument("--question-count", type=int, default=1)
    parser.add_argument("--agent-count", type=int, default=1)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--agent-role", type=str, default="MMLU Solver")
    parser.add_argument("--agent-temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--force-bfloat16", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--skip-pack-dump", action="store_true")
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "result" / "paper_repair_builder_debug"))
    return parser.parse_args()


async def _run_oldpath_minimal(*, args: argparse.Namespace, debug_log: Path, out_dir: Path) -> dict[str, Any]:
    _emit("[PACK INIT] before importing oldpath dependencies", debug_log)
    from KVCOMM.agents.analyze_agent import AnalyzeAgent
    from paper_repair.protocol.frozen_pack import load_dataset_records, load_question_indices
    _emit("[PACK INIT] oldpath dependencies imported", debug_log)

    _emit("[ENTRY] oldpath mode selected", debug_log)
    _emit("[DATA INIT] before index file load", debug_log)
    question_indices = load_question_indices(index_file=args.index_file, limit_questions=args.question_count)
    _emit(f"[DATA INIT] selected question ids resolved ids={question_indices}", debug_log)
    _emit("[DATA INIT] before dataset load", debug_log)
    records = load_dataset_records(question_indices, dataset_name="mmlu", split=args.split)
    _emit(f"[DATA INIT] dataset loaded record_count={len(records)}", debug_log)
    record = records[0]

    _emit("[MODEL INIT] before agent init", debug_log)
    agent = AnalyzeAgent(
        id="0",
        role=args.agent_role,
        temperature=args.agent_temperature,
        domain="mmlu",
        llm_name=args.llm_name,
    )
    _emit("[MODEL INIT] tokenizer loaded", debug_log)
    try:
        param = next(agent.llm.model.parameters())
        _emit(f"[MODEL INIT] model loaded dtype={param.dtype} device={param.device}", debug_log)
    except Exception as exc:
        _emit(f"[MODEL INIT] model inspect failed error={type(exc).__name__}", debug_log)

    _emit("[FIRST SAMPLE] before sample 0", debug_log)
    _emit(f"[FIRST SAMPLE] sample 0 prompt built chars={len(record['question_text'])}", debug_log)
    _emit("[FIRST SAMPLE] before agent 0 generation", debug_log)
    generation = await agent._async_execute({"task": record["question_text"]}, {}, {}, mode="default")
    _emit("[FIRST SAMPLE] agent 0 generation returned", debug_log)
    text = str(generation.text or "")
    _emit(f"[FIRST SAMPLE] candidate text extracted chars={len(text)}", debug_log)

    payload = {
        "mode": "oldpath",
        "question_indices": question_indices,
        "record_index": record["record_index"],
        "question_text": record["question_text"],
        "candidate_text": text,
        "candidate_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    out_path = out_dir / "oldpath_result.json"
    _emit("[FIRST SAMPLE] before json dump", debug_log)
    _json_dump(out_path, payload)
    _emit(f"[FIRST SAMPLE] oldpath result written path={out_path}", debug_log)
    return payload


async def _run_builder_minimal(*, args: argparse.Namespace, debug_log: Path, out_dir: Path) -> dict[str, Any]:
    _emit("[ENTRY] builder mode selected", debug_log)
    _emit("[ENTRY] args parsed", debug_log)
    _emit("[IMPORT CHAIN] before importing paper_repair.protocol.frozen_pack", debug_log)
    from paper_repair.protocol.frozen_pack import build_frozen_candidate_pack, load_question_indices
    _emit("[IMPORT CHAIN] imported paper_repair.protocol.frozen_pack", debug_log)
    _emit("[PACK INIT] before frozen_pack/runner build", debug_log)
    _emit("[PACK INIT] before pack builder build", debug_log)
    question_indices = load_question_indices(index_file=args.index_file, limit_questions=args.question_count)
    output_path = out_dir / "minimal_pack.json"
    if args.skip_pack_dump:
        output_path = out_dir / "minimal_pack_skipped.json"
    pack = await build_frozen_candidate_pack(
        llm_name=args.llm_name,
        generation_regime=args.generation_regime,
        question_indices=question_indices,
        dataset_name="mmlu",
        split=args.split,
        agent_name="AnalyzeAgent",
        agent_role=args.agent_role,
        agent_temperature=args.agent_temperature,
        agent_count=args.agent_count,
        seed=args.seed,
        output_path=None if args.skip_pack_dump else output_path,
    )
    _emit("[EXIT] builder returned successfully", debug_log)
    cases = list(pack.get("cases") or [])
    first_case = dict(cases[0] or {}) if cases else {}
    first_case_candidates = list(first_case.get("candidates") or [])
    candidate_texts = [str(item.get("text") or "") for item in first_case_candidates]
    case_summaries = []
    for case in cases:
        case_candidates = list(case.get("candidates") or [])
        case_summaries.append(
            {
                "record_index": int(case.get("record_index")),
                "candidate_count": len(case_candidates),
                "candidate_text_count": len([item for item in case_candidates if str(item.get("text") or "").strip()]),
                "agent_ids": [str(item.get("agent_id") or "") for item in case_candidates],
            }
        )
    payload = {
        "mode": "builder",
        "question_indices": question_indices,
        "agent_count": args.agent_count,
        "generation_regime": args.generation_regime,
        "candidate_count": pack.get("candidate_count"),
        "case_count": len(pack.get("cases") or []),
        "first_case_candidate_count": len(first_case_candidates),
        "first_case_candidate_text_count": len(candidate_texts),
        "first_case_agent_ids": [str(item.get("agent_id") or "") for item in first_case_candidates],
        "all_case_candidate_counts": [int(item["candidate_count"]) for item in case_summaries],
        "all_case_candidate_text_counts": [int(item["candidate_text_count"]) for item in case_summaries],
        "case_summaries": case_summaries,
        "pack_hash": hashlib.sha256(json.dumps(pack, sort_keys=True).encode("utf-8")).hexdigest(),
        "candidate_pack_hash": hashlib.sha256(json.dumps(pack, sort_keys=True).encode("utf-8")).hexdigest(),
        "output_path": None if args.skip_pack_dump else str(output_path),
    }
    _json_dump(out_dir / "builder_result.json", payload)
    return payload


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = f"{args.question_count}x{args.agent_count}"
    out_dir = Path(args.output_root).expanduser() / f"{timestamp}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_log = out_dir / "debug.log"
    configure_logging(log_path=out_dir / "loguru.log")

    if args.force_bfloat16 == "on":
        os.environ["KVCOMM_FORCE_BFLOAT16"] = "1"
    elif args.force_bfloat16 == "off":
        os.environ.pop("KVCOMM_FORCE_BFLOAT16", None)

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PAPER_REPAIR_DEBUG_LOG"] = str(debug_log)

    _emit("[ENTRY] entered main()", debug_log)
    _emit(f"[ENTRY] mode={args.mode} force_bfloat16={args.force_bfloat16}", debug_log)

    if args.mode == "builder":
        result = await _run_builder_minimal(args=args, debug_log=debug_log, out_dir=out_dir)
    else:
        result = await _run_oldpath_minimal(args=args, debug_log=debug_log, out_dir=out_dir)

    _json_dump(out_dir / "result.json", result)
    _emit("[EXIT] clean exit", debug_log)


if __name__ == "__main__":
    asyncio.run(main())
