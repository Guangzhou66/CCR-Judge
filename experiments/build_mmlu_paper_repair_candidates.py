from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging  # noqa: E402


def _emit(message: str) -> None:
    print(message, flush=True)
    debug_log = os.environ.get("PAPER_REPAIR_DEBUG_LOG", "").strip()
    if debug_log:
        path = Path(debug_log).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_FILE = str(PROJECT_ROOT / "experiments" / "mmlu_153_seed888_indices.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed MMLU paper_repair candidate cache.")
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument(
        "--generation_regime",
        type=str,
        required=True,
        choices=["parallel_exploration", "progressive_refinement"],
    )
    parser.add_argument("--index_file", type=str, default=str(DEFAULT_INDEX_FILE))
    parser.add_argument("--question-indices", nargs="*", type=int, default=None)
    parser.add_argument("--limit_questions", type=int, default=153)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--agent_name", type=str, default="AnalyzeAgent")
    parser.add_argument("--agent_role", type=str, default="MMLU Solver")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--agent_count", type=int, default=4)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=888)
    return parser.parse_args()


async def main() -> None:
    _emit("[ENTRY] entered main()")
    args = parse_args()
    _emit(f"[ENTRY] args parsed {args}")
    os.chdir(PROJECT_ROOT)
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_path.parent / "logs" / "build_mmlu_paper_repair_candidates.log")

    _emit("[PACK INIT] before importing paper_repair.protocol.frozen_pack")
    from paper_repair.protocol.frozen_pack import build_frozen_candidate_pack, load_question_indices

    _emit("[PACK INIT] paper_repair.protocol.frozen_pack imported")
    _emit("[DATA INIT] before index file load")
    question_indices = load_question_indices(
        index_file=args.index_file,
        question_indices=args.question_indices,
        limit_questions=args.limit_questions,
    )
    _emit(f"[DATA INIT] index file loaded count={len(question_indices)}")
    _emit("[PACK INIT] before build_frozen_candidate_pack")
    await build_frozen_candidate_pack(
        llm_name=args.llm_name,
        generation_regime=args.generation_regime,
        question_indices=question_indices,
        dataset_name="mmlu",
        split=args.split,
        agent_name=args.agent_name,
        agent_role=args.agent_role,
        agent_temperature=args.agent_temperature,
        agent_count=args.agent_count,
        seed=args.seed,
        output_path=output_path,
    )
    _emit("[EXIT] clean exit")


if __name__ == "__main__":
    asyncio.run(main())
