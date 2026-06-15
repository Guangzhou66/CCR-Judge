from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging  # noqa: E402
from paper_repair.protocol import build_frozen_candidate_pack, load_question_indices  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANEVAL_JSON = str(PROJECT_ROOT / "my_datasets" / "humaneval" / "humaneval-py.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed HumanEval paper_repair candidate cache.")
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument(
        "--generation_regime",
        type=str,
        required=True,
        choices=["parallel_exploration", "progressive_refinement"],
    )
    parser.add_argument("--dataset_json", type=str, default=DEFAULT_HUMANEVAL_JSON)
    parser.add_argument("--question-indices", nargs="*", type=int, default=None)
    parser.add_argument("--limit_questions", type=int, default=161)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--agent_name", type=str, default="CodeWriting")
    parser.add_argument("--agent_role", type=str, default="Programming Expert")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--agent_count", type=int, default=4)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_path.parent / "logs" / "build_humaneval_paper_repair_candidates.log")

    question_indices = load_question_indices(
        question_indices=args.question_indices,
        limit_questions=args.limit_questions,
    )
    await build_frozen_candidate_pack(
        llm_name=args.llm_name,
        generation_regime=args.generation_regime,
        question_indices=question_indices,
        dataset_name="humaneval",
        split=args.split,
        agent_name=args.agent_name,
        agent_role=args.agent_role,
        agent_temperature=args.agent_temperature,
        agent_count=args.agent_count,
        seed=args.seed,
        adapter_kwargs={"dataset_json": args.dataset_json},
        output_path=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
