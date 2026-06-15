import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging  # noqa: E402
from paper_repair.protocol import (  # noqa: E402
    DEFAULT_INDEX_FILE,
    DEFAULT_METHODS,
    run_full_153_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full paper-style MMLU-153 repair benchmark.")
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, default=str(DEFAULT_INDEX_FILE))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["parallel_exploration", "progressive_refinement"],
    )
    parser.add_argument(
        "--shuffles",
        nargs="+",
        default=["noshuffle", "shuffle"],
        help="Choose from: noshuffle shuffle",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--agent_role", type=str, default="MMLU Solver")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_shuffle_flags(values):
    parsed = []
    for value in values:
        lowered = str(value).strip().lower()
        if lowered == "shuffle":
            parsed.append(True)
        elif lowered == "noshuffle":
            parsed.append(False)
        else:
            raise ValueError("shuffles must be chosen from: noshuffle shuffle")
    return parsed


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_root / "logs" / "run_mmlu_paper_repair_153.log")

    await run_full_153_benchmark(
        llm_name=args.llm_name,
        output_root=output_root,
        index_file=args.index_file,
        methods=args.methods,
        regimes=args.regimes,
        shuffles=_parse_shuffle_flags(args.shuffles),
        split=args.split,
        agent_role=args.agent_role,
        agent_temperature=args.agent_temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    asyncio.run(main())
