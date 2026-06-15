from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging  # noqa: E402
from paper_repair.protocol import DEFAULT_METHODS, run_judge_benchmark  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-candidate paper_repair judge methods.")
    parser.add_argument("--frozen_pack", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--llm_name", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--judge-shuffle", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_dir / "logs" / "evaluate_paper_repair_judge.log")

    await run_judge_benchmark(
        frozen_pack=args.frozen_pack,
        output_dir=output_dir,
        methods=args.methods,
        llm_name=args.llm_name,
        judge_shuffle=args.judge_shuffle,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    asyncio.run(main())
