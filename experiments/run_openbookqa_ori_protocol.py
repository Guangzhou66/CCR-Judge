from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from dataset_adapters import create_dataset_adapter
from experiments.run_mmlu_ori_protocol import run_dataset_ori_protocol_once

# OpenBookQA reuses the MMLU-style A/B/C/D prompt and choice-selection path.
import KVCOMM.prompt.mmlu_prompt_set  # noqa: F401,E402
import KVCOMM.agents.analyze_agent  # noqa: F401,E402
import KVCOMM.agents.final_decision  # noqa: F401,E402
import KVCOMM.agents.ori_protocol_final_decision  # noqa: F401,E402
import KVCOMM.agents.ori_protocol_repair_final_decision  # noqa: F401,E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ori-style online OpenBookQA protocol.")
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=str(PROJECT_ROOT / "data" / "openbookqa"))
    parser.add_argument("--config_name", type=str, default="main", choices=["main", "additional"])
    parser.add_argument(
        "--include-fact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include fact1 in the prompt. Defaults to true for config_name=additional and false for main.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="FullConnected", choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--domain", type=str, default="openbookqa")
    parser.add_argument("--decision_method", type=str, default="FinalSelectBest")
    parser.add_argument("--execution_mode", type=str, default="default", choices=["default", "allow_kv_reuse"])
    parser.add_argument("--judge-compare-dense", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-mask-prev-agent-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-group-agents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-group-agents-by-position", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--agent-role", type=str, default="MMLU Solver")
    parser.add_argument(
        "--execution-temperature",
        type=float,
        default=None,
        help="Shared execution temperature. Expands to all execution agents unless --agent-temperatures is set.",
    )
    parser.add_argument("--agent-temperatures", nargs="+", type=float, default=[0.2])
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=1.0,
        help="Explicit default temperature for judge/final-decision generations.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Explicit generation cap for execution and judge generations.",
    )
    parser.add_argument(
        "--stop-conditions",
        type=str,
        default="model_eos_only",
        choices=["model_eos_only"],
        help="Explicit stop condition mode. model_eos_only preserves current/Ori generation semantics.",
    )
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[4])
    parser.add_argument("--prefix", type=str, default="The task is:\n\n")
    parser.add_argument("--limit_questions", type=int, default=10)
    parser.add_argument("--question-indices", nargs="*", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument(
        "--scorer-mode",
        type=str,
        default=None,
        help="Explicit official scorer mode label. Must match the dataset adapter's official scorer.",
    )
    parser.add_argument("--kv-threshold", type=float, default=None)
    parser.add_argument("--kv-max-anchor-num", type=int, default=20)
    parser.add_argument("--kv-window-size", type=int, default=None)
    parser.add_argument("--kv-thread-workers", type=int, default=None)
    parser.add_argument("--kv-worker-timeout", type=float, default=None)
    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    adapter = create_dataset_adapter(
        "openbookqa",
        dataset_root=args.dataset_root,
        config_name=args.config_name,
        include_fact=args.include_fact,
    )
    await run_dataset_ori_protocol_once(
        args,
        dataset_adapter=adapter,
        dataset_split=args.split,
        benchmark_name="openbookqa_ori_protocol",
    )


if __name__ == "__main__":
    asyncio.run(main())
