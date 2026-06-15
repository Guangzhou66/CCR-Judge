from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KVCOMM.utils.log import configure_logging
from experiments.run_mmlu_ori_protocol import (
    ORI_PROTOCOL_MAIN_TABLE_COLUMNS,
    _markdown_table,
    run_ori_protocol_once,
)
from paper_repair.protocol import DEFAULT_INDEX_FILE, load_question_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 153-question Ori-style online MMLU protocol.")
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, default=str(DEFAULT_INDEX_FILE))
    parser.add_argument("--question_indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument(
        "--decision_methods",
        nargs="+",
        default=["OriProtocolFinalSelectBest", "OriProtocolCCRJudgeFinalSelect"],
    )
    parser.add_argument("--execution_modes", nargs="+", default=["default", "allow_kv_reuse"])
    parser.add_argument("--judge_compare_dense", action="store_true")
    parser.add_argument("--judge_shuffles", nargs="+", default=["noshuffle", "shuffle"])
    parser.add_argument("--agent_role", type=str, default="MMLU Solver")
    parser.add_argument("--agent_temperatures", nargs="+", type=float, default=[0.2])
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[4])
    parser.add_argument("--prefix", type=str, default="The task is:\n\n")
    parser.add_argument("--limit_questions", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--kv-threshold", type=float, default=None)
    parser.add_argument("--kv-max-anchor-num", type=int, default=20)
    parser.add_argument("--kv-window-size", type=int, default=None)
    parser.add_argument("--kv-thread-workers", type=int, default=None)
    parser.add_argument("--kv-worker-timeout", type=float, default=None)
    parser.add_argument("--judge-group-agents", action="store_true")
    parser.add_argument("--judge-group-agents-by-position", action="store_true")
    parser.add_argument("--judge-mask-prev-agent-attn", action="store_true")
    return parser.parse_args()


def _parse_shuffle_flags(values: List[str]) -> List[bool]:
    flags: List[bool] = []
    for value in values:
        lowered = str(value).strip().lower()
        if lowered == "shuffle":
            flags.append(True)
        elif lowered == "noshuffle":
            flags.append(False)
        else:
            raise ValueError("judge_shuffles must be chosen from: noshuffle shuffle")
    return flags


def _build_single_run_args(base: argparse.Namespace, *, execution_mode: str, judge_shuffle: bool) -> SimpleNamespace:
    compare_dense = bool(base.judge_compare_dense) if execution_mode == "allow_kv_reuse" else False
    compare_label = "compare_dense" if compare_dense else "plain"
    shuffle_label = "shuffle" if judge_shuffle else "noshuffle"
    return SimpleNamespace(
        mode=base.mode,
        batch_size=base.batch_size,
        llm_name=base.llm_name,
        domain=base.domain,
        decision_method=None,
        execution_mode=execution_mode,
        judge_compare_dense=compare_dense,
        judge_mask_prev_agent_attn=bool(base.judge_mask_prev_agent_attn) if execution_mode == "default" else False,
        judge_shuffle=judge_shuffle,
        judge_group_agents=bool(base.judge_group_agents),
        judge_group_agents_by_position=bool(base.judge_group_agents_by_position),
        agent_role=base.agent_role,
        agent_temperatures=list(base.agent_temperatures),
        agent_names=list(base.agent_names),
        agent_nums=list(base.agent_nums),
        prefix=base.prefix,
        limit_questions=base.limit_questions,
        question_indices=list(base.question_indices) if base.question_indices else None,
        run_tag=base.run_tag,
        output_dir=str(Path(base.output_root).expanduser() / execution_mode / compare_label / shuffle_label),
        kv_threshold=base.kv_threshold,
        kv_max_anchor_num=base.kv_max_anchor_num,
        kv_window_size=base.kv_window_size,
        kv_thread_workers=base.kv_thread_workers,
        kv_worker_timeout=base.kv_worker_timeout,
    )


def _bundle_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": summary.get("method"),
        "method_label": summary.get("method_label"),
        "execution_mode": summary.get("execution_mode"),
        "judge_shuffle": summary.get("judge_shuffle"),
        "judge_compare_dense_enabled": summary.get("judge_compare_dense_enabled"),
        "acc": summary.get("acc"),
        "JCR": summary.get("JCR"),
        "parse_coverage": summary.get("parse_coverage"),
        "total_cases": summary.get("total_cases"),
    }


async def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_root / "logs" / "run_mmlu_ori_protocol_153.log")

    question_indices = load_question_indices(
        index_file=args.index_file,
        question_indices=args.question_indices,
        limit_questions=args.limit_questions,
    )
    selected_question_count = len(question_indices)
    args.question_indices = question_indices
    args.limit_questions = selected_question_count

    shuffle_flags = _parse_shuffle_flags(list(args.judge_shuffles))
    execution_modes = list(args.execution_modes)
    decision_methods = list(args.decision_methods)

    rows: List[Dict[str, Any]] = []
    bundles: List[Dict[str, Any]] = []
    for decision_method in decision_methods:
        for execution_mode in execution_modes:
            for judge_shuffle in shuffle_flags:
                run_args = _build_single_run_args(
                    args,
                    execution_mode=execution_mode,
                    judge_shuffle=judge_shuffle,
                )
                run_args.decision_method = decision_method
                run_args.output_dir = str(Path(run_args.output_dir).expanduser() / decision_method)
                benchmark_summary = await run_ori_protocol_once(run_args)
                summary = dict(benchmark_summary.get("summary") or {})
                rows.append(_bundle_row(summary))
                bundles.append(
                    {
                        "decision_method": decision_method,
                        "method": benchmark_summary.get("method"),
                        "method_label": benchmark_summary.get("method_label"),
                        "execution_mode": execution_mode,
                        "judge_shuffle": bool(judge_shuffle),
                        "judge_compare_dense_enabled": bool(run_args.judge_compare_dense),
                        "run_dir": run_args.output_dir,
                        "details_path": benchmark_summary.get("details_path"),
                        "summary_path": benchmark_summary.get("summary_path"),
                        "main_table_path": benchmark_summary.get("main_table_path"),
                        "benchmark_summary_path": str(Path(run_args.output_dir) / "benchmark_summary.json"),
                    }
                )

    main_table = (
        f"# Ori-Style MMLU {selected_question_count} Main Table\n\n"
        + _markdown_table(ORI_PROTOCOL_MAIN_TABLE_COLUMNS, rows)
    )
    (output_root / "main_table.md").write_text(main_table.rstrip() + "\n", encoding="utf-8")

    summary_payload = {
        "benchmark_name": "mmlu_ori_protocol_153",
        "protocol_name": "ori_style_online_protocol",
        "llm_name": args.llm_name,
        "mode": args.mode,
        "index_file": str(Path(args.index_file).expanduser()) if args.index_file else None,
        "question_indices": question_indices,
        "selected_question_count": selected_question_count,
        "limit_questions": args.limit_questions,
        "decision_methods": decision_methods,
        "execution_modes": execution_modes,
        "judge_shuffles": [bool(item) for item in shuffle_flags],
        "judge_compare_dense_requested": bool(args.judge_compare_dense),
        "bundles": bundles,
        "rows": rows,
    }
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
