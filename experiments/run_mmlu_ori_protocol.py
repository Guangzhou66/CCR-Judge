from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Union

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from KVCOMM.graph.graph import Graph
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.llm import LLM
from KVCOMM.utils.log import configure_logging, logger
from dataset_adapters import create_dataset_adapter
from experiments.evaluate_mmlu_ori_protocol import evaluate_ori_protocol

# Register faithful MMLU path and the Ori-style decision node locally.
import KVCOMM.prompt.mmlu_prompt_set  # noqa: F401,E402
import KVCOMM.prompt.gsm8k_prompt_set  # noqa: F401,E402
import KVCOMM.prompt.humaneval_prompt_set  # noqa: F401,E402
import KVCOMM.agents.analyze_agent  # noqa: F401,E402
import KVCOMM.agents.code_writing  # noqa: F401,E402
import KVCOMM.agents.final_decision  # noqa: F401,E402
import KVCOMM.agents.ori_protocol_final_decision  # noqa: F401,E402
import KVCOMM.agents.ori_protocol_repair_final_decision  # noqa: F401,E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def _safe_llm_label(llm_name: str) -> str:
    label = Path(llm_name).name.strip()
    if label:
        return label
    return llm_name.replace("/", "_").replace("\\", "_")


def _markdown_table(columns: Sequence[tuple[str, str]], rows: Sequence[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    if not rows:
        lines.append("| " + " | ".join("-" for _ in columns) + " |")
        return "\n".join(lines)
    for row in rows:
        rendered = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                rendered.append("-")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


ORI_PROTOCOL_MAIN_TABLE_COLUMNS = [
    ("method_label", "Method"),
    ("execution_mode", "Execution Mode"),
    ("judge_shuffle", "Shuffle"),
    ("judge_compare_dense_enabled", "Compare Dense"),
    ("acc", "Acc"),
    ("JCR", "JCR"),
    ("parse_coverage", "Parse Coverage"),
    ("total_cases", "total_cases"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ori-style online MMLU protocol.")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument("--decision_method", type=str, default="OriProtocolFinalSelectBest")
    parser.add_argument(
        "--execution_mode",
        type=str,
        default="default",
        choices=["default", "allow_kv_reuse"],
    )
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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--kv-threshold", type=float, default=None)
    parser.add_argument("--kv-max-anchor-num", type=int, default=20)
    parser.add_argument("--kv-window-size", type=int, default=None)
    parser.add_argument("--kv-thread-workers", type=int, default=None)
    parser.add_argument("--kv-worker-timeout", type=float, default=None)
    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


def _build_agent_kwargs(args: argparse.Namespace, total_agents: int) -> List[Dict[str, Any]]:
    if args.execution_temperature is not None:
        if args.agent_temperatures != [0.2]:
            raise ValueError("Use either --execution-temperature or --agent-temperatures, not both.")
        temps = [float(args.execution_temperature)]
    else:
        temps = list(args.agent_temperatures)
    if len(temps) == 1:
        temps = temps * total_agents
    elif len(temps) != total_agents:
        raise ValueError(f"--agent-temperatures expects 1 or {total_agents} values, got {len(temps)}")
    return [{"temperature": temp, "role": args.agent_role} for temp in temps]


def _effective_graph_decision_method(
    *,
    requested_decision_method: str,
    dataset_adapter: Any,
) -> str:
    dataset_name = str(getattr(dataset_adapter, "dataset_name", "") or "").strip().lower()
    if requested_decision_method == "FinalSelectBest" and dataset_name and dataset_name != "mmlu":
        # Non-MMLU dataset runners still expose the public baseline name `FinalSelectBest`,
        # but we execute through the Ori-style wrapper so compare-dense metadata is attached
        # in the same structured way as the CCR path.
        return "OriProtocolFinalSelectBest"
    return requested_decision_method


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
    ],
    n_agents: int,
) -> Dict[str, Any]:
    fixed_spatial_masks: List[List[int]] | None = None
    fixed_temporal_masks: List[List[int]] | None = None
    node_kwargs = None

    def generate_layered_graph(n: int, layer_num: int = 2) -> List[List[int]]:
        adj_matrix = [[0] * n for _ in range(n)]
        base_size = n // layer_num
        remainder = n % layer_num
        layers: List[int] = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(n):
            current_layer = layers[i]
            for j in range(n):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_mesh_graph(n: int) -> List[List[int]]:
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n: int) -> List[List[int]]:
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(1, n):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(n_agents)] for j in range(n_agents)]
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Random":
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(n_agents)] for j in range(n_agents)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(n_agents)] for j in range(n_agents)]
        fixed_temporal_masks = [[1 if i == 0 and j == n_agents - 1 else 0 for i in range(n_agents)] for j in range(n_agents)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(n_agents)] for _ in range(n_agents)]
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(n_agents)
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Mesh":
        fixed_spatial_masks = generate_mesh_graph(n_agents)
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(n_agents)
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "fixed_spatial_masks": fixed_spatial_masks,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


def _apply_generation_defaults(args: argparse.Namespace) -> None:
    # Freeze generation control at runner level without altering prompts or
    # candidate/judge algorithms.
    LLM.DEFAULT_MAX_TOKENS = int(args.max_new_tokens)
    LLM.DEFAULT_TEMPERATURE = float(args.judge_temperature)
    os.environ["KVCOMM_STOP_CONDITIONS"] = str(args.stop_conditions)


def _resolve_scorer_mode(
    *,
    args: argparse.Namespace,
    dataset_adapter: Any,
) -> str:
    adapter_mode = str(dataset_adapter.official_scorer_mode())
    requested_mode = adapter_mode if args.scorer_mode is None else str(args.scorer_mode).strip()
    if requested_mode != adapter_mode:
        raise ValueError(
            f"scorer mode mismatch for {dataset_adapter.dataset_name}: "
            f"requested {requested_mode!r}, expected {adapter_mode!r}"
        )
    return adapter_mode


def _dataset_descriptor(
    *,
    dataset_adapter: Any,
    dataset_split: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset_name": str(dataset_adapter.dataset_name),
        "dataset_split": str(dataset_split),
    }
    if hasattr(dataset_adapter, "dataset_root"):
        payload["dataset_root"] = str(getattr(dataset_adapter, "dataset_root"))
    if hasattr(dataset_adapter, "dataset_json"):
        payload["dataset_json"] = str(getattr(dataset_adapter, "dataset_json"))
    if hasattr(dataset_adapter, "config_name"):
        payload["config_name"] = str(getattr(dataset_adapter, "config_name"))
    if hasattr(dataset_adapter, "include_fact"):
        include_fact = getattr(dataset_adapter, "include_fact")
        payload["include_fact"] = None if include_fact is None else bool(include_fact)
    return payload


def _runner_config_payload(
    *,
    args: argparse.Namespace,
    dataset_adapter: Any,
    dataset_split: str,
    agent_names: Sequence[str],
    agent_kwargs: Sequence[Dict[str, Any]],
    scorer_mode: str,
) -> Dict[str, Any]:
    question_scope = {
        "limit_questions": None if args.limit_questions is None else int(args.limit_questions),
        "question_indices": None if args.question_indices is None else [int(item) for item in args.question_indices],
        "full_dataset": bool(args.limit_questions is None and not args.question_indices),
    }
    execution_temperatures = [float(item.get("temperature", 0.0)) for item in agent_kwargs]
    unique_execution_temperatures = sorted(set(execution_temperatures))
    execution_temperature: float | List[float]
    if len(unique_execution_temperatures) == 1:
        execution_temperature = unique_execution_temperatures[0]
    else:
        execution_temperature = unique_execution_temperatures
    return {
        "model_path": str(args.llm_name),
        "dataset": _dataset_descriptor(dataset_adapter=dataset_adapter, dataset_split=dataset_split),
        "question_scope": question_scope,
        "shuffle_flag": bool(args.judge_shuffle),
        "topology_mode": str(args.mode),
        "execution_mode": str(args.execution_mode),
        "decision_method": str(args.decision_method),
        "judge_compare_dense": bool(args.judge_compare_dense),
        "judge_mask_prev_agent_attn": bool(args.judge_mask_prev_agent_attn),
        "kv_max_anchor_num": None if args.execution_mode != "allow_kv_reuse" else int(args.kv_max_anchor_num),
        "judge_group_agents": bool(args.judge_group_agents),
        "judge_group_agents_by_position": bool(args.judge_group_agents_by_position),
        "execution_temperature": execution_temperature,
        "agent_temperatures": execution_temperatures,
        "judge_temperature": float(args.judge_temperature),
        "max_new_tokens": int(args.max_new_tokens),
        "stop_conditions": str(args.stop_conditions),
        "scorer_mode": str(scorer_mode),
        "agent_role": str(args.agent_role),
        "agent_names_flat": [str(name) for name in agent_names],
        "batch_size": int(args.batch_size),
        "prefix": str(args.prefix),
    }


async def run_dataset_ori_protocol_once(
    args: argparse.Namespace,
    *,
    dataset_adapter: Any,
    dataset_split: str = "val",
    benchmark_name: str = "mmlu_ori_protocol",
) -> Dict[str, Any]:
    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_dir / "logs" / "run_mmlu_ori_protocol.log")

    if args.judge_shuffle:
        os.environ["KVCOMM_JUDGE_SHUFFLE"] = "1"
    else:
        os.environ.pop("KVCOMM_JUDGE_SHUFFLE", None)
    _apply_generation_defaults(args)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    llm_label = _safe_llm_label(args.llm_name)
    details_path = output_dir / f"{args.domain}_{llm_label}_{timestamp}_details.json"
    summary_path = output_dir / f"{args.domain}_{llm_label}_{timestamp}_summary.json"
    result_path = output_dir / f"{args.domain}_{llm_label}_{timestamp}.json"
    main_table_path = output_dir / "main_table.md"
    benchmark_summary_path = output_dir / "benchmark_summary.json"

    scorer_mode = _resolve_scorer_mode(args=args, dataset_adapter=dataset_adapter)
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    agent_kwargs = _build_agent_kwargs(args, len(agent_names))
    kwargs = get_kwargs(args.mode, len(agent_names))
    kwargs["node_kwargs"] = agent_kwargs
    runner_config = _runner_config_payload(
        args=args,
        dataset_adapter=dataset_adapter,
        dataset_split=dataset_split,
        agent_names=agent_names,
        agent_kwargs=agent_kwargs,
        scorer_mode=scorer_mode,
    )

    if args.execution_mode == "allow_kv_reuse":
        kv_config = KVCommConfig.from_env().apply_overrides(
            threshold=args.kv_threshold,
            max_anchor_num=args.kv_max_anchor_num,
            window_size=args.kv_window_size,
            thread_pool_workers=args.kv_thread_workers,
            worker_timeout=args.kv_worker_timeout,
            group_judge_agent_anchors=args.judge_group_agents,
            group_judge_agent_anchors_by_position=args.judge_group_agents_by_position,
        )
    else:
        kv_config = KVCommConfig.from_env()

    graph_decision_method = _effective_graph_decision_method(
        requested_decision_method=str(args.decision_method),
        dataset_adapter=dataset_adapter,
    )

    graph = Graph(
        domain=dataset_adapter.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=graph_decision_method,
        kv_config=kv_config,
        **kwargs,
    )

    dataset_val = dataset_adapter.load_online_dataset(
        split=dataset_split,
        question_indices=None,
        limit_questions=None,
    )
    eval_kwargs: Dict[str, Any] = {}
    if args.execution_mode == "allow_kv_reuse":
        eval_kwargs = {
            "prefix": args.prefix,
            "output_dir": str(output_dir),
            "judge_compare_dense": args.judge_compare_dense,
        }
    if args.judge_mask_prev_agent_attn:
        eval_kwargs["judge_mask_prev_agent_attn"] = True

    summary_payload = await evaluate_ori_protocol(
        graph=graph,
        dataset=dataset_val,
        limit_questions=args.limit_questions,
        question_indices=args.question_indices,
        eval_batch_size=args.batch_size,
        mode=args.execution_mode,
        execution_mode=args.execution_mode,
        judge_compare_dense_enabled=bool(args.judge_compare_dense),
        judge_shuffle=bool(args.judge_shuffle),
        topology_mode=args.mode,
        decision_method=args.decision_method,
        results_path=str(details_path),
        summary_path=str(summary_path),
        main_table_path=str(main_table_path),
        dataset_name=dataset_adapter.dataset_name,
        dataset_adapter=dataset_adapter,
        **eval_kwargs,
    )
    summary_payload["scorer_mode"] = scorer_mode
    summary_payload["runner_config"] = runner_config
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_payload = {
        **summary_payload,
        "timestamp": timestamp,
        "llm_name": args.llm_name,
        "llm_label": llm_label,
        "run_tag": args.run_tag,
        "details_path": str(details_path),
        "summary_path": str(summary_path),
        "main_table_path": str(main_table_path),
        "online_protocol": True,
        "runner_config": runner_config,
    }
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    benchmark_summary = {
        "benchmark_name": benchmark_name,
        "protocol_name": "ori_style_online_protocol",
        "llm_name": args.llm_name,
        "method": summary_payload.get("method"),
        "method_label": summary_payload.get("method_label"),
        "mode": args.mode,
        "execution_mode": args.execution_mode,
        "judge_compare_dense_enabled": bool(args.judge_compare_dense),
        "judge_shuffle": bool(args.judge_shuffle),
        "decision_method": args.decision_method,
        "total_cases": summary_payload["total_cases"],
        "scorer_mode": scorer_mode,
        "runner_config": runner_config,
        "summary": summary_payload,
        "details_path": str(details_path),
        "summary_path": str(summary_path),
        "result_path": str(result_path),
        "main_table_path": str(main_table_path),
    }
    benchmark_summary_path.write_text(json.dumps(benchmark_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.opt(colors=True).info("<blue>[RESULT SAVED]</blue> {}", str(result_path))
    logger.opt(colors=True).info("<blue>[BENCHMARK SUMMARY SAVED]</blue> {}", str(benchmark_summary_path))
    return benchmark_summary


async def run_ori_protocol_once(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_adapter = create_dataset_adapter("mmlu")
    return await run_dataset_ori_protocol_once(
        args,
        dataset_adapter=dataset_adapter,
        dataset_split="val",
        benchmark_name="mmlu_ori_protocol",
    )


async def main() -> None:
    args = parse_args()
    await run_ori_protocol_once(args)


if __name__ == "__main__":
    asyncio.run(main())
