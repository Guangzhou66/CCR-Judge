import argparse
import asyncio
import copy
import json
import re
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
import random
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from KVCOMM.graph.graph import Graph
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.tools.coding.python_executor import PyExecutor
from KVCOMM.tools.reader.readers import JSONLReader
from KVCOMM.utils.globals import Time
from KVCOMM.utils.log import configure_logging, logger
from KVCOMM.utils.metrics import metrics_recorder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_result(result_file: Path) -> list:
    if not result_file.exists():
        os.makedirs(result_file.parent, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump([], file)
    with open(result_file, "r", encoding="utf-8") as file:
        return json.load(file)


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


def _extract_selected_agent_id(judge_text: str) -> str | None:
    if not isinstance(judge_text, str):
        return None
    stripped = judge_text.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    if not first_line.lower().startswith("selected agent id"):
        return None
    parts = first_line.split(":", 1)
    if len(parts) != 2:
        return None
    value = parts[1].strip()
    return value or None


def _extract_selected_agent_ids(judge_text: str) -> Dict[str, str | None]:
    """Extract selected agent ids from judge headers.

    Supports:
    - Selected agent id: <id>
    - Selected agent id (dense): <id>
    """
    result: Dict[str, str | None] = {
        "selected_agent_id": None,
        "dense_selected_agent_id": None,
    }
    if not isinstance(judge_text, str):
        return result
    stripped = judge_text.strip()
    if not stripped:
        return result

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.lower().startswith("selected agent id"):
            break
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().lower()
        value = parts[1].strip() or None
        if "dense" in key:
            result["dense_selected_agent_id"] = value
        else:
            result["selected_agent_id"] = value

    return result


def _strip_selected_agent_header(judge_text: str) -> str:
    if not isinstance(judge_text, str):
        return str(judge_text)
    stripped = judge_text.lstrip()
    if not stripped.lower().startswith("selected agent id"):
        return judge_text
    lines = stripped.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].lower().startswith("selected agent id"):
        idx += 1
    return "\n".join(lines[idx:]).lstrip()

def _extract_python_code(text: str) -> str:
    """Extract python code from a model response (best-effort)."""
    if not isinstance(text, str):
        return str(text)
    raw = text.strip()
    if not raw:
        return ""

    match = re.search(
        r"```python\s*\r?\n(.*?)\r?\n```",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\w*\s*\r?\n(.*?)\r?\n```", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) <= 1:
            return ""
        return "\n".join(lines[1:]).removesuffix("```").strip()

    return raw


def parse_args():
    parser = argparse.ArgumentParser(description="KVCOMM Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="my_datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"], help="The communication topology among agents."
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--domain", type=str, default="humaneval")
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["CodeWriting"],
        help="List of agent names in the graph.",
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[5],
        help="List of agent counts corresponding to agent names.",
    )
    parser.add_argument("--decision_method", type=str, default="FinalRefer", help="Decision method for the graph.")
    parser.add_argument(
        "--execution_mode",
        type=str,
        default="default",
        choices=["default", "allow_kv_reuse"],
        help="Execution strategy for the graph.",
    )
    parser.add_argument(
        "--judge-compare-dense",
        action="store_true",
        help="In KVCOMM mode, run an extra dense-prefill judge pass per question and log whether it selects the same agent.",
    )
    parser.add_argument(
        "--judge-shuffle",
        action="store_true",
        help="Shuffle candidate agent order for FinalSelectBest judge.",
    )
    parser.add_argument(
        "--judge-group-agents",
        action="store_true",
        help="Let judge share KV anchors across candidate agents instead of per-agent anchors.",
    )
    parser.add_argument(
        "--agent-role",
        type=str,
        default=None,
        help="Optional shared role name for all CodeWriting agents (e.g., 'Normal Programmer').",
    )
    parser.add_argument(
        "--agent-temperatures",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of temperatures for each agent instance (flattened).",
    )
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "result" / "humaneval"), help="Directory to save the output results.")
    parser.add_argument("--prefix", type=str, default="The task is:\n\n", help="The prefix text for the input query, kept the same as the default dense prefill mode.")
    parser.add_argument("--kv-threshold", type=float, default=None, help="Threshold for key-value memory usage.")
    parser.add_argument("--kv-max-anchor-num", type=int, default=None, help="Maximum number of anchors for key-value memory.")
    parser.add_argument("--kv-window-size", type=int, default=None, help="Window size for key-value memory update.")
    parser.add_argument("--kv-thread-workers", type=int, default=None, help="Number of thread workers for key-value memory processing.")
    parser.add_argument("--kv-worker-timeout", type=float, default=None, help="Timeout for key-value memory workers processing.")

    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    if args.judge_shuffle:
        os.environ["KVCOMM_JUDGE_SHUFFLE"] = "1"
    else:
        os.environ.pop("KVCOMM_JUDGE_SHUFFLE", None)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_dir / "logs/log.txt")
    dataset = JSONLReader.parse_file(args.dataset_json)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_file = output_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
    latency_target = str(output_dir)

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]

    agent_temps = None
    if args.agent_temperatures is not None:
        agent_temps = list(args.agent_temperatures)
        total_agents = len(agent_names)
        if len(agent_temps) == 1:
            agent_temps = agent_temps * total_agents
        elif len(agent_temps) != total_agents:
            raise ValueError(
                f"--agent-temperatures expects 1 or {total_agents} values, got {len(agent_temps)}"
            )

    kwargs = get_kwargs(args.mode, len(agent_names))

    if agent_temps is not None or args.agent_role is not None:
        node_kwargs = kwargs.get("node_kwargs")
        if node_kwargs is None:
            node_kwargs = [{} for _ in agent_names]
        elif len(node_kwargs) == 1 and len(agent_names) > 1:
            base = node_kwargs[0]
            node_kwargs = [dict(base) for _ in agent_names]
        elif len(node_kwargs) != len(agent_names):
            raise ValueError("Length of node_kwargs does not match number of agents.")

        if agent_temps is not None:
            for kw, temp in zip(node_kwargs, agent_temps):
                kw["temperature"] = temp

        if args.agent_role is not None:
            for kw in node_kwargs:
                kw["role"] = args.agent_role

        kwargs["node_kwargs"] = node_kwargs

    kv_config: Optional[KVCommConfig] = None
    if args.execution_mode == "allow_kv_reuse":
        kv_config = KVCommConfig.from_env().apply_overrides(
            threshold=args.kv_threshold,
            max_anchor_num=args.kv_max_anchor_num,
            window_size=args.kv_window_size,
            thread_pool_workers=args.kv_thread_workers,
            worker_timeout=args.kv_worker_timeout,
            group_judge_agent_anchors=args.judge_group_agents,
        )
    else:
        kv_config = KVCommConfig.from_env()

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        kv_config=kv_config,
        **kwargs,
    )

    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = 0, 0
    judge_agree_num, judge_agree_total = 0, 0

    for i_batch in tqdm(range(num_batches), desc="humaneval:batches", unit="batch"):
        logger.opt(colors=True).info(f"<blue>[BATCH]</blue> {i_batch} {'-' * 40}")
        start_ts = time.time()
        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if not current_batch:
            logger.warning("No more data available.")
            break

        tasks = []
        meta_info = []
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            task = record["prompt"]
            tests = record["test"]
            input_dict = {"task": task, "_batch_index": i_batch}

            mode_kwargs = {}
            if args.execution_mode == "allow_kv_reuse":
                mode_kwargs = {
                    "prefix": args.prefix,
                    "output_dir": latency_target
                }
                if args.judge_compare_dense:
                    mode_kwargs["judge_compare_dense"] = True

            tasks.append(
                asyncio.create_task(
                    realized_graph.arun(
                        input_dict,
                        1,
                        mode=args.execution_mode,
                        **mode_kwargs,
                    )
                )
            )
            meta_info.append({"task": task, "tests": tests})

        batch_results = await asyncio.gather(*tasks)
        results_by_task = {result.get("task"): result.get("answers", []) for result in batch_results}
        data = load_result(result_file)

        for info in meta_info:
            task = info["task"]
            tests = info["tests"]
            answers = results_by_task.get(task, [])
            response = answers if isinstance(answers, list) else [answers]
            if not response:
                candidate = ""
                selected_agent_id = None
                dense_selected_agent_id = None
            else:
                candidate = response[0]
                selected_ids = _extract_selected_agent_ids(str(candidate))
                selected_agent_id = selected_ids.get("selected_agent_id")
                dense_selected_agent_id = selected_ids.get("dense_selected_agent_id")
            if isinstance(candidate, str):
                content = _strip_selected_agent_header(candidate)
                code = _extract_python_code(content)
            else:
                code = _extract_python_code(_strip_selected_agent_header(str(candidate)))

            is_solved, _, _ = PyExecutor().execute(code, [tests], timeout=5)
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            judge_agree = None
            judge_agree_rate = None
            if selected_agent_id is not None and dense_selected_agent_id is not None:
                judge_agree_total += 1
                judge_agree = bool(selected_agent_id == dense_selected_agent_id)
                judge_agree_num += int(judge_agree)
                judge_agree_rate = (judge_agree_num / judge_agree_total) if judge_agree_total else None

            updated_item = {
                "Question": task,
                "Tests": tests,
                "Selected agent id": selected_agent_id,
                "Dense selected agent id": dense_selected_agent_id,
                "Judge agree": judge_agree,
                "Judge agree rate": judge_agree_rate,
                "Raw answer": str(candidate),
                "Attempt answer": code,
                "Solved": bool(is_solved),
                "Solution": code,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
            }
            data.append(updated_item)

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        logger.opt(colors=True).info(
            f"<blue>[BATCH TIME]</blue> {time.time() - start_ts:.3f}s"
        )
        logger.opt(colors=True).info(f"<blue>[ACCURACY]</blue> {accuracy:.4f}")
        metrics_recorder.log_cumulative(batch_index=i_batch)

def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
    ],
    N: int,
):
    fixed_spatial_masks: List[List[int]] = None                
    fixed_temporal_masks: List[List[int]] = None                
    node_kwargs = None

    def generate_layered_graph(n, layer_num=2):
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
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

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal Programmer"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random":
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "fixed_spatial_masks": fixed_spatial_masks,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == "__main__":
    asyncio.run(main())
