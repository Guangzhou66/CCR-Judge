from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/pychen/Test/model/Qwen2.5-7B-Instruct"
DEFAULT_HUMANEVAL_DATASET_JSON = (
    str(PROJECT_ROOT / "my_datasets" / "humaneval" / "humaneval-py.jsonl")
)


def _load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_label(model_path: str) -> str:
    return Path(str(model_path).rstrip("/")).name or str(model_path)


def _details_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("details", "records", "rows", "cases"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError("Expected an Ori details JSON list or an object with details/records/rows/cases.")


def _record_index(row: dict[str, Any], fallback: int) -> int:
    value = row.get("record_index", fallback)
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _task_id(row: dict[str, Any], *, dataset: str, record_index: int) -> str:
    question_id = row.get("question_id")
    if question_id is not None:
        return str(question_id)
    target = row.get("correct_answer")
    if isinstance(target, dict) and target.get("task_id") is not None:
        return str(target["task_id"])
    return f"{dataset}_{record_index}"


def _gold_answer(row: dict[str, Any]) -> str:
    target = row.get("correct_answer")
    if isinstance(target, dict):
        return str(target.get("entry_point") or target.get("task_id") or "")
    return str(target if target is not None else "").strip()


def _target_payload(
    row: dict[str, Any],
    *,
    dataset: str,
    humaneval_dataset_json: str,
) -> dict[str, Any]:
    target = row.get("correct_answer")
    if isinstance(target, dict):
        payload = dict(target)
        if dataset == "humaneval":
            payload.setdefault("dataset_json", humaneval_dataset_json)
        return payload
    return {"value": target}


def _candidate_item(candidate: dict[str, Any], *, fallback_agent_id: int) -> dict[str, Any]:
    metadata = dict(candidate.get("metadata") or {})
    if "final_answer" in candidate:
        metadata.setdefault("ori_final_answer", candidate.get("final_answer"))
    if "candidate_correct" in candidate:
        metadata.setdefault("ori_candidate_correct", candidate.get("candidate_correct"))
    if "candidate_execution_result" in candidate:
        metadata.setdefault("ori_candidate_execution_result", candidate.get("candidate_execution_result"))

    agent_id = candidate.get("agent_id", fallback_agent_id)
    role = candidate.get("role") or metadata.get("agent_role") or "Solver"
    text = candidate.get("text")
    if text is None:
        text = candidate.get("final_answer", "")

    return {
        "agent_id": str(agent_id),
        "role": str(role),
        "text": str(text),
        "normalized_answer": candidate.get("normalized_answer"),
        "metadata": metadata,
    }


def _case_from_ori_row(
    row: dict[str, Any],
    *,
    dataset: str,
    dataset_config: str,
    generation_regime: str,
    split: str,
    index: int,
    humaneval_dataset_json: str,
    source_details: str,
) -> dict[str, Any]:
    candidates_raw = row.get("candidate_outputs")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError(f"Row {index} has no candidate_outputs.")

    candidates = [
        _candidate_item(candidate, fallback_agent_id=pos + 1)
        for pos, candidate in enumerate(candidates_raw)
        if isinstance(candidate, dict)
    ]
    if not candidates:
        raise ValueError(f"Row {index} has no usable candidate_outputs.")

    record_index = _record_index(row, index)
    candidate_ordering = row.get("candidate_ordering")
    if isinstance(candidate_ordering, list) and len(candidate_ordering) == len(candidates):
        permutation = [str(item) for item in candidate_ordering]
    else:
        permutation = [candidate["agent_id"] for candidate in candidates]

    return {
        "record_index": record_index,
        "task_id": _task_id(row, dataset=dataset, record_index=record_index),
        "question_text": str(row.get("task") or row.get("question_text") or ""),
        "gold_answer": _gold_answer(row),
        "generation_regime": generation_regime,
        "original_agent_ids": [candidate["agent_id"] for candidate in candidates],
        "permutation": permutation,
        "candidate_count": len(candidates),
        "dataset_name": dataset,
        "dataset_config": dataset_config,
        "reference_reasoning": "",
        "target_payload": _target_payload(
            row,
            dataset=dataset,
            humaneval_dataset_json=humaneval_dataset_json,
        ),
        "source_protocol": "ori_protocol",
        "source_details": source_details,
        "source_split": split,
        "source_question_id": row.get("question_id"),
        "source_decision_method": row.get("decision_method"),
        "source_judge_shuffle": row.get("judge_shuffle"),
        "candidates": candidates,
    }


def build_pack(
    *,
    details_path: Path,
    dataset: str,
    split: str,
    dataset_config: str,
    generation_regime: str,
    llm_name: str,
    limit_cases: int | None,
    humaneval_dataset_json: str,
) -> dict[str, Any]:
    rows = _details_rows(_load_json(details_path))
    if limit_cases is not None and int(limit_cases) > 0:
        rows = rows[: int(limit_cases)]
    if not rows:
        raise ValueError(f"No rows found in {details_path}")

    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cases.append(
            _case_from_ori_row(
                row,
                dataset=dataset,
                dataset_config=dataset_config,
                generation_regime=generation_regime,
                split=split,
                index=index,
                humaneval_dataset_json=humaneval_dataset_json,
                source_details=str(details_path),
            )
        )

    candidate_counts = sorted({int(case["candidate_count"]) for case in cases})
    question_indices = [int(case["record_index"]) for case in cases]
    source_hash = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "pack_name": f"{dataset}_ori_saved_candidates_v1",
        "dataset": dataset,
        "split": split,
        "dataset_config": dataset_config,
        "candidate_build_mode": "ori_saved_candidate_outputs",
        "llm_name": llm_name,
        "llm_label": _safe_label(llm_name),
        "generation_regime": generation_regime,
        "execution_side_reuse": False,
        "candidate_count": candidate_counts[0] if len(candidate_counts) == 1 else candidate_counts,
        "agent_name": "AnalyzeAgent" if dataset == "mmlu" else "CodeWriting",
        "agent_role": "MMLU Solver" if dataset == "mmlu" else "Programming Expert",
        "agent_temperature": None,
        "seed": None,
        "question_indices": question_indices,
        "source_protocol": "ori_protocol",
        "source_details": str(details_path),
        "source_details_hash": source_hash,
        "build_timestamp": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ori protocol *_details.json candidate_outputs into a frozen candidate pack."
    )
    parser.add_argument("--details", required=True, help="Path to an Ori protocol *_details.json file.")
    parser.add_argument("--dataset", required=True, choices=["mmlu", "humaneval", "gsm8k"])
    parser.add_argument("--split", default=None, help="Dataset split label. Defaults to val for MMLU/GSM8K and test for HumanEval.")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--generation-regime", default="ori_saved")
    parser.add_argument("--llm-name", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--humaneval-dataset-json", default=DEFAULT_HUMANEVAL_DATASET_JSON)
    parser.add_argument(
        "--result-root",
        default=None,
        help="If --output is omitted, write candidate_cache/<dataset>/<dataset>_<generation-regime>.json under this root.",
    )
    parser.add_argument("--output", default=None, help="Explicit frozen pack output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details_path = Path(args.details).expanduser()
    dataset = str(args.dataset).lower()
    split = args.split or ("test" if dataset == "humaneval" else "val")

    if args.output:
        output_path = Path(args.output).expanduser()
    elif args.result_root:
        result_root = Path(args.result_root).expanduser()
        output_path = (
            result_root
            / "candidate_cache"
            / dataset
            / f"{dataset}_{args.generation_regime}.json"
        )
    else:
        raise SystemExit("Provide either --output or --result-root.")

    pack = build_pack(
        details_path=details_path,
        dataset=dataset,
        split=split,
        dataset_config=str(args.dataset_config or ""),
        generation_regime=str(args.generation_regime),
        llm_name=str(args.llm_name),
        limit_cases=args.limit_cases,
        humaneval_dataset_json=str(args.humaneval_dataset_json),
    )
    _write_json(output_path, pack)
    print(f"Wrote frozen candidate pack: {output_path}")
    print(f"cases={len(pack['cases'])} candidate_count={pack['candidate_count']}")


if __name__ == "__main__":
    main()
