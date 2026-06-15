from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm.auto import tqdm

from dataset_adapters import create_dataset_adapter
from paper_repair.protocol.candidate_generator import PaperCandidateGenerator
from paper_repair.repair.ccr_types import CCRCandidateView as CandidateView, FrozenCandidateRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_FILE = PROJECT_ROOT / "experiments" / "mmlu_153_seed888_indices.json"
DEFAULT_AGENT_COUNT = 4

DATASET_AGENT_DEFAULTS: Dict[str, Dict[str, str]] = {
    "mmlu": {"agent_name": "AnalyzeAgent", "agent_role": "MMLU Solver"},
    "gsm8k": {"agent_name": "MathSolver", "agent_role": "Math Solver"},
    "humaneval": {"agent_name": "CodeWriting", "agent_role": "Programming Expert"},
}

def _console_debug_enabled() -> bool:
    value = os.environ.get("KVCOMM_PROGRESS_ONLY", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}

def _debug_log(message: str) -> None:
    path_value = os.environ.get("PAPER_REPAIR_DEBUG_LOG", "").strip()
    line = f"[PAPER_REPAIR_DEBUG] {message}"
    if _console_debug_enabled():
        print(line, flush=True)
    if path_value:
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _safe_llm_label(llm_name: str) -> str:
    label = Path(llm_name).name.strip()
    if label:
        return label
    return llm_name.replace("/", "_").replace("\\", "_")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_question_indices(
    *,
    index_file: str | None = None,
    question_indices: Sequence[int] | None = None,
    limit_questions: int | None = None,
) -> List[int]:
    _debug_log("[DATA INIT] resolving question indices")
    if question_indices:
        _debug_log(f"[DATA INIT] using explicit question indices count={len(question_indices)}")
        return [int(item) for item in question_indices]
    if index_file:
        _debug_log(f"[DATA INIT] before index file load path={index_file}")
        payload = json.loads(Path(index_file).expanduser().read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "question_indices" in payload:
            indices = payload["question_indices"]
        else:
            indices = payload
        result = [int(item) for item in indices]
        if limit_questions is not None:
            _debug_log(f"[DATA INIT] index file loaded count={len(result)} limit={limit_questions}")
            return result[:limit_questions]
        _debug_log(f"[DATA INIT] index file loaded count={len(result)}")
        return result
    if limit_questions is None:
        raise ValueError("Either question_indices, index_file, or limit_questions must be provided.")
    _debug_log(f"[DATA INIT] falling back to range limit={limit_questions}")
    return list(range(int(limit_questions)))


def load_dataset_records(
    question_indices: Sequence[int],
    *,
    dataset_name: str = "mmlu",
    split: str = "val",
    adapter_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    _debug_log(f"[DATA INIT] before dataset load dataset={dataset_name} split={split}")
    adapter = create_dataset_adapter(dataset_name, **dict(adapter_kwargs or {}))
    dataset = adapter.load_online_dataset(
        split=split,
        question_indices=None,
        limit_questions=None,
    )
    _debug_log(f"[DATA INIT] dataset loaded dataset={dataset_name} split={split}")
    records: List[Dict[str, Any]] = []
    for record_index in question_indices:
        raw_record = dataset[int(record_index)]
        input_payload = dataset.record_to_input(raw_record)
        target_payload = dataset.record_to_target_answer(raw_record)
        if isinstance(target_payload, dict):
            serialized_target = dict(target_payload)
            task_id = str(
                target_payload.get("task_id")
                or target_payload.get("entry_point")
                or f"{adapter.dataset_name}_{split}_{int(record_index)}"
            )
            reference_reasoning = str(
                target_payload.get("canonical_solution")
                or target_payload.get("step")
                or ""
            )
        else:
            serialized_target = {"value": target_payload}
            task_id = f"{adapter.dataset_name}_{split}_{int(record_index)}"
            reference_reasoning = ""
        if not reference_reasoning and hasattr(dataset, "record_to_reasoning"):
            try:
                reference_reasoning = str(dataset.record_to_reasoning(raw_record) or "")
            except Exception:
                reference_reasoning = ""
        records.append(
            {
                "record_index": int(record_index),
                "task_id": task_id,
                "question_text": str(input_payload.get("task") or ""),
                "gold_answer": adapter.normalize_gold_answer(target_payload),
                "reference_reasoning": reference_reasoning,
                "target_payload": serialized_target,
                "dataset_name": adapter.dataset_name,
                "dataset_config": str(adapter.config.get("config_name") or ""),
            }
        )
    _debug_log(f"[DATA INIT] selected question ids resolved count={len(records)}")
    return records


def load_mmlu_records(question_indices: Sequence[int], *, split: str = "val") -> List[Dict[str, Any]]:
    return load_dataset_records(
        question_indices,
        dataset_name="mmlu",
        split=split,
    )


def _resolved_agent_defaults(
    *,
    dataset_name: str,
    agent_name: str | None,
    agent_role: str | None,
) -> tuple[str, str]:
    defaults = DATASET_AGENT_DEFAULTS.get(dataset_name, DATASET_AGENT_DEFAULTS["mmlu"])
    return str(agent_name or defaults["agent_name"]), str(agent_role or defaults["agent_role"])


def _candidate_view(agent: PaperCandidateGenerator, generation: Any, *, dataset_name: str = "mmlu") -> CandidateView:
    adapter = create_dataset_adapter(dataset_name)
    parsed = adapter.parse_candidate_output(generation.text, dict(generation.metadata or {}))
    metadata = dict(generation.metadata or {})
    return CandidateView(
        agent_id=str(agent.id),
        role=agent.role,
        text=generation.text,
        normalized_answer=parsed.get("normalized_answer"),
        conclusion=parsed.get("conclusion"),
        evidence=parsed.get("evidence"),
        strongest_alternative_considered=parsed.get("strongest_alternative_considered"),
        strongest_alternative_option=parsed.get("strongest_alternative_option"),
        rejected_option=parsed.get("rejected_option"),
        why_not_that_option=parsed.get("why_not_that_option"),
        why_not_rejected_option=parsed.get("why_not_rejected_option"),
        alternative_overturn_flag=parsed.get("alternative_overturn_flag"),
        alternative_overturn_yes=bool(parsed.get("alternative_overturn_yes")),
        alternative_supported_flag=parsed.get("alternative_supported_flag"),
        alternative_supported_yes=bool(parsed.get("alternative_supported_yes")),
        metadata=metadata,
    )


async def _generate_parallel_case(
    agents: Sequence[PaperCandidateGenerator],
    question_text: str,
    *,
    dataset_name: str = "mmlu",
    sample_idx: int | None = None,
) -> List[CandidateView]:
    _debug_log("[MULTI-AGENT] before parallel generation launch")
    if sample_idx is not None:
        _debug_log(f"[MULTI-SAMPLE] before parallel generation launch for sample {sample_idx}")
    tasks = []
    for agent in agents:
        _debug_log(f"[MULTI-AGENT] before agent {agent.id} generation")
        if sample_idx is not None:
            _debug_log(f"[MULTI-SAMPLE] before agent {agent.id} generation for sample {sample_idx}")
        tasks.append(
            asyncio.create_task(
                agent.generate(
                    question_text=question_text,
                    prior_candidates=[],
                )
            )
        )
    results = await asyncio.gather(*tasks)
    _debug_log("[MULTI-AGENT] parallel generation returned")
    if sample_idx is not None:
        _debug_log(f"[MULTI-SAMPLE] parallel generation returned for sample {sample_idx}")
    for agent, _generation in zip(agents, results):
        _debug_log(f"[MULTI-AGENT] agent {agent.id} generation returned")
        if sample_idx is not None:
            _debug_log(f"[MULTI-SAMPLE] agent {agent.id} generation returned for sample {sample_idx}")
    _debug_log(f"[MULTI-AGENT] returned candidate count = {len(results)}")
    if sample_idx is not None:
        _debug_log(f"[MULTI-SAMPLE] returned candidate count for sample {sample_idx} = {len(results)}")
    return [_candidate_view(agent, generation, dataset_name=dataset_name) for agent, generation in zip(agents, results)]


async def _generate_progressive_case(
    agents: Sequence[PaperCandidateGenerator],
    question_text: str,
    *,
    dataset_name: str = "mmlu",
) -> List[CandidateView]:
    candidates: List[CandidateView] = []
    for agent in agents:
        _debug_log(f"[FIRST SAMPLE] before progressive generation agent={agent.id}")
        generation = await agent.generate(
            question_text=question_text,
            prior_candidates=[
                {
                    "agent_id": candidate.agent_id,
                    "role": candidate.role,
                    "text": candidate.text,
                }
                for candidate in candidates
            ],
        )
        _debug_log(f"[FIRST SAMPLE] progressive generation returned agent={agent.id}")
        candidates.append(_candidate_view(agent, generation, dataset_name=dataset_name))
    return candidates


def frozen_record_from_dict(case: Dict[str, Any]) -> FrozenCandidateRecord:
    dataset_name = str(case.get("dataset_name") or "mmlu")
    dataset_config = str(case.get("dataset_config") or "")
    adapter_kwargs: Dict[str, Any] = {}
    target_payload = dict(case.get("target_payload") or {})
    if dataset_config:
        adapter_kwargs["config_name"] = dataset_config
    if isinstance(target_payload.get("dataset_json"), str):
        adapter_kwargs["dataset_json"] = target_payload["dataset_json"]
    adapter = create_dataset_adapter(dataset_name, **adapter_kwargs)

    candidates = [
        _candidate_from_frozen_item(item, adapter=adapter)
        for item in case.get("candidates") or []
    ]
    permutation = [str(item) for item in (case.get("permutation") or case.get("original_agent_ids") or [candidate.agent_id for candidate in candidates])]
    return FrozenCandidateRecord(
        record_index=int(case["record_index"]),
        question_text=str(case["question_text"]),
        gold_answer=str(case["gold_answer"]).strip(),
        generation_regime=str(case["generation_regime"]),
        candidates=candidates,
        permutation=permutation,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        task_id=str(case.get("task_id") or ""),
        reference_reasoning=str(case.get("reference_reasoning") or ""),
        target_payload=target_payload,
    )


def _candidate_from_frozen_item(item: Dict[str, Any], *, adapter: Any) -> CandidateView:
    raw_text = str(item.get("text") or "")
    metadata = dict(item.get("metadata") or {})
    try:
        parsed = adapter.parse_candidate_output(raw_text, metadata)
    except Exception:
        parsed = {}

    def parsed_or_item(key: str, default: Any = None) -> Any:
        value = parsed.get(key)
        if value is not None:
            return value
        return item.get(key, default)

    return CandidateView(
        agent_id=str(item["agent_id"]),
        role=str(item.get("role") or "MMLU Solver"),
        text=raw_text,
        normalized_answer=(
            None
            if parsed.get("normalized_answer") is None
            else str(parsed.get("normalized_answer")).strip()
        ),
        conclusion=parsed_or_item("conclusion"),
        evidence=parsed_or_item("evidence"),
        strongest_alternative_considered=parsed_or_item("strongest_alternative_considered"),
        strongest_alternative_option=parsed_or_item("strongest_alternative_option"),
        rejected_option=parsed_or_item("rejected_option"),
        why_not_that_option=parsed_or_item("why_not_that_option"),
        why_not_rejected_option=parsed_or_item("why_not_rejected_option"),
        alternative_overturn_flag=parsed_or_item("alternative_overturn_flag"),
        alternative_overturn_yes=bool(parsed_or_item("alternative_overturn_yes", False)),
        alternative_supported_flag=parsed_or_item("alternative_supported_flag"),
        alternative_supported_yes=bool(parsed_or_item("alternative_supported_yes", False)),
        metadata=metadata,
    )


def frozen_record_to_dict(record: FrozenCandidateRecord) -> Dict[str, Any]:
    return {
        "record_index": int(record.record_index),
        "task_id": record.task_id,
        "question_text": record.question_text,
        "gold_answer": record.gold_answer,
        "generation_regime": record.generation_regime,
        "original_agent_ids": record.candidate_agent_ids(),
        "permutation": list(record.permutation),
        "candidate_count": record.candidate_count,
        "dataset_name": record.dataset_name,
        "dataset_config": record.dataset_config,
        "reference_reasoning": record.reference_reasoning,
        "target_payload": dict(record.target_payload),
        "candidates": [candidate.to_dict() for candidate in record.candidates],
    }


async def build_frozen_candidate_pack(
    *,
    llm_name: str,
    generation_regime: str,
    question_indices: Sequence[int],
    dataset_name: str = "mmlu",
    split: str = "val",
    agent_name: str | None = None,
    agent_role: str | None = None,
    agent_temperature: float = 0.2,
    agent_count: int = DEFAULT_AGENT_COUNT,
    seed: int = 42,
    adapter_kwargs: Dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    if generation_regime not in {"parallel_exploration", "progressive_refinement"}:
        raise ValueError("generation_regime must be one of {'parallel_exploration', 'progressive_refinement'}")

    _debug_log("[ENTRY] entered build_frozen_candidate_pack()")
    _debug_log(
        f"[PACK INIT] dataset={dataset_name} regime={generation_regime} split={split} "
        f"question_count={len(question_indices)} agent_count={agent_count}"
    )
    random.seed(seed)
    _debug_log("[PACK INIT] pack builder created")
    adapter = create_dataset_adapter(dataset_name, **dict(adapter_kwargs or {}))
    _debug_log("[PACK INIT] runner created")
    records = load_dataset_records(
        question_indices,
        dataset_name=dataset_name,
        split=split,
        adapter_kwargs=adapter_kwargs,
    )
    resolved_agent_name, resolved_agent_role = _resolved_agent_defaults(
        dataset_name=dataset_name,
        agent_name=agent_name,
        agent_role=agent_role,
    )
    _debug_log(
        f"[MODEL INIT] before agent/model init agent_name={resolved_agent_name} role={resolved_agent_role}"
    )
    agents = [
        PaperCandidateGenerator(
            agent_id=str(agent_id),
            role=resolved_agent_role,
            domain=adapter.domain,
            llm_name=llm_name,
            temperature=agent_temperature,
        )
        for agent_id in range(int(agent_count))
    ]
    if agents:
        llm = agents[0].llm
        model = getattr(llm, "model", None)
        tokenizer = getattr(llm, "tokenizer", None)
        _debug_log(f"[MODEL INIT] tokenizer loaded={tokenizer is not None}")
        _debug_log(f"[MODEL INIT] model loaded={model is not None}")
        if model is not None:
            try:
                first_param = next(model.parameters())
                _debug_log(
                    f"[MODEL INIT] dtype={first_param.dtype} device={first_param.device} "
                    f"bf16_dtype={first_param.dtype == torch.bfloat16}"
                )
            except Exception as exc:
                _debug_log(f"[MODEL INIT] unable to inspect model dtype/device error={type(exc).__name__}")
    _debug_log("[PACK INIT] runner/agent objects created")

    cases: List[Dict[str, Any]] = []
    for case_idx, record in enumerate(tqdm(records, desc=f"build_pack:{generation_regime}", unit="question")):
        _debug_log(f"[MULTI-SAMPLE] before sample {case_idx} record_index={record['record_index']}")
        _debug_log(f"[MULTI-SAMPLE] sample {case_idx} prompt built chars={len(record['question_text'])}")
        if case_idx == 0:
            _debug_log(f"[FIRST SAMPLE] before sample 0 record_index={record['record_index']}")
            _debug_log(f"[FIRST SAMPLE] sample 0 prompt built chars={len(record['question_text'])}")
        if generation_regime == "parallel_exploration":
            candidates = await _generate_parallel_case(
                agents,
                record["question_text"],
                dataset_name=dataset_name,
                sample_idx=case_idx,
            )
        else:
            candidates = await _generate_progressive_case(
                agents,
                record["question_text"],
                dataset_name=dataset_name,
            )
        _debug_log(
            f"[MULTI-SAMPLE] sample {case_idx} candidate generation returned "
            f"candidate_count={len(candidates)}"
        )
        if case_idx == 0:
            _debug_log("[FIRST SAMPLE] candidate generation returned for sample 0")
            _debug_log(
                f"[FIRST SAMPLE] candidate text extracted count={len(candidates)} "
                f"first_chars={len(candidates[0].text) if candidates else 0}"
            )

        frozen_record = FrozenCandidateRecord(
            record_index=record["record_index"],
            task_id=record["task_id"],
            question_text=record["question_text"],
            gold_answer=record["gold_answer"],
            generation_regime=generation_regime,
            candidates=list(candidates),
            permutation=[candidate.agent_id for candidate in candidates],
            dataset_name=record.get("dataset_name", dataset_name),
            dataset_config=record.get("dataset_config", str(adapter.config.get("config_name") or "")),
            reference_reasoning=str(record.get("reference_reasoning") or ""),
            target_payload=dict(record.get("target_payload") or {}),
        )
        cases.append(frozen_record_to_dict(frozen_record))
        _debug_log(f"[MULTI-SAMPLE] frozen record assembled for sample {case_idx}")
        if case_idx == 0:
            _debug_log("[FIRST SAMPLE] frozen record assembled for sample 0")

    pack = {
        "pack_name": f"{dataset_name}_paper_frozen_candidates_v1",
        "dataset": dataset_name,
        "split": split,
        "dataset_config": str(adapter.config.get("config_name") or ""),
        "candidate_build_mode": "dense_prefill_only",
        "llm_name": llm_name,
        "llm_label": _safe_llm_label(llm_name),
        "generation_regime": generation_regime,
        "execution_side_reuse": False,
        "candidate_count": int(agent_count),
        "agent_name": resolved_agent_name,
        "agent_role": resolved_agent_role,
        "agent_temperature": agent_temperature,
        "seed": seed,
        "question_indices": [int(item) for item in question_indices],
        "build_timestamp": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
        "cases": cases,
    }
    if output_path is not None:
        _debug_log("[PACK WRITE] before pack hash")
        _debug_log("[PACK WRITE] before json dump")
        _json_dump(Path(output_path).expanduser(), pack)
        _debug_log(f"[PACK WRITE] pack written successfully path={output_path}")
    _debug_log("[EXIT] clean exit from build_frozen_candidate_pack()")
    return pack


def load_frozen_candidate_pack(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
