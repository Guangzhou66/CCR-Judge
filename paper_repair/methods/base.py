from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from KVCOMM.agents.final_decision import FinalSelectBest as BaseFinalSelectBest
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.metrics import GenerationResult
from dataset_adapters import create_dataset_adapter
from paper_repair.protocol.candidate_generator import ensure_prompt_set_loaded
from paper_repair.repair.ccr_types import CCRCandidateView as CandidateView, FrozenCandidateRecord, JudgeMethodResult


def build_spatial_info(record: FrozenCandidateRecord) -> Dict[str, Any]:
    return {
        candidate.agent_id: {
            "role": candidate.role,
            "output": candidate.text,
            "dataset_name": record.dataset_name,
            "dataset_config": record.dataset_config,
            "metadata": {
                **dict(candidate.metadata),
                "normalized_answer": candidate.normalized_answer,
                "conclusion": candidate.conclusion,
                "evidence": candidate.evidence,
                "strongest_alternative_considered": candidate.strongest_alternative_considered,
                "strongest_alternative_option": candidate.strongest_alternative_option,
                "rejected_option": candidate.rejected_option,
                "why_not_that_option": candidate.why_not_that_option,
                "why_not_rejected_option": candidate.why_not_rejected_option,
                "alternative_overturn_flag": candidate.alternative_overturn_flag,
                "alternative_overturn_yes": candidate.alternative_overturn_yes,
                "alternative_supported_flag": candidate.alternative_supported_flag,
                "alternative_supported_yes": candidate.alternative_supported_yes,
            },
        }
        for candidate in record.candidates
    }


def candidate_views_from_spatial_info(
    spatial_info: Dict[str, Any],
    ordered_agent_ids: Sequence[str],
    *,
    dataset_name: str = "mmlu",
) -> list[CandidateView]:
    views: list[CandidateView] = []
    for agent_id in ordered_agent_ids:
        info = spatial_info[str(agent_id)]
        dataset_config = str(info.get("dataset_config") or "") if isinstance(info, dict) else ""
        adapter_kwargs = {"config_name": dataset_config} if dataset_config else {}
        adapter = create_dataset_adapter(dataset_name, **adapter_kwargs)
        parsed = adapter.parse_candidate_output(
            info.get("output") if isinstance(info, dict) else "",
            dict(info.get("metadata") or {}) if isinstance(info, dict) else {},
        )
        metadata = info.get("metadata") if isinstance(info, dict) else {}
        if isinstance(metadata, dict):
            for key in (
                "normalized_answer",
                "conclusion",
                "evidence",
                "strongest_alternative_considered",
                "strongest_alternative_option",
                "rejected_option",
                "why_not_that_option",
                "why_not_rejected_option",
                "alternative_overturn_flag",
                "alternative_overturn_yes",
                "alternative_supported_flag",
                "alternative_supported_yes",
            ):
                if key == "normalized_answer":
                    continue
                if metadata.get(key) is not None:
                    parsed[key] = metadata.get(key)
        views.append(
            CandidateView(
                agent_id=str(agent_id),
                role=str(info.get("role") or "MMLU Solver"),
                text=str(info.get("output") or ""),
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
                metadata=dict(metadata or {}),
            )
        )
    return views


class PaperProtocolFinalSelectBest(BaseFinalSelectBest):
    def __init__(
        self,
        id: str | None = None,
        domain: str = "mmlu",
        llm_name: str = "",
        llm_config=None,
    ):
        super().__init__(id=id, domain=domain, llm_name=llm_name, llm_config=llm_config)
        self._paper_candidate_order: list[str] | None = None

    def _get_agent_order(self, spatial_info: Dict[str, Any]) -> list[str]:
        if self._paper_candidate_order:
            forced = [str(item) for item in self._paper_candidate_order]
            if all(agent_id in spatial_info for agent_id in forced):
                return forced
        return super()._get_agent_order(spatial_info)

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ):
        previous_order = self._paper_candidate_order
        forced_order = kwargs.get("candidate_agent_ids") or input.get("_paper_candidate_order")
        if forced_order:
            self._paper_candidate_order = [str(item) for item in forced_order]
        try:
            return await super()._async_execute(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
        finally:
            self._paper_candidate_order = previous_order


def create_protocol_judge(
    *,
    judge_cls: type[PaperProtocolFinalSelectBest] = PaperProtocolFinalSelectBest,
    judge_id: str,
    llm_name: str,
    domain: str = "mmlu",
    judge_temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> PaperProtocolFinalSelectBest:
    ensure_prompt_set_loaded(domain)
    judge = judge_cls(
        id=judge_id,
        domain=domain,
        llm_name=llm_name,
        llm_config=KVCommConfig.from_env(),
    )
    judge.llm.DEFAULT_TEMPERATURE = float(judge_temperature)
    judge.llm.DEFAULT_MAX_TOKENS = int(max_new_tokens)
    return judge


def _tokenize_candidate_output(judge: Any, text: str) -> Dict[str, torch.Tensor]:
    token_ids = judge.llm.tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if not token_ids:
        token_ids = judge.llm.tokenizer("\n", add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=judge.llm.model.device).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=judge.llm.model.device),
    }


def materialize_frozen_candidate_response_caches(
    *,
    judge: Any,
    record: FrozenCandidateRecord,
) -> None:
    message = str(record.question_text)
    for candidate in record.candidates:
        agent_id = candidate.agent_id
        owner_memory = judge.llm._ensure_agent_memory(agent_id)
        existing = owner_memory.setdefault("response", {}).get(message)
        if existing:
            continue
        token_inputs = _tokenize_candidate_output(judge, candidate.text)
        with torch.no_grad():
            output = judge.llm.model.generate(
                **token_inputs,
                use_cache=True,
                max_length=token_inputs["input_ids"].shape[-1] + 1,
                return_dict_in_generate=True,
                return_legacy_cache=False,
                do_sample=False,
                remove_invalid_values=True,
            )
        response_cache = output.past_key_values.copy().slice_(
            start=0,
            end=token_inputs["input_ids"].shape[-1],
        )
        owner_memory["response"][message] = [response_cache]
        owner_memory.setdefault("response_ids", {})[message] = [
            {
                "input_ids": token_inputs["input_ids"],
                "attention_mask": token_inputs["attention_mask"],
            }
        ]
        owner_memory.setdefault("response_drop_num", {})[message] = [0]


def _request_input(record: FrozenCandidateRecord) -> Dict[str, Any]:
    return {
        "task": record.question_text,
        "_paper_candidate_order": list(record.permutation),
        "_request_uid": uuid.uuid4().hex[:8],
    }


def _canonical_result_text(
    *,
    adapter: Any,
    selected_agent_id: str | None,
    selected_answer: str | None,
    raw_text: Any,
) -> str:
    if selected_agent_id is None or selected_answer is None:
        return str(raw_text or "")
    try:
        return str(
            adapter.format_selected_result_text(
                selected_agent_id=str(selected_agent_id),
                selected_answer=str(selected_answer),
            )
        )
    except Exception:
        return str(raw_text or "")


async def _ensure_allow_kv_reuse_prefix(
    *,
    judge: Any,
    input_dict: Dict[str, Any],
    spatial_info: Dict[str, Any],
    output_dir: Path,
) -> None:
    materialize_frozen_candidate_response_caches(judge=judge, record=FrozenCandidateRecord(
        record_index=0,
        question_text=str(input_dict["task"]),
        gold_answer="",
        generation_regime="",
        candidates=candidate_views_from_spatial_info(
            spatial_info,
            input_dict.get("_paper_candidate_order") or sorted(spatial_info.keys()),
            dataset_name=str(getattr(judge, "domain", "mmlu") or "mmlu"),
        ),
        permutation=list(input_dict.get("_paper_candidate_order") or []),
        dataset_name=str(getattr(judge, "domain", "mmlu") or "mmlu"),
    ))
    has_prefix_initialized = False
    try:
        has_prefix_initialized = bool(judge.llm.has_prefix_initialized(judge.id))
    except Exception:
        has_prefix_initialized = False
    if not has_prefix_initialized:
        await judge._process_inputs(
            input_dict,
            spatial_info,
            {},
            mode="allow_kv_reuse",
            output_dir=str(output_dir),
        )


async def run_standard_judge_case(
    *,
    judge: Any,
    method_name: str,
    record: FrozenCandidateRecord,
    output_dir: Path,
    mode: str,
) -> JudgeMethodResult:
    spatial_info = build_spatial_info(record)
    input_dict = _request_input(record)
    adapter = create_dataset_adapter(
        record.dataset_name,
        config_name=record.dataset_config,
    )

    if mode == "allow_kv_reuse":
        materialize_frozen_candidate_response_caches(judge=judge, record=record)
        has_prefix_initialized = False
        try:
            has_prefix_initialized = bool(judge.llm.has_prefix_initialized(judge.id))
        except Exception:
            has_prefix_initialized = False
        if not has_prefix_initialized:
            await judge._process_inputs(
                input_dict,
                spatial_info,
                {},
                mode="allow_kv_reuse",
                output_dir=str(output_dir),
            )

    result_payload = await judge._async_execute(
        input_dict,
        spatial_info,
        {},
        mode=mode,
        output_dir=str(output_dir),
    )
    if mode == "allow_kv_reuse":
        _, generation = result_payload
    else:
        generation = result_payload

    metadata = dict(generation.metadata or {})
    selected_agent_id = adapter.resolve_selected_agent_id(
        judge_text=generation.text,
        candidates=record.candidates,
        metadata_selected_agent_id=metadata.get("selected_agent_id"),
    )
    selected_answer = adapter.selected_answer_for_agent(record.candidates, selected_agent_id)
    parse_success = bool(selected_agent_id is not None)
    raw_generation_text = str(generation.text or "")
    canonical_text = _canonical_result_text(
        adapter=adapter,
        selected_agent_id=selected_agent_id,
        selected_answer=selected_answer,
        raw_text=raw_generation_text,
    )
    request_uid = input_dict.get("_request_uid")
    if mode == "allow_kv_reuse" and request_uid:
        try:
            judge.llm.finalize_request(request_uid)
        except Exception:
            pass

    return JudgeMethodResult(
        method_name=method_name,
        selected_agent_id=selected_agent_id,
        selected_answer=selected_answer,
        parse_success=parse_success,
        fallback_used=bool(
            metadata.get("ccr_fallback_used")
            if metadata.get("ccr_fallback_used") is not None
            else metadata.get("interaction_repair_fallback_used")
        ),
        fallback_reason=metadata.get("ccr_fallback_reason") or metadata.get("interaction_repair_fallback_reason"),
        raw_judge_text=canonical_text,
        reuse_mode=generation.mode,
        ttft=generation.ttft,
        metadata={
            **metadata,
            "raw_judge_text_before_canonicalization": raw_generation_text,
            "raw_generation_metadata": metadata,
        },
    )
