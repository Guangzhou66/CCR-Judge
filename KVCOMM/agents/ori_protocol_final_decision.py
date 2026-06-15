from __future__ import annotations

"""Ori-style baseline final selection wrapper with structured dense-compare attach."""

from typing import Any, Dict, List

from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.agents.final_decision import FinalSelectBest as _FinalSelectBest
from KVCOMM.agents.ori_protocol_dense_compare import (
    attach_ori_protocol_metadata,
    canonicalize_selected_result_text,
    run_ori_structured_dense_compare,
    selected_choice_from_spatial_info,
)


@AgentRegistry.register("OriProtocolFinalSelectBest")
class OriProtocolFinalSelectBest(_FinalSelectBest):
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config=None,
    ):
        super().__init__(id=id, domain=domain, llm_name=llm_name, llm_config=llm_config)
        self._ori_protocol_forced_candidate_order: List[str] | None = None

    def _get_agent_order(self, spatial_info: Dict[str, Any]) -> List[str]:
        forced = self._ori_protocol_forced_candidate_order
        if forced:
            forced_ids = [str(item) for item in forced]
            if all(agent_id in spatial_info for agent_id in forced_ids):
                return forced_ids
        return super()._get_agent_order(spatial_info)

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ):
        candidate_agent_ids = _FinalSelectBest._get_agent_order(self, spatial_info) if spatial_info else []
        dense_compare_requested = bool(kwargs.get("judge_compare_dense"))
        previous_order = self._ori_protocol_forced_candidate_order
        self._ori_protocol_forced_candidate_order = list(candidate_agent_ids)
        try:
            super_kwargs = dict(kwargs)
            if dense_compare_requested:
                super_kwargs["judge_compare_dense"] = False
            result = await super()._async_execute(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **super_kwargs,
            )
        finally:
            self._ori_protocol_forced_candidate_order = previous_order
        dense_compare_result = None
        if dense_compare_requested:
            dense_compare_result = await run_ori_structured_dense_compare(
                judge=self,
                question=str(input.get("task") or ""),
                candidate_agent_ids=candidate_agent_ids,
                spatial_info=spatial_info,
            )
        selected_agent_id = None
        if isinstance(result, tuple) and len(result) == 2:
            maybe_generation = result[1]
        else:
            maybe_generation = result
        if hasattr(maybe_generation, "metadata") and isinstance(maybe_generation.metadata, dict):
            selected_agent_id = maybe_generation.metadata.get("selected_agent_id")
        selected_choice = selected_choice_from_spatial_info(
            spatial_info=spatial_info,
            selected_agent_id=selected_agent_id,
            dataset_name=str(getattr(self, "domain", "mmlu") or "mmlu"),
        )
        result = canonicalize_selected_result_text(
            result,
            selected_agent_id=selected_agent_id,
            selected_choice=selected_choice,
            dataset_name=str(getattr(self, "domain", "mmlu") or "mmlu"),
        )
        return attach_ori_protocol_metadata(
            result,
            candidate_agent_ids=candidate_agent_ids,
            dense_compare_result=dense_compare_result,
        )
