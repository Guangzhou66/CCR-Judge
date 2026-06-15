from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from KVCOMM.utils.metrics import GenerationResult
from dataset_adapters import create_dataset_adapter
from paper_repair.methods.base import (
    PaperProtocolFinalSelectBest,
    candidate_views_from_spatial_info,
    create_protocol_judge,
    run_standard_judge_case,
)
from paper_repair.repair.ccr_fallback import ccr_primary_selection_is_usable
from paper_repair.repair.ccr_context import (
    build_ccr_context,
    build_ccr_context_payload,
)
from paper_repair.repair.ccr_shortlist import select_ccr_shortlist
from paper_repair.repair.ccr_state import build_ccr_state
from paper_repair.repair.ccr_types import (
    CCRCandidateView as CandidateView,
    CCRContextPayload,
    FrozenCandidateRecord,
    JudgeMethodResult,
)


class CCRJudgeFinalSelect(PaperProtocolFinalSelectBest):
    """Judge-side, interaction-aware decision-context repair for final selection.

    The method estimates structured candidate-slate interactions, constructs a
    comparative decision-context payload, serializes that payload into
    judge-consumable context text, and then performs final candidate selection.
    """

    def _candidate_outputs(self, spatial_info: Dict[str, Any]) -> list[CandidateView]:
        return candidate_views_from_spatial_info(
            spatial_info,
            self._get_agent_order(spatial_info),
            dataset_name=str(getattr(self, "domain", "mmlu") or "mmlu"),
        )

    def _compose_result_text(
        self,
        *,
        selected_agent_id: str,
        selected_choice: str | None,
        raw_text: str,
    ) -> str:
        cleaned = self._strip_selected_agent_header(raw_text)
        choice_line = selected_choice or "A"
        cleaned_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if cleaned_lines and cleaned_lines[0].strip().upper() == choice_line:
            return f"Selected agent id: {selected_agent_id}\n{cleaned}".strip()
        return f"Selected agent id: {selected_agent_id}\n{choice_line}\n{cleaned}".strip()

    def _tokenize_candidate_output(self, text: str) -> Dict[str, torch.Tensor]:
        token_ids = self.llm.tokenizer(str(text), add_special_tokens=False)["input_ids"]
        if not token_ids:
            token_ids = self.llm.tokenizer("\n", add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.llm.model.device,
        ).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=self.llm.model.device),
        }

    def _materialize_candidate_response_caches(
        self,
        *,
        candidate_outputs: list[CandidateView],
        message: str,
    ) -> None:
        for candidate in candidate_outputs:
            owner_memory = self.llm._ensure_agent_memory(candidate.agent_id)
            existing = owner_memory.setdefault("response", {}).get(message)
            if existing:
                continue
            token_inputs = self._tokenize_candidate_output(candidate.text or "")
            with torch.no_grad():
                output = self.llm.model.generate(
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

    def _build_decision_context_primary_path(
        self,
        *,
        question: str,
        candidate_outputs: list[CandidateView],
    ) -> Dict[str, Any]:
        interaction_state = build_ccr_state(candidate_outputs)
        shortlist_payload = select_ccr_shortlist(
            candidate_outputs,
            interaction_state=interaction_state,
            max_candidates=min(3, len(candidate_outputs)),
        )
        shortlisted_candidate_ids = list(shortlist_payload.get("shortlisted_candidate_ids") or [])
        context_payload = build_ccr_context_payload(
            question=question,
            candidate_outputs=candidate_outputs,
            interaction_state=interaction_state,
            shortlisted_candidate_ids=shortlisted_candidate_ids,
        )
        context_bundle = build_ccr_context(
            question=question,
            candidate_outputs=candidate_outputs,
            interaction_state=interaction_state,
            shortlisted_candidate_ids=shortlisted_candidate_ids,
        )
        return {
            "interaction_state": interaction_state,
            "shortlist_payload": shortlist_payload,
            "ccr_context_payload": context_payload,
            "ccr_context_text": context_bundle.ccr_context_text,
            "context_bundle": context_bundle,
            "judge_context_text": context_bundle.judge_context_text,
        }

    async def _run_ccr_final_select(
        self,
        *,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str,
        **kwargs,
    ) -> tuple[Any | None, GenerationResult]:
        payload = await super()._async_execute(
            input,
            spatial_info,
            temporal_info,
            mode=mode,
            **kwargs,
        )
        if mode == "allow_kv_reuse":
            message, result = payload
            return message, result
        return None, payload

    def _finalize_result(
        self,
        *,
        message: Any | None,
        source_result: GenerationResult,
        candidate_outputs: list[CandidateView],
        selected_agent_id: str | None,
        repair_metadata: Dict[str, Any],
    ):
        adapter = create_dataset_adapter(str(getattr(self, "domain", "mmlu") or "mmlu"))
        selected_choice = adapter.selected_answer_for_agent(candidate_outputs, selected_agent_id)
        if selected_agent_id is not None and selected_choice is not None:
            final_text = self._compose_result_text(
                selected_agent_id=selected_agent_id,
                selected_choice=selected_choice,
                raw_text=source_result.text,
            )
        else:
            final_text = source_result.text

        final_result = GenerationResult(
            text=final_text,
            mode=source_result.mode,
            ttft=source_result.ttft,
            raw_output=source_result.raw_output,
            metadata={
                **(source_result.metadata or {}),
                **repair_metadata,
            },
        )
        if message is not None:
            return message, final_result
        return final_result

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ):
        candidate_outputs = self._candidate_outputs(spatial_info)
        if not candidate_outputs:
            return await super()._async_execute(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )

        original_input = dict(input)
        original_question = str(original_input.get("task", ""))
        ccr_mode = "comparative_context_restoration"
        used_base_winner_in_primary_path = False

        interaction_state = None
        shortlist_payload = None
        ccr_context_payload: CCRContextPayload | None = None
        ccr_context_text = None
        context_bundle: CCRContext | None = None
        primary_selected_agent_id = None
        base_selected_agent_id = None
        interaction_repair_fallback_reason = None

        try:
            primary_path = self._build_decision_context_primary_path(
                question=original_question,
                candidate_outputs=candidate_outputs,
            )
            interaction_state = primary_path["interaction_state"]
            shortlist_payload = dict(primary_path["shortlist_payload"])
            ccr_context_payload = primary_path["ccr_context_payload"]
            ccr_context_text = primary_path["ccr_context_text"]
            context_bundle = primary_path["context_bundle"]
            decision_context_input = dict(original_input)
            decision_context_input["task"] = str(primary_path["judge_context_text"])

            if mode == "allow_kv_reuse":
                self._materialize_candidate_response_caches(
                    candidate_outputs=candidate_outputs,
                    message=decision_context_input["task"],
                )

            message, repaired_result = await self._run_ccr_final_select(
                input=decision_context_input,
                spatial_info=spatial_info,
                temporal_info=temporal_info,
                mode=mode,
                **kwargs,
            )
            repaired_metadata = dict(repaired_result.metadata or {})
            selected_agent_id = repaired_metadata.get("selected_agent_id")
            primary_selected_agent_id = str(selected_agent_id) if selected_agent_id is not None else None

            if ccr_primary_selection_is_usable(candidate_outputs, primary_selected_agent_id):
                repair_metadata = {
                    "selected_agent_id": primary_selected_agent_id,
                    "ccr_method": "comparative_context_restoration",
                    "ccr_mode": ccr_mode,
                    "ccr_state_summary": interaction_state.to_dict().get(
                        "ccr_state_summary"
                    ),
                    "ccr_shortlist_candidate_ids": shortlist_payload.get("shortlisted_candidate_ids") or [],
                    "ccr_context_summary": context_bundle.ccr_context_summary if context_bundle is not None else None,
                    "ccr_selected_agent_id": primary_selected_agent_id,
                    "base_selected_agent_id": None,
                    "ccr_used_base_winner_in_primary_path": used_base_winner_in_primary_path,
                    "ccr_single_pass_success": True,
                    "ccr_fallback_used": False,
                    "ccr_fallback_reason": None,
                    "ccr_extra_judge_calls": 0,
                    "ccr_state": interaction_state.to_dict(),
                    "ccr_shortlist_payload": shortlist_payload,
                    "ccr_context_payload": (
                        ccr_context_payload.to_dict() if ccr_context_payload is not None else None
                    ),
                    "ccr_context_text": ccr_context_text,
                    "judge_context_text": context_bundle.judge_context_text if context_bundle is not None else None,
                }
                return self._finalize_result(
                    message=message,
                    source_result=repaired_result,
                    candidate_outputs=candidate_outputs,
                    selected_agent_id=primary_selected_agent_id,
                    repair_metadata=repair_metadata,
                )

            interaction_repair_fallback_reason = "ccr_repaired_parse_failure"
        except Exception as exc:
            interaction_repair_fallback_reason = f"ccr_context_or_primary_select_failed:{type(exc).__name__}"

        fallback_message, fallback_result = await self._run_ccr_final_select(
            input=original_input,
            spatial_info=spatial_info,
            temporal_info=temporal_info,
            mode=mode,
            **kwargs,
        )
        fallback_metadata = dict(fallback_result.metadata or {})
        fallback_selected = fallback_metadata.get("selected_agent_id")
        base_selected_agent_id = str(fallback_selected) if fallback_selected is not None else None
        repair_metadata = {
            "selected_agent_id": base_selected_agent_id,
            "ccr_method": "comparative_context_restoration",
            "ccr_mode": ccr_mode,
            "ccr_state_summary": (
                interaction_state.to_dict().get("ccr_state_summary")
                if interaction_state is not None
                else None
            ),
            "ccr_shortlist_candidate_ids": (
                shortlist_payload.get("shortlisted_candidate_ids") if shortlist_payload is not None else []
            ),
            "ccr_context_summary": (
                context_bundle.ccr_context_summary if context_bundle is not None else None
            ),
            "ccr_selected_agent_id": primary_selected_agent_id,
            "base_selected_agent_id": base_selected_agent_id,
            "ccr_used_base_winner_in_primary_path": used_base_winner_in_primary_path,
            "ccr_single_pass_success": False,
            "ccr_fallback_used": True,
            "ccr_fallback_reason": interaction_repair_fallback_reason,
            "ccr_extra_judge_calls": 1,
            "ccr_state": interaction_state.to_dict() if interaction_state is not None else None,
            "ccr_shortlist_payload": shortlist_payload,
            "ccr_context_payload": ccr_context_payload.to_dict() if ccr_context_payload is not None else None,
            "ccr_context_text": ccr_context_text,
            "judge_context_text": context_bundle.judge_context_text if context_bundle is not None else None,
        }
        return self._finalize_result(
            message=fallback_message,
            source_result=fallback_result,
            candidate_outputs=candidate_outputs,
            selected_agent_id=base_selected_agent_id,
            repair_metadata=repair_metadata,
        )


def create_ccr_judge(llm_name: str, *, domain: str = "mmlu") -> CCRJudgeFinalSelect:
    return create_protocol_judge(
        judge_cls=CCRJudgeFinalSelect,
        judge_id="paper_judge_ccr",
        llm_name=llm_name,
        domain=domain,
    )


async def run_ccr_judge_case(
    *,
    judge,
    record: FrozenCandidateRecord,
    output_dir: Path,
) -> JudgeMethodResult:
    return await run_standard_judge_case(
        judge=judge,
        method_name="ccr_judge",
        record=record,
        output_dir=output_dir,
        mode="allow_kv_reuse",
    )


PaperRepairFinalSelectBest = CCRJudgeFinalSelect
create_interaction_repair_judge = create_ccr_judge
run_interaction_repair_judge_case = run_ccr_judge_case
