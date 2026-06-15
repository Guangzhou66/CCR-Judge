from __future__ import annotations

"""Ori-style wrapper for judge-side, interaction-aware decision-context repair."""

import asyncio
from collections import Counter
from typing import Any, Dict, List

from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.agents.ori_protocol_dense_compare import (
    attach_ori_protocol_metadata,
    canonicalize_selected_result_text,
    run_ori_structured_dense_compare,
    selected_choice_from_spatial_info,
)
from ablation_study.ccr_ablation_variants import (
    AblationVariant,
    CCRAblationJudgeFinalSelect,
    random_shortlist_variant,
)
from paper_repair.eval.parsing import normalize_selected_agent_id
from paper_repair.methods.ccr_judge import CCRJudgeFinalSelect


def _candidate_answers(
    *,
    candidate_agent_ids: List[str],
    spatial_info: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, str | None]:
    return {
        str(agent_id): selected_choice_from_spatial_info(
            spatial_info=spatial_info,
            selected_agent_id=str(agent_id),
            dataset_name=dataset_name,
        )
        for agent_id in candidate_agent_ids
    }


def _ccr_canonical_tiebreak_payload(
    *,
    metadata: Dict[str, Any],
    candidate_agent_ids: List[str],
    spatial_info: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    answers_by_agent = _candidate_answers(
        candidate_agent_ids=candidate_agent_ids,
        spatial_info=spatial_info,
        dataset_name=dataset_name,
    )
    valid_answers = {
        agent_id: answer
        for agent_id, answer in answers_by_agent.items()
        if answer is not None
    }
    answer_counts = Counter(valid_answers.values())
    ccr_state_summary = metadata.get("ccr_state_summary") or {}
    shortlist_candidate_ids = [
        str(item)
        for item in (metadata.get("ccr_shortlist_candidate_ids") or [])
        if str(item) in valid_answers
    ]
    shortlist_answers = {
        valid_answers[str(agent_id)]
        for agent_id in shortlist_candidate_ids
        if str(agent_id) in valid_answers
    }
    split_severity_label = str(ccr_state_summary.get("split_severity_label") or "").strip().lower()
    anchor_answer = ccr_state_summary.get("internal_consensus_anchor_answer")
    winning_answer = None if anchor_answer is None else str(anchor_answer).strip()
    if winning_answer is None and answer_counts:
        winning_answer = answer_counts.most_common(1)[0][0]
    cluster_size = int(answer_counts.get(winning_answer, 0)) if winning_answer is not None else 0
    all_same_answer = bool(valid_answers) and len(set(valid_answers.values())) == 1
    shortlist_same_answer = bool(shortlist_candidate_ids) and len(shortlist_answers) == 1
    top_support = answer_counts.get(winning_answer, 0) if winning_answer is not None else 0
    second_support = 0
    if len(answer_counts) >= 2:
        second_support = answer_counts.most_common(2)[1][1]

    reason = None
    if winning_answer is None or cluster_size <= 0:
        reason = None
    elif split_severity_label == "trivial":
        reason = "split_severity_trivial"
    elif all_same_answer:
        reason = "all_viable_candidates_same_answer"
    elif shortlist_same_answer and next(iter(shortlist_answers)) == winning_answer and top_support > second_support:
        reason = "dominant_cluster_shortlist_same_answer"

    representative_agent_id = None
    if reason is not None:
        for agent_id in candidate_agent_ids:
            if valid_answers.get(str(agent_id)) == winning_answer:
                representative_agent_id = str(agent_id)
                break

    current_selected_agent_id = normalize_selected_agent_id(metadata.get("selected_agent_id"), allowed_ids=candidate_agent_ids)
    applied = bool(
        representative_agent_id is not None
        and current_selected_agent_id is not None
        and representative_agent_id != current_selected_agent_id
    )
    return {
        "ccr_canonical_tiebreak_applied": applied,
        "ccr_canonical_tiebreak_reason": reason,
        "ccr_canonical_cluster_answer": winning_answer,
        "ccr_canonical_cluster_size": cluster_size if reason is not None else 0,
        "ccr_canonical_representative_agent_id": representative_agent_id,
        "ccr_precanonical_selected_agent_id": current_selected_agent_id,
    }


@AgentRegistry.register("OriProtocolTest3RepairFinalSelectBest")
class OriProtocolTest3RepairFinalSelectBest(CCRJudgeFinalSelect):
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
        candidate_agent_ids = super()._get_agent_order(spatial_info) if spatial_info else []
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
        selected_agent_id = None
        if isinstance(result, tuple) and len(result) == 2:
            maybe_generation = result[1]
        else:
            maybe_generation = result
        if hasattr(maybe_generation, "metadata") and isinstance(maybe_generation.metadata, dict):
            metadata = dict(maybe_generation.metadata or {})
            canonical_payload = _ccr_canonical_tiebreak_payload(
                metadata=metadata,
                candidate_agent_ids=[str(item) for item in candidate_agent_ids],
                spatial_info=spatial_info,
                dataset_name=str(getattr(self, "domain", "mmlu") or "mmlu"),
            )
            metadata.update(canonical_payload)
            if canonical_payload["ccr_canonical_tiebreak_applied"]:
                selected_agent_id = canonical_payload["ccr_canonical_representative_agent_id"]
                metadata["selected_agent_id"] = selected_agent_id
                try:
                    metadata["selected_index"] = [str(item) for item in candidate_agent_ids].index(str(selected_agent_id))
                except ValueError:
                    metadata["selected_index"] = None
            else:
                selected_agent_id = metadata.get("selected_agent_id")

            maybe_generation = maybe_generation.__class__(
                text=maybe_generation.text,
                mode=maybe_generation.mode,
                ttft=maybe_generation.ttft,
                raw_output=maybe_generation.raw_output,
                metadata=metadata,
            )
            if isinstance(result, tuple) and len(result) == 2:
                result = (result[0], maybe_generation)
            else:
                result = maybe_generation

        dense_compare_result = None
        if dense_compare_requested:
            dense_compare_result = await run_ori_structured_dense_compare(
                judge=self,
                question=str(input.get("task") or ""),
                candidate_agent_ids=candidate_agent_ids,
                spatial_info=spatial_info,
            )
        if selected_agent_id is None and hasattr(maybe_generation, "metadata") and isinstance(maybe_generation.metadata, dict):
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

    async def async_execute(self, input: Any, mode: str = "default", **kwargs):
        original_task = None
        if isinstance(input, dict):
            original_task = input.get("task")
        result = await super().async_execute(input, mode=mode, **kwargs)
        if mode == "allow_kv_reuse" and isinstance(original_task, str):
            outputs = getattr(self, "outputs", None)
            outputs_meta = getattr(self, "outputs_meta", None)
            if isinstance(outputs, dict) and original_task not in outputs and outputs:
                last_key = next(reversed(outputs))
                outputs[original_task] = list(outputs[last_key])
                if isinstance(outputs_meta, dict) and last_key in outputs_meta:
                    outputs_meta[original_task] = list(outputs_meta[last_key])
        return result


@AgentRegistry.register("OriProtocolCCRJudgeFinalSelect")
class OriProtocolCCRJudgeFinalSelect(OriProtocolTest3RepairFinalSelectBest):
    pass


class OriProtocolCCRAblationFinalSelect(
    OriProtocolTest3RepairFinalSelectBest,
    CCRAblationJudgeFinalSelect,
):
    """Ori-protocol CCR-Judge wrapper with an ablation-specific context builder."""

    ablation_variant: AblationVariant

    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config=None,
    ):
        CCRJudgeFinalSelect.__init__(self, id=id, domain=domain, llm_name=llm_name, llm_config=llm_config)
        self.ablation_variant = self.__class__.ablation_variant
        self._ori_protocol_forced_candidate_order = None
        self._last_ablation_metadata: Dict[str, Any] = {}


def _make_ori_ablation_class(
    *,
    registry_name: str,
    variant: AblationVariant,
):
    class _VariantOriProtocolCCRAblationFinalSelect(OriProtocolCCRAblationFinalSelect):
        ablation_variant = variant

    _VariantOriProtocolCCRAblationFinalSelect.__name__ = registry_name
    _VariantOriProtocolCCRAblationFinalSelect.__qualname__ = registry_name
    _VariantOriProtocolCCRAblationFinalSelect.__doc__ = (
        f"Ori-protocol CCR-Judge ablation variant: {variant.label}."
    )
    AgentRegistry.register(registry_name)(_VariantOriProtocolCCRAblationFinalSelect)
    return _VariantOriProtocolCCRAblationFinalSelect


_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeAblationFullFinalSelect",
    variant=AblationVariant(
        slug="ccr_full",
        label="CCR-Judge (Full)",
        family="full",
        description="Full CCR-Judge decision-context repair.",
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeNoComparativeGroupingFinalSelect",
    variant=AblationVariant(
        slug="ccr_wo_comparative_grouping",
        label="CCR-Judge w/o Comparative Grouping",
        family="no_grouping",
        description="Removes normalized-answer grouping, group-level statistics, and split-severity structure.",
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeNoSupportScoringFinalSelect",
    variant=AblationVariant(
        slug="ccr_wo_support_scoring",
        label="CCR-Judge w/o Support Scoring",
        family="no_support_scoring",
        description="Keeps answer grouping but removes support-score based shortlist ranking.",
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeNoShortlistCompressionFinalSelect",
    variant=AblationVariant(
        slug="ccr_wo_shortlist_compression",
        label="CCR-Judge w/o Shortlist Compression",
        family="no_shortlist_compression",
        description="Keeps comparative payload statistics but includes every candidate in the shortlist field.",
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeStatsOnlyFinalSelect",
    variant=AblationVariant(
        slug="ccr_stats_only",
        label="CCR-Judge Stats Only",
        family="stats_only",
        description="Keeps answer-group statistics, support, split severity, and risk summary while suppressing shortlist evidence.",
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeShortlistOnlyFinalSelect",
    variant=AblationVariant(
        slug="ccr_shortlist_only",
        label="CCR-Judge Shortlist Only",
        family="shortlist_only",
        description="Keeps deterministic shortlist evidence while suppressing group-level statistics and risk summary.",
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeNoComparativePayloadStatsFinalSelect",
    variant=AblationVariant(
        slug="ccr_wo_comparative_payload_statistics",
        label="CCR-Judge w/o Comparative Payload Statistics",
        family="no_payload_statistics",
        description="Keeps deterministic shortlist selection but suppresses group-level serialized statistics.",
        optional=True,
    ),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeRandomShortlistSeed42FinalSelect",
    variant=random_shortlist_variant(42),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeRandomShortlistSeed43FinalSelect",
    variant=random_shortlist_variant(43),
)
_make_ori_ablation_class(
    registry_name="OriProtocolCCRJudgeRandomShortlistSeed44FinalSelect",
    variant=random_shortlist_variant(44),
)
