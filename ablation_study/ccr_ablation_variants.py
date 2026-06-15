from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from paper_repair.methods.base import create_protocol_judge, run_standard_judge_case
from paper_repair.methods.ccr_judge import CCRJudgeFinalSelect
from paper_repair.eval.parsing import normalize_choice
from paper_repair.repair.ccr_context import (
    build_ccr_context,
    build_ccr_context_payload,
    render_ccr_context_text,
)
from paper_repair.repair.ccr_shortlist import select_ccr_shortlist
from paper_repair.repair.ccr_state import build_ccr_state
from paper_repair.repair.ccr_types import (
    CCRCandidateView,
    CCRContext,
    CCRContextPayload,
    CCRState,
    FrozenCandidateRecord,
    JudgeMethodResult,
)


SHORTLIST_CAP = 3


@dataclass(frozen=True)
class AblationVariant:
    slug: str
    label: str
    family: str
    description: str
    random_seed: int | None = None
    optional: bool = False


MAIN_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        slug="ccr_full",
        label="CCR-Judge (Full)",
        family="full",
        description="Full CCR-Judge decision-context repair.",
    ),
    AblationVariant(
        slug="ccr_wo_comparative_grouping",
        label="CCR-Judge w/o Comparative Grouping",
        family="no_grouping",
        description="Removes normalized-answer grouping, group-level statistics, and split-severity structure.",
    ),
    AblationVariant(
        slug="ccr_wo_support_scoring",
        label="CCR-Judge w/o Support Scoring",
        family="no_support_scoring",
        description="Keeps answer grouping but removes support-score based shortlist ranking.",
    ),
    AblationVariant(
        slug="ccr_wo_shortlist_compression",
        label="CCR-Judge w/o Shortlist Compression",
        family="no_shortlist_compression",
        description="Keeps comparative payload statistics but includes every candidate in the shortlist field.",
    ),
    AblationVariant(
        slug="ccr_stats_only",
        label="CCR-Judge Stats Only",
        family="stats_only",
        description="Keeps answer-group statistics, support, split severity, and risk summary while suppressing shortlist evidence.",
    ),
    AblationVariant(
        slug="ccr_shortlist_only",
        label="CCR-Judge Shortlist Only",
        family="shortlist_only",
        description="Keeps deterministic shortlist evidence while suppressing group-level statistics and risk summary.",
    ),
)

OPTIONAL_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        slug="ccr_wo_comparative_payload_statistics",
        label="CCR-Judge w/o Comparative Payload Statistics",
        family="no_payload_statistics",
        description="Keeps deterministic shortlist selection but suppresses group-level serialized statistics.",
        optional=True,
    ),
)

STABILITY_VARIANT_FAMILIES = {
    "full",
    "no_grouping",
    "no_support_scoring",
    "no_shortlist_compression",
    "stats_only",
    "shortlist_only",
}


def random_shortlist_variant(seed: int) -> AblationVariant:
    return AblationVariant(
        slug=f"ccr_random_shortlist_seed{int(seed)}",
        label=f"CCR-Judge with Random Shortlist (seed={int(seed)})",
        family="random_shortlist",
        description="Uses the same shortlist size as Full CCR-Judge but samples candidates uniformly with a fixed seed.",
        random_seed=int(seed),
    )


def variants_for_layer(
    *,
    layer: str,
    random_seeds: Sequence[int],
    include_optional: bool = False,
) -> list[AblationVariant]:
    variants = list(MAIN_VARIANTS)
    if layer == "main":
        variants.extend(random_shortlist_variant(seed) for seed in random_seeds)
        if include_optional:
            variants.extend(OPTIONAL_VARIANTS)
        return variants
    if layer == "stability":
        return [variant for variant in variants if variant.family in STABILITY_VARIANT_FAMILIES]
    raise ValueError(f"Unknown ablation layer: {layer}")


def _answer_label(value: Any) -> str:
    normalized = normalize_choice(value)
    if normalized is not None:
        return normalized
    text = str(value or "").strip()
    return text if text else "UNKNOWN"


def _candidate_positions(candidate_outputs: Sequence[CCRCandidateView]) -> Dict[str, int]:
    return {str(candidate.agent_id): index for index, candidate in enumerate(candidate_outputs)}


def _candidate_for_agent(
    candidate_outputs: Sequence[CCRCandidateView],
    agent_id: Any,
) -> CCRCandidateView | None:
    for candidate in candidate_outputs:
        if str(candidate.agent_id) == str(agent_id):
            return candidate
    return None


def _shortlist_entries(
    candidate_outputs: Sequence[CCRCandidateView],
    shortlisted_candidate_ids: Sequence[str],
    *,
    include_answers: bool,
) -> list[Dict[str, Any]]:
    positions = _candidate_positions(candidate_outputs)
    entries: list[Dict[str, Any]] = []
    for agent_id in shortlisted_candidate_ids:
        candidate = _candidate_for_agent(candidate_outputs, agent_id)
        if candidate is None:
            continue
        entry = {
            "agent_id": str(agent_id),
            "slot": int(positions.get(str(agent_id), 0)),
        }
        if include_answers:
            entry["answer"] = _answer_label(candidate.normalized_answer)
        entries.append(entry)
    return entries


def _candidate_summaries(
    candidate_outputs: Sequence[CCRCandidateView],
    *,
    include_answers: bool,
) -> list[Dict[str, Any]]:
    positions = _candidate_positions(candidate_outputs)
    summaries = []
    for candidate in candidate_outputs:
        item = {
            "agent_id": str(candidate.agent_id),
            "slot": int(positions.get(str(candidate.agent_id), 0)),
            "role": candidate.role,
        }
        if include_answers:
            item["answer"] = _answer_label(candidate.normalized_answer)
        summaries.append(item)
    return summaries


def _ordered_ids(candidate_outputs: Sequence[CCRCandidateView]) -> list[str]:
    return [str(candidate.agent_id) for candidate in candidate_outputs]


def _coverage_order_shortlist(
    candidate_outputs: Sequence[CCRCandidateView],
    *,
    max_candidates: int = SHORTLIST_CAP,
) -> list[str]:
    shortlisted: list[str] = []
    seen_answers: set[str] = set()
    for candidate in candidate_outputs:
        answer = _answer_label(candidate.normalized_answer)
        if answer in seen_answers:
            continue
        shortlisted.append(str(candidate.agent_id))
        seen_answers.add(answer)
        if len(shortlisted) >= max_candidates:
            return shortlisted
    for candidate in candidate_outputs:
        agent_id = str(candidate.agent_id)
        if agent_id in shortlisted:
            continue
        shortlisted.append(agent_id)
        if len(shortlisted) >= max_candidates:
            break
    return shortlisted


def _full_shortlist(
    candidate_outputs: Sequence[CCRCandidateView],
    interaction_state: CCRState,
    *,
    max_candidates: int = SHORTLIST_CAP,
) -> Dict[str, Any]:
    return select_ccr_shortlist(
        candidate_outputs,
        interaction_state=interaction_state,
        max_candidates=min(max_candidates, len(candidate_outputs)),
    )


def _stable_random_shortlist(
    candidate_outputs: Sequence[CCRCandidateView],
    *,
    size: int,
    seed: int,
    question: str,
) -> list[str]:
    ids = _ordered_ids(candidate_outputs)
    if size >= len(ids):
        return ids
    digest = hashlib.sha256(
        (
            str(seed)
            + "\n"
            + str(question)
            + "\n"
            + "|".join(ids)
            + "\n"
            + "|".join(str(candidate.normalized_answer) for candidate in candidate_outputs)
        ).encode("utf-8")
    ).hexdigest()
    local_seed = int(digest[:16], 16)
    rng = random.Random(local_seed)
    sampled = rng.sample(ids, int(size))
    return sampled


def _no_grouping_state(candidate_outputs: Sequence[CCRCandidateView]) -> CCRState:
    positions = _candidate_positions(candidate_outputs)
    metadata = {
        "candidate_positions": positions,
        "answer_counts": {},
        "answer_to_agent_ids": {},
        "cluster_stats": {
            "unique_normalized_answer_count": None,
            "same_answer_cluster_sizes": [],
            "nontrivial_same_answer_cluster_exists": None,
            "ablation_removed": True,
        },
        "structured_evidence_counts": {},
        "ranked_answer_groups": [],
        "competing_answer_groups": [],
        "representative_by_answer": {},
        "per_candidate_signal_scores": {str(candidate.agent_id): 0 for candidate in candidate_outputs},
        "interaction_risk_summary": {
            "consequential_split_flag": None,
            "multi_cluster_competition_flag": None,
            "later_slot_under_attention_risk_flag": None,
            "support_gap": None,
            "feature_gate_v3_fragility_score": None,
            "ablation_removed": True,
        },
        "ccr_state_summary": {
            "answer_cluster_summary": "removed_by_ablation",
            "internal_consensus_anchor_answer": None,
            "internal_consensus_anchor_strength": 0,
            "split_severity_label": "not_computed",
            "competing_answer_groups": [],
            "interaction_risk_summary": {"ablation_removed": True},
        },
    }
    metadata["slate_interaction_state_summary"] = dict(metadata["ccr_state_summary"])
    return CCRState(
        answer_support_map={},
        answer_cluster_summary="removed_by_ablation",
        internal_consensus_anchor_answer=None,
        internal_consensus_anchor_strength=0,
        split_severity_label="not_computed",
        later_slot_risk_summary={"ablation_removed": True},
        metadata=metadata,
    )


def _minimal_context(
    *,
    question: str,
    candidate_outputs: Sequence[CCRCandidateView],
    shortlisted_candidate_ids: Sequence[str],
    context_label: str,
    include_answers: bool,
    include_group_statistics: bool,
    interaction_state: CCRState | None = None,
) -> CCRContext:
    shortlist_entries = _shortlist_entries(
        candidate_outputs,
        shortlisted_candidate_ids,
        include_answers=include_answers,
    )
    candidate_summaries = _candidate_summaries(
        candidate_outputs,
        include_answers=include_answers,
    )
    payload = CCRContextPayload(
        answer_clusters=(
            interaction_state.answer_cluster_summary
            if include_group_statistics and interaction_state is not None
            else "suppressed"
        ),
        internal_anchor=(
            {
                "answer": _answer_label(interaction_state.internal_consensus_anchor_answer),
                "strength": int(interaction_state.internal_consensus_anchor_strength or 0),
            }
            if include_group_statistics and interaction_state is not None
            else {"answer": "suppressed", "strength": 0}
        ),
        split_severity=(
            interaction_state.split_severity_label
            if include_group_statistics and interaction_state is not None
            else "suppressed"
        ),
        shortlist=shortlist_entries,
        later_slot_risk=(
            dict(interaction_state.later_slot_risk_summary)
            if include_group_statistics and interaction_state is not None
            else {}
        ),
        context_flags={
            "decision_context_repair": True,
            "instance_specific": True,
            "slate_conditioned": True,
            "selection_stage_only": True,
            "modifies_candidate_generation": False,
            "ablation_context_label": context_label,
            "comparative_group_statistics_visible": bool(include_group_statistics),
        },
        competing_answer_groups=(
            list(interaction_state.to_dict().get("competing_answer_groups") or [])
            if include_group_statistics and interaction_state is not None
            else []
        ),
        risk_notes=[
            "Compare the listed candidate ids carefully before selecting a winner.",
            "Select the best candidate from the full slate.",
        ],
        original_question=str(question).strip(),
        candidate_summaries=candidate_summaries,
        canonical_tiebreak_info={},
    )
    shortlist_line = "; ".join(
        (
            f"Agent {entry.get('agent_id')}"
            + (f" -> {entry.get('answer')}" if include_answers and entry.get("answer") else "")
            + f" (slot {int(entry.get('slot', 0))})"
        )
        for entry in shortlist_entries
    )
    lines = [
        "[Decision Context]",
        f"- Context type: {context_label}",
        f"- Candidate ids for focused comparison: {shortlist_line or 'none'}",
        "- Select the best candidate from the full slate.",
        "",
        "[Original Question]",
        str(question).strip(),
    ]
    context_text = "\n".join(lines[:4]).strip()
    judge_context_text = "\n".join(lines).strip()
    return CCRContext(
        ccr_context_payload=payload,
        ccr_context_text=context_text,
        judge_context_text=judge_context_text,
        ccr_context_summary={
            "split_severity_label": payload.split_severity,
            "internal_consensus_anchor_answer": payload.internal_anchor.get("answer"),
            "internal_consensus_anchor_strength": payload.internal_anchor.get("strength"),
            "competing_answer_groups": list(payload.competing_answer_groups),
            "shortlisted_candidate_ids": [str(item.get("agent_id")) for item in payload.shortlist],
            "risk_notes": list(payload.risk_notes),
            "decision_context_label": context_label,
        },
    )


def _context_from_payload(
    *,
    question: str,
    candidate_outputs: Sequence[CCRCandidateView],
    interaction_state: CCRState,
    shortlisted_candidate_ids: list[str],
) -> CCRContext:
    payload = build_ccr_context_payload(
        question=question,
        candidate_outputs=candidate_outputs,
        interaction_state=interaction_state,
        shortlisted_candidate_ids=shortlisted_candidate_ids,
    )
    context_text = render_ccr_context_text(payload)
    judge_context_text = "\n".join(
        [
            context_text,
            "",
            "[Original Question]",
            str(question).strip(),
        ]
    ).strip()
    return CCRContext(
        ccr_context_payload=payload,
        ccr_context_text=context_text,
        judge_context_text=judge_context_text,
        ccr_context_summary={
            "split_severity_label": payload.split_severity,
            "internal_consensus_anchor_answer": payload.internal_anchor.get("answer"),
            "internal_consensus_anchor_strength": payload.internal_anchor.get("strength"),
            "competing_answer_groups": list(payload.competing_answer_groups),
            "shortlisted_candidate_ids": [str(item.get("agent_id")) for item in payload.shortlist],
            "risk_notes": list(payload.risk_notes),
            "decision_context_label": "comparative_decision_context",
        },
    )


def _shortlist_answer_group_coverage(
    candidate_outputs: Sequence[CCRCandidateView],
    shortlisted_candidate_ids: Sequence[str],
) -> Dict[str, Any]:
    answer_by_agent = {
        str(candidate.agent_id): _answer_label(candidate.normalized_answer)
        for candidate in candidate_outputs
    }
    all_groups = set(answer_by_agent.values())
    shortlist_groups = {
        answer_by_agent[str(agent_id)]
        for agent_id in shortlisted_candidate_ids
        if str(agent_id) in answer_by_agent
    }
    return {
        "ccr_ablation_shortlist_answer_group_count": len(shortlist_groups),
        "ccr_ablation_total_answer_group_count": len(all_groups),
        "ccr_ablation_shortlist_answer_group_coverage": (
            len(shortlist_groups) / len(all_groups) if all_groups else None
        ),
        "ccr_ablation_shortlist_answer_groups": sorted(shortlist_groups),
        "ccr_ablation_all_answer_groups": sorted(all_groups),
    }


class CCRAblationJudgeFinalSelect(CCRJudgeFinalSelect):
    """CCR-Judge variant wrapper used only by ablation_study."""

    def __init__(
        self,
        *args: Any,
        ablation_variant: AblationVariant,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.ablation_variant = ablation_variant
        self._last_ablation_metadata: Dict[str, Any] = {}

    def _token_count(self, text: str) -> int | None:
        try:
            return len(self.llm.tokenizer(str(text), add_special_tokens=False)["input_ids"])
        except Exception:
            return None

    def _metadata_for_context(
        self,
        *,
        candidate_outputs: Sequence[CCRCandidateView],
        shortlisted_candidate_ids: Sequence[str],
        judge_context_text: str,
        ccr_context_text: str,
        shortlist_policy: str,
    ) -> Dict[str, Any]:
        coverage = _shortlist_answer_group_coverage(candidate_outputs, shortlisted_candidate_ids)
        return {
            "ccr_ablation_variant": self.ablation_variant.slug,
            "ccr_ablation_label": self.ablation_variant.label,
            "ccr_ablation_family": self.ablation_variant.family,
            "ccr_ablation_description": self.ablation_variant.description,
            "ccr_ablation_random_seed": self.ablation_variant.random_seed,
            "ccr_ablation_shortlist_policy": shortlist_policy,
            "ccr_ablation_shortlist_size": len(list(shortlisted_candidate_ids)),
            "ccr_ablation_shortlist_cap": SHORTLIST_CAP,
            "ccr_ablation_judge_context_chars": len(str(judge_context_text)),
            "ccr_ablation_context_payload_chars": len(str(ccr_context_text)),
            "ccr_ablation_judge_context_tokens": self._token_count(judge_context_text),
            "ccr_ablation_context_payload_tokens": self._token_count(ccr_context_text),
            **coverage,
        }

    def _build_decision_context_primary_path(
        self,
        *,
        question: str,
        candidate_outputs: list[CCRCandidateView],
    ) -> Dict[str, Any]:
        variant = self.ablation_variant
        self._last_ablation_metadata = {}

        if variant.family == "full":
            primary = super()._build_decision_context_primary_path(
                question=question,
                candidate_outputs=candidate_outputs,
            )
            shortlist_payload = dict(primary["shortlist_payload"])
            shortlisted_ids = list(shortlist_payload.get("shortlisted_candidate_ids") or [])
            context_bundle = primary["context_bundle"]
            self._last_ablation_metadata = self._metadata_for_context(
                candidate_outputs=candidate_outputs,
                shortlisted_candidate_ids=shortlisted_ids,
                judge_context_text=str(primary["judge_context_text"]),
                ccr_context_text=str(primary["ccr_context_text"]),
                shortlist_policy="full_ccr_support_ranked",
            )
            return primary

        if variant.family == "no_grouping":
            interaction_state = _no_grouping_state(candidate_outputs)
            shortlisted_ids = _ordered_ids(candidate_outputs)[: min(SHORTLIST_CAP, len(candidate_outputs))]
            context_bundle = _minimal_context(
                question=question,
                candidate_outputs=candidate_outputs,
                shortlisted_candidate_ids=shortlisted_ids,
                context_label="minimal_ordered_decision_context",
                include_answers=False,
                include_group_statistics=False,
                interaction_state=interaction_state,
            )
            shortlist_payload = {
                "shortlisted_candidate_ids": shortlisted_ids,
                "shortlist_scores": [
                    {"agent_id": agent_id, "position": index, "score": None}
                    for index, agent_id in enumerate(_ordered_ids(candidate_outputs))
                ],
                "ranking_policy": "candidate_order_no_grouping",
            }
            shortlist_policy = "candidate_order_no_grouping"

        else:
            interaction_state = build_ccr_state(candidate_outputs)
            full_shortlist_payload = _full_shortlist(candidate_outputs, interaction_state)
            full_shortlisted_ids = list(full_shortlist_payload.get("shortlisted_candidate_ids") or [])

            if variant.family == "no_support_scoring":
                shortlisted_ids = _coverage_order_shortlist(
                    candidate_outputs,
                    max_candidates=min(SHORTLIST_CAP, len(candidate_outputs)),
                )
                shortlist_payload = {
                    "shortlisted_candidate_ids": shortlisted_ids,
                    "shortlist_scores": [
                        {
                            "agent_id": candidate.agent_id,
                            "answer": _answer_label(candidate.normalized_answer),
                            "position": index,
                            "score": None,
                        }
                        for index, candidate in enumerate(candidate_outputs)
                    ],
                    "ranking_policy": "coverage_order_no_support_score",
                }
                context_bundle = _context_from_payload(
                    question=question,
                    candidate_outputs=candidate_outputs,
                    interaction_state=interaction_state,
                    shortlisted_candidate_ids=shortlisted_ids,
                )
                shortlist_policy = "coverage_order_no_support_score"

            elif variant.family == "no_shortlist_compression":
                shortlisted_ids = _ordered_ids(candidate_outputs)
                shortlist_payload = {
                    **dict(full_shortlist_payload),
                    "shortlisted_candidate_ids": shortlisted_ids,
                    "ranking_policy": "all_candidates_no_shortlist_compression",
                }
                context_bundle = _context_from_payload(
                    question=question,
                    candidate_outputs=candidate_outputs,
                    interaction_state=interaction_state,
                    shortlisted_candidate_ids=shortlisted_ids,
                )
                shortlist_policy = "all_candidates_no_shortlist_compression"

            elif variant.family == "stats_only":
                shortlisted_ids = []
                shortlist_payload = {
                    **dict(full_shortlist_payload),
                    "shortlisted_candidate_ids": shortlisted_ids,
                    "ranking_policy": "stats_only_no_shortlist_evidence",
                }
                context_bundle = _minimal_context(
                    question=question,
                    candidate_outputs=candidate_outputs,
                    shortlisted_candidate_ids=shortlisted_ids,
                    context_label="stats_only_comparative_context",
                    include_answers=False,
                    include_group_statistics=True,
                    interaction_state=interaction_state,
                )
                shortlist_policy = "stats_only_no_shortlist_evidence"

            elif variant.family == "shortlist_only":
                shortlisted_ids = full_shortlisted_ids
                shortlist_payload = {
                    **dict(full_shortlist_payload),
                    "shortlisted_candidate_ids": shortlisted_ids,
                    "ranking_policy": "shortlist_only_no_group_statistics",
                }
                context_bundle = _minimal_context(
                    question=question,
                    candidate_outputs=candidate_outputs,
                    shortlisted_candidate_ids=shortlisted_ids,
                    context_label="shortlist_only_comparative_context",
                    include_answers=True,
                    include_group_statistics=False,
                    interaction_state=interaction_state,
                )
                shortlist_policy = "shortlist_only_no_group_statistics"

            elif variant.family == "random_shortlist":
                size = len(full_shortlisted_ids) or min(SHORTLIST_CAP, len(candidate_outputs))
                shortlisted_ids = _stable_random_shortlist(
                    candidate_outputs,
                    size=size,
                    seed=int(variant.random_seed or 0),
                    question=question,
                )
                shortlist_payload = {
                    "shortlisted_candidate_ids": shortlisted_ids,
                    "shortlist_scores": [
                        {"agent_id": agent_id, "score": None, "random_seed": int(variant.random_seed or 0)}
                        for agent_id in shortlisted_ids
                    ],
                    "full_ccr_shortlist_size": size,
                    "ranking_policy": "uniform_random_fixed_size",
                }
                context_bundle = _context_from_payload(
                    question=question,
                    candidate_outputs=candidate_outputs,
                    interaction_state=interaction_state,
                    shortlisted_candidate_ids=shortlisted_ids,
                )
                shortlist_policy = "uniform_random_fixed_size"

            elif variant.family == "no_payload_statistics":
                shortlisted_ids = full_shortlisted_ids
                shortlist_payload = {
                    **dict(full_shortlist_payload),
                    "ranking_policy": "full_ccr_shortlist_without_serialized_payload_statistics",
                }
                context_bundle = _minimal_context(
                    question=question,
                    candidate_outputs=candidate_outputs,
                    shortlisted_candidate_ids=shortlisted_ids,
                    context_label="minimal_shortlist_context_without_group_statistics",
                    include_answers=False,
                    include_group_statistics=False,
                    interaction_state=interaction_state,
                )
                shortlist_policy = "full_ccr_shortlist_without_serialized_payload_statistics"

            else:
                raise ValueError(f"Unsupported CCR ablation family: {variant.family}")

        self._last_ablation_metadata = self._metadata_for_context(
            candidate_outputs=candidate_outputs,
            shortlisted_candidate_ids=shortlisted_ids,
            judge_context_text=context_bundle.judge_context_text,
            ccr_context_text=context_bundle.ccr_context_text,
            shortlist_policy=shortlist_policy,
        )
        return {
            "interaction_state": interaction_state,
            "shortlist_payload": shortlist_payload,
            "ccr_context_payload": context_bundle.ccr_context_payload,
            "ccr_context_text": context_bundle.ccr_context_text,
            "context_bundle": context_bundle,
            "judge_context_text": context_bundle.judge_context_text,
        }

    def _finalize_result(
        self,
        *,
        message: Any | None,
        source_result: Any,
        candidate_outputs: list[CCRCandidateView],
        selected_agent_id: str | None,
        repair_metadata: Dict[str, Any],
    ):
        enriched_metadata = {
            **dict(repair_metadata or {}),
            **dict(self._last_ablation_metadata or {}),
        }
        return super()._finalize_result(
            message=message,
            source_result=source_result,
            candidate_outputs=candidate_outputs,
            selected_agent_id=selected_agent_id,
            repair_metadata=enriched_metadata,
        )


def create_ccr_ablation_judge(
    llm_name: str,
    *,
    domain: str,
    variant: AblationVariant,
) -> CCRAblationJudgeFinalSelect:
    return create_protocol_judge(
        judge_cls=lambda *args, **kwargs: CCRAblationJudgeFinalSelect(
            *args,
            ablation_variant=variant,
            **kwargs,
        ),
        judge_id=f"paper_judge_{variant.slug}",
        llm_name=llm_name,
        domain=domain,
    )


async def run_ccr_ablation_case(
    *,
    judge: CCRAblationJudgeFinalSelect,
    record: FrozenCandidateRecord,
    output_dir: Any,
    variant: AblationVariant,
) -> JudgeMethodResult:
    return await run_standard_judge_case(
        judge=judge,
        method_name=variant.slug,
        record=record,
        output_dir=output_dir,
        mode="allow_kv_reuse",
    )
