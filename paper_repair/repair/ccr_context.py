from __future__ import annotations

"""Comparative decision-context payload construction and serialization."""

from typing import Any, Dict, List, Sequence

from paper_repair.repair.ccr_types import (
    CCRCandidateView,
    CCRContext,
    CCRContextPayload,
    CCRState,
)


def _answer_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _candidate_for_agent(
    candidate_outputs: Sequence[CCRCandidateView],
    agent_id: Any,
) -> CCRCandidateView | None:
    for item in candidate_outputs:
        if item.agent_id == str(agent_id):
            return item
    return None


def _shortlist_entries(
    *,
    shortlisted_candidates: Sequence[CCRCandidateView],
    candidate_positions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "agent_id": candidate.agent_id,
            "answer": _answer_label(candidate.normalized_answer),
            "slot": int(candidate_positions.get(candidate.agent_id, 0)),
        }
        for candidate in shortlisted_candidates
    ]


def _candidate_summaries(
    *,
    candidate_outputs: Sequence[CCRCandidateView],
    candidate_positions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for candidate in candidate_outputs:
        summaries.append(
            {
                "agent_id": candidate.agent_id,
                "answer": _answer_label(candidate.normalized_answer),
                "slot": int(candidate_positions.get(candidate.agent_id, 0)),
                "role": candidate.role,
            }
        )
    return summaries


def build_ccr_context_payload(
    *,
    question: str,
    candidate_outputs: Sequence[CCRCandidateView],
    interaction_state: CCRState,
    shortlisted_candidate_ids: List[str],
) -> CCRContextPayload:
    shortlisted_candidates = [
        _candidate_for_agent(candidate_outputs, agent_id)
        for agent_id in shortlisted_candidate_ids
    ]
    shortlisted_candidates = [item for item in shortlisted_candidates if item is not None]

    state = interaction_state.to_dict()
    competing_groups = list(state.get("competing_answer_groups") or [])
    candidate_positions = dict(state.get("candidate_positions", {}) or {})
    later_slot_risk = dict(state.get("interaction_risk_summary") or {})
    risk_notes = [
        "Compare the whole slate before deciding on a winner.",
        "Do not assume any preliminary winner.",
    ]
    if later_slot_risk.get("later_slot_under_attention_risk_flag"):
        risk_notes.append(
            "Candidates that appear later in the slate may still carry decisive evidence and must receive equal attention."
        )
    if later_slot_risk.get("multi_cluster_competition_flag"):
        risk_notes.append(
            "Multiple answer groups remain competitive, so compare representative candidates across clusters instead of collapsing too early."
        )
    if not risk_notes:
        risk_notes.append("Compare all candidates evenly across answer groups.")

    internal_anchor = {
        "answer": _answer_label(interaction_state.internal_consensus_anchor_answer),
        "strength": int(interaction_state.internal_consensus_anchor_strength or 0),
    }
    context_flags = {
        "decision_context_repair": True,
        "instance_specific": True,
        "slate_conditioned": True,
        "selection_stage_only": True,
        "modifies_candidate_generation": False,
    }
    canonical_tiebreak_info = {
        "canonical_tiebreak_applicable": (
            str(interaction_state.split_severity_label or "").strip().lower() == "trivial"
        ),
        "canonical_representative_rule": "frontmost candidate within the winning answer cluster under candidate_ordering",
    }
    return CCRContextPayload(
        answer_clusters=interaction_state.answer_cluster_summary or "none",
        internal_anchor=internal_anchor,
        split_severity=interaction_state.split_severity_label or "unknown",
        shortlist=_shortlist_entries(
            shortlisted_candidates=shortlisted_candidates,
            candidate_positions=candidate_positions,
        ),
        later_slot_risk=later_slot_risk,
        context_flags=context_flags,
        competing_answer_groups=competing_groups,
        risk_notes=risk_notes,
        original_question=str(question).strip(),
        candidate_summaries=_candidate_summaries(
            candidate_outputs=candidate_outputs,
            candidate_positions=candidate_positions,
        ),
        canonical_tiebreak_info=canonical_tiebreak_info,
    )


def render_ccr_context_text(payload: CCRContextPayload) -> str:
    group_summary = "; ".join(
        (
            f"{item.get('answer')} support={int(item.get('support', 0))}, "
            f"evidence={int(item.get('structured_evidence', 0))}, "
            f"later_slot={int(item.get('later_slot_support', 0))}"
        )
        for item in payload.competing_answer_groups
    )
    shortlist_summary = "; ".join(
        (
            f"Agent {item.get('agent_id')} -> {item.get('answer')} "
            f"(slot {int(item.get('slot', 0))})"
        )
        for item in payload.shortlist
    )
    context_lines = [
        "[Comparative Decision Context]",
        "Structured candidate-slate interaction summary:",
        f"- Answer clusters: {payload.answer_clusters or 'none'}",
        (
            "- Internal stability anchor: "
            f"{payload.internal_anchor.get('answer', 'NONE')} "
            f"(strength {int(payload.internal_anchor.get('strength', 0))})"
        ),
        f"- Split severity: {payload.split_severity or 'unknown'}",
        f"- Strongest answer groups: {group_summary or 'none'}",
        f"- Shortlist for careful comparison: {shortlist_summary or 'none'}",
        *[f"- {note}" for note in payload.risk_notes],
        "Select the best candidate from the full slate after using this comparative decision context.",
    ]
    return "\n".join(context_lines).strip()


def build_ccr_context(
    *,
    question: str,
    candidate_outputs: Sequence[CCRCandidateView],
    interaction_state: CCRState,
    shortlisted_candidate_ids: List[str],
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
            payload.original_question,
        ]
    ).strip()
    context_summary = {
        "split_severity_label": payload.split_severity,
        "internal_consensus_anchor_answer": payload.internal_anchor.get("answer"),
        "internal_consensus_anchor_strength": payload.internal_anchor.get("strength"),
        "competing_answer_groups": list(payload.competing_answer_groups),
        "shortlisted_candidate_ids": [str(item.get("agent_id")) for item in payload.shortlist],
        "risk_notes": list(payload.risk_notes),
        "decision_context_label": "comparative_decision_context",
    }
    return CCRContext(
        ccr_context_payload=payload,
        ccr_context_text=context_text,
        judge_context_text=judge_context_text,
        ccr_context_summary=context_summary,
    )


# Backward-compatible alias retained for older imports.
build_repaired_compare_scaffold = build_ccr_context
