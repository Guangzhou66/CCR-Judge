from __future__ import annotations

from typing import Any, Dict, List, Sequence

from paper_repair.eval.parsing import normalize_choice
from paper_repair.repair.ccr_types import CCRCandidateView, CCRState


def select_ccr_shortlist(
    candidate_outputs: Sequence[CCRCandidateView],
    *,
    interaction_state: CCRState,
    max_candidates: int = 3,
) -> Dict[str, Any]:
    state = interaction_state.to_dict()
    answer_counts = dict(state.get("answer_counts") or {})
    structured_counts = dict(state.get("structured_evidence_counts") or {})
    candidate_positions = dict(state.get("candidate_positions") or {})
    per_candidate_signals = dict(state.get("per_candidate_signal_scores") or {})
    representative_by_answer = dict(state.get("representative_by_answer") or {})
    ranked_groups = list(state.get("ranked_answer_groups") or [])
    internal_anchor_answer = normalize_choice(interaction_state.internal_consensus_anchor_answer)
    top_answer_set = {
        normalize_choice(item.get("answer"))
        for item in ranked_groups[: min(2, len(ranked_groups))]
        if normalize_choice(item.get("answer")) is not None
    }
    later_slot_start = max(1, len(candidate_outputs) // 2)

    scored_candidates: List[Dict[str, Any]] = []
    for item in candidate_outputs:
        answer = normalize_choice(item.normalized_answer)
        support = int(answer_counts.get(answer or "", 0))
        structured = int(structured_counts.get(answer or "", 0))
        candidate_signal = int(per_candidate_signals.get(item.agent_id, 0))
        position = int(candidate_positions.get(item.agent_id, 0))
        later_slot_bonus = 2 if position >= later_slot_start else 0
        score = (
            (8 if answer == internal_anchor_answer else 0)
            + (5 * support)
            + (3 * structured)
            + (2 * candidate_signal)
            + (3 if representative_by_answer.get(answer or "") == item.agent_id else 0)
            + (2 if answer in top_answer_set else 0)
            + later_slot_bonus
            + (1 if support == 1 and len(answer_counts) >= 2 else 0)
        )
        scored_candidates.append(
            {
                "agent_id": item.agent_id,
                "answer": answer,
                "score": score,
                "support": support,
                "structured_evidence": structured,
                "candidate_signal_score": candidate_signal,
                "later_slot_bonus": later_slot_bonus,
                "position": position,
            }
        )

    scored_candidates.sort(
        key=lambda item: (
            -int(item["score"]),
            -int(item["support"]),
            -int(item["structured_evidence"]),
            -int(item["later_slot_bonus"]),
            int(item["position"]),
            str(item["agent_id"]),
        )
    )

    shortlisted_ids: List[str] = []
    seen_answers: set[str] = set()
    for item in scored_candidates:
        answer = normalize_choice(item.get("answer"))
        if answer is not None and answer in seen_answers:
            continue
        shortlisted_ids.append(str(item["agent_id"]))
        if answer is not None:
            seen_answers.add(answer)
        if len(shortlisted_ids) >= max_candidates:
            break

    if len(shortlisted_ids) < max_candidates:
        for item in scored_candidates:
            agent_id = str(item["agent_id"])
            if agent_id in shortlisted_ids:
                continue
            shortlisted_ids.append(agent_id)
            if len(shortlisted_ids) >= max_candidates:
                break

    return {
        "shortlisted_candidate_ids": shortlisted_ids,
        "shortlist_scores": scored_candidates,
    }


shortlist_candidate_ids_from_slate = select_ccr_shortlist
