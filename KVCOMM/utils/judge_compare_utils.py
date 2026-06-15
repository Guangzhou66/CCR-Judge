from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_compact_compare_messages(
    *,
    question: str,
    candidate_cards: List[Dict[str, Any]],
    pairwise_summary: Dict[str, Any] | None = None,
    trigger_reason: str | None = None,
) -> Dict[str, Any]:
    """Build a compact joint-compare prompt over candidate cards.

    This helper deliberately keeps the prompt short and only exposes compact
    candidate cards, rather than replaying full long-form candidate outputs.
    """

    system_prompt = (
        "You are performing a dense-like cross-candidate selection task.\n"
        "You must jointly compare all candidate cards and select the single agent"
        " that a strong dense judge would be most likely to select after comparing"
        " the whole slate together.\n"
        "Do not reduce the task to picking which answer option looks most plausible."
        " Instead, rank candidates by their overall strength against the others.\n"
        "Consider all of the following jointly:\n"
        "- final answer and normalized final answer\n"
        "- reasoning quality and completeness\n"
        "- alignment with the question\n"
        "- internal consistency between answer and rationale\n"
        "- confidence / decisiveness\n"
        "- relative strength versus competing candidates\n"
        "If multiple candidates support the same answer, prefer the stronger,"
        " clearer, more internally consistent candidate.\n"
        "Do not prefer a candidate simply because it wrote more text.\n"
        "Your output must be easy to parse."
    )

    card_lines: List[str] = []
    card_summary: List[Dict[str, Any]] = []
    for card in candidate_cards:
        agent_id = str(card.get("agent_id", ""))
        role = str(card.get("role", "unknown"))
        final_answer = str(card.get("final_answer", "") or "")
        normalized = str(card.get("normalized_final_answer", "") or "")
        answer_type = str(card.get("answer_type", "unknown"))
        rationale = str(card.get("short_rationale", "") or "")
        support_hint = str(card.get("support_hint", "") or "")
        evidence = str(card.get("optional_domain_evidence", "") or "")
        reasoning_quality_hint = str(card.get("reasoning_quality_hint", "") or "")
        confidence_hint = str(card.get("confidence_hint", "") or "")
        consistency_hint = str(card.get("consistency_hint", "") or "")
        strength_score = card.get("candidate_strength_score")

        pieces = [
            f"Agent {agent_id}",
            f"Role: {role}",
            f"Final answer: {final_answer or '[missing]'}",
            f"Normalized final answer: {normalized or '[missing]'}",
            f"Answer type: {answer_type}",
        ]
        if reasoning_quality_hint:
            pieces.append(f"Reasoning quality hint: {reasoning_quality_hint}")
        if confidence_hint:
            pieces.append(f"Confidence hint: {confidence_hint}")
        if consistency_hint:
            pieces.append(f"Consistency hint: {consistency_hint}")
        if strength_score is not None:
            pieces.append(f"Candidate strength score: {strength_score}")
        if support_hint:
            pieces.append(f"Support hint: {support_hint}")
        if rationale:
            pieces.append(f"Short rationale: {rationale}")
        if evidence:
            pieces.append(f"Domain evidence: {evidence}")
        card_lines.append("\n".join(pieces))
        card_summary.append(
            {
                "agent_id": agent_id,
                "normalized_final_answer": normalized or None,
                "answer_type": answer_type,
                "support_hint": support_hint or None,
                "reasoning_quality_hint": reasoning_quality_hint or None,
                "confidence_hint": confidence_hint or None,
                "consistency_hint": consistency_hint or None,
                "candidate_strength_score": strength_score,
            }
        )

    trigger_line = (
        f"Trigger reason: {trigger_reason}\n\n" if trigger_reason else ""
    )
    pairwise_lines: List[str] = []
    if isinstance(pairwise_summary, dict):
        raw_winner = pairwise_summary.get("raw_winner_agent_id")
        top_rival = pairwise_summary.get("top_rival_agent_id")
        second_rival = pairwise_summary.get("second_rival_agent_id")
        pairwise_lines.append(f"Raw winner: Agent {raw_winner}" if raw_winner is not None else "Raw winner: [missing]")
        pairwise_lines.append(f"Top rival: Agent {top_rival}" if top_rival is not None else "Top rival: [missing]")
        if second_rival is not None:
            pairwise_lines.append(f"Second rival: Agent {second_rival}")
        for label, entry in (
            ("Winner vs top rival", pairwise_summary.get("winner_vs_top_rival")),
            ("Winner vs second rival", pairwise_summary.get("winner_vs_second_rival")),
        ):
            if not isinstance(entry, dict):
                continue
            pairwise_lines.append(
                f"{label}: answers_differ={entry.get('answers_differ')}, "
                f"strength_margin={entry.get('winner_vs_rival_strength_margin')}, "
                f"reasoning_gap={entry.get('winner_vs_rival_reasoning_gap')}, "
                f"confidence_gap={entry.get('winner_vs_rival_confidence_gap')}, "
                f"consistency_gap={entry.get('winner_vs_rival_consistency_gap')}, "
                f"support_gap={entry.get('winner_vs_rival_support_gap')}, "
                f"dense_like_preference={entry.get('dense_like_preference')}"
            )
    pairwise_block = (
        "Winner-centered Rival Analysis:\n"
        f"{chr(10).join(pairwise_lines)}\n\n"
        if pairwise_lines
        else ""
    )
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        f"{trigger_line}"
        f"{pairwise_block}"
        "Candidate Cards:\n"
        f"{chr(10).join(card_lines)}\n\n"
        "Instructions:\n"
        "1. First compare the raw winner against its strongest rivals.\n"
        "2. Decide whether the raw winner still survives those pairwise comparisons.\n"
        "3. Then jointly compare the whole slate and pick the single candidate that a dense joint judge would be most likely to select.\n"
        "4. If several candidates support the same answer, prefer the strongest candidate rather than the longest one.\n"
        "5. Output format must be exactly:\n"
        "Selected agent id: <id>\n"
        "Reason (short): <brief reason>\n"
        "6. The selected agent id must be one of the provided candidates.\n"
        "7. Keep the reason short and agent-selection-focused.\n"
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "meta": {
            "candidate_count": len(candidate_cards),
            "trigger_reason": trigger_reason,
            "pairwise_summary": pairwise_summary,
            "card_summary": card_summary,
        },
    }


def build_pairwise_compare_messages(
    *,
    question: str,
    left_card: Dict[str, Any],
    right_card: Dict[str, Any],
    view_name: str | None = None,
    trigger_reason: str | None = None,
    pair_label: str | None = None,
) -> Dict[str, Any]:
    """Build a compact pairwise compare prompt between two anonymized cards."""

    system_prompt = (
        "You are performing a dense-like pairwise candidate comparison.\n"
        "You must compare exactly two candidate cards and decide which candidate"
        " a strong dense judge would be more likely to select.\n"
        "Do not reduce the task to only picking the more plausible answer option.\n"
        "Instead compare the two candidates holistically using:\n"
        "- final answer and normalized final answer\n"
        "- reasoning quality and closure\n"
        "- confidence / decisiveness\n"
        "- internal consistency\n"
        "- relative strength against the other candidate\n"
        "Your output must be easy to parse.\n"
        "Return a structured decision rather than a long explanation."
    )

    trigger_line = f"Trigger reason: {trigger_reason}\n\n" if trigger_reason else ""
    pair_line = f"Pair label: {pair_label}\n\n" if pair_label else ""
    view_line = f"Representation view: {view_name}\n\n" if view_name else ""

    def _render_card(alias: str, card: Dict[str, Any]) -> str:
        normalized = str(card.get("normalized_final_answer", "") or "")
        conclusion_sentence = str(card.get("conclusion_sentence", "") or "")
        evidence_sentence = str(card.get("evidence_sentence", "") or "")
        final_answer_explicitness = str(card.get("final_answer_explicitness", "") or "")
        confidence_label = str(card.get("confidence_label", "") or "")
        consistency_label = str(card.get("consistency_label", "") or "")
        same_answer_peers = card.get("same_answer_peers")
        risk_flags = str(card.get("risk_flags", "") or "none")
        pieces = [f"Candidate: {alias}", f"Normalized answer: {normalized or '[missing]'}"]
        if view_name == "answer_only_strict":
            pieces.extend(
                [
                    f"Conclusion sentence: {conclusion_sentence or '[missing]'}",
                    f"Final-answer explicitness: {final_answer_explicitness or 'weak'}",
                ]
            )
        else:
            final_answer = str(card.get("final_answer", "") or "")
            pieces.extend(
                [
                    f"Final answer: {final_answer or '[missing]'}",
                    f"Conclusion sentence: {conclusion_sentence or '[missing]'}",
                    f"Evidence sentence: {evidence_sentence or '[missing]'}",
                    f"Confidence: {confidence_label or 'weak'}",
                    f"Consistency: {consistency_label or 'mixed'}",
                    f"Same-answer peers: {same_answer_peers if same_answer_peers is not None else 0}",
                    f"Risk flags: {risk_flags}",
                ]
            )
        return "\n".join(pieces)

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        f"{trigger_line}"
        f"{pair_line}"
        f"{view_line}"
        "Pairwise Candidate Cards:\n"
        f"{_render_card('C1', left_card)}\n\n"
        f"{_render_card('C2', right_card)}\n\n"
        "Instructions:\n"
        "1. Compare these two candidates directly.\n"
        "2. Decide which candidate a dense judge would be more likely to select after comparing both side by side.\n"
        "3. Consider answer quality, reasoning quality, confidence, internal consistency, and overall relative strength.\n"
        "4. Output format must be exactly:\n"
        "Winner candidate: <C1 or C2>\n"
        "Confidence: <1-5>\n"
        "Primary reason: <answer|reasoning|consistency|confidence>\n"
        "Reason (short): <brief reason>\n"
        "5. Keep the reason short and focused on why one candidate beats the other.\n"
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "meta": {
            "trigger_reason": trigger_reason,
            "pair_label": pair_label,
            "view_name": view_name,
            "left_summary": {
                "normalized_final_answer": left_card.get("normalized_final_answer"),
                "reasoning_quality_hint": left_card.get("reasoning_quality_hint"),
                "confidence_hint": left_card.get("confidence_hint"),
                "consistency_hint": left_card.get("consistency_hint"),
                "candidate_strength_score": left_card.get("candidate_strength_score"),
            },
            "right_summary": {
                "normalized_final_answer": right_card.get("normalized_final_answer"),
                "reasoning_quality_hint": right_card.get("reasoning_quality_hint"),
                "confidence_hint": right_card.get("confidence_hint"),
                "consistency_hint": right_card.get("consistency_hint"),
                "candidate_strength_score": right_card.get("candidate_strength_score"),
            },
        },
    }


def build_decision_side_contrast_messages(
    *,
    question: str,
    candidates: List[Tuple[str, str]],
    allowed_agent_ids: List[str],
    candidate_summary_table: str,
    same_answer_cluster_summary: str,
    pairwise_contrast_table: str,
    raw_windows_summary: str,
    trigger_reason: str | None = None,
) -> Dict[str, Any]:
    allowed = ", ".join(allowed_agent_ids)
    system_prompt = (
        "You are a dense-like final judge performing a joint cross-candidate selection task.\n"
        "You will be given the full candidate answers and a Cross-Candidate Contrast Block.\n"
        "You must read both parts before making the final selection.\n"
        "Your goal is not merely to pick the most plausible answer option.\n"
        "Instead, choose the single candidate that a strong dense judge would be most likely"
        " to select after jointly comparing the entire candidate slate.\n"
        "You must compare same-answer candidates against each other for quality.\n"
        "Do not prefer a candidate only because it wrote more text.\n"
        "Use the contrast block as part of the final decision, not as optional commentary.\n"
        "Output must remain easy to parse."
    )

    formatted_candidates = "\n\n".join(
        f"Agent {agent_id}:\n{answer}" for agent_id, answer in candidates
    )
    trigger_line = f"Trigger reason: {trigger_reason}\n\n" if trigger_reason else ""
    user_prompt = (
        "# Best Agent Selection Task (Decision-Side Contrast)\n\n"
        f"## Question:\n{question}\n\n"
        "## Candidate Agent Answers:\n"
        f"{formatted_candidates}\n\n"
        "## Cross-Candidate Contrast Block:\n"
        f"{trigger_line}"
        "### Candidate Summary Table\n"
        f"{candidate_summary_table}\n\n"
        "### Raw Decision Windows\n"
        f"{raw_windows_summary}\n\n"
        "### Same-Answer Cluster Summary\n"
        f"{same_answer_cluster_summary}\n\n"
        "### Key Pairwise Contrasts\n"
        f"{pairwise_contrast_table}\n\n"
        "## Instructions:\n"
        "1. First read the Candidate Agent Answers.\n"
        "2. Then read the Cross-Candidate Contrast Block.\n"
        "3. Use both the raw candidate answers and the contrast block when choosing the winner.\n"
        "4. Compare same-answer clusters internally and prefer the stronger candidate within a cluster.\n"
        "5. Focus on which candidate a dense joint judge would most likely select.\n"
        "6. Do not select based only on text length or verbosity.\n"
        "7. Output format must be exactly:\n"
        f"   First line: Selected agent id: <id> (id must be one of: {allowed})\n"
        "   Second line: <choice> (must be exactly one of: A, B, C, D)\n"
        "   Do NOT output anything else.\n"
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "meta": {
            "candidate_count": len(candidates),
            "allowed_agent_ids": allowed_agent_ids,
            "trigger_reason": trigger_reason,
            "candidate_summary_table": candidate_summary_table,
            "same_answer_cluster_summary": same_answer_cluster_summary,
            "pairwise_contrast_table": pairwise_contrast_table,
            "raw_windows_summary": raw_windows_summary,
        },
    }
