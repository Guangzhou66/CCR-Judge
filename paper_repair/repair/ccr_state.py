from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from paper_repair.repair.ccr_types import CCRCandidateView, CCRState


def _normalized_answer_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _normalized_optional_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NONE":
        return None
    return text


def _agent_sort_key(agent_id: Any) -> tuple[int, str]:
    text = str(agent_id)
    try:
        return (0, f"{int(text):08d}")
    except Exception:
        return (1, text)


def _candidate_signal_score(candidate: CCRCandidateView) -> int:
    strongest_alt = _normalized_optional_label(candidate.strongest_alternative_option)
    rejected_option = _normalized_optional_label(candidate.rejected_option)
    supported_yes = bool(candidate.alternative_supported_yes) or (
        str(candidate.alternative_supported_flag or "NO").strip().upper() == "YES"
    )
    overturn_yes = bool(candidate.alternative_overturn_yes) or (
        str(candidate.alternative_overturn_flag or "NO").strip().upper() == "YES"
    )
    return (
        int(strongest_alt is not None)
        + int(rejected_option is not None)
        + int(supported_yes)
        + int(overturn_yes)
    )


def _compute_same_answer_clusters(candidate_outputs: Sequence[CCRCandidateView]) -> Dict[str, Any]:
    counts = Counter()
    canonicalized_distribution: Dict[str, int] = {}
    for item in candidate_outputs:
        answer = _normalized_answer_label(item.normalized_answer)
        counts[answer] += 1
        canonicalized_distribution[answer] = canonicalized_distribution.get(answer, 0) + 1
    cluster_sizes = sorted(counts.values(), reverse=True)
    if cluster_sizes:
        cluster_key = "[" + ",".join(str(value) for value in cluster_sizes) + "]"
        histogram = {cluster_key: 1}
    else:
        histogram = {}
    all_four_single_cluster = cluster_sizes == [4]
    nontrivial = bool(cluster_sizes) and not all_four_single_cluster
    return {
        "unique_normalized_answer_count": len(counts),
        "same_answer_cluster_sizes": cluster_sizes,
        "same_answer_cluster_size_histogram": histogram,
        "nontrivial_same_answer_cluster_exists": nontrivial,
        "all_four_single_cluster": all_four_single_cluster,
        "canonicalized_answer_distribution": canonicalized_distribution,
    }


def _answer_support_maps(candidate_outputs: Sequence[CCRCandidateView]) -> Dict[str, Any]:
    answer_counts: Counter[str] = Counter()
    answer_to_agent_ids: Dict[str, List[str]] = {}
    for item in candidate_outputs:
        answer = _normalized_answer_label(item.normalized_answer)
        answer_counts[answer] += 1
        answer_to_agent_ids.setdefault(answer, []).append(item.agent_id)
    for answer in answer_to_agent_ids:
        answer_to_agent_ids[answer] = sorted(answer_to_agent_ids[answer], key=_agent_sort_key)
    return {
        "answer_counts": dict(answer_counts),
        "answer_to_agent_ids": answer_to_agent_ids,
    }


def _structured_evidence_summary(candidate_outputs: Sequence[CCRCandidateView]) -> Dict[str, Dict[str, int]]:
    strongest_alt_counts: Counter[str] = Counter()
    supported_alt_counts: Counter[str] = Counter()
    overturn_alt_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()

    for item in candidate_outputs:
        answer = None if item.normalized_answer is None else str(item.normalized_answer).strip()
        strongest_alt = None if item.strongest_alternative_option in {None, "NONE"} else str(item.strongest_alternative_option).strip()
        rejected_option = None if item.rejected_option in {None, "NONE"} else str(item.rejected_option).strip()
        supported_yes = bool(item.alternative_supported_yes) or (
            str(item.alternative_supported_flag or "NO").strip().upper() == "YES"
        )
        overturn_yes = bool(item.alternative_overturn_yes) or (
            str(item.alternative_overturn_flag or "NO").strip().upper() == "YES"
        )

        if strongest_alt is not None and strongest_alt != answer:
            strongest_alt_counts[strongest_alt] += 1
            if supported_yes:
                supported_alt_counts[strongest_alt] += 1
            if overturn_yes:
                overturn_alt_counts[strongest_alt] += 1
        if rejected_option is not None:
            rejected_counts[rejected_option] += 1

    structured_evidence_counts: Counter[str] = Counter()
    for answer in (
        set(strongest_alt_counts)
        | set(supported_alt_counts)
        | set(overturn_alt_counts)
        | set(rejected_counts)
    ):
        structured_evidence_counts[answer] = (
            int(strongest_alt_counts.get(answer, 0))
            + int(supported_alt_counts.get(answer, 0))
            + int(overturn_alt_counts.get(answer, 0))
        )

    return {
        "strongest_alt_counts": dict(strongest_alt_counts),
        "supported_alt_counts": dict(supported_alt_counts),
        "overturn_alt_counts": dict(overturn_alt_counts),
        "rejected_counts": dict(rejected_counts),
        "structured_evidence_counts": dict(structured_evidence_counts),
    }


def _answer_cluster_summary(answer_counts: Dict[str, int]) -> str:
    if not answer_counts:
        return ""
    ordered = sorted(answer_counts.items(), key=lambda item: (-int(item[1]), item[0]))
    return ", ".join(f"{answer} x{count}" for answer, count in ordered)


def _ranked_answer_groups(
    *,
    answer_counts: Dict[str, int],
    structured_evidence_counts: Dict[str, int],
    later_slot_answer_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for answer, support in answer_counts.items():
        structured = int(structured_evidence_counts.get(answer, 0))
        later_support = int(later_slot_answer_counts.get(answer, 0))
        stability = (2 * int(support)) + structured
        groups.append(
            {
                "answer": answer,
                "support": int(support),
                "structured_evidence": structured,
                "later_slot_support": later_support,
                "stability_score": stability,
            }
        )
    groups.sort(
        key=lambda item: (
            -int(item["stability_score"]),
            -int(item["support"]),
            -int(item["structured_evidence"]),
            -int(item["later_slot_support"]),
            str(item["answer"]),
        )
    )
    return groups


def _split_severity_label(
    *,
    cluster_stats: Dict[str, Any],
    ranked_groups: Sequence[Dict[str, Any]],
) -> str:
    sizes = list(cluster_stats.get("same_answer_cluster_sizes") or [])
    if sizes == [4]:
        return "trivial"

    strongest_support = int(ranked_groups[0]["support"]) if ranked_groups else 0
    strongest_structured = int(ranked_groups[0]["structured_evidence"]) if ranked_groups else 0
    second_support = int(ranked_groups[1]["support"]) if len(ranked_groups) > 1 else 0
    second_structured = int(ranked_groups[1]["structured_evidence"]) if len(ranked_groups) > 1 else 0

    if sizes == [3, 1] and strongest_structured == 0 and second_support <= 1 and second_structured == 0:
        return "harmless_nontrivial"
    return "consequential_nontrivial"


def build_ccr_state(candidate_outputs: Sequence[CCRCandidateView]) -> CCRState:
    support_maps = _answer_support_maps(candidate_outputs)
    answer_counts = dict(support_maps["answer_counts"])
    answer_to_agent_ids = {
        answer: sorted(agent_ids, key=_agent_sort_key)
        for answer, agent_ids in dict(support_maps["answer_to_agent_ids"]).items()
    }
    cluster_stats = _compute_same_answer_clusters(candidate_outputs)
    evidence_maps = _structured_evidence_summary(candidate_outputs)
    structured_evidence_counts = dict(evidence_maps["structured_evidence_counts"])

    candidate_count = len(candidate_outputs)
    later_slot_start = max(1, candidate_count // 2)
    later_slot_answer_counts: Counter[str] = Counter()
    per_candidate_signals: Dict[str, int] = {}
    representative_by_answer: Dict[str, str] = {}
    candidate_positions: Dict[str, int] = {}

    for position, item in enumerate(candidate_outputs):
        answer = _normalized_answer_label(item.normalized_answer)
        candidate_positions[item.agent_id] = position
        per_candidate_signals[item.agent_id] = _candidate_signal_score(item)
        representative_by_answer.setdefault(answer, item.agent_id)
        if position >= later_slot_start:
            later_slot_answer_counts[answer] += 1

    ranked_groups = _ranked_answer_groups(
        answer_counts=answer_counts,
        structured_evidence_counts=structured_evidence_counts,
        later_slot_answer_counts=dict(later_slot_answer_counts),
    )
    top_group = ranked_groups[0] if ranked_groups else None
    second_group = ranked_groups[1] if len(ranked_groups) > 1 else None
    internal_anchor_answer = _normalized_optional_label(top_group.get("answer")) if top_group else None
    internal_anchor_strength = int(top_group.get("stability_score", 0)) if top_group else 0
    split_severity_label = _split_severity_label(
        cluster_stats=cluster_stats,
        ranked_groups=ranked_groups,
    )

    top_support = int(top_group.get("support", 0)) if top_group else 0
    second_support = int(second_group.get("support", 0)) if second_group else 0
    support_gap = top_support - second_support
    consequential_split_flag = split_severity_label == "consequential_nontrivial"
    multi_cluster_competition_flag = bool(
        len(ranked_groups) >= 2
        and second_group is not None
        and (
            int(second_group["support"]) >= 2
            or int(second_group["structured_evidence"]) > 0
            or support_gap <= 1
        )
    )
    later_slot_under_attention_risk_flag = any(
        int(group["later_slot_support"]) > 0
        and (
            _normalized_optional_label(group["answer"]) == internal_anchor_answer
            or int(group["support"]) >= max(1, second_support)
            or int(group["structured_evidence"]) > 0
        )
        for group in ranked_groups
    )
    interaction_fragility_score = min(
        1.0,
        (
            0.25 * int(cluster_stats.get("unique_normalized_answer_count", 0) >= 2)
            + 0.20 * int(multi_cluster_competition_flag)
            + 0.15 * int(consequential_split_flag)
            + 0.20 * int(support_gap <= 1 and len(ranked_groups) >= 2)
            + 0.20 * int(later_slot_under_attention_risk_flag)
        ),
    )

    competing_groups = ranked_groups[: min(3, len(ranked_groups))]
    later_slot_risk_summary = {
        "consequential_split_flag": consequential_split_flag,
        "multi_cluster_competition_flag": multi_cluster_competition_flag,
        "later_slot_under_attention_risk_flag": later_slot_under_attention_risk_flag,
        "support_gap": support_gap,
        "feature_gate_v3_fragility_score": interaction_fragility_score,
    }
    metadata = {
        "answer_counts": answer_counts,
        "answer_to_agent_ids": answer_to_agent_ids,
        "cluster_stats": cluster_stats,
        "structured_evidence_summary": evidence_maps,
        "structured_evidence_counts": structured_evidence_counts,
        "answer_stability_scores": {
            item["answer"]: int(item["stability_score"]) for item in ranked_groups
        },
        "candidate_positions": candidate_positions,
        "per_candidate_signal_scores": per_candidate_signals,
        "representative_by_answer": representative_by_answer,
        "later_slot_answer_counts": dict(later_slot_answer_counts),
        "ranked_answer_groups": ranked_groups,
        "competing_answer_groups": competing_groups,
        "interaction_risk_summary": later_slot_risk_summary,
        "ccr_state_summary": {
            "answer_cluster_summary": _answer_cluster_summary(answer_counts),
            "internal_consensus_anchor_answer": internal_anchor_answer,
            "internal_consensus_anchor_strength": internal_anchor_strength,
            "split_severity_label": split_severity_label,
            "competing_answer_groups": competing_groups,
            "interaction_risk_summary": dict(later_slot_risk_summary),
        },
    }
    metadata["slate_interaction_state_summary"] = dict(metadata["ccr_state_summary"])
    return CCRState(
        answer_support_map=answer_to_agent_ids,
        answer_cluster_summary=_answer_cluster_summary(answer_counts),
        internal_consensus_anchor_answer=internal_anchor_answer,
        internal_consensus_anchor_strength=internal_anchor_strength,
        split_severity_label=split_severity_label,
        later_slot_risk_summary=later_slot_risk_summary,
        metadata=metadata,
    )


build_slate_interaction_state = build_ccr_state
