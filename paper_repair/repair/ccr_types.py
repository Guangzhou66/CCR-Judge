from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class CCRCandidateView:
    agent_id: str
    role: str
    text: str
    normalized_answer: str | None
    conclusion: str | None = None
    evidence: str | None = None
    strongest_alternative_considered: str | None = None
    strongest_alternative_option: str | None = None
    rejected_option: str | None = None
    why_not_that_option: str | None = None
    why_not_rejected_option: str | None = None
    alternative_overturn_flag: str | None = None
    alternative_overturn_yes: bool = False
    alternative_supported_flag: str | None = None
    alternative_supported_yes: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "text": self.text,
            "normalized_answer": self.normalized_answer,
            "conclusion": self.conclusion,
            "evidence": self.evidence,
            "strongest_alternative_considered": self.strongest_alternative_considered,
            "strongest_alternative_option": self.strongest_alternative_option,
            "rejected_option": self.rejected_option,
            "why_not_that_option": self.why_not_that_option,
            "why_not_rejected_option": self.why_not_rejected_option,
            "alternative_overturn_flag": self.alternative_overturn_flag,
            "alternative_overturn_yes": self.alternative_overturn_yes,
            "alternative_supported_flag": self.alternative_supported_flag,
            "alternative_supported_yes": self.alternative_supported_yes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FrozenCandidateRecord:
    record_index: int
    question_text: str
    gold_answer: str
    generation_regime: str
    candidates: List[CCRCandidateView]
    permutation: List[str]
    dataset_name: str = "mmlu"
    dataset_config: str = ""
    task_id: str = ""
    reference_reasoning: str = ""
    target_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def candidate_agent_ids(self) -> List[str]:
        return [candidate.agent_id for candidate in self.candidates]

    def candidate_map(self) -> Dict[str, CCRCandidateView]:
        return {candidate.agent_id: candidate for candidate in self.candidates}


@dataclass(frozen=True)
class CCRState:
    answer_support_map: Dict[str, List[str]]
    answer_cluster_summary: str
    internal_consensus_anchor_answer: str | None
    internal_consensus_anchor_strength: int
    split_severity_label: str
    later_slot_risk_summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_support_map": dict(self.answer_support_map),
            "answer_cluster_summary": self.answer_cluster_summary,
            "internal_consensus_anchor_answer": self.internal_consensus_anchor_answer,
            "internal_consensus_anchor_strength": self.internal_consensus_anchor_strength,
            "split_severity_label": self.split_severity_label,
            "later_slot_risk_summary": dict(self.later_slot_risk_summary),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class CCRContextPayload:
    """Structured comparative decision-context payload for final selection."""

    answer_clusters: str
    internal_anchor: Dict[str, Any]
    split_severity: str
    shortlist: List[Dict[str, Any]] = field(default_factory=list)
    later_slot_risk: Dict[str, Any] = field(default_factory=dict)
    context_flags: Dict[str, Any] = field(default_factory=dict)
    competing_answer_groups: List[Dict[str, Any]] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    original_question: str = ""
    candidate_summaries: List[Dict[str, Any]] = field(default_factory=list)
    canonical_tiebreak_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_clusters": self.answer_clusters,
            "internal_anchor": dict(self.internal_anchor),
            "split_severity": self.split_severity,
            "shortlist": list(self.shortlist),
            "later_slot_risk": dict(self.later_slot_risk),
            "context_flags": dict(self.context_flags),
            "competing_answer_groups": list(self.competing_answer_groups),
            "risk_notes": list(self.risk_notes),
            "original_question": self.original_question,
            "candidate_summaries": list(self.candidate_summaries),
            "canonical_tiebreak_info": dict(self.canonical_tiebreak_info),
        }


@dataclass(frozen=True)
class CCRContext:
    """Rendered judge-consumable context plus its structured payload."""

    ccr_context_payload: CCRContextPayload
    ccr_context_text: str
    judge_context_text: str
    ccr_context_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def repaired_task_text(self) -> str:
        return self.judge_context_text

    @property
    def repaired_compare_scaffold_text(self) -> str:
        return self.ccr_context_text

    @property
    def repaired_compare_scaffold_summary(self) -> Dict[str, Any]:
        return self.ccr_context_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ccr_context_payload": self.ccr_context_payload.to_dict(),
            "ccr_context_text": self.ccr_context_text,
            "judge_context_text": self.judge_context_text,
            "ccr_context_summary": dict(self.ccr_context_summary),
            # Compatibility aliases for legacy readers.
            "repaired_task_text": self.repaired_task_text,
            "repaired_compare_scaffold_text": self.repaired_compare_scaffold_text,
            "repaired_compare_scaffold_summary": dict(self.repaired_compare_scaffold_summary),
        }


@dataclass(frozen=True)
class JudgeMethodResult:
    method_name: str
    selected_agent_id: str | None
    selected_answer: str | None
    parse_success: bool
    fallback_used: bool
    fallback_reason: str | None
    raw_judge_text: str
    reuse_mode: str | None = None
    ttft: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseEvaluationRow:
    record_index: int
    method_result: JudgeMethodResult
    dense_selected_agent_id: str | None
    jcr_match: bool | None
    correct: bool
    permutation: List[str]
    task_id: str = ""
    question_text: str = ""
    gold_answer: str = ""
    generation_regime: str = ""
    candidate_count: int = 0
    candidate_agent_ids: List[str] = field(default_factory=list)
    dataset_name: str = "mmlu"
    official_scoring: Dict[str, Any] = field(default_factory=dict)
    inverse_permutation: Dict[str, int] = field(default_factory=dict)
    shuffle_flag: bool = False
    selected_candidate_passed: bool | None = None
    candidate_texts: List[str] = field(default_factory=list)
    candidate_passed_list: List[bool] = field(default_factory=list)
    num_correct_candidates: int = 0
    any_candidate_upper_bound: bool = False
    all_wrong_flag: bool = False
    answer_consistent_disagreement_flag: bool = False
    ccr_specific_metadata: Dict[str, Any] = field(default_factory=dict)


# Backward-compatible aliases for shared code paths.
CandidateView = CCRCandidateView
SlateInteractionState = CCRState
RepairedCompareScaffold = CCRContext

__all__ = [
    "CCRCandidateView",
    "CCRContext",
    "CCRContextPayload",
    "CCRState",
    "CandidateView",
    "SlateInteractionState",
    "RepairedCompareScaffold",
    "FrozenCandidateRecord",
    "JudgeMethodResult",
    "CaseEvaluationRow",
]
