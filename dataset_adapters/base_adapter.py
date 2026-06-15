from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence
import asyncio


@dataclass(frozen=True)
class AdapterRecord:
    record_index: int
    task_id: str
    question_text: str
    gold_answer: str
    reference_reasoning: str = ""
    raw_record: Any = None


class DatasetAdapter(ABC):
    dataset_name: str
    domain: str

    def __init__(self, **kwargs):
        self.config = dict(kwargs)

    @abstractmethod
    def load_online_dataset(
        self,
        *,
        split: str,
        question_indices: Sequence[int] | None = None,
        limit_questions: int | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def load_records(
        self,
        *,
        split: str,
        question_indices: Sequence[int],
    ) -> List[AdapterRecord]:
        raise NotImplementedError

    @abstractmethod
    def normalize_gold_answer(self, answer: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def normalize_prediction_answer(self, answer: Any) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def parse_candidate_output(self, text: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resolve_selected_agent_id(
        self,
        *,
        judge_text: Any,
        candidates: Sequence[Any],
        metadata_selected_agent_id: Any = None,
    ) -> str | None:
        raise NotImplementedError

    def selected_answer_for_agent(self, candidates: Iterable[Any], agent_id: Any) -> str | None:
        normalized_id = None if agent_id is None else str(agent_id)
        for candidate in candidates:
            candidate_id = str(getattr(candidate, "agent_id", None) if not isinstance(candidate, dict) else candidate.get("agent_id"))
            if candidate_id != normalized_id:
                continue
            if isinstance(candidate, dict):
                raw_text = candidate.get("text", candidate.get("output"))
                metadata = dict(candidate.get("metadata") or {})
            else:
                raw_text = getattr(candidate, "text", getattr(candidate, "output", None))
                metadata = dict(getattr(candidate, "metadata", {}) or {})
            if raw_text is not None:
                try:
                    parsed = self.parse_candidate_output(raw_text, metadata)
                    for key in ("final_answer", "normalized_answer"):
                        value = parsed.get(key)
                        if value is not None and str(value).strip():
                            return str(value).strip()
                except Exception:
                    pass
            if isinstance(candidate, dict):
                value = candidate.get("final_answer", candidate.get("normalized_answer"))
            else:
                value = getattr(candidate, "final_answer", getattr(candidate, "normalized_answer", None))
            if value is not None and str(value).strip():
                return str(value).strip()
            return None
        return None

    def compute_task_metric(self, predicted: Any, target: Any) -> bool:
        predicted_norm = self.normalize_prediction_answer(predicted)
        target_norm = self.normalize_gold_answer(target)
        return predicted_norm is not None and predicted_norm == target_norm

    def answer_consistency_match(self, left: Any, right: Any) -> bool:
        left_norm = self.normalize_prediction_answer(left)
        right_norm = self.normalize_prediction_answer(right)
        return left_norm is not None and left_norm == right_norm

    def evaluate_prediction(self, predicted: Any, target: Any) -> Dict[str, Any]:
        normalized_prediction = self.normalize_prediction_answer(predicted)
        normalized_target = self.normalize_gold_answer(target)
        passed = bool(normalized_prediction is not None and normalized_prediction == normalized_target)
        return {
            "passed": passed,
            "failure_type": None if passed else "mismatch",
            "error_message": None,
            "normalized_answer": normalized_prediction,
            "target_answer": normalized_target,
        }

    async def score_official(
        self,
        *,
        final_answer: Any,
        target: Any,
        selected_execution_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if isinstance(selected_execution_result, dict):
            selected_passed = bool(selected_execution_result.get("passed"))
            return {
                "official_is_correct": selected_passed,
                "scoring_branch": "selected_candidate_passed",
                "fallback_used": False,
                "selected_candidate_passed": selected_passed,
                "fallback_correctness_result": None,
                "scorer_mode": self.official_scorer_mode(),
            }

        fallback_eval = self.evaluate_prediction(final_answer, target)
        return {
            "official_is_correct": bool(fallback_eval.get("passed")),
            "scoring_branch": "fallback_final_answer",
            "fallback_used": True,
            "selected_candidate_passed": None,
            "fallback_correctness_result": fallback_eval,
            "scorer_mode": self.official_scorer_mode(),
        }

    def online_benchmark_name(self) -> str:
        return f"{self.dataset_name}_ori_protocol"

    def paper_benchmark_name(self) -> str:
        return f"{self.dataset_name}_paper_repair"

    def official_scorer_mode(self) -> str:
        return "adapter_default"

    def build_solver_prompt(self, question_text: str) -> str:
        return question_text

    def build_final_select_prompt(
        self,
        *,
        question_text: str,
        candidates: Sequence[tuple[str, str]],
        prompt_set: Any,
    ) -> str:
        constraint = prompt_set.get_select_best_constraint([str(agent_id) for agent_id, _ in candidates])
        return prompt_set.get_select_best_agent_prompt(question_text, list(candidates), constraint)

    def build_dense_compare_messages(
        self,
        *,
        question_text: str,
        candidate_agent_ids: Sequence[str],
        candidate_blocks: str,
        retry_previous_output: str | None = None,
    ) -> list[Dict[str, str]]:
        raise NotImplementedError

    def render_dense_candidate_block(
        self,
        *,
        agent_id: str,
        parsed_candidate: Dict[str, Any],
        raw_output: str,
    ) -> str:
        answer = str(parsed_candidate.get("normalized_answer") or "UNKNOWN")
        rationale = " ".join(
            line
            for line in [
                str(parsed_candidate.get("conclusion") or ""),
                str(parsed_candidate.get("evidence") or ""),
            ]
            if line
        ).strip()
        if len(rationale) > 120:
            rationale = rationale[:117].rstrip() + "..."
        if rationale:
            return f"Agent {agent_id} | Answer: {answer} | Note: {rationale}"
        return f"Agent {agent_id} | Answer: {answer}"

    def format_selected_result_text(
        self,
        *,
        selected_agent_id: str,
        selected_answer: str,
    ) -> str:
        return f"Selected agent id: {selected_agent_id}\n{selected_answer}"
