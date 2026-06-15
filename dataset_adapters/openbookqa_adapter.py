from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from dataset_adapters.base_adapter import AdapterRecord, DatasetAdapter
from my_datasets.openbookqa_dataset import OpenBookQADataset
from paper_repair.eval.parsing import (
    extract_mmlu_choice_letter,
    normalize_choice,
    parse_candidate_output,
    resolve_selected_agent_id,
)


class OpenBookQAAdapter(DatasetAdapter):
    dataset_name = "openbookqa"
    # OpenBookQA is a four-option multiple-choice benchmark. CCR_JUDGE's Ori final
    # selector has structured A/B/C/D selection handling under the MMLU domain,
    # so this adapter deliberately reuses that execution domain while keeping
    # OpenBookQA-specific loading, ids, and scoring.
    domain = "mmlu"

    def __init__(
        self,
        *,
        dataset_root: str = "data/openbookqa",
        config_name: str = "main",
        include_fact: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_root=dataset_root,
            config_name=config_name,
            include_fact=include_fact,
            **kwargs,
        )
        self.dataset_root = str(dataset_root)
        self.config_name = str(config_name)
        self.include_fact = bool(self.config_name == "additional") if include_fact is None else bool(include_fact)

    def official_scorer_mode(self) -> str:
        return "openbookqa_choice_match"

    def load_online_dataset(
        self,
        *,
        split: str,
        question_indices: Sequence[int] | None = None,
        limit_questions: int | None = None,
    ):
        return OpenBookQADataset(
            dataset_root=self.dataset_root,
            config_name=self.config_name,
            split=split,
            include_fact=self.include_fact,
            question_indices=question_indices,
            limit_questions=limit_questions,
        )

    def load_records(
        self,
        *,
        split: str,
        question_indices: Sequence[int],
    ):
        dataset = OpenBookQADataset(
            dataset_root=self.dataset_root,
            config_name=self.config_name,
            split=split,
            include_fact=self.include_fact,
        )
        records = []
        for index in question_indices:
            raw_record = dataset[int(index)]
            raw_id = str(raw_record.get("id") or int(index)).strip()
            records.append(
                AdapterRecord(
                    record_index=int(index),
                    task_id=f"openbookqa_{self.config_name}_{split}_{raw_id}",
                    question_text=dataset.record_to_input(raw_record)["task"],
                    gold_answer=self.normalize_gold_answer(dataset.record_to_target_answer(raw_record)),
                    reference_reasoning=dataset.record_to_reasoning(raw_record),
                    raw_record=raw_record,
                )
            )
        return records

    def normalize_gold_answer(self, answer: Any) -> str:
        return normalize_choice(answer) or "A"

    def normalize_prediction_answer(self, answer: Any) -> str | None:
        return extract_mmlu_choice_letter(answer) or normalize_choice(answer)

    def parse_candidate_output(self, text: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        parsed = parse_candidate_output(text)
        normalized_answer = normalize_choice(parsed.get("normalized_answer"))
        return {
            **parsed,
            "normalized_answer": normalized_answer,
            "final_answer": normalized_answer,
        }

    def selected_answer_for_agent(self, candidates: Iterable[Any], agent_id: Any) -> str | None:
        normalized_id = None if agent_id is None else str(agent_id)
        for candidate in candidates:
            candidate_id = str(
                candidate.get("agent_id")
                if isinstance(candidate, dict)
                else getattr(candidate, "agent_id", None)
            )
            if candidate_id != normalized_id:
                continue
            raw_text = (
                candidate.get("text", candidate.get("output"))
                if isinstance(candidate, dict)
                else getattr(candidate, "text", getattr(candidate, "output", ""))
            )
            parsed = self.parse_candidate_output(raw_text)
            return normalize_choice(parsed.get("normalized_answer"))
        return None

    def resolve_selected_agent_id(
        self,
        *,
        judge_text: Any,
        candidates: Sequence[Any],
        metadata_selected_agent_id: Any = None,
    ) -> str | None:
        return resolve_selected_agent_id(
            judge_text=judge_text,
            candidates=candidates,
            metadata_selected_agent_id=metadata_selected_agent_id,
        )

    def compute_task_metric(self, predicted: Any, target: Any) -> bool:
        return self.normalize_prediction_answer(predicted) == self.normalize_gold_answer(target)

    def build_dense_compare_messages(
        self,
        *,
        question_text: str,
        candidate_agent_ids: Sequence[str],
        candidate_blocks: str,
        retry_previous_output: str | None = None,
    ) -> list[Dict[str, str]]:
        allowed = ", ".join(str(item) for item in candidate_agent_ids)
        if retry_previous_output is None:
            system_prompt = (
                "You are the strict dense reference judge for an OpenBookQA benchmark.\n"
                "Choose exactly one candidate agent.\n"
                "Output exactly two lines and nothing else."
            )
            user_prompt = "\n".join(
                [
                    "Question:",
                    question_text,
                    "",
                    "Candidates:",
                    candidate_blocks,
                    "",
                    "Output format:",
                    f"Selected agent id: <id from {allowed}>",
                    "Answer: <A/B/C/D>",
                ]
            )
        else:
            previous_preview = str(retry_previous_output or "").strip()
            if len(previous_preview) > 240:
                previous_preview = previous_preview[:237].rstrip() + "..."
            system_prompt = (
                "Your previous answer was not parseable.\n"
                "Repair it by outputting exactly two lines and nothing else."
            )
            user_prompt = "\n".join(
                [
                    "Previous invalid output:",
                    previous_preview or "<empty>",
                    "",
                    "Question:",
                    question_text,
                    "",
                    "Candidates:",
                    candidate_blocks,
                    "",
                    "You must output exactly:",
                    f"Selected agent id: <id from {allowed}>",
                    "Answer: <A/B/C/D>",
                    "No explanation.",
                ]
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
