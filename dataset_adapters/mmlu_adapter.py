from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from dataset_adapters.base_adapter import AdapterRecord, DatasetAdapter
from my_datasets.MMLU.download import download
from my_datasets.mmlu_dataset import MMLUDataset
from paper_repair.eval.parsing import (
    extract_mmlu_choice_letter,
    normalize_choice,
    parse_candidate_output,
    resolve_selected_agent_id,
)


class MMLUAdapter(DatasetAdapter):
    dataset_name = "mmlu"
    domain = "mmlu"

    def official_scorer_mode(self) -> str:
        return "mmlu_text_match_parity_aligned"

    def load_online_dataset(
        self,
        *,
        split: str,
        question_indices: Sequence[int] | None = None,
        limit_questions: int | None = None,
    ):
        download()
        return MMLUDataset(split)

    def load_records(
        self,
        *,
        split: str,
        question_indices: Sequence[int],
    ):
        download()
        dataset = MMLUDataset(split)
        records = []
        for index in question_indices:
            raw_record = dataset[int(index)]
            records.append(
                AdapterRecord(
                    record_index=int(index),
                    task_id=f"mmlu_{split}_{int(index)}",
                    question_text=dataset.record_to_input(raw_record)["task"],
                    gold_answer=self.normalize_gold_answer(dataset.record_to_target_answer(raw_record)),
                    reference_reasoning="",
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
        target_norm = self.normalize_gold_answer(target)
        predicted_choice = extract_mmlu_choice_letter(predicted)
        if predicted_choice is None:
            predicted_choice = normalize_choice(predicted)
        return predicted_choice == target_norm

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
                "You are the strict dense reference judge for an MMLU benchmark.\n"
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
