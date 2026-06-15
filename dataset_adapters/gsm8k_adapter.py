from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch

from dataset_adapters.base_adapter import AdapterRecord, DatasetAdapter
from my_datasets.gsm8k_dataset import GSM8KDataset
from KVCOMM.llm import LLMChat
from paper_repair.eval.parsing import (
    extract_selected_agent_ids,
    infer_selected_agent_id_from_text,
    normalize_selected_agent_id,
)


_FINAL_ANSWER_PATTERNS = [
    re.compile(r"####\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer(?:\s*[:=]\s*|\s+)([^\n]+)", re.IGNORECASE),
    re.compile(r"(?:the\s+)?answer(?:\s*is|\s*[:=]\s*|\s+)([^\n]+)", re.IGNORECASE),
]
_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_decimal_string(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = raw.replace(",", "").replace("$", "").replace(" ", "")
    raw = raw.rstrip(".")
    if not raw:
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _extract_numeric_answer(text: Any) -> str | None:
    if text is None:
        return None
    answer_text = str(text).strip()
    if not answer_text:
        return None
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(answer_text)
        if matches:
            candidate = str(matches[-1]).strip()
            numbers = _NUMBER_PATTERN.findall(candidate)
            if numbers:
                return _normalize_decimal_string(numbers[-1])
    for line in reversed(lines):
        for pattern in _FINAL_ANSWER_PATTERNS[1:]:
            match = pattern.search(line)
            if match:
                numbers = _NUMBER_PATTERN.findall(match.group(1))
                if numbers:
                    return _normalize_decimal_string(numbers[-1])
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if lines:
        numbers = _NUMBER_PATTERN.findall(lines[-1])
        if numbers:
            return _normalize_decimal_string(numbers[-1])
    numbers = _NUMBER_PATTERN.findall(answer_text)
    if numbers:
        return _normalize_decimal_string(numbers[-1])
    return None


def _extract_ori_parity_number_sync(answer_text: str) -> str:
    answer = str(answer_text or "")
    tokenizer = LLMChat._shared_tokenizer
    model = LLMChat._shared_model
    if tokenizer is None or model is None:
        normalized = _extract_numeric_answer(answer)
        return normalized or "0"

    messages = [
        {
            "role": "system",
            "content": (
                "You are professional in extracting the exact value from the user response. "
                "Given the following user response, you should only output the value that is argued "
                "as the most correct answer by the user for evaluation, i.e., only an integer or "
                "float-type value in one line."
            ),
        },
        {"role": "user", "content": answer},
    ]
    token_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": token_inputs.to(model.device),
        "attention_mask": torch.ones_like(token_inputs).to(model.device),
    }
    prompt_length = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "return_dict_in_generate": True,
        "return_legacy_cache": False,
    }
    outputs = model.generate(**inputs, **generate_kwargs)
    generated = outputs.sequences[:, prompt_length:]
    response_message = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    normalized = _extract_numeric_answer(response_message)
    return normalized or "0"


def _split_gold_answer(raw_answer: str) -> tuple[str, str]:
    text = str(raw_answer or "")
    lines = text.splitlines()
    final_answer = None
    reasoning_lines: List[str] = []
    for line in lines:
        if line.strip().startswith("####"):
            extracted = _extract_numeric_answer(line)
            if extracted is not None:
                final_answer = extracted
            continue
        reasoning_lines.append(line)
    if final_answer is None:
        final_answer = _extract_numeric_answer(text)
    return "\n".join(reasoning_lines).strip(), (final_answer or "0")


class GSM8KAdapter(DatasetAdapter):
    dataset_name = "gsm8k"
    domain = "gsm8k"

    def official_scorer_mode(self) -> str:
        return "ori_parity"

    def __init__(
        self,
        *,
        dataset_root: str = "data/gsm8k",
        config_name: str = "main",
        **kwargs,
    ):
        super().__init__(dataset_root=dataset_root, config_name=config_name, **kwargs)
        self.dataset_root = str(dataset_root)
        self.config_name = str(config_name)

    def load_online_dataset(
        self,
        *,
        split: str,
        question_indices: Sequence[int] | None = None,
        limit_questions: int | None = None,
    ):
        return GSM8KDataset(
            dataset_root=self.dataset_root,
            config_name=self.config_name,
            split=split,
            question_indices=question_indices,
            limit_questions=limit_questions,
        )

    def load_records(
        self,
        *,
        split: str,
        question_indices: Sequence[int],
    ):
        dataset = GSM8KDataset(
            dataset_root=self.dataset_root,
            config_name=self.config_name,
            split=split,
        )
        records = []
        for index in question_indices:
            raw_record = dataset[int(index)]
            records.append(
                AdapterRecord(
                    record_index=int(index),
                    task_id=f"gsm8k_{self.config_name}_{split}_{int(index)}",
                    question_text=dataset.record_to_input(raw_record)["task"],
                    gold_answer=self.normalize_gold_answer(dataset.record_to_target_answer(raw_record)),
                    reference_reasoning=dataset.record_to_reasoning(raw_record),
                    raw_record=raw_record,
                )
            )
        return records

    def normalize_gold_answer(self, answer: Any) -> str:
        normalized = _extract_numeric_answer(answer)
        return normalized or "0"

    def normalize_prediction_answer(self, answer: Any) -> str | None:
        return _extract_numeric_answer(answer)

    async def score_official(
        self,
        *,
        final_answer: Any,
        target: Any,
        selected_execution_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        final_text = "" if final_answer is None else str(final_answer)
        official_number = _extract_ori_parity_number_sync(final_text)
        diagnostic_number = self.normalize_prediction_answer(final_text)
        target_norm = self.normalize_gold_answer(target)
        try:
            official_correct = float(official_number) == float(target_norm)
        except Exception:
            official_correct = False
        return {
            "official_is_correct": bool(official_correct),
            "scoring_branch": "ori_parity_final_text",
            "fallback_used": False,
            "selected_candidate_passed": (
                None if not isinstance(selected_execution_result, dict) else bool(selected_execution_result.get("passed"))
            ),
            "fallback_correctness_result": None,
            "scorer_mode": "ori_parity",
            "official_gsm8k_number": official_number,
            "diagnostic_gsm8k_number": diagnostic_number,
            "ori_parity_number": official_number,
            "current_normalized_number": diagnostic_number,
        }

    def parse_candidate_output(self, text: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raw_text = text if isinstance(text, str) else str(text or "")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        normalized_answer = self.normalize_prediction_answer(raw_text)
        conclusion = lines[0] if lines else "Not stated."
        evidence = lines[1] if len(lines) > 1 else conclusion
        return {
            "text": raw_text,
            "final_answer": normalized_answer,
            "normalized_answer": normalized_answer,
            "conclusion": conclusion[:160],
            "evidence": evidence[:160],
            "strongest_alternative_considered": "NONE",
            "strongest_alternative_option": "NONE",
            "alternative_supported_flag": "NO",
            "alternative_supported_yes": False,
            "rejected_option": "NONE",
            "why_not_that_option": "Not stated.",
            "why_not_rejected_option": "Not stated.",
            "alternative_overturn_flag": "NO",
            "alternative_overturn_yes": False,
        }

    def resolve_selected_agent_id(
        self,
        *,
        judge_text: Any,
        candidates: Sequence[Any],
        metadata_selected_agent_id: Any = None,
    ) -> str | None:
        candidate_ids = [
            str(candidate.get("agent_id") if isinstance(candidate, dict) else getattr(candidate, "agent_id"))
            for candidate in candidates
        ]
        metadata_value = normalize_selected_agent_id(
            metadata_selected_agent_id,
            allowed_ids=candidate_ids,
        )
        if metadata_value is not None:
            return metadata_value
        header_value = normalize_selected_agent_id(
            extract_selected_agent_ids(judge_text).get("selected_agent_id"),
            allowed_ids=candidate_ids,
        )
        if header_value is not None:
            return header_value
        explicit = infer_selected_agent_id_from_text(judge_text, candidate_ids)
        return normalize_selected_agent_id(explicit, allowed_ids=candidate_ids)

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
                "You are the strict dense reference judge for a GSM8K benchmark.\n"
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
                    "Final Answer: <number>",
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
                    "Final Answer: <number>",
                    "No explanation.",
                ]
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
