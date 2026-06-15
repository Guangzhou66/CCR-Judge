from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dataset_adapters.base_adapter import AdapterRecord, DatasetAdapter
from my_datasets.humaneval_dataset import HumanEvalDataset
from paper_repair.eval.parsing import (
    extract_selected_agent_ids,
    infer_selected_agent_id_from_text,
    normalize_selected_agent_id,
)


_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)\n```",
    flags=re.DOTALL | re.IGNORECASE,
)
_GENERIC_CODE_BLOCK_RE = re.compile(
    r"```[\w+-]*\s*\n(.*?)\n```",
    flags=re.DOTALL,
)
_SELECTED_HEADER_RE = re.compile(r"^\s*Selected agent id(?:\s*\(\s*(?:dense|masked)\s*\))?\s*[:|]?\s*.*$", re.IGNORECASE)
_CODE_HASH_RE = re.compile(r"^CODE\[[0-9a-f]{12}\]$")
DEFAULT_HUMANEVAL_JSON = str(
    Path(__file__).resolve().parents[1] / "my_datasets" / "humaneval" / "humaneval-py.jsonl"
)


def _strip_selected_header(text: str) -> str:
    lines = str(text or "").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if not _SELECTED_HEADER_RE.match(line):
            break
        idx += 1
    return "\n".join(lines[idx:]).strip()


def _clean_code_text(text: str) -> str:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = [line.rstrip() for line in lines]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _code_label(code: str) -> str:
    digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
    return f"CODE[{digest}]"


def _classify_failure(output: str) -> str:
    text = str(output or "")
    if "SyntaxError" in text or "IndentationError" in text:
        return "syntax_error"
    if "AssertionError" in text:
        return "assertion_failure"
    return "runtime_error"


def _classify_pyexecutor_failure(output: str) -> str:
    return _classify_failure(output)


def _render_test_feedback(test_block: str, output: str) -> str:
    rendered_output = str(output or "").strip() or "Unknown execution failure."
    return "Tests passed:\n\n\nTests failed:\n" + str(test_block or "").strip() + f"\n # output: {rendered_output}"


class HumanEvalAdapter(DatasetAdapter):
    dataset_name = "humaneval"
    domain = "humaneval"

    def official_scorer_mode(self) -> str:
        return "ori_pyexecutor"

    def __init__(
        self,
        *,
        dataset_json: str = DEFAULT_HUMANEVAL_JSON,
        execution_timeout_sec: float = 5.0,
        **kwargs,
    ):
        super().__init__(dataset_json=dataset_json, execution_timeout_sec=execution_timeout_sec, **kwargs)
        self.dataset_json = str(dataset_json)
        self.execution_timeout_sec = float(execution_timeout_sec)

    def load_online_dataset(
        self,
        *,
        split: str,
        question_indices: Sequence[int] | None = None,
        limit_questions: int | None = None,
    ):
        return HumanEvalDataset(
            dataset_json=self.dataset_json,
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
        dataset = HumanEvalDataset(
            dataset_json=self.dataset_json,
            split=split,
        )
        records = []
        for index in question_indices:
            raw_record = dataset[int(index)]
            target = dataset.record_to_target_answer(raw_record)
            records.append(
                AdapterRecord(
                    record_index=int(index),
                    task_id=str(target.get("task_id") or f"humaneval_{int(index)}"),
                    question_text=dataset.record_to_input(raw_record)["task"],
                    gold_answer=str(target.get("entry_point") or ""),
                    reference_reasoning=str(target.get("canonical_solution") or ""),
                    raw_record=raw_record,
                )
            )
        return records

    def extract_code_block(self, text: Any, entry_point: str | None = None) -> str:
        raw = _strip_selected_header(text if isinstance(text, str) else str(text or ""))
        if not raw.strip():
            return ""

        python_blocks = [match.group(1).strip() for match in _CODE_BLOCK_RE.finditer(raw)]
        generic_blocks = [match.group(1).strip() for match in _GENERIC_CODE_BLOCK_RE.finditer(raw)]
        candidates = python_blocks or generic_blocks

        if entry_point:
            for block in candidates:
                if f"def {entry_point}" in block or f"class {entry_point}" in block:
                    return _clean_code_text(block)

        if candidates:
            return _clean_code_text(candidates[0])

        return _clean_code_text(raw)

    def canonicalize_code_text(self, text: Any, entry_point: str | None = None) -> str:
        return _clean_code_text(self.extract_code_block(text, entry_point=entry_point))

    def normalize_gold_answer(self, answer: Any) -> str:
        if isinstance(answer, dict):
            canonical = self.canonicalize_code_text(
                answer.get("canonical_solution") or "",
                entry_point=str(answer.get("entry_point") or ""),
            )
            if canonical:
                return _code_label(canonical)
        canonical = self.canonicalize_code_text(answer)
        return _code_label(canonical) if canonical else "CODE[empty]"

    def normalize_prediction_answer(self, answer: Any) -> str | None:
        if isinstance(answer, str):
            stripped = answer.strip()
            if _CODE_HASH_RE.match(stripped):
                return stripped
        canonical = self.canonicalize_code_text(answer)
        return _code_label(canonical) if canonical else None

    def parse_candidate_output(self, text: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raw_text = text if isinstance(text, str) else str(text or "")
        canonical_code = self.canonicalize_code_text(raw_text)
        normalized_answer = self.normalize_prediction_answer(canonical_code)
        code_lines = [line for line in canonical_code.splitlines() if line.strip()]
        preview = code_lines[0] if code_lines else "No code extracted."
        evidence = code_lines[1] if len(code_lines) > 1 else preview
        return {
            "text": raw_text,
            "final_answer": canonical_code,
            "normalized_answer": normalized_answer,
            "conclusion": preview[:160],
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

    def evaluate_prediction(self, predicted: Any, target: Any) -> Dict[str, Any]:
        target_payload = dict(target or {}) if isinstance(target, dict) else {}
        entry_point = str(target_payload.get("entry_point") or "").strip()
        tests = str(target_payload.get("test") or "")
        code = self.canonicalize_code_text(predicted, entry_point=entry_point)
        normalized = self.normalize_prediction_answer(code)
        if not code:
            return {
                "passed": False,
                "failure_type": "missing_code",
                "error_message": "No Python code could be extracted from the prediction.",
                "normalized_answer": normalized,
                "code_text": "",
            }
        if not tests.strip():
            return {
                "passed": False,
                "failure_type": "missing_tests",
                "error_message": "No HumanEval test payload was provided.",
                "normalized_answer": normalized,
                "code_text": code,
                "official_humaneval_passed": False,
                "subprocess_humaneval_passed": False,
                "ori_pyexecutor_passed": False,
                "current_subprocess_passed": False,
                "scorer_mode": "ori_pyexecutor",
                "extracted_code": code,
                "execution_error_type": "missing_tests",
            }

        def _run_script(script: str) -> Dict[str, Any]:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "",
            }
            with tempfile.TemporaryDirectory(prefix="humaneval_exec_") as tmpdir:
                script_path = Path(tmpdir) / "candidate.py"
                script_path.write_text(script, encoding="utf-8")
                try:
                    completed = subprocess.run(
                        [sys.executable, "-I", str(script_path)],
                        cwd=tmpdir,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=self.execution_timeout_sec,
                    )
                except subprocess.TimeoutExpired as exc:
                    return {
                        "passed": False,
                        "timed_out": True,
                        "output": str(exc),
                    }
            combined_output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            return {
                "passed": completed.returncode == 0,
                "timed_out": False,
                "output": combined_output,
            }

        # HumanEval official scoring should run in a fresh isolated process.
        # Reusing a long-lived Python sandbox can leak state across candidates
        # and caused spurious timeouts in the formal Ori runs.
        official_script = (
            "from typing import *\n\n"
            + code
            + "\n\n"
            + tests
            + "\n\n"
            + "if __name__ == '__main__':\n"
            + "    pass\n"
        )
        official_result = _run_script(official_script)
        official_passed = bool(official_result.get("passed"))
        official_feedback = "" if official_passed else _render_test_feedback(tests, official_result.get("output"))
        official_failure = None if official_passed else _classify_pyexecutor_failure(official_feedback)

        script = (
            code
            + "\n\n"
            + tests
            + "\n\n"
            + "if __name__ == '__main__':\n"
            + "    pass\n"
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        }
        with tempfile.TemporaryDirectory(prefix="humaneval_exec_") as tmpdir:
            script_path = Path(tmpdir) / "candidate.py"
            script_path.write_text(script, encoding="utf-8")
            try:
                completed = subprocess.run(
                    ["python", "-I", str(script_path)],
                    cwd=tmpdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "passed": bool(official_passed),
                    "failure_type": official_failure,
                    "error_message": None if official_passed else official_feedback[:2000],
                    "normalized_answer": normalized,
                    "code_text": code,
                    "official_humaneval_passed": bool(official_passed),
                    "subprocess_humaneval_passed": False,
                    "ori_pyexecutor_passed": bool(official_passed),
                    "current_subprocess_passed": False,
                    "scorer_mode": "ori_pyexecutor",
                    "extracted_code": code,
                    "execution_error_type": official_failure or "timeout",
                    "subprocess_execution_error_type": "timeout",
                    "subprocess_error_message": str(exc),
                }

        combined_output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        subprocess_passed = completed.returncode == 0
        return {
            "passed": bool(official_passed),
            "failure_type": official_failure,
            "error_message": None if official_passed else official_feedback[:2000],
            "normalized_answer": normalized,
            "code_text": code,
            "official_humaneval_passed": bool(official_passed),
            "subprocess_humaneval_passed": bool(subprocess_passed),
            "ori_pyexecutor_passed": bool(official_passed),
            "current_subprocess_passed": bool(subprocess_passed),
            "scorer_mode": "ori_pyexecutor",
            "extracted_code": code,
            "execution_error_type": official_failure,
            "subprocess_execution_error_type": (None if subprocess_passed else _classify_failure(combined_output)),
            "subprocess_error_message": None if subprocess_passed else (combined_output[:2000] if combined_output else "Unknown execution failure."),
        }

    def compute_task_metric(self, predicted: Any, target: Any) -> bool:
        return bool(self.evaluate_prediction(predicted, target).get("passed"))

    def answer_consistency_match(self, left: Any, right: Any) -> bool:
        left_norm = self.normalize_prediction_answer(left)
        right_norm = self.normalize_prediction_answer(right)
        return left_norm is not None and left_norm == right_norm

    async def score_official(
        self,
        *,
        final_answer: Any,
        target: Any,
        selected_execution_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        evaluation = self.evaluate_prediction(final_answer, target)
        return {
            "official_is_correct": bool(evaluation.get("official_humaneval_passed")),
            "scoring_branch": "ori_pyexecutor_final_code",
            "fallback_used": False,
            "selected_candidate_passed": (
                None if not isinstance(selected_execution_result, dict) else bool(selected_execution_result.get("passed"))
            ),
            "fallback_correctness_result": None,
            "scorer_mode": "ori_pyexecutor",
            "official_humaneval_passed": bool(evaluation.get("official_humaneval_passed")),
            "subprocess_humaneval_passed": bool(evaluation.get("subprocess_humaneval_passed")),
            "ori_pyexecutor_passed": bool(evaluation.get("ori_pyexecutor_passed")),
            "current_subprocess_passed": bool(evaluation.get("current_subprocess_passed")),
            "extracted_code": evaluation.get("extracted_code"),
            "execution_error_type": evaluation.get("execution_error_type"),
        }

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
                "You are the strict dense reference judge for a HumanEval benchmark.\n"
                "Choose exactly one candidate agent.\n"
                "Output exactly: a selected agent id line followed by one Python code block."
            )
            user_prompt = "\n".join(
                [
                    "Task:",
                    question_text,
                    "",
                    "Candidate implementations:",
                    candidate_blocks,
                    "",
                    "Output format:",
                    f"Selected agent id: <id from {allowed}>",
                    "```python",
                    "<selected code>",
                    "```",
                ]
            )
        else:
            previous_preview = str(retry_previous_output or "").strip()
            if len(previous_preview) > 400:
                previous_preview = previous_preview[:397].rstrip() + "..."
            system_prompt = (
                "Your previous answer was not parseable.\n"
                "Repair it by outputting exactly one selected agent id line and one Python code block."
            )
            user_prompt = "\n".join(
                [
                    "Previous invalid output:",
                    previous_preview or "<empty>",
                    "",
                    "Task:",
                    question_text,
                    "",
                    "Candidate implementations:",
                    candidate_blocks,
                    "",
                    "You must output exactly:",
                    f"Selected agent id: <id from {allowed}>",
                    "```python",
                    "<selected code>",
                    "```",
                ]
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def render_dense_candidate_block(
        self,
        *,
        agent_id: str,
        parsed_candidate: Dict[str, Any],
        raw_output: str,
    ) -> str:
        code = str(parsed_candidate.get("final_answer") or "").strip()
        code_block = code or self.extract_code_block(raw_output)
        return "\n".join(
            [
                f"Agent {agent_id}",
                "```python",
                code_block,
                "```",
            ]
        ).strip()

    def format_selected_result_text(
        self,
        *,
        selected_agent_id: str,
        selected_answer: str,
    ) -> str:
        code = self.canonicalize_code_text(selected_answer)
        if not code:
            return f"Selected agent id: {selected_agent_id}"
        return "\n".join(
            [
                f"Selected agent id: {selected_agent_id}",
                "```python",
                code,
                "```",
            ]
        )
