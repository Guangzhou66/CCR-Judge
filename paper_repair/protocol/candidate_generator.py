from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry


def _console_debug_enabled() -> bool:
    value = os.environ.get("KVCOMM_PROGRESS_ONLY", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _paper_debug_import(message: str) -> None:
    path_value = os.environ.get("PAPER_REPAIR_DEBUG_LOG", "").strip()
    if not path_value:
        return
    line = f"[IMPORT CHAIN] {message}"
    if _console_debug_enabled():
        print(line, flush=True)
    try:
        with open(path_value, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def ensure_prompt_set_loaded(domain: str) -> None:
    normalized = str(domain).strip().lower()
    if normalized == "mmlu":
        _paper_debug_import("before importing mmlu_prompt_set")
        import KVCOMM.prompt.mmlu_prompt_set  # noqa: F401
        _paper_debug_import("imported mmlu_prompt_set")
        return
    if normalized == "gsm8k":
        import KVCOMM.prompt.gsm8k_prompt_set  # noqa: F401
        return
    if normalized == "humaneval":
        import KVCOMM.prompt.humaneval_prompt_set  # noqa: F401
        return
    raise ValueError(f"Unsupported prompt-set domain: {domain}")


class PaperCandidateGenerator:
    def __init__(
        self,
        *,
        agent_id: str,
        role: str,
        domain: str,
        llm_name: str,
        temperature: float | None,
    ) -> None:
        _paper_debug_import("before importing gpt_chat")
        import KVCOMM.llm.gpt_chat  # noqa: F401
        _paper_debug_import("imported gpt_chat")
        ensure_prompt_set_loaded(domain)
        self.id = str(agent_id)
        self.role = str(role)
        self.domain = str(domain)
        self.llm = LLMRegistry.get(
            llm_name,
            prefix="",
            llm_config=KVCommConfig.from_env(),
        )
        self.llm.set_id(self.id, self.role)
        self.prompt_set = PromptSetRegistry.get(self.domain)
        self.temperature = temperature
        if self.domain == "mmlu" and hasattr(self.prompt_set, "get_analyze_constraint"):
            self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        else:
            self.constraint = self.prompt_set.get_constraint(self.role)

    def _answer_prompt(self, question_text: str) -> str:
        if self.domain == "gsm8k":
            return self.prompt_set.get_answer_prompt(question_text, role=self.role)
        return self.prompt_set.get_answer_prompt(question_text)

    def _peer_block(self, prior_candidates: Sequence[Dict[str, Any]]) -> str:
        if not prior_candidates:
            return ""
        lines = []
        for candidate in prior_candidates:
            lines.append(
                f"Agent {candidate['agent_id']} as a {candidate['role']} responded:\n{candidate['text']}"
            )
        return "\n\n".join(lines)

    def build_messages(
        self,
        *,
        question_text: str,
        prior_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        if self.domain == "mmlu":
            user_prompt = f"The task is: {question_text}\n"
        elif self.domain == "humaneval":
            user_prompt = self._answer_prompt(question_text)
        else:
            user_prompt = self._answer_prompt(question_text)
        peer_block = self._peer_block(prior_candidates)
        if peer_block:
            user_prompt += (
                "\n\nAt the same time, the outputs of other agents are as follows:\n\n"
                f"{peer_block}\n"
            )
        return [
            {"role": "system", "content": str(self.constraint)},
            {"role": "user", "content": str(user_prompt)},
        ]

    async def generate(
        self,
        *,
        question_text: str,
        prior_candidates: Sequence[Dict[str, Any]] | None = None,
    ):
        messages = self.build_messages(
            question_text=question_text,
            prior_candidates=list(prior_candidates or []),
        )
        return await self.llm.agen(
            messages,
            temperature=self.temperature,
            agent_id=self.id,
            agent_name="PaperCandidateGenerator",
            agent_role=self.role,
        )
