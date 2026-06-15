import asyncio
import os
import random
import re
from typing import Any, Dict, List
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import execute_code_get_return, PyExecutor
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.metrics import GenerationResult
from KVCOMM.utils.log import logger

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    """Final code synthesis agent that integrates peers and executes tests."""
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.role = 'FinalWriteCode'
        self.llm.set_id(self.id, 'FinalWriteCode')
        self.prompt_set = PromptSetRegistry.get(domain)
        self._executor = PyExecutor()

    @staticmethod
    def extract_example(prompt: Dict[str, Any] | str) -> List[str]:
        """Return doctest-style assertions extracted from the task description."""
        if isinstance(prompt, dict):
            prompt_text = str(prompt.get("task", ""))
        else:
            prompt_text = str(prompt)

        lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
        results: List[str] = []
        iterator = iter(lines)
        for line in iterator:
            if not line.startswith(">>>"):
                continue
            function_call = line[4:].strip()
            expected_output = next(iterator, "").strip()
            if not function_call or not expected_output:
                continue
            results.append(f"assert {function_call} == {expected_output}")
        return results

    @staticmethod
    def _is_python_code_block(text: str) -> bool:
        text = text.strip()
        return text.startswith("```python") and text.endswith("```")

    @staticmethod
    def _extract_python_code(text: str) -> str:
        """Extract pure python code from a fenced block."""
        if not FinalWriteCode._is_python_code_block(text):
            return text
        content = text.strip()[len("```python") :].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        return content

    def _summarize_agent_outputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        internal_tests: List[str],
    ) -> str:
        """Summarize peer outputs, running tests on code blocks when present."""
        paragraphs: List[str] = []
        for agent_id, info in spatial_info.items():
            role = info.get("role", "agent")
            output = info.get("output")
            if not isinstance(output, str):
                paragraphs.append(f"Agent {agent_id} as a {role} returned invalid output.\n\n")
                continue
            if self._is_python_code_block(output):
                code = self._extract_python_code(output)
                is_solved, feedback, _ = self._executor.execute(code, internal_tests, timeout=10)
                paragraphs.append(
                    f"Agent {agent_id} as a {role}:\n\n"
                    f"The code written by the agent is:\n\n{output}\n\n"
                    f"Whether it passes internal testing? {is_solved}.\n\n"
                    f"The feedback is:\n\n{feedback}.\n\n"
                )
                continue
            paragraphs.append(
                f"Agent {agent_id} as a {role} provides the following info: {output}\n\n"
            )
        return "".join(paragraphs)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Any],
        temporal_info:Dict[str,Any],
        mode: str = "default",
        **kwargs,
    )->Dict[str, Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """

        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)
            prefix_text = kwargs.get("prefix", "")

            has_shared_prefix = (
                self.llm.has_prefix_initialized(self.id)
                and "placeholder_info" in agent_memory
            )
            early_response: str | None = None

            if has_shared_prefix:
                task = raw_inputs["task"]
                for agent_id, info in spatial_info.items():
                    cond_text: str | None = None
                    cond_prefix: str | None = None

                    if self.domain == "gsm8k" and info["role"] == "Programming Expert":
                        answer = execute_code_get_return(
                            info["output"].split("```python\n")[-1].split("\n```")[0]
                        )
                        if answer is None:
                            answer = "No variable is named answer."
                        cond_text = f"the answer is {answer}"
                        cond_prefix = "the answer is "
                    elif (
                        self.domain == "humaneval"
                        and self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        code = info["output"].split("```python\n")[-1].split("\n```")[0]
                        is_solved, feedback, _ = PyExecutor().execute(
                            code, getattr(self, "internal_tests", []), timeout=10
                        )
                        cond_text = (
                            "Whether it passes internal testing?\n"
                            f"{is_solved}.\n\nThe feedback is:\n\n {feedback}."
                        )
                        cond_prefix = "Whether it passes internal testing?\n"

                    if cond_text and cond_prefix:
                        self.llm.update_condition_anchor(
                            request_uid=request_uid,
                            owner_agent_id=agent_id,
                            message=task,
                            content=cond_text,
                            prefix_text=cond_prefix,
                        )

                user_content = prefix_text + raw_inputs["task"]
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs["task"],
                    user_content=user_content,
                    prefix_text=prefix_text,
                )
                logger.opt(colors=True).info(
                    "<green>[MODE]</green> Task: {} Agent {} ({}) mode: {}",
                    raw_inputs["task"],
                    self.id,
                    self.role,
                    preferred_mode,
                )
                return {
                    "preferred_mode": preferred_mode,
                    "early_response": early_response,
                }

            system_prompt = self.prompt_set.get_decision_role()
            self.constraint = self.prompt_set.get_decision_constraint()
            system_prompt = f"{system_prompt}.\n {self.constraint}"
            user_input = "{user_question}"
            spatial_str = ""
            decision_few_shot = self.prompt_set.get_decision_few_shot()

            spatial_str = ""
            if self.domain in {"gsm8k", "mmlu"}:
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if info["role"] == "Programming Expert":
                        agent_output += "\n the result is {condition_" + agent_id + "_current}"
                    spatial_str += (
                        f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                    )
            elif self.domain == "humaneval":
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if (
                        self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        condition = "{condition_" + agent_id + "_current}"
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                            f"{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                        )
                    else:
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                        )

            decision_few_shot = self.prompt_set.get_decision_few_shot()
            user_prompt = (
                f"{decision_few_shot} {prefix_text} {user_input}\n At the same time, the output of other agents is as follows:\n\n"
                f"{spatial_str}\n\n"
            )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {
                "preferred_mode": preferred_mode,
                "early_response": early_response,
            }

        system_prompt = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{system_prompt}.\n {self.constraint}"
        prefix_text = kwargs.get("prefix", "")
        spatial_str = ""
        if self.domain in {"gsm8k", "mmlu"}:
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    agent_output += f"\n the result is {answer}"
                spatial_str += (
                    f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                )
        elif self.domain == "humaneval":
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if (
                    self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                        f"{agent_output}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                    )
                else:
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                    )

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} {prefix_text} {raw_inputs['task']}\n At the same time, the output of other agents is as follows:\n\n"
            f"{spatial_str}\n\n"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "early_response": early_response,
        }

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode="default",
                **kwargs,
            )
        )
        message = [
            {"role": "system", "content": inputs["system_prompt"]},
            {"role": "user", "content": inputs["user_prompt"]},
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
            message = [
                {"role": "system", "content": inputs["system_prompt"]},
                {"role": "user", "content": inputs["user_prompt"]},
            ]
            result = await self.llm.agen(
                message,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            return result

        mode_data = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            mode=mode,
            **kwargs,
        )
        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input["task"],
            preferred_mode=mode_data["preferred_mode"],
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return input["task"], result


@AgentRegistry.register('FinalSelectBest')
class FinalSelectBest(Node):
    """Judge agent that selects the best answer among peer agents.

    It uses domain-specific `get_select_best` prompts (when available) so that
    the underlying LLM copies one of the candidate answers verbatim. The judge
    then optionally annotates which agent was selected while keeping evaluation
    formats for existing experiments intact.
    """

    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "FinalSelectBest", domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.role = "FinalSelectBest"
        self.llm.set_id(self.id, self.role)
        self.prompt_set = PromptSetRegistry.get(domain)

        shuffle_flag = os.getenv("KVCOMM_JUDGE_SHUFFLE", "0").lower()
        self.shuffle_candidates: bool = shuffle_flag in {"1", "true", "yes", "y"}

    def _get_agent_order(self, spatial_info: Dict[str, Any]) -> List[str]:
        """Return agent ids in either fixed or shuffled order."""
        agent_ids = sorted(spatial_info.keys())
        if self.shuffle_candidates:
            agent_ids = agent_ids.copy()
            random.shuffle(agent_ids)
        return agent_ids

    @staticmethod
    def _infer_selected_agent_id_from_text(text: str, candidate_agent_ids: List[str]) -> str | None:
        """Best-effort inference of which candidate agent was selected.

        Used when the judge does not copy a candidate answer verbatim.
        Returns a candidate agent id string, or None if not confidently inferred.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        explicit = re.search(r"Selected agent id\s*:\s*(\d+)", text, re.IGNORECASE)
        if explicit:
            selected = explicit.group(1)
            return selected if selected in candidate_agent_ids else None

        # If the judge claims all candidates are identical, selection is arbitrary.
        tie_patterns = [
            r"\ball\b.*\bagents\b.*\b(same|identical|consistent)\b",
            r"\ball\b.*\bcandidate answers\b.*\b(same|identical)\b",
            r"\ball\b.*\banswers\b.*\b(same|identical)\b",
        ]
        for pat in tie_patterns:
            if re.search(pat, text, re.IGNORECASE | re.DOTALL):
                return candidate_agent_ids[0] if candidate_agent_ids else None

        # Preference-ordered patterns (more specific first).
        patterns = [
            r"most reliable answer is provided by Agent\s*(\d+)",
            r"(?:choose|select|selected)\s+Agent\s*(\d+)",
            r"Agent\s*(\d+)\s+is\s+(?:the\s+)?(?:best|most reliable|most accurate|correct)",
            r"The best answer is from Agent\s*(\d+)",
            r"Based on the analysis,\s*Agent\s*(\d+)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                selected = match.group(1)
                return selected if selected in candidate_agent_ids else None

        return None

    @staticmethod
    def _extract_mmlu_choice_letter(text: Any) -> str | None:
        if not isinstance(text, str):
            return None
        stripped = text.strip()
        if not stripped:
            return None
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([ABCD])\b", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
            break
        return None

    def _infer_selected_agent_id_from_mmlu_choice(
        self,
        judge_text: str,
        candidate_agent_ids: List[str],
        spatial_info: Dict[str, Any],
    ) -> str | None:
        cleaned = self._strip_selected_agent_header(judge_text)
        judge_choice = self._extract_mmlu_choice_letter(cleaned)
        if judge_choice is None:
            return None

        matches: List[str] = []
        for agent_id in candidate_agent_ids:
            info = spatial_info.get(agent_id) or {}
            agent_text = info.get("output")
            agent_choice = self._extract_mmlu_choice_letter(agent_text)
            if agent_choice == judge_choice:
                matches.append(agent_id)

        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _strip_selected_agent_header(text: str) -> str:
        """Remove one or more leading `Selected agent id:` header lines."""
        if not isinstance(text, str):
            return str(text)
        lines = text.splitlines()
        idx = 0
        while idx < len(lines) and re.match(
            r"^\s*Selected agent id\s*:\s*-?\d+\s*$",
            lines[idx],
            re.IGNORECASE,
        ):
            idx += 1
        return "\n".join(lines[idx:]).lstrip()

    async def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare judge prompts and anchors for KV reuse mode."""
        if mode != "allow_kv_reuse":
            # In default mode we build prompts directly in _async_execute.
            return {}

        request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")

        preferred_mode = "kv_reuse"
        agent_memory = self.llm._ensure_agent_memory(self.id)
        prefix_text = kwargs.get("prefix", "")
        candidate_agent_ids = self._get_agent_order(spatial_info)

        has_shared_prefix = (
            self.llm.has_prefix_initialized(self.id)
            and "placeholder_info" in agent_memory
        )
        early_response: str | None = None

        # Prefix KV 已经初始化：只需要根据当前任务更新 condition / input anchor
        if has_shared_prefix:
            task = raw_inputs["task"]
            for agent_id in candidate_agent_ids:
                info = spatial_info[agent_id]
                cond_text: str | None = None
                cond_prefix: str | None = None

                if self.domain == "gsm8k" and info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    if answer is None:
                        answer = "No variable is named answer."
                    cond_text = f"the answer is {answer}"
                    cond_prefix = "the answer is "
                elif (
                    self.domain == "humaneval"
                    and self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    cond_text = (
                        "Whether it passes internal testing?\n"
                        f"{is_solved}.\n\nThe feedback is:\n\n {feedback}."
                    )
                    cond_prefix = "Whether it passes internal testing?\n"

                if cond_text and cond_prefix:
                    self.llm.update_condition_anchor(
                        request_uid=request_uid,
                        owner_agent_id=agent_id,
                        message=task,
                        content=cond_text,
                        prefix_text=cond_prefix,
                    )

            user_content = prefix_text + raw_inputs["task"]
            preferred_mode = self.llm.update_input_anchor(
                request_uid=request_uid,
                agent_id=self.id,
                message=raw_inputs["task"],
                user_content=user_content,
                prefix_text=prefix_text,
            )
            logger.opt(colors=True).info(
                "<green>[MODE]</green> Task: {} Agent {} ({}) mode: {}",
                raw_inputs["task"],
                self.id,
                self.role,
                preferred_mode,
            )
            return {
                "preferred_mode": preferred_mode,
                "early_response": early_response,
                "candidate_agent_ids": candidate_agent_ids,
            }

        # Prefix KV 尚未初始化：构造带占位符的模板 prompt，并初始化前缀 KV
        if self.domain in {"gsm8k", "humaneval", "mmlu"}:
            candidates = [
                (agent_id, "{agent_" + agent_id + "_current}") for agent_id in candidate_agent_ids
            ]
            user_input = "{user_question}"

            if hasattr(self.prompt_set, "get_select_best_role"):
                system_prompt = self.prompt_set.get_select_best_role()
            else:
                system_prompt = "You are the top decision-maker and judge."

            if hasattr(self.prompt_set, "get_select_best_constraint"):
                constraint = self.prompt_set.get_select_best_constraint(candidate_agent_ids)
            elif hasattr(self.prompt_set, "get_decision_constraint"):
                constraint = self.prompt_set.get_decision_constraint()
            else:
                constraint = ""
            system_prompt = f"{system_prompt}\n\n{constraint}".strip()

            if hasattr(self.prompt_set, "get_select_best_agent_prompt"):
                user_prompt = self.prompt_set.get_select_best_agent_prompt(
                    user_input, candidates, constraint
                )
            else:
                formatted = "\n\n".join(
                    f"Agent {agent_id}:\n{ph}" for agent_id, ph in candidates
                )
                user_prompt = f"{prefix_text}{user_input}\n\n{formatted}\n"
        else:
            system_prompt = self.prompt_set.get_decision_role()
            constraint = (
                self.prompt_set.get_decision_constraint()
                if hasattr(self.prompt_set, "get_decision_constraint")
                else ""
            )
            system_prompt = f"{system_prompt}.\n {constraint}"
            user_input = "{user_question}"
            spatial_str = ""
            decision_few_shot = (
                self.prompt_set.get_decision_few_shot()
                if hasattr(self.prompt_set, "get_decision_few_shot")
                else ""
            )

            if self.domain in {"gsm8k", "mmlu"}:
                for agent_id in candidate_agent_ids:
                    info = spatial_info[agent_id]
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if info["role"] == "Programming Expert":
                        agent_output += "\n the result is {condition_" + agent_id + "_current}"
                    spatial_str += (
                        f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                    )
            elif self.domain == "humaneval":
                for agent_id in candidate_agent_ids:
                    info = spatial_info[agent_id]
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if (
                        self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        condition = "{condition_" + agent_id + "_current}"
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                            f"{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                        )
                    else:
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                        )

            user_prompt = (
                f"{decision_few_shot} {prefix_text} {user_input}\n At the same time, the output of other agents is as follows:\n\n"
                f"{spatial_str}\n\n"
            )
        await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
        return {
            "preferred_mode": preferred_mode,
            "early_response": early_response,
            "candidate_agent_ids": candidate_agent_ids,
        }

    async def _run_dense_selection(
        self,
        *,
        question: str,
        candidate_agent_ids: List[str],
        spatial_info: Dict[str, Any],
    ) -> GenerationResult:
        """Run an extra dense-prefill judge pass using the same candidate ordering."""
        if not spatial_info:
            return GenerationResult(text=question, mode="default", ttft=0.0)

        candidates = [
            (agent_id, spatial_info[agent_id]["output"]) for agent_id in candidate_agent_ids
        ]
        if (
            self.domain in {"gsm8k", "humaneval", "mmlu"}
            and hasattr(self.prompt_set, "get_select_best_agent_prompt")
        ):
            system_prompt = (
                self.prompt_set.get_select_best_role()
                if hasattr(self.prompt_set, "get_select_best_role")
                else "You are a judge that selects the best agent answer."
            )
            constraint = (
                self.prompt_set.get_select_best_constraint(candidate_agent_ids)
                if hasattr(self.prompt_set, "get_select_best_constraint")
                else ""
            )
            user_prompt = self.prompt_set.get_select_best_agent_prompt(
                question, candidates, constraint
            )
        else:
            if hasattr(self.prompt_set, "get_decision_role"):
                system_prompt = self.prompt_set.get_decision_role()
            else:
                system_prompt = "You are a judge that selects the best answer from candidates."
            constraint = (
                self.prompt_set.get_decision_constraint()
                if hasattr(self.prompt_set, "get_decision_constraint")
                else ""
            )
            candidates_block = "\n".join(
                f"Agent {agent_id}: {spatial_info[agent_id]['output']}"
                for agent_id in candidate_agent_ids
            )
            user_prompt = (
                f"{constraint}\n\nQuestion:\n{question}\n\n"
                "Candidate answers from different agents:\n"
                f"{candidates_block}\n\n"
                "Please choose the best answer and copy it exactly."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.llm.agen(
            messages,
            max_tokens=self.llm.DEFAULT_MAX_TOKENS,
            # Avoid request-scoped KVCOMM bookkeeping/anchors: this pass is only
            # for measuring what a dense-prefill judge would select.
            request_uid=None,
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ):
        """Run selection over candidate answers and optionally annotate choice."""

        async def _run_selection() -> GenerationResult:
            question = input.get("task", "") if isinstance(input, dict) else str(input)
            if not spatial_info:
                # Fallback: no peers, just echo the question back.
                return GenerationResult(text=question, mode="default", ttft=0.0)

            # Ordering over predecessors, optionally shuffled for experiments.
            agent_ids = self._get_agent_order(spatial_info)
            answers = [spatial_info[agent_id]["output"] for agent_id in agent_ids]

            # Build selection prompt.
            if self.domain in {"gsm8k", "humaneval", "mmlu"} and hasattr(self.prompt_set, "get_select_best_agent_prompt"):
                candidates = [(agent_id, spatial_info[agent_id]["output"]) for agent_id in agent_ids]
                system_prompt = (
                    self.prompt_set.get_select_best_role()
                    if hasattr(self.prompt_set, "get_select_best_role")
                    else "You are a judge that selects the best agent answer."
                )
                constraint = (
                    self.prompt_set.get_select_best_constraint(agent_ids)
                    if hasattr(self.prompt_set, "get_select_best_constraint")
                    else ""
                )
                user_prompt = self.prompt_set.get_select_best_agent_prompt(
                    question, candidates, constraint
                )
            else:
                # Generic fallback: reuse decision role/constraint if available.
                if hasattr(self.prompt_set, "get_decision_role"):
                    system_prompt = self.prompt_set.get_decision_role()
                else:
                    system_prompt = "You are a judge that selects the best answer from candidates."
                constraint = ""
                if hasattr(self.prompt_set, "get_decision_constraint"):
                    constraint = self.prompt_set.get_decision_constraint()
                candidates_block = "\n".join(
                    f"Agent {agent_id}: {spatial_info[agent_id]['output']}"
                    for agent_id in agent_ids
                )
                user_prompt = (
                    f"{constraint}\n\nQuestion:\n{question}\n\n"
                    "Candidate answers from different agents:\n"
                    f"{candidates_block}\n\n"
                    "Please choose the best answer and copy it exactly."
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            request_uid = input.get("_request_uid") or kwargs.get("request_uid")
            result = await self.llm.agen(
                messages,
                max_tokens=self.llm.DEFAULT_MAX_TOKENS,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )

            # Try to infer which agent was selected by exact match.
            selected_index = None
            selected_agent_id = None
            normalized_output = result.text.strip()
            for idx, candidate in enumerate(answers):
                if isinstance(candidate, str) and normalized_output == candidate.strip():
                    selected_index = idx
                    selected_agent_id = agent_ids[idx]
                    break

            if selected_agent_id is None:
                selected_agent_id = self._infer_selected_agent_id_from_text(result.text, agent_ids)
                if selected_agent_id is not None:
                    try:
                        selected_index = agent_ids.index(selected_agent_id)
                    except ValueError:
                        selected_index = None

            if selected_agent_id is None and self.domain == "mmlu":
                selected_agent_id = self._infer_selected_agent_id_from_mmlu_choice(
                    result.text,
                    agent_ids,
                    spatial_info,
                )
                if selected_agent_id is not None:
                    try:
                        selected_index = agent_ids.index(selected_agent_id)
                    except ValueError:
                        selected_index = None

            masked_selected_agent_id: str | None = None
            if bool(kwargs.get("judge_mask_prev_agent_attn")) and hasattr(self.llm, "agen_mask_prev_agent_attn"):
                try:
                    masked_result = await self.llm.agen_mask_prev_agent_attn(
                        messages=messages,
                        candidate_agent_ids=agent_ids,
                        max_tokens=self.llm.DEFAULT_MAX_TOKENS,
                        agent_id=self.id,
                        agent_name=self.agent_name,
                        agent_role=self.role,
                    )
                    masked_selected_agent_id = self._infer_selected_agent_id_from_text(
                        masked_result.text, agent_ids
                    )
                    if masked_selected_agent_id is None and self.domain == "mmlu":
                        masked_selected_agent_id = self._infer_selected_agent_id_from_mmlu_choice(
                            masked_result.text,
                            agent_ids,
                            spatial_info,
                        )
                    if masked_selected_agent_id is None:
                        preview_lines = [
                            line for line in str(masked_result.text).splitlines() if line.strip()
                        ][:6]
                        logger.warning(
                            "Masked judge did not yield a selectable agent id; first lines: {}",
                            preview_lines,
                        )
                except Exception as exc:
                    logger.warning("Masked prefill judge failed: {}", exc)

            # Prepend a header so runners can record which agent was selected.
            if self.domain in {"gsm8k", "humaneval", "mmlu"}:
                header_value = selected_agent_id if selected_agent_id is not None else "-1"
                cleaned = self._strip_selected_agent_header(result.text)
                masked_header = ""
                if bool(kwargs.get("judge_mask_prev_agent_attn")):
                    masked_value = masked_selected_agent_id if masked_selected_agent_id is not None else "-1"
                    masked_header = f"Selected agent id (masked): {masked_value}\n"
                new_text = f"Selected agent id: {header_value}\n{masked_header}{cleaned}"
                result = GenerationResult(
                    text=new_text,
                    mode=result.mode,
                    ttft=result.ttft,
                    raw_output=result.raw_output,
                    metadata={
                        **(result.metadata or {}),
                        "selected_agent_id": selected_agent_id,
                        "selected_index": selected_index,
                        "masked_selected_agent_id": masked_selected_agent_id,
                    },
                )

            return result

        if mode == "default":
            return await _run_selection()

        if mode == "allow_kv_reuse":
            request_uid = input.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            mode_data = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode="allow_kv_reuse",
                **kwargs,
            )
            if mode_data.get("early_response") is not None:
                early = GenerationResult(
                    text=mode_data["early_response"],
                    mode="kv_reuse",
                    ttft=0.0,
                )
                return input.get("task"), early

            candidate_agent_ids = mode_data.get("candidate_agent_ids") or self._get_agent_order(spatial_info)
            result = await self.llm.generate_for_agent(
                request_uid=request_uid,
                message=input["task"],
                preferred_mode=mode_data["preferred_mode"],
                output_dir=kwargs.get("output_dir"),
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            if self.domain in {"gsm8k", "humaneval", "mmlu"}:
                selected_agent_id = self._infer_selected_agent_id_from_text(result.text, candidate_agent_ids)
                if selected_agent_id is not None:
                    try:
                        selected_index = candidate_agent_ids.index(selected_agent_id)
                    except ValueError:
                        selected_index = None
                else:
                    selected_index = None

                if selected_agent_id is None and self.domain == "mmlu":
                    selected_agent_id = self._infer_selected_agent_id_from_mmlu_choice(
                        result.text,
                        candidate_agent_ids,
                        spatial_info,
                    )
                    if selected_agent_id is not None:
                        try:
                            selected_index = candidate_agent_ids.index(selected_agent_id)
                        except ValueError:
                            selected_index = None

                dense_selected_agent_id: str | None = None
                agreement_with_dense: bool | None = None
                if bool(kwargs.get("judge_compare_dense")):
                    dense_result = await self._run_dense_selection(
                        question=input["task"],
                        candidate_agent_ids=candidate_agent_ids,
                        spatial_info=spatial_info,
                    )
                    dense_selected_agent_id = self._infer_selected_agent_id_from_text(
                        dense_result.text, candidate_agent_ids
                    )
                    if dense_selected_agent_id is None and self.domain == "mmlu":
                        dense_selected_agent_id = self._infer_selected_agent_id_from_mmlu_choice(
                            dense_result.text,
                            candidate_agent_ids,
                            spatial_info,
                        )
                    if selected_agent_id is not None and dense_selected_agent_id is not None:
                        agreement_with_dense = (selected_agent_id == dense_selected_agent_id)

                header_value = selected_agent_id if selected_agent_id is not None else "-1"
                cleaned = self._strip_selected_agent_header(result.text)
                dense_header = ""
                if bool(kwargs.get("judge_compare_dense")):
                    dense_value = dense_selected_agent_id if dense_selected_agent_id is not None else "-1"
                    dense_header = f"Selected agent id (dense): {dense_value}\n"
                result = GenerationResult(
                    text=f"Selected agent id: {header_value}\n{dense_header}{cleaned}",
                    mode=result.mode,
                    ttft=result.ttft,
                    raw_output=result.raw_output,
                    metadata={
                        **(result.metadata or {}),
                        "selected_agent_id": selected_agent_id,
                        "selected_index": selected_index,
                        "dense_selected_agent_id": dense_selected_agent_id,
                        "agreement_with_dense": agreement_with_dense,
                    },
                )
            return input.get("task"), result

        raise ValueError(f"Unsupported async execution mode: {mode}")

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        """Synchronous wrapper for compatibility with Graph.run."""
        result = asyncio.run(
            self._async_execute(
                input,
                spatial_info,
                temporal_info,
                mode="default",
                **kwargs,
            )
        )
        if isinstance(result, GenerationResult):
            return result.text
        if isinstance(result, list) and result and isinstance(result[0], GenerationResult):
            return [item.text for item in result]
        return result


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    """Final referencing/answer agent assembling the final response."""
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.role = 'FinalRefer'
        self.llm.set_id(self.id, 'FinalRefer')
        self.prompt_set = PromptSetRegistry.get(domain)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Any],
        temporal_info:Dict[str,Any],
        mode: str = "default",
        **kwargs,
    )->Dict[str, Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)
            prefix_text = kwargs.get("prefix", "")

            has_shared_prefix = (
                self.llm.has_prefix_initialized(self.id)
                and "placeholder_info" in agent_memory
            )
            early_response: str | None = None

            if has_shared_prefix:
                task = raw_inputs["task"]
                for agent_id, info in spatial_info.items():
                    cond_text: str | None = None
                    cond_prefix: str | None = None

                    if self.domain == "gsm8k" and info["role"] == "Programming Expert":
                        answer = execute_code_get_return(
                            info["output"].split("```python\n")[-1].split("\n```")[0]
                        )
                        if answer is None:
                            answer = "No variable is named answer."
                        cond_text = f"the answer is {answer}"
                        cond_prefix = "the answer is "
                    elif (
                        self.domain == "humaneval"
                        and self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        code = info["output"].split("```python\n")[-1].split("\n```")[0]
                        is_solved, feedback, _ = PyExecutor().execute(
                            code, getattr(self, "internal_tests", []), timeout=10
                        )
                        cond_text = (
                            "Whether it passes internal testing?\n"
                            f"{is_solved}.\n\nThe feedback is:\n\n {feedback}."
                        )
                        cond_prefix = "Whether it passes internal testing?\n"

                    if cond_text and cond_prefix:
                        self.llm.update_condition_anchor(
                            request_uid=request_uid,
                            owner_agent_id=agent_id,
                            message=task,
                            content=cond_text,
                            prefix_text=cond_prefix,
                        )

                user_content = prefix_text + raw_inputs["task"]
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs["task"],
                    user_content=user_content,
                    prefix_text=prefix_text,
                )
                logger.opt(colors=True).info(
                    "<green>[MODE]</green> Task: {} Agent {} ({}) mode: {}",
                    raw_inputs["task"],
                    self.id,
                    self.role,
                    preferred_mode,
                )
                return {
                    "preferred_mode": preferred_mode,
                    "early_response": early_response,
                }

            system_prompt = self.prompt_set.get_decision_role()
            self.constraint = self.prompt_set.get_decision_constraint()
            system_prompt = f"{system_prompt}.\n {self.constraint}"
            user_input = "{user_question}"
            spatial_str = ""
            decision_few_shot = self.prompt_set.get_decision_few_shot()

            spatial_str = ""
            if self.domain in {"gsm8k", "mmlu"}:
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if info["role"] == "Programming Expert":
                        agent_output += "\n the result is {condition_" + agent_id + "_current}"
                    spatial_str += (
                        f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                    )
            elif self.domain == "humaneval":
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if (
                        self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        condition = "{condition_" + agent_id + "_current}"
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                            f"{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                        )
                    else:
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                        )

            decision_few_shot = self.prompt_set.get_decision_few_shot()
            user_prompt = (
                f"{decision_few_shot} {prefix_text} {user_input}\n At the same time, the output of other agents is as follows:\n\n"
                f"{spatial_str}\n\n"
            )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {
                "preferred_mode": preferred_mode,
                "early_response": early_response,
            }

        system_prompt = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{system_prompt}.\n {self.constraint}"
        prefix_text = kwargs.get("prefix", "")
        spatial_str = ""
        if self.domain in {"gsm8k", "mmlu"}:
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    agent_output += f"\n the result is {answer}"
                spatial_str += (
                    f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                )
        elif self.domain == "humaneval":
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if (
                    self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                        f"{agent_output}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                    )
                else:
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                    )

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} {prefix_text} {raw_inputs['task']}\n At the same time, the output of other agents is as follows:\n\n"
            f"{spatial_str}\n\n"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode="default",
                **kwargs,
            )
        )
        message = [{'role':'system','content':inputs["system_prompt"]},{'role':'user','content':inputs["user_prompt"]}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """Handle asynchronous execution across different KV cache strategies."""
        if self.domain == 'humaneval':
            self.internal_tests = self.extract_example(input)
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
            message = [{'role':'system','content':inputs["system_prompt"]},{'role':'user','content':inputs["user_prompt"]}]
            result = await self.llm.agen(
                message,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            return result

        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")

        mode_data = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            mode="allow_kv_reuse",
            **kwargs,
        )
        if mode_data["early_response"] is not None:
            early = GenerationResult(
                text=mode_data["early_response"],
                mode="kv_reuse" if mode == "allow_kv_reuse" else mode,
                ttft=0.0,
            )
            return input['task'], early
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input['task'],
            preferred_mode=mode_data["preferred_mode"],
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return input['task'], result

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        if mode == "allow_kv_reuse":
            return input.get("task"), output
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = await self.prompt_set.postprocess_answer(info['output'])
            logger.debug("Processed output: {}", processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        if mode == "allow_kv_reuse":
            return input.get("task"), max_output
        return max_output
