import asyncio
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.tools.reader.readers import JSONLReader
from KVCOMM.utils.log import logger
from KVCOMM.utils.metrics import GenerationResult
from my_datasets.gsm8k_dataset import gsm_data_process


def _load_gsm8k_examples(dataset_path: str) -> List[Dict[str, str]]:
    """Load GSM8K-style examples as candidate few-shot pool."""
    path = Path(dataset_path)
    if not path.exists():
        logger.warning("FewShotRetriever dataset path does not exist: {}", dataset_path)
        return []
    raw = JSONLReader.parse_file(str(path))
    try:
        processed = gsm_data_process(raw)
    except Exception as exc:
        logger.warning("Failed to process GSM8K dataset for few-shot retrieval: {}", exc)
        return []
    return processed


@AgentRegistry.register("FewShotRetriever")
class FewShotRetriever(Node):
    """Retrieve a single GSM8K-style solved example as few-shot context.

    This agent does not ask the LLM to generate content. Instead, it selects
    one example from a preloaded GSM8K-style dataset and exposes it as a
    reusable condition segment via KVCOMM's condition anchors.
    """

    def __init__(
        self,
        id: str | None = None,
        role: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
        dataset_path: str | None = None,
    ):
        super().__init__(id, "FewShotRetriever", domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.role = role or "FewShotRetriever"
        self.llm.set_id(self.id, self.role)

        default_path = "my_datasets/gsm8k/gsm8k_train.jsonl"
        self.dataset_path = dataset_path or os.getenv("GSM8K_FEWSHOT_PATH", default_path)
        self.examples: List[Dict[str, str]] = _load_gsm8k_examples(self.dataset_path)
        if not self.examples:
            logger.warning(
                "FewShotRetriever initialised with empty example pool from {}",
                self.dataset_path,
            )
        self._cache: Dict[str, str] = {}

    def _pick_example(self, question: str) -> str:
        """Select a single example for the given question."""
        if not self.examples:
            return "No few-shot example is available."

        base = abs(hash(question))
        try:
            node_offset = int(self.id)
        except Exception:
            node_offset = 0
        idx = (base + node_offset) % len(self.examples)
        ex = self.examples[idx]
        q = ex.get("task", "").strip()
        step = ex.get("step", "").strip()
        ans = ex.get("answer", "").strip()
        parts = []
        if q:
            parts.append(f"Q: {q}")
        if step:
            parts.append(step)
        if ans:
            parts.append(f"The answer is {ans}")
        return "\n".join(parts) if parts else "No few-shot example is available."

    async def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare condition KV for the retrieved example in KV reuse mode."""
        if mode != "allow_kv_reuse":
            return {}

        request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")

        task = raw_inputs.get("task", "")
        example_text = self._pick_example(task)
        self._cache[task] = example_text

        prefix_text = "Few-shot example:\n"
        self.llm.update_condition_anchor(
            request_uid=request_uid,
            owner_agent_id=self.id,
            message=task,
            content=example_text,
            prefix_text=prefix_text,
        )
        logger.opt(colors=True).info(
            "<green>[FewShotRetriever]</green> Prepared condition KV for task '{}' on agent {}",
            task,
            self.id,
        )
        return {"preferred_mode": "kv_reuse", "early_response": None}

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        """Synchronous execution: return the retrieved example text."""
        task = input.get("task", "")
        text = self._pick_example(task)
        self._cache[task] = text
        return text

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        mode: str = "default",
        **kwargs,
    ):
        """Asynchronous execution across default and KV reuse modes."""
        task = input.get("task", "")
        if mode == "default":
            text = self._pick_example(task)
            self._cache[task] = text
            result = GenerationResult(text=text, mode="default", ttft=0.0)
            return result

        if mode == "allow_kv_reuse":
            text = self._cache.get(task)
            if text is None:
                text = self._pick_example(task)
                self._cache[task] = text
            result = GenerationResult(text=text, mode="kv_reuse", ttft=0.0)
            return task, result

        raise ValueError(f"Unsupported async execution mode: {mode}")
