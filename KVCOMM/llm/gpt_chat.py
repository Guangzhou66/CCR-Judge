"""Vendorized into Test3 from Judge/Ori for local builder stability.

Chat backends and local HF model with KV reuse and anchor-based prefill.

This module provides two implementations:
- GPTChat: a thin adapter over an OpenAI-compatible chat API.
- LLMChat: a Hugging Face model runner that supports KV reuse between
  agents/requests and dense prefill with anchor selection.

Key concepts:
- Prefix KV: KV cache segments corresponding to static prompt parts.
- Placeholder: Markers in the prompt whose content is populated per request.
- Anchor: A remembered KV delta used to adjust prefix/payload caches.
"""
from typing import List, Union, Optional, Dict, Any, Tuple
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import threading
import asyncio
import async_timeout
from openai import AsyncOpenAI
import re
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from transformers.cache_utils import DynamicCache
from KVCOMM.llm.format import Message
from KVCOMM.llm.llm import LLM
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.config import KVCommConfig

from KVCOMM.llm.token_ops import *
from KVCOMM.llm.kvcomm_engine import KVCOMMEngine, _RequestState
from KVCOMM.utils.metrics import GenerationResult
from KVCOMM.utils.log import logger


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


def _paper_debug_runtime(message: str) -> None:
    path_value = os.environ.get("PAPER_REPAIR_DEBUG_LOG", "").strip()
    if not path_value:
        return
    if _console_debug_enabled():
        print(message, flush=True)
    try:
        with open(path_value, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception:
        pass


_paper_debug_import("imported gpt_chat")

MINE_API_KEYS = os.getenv('API_KEY')


def _escape_loguru_markup(text: Optional[str]) -> str:
    """Escape Loguru markup tokens in free-form text."""
    if text is None:
        return ""
    return text.replace("<", "\\<")


_LATENCY_IO_LOCK = threading.Lock()


def _resolve_latency_path(target: Optional[Union[str, Path]]) -> Optional[Path]:
    if target is None:
        return None
    path = Path(target)
    if path.suffix:
        return path
    return path / "Latency.json"


def _append_latency_record(target: Optional[Union[str, Path]], record: Dict[str, Any]) -> None:
    """Persist a latency record to JSON, tolerating malformed or missing files."""
    path = _resolve_latency_path(target)
    if path is None:
        return
    serializable = {key: value for key, value in record.items() if value is not None}
    with _LATENCY_IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        existing = loaded
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(serializable)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)


class _TTFTTracer(StoppingCriteria):
    """Stopping criteria to capture time-to-first-token during generation."""

    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft: Optional[float] = None

    def reset(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if self.ttft is None and input_ids.shape[-1] > self.prompt_length:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.ttft = perf_counter() - self.start_time
        return False

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    """Call an OpenAI-compatible chat endpoint asynchronously."""
    api_kwargs = dict(api_key = MINE_API_KEYS)
    try:
        aclient = AsyncOpenAI(**api_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create the async client: {e}")
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    """Thin wrapper around OpenAI-style chat completions."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        """Asynchronously generate a response via hosted chat API."""

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        response_text = await achat(self.model_name, messages)
        metadata: Dict[str, Any] = {}
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        return GenerationResult(
            text=response_text,
            mode="default",
            ttft=0.0,
            metadata=metadata,
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation not implemented for this adapter."""
        pass

@LLMRegistry.register('LLMChat')
class LLMChat(LLM):
    """Local HF model chat with KV reuse and anchor-based dense prefill.

    Provides utilities to construct chat prompts, manage shared KV caches
    across agents/requests, and generate with two strategies:
    - kv_reuse: reuses previously materialized KV segments
    - dense_prefill: regenerates prefix KV and optionally sets anchors
    """
    _shared_model = None
    _shared_tokenizer = None
    _model_lock = threading.Lock()                                        
    _THREAD_POOL: ThreadPoolExecutor | None = None
    _THREAD_POOL_WORKERS: int | None = None
    _shared_kv_cache_memory = None
    _initialization = {}
    anchors = KVCOMMEngine.anchors
    anchor_dict = KVCOMMEngine.anchor_dict
    anchor_len_dict = KVCOMMEngine.anchor_len_dict
    anchor_info_dict = KVCOMMEngine.anchor_info_dict
    weight_dict = KVCOMMEngine.weight_dict
    global_anchor_info_dict = KVCOMMEngine.global_anchor_info_dict

    _request_lock = KVCOMMEngine._request_lock
    _request_states = KVCOMMEngine._request_states
    _active_requests = KVCOMMEngine._active_requests
    _staged_commits = KVCOMMEngine._staged_commits
    _request_messages: Dict[str, set[str]] = {}
    _message_refcount: Dict[str, int] = {}

    def __init__(self, model_name: str, prefix: str = None, config: KVCommConfig | None = None):
        """Create a chat model instance and initialize shared resources.

        Args:
            model_name: HF model identifier.
            prefix: Optional legacy/template prefix configuration.
            config: KVComm runtime configuration.
        """
        self.model_name = model_name

        self.config = (config or KVCommConfig.from_env()).validate()
        self._ensure_thread_pool(self.config.thread_pool_workers)
        self.kv_engine = KVCOMMEngine(self)

        self.lock = asyncio.Lock()                       


        self._initialize_shared_resources()


        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._shared_kv_cache_memory = LLMChat._shared_kv_cache_memory
        self._initialization = LLMChat._initialization
        self._chat_markers = self._extract_chat_markers()
        self.default_assistant_prompt = "A: "
        self.base_messages_template: List[Dict[str, str]] = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        if prefix is not None:
            self._prepare_prefix_template(prefix)

    def _extract_chat_markers(self) -> Dict[str, str]:
        """Parse tokenizer chat template to identify structural markers."""
        template = getattr(self.tokenizer, "chat_template", "") or ""
        markers = {"begin": "", "start": "", "end": "", "eot": ""}
        begin_candidates = ["<|begin_of_text|>", "<s>", getattr(self.tokenizer, "bos_token", "") or ""]
        start_candidates = ["<|start_header_id|>", "<|im_start|>"]
        end_candidates = ["<|end_header_id|>", "<|im_end|>", "\n"]
        eot_candidates = ["<|eot_id|>", "<|im_end|>", getattr(self.tokenizer, "eos_token", "") or ""]

        for token in begin_candidates:
            if token and token in template:
                markers["begin"] = token
                break
        if not markers["begin"]:
            markers["begin"] = begin_candidates[-1]

        for token in start_candidates:
            if token and token in template:
                markers["start"] = token
                break

        for token in end_candidates:
            if token and token in template:
                markers["end"] = token
                break
        if not markers["end"]:
            markers["end"] = ""

        for token in eot_candidates:
            if token and token in template:
                markers["eot"] = token
                break
        if not markers["eot"]:
            markers["eot"] = eot_candidates[-1]

        return markers

    def _prepare_prefix_template(self, prefix: Union[str, List[Dict[str, str]]]) -> None:
        """Normalise various prefix formats into a base messages template."""
        if isinstance(prefix, list):
            self.base_messages_template = prefix
            return
        if isinstance(prefix, dict):
            self.base_messages_template = [prefix]
            return
        if isinstance(prefix, tuple):
            prefix = list(prefix)
        if isinstance(prefix, list) and all(isinstance(item, tuple) and len(item) == 2 for item in prefix):
            self.base_messages_template = [{"role": role, "content": tmpl} for role, tmpl in prefix]
            return
        if isinstance(prefix, str):
            self.default_assistant_prompt = self._extract_assistant_prompt(prefix)
            return
        raise TypeError("Unsupported prefix template type.")

    def _extract_assistant_prompt(self, legacy_prefix: str) -> str:
        """Extract trailing assistant prompt from a legacy text prefix."""
        start = self.start_header_id
        end = self.end_header_id
        if start and end:
            marker = f"{start}assistant{end}\n"
            if marker in legacy_prefix:
                tail = legacy_prefix.split(marker, 1)[-1]
                eot = self.eot_id
                if eot:
                    tail = tail.replace(eot, "")
                return tail
        return legacy_prefix

    @property
    def begin_of_text(self) -> str:
        return self._chat_markers.get("begin", "")

    @property
    def start_header_id(self) -> str:
        return self._chat_markers.get("start", "")

    @property
    def end_header_id(self) -> str:
        return self._chat_markers.get("end", "")

    @property
    def eot_id(self) -> str:
        return self._chat_markers.get("eot", "")

    @staticmethod
    def _normalise_messages(messages: Union[List[Message], List[Dict[str, str]], Dict[str, Any], Tuple[Any, ...], str]) -> List[Dict[str, str]]:
        """Convert mixed message representations into chat dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, tuple):
            if len(messages) == 2 and all(isinstance(item, str) for item in messages):
                system_prompt, user_prompt = messages
                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            return LLMChat._normalise_messages(list(messages))
        if isinstance(messages, dict):
            result: List[Dict[str, str]] = []
            system_prompt = messages.get("system") or messages.get("system_prompt")
            if system_prompt:
                result.append({"role": "system", "content": system_prompt})
            conversation = messages.get("messages") or messages.get("conversation")
            if conversation is not None:
                result.extend(LLMChat._normalise_messages(conversation))
            else:
                if "user" in messages:
                    user_payload = messages["user"]
                    if isinstance(user_payload, list):
                        result.extend(LLMChat._normalise_messages(user_payload))
                    else:
                        result.append({"role": "user", "content": user_payload})
                if "assistant" in messages:
                    assistant_payload = messages["assistant"]
                    if isinstance(assistant_payload, list):
                        result.extend(LLMChat._normalise_messages(assistant_payload))
                    else:
                        result.append({"role": "assistant", "content": assistant_payload})
            return result
        if not isinstance(messages, list):
            raise TypeError("messages must be a string, sequence, or a list of Message/Dict objects.")
        if messages and isinstance(messages[0], Message):
            return [{"role": m.role, "content": m.content} for m in messages]
        normalised: List[Dict[str, str]] = []
        for item in messages:
            if isinstance(item, Message):
                normalised.append({"role": item.role, "content": item.content})
            elif isinstance(item, dict):
                if "role" in item:
                    normalised.append({"role": item["role"], "content": item.get("content", "")})
                else:
                    normalised.extend(LLMChat._normalise_messages(item))
            elif isinstance(item, str):
                normalised.append({"role": "user", "content": item})
            else:
                normalised.extend(LLMChat._normalise_messages(item))
        return normalised

    def _legacy_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Fallback prompt renderer when chat_template is unavailable."""
        prompt_parts = [self.begin_of_text or getattr(self.tokenizer, "bos_token", "") or ""]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            start = self.start_header_id
            end = self.end_header_id
            eot = self.eot_id or getattr(self.tokenizer, "eos_token", "") or ""
            if start and end:
                prompt_parts.append(f"{start}{role}{end}\n{content}{eot}")
            else:
                prompt_parts.append(f"[{role.upper()}]\n{content}{eot}")
        if self.start_header_id and self.end_header_id:
            prompt_parts.append(f"{self.start_header_id}assistant{self.end_header_id}\n")
        else:
            prompt_parts.append("[ASSISTANT]\n")
        return "".join(prompt_parts)

    def _build_chat_inputs(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        assistant_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], str, int]:
        """Tokenize chat messages and return model inputs, text, and prompt length."""
        normalised = self._normalise_messages(messages)
        assistant_prompt = assistant_prompt or self.default_assistant_prompt
        prompt_text = ""
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                normalised,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ) + assistant_prompt

            tokenized = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)

            if isinstance(tokenized, dict):
                inputs = tokenized
            else:
                inputs = {
                    "input_ids": tokenized,
                    "attention_mask": torch.ones_like(tokenized),
                }
        except (ValueError, AttributeError, NotImplementedError, TypeError):

            prompt_text = self._legacy_prompt_from_messages(normalised) + assistant_prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        return inputs, prompt_text, input_length

    def _render_base_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """Render base messages from the current template with provided text."""
        rendered: List[Dict[str, str]] = []
        template = self.base_messages_template or [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        for block in template:
            content_template = block.get("content", "")
            content = content_template.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            rendered.append({"role": block.get("role", "user"), "content": content})
        return rendered

    def format_chat_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
    ) -> str:
        """Render a single chat block for the given role and content."""
        prefix = self.begin_of_text if include_begin else ""
        start = self.start_header_id
        end = self.end_header_id
        eot = self.eot_id if include_eot else ""
        if start and end:
            return f"{prefix}{start}{role}{end}\n{content}{eot}"
        upper_role = role.upper()
        return f"{prefix}[{upper_role}]\n{content}{eot}"

    def tokenize_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single chat segment and move tensors to the model device."""
        text = self.format_chat_segment(
            role,
            content,
            include_begin=include_begin,
            include_eot=include_eot,
        )
        tokens = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        return {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokens.items()
        }

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: Optional[str] = None,
        *,
        add_generation_prompt: bool = True,
        return_messages: bool = False,
    ) -> Dict[str, Any]:
        """Create model inputs from system/user prompts and optional assistant suffix."""
        messages = self._render_base_messages(system_prompt, user_prompt)
        inputs, prompt_text, prompt_length = self._build_chat_inputs(
            messages,
            assistant_prompt=assistant_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        result: Dict[str, Any] = {
            "inputs": inputs,
            "prompt_text": prompt_text,
            "prompt_length": prompt_length,
        }
        if return_messages:
            result["messages"] = messages
        return result

    @classmethod
    def _ensure_thread_pool(cls, workers: int) -> None:
        """Initialise or resize the shared thread pool used for CPU work."""
        if cls._THREAD_POOL is None or cls._THREAD_POOL_WORKERS != workers:
            if cls._THREAD_POOL is not None:
                cls._THREAD_POOL.shutdown(wait=False)
            cls._THREAD_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="LLM-chat")
            cls._THREAD_POOL_WORKERS = workers

    @classmethod
    def finalize_request(cls, request_uid: str) -> None:
        KVCOMMEngine.finalize_request(request_uid)
        cls._clear_input_cache_for_request(request_uid)

    @classmethod
    def _clear_input_cache_for_request(cls, request_uid: str) -> None:
        """Clear global input KV cache entries associated with a completed request."""
        messages = cls._request_messages.pop(request_uid, None)
        if not messages:
            return

        store = cls._shared_kv_cache_memory
        if store is None:
            return

        input_bucket = store.get("input")
        input_ids_bucket = store.get("input_ids")
        drop_bucket = store.get("input_drop_num")
        if input_bucket is None or input_ids_bucket is None or drop_bucket is None:
            return

        for message in messages:
            count = cls._message_refcount.get(message)
            if count is None:
                continue
            if count <= 1:
                cls._message_refcount.pop(message, None)
                input_bucket.pop(message, None)
                input_ids_bucket.pop(message, None)
                drop_bucket.pop(message, None)
            else:
                cls._message_refcount[message] = count - 1

    def get_request_state(self, request_uid: str) -> "_RequestState":
        """Return per-request state used by the KV engine."""
        return self.kv_engine.get_request_state(request_uid)

    def _ensure_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Return the shared memory slot for a given agent id."""
        return LLMChat._shared_kv_cache_memory.setdefault(agent_id, {})

    def _ensure_global_input_buckets(self) -> Dict[str, Dict[str, Any]]:
        """Ensure the global input buckets exist and return the shared store."""
        store = LLMChat._shared_kv_cache_memory
        store.setdefault("input", {})
        store.setdefault("input_ids", {})
        store.setdefault("input_drop_num", {})
        return store

    def has_prefix_initialized(self, agent_id: str) -> bool:
        """Check if prefix KV has been initialized for an agent."""
        return LLMChat._initialization.get(agent_id, False)

    def has_active_anchor(self, request_uid: str, message: str) -> bool:
        """Determine whether an anchor should trigger dense prefill."""
        state = self.get_request_state(request_uid)
        ph_ids = LLMChat._shared_kv_cache_memory.get(self.node_id, {}).get('placeholder_info', {}).keys()
        for ph_id in ph_ids:
            bucket = state.anchor_dict.setdefault(ph_id, {})
            if bucket.get(message) is True and f'{self.node_id}_ph_key_delta' not in state.anchors.get(ph_id, {}).get(message, {}):
                return True
        return False

    def _placeholder_is_naive(self, ph_id: str) -> bool:
        """Return True if this placeholder should use naive RoPE-only reuse.

        Controlled via KVCOMM_NAIVE_PLACEHOLDERS (comma-separated substrings),
        e.g. KVCOMM_NAIVE_PLACEHOLDERS=user_question,agent_0_ .
        """
        patterns = os.environ.get("KVCOMM_NAIVE_PLACEHOLDERS", "")
        if not patterns:
            return False
        for pat in patterns.split(","):
            pat = pat.strip()
            if pat and pat in ph_id:
                return True
        return False

    def _collect_anchors_for_placeholder(
        self,
        state: "_RequestState",
        ph_id: str,
    ) -> List[Dict]:
        """Return anchor entries for a placeholder, with optional grouping.

        By default this returns anchors exactly keyed by `ph_id`. When
        `config.group_judge_agent_anchors` is enabled, and the current
        node is a judge (`FinalSelectBest`), anchors for all candidate
        agents (placeholders starting with ``agent_``) are pooled so that
        different MathSolver agents are treated as an equivalence class.
        """
        anchors_for_node = state.anchors
        bucket = anchors_for_node.get(ph_id, {})

        group_any = getattr(self.config, "group_judge_agent_anchors", False)
        group_by_pos = getattr(self.config, "group_judge_agent_anchors_by_position", False)

        # Fast path: no grouping configured.
        if not group_any and not group_by_pos:
            return list(bucket.values())

        # Only group for the judge agent.
        if getattr(self, "role", None) != "FinalSelectBest":
            return list(bucket.values())

        # Only group placeholders that correspond to agent outputs.
        if not ph_id.startswith("agent_"):
            return list(bucket.values())

        # Position-based grouping: only pool anchors from the same display slot.
        if group_by_pos:
            pos_map = getattr(state, "judge_display_pos_by_ph_id", None) or {}
            current_pos = pos_map.get(ph_id)
            if current_pos is None:
                return list(bucket.values())

            pooled: List[Dict] = []
            for key, per_message in anchors_for_node.items():
                if not key.startswith("agent_"):
                    continue
                for _msg_key, entry in per_message.items():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("_display_pos") != current_pos:
                        continue
                    if entry.get("_source_ph_id") is None:
                        entry["_source_ph_id"] = key
                    pooled.append(entry)
        else:
            # Agent-id grouping: pool across all agent placeholders.
            pooled = []
            for key, per_message in anchors_for_node.items():
                if not key.startswith("agent_"):
                    continue
                for _msg_key, entry in per_message.items():
                    if isinstance(entry, dict) and entry.get("_source_ph_id") is None:
                        entry["_source_ph_id"] = key
                    pooled.append(entry)

        # Stable order helps weight caching/debugging.
        def _sort_key(anchor: Dict) -> str:
            if not isinstance(anchor, dict):
                return ""
            return str(anchor.get("_source_ph_id") or "")

        pooled.sort(key=_sort_key)
        return pooled

    def update_condition_anchor(
        self,
        *,
        request_uid: str,
        owner_agent_id: str,
        message: str,
        content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: Optional[str] = None,
        max_length: int = None,
    ) -> bool:
        """Materialise condition KV cache for another agent and update anchors."""
        state = self.get_request_state(request_uid)
        anchor_key = anchor_namespace or f"condition_{owner_agent_id}_current"

        owner_memory = self._ensure_agent_memory(owner_agent_id)
        condition_bucket = owner_memory.setdefault("condition", {})
        if message in condition_bucket:

            return state.anchor_dict.setdefault(anchor_key, {}).get(message, False)

        token_ids = self.tokenize_segment(
            role=role,
            content=content,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )
        if "position_ids" not in token_ids:
            position_ids = torch.arange(token_ids["input_ids"].shape[-1]).unsqueeze(0)
            token_ids["position_ids"] = position_ids.to(self.model.device)
        else:
            token_ids["position_ids"] = token_ids["position_ids"].to(self.model.device)
        token_ids["input_ids"] = token_ids["input_ids"].to(self.model.device)
        token_ids["attention_mask"] = token_ids["attention_mask"].to(self.model.device)

        prefix_ids = self.tokenize_segment(
            role=role,
            content=prefix_text,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )["input_ids"]
        drop_num = prefix_ids.shape[-1]

        if max_length is not None:
            token_ids["input_ids"] = token_ids["input_ids"][:, :drop_num + max_length]
            token_ids["attention_mask"] = token_ids["attention_mask"][:, :drop_num + max_length]
            token_ids["position_ids"] = token_ids["position_ids"][:, :drop_num + max_length]
            
        generated = self.model.generate(
            **token_ids,
            use_cache=True,
            max_length=token_ids["input_ids"].shape[-1] + 1,
            return_dict_in_generate=True,
            return_legacy_cache=False,
        )
        condition_cache = generated.past_key_values

        for key_name, value in (
            ("condition", condition_cache),
            ("condition_ids", token_ids),
            ("condition_drop_num", drop_num),
        ):
            bucket = owner_memory.setdefault(key_name, {})
            bucket.setdefault(message, []).append(value)

        # For naive placeholders, we don't maintain anchor views.
        if self._placeholder_is_naive(anchor_key):
            state.anchor_dict.setdefault(anchor_key, {})[message] = False
            return False

        anchor_store = state.anchors.setdefault(anchor_key, {})
        cond_anchor_list = list(anchor_store.values())
        cond_len_bucket = state.anchor_len_dict.setdefault(anchor_key, {})
        anchor_len_list = [
            cond_len_bucket.get(entry_key, [0, 0])
            for entry_key in anchor_store.keys()
        ]
        cond_info_bucket = state.anchor_info_dict.setdefault(anchor_key, {})
        anchor_activated_list = list(cond_info_bucket.values())

        total_prefix_len = 0
        for bucket in state.anchor_len_dict.values():
            total_prefix_len += bucket.get(message, [0, 0])[0]

        prob, anchor_activated_list = self.kv_engine.predict_as_anchor(
            condition_cache.copy().slice_(start=drop_num),
            anchor_kv_cache_list=cond_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_activated_list,
        )

        cond_flag_bucket = state.anchor_dict.setdefault(anchor_key, {})
        cond_flag_bucket[message] = prob

        global_bucket = state.global_anchor_info.setdefault(anchor_key, {})
        if not prob:
            info_items = list(cond_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                cond_info_bucket[msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
        else:
            cond_info_bucket[message] = 0
            global_bucket[message] = [
                0,
                condition_cache.get_seq_length() - drop_num,
            ]
        return prob

    def update_input_anchor(
        self,
        *,
        request_uid: str,
        agent_id: str,
        message: str,
        user_content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: str = "user_question",
        test_time: bool = False,
    ) -> str:
        """Ensure the user input placeholder cache is ready and choose a strategy."""
        state = self.get_request_state(request_uid)
        shared_mem = LLMChat._shared_kv_cache_memory
        agent_memory = self._ensure_agent_memory(agent_id)
        placeholder_info = agent_memory.get("placeholder_info")
        safe_message = _escape_loguru_markup(message)
        request_messages = LLMChat._request_messages.setdefault(request_uid, set())

        if message in shared_mem.get("input", {}):
            if message not in request_messages:
                request_messages.add(message)
                LLMChat._message_refcount[message] = LLMChat._message_refcount.get(message, 0) + 1
            if not placeholder_info:
                logger.opt(colors=True).warning(
                    f"<yellow>No placeholder info found for agent '{agent_id}' while reusing input cache.</yellow>"
                )
                return "kv_reuse"
                placeholder_entries = list(placeholder_info.items())[::-1]
                for ph_id, _ in placeholder_entries:
                    safe_ph_id = _escape_loguru_markup(ph_id)
                    bucket = state.anchor_dict.setdefault(ph_id, {})
                    if bucket.get(message):
                        if f'{self.node_id}_ph_key_delta' in state.anchors.get(ph_id, {}).get(message, {}):
                            logger.opt(colors=True).debug(
                                f"<green>The message has repeatedly received for message '{safe_message}' at placeholder '{safe_ph_id}'. So we will reuse the KV cache.</green>"
                            )
                            return "kv_reuse"
                        logger.opt(colors=True).debug(
                            f"<yellow>Existing Anchor for message '{safe_message}' at placeholder '{safe_ph_id}'.</yellow>"
                        )
                        return "dense_prefill"
            logger.opt(colors=True).debug(
                f"<green>Reusing KV caches for message '{safe_message}' in all placeholders</green>."
            )
            return "kv_reuse"

        token_ids = self.tokenize_segment(
            role=role,
            content=user_content,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )
        if "position_ids" in token_ids:
            position_ids = token_ids["position_ids"]
        else:
            position_ids = torch.arange(token_ids["input_ids"].shape[-1], dtype=torch.long)
        token_ids["position_ids"] = position_ids.unsqueeze(0).to(self.model.device)
        token_ids["input_ids"] = token_ids["input_ids"].to(self.model.device)
        token_ids["attention_mask"] = token_ids["attention_mask"].to(self.model.device)

        prefix_ids = self.tokenize_segment(
            role=role,
            content=prefix_text,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )["input_ids"]
        drop_num = prefix_ids.shape[-1]
        if test_time:
            for _ in range(10):
                if _ == 5:
                    torch.cuda.synchronize()
                    start_time = perf_counter()
                output = self.model.generate(
                    **token_ids,
                    use_cache=True,
                    max_length=token_ids["input_ids"].shape[-1] + 1,
                    return_dict_in_generate=True,
                    return_legacy_cache=False,
                )
            torch.cuda.synchronize()
            end_time = perf_counter()
            logger.opt(colors=True).info(
                f"<cyan>Latency for computing the input kv-cache of {message}: {(end_time - start_time) / 5:.3f} seconds</cyan>"
            )
        else:
            output = self.model.generate(
                **token_ids,
                use_cache=True,
                max_length=token_ids["input_ids"].shape[-1] + 1,
                return_dict_in_generate=True,
                return_legacy_cache=False,
            )
        input_cache = output.past_key_values

        global_buckets = self._ensure_global_input_buckets()
        global_buckets["input"].setdefault(message, []).append(
            input_cache.copy().slice_(start=0, end=token_ids["input_ids"].shape[-1])
        )
        global_buckets["input_ids"].setdefault(message, []).append(token_ids)
        global_buckets["input_drop_num"].setdefault(message, []).append(drop_num)

        if message not in request_messages:
            request_messages.add(message)
            LLMChat._message_refcount[message] = LLMChat._message_refcount.get(message, 0) + 1

        # For naive placeholders, we don't maintain anchor views.
        if self._placeholder_is_naive(anchor_namespace):
            state.anchor_dict.setdefault(anchor_namespace, {})[message] = False
            return "kv_reuse"

        anchor_store = state.anchors.setdefault(anchor_namespace, {})
        input_anchor_list = list(anchor_store.values())
        uq_len_bucket = state.anchor_len_dict.setdefault(anchor_namespace, {})
        anchor_len_list = [
            uq_len_bucket.get(entry_key, [0, 0])
            for entry_key in anchor_store.keys()
        ]
        uq_info_bucket = state.anchor_info_dict.setdefault(anchor_namespace, {})
        anchor_activated_list = list(uq_info_bucket.values())

        accumulate_len = 0
        for bucket in state.anchor_len_dict.values():
            accumulate_len += bucket.get(message, [0, 0])[0]

        prob, anchor_activated_list = self.kv_engine.predict_as_anchor(
            input_cache.copy().slice_(start=drop_num),
            anchor_kv_cache_list=input_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_activated_list,
            test_time=test_time,
        )
        logger.opt(colors=True).debug(
            f"<magenta>Anchor prediction for input '{safe_message}'</magenta>: {prob}"
        )

        state.anchor_dict.setdefault(anchor_namespace, {})[message] = prob
        global_bucket = state.global_anchor_info.setdefault(anchor_namespace, {})
        if not prob:
            info_items = list(uq_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                uq_info_bucket[msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
            return "kv_reuse"

        uq_info_bucket[message] = 0
        global_bucket[message] = [
            0,
            input_cache.get_seq_length() - drop_num,
        ]
        return "dense_prefill"

    async def generate_for_agent(
        self,
        *,
        request_uid: str,
        message: str,
        preferred_mode: Optional[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response using the requested strategy with sensible fallbacks."""
        latency_target = output_dir or kwargs.get("output_dir")
        if preferred_mode == "dense_prefill":
            mode = "dense_prefill"
        elif self.has_active_anchor(request_uid, message):
            mode = "dense_prefill"
        else:
            mode = "kv_reuse"

        if mode == "dense_prefill":
            return await self.generate_with_dense_prefill(
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                max_anchor_num=kwargs.get("max_anchor_num", self.config.max_anchor_num),
                window_length=kwargs.get("window_length", self.config.window_size),
                request_uid=request_uid,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=latency_target,
                **kwargs,
            )
        return await self.generate_with_kv_reuse(
            message,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=latency_target,
            **kwargs,
        )

    def _map_in_pool(self, fn, iterable, timeout=None):
        pool = LLMChat._THREAD_POOL
        if pool is None:
            raise RuntimeError("Thread pool not initialized")
        task_timeout = timeout or self.config.worker_timeout
        futures = [pool.submit(fn, *args) for args in iterable]
        for fut in as_completed(futures, timeout=task_timeout):
            try:
                yield fut.result(timeout=self.config.worker_timeout)
            except TimeoutError as exc:
                raise TimeoutError("Thread task timeout") from exc
            except Exception as exc:
                raise RuntimeError("Thread task failed") from exc

    def set_id(self, node_id: str, role: str):
        """Bind the chat instance to a graph node id and role."""
        self.node_id = node_id
        self.role = role

        if self.node_id not in LLMChat._shared_kv_cache_memory:
            self._shared_kv_cache_memory[self.node_id] = LLMChat._shared_kv_cache_memory[self.node_id] = {}
            self._initialization[self.node_id] = LLMChat._initialization[self.node_id] = False

    async def prepare_prefix_kv_segments(self, node_id: str, prefix: str, user_prompt: str):
        """Materialize and store prefix KV segments and placeholder indices.

        The rendered prompt is tokenized and executed once to obtain the KV
        cache of each text segment. These are stored in shared memory keyed by
        `node_id` for reuse during subsequent generations.
        """
        messages = self._render_base_messages(prefix, user_prompt)
        _, prompt_text, _ = self._build_chat_inputs(messages, add_generation_prompt=True)
        placeholder_info, token_ids, segments = self.locate_placeholder(prompt_text, return_segments=True)
        
        with torch.no_grad():
            out = self.model.generate(
                **token_ids,
                use_cache=True,
                max_length=token_ids['input_ids'].shape[-1] + 1,
                return_dict_in_generate=True,
                return_legacy_cache=False,
            )
        base_kv = out.past_key_values.slice_(start=0, end=token_ids['input_ids'].shape[-1])                                               
        segment_kv_list = []
        token_id_list = []
        for type_, _, token_id, s, e in segments:

            if type_ == "text":
                seg_kv = base_kv.slice(start=s, end=e)
                segment_kv_list.append(seg_kv)
                token_id_list.append(token_id)
        self._shared_kv_cache_memory[node_id]["prefix"] = LLMChat._shared_kv_cache_memory[node_id]["prefix"] = segment_kv_list          
        self._shared_kv_cache_memory[node_id]["placeholder_info"] = LLMChat._shared_kv_cache_memory[node_id]["placeholder_info"] = placeholder_info
        self._shared_kv_cache_memory[node_id]["token_ids"] = LLMChat._shared_kv_cache_memory[node_id]["token_ids"] = token_id_list            

        self._initialization[node_id] = LLMChat._initialization[node_id] = True

    def _initialize_shared_resources(self):
        """Lazy-load shared tokenizer/model and shared KV memory storage."""
        with LLMChat._model_lock:
            if LLMChat._shared_model is None:
                _paper_debug_runtime("[MODEL INIT] before tokenizer load")
                LLMChat._shared_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                _paper_debug_runtime("[MODEL INIT] tokenizer loaded")
                model_name_lower = self.model_name.lower()
                force_bfloat16 = os.environ.get("KVCOMM_FORCE_BFLOAT16", "0").lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
                if force_bfloat16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                elif "llama" in model_name_lower:
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        torch_dtype = torch.bfloat16

                _paper_debug_runtime("[MODEL INIT] before model/client load")
                LLMChat._shared_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True,
                )
                LLMChat._shared_model.eval()
                _paper_debug_runtime("[MODEL INIT] model/client loaded")
                logger.info("Model {} loaded and shared across instances.", self.model_name)
            if LLMChat._shared_kv_cache_memory is None:
                LLMChat._shared_kv_cache_memory = {}

    def locate_placeholder(self, original_text, return_segments=False):
        """Locate placeholder token spans in a templated prompt.

        Args:
            original_text: Templated prompt with placeholders such as
                "{agent_2_current}" or "{user_question}".
            return_segments: Whether to also return segment encodings.

        Returns:
            placeholder_info: Mapping placeholder -> [start, end] token indices.
            encoding or (encoding, segments): Tokenized input and optional segments.
        """

        placeholder_pattern = r'\{((?:agent|condition)_\w+_(?:current|history)|user_question)\}'

        matches = list(re.finditer(placeholder_pattern, original_text))

        last_pos = 0
        segments = []
        placeholder_info = {}
        token_num = 0
        idx_count = 0
        for m in matches:
            start, end = m.span()
            placeholder_inner = m.group(1)
            if last_pos < start:
                txt = original_text[last_pos:start]
                token_id = self.tokenizer(txt, add_special_tokens=False)['input_ids']
                encoding = {}
                encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
                encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
                if txt.strip():
                    segments.append(("text", txt, encoding, token_num, token_num + len(token_id)))
                    idx_count += 1
                token_num += len(token_id)
            token_id = self.tokenizer(f'{ {placeholder_inner}} ', add_special_tokens=False)['input_ids']
            encoding = {}
            encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
            encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
            segments.append(("placeholder", placeholder_inner, encoding, token_num, token_num + len(token_id)))
            placeholder_info[placeholder_inner] = [token_num, token_num + len(token_id)]
            token_num += len(token_id)
            idx_count += 1
            last_pos = end

        txt = original_text[last_pos:]
        token_id = self.tokenizer(txt, add_special_tokens=False)['input_ids']
        encoding = {}
        encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
        encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
        if txt.strip():
            segments.append(("text", txt, encoding, token_num, token_num + len(token_id)))
            token_num += len(token_id)

        segments.sort(key=lambda x: x[-1])
        token_ids = torch.cat([sublist[2]['input_ids'] for sublist in segments], dim=1)
        encoding = {}
        encoding['input_ids'] = token_ids
        encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)

        placeholder_info = dict(sorted(placeholder_info.items(), key=lambda x: x[1][0], reverse=True))
        if return_segments:
            return placeholder_info, encoding, segments
        return placeholder_info, encoding

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        pass

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_cache: Optional[bool] = False,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        async with self.lock:
            if max_tokens is None:
                max_tokens = self.DEFAULT_MAX_TOKENS
            explicit_temperature = temperature is not None
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE
            inputs, prompt_text, prompt_length = self._build_chat_inputs(messages)
            safe_prompt_text = _escape_loguru_markup(prompt_text)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT]</blue> Agent {} Role {} Prompt:\n{}",
                self.node_id,
                self.role,
                safe_prompt_text,
            )
            generation_kwargs = {
                "do_sample": explicit_temperature,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_dict_in_generate": True,
                "return_legacy_cache": False,
                "use_cache": True,
                "remove_invalid_values": True,
            }
            ttft_tracer = _TTFTTracer(prompt_length)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
            ttft_tracer.reset(prompt_length)
            outputs = self.model.generate(**inputs, **generation_kwargs)
            if ttft_tracer.ttft is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_value = 0.0
            else:
                ttft_value = ttft_tracer.ttft
            generated_sequence = outputs.sequences[:, prompt_length:]
            response_message = self.tokenizer.decode(
                generated_sequence[0], skip_special_tokens=True
            ).strip()
            safe_response_message = _escape_loguru_markup(response_message)
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE]</blue> Agent {} Role {} Response:\n{}",
                self.node_id,
                self.role,
                safe_response_message,
            )
            metadata: Dict[str, Any] = {}
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if agent_name:
                metadata["agent_name"] = agent_name
            if agent_role:
                metadata["agent_role"] = agent_role
            if return_cache:
                metadata["kv_cache"] = outputs.past_key_values
            return GenerationResult(
                text=response_message,
                mode="default",
                ttft=ttft_value,
                metadata=metadata,
            )

    @retry(wait=wait_random_exponential(max=1000), stop=stop_after_attempt(1))
    async def generate_with_dense_prefill(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_anchor_num: Optional[int] = 20,
        window_length: Optional[int] = 5,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate with dense prefix prefill and optional anchor update."""
        return await self.agen_kvcomm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            mode="dense_prefill",
            max_anchor_num=max_anchor_num,
            window_length=window_length,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir
        )

    @retry(wait=wait_random_exponential(max=1000), stop=stop_after_attempt(1))
    async def generate_with_kv_reuse(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate by reusing existing prefix KV (fast path)."""
        test_time = kwargs.get("test_time", False)
        if test_time:
            return await self.agen_kvcomm_time_test(
                messages=messages,
                max_tokens=max_tokens,
                min_tokens=max_tokens,
                temperature=temperature,
                request_uid=request_uid,
                mode="kv_reuse",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=output_dir,
            )
        return await self.agen_kvcomm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_uid=request_uid,
            mode="kv_reuse",
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir,
        )

    async def agen_kvcomm(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        max_anchor_num: int = 20,
        window_length: int = 5,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Core KV-aware generation entry.

        Builds merged prefix KV and token ids from stored segments and per-request
        placeholder caches, then runs generation either with:
        - dense_prefill: compute fresh prefix KV and optionally set anchors
        - kv_reuse: reuse existing prefix KV and inject as past_key_values
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        explicit_temperature = temperature is not None
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm.")
        state = self.kv_engine.resolve_request_state(request_uid)
        preprocess_start = perf_counter() if mode == "kv_reuse" else None

        if isinstance(messages, List):
            message = messages[0]
        else:
            message = messages

        prefix_store = self._shared_kv_cache_memory[self.node_id]
        prefix_kv_list: List[DynamicCache] = prefix_store.get("prefix", [])
        prefix_token_ids: List[Dict[str, torch.Tensor]] = prefix_store.get("token_ids", [])
        placeholder_info_map = prefix_store.get("placeholder_info")
        if not prefix_kv_list:
            raise RuntimeError(
                "No prefix KV found in shared memory. Make sure you've called prepare_prefix_kv_segments or init_shared_placeholder_prefix_kv."
            )
        if placeholder_info_map is None:
            raise RuntimeError("placeholder_info missing in shared KV cache memory.")

        merged_prefix_kv = prefix_kv_list[0].copy()
        merged_prefix_token_ids = prefix_token_ids[0].copy()

        placeholder_entries = list(placeholder_info_map.items())[::-1]

        meta: List[Dict[str, Any]] = []
        ph_id_list: List[str] = []
        judge_pos_map: Dict[str, int] = {}
        judge_pos_counter = 0
        cum_offset = 0
        ph_cum_len = 0
        for idx, ((ph_id, (start, end)), pf_kv, pf_token_id) in enumerate(
            zip(placeholder_entries, prefix_kv_list[1:], prefix_token_ids[1:])
        ):
            ph_cache, ph_cache_ids, drop_num = self.kv_engine.fetch_shared_cache(ph_id, message)
            real_len = ph_cache._seen_tokens - drop_num
            templ_len = end - start
            delta_len = real_len - templ_len
            meta.append(
                {
                    "idx": idx,
                    "ph_id": ph_id,
                    "start": start,
                    "end": end,
                    "drop_num": drop_num,
                    "delta": delta_len,
                    "offset_before": cum_offset,
                    "offset_after": cum_offset + delta_len,
                    "ph_cache": ph_cache,
                    "ph_cache_ids": ph_cache_ids,
                    "pf_kv": pf_kv,
                    "pf_ids": pf_token_id,
                    "cum_len": ph_cum_len,
                }
            )
            cum_offset += delta_len
            ph_cum_len += real_len
            ph_id_list.append(ph_id)
            if (
                getattr(self.config, "group_judge_agent_anchors_by_position", False)
                and getattr(self, "role", None) == "FinalSelectBest"
                and isinstance(ph_id, str)
                and ph_id.startswith("agent_")
            ):
                judge_pos_counter += 1
                judge_pos_map[ph_id] = judge_pos_counter

        if (
            getattr(self.config, "group_judge_agent_anchors_by_position", False)
            and getattr(self, "role", None) == "FinalSelectBest"
            and judge_pos_map
        ):
            # Used by _collect_anchors_for_placeholder to map placeholder->display slot.
            state.judge_display_pos_by_ph_id = judge_pos_map

        if mode == "dense_prefill":
            tasks = [(message, m) for m in meta]
            results = list(
                self._map_in_pool(self.kv_engine.process_anchor, tasks, timeout=30)
            )
        elif mode == "kv_reuse":
            results = []
            for m in meta:
                ph_id = m["ph_id"]
                if self._placeholder_is_naive(ph_id):
                    for res in self._map_in_pool(self.kv_engine.process_anchor, [(message, m)], timeout=30):
                        results.append(res)
                else:
                    anchors_for_ph = self._collect_anchors_for_placeholder(state, ph_id)
                    for res in self._map_in_pool(
                        self.kv_engine.update_kv_cache_segment,
                        [(request_uid, message, m, anchors_for_ph)],
                        timeout=30,
                    ):
                        results.append(res)
        else:
            raise ValueError(f"Unsupported mode '{mode}' for agen_kvcomm.")

        results_sorted = sorted(results, key=lambda x: x[0])

        placeholder_indices: Dict[str, Tuple[int, int]] = {}
        for m in meta:
            start = m["start"] + m["offset_before"]
            placeholder_indices[m["ph_id"]] = (
                start,
                start + m["ph_cache"]._seen_tokens - m["drop_num"],
            )

        seg_cache_list = [r[1] for r in results_sorted]
        merged_prefix_kv.concat_(seg_cache_list)
        seg_ids_list = [r[2] for r in results_sorted]
        merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)

        prefix_token_length = merged_prefix_kv.get_seq_length()
        input_length = merged_prefix_token_ids["input_ids"].shape[-1]
        if input_length != prefix_token_length:
            logger.warning(
                "prefix_token_length: {} merged_length: {}",
                prefix_token_length,
                input_length,
            )
            raise RuntimeError("prefix_token_length != merged_prefix_token_ids['input_ids'].shape[-1]")

        if "position_ids" in merged_prefix_token_ids:
            merged_prefix_token_ids["position_ids"] = (
                torch.arange(input_length).unsqueeze(0).to(self.model.device)
            )

        generation_kwargs: Dict[str, Any] = {
            "max_length": max_tokens + prefix_token_length,
            "do_sample": explicit_temperature,
            "temperature": temperature,
            "return_legacy_cache": False,
            "return_dict_in_generate": True,
            "remove_invalid_values": True,
        }

        if mode == "kv_reuse":
            merged_prefix_kv = merged_prefix_kv.slice_(start=0, end=prefix_token_length - 1)
            generation_kwargs["past_key_values"] = merged_prefix_kv

        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        preprocess_latency = 0.0
        if preprocess_start is not None:
            preprocess_latency = max(0.0, perf_counter() - preprocess_start)
        outputs = self.model.generate(**merged_prefix_token_ids, **generation_kwargs)
        if ttft_tracer.ttft is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_ttft = 0.0
        else:
            generation_ttft = ttft_tracer.ttft
        ttft_value = generation_ttft
        if preprocess_start is not None:
            ttft_value += preprocess_latency

        full_kv_cache = outputs.past_key_values

        if mode == "dense_prefill":
            base_cache = merged_prefix_kv
            real_cache = full_kv_cache.slice(start=0, end=prefix_token_length)
            real_placeholder_cache, real_prefix_cache = real_cache.split_cache_by_placeholders(
                placeholder_indices
            )
            base_placeholder_cache, base_prefix_cache = base_cache.split_cache_by_placeholders(
                placeholder_indices
            )
            filtered_ph_ids = []
            filtered_real_ph = []
            filtered_real_pf = []
            filtered_base_ph = []
            filtered_base_pf = []
            for ph_id, r_ph, r_pf, b_ph, b_pf in zip(
                ph_id_list,
                real_placeholder_cache,
                real_prefix_cache,
                base_placeholder_cache,
                base_prefix_cache,
            ):
                if self._placeholder_is_naive(ph_id):
                    continue
                filtered_ph_ids.append(ph_id)
                filtered_real_ph.append(r_ph)
                filtered_real_pf.append(r_pf)
                filtered_base_ph.append(b_ph)
                filtered_base_pf.append(b_pf)
            if filtered_ph_ids:
                self.kv_engine.set_anchor(
                    request_uid,
                    message,
                    filtered_ph_ids,
                    filtered_real_ph,
                    filtered_real_pf,
                    filtered_base_ph,
                    filtered_base_pf,
                    max_anchor_num=max_anchor_num,
                    window_length=window_length,
                )

        response_kv_cache = full_kv_cache.slice_(start=prefix_token_length)
        response_kv_cache = self.kv_engine.apply_rotary_pos_emb(
            response_kv_cache,
            offset=-prefix_token_length,
        )

        mem = LLMChat._shared_kv_cache_memory[self.node_id]
        resp = mem.setdefault("response", {})
        resp_ids = mem.setdefault("response_ids", {})
        resp_drop = mem.setdefault("response_drop_num", {})

        seq = outputs.sequences
        response_tokens = seq[:, prefix_token_length:-1]
        attn_len = response_tokens.size(1)
        response_mask = torch.ones(seq.size(0), attn_len, device=self.model.device)

        current_key = f"agent_{self.node_id}_current"
        anchor_bucket = state.anchors.setdefault(current_key, {})
        anchor_len_bucket = state.anchor_len_dict.setdefault(current_key, {})
        anchor_info_bucket = state.anchor_info_dict.setdefault(current_key, {})
        response_anchor_list = list(anchor_bucket.values())
        anchor_len_list = [
            anchor_len_bucket.get(kk, [0, 0])
            for kk in anchor_bucket.keys()
        ]
        anchor_active_list: List[int] = list(anchor_info_bucket.values())

        resp.setdefault(message, []).append(response_kv_cache)
        resp_ids.setdefault(message, []).append(
            {
                "input_ids": response_tokens,
                "attention_mask": response_mask,
            }
        )
        resp_drop.setdefault(message, []).append(0)

        accumulate_len = 0
        for key in state.anchor_len_dict.keys():
            bucket = state.anchor_len_dict.get(key, {})
            length_entry = bucket.get(message, [0, 0])
            accumulate_len += length_entry[0]

        if self._placeholder_is_naive(current_key):
            # Naive placeholders should not maintain anchor views; otherwise their
            # anchor flags can force downstream nodes (e.g. the judge) into
            # `dense_prefill` unexpectedly.
            prob = False
        else:
            prob, anchor_active_list = self.kv_engine.predict_as_anchor(
                response_kv_cache,
                anchor_kv_cache_list=response_anchor_list,
                anchor_len_list=anchor_len_list,
                anchor_activated_list=anchor_active_list,
            )
        safe_message = _escape_loguru_markup(message)
        logger.opt(colors=True).debug(
            f"<magenta>Agent {self.node_id} Role {self.role} Message {safe_message} Response Anchor Prediction: {prob}</magenta>",
        )
        state.anchor_dict.setdefault(current_key, {})[message] = prob

        if not prob and not self._placeholder_is_naive(current_key):
            global_bucket = state.global_anchor_info.setdefault(current_key, {})
            info_items = list(anchor_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                anchor_info_bucket[msg_key] = anchor_active_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_active_list[idx]

        response_message = self.tokenizer.decode(
            outputs.sequences[0, prefix_token_length:],
            skip_special_tokens=True,
        )
        prompt_preview = self.tokenizer.decode(
            merged_prefix_token_ids["input_ids"][0]
        )
        safe_prompt_preview = _escape_loguru_markup(prompt_preview)
        safe_response_message = _escape_loguru_markup(response_message)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:{mode}]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id,
            self.role,
            safe_prompt_preview,
            mode=mode,
        )
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:{mode}]</blue> Agent {} Role {} Response:\n{}",
            self.node_id,
            self.role,
            safe_response_message,
            mode=mode,
        )

        metadata: Dict[str, Any] = {
            "placeholder_ids": ph_id_list,
        }
        if (
            request_uid
            and getattr(self.config, "group_judge_agent_anchors", False)
            and getattr(self, "role", None) == "FinalSelectBest"
        ):
            try:
                state = self.kv_engine.resolve_request_state(request_uid)
                usage = getattr(state, "judge_group_anchor_usage", None)
                if usage:
                    metadata["judge_group_anchor_usage"] = usage
            except Exception:
                pass
        if preprocess_start is not None:
            metadata["preprocess_latency"] = preprocess_latency
            metadata["generation_ttft"] = generation_ttft
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        latency_record = {
            "timestamp": time.time(),
            "mode": mode,
            "ttft": float(ttft_value),
            "generation_ttft": float(generation_ttft),
            "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
            "request_uid": request_uid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "message": str(message) if message is not None else None,
            "placeholder_ids": ph_id_list,
        }
        _append_latency_record(output_dir, latency_record)
        return GenerationResult(
            text=response_message,
            mode=mode,
            ttft=ttft_value,
            metadata=metadata,
        )

    def _masked_prefill_agent_block_spans(
        self,
        *,
        prompt_text: str,
        input_ids: torch.Tensor,
        candidate_agent_ids: List[str],
    ) -> Tuple[int, List[Tuple[int, int]], int]:
        """Return (prefix_end, block_token_spans, tail_start) in token indices.

        The intent is to isolate each `Agent <id>:` block so its KV can be
        computed without attending to earlier agent blocks.
        """
        tokenized = self.tokenizer(
            prompt_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None,
        )
        offsets = tokenized.get("offset_mapping")
        if offsets is None:
            raise RuntimeError("Tokenizer did not return offset mapping; need a fast tokenizer.")
        offsets_list = offsets if isinstance(offsets, list) else offsets[0]
        seq_len = int(input_ids.shape[-1])
        if len(offsets_list) != seq_len:
            raise RuntimeError(
                f"Offset mapping length {len(offsets_list)} != input_ids length {seq_len}"
            )

        marker_re = re.compile(r"^Agent\s+(\d+)\s*:\s*$", re.MULTILINE)
        markers: Dict[str, int] = {}
        for match in marker_re.finditer(prompt_text):
            agent_id = match.group(1)
            markers.setdefault(agent_id, match.start())

        ordered_char_starts: List[int] = []
        for agent_id in candidate_agent_ids:
            start = markers.get(str(agent_id))
            if start is None:
                raise RuntimeError(f"Could not locate Agent {agent_id}: marker in prompt_text.")
            ordered_char_starts.append(start)

        prefix_char_end = min(ordered_char_starts)
        last_marker = max(ordered_char_starts)
        instr_match = re.search(
            r"^##\s+Instructions\s*:\s*$",
            prompt_text[last_marker:],
            re.MULTILINE,
        )
        tail_char_start = (last_marker + instr_match.start()) if instr_match else len(prompt_text)

        ordered_char_ends: List[int] = ordered_char_starts[1:] + [tail_char_start]

        def _char_to_tok_index(char_pos: int) -> int:
            for idx, (s, e) in enumerate(offsets_list):
                if e > char_pos:
                    return idx
            return seq_len

        def _char_span_to_tok_span(start_c: int, end_c: int) -> Tuple[int, int]:
            return _char_to_tok_index(start_c), _char_to_tok_index(end_c)

        _, prefix_end = _char_span_to_tok_span(0, prefix_char_end)
        _, tail_start = _char_span_to_tok_span(0, tail_char_start)

        block_spans: List[Tuple[int, int]] = []
        for s_c, e_c in zip(ordered_char_starts, ordered_char_ends):
            s_t, e_t = _char_span_to_tok_span(s_c, e_c)
            block_spans.append((s_t, e_t))

        return prefix_end, block_spans, tail_start

    async def agen_mask_prev_agent_attn(
        self,
        *,
        messages: List[Message],
        candidate_agent_ids: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a judge response with masked prefill between candidate agent blocks.

        Instead of modifying attention masks inside the model, we approximate the
        blocking by computing each `Agent <id>:` block's KV cache from a separate
        dense prefill over (prefix + that block), then stitching these KV segments
        back together with RoPE key rotation so their absolute positions match
        the original prompt.
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        explicit_temperature = temperature is not None
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        inputs, prompt_text, prompt_len = self._build_chat_inputs(messages)
        input_ids: torch.Tensor = inputs["input_ids"]

        prefix_end, block_spans, tail_start = self._masked_prefill_agent_block_spans(
            prompt_text=prompt_text,
            input_ids=input_ids,
            candidate_agent_ids=candidate_agent_ids,
        )
        prefix_tokens = input_ids[:, :prefix_end]
        if prefix_tokens.numel() == 0:
            raise RuntimeError("Masked prefill: empty prefix segment.")

        with torch.no_grad():
            out_prefix = self.model(
                input_ids=prefix_tokens,
                use_cache=True,
                return_dict=True,
                return_legacy_cache=False,
            )
        merged_cache = out_prefix.past_key_values

        for block_start, block_end in block_spans:
            block_tokens = input_ids[:, block_start:block_end]
            if block_tokens.numel() == 0:
                continue
            local_ids = torch.cat([prefix_tokens, block_tokens], dim=-1)
            with torch.no_grad():
                out_local = self.model(
                    input_ids=local_ids,
                    use_cache=True,
                    return_dict=True,
                    return_legacy_cache=False,
                )
            local_cache = out_local.past_key_values
            local_block_cache = local_cache.copy().slice_(
                start=prefix_end, end=prefix_end + int(block_tokens.shape[-1])
            )
            delta = int(block_start) - int(prefix_end)
            if delta != 0:
                local_block_cache = self.kv_engine.apply_rotary_pos_emb(
                    local_block_cache,
                    offset=delta,
                    drop_num=0,
                )
            merged_cache.concat_([local_block_cache])

        tail_tokens = input_ids[:, tail_start:]
        if tail_tokens.numel() > 0:
            with torch.no_grad():
                out_tail = self.model(
                    input_ids=tail_tokens,
                    past_key_values=merged_cache,
                    use_cache=True,
                    return_dict=True,
                    return_legacy_cache=False,
                )
            merged_cache = out_tail.past_key_values

        full_len = int(input_ids.shape[-1])
        if merged_cache.get_seq_length() != full_len:
            raise RuntimeError(
                f"Masked prefill cache length mismatch: {merged_cache.get_seq_length()} != {full_len}"
            )

        past_before_last = merged_cache.slice(start=0, end=full_len - 1)
        gen_kwargs: Dict[str, Any] = {
            "do_sample": explicit_temperature,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_dict_in_generate": True,
            "return_legacy_cache": False,
            "use_cache": True,
            "remove_invalid_values": True,
            "past_key_values": past_before_last,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        seq = outputs.sequences
        response_text = self.tokenizer.decode(
            seq[0, full_len:],
            skip_special_tokens=True,
        )
        metadata: Dict[str, Any] = {
            "masked_prefill": True,
            "prompt_length": int(prompt_len),
            "candidate_agent_ids": list(candidate_agent_ids),
        }
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        return GenerationResult(
            text=response_text,
            mode="masked_prefill",
            ttft=0.0,
            metadata=metadata,
        )

    async def agen_kvcomm_time_test(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        max_anchor_num: int = 20,
        window_length: int = 5,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Core KV-aware generation entry.

        Builds merged prefix KV and token ids from stored segments and per-request
        placeholder caches, then runs generation either with:
        - dense_prefill: compute fresh prefix KV and optionally set anchors
        - kv_reuse: reuse existing prefix KV and inject as past_key_values
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm.")
        min_tokens = max_tokens if min_tokens is None else min_tokens
        state = self.kv_engine.resolve_request_state(request_uid)
        preprocess_start = perf_counter() if mode == "kv_reuse" else None

        if isinstance(messages, List):
            message = messages[0]
        else:
            message = messages

        prefix_store = self._shared_kv_cache_memory[self.node_id]
        prefix_kv_list: List[DynamicCache] = prefix_store.get("prefix", [])
        prefix_token_ids: List[Dict[str, torch.Tensor]] = prefix_store.get("token_ids", [])
        placeholder_info_map = prefix_store.get("placeholder_info")
        if not prefix_kv_list:
            raise RuntimeError(
                "No prefix KV found in shared memory. Make sure you've called prepare_prefix_kv_segments or init_shared_placeholder_prefix_kv."
            )
        if placeholder_info_map is None:
            raise RuntimeError("placeholder_info missing in shared KV cache memory.")

        merged_prefix_kv = prefix_kv_list[0].copy()
        merged_prefix_token_ids = prefix_token_ids[0].copy()

        placeholder_entries = list(placeholder_info_map.items())[::-1]

        meta: List[Dict[str, Any]] = []
        ph_id_list: List[str] = []
        cum_offset = 0
        ph_cum_len = 0
        for idx, ((ph_id, (start, end)), pf_kv, pf_token_id) in enumerate(
            zip(placeholder_entries, prefix_kv_list[1:], prefix_token_ids[1:])
        ):
            ph_cache, ph_cache_ids, drop_num = self.kv_engine.fetch_shared_cache(ph_id, message)
            real_len = ph_cache._seen_tokens - drop_num
            templ_len = end - start
            delta_len = real_len - templ_len
            meta.append(
                {
                    "idx": idx,
                    "ph_id": ph_id,
                    "start": start,
                    "end": end,
                    "drop_num": drop_num,
                    "delta": delta_len,
                    "offset_before": cum_offset,
                    "offset_after": cum_offset + delta_len,
                    "ph_cache": ph_cache,
                    "ph_cache_ids": ph_cache_ids,
                    "pf_kv": pf_kv,
                    "pf_ids": pf_token_id,
                    "cum_len": ph_cum_len,
                }
            )
            cum_offset += delta_len
            ph_cum_len += real_len
            ph_id_list.append(ph_id)

        if mode == "dense_prefill":
            tasks = [(message, m) for m in meta]
            results = list(
                self._map_in_pool(self.kv_engine.process_anchor, tasks, timeout=30)
            )
        elif mode == "kv_reuse":
            tasks = [
                (
                    request_uid,
                    message,
                    m,
                    self._collect_anchors_for_placeholder(state, m["ph_id"]),
                )
                for m in meta
            ]
            results = list(
                self._map_in_pool(self.kv_engine.update_kv_cache_segment, tasks, timeout=30)
            )
        else:
            raise ValueError(f"Unsupported mode '{mode}' for agen_kvcomm.")

        results_sorted = sorted(results, key=lambda x: x[0])

        placeholder_indices: Dict[str, Tuple[int, int]] = {}
        for m in meta:
            start = m["start"] + m["offset_before"]
            placeholder_indices[m["ph_id"]] = (
                start,
                start + m["ph_cache"]._seen_tokens - m["drop_num"],
            )

        seg_cache_list = [r[1] for r in results_sorted]
        merged_prefix_kv.concat_(seg_cache_list)
        seg_ids_list = [r[2] for r in results_sorted]
        merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)

        prefix_token_length = merged_prefix_kv.get_seq_length()
        input_length = merged_prefix_token_ids["input_ids"].shape[-1]
        if input_length != prefix_token_length:
            logger.warning(
                "prefix_token_length: {} merged_length: {}",
                prefix_token_length,
                input_length,
            )
            raise RuntimeError("prefix_token_length != merged_prefix_token_ids['input_ids'].shape[-1]")

        if "position_ids" in merged_prefix_token_ids:
            merged_prefix_token_ids["position_ids"] = (
                torch.arange(input_length).unsqueeze(0).to(self.model.device)
            )

        generation_kwargs: Dict[str, Any] = {
            "max_length": max_tokens + prefix_token_length,
            "min_new_tokens": min_tokens,
            "do_sample": False,
            "temperature": temperature,
            "return_legacy_cache": False,
            "return_dict_in_generate": True,
            "remove_invalid_values": True,
        }

        if mode == "kv_reuse":
            merged_prefix_kv = merged_prefix_kv.slice_(start=0, end=prefix_token_length - 1)
            generation_kwargs["past_key_values"] = merged_prefix_kv

        preprocess_latency = 0.0
        if preprocess_start is not None:
            torch.cuda.synchronize()
            preprocess_latency = max(0.0, perf_counter() - preprocess_start)
        torch.cuda.synchronize()
        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        outputs = self.model.generate(**merged_prefix_token_ids, **generation_kwargs)
        torch.cuda.synchronize()
        if mode == "kv_reuse" and preprocess_start is not None:
            kvcomm_end_to_end_latency = perf_counter() - ttft_tracer.start_time
            kvcomm_ttft_value = ttft_tracer.ttft + preprocess_latency
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} KVCOMM E2E Latency: {kvcomm_end_to_end_latency:.4f}s TTFT: {kvcomm_ttft_value:.4f}s (Preprocess: {preprocess_latency:.4f}s)</green>",
            )
        full_kv_cache = outputs.past_key_values

        generation_kwargs.pop("past_key_values", None)
        torch.cuda.synchronize()
        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        _ = self.model.generate(**merged_prefix_token_ids, **generation_kwargs)
        torch.cuda.synchronize()
        dense_end_to_end_latency = perf_counter() - ttft_tracer.start_time
        dense_prefill_ttft = ttft_tracer.ttft
        logger.opt(colors=True).info(
            f"<cyan>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} Dense Prefill E2E Latency: {dense_end_to_end_latency:.4f}s TTFT: {dense_prefill_ttft:.4f}s</cyan>",
        )
        if mode == "kv_reuse" and preprocess_start is not None and kvcomm_ttft_value > 0:
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} KVCOMM is {dense_prefill_ttft / kvcomm_ttft_value:.2f}x faster than Dense Prefill in TTFT</green>",
            )
            ttft_value = kvcomm_ttft_value
        else:
            ttft_value = dense_prefill_ttft
        response_kv_cache = full_kv_cache.slice_(start=prefix_token_length)
        response_kv_cache = self.kv_engine.apply_rotary_pos_emb(
            response_kv_cache,
            offset=-prefix_token_length,
        )

        mem = LLMChat._shared_kv_cache_memory[self.node_id]
        resp = mem.setdefault("response", {})
        resp_ids = mem.setdefault("response_ids", {})
        resp_drop = mem.setdefault("response_drop_num", {})

        seq = outputs.sequences
        response_tokens = seq[:, prefix_token_length:-1]
        attn_len = response_tokens.size(1)
        response_mask = torch.ones(seq.size(0), attn_len, device=self.model.device)

        current_key = f"agent_{self.node_id}_current"
        anchor_bucket = state.anchors.setdefault(current_key, {})
        anchor_len_bucket = state.anchor_len_dict.setdefault(current_key, {})
        anchor_info_bucket = state.anchor_info_dict.setdefault(current_key, {})
        response_anchor_list = list(anchor_bucket.values())
        anchor_len_list = [
            anchor_len_bucket.get(kk, [0, 0])
            for kk in anchor_bucket.keys()
        ]
        anchor_active_list: List[int] = list(anchor_info_bucket.values())

        resp.setdefault(message, []).append(response_kv_cache)
        resp_ids.setdefault(message, []).append(
            {
                "input_ids": response_tokens,
                "attention_mask": response_mask,
            }
        )
        resp_drop.setdefault(message, []).append(0)

        accumulate_len = 0
        for key in state.anchor_len_dict.keys():
            bucket = state.anchor_len_dict.get(key, {})
            length_entry = bucket.get(message, [0, 0])
            accumulate_len += length_entry[0]

        prob, anchor_active_list = self.kv_engine.predict_as_anchor(
            response_kv_cache,
            anchor_kv_cache_list=response_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_active_list,
            test_time=True,
        )
        safe_message = _escape_loguru_markup(message)
        logger.opt(colors=True).debug(
            f"<magenta>Agent {self.node_id} Role {self.role} Message {safe_message} Response Anchor Prediction: {prob}</magenta>",
        )
        state.anchor_dict.setdefault(current_key, {})[message] = prob

        if not prob:
            global_bucket = state.global_anchor_info.setdefault(current_key, {})
            info_items = list(anchor_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                anchor_info_bucket[msg_key] = anchor_active_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_active_list[idx]

        response_message = self.tokenizer.decode(
            outputs.sequences[0, prefix_token_length:],
            skip_special_tokens=True,
        )
        prompt_preview = self.tokenizer.decode(
            merged_prefix_token_ids["input_ids"][0]
        )
        safe_prompt_preview = _escape_loguru_markup(prompt_preview)
        safe_response_message = _escape_loguru_markup(response_message)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:{mode}]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id,
            self.role,
            safe_prompt_preview,
            mode=mode,
        )
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:{mode}]</blue> Agent {} Role {} Response:\n{}",
            self.node_id,
            self.role,
            safe_response_message,
            mode=mode,
        )

        metadata: Dict[str, Any] = {
            "placeholder_ids": ph_id_list,
        }
        if preprocess_start is not None:
            metadata["preprocess_latency"] = preprocess_latency
            metadata["generation_ttft"] = ttft_value - preprocess_latency
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        if mode == "kv_reuse":
            latency_record = {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(ttft_value),
                "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "dense_prefill_ttft": float(dense_prefill_ttft),
                "kvcomm_end_to_end_latency": float(kvcomm_end_to_end_latency),
                "dense_end_to_end_latency": float(dense_end_to_end_latency),
                "ttft_ratio_dense_over_kvcomm": float(dense_prefill_ttft / ttft_value) if ttft_value > 0 else None,
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "message": str(message) if message is not None else None,
                "placeholder_ids": ph_id_list,
            }
        else:
            latency_record = {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(ttft_value),
                "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "dense_prefill_ttft": float(dense_prefill_ttft),
                "dense_end_to_end_latency": float(dense_end_to_end_latency),
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "message": str(message) if message is not None else None,
                "placeholder_ids": ph_id_list,
            }
        _append_latency_record(output_dir, latency_record)
        return GenerationResult(
            text=response_message,
            mode=mode,
            ttft=ttft_value,
            metadata=metadata,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['tokenizer']
        del state['lock']
        del state['_shared_kv_cache_memory']
        del state['_initialization']
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._shared_kv_cache_memory = LLMChat._shared_kv_cache_memory
        self._initialization = LLMChat._initialization
        self.lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = self.__getstate__()
        copied_state = copy.deepcopy(state, memo)
        node_id = copied_state.get('node_id', None)
        role = copied_state.get('role', None)
        if node_id is not None:
            if node_id in LLMChat._shared_kv_cache_memory:
                original_cache = LLMChat._shared_kv_cache_memory[node_id]
                LLMChat._shared_kv_cache_memory[node_id] = {
                    "prefix": original_cache.get("prefix"),
                    "placeholder_info": original_cache.get("placeholder_info"),
                    "token_ids": original_cache.get("token_ids"),
                    "input": {},
                    "response": {},
                    "response_ids": {},
                    "condition": {},
                    "condition_ids": {},
                    "input_drop_num": {},
                    "response_drop_num": {},
                    "condition_drop_num": {},
                }
                LLMChat.weight_dict = {}
            result.set_id(node_id, role)
        result.__setstate__(copied_state)
        return result
