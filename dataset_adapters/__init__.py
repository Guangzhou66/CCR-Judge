from __future__ import annotations

import os

from dataset_adapters.base_adapter import AdapterRecord, DatasetAdapter


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


def create_dataset_adapter(name: str, **kwargs) -> DatasetAdapter:
    normalized = str(name).strip().lower()
    if normalized == "mmlu":
        _paper_debug_import("before importing mmlu_dataset")
        from dataset_adapters.mmlu_adapter import MMLUAdapter

        _paper_debug_import("imported mmlu_dataset")
        return MMLUAdapter(**kwargs)
    if normalized == "gsm8k":
        from dataset_adapters.gsm8k_adapter import GSM8KAdapter

        return GSM8KAdapter(**kwargs)
    if normalized == "humaneval":
        from dataset_adapters.humaneval_adapter import HumanEvalAdapter

        return HumanEvalAdapter(**kwargs)
    if normalized == "openbookqa":
        from dataset_adapters.openbookqa_adapter import OpenBookQAAdapter

        return OpenBookQAAdapter(**kwargs)
    raise ValueError(f"Unsupported dataset adapter: {name}")


def __getattr__(name: str):
    if name == "MMLUAdapter":
        from dataset_adapters.mmlu_adapter import MMLUAdapter

        return MMLUAdapter
    if name == "GSM8KAdapter":
        from dataset_adapters.gsm8k_adapter import GSM8KAdapter

        return GSM8KAdapter
    if name == "HumanEvalAdapter":
        from dataset_adapters.humaneval_adapter import HumanEvalAdapter

        return HumanEvalAdapter
    if name == "OpenBookQAAdapter":
        from dataset_adapters.openbookqa_adapter import OpenBookQAAdapter

        return OpenBookQAAdapter
    if name in {"AdapterRecord", "DatasetAdapter", "create_dataset_adapter"}:
        return globals()[name]
    raise AttributeError(name)


__all__ = [
    "AdapterRecord",
    "DatasetAdapter",
    "MMLUAdapter",
    "GSM8KAdapter",
    "HumanEvalAdapter",
    "OpenBookQAAdapter",
    "create_dataset_adapter",
]
