__all__ = [
    "LLMRegistry",
    "VisualLLMRegistry",
    "GPTChat",
    "LLMChat",
    "KVCommConfig",
]


def __getattr__(name):
    if name == "KVCommConfig":
        from KVCOMM.llm.config import KVCommConfig

        return KVCommConfig
    if name == "LLMRegistry":
        from KVCOMM.llm.llm_registry import LLMRegistry

        return LLMRegistry
    if name == "VisualLLMRegistry":
        from KVCOMM.llm.visual_llm_registry import VisualLLMRegistry

        return VisualLLMRegistry
    if name in {"GPTChat", "LLMChat"}:
        from KVCOMM.llm.gpt_chat import GPTChat, LLMChat

        return {"GPTChat": GPTChat, "LLMChat": LLMChat}[name]
    raise AttributeError(name)
