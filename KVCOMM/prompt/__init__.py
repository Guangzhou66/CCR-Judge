from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry

__all__ = [
    "MMLUPromptSet",
    "HumanEvalPromptSet",
    "GSM8KPromptSet",
    "COPYpromptSet",
    "PromptSetRegistry",
]


def __getattr__(name):
    if name == "MMLUPromptSet":
        from KVCOMM.prompt.mmlu_prompt_set import MMLUPromptSet

        return MMLUPromptSet
    if name == "HumanEvalPromptSet":
        from KVCOMM.prompt.humaneval_prompt_set import HumanEvalPromptSet

        return HumanEvalPromptSet
    if name == "GSM8KPromptSet":
        from KVCOMM.prompt.gsm8k_prompt_set import GSM8KPromptSet

        return GSM8KPromptSet
    if name == "COPYpromptSet":
        from KVCOMM.prompt.copy_machine_prompt_set import COPYpromptSet

        return COPYpromptSet
    if name == "PromptSetRegistry":
        return PromptSetRegistry
    raise AttributeError(name)
