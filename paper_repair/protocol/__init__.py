__all__ = [
    "DEFAULT_INDEX_FILE",
    "DEFAULT_METHODS",
    "JudgeMethodSpec",
    "METHOD_SPECS",
    "build_frozen_candidate_pack",
    "load_dataset_records",
    "load_frozen_candidate_pack",
    "load_mmlu_records",
    "load_question_indices",
    "run_full_153_benchmark",
    "run_judge_benchmark",
]


def __getattr__(name):
    if name in {
        "DEFAULT_INDEX_FILE",
        "build_frozen_candidate_pack",
        "load_dataset_records",
        "load_frozen_candidate_pack",
        "load_mmlu_records",
        "load_question_indices",
    }:
        from paper_repair.protocol.frozen_pack import (
            DEFAULT_INDEX_FILE,
            build_frozen_candidate_pack,
            load_dataset_records,
            load_frozen_candidate_pack,
            load_mmlu_records,
            load_question_indices,
        )

        return {
            "DEFAULT_INDEX_FILE": DEFAULT_INDEX_FILE,
            "build_frozen_candidate_pack": build_frozen_candidate_pack,
            "load_dataset_records": load_dataset_records,
            "load_frozen_candidate_pack": load_frozen_candidate_pack,
            "load_mmlu_records": load_mmlu_records,
            "load_question_indices": load_question_indices,
        }[name]
    if name in {"DEFAULT_METHODS", "JudgeMethodSpec", "METHOD_SPECS"}:
        from paper_repair.protocol.method_specs import DEFAULT_METHODS, JudgeMethodSpec, METHOD_SPECS

        return {
            "DEFAULT_METHODS": DEFAULT_METHODS,
            "JudgeMethodSpec": JudgeMethodSpec,
            "METHOD_SPECS": METHOD_SPECS,
        }[name]
    if name in {"run_full_153_benchmark", "run_judge_benchmark"}:
        from paper_repair.protocol.runner import run_full_153_benchmark, run_judge_benchmark

        return {
            "run_full_153_benchmark": run_full_153_benchmark,
            "run_judge_benchmark": run_judge_benchmark,
        }[name]
    raise AttributeError(name)
