from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm.auto import tqdm

from paper_repair.eval.details import build_case_evaluation_row, case_evaluation_row_to_dict
from paper_repair.eval.summary import (
    MAIN_TABLE_COLUMNS,
    dump_method_outputs,
    json_dump,
    markdown_table,
    method_summary,
    write_benchmark_outputs,
)
from paper_repair.protocol.frozen_pack import (
    DEFAULT_INDEX_FILE,
    build_frozen_candidate_pack,
    frozen_record_from_dict,
    load_frozen_candidate_pack,
    load_question_indices,
)
from paper_repair.protocol.method_specs import DEFAULT_METHODS, METHOD_SPECS, validate_requested_methods
from paper_repair.protocol.permutation import candidate_permutation
from paper_repair.repair.ccr_types import FrozenCandidateRecord, JudgeMethodResult


@contextmanager
def temporary_env(overrides: Dict[str, str]):
    import os

    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _record_with_permutation(record: FrozenCandidateRecord, permutation: Sequence[str]) -> FrozenCandidateRecord:
    return FrozenCandidateRecord(
        record_index=record.record_index,
        question_text=record.question_text,
        gold_answer=record.gold_answer,
        generation_regime=record.generation_regime,
        candidates=list(record.candidates),
        permutation=[str(item) for item in permutation],
        dataset_name=record.dataset_name,
        dataset_config=record.dataset_config,
        task_id=record.task_id,
        reference_reasoning=record.reference_reasoning,
        target_payload=dict(record.target_payload),
    )


async def _run_method_results(
    *,
    spec: Any,
    cases: Sequence[FrozenCandidateRecord],
    model_name: str,
    output_root: Path,
    judge_shuffle: bool,
    shuffle_seed: int,
) -> List[tuple[FrozenCandidateRecord, JudgeMethodResult]]:
    if not cases:
        return []

    dataset_name = str(cases[0].dataset_name or "mmlu")
    records: List[tuple[FrozenCandidateRecord, JudgeMethodResult]] = []
    with temporary_env(spec.env):
        judge = spec.judge_factory(model_name, domain=dataset_name)
        for record in tqdm(
            cases,
            desc=f"judge:{spec.name}:{'shuffle' if judge_shuffle else 'noshuffle'}",
            unit="case",
        ):
            runtime_record = _record_with_permutation(
                record,
                candidate_permutation(
                    record,
                    judge_shuffle=judge_shuffle,
                    shuffle_seed=shuffle_seed,
                ),
            )
            method_result = await spec.run_case(
                judge=judge,
                record=runtime_record,
                output_dir=output_root / spec.name,
            )
            records.append((runtime_record, method_result))
    return records


async def run_judge_benchmark(
    *,
    frozen_pack: Dict[str, Any] | str | Path,
    output_dir: str | Path,
    methods: Sequence[str] | None = None,
    llm_name: str | None = None,
    judge_shuffle: bool = False,
    shuffle_seed: int = 42,
) -> Dict[str, Any]:
    if not isinstance(frozen_pack, dict):
        frozen_pack = load_frozen_candidate_pack(frozen_pack)
    requested_methods = validate_requested_methods(methods)

    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    model_name = llm_name or str(frozen_pack["llm_name"])
    generation_regime = str(frozen_pack["generation_regime"])
    dataset_name = str(frozen_pack.get("dataset") or "mmlu")
    pack_metadata = {
        "candidate_pack_hash": hashlib.sha256(
            json.dumps(frozen_pack, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "candidate_build_mode": frozen_pack.get("candidate_build_mode", "dense_prefill_only"),
        "execution_side_reuse": bool(frozen_pack.get("execution_side_reuse")),
    }
    cases = [frozen_record_from_dict(case) for case in (frozen_pack.get("cases") or [])]

    dense_spec = METHOD_SPECS["dense"]
    dense_records = await _run_method_results(
        spec=dense_spec,
        cases=cases,
        model_name=model_name,
        output_root=output_root,
        judge_shuffle=judge_shuffle,
        shuffle_seed=shuffle_seed,
    )
    dense_rows = [
        case_evaluation_row_to_dict(
            build_case_evaluation_row(
                record=runtime_record,
                method_result=method_result,
                dense_reference_result=None,
                judge_shuffle=judge_shuffle,
            ),
            pack_metadata=pack_metadata,
        )
        for runtime_record, method_result in dense_records
    ]
    dense_summary = method_summary(
        spec=dense_spec,
        rows=dense_rows,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
    )
    dump_method_outputs(
        output_root=output_root,
        spec=dense_spec,
        rows=dense_rows,
        summary=dense_summary,
    )

    results: Dict[str, Any] = {
        "dense": {
            "rows": dense_rows,
            "summary": dense_summary,
        }
    }
    dense_results_by_index = {
        runtime_record.record_index: method_result
        for runtime_record, method_result in dense_records
    }

    method_iter = tqdm(
        requested_methods,
        desc=f"methods:{generation_regime}:{'shuffle' if judge_shuffle else 'noshuffle'}",
        unit="method",
    )
    for method_name in method_iter:
        if method_name == "dense":
            continue
        spec = METHOD_SPECS[method_name]
        method_records = await _run_method_results(
            spec=spec,
            cases=cases,
            model_name=model_name,
            output_root=output_root,
            judge_shuffle=judge_shuffle,
            shuffle_seed=shuffle_seed,
        )
        rows = [
            case_evaluation_row_to_dict(
                build_case_evaluation_row(
                    record=runtime_record,
                    method_result=method_result,
                    dense_reference_result=dense_results_by_index.get(runtime_record.record_index),
                    judge_shuffle=judge_shuffle,
                ),
                pack_metadata=pack_metadata,
            )
            for runtime_record, method_result in method_records
        ]
        summary = method_summary(
            spec=spec,
            rows=rows,
            generation_regime=generation_regime,
            judge_shuffle=judge_shuffle,
        )
        results[method_name] = {"rows": rows, "summary": summary}
        dump_method_outputs(
            output_root=output_root,
            spec=spec,
            rows=rows,
            summary=summary,
        )

    ordered_method_names = ["dense"] + [method for method in requested_methods if method != "dense"]
    summary_rows = [results[name]["summary"] for name in ordered_method_names if name in results]
    return write_benchmark_outputs(
        output_root=output_root,
        benchmark_name=f"{dataset_name}_paper_repair",
        model_name=model_name,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
        shuffle_seed=shuffle_seed,
        frozen_pack=frozen_pack,
        pack_metadata=pack_metadata,
        summary_rows=summary_rows,
    )


async def run_full_153_benchmark(
    *,
    llm_name: str,
    output_root: str | Path,
    index_file: str | None = None,
    methods: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    shuffles: Sequence[bool] | None = None,
    split: str = "val",
    agent_role: str = "MMLU Solver",
    agent_temperature: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    question_indices = load_question_indices(
        index_file=index_file or str(DEFAULT_INDEX_FILE),
    )
    selected_regimes = list(regimes or ["parallel_exploration", "progressive_refinement"])
    selected_shuffles = list(shuffles or [False, True])
    combined_rows: List[Dict[str, Any]] = []
    bundles: List[Dict[str, Any]] = []

    regime_iter = tqdm(selected_regimes, desc="full_benchmark:regimes", unit="regime")
    for regime in regime_iter:
        pack_path = output_root / f"frozen_candidates_{regime}.json"
        if not pack_path.exists():
            await build_frozen_candidate_pack(
                llm_name=llm_name,
                generation_regime=regime,
                question_indices=question_indices,
                split=split,
                agent_role=agent_role,
                agent_temperature=agent_temperature,
                seed=seed,
                output_path=pack_path,
            )
        frozen_pack = load_frozen_candidate_pack(pack_path)
        shuffle_iter = tqdm(
            selected_shuffles,
            desc=f"full_benchmark:{regime}:shuffle",
            unit="setting",
            leave=False,
        )
        for judge_shuffle in shuffle_iter:
            run_dir = output_root / regime / ("shuffle" if judge_shuffle else "noshuffle")
            summary = await run_judge_benchmark(
                frozen_pack=frozen_pack,
                output_dir=run_dir,
                methods=methods or DEFAULT_METHODS,
                llm_name=llm_name,
                judge_shuffle=judge_shuffle,
                shuffle_seed=seed,
            )
            bundles.append(
                {
                    "regime": regime,
                    "judge_shuffle": bool(judge_shuffle),
                    "pack_path": str(pack_path),
                    "summary_path": str(run_dir / "benchmark_summary.json"),
                    "table_path": str(run_dir / "main_table.md"),
                }
            )
            combined_rows.extend(summary["methods"])

    combined_table = "# MMLU Paper Repair 153 Main Table\n\n" + markdown_table(MAIN_TABLE_COLUMNS, combined_rows)
    (output_root / "main_table.md").write_text(combined_table.rstrip() + "\n", encoding="utf-8")
    summary_payload = {
        "benchmark_name": "mmlu_paper_repair_153",
        "llm_name": llm_name,
        "index_file": str(Path(index_file or DEFAULT_INDEX_FILE).expanduser()),
        "question_indices": question_indices,
        "methods": list(methods or DEFAULT_METHODS),
        "regimes": selected_regimes,
        "shuffles": [bool(item) for item in selected_shuffles],
        "bundles": bundles,
        "rows": combined_rows,
        "pal_kv_available": False,
        "jcr_definition": "strict_paper_candidate_selection_agreement",
    }
    json_dump(output_root / "benchmark_summary.json", summary_payload)
    return summary_payload
