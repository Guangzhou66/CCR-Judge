from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KVCOMM.utils.log import configure_logging  # noqa: E402
from paper_repair.eval.details import build_case_evaluation_row, case_evaluation_row_to_dict  # noqa: E402
from paper_repair.eval.summary import dump_method_outputs, method_summary, write_benchmark_outputs  # noqa: E402
from paper_repair.protocol.frozen_pack import frozen_record_from_dict, load_frozen_candidate_pack  # noqa: E402
from paper_repair.protocol.method_specs import METHOD_SPECS  # noqa: E402
from paper_repair.protocol.runner import _run_method_results  # noqa: E402
from paper_repair.repair.ccr_types import FrozenCandidateRecord, JudgeMethodResult  # noqa: E402

from ablation_study.ccr_ablation_variants import (  # noqa: E402
    AblationVariant,
    create_ccr_ablation_judge,
    random_shortlist_variant,
    run_ccr_ablation_case,
    variants_for_layer,
)


def _variant_by_slug(slug: str) -> AblationVariant:
    variants = variants_for_layer(layer="main", random_seeds=[42], include_optional=True)
    by_slug = {variant.slug: variant for variant in variants}
    if slug.startswith("ccr_random_shortlist_seed"):
        seed = int(slug.rsplit("seed", 1)[1])
        return random_shortlist_variant(seed)
    if slug not in by_slug:
        raise ValueError(f"Unsupported variant slug: {slug}")
    return by_slug[slug]


def _pack_hash(pack: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(pack, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _pack_metadata(pack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_pack_hash": _pack_hash(pack),
        "candidate_build_mode": pack.get("candidate_build_mode", "dense_prefill_only"),
        "execution_side_reuse": bool(pack.get("execution_side_reuse")),
    }


def _variant_spec(variant: AblationVariant):
    async def run_case(judge: Any, record: FrozenCandidateRecord, output_dir: Path) -> JudgeMethodResult:
        return await run_ccr_ablation_case(
            judge=judge,
            record=record,
            output_dir=output_dir,
            variant=variant,
        )

    def factory(model_name: str, domain: str):
        return create_ccr_ablation_judge(
            llm_name=model_name,
            domain=domain,
            variant=variant,
        )

    return SimpleNamespace(
        name=variant.slug,
        label=variant.label,
        reuse="judge_kv_reuse",
        mode="allow_kv_reuse",
        env={},
        judge_factory=factory,
        run_case=run_case,
    )


def _dense_result_from_row(row: Dict[str, Any]) -> JudgeMethodResult:
    return JudgeMethodResult(
        method_name="dense",
        selected_agent_id=(None if row.get("selected_agent_id") is None else str(row.get("selected_agent_id"))),
        selected_answer=row.get("selected_answer"),
        parse_success=bool(row.get("parse_success")),
        fallback_used=bool(row.get("fallback_used")),
        fallback_reason=row.get("fallback_reason"),
        raw_judge_text=str(row.get("raw_judge_text") or row.get("final_result_text") or ""),
        reuse_mode=row.get("reuse_mode"),
        ttft=row.get("ttft"),
        metadata={},
    )


def _load_dense_reference(path: Path) -> Dict[int, JudgeMethodResult]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["record_index"]): _dense_result_from_row(row)
        for row in rows
        if isinstance(row, dict) and row.get("record_index") is not None
    }


async def _run_dense(
    *,
    cases: Sequence[FrozenCandidateRecord],
    model_name: str,
    output_root: Path,
    judge_shuffle: bool,
    shuffle_seed: int,
    generation_regime: str,
    pack_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    spec = METHOD_SPECS["dense"]
    records = await _run_method_results(
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
                dense_reference_result=None,
                judge_shuffle=judge_shuffle,
            ),
            pack_metadata=pack_metadata,
        )
        for runtime_record, method_result in records
    ]
    summary = method_summary(
        spec=spec,
        rows=rows,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
    )
    dump_method_outputs(output_root=output_root, spec=spec, rows=rows, summary=summary)
    return summary


async def _run_variant(
    *,
    variant: AblationVariant,
    cases: Sequence[FrozenCandidateRecord],
    dense_reference: Dict[int, JudgeMethodResult],
    model_name: str,
    output_root: Path,
    judge_shuffle: bool,
    shuffle_seed: int,
    generation_regime: str,
    pack_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    spec = _variant_spec(variant)
    records = await _run_method_results(
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
                dense_reference_result=dense_reference.get(runtime_record.record_index),
                judge_shuffle=judge_shuffle,
            ),
            pack_metadata=pack_metadata,
        )
        for runtime_record, method_result in records
    ]
    summary = method_summary(
        spec=spec,
        rows=rows,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
    )
    dump_method_outputs(output_root=output_root, spec=spec, rows=rows, summary=summary)
    return summary


async def _run_baseline_method(
    *,
    method_name: str,
    cases: Sequence[FrozenCandidateRecord],
    dense_reference: Dict[int, JudgeMethodResult],
    model_name: str,
    output_root: Path,
    judge_shuffle: bool,
    shuffle_seed: int,
    generation_regime: str,
    pack_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    spec = METHOD_SPECS[method_name]
    records = await _run_method_results(
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
                dense_reference_result=dense_reference.get(runtime_record.record_index),
                judge_shuffle=judge_shuffle,
            ),
            pack_metadata=pack_metadata,
        )
        for runtime_record, method_result in records
    ]
    summary = method_summary(
        spec=spec,
        rows=rows,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
    )
    dump_method_outputs(output_root=output_root, spec=spec, rows=rows, summary=summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one support-variant check unit.")
    parser.add_argument("--frozen-pack", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dense-details", default=None)
    parser.add_argument("--judge-shuffle", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    os.environ.setdefault("KVCOMM_FORCE_BFLOAT16", "1")
    os.environ.setdefault("KVCOMM_PROGRESS_ONLY", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_root / "logs" / "support_variant_unit.log")

    frozen_pack = load_frozen_candidate_pack(args.frozen_pack)
    pack_metadata = _pack_metadata(frozen_pack)
    cases = [frozen_record_from_dict(case) for case in (frozen_pack.get("cases") or [])]
    generation_regime = str(frozen_pack.get("generation_regime") or "")
    if args.variant == "dense":
        summary = await _run_dense(
            cases=cases,
            model_name=args.llm_name,
            output_root=output_root,
            judge_shuffle=bool(args.judge_shuffle),
            shuffle_seed=int(args.shuffle_seed),
            generation_regime=generation_regime,
            pack_metadata=pack_metadata,
        )
        summary_rows = [summary]
    else:
        if not args.dense_details:
            raise ValueError("--dense-details is required for non-dense variants")
        dense_reference = _load_dense_reference(Path(args.dense_details).expanduser())
        if args.variant in METHOD_SPECS:
            summary = await _run_baseline_method(
                method_name=args.variant,
                cases=cases,
                dense_reference=dense_reference,
                model_name=args.llm_name,
                output_root=output_root,
                judge_shuffle=bool(args.judge_shuffle),
                shuffle_seed=int(args.shuffle_seed),
                generation_regime=generation_regime,
                pack_metadata=pack_metadata,
            )
        else:
            variant = _variant_by_slug(args.variant)
            summary = await _run_variant(
                variant=variant,
                cases=cases,
                dense_reference=dense_reference,
                model_name=args.llm_name,
                output_root=output_root,
                judge_shuffle=bool(args.judge_shuffle),
                shuffle_seed=int(args.shuffle_seed),
                generation_regime=generation_regime,
                pack_metadata=pack_metadata,
            )
        summary_rows = [summary]

    dataset_name = str(frozen_pack.get("dataset") or "paper_repair")
    write_benchmark_outputs(
        output_root=output_root,
        benchmark_name=f"{dataset_name}_paper_repair_support_variant_unit",
        model_name=args.llm_name,
        generation_regime=generation_regime,
        judge_shuffle=bool(args.judge_shuffle),
        shuffle_seed=int(args.shuffle_seed),
        frozen_pack=frozen_pack,
        pack_metadata=pack_metadata,
        summary_rows=summary_rows,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
