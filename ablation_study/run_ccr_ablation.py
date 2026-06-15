from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KVCOMM.utils.log import configure_logging  # noqa: E402
from paper_repair.eval.details import build_case_evaluation_row, case_evaluation_row_to_dict  # noqa: E402
from paper_repair.eval.summary import (  # noqa: E402
    dump_method_outputs,
    json_dump,
    markdown_table,
    method_summary,
    write_benchmark_outputs,
)
from paper_repair.protocol.frozen_pack import frozen_record_from_dict, load_frozen_candidate_pack  # noqa: E402
from paper_repair.protocol.method_specs import METHOD_SPECS  # noqa: E402
from paper_repair.protocol.permutation import candidate_permutation  # noqa: E402
from paper_repair.protocol.runner import _record_with_permutation, _run_method_results  # noqa: E402
from paper_repair.repair.ccr_types import FrozenCandidateRecord, JudgeMethodResult  # noqa: E402

from ablation_study.ccr_ablation_variants import (  # noqa: E402
    AblationVariant,
    create_ccr_ablation_judge,
    random_shortlist_variant,
    run_ccr_ablation_case,
    variants_for_layer,
)


DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "paper_repair_formal_llama32_3b_20260421_164715"
DEFAULT_LLM_NAME = "/pychen/Test/model/Llama-3.2-3B-Instruct"
SETTING_TYPE = "mainline_controlled"
PROTOCOL_TYPE = "fixed_candidate_judge_only"


@dataclass(frozen=True)
class Testbed:
    layer: str
    name: str
    dataset: str
    regime: str
    ordering: str
    description: str

    @property
    def judge_shuffle(self) -> bool:
        return self.ordering == "shuffle"


DEFAULT_TESTBEDS = (
    Testbed(
        layer="main",
        name="layer1_mmlu_progressive_shuffle",
        dataset="mmlu",
        regime="progressive_refinement",
        ordering="shuffle",
        description="Layer 1 main component ablation testbed.",
    ),
    Testbed(
        layer="stability",
        name="layer2_mmlu_parallel_shuffle",
        dataset="mmlu",
        regime="parallel_exploration",
        ordering="shuffle",
        description="Layer 2 stability check with the reduced ablation matrix.",
    ),
)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pack_hash(pack: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(pack, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _candidate_pack_path(result_root: Path, testbed: Testbed) -> Path:
    return result_root / "candidate_cache" / testbed.dataset / f"{testbed.dataset}_{testbed.regime}.json"


def _parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw or "").replace(",", " ").split():
        if not part:
            continue
        seeds.append(int(part))
    return seeds or [42]


VARIANT_ALIASES = {
    "full": "ccr_full",
    "ccr_full": "ccr_full",
    "no_grouping": "ccr_wo_comparative_grouping",
    "no_comparative_grouping": "ccr_wo_comparative_grouping",
    "ccr_wo_comparative_grouping": "ccr_wo_comparative_grouping",
    "no_support": "ccr_wo_support_scoring",
    "no_support_scoring": "ccr_wo_support_scoring",
    "ccr_wo_support_scoring": "ccr_wo_support_scoring",
    "no_shortlist": "ccr_wo_shortlist_compression",
    "no_shortlist_compression": "ccr_wo_shortlist_compression",
    "ccr_wo_shortlist_compression": "ccr_wo_shortlist_compression",
    "no_payload_stats": "ccr_wo_comparative_payload_statistics",
    "no_payload_statistics": "ccr_wo_comparative_payload_statistics",
    "ccr_wo_comparative_payload_statistics": "ccr_wo_comparative_payload_statistics",
    "stats_only": "ccr_stats_only",
    "ccr_stats_only": "ccr_stats_only",
    "shortlist_only": "ccr_shortlist_only",
    "ccr_shortlist_only": "ccr_shortlist_only",
    "random_shortlist": "random_shortlist",
    "random": "random_shortlist",
    "random42": "ccr_random_shortlist_seed42",
    "random43": "ccr_random_shortlist_seed43",
    "random44": "ccr_random_shortlist_seed44",
    "ccr_random_shortlist_seed42": "ccr_random_shortlist_seed42",
    "ccr_random_shortlist_seed43": "ccr_random_shortlist_seed43",
    "ccr_random_shortlist_seed44": "ccr_random_shortlist_seed44",
}


def _filter_variants(
    variants: Sequence[AblationVariant],
    *,
    requested: Sequence[str] | None,
    random_seeds: Sequence[int],
) -> list[AblationVariant]:
    if not requested:
        return list(variants)

    available = {variant.slug: variant for variant in variants}
    for seed in random_seeds:
        variant = random_shortlist_variant(seed)
        available.setdefault(variant.slug, variant)

    result: list[AblationVariant] = []
    for raw in requested:
        canonical = VARIANT_ALIASES.get(str(raw).strip(), str(raw).strip())
        if canonical == "random_shortlist":
            for seed in random_seeds:
                result.append(available[f"ccr_random_shortlist_seed{int(seed)}"])
            continue
        if canonical not in available:
            raise SystemExit(f"Unsupported --variant {raw!r}; canonical={canonical!r}")
        result.append(available[canonical])

    deduped: list[AblationVariant] = []
    seen = set()
    for variant in result:
        if variant.slug in seen:
            continue
        seen.add(variant.slug)
        deduped.append(variant)
    return deduped


def _selected_testbeds(args: argparse.Namespace) -> list[Testbed]:
    if args.testbed:
        parsed: list[Testbed] = []
        for index, raw in enumerate(args.testbed):
            parts = raw.split(":")
            if len(parts) not in {3, 4}:
                raise SystemExit(
                    "--testbed must use dataset:regime:ordering or dataset:regime:ordering:layer"
                )
            dataset, regime, ordering = parts[:3]
            layer = parts[3] if len(parts) == 4 else "main"
            parsed.append(
                Testbed(
                    layer=layer,
                    name=f"custom_{index}_{dataset}_{regime}_{ordering}",
                    dataset=dataset,
                    regime=regime,
                    ordering=ordering,
                    description="User-specified CCR ablation testbed.",
                )
            )
        return parsed

    if args.layers == "main":
        return [DEFAULT_TESTBEDS[0]]
    if args.layers == "stability":
        return [DEFAULT_TESTBEDS[1]]
    return list(DEFAULT_TESTBEDS)


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


def _subset_pack(pack: Dict[str, Any], limit_cases: int | None) -> Dict[str, Any]:
    if limit_cases is None or int(limit_cases) <= 0:
        return pack
    limit = int(limit_cases)
    subset = dict(pack)
    cases = list(pack.get("cases") or [])[:limit]
    subset["cases"] = cases
    question_indices = pack.get("question_indices")
    if isinstance(question_indices, list):
        subset["question_indices"] = question_indices[:limit]
    subset["ablation_limit_cases"] = limit
    return subset


def _pack_metadata(pack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_pack_hash": _pack_hash(pack),
        "candidate_build_mode": pack.get("candidate_build_mode", "dense_prefill_only"),
        "execution_side_reuse": bool(pack.get("execution_side_reuse")),
    }


async def _run_dense(
    *,
    cases: Sequence[FrozenCandidateRecord],
    model_name: str,
    output_root: Path,
    generation_regime: str,
    judge_shuffle: bool,
    shuffle_seed: int,
    pack_metadata: Dict[str, Any],
) -> tuple[list[tuple[FrozenCandidateRecord, JudgeMethodResult]], Dict[str, Any]]:
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
    return dense_records, dense_summary


async def _run_variant(
    *,
    variant: AblationVariant,
    cases: Sequence[FrozenCandidateRecord],
    dense_results_by_index: Dict[int, JudgeMethodResult],
    model_name: str,
    output_root: Path,
    generation_regime: str,
    judge_shuffle: bool,
    shuffle_seed: int,
    pack_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    spec = _variant_spec(variant)
    variant_records = await _run_method_results(
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
        for runtime_record, method_result in variant_records
    ]
    summary = method_summary(
        spec=spec,
        rows=rows,
        generation_regime=generation_regime,
        judge_shuffle=judge_shuffle,
    )
    dump_method_outputs(
        output_root=output_root,
        spec=spec,
        rows=rows,
        summary=summary,
    )
    return summary


async def run_testbed(
    *,
    testbed: Testbed,
    variants: Sequence[AblationVariant],
    result_root: Path,
    output_root: Path,
    llm_name: str,
    shuffle_seed: int,
    limit_cases: int | None,
) -> Dict[str, Any]:
    pack_path = _candidate_pack_path(result_root, testbed)
    if not pack_path.exists():
        raise FileNotFoundError(f"Candidate pack not found: {pack_path}")
    frozen_pack = _subset_pack(load_frozen_candidate_pack(pack_path), limit_cases)
    pack_metadata = _pack_metadata(frozen_pack)
    cases = [frozen_record_from_dict(case) for case in (frozen_pack.get("cases") or [])]
    if not cases:
        raise ValueError(f"No cases found in candidate pack: {pack_path}")

    block_dir = output_root / testbed.layer / testbed.dataset / testbed.regime / testbed.ordering
    block_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=block_dir / "logs" / "evaluate_ccr_ablation.log")

    block_config = {
        "setting_type": SETTING_TYPE,
        "protocol_type": PROTOCOL_TYPE,
        "testbed": asdict(testbed),
        "candidate_pack_path": str(pack_path),
        "candidate_pack_hash": pack_metadata["candidate_pack_hash"],
        "llm_name": llm_name,
        "shuffle_seed": int(shuffle_seed),
        "limit_cases": limit_cases,
        "case_count": len(cases),
        "candidate_count": int(frozen_pack.get("candidate_count") or cases[0].candidate_count),
        "variants": [asdict(variant) for variant in variants],
    }
    _json_write(block_dir / "ccr_ablation_config.json", block_config)

    dense_records, dense_summary = await _run_dense(
        cases=cases,
        model_name=llm_name,
        output_root=block_dir,
        generation_regime=testbed.regime,
        judge_shuffle=testbed.judge_shuffle,
        shuffle_seed=shuffle_seed,
        pack_metadata=pack_metadata,
    )
    dense_results_by_index = {
        runtime_record.record_index: method_result
        for runtime_record, method_result in dense_records
    }

    summary_rows = [dense_summary]
    blockers: list[dict[str, Any]] = []
    for variant in variants:
        try:
            summary = await _run_variant(
                variant=variant,
                cases=cases,
                dense_results_by_index=dense_results_by_index,
                model_name=llm_name,
                output_root=block_dir,
                generation_regime=testbed.regime,
                judge_shuffle=testbed.judge_shuffle,
                shuffle_seed=shuffle_seed,
                pack_metadata=pack_metadata,
            )
            summary_rows.append(summary)
        except Exception as exc:
            blockers.append(
                {
                    "testbed": testbed.name,
                    "variant": variant.slug,
                    "label": variant.label,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    benchmark_summary = write_benchmark_outputs(
        output_root=block_dir,
        benchmark_name=f"{testbed.dataset}_paper_repair_ccr_ablation",
        model_name=llm_name,
        generation_regime=testbed.regime,
        judge_shuffle=testbed.judge_shuffle,
        shuffle_seed=shuffle_seed,
        frozen_pack=frozen_pack,
        pack_metadata=pack_metadata,
        summary_rows=summary_rows,
    )
    _json_write(block_dir / "ccr_ablation_blockers.json", blockers)
    return {
        "block_dir": str(block_dir),
        "testbed": asdict(testbed),
        "candidate_pack_path": str(pack_path),
        "candidate_pack_hash": pack_metadata["candidate_pack_hash"],
        "case_count": len(cases),
        "candidate_count": int(frozen_pack.get("candidate_count") or cases[0].candidate_count),
        "variants": [asdict(variant) for variant in variants],
        "blockers": blockers,
        "benchmark_summary_path": str(block_dir / "benchmark_summary.json"),
        "benchmark_summary": benchmark_summary,
    }


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _mean(values: Iterable[Any]) -> float | None:
    nums = []
    for value in values:
        if value is None:
            continue
        try:
            nums.append(float(value))
        except Exception:
            continue
    return (sum(nums) / len(nums)) if nums else None


def _rate(flags: Iterable[bool | None]) -> float | None:
    usable = [flag for flag in flags if flag is not None]
    if not usable:
        return None
    return sum(1 for flag in usable if flag) / len(usable)


def _load_rows(block_dir: Path, method: str) -> list[dict[str, Any]]:
    path = block_dir / f"{method}_details.json"
    if not path.exists():
        return []
    payload = _json_load(path)
    return payload if isinstance(payload, list) else []


def _load_summary(block_dir: Path, method: str) -> dict[str, Any] | None:
    path = block_dir / f"{method}_summary.json"
    if not path.exists():
        return None
    payload = _json_load(path)
    return payload if isinstance(payload, dict) else None


def _shortlist_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("ccr_shortlist_candidate_ids")
    if ids is None:
        payload = row.get("ccr_shortlist_payload") or {}
        if isinstance(payload, dict):
            ids = payload.get("shortlisted_candidate_ids")
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids]


def _extra_metrics(block_dir: Path, method: str, full_rows_by_record: Dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = _load_rows(block_dir, method)
    dense_in_shortlist = []
    full_agreement = []
    for row in rows:
        shortlist = set(_shortlist_ids(row))
        dense_id = row.get("dense_selected_agent_id")
        dense_in_shortlist.append(None if dense_id is None or not shortlist else str(dense_id) in shortlist)
        full_row = full_rows_by_record.get(int(row.get("record_index", -1)))
        if full_row is not None:
            left = row.get("selected_agent_id")
            right = full_row.get("selected_agent_id")
            full_agreement.append(None if left is None or right is None else str(left) == str(right))

    return {
        "dense_winner_in_shortlist_rate": _rate(dense_in_shortlist),
        "shortlist_answer_group_coverage": _mean(
            row.get("ccr_ablation_shortlist_answer_group_coverage") for row in rows
        ),
        "avg_judge_input_length": _mean(
            (
                row.get("ccr_ablation_judge_context_tokens")
                if row.get("ccr_ablation_judge_context_tokens") is not None
                else row.get("ccr_ablation_judge_context_chars")
            )
            for row in rows
        ),
        "avg_payload_length": _mean(
            (
                row.get("ccr_ablation_context_payload_tokens")
                if row.get("ccr_ablation_context_payload_tokens") is not None
                else row.get("ccr_ablation_context_payload_chars")
            )
            for row in rows
        ),
        "selected_agent_agreement_with_full": (
            1.0 if method == "ccr_full" and rows else _rate(full_agreement)
        ),
    }


def _collect_table_rows(block_dir: Path, variants: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    full_rows = _load_rows(block_dir, "ccr_full")
    full_rows_by_record = {
        int(row["record_index"]): row
        for row in full_rows
        if isinstance(row, dict) and row.get("record_index") is not None
    }
    rows = []
    for variant in variants:
        method = str(variant["slug"])
        summary = _load_summary(block_dir, method)
        if summary is None:
            rows.append(
                {
                    "Ablation Variant": variant.get("label", method),
                    "Acc": "MISSING",
                    "JCR": "MISSING",
                    "Reuse": "MISSING",
                    "Dense winner in shortlist rate": "MISSING",
                    "Shortlist answer-group coverage": "MISSING",
                    "Avg judge input length": "MISSING",
                    "Selection gap": "MISSING",
                    "Agreement with Full": "MISSING",
                }
            )
            continue
        extras = _extra_metrics(block_dir, method, full_rows_by_record)
        rows.append(
            {
                "Ablation Variant": variant.get("label", method),
                "Acc": _format_float(summary.get("Acc")),
                "JCR": _format_float(summary.get("JCR")),
                "Reuse": _format_float(summary.get("reuse")),
                "Dense winner in shortlist rate": _format_float(extras.get("dense_winner_in_shortlist_rate")),
                "Shortlist answer-group coverage": _format_float(extras.get("shortlist_answer_group_coverage")),
                "Avg judge input length": _format_float(extras.get("avg_judge_input_length")),
                "Selection gap": _format_float(summary.get("selection_gap")),
                "Agreement with Full": _format_float(extras.get("selected_agent_agreement_with_full")),
            }
        )
    return rows


def _delta(block_dir: Path, variant_slug: str, metric: str) -> float | None:
    full = _load_summary(block_dir, "ccr_full")
    variant = _load_summary(block_dir, variant_slug)
    if full is None or variant is None:
        return None
    try:
        return float(variant.get(metric)) - float(full.get(metric))
    except Exception:
        return None


def _analysis_sentence(
    *,
    name: str,
    block_dir: Path,
    variant_slug: str,
    component: str,
) -> str:
    acc_delta = _delta(block_dir, variant_slug, "Acc")
    jcr_delta = _delta(block_dir, variant_slug, "JCR")
    reuse_delta = _delta(block_dir, variant_slug, "reuse")
    if acc_delta is None or jcr_delta is None:
        return f"- {name}: result missing, so this component cannot be interpreted from this run."
    return (
        f"- {name}: removing {component} changes Acc by `{acc_delta:+.4f}` and JCR by "
        f"`{jcr_delta:+.4f}` while Reuse changes by `{(reuse_delta or 0.0):+.4f}`. "
        "This isolates the component as a judge-side decision factor rather than a compute-reuse factor."
    )


def generate_reports(
    *,
    output_root: Path,
    reports_dir: Path,
) -> None:
    configs = sorted(output_root.glob("*/*/*/*/ccr_ablation_config.json"))
    manifest_rows = []
    blockers = []
    table_sections = []
    analysis_sections = []

    for config_path in configs:
        block_dir = config_path.parent
        config = _json_load(config_path)
        testbed = config["testbed"]
        variants = list(config.get("variants") or [])
        blocker_path = block_dir / "ccr_ablation_blockers.json"
        block_blockers = _json_load(blocker_path) if blocker_path.exists() else []
        blockers.extend(block_blockers)

        manifest_rows.append(
            {
                "Layer": testbed.get("layer"),
                "Dataset": testbed.get("dataset"),
                "Regime": testbed.get("regime"),
                "Ordering": testbed.get("ordering"),
                "Cases": config.get("case_count"),
                "Branches": config.get("candidate_count"),
                "Variants": len(variants),
                "Result root": str(block_dir),
            }
        )
        table_rows = _collect_table_rows(block_dir, variants)
        table_sections.extend(
            [
                f"## {testbed.get('layer')} / {testbed.get('dataset')} / {testbed.get('regime')} / {testbed.get('ordering')}",
                "",
                markdown_table(
                    [
                        ("Ablation Variant", "Ablation Variant"),
                        ("Acc", "Acc"),
                        ("JCR", "JCR"),
                        ("Reuse", "Reuse"),
                        ("Dense winner in shortlist rate", "Dense winner in shortlist rate"),
                        ("Shortlist answer-group coverage", "Shortlist answer-group coverage"),
                        ("Avg judge input length", "Avg judge input length"),
                        ("Selection gap", "Selection gap"),
                        ("Agreement with Full", "Agreement with Full"),
                    ],
                    table_rows,
                ),
                "",
            ]
        )

        analysis_sections.extend(
            [
                f"## {testbed.get('layer')} / {testbed.get('dataset')} / {testbed.get('regime')} / {testbed.get('ordering')}",
                "",
                _analysis_sentence(
                    name="Comparative grouping",
                    block_dir=block_dir,
                    variant_slug="ccr_wo_comparative_grouping",
                    component="comparative grouping",
                ),
                _analysis_sentence(
                    name="Support scoring",
                    block_dir=block_dir,
                    variant_slug="ccr_wo_support_scoring",
                    component="support-based deterministic ranking",
                ),
                _analysis_sentence(
                    name="Shortlist compression",
                    block_dir=block_dir,
                    variant_slug="ccr_wo_shortlist_compression",
                    component="shortlist compression",
                ),
            ]
        )
        random_slugs = [str(variant["slug"]) for variant in variants if variant.get("family") == "random_shortlist"]
        if random_slugs:
            random_lines = [
                _analysis_sentence(
                    name=f"Random shortlist {slug}",
                    block_dir=block_dir,
                    variant_slug=slug,
                    component="deterministic shortlist selection",
                )
                for slug in random_slugs
            ]
            analysis_sections.extend(random_lines)
        if any(variant.get("family") == "no_payload_statistics" for variant in variants):
            analysis_sections.append(
                _analysis_sentence(
                    name="Comparative payload statistics",
                    block_dir=block_dir,
                    variant_slug="ccr_wo_comparative_payload_statistics",
                    component="serialized comparative payload statistics",
                )
            )
        else:
            analysis_sections.append(
                "- Comparative payload statistics: optional variant was not run in this matrix, so payload-vs-shortlist attribution remains pending."
            )
        analysis_sections.append("")

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_md = "\n".join(
        [
            "# CCR-Judge Ablation Run Manifest",
            "",
            f"Setting type: `{SETTING_TYPE}`",
            f"Protocol type: `{PROTOCOL_TYPE}`",
            "",
            markdown_table(
                [
                    ("Layer", "Layer"),
                    ("Dataset", "Dataset"),
                    ("Regime", "Regime"),
                    ("Ordering", "Ordering"),
                    ("Cases", "Cases"),
                    ("Branches", "Branches"),
                    ("Variants", "Variants"),
                    ("Result root", "Result root"),
                ],
                manifest_rows,
            ),
            "",
        ]
    )
    tables_md = "\n".join(["# CCR-Judge Ablation Tables", "", *table_sections])
    analysis_md = "\n".join(
        [
            "# CCR-Judge Ablation Analysis",
            "",
            "This report interprets component removals under the paper_repair fixed-candidate judge-only benchmark.",
            "",
            *analysis_sections,
            "## Core Questions",
            "",
            "1. CCR-Judge's core gain should be attributed to the components whose removal most reduces Acc/JCR while leaving Reuse approximately fixed. Use the deltas above to distinguish comparative grouping, support scoring, shortlist compression, and optional payload statistics.",
            "2. The gain surface is identified by the metrics that move: Acc measures answer selection quality, JCR measures decision stability relative to dense, shortlist coverage measures compression quality, and agreement with Full measures behavioral drift.",
            "",
        ]
    )
    blockers_md = "\n".join(
        [
            "# CCR-Judge Ablation Blockers",
            "",
            (
                "No blockers were recorded."
                if not blockers
                else markdown_table(
                    [
                        ("testbed", "Testbed"),
                        ("variant", "Variant"),
                        ("error_type", "Error Type"),
                        ("error_message", "Error Message"),
                    ],
                    blockers,
                )
            ),
            "",
        ]
    )

    for root in {reports_dir, output_root}:
        (root / "ccr_ablation_run_manifest.md").write_text(manifest_md, encoding="utf-8")
        (root / "ccr_ablation_tables.md").write_text(tables_md, encoding="utf-8")
        (root / "ccr_ablation_analysis.md").write_text(analysis_md, encoding="utf-8")
        (root / "ccr_ablation_blockers.md").write_text(blockers_md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CCR-Judge component ablations under paper_repair.")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default=str(ABLATION_ROOT))
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--random-shortlist-seeds", default="42")
    parser.add_argument(
        "--variant",
        action="append",
        help=(
            "Run only selected variant aliases. Examples: full, no_support, "
            "no_shortlist_compression, no_payload_stats, stats_only, shortlist_only, random_shortlist."
        ),
    )
    parser.add_argument("--layers", choices=["main", "stability", "all"], default="all")
    parser.add_argument(
        "--testbed",
        action="append",
        help="Custom testbed as dataset:regime:ordering or dataset:regime:ordering:layer.",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--include-optional-payload-statistics-ablation", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else ABLATION_ROOT / "results" / f"ccr_ablation_{timestamp}"
    )
    reports_dir = Path(args.reports_dir).expanduser()
    result_root = Path(args.result_root).expanduser()
    seeds = _parse_seed_list(args.random_shortlist_seeds)
    testbeds = _selected_testbeds(args)

    run_plan = {
        "setting_type": SETTING_TYPE,
        "protocol_type": PROTOCOL_TYPE,
        "result_root": str(result_root),
        "output_root": str(output_root),
        "reports_dir": str(reports_dir),
        "llm_name": args.llm_name,
        "shuffle_seed": int(args.shuffle_seed),
        "random_shortlist_seeds": seeds,
        "limit_cases": args.limit_cases,
        "include_optional_payload_statistics_ablation": bool(
            args.include_optional_payload_statistics_ablation
        ),
        "testbeds": [asdict(testbed) for testbed in testbeds],
    }
    _json_write(output_root / "ccr_ablation_run_plan.json", run_plan)

    if args.dry_run:
        generate_reports(output_root=output_root, reports_dir=reports_dir)
        print(json.dumps(run_plan, ensure_ascii=False, indent=2))
        return

    if not args.report_only:
        run_records = []
        for testbed in testbeds:
            variants = variants_for_layer(
                layer=testbed.layer,
                random_seeds=seeds,
                include_optional=bool(args.include_optional_payload_statistics_ablation),
            )
            variants = _filter_variants(
                variants,
                requested=args.variant,
                random_seeds=seeds,
            )
            record = await run_testbed(
                testbed=testbed,
                variants=variants,
                result_root=result_root,
                output_root=output_root,
                llm_name=args.llm_name,
                shuffle_seed=int(args.shuffle_seed),
                limit_cases=args.limit_cases,
            )
            run_records.append(record)
        _json_write(output_root / "ccr_ablation_run_records.json", run_records)

    generate_reports(output_root=output_root, reports_dir=reports_dir)
    print(f"CCR ablation outputs: {output_root}")
    print(f"CCR ablation reports: {reports_dir}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
