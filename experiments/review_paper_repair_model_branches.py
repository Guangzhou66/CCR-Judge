from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare larger-smoke model branches and emit updated readiness reports.")
    parser.add_argument("--qwen-root", required=True)
    parser.add_argument("--llama-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def markdown_table(columns: Sequence[tuple[str, str]], rows: Iterable[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    seen = False
    for row in rows:
        seen = True
        rendered = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                rendered.append("-")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    if not seen:
        lines.append("| " + " | ".join("-" for _ in columns) + " |")
    return "\n".join(lines)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_summary(result_root: Path) -> Dict[str, Any]:
    return json.loads((result_root / "reports" / "branch_validation_summary.json").read_text(encoding="utf-8"))


def branch_component(summary: Dict[str, Any], component: str) -> str:
    dataset_values = [result["status"][component] for result in summary["dataset_results"].values()]
    return "pass" if all(dataset_values) and summary["manifest_ok"] else "fail"


def main() -> None:
    args = parse_args()
    qwen_root = Path(args.qwen_root).expanduser()
    llama_root = Path(args.llama_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen = load_summary(qwen_root)
    llama = load_summary(llama_root)

    qwen_pass = bool(qwen["overall_pass"])
    llama_pass = bool(llama["overall_pass"])

    if not llama_pass:
        verdict = "not ready"
        recommended_branch = "none"
        why = "paper_default_model branch larger smoke did not pass."
    elif llama_pass:
        verdict = "ready for formal"
        recommended_branch = f"{llama['model_branch']} | {llama['model_path']}"
        why = "paper_default_model branch passed larger smoke across all three datasets with no hard protocol blocker."
    else:
        verdict = "not ready"
        recommended_branch = "none"
        why = "larger-smoke review did not produce a formal-ready paper_default_model branch."

    component_rows = []
    components = [
        ("fixed candidate sharing", "fixed_candidate_sharing"),
        ("execution-side reuse disabled", "execution_side_reuse_disabled"),
        ("official scoring boundary", "official_scoring_boundary"),
        ("shuffle permutation consistency", "shuffle_identical_permutations"),
        ("original-id JCR", "jcr_original_ids"),
        ("reuse accounting", "reuse_definition_correct"),
        ("PAL-KV baseline separation", "pal_kv_baseline_separation"),
        ("ori_protocol isolation", "ori_protocol_isolation"),
    ]
    for label, key in components:
        component_rows.append(
            {
                "component": label,
                "qwen": branch_component(qwen, key),
                "llama": branch_component(llama, key),
                "notes": "all datasets pass" if branch_component(qwen, key) == "pass" and branch_component(llama, key) == "pass" else "see branch reports",
            }
        )
    component_rows.extend(
        [
            {
                "component": "larger smoke overall",
                "qwen": "pass" if qwen_pass else "fail",
                "llama": "pass" if llama_pass else "fail",
                "notes": "branch-level larger-smoke verdict",
            },
            {
                "component": "formal candidate status",
                "qwen": "local alt evidence",
                "llama": "paper default candidate" if llama_pass else "not ready",
                "notes": "formal should not be based on the Qwen branch",
            },
        ]
    )

    comparison_columns = [
        ("component", "Component"),
        ("qwen", "Qwen Branch"),
        ("llama", "Llama Branch"),
        ("notes", "Notes"),
    ]
    comparison_text = [
        "# Paper Repair Model Branch Comparison",
        "",
        f"Qwen result root: `{qwen['result_root']}`",
        f"Llama result root: `{llama['result_root']}`",
        "",
        f"Qwen model_path: `{qwen['model_path']}`",
        f"Qwen model_branch: `{qwen['model_branch']}`",
        "",
        f"Llama model_path: `{llama['model_path']}`",
        f"Llama model_branch: `{llama['model_branch']}`",
        "",
        markdown_table(comparison_columns, component_rows),
        "",
        "## Branch Summary",
        "",
        f"- Qwen line larger-smoke passed: `{'yes' if qwen_pass else 'no'}`",
        f"- Llama line larger-smoke passed: `{'yes' if llama_pass else 'no'}`",
        "- Qwen should remain an independently retained `local_alt_model` evidence line.",
        "- Llama should be treated as the `paper_default_model` formal candidate line.",
    ]
    (output_dir / "paper_repair_model_branch_comparison.md").write_text(
        "\n".join(comparison_text).rstrip() + "\n",
        encoding="utf-8",
    )

    readiness_text = [
        "# Paper Repair Stage B Readiness After Larger Smoke",
        "",
        "## Executive Summary",
        "",
        f"- qwen model_path: `{qwen['model_path']}`",
        f"- qwen model_branch: `{qwen['model_branch']}`",
        f"- qwen result_root: `{qwen['result_root']}`",
        f"- qwen larger-smoke passed: `{'yes' if qwen_pass else 'no'}`",
        "",
        f"- llama model_path: `{llama['model_path']}`",
        f"- llama model_branch: `{llama['model_branch']}`",
        f"- llama result_root: `{llama['result_root']}`",
        f"- llama larger-smoke passed: `{'yes' if llama_pass else 'no'}`",
        "",
        "## Updated Verdict",
        "",
        f"- verdict: `{verdict}`",
        f"- recommended formal branch: `{recommended_branch}`",
        f"- reason: {why}",
        "",
        "## Interpretation",
        "",
        "- Qwen larger smoke is an additional local-alt evidence line and must remain separate from formal readiness for the paper-default branch.",
        "- Llama larger smoke is the decisive paper-default readiness line.",
    ]
    if qwen.get("confidence_risks"):
        readiness_text.extend(["", "### Qwen Confidence Notes", ""])
        for item in qwen["confidence_risks"]:
            readiness_text.append(f"- {item}")
    if llama.get("confidence_risks"):
        readiness_text.extend(["", "### Llama Confidence Notes", ""])
        for item in llama["confidence_risks"]:
            readiness_text.append(f"- {item}")
    (output_dir / "paper_repair_stageB_readiness_after_larger_smoke.md").write_text(
        "\n".join(readiness_text).rstrip() + "\n",
        encoding="utf-8",
    )

    go_nogo_lines = [
        "# Paper Repair Formal Go / No-Go",
        "",
        f"- qwen larger-smoke passed: `{'yes' if qwen_pass else 'no'}`",
        f"- llama larger-smoke passed: `{'yes' if llama_pass else 'no'}`",
        f"- final verdict: `{verdict}`",
        "",
    ]
    if verdict == "ready for formal":
        go_nogo_lines.extend(
            [
                "## Go",
                "",
                f"Formal is ready to launch on the Llama paper-default branch.",
                "",
                f"Recommended formal model_path: `{llama['model_path']}`",
                f"Recommended model_branch: `{llama['model_branch']}`",
                "",
                "Do not use Qwen results as a substitute for paper-default formal evidence.",
            ]
        )
    else:
        go_nogo_lines.extend(
            [
                "## No-Go",
                "",
                why,
                "",
                "Do not start formal yet.",
            ]
        )
        if llama.get("soft_blockers"):
            go_nogo_lines.extend(["", "Remaining blockers on the Llama line:"])
            for item in llama["soft_blockers"]:
                go_nogo_lines.append(f"- {item}")
    (output_dir / "paper_repair_formal_go_nogo.md").write_text(
        "\n".join(go_nogo_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    json_dump(
        output_dir / "paper_repair_model_branch_comparison.json",
        {
            "qwen": qwen,
            "llama": llama,
            "verdict": verdict,
            "recommended_formal_branch": recommended_branch,
            "reason": why,
        },
    )


if __name__ == "__main__":
    main()
