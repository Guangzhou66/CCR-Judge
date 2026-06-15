from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


CHOICES = ("A", "B", "C", "D")
DENSE_PARSE_FAILURE_REASONS = (
    "missing_selected_agent_header",
    "invalid_agent_id",
    "malformed_selected_agent_header",
    "multiple_candidate_ids_found",
    "answer_only_no_agent_id",
    "truncated_output",
    "runtime_exception",
    "timeout",
    "retry_failed",
    "parser_internal_error",
)
SELECTED_HEADER_RE = re.compile(
    r"^Selected agent id(?:\s*\(\s*(dense|masked)\s*\))?(?:\s*[:|]\s*|\s+)(.*?)\s*$",
    flags=re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```(?:[\w+-]+)?|```", flags=re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+|[A-Za-z])[\.\)])\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_AGENT_ID_PATTERNS = [
    re.compile(r"selected\s+agent\s+id(?:\s*[:=#|\-]\s*|\s+)(-?\d+)", re.IGNORECASE),
    re.compile(r"selected(?:\s*[:=#|\-]\s*|\s+)(?:agent\s*)?#?\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"chosen\s*(?:agent\s*id|agent)?(?:\s*[:=#|\-]\s*|\s+)#?\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"winner\s*(?:agent\s*id|agent)?(?:\s*[:=#|\-]\s*|\s+)#?\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"best\s+agent(?:\s*[:=#|\-]\s*|\s+)#?\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"agent\s*#?\s*(-?\d+)\s*(?:is\s*)?(?:selected|chosen|winner|wins|best)", re.IGNORECASE),
    re.compile(r"^id(?:\s*[:=#|\-]\s*|\s+)(-?\d+)$", re.IGNORECASE),
]
_ANSWER_PATTERNS = [
    re.compile(r"^(?:answer|final answer|choice)(?:\s*[:=#|\-]\s*|\s+)([ABCD])\b", re.IGNORECASE),
    re.compile(r"^([ABCD])\b", re.IGNORECASE),
]
_MMLU_DIRECT_CHOICE_PATTERNS = [
    re.compile(
        r"^\s*(?:the\s+)?(?:answer|final\s+answer|choice|final\s+choice|selected\s+answer)"
        r"(?:\s*[:=#|\-]\s*|\s+is\s+|\s+)?(?:option\s*)?\(?([ABCD])\)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:the\s+)?(?:correct|best|final)\s+(?:answer|choice)"
        r"(?:\s*[:=#|\-]\s*|\s+is\s+|\s+)?(?:option\s*)?\(?([ABCD])\)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:correct|best|final)\s+(?:answer|choice)"
        r"(?:\s*[:=#|\-]\s*|\s+is\s+)(?:option\s*)?\(?([ABCD])\)?\b",
        re.IGNORECASE,
    ),
]
_MMLU_OPTION_LINE_RE = re.compile(r"^\s*(?:option|choice)\s*([ABCD])(?:\b|[\).:\-])", re.IGNORECASE)
_MMLU_BARE_CHOICE_RE = re.compile(r"^\s*([ABCD])(?:\b|[\).:\-])", re.IGNORECASE)
_NUMERIC_BULLET_RE = re.compile(r"^\s*\d+\s*[\.\):]\s*")
_MMLU_PENDING_ANSWER_RE = re.compile(
    r"^\s*(?:the\s+)?(?:(?:correct|best|final)\s+)?(?:answer|choice)"
    r"(?:\s+is)?\s*[:=#|\-]?\s*$",
    re.IGNORECASE,
)
_SELECTION_HINT_RE = re.compile(
    r"(selected|chosen|winner|best\s+agent|agent\s+id|^id\s*[:=#-])",
    re.IGNORECASE,
)


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().upper()
    else:
        text = str(value).strip().upper()
    if not text:
        return None
    if text in CHOICES:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return None


def normalize_choice_or_none(value: Any) -> str:
    choice = normalize_choice(value)
    return choice if choice is not None else "NONE"


def failure_reason_counts(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if not value:
            continue
        counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def normalize_selected_agent_id(
    value: Any,
    *,
    allowed_ids: Sequence[str] | None = None,
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"-1", "none", "null", "n/a"}:
        return None
    if allowed_ids is None:
        return text
    allowed = {str(item) for item in allowed_ids}
    return text if text in allowed else None


def _extract_field(text: str, label: str) -> str | None:
    pattern = rf"(?im)^\s*{re.escape(label)}(?:\s*[:|]\s*|\s+)(.+?)\s*$"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip()


def _normalize_answer_line(line: str) -> str:
    return (
        str(line or "")
        .strip()
        .replace("：", ":")
        .replace("﹕", ":")
        .replace("\u00a0", " ")
    )


def _mmlu_line_variants(line: str) -> List[str]:
    normalized = _normalize_answer_line(line)
    variants = [normalized]
    numeric_bullet_stripped = _NUMERIC_BULLET_RE.sub("", normalized, count=1)
    if numeric_bullet_stripped != normalized:
        variants.append(numeric_bullet_stripped)
    return variants


def _direct_mmlu_choice_from_line(line: str) -> str | None:
    for normalized in _mmlu_line_variants(line):
        for pattern in _MMLU_DIRECT_CHOICE_PATTERNS:
            match = pattern.search(normalized)
            if match:
                return match.group(1).upper()
    return None


def _option_mmlu_choice_from_line(line: str) -> str | None:
    for normalized in _mmlu_line_variants(line):
        match = _MMLU_OPTION_LINE_RE.search(normalized)
        if match:
            return match.group(1).upper()
    return None


def _bare_mmlu_choice_from_line(line: str) -> str | None:
    for normalized in _mmlu_line_variants(line):
        match = _MMLU_BARE_CHOICE_RE.search(normalized)
        if match:
            return match.group(1).upper()
    return None


def _extract_mmlu_choice_from_text(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = strip_selected_agent_header(text).strip()
    if not stripped:
        return None

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    pending_answer_label = False
    early_option_fallback = None
    bare_fallback = None

    for idx, line in enumerate(lines):
        direct_choice = _direct_mmlu_choice_from_line(line)
        if direct_choice is not None:
            return direct_choice

        option_choice = _option_mmlu_choice_from_line(line)
        bare_choice = _bare_mmlu_choice_from_line(line)
        if pending_answer_label:
            if option_choice is not None:
                return option_choice
            if bare_choice is not None:
                return bare_choice

        if _MMLU_PENDING_ANSWER_RE.match(_normalize_answer_line(line)):
            pending_answer_label = True
            continue

        pending_answer_label = False
        if option_choice is not None and idx <= 2 and early_option_fallback is None:
            early_option_fallback = option_choice
        if bare_choice is not None and idx == 0 and bare_fallback is None:
            bare_fallback = bare_choice

    return bare_fallback or early_option_fallback


def parse_candidate_output(text: Any) -> Dict[str, Any]:
    raw_text = text if isinstance(text, str) else str(text or "")
    stripped = raw_text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]

    final_answer = (
        normalize_choice(_extract_field(stripped, "Final answer"))
        or normalize_choice(_extract_field(stripped, "Answer"))
        or normalize_choice(_extract_field(stripped, "Choice"))
        or _extract_mmlu_choice_from_text(stripped)
    )
    strongest_alt_raw = _extract_field(stripped, "Strongest alternative considered")
    supported_raw = _extract_field(stripped, "Is this alternative supported by another candidate?")
    rejected_option_raw = _extract_field(stripped, "Rejected option")
    why_not = _extract_field(stripped, "Why not that option")
    conclusion = _extract_field(stripped, "Conclusion")
    evidence = _extract_field(stripped, "Evidence")
    overturn = _extract_field(
        stripped, "Does the alternative overturn the current best answer?"
    )

    if overturn is None:
        for line in lines:
            if line.upper() in {"YES", "NO"}:
                overturn = line.upper()
                break
    if supported_raw is None:
        supported_raw = _extract_field(stripped, "Alternative supported")

    non_field_lines: List[str] = []
    for line in lines[1:]:
        lower = line.lower()
        if lower.startswith("final answer:"):
            continue
        if lower.startswith("conclusion:"):
            continue
        if lower.startswith("evidence:"):
            continue
        if lower.startswith("strongest alternative considered:"):
            continue
        if lower.startswith("is this alternative supported by another candidate?"):
            continue
        if lower.startswith("alternative supported:"):
            continue
        if lower.startswith("rejected option:"):
            continue
        if lower.startswith("why not that option:"):
            continue
        if lower.startswith("does the alternative overturn the current best answer?"):
            continue
        if line.upper() in {"YES", "NO"}:
            continue
        non_field_lines.append(line)

    if conclusion is None:
        conclusion = non_field_lines[0] if non_field_lines else "Not stated."
    if evidence is None:
        evidence = non_field_lines[1] if len(non_field_lines) > 1 else conclusion
    if why_not is None:
        why_not = "Not stated."

    strongest_alt = normalize_choice_or_none(strongest_alt_raw)
    rejected_option = normalize_choice_or_none(rejected_option_raw)
    supported_flag = (supported_raw or "NO").strip().upper()
    if supported_flag not in {"YES", "NO"}:
        supported_flag = "NO"
    overturn_flag = (overturn or "NO").strip().upper()
    if overturn_flag not in {"YES", "NO"}:
        overturn_flag = "NO"

    return {
        "text": raw_text,
        "final_answer": final_answer,
        "normalized_answer": final_answer,
        "conclusion": conclusion[:160],
        "evidence": evidence[:160],
        "strongest_alternative_considered": strongest_alt if strongest_alt != "NONE" else "NONE",
        "strongest_alternative_option": strongest_alt,
        "alternative_supported_flag": supported_flag,
        "alternative_supported_yes": supported_flag == "YES",
        "rejected_option": rejected_option,
        "why_not_that_option": why_not[:160],
        "why_not_rejected_option": why_not[:160],
        "alternative_overturn_flag": overturn_flag,
        "alternative_overturn_yes": overturn_flag == "YES",
    }


def extract_selected_agent_ids(judge_text: Any) -> Dict[str, str | None]:
    result: Dict[str, str | None] = {
        "selected_agent_id": None,
        "dense_selected_agent_id": None,
        "masked_selected_agent_id": None,
    }
    if not isinstance(judge_text, str):
        return result
    stripped = judge_text.strip()
    if not stripped:
        return result

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        match = SELECTED_HEADER_RE.match(line)
        if not match:
            break
        qualifier, raw_value = match.groups()
        qualifier = (qualifier or "").lower()
        value = normalize_selected_agent_id(raw_value)
        if qualifier == "dense":
            result["dense_selected_agent_id"] = value
        elif qualifier == "masked":
            result["masked_selected_agent_id"] = value
        else:
            result["selected_agent_id"] = value
    return result


def strip_selected_agent_header(judge_text: Any) -> str:
    if not isinstance(judge_text, str):
        return str(judge_text)
    stripped = judge_text.lstrip()
    if not stripped:
        return judge_text
    lines = stripped.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if not SELECTED_HEADER_RE.match(line):
            break
        idx += 1
    return "\n".join(lines[idx:]).lstrip()


def sanitize_dense_reference_text(judge_text: Any) -> str:
    text = judge_text if isinstance(judge_text, str) else str(judge_text or "")
    text = _CODE_FENCE_RE.sub("", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = text.replace("：", ":")
    text = text.replace("﹕", ":")
    text = text.replace("‒", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(?im)^\s*(Selected agent id(?:\s*\(\s*(?:dense|masked)\s*\))?)\s*\|\s*", r"\1: ", text)
    text = re.sub(r"(?im)^\s*((?:Answer|Final answer|Choice))\s*\|\s*", r"\1: ", text)
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _BULLET_RE.sub("", line).strip()
        line = _WHITESPACE_RE.sub(" ", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def normalize_dense_reference_text(
    judge_text: Any,
    *,
    allowed_ids: Sequence[str],
    line_window: int = 6,
) -> Dict[str, Any]:
    sanitized_text = sanitize_dense_reference_text(judge_text)
    lines = [line.strip() for line in sanitized_text.splitlines() if line.strip()]
    inspected_lines = lines[: max(1, int(line_window))]

    answer_value = None
    for line in inspected_lines:
        for pattern in _ANSWER_PATTERNS:
            match = pattern.search(line)
            if match:
                answer_value = normalize_choice(match.group(1))
                break
        if answer_value is not None:
            break

    selection_like_lines = [line for line in inspected_lines if _SELECTION_HINT_RE.search(line)]
    explicit_id_values: List[str] = []
    invalid_id_values: List[str] = []
    malformed_selected_header = False
    for line in selection_like_lines:
        line_has_match = False
        for pattern in _AGENT_ID_PATTERNS:
            for match in pattern.finditer(line):
                line_has_match = True
                value = str(match.group(1)).strip()
                normalized = normalize_selected_agent_id(value, allowed_ids=allowed_ids)
                if normalized is None:
                    invalid_id_values.append(value)
                else:
                    explicit_id_values.append(normalized)
        if ("selected" in line.lower() or "agent id" in line.lower()) and not line_has_match:
            malformed_selected_header = True

    unique_explicit_ids = sorted(set(explicit_id_values))
    failure_reason = None
    selected_agent_id = None
    if len(unique_explicit_ids) > 1:
        failure_reason = "multiple_candidate_ids_found"
    elif len(unique_explicit_ids) == 1:
        selected_agent_id = unique_explicit_ids[0]
    elif invalid_id_values:
        failure_reason = "invalid_agent_id"
    elif answer_value is not None:
        failure_reason = "answer_only_no_agent_id"
    elif malformed_selected_header:
        failure_reason = "malformed_selected_agent_header"
    elif re.search(r"(selected\s+agent\s+id|answer)\s*[:=#-]?\s*$", sanitized_text, re.IGNORECASE):
        failure_reason = "truncated_output"
    else:
        failure_reason = "missing_selected_agent_header"

    normalized_lines: List[str] = []
    if selected_agent_id is not None:
        normalized_lines.append(f"Selected agent id: {selected_agent_id}")
    if answer_value is not None:
        normalized_lines.append(f"Answer: {answer_value}")
    for line in inspected_lines:
        if line in normalized_lines:
            continue
        normalized_lines.append(line)
    normalized_text = "\n".join(normalized_lines).strip()
    return {
        "sanitized_text": sanitized_text,
        "normalized_text": normalized_text,
        "inspected_lines": inspected_lines,
        "selection_like_lines": selection_like_lines,
        "selected_agent_id": selected_agent_id,
        "selected_answer": answer_value,
        "found_candidate_ids": unique_explicit_ids,
        "invalid_candidate_ids": invalid_id_values,
        "parse_success": bool(selected_agent_id is not None),
        "failure_reason": None if selected_agent_id is not None else failure_reason,
    }


def parse_dense_reference_output(
    judge_text: Any,
    *,
    allowed_ids: Sequence[str],
    line_window: int = 6,
) -> Dict[str, Any]:
    try:
        return normalize_dense_reference_text(
            judge_text,
            allowed_ids=allowed_ids,
            line_window=line_window,
        )
    except Exception as exc:
        text = judge_text if isinstance(judge_text, str) else str(judge_text or "")
        return {
            "sanitized_text": sanitize_dense_reference_text(text),
            "normalized_text": sanitize_dense_reference_text(text),
            "inspected_lines": [],
            "selection_like_lines": [],
            "selected_agent_id": None,
            "selected_answer": None,
            "found_candidate_ids": [],
            "invalid_candidate_ids": [],
            "parse_success": False,
            "failure_reason": "parser_internal_error",
            "parser_error_type": type(exc).__name__,
            "parser_error_message": str(exc),
        }


def extract_mmlu_choice_letter(text: Any) -> str | None:
    return _extract_mmlu_choice_from_text(text)


def infer_selected_agent_id_from_text(
    judge_text: Any,
    candidate_agent_ids: Sequence[str],
) -> str | None:
    if not isinstance(judge_text, str) or not judge_text.strip():
        return None
    candidate_set = {str(item) for item in candidate_agent_ids}

    explicit = re.search(r"Selected agent id\s*:\s*(\d+)", judge_text, re.IGNORECASE)
    if explicit:
        selected = explicit.group(1)
        return selected if selected in candidate_set else None

    tie_patterns = [
        r"\ball\b.*\bagents\b.*\b(same|identical|consistent)\b",
        r"\ball\b.*\bcandidate answers\b.*\b(same|identical)\b",
        r"\ball\b.*\banswers\b.*\b(same|identical)\b",
    ]
    for pattern in tie_patterns:
        if re.search(pattern, judge_text, re.IGNORECASE | re.DOTALL):
            return str(candidate_agent_ids[0]) if candidate_agent_ids else None

    patterns = [
        r"most reliable answer is provided by Agent\s*(\d+)",
        r"(?:choose|select|selected)\s+Agent\s*(\d+)",
        r"Agent\s*(\d+)\s+is\s+(?:the\s+)?(?:best|most reliable|most accurate|correct)",
        r"The best answer is from Agent\s*(\d+)",
        r"Based on the analysis,\s*Agent\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, judge_text, re.IGNORECASE)
        if match:
            selected = match.group(1)
            return selected if selected in candidate_set else None
    return None


def _candidate_value(candidate: Any, field: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(field)
    return getattr(candidate, field, None)


def infer_selected_agent_id_from_choice(
    judge_text: Any,
    candidates: Sequence[Any],
) -> str | None:
    judge_choice = extract_mmlu_choice_letter(judge_text)
    if judge_choice is None:
        return None
    matches: List[str] = []
    for candidate in candidates:
        raw_text = _candidate_value(candidate, "text") or _candidate_value(candidate, "output")
        candidate_choice = extract_mmlu_choice_letter(raw_text)
        if candidate_choice is None:
            candidate_choice = normalize_choice(_candidate_value(candidate, "normalized_answer"))
        if candidate_choice == judge_choice:
            matches.append(str(_candidate_value(candidate, "agent_id")))
    if not matches:
        return None
    return matches[0]


def resolve_selected_agent_id(
    *,
    judge_text: Any,
    candidates: Sequence[Any],
    metadata_selected_agent_id: Any = None,
) -> str | None:
    candidate_ids = [str(_candidate_value(candidate, "agent_id")) for candidate in candidates]
    metadata_value = normalize_selected_agent_id(
        metadata_selected_agent_id,
        allowed_ids=candidate_ids,
    )
    if metadata_value is not None:
        return metadata_value

    header_value = normalize_selected_agent_id(
        extract_selected_agent_ids(judge_text).get("selected_agent_id"),
        allowed_ids=candidate_ids,
    )
    if header_value is not None:
        return header_value

    explicit_value = infer_selected_agent_id_from_text(judge_text, candidate_ids)
    if explicit_value is not None:
        return explicit_value

    return infer_selected_agent_id_from_choice(judge_text, candidates)


def parse_dense_retry_response(
    judge_text: Any,
    *,
    allowed_ids: Sequence[str],
) -> Dict[str, str | None]:
    parsed = parse_dense_reference_output(
        judge_text,
        allowed_ids=allowed_ids,
    )
    selected_agent_id = parsed.get("selected_agent_id")
    selected_answer = parsed.get("selected_answer")
    if selected_answer is None:
        selected_answer = extract_mmlu_choice_letter(judge_text)
    if selected_agent_id is None:
        selected_agent_id = infer_selected_agent_id_from_text(judge_text, allowed_ids)
    return {
        "selected_agent_id": selected_agent_id,
        "selected_answer": normalize_choice(selected_answer),
        "failure_reason": parsed.get("failure_reason"),
        "normalized_text": parsed.get("normalized_text"),
        "sanitized_text": parsed.get("sanitized_text"),
    }


def canonicalize_candidate_output(parsed: Dict[str, Any]) -> str:
    final_answer = normalize_choice_or_none(parsed.get("normalized_answer"))
    strongest_alt = normalize_choice_or_none(parsed.get("strongest_alternative_option"))
    supported_flag = (parsed.get("alternative_supported_flag") or "NO").strip().upper()
    if supported_flag not in {"YES", "NO"}:
        supported_flag = "NO"
    rejected_option = normalize_choice_or_none(parsed.get("rejected_option"))
    overturn_flag = (parsed.get("alternative_overturn_flag") or "NO").strip().upper()
    if overturn_flag not in {"YES", "NO"}:
        overturn_flag = "NO"
    conclusion = str(parsed.get("conclusion") or "Not stated.").strip()
    evidence = str(parsed.get("evidence") or conclusion).strip()
    why_not = str(parsed.get("why_not_that_option") or "Not stated.").strip()
    return "\n".join(
        [
            final_answer,
            f"Final answer: {final_answer}",
            f"Conclusion: {conclusion}",
            f"Evidence: {evidence}",
            f"Strongest alternative considered: {strongest_alt}",
            f"Is this alternative supported by another candidate? {supported_flag}",
            f"Rejected option: {rejected_option}",
            f"Why not that option: {why_not}",
            f"Does the alternative overturn the current best answer? {overturn_flag}",
        ]
    )


def selected_answer_for_agent(candidates: Iterable[Any], agent_id: Any) -> str | None:
    normalized_id = None if agent_id is None else str(agent_id)
    for candidate in candidates:
        if str(_candidate_value(candidate, "agent_id")) != normalized_id:
            continue
        raw_text = _candidate_value(candidate, "text") or _candidate_value(candidate, "output")
        reparsed_choice = extract_mmlu_choice_letter(raw_text)
        if reparsed_choice is not None:
            return reparsed_choice
        value = _candidate_value(candidate, "normalized_answer")
        if value is None:
            return None
        return str(value).strip()
    return None
