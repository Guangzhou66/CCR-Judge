from __future__ import annotations

from typing import Any, Sequence

from paper_repair.eval.parsing import normalize_selected_agent_id
from paper_repair.repair.ccr_types import CCRCandidateView


def ccr_primary_selection_is_usable(
    candidate_outputs: Sequence[CCRCandidateView],
    selected_agent_id: Any,
) -> bool:
    candidate_ids = [str(candidate.agent_id) for candidate in candidate_outputs]
    return normalize_selected_agent_id(selected_agent_id, allowed_ids=candidate_ids) is not None


primary_selection_is_usable = ccr_primary_selection_is_usable
