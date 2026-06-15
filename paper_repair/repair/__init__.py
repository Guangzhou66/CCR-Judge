"""Interaction-aware decision-context components for final selection."""

from paper_repair.repair.ccr_context import (
    build_ccr_context,
    build_ccr_context_payload,
    render_ccr_context_text,
)
from paper_repair.repair.ccr_fallback import ccr_primary_selection_is_usable
from paper_repair.repair.ccr_shortlist import select_ccr_shortlist
from paper_repair.repair.ccr_state import build_ccr_state
from paper_repair.repair.ccr_types import (
    CCRCandidateView,
    CCRContext,
    CCRContextPayload,
    CCRState,
)

__all__ = [
    "CCRCandidateView",
    "CCRContext",
    "CCRContextPayload",
    "CCRState",
    "build_ccr_context",
    "build_ccr_context_payload",
    "build_ccr_state",
    "ccr_primary_selection_is_usable",
    "render_ccr_context_text",
    "select_ccr_shortlist",
]
