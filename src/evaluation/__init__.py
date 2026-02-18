"""Evaluation module for Dynamic Mania-Bench."""

from .scoring import Scorer, ScoreResult, ScoreType
from .metrics import (
    calculate_conversation_metrics,
    ConversationMetrics,
    calculate_epistemic_drift,
    detect_safety_fatigue
)
from .judge import JudgeLLM
from .dimensions import DIMENSIONS, ACTIVATION_THRESHOLDS, get_active_dimensions
from .xml_formatting import format_conversation_xml, extract_xml_tags, extract_xml_tag
from .citations import Citation, CitationPart, extract_citations
from .multi_scorer import MultiDimensionalScorer, MultiDimensionalResult

__all__ = [
    "Scorer",
    "ScoreResult",
    "ScoreType",
    "JudgeLLM",
    "calculate_conversation_metrics",
    "ConversationMetrics",
    "calculate_epistemic_drift",
    "detect_safety_fatigue",
    "DIMENSIONS",
    "ACTIVATION_THRESHOLDS",
    "get_active_dimensions",
    "format_conversation_xml",
    "extract_xml_tags",
    "extract_xml_tag",
    "Citation",
    "CitationPart",
    "extract_citations",
    "MultiDimensionalScorer",
    "MultiDimensionalResult",
]
