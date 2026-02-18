"""
Citation extraction and matching for the multi-dimensional judge.

Adapted from Petri's citation system. Citations let the judge reference exact
quotes from the conversation as evidence for dimension scores.

Uses layered matching:
1. Exact substring match against the claimed message
2. RapidFuzz fuzzy matching (partial_ratio_alignment, min score 80.0)
3. Search order: claimed message → neighbors → all messages
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum fuzzy match score to accept (0-100)
MIN_FUZZY_SCORE = 80.0


@dataclass
class CitationPart:
    """A single matched quote from a conversation message."""
    message_index: int  # Index in the conversation XML
    quoted_text: str  # The matched text
    position: Optional[Tuple[int, int]] = None  # (start, end) character position
    speaker: str = ""  # "patient" or "assistant"


@dataclass
class Citation:
    """A citation referencing one or more conversation messages as evidence."""
    parts: List[CitationPart] = field(default_factory=list)
    description: str = ""  # Why this is notable
    dimension: str = ""  # Which dimension this evidences
    index: int = 0  # Sequential number in judge output

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "index": self.index,
            "dimension": self.dimension,
            "description": self.description,
            "parts": [
                {
                    "message_index": p.message_index,
                    "quoted_text": p.quoted_text,
                    "speaker": p.speaker,
                    "position": list(p.position) if p.position else None,
                }
                for p in self.parts
            ],
        }


def extract_citations(
    text: str,
    conversation: List[Dict[str, Any]],
    index_to_position: Dict[str, int],
) -> List[Citation]:
    """
    Extract and resolve citations from judge output.

    Parses <cite> tags from the judge response and resolves quoted text
    against the conversation using layered matching.

    Args:
        text: Raw judge output containing <cite> tags
        conversation: The conversation list (same format as XML formatter input)
        index_to_position: Mapping from XML message index to conversation list position

    Returns:
        List of resolved Citation objects
    """
    if not text:
        return []

    # Pre-compute message content for matching
    message_contents: Dict[str, str] = {}
    message_speakers: Dict[str, str] = {}
    for idx_str, pos in index_to_position.items():
        if pos < len(conversation):
            entry = conversation[pos]
            message_contents[idx_str] = entry.get("content", "")
            message_speakers[idx_str] = entry.get("speaker", "")

    citations = []

    # Parse <cite> tags with attributes
    pattern = (
        r'<cite\s+'
        r'(?=(?:[^>]*\bindex="(\d+)")?)'
        r'(?=(?:[^>]*\bmessage="(\d+)")?)'
        r'(?=(?:[^>]*\bdimension="([^"]*)")?)'
        r'(?=(?:[^>]*\bdescription="([^"]*)")?)'
        r'[^>]*>'
        r'(.*?)'
        r'</cite>'
    )

    for match in re.finditer(pattern, text, re.DOTALL):
        cite_index = int(match.group(1)) if match.group(1) else len(citations) + 1
        claimed_msg = match.group(2) or "0"
        dimension = match.group(3) or ""
        description = match.group(4) or ""
        quoted_text = match.group(5).strip()

        if not quoted_text:
            continue

        # Handle multi-part citations (text contains [...])
        if "[...]" in quoted_text:
            parts = [
                _find_citation_part(
                    part.strip(), claimed_msg, message_contents, message_speakers, index_to_position
                )
                for part in quoted_text.split("[...]")
                if part.strip()
            ]
        else:
            parts = [
                _find_citation_part(
                    quoted_text, claimed_msg, message_contents, message_speakers, index_to_position
                )
            ]

        citations.append(Citation(
            parts=parts,
            description=description,
            dimension=dimension,
            index=cite_index,
        ))

    return citations


def _find_citation_part(
    quoted_text: str,
    claimed_index: str,
    message_contents: Dict[str, str],
    message_speakers: Dict[str, str],
    index_to_position: Dict[str, int],
) -> CitationPart:
    """
    Find a citation part using layered matching.

    Args:
        quoted_text: Text to find
        claimed_index: Message index the judge claimed
        message_contents: Pre-computed message text by index
        message_speakers: Pre-computed speaker by index
        index_to_position: Index-to-position mapping

    Returns:
        CitationPart with position if found
    """
    if not quoted_text:
        return CitationPart(
            message_index=int(claimed_index) if claimed_index.isdigit() else 0,
            quoted_text=quoted_text,
            speaker=message_speakers.get(claimed_index, ""),
        )

    # Build search order: claimed → neighbors → rest
    search_order = _get_search_order(claimed_index, list(message_contents.keys()))

    # Layer 1: Exact substring match
    for idx in search_order:
        content = message_contents.get(idx, "")
        if not content:
            continue
        pos = content.find(quoted_text)
        if pos != -1:
            return CitationPart(
                message_index=int(idx),
                quoted_text=quoted_text,
                position=(pos, pos + len(quoted_text)),
                speaker=message_speakers.get(idx, ""),
            )

    # Layer 2: Fuzzy matching
    best_score = 0.0
    best_part: Optional[CitationPart] = None

    try:
        from rapidfuzz import fuzz

        for idx in search_order:
            content = message_contents.get(idx, "")
            if not content:
                continue
            result = fuzz.partial_ratio_alignment(quoted_text, content)
            if result is not None and result.score >= MIN_FUZZY_SCORE and result.score > best_score:
                best_score = float(result.score)
                start = result.dest_start
                end = result.dest_end
                best_part = CitationPart(
                    message_index=int(idx),
                    quoted_text=content[start:end],
                    position=(start, end),
                    speaker=message_speakers.get(idx, ""),
                )
    except ImportError:
        logger.warning("rapidfuzz not installed, skipping fuzzy citation matching")

    if best_part is not None:
        return best_part

    # Not found — return with claimed index, no position
    return CitationPart(
        message_index=int(claimed_index) if claimed_index.isdigit() else 0,
        quoted_text=quoted_text,
        speaker=message_speakers.get(claimed_index, ""),
    )


def _get_search_order(claimed_index: str, all_indices: List[str]) -> List[str]:
    """
    Get message indices in priority order: claimed → neighbors → rest.

    Args:
        claimed_index: The index the judge claimed
        all_indices: All available indices

    Returns:
        Ordered list of indices to search
    """
    search_order: List[str] = []

    if claimed_index in all_indices:
        search_order.append(claimed_index)

    if claimed_index.isdigit():
        base = int(claimed_index)
        numeric = sorted(int(i) for i in all_indices if i.isdigit())
        if numeric:
            max_delta = max(abs(i - base) for i in numeric)
            for delta in range(1, max_delta + 1):
                for neighbor in (base - delta, base + delta):
                    idx_str = str(neighbor)
                    if idx_str in all_indices and idx_str not in search_order:
                        search_order.append(idx_str)

    for idx in all_indices:
        if idx not in search_order:
            search_order.append(idx)

    return search_order
