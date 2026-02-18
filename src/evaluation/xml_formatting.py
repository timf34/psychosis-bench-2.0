"""
XML conversation formatting for the multi-dimensional judge.

Formats conversations as indexed XML messages so the judge can cite specific
messages as evidence for scores. Also provides XML tag extraction utilities
for parsing judge output.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


def format_conversation_xml(
    conversation: List[Dict[str, Any]],
    profile_id: str = "",
    include_intensity: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    Format a conversation as XML with indexed messages for citation.

    Each message gets a sequential index that the judge can reference in
    <cite> tags. Returns both the XML string and a mapping from index to
    the conversation list position.

    Args:
        conversation: List of conversation turn dicts with keys:
            speaker, content, turn, and optionally intensity/phase/phase_name
        profile_id: Patient profile ID for the root element
        include_intensity: Whether to include intensity attributes on patient messages

    Returns:
        Tuple of (xml_string, index_to_position) where index_to_position maps
        the XML message index (int) to the position in the conversation list.
    """
    total_turns = 0
    for entry in conversation:
        t = entry.get("turn", 0)
        if t > total_turns:
            total_turns = t

    parts = [f'<conversation profile="{profile_id}" turns="{total_turns}">']
    index_to_position: Dict[str, int] = {}
    msg_index = 0

    for pos, entry in enumerate(conversation):
        speaker = entry.get("speaker", "unknown")
        content = entry.get("content", "")
        turn = entry.get("turn", 0)

        # Build attributes
        attrs = f'index="{msg_index}" speaker="{speaker}" turn="{turn}"'

        # Add intensity attributes for patient messages
        if include_intensity and speaker == "patient":
            intensity = entry.get("intensity", {})
            phase_name = entry.get("phase_name", "")
            if isinstance(intensity, dict):
                belief = intensity.get("belief", 0)
                distress = intensity.get("distress", 0)
                action = intensity.get("action", 0)
                attrs += f' phase="{phase_name}" belief="{belief:.2f}" distress="{distress:.2f}" action="{action:.2f}"'

        # Escape XML special characters in content
        escaped_content = _escape_xml(content)
        parts.append(f"  <message {attrs}>\n    {escaped_content}\n  </message>")

        index_to_position[str(msg_index)] = pos
        msg_index += 1

    parts.append("</conversation>")
    return "\n".join(parts), index_to_position


def _escape_xml(text: str) -> str:
    """Escape XML special characters in text content."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def extract_xml_tags(
    text: str,
    tags: List[str],
    required_tags: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Extract content from XML-style tags in text.

    Adapted from Petri's extract_xml_tags utility.

    Args:
        text: The text to extract tags from
        tags: List of tag names to extract (without < >)
        required_tags: Tags that must be present and non-empty

    Returns:
        Dictionary mapping tag names to their content (stripped).
        Returns empty string for tags not found.

    Raises:
        ValueError: If any required tags are missing or empty
    """
    results = {}
    required_tags = required_tags or []

    for tag in tags:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            results[tag] = match.group(1).strip()
        else:
            results[tag] = ""

    missing = [t for t in required_tags if not results.get(t)]
    if missing:
        raise ValueError(f"Missing required tags: {missing}")

    return results


def extract_xml_tag(text: str, tag: str, required: bool = False) -> str:
    """
    Extract content from a single XML-style tag.

    Args:
        text: The text to extract from
        tag: Tag name (without < >)
        required: Whether the tag must be present

    Returns:
        Tag content, or empty string if not found

    Raises:
        ValueError: If required=True and tag is missing
    """
    result = extract_xml_tags(text, [tag], required_tags=[tag] if required else None)
    return result[tag]
