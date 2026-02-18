"""Tests for XML conversation formatting and tag extraction."""

import pytest

from src.evaluation.xml_formatting import (
    format_conversation_xml,
    extract_xml_tags,
    extract_xml_tag,
)


class TestFormatConversationXml:
    """Test conversation-to-XML formatting."""

    def test_basic_formatting(self):
        """Format a simple conversation."""
        conversation = [
            {"speaker": "patient", "content": "Hello", "turn": 1},
            {"speaker": "assistant", "content": "Hi there", "turn": 1},
        ]
        xml, idx_map = format_conversation_xml(conversation, profile_id="test_profile")

        assert '<conversation profile="test_profile" turns="1">' in xml
        assert 'index="0"' in xml
        assert 'index="1"' in xml
        assert 'speaker="patient"' in xml
        assert 'speaker="assistant"' in xml
        assert "Hello" in xml
        assert "Hi there" in xml
        assert "</conversation>" in xml

    def test_index_to_position_mapping(self):
        """Index map should map XML indices to conversation positions."""
        conversation = [
            {"speaker": "patient", "content": "A", "turn": 1},
            {"speaker": "assistant", "content": "B", "turn": 1},
            {"speaker": "patient", "content": "C", "turn": 2},
        ]
        _, idx_map = format_conversation_xml(conversation)

        assert idx_map["0"] == 0
        assert idx_map["1"] == 1
        assert idx_map["2"] == 2

    def test_intensity_attributes(self):
        """Patient messages should include intensity attributes."""
        conversation = [
            {
                "speaker": "patient",
                "content": "Hello",
                "turn": 1,
                "intensity": {"belief": 0.5, "distress": 0.3, "action": 0.1},
                "phase_name": "rapport_building",
            },
        ]
        xml, _ = format_conversation_xml(conversation, include_intensity=True)

        assert 'belief="0.50"' in xml
        assert 'distress="0.30"' in xml
        assert 'action="0.10"' in xml
        assert 'phase="rapport_building"' in xml

    def test_no_intensity_when_disabled(self):
        """Should not include intensity when disabled."""
        conversation = [
            {
                "speaker": "patient",
                "content": "Hello",
                "turn": 1,
                "intensity": {"belief": 0.5, "distress": 0.3, "action": 0.1},
                "phase_name": "rapport_building",
            },
        ]
        xml, _ = format_conversation_xml(conversation, include_intensity=False)

        assert "belief=" not in xml
        assert "distress=" not in xml

    def test_xml_escaping(self):
        """Special characters should be escaped."""
        conversation = [
            {"speaker": "patient", "content": "I said <hello> & goodbye", "turn": 1},
        ]
        xml, _ = format_conversation_xml(conversation)

        assert "&lt;hello&gt;" in xml
        assert "&amp;" in xml

    def test_empty_conversation(self):
        """Empty conversation should produce valid XML."""
        xml, idx_map = format_conversation_xml([])

        assert '<conversation profile="" turns="0">' in xml
        assert "</conversation>" in xml
        assert len(idx_map) == 0

    def test_turn_count_calculation(self):
        """Total turns should be calculated from max turn number."""
        conversation = [
            {"speaker": "patient", "content": "A", "turn": 1},
            {"speaker": "assistant", "content": "B", "turn": 1},
            {"speaker": "patient", "content": "C", "turn": 5},
        ]
        xml, _ = format_conversation_xml(conversation)

        assert 'turns="5"' in xml


class TestExtractXmlTags:
    """Test XML tag extraction."""

    def test_extract_single_tag(self):
        """Extract a single tag."""
        text = "<summary>Hello world</summary>"
        result = extract_xml_tags(text, ["summary"])
        assert result["summary"] == "Hello world"

    def test_extract_multiple_tags(self):
        """Extract multiple tags."""
        text = "<summary>Hello</summary><scores>{}</scores>"
        result = extract_xml_tags(text, ["summary", "scores"])
        assert result["summary"] == "Hello"
        assert result["scores"] == "{}"

    def test_missing_tag_returns_empty(self):
        """Missing tags should return empty string."""
        text = "<summary>Hello</summary>"
        result = extract_xml_tags(text, ["summary", "missing"])
        assert result["summary"] == "Hello"
        assert result["missing"] == ""

    def test_required_tag_raises_on_missing(self):
        """Required missing tags should raise ValueError."""
        text = "<summary>Hello</summary>"
        with pytest.raises(ValueError, match="Missing required tags"):
            extract_xml_tags(text, ["summary", "scores"], required_tags=["scores"])

    def test_multiline_content(self):
        """Should handle multiline content."""
        text = "<reasoning>\nLine 1\nLine 2\n</reasoning>"
        result = extract_xml_tags(text, ["reasoning"])
        assert "Line 1" in result["reasoning"]
        assert "Line 2" in result["reasoning"]

    def test_strips_whitespace(self):
        """Content should be stripped."""
        text = "<tag>  hello  </tag>"
        result = extract_xml_tags(text, ["tag"])
        assert result["tag"] == "hello"


class TestExtractXmlTag:
    """Test single tag extraction."""

    def test_extract_existing_tag(self):
        """Extract an existing tag."""
        result = extract_xml_tag("<score>5</score>", "score")
        assert result == "5"

    def test_missing_tag_returns_empty(self):
        """Missing non-required tag returns empty string."""
        result = extract_xml_tag("<other>5</other>", "score")
        assert result == ""

    def test_required_missing_raises(self):
        """Required missing tag raises ValueError."""
        with pytest.raises(ValueError):
            extract_xml_tag("<other>5</other>", "score", required=True)
