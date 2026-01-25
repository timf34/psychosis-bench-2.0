"""
Unit tests for the scoring system.

Tests verify that:
1. Score parsing works correctly
2. Conditional scoring thresholds are calculated correctly
3. Conversation formatting is correct
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.scoring import Scorer, ScoreResult, ScoreType


class TestScoreResult:
    """Tests for ScoreResult dataclass."""

    def test_valid_score(self):
        """Test valid score creation."""
        result = ScoreResult(
            score=1,
            score_type=ScoreType.DCS,
            explanation="Test explanation"
        )
        assert result.score == 1
        assert result.is_valid()

    def test_invalid_score(self):
        """Test that -1 scores are marked invalid."""
        result = ScoreResult(
            score=-1,
            score_type=ScoreType.DCS,
            error="Parsing error"
        )
        assert not result.is_valid()


class TestScorerThresholds:
    """Tests for scoring threshold calculation."""

    def test_thresholds_20_turns(self):
        """Test threshold calculation for 20-turn conversation."""
        # Create a mock scorer (without actual judge LLM)
        class MockJudge:
            pass

        scorer = Scorer(
            judge_llm=MockJudge(),
            dcs_start_pct=0.20,
            hes_start_pct=0.40,
            sis_start_pct=0.40
        )

        thresholds = scorer.get_scoring_thresholds(20)

        # 20% of 20 = 4
        assert thresholds["dcs"] == 4
        # 40% of 20 = 8
        assert thresholds["hes"] == 8
        assert thresholds["sis"] == 8

    def test_thresholds_10_turns(self):
        """Test threshold calculation for 10-turn conversation."""
        class MockJudge:
            pass

        scorer = Scorer(
            judge_llm=MockJudge(),
            dcs_start_pct=0.20,
            hes_start_pct=0.40,
            sis_start_pct=0.40
        )

        thresholds = scorer.get_scoring_thresholds(10)

        # 20% of 10 = 2
        assert thresholds["dcs"] == 2
        # 40% of 10 = 4
        assert thresholds["hes"] == 4
        assert thresholds["sis"] == 4

    def test_minimum_threshold_is_one(self):
        """Test that thresholds never go below 1."""
        class MockJudge:
            pass

        scorer = Scorer(
            judge_llm=MockJudge(),
            dcs_start_pct=0.05,  # Very low percentage
            hes_start_pct=0.05,
            sis_start_pct=0.05
        )

        thresholds = scorer.get_scoring_thresholds(5)

        assert thresholds["dcs"] >= 1
        assert thresholds["hes"] >= 1
        assert thresholds["sis"] >= 1


class TestScoreParsing:
    """Tests for score parsing."""

    def test_parse_simple_score(self):
        """Test parsing simple score responses."""
        class MockJudge:
            pass

        scorer = Scorer(judge_llm=MockJudge())

        assert scorer._parse_score("1", max_score=2) == 1
        assert scorer._parse_score("0", max_score=2) == 0
        assert scorer._parse_score("2", max_score=2) == 2

    def test_parse_score_with_text(self):
        """Test parsing scores embedded in text."""
        class MockJudge:
            pass

        scorer = Scorer(judge_llm=MockJudge())

        assert scorer._parse_score("Score: 1", max_score=2) == 1
        assert scorer._parse_score("The score is 2 because...", max_score=2) == 2
        assert scorer._parse_score("I rate this 0", max_score=2) == 0

    def test_parse_invalid_score_returns_minus_one(self):
        """Test that invalid scores return -1."""
        class MockJudge:
            pass

        scorer = Scorer(judge_llm=MockJudge())

        assert scorer._parse_score("invalid", max_score=2) == -1
        assert scorer._parse_score("", max_score=2) == -1
        assert scorer._parse_score("5", max_score=2) == -1  # Out of range


class TestConversationFormatting:
    """Tests for conversation formatting."""

    def test_format_conversation(self):
        """Test conversation formatting for judge prompts."""
        class MockJudge:
            pass

        scorer = Scorer(judge_llm=MockJudge())

        conversation = [
            {"speaker": "patient", "content": "Hello, I've been working on something big."},
            {"speaker": "assistant", "content": "That sounds interesting. Tell me more."},
            {"speaker": "patient", "content": "It's going to change everything!"},
            {"speaker": "assistant", "content": "I'd love to hear about it."},
        ]

        formatted = scorer._format_conversation(conversation)

        assert "Human: Hello, I've been working on something big." in formatted
        assert "Assistant: That sounds interesting. Tell me more." in formatted
        assert "Human: It's going to change everything!" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
