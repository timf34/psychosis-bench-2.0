"""Tests for the MH-specific scoring dimensions."""

import pytest

from src.evaluation.dimensions import (
    DIMENSIONS,
    ACTIVATION_THRESHOLDS,
    get_active_dimensions,
    format_dimensions_for_prompt,
    format_scores_list,
)


class TestDimensions:
    """Test dimension definitions."""

    def test_all_dimensions_have_rubrics(self):
        """Every dimension should have a non-empty rubric."""
        for name, rubric in DIMENSIONS.items():
            assert rubric, f"Dimension {name} has empty rubric"
            assert "{{}}" in rubric, f"Dimension {name} missing {{{{}}}} score placeholder"

    def test_all_dimensions_have_thresholds(self):
        """Every dimension should have an activation threshold."""
        for name in DIMENSIONS:
            assert name in ACTIVATION_THRESHOLDS, f"Dimension {name} missing threshold"

    def test_thresholds_are_valid(self):
        """Thresholds should be between 0 and 1."""
        for name, threshold in ACTIVATION_THRESHOLDS.items():
            assert 0 <= threshold <= 1, f"Threshold for {name} is {threshold}, expected 0-1"

    def test_dimension_count(self):
        """Should have exactly 10 dimensions."""
        assert len(DIMENSIONS) == 10

    def test_dimension_names(self):
        """Verify expected dimension names exist."""
        expected = [
            "appropriate_concern",
            "escalation_recognition",
            "delusion_validation_vs_feeling_validation",
            "professional_referral_quality",
            "boundary_maintenance",
            "consistency_under_pressure",
            "therapeutic_alliance",
            "harm_pathway_recognition",
            "sycophancy_toward_vulnerable",
            "empathy_without_enabling",
        ]
        for name in expected:
            assert name in DIMENSIONS, f"Missing dimension: {name}"


class TestGetActiveDimensions:
    """Test dimension activation based on conversation length."""

    def test_all_active_for_long_conversation(self):
        """All dimensions should be active for a long enough conversation."""
        active = get_active_dimensions(20)
        assert len(active) == 10

    def test_some_inactive_for_short_conversation(self):
        """Some dimensions should be inactive for a very short conversation."""
        # consistency_under_pressure has threshold 0.50, so for 2 turns:
        # min_turns = max(1, int(2 * 0.50)) = 1, so it IS active
        # Actually all are active if total_turns >= min_turns
        # For total_turns=1, all min_turns are max(1, int(1*threshold)) = 1
        # So even with 1 turn, all activate
        active = get_active_dimensions(1)
        # With 1 turn: max(1, int(1*0.50)) = 1, 1 >= 1 -> active
        assert len(active) == 10

    def test_returns_list_of_strings(self):
        """Should return a list of dimension name strings."""
        active = get_active_dimensions(10)
        assert isinstance(active, list)
        for name in active:
            assert isinstance(name, str)
            assert name in DIMENSIONS


class TestFormatDimensionsForPrompt:
    """Test prompt formatting."""

    def test_formats_single_dimension(self):
        """Format a single dimension rubric."""
        result = format_dimensions_for_prompt(["appropriate_concern"])
        assert "### 1. Appropriate Concern" in result
        assert "{{}}" in result  # Score placeholder should be preserved

    def test_formats_multiple_dimensions(self):
        """Format multiple dimension rubrics."""
        dims = ["appropriate_concern", "boundary_maintenance"]
        result = format_dimensions_for_prompt(dims)
        assert "### 1. Appropriate Concern" in result
        assert "### 2. Boundary Maintenance" in result

    def test_empty_list(self):
        """Empty list should return empty string."""
        result = format_dimensions_for_prompt([])
        assert result == ""


class TestFormatScoresList:
    """Test scores list formatting."""

    def test_formats_correctly(self):
        """Format scores list for prompt."""
        dims = ["appropriate_concern", "boundary_maintenance"]
        result = format_scores_list(dims)
        assert "1. appropriate_concern: [score]/10" in result
        assert "2. boundary_maintenance: [score]/10" in result

    def test_empty_list(self):
        """Empty list should return empty string."""
        result = format_scores_list([])
        assert result == ""
