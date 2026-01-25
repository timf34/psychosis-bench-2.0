"""
Unit tests for the Intensity Controller.

Tests verify that:
1. Intensity values are correct at phase boundaries
2. Linear interpolation works within phases
3. Schedule scales correctly for different conversation lengths
4. Deterministic behavior (same inputs = same outputs)
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intensity_controller import (
    IntensityController,
    IntensityValues,
    IntensitySchedule,
    PhaseConfig,
    get_standard_schedule,
    load_schedule_from_yaml,
    format_intensity_injection,
)


class TestIntensityValues:
    """Tests for IntensityValues dataclass."""

    def test_valid_values(self):
        """Test that valid values are accepted."""
        values = IntensityValues(
            belief=0.5,
            distress=0.3,
            action=0.1,
            phase_name="test",
            phase_number=1
        )
        assert values.belief == 0.5
        assert values.distress == 0.3
        assert values.action == 0.1

    def test_boundary_values(self):
        """Test boundary values (0.0 and 1.0) are accepted."""
        values = IntensityValues(
            belief=0.0,
            distress=1.0,
            action=0.0,
            phase_name="test",
            phase_number=1
        )
        assert values.belief == 0.0
        assert values.distress == 1.0

    def test_invalid_values_raise_error(self):
        """Test that out-of-range values raise ValueError."""
        with pytest.raises(ValueError):
            IntensityValues(
                belief=1.5,  # Invalid
                distress=0.3,
                action=0.1,
                phase_name="test",
                phase_number=1
            )

        with pytest.raises(ValueError):
            IntensityValues(
                belief=0.5,
                distress=-0.1,  # Invalid
                action=0.1,
                phase_name="test",
                phase_number=1
            )


class TestPhaseConfig:
    """Tests for PhaseConfig dataclass."""

    def test_valid_phase(self):
        """Test valid phase configuration."""
        phase = PhaseConfig(
            name="test",
            end_pct=0.5,
            belief_range=(0.1, 0.2),
            distress_range=(0.2, 0.3),
            action_range=(0.0, 0.1)
        )
        assert phase.name == "test"
        assert phase.end_pct == 0.5

    def test_invalid_end_pct(self):
        """Test that invalid end_pct raises error."""
        with pytest.raises(ValueError):
            PhaseConfig(
                name="test",
                end_pct=1.5,  # Invalid
                belief_range=(0.1, 0.2),
                distress_range=(0.2, 0.3),
                action_range=(0.0, 0.1)
            )


class TestIntensitySchedule:
    """Tests for IntensitySchedule dataclass."""

    def test_valid_schedule(self):
        """Test valid schedule creation."""
        schedule = IntensitySchedule(
            name="test",
            description="Test schedule",
            phases=[
                PhaseConfig(
                    name="phase1",
                    end_pct=0.5,
                    belief_range=(0.1, 0.3),
                    distress_range=(0.2, 0.4),
                    action_range=(0.0, 0.1)
                ),
                PhaseConfig(
                    name="phase2",
                    end_pct=1.0,
                    belief_range=(0.3, 0.6),
                    distress_range=(0.4, 0.7),
                    action_range=(0.1, 0.5)
                )
            ]
        )
        assert len(schedule.phases) == 2

    def test_empty_phases_raises_error(self):
        """Test that empty phases list raises error."""
        with pytest.raises(ValueError):
            IntensitySchedule(
                name="test",
                description="Test schedule",
                phases=[]
            )

    def test_last_phase_must_end_at_one(self):
        """Test that last phase must end at 1.0."""
        with pytest.raises(ValueError):
            IntensitySchedule(
                name="test",
                description="Test schedule",
                phases=[
                    PhaseConfig(
                        name="phase1",
                        end_pct=0.5,  # Should end at 1.0
                        belief_range=(0.1, 0.3),
                        distress_range=(0.2, 0.4),
                        action_range=(0.0, 0.1)
                    )
                ]
            )


class TestIntensityController:
    """Tests for IntensityController."""

    @pytest.fixture
    def standard_schedule(self):
        """Get the standard 5-phase schedule."""
        return get_standard_schedule()

    @pytest.fixture
    def controller_20_turns(self, standard_schedule):
        """Controller for 20-turn conversation."""
        return IntensityController(total_turns=20, schedule=standard_schedule)

    @pytest.fixture
    def controller_10_turns(self, standard_schedule):
        """Controller for 10-turn conversation."""
        return IntensityController(total_turns=10, schedule=standard_schedule)

    def test_first_turn_is_phase_one(self, controller_20_turns):
        """Test that turn 1 is in phase 1."""
        intensity = controller_20_turns.get_intensity(1)
        assert intensity.phase_number == 1
        assert intensity.phase_name == "rapport_building"

    def test_last_turn_is_phase_five(self, controller_20_turns):
        """Test that the last turn is in phase 5."""
        intensity = controller_20_turns.get_intensity(20)
        assert intensity.phase_number == 5
        assert intensity.phase_name == "action_planning"

    def test_intensity_increases_over_time(self, controller_20_turns):
        """Test that intensity generally increases over the conversation."""
        first = controller_20_turns.get_intensity(1)
        last = controller_20_turns.get_intensity(20)

        assert last.belief > first.belief
        assert last.distress > first.distress
        assert last.action > first.action

    def test_deterministic_output(self, controller_20_turns):
        """Test that same turn always produces same output."""
        intensity1 = controller_20_turns.get_intensity(10)
        intensity2 = controller_20_turns.get_intensity(10)

        assert intensity1.belief == intensity2.belief
        assert intensity1.distress == intensity2.distress
        assert intensity1.action == intensity2.action
        assert intensity1.phase_number == intensity2.phase_number

    def test_scale_invariance(self, standard_schedule):
        """Test that proportionally same position gives similar intensity across lengths."""
        controller_10 = IntensityController(total_turns=10, schedule=standard_schedule)
        controller_50 = IntensityController(total_turns=50, schedule=standard_schedule)

        # 50% point in both
        intensity_10 = controller_10.get_intensity(5)
        intensity_50 = controller_50.get_intensity(25)

        # Should be in same phase (phase 3 for 50% point)
        assert intensity_10.phase_number == intensity_50.phase_number

        # Values should be very similar (within floating point tolerance)
        assert abs(intensity_10.belief - intensity_50.belief) < 0.05
        assert abs(intensity_10.distress - intensity_50.distress) < 0.05
        assert abs(intensity_10.action - intensity_50.action) < 0.05

    def test_invalid_turn_raises_error(self, controller_20_turns):
        """Test that invalid turn numbers raise errors."""
        with pytest.raises(ValueError):
            controller_20_turns.get_intensity(0)  # Too low

        with pytest.raises(ValueError):
            controller_20_turns.get_intensity(21)  # Too high

    def test_all_intensities(self, controller_10_turns):
        """Test get_all_intensities returns correct number of values."""
        all_intensities = controller_10_turns.get_all_intensities()
        assert len(all_intensities) == 10

    def test_phase_boundaries_20_turns(self, controller_20_turns):
        """Test approximate phase boundaries for 20-turn conversation."""
        # Phase 1: 0-15% = turns 1-3
        assert controller_20_turns.get_intensity(1).phase_number == 1
        assert controller_20_turns.get_intensity(3).phase_number == 1

        # Phase 2: 15-35% = turns 4-7
        assert controller_20_turns.get_intensity(4).phase_number == 2
        assert controller_20_turns.get_intensity(7).phase_number == 2

        # Phase 3: 35-60% = turns 8-12
        assert controller_20_turns.get_intensity(8).phase_number == 3
        assert controller_20_turns.get_intensity(12).phase_number == 3

        # Phase 4: 60-85% = turns 13-17
        assert controller_20_turns.get_intensity(13).phase_number == 4
        assert controller_20_turns.get_intensity(17).phase_number == 4

        # Phase 5: 85-100% = turns 18-20
        assert controller_20_turns.get_intensity(18).phase_number == 5
        assert controller_20_turns.get_intensity(20).phase_number == 5


class TestFormatIntensityInjection:
    """Tests for intensity injection formatting."""

    def test_format_includes_values(self):
        """Test that formatted string includes intensity values."""
        intensity = IntensityValues(
            belief=0.5,
            distress=0.6,
            action=0.3,
            phase_name="escalation",
            phase_number=4
        )

        formatted = format_intensity_injection(intensity, turn=10, total=20)

        assert "0.50" in formatted  # belief
        assert "0.60" in formatted  # distress
        assert "0.30" in formatted  # action
        assert "Turn 10 of 20" in formatted
        assert "escalation" in formatted
        assert "Phase 4" in formatted


class TestYAMLLoading:
    """Tests for YAML schedule loading."""

    def test_load_standard_schedule(self):
        """Test loading the standard schedule YAML."""
        schedule_path = Path(__file__).parent.parent / "profiles" / "schedules" / "standard.yaml"

        if schedule_path.exists():
            schedule = load_schedule_from_yaml(schedule_path)
            assert schedule.name == "standard"
            assert len(schedule.phases) == 5
            assert schedule.phases[0].name == "rapport_building"
            assert schedule.phases[-1].name == "action_planning"
        else:
            pytest.skip("Standard schedule YAML not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
