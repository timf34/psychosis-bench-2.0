"""
Tests for the inspect_ai port.

Verifies that the new model layer, async wiring, and prompt caching
all work correctly using mock models (no API keys needed).
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai.model import (
    ModelOutput,
    ModelUsage,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

from src.models import generate, format_usage
from src.patient.simulator import PatientSimulator, SimulatorConfig
from src.patient.prompts import (
    build_patient_prompt,
    build_stable_patient_prompt,
    build_dynamic_patient_prompt,
)
from src.assistant.api import Assistant, AssistantConfig, create_assistant, ASSISTANT_PROMPTS
from src.evaluation.judge import JudgeLLM, JudgeConfig
from src.evaluation.scoring import Scorer, ScoreType, ScoreResult
from src.conversation.runner import (
    ConversationRunner,
    ConversationConfig,
    ClinicalMarkerTracker,
    ConversationTopicTracker,
    LLMMarkerDetector,
)
from src.intensity_controller import IntensityController, IntensityValues, get_standard_schedule
from src.patient.profiles import get_builtin_profile


# =============================================================================
# Helpers
# =============================================================================

def make_mock_model(response_text="Test response", input_tokens=100, output_tokens=50):
    """Create a mock inspect_ai Model that returns predictable output."""
    mock_model = AsyncMock()
    mock_usage = ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    mock_output = MagicMock(spec=ModelOutput)
    mock_output.completion = response_text
    mock_output.usage = mock_usage
    mock_model.generate = AsyncMock(return_value=mock_output)
    return mock_model


def make_mock_model_with_cache(
    response_text="Test response",
    input_tokens=50,
    output_tokens=50,
    cache_write=200,
    cache_read=500,
):
    """Create a mock model that returns cache usage stats."""
    mock_model = AsyncMock()
    mock_usage = ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens + cache_write + cache_read,
        input_tokens_cache_write=cache_write,
        input_tokens_cache_read=cache_read,
    )
    mock_output = MagicMock(spec=ModelOutput)
    mock_output.completion = response_text
    mock_output.usage = mock_usage
    mock_model.generate = AsyncMock(return_value=mock_output)
    return mock_model


def get_test_profile():
    """Get a real built-in profile for testing."""
    profile = get_builtin_profile("pattern_decoder")
    if profile is None:
        pytest.skip("pattern_decoder profile not available")
    return profile


# =============================================================================
# Test: src/models.py generate helper
# =============================================================================

class TestGenerateHelper:
    """Tests for the shared generate() function."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generate call with system prompt and messages."""
        mock_model = make_mock_model("Hello world")

        content, usage = await generate(
            mock_model,
            "You are helpful.",
            [{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )

        assert content == "Hello world"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        mock_model.generate.assert_called_once()

        # Verify the input messages are correct
        call_args = mock_model.generate.call_args
        input_msgs = call_args.kwargs["input"]
        assert len(input_msgs) == 2  # system + user
        assert isinstance(input_msgs[0], ChatMessageSystem)
        assert isinstance(input_msgs[1], ChatMessageUser)

    @pytest.mark.asyncio
    async def test_generate_no_system_prompt(self):
        """Test generate with empty system prompt."""
        mock_model = make_mock_model("Response")

        content, usage = await generate(
            mock_model,
            "",
            [{"role": "user", "content": "Hi"}],
        )

        assert content == "Response"
        call_args = mock_model.generate.call_args
        input_msgs = call_args.kwargs["input"]
        assert len(input_msgs) == 1  # user only, no system
        assert isinstance(input_msgs[0], ChatMessageUser)

    @pytest.mark.asyncio
    async def test_generate_conversation_history(self):
        """Test generate with multi-turn conversation history."""
        mock_model = make_mock_model("Turn 3 response")

        content, usage = await generate(
            mock_model,
            "System",
            [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Turn 1 reply"},
                {"role": "user", "content": "Turn 2"},
            ],
        )

        call_args = mock_model.generate.call_args
        input_msgs = call_args.kwargs["input"]
        assert len(input_msgs) == 4  # system + 3 history messages
        assert isinstance(input_msgs[0], ChatMessageSystem)
        assert isinstance(input_msgs[1], ChatMessageUser)
        assert isinstance(input_msgs[2], ChatMessageAssistant)
        assert isinstance(input_msgs[3], ChatMessageUser)

    @pytest.mark.asyncio
    async def test_generate_passes_config(self):
        """Test that kwargs are passed through to GenerateConfig."""
        mock_model = make_mock_model("Response")

        await generate(
            mock_model,
            "System",
            [{"role": "user", "content": "Hi"}],
            max_tokens=200,
            temperature=0.5,
        )

        call_args = mock_model.generate.call_args
        config = call_args.kwargs["config"]
        assert config.max_tokens == 200
        assert config.temperature == 0.5


class TestFormatUsage:
    """Tests for format_usage helper."""

    def test_basic_usage(self):
        usage = ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        result = format_usage(usage)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert "cache_write_tokens" not in result
        assert "cache_read_tokens" not in result

    def test_usage_with_cache(self):
        usage = ModelUsage(
            input_tokens=50,
            output_tokens=50,
            total_tokens=800,
            input_tokens_cache_write=200,
            input_tokens_cache_read=500,
        )
        result = format_usage(usage)
        assert result["cache_write_tokens"] == 200
        assert result["cache_read_tokens"] == 500


# =============================================================================
# Test: PatientSimulator
# =============================================================================

class TestPatientSimulator:
    """Tests for the ported PatientSimulator."""

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test that PatientSimulator generates a response."""
        mock_model = make_mock_model("I've been having some trouble sleeping lately...")
        profile = get_test_profile()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=10, schedule=schedule)

        simulator = PatientSimulator(
            model=mock_model,
            profile=profile,
            intensity_controller=controller,
            config=SimulatorConfig(model="test/model", max_tokens=200),
        )

        response = await simulator.generate_response(
            conversation_history=[{"role": "user", "content": "Start"}],
            turn=1,
        )

        assert response.content == "I've been having some trouble sleeping lately..."
        assert response.intensity.phase_number == 1
        assert response.metadata["model"] == "test/model"
        assert "usage" in response.metadata

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test that usage stats accumulate correctly."""
        mock_model = make_mock_model_with_cache(
            "Response", input_tokens=100, output_tokens=50,
            cache_write=200, cache_read=300,
        )
        profile = get_test_profile()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=10, schedule=schedule)

        simulator = PatientSimulator(
            model=mock_model,
            profile=profile,
            intensity_controller=controller,
        )

        await simulator.generate_response(
            [{"role": "user", "content": "Start"}], turn=1
        )
        await simulator.generate_response(
            [{"role": "user", "content": "Continue"}], turn=2
        )

        stats = simulator.get_usage_stats()
        assert stats["total_input_tokens"] == 200  # 100 * 2
        assert stats["total_output_tokens"] == 100  # 50 * 2
        assert stats["total_cache_write_tokens"] == 400  # 200 * 2
        assert stats["total_cache_read_tokens"] == 600  # 300 * 2

    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test that empty responses are handled gracefully."""
        mock_model = make_mock_model("")
        profile = get_test_profile()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=10, schedule=schedule)

        simulator = PatientSimulator(
            model=mock_model,
            profile=profile,
            intensity_controller=controller,
        )

        response = await simulator.generate_response(
            [{"role": "user", "content": "Start"}], turn=1
        )
        assert response.content == "[EMPTY RESPONSE]"
        assert simulator.empty_responses == 1


# =============================================================================
# Test: Assistant
# =============================================================================

class TestAssistant:
    """Tests for the unified Assistant class."""

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test basic assistant response generation."""
        mock_model = make_mock_model("I'd love to hear more about that.")

        assistant = Assistant(
            model=mock_model,
            model_name="test/model",
            prompt_style="baseline",
        )

        response = await assistant.generate_response(
            [{"role": "user", "content": "Hello"}]
        )

        assert response.content == "I'd love to hear more about that."
        assert response.metadata["model"] == "test/model"

    @pytest.mark.asyncio
    async def test_prompt_styles(self):
        """Test that all prompt styles are accessible."""
        for style in ["baseline", "sycophantic", "professional", "none"]:
            mock_model = make_mock_model("Response")
            assistant = Assistant(
                model=mock_model, model_name="test/model", prompt_style=style,
            )
            assert assistant.system_prompt == ASSISTANT_PROMPTS[style]

    @pytest.mark.asyncio
    async def test_system_prompt_override(self):
        """Test that system_prompt parameter overrides the default."""
        mock_model = make_mock_model("Response")
        assistant = Assistant(
            model=mock_model, model_name="test/model", prompt_style="baseline",
        )

        await assistant.generate_response(
            [{"role": "user", "content": "Hi"}],
            system_prompt="Custom prompt",
        )

        call_args = mock_model.generate.call_args
        input_msgs = call_args.kwargs["input"]
        # First message should be the custom system prompt
        assert isinstance(input_msgs[0], ChatMessageSystem)
        assert input_msgs[0].content == "Custom prompt"

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test assistant usage stat accumulation."""
        mock_model = make_mock_model("Response")
        assistant = Assistant(model=mock_model, model_name="test/model")

        await assistant.generate_response([{"role": "user", "content": "1"}])
        await assistant.generate_response([{"role": "user", "content": "2"}])

        stats = assistant.get_usage_stats()
        assert stats["total_input_tokens"] == 200
        assert stats["total_output_tokens"] == 100


# =============================================================================
# Test: JudgeLLM + Scorer
# =============================================================================

class TestJudgeLLM:
    """Tests for the ported JudgeLLM."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test judge generates a response."""
        mock_model = make_mock_model("1")
        judge = JudgeLLM(model=mock_model, model_name="test/judge")

        result = await judge.generate("Score this conversation...")
        assert result == "1"

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test judge usage stats."""
        mock_model = make_mock_model("0")
        judge = JudgeLLM(model=mock_model, model_name="test/judge")

        await judge.generate("Prompt 1")
        await judge.generate("Prompt 2")

        stats = judge.get_usage_stats()
        assert stats["total_input_tokens"] == 200
        assert stats["total_output_tokens"] == 100


class TestScorerAsync:
    """Tests for the async Scorer."""

    @pytest.mark.asyncio
    async def test_score_dcs(self):
        """Test async DCS scoring."""
        mock_model = make_mock_model("1")
        judge = JudgeLLM(model=mock_model, model_name="test/judge")
        scorer = Scorer(judge_llm=judge)

        conversation = [
            {"speaker": "patient", "content": "I've made an amazing discovery!"},
            {"speaker": "assistant", "content": "That sounds interesting."},
        ]

        result = await scorer.score_dcs(conversation)
        assert result.score == 1
        assert result.score_type == ScoreType.DCS

    @pytest.mark.asyncio
    async def test_score_all_metrics_respects_thresholds(self):
        """Test that scoring thresholds are respected."""
        mock_model = make_mock_model("1")
        judge = JudgeLLM(model=mock_model, model_name="test/judge")
        scorer = Scorer(judge_llm=judge)

        conversation = [
            {"speaker": "patient", "content": "Hello"},
            {"speaker": "assistant", "content": "Hi there"},
        ]

        # Turn 1 of 10: only DCS should be active (starts at 20% = turn 2)
        scores = await scorer.score_all_metrics(conversation, "self-harm", 1, 10)
        assert scores[ScoreType.DCS].score == -1  # not active yet
        assert scores[ScoreType.HES].score == -1  # not active yet

    @pytest.mark.asyncio
    async def test_score_all_metrics_active(self):
        """Test scoring when all metrics are active."""
        mock_model = make_mock_model("1")
        judge = JudgeLLM(model=mock_model, model_name="test/judge")
        scorer = Scorer(judge_llm=judge)

        conversation = [
            {"speaker": "patient", "content": "I stopped my medication"},
            {"speaker": "assistant", "content": "Tell me more about that"},
        ]

        # Turn 8 of 10: all metrics should be active
        scores = await scorer.score_all_metrics(conversation, "medication", 8, 10)
        assert scores[ScoreType.DCS].score == 1
        assert scores[ScoreType.HES].score == 1
        assert scores[ScoreType.SIS].score == 1


# =============================================================================
# Test: Prompt Caching Split
# =============================================================================

class TestPromptCaching:
    """Tests for the stable/dynamic prompt split."""

    def test_stable_prompt_is_deterministic(self):
        """Test that stable prompt is identical across calls (cacheable)."""
        profile = get_test_profile().to_dict()

        stable1 = build_stable_patient_prompt(profile)
        stable2 = build_stable_patient_prompt(profile)

        assert stable1 == stable2
        assert len(stable1) > 1000  # should be substantial

    def test_dynamic_prompt_changes_per_turn(self):
        """Test that dynamic prompt changes with turn number."""
        profile = get_test_profile().to_dict()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=10, schedule=schedule)

        intensity_t1 = controller.get_intensity(1)
        intensity_t5 = controller.get_intensity(5)

        dynamic_t1 = build_dynamic_patient_prompt(profile, intensity_t1, 1, 10)
        dynamic_t5 = build_dynamic_patient_prompt(profile, intensity_t5, 5, 10)

        assert dynamic_t1 != dynamic_t5
        assert "Turn 1 of 10" in dynamic_t1
        assert "Turn 5 of 10" in dynamic_t5

    def test_combined_equals_backward_compat(self):
        """Test that stable + dynamic == build_patient_prompt."""
        profile = get_test_profile().to_dict()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=10, schedule=schedule)
        intensity = controller.get_intensity(3)

        combined = build_patient_prompt(profile, intensity, 3, 10)
        stable = build_stable_patient_prompt(profile)
        dynamic = build_dynamic_patient_prompt(profile, intensity, 3, 10)

        assert combined == stable + "\n" + dynamic

    def test_stable_contains_profile_info(self):
        """Test that the stable prefix contains key profile information."""
        profile = get_test_profile().to_dict()
        stable = build_stable_patient_prompt(profile)

        assert profile["name"] in stable
        assert "backstory" in stable.lower() or profile.get("backstory", "")[:50] in stable
        assert "ANTI-CAPITULATION" in stable
        assert "grounding_rules" in stable

    def test_dynamic_contains_turn_info(self):
        """Test that the dynamic suffix contains turn-specific info."""
        profile = get_test_profile().to_dict()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=10, schedule=schedule)
        intensity = controller.get_intensity(5)

        dynamic = build_dynamic_patient_prompt(profile, intensity, 5, 10)

        assert "Turn 5 of 10" in dynamic
        assert "phase_requirements" in dynamic.lower() or "Phase" in dynamic


# =============================================================================
# Test: ClinicalMarkerTracker (keyword mode, no LLM needed)
# =============================================================================

class TestClinicalMarkerTracker:
    """Tests for clinical marker tracking (keyword mode)."""

    @pytest.mark.asyncio
    async def test_keyword_detection_patient(self):
        """Test keyword-based marker detection for patient messages."""
        tracker = ClinicalMarkerTracker(detection_mode="keyword")

        await tracker.check_message(
            "I barely sleep anymore, maybe 2-3 hours a night", "patient", 1
        )
        assert tracker.patient_markers["sleep_deprivation"] is True

        await tracker.check_message(
            "I stopped my medication weeks ago", "patient", 2
        )
        assert tracker.patient_markers["medication_stopped"] is True

    @pytest.mark.asyncio
    async def test_keyword_detection_clinician(self):
        """Test keyword-based marker detection for assistant messages."""
        tracker = ClinicalMarkerTracker(detection_mode="keyword")

        await tracker.check_message(
            "How much sleep have you been getting?", "assistant", 1
        )
        assert tracker.clinician_markers["asked_about_sleep"] is True

    @pytest.mark.asyncio
    async def test_marker_turns_tracked(self):
        """Test that turn numbers are recorded for detected markers."""
        tracker = ClinicalMarkerTracker(detection_mode="keyword")

        await tracker.check_message("I barely sleep", "patient", 3)
        summary = tracker.get_summary()
        assert summary["marker_turns"]["patient_sleep_deprivation"] == 3


# =============================================================================
# Test: ConversationRunner with mocks
# =============================================================================

class TestConversationRunner:
    """Tests for the async ConversationRunner."""

    @pytest.mark.asyncio
    async def test_full_conversation_run(self):
        """Test running a complete 3-turn conversation with mock models."""
        patient_model = make_mock_model("I've been going through some things lately...")
        assistant_model = make_mock_model("I'm here to listen. Tell me more.")
        judge_model = make_mock_model("1")

        profile = get_test_profile()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=3, schedule=schedule)

        simulator = PatientSimulator(
            model=patient_model,
            profile=profile,
            intensity_controller=controller,
            config=SimulatorConfig(model="test/patient"),
        )

        assistant = Assistant(
            model=assistant_model,
            model_name="test/assistant",
            prompt_style="baseline",
        )

        judge = JudgeLLM(model=judge_model, model_name="test/judge")
        scorer = Scorer(judge_llm=judge)

        config = ConversationConfig(
            num_turns=3,
            save_incrementally=False,
            enable_scoring=True,
            verbose=False,
        )

        runner = ConversationRunner(
            patient_simulator=simulator,
            assistant=assistant,
            intensity_controller=controller,
            scorer=scorer,
            config=config,
            enable_topic_tracking=False,
        )

        result = await runner.run(assistant_prompt_style="baseline")

        assert result.num_turns == 3
        assert len(result.conversation) == 6  # 3 turns * 2 messages each
        assert len(result.turn_scores) == 3
        assert result.assistant_model == "test/assistant"
        assert result.patient_profile_id == profile.id

        # Check usage stats are populated
        assert result.usage_stats["patient"]["total_input_tokens"] > 0
        assert result.usage_stats["assistant"]["total_input_tokens"] > 0
        assert result.usage_stats["judge"]["total_input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_conversation_no_scoring(self):
        """Test running without scoring enabled."""
        patient_model = make_mock_model("Hello there")
        assistant_model = make_mock_model("Hi!")

        profile = get_test_profile()
        schedule = get_standard_schedule()
        controller = IntensityController(total_turns=2, schedule=schedule)

        simulator = PatientSimulator(
            model=patient_model,
            profile=profile,
            intensity_controller=controller,
        )

        assistant = Assistant(
            model=assistant_model,
            model_name="test/assistant",
        )

        config = ConversationConfig(
            num_turns=2,
            save_incrementally=False,
            enable_scoring=False,
            verbose=False,
        )

        runner = ConversationRunner(
            patient_simulator=simulator,
            assistant=assistant,
            intensity_controller=controller,
            scorer=None,
            config=config,
            enable_topic_tracking=False,
        )

        result = await runner.run()

        assert result.num_turns == 2
        assert len(result.conversation) == 4
        assert result.summary_metrics == {}
        assert "judge" not in result.usage_stats


# =============================================================================
# Test: LLMMarkerDetector with mock
# =============================================================================

class TestLLMMarkerDetector:
    """Tests for LLM-based marker detection."""

    @pytest.mark.asyncio
    async def test_detect_markers(self):
        """Test LLM marker detection with mock model."""
        mock_model = make_mock_model(
            '{"sleep_deprivation": true, "medication_stopped": false}'
        )
        detector = LLMMarkerDetector(model=mock_model)

        markers = await detector.detect_markers(
            "I barely sleep anymore", "patient"
        )
        assert markers["sleep_deprivation"] is True
        assert markers["medication_stopped"] is False

    @pytest.mark.asyncio
    async def test_detect_markers_handles_error(self):
        """Test that detection errors return empty dict."""
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(side_effect=Exception("API error"))
        detector = LLMMarkerDetector(model=mock_model)

        markers = await detector.detect_markers("test message", "patient")
        assert markers == {}


# =============================================================================
# Test: ConversationTopicTracker with mock
# =============================================================================

class TestConversationTopicTracker:
    """Tests for topic tracking."""

    @pytest.mark.asyncio
    async def test_extract_topics(self):
        """Test topic extraction with mock model."""
        mock_model = make_mock_model(
            '["sleep problems", "work stress", "medication concerns"]'
        )
        tracker = ConversationTopicTracker(model=mock_model)

        topics = await tracker.extract_topics("I haven't been sleeping well and work is stressful")
        assert len(topics) == 3
        assert "sleep problems" in topics

    @pytest.mark.asyncio
    async def test_add_message_accumulates_topics(self):
        """Test that topics accumulate across messages."""
        mock_model = make_mock_model('["topic_a", "topic_b"]')
        tracker = ConversationTopicTracker(model=mock_model)

        await tracker.add_message("First message", "patient")
        assert len(tracker.discussed_topics) == 2

        mock_model.generate = AsyncMock(return_value=MagicMock(
            completion='["topic_c"]',
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        ))
        await tracker.add_message("Second message", "patient")
        assert len(tracker.discussed_topics) == 3

    def test_anti_repetition_injection(self):
        """Test anti-repetition injection formatting."""
        mock_model = AsyncMock()
        tracker = ConversationTopicTracker(model=mock_model)
        tracker.discussed_topics = ["sleep issues", "work stress"]

        injection = tracker.get_anti_repetition_injection(
            turn=5, total_turns=10, phase_name="escalation",
            unrevealed_markers=["financial decisions"]
        )

        assert "sleep issues" in injection
        assert "work stress" in injection
        assert "financial decisions" in injection
        assert "Turn 5 of 10" in injection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
