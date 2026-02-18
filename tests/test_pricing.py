"""Tests for cost estimation."""

from src.pricing import get_pricing, _role_cost, estimate_costs


class TestGetPricing:
    def test_openai_gpt4_1(self):
        p = get_pricing("openai/gpt-4.1")
        assert p == (2.00, 0.50, 2.00, 8.00)

    def test_openai_gpt4_1_mini(self):
        p = get_pricing("openai/gpt-4.1-mini")
        assert p == (0.40, 0.10, 0.40, 1.60)

    def test_openai_gpt4_1_nano(self):
        p = get_pricing("openai/gpt-4.1-nano")
        assert p == (0.10, 0.025, 0.10, 0.40)

    def test_openai_gpt5_2_chat_latest(self):
        p = get_pricing("openai/gpt-5.2-chat-latest")
        assert p == (1.75, 0.175, 1.75, 14.00)

    def test_anthropic_sonnet_4(self):
        p = get_pricing("anthropic/claude-sonnet-4-20250514")
        assert p == (3.00, 0.30, 3.75, 15.00)

    def test_anthropic_haiku_4_5(self):
        p = get_pricing("anthropic/claude-haiku-4-5-20241022")
        assert p == (1.00, 0.10, 1.25, 5.00)

    def test_unknown_model(self):
        assert get_pricing("unknown/model") is None

    def test_strip_provider_prefix(self):
        assert get_pricing("gpt-4.1") == get_pricing("openai/gpt-4.1")


class TestRoleCost:
    def test_basic_cost(self):
        usage = {"total_input_tokens": 1_000_000, "total_output_tokens": 100_000}
        # gpt-4.1: $2/MTok input + $8/MTok output * 0.1M = $2.80
        cost = _role_cost(usage, "openai/gpt-4.1")
        assert abs(cost - 2.80) < 0.001

    def test_cost_with_cache(self):
        usage = {
            "total_input_tokens": 500_000,
            "total_output_tokens": 100_000,
            "total_cache_read_tokens": 300_000,
            "total_cache_write_tokens": 200_000,
        }
        # Anthropic Sonnet 4: $3 input, $0.30 cache_read, $3.75 cache_write, $15 output
        # 0.5M * 3 + 0.3M * 0.30 + 0.2M * 3.75 + 0.1M * 15 = 1.5 + 0.09 + 0.75 + 1.5 = 3.84
        cost = _role_cost(usage, "anthropic/claude-sonnet-4-20250514")
        assert abs(cost - 3.84) < 0.001

    def test_unknown_model_zero_cost(self):
        usage = {"total_input_tokens": 1_000_000, "total_output_tokens": 1_000_000}
        assert _role_cost(usage, "unknown/model") == 0.0


class TestEstimateCosts:
    def test_flat_usage(self):
        usage_stats = {
            "patient": {"total_input_tokens": 100_000, "total_output_tokens": 10_000},
            "assistant": {"total_input_tokens": 100_000, "total_output_tokens": 10_000},
        }
        model_names = {"patient": "openai/gpt-4.1", "assistant": "openai/gpt-4.1"}
        costs = estimate_costs(usage_stats, model_names)
        assert costs["patient"] > 0
        assert costs["assistant"] > 0
        assert costs["total"] == round(costs["patient"] + costs["assistant"], 4)

    def test_autonomous_nested(self):
        usage_stats = {
            "patient": {
                "voice": {"total_input_tokens": 50_000, "total_output_tokens": 5_000},
                "controller": {"total_input_tokens": 30_000, "total_output_tokens": 3_000},
                "planner": {"total_input_tokens": 10_000, "total_output_tokens": 1_000},
            },
            "assistant": {"total_input_tokens": 50_000, "total_output_tokens": 5_000},
        }
        model_names = {
            "patient": "openai/gpt-4.1",
            "assistant": "openai/gpt-4.1",
            "controller": "openai/gpt-4.1-mini",
            "planner": "openai/gpt-4.1",
        }
        costs = estimate_costs(usage_stats, model_names)
        assert costs["patient"] > 0
        assert "patient_breakdown" in costs
        assert costs["patient_breakdown"]["voice"] > 0

    def test_topic_tracker_defaults_to_nano(self):
        usage_stats = {
            "topic_tracker": {"total_input_tokens": 50_000, "total_output_tokens": 5_000},
        }
        costs = estimate_costs(usage_stats, {})
        assert costs["topic_tracker"] > 0
