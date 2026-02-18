"""Cost estimation from token usage and model pricing."""

from typing import Dict, Any, Optional, Tuple

# Pricing per 1M tokens: (input, cache_read, cache_write, output)
# OpenAI: cache_write charged at input rate, cache_read at discounted rate
# Anthropic: cache_write at 1.25x input, cache_read at 0.1x input
_PRICING: list[tuple[str, float, float, float, float]] = [
    # OpenAI — most specific prefixes first
    ("gpt-5.2-pro", 21.00, 0.0, 21.00, 168.00),
    ("gpt-5.2", 1.75, 0.175, 1.75, 14.00),
    ("gpt-5.1-codex-mini", 0.25, 0.025, 0.25, 2.00),
    ("gpt-5.1", 1.25, 0.125, 1.25, 10.00),
    ("gpt-5-pro", 15.00, 0.0, 15.00, 120.00),
    ("gpt-5-mini", 0.25, 0.025, 0.25, 2.00),
    ("gpt-5-nano", 0.05, 0.005, 0.05, 0.40),
    ("gpt-5", 1.25, 0.125, 1.25, 10.00),
    ("gpt-4.1-nano", 0.10, 0.025, 0.10, 0.40),
    ("gpt-4.1-mini", 0.40, 0.10, 0.40, 1.60),
    ("gpt-4.1", 2.00, 0.50, 2.00, 8.00),
    ("gpt-4o-mini", 0.15, 0.075, 0.15, 0.60),
    ("gpt-4o", 2.50, 1.25, 2.50, 10.00),
    ("o4-mini", 1.10, 0.275, 1.10, 4.40),
    ("o3-mini", 1.10, 0.55, 1.10, 4.40),
    ("o3", 2.00, 0.50, 2.00, 8.00),
    # Google Gemini — implicit caching (automatic, like OpenAI)
    # cache_read = 0.1x input, cache_write = input (no surcharge for implicit)
    ("gemini-3-pro", 2.00, 0.20, 2.00, 12.00),
    ("gemini-3-flash", 0.50, 0.05, 0.50, 3.00),
    ("gemini-2.5-pro", 1.25, 0.125, 1.25, 10.00),
    ("gemini-2.5-flash-lite", 0.10, 0.01, 0.10, 0.40),
    ("gemini-2.5-flash", 0.30, 0.03, 0.30, 2.50),
    ("gemini-2.0-flash-lite", 0.075, 0.0, 0.075, 0.30),
    ("gemini-2.0-flash", 0.10, 0.025, 0.10, 0.40),
    # Anthropic
    ("claude-opus-4-6", 5.00, 0.50, 6.25, 25.00),
    ("claude-opus-4-5", 5.00, 0.50, 6.25, 25.00),
    ("claude-opus-4-1", 15.00, 1.50, 18.75, 75.00),
    ("claude-opus-4", 15.00, 1.50, 18.75, 75.00),
    ("claude-sonnet-4-6", 3.00, 0.30, 3.75, 15.00),
    ("claude-sonnet-4-5", 3.00, 0.30, 3.75, 15.00),
    ("claude-sonnet-4", 3.00, 0.30, 3.75, 15.00),
    ("claude-sonnet-3", 3.00, 0.30, 3.75, 15.00),
    ("claude-haiku-4-5", 1.00, 0.10, 1.25, 5.00),
    ("claude-haiku-3-5", 0.80, 0.08, 1.00, 4.00),
    ("claude-haiku-3", 0.25, 0.03, 0.30, 1.25),
]


def get_pricing(model_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Look up pricing for a model. Returns (input, cache_read, cache_write, output) per MTok."""
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]
    for prefix, inp, cr, cw, out in _PRICING:
        if model_name.startswith(prefix):
            return (inp, cr, cw, out)
    return None


def _role_cost(usage: dict, model_name: str) -> float:
    """Calculate cost for a single usage dict + model name."""
    pricing = get_pricing(model_name)
    if not pricing:
        return 0.0
    inp_price, cr_price, cw_price, out_price = pricing
    inp = usage.get("total_input_tokens", 0)
    out = usage.get("total_output_tokens", 0)
    cr = usage.get("total_cache_read_tokens", 0)
    cw = usage.get("total_cache_write_tokens", 0)
    return (inp * inp_price + cr * cr_price + cw * cw_price + out * out_price) / 1_000_000


def estimate_costs(usage_stats: Dict[str, Any], model_names: Dict[str, str]) -> Dict[str, Any]:
    """
    Estimate USD costs from usage stats and model names.

    Args:
        usage_stats: per-role usage dicts from ConversationResult
        model_names: role -> model name mapping

    Returns:
        Dict with per-role costs and "total".
    """
    costs: Dict[str, Any] = {}

    for role, usage in usage_stats.items():
        if role == "patient" and "voice" in usage:
            # Autonomous mode: nested voice/controller/planner
            v = _role_cost(usage.get("voice", {}), model_names.get("patient", ""))
            c = _role_cost(usage.get("controller", {}), model_names.get("controller", ""))
            p = _role_cost(usage.get("planner", {}), model_names.get("planner", ""))
            costs[role] = round(v + c + p, 4)
            costs["patient_breakdown"] = {
                "voice": round(v, 4), "controller": round(c, 4), "planner": round(p, 4),
            }
        elif role in model_names and model_names[role]:
            costs[role] = round(_role_cost(usage, model_names[role]), 4)
        elif role in ("topic_tracker", "marker_detector"):
            costs[role] = round(_role_cost(usage, "openai/gpt-4.1-nano"), 4)

    costs["total"] = round(
        sum(v for k, v in costs.items() if k != "patient_breakdown" and isinstance(v, (int, float))),
        4,
    )
    return costs
