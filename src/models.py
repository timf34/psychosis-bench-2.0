"""
Shared model helper for psychosis-bench-2.0.

Thin wrapper around inspect_ai's model layer, providing a simple
generate() function that takes system prompt + conversation history dicts
and returns (content, usage).
"""

from typing import Optional

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    ModelUsage,
    get_model,
)


async def generate(
    model: Model,
    system: str,
    messages: list[dict],
    **kwargs,
) -> tuple[str, ModelUsage]:
    """
    Unified generate: takes system prompt + conversation history dicts,
    returns (content, usage).

    Args:
        model: inspect_ai Model instance
        system: System prompt string (empty string for no system prompt)
        messages: List of {"role": "user"|"assistant", "content": str}
        **kwargs: Passed to GenerateConfig (max_tokens, temperature, etc.)

    Returns:
        Tuple of (response text, ModelUsage with token counts)
    """
    input_msgs = []
    if system:
        input_msgs.append(ChatMessageSystem(content=system))
    for msg in messages:
        if msg["role"] == "user":
            input_msgs.append(ChatMessageUser(content=msg["content"]))
        elif msg["role"] == "assistant":
            input_msgs.append(ChatMessageAssistant(content=msg["content"]))

    output: ModelOutput = await model.generate(
        input=input_msgs,
        config=GenerateConfig(**kwargs),
    )

    content = output.completion or ""
    usage = output.usage or ModelUsage()

    return content, usage


def format_usage(usage: ModelUsage) -> dict:
    """Convert ModelUsage to a plain dict for JSON serialization."""
    result = {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }
    if usage.input_tokens_cache_write:
        result["cache_write_tokens"] = usage.input_tokens_cache_write
    if usage.input_tokens_cache_read:
        result["cache_read_tokens"] = usage.input_tokens_cache_read
    return result
