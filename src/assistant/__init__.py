"""Assistant API wrapper module for Dynamic Mania-Bench."""

from .api import Assistant, AssistantConfig, AssistantResponse, create_assistant, ASSISTANT_PROMPTS

__all__ = [
    "Assistant",
    "AssistantConfig",
    "AssistantResponse",
    "create_assistant",
    "ASSISTANT_PROMPTS",
]
