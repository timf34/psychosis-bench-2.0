"""
LLM Judge implementation for evaluation.

Provides a unified interface for using LLMs as judges
for scoring conversation turns, via inspect_ai.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from inspect_ai.model import Model, ModelUsage

from ..models import generate, format_usage


@dataclass
class JudgeConfig:
    """Configuration for the judge LLM."""
    max_tokens: int = 100
    temperature: float = 0.0  # Low temperature for consistent scoring


class JudgeLLM:
    """
    LLM-as-judge implementation for scoring conversations.

    Uses inspect_ai's Model for provider-agnostic inference.
    """

    def __init__(
        self,
        model: Model,
        model_name: str,
        config: Optional[JudgeConfig] = None
    ):
        """
        Initialize the judge LLM.

        Args:
            model: inspect_ai Model instance
            model_name: Model name string for logging
            config: Configuration options
        """
        self.model = model
        self.model_name = model_name
        self.config = config or JudgeConfig()

        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def generate(self, prompt: str) -> str:
        """
        Generate a response from the judge LLM.

        Args:
            prompt: The evaluation prompt

        Returns:
            The judge's response
        """
        # Judge prompts are sent as user messages with no system prompt
        messages = [{"role": "user", "content": prompt}]

        content, usage = await generate(
            self.model,
            "",  # no system prompt
            messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Update statistics
        self.total_input_tokens += usage.input_tokens or 0
        self.total_output_tokens += usage.output_tokens or 0

        return content

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
