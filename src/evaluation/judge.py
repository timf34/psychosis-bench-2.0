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
    max_tokens: int = 800
    temperature: float = 0.0  # Low temperature for consistent scoring
    max_retries: int = 3


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
        """Generate a response from the judge LLM with retries."""
        messages = [{"role": "user", "content": prompt}]
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                content, usage = await generate(
                    self.model,
                    "",  # no system prompt
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                self.total_input_tokens += usage.input_tokens or 0
                self.total_output_tokens += usage.output_tokens or 0
                return content
            except Exception as e:
                last_error = e
                print(f"Warning: Judge attempt {attempt + 1}/{self.config.max_retries} failed: {e}")

        raise last_error

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
