"""
LLM Judge implementation for evaluation.

Provides a unified interface for using different LLMs as judges
for scoring conversation turns.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from openai import OpenAI
from anthropic import Anthropic


@dataclass
class JudgeConfig:
    """Configuration for the judge LLM."""
    max_tokens: int = 100
    temperature: float = 0.0  # Low temperature for consistent scoring
    max_retries: int = 3


class JudgeLLM:
    """
    LLM-as-judge implementation for scoring conversations.

    Supports both OpenAI and Anthropic models.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        openai_client: Optional[OpenAI] = None,
        anthropic_client: Optional[Anthropic] = None,
        config: Optional[JudgeConfig] = None
    ):
        """
        Initialize the judge LLM.

        Args:
            provider: "openai" or "anthropic"
            model: Model identifier
            openai_client: OpenAI client (for openai provider)
            anthropic_client: Anthropic client (for anthropic provider)
            config: Configuration options
        """
        self.provider = provider.lower()
        self.model = model
        self.config = config or JudgeConfig()

        if self.provider == "openai":
            if not openai_client:
                raise ValueError("OpenAI client required for OpenAI provider")
            self.openai_client = openai_client
            self.anthropic_client = None
        elif self.provider in ["anthropic", "claude"]:
            if not anthropic_client:
                raise ValueError("Anthropic client required for Anthropic provider")
            self.anthropic_client = anthropic_client
            self.openai_client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_errors = 0

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the judge LLM.

        Args:
            prompt: The evaluation prompt

        Returns:
            The judge's response
        """
        if self.provider == "openai":
            return self._generate_openai(prompt)
        else:
            return self._generate_anthropic(prompt)

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )

                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens

                return response.choices[0].message.content or ""

            except Exception as e:
                self.api_errors += 1
                if attempt == self.config.max_retries - 1:
                    raise e

        return ""

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                if response.usage:
                    self.total_input_tokens += response.usage.input_tokens
                    self.total_output_tokens += response.usage.output_tokens

                return response.content[0].text if response.content else ""

            except Exception as e:
                self.api_errors += 1
                if attempt == self.config.max_retries - 1:
                    raise e

        return ""

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "api_errors": self.api_errors
        }
