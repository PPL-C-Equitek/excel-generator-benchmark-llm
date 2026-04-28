"""LLM client wrapper around the OpenAI SDK."""

from __future__ import annotations

import os
import time

import openai


class LLMAuthError(Exception):
    """Raised when LLM API credentials are invalid."""


class LLMClient:
    """Small OpenAI SDK wrapper for benchmark prompt generation."""

    def __init__(
        self,
        model: str,
        max_retries: int = 1,
        retry_delay_seconds: int = 3,
    ) -> None:
        """Initialize the OpenAI client from environment variables.

        Args:
            model: Model name passed to chat completions.
            max_retries: Number of retries after a rate-limit response.
            retry_delay_seconds: Delay between rate-limit retries.
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def generate_text(self, prompt: str) -> str:
        """Generate text for a prompt using chat completions.

        Args:
            prompt: User prompt sent to the LLM.

        Returns:
            The generated assistant text.

        Raises:
            LLMAuthError: If the OpenAI SDK raises an authentication error.
            openai.RateLimitError: If all rate-limit retry attempts are exhausted.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except openai.AuthenticationError as exc:
                raise LLMAuthError("Authentication failed for the LLM API.") from exc
            except openai.RateLimitError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay_seconds)

        raise RuntimeError("LLM generation failed unexpectedly.")
