"""Generic LLM gateway client used by benchmark scripts."""

from __future__ import annotations

import os
import time
from typing import Any

import openai


API_KEY_ENV = "API_KEY"
BASE_URL_ENV = "BASE_URL"
USER_ROLE = "user"


class LLMAuthError(Exception):
    """Raised when LLM API credentials are invalid or rejected."""


class LLMClient:
    """Client for benchmark prompt generation through an LLM gateway.

    The client reads API credentials from environment variables and sends
    prompts through the gateway's chat completions endpoint. It translates
    authentication failures into ``LLMAuthError`` so callers can stop a
    benchmark run safely.
    """

    DEFAULT_MAX_RETRIES = 1
    DEFAULT_RETRY_DELAY_SECONDS = 3

    def __init__(
        self,
        model: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: int = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        """Initialize the LLM gateway client from environment variables.

        Args:
            model: Model name passed to chat completions.
            max_retries: Number of retries after a rate-limit response.
            retry_delay_seconds: Delay between rate-limit retries.

        Raises:
            ValueError: If ``max_retries`` is negative.
            openai.OpenAIError: If the underlying SDK fails during client setup.
        """
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._client = openai.OpenAI(
            api_key=os.getenv(API_KEY_ENV),
            base_url=os.getenv(BASE_URL_ENV),
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
        attempt = 0
        while True:
            try:
                response = self._create_completion(prompt)
                return self._response_text(response)
            except openai.AuthenticationError as exc:
                raise LLMAuthError("Authentication failed for the LLM API.") from exc
            except openai.RateLimitError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay_seconds)
                attempt += 1

    def _create_completion(self, prompt: str) -> Any:
        """Send a prompt to the gateway chat completions endpoint.

        Args:
            prompt: User prompt sent to the LLM.

        Returns:
            Raw OpenAI SDK chat completion response.

        Raises:
            openai.AuthenticationError: If credentials are rejected.
            openai.RateLimitError: If the provider rate-limits the request.
            openai.OpenAIError: For other SDK/API failures.
        """
        return self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": USER_ROLE, "content": prompt}],
        )

    def _response_text(self, response: Any) -> str:
        """Extract generated assistant text from a completion response.

        Args:
            response: Raw OpenAI SDK chat completion response.

        Returns:
            Text content from the first completion choice.
        """
        return str(response.choices[0].message.content)
