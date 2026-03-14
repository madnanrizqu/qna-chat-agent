"""AI client configuration and initialization."""

from typing import Type, TypeVar
from pydantic import BaseModel
from openai import OpenAI

from config import settings

ResponseT = TypeVar("T", bound=BaseModel)


class AIClient:
    """AI client manager with lazy initialization."""

    def __init__(self):
        self._client: OpenAI | None = None

    def get_client(self) -> OpenAI:
        """Get configured OpenAI client (lazy singleton pattern).

        Creates and caches a single client instance on first access.
        The OpenAI client is thread-safe and maintains its own connection pool.

        Returns:
            OpenAI: Configured OpenAI client instance
        """
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.openai_api_key, base_url=settings.openai_base_url
            )
        return self._client

    def reset(self) -> None:
        """Reset the client instance (useful for testing or configuration changes)."""
        self._client = None

    def chat_structured(
        self,
        model: str,
        messages: list[dict[str, str]],
        response_format: Type[ResponseT],
        **kwargs,
    ) -> ResponseT:
        """Chat completion with structured output using beta API.

        Args:
            model: Model identifier
            messages: Conversation messages
            response_format: Pydantic model class for response structure
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Parsed instance of response_format model
        """
        client = self.get_client()

        completion = client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=response_format, **kwargs
        )

        parsed_response = completion.choices[0].message.parsed

        if parsed_response is None:
            raise ValueError("Failed to parse structured response from AI")

        return parsed_response


ai_client = AIClient()
