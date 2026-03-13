"""AI client configuration and initialization."""

from openai import OpenAI

from config import settings


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


# Singleton instance
ai_client = AIClient()
