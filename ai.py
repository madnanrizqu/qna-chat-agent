from abc import ABC, abstractmethod
from openai import OpenAI

from config import settings


class AIClient(ABC):
    """Abstract base class for AI client implementations.

    Defines the interface that all AI client providers must implement.
    """

    @abstractmethod
    def create_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def create_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batch.

        More efficient than calling create_embedding multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        pass


class OpenAIClient(AIClient):
    """OpenAI-compatible AI client implementation.

    Supports OpenAI and OpenAI-compatible APIs (e.g., Gemini via OpenAI base URL).
    Uses lazy initialization pattern for efficient resource management.
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            base_url: Optional base URL for OpenAI-compatible endpoints
        """
        self._api_key = api_key
        self._base_url = base_url
        self._client: OpenAI | None = None

    def create_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Uses the configured embedding model and dimensions from settings.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
        )
        return response.data[0].embedding

    def create_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batch.

        Uses the configured embedding model and dimensions from settings.
        More efficient than calling create_embedding multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
            dimensions=settings.embedding_dimensions,
        )
        return [item.embedding for item in response.data]

    def _get_client(self) -> OpenAI:
        """Get configured OpenAI client (lazy singleton pattern).

        Creates and caches a single client instance on first access.
        The OpenAI client is thread-safe and maintains its own connection pool.

        Returns:
            OpenAI: Configured OpenAI client instance
        """
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client


ai_client: AIClient = OpenAIClient(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)
