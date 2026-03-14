from abc import ABC, abstractmethod
from google import genai
from google.genai import types

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


class GeminiAIClient(AIClient):
    """Native Google Gemini AI client implementation.

    Uses the official Google Generative AI SDK for embeddings.
    Implements lazy initialization pattern for efficient resource management.
    """

    def __init__(self, api_key: str):
        """Initialize Gemini client.

        Args:
            api_key: Google API key for authentication
        """
        self._api_key = api_key
        self._client: genai.Client | None = None

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

        config = types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions
        )

        result = client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
            config=config,
        )

        return result.embeddings[0].values

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

        config = types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions
        )

        result = client.models.embed_content(
            model=settings.embedding_model,
            contents=texts,
            config=config,
        )

        return [emb.values for emb in result.embeddings]

    def _get_client(self) -> genai.Client:
        """Get configured Gemini client (lazy singleton pattern).

        Creates and caches a single client instance on first access.
        The Gemini client is thread-safe and maintains its own connection pool.

        Returns:
            genai.Client: Configured Gemini client instance
        """
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client


ai_client: AIClient = GeminiAIClient(
    api_key=settings.google_api_key,
)
