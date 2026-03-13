"""Vector database abstraction for embeddings and similarity search."""

from abc import ABC, abstractmethod
from typing import Any

from supabase import create_client, Client

from ai import ai_client, AIClient
from config import settings
from models import SearchResult


class EmbeddingService(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    @abstractmethod
    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...


class VectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def get_client(self) -> Any:
        """Get the underlying database client.

        Returns:
            Database client instance
        """
        ...

    @abstractmethod
    def store_document(self, content: str) -> str:
        """Store a document with its embedding.

        Args:
            content: Document text content

        Returns:
            Document ID
        """
        ...

    @abstractmethod
    def store_documents(self, documents: list[str]) -> list[str]:
        """Store multiple documents with their embeddings in a batch.

        Args:
            documents: List of document content strings

        Returns:
            List of document IDs
        """
        ...

    @abstractmethod
    def search_similar(
        self, query: str, limit: int | None = None, threshold: float | None = None
    ) -> list[SearchResult]:
        """Search for similar documents using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results (defaults to settings.max_search_results)
            threshold: Minimum similarity threshold (defaults to settings.similarity_threshold)

        Returns:
            List of SearchResult objects ranked by similarity
        """
        ...


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using OpenAI-compatible API (currently Gemini)."""

    def __init__(self, client: AIClient):
        """Initialize with an AI client instance.

        Args:
            client: AIClient instance for generating embeddings
        """
        self._client = client

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = self._client.get_client().embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self._client.get_client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
            dimensions=settings.embedding_dimensions,
        )
        return [item.embedding for item in response.data]


class SupabaseVectorStore(VectorStore):
    """Vector store using Supabase pgvector."""

    def __init__(self, embedding_service: EmbeddingService):
        """Initialize with an embedding service.

        Args:
            embedding_service: EmbeddingService instance for generating embeddings
        """
        self._client: Client | None = None
        self._embedding_service = embedding_service

    def get_client(self) -> Client:
        """Get Supabase client (lazy initialization).

        Returns:
            Supabase client instance
        """
        if self._client is None:
            self._client = create_client(
                settings.vector_database_url, settings.vector_database_api_key
            )
        return self._client

    def store_document(self, content: str) -> str:
        """Store a document with its embedding.

        Args:
            content: Document text content

        Returns:
            Document ID (UUID as string)
        """
        embedding = self._embedding_service.generate_embedding(content)
        client = self.get_client()
        result = (
            client.table("documents")
            .insert(
                {
                    "content": content,
                    "embedding": embedding,
                }
            )
            .execute()
        )
        return result.data[0]["id"]

    def store_documents(self, documents: list[str]) -> list[str]:
        """Store multiple documents with their embeddings in a batch.

        Args:
            documents: List of document content strings

        Returns:
            List of document IDs (UUIDs as strings)
        """
        embeddings = self._embedding_service.generate_embeddings_batch(documents)
        rows = [
            {
                "content": content,
                "embedding": emb,
            }
            for content, emb in zip(documents, embeddings)
        ]
        client = self.get_client()
        result = client.table("documents").insert(rows).execute()
        return [row["id"] for row in result.data]

    def search_similar(
        self, query: str, limit: int | None = None, threshold: float | None = None
    ) -> list[SearchResult]:
        """Search for similar documents using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results (defaults to settings.max_search_results)
            threshold: Minimum similarity threshold (defaults to settings.similarity_threshold)

        Returns:
            List of SearchResult objects ranked by similarity
        """
        query_embedding = self._embedding_service.generate_embedding(query)
        client = self.get_client()
        result = client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold or settings.similarity_threshold,
                "match_count": limit or settings.max_search_results,
            },
        ).execute()
        return [SearchResult(**row) for row in result.data]


embedding_service = OpenAIEmbeddingService(ai_client)
vector_store = SupabaseVectorStore(embedding_service)
