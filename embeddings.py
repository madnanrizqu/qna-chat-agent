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


class TextSplitter(ABC):
    """Abstract interface for text chunking."""

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
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
    def store_document(self, content: str, category: str | None = None) -> str:
        """Store a document with its embedding.

        Args:
            content: Document text content
            category: Optional category label for the document

        Returns:
            Document ID
        """
        ...

    @abstractmethod
    def store_documents(
        self, documents: list[str], categories: list[str | None] | None = None
    ) -> list[str]:
        """Store multiple documents with their embeddings in a batch.

        Args:
            documents: List of document content strings
            categories: Optional list of category labels (must match length of documents)

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


class RecursiveTextSplitter(TextSplitter):
    """Text splitter using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        """Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        return self._splitter.split_text(text)


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

    def store_document(self, content: str, category: str | None = None) -> str:
        """Store a document with its embedding.

        Args:
            content: Document text content
            category: Optional category label for the document

        Returns:
            Document ID (UUID as string)
        """
        embedding = self._embedding_service.generate_embedding(content)
        client = self.get_client()
        row = {
            "content": content,
            "embedding": embedding,
        }
        if category is not None:
            row["category"] = category

        table_name = "document_chunks" if settings.use_chunked_storage else "documents"
        result = client.table(table_name).insert(row).execute()
        return result.data[0]["id"]

    def store_documents(
        self, documents: list[str], categories: list[str | None] | None = None
    ) -> list[str]:
        """Store multiple documents with their embeddings in a batch.

        Args:
            documents: List of document content strings
            categories: Optional list of category labels (must match length of documents)

        Returns:
            List of document IDs (UUIDs as strings)
        """
        if categories is not None and len(categories) != len(documents):
            raise ValueError(
                f"Length mismatch: {len(documents)} documents but {len(categories)} categories"
            )

        embeddings = self._embedding_service.generate_embeddings_batch(documents)
        rows = []
        for i, (content, emb) in enumerate(zip(documents, embeddings)):
            row = {
                "content": content,
                "embedding": emb,
            }
            if categories is not None and categories[i] is not None:
                row["category"] = categories[i]
            rows.append(row)

        client = self.get_client()
        table_name = "document_chunks" if settings.use_chunked_storage else "documents"
        result = client.table(table_name).insert(rows).execute()
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
        print(f"Generating embedding for query: {query}")
        query_embedding = self._embedding_service.generate_embedding(query)
        client = self.get_client()

        rpc_function = (
            "match_document_chunks"
            if settings.use_chunked_storage
            else "match_documents"
        )
        print(
            f"Calling {rpc_function} RPC with threshold={threshold or settings.similarity_threshold}, "
            f"limit={limit or settings.max_search_results}"
        )
        result = client.rpc(
            rpc_function,
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold or settings.similarity_threshold,
                "match_count": limit or settings.max_search_results,
            },
        ).execute()
        print(f"Received {len(result.data)} results from {rpc_function}")
        return [SearchResult(**row) for row in result.data]


embedding_service = OpenAIEmbeddingService(ai_client)
vector_store = SupabaseVectorStore(embedding_service)
text_splitter = RecursiveTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
