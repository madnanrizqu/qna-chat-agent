from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    google_api_key: str = Field(validation_alias="GOOGLE_API_KEY")
    default_model: str = Field(
        default="gemini-3.1-flash-lite-preview", validation_alias="DEFAULT_MODEL"
    )

    vector_database_url: str = Field(validation_alias="VECTOR_DATABASE_URL")
    vector_database_api_key: str = Field(validation_alias="VECTOR_DATABASE_API_KEY")

    embedding_model: str = Field(
        default="gemini-embedding-001", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(
        default=3072, validation_alias="EMBEDDING_DIMENSIONS"
    )

    similarity_threshold: float = Field(
        default=0.6, validation_alias="SIMILARITY_THRESHOLD"
    )
    max_search_results: int | None = Field(
        default=None, validation_alias="MAX_SEARCH_RESULTS"
    )

    @model_validator(mode="after")
    def set_max_search_results_default(self):
        """Set max_search_results based on use_chunked_storage if not explicitly provided."""
        if self.max_search_results is None:
            self.max_search_results = 5 if self.use_chunked_storage else 1
        return self

    chunk_size: int = Field(default=200, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=0, validation_alias="CHUNK_OVERLAP")

    use_chunked_storage: bool = Field(
        default=False, validation_alias="USE_CHUNKED_STORAGE"
    )

    environment: str = Field(default="development", validation_alias="ENVIRONMENT")


settings = Settings()
