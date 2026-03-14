from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(validation_alias="OPENAI_BASE_URL")
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
    max_search_results: int = Field(default=1, validation_alias="MAX_SEARCH_RESULTS")


settings = Settings()
