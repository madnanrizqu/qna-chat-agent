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
    default_model: str = Field(default="gemini-3.1-flash-lite-preview", validation_alias="DEFAULT_MODEL")


settings = Settings()
