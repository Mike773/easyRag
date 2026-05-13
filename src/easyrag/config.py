from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EASYRAG_",
        extra="ignore",
    )

    db_dsn: str = "postgresql+asyncpg://easyrag:easyrag@localhost:5435/easyrag"
    db_dsn_sync: str = "postgresql+psycopg://easyrag:easyrag@localhost:5435/easyrag"

    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 2048
    llm_mock: bool = False
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "EASYRAG_ANTHROPIC_API_KEY"),
    )

    embed_url: str = "http://localhost:8000/v1/embeddings"
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    embed_dim: int = 2560
    embed_api_key: str = ""
    embed_mock: bool = False

    resolve_thresh_high: float = 0.85
    resolve_thresh_low: float = 0.65
    abbr_thresh_high: float = 0.80
    graph_expand_thresh: float = 0.55


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
