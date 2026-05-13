from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "gigachat"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EASYRAG_",
        extra="ignore",
    )

    db_dsn: str = "postgresql+asyncpg://easyrag:easyrag@localhost:5435/easyrag"
    db_dsn_sync: str = "postgresql+psycopg://easyrag:easyrag@localhost:5435/easyrag"

    # --- LLM ---
    llm_provider: Provider = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0
    llm_mock: bool = False

    # OpenAI-совместимый LLM (langchain_openai.ChatOpenAI).
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "EASYRAG_OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "EASYRAG_OPENAI_BASE_URL"),
    )

    # GigaChat (langchain_gigachat.GigaChat).
    gigachat_credentials: str = Field(
        default="",
        validation_alias=AliasChoices("GIGACHAT_CREDENTIALS", "EASYRAG_GIGACHAT_CREDENTIALS"),
    )
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl: bool = False

    # --- Embeddings ---
    embed_provider: Provider = "openai"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    embed_mock: bool = False

    # OpenAI-совместимые эмбеддинги (langchain_openai.OpenAIEmbeddings).
    embed_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBED_API_KEY", "EASYRAG_EMBED_API_KEY"),
    )
    embed_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("EMBED_BASE_URL", "EASYRAG_EMBED_BASE_URL"),
    )

    resolve_thresh_high: float = 0.85
    resolve_thresh_low: float = 0.65
    abbr_thresh_high: float = 0.80
    graph_expand_thresh: float = 0.55


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
