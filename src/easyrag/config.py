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

    # Резолвер: эмбеддинги кандидатов считаются по чистому ``name``
    # (см. ingest.pipeline._embed_text), поэтому пороги ниже, чем были бы для
    # ``name. descriptor``. Зона между low и high — LLM-судья.
    # high≥thresh_high → уверенный merge без LLM; <thresh_low → уверенно новая;
    # между — судья смотрит на top-N страниц и решает.
    resolve_thresh_high: float = 0.85
    resolve_thresh_low: float = 0.45
    abbr_thresh_high: float = 0.80
    graph_expand_thresh: float = 0.55

    # Сколько символов от начала документа подавать в analyze_document
    # для построения domain brief. Brief считается один раз на ingest.
    doc_brief_window: int = 4000

    # Параметры чанкера (chars). См. easyrag.ingest.chunker.chunk_text.
    # target_size — мягкая цель группировки коротких абзацев;
    # max_size — жёсткий потолок на отдельный абзац (длиннее режется с overlap);
    # overlap применяется только к принудительным срезам внутри длинных абзацев.
    chunk_target_size: int = 1200
    chunk_max_size: int = 1800
    chunk_overlap: int = 150

    # Сколько уже существующих страниц передавать в merge-prompt как каталог
    # сущностей, на которые LLM должен ссылаться через [[…]]. На больших wiki
    # имеет смысл отбирать top-K по similarity — пока считаем, что для текущего
    # масштаба простой лимит достаточен.
    merge_catalog_limit: int = 300

    # Back-link: после каждого resolve_candidates пройтись по существующим
    # страницам и попросить LLM добавить [[…]] ссылки на сущности, появившиеся
    # в этом раунде (см. easyrag.wiki.backlinker). Без этого старые страницы
    # никогда не узнают о новых соседях.
    backlink_enabled: bool = True
    # Pre-filter: пропускать страницу, если в её body_md нет ни одного
    # substring-совпадения с title/alias свежих сущностей. Экономит LLM-вызовы
    # при «холостых» проходах. На сильно склоняющихся языках (русский) может
    # давать ложные пропуски — тогда выключай.
    backlink_prefilter: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
