# easyRag

Wiki-first RAG для бизнес-документации. CLI-only, без HTTP-слоя.

**Стек:** Python 3.11 · SQLAlchemy 2.0 (async) · PostgreSQL + pgvector · LangChain поверх OpenAI и GigaChat (tool-binding для структурированного вывода) · эмбеддинги через `OpenAIEmbeddings` / `GigaChatEmbeddings`.

**Статус:** pre-alpha / шаг 0 — скелет проекта (config, db-модели, LLM/embed клиенты, миграции, smoke-тесты). Пайплайны ingest / query / enrichment / typing будут добавлены на последующих шагах.

## Идея

Документы (источники) парсятся в чанки и используются как сырьё. Из них извлекаются сущности → нормализуются в **wiki-страницы** (markdown с `[[ссылками]]` между собой). Запросы пользователя идут не к чанкам, а к wiki-секциям. Чанки-источники сохраняются только ради провенанса (откуда какая фраза в wiki). Граф рёбер `wiki_link` — производный индекс, всегда пересобирается из markdown.

## Quickstart

```bash
# 1. Поднять Postgres + pgvector
docker compose up -d

# 2. Виртуальное окружение и зависимости
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# 3. Конфиг
cp .env.example .env
# выбрать провайдеров (EASYRAG_LLM_PROVIDER / EASYRAG_EMBED_PROVIDER)
# и заполнить креды: OPENAI_API_KEY / EMBED_API_KEY либо GIGACHAT_CREDENTIALS

# 4. Применить миграции
alembic upgrade head

# 5. Smoke-тест
pytest tests/test_smoke.py
```

## Провайдеры LLM и эмбеддингов

Провайдер выбирается независимо для LLM и для эмбеддингов. Допустимые значения — `openai` (любой OpenAI-совместимый endpoint через `langchain_openai`) или `gigachat` (через `langchain_gigachat`).

### OpenAI / OpenAI-совместимый endpoint

```env
EASYRAG_LLM_PROVIDER=openai
EASYRAG_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1   # опционально — для прокси / локального сервера

EASYRAG_EMBED_PROVIDER=openai
EASYRAG_EMBED_MODEL=text-embedding-3-small
EASYRAG_EMBED_DIM=1536
EMBED_API_KEY=sk-...
# EMBED_BASE_URL=https://api.openai.com/v1
```

Для семейства `text-embedding-3-*` параметр `EASYRAG_EMBED_DIM` пробрасывается как `dimensions` в API.

### GigaChat

```env
EASYRAG_LLM_PROVIDER=gigachat
EASYRAG_LLM_MODEL=GigaChat
GIGACHAT_CREDENTIALS=<base64 client_id:client_secret>
EASYRAG_GIGACHAT_SCOPE=GIGACHAT_API_PERS
EASYRAG_GIGACHAT_VERIFY_SSL=0

EASYRAG_EMBED_PROVIDER=gigachat
EASYRAG_EMBED_MODEL=Embeddings
EASYRAG_EMBED_DIM=1024
```

Структурированный вывод реализован одинаково для обоих провайдеров через `bind_tools(..., tool_choice=name)` — модель «вызывает» tool с заданной JSON-схемой, а мы возвращаем его аргументы.

## Структура

```
src/easyrag/
  config.py        # настройки из env (pydantic-settings)
  db/              # модели + асинхронная сессия
  llm/             # LangChain-клиенты LLM + embeddings (openai/gigachat, mock-режим)
  cli.py           # Click CLI (заглушка на шаге 0)
  wiki/            # markdown, link index (шаг 1, планируется)
  ingest/          # пайплайн загрузки (шаг 2, планируется)
  abbreviations/   # словарь и query-expand (шаг 3, планируется)
  query/           # retrieval + answer (шаг 4, планируется)
  enrichment/      # gap loop (шаг 5, планируется)
  typing_/         # эмерджентные типы (шаг 6, планируется)
```

## CLI (планируется)

На шагах 2/4/5/6 появятся команды:

```
easyrag ingest --uri <name> --file path/to/text.txt
easyrag query "ваш вопрос"
easyrag enrich --limit 50
easyrag retype
easyrag resolve-abbr
```

Сейчас доступна только `easyrag status` — выводит, что CLI пока не реализован.

## Тесты

```bash
pytest                              # все тесты
pytest tests/test_smoke.py -v       # smoke (на моках, без сети и БД)
```

Smoke-тесты используют детерминированные mock-ответы LLM и embeddings (`EASYRAG_LLM_MOCK=1`, `EASYRAG_EMBED_MOCK=1`) — не требуют ни кредов провайдера, ни запущенного Postgres.

## Лицензия

MIT — см. [LICENSE](LICENSE).
