# easyRag

Wiki-first RAG для бизнес-документации. CLI-only, без HTTP-слоя.

**Стек:** Python 3.11 · SQLAlchemy 2.0 (async) · PostgreSQL + pgvector · LangChain поверх OpenAI и GigaChat (tool-binding для структурированного вывода) · эмбеддинги через `OpenAIEmbeddings` / `GigaChatEmbeddings`.

**Статус:** pre-alpha / шаг 4 — ingest (чанки + кандидаты сущностей) с авто-резолвом в wiki через LLM-merge, query-пайплайн с retrieval (vector + graph expansion) и провенансом. Дальше: шаг 3 (аббревиатуры), шаг 5 (enrichment), шаг 6 (типизация).

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
  cli.py           # Click CLI (ingest, query)
  wiki/            # markdown, link index (шаг 1)
  ingest/          # пайплайн загрузки (шаг 2; на шаге 4 — авто-резолв в wiki)
  query/           # resolver + retrieval + answer (шаг 4)
  abbreviations/   # словарь и query-expand (шаг 3, планируется)
  enrichment/      # gap loop (шаг 5, планируется)
  typing_/         # эмерджентные типы (шаг 6, планируется)
```

## CLI

Сейчас доступны команды `status`, `ingest`, `query`:

```
easyrag status
easyrag ingest --uri contracts/7 --file path/to/text.txt
easyrag query "ваш вопрос" [--top-k 8] [--no-graph] [--no-show-sources]
```

`ingest` после извлечения сущностей автоматически резолвит кандидатов в
`wiki_page` через LLM-merge (можно отключить флагом `--no-resolve` — тогда
кандидаты останутся в `entity_candidate` без материализации). Слияние:
кандидат по вектору `name. descriptor` ищет ближайшую wiki-страницу; при
similarity ≥ `EASYRAG_RESOLVE_THRESH_HIGH` (0.85 по умолчанию) — вливается в
существующую, при similarity между `RESOLVE_THRESH_LOW` (0.65) и high —
помечается ambiguous и не материализуется, иначе создаётся новая страница.

`query` эмбеддит вопрос, делает top-K cosine-поиск по `wiki_section`,
расширяется по `wiki_link` (порог `EASYRAG_GRAPH_EXPAND_THRESH`, 0.55), просит
LLM ответить только по контексту с цитатами `slug#anchor`, после чего
печатает ответ и `section_provenance` (uri исходного документа + char-offset'ы).
Каждый запрос пишется в `query_gap`; если ни одна секция не дала валидной
цитаты — запись остаётся неразрешённой (сырьё для шага 5).

Команды `enrich`, `retype`, `resolve-abbr` появятся на шагах 5/6/3.

## Тесты

```bash
pytest                              # все тесты
pytest tests/test_smoke.py -v       # smoke (на моках, без сети и БД)
```

Smoke-тесты используют детерминированные mock-ответы LLM и embeddings (`EASYRAG_LLM_MOCK=1`, `EASYRAG_EMBED_MOCK=1`) — не требуют ни кредов провайдера, ни запущенного Postgres.

## Лицензия

MIT — см. [LICENSE](LICENSE).
