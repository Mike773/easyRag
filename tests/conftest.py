import os
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent / ".env"
def _strip_inline_comment(value: str) -> str:
    # Standard dotenv semantics: ``KEY=value   # comment`` — комментарий
    # отсекается, если он не внутри кавычек. Простая эвристика без поддержки
    # экранирования достаточна для нашего .env.
    if not value:
        return value
    if value[0] in ("'", '"'):
        return value  # пусть pydantic сам разбирается с кавычками
    hash_idx = value.find("#")
    if hash_idx < 0:
        return value.rstrip()
    return value[:hash_idx].rstrip()


if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), _strip_inline_comment(value.strip()))

os.environ.setdefault("EASYRAG_LLM_MOCK", "1")
os.environ.setdefault("EASYRAG_EMBED_MOCK", "1")
