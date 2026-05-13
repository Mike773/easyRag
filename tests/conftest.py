import os
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

os.environ.setdefault("EASYRAG_LLM_MOCK", "1")
os.environ.setdefault("EASYRAG_EMBED_MOCK", "1")
