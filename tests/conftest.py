import os

# Smoke-тесты идут полностью на моках — без сети и без Anthropic API key.
os.environ.setdefault("EASYRAG_LLM_MOCK", "1")
os.environ.setdefault("EASYRAG_EMBED_MOCK", "1")
