"""Обёртка над Anthropic Claude.

Структурированный вывод реализуется через tool use: модель «вызывает» tool с
заданной JSON-схемой, мы возвращаем его аргументы. Это надёжнее, чем парсить
свободный JSON из текста.
"""
import hashlib
import json
from typing import Any

from anthropic import AsyncAnthropic

from easyrag.config import get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._mock = settings.llm_mock
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._client: AsyncAnthropic | None = None
        if not self._mock:
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def call_json(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Запросить у модели структурированный ответ по JSON-схеме."""
        if self._mock:
            return _mock_response(tool_name, system, user, input_schema)

        if self._client is None:
            raise RuntimeError("LLMClient is in mock mode; real API client not initialised")
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return dict(block.input)
        raise RuntimeError(f"LLM не вызвал tool {tool_name}: {response.content!r}")


def _mock_response(
    tool_name: str, system: str, user: str, input_schema: dict[str, Any]
) -> dict[str, Any]:
    """Детерминированные мок-ответы — для smoke-тестов без сети.

    Возвращает «нулевую» структуру, валидную по схеме.
    """
    seed = hashlib.sha256(f"{tool_name}|{system}|{user}".encode()).hexdigest()[:8]
    return _empty_for_schema(input_schema, seed)


def _empty_for_schema(schema: dict[str, Any], seed: str) -> dict[str, Any]:
    if schema.get("type") != "object":
        return {}
    out: dict[str, Any] = {}
    for prop, ps in (schema.get("properties") or {}).items():
        out[prop] = _empty_value(ps, seed)
    return out


def _empty_value(schema: dict[str, Any], seed: str) -> Any:
    t = schema.get("type")
    if t == "string":
        return f"mock-{seed}"
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return _empty_for_schema(schema, seed)
    return None


__all__ = ["LLMClient"]


# Сахар для тестов — переопределяемая фабрика.
_default_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def set_llm_for_tests(client: LLMClient) -> None:
    global _default_client
    _default_client = client


def reset_llm() -> None:
    global _default_client
    _default_client = None


# Для отладки удобно дампить prompt'ы — экспортируем хелпер.
def dump_prompt(system: str, user: str) -> str:
    return json.dumps({"system": system, "user": user}, ensure_ascii=False, indent=2)
