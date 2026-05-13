"""LLM-обёртка поверх LangChain.

Поддерживаются два провайдера, выбираемых через ``settings.llm_provider``:
* ``openai``   — ``langchain_openai.ChatOpenAI`` (любой OpenAI-совместимый endpoint).
* ``gigachat`` — ``langchain_gigachat.GigaChat``.

Структурированный вывод реализуется через tool-binding LangChain: модель «вызывает»
tool с заданной JSON-схемой, мы возвращаем его аргументы. Это надёжнее, чем парсить
свободный JSON из текста, и единообразно работает у обоих провайдеров.
"""
import hashlib
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from easyrag.config import Provider, get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._mock = settings.llm_mock
        self._provider: Provider = settings.llm_provider
        self._model_name = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._chat: Any = None if self._mock else _build_chat(settings)

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

        if self._chat is None:
            raise RuntimeError("LLMClient is in mock mode; real client not initialised")

        tool = _schema_to_tool(tool_name, tool_description, input_schema)
        bound = self._chat.bind_tools([tool], tool_choice=tool_name)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = await bound.ainvoke(messages)

        for call in getattr(response, "tool_calls", None) or []:
            if call.get("name") == tool_name:
                return dict(call.get("args") or {})

        # Фолбэк: некоторые провайдеры (в т.ч. GigaChat) кладут аргументы
        # в additional_kwargs.function_call.arguments.
        extra = getattr(response, "additional_kwargs", {}) or {}
        fn = extra.get("function_call")
        if fn and fn.get("name") == tool_name:
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"LLM вернул не-JSON аргументы tool {tool_name}: {args!r}"
                    ) from exc
            if isinstance(args, dict):
                return dict(args)

        raise RuntimeError(f"LLM не вызвал tool {tool_name}: {response!r}")


def _build_chat(settings: Any) -> Any:
    """Сконструировать LangChain-чат под выбранного провайдера."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if settings.llm_provider == "gigachat":
        from langchain_gigachat import GigaChat

        kwargs = {
            "model": settings.llm_model,
            "credentials": settings.gigachat_credentials,
            "scope": settings.gigachat_scope,
            "verify_ssl_certs": settings.gigachat_verify_ssl,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        return GigaChat(**kwargs)

    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")


def _schema_to_tool(name: str, description: str, input_schema: dict[str, Any]) -> StructuredTool:
    """Превратить «голую» JSON-схему в LangChain-инструмент.

    Через ``args_schema`` LangChain передаёт схему как есть — это нужно, чтобы
    OpenAI и GigaChat увидели одни и те же поля и required-ограничения.
    """

    class _SchemaCarrier(BaseModel):
        @classmethod
        def model_json_schema(cls, *_: Any, **__: Any) -> dict[str, Any]:  # type: ignore[override]
            return input_schema

    def _noop(**_: Any) -> dict[str, Any]:
        return {}

    return StructuredTool.from_function(
        func=_noop,
        name=name,
        description=description,
        args_schema=_SchemaCarrier,
    )


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


__all__ = ["LLMClient", "get_llm", "set_llm_for_tests", "reset_llm", "dump_prompt"]


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
