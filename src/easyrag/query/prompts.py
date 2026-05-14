"""Промпты шага 4.

Два tool'а:

* ``save_wiki_page`` — резолвер вызывает LLM, чтобы перезаписать
  ``wiki_page.body_md`` с учётом новых ``statements`` от кандидатов. Слияние
  происходит в одном tool-вызове на страницу: модель видит текущее тело,
  список новых утверждений и обязана выдать целостный markdown без потери
  ранее зафиксированных фактов.

* ``save_answer`` — query-пайплайн просит модель ответить на вопрос
  пользователя ИСКЛЮЧИТЕЛЬНО по предоставленным секциям wiki и обязательно
  отдать список цитат ``slug#anchor``. Цитаты потом разрешаются в реальные
  секции и их провенанс.

Оба tool'а возвращают строго JSON по схеме — отдельно парсить свободный
текст не нужно.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Слияние тела wiki-страницы (resolver)
# ---------------------------------------------------------------------------

WIKI_MERGE_TOOL_NAME = "save_wiki_page"
WIKI_MERGE_TOOL_DESCRIPTION = (
    "Сохранить итоговое тело wiki-страницы (markdown) и набор алиасов. "
    "Вызывать ровно один раз. Содержимое body_md полностью заменяет текущее."
)

WIKI_MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body_md": {
            "type": "string",
            "description": (
                "Полный markdown страницы. Структурируй H2-секциями (## Заголовок). "
                "Каждое утверждение — самодостаточное предложение. "
                "При упоминании других именованных сущностей оборачивай их в [[Имя]] "
                "(имена собственные, идентифицируемые сущности; не даты, числа и общие слова). "
                "НЕ выдумывай факты — используй только то, что есть в текущем теле "
                "или в новых утверждениях."
            ),
        },
        "aliases": {
            "type": "array",
            "description": (
                "Альтернативные имена этой сущности (без основного заголовка). "
                "Включи существующие алиасы и добавь варианты написания, "
                "встретившиеся среди новых имён."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["body_md", "aliases"],
    "additionalProperties": False,
}


WIKI_MERGE_SYSTEM = (
    "Ты ведёшь wiki по сущностям, упомянутым в исходных документах. Тебе дают "
    "существующее тело страницы (может быть пустым — тогда страница создаётся "
    "впервые) и список новых утверждений, извлечённых из исходных документов. "
    "Твоя задача — выдать обновлённое тело страницы целиком.\n"
    "\n"
    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "1. Сохрани ВСЕ факты из текущего тела. Не удаляй и не сокращай ранее "
    "зафиксированные утверждения, даже если они кажутся избыточными. "
    "Если возникает прямое противоречие — оставь оба варианта и пометь "
    "коротким комментарием, какой из источников их даёт.\n"
    "2. Влей новые утверждения в подходящие секции. Если темы нет — создай "
    "новую H2-секцию с осмысленным заголовком.\n"
    "3. Оборачивай в [[Имя]] упоминания других именованных сущностей — любых "
    "идентифицируемых объектов с собственным именем, которые могут иметь "
    "отдельную wiki-страницу. НЕ ссылайся на даты, числа и общие слова. "
    "Не ограничивай себя каким-то одним классом сущностей — действуй по смыслу "
    "документа.\n"
    "4. КАТЕГОРИЧЕСКИ запрещено оборачивать в [[…]] упоминания самой текущей "
    "страницы (её заголовок указан в user-сообщении первым). Никогда не "
    "пиши [[Заголовок]], если это заголовок именно этой страницы — повторяй "
    "имя как обычный текст. Самореференция ломает граф ссылок.\n"
    "5. Каталог «Существующие сущности» (если он есть в user-сообщении) — это "
    "ОРФОГРАФИЧЕСКАЯ ПОДСКАЗКА, как правильно записать ссылку на сущность, "
    "которая УЖЕ реально упомянута в текущем теле или в новых утверждениях. "
    "Если сущность из каталога действительно встречается в материале — "
    "оборачивай её упоминание в [[Точное имя из каталога]] (не сокращённый "
    "вариант). НЕ выдумывай упоминания и не вставляй сущности из каталога, "
    "которых нет в исходном материале, только ради того, чтобы добавить "
    "ссылок. Каталог не является списком обязательных тем.\n"
    "6. Не выдумывай факты. Если новое утверждение продублировано — не дублируй "
    "его в выводе.\n"
    "7. Заголовки секций — короткие, в именительном падеже, по смыслу "
    "содержимого секции.\n"
    "8. Имена и обозначения на латинице сохраняй в оригинальном написании.\n"
    "9. Ответ — только через вызов tool save_wiki_page. Свободный текст игнорируется."
)


def build_merge_user_prompt(
    *,
    title: str,
    current_body: str,
    current_aliases: Sequence[str],
    new_descriptors: Sequence[str],
    new_statements: Sequence[str],
    source_uris: Sequence[str] = (),
    existing_entities: Sequence[tuple[str, Sequence[str]]] = (),
) -> str:
    """Собрать user-сообщение для merge-вызова.

    ``new_descriptors`` — короткие descriptor'ы кандидатов, склеившихся в эту
    страницу в текущем раунде. Они помогают модели понять, под каким углом
    новые statements относятся к сущности.

    ``source_uris`` опционально — можно перечислить uri документов-источников
    этого раунда, чтобы модель имела право упомянуть «по данным <uri>», если
    хочет (но это не обязательно).

    ``existing_entities`` — каталог уже существующих страниц wiki в формате
    ``(title, aliases)``. При непустом списке модель обязана линковать
    упоминания этих сущностей в [[…]] точно по приведённому имени —
    иначе ссылка не найдёт целевую страницу.
    """
    title_clean = title.strip()
    parts: list[str] = [
        f"Заголовок страницы: {title_clean}\n"
        f"ВАЖНО: НЕ оборачивай «{title_clean}» в [[…]] в тексте — это заголовок "
        "САМОЙ страницы, на себя не ссылаемся."
    ]

    aliases_clean = [a.strip() for a in current_aliases if a and a.strip()]
    if aliases_clean:
        parts.append("Текущие алиасы: " + ", ".join(aliases_clean))

    descriptors_clean = [d.strip() for d in new_descriptors if d and d.strip()]
    if descriptors_clean:
        bullets = "\n".join(f"- {d}" for d in descriptors_clean)
        parts.append("Контекст новых вкладов (descriptor'ы кандидатов):\n" + bullets)

    catalog_lines: list[str] = []
    for ent_title, ent_aliases in existing_entities:
        name = (ent_title or "").strip()
        if not name:
            continue
        cleaned_aliases = [a.strip() for a in (ent_aliases or ()) if a and a.strip()]
        if cleaned_aliases:
            catalog_lines.append(f"- {name} (псевдонимы: {', '.join(cleaned_aliases)})")
        else:
            catalog_lines.append(f"- {name}")
    if catalog_lines:
        parts.append(
            "Существующие сущности (используй [[Имя]] точно по этому списку, "
            "если упоминаешь любую из них):\n" + "\n".join(catalog_lines)
        )

    body_block = (current_body or "").strip()
    if body_block:
        parts.append(
            "Текущее тело страницы (между тегами):\n"
            f"<current_body>\n{body_block}\n</current_body>"
        )
    else:
        parts.append("Текущего тела страницы нет — создаёшь с нуля.")

    statements_clean = [s.strip() for s in new_statements if s and s.strip()]
    if not statements_clean:
        parts.append("Новых утверждений нет — верни текущее тело и алиасы без изменений.")
    else:
        bullets = "\n".join(f"- {s}" for s in statements_clean)
        parts.append("Новые утверждения для интеграции:\n" + bullets)

    sources_clean = [u.strip() for u in source_uris if u and u.strip()]
    if sources_clean:
        parts.append("Источники этого раунда: " + ", ".join(sources_clean))

    parts.append(
        "Сформируй обновлённое тело страницы и набор алиасов по правилам "
        "системного сообщения и верни их вызовом tool save_wiki_page."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Back-link: добавить [[…]] ссылки на новые сущности в существующую страницу
# ---------------------------------------------------------------------------

WIKI_RELINK_TOOL_NAME = "relink_wiki_page"
WIKI_RELINK_TOOL_DESCRIPTION = (
    "Сохранить обновлённое тело wiki-страницы (markdown) после простановки "
    "[[…]] ссылок на сущности из переданного каталога. Вызывать ровно один раз."
)

WIKI_RELINK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body_md": {
            "type": "string",
            "description": (
                "Полный markdown страницы. Содержимое и структура должны "
                "совпадать с current_body, отличие — только в добавленных "
                "[[Имя]] ссылках на сущности из каталога, упоминания которых "
                "уже есть в тексте."
            ),
        },
        "aliases": {
            "type": "array",
            "description": (
                "Алиасы страницы. ВЕРНИ ТЕ ЖЕ, что переданы в current_aliases — "
                "поле обязательно по схеме."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["body_md", "aliases"],
    "additionalProperties": False,
}


WIKI_RELINK_SYSTEM = (
    "Ты ведёшь wiki по сущностям. Тебе дают тело уже существующей страницы и "
    "каталог других страниц wiki (title + aliases). Твоя задача — оборачивать "
    "в [[Имя]] упоминания этих сущностей, которые УЖЕ присутствуют в текущем "
    "теле страницы, и больше ничего не менять.\n"
    "\n"
    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "1. НЕ меняй формулировки, порядок слов, структуру секций, заголовки H2.\n"
    "2. НЕ добавляй новых фактов, предложений, секций.\n"
    "3. НЕ удаляй ничего из текущего тела — ни предложений, ни ссылок.\n"
    "4. НЕ оборачивай в [[…]] упоминания САМОЙ этой страницы (её заголовок "
    "указан в user-сообщении первым). Самореференция ломает граф ссылок.\n"
    "5. Уже существующие [[…]] оставь как есть.\n"
    "6. Если упоминание сущности из каталога стоит в тексте в склонённой форме, "
    "оборачивай весь упомянутый токен в [[Имя из каталога|склонённая форма]] — "
    "так читатель видит исходный текст, а граф знает целевой slug. Не "
    "переписывай склонение в именительный.\n"
    "7. Если ни одной сущности из каталога в теле нет — верни current_body "
    "ровно как был передан.\n"
    "8. НЕ выдумывай упоминаний и не вставляй сущности из каталога, которых "
    "нет в исходном теле.\n"
    "9. Поле aliases в ответе — те же значения, что в current_aliases.\n"
    "10. Ответ — только через вызов tool relink_wiki_page."
)


def build_relink_user_prompt(
    *,
    title: str,
    current_body: str,
    current_aliases: Sequence[str],
    catalog: Sequence[tuple[str, Sequence[str]]],
) -> str:
    """Собрать user-сообщение для relink-вызова.

    ``catalog`` — список ``(title, aliases)`` всех других страниц wiki, на
    которые можно ставить ссылки. Пустой каталог означает, что ставить нечего;
    обычно в этом случае вызывать relink бессмысленно — фильтруйте до вызова.
    """
    title_clean = title.strip()
    parts: list[str] = [
        f"Заголовок страницы: {title_clean}\n"
        f"ВАЖНО: НЕ оборачивай «{title_clean}» в [[…]] в тексте — это заголовок "
        "САМОЙ страницы, на себя не ссылаемся."
    ]

    aliases_clean = [a.strip() for a in current_aliases if a and a.strip()]
    if aliases_clean:
        parts.append("Текущие алиасы (передай в ответе без изменений): " + ", ".join(aliases_clean))
    else:
        parts.append("Текущих алиасов нет — верни в ответе пустой массив.")

    catalog_lines: list[str] = []
    for ent_title, ent_aliases in catalog:
        name = (ent_title or "").strip()
        if not name:
            continue
        cleaned_aliases = [a.strip() for a in (ent_aliases or ()) if a and a.strip()]
        if cleaned_aliases:
            catalog_lines.append(f"- {name} (псевдонимы: {', '.join(cleaned_aliases)})")
        else:
            catalog_lines.append(f"- {name}")
    if catalog_lines:
        parts.append(
            "Каталог сущностей (используй [[Имя]] точно по этому списку, если "
            "упоминание встретилось в теле страницы):\n" + "\n".join(catalog_lines)
        )
    else:
        parts.append("Каталог пуст — оборачивать нечего, верни тело без изменений.")

    body_block = (current_body or "").strip()
    if body_block:
        parts.append(
            "Текущее тело страницы (между тегами, итоговое тело должно "
            "совпадать дословно, кроме добавленных [[…]] ссылок):\n"
            f"<current_body>\n{body_block}\n</current_body>"
        )
    else:
        parts.append("Текущего тела страницы нет — верни пустую строку.")

    parts.append(
        "Верни обновлённое тело и тот же набор алиасов вызовом tool "
        f"{WIKI_RELINK_TOOL_NAME}."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ответ на запрос (query)
# ---------------------------------------------------------------------------

ANSWER_TOOL_NAME = "save_answer"
ANSWER_TOOL_DESCRIPTION = (
    "Сохранить ответ на вопрос пользователя и список цитированных wiki-секций. "
    "Вызывать ровно один раз."
)

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "Ответ на вопрос пользователя в свободной форме. "
                "Опирайся ТОЛЬКО на содержимое предоставленных секций. "
                "Если в секциях нет ответа — честно скажи 'нет данных' и оставь "
                "citations пустым."
            ),
        },
        "citations": {
            "type": "array",
            "description": (
                "Список секций, реально использованных при формировании ответа. "
                "Каждая запись — пара (slug, anchor) из заголовков секций ниже. "
                "Не выдумывай slug/anchor: используй только те, что есть в контексте."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "anchor": {"type": "string"},
                },
                "required": ["slug", "anchor"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


ANSWER_SYSTEM = (
    "Ты отвечаешь на вопросы пользователя по внутренней бизнес-wiki. "
    "Тебе дают набор секций wiki (каждая помечена заголовком "
    "[slug#anchor] Название). Это ЕДИНСТВЕННЫЙ источник истины для ответа.\n"
    "\n"
    "ПРАВИЛА:\n"
    "1. Используй ТОЛЬКО факты из предоставленных секций. Никаких догадок, "
    "общих знаний, домысливания.\n"
    "2. Если в секциях нет ответа — верни короткое 'Нет данных в wiki по этому "
    "вопросу.' и пустой список citations. Не пытайся сочинить ответ.\n"
    "3. Если ответ есть, он должен быть кратким и по делу. Без вступлений типа "
    "'согласно предоставленным данным'.\n"
    "4. В citations включай только те секции, факты из которых ты реально "
    "использовал в ответе. Не дублируй цитаты на одну и ту же секцию.\n"
    "5. НЕ упоминай ID секций, slug'и, anchor'ы в самом тексте ответа — это "
    "метаданные, они уходят отдельно в поле citations.\n"
    "6. Ответ — только через вызов tool save_answer. Свободный текст игнорируется."
)


def build_answer_user_prompt(
    *,
    question: str,
    sections: Sequence["AnsweredSection"],
) -> str:
    """Собрать user-сообщение для answer-вызова.

    ``sections`` — список ``AnsweredSection`` (см. ниже) в порядке убывания
    релевантности; они уже отфильтрованы пайплайном.
    """
    parts: list[str] = [f"Вопрос пользователя: {question.strip()}"]

    if not sections:
        parts.append("Доступных секций wiki НЕТ.")
    else:
        chunks: list[str] = []
        for sec in sections:
            header = f"[{sec.slug}#{sec.anchor}] {sec.page_title} → {sec.section_title}"
            body = (sec.body_md or "").strip()
            if body:
                chunks.append(f"{header}\n{body}")
            else:
                chunks.append(f"{header}\n(секция без тела)")
        parts.append("Секции wiki (между тегами <sections>):")
        parts.append("<sections>\n" + "\n\n---\n\n".join(chunks) + "\n</sections>")

    parts.append(
        "Сформируй ответ строго по правилам системного сообщения и верни его "
        "вызовом tool save_answer."
    )
    return "\n\n".join(parts)


# Носитель данных секции в формате, ожидаемом ``build_answer_user_prompt``.
# Сам класс лежит в pipeline.py, но для аннотации тут используется forward-ref,
# чтобы не плодить циклический импорт. Контракт интерфейса:
#   slug: str, anchor: str, page_title: str, section_title: str, body_md: str
class AnsweredSection:  # noqa: D401 — protocol-like заглушка
    """Структурный протокол: поля используются в :func:`build_answer_user_prompt`."""

    slug: str
    anchor: str
    page_title: str
    section_title: str
    body_md: str


__all__ = [
    "ANSWER_SCHEMA",
    "ANSWER_SYSTEM",
    "ANSWER_TOOL_DESCRIPTION",
    "ANSWER_TOOL_NAME",
    "AnsweredSection",
    "WIKI_MERGE_SCHEMA",
    "WIKI_MERGE_SYSTEM",
    "WIKI_MERGE_TOOL_DESCRIPTION",
    "WIKI_MERGE_TOOL_NAME",
    "WIKI_RELINK_SCHEMA",
    "WIKI_RELINK_SYSTEM",
    "WIKI_RELINK_TOOL_DESCRIPTION",
    "WIKI_RELINK_TOOL_NAME",
    "build_answer_user_prompt",
    "build_merge_user_prompt",
    "build_relink_user_prompt",
]
