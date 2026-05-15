"""Юнит-тесты санитизации ``body_md``.

Без БД, без сети. См. :mod:`easyrag.wiki.sanitize`.
"""
from easyrag.wiki.markdown import parse_page
from easyrag.wiki.sanitize import sanitize_body_md

# --- 1. Unescape literal \n ---


def test_sanitize_unescapes_when_no_real_newlines():
    body = "## Заголовок\\n\\nОдин абзац.\\n\\n## Второй\\n\\nЕщё.\\n"
    out, repairs = sanitize_body_md(body)
    assert "unescaped-literal-newlines" in repairs
    assert "\n" in out
    assert "## Заголовок" in out
    assert "## Второй" in out


def test_sanitize_skips_unescape_when_real_newlines_present():
    body = "## Real\n\nОдин абзац с literal \\n внутри.\n"
    out, repairs = sanitize_body_md(body)
    assert "unescaped-literal-newlines" not in repairs
    assert "\\n" in out


def test_sanitize_unescapes_single_literal_newline_when_no_real():
    # Реальная регрессия (страница "жучка"): короткое тело «заголовок + один
    # абзац» пришло как одна строка с единственным literal \n и без реальных
    # LF. Старый порог (≥3 literal \n) такой кейс пропускал — весь абзац
    # влипал в H2-заголовок, anchor превращался в мусор.
    body = "## Участие\\nЖучка позвала кошку на помощь."
    out, repairs = sanitize_body_md(body)
    assert "unescaped-literal-newlines" in repairs
    parsed = parse_page(out)
    assert [s.title for s in parsed.sections] == ["Участие"]
    assert "Жучка позвала кошку" in parsed.sections[0].body_md


def test_sanitize_unescape_makes_parse_page_see_h2():
    # Реальная регрессия: страница "лиса" пришла одной строкой со literal \n
    # → _H2_RE с MULTILINE не находил ``## ...``, и вся страница парсилась
    # как одна гигантская секция Overview.
    body = "## Описание\\n\\nЛисичка-сестричка обманывает.\\n\\n## Сюжет\\n\\nЕст колобка."
    out, repairs = sanitize_body_md(body)
    assert "unescaped-literal-newlines" in repairs
    parsed = parse_page(out)
    assert [s.title for s in parsed.sections] == ["Описание", "Сюжет"]


# --- 2. Trailing junk trim ---


def test_sanitize_trims_trailing_json_artifacts():
    body = '## Заголовок\n\n- факт один\n- факт два\n"+\n}'
    out, repairs = sanitize_body_md(body)
    assert any(r.startswith("trimmed-tail") for r in repairs)
    assert out.endswith("- факт два")


def test_sanitize_trims_blank_tail_silently():
    # Пустые строки на хвосте режутся, но без шума в repairs.
    body = "## A\n\nтекст\n\n\n"
    out, repairs = sanitize_body_md(body)
    assert out.endswith("текст")
    assert not any(r.startswith("trimmed-tail") for r in repairs)


def test_sanitize_keeps_content_lines_with_punctuation():
    # Регресс-якорь: строка с буквами не должна считаться мусором,
    # даже если содержит фигурные/квадратные скобки.
    body = '## A\n\nфакт: {"x":1}\n'
    out, _ = sanitize_body_md(body)
    assert '{"x":1}' in out


# --- 3. Mixed-script links ---


def test_sanitize_fixes_latin_D_in_cyrillic_display():
    # Реальная регрессия: на странице "мышка" появилось [[Старик|Dед]] —
    # латинская D вместо Д. Гомоглифная замена должна это починить.
    body = "## A\n\nПриходил [[Старик|Dед]] вечером.\n"
    out, repairs = sanitize_body_md(body)
    assert any(r.startswith("link-homograph") for r in repairs)
    assert "[[Старик|Дед]]" in out
    assert "Dед" not in out


def test_sanitize_strips_display_when_unfixable():
    # Латиница без визуального аналога (q) — починить нельзя, схлопываем
    # до [[target]] без display.
    body = "## A\n\nсм. [[Старик|qед]] здесь.\n"
    out, repairs = sanitize_body_md(body)
    assert any(r.startswith("link-stripped-display") for r in repairs)
    assert "[[Старик]]" in out
    assert "qед" not in out


def test_sanitize_keeps_legitimate_latin_display():
    # target=Сбер (cyr), display=Sberbank (pure latin) — это легитимная
    # транслитерация, не трогаем.
    body = "## A\n\n[[Сбер|Sberbank]] банк.\n"
    out, repairs = sanitize_body_md(body)
    assert "[[Сбер|Sberbank]]" in out
    assert not any(r.startswith("link-") for r in repairs)


def test_sanitize_keeps_latin_target_unchanged():
    # target содержит латиницу — это не наш кейс, не вмешиваемся даже если
    # display mixed.
    body = "## A\n\n[[Acme|ООО Acme]] поставщик.\n"
    out, repairs = sanitize_body_md(body)
    assert "[[Acme|ООО Acme]]" in out
    assert not repairs


def test_sanitize_keeps_link_without_display():
    body = "## A\n\nсм. [[Старик]] выше.\n"
    out, repairs = sanitize_body_md(body)
    assert "[[Старик]]" in out
    assert not repairs


# --- 4. Happy path ---


def test_sanitize_no_op_on_clean_input():
    body = (
        "## Описание\n\n"
        "Колобок — главный герой сказки.\n\n"
        "## Сюжет\n\n"
        "Встречает [[Заяц|зайца]], [[Волк|волка]] и [[Лиса|лису]].\n"
    )
    out, repairs = sanitize_body_md(body)
    assert out == body.rstrip()
    assert repairs == []


def test_sanitize_handles_empty_input():
    out, repairs = sanitize_body_md("")
    assert out == ""
    assert repairs == []


# --- 5. Composite (real bug we saw) ---


def test_sanitize_full_repair_kolobok_like():
    # Имитация реального колобка-выхлопа: literal \n + trailing JSON + битая ссылка.
    body = (
        "## Сюжет\\n\\nКолобок убежал от [[Старик|Dед]] и [[Баба]].\\n"
        '\\n## Концовка\\n\\n[[Лиса]] съела.\\n"+\n}'
    )
    out, repairs = sanitize_body_md(body)
    parsed = parse_page(out)
    titles = [s.title for s in parsed.sections]
    assert titles == ["Сюжет", "Концовка"]
    assert "[[Старик|Дед]]" in out
    assert '"+' not in out
    assert "}" not in out.splitlines()[-1]
    # Все три типа правок отметились:
    kinds = {r.split(":", 1)[0] for r in repairs}
    assert "unescaped-literal-newlines" in kinds
    assert "trimmed-tail" in kinds
    assert "link-homograph" in kinds
