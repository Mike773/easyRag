"""Шаг 1: парсинг wiki-markdown и извлечение ссылок.

Чистые юнит-тесты без БД и сети.
"""
from easyrag.wiki import (
    ExtractedLink,
    extract_links,
    make_slug,
    parse_page,
)


def test_make_slug_basic():
    assert make_slug("Договор поставки № 1") == "dogovor-postavki-no-1"
    assert make_slug("GigaChat & OpenAI") == "gigachat-openai"
    assert make_slug("hello   world") == "hello-world"


def test_make_slug_empty_fallbacks_to_stable_hash():
    s1 = make_slug("")
    s2 = make_slug("")
    s3 = make_slug("!!!")
    assert s1 == s2  # детерминированно
    assert s1.startswith("page-")
    assert s3.startswith("page-")
    # пустой ввод и «только символы» дают разные fallback-slug.
    assert s1 != s3


def test_extract_links_simple():
    links = extract_links("см. [[Договор]] и [[Контрагент|поставщик]] подробнее")
    assert links == [
        ExtractedLink(target="Договор", display="Договор", to_slug="dogovor"),
        ExtractedLink(target="Контрагент", display="поставщик", to_slug="kontragent"),
    ]


def test_extract_links_ignores_malformed():
    # одиночные скобки, пустые ссылки, переносы строк внутри — не считаем.
    assert extract_links("[[]] и [single] и [[a\nb]] и [[ ]]") == []


def test_parse_page_h2_sections_and_overview():
    body = (
        "Кратко о странице.\n\n"
        "## Стороны\n\n"
        "Поставщик: [[ООО Поставщик|Поставщик]].\n\n"
        "## Сроки\n\n"
        "Срок поставки 10 дней. См. [[Договор]].\n"
    )
    page = parse_page(body)
    titles = [s.title for s in page.sections]
    anchors = [s.anchor for s in page.sections]
    assert titles == ["Overview", "Стороны", "Сроки"]
    assert anchors == ["overview", "storony", "sroki"]
    # ord — последовательный от 0
    assert [s.ord for s in page.sections] == [0, 1, 2]
    # ссылки извлечены и нумерация общестраничная сохраняет порядок
    all_targets = [link.target for link in page.links]
    assert all_targets == ["ООО Поставщик", "Договор"]


def test_parse_page_anchor_collision_suffix():
    body = "## Раздел\n\nA\n\n## Раздел\n\nB\n\n## Раздел\n\nC\n"
    page = parse_page(body)
    assert [s.anchor for s in page.sections] == ["razdel", "razdel-2", "razdel-3"]


def test_parse_page_no_h2_treats_body_as_overview():
    body = "Только обычный текст с [[Сущностью]]."
    page = parse_page(body)
    assert len(page.sections) == 1
    s = page.sections[0]
    assert s.title == "Overview"
    assert s.anchor == "overview"
    assert s.body_md == "Только обычный текст с [[Сущностью]]."
    assert [link.target for link in s.links] == ["Сущностью"]


def test_parse_page_empty_body():
    page = parse_page("")
    assert page.sections == ()
    assert page.links == ()


def test_parse_page_section_without_preamble():
    body = "## Только секция\n\ntext\n"
    page = parse_page(body)
    assert len(page.sections) == 1
    assert page.sections[0].title == "Только секция"
    assert page.sections[0].body_md == "text"


def test_parse_page_links_per_section_preserve_order():
    body = (
        "## A\n\n[[X]] then [[Y|why]]\n\n"
        "## B\n\n[[Z]]\n"
    )
    page = parse_page(body)
    sec_a, sec_b = page.sections
    assert [link.target for link in sec_a.links] == ["X", "Y"]
    assert [link.display for link in sec_a.links] == ["X", "why"]
    assert [link.target for link in sec_b.links] == ["Z"]


def test_parse_page_ignores_fake_h2_inside_fenced_block():
    body = (
        "## Реальная\n\n"
        "Текст до блока.\n\n"
        "```\n"
        "## НЕ заголовок\n"
        "[[НЕ ссылка]]\n"
        "```\n\n"
        "После блока — [[Реальная цель]].\n"
    )
    page = parse_page(body)
    assert [s.title for s in page.sections] == ["Реальная"]
    # Тело секции сохраняет содержимое code-блока как есть.
    assert "## НЕ заголовок" in page.sections[0].body_md
    assert "[[НЕ ссылка]]" in page.sections[0].body_md
    # Из ссылок остаётся только настоящая.
    assert [link.target for link in page.links] == ["Реальная цель"]


def test_parse_page_ignores_links_inside_inline_code():
    body = "## A\n\nИспользуйте `[[Target]]` синтаксис, и [[Реальная]] тоже.\n"
    page = parse_page(body)
    sec = page.sections[0]
    assert [link.target for link in sec.links] == ["Реальная"]
    # body_md по-прежнему хранит inline-код целиком.
    assert "`[[Target]]`" in sec.body_md


def test_parse_page_unterminated_fence_is_left_as_text():
    # Не закрытый ```-блок не должен ломать парсер — H2 после него всё равно
    # ловится (мы маскируем только закрытые fenced-блоки).
    body = "## A\n\n```\nкод без закрывающего fence\n\n## B\n\nтекст\n"
    page = parse_page(body)
    assert [s.title for s in page.sections] == ["A", "B"]
