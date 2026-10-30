"""Собрать настройки Obsidian 2c из партиалов; --check проверяет без записи.

Старая версия 2b меняется только при явном --target 2b-obsidian-violet.html.
Оболочка, стили, главная и чат редактируются непосредственно в целевом HTML.
"""

import argparse
import pathlib
import re
import sys
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = ("2c-obsidian-chat.html", "2b-obsidian-violet.html")
KEYS = (
    "voice", "commands", "ai", "hotkeys", "overlay",
    "profiles", "updates", "devtools", "privacy",
)


class Sections(HTMLParser):
    """Границы содержимого секций; вложенные <section> тоже учитываются."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=False)
        self.offsets = [0]
        self.offsets.extend(match.end() for match in re.finditer("\n", html))
        self.stack: list[tuple[str | None, int]] = []
        self.spans: dict[str, list[tuple[int, int]]] = {}
        self.feed(html)
        self.close()
        if self.stack:
            raise ValueError("Незакрытый тег <section> в целевом HTML")

    def absolute_offset(self) -> int:
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            key = dict(attrs).get("id")
            self.stack.append((key, self.absolute_offset() + len(self.get_starttag_text())))

    def handle_endtag(self, tag: str) -> None:
        if tag == "section":
            if not self.stack:
                raise ValueError("Лишний тег </section> в целевом HTML")
            key, start = self.stack.pop()
            if key:
                self.spans.setdefault(key, []).append((start, self.absolute_offset()))


def assemble(html: str) -> tuple[str, list[str]]:
    """Сначала проверить все партиалы, затем вернуть готовый документ."""
    sections = Sections(html)
    newline = "\r\n" if "\r\n" in html else "\n"
    replacements = []
    for key in KEYS:
        part = ROOT / "_sections" / f"{key}.html"
        if not part.is_file():
            raise ValueError(f"Нет партиала: {part.name}")
        spans = sections.spans.get(f"sp-{key}", [])
        if len(spans) != 1:
            raise ValueError(f"Ожидалась одна секция sp-{key}, найдено {len(spans)}")
        fragment = part.read_text(encoding="utf-8").strip()
        if not fragment:
            raise ValueError(f"Пустой партиал: {part.name}")
        Sections(fragment)
        start, end = spans[0]
        if html[start:end].replace("\r\n", "\n").strip() != fragment:
            body = newline + fragment.replace("\n", newline) + newline
            replacements.append((start, end, body, key))
    for start, end, body, _ in sorted(replacements, reverse=True):
        html = html[:start] + body + html[end:]
    return html, [key for _, _, _, key in replacements]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    parser.add_argument("--check", action="store_true", help="Проверить без записи")
    args = parser.parse_args()
    target = ROOT / args.target
    try:
        original = target.read_bytes()
        html, changed = assemble(original.decode("utf-8"))
        if not changed:
            print(f"OK: {target.name} — все {len(KEYS)} партиалов совпадают")
            return 0
        if args.check:
            print(f"Нужна сборка: {target.name} — {', '.join(changed)}")
            return 1
        if target.read_bytes() != original:
            raise ValueError("Целевой HTML изменился во время сборки; повторите запуск")
        target.write_bytes(html.encode("utf-8"))
        print(f"Собрано: {target.name} — {', '.join(changed)}")
        return 0
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Ошибка: {error}. Сборка остановлена.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
