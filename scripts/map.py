"""Карта публичного API проекта: что где лежит, без чтения самих файлов.

Зачем это нужно. Задача начинается с разведки: чтобы дописать модуль, надо
понять, какие события уже есть, как называется нужное исключение, что торчит
наружу у соседнего модуля. Разведка вслепую стоит 5-10 открытых файлов по
500-900 строк каждый, и весь этот текст потом висит в контексте до конца работы.
Один прогон этого скрипта даёт ту же информацию строк за двести.

Скрипт разбирает исходники через :mod:`ast` и ничего не импортирует: он не
тянет PySide6, не открывает аудиоустройств и не падает на модуле, у которого
нет зависимостей в текущем окружении.

Использование:

    scripts/map.py                 индекс: модуль - первая строка docstring
    scripts/map.py models          сигнатуры всех модулей, чей путь содержит "models"
    scripts/map.py core/events.py  то же для одного файла
    scripts/map.py --events        события шины (dataclass'ы из core/events.py)
    scripts/map.py --errors        дерево исключений из core/errors.py
    scripts/map.py --all           сигнатуры по всему проекту (~1500 строк)

Приватное (имена с подчёркиванием) пропускается везде, кроме ``__init__`` и
``__enter__``/``__exit__``: они говорят, как объектом пользоваться.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[1]
_SRC: Final = _ROOT / "src" / "ayris"

# Дандеры, которые несут смысл для вызывающего кода: конструктор и протокол
# контекстного менеджера. Остальные (__repr__, __eq__) - шум.
_KEPT_DUNDERS: Final = frozenset({"__init__", "__enter__", "__exit__", "__iter__", "__call__"})


def _say(text: str = "") -> None:
    """Вывести строку. Вместо ``print``: кодировка задаётся здесь, не локалью."""
    sys.stdout.write(f"{text}\n")


def _summary(node: ast.AST) -> str:
    """Первая строка docstring узла - или пусто, если его нет."""
    if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return ""
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return ""
    return doc.split("\n", 1)[0].strip()


def _is_public(name: str) -> bool:
    return not name.startswith("_") or name in _KEPT_DUNDERS


def _signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Сигнатура функции так, как она написана в исходнике.

    ``ast.unparse`` сохраняет и позиционные-только параметры, и значения по
    умолчанию, и аннотации: то есть ровно то, что нужно знать вызывающему.
    """
    args = ast.unparse(func.args)
    ret = f" -> {ast.unparse(func.returns)}" if func.returns else ""
    prefix = "async " if isinstance(func, ast.AsyncFunctionDef) else ""
    return f"{prefix}{func.name}({args}){ret}"


def _decorators(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [ast.unparse(d) for d in node.decorator_list]


def _bases(cls: ast.ClassDef) -> str:
    parts = [ast.unparse(b) for b in cls.bases]
    parts += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in cls.keywords if kw.arg]
    return f"({', '.join(parts)})" if parts else ""


def _fields(cls: ast.ClassDef) -> list[str]:
    """Аннотированные атрибуты класса - поля dataclass'а или модели pydantic."""
    out: list[str] = []
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not _is_public(node.target.id):
                continue
            value = f" = {ast.unparse(node.value)}" if node.value else ""
            out.append(f"{node.target.id}: {ast.unparse(node.annotation)}{value}")
    return out


def _rel(path: Path) -> str:
    return path.relative_to(_SRC).as_posix()


def _modules(pattern: str | None) -> list[Path]:
    """Файлы модулей, отсортированные по пути; пустые ``__init__`` отброшены."""
    found = sorted(p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts)
    if pattern:
        needle = pattern.replace("\\", "/").lower()
        found = [p for p in found if needle in _rel(p).lower()]
    return [p for p in found if p.stat().st_size > 0]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _print_index(paths: list[Path]) -> None:
    """Индекс: путь и первая строка docstring. Дешёвый способ понять, где искать."""
    width = max((len(_rel(p)) for p in paths), default=0)
    for path in paths:
        tree = _parse(path)
        doc = _summary(tree) or "(без docstring)"
        _say(f"{_rel(path):<{width}}  {doc}")


def _print_class(cls: ast.ClassDef, indent: str = "  ") -> None:
    doc = _summary(cls)
    deco = "".join(f"@{d} " for d in _decorators(cls))
    _say(f"{indent}{deco}class {cls.name}{_bases(cls)}" + (f"  # {doc}" if doc else ""))

    for field in _fields(cls):
        _say(f"{indent}    {field}")

    for node in cls.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not _is_public(node.name):
            continue
        marks = [d for d in _decorators(node) if d in {"property", "staticmethod", "classmethod"}]
        mark = f"@{marks[0]} " if marks else ""
        method_doc = _summary(node)
        line = f"{indent}    {mark}{_signature(node)}"
        _say(line + (f"  # {method_doc}" if method_doc else ""))


def _print_module(path: Path) -> None:
    tree = _parse(path)
    header = _rel(path)
    doc = _summary(tree)
    _say(f"\n=== {header} ===" + (f"\n{doc}" if doc else ""))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            _print_class(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_public(node.name):
            fn_doc = _summary(node)
            _say(f"  {_signature(node)}" + (f"  # {fn_doc}" if fn_doc else ""))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # Модульные константы: SAMPLE_RATE, KNOWN_SLOTS и прочее, на что
            # ссылаются соседние модули.
            if _is_public(node.target.id):
                _say(f"  {node.target.id}: {ast.unparse(node.annotation)}")


def _print_events() -> None:
    """Все события шины: имя, поля, назначение. Читается вместо events.py (875 строк)."""
    tree = _parse(_SRC / "core" / "events.py")
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            _print_class(node, indent="")


def _print_errors() -> None:
    """Дерево исключений: от кого наследуется каждое, чтобы не плодить своё."""
    tree = _parse(_SRC / "core" / "errors.py")
    children: dict[str, list[ast.ClassDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            parent = ast.unparse(node.bases[0]) if node.bases else ""
            children.setdefault(parent, []).append(node)

    def walk(parent: str, depth: int) -> None:
        for cls in children.get(parent, []):
            doc = _summary(cls)
            _say("  " * depth + cls.name + (f"  # {doc}" if doc else ""))
            walk(cls.name, depth + 1)

    for root in sorted(set(children) - {c.name for group in children.values() for c in group}):
        walk(root, 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Карта публичного API Ayris: модули, классы, сигнатуры.",
        epilog="Без аргументов печатает индекс модулей.",
    )
    parser.add_argument("pattern", nargs="?", help="часть пути модуля, например models или events")
    parser.add_argument("--all", action="store_true", help="сигнатуры по всему проекту")
    parser.add_argument("--events", action="store_true", help="события шины из core/events.py")
    parser.add_argument("--errors", action="store_true", help="дерево исключений из core/errors.py")
    args = parser.parse_args()

    if args.events:
        _print_events()
        return 0
    if args.errors:
        _print_errors()
        return 0

    paths = _modules(args.pattern)
    if not paths:
        _say(f"под «{args.pattern}» не подошёл ни один модуль")
        return 1

    if args.pattern or args.all:
        for path in paths:
            _print_module(path)
    else:
        _print_index(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
