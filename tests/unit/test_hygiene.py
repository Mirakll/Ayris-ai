"""Гигиена репозитория: то, что ловится дешевле теста, но не ловится ничем.

Эти проверки жили в ``.pre-commit-config.yaml`` — то есть не выполнялись почти
никогда: хуки не установлены ни на одной машине, где идёт работа, а
``pre-commit run --all-files`` в CI стоил бы 44 с и завёл бы второй список
инструментов рядом с ruff и black. Здесь они стоят десятые доли секунды, идут в
каждом джобе и в каждом локальном прогоне и падают с объяснением, а не с diff-ом.

Что именно проверяется и почему это не паранойя:

* **Кодировка.** Консоль машины разработчика — cp1251, и редактор, спасающий
  файл «как есть», приносит в проект байты, на которых CI падает при импорте.
* **Маркеры конфликта.** ``<<<<<<<`` в закоммиченном файле — это красный на
  сборе тестов вида SyntaxError, по которому не видно, что произошёл
  недоделанный merge.
* **Размер.** Веса, архивы и десятимегабайтные wav в git не лечатся: история
  тянет их за собой навсегда, даже если файл удалить следующим коммитом.
* **Перевод строки в конце.** Один — не эстетика: без него следующая правка
  показывает последнюю строку изменённой, а не добавленную.
* **LF в индексе.** ``.gitattributes`` требует LF везде, кроме ``*.ps1``,
  ``*.bat``, ``*.cmd``, ``*.iss`` и ``*.reg``, где Windows нужен CRLF в рабочей
  копии. В индексе CRLF не должно быть ни у кого: иначе на другой машине diff
  показывает файл целиком.

yaml здесь намеренно не разбирается: ``pyyaml`` не стоит в requirements-ci.txt,
а тащить зависимость ради проверки гигиены — плохая сделка. Синтаксис workflow
проверяет сам GitHub, а важные для нас флаги внутри него читает как текст
tests/unit/test_packaging.py.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from functools import cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Потолок на файл в истории. Самый большой сейчас — 135 КБ (AYRIS_ROADMAP.md),
#: так что мегабайт — это запас, а не рамка: сработает он на случайно
#: закоммиченных весах или архиве, а не на выросшей документации.
MAX_TRACKED_BYTES = 1024 * 1024

#: Расширения, которые проект пишет сам и потому обязан держать в UTF-8.
TEXT_SUFFIXES = frozenset(
    {".py", ".pyi", ".md", ".toml", ".txt", ".yml", ".yaml", ".sh", ".ps1", ".json", ".cfg"}
)

#: Фикстуры — побайтовые копии чужих ответов, вместе с их кодировкой:
#: `tests/fixtures/api/cbr_daily.xml` объявляет windows-1251, потому что именно
#: так его отдаёт ЦБ. Перекодировать — значит проверять не то, что приходит.
FIXTURES = "tests/fixtures/"


@cache
def _git(*args: str) -> str | None:
    """Вывод git-команды или ``None``, если git недоступен.

    Список файлов берётся у git, а не обходом дерева: иначе в проверку попадёт
    `_tools/venv` с десятками тысяч файлов, стоит только списку исключений
    разойтись с `.gitignore`. Цена — пропуск всех проверок этого модуля там, где
    git не установлен или репозиторий распакован из архива.
    """
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", errors="replace")


@cache
def tracked_files() -> tuple[Path, ...]:
    """Файлы под контролем версий. Пустой кортеж — git не ответил."""
    raw = _git("ls-files", "-z")
    if raw is None:
        return ()
    return tuple(PROJECT_ROOT / name for name in raw.split("\0") if name)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _text_files() -> list[Path]:
    """Текстовые файлы проекта, без фикстур с чужими кодировками."""
    return [
        path
        for path in tracked_files()
        if path.suffix.lower() in TEXT_SUFFIXES
        and not _relative(path).startswith(FIXTURES)
        and path.is_file()
    ]


requires_git = pytest.mark.skipif(
    not tracked_files(), reason="git не ответил: список файлов взять негде"
)


@pytest.mark.unit
@requires_git
def test_every_text_file_is_utf8() -> None:
    """Files the project writes itself must decode as UTF-8.

    The console on the machine where this is developed is cp1251, and an editor
    saving a file "as it was" has already put bytes into the tree that CI cannot
    import. The failure looks like a SyntaxError in an unrelated module.
    """
    broken: dict[str, str] = {}
    for path in _text_files():
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            broken[_relative(path)] = str(exc)
    assert not broken, f"файлы не в UTF-8 (пересохрани редактором в UTF-8): {broken}"


@pytest.mark.unit
@requires_git
def test_no_merge_conflict_markers_are_committed() -> None:
    """A stray ``<<<<<<<`` is a syntax error nobody reads as an unfinished merge."""
    # Шаблоны собираются из символов, иначе тест находит сам себя.
    opening, closing = "<" * 7, ">" * 7
    guilty: dict[str, int] = {}
    for path in _text_files():
        text = path.read_bytes().decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.startswith((opening, closing)):
                guilty.setdefault(_relative(path), number)
    assert not guilty, f"маркеры конфликта в файлах (файл: строка): {guilty}"


@pytest.mark.unit
@requires_git
def test_no_tracked_file_is_huge() -> None:
    """Weights, archives and multi-megabyte audio do not belong in history.

    Git keeps them forever, even if the next commit deletes the file: every clone
    pays for the mistake. Big files belong in ``_tools/`` (gitignored) or behind
    ``scripts/fetch_models.py``, which downloads by url and checks sha256.
    """
    huge = {
        _relative(path): path.stat().st_size
        for path in tracked_files()
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES
    }
    assert not huge, (
        f"файлы больше {MAX_TRACKED_BYTES // 1024} КБ под контролем версий: {huge}. "
        "Веса и архивы качаются скриптом, а не лежат в истории."
    )


@pytest.mark.unit
@requires_git
def test_text_files_end_with_exactly_one_newline() -> None:
    """One trailing newline: no more, no less.

    Without it the next edit shows the last line as changed instead of the new one
    added, and ``git diff`` prints the ``\\ No newline at end of file`` marker on
    every review. Two of them are the same noise in the other direction.
    """
    missing: list[str] = []
    extra: list[str] = []
    for path in _text_files():
        raw = path.read_bytes()
        if not raw:
            continue
        if not raw.endswith(b"\n"):
            missing.append(_relative(path))
        elif raw.endswith((b"\n\n", b"\r\n\r\n")):
            extra.append(_relative(path))
    assert not missing, f"нет перевода строки в конце: {missing}"
    assert not extra, f"лишняя пустая строка в конце: {extra}"


@pytest.mark.unit
@requires_git
def test_config_files_parse() -> None:
    """``pyproject.toml`` and every json in the tree must actually load.

    A trailing comma in ``resources/models/*.json`` is a job that dies while
    downloading weights, twenty minutes in; a broken ``pyproject.toml`` is every
    job at once. Both are one cheap parse away from being caught at commit time.
    """
    broken: dict[str, str] = {}
    for path in tracked_files():
        if path.suffix not in {".toml", ".json"} or not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        try:
            if path.suffix == ".toml":
                tomllib.loads(text)
            else:
                json.loads(text)
        except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            broken[_relative(path)] = str(exc)
    assert not broken, f"файл не разбирается: {broken}"


@pytest.mark.unit
@requires_git
def test_the_index_holds_lf_only() -> None:
    """CRLF must never reach the index, whatever the working copy looks like.

    ``.gitattributes`` deliberately checks ``*.ps1``, ``*.bat``, ``*.cmd``,
    ``*.iss`` and ``*.reg`` out as CRLF — Windows tools need it — but that is a
    working-copy conversion. A file whose *index* contents are CRLF (or mixed)
    shows up as wholly rewritten in the diff on any other machine, and that is
    what ``core.autocrlf`` set wrong looks like from the outside.
    """
    raw = _git("ls-files", "--eol")
    if raw is None:
        pytest.skip("git ls-files --eol не ответил")
    wrong: list[str] = [line for line in raw.splitlines() if line.startswith(("i/crlf", "i/mixed"))]
    assert not wrong, (
        "в индексе не LF (проверь core.autocrlf=false и .gitattributes): "
        f"{[line.split(chr(9))[-1] for line in wrong]}"
    )
