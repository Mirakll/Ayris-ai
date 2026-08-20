"""Проверка, что установленные версии совпадают с пинами, которые ставил CI.

Зачем не `pip check`. Он там стоял и не мог пройти никогда: джоб с весами ставит
и `ayris`, и `pyrnnoise` флагом `--no-deps`, а `pip check` считает каждую
недостающую транзитивную зависимость ошибкой. В его выхлопе двенадцать строк
вида «ayris требует anthropic, который не установлен» — все двенадцать
намеренные, — и в них тонет единственная строка, которая действительно важна.

А важна вот какая. `openwakeword` тянет scipy и scikit-learn, те требуют
`numpy>=2.3`, и pip молча поднимает numpy поверх пина 1.26.4. Джоб при этом
зеленеет, но проверяет уже не то окружение, в котором работает приложение:
`vosk` и `faster-whisper` собраны против ABI numpy 1.x. Файл ограничений
(`-c requirements-ci.txt`) обязан это предотвратить, и именно его работу здесь
и подтверждаем — сравнением того, что просили, с тем, что стоит.

Проверяются ровно те пакеты, которые CI перечислил в `requirements-ci*.txt`.
Пакеты, которых там нет (`pluggy` и `requests` из зависимостей приложения,
подтянутые попутно чужими колёсами), не проверяются: CI их не ставил, и их
версия — не его обещание.

    python scripts/verify_pins.py            # все три файла
    python scripts/verify_pins.py -v         # и совпадения тоже печатать
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, NamedTuple

_ROOT: Final = Path(__file__).resolve().parents[1]

#: Файлы, из которых CI ставит пакеты. Порядок — как в workflow: следующий может
#: перекрыть версию предыдущего, и тогда обещанием считается последняя.
REQUIREMENTS: Final[tuple[str, ...]] = (
    "requirements-ci.txt",
    "requirements-ci-nodeps.txt",
    "requirements-ci-models.txt",
)

#: ``name==version`` с отброшенным хвостом из extras и маркеров.
_PIN: Final = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s*==\s*(?P<version>[^\s;#]+)")

#: Маркер платформы: `; sys_platform == 'win32'` и подобное.
_MARKER: Final = re.compile(r";\s*sys_platform\s*==\s*['\"](?P<platform>[a-z0-9]+)['\"]")


class Pin(NamedTuple):
    """Один пин и то, откуда он взялся."""

    name: str
    wanted: str
    source: str


def normalize(name: str) -> str:
    """Нормализация имени по PEP 503: `PySide6` и `pyside6` — один пакет."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(paths: tuple[str, ...]) -> dict[str, Pin]:
    """Пины из файлов требований, с учётом маркера платформы.

    Строка с `; sys_platform == 'win32'` на линуксе пропускается: pip её тоже
    пропустил, и требовать установленный pycaw там значит красить джоб зря.
    """
    pins: dict[str, Pin] = {}
    for name in paths:
        path = _ROOT / name
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _PIN.match(line)
            if match is None:
                continue
            marker = _MARKER.search(line)
            if marker is not None and marker["platform"] != sys.platform:
                continue
            key = normalize(match["name"])
            pins[key] = Pin(name=key, wanted=match["version"], source=name)
    return pins


def installed_version(name: str) -> str | None:
    """Версия установленного пакета или `None`, если его нет."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="печатать и совпадения")
    args = parser.parse_args(argv)

    pins = read_pins(REQUIREMENTS)
    if not pins:
        print("не нашёл ни одного пина — файлы требований пропали?", file=sys.stderr)
        return 2

    wrong: list[tuple[Pin, str]] = []
    absent: list[Pin] = []
    for pin in sorted(pins.values()):
        actual = installed_version(pin.name)
        if actual is None:
            absent.append(pin)
        elif actual != pin.wanted:
            wrong.append((pin, actual))
        elif args.verbose:
            print(f"  ok       {pin.name}=={actual}")

    for pin in absent:
        print(f"НЕ СТОИТ  {pin.name}=={pin.wanted} (из {pin.source})", file=sys.stderr)
    for pin, actual in wrong:
        print(
            f"РАЗОШЛОСЬ {pin.name}: просили {pin.wanted} (из {pin.source}), стоит {actual}",
            file=sys.stderr,
        )

    if wrong or absent:
        print(
            f"\nокружение не то, что заявлено: разошлось {len(wrong)}, не установлено {len(absent)}",
            file=sys.stderr,
        )
        return 1
    print(f"окружение то, что заявлено: {len(pins)} пакетов совпали")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
