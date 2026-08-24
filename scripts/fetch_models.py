"""Скачать веса, на которых идут тесты с маркером ``models``.

Зачем отдельный скрипт. Тесты с этим маркером проверяют то, чего заглушка
проверить не может: что GigaAM действительно распознаёт фикстуру, что Piper
действительно выдаёт звук, что openWakeWord вообще загружается. Весов в
репозитории нет и не будет — один GigaAM это 215 МБ, — поэтому раньше эти тесты
не запускались нигде, кроме машины разработчика, и пять настоящих багов в путях к
моделям всплыли только тогда, когда я эти веса наконец скачал.

Скрипт качает их **через собственный каталог проекта** (`resources/models/*.json`,
`ModelCatalog` → `Downloader` → `Installer`), а не своим кодом с curl. Так
проверяется заодно и вся эта машинерия: адреса живы, sha256 совпадают, архив
распаковывается, файлы ложатся под теми именами, которые потом ищут движки.
Именно последнее и было сломано у openWakeWord.

Использование:

    python scripts/fetch_models.py                       # в .models-cache/
    python scripts/fetch_models.py --root D:/веса        # куда угодно
    python scripts/fetch_models.py --skip-whisper        # без гигабайтного CT2
    eval "$(python scripts/fetch_models.py --quiet)"     # и сразу в окружение

Готовые модели не перекачиваются, так что повторный запуск на закэшированной
папке ничего не тянет из сети. Переменные `AYRIS_TEST_*` уходят в `$GITHUB_ENV`,
если он задан, иначе печатаются как `export`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final, NamedTuple

_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from ayris.models.catalog import ModelEntry, catalog_dir, load_catalog  # noqa: E402
from ayris.models.downloader import Downloader, human_size  # noqa: E402
from ayris.models.installer import Installer  # noqa: E402

#: Куда по умолчанию. Рядом с проектом и в `.gitignore`, чтобы `actions/cache`
#: указывал на один путь и на раннере, и здесь.
DEFAULT_ROOT: Final = _ROOT / ".models-cache"

#: Модель CTranslate2 для faster-whisper. В каталоге её нет: экспорт CT2 — это
#: каталог из четырёх файлов, и до появления `ModelEntry.directory` положить их
#: было некуда (`ModelFile.target` запрещает разделители пути, а плоско — значит
#: столкнуть два разных whisper'а на общем `config.json`). Теперь подпапки есть,
#: но перевод whisper на каталог — отдельная задача; здесь по-прежнему HF.
WHISPER_REPO: Final = "Systran/faster-whisper-tiny"
WHISPER_DIR_NAME: Final = "faster-whisper-tiny"


class Wanted(NamedTuple):
    """Одна запись каталога и переменная окружения, которая на неё показывает.

    Attributes:
        catalog_id: `id` из `resources/models/*.json`.
        variable: Имя `AYRIS_TEST_*`, или пустая строка, если модель нужна как
            часть чего-то другого — мел-спектрограмма openWakeWord сама по себе
            ни одному тесту не нужна, но без неё не грузится ни одна фраза.
        points_at_dir: Показывать на папку вида, а не на сам файл. Так устроен
            `AYRIS_TEST_WAKE_MODELS`: движок ищет в папке по имени фразы.
    """

    catalog_id: str
    variable: str = ""
    points_at_dir: bool = False


#: Что качаем. Порядок — от нужного всем к необязательному.
WANTED: Final[tuple[Wanted, ...]] = (
    # Движок распознавания по умолчанию. Показывает на саму папку, а не на файл
    # внутри: `onnx-asr` берёт каталог и находит в нём веса с словарём сам.
    Wanted("gigaam-v3-ctc", "AYRIS_TEST_GIGAAM_MODEL"),
    Wanted("vosk-ru-small", "AYRIS_TEST_STT_MODEL"),
    Wanted("piper-ru-irina", "AYRIS_TEST_PIPER_VOICE"),
    Wanted("oww-melspectrogram"),
    Wanted("oww-embedding"),
    Wanted("oww-hey-jarvis", "AYRIS_TEST_WAKE_MODELS", points_at_dir=True),
)


def _progress(model_id: str, downloaded: int, total: int, _speed: float, _eta: float) -> None:
    """Одна строка на модель, и только по завершении.

    В логе Actions «\\r» не перерисовывает строку, а добавляет новую, поэтому
    живой прогресс-бар превращается в сотни строк мусора.
    """
    if total and downloaded >= total:
        print(f"    {model_id}: {human_size(total)}")


def _fetch_catalog_models(
    root: Path, wanted: tuple[Wanted, ...], *, work_dir: Path
) -> dict[str, str]:
    """Установить записи каталога в `root` и вернуть переменные окружения."""
    catalog = load_catalog(catalog_dir(), strict=True)
    installer = Installer(root)
    variables: dict[str, str] = {}

    with Downloader(work_dir) as downloader:
        for item in wanted:
            entry: ModelEntry = catalog.require(item.catalog_id)
            destination = installer.destination(entry)
            if installer.is_installed(entry):
                print(f"  есть: {entry.id} → {destination.name}")
            else:
                print(f"  качаю {entry.id} ({human_size(entry.total_bytes)})")
                downloads = downloader.fetch_all(entry, on_progress=_progress)
                result = installer.install(entry, downloads)
                destination = result.path
                print(f"    установлено в {destination}")
            if item.variable:
                target = destination.parent if item.points_at_dir else destination
                variables[item.variable] = str(target)
    return variables


def _fetch_whisper(root: Path) -> dict[str, str]:
    """Забрать экспорт CT2 с Hugging Face.

    `huggingface_hub` уже стоит — это зависимость faster-whisper, — поэтому
    отдельного пакета для этого не нужно. Модель кладётся распакованной папкой,
    а не в кэш HF: тесту нужен путь, а не идентификатор репозитория.
    """
    destination = root / "stt" / WHISPER_DIR_NAME
    if (destination / "model.bin").is_file():
        print(f"  есть: {WHISPER_REPO} → {destination.name}")
        return {"AYRIS_TEST_WHISPER_MODEL": str(destination)}

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub не установлен, whisper пропущен", file=sys.stderr)
        return {}

    print(f"  качаю {WHISPER_REPO}")
    snapshot_download(
        WHISPER_REPO,
        local_dir=destination,
        allow_patterns=["*.json", "*.bin", "*.txt"],
    )
    print(f"    установлено в {destination}")
    return {"AYRIS_TEST_WHISPER_MODEL": str(destination)}


def _publish(variables: dict[str, str], *, quiet: bool) -> None:
    """Отдать переменные тому, кто нас позвал: Actions или шеллу."""
    github_env = os.environ.get("GITHUB_ENV", "")
    if github_env:
        with Path(github_env).open("a", encoding="utf-8") as handle:
            for name, value in variables.items():
                handle.write(f"{name}={value}\n")
        if not quiet:
            print(f"\n{len(variables)} переменных записано в GITHUB_ENV")
        return
    for name, value in variables.items():
        print(f'export {name}="{value}"')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="куда ставить модели")
    parser.add_argument("--skip-whisper", action="store_true", help="без экспорта CT2 с HF")
    parser.add_argument("--quiet", action="store_true", help="только строки export, без отчёта")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        print(f"веса → {root}")

    stdout = sys.stdout
    if args.quiet:
        # Отчёт уходит в stderr, чтобы `eval "$(...)"` получил только export.
        sys.stdout = sys.stderr
    try:
        variables = _fetch_catalog_models(root, WANTED, work_dir=root / ".downloads")
        if not args.skip_whisper:
            variables |= _fetch_whisper(root)
    finally:
        sys.stdout = stdout

    _publish(variables, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
