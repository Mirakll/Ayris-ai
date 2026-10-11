"""Какие файлы моделей лежат на диске — список для встроенного llama.cpp.

Облачный провайдер и локальные серверы (Ollama, LM Studio) отвечают на вопрос
«какие у тебя модели» сами — по HTTP. У встроенного llama.cpp спрашивать некого:
он открывает конкретный ``.gguf`` по пути, и единственный честный список — это
файлы, которые действительно лежат на диске. Этот модуль их и собирает, чтобы
пикер на вкладке «ИИ» показывал скачанные модели, а не одну строку с текущим
именем.

**Имя — это стем файла, а не метка из каталога.** Каталог рекомендаций
(:mod:`~ayris.nlu.llm.catalog`) знает лишь горстку моделей; на диске у человека
лежит что угодно, включая то, чего в каталоге нет и не будет. Поэтому имя берётся
из имени файла, а совпадения стемов из разных папок разводятся суффиксом с именем
папки — иначе две разные модели слились бы в один пункт списка.

**Обход мелкий и дешёвый.** Смотрим саму папку и её прямые подпапки (менеджер
моделей кладёт загрузку в подкаталог), не глубже: пикер открывают интерактивно, и
рекурсия по чужому диску с сетевыми томами там неуместна. Любая ошибка чтения —
это «папки нет» и пустой список, а не исключение в UI-потоке.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ayris.utils.logger import get_logger

__all__ = [
    "GGUF_SUFFIX",
    "LocalModel",
    "discover_local_models",
    "local_model_dirs",
]

_log = get_logger(__name__)

#: Расширение файла модели встроенного движка llama.cpp.
GGUF_SUFFIX = ".gguf"

#: Потолок на число найденных моделей: список открывается мышкой, и пятьсот
#: пунктов в выпадашке — это не выбор, а зависший интерфейс.
MAX_LOCAL_MODELS = 64


@dataclass(frozen=True, slots=True)
class LocalModel:
    """Один найденный файл модели: как показать и что записать в конфиг."""

    name: str
    path: Path
    size_bytes: int = 0

    @property
    def human_size(self) -> str:
        """Размер файла строкой «≈1.7 ГБ» для подписи под пикером."""
        gb = self.size_bytes / (1024**3)
        if gb >= 1.0:
            return f"≈{gb:.1f} ГБ"
        mb = self.size_bytes / (1024**2)
        return f"≈{mb:.0f} МБ"


def local_model_dirs(
    *,
    models_dir: Path | str | None = None,
    current_path: Path | str = "",
    extra: Iterable[Path | str] = (),
) -> tuple[Path, ...]:
    """Где искать ``.gguf``: папка профиля, папка текущей модели и указанные явно.

    Папка текущей модели важнее, чем кажется: человек, который однажды выбрал
    файл кнопкой «Указать .gguf», держит там и остальные свои модели, и без этой
    папки список показал бы ему всё, кроме того, чем он пользуется.

    Дубликаты и несуществующие пути отсеиваются, порядок сохраняется.
    """
    candidates: list[Path | str] = []
    if models_dir:
        candidates.append(models_dir)
    if current_path:
        candidates.append(Path(current_path).parent)
    candidates.extend(extra)

    seen: dict[Path, None] = {}
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = Path(candidate)
            if not resolved.is_dir():
                continue
        except OSError:  # сетевой том отвалился — просто не ищем там
            continue
        seen.setdefault(resolved, None)
    return tuple(seen)


def discover_local_models(dirs: Iterable[Path | str]) -> tuple[LocalModel, ...]:
    """Найти файлы моделей в ``dirs``, по одному пункту на файл.

    Результат отсортирован по имени (регистр не важен) и ограничен
    :data:`MAX_LOCAL_MODELS`. Один и тот же файл, встреченный через разные папки,
    попадает в список один раз; одинаковые имена из разных папок разводятся
    суффиксом «(папка)».
    """
    found: dict[Path, LocalModel] = {}
    taken: dict[str, Path] = {}
    for file in _walk(dirs):
        if file in found:
            continue
        name = _unique_name(file, taken)
        taken[name] = file
        found[file] = LocalModel(name=name, path=file, size_bytes=_size_of(file))
        if len(found) >= MAX_LOCAL_MODELS:
            _log.info("файлов моделей больше %d — список обрезан", MAX_LOCAL_MODELS)
            break
    return tuple(sorted(found.values(), key=lambda model: model.name.casefold()))


def _walk(dirs: Iterable[Path | str]) -> Iterator[Path]:
    """Файлы ``.gguf`` в каждой папке и её прямых подпапках, без рекурсии глубже."""
    for raw in dirs:
        directory = Path(raw)
        for pattern in (f"*{GGUF_SUFFIX}", f"*/*{GGUF_SUFFIX}"):
            try:
                entries = sorted(directory.glob(pattern))
            except OSError:
                _log.debug("не удалось прочитать папку моделей %s", directory, exc_info=True)
                continue
            for entry in entries:
                if entry.is_file():
                    yield entry


def _unique_name(file: Path, taken: dict[str, Path]) -> str:
    """Стем файла, а при столкновении — со скобкой с именем папки или номером."""
    stem = file.stem
    if stem not in taken:
        return stem
    scoped = f"{stem} ({file.parent.name})"
    if scoped not in taken:
        return scoped
    index = 2
    while f"{scoped} {index}" in taken:
        index += 1
    return f"{scoped} {index}"


def _size_of(file: Path) -> int:
    """Размер файла, или ``0``, если его не спросить (битая ссылка, нет прав)."""
    try:
        return file.stat().st_size
    except OSError:
        return 0
