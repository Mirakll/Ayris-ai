"""Файловые действия: поиск, чтение и безопасные изменения в разрешённых папках.

Любой путь сначала проходит :func:`_resolve_inside` — операция касается только
тех корней, что пользователь сам перечислил в ``[actions.files] allowed_dirs``.
Пустой список означает «доступа нет»: действие честно отказывает, а не идёт по
всему диску. Проверка — разрешающий список, а не запреты, и путь резолвится и
лексически, и по-настоящему (``realpath``), поэтому символической ссылкой из
разрешённой папки нельзя дотянуться наружу.

Чтение (список, поиск, текст, сведения) безвредно и идёт прямо по файловой
системе. Изменения (удалить, переместить, скопировать, создать папку) спрятаны
за швом :class:`FileSystem`, чтобы тест не трогал настоящий диск; удаление и
перемещение помечены ``is_dangerous`` и спрашивают подтверждение, а удаление по
умолчанию отправляет в корзину, а не стирает насовсем.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.core.config import FilesActionsConfig

__all__ = [
    "CopyFile",
    "CreateFolder",
    "DeleteFile",
    "FileEntry",
    "FileFacts",
    "FileInfo",
    "FileSystem",
    "FindFiles",
    "ListDir",
    "MoveFile",
    "ReadTextFile",
    "RealFileSystem",
    "RecordingFileSystem",
    "get_file_system",
    "set_file_system",
]

_log = get_logger(__name__)

#: Encodings tried in order when reading text (Russian Windows is cp1251).
_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1251")

#: Cap on how many entries a listing or search returns, so a huge tree does not
#: flood the spoken answer or the model's context.
_MAX_ENTRIES: int = 1000


@dataclass(frozen=True, slots=True)
class FileEntry:
    """Одна строка листинга или результата поиска."""

    name: str
    path: str
    is_dir: bool
    size_bytes: int

    def as_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FileFacts:
    """Сведения об одном файле или папке для :class:`FileInfo`."""

    path: str
    exists: bool
    is_dir: bool
    size_bytes: int
    modified_unix: float

    def as_data(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_dir": self.is_dir,
            "size_bytes": self.size_bytes,
            "modified_unix": self.modified_unix,
        }


def _config() -> FilesActionsConfig:
    """Live ``[actions.files]`` settings; re-read every call (hot-reloadable)."""
    from ayris.core.config import get_settings

    return get_settings().actions.files


def _allowed_roots(config: FilesActionsConfig) -> list[Path]:
    """Configured roots as fully-resolved absolute paths; bad entries dropped."""
    roots: list[Path] = []
    for raw in config.allowed_dirs:
        try:
            roots.append(Path(str(raw)).expanduser().resolve())
        except (OSError, ValueError):
            continue
    return roots


def _within(path: Path, roots: Sequence[Path]) -> bool:
    """Whether ``path`` is one of the roots or lives under one of them."""
    return any(path == root or path.is_relative_to(root) for root in roots)


def _resolve_inside(raw: str, *, must_exist: bool = False) -> Path:
    """Разрешить путь и убедиться, что он внутри разрешённого корня.

    Пустой ``allowed_dirs`` — доступа нет. Путь проверяется и лексически
    (``abspath``, нормализует ``..`` без перехода по ссылкам), и по-настоящему
    (``realpath``); оба обязаны лежать внутри корня, поэтому ни ``..``, ни
    символическая ссылка не выведут наружу. При ``follow_symlinks=False``
    расхождение этих двух путей — признак перехода по ссылке, и он отклоняется.
    """
    config = _config()
    roots = _allowed_roots(config)
    if not roots:
        raise ActionError(
            "no allowed_dirs configured",
            user_message="Доступ к файлам не настроен. Разрешите папку в настройках.",
        )
    text = str(raw).strip()
    if not text:
        raise ActionError("empty path", user_message="Не указан путь.")
    base = Path(text).expanduser()
    try:
        lexical = Path(os.path.abspath(base))  # noqa: PTH100 — свернуть .. без перехода по ссылкам
        real = base.resolve()
    except (OSError, ValueError) as exc:
        raise ActionError(f"bad path {text!r}", user_message="Не разобрала путь.") from exc
    if not _within(lexical, roots) or not _within(real, roots):
        raise ActionError(
            f"path outside allowed roots: {lexical}",
            user_message="Эта папка не разрешена в настройках.",
        )
    if not config.follow_symlinks and lexical.is_symlink():
        raise ActionError(
            f"symlink refused: {lexical}",
            user_message="Переход по символической ссылке запрещён настройками.",
        )
    if must_exist and not lexical.exists():
        raise ActionError(f"not found: {lexical}", user_message="Не нашла такой файл или папку.")
    return lexical


def _entry_of(path: Path) -> FileEntry:
    """One listing row; a stat failure degrades to zero size, never raises."""
    try:
        is_dir = path.is_dir()
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        is_dir = False
        size = 0
    return FileEntry(name=path.name, path=str(path), is_dir=is_dir, size_bytes=size)


def _decode_text(data: bytes) -> str:
    """Decode bytes trying UTF-8 then cp1251; last resort replaces bad bytes."""
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _human_size(size: int) -> str:
    """Short Russian size label, e.g. «3.4 МБ»."""
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            return f"{int(value)} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} Б"


class FileSystem(Protocol):
    """ОС в объёме изменяющих файловых операций — без решений, только вызовы."""

    def trash(self, path: str) -> None: ...
    def remove(self, path: str, *, recursive: bool) -> None: ...
    def copy(self, source: str, destination: str) -> None: ...
    def move(self, source: str, destination: str) -> None: ...
    def make_dir(self, path: str) -> None: ...


class RealFileSystem:
    """Настоящие изменения: ``shutil`` для копирования и корзина через ``send2trash``.

    ``send2trash`` импортируется лениво: на линуксовом джобе CI его может не быть,
    а модуль обязан импортироваться там всё равно — изменяющие тесты подставляют
    :class:`RecordingFileSystem` и настоящего диска не касаются.
    """

    def trash(self, path: str) -> None:
        try:
            from send2trash import send2trash
        except ImportError as exc:
            raise ActionUnavailable(
                "send2trash is not installed",
                user_message="Удаление в корзину недоступно: не установлен компонент send2trash.",
            ) from exc
        try:
            send2trash(path)
        except OSError as exc:
            raise ActionError(
                f"send2trash failed: {exc}", user_message="Не смогла отправить в корзину."
            ) from exc

    def remove(self, path: str, *, recursive: bool) -> None:
        target = Path(path)
        try:
            if target.is_dir() and not target.is_symlink():
                if recursive:
                    shutil.rmtree(target)
                else:
                    target.rmdir()
            else:
                target.unlink()
        except OSError as exc:
            raise ActionError(f"remove failed: {exc}", user_message="Не смогла удалить.") from exc

    def copy(self, source: str, destination: str) -> None:
        src = Path(source)
        try:
            if src.is_dir():
                shutil.copytree(src, destination)
            else:
                shutil.copy2(source, destination)
        except OSError as exc:
            raise ActionError(f"copy failed: {exc}", user_message="Не смогла скопировать.") from exc

    def move(self, source: str, destination: str) -> None:
        try:
            shutil.move(source, destination)
        except OSError as exc:
            raise ActionError(f"move failed: {exc}", user_message="Не смогла переместить.") from exc

    def make_dir(self, path: str) -> None:
        try:
            Path(path).mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise ActionError(
                f"mkdir failed: {exc}", user_message="Не смогла создать папку."
            ) from exc


@dataclass(slots=True)
class RecordingFileSystem:
    """Пишет вызов в ``calls`` и ничего не делает — для тестов и dry-run."""

    calls: list[tuple[object, ...]] = field(default_factory=list)

    def trash(self, path: str) -> None:
        self.calls.append(("trash", path))

    def remove(self, path: str, *, recursive: bool) -> None:
        self.calls.append(("remove", path, recursive))

    def copy(self, source: str, destination: str) -> None:
        self.calls.append(("copy", source, destination))

    def move(self, source: str, destination: str) -> None:
        self.calls.append(("move", source, destination))

    def make_dir(self, path: str) -> None:
        self.calls.append(("make_dir", path))


_backend: FileSystem | None = None
_default: FileSystem = RealFileSystem()


def get_file_system() -> FileSystem:
    """Активный бэкенд изменений; по умолчанию настоящий (как у яркости/браузера)."""
    return _backend if _backend is not None else _default


def set_file_system(backend: FileSystem | None) -> None:
    """Поставить бэкенд (тест ставит запись); ``None`` возвращает настоящий."""
    global _backend
    _backend = backend


@register
class ListDir(Action):
    """Содержимое папки: файлы и вложенные папки."""

    meta: ClassVar = ActionMeta(
        name="ListDir",
        category=ActionCategory.FILES,
        title_ru="Список файлов",
        description_ru="Показать содержимое папки",
    )

    class Params(ActionParams):
        path: str = Field(
            min_length=1,
            max_length=32_767,
            description="Папка, содержимое которой показать",
        )

    def run(self, params: Params) -> ActionResult[tuple[FileEntry, ...]]:
        target = _resolve_inside(params.path, must_exist=True)
        if not target.is_dir():
            return ActionResult.failed("Это не папка.", detail=f"not a directory: {target}")
        try:
            children = sorted(
                target.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())
            )
        except OSError as exc:
            raise ActionError(
                f"iterdir failed: {exc}", user_message="Не смогла прочитать папку."
            ) from exc
        entries = tuple(_entry_of(item) for item in children[:_MAX_ENTRIES])
        dirs = sum(1 for entry in entries if entry.is_dir)
        files = len(entries) - dirs
        return ActionResult.done(
            f"В папке {dirs} папок и {files} файлов.",
            value=entries,
            detail=f"listed {len(entries)} entries in {target}",
            data={"path": str(target), "dirs": dirs, "files": files, "count": len(entries)},
        )


@register
class FindFiles(Action):
    """Поиск файлов по маске имени внутри разрешённой папки."""

    meta: ClassVar = ActionMeta(
        name="FindFiles",
        category=ActionCategory.FILES,
        title_ru="Найти файлы",
        description_ru="Поиск по маске имени в папке и подпапках",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        root: str = Field(min_length=1, max_length=32_767, description="Где искать")
        pattern: str = Field(
            default="*",
            max_length=256,
            title="Маска",
            description="Маска имени, например *.pdf или отчёт*",
        )
        recursive: bool = Field(
            default=True, title="В подпапках", description="Искать и во вложенных папках"
        )
        limit: int = Field(default=100, ge=1, le=_MAX_ENTRIES, title="Сколько максимум")

    def run(self, params: Params) -> ActionResult[tuple[FileEntry, ...]]:
        root = _resolve_inside(params.root, must_exist=True)
        if not root.is_dir():
            return ActionResult.failed("Это не папка.", detail=f"not a directory: {root}")
        globber = root.rglob if params.recursive else root.glob
        try:
            found: list[Path] = []
            for match in globber(params.pattern):
                found.append(match)
                if len(found) >= params.limit:
                    break
        except (OSError, ValueError) as exc:
            raise ActionError(
                f"glob failed: {exc}", user_message="Не смогла выполнить поиск."
            ) from exc
        entries = tuple(_entry_of(item) for item in found)
        if entries:
            message = f"Нашла {len(entries)} совпадений."
        else:
            message = f"Ничего не нашла по маске «{params.pattern}»."
        return ActionResult.done(
            message,
            value=entries,
            detail=f"found {len(entries)} for {params.pattern!r} in {root}",
            data={"root": str(root), "pattern": params.pattern, "count": len(entries)},
        )


@register
class ReadTextFile(Action):
    """Прочитать текстовый файл (с лимитом размера из настроек)."""

    meta: ClassVar = ActionMeta(
        name="ReadTextFile",
        category=ActionCategory.FILES,
        title_ru="Прочитать файл",
        description_ru="Прочитать текст из файла",
    )

    class Params(ActionParams):
        path: str = Field(min_length=1, max_length=32_767, description="Файл для чтения")

    def run(self, params: Params) -> ActionResult[str]:
        target = _resolve_inside(params.path, must_exist=True)
        if not target.is_file():
            return ActionResult.failed("Это не файл.", detail=f"not a file: {target}", value="")
        cap = _config().max_read_bytes
        try:
            with target.open("rb") as handle:
                chunk = handle.read(cap + 1)
        except OSError as exc:
            raise ActionError(
                f"read failed: {exc}", user_message="Не смогла прочитать файл."
            ) from exc
        truncated = len(chunk) > cap
        text = _decode_text(chunk[:cap])
        note = " (прочитала не весь файл — слишком большой)" if truncated else ""
        return ActionResult.done(
            f"Прочитала {target.name}{note}.",
            value=text,
            detail=f"read {min(len(chunk), cap)} bytes from {target}",
            data={"path": str(target), "bytes": min(len(chunk), cap), "truncated": truncated},
        )


@register
class FileInfo(Action):
    """Сведения о файле или папке: размер, дата изменения, тип."""

    meta: ClassVar = ActionMeta(
        name="FileInfo",
        category=ActionCategory.FILES,
        title_ru="Сведения о файле",
        description_ru="Размер, дата изменения и тип файла или папки",
    )

    class Params(ActionParams):
        path: str = Field(min_length=1, max_length=32_767, description="Файл или папка")

    def run(self, params: Params) -> ActionResult[FileFacts]:
        target = _resolve_inside(params.path)
        try:
            stat = target.stat()
        except OSError:
            missing = FileFacts(
                path=str(target), exists=False, is_dir=False, size_bytes=0, modified_unix=0.0
            )
            return ActionResult.failed(
                "Такого файла или папки нет.", detail=f"missing: {target}", value=missing
            )
        facts = FileFacts(
            path=str(target),
            exists=True,
            is_dir=target.is_dir(),
            size_bytes=stat.st_size,
            modified_unix=stat.st_mtime,
        )
        kind = "папка" if facts.is_dir else "файл"
        size = "" if facts.is_dir else f", {_human_size(facts.size_bytes)}"
        return ActionResult.done(
            f"{target.name}: {kind}{size}.",
            value=facts,
            detail=f"stat {target}",
            data=facts.as_data(),
        )


@register
class DeleteFile(Action):
    """Удалить файл или папку — по умолчанию в корзину."""

    meta: ClassVar = ActionMeta(
        name="DeleteFile",
        category=ActionCategory.FILES,
        title_ru="Удалить файл",
        description_ru="Отправить файл или папку в корзину (или удалить насовсем)",
        is_dangerous=True,
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        path: str = Field(
            min_length=1, max_length=32_767, description="Файл или папка для удаления"
        )
        recursive: bool = Field(
            default=False,
            title="Со всем содержимым",
            description="Удалять папку вместе с содержимым",
        )
        permanent: bool = Field(
            default=False,
            title="Насовсем",
            description="Удалить мимо корзины, без возможности восстановить",
        )

    def run(self, params: Params) -> ActionResult[str]:
        target = _resolve_inside(params.path, must_exist=True)
        to_trash = _config().delete_to_trash and not params.permanent
        backend = get_file_system()
        if to_trash:
            backend.trash(str(target))
            return ActionResult.done(
                f"Отправила {target.name} в корзину.",
                value=str(target),
                detail=f"trashed {target}",
                data={"path": str(target), "trash": True},
            )
        backend.remove(str(target), recursive=params.recursive)
        return ActionResult.done(
            f"Удалила {target.name} насовсем.",
            value=str(target),
            detail=f"removed {target}",
            data={"path": str(target), "trash": False},
        )


@register
class MoveFile(Action):
    """Переместить или переименовать файл либо папку."""

    meta: ClassVar = ActionMeta(
        name="MoveFile",
        category=ActionCategory.FILES,
        title_ru="Переместить файл",
        description_ru="Переместить или переименовать файл или папку",
        is_dangerous=True,
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        source: str = Field(min_length=1, max_length=32_767, description="Что перемещаем")
        destination: str = Field(
            min_length=1, max_length=32_767, description="Куда: новый путь или имя"
        )
        overwrite: bool = Field(
            default=False,
            title="Перезаписать",
            description="Заменить файл в месте назначения, если он уже есть",
        )

    def run(self, params: Params) -> ActionResult[str]:
        src = _resolve_inside(params.source, must_exist=True)
        dst = _resolve_inside(params.destination)
        if dst.exists() and not params.overwrite:
            return ActionResult.failed(
                "В месте назначения уже есть такой файл.",
                detail=f"destination exists: {dst}",
                value=str(dst),
            )
        get_file_system().move(str(src), str(dst))
        return ActionResult.done(
            f"Переместила {src.name}.",
            value=str(dst),
            detail=f"moved {src} -> {dst}",
            data={"source": str(src), "destination": str(dst)},
        )


@register
class CopyFile(Action):
    """Скопировать файл или папку."""

    meta: ClassVar = ActionMeta(
        name="CopyFile",
        category=ActionCategory.FILES,
        title_ru="Скопировать файл",
        description_ru="Создать копию файла или папки",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        source: str = Field(min_length=1, max_length=32_767, description="Что копируем")
        destination: str = Field(min_length=1, max_length=32_767, description="Куда копируем")
        recursive: bool = Field(
            default=False,
            title="Со всем содержимым",
            description="Копировать папку вместе с содержимым",
        )

    def run(self, params: Params) -> ActionResult[str]:
        src = _resolve_inside(params.source, must_exist=True)
        dst = _resolve_inside(params.destination)
        if src.is_dir() and not params.recursive:
            return ActionResult.failed(
                "Это папка — включите «со всем содержимым».",
                detail=f"directory without recursive: {src}",
                value=str(src),
            )
        if dst.exists():
            return ActionResult.failed(
                "В месте назначения уже есть такой файл.",
                detail=f"destination exists: {dst}",
                value=str(dst),
            )
        get_file_system().copy(str(src), str(dst))
        return ActionResult.done(
            f"Скопировала {src.name}.",
            value=str(dst),
            detail=f"copied {src} -> {dst}",
            data={"source": str(src), "destination": str(dst)},
        )


@register
class CreateFolder(Action):
    """Создать папку."""

    meta: ClassVar = ActionMeta(
        name="CreateFolder",
        category=ActionCategory.FILES,
        title_ru="Создать папку",
        description_ru="Создать новую папку",
    )

    class Params(ActionParams):
        path: str = Field(min_length=1, max_length=32_767, description="Путь новой папки")

    def run(self, params: Params) -> ActionResult[str]:
        target = _resolve_inside(params.path)
        if target.exists():
            return ActionResult.failed(
                "Такая папка уже есть.", detail=f"already exists: {target}", value=str(target)
            )
        get_file_system().make_dir(str(target))
        return ActionResult.done(
            f"Создала папку {target.name}.",
            value=str(target),
            detail=f"mkdir {target}",
            data={"path": str(target)},
        )
