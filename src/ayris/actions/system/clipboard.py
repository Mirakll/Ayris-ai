"""The clipboard: reading it, writing it, and the history of what passed through.

Four actions live here — ``ClipboardSet``, ``ClipboardGet``, ``ClipboardHistory``
and ``ClipboardPaste`` — plus the single wrapper the whole win32 clipboard layer
hides behind.

**Why a wrapper.** ``OpenClipboard`` is one exclusive lock for the entire desktop
and it genuinely fails: right after a Ctrl+C the source program, a password
manager and Windows' own history service all reach for it. Every call in
:mod:`ayris.utils.winapi` already retries, and :class:`ClipboardBackend` puts one
more thing on top — a seam. Deduplication, eviction with pinned entries kept,
length limits, newest-first numbering and «wipe the clipboard after pasting a
password» are all policy, and policy is worth testing on a machine with no
clipboard at all. :class:`FakeClipboard` is what the tests drive, and it is the
same object shape the real one has.

**What the history stores.** Text, and nothing else. A picture and a set of
dragged files are recognised — :class:`ClipboardKind` reports which — but never
copied into the database: a bitmap has no useful preview to read out, and a
history that quietly holds a megabyte per copy is a different feature. Over-long
text is skipped rather than truncated, because a truncated entry looks fine in
the list and pastes half a document.

**Numbering.** «Вставь третий» counts from the newest entry, always, whatever is
pinned — see :meth:`ClipboardRepository.history`. The ordinal itself is an NLU
slot; this module takes an integer and never parses Russian numerals.
"""

from __future__ import annotations

import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.input.backend import get_input_backend
from ayris.actions.input.keys import parse_combo, press_combo
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.config import get_settings
from ayris.core.database import get_database
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.core.repositories import Repositories
from ayris.utils import winapi
from ayris.utils.logger import forget_secret, get_logger, guard_secret

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.core.config import ClipboardActionsConfig
    from ayris.core.models import ClipboardEntry
    from ayris.core.repositories import ClipboardRepository

__all__ = [
    "EXCLUSION_FORMATS",
    "ClipboardBackend",
    "ClipboardBusy",
    "ClipboardEntryView",
    "ClipboardGet",
    "ClipboardHistory",
    "ClipboardKind",
    "ClipboardPaste",
    "ClipboardSet",
    "ClipboardSnapshot",
    "FakeClipboard",
    "WinClipboard",
    "clipboard_settings",
    "get_clipboard",
    "get_clipboard_store",
    "record_clipboard",
    "reset_clipboard",
    "reset_clipboard_store",
    "set_clipboard",
    "set_clipboard_store",
    "suppress_record",
    "windows_history_enabled",
]

_log = get_logger(__name__)

#: The two formats a password manager sets to ask clipboard monitors to look
#: away. ``Clipboard Viewer Ignore`` is the old convention every manager honours;
#: ``ExcludeClipboardContentFromMonitorProcessing`` is the one Windows itself
#: documents and what Win+V history obeys. Both are read as markers only — their
#: payload is irrelevant, presence is the whole message.
EXCLUSION_FORMATS: Final = (
    "Clipboard Viewer Ignore",
    "ExcludeClipboardContentFromMonitorProcessing",
    "CanIncludeInClipboardHistory",
)

#: How long to wait after Ctrl+V before putting the previous clipboard back. The
#: receiving program reads the clipboard when it processes the keystroke, which is
#: on its own message loop, not ours.
_PASTE_SETTLE_S: Final = 0.12


class ClipboardKind(StrEnum):
    """What is on the clipboard right now.

    Only :attr:`TEXT` is ever stored. The others exist so ``ClipboardGet`` can say
    «в буфере картинка» instead of «буфер пуст», which is a different answer and
    the one a person needs to hear.
    """

    EMPTY = "empty"
    TEXT = "text"
    IMAGE = "image"
    FILES = "files"
    OTHER = "other"

    @property
    def title_ru(self) -> str:
        return _KIND_TITLES[self]


_KIND_TITLES: Final[dict[ClipboardKind, str]] = {
    ClipboardKind.EMPTY: "пусто",
    ClipboardKind.TEXT: "текст",
    ClipboardKind.IMAGE: "изображение",
    ClipboardKind.FILES: "файлы",
    ClipboardKind.OTHER: "другое содержимое",
}


class ClipboardBusy(ActionError):
    """The clipboard stayed locked by another process through every retry."""


@dataclass(frozen=True, slots=True)
class ClipboardSnapshot:
    """One look at the clipboard, as the policy layer needs to see it."""

    kind: ClipboardKind = ClipboardKind.EMPTY
    text: str = ""
    files: tuple[str, ...] = ()
    excluded: bool = False
    sequence: int = 0

    @property
    def is_text(self) -> bool:
        return self.kind is ClipboardKind.TEXT and bool(self.text)

    def describe_ru(self) -> str:
        """What is on the clipboard, in a sentence. Never quotes secret-looking text."""
        if self.kind is ClipboardKind.FILES:
            if len(self.files) == 1:
                return f"В буфере файл: {self.files[0]}"
            return f"В буфере {len(self.files)} файлов"
        if self.kind is ClipboardKind.IMAGE:
            return "В буфере изображение"
        if self.kind is ClipboardKind.EMPTY:
            return "Буфер обмена пуст"
        if self.kind is ClipboardKind.OTHER:
            return "В буфере содержимое, которое Айрис не умеет читать"
        return f"В буфере {len(self.text)} символов текста"


class ClipboardBackend(ABC):
    """Everything Ayris does to the Windows clipboard, and nothing more.

    Four methods, so that a fake is four methods long. Anything more interesting
    — what counts as a duplicate, which entry is «третий», when to wipe the
    clipboard — belongs above this line, where it can be tested.
    """

    @abstractmethod
    def read(self) -> ClipboardSnapshot:
        """What is on the clipboard now."""

    @abstractmethod
    def write_text(self, text: str) -> None:
        """Replace the clipboard contents with ``text``."""

    @abstractmethod
    def clear(self) -> None:
        """Empty the clipboard."""

    @abstractmethod
    def sequence(self) -> int:
        """Windows' clipboard change counter, or ``0`` where there is none."""


class WinClipboard(ClipboardBackend):
    """The real thing: win32 first, ``pyperclip`` only if win32 is not there.

    The win32 path is the one that matters — it is the only one that can tell a
    picture from text, see the password-manager exclusion markers, or read the
    change counter, and it is the one that retries when the clipboard is locked.
    ``pyperclip`` is kept as a text-only fallback for a developer machine that is
    not Windows; it is imported inside the method so a missing package is never an
    import-time failure.
    """

    def read(self) -> ClipboardSnapshot:
        if not winapi.available():
            return self._fallback_read()
        try:
            data = winapi.read_clipboard(blobs=self._exclusion_formats())
        except winapi.WinApiError as error:
            raise self._busy(error) from error
        return _snapshot_from(data, self._exclusion_formats())

    def write_text(self, text: str) -> None:
        if not winapi.available():
            self._fallback_write(text)
            return
        try:
            winapi.clipboard_set_text(text)
        except winapi.WinApiError as error:
            raise self._busy(error) from error

    def clear(self) -> None:
        if not winapi.available():
            self._fallback_write("")
            return
        try:
            winapi.clipboard_clear()
        except winapi.WinApiError as error:
            raise self._busy(error) from error

    def sequence(self) -> int:
        if not winapi.available():
            return 0
        return winapi.clipboard_sequence_number()

    @staticmethod
    def _busy(error: winapi.WinApiError) -> ClipboardBusy:
        """Turn a WinAPI refusal into the one message a user can act on."""
        return ClipboardBusy(
            f"clipboard locked by another process: {error}",
            user_message=("Буфер обмена занят другой программой. Подождите секунду и повторите."),
        )

    @staticmethod
    def _exclusion_formats() -> tuple[int, ...]:
        """Ids of :data:`EXCLUSION_FORMATS`, registering them on first use."""
        ids = []
        for name in EXCLUSION_FORMATS:
            registered = winapi.register_clipboard_format(name)
            if registered:
                ids.append(registered)
        return tuple(ids)

    @staticmethod
    def _fallback_read() -> ClipboardSnapshot:
        try:
            import pyperclip
        except ImportError:  # pragma: no cover - pyperclip is a pinned dependency
            raise _no_clipboard() from None
        try:
            text = str(pyperclip.paste())
        except Exception as error:  # pyperclip raises its own, undeclared types
            raise _no_clipboard() from error
        kind = ClipboardKind.TEXT if text else ClipboardKind.EMPTY
        return ClipboardSnapshot(kind=kind, text=text)

    @staticmethod
    def _fallback_write(text: str) -> None:
        try:
            import pyperclip
        except ImportError:  # pragma: no cover - pyperclip is a pinned dependency
            raise _no_clipboard() from None
        try:
            pyperclip.copy(text)
        except Exception as error:
            raise _no_clipboard() from error


class FakeClipboard(ClipboardBackend):
    """A clipboard in a variable. What the policy tests run against.

    ``busy_reads`` and ``busy_writes`` make the next N calls fail the way a locked
    clipboard does, which is how the retry-then-explain path gets covered on a
    machine that has no clipboard to lock.
    """

    def __init__(self, snapshot: ClipboardSnapshot | None = None) -> None:
        self.snapshot = snapshot if snapshot is not None else ClipboardSnapshot()
        self.busy_reads = 0
        self.busy_writes = 0
        self.writes: list[str] = []
        self.clears = 0
        self._sequence = 1 if snapshot is not None else 0

    def read(self) -> ClipboardSnapshot:
        if self.busy_reads > 0:
            self.busy_reads -= 1
            raise _fake_busy()
        return self.snapshot

    def write_text(self, text: str) -> None:
        if self.busy_writes > 0:
            self.busy_writes -= 1
            raise _fake_busy()
        self.writes.append(text)
        self._sequence += 1
        kind = ClipboardKind.TEXT if text else ClipboardKind.EMPTY
        self.snapshot = ClipboardSnapshot(kind=kind, text=text, sequence=self._sequence)

    def clear(self) -> None:
        self.clears += 1
        self._sequence += 1
        self.snapshot = ClipboardSnapshot(sequence=self._sequence)

    def sequence(self) -> int:
        return self._sequence

    def put(self, snapshot: ClipboardSnapshot) -> None:
        """Set what a read will see, the way another program would."""
        self._sequence += 1
        self.snapshot = ClipboardSnapshot(
            kind=snapshot.kind,
            text=snapshot.text,
            files=snapshot.files,
            excluded=snapshot.excluded,
            sequence=self._sequence,
        )


def _fake_busy() -> ClipboardBusy:
    return ClipboardBusy(
        "clipboard locked by another process (fake)",
        user_message="Буфер обмена занят другой программой. Подождите секунду и повторите.",
    )


def _no_clipboard() -> ActionUnavailable:
    return ActionUnavailable(
        "no clipboard implementation available on this platform",
        user_message="Буфер обмена недоступен: эта операция работает только в Windows.",
    )


def _snapshot_from(data: winapi.ClipboardData, exclusion_ids: Sequence[int]) -> ClipboardSnapshot:
    """Classify a raw win32 read. The one place format ids turn into meaning."""
    excluded = _is_excluded(data, exclusion_ids)
    if data.has(winapi.CF_HDROP) and data.files:
        return ClipboardSnapshot(
            kind=ClipboardKind.FILES,
            files=data.files,
            excluded=excluded,
            sequence=data.sequence,
        )
    if any(data.has(fmt) for fmt in winapi.IMAGE_FORMATS):
        return ClipboardSnapshot(
            kind=ClipboardKind.IMAGE, excluded=excluded, sequence=data.sequence
        )
    if data.text:
        return ClipboardSnapshot(
            kind=ClipboardKind.TEXT, text=data.text, excluded=excluded, sequence=data.sequence
        )
    kind = ClipboardKind.OTHER if data.formats else ClipboardKind.EMPTY
    return ClipboardSnapshot(kind=kind, excluded=excluded, sequence=data.sequence)


def _is_excluded(data: winapi.ClipboardData, exclusion_ids: Sequence[int]) -> bool:
    """Whether whoever filled the clipboard asked monitors to ignore it.

    Two of the three markers mean «ignore me» by merely being present. The third,
    ``CanIncludeInClipboardHistory``, carries a DWORD and inverts the question: a
    zero means no. A manager that sets it to one is explicitly allowing storage,
    so only the zero is treated as a refusal.
    """
    allow_id = winapi.register_clipboard_format(EXCLUSION_FORMATS[2])
    for fmt in exclusion_ids:
        if not data.has(fmt):
            continue
        if fmt == allow_id and allow_id:
            payload = data.blobs.get(fmt, b"")
            if payload and int.from_bytes(payload[:4], "little") != 0:
                continue
        return True
    return False


# --------------------------------------------------------------------------- #
# Process-wide backend and history store
# --------------------------------------------------------------------------- #

_backend: ClipboardBackend | None = None
_backend_lock = threading.Lock()
_store: ClipboardRepository | None = None


def get_clipboard() -> ClipboardBackend:
    """The clipboard in force, building the real one on first use."""
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = WinClipboard()
        return _backend


def set_clipboard(backend: ClipboardBackend | None) -> None:
    """Install a backend, or restore the real one with ``None``. Test seam."""
    global _backend
    with _backend_lock:
        _backend = backend


def reset_clipboard() -> None:
    """Forget the cached backend, so the next call builds a fresh one."""
    set_clipboard(None)


def get_clipboard_store() -> ClipboardRepository:
    """The ``clipboard_history`` repository, opening the database on first use."""
    global _store
    if _store is None:
        _store = Repositories(get_database()).clipboard
    return _store


def set_clipboard_store(store: ClipboardRepository | None) -> None:
    """Point the history at another repository. Test and profile-switch seam."""
    global _store
    _store = store


def reset_clipboard_store() -> None:
    """Forget the cached repository — after a profile switch closed its database."""
    set_clipboard_store(None)


def clipboard_settings() -> ClipboardActionsConfig:
    """The ``[actions.clipboard]`` section."""
    return get_settings().actions.clipboard


def windows_history_enabled() -> bool:
    """Whether Win+V clipboard history is on, as far as the registry admits.

    Worth knowing because it means a second copy of everything Ayris records
    exists outside Ayris — and, with cloud sync, outside the machine. The settings
    tab shows a warning; nothing here changes behaviour because of it.
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg
    except ImportError:  # pragma: no cover - winreg ships with Windows Python
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Clipboard") as key:
            value, _ = winreg.QueryValueEx(key, "EnableClipboardHistory")
    except OSError:
        return False
    return bool(value)


# --------------------------------------------------------------------------- #
# Recording an entry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RecordOutcome:
    """Why a clipboard change did or did not become a history entry.

    Every rejection has its own reason so that a DEBUG log line explains a
    history that stayed empty, which is otherwise the hardest thing here to
    diagnose: the value cannot be printed, so the reason has to be enough.
    """

    stored: bool = False
    reason: str = ""
    entry_id: int | None = None


#: Values Ayris itself just put on the clipboard, waiting to be ignored once.
#:
#: Pasting entry three writes it to the clipboard, and the monitor sees that write
#: exactly as it would see a Ctrl+C. Without this, «вставь третий» would move that
#: entry to the top of the history and renumber everything the user had just been
#: read — so the next «вставь третий» would land somewhere else. Bounded, because a
#: suppression nobody claims (the monitor is off, the write failed) must not
#: accumulate; each entry is claimed at most once, so a genuine later copy of the
#: same text is still recorded.
_suppressed: dict[str, None] = {}
_suppressed_lock = threading.Lock()
_MAX_SUPPRESSED: Final = 8


def suppress_record(text: str) -> None:
    """Have the monitor ignore the next clipboard change that carries ``text``."""
    if not text:
        return
    with _suppressed_lock:
        _suppressed.pop(text, None)
        _suppressed[text] = None
        while len(_suppressed) > _MAX_SUPPRESSED:
            del _suppressed[next(iter(_suppressed))]


def _claim_suppressed(text: str) -> bool:
    """Whether ``text`` was one of ours, consuming the suppression if so."""
    with _suppressed_lock:
        return _suppressed.pop(text, "missing") is None


def record_clipboard(
    snapshot: ClipboardSnapshot,
    *,
    store: ClipboardRepository | None = None,
    settings: ClipboardActionsConfig | None = None,
    secret: bool = False,
) -> RecordOutcome:
    """Put one clipboard change into the history, if it belongs there.

    The whole policy of the monitor, in one function with no I/O of its own beyond
    the repository — which is what makes it testable without a clipboard, a window
    or a thread:

    * only text is stored, and only text that is not blank;
    * a value identical to the newest entry is a duplicate, and a Ctrl+C pressed
      twice must not fill the list with the same line;
    * anything longer than ``max_length`` is skipped, not truncated;
    * a value marked secret, or coming from a program that set an exclusion
      marker, is never written at all;
    * a value Ayris itself just wrote while pasting is not a new copy;
    * after a successful write, unpinned entries beyond ``limit`` are evicted.
    """
    config = settings if settings is not None else clipboard_settings()
    repository = store if store is not None else get_clipboard_store()

    if secret:
        return RecordOutcome(reason="secret")
    if snapshot.excluded and config.skip_password_managers:
        return RecordOutcome(reason="excluded")
    if not snapshot.is_text:
        return RecordOutcome(reason=f"kind:{snapshot.kind.value}")
    text = snapshot.text
    if not text.strip():
        return RecordOutcome(reason="blank")
    if _claim_suppressed(text):
        return RecordOutcome(reason="self")
    if len(text) > config.max_length:
        return RecordOutcome(reason="too-long")

    newest = repository.newest()
    if newest is not None and newest.content == text:
        return RecordOutcome(reason="duplicate", entry_id=newest.id)

    entry = repository.add(text)
    evicted = repository.trim_to_limit(config.limit)
    if evicted:
        _log.debug("clipboard history evicted %d entries", evicted)
    return RecordOutcome(stored=True, entry_id=entry.id)


# --------------------------------------------------------------------------- #
# The history as the voice and the editor see it
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ClipboardEntryView:
    """One history entry, numbered and shortened for reading out.

    ``number`` counts from the newest entry and starts at one, because that is
    what «вставь третий» means. It is a position in the answer, not a database id:
    ``entry_id`` is the one that survives the next copy.
    """

    number: int
    entry_id: int
    preview: str
    length: int
    pinned: bool = False
    kind: ClipboardKind = ClipboardKind.TEXT
    ts_ru: str = ""

    def describe_ru(self) -> str:
        pin = "★ " if self.pinned else ""
        stamp = f" ({self.ts_ru})" if self.ts_ru else ""
        return f"{self.number}. {pin}{self.preview}{stamp}"


def _preview(text: str, limit: int) -> str:
    """One line, at most ``limit`` characters, ending in an ellipsis when cut."""
    single = " ".join(text.split())
    if len(single) <= limit:
        return single
    return single[: max(1, limit - 1)].rstrip() + "…"


def _view_of(entry: ClipboardEntry, number: int, limit: int) -> ClipboardEntryView:
    return ClipboardEntryView(
        number=number,
        entry_id=entry.id or 0,
        preview=_preview(entry.content, limit),
        length=len(entry.content),
        pinned=entry.pinned,
        ts_ru=entry.ts.astimezone().strftime("%H:%M") if entry.ts is not None else "",
    )


def history_views(
    *,
    limit: int = 0,
    query: str = "",
    pinned_only: bool = False,
    store: ClipboardRepository | None = None,
    settings: ClipboardActionsConfig | None = None,
) -> tuple[ClipboardEntryView, ...]:
    """The history, newest first, numbered from one.

    Numbering is assigned *after* filtering, so «вставь второй» matches the second
    line of the list the user was just read, not the second entry overall.
    """
    config = settings if settings is not None else clipboard_settings()
    repository = store if store is not None else get_clipboard_store()
    count = limit if limit > 0 else config.limit
    entries = repository.history(limit=count, query=query, pinned_only=pinned_only)
    return tuple(
        _view_of(entry, number, config.preview_length)
        for number, entry in enumerate(entries, start=1)
    )


def entry_by_number(
    number: int,
    *,
    store: ClipboardRepository | None = None,
) -> ClipboardEntry | None:
    """The ``number``-th entry counting from the newest, or ``None``."""
    if number < 1:
        return None
    repository = store if store is not None else get_clipboard_store()
    entries = repository.history(limit=max(number, 1))
    if len(entries) < number:
        return None
    return entries[number - 1]


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


@register
class ClipboardSet(Action):
    """Put text on the clipboard.

    ``secret=True`` is the caller saying «this is a password»: the value is kept
    out of the audit row, out of the log and out of the history, and the clipboard
    is wiped after ``clear_after_s`` seconds so it does not sit there until the
    next copy.
    """

    meta: ClassVar = ActionMeta(
        name="ClipboardSet",
        category=ActionCategory.CLIPBOARD,
        title_ru="Положить в буфер",
        description_ru="Помещает текст в буфер обмена.",
    )

    class Params(ActionParams):
        text: str = Field(max_length=1_000_000, description="Текст для буфера обмена")
        secret: bool = Field(
            default=False,
            description="Не писать значение в историю, аудит и логи",
        )
        clear_after_s: float = Field(
            default=0.0,
            ge=0.0,
            le=600.0,
            description="Через сколько секунд очистить буфер (0 — не очищать)",
        )

        def secret_now(self) -> frozenset[str]:
            return frozenset({"text"}) if self.secret else frozenset()

    def run(self, params: Params) -> ActionResult[int]:
        clipboard = get_clipboard()
        if params.secret:
            guard_secret(params.text)
            suppress_record(params.text)
        clipboard.write_text(params.text)
        if params.secret:
            _log.info("clipboard set to a secret value (%d characters)", len(params.text))
        else:
            _log.debug("clipboard set to %d characters", len(params.text))

        if params.clear_after_s > 0:
            _schedule_clear(clipboard, params.clear_after_s, params.text if params.secret else "")
            return ActionResult.done(
                f"Скопировано в буфер, очищу через {int(params.clear_after_s)} с",
                value=len(params.text),
            )
        return ActionResult.done("Скопировано в буфер обмена", value=len(params.text))


@register
class ClipboardGet(Action):
    """Read the clipboard: the text, or what kind of thing is there instead."""

    meta: ClassVar = ActionMeta(
        name="ClipboardGet",
        category=ActionCategory.CLIPBOARD,
        title_ru="Прочитать буфер",
        description_ru="Читает буфер обмена: текст либо тип содержимого.",
    )

    class Params(ActionParams):
        speak_limit: int = Field(
            default=400,
            ge=0,
            le=4000,
            description="Сколько символов озвучивать (0 — только тип содержимого)",
        )

    def run(self, params: Params) -> ActionResult[str]:
        snapshot = get_clipboard().read()
        data: dict[str, Any] = {
            "kind": snapshot.kind.value,
            "length": len(snapshot.text),
            "files": list(snapshot.files),
        }
        if not snapshot.is_text:
            return ActionResult.done(snapshot.describe_ru(), value="", data=data)
        spoken = (
            _preview(snapshot.text, params.speak_limit)
            if params.speak_limit
            else snapshot.describe_ru()
        )
        return ActionResult.done(spoken, value=snapshot.text, data=data)


@register
class ClipboardHistory(Action):
    """List the clipboard history, or pin, unpin, delete and clear its entries.

    One action rather than five: the operations share the numbering, and a macro
    that lists entries and then pins one should not have to know two block names
    to do it. ``number`` is always a position in the newest-first list.
    """

    meta: ClassVar = ActionMeta(
        name="ClipboardHistory",
        category=ActionCategory.CLIPBOARD,
        title_ru="История буфера",
        description_ru="Показывает историю копирований, закрепляет и удаляет записи.",
    )

    class Params(ActionParams):
        operation: str = Field(
            default="list",
            pattern="^(list|pin|unpin|delete|clear)$",
            description="Что сделать: list, pin, unpin, delete, clear",
        )
        number: int = Field(
            default=0,
            ge=0,
            le=1000,
            description="Номер записи для pin/unpin/delete, считая от самой свежей",
        )
        query: str = Field(default="", max_length=200, description="Искать по подстроке")
        pinned_only: bool = Field(default=False, description="Только закреплённые записи")
        limit: int = Field(default=0, ge=0, le=1000, description="Сколько записей вернуть")
        include_pinned: bool = Field(
            default=False,
            description="При очистке удалять и закреплённые записи",
        )

    def run(self, params: Params) -> ActionResult[Any]:
        store = get_clipboard_store()
        if params.operation == "clear":
            removed = store.clear(keep_pinned=not params.include_pinned)
            kept = store.count(pinned=True) if not params.include_pinned else 0
            message = f"История очищена: удалено записей — {removed}"
            if kept:
                message += f", закреплённых оставлено — {kept}"
            return ActionResult.done(message, value=removed, data={"kept_pinned": kept})

        if params.operation == "list":
            views = history_views(
                limit=params.limit, query=params.query, pinned_only=params.pinned_only, store=store
            )
            return ActionResult.done(
                _describe_history(views, query=params.query, pinned_only=params.pinned_only),
                value=[view.describe_ru() for view in views],
                data={"entries": [view.entry_id for view in views], "count": len(views)},
            )

        entry = entry_by_number(params.number, store=store)
        if entry is None or entry.id is None:
            return ActionResult.failed(
                f"В истории буфера нет записи номер {params.number}",
                detail=f"number={params.number}",
            )
        if params.operation == "delete":
            store.delete(entry.id)
            return ActionResult.done(f"Запись {params.number} удалена", value=entry.id)
        pinned = params.operation == "pin"
        store.set_pinned(entry.id, pinned=pinned)
        word = "закреплена" if pinned else "откреплена"
        return ActionResult.done(f"Запись {params.number} {word}", value=entry.id)


@register
class ClipboardPaste(Action):
    """Paste a history entry: «вставь третий».

    The number comes from an NLU slot already parsed into an integer — the ordinal
    «третий» is a language problem and belongs in the recogniser, not here.

    The entry is put on the clipboard and Ctrl+V is synthesised through the input
    backend of task 23, because there is no way to make another program insert
    text without going through its own paste handler. What was on the clipboard
    before is restored afterwards when it was text, so a paste does not silently
    eat what the user had copied.
    """

    meta: ClassVar = ActionMeta(
        name="ClipboardPaste",
        category=ActionCategory.CLIPBOARD,
        title_ru="Вставить из истории",
        description_ru="Вставляет запись истории буфера по номеру, например «вставь третий».",
    )

    class Params(ActionParams):
        number: int = Field(default=1, ge=1, le=1000, description="Номер записи от самой свежей")
        paste: bool = Field(default=True, description="Нажать Ctrl+V, а не только положить в буфер")
        restore: bool = Field(default=True, description="Вернуть прежнее содержимое буфера")
        secret: bool = Field(
            default=False,
            description="Считать значение секретным: очистить буфер сразу после вставки",
        )

    def run(self, params: Params) -> ActionResult[int]:
        store = get_clipboard_store()
        entry = entry_by_number(params.number, store=store)
        if entry is None or entry.id is None:
            return ActionResult.failed(
                f"В истории буфера нет записи номер {params.number}",
                detail=f"number={params.number}",
            )

        clipboard = get_clipboard()
        previous = ""
        if params.restore and not params.secret:
            try:
                snapshot = clipboard.read()
            except ClipboardBusy:
                snapshot = ClipboardSnapshot()
            previous = snapshot.text if snapshot.is_text else ""

        if params.secret:
            guard_secret(entry.content)
        suppress_record(entry.content)
        clipboard.write_text(entry.content)
        if not params.paste:
            return ActionResult.done(
                f"Запись {params.number} в буфере обмена", value=len(entry.content)
            )

        paste_shortcut()
        if params.secret and clipboard_settings().clear_after_secret:
            clipboard.clear()
            forget_secret(entry.content)
        elif previous:
            suppress_record(previous)
            clipboard.write_text(previous)
        return ActionResult.done(f"Вставлено: запись {params.number}", value=len(entry.content))


def _describe_history(views: Sequence[ClipboardEntryView], *, query: str, pinned_only: bool) -> str:
    """The list as a sentence, or why it is empty."""
    if views:
        return "\n".join(view.describe_ru() for view in views)
    if query:
        return f"В истории буфера нет записей со словом «{query}»"
    if pinned_only:
        return "Закреплённых записей нет"
    return "История буфера пуста"


def paste_shortcut() -> None:
    """Synthesise Ctrl+V through the configured input backend."""
    press_combo(
        parse_combo("ctrl+v"),
        backend=get_input_backend(),
        hold_ms=get_settings().actions.input.key_hold_ms,
    )
    time.sleep(_PASTE_SETTLE_S)


def _schedule_clear(backend: ClipboardBackend, delay: float, guarded: str) -> None:
    """Wipe the clipboard after ``delay`` seconds, on a daemon timer.

    Only if the clipboard still holds what was put there: clearing it blindly
    would throw away whatever the user copied in the meantime. A daemon thread on
    purpose — quitting Ayris must not wait out a ten-minute timer, and a clipboard
    that outlives the process is the operating system's business.
    """

    def clear_later() -> None:
        time.sleep(delay)
        try:
            snapshot = backend.read()
            if snapshot.is_text and guarded and snapshot.text != guarded:
                return
            backend.clear()
        except ActionError as error:
            _log.warning("delayed clipboard clear failed: %s", error)
        finally:
            if guarded:
                forget_secret(guarded)

    threading.Thread(target=clear_later, name="ayris-clipboard-clear", daemon=True).start()
