"""Задача 27: буфер обмена и его история — без буфера обмена.

Everything here runs on a machine whose clipboard is a variable. That is possible
because the win32 layer is behind one four-method seam
(:class:`~ayris.actions.system.clipboard.ClipboardBackend`) and every decision
worth getting wrong sits above it:
:func:`~ayris.actions.system.clipboard.record_clipboard` is the whole policy of the
monitor as a pure function of a snapshot, and
:func:`~ayris.actions.system.clipboard.history_views` is the whole of numbering.

Five things carry the weight.

*A history that stores the wrong thing is a liability, not a feature.* An image, a
file drop, a blank line, a value a password manager asked monitors to ignore, and
anything longer than the limit must not reach ``clipboard_history`` — and the
over-long one is *skipped*, not truncated, because a truncated entry looks fine in
the list and pastes half a document.

*Numbering must not move under the user.* «Вставь третий» refers to the third line
of the list just read out, so numbering is newest-first, assigned after filtering,
and pinning an entry may not renumber anything.

*Pasting is not copying.* Putting entry three on the clipboard looks exactly like a
Ctrl+C to the monitor, and without suppression «вставь третий» would promote that
entry to the top and renumber the rest.

*A busy clipboard is normal.* ``OpenClipboard`` fails whenever another process holds
the clipboard, which right after a Ctrl+C is most of the time. The wrapper retries,
and when it finally gives up the user gets a Russian sentence rather than a
traceback.

*A secret must be absent from the places that persist.* The database file, the
audit row and the log file are each checked for the value itself, not for a flag
saying it was handled.

Groups:

* :class:`TestSnapshots` — classifying a raw win32 read; exclusion markers.
* :class:`TestBusyClipboard` — retries, and the Russian error at the end of them.
* :class:`TestRecording` — the monitor's policy: dedup, limits, kinds, secrets.
* :class:`TestEviction` — the count limit, and pinned entries surviving it.
* :class:`TestNumbering` — newest-first, filtered, and stable under pinning.
* :class:`TestClipboardSet` — writing, marking secret, the delayed wipe.
* :class:`TestClipboardGet` — text, images, file drops, the spoken limit.
* :class:`TestClipboardHistory` — list, pin, unpin, delete, clear.
* :class:`TestClipboardPaste` — Ctrl+V, restoring, wiping after a secret.
* :class:`TestMonitor` — the listener's counters, its silence on failure.
* :class:`TestSecretsStayOut` — the database, the audit and the log.
* :class:`TestRealClipboard` — Windows only: a real round-trip, a real window.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import TYPE_CHECKING

import pytest

from ayris.actions.input.backend import RecordingBackend, reset_input_backend, set_input_backend
from ayris.actions.registry import ActionRegistry
from ayris.actions.system import clipboard as clipboard_module
from ayris.actions.system import clipboard_monitor as monitor_module
from ayris.actions.system.clipboard import (
    EXCLUSION_FORMATS,
    ClipboardBusy,
    ClipboardGet,
    ClipboardHistory,
    ClipboardKind,
    ClipboardPaste,
    ClipboardSet,
    ClipboardSnapshot,
    FakeClipboard,
    WinClipboard,
    entry_by_number,
    history_views,
    record_clipboard,
    reset_clipboard,
    reset_clipboard_store,
    set_clipboard,
    set_clipboard_store,
    suppress_record,
)
from ayris.actions.system.clipboard_monitor import ClipboardMonitor
from ayris.core.config import ClipboardActionsConfig
from ayris.core.database import Database, reset_database
from ayris.core.models import ExecutionResult
from ayris.core.repositories import Repositories
from ayris.utils import logger as logger_module
from ayris.utils import winapi

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    from ayris.core.repositories import ClipboardRepository

SECRET = "Пароль-от-Госуслуг-77"


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the paste settle delay out of the suite.

    It exists so a real application has time to read the clipboard before it is
    wiped; multiplied by every test that pastes it is only wall-clock.
    """
    monkeypatch.setattr(clipboard_module, "_PASTE_SETTLE_S", 0.0)


@pytest.fixture(autouse=True)
def _clean_seams() -> Iterator[None]:
    """Leave no backend, store or suppression behind for the next test."""
    yield
    reset_clipboard()
    reset_clipboard_store()
    # The reset drops the cached backend; the override has to be taken off
    # separately, or it stays installed for every later test file.
    set_input_backend(None)
    reset_input_backend()
    clipboard_module._suppressed.clear()


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    handle = Database.open(tmp_path / "ayris.db")
    yield handle
    handle.close()
    reset_database()


@pytest.fixture
def store(database: Database) -> ClipboardRepository:
    """The history repository, also installed as the process-wide one."""
    repository = Repositories(database).clipboard
    set_clipboard_store(repository)
    return repository


@pytest.fixture
def fake() -> FakeClipboard:
    """A clipboard in a variable, installed as the process-wide backend."""
    backend = FakeClipboard()
    set_clipboard(backend)
    return backend


@pytest.fixture
def keyboard() -> RecordingBackend:
    """A keyboard that writes down what was pressed instead of pressing it."""
    backend = RecordingBackend()
    set_input_backend(backend)
    return backend


def config(**overrides: object) -> ClipboardActionsConfig:
    """A ``[actions.clipboard]`` section with the defaults a test wants changed."""
    return ClipboardActionsConfig(**overrides)  # type: ignore[arg-type]


def text_of(*lines: str) -> ClipboardSnapshot:
    return ClipboardSnapshot(kind=ClipboardKind.TEXT, text="\n".join(lines))


# --------------------------------------------------------------------------- #
# Что лежит в буфере
# --------------------------------------------------------------------------- #


class TestSnapshots:
    """Turning a raw win32 read into something the policy can decide about."""

    @staticmethod
    def data(**kwargs: object) -> winapi.ClipboardData:
        base: dict[str, object] = {"formats": (), "text": "", "files": (), "blobs": {}}
        base.update(kwargs)
        return winapi.ClipboardData(**base)  # type: ignore[arg-type]

    def test_text_is_text(self) -> None:
        snapshot = clipboard_module._snapshot_from(
            self.data(formats=(winapi.CF_UNICODETEXT,), text="привет"), ()
        )
        assert snapshot.kind is ClipboardKind.TEXT
        assert snapshot.is_text
        assert snapshot.text == "привет"

    def test_a_file_drop_wins_over_its_own_text(self) -> None:
        """Explorer puts both a CF_HDROP and the file names on the clipboard.

        Recording the names as text would put «C:\\...\\фото.jpg» in the history and
        paste a path where the user expects a file.
        """
        snapshot = clipboard_module._snapshot_from(
            self.data(
                formats=(winapi.CF_HDROP, winapi.CF_UNICODETEXT),
                text="C:\\фото.jpg",
                files=("C:\\фото.jpg",),
            ),
            (),
        )
        assert snapshot.kind is ClipboardKind.FILES
        assert not snapshot.is_text
        assert snapshot.files == ("C:\\фото.jpg",)

    def test_an_image_is_recognised_and_carries_no_text(self) -> None:
        snapshot = clipboard_module._snapshot_from(self.data(formats=(winapi.CF_DIB,)), ())
        assert snapshot.kind is ClipboardKind.IMAGE
        assert snapshot.text == ""
        assert snapshot.describe_ru() == "В буфере изображение"

    def test_an_unknown_format_is_other_and_nothing_is_empty(self) -> None:
        assert (
            clipboard_module._snapshot_from(self.data(formats=(0xC123,)), ()).kind
            is ClipboardKind.OTHER
        )
        assert clipboard_module._snapshot_from(self.data(), ()).kind is ClipboardKind.EMPTY

    def test_описание_не_цитирует_текст(self) -> None:
        """A description says how much text there is, never what it says.

        It is spoken and logged, and the text may be the password a manager just
        put there.
        """
        described = text_of(SECRET).describe_ru()
        assert SECRET not in described
        assert described == f"В буфере {len(SECRET)} символов текста"

    @pytest.mark.skipif(sys.platform != "win32", reason="нужны id форматов от Windows")
    def test_an_exclusion_marker_is_seen(self) -> None:
        ignore = winapi.register_clipboard_format(EXCLUSION_FORMATS[0])
        assert ignore
        snapshot = clipboard_module._snapshot_from(
            self.data(formats=(winapi.CF_UNICODETEXT, ignore), text=SECRET), (ignore,)
        )
        assert snapshot.excluded

    @pytest.mark.skipif(sys.platform != "win32", reason="нужны id форматов от Windows")
    def test_can_include_set_to_one_is_permission_not_refusal(self) -> None:
        """``CanIncludeInClipboardHistory`` inverts the question, and a 1 means yes."""
        allow = winapi.register_clipboard_format(EXCLUSION_FORMATS[2])
        assert allow
        permitted = clipboard_module._snapshot_from(
            self.data(
                formats=(winapi.CF_UNICODETEXT, allow),
                text="ссылка",
                blobs={allow: (1).to_bytes(4, "little")},
            ),
            (allow,),
        )
        refused = clipboard_module._snapshot_from(
            self.data(
                formats=(winapi.CF_UNICODETEXT, allow),
                text=SECRET,
                blobs={allow: (0).to_bytes(4, "little")},
            ),
            (allow,),
        )
        assert not permitted.excluded
        assert refused.excluded


# --------------------------------------------------------------------------- #
# Занятый буфер
# --------------------------------------------------------------------------- #


class TestBusyClipboard:
    """The clipboard is a desktop-wide lock, and someone else usually holds it."""

    def test_открытие_повторяется_и_кончается_понятной_ошибкой(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal is retried, and only then becomes an error.

        Tested on the private context manager because it is the only place the
        retry loop exists — and because the alternative, a real clipboard held
        open by another process, is not something a test can arrange.
        """
        attempts = 0

        def failing_open(_: object) -> bool:
            nonlocal attempts
            attempts += 1
            return False

        def entry(_library: str, function: str) -> object:
            return failing_open if function == "OpenClipboard" else (lambda: True)

        monkeypatch.setattr(winapi, "_require", entry)
        monkeypatch.setattr(winapi, "_CLIPBOARD_RETRY_S", 0.0)
        with pytest.raises(winapi.WinApiError), winapi._clipboard_open():
            pass  # pragma: no cover - the body is never reached
        assert attempts == winapi._CLIPBOARD_TRIES

    def test_отказ_winapi_превращается_в_русскую_ошибку(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every one of the three operations answers with the same sentence.

        The user cannot tell reading from writing, and «подождите и повторите» is
        the only actionable thing to say about either.
        """
        monkeypatch.setattr(winapi, "available", lambda: True)
        monkeypatch.setattr(winapi, "register_clipboard_format", lambda _name: 0)
        for name in ("read_clipboard", "clipboard_set_text", "clipboard_clear"):
            monkeypatch.setattr(
                winapi,
                name,
                lambda *_args, **_kwargs: (_ for _ in ()).throw(winapi.WinApiError("занят")),
            )
        backend = WinClipboard()
        for call in (backend.read, lambda: backend.write_text("x"), backend.clear):
            with pytest.raises(ClipboardBusy) as info:
                call()
            assert info.value.user_message == (
                "Буфер обмена занят другой программой. Подождите секунду и повторите."
            )

    def test_монитор_не_падает_на_занятом_буфере(self, store: ClipboardRepository) -> None:
        backend = FakeClipboard(text_of("что-то"))
        backend.busy_reads = 1
        monitor = ClipboardMonitor(backend=backend, store=store, settings=config())
        monitor.handle_update()
        assert monitor.stats.errors == 1
        assert store.count() == 0


# --------------------------------------------------------------------------- #
# Политика записи в историю
# --------------------------------------------------------------------------- #


class TestRecording:
    """What becomes a history entry, and what is silently not one."""

    def test_текст_записывается(self, store: ClipboardRepository) -> None:
        outcome = record_clipboard(text_of("первое"), store=store, settings=config())
        assert outcome.stored
        assert outcome.entry_id is not None
        assert [entry.content for entry in store.history()] == ["первое"]

    def test_повторное_копирование_того_же_не_дублируется(self, store: ClipboardRepository) -> None:
        """Ctrl+C pressed twice is one entry, not two identical lines."""
        record_clipboard(text_of("адрес"), store=store, settings=config())
        again = record_clipboard(text_of("адрес"), store=store, settings=config())
        assert not again.stored
        assert again.reason == "duplicate"
        assert store.count() == 1

    def test_тот_же_текст_после_другого_записывается_снова(
        self, store: ClipboardRepository
    ) -> None:
        """Dedup is about the *newest* entry only.

        Copying A, then B, then A again is three deliberate copies, and the user
        who says «вставь первый" means the A they just copied.
        """
        for value in ("а", "б", "а"):
            record_clipboard(text_of(value), store=store, settings=config())
        assert [entry.content for entry in store.history()] == ["а", "б", "а"]

    @pytest.mark.parametrize("blank", ["   ", "\n\t ", " "])
    def test_пробелы_не_записываются(self, store: ClipboardRepository, blank: str) -> None:
        """Selecting an empty line and pressing Ctrl+C is not a history entry.

        Real reads do produce this: whitespace is text as far as Windows is
        concerned, so the blank check has to live above the kind check.
        """
        outcome = record_clipboard(
            ClipboardSnapshot(kind=ClipboardKind.TEXT, text=blank), store=store, settings=config()
        )
        assert not outcome.stored
        assert outcome.reason == "blank"
        assert store.count() == 0

    def test_слишком_длинное_пропускается_целиком(self, store: ClipboardRepository) -> None:
        """Skipped, not truncated: half a document pasted is worse than nothing."""
        outcome = record_clipboard(text_of("я" * 100), store=store, settings=config(max_length=32))
        assert not outcome.stored
        assert outcome.reason == "too-long"
        assert store.count() == 0

    @pytest.mark.parametrize(
        "snapshot",
        [
            ClipboardSnapshot(kind=ClipboardKind.IMAGE),
            ClipboardSnapshot(kind=ClipboardKind.FILES, files=("C:\\ф.jpg",)),
            ClipboardSnapshot(kind=ClipboardKind.OTHER),
            ClipboardSnapshot(),
        ],
    )
    def test_нетекстовое_не_записывается(
        self, store: ClipboardRepository, snapshot: ClipboardSnapshot
    ) -> None:
        """The kind is reported by ``ClipboardGet`` and never stored."""
        outcome = record_clipboard(snapshot, store=store, settings=config())
        assert not outcome.stored
        assert outcome.reason == f"kind:{snapshot.kind.value}"
        assert store.count() == 0

    def test_метка_менеджера_паролей_уважается(self, store: ClipboardRepository) -> None:
        marked = ClipboardSnapshot(kind=ClipboardKind.TEXT, text=SECRET, excluded=True)
        outcome = record_clipboard(marked, store=store, settings=config())
        assert not outcome.stored
        assert outcome.reason == "excluded"
        assert store.count() == 0

    def test_метку_можно_игнорировать_настройкой(self, store: ClipboardRepository) -> None:
        """Deliberately possible: a manager can mark something harmless."""
        marked = ClipboardSnapshot(kind=ClipboardKind.TEXT, text="счёт на оплату", excluded=True)
        outcome = record_clipboard(
            marked, store=store, settings=config(skip_password_managers=False)
        )
        assert outcome.stored

    def test_секрет_не_записывается_никогда(self, store: ClipboardRepository) -> None:
        outcome = record_clipboard(text_of(SECRET), store=store, settings=config(), secret=True)
        assert not outcome.stored
        assert outcome.reason == "secret"
        assert store.count() == 0

    def test_своя_запись_гасится_один_раз(self, store: ClipboardRepository) -> None:
        """A paste is not a copy — but the *next* real copy of the same text is."""
        suppress_record("вставленное")
        first = record_clipboard(text_of("вставленное"), store=store, settings=config())
        second = record_clipboard(text_of("вставленное"), store=store, settings=config())
        assert not first.stored
        assert first.reason == "self"
        assert second.stored

    def test_гасится_ограниченное_число_значений(self) -> None:
        """A suppression nobody claims must not accumulate."""
        for index in range(clipboard_module._MAX_SUPPRESSED * 3):
            suppress_record(f"значение {index}")
        assert len(clipboard_module._suppressed) == clipboard_module._MAX_SUPPRESSED


class TestEviction:
    """The count limit, and what the limit is not allowed to touch."""

    def test_старые_записи_вытесняются(self, store: ClipboardRepository) -> None:
        for index in range(6):
            record_clipboard(text_of(f"строка {index}"), store=store, settings=config(limit=3))
        assert [entry.content for entry in store.history()] == [
            "строка 5",
            "строка 4",
            "строка 3",
        ]

    def test_закреплённые_не_вытесняются(self, store: ClipboardRepository) -> None:
        """Pinning an address is exactly the request «не терять это»."""
        record_clipboard(text_of("Москва, Тверская 1"), store=store, settings=config(limit=2))
        pinned = store.newest()
        assert pinned is not None and pinned.id is not None
        store.set_pinned(pinned.id, pinned=True)
        for index in range(5):
            record_clipboard(text_of(f"мусор {index}"), store=store, settings=config(limit=2))
        contents = [entry.content for entry in store.history()]
        assert "Москва, Тверская 1" in contents
        assert len(contents) == 3  # два незакреплённых плюс закреплённая

    def test_закреплённые_не_занимают_лимит(self, store: ClipboardRepository) -> None:
        for index in range(4):
            record_clipboard(text_of(f"важное {index}"), store=store, settings=config(limit=2))
            entry = store.newest()
            assert entry is not None and entry.id is not None
            store.set_pinned(entry.id, pinned=True)
        assert store.count(pinned=True) == 4
        assert store.count(pinned=False) == 0


# --------------------------------------------------------------------------- #
# Нумерация
# --------------------------------------------------------------------------- #


class TestNumbering:
    """«Вставь третий» has to mean the third line the user was just read."""

    @staticmethod
    def fill(store: ClipboardRepository, *values: str) -> None:
        for value in values:
            record_clipboard(text_of(value), store=store, settings=config())

    def test_нумерация_от_самой_свежей(self, store: ClipboardRepository) -> None:
        self.fill(store, "первое", "второе", "третье")
        views = history_views(store=store, settings=config())
        assert [view.number for view in views] == [1, 2, 3]
        assert [view.preview for view in views] == ["третье", "второе", "первое"]

    def test_запись_по_номеру(self, store: ClipboardRepository) -> None:
        self.fill(store, "первое", "второе", "третье")
        entry = entry_by_number(3, store=store)
        assert entry is not None
        assert entry.content == "первое"

    def test_несуществующий_номер_даёт_none(self, store: ClipboardRepository) -> None:
        self.fill(store, "одно")
        assert entry_by_number(2, store=store) is None
        assert entry_by_number(0, store=store) is None

    def test_закрепление_не_меняет_нумерацию(self, store: ClipboardRepository) -> None:
        """The picker shows pinned entries first; spoken numbering must not.

        Otherwise pinning an entry silently renumbers the list the user is
        holding in their head, and the next «вставь третий» lands elsewhere.
        """
        self.fill(store, "первое", "второе", "третье")
        oldest = entry_by_number(3, store=store)
        assert oldest is not None and oldest.id is not None
        store.set_pinned(oldest.id, pinned=True)
        views = history_views(store=store, settings=config())
        assert [view.preview for view in views] == ["третье", "второе", "первое"]
        assert views[2].pinned
        assert views[2].describe_ru().startswith("3. ★ ")

    def test_поиск_перенумеровывает_ответ(self, store: ClipboardRepository) -> None:
        self.fill(store, "адрес почты", "телефон", "адрес дома")
        views = history_views(query="адрес", store=store, settings=config())
        assert [(view.number, view.preview) for view in views] == [
            (1, "адрес дома"),
            (2, "адрес почты"),
        ]

    def test_только_закреплённые(self, store: ClipboardRepository) -> None:
        self.fill(store, "обычное", "нужное")
        entry = entry_by_number(1, store=store)
        assert entry is not None and entry.id is not None
        store.set_pinned(entry.id, pinned=True)
        views = history_views(pinned_only=True, store=store, settings=config())
        assert [view.preview for view in views] == ["нужное"]

    def test_превью_укорачивается_и_склеивает_строки(self, store: ClipboardRepository) -> None:
        self.fill(store, "первая строка\nвторая строка\nтретья строка")
        view = history_views(store=store, settings=config(preview_length=20))[0]
        assert "\n" not in view.preview
        assert len(view.preview) <= 20
        assert view.preview.endswith("…")
        assert view.length == len("первая строка\nвторая строка\nтретья строка")


# --------------------------------------------------------------------------- #
# Действия
# --------------------------------------------------------------------------- #


class TestClipboardSet:
    def test_текст_попадает_в_буфер(self, fake: FakeClipboard) -> None:
        result = ClipboardSet().run(ClipboardSet.Params(text="привет"))
        assert result.ok
        assert result.value == 6
        assert fake.writes == ["привет"]

    def test_секрет_гасится_и_маскируется(self, fake: FakeClipboard) -> None:
        """A secret write is suppressed for the monitor and masked for the audit."""
        params = ClipboardSet.Params(text=SECRET, secret=True)
        assert params.secret_now() == frozenset({"text"})
        assert ClipboardSet.Params(text="ссылка").secret_now() == frozenset()
        ClipboardSet().run(params)
        assert fake.writes == [SECRET]
        assert clipboard_module._claim_suppressed(SECRET)
        logger_module.forget_secret(SECRET)

    def test_отложенная_очистка_срабатывает(self, fake: FakeClipboard) -> None:
        ClipboardSet().run(ClipboardSet.Params(text=SECRET, secret=True, clear_after_s=0.01))
        deadline = 2.0
        step = 0.02
        waited = 0.0
        while fake.clears == 0 and waited < deadline:
            import time

            time.sleep(step)
            waited += step
        assert fake.clears == 1
        assert fake.read().kind is ClipboardKind.EMPTY

    def test_отложенная_очистка_не_трогает_чужое(self, fake: FakeClipboard) -> None:
        """The user copied something else in the meantime; that is theirs."""
        fake.put(text_of("новое, скопированное пользователем"))
        clipboard_module._schedule_clear(fake, 0.01, "старое")
        import time

        time.sleep(0.2)
        assert fake.clears == 0
        assert fake.read().text == "новое, скопированное пользователем"


class TestClipboardGet:
    def test_текст_возвращается_целиком_а_озвучивается_кратко(self, fake: FakeClipboard) -> None:
        fake.put(text_of("ю" * 500))
        result = ClipboardGet().run(ClipboardGet.Params(speak_limit=50))
        assert result.value == "ю" * 500
        assert len(result.message_ru) <= 50
        assert result.data["length"] == 500

    def test_изображение_описывается_а_не_возвращается(self, fake: FakeClipboard) -> None:
        fake.put(ClipboardSnapshot(kind=ClipboardKind.IMAGE))
        result = ClipboardGet().run(ClipboardGet.Params())
        assert result.value == ""
        assert result.message_ru == "В буфере изображение"
        assert result.data["kind"] == "image"

    def test_файлы_перечисляются(self, fake: FakeClipboard) -> None:
        fake.put(ClipboardSnapshot(kind=ClipboardKind.FILES, files=("C:\\а.txt", "C:\\б.txt")))
        result = ClipboardGet().run(ClipboardGet.Params())
        assert result.message_ru == "В буфере 2 файлов"
        assert result.data["files"] == ["C:\\а.txt", "C:\\б.txt"]

    def test_нулевой_лимит_озвучивает_только_тип(self, fake: FakeClipboard) -> None:
        fake.put(text_of(SECRET))
        result = ClipboardGet().run(ClipboardGet.Params(speak_limit=0))
        assert SECRET not in result.message_ru
        assert result.value == SECRET


class TestClipboardHistory:
    @pytest.fixture(autouse=True)
    def _filled(self, store: ClipboardRepository) -> None:
        for value in ("первое", "второе", "третье"):
            record_clipboard(text_of(value), store=store, settings=config())

    def test_список_читается_вслух_по_номерам(self) -> None:
        result = ClipboardHistory().run(ClipboardHistory.Params())
        assert result.ok
        assert result.message_ru.startswith("1. третье")
        assert result.data["count"] == 3

    def test_пустая_история_объясняет_себя(self, store: ClipboardRepository) -> None:
        store.clear(keep_pinned=False)
        assert (
            ClipboardHistory().run(ClipboardHistory.Params()).message_ru == "История буфера пуста"
        )
        assert (
            ClipboardHistory().run(ClipboardHistory.Params(query="нет такого")).message_ru
            == "В истории буфера нет записей со словом «нет такого»"
        )

    def test_закрепить_и_открепить(self, store: ClipboardRepository) -> None:
        pinned = ClipboardHistory().run(ClipboardHistory.Params(operation="pin", number=2))
        assert pinned.ok
        assert store.count(pinned=True) == 1
        unpinned = ClipboardHistory().run(ClipboardHistory.Params(operation="unpin", number=2))
        assert unpinned.ok
        assert store.count(pinned=True) == 0

    def test_удалить_одну(self, store: ClipboardRepository) -> None:
        result = ClipboardHistory().run(ClipboardHistory.Params(operation="delete", number=1))
        assert result.ok
        assert [entry.content for entry in store.history()] == ["второе", "первое"]

    def test_несуществующий_номер_отказ_а_не_исключение(self) -> None:
        result = ClipboardHistory().run(ClipboardHistory.Params(operation="pin", number=99))
        assert not result.ok
        assert "номер 99" in result.message_ru

    def test_очистка_щадит_закреплённые(self, store: ClipboardRepository) -> None:
        ClipboardHistory().run(ClipboardHistory.Params(operation="pin", number=3))
        result = ClipboardHistory().run(ClipboardHistory.Params(operation="clear"))
        assert result.ok
        assert result.data["kept_pinned"] == 1
        assert [entry.content for entry in store.history()] == ["первое"]

    def test_очистка_с_флагом_убирает_всё(self, store: ClipboardRepository) -> None:
        ClipboardHistory().run(ClipboardHistory.Params(operation="pin", number=3))
        ClipboardHistory().run(ClipboardHistory.Params(operation="clear", include_pinned=True))
        assert store.count() == 0

    def test_неизвестная_операция_не_проходит_валидацию(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClipboardHistory.Params(operation="сжечь")


class TestClipboardPaste:
    @pytest.fixture(autouse=True)
    def _filled(self, store: ClipboardRepository) -> None:
        for value in ("первое", "второе", "третье"):
            record_clipboard(text_of(value), store=store, settings=config())

    def test_вставить_третий(self, fake: FakeClipboard, keyboard: RecordingBackend) -> None:
        result = ClipboardPaste().run(ClipboardPaste.Params(number=3))
        assert result.ok
        assert fake.writes[0] == "первое"
        assert keyboard.keys == ("+ctrl", "+v", "-v", "-ctrl")

    def test_только_в_буфер_без_нажатия(
        self, fake: FakeClipboard, keyboard: RecordingBackend
    ) -> None:
        result = ClipboardPaste().run(ClipboardPaste.Params(number=1, paste=False))
        assert result.ok
        assert fake.writes == ["третье"]
        assert keyboard.keys == ()

    def test_прежнее_содержимое_возвращается(
        self, fake: FakeClipboard, keyboard: RecordingBackend
    ) -> None:
        fake.put(text_of("то, что было"))
        ClipboardPaste().run(ClipboardPaste.Params(number=2))
        assert fake.writes == ["второе", "то, что было"]

    def test_вставка_не_переставляет_историю(
        self, store: ClipboardRepository, fake: FakeClipboard, keyboard: RecordingBackend
    ) -> None:
        """The monitor sees the paste as a copy; suppression is what stops it.

        Without this, «вставь третий» promotes the third entry to first and the
        list the user just heard is wrong.
        """
        ClipboardPaste().run(ClipboardPaste.Params(number=3))
        outcome = record_clipboard(text_of("первое"), store=store, settings=config())
        assert outcome.reason == "self"
        assert [entry.content for entry in store.history()] == ["третье", "второе", "первое"]

    def test_секрет_очищает_буфер_и_не_возвращает_прежнее(
        self, fake: FakeClipboard, keyboard: RecordingBackend
    ) -> None:
        fake.put(text_of("то, что было"))
        result = ClipboardPaste().run(ClipboardPaste.Params(number=1, secret=True))
        assert result.ok
        assert fake.clears == 1
        assert fake.read().kind is ClipboardKind.EMPTY
        assert "то, что было" not in fake.writes[1:]

    def test_настройка_может_оставить_буфер(
        self, fake: FakeClipboard, keyboard: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            clipboard_module, "clipboard_settings", lambda: config(clear_after_secret=False)
        )
        ClipboardPaste().run(ClipboardPaste.Params(number=1, secret=True))
        assert fake.clears == 0

    def test_несуществующий_номер_отказ(self, fake: FakeClipboard) -> None:
        result = ClipboardPaste().run(ClipboardPaste.Params(number=9))
        assert not result.ok
        assert fake.writes == []


# --------------------------------------------------------------------------- #
# Монитор
# --------------------------------------------------------------------------- #


class TestMonitor:
    def test_изменение_буфера_становится_записью(self, store: ClipboardRepository) -> None:
        backend = FakeClipboard()
        monitor = ClipboardMonitor(backend=backend, store=store, settings=config())
        backend.put(text_of("скопировано"))
        monitor.handle_update()
        assert monitor.stats.stored == 1
        assert [entry.content for entry in store.history()] == ["скопировано"]

    def test_причина_пропуска_попадает_в_счётчики(self, store: ClipboardRepository) -> None:
        """A history that stayed empty is otherwise undiagnosable: the value
        cannot be printed, so the reason has to be enough."""
        backend = FakeClipboard()
        monitor = ClipboardMonitor(backend=backend, store=store, settings=config())
        backend.put(ClipboardSnapshot(kind=ClipboardKind.IMAGE))
        monitor.handle_update()
        backend.put(ClipboardSnapshot(kind=ClipboardKind.TEXT, text=SECRET, excluded=True))
        monitor.handle_update()
        assert monitor.stats.stored == 0
        assert monitor.stats.skipped == 2
        assert monitor.stats.reasons == {"kind:image": 1, "excluded": 1}

    def test_выключенный_монитор_не_запускается(self, store: ClipboardRepository) -> None:
        monitor = ClipboardMonitor(
            backend=FakeClipboard(), store=store, settings=config(monitor=False)
        )
        assert monitor.start() is False
        assert not monitor.running

    @pytest.mark.skipif(sys.platform == "win32", reason="про поведение вне Windows")
    def test_вне_windows_монитор_молча_не_стартует(self, store: ClipboardRepository) -> None:
        monitor = ClipboardMonitor(backend=FakeClipboard(), store=store, settings=config())
        assert monitor.start() is False
        monitor.stop()

    def test_остановка_без_старта_безопасна(self) -> None:
        monitor_module.reset_clipboard_monitor()
        ClipboardMonitor(backend=FakeClipboard(), settings=config()).stop()


# --------------------------------------------------------------------------- #
# Секреты не остаются нигде
# --------------------------------------------------------------------------- #


class TestSecretsStayOut:
    """The three places a value would survive the moment it was needed."""

    def test_секрета_нет_в_файле_базы(
        self, tmp_path: Path, store: ClipboardRepository, database: Database, fake: FakeClipboard
    ) -> None:
        ClipboardSet().run(ClipboardSet.Params(text=SECRET, secret=True))
        record_clipboard(text_of(SECRET), store=store, settings=config(), secret=True)
        record_clipboard(
            ClipboardSnapshot(kind=ClipboardKind.TEXT, text=SECRET, excluded=True),
            store=store,
            settings=config(),
        )
        record_clipboard(text_of("безобидное"), store=store, settings=config())
        database.close()
        raw = (tmp_path / "ayris.db").read_bytes()
        assert SECRET.encode("utf-8") not in raw
        assert SECRET.encode("utf-16-le") not in raw
        assert "безобидное".encode() in raw
        logger_module.forget_secret(SECRET)

    def test_секрета_нет_в_журнале_аудита(self, database: Database, fake: FakeClipboard) -> None:
        repos = Repositories(database)
        registry = ActionRegistry(audit=repos.audit, audit_enabled=lambda: True)
        registry.add(ClipboardSet)
        try:
            registry.execute("ClipboardSet", {"text": SECRET, "secret": True})
        finally:
            registry.shutdown()
        entry = repos.audit.recent(1)[0]
        assert entry.result is ExecutionResult.OK
        assert SECRET not in str(entry.params)
        assert entry.params["text"] != SECRET
        logger_module.forget_secret(SECRET)

    def test_секрета_нет_в_логе_даже_на_debug(
        self, tmp_path: Path, fake: FakeClipboard, keyboard: RecordingBackend
    ) -> None:
        """The filter sits on the handlers, so DEBUG is covered too.

        DEBUG is where a secret leaks in practice: someone adds «%s» to a
        troubleshooting line, and it is invisible until a user sends their log.
        """
        log_dir = tmp_path / "logs"
        logger_module.setup_logging("DEBUG", console=False, log_dir=log_dir)
        try:
            ClipboardSet().run(ClipboardSet.Params(text=SECRET, secret=True))
            logging.getLogger("ayris.test").debug("что-то с буфером: %s", SECRET)
            logging.getLogger("ayris.test").error("и в ошибке тоже: %s", SECRET)
        finally:
            logger_module.shutdown_logging()
            logger_module.forget_secret(SECRET)
        written = "\n".join(path.read_text("utf-8") for path in log_dir.glob("*.log"))
        assert written
        assert SECRET not in written
        assert logger_module.SECRET_PLACEHOLDER in written

    def test_значение_перестаёт_прятаться_после_forget(self) -> None:
        before = logger_module.guarded_secret_count()
        logger_module.guard_secret(SECRET)
        assert logger_module.redact(f"вот {SECRET}") == f"вот {logger_module.SECRET_PLACEHOLDER}"
        logger_module.forget_secret(SECRET)
        assert logger_module.redact(f"вот {SECRET}") == f"вот {SECRET}"
        assert logger_module.guarded_secret_count() == before


# --------------------------------------------------------------------------- #
# Настоящий буфер
# --------------------------------------------------------------------------- #


@pytest.mark.xdist_group("clipboard")
@pytest.mark.skipif(sys.platform != "win32", reason="нужен настоящий буфер обмена Windows")
class TestRealClipboard:
    """The parts a fake cannot check: that the win32 calls are spelled right.

    ``xdist_group`` holds this class on the same worker as
    ``test_input.py::TestRealClipboardPaste``: the clipboard is one lock for the
    whole desktop, and the listener below reads it on every update, so the two
    classes running side by side on different workers make each other fail with
    ``OpenClipboard [5]``. The run passes ``--dist=loadgroup`` for the group to
    mean anything.
    """

    def test_запись_чтение_и_очистка(
        self, clipboard_or_skip: Callable[[], AbstractContextManager[None]]
    ) -> None:
        backend = WinClipboard()
        marker = "Айрис: проверка буфера 27"
        with clipboard_or_skip():
            backend.write_text(marker)
            snapshot = backend.read()
            assert snapshot.kind is ClipboardKind.TEXT
            assert snapshot.text == marker
            assert backend.sequence() > 0
            backend.clear()
            assert backend.read().kind is ClipboardKind.EMPTY

    def test_окно_слушателя_поднимается_и_получает_сообщения(
        self,
        store: ClipboardRepository,
        clipboard_or_skip: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """The listener really registers, really pumps, and really stops.

        This is the one test that exercises ``AddClipboardFormatListener`` —
        everything above it runs against ``handle_update`` directly.
        """
        real = WinClipboard()
        monitor = ClipboardMonitor(backend=real, store=store, settings=config())
        assert monitor.start() is True
        try:
            with clipboard_or_skip():
                deadline = 3.0
                waited = 0.0
                real.write_text("Айрис: проверка слушателя 27")
                while monitor.stats.received == 0 and waited < deadline:
                    import time

                    time.sleep(0.05)
                    waited += 0.05
                assert monitor.stats.received >= 1
        finally:
            monitor.stop()
            # Очистка — гигиена, а не проверка: если буфер в этот момент забрала
            # чужая программа, тест уже сказал всё, что хотел.
            with contextlib.suppress(ClipboardBusy):
                real.clear()
        assert not monitor.running
