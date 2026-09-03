"""Задача 29: Яндекс Музыка — без Яндекс Музыки и без Windows.

Управление плеером состоит из двух уровней, и оба здесь проверяются целиком без
запущенного приложения. Первый — транспорт Windows
(:mod:`ayris.actions.media.smtc`): за ним стоит один класс с двумя методами, и всё,
что можно решить неправильно, решается выше него — :func:`pick_session` целиком
чистая функция от списка сессий, :func:`now_playing_ru` — от одной. Второй уровень
(:mod:`ayris.actions.media.cdp`, :mod:`ayris.actions.media.yandex_music`) — это
рукописный WebSocket и строка JavaScript, то есть две вещи, которые проверяются
почти байт в байт: кадр — round-trip'ом, скрипт — тем, что в нём осталось.

Шесть вещей несут вес.

*Ни одной синтезированной клавиши.* Это не пожелание, а требование задачи:
пользователь играет, а нажатие уходит туда, где фокус. Поэтому здесь есть тест,
который разбирает импорты всех четырёх модулей и требует, чтобы клавиатура
импортировалась ровно в одном месте — внутри ``_press_media_key``, — и тест, который
читает все шаблоны JavaScript и ищет в них ``KeyboardEvent`` и ``dispatchKeyEvent``.
Сам фоллбек проверяется отдельно, через :class:`RecordingBackend` и по VK-кодам:
путь есть, он выключен по умолчанию и должен работать, когда его попросили.

*Настроенный плеер важнее играющего.* «Следующий трек», сказанное поверх видео в
браузере, — про музыку. Это два теста на :func:`pick_session`, и оба про сессию,
которая на паузе, пока играет другая.

*Кадр обязан быть маскированным.* Сервер, получивший немаскированный кадр от
клиента, обязан разорвать соединение — то есть ошибка здесь выглядит не как
«неверный ответ», а как «сокет молча закрылся». Отсюда round-trip и проверка бита
маски прямо в байтах.

*Заголовка ``Origin`` быть не должно.* Chromium 111 и новее отвергает
DevTools-сокет с чужим ``Origin``, а библиотеки WebSocket посылают его сами. Здесь
поднимается настоящий сервер на loopback, который записывает заголовки запроса, и
отсутствие ``Origin`` — утверждение теста, а не комментарий в коде.

*Приложение не перезапускается за спиной.* Запущенная без порта Яндекс Музыка
означает отказ с внятной фразой, а не тихий рестарт посреди трека. Проверяется
тем, что лаунчер не получил ни одного запроса.

*Отмена не должна убрать не тот трек.* «Отмени» после «добавь в Любимку» удаляет
именно то, что было добавлено, а если играет уже другое — отказывается.

Groups:

* :class:`TestMediaSession` — снимок сессии: фразы, флаги, плоские данные.
* :class:`TestStatusesAndCommands` — русские названия и числа ABI WinRT.
* :class:`TestPickSession` — какой из плееров слушать; настроенный против играющего.
* :class:`TestNowPlaying` — «что играет» во всех состояниях.
* :class:`TestTransport` — шесть действий транспорта поверх записывающего бэкенда.
* :class:`TestMediaKeys` — фоллбек: выключен по умолчанию, VK-коды когда включён.
* :class:`TestNoKeys` — ни одного нажатия ни в питоне, ни в JavaScript.
* :class:`TestFrames` — кадры WebSocket: round-trip, маска, размер, ключ RFC.
* :class:`TestTargets` — ``/json/list`` в цели и выбор страницы приложения.
* :class:`TestDevTools` — живой сокет на loopback: рукопожатие, ping, фрагменты.
* :class:`TestAttach` — отказы, когда порта нет: запуск, таймаут, запрет рестарта.
* :class:`TestSelectors` — таблица селекторов как данные, переопределение, полнота.
* :class:`TestScripts` — что именно уезжает в страницу: селекторы, маршруты, клики.
* :class:`TestActions` — пять действий второго уровня и их фразы.
* :class:`TestUndo` — отмена добавления в плейлист и её отказы.
* :class:`TestSchemas` — регистрация, категория, поля для редактора макросов.
* :class:`TestRealSmtc` — только Windows: настоящий список плееров системы.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import socket
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from string import Template
from typing import Any

import pytest

from ayris.actions.base import FieldKind, build_schema
from ayris.actions.input.backend import (
    RecordingBackend,
    reset_input_backend,
    set_input_backend,
)
from ayris.actions.media import cdp
from ayris.actions.media import smtc as smtc_module
from ayris.actions.media import yandex_music as ym_module
from ayris.actions.media.cdp import (
    MAX_FRAME_BYTES,
    MUSIC_PAGE_PREFIX,
    OP_CLOSE,
    OP_PING,
    OP_TEXT,
    CdpClient,
    CdpError,
    CdpUnavailable,
    PageTarget,
    RecordingCdp,
    accept_token,
    decode_frame,
    encode_frame,
    parse_targets,
    pick_page,
)
from ayris.actions.media.smtc import (
    YANDEX_MUSIC_APP_ID,
    MediaSession,
    NowPlaying,
    NullSessions,
    PlaybackStatus,
    RecordingSessions,
    TransportCommand,
    YMNext,
    YMPause,
    YMPlay,
    YMPrev,
    YMToggle,
    now_playing_ru,
    pick_session,
    set_sessions,
)
from ayris.actions.media.yandex_music import (
    ROUTES,
    SELECTORS,
    LikeMode,
    SearchKind,
    YMAddToPlaylist,
    YMLike,
    YMPlaylist,
    YMSearch,
    YMWave,
    debug_flags,
    selector,
    set_transport,
)
from ayris.actions.registry import ActionRegistry
from ayris.actions.system.apps import LaunchRequest, set_launcher
from ayris.core.config import MediaActionsConfig
from ayris.core.errors import ActionError, ActionUnavailable

#: Сессия Яндекс Музыки с треком, который слышно.
PLAYING = MediaSession(
    app_id=YANDEX_MUSIC_APP_ID,
    title="Выхода нет",
    artist="Сплин",
    album="Гранатовый альбом",
    status=PlaybackStatus.PLAYING,
    position_s=61.5,
    duration_s=245.0,
    can_pause=True,
    can_next=True,
    can_previous=True,
)

#: Та же, но на паузе — то состояние, в котором её всё равно надо предпочесть.
PAUSED = MediaSession(
    app_id=YANDEX_MUSIC_APP_ID,
    title="Романс",
    artist="Сплин",
    status=PlaybackStatus.PAUSED,
)

#: Чужой плеер, который в этот момент играет.
BROWSER = MediaSession(
    app_id="msedge.exe",
    title="Лекция про кольца",
    artist="YouTube",
    status=PlaybackStatus.PLAYING,
)

#: Сессия есть, а трека в ней нет: так выглядит только что открытое приложение.
BLANK = MediaSession(app_id=YANDEX_MUSIC_APP_ID, status=PlaybackStatus.OPENED)

#: Ответ ``/json/list``, снятый с настоящего приложения 5.115.3.
TARGETS_JSON = json.dumps(
    [
        {
            "id": "5A8",
            "type": "page",
            "title": "Яндекс Музыка",
            "url": "music-application://desktop/",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/5A8",
        },
        {
            "id": "B21",
            "type": "page",
            "title": "about:blank",
            "url": "about:blank",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/B21",
        },
        {
            "id": "C33",
            "type": "service_worker",
            "title": "sw.js",
            "url": "music-application://desktop/sw.js",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/C33",
        },
        {
            "id": "D44",
            "type": "page",
            "title": "уже отлаживается кем-то",
            "url": "music-application://desktop/other",
        },
    ]
)


def config(**overrides: Any) -> MediaActionsConfig:
    """Секция ``[actions.media]`` с изменёнными по вкусу теста значениями."""
    return MediaActionsConfig(**overrides)


def use_config(monkeypatch: pytest.MonkeyPatch, section: MediaActionsConfig) -> None:
    """Подсунуть секцию обоим модулям, которые её читают.

    Патч ложится на имя ``get_settings`` внутри каждого модуля: оба взяли его
    ``from ayris.core.config import get_settings``, то есть держат свою ссылку.
    """
    from ayris.core.config import get_settings

    base = get_settings()
    patched = base.model_copy(
        update={"actions": base.actions.model_copy(update={"media": section})}
    )
    monkeypatch.setattr(smtc_module, "get_settings", lambda: patched)
    monkeypatch.setattr(ym_module, "get_settings", lambda: patched)


class RecordingLauncher:
    """Запоминает, что просили запустить, и ничего не запускает."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.requests: list[LaunchRequest] = []

    def launch(self, request: LaunchRequest) -> int:
        self.requests.append(request)
        return self.pid


class FakeClock:
    """Часы вместо :mod:`time` внутри модуля: сон мгновенный, но время идёт.

    Ставится на место самого ``time`` в :mod:`ayris.actions.media.yandex_music`, а
    не на ``time.sleep`` глобально: модуль зовёт только ``monotonic`` и ``sleep``,
    а подменять их всему процессу — значит попутно сломать часы самому pytest.
    """

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _reset_media_seams() -> Iterator[None]:
    """Ни один тест не должен унести с собой ни бэкенд, ни транспорт, ни лаунчер."""
    yield
    set_sessions(None)
    set_transport(None)
    set_launcher(None)
    set_input_backend(None)
    reset_input_backend()
    # Отметка о последнем клике по плейлисту — тоже состояние модуля: иначе второй
    # тест подряд честно ждал бы больше секунды, дожидаясь несуществующего меню.
    ym_module._last_toggle = 0.0


@pytest.fixture
def sessions() -> RecordingSessions:
    """Записывающий транспорт с одной играющей Яндекс Музыкой."""
    backend = RecordingSessions([PLAYING])
    set_sessions(backend)
    return backend


@pytest.fixture
def keyboard() -> RecordingBackend:
    backend = RecordingBackend()
    set_input_backend(backend)
    return backend


def answering(*results: Any) -> RecordingCdp:
    """Страница с заранее заданными ответами, установленная как транспорт."""
    transport = RecordingCdp(results=list(results))
    set_transport(transport)
    return transport


def run(action_class: Any, **params: Any) -> Any:
    """Вызвать действие так, как это делает реестр: через провалидированные Params."""
    action = action_class()
    return action.run(action_class.params_model()(**params))


def source_text(name: str) -> str:
    """Исходник одного модуля пакета медиа."""
    return (Path(ym_module.__file__).parent / name).read_text(encoding="utf-8")


def as_literal(name: str) -> str:
    """Селектор так, как он уезжает в страницу: литералом JavaScript, а не сырым CSS.

    Внутри селекторов двойные кавычки, в скрипте они экранированы. Искать там сырое
    значение из :data:`SELECTORS` — значит не найти его никогда.
    """
    return json.dumps(SELECTORS[name], ensure_ascii=False)


def js_templates() -> dict[str, str]:
    """Все шаблоны JavaScript модуля, как текст, и ничего кроме них.

    Отдельно от исходника: в докстрингах этого модуля разбирается, почему в нём нет
    ``location.assign`` и синтезированных клавиш, — то есть поиск по файлу нашёл бы
    ровно те слова, которых там не должно быть в коде.
    """
    found = {
        name: value.template if isinstance(value, Template) else str(value)
        for name, value in vars(ym_module).items()
        if name.startswith(("_JS_", "_PRELUDE"))
    }
    assert len(found) >= 5, "шаблоны переименовали — тест перестал их находить"
    return found


def js_scripts() -> dict[str, str]:
    """Только большие скрипты — те, что собраны :func:`_script` и уезжают целиком.

    Отдельно от :func:`js_templates`: там ещё короткий вопрос про готовность
    интерфейса, к которому требования «обёрнут в функцию» не относятся.
    """
    found = {
        name: value.template
        for name, value in vars(ym_module).items()
        if isinstance(value, Template)
    }
    assert len(found) == 4, "скрипты переименовали — тест перестал их находить"
    return found


class TestMediaSession:
    """Снимок одной сессии: что из него можно сказать и что записать."""

    def test_a_playing_session_knows_it_is_playing(self) -> None:
        assert PLAYING.playing is True
        assert PAUSED.playing is False

    def test_a_session_without_a_track_is_empty(self) -> None:
        assert BLANK.empty is True
        assert PLAYING.empty is False

    def test_the_label_is_artist_then_title(self) -> None:
        assert PLAYING.label_ru == "Сплин — Выхода нет"

    @pytest.mark.parametrize(
        ("session", "expected"),
        [
            (MediaSession(app_id="x", title="Романс"), "Романс"),
            (MediaSession(app_id="x", artist="Сплин"), "Сплин"),
            (MediaSession(app_id="x"), ""),
        ],
    )
    def test_half_a_label_is_still_a_label(self, session: MediaSession, expected: str) -> None:
        assert session.label_ru == expected

    def test_the_data_row_is_flat_and_rounded(self) -> None:
        assert PLAYING.as_data() == {
            "app_id": YANDEX_MUSIC_APP_ID,
            "title": "Выхода нет",
            "artist": "Сплин",
            "album": "Гранатовый альбом",
            "status": "playing",
            "position_s": 61.5,
            "duration_s": 245.0,
        }

    def test_a_snapshot_cannot_be_edited(self) -> None:
        with pytest.raises(AttributeError):
            PLAYING.title = "другое"  # type: ignore[misc]


class TestStatusesAndCommands:
    """Русские названия состояний и приказов, и числа, которыми их зовёт WinRT."""

    @pytest.mark.parametrize("status", list(PlaybackStatus))
    def test_every_status_has_a_russian_name(self, status: PlaybackStatus) -> None:
        assert status.title_ru
        assert status.title_ru.isprintable()

    @pytest.mark.parametrize("command", list(TransportCommand))
    def test_every_command_has_a_russian_name(self, command: TransportCommand) -> None:
        assert command.title_ru

    def test_the_abi_numbers_are_the_ones_winrt_uses(self) -> None:
        # Часть ABI: выписаны руками, чтобы не тянуть проекцию ради шести чисел.
        assert smtc_module._STATUS_BY_VALUE == {
            0: PlaybackStatus.CLOSED,
            1: PlaybackStatus.OPENED,
            2: PlaybackStatus.CHANGING,
            3: PlaybackStatus.STOPPED,
            4: PlaybackStatus.PLAYING,
            5: PlaybackStatus.PAUSED,
        }

    def test_only_dedicated_media_keys_are_ever_named(self) -> None:
        # Переключение пауза/плей — это и есть клавиша play/pause, а не буква K.
        assert smtc_module._COMMAND_KEYS[TransportCommand.TOGGLE] == "mediaplay"
        assert set(smtc_module._COMMAND_KEYS.values()) <= {
            "mediaplay",
            "medianext",
            "mediaprev",
            "mediastop",
        }


class TestPickSession:
    """Какой из плееров слушает команда. Чистая функция, поэтому таблица."""

    def test_no_sessions_is_no_session(self) -> None:
        assert pick_session([], YANDEX_MUSIC_APP_ID) is None

    def test_one_session_is_the_session(self) -> None:
        assert pick_session([BROWSER], YANDEX_MUSIC_APP_ID) is BROWSER

    def test_the_configured_player_wins_even_on_pause(self) -> None:
        # «Следующий трек» поверх видео в браузере — всё равно про музыку.
        assert pick_session([BROWSER, PAUSED], YANDEX_MUSIC_APP_ID) is PAUSED

    def test_among_two_of_the_same_app_the_playing_one_wins(self) -> None:
        # Яндекс Музыка открывает вторую сессию под видеоклип.
        assert pick_session([PAUSED, PLAYING], YANDEX_MUSIC_APP_ID) is PLAYING

    def test_without_a_configured_player_the_playing_one_wins(self) -> None:
        assert pick_session([PAUSED, BROWSER], "") is BROWSER

    def test_without_a_configured_player_and_nothing_playing_the_first_wins(self) -> None:
        assert pick_session([PAUSED, BLANK], "") is PAUSED

    def test_the_app_id_is_matched_case_insensitively(self) -> None:
        assert pick_session([BROWSER, PAUSED], YANDEX_MUSIC_APP_ID.upper()) is PAUSED


class TestNowPlaying:
    """«Что играет» — одна фраза на каждое состояние."""

    def test_nothing_playing_says_so(self) -> None:
        assert now_playing_ru(None) == "Сейчас ничего не играет."

    def test_an_empty_session_counts_as_nothing(self) -> None:
        assert now_playing_ru(BLANK) == "Сейчас ничего не играет."

    def test_a_playing_track_is_named(self) -> None:
        assert now_playing_ru(PLAYING) == "Сейчас играет Сплин — Выхода нет."

    def test_a_paused_track_is_named_as_paused(self) -> None:
        assert now_playing_ru(PAUSED) == "На паузе Сплин — Романс."

    def test_any_other_state_is_spelled_out(self) -> None:
        stopped = MediaSession(app_id="x", title="Романс", status=PlaybackStatus.STOPPED)
        assert now_playing_ru(stopped) == f"Романс — {PlaybackStatus.STOPPED.title_ru}."

    def test_the_action_answers_from_the_backend(self, sessions: RecordingSessions) -> None:
        result = run(NowPlaying)
        assert result.message_ru == "Сейчас играет Сплин — Выхода нет."
        assert result.data["title"] == "Выхода нет"
        assert sessions.commands == [], "спросили, а не приказали"

    def test_the_action_refuses_when_there_is_no_backend(self) -> None:
        set_sessions(NullSessions())
        with pytest.raises(ActionUnavailable) as info:
            run(NowPlaying)
        assert info.value.user_message == "На этой системе не видно, что играет."


class TestTransport:
    """Шесть действий транспорта, поверх бэкенда, который ничего не отправляет."""

    @pytest.mark.parametrize(
        ("action_class", "command", "message"),
        [
            (YMPlay, TransportCommand.PLAY, "Включаю."),
            (YMPause, TransportCommand.PAUSE, "Ставлю на паузу."),
            (YMNext, TransportCommand.NEXT, "Следующий трек."),
            (YMPrev, TransportCommand.PREVIOUS, "Предыдущий трек."),
        ],
    )
    def test_a_command_reaches_the_configured_player(
        self,
        sessions: RecordingSessions,
        action_class: Any,
        command: TransportCommand,
        message: str,
    ) -> None:
        result = run(action_class)
        assert sessions.commands == [(YANDEX_MUSIC_APP_ID, command)]
        assert result.message_ru == message
        assert result.data["app_id"] == YANDEX_MUSIC_APP_ID

    def test_a_toggle_says_what_it_did_reading_the_state_first(self) -> None:
        set_sessions(RecordingSessions([PLAYING]))
        assert run(YMToggle).message_ru == "Ставлю на паузу."
        set_sessions(RecordingSessions([PAUSED]))
        assert run(YMToggle).message_ru == "Продолжаю."

    def test_a_command_goes_to_the_configured_player_not_the_playing_one(self) -> None:
        backend = RecordingSessions([BROWSER, PAUSED])
        set_sessions(backend)
        run(YMNext)
        assert backend.commands == [(YANDEX_MUSIC_APP_ID, TransportCommand.NEXT)]

    def test_no_player_asks_the_user_to_open_one(self) -> None:
        set_sessions(RecordingSessions([]))
        with pytest.raises(ActionUnavailable) as info:
            run(YMPause)
        assert info.value.user_message == "Не нашёл запущенный плеер. Открой Яндекс Музыку."

    def test_a_refusal_is_reported_in_russian_naming_the_command(self) -> None:
        # Так отвечает плеер, у которого нет следующего трека: False, не исключение.
        set_sessions(RecordingSessions([PLAYING], accept=False))
        with pytest.raises(ActionError) as info:
            run(YMNext)
        assert info.value.user_message == "Плеер не смог следующий трек."

    def test_a_missing_backend_is_the_same_as_no_player(self) -> None:
        set_sessions(NullSessions())
        with pytest.raises(ActionUnavailable):
            run(YMPlay)


class TestMediaKeys:
    """Фоллбек на медиа-клавиши: его нет, пока его не включили."""

    def test_it_is_off_by_default(self) -> None:
        assert MediaActionsConfig().media_keys_fallback is False

    def test_off_means_a_refusal_rather_than_a_keystroke(
        self,
        monkeypatch: pytest.MonkeyPatch,
        keyboard: RecordingBackend,
    ) -> None:
        use_config(monkeypatch, config(media_keys_fallback=False))
        set_sessions(RecordingSessions([]))
        with pytest.raises(ActionUnavailable):
            run(YMNext)
        assert keyboard.events == []

    @pytest.mark.parametrize(
        ("action_class", "vk"),
        [(YMPlay, 0xB3), (YMNext, 0xB0), (YMPrev, 0xB1)],
    )
    def test_on_it_sends_the_dedicated_media_key_and_nothing_else(
        self,
        monkeypatch: pytest.MonkeyPatch,
        keyboard: RecordingBackend,
        action_class: Any,
        vk: int,
    ) -> None:
        use_config(monkeypatch, config(media_keys_fallback=True))
        set_sessions(RecordingSessions([]))
        result = run(action_class)
        assert [event.vk for event in keyboard.events] == [vk, vk]
        assert [event.kind for event in keyboard.events] == ["key_down", "key_up"]
        assert result.data == {"app_id": "", "via_keys": True}

    def test_a_player_that_refuses_falls_back_too(
        self,
        monkeypatch: pytest.MonkeyPatch,
        keyboard: RecordingBackend,
    ) -> None:
        use_config(monkeypatch, config(media_keys_fallback=True))
        set_sessions(RecordingSessions([PLAYING], accept=False))
        assert run(YMNext).data["via_keys"] is True
        assert [event.vk for event in keyboard.events] == [0xB0, 0xB0]


class TestNoKeys:
    """Требование задачи, а не стиль: нажатие уходит туда, где фокус."""

    #: Все четыре модуля пакета.
    MODULES = ("__init__.py", "smtc.py", "cdp.py", "yandex_music.py")

    @staticmethod
    def imported_modules(source: str, inside: str = "") -> list[str]:
        """Имена импортируемых модулей — всего файла или одной функции в нём.

        Разбор через :mod:`ast`, а не поиск подстроки: слова ``SendInput`` и
        ``input.keys`` встречаются в комментариях и докстрингах этих модулей
        именно потому, что там объясняется, почему их тут нет.
        """
        tree: ast.AST = ast.parse(source)
        if inside:
            tree = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == inside
            )
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.append(node.module)
        return found

    def test_three_of_the_four_modules_never_import_the_keyboard(self) -> None:
        for name in ("__init__.py", "cdp.py", "yandex_music.py"):
            imported = self.imported_modules(source_text(name))
            assert not [item for item in imported if "actions.input" in item], name

    def test_the_fourth_imports_it_only_inside_the_fallback(self) -> None:
        source = source_text("smtc.py")
        everywhere = [item for item in self.imported_modules(source) if "actions.input" in item]
        inside = [
            item
            for item in self.imported_modules(source, inside="_press_media_key")
            if "actions.input" in item
        ]
        assert everywhere == inside
        assert sorted(inside) == [
            "ayris.actions.input.backend",
            "ayris.actions.input.keys",
        ]

    def test_no_javascript_template_synthesises_a_key(self) -> None:
        for name, text in js_templates().items():
            for word in ("KeyboardEvent", "dispatchKeyEvent", "keydown", "keyCode", "insertText"):
                assert word not in text, f"{name} синтезирует клавишу {word}"


def reader_over(raw: bytes) -> Callable[[int], bytes]:
    """``read(n)`` над готовым буфером, как его ждёт :func:`decode_frame`."""
    view = bytearray(raw)

    def read(count: int) -> bytes:
        chunk = bytes(view[:count])
        del view[:count]
        return chunk

    return read


def server_frame(payload: bytes, *, opcode: int = OP_TEXT, final: bool = True) -> bytes:
    """Кадр так, как его посылает сервер: без маски, поэтому руками.

    :func:`encode_frame` маскирует всегда — она клиентская, и это правильно, —
    поэтому серверную сторону тест собирает сам.
    """
    header = bytearray([(0x80 if final else 0x00) | opcode])
    size = len(payload)
    if size < 126:
        header.append(size)
    elif size <= 0xFFFF:
        header.append(126)
        header += size.to_bytes(2, "big")
    else:
        header.append(127)
        header += size.to_bytes(8, "big")
    return bytes(header) + payload


class TestFrames:
    """Кадры WebSocket. Ошибка здесь выглядит как молча закрытый сокет."""

    def test_the_rfc_example_key_gives_the_rfc_example_token(self) -> None:
        # RFC 6455, раздел 1.3 — вектор из самого стандарта.
        assert accept_token("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    @pytest.mark.parametrize("size", [0, 5, 125, 126, 127, 0xFFFF, 0x10000])
    def test_a_frame_survives_a_round_trip_at_every_length_boundary(self, size: int) -> None:
        payload = bytes(index % 251 for index in range(size))
        frame = decode_frame(reader_over(encode_frame(payload)))
        assert frame.payload == payload
        assert frame.opcode == OP_TEXT
        assert frame.final is True

    def test_a_client_frame_is_always_masked_and_final(self) -> None:
        mask = b"\x01\x02\x03\x04"
        raw = encode_frame(b"ping", mask=mask)
        assert raw[0] & 0x80, "FIN не выставлен"
        assert raw[1] & 0x80, "бит маски не выставлен"
        assert raw[2:6] == mask
        assert raw[6:] == bytes(a ^ b for a, b in zip(b"ping", mask, strict=True))

    def test_a_mask_of_the_wrong_length_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="4 bytes"):
            encode_frame(b"x", mask=b"\x01\x02")

    def test_an_unmasked_server_frame_is_read_as_it_is(self) -> None:
        frame = decode_frame(reader_over(server_frame(b'{"id":1}')))
        assert frame.payload == b'{"id":1}'

    def test_an_absurd_length_is_refused_before_anything_is_read(self) -> None:
        header = bytes([0x81, 127]) + (MAX_FRAME_BYTES + 1).to_bytes(8, "big")
        with pytest.raises(CdpError, match="refusing"):
            decode_frame(reader_over(header))

    def test_utf8_survives_the_round_trip(self) -> None:
        text = "Сплин — Выхода нет"
        frame = decode_frame(reader_over(encode_frame(text.encode("utf-8"))))
        assert frame.payload.decode("utf-8") == text


class TestTargets:
    """``/json/list`` в цели, и выбор страницы приложения среди них."""

    def test_only_pages_with_a_socket_survive(self) -> None:
        assert [target.id for target in parse_targets(TARGETS_JSON)] == ["5A8", "B21"]

    def test_the_music_page_is_recognised_by_its_scheme(self) -> None:
        targets = parse_targets(TARGETS_JSON)
        assert targets[0].is_music is True
        assert targets[1].is_music is False

    def test_the_music_page_is_chosen_even_when_it_is_not_first(self) -> None:
        targets = parse_targets(TARGETS_JSON)
        assert pick_page(tuple(reversed(targets))) is targets[0]

    def test_without_a_music_page_the_first_page_is_taken(self) -> None:
        other = PageTarget(id="B21", title="", url="about:blank", ws_url="ws://x/1")
        assert pick_page([other]) is other

    def test_no_pages_is_no_page(self) -> None:
        assert pick_page([]) is None

    def test_a_non_json_answer_is_a_missing_port_not_a_crash(self) -> None:
        with pytest.raises(CdpUnavailable, match="not JSON"):
            parse_targets("<html>404</html>")

    def test_a_json_object_is_not_a_target_list(self) -> None:
        with pytest.raises(CdpUnavailable, match="expected a list"):
            parse_targets('{"error": "nope"}')

    def test_the_music_prefix_is_the_electron_scheme(self) -> None:
        assert MUSIC_PAGE_PREFIX == "music-application://"


class FakeDevTools:
    """DevTools на loopback: ``/json/list`` по HTTP и настоящее рукопожатие WS.

    Настоящий сокет, а не мок, ради одного утверждения: у запроса не должно быть
    заголовка ``Origin``. Проверить это можно только там, где заголовки правда
    уходят в сеть, поэтому сервер записывает их в :attr:`ws_headers`.

    Соединения обслуживаются по одному и последовательно — этого хватает: клиент
    сначала спрашивает ``/json/list``, потом открывает сокет.
    """

    def __init__(
        self,
        *,
        answers: Sequence[bytes] = (),
        upgrade: str = "HTTP/1.1 101 Switching Protocols",
        accept: str = "",
        ping_first: bool = False,
        close_first: bool = False,
        mute: bool = False,
    ) -> None:
        self.answers = list(answers)
        self.upgrade = upgrade
        self.accept = accept
        self.ping_first = ping_first
        self.close_first = close_first
        self.mute = mute
        self.ws_headers: dict[str, str] = {}
        self.requests: list[str] = []
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(10.0)
        self.port: int = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, name="fake-devtools", daemon=True)
        self._stop = threading.Event()

    def __enter__(self) -> FakeDevTools:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        # Разбудить accept(): своё же соединение дешевле, чем ждать таймаут.
        with contextlib.suppress(OSError), socket.socket() as waker:
            waker.settimeout(1.0)
            waker.connect(("127.0.0.1", self.port))
        self._thread.join(timeout=10.0)
        self._server.close()

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/devtools/page/5A8"

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except (TimeoutError, OSError):
                return
            with client, contextlib.suppress(OSError):
                self._handle(client)

    def _handle(self, client: socket.socket) -> None:
        client.settimeout(10.0)
        pending = bytearray()
        head = self._read_head(client, pending)
        if not head:
            return
        lines = head.decode("latin-1").split("\r\n")
        self.requests.append(lines[0])
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            if name:
                headers[name.strip().casefold()] = value.strip()
        if "/json/list" in lines[0]:
            body = TARGETS_JSON.replace("9222", str(self.port)).encode("utf-8")
            # «Connection: close» здесь не украшение: без него http.client оставляет
            # сокет открытым для повторного использования, и он всплывает
            # ResourceWarning'ом посреди следующего теста.
            client.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\n\r\n"
                + body
            )
            return
        self.ws_headers = headers
        if self.mute:
            return  # приняли соединение и молчим: порт живой, рукопожатия не будет
        token = self.accept or accept_token(headers.get("sec-websocket-key", ""))
        client.sendall(
            f"{self.upgrade}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {token}\r\n\r\n".encode("latin-1")
        )
        if not self.upgrade.endswith("101 Switching Protocols"):
            return
        if self.close_first:
            # Сокет держим открытым: иначе клиентский sendall упал бы по RST и
            # тест проверял бы «не смог отправить» вместо «страница отключилась».
            client.sendall(server_frame(b"", opcode=OP_CLOSE))
            with contextlib.suppress(OSError):
                client.recv(4096)
            return
        if self.ping_first:
            client.sendall(server_frame(b"hi", opcode=OP_PING))
        read = self._reader(client, pending)
        for answer in self.answers:
            frame = decode_frame(read)
            while frame.opcode not in (OP_TEXT, OP_CLOSE):
                frame = decode_frame(read)  # PONG на наш PING — это не запрос
            if frame.opcode != OP_TEXT:
                return  # клиент прощается своим CLOSE — отвечать больше нечем
            request = json.loads(frame.payload)
            # Номер запроса здесь всегда одноцифровой — клиент в каждом тесте свой, —
            # поэтому подстановка не меняет длину кадра и не рвёт заголовок.
            client.sendall(answer.replace(b'"id":0', f'"id":{request["id"]}'.encode("ascii")))
        self._wait_for_goodbye(client)

    @staticmethod
    def _wait_for_goodbye(client: socket.socket) -> None:
        """Дочитать всё, что клиент ещё пришлёт, и только потом закрыть сокет.

        Закрытие с непрочитанными данными в буфере — это RST, а RST на Windows
        стирает у клиента и уже присланный, но не разобранный ответ: тест падает
        с ``WinError 10054`` вместо проверки ответа.
        """
        with contextlib.suppress(OSError):
            client.settimeout(5.0)
            while client.recv(65536):
                pass

    @staticmethod
    def _reader(client: socket.socket, pending: bytearray) -> Callable[[int], bytes]:
        """``read(n)`` над сокетом, начиная с того, что уже пришло раньше."""

        def read(count: int) -> bytes:
            while len(pending) < count:
                chunk = client.recv(65536)
                if not chunk:
                    raise OSError("клиент ушёл посреди кадра")
                pending.extend(chunk)
            taken = bytes(pending[:count])
            del pending[:count]
            return taken

        return read

    @staticmethod
    def _read_head(client: socket.socket, pending: bytearray) -> bytes:
        """Заголовки запроса. Всё, что пришло за ними, остаётся в ``pending``.

        Первый кадр WebSocket может приехать тем же ``recv``, что и рукопожатие:
        выбросить остаток — значит потерять запрос и подвесить тест.
        """
        while b"\r\n\r\n" not in pending:
            chunk = client.recv(4096)
            if not chunk:
                return b""
            pending += chunk
        head, _, rest = bytes(pending).partition(b"\r\n\r\n")
        pending[:] = rest
        return head


def compact(message: dict[str, Any]) -> bytes:
    """JSON без пробелов: сервер подменяет в нём ``"id":0`` на настоящий номер."""
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def evaluated(value: Any) -> bytes:
    """Ответ ``Runtime.evaluate`` со значением ``value``, как кадр сервера."""
    return server_frame(
        compact({"id": 0, "result": {"result": {"type": "object", "value": value}}})
    )


class TestDevTools:
    """Живой сокет: рукопожатие, отсутствие ``Origin``, ping, фрагменты, отказы."""

    def test_a_value_comes_back_from_the_page(self) -> None:
        with (
            FakeDevTools(answers=[evaluated({"ok": True, "label": "сплин"})]) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
        ):
            assert client.evaluate("1 + 1") == {"ok": True, "label": "сплин"}

    def test_the_handshake_carries_no_origin(self) -> None:
        # Chromium 111 и новее отвергает DevTools-сокет с чужим Origin, а
        # готовые библиотеки WebSocket посылают его по умолчанию.
        with FakeDevTools(answers=[evaluated(1)]) as server:
            with CdpClient(server.ws_url, timeout=5.0) as client:
                client.evaluate("1")
            assert "origin" not in server.ws_headers
            assert server.ws_headers["upgrade"] == "websocket"
            assert server.ws_headers["sec-websocket-version"] == "13"
            assert "sec-websocket-extensions" not in server.ws_headers

    def test_a_ping_is_answered_and_does_not_confuse_the_reply(self) -> None:
        with (
            FakeDevTools(answers=[evaluated("ок")], ping_first=True) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
        ):
            assert client.evaluate("1") == "ок"

    def test_an_event_before_the_answer_is_skipped(self) -> None:
        event = server_frame(compact({"method": "Page.loadEventFired"}))
        with (
            FakeDevTools(answers=[event + evaluated("ок")]) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
        ):
            assert client.evaluate("1") == "ок"

    def test_a_fragmented_answer_is_reassembled(self) -> None:
        body = compact({"id": 0, "result": {"result": {"value": "склеено"}}})
        half = len(body) // 2
        pieces = server_frame(body[:half], final=False) + server_frame(
            body[half:], opcode=0x0, final=True
        )
        with (
            FakeDevTools(answers=[pieces]) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
        ):
            assert client.evaluate("1") == "склеено"

    def test_a_refused_upgrade_reads_as_a_missing_port(self) -> None:
        with (
            FakeDevTools(upgrade="HTTP/1.1 403 Forbidden") as server,
            pytest.raises(CdpUnavailable, match="403"),
        ):
            CdpClient(server.ws_url, timeout=5.0).connect()

    def test_a_wrong_accept_token_is_refused(self) -> None:
        # Токен именно ASCII: заголовки уезжают latin-1, и кириллица здесь уронила бы
        # сам сервер вместо проверки ветки с неверным ответом.
        with (
            FakeDevTools(accept="0000000000000000000000000000=") as server,
            pytest.raises(CdpUnavailable, match="Accept"),
        ):
            CdpClient(server.ws_url, timeout=5.0).connect()

    def test_a_handshake_that_dies_halfway_leaves_no_socket_behind(self) -> None:
        # Порт принимает соединение и молчит. Раньше `connect()` ловил только
        # `OSError` и `CdpUnavailable`, а `CdpError` из оборванного чтения уносил
        # исключение наружу вместе с открытым сокетом.
        with FakeDevTools(mute=True) as server:
            client = CdpClient(server.ws_url, timeout=1.0)
            with pytest.raises((CdpError, CdpUnavailable)):
                client.connect()
            assert client.connected is False

    def test_a_javascript_exception_becomes_a_readable_error(self) -> None:
        body = compact(
            {
                "id": 0,
                "result": {
                    "result": {"type": "object"},
                    "exceptionDetails": {
                        "text": "Uncaught",
                        "exception": {"description": "TypeError: el is null"},
                    },
                },
            }
        )
        with (
            FakeDevTools(answers=[server_frame(body)]) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
            pytest.raises(CdpError, match="TypeError"),
        ):
            client.evaluate("null.click()")

    def test_a_protocol_error_names_the_method(self) -> None:
        body = compact({"id": 0, "error": {"code": -32000, "message": "не вышло"}})
        with (
            FakeDevTools(answers=[server_frame(body)]) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
            pytest.raises(CdpError, match="Runtime.evaluate"),
        ):
            client.evaluate("1")

    def test_a_page_that_hangs_up_is_reported_as_such(self) -> None:
        with (
            FakeDevTools(close_first=True) as server,
            CdpClient(server.ws_url, timeout=5.0) as client,
            pytest.raises(CdpError, match="closed"),
        ):
            client.evaluate("1")

    def test_the_page_is_found_over_http_and_then_connected_to(self) -> None:
        with FakeDevTools(answers=[evaluated("нашлась")]) as server:
            client = cdp.connect(server.port, timeout=5.0)
            try:
                assert client.evaluate("1") == "нашлась"
            finally:
                client.close()
            assert any("/json/list" in line for line in server.requests)

    def test_closing_twice_is_harmless(self) -> None:
        with FakeDevTools(answers=[evaluated(1)]) as server:
            client = CdpClient(server.ws_url, timeout=5.0)
            client.connect()
            client.close()
            client.close()
            assert client.connected is False

    def test_an_open_port_is_seen_as_open(self) -> None:
        with FakeDevTools() as server:
            assert cdp.is_port_open(server.port, timeout=1.0) is True

    def test_a_closed_port_is_seen_as_closed_and_lists_nothing(self) -> None:
        with FakeDevTools() as server:
            port = server.port
        assert cdp.is_port_open(port, timeout=0.3) is False
        with pytest.raises(CdpUnavailable):
            cdp.list_targets(port, timeout=0.3)


class TestAttach:
    """Что происходит, когда отладочного порта нет."""

    def test_a_running_app_without_the_port_is_not_restarted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        launcher = RecordingLauncher()
        set_launcher(launcher)
        use_config(monkeypatch, config(launch_app=True))
        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: False)
        set_sessions(RecordingSessions([PLAYING]))
        with pytest.raises(CdpUnavailable) as info:
            run(YMWave)
        assert "Закрой и снова открой её через Ayris" in (info.value.user_message or "")
        assert launcher.requests == [], "приложение перезапускали за спиной пользователя"

    def test_launching_disabled_says_the_app_is_not_running(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_config(monkeypatch, config(launch_app=False))
        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: False)
        set_sessions(RecordingSessions([]))
        with pytest.raises(CdpUnavailable) as info:
            run(YMLike)
        assert info.value.user_message == "Яндекс Музыка не запущена."

    def test_the_app_is_launched_with_both_debug_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        launcher = RecordingLauncher()
        set_launcher(launcher)
        use_config(
            monkeypatch,
            config(launch_app=True, launch_timeout_s=1.0, player_path="C:/Music/Music.exe"),
        )
        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: False)
        set_sessions(RecordingSessions([]))
        with pytest.raises(CdpUnavailable) as info:
            run(YMLike)
        assert info.value.user_message == "Яндекс Музыка не успела запуститься."
        assert len(launcher.requests) == 1
        assert "--remote-debugging-port=9222" in launcher.requests[0].arguments
        assert "--remote-allow-origins=*" in launcher.requests[0].arguments

    def test_a_port_that_opens_while_waiting_is_used(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        set_launcher(RecordingLauncher())
        use_config(
            monkeypatch,
            config(launch_app=True, launch_timeout_s=5.0, player_path="C:/Music/Music.exe"),
        )
        answers = iter([False, True])
        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: next(answers, True))
        # Плеер не запущен: иначе сработала бы ветка «работает без отладочного порта».
        set_sessions(RecordingSessions([]))
        closed: list[str] = []

        class Stub:
            def evaluate(self, expression: str, **_kw: Any) -> Any:
                # Первым спрашивают про роутер: пока он не поднялся, команду не
                # отправляют вообще. Заглушка отвечает, что интерфейс готов.
                if expression == ym_module._JS_READY:
                    return True
                return {"ok": True, "liked": True, "changed": True}

            def close(self) -> None:
                closed.append("closed")

        monkeypatch.setattr(cdp, "connect", lambda *_a, **_kw: Stub())
        assert run(YMLike).message_ru == "Поставил лайк."
        assert closed == ["closed"], "соединение не закрыли"

    def test_the_flags_are_exactly_the_two_that_are_needed(self) -> None:
        assert debug_flags(9333) == "--remote-debugging-port=9333 --remote-allow-origins=*"

    def test_the_router_is_asked_about_before_any_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Отладочный порт открывается раньше интерфейса: сразу после холодного
        # старта окно показывает заставку, `window.next.router` ещё нет, и первая
        # команда падала бы «не получилось перейти по странице» — при том что
        # приложение поднялось нормально, просто на полсекунды позже.
        asked: list[str] = []

        class Stub:
            def __init__(self) -> None:
                self.ready = iter([False, False, True])

            def evaluate(self, expression: str, **_kw: Any) -> Any:
                asked.append(expression)
                if expression == ym_module._JS_READY:
                    return next(self.ready, True)
                return {"ok": True, "preset": "", "reset": ""}

            def close(self) -> None:
                asked.append("close")

        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: True)
        monkeypatch.setattr(cdp, "connect", lambda *_a, **_kw: Stub())
        monkeypatch.setattr(ym_module, "time", FakeClock())
        set_sessions(RecordingSessions([PLAYING]))
        assert run(YMWave).message_ru == "Включаю Мою волну."
        assert asked.count(ym_module._JS_READY) == 3
        assert asked[-1] == "close"

    def test_an_interface_that_never_loads_is_a_readable_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed: list[str] = []

        class Stub:
            def evaluate(self, expression: str, **_kw: Any) -> Any:
                assert expression == ym_module._JS_READY, "команду отправили в незагруженное окно"
                return False

            def close(self) -> None:
                closed.append("closed")

        clock = FakeClock()
        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: True)
        monkeypatch.setattr(cdp, "connect", lambda *_a, **_kw: Stub())
        monkeypatch.setattr(ym_module, "time", clock)
        use_config(monkeypatch, config(launch_timeout_s=1.0))
        set_sessions(RecordingSessions([PLAYING]))
        with pytest.raises(CdpUnavailable) as info:
            run(YMLike)
        assert info.value.user_message == "Яндекс Музыка ещё загружается. Попробуй ещё раз."
        assert closed == ["closed"], "сокет остался открытым"
        assert sum(clock.slept) >= 1.0, "сдались раньше отведённого времени"

    def test_a_context_replaced_while_loading_is_not_a_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Пока страница грузится, контекст исполнения могут заменить, и `evaluate`
        # отвечает ошибкой протокола. Это не отказ порта, а середина загрузки.
        answers: list[Any] = [CdpError("Cannot find context with specified id"), True]

        class Stub:
            def evaluate(self, expression: str, **_kw: Any) -> Any:
                if expression == ym_module._JS_READY:
                    answer = answers.pop(0)
                    if isinstance(answer, Exception):
                        raise answer
                    return answer
                return {"ok": True, "liked": True, "changed": True}

            def close(self) -> None:
                pass

        monkeypatch.setattr(cdp, "is_port_open", lambda *_a, **_kw: True)
        monkeypatch.setattr(cdp, "connect", lambda *_a, **_kw: Stub())
        monkeypatch.setattr(ym_module, "time", FakeClock())
        set_sessions(RecordingSessions([PLAYING]))
        assert run(YMLike).message_ru == "Поставил лайк — Сплин — Выхода нет."
        assert answers == [], "второй попытки не было — отказались после первой ошибки"


class TestSelectors:
    """Таблица селекторов как данные: полнота, форма, переопределение."""

    def test_every_selector_is_a_trimmed_css_selector(self) -> None:
        for name, value in SELECTORS.items():
            assert value, name
            assert value.strip() == value, name
            assert "{" not in value, name

    def test_everything_is_addressed_by_data_test_id(self) -> None:
        # Раньше карточки альбома и плейлиста в выдаче опознавались по href, и это
        # было мимо: `a[href*="playlist"]` попадал в закладки боковой панели, а
        # `a[href*="/album"]` — в ссылку /album/track?… внутри карточки трека. У
        # карточек есть свои HORIZONTAL_*_CARD, проверено на живом приложении.
        by_href = {name for name, value in SELECTORS.items() if "data-test-id" not in value}
        assert by_href == set()

    def test_a_play_selector_ends_at_a_button(self) -> None:
        for name, value in SELECTORS.items():
            if name.endswith(("_play", "_pause")):
                assert value.endswith(('PLAY_BUTTON"]', 'PAUSE_BUTTON"]')), name

    def test_the_settings_file_can_replace_one_entry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_config(monkeypatch, config(selectors={"playerbar_like": ".new-like"}))
        assert selector("playerbar_like") == ".new-like"
        assert selector("playerbar_menu") == SELECTORS["playerbar_menu"]

    def test_a_blank_override_falls_back_to_the_built_in_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use_config(monkeypatch, config(selectors={"nav_home": "   "}))
        assert selector("nav_home") == SELECTORS["nav_home"]

    def test_an_unknown_name_is_a_programming_mistake(self) -> None:
        with pytest.raises(KeyError):
            selector("нет такого")

    def test_every_name_the_code_asks_for_exists_in_the_table(self) -> None:
        # Опечатка в имени иначе всплыла бы только на живом приложении.
        asked = set(re.findall(r'selector\("([a-z_]+)"\)', source_text("yandex_music.py")))
        assert asked, "тест перестал что-либо находить — изменился синтаксис вызова"
        assert asked <= set(SELECTORS), sorted(asked - set(SELECTORS))

    def test_the_home_route_is_deliberately_absent(self) -> None:
        # Роутер приложения игнорирует push('/'), главная открывается кликом.
        assert "/" not in ROUTES.values()
        assert set(ROUTES) == {"search", "playlist", "artist", "album", "track", "collection"}

    def test_every_kind_of_search_names_a_card_and_optional_lines(self) -> None:
        for kind, plan in ym_module._SEARCH_PLAN.items():
            card, label, extra = plan
            assert card in SELECTORS, kind
            for name in (label, extra):
                assert name == "" or name in SELECTORS, (kind, name)

    def test_a_card_selector_is_a_card_and_a_line_is_a_line(self) -> None:
        # Перепутанные местами карточка и заголовок — это «включи» по всей выдаче
        # сразу, а такое видно только на живом приложении.
        for kind, (card, label, extra) in ym_module._SEARCH_PLAN.items():
            assert "CARD" in SELECTORS[card], kind
            for name in (label, extra):
                assert name == "" or "TITLE" in SELECTORS[name], (kind, name)

    def test_the_bottom_bar_of_the_home_page_is_a_second_bar_with_its_own_names(self) -> None:
        # У главной страницы нижняя панель своя, и внутри неё свои имена. Один раз
        # здесь стоял `TRACK_CONTEXT_MENU_BUTTON` — его в приложении нет вообще, и
        # «добавь в плейлист» с главной было нечем сделать. Имя ниже снято с живого
        # приложения 5.115.3; если оно опять уедет, пусть падает тест, а не действие.
        assert SELECTORS["vibe_menu"] == (
            '[data-test-id="VIBE_PLAYERBAR"] [data-test-id="VIBE_CONTEXT_MENU_BUTTON"]'
        )
        vibe = {name: value for name, value in SELECTORS.items() if name.startswith("vibe_")}
        inside_the_bar = {name for name, value in vibe.items() if "VIBE_PLAYERBAR" in value}
        assert inside_the_bar == {
            "vibe_playerbar",  # сама панель, остальные четыре — внутри неё
            "vibe_play",
            "vibe_pause",
            "vibe_like",
            "vibe_menu",
        }

    def test_the_two_bars_are_never_confused_with_each_other(self) -> None:
        # Скрипты пробуют сначала обычную панель, потом волновую, поэтому селектор
        # каждой должен отвечать только за свою: панель, найденная в обеих, — это
        # клик не туда на одной из страниц.
        for name, value in SELECTORS.items():
            if name.startswith("playerbar"):
                assert "VIBE_PLAYERBAR" not in value, name
            if name.startswith("vibe_") and "PLAYERBAR" in value:
                assert "PLAYERBAR_DESKTOP" not in value, name


class TestScripts:
    """Что именно уезжает в страницу. Скрипт — это данные, поэтому его читают."""

    def test_search_puts_the_query_in_the_route_and_never_types_it(self) -> None:
        transport = answering({"ok": True, "label": "сплин", "via": "card"})
        run(YMSearch, query="Сплин", kind="artist")
        script = transport.expressions[0]
        assert "/search?text=Сплин" in script
        assert as_literal("search_artist_card") in script
        assert ".click()" in script

    @pytest.mark.parametrize("kind", list(SearchKind))
    def test_every_kind_of_search_sends_its_own_card_selector(self, kind: SearchKind) -> None:
        transport = answering({"ok": True, "label": "что-то"})
        run(YMSearch, query="что-то", kind=str(kind))
        card_name = ym_module._SEARCH_PLAN[kind][0]
        assert as_literal(card_name) in transport.expressions[0]

    @pytest.mark.parametrize(
        ("action_class", "params"),
        [
            (YMSearch, {"query": "Сплин"}),
            (YMPlaylist, {"name": "Любимка"}),
            (YMWave, {}),
            (YMLike, {}),
            (YMAddToPlaylist, {"playlist": "Любимка"}),
        ],
    )
    def test_no_placeholder_is_left_unsubstituted(
        self,
        action_class: Any,
        params: dict[str, Any],
    ) -> None:
        transport = answering({"ok": True, "label": "x", "liked": True, "changed": True})
        run(action_class, **params)
        script = transport.expressions[0]
        assert "$" not in script, "остался незаполненный placeholder"
        assert "undefined" not in script

    def test_navigation_goes_through_the_router_and_never_reloads(self) -> None:
        templates = js_templates()
        assert "window.next && window.next.router" in templates["_PRELUDE"]
        # location.assign и Page.navigate перезагружают страницу и глушат музыку.
        for name, text in templates.items():
            for word in ("location.assign", "location.href", "location.reload", "Page.navigate"):
                assert word not in text, f"{name} перезагружает страницу через {word}"

    def test_the_wave_clicks_reset_when_no_preset_was_asked_for(self) -> None:
        transport = answering({"ok": True, "preset": "", "reset": "популярные треки артиста"})
        run(YMWave)
        script = transport.expressions[0]
        assert as_literal("vibe_reset") in script
        # Именно `if (true)`: сброс контекста включён, а значит кнопку нажмут.
        assert "if (true)" in script

    def test_a_preset_suppresses_the_reset_because_it_sets_its_own_context(self) -> None:
        transport = answering({"ok": True, "preset": "по артисту сплин"})
        run(YMWave, preset="по артисту Сплин")
        assert "if (false)" in transport.expressions[0]

    def test_the_reset_can_also_be_switched_off_by_hand(self) -> None:
        transport = answering({"ok": True, "preset": ""})
        run(YMWave, reset_context=False)
        assert "if (false)" in transport.expressions[0]

    @pytest.mark.parametrize(
        ("mode", "wanted"),
        [(LikeMode.LIKE, "true"), (LikeMode.UNLIKE, "false"), (LikeMode.TOGGLE, "null")],
    )
    def test_the_like_mode_reaches_the_page_as_the_state_it_wants(
        self,
        mode: LikeMode,
        wanted: str,
    ) -> None:
        transport = answering({"ok": True, "liked": True, "changed": False})
        run(YMLike, mode=str(mode))
        assert f"before === {wanted}" in transport.expressions[0]

    def test_nothing_is_declared_in_the_page_own_scope(self) -> None:
        # Область видимости у `Runtime.evaluate` одна на все соединения и живёт до
        # перезагрузки. Объявленный на верхнем уровне `const sleep` переживает вызов,
        # и вторая команда пользователя падала бы с «Identifier 'sleep' has already
        # been declared» — то есть работал бы ровно один запрос за запуск приложения.
        for name, script in js_scripts().items():
            assert script.startswith("(() => {"), name
            assert script.endswith(";})()"), name

    def test_every_script_carries_the_helpers_with_it(self) -> None:
        prelude = js_templates()["_PRELUDE"]
        for name, script in js_scripts().items():
            assert prelude in script, name

    def test_the_readiness_question_needs_no_helpers_at_all(self) -> None:
        # Его задают до того, как страница дозагрузилась, и отвечать он должен
        # мгновенно: ни `await`, ни помощников, ни обёртки.
        assert ym_module._JS_READY == "!!(window.next && window.next.router)"

    def test_the_search_fingerprints_the_old_results_before_navigating(self) -> None:
        # Роутер меняет данные страницы, а не её DOM: секунду после `push` в выдаче
        # лежат карточки прошлого запроса, и `waitFor` находит их мгновенно. Отпечаток
        # снимается до перехода, иначе «включи Сплина» включит Кино.
        script = js_scripts()["_JS_START"]
        body = script[script.index("(async () =>") :]
        assert body.index("mark(") < body.index("route("), "отпечаток снят после перехода"
        assert "settle(" in body

    def test_a_track_is_matched_together_with_its_artist(self) -> None:
        # «включи Сплин Выхода нет» — это карточка, в заголовке которой Сплина нет.
        transport = answering({"ok": True, "label": "Выхода нет", "via": "card"})
        run(YMSearch, query="Сплин Выхода нет", kind="track")
        assert as_literal("artist_title") in transport.expressions[0]

    def test_a_search_without_a_second_line_sends_an_empty_selector(self) -> None:
        transport = answering({"ok": True, "label": "Кино", "via": "card"})
        run(YMSearch, query="Кино", kind="artist")
        # Пустая строка вместо селектора — это «второй строки у карточки нет», и
        # скрипт обязан такое пережить, а не искать элемент по селектору "".
        assert '""' in transport.expressions[0]
        assert as_literal("artist_title") not in transport.expressions[0]

    def test_the_playlist_menu_is_closed_after_the_click(self) -> None:
        # Приложение закрывает меню само, но если перестанет — открытое подменю
        # останется висеть поверх интерфейса, и это увидит пользователь.
        script = js_scripts()["_JS_ADD_TO_PLAYLIST"]
        assert script.count("closeMenus();") == 3, "ветка без закрытия меню"
        assert script.index("found.el.click()") < script.rindex("closeMenus();")

    def test_the_menu_is_looked_for_in_both_bars(self) -> None:
        # «Добавь в плейлист» одинаково нужен и на странице трека, и на главной, а
        # панели там разные. Пропавший из скрипта второй селектор — это отказ
        # «ничего не играет» ровно на половине страниц.
        script = js_scripts()["_JS_ADD_TO_PLAYLIST"]
        assert "$playerbar_menu" in script and "$vibe_menu" in script
        transport = answering({"ok": True, "changed": True, "label": "Бег", "inside": False})
        run(YMAddToPlaylist, playlist="Бег")
        sent = transport.expressions[0]
        assert as_literal("playerbar_menu") in sent
        assert as_literal("vibe_menu") in sent


class TestActions:
    """Пять действий второго уровня: их фразы и их отказы."""

    def test_search_says_what_it_found_not_what_was_asked(self) -> None:
        answering({"ok": True, "label": "сплин", "via": "card"})
        result = run(YMSearch, query="спли", kind="artist")
        assert result.message_ru == "Включаю сплин."
        assert result.data == {"kind": "artist", "query": "спли", "label": "сплин"}

    def test_a_playlist_is_announced_as_a_playlist(self) -> None:
        answering({"ok": True, "label": "любимка", "via": "page"})
        assert run(YMPlaylist, name="Любимка").message_ru == "Включаю плейлист любимка."

    def test_the_plain_wave_is_announced_plainly(self) -> None:
        answering({"ok": True, "preset": "", "reset": ""})
        assert run(YMWave).message_ru == "Включаю Мою волну."

    def test_a_preset_wave_names_the_preset(self) -> None:
        answering({"ok": True, "preset": "радостно на душе"})
        assert run(YMWave, preset="радостно").message_ru == "Включаю волну: радостно на душе."

    @pytest.mark.parametrize(
        ("answer", "message"),
        [
            ({"liked": True, "changed": True}, "Поставил лайк — Сплин — Выхода нет."),
            ({"liked": False, "changed": True}, "Убрал лайк — Сплин — Выхода нет."),
            ({"liked": True, "changed": False}, "Лайк уже стоял — Сплин — Выхода нет."),
            ({"liked": False, "changed": False}, "Лайка и не было — Сплин — Выхода нет."),
        ],
    )
    def test_a_like_says_which_of_the_four_things_happened(
        self,
        sessions: RecordingSessions,
        answer: dict[str, Any],
        message: str,
    ) -> None:
        answering({"ok": True, **answer})
        assert run(YMLike, mode="toggle").message_ru == message

    def test_a_like_without_a_session_says_less_but_still_says_it(self) -> None:
        set_sessions(RecordingSessions([]))
        answering({"ok": True, "liked": True, "changed": True})
        assert run(YMLike).message_ru == "Поставил лайк."

    def test_adding_a_track_names_the_track_and_the_playlist(
        self,
        sessions: RecordingSessions,
    ) -> None:
        answering({"ok": True, "changed": True, "label": "Любимка"})
        result = run(YMAddToPlaylist, playlist="любимк")
        assert result.message_ru == "Добавил Сплин — Выхода нет в Любимка."
        assert result.data["changed"] is True

    def test_a_track_already_in_the_playlist_is_left_alone(
        self,
        sessions: RecordingSessions,
    ) -> None:
        answering({"ok": True, "changed": False, "label": "Любимка", "inside": True})
        result = run(YMAddToPlaylist, playlist="Любимка")
        assert result.message_ru == "Сплин — Выхода нет уже был в плейлисте Любимка."
        assert result.undo_token is None, "нечего отменять — ничего не делали"

    @pytest.mark.parametrize(
        ("reason", "message"),
        [
            ("no-results", "Яндекс Музыка ничего не нашла по запросу."),
            ("no-match", "Не нашёл этого в выдаче Яндекс Музыки."),
            ("no-router", "Не получилось перейти по странице Яндекс Музыки."),
            ("что-то новое", "Яндекс Музыка не ответила так, как ожидалось."),
        ],
    )
    def test_a_machine_reason_is_spoken_as_a_sentence(self, reason: str, message: str) -> None:
        answering({"ok": False, "reason": reason})
        with pytest.raises(ActionError) as info:
            run(YMSearch, query="Сплин")
        assert info.value.user_message == message

    def test_the_seen_list_goes_to_the_log_and_not_to_the_user(self) -> None:
        answering({"ok": False, "reason": "no-playlist", "seen": ["Любимка", "Дорога"]})
        with pytest.raises(ActionError) as info:
            run(YMAddToPlaylist, playlist="нет такого")
        assert info.value.user_message == "Не нашёл такой плейлист."
        assert "Любимка" in str(info.value)

    def test_a_page_that_answers_with_a_non_object_is_a_protocol_error(self) -> None:
        answering("вообще не то")
        with pytest.raises(CdpError, match="expected an object"):
            run(YMWave)

    def test_every_reason_in_the_table_has_a_russian_sentence(self) -> None:
        for reason, message in ym_module._REASONS.items():
            assert message.endswith("."), reason
            assert message[0].isupper(), reason


class TestUndo:
    """Отмена добавления в плейлист — то самое «да, с откатом»."""

    def test_adding_hands_out_a_token_that_remembers_both_names(
        self,
        sessions: RecordingSessions,
    ) -> None:
        answering({"ok": True, "changed": True, "label": "Любимка"})
        result = run(YMAddToPlaylist, playlist="любимк")
        assert result.undo_token == "Любимка|Сплин — Выхода нет"
        assert result.undoable is True

    def test_adding_clicks_only_when_the_track_is_not_in_the_playlist(
        self,
        sessions: RecordingSessions,
    ) -> None:
        transport = answering({"ok": True, "changed": True, "label": "Любимка"})
        run(YMAddToPlaylist, playlist="Любимка")
        assert "inside !== false" in transport.expressions[0]

    def test_undo_clicks_the_same_item_the_other_way_round(
        self,
        sessions: RecordingSessions,
    ) -> None:
        transport = answering({"ok": True, "changed": True, "label": "Любимка"})
        result = YMAddToPlaylist().undo("Любимка|Сплин — Выхода нет")
        assert result.message_ru == "Убрал Сплин — Выхода нет из Любимка."
        # Тот же скрипт, но условие пропуска инвертировано: клик только когда
        # трек в плейлисте уже есть.
        assert "inside !== true" in transport.expressions[0]
        assert as_literal("submenu_in_playlist") in transport.expressions[0]

    def test_undo_refuses_when_a_different_track_is_playing_now(self) -> None:
        # Иначе «отмени» убрало бы из плейлиста не тот трек, а это хуже, чем
        # лишний трек в плейлисте.
        set_sessions(RecordingSessions([BROWSER]))
        transport = answering({"ok": True, "changed": True, "label": "Любимка"})
        with pytest.raises(ActionError) as info:
            YMAddToPlaylist().undo("Любимка|Сплин — Выхода нет")
        assert "Сплин — Выхода нет" in (info.value.user_message or "")
        assert "YouTube — Лекция про кольца" in (info.value.user_message or "")
        assert transport.expressions == [], "страницу всё-таки трогали"

    def test_undo_proceeds_when_nothing_is_playing_at_all(self) -> None:
        # Плеер закрыли — трек всё равно тот, и убрать его безопасно.
        set_sessions(RecordingSessions([]))
        answering({"ok": True, "changed": True, "label": "Любимка"})
        result = YMAddToPlaylist().undo("Любимка|Сплин — Выхода нет")
        assert result.message_ru == "Убрал Сплин — Выхода нет из Любимка."

    def test_a_track_already_gone_is_said_plainly(self, sessions: RecordingSessions) -> None:
        answering({"ok": True, "changed": False, "label": "Любимка", "inside": False})
        result = YMAddToPlaylist().undo("Любимка|Сплин — Выхода нет")
        assert result.message_ru == "Сплин — Выхода нет и так не было в плейлисте Любимка."

    def test_a_stale_token_is_refused_in_russian(self) -> None:
        with pytest.raises(ActionError) as info:
            YMAddToPlaylist().undo("мусор без разделителя")
        assert info.value.user_message == "Не помню, что именно добавлял — не могу отменить."

    def test_the_registry_lets_this_one_be_undone_and_the_others_not(self) -> None:
        registry = ActionRegistry()
        registry.discover()
        assert registry.get("YMAddToPlaylist").meta.supports_undo is True
        with pytest.raises(ActionUnavailable):
            registry.undo("YMLike", "неважно")

    def test_undo_right_after_adding_waits_for_the_icon_to_catch_up(
        self,
        sessions: RecordingSessions,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Замерено на живом приложении: заново открытое меню секунду показывает
        # состояние *до* клика. Без паузы «отмени» прочитало бы трек не добавленным
        # и честно ничего не сняло — а пользователю сказали бы, что отменили.
        clock = FakeClock()
        monkeypatch.setattr(ym_module, "time", clock)
        answering(
            {"ok": True, "changed": True, "label": "Любимка"},
            {"ok": True, "changed": True, "label": "Любимка"},
        )
        run(YMAddToPlaylist, playlist="Любимка")
        assert clock.slept == [], "первый клик ждать нечего"
        YMAddToPlaylist().undo("Любимка|Сплин — Выхода нет")
        assert clock.slept == [ym_module._SETTLE_S]

    def test_a_click_that_changed_nothing_leaves_nothing_to_wait_for(
        self,
        sessions: RecordingSessions,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = FakeClock()
        monkeypatch.setattr(ym_module, "time", clock)
        answering(
            {"ok": True, "changed": False, "label": "Любимка", "inside": True},
            {"ok": True, "changed": True, "label": "Любимка"},
        )
        run(YMAddToPlaylist, playlist="Любимка")
        YMAddToPlaylist().undo("Любимка|Сплин — Выхода нет")
        assert clock.slept == [], "ждали отметку, которая никуда не двигалась"


class TestSchemas:
    """Что редактор макросов нарисует для одиннадцати действий."""

    NAMES = (
        "YMPlay",
        "YMPause",
        "YMToggle",
        "YMNext",
        "YMPrev",
        "NowPlaying",
        "YMSearch",
        "YMPlaylist",
        "YMWave",
        "YMLike",
        "YMAddToPlaylist",
    )

    def test_all_eleven_actions_are_registered(self) -> None:
        registry = ActionRegistry()
        registry.discover()
        assert set(self.NAMES) <= set(registry.names)

    @pytest.mark.parametrize(
        "action_class",
        [YMPlay, YMPause, YMToggle, YMNext, YMPrev, NowPlaying, YMSearch, YMPlaylist, YMWave],
    )
    def test_every_action_describes_itself_in_russian(self, action_class: Any) -> None:
        schema = build_schema(action_class)
        assert schema.title_ru
        assert schema.description_ru
        assert schema.category_title_ru == "Музыка и медиа"
        assert all(field.label_ru or field.description_ru for field in schema.fields)

    def test_the_transport_actions_take_no_parameters(self) -> None:
        for action_class in (YMPlay, YMPause, YMToggle, YMNext, YMPrev, NowPlaying):
            assert build_schema(action_class).fields == ()

    def test_a_search_query_is_required_text(self) -> None:
        field = build_schema(YMSearch).field_by_name("query")
        assert field is not None
        assert field.kind is FieldKind.TEXT
        assert field.required is True

    def test_the_kind_of_search_is_a_choice_of_four(self) -> None:
        field = build_schema(YMSearch).field_by_name("kind")
        assert field is not None
        assert field.kind is FieldKind.CHOICE
        assert {choice.value for choice in field.choices} == {
            "track",
            "artist",
            "album",
            "playlist",
        }

    def test_the_wave_reset_is_a_switch_that_defaults_to_on(self) -> None:
        field = build_schema(YMWave).field_by_name("reset_context")
        assert field is not None
        assert field.kind is FieldKind.BOOLEAN
        assert field.default is True

    def test_no_media_action_is_dangerous_or_needs_admin(self) -> None:
        registry = ActionRegistry()
        registry.discover()
        for name in self.NAMES:
            meta = registry.get(name).meta
            assert meta.is_dangerous is False, name
            assert meta.require_admin is False, name


@pytest.mark.skipif(sys.platform != "win32", reason="SMTC есть только в Windows")
class TestRealSmtc:
    """Настоящий список плееров системы. Ни один не запущен — это тоже ответ."""

    @staticmethod
    def real_backend() -> smtc_module.SessionsBackend:
        set_sessions(None)
        backend = smtc_module.get_sessions()
        if not backend.available:
            pytest.skip("пакеты winrt не установлены")
        return backend

    def test_the_backend_answers_without_a_player_running(self) -> None:
        for session in self.real_backend().sessions():
            assert session.app_id
            assert session.status in set(PlaybackStatus)

    def test_asking_what_plays_never_raises(self) -> None:
        self.real_backend()
        assert now_playing_ru(smtc_module.current_session())

    def test_a_command_to_a_player_that_is_not_there_is_refused_not_crashed(self) -> None:
        backend = self.real_backend()
        assert backend.send("нет.такого.плеера", TransportCommand.PAUSE) is False
