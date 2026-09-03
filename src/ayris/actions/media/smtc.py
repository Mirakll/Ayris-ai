"""The basic level: Windows' own media transport, no debug port and no browser.

``GlobalSystemMediaTransportControlsSessionManager`` is the list of players
Windows knows about — the same list the volume flyout draws — and each entry
answers two questions and takes four orders. What is playing (title, artist,
album, position, duration, and which of the transport buttons the player says are
usable) and whether it is playing; play, pause, next, previous. That is exactly
the half of task 29 that must work without anything special being running, so it
is the half that lives here.

Three things about it are worth knowing before reading the code.

**It is not a keystroke.** A media key is a broadcast: Windows sends it to
whatever it thinks should get it, and the person using Ayris plays games, where a
synthesised key press is a bind. SMTC is the opposite — the order goes to one named
session object, so nothing else on the machine can receive it. The media-key path
below therefore exists only for a player with no session at all, and is off by
default.

**Sessions are re-enumerated for every command.** Holding on to a session object
between calls looks cheaper and is wrong: the player closes and reopens its session
when the track ends, when it is minimised to the tray, when it loses the audio
device. Enumerating costs a WinRT round trip, which is under a millisecond, and it
is the enumeration that tells us whether the player is still there at all.

**The async operations block on the wrong apartment.** Same problem, same fix, as
in :mod:`ayris.actions.system.ocr_engines.windows_ocr`: ``comtypes`` elsewhere in
the process leaves worker threads initialised as single-threaded apartments, and
pywinrt refuses a blocking wait on those. :func:`_await` retries on a thread of its
own, which WinRT initialises as multithreaded.

The actions are named after Яндекс Музыка because that is what they are configured
for — ``[actions.media] player_app_id`` — but nothing here is Yandex-specific: with
the setting cleared they drive whichever player is currently playing, which is the
right behaviour for «поставь на паузу» said over a film.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Final

from ayris.actions.base import Action, ActionCategory, ActionMeta
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.config import get_settings
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils.logger import get_logger

__all__ = [
    "MediaSession",
    "NowPlaying",
    "NullSessions",
    "PlaybackStatus",
    "RecordingSessions",
    "SessionsBackend",
    "TransportCommand",
    "WinRtSessions",
    "YMNext",
    "YMPause",
    "YMPlay",
    "YMPrev",
    "YMToggle",
    "get_sessions",
    "now_playing_ru",
    "pick_session",
    "send_command",
    "set_sessions",
]

_log = get_logger(__name__)

#: The desktop app's identifier in Windows' session list. Not a process name and
#: not a window title: this is what ``SourceAppUserModelId`` reports, verified on
#: the running app.
YANDEX_MUSIC_APP_ID: Final = "ru.yandex.desktop.music"

#: WinRT reports position and duration as ``TimeSpan``, which pywinrt projects as
#: ``timedelta`` — but a build that hands over raw 100-nanosecond ticks is not
#: unheard of, so :func:`_seconds` accepts both. This is the tick scale.
_TICKS_PER_SECOND: Final = 10_000_000

#: Sentinel for pywinrt's apartment complaint, matched on the word: the sentence
#: around it is Microsoft's and may change.
_APARTMENT: Final = "apartment"


class PlaybackStatus(StrEnum):
    """State of one player, mirroring WinRT's ``PlaybackStatus`` enumeration."""

    CLOSED = "closed"
    OPENED = "opened"
    CHANGING = "changing"
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"

    @property
    def title_ru(self) -> str:
        """How this state is named when Ayris has to say it out loud."""
        return _STATUS_TITLES[self]


_STATUS_TITLES: Final[dict[PlaybackStatus, str]] = {
    PlaybackStatus.CLOSED: "закрыт",
    PlaybackStatus.OPENED: "открыт",
    PlaybackStatus.CHANGING: "переключается",
    PlaybackStatus.STOPPED: "остановлен",
    PlaybackStatus.PLAYING: "играет",
    PlaybackStatus.PAUSED: "на паузе",
}

#: WinRT's integer status to ours. The numbers are part of the ABI, so writing
#: them out is safer than importing the projection just to read its members.
_STATUS_BY_VALUE: Final[dict[int, PlaybackStatus]] = {
    0: PlaybackStatus.CLOSED,
    1: PlaybackStatus.OPENED,
    2: PlaybackStatus.CHANGING,
    3: PlaybackStatus.STOPPED,
    4: PlaybackStatus.PLAYING,
    5: PlaybackStatus.PAUSED,
}


class TransportCommand(StrEnum):
    """The orders a transport control can carry."""

    PLAY = "play"
    PAUSE = "pause"
    TOGGLE = "toggle"
    NEXT = "next"
    PREVIOUS = "previous"
    STOP = "stop"

    @property
    def title_ru(self) -> str:
        """Wording for the log line and for a failure the user hears about."""
        return _COMMAND_TITLES[self]


_COMMAND_TITLES: Final[dict[TransportCommand, str]] = {
    TransportCommand.PLAY: "включить",
    TransportCommand.PAUSE: "поставить на паузу",
    TransportCommand.TOGGLE: "переключить пауза/плей",
    TransportCommand.NEXT: "следующий трек",
    TransportCommand.PREVIOUS: "предыдущий трек",
    TransportCommand.STOP: "остановить",
}

#: Key names from :mod:`ayris.actions.input.keys` for the last-resort path. Only
#: dedicated media keys are here: they are the one part of the keyboard that no
#: game binds, and there is nothing to translate a toggle into but the play/pause
#: key, which is what a toggle already is.
_COMMAND_KEYS: Final[dict[TransportCommand, str]] = {
    TransportCommand.PLAY: "mediaplay",
    TransportCommand.PAUSE: "mediaplay",
    TransportCommand.TOGGLE: "mediaplay",
    TransportCommand.NEXT: "medianext",
    TransportCommand.PREVIOUS: "mediaprev",
    TransportCommand.STOP: "mediastop",
}


@dataclass(frozen=True, slots=True)
class MediaSession:
    """One player, as Windows describes it at the moment it was asked.

    A snapshot, not a handle: nothing here can be used to talk to the player
    afterwards, and :attr:`app_id` is how a command finds it again.
    """

    app_id: str
    title: str = ""
    artist: str = ""
    album: str = ""
    status: PlaybackStatus = PlaybackStatus.CLOSED
    position_s: float = 0.0
    duration_s: float = 0.0
    can_play: bool = False
    can_pause: bool = False
    can_next: bool = False
    can_previous: bool = False

    @property
    def playing(self) -> bool:
        """Whether sound is coming out of it right now."""
        return self.status is PlaybackStatus.PLAYING

    @property
    def empty(self) -> bool:
        """Whether the player has a session but nothing loaded in it."""
        return not self.title and not self.artist

    @property
    def label_ru(self) -> str:
        """«Исполнитель — Трек», or whichever half of it is known."""
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.artist

    def as_data(self) -> dict[str, Any]:
        """Flat mapping for the audit row and for the event payload."""
        return {
            "app_id": self.app_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "status": str(self.status),
            "position_s": round(self.position_s, 1),
            "duration_s": round(self.duration_s, 1),
        }


class SessionsBackend:
    """Where «what is playing» and «next track» actually go.

    Three implementations: WinRT on Windows, a refusing one everywhere else, and a
    recording one for the tests. The seam is a class rather than a ``Protocol``
    because the two stubs want the shared docstrings more than they want structural
    typing.
    """

    #: Whether this backend can answer at all. ``False`` makes every action fail
    #: with :class:`~ayris.core.errors.ActionUnavailable` and a Russian reason.
    available: ClassVar[bool] = False

    def sessions(self) -> tuple[MediaSession, ...]:
        """Every player Windows currently knows about, in its own order."""
        raise NotImplementedError

    def send(self, app_id: str, command: TransportCommand) -> bool:
        """Give one order to the player named ``app_id``.

        Returns:
            ``False`` when the player is gone, or refused. WinRT's ``TryX`` methods
            report a refusal by returning ``False`` rather than by failing, which is
            how a player that has no next track answers ``next``.
        """
        raise NotImplementedError


class NullSessions(SessionsBackend):
    """No transport at all: not Windows, or the projections are not installed."""

    available: ClassVar[bool] = False

    def sessions(self) -> tuple[MediaSession, ...]:
        return ()

    def send(self, app_id: str, command: TransportCommand) -> bool:  # noqa: ARG002 - контракт
        return False


class RecordingSessions(SessionsBackend):
    """Writes the orders down instead of sending them. Test seam.

    Built over a fixed list of sessions, so a test says «two players, one of them
    Яндекс Музыка, paused» as data and then asserts on :attr:`commands`.
    """

    available: ClassVar[bool] = True

    def __init__(self, sessions: Sequence[MediaSession] = (), *, accept: bool = True) -> None:
        self._sessions = tuple(sessions)
        self._accept = accept
        self.commands: list[tuple[str, TransportCommand]] = []

    def sessions(self) -> tuple[MediaSession, ...]:
        return self._sessions

    def send(self, app_id: str, command: TransportCommand) -> bool:
        self.commands.append((app_id, command))
        return self._accept and any(item.app_id == app_id for item in self._sessions)


def _await(operation: Any) -> Any:
    """Wait for a WinRT async operation from whichever COM apartment we are on.

    See :func:`ayris.actions.system.ocr_engines.windows_ocr._await`: same trap, and
    it is copied rather than shared because the two modules must not import each
    other for a four-line helper.
    """
    try:
        return operation.get()
    except RuntimeError as exc:
        if _APARTMENT not in str(exc).lower():
            raise
    done: list[Any] = []
    failed: list[Exception] = []

    def wait() -> None:
        try:
            done.append(operation.get())
        except Exception as exc:  # re-raised below, on the thread that asked
            failed.append(exc)

    worker = threading.Thread(target=wait, name="ayris-smtc-await", daemon=True)
    worker.start()
    worker.join()
    if failed:
        raise failed[0]
    return done[0]


def _seconds(value: Any) -> float:
    """A WinRT ``TimeSpan`` as seconds, however this pywinrt projects it."""
    total = getattr(value, "total_seconds", None)
    if callable(total):
        return float(total())
    try:
        return float(value) / _TICKS_PER_SECOND
    except (TypeError, ValueError):
        return 0.0


class _Winrt:
    """``Windows.Media.Control``, imported once and only when it is needed.

    At module level this import would break the whole media package on the Linux
    CI runner, where the ``winrt-*`` wheels are deliberately absent and these tests
    run against :class:`RecordingSessions`.
    """

    loaded: ClassVar[bool] = False
    error: ClassVar[str] = ""
    manager: ClassVar[Any] = None

    @classmethod
    def load(cls) -> bool:
        """Import the projection. ``False`` when it is not installed."""
        if cls.loaded:
            return not cls.error
        cls.loaded = True
        try:
            from winrt.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager,
            )
        except ImportError as exc:  # not Windows, or a trimmed install
            cls.error = str(exc)
            _log.debug("Windows.Media.Control is not importable: %s", exc)
            return False
        except OSError as exc:  # the runtime is there but refuses to activate
            cls.error = str(exc)
            _log.warning("Windows.Media.Control failed to load: %s", exc)
            return False
        cls.manager = GlobalSystemMediaTransportControlsSessionManager
        return True


class WinRtSessions(SessionsBackend):
    """The real thing: ``GlobalSystemMediaTransportControlsSessionManager``."""

    available: ClassVar[bool] = True

    def _manager(self) -> Any:
        """A fresh session manager.

        Asked for on every call on purpose. The manager caches its session list
        internally and a stale one reports a player that closed two songs ago; it is
        a cheap object, and the request is the liveness check.
        """
        if not _Winrt.load():
            raise ActionUnavailable(
                f"winrt.windows.media.control is not importable: {_Winrt.error}",
                user_message=(
                    "Управление плеером не подключено: не установлены пакеты winrt. "
                    "Переустанови Ayris."
                ),
            )
        try:
            return _await(_Winrt.manager.request_async())
        except (OSError, RuntimeError) as exc:
            raise ActionUnavailable(
                f"RequestAsync failed: {exc}",
                user_message="Windows не отдал список плееров.",
            ) from exc

    def sessions(self) -> tuple[MediaSession, ...]:
        manager = self._manager()
        try:
            raw = list(manager.get_sessions())
        except (OSError, RuntimeError) as exc:
            _log.warning("could not list media sessions: %s", exc)
            return ()
        found: list[MediaSession] = []
        for session in raw:
            snapshot = self._snapshot(session)
            if snapshot is not None:
                found.append(snapshot)
        return tuple(found)

    def send(self, app_id: str, command: TransportCommand) -> bool:
        manager = self._manager()
        try:
            raw = list(manager.get_sessions())
        except (OSError, RuntimeError) as exc:
            _log.warning("could not list media sessions: %s", exc)
            return False
        for session in raw:
            if str(session.source_app_user_model_id) != app_id:
                continue
            return self._order(session, command)
        return False

    @staticmethod
    def _order(session: Any, command: TransportCommand) -> bool:
        """Call the ``TryX`` method matching ``command`` and wait for its answer."""
        calls = {
            TransportCommand.PLAY: session.try_play_async,
            TransportCommand.PAUSE: session.try_pause_async,
            TransportCommand.TOGGLE: session.try_toggle_play_pause_async,
            TransportCommand.NEXT: session.try_skip_next_async,
            TransportCommand.PREVIOUS: session.try_skip_previous_async,
            TransportCommand.STOP: session.try_stop_async,
        }
        try:
            return bool(_await(calls[command]()))
        except (OSError, RuntimeError) as exc:
            raise ActionError(
                f"{command} failed: {exc}",
                user_message="Плеер не принял команду.",
            ) from exc

    @staticmethod
    def _snapshot(session: Any) -> MediaSession | None:
        """One session read out, or ``None`` if it disappeared mid-read.

        Which happens: the app closes its session between the enumeration and the
        property read, and every getter then raises. A player that vanished is not
        an error worth an exception — it is simply not in the list any more.
        """
        try:
            app_id = str(session.source_app_user_model_id)
            playback = session.get_playback_info()
            controls = playback.controls
            timeline = session.get_timeline_properties()
            media = _await(session.try_get_media_properties_async())
        except (OSError, RuntimeError) as exc:
            _log.debug("media session went away while being read: %s", exc)
            return None
        return MediaSession(
            app_id=app_id,
            title=str(media.title or ""),
            artist=str(media.artist or media.album_artist or ""),
            album=str(media.album_title or ""),
            status=_STATUS_BY_VALUE.get(int(playback.playback_status), PlaybackStatus.CLOSED),
            position_s=_seconds(timeline.position),
            duration_s=_seconds(timeline.end_time),
            can_play=bool(controls.is_play_enabled),
            can_pause=bool(controls.is_pause_enabled),
            can_next=bool(controls.is_next_enabled),
            can_previous=bool(controls.is_previous_enabled),
        )


_sessions: SessionsBackend | None = None


def get_sessions() -> SessionsBackend:
    """The transport backend in force, built on first use."""
    global _sessions
    if _sessions is None:
        candidate: SessionsBackend = WinRtSessions() if _Winrt.load() else NullSessions()
        _sessions = candidate
    return _sessions


def set_sessions(backend: SessionsBackend | None) -> None:
    """Install a backend, or forget the cached one with ``None``. Test seam."""
    global _sessions
    _sessions = backend


def pick_session(
    sessions: Sequence[MediaSession],
    prefer_app: str = "",
) -> MediaSession | None:
    """Which of the players a command is about. Pure, so it is testable as data.

    The configured player wins even when it is paused and something else is
    playing — «следующий трек» said while a video plays in a browser is still about
    the music. Among several sessions of the same app (Яндекс Музыка opens a second
    one for a video clip) the playing one wins. With no configured player, or none
    of them present, the playing session wins, and failing that the first: Windows
    lists them most-recently-active first.
    """
    if not sessions:
        return None
    if prefer_app:
        wanted = prefer_app.casefold()
        mine = [item for item in sessions if item.app_id.casefold() == wanted]
        if mine:
            return next((item for item in mine if item.playing), mine[0])
    return next((item for item in sessions if item.playing), sessions[0])


def current_session() -> MediaSession | None:
    """The player the actions here are about, right now."""
    backend = get_sessions()
    if not backend.available:
        return None
    return pick_session(backend.sessions(), get_settings().actions.media.player_app_id)


def now_playing_ru(session: MediaSession | None) -> str:
    """What Ayris says when asked what is playing."""
    if session is None or session.empty:
        return "Сейчас ничего не играет."
    if session.status is PlaybackStatus.PLAYING:
        return f"Сейчас играет {session.label_ru}."
    if session.status is PlaybackStatus.PAUSED:
        return f"На паузе {session.label_ru}."
    return f"{session.label_ru} — {session.status.title_ru}."


def _press_media_key(command: TransportCommand) -> bool:
    """Last resort: the dedicated media key, if the settings allow it.

    Off by default, and the reason is in this package's docstring — a synthesised
    key press goes to whatever has focus, and games bind keys. It stays available
    because a player that publishes no SMTC session leaves nothing else to try, and
    a media key is at least one no game binds.
    """
    if not get_settings().actions.media.media_keys_fallback:
        return False
    # Imported here rather than at module level: the input package builds a
    # backend, and a media action that never reaches this line should not drag
    # ``SendInput`` into the process.
    from ayris.actions.input.backend import get_input_backend
    from ayris.actions.input.keys import press_combo, resolve_key

    key = resolve_key(_COMMAND_KEYS[command])
    try:
        press_combo((key,), backend=get_input_backend(), hold_ms=30)
    except ActionError as exc:
        _log.warning("media key %s was refused: %s", key.name, exc)
        return False
    _log.info("%s: медиа-клавишей %s, сессии SMTC нет", command, key.name)
    return True


def send_command(command: TransportCommand) -> MediaSession | None:
    """Order the configured player about, and say which one got it.

    Returns:
        The session the order went to, or ``None`` when it went out as a media key
        because no player publishes a session.

    Raises:
        ActionUnavailable: nothing to talk to — no session, and the key fallback is
            off or unavailable.
        ActionError: the player was there and refused.
    """
    backend = get_sessions()
    sessions = backend.sessions() if backend.available else ()
    target = pick_session(sessions, get_settings().actions.media.player_app_id)
    if target is None:
        if _press_media_key(command):
            return None
        raise ActionUnavailable(
            f"no media session to {command}",
            user_message="Не нашёл запущенный плеер. Открой Яндекс Музыку.",
        )
    if not backend.send(target.app_id, command):
        if _press_media_key(command):
            return None
        raise ActionError(
            f"{target.app_id} refused {command}",
            user_message=f"Плеер не смог {command.title_ru}.",
        )
    _log.info("%s → %s", command, target.app_id)
    return target


def _transport_result(command: TransportCommand, message_ru: str) -> ActionResult[MediaSession]:
    """Run one transport command and dress its outcome as an action result."""
    session = send_command(command)
    data = session.as_data() if session is not None else {"app_id": "", "via_keys": True}
    return ActionResult.done(
        message_ru,
        value=session,
        detail=f"{command} -> {session.app_id if session else 'media key'}",
        data=data,
    )


@register
class YMPlay(Action):
    """Resume the player, whatever it was that got paused."""

    meta: ClassVar = ActionMeta(
        name="YMPlay",
        category=ActionCategory.MEDIA,
        title_ru="Включить музыку",
        description_ru="Продолжить воспроизведение в Яндекс Музыке или другом плеере",
        timeout_ms=10_000,
    )

    def run(self, _params: Any) -> ActionResult[MediaSession]:
        return _transport_result(TransportCommand.PLAY, "Включаю.")


@register
class YMPause(Action):
    """Pause the player."""

    meta: ClassVar = ActionMeta(
        name="YMPause",
        category=ActionCategory.MEDIA,
        title_ru="Пауза",
        description_ru="Поставить воспроизведение на паузу",
        timeout_ms=10_000,
    )

    def run(self, _params: Any) -> ActionResult[MediaSession]:
        return _transport_result(TransportCommand.PAUSE, "Ставлю на паузу.")


@register
class YMToggle(Action):
    """One block for the hotkey that means «пауза» and «продолжи» alike."""

    meta: ClassVar = ActionMeta(
        name="YMToggle",
        category=ActionCategory.MEDIA,
        title_ru="Пауза или продолжить",
        description_ru="Переключить: играет — на паузу, на паузе — продолжить",
        timeout_ms=10_000,
    )

    def run(self, _params: Any) -> ActionResult[MediaSession]:
        # Read before ordering: afterwards the session reports the new state, and
        # the phrase has to describe what was just done.
        before = current_session()
        was_playing = before is not None and before.playing
        message = "Ставлю на паузу." if was_playing else "Продолжаю."
        return _transport_result(TransportCommand.TOGGLE, message)


@register
class YMNext(Action):
    """Skip to the next track."""

    meta: ClassVar = ActionMeta(
        name="YMNext",
        category=ActionCategory.MEDIA,
        title_ru="Следующий трек",
        description_ru="Переключить на следующий трек",
        timeout_ms=10_000,
    )

    def run(self, _params: Any) -> ActionResult[MediaSession]:
        return _transport_result(TransportCommand.NEXT, "Следующий трек.")


@register
class YMPrev(Action):
    """Go back to the previous track."""

    meta: ClassVar = ActionMeta(
        name="YMPrev",
        category=ActionCategory.MEDIA,
        title_ru="Предыдущий трек",
        description_ru="Вернуться к предыдущему треку",
        timeout_ms=10_000,
    )

    def run(self, _params: Any) -> ActionResult[MediaSession]:
        return _transport_result(TransportCommand.PREVIOUS, "Предыдущий трек.")


@register
class NowPlaying(Action):
    """Answer «что играет» from Windows, with no player interface involved."""

    meta: ClassVar = ActionMeta(
        name="NowPlaying",
        category=ActionCategory.MEDIA,
        title_ru="Что играет",
        description_ru="Назвать исполнителя и трек, который играет сейчас",
        timeout_ms=10_000,
    )

    def run(self, _params: Any) -> ActionResult[MediaSession]:
        backend = get_sessions()
        if not backend.available:
            raise ActionUnavailable(
                "no SMTC backend available",
                user_message="На этой системе не видно, что играет.",
            )
        session = current_session()
        data = session.as_data() if session is not None else {}
        return ActionResult.done(
            now_playing_ru(session),
            value=session,
            detail=f"session {session.app_id}" if session else "no session",
            data=data,
        )
