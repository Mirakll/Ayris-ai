"""System volume, the Windows mixer, and the microphone.

Three things a user means by «звук», and WASAPI answers each of them with a
different interface. The master volume of an endpoint is
``IAudioEndpointVolume``; one program's slider in the mixer is
``ISimpleAudioVolume`` behind ``IAudioSessionManager2``; the microphone is the
master volume again, only of a capture endpoint. All three are reached here
through :class:`AudioBackend`, one narrow protocol with six methods, and
everything above it — clamping, the configured step, «на 10 процентов тише»,
matching a said program name against the mixer — is ordinary Python that a fake
backend tests without a sound card.

**Percent in, scalar out, converted in one place.** WASAPI speaks 0.0–1.0 and
people speak 0–100, and mixing the two up is the kind of bug that sets the
volume to 100% when asked for 60. :func:`percent_to_scalar` and
:func:`scalar_to_percent` are the only two functions in Ayris allowed to know
that, they clamp on the way through, and the tests assert that what reaches the
backend is ``0.6`` rather than ``60``.

**A relative change is not computed here either.** «Громче» is
:class:`~ayris.nlu.slot_types.RelativeValue`, and
:meth:`~ayris.nlu.slot_types.RelativeValue.resolve` already knows that a percent
is a percentage of the scale rather than of the current value and that the result
is clamped. The step it applies to a bare «громче» comes from
``[actions.audio] volume_step``.

The microphone deserves one warning of its own. A muted capture endpoint is not
the assistant's «микрофон выключен»: the first is a Windows-wide setting that
silences the device for every program, the second is Ayris choosing not to
listen. :class:`SetMicVolume` therefore publishes nothing on the bus and never
touches the assistant's own state — a user who says «выключи микрофон» meaning
«перестань меня слушать» is served by the microphone toggle, not by this action.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from pydantic import Field, model_validator

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.app_index import (
    AppAmbiguous,
    AppNotFound,
    get_app_index,
    phrase_key,
)
from ayris.actions.system.apps import executable_stem
from ayris.actions.system.audio_devices import (
    DeviceKind,
    audio_utilities,
    endpoint_volume,
    find_device,
    initialize_com,
    list_audio_devices,
    start_device_watcher,
)
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.nlu.numbers import plural_form
from ayris.nlu.slot_types import (
    MAX_VOLUME,
    MIN_VOLUME,
    Direction,
    RelativeUnit,
    RelativeValue,
)
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "AdjustVolume",
    "AudioBackend",
    "AudioSession",
    "MixerUnavailable",
    "MuteMode",
    "MuteToggle",
    "SessionNotFound",
    "SetAppVolume",
    "SetMicVolume",
    "SetVolume",
    "VolumeState",
    "WasapiAudio",
    "clamp_volume",
    "find_sessions",
    "get_audio_backend",
    "match_sessions",
    "percent_to_scalar",
    "scalar_to_percent",
    "set_audio_backend",
    "volume_step",
]

_log = get_logger(__name__)

#: Longest mixer listing that is worth building. A busy machine has a dozen
#: sessions; anything past this is a runaway and not a browser.
MAX_SESSIONS: Final = 64


class MixerUnavailable(ActionUnavailable):
    """The volume mixer cannot be read on this machine.

    What a machine with no sound card answers — the CI runner included, where
    this is the expected outcome rather than a failure.
    """

    default_user_message = "Не смогла прочитать микшер Windows."


class SessionNotFound(ActionError):
    """No mixer session belongs to the program that was named."""

    default_user_message = "Не нашла это приложение в микшере."


# --------------------------------------------------------------------------- #
# Percent ↔ scalar, in one place and nowhere else
# --------------------------------------------------------------------------- #


def clamp_volume(level: int) -> int:
    """``level`` brought inside 0..100.

    Clamping rather than refusing, because every source of a level has already
    decided that out-of-range means «as far as it goes»: «громче» at 95 asks for
    100, and :meth:`RelativeValue.resolve` clamps for the same reason.
    """
    return max(MIN_VOLUME, min(MAX_VOLUME, level))


def percent_to_scalar(level: int) -> float:
    """Percent 0..100 as the 0.0–1.0 scalar WASAPI wants."""
    return clamp_volume(level) / 100.0


def scalar_to_percent(scalar: float) -> int:
    """A WASAPI scalar as whole percent, rounded and clamped.

    Drivers do not store the scalar exactly: setting ``0.6`` reads back as
    ``0.6000000238418579``, and truncating that would report 59 after asking for
    60. Rounding is what makes the level read back the way it was set.
    """
    return clamp_volume(round(scalar * 100.0))


def volume_step() -> int:
    """How much one «громче» moves the volume, from ``[actions.audio]``.

    Read on every call and not cached: settings are hot-reloadable, and a step
    remembered at import time would ignore the slider the user just moved.
    """
    from ayris.core.config import get_settings

    return get_settings().actions.audio.volume_step


# --------------------------------------------------------------------------- #
# What the actions return
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VolumeState:
    """Loudness of one endpoint, in the units a person hears them in."""

    level: int = 0
    muted: bool = False
    kind: DeviceKind = DeviceKind.OUTPUT
    device: str = ""

    @property
    def spoken_ru(self) -> str:
        """The state as one short sentence: «громкость 60», «без звука»."""
        if self.muted:
            return "без звука"
        return f"громкость {self.level}"

    def with_level(self, level: int) -> VolumeState:
        """Copy at another level, clamped. What a setter reports back."""
        return VolumeState(
            level=clamp_volume(level),
            muted=self.muted,
            kind=self.kind,
            device=self.device,
        )

    def with_mute(self, muted: bool) -> VolumeState:
        """Copy with the mute flag flipped to ``muted``."""
        return VolumeState(level=self.level, muted=muted, kind=self.kind, device=self.device)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, for :attr:`ActionResult.data` and the audit trail."""
        return {
            "level": self.level,
            "muted": self.muted,
            "kind": str(self.kind),
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class AudioSession:
    """One program's slider in the volume mixer.

    ``process`` is the executable stem, folded and without ``.exe`` — the shape
    :func:`~ayris.actions.system.apps.executable_stem` produces, so that the
    resolver behind ``{app}`` and the mixer agree on what «хром» is called.

    ``pid`` of zero is not a broken session: Windows keeps the system sounds
    under it, and its display name is a resource reference such as
    ``@%SystemRoot%\\System32\\AudioSrv.Dll,-202`` rather than anything sayable.
    """

    pid: int = 0
    process: str = ""
    display_name: str = ""
    level: int = 0
    muted: bool = False
    active: bool = True

    @property
    def spoken_name(self) -> str:
        """Display name when there is a real one, ``""`` when it is a resource id."""
        name = self.display_name.strip()
        return "" if name.startswith("@") else name

    @property
    def label(self) -> str:
        """How this session is named out loud."""
        return self.spoken_name or self.process or ("системные звуки" if not self.pid else "")

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, for :attr:`ActionResult.data` and the audit trail."""
        return {
            "pid": self.pid,
            "process": self.process,
            "display_name": self.display_name,
            "level": self.level,
            "muted": self.muted,
            "active": self.active,
        }


# --------------------------------------------------------------------------- #
# Which session did they mean
# --------------------------------------------------------------------------- #


def _session_keys(session: AudioSession) -> tuple[str, str]:
    """Folded process stem and display name of one session.

    Folded through :func:`~ayris.actions.system.app_index.phrase_key`, the same
    normaliser the ``{app}`` resolver uses, so that «хром» reaches a session the
    same way it reaches an installed program. A resource-string display name
    folds to ``""``: matching «аудио» against ``@%SystemRoot%\\…AudioSrv.Dll,-202``
    would hit the system sounds for half the words a user says.
    """
    process = phrase_key(session.process)
    name = session.spoken_name
    return process, phrase_key(name) if name else ""


def _session_score(session: AudioSession, needle: str, words: Sequence[str]) -> int:
    """How well one session answers a said name: 5 the stem … 1 all words.

    The process stem outranks the display name because it is what a program is
    actually called: Chrome's sessions are named «Google Chrome» while the stem
    is ``chrome``, and a user saying «хром» resolves to the stem through the app
    index, never to the marketing name.
    """
    process, display = _session_keys(session)
    if process and process == needle:
        return 5
    if display and display == needle:
        return 4
    if display.startswith(needle) or (process and process.startswith(needle)):
        return 3
    if needle in display or (process and needle in process):
        return 2
    if len(words) > 1 and display and all(word in display for word in words):
        return 1
    return 0


def match_sessions(sessions: Iterable[AudioSession], query: str) -> list[AudioSession]:
    """Mixer sessions that answer to a said program name, best match first.

    Active sessions come before idle ones at the same score: a browser that is
    playing something is what «сделай хром тише» is about, even though every one
    of its background processes owns a session too.
    """
    needle = phrase_key(query)
    if not needle:
        return []
    words = needle.split()
    scored: list[tuple[int, int, int, AudioSession]] = []
    for session in sessions:
        score = _session_score(session, needle, words)
        if score:
            scored.append((-score, 0 if session.active else 1, session.pid, session))
    scored.sort(key=lambda item: item[:3])
    return [session for *_, session in scored]


def _same_program(sessions: Iterable[AudioSession], chosen: AudioSession) -> list[AudioSession]:
    """Every session of the program ``chosen`` belongs to.

    Chrome answers with one session per renderer process, and a user who says
    «хром потише» means the tab that is playing *and* the one that will play
    next. Sessions are grouped by process stem, falling back to the pid when the
    stem is unknown — grouping those by an empty stem would collect every
    unnameable session on the machine.
    """
    if not chosen.process:
        return [session for session in sessions if session.pid == chosen.pid]
    return [session for session in sessions if session.process == chosen.process]


def find_sessions(sessions: Iterable[AudioSession], query: str) -> list[AudioSession]:
    """Every mixer session of the program ``query`` names.

    Two passes, in this order for a reason. First the said name is matched
    against the sessions themselves, which costs nothing. Only when that finds
    nothing is the application index asked — and that call may block on a
    registry scan, which is far too much work for «сделай хром тише» when the
    mixer already knows a session called ``chrome``.

    Raises:
        SessionNotFound: no session belongs to that program, whether because it
            is not playing anything or because nothing of that name is installed.
    """
    listed = list(sessions)
    matched = match_sessions(listed, query)
    if matched:
        return _same_program(listed, matched[0])
    stem = _resolved_stem(query)
    if stem:
        by_stem = [session for session in listed if session.process == stem]
        if by_stem:
            return by_stem
    raise SessionNotFound(
        f"no mixer session matches {query!r} (resolved stem {stem or '-'})",
        user_message=f"Не нашла приложение «{query.strip()}» в микшере.",
    )


def _resolved_stem(query: str) -> str:
    """Executable stem of the installed program ``query`` names, or ``""``.

    The fallback path: the mixer says ``msedge`` and the user said «эдж», and
    only the application index knows those are the same thing. An unresolvable
    name is not an error here — the caller has a better message for it than
    «не нашла программу», because the program may well be installed and simply
    not playing anything.
    """
    try:
        candidate = get_app_index().resolve(query)
    except (AppNotFound, AppAmbiguous) as exc:
        _log.debug("«%s» не разрешилось в программу: %s", query, exc)
        return ""
    except ActionError as exc:  # pragma: no cover - index unavailable, not fatal here
        _log.debug("индекс программ недоступен: %s", exc)
        return ""
    return executable_stem(candidate)


# --------------------------------------------------------------------------- #
# The backend: everything that touches WASAPI, and nothing that decides anything
# --------------------------------------------------------------------------- #


class AudioBackend(Protocol):
    """Everything the volume actions need from WASAPI.

    Six methods, all of them in scalars: ``0.0`` to ``1.0`` going down, percent
    only above this line. Faked wholesale in the tests, which is how clamping and
    the configured step are checked without a sound card — and how the fake can
    assert that ``0.6`` arrived rather than ``60``.
    """

    def get_master_volume(
        self,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> VolumeState:
        """Level and mute of one endpoint, or of the default one."""
        ...

    def set_master_volume(
        self,
        scalar: float,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> None:
        """Set the endpoint's master volume to a 0.0–1.0 scalar."""
        ...

    def set_master_mute(
        self,
        muted: bool,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> None:
        """Mute or unmute the endpoint."""
        ...

    def list_sessions(self) -> list[AudioSession]:
        """Every program in the mixer right now."""
        ...

    def set_session_volume(self, pid: int, scalar: float) -> None:
        """Set one process's mixer slider to a 0.0–1.0 scalar."""
        ...

    def set_session_mute(self, pid: int, muted: bool) -> None:
        """Mute or unmute one process in the mixer."""
        ...


class WasapiAudio:
    """The real backend, over ``pycaw``.

    Endpoints are resolved on every call through
    :func:`~ayris.actions.system.audio_devices.endpoint_volume` rather than
    cached: between «громкость 60» and «громче» the user may have unplugged the
    headphones, and a remembered ``IAudioEndpointVolume`` then fails in a way that
    reads like a bug in the action.
    """

    def supported(self) -> bool:
        """Whether WASAPI can be reached at all in this process."""
        if sys.platform != "win32":
            return False
        try:
            import comtypes  # noqa: F401
            import pycaw.utils  # noqa: F401
        except ImportError:
            _log.debug("pycaw недоступен, громкость не читается")
            return False
        return True

    def get_master_volume(
        self,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> VolumeState:
        """Read one endpoint.

        Raises:
            DeviceUnavailable: there is no such endpoint, or it stopped answering.
        """
        volume, name = endpoint_volume(kind, device_id)
        try:
            scalar = float(volume.GetMasterVolumeLevelScalar())
            muted = bool(volume.GetMute())
        except OSError as exc:
            raise _endpoint_failed(kind, exc) from exc
        return VolumeState(
            level=scalar_to_percent(scalar),
            muted=muted,
            kind=kind,
            device=name,
        )

    def set_master_volume(
        self,
        scalar: float,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> None:
        """Move the endpoint's master slider.

        The ``None`` second argument is the event-context GUID: WASAPI uses it to
        tell an application its own change from someone else's, and Ayris has no
        listener that cares.

        Raises:
            DeviceUnavailable: the endpoint refused or is gone.
        """
        volume, _ = endpoint_volume(kind, device_id)
        try:
            volume.SetMasterVolumeLevelScalar(_clamp_scalar(scalar), None)
        except OSError as exc:
            raise _endpoint_failed(kind, exc) from exc

    def set_master_mute(
        self,
        muted: bool,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> None:
        """Mute or unmute the endpoint.

        Raises:
            DeviceUnavailable: the endpoint refused or is gone.
        """
        volume, _ = endpoint_volume(kind, device_id)
        try:
            volume.SetMute(bool(muted), None)
        except OSError as exc:
            raise _endpoint_failed(kind, exc) from exc

    def list_sessions(self) -> list[AudioSession]:
        """Describe every mixer session, skipping the ones that will not answer.

        One broken session must not hide the rest of the mixer, so a session that
        raises while being read is dropped with a line in the debug log.

        Raises:
            MixerUnavailable: the mixer itself cannot be reached — which is what a
                machine with no sound card says.
        """
        sessions: list[AudioSession] = []
        for raw in self._raw_sessions():
            described = self._describe(raw)
            if described is not None:
                sessions.append(described)
            if len(sessions) >= MAX_SESSIONS:
                break
        return sessions

    def set_session_volume(self, pid: int, scalar: float) -> None:
        """Move one process's mixer slider, on every session it owns."""
        self._each_session(pid, "SetMasterVolume", _clamp_scalar(scalar))

    def set_session_mute(self, pid: int, muted: bool) -> None:
        """Mute or unmute one process in the mixer."""
        self._each_session(pid, "SetMute", bool(muted))

    def _raw_sessions(self) -> list[Any]:
        """``pycaw`` session objects, or a spoken refusal.

        Raises:
            MixerUnavailable: the session manager is not there.
        """
        initialize_com()
        try:
            raw = audio_utilities().GetAllSessions()
        except OSError as exc:
            raise MixerUnavailable(f"IAudioSessionManager2 is unavailable: {exc}") from exc
        return list(raw or ())

    def _describe(self, raw: Any) -> AudioSession | None:
        """One ``pycaw`` session as an :class:`AudioSession`, or ``None``."""
        try:
            pid = int(getattr(raw, "ProcessId", 0) or 0)
            display = str(getattr(raw, "DisplayName", "") or "")
            state = int(getattr(raw, "State", 0) or 0)
            simple = raw.SimpleAudioVolume
            level = scalar_to_percent(float(simple.GetMasterVolume()))
            muted = bool(simple.GetMute())
        except (OSError, AttributeError, TypeError, ValueError) as exc:
            _log.debug("сессия микшера не читается: %s", exc)
            return None
        return AudioSession(
            pid=pid,
            process=_process_stem(raw, pid),
            display_name=display,
            level=level,
            muted=muted,
            active=state == _SESSION_ACTIVE,
        )

    def _each_session(self, pid: int, method: str, value: Any) -> None:
        """Call one ``ISimpleAudioVolume`` setter on every session of ``pid``.

        A process that exited between the listing and the call is not an error:
        the user asked for silence and silence is what an exited process gives.
        """
        touched = 0
        for raw in self._raw_sessions():
            try:
                if int(getattr(raw, "ProcessId", 0) or 0) != pid:
                    continue
                getattr(raw.SimpleAudioVolume, method)(value, None)
            except (OSError, AttributeError, TypeError, ValueError) as exc:
                _log.debug("сессия pid %s не приняла %s: %s", pid, method, exc)
                continue
            touched += 1
        if not touched:
            _log.debug("у процесса %s больше нет сессий в микшере", pid)


#: ``AudioSessionState.Active``. A session at ``Inactive`` is a program that is
#: running with nothing playing — it belongs in the mixer, just not at the top.
_SESSION_ACTIVE: Final = 1


def _clamp_scalar(scalar: float) -> float:
    """A scalar brought inside 0.0..1.0.

    The last gate before WASAPI: it rejects anything outside the range with a
    ``COMError`` that says nothing about which value was wrong.
    """
    return max(0.0, min(1.0, float(scalar)))


def _endpoint_failed(kind: DeviceKind, exc: OSError) -> ActionError:
    """The error to raise when an endpoint stops answering mid-call."""
    from ayris.actions.system.audio_devices import DeviceUnavailable

    return DeviceUnavailable(
        f"{kind} endpoint volume call failed: {exc}",
        user_message=f"Не смогла управлять громкостью: {kind.noun_ru} не отвечает.",
    )


def _process_stem(raw: Any, pid: int) -> str:
    """Executable stem of a session's process, folded, without ``.exe``.

    ``QueryFullProcessImageNameW`` first, because it works across integrity
    levels and needs no third-party package. ``session.Process`` is ``pycaw``'s
    own answer and comes from ``psutil``, which is not pinned in
    ``pyproject.toml`` — it arrives as a transitive dependency and may not — so it
    is read defensively and only as a fallback.
    """
    name = winapi.process_image_name(pid) if pid > 0 else ""
    if not name:
        name = _psutil_name(raw)
    if not name:
        return ""
    stem = name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if stem.casefold().endswith(".exe"):
        stem = stem[: -len(".exe")]
    return stem.casefold()


def _psutil_name(raw: Any) -> str:
    """``session.Process.name()`` if ``pycaw`` managed to attach one."""
    process = getattr(raw, "Process", None)
    getter = getattr(process, "name", None)
    if not callable(getter):
        return ""
    try:
        return str(getter() or "")
    except Exception as exc:
        _log.debug("psutil не назвал процесс: %s", exc)
        return ""


_backend: AudioBackend | None = None
_real_backend: WasapiAudio | None = None
_lock = threading.Lock()


def get_audio_backend() -> AudioBackend:
    """The backend in force. Real WASAPI unless a test replaced it.

    Creating the real one also registers the hot-plug watcher, so that a session
    which only ever changes the volume still drops its cached COM objects when the
    sound stack changes underneath it.

    Raises:
        ActionUnavailable: this is not Windows, or ``pycaw`` is missing.
    """
    if _backend is not None:
        return _backend
    global _real_backend
    with _lock:
        if _real_backend is None:
            real = WasapiAudio()
            if not real.supported():
                raise ActionUnavailable(
                    "volume actions require Windows with pycaw",
                    user_message="Управление громкостью работает только в Windows.",
                )
            _real_backend = real
            start_device_watcher()
        return _real_backend


def set_audio_backend(backend: AudioBackend | None) -> None:
    """Install a backend, or restore the real one with ``None``. Test seam."""
    global _backend
    _backend = backend


# --------------------------------------------------------------------------- #
# Master volume
# --------------------------------------------------------------------------- #


def _target_device(kind: DeviceKind, device: str) -> str:
    """Endpoint id for a spoken device name, or ``""`` for the default one.

    Raises:
        DeviceNotFound: nothing of this kind answers to that name.
    """
    if not device.strip():
        return ""
    return find_device(list_audio_devices(kind), device, kind=kind).device_id


def _read_state(
    backend: AudioBackend,
    kind: DeviceKind,
    device_id: str,
    fallback: VolumeState,
) -> VolumeState:
    """Read the endpoint back, keeping ``fallback`` if the read fails.

    A setter that worked and a read-back that did not is still a success, and
    answering «сделала 60» from what we asked for beats answering with an error
    about something that already happened.
    """
    try:
        return backend.get_master_volume(kind, device_id)
    except (ActionError, OSError) as exc:
        _log.debug("не удалось перечитать громкость: %s", exc)
        return fallback


def _apply_level(
    backend: AudioBackend,
    level: int,
    kind: DeviceKind,
    device_id: str,
    current: VolumeState,
) -> VolumeState:
    """Set one endpoint to ``level`` percent and describe where it ended up.

    Setting a level also lifts mute, which is what «сделай громкость 40» means to
    everyone who has ever said it to a muted machine.
    """
    backend.set_master_volume(percent_to_scalar(level), kind, device_id)
    if current.muted:
        backend.set_master_mute(False, kind, device_id)
    return _read_state(backend, kind, device_id, current.with_level(level).with_mute(False))


def _undo_volume(kind: DeviceKind, token: str) -> ActionResult[VolumeState]:
    """Put one endpoint back to the level a ``token`` remembers.

    Shared by :class:`SetVolume` and :class:`AdjustVolume`, because «отмени» after
    either of them means the same thing and the token is the same string.
    """
    level, muted, device_id = _parse_token(token)
    backend = get_audio_backend()
    backend.set_master_volume(percent_to_scalar(level), kind, device_id)
    backend.set_master_mute(muted, kind, device_id)
    state = _read_state(
        backend,
        kind,
        device_id,
        VolumeState(level=level, muted=muted, kind=kind),
    )
    return ActionResult.done(f"Вернула {state.level}%.", value=state, data=state.as_dict())


def _make_token(state: VolumeState, device_id: str) -> str:
    """Remember a level well enough to restore it: ``level|muted|device_id``."""
    return f"{state.level}|{int(state.muted)}|{device_id}"


def _volume_said(state: VolumeState) -> str:
    """Confirmation of a level change, as a whole sentence."""
    if state.muted:
        return "Выключила звук."
    return f"Громкость {state.level}%."


def _parse_token(token: str) -> tuple[int, bool, str]:
    """Read back a token made by :func:`_make_token`.

    Raises:
        ActionError: the token is not one of ours — a stale history entry, or a
            hand-edited macro.
    """
    parts = token.split("|", 2)
    try:
        level = clamp_volume(int(parts[0]))
        muted = bool(int(parts[1])) if len(parts) > 1 else False
    except (ValueError, IndexError) as exc:
        raise ActionError(
            f"malformed volume undo token {token!r}",
            user_message="Не помню, какая громкость была до этого.",
        ) from exc
    return level, muted, parts[2] if len(parts) > 2 else ""


class _EndpointParams(ActionParams):
    """The device field that every master-volume action shares."""

    device: str = Field(
        default="",
        max_length=160,
        title="Устройство",
        description="Часть названия устройства; пусто — то, что используется сейчас",
    )


@register
class SetVolume(Action):
    """Set the system volume to an absolute percentage."""

    meta: ClassVar = ActionMeta(
        name="SetVolume",
        category=ActionCategory.AUDIO,
        title_ru="Установить громкость",
        description_ru="Задать громкость системы в процентах",
        supports_undo=True,
        timeout_ms=5_000,
    )

    class Params(_EndpointParams):
        level: int = Field(
            ge=MIN_VOLUME,
            le=MAX_VOLUME,
            title="Громкость",
            description="Ноль — тишина, сто — максимум",
            json_schema_extra={"unit_ru": "%"},
        )

    def run(self, params: Params) -> ActionResult[VolumeState]:
        backend = get_audio_backend()
        device_id = _target_device(DeviceKind.OUTPUT, params.device)
        before = backend.get_master_volume(DeviceKind.OUTPUT, device_id)
        level = clamp_volume(params.level)
        if level == before.level and not before.muted:
            return ActionResult.done(
                f"Громкость уже {level}%.",
                value=before,
                data=before.as_dict(),
            )
        state = _apply_level(backend, level, DeviceKind.OUTPUT, device_id, before)
        return ActionResult.done(
            _volume_said(state),
            value=state,
            undo_token=_make_token(before, device_id),
            data=state.as_dict(),
        )

    def undo(self, token: str) -> ActionResult[VolumeState]:
        """Restore the level and the mute this call replaced."""
        return _undo_volume(DeviceKind.OUTPUT, token)


@register
class AdjustVolume(Action):
    """Make it louder or quieter, by the configured step or by a said amount."""

    meta: ClassVar = ActionMeta(
        name="AdjustVolume",
        category=ActionCategory.AUDIO,
        title_ru="Изменить громкость",
        description_ru="Сделать громче или тише на шаг либо на заданное число процентов",
        supports_undo=True,
        timeout_ms=5_000,
    )

    class Params(_EndpointParams):
        direction: Direction = Field(
            title="Куда",
            description="Громче или тише",
            json_schema_extra={
                "choices_ru": {
                    str(Direction.UP): "Громче",
                    str(Direction.DOWN): "Тише",
                }
            },
        )
        amount: int | None = Field(
            default=None,
            ge=1,
            le=MAX_VOLUME,
            title="На сколько",
            description="Процентов; пусто — шаг из настроек",
            json_schema_extra={"unit_ru": "%"},
        )

    def run(self, params: Params) -> ActionResult[VolumeState]:
        backend = get_audio_backend()
        device_id = _target_device(DeviceKind.OUTPUT, params.device)
        before = backend.get_master_volume(DeviceKind.OUTPUT, device_id)
        change = RelativeValue(
            direction=params.direction,
            amount=None if params.amount is None else Decimal(params.amount),
            unit=RelativeUnit.STEP if params.amount is None else RelativeUnit.PERCENT,
        )
        level = change.resolve(before.level, step=volume_step())
        if level == before.level and not before.muted:
            return ActionResult.done(
                self._at_the_end(params.direction, level),
                value=before,
                data=before.as_dict(),
            )
        state = _apply_level(backend, level, DeviceKind.OUTPUT, device_id, before)
        return ActionResult.done(
            _volume_said(state),
            value=state,
            undo_token=_make_token(before, device_id),
            data=state.as_dict(),
        )

    def undo(self, token: str) -> ActionResult[VolumeState]:
        """Restore the level and the mute this call replaced."""
        return _undo_volume(DeviceKind.OUTPUT, token)

    def _at_the_end(self, direction: Direction, level: int) -> str:
        """What to say when the slider is already against its stop."""
        if direction is Direction.UP:
            return "Громкость уже на максимуме."
        return "Тише уже некуда." if level == MIN_VOLUME else f"Громкость уже {level}%."


class MuteMode(StrEnum):
    """What «выключи звук» is being asked to do to the mute switch."""

    ON = "on"
    OFF = "off"
    TOGGLE = "toggle"

    @property
    def title_ru(self) -> str:
        """Label for the macro editor."""
        return {
            MuteMode.ON: "Выключить звук",
            MuteMode.OFF: "Включить звук",
            MuteMode.TOGGLE: "Переключить",
        }[self]

    def applies(self, muted: bool) -> bool:
        """The mute state this mode wants, given the current one."""
        if self is MuteMode.TOGGLE:
            return not muted
        return self is MuteMode.ON


@register
class MuteToggle(Action):
    """Mute or unmute an endpoint, or flip whatever it is now."""

    meta: ClassVar = ActionMeta(
        name="MuteToggle",
        category=ActionCategory.AUDIO,
        title_ru="Выключить или включить звук",
        description_ru="Заглушить устройство, снять заглушение или переключить",
        timeout_ms=5_000,
    )

    class Params(_EndpointParams):
        mode: MuteMode = Field(
            default=MuteMode.TOGGLE,
            title="Что сделать",
            description="Выключить, включить или переключить",
            json_schema_extra={"choices_ru": {str(mode): mode.title_ru for mode in MuteMode}},
        )
        kind: DeviceKind = Field(
            default=DeviceKind.OUTPUT,
            title="Направление",
            description="Звук или микрофон",
            json_schema_extra={"choices_ru": {str(k): k.title_ru for k in DeviceKind}},
        )

    def run(self, params: Params) -> ActionResult[VolumeState]:
        backend = get_audio_backend()
        device_id = _target_device(params.kind, params.device)
        before = backend.get_master_volume(params.kind, device_id)
        muted = params.mode.applies(before.muted)
        if muted == before.muted:
            return ActionResult.done(
                self._already(params.kind, muted=muted),
                value=before,
                data=before.as_dict(),
            )
        backend.set_master_mute(muted, params.kind, device_id)
        state = _read_state(backend, params.kind, device_id, before.with_mute(muted))
        return ActionResult.done(
            self._said(params.kind, state),
            value=state,
            data=state.as_dict(),
        )

    def _already(self, kind: DeviceKind, *, muted: bool) -> str:
        """Nothing to do, said in a way that still confirms the state."""
        if kind is DeviceKind.INPUT:
            return "Микрофон и так выключен." if muted else "Микрофон и так включён."
        return "Звук и так выключен." if muted else "Звук и так включён."

    def _said(self, kind: DeviceKind, state: VolumeState) -> str:
        """Confirmation of a change that happened."""
        if kind is DeviceKind.INPUT:
            return "Выключила микрофон." if state.muted else "Включила микрофон."
        return "Выключила звук." if state.muted else f"Включила звук, громкость {state.level}%."


# --------------------------------------------------------------------------- #
# One program in the mixer, and the microphone
# --------------------------------------------------------------------------- #


class _LevelAndMuteParams(ActionParams):
    """Level, mute, and the rule that at least one of them was asked for."""

    level: int | None = Field(
        default=None,
        ge=MIN_VOLUME,
        le=MAX_VOLUME,
        title="Громкость",
        description="Ноль — тишина, сто — максимум; пусто — не менять",
        json_schema_extra={"unit_ru": "%"},
    )
    mute: bool | None = Field(
        default=None,
        title="Без звука",
        description="Да — заглушить, нет — снять заглушение; пусто — не менять",
    )

    @model_validator(mode="after")
    def _something_to_do(self) -> _LevelAndMuteParams:
        """Refuse a call that would change nothing.

        A macro with both fields empty is a mistake in the macro, and doing
        nothing quietly is how it stays a mistake.
        """
        if self.level is None and self.mute is None:
            raise ValueError("укажите громкость, заглушение или и то и другое")
        return self


def _mixer_said(label: str, level: int | None, mute: bool | None) -> str:
    """What one changed program is reported as.

    Both halves in one sentence when both were asked for, because «сделай хром
    тише и включи ему звук» is one request and deserves one answer.
    """
    name = f"«{label}»" if label else "приложения"
    if mute is True:
        return f"Заглушила {name}."
    if level is None:
        return f"Включила звук у {name}."
    if mute is False:
        return f"Включила звук у {name}, громкость {level}%."
    return f"Громкость {name} — {level}%."


@register
class SetAppVolume(Action):
    """Change one program's slider in the volume mixer, by name."""

    meta: ClassVar = ActionMeta(
        name="SetAppVolume",
        category=ActionCategory.AUDIO,
        title_ru="Громкость приложения",
        description_ru="Задать громкость или заглушить отдельную программу в микшере",
        timeout_ms=8_000,
    )

    class Params(_LevelAndMuteParams):
        app: str = Field(
            min_length=1,
            max_length=160,
            title="Приложение",
            description="Как пользователь называет программу: «хром», «спотифай», «discord»",
        )

    def run(self, params: Params) -> ActionResult[list[AudioSession]]:
        backend = get_audio_backend()
        sessions = find_sessions(backend.list_sessions(), params.app)
        changed: list[AudioSession] = []
        for session in sessions:
            if params.level is not None:
                backend.set_session_volume(session.pid, percent_to_scalar(params.level))
            if params.mute is not None:
                backend.set_session_mute(session.pid, params.mute)
            changed.append(self._after(session, params))
        return ActionResult.done(
            _mixer_said(changed[0].label, params.level, params.mute),
            value=changed,
            detail=self._detail(changed),
            data={"sessions": [session.as_dict() for session in changed]},
        )

    def _after(self, session: AudioSession, params: Params) -> AudioSession:
        """The session as it now is, without reading the mixer a second time."""
        return AudioSession(
            pid=session.pid,
            process=session.process,
            display_name=session.display_name,
            level=session.level if params.level is None else clamp_volume(params.level),
            muted=session.muted if params.mute is None else params.mute,
            active=session.active,
        )

    def _detail(self, changed: Sequence[AudioSession]) -> str:
        """A line for the log when a program held more than one session."""
        if len(changed) < 2:
            return ""
        count = len(changed)
        noun = plural_form(count, "сессию", "сессии", "сессий")
        return f"задето {count} {noun}: {', '.join(str(item.pid) for item in changed)}"


@register
class SetMicVolume(Action):
    """Set the microphone's input level, or mute the device itself.

    Muting here is the Windows-wide switch on a capture endpoint: it silences the
    microphone for every program on the machine, Ayris included. It is *not* the
    assistant's own «перестань слушать» — that is a different switch entirely, it
    lives on the bus, and this action deliberately leaves it alone so that
    «выключи микрофон» never turns out to have half-muted two different things.
    """

    meta: ClassVar = ActionMeta(
        name="SetMicVolume",
        category=ActionCategory.AUDIO,
        title_ru="Громкость микрофона",
        description_ru="Задать чувствительность микрофона или заглушить его в Windows",
        timeout_ms=5_000,
    )

    class Params(_LevelAndMuteParams):
        device: str = Field(
            default="",
            max_length=160,
            title="Микрофон",
            description="Часть названия устройства; пусто — микрофон по умолчанию",
        )

    def run(self, params: Params) -> ActionResult[VolumeState]:
        backend = get_audio_backend()
        kind = DeviceKind.INPUT
        device_id = _target_device(kind, params.device)
        before = backend.get_master_volume(kind, device_id)

        if params.level is not None:
            backend.set_master_volume(percent_to_scalar(params.level), kind, device_id)
        if params.mute is not None:
            backend.set_master_mute(params.mute, kind, device_id)

        expected = before
        if params.level is not None:
            expected = expected.with_level(params.level)
        if params.mute is not None:
            expected = expected.with_mute(params.mute)
        state = _read_state(backend, kind, device_id, expected)
        return ActionResult.done(
            self._said(state, params),
            value=state,
            data=state.as_dict(),
        )

    def _said(self, state: VolumeState, params: Params) -> str:
        """Confirmation that names the device when it was not the default one."""
        where = f" на «{state.device}»" if params.device.strip() and state.device else ""
        if params.mute is True:
            return f"Выключила микрофон{where}."
        if params.level is None:
            return f"Включила микрофон{where}, чувствительность {state.level}%."
        return f"Чувствительность микрофона{where} — {state.level}%."
