"""Global hotkeys: registrations, command triggers, capture and push-to-talk.

The manager owns policy and delegates OS calls to a small backend.  Its WinAPI
backend has a dedicated message-loop thread, so registration and removal always
happen on the same thread as required by RegisterHotKey.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from ayris.actions.input.keys import KEYS, MODIFIERS
from ayris.core.errors import HotkeyError
from ayris.core.events import (
    CancelRequested,
    CommandsChanged,
    ConfigChanged,
    EventBus,
    HotkeyTriggered,
    MicToggleRequested,
    NotificationRequested,
    OverlayToggleRequested,
    PttPressed,
    PttReleased,
    WakeToggleRequested,
)
from ayris.core.models import TriggerType
from ayris.core.profile import ProfileSwitched
from ayris.utils.hotkey_backends.interception import InterceptionBackend
from ayris.utils.hotkey_backends.winapi import HotkeyBackendUnavailable, WinApiBackend
from ayris.utils.hotkeys import Hotkey, parse_hotkey
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.macros.engine import MacroEngine
    from ayris.core.app import AyrisApp
    from ayris.core.config import HotkeysConfig
    from ayris.core.repositories import Repositories

__all__ = [
    "CaptureCancelled",
    "HotkeyBinding",
    "HotkeyConflict",
    "HotkeyManager",
    "detect_conflicts",
    "install_hotkeys",
]

_log = get_logger(__name__)
_START_TIMEOUT_S: Final = 3.0
_PTT_POLL_S: Final = 0.01
_PTT_MINIMUM_MS: Final = 120
_CAPTURE_FORBIDDEN: Final[frozenset[str]] = frozenset(
    {"escape", "printscreen", "pause", "capslock", "numlock", "scrolllock"}
)
_SYSTEM_LABELS: Final[dict[str, str]] = {
    "push_to_talk": "Push-to-Talk",
    "toggle_wake": "переключение Wake Word",
    "toggle_overlay": "оверлей",
    "toggle_mute": "Mute микрофона",
    "cancel": "отмена действия",
}


class _Backend(Protocol):
    def run(self, callback: Callable[[int], None]) -> None: ...

    def wait_ready(self, timeout: float) -> bool: ...

    def invoke(self, command: Callable[[], None]) -> bool: ...

    def register(self, identifier: int, hotkey: Hotkey, owner: str) -> None: ...

    def unregister(self, identifier: int) -> None: ...

    def key_down(self, hotkey: Hotkey) -> bool: ...

    def start_capture(self, callback: Callable[[int, bool], bool]) -> None: ...

    def stop_capture(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class HotkeyBinding:
    """One claimed combination and what firing it means."""

    hotkey: Hotkey
    owner: str
    action: str
    command_id: int | None = None


@dataclass(frozen=True, slots=True)
class HotkeyConflict:
    """Two Ayris owners requested the same canonical combination."""

    hotkey: Hotkey
    owners: tuple[str, ...]

    @property
    def user_message(self) -> str:
        return f"{self.hotkey.label_ru} уже используется: {', '.join(self.owners)}."


class CaptureCancelled(HotkeyError):  # noqa: N818 - describes a UI outcome
    """Esc cancelled the temporary capture dialog."""

    default_user_message = "Захват сочетания отменён."


def detect_conflicts(bindings: Iterable[HotkeyBinding]) -> tuple[HotkeyConflict, ...]:
    """Return every duplicate without touching the operating system."""
    owners: dict[Hotkey, list[str]] = {}
    for binding in bindings:
        owners.setdefault(binding.hotkey, []).append(binding.owner)
    return tuple(
        HotkeyConflict(hotkey, tuple(names)) for hotkey, names in owners.items() if len(names) > 1
    )


class HotkeyManager:
    """Live global-hotkey registry and the public UI-facing hotkey API."""

    def __init__(
        self,
        bus: EventBus,
        settings: HotkeysConfig,
        *,
        repositories: Repositories | None = None,
        profile_id: int | None = None,
        engine: MacroEngine | None = None,
        backend_factory: Callable[[], _Backend] = WinApiBackend,
        interception_factory: Callable[[], InterceptionBackend] = InterceptionBackend,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._repositories = repositories
        self._profile_id = profile_id
        _ = engine  # compatibility: execution now belongs exclusively to TriggerDispatcher
        self._backend_factory = backend_factory
        self._interception_factory = interception_factory
        self._clock = clock
        self._backend: _Backend | None = None
        self._thread: threading.Thread | None = None
        self._bindings: dict[int, HotkeyBinding] = {}
        self._next_id = 1
        self._registration_errors: dict[str, str] = {}
        self._ptt_started: float | None = None
        self._ptt_thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._capture_done = threading.Event()
        self._capture_result: Hotkey | BaseException | None = None
        self._capture_modifiers: set[str] = set()
        self._capture_callback: Callable[[Hotkey | None], None] | None = None
        self._interception: InterceptionBackend | None = None
        self._interception_thread: threading.Thread | None = None
        self._subscriptions = [
            bus.subscribe(ConfigChanged, self._on_config_changed, weak=False),
            bus.subscribe(CommandsChanged, self._on_commands_changed, weak=False),
            bus.subscribe(ProfileSwitched, self._on_profile_switched, weak=False),
        ]

    @property
    def occupied(self) -> Mapping[str, str]:
        """Canonical combination to owner, including rejected external claims."""
        result = {binding.hotkey.canonical: binding.owner for binding in self._bindings.values()}
        result.update(self._registration_errors)
        return result

    @property
    def conflicts(self) -> tuple[HotkeyConflict, ...]:
        return detect_conflicts(self.desired_bindings())

    def conflict_for(
        self, hotkey: Hotkey, *, excluding_command_id: int | None = None
    ) -> HotkeyConflict | None:
        """Describe who already claims ``hotkey`` before a UI save."""
        owners = tuple(
            binding.owner
            for binding in self.desired_bindings()
            if binding.hotkey == hotkey and binding.command_id != excluding_command_id
        )
        return HotkeyConflict(hotkey, owners) if owners else None

    def desired_bindings(self) -> tuple[HotkeyBinding, ...]:
        listed = [
            HotkeyBinding(parse_hotkey(getattr(self._settings, field)), label, field)
            for field, label in _SYSTEM_LABELS.items()
            if getattr(self._settings, field, "")
        ]
        listed.extend(self._command_bindings())
        return tuple(listed)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopping.clear()
        if self._settings.use_interception:
            self._start_interception()
        try:
            backend = self._backend_factory()
        except HotkeyBackendUnavailable as exc:
            self._notify(exc.user_message)
            return
        self._backend = backend
        self._thread = threading.Thread(target=self._run_backend, name="ayris-hotkeys", daemon=True)
        self._thread.start()
        if not backend.wait_ready(_START_TIMEOUT_S):
            self.stop()
            raise HotkeyBackendUnavailable(
                "hotkey message loop did not start",
                user_message="Не удалось запустить обработчик глобальных горячих клавиш.",
            )
        self.reload()

    def _run_backend(self) -> None:
        assert self._backend is not None
        try:
            self._backend.run(self._on_hotkey)
        except Exception:
            _log.exception("цикл глобальных горячих клавиш завершился с ошибкой")
            self._notify("Глобальные горячие клавиши перестали работать.", level="error")

    def stop(self) -> None:
        self._stopping.set()
        if self._interception is not None:
            self._interception.stop()
        backend = self._backend
        thread = self._thread
        if backend is not None:
            backend.invoke(self._unregister_all)
            backend.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(_START_TIMEOUT_S)
        self._thread = None
        self._backend = None
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._subscriptions.clear()

    def reload(self) -> None:
        """Atomically replace registrations on the message-loop thread."""
        desired = self.desired_bindings()
        conflicts = detect_conflicts(desired)
        for conflict in conflicts:
            self._notify(conflict.user_message, level="warning")
        # Configuration validation normally catches this before saving.  If the
        # database was edited externally, preserve the first (system bindings are
        # listed first) and skip only later claimants.
        seen: set[Hotkey] = set()
        accepted: list[HotkeyBinding] = []
        for binding in desired:
            if binding.hotkey in seen:
                continue
            seen.add(binding.hotkey)
            accepted.append(binding)
        backend = self._backend
        if backend is None:
            return

        def replace() -> None:
            self._unregister_all()
            self._registration_errors.clear()
            for binding in accepted:
                identifier = self._next_id
                self._next_id += 1
                if self._interception is None:
                    try:
                        backend.register(identifier, binding.hotkey, binding.owner)
                    except HotkeyError as exc:
                        self._registration_errors[binding.hotkey.canonical] = exc.user_message
                        self._notify(exc.user_message, level="warning")
                        continue
                self._bindings[identifier] = binding
            if self._interception is not None:
                self._interception.set_bindings(
                    {identifier: binding.hotkey for identifier, binding in self._bindings.items()}
                )

        if not backend.invoke(replace):
            raise HotkeyBackendUnavailable(
                "hotkey message loop is not accepting commands",
                user_message="Не удалось обновить глобальные горячие клавиши.",
            )

    def _unregister_all(self) -> None:
        backend = self._backend
        if backend is not None:
            for identifier in tuple(self._bindings):
                backend.unregister(identifier)
        self._bindings.clear()

    def _command_bindings(self) -> list[HotkeyBinding]:
        if self._repositories is None or self._profile_id is None:
            return []
        triggers = self._repositories.triggers.list_for_profile(
            self._profile_id, trigger_type=TriggerType.HOTKEY, enabled_only=True
        )
        bindings: list[HotkeyBinding] = []
        for trigger in triggers:
            if trigger.payload.get("enabled", True) is False:
                continue
            combo = trigger.payload.get("combo")
            command = self._repositories.commands.get(trigger.command_id)
            if not isinstance(combo, str) or command is None or not command.enabled:
                continue
            bindings.append(
                HotkeyBinding(
                    hotkey=parse_hotkey(combo),
                    owner=f"команда «{command.name}»",
                    action="command",
                    command_id=trigger.command_id,
                )
            )
        return bindings

    def _on_hotkey(self, identifier: int) -> None:
        binding = self._bindings.get(identifier)
        if binding is None:
            return
        if binding.action == "push_to_talk":
            self._ptt_down(binding)
        elif binding.action == "toggle_wake":
            self._bus.publish(WakeToggleRequested())
        elif binding.action == "toggle_overlay":
            self._bus.publish(OverlayToggleRequested())
        elif binding.action == "toggle_mute":
            self._bus.publish(MicToggleRequested())
        elif binding.action == "cancel":
            self._bus.publish(CancelRequested(reason="hotkey"))
        elif binding.action == "command":
            self._run_command(binding.command_id)

    def _run_command(self, command_id: int | None) -> None:
        if command_id is not None:
            self._bus.publish(HotkeyTriggered(command_id))

    def _ptt_down(self, binding: HotkeyBinding, *, poll_release: bool = True) -> None:
        if self._ptt_started is not None:
            return
        self._ptt_started = self._clock()
        self._bus.publish(PttPressed(hotkey=binding.hotkey.canonical))
        backend = self._backend
        if backend is None or not poll_release:
            return

        def watch_release() -> None:
            while not self._stopping.wait(_PTT_POLL_S):
                if not backend.key_down(binding.hotkey):
                    self._ptt_up(binding)
                    return

        self._ptt_thread = threading.Thread(
            target=watch_release, name="ayris-ptt-release", daemon=True
        )
        self._ptt_thread.start()

    def _ptt_up(self, binding: HotkeyBinding) -> None:
        started, self._ptt_started = self._ptt_started, None
        if started is None:
            return
        duration_ms = max(0, round((self._clock() - started) * 1000))
        self._bus.publish(
            PttReleased(
                hotkey=binding.hotkey.canonical,
                duration_ms=duration_ms,
                accepted=duration_ms >= _PTT_MINIMUM_MS,
            )
        )

    def _start_interception(self) -> None:
        if self._interception is not None:
            return
        try:
            interception = self._interception_factory()
        except HotkeyBackendUnavailable as exc:
            self._notify(exc.user_message, level="warning")
            return
        self._interception = interception
        interception.set_bindings({})

        def dispatch(identifier: int, pressed: bool) -> None:
            binding = self._bindings.get(identifier)
            if binding is None:
                return
            if binding.action == "push_to_talk":
                if pressed:
                    self._ptt_down(binding, poll_release=False)
                else:
                    self._ptt_up(binding)
            elif pressed:
                self._on_hotkey(identifier)

        self._interception_thread = threading.Thread(
            target=interception.run,
            args=(dispatch,),
            name="ayris-hotkeys-interception",
            daemon=True,
        )
        self._interception_thread.start()

    def capture(
        self,
        callback: Callable[[Hotkey | None], None] | None = None,
        *,
        timeout: float | None = None,
    ) -> Hotkey | None:
        """Temporarily hook the keyboard and return one normalized combination.

        With ``callback`` this is asynchronous for a UI dialog.  Without it the
        call waits until a combination, Esc, or ``timeout``.
        """
        backend = self._backend
        if backend is None:
            raise HotkeyBackendUnavailable("hotkey backend is not running")
        self._capture_done.clear()
        self._capture_result = None
        self._capture_modifiers.clear()
        self._capture_callback = callback
        if not backend.invoke(lambda: backend.start_capture(self._capture_key)):
            raise HotkeyBackendUnavailable("capture hook could not be queued")
        if callback is not None:
            return None
        if not self._capture_done.wait(timeout):
            backend.invoke(backend.stop_capture)
            return None
        if isinstance(self._capture_result, BaseException):
            raise self._capture_result
        return self._capture_result

    def cancel_capture(self) -> None:
        backend = self._backend
        if backend is not None:
            backend.invoke(lambda: self._finish_capture(None))

    def _capture_key(self, vk: int, pressed: bool) -> bool:
        name = next((name for name, key in KEYS.items() if key.vk == vk), "")
        if not name:
            return False
        canonical_modifier = _modifier_name(name)
        if canonical_modifier:
            if pressed:
                self._capture_modifiers.add(canonical_modifier)
            else:
                self._capture_modifiers.discard(canonical_modifier)
            return True
        if not pressed:
            return True
        if name == "escape":
            self._finish_capture(None)
            return True
        if name in _CAPTURE_FORBIDDEN:
            return True
        hotkey = Hotkey(
            key=name,
            ctrl="ctrl" in self._capture_modifiers,
            alt="alt" in self._capture_modifiers,
            shift="shift" in self._capture_modifiers,
            win="win" in self._capture_modifiers,
        )
        self._finish_capture(hotkey)
        return True

    def _finish_capture(self, result: Hotkey | None) -> None:
        backend = self._backend
        if backend is not None:
            backend.stop_capture()
        self._capture_result = result
        self._capture_done.set()
        callback, self._capture_callback = self._capture_callback, None
        if callback is not None:
            callback(result)

    def _on_config_changed(self, event: ConfigChanged) -> None:
        if not event.touches("hotkeys"):
            return
        old_interception = self._settings.use_interception
        self._settings = event.diff.settings.hotkeys
        if self._settings.use_interception and not old_interception:
            self._start_interception()
        elif old_interception and not self._settings.use_interception:
            if self._interception is not None:
                self._interception.stop()
            self._interception = None
        self.reload()

    def _on_commands_changed(self, _event: CommandsChanged) -> None:
        self.reload()

    def _on_profile_switched(self, event: ProfileSwitched) -> None:
        self._profile_id = event.profile.id
        self.reload()

    def _notify(self, message: str, *, level: str = "info") -> None:
        if level == "info":
            _log.info("%s", message)
        else:
            _log.warning("%s", message)
        self._bus.publish(
            NotificationRequested(title="Горячие клавиши", message=message, level=level)
        )


def _modifier_name(name: str) -> str:
    if name not in MODIFIERS:
        return ""
    for modifier in ("ctrl", "alt", "shift", "win"):
        if modifier in name or (modifier == "win" and name in {"lwin", "rwin"}):
            return modifier
    return ""


def install_hotkeys(app: AyrisApp) -> HotkeyManager:
    """Attach global hotkeys and their event consumers to the lifecycle."""
    from ayris.core.app import Component, LifecycleStage

    profile_id = app.profile.id
    if profile_id is None:
        raise HotkeyError(
            "active profile has no database id",
            user_message="Не удалось загрузить горячие клавиши активного профиля.",
        )
    manager = HotkeyManager(
        app.bus,
        app.settings.hotkeys,
        repositories=app.repositories,
        profile_id=profile_id,
    )

    def toggle_mic(_event: MicToggleRequested) -> None:
        app.state.toggle_mic()

    unsubscribers = [app.bus.subscribe(MicToggleRequested, toggle_mic, weak=False)]

    def stop() -> None:
        manager.stop()
        for unsubscribe in unsubscribers:
            unsubscribe()
        unsubscribers.clear()

    app.add_component(
        Component(
            name="глобальные горячие клавиши",
            stage=LifecycleStage.ACTIONS,
            start=manager.start,
            stop=stop,
        )
    )
    return manager
