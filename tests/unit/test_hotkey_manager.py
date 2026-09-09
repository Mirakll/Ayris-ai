from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from ayris.actions.input.keys import KEYS
from ayris.core.config import HotkeysConfig, Settings, diff_settings
from ayris.core.events import (
    CancelRequested,
    CommandsChanged,
    ConfigChanged,
    Event,
    EventBus,
    HotkeyTriggered,
    MicToggleRequested,
    NotificationRequested,
    OverlayToggleRequested,
    PttPressed,
    PttReleased,
    WakeToggleRequested,
)
from ayris.core.models import Command, Trigger, TriggerType
from ayris.utils.hotkey_backends.winapi import (
    ERROR_HOTKEY_ALREADY_REGISTERED,
    HotkeyBackendUnavailable,
    HotkeyRegistrationError,
)
from ayris.utils.hotkey_manager import HotkeyBinding, HotkeyManager, detect_conflicts
from ayris.utils.hotkeys import Hotkey, parse_hotkey


class FakeBackend:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.done = threading.Event()
        self.callback: Any = None
        self.registered: dict[int, tuple[Hotkey, str]] = {}
        self.unregistered: list[int] = []
        self.down = True
        self.capture_callback: Any = None
        self.fail_combo = ""

    def run(self, callback: Any) -> None:
        self.callback = callback
        self.ready.set()
        self.done.wait(2)

    def wait_ready(self, timeout: float) -> bool:
        return self.ready.wait(timeout)

    def invoke(self, command: Any) -> bool:
        command()
        return True

    def register(self, identifier: int, hotkey: Hotkey, owner: str) -> None:
        if hotkey.canonical == self.fail_combo:
            raise HotkeyRegistrationError(hotkey, owner, ERROR_HOTKEY_ALREADY_REGISTERED)
        self.registered[identifier] = (hotkey, owner)

    def unregister(self, identifier: int) -> None:
        self.unregistered.append(identifier)
        self.registered.pop(identifier, None)

    def key_down(self, _hotkey: Hotkey) -> bool:
        return self.down

    def start_capture(self, callback: Any) -> None:
        self.capture_callback = callback

    def stop_capture(self) -> None:
        self.capture_callback = None

    def stop(self) -> None:
        self.done.set()

    def fire(self, combo: str) -> None:
        identifier = next(
            key for key, (hotkey, _owner) in self.registered.items() if hotkey.canonical == combo
        )
        self.callback(identifier)

    def key(self, name: str, pressed: bool = True) -> None:
        assert self.capture_callback is not None
        self.capture_callback(KEYS[name].vk, pressed)


class FakeCommands:
    def __init__(self, commands: list[Command]) -> None:
        self.rows = {command.id: command for command in commands}

    def get(self, command_id: int) -> Command | None:
        return self.rows.get(command_id)


class FakeTriggers:
    def __init__(self, triggers: list[Trigger]) -> None:
        self.rows = triggers

    def list_for_profile(self, *_args: Any, **_kwargs: Any) -> list[Trigger]:
        return list(self.rows)

    def list_for_command(self, command_id: int) -> list[Trigger]:
        return [item for item in self.rows if item.command_id == command_id]


class FakeRepositories:
    def __init__(self, commands: list[Command], triggers: list[Trigger]) -> None:
        self.commands = FakeCommands(commands)
        self.triggers = FakeTriggers(triggers)


def make_manager(
    *,
    backend: FakeBackend | None = None,
    settings: HotkeysConfig | None = None,
    repositories: Any = None,
    engine: Any = None,
    clock: Any = time.monotonic,
) -> tuple[HotkeyManager, FakeBackend, EventBus, list[Event]]:
    fake = backend or FakeBackend()
    bus = EventBus(thread_id=None)
    events: list[Event] = []
    bus.subscribe(Event, events.append, weak=False)
    manager = HotkeyManager(
        bus,
        settings or HotkeysConfig(),
        repositories=repositories,
        profile_id=1 if repositories is not None else None,
        engine=engine,
        backend_factory=lambda: fake,
        clock=clock,
    )
    manager.start()
    return manager, fake, bus, events


def test_detects_conflicts_with_both_owner_names() -> None:
    combo = parse_hotkey("ctrl+alt+k")
    conflicts = detect_conflicts(
        [
            HotkeyBinding(combo, "команда «Первая»", "command", 1),
            HotkeyBinding(combo, "команда «Вторая»", "command", 2),
        ]
    )
    assert conflicts[0].owners == ("команда «Первая»", "команда «Вторая»")
    assert "Первая" in conflicts[0].user_message
    assert "Вторая" in conflicts[0].user_message


def test_system_binding_wins_over_conflicting_command() -> None:
    command = Command(id=1, profile_id=1, name="Дубль")
    trigger = Trigger(command_id=1, type=TriggerType.HOTKEY, payload={"combo": "ctrl+shift+a"})
    manager, backend, _bus, events = make_manager(
        repositories=FakeRepositories([command], [trigger])
    )
    try:
        owners = [owner for _hotkey, owner in backend.registered.values()]
        assert "переключение Wake Word" in owners
        assert "команда «Дубль»" not in owners
        assert any(isinstance(event, NotificationRequested) for event in events)
    finally:
        manager.stop()


def test_external_registration_conflict_is_reported_in_russian() -> None:
    backend = FakeBackend()
    backend.fail_combo = "ctrl+shift+o"
    manager, _backend, _bus, events = make_manager(backend=backend)
    try:
        assert "ctrl+shift+o" in manager.occupied
        notices = [event.message for event in events if isinstance(event, NotificationRequested)]
        assert any("уже занято Windows или другой программой" in text for text in notices)
    finally:
        manager.stop()


def test_system_hotkeys_publish_typed_events_only() -> None:
    manager, backend, _bus, events = make_manager()
    try:
        backend.fire("ctrl+shift+a")
        backend.fire("ctrl+shift+o")
        backend.fire("ctrl+shift+m")
        backend.fire("escape")
        assert any(isinstance(event, WakeToggleRequested) for event in events)
        assert any(isinstance(event, OverlayToggleRequested) for event in events)
        assert any(isinstance(event, MicToggleRequested) for event in events)
        assert any(isinstance(event, CancelRequested) for event in events)
    finally:
        manager.stop()


def test_command_hotkey_publishes_for_dispatcher() -> None:
    command = Command(id=7, profile_id=1, name="Открыть карту")
    trigger = Trigger(command_id=7, type=TriggerType.HOTKEY, payload={"combo": "ctrl+alt+k"})
    manager, backend, _bus, events = make_manager(
        repositories=FakeRepositories([command], [trigger])
    )
    try:
        backend.fire("ctrl+alt+k")
        fired = [event for event in events if isinstance(event, HotkeyTriggered)]
        assert [event.command_id for event in fired] == [7]
    finally:
        manager.stop()


def test_disabled_command_and_trigger_do_not_claim_hotkeys() -> None:
    commands = [
        Command(id=1, profile_id=1, name="Off", enabled=False),
        Command(id=2, profile_id=1, name="Trigger off"),
    ]
    triggers = [
        Trigger(command_id=1, type=TriggerType.HOTKEY, payload={"combo": "ctrl+1"}),
        Trigger(
            command_id=2,
            type=TriggerType.HOTKEY,
            payload={"combo": "ctrl+2", "enabled": False},
        ),
    ]
    manager, backend, _bus, _events = make_manager(
        repositories=FakeRepositories(commands, triggers)
    )
    try:
        combos = {hotkey.canonical for hotkey, _owner in backend.registered.values()}
        assert "ctrl+1" not in combos
        assert "ctrl+2" not in combos
    finally:
        manager.stop()


def test_commands_changed_reregisters_without_restart() -> None:
    command = Command(id=1, profile_id=1, name="Команда")
    trigger = Trigger(command_id=1, type=TriggerType.HOTKEY, payload={"combo": "ctrl+1"})
    repositories = FakeRepositories([command], [trigger])
    manager, backend, bus, _events = make_manager(repositories=repositories)
    try:
        repositories.triggers.rows[0] = replace(trigger, payload={"combo": "ctrl+2"})
        bus.publish(CommandsChanged(command_id=1))
        combos = {hotkey.canonical for hotkey, _owner in backend.registered.values()}
        assert "ctrl+1" not in combos
        assert "ctrl+2" in combos
        assert backend.unregistered
    finally:
        manager.stop()


def test_config_changed_reregisters_without_restart() -> None:
    manager, backend, bus, _events = make_manager()
    try:
        old = Settings()
        new = old.model_copy(
            update={"hotkeys": old.hotkeys.model_copy(update={"toggle_overlay": "ctrl+alt+o"})}
        )
        bus.publish(ConfigChanged(diff=diff_settings(old, new)))
        combos = {hotkey.canonical for hotkey, _owner in backend.registered.values()}
        assert "ctrl+shift+o" not in combos
        assert "ctrl+alt+o" in combos
    finally:
        manager.stop()


def test_ptt_press_is_debounced_and_short_release_is_rejected() -> None:
    now = [10.0]
    manager, backend, _bus, events = make_manager(clock=lambda: now[0])
    try:
        backend.fire("ctrl+shift+space")
        backend.fire("ctrl+shift+space")
        assert len([event for event in events if isinstance(event, PttPressed)]) == 1
        now[0] += 0.05
        backend.down = False
        deadline = time.monotonic() + 1
        while not any(isinstance(event, PttReleased) for event in events):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        released = next(event for event in events if isinstance(event, PttReleased))
        assert released.duration_ms == 50
        assert released.accepted is False
    finally:
        manager.stop()


def test_capture_normalizes_combo_and_escape_cancels() -> None:
    manager, backend, _bus, _events = make_manager()
    captured: list[Hotkey | None] = []
    try:
        manager.capture(captured.append)
        backend.key("lctrl")
        backend.key("lshift")
        backend.key("capslock")
        assert captured == []
        backend.key("k")
        assert captured == [parse_hotkey("ctrl+shift+k")]

        manager.capture(captured.append)
        backend.key("escape")
        assert captured[-1] is None
    finally:
        manager.stop()


def test_interception_failure_falls_back_to_winapi() -> None:
    def unavailable() -> Any:
        raise HotkeyBackendUnavailable(
            "no driver", user_message="Драйвер Interception не установлен. Использую WinAPI."
        )

    backend = FakeBackend()
    bus = EventBus(thread_id=None)
    events: list[Event] = []
    bus.subscribe(Event, events.append, weak=False)
    manager = HotkeyManager(
        bus,
        HotkeysConfig(use_interception=True),
        backend_factory=lambda: backend,
        interception_factory=unavailable,
    )
    manager.start()
    try:
        assert backend.registered
        assert any(
            isinstance(event, NotificationRequested) and "Interception" in event.message
            for event in events
        )
    finally:
        manager.stop()


@pytest.mark.parametrize("combo", ["ctrl+k", "alt+f8"])
def test_conflict_for_is_available_to_ui(combo: str) -> None:
    command = Command(id=1, profile_id=1, name="Команда")
    trigger = Trigger(command_id=1, type=TriggerType.HOTKEY, payload={"combo": combo})
    manager = HotkeyManager(
        EventBus(thread_id=None),
        HotkeysConfig(),
        repositories=FakeRepositories([command], [trigger]),
        profile_id=1,
    )
    conflict = manager.conflict_for(parse_hotkey(combo))
    assert conflict is not None
    assert conflict.owners == ("команда «Команда»",)
