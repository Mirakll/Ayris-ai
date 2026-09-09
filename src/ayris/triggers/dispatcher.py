"""The only route from every trigger source to ``MacroEngine.start``."""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ayris.actions.macros.context import TriggerSource
from ayris.actions.macros.serializer import command_from_rows
from ayris.core.events import CommandsChanged, EventBus, HotkeyTriggered, IntentMatched
from ayris.core.models import Trigger, TriggerType, from_db_timestamp
from ayris.core.profile import ProfileSwitched
from ayris.triggers.debounce import Debouncer, RateLimiter
from ayris.triggers.schedule import ScheduleEntry, TriggerSchedule
from ayris.triggers.system_events import SystemEvent, SystemEventMonitor

if TYPE_CHECKING:
    from ayris.actions.macros.engine import MacroEngine
    from ayris.actions.macros.schema import CommandModel
    from ayris.core.app import AyrisApp
    from ayris.core.pipeline_states import Scheduler
    from ayris.core.repositories import Repositories


_DEFAULT_DEBOUNCE = {
    "active_window_changed": 0.25,
    "device_connected": 1.0,
    "device_disconnected": 1.0,
    "power_changed": 1.0,
    "fullscreen_entered": 0.5,
    "fullscreen_exited": 0.5,
    "process_started": 0.1,
    "process_stopped": 0.1,
}


class TriggerDispatcher:
    """Index active triggers, resolve them, and launch their commands."""

    def __init__(
        self,
        bus: EventBus,
        repositories: Repositories,
        engine: MacroEngine,
        profile_id: int,
        *,
        monitor: SystemEventMonitor | None = None,
        scheduler: Scheduler | None = None,
        wall_clock: Any = None,
        monotonic: Any = time.monotonic,
        launches_per_second: int = 10,
        profile_changed: Callable[[int], None] | None = None,
    ) -> None:
        self._bus = bus
        self._repositories = repositories
        self._engine = engine
        self._profile_id = profile_id
        self._profile_changed = profile_changed
        self._monitor = monitor or SystemEventMonitor(bus)
        self._debounce = Debouncer(clock=monotonic)
        self._rate = RateLimiter(launches_per_second, clock=monotonic)
        self._schedule = TriggerSchedule(
            self._on_timer, scheduler=scheduler, wall_clock=wall_clock, monotonic=monotonic
        )
        self._triggers: dict[int, Trigger] = {}
        self._by_command: dict[int, set[TriggerType]] = {}
        self._events: dict[str, list[Trigger]] = {}
        self._commands: dict[int, CommandModel] = {}
        self._subscriptions = [
            bus.subscribe(IntentMatched, self._on_voice, weak=False),
            bus.subscribe(HotkeyTriggered, self._on_hotkey, weak=False),
            bus.subscribe(SystemEvent, self._on_system_event, weak=False),
            bus.subscribe(CommandsChanged, self._on_commands_changed, weak=False),
            bus.subscribe(ProfileSwitched, self._on_profile_switched, weak=False),
        ]
        self.reload()

    @property
    def profile_id(self) -> int:
        return self._profile_id

    @property
    def active_trigger_ids(self) -> frozenset[int]:
        return frozenset(self._triggers)

    def reload(self) -> None:
        triggers = self._repositories.triggers.list_for_profile(self._profile_id, enabled_only=True)
        active = [
            trigger
            for trigger in triggers
            if trigger.id is not None and trigger.payload.get("enabled", True) is not False
        ]
        commands: dict[int, CommandModel] = {}
        by_command: dict[int, set[TriggerType]] = {}
        events: dict[str, list[Trigger]] = {}
        schedules: list[ScheduleEntry] = []
        for trigger in active:
            row = self._repositories.commands.get(trigger.command_id)
            if row is None or not row.enabled or row.id is None:
                continue
            if row.id not in commands:
                commands[row.id] = command_from_rows(
                    row, self._repositories.triggers.list_for_command(row.id)
                )
            by_command.setdefault(row.id, set()).add(trigger.type)
            if trigger.type is TriggerType.EVENT:
                event_name = trigger.payload.get("event_name")
                if isinstance(event_name, str):
                    events.setdefault(event_name, []).append(trigger)
            elif trigger.type is TriggerType.TIMER:
                schedule = self._schedule_entry(trigger)
                if schedule is not None:
                    schedules.append(schedule)
        self._triggers = {trigger.id: trigger for trigger in active if trigger.id is not None}
        self._commands = commands
        self._by_command = by_command
        self._events = events
        self._debounce.clear()
        self._rate.clear()
        self._schedule.replace(schedules)
        self._monitor.replace_subscriptions(events)

    def enable(self, trigger_id: int) -> None:
        self._set_enabled(trigger_id, True)

    def disable(self, trigger_id: int) -> None:
        self._set_enabled(trigger_id, False)

    def close(self) -> None:
        self._schedule.close()
        self._monitor.stop()
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._subscriptions.clear()

    def _set_enabled(self, trigger_id: int, enabled: bool) -> None:
        trigger = self._repositories.triggers.get(trigger_id)
        if trigger is None:
            raise KeyError(trigger_id)
        payload = dict(trigger.payload)
        payload["enabled"] = enabled
        self._repositories.triggers.update(replace(trigger, payload=payload))
        self.reload()
        self._bus.publish(CommandsChanged(command_id=trigger.command_id))

    def _on_voice(self, event: IntentMatched) -> None:
        if event.command_id is not None and self._has(event.command_id, TriggerType.VOICE):
            self._launch(
                event.command_id,
                TriggerSource.VOICE,
                slots=event.slots,
                request_id=event.request_id,
            )

    def _on_hotkey(self, event: HotkeyTriggered) -> None:
        if self._has(event.command_id, TriggerType.HOTKEY):
            self._launch(event.command_id, TriggerSource.HOTKEY)

    def _on_timer(self, trigger_id: int) -> None:
        trigger = self._triggers.get(trigger_id)
        if trigger is not None:
            self._launch(trigger.command_id, TriggerSource.TIMER)

    def _on_system_event(self, event: SystemEvent) -> None:
        values = (
            event.kind,
            event.process.casefold(),
            event.title.casefold(),
            event.device_type.casefold(),
            event.device_id.casefold(),
            event.on_ac,
            event.fullscreen,
        )
        window = _DEFAULT_DEBOUNCE.get(event.kind, 0.25)
        for trigger in self._events.get(event.kind, ()):
            custom = trigger.payload.get("debounce_ms")
            debounce = float(custom) / 1000.0 if isinstance(custom, int | float) else window
            if not _matches_filter(event, trigger.payload.get("filter_json")):
                continue
            if self._debounce.accept((trigger.id, values), debounce):
                self._launch(trigger.command_id, TriggerSource.EVENT)

    def _launch(
        self,
        command_id: int,
        source: TriggerSource,
        *,
        slots: Mapping[str, Any] | None = None,
        request_id: str = "",
    ) -> None:
        command = self._commands.get(command_id)
        if command is not None and self._rate.accept(command_id):
            self._engine.start(command, slots=slots, trigger=source, request_id=request_id)

    def _has(self, command_id: int, trigger_type: TriggerType) -> bool:
        return trigger_type in self._by_command.get(command_id, ())

    def _on_commands_changed(self, _event: CommandsChanged) -> None:
        self.reload()

    def _on_profile_switched(self, event: ProfileSwitched) -> None:
        if event.profile.id is not None:
            self._profile_id = event.profile.id
            if self._profile_changed is not None:
                self._profile_changed(self._profile_id)
            self.reload()

    @staticmethod
    def _schedule_entry(trigger: Trigger) -> ScheduleEntry | None:
        assert trigger.id is not None
        raw_fire_at = trigger.payload.get("fire_at")
        fire_at = from_db_timestamp(raw_fire_at) if raw_fire_at else None
        cron = trigger.payload.get("cron")
        policy = trigger.payload.get("missed", "run_once")
        if policy not in {"run_once", "skip"}:
            policy = "run_once"
        if fire_at is None and not isinstance(cron, str):
            return None
        return ScheduleEntry(
            trigger.id,
            fire_at=fire_at,
            cron=cron if isinstance(cron, str) else "",
            missed=policy,
        )


def _matches_filter(event: SystemEvent, raw_filter: object) -> bool:
    if not isinstance(raw_filter, dict):
        return True
    for field, expected in raw_filter.items():
        if not hasattr(event, field):
            return False
        actual = getattr(event, field)
        if isinstance(expected, str):
            if not fnmatch.fnmatchcase(str(actual).casefold(), expected.casefold()):
                return False
        elif actual != expected:
            return False
    return True


def install_triggers(app: AyrisApp) -> TriggerDispatcher:
    """Attach one engine, dispatcher, system monitor, scheduler and hotkeys."""
    from ayris.actions.macros.engine import MacroEngine
    from ayris.actions.macros.variables import DatabaseVariables
    from ayris.actions.registry import ActionRegistry
    from ayris.core.app import Component, LifecycleStage
    from ayris.utils.hotkey_manager import install_hotkeys

    profile_id = app.profile.id
    if profile_id is None:
        raise RuntimeError("active profile has no database id")
    registry = ActionRegistry(
        bus=app.bus,
        audit=app.repositories.audit,
        audit_enabled=lambda: app.settings.privacy.audit_commands,
    )
    registry.discover()
    from ayris.security.confirmation import ConfirmationManager, windows_dialog_prompt
    from ayris.security.pin import PinManager

    registry.set_confirmation(
        ConfirmationManager(
            lambda: app.settings.privacy,
            action=lambda name: registry.get(name) if registry.has(name) else None,
            pin=PinManager(),
            dialog=windows_dialog_prompt,
        )
    )

    def command_by_name(name: str) -> CommandModel | None:
        row = app.repositories.commands.get_by_name(dispatcher.profile_id, name)
        if row is None or row.id is None:
            return None
        return command_from_rows(row, app.repositories.triggers.list_for_command(row.id))

    engine = MacroEngine(
        registry,
        bus=app.bus,
        store=DatabaseVariables(app.repositories.variables, profile_id=profile_id),
        library=command_by_name,
    )

    def switch_variables(new_profile_id: int) -> None:
        engine.replace_variables(
            DatabaseVariables(app.repositories.variables, profile_id=new_profile_id)
        )

    dispatcher = TriggerDispatcher(
        app.bus, app.repositories, engine, profile_id, profile_changed=switch_variables
    )

    def stop() -> None:
        dispatcher.close()
        engine.shutdown()
        registry.shutdown()

    app.add_component(
        Component(
            name="диспетчер триггеров",
            stage=LifecycleStage.ACTIONS,
            stop=stop,
        )
    )
    install_hotkeys(app)
    return dispatcher
