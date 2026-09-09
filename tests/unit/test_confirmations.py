"""Confirmation policy and adapters for dangerous actions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

import pytest

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import ActionRegistry, ConfirmationRequest
from ayris.actions.result import ActionResult
from ayris.core.config import PrivacyConfig
from ayris.core.database import Database, reset_database
from ayris.core.errors import ActionNotConfirmed
from ayris.core.migrations import apply_migrations
from ayris.core.models import ExecutionResult
from ayris.core.repositories import Repositories
from ayris.security.confirmation import ConfirmationManager
from ayris.security.hello import WindowsHello
from ayris.security.pin import PinManager
from ayris.security.policy import ConfirmationMethod, ConfirmationPolicy

pytestmark = pytest.mark.unit


class MemoryStore:
    def __init__(self) -> None:
        self.value: str | None = None

    def get_password(self, _service: str, _account: str) -> str | None:
        return self.value

    def set_password(self, _service: str, _account: str, value: str) -> None:
        self.value = value

    def delete_password(self, _service: str, _account: str) -> None:
        self.value = None


class Dangerous(Action):
    meta = ActionMeta(
        name="Dangerous",
        category=ActionCategory.SYSTEM,
        title_ru="Опасное действие",
        is_dangerous=True,
    )

    def __init__(self) -> None:
        self.calls = 0

    def run(self, params: ActionParams) -> ActionResult[int]:
        self.calls += 1
        return ActionResult.done(value=self.calls)


class Ordinary(Action):
    meta = ActionMeta(name="Ordinary", category=ActionCategory.LOGIC, title_ru="Обычное действие")

    def __init__(self) -> None:
        self.calls = 0

    def run(self, params: ActionParams) -> ActionResult[int]:
        self.calls += 1
        return ActionResult.done(value=self.calls)


REQUEST = ConfirmationRequest(action="Dangerous", title_ru="Опасное действие")


def privacy(**changes: Any) -> PrivacyConfig:
    return PrivacyConfig.model_validate(changes)


def test_voice_yes_no_timeout_and_pipeline_guard() -> None:
    events: list[str] = []

    @contextmanager
    def paused() -> Iterator[None]:
        events.append("pause")
        try:
            yield
        finally:
            events.append("resume")

    manager = ConfirmationManager(
        privacy(confirmation_method="voice"), voice=lambda _q, _t: "да", pause_pipeline=paused
    )
    assert manager(REQUEST).confirmed is True
    assert events == ["pause", "resume"]

    manager = ConfirmationManager(privacy(confirmation_method="voice"), voice=lambda _q, _t: "нет")
    assert manager(REQUEST).reason == "rejected"
    manager = ConfirmationManager(privacy(confirmation_method="voice"), voice=lambda _q, _t: None)
    assert manager(REQUEST).reason == "timeout"


def test_pin_is_salted_hashed_and_limited() -> None:
    store = MemoryStore()
    sleeps: list[float] = []
    pin = PinManager(store, attempts=3, delay_sec=0.25, sleeper=sleeps.append)
    pin.set_pin("1234")
    assert store.value is not None
    assert "1234" not in store.value
    payload = json.loads(store.value)
    assert payload["salt"] and payload["hash"]
    assert pin.verify("1234") is True
    assert pin.verify("4321") is False
    attempts: list[int] = []
    assert pin.prompt(lambda number: attempts.append(number) or "0000") is False
    assert attempts == [1, 2, 3]
    assert sleeps == [0.25, 0.25]


class Value:
    def __init__(self, value: object) -> None:
        self.value = value
        self.cancelled = False

    def get(self) -> object:
        return self.value

    def cancel(self) -> None:
        self.cancelled = True


class HelloBackend:
    def __init__(self, availability: object, result: object = "verified") -> None:
        self.availability = availability
        self.result = result

    def check_availability_async(self) -> Value:
        return Value(self.availability)

    def request_verification_async(self, _message: str) -> Value:
        return Value(self.result)


class Availability(Enum):
    AVAILABLE = 0
    DEVICE_NOT_PRESENT = 1


class Verification(Enum):
    VERIFIED = 0
    CANCELED = 6


def test_windows_hello_verifies_and_unavailable_falls_back() -> None:
    hello = WindowsHello(HelloBackend(Availability.AVAILABLE, Verification.VERIFIED))
    assert hello.verify("Подтвердите").confirmed is True

    unavailable = WindowsHello(HelloBackend(Availability.DEVICE_NOT_PRESENT))
    manager = ConfirmationManager(
        privacy(confirmation_method="hello", confirmation_fallback="voice"),
        voice=lambda _q, _t: "да",
        hello=unavailable,
    )
    assert manager(REQUEST).confirmed is True


def test_policy_uses_metadata_and_package_category() -> None:
    settings = privacy(
        confirmation_method="dialog",
        confirmation_by_category={"admin": "pin", "power": "hello"},
    )
    admin = Dangerous()
    admin.meta = ActionMeta(
        name="Dangerous",
        category=ActionCategory.SYSTEM,
        title_ru="Опасное действие",
        require_admin=True,
        is_dangerous=True,
    )
    assert ConfirmationPolicy(settings).method_for(admin) is ConfirmationMethod.PIN


def test_editable_action_list_uses_the_registry_barrier() -> None:
    settings = privacy(confirmation_method="voice", confirmation_actions=("Ordinary",))
    registry = ActionRegistry(audit_enabled=lambda: False)
    registry.add(Ordinary)
    registry.set_confirmation(ConfirmationManager(settings, voice=lambda _q, _t: "нет"))
    try:
        with pytest.raises(ActionNotConfirmed):
            registry.execute("Ordinary")
        assert isinstance(registry.get("Ordinary"), Ordinary)
        assert registry.get("Ordinary").calls == 0
    finally:
        registry.shutdown()


def test_registry_end_to_end_yes_executes_and_no_is_audited(tmp_path: Any) -> None:
    db = Database.open(tmp_path / "confirmations.db")
    apply_migrations(db)
    repos = Repositories(db)
    registry = ActionRegistry(audit=repos.audit, audit_enabled=lambda: True)
    registry.add(Dangerous)
    settings = privacy(confirmation_method="voice")
    try:
        registry.set_confirmation(ConfirmationManager(settings, voice=lambda _q, _t: "да"))
        assert registry.execute("Dangerous").value == 1
        registry.set_confirmation(ConfirmationManager(settings, voice=lambda _q, _t: "нет"))
        with pytest.raises(ActionNotConfirmed):
            registry.execute("Dangerous")
        action = registry.get("Dangerous")
        assert isinstance(action, Dangerous)
        assert action.calls == 1
        entries = repos.audit.recent(10)
        assert len(entries) == 2
        assert entries[0].confirmed is False
        assert entries[0].result is ExecutionResult.CANCELLED
        assert entries[1].confirmed is True
    finally:
        registry.shutdown()
        db.close()
        reset_database()
