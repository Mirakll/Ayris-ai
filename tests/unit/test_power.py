"""Tests for the power actions: sleep, hibernation, reboot, shutdown, lock.

**Nothing here is allowed near the real backend.** The suite runs on a developer's
machine and on a CI runner, and both would be lost to a single ``ExitWindowsEx``
that slipped through — the runner mid-job, the developer mid-thought. So the
:func:`_ban_real_power` fixture replaces every WinAPI power entry point with
something that fails the test instead of the machine, and every test asserts on a
:class:`RecordingPowerBackend` that writes the call down rather than making it.
That fixture is the point of this file as much as any assertion in it.

The parser fixtures below are real ``powercfg /a`` output, on a Russian and an
English system. They are transcribed rather than generated because what is being
tested is exactly the thing a generated fixture would get wrong: that the parser
reads the *structure* and not the words.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ayris.actions.registry import ActionRegistry
from ayris.actions.system.power import (
    MAX_DELAY_S,
    CancelPowerAction,
    LockWorkstation,
    PowerAction,
    PowerCall,
    PowerOperation,
    PowerRequest,
    RecordingPowerBackend,
    WinApiPowerBackend,
    cancel_pending,
    format_delay_ru,
    get_power_backend,
    hibernation_from_powercfg,
    pending_delay,
    perform,
    set_power_backend,
)
from ayris.core.errors import ActionNotConfirmed, ActionUnavailable
from ayris.utils import winapi

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# recorded powercfg output
# --------------------------------------------------------------------------- #

POWERCFG_RU_ENABLED = """\
Следующие состояния сна доступны в этой системе:
    Ждущий режим (S3)
    Гибернация
    Ждущий режим (S0 с низким энергопотреблением)
    Гибридный спящий режим
    Быстрый запуск

Следующие состояния сна недоступны в этой системе:
    Ждущий режим (S1)
        Встроенное ПО системы не поддерживает данное состояние ожидания.
    Ждущий режим (S2)
        Встроенное ПО системы не поддерживает данное состояние ожидания.
"""

POWERCFG_RU_DISABLED = """\
Следующие состояния сна доступны в этой системе:
    Ждущий режим (S3)

Следующие состояния сна недоступны в этой системе:
    Гибернация
        Гибернация не включена.
    Гибридный спящий режим
        Отсутствует режим гибернации.
    Быстрый запуск
        Отсутствует режим гибернации.
"""

POWERCFG_EN_ENABLED = """\
The following sleep states are available on this system:
    Standby (S3)
    Hibernate
    Hybrid Sleep
    Fast Startup

The following sleep states are not available on this system:
    Standby (S1)
        The system firmware does not support this standby state.
"""

POWERCFG_EN_DISABLED = """\
The following sleep states are available on this system:
    Standby (S3)

The following sleep states are not available on this system:
    Hibernate
        Hibernation has not been enabled.
    Hybrid Sleep
        Hibernation is not available.
"""


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _ban_real_power(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every real power call a test failure rather than a reboot.

    Belt and braces on top of the recording backend: the backend is what the code
    under test is supposed to go through, and this is what catches the day
    somebody adds a call that bypasses it.
    """

    def forbidden(name: str) -> object:
        def refuse(*args: object, **kwargs: object) -> None:
            message = f"тест дошёл до настоящего {name}"
            raise AssertionError(message)

        return refuse

    for name in (
        "set_suspend_state",
        "exit_windows",
        "initiate_shutdown",
        "abort_shutdown",
        "lock_workstation",
        "enable_privilege",
    ):
        monkeypatch.setattr(winapi, name, forbidden(name))


@pytest.fixture
def backend() -> Iterator[RecordingPowerBackend]:
    """A recording backend installed for the duration of one test."""
    recorder = RecordingPowerBackend()
    set_power_backend(recorder)
    try:
        yield recorder
    finally:
        set_power_backend(None)
        cancel_pending()


@pytest.fixture
def registry(backend: RecordingPowerBackend) -> Iterator[ActionRegistry]:
    """A registry holding the power actions and nothing else."""
    del backend
    instance = ActionRegistry(audit_enabled=lambda: False)
    for action in (PowerAction, LockWorkstation, CancelPowerAction):
        instance.add(action)
    try:
        yield instance
    finally:
        instance.shutdown()


# --------------------------------------------------------------------------- #
# the enum
# --------------------------------------------------------------------------- #


class TestPowerOperation:
    def test_every_member_has_russian_names(self) -> None:
        for operation in PowerOperation:
            assert operation.title_ru
            assert operation.announce_ru.endswith(".")

    def test_suspend_operations_are_the_two_sleeps(self) -> None:
        suspends = {op for op in PowerOperation if op.suspends}
        assert suspends == {PowerOperation.SLEEP, PowerOperation.HIBERNATE}

    def test_only_shutdown_and_reboot_can_be_scheduled(self) -> None:
        schedulable = {op for op in PowerOperation if op.schedulable}
        assert schedulable == {PowerOperation.REBOOT, PowerOperation.SHUTDOWN}

    def test_exit_flags_match_the_winapi_constants(self) -> None:
        assert PowerOperation.LOGOFF.exit_flags == winapi.EWX_LOGOFF
        assert PowerOperation.REBOOT.exit_flags == winapi.EWX_REBOOT
        assert PowerOperation.SHUTDOWN.exit_flags == winapi.EWX_SHUTDOWN | winapi.EWX_POWEROFF

    def test_suspend_operations_have_no_exit_flags(self) -> None:
        assert PowerOperation.SLEEP.exit_flags == 0
        assert PowerOperation.HIBERNATE.exit_flags == 0

    def test_shutdown_cuts_the_power(self) -> None:
        # EWX_SHUTDOWN alone leaves the machine on with the power-off screen.
        assert PowerOperation.SHUTDOWN.exit_flags & winapi.EWX_POWEROFF


class TestDelayWording:
    @pytest.mark.parametrize(
        ("delay_s", "expected"),
        [
            (0, "сейчас"),
            (1, "через 1 секунду"),
            (2, "через 2 секунды"),
            (5, "через 5 секунд"),
            (11, "через 11 секунд"),
            (21, "через 21 секунду"),
            (30, "через 30 секунд"),
            (60, "через 1 минуту"),
            (120, "через 2 минуты"),
            (300, "через 5 минут"),
            (660, "через 11 минут"),
            (90, "через 90 секунд"),
        ],
    )
    def test_wording(self, delay_s: int, expected: str) -> None:
        assert format_delay_ru(delay_s) == expected

    def test_request_summary_mentions_the_delay(self) -> None:
        request = PowerRequest(operation=PowerOperation.SHUTDOWN, delay_s=300)
        assert request.summary_ru == "выключение через 5 минут"
        assert request.delayed
        assert request.cancellable

    def test_immediate_request_is_not_cancellable(self) -> None:
        request = PowerRequest(operation=PowerOperation.SHUTDOWN)
        assert request.summary_ru == "выключение"
        assert not request.delayed
        assert not request.cancellable


# --------------------------------------------------------------------------- #
# powercfg parsing
# --------------------------------------------------------------------------- #


class TestHibernationParser:
    @pytest.mark.parametrize("output", [POWERCFG_RU_ENABLED, POWERCFG_EN_ENABLED])
    def test_enabled_on_both_locales(self, output: str) -> None:
        assert hibernation_from_powercfg(output) is True

    @pytest.mark.parametrize("output", [POWERCFG_RU_DISABLED, POWERCFG_EN_DISABLED])
    def test_disabled_on_both_locales(self, output: str) -> None:
        assert hibernation_from_powercfg(output) is False

    def test_hibernation_in_the_unavailable_list_only_is_not_available(self) -> None:
        # The word appears in the output either way; what decides is which
        # section it appears in.
        assert "Гибернация" in POWERCFG_RU_DISABLED
        assert hibernation_from_powercfg(POWERCFG_RU_DISABLED) is False

    def test_unparseable_output_reads_as_available(self) -> None:
        # Better to let Windows refuse the call than to refuse it here on a guess.
        assert hibernation_from_powercfg("") is True
        assert hibernation_from_powercfg("ERROR: недостаточно прав.") is True

    def test_indented_reasons_are_not_mistaken_for_states(self) -> None:
        output = (
            "Available:\n"
            "    Standby (S3)\n"
            "Not available:\n"
            "    Hibernate\n"
            "        Hibernation has not been enabled.\n"
        )
        assert hibernation_from_powercfg(output) is False


# --------------------------------------------------------------------------- #
# performing a request
# --------------------------------------------------------------------------- #


class TestImmediate:
    def test_sleep_goes_to_set_suspend_state(self, backend: RecordingPowerBackend) -> None:
        result = perform(PowerRequest(operation=PowerOperation.SLEEP))
        assert result.ok
        assert backend.calls == [PowerCall(kind="suspend", hibernate=False)]

    def test_hibernate_asks_for_hibernation(self, backend: RecordingPowerBackend) -> None:
        perform(PowerRequest(operation=PowerOperation.HIBERNATE))
        assert backend.calls == [PowerCall(kind="suspend", hibernate=True)]

    def test_force_is_passed_through(self, backend: RecordingPowerBackend) -> None:
        perform(PowerRequest(operation=PowerOperation.SLEEP, force=True))
        assert backend.calls[0].force is True

    @pytest.mark.parametrize(
        ("operation", "flags"),
        [
            (PowerOperation.LOGOFF, winapi.EWX_LOGOFF),
            (PowerOperation.REBOOT, winapi.EWX_REBOOT),
            (PowerOperation.SHUTDOWN, winapi.EWX_SHUTDOWN | winapi.EWX_POWEROFF),
        ],
    )
    def test_session_end_flags(
        self, backend: RecordingPowerBackend, operation: PowerOperation, flags: int
    ) -> None:
        perform(PowerRequest(operation=operation))
        assert backend.calls == [PowerCall(kind="exit_windows", flags=flags)]

    def test_hibernation_disabled_is_refused_before_anything_happens(
        self, backend: RecordingPowerBackend
    ) -> None:
        backend.hibernation = False
        with pytest.raises(ActionUnavailable) as info:
            perform(PowerRequest(operation=PowerOperation.HIBERNATE))
        assert "Гибернация отключена" in info.value.user_message
        assert "powercfg" in info.value.user_message
        assert backend.calls == []

    def test_sleep_is_unaffected_by_disabled_hibernation(
        self, backend: RecordingPowerBackend
    ) -> None:
        backend.hibernation = False
        assert perform(PowerRequest(operation=PowerOperation.SLEEP)).ok


class TestScheduled:
    def test_shutdown_is_handed_to_windows(self, backend: RecordingPowerBackend) -> None:
        result = perform(PowerRequest(operation=PowerOperation.SHUTDOWN, delay_s=300))
        assert result.ok
        assert "через 5 минут" in result.message_ru
        assert "отмени" in result.message_ru
        assert backend.calls == [PowerCall(kind="schedule", delay_s=300, reboot=False)]

    def test_reboot_is_scheduled_as_a_reboot(self, backend: RecordingPowerBackend) -> None:
        perform(PowerRequest(operation=PowerOperation.REBOOT, delay_s=60, message="ремонт"))
        assert backend.calls == [
            PowerCall(kind="schedule", delay_s=60, reboot=True, message="ремонт")
        ]

    def test_scheduling_returns_without_waiting(self, backend: RecordingPowerBackend) -> None:
        started = time.monotonic()
        perform(PowerRequest(operation=PowerOperation.SHUTDOWN, delay_s=MAX_DELAY_S))
        assert time.monotonic() - started < 1.0
        assert backend.calls[0].delay_s == MAX_DELAY_S

    def test_cancel_takes_back_a_scheduled_shutdown(self, backend: RecordingPowerBackend) -> None:
        perform(PowerRequest(operation=PowerOperation.SHUTDOWN, delay_s=300))
        assert "Отменила" in cancel_pending()
        assert [call.kind for call in backend.calls] == ["schedule", "abort"]

    def test_cancel_with_nothing_pending_says_so(self, backend: RecordingPowerBackend) -> None:
        assert cancel_pending() == ""
        assert [call.kind for call in backend.calls] == ["abort"]


class TestDelayedLocally:
    """A delayed sleep or log-off, which Ayris waits out itself.

    Windows has no scheduled form of ``SetSuspendState``, so these run on the
    worker thread the registry handed the action. The tests drive that from a
    second thread, which is also how the real cancellation arrives.
    """

    def _wait_for_pending(self, timeout_s: float = 5.0) -> PowerRequest:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            request = pending_delay()
            if request is not None:
                return request
            time.sleep(0.01)
        message = "отложенная операция так и не зарегистрировалась"
        raise AssertionError(message)

    def test_cancel_stops_a_delayed_sleep(self, backend: RecordingPowerBackend) -> None:
        outcome: list[bool] = []

        def run() -> None:
            outcome.append(perform(PowerRequest(operation=PowerOperation.SLEEP, delay_s=60)).ok)

        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert self._wait_for_pending().operation is PowerOperation.SLEEP
            assert "Отменила" in cancel_pending()
        finally:
            worker.join(10)
        assert outcome == [False]
        # The machine was never touched: only the cancellation reached the backend.
        assert [call.kind for call in backend.calls] == ["abort"]

    def test_delayed_sleep_runs_when_it_is_not_cancelled(
        self, backend: RecordingPowerBackend
    ) -> None:
        # One second is long enough to prove the wait happened and short enough
        # to keep the suite quick; the granularity of the wait is 250 ms.
        started = time.monotonic()
        result = perform(PowerRequest(operation=PowerOperation.SLEEP, delay_s=1))
        assert result.ok
        assert time.monotonic() - started >= 0.5
        assert backend.calls == [PowerCall(kind="suspend", hibernate=False)]

    def test_nothing_is_pending_afterwards(self, backend: RecordingPowerBackend) -> None:
        del backend
        perform(PowerRequest(operation=PowerOperation.SLEEP, delay_s=1))
        assert pending_delay() is None

    def test_a_second_request_supersedes_the_first(self, backend: RecordingPowerBackend) -> None:
        # Two delays cannot both be waiting: the machine has one power state, and
        # a forgotten «усни через час» must not fire an hour after «усни через
        # секунду» already did.
        first: list[bool] = []

        def run() -> None:
            first.append(perform(PowerRequest(operation=PowerOperation.LOGOFF, delay_s=60)).ok)

        worker = threading.Thread(target=run)
        worker.start()
        try:
            self._wait_for_pending()
            assert perform(PowerRequest(operation=PowerOperation.SLEEP, delay_s=1)).ok
        finally:
            worker.join(10)
        assert first == [False]
        assert [call.kind for call in backend.calls] == ["suspend"]


# --------------------------------------------------------------------------- #
# the actions themselves
# --------------------------------------------------------------------------- #


class TestParams:
    def test_operation_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PowerAction.Params()

    def test_unknown_operation_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowerAction.Params(operation="destroy")

    @pytest.mark.parametrize("value", ["sleep", "hibernate", "logoff", "reboot", "shutdown"])
    def test_every_documented_value_is_accepted(self, value: str) -> None:
        assert PowerAction.Params(operation=value).operation == value

    def test_negative_delay_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowerAction.Params(operation="sleep", delay_s=-1)

    def test_absurd_delay_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowerAction.Params(operation="sleep", delay_s=MAX_DELAY_S + 1)

    def test_the_longest_allowed_delay_is_a_day(self) -> None:
        assert PowerAction.Params(operation="shutdown", delay_s=MAX_DELAY_S).delay_s == 86400

    def test_defaults_are_immediate_and_polite(self) -> None:
        params = PowerAction.Params(operation="sleep")
        assert params.delay_s == 0
        assert params.force is False
        assert params.message == ""


class TestMetadata:
    def test_power_action_is_dangerous(self) -> None:
        assert PowerAction.meta.is_dangerous is True

    def test_locking_is_not_dangerous(self) -> None:
        # Nothing is lost, and a prompt would defeat saying it while walking away.
        assert LockWorkstation.meta.is_dangerous is False

    def test_cancelling_is_not_dangerous(self) -> None:
        assert CancelPowerAction.meta.is_dangerous is False

    def test_power_action_has_no_timeout(self) -> None:
        # A delayed sleep waits out its delay on the worker thread.
        assert PowerAction.meta.timeout_ms == 0

    def test_nothing_requires_admin(self) -> None:
        # SeShutdownPrivilege is held by ordinary users; elevation is not the gate.
        assert PowerAction.meta.require_admin is False
        assert LockWorkstation.meta.require_admin is False


class TestActions:
    def test_lock_reaches_the_backend(
        self, registry: ActionRegistry, backend: RecordingPowerBackend
    ) -> None:
        assert registry.execute("LockWorkstation").ok
        assert backend.calls == [PowerCall(kind="lock")]

    def test_cancel_action_reports_when_there_is_nothing_to_cancel(
        self, registry: ActionRegistry
    ) -> None:
        result = registry.execute("CancelPowerAction")
        assert not result.ok
        assert "Нечего отменять" in result.message_ru

    def test_cancel_action_takes_back_a_schedule(
        self, registry: ActionRegistry, backend: RecordingPowerBackend
    ) -> None:
        perform(PowerRequest(operation=PowerOperation.SHUTDOWN, delay_s=600))
        assert registry.execute("CancelPowerAction").ok
        assert [call.kind for call in backend.calls] == ["schedule", "abort"]


class TestConfirmation:
    """The reason the registry asks before it acts, tested from the outside."""

    def test_without_a_mechanism_nothing_happens(
        self, registry: ActionRegistry, backend: RecordingPowerBackend
    ) -> None:
        with pytest.raises(ActionNotConfirmed):
            registry.execute("PowerAction", {"operation": "shutdown"})
        assert backend.calls == []

    def test_a_refusal_stops_the_action(self, backend: RecordingPowerBackend) -> None:
        instance = ActionRegistry(audit_enabled=lambda: False, confirm=lambda _request: False)
        instance.add(PowerAction)
        try:
            with pytest.raises(ActionNotConfirmed) as info:
                instance.execute("PowerAction", {"operation": "reboot"})
        finally:
            instance.shutdown()
        assert "подтверждения" in info.value.user_message
        assert backend.calls == []

    def test_a_confirmation_lets_it_through(self, backend: RecordingPowerBackend) -> None:
        instance = ActionRegistry(audit_enabled=lambda: False, confirm=lambda _request: True)
        instance.add(PowerAction)
        try:
            assert instance.execute("PowerAction", {"operation": "reboot"}).ok
        finally:
            instance.shutdown()
        assert backend.calls == [PowerCall(kind="exit_windows", flags=winapi.EWX_REBOOT)]

    def test_the_prompt_names_the_action(self, backend: RecordingPowerBackend) -> None:
        del backend
        asked: list[str] = []
        instance = ActionRegistry(
            audit_enabled=lambda: False,
            confirm=lambda request: asked.append(request.question_ru) is None,
        )
        instance.add(PowerAction)
        try:
            instance.execute("PowerAction", {"operation": "sleep"})
        finally:
            instance.shutdown()
        assert asked == ["Подтвердите: «Питание»."]

    def test_locking_is_not_asked_about(
        self, registry: ActionRegistry, backend: RecordingPowerBackend
    ) -> None:
        # No confirmation mechanism is installed on this registry, and the lock
        # still runs — that is what is_dangerous=False has to mean.
        assert registry.execute("LockWorkstation").ok
        assert backend.calls == [PowerCall(kind="lock")]


class TestDefaultBackend:
    def test_the_default_records_rather_than_acts(self) -> None:
        set_power_backend(None)
        assert isinstance(get_power_backend(), RecordingPowerBackend)

    def test_the_real_backend_is_only_reached_by_name(self) -> None:
        # set_power_backend(None) means «record», not «do it for real». A build
        # that forgets to install the real backend must be inert, not lethal.
        set_power_backend(None)
        assert not isinstance(get_power_backend(), WinApiPowerBackend)


@pytest.mark.hardware
@pytest.mark.skipif(sys.platform != "win32", reason="powercfg is Windows-only")
class TestRealPowercfg:
    """The one thing the real backend is allowed to do here: read the policy.

    ``powercfg /a`` changes nothing, but it is still a live Windows call, so it is
    marked ``hardware`` and never runs in CI.
    """

    def test_powercfg_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        assert isinstance(WinApiPowerBackend().hibernation_available(), bool)
