"""Tests for the action registry: registration, validation, timeouts, audit.

Every action here is a dummy declared in this module. Real system actions import
WinAPI, pycaw and WinRT (tasks 20-29), so testing the registry against them would
make these tests both slow and platform-dependent for no gain: what is under test
is the machinery around an action, not any particular action.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic import Field, ValidationError

from ayris.actions.base import (
    DEFAULT_TIMEOUT_MS,
    SECRET_MASK,
    Action,
    ActionCategory,
    ActionMeta,
    ActionParams,
    FieldKind,
    build_schema,
    mask_params,
    secret_fields,
)
from ayris.actions.registry import (
    SYSTEM_PACKAGE,
    ActionRegistry,
    RegisteredAction,
    register,
    registered_actions,
)
from ayris.actions.result import ActionResult
from ayris.core.database import Database, reset_database
from ayris.core.errors import (
    ActionError,
    ActionNotFound,
    ActionParamsInvalid,
    ActionRequiresAdmin,
    ActionTimeout,
    ActionUnavailable,
    AudioError,
    ParamProblem,
)
from ayris.core.events import ActionFailed, ActionFinished, ActionStarted, Event, EventBus
from ayris.core.models import ExecutionResult
from ayris.core.repositories import Repositories

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# dummy actions
# --------------------------------------------------------------------------- #


class Echo(Action):
    """Returns its own argument. The simplest useful shape."""

    meta = ActionMeta(
        name="Echo",
        category=ActionCategory.LOGIC,
        title_ru="Эхо",
        description_ru="Повторяет переданный текст",
    )

    class Params(ActionParams):
        text: str = ""
        times: int = Field(default=1, ge=1, le=10)

    def run(self, params: Echo.Params) -> ActionResult[str]:
        value = params.text * params.times
        return ActionResult.done(f"Сказала: {value}", value=value, detail="echoed")


class Counter(Action):
    """Counts its calls, so a test can prove the action really ran."""

    meta = ActionMeta(name="Counter", category=ActionCategory.LOGIC, title_ru="Счётчик")

    def __init__(self) -> None:
        self.calls = 0

    def run(self, params: ActionParams) -> ActionResult[int]:
        self.calls += 1
        return ActionResult.done(value=self.calls)


class Failing(Action):
    """Raises something that is not an ActionError."""

    meta = ActionMeta(name="Failing", category=ActionCategory.LOGIC, title_ru="Падает")

    def run(self, params: ActionParams) -> ActionResult[None]:
        raise RuntimeError("boom")


class Refusing(Action):
    """Returns a handled failure instead of raising."""

    meta = ActionMeta(name="Refusing", category=ActionCategory.WINDOWS, title_ru="Отказывает")

    def run(self, params: ActionParams) -> ActionResult[None]:
        return ActionResult.failed("Окно не найдено.", detail="no window matched")


class Raising(Action):
    """Raises a typed action error, the way a real action reports a failure."""

    meta = ActionMeta(name="Raising", category=ActionCategory.AUDIO, title_ru="Бросает")

    def run(self, params: ActionParams) -> ActionResult[None]:
        raise ActionUnavailable("no audio endpoint", user_message="Нет устройства вывода.")


class Elevated(Action):
    """Needs administrator rights."""

    meta = ActionMeta(
        name="Elevated",
        category=ActionCategory.SYSTEM,
        title_ru="Только для админа",
        require_admin=True,
        is_dangerous=True,
    )

    def run(self, params: ActionParams) -> ActionResult[None]:
        return ActionResult.done("Готово.")


class Wedging(Action):
    """Blocks until a test releases it. Stands in for a hung WinAPI call."""

    meta = ActionMeta(
        name="Wedging",
        category=ActionCategory.WINDOWS,
        title_ru="Зависает",
        timeout_ms=30,
    )

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()

    def run(self, params: ActionParams) -> ActionResult[None]:
        self.entered.set()
        self.release.wait(30)
        return ActionResult.done("Отпустили.")


class SlowAsync(Action):
    """Async action that outlives its timeout."""

    meta = ActionMeta(
        name="SlowAsync",
        category=ActionCategory.WEB,
        title_ru="Медленная сеть",
        timeout_ms=20,
    )

    async def arun(self, params: ActionParams) -> ActionResult[None]:
        await asyncio.sleep(5)
        return ActionResult.done("Не должно случиться.")


class AsyncOnly(Action):
    """Implements only the async path."""

    meta = ActionMeta(name="AsyncOnly", category=ActionCategory.WEB, title_ru="Только async")

    async def arun(self, params: ActionParams) -> ActionResult[str]:
        await asyncio.sleep(0)
        return ActionResult.done(value="async")


class WithSecret(Action):
    """Takes a credential, so masking has something to hide."""

    meta = ActionMeta(name="WithSecret", category=ActionCategory.WEB, title_ru="С секретом")

    class Params(ActionParams):
        url: str
        token: str = Field(default="", json_schema_extra={"secret": True})

    def run(self, params: WithSecret.Params) -> ActionResult[str]:
        return ActionResult.done(value=params.token)


class Undoable(Action):
    """Remembers what it did and can put it back."""

    meta = ActionMeta(
        name="Undoable",
        category=ActionCategory.AUDIO,
        title_ru="С отменой",
        supports_undo=True,
    )

    class Params(ActionParams):
        level: int = Field(default=50, ge=0, le=100)

    def __init__(self) -> None:
        self.level = 10

    def run(self, params: Undoable.Params) -> ActionResult[int]:
        previous = self.level
        self.level = params.level
        return ActionResult.done(value=self.level, undo_token=str(previous))

    def undo(self, token: str) -> ActionResult[int]:
        self.level = int(token)
        return ActionResult.done("Вернула.", value=self.level)


class Rich(Action):
    """Every parameter shape the editor has to render."""

    meta = ActionMeta(
        name="Rich",
        category=ActionCategory.DISPLAY,
        title_ru="Богатые параметры",
        description_ru="Для проверки интроспекции",
        timeout_ms=0,
    )

    class Params(ActionParams):
        title: str = Field(description="Заголовок окна", max_length=80)
        percent: int = Field(default=50, ge=0, le=100, json_schema_extra={"unit_ru": "%"})
        ratio: float = Field(default=1.5, gt=0.0, lt=10.0)
        enabled: bool = True
        mode: Literal["window", "fullscreen"] = Field(
            default="window",
            json_schema_extra={"choices_ru": {"window": "В окне", "fullscreen": "Во весь экран"}},
        )
        tags: list[str] = Field(default_factory=list)
        note: str | None = Field(default=None, json_schema_extra={"multiline": True})
        password: str = Field(default="", json_schema_extra={"secret": True})

    def run(self, params: Rich.Params) -> ActionResult[None]:
        return ActionResult.done()


ALL_DUMMIES: tuple[type[Action], ...] = (
    Echo,
    Counter,
    Failing,
    Refusing,
    Raising,
    Elevated,
    Wedging,
    SlowAsync,
    AsyncOnly,
    WithSecret,
    Undoable,
    Rich,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


class Recorder:
    """Collects published events in order."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    @property
    def names(self) -> list[str]:
        return [event.name for event in self.events]

    def of(self, kind: type[Event]) -> list[Any]:
        return [event for event in self.events if isinstance(event, kind)]


@pytest.fixture
def bus() -> Iterator[EventBus]:
    instance = EventBus(thread_id=None)
    yield instance
    instance.clear()


@pytest.fixture
def recorder(bus: EventBus) -> Recorder:
    listener = Recorder()
    for kind in (ActionStarted, ActionFinished, ActionFailed):
        bus.subscribe(kind, listener)
    return listener


@pytest.fixture
def registry(bus: EventBus) -> Iterator[ActionRegistry]:
    instance = ActionRegistry(
        bus=bus,
        audit_enabled=lambda: False,
        is_elevated=lambda: False,
        max_workers=2,
    )
    for action_class in ALL_DUMMIES:
        instance.add(action_class)
    yield instance
    instance.shutdown()


@pytest.fixture
def bare() -> Iterator[ActionRegistry]:
    """Registry with no bus and no audit, for the lookup tests."""
    instance = ActionRegistry(audit_enabled=lambda: False, is_elevated=lambda: False)
    yield instance
    instance.shutdown()


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    instance = Database.open(tmp_path / "ayris.db")
    yield instance
    instance.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    return Repositories(database)


# --------------------------------------------------------------------------- #
# registration and discovery
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_add_returns_the_registered_name(self, bare: ActionRegistry) -> None:
        assert bare.add(Echo) == "Echo"
        assert bare.has("Echo")
        assert "Echo" in bare
        assert len(bare) == 1

    def test_duplicate_name_is_refused(self, bare: ActionRegistry) -> None:
        bare.add(Echo)
        with pytest.raises(ValueError, match="already registered"):
            bare.add(Echo)

    def test_duplicate_name_can_be_replaced_explicitly(self, bare: ActionRegistry) -> None:
        bare.add(Echo)
        bare.add(Echo, replace=True)
        assert len(bare) == 1

    def test_plugin_prefix_keeps_both_actions(self, bare: ActionRegistry) -> None:
        bare.add(Echo)
        assert bare.add(Echo, plugin="demo") == "demo.Echo"
        assert bare.names == ("Echo", "demo.Echo")
        assert bare.get("demo.Echo").meta.plugin == "demo"
        assert bare.get("demo.Echo").meta.short_name == "Echo"
        # The unprefixed instance must not have been renamed along with it.
        assert bare.get("Echo").meta.name == "Echo"

    def test_unknown_name_raises_with_a_russian_message(self, bare: ActionRegistry) -> None:
        with pytest.raises(ActionNotFound) as info:
            bare.get("Nope")
        assert "Nope" in info.value.user_message
        assert info.value.user_message_ru == info.value.user_message

    def test_class_without_meta_is_refused(self, bare: ActionRegistry) -> None:
        class Anonymous(Action):
            def run(self, params: ActionParams) -> ActionResult[None]:
                return ActionResult.done()

        with pytest.raises(TypeError, match="no ActionMeta"):
            bare.add(Anonymous)

    def test_class_without_an_implementation_is_refused(self, bare: ActionRegistry) -> None:
        class Empty(Action):
            meta = ActionMeta(name="Empty", category=ActionCategory.LOGIC, title_ru="Пусто")

        with pytest.raises(TypeError, match="neither run"):
            bare.add(Empty)

    def test_undo_promise_must_be_kept(self, bare: ActionRegistry) -> None:
        class Liar(Action):
            meta = ActionMeta(
                name="Liar",
                category=ActionCategory.LOGIC,
                title_ru="Обещает отмену",
                supports_undo=True,
            )

            def run(self, params: ActionParams) -> ActionResult[None]:
                return ActionResult.done()

        with pytest.raises(TypeError, match="supports_undo"):
            bare.add(Liar)

    def test_non_action_is_refused(self, bare: ActionRegistry) -> None:
        with pytest.raises(TypeError, match="not a subclass"):
            bare.add(str)  # type: ignore[arg-type]

    def test_register_marks_the_class_and_is_idempotent(self) -> None:
        before = len(registered_actions())

        @register
        class Marked(Action):
            meta = ActionMeta(
                name="MarkedForTest", category=ActionCategory.LOGIC, title_ru="Помечено"
            )

            def run(self, params: ActionParams) -> ActionResult[None]:
                return ActionResult.done()

        try:
            entries = registered_actions()
            assert len(entries) == before + 1
            assert any(entry.name == "MarkedForTest" for entry in entries)
            assert register(Marked) is Marked
            assert len(registered_actions()) == before + 1
        finally:
            from ayris.actions import registry as registry_module

            registry_module._MARKED.pop("MarkedForTest", None)

    def test_register_rejects_a_taken_name(self) -> None:
        from ayris.actions import registry as registry_module

        @register
        class First(Action):
            meta = ActionMeta(name="ClashForTest", category=ActionCategory.LOGIC, title_ru="Первое")

            def run(self, params: ActionParams) -> ActionResult[None]:
                return ActionResult.done()

        try:
            with pytest.raises(ValueError, match="already used by"):

                @register
                class Second(Action):
                    meta = ActionMeta(
                        name="ClashForTest", category=ActionCategory.LOGIC, title_ru="Второе"
                    )

                    def run(self, params: ActionParams) -> ActionResult[None]:
                        return ActionResult.done()

        finally:
            registry_module._MARKED.pop("ClashForTest", None)
        assert First.meta.name == "ClashForTest"

    def test_add_all_skips_what_is_already_there(self, bare: ActionRegistry) -> None:
        entries = (RegisteredAction(Echo), RegisteredAction(Counter))
        assert bare.add_all(entries) == 2
        assert bare.add_all(entries) == 0
        assert len(bare) == 2

    def test_discover_survives_a_package_that_imports_nothing(
        self, bare: ActionRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Nothing under actions/system yet (tasks 20-29 fill it), so discovery has
        # to be a no-op rather than a failure.
        with caplog.at_level("WARNING", logger="ayris"):
            gained = bare.discover(SYSTEM_PACKAGE)
        assert gained >= 0
        assert "Traceback" not in caplog.text

    def test_discover_reports_a_missing_package(
        self, bare: ActionRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("ERROR", logger="ayris"):
            assert bare.discover("ayris.actions.nonexistent") == 0
        assert "not importable" in caplog.text

    def test_discover_is_done_once(self, bare: ActionRegistry) -> None:
        first = bare.discover(SYSTEM_PACKAGE)
        second = bare.discover(SYSTEM_PACKAGE)
        assert first == second == 0 or second == 0


# --------------------------------------------------------------------------- #
# lookup for the editor
# --------------------------------------------------------------------------- #


class TestLookup:
    def test_find_without_filters_returns_everything_sorted(self, registry: ActionRegistry) -> None:
        found = registry.find()
        assert len(found) == len(ALL_DUMMIES)
        keys = [(item.meta.category.value, item.meta.name) for item in found]
        assert keys == sorted(keys)

    def test_find_by_category(self, registry: ActionRegistry) -> None:
        found = registry.find(category=ActionCategory.AUDIO)
        assert {item.meta.name for item in found} == {"Raising", "Undoable"}

    def test_find_by_query_matches_name_title_and_description(
        self, registry: ActionRegistry
    ) -> None:
        assert [item.meta.name for item in registry.find(query="echo")] == ["Echo"]
        assert [item.meta.name for item in registry.find(query="ЭХО")] == ["Echo"]
        assert [item.meta.name for item in registry.find(query="повторяет")] == ["Echo"]

    def test_find_by_plugin(self, registry: ActionRegistry) -> None:
        registry.add(Echo, plugin="demo")
        assert [item.meta.name for item in registry.find(plugin="demo")] == ["demo.Echo"]
        assert all(item.meta.plugin == "" for item in registry.find(plugin=""))

    def test_list_categories_only_shows_the_populated_ones(self, registry: ActionRegistry) -> None:
        categories = registry.list_categories()
        assert ActionCategory.LOGIC in categories
        assert ActionCategory.CLIPBOARD not in categories
        # Editor order is the declaration order of the enum, not alphabetical.
        assert categories == [
            category for category in ActionCategory if category in set(categories)
        ]

    def test_iteration_yields_the_instances(self, registry: ActionRegistry) -> None:
        assert len(list(registry)) == len(ALL_DUMMIES)
        assert "Echo" not in list(registry)


# --------------------------------------------------------------------------- #
# parameter validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_valid_params_reach_the_action(self, registry: ActionRegistry) -> None:
        result = registry.execute("Echo", {"text": "ку", "times": 2})
        assert result.ok
        assert result.value == "куку"
        assert result.message_ru == "Сказала: куку"

    def test_missing_params_use_the_defaults(self, registry: ActionRegistry) -> None:
        assert registry.execute("Echo").value == ""
        assert registry.execute("Echo", None).value == ""

    def test_out_of_range_value_names_the_field(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid) as info:
            registry.execute("Echo", {"times": 99})
        error = info.value
        assert error.fields == ("times",)
        assert "велико" in str(error.problems[0])
        assert "le 10" in str(error.problems[0])
        assert "Эхо" in error.user_message

    def test_wrong_type_is_reported_in_russian(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid) as info:
            registry.execute("Echo", {"times": "много"})
        assert info.value.problems[0].message.startswith("нужно целое число")

    def test_unknown_field_is_refused(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid) as info:
            registry.execute("Echo", {"text": "a", "colour": "red"})
        assert info.value.fields == ("colour",)
        assert info.value.problems[0].message == "лишний параметр"

    def test_missing_required_field_is_reported(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid) as info:
            registry.execute("Rich", {})
        assert info.value.fields == ("title",)
        assert info.value.problems[0].message == "не заполнено"

    def test_every_problem_is_listed(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid) as info:
            registry.execute("Rich", {"percent": 500, "ratio": -1.0})
        assert set(info.value.fields) == {"title", "percent", "ratio"}

    def test_a_rejected_call_publishes_nothing(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        with pytest.raises(ActionParamsInvalid):
            registry.execute("Echo", {"times": 0})
        assert recorder.events == []

    def test_params_are_frozen_for_the_action(self) -> None:
        params = Echo.Params(text="a", times=1)
        with pytest.raises(ValidationError):
            params.text = "b"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


class TestExecute:
    def test_the_action_actually_runs(self, registry: ActionRegistry) -> None:
        assert registry.execute("Counter").value == 1
        assert registry.execute("Counter").value == 2

    def test_duration_is_stamped_by_the_registry(self) -> None:
        ticks = iter([100.0, 100.25])
        instance = ActionRegistry(
            audit_enabled=lambda: False, clock=lambda: next(ticks), max_workers=1
        )
        instance.add(Counter)
        try:
            assert instance.execute("Counter").duration_ms == 250
        finally:
            instance.shutdown()

    def test_handled_failure_comes_back_as_a_result(self, registry: ActionRegistry) -> None:
        result = registry.execute("Refusing")
        assert not result.ok
        assert result.message_ru == "Окно не найдено."
        assert result.execution is ExecutionResult.ERROR

    def test_unexpected_exception_becomes_an_action_error(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionError) as info:
            registry.execute("Failing")
        error = info.value
        assert type(error) is ActionError
        assert "RuntimeError" in error.technical
        assert error.user_message == "Не удалось выполнить «Падает»."
        assert isinstance(error.__cause__, RuntimeError)

    def test_typed_action_error_passes_through_untouched(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionUnavailable) as info:
            registry.execute("Raising")
        assert info.value.user_message == "Нет устройства вывода."

    def test_other_ayris_errors_keep_their_russian_message(self, bare: ActionRegistry) -> None:
        class Audio(Action):
            meta = ActionMeta(name="Audio", category=ActionCategory.AUDIO, title_ru="Звук")

            def run(self, params: ActionParams) -> ActionResult[None]:
                raise AudioError("device gone")

        bare.add(Audio)
        with pytest.raises(ActionError) as info:
            bare.execute("Audio")
        assert info.value.user_message == AudioError.default_user_message

    def test_not_implemented_means_unavailable(self, bare: ActionRegistry) -> None:
        class WindowsOnly(Action):
            meta = ActionMeta(
                name="WindowsOnly", category=ActionCategory.SYSTEM, title_ru="Только Windows"
            )

            def run(self, params: ActionParams) -> ActionResult[None]:
                raise NotImplementedError("win32 only")

        bare.add(WindowsOnly)
        with pytest.raises(ActionUnavailable) as info:
            bare.execute("WindowsOnly")
        assert info.value.user_message == "«Только Windows» здесь недоступно."

    def test_permission_error_means_missing_rights(self, bare: ActionRegistry) -> None:
        class Locked(Action):
            meta = ActionMeta(name="Locked", category=ActionCategory.SYSTEM, title_ru="Закрыто")

            def run(self, params: ActionParams) -> ActionResult[None]:
                raise PermissionError(13, "denied")

        bare.add(Locked)
        with pytest.raises(ActionRequiresAdmin):
            bare.execute("Locked")

    def test_keyboard_interrupt_is_not_swallowed(self, bare: ActionRegistry) -> None:
        class Interrupted(Action):
            meta = ActionMeta(name="Interrupted", category=ActionCategory.LOGIC, title_ru="Ctrl-C")

            def run(self, params: ActionParams) -> ActionResult[None]:
                raise KeyboardInterrupt

        bare.add(Interrupted)
        with pytest.raises(KeyboardInterrupt):
            bare.execute("Interrupted")

    def test_sync_action_runs_off_the_calling_thread(self, bare: ActionRegistry) -> None:
        seen: list[int] = []

        class Threaded(Action):
            meta = ActionMeta(name="Threaded", category=ActionCategory.LOGIC, title_ru="В потоке")

            def run(self, params: ActionParams) -> ActionResult[None]:
                seen.append(threading.get_ident())
                return ActionResult.done()

        bare.add(Threaded)
        bare.execute("Threaded")
        assert seen == [seen[0]]
        assert seen[0] != threading.get_ident()

    def test_async_action_is_reachable_from_sync_code(self, registry: ActionRegistry) -> None:
        assert registry.execute("AsyncOnly").value == "async"

    @pytest.mark.asyncio
    async def test_aexecute_runs_an_async_action(self, registry: ActionRegistry) -> None:
        result = await registry.aexecute("AsyncOnly")
        assert result.value == "async"

    @pytest.mark.asyncio
    async def test_aexecute_runs_a_sync_action_in_a_thread(self, registry: ActionRegistry) -> None:
        result = await registry.aexecute("Echo", {"text": "ok"})
        assert result.value == "ok"

    @pytest.mark.asyncio
    async def test_aexecute_validates_the_same_way(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid):
            await registry.aexecute("Echo", {"times": 0})

    @pytest.mark.asyncio
    async def test_async_only_action_refuses_the_sync_path_inside_a_loop(
        self, registry: ActionRegistry
    ) -> None:
        # asyncio.run() inside a running loop would raise RuntimeError; the action
        # has to say what to do instead.
        with pytest.raises(ActionError) as info:
            AsyncOnly().run(ActionParams())
        assert "aexecute" in info.value.technical


# --------------------------------------------------------------------------- #
# rights
# --------------------------------------------------------------------------- #


class TestElevation:
    def test_admin_action_is_refused_without_rights(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        with pytest.raises(ActionRequiresAdmin) as info:
            registry.execute("Elevated")
        assert "администратора" in info.value.user_message
        assert recorder.names == ["ActionFailed"]
        assert recorder.of(ActionFailed)[0].reason == "denied"

    def test_admin_action_runs_when_elevated(self, bus: EventBus) -> None:
        instance = ActionRegistry(bus=bus, audit_enabled=lambda: False, is_elevated=lambda: True)
        instance.add(Elevated)
        try:
            assert instance.execute("Elevated").ok
        finally:
            instance.shutdown()

    def test_broken_elevation_check_denies(self, bare: ActionRegistry) -> None:
        def explode() -> bool:
            raise OSError("no shell32")

        instance = ActionRegistry(audit_enabled=lambda: False, is_elevated=explode)
        instance.add(Elevated)
        try:
            with pytest.raises(ActionRequiresAdmin):
                instance.execute("Elevated")
        finally:
            instance.shutdown()

    @pytest.mark.skipif(sys.platform != "win32", reason="IsUserAnAdmin is Windows-only")
    def test_default_elevation_check_answers_on_windows(self) -> None:
        from ayris.actions.registry import _process_is_elevated

        assert isinstance(_process_is_elevated(), bool)

    @pytest.mark.skipif(sys.platform == "win32", reason="checked separately on Windows")
    def test_default_elevation_check_is_false_elsewhere(self) -> None:
        from ayris.actions.registry import _process_is_elevated

        assert _process_is_elevated() is False


# --------------------------------------------------------------------------- #
# timeouts
# --------------------------------------------------------------------------- #


class TestTimeout:
    def test_sync_timeout_raises_and_marks_the_thread_as_stuck(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        wedging = registry.get("Wedging")
        assert isinstance(wedging, Wedging)
        try:
            assert registry.free_workers == 2
            with pytest.raises(ActionTimeout):
                registry.execute("Wedging")
            assert wedging.entered.is_set()
            # The point of the test: the thread is still inside the action, so it
            # must not be counted as available.
            assert registry.stuck_workers == 1
            assert registry.free_workers == 1
            assert recorder.names == ["ActionStarted", "ActionFailed"]
            assert recorder.of(ActionFailed)[0].reason == "timeout"
            # The remaining worker still takes work.
            assert registry.execute("Counter").ok
        finally:
            wedging.release.set()

    def test_stuck_worker_is_released_when_the_action_finally_returns(
        self, registry: ActionRegistry
    ) -> None:
        wedging = registry.get("Wedging")
        assert isinstance(wedging, Wedging)
        with pytest.raises(ActionTimeout):
            registry.execute("Wedging")
        assert registry.stuck_workers == 1
        wedging.release.set()
        deadline = time.monotonic() + 5
        while registry.stuck_workers and time.monotonic() < deadline:
            time.sleep(0.01)
        assert registry.stuck_workers == 0
        assert registry.free_workers == 2

    def test_exhausted_pool_fails_fast(self, registry: ActionRegistry) -> None:
        wedging = registry.get("Wedging")
        assert isinstance(wedging, Wedging)
        try:
            for _ in range(2):
                wedging.entered.clear()
                with pytest.raises(ActionTimeout):
                    registry.execute("Wedging")
            assert registry.free_workers == 0
            with pytest.raises(ActionUnavailable) as info:
                registry.execute("Counter")
            assert "Перезапустите" in info.value.user_message
        finally:
            wedging.release.set()

    def test_unlimited_timeout_is_allowed(self, registry: ActionRegistry) -> None:
        assert registry.get("Rich").meta.timeout_s is None
        assert registry.execute("Rich", {"title": "окно"}).ok

    @pytest.mark.asyncio
    async def test_async_timeout_raises(self, registry: ActionRegistry, recorder: Recorder) -> None:
        with pytest.raises(ActionTimeout):
            await registry.aexecute("SlowAsync")
        assert recorder.names == ["ActionStarted", "ActionFailed"]

    @pytest.mark.asyncio
    async def test_async_timeout_cancels_the_coroutine(self, bare: ActionRegistry) -> None:
        cancelled = threading.Event()

        class Cancellable(Action):
            meta = ActionMeta(
                name="Cancellable",
                category=ActionCategory.WEB,
                title_ru="Отменяемая",
                timeout_ms=20,
            )

            async def arun(self, params: ActionParams) -> ActionResult[None]:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return ActionResult.done()

        bare.add(Cancellable)
        with pytest.raises(ActionTimeout):
            await bare.aexecute("Cancellable")
        assert cancelled.is_set()


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #


class TestEvents:
    def test_success_publishes_started_then_finished(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        registry.execute("Echo", {"text": "да"}, request_id="req-1", command_id=7)
        assert recorder.names == ["ActionStarted", "ActionFinished"]
        started = recorder.of(ActionStarted)[0]
        assert started.action == "Echo"
        assert started.request_id == "req-1"
        assert started.command_id == 7
        assert started.params == {"text": "да", "times": 1}
        finished = recorder.of(ActionFinished)[0]
        assert finished.ok is True
        assert finished.result == "Сказала: да"
        assert finished.request_id == "req-1"
        assert finished.duration_ms >= 0

    def test_handled_failure_still_finishes(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        registry.execute("Refusing")
        assert recorder.names == ["ActionStarted", "ActionFinished"]
        assert recorder.of(ActionFinished)[0].ok is False

    def test_raised_failure_publishes_failed(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        with pytest.raises(ActionUnavailable):
            registry.execute("Raising", request_id="req-2")
        assert recorder.names == ["ActionStarted", "ActionFailed"]
        failed = recorder.of(ActionFailed)[0]
        assert failed.action == "Raising"
        assert failed.error == "no audio endpoint"
        assert failed.user_message == "Нет устройства вывода."
        assert failed.request_id == "req-2"
        assert failed.reason == "error"

    def test_secrets_never_reach_the_bus(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        registry.execute("WithSecret", {"url": "https://x", "token": "sk-live-123"})
        started = recorder.of(ActionStarted)[0]
        assert started.params == {"url": "https://x", "token": SECRET_MASK}
        assert "sk-live-123" not in repr(recorder.events)

    def test_a_broken_subscriber_does_not_break_the_action(
        self, bus: EventBus, registry: ActionRegistry
    ) -> None:
        def bad(event: Event) -> None:
            raise AudioError("subscriber blew up")

        bus.subscribe(ActionStarted, bad)
        assert registry.execute("Counter").ok

    def test_registry_without_a_bus_still_runs(self, bare: ActionRegistry) -> None:
        bare.add(Counter)
        assert bare.execute("Counter").ok


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #


class TestAudit:
    def test_success_is_journalled(self, repos: Repositories) -> None:
        instance = ActionRegistry(
            audit=repos.audit, audit_enabled=lambda: True, is_elevated=lambda: False
        )
        instance.add(Echo)
        try:
            instance.execute("Echo", {"text": "ok"})
        finally:
            instance.shutdown()
        entries = repos.audit.recent(10)
        assert len(entries) == 1
        assert entries[0].command_name == "Echo"
        assert entries[0].result is ExecutionResult.OK
        assert entries[0].params == {"text": "ok", "times": 1}
        assert entries[0].elevated is False

    def test_secrets_are_masked_in_the_journal(self, repos: Repositories) -> None:
        instance = ActionRegistry(audit=repos.audit, audit_enabled=lambda: True)
        instance.add(WithSecret)
        try:
            instance.execute("WithSecret", {"url": "https://x", "token": "sk-live-999"})
        finally:
            instance.shutdown()
        entry = repos.audit.recent(1)[0]
        assert entry.params["token"] == SECRET_MASK
        assert "sk-live-999" not in str(entry.params)

    def test_failures_are_journalled_with_their_outcome(self, repos: Repositories) -> None:
        instance = ActionRegistry(
            audit=repos.audit, audit_enabled=lambda: True, is_elevated=lambda: False
        )
        for action_class in (Failing, Elevated, Wedging):
            instance.add(action_class)
        wedging = instance.get("Wedging")
        assert isinstance(wedging, Wedging)
        try:
            with pytest.raises(ActionError):
                instance.execute("Failing")
            with pytest.raises(ActionRequiresAdmin):
                instance.execute("Elevated")
            with pytest.raises(ActionTimeout):
                instance.execute("Wedging")
        finally:
            wedging.release.set()
            instance.shutdown()
        outcomes = {entry.command_name: entry.result for entry in repos.audit.recent(10)}
        assert outcomes == {
            "Failing": ExecutionResult.ERROR,
            "Elevated": ExecutionResult.DENIED,
            "Wedging": ExecutionResult.TIMEOUT,
        }

    def test_admin_flag_is_recorded(self, repos: Repositories) -> None:
        instance = ActionRegistry(
            audit=repos.audit, audit_enabled=lambda: True, is_elevated=lambda: True
        )
        instance.add(Elevated)
        try:
            instance.execute("Elevated")
        finally:
            instance.shutdown()
        entry = repos.audit.recent(1)[0]
        assert entry.require_admin is True
        assert entry.elevated is True

    def test_disabled_audit_writes_nothing(self, repos: Repositories) -> None:
        instance = ActionRegistry(audit=repos.audit, audit_enabled=lambda: False)
        instance.add(Echo)
        try:
            instance.execute("Echo")
        finally:
            instance.shutdown()
        assert repos.audit.count() == 0

    def test_audit_can_be_switched_off_while_running(self, repos: Repositories) -> None:
        enabled = True
        instance = ActionRegistry(audit=repos.audit, audit_enabled=lambda: enabled)
        instance.add(Echo)
        try:
            instance.execute("Echo")
            enabled = False
            instance.execute("Echo")
        finally:
            instance.shutdown()
        assert repos.audit.count() == 1

    def test_a_broken_journal_does_not_fail_the_action(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class Broken:
            def add(self, entry: Any) -> Any:
                raise AudioError("disk on fire")

        instance = ActionRegistry(audit=Broken(), audit_enabled=lambda: True)  # type: ignore[arg-type]
        instance.add(Counter)
        try:
            with caplog.at_level("ERROR", logger="ayris"):
                assert instance.execute("Counter").ok
        finally:
            instance.shutdown()
        assert "audit write failed" in caplog.text


# --------------------------------------------------------------------------- #
# undo
# --------------------------------------------------------------------------- #


class TestUndo:
    def test_undo_token_round_trip(self, registry: ActionRegistry) -> None:
        undoable = registry.get("Undoable")
        assert isinstance(undoable, Undoable)
        result = registry.execute("Undoable", {"level": 80})
        assert result.undoable
        assert result.undo_token == "10"
        assert undoable.level == 80
        back = registry.undo("Undoable", result.undo_token or "")
        assert back.ok
        assert undoable.level == 10

    def test_undo_is_refused_for_an_action_that_cannot(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionUnavailable) as info:
            registry.undo("Echo", "whatever")
        assert "нельзя отменить" in info.value.user_message

    def test_failing_undo_is_wrapped(self, bare: ActionRegistry) -> None:
        class BadUndo(Action):
            meta = ActionMeta(
                name="BadUndo",
                category=ActionCategory.LOGIC,
                title_ru="Плохая отмена",
                supports_undo=True,
            )

            def run(self, params: ActionParams) -> ActionResult[None]:
                return ActionResult.done(undo_token="x")

            def undo(self, token: str) -> ActionResult[None]:
                raise RuntimeError(f"cannot undo {token}")

        bare.add(BadUndo)
        with pytest.raises(ActionError) as info:
            bare.undo("BadUndo", "x")
        assert "отмена не удалась" in info.value.user_message

    def test_undo_publishes_a_finished_event(
        self, registry: ActionRegistry, recorder: Recorder
    ) -> None:
        result = registry.execute("Undoable", {"level": 30})
        recorder.events.clear()
        registry.undo("Undoable", result.undo_token or "")
        assert recorder.names == ["ActionFinished"]


# --------------------------------------------------------------------------- #
# introspection for the macro editor
# --------------------------------------------------------------------------- #


class TestIntrospection:
    def test_describe_copies_the_metadata(self, registry: ActionRegistry) -> None:
        schema = registry.describe("Elevated")
        assert schema.name == "Elevated"
        assert schema.category is ActionCategory.SYSTEM
        assert schema.title_ru == "Только для админа"
        assert schema.category_title_ru == ActionCategory.SYSTEM.title_ru
        assert schema.require_admin is True
        assert schema.is_dangerous is True
        assert schema.supports_undo is False
        assert schema.timeout_ms == DEFAULT_TIMEOUT_MS
        assert schema.fields == ()

    def test_describe_marks_an_async_action(self, registry: ActionRegistry) -> None:
        assert registry.describe("AsyncOnly").is_async is True
        assert registry.describe("Echo").is_async is False

    def test_describe_all_covers_every_action(self, registry: ActionRegistry) -> None:
        schemas = registry.describe_all()
        assert len(schemas) == len(ALL_DUMMIES)
        assert {schema.name for schema in schemas} == {action.meta.name for action in ALL_DUMMIES}

    def test_describe_unknown_action_raises(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionNotFound):
            registry.describe("Nope")

    def test_text_field(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("title")
        assert field is not None
        assert field.kind is FieldKind.TEXT
        assert field.required is True
        assert field.max_length == 80
        # A single short description is the caption, not a hint repeated under it.
        assert field.label_ru == "Заголовок окна"
        assert field.description_ru == ""

    def test_explicit_title_wins_over_the_description(self, bare: ActionRegistry) -> None:
        class Titled(Action):
            meta = ActionMeta(name="Titled", category=ActionCategory.LOGIC, title_ru="С подписью")

            class Params(ActionParams):
                level: int = Field(default=0, title="Громкость", description="От 0 до 100")

            def run(self, params: Titled.Params) -> ActionResult[None]:
                return ActionResult.done()

        bare.add(Titled)
        field = bare.describe("Titled").field_by_name("level")
        assert field is not None
        assert field.label_ru == "Громкость"
        assert field.description_ru == "От 0 до 100"

    def test_long_description_stays_a_hint(self, bare: ActionRegistry) -> None:
        long_text = (
            "Полный путь к исполняемому файлу либо имя программы так, " "как её знает меню Пуск"
        )

        class Wordy(Action):
            meta = ActionMeta(name="Wordy", category=ActionCategory.APPS, title_ru="Многословное")

            class Params(ActionParams):
                target: str = Field(default="", description=long_text)

            def run(self, params: Wordy.Params) -> ActionResult[None]:
                return ActionResult.done()

        bare.add(Wordy)
        field = bare.describe("Wordy").field_by_name("target")
        assert field is not None
        assert field.label_ru == "target"
        assert field.description_ru == long_text

    def test_integer_field_with_range_and_unit(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("percent")
        assert field is not None
        assert field.kind is FieldKind.INTEGER
        assert field.required is False
        assert field.default == 50
        assert (field.minimum, field.maximum) == (0, 100)
        assert field.has_range
        assert field.unit_ru == "%"

    def test_exclusive_bounds_are_folded_in(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("ratio")
        assert field is not None
        assert field.kind is FieldKind.NUMBER
        assert field.minimum == 0.0
        assert field.maximum == 10.0

    def test_boolean_field(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("enabled")
        assert field is not None
        assert field.kind is FieldKind.BOOLEAN
        assert field.default is True

    def test_choice_field_carries_russian_labels(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("mode")
        assert field is not None
        assert field.kind is FieldKind.CHOICE
        assert [(choice.value, choice.label_ru) for choice in field.choices] == [
            ("window", "В окне"),
            ("fullscreen", "Во весь экран"),
        ]

    def test_list_field_knows_its_item_kind(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("tags")
        assert field is not None
        assert field.kind is FieldKind.LIST
        assert field.item_kind is FieldKind.TEXT

    def test_optional_field_is_unwrapped(self, registry: ActionRegistry) -> None:
        field = registry.describe("Rich").field_by_name("note")
        assert field is not None
        assert field.kind is FieldKind.TEXT
        assert field.required is False
        assert field.default is None
        assert field.multiline is True

    def test_secret_field_is_flagged(self, registry: ActionRegistry) -> None:
        schema = registry.describe("Rich")
        field = schema.field_by_name("password")
        assert field is not None
        assert field.secret is True
        assert schema.secret_fields == ("password",)

    def test_unknown_field_returns_none(self, registry: ActionRegistry) -> None:
        assert registry.describe("Rich").field_by_name("absent") is None

    def test_json_schema_is_kept_for_the_macro_format(self, registry: ActionRegistry) -> None:
        schema = registry.describe("Echo")
        assert schema.json_schema["type"] == "object"
        assert set(schema.json_schema["properties"]) == {"text", "times"}

    def test_build_schema_works_on_a_class_too(self) -> None:
        assert build_schema(Echo).name == build_schema(Echo()).name


# --------------------------------------------------------------------------- #
# metadata and masking helpers
# --------------------------------------------------------------------------- #


class TestMeta:
    def test_name_must_look_like_an_action(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ActionMeta(name="run app", category=ActionCategory.APPS, title_ru="Запуск")

    def test_title_is_required(self) -> None:
        with pytest.raises(ValueError, match="needs a Russian title"):
            ActionMeta(name="RunApp", category=ActionCategory.APPS, title_ru="  ")

    def test_negative_timeout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative timeout"):
            ActionMeta(
                name="RunApp", category=ActionCategory.APPS, title_ru="Запуск", timeout_ms=-1
            )

    def test_timeout_seconds(self) -> None:
        meta = ActionMeta(name="RunApp", category=ActionCategory.APPS, title_ru="Запуск")
        assert meta.timeout_s == DEFAULT_TIMEOUT_MS / 1000
        assert meta.with_prefix("").timeout_s == meta.timeout_s

    def test_prefixing_twice_does_not_stack(self) -> None:
        meta = ActionMeta(name="RunApp", category=ActionCategory.APPS, title_ru="Запуск")
        once = meta.with_prefix("demo")
        assert once.name == "demo.RunApp"
        assert once.with_prefix("demo").name == "demo.RunApp"
        assert once.plugin == "demo"

    def test_every_category_has_a_russian_title(self) -> None:
        for category in ActionCategory:
            assert category.title_ru
            assert category.title_ru != category.value

    def test_secret_fields_are_discovered_from_the_model(self) -> None:
        assert secret_fields(WithSecret.Params) == frozenset({"token"})
        assert secret_fields(Echo.Params) == frozenset()

    def test_masking_leaves_other_values_alone(self) -> None:
        masked = mask_params(WithSecret.Params(url="https://x", token="secret"))
        assert masked == {"url": "https://x", "token": SECRET_MASK}

    def test_masking_is_recursive(self) -> None:
        class Outer(ActionParams):
            inner: WithSecret.Params
            name: str = "x"

        masked = mask_params(Outer(inner=WithSecret.Params(url="u", token="t")))
        assert masked["inner"]["token"] == SECRET_MASK

    def test_param_problem_renders_both_ways(self) -> None:
        assert str(ParamProblem(field="times", message="нужно целое число")) == (
            "times: нужно целое число"
        )
        assert str(ParamProblem(field="", message="параметры не подходят")) == (
            "параметры не подходят"
        )

    def test_result_helpers(self) -> None:
        done: ActionResult[int] = ActionResult.done("Готово.", value=5)
        assert (done.ok, done.value, done.execution) == (True, 5, ExecutionResult.OK)
        assert done.with_duration(-5).duration_ms == 0
        failed: ActionResult[int] = ActionResult.failed("Не вышло.")
        assert not failed.ok
        assert failed.detail == "Не вышло."
        from_error: ActionResult[int] = ActionResult.from_error(ActionTimeout("too slow"))
        assert from_error.message_ru == ActionTimeout.default_user_message
        assert from_error.detail == "too slow"
        assert not from_error.undoable


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #


class TestConcurrency:
    def test_two_actions_run_at_the_same_time(self, bare: ActionRegistry) -> None:
        barrier = threading.Barrier(2, timeout=5)

        class Meeting(Action):
            meta = ActionMeta(
                name="Meeting", category=ActionCategory.LOGIC, title_ru="Встреча", timeout_ms=5000
            )

            def run(self, params: ActionParams) -> ActionResult[int]:
                return ActionResult.done(value=barrier.wait())

        bare.add(Meeting)
        results: list[Any] = []

        def runner() -> None:
            results.append(bare.execute("Meeting"))

        threads = [threading.Thread(target=runner) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert len(results) == 2
        assert all(result.ok for result in results)
