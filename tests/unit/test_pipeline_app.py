"""Task 47 wiring: workers as a lifecycle component and the text pipeline.

These check the composition seams :func:`~ayris.workers.manager.install_workers`
and :func:`~ayris.core.pipeline_app.install_pipeline` add to the application —
without spawning a real worker process (that needs models and a microphone) and
without a full :class:`~ayris.core.app.AyrisApp` (there is no fixture for a
started one). A light duck-typed application carries exactly the accessors the two
functions read, over a real in-memory database, event bus and state machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ayris.core.app import Component, LifecycleStage
from ayris.core.config import RestartScope, Settings
from ayris.core.database import Database, reset_database
from ayris.core.events import EventBus, IntentMatched
from ayris.core.models import Command
from ayris.core.pipeline_app import install_pipeline
from ayris.core.repositories import Repositories
from ayris.core.state import StateMachine
from ayris.workers.manager import WorkerManager, install_workers

pytestmark = pytest.mark.unit


@dataclass
class _FakeApp:
    """Only what the two installers touch on :class:`AyrisApp`."""

    bus: EventBus
    state: StateMachine
    settings: Settings
    repositories: Repositories
    profile: object
    paths: object
    components: list[Component] = field(default_factory=list)
    restart_handlers: dict[RestartScope, list[object]] = field(default_factory=dict)

    def add_component(self, component: Component) -> None:
        self.components.append(component)

    def register_restart_handler(self, scope: RestartScope, handler: object) -> object:
        self.restart_handlers.setdefault(scope, []).append(handler)
        return lambda: None


@dataclass
class _Paths:
    logs_dir: Path


@pytest.fixture
def bus() -> Iterator[EventBus]:
    instance = EventBus(thread_id=None)
    yield instance
    instance.clear()


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    instance = Database.open(tmp_path / "ayris.db")
    yield instance
    instance.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    return Repositories(database)


@pytest.fixture
def app(bus: EventBus, repos: Repositories, tmp_path: Path) -> _FakeApp:
    profile = repos.profiles.create("Тест", activate=True)
    return _FakeApp(
        bus=bus,
        state=StateMachine(bus),
        settings=Settings.model_validate({}),
        repositories=repos,
        profile=profile,
        paths=_Paths(logs_dir=tmp_path / "logs"),
    )


# --------------------------------------------------------------------------- #
# workers as a lifecycle component
# --------------------------------------------------------------------------- #


def test_install_workers_registers_the_lifecycle_component(app: _FakeApp) -> None:
    manager = install_workers(app)
    try:
        assert isinstance(manager, WorkerManager)
        workers = [c for c in app.components if c.stage is LifecycleStage.WORKERS]
        assert len(workers) == 1
        component = workers[0]
        assert component.start is not None
        assert component.stop is not None
        assert component.kill is not None
    finally:
        manager.shutdown()


def test_install_workers_registers_a_handler_per_restart_scope(app: _FakeApp) -> None:
    manager = install_workers(app)
    try:
        for scope in (
            RestartScope.AUDIO,
            RestartScope.WAKE,
            RestartScope.STT,
            RestartScope.TTS,
            RestartScope.LLM,
        ):
            assert app.restart_handlers.get(scope)
    finally:
        manager.shutdown()


# --------------------------------------------------------------------------- #
# the text pipeline
# --------------------------------------------------------------------------- #


def _command_with_voice(repos: Repositories, profile_id: int, phrase: str) -> int:
    command = repos.commands.create(
        Command(name="Открыть блокнот", profile_id=profile_id, actions=())
    )
    assert command.id is not None
    repos.triggers.add_voice(command.id, phrase)
    return command.id


def test_pipeline_matches_a_typed_command_from_the_library(app: _FakeApp) -> None:
    phrase = "открой блокнот"
    command_id = _command_with_voice(app.repositories, app.profile.id, phrase)  # type: ignore[attr-defined]
    matched: list[IntentMatched] = []
    app.bus.subscribe(IntentMatched, matched.append, weak=False)

    pipeline = install_pipeline(app)
    stop = app.components[-1].stop
    try:
        result = pipeline.run_text(phrase)
        # With no in-pipeline runner the match is handed to the dispatcher, so the
        # pass reports success and the command goes out on the bus for it to run.
        assert result.ok
        assert len(matched) == 1
        assert matched[0].command_id == command_id
    finally:
        if stop is not None:
            stop()


def test_pipeline_rebuilds_its_index_when_commands_change(app: _FakeApp) -> None:
    from ayris.core.events import CommandsChanged

    pipeline = install_pipeline(app)
    stop = app.components[-1].stop
    matched: list[IntentMatched] = []
    app.bus.subscribe(IntentMatched, matched.append, weak=False)
    try:
        phrase = "открой калькулятор"
        # Nothing in the library yet: the phrase matches no command, so nothing
        # goes out as IntentMatched (whatever the NLU mode does with a miss).
        pipeline.run_text(phrase)
        assert not matched

        command_id = _command_with_voice(app.repositories, app.profile.id, phrase)  # type: ignore[attr-defined]
        app.bus.publish(CommandsChanged(command_id=command_id))

        assert pipeline.run_text(phrase).ok
        assert matched and matched[-1].command_id == command_id
    finally:
        if stop is not None:
            stop()


def test_installed_pipeline_stop_unsubscribes(app: _FakeApp) -> None:
    _command_with_voice(app.repositories, app.profile.id, "открой блокнот")  # type: ignore[attr-defined]
    install_pipeline(app)
    stop = app.components[-1].stop
    assert stop is not None
    stop()
    # After stop the index no longer follows the bus; a change must not raise.
    from ayris.core.events import CommandsChanged

    app.bus.publish(CommandsChanged(command_id=1))
