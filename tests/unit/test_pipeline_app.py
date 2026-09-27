"""Task 47 wiring: workers as a lifecycle component and the text pipeline.

These check the composition seams :func:`~ayris.workers.manager.install_workers`
and :func:`~ayris.core.pipeline_app.install_pipeline` add to the application —
without spawning a real worker process (that needs models and a microphone) and
without a full :class:`~ayris.core.app.AyrisApp` (there is no fixture for a
started one). A light duck-typed application carries exactly the accessors the two
functions read, over a real in-memory database, event bus and state machine.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ayris.audio.stt.base import STT_SAMPLE_RATE, TranscriptResult
from ayris.core.app import Component, LifecycleStage
from ayris.core.config import RestartScope, Settings, diff_settings
from ayris.core.database import Database, reset_database
from ayris.core.events import (
    ConfigChanged,
    EventBus,
    IntentMatched,
    SpeechEnded,
    WakeWordDetected,
)
from ayris.core.models import Command
from ayris.core.pipeline_app import install_pipeline
from ayris.core.repositories import Repositories
from ayris.core.state import StateMachine
from ayris.workers.manager import WorkerManager, install_workers
from ayris.workers.registry import WorkerKind

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


# --------------------------------------------------------------------------- #
# the voice input path
# --------------------------------------------------------------------------- #


@dataclass
class _FakeManager:
    """A worker supervisor answering ``segment`` and ``transcribe`` from memory.

    Stands in for :class:`~ayris.workers.manager.WorkerManager` so the whole voice
    input path — wake word → phrase → recognition → match — runs without a worker
    process, a model or a microphone. Only :meth:`call_sync` is needed: it is the
    one method the two adapters in :mod:`ayris.core.worker_speech` reach for.
    """

    segment: object = None
    transcript: object = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call_sync(
        self,
        worker: str,
        method: str,
        params: Any = None,
        *,
        timeout: float | None = None,
        audio: bytes | None = None,
        sample_rate: int = STT_SAMPLE_RATE,
        channels: int = 1,
        sample_format: str = "int16",
    ) -> Any:
        self.calls.append((worker, method))
        if worker == WorkerKind.AUDIO.value and method == "segment":
            return self.segment
        if worker == WorkerKind.STT.value and method == "transcribe":
            return self.transcript
        raise AssertionError(f"неожиданный вызов воркера: {worker}.{method}")


def test_a_spoken_phrase_runs_the_same_path_as_a_typed_one(app: _FakeApp) -> None:
    phrase = "открой блокнот"
    command_id = _command_with_voice(app.repositories, app.profile.id, phrase)  # type: ignore[attr-defined]
    manager = _FakeManager(
        segment={
            "available": True,
            "pcm": b"\x01\x02\x03\x04",
            "sample_rate": STT_SAMPLE_RATE,
        },
        transcript=TranscriptResult(text=phrase, confidence=0.0, engine="fake").to_params(),
    )
    matched: list[IntentMatched] = []
    done = threading.Event()

    def on_match(event: IntentMatched) -> None:
        matched.append(event)
        done.set()

    app.bus.subscribe(IntentMatched, on_match, weak=False)

    pipeline = install_pipeline(app, manager)  # type: ignore[arg-type]
    stop = app.components[-1].stop
    try:
        # A wake word opens the session; the finished phrase's PCM is pulled from
        # the audio worker, recognised by the STT worker and matched — the very
        # path a typed command runs, only with a microphone where the keyboard was.
        app.bus.publish(WakeWordDetected(phrase="айрис"))
        assert pipeline.session_id != ""
        app.bus.publish(SpeechEnded(duration_ms=800, reason="silence"))

        assert done.wait(5.0), "произнесённая команда не дошла до IntentMatched"
        assert len(matched) == 1
        assert matched[0].command_id == command_id
        assert (WorkerKind.AUDIO.value, "segment") in manager.calls
        assert (WorkerKind.STT.value, "transcribe") in manager.calls
    finally:
        if stop is not None:
            stop()


def test_voice_loop_follows_the_wake_settings(app: _FakeApp) -> None:
    manager = _FakeManager()
    pipeline = install_pipeline(app, manager)  # type: ignore[arg-type]
    stop = app.components[-1].stop
    try:
        # Default settings allow a spoken activation, so the loop is attached and
        # a wake word opens a session.
        app.bus.publish(WakeWordDetected(phrase="айрис"))
        assert pipeline.session_id != ""
        pipeline.cancel()
        assert pipeline.session_id == ""

        # Wake word off and the microphone on «always» leave nothing that could
        # open a session: the loop detaches and the wake word is now ignored.
        muted = Settings.model_validate(
            {"voice": {"wake": {"enabled": False, "mic_mode": "always"}}}
        )
        app.bus.publish(ConfigChanged(diff=diff_settings(app.settings, muted)))
        app.bus.publish(WakeWordDetected(phrase="айрис"))
        assert pipeline.session_id == ""

        # Turning the wake word back on re-attaches the loop on the fly.
        live = Settings.model_validate({"voice": {"wake": {"enabled": True}}})
        app.bus.publish(ConfigChanged(diff=diff_settings(muted, live)))
        app.bus.publish(WakeWordDetected(phrase="айрис"))
        assert pipeline.session_id != ""
        pipeline.cancel()
    finally:
        if stop is not None:
            stop()


def test_a_text_only_pipeline_ignores_wake_words(app: _FakeApp) -> None:
    # With no worker manager there is nothing to feed the loop, so install_pipeline
    # never attaches: a wake word cannot open a session, only run_text drives it.
    pipeline = install_pipeline(app)
    stop = app.components[-1].stop
    try:
        app.bus.publish(WakeWordDetected(phrase="айрис"))
        assert pipeline.session_id == ""
    finally:
        if stop is not None:
            stop()
