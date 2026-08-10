"""Which workers exist, which ones this configuration needs, and which start now.

The registry is the single place that answers "what should be running?". It maps
each worker type to its entrypoint, its restart scope and the slice of
:class:`~ayris.core.config.Settings` that type actually needs, then turns a
settings object into a :class:`WorkerPlan` the supervisor can execute.

Two questions are kept apart on purpose:

*needed* — will this worker ever be used under this configuration? A user who
picked online-only recognition still needs the STT worker: it holds the cloud
client. A user who turned off the wake word does not need a wake worker at all.

*preferred* — is it on the primary path right now? Online recognition means the
local model is a fallback that may never be reached.

Eco mode (section 12 of the specification) reads the second answer: a worker that
is needed but not preferred is registered without being started, and comes up on
first use — trading a few seconds of latency the first time for a gigabyte of RAM
and a cold GPU. With eco mode off, everything needed is started up front so the
first command is as fast as the tenth. Nothing outside the plan ever starts,
under either mode.

Worker modules land over the following tasks (audio in 06, STT in 09, TTS in 12,
LLM in 55), so an entrypoint is checked for importability before it is planned and
a missing one is skipped with a log line instead of failing the launch.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self, cast

from ayris.core.config import RestartScope, Settings
from ayris.core.models import JsonObject
from ayris.workers.protocol import PROTOCOL_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from ayris.core.events import Event

    #: What a worker event becomes on the bus. ``None`` drops the event.
    EventTranslator = Callable[[str, JsonObject], Event | None]

__all__ = [
    "TRANSLATOR_ATTR",
    "WORKER_TYPES",
    "PlannedWorker",
    "WorkerKind",
    "WorkerPlan",
    "WorkerSpec",
    "WorkerType",
    "event_translator",
    "plan_workers",
    "worker_type",
]

_log = logging.getLogger("ayris.workers.registry")

#: Default heartbeat cadence. The supervisor's patience is a multiple of it.
DEFAULT_HEARTBEAT: Final = 2.0

#: How many missed heartbeats mean "hung". Three keeps a worker that lost a
#: scheduling slice under load from being killed for it.
DEFAULT_HEARTBEAT_MISSES: Final = 3


class WorkerKind(StrEnum):
    """The worker types Ayris knows about.

    A string enum because the value travels over the pipe, lands in log lines and
    is shown in DevTools; ``str(kind)`` has to be readable everywhere.
    """

    AUDIO = "audio"
    STT = "stt"
    TTS = "tts"
    LLM = "llm"

    @property
    def label(self) -> str:
        """Russian name for the settings window and DevTools."""
        return _KIND_LABELS[self]


_KIND_LABELS: Final[Mapping[WorkerKind, str]] = MappingProxyType(
    {
        WorkerKind.AUDIO: "Захват звука",
        WorkerKind.STT: "Распознавание речи",
        WorkerKind.TTS: "Синтез речи",
        WorkerKind.LLM: "Языковая модель",
    }
)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Everything needed to start one worker process.

    The supervisor treats this as opaque configuration: it starts what the spec
    describes and never inspects ``params``. Tests build a spec directly, which
    is why nothing here reaches into :class:`~ayris.core.config.Settings`.

    Args:
        name: Unique identifier. One process per name.
        kind: Worker type; ``"test"`` and other free-form values are allowed so
            the suite does not have to pretend to be a real subsystem.
        entrypoint: ``"module:Class"`` resolved inside the child process.
        params: Configuration handed to the worker. Must be picklable — settings
            objects are flattened to plain dicts by :func:`plan_workers`.
        priority: Process priority applied by the child at start-up.
        heartbeat_interval: Seconds between the worker's heartbeats.
        heartbeat_misses: Missed beats before the worker is declared hung.
        start_timeout: How long to wait for the ready message.
        stop_timeout: How long a polite stop is given before terminating.
        call_timeout: Default deadline for :meth:`WorkerManager.call`.
        max_restarts: Restarts allowed inside ``restart_window`` before the
            worker is marked failed and left alone.
        restart_window: Seconds over which restarts are counted.
        restart_delay: Delay before the first restart; doubles per attempt.
        max_restart_delay: Ceiling for that backoff.
        autostart: Start with the application, rather than on first call.
        python_path: Extra ``sys.path`` entries for the child. Plugin workers and
            the test fixture worker live outside the installed package.
        restart_scope: Which settings changes require restarting this worker.
        log_dir: Where the child writes its own log file, if at all.
        log_level: Threshold for the child's logger.
    """

    name: str
    entrypoint: str
    kind: str = WorkerKind.AUDIO
    params: JsonObject = field(default_factory=dict)
    priority: str = "normal"
    heartbeat_interval: float = DEFAULT_HEARTBEAT
    heartbeat_misses: int = DEFAULT_HEARTBEAT_MISSES
    start_timeout: float = 30.0
    stop_timeout: float = 5.0
    call_timeout: float = 30.0
    max_restarts: int = 5
    restart_window: float = 60.0
    restart_delay: float = 1.0
    max_restart_delay: float = 30.0
    autostart: bool = True
    python_path: tuple[str, ...] = ()
    restart_scope: RestartScope = RestartScope.NONE
    log_dir: Path | None = None
    log_level: str = "INFO"
    protocol_version: int = PROTOCOL_VERSION

    @property
    def heartbeat_timeout(self) -> float:
        """Silence after which the worker counts as hung."""
        return self.heartbeat_interval * max(self.heartbeat_misses, 1)

    def with_params(self, params: JsonObject) -> Self:
        """A copy carrying new parameters, for a reconfigure-and-restart."""
        return replace(self, params=dict(params))


@dataclass(frozen=True, slots=True)
class WorkerType:
    """A worker type and the rules that decide when it runs.

    Args:
        kind: The type itself.
        entrypoint: ``"module:Class"`` of its implementation.
        build: Turns settings into the spec for this type.
        is_needed: Whether the configuration uses this worker at all.
        is_preferred: Whether it is on the primary path, as opposed to being a
            fallback. Only consulted in eco mode.
        loads_local_model: Whether starting it costs real memory. Purely
            informational, shown in DevTools next to the eco-mode switch.
    """

    kind: WorkerKind
    entrypoint: str
    build: Callable[[Settings], WorkerSpec]
    is_needed: Callable[[Settings], bool]
    is_preferred: Callable[[Settings], bool]
    loads_local_model: bool = True

    @property
    def available(self) -> bool:
        """Whether the entrypoint's module can be imported in this build.

        Guards against a half-finished checkout and against an optional engine
        whose package the user never installed.
        """
        module_name = self.entrypoint.partition(":")[0]
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class PlannedWorker:
    """One worker in a plan, with the reason it is in that state.

    Args:
        spec: What to start.
        needed: Whether the configuration uses this worker at all.
        autostart: Whether to start it now rather than on first use.
        reason: Human-readable justification, shown in DevTools and logged when
            eco mode changes what runs.
    """

    spec: WorkerSpec
    needed: bool
    autostart: bool
    reason: str = ""

    @property
    def kind(self) -> str:
        """Shorthand for ``planned.spec.kind``."""
        return self.spec.kind


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    """What the supervisor should be running for a given configuration.

    Includes workers that are needed but deferred, because the supervisor still
    registers those: a deferred worker has to be startable on first call without
    consulting the settings again.
    """

    workers: tuple[PlannedWorker, ...] = ()
    eco_mode: bool = False

    def __iter__(self) -> Iterator[PlannedWorker]:
        return iter(self.workers)

    def __len__(self) -> int:
        return len(self.workers)

    @property
    def autostart(self) -> tuple[PlannedWorker, ...]:
        """Workers to bring up during start-up, in registry order."""
        return tuple(planned for planned in self.workers if planned.autostart)

    @property
    def deferred(self) -> tuple[PlannedWorker, ...]:
        """Workers registered but left cold until first use."""
        return tuple(planned for planned in self.workers if not planned.autostart)

    def by_kind(self, kind: str) -> PlannedWorker | None:
        """The planned worker of this type, or ``None`` if the plan omits it."""
        for planned in self.workers:
            if planned.spec.kind == kind:
                return planned
        return None

    def describe(self) -> str:
        """One-line summary for the log at start-up."""
        if not self.workers:
            return "воркеры не нужны"
        parts = [
            f"{planned.spec.name}{'' if planned.autostart else ' (по требованию)'}"
            for planned in self.workers
        ]
        suffix = ", режим экономии" if self.eco_mode else ""
        return ", ".join(parts) + suffix


# ----------------------------------------------------------------------
# per-type rules
# ----------------------------------------------------------------------


def _common(settings: Settings) -> JsonObject:
    """Fields every worker gets, so none of them has to load settings itself."""
    return {
        "log_level": settings.devtools.log_level,
        "gpu": settings.performance.gpu,
        "eco_mode": settings.performance.eco_mode,
    }


def _audio_spec(settings: Settings) -> WorkerSpec:
    audio = settings.voice.audio_input
    wake = settings.voice.wake
    return WorkerSpec(
        name="audio",
        kind=WorkerKind.AUDIO,
        entrypoint=_ENTRYPOINTS[WorkerKind.AUDIO],
        params={
            "device": audio.device,
            "sample_rate": audio.sample_rate,
            "frame_ms": audio.frame_ms,
            "gain": audio.gain,
            "vad_threshold": audio.vad_threshold,
            "vad_aggressiveness": audio.vad_aggressiveness,
            "silence_ms": audio.silence_ms,
            "max_utterance_sec": audio.max_utterance_sec,
            "denoise": audio.denoise,
            "noise_floor_db": audio.noise_floor_db,
            "wake_enabled": wake.enabled,
            "wake_engine": wake.engine,
            "wake_debounce_ms": wake.debounce_ms,
            "wake_phrases": [
                {"phrase": item.phrase, "sensitivity": item.sensitivity}
                for item in wake.phrases
                if item.enabled
            ],
            "mic_mode": wake.mic_mode,
            "listen_window_sec": wake.listen_window_sec,
            # The name of the credential, never the credential: the audio worker
            # reads the key out of the Windows store itself, so a vendor key
            # cannot end up in a worker spec, a log line or a crash report.
            "wake_credential_ref": wake.credential_ref,
        },
        # A dropped input buffer is audible and unrecoverable, so this is the one
        # worker allowed above normal priority by default.
        priority=settings.performance.audio_priority,
        # Capture must not stall: the shorter beat catches a stuck device fast.
        heartbeat_interval=1.0,
        call_timeout=10.0,
        restart_scope=RestartScope.AUDIO,
    )


def _stt_spec(settings: Settings) -> WorkerSpec:
    stt = settings.voice.stt
    return WorkerSpec(
        name="stt",
        kind=WorkerKind.STT,
        entrypoint=_ENTRYPOINTS[WorkerKind.STT],
        params={
            "mode": stt.mode,
            "offline_engine": stt.offline_engine,
            "offline_model": stt.offline_model,
            "online_provider": stt.online_provider,
            "credential_ref": stt.credential_ref,
            "online_timeout_sec": stt.online_timeout_sec,
            "punctuation": stt.punctuation,
            "partial_results": stt.partial_results,
            "min_confidence": stt.min_confidence,
            "sample_rate": settings.voice.audio_input.sample_rate,
            "threads": settings.performance.stt_threads,
            "gpu": settings.performance.gpu,
            # The worker enforces the memory cap itself, before a load: it is the
            # only process that knows how big the model on disk actually is.
            "ram_limit_mb": settings.performance.ram_limit_mb,
            "model_idle_sec": settings.performance.model_idle_sec,
        },
        priority=settings.performance.process_priority,
        # Loading a recognition model takes tens of seconds on a cold disk.
        start_timeout=120.0,
        call_timeout=max(60.0, stt.online_timeout_sec * 4),
        restart_scope=RestartScope.STT,
    )


def _tts_spec(settings: Settings) -> WorkerSpec:
    tts = settings.voice.tts
    return WorkerSpec(
        name="tts",
        kind=WorkerKind.TTS,
        entrypoint=_ENTRYPOINTS[WorkerKind.TTS],
        params={
            "engine": tts.engine,
            "voice": tts.voice,
            "speed": tts.speed,
            "pitch": tts.pitch,
            "volume": tts.volume,
            "output_device": tts.output_device,
            "cloud_fallback": tts.cloud_fallback,
            "credential_ref": tts.credential_ref,
            "cache_size_mb": tts.cache_size_mb,
            "threads": settings.performance.tts_threads,
            "gpu": settings.performance.gpu,
            # Same reasoning as for STT: the worker is the only process that can
            # measure the voice on disk, and it unloads it again when it goes cold.
            "ram_limit_mb": settings.performance.ram_limit_mb,
            "model_idle_sec": settings.performance.model_idle_sec,
            "eco_mode": settings.performance.eco_mode,
        },
        priority=settings.performance.process_priority,
        start_timeout=90.0,
        call_timeout=60.0,
        restart_scope=RestartScope.TTS,
    )


def _llm_spec(settings: Settings) -> WorkerSpec:
    ai = settings.ai
    return WorkerSpec(
        name="llm",
        kind=WorkerKind.LLM,
        entrypoint=_ENTRYPOINTS[WorkerKind.LLM],
        params={
            "provider": ai.provider,
            "model": ai.model,
            "host": ai.host,
            "credential_ref": ai.credential_ref,
            "temperature": ai.temperature,
            "max_tokens": ai.max_tokens,
            "request_timeout_sec": ai.request_timeout_sec,
            "threads": settings.performance.llm_threads,
            "gpu": settings.performance.gpu,
        },
        priority=settings.performance.process_priority,
        start_timeout=120.0,
        # Generation is the slowest thing Ayris does; the worker's own timeout
        # fires first and returns a proper error.
        call_timeout=ai.request_timeout_sec + 30.0,
        restart_scope=RestartScope.LLM,
    )


_ENTRYPOINTS: Final[Mapping[WorkerKind, str]] = MappingProxyType(
    {
        WorkerKind.AUDIO: "ayris.workers.audio_worker:AudioWorker",
        WorkerKind.STT: "ayris.workers.stt_worker:SttWorker",
        WorkerKind.TTS: "ayris.workers.tts_worker:TtsWorker",
        WorkerKind.LLM: "ayris.nlu.llm.worker:LlmWorker",
    }
)

#: Attribute a worker module may expose to say what its events mean on the bus.
#: See :func:`event_translator`.
TRANSLATOR_ATTR: Final = "EVENT_TRANSLATOR"


def event_translator(kind: str) -> EventTranslator | None:
    """Return the bus translator a worker type publishes, if it has one.

    The supervisor must not know that the audio worker's ``level`` event is an
    :class:`~ayris.core.events.AudioLevelChanged` - that knowledge belongs to
    the subsystem - but something has to introduce the two. Each worker module
    may expose :data:`TRANSLATOR_ATTR`, and this looks it up by worker type.

    Importing the module here is safe and cheap: the same module is imported in
    the child process anyway, and :meth:`WorkerType.available` has already
    checked that it exists. A worker type without a translator, one that is not
    a known kind (the test fixture worker is not), or one whose module fails to
    import, yields ``None`` rather than an error - its events then reach the log
    and stop there.
    """
    try:
        entrypoint = _ENTRYPOINTS[WorkerKind(kind)]
    except ValueError:
        return None
    module_name = entrypoint.partition(":")[0]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        _log.debug("нет модуля %s для трансляции событий: %s", module_name, exc)
        return None
    translator = getattr(module, TRANSLATOR_ATTR, None)
    if not callable(translator):
        return None
    return cast("EventTranslator", translator)


def _stt_needed(settings: Settings) -> bool:
    """Always: the worker owns the cloud client as well as the local model."""
    del settings
    return True


def _stt_preferred(settings: Settings) -> bool:
    """Offline recognition is primary unless the user chose online-only.

    In ``auto`` the local model is the fallback, but it is the one that has to be
    warm when the network drops, so it counts as preferred.
    """
    return settings.voice.stt.mode != "online"


def _tts_preferred(settings: Settings) -> bool:
    """A cloud voice does not need a local synthesiser warm at start-up."""
    return settings.voice.tts.engine in {"piper", "silero", "xtts", "sapi"}


def _llm_needed(settings: Settings) -> bool:
    """Only if some feature actually asks a model something."""
    ai = settings.ai
    return ai.fallback_to_llm or ai.llm_understanding or ai.free_chat


def _llm_preferred(settings: Settings) -> bool:
    """Free chat or model-based understanding keeps a local model hot.

    Plain fallback fires rarely, so under eco mode it waits for first use. A
    remote provider costs nothing to keep "loaded", but the worker still holds a
    connection pool, so the same rule applies.
    """
    return settings.ai.llm_understanding or settings.ai.free_chat


def _always(settings: Settings) -> bool:
    del settings
    return True


WORKER_TYPES: Final[tuple[WorkerType, ...]] = (
    WorkerType(
        kind=WorkerKind.AUDIO,
        entrypoint=_ENTRYPOINTS[WorkerKind.AUDIO],
        build=_audio_spec,
        # Push-to-talk and the hotkey still need capture, so this is not tied to
        # the wake word being on.
        is_needed=_always,
        is_preferred=_always,
        loads_local_model=False,
    ),
    WorkerType(
        kind=WorkerKind.STT,
        entrypoint=_ENTRYPOINTS[WorkerKind.STT],
        build=_stt_spec,
        is_needed=_stt_needed,
        is_preferred=_stt_preferred,
    ),
    WorkerType(
        kind=WorkerKind.TTS,
        entrypoint=_ENTRYPOINTS[WorkerKind.TTS],
        build=_tts_spec,
        is_needed=_always,
        is_preferred=_tts_preferred,
    ),
    WorkerType(
        kind=WorkerKind.LLM,
        entrypoint=_ENTRYPOINTS[WorkerKind.LLM],
        build=_llm_spec,
        is_needed=_llm_needed,
        is_preferred=_llm_preferred,
    ),
)


def worker_type(kind: str) -> WorkerType | None:
    """Look up a registered type by its string value."""
    for entry in WORKER_TYPES:
        if entry.kind == kind:
            return entry
    return None


# ----------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------


def plan_workers(
    settings: Settings,
    *,
    log_dir: Path | None = None,
    include_unavailable: bool = False,
) -> WorkerPlan:
    """Decide what should run under ``settings``.

    Args:
        settings: Current configuration.
        log_dir: Passed to each worker so children write beside the main log.
        include_unavailable: Plan types whose module cannot be imported. Off by
            default — a half-implemented type must not break the launch — and on
            in tests, which check the rules rather than the modules.

    Returns:
        The plan, in registry order: audio first, because everything downstream
        waits on it.
    """
    eco = settings.performance.eco_mode
    common = _common(settings)
    planned: list[PlannedWorker] = []

    for entry in WORKER_TYPES:
        if not entry.is_needed(settings):
            _log.debug("воркер %s не нужен при текущих настройках", entry.kind)
            continue
        if not include_unavailable and not entry.available:
            _log.warning(
                "воркер %s пропущен: модуль %s ещё не реализован",
                entry.kind,
                entry.entrypoint.partition(":")[0],
            )
            continue

        spec = entry.build(settings)
        params = dict(common)
        params.update(spec.params)
        spec = replace(
            spec,
            params=params,
            log_dir=log_dir,
            log_level=str(params.get("log_level", "INFO")),
        )

        preferred = entry.is_preferred(settings)
        autostart = preferred or not eco
        if autostart:
            reason = "основной путь" if preferred else "предзагрузка"
        elif entry.loads_local_model:
            reason = "режим экономии: запуск при первом обращении"
        else:
            reason = "режим экономии: запасной путь"

        planned.append(
            PlannedWorker(
                spec=replace(spec, autostart=autostart),
                needed=True,
                autostart=autostart,
                reason=reason,
            )
        )

    plan = WorkerPlan(workers=tuple(planned), eco_mode=eco)
    _log.debug("план воркеров: %s", plan.describe())
    return plan
