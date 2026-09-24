"""Compose the runtime :class:`~ayris.audio.tts.router.TtsRouter` and share it.

Task 47 wired the pipeline's understanding path but left speech unattached: the
TTS engines only ran inside the worker, and nothing turned ``voice.tts`` into a
router the running application could actually speak through. This module is that
seam.

:func:`build_tts_router` reads ``voice.tts``, builds one router over the single
:class:`~ayris.audio.tts.player.TtsPlayer` the macro sounds already own (passed
in, never created here — one device owner, not two), and wires it to the bus so
:class:`~ayris.core.pipeline.Pipeline` speaks answers through it and the macro
:class:`~ayris.actions.macros.sounds.library.SoundLibrary` synthesises ``tts:``
bindings through the very same voice.

The router takes two *providers* — callables that build **and load** an engine on
first use (:class:`~ayris.audio.tts.router._Slot` calls the provider once and
then only ``synthesize_stream``; it never calls ``load`` itself). Which engine
goes in which slot follows from the settings:

* a cloud ``engine`` → the cloud slot speaks it and the local slot is the offline
  safety net (bundled Piper); the mode is
  :attr:`~ayris.audio.tts.router.TtsMode.AUTO`;
* a local ``engine`` → the local slot speaks it and the cloud slot is wired to
  ``credential_ref`` (when that names a provider) so ``cloud_fallback`` can be
  toggled live without a restart — the mode gates whether it is used.

``engine``/``voice``/``output_device`` are ``RestartScope.TTS``: a change to them
takes effect on restart, so the slots are read once. Speed, pitch, volume and
``cloud_fallback`` are live, so the ``ConfigChanged`` handler pushes them into the
running router with ``set_params``/``set_mode``.

:func:`set_active_tts_router` / :func:`active_tts_router` are the shared handle,
the same shape as
:func:`~ayris.actions.macros.sounds.runtime.set_active_sound_library`: the
dispatcher registers the router it built so :func:`~ayris.core.pipeline_app.install_pipeline`
can hand it to the pipeline as its :class:`~ayris.core.pipeline.SpeechOutput`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayris.audio.tts.base import TtsOptions, VoiceSpec, create_engine, engine_class
from ayris.audio.tts.cloud_base import create_cloud_engine, is_cloud_engine
from ayris.audio.tts.router import TtsRouter, VoiceParams, mode_from_config
from ayris.audio.tts.service import connect_player
from ayris.core.events import ConfigChanged
from ayris.core.paths import get_paths
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayris.audio.tts.base import TtsEngine
    from ayris.audio.tts.player import TtsPlayer
    from ayris.core.app import AyrisApp
    from ayris.core.config import TtsConfig

    EngineProvider = Callable[[], TtsEngine]

__all__ = [
    "active_tts_router",
    "build_tts_router",
    "set_active_tts_router",
]

_log = get_logger(__name__)

#: The voice that ships with Ayris and needs no key — the offline safety net a
#: cloud engine falls back to, and what the sound library speaks through until a
#: local voice is installed.
_DEFAULT_LOCAL_ENGINE: Final = "piper"
_DEFAULT_LOCAL_VOICE: Final = "ru_RU-irina-medium"


def build_tts_router(app: AyrisApp, player: TtsPlayer) -> tuple[TtsRouter, Callable[[], None]]:
    """Build the runtime router over ``player`` and return it with a teardown.

    Args:
        app: The running application — ``voice.tts`` for the voice, the bus for
            completions and connectivity.
        player: The one device owner, shared with the macro sound library. The
            router speaks through it; it is never created or stopped here (the
            dispatcher that built it owns its lifetime).

    Returns:
        The router and a ``close`` callable that unsubscribes the config handler,
        closes the router (dropping its bus subscriptions and unloading the
        engines) and closes the player bridge. ``close`` does **not** stop the
        player — that stays with whoever built it.
    """
    cfg = app.settings.voice.tts
    built_engine = cfg.engine
    params = VoiceParams(speed=cfg.speed, pitch=cfg.pitch, volume=cfg.volume)
    mode = mode_from_config(built_engine, cloud_fallback=cfg.cloud_fallback)

    cloud, local = _providers(cfg)

    bridge = connect_player(app.bus, player)
    router = TtsRouter(
        player,
        mode=mode,
        cloud=cloud,
        local=local,
        params=params,
        bus=app.bus,
        monitor=None,
        on_spoken=None,
    )

    def on_config(_event: ConfigChanged) -> None:
        # engine/voice are RestartScope.TTS, so the slots stand until restart; only
        # the live knobs move. The mode is recomputed from the engine that was
        # actually built, so ticking cloud_fallback flips the fallback direction
        # without pretending a not-yet-loaded new engine is already in the slot.
        live = app.settings.voice.tts
        router.set_params(VoiceParams(speed=live.speed, pitch=live.pitch, volume=live.volume))
        router.set_mode(mode_from_config(built_engine, cloud_fallback=live.cloud_fallback))

    unsub_config = app.bus.subscribe(ConfigChanged, on_config, weak=False)

    def close() -> None:
        unsub_config()
        router.close()
        bridge.close()

    _log.info(
        "маршрутизатор TTS собран: движок %s, режим %s, облачный резерв %s",
        built_engine,
        mode,
        cfg.credential_ref if is_cloud_engine(cfg.credential_ref) else "нет",
    )
    return router, close


def _providers(cfg: TtsConfig) -> tuple[EngineProvider | None, EngineProvider | None]:
    """The ``(cloud, local)`` providers for the configured engine.

    A cloud engine is the primary voice with the bundled local one behind it; a
    local engine is primary with the ``credential_ref`` provider wired behind it
    when that names a cloud service, so ``cloud_fallback`` works the moment it is
    ticked rather than only after the next restart.
    """
    if is_cloud_engine(cfg.engine):
        cloud: EngineProvider | None = _cloud_provider(
            cfg.engine, cfg.voice, cfg.credential_ref, endpoint=cfg.endpoint, model=cfg.model
        )
        local = _local_provider(_DEFAULT_LOCAL_ENGINE, _DEFAULT_LOCAL_VOICE)
        return cloud, local
    local = _local_provider(cfg.engine, cfg.voice)
    cloud = (
        _cloud_provider(
            cfg.credential_ref, "", cfg.credential_ref, endpoint=cfg.endpoint, model=cfg.model
        )
        if is_cloud_engine(cfg.credential_ref)
        else None
    )
    return cloud, local


def _local_provider(engine_name: str, voice_id: str) -> EngineProvider:
    """A provider that constructs and loads a local engine on first use."""

    def provide() -> TtsEngine:
        engine = create_engine(engine_name)
        engine.load(_resolve_local_voice(engine_name, voice_id), TtsOptions())
        return engine

    return provide


def _cloud_provider(
    engine_name: str,
    voice_id: str,
    credential_ref: str,
    *,
    endpoint: str = "",
    model: str = "",
) -> EngineProvider:
    """A provider that constructs and loads a cloud engine on first use.

    Only a *reference* to the credential is passed; the engine reads the key from
    the Windows credential store itself and it never enters the configuration. An
    empty ``voice_id`` lets the provider speak in its own default voice, which is
    what the fallback slot wants — the user picked a local voice, not this one.

    ``endpoint`` and ``model`` are the generic OpenAI-compatible engine's settings;
    they are threaded in only when set, so a blank endpoint never overrides another
    provider's built-in default and a named provider (Yandex, Google, …) keeps its
    own base URL when it happens to be wired as the fallback slot.
    """

    def provide() -> TtsEngine:
        engine = create_cloud_engine(engine_name)
        voice = VoiceSpec(engine=engine_name, voice_id=voice_id)
        extra: dict[str, str] = {"credential_ref": credential_ref}
        if endpoint:
            extra["endpoint"] = endpoint
        if model:
            extra["model"] = model
        engine.load(voice, TtsOptions(extra=extra))
        return engine

    return provide


def _resolve_local_voice(engine_name: str, voice_id: str) -> VoiceSpec:
    """Resolve a configured local voice name to a spec, mirroring the worker.

    An enumerable voice is used as enumerated — that carries the real path,
    language and sample rate; anything else becomes a bare spec the engine
    resolves by its own conventions, so a hand-written ``config.toml`` naming just
    ``ru_RU-irina-medium`` keeps working.
    """
    directory = get_paths().tts_models_dir
    candidate = Path(voice_id)
    for spec in engine_class(engine_name).voices(directory):
        if voice_id in {spec.voice_id, spec.path, spec.display_name}:
            return spec
    return VoiceSpec(
        engine=engine_name,
        voice_id=candidate.stem if candidate.is_absolute() else voice_id,
        path=str(candidate) if candidate.is_absolute() else "",
    )


_ACTIVE_ROUTER: TtsRouter | None = None
_ROUTER_LOCK: Final = threading.Lock()


def set_active_tts_router(router: TtsRouter | None) -> None:
    """Register (or clear) the router the pipeline and sound library speak through."""
    global _ACTIVE_ROUTER
    with _ROUTER_LOCK:
        _ACTIVE_ROUTER = router


def active_tts_router() -> TtsRouter | None:
    """The router registered with :func:`set_active_tts_router`, if any."""
    with _ROUTER_LOCK:
        return _ACTIVE_ROUTER
