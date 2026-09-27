"""Bring the dispatcher (task 18) up inside the application and feed it text.

Task 18 built :class:`~ayris.core.pipeline.Pipeline` against protocols and left
its collaborators as stubs; task 47 is the seam that connects it to the running
application so a command typed in the dashboard actually runs.

The text path is deliberately narrow. :func:`install_pipeline` gives the pipeline
a live :class:`~ayris.nlu.matcher.Matcher` built from the profile's voice triggers
and *no* action runner: a matched command is published as
:class:`~ayris.core.events.IntentMatched`, and the
:class:`~ayris.triggers.dispatcher.TriggerDispatcher` — the single route to
``MacroEngine.start`` — executes it off that event. Wiring an in-pipeline runner
too would run the command twice, so the pipeline stays the understanding
front-end and the dispatcher stays the executor.

The **input** side is now wired too (task 47's closing seam). When a
:class:`~ayris.workers.manager.WorkerManager` is supplied, :func:`install_pipeline`
builds the two adapters in :mod:`ayris.core.worker_speech` —
:class:`~ayris.core.worker_speech.WorkerSttSource` and
:class:`~ayris.core.worker_speech.WorkerPhraseSource` — over that manager and calls
:meth:`~ayris.core.pipeline.Pipeline.attach`, so a wake word (or the push-to-talk
key, which arrives as the same :class:`~ayris.core.events.WakeWordDetected`) opens
a session, the finished phrase's PCM is pulled from the audio worker and handed to
the STT worker, and the recognised text runs the very same understanding path a
typed command does. Both adapters degrade softly — an unavailable or slow worker
becomes «не расслышала», never a crash — and the loop is attached only while the
voice input is actually enabled, so a profile or settings change that turns the
microphone off detaches it on the fly. With no manager (a text-only pipeline, as
in the tests) the input stays unattached and only :meth:`run_text` drives it.

The **output** side is wired: :func:`install_pipeline` hands the pipeline the
runtime :class:`~ayris.audio.tts.router.TtsRouter` (built by the dispatcher over
the player shared with the macro sounds — see :mod:`ayris.audio.tts.app_router`)
as its :class:`~ayris.core.pipeline.SpeechOutput`, so a matched command's answer
is spoken. When no router could be built the pipeline gets ``None`` and runs
silently instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ayris.core.events import ConfigChanged
from ayris.core.pipeline import Pipeline
from ayris.core.profile import ProfileSwitched

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.core.app import AyrisApp
    from ayris.core.config import Settings
    from ayris.nlu.matcher import Trigger
    from ayris.workers.manager import WorkerManager

__all__ = ["install_pipeline"]

_log = logging.getLogger("ayris.core.pipeline_app")


def _voice_input_live(settings: Settings) -> bool:
    """Whether any path can feed the wake→STT loop under these settings.

    The pipeline listens only when a spoken activation could actually arrive:
    the wake word is enabled, or the microphone mode allows the push-to-talk key
    (which reaches the pipeline as the same wake event). With neither, nothing
    would ever open a session, so the handlers are dropped rather than left
    waiting on events that cannot come — and re-added the moment the setting
    flips back. The audio worker still owns *when* it emits those events; this is
    the coarser gate above it.
    """
    wake = settings.voice.wake
    return wake.enabled or wake.mic_mode in ("ptt", "hybrid")


def install_pipeline(app: AyrisApp, manager: WorkerManager | None = None) -> Pipeline:
    """Attach the pipeline to the application and return it.

    Registers under :attr:`~ayris.core.app.LifecycleStage.NLU`, so the pipeline
    comes up after the workers and before the GUI, and its subscriptions are
    dropped before the workers stop. The returned pipeline's
    :meth:`~ayris.core.pipeline.Pipeline.run_text` is what the dashboard's text
    field is wired to.

    Args:
        app: The running application, for the bus, state, settings, profile and
            repositories.
        manager: The worker supervisor. When given, the voice **input** path is
            wired: the STT and phrase adapters in
            :mod:`ayris.core.worker_speech` are built over it and the pipeline is
            attached to the wake/speech events while voice input is enabled. When
            ``None`` the pipeline is text-only — only
            :meth:`~ayris.core.pipeline.Pipeline.run_text` drives it — which is
            what the unit tests use.
    """
    from ayris.audio.tts.app_router import active_tts_router
    from ayris.core.app import Component, LifecycleStage
    from ayris.core.models import TriggerType
    from ayris.core.worker_speech import WorkerPhraseSource, WorkerSttSource
    from ayris.nlu.index import TriggerIndex
    from ayris.nlu.matcher import Matcher, trigger_from_db

    repos = app.repositories
    # A one-element holder so the profile-switch handler can retarget the loader
    # without rebuilding it: the closure closes over the box, not the value.
    profile_box = {"id": app.profile.id}

    def load(command_id: int | None) -> Sequence[Trigger]:
        """Voice triggers as the matcher wants them: whole library, or one command."""
        if command_id is None:
            profile_id = profile_box["id"]
            if profile_id is None:
                return ()
            rows = repos.triggers.list_for_profile(
                profile_id, trigger_type=TriggerType.VOICE, enabled_only=True
            )
        else:
            rows = [
                row
                for row in repos.triggers.list_for_command(command_id)
                if row.type is TriggerType.VOICE
            ]
        triggers: list[Trigger] = []
        for row in rows:
            command = repos.commands.get(row.command_id)
            if command is None or not command.enabled or command.id is None:
                continue
            converted = trigger_from_db(
                row, command_priority=command.priority, enabled=command.enabled
            )
            if converted is not None:
                triggers.append(converted)
        return triggers

    index = TriggerIndex()
    index.replace_all(load(None))
    unbind = index.bind(app.bus, load)
    matcher = Matcher(index)

    # The input adapters exist only when there is a worker manager to feed them;
    # a text-only pipeline (the tests) keeps stt/phrase_source as None and never
    # attaches. actions stays None regardless: a match is published as
    # IntentMatched and the dispatcher is the single executor (task 47).
    stt = WorkerSttSource(manager) if manager is not None else None
    phrase_source = WorkerPhraseSource(manager) if manager is not None else None

    pipeline = Pipeline(
        app.bus,
        state=app.state,
        matcher=matcher,
        settings=app.settings,
        history=repos.history,
        tts=active_tts_router(),
        stt=stt,
        phrase_source=phrase_source,
    )

    def sync_voice_loop(settings: Settings) -> None:
        # attach()/detach() are idempotent, so this can run on every config
        # change. With no manager there is nothing to feed the loop, so it stays
        # detached whatever the settings say.
        if manager is not None and _voice_input_live(settings):
            pipeline.attach()
        else:
            pipeline.detach()

    def on_profile(event: ProfileSwitched) -> None:
        # The dispatcher rebuilds its own index on this event; the matcher's
        # library is per-profile too, so it has to follow.
        profile_box["id"] = event.profile.id
        index.replace_all(load(None))

    def on_config(event: ConfigChanged) -> None:
        # The «ИИ» toggles decide command-only / hybrid / model-only, and they can
        # change mid-session; the mode is read off the settings the pipeline holds.
        pipeline.apply_settings(event.diff.settings)
        # The voice toggles (wake word, mic mode) can flip the input loop on or
        # off between sessions; follow them so a muted microphone stops listening.
        sync_voice_loop(event.diff.settings)

    unsub_profile = app.bus.subscribe(ProfileSwitched, on_profile, weak=False)
    unsub_config = app.bus.subscribe(ConfigChanged, on_config, weak=False)

    # Attach the input loop now if the current settings want it; run_text works
    # either way.
    listening = manager is not None and _voice_input_live(app.settings)
    sync_voice_loop(app.settings)

    def stop() -> None:
        unsub_profile()
        unsub_config()
        unbind()
        pipeline.close()

    app.add_component(Component(name="пайплайн", stage=LifecycleStage.NLU, stop=stop))
    _log.info(
        "пайплайн подключён: %d голосовых триггеров в индексе, голосовой вход %s",
        index.stats().triggers,
        "включён" if listening else "выключен",
    )
    return pipeline
