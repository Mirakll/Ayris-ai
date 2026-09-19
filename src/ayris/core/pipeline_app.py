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

The voice path (wake word, STT, TTS) is **not** attached here: the recognition and
synthesis engines still run in workers that nothing adapts into the pipeline's
``SttSource``/``SpeechOutput`` yet, and attaching a half-wired voice path would
turn every real audio event into an error. Only :meth:`Pipeline.run_text` is used,
so the microphone button and the audio worker keep their own behaviour untouched
until those adapters exist.
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
    from ayris.nlu.matcher import Trigger

__all__ = ["install_pipeline"]

_log = logging.getLogger("ayris.core.pipeline_app")


def install_pipeline(app: AyrisApp) -> Pipeline:
    """Attach the pipeline to the application and return it.

    Registers under :attr:`~ayris.core.app.LifecycleStage.NLU`, so the pipeline
    comes up after the workers and before the GUI, and its subscriptions are
    dropped before the workers stop. The returned pipeline's
    :meth:`~ayris.core.pipeline.Pipeline.run_text` is what the dashboard's text
    field is wired to.
    """
    from ayris.core.app import Component, LifecycleStage
    from ayris.core.models import TriggerType
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

    pipeline = Pipeline(
        app.bus,
        state=app.state,
        matcher=matcher,
        settings=app.settings,
        history=repos.history,
    )

    def on_profile(event: ProfileSwitched) -> None:
        # The dispatcher rebuilds its own index on this event; the matcher's
        # library is per-profile too, so it has to follow.
        profile_box["id"] = event.profile.id
        index.replace_all(load(None))

    def on_config(event: ConfigChanged) -> None:
        # The «ИИ» toggles decide command-only / hybrid / model-only, and they can
        # change mid-session; the mode is read off the settings the pipeline holds.
        pipeline.apply_settings(event.diff.settings)

    unsub_profile = app.bus.subscribe(ProfileSwitched, on_profile, weak=False)
    unsub_config = app.bus.subscribe(ConfigChanged, on_config, weak=False)

    def stop() -> None:
        unsub_profile()
        unsub_config()
        unbind()
        pipeline.close()

    app.add_component(Component(name="пайплайн", stage=LifecycleStage.NLU, stop=stop))
    _log.info("пайплайн подключён: %d голосовых триггеров в индексе", index.stats().triggers)
    return pipeline
