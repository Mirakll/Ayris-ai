"""Speaking, playing a sound, and stopping one — the «Голос/Звук» macro blocks.

Task 47 built the speech runtime (:class:`~ayris.audio.tts.router.TtsRouter` over
one :class:`~ayris.audio.tts.player.TtsPlayer`) and the macro sound path
(:class:`~ayris.actions.macros.sounds.library.SoundLibrary` over the *same*
player), and the dispatcher wired both to shared handles at start-up
(:func:`~ayris.audio.tts.app_router.active_tts_router`,
:func:`~ayris.actions.macros.sounds.runtime.active_sound_library`). What was
missing was the seam from a *macro* to that runtime: the palette showed ``Say``,
``PlaySound``, ``StopSound`` and ``SetTTSVoice`` greyed out with «ещё не
подключено в этой сборке», because nothing in the registry answered those names.

This module is that seam. Each block is an ordinary :class:`~ayris.actions.base.Action`
registered under :class:`~ayris.actions.base.ActionCategory.AUDIO`, exactly like
:class:`~ayris.actions.system.audio.SetVolume` next door — so it inherits the one
audit writer, the confirmation/admin gates, and the timeout pool, and the block
catalog flips it to ``available`` the moment discovery finds it, no catalog edit
required.

Two disciplines carry weight here:

* **The runtime is reached lazily, inside** ``run``. Importing the TTS router or
  the sound library at module load would pull PortAudio and PyAV into discovery
  and could refuse to import on a CI runner with no audio stack, which would make
  the block silently vanish rather than fail loudly when actually used. So the
  global handles are imported per call, the same way
  :mod:`~ayris.actions.system.audio` imports pycaw lazily.
* **A missing runtime is a handled failure, not a crash.** When the app started
  without a working audio path (``active_tts_router() is None``), the action
  returns :meth:`~ayris.actions.result.ActionResult.failed` with a Russian
  sentence, not an exception — the macro sees ``ok=False`` and can branch on it.

:class:`Say` speaks through the very voice the assistant answers in;
:class:`SetTTSVoice` changes that voice for the running session (it is not
persisted — engine and voice are ``RestartScope.TTS`` — so a macro that sets it
is setting it *until restart*, which is what a session-scoped block should do).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.audio.tts.base import VoiceSpec
    from ayris.audio.tts.router import TtsRouter

__all__ = [
    "PlaySound",
    "Say",
    "SetTTSVoice",
    "StopSound",
]

_log = get_logger(__name__)

#: The engine assumed when neither the running voice nor the settings name one —
#: the same bundled voice the router falls back to. Only used to resolve a voice
#: *name* to a spec; the router still speaks through whatever slot it actually has.
_DEFAULT_ENGINE = "piper"

#: No spoken confirmation for speaking or playing: the sound *is* the feedback,
#: and narrating «сказала» after every phrase would echo. ``message_ru`` stays
#: empty for those, and carries a short line only where a change is otherwise
#: silent (:class:`SetTTSVoice`, :class:`StopSound`).


@register
class Say(Action):
    """Speak a phrase in the assistant's own voice."""

    meta: ClassVar = ActionMeta(
        name="Say",
        category=ActionCategory.AUDIO,
        title_ru="Сказать",
        description_ru="Произнести текст голосом ассистента через синтез речи",
        # Speaking a long passage is a legitimate wait, like the «Пауза» block;
        # the handle always resolves (the player reports completion or error, and
        # a shutdown resolves anything still pending), so an unbounded wait cannot
        # wedge the worker forever.
        timeout_ms=0,
    )

    class Params(ActionParams):
        text: str = Field(
            min_length=1,
            max_length=2000,
            title="Текст",
            description="Что произнести",
            json_schema_extra={"multiline": True},
        )
        wait: bool = Field(
            default=True,
            title="Дождаться",
            description="Продолжить макрос только после того, как фраза договорена",
        )

    def run(self, params: Params) -> ActionResult[None]:
        from ayris.audio.tts.app_router import active_tts_router

        router = active_tts_router()
        if router is None:
            return ActionResult.failed(
                "Синтез речи недоступен: голос не настроен.",
                detail="no active tts router",
            )
        handle = router.say(params.text)
        if params.wait:
            handle.wait()
        return ActionResult.done(
            detail=f"spoke {len(params.text)} chars via {handle.engine or 'tts'}",
        )


@register
class PlaySound(Action):
    """Play a built-in or user sound (or a «tts:…» phrase) through the mixer."""

    meta: ClassVar = ActionMeta(
        name="PlaySound",
        category=ActionCategory.AUDIO,
        title_ru="Проиграть звук",
        description_ru="Проиграть встроенный или свой звук, либо фразу «tts:текст»",
        timeout_ms=0,
    )

    class Params(ActionParams):
        sound: str = Field(
            min_length=1,
            max_length=500,
            title="Звук",
            description="«notification», «builtin:…», «custom:файл.wav» или «tts:фраза»",
        )
        wait: bool = Field(
            default=False,
            title="Дождаться",
            description="Продолжить макрос только после того, как звук доиграет",
        )
        volume: int | None = Field(
            default=None,
            ge=0,
            le=100,
            title="Громкость",
            description="Процентов; пусто — как в звуке",
            json_schema_extra={"unit_ru": "%"},
        )

    def run(self, params: Params) -> ActionResult[None]:
        from ayris.actions.macros.schema import SoundBinding
        from ayris.actions.macros.sounds.runtime import active_sound_library

        library = active_sound_library()
        if library is None:
            return ActionResult.failed(
                "Звук недоступен: вывод звука не настроен.",
                detail="no active sound library",
            )
        # A bad reference (unknown builtin, missing file, tts without a synthesiser)
        # raises a SoundLibraryError, which is an AyrisError the registry turns into
        # a spoken failure on its own — no need to catch it here.
        binding = SoundBinding(value=params.sound, volume=params.volume, wait=params.wait)
        library.play_binding(binding, owner="macro:play")
        return ActionResult.done(detail=f"played {binding.reference}")


@register
class StopSound(Action):
    """Stop sounds started by commands. The assistant's own speech is untouched."""

    meta: ClassVar = ActionMeta(
        name="StopSound",
        category=ActionCategory.AUDIO,
        title_ru="Остановить звук",
        description_ru="Остановить звуки, запущенные командами (речь ассистента не трогает)",
        timeout_ms=5_000,
    )

    class Params(ActionParams):
        pass

    def run(self, params: Params) -> ActionResult[None]:
        del params  # «Остановить звук» без параметров
        from ayris.actions.macros.sounds.runtime import active_sound_library

        library = active_sound_library()
        if library is None:
            return ActionResult.done(detail="no active sound library")
        stopped = library.mixer.stop()
        message = "Звук остановлен." if stopped else ""
        return ActionResult.done(message, detail=f"stopped {stopped} macro sound(s)")


@register
class SetTTSVoice(Action):
    """Change the synthesis voice for the phrases that follow, this session.

    The router speaks every phrase through ``params.voice`` when a call names no
    voice of its own, so replacing it here changes the assistant's voice from the
    next phrase on. It is not written to the config — engine and voice are
    ``RestartScope.TTS`` — so the change lasts until restart, which is what a
    macro that flips the voice mid-session means.
    """

    meta: ClassVar = ActionMeta(
        name="SetTTSVoice",
        category=ActionCategory.AUDIO,
        title_ru="Сменить голос",
        description_ru="Сменить голос синтеза речи для последующих фраз (до перезапуска)",
        timeout_ms=5_000,
    )

    class Params(ActionParams):
        voice: str = Field(
            min_length=1,
            max_length=200,
            title="Голос",
            description="Имя голоса, как в настройках, либо путь к модели",
        )

    def run(self, params: Params) -> ActionResult[None]:
        from ayris.audio.tts.app_router import active_tts_router

        router = active_tts_router()
        if router is None:
            return ActionResult.failed(
                "Синтез речи недоступен: голос не настроен.",
                detail="no active tts router",
            )
        engine_name = _configured_engine(router)
        spec = _resolve_voice_spec(engine_name, params.voice)
        router.set_params(router.params.merged(voice=spec))
        return ActionResult.done(
            f"Голос: {spec.label}.",
            detail=f"voice -> {spec.key}",
        )


def _configured_engine(router: TtsRouter) -> str:
    """The engine name a voice should be resolved against.

    The voice already in force names its engine; failing that, the configured
    ``voice.tts.engine`` does; failing even that, the bundled default. A resolved
    spec still only matters to whichever slot actually speaks — this just picks a
    sane namespace for the name the user typed.
    """
    current = router.params.voice
    if current is not None and current.engine:
        return current.engine
    try:
        from ayris.core.config import get_settings

        return get_settings().voice.tts.engine or _DEFAULT_ENGINE
    except Exception:  # настройки читаем по возможности — их отсутствие не беда
        return _DEFAULT_ENGINE


def _resolve_voice_spec(engine_name: str, name: str) -> VoiceSpec:
    """Resolve a voice name to a spec, mirroring the runtime router's own rule.

    An enumerable voice is used as enumerated (it carries the real path, language
    and sample rate); anything else becomes a bare spec the engine resolves by its
    own conventions, so a short name like ``irina`` or a cloud voice id both work.
    """
    from pathlib import Path

    from ayris.audio.tts.base import VoiceSpec, engine_class
    from ayris.core.paths import get_paths

    try:
        directory = get_paths().tts_models_dir
        for spec in engine_class(engine_name).voices(directory):
            if name in {spec.voice_id, spec.path, spec.display_name}:
                return spec
    except Exception:  # перечисление по возможности — иначе голый VoiceSpec
        _log.debug("не удалось перечислить голоса движка %s", engine_name, exc_info=True)
    candidate = Path(name)
    return VoiceSpec(
        engine=engine_name,
        voice_id=candidate.stem if candidate.is_absolute() else name,
        path=str(candidate) if candidate.is_absolute() else "",
    )
