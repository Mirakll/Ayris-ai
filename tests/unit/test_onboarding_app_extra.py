"""Добор покрытия ядра: пайплайн, менеджер настроек и режимы NLU — на моках.

Ни сети, ни железа, ни моделей, ни живого Qt: каждый внешний шов — фейк, время
подаётся вручную через ``ManualScheduler``, часы инъектируются. Файл добирает те
ветки ``ayris.core.pipeline`` и ``ayris.core.config``, до которых не доходит
``test_llm_modes``: голосовой путь (PTT/wake, распознавание, таймауты стадий),
разговорные ветки (отмена, повтор, ответ на вопрос), сброс памяти, ошибки
действия и озвучки, пустой ответ модели и незаданная модель, а также менеджер
настроек — diff, apply, подписка, перечитывание и слежение за файлом.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from ayris.audio.stt.base import AudioBuffer, TranscriptResult
from ayris.core import config as config_module
from ayris.core.app import (
    AlreadyRunningError,
    AppOptions,
    AyrisApp,
    Component,
    LifecycleStage,
)
from ayris.core.config import (
    ConfigError,
    ConfigManager,
    RestartScope,
    Settings,
    diff_settings,
    init_config,
    reset_config_manager,
    save_settings,
)
from ayris.core.errors import ActionError, AyrisError, TtsError
from ayris.core.events import ConfigChanged, EventBus, TranscriptReady
from ayris.core.models import ExecutionResult
from ayris.core.pipeline import (
    ACTION_FAILED_MESSAGE,
    CANCELLED_MESSAGE,
    MEMORY_RESET_MESSAGE,
    NOT_HEARD_MESSAGE,
    NOT_MATCHED_MESSAGE,
    NOTHING_SAID_MESSAGE,
    SOURCE_PTT,
    SOURCE_WAKE,
    TIMEOUT_MESSAGE,
    ActionOutcome,
    ActionRequest,
    NluMode,
    Pipeline,
    inline_runner,
    mode_from_config,
)
from ayris.core.pipeline_states import ManualScheduler, PipelineState
from ayris.nlu.context import DialogContext, PendingKind, PendingRequest
from ayris.nlu.llm.base import (
    NOT_CONFIGURED_MESSAGE,
    FinishReason,
    LlmClient,
    LlmMessage,
    LlmResponse,
    LlmTool,
    LlmToolCall,
)
from ayris.nlu.llm.memory import DialogMemory, InMemoryStore
from ayris.nlu.llm.tools import CommandCard, SlotSpec
from ayris.nlu.matcher import Matcher, Trigger

pytestmark = pytest.mark.unit

Runner = Callable[[Callable[[], None]], None]


# --- фейки внешних швов -----------------------------------------------------


class Handle:
    """Заглушка ручки озвучки: барджа-ина в этих тестах нет."""

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def cancel(self) -> bool:
        return True


class FakeTts:
    """Копит сказанное; по флагу падает, чтобы проверить проглатывание ошибки."""

    def __init__(self, *, fail: bool = False) -> None:
        self.said: list[str] = []
        self._fail = fail

    def say(self, text: str) -> Handle:
        if self._fail:
            raise TtsError("нет голоса")
        self.said.append(text)
        return Handle()

    @property
    def last(self) -> str:
        return self.said[-1] if self.said else ""


class FakeStt:
    """Распознаватель на записанном результате — ничего не считает и не ходит."""

    def __init__(self, result: TranscriptResult) -> None:
        self._result = result

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        return self._result


class RecordingActions:
    """Исполнитель совпавшей команды: запоминает запрос, отдаёт заданный итог."""

    def __init__(self, outcome: ActionOutcome | None = None) -> None:
        self.seen: list[ActionRequest] = []
        self._outcome = outcome if outcome is not None else ActionOutcome(speak="Готово.")

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        self.seen.append(request)
        return self._outcome

    @property
    def calls(self) -> int:
        return len(self.seen)

    @property
    def last(self) -> ActionRequest:
        return self.seen[-1]


class RaisingActions:
    """Раннер, роняющий :class:`ActionError`: проверяет ветку ошибки действия."""

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        raise ActionError("сломалось", user_message="Команда сломалась.")


class ScriptedLlm(LlmClient):
    """Клиент на записанных ответах; последний повторяется, если их не хватило."""

    name: ClassVar[str] = "scripted"

    def __init__(self, *responses: LlmResponse) -> None:
        self._responses = list(responses) or [_text("Готово.")]
        self.calls = 0

    @property
    def configured(self) -> bool:
        return True

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


class ToolLlm(ScriptedLlm):
    """То же, но с tool calling — иначе пайплайн не предложит инструменты."""

    supports_tools: ClassVar[bool] = True


# --- фабрики окружения ------------------------------------------------------


def _text(text: str) -> LlmResponse:
    return LlmResponse(text=text, engine="scripted", finish_reason=FinishReason.STOP)


def _tool(*calls: LlmToolCall) -> LlmResponse:
    return LlmResponse(tool_calls=calls, engine="scripted", finish_reason=FinishReason.TOOL_CALLS)


def _library() -> Matcher:
    """Матчер знает ровно «открой браузер» (command_id 8) — остальное мимо."""
    return Matcher.from_triggers([Trigger(id=2, command_id=8, pattern="открой браузер")])


def _cards() -> list[CommandCard]:
    """Каталог для JSON-NLU и tool calling: «Громкость» со слотом value."""
    return [
        CommandCard(
            command_id=7,
            name="Громкость",
            description="громкость звука",
            phrases=("громкость 50",),
            slots=(SlotSpec(name="value", type="int"),),
        )
    ]


def _settings(**ai: object) -> Settings:
    return Settings.model_validate({"ai": ai})


def _good_audio() -> AudioBuffer:
    return AudioBuffer(pcm=b"\x00\x00")


def _context() -> DialogContext:
    """Контекст без обращения к WinAPI — окно всегда «нет»."""
    return DialogContext(window_probe=lambda: None)


def _build(
    *,
    settings: Settings,
    bus: EventBus | None = None,
    llm: LlmClient | None = None,
    actions: Callable[[ActionRequest], ActionOutcome] | None = None,
    stt: FakeStt | None = None,
    tts: FakeTts | None = None,
    context: DialogContext | None = None,
    memory: DialogMemory | None = None,
    catalog: Callable[[], Sequence[CommandCard]] | None = None,
    scheduler: ManualScheduler | None = None,
    runner: Runner = inline_runner,
) -> Pipeline:
    """Собрать пайплайн задачи 18 с моками и включённым LLM-швом задачи 63."""
    return Pipeline(
        bus if bus is not None else EventBus(),
        matcher=_library(),
        tts=tts,
        actions=actions,
        stt=stt,
        llm=llm,
        settings=settings,
        scheduler=scheduler if scheduler is not None else ManualScheduler(),
        runner=runner,
        context=context,
        memory=memory,
        catalog=catalog,
    )


# --- голосовой путь: активация, распознавание, таймауты ---------------------


def _voice(text: str, confidence: float = 0.9) -> FakeStt:
    return FakeStt(TranscriptResult(text=text, confidence=confidence))


def test_ptt_timeout_speaks_nothing_said() -> None:
    """Окно слушания по горячей клавише молча закрылось — «Ничего не услышала»."""
    tts, scheduler = FakeTts(), ManualScheduler()
    pipeline = _build(
        settings=_settings(fallback_to_llm=False),
        tts=tts,
        stt=_voice("открой браузер"),
        scheduler=scheduler,
    )
    session_id = pipeline.activate(source=SOURCE_PTT)
    assert session_id
    assert pipeline.state is PipelineState.LISTENING
    assert len(scheduler.pending) == 1
    assert scheduler.fire_all() == 1
    assert tts.last == NOTHING_SAID_MESSAGE
    assert pipeline.state is PipelineState.IDLE


def test_wake_timeout_is_silent() -> None:
    """Ложное срабатывание слова активации не должно ничего произносить."""
    tts, scheduler = FakeTts(), ManualScheduler()
    pipeline = _build(
        settings=_settings(fallback_to_llm=False),
        tts=tts,
        stt=_voice("открой браузер"),
        scheduler=scheduler,
    )
    pipeline.activate(source=SOURCE_WAKE)
    assert scheduler.fire_all() == 1
    assert tts.said == []


def test_transcribing_timeout_speaks_timeout() -> None:
    """Стадия распознавания просрочена — общий «Не успела»."""
    tts, scheduler = FakeTts(), ManualScheduler()
    captured: list[Callable[[], None]] = []
    pipeline = _build(
        settings=_settings(fallback_to_llm=False),
        tts=tts,
        stt=_voice("открой браузер"),
        scheduler=scheduler,
        runner=captured.append,
    )
    pipeline.activate(source=SOURCE_PTT)
    pipeline.submit_audio(_good_audio())
    assert pipeline.state is PipelineState.TRANSCRIBING
    assert len(scheduler.pending) == 1
    assert len(captured) == 1
    scheduler.fire_all()
    assert tts.last == TIMEOUT_MESSAGE
    assert pipeline.state is PipelineState.IDLE


def test_submit_good_audio_matches_command() -> None:
    """Распознанная фраза совпала с библиотекой: команда пошла, TranscriptReady был."""
    bus = EventBus()
    heard: list[str] = []
    bus.subscribe(TranscriptReady, lambda event: heard.append(event.text), weak=False)
    actions = RecordingActions()
    pipeline = _build(
        settings=_settings(fallback_to_llm=False),
        bus=bus,
        actions=actions,
        stt=_voice("открой браузер"),
    )
    pipeline.activate(source=SOURCE_PTT)
    pipeline.submit_audio(_good_audio())
    assert heard == ["открой браузер"]
    assert actions.calls == 1
    assert actions.last.command_id == 8
    assert pipeline.state is PipelineState.IDLE


def test_submit_empty_transcription_is_not_heard() -> None:
    """Пустой текст распознавания — честное «Не расслышала», модель ни при чём."""
    tts = FakeTts()
    pipeline = _build(settings=_settings(fallback_to_llm=False), tts=tts, stt=_voice(""))
    pipeline.activate(source=SOURCE_PTT)
    pipeline.submit_audio(_good_audio())
    assert tts.last == NOT_HEARD_MESSAGE


def test_submit_low_confidence_is_not_heard() -> None:
    """Уверенность ниже порога (0.4 по умолчанию) — фраза отброшена."""
    tts = FakeTts()
    pipeline = _build(settings=_settings(fallback_to_llm=False), tts=tts, stt=_voice("привет", 0.1))
    pipeline.activate(source=SOURCE_PTT)
    pipeline.submit_audio(_good_audio())
    assert tts.last == NOT_HEARD_MESSAGE


def test_submit_none_audio_is_not_heard() -> None:
    """Пустой сегмент под открытой сессией — «Не расслышала», возврат в покой."""
    tts = FakeTts()
    pipeline = _build(settings=_settings(fallback_to_llm=False), tts=tts, stt=_voice("x"))
    pipeline.activate(source=SOURCE_PTT)
    assert pipeline.submit_audio(None) == ""
    assert tts.last == NOT_HEARD_MESSAGE
    assert pipeline.state is PipelineState.IDLE


def test_submit_without_session_is_dropped() -> None:
    """Фраза без открытой сессии — кто-то говорит в комнате, аудио отбрасывается."""
    tts = FakeTts()
    pipeline = _build(settings=_settings(fallback_to_llm=False), tts=tts, stt=_voice("x"))
    assert pipeline.submit_audio(_good_audio()) == ""
    assert tts.said == []


# --- run_text: пустой ввод и сухой прогон -----------------------------------


def test_empty_text_is_rejected() -> None:
    """Пустая строка не открывает сессию — сразу ошибка «empty text»."""
    pipeline = _build(settings=_settings(fallback_to_llm=False))
    result = pipeline.run_text("   ")
    assert result.outcome is ExecutionResult.ERROR
    assert result.error == "empty text"


def test_dry_run_matches_without_executing() -> None:
    """«Сухой прогон»: команда выбрана и записана, но не выполняется."""
    actions = RecordingActions()
    pipeline = _build(settings=_settings(fallback_to_llm=False), actions=actions)
    result = pipeline.run_text("открой браузер", execute=False)
    assert result.outcome is ExecutionResult.OK
    assert result.command_id == 8
    assert actions.calls == 0


# --- разговорные ветки: отмена, повтор, ответ на вопрос ----------------------


def test_cancel_phrase_closes_silently() -> None:
    """«Отмена» закрывает сессию молча: отмена — не ответ."""
    pipeline = _build(settings=_settings(fallback_to_llm=False), context=_context())
    result = pipeline.run_text("отмена")
    assert result.outcome is ExecutionResult.CANCELLED
    assert result.spoken == ""


def test_repeat_with_nothing_to_repeat() -> None:
    """«Повтори» без предыдущего ответа — честное «Мне нечего повторить»."""
    pipeline = _build(settings=_settings(fallback_to_llm=False), context=_context())
    result = pipeline.run_text("повтори")
    assert result.outcome is ExecutionResult.OK
    assert result.spoken == "Мне нечего повторить."


def test_pending_confirm_yes_runs_command() -> None:
    """«Да» на открытый вопрос-подтверждение запускает предзаполненную команду."""
    context = _context()
    context.set_pending(
        PendingRequest(kind=PendingKind.CONFIRM, question="Точно?", command_id=8, intent="open")
    )
    actions = RecordingActions()
    pipeline = _build(settings=_settings(fallback_to_llm=False), context=context, actions=actions)
    result = pipeline.run_text("да")
    assert result.outcome is ExecutionResult.OK
    assert result.command_id == 8
    assert actions.last.confirmed is True


def test_pending_confirm_no_declines() -> None:
    """«Нет» отклоняет вопрос: «Отменено», команда не запускается."""
    context = _context()
    context.set_pending(
        PendingRequest(kind=PendingKind.CONFIRM, question="Точно?", command_id=8, intent="open")
    )
    actions = RecordingActions()
    pipeline = _build(settings=_settings(fallback_to_llm=False), context=context, actions=actions)
    result = pipeline.run_text("нет")
    assert result.outcome is ExecutionResult.ERROR
    assert result.spoken == CANCELLED_MESSAGE
    assert actions.calls == 0


# --- память диалога: сброс по «забудь» --------------------------------------


def test_memory_reset_clears_and_confirms() -> None:
    """«Забудь» в режиме ИИ стирает память диалога и подтверждает это вслух."""
    memory = DialogMemory(store=InMemoryStore())
    memory.remember("привет", "здравствуй")
    pipeline = _build(
        settings=_settings(free_chat=True), llm=ScriptedLlm(_text("ответ")), memory=memory
    )
    result = pipeline.run_text("забудь")
    assert result.outcome is ExecutionResult.OK
    assert result.spoken == MEMORY_RESET_MESSAGE
    assert memory.messages() == []


# --- ошибки: действие, озвучка, пустой ответ, незаданная модель --------------


def test_action_error_is_spoken() -> None:
    """Раннер бросил ActionError — пайплайн ловит и произносит его сообщение."""
    pipeline = _build(settings=_settings(fallback_to_llm=False), actions=RaisingActions())
    result = pipeline.run_text("открой браузер")
    assert result.outcome is ExecutionResult.ERROR
    assert result.spoken == "Команда сломалась."


def test_action_failure_without_speech_uses_default() -> None:
    """Действие упало молча — вместо тишины «Не получилось выполнить»."""
    actions = RecordingActions(ActionOutcome(result=ExecutionResult.ERROR, speak=""))
    pipeline = _build(settings=_settings(fallback_to_llm=False), actions=actions)
    result = pipeline.run_text("открой браузер")
    assert result.outcome is ExecutionResult.ERROR
    assert result.spoken == ACTION_FAILED_MESSAGE


def test_text_path_without_runner_defers_to_dispatcher() -> None:
    """Без встроенного раннера команда уже ушла событием — двойного запуска нет."""
    pipeline = _build(settings=_settings(fallback_to_llm=False), actions=None)
    result = pipeline.run_text("открой браузер")
    assert result.outcome is ExecutionResult.OK
    assert result.command_id == 8
    assert result.spoken == ""


def test_tts_error_does_not_lose_command() -> None:
    """Немой синтез — не повод терять выполненную команду: ответ в трейсе есть."""
    actions = RecordingActions()
    pipeline = _build(
        settings=_settings(fallback_to_llm=False), actions=actions, tts=FakeTts(fail=True)
    )
    result = pipeline.run_text("открой браузер")
    assert result.outcome is ExecutionResult.OK
    assert result.spoken == "Готово."


def test_empty_model_answer_is_unmatched() -> None:
    """Пустой ответ модели — честное «Не поняла», а не выдумка."""
    pipeline = _build(settings=_settings(free_chat=True), llm=ScriptedLlm(_text("")))
    result = pipeline.run_text("скажи что-нибудь")
    assert result.outcome is ExecutionResult.UNMATCHED
    assert result.spoken == NOT_MATCHED_MESSAGE


def test_missing_model_reports_not_configured() -> None:
    """Режим ИИ без модели каждую фразу отвечает «ИИ не настроен», не падая."""
    pipeline = _build(settings=_settings(free_chat=True), llm=None)
    result = pipeline.run_text("привет")
    assert result.outcome is ExecutionResult.ERROR
    assert result.spoken == NOT_CONFIGURED_MESSAGE


# --- JSON-NLU и tool calling через каталог -----------------------------------


def test_nlu_maps_phrase_to_command() -> None:
    """Гибрид с пониманием: модель разобрала фразу в команду строгим JSON."""
    good = _text('{"command": "Громкость", "params": {"value": 30}, "confidence": 0.9}')
    llm = ScriptedLlm(good)
    actions = RecordingActions()
    pipeline = _build(
        settings=_settings(llm_understanding=True, fallback_to_llm=False),
        llm=llm,
        actions=actions,
        catalog=_cards,
    )
    result = pipeline.run_text("сделай погромче на тридцать")
    assert llm.calls == 1
    assert actions.calls == 1
    assert actions.last.command_id == 7
    assert actions.last.slots == {"value": 30}
    assert actions.last.confirmed is False
    assert result.command_id == 7


def test_nlu_miss_falls_through_to_chat() -> None:
    """``{"command": null}`` — не команда: с фоллбеком фраза уходит в чат."""
    llm = ScriptedLlm(_text('{"command": null}'), _text("Не команда, но вот факт."))
    actions = RecordingActions()
    pipeline = _build(
        settings=_settings(llm_understanding=True, fallback_to_llm=True),
        llm=llm,
        actions=actions,
        catalog=_cards,
    )
    result = pipeline.run_text("посоветуй фильм")
    assert llm.calls == 2
    assert result.spoken == "Не команда, но вот факт."
    assert result.outcome is ExecutionResult.OK
    assert actions.calls == 0


def test_nlu_invalid_twice_is_honest_miss() -> None:
    """Кривой JSON даёт один переспрос, затем честное «не нашла» без фоллбека."""
    llm = ScriptedLlm(_text("совсем не json"), _text("и снова не json"))
    pipeline = _build(
        settings=_settings(llm_understanding=True, fallback_to_llm=False),
        llm=llm,
        catalog=_cards,
    )
    result = pipeline.run_text("посоветуй фильм")
    assert llm.calls == 2
    assert result.outcome is ExecutionResult.UNMATCHED
    assert result.spoken == NOT_MATCHED_MESSAGE


def test_command_tool_call_routes_like_a_match() -> None:
    """Tool call ``cmd_<id>`` едет тем же путём, что и совпадение матчера."""
    llm = ToolLlm(_tool(LlmToolCall(name="cmd_7", arguments={"value": 40})))
    actions = RecordingActions()
    pipeline = _build(settings=_settings(free_chat=True), llm=llm, actions=actions, catalog=_cards)
    result = pipeline.run_text("сделай громче")
    assert actions.calls == 1
    assert actions.last.command_id == 7
    assert actions.last.slots == {"value": 40}
    assert actions.last.confirmed is False
    assert result.command_id == 7


# --- поверхность пайплайна: занятость, режимы, настройки, подписки -----------


def test_busy_pipeline_refuses_second_activation() -> None:
    """Активация посреди распознавания предыдущей фразы отклоняется."""
    pipeline = _build(settings=_settings(fallback_to_llm=False))
    first = pipeline.activate(source=SOURCE_WAKE)
    assert first
    assert pipeline.busy is True
    assert pipeline.session_id == first
    assert pipeline.activate(source=SOURCE_WAKE) == ""
    assert pipeline.cancel() is True
    assert pipeline.cancel() is False


def test_mode_from_config_reads_three_toggles() -> None:
    """Свободный чат → AI; любой командный тумблер → HYBRID; иначе COMMANDS."""
    assert mode_from_config(_settings(free_chat=True)) is NluMode.AI
    assert mode_from_config(_settings(llm_understanding=True)) is NluMode.HYBRID
    assert mode_from_config(_settings(fallback_to_llm=True)) is NluMode.HYBRID
    assert mode_from_config(_settings(fallback_to_llm=False)) is NluMode.COMMANDS


def test_apply_settings_and_setters_are_live() -> None:
    """Смена настроек и швов на лету не роняет пайплайн; attach/detach идемпотентны."""
    pipeline = _build(settings=_settings(fallback_to_llm=False))
    assert pipeline.mode is NluMode.COMMANDS
    pipeline.apply_settings(_settings(free_chat=True))
    assert pipeline.mode is NluMode.AI
    pipeline.set_matcher(_library())
    pipeline.set_llm(ScriptedLlm(_text("ага")))
    pipeline.set_catalog(_cards)
    pipeline.set_gateway(None)
    pipeline.set_memory(DialogMemory(store=InMemoryStore()))
    pipeline.attach()
    pipeline.attach()
    pipeline.detach()
    pipeline.close()
    assert pipeline.traces() == ()


# --- менеджер настроек: diff, apply, подписка, перечитывание, слежение -------


def _tweaked(settings: Settings, **general: object) -> Settings:
    return settings.model_copy(update={"general": settings.general.model_copy(update=general)})


def test_diff_settings_splits_live_and_restart() -> None:
    """Diff раскладывает изменения на «сразу» и «нужен перезапуск» по scope."""
    base = Settings()
    changed = _tweaked(base, autostart=True, always_run_as_admin=True)
    diff = diff_settings(base, changed)
    assert bool(diff) is True
    assert len(diff) == 2
    assert diff.touches("general.autostart") is True
    assert diff.touches("general") is True
    assert diff.touches("voice.tts") is False
    assert [change.path for change in diff.live] == ["general.autostart"]
    assert [change.path for change in diff.restart_required] == ["general.always_run_as_admin"]
    assert diff.restart_scopes == frozenset({RestartScope.APP})
    assert "general.autostart" in diff.summary()


def test_diff_summary_truncates_over_five() -> None:
    """Пустой diff молчит, а больше пяти изменений — сводка обрывается «…и ещё N»."""
    base = Settings()
    empty = diff_settings(base, base)
    assert bool(empty) is False
    assert empty.summary() == "настройки не изменились"

    changed = base.model_copy(
        update={
            "general": base.general.model_copy(
                update={"autostart": True, "always_run_as_admin": True}
            ),
            "ai": base.ai.model_copy(
                update={
                    "temperature": 0.1,
                    "max_tokens": base.ai.max_tokens + 7,
                    "provider": "openai",
                    "request_timeout_sec": base.ai.request_timeout_sec + 5.0,
                }
            ),
        }
    )
    diff = diff_settings(base, changed)
    assert len(diff) > 5
    assert diff.summary().startswith("изменено: ")
    assert "и ещё" in diff.summary()


def test_config_manager_apply_and_subscribe(tmp_path: Path) -> None:
    """apply меняет живое поле и требующее перезапуска, извещая подписчиков."""
    manager = ConfigManager(tmp_path / "config.toml")
    assert repr(manager) == f"ConfigManager(path={str(manager.path)!r}, loaded=False)"
    manager.load()
    assert manager.path.is_file()
    assert manager.dropped_fields == ()
    assert "loaded=True" in repr(manager)

    seen: list[tuple[str, ...]] = []
    unsubscribe = manager.subscribe(lambda change: seen.append(change.paths))

    live = manager.apply({"ai.temperature": 0.5})
    assert live.paths == ("ai.temperature",)
    assert manager.pending_restarts == frozenset()

    restart = manager.apply({"ai.provider": "openai"})
    assert restart.restart_scopes == frozenset({RestartScope.LLM})
    assert manager.pending_restarts == frozenset({RestartScope.LLM})
    manager.acknowledge_restart(RestartScope.LLM)
    assert manager.pending_restarts == frozenset()

    assert bool(manager.apply({"ai.temperature": 0.5})) is False
    unsubscribe()
    manager.apply({"ai.temperature": 0.8})
    assert seen == [("ai.temperature",), ("ai.provider",)]


def test_apply_rejects_invalid_with_summary(tmp_path: Path) -> None:
    """Больше пяти неверных полей — apply падает ConfigError с обрезанным списком."""
    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    before = manager.settings
    payload = {
        "ai.temperature": "x",
        "voice.stt.min_confidence": 5.0,
        "ai.max_tokens": "y",
        "overlay.rotation_speed": "z",
        "voice.tts.speed": "q",
        "ai.request_timeout_sec": "w",
    }
    with pytest.raises(ConfigError) as excinfo:
        manager.apply(payload)
    assert "и ещё" in excinfo.value.user_message
    assert manager.settings is before  # ничего не сохранилось


def test_reload_and_watch_lifecycle(tmp_path: Path) -> None:
    """reload видит внешнюю правку, молчит без неё, а слежение снимается начисто."""
    path = tmp_path / "config.toml"
    manager = ConfigManager(path)
    manager.load()

    assert manager.reload() is None  # файл не трогали

    modified = manager.settings.model_copy(
        update={"ai": manager.settings.ai.model_copy(update={"temperature": 0.55})}
    )
    save_settings(modified, path)
    change = manager.reload()
    assert change is not None
    assert change.paths == ("ai.temperature",)
    assert manager.reload() is None  # повторный reload уже без изменений

    try:
        manager.start_watching(interval=60.0)
        manager.start_watching(interval=60.0)  # идемпотентно
    finally:
        manager.stop_watching()
        manager.stop_watching()  # второй вызов — ранний выход


def test_init_and_reset_config_manager(tmp_path: Path) -> None:
    """init_config ставит глобальный менеджер, reset_config_manager его снимает."""
    saved = config_module._manager
    try:
        manager = init_config(tmp_path / "config.toml", watch=False)
        assert config_module._manager is manager
        assert manager.path.is_file()
        reset_config_manager()
        assert config_module._manager is None
    finally:
        config_module._manager = saved


# --- жизненный цикл приложения: этапы, компоненты, живые настройки -----------


def _provider_diff(provider: str = "openai") -> ConfigChanged:
    """Diff, меняющий только провайдера LLM — scope LLM, автозапуск не трогает."""
    base = Settings()
    changed = base.model_copy(update={"ai": base.ai.model_copy(update={"provider": provider})})
    return diff_settings(base, changed)


def test_lifecycle_stage_order_and_dataclasses() -> None:
    """Порядок этапов — контракт; дефолты AppOptions и поля Component на месте."""
    stages = tuple(LifecycleStage)
    assert stages[0] is LifecycleStage.PATHS
    assert stages[-1] is LifecycleStage.GUI
    assert len(stages) == len(set(stages)) == 13

    options = AppOptions()
    assert options.profile is None
    assert options.portable is False
    assert options.watch_config is True
    assert options.single_instance is None

    component = Component(name="демо", stage=LifecycleStage.GUI, stop_timeout=0.0)
    assert component.start is None and component.stop is None
    assert component.stage is LifecycleStage.GUI

    error = AlreadyRunningError()
    assert "уже запущен" in error.user_message
    assert error.recoverable is False


def test_app_properties_guard_before_startup() -> None:
    """До запуска приложение пустое, а обращение к неготовым частям — честная ошибка."""
    app = AyrisApp(AppOptions(log_level="INFO"))
    assert app.running is False
    assert app.stages_started == ()
    assert app.pending_restarts == frozenset()
    assert app.options.log_level == "INFO"
    assert isinstance(app.bus, EventBus)
    assert repr(app) == "AyrisApp(running=False, stages=0)"

    with pytest.raises(AyrisError):
        _ = app.config
    with pytest.raises(AyrisError):
        _ = app.paths
    with pytest.raises(AyrisError):
        _ = app.database
    with pytest.raises(AyrisError):
        _ = app.repositories
    with pytest.raises(AyrisError):
        _ = app.state
    with pytest.raises(AyrisError):
        _ = app.profile
    with pytest.raises(AyrisError):
        _ = app.profile_manager
    with pytest.raises(AyrisError):
        _ = app.settings

    # Компонент, добавленный до запуска, только копится — стартовать некому.
    app.add_component(Component(name="later", stage=LifecycleStage.GUI, stop_timeout=0.0))
    assert app._components_of(LifecycleStage.GUI) == []

    # Снятие обработчика идемпотентно: второй вызов уже никого не ищет.
    unregister = app.register_restart_handler(RestartScope.LLM, lambda _settings: None)
    unregister()
    unregister()

    app.shutdown()  # ничего не запускали — тихий ранний выход
    assert app.running is False


def test_add_component_while_running_starts_it_immediately() -> None:
    """Компонент, добавленный на уже пройденном этапе, стартует сразу и виден в порядке."""
    app = AyrisApp(AppOptions(log_level="INFO"))
    # Имитируем состояние «этап GUI уже поднят», не запуская тяжёлый startup().
    app._running = True
    app._stages_started.append(LifecycleStage.GUI)

    started: list[str] = []
    first = Component(
        name="first",
        stage=LifecycleStage.GUI,
        start=lambda: started.append("first"),
        stop_timeout=0.0,
    )
    second = Component(
        name="second",
        stage=LifecycleStage.GUI,
        start=lambda: started.append("second"),
        stop_timeout=0.0,
    )
    app.add_component(first)
    app.add_component(second)

    assert started == ["first", "second"]
    # _components_of отдаёт в порядке остановки — новейшие первыми.
    assert app._components_of(LifecycleStage.GUI) == [second, first]


def test_config_event_runs_restart_handler_and_clears_pending() -> None:
    """Смена провайдера через шину зовёт обработчик перезапуска и гасит pending."""
    app = AyrisApp(AppOptions(log_level="INFO"))
    fired: list[Settings] = []
    app.register_restart_handler(RestartScope.LLM, lambda settings: fired.append(settings))

    diff = _provider_diff("openai")
    assert diff.restart_scopes == frozenset({RestartScope.LLM})
    app.bus.publish(ConfigChanged(diff=diff))

    assert len(fired) == 1
    assert fired[0].ai.provider == "openai"
    assert app.pending_restarts == frozenset()


def test_config_event_without_handler_leaves_scope_pending() -> None:
    """Без обработчика и при его падении scope остаётся в pending_restarts."""
    app = AyrisApp(AppOptions(log_level="INFO"))

    app.bus.publish(ConfigChanged(diff=_provider_diff("openai")))
    assert app.pending_restarts == frozenset({RestartScope.LLM})

    def boom(_settings: Settings) -> None:
        raise RuntimeError("не смог перезапуститься")

    app.register_restart_handler(RestartScope.LLM, boom)
    app.bus.publish(ConfigChanged(diff=_provider_diff("lmstudio")))
    assert app.pending_restarts == frozenset({RestartScope.LLM})
