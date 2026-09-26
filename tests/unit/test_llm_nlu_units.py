"""Задача 63: юнит-покрытие строительных блоков LLM-NLU — без пайплайна и сети.

Здесь по отдельности проверяются кирпичи, из которых собран LLM-шов задачи 63:
разбор кривого JSON-ответа модели, каталог команд и его рендер в инструменты,
история диалога (сброс, таймаут, вытеснение по символам, суммаризация), сборка
системных промптов из ресурсов с фоллбеком и предикаты режимов для будущей
вкладки ИИ (задача 64). Модель нигде не настоящая — все входы записаны заранее,
шлюз реестра проверяется на заглушке, так что тест детерминирован и молчалив.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayris.core.config import Settings
from ayris.core.pipeline import NluMode
from ayris.nlu.llm.base import LlmMessage, LlmRole
from ayris.nlu.llm.json_nlu import (
    DEFAULT_MIN_CONFIDENCE,
    NluDecision,
    interpret,
    parse_json_object,
    resolve_command,
)
from ayris.nlu.llm.memory import (
    DialogMemory,
    InMemoryStore,
    MemoryState,
    summary_prompt,
)
from ayris.nlu.llm.modes import MODE_INFO, ModeInfo, describe_mode, needs_llm
from ayris.nlu.llm.prompts import (
    COMMANDS_PLACEHOLDER,
    build_chat_prompt,
    build_nlu_prompt,
    load_template,
)
from ayris.nlu.llm.tools import (
    ActionReply,
    CommandCard,
    RegistryGateway,
    SlotSpec,
    command_tool_name,
    command_tools,
    parse_command_tool,
    render_catalog,
)

pytestmark = pytest.mark.unit

# --- json_nlu: разбор кривого ответа модели --------------------------------


def _cards() -> list[CommandCard]:
    """Каталог из двух команд: «Громкость» со слотом value и «Открыть браузер»."""
    return [
        CommandCard(
            command_id=7,
            name="Громкость",
            description="громкость звука",
            phrases=("громкость 50",),
            slots=(SlotSpec(name="value", type="int"),),
        ),
        CommandCard(command_id=8, name="Открыть браузер", phrases=("открой браузер",)),
    ]


def test_parse_json_object_reads_fenced_block() -> None:
    """```json … ``` разворачивается, а не ломает разбор."""
    assert parse_json_object('```json\n{"command": "Громкость"}\n```') == {"command": "Громкость"}


def test_parse_json_object_finds_object_amid_prose() -> None:
    """Первый сбалансированный {…} вытаскивается из болтовни, } внутри строки — не конец."""
    obj = parse_json_object('Вот ответ: {"command": "x", "note": "}"} — готово.')
    assert obj == {"command": "x", "note": "}"}


def test_parse_json_object_returns_none_without_object() -> None:
    assert parse_json_object("совсем не json") is None
    assert parse_json_object("") is None


@pytest.mark.parametrize(
    ("command", "expected_id"),
    [(7, 7), ("Громкость", 7), ("громкость", 7), ("cmd_8", 8)],
)
def test_resolve_command_hits(command: object, expected_id: int) -> None:
    """По id, по имени без регистра и по имени инструмента cmd_<id>."""
    card = resolve_command(_cards(), command)
    assert card is not None and card.command_id == expected_id


@pytest.mark.parametrize("command", [None, True, 3.5, "", "   ", "нет такой", 999, "cmd_999"])
def test_resolve_command_misses(command: object) -> None:
    """None, bool, дробь, пустышка, чужое имя и несуществующий id — не команда."""
    assert resolve_command(_cards(), command) is None


def test_nludecision_default_confidence_threshold() -> None:
    """Порог доверия — доля, а честный промах возвращает именно NluDecision."""
    assert 0.0 < DEFAULT_MIN_CONFIDENCE < 1.0
    assert isinstance(interpret('{"command": null}', _cards()), NluDecision)


def test_interpret_missing_command_field_is_invalid() -> None:
    decision = interpret('{"params": {}}', _cards())
    assert decision.resolved is False and decision.invalid is True
    assert "нет поля command" in decision.reason


def test_interpret_null_command_is_honest_miss() -> None:
    """Честный null — не команда и переспрашивать бесполезно (invalid=False)."""
    decision = interpret('{"command": null}', _cards())
    assert decision.resolved is False and decision.invalid is False


def test_interpret_unknown_command_is_retryable_invalid() -> None:
    """Выдуманное имя — повод для одного переспроса, а не окончательный промах."""
    decision = interpret('{"command": "Полёт на Марс"}', _cards())
    assert decision.resolved is False and decision.invalid is True
    assert decision.command_name == "Полёт на Марс"


def test_interpret_low_confidence_is_miss_not_action() -> None:
    """Сомнительная команда — «не поняла», а не не то действие."""
    decision = interpret('{"command": "Громкость", "confidence": 0.1}', _cards())
    assert decision.resolved is False and decision.invalid is False
    assert decision.command_id == 7 and decision.confidence == pytest.approx(0.1)


def test_interpret_keeps_only_declared_slots() -> None:
    """Чужие ключи в params модель выдумала — берём только объявленные слоты."""
    decision = interpret(
        '{"command": "Громкость", "params": {"value": 30, "мусор": 1}, "confidence": 0.9}',
        _cards(),
    )
    assert decision.resolved is True and decision.params == {"value": 30}


def test_interpret_params_not_mapping_are_dropped() -> None:
    decision = interpret(
        '{"command": "Открыть браузер", "params": [1, 2], "confidence": 1}', _cards()
    )
    assert decision.resolved is True and decision.params == {}


@pytest.mark.parametrize(
    ("payload", "resolved"),
    [
        ('{"command": "Громкость", "confidence": true}', True),  # bool -> 1.0
        ('{"command": "Громкость", "confidence": false}', False),  # bool -> 0.0
        ('{"command": "Громкость", "confidence": "0.9"}', True),  # строка -> 0.9
        ('{"command": "Громкость", "confidence": "ага"}', False),  # мусор -> 0.0
        ('{"command": "Громкость"}', True),  # поля нет -> уверена
        ('{"command": "Громкость", "confidence": [1]}', False),  # чужой тип -> 0.0
    ],
)
def test_interpret_confidence_coercion(payload: str, resolved: bool) -> None:
    """Уверенность приводится из bool/строки/пустоты, мусор — в ноль."""
    assert interpret(payload, _cards()).resolved is resolved


# --- memory: окно, сброс, таймаут, суммаризация ----------------------------


def test_remember_skips_blank_turns() -> None:
    """Пустые реплики не засоряют окно; без стора память ничего не персистит."""
    memory = DialogMemory(max_turns=10)
    memory.remember("привет", "   ")  # ассистент пуст
    memory.remember("   ", "и тебе")  # пользователь пуст
    assert [message.content for message in memory.messages()] == ["привет", "и тебе"]


def test_reset_clears_window_summary_and_store() -> None:
    store = InMemoryStore()
    memory = DialogMemory(store=store, profile_id=5)
    memory.remember("вопрос", "ответ")
    assert store.load(5).turns  # сохранилось
    memory.reset()
    assert memory.messages() == [] and memory.summary == ""
    assert store.load(5) == MemoryState()  # store.clear отработал


def test_profile_switch_persists_old_and_loads_new() -> None:
    store = InMemoryStore()
    memory = DialogMemory(store=store, profile_id=1)
    memory.remember("для первого", "ага")
    memory.profile(1)  # тот же профиль — ничего не делаем
    assert [message.content for message in memory.messages()] == ["для первого", "ага"]
    memory.profile(2)  # переключились — окно чистое
    assert memory.messages() == []
    memory.profile(1)  # вернулись — старое подгрузилось из стора
    assert [message.content for message in memory.messages()] == ["для первого", "ага"]


def test_maybe_expire_resets_only_after_timeout() -> None:
    now = [1000.0]
    memory = DialogMemory(session_ttl_s=60.0, clock=lambda: now[0])
    assert memory.maybe_expire() is False  # пусто — нечего сбрасывать
    memory.remember("привет", "здравствуй")
    now[0] += 30.0
    assert memory.maybe_expire() is False  # ещё в пределах окна
    now[0] += 40.0
    assert memory.maybe_expire() is True  # молчали дольше ttl — сброс
    assert memory.messages() == []


def test_maybe_expire_off_without_ttl() -> None:
    memory = DialogMemory(session_ttl_s=0.0)
    memory.remember("привет", "здравствуй")
    assert memory.maybe_expire() is False


def test_long_turns_evicted_by_char_budget() -> None:
    """Длинная реплика внутри окна всё равно режется по бюджету символов."""
    memory = DialogMemory(max_turns=100, max_chars=40)
    memory.remember("к" * 30, "о" * 30)  # 60 символов > 40 бюджета
    assert [message.content for message in memory.messages()] == ["о" * 30]
    assert memory.approx_tokens() == 30 // 4


def test_summary_absorbs_evicted_and_survives_summarizer_error() -> None:
    """Первое вытеснение сворачивается в резюме; падение суммаризатора его не портит."""
    calls: list[int] = []

    def flaky(messages: object) -> str:
        calls.append(len(messages))  # type: ignore[arg-type]
        if len(calls) == 1:
            return "Резюме первого вытеснения."
        raise RuntimeError("суммаризатор упал")

    memory = DialogMemory(summarizer=flaky, max_turns=2, summarize_after=2)
    memory.remember("а", "б")  # окно [а, б], seen=2
    memory.remember("в", "г")  # вытесняет [а, б] -> резюме №1 (без прежнего резюме)
    assert memory.summary == "Резюме первого вытеснения."
    memory.remember("д", "е")  # вытесняет [в, г]; в payload — прежнее резюме; суммаризатор падает
    assert memory.summary == "Резюме первого вытеснения."  # прежнее уцелело
    assert calls[-1] >= 3  # system + assistant(резюме) + вытесненные реплики


def test_empty_summary_is_not_stored() -> None:
    memory = DialogMemory(summarizer=lambda _messages: "   ", max_turns=2, summarize_after=2)
    memory.remember("а", "б")
    memory.remember("в", "г")  # вытеснение -> суммаризатор вернул пустое
    assert memory.summary == ""


def test_summary_prompt_builds_cheap_summarizer_input() -> None:
    evicted = [LlmMessage.user("а"), LlmMessage.assistant("б")]
    messages = summary_prompt("прежнее резюме", evicted)
    assert messages[0].role is LlmRole.SYSTEM
    assert any(message.content == "прежнее резюме" for message in messages)
    assert messages[-1].content == "б"
    assert len(summary_prompt("", evicted)) == 1 + len(evicted)  # без прежнего резюме


# --- modes: описания режимов и предикат воркера модели ---------------------


@pytest.mark.parametrize("mode", list(NluMode))
def test_describe_mode_covers_every_mode(mode: NluMode) -> None:
    info = describe_mode(mode)
    assert isinstance(info, ModeInfo)
    assert info.mode is mode
    assert info.label_ru and info.note_ru
    assert info.uses_matcher == mode.uses_matcher
    assert info.uses_llm == mode.uses_llm


def test_only_ai_mode_warns_about_lost_commands() -> None:
    assert describe_mode(NluMode.AI).warn_ru
    assert not describe_mode(NluMode.COMMANDS).warn_ru
    assert not describe_mode(NluMode.HYBRID).warn_ru
    assert set(MODE_INFO) == set(NluMode)


@pytest.mark.parametrize(
    ("ai", "expected"),
    [
        ({"fallback_to_llm": False}, False),  # только команды — воркер не нужен
        ({"fallback_to_llm": True}, True),  # гибрид с фоллбеком
        ({"llm_understanding": True, "fallback_to_llm": False}, True),  # гибрид-понимание
        ({"free_chat": True}, True),  # только ИИ
    ],
)
def test_needs_llm_matches_mode_policy(ai: dict[str, object], expected: bool) -> None:
    assert needs_llm(Settings.model_validate({"ai": ai})) is expected


# --- prompts: сборка системного промпта из ресурсов ------------------------


def test_load_template_falls_back_when_no_file(tmp_path: Path) -> None:
    """Ни переопределения, ни ресурса — берётся непустая встроенная заглушка."""
    text = load_template("chat", resources_dir=tmp_path)
    assert text and "markdown" in text.lower()


def test_override_wins_and_resource_is_the_fallback(tmp_path: Path) -> None:
    resources = tmp_path / "res"
    resources.mkdir()
    override = tmp_path / "over"
    override.mkdir()
    (resources / "chat.txt").write_text("из ресурса", encoding="utf-8")
    (override / "chat.txt").write_text("из профиля", encoding="utf-8")
    assert load_template("chat", resources_dir=resources, override_dir=override) == "из профиля"
    # для nlu переопределения нет — падаем в ресурс (ветка continue в цикле)
    (resources / "nlu.txt").write_text("нлу из ресурса", encoding="utf-8")
    assert load_template("nlu", resources_dir=resources, override_dir=override) == "нлу из ресурса"


def test_read_survives_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Нечитаемый файл шаблона — заглушка, а не исключение наружу."""
    import ayris.nlu.llm.prompts as prompts_mod

    (tmp_path / "chat.txt").write_text("ok", encoding="utf-8")

    def boom(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("нет доступа")

    monkeypatch.setattr(Path, "read_text", boom)
    assert load_template("chat", resources_dir=tmp_path) == prompts_mod._FALLBACKS["chat"]


def test_build_nlu_prompt_appends_commands_without_placeholder(tmp_path: Path) -> None:
    (tmp_path / "nlu.txt").write_text("Правила без маркера.", encoding="utf-8")
    prompt = build_nlu_prompt("Ты Айрис.", "- Громкость", resources_dir=tmp_path)
    assert "Ты Айрис." in prompt
    assert "Доступные команды:" in prompt and "- Громкость" in prompt


def test_build_nlu_prompt_fills_placeholder_and_empty_list(tmp_path: Path) -> None:
    (tmp_path / "nlu.txt").write_text("Список:\n" + COMMANDS_PLACEHOLDER, encoding="utf-8")
    filled = build_nlu_prompt("", "- Открыть браузер", resources_dir=tmp_path)
    assert "- Открыть браузер" in filled and COMMANDS_PLACEHOLDER not in filled
    empty = build_nlu_prompt("", "   ", resources_dir=tmp_path)
    assert "(команд нет)" in empty


def test_build_chat_prompt_joins_persona_and_template(tmp_path: Path) -> None:
    (tmp_path / "chat.txt").write_text("Правила чата.", encoding="utf-8")
    assert build_chat_prompt("Персона.", resources_dir=tmp_path) == "Персона.\n\nПравила чата."
    # пустая персона -> только шаблон (ветка _join с одной стороной)
    assert build_chat_prompt("   ", resources_dir=tmp_path) == "Правила чата."


# --- tools: каталог команд, инструменты, шлюз реестра ----------------------


def test_slot_as_property_carries_type_and_description() -> None:
    assert SlotSpec(name="value", type="int", description="уровень").as_property() == {
        "type": "integer",
        "description": "уровень",
    }
    assert SlotSpec(name="q").as_property() == {"type": "string"}


def test_command_card_render_lists_phrases_and_slots() -> None:
    line = CommandCard(
        command_id=7,
        name="Громкость",
        description="звук",
        phrases=("громче", "тише"),
        slots=(SlotSpec(name="value", type="int"),),
    ).render()
    assert "Громкость" in line and "звук" in line
    assert "например: громче; тише" in line and "value:int" in line


def test_render_catalog_respects_char_budget() -> None:
    """Бюджет символов режет список: попадает меньше карточек, чем есть."""
    cards = [CommandCard(command_id=i, name=f"Команда {i}", description="x" * 50) for i in range(6)]
    lines = render_catalog(cards, "команда", budget_chars=60).splitlines()
    assert 0 < len(lines) < len(cards)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("cmd_7", 7), ("cmd_0", 0), ("cmd_x", None), ("open", None), ("cmd_", None)],
)
def test_parse_command_tool(name: str, expected: int | None) -> None:
    assert parse_command_tool(name) is expected
    assert command_tool_name(7) == "cmd_7"


def test_command_tools_marks_required_and_uses_phrase() -> None:
    cards = [
        CommandCard(
            command_id=7,
            name="Громкость",
            phrases=("сделай громче",),
            slots=(SlotSpec(name="value", type="int", required=True),),
        )
    ]
    tool = command_tools(cards)[0]
    assert tool.name == "cmd_7"
    assert tool.parameters["required"] == ["value"]
    assert "сделай громче" in tool.description


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (ActionReply(ok=True, spoken="готово, шеф"), "готово, шеф"),
        (ActionReply(ok=True, detail="деталь"), "деталь"),
        (ActionReply(ok=False, error="boom"), "ошибка: boom"),
        (ActionReply(ok=True), "готово"),
    ],
)
def test_action_reply_as_tool_result(reply: ActionReply, expected: str) -> None:
    assert reply.as_tool_result == expected


class _Result:
    """Заглушка ExecutionResult: реестр отдаёт ровно то, что нужно run_action."""

    def __init__(self, *, ok: bool = True, message_ru: str = "ок", detail: str = "") -> None:
        self.ok = ok
        self.message_ru = message_ru
        self.detail = detail


class _FakeRegistry:
    """Реестр-заглушка: отдаёт готовый результат или бросает заданную ошибку."""

    def __init__(
        self, *, result: _Result | None = None, raises: Exception | None = None, has: bool = True
    ) -> None:
        self._result = result if result is not None else _Result()
        self._raises = raises
        self._has = has
        self.calls: list[tuple[str, dict[str, object], str, int | None]] = []

    def execute(
        self,
        name: str,
        params: object = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> _Result:
        self.calls.append((name, dict(params or {}), request_id, command_id))  # type: ignore[arg-type]
        if self._raises is not None:
            raise self._raises
        return self._result

    def has(self, name: str) -> bool:
        return self._has

    def describe_all(self) -> list[object]:
        return []


def test_instant_answer_requires_detector_kind_and_action() -> None:
    """Мгновенный ответ идёт только когда есть детектор, вид и действие-провайдер."""
    registry = _FakeRegistry(result=_Result(message_ru="Сейчас 14:30."))
    assert RegistryGateway(registry).instant_answer("сколько времени") is None  # детектора нет
    quiet = RegistryGateway(registry, detect=lambda _u: "")
    assert quiet.instant_answer("просто болтаю") is None  # детектор молчит
    missing = RegistryGateway(_FakeRegistry(has=False), detect=lambda _u: "time")
    assert missing.instant_answer("сколько времени") is None  # провайдер не зарегистрирован
    ok = RegistryGateway(registry, detect=lambda _u: "time", instant_action="TimeReading")
    assert ok.instant_answer("сколько времени") == "Сейчас 14:30."
    assert registry.calls[-1][0] == "TimeReading"


def test_run_action_maps_registry_outcomes() -> None:
    """Успех, отказ в подтверждении (опасное) и ошибка сводятся к своей реплике."""
    from ayris.core.errors import ActionError, ActionNotConfirmed

    done = RegistryGateway(
        _FakeRegistry(result=_Result(ok=True, message_ru="готово", detail="d"))
    ).run_action("EchoPing")
    assert done.ok and done.spoken == "готово" and done.detail == "d"

    refused = RegistryGateway(
        _FakeRegistry(raises=ActionNotConfirmed("нужно да", user_message="Не стираю без спроса."))
    ).run_action("WipeDisk")
    assert refused.ok is False and refused.dangerous is True
    assert refused.spoken == "Не стираю без спроса."

    failed = RegistryGateway(
        _FakeRegistry(raises=ActionError("boom", user_message="Не вышло."))
    ).run_action("Broken")
    assert failed.ok is False and failed.dangerous is False and failed.spoken == "Не вышло."
