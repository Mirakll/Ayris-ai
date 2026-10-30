"""Тесты долговременной памяти MemGPT (Core Memory + Archival Memory FTS5).

Проверяет:
- SqliteMemoryStore: сохранение фактов, состояние диалога, FTS5 полнотекстовый поиск.
- InMemoryStore: реализация протокола для тестов.
- DialogMemory: режимы sliding_window (откат) и memgpt (расширенная память).
- Интерактивные команды: «Запомни: ...», «Что ты обо мне знаешь?», «Забудь факт ...».
- Интеграция с Pipeline: перехват команд и подмешивание контекста в системный промпт.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from ayris.core.config import AiConfig, Settings
from ayris.core.events import EventBus
from ayris.core.pipeline import ExecutionResult, Pipeline
from ayris.nlu.llm.base import LlmClient, LlmMessage, LlmResponse
from ayris.nlu.llm.memory import (
    DialogMemory,
    InMemoryStore,
    MemoryState,
    SqliteMemoryStore,
    Turn,
    extract_forget_fact,
    extract_remember_fact,
    is_query_facts,
)

pytestmark = pytest.mark.unit


def _chat_response(text: str) -> LlmResponse:
    return LlmResponse(text=text)


class ScriptedLlm(LlmClient):
    def __init__(self, *responses: LlmResponse) -> None:
        self.responses = list(responses)
        self.last_prompt: list[LlmMessage] = []
        self._calls = 0

    @property
    def configured(self) -> bool:
        return True

    def complete(self, messages: Any, tools: Any = (), **kwargs: Any) -> LlmResponse:
        self.last_prompt = list(messages)
        self._calls += 1
        if self.responses:
            return self.responses.pop(0)
        return _chat_response("Ответ по умолчанию")


def _make_pipeline(
    memory: DialogMemory,
    llm: LlmClient | None = None,
    mode: str = "memgpt",
) -> tuple[Pipeline, list[str]]:
    bus = EventBus()
    spoken: list[str] = []

    class DummyHandle:
        def wait(self, timeout: float | None = None) -> bool:
            return True

        def stop(self) -> None:
            pass

    class DummyTts:
        def say(self, text: str, **_: Any) -> Any:
            spoken.append(text)
            return DummyHandle()

        def stop(self) -> None:
            pass

    settings = Settings(
        ai=AiConfig(
            mode="llm",
            memory_mode=mode,  # type: ignore[arg-type]
            long_term_recall_limit=3,
        )
    )

    pipeline = Pipeline(
        bus=bus,
        settings=settings,
        tts=DummyTts(),  # type: ignore[arg-type]
        memory=memory,
    )
    if llm is not None:
        pipeline.set_llm(llm)
    return pipeline, spoken


# ----------------------------------------------------------------------
# Хелперы извлечения команд памяти
# ----------------------------------------------------------------------


def test_extract_remember_fact() -> None:
    assert extract_remember_fact("Запомни: меня зовут Артём") == "меня зовут Артём"
    assert extract_remember_fact("Айрис, запомни факт я люблю кофе") == "я люблю кофе"
    assert extract_remember_fact("запомни, что мой кот рыжий") == "мой кот рыжий"
    assert (
        extract_remember_fact("сохрани факт: проект называется Ayris") == "проект называется Ayris"
    )
    assert extract_remember_fact("просто фраза без команды") is None


def test_is_query_facts() -> None:
    assert is_query_facts("Что ты обо мне знаешь?") is True
    assert is_query_facts("вспомни обо мне") is True
    assert is_query_facts("какие факты обо мне сохранены") is True
    assert is_query_facts("что ты помнишь") is True
    assert is_query_facts("расскажи анекдот") is False


def test_extract_forget_fact() -> None:
    assert extract_forget_fact("Забудь факт: кофе") == "кофе"
    assert extract_forget_fact("удали факт про кота") == "про кота"
    assert extract_forget_fact("забудь факт день рождения") == "день рождения"
    assert extract_forget_fact("забудь все") is None  # это сброс диалога is_reset


# ----------------------------------------------------------------------
# SqliteMemoryStore и FTS5 поиск
# ----------------------------------------------------------------------


def test_sqlite_store_facts_and_archive() -> None:
    conn = sqlite3.connect(":memory:")
    store = SqliteMemoryStore(conn)

    # Core Memory: добавление и листинг
    store.save_fact(1, "Пользователь живёт в Москве")
    store.save_fact(1, "Любимый язык программирования — Python")
    assert store.list_facts(1) == [
        "Пользователь живёт в Москве",
        "Любимый язык программирования — Python",
    ]

    # Удаление факта (регистронезависимо для кириллицы)
    assert store.delete_fact(1, "москве") is True
    assert store.list_facts(1) == ["Любимый язык программирования — Python"]

    # Архив реплик и FTS5 BM25 поиск
    store.archive_turn(1, "user", "Как запустить квантовый симулятор?")
    store.archive_turn(1, "assistant", "Для квантового симулятора используйте Qiskit или Cirq.")
    store.archive_turn(1, "user", "Купи хлеб и молоко.")
    store.archive_turn(1, "assistant", "Добавила хлеб и молоко в список покупок.")

    # Поиск по архиву
    results = store.search_archive(1, "квантовый симулятор", limit=2)
    assert len(results) >= 1
    found_texts = " ".join(f"{u} {a}" for ts, u, a in results)
    assert "квантовый" in found_texts.lower()

    # Поиск несуществующего
    empty_res = store.search_archive(1, "космодром байконур", limit=2)
    assert len(empty_res) == 0


def test_sqlite_store_state_roundtrip() -> None:
    conn = sqlite3.connect(":memory:")
    store = SqliteMemoryStore(conn)

    store.save_fact(1, "Факт 1")
    store.save_fact(1, "Факт 2")
    turns = (
        Turn(role="user", text="Привет", ts="2026-01-01T10:00:00"),  # type: ignore[arg-type]
        Turn(role="assistant", text="Приветствую!", ts="2026-01-01T10:00:01"),  # type: ignore[arg-type]
    )
    state = MemoryState(summary="Разговор о разработке", turns=turns)

    store.save(1, state)
    loaded = store.load(1)
    assert loaded.summary == "Разговор о разработке"
    assert len(loaded.turns) == 2
    assert loaded.turns[0].text == "Привет"
    assert loaded.facts == ("Факт 1", "Факт 2")


# ----------------------------------------------------------------------
# DialogMemory: Режим sliding_window vs memgpt
# ----------------------------------------------------------------------


def test_dialog_memory_sliding_window_ignores_memgpt() -> None:
    store = InMemoryStore()
    memory = DialogMemory(store=store, mode="sliding_window")

    # Добавляем факт в хранилище напрямую
    store.save_fact(None, "Секретный факт")

    # В режиме sliding_window системный аддон НЕ должен содержать факты
    addon = memory.format_system_addon("Любой запрос")
    assert "Долговременная память о пользователе" not in addon
    assert "Секретный факт" not in addon


def test_dialog_memory_memgpt_addons_and_recall() -> None:
    store = InMemoryStore()
    # Добавляем прошлый разговор в долговременный архив
    store.archive_turn(
        None,
        "Где находится библиотека?",
        "Библиотека на углу улицы Ленина.",
        ts="2026-10-01 12:00",
    )

    memory = DialogMemory(store=store, mode="memgpt", recall_limit=2)
    memory.remember_fact("Пользователь пьет капучино с корицей")

    addon = memory.format_system_addon("как пройти в библиотеку")
    assert "Долговременная память о пользователе:" in addon
    assert "Пользователь пьет капучино с корицей" in addon
    assert "Найдено в архиве прошлых разговоров:" in addon
    assert "Библиотека" in addon


def test_dialog_memory_forget_fact() -> None:
    memory = DialogMemory(mode="memgpt")
    memory.remember_fact("Любимый цвет синий")
    memory.remember_fact("День рождения 15 августа")

    assert len(memory.list_facts()) == 2
    deleted = memory.forget_fact("синий")
    assert deleted is True
    facts = memory.list_facts()
    assert len(facts) == 1
    assert "синий" not in facts[0]


# ----------------------------------------------------------------------
# Pipeline: Обработка интерактивных команд и откат
# ----------------------------------------------------------------------


def test_pipeline_memgpt_interactive_commands() -> None:
    memory = DialogMemory(mode="memgpt")
    pipeline, spoken = _make_pipeline(memory=memory, mode="memgpt")

    # 1. Запомни факт
    res = pipeline.run_text("Запомни: я люблю читать фантастику")
    assert res.outcome == ExecutionResult.OK
    assert "Запомнила: я люблю читать фантастику" in spoken[-1]
    assert "я люблю читать фантастику" in memory.list_facts()

    # 2. Что ты обо мне знаешь?
    res = pipeline.run_text("Что ты обо мне знаешь?")
    assert res.outcome == ExecutionResult.OK
    assert "Вот что я о вас знаю:" in spoken[-1]
    assert "я люблю читать фантастику" in spoken[-1]

    # 3. Забудь факт
    res = pipeline.run_text("Забудь факт фантастику")
    assert res.outcome == ExecutionResult.OK
    assert "Забыла факт: фантастику" in spoken[-1]
    assert len(memory.list_facts()) == 0


def test_pipeline_sliding_window_does_not_intercept_remember() -> None:
    """В режиме sliding_window (откат) «Запомни: ...» не перехватывается, а уходит в LLM."""
    memory = DialogMemory(mode="sliding_window")
    llm = ScriptedLlm(_chat_response("Ответ от нейросети"))
    pipeline, spoken = _make_pipeline(memory=memory, llm=llm, mode="sliding_window")

    pipeline.run_text("Запомни: я люблю яблоки")
    # Должен был вызваться LLM, а не перехват пайплайна
    assert llm._calls == 1
    assert spoken[-1] == "Ответ от нейросети"
    # Память фактов осталась пустой
    assert len(memory.list_facts()) == 0


def test_pipeline_switch_mode_on_the_fly() -> None:
    """Проверка переключения режимов на лету через apply_settings (откат в один клик)."""
    memory = DialogMemory(mode="memgpt")
    llm = ScriptedLlm(_chat_response("Ответ LLM"))
    pipeline, spoken = _make_pipeline(memory=memory, llm=llm, mode="memgpt")

    # Сначала в режиме memgpt: перехватывает команду
    pipeline.run_text("Запомни: проект называется Ayris")
    assert "Запомнила: проект называется Ayris" in spoken[-1]
    assert len(memory.list_facts()) == 1

    # Пользователь переключает режим на sliding_window (откат)
    new_settings = Settings(
        ai=AiConfig(
            mode="llm",
            memory_mode="sliding_window",
        )
    )
    pipeline.apply_settings(new_settings)

    assert memory.mode == "sliding_window"

    # Теперь команда не перехватывается
    pipeline.run_text("Запомни: новая фраза")
    assert llm._calls == 1
    assert spoken[-1] == "Ответ LLM"
