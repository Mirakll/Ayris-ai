"""История диалога для модели: окно по сообщениям и токенам, суммаризация, сброс.

Отдельно от :class:`~ayris.nlu.context.DialogContext` (тот помнит «его/её» и
последнюю команду от фразы к фразе): здесь — многоходовая переписка с моделью,
которую надо удержать в пределах контекста. Свежие реплики хранятся дословно в
окне ``max_turns`` / ``max_chars``; всё, что вытесняется, при включённой
суммаризации сворачивается отдельным дешёвым вызовом в бегущее резюме, а не
теряется. Память привязана к профилю и переживает перезапуск через
:class:`MemoryStore`; сбрасывается по «Айрис, забудь» (:func:`is_reset`) и по
таймауту сессии.

Резюме отдаётся не сообщением, а строкой :attr:`DialogMemory.summary`, которую
пайплайн вкладывает в system-промпт: так у запроса остаётся ровно один system, и
провайдеры вроде Anthropic, у которых он отдельным полем, не спотыкаются о второй.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from ayris.nlu.llm.base import LlmMessage, LlmRole
from ayris.nlu.normalize import normalize

__all__ = [
    "DialogMemory",
    "InMemoryStore",
    "MemoryState",
    "MemoryStore",
    "Summarizer",
    "Turn",
    "is_reset",
    "summary_prompt",
]

_log = logging.getLogger("ayris.nlu.llm.memory")

#: Грубая оценка «символов на токен» для бюджета: без токенайзера модели считаем
#: по ней, с запасом в меньшую сторону.
_CHARS_PER_TOKEN: Final = 4

#: Нормализованные фразы, по которым память стирается.
_RESET_PHRASES: Final = (
    "айрис забудь",
    "забудь все",
    "забудь всё",
    "забудь",
    "начни сначала",
    "новый разговор",
)

#: Промпт для дешёвого вызова суммаризации вытесненных реплик.
_SUMMARY_SYSTEM: Final = (
    "Сожми переписку в короткое резюме на русском: только факты и решения, "
    "которые понадобятся дальше. Одно-два предложения, без markdown."
)

#: Что даёт функция суммаризации: реплики на сжатие -> строка резюме.
Summarizer = Callable[[Sequence[LlmMessage]], str]


def is_reset(text: str) -> bool:
    """Просит ли пользователь забыть разговор."""
    normalized = normalize(text)
    return any(phrase in normalized.text for phrase in _RESET_PHRASES)


@dataclass(frozen=True, slots=True)
class Turn:
    """Одна реплика в истории — пользователя или ассистента."""

    role: LlmRole
    text: str

    def as_message(self) -> LlmMessage:
        if self.role is LlmRole.ASSISTANT:
            return LlmMessage.assistant(self.text)
        return LlmMessage.user(self.text)


@dataclass(frozen=True, slots=True)
class MemoryState:
    """Сохраняемое состояние памяти профиля: резюме и дословное окно реплик."""

    summary: str = ""
    turns: tuple[Turn, ...] = ()


class MemoryStore:
    """Куда память профиля кладётся, чтобы пережить перезапуск.

    База — необязательна: без стора память живёт только в текущей сессии. Все
    реализации глотают ошибки хранилища и возвращают пустое состояние, потому что
    потеря истории диалога — не повод ронять разбор фразы.
    """

    def load(self, profile_id: int | None) -> MemoryState:  # pragma: no cover - протокол
        raise NotImplementedError

    def save(self, profile_id: int | None, state: MemoryState) -> None:  # pragma: no cover
        raise NotImplementedError

    def clear(self, profile_id: int | None) -> None:  # pragma: no cover - протокол
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """Стор в памяти процесса — для тестов и для работы без базы."""

    def __init__(self) -> None:
        self._states: dict[int | None, MemoryState] = {}

    def load(self, profile_id: int | None) -> MemoryState:
        return self._states.get(profile_id, MemoryState())

    def save(self, profile_id: int | None, state: MemoryState) -> None:
        self._states[profile_id] = state

    def clear(self, profile_id: int | None) -> None:
        self._states.pop(profile_id, None)


class DialogMemory:
    """Окно диалога с моделью: ограничение, суммаризация, сброс, персистентность."""

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        summarizer: Summarizer | None = None,
        max_turns: int = 10,
        summarize_after: int = 0,
        max_chars: int = 6000,
        profile_id: int | None = None,
        session_ttl_s: float = 0.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        from time import monotonic

        self._store = store
        self._summarizer = summarizer
        self._max_turns = max(0, max_turns)
        self._summarize_after = max(0, summarize_after)
        self._max_chars = max(0, max_chars)
        self._profile_id = profile_id
        self._session_ttl = max(0.0, session_ttl_s)
        self._clock = clock if clock is not None else monotonic
        self._summary = ""
        self._turns: list[Turn] = []
        self._seen = 0
        self._last_at = self._clock()
        self._load()

    # --- чтение -------------------------------------------------------------

    @property
    def summary(self) -> str:
        """Бегущее резюме вытесненных реплик — его пайплайн кладёт в system."""
        return self._summary

    def messages(self) -> list[LlmMessage]:
        """Дословное окно реплик для вставки между system-промптом и новой фразой."""
        return [turn.as_message() for turn in self._turns]

    def state(self) -> MemoryState:
        return MemoryState(summary=self._summary, turns=tuple(self._turns))

    # --- запись -------------------------------------------------------------

    def remember(self, user_text: str, assistant_text: str) -> None:
        """Дописать обмен репликами и подрезать окно под лимиты."""
        now = self._clock()
        if user_text.strip():
            self._turns.append(Turn(LlmRole.USER, user_text.strip()))
            self._seen += 1
        if assistant_text.strip():
            self._turns.append(Turn(LlmRole.ASSISTANT, assistant_text.strip()))
            self._seen += 1
        self._compact()
        self._last_at = now
        self._persist()

    def profile(self, profile_id: int | None) -> None:
        """Переключиться на память другого профиля, сохранив текущую."""
        if profile_id == self._profile_id:
            return
        self._persist()
        self._profile_id = profile_id
        self._summary = ""
        self._turns = []
        self._seen = 0
        self._load()

    def reset(self) -> None:
        """Стереть память — «Айрис, забудь» или таймаут сессии."""
        self._summary = ""
        self._turns = []
        self._seen = 0
        self._last_at = self._clock()
        if self._store is not None:
            self._store.clear(self._profile_id)

    def maybe_expire(self) -> bool:
        """Сбросить память, если сессия молчала дольше таймаута. ``True`` — сбросили."""
        if self._session_ttl <= 0.0:
            return False
        if not self._turns and not self._summary:
            return False
        if self._clock() - self._last_at < self._session_ttl:
            return False
        self.reset()
        return True

    # --- внутреннее ---------------------------------------------------------

    def _compact(self) -> None:
        if self._max_turns and len(self._turns) > self._max_turns:
            overflow = self._turns[: len(self._turns) - self._max_turns]
            self._turns = self._turns[len(self._turns) - self._max_turns :]
            self._absorb(overflow)
        # Даже внутри окна длинные реплики могут пробить бюджет символов.
        while self._max_chars and len(self._turns) > 1 and self._chars() > self._max_chars:
            self._absorb([self._turns.pop(0)])

    def _absorb(self, evicted: Sequence[Turn]) -> None:
        if not evicted:
            return
        if (
            self._summarizer is None
            or self._summarize_after <= 0
            or self._seen < self._summarize_after
        ):
            return
        payload: list[LlmMessage] = []
        if self._summary:
            payload.append(LlmMessage.system(_SUMMARY_SYSTEM))
            payload.append(LlmMessage.assistant(self._summary))
        payload.extend(turn.as_message() for turn in evicted)
        try:
            summary = self._summarizer(payload).strip()
        except Exception as exc:  # суммаризация не должна ронять разбор
            _log.warning("суммаризация памяти не удалась: %s", exc)
            return
        if summary:
            self._summary = summary[: self._max_chars] if self._max_chars else summary

    def _chars(self) -> int:
        return sum(len(turn.text) for turn in self._turns) + len(self._summary)

    def approx_tokens(self) -> int:
        """Грубая оценка занятого моделью контекста, для лимита в настройках."""
        return self._chars() // _CHARS_PER_TOKEN

    def _load(self) -> None:
        if self._store is None:
            return
        state = self._store.load(self._profile_id)
        self._summary = state.summary
        self._turns = list(state.turns)

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save(self._profile_id, self.state())


def summary_prompt(existing: str, evicted: Sequence[LlmMessage]) -> list[LlmMessage]:
    """Собрать вход для дешёвого вызова суммаризации (для обёртки в пайплайне)."""
    messages = [LlmMessage.system(_SUMMARY_SYSTEM)]
    if existing:
        messages.append(LlmMessage.assistant(existing))
    messages.extend(evicted)
    return messages
