"""История диалога для модели: окно по сообщениям и токенам, суммаризация, сброс.

Отдельно от :class:`~ayris.nlu.context.DialogContext` (тот помнит «его/её» и
последнюю команду от фразы к фразе): здесь — многоходовая переписка с моделью,
которую надо удержать в пределах контекста. Свежие реплики хранятся дословно в
окне ``max_turns`` / ``max_chars``; всё, что вытесняется, при включённой
суммаризации сворачивается отдельным дешёвым вызовом в бегущее резюме, а не
теряется. Память привязана к профилю и переживает перезапуск через
:class:`MemoryStore`; сбрасывается по «Айрис, забудь» (:func:`is_reset`) и по
таймауту сессии.

Поддерживает два режима (настройка ``ai.memory_mode``):
* ``sliding_window`` — классическое окно последних реплик (по умолчанию, для отката);
* ``memgpt`` — долговременная память MemGPT: факты о пользователе (Core Memory)
  плюс семантически индексированный архив всех прошлых разговоров через SQLite FTS5.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from ayris.nlu.llm.base import LlmMessage, LlmRole
from ayris.nlu.normalize import normalize

__all__ = [
    "DialogMemory",
    "InMemoryStore",
    "MemoryState",
    "MemoryStore",
    "SqliteMemoryStore",
    "Summarizer",
    "Turn",
    "extract_forget_fact",
    "extract_remember_fact",
    "is_query_facts",
    "is_reset",
    "summary_prompt",
]

_log = logging.getLogger("ayris.nlu.llm.memory")

#: Грубая оценка «символов на токен» для бюджета: без токенайзера модели считаем
#: по ней, с запасом в меньшую сторону.
_CHARS_PER_TOKEN: Final = 4

#: Нормализованные фразы, по которым память стирается целиком.
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

#: Стоп-слова для исключения из FTS5-поиска по архиву.
_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "айрис",
        "были",
        "было",
        "быть",
        "вчера",
        "где",
        "говорили",
        "для",
        "зачем",
        "или",
        "как",
        "когда",
        "меня",
        "пожалуйста",
        "помнишь",
        "почему",
        "про",
        "скажи",
        "тебя",
        "что",
        "это",
    }
)

#: Регулярное выражение для извлечения факта из «Запомни: ...» или «Запомни, что ...».
_FACT_PREFIX_RE: Final = re.compile(
    r"^(?:айрис\s*,?\s*)?(?:запомни|сохрани)(?:\s+факт:|\s+факт\b|\s+что\b|,\s*что\b|:)?\s*(.+)$",
    re.IGNORECASE,
)

#: Фразы запроса списка известных фактов.
_QUERY_FACTS_PHRASES: Final = (
    "что ты обо мне знаешь",
    "что ты обо мне помнишь",
    "что ты помнишь обо мне",
    "что ты знаешь обо мне",
    "что ты помнишь",
    "вспомни обо мне",
    "вспомни факты",
    "какие факты обо мне",
    "какие факты ты знаешь",
    "покажи воспоминания",
    "что у тебя в памяти",
)

#: Регулярное выражение для удаления факта «Забудь обо мне: ...».
_FORGET_FACT_RE: Final = re.compile(
    r"^(?:айрис\s*,?\s*)?(?:забудь\s+обо\s+мне|забудь\s+факт|удали\s+факт|удали\s+из\s+памяти)\s*[:,\s]\s*(.+)$",
    re.IGNORECASE,
)

#: Что даёт функция суммаризации: реплики на сжатие -> строка резюме.
Summarizer = Callable[[Sequence[LlmMessage]], str]


def is_reset(text: str) -> bool:
    """Просит ли пользователь забыть разговор целиком."""
    normalized = normalize(text)
    return any(phrase in normalized.text for phrase in _RESET_PHRASES)


def extract_remember_fact(text: str) -> str | None:
    """Извлечь факт из фразы вроде «Запомни: я люблю кофе» или «Запомни, что меня зовут Артём».

    Возвращает очищенный текст факта или None, если фраза не про сохранение факта.
    """
    stripped = text.strip()
    m = _FACT_PREFIX_RE.match(stripped)
    if not m:
        return None
    fact = m.group(1).strip()
    return fact if len(fact) >= 2 else None


def is_query_facts(text: str) -> bool:
    """Спрашивает ли пользователь, что ассистент о нём помнит."""
    norm = normalize(text).text
    return any(p in norm for p in _QUERY_FACTS_PHRASES)


def extract_forget_fact(text: str) -> str | None:
    """Извлечь цель удаления из «Забудь обо мне: кофе» или «Удали из памяти кофе»."""
    stripped = text.strip()
    m = _FORGET_FACT_RE.match(stripped)
    if not m:
        return None
    target = m.group(1).strip()
    return target if target else None


@dataclass(frozen=True, slots=True)
class Turn:
    """Одна реплика в истории — пользователя или ассистента."""

    role: LlmRole
    text: str
    ts: str = ""

    def as_message(self) -> LlmMessage:
        if self.role is LlmRole.ASSISTANT:
            return LlmMessage.assistant(self.text)
        return LlmMessage.user(self.text)


@dataclass(frozen=True, slots=True)
class MemoryState:
    """Сохраняемое состояние памяти профиля: резюме, дословное окно реплик и факты."""

    summary: str = ""
    turns: tuple[Turn, ...] = ()
    facts: tuple[str, ...] = ()


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

    def save_fact(self, profile_id: int | None, fact: str) -> None:
        """Сохранить факт о пользователе в Core Memory."""

    def delete_fact(self, profile_id: int | None, pattern: str) -> bool:  # noqa: ARG002
        """Удалить факты, содержащие pattern."""
        return False

    def list_facts(self, profile_id: int | None) -> list[str]:  # noqa: ARG002
        """Получить все факты профиля."""
        return []

    def archive_turn(
        self,
        profile_id: int | None,
        user_text: str,
        assistant_text: str,
        ts: str | None = None,
    ) -> None:
        """Добавить завершённый диалог в долговременный архив."""

    def search_archive(
        self,
        profile_id: int | None,  # noqa: ARG002
        query: str,  # noqa: ARG002
        limit: int = 3,  # noqa: ARG002
        exclude: Sequence[str] = (),  # noqa: ARG002
    ) -> list[tuple[str, str, str]]:
        """Найти релевантные реплики в архиве (ts, user, assistant)."""
        return []


class InMemoryStore(MemoryStore):
    """Стор в памяти процесса — для тестов и для работы без базы."""

    def __init__(self) -> None:
        self._states: dict[int | None, MemoryState] = {}
        self._facts: dict[int | None, list[str]] = {}
        self._archive: dict[int | None, list[tuple[str, str, str]]] = {}

    def load(self, profile_id: int | None) -> MemoryState:
        base = self._states.get(profile_id, MemoryState())
        facts = tuple(self._facts.get(profile_id, list(base.facts)))
        return MemoryState(summary=base.summary, turns=base.turns, facts=facts)

    def save(self, profile_id: int | None, state: MemoryState) -> None:
        self._states[profile_id] = state
        self._facts[profile_id] = list(state.facts)

    def clear(self, profile_id: int | None) -> None:
        self._states.pop(profile_id, None)
        self._facts.pop(profile_id, None)
        self._archive.pop(profile_id, None)

    def save_fact(self, profile_id: int | None, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        facts = self._facts.setdefault(profile_id, [])
        if fact not in facts:
            facts.append(fact)

    def delete_fact(self, profile_id: int | None, pattern: str) -> bool:
        pattern = pattern.strip().lower()
        if not pattern:
            return False
        facts = self._facts.get(profile_id, [])
        kept = [f for f in facts if pattern not in f.lower()]
        removed = len(kept) != len(facts)
        self._facts[profile_id] = kept
        return removed

    def list_facts(self, profile_id: int | None) -> list[str]:
        return list(self._facts.get(profile_id, []))

    def archive_turn(
        self, profile_id: int | None, user_text: str, assistant_text: str, ts: str | None = None
    ) -> None:
        stamp = ts or datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        self._archive.setdefault(profile_id, []).append((stamp, user_text, assistant_text))

    def search_archive(
        self, profile_id: int | None, query: str, limit: int = 3, exclude: Sequence[str] = ()
    ) -> list[tuple[str, str, str]]:
        words = [
            w.lower()
            for w in re.findall(r"\w+", query)
            if len(w) >= 3 and w.lower() not in _STOP_WORDS
        ]
        if not words:
            return []
        exclude_set = set(exclude)
        results: list[tuple[str, str, str]] = []

        def _match_word(w: str, target: str) -> bool:
            if w in target:
                return True
            # Для русского языка ищем основу слова без окончания
            stem = w[: max(3, len(w) - 2)] if len(w) >= 5 else w
            return stem in target

        for ts, u, a in reversed(self._archive.get(profile_id, [])):
            if u in exclude_set or a in exclude_set:
                continue
            text = f"{u} {a}".lower()
            if any(_match_word(w, text) for w in words):
                results.append((ts, u, a))
                if len(results) >= limit:
                    break
        return results


class SqliteMemoryStore(MemoryStore):
    """Персистентное хранилище памяти в SQLite: окно, факты и FTS5-архив."""

    def __init__(self, db: Any = None, *, path: Path | str | None = None) -> None:
        self._db = db
        self._path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._db is not None and hasattr(self._db, "connect"):
            return self._db.connect()  # type: ignore[no-any-return]
        if self._conn is not None:
            return self._conn
        with self._lock:
            if self._conn is None:
                target = str(self._path) if self._path is not None else ":memory:"
                self._conn = sqlite3.connect(target, check_same_thread=False)
            return self._conn

    def _ensure_schema(self) -> None:
        try:
            conn = self._get_conn()
            with self._lock:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memgpt_state (
                        profile_id INTEGER PRIMARY KEY,
                        summary TEXT NOT NULL DEFAULT '',
                        turns_json TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS memgpt_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        profile_id INTEGER,
                        fact TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_memgpt_facts_profile
                    ON memgpt_facts (profile_id);

                    CREATE TABLE IF NOT EXISTS memgpt_archive (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        profile_id INTEGER,
                        ts TEXT NOT NULL,
                        user_text TEXT NOT NULL,
                        assistant_text TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_memgpt_archive_profile
                    ON memgpt_archive (profile_id, ts DESC);

                    CREATE VIRTUAL TABLE IF NOT EXISTS memgpt_archive_fts USING fts5(
                        user_text,
                        assistant_text,
                        content='memgpt_archive',
                        content_rowid='id'
                    );

                    CREATE TRIGGER IF NOT EXISTS memgpt_archive_ai
                    AFTER INSERT ON memgpt_archive BEGIN
                        INSERT INTO memgpt_archive_fts(rowid, user_text, assistant_text)
                        VALUES (new.id, new.user_text, new.assistant_text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS memgpt_archive_ad
                    AFTER DELETE ON memgpt_archive BEGIN
                        INSERT INTO memgpt_archive_fts(
                            memgpt_archive_fts, rowid, user_text, assistant_text
                        )
                        VALUES ('delete', old.id, old.user_text, old.assistant_text);
                    END;
                    """
                )
                conn.commit()
        except Exception as exc:
            _log.warning("не удалось инициализировать memgpt-таблицы в sqlite: %s", exc)

    def load(self, profile_id: int | None) -> MemoryState:
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            row = conn.execute(
                "SELECT summary, turns_json FROM memgpt_state WHERE profile_id = ?",
                (key,),
            ).fetchone()
            summary = str(row[0]) if row is not None else ""
            turns_data = json.loads(row[1]) if row is not None and row[1] else []
            turns = tuple(
                Turn(
                    role=LlmRole(item.get("role", "user")),
                    text=str(item.get("text", "")),
                    ts=str(item.get("ts", "")),
                )
                for item in turns_data
            )
            fact_rows = conn.execute(
                "SELECT fact FROM memgpt_facts "
                "WHERE (profile_id = ? OR profile_id IS NULL) ORDER BY id ASC",
                (key,),
            ).fetchall()
            facts = tuple(str(r[0]) for r in fact_rows)
            return MemoryState(summary=summary, turns=turns, facts=facts)
        except Exception as exc:
            _log.warning("ошибка загрузки памяти из sqlite: %s", exc)
            return MemoryState()

    def save(self, profile_id: int | None, state: MemoryState) -> None:
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            turns_payload = json.dumps(
                [
                    {
                        "role": (
                            turn.role.value if hasattr(turn.role, "value") else str(turn.role)
                        ),
                        "text": turn.text,
                        "ts": turn.ts,
                    }
                    for turn in state.turns
                ],
                ensure_ascii=False,
            )
            now = datetime.now(UTC).isoformat()
            with self._lock:
                conn.execute(
                    """
                    INSERT INTO memgpt_state (profile_id, summary, turns_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        summary=excluded.summary,
                        turns_json=excluded.turns_json,
                        updated_at=excluded.updated_at
                    """,
                    (key, state.summary, turns_payload, now),
                )
                conn.commit()
        except Exception as exc:
            _log.warning("ошибка сохранения памяти в sqlite: %s", exc)

    def clear(self, profile_id: int | None) -> None:
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            with self._lock:
                conn.execute("DELETE FROM memgpt_state WHERE profile_id = ?", (key,))
                conn.execute(
                    "DELETE FROM memgpt_facts WHERE profile_id = ? OR profile_id IS NULL",
                    (key,),
                )
                conn.execute(
                    "DELETE FROM memgpt_archive WHERE profile_id = ? OR profile_id IS NULL",
                    (key,),
                )
                conn.commit()
        except Exception as exc:
            _log.warning("ошибка очистки памяти в sqlite: %s", exc)

    def save_fact(self, profile_id: int | None, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            existing = conn.execute(
                "SELECT id FROM memgpt_facts "
                "WHERE (profile_id = ? OR profile_id IS NULL) AND lower(fact) = lower(?)",
                (key, fact),
            ).fetchone()
            if existing is not None:
                return
            now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
            with self._lock:
                conn.execute(
                    "INSERT INTO memgpt_facts (profile_id, fact, created_at) VALUES (?, ?, ?)",
                    (key, fact, now),
                )
                conn.commit()
        except Exception as exc:
            _log.warning("ошибка сохранения факта в sqlite: %s", exc)

    def delete_fact(self, profile_id: int | None, pattern: str) -> bool:
        pattern = pattern.strip()
        if not pattern:
            return False
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            pat_lower = pattern.lower()
            with self._lock:
                rows = conn.execute(
                    "SELECT id, fact FROM memgpt_facts "
                    "WHERE (profile_id = ? OR profile_id IS NULL)",
                    (key,),
                ).fetchall()
                to_delete = [r[0] for r in rows if pat_lower in str(r[1]).lower()]
                if not to_delete:
                    return False
                placeholders = ",".join("?" for _ in to_delete)
                conn.execute(
                    f"DELETE FROM memgpt_facts WHERE id IN ({placeholders})",
                    tuple(to_delete),
                )
                conn.commit()
                return True
        except Exception as exc:
            _log.warning("ошибка удаления факта в sqlite: %s", exc)
            return False

    def list_facts(self, profile_id: int | None) -> list[str]:
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            rows = conn.execute(
                "SELECT fact FROM memgpt_facts "
                "WHERE (profile_id = ? OR profile_id IS NULL) ORDER BY id ASC",
                (key,),
            ).fetchall()
            return [str(r[0]) for r in rows]
        except Exception as exc:
            _log.warning("ошибка чтения фактов из sqlite: %s", exc)
            return []

    def archive_turn(
        self, profile_id: int | None, user_text: str, assistant_text: str, ts: str | None = None
    ) -> None:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text and not assistant_text:
            return
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            stamp = ts or datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
            with self._lock:
                conn.execute(
                    "INSERT INTO memgpt_archive "
                    "(profile_id, ts, user_text, assistant_text) VALUES (?, ?, ?, ?)",
                    (key, stamp, user_text, assistant_text),
                )
                conn.commit()
        except Exception as exc:
            _log.warning("ошибка архивации реплики в sqlite: %s", exc)

    def search_archive(
        self, profile_id: int | None, query: str, limit: int = 3, exclude: Sequence[str] = ()
    ) -> list[tuple[str, str, str]]:
        words = [
            w for w in re.findall(r"\w+", query) if len(w) >= 3 and w.lower() not in _STOP_WORDS
        ]
        if not words:
            return []
        try:
            conn = self._get_conn()
            key = profile_id if profile_id is not None else 0
            match_str = " OR ".join(f'"{w}"' for w in words)
            rows = conn.execute(
                """
                SELECT m.ts, m.user_text, m.assistant_text
                FROM memgpt_archive_fts f
                JOIN memgpt_archive m ON f.rowid = m.id
                WHERE (m.profile_id = ? OR m.profile_id IS NULL)
                  AND memgpt_archive_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (key, match_str, limit * 2),
            ).fetchall()
            exclude_set = set(exclude)
            results: list[tuple[str, str, str]] = []
            for r in rows:
                ts, u, a = str(r[0]), str(r[1]), str(r[2])
                if u in exclude_set or a in exclude_set:
                    continue
                results.append((ts, u, a))
                if len(results) >= limit:
                    break
            return results
        except Exception as exc:
            _log.warning("ошибка FTS5-поиска в архиве sqlite: %s", exc)
            return []


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
        mode: str = "sliding_window",
        recall_limit: int = 3,
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
        self._mode = mode
        self._recall_limit = max(0, recall_limit)
        self._summary = ""
        self._turns: list[Turn] = []
        self._facts: list[str] = []
        self._seen = 0
        self._last_at = self._clock()
        self._load()

    # --- свойства и чтение --------------------------------------------------

    @property
    def mode(self) -> str:
        """Режим памяти: ``sliding_window`` или ``memgpt``."""
        return self._mode

    @property
    def summary(self) -> str:
        """Бегущее резюме вытесненных реплик — его пайплайн кладёт в system."""
        return self._summary

    def messages(self) -> list[LlmMessage]:
        """Дословное окно реплик для вставки между system-промптом и новой фразой."""
        return [turn.as_message() for turn in self._turns]

    def state(self) -> MemoryState:
        return MemoryState(
            summary=self._summary,
            turns=tuple(self._turns),
            facts=tuple(self._facts),
        )

    # --- долговременная память MemGPT ---------------------------------------

    def remember_fact(self, fact: str) -> None:
        """Сохранить факт в долговременную память пользователя (Core Memory)."""
        cleaned = fact.strip()
        if not cleaned:
            return
        if cleaned not in self._facts:
            self._facts.append(cleaned)
        if self._store is not None:
            self._store.save_fact(self._profile_id, cleaned)
        self._persist()

    def forget_fact(self, pattern: str) -> bool:
        """Удалить факт, содержащий pattern (регистронезависимо)."""
        pat = pattern.strip().lower()
        if not pat:
            return False
        kept = [f for f in self._facts if pat not in f.lower()]
        removed = len(kept) != len(self._facts)
        self._facts = kept
        if self._store is not None:
            store_removed = self._store.delete_fact(self._profile_id, pat)
            removed = removed or store_removed
        self._persist()
        return removed

    def list_facts(self) -> list[str]:
        """Список сохранённых фактов о пользователе."""
        return list(self._facts)

    def recall(self, query: str, limit: int | None = None) -> list[tuple[str, str, str]]:
        """Найти релевантные реплики в архиве прошлых разговоров."""
        if self._mode != "memgpt" or self._store is None:
            return []
        active_texts = [turn.text for turn in self._turns]
        lim = limit if limit is not None else self._recall_limit
        return self._store.search_archive(
            self._profile_id,
            query,
            limit=lim,
            exclude=active_texts,
        )

    def format_system_addon(self, query: str = "") -> str:
        """Сформировать добавку к системному промпту (резюме, факты, архивные воспоминания)."""
        parts: list[str] = []
        if self._summary:
            parts.append(f"Ранее в разговоре: {self._summary}")
        if self._mode == "memgpt":
            facts = self.list_facts()
            if facts:
                facts_lines = "\n".join(f"- {f}" for f in facts)
                parts.append(f"Долговременная память о пользователе:\n{facts_lines}")
            recalled = self.recall(query, limit=self._recall_limit)
            if recalled:
                recalled_lines = "\n".join(
                    f"- ({ts}) Пользователь: {u} | Айрис: {a}" for ts, u, a in recalled
                )
                parts.append(f"Найдено в архиве прошлых разговоров:\n{recalled_lines}")
        return "\n\n".join(parts)

    # --- запись -------------------------------------------------------------

    def remember(self, user_text: str, assistant_text: str) -> None:
        """Дописать обмен репликами и подрезать окно под лимиты."""
        now = self._clock()
        u = user_text.strip()
        a = assistant_text.strip()
        if u:
            self._turns.append(Turn(LlmRole.USER, u))
            self._seen += 1
        if a:
            self._turns.append(Turn(LlmRole.ASSISTANT, a))
            self._seen += 1
        self._compact()
        self._last_at = now
        if self._mode == "memgpt" and self._store is not None and u and a:
            self._store.archive_turn(self._profile_id, u, a)
        self._persist()

    def profile(self, profile_id: int | None) -> None:
        """Переключиться на память другого профиля, сохранив текущую."""
        if profile_id == self._profile_id:
            return
        self._persist()
        self._profile_id = profile_id
        self._summary = ""
        self._turns = []
        self._facts = []
        self._seen = 0
        self._load()

    def reset(self) -> None:
        """Стереть память — «Айрис, забудь» или таймаут сессии."""
        self._summary = ""
        self._turns = []
        self._facts = []
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

    def configure(
        self,
        *,
        max_turns: int | None = None,
        summarize_after: int | None = None,
        max_chars: int | None = None,
        mode: str | None = None,
        recall_limit: int | None = None,
    ) -> None:
        """Обновить лимиты реплик и режим памяти на лету."""
        if max_turns is not None:
            self._max_turns = max(0, max_turns)
        if summarize_after is not None:
            self._summarize_after = max(0, summarize_after)
        if max_chars is not None:
            self._max_chars = max(0, max_chars)
        if mode is not None:
            self._mode = mode
        if recall_limit is not None:
            self._recall_limit = max(0, recall_limit)
        self._compact()

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
        self._facts = list(state.facts)

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
