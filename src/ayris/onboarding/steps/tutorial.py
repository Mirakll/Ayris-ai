"""Шаг «Проба»: интерактивная тренировка на трёх настоящих командах.

Пользователю предлагают произнести три фразы из набора примеров — открыть
браузер, задать громкость, поставить таймер. Успех засчитывается не по факту
распознавания слов, а по тому, что команда действительно сработала: шаг слушает
шину и ждёт :class:`~ayris.core.events.IntentMatched` (единственный надёжный
сигнал «команда совпала», приходит и для голоса, и для набранного текста).
Распознанный текст показывается из :class:`~ayris.core.events.TranscriptReady`.

Если микрофон ещё не настроен, но есть канал ``submit_text``, ту же фразу можно
прогнать текстом кнопкой «Проверить» — тот же путь пайплайна, тот же сигнал
успеха. Любую пробу можно повторить или пропустить: шаг пропускаемый и никогда
не запирает мастер.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.events import EventBus, IntentMatched, TranscriptReady
from ayris.gui.theme import ThemeManager
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep
from ayris.utils.logger import get_logger

__all__ = ["TutorialStep", "TutorialTask"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TutorialTask:
    """Одна проба: что сказать и что при этом произойдёт."""

    phrase: str
    hint: str


#: Три пробы поверх команд из ``resources/examples``. Формулировки совпадают с
#: голосовыми триггерами примеров, чтобы проба реально запускала команду.
DEFAULT_TASKS: tuple[TutorialTask, ...] = (
    TutorialTask("Айрис, открой браузер", "Откроется браузер на стартовой странице."),
    TutorialTask("Айрис, громкость 50", "Громкость системы станет 50 %."),
    TutorialTask("Айрис, поставь таймер 5 минут", "Запустится таймер на пять минут."),
)


class _PipelineRelay(QObject):
    """Переносит события пайплайна с рабочего потока на поток GUI."""

    matched = Signal(object)
    transcript = Signal(object)


class TutorialStep(WizardStep):
    """Интерактивная проба: произнести три команды и увидеть, что они работают."""

    def __init__(
        self,
        theme: ThemeManager,
        bus: EventBus | None,
        submit_text: Callable[[str], None] | None,
        *,
        tasks: tuple[TutorialTask, ...] = DEFAULT_TASKS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = "tutorial"
        self.title = "Проба"
        self._bus = bus
        self._submit_text = submit_text
        self._tasks = tasks
        self._index = 0
        self._done = [False] * len(tasks)
        self._unsubs: list[Callable[[], None]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("Проба"))
        layout.addWidget(
            caption(
                "Произнесите фразу вслух после слова активации. Проба засчитается, "
                "когда команда действительно сработает."
            )
        )

        self._progress = QLabel("")
        self._progress.setProperty("role", "muted")
        layout.addWidget(self._progress)

        self._phrase = QLabel("")
        self._phrase.setProperty("role", "h2")
        self._phrase.setWordWrap(True)
        layout.addWidget(self._phrase)

        self._hint = QLabel("")
        self._hint.setProperty("role", "secondary")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self._recognised = QLabel("")
        self._recognised.setProperty("role", "muted")
        self._recognised.setWordWrap(True)
        layout.addWidget(self._recognised)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        row = QHBoxLayout()
        self._check_button = QPushButton("Проверить текстом")
        self._check_button.clicked.connect(self._check_by_text)
        self._check_button.setVisible(submit_text is not None)
        row.addWidget(self._check_button)
        row.addStretch(1)
        self._repeat_button = QPushButton("Повторить")
        self._repeat_button.clicked.connect(self._repeat)
        row.addWidget(self._repeat_button)
        self._skip_button = QPushButton("Пропустить пробу")
        self._skip_button.clicked.connect(self._skip_task)
        row.addWidget(self._skip_button)
        layout.addLayout(row)
        layout.addStretch(1)

        self._relay = _PipelineRelay(self)
        self._relay.matched.connect(self._on_matched)
        self._relay.transcript.connect(self._on_transcript)

        self._show_task()

    # -- показ пробы -------------------------------------------------------

    def _show_task(self) -> None:
        total = len(self._tasks)
        if self._index >= total:
            self._progress.setText(f"Готово: {total} из {total}")
            self._phrase.setText("Все пробы пройдены 🎉")
            self._hint.setText("")
            self._recognised.setText("")
            self._status.setText("")
            self._repeat_button.setEnabled(False)
            self._skip_button.setEnabled(False)
            self._check_button.setEnabled(False)
            self.completion_changed.emit()
            return
        task = self._tasks[self._index]
        self._progress.setText(f"Проба {self._index + 1} из {total}")
        self._phrase.setText(f"«{task.phrase}»")
        self._hint.setText(task.hint)
        self._recognised.setText("")
        self._status.setText("Слушаю…" if self._bus is not None else "Скажите фразу вслух.")
        self._repeat_button.setEnabled(True)
        self._skip_button.setEnabled(True)
        self._check_button.setEnabled(self._submit_text is not None)

    def _advance(self) -> None:
        self._index += 1
        self._show_task()

    def _repeat(self) -> None:
        self._recognised.setText("")
        self._status.setText("Слушаю…" if self._bus is not None else "Скажите фразу вслух.")

    def _skip_task(self) -> None:
        self._advance()

    def _check_by_text(self) -> None:
        if self._submit_text is None or self._index >= len(self._tasks):
            return
        phrase = self._tasks[self._index].phrase
        self._status.setText("Проверяю…")
        try:
            self._submit_text(phrase)
        except Exception as exc:
            _log.exception("не удалось прогнать пробу текстом")
            self._status.setText(f"Не удалось проверить: {exc}")

    # -- события пайплайна -------------------------------------------------

    def _on_matched(self, _event: IntentMatched) -> None:
        if self._index >= len(self._tasks):
            return
        self._done[self._index] = True
        self._status.setText("Сработало ✓")
        self._advance()

    def _on_transcript(self, event: TranscriptReady) -> None:
        if event.text:
            self._recognised.setText(f"Распознано: {event.text}")

    # -- контракт шага -----------------------------------------------------

    def is_complete(self) -> bool:
        return all(self._done)

    def activate(self) -> None:
        if self._bus is not None and not self._unsubs:
            self._unsubs = [
                self._bus.subscribe(IntentMatched, self._relay.matched.emit),
                self._bus.subscribe(TranscriptReady, self._relay.transcript.emit),
            ]

    def deactivate(self) -> None:
        self._unsubscribe()

    def teardown(self) -> None:
        self._unsubscribe()

    def _unsubscribe(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                _log.exception("не удалось отписаться от событий пайплайна")
        self._unsubs = []
