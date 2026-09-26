"""Тестовая панель «спросить у модели»: ответ, время, токены, цена.

Панель вкладки «ИИ» (задача 64, §6): пользователь пишет вопрос, панель отправляет
его выбранной сейчас моделью и показывает ответ по мере генерации, время до первого
токена и общее время, число токенов и грубую цену запроса. Это пробник настроек, а
не разговор: запрос НЕ трогает историю диалога — панель получает готовый
:class:`TestRequest` от вкладки, где system-промпт собран без прошлых реплик, и после
ответа ничего никуда не сохраняет.

Сетевой вызов идёт не в UI-потоке: :class:`_StreamRunner` крутит ``client.stream`` на
демоне и шлёт фрагменты назад сигналами, которые Qt доставляет в поток панели. У
запроса есть отмена (:meth:`_StreamRunner.cancel` взводит событие, которое стрим
опрашивает между фрагментами), а клиент закрывается в ``finally``, чтобы отменённый
или упавший запрос не оставил открытый сокет.

Токены и цену считают :func:`resolve_usage` и :func:`price_for`: провайдер, который
не отдал счётчики, добивается эвристикой и помечается «≈», а локальные модели честно
показывают «без оплаты» вместо выдуманного нуля в долларах.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.errors import AyrisError
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import BusyIndicator
from ayris.nlu.llm.base import (
    NOT_CONFIGURED_MESSAGE,
    FinishReason,
    LlmClient,
    LlmDoneDelta,
    LlmMessage,
    LlmTextDelta,
    LlmUsage,
    LlmUsageDelta,
)
from ayris.nlu.llm.usage import price_for, resolve_usage
from ayris.utils.logger import get_logger

__all__ = ["LlmTestPanel", "TestRequest"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TestRequest:
    """Готовый одноразовый запрос к модели: клиент, сообщения и параметры.

    Собирает его вкладка, а не панель: только вкладка знает про фабрику клиентов,
    выбранного провайдера и system-промпт. ``messages`` — ровно system + вопрос, без
    истории: панель ничего не помнит между запросами.
    """

    client: LlmClient
    messages: Sequence[LlmMessage]
    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class _TestResult:
    """Итог одного тестового запроса — то, что панель показывает под ответом."""

    text: str
    total_s: float
    usage: LlmUsage
    estimated: bool
    cost_usd: float
    price_known: bool
    finish: FinishReason
    ttft_s: float | None = None


class _StreamRunner(QObject):
    """Крутит ``client.stream`` вне UI-потока и шлёт результат назад сигналами."""

    first_token = Signal(float)
    chunk = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Взвести отмену — стрим опрашивает её между фрагментами."""
        self._cancel.set()

    def reset(self) -> None:
        """Свежее событие отмены перед новым запросом."""
        self._cancel = threading.Event()

    def run(self, request: TestRequest) -> None:
        """Запустить запрос на демоне; ответ придёт сигналами в поток панели."""
        threading.Thread(target=self._run, args=(request,), daemon=True).start()

    def _run(self, request: TestRequest) -> None:
        start = perf_counter()
        ttft: float | None = None
        pieces: list[str] = []
        usage = LlmUsage()
        finish = FinishReason.STOP
        try:
            for delta in request.client.stream(
                request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                cancel=self._cancel.is_set,
            ):
                if isinstance(delta, LlmTextDelta):
                    if ttft is None:
                        ttft = perf_counter() - start
                        self.first_token.emit(ttft)
                    if delta.text:
                        pieces.append(delta.text)
                        self.chunk.emit(delta.text)
                elif isinstance(delta, LlmUsageDelta):
                    usage = delta.usage
                elif isinstance(delta, LlmDoneDelta):
                    finish = delta.finish_reason
        except AyrisError as exc:
            self.failed.emit(exc.user_message)
            return
        except Exception as exc:  # клиент может кинуть что угодно — UI не должен упасть
            _log.exception("тестовый запрос к модели упал")
            self.failed.emit(str(exc))
            return
        finally:
            request.client.close()
        completion = "".join(pieces)
        resolved, estimated = resolve_usage(request.provider, usage, request.messages, completion)
        price = price_for(request.provider, request.model)
        self.finished.emit(
            _TestResult(
                text=completion,
                total_s=perf_counter() - start,
                usage=resolved,
                estimated=estimated,
                cost_usd=price.cost(resolved),
                price_known=price.known,
                finish=finish,
                ttft_s=ttft,
            )
        )


class LlmTestPanel(QWidget):
    """Поле вопроса, ответ по мере генерации и строка со временем, токенами и ценой.

    Панель самодостаточна: как собрать запрос — знает вкладка, а панель лишь зовёт
    ``build_request`` и рисует ответ. Между запросами она ничего не хранит, поэтому
    один пробник не влияет на следующий и на настоящий диалог.
    """

    def __init__(
        self,
        theme: ThemeManager,
        *,
        build_request: Callable[[str], TestRequest],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._build_request = build_request
        self._running = False
        self.setProperty("transparent", True)
        self._runner = _StreamRunner(self)
        self._runner.first_token.connect(self._on_first_token)
        self._runner.chunk.connect(self._on_chunk)
        self._runner.finished.connect(self._on_finished)
        self._runner.failed.connect(self._on_failed)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))

        row = QHBoxLayout()
        row.setSpacing(self._theme.metric("spacing_sm"))
        self.question = QLineEdit()
        self.question.setPlaceholderText("Спросите модель, чтобы проверить настройки…")
        self.question.setAccessibleName("Вопрос модели")
        self.question.textChanged.connect(self._sync_ask_enabled)
        self.question.returnPressed.connect(self._ask)
        row.addWidget(self.question, 1)
        self.ask_button = QPushButton("Спросить")
        self.ask_button.setProperty("kind", "primary")
        self.ask_button.setEnabled(False)
        self.ask_button.clicked.connect(self._ask)
        row.addWidget(self.ask_button)
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        row.addWidget(self.cancel_button)
        self.busy = BusyIndicator(self._theme, active=False)
        self.busy.hide()
        row.addWidget(self.busy)
        layout.addLayout(row)

        self.answer = QPlainTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setAccessibleName("Ответ модели")
        self.answer.setPlaceholderText("Ответ появится здесь.")
        layout.addWidget(self.answer)

        self.stats = QLabel("")
        self.stats.setProperty("role", "muted")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)

    # -- взаимодействие -----------------------------------------------------

    def _sync_ask_enabled(self) -> None:
        ready = bool(self.question.text().strip()) and not self._running
        self.ask_button.setEnabled(ready)

    def _ask(self) -> None:
        text = self.question.text().strip()
        if not text or self._running:
            return
        try:
            request = self._build_request(text)
        except AyrisError as exc:
            self.stats.setText(exc.user_message)
            return
        if not request.client.configured:
            request.client.close()
            self.answer.setPlainText(NOT_CONFIGURED_MESSAGE)
            self.stats.setText("")
            return
        self.answer.clear()
        self.stats.setText("Жду ответ…")
        self._set_running(True)
        self._runner.reset()
        self._runner.run(request)

    def _cancel(self) -> None:
        self._runner.cancel()
        self.cancel_button.setEnabled(False)

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        self.busy.setVisible(running)
        self.busy.setActive(running)
        self.question.setReadOnly(running)
        self._sync_ask_enabled()

    # -- сигналы стрима ------------------------------------------------------

    def _on_first_token(self, _ttft: float) -> None:
        self.stats.setText("Печатает…")

    def _on_chunk(self, text: str) -> None:
        self.answer.moveCursor(QTextCursor.MoveOperation.End)
        self.answer.insertPlainText(text)

    def _on_finished(self, result: object) -> None:
        self._set_running(False)
        if not isinstance(result, _TestResult):
            return
        if result.text and not self.answer.toPlainText():
            self.answer.setPlainText(result.text)
        self.stats.setText(self._format_stats(result))

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.stats.setText(f"Не удалось получить ответ: {message}")

    def _format_stats(self, result: _TestResult) -> str:
        if result.finish is FinishReason.CANCELLED:
            return f"Отменено  ·  {result.total_s:.2f} с"
        parts: list[str] = []
        if result.ttft_s is not None:
            parts.append(f"первый токен {result.ttft_s:.2f} с")
        parts.append(f"всего {result.total_s:.2f} с")
        prefix = "≈" if result.estimated else ""
        parts.append(f"{prefix}{result.usage.total_tokens} токенов")
        parts.append(f"≈${result.cost_usd:.4f}" if result.price_known else "без оплаты")
        return "  ·  ".join(parts)
