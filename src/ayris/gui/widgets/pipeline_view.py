"""Pipeline trace table for the «Логи / DevTools» tab (task 58).

Task 18 gives every pass through the pipeline a trace: the raw recognition, the
intent it matched, the action it ran, the result, and a stopwatch on each stage.
Section 15 asks DevTools to show them as ``STT raw → NLU intent → Action →
Result`` with the timings, and to let a slow pass be opened up stage by stage.

Two facts shape this widget.

*There is no «trace finished» event.* The pipeline keeps the last hundred frozen
traces in a ring buffer and hands them out with :meth:`Pipeline.traces`; nothing
is published when one lands. So the view polls that snapshot on a timer while it
is visible — the same visibility contract the resource panel uses — instead of
subscribing to a bus.

*A trace carries no timestamp.* :class:`TraceRecord` has durations but no wall
clock, so the model stamps arrival the first time it sees a ``session_id`` and
keeps that stamp across refreshes.

:class:`PipelineViewModel` holds the ingest, the stamping, the result filter and
the longest-stage arithmetic without a widget, so the tests read it directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ayris.core.models import ExecutionResult
from ayris.core.pipeline_trace import StageTiming, TraceRecord
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox

__all__ = [
    "PipelineRow",
    "PipelineView",
    "PipelineViewModel",
    "result_label",
]

#: How often the table repolls the trace ring buffer while it is visible.
_POLL_MS: Final = 1000
_ALL_RESULTS = "все результаты"
#: Width, in block characters, of the bar drawn for the slowest stage of a pass.
_BAR_WIDTH: Final = 24
#: The «Результат» column, coloured by outcome.
_RESULT_COLUMN: Final = 5
_COLUMNS: Final = ("Время", "request_id", "STT raw", "Интент", "Действие", "Результат", "мс")

_RESULT_LABELS: Final[Mapping[ExecutionResult, str]] = {
    ExecutionResult.OK: "успех",
    ExecutionResult.ERROR: "ошибка",
    ExecutionResult.CANCELLED: "отменено",
    ExecutionResult.TIMEOUT: "таймаут",
    ExecutionResult.DENIED: "отклонено",
    ExecutionResult.UNMATCHED: "не распознано",
}


def result_label(outcome: ExecutionResult) -> str:
    """Russian label for a result, so the model and the filter combo agree."""
    return _RESULT_LABELS.get(outcome, outcome.value)


def _result_color_token(outcome: ExecutionResult) -> str:
    """Theme colour token for a result — green for success, red for a failure."""
    if outcome is ExecutionResult.OK:
        return "success"
    if outcome in (ExecutionResult.ERROR, ExecutionResult.TIMEOUT):
        return "error"
    if outcome is ExecutionResult.DENIED:
        return "warning"
    return "text_secondary"


def _bar_text(duration_ms: int, peak_ms: int) -> str:
    """A block-character bar sized against the slowest stage of the same pass."""
    if peak_ms <= 0 or duration_ms <= 0:
        return ""
    return "█" * max(1, round(_BAR_WIDTH * duration_ms / peak_ms))


def _now() -> datetime:
    return datetime.now()


@dataclass(frozen=True, slots=True)
class PipelineRow:
    """One finished pass: the frozen trace and the moment the view first saw it."""

    seen_at: datetime
    record: TraceRecord

    @property
    def session_id(self) -> str:
        return self.record.session_id

    @property
    def time_text(self) -> str:
        return self.seen_at.strftime("%H:%M:%S")

    @property
    def intent_text(self) -> str:
        """Intent with its slots, so a matched pass shows what it captured."""
        base = self.record.intent or "—"
        slots = self.record.payload.get("slots")
        if isinstance(slots, Mapping) and slots:
            joined = ", ".join(f"{key}={value}" for key, value in slots.items())
            return f"{base} ({joined})"
        return base

    def longest_timing(self) -> StageTiming | None:
        """The single slowest stage — the one the view highlights and scales to."""
        if not self.record.stages:
            return None
        return max(self.record.stages, key=lambda timing: timing.duration_ms)


class PipelineViewModel:
    """The trace table's data, without a widget: stamped, ordered, filtered.

    It mirrors the pipeline's ring buffer on every :meth:`ingest`: a
    ``session_id`` still in the snapshot keeps the arrival time it was first
    given, one that dropped out of the buffer drops out here too, and a new one
    is stamped with the injected clock. So the view never drifts from what the
    pipeline actually retains, and the timestamps stay put while it polls.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock if clock is not None else _now
        self._rows: dict[str, PipelineRow] = {}
        self._result = ""

    # -- ingest -------------------------------------------------------------

    def ingest(self, records: Sequence[TraceRecord]) -> bool:
        """Rebuild rows from a trace snapshot. Returns whether anything changed."""
        rebuilt: dict[str, PipelineRow] = {}
        changed = len(records) != len(self._rows)
        for record in records:
            previous = self._rows.get(record.session_id)
            if previous is None:
                rebuilt[record.session_id] = PipelineRow(self._clock(), record)
                changed = True
            else:
                if previous.record != record:
                    changed = True
                rebuilt[record.session_id] = PipelineRow(previous.seen_at, record)
        self._rows = rebuilt
        return changed

    def clear(self) -> None:
        self._rows = {}

    # -- filter -------------------------------------------------------------

    def set_result_filter(self, result: str) -> None:
        self._result = "" if not result or result == _ALL_RESULTS else result

    def matches(self, row: PipelineRow) -> bool:
        return not self._result or row.record.outcome.value == self._result

    # -- read ---------------------------------------------------------------

    @property
    def rows(self) -> list[PipelineRow]:
        """Every stamped row, oldest first."""
        return list(self._rows.values())

    def visible_rows(self) -> list[PipelineRow]:
        """Rows that pass the filter, newest first — how the table stacks them."""
        return [row for row in reversed(self._rows.values()) if self.matches(row)]

    def count(self) -> int:
        return len(self._rows)


class PipelineView(QFrame):
    """A table of pipeline passes: one row each, expandable into timed stages.

    Args:
        theme: Active theme, for metrics and the per-result and stage colours.
        source: Returns the current trace snapshot (``pipeline.traces``). ``None``
            leaves the table empty and static, which is what the tests use.
        clock: Arrival-time source for the model; injected in tests.
    """

    #: Emitted with a ``request_id`` when the user asks to see a pass in the log.
    jump_to_log = Signal(str)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        source: Callable[[], Sequence[TraceRecord]] | None = None,
        clock: Callable[[], datetime] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._source = source
        self._model = PipelineViewModel(clock=clock)
        self._selected_session: str = ""

        self.setProperty("card", True)
        self.setAccessibleName("Таблица пайплайна")
        self._outer = QVBoxLayout(self)
        self._build_controls()
        self._build_tree()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self.refresh)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel("Результат:"))
        self._result_combo = ThemedComboBox()
        self._result_combo.addItem(_ALL_RESULTS)
        for outcome in ExecutionResult:
            self._result_combo.addItem(result_label(outcome), outcome.value)
        self._result_combo.currentIndexChanged.connect(self._on_result_filter)
        row.addWidget(self._result_combo)
        row.addStretch(1)
        self._jump_button = QPushButton("К логу")
        self._jump_button.setEnabled(False)
        self._jump_button.clicked.connect(self._emit_jump)
        row.addWidget(self._jump_button)
        self._outer.addLayout(row)

    def _build_tree(self) -> None:
        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(_COLUMNS))
        self._tree.setHeaderLabels(list(_COLUMNS))
        self._tree.setRootIsDecorated(True)
        self._tree.setAccessibleName("Проходы пайплайна")
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        self._tree.itemSelectionChanged.connect(self._on_selection)
        self._outer.addWidget(self._tree, 1)

    # -- polling lifecycle --------------------------------------------------

    def is_polling(self) -> bool:
        """Whether the refresh timer is running. The visibility contract, for tests."""
        return self._timer.isActive()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        """Poll the trace snapshot and rebuild the tree only when it changed."""
        if self._source is None:
            return
        if self._model.ingest(tuple(self._source())):
            self._rebuild()

    # -- rendering ----------------------------------------------------------

    def _rebuild(self) -> None:
        keep = self._selected_session
        self._tree.clear()
        restored: QTreeWidgetItem | None = None
        for row in self._model.visible_rows():
            item = self._make_trace_item(row)
            self._tree.addTopLevelItem(item)
            self._add_stage_children(item, row)
            if row.session_id == keep:
                restored = item
        if restored is not None:
            restored.setSelected(True)
        else:
            self._selected_session = ""
            self._jump_button.setEnabled(False)

    def _make_trace_item(self, row: PipelineRow) -> QTreeWidgetItem:
        rec = row.record
        item = QTreeWidgetItem(
            [
                row.time_text,
                rec.session_id,
                rec.stt_raw or "—",
                row.intent_text,
                rec.action or "—",
                result_label(rec.outcome),
                f"{rec.total_ms} мс",
            ]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, rec.session_id)
        color = self._theme.theme.color(_result_color_token(rec.outcome))
        item.setForeground(_RESULT_COLUMN, QColor(color))
        return item

    def _add_stage_children(self, parent: QTreeWidgetItem, row: PipelineRow) -> None:
        longest = row.longest_timing()
        peak = longest.duration_ms if longest is not None else 0
        action_color = QColor(self._theme.theme.color("role_action"))
        accent_color = QColor(self._theme.theme.color("accent"))
        for timing in row.record.stages:
            child = QTreeWidgetItem(
                [
                    f"    {timing.stage.label}",
                    _bar_text(timing.duration_ms, peak),
                    "",
                    "",
                    "" if timing.ok else "сбой",
                    "",
                    f"{timing.duration_ms} мс",
                ]
            )
            is_longest = timing is longest
            child.setForeground(1, accent_color if is_longest else action_color)
            if is_longest:
                child.setForeground(6, accent_color)
            parent.addChild(child)

    # -- handlers -----------------------------------------------------------

    def _on_result_filter(self, _index: int) -> None:
        data = self._result_combo.currentData()
        self._model.set_result_filter(data if isinstance(data, str) else "")
        self._rebuild()

    def _current_session(self) -> str:
        item = self._tree.currentItem()
        if item is not None:
            top = item if item.parent() is None else item.parent()
            data = top.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, str):
                return data
        return ""

    def _on_selection(self) -> None:
        session = self._current_session()
        self._selected_session = session
        self._jump_button.setEnabled(bool(session))

    def _on_double_click(self, _item: QTreeWidgetItem, _column: int) -> None:
        self._emit_jump()

    def _emit_jump(self) -> None:
        session = self._current_session()
        if session:
            self.jump_to_log.emit(session)

    # -- lifecycle ----------------------------------------------------------

    @property
    def model(self) -> PipelineViewModel:
        return self._model

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_md")
        self._outer.setContentsMargins(pad, pad, pad, pad)
        self._outer.setSpacing(self._theme.metric("spacing_sm"))

    def stop(self) -> None:
        """Stop the poll timer. A live timer keeps the event loop from ending."""
        if self._timer.isActive():
            self._timer.stop()

    def dispose(self) -> None:
        self.stop()
