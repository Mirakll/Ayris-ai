"""Tab «Логи / DevTools» (task 58): the window onto a running Ayris.

The page is a thin shell that wires five widgets — each of which owns its own
logic and its own tests — to the live application, and adds the two things a
widget cannot reach on its own: a text field that runs a phrase through the real
:meth:`~ayris.core.pipeline.Pipeline.run_text`, and the log-level controls that
retune :mod:`ayris.utils.logger` on the fly.

The live objects are handed in through :class:`DevToolsServices`, set once at
startup by ``__main__`` via :func:`set_active_devtools_services` and cleared
before the GUI tears down. The settings window builds the tab through the plain
``register_tab`` factory, which knows nothing of the pipeline, so the tab reads
the seam itself. With no services — the state a bare settings window or a test
starts in — every widget falls back to its inert, empty form.

Nothing here is a way around the guards: a phrase typed into the field is an
unconfirmed command and passes the same task-40 confirmation as a spoken one;
the REPL is gated once behind «Понимаю риски» and audits every submission.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigManager
from ayris.core.events import EventBus
from ayris.core.paths import get_paths
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.tabs.voice import AsyncRunner
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.log_view import LogView
from ayris.gui.widgets.pipeline_view import PipelineView
from ayris.gui.widgets.repl_console import ReplConsole
from ayris.gui.widgets.worker_health import WorkerHealth, WorkerSupervisor
from ayris.utils.bug_report import BugReportResult, build_bug_report, open_report_folder
from ayris.utils.logger import (
    LOG_LEVELS,
    get_level_state,
    get_logger,
    reset_module_level,
    set_level,
    set_module_level,
)

if TYPE_CHECKING:
    from ayris.core.pipeline import Pipeline

__all__ = [
    "DevToolsServices",
    "DevToolsTab",
    "active_devtools_services",
    "set_active_devtools_services",
]

_log = get_logger(__name__)


@dataclass(slots=True)
class DevToolsServices:
    """The live application objects the tab needs from ``__main__`` — all optional.

    Kept optional so the settings window can build the page before the pipeline
    exists (and a test can build it with nothing), in which case each widget
    shows its inert form.
    """

    #: The running pipeline; feeds the trace table and the text field.
    pipeline: Pipeline | None = None
    #: The worker supervisor (task 05), for the health panel.
    worker_control: WorkerSupervisor | None = None
    #: The globals the REPL runs against — the live ``app``, ``config``, engine…
    repl_namespace: dict[str, Any] = field(default_factory=dict)
    #: Writes one REPL submission to the security journal before it runs.
    audit: Callable[[str], None] | None = None
    #: Whether a pipeline session is live, so worker controls can warn first.
    is_session_active: Callable[[], bool] | None = None


_active_services: DevToolsServices | None = None


def set_active_devtools_services(services: DevToolsServices | None) -> None:
    """Publish (or clear) the services the next DevTools tab will read."""
    global _active_services
    _active_services = services


def active_devtools_services() -> DevToolsServices | None:
    """The services set by ``__main__``, or ``None`` before startup wires them."""
    return _active_services


class _HistoryLineEdit(QLineEdit):
    """The text-command field. Up/Down walk the input history; Enter submits."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.on_history_previous: Callable[[], None] | None = None
        self.on_history_next: Callable[[], None] | None = None

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Up and self.on_history_previous is not None:
            self.on_history_previous()
            return
        if event.key() == Qt.Key.Key_Down and self.on_history_next is not None:
            self.on_history_next()
            return
        super().keyPressEvent(event)


class DevToolsTab(SettingsTab):
    """The «Логи / DevTools» settings page (task 58)."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        services: DevToolsServices | None = None,
    ) -> None:
        super().__init__("devtools", "Логи / DevTools", ("devtools",), manager, theme, bus)
        # The page drives logging and the pipeline directly, not through bound
        # config fields, so the inherited «Сбросить секцию» button has nothing
        # to reset — hide it.
        self.reset_button.hide()
        resolved = services if services is not None else active_devtools_services()
        self._services = resolved if resolved is not None else DevToolsServices()
        self._bus = bus
        self._history: list[str] = []
        self._history_index: int | None = None
        self._report: BugReportResult | None = None

        self._export_runner = AsyncRunner()
        self._export_runner.finished.connect(self._on_export_done)
        self._export_runner.failed.connect(self._on_export_failed)
        self._text_runner = AsyncRunner()
        self._text_runner.finished.connect(self._on_text_done)
        self._text_runner.failed.connect(self._on_text_failed)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self._content = QVBoxLayout(container)
        self._content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        self._build_log_section()
        self._build_pipeline_section()
        self._build_repl_section()
        self._build_workers_section()
        self._content.addStretch(1)

    # -- headers ------------------------------------------------------------

    def _add_header(self, text: str) -> None:
        header = QLabel(text)
        header.setProperty("role", "h2")
        self._content.addWidget(header)

    def _add_caption(self, text: str) -> None:
        caption = QLabel(text)
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

    # -- log section --------------------------------------------------------

    def _build_log_section(self) -> None:
        self._add_header("Журнал")
        self._add_caption(
            "Живой журнал приложения. Значения уже замаскированы — секреты сюда не "
            "попадают. Уровень можно менять на лету, не перезапуская Ayris."
        )
        self._log_view = LogView(
            self._theme,
            self._bus,
            capacity=self._manager.settings.devtools.log_buffer_lines,
        )
        self._content.addWidget(self._log_view)
        self._build_level_controls()
        self._build_log_actions()

    def _build_level_controls(self) -> None:
        common, overrides = get_level_state()
        row = QHBoxLayout()
        row.addWidget(QLabel("Общий уровень:"))
        self._level_combo = ThemedComboBox()
        for level in LOG_LEVELS:
            self._level_combo.addItem(level, level)
        index = self._level_combo.findData(common)
        if index >= 0:
            self._level_combo.setCurrentIndex(index)
        self._level_combo.currentIndexChanged.connect(self._on_common_level)
        row.addWidget(self._level_combo)
        row.addSpacing(self._theme.metric("spacing_md"))
        row.addWidget(QLabel("Модуль:"))
        self._module_edit = QLineEdit()
        self._module_edit.setPlaceholderText("напр. ayris.core.pipeline")
        row.addWidget(self._module_edit, 1)
        self._module_level = ThemedComboBox()
        for level in LOG_LEVELS:
            self._module_level.addItem(level, level)
        row.addWidget(self._module_level)
        set_button = QPushButton("Задать")
        set_button.clicked.connect(self._on_module_level)
        row.addWidget(set_button)
        reset_button = QPushButton("Сбросить")
        reset_button.clicked.connect(self._on_module_reset)
        row.addWidget(reset_button)
        self._content.addLayout(row)
        self._level_status = QLabel("")
        self._level_status.setProperty("role", "secondary")
        self._content.addWidget(self._level_status)
        self._refresh_level_status(overrides)

    def _refresh_level_status(self, overrides: dict[str, str]) -> None:
        if not overrides:
            self._level_status.setText("Переопределений по модулям нет.")
            return
        parts = ", ".join(f"{name} → {level}" for name, level in sorted(overrides.items()))
        self._level_status.setText(f"Переопределения: {parts}")

    def _on_common_level(self, _index: int) -> None:
        data = self._level_combo.currentData()
        if not isinstance(data, str):
            return
        applied = set_level(data)
        # Persist so the choice survives a restart; the live change already happened.
        try:
            self._manager.apply({"devtools.log_level": applied})
        except Exception:
            _log.exception("не удалось сохранить уровень логирования")

    def _on_module_level(self) -> None:
        module = self._module_edit.text().strip()
        level = self._module_level.currentData()
        if not module or not isinstance(level, str):
            return
        set_module_level(module, level)
        self._refresh_level_status(get_level_state()[1])

    def _on_module_reset(self) -> None:
        module = self._module_edit.text().strip()
        if module:
            reset_module_level(module)
            self._refresh_level_status(get_level_state()[1])

    def _build_log_actions(self) -> None:
        row = QHBoxLayout()
        open_button = QPushButton("Открыть папку логов")
        open_button.clicked.connect(self._open_logs_folder)
        row.addWidget(open_button)
        self._export_button = QPushButton("Экспорт лога для баг-репорта…")
        self._export_button.clicked.connect(self._export_report)
        row.addWidget(self._export_button)
        self._open_report_button = QPushButton("Открыть папку архива")
        self._open_report_button.setEnabled(False)
        self._open_report_button.clicked.connect(self._open_report_folder)
        row.addWidget(self._open_report_button)
        row.addStretch(1)
        self._content.addLayout(row)
        self._export_status = QLabel("")
        self._export_status.setProperty("role", "secondary")
        self._export_status.setWordWrap(True)
        self._content.addWidget(self._export_status)

    def _open_logs_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_paths().logs_dir)))

    def _export_report(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Куда сохранить архив баг-репорта", str(Path.home())
        )
        if not chosen:
            return
        destination = Path(chosen)
        self._export_button.setEnabled(False)
        self._export_status.setText("Собираем архив…")
        self._export_runner.run(lambda: build_bug_report(destination))

    def _on_export_done(self, result: object) -> None:
        self._export_button.setEnabled(True)
        if not isinstance(result, BugReportResult):
            return
        self._report = result
        self._open_report_button.setEnabled(True)
        contents = ", ".join(result.included) if result.included else "—"
        self._export_status.setText(
            f"Архив собран: {result.path.name}\nВнутри: {contents}.\n"
            "Отправьте файл разработчику вручную — Ayris ничего никуда не загружает."
        )

    def _on_export_failed(self, message: str) -> None:
        self._export_button.setEnabled(True)
        self._export_status.setText(message or "Не удалось собрать архив.")

    def _open_report_folder(self) -> None:
        if self._report is not None:
            open_report_folder(self._report.path)

    # -- pipeline section ---------------------------------------------------

    def _build_pipeline_section(self) -> None:
        self._add_header("Пайплайн")
        self._add_caption(
            "Каждый проход: распознавание → интент → действие → результат, с "
            "таймингами. Двойной клик по строке — перейти к её логу."
        )
        source = None
        pipeline = self._services.pipeline
        if pipeline is not None:
            source = pipeline.traces
        self._pipeline_view = PipelineView(self._theme, source=source)
        self._pipeline_view.jump_to_log.connect(self._log_view.jump_to_request)
        self._content.addWidget(self._pipeline_view)
        self._build_text_input()

    def _build_text_input(self) -> None:
        if not self._manager.settings.devtools.text_input:
            return
        self._add_caption(
            "Ввод команды текстом идёт тем же путём, что голос: опасное действие "
            "спросит подтверждение. «Сухой прогон» — разобрать, но не выполнять."
        )
        row = QHBoxLayout()
        self._text_edit = _HistoryLineEdit()
        self._text_edit.setPlaceholderText("Команда текстом, напр. «который час»")
        self._text_edit.on_history_previous = self._history_previous
        self._text_edit.on_history_next = self._history_next
        self._text_edit.returnPressed.connect(self._submit_text)
        row.addWidget(self._text_edit, 1)
        self._dry_run_box = QCheckBox("Сухой прогон")
        row.addWidget(self._dry_run_box)
        self._run_button = QPushButton("Выполнить")
        self._run_button.setProperty("kind", "primary")
        self._run_button.clicked.connect(self._submit_text)
        row.addWidget(self._run_button)
        self._repeat_button = QPushButton("Повторить последнюю")
        self._repeat_button.clicked.connect(self._repeat_last)
        row.addWidget(self._repeat_button)
        self._content.addLayout(row)
        self._text_status = QLabel("")
        self._text_status.setProperty("role", "secondary")
        self._text_status.setWordWrap(True)
        self._content.addWidget(self._text_status)
        self._sync_text_enabled()

    def _sync_text_enabled(self) -> None:
        """The field only works with a live pipeline; without one it is inert."""
        enabled = self._services.pipeline is not None
        self._text_edit.setEnabled(enabled)
        self._run_button.setEnabled(enabled)
        self._repeat_button.setEnabled(enabled and bool(self._history))
        if not enabled:
            self._text_status.setText("Пайплайн не запущен — ввод недоступен.")

    def _submit_text(self) -> None:
        pipeline = self._services.pipeline
        if pipeline is None:
            return
        text = self._text_edit.text().strip()
        if not text:
            return
        self._remember(text)
        self._history_index = None
        self._text_edit.clear()
        dry_run = self._dry_run_box.isChecked()
        self._run_button.setEnabled(False)
        self._text_status.setText("Разбираю…" if dry_run else "Выполняю…")
        self._text_runner.run(lambda: pipeline.run_text(text, execute=not dry_run))

    def _remember(self, text: str) -> None:
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._repeat_button.setEnabled(True)

    def _repeat_last(self) -> None:
        if self._history:
            self._text_edit.setText(self._history[-1])
            self._submit_text()

    def _history_previous(self) -> None:
        if not self._history:
            return
        if self._history_index is None:
            self._history_index = len(self._history)
        self._history_index = max(0, self._history_index - 1)
        self._text_edit.setText(self._history[self._history_index])

    def _history_next(self) -> None:
        if self._history_index is None:
            return
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = None
            self._text_edit.clear()
        else:
            self._text_edit.setText(self._history[self._history_index])

    def _on_text_done(self, result: object) -> None:
        self._run_button.setEnabled(self._services.pipeline is not None)
        from ayris.core.pipeline import PipelineResult
        from ayris.gui.widgets.pipeline_view import result_label

        if not isinstance(result, PipelineResult):
            return
        parts = [f"Результат: {result_label(result.outcome)}"]
        if result.intent:
            parts.append(f"интент: {result.intent}")
        if result.spoken:
            parts.append(f"ответ: {result.spoken}")
        if result.error:
            parts.append(f"ошибка: {result.error}")
        prefix = "Сухой прогон. " if self._dry_run_box.isChecked() else ""
        self._text_status.setText(prefix + ", ".join(parts))

    def _on_text_failed(self, message: str) -> None:
        self._run_button.setEnabled(self._services.pipeline is not None)
        self._text_status.setText(message or "Не удалось выполнить команду.")

    # -- repl section -------------------------------------------------------

    def _build_repl_section(self) -> None:
        self._add_header("Консоль")
        self._add_caption(
            "Python в контексте приложения. Включается один раз за «Понимаю риски», "
            "каждый ввод пишется в аудит. Автозапуска сохранённого нет."
        )
        settings = self._manager.settings.devtools
        self._repl_console = ReplConsole(
            self._theme,
            namespace=self._services.repl_namespace,
            on_audit=self._services.audit,
            acknowledged=settings.repl_enabled,
            on_acknowledge=self._persist_repl_enabled,
            history_path=get_paths().cache_dir / "devtools_repl_history.json",
        )
        self._content.addWidget(self._repl_console)

    def _persist_repl_enabled(self) -> None:
        try:
            self._manager.apply({"devtools.repl_enabled": True})
        except Exception:
            _log.exception("не удалось сохранить согласие на REPL")

    # -- workers section ----------------------------------------------------

    def _build_workers_section(self) -> None:
        self._add_header("Воркеры")
        self._add_caption(
            "Каждая подсистема живёт в своём процессе. Перезапуск или пауза во время "
            "активной сессии прервёт её — Ayris сначала предупредит."
        )
        self._worker_health = WorkerHealth(
            self._theme,
            control=self._services.worker_control,
            is_session_active=self._services.is_session_active,
        )
        self._content.addWidget(self._worker_health)

    # -- lifecycle ----------------------------------------------------------

    def dispose(self) -> None:
        for widget in (
            getattr(self, "_log_view", None),
            getattr(self, "_pipeline_view", None),
            getattr(self, "_repl_console", None),
            getattr(self, "_worker_health", None),
        ):
            if widget is not None:
                widget.dispose()
        super().dispose()


register_tab("devtools", DevToolsTab)
