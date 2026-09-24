"""In-app Python console for the «Логи / DevTools» tab (task 58).

The console runs real Python against the running application — ``app``,
``config``, the repositories, the registry, the macro engine. That is the most
dangerous surface in the whole program, and three rules from the task keep it
from being a footgun.

*It is off by default and gated once.* The first time it is opened the console
shows a warning and does nothing until the user presses «Понимаю риски»; the
acknowledgement is remembered in ``devtools.repl_enabled`` so the gate is not
asked again. Nothing runs on its own — there is no autorun of saved snippets,
only what the user types now.

*Every submission is audited.* Before a line executes it is written to the
security journal, so even a command that hangs or crashes leaves a trace of what
was asked.

*It never blocks the interface.* Execution happens on a worker thread; the
result is marshalled back as a signal. A running command can be interrupted, on
a best-effort basis, from the GUI thread.

:class:`ReplModel` holds the namespace, the history and the execute/complete
logic without a widget, so the tests drive it directly.
"""

from __future__ import annotations

import builtins
import codeop
import io
import json
import keyword
import os
import threading
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.utils.logger import get_logger

__all__ = [
    "ReplConsole",
    "ReplModel",
    "ReplResult",
]

_log = get_logger(__name__)

#: How many past inputs to keep on disk between runs.
_HISTORY_LIMIT: Final = 200


def _async_raise(thread_id: int | None, exctype: type[BaseException]) -> None:
    """Raise ``exctype`` in another thread — the console's best-effort interrupt.

    There is no public way to stop a running ``exec``; this uses the same CPython
    C-API hook the interpreter uses for Ctrl-C. It only lands between bytecodes,
    so a command blocked in a C call (a socket read, ``time.sleep`` on some
    platforms) keeps running until it returns — the honest limit of interrupting
    a thread that never agreed to yield. ``argtypes`` are declared because an
    undeclared 64-bit pointer arg silently truncates on Win64.
    """
    if thread_id is None:
        return
    import ctypes

    set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
    set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)
    set_async_exc.restype = ctypes.c_int
    raised = set_async_exc(ctypes.c_ulong(thread_id), ctypes.py_object(exctype))
    if raised > 1:  # landed in more than one thread — undo, or the VM is left broken
        set_async_exc(ctypes.c_ulong(thread_id), None)


def _trailing_token(text: str) -> str:
    """The dotted identifier at the end of ``text`` — what completion works on."""
    token: list[str] = []
    for char in reversed(text):
        if char.isalnum() or char in "_.":
            token.append(char)
        else:
            break
    return "".join(reversed(token))


@dataclass(frozen=True, slots=True)
class ReplResult:
    """The outcome of one submission: what it printed, returned, or raised."""

    source: str
    output: str = ""
    value_repr: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class ReplModel:
    """The console without a widget: a namespace, a history, execute and complete.

    Args:
        namespace: The globals every submission runs against. The tab fills it
            with the live application objects; a test passes whatever it wants to
            reach. A copy is taken so the caller's dict is never mutated by ``_``.
        history_path: Where past inputs are persisted as a JSON list. ``None``
            keeps history in memory only, which is what the tests use.
    """

    def __init__(
        self,
        namespace: dict[str, Any] | None = None,
        *,
        history_path: Path | None = None,
        history_limit: int = _HISTORY_LIMIT,
    ) -> None:
        self._namespace: dict[str, Any] = dict(namespace or {})
        self._history_path = history_path
        self._history_limit = max(1, history_limit)
        self._history: list[str] = self._load_history()

    # -- namespace & history ------------------------------------------------

    @property
    def namespace(self) -> dict[str, Any]:
        return self._namespace

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def remember(self, source: str) -> None:
        """Append a submission to history and persist it, skipping repeats."""
        source = source.rstrip("\n")
        if not source.strip():
            return
        if not self._history or self._history[-1] != source:
            self._history.append(source)
            if len(self._history) > self._history_limit:
                del self._history[: -self._history_limit]
            self._save_history()

    def _load_history(self) -> list[str]:
        if self._history_path is None or not self._history_path.exists():
            return []
        try:
            raw = json.loads(self._history_path.read_text("utf-8"))
        except (OSError, ValueError):
            _log.warning("не удалось прочитать историю REPL, начинаю с пустой")
            return []
        return [str(item) for item in raw][-self._history_limit :] if isinstance(raw, list) else []

    def _save_history(self) -> None:
        if self._history_path is None:
            return
        try:
            self._history_path.write_text(json.dumps(self._history, ensure_ascii=False), "utf-8")
        except OSError:
            _log.warning("не удалось сохранить историю REPL")

    # -- execution ----------------------------------------------------------

    def is_complete(self, source: str) -> bool:
        """Whether ``source`` is a finished statement, for Enter-vs-newline.

        ``codeop.compile_command`` returns ``None`` while more lines are needed
        (an open block, a dangling bracket). A syntax error is treated as
        complete so that pressing Enter surfaces it instead of trapping the user
        on a line that will never compile.
        """
        try:
            return codeop.compile_command(source, "<repl>", "single") is not None
        except (SyntaxError, OverflowError, ValueError):
            return True

    def execute(self, source: str) -> ReplResult:
        """Run ``source`` against the namespace, capturing output and errors.

        Called on a worker thread. An expression has its value echoed and bound
        to ``_`` like a real REPL; a statement just runs. Any exception —
        including the :class:`KeyboardInterrupt` an interrupt injects — is caught
        and returned as a traceback, never propagated into the worker.
        """
        buffer = io.StringIO()
        value_repr = ""
        error = ""
        with redirect_stdout(buffer), redirect_stderr(buffer):
            try:
                try:
                    code = compile(source, "<repl>", "eval")
                except SyntaxError:
                    exec(compile(source, "<repl>", "exec"), self._namespace)
                else:
                    result = eval(code, self._namespace)
                    if result is not None:
                        value_repr = repr(result)
                        self._namespace["_"] = result
            except BaseException:
                error = traceback.format_exc()
        return ReplResult(
            source=source, output=buffer.getvalue(), value_repr=value_repr, error=error
        )

    # -- completion ---------------------------------------------------------

    def completions(self, fragment: str) -> list[str]:
        """Names that could finish ``fragment`` — ``dir()`` behind the last dot."""
        fragment = fragment.strip()
        if "." in fragment:
            head, _, prefix = fragment.rpartition(".")
            try:
                target = eval(head, self._namespace)
            except BaseException:
                return []
            names = dir(target)
        else:
            prefix = fragment
            names = [*self._namespace, *dir(builtins), *keyword.kwlist]
        hide_dunder = not prefix.startswith("_")
        matches = {
            name
            for name in names
            if name.startswith(prefix) and not (hide_dunder and name.startswith("__"))
        }
        return sorted(matches)


class _ReplInput(QPlainTextEdit):
    """The input box. Turns key presses into the intents the console handles.

    Enter submits a complete statement and inserts a newline for an unfinished
    one (Shift+Enter always inserts one); Up/Down walk history at the edges;
    Tab asks for completion; Esc asks to interrupt a running command.
    """

    submit_requested = Signal()
    interrupt_requested = Signal()
    history_previous = Signal()
    history_next = Signal()
    complete_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.is_complete: Callable[[str], bool] = lambda _source: True

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if mods & Qt.KeyboardModifier.ShiftModifier or not self.is_complete(self.toPlainText()):
                super().keyPressEvent(event)
            else:
                self.submit_requested.emit()
            return
        if key == Qt.Key.Key_Escape:
            self.interrupt_requested.emit()
            return
        if key == Qt.Key.Key_Tab:
            self.complete_requested.emit()
            return
        if key == Qt.Key.Key_Up and self.textCursor().blockNumber() == 0:
            self.history_previous.emit()
            return
        if key == Qt.Key.Key_Down and self.textCursor().blockNumber() == self.blockCount() - 1:
            self.history_next.emit()
            return
        super().keyPressEvent(event)


class ReplConsole(QFrame):
    """The DevTools Python console: a gated, audited, off-thread REPL.

    Args:
        theme: Active theme, for metrics.
        namespace: Globals the REPL runs against (the live app objects). The tab
            assembles it; tests pass their own.
        on_audit: Called with the source of every submission before it runs.
        acknowledged: Whether «Понимаю риски» was accepted before (the persisted
            ``devtools.repl_enabled`` flag). When ``False`` the console is gated.
        on_acknowledge: Persists the acknowledgement the first time it is given.
        history_path: Where input history lives between runs.
    """

    _result_ready = Signal(object)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        namespace: dict[str, Any] | None = None,
        on_audit: Callable[[str], None] | None = None,
        acknowledged: bool = False,
        on_acknowledge: Callable[[], None] | None = None,
        history_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._model = ReplModel(namespace, history_path=history_path)
        self._on_audit = on_audit
        self._on_acknowledge = on_acknowledge
        self._acknowledged = acknowledged
        self._thread: threading.Thread | None = None
        self._running = False
        self._disposed = False
        self._history_index: int | None = None

        self.setProperty("card", True)
        self.setAccessibleName("Консоль Python")
        self._outer = QVBoxLayout(self)
        self._build_gate()
        self._build_console()
        self._result_ready.connect(self._on_result)
        self._gate.setVisible(not acknowledged)
        self._console.setVisible(acknowledged)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    # -- construction -------------------------------------------------------

    def _build_gate(self) -> None:
        self._gate = QFrame()
        self._gate.setProperty("notice", True)
        self._gate.setProperty("status", "warning")
        layout = QVBoxLayout(self._gate)
        warning = QLabel(
            "Консоль выполняет любой код Python в контексте приложения — с полным "
            "доступом к настройкам, базе и командам. Ошибка здесь может повредить "
            "данные. Автозапуска сохранённых сниппетов нет; каждый ввод пишется в "
            "журнал безопасности."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._ack_button = QPushButton("Понимаю риски")
        self._ack_button.clicked.connect(self._acknowledge)
        button_row.addWidget(self._ack_button)
        layout.addLayout(button_row)
        self._outer.addWidget(self._gate)

    def _build_console(self) -> None:
        self._console = QWidget()
        layout = QVBoxLayout(self._console)
        layout.setContentsMargins(0, 0, 0, 0)
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setAccessibleName("Вывод консоли")
        self._output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._output.setFont(_mono_font())
        layout.addWidget(self._output, 1)

        self._input = _ReplInput()
        self._input.setAccessibleName("Ввод консоли")
        self._input.setFont(_mono_font())
        self._input.is_complete = self._model.is_complete
        self._input.submit_requested.connect(self._submit)
        self._input.interrupt_requested.connect(self._interrupt)
        self._input.complete_requested.connect(self._complete)
        self._input.history_previous.connect(self._history_previous)
        self._input.history_next.connect(self._history_next)
        layout.addWidget(self._input)

        actions = QHBoxLayout()
        self._run_button = QPushButton("Выполнить")
        self._run_button.clicked.connect(self._submit)
        actions.addWidget(self._run_button)
        self._interrupt_button = QPushButton("Прервать")
        self._interrupt_button.setEnabled(False)
        self._interrupt_button.clicked.connect(self._interrupt)
        actions.addWidget(self._interrupt_button)
        actions.addStretch(1)
        self._clear_button = QPushButton("Очистить вывод")
        self._clear_button.clicked.connect(self._output.clear)
        actions.addWidget(self._clear_button)
        layout.addLayout(actions)
        self._outer.addWidget(self._console)

    def _acknowledge(self) -> None:
        self._acknowledged = True
        self._gate.setVisible(False)
        self._console.setVisible(True)
        self._input.setFocus()
        if self._on_acknowledge is not None:
            self._on_acknowledge()

    # -- execution ----------------------------------------------------------

    def _submit(self) -> None:
        source = self._input.toPlainText()
        if not source.strip() or self._running:
            return
        self._model.remember(source)
        self._history_index = None
        if self._on_audit is not None:
            self._on_audit(source)
        self._echo_input(source)
        self._input.clear()
        self._set_running(True)
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def _run(self, source: str) -> None:
        """Worker-thread body: execute and hand the result back as a signal."""
        result = self._model.execute(source)
        self._result_ready.emit(result)

    def _on_result(self, result: ReplResult) -> None:
        if self._disposed:
            return
        self._set_running(False)
        self._thread = None
        if result.output:
            self._append(result.output.rstrip("\n"))
        if result.value_repr:
            self._append(result.value_repr)
        if result.error:
            self._append(result.error.rstrip("\n"))

    def _interrupt(self) -> None:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        self._append("— прерывание —")
        _async_raise(thread.ident, KeyboardInterrupt)

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._run_button.setEnabled(not running)
        self._interrupt_button.setEnabled(running)

    # -- history & completion ----------------------------------------------

    def _history_previous(self) -> None:
        history = self._model.history
        if not history:
            return
        if self._history_index is None:
            self._history_index = len(history)
        self._history_index = max(0, self._history_index - 1)
        self._input.setPlainText(history[self._history_index])

    def _history_next(self) -> None:
        history = self._model.history
        if self._history_index is None:
            return
        self._history_index += 1
        if self._history_index >= len(history):
            self._history_index = None
            self._input.clear()
        else:
            self._input.setPlainText(history[self._history_index])

    def _complete(self) -> None:
        token = _trailing_token(self._input.toPlainText())
        matches = self._model.completions(token)
        if not matches:
            return
        prefix = token.rpartition(".")[2]
        if len(matches) == 1:
            self._input.insertPlainText(matches[0][len(prefix) :])
            return
        common = os.path.commonprefix(matches)
        if len(common) > len(prefix):
            self._input.insertPlainText(common[len(prefix) :])
        self._append("  ".join(matches[:40]))

    # -- output & lifecycle -------------------------------------------------

    def _echo_input(self, source: str) -> None:
        lines = source.splitlines() or [""]
        prefixed = [f">>> {lines[0]}"] + [f"... {line}" for line in lines[1:]]
        self._append("\n".join(prefixed))

    def _append(self, text: str) -> None:
        self._output.appendPlainText(text)
        scrollbar = self._output.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    @property
    def model(self) -> ReplModel:
        return self._model

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_acknowledged(self) -> bool:
        return self._acknowledged

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        gap = self._theme.metric("spacing_sm")
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(gap)

    def dispose(self) -> None:
        """Stop caring about a still-running command; its late result is dropped."""
        self._disposed = True


def _mono_font() -> QFont:
    return QFont("Consolas")
