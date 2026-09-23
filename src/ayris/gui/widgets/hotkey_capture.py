"""The «Нажмите сочетание» dialog: capture one combination through task 37.

The dialog never reads the keyboard itself. It asks the live
:class:`~ayris.utils.hotkey_manager.HotkeyManager` to hook the keyboard
(:meth:`~ayris.utils.hotkey_manager.HotkeyManager.capture`) and hands back the one
combination task 37 normalises for it — the same parser the config and the command
triggers use, so a combo assigned here can never disagree with one stored elsewhere.

Two things the task insists on shape the code:

*The hook is released on every exit.* Capture holds a low-level keyboard hook, and a
hook left installed swallows the user's keystrokes system-wide. :meth:`done` — the one
method Qt routes **every** close through (the buttons, Esc, the window's ✕, an
``accept``/``reject`` from anywhere) — cancels the capture, so there is exactly one
release path and it cannot be missed.

*The callbacks arrive on the wrong thread.* ``capture`` reports the result and the live
modifiers from the backend's message-loop thread, never the GUI thread. Touching a
widget there would crash, so both are bounced through a queued signal onto the thread
that owns the dialog, the same marshalling the voice tab's ``AsyncRunner`` uses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.utils.hotkeys import Hotkey
from ayris.utils.logger import get_logger

__all__ = ["CaptureManager", "CaptureResult", "HotkeyCaptureDialog"]

_log = get_logger(__name__)

#: Display captions for the held-modifier preview. The final key is not shown live —
#: pressing it ends the capture — so only the four modifiers need a caption here.
_MODIFIER_LABELS: dict[str, str] = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}


class CaptureManager(Protocol):
    """The slice of :class:`~ayris.utils.hotkey_manager.HotkeyManager` the dialog needs.

    A ``Protocol`` so the real manager satisfies it structurally and a test can pass a
    fake that completes the capture synchronously.
    """

    def capture(
        self,
        callback: Callable[[Hotkey | None], None] | None = None,
        *,
        on_modifiers: Callable[[tuple[str, ...]], None] | None = None,
        timeout: float | None = None,
    ) -> Hotkey | None: ...

    def cancel_capture(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """What the dialog came back with: a chosen combination, or a request to clear.

    ``hotkey`` carries the captured combination; ``cleared`` is ``True`` when the user
    asked to unbind the shortcut (Delete or the «Очистить» button). Cancelling the
    dialog returns ``None`` instead of a result, so «отмена» never touches the binding.
    """

    hotkey: Hotkey | None = None
    cleared: bool = False


class HotkeyCaptureDialog(QDialog):
    """Modal «press a combination» dialog over a live capture manager."""

    #: Emitted from the capture (backend) thread; delivered on the GUI thread queued.
    _result_ready = Signal(object)
    _modifiers_ready = Signal(object)

    def __init__(
        self,
        theme: ThemeManager,
        manager: CaptureManager,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._manager = manager
        self._outcome: CaptureResult | None = None
        self._capturing = False
        self._finished = False

        self.setWindowTitle("Назначение сочетания")
        self.setAccessibleName("Назначение сочетания клавиш")
        self.setModal(True)

        self._layout = QVBoxLayout(self)
        heading = QLabel("Нажмите сочетание")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)

        # The live preview of held modifiers; the placeholder stands in until the
        # first key goes down so the row never collapses to nothing.
        self._preview = QLabel("…")
        self._preview.setProperty("role", "h1")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._preview)

        self._hint = QLabel(
            "Модификатор сам по себе не подойдёт — добавьте клавишу. "
            "Esc — отмена, Delete — очистить сочетание. "
            "Клавиши-переключатели (Caps Lock, Num Lock, Scroll Lock, Pause) "
            "назначить нельзя."
        )
        self._hint.setProperty("role", "secondary")
        self._hint.setWordWrap(True)
        self._layout.addWidget(self._hint)

        buttons = QDialogButtonBox()
        self._clear_button = QPushButton("Очистить")
        self._clear_button.setAccessibleName("Очистить сочетание")
        self._clear_button.clicked.connect(self._on_clear)
        self._cancel_button = QPushButton("Отмена")
        self._cancel_button.clicked.connect(self.reject)
        buttons.addButton(self._clear_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self._cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        self._layout.addWidget(buttons)

        # Queued so a signal emitted from the backend thread runs on the GUI thread.
        self._result_ready.connect(self._on_result, Qt.ConnectionType.QueuedConnection)
        self._modifiers_ready.connect(self._on_modifiers, Qt.ConnectionType.QueuedConnection)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def run(self) -> CaptureResult | None:
        """Show the dialog modally and return the outcome, or ``None`` if cancelled."""
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        return self._outcome

    # -- capture lifecycle --------------------------------------------------

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)  # type: ignore[arg-type]
        if self._capturing or self._finished:
            return
        self._capturing = True
        try:
            self._manager.capture(
                self._result_ready.emit,
                on_modifiers=self._modifiers_ready.emit,
            )
        except Exception:
            # The backend is not running (window opened before the hotkey loop, or a
            # machine where it never starts). Say so plainly and leave «Отмена».
            _log.exception("не удалось начать захват сочетания")
            self._capturing = False
            self._preview.setText("—")
            self._hint.setText(
                "Захват сочетания сейчас недоступен: обработчик горячих клавиш не запущен."
            )

    def _on_result(self, result: object) -> None:
        """A combination (or Esc → ``None``) arrived from the capture thread."""
        self._capturing = False
        if result is None:
            # Esc inside the capture cancels, matching the «Отмена» button.
            self.reject()
            return
        if not isinstance(result, Hotkey):
            return
        # A bare Delete means «unbind», per the task; Delete with a modifier is a
        # normal combination the user may legitimately want.
        if result.key == "delete" and not result.modifiers:
            self._outcome = CaptureResult(cleared=True)
        else:
            self._outcome = CaptureResult(hotkey=result)
        self.accept()

    def _on_modifiers(self, held: object) -> None:
        """Repaint the live preview as modifiers are pressed and released."""
        if not isinstance(held, tuple):
            return
        labels = [_MODIFIER_LABELS[name] for name in held if name in _MODIFIER_LABELS]
        self._preview.setText(" + ".join([*labels, "…"]) if labels else "…")

    def _on_clear(self) -> None:
        self._outcome = CaptureResult(cleared=True)
        self.accept()

    def done(self, result: int) -> None:
        # Every close routes through done(): the buttons, Esc, the window ✕, and any
        # accept()/reject() above. Releasing the hook here is the single guarantee the
        # task demands — a hook left installed would swallow the user's keystrokes.
        if not self._finished:
            self._finished = True
            try:
                self._manager.cancel_capture()
            except Exception:
                _log.exception("не удалось снять хук захвата сочетания")
        super().done(result)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_lg"))
        self.setMinimumWidth(self._theme.metric("dialog_width"))
