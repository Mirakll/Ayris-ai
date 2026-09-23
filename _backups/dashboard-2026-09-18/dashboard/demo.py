"""Run the single dashboard window standalone.

``python -m ayris.gui.dashboard.demo`` opens the real window (WebGL sphere and
all) against a throwaway settings file and cycles the assistant through its five
states every few seconds, so the composition and the state transitions can be
seen without launching the whole application.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ayris.core.config import ConfigManager
from ayris.core.state import AssistantState
from ayris.gui.main_window import MainWindow
from ayris.gui.overlay.dialog_log import DialogKind
from ayris.gui.theme import ThemeManager

_STATES: tuple[AssistantState, ...] = (
    AssistantState.IDLE,
    AssistantState.LISTENING,
    AssistantState.THINKING,
    AssistantState.SPEAKING,
    AssistantState.ERROR,
)


def main(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv[:1])
    theme = ThemeManager(app)
    theme.apply()
    manager = ConfigManager(Path(tempfile.gettempdir()) / "ayris-dashboard-demo.toml")
    window = MainWindow(theme=theme, manager=manager)
    window.resize(1100, 680)
    window.show()

    index = {"value": 0}

    def step() -> None:
        state = _STATES[index["value"] % len(_STATES)]
        window.set_state(state)
        if state is AssistantState.SPEAKING:
            window.add_message(DialogKind.ANSWER, "Готово, включаю музыку.")
        elif state is AssistantState.ERROR:
            window.set_status("Не расслышала — повторите, пожалуйста")
        index["value"] += 1

    timer = QTimer(app)
    timer.timeout.connect(step)
    timer.start(2500)
    step()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
