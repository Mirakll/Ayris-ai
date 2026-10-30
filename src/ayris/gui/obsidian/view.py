"""QWebEngineView wrapper for the Obsidian 2c interface."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from ayris.gui.main_window import MainWindow
    from ayris.gui.obsidian.server import ObsidianServer

_log = logging.getLogger(__name__)


class _LoggingPage(QWebEnginePage):
    """Forwards JavaScript console output to logger for debugging."""

    def javaScriptConsoleMessage(  # noqa: N802
        self, level: object, message: str, line: int, source: str
    ) -> None:
        _log.debug("obsidian-ui [%s] %s:%s %s", level, source, line, message)


class ObsidianView(QWidget):
    """Hosts the Obsidian 2c HTML interface inside a QWebEngineView."""

    def __init__(
        self,
        window: MainWindow,
        server: ObsidianServer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._server = server
        self._ready = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = QWebEngineView(self)
        self._page = _LoggingPage(self._view)
        self._view.setPage(self._page)

        # Transparent background for seamless frameless window styling
        self._page.setBackgroundColor(Qt.GlobalColor.transparent)
        self._view.setStyleSheet("background: transparent;")

        # Enable local storage and WebGL
        settings = self._page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)

        layout.addWidget(self._view)

        # Connect bridge signals to window actions on the Qt thread
        signals = self._server.signals
        signals.command_requested.connect(self._on_command_requested)
        signals.voice_toggled.connect(self._window._on_voice)
        signals.voice_triggered.connect(self._window._on_trigger_voice)
        signals.mode_requested.connect(self._on_mode_requested)
        signals.profile_requested.connect(self._on_profile_requested)
        signals.setting_update_requested.connect(self._on_setting_update_requested)
        signals.window_action_requested.connect(self._on_window_action)
        signals.pick_model_requested.connect(self._window._on_pick_model)
        signals.wake_reset_requested.connect(self._window._on_wake_reset)
        signals.wake_mark_false_requested.connect(self._window._on_wake_mark_false)
        signals.pick_color_requested.connect(self._window._on_pick_color)
        signals.pick_gguf_requested.connect(self._window._on_pick_gguf)
        signals.pick_folder_requested.connect(self._window._on_pick_folder)

        self._view.loadFinished.connect(self._on_load_finished)

        # Load the Obsidian 2c page from loopback server
        port = self._server.server_address[1]
        url = QUrl(f"http://127.0.0.1:{port}/redesign2/2c-obsidian-chat.html?runtime=1")
        self._view.setUrl(url)

    def _on_load_finished(self, ok: bool) -> None:
        if not ok:
            _log.warning("Не удалось загрузить Obsidian 2c HTML интерфейс")
            return
        self._ready = True
        self.run_js(
            "document.documentElement.classList.add('runtime-env');"
            "document.body.classList.add('runtime');"
        )
        # Synchronise current assistant state and profile immediately
        state = getattr(self._window, "_current_state", "idle")
        mode = getattr(self._window, "_current_mode_name", "hybrid")
        profile = getattr(self._window, "_current_profile_name", "По умолчанию")
        self.set_state(state)
        self.set_mode(mode)
        self.set_profile(profile)

    def run_js(self, code: str) -> None:
        if self._ready:
            self._page.runJavaScript(code)

    _run_js = run_js

    def _on_command_requested(self, text: str) -> None:
        self._window._on_command(text)

    def _on_mode_requested(self, mode: str) -> None:
        from ayris.core.pipeline import NluMode
        from ayris.gui.tabs.ai import _MODE_FLAGS

        mode_map = {
            "commands": NluMode.COMMANDS,
            "hybrid": NluMode.HYBRID,
            "chat": NluMode.AI,
            "ai": NluMode.AI,
            "команды": NluMode.COMMANDS,
            "гибрид": NluMode.HYBRID,
            "чат": NluMode.AI,
            "ии": NluMode.AI,
            "к": NluMode.COMMANDS,
            "г": NluMode.HYBRID,
            "ч": NluMode.AI,
        }
        target = mode_map.get(mode.lower(), NluMode.HYBRID)
        self._window._manager.apply(dict(_MODE_FLAGS[target]))
        self.set_mode(target.value)

    def _on_profile_requested(self, name: str) -> None:
        if self._window._profiles is not None:
            for prof in self._window._profiles():
                if prof.name == name:
                    self._window._on_profile_chosen(prof)
                    break

    def _on_setting_update_requested(self, path: str, value: object) -> None:
        try:
            self._window._manager.apply({path: value})
        except Exception:
            _log.exception("Ошибка обновления настройки %s", path)

    def _on_window_action(self, action: str) -> None:
        if action == "minimize":
            self._window.showMinimized()
        elif action == "maximize":
            self._window.toggle_fullscreen()
        elif action == "close":
            self._window.close()
        elif action == "exit":
            self._window.exit()
        elif action == "start-drag":
            self._window._start_system_move()

    # -- Public API called by MainWindow ------------------------------------

    def set_state(self, state: str) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setState({json.dumps(state)});")

    def set_status(self, text: str) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setStatus({json.dumps(text)});")

    def add_message(self, kind: str, text: str) -> None:
        payload = f"{json.dumps(kind)}, {json.dumps(text)}"
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.addMessage({payload});")

    def set_mic(self, enabled: bool, mode: str) -> None:
        self._run_js(
            f"window.ayrisAPI && window.ayrisAPI.setMic({json.dumps(enabled)}, {json.dumps(mode)});"
        )

    def set_mode(self, mode: str) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setMode({json.dumps(mode)});")

    def set_profile(self, name: str) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setProfile({json.dumps(name)});")

    def set_profiles(self, profiles: list[str]) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setProfiles({json.dumps(profiles)});")

    def set_telemetry(self, cpu: int, ram: int, gpu: int) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setTelemetry({cpu}, {ram}, {gpu});")

    def set_level(self, rms: float) -> None:
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.setLevel({rms});")

    def on_wake_detected(
        self, phrase: str, confidence: float, detections: int, false_positives: int
    ) -> None:
        payload = f"{json.dumps(phrase)}, {confidence}, {detections}, {false_positives}"
        self._run_js(f"window.ayrisAPI && window.ayrisAPI.onWakeDetected({payload});")
