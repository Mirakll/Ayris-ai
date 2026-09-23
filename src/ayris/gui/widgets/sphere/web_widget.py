"""Sphere widget backed by the bundled WebGL prototype — pixel-identical to the
``design/sphere_ref`` HTML, but fully offline.

Three.js and its post-processing passes are vendored under ``assets/vendor`` and
served from a loopback HTTP server bound to ``127.0.0.1`` on a random port. A
local server (rather than ``file://``) is used because Chromium blocks ES-module
imports over ``file://`` (null origin); ``127.0.0.1`` needs no external network.
"""

from __future__ import annotations

import functools
import http.server
import json
import logging
import threading
from pathlib import Path
from typing import Final

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget

from ayris.gui.widgets.sphere.states import AnimationProfile, SphereState

_log = logging.getLogger(__name__)

_ASSETS: Final = Path(__file__).resolve().parent / "assets"

_STATE_TO_MODE: Final[dict[SphereState, str]] = {
    SphereState.IDLE: "calm",
    SphereState.LISTENING: "listening",
    SphereState.THINKING: "thinking",
    SphereState.SPEAKING: "speaking",
    SphereState.ERROR: "error",
}


def _profile_payload(profile: AnimationProfile) -> str:
    """Serialise a motion profile for ``window.setProfile`` (camelCase keys)."""
    return json.dumps(
        {
            "rotationSpeed": profile.rotation_speed,
            "pulseAmplitude": profile.pulse_amplitude,
            "waveIntensity": profile.wave_intensity,
            "rotation": profile.rotation,
            "pulsation": profile.pulsation,
            "waves": profile.waves,
            "errorFlash": profile.error_flash,
        }
    )


_server_lock = threading.Lock()
_server_port: int | None = None


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


def _ensure_server() -> int:
    """Start (once per process) a loopback static server for the assets dir."""

    global _server_port
    with _server_lock:
        if _server_port is not None:
            return _server_port
        handler = functools.partial(_QuietHandler, directory=str(_ASSETS))
        # Threaded so Chromium's parallel module requests don't serialise.
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _server_port = int(httpd.server_address[1])
        return _server_port


class _LoggingPage(QWebEnginePage):
    """Forwards page console output to stderr — invaluable for WebGL debugging."""

    def javaScriptConsoleMessage(  # noqa: N802 - fixed Qt override signature
        self, level: object, message: str, line: int, source: str
    ) -> None:
        _log.warning("sphere-web [%s] %s:%s %s", level, source, line, message)


class SphereWidget(QWebEngineView):
    """WebGL neural sphere with five voice states. API: ``set_state`` / ``set_level``."""

    def __init__(self, parent: QWidget | None = None, *, show_controls: bool = False) -> None:
        super().__init__(parent)
        self._ready = False
        self._state = SphereState.IDLE
        self._level = 0.0
        self._show_controls = show_controls
        self._background: str | None = None
        self._point_count: int | None = None
        self._target_fps: int | None = None
        self._animations = True
        self._stop_when_hidden = True
        self._accent: str | None = None
        self._profile: AnimationProfile | None = None
        self.setPage(_LoggingPage(self))
        self.page().setBackgroundColor(Qt.GlobalColor.transparent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.loadFinished.connect(self._on_loaded)
        port = _ensure_server()
        self.load(QUrl(f"http://127.0.0.1:{port}/sphere.html"))

    @property
    def state(self) -> SphereState:
        return self._state

    def set_state(self, state: SphereState | str) -> None:
        self._state = SphereState(state)
        self._run(f"window.setMode({json.dumps(_STATE_TO_MODE[self._state])})")

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, float(level)))
        self._run(f"window.setLevel({self._level})")

    def set_controls_visible(self, visible: bool) -> None:
        self._show_controls = visible
        self._run(f"window.setPanel({str(visible).lower()})")

    def set_point_count(self, count: int) -> None:
        """Rebuild the wire grid at a new density (task 56, live)."""
        self._point_count = max(100, min(3000, int(count)))
        self._run(f"window.setPointCount({self._point_count})")

    def set_target_fps(self, target_fps: int) -> None:
        """Cap the RAF loop's frame rate."""
        self._target_fps = max(15, min(144, int(target_fps)))
        self._run(f"window.setFps({self._target_fps})")

    def set_animations_enabled(self, enabled: bool) -> None:
        """Freeze or resume the sphere (economy master switch)."""
        self._animations = bool(enabled)
        self._run(f"window.setAnimations({str(self._animations).lower()})")

    def set_stop_when_hidden(self, stop: bool) -> None:
        """Kept for API parity with the painter sphere.

        Chromium already suspends ``requestAnimationFrame`` for a hidden page, so
        the WebGL loop idles when the window is minimised regardless; we only
        remember the flag so callers can treat both spheres identically.
        """
        self._stop_when_hidden = bool(stop)

    def set_profile(self, profile: AnimationProfile) -> None:
        """Apply the user's motion overrides (speed, amplitudes, per-animation)."""
        self._profile = profile
        self._run(f"window.setProfile({_profile_payload(profile)})")

    def set_accent(self, colour: str | None) -> None:
        """Override the sphere accent colour; ``None``/empty keeps the theme."""
        self._accent = colour or None
        self._run(f"window.setAccent({json.dumps(self._accent)})")

    def set_background(self, color: str | None) -> None:
        """Paint the page background a solid colour so it blends with the panel.

        ``QWebEngineView`` on Windows does not reliably composite a transparent
        surface, so instead of leaving it transparent we fill the page with the
        host panel's colour — the sphere then has no visible square behind it.
        Passing ``None`` clears back to the stylesheet default.
        """
        self._background = color
        self._run(f"window.setBackground({json.dumps(color)})")

    def _on_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        if not self._ready:
            return
        self.set_controls_visible(self._show_controls)
        if self._background is not None:
            self.set_background(self._background)
        if self._point_count is not None:
            self.set_point_count(self._point_count)
        if self._target_fps is not None:
            self.set_target_fps(self._target_fps)
        self.set_animations_enabled(self._animations)
        if self._accent is not None:
            self.set_accent(self._accent)
        if self._profile is not None:
            self.set_profile(self._profile)
        self.set_state(self._state)
        self.set_level(self._level)

    def _run(self, js: str) -> None:
        if self._ready:
            self.page().runJavaScript(js)
