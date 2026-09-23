"""The left «showcase» column: brand, state sphere and a quiet caption.

The panel is the darkest surface in the window. The sphere sits centred in a
square that scales with the panel; it animates itself according to its state.
Empty areas drag the frameless window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ayris.gui.dashboard.sphere_host import SphereLike, make_sphere
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.states import SphereState

__all__ = ["ShowcasePanel"]


class _SphereHost(QWidget):
    """Lets the sphere canvas fill the whole area; the sphere centres itself.

    The canvas must cover the entire host (not a centred square): the WebGL
    frame is opaque — filled with the panel colour — and bloom lifts the pixels
    around the sphere. A smaller square would clip that glow at its edge and show
    a lighter box behind the sphere. Full-bleed lets the glow fade to zero inside
    the frame, so its border matches the panel and no box is visible. The sphere's
    camera auto-fits any aspect ratio, so a non-square canvas never clips it.
    """

    def __init__(self, child: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._child = child
        child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event: object) -> None:  # noqa: N802, ARG002
        self._child.setGeometry(0, 0, self.width(), self.height())


class ShowcasePanel(QFrame):
    """Logo, centred sphere and caption; empty space moves the window."""

    drag_started = Signal()

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setObjectName("showcasePanel")

        self._layout = QVBoxLayout(self)

        self.logo = QLabel("AYRIS", self)
        self.logo.setObjectName("showcaseLogo")
        self.logo.setAccessibleName("Айрис")

        sphere_widget = make_sphere(theme, self)
        self._sphere: SphereLike = sphere_widget  # type: ignore[assignment]
        self._sphere_host = _SphereHost(sphere_widget, self)

        self.caption = QLabel("A Y R I S   A S S I S T A N T", self)
        self.caption.setObjectName("showcaseCaption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._layout.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._layout.addWidget(self._sphere_host, 1)
        self._layout.addWidget(self.caption, 0, Qt.AlignmentFlag.AlignHCenter)

        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    @property
    def sphere(self) -> SphereLike:
        return self._sphere

    def set_state(self, state: SphereState | str) -> None:
        self._sphere.set_state(state)

    def set_level(self, level: float) -> None:
        self._sphere.set_level(level)

    # Empty areas drag the frameless window.
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_started.emit()
        super().mousePressEvent(event)

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        metric = self._theme.metric
        typography = self._theme.theme.typography

        background = color("background")
        text = color("text_primary")
        muted = color("text_muted")
        pad = metric("spacing_xl")

        # QWebEngineView won't composite transparently on Windows, so blend the
        # sphere in by painting its page the same colour as this panel.
        set_background = getattr(self._sphere, "set_background", None)
        if callable(set_background):
            set_background(background)

        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(metric("spacing_lg"))
        self.setStyleSheet(
            f"#showcasePanel {{ background: {background}; }}"
            f"#showcaseLogo {{ color: {text}; font-size: {typography.h1_size}px;"
            f" font-weight: {typography.weight_bold}; letter-spacing: 2px; }}"
            f"#showcaseCaption {{ color: {muted}; font-size: {typography.caption_size}px;"
            f" letter-spacing: 4px; }}"
        )
