"""The right «dialogue» column: top bar, conversation, status and input.

This is where the old floating overlay's parts live now — the dialogue log, the
active-timers line and the command field — plus the window's own chrome
(hamburger for settings, a profile selector, minimise and close). When the log
is empty the column shows a large status line under a soft glyph instead.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ayris.core.models import Profile
from ayris.core.profile import DEFAULT_PROFILE_NAME
from ayris.core.state import MicMode
from ayris.gui.dashboard.input_bar import InputBar
from ayris.gui.overlay.dialog_log import DialogKind, DialogLog
from ayris.gui.overlay.timers_panel import TimerProvider, TimersPanel
from ayris.gui.theme import ThemeManager

__all__ = ["DialogView"]

_DEFAULT_STATUS = "Привет, чем помочь?"


def _avatar_pixmap(color: str, size: int) -> QPixmap:
    """A soft, rounded triangle glyph used above the idle status line."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = size / 100.0
    path = QPainterPath()
    path.moveTo(50 * unit, 18 * unit)
    path.lineTo(82 * unit, 74 * unit)
    path.lineTo(18 * unit, 74 * unit)
    path.closeSubpath()

    # Приглушаем акцент: на тёмном фоне насыщенный цвет «вибрирует» и режет
    # глаз, поэтому сбрасываем насыщенность и яркость до спокойной лаванды.
    src = QColor(color)
    calm = QColor.fromHsv(src.hue(), int(src.saturation() * 0.45), min(src.value(), 165))

    top = calm.lighter(110)
    top.setAlphaF(0.7)
    bottom = QColor(calm)
    bottom.setAlphaF(0.58)
    gradient = QLinearGradient(0.0, 18 * unit, 0.0, 74 * unit)
    gradient.setColorAt(0.0, top)
    gradient.setColorAt(1.0, bottom)

    edge = QColor(calm)
    edge.setAlphaF(0.66)
    pen = painter.pen()
    pen.setColor(edge)
    pen.setWidthF(6 * unit)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QBrush(gradient))
    painter.drawPath(path)
    painter.end()
    return pixmap


class DialogView(QFrame):
    """Top bar + conversation/status + timers + command input."""

    settings_requested = Signal()
    minimize_requested = Signal()
    close_requested = Signal()
    command_submitted = Signal(str)
    voice_requested = Signal()
    code_requested = Signal()
    profile_selected = Signal(object)
    drag_started = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        timer_provider: TimerProvider | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._profiles: tuple[Profile, ...] = ()
        self.setObjectName("dialogPanel")

        self._root = QVBoxLayout(self)

        self._top = self._build_top_bar()
        self._root.addLayout(self._top)

        self._center = QStackedWidget(self)
        self._center.setObjectName("dialogCenter")
        self._empty_page = self._build_empty_page()
        self.log = DialogLog(theme)
        self._center.addWidget(self._empty_page)
        self._center.addWidget(self.log)
        self._center.setCurrentWidget(self._empty_page)
        self._root.addWidget(self._center, 1)

        self._status_row = QHBoxLayout()
        self.timers = TimersPanel(theme, provider=timer_provider, show_empty=False)
        self._status_row.addWidget(self.timers, 1)
        self._root.addLayout(self._status_row)

        self.input_bar = InputBar(theme, self)
        self.input_bar.submitted.connect(self.command_submitted.emit)
        self.input_bar.voice_requested.connect(self.voice_requested.emit)
        self.input_bar.code_requested.connect(self.code_requested.emit)
        self._root.addWidget(self.input_bar)

        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    # -- construction -------------------------------------------------------

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.menu_button = QPushButton("≡", self)  # ≡
        self.menu_button.setObjectName("topGhost")
        self.menu_button.setAccessibleName("Открыть настройки")
        self.menu_button.setToolTip("Настройки")
        self.menu_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.menu_button.clicked.connect(self.settings_requested.emit)

        self.profile_button = QPushButton(f"{DEFAULT_PROFILE_NAME}  ▾", self)
        self.profile_button.setObjectName("profileSelector")
        self.profile_button.setAccessibleName("Профиль ассистента")
        self.profile_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.profile_button.clicked.connect(self._show_profile_menu)

        # Офлайн-подпись живёт тихой строкой под именем профиля: показывается
        # только когда сети нет и сворачивается (visible=False), не раздвигая бар.
        self.offline_hint = QLabel("", self)
        self.offline_hint.setObjectName("offlineHint")
        self.offline_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.offline_hint.setVisible(False)

        profile_box = QWidget(self)
        # Глобальный QSS красит любой QWidget в тёмный background — гасим его,
        # чтобы колонка профиля сливалась с панелью, а не рисовала тёмную плашку.
        profile_box.setObjectName("profileGroup")
        self._profile_col = QVBoxLayout(profile_box)
        self._profile_col.setContentsMargins(0, 0, 0, 0)
        self._profile_col.addWidget(self.profile_button, 0, Qt.AlignmentFlag.AlignHCenter)
        self._profile_col.addWidget(self.offline_hint, 0, Qt.AlignmentFlag.AlignHCenter)

        self.minimize_button = QPushButton("–", self)  # –
        self.minimize_button.setObjectName("topGhost")
        self.minimize_button.setAccessibleName("Свернуть окно")
        self.minimize_button.setToolTip("Свернуть")
        self.minimize_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.minimize_button.clicked.connect(self.minimize_requested.emit)

        self.close_button = QPushButton("✕", self)  # ✕
        self.close_button.setObjectName("topClose")
        self.close_button.setAccessibleName("Закрыть окно")
        self.close_button.setToolTip("Закрыть")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.close_requested.emit)

        # Кнопки по краям крепим к верху, чтобы появление офлайн-строки под
        # профилем не сдвигало их вниз вместе с центрированием бара.
        top = Qt.AlignmentFlag.AlignTop
        bar.addWidget(self.menu_button, 0, top)
        bar.addStretch(1)
        bar.addWidget(profile_box, 0, top)
        bar.addStretch(1)
        bar.addWidget(self.minimize_button, 0, top)
        bar.addWidget(self.close_button, 0, top)
        return bar

    def _build_empty_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("dialogEmpty")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(2)
        self.avatar = QLabel(page)
        self.avatar.setObjectName("statusAvatar")
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel(_DEFAULT_STATUS, page)
        self.status_label.setObjectName("statusLine")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setAccessibleName("Статус ассистента")
        layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(3)
        return page

    # -- public API ---------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.status_label.setText(text or _DEFAULT_STATUS)

    def add_message(self, kind: DialogKind, text: str) -> None:
        self.log.add_entry(kind, text)
        if self.log.entries:
            self._center.setCurrentWidget(self.log)

    def clear_dialogue(self) -> None:
        self.log.clear()
        self._center.setCurrentWidget(self._empty_page)

    def set_voice_active(self, active: bool) -> None:
        self.input_bar.set_voice_active(active)

    def set_mic(self, *, enabled: bool, mode: MicMode) -> None:
        # A muted microphone dims the voice button; the mode rides along for a11y.
        name = f"Голосовой ввод, режим: {mode.label}" if enabled else "Микрофон выключен"
        self.input_bar.mic_button.setAccessibleName(name)
        self.input_bar.mic_button.setEnabled(enabled)

    def set_online(self, *, online: bool, detail: str = "") -> None:
        self.offline_hint.setVisible(not online)
        if not online:
            self.offline_hint.setText(detail or "Офлайн")
            self.offline_hint.setAccessibleName(detail or "Офлайн-режим")

    def set_profile(self, name: str) -> None:
        self.profile_button.setText(f"{name}  ▾")

    def set_profiles(self, profiles: tuple[Profile, ...]) -> None:
        self._profiles = profiles

    def focus_command(self) -> None:
        self.input_bar.focus_command()

    # -- interaction --------------------------------------------------------

    def _show_profile_menu(self) -> None:
        if not self._profiles:
            return
        menu = QMenu(self.profile_button)
        for profile in self._profiles:
            action = menu.addAction(profile.name)
            action.triggered.connect(
                lambda _checked=False, p=profile: self.profile_selected.emit(p)
            )
        menu.exec(self.profile_button.mapToGlobal(self.profile_button.rect().bottomLeft()))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_started.emit()
        super().mousePressEvent(event)

    # -- theme --------------------------------------------------------------

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        metric = self._theme.metric
        typography = self._theme.theme.typography

        surface = color("surface")
        surface_high = color("surface_highlight")
        text = color("text_primary")
        secondary = color("text_secondary")
        muted = color("text_muted")
        accent = color("accent")
        radius_md = metric("radius_md")
        radius_lg = metric("radius_lg")
        pad = metric("spacing_lg")
        control = metric("control_height")
        control_lg = metric("control_height_lg")

        self._root.setContentsMargins(pad, pad, pad, pad)
        self._root.setSpacing(metric("spacing_md"))
        self._top.setSpacing(metric("spacing_sm"))
        self._profile_col.setSpacing(metric("spacing_xs"))
        self._status_row.setSpacing(metric("spacing_sm"))

        # Round only the outer (right) corners so the panel follows the window's
        # rounded card; the left edge butts against the seam. Qt does not clip
        # children to the root's border-radius, so a square fill would otherwise
        # cover the rounded frame and the window's corners look bitten off.
        self.setStyleSheet(
            f"#dialogPanel {{ background: {surface};"
            f" border-top-right-radius: {radius_lg}px;"
            f" border-bottom-right-radius: {radius_lg}px; }}"
            # The global QSS paints every QWidget with the darkest `background`
            # colour; without this the empty-status page draws a dark box over
            # the panel's surface. Keep the stack and its idle page transparent.
            f"#dialogCenter, #dialogEmpty {{ background: transparent; }}"
            f"#topGhost {{ background: transparent; border: none; color: {secondary};"
            f" font-size: {typography.h2_size}px; border-radius: {radius_md}px; }}"
            f"#topGhost:hover {{ color: {text}; background: {surface_high}; }}"
            f"#topClose {{ background: transparent; border: none; color: {accent};"
            f" font-size: {typography.h2_size}px; border-radius: {radius_md}px; }}"
            f"#topClose:hover {{ background: {surface_high}; }}"
            f"#profileGroup {{ background: transparent; }}"
            f"#profileSelector {{ background: transparent; border: none; color: {text};"
            f" font-weight: {typography.weight_medium}; }}"
            f"#profileSelector:hover {{ color: {accent}; }}"
            f"#statusAvatar {{ background: transparent; }}"
            f"#statusLine {{ background: transparent; color: {text};"
            f" font-size: {typography.h1_size}px; font-weight: {typography.weight_bold}; }}"
            f"#offlineHint {{ background: transparent; color: {muted};"
            f" font-size: {typography.caption_size}px; }}"
        )

        self.menu_button.setFixedSize(QSize(control, control))
        # Гамбургер читается мельче собратьев — даём глифу пару пикселей.
        self.menu_button.setStyleSheet(f"#topGhost {{ font-size: {typography.h2_size + 3}px; }}")
        self.minimize_button.setFixedSize(QSize(control, control))
        self.close_button.setFixedSize(QSize(control, control))
        self.profile_button.setMinimumHeight(control)
        for label in (self.menu_button, self.minimize_button, self.close_button):
            label.setMaximumHeight(control_lg)

        avatar_side = metric("icon_lg") + metric("spacing_xl")
        self.avatar.setPixmap(_avatar_pixmap(accent, avatar_side))
        self.avatar.setFixedSize(QSize(avatar_side, avatar_side))
