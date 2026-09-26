"""Редактор системного промпта: поле, сброс к дефолту, предпросмотр, счётчик токенов.

Вкладка «ИИ» правит два промпта — чата и NLU (задача 64). Оба ведут себя
одинаково: пользователь пишет свою «персону», а модель получает её вместе с
подробным шаблоном поведения и (для NLU) списком команд. Виджет держит только
персону — ту строку, что уходит в ``ai.chat_system_prompt`` /
``ai.nlu_system_prompt``, — а итоговый промпт собирает переданный ``build_preview``
и показывает в отдельном поле только для чтения, чтобы было видно, что именно
увидит модель.

Как собрать итог, виджет не знает: колбэк приходит из вкладки, где живут
``build_chat_prompt`` / ``build_nlu_prompt`` и каталог команд. Так один виджет
обслуживает оба промпта, а предпросмотр остаётся правдивым, ничего не запуская.
Счётчик токенов считает по :func:`estimate_tokens` — груб, как и везде, и помечен
знаком «≈».
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import ConfirmDialog
from ayris.nlu.llm.usage import estimate_tokens

__all__ = ["PromptEditor"]


class PromptEditor(QWidget):
    """Многострочный редактор одного промпта с предпросмотром итога и сбросом."""

    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        default_text: str,
        build_preview: Callable[[str], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._default_text = default_text
        self._build_preview = build_preview
        self.setProperty("transparent", True)
        self._build()
        self.refresh_preview()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))

        self.editor = QPlainTextEdit()
        self.editor.setAccessibleName("Текст промпта")
        self.editor.setTabChangesFocus(True)
        self.editor.textChanged.connect(self._on_edited)
        layout.addWidget(self.editor)

        controls = QHBoxLayout()
        controls.setSpacing(self._theme.metric("spacing_sm"))
        self.reset_button = QPushButton("Сбросить к дефолту")
        self.reset_button.clicked.connect(self._confirm_reset)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)
        self.token_label = QLabel()
        self.token_label.setProperty("role", "muted")
        controls.addWidget(self.token_label)
        layout.addLayout(controls)

        preview_caption = QLabel("Итоговый промпт для модели:")
        preview_caption.setProperty("role", "secondary")
        layout.addWidget(preview_caption)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setAccessibleName("Предпросмотр итогового промпта")
        self.preview.setProperty("role", "muted")
        layout.addWidget(self.preview)

    def text(self) -> str:
        """Персона — та строка, что уходит в конфиг (без шаблона и команд)."""
        return self.editor.toPlainText()

    def setText(self, value: str) -> None:  # noqa: N802 - зеркалит QLineEdit для _bind
        """Загрузить значение из конфига, не поднимая ``changed`` на программный ввод."""
        blocker = QSignalBlocker(self.editor)
        self.editor.setPlainText(value)
        del blocker
        self.refresh_preview()

    def refresh_preview(self) -> None:
        """Пересобрать итоговый промпт и пересчитать токены."""
        persona = self.text()
        try:
            preview = self._build_preview(persona)
        except Exception:
            preview = persona
        blocker = QSignalBlocker(self.preview)
        self.preview.setPlainText(preview)
        del blocker
        self.token_label.setText(f"≈{estimate_tokens(preview)} токенов")

    def _on_edited(self) -> None:
        self.refresh_preview()
        self.changed.emit()

    def _confirm_reset(self) -> None:
        dialog = ConfirmDialog(
            "Сбросить промпт?",
            "Текст вернётся к значению по умолчанию. Ваши изменения будут потеряны.",
            self._theme,
            confirm_text="Сбросить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.setText(self._default_text)
            self.changed.emit()
