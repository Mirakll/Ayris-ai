"""Мелкие общие элементы оформления шагов, чтобы заголовки и подписи выглядели
одинаково во всех шагах и брали цвета из ролей темы (``h2``/``secondary``)."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

__all__ = ["caption", "heading"]


def heading(text: str) -> QLabel:
    """Заголовок шага (роль темы ``h2``)."""
    label = QLabel(text)
    label.setProperty("role", "h2")
    label.setWordWrap(True)
    return label


def caption(text: str) -> QLabel:
    """Пояснительная подпись (приглушённый вторичный текст)."""
    label = QLabel(text)
    label.setProperty("role", "secondary")
    label.setWordWrap(True)
    return label
