"""Шаг «Профиль»: с чего начать — примеры, импорт из VoiceAttack или чисто.

Импорт делает внедрённый :class:`ProfileImporter` (в приложении — обёртка над
``CommandTreeStore`` и импортёром VoiceAttack, в тестах — заглушка), поэтому шаг
не тянет за собой БД и парсер .vap. Перед применением показывается предпросмотр:
список команд, которые добавятся. Пустой профиль ничего не пишет.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep
from ayris.utils.logger import get_logger

__all__ = ["ProfileImporter", "ProfileStep"]

_log = get_logger(__name__)


class ProfileImporter(Protocol):
    """Узкий контракт импорта, чтобы шаг не зависел от БД и парсеров."""

    def preview_examples(self) -> list[str]:
        """Имена команд из набора примеров (без записи)."""

    def import_examples(self) -> int:
        """Добавить примеры в профиль, вернуть число добавленных команд."""

    def preview_voiceattack(self, path: Path) -> list[str]:
        """Имена команд из файла VoiceAttack ``.vap`` (без записи)."""

    def import_voiceattack(self, path: Path) -> int:
        """Импортировать ``.vap`` в профиль, вернуть число добавленных команд."""


class ProfileStep(WizardStep):
    """Стартовое наполнение профиля командами."""

    def __init__(
        self,
        theme: ThemeManager,
        importer: ProfileImporter | None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = "profile"
        self.title = "Профиль"
        self._importer = importer
        self._vap_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("С чего начать"))
        layout.addWidget(caption("Команды можно добавлять и менять позже на вкладке «Команды»."))

        self._group = QButtonGroup(self)
        self._clean = QRadioButton("Чистый профиль")
        self._examples = QRadioButton("Примеры команд")
        self._voiceattack = QRadioButton("Импорт из VoiceAttack (.vap)")
        for button in (self._clean, self._examples, self._voiceattack):
            self._group.addButton(button)
            layout.addWidget(button)
        self._examples.setChecked(True)

        # placeholder-profile-body

        self._pick_row = QHBoxLayout()
        self._pick_btn = QPushButton("Выбрать файл .vap…")
        self._pick_btn.clicked.connect(self._choose_vap)
        self._pick_row.addWidget(self._pick_btn)
        self._pick_caption = caption("Файл не выбран.")
        self._pick_row.addWidget(self._pick_caption, 1)
        layout.addLayout(self._pick_row)

        self._preview = QListWidget()
        self._preview.setEnabled(False)
        layout.addWidget(self._preview, 1)

        if self._importer is None:
            self._examples.setEnabled(False)
            self._voiceattack.setEnabled(False)
            self._clean.setChecked(True)

        for button in (self._clean, self._examples, self._voiceattack):
            button.toggled.connect(self._on_choice)
        self._on_choice()

    # -- выбор варианта ----------------------------------------------------

    def _on_choice(self) -> None:
        voiceattack = self._voiceattack.isChecked()
        self._pick_btn.setVisible(voiceattack)
        self._pick_caption.setVisible(voiceattack)
        self._refresh_preview()
        self.completion_changed.emit()

    def _choose_vap(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Файл VoiceAttack", "", "Профили VoiceAttack (*.vap)"
        )
        if selected:
            self._vap_path = Path(selected)
            self._pick_caption.setText(self._vap_path.name)
            self._refresh_preview()
            self.completion_changed.emit()

    def _refresh_preview(self) -> None:
        self._preview.clear()
        if self._importer is None:
            return
        try:
            if self._examples.isChecked():
                names = self._importer.preview_examples()
            elif self._voiceattack.isChecked() and self._vap_path is not None:
                names = self._importer.preview_voiceattack(self._vap_path)
            else:
                names = []
        except Exception:
            _log.exception("не удалось построить предпросмотр импорта")
            names = []
        self._preview.addItems(names)

    # -- контракт шага -----------------------------------------------------

    def validate(self) -> str | None:
        if self._voiceattack.isChecked() and self._vap_path is None:
            return "Выберите файл .vap или выберите другой вариант."
        return None

    def apply(self) -> None:
        if self._importer is None:
            return
        if self._examples.isChecked():
            count = self._importer.import_examples()
            _log.info("онбординг: добавлено примеров команд: %d", count)
        elif self._voiceattack.isChecked() and self._vap_path is not None:
            count = self._importer.import_voiceattack(self._vap_path)
            _log.info("онбординг: импортировано из VoiceAttack: %d", count)
