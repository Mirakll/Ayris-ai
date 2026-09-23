"""Export a profile to a portable ``.zip``: what goes in, and where it lands.

The dialog gathers the two real choices the archive format offers — whether the
settings and the custom sounds travel with the commands — and a destination path.
Commands, folders, variables and the model manifest are always included: they are
what a profile *is*, so they show as ticked and disabled rather than as a way to
export an empty bundle. It applies nothing; the tab reads :attr:`destination`,
:attr:`include_settings` and :attr:`include_sounds` and calls
:meth:`ProfileManager.export` on a background thread.

The one thing the dialog states outright, not in fine print, is that **API keys do
not travel with the archive**: they live in the Windows Credential Manager and only
a reference to them is in ``config.toml``, which export strips as well. The notice
says the keys must be entered again after importing elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.portable_profile import BUNDLE_SUFFIX
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.notice import InlineNotice
from ayris.models.downloader import human_size

__all__ = ["ProfileExportDialog"]


def _safe_stem(name: str) -> str:
    """A file-name stem from a profile name, spaces and separators tamed."""
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return cleaned.replace(" ", "_") or "профиль"


class ProfileExportDialog(QDialog):
    """Composition and destination for a profile export."""

    def __init__(
        self,
        profile_name: str,
        theme: ThemeManager,
        *,
        estimate: Callable[[bool], int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._estimate = estimate
        self._destination: Path | None = None
        self._default_name = f"{_safe_stem(profile_name)}{BUNDLE_SUFFIX}"
        self.setWindowTitle("Экспорт профиля")
        self.setAccessibleName("Экспорт профиля")
        self.setModal(True)

        self._layout = QVBoxLayout(self)
        heading = QLabel(f"Экспорт профиля «{profile_name}»")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)

        caption = QLabel("Выберите, что войдёт в архив. Его можно перенести на другой компьютер.")
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._layout.addWidget(caption)

        self._commands = _mandatory_check("Команды и папки")
        self._variables = _mandatory_check("Переменные профиля")
        self._models = _mandatory_check("Список моделей (манифесты, не сами файлы)")
        self._settings = QCheckBox("Настройки (без секретов)")
        self._settings.setChecked(True)
        self._sounds = QCheckBox("Пользовательские звуки")
        self._sounds.setChecked(True)
        for box in (self._commands, self._variables, self._settings, self._sounds, self._models):
            self._layout.addWidget(box)
        self._sounds.toggled.connect(self._update_estimate)

        self._notice = InlineNotice(
            "API-ключи и токены в архив не попадают — их хранит Windows Credential "
            "Manager, а в настройках остаётся только ссылка на них. После импорта на "
            "другом компьютере ключи придётся ввести заново.",
            theme,
            kind="warning",
        )
        self._notice.close_button.hide()
        self._layout.addWidget(self._notice)

        self._size = QLabel("")
        self._size.setProperty("role", "secondary")
        self._layout.addWidget(self._size)

        destination = QHBoxLayout()
        self._path_label = QLabel("Путь не выбран")
        self._path_label.setProperty("role", "secondary")
        self._path_label.setWordWrap(True)
        browse = QPushButton("Обзор…")
        browse.clicked.connect(self._choose_destination)
        destination.addWidget(self._path_label, 1)
        destination.addWidget(browse)
        self._layout.addLayout(destination)

        buttons = QDialogButtonBox()
        self._export_button = QPushButton("Экспортировать")
        self._export_button.setProperty("kind", "primary")
        self._export_button.setEnabled(False)
        cancel = QPushButton("Отмена")
        self._export_button.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addButton(cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(self._export_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self._layout.addWidget(buttons)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self._update_estimate()

    @property
    def include_settings(self) -> bool:
        return self._settings.isChecked()

    @property
    def include_sounds(self) -> bool:
        return self._sounds.isChecked()

    @property
    def destination(self) -> Path | None:
        return self._destination

    def set_destination(self, path: Path) -> None:
        """Set the target archive path (used by the browse button and by tests)."""
        self._destination = path
        self._path_label.setText(f"Сохранить в:\n{path}")
        self._export_button.setEnabled(True)

    def _choose_destination(self) -> None:
        start = self._destination or Path.home() / self._default_name
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Сохранить профиль", str(start), f"Архив профиля (*{BUNDLE_SUFFIX})"
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() != BUNDLE_SUFFIX:
            path = path.with_suffix(BUNDLE_SUFFIX)
        self.set_destination(path)

    def _update_estimate(self, _checked: bool | None = None) -> None:
        try:
            estimated = self._estimate(self.include_sounds)
        except Exception:  # an estimate must never block the export dialog
            self._size.setText("")
            return
        self._size.setText(f"Примерный размер архива: ~{human_size(estimated)}")

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_md"))
        self.setMinimumWidth(self._theme.metric("dialog_width"))


def _mandatory_check(text: str) -> QCheckBox:
    """A ticked, disabled box for a part that is always exported."""
    box = QCheckBox(text)
    box.setChecked(True)
    box.setEnabled(False)
    box.setToolTip("Всегда входит в архив")
    return box
