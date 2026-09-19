"""One model in the manager: its facts, its status, and what can be done to it.

A card is a pure view. It never downloads, deletes or verifies anything itself —
it emits a signal and lets :class:`~ayris.gui.widgets.model_manager.ModelManager`
drive the backend, so the same card works against a real registry and against a
fake in tests. :meth:`update_view` re-renders the static facts, :meth:`set_status`
and :meth:`apply_progress` move the mutable state, and the embedded
:class:`~ayris.gui.widgets.download_progress.DownloadProgress` appears only while
bytes are moving.

The active-model choice is a radio button the card exposes but does not own: the
manager puts every card of one kind into a single :class:`QButtonGroup`, so
choosing one deselects the rest, which is what «выбор активной модели на вид»
means. Selecting emits :attr:`activate_requested`; the manager calls the registry
and lets the resulting :class:`~ayris.core.events.ActiveModelChanged` settle the
radios, so a rejected activation does not leave a lie on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.download_progress import DownloadProgress
from ayris.models.downloader import human_size

if TYPE_CHECKING:
    from ayris.core.models import ModelRecord
    from ayris.core.paths import ModelKind
    from ayris.models.catalog import ModelEntry

__all__ = ["CardStatus", "ModelCard", "ModelView"]


class CardStatus(StrEnum):
    """Where a model stands, as far as the card shows."""

    AVAILABLE = "available"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    ERROR = "error"


#: Status → (Russian label, badge colour class understood by the stylesheet).
_STATUS_TEXT: dict[CardStatus, tuple[str, str]] = {
    CardStatus.AVAILABLE: ("Доступна", "muted"),
    CardStatus.QUEUED: ("В очереди", "info"),
    CardStatus.DOWNLOADING: ("Загрузка", "info"),
    CardStatus.INSTALLED: ("Установлена", "success"),
    CardStatus.CORRUPTED: ("Повреждена", "error"),
    CardStatus.MISSING: ("Файлы пропали", "warning"),
    CardStatus.ERROR: ("Ошибка", "error"),
}


@dataclass(frozen=True, slots=True)
class ModelView:
    """Everything a card shows about one model, catalog and disk merged.

    Either ``entry`` (a catalog model that can be downloaded) or ``record`` (a
    model already on disk) is always set; usually both, for a catalog model that
    is installed. The properties prefer catalog facts for description and
    requirements and disk facts for the installed size.
    """

    model_id: str
    kind: ModelKind
    name: str
    engine: str
    status: CardStatus
    entry: ModelEntry | None = None
    record: ModelRecord | None = None
    description: str = ""
    language: str = ""
    download_bytes: int = 0
    installed_bytes: int = 0
    requires_ram_mb: int = 0
    requires_gpu: bool = False
    is_active: bool = False
    detail: str = ""

    @property
    def is_installed(self) -> bool:
        return self.record is not None

    @property
    def can_download(self) -> bool:
        return self.entry is not None

    @property
    def search_text(self) -> str:
        return f"{self.name} {self.engine} {self.description} {self.language}".casefold()


#: Languages spelled out; anything else is shown verbatim.
_LANGUAGE_LABELS: dict[str, str] = {
    "ru": "Русский",
    "en": "Английский",
    "multi": "Многоязычная",
    "any": "Любой язык",
    "": "",
}


class ModelCard(QFrame):
    """A settings card for one model with download, verify and delete actions."""

    download_requested = Signal(str)
    cancel_requested = Signal(str)
    delete_requested = Signal(str)
    verify_requested = Signal(str)
    retry_requested = Signal(str)
    #: Emitted when the user picks this card's radio; carries the model id.
    activate_requested = Signal(str)

    def __init__(self, view: ModelView, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._view = view
        self.setProperty("card", True)
        self.setAccessibleName(view.name)

        self._layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._name = QLabel(view.name)
        self._name.setProperty("role", "h2")
        self._name.setWordWrap(True)
        self._status = QLabel("")
        self._status.setProperty("badge", True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._name, 1)
        header.addWidget(self._status, 0, Qt.AlignmentFlag.AlignTop)
        self._layout.addLayout(header)

        self._meta = QLabel("")
        self._meta.setProperty("role", "muted")
        self._meta.setWordWrap(True)
        self._layout.addWidget(self._meta)

        self._description = QLabel("")
        self._description.setProperty("role", "secondary")
        self._description.setWordWrap(True)
        self._layout.addWidget(self._description)

        self._progress = DownloadProgress(theme)
        self._progress.cancel_requested.connect(
            lambda: self.cancel_requested.emit(self._view.model_id)
        )
        self._progress.hide()
        self._layout.addWidget(self._progress)

        self._error = QLabel("")
        self._error.setProperty("role", "muted")
        self._error.setProperty("badge", "error")
        self._error.setWordWrap(True)
        self._error.hide()
        self._layout.addWidget(self._error)

        actions = QHBoxLayout()
        self._radio = QRadioButton("Активная")
        self._radio.setAccessibleName(f"Сделать «{view.name}» активной")
        self._radio.toggled.connect(self._on_radio)
        actions.addWidget(self._radio)
        actions.addStretch(1)
        self._verify_button = QPushButton("Проверить")
        self._verify_button.setAccessibleName(f"Проверить целостность «{view.name}»")
        self._verify_button.clicked.connect(lambda: self.verify_requested.emit(self._view.model_id))
        self._delete_button = QPushButton("Удалить")
        self._delete_button.setProperty("kind", "danger")
        self._delete_button.setAccessibleName(f"Удалить «{view.name}»")
        self._delete_button.clicked.connect(lambda: self.delete_requested.emit(self._view.model_id))
        self._action_button = QPushButton("Скачать")
        self._action_button.setProperty("kind", "primary")
        self._action_button.clicked.connect(self._on_action)
        actions.addWidget(self._verify_button)
        actions.addWidget(self._delete_button)
        actions.addWidget(self._action_button)
        self._layout.addLayout(actions)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self.update_view(view)

    # -- radio grouping -----------------------------------------------------

    @property
    def radio(self) -> QRadioButton:
        """The active-model radio; the manager adds it to a per-kind group."""
        return self._radio

    def set_active(self, active: bool) -> None:
        """Reflect the active model without emitting :attr:`activate_requested`."""
        blocked = self._radio.blockSignals(True)
        self._radio.setChecked(active)
        self._radio.blockSignals(blocked)
        self._view = _replace_active(self._view, active)

    # -- rendering ----------------------------------------------------------

    def view(self) -> ModelView:
        return self._view

    def update_view(self, view: ModelView) -> None:
        """Re-render the static facts and re-derive the button states."""
        self._view = view
        self._name.setText(view.name)
        self._description.setText(view.description)
        self._description.setVisible(bool(view.description))
        self._meta.setText(self._meta_text(view))
        self.set_active(view.is_active)
        self.set_status(view.status, detail=view.detail)

    def set_status(self, status: CardStatus, *, detail: str = "") -> None:
        """Move the mutable state: badge, buttons, progress and error line."""
        self._view = _replace_status(self._view, status, detail)
        label, colour = _STATUS_TEXT[status]
        self._status.setText(label)
        self._status.setProperty("badge", colour)
        _restyle(self._status)

        downloading = status in (CardStatus.DOWNLOADING, CardStatus.QUEUED)
        self._progress.setVisible(downloading)
        if not downloading:
            self._progress.reset()
        if status == CardStatus.QUEUED:
            self._progress.set_progress(0, self._view.download_bytes, 0.0, 0.0)

        installed = status in (CardStatus.INSTALLED, CardStatus.CORRUPTED, CardStatus.MISSING)
        self._verify_button.setVisible(installed)
        self._delete_button.setVisible(installed)
        self._radio.setVisible(installed)
        self._radio.setEnabled(status == CardStatus.INSTALLED)

        self._action_button.setVisible(not downloading)
        self._action_button.setEnabled(self._view.can_download)
        if status in (CardStatus.CORRUPTED, CardStatus.MISSING, CardStatus.ERROR):
            self._action_button.setText("Скачать заново")
            self._action_button.setVisible(self._view.can_download)
        elif installed:
            self._action_button.setVisible(False)
        else:
            self._action_button.setText("Скачать")

        show_error = bool(detail) and status in (
            CardStatus.CORRUPTED,
            CardStatus.MISSING,
            CardStatus.ERROR,
        )
        self._error.setText(detail)
        self._error.setVisible(show_error)

    def apply_progress(self, downloaded: int, total: int, speed_bps: float, eta_s: float) -> None:
        """Feed one throttled progress sample to the embedded bar."""
        if not self._progress.isVisible():
            self.set_status(CardStatus.DOWNLOADING)
        self._progress.set_progress(downloaded, total, speed_bps, eta_s)

    def finish_progress(self) -> None:
        """Draw the final sample immediately — a transfer just ended."""
        self._progress.flush()

    # -- internals ----------------------------------------------------------

    def _on_action(self) -> None:
        if self._view.status in (CardStatus.CORRUPTED, CardStatus.MISSING, CardStatus.ERROR):
            self.retry_requested.emit(self._view.model_id)
        else:
            self.download_requested.emit(self._view.model_id)

    def _on_radio(self, checked: bool) -> None:
        if checked:
            self.activate_requested.emit(self._view.model_id)

    def _meta_text(self, view: ModelView) -> str:
        parts: list[str] = [f"Движок: {view.engine}"]
        language = _LANGUAGE_LABELS.get(view.language, view.language)
        if language:
            parts.append(language)
        if view.is_installed and view.installed_bytes:
            parts.append(f"на диске {human_size(view.installed_bytes)}")
        elif view.download_bytes:
            parts.append(f"загрузка {human_size(view.download_bytes)}")
        if view.requires_ram_mb:
            parts.append(f"память ~{_ram(view.requires_ram_mb)}")
        if view.requires_gpu:
            parts.append("нужен GPU")
        return "   ·   ".join(parts)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_lg")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_sm"))


def _ram(mb: int) -> str:
    """``2048`` → ``2 ГБ``; small values stay in megabytes."""
    if mb >= 1024:
        gb = mb / 1024
        text = f"{gb:.0f}" if gb.is_integer() else f"{gb:.1f}"
        return f"{text.replace('.', ',')} ГБ"
    return f"{mb} МБ"


def _restyle(widget: QWidget) -> None:
    """Re-evaluate a property selector after a dynamic property changed."""
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


def _replace_status(view: ModelView, status: CardStatus, detail: str) -> ModelView:
    from dataclasses import replace

    return replace(view, status=status, detail=detail)


def _replace_active(view: ModelView, active: bool) -> ModelView:
    from dataclasses import replace

    return replace(view, is_active=active)
