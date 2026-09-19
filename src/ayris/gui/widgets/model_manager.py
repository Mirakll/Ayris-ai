"""The model manager: a catalog of models grouped by kind, with downloads.

This is the working half of the «Обновления» tab. It reads the catalog and the
installed rows through a small :class:`ModelManagerBackend` seam — a real one
wraps :class:`~ayris.models.registry.ModelRegistry`, a fake one stands in for
tests — builds one :class:`~ayris.gui.widgets.model_card.ModelCard` per model
grouped by kind, and keeps them in step with the download events on the bus.

Three things earn their weight here:

*Downloads outlive the window.* The bytes are owned by a
:class:`DownloadCoordinator` that lives with the backend, not with this widget,
so closing the settings window tears down the cards while the transfer keeps
running. Reopening rebuilds from the backend and re-attaches to whatever is still
in flight.

*Progress is marshalled, then thrown at one card.* :class:`ModelDownloadProgress`
is published from the download thread, so it crosses to the GUI thread through a
relay and lands on the single card it names — never a full rebuild, which at
several events a second would be unusable. The card throttles the repaint.

*Filters hide, they do not rebuild.* Search, kind, language and «только
установленные» toggle card visibility, so typing stays smooth and an in-flight
download is never interrupted by a re-layout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ayris.core.events import (
    ActiveModelChanged,
    EventBus,
    ModelDownloadFailed,
    ModelDownloadFinished,
    ModelDownloadProgress,
    ModelDownloadStarted,
    ModelRemoved,
)
from ayris.core.models import MODEL_KINDS, ModelRecord
from ayris.core.paths import ModelKind
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.empty_state import EmptyState
from ayris.gui.widgets.model_card import CardStatus, ModelCard, ModelView
from ayris.gui.widgets.notice import InlineNotice
from ayris.gui.widgets.search_field import SearchField
from ayris.gui.widgets.toggle import ToggleSwitch
from ayris.models.downloader import human_size
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.models.catalog import ModelCatalog, ModelEntry
    from ayris.models.registry import DiskUsage, IntegrityReport

__all__ = ["ModelManager", "ModelManagerBackend"]

_log = get_logger(__name__)

#: Kind → the section heading shown above its cards.
_KIND_LABELS: dict[str, str] = {
    "stt": "Распознавание речи (STT)",
    "tts": "Синтез речи (TTS)",
    "wake": "Активация (Wake)",
    "llm": "Языковая модель (LLM)",
}

#: Short kind labels for the filter combo, so a long heading never widens the row.
_KIND_SHORT: dict[str, str] = {
    "stt": "STT",
    "tts": "TTS",
    "wake": "Wake",
    "llm": "LLM",
}

_LANGUAGE_FILTER_LABELS: dict[str, str] = {
    "ru": "Русский",
    "en": "Английский",
    "multi": "Многоязычные",
    "any": "Любой язык",
}


class ModelManagerBackend(Protocol):
    """What the manager needs from the model registry, and nothing more.

    A real implementation wraps :class:`~ayris.models.registry.ModelRegistry`; a
    test supplies a fake so no download touches the network. Every method here is
    a read or a single registry call — the widget owns no disk and no socket.
    """

    def catalog(self) -> ModelCatalog: ...

    def installed(self) -> list[ModelRecord]: ...

    def active(self, kind: ModelKind) -> ModelRecord | None: ...

    def disk_usage(self) -> DiskUsage: ...

    def free_disk_bytes(self) -> int: ...

    def set_active(self, record: ModelRecord) -> None: ...

    def remove(self, record: ModelRecord, *, force: bool) -> int: ...

    def verify(self, record: ModelRecord, *, full: bool) -> IntegrityReport: ...

    def start_download(self, model_id: str) -> None: ...

    def cancel_download(self, model_id: str) -> None: ...

    def is_downloading(self, model_id: str) -> bool: ...


class _BusRelay(QObject):
    """Marshals bus events onto the GUI thread.

    Download progress is published from a worker thread; touching a widget from
    there is undefined. Each Qt signal here is emitted from the subscriber and
    delivered on the thread that owns the relay — the GUI thread — as a queued
    call, exactly the pattern the settings base class uses for config changes.
    """

    progress = Signal(object)
    started = Signal(object)
    finished = Signal(object)
    failed = Signal(object)
    removed = Signal(object)
    active = Signal(object)


class ModelManager(QWidget):
    """Catalog of models grouped by kind, with search, filters and downloads."""

    def __init__(
        self,
        backend: ModelManagerBackend,
        theme: ThemeManager,
        bus: EventBus | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._theme = theme
        self._bus = bus
        self._cards: dict[str, ModelCard] = {}
        self._sections: dict[str, QWidget] = {}
        self._groups: dict[str, QButtonGroup] = {}
        self._unsubscribers: list[Callable[[], None]] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._build_controls()

        self._usage = QLabel("")
        self._usage.setProperty("role", "muted")
        self._usage.setWordWrap(True)
        self._layout.addWidget(self._usage)

        self._empty = EmptyState(
            "Ничего не найдено",
            "Измените запрос или фильтры, чтобы увидеть модели.",
            theme,
        )
        self._empty.hide()
        self._layout.addWidget(self._empty)

        self._sections_box = QWidget()
        self._sections_box.setProperty("transparent", True)
        self._sections_layout = QVBoxLayout(self._sections_box)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(theme.metric("spacing_md"))
        self._layout.addWidget(self._sections_box)

        self._relay = _BusRelay(self)
        self._relay.progress.connect(self._on_progress)
        self._relay.started.connect(self._on_started)
        self._relay.finished.connect(self._on_finished)
        self._relay.failed.connect(self._on_failed)
        self._relay.removed.connect(self._on_removed)
        self._relay.active.connect(self._on_active_changed)
        self._subscribe(bus)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self.rebuild()

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> None:
        controls = QHBoxLayout()
        self._search = SearchField(placeholder="Поиск моделей", theme=self._theme)
        self._search.search_changed.connect(lambda _text: self._apply_filters())
        controls.addWidget(self._search, 1)

        self._kind_combo = ThemedComboBox()
        self._kind_combo.addItem("Все виды", "")
        for kind in MODEL_KINDS:
            self._kind_combo.addItem(_KIND_SHORT.get(kind, kind), kind)
        self._kind_combo.currentIndexChanged.connect(self._apply_filters)
        _tame_combo(self._kind_combo)
        controls.addWidget(self._kind_combo)

        self._language_combo = ThemedComboBox()
        self._language_combo.addItem("Все языки", "")
        self._language_combo.currentIndexChanged.connect(self._apply_filters)
        _tame_combo(self._language_combo)
        controls.addWidget(self._language_combo)

        self._layout.addLayout(controls)

        installed_row = QHBoxLayout()
        self._only_installed = ToggleSwitch(self._theme, label="Только установленные")
        self._only_installed.toggled.connect(lambda _checked: self._apply_filters())
        installed_label = QLabel("Только установленные")
        installed_label.setProperty("role", "secondary")
        installed_row.addWidget(self._only_installed)
        installed_row.addWidget(installed_label)
        installed_row.addStretch(1)
        self._layout.addLayout(installed_row)

    def _refill_language_filter(self, views: Iterable[ModelView]) -> None:
        languages = sorted({view.language for view in views if view.language})
        current = self._language_combo.currentData()
        blocked = self._language_combo.blockSignals(True)
        self._language_combo.clear()
        self._language_combo.addItem("Все языки", "")
        for code in languages:
            self._language_combo.addItem(_LANGUAGE_FILTER_LABELS.get(code, code), code)
        index = self._language_combo.findData(current)
        self._language_combo.setCurrentIndex(index if index >= 0 else 0)
        self._language_combo.blockSignals(blocked)

    # -- data ---------------------------------------------------------------

    def rebuild(self) -> None:
        """Recreate every section and card from the backend's current state.

        Called on first show and after a structural change — install, delete —
        never on a progress tick, which only nudges a single card.
        """
        self._clear_sections()
        views = self._collect_views()
        self._refill_language_filter(views)
        by_kind: dict[str, list[ModelView]] = {}
        for view in views:
            by_kind.setdefault(view.kind, []).append(view)

        for kind in MODEL_KINDS:
            entries = by_kind.get(kind, [])
            if not entries:
                continue
            self._add_section(kind, entries)
        self._update_usage()
        self._apply_filters()

    def _collect_views(self) -> list[ModelView]:
        catalog = self._backend.catalog()
        installed = {record.catalog_id: record for record in self._backend.installed()}
        local = [record for record in self._backend.installed() if not record.catalog_id]
        active_ids = {
            record.id for kind in MODEL_KINDS if (record := self._backend.active(kind)) is not None
        }

        views: list[ModelView] = []
        seen: set[str] = set()
        for entry in catalog:
            record = installed.get(entry.id)
            views.append(self._view_for(entry, record, active_ids))
            seen.add(entry.id)
        # A model the user dropped in by hand still deserves a card so it can be
        # activated, verified or removed — even without a catalog entry.
        for record in local:
            model_id = f"local:{record.kind}:{record.name}"
            if model_id in seen:
                continue
            views.append(self._local_view(model_id, record, active_ids))
        return views

    def _view_for(
        self,
        entry: ModelEntry,
        record: ModelRecord | None,
        active_ids: set[int | None],
    ) -> ModelView:
        if self._backend.is_downloading(entry.id):
            status = CardStatus.DOWNLOADING
        elif record is not None:
            status = CardStatus.INSTALLED
        else:
            status = CardStatus.AVAILABLE
        return ModelView(
            model_id=entry.id,
            kind=entry.kind,
            name=entry.name,
            engine=entry.engine,
            status=status,
            entry=entry,
            record=record,
            description=entry.description,
            language=entry.language,
            download_bytes=entry.total_bytes,
            installed_bytes=record.size_bytes if record is not None else 0,
            requires_ram_mb=entry.requires_ram_mb,
            requires_gpu=entry.requires_gpu,
            is_active=record is not None and record.id in active_ids,
        )

    def _local_view(
        self, model_id: str, record: ModelRecord, active_ids: set[int | None]
    ) -> ModelView:
        return ModelView(
            model_id=model_id,
            kind=record.kind,
            name=record.name,
            engine=record.engine or "—",
            status=CardStatus.INSTALLED,
            record=record,
            description="Установлена вручную, не из каталога.",
            installed_bytes=record.size_bytes,
            is_active=record.id in active_ids,
        )

    def _add_section(self, kind: str, views: list[ModelView]) -> None:
        section = QWidget()
        section.setProperty("transparent", True)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))

        heading = QLabel(_KIND_LABELS.get(kind, kind))
        heading.setProperty("role", "h2")
        layout.addWidget(heading)

        group = QButtonGroup(section)
        group.setExclusive(True)
        self._groups[kind] = group
        for view in views:
            card = ModelCard(view, self._theme)
            card.download_requested.connect(self._download)
            card.cancel_requested.connect(self._cancel)
            card.delete_requested.connect(self._delete)
            card.verify_requested.connect(self._verify)
            card.retry_requested.connect(self._download)
            card.activate_requested.connect(self._activate)
            group.addButton(card.radio)
            layout.addWidget(card)
            self._cards[view.model_id] = card

        self._sections_layout.addWidget(section)
        self._sections[kind] = section

    def _clear_sections(self) -> None:
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        for section in self._sections.values():
            section.setParent(None)
            section.deleteLater()
        self._cards.clear()
        self._sections.clear()
        self._groups.clear()

    # -- filtering ----------------------------------------------------------

    def _apply_filters(self, *_args: object) -> None:
        query = self._search.text().casefold().strip()
        wanted_kind = self._kind_combo.currentData()
        wanted_language = self._language_combo.currentData()
        only_installed = self._only_installed.isChecked()

        any_visible = False
        for kind, section in self._sections.items():
            section_visible = False
            for card in self._cards.values():
                view = card.view()
                if view.kind != kind:
                    continue
                visible = True
                if wanted_kind and view.kind != wanted_kind:
                    visible = False
                if wanted_language and view.language != wanted_language:
                    visible = False
                if only_installed and not view.is_installed:
                    visible = False
                if query and query not in view.search_text:
                    visible = False
                card.setVisible(visible)
                section_visible = section_visible or visible
            section.setVisible(section_visible)
            any_visible = any_visible or section_visible

        self._empty.setVisible(not any_visible)
        self._sections_box.setVisible(any_visible)

    # -- usage --------------------------------------------------------------

    def _update_usage(self) -> None:
        usage = self._backend.disk_usage()
        free = self._backend.free_disk_bytes()
        parts: list[str] = []
        for kind in MODEL_KINDS:
            used = usage.by_kind.get(kind, 0)
            if used <= 0:
                continue
            short = _KIND_LABELS.get(kind, kind).split(" (")[0]
            parts.append(f"{short}: {human_size(used)}")
        prefix = "Занято моделями: " + (", ".join(parts) if parts else "пусто")
        total = f"всего {human_size(usage.total)}"
        free_text = f"свободно на диске {human_size(free)}"
        self._usage.setText(f"{prefix}. {total}; {free_text}.")

    # -- actions ------------------------------------------------------------

    def _card(self, model_id: str) -> ModelCard | None:
        return self._cards.get(model_id)

    def _download(self, model_id: str) -> None:
        card = self._card(model_id)
        if card is not None:
            card.set_status(CardStatus.QUEUED)
        try:
            self._backend.start_download(model_id)
        except Exception as exc:  # a rejected start is shown, not swallowed
            _log.exception("не удалось начать загрузку %s", model_id)
            if card is not None:
                card.set_status(CardStatus.ERROR, detail=_message(exc))

    def _cancel(self, model_id: str) -> None:
        try:
            self._backend.cancel_download(model_id)
        except Exception:
            _log.exception("не удалось отменить загрузку %s", model_id)
        card = self._card(model_id)
        if card is not None:
            view = card.view()
            card.set_status(CardStatus.INSTALLED if view.is_installed else CardStatus.AVAILABLE)

    def _activate(self, model_id: str) -> None:
        card = self._card(model_id)
        if card is None or card.view().record is None:
            return
        record = card.view().record
        try:
            self._backend.set_active(record)  # type: ignore[arg-type]
        except Exception as exc:
            _log.exception("не удалось выбрать активную модель %s", model_id)
            card.set_status(CardStatus.INSTALLED, detail=_message(exc))

    def _verify(self, model_id: str) -> None:
        card = self._card(model_id)
        if card is None or card.view().record is None:
            return
        record = card.view().record
        try:
            report = self._backend.verify(record, full=True)  # type: ignore[arg-type]
        except Exception as exc:
            _log.exception("не удалось проверить модель %s", model_id)
            card.set_status(CardStatus.ERROR, detail=_message(exc))
            return
        self._apply_report(card, report)

    def _apply_report(self, card: ModelCard, report: IntegrityReport) -> None:
        from ayris.models.registry import IntegrityStatus

        if report.status == IntegrityStatus.OK:
            card.set_status(CardStatus.INSTALLED, detail="")
            self._notice("Модель в порядке: контрольная сумма совпала.", "success")
        elif report.status == IntegrityStatus.UNVERIFIED:
            card.set_status(CardStatus.INSTALLED, detail="")
            self._notice("Проверить нечем: нет записанной контрольной суммы.", "info")
        elif report.status == IntegrityStatus.MISSING:
            card.set_status(CardStatus.MISSING, detail=report.detail or "Файлы модели пропали.")
        else:
            card.set_status(
                CardStatus.CORRUPTED,
                detail=(report.detail or "Контрольная сумма не совпала.")
                + " Модель можно скачать заново.",
            )

    def _delete(self, model_id: str) -> None:
        card = self._card(model_id)
        if card is None or card.view().record is None:
            return
        record = card.view().record
        view = card.view()
        message = f"Файлы модели «{view.name}» будут удалены с диска."
        if view.is_active:
            message += "\nЭто активная модель для своего вида — её работа прервётся."
        dialog = ConfirmDialog(
            "Удалить модель?",
            message,
            self._theme,
            confirm_text="Удалить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            freed = self._backend.remove(record, force=True)  # type: ignore[arg-type]
        except Exception as exc:
            _log.exception("не удалось удалить модель %s", model_id)
            card.set_status(card.view().status, detail=_message(exc))
            return
        self._notice(f"Модель удалена, освобождено {human_size(freed)}.", "success")
        self.rebuild()

    # -- events -------------------------------------------------------------

    def _on_progress(self, event: ModelDownloadProgress) -> None:
        card = self._card(event.model_id)
        if card is not None:
            card.apply_progress(event.downloaded, event.total, event.speed_bps, event.eta_s)

    def _on_started(self, event: ModelDownloadStarted) -> None:
        card = self._card(event.model_id)
        if card is not None:
            card.set_status(CardStatus.DOWNLOADING)
            card.apply_progress(event.downloaded, event.total, 0.0, 0.0)

    def _on_finished(self, event: ModelDownloadFinished) -> None:
        card = self._card(event.model_id)
        if card is not None:
            card.finish_progress()
        self.rebuild()
        self._notice("Модель установлена.", "success")

    def _on_failed(self, event: ModelDownloadFailed) -> None:
        card = self._card(event.model_id)
        if card is None:
            return
        if event.cancelled:
            view = card.view()
            card.set_status(CardStatus.INSTALLED if view.is_installed else CardStatus.AVAILABLE)
            return
        card.set_status(
            CardStatus.ERROR,
            detail=event.user_message or "Загрузка не удалась.",
        )

    def _on_removed(self, _event: ModelRemoved) -> None:
        self.rebuild()

    def _on_active_changed(self, event: ActiveModelChanged) -> None:
        # Settle radios from the authoritative event: reflect the new active
        # model and clear the rest of its kind, without re-emitting activation.
        for card in self._cards.values():
            view = card.view()
            if view.kind != event.kind or not view.is_installed:
                continue
            record = view.record
            is_active = record is not None and not event.cleared and record.id == event.model_id
            card.set_active(is_active)

    # -- notices ------------------------------------------------------------

    def _notice(self, text: str, kind: str) -> None:
        # auto_hide_ms=0 keeps the notice's own timer off; the context-bound
        # single-shot below is cancelled automatically if the notice is destroyed
        # first, so a closed settings window never fires a timer at a dead widget.
        notice = InlineNotice(text, self._theme, kind=kind, auto_hide_ms=0)  # type: ignore[arg-type]
        self._layout.insertWidget(self._layout.indexOf(self._usage), notice)
        notice.closed.connect(notice.deleteLater)
        QTimer.singleShot(5000, notice, notice.dismiss)

    # -- lifecycle ----------------------------------------------------------

    def _subscribe(self, bus: EventBus | None) -> None:
        if bus is None:
            return
        self._unsubscribers = [
            bus.subscribe(ModelDownloadProgress, self._relay.progress.emit),
            bus.subscribe(ModelDownloadStarted, self._relay.started.emit),
            bus.subscribe(ModelDownloadFinished, self._relay.finished.emit),
            bus.subscribe(ModelDownloadFailed, self._relay.failed.emit),
            bus.subscribe(ModelRemoved, self._relay.removed.emit),
            bus.subscribe(ActiveModelChanged, self._relay.active.emit),
        ]

    def dispose(self) -> None:
        """Detach from the bus. Never cancels downloads — they outlive the tab."""
        for unsubscribe in self._unsubscribers:
            try:
                unsubscribe()
            except Exception:
                _log.exception("не удалось отписаться от шины в менеджере моделей")
        self._unsubscribers.clear()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        self._layout.setSpacing(self._theme.metric("spacing_md"))


def _message(exc: Exception) -> str:
    """The Russian user message of an :class:`AyrisError`, else its text."""
    user_message = getattr(exc, "user_message", "")
    return user_message or str(exc) or "Произошла ошибка."


def _tame_combo(combo: ThemedComboBox) -> None:
    """Keep a filter combo from widening the controls row: size to a short length."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
