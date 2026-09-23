"""Tab «Профили»: the profile list, export and import, backups and the data folder.

Everything here is a thin, testable shell over :class:`~ayris.core.profile.ProfileManager`
(task 06), which owns all the logic and all the safety — atomic import, a backup
before every destructive step, the integrity and free-space checks on a root move.
The tab adds only what a manager cannot: a list model the tests can read without a
window (:class:`ProfileListModel`), the two dialogs, and the rule that the slow file
operations — export, import, backup restore, moving the root — run on a background
thread through :class:`~ayris.gui.tabs.voice.AsyncRunner` so the interface never
freezes. Switching, renaming and creating a profile are quick database writes and
stay on the UI thread.

Two things the manager cannot know are injected through :class:`ProfilesServices`: a
predicate for «есть несохранённые правки в редакторе» (task 54) that gates a switch,
and a way to open the model manager (task 50) from the import report when a bundle
references models that are not installed.

Because profiles share one database and one installation root, a profile's «размер»
is its content — commands, folders, variables — not a byte count; the byte size that
matters, free space on the data disk, is shown on the data-folder card.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigManager
from ayris.core.errors import AyrisError
from ayris.core.events import EventBus
from ayris.core.models import Profile, VariableScope
from ayris.core.paths import APP_DIR_NAME, get_paths, native_path, native_path_problem
from ayris.core.portable_profile import (
    BUNDLE_SUFFIX,
    BundlePreview,
    ConflictPolicy,
    ImportReport,
)
from ayris.core.profile import MAX_BACKUPS, ProfilesChanged, ProfileSwitched
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.tabs.voice import AsyncRunner
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.backup_list import BackupList
from ayris.gui.widgets.busy_indicator import BusyIndicator
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.notice import InlineNotice
from ayris.gui.widgets.profile_export_dialog import ProfileExportDialog
from ayris.gui.widgets.profile_import_dialog import ProfileImportDialog
from ayris.models.downloader import human_size
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.profile import ProfileManager

__all__ = ["ProfileListModel", "ProfileRow", "ProfilesServices", "ProfilesTab"]

_log = get_logger(__name__)


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Russian count phrase: ``1 команда``, ``2 команды``, ``5 команд``."""
    tail = count % 100
    if 11 <= tail <= 14:
        word = many
    else:
        unit = count % 10
        word = one if unit == 1 else few if 2 <= unit <= 4 else many
    return f"{count} {word}"


def _format_date(value: datetime | None) -> str:
    return value.strftime("%d.%m.%Y") if value is not None else "—"


def _dir_size(directory: Path) -> int:
    """Total size of every file under ``directory``; ``0`` if it is missing."""
    total = 0
    try:
        for entry in directory.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


@dataclass(frozen=True, slots=True)
class ProfileRow:
    """One profile as the list shows it: identity, active flag and content counts."""

    profile: Profile
    active: bool
    commands: int
    folders: int
    variables: int

    @property
    def summary(self) -> str:
        return " · ".join(
            (
                _plural(self.commands, "команда", "команды", "команд"),
                _plural(self.folders, "папка", "папки", "папок"),
                _plural(self.variables, "переменная", "переменные", "переменных"),
            )
        )

    @property
    def created(self) -> str:
        return _format_date(self.profile.created_at)

    @property
    def display(self) -> str:
        mark = "● " if self.active else ""
        return f"{mark}{self.profile.name}\n{self.summary} · создан {self.created}"


class ProfileListModel:
    """The profile list without a widget: what the tab renders and the tests read.

    A plain object rather than a ``QAbstractListModel`` on purpose — the list is
    small, rebuilt wholesale on any change, and the tests want to assert its rows
    against a real :class:`ProfileManager` over a temporary database with no Qt in
    the picture.
    """

    def __init__(self, manager: ProfileManager) -> None:
        self._manager = manager
        self._rows: list[ProfileRow] = []
        self.refresh()

    def refresh(self) -> None:
        repos = self._manager.repositories
        rows: list[ProfileRow] = []
        for profile in self._manager.list_all():
            pid = profile.id
            if pid is None:
                continue
            folders = sum(1 for f in repos.folders.list_for_profile(pid) if f.profile_id == pid)
            variables = len(repos.variables.list_all(scope=VariableScope.PROFILE, profile_id=pid))
            rows.append(
                ProfileRow(
                    profile=profile,
                    active=profile.is_active,
                    commands=repos.commands.count(pid),
                    folders=folders,
                    variables=variables,
                )
            )
        self._rows = rows

    @property
    def rows(self) -> list[ProfileRow]:
        return list(self._rows)

    @property
    def active_row(self) -> ProfileRow | None:
        return next((row for row in self._rows if row.active), None)

    def row_for(self, profile_id: int) -> ProfileRow | None:
        return next((row for row in self._rows if row.profile.id == profile_id), None)

    def can_delete(self) -> bool:
        """Whether deleting is allowed at all: the last profile is protected."""
        return len(self._rows) > 1

    def count(self) -> int:
        return len(self._rows)


@dataclass(slots=True)
class ProfilesServices:
    """What the tab needs from the rest of Ayris beyond the manager — all optional."""

    profile_manager: ProfileManager | None = None
    #: True when the command editor (task 54) holds unsaved edits; gates a switch.
    unsaved_edits: Callable[[], bool] | None = None
    #: Opens the model manager (task 50) from the import report. ``None`` hides the link.
    open_model_manager: Callable[[], None] | None = None
    #: Install names present on disk, for flagging models a bundle is missing.
    installed_models: Callable[[], frozenset[str]] | None = None


def _default_profile_manager(bus: EventBus | None, config: ConfigManager) -> ProfileManager | None:
    """Build a manager over the live database, or ``None`` if that is not possible."""
    try:
        from ayris.core.database import get_database
        from ayris.core.profile import ProfileManager
        from ayris.core.repositories import Repositories

        repositories = Repositories(get_database())
        return ProfileManager(repositories, paths=get_paths(), bus=bus, config=config)
    except Exception:
        _log.exception("не удалось построить менеджер профилей для вкладки «Профили»")
        return None


class _ProfileRelay(QObject):
    """Marshals bus notifications about profiles onto the GUI thread."""

    changed = Signal()


def _restyle(widget: QWidget) -> None:
    """Re-run the stylesheet after a dynamic property changed."""
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


class _BusyRow(QWidget):
    """An indeterminate spinner and a status line for one background section.

    The portable-profile engine reports no percentage, so «прогресс» here is an
    honest indeterminate spinner beside a message rather than a fake bar.
    """

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._busy = BusyIndicator(theme, active=False)
        self._busy.hide()
        self._message = QLabel("")
        self._message.setProperty("role", "secondary")
        self._message.setWordWrap(True)
        self._message.hide()
        self._layout.addWidget(self._busy)
        self._layout.addWidget(self._message, 1)

    def set_busy(self, busy: bool, text: str = "") -> None:
        self._busy.setVisible(busy)
        self._busy.setActive(busy)
        if busy:
            self._message.setProperty("badge", "info")
            self._message.setText(text)
            _restyle(self._message)
            self._message.setVisible(bool(text))

    def show_message(self, text: str, kind: str) -> None:
        self._busy.setActive(False)
        self._busy.hide()
        self._message.setProperty("badge", kind)
        self._message.setText(text)
        _restyle(self._message)
        self._message.show()

    def clear(self) -> None:
        self._message.hide()


class _ReportDialog(QDialog):
    """Shows an :class:`ImportReport` after an import, with a jump to the models."""

    def __init__(
        self,
        report: ImportReport,
        theme: ThemeManager,
        *,
        open_model_manager: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._opener = open_model_manager
        self.setWindowTitle("Импорт завершён")
        self.setAccessibleName("Отчёт об импорте")
        self.setModal(True)
        self._layout = QVBoxLayout(self)

        heading = QLabel(f"Профиль «{report.profile_name}» обновлён")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)

        summary = QLabel(report.describe())
        summary.setProperty("role", "secondary")
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._layout.addWidget(summary)

        if report.missing_models:
            names = ", ".join(report.missing_models)
            self._layout.addWidget(
                InlineNotice(
                    f"Не хватает моделей: {names}. Команды сохранены, но не "
                    "заработают, пока модели не установлены.",
                    theme,
                    kind="warning",
                )
            )
        self._build_buttons(report)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _build_buttons(self, report: ImportReport) -> None:
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        if report.missing_models and self._opener is not None:
            open_button = QPushButton("Открыть менеджер моделей")
            open_button.clicked.connect(self._open_models)
            buttons.addWidget(open_button)
        close = QPushButton("Готово")
        close.setProperty("kind", "primary")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        self._layout.addLayout(buttons)

    def _open_models(self) -> None:
        if self._opener is not None:
            self._opener()
        self.accept()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_md"))
        self.setMinimumWidth(self._theme.metric("dialog_width"))


class ProfilesTab(SettingsTab):
    """The «Профили» settings page (task 57)."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        services: ProfilesServices | None = None,
    ) -> None:
        super().__init__("profiles", "Профили", (), manager, theme, bus)
        # The section owns no config keys, so the inherited «Сбросить секцию»
        # header button has nothing to reset — hide it.
        self.reset_button.hide()
        self.services = services if services is not None else ProfilesServices()
        if self.services.profile_manager is None:
            self.services.profile_manager = _default_profile_manager(bus, manager)
        self._pm = self.services.profile_manager
        self._bus = bus
        self._extra_unsub: list[Callable[[], None]] = []
        self._model: ProfileListModel | None = None
        self._pending_root: Path | None = None
        self._backup_status_row: _BusyRow | None = None
        self._backup_done_msg = ""

        self._export_runner = AsyncRunner()
        self._export_runner.finished.connect(self._on_export_done)
        self._export_runner.failed.connect(self._on_export_failed)
        self._import_runner = AsyncRunner()
        self._import_runner.finished.connect(self._on_import_done)
        self._import_runner.failed.connect(self._on_import_failed)
        self._backup_runner = AsyncRunner()
        self._backup_runner.finished.connect(self._on_backup_done)
        self._backup_runner.failed.connect(self._on_backup_failed)
        self._folder_runner = AsyncRunner()
        self._folder_runner.finished.connect(self._on_folder_moved)
        self._folder_runner.failed.connect(self._on_folder_failed)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self._content = QVBoxLayout(container)
        self._content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)
        self._init_sections(theme, bus)

    def _init_sections(self, theme: ThemeManager, bus: EventBus | None) -> None:
        if self._pm is None:
            self._content.addWidget(
                InlineNotice(
                    "Профили сейчас недоступны: не удалось открыть базу данных.",
                    theme,
                    kind="error",
                )
            )
            self._content.addStretch(1)
            return

        self._model = ProfileListModel(self._pm)
        self._profile_relay = _ProfileRelay(self)
        self._profile_relay.changed.connect(self._reload_profiles)
        if bus is not None:
            self._extra_unsub.append(bus.subscribe(ProfilesChanged, self._on_bus_event, weak=False))
            self._extra_unsub.append(bus.subscribe(ProfileSwitched, self._on_bus_event, weak=False))

        self._build_profiles()
        self._build_transfer()
        self._build_backups()
        self._build_data_folder()
        self._content.addStretch(1)
        self._reload_profiles()
        self._reload_backups()
        self._refresh_folder_card()

    def _add_header(self, text: str) -> None:
        header = QLabel(text)
        header.setProperty("role", "h2")
        self._content.addWidget(header)

    def _on_bus_event(self, _event: object) -> None:
        self._profile_relay.changed.emit()

    def dispose(self) -> None:
        for unsub in self._extra_unsub:
            unsub()
        self._extra_unsub.clear()
        super().dispose()

    # -- profiles list ------------------------------------------------------

    def _build_profiles(self) -> None:
        self._add_header("Профили")
        caption = QLabel(
            "Каждый профиль — свой набор команд, папок и переменных. Все профили "
            "живут в одной установке; активный отмечен точкой."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

        self._list = QListWidget()
        self._list.setAccessibleName("Список профилей")
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        self._list.itemDoubleClicked.connect(lambda _item: self._switch_selected())
        self._content.addWidget(self._list)

        row = QHBoxLayout()
        self._create_button = QPushButton("Создать")
        self._create_button.clicked.connect(self._create_profile)
        self._duplicate_button = QPushButton("Дублировать")
        self._duplicate_button.clicked.connect(self._duplicate_profile)
        self._rename_button = QPushButton("Переименовать")
        self._rename_button.clicked.connect(self._rename_profile)
        self._delete_button = QPushButton("Удалить")
        self._delete_button.clicked.connect(self._delete_profile)
        self._switch_button = QPushButton("Сделать активным")
        self._switch_button.setProperty("kind", "primary")
        self._switch_button.clicked.connect(self._switch_selected)
        for button in (
            self._create_button,
            self._duplicate_button,
            self._rename_button,
            self._delete_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self._switch_button)
        self._content.addLayout(row)
        self._profiles_status = _BusyRow(self._theme)
        self._content.addWidget(self._profiles_status)

    def _reload_profiles(self) -> None:
        model = self._model
        if model is None:
            return
        selected = self._current_id()
        model.refresh()
        blocker = QSignalBlocker(self._list)
        self._list.clear()
        for row in model.rows:
            item = QListWidgetItem(row.display)
            if row.profile.id is not None:
                item.setData(Qt.ItemDataRole.UserRole, row.profile.id)
            self._list.addItem(item)
        del blocker
        self._reselect(selected)
        self._sync_buttons()

    def _reselect(self, profile_id: int | None) -> None:
        target = profile_id
        if target is None and self._model is not None:
            active = self._model.active_row
            target = active.profile.id if active is not None else None
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == target:
                self._list.setCurrentItem(item)
                return

    def _current_id(self) -> int | None:
        if self._list.currentRow() < 0:
            return None
        item = self._list.currentItem()
        data = item.data(Qt.ItemDataRole.UserRole)
        return int(data) if isinstance(data, int) else None

    def _selected_profile(self) -> Profile | None:
        model = self._model
        if model is None:
            return None
        pid = self._current_id()
        row = model.row_for(pid) if pid is not None else model.active_row
        return row.profile if row is not None else None

    def _sync_buttons(self) -> None:
        model = self._model
        if model is None:
            return
        pid = self._current_id()
        row = model.row_for(pid) if pid is not None else None
        has_selection = row is not None
        self._duplicate_button.setEnabled(has_selection)
        self._rename_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection and model.can_delete())
        self._switch_button.setEnabled(bool(row is not None and not row.active))

    def _create_profile(self) -> None:
        pm = self._pm
        if pm is None:
            return
        name, ok = QInputDialog.getText(self, "Новый профиль", "Имя профиля:")
        if not ok or not name.strip():
            return
        try:
            pm.create(name.strip())
        except AyrisError as exc:
            self._profiles_status.show_message(exc.user_message, "error")
            return
        self._reload_profiles()
        self._profiles_status.show_message(f"Профиль «{name.strip()}» создан.", "success")

    def _duplicate_profile(self) -> None:
        pm = self._pm
        target = self._selected_profile()
        if pm is None or target is None:
            return
        try:
            copy = pm.copy(target)
        except AyrisError as exc:
            self._profiles_status.show_message(exc.user_message, "error")
            return
        self._reload_profiles()
        self._profiles_status.show_message(f"Создана копия «{copy.name}».", "success")

    def _rename_profile(self) -> None:
        pm = self._pm
        target = self._selected_profile()
        if pm is None or target is None:
            return
        name, ok = QInputDialog.getText(
            self, "Переименовать профиль", "Новое имя:", text=target.name
        )
        if not ok or not name.strip() or name.strip() == target.name:
            return
        try:
            pm.rename(target, name.strip())
        except AyrisError as exc:
            self._profiles_status.show_message(exc.user_message, "error")
            return
        self._reload_profiles()

    def _switch_selected(self) -> None:
        pm = self._pm
        target = self._selected_profile()
        if pm is None or target is None or target.is_active:
            return
        guard = self.services.unsaved_edits
        if guard is not None and guard():
            dialog = ConfirmDialog(
                "Переключить профиль?",
                "В редакторе команд есть несохранённые правки. При переключении "
                "профиля они будут потеряны.",
                self._theme,
                confirm_text="Переключить",
                dangerous=True,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
        try:
            pm.switch(target)
        except AyrisError as exc:
            self._profiles_status.show_message(exc.user_message, "error")
            return
        self._reload_profiles()
        self._profiles_status.show_message(f"Активен профиль «{target.name}».", "success")

    def _delete_profile(self) -> None:
        pm = self._pm
        model = self._model
        target = self._selected_profile()
        if pm is None or model is None or target is None:
            return
        if not model.can_delete():
            self._profiles_status.show_message("Нельзя удалить единственный профиль.", "error")
            return
        dialog = ConfirmDialog(
            "Удалить профиль?",
            f"Профиль «{target.name}» и все его команды, папки и переменные будут "
            "удалены без возможности отмены. Перед удалением будет создана "
            "резервная копия.",
            self._theme,
            confirm_text="Удалить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._backup_status_row = self._profiles_status
        self._backup_done_msg = f"Профиль «{target.name}» удалён."
        self._profiles_status.set_busy(True, "Создаём копию и удаляем…")
        self._backup_runner.run(lambda: self._backup_then_delete(target))

    def _backup_then_delete(self, target: Profile) -> Path:
        assert self._pm is not None
        backup = self._pm.backup(profile=target, reason="manual")
        self._pm.delete(target)
        return backup

    def _on_backup_done(self, _result: object) -> None:
        self._reload_profiles()
        self._reload_backups()
        self._backup_button.setEnabled(True)
        row = self._backup_status_row
        if row is not None:
            row.set_busy(False)
            row.show_message(self._backup_done_msg or "Готово.", "success")

    def _on_backup_failed(self, message: str) -> None:
        self._backup_button.setEnabled(True)
        row = self._backup_status_row
        if row is not None:
            row.set_busy(False)
            row.show_message(message or "Операция не удалась.", "error")

    # -- export / import ----------------------------------------------------

    def _build_transfer(self) -> None:
        self._add_header("Экспорт и импорт")
        caption = QLabel(
            "Профиль можно сохранить в архив .zip и перенести на другой компьютер. "
            "Секреты — API-ключи и токены — в архив не попадают."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

        row = QHBoxLayout()
        self._export_button = QPushButton("Экспортировать профиль…")
        self._export_button.clicked.connect(self._export_profile)
        self._import_button = QPushButton("Импортировать из архива…")
        self._import_button.setProperty("kind", "primary")
        self._import_button.clicked.connect(self._import_profile)
        row.addWidget(self._export_button)
        row.addWidget(self._import_button)
        row.addStretch(1)
        self._content.addLayout(row)
        self._transfer_status = _BusyRow(self._theme)
        self._content.addWidget(self._transfer_status)

    def _export_profile(self) -> None:
        pm = self._pm
        target = self._selected_profile()
        if pm is None or target is None:
            return
        manager = pm
        dialog = ProfileExportDialog(
            target.name,
            self._theme,
            estimate=lambda sounds: self._estimate(target, sounds),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        destination = dialog.destination
        if destination is None:
            return
        include_sounds = dialog.include_sounds
        include_settings = dialog.include_settings
        self._transfer_status.set_busy(True, "Экспортируем…")
        self._export_button.setEnabled(False)
        self._export_runner.run(
            lambda: manager.export(
                destination,
                profile=target,
                include_sounds=include_sounds,
                include_settings=include_settings,
            )
        )

    def _on_export_done(self, _result: object) -> None:
        self._export_button.setEnabled(True)
        self._transfer_status.set_busy(False)
        self._transfer_status.show_message("Профиль экспортирован в архив.", "success")

    def _on_export_failed(self, message: str) -> None:
        self._export_button.setEnabled(True)
        self._transfer_status.set_busy(False)
        self._transfer_status.show_message(message or "Не удалось экспортировать профиль.", "error")

    def _estimate(self, profile: Profile, include_sounds: bool) -> int:
        model = self._model
        pid = profile.id
        row = model.row_for(pid) if model is not None and pid is not None else None
        commands = row.commands if row is not None else 0
        variables = row.variables if row is not None else 0
        total = 8192 + commands * 512 + variables * 128
        if include_sounds and self._pm is not None:
            total += _dir_size(self._pm.paths.sounds_dir)
        return total

    def _missing_models(self, preview: BundlePreview) -> tuple[str, ...]:
        installed = self.services.installed_models
        if installed is None:
            return ()
        present = installed()
        return tuple(name for name in preview.models if name not in present)

    def _import_profile(self) -> None:
        pm = self._pm
        if pm is None:
            return
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите архив профиля",
            str(Path.home()),
            f"Архив профиля (*{BUNDLE_SUFFIX})",
        )
        if not chosen:
            return
        archive = Path(chosen)
        try:
            preview = pm.preview_import(archive)
        except AyrisError as exc:
            self._transfer_status.show_message(exc.user_message, "error")
            return
        except Exception:
            _log.exception("не удалось прочитать архив профиля")
            self._transfer_status.show_message("Не удалось прочитать архив.", "error")
            return
        dialog = ProfileImportDialog(
            preview,
            self._theme,
            active_profile_name=pm.active.name,
            missing_models=self._missing_models(preview),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._start_import(archive, preview, dialog)

    def _start_import(
        self, archive: Path, preview: BundlePreview, dialog: ProfileImportDialog
    ) -> None:
        pm = self._pm
        if pm is None:
            return
        manager = pm
        target_new = dialog.target_new
        new_name = dialog.new_profile_name or preview.manifest.profile_name
        policy = dialog.policy
        apply_config = dialog.apply_config

        def work() -> ImportReport:
            if target_new:
                profile = manager.create(new_name)
                return manager.import_bundle(
                    archive,
                    profile=profile,
                    policy=policy,
                    backup=False,
                    apply_config=apply_config,
                )
            return manager.import_bundle(
                archive, policy=policy, backup=True, apply_config=apply_config
            )

        self._transfer_status.set_busy(True, "Импортируем…")
        self._import_button.setEnabled(False)
        self._import_runner.run(work)

    def _on_import_done(self, result: object) -> None:
        self._import_button.setEnabled(True)
        self._transfer_status.set_busy(False)
        self._reload_profiles()
        self._reload_backups()
        if not isinstance(result, ImportReport):
            return
        self._transfer_status.show_message("Импорт завершён.", "success")
        report_dialog = _ReportDialog(
            result,
            self._theme,
            open_model_manager=self.services.open_model_manager,
            parent=self,
        )
        report_dialog.exec()

    def _on_import_failed(self, message: str) -> None:
        self._import_button.setEnabled(True)
        self._transfer_status.set_busy(False)
        self._transfer_status.show_message(message or "Не удалось импортировать профиль.", "error")

    # -- backups ------------------------------------------------------------

    def _build_backups(self) -> None:
        self._add_header("Резервные копии")
        caption = QLabel(
            "Копии создаются автоматически перед импортом и сбросом, и вручную "
            f"кнопкой ниже. Хранятся последние {MAX_BACKUPS}; папку можно открыть "
            "из раздела «Папка данных»."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

        row = QHBoxLayout()
        self._backup_button = QPushButton("Создать резервную копию")
        self._backup_button.clicked.connect(self._create_backup)
        row.addWidget(self._backup_button)
        row.addStretch(1)
        self._content.addLayout(row)

        self._backups_status = _BusyRow(self._theme)
        self._content.addWidget(self._backups_status)
        self._backups = BackupList(self._theme)
        self._backups.restore_requested.connect(self._restore_backup)
        self._backups.delete_requested.connect(self._delete_backup)
        self._content.addWidget(self._backups)

    def _reload_backups(self) -> None:
        pm = self._pm
        if pm is None:
            return
        try:
            self._backups.set_backups(pm.list_backups())
        except OSError:
            _log.exception("не удалось прочитать список резервных копий")

    def _create_backup(self) -> None:
        pm = self._pm
        if pm is None:
            return
        manager = pm
        self._backup_status_row = self._backups_status
        self._backup_done_msg = "Резервная копия создана."
        self._backups_status.set_busy(True, "Создаём копию…")
        self._backup_button.setEnabled(False)
        self._backup_runner.run(lambda: manager.backup(reason="manual"))

    def _restore_backup(self, path: Path) -> None:
        pm = self._pm
        if pm is None:
            return
        manager = pm
        active_name = manager.active.name
        dialog = ConfirmDialog(
            "Восстановить копию?",
            f"Содержимое активного профиля «{active_name}» будет заменено данными "
            "из копии. Текущее состояние необратимо, поэтому перед восстановлением "
            "будет создана ещё одна резервная копия.",
            self._theme,
            confirm_text="Восстановить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._backup_status_row = self._backups_status
        self._backup_done_msg = "Копия восстановлена."
        self._backups_status.set_busy(True, "Восстанавливаем…")
        self._backup_button.setEnabled(False)
        self._backup_runner.run(lambda: self._restore_from(manager, path))

    def _restore_from(self, manager: ProfileManager, path: Path) -> ImportReport:
        manager.reset(backup=True)
        return manager.import_bundle(
            path, policy=ConflictPolicy.OVERWRITE, backup=False, apply_config=False
        )

    def _delete_backup(self, path: Path) -> None:
        dialog = ConfirmDialog(
            "Удалить копию?",
            f"Файл резервной копии будет удалён без возможности отмены:\n{path.name}",
            self._theme,
            confirm_text="Удалить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            path.unlink()
        except OSError as exc:
            self._backups_status.show_message(f"Не удалось удалить копию: {exc}", "error")
            return
        self._reload_backups()
        self._backups_status.show_message("Копия удалена.", "success")

    # -- data folder --------------------------------------------------------

    def _build_data_folder(self) -> None:
        self._add_header("Папка данных")
        caption = QLabel(
            "Здесь хранятся база профилей, модели, настройки и резервные копии. "
            "Папку можно перенести на другой диск: данные скопируются, а новая "
            "папка вступит в силу после перезапуска."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

        self._cloud_notice = InlineNotice(
            "Не храните папку данных в облачных клиентах (Syncthing, Dropbox, "
            "OneDrive): они синхронизируют файлы во время записи и могут повредить "
            "базу. Для переноса на другой компьютер пользуйтесь экспортом профиля.",
            self._theme,
            kind="warning",
        )
        self._cloud_notice.close_button.hide()
        self._content.addWidget(self._cloud_notice)

        self._folder_path = QLabel("—")
        self._folder_path.setProperty("role", "h2")
        self._folder_path.setWordWrap(True)
        self._folder_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._content.addWidget(self._folder_path)
        self._folder_detail = QLabel("")
        self._folder_detail.setProperty("role", "secondary")
        self._folder_detail.setWordWrap(True)
        self._content.addWidget(self._folder_detail)

        row = QHBoxLayout()
        self._open_folder_button = QPushButton("Открыть папку профиля")
        self._open_folder_button.clicked.connect(self._open_data_folder)
        self._change_folder_button = QPushButton("Изменить папку…")
        self._change_folder_button.setProperty("kind", "primary")
        self._change_folder_button.clicked.connect(self._change_data_folder)
        row.addWidget(self._open_folder_button)
        row.addWidget(self._change_folder_button)
        row.addStretch(1)
        self._content.addLayout(row)
        self._folder_status = _BusyRow(self._theme)
        self._content.addWidget(self._folder_status)

    def _refresh_folder_card(self) -> None:
        if self._pm is None:
            return
        paths = self._pm.paths
        self._folder_path.setText(f"Данные и модели хранятся в:\n{paths.root}")
        detail = f"Расположение: {paths.source_label}."
        try:
            free = shutil.disk_usage(paths.root).free
        except OSError:
            free = None
        if free is not None:
            detail += f" Свободно на диске: {human_size(free)}."
        self._folder_detail.setText(detail)

    def _open_data_folder(self) -> None:
        pm = self._pm
        if pm is None:
            return
        try:
            pm.open_folder()
        except AyrisError as exc:
            self._folder_status.show_message(exc.user_message, "error")

    def _change_data_folder(self) -> None:
        pm = self._pm
        if pm is None:
            return
        manager = pm
        current = manager.paths.root
        chosen = QFileDialog.getExistingDirectory(
            self, "Выберите папку для данных Ayris", str(current.parent)
        )
        if not chosen:
            return
        target = (Path(chosen).expanduser() / APP_DIR_NAME).resolve()
        if target == current:
            self._folder_status.show_message("Это уже текущая папка данных.", "info")
            return
        if native_path(target) is None:
            self._folder_status.show_message(
                native_path_problem(target, what="модели и данные"), "error"
            )
            return
        self._confirm_and_move(manager, target)

    def _confirm_and_move(self, manager: ProfileManager, target: Path) -> None:
        dialog = ConfirmDialog(
            "Перенести папку данных?",
            f"Данные Ayris будут скопированы в:\n{target}\n\n"
            "Не выбирайте папку облачного клиента (Syncthing, Dropbox, OneDrive) — "
            "синхронизация во время записи может повредить базу. После копирования "
            "нужно перезапустить Ayris; старая папка останется резервной.",
            self._theme,
            confirm_text="Скопировать",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._pending_root = target
        self._folder_status.set_busy(True, "Копируем данные…")
        self._change_folder_button.setEnabled(False)
        self._folder_runner.run(lambda: manager.stage_root_change(target))

    def _on_folder_moved(self, result: object) -> None:
        self._change_folder_button.setEnabled(True)
        self._folder_status.set_busy(False)
        target = result if isinstance(result, Path) else self._pending_root
        self._folder_status.show_message(
            f"Готово. Данные скопированы в:\n{target}\n"
            "Перезапустите Ayris, чтобы перейти на новую папку.",
            "success",
        )
        dialog = ConfirmDialog(
            "Перезапустить Ayris?",
            "Новая папка данных вступит в силу после перезапуска. До него Ayris "
            "продолжит работать со старой папкой.",
            self._theme,
            confirm_text="Перезапустить сейчас",
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._restart_app()

    def _on_folder_failed(self, message: str) -> None:
        self._change_folder_button.setEnabled(True)
        self._folder_status.set_busy(False)
        self._folder_status.show_message(message or "Не удалось перенести папку данных.", "error")

    def _restart_app(self) -> None:
        """Relaunch a packaged build, then quit; a dev run just quits to reopen."""
        import subprocess
        import sys

        try:
            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable, *sys.argv[1:]])
        except Exception:
            _log.exception("не удалось перезапустить приложение автоматически")
        app = QApplication.instance()
        if app is not None:
            app.quit()


register_tab("profiles", ProfilesTab)
