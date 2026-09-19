"""Tab «Обновления»: application updates on top, the model manager below.

The page is two stacked concerns of one settings section. The top is the
application updater — check mode, «Проверить сейчас», the current and available
version, the changelog from GitHub Releases, and the install action. The bottom
is the whole :class:`~ayris.gui.widgets.model_manager.ModelManager`, which does
the real work of task 50.

Two things are worth knowing:

*The updater talks to nothing by default.* Checking GitHub and installing a build
belong to the installer of task 72; here they are injected services on
:class:`UpdatesServices`. Without them «Проверить сейчас» reports that checking is
unavailable rather than reaching the network from a settings tab — the same shape
the «Голос» tab uses for a service it does not yet have. :func:`parse_github_release`
is pure and testable, so the changelog rendering is verified against a recorded
response with no network at all.

*The model backend is built lazily.* The settings window constructs every tab with
``(manager, theme, bus)`` only, so when no backend is injected the tab wires one
over the real registry from :func:`~ayris.core.database.get_database` and
:func:`~ayris.core.paths.get_paths`, degrading to an empty catalog if that is not
available yet. The download coordinator it owns lives with the backend, not the
widgets, so closing the window never cancels a running download.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris import __version__
from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.config import ConfigManager
from ayris.core.errors import AyrisError
from ayris.core.events import EventBus
from ayris.core.paths import APP_DIR_NAME, get_paths, native_path, native_path_problem
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.tabs.voice import AsyncRunner
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import SettingCard, ThemedComboBox, ToggleSwitch
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.model_manager import ModelManager, ModelManagerBackend
from ayris.models.downloader import DownloadCancelled, DownloadHandle, human_size
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.models import ModelRecord
    from ayris.core.paths import ModelKind
    from ayris.core.profile import ProfileManager
    from ayris.models.catalog import ModelCatalog
    from ayris.models.registry import DiskUsage, IntegrityReport, ModelRegistry

__all__ = [
    "DownloadCoordinator",
    "RegistryBackend",
    "ReleaseInfo",
    "UpdatesServices",
    "UpdatesTab",
    "parse_github_release",
]

_log = get_logger(__name__)

#: The three check modes shown to the user, mapped onto the real config below.
_CHECK_MODES: tuple[tuple[str, str], ...] = (
    ("startup", "При запуске"),
    ("daily", "Раз в день"),
    ("manual", "Только вручную"),
)


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """A GitHub release, reduced to what the updater section shows."""

    version: str
    name: str = ""
    notes: str = ""
    url: str = ""
    published_at: str = ""
    prerelease: bool = False
    download_url: str = ""

    @property
    def title(self) -> str:
        return self.name or f"Версия {self.version}"


def parse_github_release(payload: dict[str, Any]) -> ReleaseInfo:
    """Turn one GitHub Releases API object into a :class:`ReleaseInfo`.

    Pure and total: every field is optional and defaults to empty, so a response
    with a missing ``body`` or no assets still yields a usable record instead of
    raising. The version is the tag with a leading ``v`` stripped, because that is
    how the catalog and :data:`ayris.__version__` spell it.
    """
    tag = str(payload.get("tag_name", "")).strip()
    version = tag[1:] if tag[:1] in {"v", "V"} else tag
    assets = payload.get("assets")
    download_url = ""
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "")).lower()
            url = asset.get("browser_download_url")
            if isinstance(url, str) and name.endswith((".exe", ".msi", ".zip")):
                download_url = url
                break
    return ReleaseInfo(
        version=version,
        name=str(payload.get("name", "")).strip(),
        notes=str(payload.get("body", "")).strip(),
        url=str(payload.get("html_url", "")).strip(),
        published_at=str(payload.get("published_at", "")).strip(),
        prerelease=bool(payload.get("prerelease", False)),
        download_url=download_url,
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    """A comparable tuple from a dotted version, non-numeric parts dropped."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def is_newer(available: str, current: str) -> bool:
    """Whether ``available`` is a strictly newer version than ``current``."""
    left, right = _version_tuple(available), _version_tuple(current)
    if not left:
        return False
    return left > right


#: Checks for a newer build; ``None`` when up to date. Raises on failure. Wired by
#: the installer of task 72; unset here so the tab reaches no network on its own.
UpdateChecker = Callable[[], ReleaseInfo | None]
#: Downloads and installs a release. Task 72; unset disables the install button.
UpdateInstaller = Callable[[ReleaseInfo], None]


@dataclass(slots=True)
class UpdatesServices:
    """Everything the tab needs from the rest of Ayris, all optional."""

    backend: ModelManagerBackend | None = None
    check_updates: UpdateChecker | None = None
    install_update: UpdateInstaller | None = None
    #: Moves the whole profile — models, database, settings — to another folder.
    #: Built lazily over the live database when omitted; ``None`` only if that
    #: fails, which greys out «Изменить папку…» rather than crashing the tab.
    profile_manager: ProfileManager | None = None


# ----------------------------------------------------------------------
# model backend over the registry
# ----------------------------------------------------------------------


class DownloadCoordinator:
    """Owns the download threads and cancellation tokens, keyed by model id.

    Lives with the backend, not with any widget, so a download the user started
    keeps running after the settings window closes — the requirement that
    «закрытие окна настроек не должно отменять загрузку». Threads are daemons and
    the registry publishes progress and completion on the bus; this only keeps the
    handles so «Отмена» has something to cancel and a rebuilt page can see what is
    still in flight.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._handles: dict[str, DownloadHandle] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, model_id: str) -> None:
        with self._lock:
            existing = self._threads.get(model_id)
            if existing is not None and existing.is_alive():
                return
            handle = DownloadHandle()
            self._handles[model_id] = handle
            thread = threading.Thread(
                target=self._run,
                args=(model_id, handle),
                name=f"ayris-download-{model_id}",
                daemon=True,
            )
            self._threads[model_id] = thread
            thread.start()

    def _run(self, model_id: str, handle: DownloadHandle) -> None:
        try:
            self._registry.install(model_id, handle=handle)
        except DownloadCancelled:
            _log.info("загрузка %s отменена", model_id)
        except AyrisError as exc:
            # The registry already published ModelDownloadFailed for the UI; this
            # is the log side of the same event.
            _log.warning("загрузка %s не удалась: %s", model_id, exc.technical)
        except Exception:
            _log.exception("загрузка %s упала", model_id)
        finally:
            with self._lock:
                self._handles.pop(model_id, None)
                self._threads.pop(model_id, None)

    def cancel(self, model_id: str) -> None:
        with self._lock:
            handle = self._handles.get(model_id)
        if handle is not None:
            handle.cancel()

    def is_active(self, model_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(model_id)
        return thread is not None and thread.is_alive()


class RegistryBackend:
    """A :class:`ModelManagerBackend` over the real registry and coordinator."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._coordinator = DownloadCoordinator(registry)

    def catalog(self) -> ModelCatalog:
        return self._registry.catalog

    def installed(self) -> list[ModelRecord]:
        return self._registry.installed()

    def active(self, kind: ModelKind) -> ModelRecord | None:
        return self._registry.active(kind)

    def disk_usage(self) -> DiskUsage:
        return self._registry.disk_usage()

    def free_disk_bytes(self) -> int:
        return self._registry.free_disk_bytes()

    def set_active(self, record: ModelRecord) -> None:
        self._registry.set_active(record)

    def remove(self, record: ModelRecord, *, force: bool) -> int:
        return self._registry.remove(record, force=force)

    def verify(self, record: ModelRecord, *, full: bool) -> IntegrityReport:
        return self._registry.verify(record, full=full)

    def start_download(self, model_id: str) -> None:
        # Fail fast on space before a thread is spawned, so «недостаточно места»
        # reaches the card immediately rather than after a failed transfer.
        self._registry.check_space(model_id)
        self._coordinator.start(model_id)

    def cancel_download(self, model_id: str) -> None:
        self._coordinator.cancel(model_id)

    def is_downloading(self, model_id: str) -> bool:
        return self._coordinator.is_active(model_id)


class _EmptyBackend:
    """Stand-in when no registry is available yet: an empty, inert catalog."""

    def catalog(self) -> ModelCatalog:
        from ayris.models.catalog import ModelCatalog

        return ModelCatalog(entries=())

    def installed(self) -> list[ModelRecord]:
        return []

    def active(self, _kind: ModelKind) -> ModelRecord | None:
        return None

    def disk_usage(self) -> DiskUsage:
        from ayris.models.registry import DiskUsage

        return DiskUsage()

    def free_disk_bytes(self) -> int:
        return 0

    def set_active(self, _record: ModelRecord) -> None:
        return None

    def remove(self, _record: ModelRecord, *, force: bool) -> int:  # noqa: ARG002
        return 0

    def verify(self, record: ModelRecord, *, full: bool) -> IntegrityReport:  # noqa: ARG002
        from ayris.models.registry import IntegrityReport, IntegrityStatus

        return IntegrityReport(record=record, status=IntegrityStatus.UNVERIFIED)

    def start_download(self, _model_id: str) -> None:
        return None

    def cancel_download(self, _model_id: str) -> None:
        return None

    def is_downloading(self, _model_id: str) -> bool:
        return False


def _default_backend(bus: EventBus | None) -> ModelManagerBackend:
    """Build a registry-backed backend, or an empty one if that is not possible."""
    try:
        from ayris.core.database import get_database
        from ayris.core.repositories import Repositories
        from ayris.models.registry import ModelRegistry

        repositories = Repositories(get_database())
        registry = ModelRegistry(repositories.models, get_paths(), bus=bus)
        return RegistryBackend(registry)
    except Exception:
        _log.exception("не удалось построить бэкенд менеджера моделей, показываю пустой каталог")
        return _EmptyBackend()


def _default_profile_manager(bus: EventBus | None, config: ConfigManager) -> ProfileManager | None:
    """Build a profile manager over the live database for the «Папка данных» card.

    Shares the process-wide database and paths, so :meth:`stage_root_change`
    checkpoints the same connection the application is using. Returns ``None`` if
    that is not possible — the card then reports the move is unavailable rather
    than taking the tab down.
    """
    try:
        from ayris.core.database import get_database
        from ayris.core.profile import ProfileManager
        from ayris.core.repositories import Repositories

        repositories = Repositories(get_database())
        return ProfileManager(repositories, paths=get_paths(), bus=bus, config=config)
    except Exception:
        _log.exception("не удалось построить менеджер профилей для смены папки данных")
        return None


# ----------------------------------------------------------------------
# the tab
# ----------------------------------------------------------------------


class UpdatesTab(SettingsTab):
    """The «Обновления» settings page (task 50)."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        services: UpdatesServices | None = None,
    ) -> None:
        super().__init__("updates", "Обновления", ("updates",), manager, theme, bus)
        self.services = services if services is not None else UpdatesServices()
        if self.services.backend is None:
            self.services.backend = _default_backend(bus)
        # The profile manager is built on first use, not here: constructing one
        # touches the live database, and the tab must build cheaply for a page
        # the user may never scroll to.
        self._profile_manager_bus = bus
        self._runner = AsyncRunner()
        self._runner.finished.connect(self._on_check_finished)
        self._runner.failed.connect(self._on_check_failed)
        self._latest: ReleaseInfo | None = None
        self._pending_root: Path | None = None
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

        self._build_app_updates()
        self._build_data_folder()
        self._build_model_manager(bus)
        self._content.addStretch(1)

    # -- application updates ------------------------------------------------

    def _add_header(self, text: str) -> None:
        header = QLabel(text)
        header.setProperty("role", "h2")
        self._content.addWidget(header)

    def _build_app_updates(self) -> None:
        self._add_header("Обновления приложения")

        self._version_card = _VersionPanel(self._theme)
        self._version_card.check_requested.connect(self._check_now)
        self._version_card.install_requested.connect(self._install_now)
        self._content.addWidget(self._version_card)

        self._mode_combo = ThemedComboBox()
        for value, label in _CHECK_MODES:
            self._mode_combo.addItem(label, value)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._content.addWidget(
            SettingCard(
                "Проверка обновлений",
                "Когда искать новую версию. Ручную проверку можно запустить в любой момент.",
                self._mode_combo,
                self._theme,
            )
        )

        self._channel_combo = ThemedComboBox()
        self._channel_combo.addItem("Стабильный", "stable")
        self._channel_combo.addItem("Бета", "beta")
        self.bind_combo(self._channel_combo, "updates.channel", "Канал обновлений")
        self._content.addWidget(
            SettingCard(
                "Канал",
                "Стабильные сборки или ранние бета-версии с новыми возможностями.",
                self._channel_combo,
                self._theme,
            )
        )

        self._auto_download = ToggleSwitch(self._theme, label="Скачивать автоматически")
        self.bind_toggle(self._auto_download, "updates.auto_download", "Скачивать автоматически")
        self._content.addWidget(
            SettingCard(
                "Скачивать обновления автоматически",
                "Найденное обновление скачивается в фоне; установка — по вашему подтверждению.",
                self._auto_download,
                self._theme,
            )
        )

        self._auto_install = ToggleSwitch(self._theme, label="Ставить при следующем запуске")
        self.bind_toggle(
            self._auto_install, "updates.auto_install", "Ставить при следующем запуске"
        )
        self._content.addWidget(
            SettingCard(
                "Устанавливать при следующем запуске",
                "Скачанное обновление ставится без вопроса при следующем старте Ayris.",
                self._auto_install,
                self._theme,
            )
        )

        self._check_models = ToggleSwitch(self._theme, label="Обновлять модели")
        self.bind_toggle(self._check_models, "updates.check_models", "Проверять обновления моделей")
        self._content.addWidget(
            SettingCard(
                "Проверять обновления моделей",
                "Вместе с программой проверять, не вышли ли новые версии моделей из каталога.",
                self._check_models,
                self._theme,
            )
        )

    def _build_model_manager(self, bus: EventBus | None) -> None:
        self._add_header("Модели")
        caption = QLabel(
            "Каталог моделей распознавания, синтеза, активации и языковой модели. "
            "Скачивание идёт в фоне и продолжается, даже если закрыть это окно."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

        assert self.services.backend is not None
        self._manager_widget = ModelManager(self.services.backend, self._theme, bus)
        self._content.addWidget(self._manager_widget)

    # -- data folder --------------------------------------------------------

    def _build_data_folder(self) -> None:
        self._add_header("Папка данных")
        caption = QLabel(
            "Здесь хранятся модели, база и настройки. По умолчанию это системный "
            "диск. Папку можно перенести на другой диск — данные скопируются, а "
            "новая папка вступит в силу после перезапуска Ayris."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        self._content.addWidget(caption)

        self._folder_card = _DataFolderCard(self._theme)
        self._folder_card.change_requested.connect(self._change_data_folder)
        self._folder_card.open_requested.connect(self._open_data_folder)
        self._content.addWidget(self._folder_card)
        self._refresh_folder_card()

    def _profile_manager(self) -> ProfileManager | None:
        """The injected manager, or one built lazily over the live database."""
        if self.services.profile_manager is None:
            self.services.profile_manager = _default_profile_manager(
                self._profile_manager_bus, self._manager
            )
        return self.services.profile_manager

    def _refresh_folder_card(self) -> None:
        paths = get_paths()
        try:
            free_text = f"свободно на диске {human_size(shutil.disk_usage(paths.root).free)}"
        except OSError:
            free_text = ""
        self._folder_card.set_location(str(paths.root), paths.source_label, free_text)

    def _open_data_folder(self) -> None:
        manager = self._profile_manager()
        if manager is None:
            self._folder_card.show_message("Открыть папку сейчас не удалось.", "error")
            return
        try:
            manager.open_folder()
        except AyrisError as exc:
            self._folder_card.show_message(exc.user_message, "error")

    def _change_data_folder(self) -> None:
        manager = self._profile_manager()
        if manager is None:
            self._folder_card.show_message("Смена папки сейчас недоступна.", "error")
            return
        current = get_paths().root
        chosen = QFileDialog.getExistingDirectory(
            self, "Выберите папку для данных Ayris", str(current.parent)
        )
        if not chosen:
            return
        # A fresh «Ayris» subfolder, so an existing, non-empty pick is never
        # refused and the layout matches the default %APPDATA%\Ayris shape.
        target = (Path(chosen).expanduser() / APP_DIR_NAME).resolve()
        if target == current:
            self._folder_card.show_message("Это уже текущая папка данных.", "info")
            return
        # Native engines (Vosk, espeak) cannot open non-Latin paths, so a folder
        # they could not read is refused before anything is copied.
        if native_path(target) is None:
            self._folder_card.show_message(
                native_path_problem(target, what="модели и данные"), "error"
            )
            return
        dialog = ConfirmDialog(
            "Перенести папку данных?",
            f"Данные Ayris (модели, база, настройки) будут скопированы в:\n{target}\n\n"
            "После копирования нужно перезапустить Ayris, чтобы перейти на новую "
            "папку. Старая папка останется как резервная копия.",
            self._theme,
            confirm_text="Скопировать",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._pending_root = target
        self._folder_card.clear_message()
        self._folder_card.set_busy(True)
        self._folder_runner.run(lambda: manager.stage_root_change(target))

    def _on_folder_moved(self, result: object) -> None:
        self._folder_card.set_busy(False)
        target = result if isinstance(result, Path) else self._pending_root
        self._folder_card.show_message(
            f"Готово. Данные скопированы в:\n{target}\n"
            "Перезапустите Ayris, чтобы начать пользоваться новой папкой.",
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
        self._folder_card.set_busy(False)
        self._folder_card.show_message(message or "Не удалось перенести папку данных.", "error")

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

    # -- check mode round-trip ----------------------------------------------

    def _mode_from_config(self) -> str:
        updates = self._manager.settings.updates
        if not updates.check_on_start:
            return "manual"
        return "daily" if updates.interval_hours <= 24 else "startup"

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentData()
        if mode == "manual":
            self._manager.apply({"updates.check_on_start": False})
        elif mode == "daily":
            self._manager.apply({"updates.check_on_start": True, "updates.interval_hours": 24})
        else:
            self._manager.apply({"updates.check_on_start": True, "updates.interval_hours": 168})

    def _sync_mode_combo(self) -> None:
        blocker = QSignalBlocker(self._mode_combo)
        index = self._mode_combo.findData(self._mode_from_config())
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)
        del blocker

    # -- checking -----------------------------------------------------------

    def _check_now(self) -> None:
        check = self.services.check_updates
        if check is None:
            self._version_card.show_message(
                "Проверка обновлений появится вместе с автообновлением приложения.",
                "info",
            )
            return
        self._version_card.set_checking(True)
        self._runner.run(check)

    def _on_check_finished(self, result: object) -> None:
        self._version_card.set_checking(False)
        if result is None or not isinstance(result, ReleaseInfo):
            self._latest = None
            self._version_card.show_up_to_date()
            return
        if not is_newer(result.version, __version__):
            self._latest = None
            self._version_card.show_up_to_date()
            return
        self._latest = result
        can_install = self.services.install_update is not None
        self._version_card.show_release(result, can_install=can_install)

    def _on_check_failed(self, message: str) -> None:
        self._version_card.set_checking(False)
        self._version_card.show_message(
            message or "Не удалось проверить обновления. Проверьте подключение.",
            "error",
        )

    def _install_now(self) -> None:
        install = self.services.install_update
        if install is None or self._latest is None:
            return
        try:
            install(self._latest)
        except AyrisError as exc:
            self._version_card.show_message(exc.user_message, "error")
        except Exception as exc:
            _log.exception("не удалось установить обновление")
            self._version_card.show_message(str(exc) or "Ошибка установки.", "error")
        else:
            self._version_card.show_message("Обновление устанавливается…", "success")

    # -- lifecycle ----------------------------------------------------------

    def load_from_config(self) -> None:
        super().load_from_config()
        self._sync_mode_combo()
        self._version_card.set_current_version(__version__)

    def _on_config_changed(self, diff: SettingsDiff) -> None:
        super()._on_config_changed(diff)
        paths = set(diff.paths)
        if {"updates.check_on_start", "updates.interval_hours"} & paths:
            self._sync_mode_combo()

    def dispose(self) -> None:
        self._manager_widget.dispose()
        super().dispose()


class _VersionPanel(QFrame):
    """The version card: current/available version, changelog and actions."""

    check_requested = Signal()
    install_requested = Signal()

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("card", True)
        self.setAccessibleName("Обновления приложения")
        self._layout = QVBoxLayout(self)

        self._current = QLabel("Текущая версия: —")
        self._current.setProperty("role", "h2")
        self._layout.addWidget(self._current)

        self._available = QLabel("")
        self._available.setProperty("badge", "info")
        self._available.setWordWrap(True)
        self._available.hide()
        self._layout.addWidget(self._available)

        self._message = QLabel("")
        self._message.setProperty("role", "secondary")
        self._message.setWordWrap(True)
        self._message.hide()
        self._layout.addWidget(self._message)

        self._changelog = QLabel("")
        self._changelog.setProperty("role", "secondary")
        self._changelog.setWordWrap(True)
        self._changelog.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._changelog.hide()
        self._layout.addWidget(self._changelog)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self._check_button = QPushButton("Проверить сейчас")
        self._check_button.clicked.connect(self.check_requested)
        self._install_button = QPushButton("Скачать и установить")
        self._install_button.setProperty("kind", "primary")
        self._install_button.clicked.connect(self.install_requested)
        self._install_button.hide()
        actions.addWidget(self._check_button)
        actions.addWidget(self._install_button)
        self._layout.addLayout(actions)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def set_current_version(self, version: str) -> None:
        self._current.setText(f"Текущая версия: {version}")

    def set_checking(self, checking: bool) -> None:
        self._check_button.setEnabled(not checking)
        self._check_button.setText("Проверяем…" if checking else "Проверить сейчас")

    def show_up_to_date(self) -> None:
        self._available.hide()
        self._changelog.hide()
        self._install_button.hide()
        self.show_message("Установлена последняя версия.", "success")

    def show_release(self, release: ReleaseInfo, *, can_install: bool) -> None:
        self._message.hide()
        label = f"Доступна {release.title}"
        if release.published_at:
            label += f" — {_human_date(release.published_at)}"
        self._available.setText(label)
        self._available.setProperty("badge", "info")
        _restyle(self._available)
        self._available.show()
        self._changelog.setText(release.notes or "Список изменений не опубликован.")
        self._changelog.show()
        self._install_button.setVisible(can_install)
        self._install_button.setEnabled(can_install)

    def show_message(self, text: str, kind: str) -> None:
        self._message.setText(text)
        self._message.setProperty("badge", kind)
        _restyle(self._message)
        self._message.show()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_lg")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_sm"))


class _DataFolderCard(QFrame):
    """Shows where the profile lives and offers to move it to another folder."""

    change_requested = Signal()
    open_requested = Signal()

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("card", True)
        self.setAccessibleName("Папка данных")
        self._layout = QVBoxLayout(self)

        self._path = QLabel("—")
        self._path.setProperty("role", "h2")
        self._path.setWordWrap(True)
        self._path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._layout.addWidget(self._path)

        self._detail = QLabel("")
        self._detail.setProperty("role", "secondary")
        self._detail.setWordWrap(True)
        self._layout.addWidget(self._detail)

        self._message = QLabel("")
        self._message.setProperty("role", "secondary")
        self._message.setWordWrap(True)
        self._message.hide()
        self._layout.addWidget(self._message)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self._open_button = QPushButton("Открыть папку")
        self._open_button.clicked.connect(self.open_requested)
        self._change_button = QPushButton("Изменить папку…")
        self._change_button.setProperty("kind", "primary")
        self._change_button.clicked.connect(self.change_requested)
        actions.addWidget(self._open_button)
        actions.addWidget(self._change_button)
        self._layout.addLayout(actions)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def set_location(self, path: str, source_label: str, free_text: str) -> None:
        self._path.setText(f"Данные и модели хранятся в:\n{path}")
        detail = f"Расположение: {source_label}."
        if free_text:
            detail += f" {free_text}."
        self._detail.setText(detail)

    def set_busy(self, busy: bool) -> None:
        self._change_button.setEnabled(not busy)
        self._change_button.setText("Копируем…" if busy else "Изменить папку…")
        self._open_button.setEnabled(not busy)

    def show_message(self, text: str, kind: str) -> None:
        self._message.setText(text)
        self._message.setProperty("badge", kind)
        _restyle(self._message)
        self._message.show()

    def clear_message(self) -> None:
        self._message.hide()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_lg")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_sm"))


def _human_date(iso: str) -> str:
    """``2026-09-01T10:00:00Z`` → ``01.09.2026``; the raw string if unparseable."""
    try:
        parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    return parsed.strftime("%d.%m.%Y")


def _restyle(widget: QWidget) -> None:
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


register_tab("updates", UpdatesTab)
