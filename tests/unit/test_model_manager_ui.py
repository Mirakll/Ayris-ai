"""Tab «Обновления» and the model manager (task 50), offscreen on a fake backend.

No test opens a socket or downloads a byte: a fake backend stands in for the
registry, and the update checker is called directly rather than through its
thread, so the assertions are deterministic. The checks are structural — cards
build from the catalog grouped by kind, a progress event moves one card, the
throttle caps repaints, deleting the active model asks first, a recorded GitHub
release parses into a changelog, and disk usage sums over the real Windows paths
(the last verified by CI on the Windows runner). Every widget is closed in
teardown: an un-closed page keeps a bus subscription and a download-progress
timer, and CI hangs on the leak.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from ayris.core.database import Database
from ayris.core.events import (
    EventBus,
    ModelDownloadFailed,
    ModelDownloadFinished,
    ModelDownloadProgress,
    ModelRemoved,
)
from ayris.core.models import ModelRecord
from ayris.core.paths import AppPaths, ModelKind
from ayris.core.repositories import Repositories
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.updates import (
    RegistryBackend,
    ReleaseInfo,
    UpdatesServices,
    UpdatesTab,
    is_newer,
    parse_github_release,
)
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import ModelManager
from ayris.gui.widgets import model_manager as manager_module
from ayris.gui.widgets.download_progress import DownloadProgress, human_eta
from ayris.gui.widgets.model_card import CardStatus
from ayris.models.catalog import ModelCatalog, ModelEntry
from ayris.models.registry import DiskUsage, IntegrityReport, IntegrityStatus, ModelRegistry

pytestmark = pytest.mark.unit

_SHA = "0" * 64


# ----------------------------------------------------------------------
# fixtures and fakes
# ----------------------------------------------------------------------


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


def _entry(**overrides: object) -> ModelEntry:
    # model_validate, not the constructor: ModelEntry's validator rebuilds the
    # instance to fill derived targets, which pydantic only allows off __init__.
    payload: dict[str, object] = {
        "url": "https://example.com/model.bin",
        "sha256": _SHA,
        "size_bytes": 1000,
        "id": "model",
        "name": "Модель",
        "kind": "stt",
        "engine": "gigaam",
    }
    payload.update(overrides)
    return ModelEntry.model_validate(payload)


def _catalog() -> ModelCatalog:
    return ModelCatalog(
        entries=(
            _entry(
                id="gigaam-v3-ctc",
                name="GigaAM v3 CTC",
                kind="stt",
                engine="gigaam",
                url="https://example.com/gigaam.zip",
                archive="zip",
                language="ru",
                description="Русское распознавание по умолчанию.",
                size_bytes=200,
                requires_ram_mb=2048,
            ),
            _entry(
                id="vosk-ru-small",
                name="Vosk RU small",
                kind="stt",
                engine="vosk",
                url="https://example.com/vosk.zip",
                archive="zip",
                language="ru",
                description="Потоковое распознавание.",
                size_bytes=100,
            ),
            _entry(
                id="ru-irina",
                name="Ирина",
                kind="tts",
                engine="piper",
                url="https://example.com/irina.onnx",
                language="ru",
                description="Женский голос.",
                size_bytes=300,
            ),
            _entry(
                id="qwen-1_5b",
                name="Qwen 1.5B",
                kind="llm",
                engine="llamacpp",
                url="https://example.com/qwen.gguf",
                language="multi",
                description="Маленькая языковая модель.",
                size_bytes=500,
            ),
        )
    )


def _record(model_id: str, kind: ModelKind, **overrides: object) -> ModelRecord:
    values: dict[str, object] = {
        "kind": kind,
        "name": model_id,
        "id": abs(hash(model_id)) % 100000,
        "engine": "gigaam",
        "catalog_id": model_id,
        "size_bytes": 12345,
        "sha256": _SHA,
    }
    values.update(overrides)
    return ModelRecord(**values)  # type: ignore[arg-type]


class FakeBackend:
    """A :class:`ModelManagerBackend` that records calls and touches no disk."""

    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        installed: tuple[ModelRecord, ...] = (),
        active: dict[str, ModelRecord] | None = None,
    ) -> None:
        self._catalog = catalog if catalog is not None else _catalog()
        self._installed = list(installed)
        self._active = dict(active or {})
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.removed: list[tuple[ModelRecord, bool]] = []
        self.activated: list[ModelRecord] = []
        self.verify_result: IntegrityReport | None = None
        self.space_error: Exception | None = None
        self._downloading: set[str] = set()
        self.free = 500 * 1024 * 1024

    def catalog(self) -> ModelCatalog:
        return self._catalog

    def installed(self) -> list[ModelRecord]:
        return list(self._installed)

    def active(self, kind: ModelKind) -> ModelRecord | None:
        return self._active.get(kind)

    def disk_usage(self) -> DiskUsage:
        by_kind = {"stt": 12345, "tts": 0, "wake": 0, "llm": 0}
        return DiskUsage(by_kind=by_kind, total=sum(by_kind.values()), cache_bytes=0)

    def free_disk_bytes(self) -> int:
        return self.free

    def set_active(self, record: ModelRecord) -> None:
        self.activated.append(record)
        self._active[record.kind] = record

    def remove(self, record: ModelRecord, *, force: bool) -> int:
        self.removed.append((record, force))
        self._installed = [r for r in self._installed if r.id != record.id]
        self._active = {k: v for k, v in self._active.items() if v.id != record.id}
        return 9999

    def verify(self, record: ModelRecord, *, full: bool) -> IntegrityReport:
        if self.verify_result is not None:
            return self.verify_result
        return IntegrityReport(record=record, status=IntegrityStatus.OK, full=full)

    def start_download(self, model_id: str) -> None:
        if self.space_error is not None:
            raise self.space_error
        self.started.append(model_id)
        self._downloading.add(model_id)

    def cancel_download(self, model_id: str) -> None:
        self.cancelled.append(model_id)
        self._downloading.discard(model_id)

    def is_downloading(self, model_id: str) -> bool:
        return model_id in self._downloading


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _manager(app: QApplication, backend: FakeBackend, bus: EventBus | None = None) -> ModelManager:
    return ModelManager(backend, ThemeManager(app), bus)


# ----------------------------------------------------------------------
# registration and layout
# ----------------------------------------------------------------------


def test_tab_registers_itself_as_the_updates_factory() -> None:
    assert tab_spec("updates").factory is UpdatesTab


def test_manager_builds_a_section_and_card_per_catalog_entry(app: QApplication) -> None:
    backend = FakeBackend()
    manager = _manager(app, backend)
    assert set(manager._cards) == {"gigaam-v3-ctc", "vosk-ru-small", "ru-irina", "qwen-1_5b"}
    # Three kinds are present in the catalog: stt, tts, llm — wake has no entry.
    assert set(manager._sections) == {"stt", "tts", "llm"}
    manager.dispose()
    manager.close()


def test_installed_model_shows_installed_status(app: QApplication) -> None:
    record = _record("ru-irina", "tts")
    backend = FakeBackend(installed=(record,), active={"tts": record})
    manager = _manager(app, backend)
    card = manager._cards["ru-irina"]
    assert card.view().status == CardStatus.INSTALLED
    assert card.view().is_active
    assert card.radio.isChecked()
    manager.dispose()
    manager.close()


def test_usage_line_shows_used_and_free(app: QApplication) -> None:
    backend = FakeBackend(installed=(_record("ru-irina", "tts"),))
    manager = _manager(app, backend)
    text = manager._usage.text()
    assert "свободно на диске" in text
    assert "всего" in text
    manager.dispose()
    manager.close()


# ----------------------------------------------------------------------
# downloads and progress
# ----------------------------------------------------------------------


def test_download_button_starts_the_backend(app: QApplication) -> None:
    backend = FakeBackend()
    manager = _manager(app, backend)
    manager._download("gigaam-v3-ctc")
    assert backend.started == ["gigaam-v3-ctc"]
    assert manager._cards["gigaam-v3-ctc"].view().status == CardStatus.QUEUED
    manager.dispose()
    manager.close()


def test_not_enough_space_is_shown_on_the_card(app: QApplication) -> None:
    from ayris.models.downloader import NotEnoughSpaceError

    backend = FakeBackend()
    backend.space_error = NotEnoughSpaceError("full", user_message="Недостаточно места.")
    manager = _manager(app, backend)
    manager._download("gigaam-v3-ctc")
    card = manager._cards["gigaam-v3-ctc"]
    assert card.view().status == CardStatus.ERROR
    assert "места" in card.view().detail
    manager.dispose()
    manager.close()


def test_progress_event_moves_one_card(app: QApplication) -> None:
    bus = EventBus()
    backend = FakeBackend()
    manager = _manager(app, backend, bus)
    bus.publish(ModelDownloadProgress(model_id="gigaam-v3-ctc", downloaded=100, total=200))
    app.processEvents()
    card = manager._cards["gigaam-v3-ctc"]
    assert card.view().status == CardStatus.DOWNLOADING
    # isHidden, not isVisible: the page is never shown offscreen, so a visible
    # child still reports isVisible()==False; isHidden reflects the explicit call.
    assert not card._progress.isHidden()
    manager.dispose()
    manager.close()


def test_finished_event_rebuilds_and_marks_installed(app: QApplication) -> None:
    bus = EventBus()
    backend = FakeBackend()
    manager = _manager(app, backend, bus)
    # The install landed: the backend now reports it installed, and the rebuild
    # on the finished event should pick that up.
    backend._installed.append(_record("gigaam-v3-ctc", "stt"))
    bus.publish(ModelDownloadFinished(model_id="gigaam-v3-ctc", kind="stt"))
    app.processEvents()
    assert manager._cards["gigaam-v3-ctc"].view().status == CardStatus.INSTALLED
    manager.dispose()
    manager.close()


def test_failed_event_shows_error_with_message(app: QApplication) -> None:
    bus = EventBus()
    backend = FakeBackend()
    manager = _manager(app, backend, bus)
    bus.publish(
        ModelDownloadFailed(model_id="vosk-ru-small", user_message="Сервер отклонил загрузку.")
    )
    app.processEvents()
    card = manager._cards["vosk-ru-small"]
    assert card.view().status == CardStatus.ERROR
    assert card.view().detail == "Сервер отклонил загрузку."
    manager.dispose()
    manager.close()


def test_cancel_calls_backend_and_resets_card(app: QApplication) -> None:
    backend = FakeBackend()
    manager = _manager(app, backend)
    manager._download("gigaam-v3-ctc")
    manager._cancel("gigaam-v3-ctc")
    assert backend.cancelled == ["gigaam-v3-ctc"]
    assert manager._cards["gigaam-v3-ctc"].view().status == CardStatus.AVAILABLE
    manager.dispose()
    manager.close()


def test_removed_event_does_not_cancel_downloads(app: QApplication) -> None:
    bus = EventBus()
    backend = FakeBackend()
    manager = _manager(app, backend, bus)
    manager._download("gigaam-v3-ctc")
    bus.publish(ModelRemoved(model_id="other", kind="stt"))
    app.processEvents()
    # A rebuild happened, but the running download was never cancelled.
    assert backend.cancelled == []
    manager.dispose()
    manager.close()


# ----------------------------------------------------------------------
# throttle
# ----------------------------------------------------------------------


def test_progress_widget_throttles_to_ten_per_second(app: QApplication) -> None:
    clock = FakeClock()
    widget = DownloadProgress(ThemeManager(app), clock=clock)
    for index in range(1000):
        widget.set_progress(index, 1000, 1000.0, 1.0)
        clock.advance(0.001)  # 1000 samples over one simulated second
    # At most ten applied updates a second, plus the very first sample.
    assert widget.updates_applied <= 12
    widget.flush()
    widget.close()


def test_progress_widget_indeterminate_when_total_unknown(app: QApplication) -> None:
    widget = DownloadProgress(ThemeManager(app))
    widget.set_progress(1024, 0, 0.0, 0.0)
    assert widget._bar.maximum() == 0  # indeterminate bar
    widget.reset()
    widget.close()


def test_human_eta_is_russian_and_empty_when_unknown() -> None:
    assert human_eta(0) == ""
    assert human_eta(30) == "осталось 30 с"
    assert human_eta(95) == "осталось 1 мин 35 с"
    assert "ч" in human_eta(3700)


# ----------------------------------------------------------------------
# activation, verification, deletion
# ----------------------------------------------------------------------


def test_activate_radio_calls_set_active(app: QApplication) -> None:
    record = _record("ru-irina", "tts")
    backend = FakeBackend(installed=(record,))
    manager = _manager(app, backend)
    manager._cards["ru-irina"].radio.setChecked(True)
    assert backend.activated and backend.activated[0].catalog_id == "ru-irina"
    manager.dispose()
    manager.close()


def test_verify_ok_reports_intact(app: QApplication) -> None:
    record = _record("ru-irina", "tts")
    backend = FakeBackend(installed=(record,))
    manager = _manager(app, backend)
    manager._verify("ru-irina")
    assert manager._cards["ru-irina"].view().status == CardStatus.INSTALLED
    manager.dispose()
    manager.close()


def test_verify_corrupted_offers_redownload(app: QApplication) -> None:
    record = _record("ru-irina", "tts")
    backend = FakeBackend(installed=(record,))
    backend.verify_result = IntegrityReport(
        record=record, status=IntegrityStatus.CORRUPTED, detail="sha256 mismatch", full=True
    )
    manager = _manager(app, backend)
    manager._verify("ru-irina")
    card = manager._cards["ru-irina"]
    assert card.view().status == CardStatus.CORRUPTED
    assert "заново" in card.view().detail
    manager.dispose()
    manager.close()


def test_delete_active_asks_confirmation_and_can_decline(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record("ru-irina", "tts")
    backend = FakeBackend(installed=(record,), active={"tts": record})
    manager = _manager(app, backend)

    asked: list[str] = []

    def _decline(self: object) -> QDialog.DialogCode:
        asked.append(getattr(self, "windowTitle", lambda: "")())
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(manager_module.ConfirmDialog, "exec", _decline)
    manager._delete("ru-irina")
    assert asked  # a confirmation dialog was shown
    assert backend.removed == []  # declined: nothing deleted
    manager.dispose()
    manager.close()


def test_delete_confirmed_removes_the_model(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record("ru-irina", "tts")
    backend = FakeBackend(installed=(record,))
    manager = _manager(app, backend)
    monkeypatch.setattr(
        manager_module.ConfirmDialog, "exec", lambda *_: QDialog.DialogCode.Accepted
    )
    manager._delete("ru-irina")
    assert backend.removed and backend.removed[0][0].catalog_id == "ru-irina"
    manager.dispose()
    manager.close()


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------


def test_search_filters_cards(app: QApplication) -> None:
    backend = FakeBackend()
    manager = _manager(app, backend)
    manager._search.setText("vosk")
    assert not manager._cards["vosk-ru-small"].isHidden()
    assert manager._cards["gigaam-v3-ctc"].isHidden()
    manager.dispose()
    manager.close()


def test_only_installed_filter_hides_available(app: QApplication) -> None:
    backend = FakeBackend(installed=(_record("ru-irina", "tts"),))
    manager = _manager(app, backend)
    manager._only_installed.setChecked(True)
    assert not manager._cards["ru-irina"].isHidden()
    assert manager._cards["gigaam-v3-ctc"].isHidden()
    manager.dispose()
    manager.close()


def test_empty_state_when_nothing_matches(app: QApplication) -> None:
    backend = FakeBackend()
    manager = _manager(app, backend)
    manager._search.setText("нет такой модели вообще")
    assert not manager._empty.isHidden()
    manager.dispose()
    manager.close()


# ----------------------------------------------------------------------
# application updates
# ----------------------------------------------------------------------


_RELEASE_PAYLOAD = {
    "tag_name": "v9.9.9",
    "name": "Ayris 9.9.9",
    "body": "## Что нового\n- Быстрее запуск\n- Меньше памяти",
    "html_url": "https://github.com/Mirakll/Ayris-ai/releases/tag/v9.9.9",
    "published_at": "2026-09-01T10:00:00Z",
    "prerelease": False,
    "assets": [
        {"name": "Ayris-Setup.exe", "browser_download_url": "https://example.com/Ayris-Setup.exe"}
    ],
}


def test_parse_github_release_reads_the_fields() -> None:
    info = parse_github_release(_RELEASE_PAYLOAD)
    assert info.version == "9.9.9"
    assert info.name == "Ayris 9.9.9"
    assert "Быстрее запуск" in info.notes
    assert info.download_url.endswith("Ayris-Setup.exe")
    assert not info.prerelease


def test_parse_github_release_tolerates_missing_fields() -> None:
    info = parse_github_release({"tag_name": "1.2.3"})
    assert info.version == "1.2.3"
    assert info.notes == ""
    assert info.download_url == ""


def test_is_newer_compares_versions() -> None:
    assert is_newer("1.2.0", "1.1.9")
    assert is_newer("2.0.0", "1.9.9")
    assert not is_newer("1.0.0", "1.0.0")
    assert not is_newer("0.9.0", "1.0.0")


def _tab(app: QApplication, **services: object) -> UpdatesTab:
    from ayris.core.config import ConfigManager

    manager = ConfigManager(Path("nonexistent") / "config.toml")
    tab = UpdatesTab(
        manager,
        ThemeManager(app),
        None,
        services=UpdatesServices(backend=FakeBackend(), **services),  # type: ignore[arg-type]
    )
    tab.load_from_config()
    return tab


def test_tab_shows_current_version(app: QApplication, tmp_path: Path) -> None:
    from ayris.core.config import ConfigManager

    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    tab = UpdatesTab(
        manager, ThemeManager(app), None, services=UpdatesServices(backend=FakeBackend())
    )
    tab.load_from_config()
    from ayris import __version__

    assert __version__ in tab._version_card._current.text()
    tab.dispose()
    tab.close()


def test_check_finds_newer_version_and_renders_changelog(app: QApplication, tmp_path: Path) -> None:
    from ayris.core.config import ConfigManager

    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    release = parse_github_release(_RELEASE_PAYLOAD)
    installed: list[ReleaseInfo] = []
    tab = UpdatesTab(
        manager,
        ThemeManager(app),
        None,
        services=UpdatesServices(
            backend=FakeBackend(),
            check_updates=lambda: release,
            install_update=installed.append,
        ),
    )
    tab.load_from_config()
    # Drive the result path directly instead of the worker thread.
    tab._on_check_finished(release)
    assert tab._latest is release
    assert "Быстрее запуск" in tab._version_card._changelog.text()
    assert not tab._version_card._install_button.isHidden()
    tab._install_now()
    assert installed == [release]
    tab.dispose()
    tab.close()


def test_check_up_to_date_hides_install(app: QApplication, tmp_path: Path) -> None:
    from ayris.core.config import ConfigManager

    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    old = ReleaseInfo(version="0.0.1", notes="старьё")
    tab = UpdatesTab(
        manager,
        ThemeManager(app),
        None,
        services=UpdatesServices(backend=FakeBackend(), check_updates=lambda: old),
    )
    tab.load_from_config()
    tab._on_check_finished(old)
    assert tab._latest is None
    assert tab._version_card._install_button.isHidden()
    tab.dispose()
    tab.close()


def test_check_mode_round_trips_through_config(app: QApplication, tmp_path: Path) -> None:
    from ayris.core.config import ConfigManager

    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    tab = UpdatesTab(
        manager, ThemeManager(app), None, services=UpdatesServices(backend=FakeBackend())
    )
    tab.load_from_config()

    tab._mode_combo.setCurrentIndex(tab._mode_combo.findData("manual"))
    assert not manager.settings.updates.check_on_start
    tab._mode_combo.setCurrentIndex(tab._mode_combo.findData("daily"))
    assert manager.settings.updates.check_on_start
    assert manager.settings.updates.interval_hours == 24
    tab.dispose()
    tab.close()


def test_tab_has_no_horizontal_scrollbar(app: QApplication, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea

    from ayris.core.config import ConfigManager

    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    tab = UpdatesTab(
        manager, ThemeManager(app), None, services=UpdatesServices(backend=FakeBackend())
    )
    tab.load_from_config()
    scroll = tab.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    tab.dispose()
    tab.close()


# ----------------------------------------------------------------------
# data folder (moving the profile to another disk)
# ----------------------------------------------------------------------


class _FakeProfileManager:
    """Stands in for :class:`ProfileManager`: records the move, touches no disk."""

    def __init__(self) -> None:
        self.staged: list[Path] = []
        self.opened = 0

    def stage_root_change(self, target: Path) -> Path:
        self.staged.append(Path(target))
        return Path(target)

    def open_folder(self, target: Path | None = None) -> Path:
        self.opened += 1
        return Path("x")


def test_data_folder_card_shows_the_current_root(app: QApplication) -> None:
    from ayris.core.paths import get_paths

    tab = _tab(app, profile_manager=_FakeProfileManager())
    assert str(get_paths().root) in tab._folder_card._path.text()
    tab.dispose()
    tab.close()


def test_change_data_folder_confirms_then_stages_and_offers_restart(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PySide6.QtWidgets import QFileDialog

    from ayris.gui.widgets.confirm_dialog import ConfirmDialog

    pm = _FakeProfileManager()
    tab = _tab(app, profile_manager=pm)
    chosen = tmp_path / "target"
    restarts: list[bool] = []
    monkeypatch.setattr(tab, "_restart_app", lambda: restarts.append(True))
    # The temp dir sits under a Cyrillic home here, so bypass the native-path
    # guard: this test is about the confirm → stage → restart flow, not the
    # guard, which has its own test below.
    monkeypatch.setattr("ayris.gui.tabs.updates.native_path", lambda path: str(path))
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *_a, **_k: str(chosen))
    )
    monkeypatch.setattr(ConfirmDialog, "exec", lambda _self: QDialog.DialogCode.Accepted)
    # Run the copy inline instead of on the worker thread.
    monkeypatch.setattr(tab._folder_runner, "run", lambda work: tab._on_folder_moved(work()))

    tab._change_data_folder()

    assert pm.staged == [(chosen / "Ayris").resolve()]
    assert "Готово" in tab._folder_card._message.text()
    assert restarts == [True]
    tab.dispose()
    tab.close()


def test_change_data_folder_refuses_a_non_latin_path(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PySide6.QtWidgets import QFileDialog

    pm = _FakeProfileManager()
    tab = _tab(app, profile_manager=pm)
    cyrillic = tmp_path / "кириллица"
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *_a, **_k: str(cyrillic))
    )

    tab._change_data_folder()

    assert pm.staged == []
    assert "латиниц" in tab._folder_card._message.text()
    tab.dispose()
    tab.close()


def test_change_data_folder_cancelled_confirm_stages_nothing(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PySide6.QtWidgets import QFileDialog

    from ayris.gui.widgets.confirm_dialog import ConfirmDialog

    pm = _FakeProfileManager()
    tab = _tab(app, profile_manager=pm)
    monkeypatch.setattr("ayris.gui.tabs.updates.native_path", lambda path: str(path))
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *_a, **_k: str(tmp_path / "cancel")),
    )
    monkeypatch.setattr(ConfirmDialog, "exec", lambda _self: QDialog.DialogCode.Rejected)

    tab._change_data_folder()

    assert pm.staged == []
    tab.dispose()
    tab.close()


def test_change_data_folder_reports_a_failed_move(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = _tab(app, profile_manager=_FakeProfileManager())
    tab._on_folder_failed("Недостаточно места в выбранной папке.")
    assert "места" in tab._folder_card._message.text()
    tab.dispose()
    tab.close()


# ----------------------------------------------------------------------
# disk usage over real Windows paths (CI on the Windows runner)
# ----------------------------------------------------------------------


def test_registry_backend_disk_usage_over_real_paths(app: QApplication, tmp_path: Path) -> None:
    paths = AppPaths(root=tmp_path)
    with Database.open(paths.database_file) as database:
        repositories = Repositories(database)
        registry = ModelRegistry(repositories.models, paths, catalog=_catalog())
        try:
            backend = RegistryBackend(registry)
            usage = backend.disk_usage()
            assert set(usage.by_kind) == {"stt", "tts", "wake", "llm"}
            # Free space on the profile volume is a real, positive number.
            assert backend.free_disk_bytes() > 0
            manager = ModelManager(backend, ThemeManager(app), None)
            assert "свободно на диске" in manager._usage.text()
            manager.dispose()
            manager.close()
        finally:
            registry.close()
