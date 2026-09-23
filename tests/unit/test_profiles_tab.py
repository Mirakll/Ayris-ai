"""Task 57: the «Профили» tab — its list model, the two dialogs and the report.

The heavy lifting (export, import, backups, moving the root) lives in
:class:`~ayris.core.profile.ProfileManager` and is proven by ``test_profile``; the
checks here are the tab's own surface, the part a manager cannot cover: the pure
list model over a real manager, the Russian pluralisation and backup-name parsing,
and that the export/import dialogs collect the right decisions and speak plainly
about what leaves the machine. Everything is offscreen; every widget is closed and
the tab disposed in teardown, or a lingering config subscription hangs CI. No test
calls :meth:`AsyncRunner.run`, so no background thread is left running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from ayris.core import paths as paths_module
from ayris.core.config import ConfigManager
from ayris.core.database import Database, reset_database
from ayris.core.events import EventBus
from ayris.core.models import Command, CommandFolder, VariableScope
from ayris.core.portable_profile import ConflictPolicy, ImportReport
from ayris.core.profile import ProfileManager
from ayris.core.repositories import Repositories
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.profiles import (
    ProfileListModel,
    ProfileRow,
    ProfilesServices,
    ProfilesTab,
    _plural,
    _ReportDialog,
)
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.backup_list import parse_backup_name
from ayris.gui.widgets.notice import InlineNotice
from ayris.gui.widgets.profile_export_dialog import ProfileExportDialog
from ayris.gui.widgets.profile_import_dialog import ProfileImportDialog

pytestmark = pytest.mark.unit


@pytest.fixture
def paths(tmp_path: Path) -> paths_module.AppPaths:
    return paths_module.init_paths(profile=tmp_path / "root")


@pytest.fixture
def database(paths: paths_module.AppPaths) -> Iterator[Database]:
    handle = Database.open(paths.database_file)
    yield handle
    handle.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    return Repositories(database)


@pytest.fixture
def bus() -> EventBus:
    return EventBus(thread_id=None)


@pytest.fixture
def config(paths: paths_module.AppPaths) -> ConfigManager:
    manager = ConfigManager(paths.config_file)
    manager.load()
    return manager


@pytest.fixture
def pm(
    repos: Repositories,
    paths: paths_module.AppPaths,
    bus: EventBus,
    config: ConfigManager,
) -> ProfileManager:
    return ProfileManager(repos, paths=paths, bus=bus, config=config)


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def theme(app: QApplication) -> ThemeManager:
    return ThemeManager(app)


def seed(repos: Repositories, profile_id: int) -> None:
    """Two folders, two commands and two profile-scoped variables, all in Russian."""
    outer = repos.folders.create(CommandFolder(name="Свет", profile_id=profile_id, sort_order=1))
    assert outer.id is not None
    repos.folders.create(
        CommandFolder(name="Кухня", profile_id=profile_id, parent_id=outer.id, sort_order=2)
    )
    repos.commands.create(Command(name="Включи свет", profile_id=profile_id, folder_id=outer.id))
    repos.commands.create(Command(name="Заметка", profile_id=profile_id))
    repos.variables.set(
        "город", "Москва", scope=VariableScope.PROFILE, profile_id=profile_id, persistent=True
    )
    repos.variables.set(
        "черновик", "temp", scope=VariableScope.PROFILE, profile_id=profile_id, persistent=False
    )


def _make_tab(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    pm: ProfileManager,
    services: ProfilesServices | None = None,
) -> ProfilesTab:
    if services is None:
        services = ProfilesServices(profile_manager=pm)
    return ProfilesTab(config, theme, bus, services=services)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "1 команда"),
        (2, "2 команды"),
        (4, "4 команды"),
        (5, "5 команд"),
        (11, "11 команд"),
        (14, "14 команд"),
        (21, "21 команда"),
        (0, "0 команд"),
    ],
)
def test_plural_picks_the_right_russian_form(count: int, expected: str) -> None:
    assert _plural(count, "команда", "команды", "команд") == expected


def test_parse_backup_name_reads_a_well_formed_stamp(tmp_path: Path) -> None:
    archive = tmp_path / "Основной_manual_20260115_083000.zip"
    archive.write_bytes(b"PK")
    info = parse_backup_name(archive)
    assert info.reason == "manual"
    assert info.reason_label == "вручную"
    assert info.created == datetime(2026, 1, 15, 8, 30, 0)
    assert info.size == 2


def test_parse_backup_name_survives_a_foreign_file(tmp_path: Path) -> None:
    archive = tmp_path / "случайный-файл.zip"
    archive.write_bytes(b"")
    info = parse_backup_name(archive)
    assert info.reason == ""
    assert info.created is None
    assert info.reason_label == "неизвестно"


def test_tab_registers_itself_as_the_profiles_factory() -> None:
    assert tab_spec("profiles").factory is ProfilesTab


def test_profiles_page_is_wired_through_the_package_import() -> None:
    """A fresh interpreter must reach the real page through ``ayris.gui.tabs`` alone.

    The app never imports :mod:`ayris.gui.tabs.profiles` directly — ``main_window``
    imports the package, whose ``__init__`` pulls each page module so its
    ``register_tab`` runs at startup. The assertion above passes in-process merely
    because this test file imported the submodule; only a clean subprocess proves
    the wire that makes the section show its page instead of the placeholder.
    """
    src = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(src), env.get("PYTHONPATH", "")) if p)
    code = (
        "from ayris.gui.tabs import tab_spec\n"
        "spec = tab_spec('profiles')\n"
        "assert spec.factory is not None, 'profiles still shows the placeholder'\n"
        "print(spec.factory.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ProfilesTab"


def test_list_model_reads_the_active_profile_with_its_counts(
    pm: ProfileManager, repos: Repositories
) -> None:
    active_id = pm.active.id
    assert active_id is not None
    seed(repos, active_id)

    model = ProfileListModel(pm)
    row = model.active_row
    assert isinstance(row, ProfileRow)
    assert row.active is True
    assert (row.commands, row.folders, row.variables) == (2, 2, 2)
    assert model.count() == 1
    assert model.can_delete() is False
    assert row.display.startswith("● ")
    assert "2 команды" in row.summary


def test_list_model_tracks_several_profiles(pm: ProfileManager) -> None:
    created = pm.create("Второй")
    model = ProfileListModel(pm)
    assert model.count() == 2
    assert model.can_delete() is True
    assert sum(1 for r in model.rows if r.active) == 1
    assert created.id is not None
    assert model.row_for(created.id) is not None


def test_tab_assembles_and_protects_the_last_profile(
    pm: ProfileManager,
    repos: Repositories,
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
) -> None:
    active_id = pm.active.id
    assert active_id is not None
    seed(repos, active_id)

    tab = _make_tab(config, theme, bus, pm)
    assert tab._model is not None
    assert tab._list.count() == 1
    # One profile: delete is blocked and the active row cannot be «made active».
    assert tab._delete_button.isEnabled() is False
    assert tab._switch_button.isEnabled() is False
    tab.dispose()
    tab.close()


def test_tab_reloads_when_a_profile_appears_on_the_bus(
    pm: ProfileManager,
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
) -> None:
    tab = _make_tab(config, theme, bus, pm)
    before = tab._list.count()
    pm.create("Второй")  # publishes ProfilesChanged on the shared synchronous bus
    assert tab._list.count() == before + 1
    assert tab._delete_button.isEnabled() is True
    tab.dispose()
    tab.close()


def test_tab_data_folder_card_warns_about_cloud_clients(
    pm: ProfileManager,
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
) -> None:
    tab = _make_tab(config, theme, bus, pm)
    assert str(pm.paths.root) in tab._folder_path.text()
    notices = [n.label.text() for n in tab.findChildren(InlineNotice)]
    cloud = next(t for t in notices if "Syncthing" in t)
    assert "Dropbox" in cloud and "OneDrive" in cloud
    tab.dispose()
    tab.close()


def test_export_dialog_collects_composition_and_states_no_secrets(
    theme: ThemeManager, tmp_path: Path
) -> None:
    dialog = ProfileExportDialog("Игры", theme, estimate=lambda _sounds: 4096)
    assert dialog.include_settings is True
    assert dialog.include_sounds is True
    # The command/folder box is always on and cannot be turned off.
    assert dialog._commands.isChecked() is True
    assert dialog._commands.isEnabled() is False
    dialog._settings.setChecked(False)
    dialog._sounds.setChecked(False)
    assert dialog.include_settings is False
    assert dialog.include_sounds is False
    # No destination yet, so the export button stays disabled until one is chosen.
    assert dialog.destination is None
    assert dialog._export_button.isEnabled() is False
    dialog.set_destination(tmp_path / "игры.zip")
    assert dialog.destination == tmp_path / "игры.zip"
    assert dialog._export_button.isEnabled() is True
    notices = [n.label.text() for n in dialog.findChildren(InlineNotice)]
    assert any("ключи" in t for t in notices)
    dialog.close()


def test_import_dialog_reads_a_real_bundle_and_flags_conflicts(
    pm: ProfileManager,
    repos: Repositories,
    theme: ThemeManager,
    tmp_path: Path,
) -> None:
    active_id = pm.active.id
    assert active_id is not None
    seed(repos, active_id)
    archive = tmp_path / "профиль.zip"
    pm.export(archive)
    preview = pm.preview_import(archive)  # against the same profile → name clashes

    dialog = ProfileImportDialog(
        preview,
        theme,
        active_profile_name=pm.active.name,
        missing_models=("vosk-ru-0.42",),
    )
    assert dialog.target_new is True
    assert dialog.new_profile_name == preview.manifest.profile_name
    assert dialog.policy is ConflictPolicy.RENAME
    items = [
        it.text()
        for i in range(dialog._contents.count())
        if (it := dialog._contents.item(i)) is not None
    ]
    assert any("⚠ уже есть" in text for text in items)
    notices = [n.label.text() for n in dialog.findChildren(InlineNotice)]
    assert any("vosk-ru-0.42" in t for t in notices)

    # The clash policy is meaningless for a new profile and lights up on merge.
    assert dialog._policy_combo.isEnabled() is False
    dialog._merge_radio.setChecked(True)
    assert dialog._policy_combo.isEnabled() is True
    dialog.close()


def test_import_dialog_disables_config_when_the_archive_has_none(
    repos: Repositories,
    paths: paths_module.AppPaths,
    bus: EventBus,
    theme: ThemeManager,
    tmp_path: Path,
) -> None:
    no_config = ProfileManager(repos, paths=paths, bus=bus)  # exports without settings
    archive = tmp_path / "без-настроек.zip"
    no_config.export(archive)
    preview = no_config.preview_import(archive)
    assert preview.has_config is False

    dialog = ProfileImportDialog(preview, theme, active_profile_name=no_config.active.name)
    assert dialog.apply_config is False
    assert dialog._apply_config.isEnabled() is False
    dialog.close()


def test_report_dialog_lists_missing_models_and_opens_the_manager(theme: ThemeManager) -> None:
    report = ImportReport(
        profile_name="Второй",
        added_commands=("Свет",),
        missing_models=("vosk-ru-0.42",),
    )
    opened: list[bool] = []
    dialog = _ReportDialog(report, theme, open_model_manager=lambda: opened.append(True))
    notices = [n.label.text() for n in dialog.findChildren(InlineNotice)]
    assert any("vosk-ru-0.42" in t for t in notices)
    button = next(
        b for b in dialog.findChildren(QPushButton) if b.text() == "Открыть менеджер моделей"
    )
    button.click()
    assert opened == [True]
    dialog.close()
