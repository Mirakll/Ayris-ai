"""Settings window navigation, binding, search and persisted state."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QLineEdit

from ayris.core.config import ConfigManager, WindowConfig
from ayris.gui.main_window import MainWindow, restored_geometry
from ayris.gui.tabs import SECTIONS, SettingsTab
from ayris.gui.theme import ThemeManager

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> QApplication:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        if isinstance(widget, MainWindow):
            widget.exit()
        else:
            widget.close()
    application.processEvents()


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    result = ConfigManager(tmp_path / "config.toml")
    result.load()
    return result


def test_registry_and_window_pages_are_lazy(app: QApplication, manager: ConfigManager) -> None:
    assert len(SECTIONS) == 11
    window = MainWindow(theme=ThemeManager(app), manager=manager)
    assert window.created_sections == ("general",)
    first = window.open_section("voice")
    assert window.created_sections == ("general", "voice")
    assert window.open_section("voice") is first
    window.exit()


def test_binding_debounces_and_external_change_has_no_echo(
    app: QApplication, manager: ConfigManager
) -> None:
    theme = ThemeManager(app)
    tab = SettingsTab("general", "Общие", ("general",), manager, theme)
    field = QLineEdit()
    tab.body.addWidget(field)
    tab.bind_line_edit(field, "general.theme", "Тема")
    tab.load_from_config()
    assert field.text() == "dark_purple"

    field.setText("light")
    assert tab.is_dirty
    tab.flush_pending()
    assert manager.settings.general.theme == "light"
    assert 'theme = "light"' in manager.path.read_text(encoding="utf-8")
    assert not tab.is_dirty

    manager.apply({"general.theme": "system"})
    app.processEvents()
    assert field.text() == "system"
    assert not tab.is_dirty
    tab.dispose()
    tab.close()


def test_reset_uses_pydantic_defaults(app: QApplication, manager: ConfigManager) -> None:
    manager.apply({"general.theme": "light"})
    tab = SettingsTab("general", "Общие", ("general",), manager, ThemeManager(app))
    tab.reset_to_defaults()
    assert manager.settings.general.theme == "dark_purple"
    tab.dispose()
    tab.close()


def test_search_index_contains_registered_field(app: QApplication, manager: ConfigManager) -> None:
    tab = SettingsTab("general", "Общие", ("general",), manager, ThemeManager(app))
    field = QLineEdit()
    tab.bind_line_edit(field, "general.language", "Язык интерфейса")
    entries = tab.search_entries
    assert len(entries) == 1
    assert entries[0].search_text == "general.language язык интерфейса общие"
    tab.dispose()
    tab.close()


def test_geometry_is_restored_or_recentred() -> None:
    primary = QRect(0, 0, 1920, 1080)
    visible = WindowConfig(x=200, y=100, width=900, height=620)
    assert restored_geometry(visible, (primary,), primary).topLeft() == QPoint(200, 100)

    missing = WindowConfig(x=4000, y=100, width=900, height=620)
    result = restored_geometry(missing, (primary,), primary)
    assert result.center() == primary.center()


def test_window_state_and_close_to_tray(app: QApplication, manager: ConfigManager) -> None:
    window = MainWindow(theme=ThemeManager(app), manager=manager)
    window.show()
    window.open_section("privacy")
    window.setGeometry(120, 140, 1000, 700)
    window.close()
    assert not window.isVisible()
    assert manager.settings.window.section == "privacy"
    assert manager.settings.window.width == 1000
    window.exit()

    restored = MainWindow(theme=ThemeManager(app), manager=manager)
    assert restored.current_section == "privacy"
    assert restored.geometry().size().width() == 1000
    restored.exit()
