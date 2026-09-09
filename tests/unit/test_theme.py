"""Theme schema, QSS substitution and live Qt application."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from ayris.gui.theme import ThemeLoadError, ThemeManager, bundled_theme_path, load_theme, render_qss
from ayris.gui.widgets import (
    BusyIndicator,
    ConfirmDialog,
    EmptyState,
    IconButton,
    InlineNotice,
    SearchField,
    SettingCard,
    SliderField,
    Toast,
    ToggleSwitch,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> QApplication:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


def test_bundled_themes_share_the_complete_schema() -> None:
    dark = load_theme(bundled_theme_path("dark_purple"))
    light = load_theme(bundled_theme_path("light"))
    assert dark.color("accent") == "#9B6DFF"
    assert dark.colors.model_fields_set == light.colors.model_fields_set
    assert dark.metrics.model_fields_set == light.metrics.model_fields_set
    assert dark.typography.model_fields_set == light.typography.model_fields_set


def test_missing_token_has_a_readable_path(tmp_path: Path) -> None:
    data = json.loads(bundled_theme_path("light").read_text(encoding="utf-8"))
    del data["colors"]["accent"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ThemeLoadError, match=r"colors\.accent.*отсутствует обязательный ключ"):
        load_theme(path)


def test_qss_substitutes_tokens(app: QApplication) -> None:
    theme = load_theme(bundled_theme_path("dark_purple"))
    qss = render_qss(
        theme, "QWidget { color: {{color.accent}}; padding: {{metric.spacing_sm}}px; }"
    )
    assert "#9B6DFF" in qss
    assert "padding: 8px" in qss
    assert "{{" not in qss


def test_theme_manager_replaces_stylesheet_and_emits(app: QApplication, qtbot: object) -> None:
    manager = ThemeManager(app)
    received: list[str] = []
    manager.theme_changed.connect(lambda theme: received.append(theme.mode))
    manager.set_mode("dark")
    old = app.styleSheet()
    manager.set_mode("light")
    new = app.styleSheet()
    assert manager.theme.mode == "light"
    assert received[-1] == "light"
    assert old != new
    assert "#7044D6" in new
    assert "#9B6DFF" not in new


def test_slider_and_toggle_state(app: QApplication) -> None:
    manager = ThemeManager(app)
    slider = SliderField(manager, value=20, unit="%")
    toggle = ToggleSwitch(manager)
    try:
        slider.spin_box.setValue(42)
        toggle.setChecked(True)
        assert slider.value() == 42
        assert toggle.isChecked()
    finally:
        slider.close()
        toggle.close()


def test_all_widgets_construct_and_close(app: QApplication) -> None:
    manager = ThemeManager(app)
    toggle = ToggleSwitch(manager)
    widgets: list[QWidget] = [
        toggle,
        SettingCard("Заголовок", "Описание", toggle, manager),
        SliderField(manager, unit="%"),
        SearchField(theme=manager),
        IconButton(
            app.style().standardIcon(app.style().StandardPixmap.SP_BrowserReload),
            "Обновить",
            manager,
        ),
        InlineNotice("Сообщение", manager),
        Toast("Готово", "Сообщение", manager, auto_hide_ms=0),
        EmptyState("Пусто", "Здесь пока ничего нет", manager),
        BusyIndicator(manager),
        ConfirmDialog("Подтверждение", "Продолжить?", manager),
    ]
    for widget in widgets:
        widget.close()
    app.processEvents()


def test_widgets_contain_no_hardcoded_hex_colours() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "ayris" / "gui" / "widgets"
    offenders = {
        path.name: line
        for path in root.glob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if any(
            len(word) in (7, 9) and word.startswith("#") for word in line.replace(";", " ").split()
        )
    }
    assert not offenders
