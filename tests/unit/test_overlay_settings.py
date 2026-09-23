"""Вкладка «Панель / Сфера» (задача 56): конфиг, миграция и предпросмотр.

Проверяем три обещания задачи. Конфиг: правки контролов доезжают до
:class:`OverlayConfig` и грузятся обратно. Миграция: старый ``config.toml`` с
полями плавающего оверлея (позиция, монитор, прозрачность, масштаб,
click-through) грузится без ошибки — мёртвые ключи молча отброшены, живые
сохранены. Предпросмотр: встроенная сфера задачи 45 переключает все пять
состояний и отражает облик сразу, ещё до отложенного сохранения.

WebGL-сфера вешает CI, поэтому ``make_sphere`` в модуле предпросмотра подменяем
на лёгкую QPainter-сферу, а каждый виджет закрываем в ``finally``: незакрытая
страница настроек держит подписку на конфиг и подвешивает прогон.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from ayris.core.config import ConfigManager, OverlayConfig
from ayris.gui.dashboard.dialog_view import DialogView
from ayris.gui.overlay.dialog_log import DialogKind
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.overlay_settings import OVERLAY_PRESETS, OverlayTab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import sphere_preview as sphere_preview_module
from ayris.gui.widgets.sphere.sphere_widget import SphereWidget as PainterSphere
from ayris.gui.widgets.sphere.states import SphereState
from ayris.gui.widgets.sphere_preview import SpherePreview

pytestmark = pytest.mark.unit


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


@pytest.fixture(autouse=True)
def _painter_sphere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the QPainter sphere: the WebGL one needs WebEngine and hangs CI.

    ``SpherePreview`` looks up ``make_sphere`` in its own module namespace, so
    that is where the patch has to land.
    """

    def factory(theme_: ThemeManager, parent: QWidget | None = None) -> QWidget:
        return PainterSphere(theme_, parent=parent)

    monkeypatch.setattr(sphere_preview_module, "make_sphere", factory)


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    result = ConfigManager(tmp_path / "config.toml")
    result.load()
    return result


def _tab(manager: ConfigManager, theme: ThemeManager) -> OverlayTab:
    tab = OverlayTab(manager, theme)
    tab.load_from_config()
    return tab


# --- регистрация и загрузка ------------------------------------------------


def test_tab_registers_itself_as_the_overlay_factory() -> None:
    spec = tab_spec("overlay")
    assert spec.factory is OverlayTab
    assert spec.config_paths == ("overlay",)


def test_defaults_load_into_controls(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        assert tab._fps_combo.currentData() == 60
        assert tab._bindings["overlay.show_log"].widget.isChecked() is True
        assert tab._bindings["overlay.sphere_points"].widget.value() == 600
        assert tab._bindings["overlay.rotation_speed"].widget.value() == 100
        assert tab.preview.sphere.point_count == 600
    finally:
        tab.dispose()
        tab.close()


# --- конфиг: контролы ↔ OverlayConfig --------------------------------------


def test_edits_round_trip_to_config(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        tab._bindings["overlay.sphere_points"].widget.setValue(1200)
        tab._bindings["overlay.rotation_speed"].widget.setValue(150)  # → 1.5
        tab._bindings["overlay.show_log"].widget.setChecked(False)
        tab._bindings["overlay.waves"].widget.setChecked(False)
        tab._fps_combo.setCurrentIndex(tab._fps_combo.findData(30))
        tab.flush_pending()

        overlay = manager.settings.overlay
        assert overlay.sphere_points == 1200
        assert overlay.rotation_speed == pytest.approx(1.5)
        assert overlay.show_log is False
        assert overlay.waves is False
        assert overlay.target_fps == 30
    finally:
        tab.dispose()
        tab.close()


def test_accent_field_round_trips_and_clears(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        tab._accent.set_value("#12ab34", emit=True)
        tab.flush_pending()
        assert manager.settings.overlay.sphere_accent == "#12ab34"

        tab._accent.set_value("", emit=True)
        tab.flush_pending()
        assert manager.settings.overlay.sphere_accent == ""
    finally:
        tab.dispose()
        tab.close()


def test_external_change_updates_controls_and_preview(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        manager.apply({"overlay.sphere_points": 800})
        app.processEvents()
        assert tab._bindings["overlay.sphere_points"].widget.value() == 800
        assert tab.preview.sphere.point_count == 800
    finally:
        tab.dispose()
        tab.close()


# --- пресеты ----------------------------------------------------------------


def test_presets_apply_expected_bundles(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        tab._apply_preset("economy")
        overlay = manager.settings.overlay
        for key, value in OVERLAY_PRESETS["economy"].items():
            assert getattr(overlay, key) == value
        app.processEvents()
        assert tab._fps_combo.currentData() == 30
        assert "overlay.target_fps" not in tab._pending
    finally:
        tab.dispose()
        tab.close()


def test_unknown_preset_is_a_no_op(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        before = manager.settings.overlay
        tab._apply_preset("nope")
        assert manager.settings.overlay is before
    finally:
        tab.dispose()
        tab.close()


# --- миграция старого config.toml ------------------------------------------

_OLD_TOML = """\
[overlay]
position = "top_right"
custom_x = 120
custom_y = 240
monitor = 1
opacity = 0.75
scale = 1.25
click_through = true
hide_when_idle = true
idle_hide_sec = 45
sphere_points = 900
show_log = false
sphere_accent = "#123abc"
"""


def test_old_config_migrates_by_dropping_dead_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_OLD_TOML, encoding="utf-8")
    manager = ConfigManager(path)
    manager.load()

    overlay = manager.settings.overlay
    assert overlay.sphere_points == 900
    assert overlay.show_log is False
    assert overlay.sphere_accent == "#123abc"

    dead = (
        "position",
        "custom_x",
        "custom_y",
        "monitor",
        "opacity",
        "scale",
        "click_through",
        "hide_when_idle",
        "idle_hide_sec",
    )
    for name in dead:
        assert name not in type(overlay).model_fields
        assert not hasattr(overlay, name)
    assert manager.dropped_fields == ()


def test_out_of_range_value_is_dropped_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[overlay]\nsphere_points = 99999\n", encoding="utf-8")
    manager = ConfigManager(path)
    manager.load()

    assert manager.settings.overlay.sphere_points == 600
    assert "overlay.sphere_points" in manager.dropped_fields


# --- предпросмотр сферы -----------------------------------------------------


def test_preview_cycles_through_all_five_states(app: QApplication, theme: ThemeManager) -> None:
    preview = SpherePreview(theme)
    try:
        states = [
            SphereState.IDLE,
            SphereState.LISTENING,
            SphereState.THINKING,
            SphereState.SPEAKING,
            SphereState.ERROR,
        ]
        for index, state in enumerate(states):
            preview.show_state(state)
            # Кнопка выбранного состояния подсвечивается для всех пяти — это и есть
            # контракт сегментного переключателя предпросмотра.
            button = preview._buttons.button(index)
            assert button is not None
            assert button.isChecked()
            # «Ошибка» у сферы задачи 45 — вспышка: set_state лишь взводит её, а
            # _state доезжает до ERROR только когда тикает часами видимый виджет.
            # Остальные четыре переключаются синхронно — их и сверяем.
            if state is not SphereState.ERROR:
                assert preview.state is state
    finally:
        preview.close()


def test_preview_applies_overlay_appearance_to_the_sphere(
    app: QApplication, theme: ThemeManager
) -> None:
    preview = SpherePreview(theme)
    try:
        preview.apply_overlay(OverlayConfig(sphere_points=250, sphere_accent="#00ff00"))
        assert preview.sphere.point_count == 250
        accent = preview.sphere._accent
        assert accent is not None
        assert accent.name() == "#00ff00"

        preview.apply_overlay(OverlayConfig(sphere_points=250, sphere_accent=""))
        assert preview.sphere._accent is None
    finally:
        preview.close()


def test_preview_reflects_pending_edits_before_save(
    app: QApplication, manager: ConfigManager, theme: ThemeManager
) -> None:
    tab = _tab(manager, theme)
    try:
        tab._bindings["overlay.sphere_points"].widget.setValue(300)
        # Правка ещё не сохранена...
        assert tab.is_dirty is True
        assert "overlay.sphere_points" in tab._pending
        assert manager.settings.overlay.sphere_points == 600
        # ...но предпросмотр уже показывает её.
        assert tab.preview.sphere.point_count == 300
    finally:
        tab.dispose()
        tab.close()


# --- видимость блоков управления в колонке диалога --------------------------


def test_dialog_view_hides_control_blocks_per_overlay(
    app: QApplication, theme: ThemeManager
) -> None:
    view = DialogView(theme)
    try:
        view.apply_overlay(
            OverlayConfig(
                show_log=False,
                show_timers=False,
                show_profile=False,
                show_settings_button=False,
                show_mic=False,
                show_text_input=False,
            )
        )
        assert view.timers.isVisibleTo(view) is False
        assert view._profile_box.isVisibleTo(view) is False
        assert view.menu_button.isVisibleTo(view) is False
        assert view.input_bar.isVisibleTo(view) is False

        view.add_message(DialogKind.ANSWER, "готово")
        assert view._center.currentWidget() is view._empty_page
    finally:
        view.close()


def test_dialog_view_keeps_input_bar_when_only_mic_visible(
    app: QApplication, theme: ThemeManager
) -> None:
    view = DialogView(theme)
    try:
        view.apply_overlay(OverlayConfig(show_mic=True, show_text_input=False))
        assert view.input_bar.isVisibleTo(view) is True
        assert view.input_bar.mic_button.isVisibleTo(view.input_bar) is True
        assert view.input_bar.command.isVisibleTo(view.input_bar) is False
    finally:
        view.close()


def test_dialog_view_applies_log_line_cap(app: QApplication, theme: ThemeManager) -> None:
    view = DialogView(theme)
    try:
        view.apply_overlay(OverlayConfig(show_log=True, log_lines=3))
        for index in range(6):
            view.add_message(DialogKind.ANSWER, f"строка {index}")
        assert len(view.log.entries) == 3
        assert view._center.currentWidget() is view.log
    finally:
        view.close()
