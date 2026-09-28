"""GUI-сценарии единого окна через pytest-qt в offscreen-режиме.

Точечные тесты на вкладку `test_main_window.py` уже проверяют сферу, лог и прямое
открытие слоя настроек. Здесь — стыки, которых там нет: переключение раздела
через сигнал бокового меню (а не прямым `open_section`), запись настройки через
страницу, собранную фабрикой реестра, «погружение» при открытии редактора команды
и реакция окна на событие `OpenCommandRequested` с шины. То есть проводка между
подсистемами GUI, а не внутренности одного виджета.

Весь набор идёт в offscreen (`QT_QPA_PLATFORM=offscreen` — умолчание набора) и по
проверенному образцу: локальная фикстура `app` + `app.processEvents()`, а не
`qtbot`. WebGL-сфера повесила бы headless-прогон, поэтому `make_sphere` в модуле
`showcase` заменена на лёгкую сферу на QPainter для каждого теста. Окна и вкладки
закрываются в `finally`, иначе незакрытый виджет всплывёт как `ResourceWarning`,
а `filterwarnings=["error"]` превратит его в падение.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from ayris.core.config import ConfigManager
from ayris.core.database import init_database, reset_database
from ayris.core.events import EventBus, OpenCommandRequested
from ayris.core.models import Command
from ayris.core.repositories import Repositories
from ayris.gui.dashboard import showcase as showcase_module
from ayris.gui.main_window import MainWindow
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.sphere_widget import SphereWidget as PainterSphere

pytestmark = pytest.mark.gui


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
def _painter_sphere(monkeypatch: pytest.MonkeyPatch, theme: ThemeManager) -> None:
    def factory(theme_: ThemeManager, parent: QWidget | None = None) -> QWidget:
        return PainterSphere(theme_, parent=parent)

    monkeypatch.setattr(showcase_module, "make_sphere", factory)


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path / "config.toml")


def _settle(app: QApplication, *, ticks: int = 12) -> None:
    """Прокрутить цикл событий, чтобы фоновая валидация редактора доставила сигнал.

    Редактор гоняет разбор команды в демон-нити и возвращает результат очередью;
    если снести редактор, пока сигнал в полёте, Qt ругнётся в удалённый объект.
    Короткая прокрутка даёт быстрой проверке одного блока успеть до разборки.
    """
    for _ in range(ticks):
        app.processEvents()
        time.sleep(0.01)


def _window(
    theme: ThemeManager,
    manager: ConfigManager,
    *,
    bus: EventBus | None = None,
    **kwargs: object,
) -> MainWindow:
    return MainWindow(theme=theme, manager=manager, bus=bus, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Проводка бокового меню: сигнал раздела ведёт окно в open_section
# --------------------------------------------------------------------------- #


def test_sidebar_signal_switches_section(theme: ThemeManager, manager: ConfigManager) -> None:
    """Выбор в боковом меню эмитит `section_selected`, и окно открывает раздел.

    `open_section` тесты уже зовут напрямую; здесь проверяется звено до него —
    что клик по пункту меню (через `currentItemChanged` → сигнал) доводит запрос
    до окна, создаёт страницу и делает её текущей. Окно строится на «Общих», так
    что переход на «overlay» — настоящая смена, а не повтор стартового раздела.
    """
    window = _window(theme, manager)
    try:
        assert window.current_section == "general"
        selected: list[str] = []
        window._sidebar.section_selected.connect(selected.append)

        assert window._sidebar.select_section("overlay") is True

        assert selected == ["overlay"]  # прошло через сигнал, а не прямой вызов
        assert window.current_section == "overlay"
        assert "overlay" in window.created_sections
    finally:
        window.exit()


# --------------------------------------------------------------------------- #
# Запись настройки через страницу, собранную фабрикой реестра
# --------------------------------------------------------------------------- #


def test_factory_page_writes_setting_through_manager(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    """Тумблер на странице, которую собрала фабрика реестра, пишет в конфиг.

    Окно не знает полей «Общих» — их несёт `GeneralTab`, поднятый `open_section`
    через `register_tab`. Тест правит привязанный тумблер, сбрасывает отложенную
    запись и убеждается, что значение дошло до `manager.settings`, а не осело в
    виджете. Флип от текущего значения делает проверку независимой от умолчания.
    """
    window = _window(theme, manager)
    try:
        page = window.open_section("general")
        page.load_from_config()
        toggle = page._bindings["general.start_minimized"].widget
        target = not manager.settings.general.start_minimized

        toggle.setChecked(target)
        page.flush_pending()

        assert manager.settings.general.start_minimized is target
    finally:
        window.exit()


def _seed_command(path: Path) -> int:
    """Поднять процесс-БД в файле, создать активный профиль и одну команду.

    Фабрика «Команд» строит хранилище лениво через `get_database()`, поэтому окну
    нужна установленная процесс-БД с активным профилем — иначе вкладка выродится в
    заглушку «библиотека недоступна». Возвращает id созданной команды.
    """
    database = init_database(path)
    repositories = Repositories(database)
    profile = repositories.profiles.create("Тест", activate=True)
    command = repositories.commands.create(Command(name="Свет", profile_id=profile.id))
    assert command.id is not None
    return command.id


# --------------------------------------------------------------------------- #
# «Погружение»: открытие редактора команды прячет меню и поиск слоя
# --------------------------------------------------------------------------- #


def test_opening_command_editor_collapses_nav(
    app: QApplication, theme: ThemeManager, manager: ConfigManager, tmp_path: Path
) -> None:
    """Открытие редактора команды сворачивает боковое меню и поиск настроек.

    `CommandsTab` эмитит `immersive_changed(True)`, когда встаёт на экран редактора,
    а окно связывает это со скрытием навигации и поля поиска — нодовому холсту отдан
    весь слой. Тест ведёт всю цепочку через окно: поднимает раздел, открывает команду,
    затем возвращается в список и проверяет, что меню с поиском вернулись.
    """
    command_id = _seed_command(tmp_path / "ayris.db")
    try:
        window = _window(theme, manager, bus=EventBus(thread_id=None))
        try:
            page = window.open_section("commands")
            assert window._sidebar.isHidden() is False
            assert window._search_field.isHidden() is False

            page._open_editor(command_id)
            assert page._stack.currentIndex() == 1
            assert window._sidebar.isHidden() is True
            assert window._search_field.isHidden() is True

            page._back_to_library()
            assert page._stack.currentIndex() == 0
            assert window._sidebar.isHidden() is False
            assert window._search_field.isHidden() is False
        finally:
            _settle(app)  # дать фоновой валидации редактора долететь до разборки
            window.exit()
    finally:
        reset_database()


# --------------------------------------------------------------------------- #
# Событие шины «Открыть команду» поднимает настройки и раздел «Команды»
# --------------------------------------------------------------------------- #


def test_open_command_request_reveals_it_in_settings(
    app: QApplication, theme: ThemeManager, manager: ConfigManager, tmp_path: Path
) -> None:
    """`OpenCommandRequested` открывает слой настроек и раздел «Команды».

    Ссылка «Открыть команду» из вкладки «Горячие клавиши» кидает событие на шину;
    окно ловит его, поднимает слой настроек, переключается на «Команды» и просит
    вкладку показать нужную. Тест проверяет реакцию окна на событие целиком, а не
    прямой вызов `open_section`.
    """
    command_id = _seed_command(tmp_path / "ayris.db")
    bus = EventBus(thread_id=None)
    try:
        window = _window(theme, manager, bus=bus)
        try:
            assert window.settings_open is False

            bus.publish(OpenCommandRequested(command_id=command_id))

            assert window.settings_open is True
            assert window.current_section == "commands"
            assert "commands" in window.created_sections
        finally:
            _settle(app)
            window.exit()
    finally:
        reset_database()
