"""Редактор команд списком (задача 52), offscreen.

Ни один пиксель не проверяется. Каждое утверждение — про состояние рабочей
:class:`CommandModel`, про сигналы редактора (``command_saved``,
``dirty_changed``) и про текст статуса. Сервисы, которым нужно железо или живой
воркер, подменены фейками: раннер теста, предпросмотр звука. Валидация и тест
редактора идут в daemon-потоке через queued-сигнал, поэтому там, где важен
результат, событийный цикл прокручивается ``processEvents`` в цикле ожидания, а
чистые обработчики (``_on_validation`` / ``_on_test_finished``) в остальных
местах зовутся напрямую с готовым объектом — на реальный тайминг тест не
опирается. Каждый созданный виджет закрывается в фикстуре ``app``; редактор
держит несколько ``QCompleter``, чьи всплывающие окна иначе удержали бы цикл.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.actions.macros.validator import (
    Problem,
    Severity,
    ValidationReport,
)
from ayris.core.database import Database
from ayris.core.errors import AyrisError
from ayris.core.models import Command, CommandFolder
from ayris.core.repositories import Repositories
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.macro_editor import (
    _SELECT_HINT,
    MacroEditor,
    MacroEditorServices,
    MacroTestResult,
    MacroTestRunner,
    _AsyncRunner,
    _referenced_names,
    _status_glyph,
    _TestStage,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# фикстуры (app + theme взяты дословно из test_param_form)
# ----------------------------------------------------------------------


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
    manager = ThemeManager(app)
    manager.apply()
    return manager


@pytest.fixture
def repos() -> Iterator[Repositories]:
    database = Database.open(":memory:")
    yield Repositories(database)
    database.close()


@pytest.fixture
def store(repos: Repositories) -> CommandTreeStore:
    profile = repos.profiles.create("Основной", activate=True)
    assert profile.id is not None
    folder = repos.folders.create(CommandFolder(name="Работа", profile_id=profile.id))
    assert folder.id is not None
    # Две команды в корне — одна папка с командой. «Свет» и «Тьма» лежат в корне
    # вместе, так что переименование одной в другую даёт коллизию имён сиблингов.
    repos.commands.create(Command(name="Свет", profile_id=profile.id, priority=10))
    repos.commands.create(Command(name="Тьма", profile_id=profile.id))
    repos.commands.create(Command(name="Игра", profile_id=profile.id, folder_id=folder.id))
    return CommandTreeStore(repos, profile.id)


# Реестр действий поднимается один раз на модуль: BlockCatalog() без реестра
# зовёт discover(), а это дорого повторять для каждого редактора.
@pytest.fixture(scope="module")
def catalog() -> BlockCatalog:
    return BlockCatalog()


class FakeRunner:
    """Раннер «Теста» без воркера: запоминает вызовы и отдаёт готовый результат."""

    def __init__(self, result: MacroTestResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def run_test(
        self,
        command: CommandModel,
        *,
        slots: Mapping[str, Any],
        dry_run: bool,
    ) -> MacroTestResult:
        self.calls.append({"command": command, "slots": dict(slots), "dry_run": dry_run})
        return self._result


class FakePreview:
    """Предпросмотр звука без аудио-устройства."""

    def __init__(self) -> None:
        self.stopped = 0

    def preview_binding(self, binding: object) -> object:
        return object()

    def stop(self) -> None:
        self.stopped += 1

    def duration_ms(self, binding: object) -> int | None:
        return 1200


_OK_RESULT = MacroTestResult(
    outcome="ok",
    message="Готово",
    stages=(_TestStage(path="actions[0]", status="ok", duration_ms=12.0, message="шаг"),),
    dry_run=False,
)


def _command_id(store: CommandTreeStore, name: str) -> int:
    for command in store.commands():
        if command.name == name and command.id is not None:
            return command.id
    raise AssertionError(name)


def _editor(
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    *,
    runner: MacroTestRunner | None = None,
    preview: object | None = None,
) -> MacroEditor:
    services = MacroEditorServices(
        catalog=catalog,
        event_names=("system.ready", "system.sleep"),
        sound_preview=preview,  # type: ignore[arg-type]
        test_runner=runner,
    )
    return MacroEditor(store, theme, services=services)


def _pump(app: QApplication, predicate: object, *, timeout: float = 5.0) -> bool:
    """Крутить событийный цикл, пока предикат не станет истинным или не выйдет время."""
    assert callable(predicate)
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


# ----------------------------------------------------------------------
# чистые хелперы модуля
# ----------------------------------------------------------------------


def test_status_glyph_maps_known_and_unknown() -> None:
    assert _status_glyph("ok") == "✓"
    assert _status_glyph("failed") == "✗"
    assert _status_glyph("skipped") == "⊘"
    assert _status_glyph("cancelled") == "■"
    assert _status_glyph("что-то другое") == "•"


def test_referenced_names_collects_braced_from_all_blocks() -> None:
    command = CommandModel(
        name="Ссылки",
        actions=[
            ActionBlock(type="Say", params={"text": "привет, {имя}"}),
            ActionBlock(
                type="If",
                params={"condition": "{готово}"},
                then=[ActionBlock(type="TypeText", params={"text": "{адрес}", "count": 3})],
            ),
        ],
    )
    # Собираются имена из всех блоков, включая ветку then; нестроковый параметр
    # (count=3) не мешает.
    assert _referenced_names(command) == {"имя", "готово", "адрес"}


def test_referenced_names_empty_for_plain_command() -> None:
    command = CommandModel(name="Пусто", actions=[ActionBlock(type="Say", params={"text": "тут"})])
    assert _referenced_names(command) == set()


# ----------------------------------------------------------------------
# точки останова: холст ↔ хранилище сессии отладки
# ----------------------------------------------------------------------


def test_breakpoint_toggle_persists_and_reloads(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    cmd_id = _command_id(store, "Свет")
    editor.load_command(cmd_id)
    # Дать команде блок и сохранить, чтобы у ноды был реальный путь и после reload.
    editor._action_model.insert_type("Say", ("actions",), 0)
    block = editor._action_model.block_at(("actions", 0))
    assert block is not None
    block.params = {"text": "раз"}
    editor._node_editor.rebuild()
    editor.save()

    # Двойной клик по ноде (через сцену) ставит точку останова → она уходит в
    # хранилище сессии отладки, откуда её читает MacroDebugger при запуске.
    editor._node_editor._scene.toggle_breakpoint("actions[0]")
    snapshot = store.debug_store.load(cmd_id)
    assert snapshot is not None
    assert [bp.path for bp in snapshot.breakpoints] == ["actions[0]"]

    # Повторное открытие команды возвращает точку на холст.
    editor.load_command(cmd_id)
    assert editor._node_editor.breakpoints() == {"actions[0]"}

    # Снятие точки очищает запись в хранилище.
    editor._node_editor._scene.toggle_breakpoint("actions[0]")
    reloaded = store.debug_store.load(cmd_id)
    assert reloaded is not None
    assert reloaded.breakpoints == []


def test_breakpoint_save_preserves_watches_and_slots(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    from ayris.actions.macros.debug_session import DebugSessionSnapshot

    editor = _editor(store, theme, catalog)
    cmd_id = _command_id(store, "Свет")
    editor.load_command(cmd_id)
    editor._action_model.insert_type("Say", ("actions",), 0)
    block = editor._action_model.block_at(("actions", 0))
    assert block is not None
    block.params = {"text": "раз"}
    editor._node_editor.rebuild()
    editor.save()

    # Отладчик уже сохранил watch и подмену слота для этой команды.
    store.debug_store.save(
        DebugSessionSnapshot(
            command_id=cmd_id,
            watches=["{x}"],
            slots_override={"phrase": "тест"},
        )
    )
    # Постановка точки на холсте не должна затирать watch/slots.
    editor._node_editor._scene.toggle_breakpoint("actions[0]")
    snapshot = store.debug_store.load(cmd_id)
    assert snapshot is not None
    assert [bp.path for bp in snapshot.breakpoints] == ["actions[0]"]
    assert snapshot.watches == ["{x}"]
    assert snapshot.slots_override == {"phrase": "тест"}


# ----------------------------------------------------------------------
# _AsyncRunner: успех, ошибка Ayris, любое исключение, реальный поток
# ----------------------------------------------------------------------


def test_async_runner_finished_carries_result(app: QApplication) -> None:
    runner = _AsyncRunner()
    seen: list[object] = []
    runner.finished.connect(seen.append)
    # Прямой вызов _run синхронен — очередь событий не нужна.
    runner._run(lambda: "готово")
    assert seen == ["готово"]


def test_async_runner_reports_ayris_error_user_message(app: QApplication) -> None:
    runner = _AsyncRunner()
    failures: list[str] = []
    runner.failed.connect(failures.append)

    def boom() -> object:
        raise AyrisError("boom", user_message="Не вышло по-русски.")

    runner._run(boom)
    assert failures == ["Не вышло по-русски."]


def test_async_runner_reports_generic_exception(app: QApplication) -> None:
    runner = _AsyncRunner()
    failures: list[str] = []
    runner.failed.connect(failures.append)

    def boom() -> object:
        raise ValueError("сырое исключение")

    runner._run(boom)
    assert failures == ["сырое исключение"]


def test_async_runner_real_thread_delivers_via_queue(app: QApplication) -> None:
    runner = _AsyncRunner()
    seen: list[object] = []
    runner.finished.connect(seen.append)
    runner.run(lambda: 42)  # запускает daemon-поток
    assert _pump(app, lambda: seen == [42])


# ----------------------------------------------------------------------
# загрузка команды
# ----------------------------------------------------------------------


def test_load_command_fills_sections_and_shows_content(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    light = _command_id(store, "Свет")
    editor.load_command(light)
    assert editor.command_id == light
    assert editor.is_dirty is False
    # Контент показан, плейсхолдер спрятан.
    assert editor._content.isHidden() is False
    assert editor._placeholder.isHidden() is True


def test_load_missing_command_shows_placeholder(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(999_999)  # такого id нет
    assert editor.command_id is None
    assert editor._placeholder.isHidden() is False
    assert editor._content.isHidden() is True
    assert "не найдена" in editor._placeholder.text().lower()


def test_load_runs_validation_and_reports_warnings(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    # У «Свет» нет ни блоков, ни триггеров — валидатор в потоке даёт два
    # предупреждения; ждём queued-сигнал _on_validation.
    assert _pump(app, lambda: editor._status.text() == "Предупреждений: 2.")


# ----------------------------------------------------------------------
# правка секций метит dirty
# ----------------------------------------------------------------------


def test_header_edit_marks_dirty(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    dirty: list[bool] = []
    editor.dirty_changed.connect(dirty.append)
    editor._header._name.setText("Свет новый")
    assert editor.is_dirty is True
    assert dirty and dirty[-1] is True
    assert editor._model is not None and editor._model.name == "Свет новый"


def test_trigger_edit_marks_dirty_and_updates_conflicts(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._triggers._new_voice()  # добавляет голосовую карточку и коммитит
    assert editor.is_dirty is True
    assert editor._model is not None and editor._model.triggers


def test_variable_edit_marks_dirty(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._variables._add_row()
    assert editor.is_dirty is True
    assert editor._model is not None and editor._model.variables


def test_sound_edit_marks_dirty(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, preview=FakePreview())
    editor.load_command(_command_id(store, "Свет"))
    editor._sounds.changed.emit()  # SoundBindingSection.changed → _on_model_changed
    assert editor.is_dirty is True


# ----------------------------------------------------------------------
# палитра и форма параметров
# ----------------------------------------------------------------------


def test_palette_choice_inserts_block_at_root(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("Say")
    assert editor._model is not None
    assert [b.type for b in editor._model.actions] == ["Say"]
    assert editor.is_dirty is True


def test_palette_choice_inserts_after_selection(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("Say")  # actions[0]
    editor._action_view.select_path(("actions", 0))
    editor._on_palette_choice("Wait")  # вставляется после выбранного
    assert editor._model is not None
    assert [b.type for b in editor._model.actions] == ["Say", "Wait"]


def test_palette_add_in_node_view_is_a_free_node(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # BUG 2: на нодовом холсте блок из палитры добавлялся НЕ свободным и авто-сцеплялся с
    # соседом по списку. На холсте поток задают провода, поэтому палитра кладёт ноду
    # detached — ровно как собственная «＋ Нода» редактора.
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    assert editor._current_action_view() is editor._node_editor
    editor._on_palette_choice("Say")
    block = editor._action_model.block_at(("actions", 0))
    assert block is not None and block.detached is True


def test_palette_add_in_list_view_stays_wired(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # В списке порядок и есть поток, поэтому добавленный блок остаётся подключённым
    # (detached=False) — свободными ноды делает только нодовый холст.
    services = MacroEditorServices(catalog=catalog, action_view="list")
    editor = MacroEditor(store, theme, services=services)
    editor.load_command(_command_id(store, "Свет"))
    assert editor._current_action_view() is editor._action_view
    editor._on_palette_choice("Say")
    block = editor._action_model.block_at(("actions", 0))
    assert block is not None and block.detached is False


def test_block_selected_builds_param_form(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("If")  # выбирается новый путь, форма строится
    assert editor._param_title.text() == "Параметры — Если"
    # У блока If есть параметр condition — форма непустая.
    assert editor._param_form.values() is not None
    assert editor._param_form._rows


def test_block_selected_unknown_type_uses_raw_title(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    # Блок неизвестного каталогу типа: заголовок — сам тип, полей нет.
    editor._action_model.insert(ActionBlock(type="MysteryBlock"), ("actions",), 0)
    editor._action_view.rebuild()
    editor._on_block_selected(("actions", 0))
    assert editor._param_title.text() == "Параметры — MysteryBlock"
    assert editor._param_form._rows == []


def test_block_selected_empty_path_clears_form(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("If")
    editor._on_block_selected(())  # пустой путь
    assert editor._param_title.text() == "Параметры"
    assert editor._param_form._rows == []


def test_block_selected_missing_block_clears_form(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("If")
    editor._on_block_selected(("actions", 42))  # блока по такому пути нет
    assert editor._param_title.text() == "Параметры"


def test_inspector_reads_as_a_panel_with_a_prompt_when_empty(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Панель параметров — карта (surface+рамка), а не прозрачная полоса, иначе на
    # холсте её не видно, пока нода не выбрана. Пустое состояние — подсказка выбрать
    # ноду, а не «нет параметров» (это честный текст только для выбранного блока).
    editor = _editor(store, theme, catalog)
    assert editor._param_form.parentWidget().property("card") is True
    editor.load_command(_command_id(store, "Свет"))
    # Ничего не выбрано — панель зовёт выбрать ноду, а не молчит пустой.
    assert editor._param_form._empty.text() == _SELECT_HINT
    # Выбор реального блока строит форму и снимает подсказку выбора: пустой текст
    # снова честный «нет параметров» (виден лишь у блока без полей, но не подсказка).
    editor._on_palette_choice("If")  # у If есть параметр — форма непустая
    assert editor._param_form._rows
    assert editor._param_form._empty.text() != _SELECT_HINT


def test_params_changed_writes_into_block(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("If")  # actions[0] выбран, форма построена
    editor._param_form.set_values({"condition": "{ready}"})
    editor._on_params_changed()
    block = editor._action_model.block_at(("actions", 0))
    assert block is not None and block.params.get("condition") == "{ready}"


def test_params_changed_without_selection_is_noop(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("Say")
    editor._action_view.clearSelection()
    before = editor._model.actions[0].params if editor._model else {}
    editor._on_params_changed()  # блок не выбран — ничего не меняется
    assert editor._model is not None and editor._model.actions[0].params == before


def test_typing_param_in_node_view_survives_rebuild(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Регресс: правка текстового поля параметра при активном НОДОВОМ виде роняла
    # приложение. Ввод → _on_params_changed → rebuild нодов → повторный выбор того же
    # блока → _on_block_selected пересобирал форму и удалял QLineEdit прямо во время
    # его textChanged (use-after-free). Форма обязана пережить правку.
    editor = _editor(store, theme, catalog)
    assert editor._current_action_view() is editor._node_editor
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("SetVolume")  # блок вставлен и выбран, форма построена
    rows = {row.field.name: row for row in editor._param_form._rows}
    assert "device" in rows  # текстовое поле «Часть названия устройства»
    setter, getter = rows["device"].setter, rows["device"].getter
    assert callable(setter) and callable(getter)
    setter("Нау")  # имитируем ввод: textChanged → _on_params_changed → rebuild
    # Поле пережило правку (у удалённого C++-объекта .text() бросил бы) и значение
    # дошло до блока.
    assert getter() == "Нау"
    block = editor._action_model.block_at(("actions", 0))
    assert block is not None and block.params.get("device") == "Нау"


# ----------------------------------------------------------------------
# режим редактора действий по умолчанию (ноды) и его запоминание
# ----------------------------------------------------------------------


def test_default_action_view_is_nodes(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # По умолчанию редактор действий открывается нодовым холстом, а не списком:
    # это и есть «плавающие ноды», которые иначе спрятаны за невыбранным тумблером.
    editor = _editor(store, theme, catalog)
    assert editor._mode_stack.currentIndex() == 1
    assert editor._nodes_button.isChecked()
    assert not editor._list_button.isChecked()
    assert editor._current_action_view() is editor._node_editor


def test_list_view_default_when_configured(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Запомненный выбор «Список» уважается на следующем открытии.
    services = MacroEditorServices(catalog=catalog, action_view="list")
    editor = MacroEditor(store, theme, services=services)
    assert editor._mode_stack.currentIndex() == 0
    assert editor._list_button.isChecked()
    assert editor._current_action_view() is editor._action_view


def test_load_command_opens_on_actions_tab(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Команду открывают, чтобы увидеть её действия, поэтому load сразу встаёт на
    # вкладку «Действия», а не на «Обзор» с именем и триггерами.
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    assert editor._tabs.tabText(editor._tabs.currentIndex()) == "Действия"


def test_toggle_action_view_reports_choice(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Каждый переворот тумблера отдаётся наружу, чтобы вкладка его сохранила —
    # так следующая команда и следующий запуск открываются в том же виде.
    seen: list[str] = []
    services = MacroEditorServices(catalog=catalog, on_action_view_changed=seen.append)
    editor = MacroEditor(store, theme, services=services)
    editor.load_command(_command_id(store, "Свет"))
    editor._list_button.click()
    editor._nodes_button.click()
    assert seen == ["list", "nodes"]


def test_second_click_on_list_collapses_back_to_nodes(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Тумблер читается как «открыть/закрыть»: первый клик по «Список» открывает
    # список, повторный — сворачивает обратно к нодам (домашний вид), а не залипает.
    editor = _editor(store, theme, catalog)
    editor._list_button.click()
    assert editor._mode_stack.currentIndex() == 0
    assert editor._list_button.isChecked()
    editor._list_button.click()
    assert editor._mode_stack.currentIndex() == 1
    assert editor._nodes_button.isChecked()
    assert not editor._list_button.isChecked()


def test_second_click_on_active_nodes_stays_home(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # У домашнего вида нет запасного, поэтому повторный клик по активным «Нодам»
    # ничего не переключает — иначе праздный клик выбросил бы в список.
    editor = _editor(store, theme, catalog)
    editor._nodes_button.click()
    assert editor._mode_stack.currentIndex() == 1
    assert editor._nodes_button.isChecked()


# ----------------------------------------------------------------------
# сохранение
# ----------------------------------------------------------------------


def test_save_persists_and_emits_command_saved(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    light = _command_id(store, "Свет")
    editor.load_command(light)
    saved: list[int] = []
    editor.command_saved.connect(saved.append)
    editor._header._name.setText("Свет 2")  # свободное имя
    editor.save()
    assert editor._status.text() == "Команда сохранена."
    assert editor.is_dirty is False
    assert saved == [light]
    assert store.command_model(light).name == "Свет 2"


def test_save_rejects_empty_name(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    light = _command_id(store, "Свет")
    editor.load_command(light)
    saved: list[int] = []
    editor.command_saved.connect(saved.append)
    editor._header._name.setText("")  # пустое имя невалидно
    editor.save()
    assert "Исправьте имя" in editor._status.text()
    assert editor._tabs.currentIndex() == 0
    assert saved == []
    assert store.command_model(light).name == "Свет"  # ничего не сохранено


def test_save_rejects_duplicate_name(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    light = _command_id(store, "Свет")
    editor.load_command(light)
    editor._header._name.setText("Тьма")  # имя сиблинга в том же корне
    editor.save()
    assert "Исправьте имя" in editor._status.text()
    assert store.command_model(light).name == "Свет"


def test_save_without_model_is_noop(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.save()  # команда не загружена
    assert editor.command_id is None
    # заодно ранний выход _refresh_completions без модели
    editor._refresh_completions()


def test_save_store_error_shows_user_message(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    assert editor._model is not None
    editor._model.id = None  # store.save_command бросит AyrisError (нет id)
    editor.save()  # имя валидно, падаем на store
    assert "нельзя сохранить" in editor._status.text().lower()


# ----------------------------------------------------------------------
# set_store и свойства
# ----------------------------------------------------------------------


def test_set_store_clears_model_and_shows_placeholder(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor.set_store(store)
    assert editor.command_id is None
    assert editor._placeholder.isHidden() is False
    assert editor._content.isHidden() is True


# ----------------------------------------------------------------------
# валидация: чистый слот _on_validation
# ----------------------------------------------------------------------


def test_on_validation_reports_errors_and_warnings(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    report = ValidationReport(
        problems=(
            Problem(message="ошибка", severity=Severity.ERROR),
            Problem(message="ещё ошибка", severity=Severity.ERROR),
            Problem(message="предупреждение", severity=Severity.WARNING),
        )
    )
    editor._on_validation(report)
    assert editor._status.text() == "Ошибок: 2, предупреждений: 1."


def test_on_validation_reports_only_warnings(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    report = ValidationReport(problems=(Problem(message="w", severity=Severity.WARNING),))
    editor._on_validation(report)
    assert editor._status.text() == "Предупреждений: 1."


def test_on_validation_reports_pass(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_validation(ValidationReport(problems=()))
    assert editor._status.text() == "Проверка пройдена."


def test_on_validation_ignores_non_report(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._status.setText("исходный")
    editor._on_validation("не отчёт")  # не ValidationReport — статус не трогаем
    assert editor._status.text() == "исходный"


def test_schedule_validation_without_model_is_noop(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor._schedule_validation()  # модель не загружена — ранний выход
    assert editor._model is None


# ----------------------------------------------------------------------
# тест-раннер
# ----------------------------------------------------------------------


def test_test_button_disabled_without_runner(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    assert editor._test_button.isEnabled() is False


def test_test_button_enabled_with_runner(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    assert editor._test_button.isEnabled() is True


def test_on_test_without_runner_is_noop(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)  # раннера нет
    editor.load_command(_command_id(store, "Свет"))
    editor._status.setText("тихо")
    editor._on_test()
    assert editor._status.text() == "тихо"


def test_on_test_without_model_is_noop(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    editor._status.setText("тихо")
    editor._on_test()  # команда не загружена
    assert editor._status.text() == "тихо"


def test_on_test_rejects_invalid_name(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    editor.load_command(_command_id(store, "Свет"))
    editor._header._name.setText("")  # имя невалидно
    editor._on_test()
    assert "Исправьте имя" in editor._status.text()


def test_on_test_store_error_shows_message(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    editor.load_command(_command_id(store, "Свет"))
    assert editor._model is not None
    editor._model.id = None  # save_command внутри теста бросит AyrisError
    editor._on_test()
    assert "нельзя сохранить" in editor._status.text().lower()


def test_on_test_runs_and_writes_log(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    runner = FakeRunner(_OK_RESULT)
    editor = _editor(store, theme, catalog, runner=runner)
    editor.load_command(_command_id(store, "Свет"))
    # Дождаться валидации после загрузки, чтобы её сигнал не пришёл поверх теста.
    assert _pump(app, lambda: editor._status.text() == "Предупреждений: 2.")
    editor._on_test()  # летит в daemon-поток, результат — queued-сигналом
    assert _pump(app, lambda: bool(editor._log.toPlainText()))
    assert "actions[0]" in editor._log.toPlainText()
    assert editor._status.text() == "Готово"
    assert len(runner.calls) == 1
    assert runner.calls[0]["dry_run"] is False  # у «Свет» нет опасных блоков


def test_on_test_dry_run_for_dangerous_block(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    runner = FakeRunner(_OK_RESULT)
    editor = _editor(store, theme, catalog, runner=runner)
    editor.load_command(_command_id(store, "Свет"))
    editor._on_palette_choice("RunShell")  # опасный блок в каталоге
    assert _pump(app, lambda: bool(editor._status.text()))
    editor._on_test()
    assert _pump(app, lambda: bool(editor._log.toPlainText()))
    assert runner.calls and runner.calls[-1]["dry_run"] is True


def test_is_dangerous_true_for_runshell(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    assert editor._is_dangerous("RunShell") is True
    assert editor._is_dangerous("НетТакого") is False


def test_on_test_finished_dry_run_prefix_and_empty_message(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    editor.load_command(_command_id(store, "Свет"))
    result = MacroTestResult(
        outcome="ok",
        message="Проверено",
        stages=(
            _TestStage(path="actions[0]", status="ok", duration_ms=5.0),  # без message
            _TestStage(path="actions[1]", status="failed", duration_ms=7.0, message="упал"),
        ),
        dry_run=True,
    )
    editor._on_test_finished(result)
    assert editor._status.text() == "Сухой прогон. Проверено"
    log = editor._log.toPlainText()
    assert "actions[0]" in log and "упал" in log
    assert editor._test_button.isEnabled() is True


def test_on_test_finished_ignores_non_result(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    editor.load_command(_command_id(store, "Свет"))
    editor._log.setPlainText("старый лог")
    editor._on_test_finished("не результат")
    assert editor._log.toPlainText() == "старый лог"  # лог не тронут
    assert editor._test_button.isEnabled() is True


def test_on_test_failed_shows_message(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog, runner=FakeRunner(_OK_RESULT))
    editor.load_command(_command_id(store, "Свет"))
    editor._on_test_failed("воркер умер")
    assert editor._status.text() == "Тест не выполнен: воркер умер"
    assert editor._test_button.isEnabled() is True


# ----------------------------------------------------------------------
# Protocol MacroTestRunner: тело метода для покрытия
# ----------------------------------------------------------------------


def test_runner_protocol_body_is_callable() -> None:
    # Тело метода протокола — многоточие; вызвать его напрямую, чтобы строка
    # засчиталась и MacroTestRunner был реально задействован.
    class Stub:
        pass

    result = MacroTestRunner.run_test(Stub(), CommandModel(name="x"), slots={}, dry_run=False)
    assert result is None
