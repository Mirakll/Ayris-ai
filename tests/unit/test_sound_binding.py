"""Секция звуков команды (задача 52), offscreen.

Ничего не рисуется и не проигрывается. Каждая проверка — про состояние модели и
про сигналы: что три стадии (запуск/успех/ошибка) кладут по одной привязке в
``model.sounds``, что смена источника и значения летит в модель и шлёт ``changed``,
что превью зовёт инжектированный фейк :class:`SoundPreview` (а без сервиса кнопка
выключена) и что длительность берётся из ``duration_ms``. Реального звука,
сети и буфера обмена здесь нет — только фейк со счётчиками вызовов.

Каждый созданный виджет закрывается в фикстуре ``app``; поповеров и таймеров этот
виджет не держит, так что закрытия верхнеуровневых окон достаточно.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.schema import (
    CommandModel,
    SoundBinding,
    SoundSource,
    SoundStage,
)
from ayris.actions.macros.sounds import SoundImportError
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sound_binding import SoundBindingRow, SoundBindingSection, _ImportRunner

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


def _pump(app: QApplication, predicate: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    """Крутить событийный цикл, пока предикат не станет истинным или не выйдет время."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


@pytest.fixture
def theme(app: QApplication) -> ThemeManager:
    manager = ThemeManager(app)
    manager.apply()
    return manager


class FakePreview:
    """Фейк сервиса воспроизведения — только считает вызовы, звука не издаёт.

    ``fail_preview`` заставляет :meth:`preview_binding` кинуть, чтобы проверить, что
    секция глотает ошибку; ``fail_duration`` — то же для :meth:`duration_ms`.
    ``duration`` — что вернуть как длительность (или ``None``).
    """

    def __init__(
        self,
        *,
        duration: int | None = None,
        fail_preview: bool = False,
        fail_duration: bool = False,
    ) -> None:
        self.duration = duration
        self.fail_preview = fail_preview
        self.fail_duration = fail_duration
        self.previews: list[SoundBinding] = []
        self.stops = 0
        self.duration_calls: list[SoundBinding] = []

    def preview_binding(self, binding: SoundBinding) -> object:
        self.previews.append(binding)
        if self.fail_preview:
            raise RuntimeError("сломанное воспроизведение")
        return object()  # дескриптор с cancel(); виджет его хранит, но не зовёт

    def stop(self) -> None:
        self.stops += 1

    def duration_ms(self, binding: SoundBinding) -> int | None:
        self.duration_calls.append(binding)
        if self.fail_duration:
            raise RuntimeError("не удалось измерить")
        return self.duration


class FakeImporter:
    """Фейк импортёра — вместо копирования файла возвращает имя, считает вызовы.

    ``result`` — имя файла, которое отдать как сохранённое в папке профиля.
    ``fail`` заставляет :meth:`import_file` кинуть доменную ошибку (её сообщение
    секция показывает у строки), ``crash`` — неожиданную (её текст наружу не идёт).
    """

    def __init__(
        self,
        *,
        result: str = "мой звук.wav",
        fail: bool = False,
        crash: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
        self.crash = crash
        self.imported: list[Path] = []

    def import_file(self, source: Path) -> str:
        self.imported.append(source)
        if self.crash:
            raise RuntimeError("неожиданный сбой")
        if self.fail:
            raise SoundImportError("нельзя декодировать", user_message="Файл повреждён.")
        return self.result


def _builtin(stage: SoundStage = SoundStage.ON_START, name: str = "chime") -> SoundBinding:
    return SoundBinding(stage=stage, source=SoundSource.BUILTIN, value=name)


def _tts(stage: SoundStage = SoundStage.ON_SUCCESS, text: str = "Готово") -> SoundBinding:
    return SoundBinding(stage=stage, source=SoundSource.TTS, value=text)


def _file(stage: SoundStage = SoundStage.ON_ERROR, name: str = "alarm.wav") -> SoundBinding:
    return SoundBinding(stage=stage, source=SoundSource.FILE, value=name)


# ----------------------------------------------------------------------
# SoundBindingRow: состояние по умолчанию и set_binding
# ----------------------------------------------------------------------


def test_row_default_has_no_sound(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    # «Без звука» выбран, значение/громкость/ожидание выключены, привязки нет.
    assert row.binding() is None
    assert not row._value.isEnabled()
    assert not row._volume.isEnabled()
    assert not row._wait.isEnabled()
    assert row._value.placeholderText() == "Звук не задан"
    assert not row._stop.isEnabled()


def test_row_set_binding_fills_all_controls(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    seen: list[int] = []
    row.changed.connect(lambda: seen.append(1))
    binding = SoundBinding(
        stage=SoundStage.ON_START,
        source=SoundSource.BUILTIN,
        value="chime",
        volume=40,
        wait=True,
    )
    row.set_binding(binding)
    # Загрузка не выглядит как правка пользователя.
    assert seen == []
    # currentData() у ThemedComboBox отдаёт строковое значение StrEnum, отсюда ==.
    assert row._enabled.currentData() == SoundSource.BUILTIN
    assert row._value.text() == "chime"
    assert row._volume.value() == 40
    assert row._wait.isChecked() is True
    assert row._value.isEnabled()
    # binding() отдаёт то же самое обратно.
    assert row.binding() == binding


def test_row_set_binding_none_resets(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row.set_binding(_builtin(SoundStage.ON_START, "chime"))
    row.set_binding(None)
    assert row.binding() is None
    assert row._enabled.currentIndex() == 0
    assert row._value.text() == ""
    assert row._volume.value() == 100
    assert row._wait.isChecked() is False
    assert not row._value.isEnabled()


def test_row_set_binding_default_volume_is_full(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
    # У привязки без явной громкости в виджете 100 %.
    row.set_binding(_tts(SoundStage.ON_SUCCESS, "Готово"))
    assert row._volume.value() == 100
    assert row._value.placeholderText() == "Текст для озвучивания"


def test_row_set_binding_file_source(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_ERROR)
    binding = _file(SoundStage.ON_ERROR, "звук.wav")
    row.set_binding(binding)
    assert row._enabled.currentData() == SoundSource.FILE
    assert row._value.placeholderText() == "custom:имя-файла.wav"
    assert row.binding() == binding


# ----------------------------------------------------------------------
# SoundBindingRow: binding() и смена источника
# ----------------------------------------------------------------------


def test_row_binding_none_when_value_empty(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
    # Источник задан, но значение пустое — привязки нет.
    idx = row._enabled.findData(SoundSource.TTS)
    row._enabled.setCurrentIndex(idx)
    assert row._value.text() == ""
    assert row.binding() is None


def test_row_source_change_emits_and_enables(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
    seen: list[int] = []
    row.changed.connect(lambda: seen.append(1))
    idx = row._enabled.findData(SoundSource.TTS)
    row._enabled.setCurrentIndex(idx)
    # Смена источника вне загрузки — это правка: сигнал есть, поля включились.
    assert seen != []
    assert row._value.isEnabled()
    assert row._volume.isEnabled()
    assert row._wait.isEnabled()
    assert row._value.placeholderText() == "Текст для озвучивания"


def test_row_value_edit_emits_changed(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
    idx = row._enabled.findData(SoundSource.TTS)
    row._enabled.setCurrentIndex(idx)
    seen: list[int] = []
    row.changed.connect(lambda: seen.append(1))
    row._value.setText("Привет")
    assert seen != []
    got = row.binding()
    assert got is not None
    assert got.source is SoundSource.TTS
    assert got.value == "Привет"


def test_row_volume_and_wait_feed_binding(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    row._value.setText("Раз")
    row._volume.setValue(55)
    row._wait.setChecked(True)
    got = row.binding()
    assert got is not None
    assert got.volume == 55
    assert got.wait is True


# ----------------------------------------------------------------------
# SoundBindingRow: превью, стоп, длительность
# ----------------------------------------------------------------------


def test_row_preview_button_tracks_service_and_binding(
    app: QApplication, theme: ThemeManager
) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    # Без сервиса кнопка выключена, даже когда есть валидная привязка.
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    row._value.setText("Раз")
    assert not row._preview.isEnabled()
    # С сервисом и валидной привязкой — включена…
    row.set_preview_enabled(True)
    assert row._preview.isEnabled()
    # …а стоит очистить значение — снова выключена.
    row._value.clear()
    assert not row._preview.isEnabled()


def test_row_preview_click_emits_binding(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row.set_preview_enabled(True)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    row._value.setText("Раз")
    seen: list[object] = []
    row.preview_requested.connect(seen.append)
    row._preview.click()
    assert len(seen) == 1
    assert isinstance(seen[0], SoundBinding)
    assert seen[0].value == "Раз"


def test_row_preview_noop_without_binding(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    seen: list[object] = []
    row.preview_requested.connect(seen.append)
    # Прямой вызов при пустом значении: сигнал не летит.
    row._on_preview()
    assert seen == []


def test_row_stop_button_and_signal(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    assert not row._stop.isEnabled()
    row.set_playing(True)
    assert row._stop.isEnabled()
    seen: list[int] = []
    row.stop_requested.connect(lambda: seen.append(1))
    row._stop.click()
    assert seen == [1]
    row.set_playing(False)
    assert not row._stop.isEnabled()


def test_row_duration_label(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row.set_duration(2500)
    assert row._duration.text() == "2.5 с"
    row.set_duration(None)
    assert row._duration.text() == ""


# ----------------------------------------------------------------------
# SoundBindingSection: конструкция и set_command
# ----------------------------------------------------------------------


def test_section_stages_are_three_rows(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)
    rows = section.stages()
    assert len(rows) == 3
    assert all(isinstance(row, SoundBindingRow) for row in rows)


def test_section_without_preview_disables_buttons(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # preview=None
    model = CommandModel(name="Тест", sounds=[_tts(SoundStage.ON_SUCCESS, "Готово")])
    section.set_command(model)
    # Ни у одной строки превью не активно, длительность пустая.
    for row in section.stages():
        assert not row._preview.isEnabled()
    assert all(row._duration.text() == "" for row in section.stages())


def test_section_set_command_maps_bindings_by_stage(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview(duration=2500)
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(
        name="Тест",
        sounds=[_builtin(SoundStage.ON_START, "chime"), _file(SoundStage.ON_ERROR, "alarm.wav")],
    )
    section.set_command(model)
    rows = {row._stage: row for row in section.stages()}
    assert rows[SoundStage.ON_START].binding() == _builtin(SoundStage.ON_START, "chime")
    # У средней стадии звука нет — строка пустая.
    assert rows[SoundStage.ON_SUCCESS].binding() is None
    assert rows[SoundStage.ON_ERROR].binding() == _file(SoundStage.ON_ERROR, "alarm.wav")
    # Длительность спросили только для двух заданных стадий.
    assert len(fake.duration_calls) == 2
    assert rows[SoundStage.ON_START]._duration.text() == "2.5 с"


def test_section_set_command_duration_none(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview(duration=None)
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(name="Тест", sounds=[_builtin(SoundStage.ON_START, "chime")])
    section.set_command(model)
    rows = {row._stage: row for row in section.stages()}
    assert rows[SoundStage.ON_START]._duration.text() == ""


def test_section_set_command_swallows_duration_error(
    app: QApplication, theme: ThemeManager
) -> None:
    fake = FakePreview(fail_duration=True)
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(name="Тест", sounds=[_builtin(SoundStage.ON_START, "chime")])
    # Ошибка измерения длительности не всплывает наружу.
    section.set_command(model)
    rows = {row._stage: row for row in section.stages()}
    assert rows[SoundStage.ON_START]._duration.text() == ""


# ----------------------------------------------------------------------
# SoundBindingSection: правки летят в model.sounds
# ----------------------------------------------------------------------


def test_section_edit_writes_sound_into_model(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)
    model = CommandModel(name="Тест")
    section.set_command(model)
    seen: list[int] = []
    section.changed.connect(lambda: seen.append(1))
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    row._value.setText("Начали")
    assert seen != []
    assert len(model.sounds) == 1
    assert model.sounds[0].stage is SoundStage.ON_START
    assert model.sounds[0].source is SoundSource.TTS
    assert model.sounds[0].value == "Начали"


def test_section_clearing_source_removes_binding(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)
    model = CommandModel(name="Тест", sounds=[_tts(SoundStage.ON_START, "Начали")])
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    # Пользователь возвращает «Без звука» — привязка уходит из модели.
    row._enabled.setCurrentIndex(0)
    assert model.sounds == []


def test_section_edit_without_model_is_silent(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # set_command не вызывали → model=None
    seen: list[int] = []
    section.changed.connect(lambda: seen.append(1))
    row = section.stages()[0]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    row._value.setText("Раз")
    # _commit возвращается рано на model=None: сигнала секции нет.
    assert seen == []


# ----------------------------------------------------------------------
# SoundBindingSection: превью и стоп через фейк
# ----------------------------------------------------------------------


def test_section_preview_routes_to_service(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview()
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(name="Тест", sounds=[_tts(SoundStage.ON_START, "Раз")])
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    assert row._preview.isEnabled()
    stops_before = fake.stops
    row._preview.click()
    assert len(fake.previews) == 1
    assert fake.previews[0].value == "Раз"
    # Именно эта строка помечена играющей, стоп доступен.
    assert row._stop.isEnabled()
    # Перед стартом секция сначала всё останавливает.
    assert fake.stops == stops_before + 1


def test_section_preview_twice_restarts(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview()
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(name="Тест", sounds=[_tts(SoundStage.ON_START, "Раз")])
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._preview.click()
    row._preview.click()
    assert len(fake.previews) == 2
    # Второй запуск снова остановил предыдущий.
    assert fake.stops == 2


def test_section_stop_button_stops_service(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview()
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(name="Тест", sounds=[_tts(SoundStage.ON_START, "Раз")])
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._preview.click()
    stops_before = fake.stops
    row._stop.click()
    assert fake.stops == stops_before + 1
    assert not row._stop.isEnabled()


def test_section_preview_failure_is_swallowed(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview(fail_preview=True)
    section = SoundBindingSection(theme, preview=fake)
    model = CommandModel(name="Тест", sounds=[_tts(SoundStage.ON_START, "Раз")])
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    # Клик по превью не должен уронить UI, хотя сервис кинул.
    row._preview.click()
    assert len(fake.previews) == 1
    # Играющей строка не осталась: дескриптор сброшен, стоп выключен.
    assert not row._stop.isEnabled()


def test_section_on_preview_ignores_non_binding(app: QApplication, theme: ThemeManager) -> None:
    fake = FakePreview()
    section = SoundBindingSection(theme, preview=fake)
    # Аргумент не SoundBinding — ранний возврат, сервис не тронут.
    section._on_preview("не привязка")
    assert fake.previews == []


def test_section_on_preview_without_service(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # preview=None
    # Без сервиса вызов безопасно ничего не делает.
    section._on_preview(_tts(SoundStage.ON_START, "Раз"))
    for row in section.stages():
        assert not row._stop.isEnabled()


def test_section_stop_without_service(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # preview=None
    for row in section.stages():
        row.set_playing(True)
    # Стоп без сервиса всё равно снимает признак игры со всех строк.
    section._on_stop()
    assert all(not row._stop.isEnabled() for row in section.stages())


def test_section_safe_duration_without_service(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # preview=None
    assert section._safe_duration(_builtin(SoundStage.ON_START, "chime")) is None


# ----------------------------------------------------------------------
# SoundBindingRow: кнопка выбора файла
# ----------------------------------------------------------------------


def test_row_browse_hidden_until_file_source(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    # Видимость проверяем через isHidden(): верхнеуровневый виджет в тесте не показан.
    # По умолчанию «Без звука» — кнопки выбора файла нет.
    assert row._browse.isHidden()
    # Для «Файл» — появляется, для остальных источников — снова прячется.
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    assert not row._browse.isHidden()
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    assert row._browse.isHidden()


def test_row_browse_enabled_tracks_import_service(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    # Без сервиса импорта кнопка выключена (но видима, чтобы объяснить почему).
    assert not row._browse.isEnabled()
    row.set_import_enabled(True)
    assert row._browse.isEnabled()


def test_row_apply_imported_sets_value_and_status(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_ERROR)
    row.set_import_enabled(True)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    seen: list[int] = []
    row.changed.connect(lambda: seen.append(1))
    row.apply_imported("сигнал.wav")
    # Имя попало в значение, привязка собралась, статус показан.
    assert row._value.text() == "сигнал.wav"
    assert seen != []
    got = row.binding()
    assert got is not None
    assert got.source is SoundSource.FILE
    assert got.value == "сигнал.wav"
    assert not row._import_status.isHidden()
    assert "сигнал.wav" in row._import_status.text()


def test_row_import_error_shows_without_touching_value(
    app: QApplication, theme: ThemeManager
) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    row.set_import_error("Файл повреждён.")
    assert row._value.text() == ""
    assert not row._import_status.isHidden()
    assert row._import_status.text() == "Файл повреждён."


def test_row_source_change_clears_import_status(app: QApplication, theme: ThemeManager) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    row.set_import_error("Файл повреждён.")
    assert not row._import_status.isHidden()
    # Смена источника снимает устаревшее сообщение.
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.TTS))
    assert row._import_status.isHidden()
    assert row._import_status.text() == ""


def test_row_browse_emits_chosen_path(
    app: QApplication, theme: ThemeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row.set_import_enabled(True)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    seen: list[str] = []
    row.import_requested.connect(seen.append)
    # Диалог подменён: имитируем выбор файла.
    monkeypatch.setattr(
        "ayris.gui.widgets.sound_binding.QFileDialog.getOpenFileName",
        lambda *_a, **_k: ("C:/музыка/трек.mp3", "Звуковые файлы (*.wav *.mp3 *.ogg)"),
    )
    row._browse.click()
    assert seen == ["C:/музыка/трек.mp3"]


def test_row_browse_cancel_emits_nothing(
    app: QApplication, theme: ThemeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = SoundBindingRow(theme, SoundStage.ON_START)
    row.set_import_enabled(True)
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    seen: list[str] = []
    row.import_requested.connect(seen.append)
    # Пользователь закрыл диалог — путь пустой, сигнала нет.
    monkeypatch.setattr(
        "ayris.gui.widgets.sound_binding.QFileDialog.getOpenFileName",
        lambda *_a, **_k: ("", ""),
    )
    row._browse.click()
    assert seen == []


# ----------------------------------------------------------------------
# SoundBindingSection: импорт файла через фейк
# ----------------------------------------------------------------------


def test_section_without_importer_disables_browse(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # importer=None
    for row in section.stages():
        row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
        assert not row._browse.isEnabled()


def test_section_import_writes_file_into_model(app: QApplication, theme: ThemeManager) -> None:
    importer = FakeImporter(result="звук.wav")
    section = SoundBindingSection(theme, importer=importer)
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    # Импорт идёт в фоновом потоке — ждём его завершения через событийный цикл.
    row.import_requested.emit("C:/музыка/трек.mp3")
    assert _pump(app, lambda: bool(model.sounds))
    # Импортёр вызван с выбранным путём, результат лёг в модель как FILE-привязка.
    assert importer.imported == [Path("C:/музыка/трек.mp3")]
    assert len(model.sounds) == 1
    assert model.sounds[0].source is SoundSource.FILE
    assert model.sounds[0].value == "звук.wav"
    assert "звук.wav" in row._import_status.text()


def test_section_import_shows_busy_then_result(app: QApplication, theme: ThemeManager) -> None:
    # Импортёр блокируется, пока тест не разрешит, — так ловим промежуточное «занято».
    gate = threading.Event()

    class Blocking:
        def import_file(self, source: Path) -> str:
            gate.wait(2.0)
            return "готово.wav"

    section = SoundBindingSection(theme, importer=Blocking())
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    row.import_requested.emit("C:/музыка/трек.mp3")
    # Пока импорт идёт: кнопка заблокирована, статус «Импортирую…».
    assert not row._browse.isEnabled()
    assert row._import_status.text() == "Импортирую…"
    # Разрешаем импортёру закончить — кнопка снова активна, имя в модели.
    gate.set()
    assert _pump(app, lambda: bool(model.sounds))
    assert row._browse.isEnabled()
    assert model.sounds[0].value == "готово.wav"


def test_section_import_shows_duration(app: QApplication, theme: ThemeManager) -> None:
    importer = FakeImporter(result="звук.wav")
    preview = FakePreview(duration=1800)
    section = SoundBindingSection(theme, preview=preview, importer=importer)
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    row.import_requested.emit("C:/музыка/трек.mp3")
    # После импорта показана длительность свежесохранённого звука.
    assert _pump(app, lambda: row._duration.text() == "1.8 с")


def test_section_import_domain_error_is_shown(app: QApplication, theme: ThemeManager) -> None:
    importer = FakeImporter(fail=True)
    section = SoundBindingSection(theme, importer=importer)
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    row.import_requested.emit("C:/битый.mp3")
    # Понятное сообщение у строки, модель не тронута, кнопка снова активна.
    assert _pump(app, lambda: row._import_status.text() == "Файл повреждён.")
    assert model.sounds == []
    assert row._browse.isEnabled()


def test_section_import_unexpected_error_is_swallowed(
    app: QApplication, theme: ThemeManager
) -> None:
    importer = FakeImporter(crash=True)
    section = SoundBindingSection(theme, importer=importer)
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    # Неожиданная ошибка не всплывает наружу — только общий текст у строки.
    row.import_requested.emit("C:/трек.mp3")
    assert _pump(app, lambda: "Не удалось импортировать" in row._import_status.text())
    assert model.sounds == []


def test_section_import_without_service_is_silent(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme)  # importer=None
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    # Без сервиса вызов безопасно ничего не делает.
    row.import_requested.emit("C:/трек.mp3")
    assert model.sounds == []


# ----------------------------------------------------------------------
# _ImportRunner и чистые обработчики секции
# ----------------------------------------------------------------------


def test_import_runner_finished_carries_name(app: QApplication) -> None:
    runner = _ImportRunner()
    seen: list[str] = []
    runner.finished.connect(seen.append)
    # Прямой вызов _run синхронен — очередь событий не нужна.
    runner._run(lambda: "звук.wav")
    assert seen == ["звук.wav"]


def test_import_runner_reports_ayris_user_message(app: QApplication) -> None:
    runner = _ImportRunner()
    failures: list[str] = []
    runner.failed.connect(failures.append)

    def boom() -> str:
        raise SoundImportError("boom", user_message="Файл повреждён.")

    runner._run(boom)
    assert failures == ["Файл повреждён."]


def test_import_runner_hides_generic_exception(app: QApplication) -> None:
    runner = _ImportRunner()
    failures: list[str] = []
    runner.failed.connect(failures.append)

    def boom() -> str:
        raise ValueError("сырое исключение")

    runner._run(boom)
    # Наружу — только общий русский текст, без деталей исключения.
    assert failures == ["Не удалось импортировать звук. Проверьте файл."]


def test_import_runner_real_thread_delivers_via_queue(app: QApplication) -> None:
    runner = _ImportRunner()
    seen: list[str] = []
    runner.finished.connect(seen.append)
    runner.run(lambda: "готово.wav")  # запускает daemon-поток
    assert _pump(app, lambda: seen == ["готово.wav"])


def test_section_on_import_finished_lands_binding(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme, importer=FakeImporter())
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    # Чистый обработчик кладёт имя в модель и снимает признак импорта.
    section._on_import_finished(row, "финал.wav")
    assert model.sounds[0].value == "финал.wav"
    assert row._browse.isEnabled()


def test_section_on_import_failed_shows_message(app: QApplication, theme: ThemeManager) -> None:
    section = SoundBindingSection(theme, importer=FakeImporter())
    model = CommandModel(name="Тест")
    section.set_command(model)
    row = {r._stage: r for r in section.stages()}[SoundStage.ON_START]
    row._enabled.setCurrentIndex(row._enabled.findData(SoundSource.FILE))
    section._on_import_failed(row, "Файл повреждён.")
    assert row._import_status.text() == "Файл повреждён."
    assert model.sounds == []
    assert row._browse.isEnabled()
