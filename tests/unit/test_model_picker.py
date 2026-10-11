"""Пикер модели (:class:`ModelPicker`), offscreen.

Проверяем контракт виджета, а не пиксели: список всегда несёт «➕ Добавить свою
модель…», текущая модель приколота даже когда сервер её не вернул, дубликаты
схлопываются с сохранением порядка, выбор готовой модели эмитит ``model_chosen``,
сентинел — ``add_new_requested`` (и не становится «выбранной»), открытие списка
поднимает ``list_requested``, а ``reload_open`` при закрытом списке — тихий no-op.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from ayris.gui.widgets.model_picker import ModelPicker

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


def _add_new_index(picker: ModelPicker) -> int:
    """Индекс пункта «добавить свою» — он всегда последний."""
    return picker.count() - 1


def test_empty_picker_still_offers_add_new(app: QApplication) -> None:
    picker = ModelPicker()
    assert picker.current_model() == ""
    # Плейсхолдер, разделитель и «добавить свою».
    assert picker.itemData(_add_new_index(picker)) == "\x00add-new"
    picker.close()


def test_current_model_is_pinned_even_without_a_server_list(app: QApplication) -> None:
    picker = ModelPicker()
    picker.set_current_model("gpt-4o")
    assert picker.current_model() == "gpt-4o"
    assert picker.findData("gpt-4o") >= 0
    picker.close()


def test_set_models_dedupes_and_keeps_order(app: QApplication) -> None:
    picker = ModelPicker()
    picker.set_models(["b", "a", "b", " ", "a"])
    assert picker.models() == ("b", "a")
    picker.close()


def test_current_not_in_list_is_prepended(app: QApplication) -> None:
    picker = ModelPicker()
    picker.set_models(["a", "b"])
    picker.set_current_model("custom-model")
    assert picker.findData("custom-model") == 0
    picker.close()


def test_choosing_a_model_emits_model_chosen(app: QApplication) -> None:
    picker = ModelPicker()
    picker.set_models(["a", "b"])
    chosen: list[str] = []
    picker.model_chosen.connect(chosen.append)
    picker._on_activated(picker.findData("b"))
    assert chosen == ["b"]
    assert picker.current_model() == "b"
    picker.close()


def test_add_new_sentinel_emits_request_without_changing_model(app: QApplication) -> None:
    picker = ModelPicker()
    picker.set_models(["a"])
    picker.set_current_model("a")
    requested: list[bool] = []
    chosen: list[str] = []
    picker.add_new_requested.connect(lambda: requested.append(True))
    picker.model_chosen.connect(chosen.append)
    picker._on_activated(_add_new_index(picker))
    assert requested == [True]
    assert chosen == []  # сентинел не считается выбором модели
    assert picker.current_model() == "a"
    picker.close()


def test_show_popup_requests_the_list(app: QApplication) -> None:
    picker = ModelPicker()
    requests: list[bool] = []
    picker.list_requested.connect(lambda: requests.append(True))
    try:
        picker.showPopup()
        assert requests == [True]
    finally:
        picker.hidePopup()
    picker.close()


def test_reload_open_is_a_noop_when_closed(app: QApplication) -> None:
    picker = ModelPicker()
    requests: list[bool] = []
    picker.list_requested.connect(lambda: requests.append(True))
    picker.reload_open()  # список закрыт — ничего не делает и не запрашивает
    assert requests == []
    picker.close()
