"""Выпадающий выбор модели: список доступных + пункт «добавить свою».

Заменяет на вкладке «ИИ» пару «текстовое поле + комбобокс "модели с сервера"»
одним контролом. Список моделей приходит с сервера или из недавних выборов,
текущая модель всегда присутствует (чтобы значение переживало round-trip через
конфиг), а последним пунктом идёт сентинел «➕ Добавить свою модель…»: выбор его
не меняет модель, а просит владельца показать поле ввода — так задаётся облачная
модель, которой ещё нет в списке.

Открытие списка (:meth:`showPopup`) поднимает сигнал :attr:`list_requested` —
владелец на вкладке подтягивает модели в фоне и зовёт :meth:`set_models`. Пока
список пуст, в нём всё равно есть текущая модель и «добавить свою», так что
контрол не бывает пустым и до первой загрузки. Когда фон отвечает уже при
открытом списке, :meth:`reload_open` пересобирает его геометрию, не запрашивая
модели заново — иначе повторный :attr:`list_requested` зациклил бы загрузку.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QWidget

from ayris.gui.widgets.combo_box import ThemedComboBox

__all__ = ["ModelPicker"]

#: Служебное значение пункта «добавить свою модель» — заведомо не имя модели.
_ADD_NEW = "\x00add-new"


class ModelPicker(ThemedComboBox):
    """Комбобокс выбора модели с пунктом «добавить свою» внизу списка."""

    #: Пользователь выбрал готовую модель из списка (имя модели).
    model_chosen = Signal(str)
    #: Пользователь выбрал «➕ Добавить свою модель…» — пора показать поле ввода.
    add_new_requested = Signal()
    #: Список открыли — самое время подтянуть модели с сервера/диска.
    list_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current = ""
        self._models: tuple[str, ...] = ()
        self._loading = False
        self._suppress_request = False
        self.activated.connect(self._on_activated)
        self._rebuild()

    # --- публичный API -----------------------------------------------------

    def current_model(self) -> str:
        """Отмеченная сейчас модель (может быть пустой строкой)."""
        return self._current

    def models(self) -> tuple[str, ...]:
        """Модели, показанные в списке помимо текущей и «добавить свою»."""
        return self._models

    def set_current_model(self, model: str) -> None:
        """Отметить выбранную модель; добавит её в список, если её там нет."""
        cleaned = model.strip()
        if cleaned == self._current:
            return
        self._current = cleaned
        self._rebuild()

    def set_models(self, models: Sequence[str], *, loading: bool = False) -> None:
        """Заполнить список моделями (порядок стабилен, дубликаты убраны)."""
        seen: dict[str, None] = {}
        for name in models:
            cleaned = name.strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        self._models = tuple(seen)
        self._loading = loading
        self._rebuild()

    def set_loading(self, loading: bool) -> None:
        """Показать/снять пометку «загрузка списка…» без смены самого списка."""
        if self._loading == loading:
            return
        self._loading = loading
        self._rebuild()

    def reload_open(self) -> None:
        """Пересобрать геометрию открытого списка, не запрашивая модели заново.

        Qt считает размер выпадашки при показе; если модели пришли, когда список
        уже открыт, новые пункты влезают только под прокрутку. Переоткрытие даёт
        правильную высоту, а флаг :attr:`_suppress_request` гасит повторный
        :attr:`list_requested`, иначе переоткрытие снова бы дёрнуло загрузку.
        """
        view = self.view()
        if view is None or not view.isVisible():
            return
        self._suppress_request = True
        try:
            self.hidePopup()
            self.showPopup()
        finally:
            self._suppress_request = False

    # --- построение списка -------------------------------------------------

    def _rebuild(self) -> None:
        with QSignalBlocker(self):
            self.clear()
            names = list(self._models)
            if self._current and self._current not in names:
                names.insert(0, self._current)
            for name in names:
                self.addItem(name, name)
            if not names:
                placeholder = (
                    "Загрузка списка моделей…" if self._loading else "Список появится при открытии"
                )
                self.addItem(placeholder, "")
            self.insertSeparator(self.count())
            self.addItem("➕ Добавить свою модель…", _ADD_NEW)
            index = self.findData(self._current) if self._current else -1
            self.setCurrentIndex(index if index >= 0 else 0)

    # --- события -----------------------------------------------------------

    def showPopup(self) -> None:  # noqa: N802
        if not self._suppress_request:
            self.list_requested.emit()
        super().showPopup()

    def _on_activated(self, index: int) -> None:
        data = self.itemData(index)
        if data == _ADD_NEW:
            # Не даём сентинелу стать «выбранной» моделью — возвращаем текущую.
            self._restore_current()
            self.add_new_requested.emit()
            return
        if not data:  # плейсхолдер «список появится…» / «загрузка…»
            self._restore_current()
            return
        model = str(data)
        if model != self._current:
            self._current = model
            self.model_chosen.emit(model)

    def _restore_current(self) -> None:
        index = self.findData(self._current) if self._current else 0
        with QSignalBlocker(self):
            self.setCurrentIndex(index if index >= 0 else 0)
