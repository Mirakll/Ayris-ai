"""Base settings page with validated, debounced two-way config binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from PySide6.QtCore import QObject, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.config import ConfigManager, Settings
from ayris.core.events import ConfigChanged, EventBus
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import ConfirmDialog, ToggleSwitch
from ayris.utils.logger import get_logger

__all__ = ["SearchEntry", "SettingsTab"]

_log = get_logger(__name__)
Getter = Callable[[], Any]
Setter = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class SearchEntry:
    path: str
    label: str
    section_key: str
    section_title: str
    widget: QWidget
    scroll_area: QScrollArea | None

    @property
    def search_text(self) -> str:
        return f"{self.path} {self.label} {self.section_title}".casefold()


@dataclass(slots=True)
class _Binding:
    path: str
    widget: QWidget
    getter: Getter
    setter: Setter


class _ConfigRelay(QObject):
    changed = Signal(object)


class SettingsTab(QWidget):
    """Shared page chrome and config binding used by tasks 48-59."""

    search_entries_changed = Signal()

    def __init__(
        self,
        section_key: str,
        title: str,
        config_paths: tuple[str, ...],
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        parent: QWidget | None = None,
        *,
        debounce_ms: int = 350,
    ) -> None:
        super().__init__(parent)
        self.section_key = section_key
        self.section_title = title
        self.config_paths = config_paths
        self._manager = manager
        self._theme = theme
        self._bindings: dict[str, _Binding] = {}
        self._search_entries: list[SearchEntry] = []
        self._pending: dict[str, Any] = {}
        self._programmatic = False
        self._loaded = False
        self._relay = _ConfigRelay(self)
        self._relay.changed.connect(self._on_config_changed)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(debounce_ms)
        self._save_timer.timeout.connect(self.flush_pending)

        self._layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setProperty("role", "h1")
        self.dirty_label = QLabel("● Не сохранено")
        self.dirty_label.setProperty("status", "warning")
        self.dirty_label.hide()
        self.reset_button = QPushButton("Сбросить секцию")
        self.reset_button.clicked.connect(self.confirm_reset)
        header.addWidget(self.title_label)
        header.addWidget(self.dirty_label)
        header.addStretch(1)
        header.addWidget(self.reset_button)
        self._layout.addLayout(header)
        self.body = QVBoxLayout()
        self._layout.addLayout(self.body, 1)
        self._layout.setContentsMargins(*(theme.metric("spacing_2xl"),) * 4)
        self._layout.setSpacing(theme.metric("spacing_lg"))

        if bus is not None:
            self._unsubscribe = bus.subscribe(ConfigChanged, self._receive_event)
        else:
            self._unsubscribe = manager.subscribe(self._receive_diff)
        self._subscribed = True

    @property
    def search_entries(self) -> tuple[SearchEntry, ...]:
        return tuple(self._search_entries)

    @property
    def is_dirty(self) -> bool:
        return bool(self._pending)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        if not self._loaded:
            self.load_from_config()
            self._loaded = True
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.dispose()
        super().closeEvent(event)

    def dispose(self) -> None:
        """Save pending values and detach long-lived config listeners."""
        self.flush_pending()
        if self._subscribed:
            self._unsubscribe()
            self._subscribed = False

    def load_from_config(self) -> None:
        self._programmatic = True
        try:
            for binding in self._bindings.values():
                value = _read_path(self._manager.settings, binding.path)
                blocker = QSignalBlocker(binding.widget)
                binding.setter(value)
                del blocker
        finally:
            self._programmatic = False

    def bind_toggle(self, widget: ToggleSwitch, path: str, label: str) -> None:
        self._bind(widget, path, label, widget.isChecked, widget.setChecked, widget.toggled)

    def bind_slider(self, widget: QSlider, path: str, label: str) -> None:
        self._bind(widget, path, label, widget.value, widget.setValue, widget.valueChanged)

    def bind_combo(self, widget: QComboBox, path: str, label: str) -> None:
        self._bind(
            widget,
            path,
            label,
            widget.currentData,
            _set_combo_value(widget),
            widget.currentIndexChanged,
        )

    def bind_line_edit(self, widget: QLineEdit, path: str, label: str) -> None:
        self._bind(widget, path, label, widget.text, widget.setText, widget.textChanged)

    def register_search_field(
        self, widget: QWidget, path: str, label: str, *, scroll_area: QScrollArea | None = None
    ) -> None:
        widget.setAccessibleName(label)
        self._search_entries.append(
            SearchEntry(path, label, self.section_key, self.section_title, widget, scroll_area)
        )
        self.search_entries_changed.emit()

    def flush_pending(self) -> None:
        if not self._pending:
            return
        pending = dict(self._pending)
        try:
            self._manager.apply(pending)
        except Exception:
            _log.exception("Не удалось сохранить настройки раздела %s", self.section_key)
            self.load_from_config()
            self._pending.clear()
            self.dirty_label.hide()
            return
        self._pending.clear()
        self.dirty_label.hide()

    def confirm_reset(self) -> None:
        dialog = ConfirmDialog(
            "Сбросить раздел?",
            f"Все настройки раздела «{self.section_title}» вернутся к значениям по умолчанию.",
            self._theme,
            confirm_text="Сбросить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.reset_to_defaults()

    def reset_to_defaults(self) -> None:
        self._save_timer.stop()
        defaults = Settings()
        values: dict[str, Any] = {}
        for prefix in self.config_paths:
            section = _read_path(defaults, prefix)
            if isinstance(section, BaseModel):
                for path, value in _leaves(section, prefix):
                    values[path] = value
        if values:
            self._manager.apply(values)
        self._pending.clear()
        self.dirty_label.hide()
        self.load_from_config()

    def _bind(
        self,
        widget: QWidget,
        path: str,
        label: str,
        getter: Getter,
        setter: Setter,
        signal: Any,
    ) -> None:
        if path in self._bindings:
            raise ValueError(f"Поле уже привязано: {path}")
        self._bindings[path] = _Binding(path, widget, getter, setter)
        self.register_search_field(widget, path, label)
        signal.connect(lambda *_args, bound_path=path: self._control_changed(bound_path))

    def _control_changed(self, path: str) -> None:
        if self._programmatic:
            return
        self._pending[path] = self._bindings[path].getter()
        self.dirty_label.show()
        self._save_timer.start()

    def _receive_event(self, event: ConfigChanged) -> None:
        self._relay.changed.emit(event.diff)

    def _receive_diff(self, diff: SettingsDiff) -> None:
        self._relay.changed.emit(diff)

    def _on_config_changed(self, diff: SettingsDiff) -> None:
        touched = set(diff.paths) & set(self._bindings)
        if not touched:
            return
        self._programmatic = True
        try:
            for path in touched:
                binding = self._bindings[path]
                blocker = QSignalBlocker(binding.widget)
                binding.setter(_read_path(diff.settings, path))
                del blocker
                self._pending.pop(path, None)
        finally:
            self._programmatic = False
        self.dirty_label.setVisible(bool(self._pending))


def _read_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _leaves(model: BaseModel, prefix: str) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for name in type(model).model_fields:
        value = getattr(model, name)
        path = f"{prefix}.{name}"
        if isinstance(value, BaseModel):
            result.extend(_leaves(value, path))
        else:
            result.append((path, value))
    return result


def _set_combo_value(widget: QComboBox) -> Setter:
    def setter(value: Any) -> None:
        index = widget.findData(value)
        if index < 0:
            index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)

    return setter
