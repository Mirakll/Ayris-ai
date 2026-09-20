"""Stage sounds of a command: what plays on start, on success, on error.

The three stages of :class:`~ayris.actions.macros.schema.SoundStage` each get one
optional binding, editing ``model.sounds`` in place. A binding names a source — a
sound from the built-in library, a phrase spoken by TTS, or an imported file — a
volume and whether the command waits for it to finish.

Playback is never done here. The widget calls a :class:`SoundPreview` the editor
injects, which routes to the same :class:`~ayris.actions.macros.sounds.library.SoundLibrary`
the macro engine uses; without one the preview button is disabled rather than the
widget reaching for an audio library of its own (the task's «никаких прямых вызовов
аудио-библиотек из виджета»). Resolving a file's duration is equally the service's
job, so a slow disk read never happens on the UI thread inside this widget.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.schema import CommandModel, SoundBinding, SoundSource, SoundStage
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox

__all__ = ["SoundBindingRow", "SoundBindingSection", "SoundPreview"]


class SoundPreview(Protocol):
    """The playback the editor lends the sound section — a facade over the library."""

    def preview_binding(self, binding: SoundBinding) -> object:
        """Play one binding and return a handle with ``cancel()``."""

    def stop(self) -> None:
        """Stop whatever is currently previewing."""

    def duration_ms(self, binding: SoundBinding) -> int | None:
        """The length of a resolved binding in ms, or ``None`` when unknown."""


_STAGE_LABELS: dict[SoundStage, str] = {
    SoundStage.ON_START: "При запуске",
    SoundStage.ON_SUCCESS: "При успехе",
    SoundStage.ON_ERROR: "При ошибке",
}

_SOURCE_LABELS: dict[SoundSource, str] = {
    SoundSource.BUILTIN: "Из библиотеки",
    SoundSource.TTS: "Синтез речи",
    SoundSource.FILE: "Файл",
}

_VALUE_PLACEHOLDERS: dict[SoundSource, str] = {
    SoundSource.BUILTIN: "Идентификатор звука, например builtin:chime",
    SoundSource.TTS: "Текст для озвучивания",
    SoundSource.FILE: "custom:имя-файла.wav",
}


class SoundBindingRow(QWidget):
    """The controls of one stage: source, value, volume, wait, and preview."""

    changed = Signal()
    preview_requested = Signal(object)
    stop_requested = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        stage: SoundStage,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._stage = stage
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_xs"))

        title = QLabel(_STAGE_LABELS[stage])
        title.setProperty("role", "muted")
        outer.addWidget(title)

        top = QHBoxLayout()
        top.setSpacing(theme.metric("spacing_sm"))
        self._enabled = ThemedComboBox()
        self._enabled.addItem("Без звука", None)
        for source, label in _SOURCE_LABELS.items():
            self._enabled.addItem(label, source)
        self._enabled.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self._enabled, 0)

        self._value = QLineEdit()
        self._value.textChanged.connect(self._on_changed)
        top.addWidget(self._value, 1)
        outer.addLayout(top)

        bottom = QHBoxLayout()
        bottom.setSpacing(theme.metric("spacing_sm"))
        self._volume = QSpinBox()
        self._volume.setRange(0, 100)
        self._volume.setValue(100)
        self._volume.setSuffix(" %")
        self._volume.setToolTip("Громкость воспроизведения.")
        self._volume.valueChanged.connect(self._on_changed)
        bottom.addWidget(self._volume)

        self._wait = ThemedComboBox()
        self._wait.addItem("Не ждать", False)
        self._wait.addItem("Ждать окончания", True)
        self._wait.currentIndexChanged.connect(self._on_changed)
        bottom.addWidget(self._wait)

        self._duration = QLabel("")
        self._duration.setProperty("role", "muted")
        bottom.addWidget(self._duration)
        bottom.addStretch(1)

        self._preview = QPushButton("Прослушать")
        self._preview.clicked.connect(self._on_preview)
        self._stop = QPushButton("Стоп")
        self._stop.clicked.connect(self.stop_requested)
        self._stop.setEnabled(False)
        bottom.addWidget(self._preview)
        bottom.addWidget(self._stop)
        outer.addLayout(bottom)

        self._can_preview = False
        self._apply_source(None)

    # -- public API ---------------------------------------------------------

    def set_binding(self, binding: SoundBinding | None) -> None:
        self._loading = True
        try:
            if binding is None:
                self._enabled.setCurrentIndex(0)
                self._value.clear()
                self._volume.setValue(100)
                self._wait.setCurrentIndex(0)
                self._apply_source(None)
            else:
                self._enabled.setCurrentIndex(self._enabled.findData(binding.source))
                self._value.setText(binding.value)
                self._volume.setValue(binding.volume if binding.volume is not None else 100)
                self._wait.setCurrentIndex(1 if binding.wait else 0)
                self._apply_source(binding.source)
        finally:
            self._loading = False

    def binding(self) -> SoundBinding | None:
        """The binding for this stage, or ``None`` when the stage has no sound."""
        source = self._enabled.currentData()
        value = self._value.text().strip()
        if source is None or not value:
            return None
        return SoundBinding(
            stage=self._stage,
            source=source,
            value=value,
            volume=self._volume.value(),
            wait=bool(self._wait.currentData()),
        )

    def set_preview_enabled(self, enabled: bool) -> None:
        self._can_preview = enabled
        self._refresh_preview_button()

    def set_playing(self, playing: bool) -> None:
        self._stop.setEnabled(playing)

    def set_duration(self, duration_ms: int | None) -> None:
        if duration_ms is None:
            self._duration.clear()
        else:
            self._duration.setText(f"{duration_ms / 1000:.1f} с")

    # -- editing ------------------------------------------------------------

    def _on_source_changed(self) -> None:
        self._apply_source(self._enabled.currentData())
        self._on_changed()

    def _apply_source(self, source: SoundSource | None) -> None:
        has_source = source is not None
        self._value.setEnabled(has_source)
        self._volume.setEnabled(has_source)
        self._wait.setEnabled(has_source)
        self._value.setPlaceholderText(
            _VALUE_PLACEHOLDERS[source] if source is not None else "Звук не задан"
        )
        self._refresh_preview_button()

    def _refresh_preview_button(self) -> None:
        self._preview.setEnabled(self._can_preview and self.binding() is not None)

    def _on_changed(self, *_args: object) -> None:
        self._refresh_preview_button()
        if not self._loading:
            self.changed.emit()

    def _on_preview(self) -> None:
        binding = self.binding()
        if binding is not None:
            self.preview_requested.emit(binding)


class SoundBindingSection(QWidget):
    """The three stage rows together, editing ``model.sounds`` as one list."""

    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        preview: SoundPreview | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._preview = preview
        self._model: CommandModel | None = None
        self._playing_handle: object | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_md"))

        self._rows: dict[SoundStage, SoundBindingRow] = {}
        for stage in (SoundStage.ON_START, SoundStage.ON_SUCCESS, SoundStage.ON_ERROR):
            row = SoundBindingRow(theme, stage)
            row.changed.connect(self._commit)
            row.preview_requested.connect(self._on_preview)
            row.stop_requested.connect(self._on_stop)
            row.set_preview_enabled(preview is not None)
            self._rows[stage] = row
            outer.addWidget(row)

    # -- public API ---------------------------------------------------------

    def set_command(self, model: CommandModel) -> None:
        self._model = model
        by_stage = {binding.stage: binding for binding in model.sounds}
        for stage, row in self._rows.items():
            binding = by_stage.get(stage)
            row.set_binding(binding)
            if binding is not None and self._preview is not None:
                row.set_duration(self._safe_duration(binding))

    def stages(self) -> Sequence[SoundBindingRow]:
        return tuple(self._rows.values())

    # -- editing ------------------------------------------------------------

    def _commit(self) -> None:
        if self._model is None:
            return
        bindings = [row.binding() for row in self._rows.values()]
        self._model.sounds = [binding for binding in bindings if binding is not None]
        self.changed.emit()

    def _on_preview(self, binding: object) -> None:
        if self._preview is None or not isinstance(binding, SoundBinding):
            return
        self._on_stop()
        try:
            self._playing_handle = self._preview.preview_binding(binding)
        except Exception:
            self._playing_handle = None
            return
        for row in self._rows.values():
            row.set_playing(row.binding() == binding)

    def _on_stop(self) -> None:
        if self._preview is not None:
            self._preview.stop()
        self._playing_handle = None
        for row in self._rows.values():
            row.set_playing(False)

    def _safe_duration(self, binding: SoundBinding) -> int | None:
        if self._preview is None:
            return None
        try:
            return self._preview.duration_ms(binding)
        except Exception:
            return None
