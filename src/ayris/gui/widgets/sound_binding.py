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

Choosing a file works the same way. A «Файл» source shows a «Выбрать файл…» button
that opens the native file dialog for ``.wav``, ``.mp3`` and ``.ogg``; the chosen
path is handed to a :class:`SoundImporter` the editor injects, which decodes and
copies it into the profile's own sounds folder (a portable mono WAV) and returns
the stored filename. That name — never the original absolute path — is what the
binding keeps, so an exported ``.ayris`` still travels with its sounds.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.schema import CommandModel, SoundBinding, SoundSource, SoundStage
from ayris.core.errors import AyrisError
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.spin_box import ThemedSpinBox
from ayris.gui.widgets.toggle import ToggleSwitch
from ayris.utils.logger import get_logger

__all__ = ["SoundBindingRow", "SoundBindingSection", "SoundImporter", "SoundPreview"]

_log = get_logger(__name__)

#: The dialog filter for picking a sound to import. Only the containers
#: :data:`~ayris.actions.macros.schema.SOUND_EXTENSIONS` allows, plus an «all files»
#: escape hatch, since the importer refuses anything it cannot decode anyway.
_SOUND_FILE_FILTER = "Звуковые файлы (*.wav *.mp3 *.ogg);;Все файлы (*)"


class SoundPreview(Protocol):
    """The playback the editor lends the sound section — a facade over the library."""

    def preview_binding(self, binding: SoundBinding) -> object:
        """Play one binding and return a handle with ``cancel()``."""

    def stop(self) -> None:
        """Stop whatever is currently previewing."""

    def duration_ms(self, binding: SoundBinding) -> int | None:
        """The length of a resolved binding in ms, or ``None`` when unknown."""


class SoundImporter(Protocol):
    """Brings a file the user picked into the profile's sounds folder.

    The editor injects one over :func:`~ayris.actions.macros.sounds.importer.import_sound`;
    without it the «Выбрать файл…» button is disabled rather than the widget copying
    files itself. ``import_file`` returns the stored filename (``work_mode_start.wav``),
    which is what a :attr:`~ayris.actions.macros.schema.SoundSource.FILE` binding keeps.
    """

    def import_file(self, source: Path) -> str:
        """Decode and store one file, returning its portable name in the folder."""


class _ImportRunner(QObject):
    """Runs one blocking import off the UI thread, delivering the result by signal.

    Decoding and resampling a long track is slow enough to freeze the window if done
    inline, so :class:`SoundBindingSection` hands the work here and reacts to the
    queued ``finished``/``failed`` signal. Mirrors ``macro_editor._AsyncRunner`` rather
    than importing it, to keep this widget free of an editor dependency.
    """

    finished = Signal(str)
    failed = Signal(str)

    def run(self, work: Callable[[], str]) -> None:
        threading.Thread(target=self._run, args=(work,), daemon=True).start()

    def _run(self, work: Callable[[], str]) -> None:
        try:
            result = work()
        except AyrisError as exc:
            self.failed.emit(exc.user_message)
        except Exception:
            _log.exception("не удалось импортировать звук")
            self.failed.emit("Не удалось импортировать звук. Проверьте файл.")
        else:
            self.finished.emit(result)


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
    import_requested = Signal(str)

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

        # Only shown for the «Файл» source: pick a sound off disk instead of typing
        # its name. Hidden for the other sources so the row stays a single line.
        self._browse = QPushButton("Выбрать файл…")
        self._browse.setToolTip("Выбрать файл WAV, MP3 или OGG на компьютере.")
        self._browse.clicked.connect(self._on_browse)
        self._browse.hide()
        top.addWidget(self._browse, 0)
        outer.addLayout(top)

        bottom = QHBoxLayout()
        bottom.setSpacing(theme.metric("spacing_sm"))
        self._volume = ThemedSpinBox()
        self._volume.setRange(0, 100)
        self._volume.setValue(100)
        self._volume.setSuffix(" %")
        self._volume.setToolTip("Громкость воспроизведения.")
        self._volume.valueChanged.connect(self._on_changed)
        bottom.addWidget(self._volume)

        self._wait_label = QLabel("Ждать окончания")
        self._wait_label.setProperty("role", "muted")
        self._wait = ToggleSwitch(theme, label="Ждать окончания звука")
        self._wait.setToolTip("Команда продолжит следующие действия только после конца звука.")
        self._wait.toggled.connect(self._on_changed)
        bottom.addWidget(self._wait_label)
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

        # Result of the last import — a confirmation or a Russian error. Kept out of
        # the way (empty and hidden) until there is something to say.
        self._import_status = QLabel("")
        self._import_status.setProperty("role", "muted")
        self._import_status.setWordWrap(True)
        self._import_status.hide()
        outer.addWidget(self._import_status)

        self._can_preview = False
        self._can_import = False
        self._importing = False
        self._apply_source(None)

    # -- public API ---------------------------------------------------------

    def set_binding(self, binding: SoundBinding | None) -> None:
        self._loading = True
        try:
            if binding is None:
                self._enabled.setCurrentIndex(0)
                self._value.clear()
                self._volume.setValue(100)
                self._wait.setChecked(False)
                self._apply_source(None)
            else:
                self._enabled.setCurrentIndex(self._enabled.findData(binding.source))
                self._value.setText(binding.value)
                self._volume.setValue(binding.volume if binding.volume is not None else 100)
                self._wait.setChecked(bool(binding.wait))
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
            wait=self._wait.isChecked(),
        )

    def set_preview_enabled(self, enabled: bool) -> None:
        self._can_preview = enabled
        self._refresh_preview_button()

    def set_import_enabled(self, enabled: bool) -> None:
        self._can_import = enabled
        self._refresh_import_button()

    def set_importing(self, importing: bool) -> None:
        """Show the copy is in progress and lock the picker while it runs."""
        # The decode/resample runs off the UI thread; the button must not launch a
        # second import over the first, and the label tells the user why it waits.
        self._importing = importing
        self._refresh_import_button()
        if importing:
            self._set_import_status("Импортирую…")

    def apply_imported(self, filename: str) -> None:
        """Put an imported file's name into the value and confirm it."""
        # Setting the text runs _on_changed, which commits the new binding upward;
        # the import status label reports the copy the user cannot otherwise see.
        self._importing = False
        self._value.setText(filename)
        self._set_import_status(f"Звук «{filename}» добавлен.")
        self._refresh_import_button()

    def set_import_error(self, message: str) -> None:
        self._importing = False
        self._set_import_status(message)
        self._refresh_import_button()

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
        self._wait_label.setEnabled(has_source)
        self._value.setPlaceholderText(
            _VALUE_PLACEHOLDERS[source] if source is not None else "Звук не задан"
        )
        # The picker belongs to the «Файл» source only, and a stale «added» note from
        # a previous file has no meaning once the source changes. Compare with ``==``,
        # not ``is``: ThemedComboBox hands back the StrEnum's string value on a source
        # change, while set_binding passes the enum member — both equal SoundSource.FILE.
        self._browse.setVisible(source == SoundSource.FILE)
        self._set_import_status("")
        self._refresh_preview_button()
        self._refresh_import_button()

    def _refresh_preview_button(self) -> None:
        self._preview.setEnabled(self._can_preview and self.binding() is not None)

    def _refresh_import_button(self) -> None:
        # Disabled while an import is running so a second file cannot start over it.
        self._browse.setEnabled(self._can_import and not self._importing)
        if self._importing:
            self._browse.setToolTip("Идёт импорт звука…")
        elif self._can_import:
            self._browse.setToolTip("Выбрать файл WAV, MP3 или OGG на компьютере.")
        else:
            self._browse.setToolTip(
                "Импорт звуков недоступен: не удалось открыть папку звуков профиля."
            )

    def _set_import_status(self, message: str) -> None:
        self._import_status.setText(message)
        self._import_status.setVisible(bool(message))

    def _on_changed(self, *_args: object) -> None:
        self._refresh_preview_button()
        if not self._loading:
            self.changed.emit()

    def _on_preview(self) -> None:
        binding = self.binding()
        if binding is not None:
            self.preview_requested.emit(binding)

    def _on_browse(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            "Выберите звуковой файл",
            "",
            _SOUND_FILE_FILTER,
        )
        if chosen:
            self.import_requested.emit(chosen)


class SoundBindingSection(QWidget):
    """The three stage rows together, editing ``model.sounds`` as one list."""

    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        preview: SoundPreview | None = None,
        importer: SoundImporter | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._preview = preview
        self._importer = importer
        self._model: CommandModel | None = None
        self._playing_handle: object | None = None
        # Runners are kept alive here until their signal fires; a daemon thread whose
        # QObject was collected mid-flight would drop the queued result.
        self._import_runners: set[_ImportRunner] = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_md"))

        self._rows: dict[SoundStage, SoundBindingRow] = {}
        for stage in (SoundStage.ON_START, SoundStage.ON_SUCCESS, SoundStage.ON_ERROR):
            row = SoundBindingRow(theme, stage)
            row.changed.connect(self._commit)
            row.preview_requested.connect(self._on_preview)
            row.stop_requested.connect(self._on_stop)
            # Bind the row to the handler so a shared slot knows which one asked.
            row.import_requested.connect(lambda path, r=row: self._on_import(r, path))
            row.set_preview_enabled(preview is not None)
            row.set_import_enabled(importer is not None)
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

    def _on_import(self, row: SoundBindingRow, path: str) -> None:
        # Decoding a full track is slow, so it runs on a daemon thread; the row shows
        # «Импортирую…» and locks its button meanwhile, and the finished/failed slots
        # (pure, and unit-tested directly) land the result back on the UI thread.
        if self._importer is None:
            return
        importer = self._importer
        source = Path(path)
        row.set_importing(True)
        runner = _ImportRunner(self)
        self._import_runners.add(runner)
        runner.finished.connect(
            lambda name, r=row, run=runner: self._on_import_finished(r, name, run)
        )
        runner.failed.connect(
            lambda message, r=row, run=runner: self._on_import_failed(r, message, run)
        )
        runner.run(lambda: importer.import_file(source))

    def _on_import_finished(
        self, row: SoundBindingRow, filename: str, runner: _ImportRunner | None = None
    ) -> None:
        if runner is not None:
            self._import_runners.discard(runner)
        # apply_imported writes the name, which _commit turns into the FILE binding;
        # then show the duration of the freshly stored sound like set_command does.
        row.apply_imported(filename)
        if self._preview is not None:
            binding = row.binding()
            if binding is not None:
                row.set_duration(self._safe_duration(binding))

    def _on_import_failed(
        self, row: SoundBindingRow, message: str, runner: _ImportRunner | None = None
    ) -> None:
        if runner is not None:
            self._import_runners.discard(runner)
        row.set_import_error(message)

    def _safe_duration(self, binding: SoundBinding) -> int | None:
        if self._preview is None:
            return None
        try:
            return self._preview.duration_ms(binding)
        except Exception:
            return None
