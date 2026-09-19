"""Tab «Голос»: STT, TTS, wake word and the microphone, in one page.

The tab is the section chrome and the shared plumbing; the four parts of the
speech chain live in :mod:`ayris.gui.tabs.voice_sections`. Each section builds
its own cards through the helpers here, so the binding, the restart plaques and
the config round-trip all go through :class:`~ayris.gui.tabs.base.SettingsTab`
exactly once — there is no second settings-writer hiding in a section.

Three things are not the usual bind-and-forget:

*Float fields ride an int slider.* Speed, gain and the VAD threshold are floats,
but :class:`~ayris.gui.widgets.SliderField` is integer. :meth:`bind_scaled_slider`
binds one with a getter and setter that scale, so the config still sees the float
and the widget still shows a whole number.

*Some fields need a worker restart.* Changing the engine, the model or the audio
device cannot happen under a running worker, so those fields are grouped under a
per-section :class:`_RestartBar` that appears from :attr:`ConfigManager.pending_restarts`
and recycles exactly that worker through the supervisor.

*API keys never touch the config.* ``config.toml`` holds only ``credential_ref``;
the key itself goes to the Windows Credential Manager through
:class:`~ayris.core.secrets.SecretsStore`. Nothing in this tab writes a key to a
bound field, a log line or the settings file.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_args

from pydantic import BaseModel
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.config import ConfigManager, RestartScope
from ayris.core.errors import AyrisError
from ayris.core.events import EventBus
from ayris.core.secrets import SecretsStore, get_secrets
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import (
    SettingCard,
    SliderField,
    active_worker_control,
)
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.audio.calibration import AudioSource
    from ayris.audio.devices import DeviceEnumerator
    from ayris.models.catalog import ModelCatalog

__all__ = ["AsyncRunner", "VoiceServices", "VoiceTab", "combo_options"]

_log = get_logger(__name__)


def combo_options(
    model: type[BaseModel], field_name: str, labels: Mapping[Any, str]
) -> list[tuple[Any, str]]:
    """Combo entries ``(value, label)`` for a ``Literal`` field, straight from the schema.

    The value keeps its schema type — a string for an engine, an int for a sample
    rate — so it round-trips through :meth:`QComboBox.currentData` back into the
    config unchanged. An unlabelled value falls back to its own text.
    """
    annotation = model.model_fields[field_name].annotation
    return [(value, labels.get(value, str(value))) for value in get_args(annotation)]


#: Callable that voices a short test phrase with the current settings.
SpeakSample = Callable[[], None]
#: Callable that records one phrase and returns the recognised text and timing.
TranscribeOnce = Callable[[], "tuple[str, int]"]


@dataclass(slots=True)
class VoiceServices:
    """Everything the tab needs from the rest of Ayris, all optional.

    Injected so the page is testable with fakes and works before the voice
    pipeline is wired: a missing service disables its button rather than failing.
    """

    secrets: SecretsStore = field(default_factory=get_secrets)
    devices: DeviceEnumerator | None = None
    catalog: ModelCatalog | None = None
    #: Install names present on disk per model kind (``"stt"`` → ``{"gigaam-v3-ctc"}``).
    installed: Callable[[str], frozenset[str]] | None = None
    #: Builds the audio source calibration records from; ``None`` disables «Калибровать».
    audio_source: Callable[[], AudioSource] | None = None
    #: Runs the recognition self-test; ``None`` disables «Тест распознавания».
    transcribe_once: TranscribeOnce | None = None
    #: Voices the TTS sample; ``None`` disables «Прослушать».
    speak_sample: SpeakSample | None = None
    #: Opens the model manager (task 50); ``None`` leaves the link inert.
    open_model_manager: Callable[[], None] | None = None


class AsyncRunner(QObject):
    """Runs a blocking call off the UI thread and reports the result back on it.

    The tests in this tab («прослушать», «тест распознавания», «калибровать»)
    must never freeze the interface, so each runs through here. ``finished`` and
    ``failed`` are emitted from a worker thread; Qt marshals them to the thread
    that owns this object — the GUI thread — as queued signals.
    """

    finished = Signal(object)
    failed = Signal(str)

    def run(self, work: Callable[[], object]) -> None:
        """Start ``work`` on a daemon thread; the result comes back as a signal."""
        threading.Thread(target=self._run, args=(work,), daemon=True).start()

    def _run(self, work: Callable[[], object]) -> None:
        try:
            result = work()
        except AyrisError as exc:
            self.failed.emit(exc.user_message)
        except Exception as exc:  # a service may raise anything; keep the UI alive
            _log.exception("фоновая операция вкладки «Голос» упала")
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class _RestartBar(QFrame):
    """The «требуется перезапуск» plaque shown once per section that needs one."""

    restart_requested = Signal()

    def __init__(self, scope_hint: str, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("notice", True)
        self.setProperty("status", "warning")
        self._layout = QHBoxLayout(self)
        self.label = QLabel("Изменение применится после перезапуска воркера")
        self.label.setWordWrap(True)
        self.label.setToolTip(scope_hint)
        self._layout.addWidget(self.label, 1)
        self.button = QPushButton("Перезапустить")
        self.button.clicked.connect(self.restart_requested)
        self._layout.addWidget(self.button)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self.hide()

    def set_pending(self, pending: bool, *, has_control: bool) -> None:
        self.button.setEnabled(has_control)
        self.button.setToolTip("" if has_control else "Воркеры сейчас не запущены")
        self.setVisible(pending)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_sm")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(pad)
        self.setMinimumHeight(self._theme.metric("notice_min_height"))


class VoiceTab(SettingsTab):
    """The «Голос» settings page (task 49)."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        services: VoiceServices | None = None,
    ) -> None:
        super().__init__("voice", "Голос", ("voice",), manager, theme, bus)
        self.services = services if services is not None else VoiceServices()
        self.event_bus = bus
        self._restart_bars: dict[RestartScope, _RestartBar] = {}
        self._refreshers: list[Callable[[], None]] = []
        self._teardown: list[Callable[[], None]] = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # The page scrolls vertically only: a long combo label must elide inside
        # its card, never widen the page into a horizontal scrollbar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self.content = QVBoxLayout(container)
        self.content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        # Imported here, not at module scope: the section modules import this
        # class for typing, so importing them at the top would be circular.
        from ayris.gui.tabs.voice_sections import (
            AudioInputSection,
            SttSection,
            TtsSection,
            WakeWordSection,
        )

        self._sections = (
            SttSection(self),
            TtsSection(self),
            WakeWordSection(self),
            AudioInputSection(self),
        )
        self.content.addStretch(1)

    # -- section-facing helpers --------------------------------------------

    @property
    def manager(self) -> ConfigManager:
        """The config manager; sections apply structured values through it."""
        return self._manager

    @property
    def theme(self) -> ThemeManager:
        """The active theme, for section widgets that paint themselves."""
        return self._theme

    def add_header(self, text: str) -> QLabel:
        header = QLabel(text)
        header.setProperty("role", "h2")
        self.content.addWidget(header)
        return header

    def add_caption(self, text: str) -> QLabel:
        caption = QLabel(text)
        caption.setProperty("role", "muted")
        caption.setWordWrap(True)
        self.content.addWidget(caption)
        return caption

    def add_card(self, title: str, description: str, control: QWidget) -> SettingCard:
        if isinstance(control, QComboBox):
            self.tame_combo(control)
        card = SettingCard(title, description, control, self._theme)
        self.content.addWidget(card)
        return card

    def add_block(self, title: str, description: str, body: QWidget) -> QFrame:
        """A full-width card: title and description stacked *above* the body.

        :class:`SettingCard` puts the control to the right of the text, which
        doubles the minimum width for a body that is itself a row of buttons or a
        list. Stacking keeps a rich panel — the key field, the wake-word list, the
        level meter — inside the page instead of forcing a horizontal scrollbar.
        """
        frame = QFrame()
        frame.setProperty("card", True)
        frame.setAccessibleName(title)
        pad = self._theme.metric("spacing_lg")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        heading = QLabel(title)
        heading.setProperty("role", "h2")
        layout.addWidget(heading)
        if description:
            caption = QLabel(description)
            caption.setProperty("role", "secondary")
            caption.setWordWrap(True)
            layout.addWidget(caption)
        layout.addWidget(body)
        self.content.addWidget(frame)
        return frame

    def tame_combo(self, combo: QComboBox) -> None:
        """Stop a long item from widening the page: size to a fixed length, elide.

        Without this a combo's width tracks its longest item, so «Google Cloud
        Speech-to-Text» or «Авто: облако с откатом на офлайн» pushes the card past
        the viewport and a horizontal scrollbar appears. Sizing to a fixed
        character count and letting the closed combo elide keeps every card inside
        the page.
        """
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def add_widget(self, widget: QWidget) -> None:
        self.content.addWidget(widget)

    def panel(self) -> QWidget:
        """A layout-only container that stays transparent over its card.

        A bare ``QWidget`` inherits the window fill from the stylesheet and paints
        a dark rectangle on the card; the ``transparent`` property opts it out.
        """
        container = QWidget()
        container.setProperty("transparent", True)
        return container

    def add_restart_bar(self, scope: RestartScope) -> _RestartBar:
        bar = _RestartBar(scope.label, self._theme)
        bar.restart_requested.connect(lambda: self._restart_scope(scope))
        self._restart_bars[scope] = bar
        self.content.addWidget(bar)
        return bar

    def bind_int_slider(self, field_widget: SliderField, path: str, label: str) -> None:
        self._bind(
            field_widget,
            path,
            label,
            field_widget.value,
            field_widget.setValue,
            field_widget.value_changed,
        )

    def bind_scaled_slider(
        self, field_widget: SliderField, path: str, label: str, *, factor: float
    ) -> None:
        """Bind an int slider to a float config field, scaling both ways."""
        self._bind(
            field_widget,
            path,
            label,
            lambda: round(field_widget.value() / factor, 3),
            lambda value: field_widget.setValue(round(float(value) * factor)),
            field_widget.value_changed,
        )

    def register_refresh(self, refresher: Callable[[], None]) -> None:
        """Call ``refresher`` whenever settings change externally or on load."""
        self._refreshers.append(refresher)

    def add_teardown(self, callback: Callable[[], None]) -> None:
        """Register cleanup — a bus unsubscribe, a timer stop — run on dispose."""
        self._teardown.append(callback)

    def dispose(self) -> None:
        for callback in self._teardown:
            try:
                callback()
            except Exception:
                _log.exception("сбой очистки вкладки «Голос»")
        self._teardown.clear()
        super().dispose()

    def model_catalog(self) -> ModelCatalog:
        """The download catalog, loaded once and cached; empty if it cannot be read."""
        if self.services.catalog is None:
            from ayris.models.catalog import ModelCatalog, load_catalog

            try:
                self.services.catalog = load_catalog()
            except Exception:
                _log.exception("не удалось загрузить каталог моделей")
                self.services.catalog = ModelCatalog(entries=())
        return self.services.catalog

    def installed_models(self, kind: str) -> frozenset[str]:
        """Install names present on disk for a model kind, or empty if unknown."""
        if self.services.installed is None:
            return frozenset()
        try:
            return self.services.installed(kind)
        except Exception:
            _log.exception("не удалось получить список установленных моделей %s", kind)
            return frozenset()

    # -- lifecycle ----------------------------------------------------------

    def load_from_config(self) -> None:
        super().load_from_config()
        self._refresh_dynamic()

    def _restart_scope(self, scope: RestartScope) -> None:
        control = active_worker_control()
        if control is None:
            return
        try:
            control.restart_scope(scope, "перезапуск из настроек")
        except Exception:
            _log.exception("не удалось перезапустить воркеры области %s", scope.value)
            return
        self._manager.acknowledge_restart(scope)
        self._refresh_dynamic()

    def _refresh_dynamic(self) -> None:
        pending = self._manager.pending_restarts
        has_control = active_worker_control() is not None
        for scope, bar in self._restart_bars.items():
            bar.set_pending(scope in pending, has_control=has_control)
        for refresher in self._refreshers:
            try:
                refresher()
            except Exception:
                _log.exception("сбой обновления секции вкладки «Голос»")

    def _on_config_changed(self, diff: SettingsDiff) -> None:
        super()._on_config_changed(diff)
        self._refresh_dynamic()


register_tab("voice", VoiceTab)
