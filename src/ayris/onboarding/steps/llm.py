"""Шаг «Языковая модель»: выбрать и скачать локальную LLM для встроенного llama.cpp.

Необязательный, пропускаемый шаг. Базовый сценарий (активация → команда → ответ)
работает без языковой модели, поэтому по умолчанию выбран вариант «Позже»: мастер
не тянет несколько гигабайт, если пользователь этого не просил.

Выбранная модель качается тем же бэкендом менеджера моделей, что и шаг «Модели»:
его :class:`~ayris.gui.tabs.updates.DownloadCoordinator` крутит загрузку в фоновом
потоке и продолжает её даже после закрытия мастера. Поэтому «Далее» не ждёт конца
загрузки — прогрев модели произойдёт на следующем старте Айрис.

:meth:`apply` переключает провайдера на встроенный ``llamacpp`` и записывает имя
модели; путь к файлу (``ai.model_path``) пишется по факту завершения загрузки —
здесь, если мастер ещё открыт, либо обработчиком активной модели в приложении,
когда скачанная модель становится активной (``RestartScope.LLM`` перезапустит
воркер с новым файлом).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigManager
from ayris.core.errors import AyrisError
from ayris.core.events import (
    EventBus,
    ModelDownloadFailed,
    ModelDownloadFinished,
    ModelDownloadProgress,
)
from ayris.gui.theme import ThemeManager
from ayris.nlu.llm.catalog import (
    RECOMMENDED,
    LlmModelSpec,
    detect_total_ram_mb,
    verdict_for,
    verdict_label,
)
from ayris.nlu.llm.factory import LLAMACPP_PROVIDER
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.models import ModelRecord
    from ayris.gui.widgets.model_manager import ModelManagerBackend

__all__ = ["LlmStep", "downloadable_specs"]

_log = get_logger(__name__)


def downloadable_specs() -> tuple[LlmModelSpec, ...]:
    """Рекомендованные модели, которые Айрис умеет скачать для llama.cpp.

    Только те, для которых поставляется GGUF (``gguf_catalog_id``) и которые
    заявлены как исполнимые движком ``llamacpp``.
    """
    return tuple(
        spec for spec in RECOMMENDED if spec.gguf_catalog_id and spec.runs_on(LLAMACPP_PROVIDER)
    )


class _LlmRelay(QObject):
    """Переносит события загрузки с фонового потока на поток GUI."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(object)


class LlmStep(WizardStep):
    """Выбор и фоновая загрузка локальной языковой модели (llama.cpp)."""

    def __init__(
        self,
        theme: ThemeManager,
        config: ConfigManager,
        backend: ModelManagerBackend,
        bus: EventBus | None,
        *,
        ram_mb: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = "llm"
        self.title = "Языковая модель"
        self._config = config
        self._backend = backend
        self._bus = bus
        self._ram_mb = ram_mb or detect_total_ram_mb()
        self._specs = downloadable_specs()
        self._selected: LlmModelSpec | None = None
        self._downloading_id = ""
        self._unsubs: list[Callable[[], None]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("Языковая модель"))
        layout.addWidget(
            caption(
                "Необязательно. Локальная модель работает внутри Айрис — отдельная "
                "программа не нужна. Можно пропустить и выбрать позже в разделе «ИИ»."
            )
        )

        self._group = QButtonGroup(self)
        self._later = QRadioButton("Позже — базовые команды работают без модели")
        self._later.setChecked(True)
        self._later.toggled.connect(lambda _c: self._on_choice())
        self._group.addButton(self._later)
        layout.addWidget(self._later)

        self._radios: dict[str, tuple[QRadioButton, LlmModelSpec]] = {}
        for spec in self._specs:
            verdict = verdict_label(verdict_for(spec, self._ram_mb))
            radio = QRadioButton(f"{spec.name}  ·  {spec.human_size}  ·  {verdict}")
            radio.toggled.connect(lambda _c: self._on_choice())
            self._group.addButton(radio)
            self._radios[spec.gguf_catalog_id] = (radio, spec)
            layout.addWidget(radio)

        button_row = QHBoxLayout()
        self._download_button = QPushButton("Скачать выбранную")
        self._download_button.setProperty("kind", "primary")
        self._download_button.setEnabled(False)
        self._download_button.clicked.connect(self._start_download)
        button_row.addWidget(self._download_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.hide()
        layout.addWidget(self._bar)

        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._relay = _LlmRelay(self)
        self._relay.progress.connect(self._on_progress)
        self._relay.finished.connect(self._on_finished)
        self._relay.failed.connect(self._on_failed)

        self._refresh_installed()

    # -- выбор и загрузка --------------------------------------------------

    def _on_choice(self) -> None:
        self._selected = None
        for _catalog_id, (radio, spec) in self._radios.items():
            if radio.isChecked():
                self._selected = spec
                break
        selected = self._selected
        installed = selected is not None and selected.gguf_catalog_id in self._installed_ids()
        self._download_button.setEnabled(selected is not None and not installed)
        if installed:
            self._status.setText("Эта модель уже установлена — нажмите «Далее».")

    def _start_download(self) -> None:
        spec = self._selected
        if spec is None or not spec.gguf_catalog_id:
            return
        try:
            self._backend.start_download(spec.gguf_catalog_id)
        except AyrisError as exc:
            self._status.setText(f"Не удалось начать загрузку: {exc.user_message}")
            return
        except Exception as exc:  # бэкенд может кинуть что угодно — шаг не падает
            _log.exception("не удалось запустить загрузку GGUF в мастере")
            self._status.setText(f"Не удалось начать загрузку: {exc}")
            return
        self._downloading_id = spec.gguf_catalog_id
        self._download_button.setEnabled(False)
        self._bar.setRange(0, 0)  # неопределённый, пока не пришёл первый прогресс
        self._bar.show()
        self._status.setText(
            f"Скачиваю «{spec.name}» в фоне — можно продолжить, не дожидаясь конца."
        )

    # -- события шины ------------------------------------------------------

    def _on_progress(self, event: object) -> None:
        if not isinstance(event, ModelDownloadProgress) or event.model_id != self._downloading_id:
            return
        if event.total <= 0:
            self._bar.setRange(0, 0)
            return
        self._bar.setRange(0, 100)
        self._bar.setValue(int(event.fraction * 100))

    def _on_finished(self, event: object) -> None:
        if not isinstance(event, ModelDownloadFinished) or event.model_id != self._downloading_id:
            return
        self._bar.setRange(0, 100)
        self._bar.setValue(100)
        spec = self._selected
        name = spec.name if spec is not None else event.model_id
        self._write_choice(spec, event.path, event.model_id)
        self._status.setText(f"Модель «{name}» загружена и выбрана. Нажмите «Далее».")
        self._downloading_id = ""

    def _on_failed(self, event: object) -> None:
        if not isinstance(event, ModelDownloadFailed) or event.model_id != self._downloading_id:
            return
        self._bar.hide()
        self._downloading_id = ""
        self._download_button.setEnabled(True)
        if event.cancelled:
            self._status.setText("Загрузка отменена.")
        else:
            self._status.setText(f"Не удалось загрузить: {event.user_message or event.error}")

    # -- контракт шага -----------------------------------------------------

    def can_skip(self) -> bool:
        return True

    def activate(self) -> None:
        self._refresh_installed()
        if self._bus is not None and not self._unsubs:
            self._unsubs = [
                self._bus.subscribe(ModelDownloadProgress, self._relay.progress.emit),
                self._bus.subscribe(ModelDownloadFinished, self._relay.finished.emit),
                self._bus.subscribe(ModelDownloadFailed, self._relay.failed.emit),
            ]

    def deactivate(self) -> None:
        self._unsubscribe()

    def apply(self) -> None:
        # «Позже» ничего не пишет — провайдер остаётся дефолтным. Выбранная модель
        # переключает движок на встроенный llama.cpp; путь к файлу допишется по
        # завершении загрузки (здесь или обработчиком активной модели).
        spec = self._selected
        if spec is None:
            return
        record = self._find_record(spec.gguf_catalog_id)
        path = record.path if record is not None else ""
        self._write_choice(spec, path, spec.gguf_catalog_id)

    def teardown(self) -> None:
        self._unsubscribe()

    # -- служебное ---------------------------------------------------------

    def _write_choice(self, spec: LlmModelSpec | None, path: str, catalog_id: str) -> None:
        if spec is None:
            return
        updates: dict[str, object] = {
            "ai.provider": LLAMACPP_PROVIDER,
            "ai.model": spec.name,
        }
        resolved = path or self._record_path(catalog_id)
        if resolved:
            updates["ai.model_path"] = resolved
        try:
            self._config.apply(updates)
        except Exception:  # запись конфига не должна ронять мастер
            _log.exception("не удалось записать выбор локальной модели")

    def _record_path(self, catalog_id: str) -> str:
        record = self._find_record(catalog_id)
        return record.path if record is not None else ""

    def _installed_records(self) -> list[ModelRecord]:
        try:
            return self._backend.installed()
        except Exception:
            _log.exception("не удалось прочитать установленные модели в мастере")
            return []

    def _installed_ids(self) -> set[str]:
        return {record.catalog_id for record in self._installed_records() if record.catalog_id}

    def _find_record(self, catalog_id: str) -> ModelRecord | None:
        for record in self._installed_records():
            if record.kind == "llm" and record.catalog_id == catalog_id:
                return record
        return None

    def _refresh_installed(self) -> None:
        self._on_choice()

    def _unsubscribe(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                _log.exception("не удалось отписаться от событий загрузки в мастере")
        self._unsubs = []
