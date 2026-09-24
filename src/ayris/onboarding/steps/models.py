"""Шаг «Модели»: скачать минимальный набор для работы Айрис из коробки.

Набор — пять моделей на ~280 МБ: две общие части openWakeWord, обученная фраза
«айрис», распознавание речи GigaAM и русский голос Piper. Никаких LLM: базовый
сценарий (активация → команда → ответ) работает и без них.

Шаг ничего не качает сам — вся работа идёт через уже готовый бэкенд менеджера
моделей (:class:`~ayris.gui.widgets.model_manager.ModelManagerBackend`, задачи 14
и 50): его :class:`~ayris.gui.tabs.updates.DownloadCoordinator` крутит загрузки в
фоновых потоках и продолжает их даже после закрытия окна, а прогресс и итог
приходят событиями шины. Поэтому шаг пропускаемый и не блокирует мастер: можно
уйти дальше, пока качается.

Здесь же решается «Айрис из коробки» (пункт 13 задачи): по факту установленного
набора выбирается движок активации — :func:`resolve_wake_engine` — чтобы дефолтная
фраза «айрис» никогда не осталась без модели (мёртвый openWakeWord). Функция
чистая и проверяется тестами отдельно от виджета.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
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
    ModelDownloadStarted,
)
from ayris.gui.theme import ThemeManager
from ayris.models.downloader import human_size
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.models import ModelRecord
    from ayris.gui.widgets.model_manager import ModelManagerBackend
    from ayris.models.catalog import ModelCatalog

__all__ = ["MINIMAL_MODEL_IDS", "ModelsStep", "resolve_wake_engine"]

_log = get_logger(__name__)

#: Минимальный набор «из коробки», в порядке показа. Без LLM — базовый сценарий
#: (активация → распознавание → голос) работает и на нём.
MINIMAL_MODEL_IDS: tuple[str, ...] = (
    "oww-melspectrogram",
    "oww-embedding",
    "oww-airis-ru",
    "gigaam-v3-ctc",
    "piper-ru-irina",
)

#: openWakeWord для «айрис» — это обе общие части движка плюс файл фразы. Без всех
#: трёх дефолтная активация мертва.
_AIRIS_WAKE_IDS: tuple[str, ...] = ("oww-melspectrogram", "oww-embedding", "oww-airis-ru")


def resolve_wake_engine(installed: Iterable[ModelRecord]) -> dict[str, object]:
    """Значения конфига активации по факту установленных моделей (пункт 13).

    - Есть все части «айрис» → дефолтный openWakeWord с фразой «айрис».
    - Иначе есть Vosk-STT → активация через Vosk KWS (переиспользует его модель).
    - Иначе моделей нет → активацию выключаем, чтобы дефолтная фраза не осталась
      без модели (мёртвый движок). Пользователь включит её на вкладке «Голос»,
      когда модель появится.

    Чистая функция: возвращает набор dotted-path значений для
    :meth:`ConfigManager.apply`, ничего не пишет сама.
    """
    installed = list(installed)
    catalog_ids = {record.catalog_id for record in installed if record.catalog_id}
    if all(part in catalog_ids for part in _AIRIS_WAKE_IDS):
        return {"voice.wake.enabled": True, "voice.wake.engine": "openwakeword"}
    vosk = next(
        (record for record in installed if record.kind == "stt" and record.engine == "vosk"),
        None,
    )
    if vosk is not None:
        options = {"model_path": vosk.path} if vosk.path else {}
        return {
            "voice.wake.enabled": True,
            "voice.wake.engine": "vosk",
            "voice.wake.options": options,
        }
    return {"voice.wake.enabled": False}


def reconcile_wake(config: ConfigManager, installed: Iterable[ModelRecord]) -> None:
    """Применить :func:`resolve_wake_engine` в конфиг, не роняя вызывающего."""
    try:
        config.apply(resolve_wake_engine(installed))
    except Exception:
        _log.exception("не удалось согласовать движок активации с набором моделей")


class _ModelRelay(QObject):
    """Переносит события загрузки с фонового потока на поток GUI."""

    started = Signal(object)
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(object)


class _ModelRow:
    """Строка одной модели: имя, размер и полоса прогресса со статусом."""

    def __init__(self, theme: ThemeManager, name: str, total_bytes: int) -> None:
        self._total = total_bytes
        self.widget = QWidget()
        self.widget.setProperty("transparent", True)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_xs"))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(name)
        header.addWidget(title, 1)
        size_text = human_size(total_bytes) if total_bytes else ""
        self._size = QLabel(size_text)
        self._size.setProperty("role", "muted")
        header.addWidget(self._size)
        layout.addLayout(header)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)

        self.status = QLabel("Ожидание")
        self.status.setProperty("role", "muted")
        layout.addWidget(self.status)

    def mark_installed(self) -> None:
        self.bar.setValue(100)
        self.status.setText("Установлено")

    def mark_waiting(self) -> None:
        self.bar.setValue(0)
        self.status.setText("Ожидание")

    def mark_started(self, downloaded: int, total: int) -> None:
        if total:
            self._total = total
            self._size.setText(human_size(total))
        self.status.setText("Загрузка…")
        self._set_fraction(downloaded, total or self._total)

    def mark_progress(self, downloaded: int, total: int) -> None:
        self._set_fraction(downloaded, total or self._total)

    def mark_failed(self, message: str) -> None:
        self.status.setText(f"Ошибка: {message}" if message else "Ошибка загрузки")

    def _set_fraction(self, downloaded: int, total: int) -> None:
        if total <= 0:
            self.bar.setRange(0, 0)  # неопределённый прогресс
            return
        self.bar.setRange(0, 100)
        self.bar.setValue(int(min(1.0, downloaded / total) * 100))


class ModelsStep(WizardStep):
    """Скачивание минимального набора моделей в фоне."""

    def __init__(
        self,
        theme: ThemeManager,
        config: ConfigManager,
        backend: ModelManagerBackend,
        bus: EventBus | None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = "models"
        self.title = "Модели"
        self._config = config
        self._backend = backend
        self._bus = bus
        self._unsubs: list[Callable[[], None]] = []
        self._rows: dict[str, _ModelRow] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("Модели"))

        catalog = self._catalog()
        total = 0
        specs: list[tuple[str, str, int]] = []
        for model_id in MINIMAL_MODEL_IDS:
            entry = catalog.get(model_id)
            name = entry.name if entry is not None else model_id
            size = entry.total_bytes if entry is not None else 0
            total += size
            specs.append((model_id, name, size))

        layout.addWidget(
            caption(
                f"Минимальный набор для работы «из коробки» — {human_size(total)}. "
                "Активация, распознавание речи и голос ответа. Модели языковых моделей "
                "(LLM) сюда не входят — базовые команды работают и без них."
            )
        )

        for model_id, name, size in specs:
            row = _ModelRow(theme, name, size)
            self._rows[model_id] = row
            layout.addWidget(row.widget)

        button_row = QHBoxLayout()
        self._download_button = QPushButton(f"Скачать набор ({human_size(total)})")
        self._download_button.setProperty("kind", "primary")
        self._download_button.clicked.connect(self._download_all)
        button_row.addWidget(self._download_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._notice = QLabel("")
        self._notice.setProperty("role", "muted")
        self._notice.setWordWrap(True)
        layout.addWidget(self._notice)
        layout.addStretch(1)

        self._relay = _ModelRelay(self)
        self._relay.started.connect(self._on_started)
        self._relay.progress.connect(self._on_progress)
        self._relay.finished.connect(self._on_finished)
        self._relay.failed.connect(self._on_failed)

        self._refresh_installed()

    # -- бэкенд ------------------------------------------------------------

    def _catalog(self) -> ModelCatalog:
        try:
            return self._backend.catalog()
        except Exception:
            _log.exception("не удалось получить каталог моделей")
            from ayris.models.catalog import ModelCatalog

            return ModelCatalog(entries=())

    def _installed_records(self) -> list[ModelRecord]:
        try:
            return self._backend.installed()
        except Exception:
            _log.exception("не удалось получить список установленных моделей")
            return []

    def _installed_ids(self) -> set[str]:
        return {record.catalog_id for record in self._installed_records() if record.catalog_id}

    def _refresh_installed(self) -> None:
        installed = self._installed_ids()
        pending = 0
        for model_id, row in self._rows.items():
            if model_id in installed:
                row.mark_installed()
            elif self._is_downloading(model_id):
                row.status.setText("Загрузка…")
                pending += 1
            else:
                row.mark_waiting()
                pending += 1
        self._download_button.setEnabled(pending > 0)
        if pending == 0:
            self._notice.setText("Весь набор установлен.")
            self._download_button.setText("Набор установлен")

    def _is_downloading(self, model_id: str) -> bool:
        try:
            return self._backend.is_downloading(model_id)
        except Exception:
            _log.exception("не удалось проверить состояние загрузки %s", model_id)
            return False

    # -- загрузка ----------------------------------------------------------

    def _download_all(self) -> None:
        installed = self._installed_ids()
        problems: list[str] = []
        for model_id, row in self._rows.items():
            if model_id in installed or self._is_downloading(model_id):
                continue
            try:
                self._backend.start_download(model_id)
            except AyrisError as exc:
                row.mark_failed(exc.user_message)
                problems.append(exc.user_message)
            except Exception as exc:
                _log.exception("не удалось запустить загрузку %s", model_id)
                row.mark_failed(str(exc))
                problems.append(str(exc))
            else:
                row.status.setText("Загрузка…")
        if problems:
            self._notice.setText("; ".join(dict.fromkeys(problems)))
        else:
            self._notice.setText("Загрузка идёт в фоне — можно продолжить, не дожидаясь конца.")
        self._download_button.setEnabled(False)

    # -- события шины ------------------------------------------------------

    def _on_started(self, event: ModelDownloadStarted) -> None:
        row = self._rows.get(event.model_id)
        if row is not None:
            row.mark_started(event.downloaded, event.total)

    def _on_progress(self, event: ModelDownloadProgress) -> None:
        row = self._rows.get(event.model_id)
        if row is not None:
            row.mark_progress(event.downloaded, event.total)

    def _on_finished(self, event: ModelDownloadFinished) -> None:
        row = self._rows.get(event.model_id)
        if row is not None:
            row.mark_installed()
        # Набор мог только что стать полным — сразу чиним движок активации.
        reconcile_wake(self._config, self._installed_records())
        self._refresh_installed()

    def _on_failed(self, event: ModelDownloadFailed) -> None:
        row = self._rows.get(event.model_id)
        if row is None:
            return
        if event.cancelled:
            row.mark_waiting()
        else:
            row.mark_failed(event.user_message)
        self._download_button.setEnabled(True)

    # -- контракт шага -----------------------------------------------------

    def activate(self) -> None:
        self._refresh_installed()
        if self._bus is not None and not self._unsubs:
            self._unsubs = [
                self._bus.subscribe(ModelDownloadStarted, self._relay.started.emit),
                self._bus.subscribe(ModelDownloadProgress, self._relay.progress.emit),
                self._bus.subscribe(ModelDownloadFinished, self._relay.finished.emit),
                self._bus.subscribe(ModelDownloadFailed, self._relay.failed.emit),
            ]

    def deactivate(self) -> None:
        self._unsubscribe()

    def apply(self) -> None:
        # «Далее» согласует движок активации с тем, что реально установлено.
        reconcile_wake(self._config, self._installed_records())

    def teardown(self) -> None:
        self._unsubscribe()

    def _unsubscribe(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                _log.exception("не удалось отписаться от событий загрузки")
        self._unsubs = []
