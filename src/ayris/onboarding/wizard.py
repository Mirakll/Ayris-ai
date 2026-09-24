"""Каркас мастера первого запуска.

Мастер — модальный :class:`QDialog` с горизонтальной лентой шагов сверху,
областью текущего шага в центре и панелью навигации снизу. Каждый шаг —
самостоятельный виджет за общим интерфейсом :class:`WizardStep`; мастер ничего
не знает об их внутренностях, только вызывает ``validate()``/``apply()`` и слушает
``completion_changed``.

Состояние живёт в конфиге (``general.onboarding_last_step`` и
``general.onboarding_completed``): прерванный на середине мастер при следующем
запуске открывается на том же шаге, а пройденный до конца больше не показывается.
Каждый шаг применяется атомарно в своём ``apply()`` на переходе «Далее» — «Назад»
и «Пропустить шаг» ничего не пишут, поэтому брошенный посередине мастер не
оставляет полуприменённых настроек.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigManager
from ayris.gui.theme import ThemeManager
from ayris.utils.logger import get_logger

__all__ = ["OnboardingWizard", "WizardStep"]

_log = get_logger(__name__)


class WizardStep(QWidget):
    """Общий интерфейс шага мастера.

    Наследники задают ``key`` и ``title`` и по необходимости переопределяют
    ``is_complete``/``validate``/``apply`` и хуки жизненного цикла. База даёт
    рабочие значения по умолчанию: шаг «выполнен», без ошибок валидации, без
    побочных эффектов и с возможностью пропуска — так любой шаг деградирует до
    «пропустить и продолжить», ничего не ломая.
    """

    #: Мастер переспрашивает навигацию, когда шаг сообщает об изменении.
    completion_changed = Signal()

    key: str = ""
    title: str = ""

    def is_complete(self) -> bool:
        """Достигнута ли цель шага (галочка в индикаторе прогресса)."""
        return True

    def validate(self) -> str | None:
        """Причина, по которой нельзя перейти дальше, или ``None`` если можно."""
        return None

    def apply(self) -> None:
        """Атомарно применить выбор шага в конфиг. По умолчанию — ничего."""

    def can_skip(self) -> bool:
        """Можно ли пропустить шаг кнопкой «Пропустить шаг»."""
        return True

    # -- жизненный цикл ----------------------------------------------------

    def activate(self) -> None:
        """Шаг стал видимым: можно запускать анимацию/подписки."""

    def deactivate(self) -> None:
        """Ушли с шага (вперёд/назад): приостановить то, что не нужно скрытым."""

    def teardown(self) -> None:
        """Мастер закрывается: погасить таймеры, потоки и подписки."""


class _StepRibbon(QFrame):
    """Горизонтальная лента шагов с подсветкой текущего.

    Подсветка держится на жирности шрифта (работает без завязки на QSS) плюс
    свойстве ``role`` для темы, если она умеет его красить.
    """

    def __init__(self, theme: ThemeManager, titles: list[str]) -> None:
        super().__init__()
        self.setObjectName("wizardRibbon")
        self._labels: list[QLabel] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_lg"))
        for i, title in enumerate(titles):
            if i:
                sep = QLabel("›")
                sep.setProperty("role", "muted")
                layout.addWidget(sep)
            label = QLabel(f"{i + 1}. {title}")
            label.setProperty("role", "muted")
            self._labels.append(label)
            layout.addWidget(label)
        layout.addStretch(1)

    def set_current(self, index: int) -> None:
        for i, label in enumerate(self._labels):
            font = label.font()
            font.setBold(i == index)
            label.setFont(font)
            label.setProperty("role", "accent" if i == index else "muted")
            style = label.style()
            style.unpolish(label)
            style.polish(label)


# placeholder-onboarding-wizard


class OnboardingWizard(QDialog):
    """Модальный мастер первого запуска.

    Принимает готовый список шагов и сам ведёт навигацию, прогресс и сохранение
    состояния. Шаги строит фабрика (см. ``build_default_steps``), поэтому мастер
    не зависит от сервисов приложения и легко проверяется на подставных шагах.
    """

    #: Мастер пройден до конца (в отличие от «Завершить позже»).
    completed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        config: ConfigManager,
        steps: list[WizardStep],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not steps:
            raise ValueError("мастеру нужен хотя бы один шаг")
        self._theme = theme
        self._config = config
        self._steps = steps
        self._index = 0
        self._torn_down = False

        self.setWindowTitle("Первый запуск — Айрис")
        self.setModal(True)
        self.setMinimumSize(760, 580)

        self._build_ui()
        self._connect_steps()
        self.finished.connect(self._on_finished)

        self._navigate(self._resume_index())

    # -- построение интерфейса --------------------------------------------

    def _build_ui(self) -> None:
        metric = self._theme.metric
        outer = metric("spacing_2xl")
        root = QVBoxLayout(self)
        root.setContentsMargins(outer, outer, outer, outer)
        root.setSpacing(metric("spacing_lg"))

        self._ribbon = _StepRibbon(self._theme, [step.title for step in self._steps])
        root.addWidget(self._ribbon)

        self._stack = QStackedWidget()
        for step in self._steps:
            self._stack.addWidget(step)
        root.addWidget(self._stack, 1)

        self._error = QLabel()
        self._error.setObjectName("wizardError")
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color: {self._theme.theme.color('error')};")
        self._error.hide()
        root.addWidget(self._error)

        root.addLayout(self._build_nav())

    def _build_nav(self) -> QHBoxLayout:
        nav = QHBoxLayout()
        nav.setSpacing(self._theme.metric("spacing_lg"))
        self._back_btn = QPushButton("Назад")
        self._back_btn.clicked.connect(self._on_back)
        self._skip_btn = QPushButton("Пропустить шаг")
        self._skip_btn.clicked.connect(self._on_skip)
        self._later_btn = QPushButton("Завершить позже")
        self._later_btn.clicked.connect(self._on_finish_later)
        self._next_btn = QPushButton("Далее")
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self._on_next)

        nav.addWidget(self._back_btn)
        nav.addStretch(1)
        nav.addWidget(self._skip_btn)
        nav.addWidget(self._later_btn)
        nav.addWidget(self._next_btn)
        return nav

    def _connect_steps(self) -> None:
        for step in self._steps:
            step.completion_changed.connect(self._refresh_nav)

    # placeholder-wizard-nav

    # -- навигация ---------------------------------------------------------

    def _resume_index(self) -> int:
        raw = self._config.settings.general.onboarding_last_step
        return max(0, min(raw, len(self._steps) - 1))

    def _navigate(self, index: int) -> None:
        """Показать шаг ``index`` без применения настроек."""
        if not 0 <= index < len(self._steps):
            return
        if index != self._index:
            self._steps[self._index].deactivate()
        self._index = index
        self._stack.setCurrentIndex(index)
        self._ribbon.set_current(index)
        self._error.hide()
        self._steps[index].activate()
        self._refresh_nav()

    def _on_next(self) -> None:
        step = self._steps[self._index]
        problem = step.validate()
        if problem:
            self._show_error(problem)
            return
        try:
            step.apply()
        except Exception:
            _log.exception("шаг %s: apply() упал", step.key)
            self._show_error(
                "Не удалось применить настройки шага. Можно пропустить его и продолжить."
            )
            return
        self._advance()

    def _on_skip(self) -> None:
        self._advance()

    def _advance(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._finish()
            return
        self._persist_step(self._index + 1)
        self._navigate(self._index + 1)

    def _on_back(self) -> None:
        if self._index > 0:
            self._navigate(self._index - 1)

    def _on_finish_later(self) -> None:
        self._persist_step(self._index)
        self.reject()

    def _finish(self) -> None:
        self._config.apply(
            {
                "general.onboarding_completed": True,
                "general.show_onboarding": False,
                "general.onboarding_last_step": len(self._steps),
            }
        )
        self.completed.emit()
        self.accept()

    # -- служебное ---------------------------------------------------------

    def _persist_step(self, index: int) -> None:
        try:
            self._config.apply({"general.onboarding_last_step": index})
        except Exception:
            _log.exception("не удалось сохранить прогресс мастера")

    def _refresh_nav(self) -> None:
        last = self._index >= len(self._steps) - 1
        step = self._steps[self._index]
        self._back_btn.setEnabled(self._index > 0)
        self._skip_btn.setVisible(step.can_skip())
        self._next_btn.setText("Завершить" if last else "Далее")

    def _show_error(self, text: str) -> None:
        self._error.setText(text)
        self._error.show()

    def _on_finished(self, _result: int) -> None:
        self._teardown_all()

    def _teardown_all(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        for step in self._steps:
            try:
                step.teardown()
            except Exception:
                _log.exception("шаг %s: teardown() упал", step.key)
