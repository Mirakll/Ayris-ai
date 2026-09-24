"""Сборка мастера первого запуска из живых сервисов приложения.

Здесь смыкаются два мира: чистые шаги (:mod:`ayris.onboarding.steps`), которые
знают только свои узкие протоколы, и настоящее приложение с БД, каталогом
моделей, парсером VoiceAttack и звуковым железом. Фабрика строит конкретные
реализации этих протоколов поверх тех же кирпичей, что использует остальной GUI
(бэкенд менеджера моделей из вкладки «Обновления», ``CommandTreeStore`` вкладки
«Команды», перечислитель устройств вкладки «Голос»), собирает :class:`WizardServices`
и по ним — список шагов и сам :class:`OnboardingWizard`.

Каждый недоступный сервис вырождается в ``None`` (нет профиля — нет импортёра,
нет каталога — нет шага моделей), а не роняет мастер: онбординг обязан подниматься
даже на неполном окружении первого запуска.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ayris.core.config import ConfigManager
from ayris.core.events import EventBus
from ayris.gui.theme import ThemeManager
from ayris.onboarding.services import WizardServices
from ayris.onboarding.steps.audio import AudioStep
from ayris.onboarding.steps.mode import ModeStep
from ayris.onboarding.steps.models import ModelsStep, reconcile_wake
from ayris.onboarding.steps.profile import ProfileStep
from ayris.onboarding.steps.theme import ThemeStep
from ayris.onboarding.steps.tutorial import TutorialStep
from ayris.onboarding.steps.welcome import WelcomeStep
from ayris.onboarding.wizard import OnboardingWizard, WizardStep
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from PySide6.QtWidgets import QWidget

    from ayris.audio.calibration import CalibrationReport
    from ayris.core.database import Database
    from ayris.gui.widgets.command_tree_model import CommandTreeStore
    from ayris.gui.widgets.model_manager import ModelManagerBackend

__all__ = [
    "build_default_steps",
    "build_services",
    "run_onboarding",
    "should_run_onboarding",
]

_log = get_logger(__name__)


class _AppProfileImporter:
    """Импорт стартового профиля поверх живой БД (контракт ``ProfileImporter``).

    «Примеры» — это ``.ayris`` из ``resources/examples``, каждый прогоняется через
    :meth:`CommandTreeStore.apply_import` (та же дорога, что и импорт на вкладке
    «Команды»). «VoiceAttack» — через штатный :class:`VoiceAttackImporter`. Ни один
    сбой на одном файле не роняет остальные и мастер.
    """

    def __init__(
        self,
        store: CommandTreeStore,
        database: Database,
        profile_id: int,
        sounds_dir: Path,
        examples_dir: Path,
    ) -> None:
        self._store = store
        self._database = database
        self._profile_id = profile_id
        self._sounds_dir = sounds_dir
        self._examples_dir = examples_dir

    def _example_files(self) -> list[Path]:
        try:
            return sorted(self._examples_dir.glob("*.ayris"))
        except OSError:
            _log.exception("не удалось перечислить примеры команд в %s", self._examples_dir)
            return []

    def preview_examples(self) -> list[str]:
        from ayris.actions.macros.serializer import load_document

        names: list[str] = []
        for path in self._example_files():
            try:
                document = load_document(path.read_text(encoding="utf-8"))
            except Exception:
                _log.exception("не удалось прочитать пример %s", path)
                continue
            names.extend(command.name for command in document.commands)
        return names

    def import_examples(self) -> int:
        from ayris.gui.widgets.command_tree_model import ConflictStrategy

        total = 0
        for path in self._example_files():
            try:
                outcome = self._store.apply_import(
                    path.read_text(encoding="utf-8"),
                    target_folder_id=None,
                    strategy=ConflictStrategy.RENAME,
                )
            except Exception:
                _log.exception("не удалось импортировать пример %s", path)
                continue
            total += outcome.imported
        return total

    def preview_voiceattack(self, path: Path) -> list[str]:
        from ayris.actions.macros.importers.voiceattack import VoiceAttackImporter

        try:
            result = VoiceAttackImporter().parse(path)
        except Exception:
            _log.exception("не удалось разобрать файл VoiceAttack %s", path)
            return []
        return [command.name for command in result.commands]

    def import_voiceattack(self, path: Path) -> int:
        from ayris.actions.macros.importers.voiceattack import VoiceAttackImporter

        importer = VoiceAttackImporter()
        result = importer.parse(path)
        report = importer.apply(
            result,
            ("Импорт VoiceAttack",),
            database=self._database,
            profile_id=self._profile_id,
            sounds_dir=self._sounds_dir,
        )
        return report.imported


class _AppAudioProbe:
    """Пробник железа поверх PortAudio (контракт ``AudioProbe``).

    Перечисляет устройства записи для выбора в шаге «Микрофон». Калибровку в
    мастере не запускаем: на первом запуске микрофоном уже владеет аудио-воркер,
    и открывать устройство вторым владельцем ради калибровки небезопасно. Поэтому
    :meth:`can_calibrate` возвращает ``False`` — шаг честно вырождает калибровку в
    «недоступно», оставляя выбор устройства, усиление и живой уровень. Полная
    калибровка остаётся на вкладке «Голос», где источник звука уже её собственный.
    """

    def input_devices(self) -> list[tuple[str, str]]:
        from ayris.audio.devices import DeviceDirection, SoundDeviceBackend, list_devices

        devices = list_devices(SoundDeviceBackend(), DeviceDirection.INPUT)
        return [(device.id, device.label) for device in devices]

    def can_calibrate(self) -> bool:
        return False

    def calibrate(self, *, base_gain: float) -> CalibrationReport:
        del base_gain  # часть сигнатуры протокола; здесь калибровки нет
        raise RuntimeError("калибровка в мастере первого запуска недоступна")


def _build_backend(bus: EventBus | None) -> ModelManagerBackend:
    """Бэкенд менеджера моделей — тот же, что у вкладки «Обновления».

    Загрузки живут в его ``DownloadCoordinator`` и продолжаются даже после закрытия
    мастера. Если хранилище не готово, вернётся пустой бэкенд, и шаг моделей просто
    покажет каталог без загрузок.
    """
    from ayris.gui.tabs.updates import _default_backend

    return _default_backend(bus)


def _build_importer() -> _AppProfileImporter | None:
    """Импортёр стартового профиля, или ``None`` если профиль/БД недоступны.

    Повторяет ленивую сборку :func:`ayris.gui.tabs.commands.build_store`: без
    активного профиля шаг «Профиль» деградирует до «Чистый», а не падает.
    """
    try:
        from ayris.core.database import get_database
        from ayris.core.paths import executable_dir, get_paths
        from ayris.core.repositories import Repositories
        from ayris.gui.widgets.command_tree_model import CommandTreeStore

        database = get_database()
        repositories = Repositories(database)
        active = repositories.profiles.active()
        if active is None or active.id is None:
            return None
        store = CommandTreeStore(repositories, active.id)
        examples_dir = executable_dir() / "resources" / "examples"
        return _AppProfileImporter(store, database, active.id, get_paths().sounds_dir, examples_dir)
    except Exception:
        _log.exception("не удалось подготовить импорт стартового профиля")
        return None


def build_services(
    theme: ThemeManager,
    config: ConfigManager,
    bus: EventBus | None,
    *,
    submit_text: Callable[[str], None] | None = None,
) -> WizardServices:
    """Собрать зависимости мастера из живых сервисов приложения."""
    return WizardServices(
        theme=theme,
        config=config,
        bus=bus,
        backend=_build_backend(bus),
        importer=_build_importer(),
        audio_probe=_AppAudioProbe(),
        submit_text=submit_text,
    )


def build_default_steps(services: WizardServices) -> list[WizardStep]:
    """Штатная последовательность шагов первого запуска, в порядке показа.

    Приветствие → Тема → Режим → Микрофон → Модели → Профиль → Проба. Шаг моделей
    появляется только при наличии бэкенда каталога — без него скачивать нечего.
    """
    theme = services.theme
    config = services.config
    steps: list[WizardStep] = [
        WelcomeStep(theme),
        ThemeStep(theme, config),
        ModeStep(theme, config),
        AudioStep(theme, config, services.audio_probe, services.bus),
    ]
    if services.backend is not None:
        steps.append(ModelsStep(theme, config, services.backend, services.bus))
    steps.append(ProfileStep(theme, services.importer))
    steps.append(TutorialStep(theme, services.bus, services.submit_text))
    return steps


def should_run_onboarding(config: ConfigManager) -> bool:
    """Показывать ли мастер на этом запуске.

    ``show_onboarding`` включён и мастер ещё не пройден до конца. «Завершить позже»
    оставляет оба флага, поэтому прерванный мастер вернётся на том же шаге; финал
    гасит ``show_onboarding`` и ставит ``onboarding_completed``.
    """
    general = config.settings.general
    return general.show_onboarding and not general.onboarding_completed


def run_onboarding(services: WizardServices, *, parent: QWidget | None = None) -> bool:
    """Показать мастер модально и вернуть, пройден ли он до конца.

    При любом закрытии (готово/позже/крестик) согласуем движок активации с реально
    установленными моделями — так дефолтная фраза «айрис» не остаётся без модели,
    даже если шаг моделей пропустили (пункты 12–13 задачи).
    """
    steps = build_default_steps(services)
    wizard = OnboardingWizard(services.theme, services.config, steps, parent=parent)

    def _reconcile(_result: int) -> None:
        backend = services.backend
        if backend is None:
            return
        try:
            installed = backend.installed()
        except Exception:
            _log.exception("не удалось прочитать установленные модели после мастера")
            return
        reconcile_wake(services.config, installed)

    wizard.finished.connect(_reconcile)
    wizard.exec()
    return bool(services.config.settings.general.onboarding_completed)
