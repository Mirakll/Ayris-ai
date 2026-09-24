"""Мастер первого запуска: шаги, прерывание/возобновление и «из коробки».

Проверяется четыре обещания задачи 60: последовательность шагов и их
деградация до «пропустить»; прерванный мастер продолжается с того же шага;
примеры команд валидны и ничего опасного не несут; дефолтная фраза активации
«айрис» никогда не остаётся без модели (:func:`resolve_wake_engine`).

Всё оффскрин и без железа/сети: аудио-пробник, бэкенд загрузок и импортёр
профиля — заглушки. Каждый тест закрывает свой мастер/шаг, чтобы таймер сферы
приветствия не протёк в следующий; ``app``-фикстура добивает остаток.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.serializer import load_document
from ayris.actions.macros.validator import validate_command
from ayris.actions.registry import ActionRegistry
from ayris.core.config import ConfigManager
from ayris.core.events import EventBus, IntentMatched
from ayris.core.models import ModelRecord
from ayris.core.paths import ModelKind
from ayris.gui.theme import ThemeManager
from ayris.models.catalog import ModelCatalog
from ayris.models.registry import DiskUsage, IntegrityReport, IntegrityStatus
from ayris.onboarding.factory import build_default_steps, should_run_onboarding
from ayris.onboarding.services import WizardServices
from ayris.onboarding.steps.audio import AudioStep
from ayris.onboarding.steps.mode import ModeStep
from ayris.onboarding.steps.models import (
    MINIMAL_MODEL_IDS,
    ModelsStep,
    reconcile_wake,
    resolve_wake_engine,
)
from ayris.onboarding.steps.profile import ProfileStep
from ayris.onboarding.steps.theme import ThemeStep
from ayris.onboarding.steps.tutorial import DEFAULT_TASKS, TutorialStep
from ayris.onboarding.steps.welcome import WelcomeStep
from ayris.onboarding.wizard import OnboardingWizard, WizardStep

if TYPE_CHECKING:
    from ayris.audio.calibration import CalibrationReport

pytestmark = pytest.mark.unit

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "resources" / "examples"
_SHA = "0" * 64
#: Действия, которых в примерах быть не должно (пункт 9): необратимое и запуск
#: произвольных программ — ничего из этого учебные команды не делают.
_FORBIDDEN_BLOCKS = {"PowerAction", "RunApp"}
#: Полный набор ключей шагов в штатном порядке показа.
_FULL_SEQUENCE = ["welcome", "theme", "mode", "audio", "models", "profile", "tutorial"]

_EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.ayris"))


# ----------------------------------------------------------------------
# фикстуры
# ----------------------------------------------------------------------


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def theme(app: QApplication) -> ThemeManager:
    return ThemeManager(app)


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    config = ConfigManager(tmp_path / "config.toml")
    config.load()
    return config


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg.discover()
    return reg


# ----------------------------------------------------------------------
# заглушки железа, загрузок и импорта
# ----------------------------------------------------------------------


class FakeAudioProbe:
    """Пробник без звуковой карты: перечисляет устройства, калибровку не даёт."""

    def __init__(
        self, devices: tuple[tuple[str, str], ...] = (), *, can_calibrate: bool = False
    ) -> None:
        self._devices = list(devices)
        self._can = can_calibrate

    def input_devices(self) -> list[tuple[str, str]]:
        return list(self._devices)

    def can_calibrate(self) -> bool:
        return self._can

    def calibrate(self, *, base_gain: float) -> CalibrationReport:  # pragma: no cover
        raise RuntimeError("калибровка в тестах недоступна")


class FakeImporter:
    """Импортёр профиля, который ничего не пишет, а лишь считает вызовы."""

    def __init__(self, examples: tuple[str, ...] = (), vap: tuple[str, ...] = ()) -> None:
        self._examples = list(examples)
        self._vap = list(vap)
        self.imported_examples = 0
        self.imported_vap = 0

    def preview_examples(self) -> list[str]:
        return list(self._examples)

    def import_examples(self) -> int:
        self.imported_examples += 1
        return len(self._examples)

    def preview_voiceattack(self, path: Path) -> list[str]:
        return list(self._vap)

    def import_voiceattack(self, path: Path) -> int:
        self.imported_vap += 1
        return len(self._vap)


def _record(catalog_id: str, kind: ModelKind, *, engine: str = "", path: str = "") -> ModelRecord:
    return ModelRecord(
        kind=kind,
        name=catalog_id,
        id=abs(hash(catalog_id)) % 100000,
        engine=engine,
        catalog_id=catalog_id,
        path=path,
        sha256=_SHA,
        size_bytes=1234,
    )


class FakeBackend:
    """Бэкенд менеджера моделей без сети и диска: пишет вызовы, не качает."""

    def __init__(self, installed: tuple[ModelRecord, ...] = ()) -> None:
        self._installed = list(installed)
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self._downloading: set[str] = set()

    def catalog(self) -> ModelCatalog:
        return ModelCatalog(entries=())

    def installed(self) -> list[ModelRecord]:
        return list(self._installed)

    def active(self, kind: ModelKind) -> ModelRecord | None:
        return next((r for r in self._installed if r.kind == kind), None)

    def disk_usage(self) -> DiskUsage:
        by_kind = {"stt": 0, "tts": 0, "wake": 0, "llm": 0}
        return DiskUsage(by_kind=by_kind, total=0, cache_bytes=0)

    def free_disk_bytes(self) -> int:
        return 500 * 1024 * 1024

    def set_active(self, record: ModelRecord) -> None:
        self._installed.append(record)

    def remove(self, record: ModelRecord, *, force: bool) -> int:
        self._installed = [r for r in self._installed if r.id != record.id]
        return 0

    def verify(self, record: ModelRecord, *, full: bool) -> IntegrityReport:
        return IntegrityReport(record=record, status=IntegrityStatus.OK, full=full)

    def start_download(self, model_id: str) -> None:
        self.started.append(model_id)
        self._downloading.add(model_id)

    def cancel_download(self, model_id: str) -> None:
        self.cancelled.append(model_id)
        self._downloading.discard(model_id)

    def is_downloading(self, model_id: str) -> bool:
        return model_id in self._downloading


class _DummyStep(WizardStep):
    """Минимальный шаг для проверки навигации мастера без реальных сервисов."""

    def __init__(self, key: str, *, skippable: bool = True) -> None:
        super().__init__()
        self.key = key
        self.title = key.title()
        self.applied = 0
        self._skippable = skippable

    def apply(self) -> None:
        self.applied += 1

    def can_skip(self) -> bool:
        return self._skippable


def _services(
    theme: ThemeManager,
    manager: ConfigManager,
    *,
    backend: FakeBackend | None = None,
    importer: FakeImporter | None = None,
    audio_probe: FakeAudioProbe | None = None,
    bus: EventBus | None = None,
    submit_text: Callable[[str], None] | None = None,
) -> WizardServices:
    return WizardServices(
        theme=theme,
        config=manager,
        bus=bus,
        backend=backend,  # type: ignore[arg-type]
        importer=importer,  # type: ignore[arg-type]
        audio_probe=audio_probe,  # type: ignore[arg-type]
        submit_text=submit_text,
    )


def _teardown_steps(steps: list[WizardStep]) -> None:
    for step in steps:
        step.teardown()
        step.close()


# ----------------------------------------------------------------------
# последовательность шагов и её деградация
# ----------------------------------------------------------------------


def test_default_sequence_with_backend(theme: ThemeManager, manager: ConfigManager) -> None:
    steps = build_default_steps(_services(theme, manager, backend=FakeBackend()))
    assert [step.key for step in steps] == _FULL_SEQUENCE
    _teardown_steps(steps)


def test_sequence_drops_models_without_backend(theme: ThemeManager, manager: ConfigManager) -> None:
    steps = build_default_steps(_services(theme, manager, backend=None))
    keys = [step.key for step in steps]
    assert "models" not in keys
    assert keys == ["welcome", "theme", "mode", "audio", "profile", "tutorial"]
    _teardown_steps(steps)


# ----------------------------------------------------------------------
# прерывание и возобновление
# ----------------------------------------------------------------------


def test_wizard_resumes_at_saved_step(theme: ThemeManager, manager: ConfigManager) -> None:
    manager.apply({"general.onboarding_last_step": 2})
    steps: list[WizardStep] = [_DummyStep("a"), _DummyStep("b"), _DummyStep("c")]
    wizard = OnboardingWizard(theme, manager, steps)
    assert wizard._index == 2
    wizard._teardown_all()
    wizard.close()


def test_wizard_clamps_out_of_range_resume(theme: ThemeManager, manager: ConfigManager) -> None:
    # «Завершить» пишет last_step == len(steps); при следующем показе (если бы он
    # случился) индекс не должен вылезти за последний шаг.
    manager.apply({"general.onboarding_last_step": 99})
    steps: list[WizardStep] = [_DummyStep("a"), _DummyStep("b")]
    wizard = OnboardingWizard(theme, manager, steps)
    assert wizard._index == 1
    wizard._teardown_all()
    wizard.close()


def test_finish_later_keeps_flags_and_saves_step(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    steps: list[WizardStep] = [_DummyStep("a"), _DummyStep("b"), _DummyStep("c")]
    wizard = OnboardingWizard(theme, manager, steps)
    wizard._navigate(1)
    wizard._on_finish_later()
    general = manager.settings.general
    assert general.onboarding_last_step == 1
    assert general.show_onboarding is True
    assert general.onboarding_completed is False
    wizard.close()


def test_completing_sets_flags(theme: ThemeManager, manager: ConfigManager) -> None:
    steps: list[WizardStep] = [_DummyStep("a"), _DummyStep("b"), _DummyStep("c")]
    wizard = OnboardingWizard(theme, manager, steps)
    for _ in steps:
        wizard._on_skip()
    general = manager.settings.general
    assert general.onboarding_completed is True
    assert general.show_onboarding is False
    assert general.onboarding_last_step == len(steps)
    wizard.close()


def test_next_applies_and_advances(theme: ThemeManager, manager: ConfigManager) -> None:
    steps: list[WizardStep] = [_DummyStep("a"), _DummyStep("b")]
    wizard = OnboardingWizard(theme, manager, steps)
    wizard._on_next()
    assert steps[0].applied == 1  # type: ignore[attr-defined]
    assert wizard._index == 1
    wizard._teardown_all()
    wizard.close()


def test_back_and_skip_do_not_apply(theme: ThemeManager, manager: ConfigManager) -> None:
    steps: list[WizardStep] = [_DummyStep("a"), _DummyStep("b"), _DummyStep("c")]
    wizard = OnboardingWizard(theme, manager, steps)
    wizard._on_skip()  # 0 -> 1, без apply
    wizard._on_back()  # 1 -> 0
    assert wizard._index == 0
    assert all(step.applied == 0 for step in steps)  # type: ignore[attr-defined]
    wizard._teardown_all()
    wizard.close()


# ----------------------------------------------------------------------
# отдельные шаги: применение и деградация
# ----------------------------------------------------------------------


def test_welcome_step_cannot_be_skipped(theme: ThemeManager) -> None:
    step = WelcomeStep(theme)
    assert step.can_skip() is False
    step.close()


def test_theme_step_writes_config(theme: ThemeManager, manager: ConfigManager) -> None:
    step = ThemeStep(theme, manager)
    step._buttons["light"].setChecked(True)
    step.apply()
    assert manager.settings.general.theme == "light"
    step.close()


def test_mode_step_writes_config(theme: ThemeManager, manager: ConfigManager) -> None:
    step = ModeStep(theme, manager)
    step._buttons["offline"].setChecked(True)
    step.apply()
    assert manager.settings.voice.stt.mode == "offline"
    step.close()


def test_profile_step_imports_examples(theme: ThemeManager) -> None:
    importer = FakeImporter(examples=("Открой браузер", "Громкость"))
    step = ProfileStep(theme, importer)
    assert step._examples.isChecked()  # примеры выбраны по умолчанию
    step.apply()
    assert importer.imported_examples == 1
    assert importer.imported_vap == 0
    step.close()


def test_profile_clean_imports_nothing(theme: ThemeManager) -> None:
    importer = FakeImporter(examples=("Открой браузер",))
    step = ProfileStep(theme, importer)
    step._clean.setChecked(True)
    step.apply()
    assert importer.imported_examples == 0
    step.close()


def test_profile_without_importer_forces_clean(theme: ThemeManager) -> None:
    step = ProfileStep(theme, None)
    assert step._clean.isChecked()
    assert not step._examples.isEnabled()
    assert not step._voiceattack.isEnabled()
    step.apply()  # ничего не должно упасть без импортёра
    step.close()


def test_audio_step_without_mic_degrades(theme: ThemeManager, manager: ConfigManager) -> None:
    step = AudioStep(theme, manager, None, None)
    assert step.can_skip() is True
    assert not step._calibrate_button.isEnabled()
    assert "Микрофон не найден" in step._device_notice.text()
    step.teardown()
    step.close()


def test_audio_step_lists_devices_with_calibration_disabled(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    probe = FakeAudioProbe(devices=(("dev-1", "Микрофон 1"),), can_calibrate=False)
    step = AudioStep(theme, manager, probe, None)
    # «Системное по умолчанию» плюс найденное устройство.
    assert step._device_combo.count() >= 2
    assert not step._calibrate_button.isEnabled()
    step.teardown()
    step.close()


def test_models_step_downloads_full_minimal_set(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    backend = FakeBackend()
    step = ModelsStep(theme, manager, backend, None)  # type: ignore[arg-type]
    assert step.can_skip() is True
    step._download_all()
    assert backend.started == list(MINIMAL_MODEL_IDS)
    step.teardown()
    step.close()


def test_models_step_skips_already_installed(theme: ThemeManager, manager: ConfigManager) -> None:
    installed = _record("gigaam-v3-ctc", "stt", engine="gigaam")
    backend = FakeBackend(installed=(installed,))
    step = ModelsStep(theme, manager, backend, None)  # type: ignore[arg-type]
    step._download_all()
    assert "gigaam-v3-ctc" not in backend.started
    assert set(backend.started) == set(MINIMAL_MODEL_IDS) - {"gigaam-v3-ctc"}
    step.teardown()
    step.close()


def test_tutorial_marks_done_on_matched(app: QApplication, theme: ThemeManager) -> None:
    bus = EventBus()
    step = TutorialStep(theme, bus, None)
    step.activate()
    assert step.is_complete() is False
    for _ in DEFAULT_TASKS:
        bus.publish(IntentMatched(intent="проба"))
    app.processEvents()
    assert step.is_complete() is True
    step.teardown()
    step.close()


def test_tutorial_check_by_text_submits_current_phrase(theme: ThemeManager) -> None:
    calls: list[str] = []
    step = TutorialStep(theme, None, calls.append)
    step._check_by_text()
    assert calls == [DEFAULT_TASKS[0].phrase]
    step.teardown()
    step.close()


# ----------------------------------------------------------------------
# примеры команд: валидны и ничего опасного не несут
# ----------------------------------------------------------------------


def test_examples_directory_is_not_empty() -> None:
    assert _EXAMPLE_FILES, f"нет ни одного примера в {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=[p.name for p in _EXAMPLE_FILES])
def test_example_command_is_valid(path: Path, registry: ActionRegistry) -> None:
    document = load_document(path.read_text(encoding="utf-8"))
    assert document.commands, f"{path.name}: файл без команд"
    for command in document.commands:
        report = validate_command(command, registry=registry)
        assert report.ok, f"{path.name}: {report.user_message}"


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=[p.name for p in _EXAMPLE_FILES])
def test_example_command_has_no_dangerous_actions(path: Path) -> None:
    document = load_document(path.read_text(encoding="utf-8"))
    for command in document.commands:
        for location in command.blocks():
            block_type = location.block.type
            assert block_type not in _FORBIDDEN_BLOCKS, f"{path.name}: опасно — {block_type}"


@pytest.mark.parametrize("path", _EXAMPLE_FILES, ids=[p.name for p in _EXAMPLE_FILES])
def test_example_has_no_machine_paths(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "\\" not in text, f"{path.name}: обратный слэш выглядит как машинный путь"
    # Буква диска вида ``C:\`` или ``C:/``, но не схема URL (``https://``): второй
    # слэш отсекается отрицательным просмотром, иначе «s://» из ссылки даст ложное
    # срабатывание.
    drive = re.search(r"[A-Za-z]:[\\/](?![\\/])", text)
    assert drive is None, f"{path.name}: путь с буквой диска"


# ----------------------------------------------------------------------
# «Айрис из коробки»: движок активации по факту моделей
# ----------------------------------------------------------------------


def test_resolve_wake_prefers_openwakeword_when_airis_installed() -> None:
    installed = [
        _record("oww-melspectrogram", "wake"),
        _record("oww-embedding", "wake"),
        _record("oww-airis-ru", "wake"),
    ]
    assert resolve_wake_engine(installed) == {
        "voice.wake.enabled": True,
        "voice.wake.engine": "openwakeword",
    }


def test_resolve_wake_falls_back_to_vosk() -> None:
    installed = [_record("vosk-ru-small", "stt", engine="vosk", path="models/vosk")]
    result = resolve_wake_engine(installed)
    assert result["voice.wake.enabled"] is True
    assert result["voice.wake.engine"] == "vosk"
    assert result["voice.wake.options"] == {"model_path": "models/vosk"}


def test_resolve_wake_vosk_without_path_omits_model_path() -> None:
    installed = [_record("vosk-ru-small", "stt", engine="vosk")]
    result = resolve_wake_engine(installed)
    assert result["voice.wake.engine"] == "vosk"
    assert result["voice.wake.options"] == {}


def test_resolve_wake_disables_when_nothing_installed() -> None:
    assert resolve_wake_engine([]) == {"voice.wake.enabled": False}


@pytest.mark.parametrize(
    "installed_ids",
    [
        (),
        ("oww-melspectrogram",),
        ("oww-melspectrogram", "oww-embedding"),
        ("oww-embedding", "oww-airis-ru"),
        ("oww-airis-ru",),
    ],
)
def test_default_phrase_never_left_on_dead_openwakeword(installed_ids: tuple[str, ...]) -> None:
    # Обещание задачи: при любой НЕПОЛНОЙ установке openWakeWord активацию нельзя
    # оставить включённой на нём — иначе дефолтная фраза «айрис» останется без модели.
    installed = [_record(cid, "wake") for cid in installed_ids]
    result = resolve_wake_engine(installed)
    enabled = bool(result.get("voice.wake.enabled"))
    engine = result.get("voice.wake.engine")
    assert not (enabled and engine == "openwakeword")


def test_reconcile_wake_disables_when_no_models(manager: ConfigManager) -> None:
    assert manager.settings.voice.wake.enabled is True  # дефолт — openWakeWord
    reconcile_wake(manager, [])
    assert manager.settings.voice.wake.enabled is False


def test_reconcile_wake_keeps_openwakeword_when_airis_installed(manager: ConfigManager) -> None:
    reconcile_wake(manager, [])
    assert manager.settings.voice.wake.enabled is False
    installed = [
        _record("oww-melspectrogram", "wake"),
        _record("oww-embedding", "wake"),
        _record("oww-airis-ru", "wake"),
    ]
    reconcile_wake(manager, installed)
    assert manager.settings.voice.wake.enabled is True
    assert manager.settings.voice.wake.engine == "openwakeword"


# ----------------------------------------------------------------------
# показ мастера и состав минимального набора
# ----------------------------------------------------------------------


def test_should_run_onboarding_matrix(manager: ConfigManager) -> None:
    assert should_run_onboarding(manager) is True  # дефолт: показать
    manager.apply({"general.onboarding_completed": True})
    assert should_run_onboarding(manager) is False  # пройден — не показывать
    manager.apply({"general.onboarding_completed": False, "general.show_onboarding": False})
    assert should_run_onboarding(manager) is False  # выключен явно


def test_minimal_set_covers_wake_stt_and_voice() -> None:
    ids = set(MINIMAL_MODEL_IDS)
    # Все три части фразы «айрис» для openWakeWord.
    assert {"oww-melspectrogram", "oww-embedding", "oww-airis-ru"} <= ids
    # Распознавание речи и голос ответа.
    assert "gigaam-v3-ctc" in ids
    assert "piper-ru-irina" in ids
    # Без LLM: базовый сценарий работает и без языковой модели.
    assert not any("qwen" in cid or "llm" in cid for cid in ids)
