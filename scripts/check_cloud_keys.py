"""Проверка облачных ключей распознавания речи на живых сервисах.

Скрипт запускается на Windows пользователя, а не в песочнице: доменов Яндекса,
Google, Azure и OpenAI нет в allowlist песочницы, да и ключ не должен уезжать с
машины владельца. Обёртка ``check-keys.ps1`` приносит портативный CPython 3.12 и
колёса, после чего запускает этот файл.

Порядок работы:

1. для каждого выбранного провайдера ключ спрашивается скрытым вводом и кладётся в
   диспетчер учётных данных Windows под тем же именем, которое читает Ayris
   (:data:`~ayris.core.secrets.KNOWN_SLOTS`);
2. пишется несколько секунд с микрофона — или берётся WAV из ``--wav``;
3. тот же движок, что и в приложении, делает один живой запрос на провайдера;
4. печатается распознанный текст либо текст ошибки.

Ключ не передаётся аргументом командной строки: он попал бы в историю PowerShell.
В вывод он тоже не попадает — только маска из :func:`ayris.core.secrets.mask`.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from ayris.audio.stt.base import STT_SAMPLE_RATE, AudioBuffer, SttOptions  # noqa: E402
from ayris.audio.stt.cloud_base import as_wav, create_cloud_engine  # noqa: E402
from ayris.core.errors import AyrisError  # noqa: E402
from ayris.core.secrets import KNOWN_SLOTS, SecretsStore, mask, reset_secrets  # noqa: E402

if TYPE_CHECKING:
    from ayris.audio.stt.base import TranscriptResult

#: Провайдеры в порядке проверки.
PROVIDERS: Final = ("yandex", "google", "azure", "openai")

#: Несекретные параметры, без которых запрос не уходит. Лежат рядом со скриптом в
#: открытом виде: это не ключи, а адресация — каталог облака и регион ресурса.
EXTRAS: Final = {
    "yandex": ("folder_id", "Идентификатор каталога (folderId, вида b1g…)"),
    "azure": ("region", "Регион ресурса Speech (например westeurope)"),
}

_PARAMS_PATH: Final = _ROOT / "_tools" / "cloud_check_params.json"
_RECORD_PATH: Final = _ROOT / "_tools" / "last_record.wav"
_FALLBACK_WAV: Final = _ROOT / "tests" / "fixtures" / "audio" / "stt_phrase.wav"
_RECORD_SECONDS: Final = 5.0
_PHRASE: Final = "Айрис, открой браузер"


def _say(text: str = "") -> None:
    """Вывести строку. Вместо ``print``, чтобы кодировка задавалась здесь."""
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()


def _utf8_output() -> None:
    """Переключить вывод на UTF-8.

    Портативный Python с файлом ``._pth`` игнорирует переменные окружения, в том
    числе ``PYTHONIOENCODING``. Без этого перенаправление вывода в файл падает на
    первой же кириллической строке: локаль там cp1251.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _load_params() -> dict[str, str]:
    """Прочитать несекретные параметры прошлого запуска."""
    try:
        raw = json.loads(_PARAMS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(key): str(value) for key, value in raw.items()} if isinstance(raw, dict) else {}


def _save_params(params: dict[str, str]) -> None:
    """Запомнить folderId и регион, чтобы не спрашивать их второй раз."""
    _PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PARAMS_PATH.write_text(
        json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _store() -> SecretsStore:
    """Открыть диспетчер учётных данных Windows.

    Бэкенд выбирается явно, а не через точки входа ``keyring``: в портативном
    Python метаданные пакетов могут не читаться, и тогда ``keyring`` молча
    отдаёт заглушку, которая «сохраняет» ключ в никуда.
    """
    import keyring
    from keyring.backends.Windows import WinVaultKeyring

    if not WinVaultKeyring.viable:  # pragma: no cover - только на не-Windows
        raise SystemExit("Диспетчер учётных данных Windows недоступен на этой системе.")
    backend = WinVaultKeyring()  # type: ignore[no-untyped-call]
    keyring.set_keyring(backend)
    store = SecretsStore(backend=backend)
    reset_secrets(store)  # движки читают ключ через get_secrets()
    return store


def _ask_key(ref: str, store: SecretsStore) -> bool:
    """Спросить ключ скрытым вводом и сохранить его.

    Returns:
        ``False``, если пользователь ввёл пустую строку — провайдер тогда
        пропускается.
    """
    slot = KNOWN_SLOTS[ref]
    _say(f"  {slot.title}: {slot.hint}")
    value = getpass.getpass("  ключ (ввод скрыт, Enter — пропустить): ").strip()
    if not value:
        return False
    store.save(ref, value)
    _say(f"  сохранено в хранилище Windows как «{ref}»: {mask(value)}")
    return True


def _ask_extra(provider: str, params: dict[str, str], *, force: bool) -> bool:
    """Спросить несекретный параметр провайдера, если он ещё не известен.

    Returns:
        Есть ли теперь значение. Без него запрос всё равно не уйдёт, так что
        провайдер лучше пропустить, чем идти в предсказуемую ошибку.
    """
    field = EXTRAS.get(provider)
    if field is None:
        return True
    name, hint = field
    key = f"{provider}_{name}"
    known = params.get(key, "")
    if known and not force:
        _say(f"  {name}: {known}")
        return True
    prompt = f"  {hint}: " if not known else f"  {hint} [{known}]: "
    value = input(prompt).strip() or known
    if not value:
        _say(f"  без {name} запрос не уйдёт")
        value = input(prompt).strip()
    if not value:
        return False
    params[key] = value
    return True


def _record(seconds: float) -> AudioBuffer | None:
    """Записать моно 16 кГц с устройства ввода по умолчанию.

    Читается сырой поток, а не ``sounddevice.rec``: тот отдаёт массив numpy, а
    numpy сюда тащить не за чем.

    Returns:
        Буфер, либо ``None``, если звуковой ввод недоступен — тогда вызывающий
        берёт файл.
    """
    try:
        import sounddevice
    except Exception as exc:  # причина важнее типа
        _say(f"  микрофон недоступен ({type(exc).__name__}: {exc})")
        return None

    chunks: list[bytes] = []

    def collect(data: object, frames: int, moment: object, status: object) -> None:
        del frames, moment, status
        chunks.append(bytes(memoryview(data)))  # type: ignore[arg-type]

    try:
        stream = sounddevice.RawInputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=320,
            callback=collect,
        )
        _say(f"  говорите: «{_PHRASE}» ({seconds:.0f} с)")
        with stream:
            time.sleep(seconds)
    except Exception as exc:  # любая ошибка драйвера
        _say(f"  запись не удалась ({type(exc).__name__}: {exc})")
        return None

    audio = AudioBuffer(pcm=b"".join(chunks), sample_rate=STT_SAMPLE_RATE, channels=1)
    if audio.is_empty:
        _say("  с микрофона не пришло ни одного кадра")
        return None
    return audio


def _audio(args: argparse.Namespace) -> AudioBuffer:
    """Взять звук: указанный файл, микрофон или фикстуру."""
    if args.wav:
        path = Path(args.wav)
        _say(f"Звук из файла: {path}")
        return AudioBuffer.from_wav(path)

    _say("Запись с микрофона")
    audio = _record(args.seconds)
    if audio is not None:
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECORD_PATH.write_bytes(as_wav(audio))
        _say(
            f"  записано {audio.duration_ms / 1000:.1f} с, "
            f"громкость {audio.rms_dbfs:.1f} dBFS, копия: {_RECORD_PATH}"
        )
        if audio.is_silent():
            _say("  тишина: проверьте, что выбран нужный микрофон и он не выключен")
        return audio

    _say(f"  беру синтетическую фикстуру {_FALLBACK_WAV.name}: она проверит только ключ,")
    _say("  распознанный текст на ней будет пустым или случайным")
    return AudioBuffer.from_wav(_FALLBACK_WAV)


def _options(provider: str, params: dict[str, str]) -> SttOptions:
    """Собрать параметры движка: имя записи с ключом плюс несекретная адресация."""
    extra: dict[str, str] = {"credential_ref": provider}
    field = EXTRAS.get(provider)
    if field is not None:
        name, _ = field
        value = params.get(f"{provider}_{name}", "")
        if value:
            extra[name] = value
    return SttOptions(extra=extra)


def _check(provider: str, audio: AudioBuffer, params: dict[str, str]) -> bool:
    """Сделать один живой запрос и напечатать, чем он кончился.

    Returns:
        Успешен ли запрос. Пустой текст при HTTP 200 считается успехом: ключ
        приняли, а что именно услышали — вопрос к звуку, не к ключу.
    """
    engine = create_cloud_engine(provider)
    try:
        engine.load(Path(), _options(provider, params))
    except AyrisError as exc:
        _say(f"  ✘ {exc.user_message}")
        return False

    try:
        result: TranscriptResult = engine.transcribe(audio)
    except AyrisError as exc:
        _say(f"  ✘ {exc.user_message}")
        _say(f"    подробности: {exc.technical}")
        return False
    except Exception as exc:  # показать любой сюрприз, а не падать
        _say(f"  ✘ неожиданная ошибка: {type(exc).__name__}: {exc}")
        return False
    finally:
        engine.unload()

    if result.is_empty:
        _say(f"  ✔ ключ принят, но текста нет (за {result.inference_ms:.0f} мс)")
        return True
    _say(f"  ✔ «{result.text}»")
    _say(
        f"    уверенность {result.confidence:.2f}, модель {result.model or '—'}, "
        f"{result.inference_ms:.0f} мс"
    )
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_cloud_keys",
        description="Сохранить облачные ключи в хранилище Windows и проверить их живым запросом.",
    )
    parser.add_argument(
        "--providers",
        default="all",
        help=f"через запятую из: {', '.join(PROVIDERS)}; по умолчанию все",
    )
    parser.add_argument("--wav", default="", help="взять звук из файла вместо микрофона")
    parser.add_argument(
        "--seconds",
        type=float,
        default=_RECORD_SECONDS,
        help=f"сколько писать с микрофона, по умолчанию {_RECORD_SECONDS:.0f}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="спрашивать ключ заново, даже если он уже сохранён",
    )
    return parser.parse_args(argv)


def _chosen(raw: str) -> tuple[str, ...]:
    """Разобрать ``--providers``."""
    if raw.strip().lower() in {"all", "все", ""}:
        return PROVIDERS
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = [name for name in names if name not in PROVIDERS]
    if unknown:
        raise SystemExit(f"Неизвестный провайдер: {', '.join(unknown)}")
    return names


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    args = _parse_args(argv)
    providers = _chosen(args.providers)
    store = _store()
    params = _load_params()

    _say("Ключи сохраняются в диспетчере учётных данных Windows (служба «Ayris»).")
    _say("Ввод ключа скрыт, в вывод и в логи он не попадает.")
    _say("")

    ready: list[str] = []
    for provider in providers:
        _say(f"[{KNOWN_SLOTS[provider].title}]")
        if args.force or not store.has(provider):
            if not _ask_key(provider, store):
                _say("  пропущен")
                _say("")
                continue
        else:
            _say(f"  ключ уже сохранён ({mask(store.get(provider))}), --force чтобы заменить")
        if not _ask_extra(provider, params, force=args.force):
            _say("  пропущен")
            _say("")
            continue
        ready.append(provider)
        _say("")

    if not ready:
        _say("Ни одного ключа не задано, проверять нечего.")
        return 1

    _save_params(params)
    audio = _audio(args)
    _say("")

    failed: list[str] = []
    for provider in ready:
        _say(f"[{KNOWN_SLOTS[provider].title}]")
        if not _check(provider, audio, params):
            failed.append(provider)
        _say("")

    good = [name for name in ready if name not in failed]
    _say(f"Работают: {', '.join(good) or '—'}")
    if failed:
        _say(f"Не работают: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
