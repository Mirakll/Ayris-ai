"""Piper: the default voice, and the reason local synthesis is fast enough.

Piper runs a VITS model through ONNX Runtime on the CPU. A medium Russian voice
is about 60 MB on disk, takes roughly 400 ms to load and then synthesizes faster
than real time on any machine that can run this assistant at all - which is what
makes the specification's "speech starts under 500 ms" achievable without a GPU.

**A voice is two files.** ``ru_RU-irina-medium.onnx`` and its
``ru_RU-irina-medium.onnx.json``, which carries the sample rate, the phoneme map
and the model's default inference scales. Piper's own loader assumes the config
sits next to the model with ``.json`` appended; :meth:`PiperTtsEngine.load`
accepts that layout and also a bare ``.json`` next to a differently-named model,
because users renaming downloaded voices is not a hypothetical.

**Where voices come from.** ``models/tts`` in the profile. Anything with an
``.onnx`` file and a readable config shows up in :meth:`PiperTtsEngine.voices`,
so a user who drops a voice from the Piper release page into that folder finds it
in the settings without Ayris needing to know the name in advance.

**Speed is length_scale, and it is inverted.** Piper stretches phoneme durations
by ``length_scale``: 2.0 is half speed. The multiplier the settings expose runs
the other way - 2.0 means twice as fast - so ``length_scale = base / speed``. The
base comes from the voice's own config rather than 1.0, because some voices ship
tuned to a value other than one and overriding it would change how they sound as
well as how fast.

**Pitch is done on the samples, not in the model.** VITS has no pitch parameter,
so «Тон» is applied the way Silero does it: the voice is synthesized a touch
longer (``length_scale`` multiplied by the pitch factor) and then resampled back
by that same factor, which shifts the frequency while returning the phrase to the
tempo the speed setting asked for. It reuses
:class:`~ayris.audio.capture.Resampler` rather than a phase vocoder - one extra
interpolation pass per phrase, and only when the control is off neutral - so it is
cheap and introduces no new dependency. The trade-off is honest: this is plain
resampling, not formant-preserving, so a large shift thins or thickens the timbre
the way speeding a tape up does. Left at 1.0 it costs nothing and changes nothing.

**Expressiveness is the model's ``noise_scale``.** VITS samples its prosody from
noise; turning that scale down flattens the delivery, up livens it. It is a
load-time option (:attr:`~ayris.audio.tts.base.TtsOptions.noise_scale`), so a
change takes effect when the engine is next loaded - which is why the setting is
``RestartScope.TTS``. ``None`` leaves the value the voice config shipped with.

Every import of ``piper`` happens inside :meth:`load`. It pulls in ONNX Runtime
and NumPy, neither of which a Silero user should pay for, and neither of which is
installed on the CI runner - a module-level import would break collection of
every test that merely mentions TTS.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import suppress
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.tts.base import (
    AudioChunk,
    TtsEngine,
    TtsOptions,
    VoiceSpec,
    concat_chunks,
)
from ayris.core.errors import TtsError
from ayris.core.paths import get_paths, is_ascii_path, native_path, native_path_problem

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["PIPER_SAMPLE_RATE", "PiperTtsEngine", "find_voice_files"]

_log = logging.getLogger(__name__)

#: What the medium Russian voices produce. Read from the voice config when it is
#: present; this is the fallback for a config that omits it.
PIPER_SAMPLE_RATE: Final = 22050

#: Model file extension.
_MODEL_SUFFIX: Final = ".onnx"

#: Length scale outside this range is either unintelligible or so slow the user
#: assumes it hung. Wider than the settings allow on purpose - this catches a
#: voice config with an odd base scale, not a user mistake.
_MIN_LENGTH_SCALE: Final = 0.25
_MAX_LENGTH_SCALE: Final = 4.0

#: Pitch this close to 1.0 is treated as neutral: the resample pass costs real
#: time on a long phrase and a 2% shift is inaudible. Mirrors Silero's own
#: threshold so both engines round «Тон» the same way.
_PITCH_EPSILON: Final = 0.02

#: Where the ``piper`` package keeps the espeak-ng data it ships with.
_ESPEAK_DATA_DIR_NAME: Final = "espeak-ng-data"

#: Present in every espeak data folder; used to tell a finished copy from a
#: half-written one.
_ESPEAK_MARKER: Final = "phontab"


def find_voice_files(model: Path) -> tuple[Path, Path]:
    """Locate the model and its config.

    Accepts either the ``.onnx`` file or the ``.json`` beside it, and tolerates
    both naming conventions: ``voice.onnx.json`` (what Piper ships) and
    ``voice.json`` (what a user gets after renaming).

    Raises:
        TtsError: Neither file can be found.
    """
    model_path = model if model.suffix == _MODEL_SUFFIX else model.with_suffix(_MODEL_SUFFIX)
    if not model_path.is_file():
        raise TtsError(
            f"piper: model {model_path} not found",
            user_message=(
                f"Файл голоса не найден:\n{model_path}\n"
                "Скачайте голос в настройках или укажите другой."
            ),
        )
    candidates = (
        Path(f"{model_path}.json"),
        model_path.with_suffix(".json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return model_path, candidate
    raise TtsError(
        f"piper: config for {model_path.name} not found",
        user_message=(
            f"Рядом с голосом «{model_path.name}» нет файла настроек .json. "
            "Скачайте голос целиком: модель и конфиг идут парой."
        ),
    )


class PiperTtsEngine(TtsEngine):
    """Piper VITS synthesis through ONNX Runtime."""

    name: ClassVar[str] = "piper"
    package: ClassVar[str] = "piper-tts"
    module: ClassVar[str] = "piper"
    optional: ClassVar[bool] = False
    #: ONNX Runtime holds the graph plus its arena; measured at roughly 2.5x the
    #: file for a medium voice, rounded up because the arena grows with use.
    memory_factor: ClassVar[float] = 3.0
    native_sample_rate: ClassVar[int] = PIPER_SAMPLE_RATE

    __slots__ = ("_base_length_scale", "_use_cuda", "_voice_object")

    def __init__(self) -> None:
        super().__init__()
        self._voice_object: Any | None = None
        self._base_length_scale = 1.0
        self._use_cuda = False

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """The language of the loaded voice, since one model speaks one."""
        voice = self.voice
        return (voice.language,) if voice is not None and voice.language else ()

    @property
    def device(self) -> str:
        """``cuda`` only when the user asked for it and ONNX accepted."""
        return "cuda" if self._use_cuda else "cpu"

    # ----------------------------------------------------------- enumeration

    @classmethod
    def voices(cls, directory: Path | None = None) -> tuple[VoiceSpec, ...]:
        """Every usable voice in the model directory.

        A model without a config, or with a config that will not parse, is
        skipped silently: half a voice in the combo box would only produce an
        error when the user selected it.
        """
        root = directory if directory is not None else get_paths().tts_models_dir
        try:
            models = sorted(root.rglob(f"*{_MODEL_SUFFIX}"))
        except OSError as exc:
            _log.debug("piper: каталог голосов %s недоступен: %s", root, exc)
            return ()

        found: list[VoiceSpec] = []
        for model in models:
            try:
                model_path, config_path = find_voice_files(model)
            except TtsError:
                continue
            config = _read_config(config_path)
            if config is None:
                continue
            found.append(
                VoiceSpec(
                    engine=cls.name,
                    voice_id=model_path.stem,
                    path=str(model_path),
                    language=_config_language(config),
                    display_name=_config_display_name(config, model_path.stem),
                    sample_rate=_config_sample_rate(config),
                )
            )
        return tuple(found)

    # ------------------------------------------------------------- lifecycle

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        """Open the ONNX session for one voice.

        The previous voice is released first: two VITS graphs in one process is
        half a gigabyte of arena for no benefit, since only one can be speaking.

        Raises:
            TtsError: ``piper-tts`` is not installed, the files are missing, or
                ONNX Runtime refused the model.
        """
        model_path, config_path = find_voice_files(_resolve_model_path(voice))
        config = _read_config(config_path)
        if config is None:
            raise TtsError(
                f"piper: cannot parse {config_path}",
                user_message=(
                    f"Файл настроек голоса повреждён:\n{config_path.name}\n"
                    "Скачайте голос заново."
                ),
            )

        piper = self._import()
        self.unload()

        use_cuda = options.gpu == "cuda"
        try:
            loaded = piper.PiperVoice.load(
                str(model_path),
                config_path=str(config_path),
                use_cuda=use_cuda,
                **_espeak_argument(piper),
            )
        except Exception as exc:  # onnxruntime raises its own exception types
            raise TtsError(
                f"piper: cannot load {model_path.name}: {exc}",
                user_message=(
                    f"Не удалось загрузить голос «{model_path.stem}». "
                    "Возможно, файл повреждён или не хватает памяти."
                ),
            ) from exc

        self._voice_object = loaded
        self._use_cuda = use_cuda
        self._base_length_scale = _config_length_scale(config)
        self._voice = VoiceSpec(
            engine=self.name,
            voice_id=voice.voice_id or model_path.stem,
            path=str(model_path),
            language=voice.language or _config_language(config),
            display_name=voice.display_name or _config_display_name(config, model_path.stem),
            sample_rate=_config_sample_rate(config),
        )
        self._options = options
        _log.info(
            "piper: голос %s загружен, %d Гц, устройство %s",
            self._voice.voice_id,
            self.sample_rate,
            self.device,
        )

    def unload(self) -> None:
        """Drop the ONNX session. Safe to call twice; never raises."""
        self._voice_object = None
        self._voice = None
        self._base_length_scale = 1.0

    # ------------------------------------------------------------- synthesis

    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        """Run one phrase through the model.

        Piper yields raw ``int16`` per internal sentence; they are joined here
        because the caller has already decided what one phrase is, and handing
        back Piper's own sub-splits would put pauses where
        :mod:`ayris.audio.tts.sentence_split` decided there were none.

        Raises:
            TtsError: The session failed mid-synthesis.
        """
        return concat_chunks(list(self._stream_raw(text, speed, pitch)))

    def synthesize_stream(
        self,
        text: str,
        voice: VoiceSpec | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield audio as Piper produces it, one internal sentence at a time.

        Overridden because Piper streams natively: the base class would wait for
        a whole sentence, and for a long one this starts the sound sooner still.
        """
        from ayris.audio.tts.base import clamp_pitch, clamp_speed

        if not text.strip():
            return
        self._ensure_voice(voice)
        resolved_speed = clamp_speed(self._options.speed if speed is None else speed)
        resolved_pitch = clamp_pitch(self._options.pitch if pitch is None else pitch)
        for chunk in self._stream_raw(text, resolved_speed, resolved_pitch):
            if not chunk.empty:
                yield chunk

    def _stream_raw(self, text: str, speed: float, pitch: float) -> Iterator[AudioChunk]:
        """Piper's raw stream, wrapped in :class:`AudioChunk`.

        Raises:
            TtsError: No voice loaded, or ONNX Runtime failed.
        """
        loaded = self._voice_object
        if loaded is None:
            self._require_loaded()  # raises with the right message
            return
        rate = self.sample_rate
        shift = abs(pitch - 1.0) > _PITCH_EPSILON
        length_scale = self._length_scale(speed, pitch)
        try:
            for pcm in self._vendor_stream(loaded, text, length_scale):
                shaped = _resample_pitch(pcm, rate, pitch) if shift else pcm
                yield AudioChunk(pcm=shaped, sample_rate=rate, channels=1)
        except TtsError:
            raise
        except Exception as exc:  # onnxruntime and piper-phonemize both raise
            raise TtsError(
                f"piper: synthesis failed: {exc}",
                user_message="Не удалось озвучить ответ: движок Piper вернул ошибку.",
            ) from exc

    def _vendor_stream(self, loaded: Any, text: str, length_scale: float) -> Iterator[bytes]:
        """Yield raw PCM from whichever synthesis API this Piper offers.

        Piper 1.3 replaced ``synthesize_stream_raw(text, length_scale=...)``,
        which yielded plain PCM, with ``synthesize(text, syn_config=...)``,
        which yields ``AudioChunk`` objects carrying their own sample rate.  We
        speak both: the old one is what ships in Debian and in the frozen
        builds people already have, the new one is what ``pip install
        piper-tts`` gives today.

        Expressiveness (``noise_scale``) rides only the new API - the legacy
        stream took no such argument, so an old Piper simply keeps the voice's
        own value there.
        """
        legacy = getattr(loaded, "synthesize_stream_raw", None)
        if legacy is not None:
            for pcm in legacy(text, length_scale=length_scale):
                yield bytes(pcm)
            return
        piper = self._import()
        config_type = getattr(piper, "SynthesisConfig", None)
        if config_type is None:  # pragma: no cover - no such Piper release
            raise TtsError(
                "piper: neither synthesize_stream_raw nor SynthesisConfig is available",
                user_message=(
                    "Установленная версия Piper не поддерживается. "
                    "Обновите пакет piper-tts или выберите другой движок в настройках."
                ),
            )
        syn_config = config_type(**self._synthesis_kwargs(length_scale))
        for chunk in loaded.synthesize(text, syn_config=syn_config):
            yield bytes(chunk.audio_int16_bytes)

    def _synthesis_kwargs(self, length_scale: float) -> dict[str, float]:
        """Arguments for ``SynthesisConfig``: always the tempo, sometimes the noise.

        ``noise_scale`` is passed only when the settings set it, so a voice keeps
        the value baked into its own config until the user moves the slider.
        """
        kwargs: dict[str, float] = {"length_scale": length_scale}
        noise_scale = self._options.noise_scale
        if noise_scale is not None:
            kwargs["noise_scale"] = noise_scale
        return kwargs

    def _length_scale(self, speed: float, pitch: float = 1.0) -> float:
        """Convert the speed multiplier to Piper's duration stretch.

        When «Тон» is off neutral the scale is multiplied by the pitch factor:
        the phrase is synthesized proportionally longer so that resampling it by
        that factor afterwards (which shifts pitch *and* time together) lands back
        on the tempo the speed asked for. Clamped once, after that multiply, so an
        extreme speed-and-pitch combination stays intelligible rather than doubling
        past the limit.
        """
        scale = self._base_length_scale / max(speed, 0.01)
        if abs(pitch - 1.0) > _PITCH_EPSILON:
            scale *= pitch
        return min(max(scale, _MIN_LENGTH_SCALE), _MAX_LENGTH_SCALE)


# ------------------------------------------------------------------- shaping


def _resample_pitch(pcm: bytes, rate: int, factor: float) -> bytes:
    """Shift pitch by ``factor`` by resampling and claiming the same rate.

    ``factor`` above 1.0 yields fewer samples - higher and shorter when played at
    ``rate`` - which is why :meth:`PiperTtsEngine._length_scale` lengthens the
    synthesis by the same factor first, so the phrase comes out at its intended
    tempo. Reuses :class:`~ayris.audio.capture.Resampler`, imported here rather
    than at module load so a Piper user pays for it only when «Тон» is off neutral.
    """
    source = max(1, int(round(rate * factor)))
    if source == rate or not pcm:
        return pcm
    from ayris.audio.capture import Resampler

    return Resampler(source, rate).process(pcm)


# --------------------------------------------------------------- espeak data


def _espeak_argument(piper: Any) -> dict[str, str]:
    """``espeak_data_dir`` for :meth:`PiperVoice.load`, when it needs fixing.

    Piper phonemizes Russian through espeak-ng, and espeak-ng takes the path to
    its data as a narrow string: a folder whose name is not Latin never opens.
    That folder ships inside the ``piper`` package, so whether it opens has
    nothing to do with the user's profile and everything to do with where Python
    is installed - and when it fails, espeak calls ``exit()`` from C and takes
    the process with it, after printing the path of the machine it was built on.

    So hand Piper an ASCII spelling of its own data folder when the plain one is
    unsafe.  Older Piper releases have no such parameter; there we stay silent
    rather than raise, because the same release also has the old synthesis API
    and it is not our business to demand an upgrade.
    """
    data_dir = Path(piper.__file__).parent / _ESPEAK_DATA_DIR_NAME
    if not data_dir.is_dir() or is_ascii_path(data_dir):
        return {}
    if "espeak_data_dir" not in signature(piper.PiperVoice.load).parameters:
        return {}
    safe = native_path(data_dir) or _ascii_espeak_copy(data_dir)
    if safe is None:
        raise TtsError(
            f"piper: espeak data at {data_dir} has no ASCII spelling",
            user_message=native_path_problem(data_dir, what="данные espeak для Piper"),
        )
    return {"espeak_data_dir": safe}


def _ascii_espeak_copy(data_dir: Path) -> str | None:
    """Copy espeak's data somewhere with an ASCII name, and say where.

    Nineteen megabytes, once, and only for an installation Python cannot even
    name in ASCII - which is a Python unpacked into a folder like
    ``D:\\Программы`` on a disk with 8.3 names switched off.  Copying beats
    refusing: without this, Russian synthesis is simply unavailable.
    """
    for parent in _copy_destinations():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - an unwritable cache is the next case
            continue
        # Asked after mkdir on purpose: Windows has no short name for a
        # directory that does not exist yet.
        target = native_path(parent)
        if target is None:
            continue
        destination = Path(target) / _ESPEAK_DATA_DIR_NAME
        if (destination / _ESPEAK_MARKER).is_file():
            return str(destination)
        staging = destination.with_name(f"{_ESPEAK_DATA_DIR_NAME}.partial")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        try:
            shutil.copytree(data_dir, staging)
            staging.rename(destination)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            _log.warning("piper: не удалось скопировать данные espeak в %s: %s", parent, exc)
            continue
        _log.info(
            "piper: данные espeak скопированы в %s (в пути установки есть не-латиница)",
            destination,
        )
        return str(destination)
    return None


def _copy_destinations() -> Iterator[Path]:
    """Where a copy of espeak's data may live, best first.

    The profile cache is the right home for a derived copy of somebody else's
    data.  But a profile can itself be unnameable - a portable build on a stick
    called ``Флешка`` - and that install still needs Russian speech, so the
    temporary directory follows: it is on the system disk, where Windows leaves
    8.3 names on by default.
    """
    with suppress(Exception):  # paths refuse only if the profile is broken
        yield get_paths().cache_dir
    yield Path(tempfile.gettempdir())


# ------------------------------------------------------------- config reading


def _resolve_model_path(voice: VoiceSpec) -> Path:
    """Where the voice's model file should be.

    An explicit path wins; otherwise the identifier is looked up in the profile's
    ``models/tts``, which is what a config file holding just ``ru_RU-irina-medium``
    means.
    """
    if voice.path:
        return Path(voice.path)
    # The id is normally a stem, but a voice added through «Своя модель» keeps its
    # file name, so it may already carry «.onnx». Appending unconditionally gave
    # «voice.onnx.onnx», which no file matches - mirror find_voice_files and only
    # add the suffix when it is missing.
    stem = voice.voice_id
    name = stem if stem.endswith(_MODEL_SUFFIX) else f"{stem}{_MODEL_SUFFIX}"
    return get_paths().tts_models_dir / name


def _read_config(path: Path) -> dict[str, Any] | None:
    """Parse a voice config, or ``None`` if it is unreadable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, ValueError) as exc:
        _log.debug("piper: конфиг %s не читается: %s", path.name, exc)
        return None
    return parsed if isinstance(parsed, dict) else None


def _config_sample_rate(config: dict[str, Any]) -> int:
    """Rate the voice produces, from ``audio.sample_rate``."""
    audio = config.get("audio")
    if isinstance(audio, dict):
        rate = audio.get("sample_rate")
        if isinstance(rate, int) and rate > 0:
            return rate
    return PIPER_SAMPLE_RATE


def _config_length_scale(config: dict[str, Any]) -> float:
    """The voice's own duration scale, the baseline speed multiplies against."""
    inference = config.get("inference")
    if isinstance(inference, dict):
        scale = inference.get("length_scale")
        if isinstance(scale, int | float) and not isinstance(scale, bool) and scale > 0:
            return float(scale)
    return 1.0


def _config_language(config: dict[str, Any]) -> str:
    """Two-letter code from ``language.code`` or the espeak voice."""
    language = config.get("language")
    if isinstance(language, dict):
        code = language.get("code") or language.get("family")
        if isinstance(code, str) and code:
            return code.split("_")[0].split("-")[0].lower()
    espeak = config.get("espeak")
    if isinstance(espeak, dict):
        voice = espeak.get("voice")
        if isinstance(voice, str) and voice:
            return voice.split("-")[0].lower()
    return "ru"


def _config_display_name(config: dict[str, Any], fallback: str) -> str:
    """Human-readable name: ``irina (ru, medium)`` from the dataset fields."""
    dataset = config.get("dataset")
    quality = config.get("quality")
    if not isinstance(dataset, str) or not dataset:
        return fallback
    language = _config_language(config)
    if isinstance(quality, str) and quality:
        return f"{dataset} ({language}, {quality})"
    return f"{dataset} ({language})"
