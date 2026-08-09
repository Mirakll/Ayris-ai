"""The contract every wake word engine implements.

A wake word engine answers one question - "did the user just say the word" - and
nothing else.  It does not own the microphone, it does not resample, it does not
cut the stream into frames of the size it needs, and it does not decide whether a
score is high enough.  All of that lives in
:mod:`ayris.audio.wake_word.manager`, for a practical reason: the three engines
disagree about almost every detail of their input.  openWakeWord wants blocks of
1280 samples, Porcupine wants exactly 512, Vosk takes whatever it is given, and
duplicating the resampler and the splitter three times would mean three places to
get the seam between two blocks wrong.

What an engine *does* own is its model.  :meth:`WakeWordEngine.load` is handed a
:class:`ModelSpec` - the phrase list with per-phrase sensitivities, the directory
models live in, and a credential when the vendor demands one - and everything
after that is per-frame work.

**Scores, thresholds and sensitivity.**  ``sensitivity`` is what the user sets in
the settings window, 0.0-1.0, and it reads the way a user expects: higher means
the assistant reacts more eagerly and gets it wrong more often.  Engines think in
the opposite direction - they emit a *score* and fire when it clears a threshold -
so :attr:`WakePhrase.threshold` does the conversion once, here, rather than in
each engine.  An engine reports raw scores; the manager compares them against the
threshold of the specific phrase.  That is the whole reason a variant can be made
deaf without touching the others.

**What ``process`` returns.**  ``None`` for the overwhelming majority of frames.
When something did score, the engine returns its best candidate and, in
:attr:`WakeDetection.scores`, whatever it knows about the other phrases in the
same frame - openWakeWord scores every model on every frame anyway, and throwing
that away would make a low-threshold variant unreachable behind a high-scoring
one.  Anything below :data:`MIN_THRESHOLD` is not worth reporting: no phrase can
be configured loosely enough for it to matter.

**Missing libraries are a message, not a crash.**  Every engine imports its
vendor package inside :meth:`WakeWordEngine.load`, never at module import time.
A user who never enables Porcupine must not need ``pvporcupine`` installed, and
the test suite has to stay collectable on a runner where the wheel did not build.
:meth:`WakeWordEngine.available` answers the settings window's question without
importing anything at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.core.errors import WakeWordError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "DEFAULT_SENSITIVITY",
    "ENGINE_ENTRYPOINTS",
    "MAX_THRESHOLD",
    "MIN_THRESHOLD",
    "PTT_PHRASE",
    "WAKE_SAMPLE_RATE",
    "ModelSpec",
    "WakeDetection",
    "WakePhrase",
    "WakeWordEngine",
    "create_engine",
    "engine_class",
    "engine_names",
    "normalise_phrase",
    "phrases_from",
]

#: Every engine here is trained at 16 kHz mono and accepts nothing else, which is
#: also what :mod:`ayris.audio.capture` normalises to.
WAKE_SAMPLE_RATE: Final = 16000

#: Neutral sensitivity, matching ``voice.wake.phrases[].sensitivity``.
DEFAULT_SENSITIVITY: Final = 0.5

#: Threshold at ``sensitivity == 1.0``.  Not zero: a threshold of zero fires on
#: silence, and a user dragging the slider to the end wants "very eager", not
#: "permanently triggered".
MIN_THRESHOLD: Final = 0.05

#: Threshold at ``sensitivity == 0.0``.  Not one: models rarely output a clean
#: 1.0, so a maximum of exactly one would mean "never fires" rather than "only on
#: a perfect match".
MAX_THRESHOLD: Final = 0.95

#: Phrase reported when activation came from the hotkey instead of the
#: microphone.  Deliberately not a pronounceable word, so it can never collide
#: with a variant a user adds.
PTT_PHRASE: Final = "<ptt>"

#: Engine name (the value of ``voice.wake.engine``) to ``module:Class``.
#: Resolved lazily by :func:`create_engine` so that importing this module costs
#: nothing and a broken engine module cannot take the others down with it.
ENGINE_ENTRYPOINTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openwakeword": "ayris.audio.wake_word.openwakeword:OpenWakeWordEngine",
        "porcupine": "ayris.audio.wake_word.porcupine:PorcupineEngine",
        "vosk": "ayris.audio.wake_word.vosk_kws:VoskKwsEngine",
    }
)


def normalise_phrase(text: str) -> str:
    """Fold a wake word to the form phrases are compared in.

    The same rule as ``voice.wake.phrases[].phrase`` in the settings model:
    collapse whitespace and lower-case.  Applied again here because phrases also
    arrive from worker parameters and from tests, which do not go through
    pydantic.
    """
    return " ".join(text.split()).lower()


@dataclass(frozen=True, slots=True)
class WakePhrase:
    """One variant of the wake word with its own sensitivity.

    Attributes:
        text: What the user says.  Folded by :func:`normalise_phrase`.
        sensitivity: 0.0-1.0 as shown in the settings window.  Higher fires more
            easily and produces more false positives.
        engine_model: Which model file recognises this phrase.  Empty means "the
            engine works it out from :attr:`text`", which is what Vosk does and
            what openWakeWord does when a model is named after the phrase.
        enabled: A variant switched off in the settings stays in the list so its
            sensitivity is not lost, but is not loaded into the engine.

    Raises:
        WakeWordError: If the text is empty or the sensitivity is out of range.
            The settings model validates the same two things; this catches a
            worker started with hand-written parameters.
    """

    text: str
    sensitivity: float = DEFAULT_SENSITIVITY
    engine_model: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        folded = normalise_phrase(self.text)
        if not folded:
            raise WakeWordError(
                "wake phrase must not be empty",
                user_message="Слово активации не может быть пустым.",
            )
        if not 0.0 <= self.sensitivity <= 1.0:
            raise WakeWordError(
                f"sensitivity of {folded!r} must be 0.0-1.0, got {self.sensitivity}",
                user_message="Чувствительность слова активации задаётся числом от 0 до 1.",
            )
        object.__setattr__(self, "text", folded)

    @property
    def threshold(self) -> float:
        """Score this phrase has to reach, derived from :attr:`sensitivity`.

        Linear between :data:`MAX_THRESHOLD` at sensitivity 0 and
        :data:`MIN_THRESHOLD` at sensitivity 1, so the default 0.5 lands on 0.5 -
        which is also the threshold openWakeWord's own documentation recommends.
        """
        return MAX_THRESHOLD - (MAX_THRESHOLD - MIN_THRESHOLD) * self.sensitivity

    def with_sensitivity(self, sensitivity: float) -> WakePhrase:
        """A copy at a different sensitivity, for the settings slider."""
        return replace(self, sensitivity=sensitivity)


@dataclass(frozen=True, slots=True)
class WakeDetection:
    """A frame the engine thinks contains a wake word.

    Attributes:
        phrase: Best-scoring variant, folded the same way as
            :attr:`WakePhrase.text`.
        score: Its raw score, 0.0-1.0.  Not yet compared with any threshold - the
            manager does that, because only it knows the current sensitivity.
        timestamp: Seconds of audio since the detector was last reset.  Audio
            time, not wall-clock: a machine that stalls for half a second must
            not have its debounce window slide with it.  Engines leave this at
            zero and the manager stamps it.
        engine: Name of the engine that produced the detection.
        scores: Every phrase the engine scored in this frame, when it knows more
            than one.  openWakeWord runs all its models on every frame, and
            dropping the rest would hide a variant with a low threshold behind a
            louder-scoring neighbour.
    """

    phrase: str
    score: float
    timestamp: float = 0.0
    engine: str = ""
    scores: Mapping[str, float] = field(default_factory=dict)

    def all_scores(self) -> Mapping[str, float]:
        """:attr:`scores` if the engine filled it, otherwise the winner alone."""
        return self.scores if self.scores else {self.phrase: self.score}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What an engine needs in order to load.

    Attributes:
        phrases: The variants to listen for, already filtered to the enabled
            ones.  An engine that keys on text (Vosk) reads
            :attr:`WakePhrase.text`; one that keys on model files
            (openWakeWord, Porcupine) reads :attr:`WakePhrase.engine_model` and
            falls back to the text.
        models_dir: Where model files live - ``<profile>/models/wake`` in
            production.  ``None`` means "wherever the library looks by default",
            which for openWakeWord is its bundled set.
        access_key: Vendor credential, read from the Windows credential store by
            the caller.  Only Porcupine uses one.
        options: Engine-specific extras that have no home in the settings model
            yet, e.g. ``inference_framework`` for openWakeWord.
    """

    phrases: tuple[WakePhrase, ...] = ()
    models_dir: Path | None = None
    access_key: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def enabled_phrases(self) -> tuple[WakePhrase, ...]:
        """The variants that should actually be loaded."""
        return tuple(phrase for phrase in self.phrases if phrase.enabled)

    def option(self, name: str, fallback: str = "") -> str:
        """Read an extra as a string, tolerating a missing key."""
        value = self.options.get(name)
        return fallback if value is None else str(value)


class WakeWordEngine(ABC):
    """Base class for a keyword spotter.

    Subclasses declare the shape of the audio they want
    (:attr:`sample_rate`, :attr:`frame_samples`) and implement three methods.
    Everything else - resampling, framing, thresholds, debounce, counters -
    belongs to :class:`~ayris.audio.wake_word.manager.WakeWordDetector`.

    Instances are not thread-safe.  One engine belongs to one detector, and in
    Ayris that is the audio worker's inference thread.
    """

    #: Value of ``voice.wake.engine`` this class implements.
    name: ClassVar[str] = "wake"

    #: Distribution to install, named in the error a user sees when it is
    #: missing.  Usually equal to :attr:`module`.
    package: ClassVar[str] = ""

    #: Top-level module imported at load time.  Checked by :meth:`available`
    #: without importing it.
    module: ClassVar[str] = ""

    #: Whether changing a sensitivity requires rebuilding the engine.  True for
    #: vendors that take thresholds at construction time and do their own
    #: comparison internally (Porcupine); false for engines that hand back a raw
    #: score the manager can re-threshold for free.
    reload_on_sensitivity: ClassVar[bool] = False

    #: Whether the vendor demands a paid or registered access key.  The manager
    #: reads it out of the Windows credential store and puts it in
    #: :attr:`ModelSpec.access_key`, so that no engine has to know about
    #: :mod:`ayris.core.secrets`.
    needs_credential: ClassVar[bool] = False

    __slots__ = ("_spec",)

    def __init__(self) -> None:
        self._spec: ModelSpec | None = None

    # ------------------------------------------------------------ description

    @property
    def sample_rate(self) -> int:
        """Rate the engine's frames must be at.  16 kHz for all three."""
        return WAKE_SAMPLE_RATE

    @property
    @abstractmethod
    def frame_samples(self) -> int:
        """Samples in one frame :meth:`process` accepts.

        Fixed per engine and not negotiable: Porcupine raises on anything other
        than its ``frame_length``, and openWakeWord's model has 1280 samples of
        input baked into its graph.
        """

    @property
    def frame_bytes(self) -> int:
        """Bytes in one frame, mono ``int16``."""
        return self.frame_samples * SAMPLE_WIDTH

    @property
    def frame_ms(self) -> float:
        """Length of one frame in milliseconds."""
        return self.frame_samples * 1000.0 / self.sample_rate

    @property
    def loaded(self) -> bool:
        """Whether a model is in memory and :meth:`process` will do anything."""
        return self._spec is not None

    @property
    def spec(self) -> ModelSpec | None:
        """The specification the engine was loaded with."""
        return self._spec

    @property
    def phrases(self) -> tuple[WakePhrase, ...]:
        """Variants currently loaded."""
        return self._spec.enabled_phrases if self._spec is not None else ()

    # ----------------------------------------------------------------- lifecycle

    @abstractmethod
    def load(self, spec: ModelSpec) -> None:
        """Bring the model into memory.

        Slow on purpose - a few hundred milliseconds for openWakeWord, seconds
        for a Vosk model - which is why it runs in the audio worker process and
        not on the UI thread.

        Raises:
            WakeWordError: The library is missing, the model file is not there,
                the credential was rejected, or the phrase list is empty.  The
                Russian message says which, because every one of those is
                something the user can fix in the settings window.
        """

    @abstractmethod
    def process(self, frame: bytes) -> WakeDetection | None:
        """Score exactly one frame of :attr:`frame_bytes` bytes.

        Returns:
            ``None`` when nothing reached :data:`MIN_THRESHOLD` - the normal
            answer, tens of times per second.  Otherwise the best candidate,
            with :attr:`WakeDetection.timestamp` left at zero for the manager to
            fill in.

        Raises:
            WakeWordError: If no model is loaded or the frame has the wrong
                length.  Both mean the caller is driving the engine directly
                instead of through the manager.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the model.  Must be safe to call twice and must not raise."""

    def reset(self) -> None:  # noqa: B027 - optional on purpose, see below
        """Forget whatever the engine carries between frames.

        A wake word is longer than one frame, so every engine here keeps some
        history - a rolling buffer of embeddings, a partial decoder hypothesis.
        When the stream restarts, that history belongs to audio the user is no
        longer speaking into, and scoring the new stream against it is how a
        wake word appears to fire the instant the microphone is switched.

        Deliberately not abstract: Porcupine exposes no way to clear its state,
        and forcing every engine to declare that it has nothing to forget would
        turn a genuine no-op into three lines of ceremony.
        """

    def update_phrases(self, phrases: Sequence[WakePhrase]) -> None:
        """Swap the phrase list without a full restart of the audio worker.

        The default reloads the model with the new list, which is correct for
        every engine and cheap for none of them.  An engine whose scores are
        independent of the thresholds overrides this to do nothing when only a
        sensitivity moved - see :attr:`reload_on_sensitivity`.

        Raises:
            WakeWordError: Same as :meth:`load`.
        """
        spec = self._spec
        if spec is None:
            raise WakeWordError(
                f"{self.name}: cannot update phrases before the model is loaded",
                user_message="Распознавание слова активации ещё не запущено.",
            )
        self.unload()
        self.load(replace(spec, phrases=tuple(phrases)))

    # ------------------------------------------------------------------ helpers

    @classmethod
    def available(cls) -> bool:
        """Whether the vendor library can be imported.

        Uses :func:`importlib.util.find_spec`, so asking the question does not
        pay for the answer: the settings window can grey out an engine without
        loading a machine learning runtime into the GUI process.
        """
        if not cls.module:
            return False
        try:
            return importlib.util.find_spec(cls.module) is not None
        except (ImportError, ValueError):
            # A namespace package shadowed by a broken install raises rather
            # than returning None.  Either way the import would fail.
            return False

    @classmethod
    def _import(cls, module: str | None = None) -> Any:
        """Import the vendor library.

        Raises:
            WakeWordError: If it is not installed, naming the package to install.
        """
        target = module or cls.module
        try:
            return importlib.import_module(target)
        except ImportError as exc:
            package = cls.package or target
            raise WakeWordError(
                f"{cls.name}: {target} is not installed: {exc}",
                user_message=(
                    f"Движок «{cls.name}» не установлен. "
                    f"Установите пакет {package} или выберите другой движок "
                    f"в настройках слова активации."
                ),
            ) from exc

    def _require_loaded(self) -> ModelSpec:
        """The loaded specification.

        Raises:
            WakeWordError: If :meth:`load` has not run.
        """
        spec = self._spec
        if spec is None:
            raise WakeWordError(
                f"{self.name}: no model is loaded",
                user_message="Распознавание слова активации ещё не запущено.",
            )
        return spec

    def _check_frame(self, frame: bytes) -> None:
        """Reject a frame of the wrong size.

        Raises:
            WakeWordError: Always, when the size is wrong.  The manager never
                produces one; a direct caller that does is better off knowing.
        """
        expected = self.frame_bytes
        if len(frame) != expected:
            raise WakeWordError(
                f"{self.name}: frame must be {expected} bytes, got {len(frame)}",
                user_message="Внутренняя ошибка распознавания слова активации.",
            )

    @staticmethod
    def _require_phrases(spec: ModelSpec) -> tuple[WakePhrase, ...]:
        """The enabled phrases.

        Raises:
            WakeWordError: If there are none - an engine with nothing to listen
                for would silently never fire, which looks exactly like a broken
                microphone.
        """
        phrases = spec.enabled_phrases
        if not phrases:
            raise WakeWordError(
                "wake word engine loaded without any enabled phrase",
                user_message=(
                    "Не задано ни одного варианта слова активации. "
                    "Добавьте вариант в настройках или включите Push-to-Talk."
                ),
            )
        return phrases

    def __repr__(self) -> str:
        return f"{type(self).__name__}(loaded={self.loaded}, phrases={len(self.phrases)})"


def engine_names() -> tuple[str, ...]:
    """Every engine name the settings window may offer."""
    return tuple(ENGINE_ENTRYPOINTS)


def engine_class(name: str) -> type[WakeWordEngine]:
    """Resolve an engine name to its class without constructing anything.

    The module is imported here and not at the top of this file, so that a
    vendor library with an expensive import - openWakeWord pulls in an ONNX
    runtime - costs nothing until somebody selects it, and so that an engine
    module which fails to import cannot stop the other two from working.

    Raises:
        WakeWordError: If the name is unknown or its module is broken.  Not
            silently downgraded to another engine: a user who picked Porcupine
            and got openWakeWord would be debugging the wrong component.
    """
    entrypoint = ENGINE_ENTRYPOINTS.get(name)
    if entrypoint is None:
        known = ", ".join(sorted(ENGINE_ENTRYPOINTS))
        raise WakeWordError(
            f"unknown wake word engine {name!r}, expected one of {known}",
            user_message=f"Неизвестный движок слова активации: {name}.",
        )
    module_name, _, attribute = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only a broken checkout
        raise WakeWordError(
            f"cannot import wake word engine {name!r}: {exc}",
            user_message=f"Не удалось загрузить движок слова активации «{name}».",
        ) from exc
    factory: type[WakeWordEngine] = getattr(module, attribute)
    return factory


def create_engine(name: str) -> WakeWordEngine:
    """Build the engine ``voice.wake.engine`` names.

    Raises:
        WakeWordError: Same as :func:`engine_class`.
    """
    return engine_class(name)()


def phrases_from(items: Iterable[Mapping[str, Any]]) -> tuple[WakePhrase, ...]:
    """Build phrases from worker parameters.

    The audio worker receives ``[{"phrase": ..., "sensitivity": ...}, ...]``
    over the pipe - plain JSON-ish dictionaries, because a pydantic model cannot
    be pickled into a spawned process without importing the settings module
    there.  Unknown keys are ignored and a missing sensitivity falls back to the
    default, so an older supervisor can still start a newer worker.

    Raises:
        WakeWordError: If an entry has an empty phrase or an out-of-range
            sensitivity.
    """
    built: list[WakePhrase] = []
    seen: set[str] = set()
    for item in items:
        text = str(item.get("phrase", item.get("text", "")))
        if not normalise_phrase(text):
            continue
        raw = item.get("sensitivity", DEFAULT_SENSITIVITY)
        phrase = WakePhrase(
            text=text,
            sensitivity=float(raw) if isinstance(raw, int | float) else DEFAULT_SENSITIVITY,
            engine_model=str(item.get("model", item.get("engine_model", ""))),
            enabled=bool(item.get("enabled", True)),
        )
        if phrase.text in seen:
            # Two identical variants would double every event they produce.
            continue
        seen.add(phrase.text)
        built.append(phrase)
    return tuple(built)
