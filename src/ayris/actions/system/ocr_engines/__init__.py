"""Choosing an OCR engine: what is asked for, what is actually installed.

Three engines, one interface, and one function worth reading:
:func:`select_engine`. The configured preference is a wish, not an instruction —
Windows can be missing its language packs, Tesseract is usually not installed at
all, PaddleOCR almost never is — so the choice is always made against what answers
:meth:`~ayris.actions.system.ocr_engines.base.OcrEngine.is_available` right now,
and a refused preference is reported rather than silently swallowed.

The order is deliberate. Windows first: offline, instant, already there, Russian
included. Tesseract second: has to be installed by hand, costs a process launch per
recognition, but is the one people already have when they have anything. Paddle
last: it is the best of the three at hard pictures and it is a hundred megabytes of
weights and half a gigabyte of RAM, so it is only used when it is asked for by name
or when nothing else answers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayris.actions.system.ocr_engines.base import (
    OcrBlock,
    OcrEngine,
    OcrEngineError,
    OcrText,
    primary_subtag,
)
from ayris.actions.system.ocr_engines.paddle import PaddleOcr
from ayris.actions.system.ocr_engines.tesseract import TesseractOcr
from ayris.actions.system.ocr_engines.windows_ocr import WindowsOcr
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "ENGINES",
    "Choice",
    "OcrBlock",
    "OcrEngine",
    "OcrEngineError",
    "OcrText",
    "PaddleOcr",
    "TesseractOcr",
    "WindowsOcr",
    "engine_by_name",
    "primary_subtag",
    "reset_engine_cache",
    "select_engine",
]

_log = get_logger(__name__)

#: Every engine, in the order they are tried when the config says ``auto``.
ENGINES: Final[tuple[type[OcrEngine], ...]] = (WindowsOcr, TesseractOcr, PaddleOcr)

#: Built engines, keyed by name and language. Both the Windows recogniser and the
#: Paddle models are expensive to create and cheap to keep, and an action that runs
#: on every «прочитай экран» would otherwise pay for it every time.
_cache: dict[tuple[str, str], OcrEngine] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class Choice:
    """Which engine was chosen, and what was given up to get there.

    ``fallback_from`` and ``reason`` are the interesting fields: an action that
    silently used Tesseract when the settings said Windows is confusing, and the
    user is told once, in the same sentence as the result.
    """

    engine: OcrEngine
    language: str
    fallback_from: str = ""
    reason: str = ""

    @property
    def is_fallback(self) -> bool:
        return bool(self.fallback_from)

    @property
    def note_ru(self) -> str:
        """One sentence about the substitution, empty when there was none."""
        if not self.is_fallback:
            return ""
        wanted = engine_by_name(self.fallback_from)
        title = wanted.title_ru if wanted is not None else self.fallback_from
        return f"Вместо «{title}» использовала {self.engine.title_ru}: {self.reason}"


def engine_by_name(name: str) -> type[OcrEngine] | None:
    """The engine class the config's ``engine`` value names, if it names one."""
    wanted = name.strip().lower()
    for engine in ENGINES:
        if engine.name == wanted:
            return engine
    return None


def reset_engine_cache() -> None:
    """Forget the built engines. For tests, and for a change of settings."""
    with _cache_lock:
        _cache.clear()


def _build(engine: type[OcrEngine], language: str) -> OcrEngine:
    """One engine per (name, language), created once."""
    key = (engine.name, language)
    with _cache_lock:
        existing = _cache.get(key)
        if existing is not None:
            return existing
        built = engine(language)
        _cache[key] = built
        return built


def _try(engine: type[OcrEngine], languages: Sequence[str]) -> tuple[OcrEngine, str] | None:
    """Build the engine if it is available and speaks one of the languages.

    Availability is asked first and the answer is never allowed to raise: an engine
    that throws while being checked would take down the choice of every engine
    after it, and «Tesseract сломан» must not mean «распознавания нет».
    """
    try:
        if not engine.is_available():
            return None
        matched = engine.match_language(languages)
    except Exception as exc:  # a broken engine must not block the rest
        _log.warning("%s failed its availability check: %s", engine.name, exc)
        return None
    if matched is None:
        return None
    return (_build(engine, matched), matched)


def select_engine(
    preference: str = "auto",
    languages: Iterable[str] = ("ru", "en"),
) -> Choice:
    """Pick the engine to use, honouring the preference where reality allows.

    ``preference`` is ``[actions.ocr] engine``: an engine name, or ``auto`` for the
    :data:`ENGINES` order. ``languages`` is the wish list in Ayris' own two-letter
    codes; the chosen engine's own notation comes back in :attr:`Choice.language`.

    Returns:
        The chosen engine, the language tag it was built for, and — when the
        preference could not be honoured — what was asked for and why it was not.

    Raises:
        OcrEngineError: nothing on this machine can read the requested languages.
            The message names what to install, taken from the preferred engine when
            there was one and from Windows otherwise, since that is the one every
            Windows user can fix in Settings.
    """
    wanted = [code for code in languages if code.strip()] or ["ru", "en"]
    requested = preference.strip().lower() or "auto"

    if requested != "auto":
        chosen = engine_by_name(requested)
        if chosen is None:
            _log.warning("unknown OCR engine %r in the settings, falling back to auto", requested)
        else:
            found = _try(chosen, wanted)
            if found is not None:
                return Choice(engine=found[0], language=found[1])
            reason = _refusal(chosen, wanted)
            _log.info("%s is not usable: %s", chosen.name, reason)
            for engine in ENGINES:
                if engine is chosen:
                    continue
                alternative = _try(engine, wanted)
                if alternative is not None:
                    return Choice(
                        engine=alternative[0],
                        language=alternative[1],
                        fallback_from=chosen.name,
                        reason=reason,
                    )
            raise OcrEngineError(
                f"no OCR engine can read {wanted}; {chosen.name} was asked for",
                user_message=chosen.describe_missing(),
            )

    for engine in ENGINES:
        found = _try(engine, wanted)
        if found is not None:
            return Choice(engine=found[0], language=found[1])
    raise OcrEngineError(
        f"no OCR engine can read {wanted}",
        user_message=WindowsOcr.describe_missing(),
    )


def _refusal(engine: type[OcrEngine], languages: Sequence[str]) -> str:
    """Why an engine was not used: absent, or present without the language.

    The distinction is the whole point of the sentence the user hears. "Tesseract
    не установлен" and "Tesseract есть, но не знает русского" need different
    actions from them, and both look identical from inside the code unless asked
    apart here.
    """
    try:
        installed = engine.available_languages()
    except Exception as exc:  # already logged in _try, keep it short
        return f"движок не отвечает ({exc})"
    if not installed:
        return "не установлен"
    names = ", ".join(languages)
    return f"установлен, но не знает языков {names} (есть: {', '.join(installed)})"


def available_engines() -> tuple[str, ...]:
    """Names of the engines usable on this machine, in preference order.

    For the settings window: a dropdown that offers PaddleOCR on a machine without
    it is a dropdown that produces an error message on the first «прочитай экран».
    """
    usable: list[str] = []
    for engine in ENGINES:
        try:
            if engine.is_available():
                usable.append(engine.name)
        except Exception as exc:  # a broken engine is just an unavailable one
            _log.warning("%s failed its availability check: %s", engine.name, exc)
    return tuple(usable)
