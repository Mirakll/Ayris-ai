"""Tesseract: the fallback, and the one the user has to install by hand.

Every other engine here is either already in Windows or a pip install. Tesseract is
a separate program, and its Windows build does not come with Russian: the installer
has a language-selection page most people click past. So this module spends more
code on "is it there, and does it have what we need" than on the recognition
itself, and answers with an instruction rather than a traceback when the answer is
no.

``pytesseract`` is a thin wrapper that runs the binary and parses its output. That
has two consequences worth knowing. Recognition costs a process launch — roughly a
tenth of a second before any pixels are read — which is why this is the fallback
and not the default. And the binary's location is a global on the wrapper module,
not a parameter, so it is set once from the config here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.actions.system.ocr_engines.base import OcrBlock, OcrEngine, OcrEngineError, OcrText
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["TESSERACT_CODES", "TesseractOcr", "find_tesseract"]

_log = get_logger(__name__)

#: Our two-letter codes to Tesseract's ISO 639-2 ones. Only the languages Ayris
#: speaks: the rest of the list is long and none of it is reachable from the
#: config's validation.
TESSERACT_CODES: Final[dict[str, str]] = {
    "ru": "rus",
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "uk": "ukr",
    "be": "bel",
    "kk": "kaz",
    "pl": "pol",
    "zh": "chi_sim",
    "ja": "jpn",
}

#: Where the official Windows installer puts it, and where a Chocolatey or Scoop
#: install ends up. Checked in order when the binary is not on ``PATH`` — which it
#: is not by default, because the installer's "add to PATH" box is off.
_KNOWN_PATHS: Final[tuple[str, ...]] = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\ProgramData\chocolatey\bin\tesseract.exe",
)

#: How long to wait for one recognition. A full 4K screen takes a few seconds on a
#: slow machine; past this the binary is stuck and the action must not be.
_TIMEOUT_S: Final = 60

#: Word level. ``image_to_data`` can also report blocks and paragraphs, and words
#: are the only level where the confidence means anything per box.
_WORD_LEVEL: Final = 5

#: ``image_to_data`` marks a box it could not read at all with this confidence.
_NO_CONFIDENCE: Final = -1.0

#: The URL to send the user to. The upstream project has no Windows binaries; this
#: is the build everyone actually installs, and it is the one whose language page
#: matters for Russian.
_INSTALL_URL: Final = "https://github.com/UB-Mannheim/tesseract/wiki"


def find_tesseract(configured: str = "") -> str:
    """Where the binary is, or ``""``.

    ``configured`` — ``[actions.ocr] tesseract_path`` — wins, then ``PATH``, then
    the three places the Windows installers use. Looking in those is not a hack:
    the installer does not add itself to ``PATH``, so "installed and invisible" is
    the *normal* state on Windows, and an assistant that says «не установлен» about
    a program sitting in Program Files is wrong.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        _log.warning("tesseract_path points at nothing: %s", candidate)
    found = shutil.which("tesseract")
    if found:
        return found
    for path in _KNOWN_PATHS:
        if Path(path).is_file():
            return path
    return ""


class TesseractOcr(OcrEngine):
    """Recognition by shelling out to ``tesseract``."""

    name: ClassVar[str] = "tesseract"
    title_ru: ClassVar[str] = "Tesseract"
    # Trained on 300-DPI scans, and visibly worse on screen-sized text: a 16-pixel
    # font gains several percent of accuracy from being tripled.
    wants_upscale: ClassVar[bool] = True

    #: Set from the config before availability is first asked about. A class
    #: attribute because ``pytesseract`` keeps the binary path in a module global
    #: too, and threading it through every call would only hide that.
    binary: ClassVar[str] = ""

    @classmethod
    def configure(cls, binary: str) -> None:
        """Point the engine at a binary, ``""`` to go back to searching for one."""
        cls.binary = binary
        cls._languages = None

    #: Cached answer of the language query, cleared by :meth:`configure`. The query
    #: launches the binary, and it is asked on every engine selection.
    _languages: ClassVar[tuple[str, ...] | None] = None

    @classmethod
    def executable(cls) -> str:
        return find_tesseract(cls.binary)

    @classmethod
    def is_available(cls) -> bool:
        """Whether the binary exists *and* has at least one language pack."""
        return bool(cls.available_languages())

    @classmethod
    def available_languages(cls) -> tuple[str, ...]:
        """ISO 639-2 codes of the installed ``.traineddata`` files.

        Asked of the binary rather than read off the ``tessdata`` directory: the
        directory can be moved with ``TESSDATA_PREFIX``, and the binary is the only
        thing that knows where it is actually looking.
        """
        if cls._languages is not None:
            return cls._languages
        executable = cls.executable()
        if not executable:
            cls._languages = ()
            return cls._languages
        try:
            completed = subprocess.run(  # the path comes from PATH or the config
                [executable, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _log.warning("tesseract --list-langs failed: %s", exc)
            cls._languages = ()
            return cls._languages
        # The first line is a header ("List of available languages (4):"); the rest
        # are codes, one per line, and stderr is where some builds print them.
        lines = (completed.stdout or completed.stderr or "").splitlines()
        codes = tuple(
            line.strip()
            for line in lines[1:]
            if line.strip() and not line.strip().endswith(":") and " " not in line.strip()
        )
        cls._languages = codes
        _log.debug("tesseract %s has languages: %s", executable, ", ".join(codes) or "none")
        return codes

    @classmethod
    def describe_missing(cls) -> str:
        """Which of the two things is missing, and what to do about it."""
        if not cls.executable():
            return (
                "Tesseract не установлен. Скачай установщик со страницы "
                f"{_INSTALL_URL} и на шаге выбора компонентов отметь "
                "Additional language data → Russian."
            )
        return (
            "Tesseract установлен, но без языковых данных. Запусти установщик "
            "заново и отметь Additional language data → Russian, "
            "или положи rus.traineddata в папку tessdata."
        )

    @classmethod
    def code_for(cls, language: str) -> str:
        """Our ``ru`` as Tesseract's ``rus``; anything else is passed through."""
        return TESSERACT_CODES.get(language.strip().lower(), language.strip().lower())

    def recognise(self, image: Image.Image) -> OcrText:
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrEngineError(
                f"pytesseract is not installed: {exc}",
                user_message=(
                    "Не установлена обёртка pytesseract. "
                    "Выбери в настройках другой движок распознавания."
                ),
            ) from exc

        executable = self.executable()
        if not executable:
            raise OcrEngineError(
                "no tesseract binary",
                user_message=self.describe_missing(),
            )
        pytesseract.pytesseract.tesseract_cmd = executable
        code = self.code_for(self.language)
        try:
            data = pytesseract.image_to_data(
                image,
                lang=code,
                output_type=pytesseract.Output.DICT,
                timeout=_TIMEOUT_S,
            )
        except RuntimeError as exc:  # pytesseract's own timeout
            raise OcrEngineError(
                f"tesseract timed out after {_TIMEOUT_S}s: {exc}",
                user_message="Распознавание заняло слишком долго и было прервано.",
            ) from exc
        except pytesseract.TesseractError as exc:
            raise OcrEngineError(
                f"tesseract failed for lang={code!r}: {exc}",
                user_message=self.describe_missing(),
            ) from exc
        except (OSError, pytesseract.TesseractNotFoundError) as exc:
            raise OcrEngineError(
                f"cannot run {executable}: {exc}",
                user_message=self.describe_missing(),
            ) from exc

        blocks = tuple(parse_image_data(data))
        return OcrText(
            text=join_lines(data),
            blocks=blocks,
            engine=self.name,
            language=code,
        )


def parse_image_data(data: dict[str, list[object]]) -> list[OcrBlock]:
    """Turn ``image_to_data``'s parallel lists into blocks.

    The output is a dict of equally long lists — one entry per detected box at
    every level, from page down to word — so this filters to words with text and
    zips the coordinates back together. Boxes with a confidence of ``-1`` are
    Tesseract saying it found something and could not read it; they are dropped
    rather than reported as empty text.
    """
    texts = [str(value) for value in data.get("text", [])]
    levels = data.get("level", [])
    blocks: list[OcrBlock] = []
    for index, text in enumerate(texts):
        if not text.strip():
            continue
        if levels and _as_int(levels[index]) != _WORD_LEVEL:
            continue
        confidence = _as_float(data.get("conf", [])[index] if data.get("conf") else None)
        if confidence == _NO_CONFIDENCE:
            continue
        left = _as_int(data["left"][index])
        top = _as_int(data["top"][index])
        blocks.append(
            OcrBlock(
                text=text,
                rect=winapi.Rect(
                    left,
                    top,
                    left + _as_int(data["width"][index]),
                    top + _as_int(data["height"][index]),
                ),
                confidence=None if confidence is None else confidence / 100.0,
            )
        )
    return blocks


def join_lines(data: dict[str, list[object]]) -> str:
    """Rebuild the text with line breaks where Tesseract saw them.

    ``image_to_string`` would give this directly, at the price of running the
    binary a second time. The line identity is already in the data: words carry the
    block, paragraph and line numbers they belong to.
    """
    texts = [str(value) for value in data.get("text", [])]
    keys = list(
        zip(
            data.get("block_num", []),
            data.get("par_num", []),
            data.get("line_num", []),
            strict=False,
        )
    )
    lines: list[list[str]] = []
    previous: tuple[object, ...] | None = None
    for index, text in enumerate(texts):
        if not text.strip():
            continue
        key = keys[index] if index < len(keys) else None
        if key != previous or not lines:
            lines.append([])
            previous = key
        lines[-1].append(text)
    return "\n".join(" ".join(words) for words in lines if words)


def _as_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
