"""What every OCR engine looks like from the outside.

Three engines end up behind this: Windows' own recogniser, Tesseract and
PaddleOCR. They agree on very little — one is a COM API that speaks one language
per instance, one is a subprocess, one is a neural network that wants a numpy
array — so the interface here is deliberately narrow: hand over an image, get back
text and where on the image it was.

The result type is shaped by what the engines can actually promise. Every one of
them gives text and a bounding box; only Tesseract and Paddle give a confidence,
so :attr:`OcrBlock.confidence` is optional rather than a zero — a filter on
confidence must be able to tell "the engine was unsure" from "the engine does not
report certainty". Coordinates are in the pixels of the image that was recognised,
which is not the screen: it has been cropped and scaled on the way in, and
:meth:`OcrText.on_screen` is what puts it back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ayris.core.errors import ActionError
from ayris.utils import winapi

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PIL import Image

__all__ = [
    "OcrBlock",
    "OcrEngine",
    "OcrEngineError",
    "OcrText",
    "primary_subtag",
]


class OcrEngineError(ActionError):
    """The engine was there and still could not read the picture."""

    default_user_message = "Не получилось разобрать текст на картинке."


def primary_subtag(tag: str) -> str:
    """``"ru-RU"`` → ``"ru"``, ``"eng"`` → ``"eng"``, whitespace and case ignored.

    Enough language matching for the job. Engines name languages three different
    ways — BCP-47 for Windows, ISO 639-2 for Tesseract, its own short list for
    Paddle — and all three agree on the first subtag being the language, which is
    the only part the user ever says out loud.
    """
    return tag.strip().lower().replace("_", "-").split("-", 1)[0]


@dataclass(frozen=True, slots=True)
class OcrBlock:
    """One recognised piece of text and where it was.

    A "block" is whatever unit the engine returns — a line for Windows and Paddle,
    a word for Tesseract at word level. It is not normalised on purpose: joining
    words into lines guesses at the layout, and the guess would be wrong exactly
    where it matters, on a two-column page.
    """

    text: str
    rect: winapi.Rect = field(default_factory=winapi.Rect)
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "rect": self.rect.as_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class OcrText:
    """Everything one recognition produced.

    ``text`` is the engine's own rendering of the whole picture, kept as it came
    rather than rebuilt from the blocks: the engines put the lines in reading order
    and know more about the layout than a sort by coordinate would.
    """

    text: str = ""
    blocks: tuple[OcrBlock, ...] = ()
    engine: str = ""
    language: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(line for line in self.text.splitlines() if line.strip())

    def filtered(self, min_confidence: float) -> OcrText:
        """Drop the blocks the engine was unsure about.

        Blocks with no confidence at all survive: Windows does not report one, and
        dropping everything it returns would make the setting mean "turn the
        default engine off".
        """
        if min_confidence <= 0:
            return self
        kept = tuple(
            block
            for block in self.blocks
            if block.confidence is None or block.confidence >= min_confidence
        )
        if len(kept) == len(self.blocks):
            return self
        return OcrText(
            text="\n".join(block.text for block in kept),
            blocks=kept,
            engine=self.engine,
            language=self.language,
        )

    def on_screen(self, *, scale: float, origin: winapi.Rect) -> OcrText:
        """The same text with block coordinates moved onto the virtual desktop.

        ``scale`` is what the preprocessing multiplied the image by and ``origin``
        is the rectangle that was captured, so this undoes both and the caller gets
        coordinates it can click on.
        """
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        moved = tuple(
            OcrBlock(
                text=block.text,
                rect=winapi.Rect(
                    left=origin.left + round(block.rect.left / scale),
                    top=origin.top + round(block.rect.top / scale),
                    right=origin.left + round(block.rect.right / scale),
                    bottom=origin.top + round(block.rect.bottom / scale),
                ),
                confidence=block.confidence,
            )
            for block in self.blocks
        )
        return OcrText(text=self.text, blocks=moved, engine=self.engine, language=self.language)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "engine": self.engine,
            "language": self.language,
            "blocks": [block.as_dict() for block in self.blocks],
        }


class OcrEngine(ABC):
    """One way of turning a picture into text.

    An instance is bound to one language, because the default engine is: Windows
    creates a separate recogniser per language and reads Latin letters as Cyrillic
    if asked in the wrong one — "Hello" out of the Russian engine comes back as
    "Нено". So the language is chosen once, when the engine is built, and the
    expensive native object behind it is created on first use and kept.
    """

    #: Stable identifier, the same string the config's ``engine`` setting takes.
    name: ClassVar[str] = ""

    #: What the user is told when Ayris says which engine read the screen.
    title_ru: ClassVar[str] = ""

    #: Whether the engine wants the image upscaled before it sees it. Windows is
    #: happy with screen-sized text; Tesseract is trained on 300-DPI scans and
    #: loses accuracy badly below roughly a 30-pixel cap height.
    wants_upscale: ClassVar[bool] = True

    #: Longest side the engine accepts, 0 for no limit of its own.
    max_dimension: ClassVar[int] = 0

    def __init__(self, language: str) -> None:
        self.language = language

    def __repr__(self) -> str:
        return f"{type(self).__name__}(language={self.language!r})"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this engine can be used at all on this machine right now.

        Must not raise and must not be slow: it is called to choose an engine, on
        every recognition, and one that shells out to a missing binary has to
        answer ``False`` rather than throw.
        """

    @classmethod
    @abstractmethod
    def available_languages(cls) -> tuple[str, ...]:
        """Language tags the engine can read here, in the engine's own notation.

        Empty when the engine is not installed. Also empty — and this is the case
        that matters — when it is installed without its language data, which is
        Tesseract's normal state on a fresh Windows.
        """

    @abstractmethod
    def recognise(self, image: Image.Image) -> OcrText:
        """Read the image. Coordinates come back in the image's own pixels.

        Raises:
            OcrEngineError: the engine was available and the recognition failed.
        """

    @classmethod
    def code_for(cls, language: str) -> str:
        """This engine's own name for one of Ayris' two-letter language codes.

        The identity by default, and overridden where the engine speaks another
        notation — ``ru`` is ``rus`` to Tesseract. It exists on the base class
        because :meth:`match_language` needs it: comparing a wish for ``ru`` against
        an installed ``rus`` on the first subtag finds nothing at all.
        """
        return language.strip().lower()

    @classmethod
    def match_language(cls, wanted: Iterable[str]) -> str | None:
        """The first wanted language this engine has data for, or ``None``.

        Matching is on the primary subtag, so a config asking for ``ru`` is
        answered with the engine's own ``ru-RU`` or ``rus`` — the user should not
        have to know which of the three notations the chosen engine speaks.
        """
        installed = cls.available_languages()
        by_subtag: dict[str, str] = {}
        for tag in installed:
            by_subtag.setdefault(primary_subtag(tag), tag)
        for tag in wanted:
            # The engine's own code first, then the tag as the user wrote it: a
            # config saying "rus" outright should work as well as "ru".
            for candidate in (cls.code_for(tag), tag):
                found = by_subtag.get(primary_subtag(candidate))
                if found is not None:
                    return found
        return None

    @classmethod
    def max_side(cls) -> int:
        """Longest side this engine will accept, ``0`` for no limit.

        A method rather than the bare :attr:`max_dimension` because Windows reports
        the number at runtime and the answer needs a WinRT call. Preprocessing has
        to respect it: Windows OCR rejects an oversized bitmap outright instead of
        scaling it down itself.
        """
        return cls.max_dimension

    @classmethod
    def describe_missing(cls) -> str:
        """What to tell the user when this engine was asked for and is not there.

        An instruction, not a diagnosis: every engine here needs a different thing
        installed, and «поставь пакет» is not an answer a person can act on.
        """
        return f"Движок {cls.title_ru or cls.name} недоступен."
