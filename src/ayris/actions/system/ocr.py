"""Reading text off the screen: capture, prepare, recognise, hand back.

Two actions — :class:`OcrScreen` over a monitor or a window, :class:`OcrRegion` over
a rectangle the user draws — and both are the same four steps. The frame comes from
:mod:`ayris.actions.system.screenshot`, the recognition from an engine chosen by
:func:`~ayris.actions.system.ocr_engines.select_engine`, and this module is what sits
between them: the preparation of the picture, and what happens to the text after.

Three decisions in here are worth the words.

**Preparation is per engine, not per image.** Tesseract is trained on 300-DPI scans
and reads 14-pixel interface text badly; Windows OCR is trained on screens and gains
nothing from being handed a tripled one. So the scale factor comes from the engine's
own ``wants_upscale``, and it is capped by area rather than by side: tripling a small
region is cheap, tripling a 4K frame is a hundred megapixels and several seconds of
the user's time for no gain at all.

**Blocks come back in screen coordinates.** An engine reports boxes in the pixels of
the image it was given — a scaled crop of somewhere on the desktop — and those
numbers mean nothing to anything downstream. Undoing the scale and adding the capture
origin turns them into coordinates a click can use, which is the whole reason for
keeping them.

**Recognition and capture are separate functions.** :func:`read_frame` takes a frame
and never touches a screen, so the engine choice, the fallback, the scaling and the
coordinate mapping are all testable with a synthetic picture and a fake engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final

from PIL import Image, ImageOps
from pydantic import Field, model_validator

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.clipboard import get_clipboard
from ayris.actions.system.ocr_engines import Choice, OcrEngine, OcrText, select_engine
from ayris.actions.system.screenshot import (
    MAX_SIDE,
    capture_all,
    capture_monitor,
    capture_rect,
    capture_region,
    capture_window,
)
from ayris.actions.system.windows import WindowQuery, list_windows, select_window
from ayris.core.config import get_settings
from ayris.core.errors import ActionError
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.system.screenshot import Frame

__all__ = [
    "MAX_PIXELS",
    "MAX_SCALE",
    "OcrFailed",
    "OcrRegion",
    "OcrScreen",
    "Reading",
    "TextOutput",
    "binarize",
    "for_speech",
    "prepare",
    "read_frame",
    "scale_for",
]

_log = get_logger(__name__)

#: A screen is 96 DPI by definition, so the factor asked of a capture to reach the
#: configured ``upscale_to_dpi`` is that over this.
SCREEN_DPI: Final = 96

#: The most pixels we will hand an engine. A 4K frame is 8.3 MP and passes untouched;
#: a region can still be tripled. Past the cap the factor comes down rather than the
#: picture being cropped — a smaller chance at the text beats none.
MAX_PIXELS: Final = 40_000_000

#: Never scale by more than this, whatever the DPI setting asks. Past about three
#: times, interpolation invents detail rather than revealing it.
MAX_SCALE: Final = 4.0

#: Below this a resize is not worth its own copy of the image.
MIN_SCALE: Final = 1.05

#: Appended after a cut when the text is too long to read out.
_TRUNCATED_RU: Final = "… дальше не читаю, текст в буфере обмена."

#: Longest fragment of recognised text to put in a result's message. The message goes
#: into the log and the history list, and a full page of OCR there is unreadable.
_PREVIEW_LIMIT: Final = 120

#: Where :func:`binarize` splits black from white, after autocontrast has spread the
#: picture across the full range.
_MIDPOINT: Final = 128


class OcrFailed(ActionError):
    """The capture worked and the recognition did not."""

    default_user_message = "Не получилось распознать текст."


class TextOutput(StrEnum):
    """Where recognised text goes. Mirrors ``[actions.ocr] output``."""

    CLIPBOARD = "clipboard"
    SPEAK = "speak"
    BOTH = "both"

    @property
    def to_clipboard(self) -> bool:
        return self is not TextOutput.SPEAK

    @property
    def to_speech(self) -> bool:
        return self is not TextOutput.CLIPBOARD

    @property
    def title_ru(self) -> str:
        return {
            TextOutput.CLIPBOARD: "в буфер обмена",
            TextOutput.SPEAK: "прочитать вслух",
            TextOutput.BOTH: "и в буфер, и вслух",
        }[self]


@dataclass(frozen=True, slots=True)
class Reading:
    """Everything one recognition produced.

    Attributes:
        text: The recognised text, with block rectangles in screen coordinates.
        frame: The capture it was read from, kept so a caller can save or re-read it.
        choice: Which engine ran, and whether it was the one that was asked for.
        scale: How much the picture was enlarged before recognition.
        clipboard: Whether the text reached the clipboard.
    """

    text: OcrText
    frame: Frame
    choice: Choice
    scale: float = 1.0
    clipboard: bool = False

    @property
    def is_empty(self) -> bool:
        return self.text.is_empty

    def with_clipboard(self, *, copied: bool) -> Reading:
        """The same reading, recording that the text was copied."""
        return Reading(
            text=self.text,
            frame=self.frame,
            choice=self.choice,
            scale=self.scale,
            clipboard=copied,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text.text,
            "blocks": [block.as_dict() for block in self.text.blocks],
            "engine": self.choice.engine.name,
            "language": self.choice.language,
            "fallback_from": self.choice.fallback_from,
            "scale": round(self.scale, 3),
            "clipboard": self.clipboard,
            "frame": self.frame.as_dict(),
        }


# --------------------------------------------------------------------------- #
# Preparing the picture
# --------------------------------------------------------------------------- #


def scale_for(engine: OcrEngine, frame_pixels: int, *, target_dpi: int) -> float:
    """How much to enlarge a capture before handing it to ``engine``.

    ``1.0`` when the engine does not want it, when the picture is already large
    enough to reach :data:`MAX_PIXELS`, or when the gain would be under
    :data:`MIN_SCALE`.

    Args:
        engine: The engine that will read the picture; its ``wants_upscale`` decides
            whether any of this applies at all.
        frame_pixels: Area of the capture, in pixels.
        target_dpi: ``[actions.ocr] upscale_to_dpi``.
    """
    if not engine.wants_upscale or frame_pixels <= 0:
        return 1.0
    wanted = min(max(target_dpi / SCREEN_DPI, 1.0), MAX_SCALE)
    if wanted < MIN_SCALE:
        return 1.0
    # Area grows with the square of the factor, so the cap is a square root.
    allowed = (MAX_PIXELS / frame_pixels) ** 0.5
    factor = min(wanted, allowed)
    return factor if factor >= MIN_SCALE else 1.0


def binarize(image: Image.Image) -> Image.Image:
    """Grayscale, autocontrast, then split black from white at the midpoint.

    Off by default, and rightly so: subpixel-antialiased screen text loses its edges
    to a hard threshold and comes out *worse*. What it is for is the opposite case —
    a dark theme, a washed-out scan, a photograph of a monitor — where the engine
    cannot find letters at all and a flattened picture is the only thing that works.
    The autocontrast before the threshold is what makes a fixed midpoint usable
    whatever the original's brightness was.
    """
    gray = ImageOps.autocontrast(image if image.mode == "L" else image.convert("L"))
    return gray.point(lambda value: 255 if value >= _MIDPOINT else 0, mode="L")


def prepare(
    frame: Frame,
    engine: OcrEngine,
    *,
    target_dpi: int = 300,
    binarise: bool = False,
) -> tuple[Image.Image, float]:
    """Turn a capture into the picture this engine reads best.

    Returns:
        The prepared image and the factor it ended up scaled by — which the caller
        needs in order to map the recognised boxes back onto the screen.
    """
    image = frame.to_image()
    factor = scale_for(engine, frame.pixels, target_dpi=target_dpi)
    if factor > 1.0:
        # LANCZOS over BICUBIC: on text the difference is visible, and it costs a
        # few milliseconds on a picture this size.
        size = (round(image.width * factor), round(image.height * factor))
        image = image.resize(size, Image.Resampling.LANCZOS)
    if binarise:
        image = binarize(image)
    side = engine.max_side()
    if side and max(image.size) > side:
        # Windows refuses anything over 10000 px a side outright, so shrinking is
        # the difference between a result and an exception.
        image.thumbnail((side, side), Image.Resampling.LANCZOS)
        factor = image.width / frame.width if frame.width else factor
        _log.debug("shrank the capture to %dx%d for %s", *image.size, engine.name)
    return (image, factor)


# --------------------------------------------------------------------------- #
# Recognition, without a screen in sight
# --------------------------------------------------------------------------- #


def read_frame(
    frame: Frame,
    *,
    preference: str = "",
    languages: tuple[str, ...] = (),
    target_dpi: int = 0,
    binarise: bool | None = None,
    min_confidence: float | None = None,
) -> Reading:
    """Recognise the text in a capture. The seam every OCR action goes through.

    Anything left unspecified comes from ``[actions.ocr]``.

    Raises:
        OcrFailed: the frame came back black — capture was blocked, and no engine
            will find text in it.
        ~ayris.actions.system.ocr_engines.base.OcrEngineError: no usable engine on
            this machine, or the engine failed on the picture.
    """
    settings = get_settings().actions.ocr
    if frame.is_blank:
        raise OcrFailed(
            f"the {frame.width}x{frame.height} capture is black",
            user_message=(
                "Кадр получился чёрный — окно защищено от захвата или рисуется "
                "через эксклюзивный полный экран. Распознавать нечего."
            ),
        )
    choice = select_engine(preference or settings.engine, languages or tuple(settings.languages))
    image, factor = prepare(
        frame,
        choice.engine,
        target_dpi=target_dpi or settings.upscale_to_dpi,
        binarise=settings.binarize if binarise is None else binarise,
    )
    text = choice.engine.recognise(image)
    threshold = settings.min_confidence if min_confidence is None else min_confidence
    if threshold > 0:
        text = text.filtered(threshold)
    text = text.on_screen(scale=factor, origin=frame.rect)
    _log.info(
        "%s read %d block(s) from %dx%d at x%.2f",
        choice.engine.name,
        len(text.blocks),
        frame.width,
        frame.height,
        factor,
    )
    return Reading(text=text, frame=frame, choice=choice, scale=factor)


# --------------------------------------------------------------------------- #
# What happens to the text
# --------------------------------------------------------------------------- #


def copy_text(text: str) -> bool:
    """Put the text on the clipboard, ``False`` if it would not go.

    Text, unlike the pictures in :mod:`ayris.actions.system.screenshot`, needs no
    win32 code of its own here: the project's one clipboard wrapper already writes
    ``CF_UNICODETEXT`` and already retries a clipboard another program is holding
    open. Recognised text is deliberately *not* hidden from the history monitor —
    the user asked for it to be on the clipboard, so it belongs in «вставь второй»
    like any other copy.

    A refusal never fails the recognition: the text is still returned and still
    read out, so a locked clipboard costs a warning and nothing else.
    """
    if not text.strip():
        return False
    try:
        get_clipboard().write_text(text)
    except ActionError:
        _log.warning("не удалось скопировать распознанный текст", exc_info=True)
        return False
    return True


def for_speech(text: str, limit: int) -> str:
    """The text as it should be read out: whole, or cut at a word with a note.

    A screen holds thousands of characters and a spoken sentence must not. The cut
    lands on the last space before the limit so a word is not sliced in half, and the
    user is told where the rest went instead of being left wondering.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    head = flat[:limit]
    cut = head.rfind(" ")
    if cut > limit // 2:
        head = head[:cut]
    return head.rstrip(" ,.;:—-") + _TRUNCATED_RU


def deliver(reading: Reading, mode: TextOutput | None = None) -> Reading:
    """Copy the text if the mode says so.

    Speaking is not done here: an action returns its sentence and the pipeline says
    it, so the spoken form goes into the result's ``message_ru`` instead.
    """
    wanted = mode or TextOutput(get_settings().actions.ocr.output)
    if reading.is_empty or not wanted.to_clipboard:
        return reading
    return reading.with_clipboard(copied=copy_text(reading.text.text))


def _spoken(reading: Reading, *, what: str, mode: TextOutput | None = None) -> str:
    """The sentence the assistant says about a reading."""
    settings = get_settings().actions.ocr
    wanted = mode or TextOutput(settings.output)
    note = reading.choice.note_ru
    if reading.is_empty:
        return _joined(f"{what} текста не нашла.", note)
    if wanted.to_speech:
        return _joined(for_speech(reading.text.text, settings.speak_limit), note)
    lines = len(reading.text.lines)
    where = "в буфере обмена" if reading.clipboard else "не поместился в буфер обмена"
    preview = for_speech(reading.text.text, _PREVIEW_LIMIT)
    return _joined(f"{what} {lines} строк(и), {where}. {preview}", note)


def _joined(body: str, note: str) -> str:
    return f"{body} {note}" if note else body


def _result(
    reading: Reading,
    *,
    what: str,
    mode: TextOutput | None = None,
) -> ActionResult[Reading]:
    """One reading as an action result."""
    detail = (
        f"{reading.choice.engine.name}/{reading.choice.language}, "
        f"{len(reading.text.blocks)} блок(ов), x{reading.scale:.2f}"
    )
    return ActionResult.done(
        _spoken(reading, what=what, mode=mode),
        value=reading,
        detail=detail,
        data=reading.as_dict(),
    )


class _OcrParams(ActionParams):
    """The overrides both OCR actions share."""

    output: TextOutput | None = Field(
        default=None,
        description="Куда девать текст; пусто — как в настройках",
        json_schema_extra={
            "choices_ru": {str(mode): mode.title_ru for mode in TextOutput},
        },
    )
    language: str = Field(
        default="",
        max_length=16,
        title="Язык",
        description="ru, en…; пусто — порядок языков из настроек",
    )

    @property
    def languages(self) -> tuple[str, ...]:
        """The language override as a one-item wish list, or nothing."""
        wanted = self.language.strip()
        return (wanted,) if wanted else ()


@register
class OcrScreen(Action):
    """Text on the whole desktop, on one monitor, or in one window."""

    meta: ClassVar = ActionMeta(
        name="OcrScreen",
        category=ActionCategory.CAPTURE,
        title_ru="Распознать текст на экране",
        description_ru="Снять экран, монитор или окно и прочитать текст на снимке",
        # Capture is instant, recognition is not: Tesseract on an upscaled 4K frame
        # takes the better part of a minute on a slow machine, and PaddleOCR builds
        # its models on the first call.
        timeout_ms=120_000,
    )

    class Params(_OcrParams):
        monitor: str = Field(
            default="",
            max_length=120,
            title="Монитор",
            description="primary, external_1, номер или часть имени; пусто — все мониторы",
        )
        window: str = Field(
            default="",
            max_length=200,
            title="Окно",
            description="Часть заголовка окна; задано — читается окно, а не монитор",
        )

    def run(self, params: Params) -> ActionResult[Reading]:
        frame, what = self._capture(params)
        reading = deliver(read_frame(frame, languages=params.languages), params.output)
        return _result(reading, what=what, mode=params.output)

    @staticmethod
    def _capture(params: Params) -> tuple[Frame, str]:
        """The frame to read, and how to refer to it out loud."""
        title = params.window.strip()
        if title:
            query = WindowQuery(title=title)
            found = select_window(list_windows(query), query)
            return (capture_window(found.hwnd, title=found.title), f"В окне «{found.title}»:")
        address = params.monitor.strip()
        if address:
            frame = capture_monitor(address)
            return (frame, f"На мониторе «{frame.monitor or address}»:")
        return (capture_all(), "На экране:")


@register
class OcrRegion(Action):
    """Text in a rectangle, either given in numbers or dragged with the mouse."""

    meta: ClassVar = ActionMeta(
        name="OcrRegion",
        category=ActionCategory.CAPTURE,
        title_ru="Распознать текст в области",
        description_ru="Выделить область экрана мышью или задать координаты и прочитать текст",
        # Long, because the interactive branch waits for a person: the selection has
        # its own [actions.screenshot] selection_timeout_s, and this only has to be
        # comfortably longer than that plus one recognition.
        timeout_ms=300_000,
    )

    class Params(_OcrParams):
        left: int = Field(default=0, ge=-MAX_SIDE, le=MAX_SIDE, description="Левый край области")
        top: int = Field(default=0, ge=-MAX_SIDE, le=MAX_SIDE, description="Верхний край области")
        width: int = Field(default=0, ge=0, le=MAX_SIDE, description="Ширина области")
        height: int = Field(default=0, ge=0, le=MAX_SIDE, description="Высота области")

        @property
        def has_rect(self) -> bool:
            """Whether numbers were given, so nothing has to be asked of the user."""
            return self.width > 0 and self.height > 0

        @model_validator(mode="after")
        def _both_sides(self) -> OcrRegion.Params:
            """A width without a height is half a rectangle, not a default."""
            if (self.width > 0) != (self.height > 0):
                raise ValueError("укажите и ширину, и высоту области")
            return self

    def run(self, params: Params) -> ActionResult[Reading]:
        if params.has_rect:
            rect = winapi.Rect(
                params.left,
                params.top,
                params.left + params.width,
                params.top + params.height,
            )
            frame = capture_rect(rect)
        else:
            settings = get_settings().actions.screenshot
            captured = capture_region(
                timeout_s=settings.selection_timeout_s,
                dim=settings.dim_opacity,
            )
            if captured is None:
                return ActionResult.done("Отменила выделение области.", data={"cancelled": True})
            frame = captured
        reading = deliver(read_frame(frame, languages=params.languages), params.output)
        return _result(reading, what="В области:", mode=params.output)
