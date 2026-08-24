"""PaddleOCR: optional, heavy, and better than the other two on hard pictures.

A detection network plus a recognition network, around a hundred megabytes of
weights and half a gigabyte of resident memory once loaded. In exchange it reads
what the others do not: rotated text, low contrast, photographs of screens,
handwriting-adjacent fonts. That trade — a slow first call and a lot of RAM for
accuracy — is why this is an ``ocr`` extra and never the default.

Everything about it is deferred. The import happens inside the methods, because
importing ``paddleocr`` pulls in ``paddle`` itself and takes several seconds; the
model is built on first recognition and kept, because building it downloads weights
the first time. Which means the availability check must not touch either: it looks
for the module in the import system without importing it.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.actions.system.ocr_engines.base import OcrBlock, OcrEngine, OcrEngineError, OcrText
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["PADDLE_CODES", "PaddleOcr"]

_log = get_logger(__name__)

#: Our codes to Paddle's. Cyrillic languages share one recognition model — the
#: ``cyrillic`` one — while ``ru`` also exists as an alias for it; ``en`` has its
#: own. Only the languages Ayris speaks are listed.
PADDLE_CODES: Final[dict[str, str]] = {
    "ru": "ru",
    "en": "en",
    "uk": "uk",
    "be": "be",
    "de": "german",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "pl": "pl",
    "zh": "ch",
    "ja": "japan",
}

#: What ``paddleocr`` calls itself in the import system. Checked without importing.
_MODULE: Final = "paddleocr"


class PaddleOcr(OcrEngine):
    """Recognition through PaddleOCR, loaded on demand."""

    name: ClassVar[str] = "paddle"
    title_ru: ClassVar[str] = "PaddleOCR"
    # The detector works on the image at its own scale and upscaling only costs
    # time, but it is trained on photographs rather than on 12-pixel UI text, so a
    # small crop still benefits.
    wants_upscale: ClassVar[bool] = True

    def __init__(self, language: str) -> None:
        super().__init__(language)
        self._reader: Any | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Whether the package is installed, without paying to import it.

        ``find_spec`` walks the path finders and stops at the module's location, so
        this costs a directory listing rather than the several seconds and hundreds
        of megabytes that importing ``paddle`` does.
        """
        try:
            return importlib.util.find_spec(_MODULE) is not None
        except (ImportError, ValueError):  # a broken or partially removed install
            return False

    @classmethod
    def available_languages(cls) -> tuple[str, ...]:
        """What we are prepared to ask it for, when it is installed at all.

        Not a query: Paddle has no "installed languages" — it downloads the model
        for whichever language it is asked about, on first use. So the honest answer
        is the list this module knows how to name.
        """
        if not cls.is_available():
            return ()
        return tuple(PADDLE_CODES)

    @classmethod
    def describe_missing(cls) -> str:
        return (
            "PaddleOCR не установлен. Он необязательный и тяжёлый: "
            "pip install ayris[ocr], плюс paddlepaddle под свою платформу."
        )

    @classmethod
    def code_for(cls, language: str) -> str:
        return PADDLE_CODES.get(language.strip().lower(), language.strip().lower())

    def _reader_for(self, code: str) -> Any:
        """The ``PaddleOCR`` object, built once and kept.

        The first build downloads two models into the user's cache, which is why
        the failure here is reported as "не смогла загрузить" and not as a
        recognition error: at this point nothing has been read yet.
        """
        if self._reader is not None:
            return self._reader
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrEngineError(
                f"paddleocr is not importable: {exc}",
                user_message=self.describe_missing(),
            ) from exc
        try:
            # show_log off: the wrapper logs a page of banner text per call
            # straight to stdout, and print() is banned in this project for a
            # reason — it would go into the packaged app's void.
            self._reader = PaddleOCR(lang=code, use_angle_cls=True, show_log=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise OcrEngineError(
                f"PaddleOCR({code!r}) failed to initialise: {exc}",
                user_message=(
                    "Не получилось загрузить модели PaddleOCR. "
                    "Проверь, что paddlepaddle установлен и есть место на диске."
                ),
            ) from exc
        return self._reader

    def recognise(self, image: Image.Image) -> OcrText:
        code = self.code_for(self.language)
        reader = self._reader_for(code)
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        try:
            import numpy as np

            raw = reader.ocr(np.asarray(rgb), cls=True)
        except ImportError as exc:  # pragma: no cover - numpy is a hard dependency
            raise OcrEngineError(
                f"numpy is required to feed PaddleOCR: {exc}",
                user_message=self.describe_missing(),
            ) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise OcrEngineError(
                f"PaddleOCR failed on a {rgb.width}×{rgb.height} image: {exc}",
                user_message="PaddleOCR не смог прочитать картинку.",
            ) from exc
        blocks = tuple(parse_paddle_result(raw))
        return OcrText(
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            engine=self.name,
            language=code,
        )


def parse_paddle_result(raw: object) -> list[OcrBlock]:
    """Flatten Paddle's nested output into blocks.

    The shape is ``[[[box, (text, confidence)], ...]]`` — a list per input image,
    and this module always passes one, but a ``None`` in place of the inner list is
    how "found nothing" is reported and an older version returns the inner list
    directly. Anything that does not fit is skipped rather than raised on: the
    wrapper's output shape has changed twice between releases, and a partially
    understood result is still worth reading out.
    """
    if not isinstance(raw, list) or not raw:
        return []
    pages = [raw] if _is_entry(raw[0]) else raw
    blocks: list[OcrBlock] = []
    for page in pages:
        if not isinstance(page, list):
            continue
        for entry in page:
            block = _parse_entry(entry)
            if block is not None:
                blocks.append(block)
    return blocks


def _is_entry(value: object) -> bool:
    """Whether this is one ``[box, (text, confidence)]`` rather than a page of them.

    Told apart by the second element: in an entry it is the payload, whose first item
    is the recognised string; in a page it is another entry, whose first item is a box
    of four points. Guessing by nesting depth instead — the obvious approach — reads
    a one-entry flat result as a page and returns the box's numbers as text.
    """
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    payload = value[1]
    return isinstance(payload, list | tuple) and bool(payload) and isinstance(payload[0], str)


def _parse_entry(entry: object) -> OcrBlock | None:
    """One ``[box, (text, confidence)]`` pair, or ``None`` if it is not one."""
    if not isinstance(entry, list | tuple) or len(entry) < 2:
        return None
    box, payload = entry[0], entry[1]
    if not isinstance(payload, list | tuple) or not payload:
        return None
    text = str(payload[0])
    if not text.strip():
        return None
    confidence = None
    if len(payload) > 1:
        try:
            confidence = float(payload[1])
        except (TypeError, ValueError):
            confidence = None
    return OcrBlock(text=text, rect=_box_rect(box), confidence=confidence)


def _box_rect(box: object) -> winapi.Rect:
    """The bounding box of Paddle's four corner points.

    The corners are a quadrilateral, not a rectangle — that is the point of the
    detector, it finds rotated text — so what comes back here is the upright box
    around it, which is what a caller can click on.
    """
    if not isinstance(box, list | tuple) or not box:
        return winapi.Rect()
    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return winapi.Rect()
    return winapi.Rect(round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))
