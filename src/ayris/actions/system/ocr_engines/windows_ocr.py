"""Windows' own OCR: offline, already installed, Russian out of the box.

``Windows.Media.Ocr`` is the right default for an offline assistant on Windows. It
ships with the system, needs no model download, reads Russian and English on a
stock installation, and runs in about thirty milliseconds on a screenshot-sized
image. What it costs is a WinRT round trip, and the awkward parts of that are all
here:

* **One language per engine object.** There is no multilingual mode. An engine
  created for Russian reads Latin words as Cyrillic — "Hello world" comes back as
  "Нено world" — so the language is picked when the engine is built and the
  configured order matters.
* **The picture has to become a ``SoftwareBitmap``.** Going through PNG bytes and
  a ``BitmapDecoder`` is the recipe found everywhere and it is two encodes too
  many; ``CreateCopyFromBuffer`` takes the raw pixels, and any Python buffer is
  accepted, so a grayscale ``Image`` is handed over as-is.
* **The API is asynchronous.** ``IAsyncOperation.get()`` blocks, which is what an
  action running on a worker thread wants. No event loop, no asyncio — but see
  :func:`_await` for the apartment the wait is allowed to happen on.
* **Coordinates come back as floats** in the bitmap's pixels, and per *word* —
  lines have no rectangle of their own, so a line's box is the union of its words'.

Language packs are the one thing that can be missing: a Windows installed in, say,
German has neither ``ru`` nor ``en`` OCR data, and
``available_recognizer_languages`` is then short. That is a normal condition, not
an error — :meth:`WindowsOcr.is_available` reports it and the caller falls back.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.actions.system.ocr_engines.base import OcrBlock, OcrEngine, OcrEngineError, OcrText
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["WindowsOcr"]

_log = get_logger(__name__)

#: Gray8 rather than Bgra8: the recogniser converts to grayscale itself, the buffer
#: is a quarter of the size, and the alpha byte of a GDI capture is undefined —
#: handing it over as Bgra8 risks a bitmap the compositor considers fully
#: transparent.
_GRAY8: Final = "Gray8"


class _Winrt:
    """The WinRT names, imported once and only when they are needed.

    A module-level import would make every action module fail to load on a machine
    without the ``winrt-*`` packages — including on the Linux CI runner, where they
    are deliberately not installed and the OCR tests run against fakes.

    The four names are ``Any`` because the projected classes keep their static
    members on a metaclass — ``OcrEngine.try_create_from_language`` is not an
    attribute of ``OcrEngine`` as far as a type checker can see — and a lazily bound
    class object has no useful static type here anyway.
    """

    loaded: ClassVar[bool] = False
    error: ClassVar[str] = ""
    ocr_engine: ClassVar[Any] = None
    language: ClassVar[Any] = None
    software_bitmap: ClassVar[Any] = None
    pixel_format: ClassVar[Any] = None

    @classmethod
    def load(cls) -> bool:
        """Import the projections. ``False`` when they are not installed."""
        if cls.loaded:
            return not cls.error
        cls.loaded = True
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
            from winrt.windows.media.ocr import OcrEngine as WinOcrEngine
        except ImportError as exc:  # packages absent: not Windows, or a trimmed install
            cls.error = str(exc)
            _log.debug("Windows.Media.Ocr is not importable: %s", exc)
            return False
        except OSError as exc:  # the runtime is there but refuses to activate
            cls.error = str(exc)
            _log.warning("Windows.Media.Ocr failed to load: %s", exc)
            return False
        cls.ocr_engine = WinOcrEngine
        cls.language = Language
        cls.software_bitmap = SoftwareBitmap
        cls.pixel_format = BitmapPixelFormat
        return True


#: Sentinel for the apartment complaint pywinrt raises: matched on the word rather
#: than on the whole sentence, which is Microsoft's wording and not ours.
_APARTMENT: Final = "apartment"


def _await(operation: Any) -> Any:
    """Wait for a WinRT async operation, whichever COM apartment we are called on.

    ``IAsyncOperation.get()`` is exactly the blocking wait an action on a worker
    thread wants — except that pywinrt refuses it when the calling thread is a
    single-threaded apartment, because blocking there stalls the message pump the STA
    itself needs to deliver the completion. Then it raises ``RuntimeError: Cannot
    call blocking method from single-threaded apartment``.

    Which is not hypothetical. Anything using ``comtypes`` in the process — the volume
    actions do — initialises the thread it ran on as an STA and never uninitialises
    it, so an OCR action landing on that same thread afterwards would fail for a
    reason that has nothing to do with OCR. So the wait moves onto a thread of our
    own, which WinRT initialises as a multithreaded apartment. Handing the operation
    across is safe: WinRT objects are agile unless they say otherwise.
    """
    try:
        return operation.get()
    except RuntimeError as exc:
        if _APARTMENT not in str(exc).lower():
            raise
    done: list[Any] = []
    failed: list[Exception] = []

    def wait() -> None:
        try:
            done.append(operation.get())
        except Exception as exc:  # re-raised below, on the thread that asked
            failed.append(exc)

    worker = threading.Thread(target=wait, name="ayris-ocr-await", daemon=True)
    worker.start()
    worker.join()
    if failed:
        raise failed[0]
    return done[0]


class WindowsOcr(OcrEngine):
    """Recognition through ``Windows.Media.Ocr``."""

    name: ClassVar[str] = "windows"
    title_ru: ClassVar[str] = "Windows OCR"
    # Trained on screen text, so a screenshot needs no help. Upscaling a full 4K
    # frame to 300 DPI would cost nine times the memory for nothing.
    wants_upscale: ClassVar[bool] = False

    def __init__(self, language: str) -> None:
        super().__init__(language)
        self._engine: Any = None

    @classmethod
    def is_available(cls) -> bool:
        return bool(cls.available_languages())

    @classmethod
    def available_languages(cls) -> tuple[str, ...]:
        """BCP-47 tags of the OCR language packs installed in Windows."""
        if not _Winrt.load():
            return ()
        try:
            languages = _Winrt.ocr_engine.available_recognizer_languages
            return tuple(str(item.language_tag) for item in languages)
        except OSError as exc:
            _log.warning("could not list OCR languages: %s", exc)
            return ()

    @classmethod
    def max_side(cls) -> int:
        """Longest side the recogniser accepts, 10000 on every Windows so far."""
        if not _Winrt.load():
            return 0
        try:
            return int(_Winrt.ocr_engine.max_image_dimension)
        except OSError:
            return 0

    @classmethod
    def describe_missing(cls) -> str:
        if not _Winrt.load():
            return (
                "Windows OCR не подключён: не установлены пакеты winrt. "
                "Переустанови Ayris или выбери в настройках Tesseract."
            )
        return (
            "В Windows не установлен языковой пакет распознавания. "
            "Параметры → Время и язык → Язык и регион → у нужного языка "
            "«Параметры языка» → «Оптическое распознавание символов»."
        )

    def _native(self) -> Any:
        """The ``OcrEngine`` for our language, created once.

        Raises:
            OcrEngineError: the projections are missing, or Windows has no data for
                the language we were built for.
        """
        if self._engine is not None:
            return self._engine
        if not _Winrt.load():
            raise OcrEngineError(
                f"winrt projections are not importable: {_Winrt.error}",
                user_message=self.describe_missing(),
            )
        try:
            engine = _Winrt.ocr_engine.try_create_from_language(_Winrt.language(self.language))
        except OSError as exc:
            raise OcrEngineError(
                f"OcrEngine.TryCreateFromLanguage({self.language!r}) failed: {exc}",
                user_message="Windows не смог запустить распознавание текста.",
            ) from exc
        if engine is None:
            raise OcrEngineError(
                f"no OCR language pack for {self.language!r}",
                user_message=self.describe_missing(),
            )
        self._engine = engine
        return engine

    def recognise(self, image: Image.Image) -> OcrText:
        engine = self._native()
        bitmap = self._to_bitmap(image)
        try:
            result = _await(engine.recognize_async(bitmap))
        except (OSError, RuntimeError) as exc:
            raise OcrEngineError(
                f"RecognizeAsync failed: {exc}",
                user_message="Windows не смог прочитать картинку.",
            ) from exc
        finally:
            bitmap.close()
        blocks = tuple(self._blocks(result))
        # Not ``result.text``: that joins every line with a space, so a screen comes
        # back as one endless sentence and ``OcrText.lines`` finds a single line in
        # it. The line structure is already in the blocks — use it.
        return OcrText(
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            engine=self.name,
            language=self.language,
        )

    def _to_bitmap(self, image: Image.Image) -> Any:
        """A ``SoftwareBitmap`` holding a copy of the image's grayscale pixels."""
        gray = image if image.mode == "L" else image.convert("L")
        width, height = gray.size
        try:
            return _Winrt.software_bitmap.create_copy_from_buffer(
                gray.tobytes(),
                getattr(_Winrt.pixel_format, _GRAY8.upper()),
                width,
                height,
            )
        except (OSError, ValueError) as exc:
            raise OcrEngineError(
                f"could not build a {width}×{height} SoftwareBitmap: {exc}",
                user_message="Не получилось передать картинку в распознавание.",
            ) from exc

    @staticmethod
    def _blocks(result: Any) -> list[OcrBlock]:
        """One block per recognised line, boxed around its words.

        ``OcrLine`` carries no rectangle, so the line's box is the union of the
        word boxes — which is also the only way to get a box for a line at all.
        """
        blocks: list[OcrBlock] = []
        for line in result.lines:
            rect = winapi.Rect()
            for word in line.words:
                box = word.bounding_rect
                left, top = int(box.x), int(box.y)
                word_rect = winapi.Rect(
                    left, top, left + int(round(box.width)), top + int(round(box.height))
                )
                rect = word_rect if rect.is_empty else _union(rect, word_rect)
            blocks.append(OcrBlock(text=str(line.text), rect=rect))
        return blocks


def _union(first: winapi.Rect, second: winapi.Rect) -> winapi.Rect:
    return winapi.Rect(
        left=min(first.left, second.left),
        top=min(first.top, second.top),
        right=max(first.right, second.right),
        bottom=max(first.bottom, second.bottom),
    )
