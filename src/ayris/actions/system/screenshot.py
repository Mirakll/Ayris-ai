"""Screenshots: what to capture, and where the picture goes.

The capture itself is four lines of ``mss``. Everything around it is what makes a
screenshot useful, and all of it is here:

* **Which rectangle.** All monitors as one image, one monitor by the address
  :mod:`ayris.utils.monitors` resolves, an arbitrary rectangle, or a window. Every
  one of them ends up as a rectangle in virtual-desktop physical pixels, and that
  space has no fixed origin — a monitor to the left of the primary one gives it a
  negative ``left``. The arithmetic is in free functions so it can be tested
  against monitor layouts that no single machine has.
* **Window bounds from the compositor**, not from ``GetWindowRect``: since Aero the
  resize border sits outside the visible frame and is transparent, so
  ``GetWindowRect`` overshoots by about seven pixels on three sides and the
  screenshot gets a strip of the desktop around the window.
* **Where it goes.** A file, the clipboard, or both, from
  ``[actions.screenshot] output``. The file name comes from a template, and the
  clipboard gets the picture twice — as ``CF_DIB``, which everything back to Paint
  understands, and as PNG, which the programs that care about alpha prefer.

Two things the capture cannot do anything about but must not hide. A frame can
come back entirely black — a window with hardware overlay, a DRM-protected video,
an exclusive-fullscreen game — and that is reported as the reason rather than
saved silently as a black rectangle. And without per-monitor DPI awareness the
frame arrives scaled or cropped on a monitor whose scaling is not 100%;
:func:`ayris.utils.dpi.enable_per_monitor_dpi_awareness` runs at startup, and this
module works in physical pixels throughout so nothing undoes it.

The interactive selection lives in :mod:`ayris.gui.widgets.region_selector` and is
reached through a seam, so this module imports no Qt: action modules are imported
by :meth:`ayris.actions.registry.ActionRegistry.discover` in worker processes that
have no GUI at all.
"""

from __future__ import annotations

import re
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

import mss
from PIL import Image
from pydantic import Field, model_validator

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.windows import WindowQuery, list_windows, select_window
from ayris.core.config import get_settings
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.core.paths import get_paths
from ayris.utils import monitors, winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "BLANK_LEVEL",
    "CaptureBackend",
    "CaptureFailed",
    "Frame",
    "MssCapture",
    "OutputMode",
    "Screenshot",
    "ScreenshotRegion",
    "ScreenshotWindow",
    "Shot",
    "build_path",
    "capture_all",
    "capture_monitor",
    "capture_rect",
    "capture_window",
    "clamp_rect",
    "deliver",
    "get_capture_backend",
    "get_region_provider",
    "normalize_rect",
    "render_filename",
    "safe_component",
    "set_capture_backend",
    "set_region_provider",
    "window_bounds",
]

_log = get_logger(__name__)

#: A capture is called black when no channel of any pixel gets above this. Not
#: zero, because a lossless capture of a dark desktop still has a pixel or two of
#: noise, and not higher, because a legitimately dark screenshot — a video paused
#: on a night scene — must not be reported as a failure.
BLANK_LEVEL: Final = 8

#: Smallest rectangle worth capturing. Below it the user has clicked rather than
#: dragged, and an 8×3 picture is never what they meant.
MIN_SIDE: Final = 4

#: Longest capture in pixels along one side. A quadruple-4K desktop is 15360 wide;
#: anything past this is a bad coordinate, and allocating for it would be a hang.
MAX_SIDE: Final = 32_768

#: How many names ``{n}`` and the overwrite guard will try before giving up.
MAX_SERIES: Final = 9_999

#: Characters Windows forbids in a file name, plus the control range.
_ILLEGAL_CHARS: Final = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

#: Device names that cannot be a file name on any Windows, extension or not.
_RESERVED_NAMES: Final = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)

#: ``{date}``-style placeholders. Unknown ones are left in the name untouched.
_PLACEHOLDER: Final = re.compile(r"\{(\w+)\}")

#: Longest one substituted value may be. A browser window title runs to hundreds
#: of characters and the whole path is capped at 260 on a default Windows.
_MAX_COMPONENT: Final = 64

#: Longest whole file name, extension included.
_MAX_STEM: Final = 120

#: How long to let the compositor redraw after the selection overlay hides. One
#: frame at 60 Hz is 17 ms; this is generous on purpose, because catching our own
#: dimmed overlay in the screenshot is worse than a tenth of a second of delay.
OVERLAY_SETTLE_S: Final = 0.12


class CaptureFailed(ActionError):
    """The screen could not be captured at all."""

    default_user_message = "Не получилось снять экран."


class OutputMode(StrEnum):
    """Where a finished screenshot goes."""

    FILE = "file"
    CLIPBOARD = "clipboard"
    BOTH = "both"

    @property
    def to_file(self) -> bool:
        return self in (OutputMode.FILE, OutputMode.BOTH)

    @property
    def to_clipboard(self) -> bool:
        return self in (OutputMode.CLIPBOARD, OutputMode.BOTH)

    @property
    def title_ru(self) -> str:
        return {
            OutputMode.FILE: "в файл",
            OutputMode.CLIPBOARD: "в буфер обмена",
            OutputMode.BOTH: "в файл и в буфер",
        }[self]


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured picture, still in the layout the screen gave it.

    ``bgra`` is ``height`` rows of ``width`` pixels, four bytes each, blue first
    and the top row first — what ``BitBlt`` writes and what ``mss`` hands over
    without touching. The fourth byte is whatever GDI left there and means
    nothing: :meth:`to_image` drops it, :meth:`to_dib` overwrites it with opaque.

    ``rect`` is where on the virtual desktop the pixels came from, so a caller can
    map a recognised word back to a screen coordinate. ``monitor`` and ``window``
    are labels for the file-name template and for the spoken confirmation.
    """

    width: int
    height: int
    bgra: bytes
    rect: winapi.Rect = field(default_factory=winapi.Rect)
    monitor: str = ""
    window: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"frame has no area: {self.width}×{self.height}")
        expected = self.width * self.height * 4
        if len(self.bgra) != expected:
            raise ValueError(
                f"frame is {len(self.bgra)} bytes, expected {expected} "
                f"for {self.width}×{self.height} BGRA"
            )

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def to_image(self) -> Image.Image:
        """The frame as an RGB image, alpha discarded.

        ``BGRX`` rather than ``BGRA`` on purpose: the fourth byte of a GDI capture
        is undefined, and reading it as alpha turns a perfectly good screenshot
        into a transparent one.
        """
        return Image.frombuffer("RGB", self.size, self.bgra, "raw", "BGRX", self.width * 4, 1)

    def to_png(self) -> bytes:
        """PNG bytes. What goes to disk and onto the clipboard."""
        buffer = BytesIO()
        # compress_level 6 rather than the default 9: on a 4K frame the last three
        # levels cost about half a second and save around two per cent.
        self.to_image().save(buffer, format="PNG", compress_level=6)
        return buffer.getvalue()

    def to_jpeg(self, quality: int = 92) -> bytes:
        """JPEG bytes, without chroma subsampling.

        ``subsampling=0`` costs a few per cent of size and is not optional for a
        screenshot: 4:2:0 averages colour over 2×2 blocks, which is exactly the
        scale of the text a screenshot is usually taken for.
        """
        buffer = BytesIO()
        self.to_image().save(buffer, format="JPEG", quality=quality, subsampling=0)
        return buffer.getvalue()

    def to_dib(self) -> bytes:
        """A bottom-up 32-bit ``BITMAPINFOHEADER`` DIB, as ``CF_DIB`` wants.

        Three differences from the captured buffer, all mandatory. The rows are
        reversed, because a positive ``biHeight`` means bottom-up and that is the
        only form every consumer reads. The fourth byte of every pixel is forced
        opaque: GDI leaves it undefined, and the programs that do look at it — a
        browser, Office — paste an invisible picture when it happens to be zero.
        And the ``BITMAPFILEHEADER`` is *not* prepended: that one belongs to a
        ``.bmp`` file, and its presence is what makes a paste come out as noise.
        """
        stride = self.width * 4
        view = memoryview(self.bgra)
        pixels = bytearray(
            b"".join(
                view[start : start + stride] for start in range(len(view) - stride, -1, -stride)
            )
        )
        pixels[3::4] = b"\xff" * self.pixels
        header = struct.pack(
            "<IiiHHIIiiII",
            40,  # biSize
            self.width,  # biWidth
            self.height,  # biHeight, positive: rows bottom-up
            1,  # biPlanes
            32,  # biBitCount
            0,  # biCompression = BI_RGB
            len(pixels),  # biSizeImage
            0,  # biXPelsPerMeter
            0,  # biYPelsPerMeter
            0,  # biClrUsed
            0,  # biClrImportant
        )
        return header + bytes(pixels)

    @property
    def is_blank(self) -> bool:
        """Whether nothing at all came through — see :data:`BLANK_LEVEL`.

        Read straight off the buffer, three channels at a time, because the fourth
        byte is the one GDI leaves undefined: a frame of pure black with the alpha
        byte set to ``0xff`` would look bright to anything that scanned all four.
        """
        brightest = max(max(self.bgra[channel::4]) for channel in range(3))
        return brightest <= BLANK_LEVEL

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready description. The pixels are deliberately not in it."""
        return {
            "width": self.width,
            "height": self.height,
            "rect": self.rect.as_dict(),
            "monitor": self.monitor,
            "window": self.window,
        }


@dataclass(frozen=True, slots=True)
class Shot:
    """A delivered screenshot: the frame, where it was written, and what was said.

    ``path`` is ``None`` when the mode was clipboard-only, and ``clipboard`` is
    ``False`` when it was file-only, so a caller can tell what actually happened
    rather than what was configured.
    """

    frame: Frame
    path: Path | None = None
    clipboard: bool = False
    blank: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.frame.as_dict(),
            "path": str(self.path) if self.path is not None else "",
            "clipboard": self.clipboard,
            "blank": self.blank,
        }


# --------------------------------------------------------------------------- #
# Rectangles
# --------------------------------------------------------------------------- #


def normalize_rect(x1: int, y1: int, x2: int, y2: int) -> winapi.Rect:
    """A rectangle from two corners in any order.

    A drag that went up and to the left is the normal case, not an error.
    """
    return winapi.Rect(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def clamp_rect(rect: winapi.Rect, bounds: winapi.Rect) -> winapi.Rect:
    """``rect`` cut down to what is actually on screen.

    Returns an empty rectangle when the two do not overlap — the caller decides
    whether that is worth an error, because «область целиком за экраном» and
    «пользователь ничего не выделил» deserve different sentences.

    The clamping is what makes a spoken «сними левую половину» safe: the numbers
    may name a region wider than the desktop, and ``mss`` asked for pixels that do
    not exist returns a frame padded with black instead of failing.
    """
    return winapi.Rect(
        left=max(rect.left, bounds.left),
        top=max(rect.top, bounds.top),
        right=min(rect.right, bounds.right),
        bottom=min(rect.bottom, bounds.bottom),
    )


def _check_rect(rect: winapi.Rect, *, what: str) -> None:
    """Reject a rectangle that cannot be captured, in Russian.

    Raises:
        CaptureFailed: the rectangle is empty, too thin to be a picture, or so
            large that it is a mistyped coordinate rather than a desktop.
    """
    if rect.is_empty:
        raise CaptureFailed(
            f"{what} is empty: {rect.as_tuple()}",
            user_message="Эта область не попадает на экран.",
        )
    if rect.width < MIN_SIDE or rect.height < MIN_SIDE:
        raise CaptureFailed(
            f"{what} is {rect.width}×{rect.height}, below the {MIN_SIDE}px minimum",
            user_message="Область слишком маленькая для снимка.",
        )
    if rect.width > MAX_SIDE or rect.height > MAX_SIDE:
        raise CaptureFailed(
            f"{what} is {rect.width}×{rect.height}, above the {MAX_SIDE}px maximum",
            user_message="Область слишком большая, что-то не так с координатами.",
        )


def window_bounds(hwnd: int) -> winapi.Rect:
    """What the window covers on screen, shadow margins excluded.

    :func:`ayris.utils.winapi.extended_frame_bounds` asks the compositor and falls
    back to ``GetWindowRect``; the fallback captures a few pixels of whatever is
    behind the window on three sides, which is worth having when the alternative
    is no screenshot.
    """
    return winapi.extended_frame_bounds(hwnd)


# --------------------------------------------------------------------------- #
# Capture backend
# --------------------------------------------------------------------------- #


class CaptureBackend(Protocol):
    """Everything this module needs from the screen. Faked wholesale in tests."""

    def grab(self, rect: winapi.Rect) -> Frame:
        """Pixels of one rectangle of the virtual desktop, in physical pixels."""
        ...


class MssCapture:
    """The real backend: ``mss``, one grabber per call.

    A grabber is not safe to share between threads — it holds a device context —
    and actions run on a pool, so it is created and closed inside the call. That
    costs around a millisecond, which is nothing next to the copy of the pixels.
    """

    def grab(self, rect: winapi.Rect) -> Frame:
        region = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        try:
            with mss.mss() as grabber:
                shot = grabber.grab(region)
                raw = bytes(shot.bgra)
                width, height = int(shot.width), int(shot.height)
        except mss.exception.ScreenShotError as exc:
            raise CaptureFailed(
                f"mss refused to grab {region}: {exc}",
                user_message="Не получилось снять экран — Windows не отдал картинку.",
            ) from exc
        except OSError as exc:  # no display at all: a service session, or Linux
            raise ActionUnavailable(
                f"no screen to capture: {exc}",
                user_message="Не вижу экрана, с которого можно снять картинку.",
            ) from exc
        return Frame(width=width, height=height, bgra=raw, rect=rect)


_backend: CaptureBackend | None = None


def get_capture_backend() -> CaptureBackend:
    """The backend in force. Real ``mss`` unless a test replaced it."""
    if _backend is not None:
        return _backend
    return MssCapture()


def set_capture_backend(backend: CaptureBackend | None) -> None:
    """Install a backend, or restore the real one with ``None``. Test seam."""
    global _backend
    _backend = backend


# --------------------------------------------------------------------------- #
# Interactive selection, through a seam so that Qt stays out of this module
# --------------------------------------------------------------------------- #

_region_provider: Callable[[float, float], winapi.Rect | None] | None = None


def _qt_region_provider(timeout_s: float, dim: float) -> winapi.Rect | None:
    """Ask the user to drag a rectangle, using the Qt overlay.

    Imported here and not at module scope: :mod:`ayris.actions` is imported in
    worker processes that never build a ``QApplication``, and importing PySide6
    there costs both memory and a chance of failing on a machine without the
    Visual C++ runtime.
    """
    from ayris.gui.widgets.region_selector import select_region

    return select_region(timeout_s=timeout_s, dim=dim)


def get_region_provider() -> Callable[[float, float], winapi.Rect | None]:
    """Whatever will be asked for a rectangle. The Qt overlay unless replaced."""
    if _region_provider is not None:
        return _region_provider
    return _qt_region_provider


def set_region_provider(
    provider: Callable[[float, float], winapi.Rect | None] | None,
) -> None:
    """Install a selection provider, or restore the overlay with ``None``."""
    global _region_provider
    _region_provider = provider


# --------------------------------------------------------------------------- #
# Capturing
# --------------------------------------------------------------------------- #


def capture_rect(
    rect: winapi.Rect,
    *,
    bounds: winapi.Rect | None = None,
    monitor: str = "",
    window: str = "",
    backend: CaptureBackend | None = None,
) -> Frame:
    """Capture one rectangle of the virtual desktop.

    ``bounds`` is what to clamp against, the whole virtual desktop by default.
    Pass it explicitly when capturing inside one monitor, so a rectangle that
    spills onto the neighbour is trimmed rather than silently widened.

    Raises:
        CaptureFailed: the rectangle is off-screen, degenerate or absurd.
    """
    limits = monitors.virtual_bounds() if bounds is None else bounds
    wanted = clamp_rect(rect, limits) if not limits.is_empty else rect
    _check_rect(wanted, what="capture rect")
    frame = (backend or get_capture_backend()).grab(wanted)
    if (frame.monitor, frame.window) == (monitor, window):
        return frame
    return Frame(
        width=frame.width,
        height=frame.height,
        bgra=frame.bgra,
        rect=frame.rect if not frame.rect.is_empty else wanted,
        monitor=monitor,
        window=window,
    )


def capture_all(*, backend: CaptureBackend | None = None) -> Frame:
    """Every monitor in one image, in the layout the user sees them.

    The gap two monitors of different heights leave between them is captured as
    black — it is part of the bounding box and there is nothing else to put there.
    """
    bounds = monitors.virtual_bounds()
    if bounds.is_empty:
        raise ActionUnavailable(
            "the virtual desktop has no area",
            user_message="Не вижу ни одного монитора.",
        )
    return capture_rect(bounds, bounds=bounds, monitor="все мониторы", backend=backend)


def capture_monitor(
    address: str | int | None = None,
    *,
    backend: CaptureBackend | None = None,
) -> Frame:
    """One monitor, addressed the way the whole application addresses monitors.

    Raises:
        CaptureFailed: no monitor answers to that address.
    """
    try:
        display = monitors.resolve_monitor(address)
    except monitors.MonitorNotFound as exc:
        attached = ", ".join(item.label for item in exc.available)
        raise CaptureFailed(
            str(exc),
            user_message=(
                f"Не нашла монитор «{address}»." + (f" Подключены: {attached}." if attached else "")
            ),
        ) from exc
    return capture_rect(
        display.rect,
        bounds=display.rect,
        monitor=display.label,
        backend=backend,
    )


def capture_window(hwnd: int, *, title: str = "", backend: CaptureBackend | None = None) -> Frame:
    """One window, cropped to what it visibly covers.

    The window is captured where it sits, so anything on top of it is captured
    too. Raising it first is the caller's decision — a screenshot of a background
    window is a legitimate thing to want, and stealing the focus to take one is
    not something an assistant should do behind the user's back.
    """
    frame_rect = window_bounds(hwnd)
    if frame_rect.is_empty:
        raise CaptureFailed(
            f"window {hwnd:#x} has no visible frame",
            user_message="Это окно свёрнуто или у него нет видимой рамки.",
        )
    return capture_rect(frame_rect, monitor="", window=title, backend=backend)


def capture_region(
    *,
    timeout_s: float,
    dim: float,
    backend: CaptureBackend | None = None,
) -> Frame | None:
    """Let the user drag a rectangle, then capture it. ``None`` when cancelled.

    The overlay hides itself before this returns, and the capture waits
    :data:`OVERLAY_SETTLE_S` on top of that: hiding a window is a request to the
    compositor, not an event that has already happened, and grabbing too early
    catches our own dimming in the picture.
    """
    selected = get_region_provider()(timeout_s, dim)
    if selected is None:
        return None
    time.sleep(OVERLAY_SETTLE_S)
    return capture_rect(selected, backend=backend)


# --------------------------------------------------------------------------- #
# Names and files
# --------------------------------------------------------------------------- #


def safe_component(text: str, *, limit: int = _MAX_COMPONENT) -> str:
    """Turn a window title or a monitor name into part of a file name.

    Windows forbids nine characters and the whole control range, silently drops
    trailing dots and spaces, and refuses a name that *starts* with a reserved
    device word — ``nul.png`` cannot be created however hard one tries. Long
    titles are cut because the path as a whole has a limit and a browser tab title
    can be several hundred characters on its own.
    """
    cleaned = _ILLEGAL_CHARS.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        return ""
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].strip(" .")
    if cleaned.split(".", 1)[0].lower() in _RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def render_filename(
    template: str,
    *,
    when: datetime,
    monitor: str = "",
    window: str = "",
    index: int = 1,
) -> str:
    """Fill in a file-name template, without an extension.

    ``{date}`` is ``2026-08-24`` and ``{time}`` is ``14-32-05`` — dashes, because a
    colon cannot appear in a Windows file name and a name with one silently fails
    to be created. ``{monitor}`` and ``{window}`` are cleaned by
    :func:`safe_component` and collapse to nothing when there is nothing to put
    there, taking the separator next to them along. An unknown placeholder is left
    exactly as written: a typo in the settings should produce a strange file name,
    not an exception in the middle of a screenshot.
    """
    values = {
        "date": when.strftime("%Y-%m-%d"),
        "time": when.strftime("%H-%M-%S"),
        "monitor": safe_component(monitor),
        "window": safe_component(window),
        "n": str(index),
    }

    def substitute(match: re.Match[str]) -> str:
        return values.get(match.group(1), match.group(0))

    filled = _PLACEHOLDER.sub(substitute, template)
    # An absent {window} leaves «ayris__2026-08-24»; collapse the doubled
    # separators it left behind rather than shipping them in the name.
    filled = re.sub(r"[ _-]{2,}", "_", filled)
    filled = safe_component(filled, limit=_MAX_STEM)
    return filled or "screenshot"


def build_path(
    directory: Path,
    template: str,
    *,
    extension: str,
    when: datetime,
    monitor: str = "",
    window: str = "",
) -> Path:
    """A file path that does not exist yet.

    Two ways of getting there, and which one applies is decided by the template.
    With ``{n}`` in it the number is a series counter: the first free number wins,
    so a run of screenshots comes out as ``shot_1``, ``shot_2``, ``shot_3``.
    Without it the rendered name is used as-is, and only if something is already
    there does ``_2`` get appended — a name with a timestamp in it collides about
    once a year, and renaming it every time would be noise.

    Raises:
        CaptureFailed: the directory cannot be created, or all
            :data:`MAX_SERIES` candidate names are taken.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CaptureFailed(
            f"cannot create {directory}: {exc}",
            user_message=f"Не получилось создать папку для снимков: {directory}",
        ) from exc

    suffix = extension if extension.startswith(".") else f".{extension}"
    if "{n}" in template:
        for number in range(1, MAX_SERIES + 1):
            stem = render_filename(
                template, when=when, monitor=monitor, window=window, index=number
            )
            candidate = directory / f"{stem}{suffix}"
            if not candidate.exists():
                return candidate
        raise CaptureFailed(
            f"all {MAX_SERIES} numbered names are taken in {directory}",
            user_message="В папке слишком много снимков, не могу придумать имя.",
        )

    stem = render_filename(template, when=when, monitor=monitor, window=window)
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for number in range(2, MAX_SERIES + 1):
        candidate = directory / f"{stem}_{number}{suffix}"
        if not candidate.exists():
            return candidate
    raise CaptureFailed(
        f"cannot find a free name for {stem} in {directory}",
        user_message="В папке слишком много снимков, не могу придумать имя.",
    )


def _output_directory(configured: str) -> Path:
    """Where files go: the configured folder, or the profile's own."""
    text = configured.strip()
    if not text:
        return get_paths().screenshots_dir
    return Path(text).expanduser()


def save_frame(
    frame: Frame,
    *,
    directory: Path | None = None,
    template: str | None = None,
    image_format: str | None = None,
    quality: int | None = None,
    when: datetime | None = None,
) -> Path:
    """Write the frame to a new file and return where it went.

    Every argument defaults to ``[actions.screenshot]``; they exist so the OCR
    actions and the tests can override one thing without assembling a config.

    Raises:
        CaptureFailed: the file could not be written.
    """
    settings = get_settings().actions.screenshot
    kind = (image_format or settings.format).lower()
    target = build_path(
        directory if directory is not None else _output_directory(settings.directory),
        template or settings.filename,
        extension="jpg" if kind == "jpeg" else "png",
        when=when or datetime.now(),
        monitor=frame.monitor,
        window=frame.window,
    )
    payload = (
        frame.to_jpeg(quality if quality is not None else settings.jpeg_quality)
        if kind == "jpeg"
        else frame.to_png()
    )
    try:
        target.write_bytes(payload)
    except OSError as exc:
        raise CaptureFailed(
            f"cannot write {target}: {exc}",
            user_message=f"Не получилось сохранить снимок в {target.parent}.",
        ) from exc
    _log.debug("saved %d×%d screenshot to %s", frame.width, frame.height, target)
    return target


def copy_frame(frame: Frame) -> bool:
    """Put the frame on the clipboard as ``CF_DIB`` and as PNG.

    Returns ``False`` instead of raising when the clipboard refuses: the file has
    usually been written by then, and losing the copy is worth a sentence, not a
    failed action.
    """
    payloads: list[tuple[int, bytes]] = [(winapi.CF_DIB, frame.to_dib())]
    png_format = winapi.register_clipboard_format("PNG")
    if png_format:
        payloads.append((png_format, frame.to_png()))
    try:
        winapi.clipboard_set_binary(payloads)
    except winapi.WinApiError as exc:
        _log.warning("clipboard refused the screenshot: %s", exc)
        return False
    return True


def deliver(frame: Frame, mode: OutputMode | None = None, **saving: Any) -> Shot:
    """Save and/or copy the frame according to ``mode``, and report what happened.

    ``mode`` defaults to ``[actions.screenshot] output``. A black frame is
    delivered like any other — see :meth:`Frame.is_blank` for why it is flagged
    rather than refused.
    """
    settings = get_settings().actions.screenshot
    chosen = mode if mode is not None else OutputMode(settings.output)
    path = save_frame(frame, **saving) if chosen.to_file else None
    copied = copy_frame(frame) if chosen.to_clipboard else False
    return Shot(frame=frame, path=path, clipboard=copied, blank=frame.is_blank)


def _spoken(shot: Shot, *, what: str) -> str:
    """The sentence Ayris says about a finished screenshot."""
    size = f"{shot.frame.width}×{shot.frame.height}"
    where: list[str] = []
    if shot.path is not None:
        where.append(f"сохранила как {shot.path.name}")
    if shot.clipboard:
        where.append("скопировала в буфер")
    tail = ", ".join(where) if where else "никуда не сохранила"
    line = f"Снимок {what} {size} — {tail}."
    if shot.blank:
        line += (
            " Кадр вышел чёрным: так бывает с защищённым видео, "
            "полноэкранной игрой или окном с аппаратным ускорением."
        )
    return line


class _OutputParams(ActionParams):
    """The output override every screenshot action shares."""

    output: OutputMode | None = Field(
        default=None,
        description="Куда девать снимок; пусто — как в настройках",
        json_schema_extra={
            "choices_ru": {str(mode): mode.title_ru for mode in OutputMode},
        },
    )


@register
class Screenshot(Action):
    """Snapshot of every monitor at once, or of one named monitor."""

    meta: ClassVar = ActionMeta(
        name="Screenshot",
        category=ActionCategory.CAPTURE,
        title_ru="Снимок экрана",
        description_ru="Снять все мониторы одним изображением или один монитор",
        timeout_ms=20_000,
    )

    class Params(_OutputParams):
        monitor: str = Field(
            default="",
            max_length=120,
            title="Монитор",
            description="primary, external_1, номер или часть имени; пусто — все мониторы",
        )

    def run(self, params: Params) -> ActionResult[Shot]:
        address = params.monitor.strip()
        frame = capture_monitor(address) if address else capture_all()
        shot = deliver(frame, params.output)
        what = f"монитора «{frame.monitor}»" if address else "всех мониторов"
        return ActionResult.done(_spoken(shot, what=what), value=shot, data=shot.as_dict())


@register
class ScreenshotRegion(Action):
    """Snapshot of a rectangle, either given in numbers or dragged with the mouse."""

    meta: ClassVar = ActionMeta(
        name="ScreenshotRegion",
        category=ActionCategory.CAPTURE,
        title_ru="Снимок области",
        description_ru="Снять прямоугольную область экрана, выделив её мышью или задав координаты",
        # Long, because the interactive branch waits for a person: the selection
        # has its own [actions.screenshot] selection_timeout_s, and this only has
        # to be comfortably longer than that so the two do not race.
        timeout_ms=180_000,
    )

    class Params(_OutputParams):
        left: int = Field(default=0, ge=-MAX_SIDE, le=MAX_SIDE, description="Левый край области")
        top: int = Field(default=0, ge=-MAX_SIDE, le=MAX_SIDE, description="Верхний край области")
        width: int = Field(default=0, ge=0, le=MAX_SIDE, description="Ширина области")
        height: int = Field(default=0, ge=0, le=MAX_SIDE, description="Высота области")

        @property
        def has_rect(self) -> bool:
            """Whether numbers were given, so nothing has to be asked of the user."""
            return self.width > 0 and self.height > 0

        @model_validator(mode="after")
        def _both_sides(self) -> ScreenshotRegion.Params:
            """A width without a height is half a rectangle, not a default."""
            if (self.width > 0) != (self.height > 0):
                raise ValueError("укажите и ширину, и высоту области")
            return self

    def run(self, params: Params) -> ActionResult[Shot]:
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
        shot = deliver(frame, params.output)
        return ActionResult.done(_spoken(shot, what="области"), value=shot, data=shot.as_dict())


@register
class ScreenshotWindow(Action):
    """Snapshot of one window, cropped to its visible frame."""

    meta: ClassVar = ActionMeta(
        name="ScreenshotWindow",
        category=ActionCategory.CAPTURE,
        title_ru="Снимок окна",
        description_ru="Снять активное окно или найденное по названию, без полей тени",
        timeout_ms=20_000,
    )

    class Params(_OutputParams):
        title: str = Field(
            default="",
            max_length=200,
            title="Название окна",
            description="Часть заголовка; пусто — активное окно",
        )
        process: str = Field(
            default="",
            max_length=120,
            title="Программа",
            description="Имя процесса, если заголовок не помогает",
        )

    def run(self, params: Params) -> ActionResult[Shot]:
        query = WindowQuery(title=params.title.strip(), process=params.process.strip())
        record = select_window(list_windows(query), query)
        frame = capture_window(record.hwnd, title=record.label)
        shot = deliver(frame, params.output)
        return ActionResult.done(
            _spoken(shot, what=f"окна «{record.label}»"),
            value=shot,
            data={**shot.as_dict(), "hwnd": record.hwnd},
        )


def describe_layout(displays: Sequence[monitors.MonitorInfo] | None = None) -> str:
    """One line per monitor, for the log when a capture comes out wrong.

    The three numbers that explain almost every «скриншот получился не такой»:
    where the monitor starts, how big it is in physical pixels, and its scaling.
    """
    available = monitors.list_monitors() if displays is None else displays
    return "; ".join(
        f"{item.address} {item.rect.width}×{item.rect.height}"
        f"@{item.rect.left},{item.rect.top} ×{item.scale:g}"
        for item in available
    )
