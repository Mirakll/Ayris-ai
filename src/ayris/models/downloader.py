"""Streaming downloads with resume, progress and cancellation.

Models are big — a Vosk archive is 45 MB, a whisper.cpp ``large`` is 1.5 GB — and
most of what is here follows from that one fact.

**Nothing is held in memory.** Bytes go from the socket to the file in 64 KiB
chunks, and the SHA-256 is fed the same chunks on the way past. Reading the
finished file back to verify it would double the I/O for no gain; buffering it to
verify would need a spare gigabyte of RAM.

**An interrupted download resumes.** The partial file lives in the work directory
as ``<model>__<file>.part`` with a small JSON sidecar recording the URL, the
length and whatever validator the server offered. The next attempt asks for
``Range: bytes=N-`` and, when a validator was stored, ``If-Range``. That single
header is the whole safety argument: a server whose copy still matches answers
``206`` and the transfer continues, a server whose copy changed answers ``200``
with the whole file, and the partial bytes are dropped. Splicing half of one
version onto half of another would produce a file that looks plausible and fails
only at the checksum, an hour later.

Resuming has a cost that is easy to miss: :mod:`hashlib` state cannot be
persisted, so bytes already on disk are re-hashed before new ones are appended to
the digest. That is a local read at disk speed, and still far cheaper than
fetching them again.

**Cancellation is prompt.** :class:`DownloadHandle` is checked between chunks, so
the worst case is one chunk of latency. Cancelling keeps the ``.part`` file: the
user who cancelled at 90 % because they needed the bandwidth should not lose the
90 %.

**One download per model.** A per-``model_id`` lock stops two callers from writing
one ``.part`` file through two handles, which on Windows fails outright and on
Linux silently interleaves.

Progress goes to two audiences: the ``on_progress`` callback fires on the
downloading thread for a caller that wants every update, and
:class:`~ayris.core.events.ModelDownloadProgress` reaches the bus a few times a
second for the UI. Both are throttled by wall clock rather than by chunk count,
so a fast link and a slow one produce the same event rate.

There is no bandwidth limit: the task does not ask for one, and a download the
user started and is watching should use the link they have.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import httpx

from ayris.core.errors import ModelError
from ayris.core.events import ModelDownloadFailed, ModelDownloadProgress
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from ayris.core.events import EventBus
    from ayris.models.catalog import ModelEntry, ModelFile

__all__ = [
    "CHUNK_BYTES",
    "META_SUFFIX",
    "PART_SUFFIX",
    "SPACE_MARGIN_BYTES",
    "DownloadCancelled",
    "DownloadError",
    "DownloadHandle",
    "DownloadResult",
    "Downloader",
    "IntegrityError",
    "NotEnoughSpaceError",
    "ProgressCallback",
    "human_size",
    "sha256_file",
]

_log = get_logger(__name__)

#: Bytes per read. Large enough that per-chunk bookkeeping disappears next to the
#: I/O, small enough that cancellation is felt immediately.
CHUNK_BYTES: Final = 64 * 1024

#: Extension of the partial file.
PART_SUFFIX: Final = ".part"

#: Extension of the sidecar holding the resume metadata.
META_SUFFIX: Final = ".part.json"

#: Slack demanded on top of the download size. Windows behaves badly on a volume
#: with nothing left, and an unpacked model needs somewhere to go.
SPACE_MARGIN_BYTES: Final = 64 * 1024 * 1024

#: Seconds between progress notifications.
PROGRESS_INTERVAL_SEC: Final = 0.2

#: Connect and per-read timeouts. The read timeout applies to one chunk, not to
#: the whole transfer, so a slow link is fine and a dead one is not.
CONNECT_TIMEOUT_SEC: Final = 10.0
READ_TIMEOUT_SEC: Final = 60.0

#: Retries for a connection that drops mid-transfer. Each one resumes from the
#: bytes already written, so a retry costs a request rather than a restart.
MAX_RETRIES: Final = 3
BACKOFF_BASE_SEC: Final = 1.0
BACKOFF_MAX_SEC: Final = 8.0

_USER_AGENT: Final = "Ayris"

#: Window the reported speed is averaged over. One chunk divided by one interval
#: swings wildly; the whole transfer averaged from the start reacts too slowly to
#: a link that changes. The ETA is derived from this.
_SPEED_WINDOW_SEC: Final = 5.0

#: HTTP status codes this module reasons about by name.
_HTTP_OK: Final = 200
_HTTP_PARTIAL: Final = 206
_HTTP_ERROR: Final = 400


class DownloadError(ModelError):
    """A download failed for a reason the user may be able to fix."""

    default_user_message = "Не удалось скачать модель."


class DownloadCancelled(DownloadError):
    """The download was cancelled. Not a fault, but it unwinds like one."""

    default_user_message = "Загрузка отменена."


class IntegrityError(ModelError):
    """Downloaded bytes do not match the expected SHA-256."""

    default_user_message = "Файл модели повреждён: контрольная сумма не совпала."


class NotEnoughSpaceError(ModelError):
    """The destination volume cannot hold the model."""

    default_user_message = "Недостаточно места на диске."


#: Called on the downloading thread, so it must be quick and must not touch the
#: UI directly. The signature is the one the task specifies:
#: ``(model_id, downloaded, total, speed_bps, eta_s)``.
ProgressCallback = Callable[[str, int, int, float, float], None]


class DownloadHandle:
    """Cancellation token for one download.

    Handed to the caller before the transfer starts, so a UI can wire it to a
    button. :meth:`cancel` is safe from any thread and takes effect within one
    chunk.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask the download to stop. Idempotent."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise DownloadCancelled("download cancelled by the caller")


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Where a finished download landed and what it cost."""

    model_id: str
    target: str
    path: Path
    size_bytes: int
    sha256: str
    duration_ms: int = 0
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class _Resume:
    """What the sidecar remembers about an interrupted transfer."""

    url: str
    total: int = 0
    etag: str = ""
    last_modified: str = ""

    @property
    def validator(self) -> str:
        """Value for ``If-Range``: an ``ETag`` if there is one, else a date.

        Both forms are allowed by RFC 9110, and between them they cover the CDNs
        the catalog points at and a plain static file server.
        """
        return self.etag or self.last_modified


class _Sink:
    """The ``.part`` file and its running digest, kept in step.

    They have to move together — appending bytes without hashing them, or hashing
    without appending, produces a file that fails verification for no visible
    reason — so one object owns both and nothing else touches either.
    """

    __slots__ = ("_digest", "path", "written")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.written = 0
        self._digest = hashlib.sha256()

    def adopt(self, handle: DownloadHandle) -> int:
        """Take over the bytes already on disk, feeding them through the digest."""
        self._digest = hashlib.sha256()
        self.written = 0
        if not self.path.is_file():
            return 0
        with self.path.open("rb") as source:
            while True:
                handle.raise_if_cancelled()
                chunk = source.read(CHUNK_BYTES)
                if not chunk:
                    break
                self._digest.update(chunk)
                self.written += len(chunk)
        return self.written

    def reset(self) -> None:
        """Throw away everything and start from an empty file."""
        self._digest = hashlib.sha256()
        self.written = 0
        self.path.unlink(missing_ok=True)

    def absorb(
        self,
        chunks: Iterable[bytes],
        handle: DownloadHandle,
        reporter: _Reporter,
    ) -> None:
        """Append an iterable of byte chunks, hashing and reporting as it goes."""
        mode = "ab" if self.written else "wb"
        with self.path.open(mode) as target:
            for chunk in chunks:
                handle.raise_if_cancelled()
                target.write(chunk)
                self._digest.update(chunk)
                self.written += len(chunk)
                reporter.update(self.written)
        reporter.flush(self.written)

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


class Downloader:
    """Fetches catalog artifacts into a working directory.

    Args:
        work_dir: Where ``.part`` files live and where a verified file waits for
            the installer to move it. Anything left here is a resumable download.
        bus: Event bus for progress and failure events. Optional, because the
            worker-side callers have no bus.
        transport: Injected by tests. The project's rule is that no test opens a
            socket, so every HTTP path here runs through
            :class:`httpx.MockTransport` in the suite.
        client: A ready-made client, if the caller already has one. Takes
            precedence over ``transport`` and is not closed by the downloader.
    """

    __slots__ = ("_bus", "_client", "_locks", "_locks_guard", "_owns_client", "_work_dir")

    def __init__(
        self,
        work_dir: Path,
        *,
        bus: EventBus | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._bus = bus
        self._owns_client = client is None
        self._client = client if client is not None else _build_client(transport)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the HTTP client, unless it was handed in."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Downloader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def part_path(self, model_id: str, file: ModelFile) -> Path:
        """Where the partial download for one artifact lives."""
        return self._work_dir / f"{model_id}__{file.target}{PART_SUFFIX}"

    def staged_path(self, model_id: str, file: ModelFile) -> Path:
        """Where a verified artifact waits for the installer."""
        return self._work_dir / f"{model_id}__{file.target}"

    def pending_bytes(self, model_id: str, file: ModelFile) -> int:
        """Bytes already on disk for this artifact, ``0`` if none."""
        part = self.part_path(model_id, file)
        return part.stat().st_size if part.is_file() else 0

    def check_space(self, entry: ModelEntry, destination: Path) -> None:
        """Refuse to start a download that cannot possibly fit.

        Checked before the first byte rather than discovered at the last one. An
        archive counts twice, because the download and the unpacked result
        coexist until the installer removes the archive.

        Raises:
            NotEnoughSpaceError: The volume holding ``destination`` is too small.
        """
        needed = entry.total_bytes * (2 if entry.is_archive else 1) + SPACE_MARGIN_BYTES
        free = _free_bytes(destination)
        if free is None or free >= needed:
            return
        raise NotEnoughSpaceError(
            f"need {needed} bytes for {entry.id}, {free} free at {destination}",
            user_message=(
                f"Недостаточно места для «{entry.name}».\n"
                f"Требуется {human_size(needed)}, свободно {human_size(free)}."
            ),
        )

    def discard(self, model_id: str, entry: ModelEntry) -> None:
        """Delete every partial and staged file of a model.

        What "скачать заново" calls after a failure the user chose not to resume.
        """
        for file in entry.files:
            part = self.part_path(model_id, file)
            part.unlink(missing_ok=True)
            _meta_path(part).unlink(missing_ok=True)
            self.staged_path(model_id, file).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # downloading
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        entry: ModelEntry,
        *,
        handle: DownloadHandle | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[DownloadResult, ...]:
        """Download every artifact of a model, primary first.

        A Piper voice is two files and is useless as one, so the caller gets all
        or nothing. Progress runs continuously across them instead of restarting
        per file, which is why each :meth:`fetch` is told what came before it.
        """
        token = handle if handle is not None else DownloadHandle()
        lock = self._lock_for(entry.id)
        if not lock.acquire(blocking=False):
            raise DownloadError(
                f"model {entry.id} is already downloading",
                user_message=f"Модель «{entry.name}» уже загружается.",
            )
        try:
            results: list[DownloadResult] = []
            done = 0
            for file in entry.files:
                result = self._fetch(
                    entry.id,
                    file,
                    handle=token,
                    on_progress=on_progress,
                    offset=done,
                    total=entry.total_bytes,
                )
                results.append(result)
                done += result.size_bytes
            return tuple(results)
        finally:
            lock.release()

    def fetch(
        self,
        model_id: str,
        file: ModelFile,
        *,
        handle: DownloadHandle | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> DownloadResult:
        """Download one artifact into the work directory and verify it.

        Raises:
            DownloadCancelled: :meth:`DownloadHandle.cancel` was called. The
                partial file is kept so the next attempt can resume.
            IntegrityError: The checksum did not match. The file is deleted,
                because a corrupt model that stays on disk gets resumed into
                forever.
            DownloadError: Network failure, an HTTP error status, or a disk that
                would not take the bytes.
        """
        token = handle if handle is not None else DownloadHandle()
        lock = self._lock_for(model_id)
        if not lock.acquire(blocking=False):
            raise DownloadError(
                f"model {model_id} is already downloading",
                user_message="Эта модель уже загружается.",
            )
        try:
            return self._fetch(
                model_id,
                file,
                handle=token,
                on_progress=on_progress,
                offset=0,
                total=file.size_bytes,
            )
        finally:
            lock.release()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fetch(
        self,
        model_id: str,
        file: ModelFile,
        *,
        handle: DownloadHandle,
        on_progress: ProgressCallback | None,
        offset: int,
        total: int,
    ) -> DownloadResult:
        self._work_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        sink = _Sink(self.part_path(model_id, file))
        meta = _meta_path(sink.path)
        resume = _plan_resume(sink.path, meta, file)
        if resume is None:
            sink.reset()
            meta.unlink(missing_ok=True)
        else:
            sink.adopt(handle)

        reporter = _Reporter(
            model_id=model_id,
            bus=self._bus,
            callback=on_progress,
            offset=offset,
            total=total or file.size_bytes,
        )
        try:
            self._stream(sink, meta, file, resume, handle, reporter)
            return self._finish(model_id, file, sink, meta, started)
        except DownloadCancelled as exc:
            self._publish_failed(model_id, exc, cancelled=True, resumable=sink.path.is_file())
            raise
        except ModelError as exc:
            self._publish_failed(model_id, exc, cancelled=False, resumable=sink.path.is_file())
            raise
        except OSError as exc:
            wrapped = DownloadError(
                f"cannot write {sink.path}: {exc}",
                user_message=f"Не удалось записать файл модели:\n{sink.path}",
            )
            self._publish_failed(model_id, wrapped, cancelled=False, resumable=False)
            raise wrapped from exc

    def _finish(
        self,
        model_id: str,
        file: ModelFile,
        sink: _Sink,
        meta: Path,
        started: float,
    ) -> DownloadResult:
        actual = sink.hexdigest
        if actual != file.sha256:
            sink.discard()
            meta.unlink(missing_ok=True)
            # Not published here: the caller's ``except ModelError`` publishes the
            # failure, and doing both said one download failed twice.
            raise IntegrityError(
                f"sha256 mismatch for {model_id}/{file.target}: "
                f"expected {file.sha256}, got {actual}",
                user_message=(
                    f"Файл «{file.target}» скачан с ошибкой: контрольная сумма не совпала.\n"
                    "Файл удалён, попробуйте скачать заново."
                ),
            )

        final = self.staged_path(model_id, file)
        final.unlink(missing_ok=True)
        sink.path.replace(final)
        meta.unlink(missing_ok=True)
        duration_ms = int((time.monotonic() - started) * 1000)
        _log.info(
            "модель %s: файл %s готов (%s за %d мс)",
            model_id,
            file.target,
            human_size(sink.written),
            duration_ms,
        )
        return DownloadResult(
            model_id=model_id,
            target=file.target,
            path=final,
            size_bytes=sink.written,
            sha256=actual,
            duration_ms=duration_ms,
            resumed=False,
        )

    def _stream(
        self,
        sink: _Sink,
        meta: Path,
        file: ModelFile,
        resume: _Resume | None,
        handle: DownloadHandle,
        reporter: _Reporter,
    ) -> None:
        """Run the transfer, retrying a dropped connection where it stopped."""
        attempt = 0
        while True:
            handle.raise_if_cancelled()
            try:
                self._stream_once(sink, meta, file, resume, handle, reporter)
            except (httpx.HTTPError, OSError) as exc:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise DownloadError(
                        f"download of {file.url} failed after {MAX_RETRIES} retries: {exc}",
                        user_message=(
                            f"Не удалось скачать файл «{file.target}».\n"
                            "Проверьте подключение к интернету и попробуйте снова."
                        ),
                    ) from exc
                if handle.cancelled:
                    raise DownloadCancelled("download cancelled by the caller") from exc
                delay = min(BACKOFF_BASE_SEC * 2 ** (attempt - 1), BACKOFF_MAX_SEC)
                _log.warning(
                    "обрыв загрузки %s (%s), попытка %d через %.1f с",
                    file.target,
                    exc,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                # Re-read what actually reached the disk: the failure may have
                # landed between the write and the digest update, and the file is
                # the only thing that knows the truth.
                sink.adopt(handle)
                resume = _read_meta(meta)
                reporter.restart(sink.written)
                continue
            return

    def _stream_once(
        self,
        sink: _Sink,
        meta: Path,
        file: ModelFile,
        resume: _Resume | None,
        handle: DownloadHandle,
        reporter: _Reporter,
    ) -> None:
        headers = {"Accept-Encoding": "identity"}
        if sink.written:
            headers["Range"] = f"bytes={sink.written}-"
            # If-Range turns "give me the tail" into "give me the tail only if
            # the file is the one I already have half of, otherwise send it all".
            # One header instead of a HEAD request plus a race.
            if resume is not None and resume.validator:
                headers["If-Range"] = resume.validator

        with self._client.stream("GET", file.url, headers=headers) as response:
            if response.status_code >= _HTTP_ERROR:
                response.read()
                raise DownloadError(
                    f"{file.url} returned HTTP {response.status_code}",
                    user_message=(
                        f"Сервер отклонил загрузку «{file.target}» "
                        f"(код {response.status_code})."
                    ),
                )
            lowered = {key.lower(): value for key, value in response.headers.items()}
            if sink.written and response.status_code == _HTTP_OK:
                # The file changed, or the server does not do ranges. Either way
                # it is sending the whole thing, and appending it to what we have
                # would splice two files together.
                _log.info("докачка %s невозможна, качаем заново", file.target)
                sink.reset()
                reporter.restart(0)
            elif sink.written and response.status_code != _HTTP_PARTIAL:
                raise DownloadError(
                    f"{file.url} answered {response.status_code} to a range request",
                    user_message=f"Сервер не поддержал докачку файла «{file.target}».",
                )
            _write_meta(meta, file, lowered)
            sink.absorb(response.iter_bytes(CHUNK_BYTES), handle, reporter)

    def _lock_for(self, model_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(model_id, threading.Lock())

    def _publish_failed(
        self,
        model_id: str,
        error: ModelError,
        *,
        cancelled: bool,
        resumable: bool,
    ) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            ModelDownloadFailed(
                model_id=model_id,
                error=error.technical,
                user_message=error.user_message,
                cancelled=cancelled,
                resumable=resumable,
            )
        )


class _Reporter:
    """Throttles progress into a callback and onto the bus.

    Both outputs are rate-limited by wall clock, so the number of updates depends
    on how long the download takes rather than on how fast the link is.
    """

    __slots__ = (
        "_bus",
        "_callback",
        "_last_at",
        "_mark_at",
        "_mark_bytes",
        "_model_id",
        "_offset",
        "_speed",
        "_total",
    )

    def __init__(
        self,
        *,
        model_id: str,
        bus: EventBus | None,
        callback: ProgressCallback | None,
        offset: int,
        total: int,
    ) -> None:
        self._model_id = model_id
        self._bus = bus
        self._callback = callback
        self._offset = offset
        self._total = total
        self._last_at = 0.0
        self._mark_at = time.monotonic()
        self._mark_bytes = 0
        self._speed = 0.0

    def restart(self, written: int) -> None:
        """The transfer moved backwards; forget the sampling window."""
        self._mark_at = time.monotonic()
        self._mark_bytes = written
        self._speed = 0.0

    def update(self, written: int) -> None:
        now = time.monotonic()
        if now - self._last_at < PROGRESS_INTERVAL_SEC:
            return
        self._emit(written, now)

    def flush(self, written: int) -> None:
        """Report unconditionally — used when a stream ends."""
        self._emit(written, time.monotonic())

    def _emit(self, written: int, now: float) -> None:
        self._last_at = now
        elapsed = now - self._mark_at
        if elapsed >= _SPEED_WINDOW_SEC or (self._speed == 0.0 and elapsed > 0):
            self._speed = max(0.0, (written - self._mark_bytes) / elapsed)
            self._mark_at = now
            self._mark_bytes = written

        done = self._offset + written
        remaining = max(0, self._total - done)
        eta = remaining / self._speed if self._speed > 0 and self._total else 0.0

        if self._callback is not None:
            self._callback(self._model_id, done, self._total, self._speed, eta)
        if self._bus is not None:
            self._bus.publish(
                ModelDownloadProgress(
                    model_id=self._model_id,
                    downloaded=done,
                    total=self._total,
                    speed_bps=self._speed,
                    eta_s=eta,
                )
            )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def sha256_file(path: Path, *, chunk: int = CHUNK_BYTES) -> str:
    """Hash a file without reading it into memory.

    Shared by the downloader and by the registry's integrity check, which is why
    it lives here instead of inside either.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def human_size(size: float) -> str:
    """``1536`` → ``1,5 КБ``. Russian decimal comma, because this reaches the user."""
    units = ("Б", "КБ", "МБ", "ГБ", "ТБ")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            text = f"{value:.0f}" if unit == "Б" else f"{value:.1f}"
            return f"{text.replace('.', ',')} {unit}"
        value /= 1024
    raise AssertionError  # pragma: no cover - the loop always returns


def _build_client(transport: httpx.BaseTransport | None) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT_SEC,
            read=READ_TIMEOUT_SEC,
            write=READ_TIMEOUT_SEC,
            pool=CONNECT_TIMEOUT_SEC,
        ),
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
        transport=transport,
    )


def _meta_path(part: Path) -> Path:
    return part.parent / f"{part.name.removesuffix(PART_SUFFIX)}{META_SUFFIX}"


def _plan_resume(part: Path, meta: Path, file: ModelFile) -> _Resume | None:
    """Whether the existing ``.part`` may be continued, and with what validator.

    ``None`` means start over. The decision is deliberately conservative: the
    server gets the final say through ``If-Range``, but there is no point asking
    when the sidecar is missing, points at another URL, or describes a file
    already longer than the one we want.
    """
    if not part.is_file():
        return None
    size = part.stat().st_size
    if size == 0:
        return None
    stored = _read_meta(meta)
    if stored is None or stored.url != file.url:
        return None
    if file.size_bytes and size >= file.size_bytes:
        # Longer than the file is supposed to be: something other than this
        # downloader wrote it, and there is nothing to salvage.
        return None
    if stored.total and file.size_bytes and stored.total != file.size_bytes:
        return None
    return stored


def _read_meta(path: Path) -> _Resume | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    url = payload.get("url")
    if not isinstance(url, str):
        return None
    total = payload.get("total")
    return _Resume(
        url=url,
        total=total if isinstance(total, int) else 0,
        etag=str(payload.get("etag", "")),
        last_modified=str(payload.get("last_modified", "")),
    )


def _write_meta(path: Path, file: ModelFile, headers: Mapping[str, str]) -> None:
    payload = {
        "url": file.url,
        "total": file.size_bytes or _content_length(headers),
        "etag": headers.get("etag", ""),
        "last_modified": headers.get("last-modified", ""),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - resume metadata is best-effort
        _log.debug("не удалось записать метаданные докачки %s: %s", path, exc)


def _content_length(headers: Mapping[str, str]) -> int:
    try:
        return int(headers.get("content-length", "0"))
    except ValueError:
        return 0


def _free_bytes(path: Path) -> int | None:
    """Free space on the volume holding ``path``, walking up to a directory that exists."""
    probe = path
    for _ in range(_PARENT_PROBES):
        if probe.exists():
            try:
                return shutil.disk_usage(probe).free
            except OSError:  # pragma: no cover - unreadable mount point
                return None
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    return None  # pragma: no cover - that many missing parents cannot happen


#: How far up the tree :func:`_free_bytes` looks for something that exists.
_PARENT_PROBES: Final = 16
