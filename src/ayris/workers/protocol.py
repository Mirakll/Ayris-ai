"""Wire vocabulary of the worker channel: messages, framing, shared audio, errors.

Both ends of a worker pipe import this module and nothing else of each other, so
it stays deliberately light: no Qt, no manager, no engine. A worker process
imports :mod:`ayris.workers.protocol` and :mod:`ayris.workers.base`; the
supervisor adds :mod:`ayris.workers.manager` on top.

**Framing.** Every message travels as ``b"AYW" + version + pickle`` over
``Connection.send_bytes``, rather than through ``Connection.send``. The four
header bytes buy two things worth the trouble: a version check that turns a
mismatched pair of processes into one readable error instead of an
``AttributeError`` halfway through a run, and a single choke point where the
payload is produced and consumed, which is what makes the encoding testable
without a live process.

**Pickle is safe here and only here.** The channel is an anonymous OS pipe
created by the parent and handed to a child it spawned itself; nothing else can
open it, so the usual objection to unpickling untrusted bytes does not apply.
Nothing from the network, a plugin manifest or a config file may ever be fed to
:func:`decode`.

**Audio does not travel through the pipe.** Ten seconds of 16 kHz mono PCM is
320 KB, and pickling that per utterance would copy it three times. Buffers go
into :class:`multiprocessing.shared_memory.SharedMemory` and only the
:class:`AudioChunk` descriptor is sent — see :class:`SharedAudioBlock` on the
writing side and :func:`open_audio` on the reading side.
"""

from __future__ import annotations

import pickle
import sys
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from multiprocessing.shared_memory import SharedMemory
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Final, Self

from ayris.core import errors as error_module
from ayris.core.errors import AyrisError
from ayris.core.models import JsonObject

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from multiprocessing.connection import Connection

__all__ = [
    "AUDIO_PARAM",
    "MAGIC",
    "MAX_INLINE_BYTES",
    "MESSAGE_TYPES",
    "PROTOCOL_VERSION",
    "SAMPLE_WIDTHS",
    "AudioChunk",
    "Channel",
    "Control",
    "ControlKind",
    "ErrorInfo",
    "Heartbeat",
    "Message",
    "Ready",
    "Request",
    "Response",
    "SharedAudioBlock",
    "Stopped",
    "WorkerCallError",
    "WorkerCancelledError",
    "WorkerCrashError",
    "WorkerError",
    "WorkerEvent",
    "WorkerProtocolError",
    "WorkerStartError",
    "WorkerTimeoutError",
    "WorkerUnavailableError",
    "decode",
    "encode",
    "iter_frames",
    "open_audio",
]

#: Bumped whenever a message class changes shape. A mismatch is refused rather
#: than guessed at, which only happens when a stale worker survives an update.
PROTOCOL_VERSION: Final = 1

#: Frame prefix. Short on purpose: it is paid per message.
MAGIC: Final = b"AYW"

_HEADER: Final = MAGIC + bytes((PROTOCOL_VERSION,))
_HEADER_SIZE: Final = len(_HEADER)

#: Above this, a payload belongs in shared memory instead of in a message.
#: Roughly 8 seconds of 16 kHz mono PCM — under it, copying is cheaper than
#: allocating and mapping a block.
MAX_INLINE_BYTES: Final = 256 * 1024

#: Reserved key under which a request carries its :class:`AudioChunk`.
AUDIO_PARAM: Final = "audio"

#: Bytes per sample of the PCM formats the audio path uses.
SAMPLE_WIDTHS: Final[Mapping[str, int]] = {
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "float32": 4,
}


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------


class WorkerError(AyrisError):
    """Base class for every failure of the worker infrastructure itself."""

    default_user_message = "Сбой фонового процесса Ayris."


class WorkerProtocolError(WorkerError):
    """A frame was truncated, mislabelled or produced by another version."""

    default_user_message = "Фоновый процесс Ayris говорит на другом языке. Перезапустите Ayris."


class WorkerStartError(WorkerError):
    """A worker could not be spawned, or its entrypoint does not resolve."""

    default_user_message = "Не удалось запустить фоновый процесс Ayris."


class WorkerUnavailableError(WorkerError):
    """The worker is not running and the caller asked not to start it.

    Also raised for a worker whose entrypoint has not been implemented yet, which
    is how the registry reports a subsystem that a later task will fill in.
    """

    default_user_message = "Подсистема Ayris сейчас недоступна."


class WorkerTimeoutError(WorkerError):
    """A call did not answer within its timeout. The worker may still be alive."""

    default_user_message = "Фоновый процесс Ayris не ответил вовремя."


class WorkerCrashError(WorkerError):
    """The worker died — or was killed for wedging — while a call was pending."""

    default_user_message = "Фоновый процесс Ayris аварийно завершился и будет перезапущен."


class WorkerCallError(WorkerError):
    """A handler raised something that is not an :class:`AyrisError`.

    The remote traceback is kept in :attr:`remote_traceback` for the log; the
    exception type name survives in :attr:`remote_type` even though the class
    itself cannot be reconstructed.
    """

    default_user_message = "Не удалось выполнить операцию в фоновом процессе."

    def __init__(
        self,
        technical: str,
        *,
        remote_type: str = "",
        remote_traceback: str = "",
        user_message: str | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(technical, user_message=user_message, recoverable=recoverable)
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback


class WorkerCancelledError(WorkerError):
    """Raised inside a handler that polled :meth:`WorkerContext.check_cancelled`.

    Travels back as an ordinary failed response, so the supervisor can tell a
    cancelled call apart from a broken one.
    """

    default_user_message = "Операция отменена."


#: Every :class:`~ayris.core.errors.AyrisError` subclass, by class name, so a
#: typed failure raised in a worker is re-raised as the same class in the parent.
#: Built by scanning the module rather than by listing the classes, because a
#: list here would silently go stale every time a new error class is added.
_ERROR_CLASSES: Final[Mapping[str, type[AyrisError]]] = {
    name: value
    for name, value in vars(error_module).items()
    if isinstance(value, type) and issubclass(value, AyrisError)
}


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A failure, flattened enough to survive the pipe.

    An exception instance is not sent as-is: a handler may raise something
    carrying an open file, a device handle or a model, none of which pickles.
    """

    error_type: str
    message: str
    user_message: str
    traceback: str = ""
    recoverable: bool = True

    @classmethod
    def from_exception(cls, exc: BaseException, *, with_traceback: bool = True) -> Self:
        """Flatten ``exc``, keeping its Russian message when it has one."""
        if isinstance(exc, AyrisError):
            user_message = exc.user_message
            recoverable = exc.recoverable
        else:
            user_message = AyrisError.default_user_message
            recoverable = True
        return cls(
            error_type=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
            user_message=user_message,
            traceback="".join(format_exception(exc)) if with_traceback else "",
            recoverable=recoverable,
        )

    def to_exception(self) -> AyrisError:
        """Rebuild the exception for the caller of the failed call.

        A known :class:`~ayris.core.errors.AyrisError` subclass is reconstructed
        exactly, so ``except SttError`` in the pipeline works across the process
        boundary. Anything else becomes a :class:`WorkerCallError` that keeps the
        original type name and traceback.
        """
        known = _ERROR_CLASSES.get(self.error_type)
        if known is not None:
            return known(
                self.message,
                user_message=self.user_message,
                recoverable=self.recoverable,
            )
        return WorkerCallError(
            self.message,
            remote_type=self.error_type,
            remote_traceback=self.traceback,
            user_message=self.user_message,
            recoverable=self.recoverable,
        )


# ----------------------------------------------------------------------
# messages
# ----------------------------------------------------------------------


class ControlKind(StrEnum):
    """Out-of-band instruction, handled ahead of any queued request."""

    STOP = "stop"
    PING = "ping"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class Request:
    """Parent to worker: run ``method`` with ``params``.

    ``params`` is a plain dict so a handler can validate it itself; the audio
    descriptor, when there is one, sits under :data:`AUDIO_PARAM`.
    """

    id: int
    method: str
    params: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Response:
    """Worker to parent: the outcome of exactly one :class:`Request`."""

    id: int
    ok: bool
    payload: Any = None
    error: ErrorInfo | None = None
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    """Worker to parent, unsolicited: audio levels, partial text, log records.

    ``kind`` is namespaced by the worker that sends it (``level``, ``partial``,
    ``log``); the supervisor turns known kinds into bus events through the
    translator on the worker's spec.
    """

    kind: str
    payload: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """Worker to parent: still alive, and what it is doing.

    ``at`` is :func:`time.time`, for the log; liveness is judged by arrival time
    in the parent, so the two clocks never have to agree.
    """

    seq: int
    at: float
    busy: bool = False
    handled: int = 0


@dataclass(frozen=True, slots=True)
class Control:
    """Parent to worker: stop, answer now, or abandon a request."""

    kind: ControlKind
    request_id: int | None = None


@dataclass(frozen=True, slots=True)
class Ready:
    """Worker to parent: ``on_start`` finished, requests may be sent."""

    name: str
    pid: int
    protocol_version: int = PROTOCOL_VERSION
    methods: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Stopped:
    """Worker to parent: leaving cleanly, this is the last frame."""

    reason: str = ""


#: Anything that may cross the channel.
Message = Request | Response | WorkerEvent | Heartbeat | Control | Ready | Stopped

#: Runtime counterpart of :data:`Message`, for the check in :func:`decode`.
MESSAGE_TYPES: Final = (Request, Response, WorkerEvent, Heartbeat, Control, Ready, Stopped)


def encode(message: Message) -> bytes:
    """Frame ``message`` for :meth:`Channel.send`."""
    return _HEADER + pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def decode(raw: bytes) -> Message:
    """Parse a frame produced by :func:`encode`.

    Raises:
        WorkerProtocolError: The frame is truncated, not ours, written by another
            protocol version, or does not carry a known message type.
    """
    if len(raw) < _HEADER_SIZE or not raw.startswith(MAGIC):
        raise WorkerProtocolError(f"not a worker frame: {raw[:8]!r}")
    version = raw[len(MAGIC)]
    if version != PROTOCOL_VERSION:
        raise WorkerProtocolError(
            f"worker protocol version {version}, expected {PROTOCOL_VERSION}"
        )
    try:
        message = pickle.loads(raw[_HEADER_SIZE:])
    except Exception as exc:
        raise WorkerProtocolError(f"cannot unpickle worker frame: {exc}") from exc
    if not isinstance(message, MESSAGE_TYPES):
        raise WorkerProtocolError(f"unexpected worker payload: {type(message).__name__}")
    return message


class Channel:
    """Framed message channel over one end of a duplex pipe.

    Sends are serialised: a worker writes from its dispatch thread, its heartbeat
    thread and its log handler, and two interleaved ``send_bytes`` calls would
    corrupt the stream. Receives are not locked — exactly one thread per end
    reads.
    """

    __slots__ = ("_closed", "_connection", "_send_lock")

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._send_lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, message: Message) -> None:
        """Write one message.

        Raises:
            OSError: The pipe is closed, which is what a dead peer looks like.
        """
        raw = encode(message)
        with self._send_lock:
            if self._closed:
                raise OSError("worker channel is closed")
            self._connection.send_bytes(raw)

    def recv(self) -> Message:
        """Block until one message arrives.

        Raises:
            EOFError: The peer closed its end — the normal end of a worker's life.
            OSError: This end is closed.
            WorkerProtocolError: The frame did not parse.
        """
        return decode(self._connection.recv_bytes())

    def poll(self, timeout: float | None = 0.0) -> bool:
        """Whether a message is waiting."""
        return self._connection.poll(timeout)

    def close(self) -> None:
        """Close this end. Idempotent; a peer blocked in ``recv`` gets EOF."""
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
        self._connection.close()

    def __repr__(self) -> str:
        return f"Channel(closed={self._closed})"


# ----------------------------------------------------------------------
# audio through shared memory
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Descriptor of a PCM buffer living in shared memory.

    Cheap to pickle — a name and five numbers — which is the whole point: the
    samples themselves never enter the pipe.
    """

    block: str
    nbytes: int
    sample_rate: int = 16000
    channels: int = 1
    sample_format: str = "int16"

    @property
    def sample_width(self) -> int:
        """Bytes per sample, defaulting to 16-bit for an unknown format."""
        return SAMPLE_WIDTHS.get(self.sample_format, 2)

    @property
    def frames(self) -> int:
        """Sample frames in the buffer, counting all channels as one frame."""
        stride = self.sample_width * max(self.channels, 1)
        return self.nbytes // stride if stride else 0

    @property
    def duration_ms(self) -> float:
        """Length of the buffer in milliseconds."""
        return self.frames * 1000.0 / self.sample_rate if self.sample_rate else 0.0

    def to_params(self, params: JsonObject | None = None) -> JsonObject:
        """Copy ``params`` with this descriptor added under :data:`AUDIO_PARAM`."""
        merged: JsonObject = dict(params) if params else {}
        merged[AUDIO_PARAM] = self
        return merged

    @classmethod
    def from_params(cls, params: JsonObject) -> Self | None:
        """Pull the descriptor out of a request's params, or ``None``."""
        candidate = params.get(AUDIO_PARAM)
        return candidate if isinstance(candidate, cls) else None


class SharedAudioBlock:
    """A PCM buffer in shared memory, owned by the process that created it.

    The owner unlinks the block in :meth:`close`; readers only ever attach, so
    the block outlives no more than one call. Use it as a context manager, or let
    the supervisor tie its lifetime to the call it belongs to.
    """

    __slots__ = ("_chunk", "_memory", "_released")

    def __init__(self, memory: SharedMemory, chunk: AudioChunk) -> None:
        self._memory = memory
        self._chunk = chunk
        self._released = False

    @classmethod
    def create(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_format: str = "int16",
    ) -> Self:
        """Allocate a block and copy ``data`` into it.

        Raises:
            WorkerError: The operating system refused the allocation.
        """
        payload = memoryview(data).cast("B")
        size = payload.nbytes
        if size <= 0:
            raise WorkerError("refusing to share an empty audio buffer")
        try:
            memory = SharedMemory(create=True, size=size)
        except OSError as exc:
            raise WorkerError(f"cannot allocate {size} bytes of shared memory: {exc}") from exc
        memory.buf[:size] = payload
        chunk = AudioChunk(
            block=memory.name,
            nbytes=size,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
        )
        return cls(memory, chunk)

    @property
    def chunk(self) -> AudioChunk:
        """The descriptor to put into a request."""
        return self._chunk

    @property
    def buffer(self) -> memoryview:
        """The bytes, writable. Invalid after :meth:`close`."""
        return self._memory.buf[: self._chunk.nbytes]

    def close(self) -> None:
        """Release and unlink the block. Idempotent."""
        if self._released:
            return
        self._released = True
        try:
            self._memory.close()
            self._memory.unlink()
        except (OSError, BufferError):
            # Already gone, or a reader still holds a view. Either way there is
            # nothing useful left to do and the caller is on a teardown path.
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SharedAudioBlock({self._chunk.block}, {self._chunk.nbytes} bytes)"


def _untrack(name: str) -> None:
    """Stop the resource tracker from claiming a block this process only reads.

    A process that attaches to shared memory registers it with the multiprocessing
    resource tracker, which then reports it as leaked when that process exits —
    even though the owner unlinks it correctly (CPython gh-82300). Unregistering
    keeps a worker's exit quiet without weakening the owner's cleanup.
    """
    if sys.platform != "win32":
        from multiprocessing import resource_tracker

        try:
            resource_tracker.unregister(f"/{name}", "shared_memory")
        except (KeyError, OSError):
            pass


def open_audio(chunk: AudioChunk) -> _AudioReader:
    """Attach to ``chunk`` for reading.

    The view is valid only inside the ``with`` block — copy out whatever has to
    outlive it::

        with open_audio(chunk) as pcm:
            samples = numpy.frombuffer(pcm, dtype="int16")

    Raises:
        WorkerError: The block is gone, usually because the call timed out and
            the supervisor already released it.
    """
    return _AudioReader(chunk)


class _AudioReader:
    """Context manager returned by :func:`open_audio`."""

    __slots__ = ("_chunk", "_memory", "_view")

    def __init__(self, chunk: AudioChunk) -> None:
        self._chunk = chunk
        self._memory: SharedMemory | None = None
        self._view: memoryview | None = None

    def __enter__(self) -> memoryview:
        try:
            memory = SharedMemory(name=self._chunk.block)
        except (FileNotFoundError, OSError) as exc:
            raise WorkerError(f"shared audio block {self._chunk.block} is gone: {exc}") from exc
        _untrack(self._chunk.block)
        self._memory = memory
        self._view = memory.buf[: self._chunk.nbytes]
        return self._view

    def __exit__(self, *_exc: object) -> None:
        # Order matters: SharedMemory.close() raises BufferError while an
        # exported slice of its buffer is still alive.
        if self._view is not None:
            self._view.release()
            self._view = None
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    def __repr__(self) -> str:
        return f"_AudioReader({self._chunk.block})"


def iter_frames(pcm: memoryview, chunk: AudioChunk, frames: int) -> Iterator[memoryview]:
    """Slice ``pcm`` into groups of ``frames``, for engines that want fixed sizes.

    The last slice is short when the buffer does not divide evenly; a VAD or a
    streaming recogniser is expected to pad or drop it itself.
    """
    stride = chunk.sample_width * max(chunk.channels, 1) * max(frames, 1)
    for start in range(0, pcm.nbytes, stride):
        yield pcm[start : start + stride]
