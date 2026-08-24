"""Task 14: the model manager, on mocked HTTP and a temporary disk.

Only one test here opens a socket, and it is marked ``network`` for it:
:meth:`TestShippedCatalog.test_every_url_is_still_there`, which asks whether the
twenty-two addresses in the shipped catalog still answer. Everything else is
offline. :class:`httpx.MockTransport` goes in through
:class:`~ayris.models.downloader.Downloader`'s ``transport=`` seam, so the
requests the downloader *would* have sent are inspected as objects — which is the
only way to assert the thing that actually matters about resume: that the second
attempt asked for ``bytes=N-`` and received exactly the missing tail, rather than
quietly fetching the file twice and looking fine.

The catalog entries are built in-process rather than read from
``resources/models``: a test that downloads a real 45 MB Vosk archive is not a
unit test, and one that hardcodes the real SHA-256 would fail whenever upstream
re-publishes a file. What the shipped catalog *is* checked for is that it parses
and that every entry is well-formed — see :class:`TestShippedCatalog`.

Groups:

* :class:`TestCatalog` — parsing, validation and the error a broken file gives.
* :class:`TestShippedCatalog` — ``resources/models/*.json`` loads, and its
  addresses are still live (that one test is marked ``network``).
* :class:`TestDownload` — the happy path, progress and the staged result.
* :class:`TestResume` — interruption, ``Range``, ``If-Range`` and a changed file.
* :class:`TestIntegrity` — a mismatched digest deletes the file and raises.
* :class:`TestCancel` — cancellation keeps the partial file.
* :class:`TestSpace` — the free-space check happens before the first byte.
* :class:`TestArchives` — unpacking, single-root stripping, traversal attempts.
* :class:`TestSubdirectory` — ``directory``: loose files staged into their own folder.
* :class:`TestRegistry` — registration, activation events, deletion, accounting.
* :class:`TestIntegrityChecks` — OK, повреждена, and rows whose files are gone.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path as PathType
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from ayris.core.config import SttConfig
from ayris.core.database import Database
from ayris.core.events import (
    ActiveModelChanged,
    Event,
    EventBus,
    ModelDownloadFailed,
    ModelDownloadProgress,
    ModelRemoved,
)
from ayris.core.models import ModelRecord
from ayris.core.repositories import Repositories
from ayris.models.catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogError,
    ModelCatalog,
    ModelEntry,
    catalog_dir,
    load_catalog,
    load_catalog_file,
)
from ayris.models.downloader import (
    CHUNK_BYTES,
    PART_SUFFIX,
    DownloadCancelled,
    Downloader,
    DownloadError,
    DownloadHandle,
    IntegrityError,
    NotEnoughSpaceError,
    human_size,
    sha256_file,
)
from ayris.models.installer import (
    MANIFEST_NAME,
    ArchiveError,
    Installer,
    InstallError,
    read_manifest,
)
from ayris.models.registry import (
    IntegrityStatus,
    ModelInUseError,
    ModelRegistry,
    NotInstalledError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from ayris.core.paths import AppPaths

#: Body of the model file every download test fetches. Big enough to span several
#: 64 KiB chunks, so resume has a real boundary to land on rather than a single
#: read that either works or does not.
PAYLOAD = bytes(range(256)) * 1024  # 256 KiB
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()

#: Companion of a directory model — the vocabulary beside the weights. Tiny on
#: purpose: what it is here for is the file *name*, not its size.
VOCAB = "а б в\nг д е\n".encode()
VOCAB_SHA = hashlib.sha256(VOCAB).hexdigest()

URL = "https://models.test/ayris/model.bin"

#: Registries built by :func:`make_registry`, drained by ``_close_registries``.
_OPEN_REGISTRIES: list[ModelRegistry] = []


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def entry(**overrides: Any) -> ModelEntry:
    """A catalog entry for :data:`PAYLOAD`, with fields overridable per test."""
    fields: dict[str, Any] = {
        "id": "test-model",
        "name": "Тестовая модель",
        "kind": "stt",
        "engine": "vosk",
        "url": URL,
        "sha256": PAYLOAD_SHA,
        "size_bytes": len(PAYLOAD),
    }
    fields.update(overrides)
    return ModelEntry.model_validate(fields)


def zip_bytes(members: dict[str, bytes]) -> bytes:
    """A zip archive built in memory, so no fixture file has to be checked in."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, body in members.items():
            bundle.writestr(name, body)
    return buffer.getvalue()


def tar_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


class Server:
    """A recording HTTP stand-in that honours ``Range`` and ``If-Range``.

    Deliberately not a bare lambda: every resume assertion in this module is
    really an assertion about the request the downloader built, so the requests
    have to be kept. ``cut_after`` reproduces the case the whole resume path
    exists for — a connection that dies mid-body — by raising once the response
    has already started.
    """

    def __init__(
        self,
        body: bytes = PAYLOAD,
        *,
        etag: str = '"v1"',
        supports_range: bool = True,
        cut_after: int | None = None,
    ) -> None:
        self.body = body
        self.etag = etag
        self.supports_range = supports_range
        self.cut_after = cut_after
        self.requests: list[httpx.Request] = []
        self.status_codes: list[int] = []
        self.served: list[int] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @property
    def ranges(self) -> list[str]:
        """The ``Range`` header of each request, ``""`` when there was none."""
        return [request.headers.get("range", "") for request in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        start = 0
        raw_range = request.headers.get("range", "")
        wants_range = bool(raw_range) and self.supports_range
        if wants_range and self._validator_matches(request):
            start = int(raw_range.removeprefix("bytes=").split("-")[0])
        elif raw_range and not self.supports_range:
            wants_range = False

        chunk = self.body[start:]
        status = 206 if wants_range and start else 200
        self.status_codes.append(status)
        headers = {
            "Content-Length": str(len(chunk)),
            "ETag": self.etag,
            "Accept-Ranges": "bytes",
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{len(self.body) - 1}/{len(self.body)}"

        if self.cut_after is not None:
            served = chunk[: self.cut_after]
            self.cut_after = None
            self.served.append(len(served))
            return httpx.Response(
                status,
                headers=headers,
                content=_truncated(served),
            )
        self.served.append(len(chunk))
        return httpx.Response(status, headers=headers, content=chunk)

    def _validator_matches(self, request: httpx.Request) -> bool:
        """``If-Range`` decides whether the tail may be continued."""
        offered = request.headers.get("if-range", "")
        return not offered or offered == self.etag


def _truncated(served: bytes) -> Iterator[bytes]:
    """Yield a partial body and then drop the connection."""

    def stream() -> Iterator[bytes]:
        yield served
        raise httpx.ReadError("connection reset")

    return stream()


class Recorder:
    """Collects published events, keeping them in order."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        self._token = bus.subscribe(Event, self.events.append, weak=False)

    def of(self, event_type: type[Event]) -> list[Any]:
        return [item for item in self.events if isinstance(item, event_type)]


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the retry backoff: three retries otherwise sleep seven seconds."""
    monkeypatch.setattr("ayris.models.downloader.BACKOFF_BASE_SEC", 0.0)
    monkeypatch.setattr("ayris.models.downloader.BACKOFF_MAX_SEC", 0.0)


@pytest.fixture(autouse=True)
def _close_registries() -> Iterator[None]:
    """Close every registry :func:`make_registry` built, even on a failed assert.

    Each registry owns an :class:`httpx.Client`; the explicit ``close()`` at the
    end of a test does not run when an assertion above it fails.
    """
    yield
    while _OPEN_REGISTRIES:
        _OPEN_REGISTRIES.pop().close()


@pytest.fixture
def bus() -> EventBus:
    """A bus that delivers inline, so a test never has to drain it."""
    return EventBus(thread_id=None)


@pytest.fixture
def recorder(bus: EventBus) -> Recorder:
    return Recorder(bus)


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path / "work"


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def repos(tmp_path: Path) -> Iterator[Repositories]:
    with Database.open(tmp_path / "ayris.db") as database:
        yield Repositories(database)


def make_registry(
    repos: Repositories,
    paths: AppPaths,
    *,
    catalog: ModelCatalog,
    bus: EventBus | None = None,
    transport: httpx.BaseTransport | None = None,
    is_busy: Callable[[ModelRecord], bool] | None = None,
) -> ModelRegistry:
    """A registry wired to a mock transport and a temporary profile."""
    downloader = Downloader(
        paths.cache_dir / "models" / "downloads",
        bus=bus,
        transport=transport,
    )
    registry = ModelRegistry(
        repos.models,
        paths,
        catalog=catalog,
        bus=bus,
        downloader=downloader,
        installer=Installer(paths.models_dir),
        is_busy=is_busy,
    )
    _OPEN_REGISTRIES.append(registry)
    return registry


# ----------------------------------------------------------------------
# catalog
# ----------------------------------------------------------------------


class TestCatalog:
    def test_entry_derives_its_target_from_the_url(self) -> None:
        assert entry().target == "model.bin"

    def test_archive_entry_derives_its_target_from_the_id(self) -> None:
        assert entry(archive="zip").target == "test-model"

    def test_total_bytes_counts_companions(self) -> None:
        voice = entry(
            extra_files=[
                {"url": "https://models.test/v.json", "sha256": PAYLOAD_SHA, "size_bytes": 100}
            ]
        )
        assert voice.total_bytes == len(PAYLOAD) + 100
        assert [item.target for item in voice.files] == ["model.bin", "v.json"]

    def test_a_short_digest_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            entry(sha256="abc")

    def test_an_uppercase_digest_is_folded(self) -> None:
        assert entry(sha256=PAYLOAD_SHA.upper()).sha256 == PAYLOAD_SHA

    def test_a_relative_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="http"):
            entry(url="models/model.bin")

    def test_a_target_with_a_path_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="путь"):
            entry(target="../escape")

    def test_companions_to_an_archive_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra_files"):
            entry(
                archive="zip",
                extra_files=[
                    {"url": "https://models.test/v.json", "sha256": PAYLOAD_SHA, "size_bytes": 1}
                ],
            )

    def test_a_subdirectory_with_a_path_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="путь"):
            entry(directory="../escape")

    def test_a_subdirectory_on_an_archive_is_rejected(self) -> None:
        """An archive brings its own top-level folder; naming a second one is a
        catalog mistake, not a layout to guess at."""
        with pytest.raises(ValueError, match="directory"):
            entry(archive="zip", directory="somewhere")

    def test_install_name_is_the_subdirectory_when_there_is_one(self) -> None:
        """The name the settings hold: ``offline_model`` is this string verbatim."""
        assert entry().install_name == "model.bin"
        assert entry(directory="gigaam-v3-ctc").install_name == "gigaam-v3-ctc"

    def test_broken_json_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "stt.json"
        path.write_text("{не json", encoding="utf-8")

        with pytest.raises(CatalogError) as excinfo:
            load_catalog_file(path)

        assert "stt.json" in excinfo.value.user_message

    def test_a_bad_entry_is_reported_in_russian(self, tmp_path: Path) -> None:
        path = tmp_path / "stt.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "stt",
                    "models": [{"id": "x", "name": "X", "kind": "stt", "engine": "vosk"}],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(CatalogError) as excinfo:
            load_catalog_file(path)

        assert "stt.json" in excinfo.value.user_message
        assert "url" in excinfo.value.user_message

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        record = {
            "id": "dup",
            "name": "Первая",
            "kind": "stt",
            "engine": "vosk",
            "url": URL,
            "sha256": PAYLOAD_SHA,
            "size_bytes": 1,
        }
        path = tmp_path / "stt.json"
        path.write_text(
            json.dumps({"schema_version": 1, "kind": "stt", "models": [record, record]}),
            encoding="utf-8",
        )

        with pytest.raises(CatalogError) as excinfo:
            load_catalog_file(path)

        assert "dup" in excinfo.value.technical

    def test_a_newer_schema_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "stt.json"
        path.write_text(
            json.dumps({"schema_version": CATALOG_SCHEMA_VERSION + 1, "kind": "stt", "models": []}),
            encoding="utf-8",
        )

        with pytest.raises(CatalogError) as excinfo:
            load_catalog_file(path)

        assert "новее" in excinfo.value.user_message

    def test_a_missing_directory_yields_an_empty_catalog(self, tmp_path: Path) -> None:
        assert len(load_catalog(tmp_path / "nowhere")) == 0

    def test_an_entry_inherits_the_kind_of_its_file(self, tmp_path: Path) -> None:
        """``stt.json`` says ``stt`` once; the entries do not repeat it."""
        path = tmp_path / "stt.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "stt",
                    "models": [
                        {
                            "id": "x",
                            "name": "X",
                            "engine": "vosk",
                            "url": URL,
                            "sha256": PAYLOAD_SHA,
                            "size_bytes": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        assert load_catalog_file(path).models[0].kind == "stt"

    def test_an_entry_filed_under_the_wrong_kind_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "stt.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "stt",
                    "models": [
                        {
                            "id": "voice",
                            "name": "Голос",
                            "kind": "tts",
                            "engine": "piper",
                            "url": URL,
                            "sha256": PAYLOAD_SHA,
                            "size_bytes": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(CatalogError) as excinfo:
            load_catalog_file(path)

        assert "не того вида" in excinfo.value.user_message

    def test_strict_mode_complains_about_a_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogError):
            load_catalog(tmp_path / "nowhere", strict=True)

    def test_lookup_helpers(self) -> None:
        catalog = ModelCatalog((entry(), entry(id="other", engine="whisper")))

        assert catalog.get("test-model") is not None
        assert catalog.get("нет такой") is None
        assert catalog.for_engine("stt", "whisper")[0].id == "other"
        assert catalog.engines("stt") == ("vosk", "whisper")
        assert "test-model" in catalog

    def test_require_names_the_missing_model(self) -> None:
        with pytest.raises(CatalogError) as excinfo:
            ModelCatalog().require("пропавшая")

        assert "пропавшая" in excinfo.value.user_message


class TestShippedCatalog:
    """``resources/models/*.json`` must parse — it ships with the application."""

    #: One test in here opens a socket and the rest do not, so the marker sits on
    #: the method rather than the class.

    def test_every_shipped_file_loads(self) -> None:
        directory = catalog_dir()
        if not directory.is_dir():  # pragma: no cover - only in a stripped checkout
            pytest.skip("resources/models отсутствует")

        catalog = load_catalog(directory, strict=True)

        assert len(catalog) > 0
        assert len(set(catalog.ids)) == len(catalog)

    @pytest.mark.network
    def test_every_url_is_still_there(self) -> None:
        """Every address in the catalog answers, with the size the catalog claims.

        The one test here that opens a socket, and the reason it is worth the
        exception: these are twenty-two links to somebody else's files. Upstream
        renames a release asset, Hugging Face moves a voice, and nothing in the
        repository changes — the failure lands on a user's first run, in the one
        moment when the application has nothing to fall back on. A ``HEAD`` per
        entry costs a couple of seconds and moves that discovery here.

        Sizes are compared but digests are not: verifying sha256 means
        downloading two gigabytes, which belongs to the job that installs the
        weights, not to a liveness check.
        """
        directory = catalog_dir()
        if not directory.is_dir():  # pragma: no cover - only in a stripped checkout
            pytest.skip("resources/models отсутствует")

        broken: list[str] = []
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            for entry in load_catalog(directory):
                for file in entry.files:
                    try:
                        response = client.head(file.url)
                    except httpx.HTTPError as exc:
                        broken.append(f"{entry.id}/{file.target}: {type(exc).__name__} {exc}")
                        continue
                    if response.status_code != httpx.codes.OK:
                        broken.append(f"{entry.id}/{file.target}: HTTP {response.status_code}")
                        continue
                    # Не у всякого сервера есть Content-Length на HEAD; если он
                    # есть, он обязан совпасть - иначе файл на том же адресе
                    # заменили другим, и sha256 не сойдётся уже у пользователя.
                    length = response.headers.get("content-length")
                    if length is not None and int(length) != file.size_bytes:
                        broken.append(
                            f"{entry.id}/{file.target}: {length} байт "
                            f"вместо {file.size_bytes} из каталога"
                        )

        assert not broken, "адреса в каталоге больше не отдают то, что заявлено:\n" + "\n".join(
            broken
        )

    def test_the_default_stt_model_is_offered(self) -> None:
        """The settings default has to be installable, or a fresh profile is stuck.

        Compared against ``install_name`` and not ``target``: the default is a
        GigaAM export, which is several files in a folder of their own, and the
        name the setting holds is that folder's.
        """
        directory = catalog_dir()
        if not directory.is_dir():  # pragma: no cover - only in a stripped checkout
            pytest.skip("resources/models отсутствует")

        names = {item.install_name for item in load_catalog(directory).for_kind("stt")}

        assert SttConfig().offline_model in names


# ----------------------------------------------------------------------
# downloading
# ----------------------------------------------------------------------


class TestDownload:
    def test_a_download_lands_verified_in_the_work_directory(self, work_dir: Path) -> None:
        server = Server()
        model = entry()

        with Downloader(work_dir, transport=server.transport) as downloader:
            result = downloader.fetch(model.id, model.files[0])

        assert result.path.read_bytes() == PAYLOAD
        assert result.sha256 == PAYLOAD_SHA
        assert result.size_bytes == len(PAYLOAD)

    def test_the_part_file_is_gone_afterwards(self, work_dir: Path) -> None:
        server = Server()
        model = entry()

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloader.fetch(model.id, model.files[0])

            assert not downloader.part_path(model.id, model.files[0]).exists()
            assert not list(work_dir.glob(f"*{PART_SUFFIX}.json"))

    def test_progress_reaches_the_callback_and_the_bus(
        self,
        work_dir: Path,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        server = Server()
        model = entry()
        seen: list[tuple[str, int, int]] = []

        with Downloader(work_dir, bus=bus, transport=server.transport) as downloader:
            downloader.fetch(
                model.id,
                model.files[0],
                on_progress=lambda mid, done, total, _speed, _eta: seen.append((mid, done, total)),
            )

        assert seen, "прогресс не пришёл ни разу"
        assert seen[-1] == (model.id, len(PAYLOAD), len(PAYLOAD))
        progress = recorder.of(ModelDownloadProgress)
        assert progress and progress[-1].fraction == pytest.approx(1.0)

    def test_all_files_of_a_multi_file_model_are_fetched(self, work_dir: Path) -> None:
        config = b'{"voice": "irina"}'
        server = Server()

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".json"):
                return httpx.Response(200, content=config)
            return server.handle(request)

        voice = entry(
            extra_files=[
                {
                    "url": "https://models.test/ayris/voice.onnx.json",
                    "sha256": hashlib.sha256(config).hexdigest(),
                    "size_bytes": len(config),
                }
            ]
        )

        with Downloader(work_dir, transport=httpx.MockTransport(handle)) as downloader:
            results = downloader.fetch_all(voice)

        assert [item.target for item in results] == ["model.bin", "voice.onnx.json"]
        assert results[1].path.read_bytes() == config

    def test_an_http_error_is_reported_with_the_status(self, work_dir: Path) -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(404))
        model = entry()

        with (
            Downloader(work_dir, transport=transport) as downloader,
            pytest.raises(DownloadError, match="404"),
        ):
            downloader.fetch(model.id, model.files[0])

    def test_a_failure_is_published(
        self,
        work_dir: Path,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(500))
        model = entry()

        with (
            Downloader(work_dir, bus=bus, transport=transport) as downloader,
            pytest.raises(DownloadError),
        ):
            downloader.fetch(model.id, model.files[0])

        failures = recorder.of(ModelDownloadFailed)
        assert len(failures) == 1
        assert failures[-1].cancelled is False

    def test_two_downloads_of_one_model_do_not_overlap(self, work_dir: Path) -> None:
        """The per-model lock, exercised by re-entering ``fetch`` from the handler.

        No threads: the lock is not re-entrant, so ``acquire(blocking=False)``
        from the same thread already refuses, and the nested call gives up before
        it reaches HTTP — which is why this cannot recurse forever.
        """
        model = entry()
        server = Server()
        refused: list[str] = []
        holder: list[Downloader] = []

        def reenter(request: httpx.Request) -> httpx.Response:
            try:
                holder[0].fetch(model.id, model.files[0])
            except DownloadError as exc:
                refused.append(exc.user_message)
            return server.handle(request)

        with Downloader(work_dir, transport=httpx.MockTransport(reenter)) as downloader:
            holder.append(downloader)
            result = downloader.fetch(model.id, model.files[0])

        assert refused == ["Эта модель уже загружается."]
        assert result.sha256 == PAYLOAD_SHA
        assert len(server.requests) == 1, "вложенная попытка не должна доходить до сети"


class TestResume:
    def test_an_interrupted_download_asks_for_the_missing_tail(self, work_dir: Path) -> None:
        # One whole chunk lands before the connection dies: httpx hands the sink
        # only complete 64 KiB pieces, so the resume offset is exactly CHUNK_BYTES
        # rather than whatever the chunker happened to have buffered.
        server = Server(cut_after=CHUNK_BYTES)
        model = entry()

        with Downloader(work_dir, transport=server.transport) as downloader:
            result = downloader.fetch(model.id, model.files[0])

        assert result.path.read_bytes() == PAYLOAD
        assert result.sha256 == PAYLOAD_SHA
        # The first request had no Range; the retry asked to continue where the
        # bytes on disk stopped — not from zero, and not from a guess.
        assert server.ranges[0] == ""
        assert server.ranges[1] == f"bytes={CHUNK_BYTES}-"
        assert server.status_codes[1] == 206
        # The tail was sent once, so the payload crossed the wire 1.25 times, not twice.
        assert server.served == [CHUNK_BYTES, len(PAYLOAD) - CHUNK_BYTES]

    def test_a_second_run_continues_the_part_file(self, work_dir: Path) -> None:
        """The real resume case: the process died, the ``.part`` file survived."""
        model = entry()
        first = Server(cut_after=CHUNK_BYTES)
        handle = DownloadHandle()

        # Fail the first attempt outright, leaving a partial file behind.
        with Downloader(work_dir, transport=_flaky(first)) as downloader:
            with pytest.raises(DownloadError):
                downloader.fetch(model.id, model.files[0], handle=handle)
            partial = downloader.pending_bytes(model.id, model.files[0])

        assert partial == CHUNK_BYTES

        second = Server()
        with Downloader(work_dir, transport=second.transport) as downloader:
            result = downloader.fetch(model.id, model.files[0])

        assert result.path.read_bytes() == PAYLOAD
        assert second.ranges[0] == f"bytes={partial}-"
        assert second.requests[0].headers.get("if-range") == '"v1"'
        # Exactly the tail: the run took one response and it was short by what
        # was already on disk, so nothing was fetched twice.
        assert second.served == [len(PAYLOAD) - partial]
        assert first.served[0] + second.served[0] == len(PAYLOAD)

    def test_a_changed_file_restarts_the_download(self, work_dir: Path) -> None:
        model = entry()
        with Downloader(work_dir, transport=_flaky(Server(cut_after=CHUNK_BYTES))) as downloader:
            with pytest.raises(DownloadError):
                downloader.fetch(model.id, model.files[0])
            partial = downloader.pending_bytes(model.id, model.files[0])
        assert partial > 0

        # Same URL, different content: the server answers 200 to the range
        # request, which must throw the partial bytes away rather than splice.
        changed = Server(etag='"v2"')
        with Downloader(work_dir, transport=changed.transport) as downloader:
            result = downloader.fetch(model.id, model.files[0])

        assert changed.status_codes[0] == 200
        assert result.path.read_bytes() == PAYLOAD
        assert result.sha256 == PAYLOAD_SHA

    def test_a_server_without_range_support_starts_over(self, work_dir: Path) -> None:
        model = entry()
        with (
            Downloader(work_dir, transport=_flaky(Server(cut_after=CHUNK_BYTES))) as downloader,
            pytest.raises(DownloadError),
        ):
            downloader.fetch(model.id, model.files[0])

        plain = Server(supports_range=False)
        with Downloader(work_dir, transport=plain.transport) as downloader:
            result = downloader.fetch(model.id, model.files[0])

        assert plain.status_codes[0] == 200
        assert plain.served == [len(PAYLOAD)]
        assert result.sha256 == PAYLOAD_SHA

    def test_a_part_file_without_a_sidecar_is_not_resumed(self, work_dir: Path) -> None:
        model = entry()
        work_dir.mkdir(parents=True)
        server = Server()
        with Downloader(work_dir, transport=server.transport) as downloader:
            downloader.part_path(model.id, model.files[0]).write_bytes(b"garbage")

            result = downloader.fetch(model.id, model.files[0])

        assert server.ranges[0] == ""
        assert result.sha256 == PAYLOAD_SHA


class TestIntegrity:
    def test_a_mismatched_digest_raises_and_deletes(self, work_dir: Path) -> None:
        server = Server(body=b"\x00" * len(PAYLOAD))
        model = entry()

        with Downloader(work_dir, transport=server.transport) as downloader:
            with pytest.raises(IntegrityError) as excinfo:
                downloader.fetch(model.id, model.files[0])

            assert not downloader.part_path(model.id, model.files[0]).exists()
            assert not downloader.staged_path(model.id, model.files[0]).exists()

        assert "контрольная сумма" in excinfo.value.user_message

    def test_the_failure_is_not_marked_resumable(
        self,
        work_dir: Path,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        server = Server(body=b"\x00" * len(PAYLOAD))
        model = entry()

        with (
            Downloader(work_dir, bus=bus, transport=server.transport) as downloader,
            pytest.raises(IntegrityError),
        ):
            downloader.fetch(model.id, model.files[0])

        failures = recorder.of(ModelDownloadFailed)
        assert len(failures) == 1, "о сорванной проверке сообщают один раз"
        assert failures[-1].resumable is False

    def test_sha256_file_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "blob"
        path.write_bytes(PAYLOAD)

        assert sha256_file(path) == PAYLOAD_SHA


class TestCancel:
    def test_cancelling_stops_the_transfer_and_keeps_the_part(self, work_dir: Path) -> None:
        """Cancel *during* the body, so the ``.part`` file has something in it.

        Cancelling before the first byte would leave an empty file and prove
        nothing: the point is that what was already transferred survives.
        """
        model = entry()
        handle = DownloadHandle()

        def stream() -> Iterator[bytes]:
            yield PAYLOAD[:CHUNK_BYTES]
            handle.cancel()
            yield PAYLOAD[CHUNK_BYTES:]

        def serve(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(PAYLOAD)), "Accept-Ranges": "bytes"},
                content=stream(),
            )

        with Downloader(work_dir, transport=httpx.MockTransport(serve)) as downloader:
            with pytest.raises(DownloadCancelled):
                downloader.fetch(model.id, model.files[0], handle=handle)

            assert downloader.part_path(model.id, model.files[0]).exists()
            assert downloader.pending_bytes(model.id, model.files[0]) == CHUNK_BYTES

    def test_a_cancelled_download_is_published_as_cancelled(
        self,
        work_dir: Path,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        model = entry()
        handle = DownloadHandle()
        handle.cancel()

        with (
            Downloader(work_dir, bus=bus, transport=Server().transport) as downloader,
            pytest.raises(DownloadCancelled),
        ):
            downloader.fetch(model.id, model.files[0], handle=handle)

        failure = recorder.of(ModelDownloadFailed)[-1]
        assert failure.cancelled is True

    def test_discard_removes_every_trace(self, work_dir: Path) -> None:
        model = entry()
        handle = DownloadHandle()
        handle.cancel()

        with Downloader(work_dir, transport=Server().transport) as downloader:
            with pytest.raises(DownloadCancelled):
                downloader.fetch(model.id, model.files[0], handle=handle)
            downloader.discard(model.id, model)

            assert downloader.pending_bytes(model.id, model.files[0]) == 0


class TestSpace:
    def test_an_oversized_model_is_refused_before_the_request(
        self,
        work_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        server = Server()
        huge = entry(size_bytes=10**12)
        monkeypatch.setattr("ayris.models.downloader._free_bytes", lambda _path: 10**6)

        with (
            Downloader(work_dir, transport=server.transport) as downloader,
            pytest.raises(NotEnoughSpaceError) as excinfo,
        ):
            downloader.check_space(huge, tmp_path)

        assert not server.requests, "проверка места должна идти до запроса"
        assert "Недостаточно места" in excinfo.value.user_message

    def test_a_model_that_fits_passes(
        self,
        work_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ayris.models.downloader._free_bytes", lambda _path: 10**10)

        with Downloader(work_dir, transport=Server().transport) as downloader:
            downloader.check_space(entry(), tmp_path)

    def test_human_size_uses_a_russian_comma(self) -> None:
        assert human_size(1536) == "1,5 КБ"
        assert human_size(512) == "512 Б"


# ----------------------------------------------------------------------
# archives
# ----------------------------------------------------------------------


class TestArchives:
    def test_a_zip_is_unpacked_into_the_kind_directory(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        body = zip_bytes({"model/conf.json": b"{}", "model/am/final.mdl": b"weights"})
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)
        result = Installer(models_root).install(model, downloads)

        assert (result.path / "conf.json").read_bytes() == b"{}"
        assert (result.path / "am" / "final.mdl").read_bytes() == b"weights"

    def test_a_single_top_level_directory_is_stripped(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        """Vosk archives wrap everything in one folder; installing it twice-nested
        would leave the settings pointing at a path that does not exist."""
        body = zip_bytes({"vosk-model/conf/model.conf": b"conf"})
        model = _archive_entry(body, target="vosk-model")
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)
        result = Installer(models_root).install(model, downloads)

        assert result.path == models_root / "stt" / "vosk-model"
        assert (result.path / "conf" / "model.conf").exists()
        assert not (result.path / "vosk-model").exists()

    def test_a_tar_is_unpacked_too(self, work_dir: Path, models_root: Path) -> None:
        body = tar_bytes({"a.txt": b"first", "b.txt": b"second"})
        model = _archive_entry(body, archive="tar")
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)
        result = Installer(models_root).install(model, downloads)

        assert (result.path / "a.txt").read_bytes() == b"first"

    def test_a_traversing_member_is_refused(self, work_dir: Path, models_root: Path) -> None:
        body = zip_bytes({"../../evil.txt": b"pwned"})
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)

        with pytest.raises(ArchiveError) as excinfo:
            Installer(models_root).install(model, downloads)

        assert "за пределы" in excinfo.value.user_message
        assert not (models_root.parent / "evil.txt").exists()

    def test_a_windows_style_traversal_is_refused(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        """``zipfile`` does not treat a backslash as a separator, so the check has
        to normalise before deciding."""
        body = zip_bytes({r"..\..\evil.txt": b"pwned"})
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)

        with pytest.raises(ArchiveError):
            Installer(models_root).install(model, downloads)

    def test_an_absolute_member_is_refused(self, work_dir: Path, models_root: Path) -> None:
        body = zip_bytes({"/etc/passwd": b"pwned"})
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)

        with pytest.raises(ArchiveError):
            Installer(models_root).install(model, downloads)

    def test_a_failed_extraction_leaves_nothing_behind(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        body = zip_bytes({"good.txt": b"ok", "../evil.txt": b"pwned"})
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)

        with pytest.raises(ArchiveError):
            Installer(models_root).install(model, downloads)

        assert not (models_root / "stt" / model.target).exists()
        assert not list((models_root / "stt").glob("*.incomplete"))

    def test_a_corrupt_archive_is_reported_in_russian(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        body = b"not a zip" * 100
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)

        with pytest.raises(ArchiveError) as excinfo:
            Installer(models_root).install(model, downloads)

        assert "повреждён" in excinfo.value.user_message

    def test_installing_writes_a_manifest(self, work_dir: Path, models_root: Path) -> None:
        body = zip_bytes({"model/a.bin": b"first", "model/b.bin": b"second"})
        model = _archive_entry(body)
        server = Server(body=body)

        with Downloader(work_dir, transport=server.transport) as downloader:
            downloads = downloader.fetch_all(model)
        result = Installer(models_root).install(model, downloads)

        manifest = read_manifest(result.path)
        assert manifest is not None
        assert manifest.catalog_id == model.id
        names = {item.name for item in manifest.files}
        assert {"a.bin", "b.bin"} <= names

    def test_reinstalling_replaces_the_previous_copy(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        installer = Installer(models_root)
        first = zip_bytes({"model/old.txt": b"old"})
        model = _archive_entry(first)
        with Downloader(work_dir, transport=Server(body=first).transport) as downloader:
            installer.install(model, downloader.fetch_all(model))

        second = zip_bytes({"model/new.txt": b"new"})
        updated = _archive_entry(second)
        with Downloader(work_dir, transport=Server(body=second).transport) as downloader:
            result = installer.install(updated, downloader.fetch_all(updated))

        assert (result.path / "new.txt").exists()
        assert not (result.path / "old.txt").exists()
        assert not list((models_root / "stt").glob("*.old"))

    def test_the_archive_is_deleted_after_unpacking(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        body = zip_bytes({"model/a.bin": b"payload"})
        model = _archive_entry(body)

        with Downloader(work_dir, transport=Server(body=body).transport) as downloader:
            downloads = downloader.fetch_all(model)
            Installer(models_root).install(model, downloads)

        assert not downloads[0].path.exists()


# ----------------------------------------------------------------------
# subdirectories without an archive
# ----------------------------------------------------------------------


class TestSubdirectory:
    """``directory``: several loose files that need a folder of their own.

    A GigaAM export is weights plus a vocabulary, and ``onnx-asr`` looks for them
    by globbing a folder — ``v?_ctc*.onnx``, ``v?_vocab.txt`` — rather than by the
    path it was given. Laid out flat in ``models/stt`` the second variant would
    match the first one's glob and the engine would load whichever came first, so
    the record names a folder and every file of it lands inside. Staged and
    swapped like an archive, for the same reason: a half-moved set of files looks
    installed and then fails on load.
    """

    def test_the_files_land_in_the_subdirectory(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        model = _directory_entry()

        with Downloader(work_dir, transport=_vendor_transport()) as downloader:
            downloads = downloader.fetch_all(model)
        result = Installer(models_root).install(model, downloads)

        assert result.path == models_root / "stt" / "gigaam-v3-ctc"
        assert result.name == "gigaam-v3-ctc"
        assert (result.path / "v3_ctc.int8.onnx").read_bytes() == PAYLOAD
        assert (result.path / "v3_vocab.txt").read_bytes() == VOCAB
        assert not (models_root / "stt" / "v3_ctc.int8.onnx").exists()

    def test_the_destination_is_the_folder_before_anything_is_downloaded(
        self,
        models_root: Path,
    ) -> None:
        """``fetch_models.py`` and the settings ask this before spending 215 МБ."""
        installer = Installer(models_root)
        model = _directory_entry()

        assert installer.destination(model) == models_root / "stt" / "gigaam-v3-ctc"
        assert not installer.is_installed(model)

    def test_two_variants_keep_their_own_folders(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        """The whole point: identical file names, and neither shadows the other."""
        installer = Installer(models_root)
        plain = _directory_entry()
        punctuated = _directory_entry(id="test-model-e2e", directory="gigaam-v3-e2e-ctc")

        for model in (plain, punctuated):
            with Downloader(work_dir, transport=_vendor_transport()) as downloader:
                installer.install(model, downloader.fetch_all(model))

        stt = models_root / "stt"
        assert (stt / "gigaam-v3-ctc" / "v3_ctc.int8.onnx").is_file()
        assert (stt / "gigaam-v3-e2e-ctc" / "v3_ctc.int8.onnx").is_file()
        assert sorted(path.name for path in stt.iterdir()) == [
            "gigaam-v3-ctc",
            "gigaam-v3-e2e-ctc",
        ]

    def test_installing_writes_a_manifest(self, work_dir: Path, models_root: Path) -> None:
        """Without it «проверить целостность» could only ever compare sizes: the
        catalog's ``sha256`` covers one file of the several that arrived."""
        model = _directory_entry()

        with Downloader(work_dir, transport=_vendor_transport()) as downloader:
            result = Installer(models_root).install(model, downloader.fetch_all(model))

        manifest = read_manifest(result.path)
        assert manifest is not None
        assert manifest.catalog_id == model.id
        assert {item.name for item in manifest.files} == {"v3_ctc.int8.onnx", "v3_vocab.txt"}
        assert manifest.total_bytes == model.total_bytes
        assert (result.path / MANIFEST_NAME).is_file()

    def test_reinstalling_replaces_the_previous_copy(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        """A file the new release dropped has to go, or the loader's glob finds it."""
        installer = Installer(models_root)
        model = _directory_entry()
        with Downloader(work_dir, transport=_vendor_transport()) as downloader:
            first = installer.install(model, downloader.fetch_all(model))
        (first.path / "v3_ctc_stale.onnx").write_bytes("из прошлого релиза".encode())

        with Downloader(work_dir, transport=_vendor_transport()) as downloader:
            again = installer.install(model, downloader.fetch_all(model))

        assert not (again.path / "v3_ctc_stale.onnx").exists()
        assert (again.path / "v3_vocab.txt").read_bytes() == VOCAB
        assert not list((models_root / "stt").glob("*.old"))
        assert not list((models_root / "stt").glob("*.incomplete"))

    def test_a_failed_install_keeps_the_working_copy(
        self,
        work_dir: Path,
        models_root: Path,
    ) -> None:
        """All of them or none. Weights that arrived without their vocabulary are
        not a model, they are a folder that makes the engine raise on load — and
        the copy that was already installed has to survive the attempt."""
        installer = Installer(models_root)
        model = _directory_entry()
        with Downloader(work_dir, transport=_vendor_transport()) as downloader:
            first = installer.install(model, downloader.fetch_all(model))
        (first.path / "keep.txt").write_bytes("эта копия должна остаться".encode())

        with Downloader(work_dir, transport=_vendor_transport()) as downloader:
            downloads = downloader.fetch_all(model)
        downloads[1].path.unlink()

        with pytest.raises(InstallError) as excinfo:
            installer.install(model, downloads)

        assert "не открыты другой программой" in excinfo.value.user_message
        assert (first.path / "keep.txt").exists()
        assert (first.path / "v3_vocab.txt").exists()
        assert not list((models_root / "stt").glob("*.incomplete"))

    def test_the_registry_registers_the_folder_and_deletes_it_whole(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        """The row points at the folder, and removal takes the folder with it.

        The companion lookup has to keep its hands off a record like this: the
        vocabulary lives *inside* the folder, so the path it would build —
        ``models/stt/v3_vocab.txt`` — is somebody else's file or nothing at all.
        """
        model = _directory_entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=_vendor_transport(),
        )

        record = registry.install(model.id)

        folder = profile_paths.model_dir("stt") / "gigaam-v3-ctc"
        assert record_path(record) == folder
        assert record.name == "gigaam-v3-ctc"
        assert registry.installed("stt") == [record]

        freed = registry.remove(record, force=True)

        assert freed >= model.total_bytes
        assert not folder.exists()
        assert registry.installed("stt") == []


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


class TestRegistry:
    def test_installing_registers_the_model(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )

        record = registry.install(model.id)

        assert record.id is not None
        assert record.catalog_id == model.id
        assert record.engine == "vosk"
        assert record.size_bytes == len(PAYLOAD)
        assert (profile_paths.model_dir("stt") / "model.bin").read_bytes() == PAYLOAD

    def test_the_first_model_of_a_kind_becomes_active(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )

        record = registry.install(model.id)

        assert record.is_active is True
        changed = recorder.of(ActiveModelChanged)
        assert changed and changed[-1].name == "model.bin"
        assert changed[-1].path.endswith("model.bin")

    def test_a_second_model_does_not_steal_the_selection(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        first = entry()
        second = entry(id="second", url="https://models.test/ayris/second.bin")
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((first, second)),
            bus=bus,
            transport=Server().transport,
        )

        registry.install(first.id)
        registry.install(second.id)

        active = registry.active("stt")
        assert active is not None and active.catalog_id == first.id

    def test_switching_the_active_model_publishes_the_change(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        first = entry()
        second = entry(id="second", url="https://models.test/ayris/second.bin")
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((first, second)),
            bus=bus,
            transport=Server().transport,
        )
        registry.install(first.id)
        other = registry.install(second.id)

        activated = registry.set_active(other)

        assert activated.is_active is True
        assert registry.active("stt") is not None
        latest = recorder.of(ActiveModelChanged)[-1]
        assert latest.catalog_id == "second"
        assert latest.cleared is False

    def test_activating_an_unknown_model_raises(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
    ) -> None:
        registry = make_registry(repos, profile_paths, catalog=ModelCatalog())

        with pytest.raises(NotInstalledError):
            registry.set_active(404)

    def test_reinstalling_updates_the_row_instead_of_adding_one(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )

        first = registry.install(model.id)
        second = registry.install(model.id)

        assert first.id == second.id
        assert len(registry.installed("stt")) == 1

    def test_a_hand_placed_model_can_be_registered(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        target = profile_paths.model_dir("stt") / "своя-модель"
        target.mkdir(parents=True)
        (target / "final.mdl").write_bytes(b"weights")
        registry = make_registry(repos, profile_paths, catalog=ModelCatalog(), bus=bus)

        record = registry.register_local("stt", "своя-модель", engine="vosk", activate=True)

        assert record.catalog_id == ""
        assert record.size_bytes == len(b"weights")
        assert record.is_active is True

    def test_registering_something_absent_raises(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
    ) -> None:
        registry = make_registry(repos, profile_paths, catalog=ModelCatalog())

        with pytest.raises(NotInstalledError):
            registry.register_local("stt", "нет-такой")


class TestRemoval:
    def test_removing_deletes_files_and_row(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)

        freed = registry.remove(record, force=True)

        assert freed == len(PAYLOAD)
        assert not (profile_paths.model_dir("stt") / "model.bin").exists()
        assert registry.installed("stt") == []
        removed = recorder.of(ModelRemoved)
        assert removed and removed[-1].freed_bytes == len(PAYLOAD)

    def test_removing_the_active_model_needs_force(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)

        with pytest.raises(ModelInUseError):
            registry.remove(record)

        assert (profile_paths.model_dir("stt") / "model.bin").exists()

    def test_removing_the_active_model_clears_the_selection_first(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
        recorder: Recorder,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)

        registry.remove(record, force=True)

        cleared = [item for item in recorder.of(ActiveModelChanged) if item.cleared]
        assert cleared, "подписчики не узнали, что модель снята"
        assert registry.active("stt") is None

    def test_a_model_a_worker_holds_is_not_deleted(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
            is_busy=lambda _record: True,
        )
        record = registry.install(model.id)

        with pytest.raises(ModelInUseError) as excinfo:
            registry.remove(record, force=True)

        assert "используется" in excinfo.value.user_message
        assert (profile_paths.model_dir("stt") / "model.bin").exists()

    def test_companions_are_deleted_with_the_model(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        config = b'{"voice": "irina"}'

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".json"):
                return httpx.Response(200, content=config)
            return Server().handle(request)

        voice = entry(
            kind="tts",
            engine="piper",
            extra_files=[
                {
                    "url": "https://models.test/ayris/model.bin.json",
                    "sha256": hashlib.sha256(config).hexdigest(),
                    "size_bytes": len(config),
                }
            ],
        )
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((voice,)),
            bus=bus,
            transport=httpx.MockTransport(handle),
        )
        record = registry.install(voice.id)

        registry.remove(record, force=True)

        assert not (profile_paths.model_dir("tts") / "model.bin").exists()
        assert not (profile_paths.model_dir("tts") / "model.bin.json").exists()

    def test_purging_downloads_keeps_installed_models(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        registry.install(model.id)
        registry.download_dir.mkdir(parents=True, exist_ok=True)
        (registry.download_dir / "leftover.part").write_bytes(b"x" * 10)

        freed = registry.purge_downloads()

        assert freed >= 10
        assert (profile_paths.model_dir("stt") / "model.bin").exists()


class TestAccounting:
    def test_disk_usage_sums_per_kind(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        stt = entry()
        tts = entry(id="voice", kind="tts", engine="piper", url="https://models.test/a/voice.bin")
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((stt, tts)),
            bus=bus,
            transport=Server().transport,
        )
        registry.install(stt.id)
        registry.install(tts.id)

        usage = registry.disk_usage()

        assert usage.by_kind["stt"] == len(PAYLOAD)
        assert usage.by_kind["tts"] == len(PAYLOAD)
        assert usage.total == 2 * len(PAYLOAD)
        assert "КБ" in usage.human_total

    def test_measuring_walks_the_directories(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        registry.install(model.id)

        measured = registry.disk_usage(measure=True)

        assert measured.by_kind["stt"] == len(PAYLOAD)


class TestIntegrityChecks:
    def test_an_untouched_model_checks_out(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)

        report = registry.verify(record, full=True)

        assert report.status is IntegrityStatus.OK
        assert report.ok is True

    def test_a_damaged_file_is_reported_as_corrupted(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)
        installed = profile_paths.model_dir("stt") / "model.bin"
        installed.write_bytes(PAYLOAD[:-1])

        report = registry.verify(record)

        assert report.status is IntegrityStatus.CORRUPTED
        assert report.can_redownload is True

    def test_a_silently_edited_file_needs_the_full_check(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        """Same length, different bytes — only the digest can tell."""
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)
        installed = profile_paths.model_dir("stt") / "model.bin"
        installed.write_bytes(b"\x00" * len(PAYLOAD))

        assert registry.verify(record).status is IntegrityStatus.OK
        assert registry.verify(record, full=True).status is IntegrityStatus.CORRUPTED

    def test_a_deleted_model_is_reported_as_missing(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        model = entry()
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server().transport,
        )
        record = registry.install(model.id)
        (profile_paths.model_dir("stt") / "model.bin").unlink()

        report = registry.verify(record)

        assert report.status is IntegrityStatus.MISSING
        assert [item.id for item in registry.missing()] == [record.id]

    def test_an_unpacked_model_is_verified_against_its_manifest(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        body = zip_bytes({"model/a.bin": b"first", "model/b.bin": b"second"})
        model = _archive_entry(body)
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server(body=body).transport,
        )
        record = registry.install(model.id)

        assert registry.verify(record, full=True).status is IntegrityStatus.OK

        # Same length, different bytes: the quick pass cannot see it, the digest can.
        (record_path(record) / "a.bin").write_bytes(b"FIRST")
        assert registry.verify(record).status is IntegrityStatus.OK
        assert registry.verify(record, full=True).status is IntegrityStatus.CORRUPTED

    def test_a_file_missing_from_the_manifest_is_noticed(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        body = zip_bytes({"model/a.bin": b"first", "model/b.bin": b"second"})
        model = _archive_entry(body)
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server(body=body).transport,
        )
        record = registry.install(model.id)

        (record_path(record) / "b.bin").unlink()

        assert registry.verify(record).status is IntegrityStatus.MISSING

    def test_a_hand_placed_model_is_unverified_not_broken(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        target = profile_paths.model_dir("stt") / "своя-модель"
        target.mkdir(parents=True)
        (target / "final.mdl").write_bytes(b"weights")
        registry = make_registry(repos, profile_paths, catalog=ModelCatalog(), bus=bus)
        record = registry.register_local("stt", "своя-модель")

        report = registry.verify(record, full=True)

        assert report.status is IntegrityStatus.UNVERIFIED
        assert report.ok is True
        assert report.can_redownload is False

    def test_verify_all_covers_every_row(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        stt = entry()
        tts = entry(id="voice", kind="tts", engine="piper", url="https://models.test/a/voice.bin")
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((stt, tts)),
            bus=bus,
            transport=Server().transport,
        )
        registry.install(stt.id)
        registry.install(tts.id)

        reports = registry.verify_all()

        assert len(reports) == 2
        assert all(report.ok for report in reports)

    def test_a_manifest_survives_being_read_back(
        self,
        repos: Repositories,
        profile_paths: AppPaths,
        bus: EventBus,
    ) -> None:
        body = zip_bytes({"model/a.bin": b"payload"})
        model = _archive_entry(body)
        registry = make_registry(
            repos,
            profile_paths,
            catalog=ModelCatalog((model,)),
            bus=bus,
            transport=Server(body=body).transport,
        )
        record = registry.install(model.id)

        manifest = read_manifest(record_path(record))

        assert manifest is not None
        assert manifest.archive_sha256 == hashlib.sha256(body).hexdigest()
        assert (record_path(record) / MANIFEST_NAME).exists()


# ----------------------------------------------------------------------
# small helpers used by several groups
# ----------------------------------------------------------------------


def _archive_entry(body: bytes, *, archive: str = "zip", target: str = "") -> ModelEntry:
    """A catalog entry describing an in-memory archive."""
    fields: dict[str, Any] = {
        "archive": archive,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
    }
    if target:
        fields["target"] = target
    return entry(**fields)


def _directory_entry(*, directory: str = "gigaam-v3-ctc", **overrides: Any) -> ModelEntry:
    """A record shaped like a GigaAM export: weights, a vocabulary, one folder.

    The names matter and are the vendor's, not ours: ``onnx-asr`` globs the folder
    for ``v?_ctc*.onnx`` and ``v?_vocab.txt``, so renaming either file to
    something tidier is how a model stops loading.
    """
    return entry(
        url="https://models.test/ayris/v3_ctc.int8.onnx",
        directory=directory,
        extra_files=[
            {
                "url": "https://models.test/ayris/v3_vocab.txt",
                "sha256": VOCAB_SHA,
                "size_bytes": len(VOCAB),
            }
        ],
        **overrides,
    )


def _vendor_transport() -> httpx.MockTransport:
    """Serves :data:`PAYLOAD` for the weights and :data:`VOCAB` for the vocabulary."""
    server = Server()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".txt"):
            return httpx.Response(200, content=VOCAB)
        return server.handle(request)

    return httpx.MockTransport(handle)


def _flaky(server: Server) -> httpx.MockTransport:
    """A transport that serves ``server`` and then refuses every later request.

    Reproduces "the download died and the process went away": the first attempt
    leaves a ``.part`` file, and the retries inside one ``fetch`` cannot quietly
    finish the job and hide the fact that resume was never exercised.
    """
    state = {"first": True}

    def handle(request: httpx.Request) -> httpx.Response:
        if state["first"]:
            state["first"] = False
            return server.handle(request)
        raise httpx.ConnectError("сеть пропала")

    return httpx.MockTransport(handle)


def record_path(record: ModelRecord) -> Path:
    """The installed location of a registered model."""
    return PathType(record.path)
