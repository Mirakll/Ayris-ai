"""Task 11: the four cloud recognisers, on mocked HTTP.

Not one test here opens a socket. ``httpx.MockTransport`` is handed to the engine
through ``SttOptions.extra["transport"]``, which is the seam
:class:`~ayris.audio.stt.cloud_base.CloudSttEngine` builds its client around, so
the request that a provider *would* have sent is inspected as an object instead
of being guessed at. That is the strict version of this test: an assertion can
say "the folder id ended up in the query string" and "the key ended up in an
``Api-Key`` header", which a live account would only tell us by working or not.

What mocks cannot check is whether the real service agrees with our idea of its
request format. The task tracks that separately as a manual step with a real key;
everything mechanically checkable is here.

Groups:

* :class:`TestCredentials` — the key comes from keyring, by reference.
* :class:`TestYandex` / :class:`TestGoogle` / :class:`TestAzure` /
  :class:`TestOpenAi` — the request each provider builds and the answer it parses.
* :class:`TestHttpErrors` — 401, 429 and 5xx become different exception types.
* :class:`TestRetries` — what is retried, what is not, and for how long.
* :class:`TestLogging` — no key and no audio in the log.
* :class:`TestRegistry` — the four names resolve to the four classes.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import pytest

from ayris.audio.stt.azure_engine import AzureSttEngine
from ayris.audio.stt.base import AudioBuffer, SttOptions
from ayris.audio.stt.cloud_base import (
    ASSUMED_CONFIDENCE,
    CLOUD_ENGINE_ENTRYPOINTS,
    CloudSttEngine,
    NetworkError,
    QuotaError,
    as_wav,
    cloud_engine_names,
    create_cloud_engine,
)
from ayris.audio.stt.google_engine import GoogleSttEngine
from ayris.audio.stt.openai_engine import OpenAiSttEngine
from ayris.audio.stt.yandex_engine import YandexSttEngine
from ayris.core.errors import SttError
from ayris.core.secrets import SecretsStore, reset_secrets
from ayris.utils.logger import ROOT_LOGGER_NAME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

#: The key every test stores, so an assertion can look for it by value.
KEY = "test-secret-key"


class FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager.

    The sandbox has no keyring backend at all, so without this the engines could
    not even be constructed here. Duplicated from ``test_config.py`` on purpose:
    a shared fixture would make the two modules fail together.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.entries.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.entries[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self.entries[(service_name, username)]


@pytest.fixture
def keyring_store() -> Iterator[SecretsStore]:
    """A process-wide store holding :data:`KEY` under every provider's ref."""
    store = SecretsStore("Ayris-test", backend=FakeKeyring())
    for ref in CLOUD_ENGINE_ENTRYPOINTS:
        store.save(ref, KEY)
    reset_secrets(store)
    yield store
    reset_secrets()


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree.

    ``setup_logging`` sets ``propagate = False`` on that logger, and ``caplog``
    listens on the interpreter root — so plain ``caplog.at_level`` sees nothing
    once any earlier test in the run has configured logging. Attaching the
    handler directly makes the masking assertions independent of test order.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


def speech(ms: int = 900, amplitude: int = 9000) -> AudioBuffer:
    """Audible 16 kHz mono audio, long enough to pass the silence check."""
    from array import array

    rate = 16000
    count = rate * ms // 1000
    samples = array(
        "h",
        (int(amplitude * math.sin(2.0 * math.pi * 220.0 * i / rate)) for i in range(count)),
    )
    return AudioBuffer(samples.tobytes(), sample_rate=rate, channels=1)


class Recorder:
    """Captures the request an engine sent and replies with a canned answer."""

    def __init__(self, status: int = 200, payload: Any = None, body: bytes | None = None) -> None:
        self.status = status
        self.payload = payload if payload is not None else {}
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        # read() so the body is available after the transport returns.
        request.read()
        self.requests.append(request)
        if self.body is not None:
            return httpx.Response(self.status, content=self.body)
        return httpx.Response(self.status, json=self.payload)

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "engine sent no request"
        return self.requests[-1]

    @property
    def count(self) -> int:
        return len(self.requests)


def raiser(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    """A transport handler that always fails, to test the retry loop."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise exc

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def options(**extra: Any) -> SttOptions:
    """Engine options with the mock transport and the usual provider settings."""
    handler = extra.pop("handler", None)
    transport: dict[str, Any] = {}
    if handler is not None:
        transport["transport"] = httpx.MockTransport(handler)
    return SttOptions(
        language="ru",
        punctuation=True,
        extra={
            "folder_id": "b1gtest",
            "region": "westeurope",
            **transport,
            **extra,
        },
    )


def loaded(engine: CloudSttEngine, tmp_path: Path, **extra: Any) -> CloudSttEngine:
    """Load an engine against the mock transport. ``model_path`` is ignored."""
    engine.load(tmp_path, options(**extra))
    return engine


class TestCredentials:
    """«Готово когда»: keys are read from keyring, and their absence is explained."""

    def test_the_key_comes_from_the_store_by_reference(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "привет"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(speech())
        assert recorder.last.headers["Authorization"] == f"Api-Key {KEY}"

    def test_a_missing_key_names_the_reference_and_offers_offline(self, tmp_path: Path) -> None:
        reset_secrets(SecretsStore("Ayris-empty", backend=FakeKeyring()))
        try:
            with pytest.raises(SttError) as info:
                loaded(YandexSttEngine(), tmp_path, handler=Recorder())
        finally:
            reset_secrets()
        assert "yandex" in info.value.technical
        assert "офлайн" in info.value.user_message

    def test_an_explicit_credential_beats_the_store(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "да"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder, credential="inline-key")
        engine.transcribe(speech())
        assert recorder.last.headers["Authorization"] == "Api-Key inline-key"

    def test_the_engine_reports_itself_as_a_cloud_device(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(GoogleSttEngine(), tmp_path, handler=Recorder())
        assert engine.device == "cloud"
        assert engine.loaded
        engine.unload()
        assert not engine.loaded


class TestYandex:
    """SpeechKit: raw LPCM in the body, everything else in the query string."""

    def test_a_transcript_carries_text_confidence_and_timing(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "привет мир"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder)
        result = engine.transcribe(speech())
        assert result.text == "привет мир"
        assert result.engine == "yandex"
        assert result.device == "cloud"
        assert result.confidence == pytest.approx(ASSUMED_CONFIDENCE)
        assert result.duration_ms == pytest.approx(900.0, abs=1.0)
        assert result.inference_ms > 0.0

    def test_the_request_carries_the_folder_and_the_raw_pcm(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "да"})
        audio = speech()
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(audio)
        query = dict(recorder.last.url.params)
        assert query["folderId"] == "b1gtest"
        assert query["lang"] == "ru-RU"
        assert query["format"] == "lpcm"
        assert query["sampleRateHertz"] == "16000"
        assert recorder.last.content == audio.pcm

    def test_punctuation_off_asks_for_raw_results(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "да"})
        engine = YandexSttEngine()
        engine.load(
            tmp_path,
            SttOptions(
                language="ru",
                punctuation=False,
                extra={"folder_id": "b1", "transport": httpx.MockTransport(recorder)},
            ),
        )
        engine.transcribe(speech())
        assert dict(recorder.last.url.params)["rawResults"] == "true"

    def test_an_iam_token_uses_the_bearer_scheme(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "да"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder, auth_scheme="iam")
        engine.transcribe(speech())
        assert recorder.last.headers["Authorization"] == f"Bearer {KEY}"

    def test_an_unknown_auth_scheme_is_refused_before_the_request(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "да"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder, auth_scheme="oauth2")
        with pytest.raises(SttError, match="unknown auth scheme"):
            engine.transcribe(speech())
        assert recorder.count == 0

    def test_a_missing_folder_id_is_caught_locally(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "да"})
        engine = YandexSttEngine()
        engine.load(
            tmp_path,
            SttOptions(extra={"transport": httpx.MockTransport(recorder)}),
        )
        with pytest.raises(SttError) as info:
            engine.transcribe(speech())
        assert "folder_id" in info.value.technical
        assert recorder.count == 0

    def test_an_empty_result_is_an_empty_transcript_not_an_error(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(YandexSttEngine(), tmp_path, handler=Recorder(payload={"result": ""}))
        result = engine.transcribe(speech())
        assert result.is_empty
        assert result.text == ""

    def test_a_response_without_the_result_field_is_an_error(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(YandexSttEngine(), tmp_path, handler=Recorder(payload={"oops": 1}))
        with pytest.raises(SttError, match="no 'result' field"):
            engine.transcribe(speech())


class TestGoogle:
    """Cloud Speech-to-Text v1: JSON in, base64 audio, word offsets out."""

    RESPONSE: ClassVar[dict[str, Any]] = {
        "results": [
            {
                "alternatives": [
                    {
                        "transcript": "привет мир",
                        "confidence": 0.87,
                        "words": [
                            {"word": "привет", "startTime": "0.100s", "endTime": "0.500s"},
                            {"word": "мир", "startTime": "0.600s", "endTime": "0.900s"},
                        ],
                    }
                ]
            }
        ]
    }

    def test_the_transcript_keeps_the_reported_confidence(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(GoogleSttEngine(), tmp_path, handler=Recorder(payload=self.RESPONSE))
        result = engine.transcribe(speech())
        assert result.text == "привет мир"
        assert result.confidence == pytest.approx(0.87)
        assert result.engine == "google"

    def test_word_offsets_become_segments(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(GoogleSttEngine(), tmp_path, handler=Recorder(payload=self.RESPONSE))
        result = engine.transcribe(speech())
        assert [segment.text for segment in result.segments] == ["привет", "мир"]
        assert result.segments[0].start_ms == pytest.approx(100.0)
        assert result.segments[1].end_ms == pytest.approx(900.0)

    def test_the_audio_travels_base64_encoded_with_a_linear16_config(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload=self.RESPONSE)
        audio = speech()
        engine = loaded(GoogleSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(audio)
        sent = json.loads(recorder.last.content)
        assert sent["config"]["encoding"] == "LINEAR16"
        assert sent["config"]["languageCode"] == "ru-RU"
        assert sent["config"]["enableAutomaticPunctuation"] is True
        assert base64.b64decode(sent["audio"]["content"]) == audio.pcm

    def test_the_key_travels_in_a_header_not_the_query_string(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload=self.RESPONSE)
        engine = loaded(GoogleSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(speech())
        assert recorder.last.headers["X-Goog-Api-Key"] == KEY
        assert KEY not in str(recorder.last.url)

    def test_no_results_means_nothing_was_recognised(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(GoogleSttEngine(), tmp_path, handler=Recorder(payload={"results": []}))
        assert engine.transcribe(speech()).is_empty


class TestAzure:
    """Azure Speech: WAV upload, detailed JSON, region in the host name."""

    RESPONSE: ClassVar[dict[str, Any]] = {
        "RecognitionStatus": "Success",
        "DisplayText": "привет мир",
        "NBest": [{"Confidence": 0.81, "Display": "привет мир"}],
        "Offset": 1000000,
        "Duration": 9000000,
    }

    def test_the_display_text_and_confidence_are_used(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(AzureSttEngine(), tmp_path, handler=Recorder(payload=self.RESPONSE))
        result = engine.transcribe(speech())
        assert result.text == "привет мир"
        assert result.confidence == pytest.approx(0.81)
        assert result.engine == "azure"

    def test_the_region_builds_the_host_and_the_body_is_a_wav(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload=self.RESPONSE)
        engine = loaded(AzureSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(speech())
        assert recorder.last.url.host == "westeurope.stt.speech.microsoft.com"
        assert recorder.last.headers["Ocp-Apim-Subscription-Key"] == KEY
        assert recorder.last.content[:4] == b"RIFF"

    def test_a_missing_region_is_caught_locally(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload=self.RESPONSE)
        engine = AzureSttEngine()
        engine.load(tmp_path, SttOptions(extra={"transport": httpx.MockTransport(recorder)}))
        with pytest.raises(SttError) as info:
            engine.transcribe(speech())
        assert "region" in info.value.technical
        assert recorder.count == 0

    def test_no_match_is_an_empty_transcript(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(
            AzureSttEngine(),
            tmp_path,
            handler=Recorder(payload={"RecognitionStatus": "NoMatch"}),
        )
        assert engine.transcribe(speech()).is_empty

    def test_an_error_status_from_the_service_is_an_error(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(
            AzureSttEngine(),
            tmp_path,
            handler=Recorder(payload={"RecognitionStatus": "Error"}),
        )
        with pytest.raises(SttError, match="RecognitionStatus"):
            engine.transcribe(speech())


class TestOpenAi:
    """Whisper API: multipart upload, verbose JSON, logprobs as confidence."""

    RESPONSE: ClassVar[dict[str, Any]] = {
        "text": "привет мир",
        "segments": [
            {"start": 0.1, "end": 0.5, "text": " привет", "avg_logprob": -0.2},
            {"start": 0.6, "end": 0.9, "text": " мир", "avg_logprob": -0.4},
        ],
    }

    def test_segments_and_confidence_come_from_the_verbose_answer(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(OpenAiSttEngine(), tmp_path, handler=Recorder(payload=self.RESPONSE))
        result = engine.transcribe(speech())
        assert result.text == "привет мир"
        assert [segment.text for segment in result.segments] == ["привет", "мир"]
        # exp(mean(-0.2, -0.4)) ≈ 0.74: a plausible transcript, not a certain one.
        assert result.confidence == pytest.approx(math.exp(-0.3), abs=0.01)

    def test_the_upload_is_multipart_with_a_wav_and_the_model(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload=self.RESPONSE)
        engine = loaded(OpenAiSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(speech())
        assert recorder.last.headers["Authorization"] == f"Bearer {KEY}"
        assert "multipart/form-data" in recorder.last.headers["content-type"]
        body = recorder.last.content
        assert b'name="model"' in body
        assert b"whisper-1" in body
        assert b"RIFF" in body

    def test_a_plain_text_answer_still_produces_a_transcript(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(OpenAiSttEngine(), tmp_path, handler=Recorder(payload={"text": "привет"}))
        result = engine.transcribe(speech())
        assert result.text == "привет"
        assert result.confidence == pytest.approx(ASSUMED_CONFIDENCE)

    def test_an_empty_text_is_an_empty_transcript(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(OpenAiSttEngine(), tmp_path, handler=Recorder(payload={"text": "  "}))
        assert engine.transcribe(speech()).is_empty


class TestHttpErrors:
    """«Готово когда»: 401, 429 and 5xx are three different problems."""

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_rejected_key_says_so_and_is_not_a_network_error(
        self, keyring_store: SecretsStore, tmp_path: Path, status: int
    ) -> None:
        engine = loaded(
            YandexSttEngine(), tmp_path, handler=Recorder(status=status, payload={"e": "no"})
        )
        with pytest.raises(SttError) as info:
            engine.transcribe(speech())
        assert not isinstance(info.value, NetworkError | QuotaError)
        assert "ключ" in info.value.user_message

    def test_a_quota_error_is_its_own_type(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(YandexSttEngine(), tmp_path, handler=Recorder(status=429))
        with pytest.raises(QuotaError) as info:
            engine.transcribe(speech())
        assert "лимит" in info.value.user_message

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_a_server_error_is_a_network_error(
        self, keyring_store: SecretsStore, tmp_path: Path, status: int
    ) -> None:
        engine = loaded(
            YandexSttEngine(),
            tmp_path,
            handler=Recorder(status=status),
            max_retries=0,
        )
        with pytest.raises(NetworkError):
            engine.transcribe(speech())

    def test_a_body_that_is_not_json_is_reported_as_such(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        engine = loaded(YandexSttEngine(), tmp_path, handler=Recorder(body=b"<html>gateway</html>"))
        with pytest.raises(SttError, match="not json|JSON"):
            engine.transcribe(speech())

    def test_silence_never_reaches_the_network(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(payload={"result": "не должно быть"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder)
        result = engine.transcribe(AudioBuffer(b"\x00\x00" * 16000, sample_rate=16000))
        assert result.is_empty
        assert recorder.count == 0


class TestRetries:
    """A timeout is retried, a quota is not, and the deadline caps both."""

    def test_a_timeout_is_retried_then_raised_as_a_network_error(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        handler = raiser(httpx.ConnectTimeout("too slow"))
        engine = loaded(
            YandexSttEngine(),
            tmp_path,
            handler=handler,
            max_retries=2,
            deadline_sec=30.0,
        )
        with pytest.raises(NetworkError) as info:
            engine.transcribe(speech())
        assert len(handler.calls) == 3  # type: ignore[attr-defined]
        assert "интернет" in info.value.user_message

    def test_a_quota_error_is_never_retried(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(status=429)
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder, max_retries=3)
        with pytest.raises(QuotaError):
            engine.transcribe(speech())
        assert recorder.count == 1

    def test_a_rejected_key_is_never_retried(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        recorder = Recorder(status=401)
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder, max_retries=3)
        with pytest.raises(SttError):
            engine.transcribe(speech())
        assert recorder.count == 1

    def test_zero_retries_means_one_attempt(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        handler = raiser(httpx.ReadTimeout("nope"))
        engine = loaded(YandexSttEngine(), tmp_path, handler=handler, max_retries=0)
        with pytest.raises(NetworkError):
            engine.transcribe(speech())
        assert len(handler.calls) == 1  # type: ignore[attr-defined]

    def test_a_tight_deadline_stops_the_retries_early(
        self, keyring_store: SecretsStore, tmp_path: Path
    ) -> None:
        handler = raiser(httpx.ConnectError("down"))
        engine = loaded(
            YandexSttEngine(),
            tmp_path,
            handler=handler,
            max_retries=5,
            deadline_sec=0.01,
        )
        with pytest.raises(NetworkError):
            engine.transcribe(speech())
        assert len(handler.calls) == 1  # type: ignore[attr-defined]

    def test_the_backoff_grows_and_stays_under_the_cap(self) -> None:
        engine = YandexSttEngine()
        delays = [engine._backoff_delay(attempt) for attempt in range(1, 8)]
        assert all(delay > 0.0 for delay in delays)
        assert delays[0] < 1.0
        assert max(delays) <= 4.0


class TestLogging:
    """«Готово когда»: the key is not in the log, and neither is the audio."""

    def test_the_authorization_header_is_masked_in_the_debug_log(
        self,
        keyring_store: SecretsStore,
        tmp_path: Path,
        ayris_log: pytest.LogCaptureFixture,
    ) -> None:
        recorder = Recorder(payload={"result": "привет"})
        engine = loaded(YandexSttEngine(), tmp_path, handler=recorder)
        engine.transcribe(speech())
        blob = "\n".join(record.getMessage() for record in ayris_log.records)
        assert KEY not in blob
        assert "***" in blob

    def test_the_load_line_masks_the_credential(
        self,
        keyring_store: SecretsStore,
        tmp_path: Path,
        ayris_log: pytest.LogCaptureFixture,
    ) -> None:
        loaded(YandexSttEngine(), tmp_path, handler=Recorder())
        blob = "\n".join(record.getMessage() for record in ayris_log.records)
        assert KEY not in blob

    def test_the_audio_body_is_not_logged(
        self,
        keyring_store: SecretsStore,
        tmp_path: Path,
        ayris_log: pytest.LogCaptureFixture,
    ) -> None:
        audio = speech(ms=200, amplitude=12000)
        engine = loaded(YandexSttEngine(), tmp_path, handler=Recorder(payload={"result": "да"}))
        engine.transcribe(audio)
        blob = "\n".join(record.getMessage() for record in ayris_log.records)
        assert audio.pcm[:32].hex() not in blob
        assert "bytes of audio" in blob


class TestRegistry:
    """The four provider names resolve, and nothing else does."""

    def test_all_four_providers_are_offered(self) -> None:
        assert set(cloud_engine_names()) == {"yandex", "google", "azure", "openai"}

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("yandex", YandexSttEngine),
            ("google", GoogleSttEngine),
            ("azure", AzureSttEngine),
            ("openai", OpenAiSttEngine),
        ],
    )
    def test_a_name_resolves_to_its_class(self, name: str, expected: type[CloudSttEngine]) -> None:
        engine = create_cloud_engine(name)
        assert isinstance(engine, expected)
        assert engine.name == name
        assert engine.title

    def test_an_unknown_provider_is_refused_by_name(self) -> None:
        with pytest.raises(SttError, match="unknown cloud stt engine"):
            create_cloud_engine("nuance")

    def test_as_wav_wraps_pcm_in_a_readable_container(self) -> None:
        import wave

        audio = speech(ms=300)
        data = as_wav(audio)
        assert data[:4] == b"RIFF"
        import io

        with wave.open(io.BytesIO(data), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getnframes() == len(audio.pcm) // 2
