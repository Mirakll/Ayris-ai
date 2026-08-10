"""Task 13: the four cloud voices, on mocked HTTP.

No socket is opened here. ``httpx.MockTransport`` is handed to the engine through
``TtsOptions.extra["transport"]``, which is the seam
:class:`~ayris.audio.tts.cloud_base.CloudTtsEngine` builds its client around, so
the request a provider *would* have sent is inspected as an object instead of
being guessed at. That makes the strict assertions possible: "the speed ended up
inside ElevenLabs' 0.7–1.2 window", "the key ended up in ``Ocp-Apim-Subscription-Key``",
"the SSML escaped the ampersand".

What a mock cannot check is whether the real service agrees with our idea of its
request format. The task tracks that separately as a manual step with a real key;
everything mechanically checkable is here.

Groups:

* :class:`TestScales` — the arithmetic that turns 0.5–2.0 into provider units.
* :class:`TestDecoding` — PCM through, RIFF unwrapped, odd bytes trimmed.
* :class:`TestCredentials` — the key comes from keyring, by reference.
* :class:`TestElevenLabs` / :class:`TestYandex` / :class:`TestGoogle` /
  :class:`TestAzure` — the request each builds and the answer each parses.
* :class:`TestStreaming` — audio arriving in blocks before the body is complete.
* :class:`TestHttpErrors` — 401/402/429/5xx become different exception types.
* :class:`TestRetries` — what is retried, what is not.
* :class:`TestUsage` — characters counted, and counted once.
* :class:`TestVoiceList` — cached, and a failure is not fatal.
* :class:`TestLogging` — no key and no spoken text in the log.
* :class:`TestRegistry` — the four names resolve to the four classes.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import wave
from array import array
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import parse_qs

import httpx
import pytest

from ayris.audio.tts.azure_tts_engine import AzureTtsEngine
from ayris.audio.tts.base import (
    MAX_PITCH,
    MAX_SPEED,
    MIN_PITCH,
    MIN_SPEED,
    SAMPLE_WIDTH,
    TtsOptions,
    VoiceSpec,
)
from ayris.audio.tts.cloud_base import (
    CLOUD_ENGINE_ENTRYPOINTS,
    STREAM_CHUNK_BYTES,
    AudioFormat,
    CloudTtsEngine,
    TtsAuthError,
    TtsNetworkError,
    TtsQuotaError,
    cloud_engine_names,
    create_cloud_engine,
    decibels,
    decode_audio,
    is_cloud_engine,
    map_range,
    percent_offset,
    scrub_headers,
    semitones,
)
from ayris.audio.tts.elevenlabs_engine import ElevenLabsTtsEngine
from ayris.audio.tts.google_tts_engine import GoogleTtsEngine
from ayris.audio.tts.yandex_tts_engine import YandexTtsEngine
from ayris.core.errors import TtsError
from ayris.core.secrets import SecretsStore, mask, reset_secrets
from ayris.utils.logger import ROOT_LOGGER_NAME

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

#: The key every test stores, so an assertion can look for it by value.
KEY = "test-secret-key"

#: Text short enough to read in a failure message, long enough to count.
PHRASE = "Привет, мир."


class FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager.

    The sandbox has no keyring backend at all, so without this the engines could
    not even be constructed here. Duplicated from ``test_stt_online.py`` on
    purpose: a shared fixture would make the two modules fail together.
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

    ``setup_logging`` sets ``propagate = False`` on that logger and ``caplog``
    listens on the interpreter root, so a plain ``caplog.at_level`` sees nothing
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


# ----------------------------------------------------------------------
# audio helpers
# ----------------------------------------------------------------------


def pcm(frames: int = 480, level: int = 6000) -> bytes:
    """``frames`` of a constant int16 level: audio whose peak is a number."""
    return array("h", [level] * frames).tobytes()


def wav(data: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    """Wrap PCM in a RIFF container, the way Google answers."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(data)
    return buffer.getvalue()


class Recorder:
    """A :class:`httpx.MockTransport` handler that keeps what it was asked.

    Every provider test needs the same two things - the requests as objects and a
    canned answer - and a class makes "the second request went to /stream" an
    index rather than a closure over a list.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
        responses: list[tuple[int, bytes]] | None = None,
        error: Exception | None = None,
        error_times: int = 0,
    ) -> None:
        self.status = status
        self.content = content
        self.headers = headers or {}
        #: Consumed in order when set, so a retry can be answered differently.
        self.responses = list(responses or [])
        self.error = error
        #: How many opening calls raise ``error`` before the canned answer.
        self.error_times = error_times
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None and len(self.requests) <= self.error_times:
            raise self.error
        if self.responses:
            status, content = self.responses.pop(0)
            return httpx.Response(status, content=content, headers=self.headers)
        return httpx.Response(self.status, content=self.content, headers=self.headers)

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "движок не отправил ни одного запроса"
        return self.requests[-1]

    @property
    def first(self) -> httpx.Request:
        assert self.requests, "движок не отправил ни одного запроса"
        return self.requests[0]

    @property
    def calls(self) -> int:
        return len(self.requests)

    def json_body(self, index: int = -1) -> dict[str, Any]:
        """The request body parsed as a JSON object."""
        payload = json.loads(self.requests[index].content.decode("utf-8"))
        assert isinstance(payload, dict)
        return payload

    def form(self, index: int = -1) -> dict[str, str]:
        """The request body parsed as a urlencoded form, single values."""
        raw = parse_qs(self.requests[index].content.decode("utf-8"))
        return {key: values[0] for key, values in raw.items()}


def options(recorder: Recorder, **extra: object) -> TtsOptions:
    """Options wiring ``recorder`` in as the transport, plus test-fast retries."""
    payload: dict[str, object] = {
        "transport": httpx.MockTransport(recorder),
        "max_retries": 0,
        **extra,
    }
    return TtsOptions(extra=payload)


def load(
    engine: CloudTtsEngine,
    recorder: Recorder,
    *,
    voice_id: str = "",
    speed: float = 1.0,
    pitch: float = 1.0,
    **extra: object,
) -> CloudTtsEngine:
    """Load ``engine`` against ``recorder``. Returns it, so calls can chain."""
    engine.load(
        VoiceSpec(engine=engine.name, voice_id=voice_id),
        TtsOptions(speed=speed, pitch=pitch, extra=options(recorder, **extra).extra),
    )
    return engine


# ----------------------------------------------------------------------
# the scales
# ----------------------------------------------------------------------


class TestScales:
    """Ayris's 0.5–2.0 onto four different provider ranges.

    These are the conversions a listener notices instantly and a test notices
    never, unless it checks the endpoints: a pitch mapped linearly rather than
    logarithmically is an octave out, and speed 1.0 landing anywhere but the
    provider's own neutral makes every unmodified phrase drawl.
    """

    def test_neutral_stays_neutral(self) -> None:
        """1.0 maps onto 1.0 in every provider's range that contains it."""
        for limits in ((0.7, 1.2), (0.1, 3.0), (0.25, 4.0), (0.5, 2.0)):
            assert map_range(1.0, (MIN_SPEED, MAX_SPEED), limits) == pytest.approx(1.0)

    def test_endpoints_reach_the_providers_endpoints(self) -> None:
        """0.5 and 2.0 reach the provider's own minimum and maximum."""
        limits = (0.7, 1.2)
        assert map_range(MIN_SPEED, (MIN_SPEED, MAX_SPEED), limits) == pytest.approx(0.7)
        assert map_range(MAX_SPEED, (MIN_SPEED, MAX_SPEED), limits) == pytest.approx(1.2)

    def test_halves_are_mapped_separately(self) -> None:
        """Below neutral compresses into the lower half, not across the range.

        A plain end-to-end interpolation would put 1.0 at 0.75 of ElevenLabs'
        window - this is the assertion that catches that regression.
        """
        limits = (0.7, 1.2)
        slow = map_range(0.75, (MIN_SPEED, MAX_SPEED), limits)
        assert 0.7 < slow < 1.0
        assert slow == pytest.approx(0.85)

    def test_out_of_range_is_clamped(self) -> None:
        """A value past the Ayris scale never leaves the provider's range."""
        limits = (0.1, 3.0)
        assert map_range(9.0, (MIN_SPEED, MAX_SPEED), limits) == pytest.approx(3.0)
        assert map_range(-1.0, (MIN_SPEED, MAX_SPEED), limits) == pytest.approx(0.1)

    def test_semitones_are_a_real_frequency_ratio(self) -> None:
        """Google's pitch: doubling is +12 semitones, halving is -12."""
        limits = (-20.0, 20.0)
        assert semitones(2.0, limits) == pytest.approx(12.0)
        assert semitones(1.0, limits) == pytest.approx(0.0)
        assert semitones(0.5, limits) == pytest.approx(-12.0)

    def test_semitones_are_clamped_and_survive_zero(self) -> None:
        """A nonsense multiplier gives the floor rather than a math domain error."""
        limits = (-20.0, 20.0)
        assert semitones(0.0, limits) == pytest.approx(-20.0)
        assert semitones(1000.0, limits) == pytest.approx(20.0)

    def test_percent_offset_is_what_ssml_writes(self) -> None:
        """Azure's prosody: 1.0 is 0%, 1.5 is +50%, 0.5 is -50%."""
        limits = (-50.0, 100.0)
        assert percent_offset(1.0, limits) == 0
        assert percent_offset(1.5, limits) == 50
        assert percent_offset(0.5, limits) == -50

    def test_percent_offset_clamps_to_the_documented_range(self) -> None:
        """2.0 is +100% for rate but only +50% for pitch, as Azure documents."""
        assert percent_offset(2.0, (-50.0, 100.0)) == 100
        assert percent_offset(2.0, (-50.0, 50.0)) == 50

    def test_decibels_are_logarithmic(self) -> None:
        """Volume 100 is unchanged; half the slider is about -6 dB, not -50."""
        limits = (-96.0, 16.0)
        assert decibels(100, limits) == pytest.approx(0.0)
        assert decibels(50, limits) == pytest.approx(20.0 * math.log10(0.5))
        assert decibels(0, limits) == pytest.approx(-96.0)

    def test_pitch_scale_uses_the_pitch_limits(self) -> None:
        """The base helper reads :attr:`pitch_limits`, not the speed ones."""
        engine = ElevenLabsTtsEngine()
        assert engine._pitch_for_provider(MAX_PITCH) == pytest.approx(engine.pitch_limits[1])
        assert engine._pitch_for_provider(MIN_PITCH) == pytest.approx(engine.pitch_limits[0])


# ----------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------


class TestDecoding:
    """Four containers, one thing the player understands."""

    def test_pcm_passes_through(self) -> None:
        data = pcm(100)
        chunk = decode_audio(data, AudioFormat.PCM, 24000)
        assert chunk.pcm == data
        assert chunk.sample_rate == 24000
        assert chunk.channels == 1

    def test_odd_byte_count_is_trimmed(self) -> None:
        """A truncated frame would shift every sample after it into noise."""
        chunk = decode_audio(pcm(100) + b"\x01", AudioFormat.PCM, 24000)
        assert len(chunk.pcm) % SAMPLE_WIDTH == 0
        assert len(chunk.pcm) == 200

    def test_empty_body_is_an_empty_chunk(self) -> None:
        """200 with no audio is odd but not fatal: the router says it locally."""
        chunk = decode_audio(b"", AudioFormat.PCM, 24000)
        assert chunk.empty
        assert chunk.sample_rate == 24000

    def test_wav_is_unwrapped_and_keeps_its_own_rate(self) -> None:
        """The header wins over the requested rate: it is what was synthesized."""
        data = pcm(240)
        chunk = decode_audio(wav(data, rate=16000), AudioFormat.WAV, 24000)
        assert chunk.pcm == data
        assert chunk.sample_rate == 16000

    def test_riff_is_detected_even_when_pcm_was_declared(self) -> None:
        """A provider that ignores the format parameter still plays."""
        data = pcm(120)
        chunk = decode_audio(wav(data), AudioFormat.WAV, 24000)
        assert chunk.pcm == data

    def test_eight_bit_wav_is_refused_with_a_sentence(self) -> None:
        """Converting sample widths by hand would hide a wrongly built request."""
        with pytest.raises(TtsError) as info:
            decode_audio(wav(b"\x40" * 100, width=1), AudioFormat.WAV, 24000, provider="Тест")
        assert "8-bit" in info.value.technical
        assert info.value.user_message

    def test_broken_wav_names_the_provider(self) -> None:
        with pytest.raises(TtsError) as info:
            decode_audio(b"RIFFxxxxWAVEjunk", AudioFormat.WAV, 24000, provider="Google")
        assert "Google" in info.value.technical

    def test_compressed_without_a_decoder_says_what_to_install(self) -> None:
        """Reaching this means a provider ignored the format we asked for."""
        try:
            import soundfile  # noqa: F401
        except ImportError:
            pass
        else:  # pragma: no cover - soundfile is not a dependency
            pytest.skip("soundfile установлен, ветка недостижима")
        with pytest.raises(TtsError) as info:
            decode_audio(b"\xff\xfb\x90\x00mp3", AudioFormat.COMPRESSED, 24000, provider="Тест")
        assert "soundfile" in info.value.user_message


# ----------------------------------------------------------------------
# credentials
# ----------------------------------------------------------------------


class TestCredentials:
    """Where the key comes from, and what happens when it is not there."""

    def test_key_is_read_from_the_store_by_default_ref(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder)
        engine.synthesize(PHRASE)
        assert recorder.last.headers["xi-api-key"] == KEY

    def test_configured_ref_wins_over_the_default(self, keyring_store: SecretsStore) -> None:
        """A user with two accounts names the entry; the default is a fallback."""
        keyring_store.save("second-account", "other-key")
        recorder = Recorder(content=pcm())
        engine = ElevenLabsTtsEngine()
        engine.load(
            VoiceSpec(engine="elevenlabs", voice_id="v"),
            options(recorder, credential_ref="second-account"),
        )
        engine.synthesize(PHRASE)
        assert recorder.last.headers["xi-api-key"] == "other-key"

    def test_explicit_credential_skips_the_store(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = ElevenLabsTtsEngine()
        spec = VoiceSpec(engine="elevenlabs", voice_id="v")
        engine.load(spec, options(recorder, credential=" k "))
        engine.synthesize(PHRASE)
        assert recorder.last.headers["xi-api-key"] == "k"

    def test_missing_key_fails_at_load_not_mid_phrase(self) -> None:
        """The router must learn there is no key before promising any sound."""
        reset_secrets(SecretsStore("Ayris-empty", backend=FakeKeyring()))
        try:
            recorder = Recorder(content=pcm())
            with pytest.raises(TtsAuthError) as info:
                load(ElevenLabsTtsEngine(), recorder)
        finally:
            reset_secrets()
        assert recorder.calls == 0
        assert "ElevenLabs" in info.value.user_message

    def test_unload_forgets_the_key(self, keyring_store: SecretsStore) -> None:
        engine = load(ElevenLabsTtsEngine(), Recorder(content=pcm()))
        engine.unload()
        assert engine._credential == ""
        assert not engine.loaded

    def test_unload_is_safe_twice(self, keyring_store: SecretsStore) -> None:
        engine = load(YandexTtsEngine(), Recorder(content=pcm()))
        engine.unload()
        engine.unload()


# ----------------------------------------------------------------------
# per-provider requests and answers
# ----------------------------------------------------------------------


class TestElevenLabs:
    """The fussiest speed range and the only cloned-voice list."""

    def test_audio_is_parsed_from_headerless_pcm(self, keyring_store: SecretsStore) -> None:
        data = pcm(480)
        engine = load(ElevenLabsTtsEngine(), Recorder(content=data))
        chunk = engine.synthesize(PHRASE)
        assert chunk.pcm == data
        assert chunk.sample_rate == 24000

    def test_request_asks_for_pcm_at_the_native_rate(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder, voice_id="voice-42")
        engine.synthesize(PHRASE)
        url = str(recorder.last.url)
        assert "/text-to-speech/voice-42" in url
        assert "output_format=pcm_24000" in url
        assert recorder.last.method == "POST"

    def test_text_and_model_are_in_the_body(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder, model="eleven_multilingual_v2")
        engine.synthesize(PHRASE)
        body = recorder.json_body()
        assert body["text"] == PHRASE
        assert body["model_id"] == "eleven_multilingual_v2"

    @pytest.mark.parametrize(
        ("speed", "expected"),
        [(MIN_SPEED, 0.7), (1.0, 1.0), (MAX_SPEED, 1.2), (0.75, 0.85)],
    )
    def test_speed_lands_inside_the_documented_window(
        self, keyring_store: SecretsStore, speed: float, expected: float
    ) -> None:
        """0.5–2.0 compressed into 0.7–1.2, which is all this API accepts."""
        recorder = Recorder(content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder)
        engine.synthesize(PHRASE, speed=speed)
        settings = recorder.json_body()["voice_settings"]
        assert isinstance(settings, dict)
        assert settings["speed"] == pytest.approx(expected, abs=0.01)

    def test_pitch_is_accepted_and_dropped(self, keyring_store: SecretsStore) -> None:
        """There is no pitch field in this API; the slider must not break it."""
        recorder = Recorder(content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder)
        engine.synthesize(PHRASE, pitch=1.8)
        assert "pitch" not in json.dumps(recorder.json_body())


class TestYandex:
    """A form body, a 48 kHz answer, and two shapes of credential."""

    def test_form_carries_text_voice_and_format(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(YandexTtsEngine(), recorder, voice_id="filipp")
        engine.synthesize(PHRASE)
        fields = recorder.form()
        assert fields["text"] == PHRASE
        assert fields["voice"] == "filipp"
        assert fields["format"] == "lpcm"
        assert fields["sampleRateHertz"] == "48000"

    def test_answer_is_pcm_at_48_khz(self, keyring_store: SecretsStore) -> None:
        data = pcm(960)
        engine = load(YandexTtsEngine(), Recorder(content=data))
        chunk = engine.synthesize(PHRASE)
        assert chunk.pcm == data
        assert chunk.sample_rate == 48000

    def test_api_key_and_iam_token_get_different_schemes(self, keyring_store: SecretsStore) -> None:
        """Told apart by shape, so a pasted IAM token still builds a valid request."""
        recorder = Recorder(content=pcm())
        engine = YandexTtsEngine()
        spec = VoiceSpec(engine="yandex", voice_id="alena")
        engine.load(spec, options(recorder, credential="t1.abc"))
        engine.synthesize(PHRASE)
        assert recorder.last.headers["Authorization"] == "Bearer t1.abc"

        other = load(YandexTtsEngine(), recorder)
        other.synthesize(PHRASE)
        assert recorder.last.headers["Authorization"] == f"Api-Key {KEY}"

    def test_folder_id_and_emotion_are_optional(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(YandexTtsEngine(), recorder)
        engine.synthesize(PHRASE)
        assert "folderId" not in recorder.form()

        with_folder = load(YandexTtsEngine(), recorder, folder_id="b1g", emotion="good")
        with_folder.synthesize(PHRASE)
        fields = recorder.form()
        assert fields["folderId"] == "b1g"
        assert fields["emotion"] == "good"

    @pytest.mark.parametrize(
        ("speed", "expected"), [(MIN_SPEED, 0.1), (1.0, 1.0), (MAX_SPEED, 3.0)]
    )
    def test_speed_uses_the_full_speechkit_range(
        self, keyring_store: SecretsStore, speed: float, expected: float
    ) -> None:
        recorder = Recorder(content=pcm())
        engine = load(YandexTtsEngine(), recorder)
        engine.synthesize(PHRASE, speed=speed)
        assert float(recorder.form()["speed"]) == pytest.approx(expected, abs=0.01)

    def test_documented_voices_need_no_network(self) -> None:
        voices = YandexTtsEngine.voices()
        assert voices
        assert all(spec.language == "ru" for spec in voices)
        assert any(spec.voice_id == "alena" for spec in voices)


class TestGoogle:
    """base64 inside JSON, a RIFF container inside that."""

    def test_audio_content_is_base64_decoded_then_unwrapped(
        self, keyring_store: SecretsStore
    ) -> None:
        data = pcm(240)
        payload = json.dumps({"audioContent": base64.b64encode(wav(data)).decode()}).encode()
        engine = load(GoogleTtsEngine(), Recorder(content=payload))
        chunk = engine.synthesize(PHRASE)
        assert chunk.pcm == data

    def test_request_is_the_documented_synthesize_body(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(
            content=json.dumps({"audioContent": base64.b64encode(wav(pcm())).decode()}).encode()
        )
        engine = load(GoogleTtsEngine(), recorder, voice_id="ru-RU-Wavenet-C")
        engine.synthesize(PHRASE)
        body = recorder.json_body()
        assert body["input"] == {"text": PHRASE}
        voice = body["voice"]
        assert isinstance(voice, dict)
        assert voice["languageCode"] == "ru-RU"
        assert voice["name"] == "ru-RU-Wavenet-C"
        assert body["audioConfig"]["audioEncoding"] == "LINEAR16"  # type: ignore[index]
        assert recorder.last.headers["X-Goog-Api-Key"] == KEY

    def test_language_follows_the_voice_name(self, keyring_store: SecretsStore) -> None:
        """A user who picks an English voice gets English, not a 400."""
        recorder = Recorder(
            content=json.dumps({"audioContent": base64.b64encode(wav(pcm())).decode()}).encode()
        )
        engine = load(GoogleTtsEngine(), recorder, voice_id="en-US-Neural2-F")
        engine.synthesize(PHRASE)
        assert recorder.json_body()["voice"]["languageCode"] == "en-US"  # type: ignore[index]

    @pytest.mark.parametrize(("pitch", "expected"), [(2.0, 12.0), (1.0, 0.0), (0.5, -12.0)])
    def test_pitch_is_sent_in_semitones(
        self, keyring_store: SecretsStore, pitch: float, expected: float
    ) -> None:
        recorder = Recorder(
            content=json.dumps({"audioContent": base64.b64encode(wav(pcm())).decode()}).encode()
        )
        engine = load(GoogleTtsEngine(), recorder)
        engine.synthesize(PHRASE, pitch=pitch)
        config = recorder.json_body()["audioConfig"]
        assert isinstance(config, dict)
        assert config["pitch"] == pytest.approx(expected, abs=0.01)

    def test_volume_is_sent_in_decibels(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(
            content=json.dumps({"audioContent": base64.b64encode(wav(pcm())).decode()}).encode()
        )
        engine = load(GoogleTtsEngine(), recorder, volume=50)
        engine.synthesize(PHRASE)
        config = recorder.json_body()["audioConfig"]
        assert isinstance(config, dict)
        assert config["volumeGainDb"] == pytest.approx(20.0 * math.log10(0.5), abs=0.01)

    def test_missing_audio_field_is_a_clear_error(self, keyring_store: SecretsStore) -> None:
        """The API changed shape: say so rather than play zero bytes."""
        engine = load(GoogleTtsEngine(), Recorder(content=b'{"name":"operations/1"}'))
        with pytest.raises(TtsError) as info:
            engine.synthesize(PHRASE)
        assert "audioContent" in info.value.technical

    def test_non_json_answer_is_a_clear_error(self, keyring_store: SecretsStore) -> None:
        """A proxy's HTML error page must not reach the sound card."""
        engine = load(GoogleTtsEngine(), Recorder(content=b"<html>502</html>"))
        with pytest.raises(TtsError) as info:
            engine.synthesize(PHRASE)
        assert "not JSON" in info.value.technical


class TestAzure:
    """SSML, a region in the host, and escaping that keeps 400s away."""

    def test_ssml_carries_voice_rate_and_pitch(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(
            AzureTtsEngine(), recorder, voice_id="ru-RU-DmitryNeural", region="westeurope"
        )
        engine.synthesize(PHRASE, speed=1.5, pitch=1.5)
        body = recorder.last.content.decode("utf-8")
        assert '<voice name="ru-RU-DmitryNeural">' in body
        assert 'rate="+50%"' in body
        assert 'pitch="+50%"' in body
        assert PHRASE in body

    def test_pitch_clamps_tighter_than_rate(self, keyring_store: SecretsStore) -> None:
        """Azure documents ±50% for pitch but -50…+100% for rate."""
        recorder = Recorder(content=pcm())
        engine = load(AzureTtsEngine(), recorder, region="westeurope")
        engine.synthesize(PHRASE, speed=MAX_SPEED, pitch=MAX_PITCH)
        body = recorder.last.content.decode("utf-8")
        assert 'rate="+100%"' in body
        assert 'pitch="+50%"' in body

    def test_ampersand_is_escaped(self, keyring_store: SecretsStore) -> None:
        """«чай & кофе» is the difference between speech and a 400."""
        recorder = Recorder(content=pcm())
        engine = load(AzureTtsEngine(), recorder, region="westeurope")
        engine.synthesize("Чай & кофе.")
        body = recorder.last.content.decode("utf-8")
        assert "&amp;" in body
        assert "чай & кофе" not in body.lower()

    def test_region_becomes_the_host(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(AzureTtsEngine(), recorder, region="northeurope")
        engine.synthesize(PHRASE)
        assert str(recorder.last.url).startswith("https://northeurope.tts.speech.microsoft.com/")
        assert recorder.last.headers["Ocp-Apim-Subscription-Key"] == KEY
        assert recorder.last.headers["X-Microsoft-OutputFormat"] == "raw-24khz-16bit-mono-pcm"

    def test_missing_region_reads_as_a_settings_problem(self, keyring_store: SecretsStore) -> None:
        """It is one screen and one thought for the user: key plus region."""
        recorder = Recorder(content=pcm())
        engine = load(AzureTtsEngine(), recorder)
        with pytest.raises(TtsAuthError) as info:
            engine.synthesize(PHRASE)
        assert "регион" in info.value.user_message.lower()
        assert recorder.calls == 0

    def test_voice_list_keeps_only_neural_voices(self, keyring_store: SecretsStore) -> None:
        """The retired standard voices synthesize and sound a decade old."""
        listing = json.dumps(
            [
                {"ShortName": "ru-RU-SvetlanaNeural", "VoiceType": "Neural", "Locale": "ru-RU"},
                {"ShortName": "ru-RU-Irina", "VoiceType": "Standard", "Locale": "ru-RU"},
                {"NoShortName": True},
            ]
        ).encode()
        engine = load(AzureTtsEngine(), Recorder(content=listing), region="westeurope")
        voices = engine.list_voices()
        assert [spec.voice_id for spec in voices] == ["ru-RU-SvetlanaNeural"]


# ----------------------------------------------------------------------
# streaming
# ----------------------------------------------------------------------


class TestStreaming:
    """Audio handed on while the rest is still arriving."""

    def test_elevenlabs_streams_from_the_stream_endpoint(self, keyring_store: SecretsStore) -> None:
        data = pcm(STREAM_CHUNK_BYTES)  # several blocks' worth
        recorder = Recorder(content=data)
        engine = load(ElevenLabsTtsEngine(), recorder, voice_id="v1")
        chunks = list(engine.synthesize_stream(PHRASE))
        assert len(chunks) > 1, "поток пришёл одним куском - стриминг не задействован"
        assert b"".join(chunk.pcm for chunk in chunks) == data
        assert "/stream" in str(recorder.first.url)

    def test_azure_streams_from_the_same_endpoint(self, keyring_store: SecretsStore) -> None:
        data = pcm(STREAM_CHUNK_BYTES)
        recorder = Recorder(content=data)
        engine = load(AzureTtsEngine(), recorder, region="westeurope")
        chunks = list(engine.synthesize_stream(PHRASE))
        assert b"".join(chunk.pcm for chunk in chunks) == data
        assert all(chunk.sample_rate == 24000 for chunk in chunks)

    def test_stream_chunks_never_split_a_frame(self, keyring_store: SecretsStore) -> None:
        """An odd split would shift every following sample into white noise."""
        recorder = Recorder(content=pcm(STREAM_CHUNK_BYTES) + b"\x00")
        engine = load(ElevenLabsTtsEngine(), recorder)
        for chunk in engine.synthesize_stream(PHRASE):
            assert len(chunk.pcm) % SAMPLE_WIDTH == 0

    def test_non_streaming_provider_falls_back_to_per_sentence(
        self, keyring_store: SecretsStore
    ) -> None:
        """Google has no streaming endpoint; the base class splits instead."""
        recorder = Recorder(
            content=json.dumps({"audioContent": base64.b64encode(wav(pcm())).decode()}).encode()
        )
        engine = load(GoogleTtsEngine(), recorder)
        chunks = list(engine.synthesize_stream("Первое предложение. Второе предложение."))
        assert len(chunks) == 2
        assert recorder.calls == 2

    def test_stream_failure_is_not_retried(self, keyring_store: SecretsStore) -> None:
        """A retry would replay a phrase the player is already speaking."""
        recorder = Recorder(error=httpx.ConnectError("сеть пропала"), error_times=5, content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder, max_retries=3)
        with pytest.raises(TtsNetworkError):
            list(engine.synthesize_stream(PHRASE))
        assert recorder.calls == 1

    def test_stream_status_error_keeps_its_type(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(status=429, content=b"quota")
        engine = load(ElevenLabsTtsEngine(), recorder)
        with pytest.raises(TtsQuotaError):
            list(engine.synthesize_stream(PHRASE))

    def test_unspeakable_text_makes_no_request(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder)
        assert list(engine.synthesize_stream("   ...   ")) == []
        assert recorder.calls == 0


# ----------------------------------------------------------------------
# HTTP failures
# ----------------------------------------------------------------------


class TestHttpErrors:
    """Four statuses, four different things for the router to do."""

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_key_is_an_auth_error(self, keyring_store: SecretsStore, status: int) -> None:
        engine = load(ElevenLabsTtsEngine(), Recorder(status=status, content=b"nope"))
        with pytest.raises(TtsAuthError) as info:
            engine.synthesize(PHRASE)
        assert "ключ" in info.value.user_message.lower()

    @pytest.mark.parametrize("status", [402, 429])
    def test_spent_budget_is_a_quota_error(self, keyring_store: SecretsStore, status: int) -> None:
        engine = load(YandexTtsEngine(), Recorder(status=status, content=b"limit"))
        with pytest.raises(TtsQuotaError) as info:
            engine.synthesize(PHRASE)
        assert "лимит" in info.value.user_message.lower()

    @pytest.mark.parametrize("status", [500, 503])
    def test_server_error_is_a_network_error(
        self, keyring_store: SecretsStore, status: int
    ) -> None:
        engine = load(GoogleTtsEngine(), Recorder(status=status, content=b"oops"))
        with pytest.raises(TtsNetworkError) as info:
            engine.synthesize(PHRASE)
        assert str(status) in info.value.user_message

    def test_other_4xx_is_a_plain_tts_error(self, keyring_store: SecretsStore) -> None:
        """400 means Ayris built the request wrong: not a fallback-worthy fact."""
        engine = load(AzureTtsEngine(), Recorder(status=400, content=b"bad ssml"), region="we")
        with pytest.raises(TtsError) as info:
            engine.synthesize(PHRASE)
        assert not isinstance(info.value, TtsNetworkError | TtsQuotaError | TtsAuthError)

    def test_timeout_becomes_a_network_error(self, keyring_store: SecretsStore) -> None:
        engine = load(
            ElevenLabsTtsEngine(),
            Recorder(error=httpx.ReadTimeout("медленно"), error_times=9, content=pcm()),
        )
        with pytest.raises(TtsNetworkError) as info:
            engine.synthesize(PHRASE)
        assert "интернет" in info.value.user_message.lower()

    def test_connect_error_becomes_a_network_error(self, keyring_store: SecretsStore) -> None:
        engine = load(
            YandexTtsEngine(),
            Recorder(error=httpx.ConnectError("нет сети"), error_times=9, content=pcm()),
        )
        with pytest.raises(TtsNetworkError):
            engine.synthesize(PHRASE)

    def test_error_body_reaches_the_technical_message_only(
        self, keyring_store: SecretsStore
    ) -> None:
        """The provider's own phrase is what makes a 400 debuggable."""
        engine = load(GoogleTtsEngine(), Recorder(status=400, content=b"voice not found"))
        with pytest.raises(TtsError) as info:
            engine.synthesize(PHRASE)
        assert "voice not found" in info.value.technical
        assert "voice not found" not in info.value.user_message


class TestRetries:
    """What is worth trying again, and what only wastes a second."""

    def test_server_error_is_retried_then_succeeds(self, keyring_store: SecretsStore) -> None:
        data = pcm()
        recorder = Recorder(responses=[(503, b"busy"), (200, data)])
        engine = load(ElevenLabsTtsEngine(), recorder, max_retries=2)
        assert engine.synthesize(PHRASE).pcm == data
        assert recorder.calls == 2

    def test_quota_is_never_retried(self, keyring_store: SecretsStore) -> None:
        """The wall does not move; a retry spends a second hitting it again."""
        recorder = Recorder(status=429, content=b"limit")
        engine = load(YandexTtsEngine(), recorder, max_retries=3)
        with pytest.raises(TtsQuotaError):
            engine.synthesize(PHRASE)
        assert recorder.calls == 1

    def test_auth_is_never_retried(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(status=401, content=b"bad key")
        engine = load(GoogleTtsEngine(), recorder, max_retries=3)
        with pytest.raises(TtsAuthError):
            engine.synthesize(PHRASE)
        assert recorder.calls == 1

    def test_retries_are_bounded_by_the_budget(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder(error=httpx.ConnectError("нет сети"), error_times=99, content=pcm())
        engine = load(ElevenLabsTtsEngine(), recorder, max_retries=2)
        with pytest.raises(TtsNetworkError):
            engine.synthesize(PHRASE)
        assert recorder.calls == 3

    def test_deadline_cuts_the_retries_short(self, keyring_store: SecretsStore) -> None:
        """Past the deadline the router is already speaking locally."""
        recorder = Recorder(error=httpx.ConnectError("нет сети"), error_times=99, content=pcm())
        engine = load(YandexTtsEngine(), recorder, max_retries=9, deadline_sec=0.001)
        with pytest.raises(TtsNetworkError):
            engine.synthesize(PHRASE)
        assert recorder.calls == 1

    def test_server_error_keeps_its_own_message_after_giving_up(
        self, keyring_store: SecretsStore
    ) -> None:
        """ "Check your connection" would send the user to a working router."""
        engine = load(GoogleTtsEngine(), Recorder(status=503, content=b"maintenance"))
        with pytest.raises(TtsNetworkError) as info:
            engine.synthesize(PHRASE)
        assert "503" in info.value.user_message


# ----------------------------------------------------------------------
# usage
# ----------------------------------------------------------------------


class Usage:
    """A list that satisfies :class:`~ayris.audio.tts.cloud_base.UsageRecorder`."""

    def __init__(self) -> None:
        self.records: list[tuple[str, int]] = []

    def record(self, provider: str, characters: int) -> None:
        self.records.append((provider, characters))

    @property
    def total(self) -> int:
        return sum(count for _, count in self.records)


class TestUsage:
    """Characters counted, and nothing else reported."""

    def test_characters_are_counted_per_provider(self, keyring_store: SecretsStore) -> None:
        usage = Usage()
        engine = load(ElevenLabsTtsEngine(), Recorder(content=pcm()), usage=usage)
        engine.synthesize(PHRASE)
        assert usage.records == [("elevenlabs", len(PHRASE))]

    def test_streaming_counts_once_per_request(self, keyring_store: SecretsStore) -> None:
        """Not once per block: the provider bills the request, not the chunks."""
        usage = Usage()
        engine = load(ElevenLabsTtsEngine(), Recorder(content=pcm(STREAM_CHUNK_BYTES)), usage=usage)
        list(engine.synthesize_stream(PHRASE))
        assert usage.records == [("elevenlabs", len(PHRASE))]

    def test_each_sentence_of_a_split_answer_is_counted(self, keyring_store: SecretsStore) -> None:
        usage = Usage()
        recorder = Recorder(
            content=json.dumps({"audioContent": base64.b64encode(wav(pcm())).decode()}).encode()
        )
        engine = load(GoogleTtsEngine(), recorder, usage=usage)
        parts = ("Первое предложение.", "Второе предложение.")
        list(engine.synthesize_stream(" ".join(parts)))
        assert len(usage.records) == 2
        assert usage.total == sum(len(part) for part in parts)

    def test_a_failed_request_is_not_counted(self, keyring_store: SecretsStore) -> None:
        usage = Usage()
        engine = load(YandexTtsEngine(), Recorder(status=500, content=b"x"), usage=usage)
        with pytest.raises(TtsNetworkError):
            engine.synthesize(PHRASE)
        assert usage.records == []

    def test_voice_list_is_not_billed_as_speech(self, keyring_store: SecretsStore) -> None:
        usage = Usage()
        engine = load(ElevenLabsTtsEngine(), Recorder(content=b'{"voices": []}'), usage=usage)
        engine.list_voices()
        assert usage.records == []

    def test_a_recorder_that_raises_does_not_swallow_audio(
        self, keyring_store: SecretsStore
    ) -> None:
        """The user already paid for that phrase; they should still hear it."""

        class Broken:
            def record(self, provider: str, characters: int) -> None:
                raise RuntimeError("база упала")

        data = pcm()
        engine = load(YandexTtsEngine(), Recorder(content=data), usage=Broken())
        assert engine.synthesize(PHRASE).pcm == data


# ----------------------------------------------------------------------
# voice list
# ----------------------------------------------------------------------


class TestVoiceList:
    """A billed request the settings window would otherwise make on every open."""

    def test_elevenlabs_voices_are_parsed(self, keyring_store: SecretsStore) -> None:
        listing = json.dumps(
            {
                "voices": [
                    {"voice_id": "a1", "name": "Аня", "labels": {"language": "ru"}},
                    {"name": "без id"},
                    {"voice_id": "b2", "name": "Bob", "labels": {"language": "en-US"}},
                ]
            }
        ).encode()
        engine = load(ElevenLabsTtsEngine(), Recorder(content=listing))
        voices = engine.list_voices()
        assert [spec.voice_id for spec in voices] == ["a1", "b2"]
        assert voices[1].language == "en"

    def test_second_call_is_served_from_the_cache(self, keyring_store: SecretsStore) -> None:
        listing = json.dumps({"voices": [{"voice_id": "a1", "name": "Аня"}]}).encode()
        recorder = Recorder(content=listing)
        engine = load(ElevenLabsTtsEngine(), recorder)
        engine.list_voices()
        engine.list_voices()
        assert recorder.calls == 1

    def test_refresh_asks_again(self, keyring_store: SecretsStore) -> None:
        listing = json.dumps({"voices": [{"voice_id": "a1", "name": "Аня"}]}).encode()
        recorder = Recorder(content=listing)
        engine = load(ElevenLabsTtsEngine(), recorder)
        engine.list_voices()
        engine.list_voices(refresh=True)
        assert recorder.calls == 2

    def test_a_failed_list_is_empty_rather_than_fatal(self, keyring_store: SecretsStore) -> None:
        """An error box because a combo box did not fill would be worse."""
        engine = load(ElevenLabsTtsEngine(), Recorder(status=500, content=b"down"))
        assert engine.list_voices() == ()

    def test_a_provider_without_a_list_endpoint_returns_nothing(
        self, keyring_store: SecretsStore
    ) -> None:
        recorder = Recorder(content=pcm())
        engine = load(YandexTtsEngine(), recorder)
        assert engine.list_voices() == ()
        assert recorder.calls == 0


# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------


class TestLogging:
    """The key and the spoken text stay out of the log."""

    def test_headers_are_scrubbed_by_name(self) -> None:
        scrubbed = scrub_headers(
            {
                "xi-api-key": KEY,
                "Authorization": f"Api-Key {KEY}",
                "Ocp-Apim-Subscription-Key": KEY,
                "X-Goog-Api-Key": KEY,
                "Content-Type": "application/json",
            }
        )
        assert scrubbed["Content-Type"] == "application/json"
        assert KEY not in json.dumps(scrubbed)

    def test_no_key_reaches_the_log_on_a_normal_request(
        self, keyring_store: SecretsStore, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        engine = load(ElevenLabsTtsEngine(), Recorder(content=pcm()))
        engine.synthesize(PHRASE)
        assert KEY not in ayris_log.text

    def test_no_key_reaches_the_log_on_a_failure(
        self, keyring_store: SecretsStore, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        engine = load(YandexTtsEngine(), Recorder(status=500, content=b"boom"))
        with pytest.raises(TtsNetworkError):
            engine.synthesize(PHRASE)
        assert KEY not in ayris_log.text

    def test_the_spoken_text_is_not_logged(
        self, keyring_store: SecretsStore, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """It is the user's, and a request body is where it would leak from."""
        secret_phrase = "Мой пароль от почты - земляника."
        engine = load(AzureTtsEngine(), Recorder(content=pcm()), region="westeurope")
        engine.synthesize(secret_phrase)
        assert "земляника" not in ayris_log.text

    def test_load_logs_the_key_masked(
        self, keyring_store: SecretsStore, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """Enough to tell two accounts apart, not enough to use."""
        load(ElevenLabsTtsEngine(), Recorder(content=pcm()))
        assert KEY not in ayris_log.text
        assert mask(KEY) in ayris_log.text
        assert ayris_log.text.count(KEY[:-4]) == 0


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


class TestRegistry:
    """Four names, four classes, and nothing silently substituted."""

    expected: ClassVar[dict[str, type[CloudTtsEngine]]] = {
        "elevenlabs": ElevenLabsTtsEngine,
        "yandex": YandexTtsEngine,
        "google": GoogleTtsEngine,
        "azure": AzureTtsEngine,
    }

    def test_names_are_the_four_providers(self) -> None:
        assert set(cloud_engine_names()) == set(self.expected)

    @pytest.mark.parametrize("name", sorted(expected))
    def test_each_name_builds_its_own_class(self, name: str) -> None:
        assert isinstance(create_cloud_engine(name), self.expected[name])

    def test_an_unknown_name_is_refused_not_substituted(self) -> None:
        """A user who configured Azure and got Yandex debugs the wrong account."""
        with pytest.raises(TtsError) as info:
            create_cloud_engine("polly")
        assert "polly" in info.value.technical

    def test_is_cloud_engine_tells_the_two_lists_apart(self) -> None:
        assert is_cloud_engine("azure")
        assert not is_cloud_engine("piper")

    def test_cloud_engines_declare_no_resident_memory(self) -> None:
        """Nothing is loaded on this machine, so the worker reserves nothing."""
        for name in cloud_engine_names():
            engine = create_cloud_engine(name)
            assert engine.memory_factor == 0.0
            assert engine.device == "cloud"

    def test_a_cloud_engine_needs_only_httpx(self) -> None:
        for name in cloud_engine_names():
            assert type(create_cloud_engine(name)).available()


def test_recorder_helper_is_not_used_by_accident(
    keyring_store: SecretsStore,
) -> None:
    """Guard for the fixture itself: the transport really is the mock.

    If ``extra["transport"]`` ever stopped being honoured, every test above would
    still pass while quietly opening real sockets - which the sandbox would fail
    on but a developer machine would not.
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=pcm())

    engine = ElevenLabsTtsEngine()
    engine.load(
        VoiceSpec(engine="elevenlabs", voice_id="v"),
        TtsOptions(extra={"transport": httpx.MockTransport(handler), "max_retries": 0}),
    )
    engine.synthesize(PHRASE)
    assert len(calls) == 1
