"""Яндекс SpeechKit: the one that sounds right in Russian and costs the least.

SpeechKit is form-encoded rather than JSON, which is the only structural
surprise here, and it wants a folder id alongside the key: the key says who is
paying and the folder says which project is billed. Both come out of the
settings - the key from the credential store, the folder from ``folder_id`` in
the extras - and a missing folder is an error the service returns rather than
something this module guesses at, because guessing would bill the wrong project.

``format=lpcm`` gives headerless ``int16`` at the rate asked for. SpeechKit only
offers 8000, 16000 and 48000, so the cloud default of 24 kHz is not available
and 48 kHz is used instead - the highest of the three, since the player
resamples once per device and downsampling later is cheaper than inventing
detail that was never synthesized.

Speed is ``speed``, 0.1–3.0, wide enough that Ayris's 0.5–2.0 maps onto it
without compression. Pitch has no equivalent: SpeechKit's voices are fixed, and
the ``emotion`` parameter changes delivery rather than frequency. The slider is
therefore ignored, as it is on Piper and ElevenLabs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final
from urllib.parse import urlencode

from ayris.audio.tts.base import VoiceSpec
from ayris.audio.tts.cloud_base import AudioFormat, CloudRequest, CloudTtsEngine

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["YandexTtsEngine"]

#: Rate to ask for. One of 8000/16000/48000; the top one, because the extra
#: bandwidth is free and the quality is not recoverable afterwards.
_SAMPLE_RATE: Final = 48000

#: What ``speed`` accepts.
_SPEED_LIMITS: Final = (0.1, 3.0)

#: Voice used when the settings name none. Alena is the Russian voice SpeechKit
#: documents first and the only one available in every region.
_DEFAULT_VOICE: Final = "alena"

#: Voices SpeechKit offers in Russian, as a table rather than an API call: the
#: list is documented, does not change between deployments, and there is no
#: endpoint that returns it.
_VOICES: Final = (
    ("alena", "Алёна"),
    ("filipp", "Филипп"),
    ("ermil", "Ермил"),
    ("jane", "Джейн"),
    ("madirus", "Мадирус"),
    ("omazh", "Омаж"),
    ("zahar", "Захар"),
    ("dasha", "Даша"),
    ("julia", "Юлия"),
    ("lera", "Лера"),
    ("masha", "Маша"),
    ("marina", "Марина"),
    ("alexander", "Александр"),
    ("kirill", "Кирилл"),
    ("anton", "Антон"),
)


class YandexTtsEngine(CloudTtsEngine):
    """SpeechKit synthesis over HTTPS."""

    name: ClassVar[str] = "yandex"
    title: ClassVar[str] = "Яндекс SpeechKit"
    default_ref: ClassVar[str] = "yandex"
    default_endpoint: ClassVar[str] = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    native_sample_rate: ClassVar[int] = _SAMPLE_RATE
    speed_limits: ClassVar[tuple[float, float]] = _SPEED_LIMITS

    __slots__ = ()

    @classmethod
    def voices(cls, directory: Path | None = None) -> tuple[VoiceSpec, ...]:
        """The documented Russian voices. No network, no key, no cache."""
        del directory
        return tuple(
            VoiceSpec(
                engine=cls.name,
                voice_id=voice_id,
                language="ru",
                display_name=label,
                sample_rate=_SAMPLE_RATE,
            )
            for voice_id, label in _VOICES
        )

    def _build_request(
        self,
        text: str,
        speed: float,
        pitch: float,
        *,
        stream: bool,
    ) -> CloudRequest:
        """POST a form body. ``pitch`` and ``stream`` have no counterpart here."""
        del pitch, stream
        voice = self._require_loaded()
        fields: dict[str, str] = {
            "text": text,
            "lang": self._options.option("language", "ru-RU"),
            "voice": voice.voice_id or _DEFAULT_VOICE,
            "format": "lpcm",
            "sampleRateHertz": str(_SAMPLE_RATE),
            "speed": f"{self._speed_for_provider(speed):.2f}",
        }
        emotion = self._options.option("emotion")
        if emotion:
            # ``neutral``, ``good`` or ``evil``, and only some voices accept it.
            # Sent only when asked for, so a voice that does not support it is
            # not refused over a parameter nobody set.
            fields["emotion"] = emotion
        folder = self._options.option("folder_id")
        if folder:
            fields["folderId"] = folder
        return CloudRequest(
            url=self._endpoint(),
            headers={
                "Authorization": self._authorization(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body=urlencode(fields).encode("utf-8"),
            audio_format=AudioFormat.PCM,
            sample_rate=_SAMPLE_RATE,
        )

    def _authorization(self) -> str:
        """``Api-Key`` for a service-account key, ``Bearer`` for an IAM token.

        Told apart by shape rather than by a setting: an IAM token starts with
        ``t1.`` and expires in twelve hours, an API key does not. A user who
        pasted one where the other was expected still gets a working request.
        """
        credential = self._credential
        scheme = "Bearer" if credential.startswith("t1.") else "Api-Key"
        return f"{scheme} {credential}"
