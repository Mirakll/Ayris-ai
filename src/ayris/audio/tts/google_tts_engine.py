"""Google Cloud Text-to-Speech: the only one that answers in JSON.

Everything about this provider follows from that. ``audioContent`` carries the
audio base64-encoded inside a JSON object, so there is no streaming to be had -
the body means nothing until the last byte of it has arrived - and
:class:`~ayris.audio.tts.cloud_base.CloudRequest` names the field so the base
class can unwrap it. Asking for ``LINEAR16`` gets a complete RIFF file rather
than raw samples, which is why the format is declared as
:attr:`~ayris.audio.tts.cloud_base.AudioFormat.WAV` and the rate is read out of
the header instead of being assumed.

It is also the only one of the four with a real pitch control, and it measures it
in semitones: :func:`~ayris.audio.tts.cloud_base.semitones` converts Ayris's
multiplier properly, so 2.0 is an octave up rather than an arbitrary number that
happened to be in range. Speed is ``speakingRate`` and spans 0.25–4.0, wide
enough that 0.5–2.0 passes through unchanged.

Authentication is an API key in ``X-Goog-Api-Key``. Service-account JSON and OAuth
are the other documented options and neither is offered here: both need a signing
step and a token cache, and the settings window has one field per provider.
"""

from __future__ import annotations

import json
from typing import ClassVar, Final

from ayris.audio.tts.cloud_base import (
    AudioFormat,
    CloudRequest,
    CloudTtsEngine,
    decibels,
    semitones,
)

__all__ = ["GoogleTtsEngine"]

#: What ``speakingRate`` accepts.
_SPEED_LIMITS: Final = (0.25, 4.0)

#: What ``pitch`` accepts, in semitones.
_PITCH_SEMITONES: Final = (-20.0, 20.0)

#: What ``volumeGainDb`` accepts.
_VOLUME_DB: Final = (-96.0, 16.0)

#: Voice used when the settings name none. A WaveNet Russian voice: Standard is
#: cheaper and audibly worse, and Neural2 has no Russian.
_DEFAULT_VOICE: Final = "ru-RU-Wavenet-C"


class GoogleTtsEngine(CloudTtsEngine):
    """Google Cloud synthesis over HTTPS."""

    name: ClassVar[str] = "google"
    title: ClassVar[str] = "Google Cloud TTS"
    default_ref: ClassVar[str] = "google"
    default_endpoint: ClassVar[str] = "https://texttospeech.googleapis.com/v1/text:synthesize"
    speed_limits: ClassVar[tuple[float, float]] = _SPEED_LIMITS

    __slots__ = ()

    def _build_request(
        self,
        text: str,
        speed: float,
        pitch: float,
        *,
        stream: bool,
    ) -> CloudRequest:
        """POST the documented ``text:synthesize`` body.

        ``stream`` is ignored: there is no streaming endpoint, and the base class
        falls back to a chunk per sentence, which is where the latency budget is
        actually met for this provider.
        """
        del stream
        voice = self._require_loaded()
        voice_id = voice.voice_id or _DEFAULT_VOICE
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": self._language_of(voice_id),
                "name": voice_id,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": self.native_sample_rate,
                "speakingRate": round(self._speed_for_provider(speed), 3),
                "pitch": round(semitones(pitch, _PITCH_SEMITONES), 2),
                "volumeGainDb": round(decibels(self._volume(), _VOLUME_DB), 2),
            },
        }
        return CloudRequest(
            url=self._endpoint(),
            headers={
                "X-Goog-Api-Key": self._credential,
                "Content-Type": "application/json; charset=utf-8",
            },
            body=json.dumps(payload).encode("utf-8"),
            # LINEAR16 arrives as a RIFF file, header included, whatever the
            # name suggests. The rate then comes from the header rather than
            # from what was asked for.
            audio_format=AudioFormat.WAV,
            sample_rate=self.native_sample_rate,
            json_field="audioContent",
        )

    def _language_of(self, voice_id: str) -> str:
        """The locale a voice belongs to.

        Google requires ``languageCode`` and requires it to agree with the voice
        name, which already begins with the locale - ``ru-RU-Wavenet-C``. Taking
        it from the name rather than from the settings means a user who picks an
        English voice gets English rather than a rejected request.
        """
        configured = self._options.option("language")
        if configured:
            return configured
        parts = voice_id.split("-")
        if len(parts) >= 2 and len(parts[0]) == 2:
            return f"{parts[0]}-{parts[1]}"
        return "ru-RU"

    def _volume(self) -> int:
        """The 0–100 volume from the extras, or 100 when nobody set one.

        Google is the only provider here whose gain is applied server-side and
        is worth using: it is applied before the codec rather than after, so a
        quiet setting does not lose bits the way scaling ``int16`` afterwards
        would.
        """
        raw = self._options.extra.get("volume")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return 100
        return max(0, min(100, int(raw)))
