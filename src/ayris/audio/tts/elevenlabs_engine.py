"""ElevenLabs: the best-sounding of the four, and the fussiest about speed.

Two things make this provider different from the other three. It has a real
streaming endpoint - ``/stream`` returns audio while it is still being generated,
which is what gets a long answer sounding inside the 1.5 s the specification
allows - and it has almost no speed range: ``voice_settings.speed`` is documented
as 0.7–1.2 and values outside it are rejected rather than clamped. Ayris's
0.5–2.0 is therefore compressed into that window by
:func:`~ayris.audio.tts.cloud_base.map_range`, which keeps 1.0 on 1.0; the user
who drags the slider to 2.0 gets 1.2 and the fastest speech this service will
produce, not an error.

There is no pitch control at all. The field stays in the API because the settings
window has one slider for every engine, and ignoring it here is the same choice
Piper already makes.

``output_format=pcm_24000`` asks for headerless ``int16``, so
:func:`~ayris.audio.tts.cloud_base.decode_audio` passes the bytes straight
through and the streaming path can hand blocks to the player as they land.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.audio.tts.base import VoiceSpec
from ayris.audio.tts.cloud_base import AudioFormat, CloudRequest, CloudTtsEngine

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

__all__ = ["ElevenLabsTtsEngine"]

#: Model used when nothing is configured. Turbo v2.5 because it is the one with
#: Russian support and the lowest latency of those that have it; ``eleven_v3``
#: sounds better and is slow enough to miss the budget on a long answer.
_DEFAULT_MODEL: Final = "eleven_turbo_v2_5"

#: A voice from the shared library, used when the settings name none. Rachel is
#: the account-independent default every ElevenLabs key can reach.
_DEFAULT_VOICE: Final = "21m00Tcm4TlvDq8ikWAM"

#: What ``voice_settings.speed`` accepts. Narrow, and enforced server-side.
_SPEED_LIMITS: Final = (0.7, 1.2)


class ElevenLabsTtsEngine(CloudTtsEngine):
    """ElevenLabs text-to-speech over HTTPS, streaming where it helps."""

    name: ClassVar[str] = "elevenlabs"
    title: ClassVar[str] = "ElevenLabs"
    default_ref: ClassVar[str] = "elevenlabs"
    default_endpoint: ClassVar[str] = "https://api.elevenlabs.io/v1"
    supports_streaming: ClassVar[bool] = True
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
        """POST the phrase to ``/text-to-speech/<voice>``.

        ``pitch`` is accepted and dropped: the API has no field for it.
        """
        del pitch
        voice = self._require_loaded()
        voice_id = voice.voice_id or _DEFAULT_VOICE
        suffix = "/stream" if stream else ""
        rate = self.native_sample_rate
        url = (
            f"{self._endpoint()}/text-to-speech/{voice_id}{suffix}"
            f"?output_format=pcm_{rate}&optimize_streaming_latency=2"
        )
        payload: JsonObject = {
            "text": text,
            "model_id": self._options.option("model", _DEFAULT_MODEL),
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": round(self._speed_for_provider(speed), 2),
            },
        }
        return CloudRequest(
            url=url,
            headers={
                "xi-api-key": self._credential,
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
            body=json.dumps(payload).encode("utf-8"),
            audio_format=AudioFormat.PCM,
            sample_rate=rate,
        )

    def _voice_request(self) -> CloudRequest:
        """GET ``/voices``: what this key may actually use.

        Worth a request, unlike the other three providers: an ElevenLabs account
        holds cloned voices that no documentation can list.
        """
        return CloudRequest(
            url=f"{self._endpoint()}/voices",
            headers={"xi-api-key": self._credential, "Accept": "application/json"},
            method="GET",
        )

    def _parse_voices(self, data: bytes) -> tuple[VoiceSpec, ...]:
        """Read ``{"voices": [{"voice_id": ..., "name": ...}]}``.

        Entries missing an id are skipped rather than raised on: one malformed
        record in a list of thirty must not empty the settings combo box.
        """
        payload = self._decode_json(data)
        entries = payload.get("voices")
        if not isinstance(entries, list):
            return ()
        found: list[VoiceSpec] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            voice_id = entry.get("voice_id")
            if not isinstance(voice_id, str) or not voice_id:
                continue
            labels = entry.get("labels")
            language = "ru"
            if isinstance(labels, dict):
                candidate = labels.get("language")
                if isinstance(candidate, str) and candidate:
                    language = candidate[:2].lower()
            name = entry.get("name")
            found.append(
                VoiceSpec(
                    engine=self.name,
                    voice_id=voice_id,
                    language=language,
                    display_name=name if isinstance(name, str) else "",
                    sample_rate=self.native_sample_rate,
                )
            )
        return tuple(found)
