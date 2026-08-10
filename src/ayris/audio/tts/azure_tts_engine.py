"""Azure Speech: SSML in, raw PCM out, and the region is part of the hostname.

Azure is the only one of the four that takes markup rather than a text field.
That is a small advantage - ``prosody`` carries rate, pitch and volume in one
element, so all three sliders reach the service - and one real hazard: the
user's text lands inside a document, and a phrase containing ``&`` or ``<`` would
produce either malformed XML or, worse, markup the service interprets.
:func:`xml.sax.saxutils.escape` runs on every phrase for that reason.

The endpoint carries the region: ``https://<region>.tts.speech.microsoft.com``.
There is no global host to fall back to, so a missing region is refused here with
a sentence naming the setting rather than sent off to fail as a DNS error the
user cannot read.

``X-Microsoft-OutputFormat: raw-24khz-16bit-mono-pcm`` asks for headerless
``int16``, and Azure streams it: the response body starts arriving while the rest
is still being synthesized, which is what
:attr:`~ayris.audio.tts.cloud_base.CloudTtsEngine.supports_streaming` turns on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final
from xml.sax.saxutils import escape

from ayris.audio.tts.base import VoiceSpec
from ayris.audio.tts.cloud_base import (
    AudioFormat,
    CloudRequest,
    CloudTtsEngine,
    TtsAuthError,
    percent_offset,
)

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

__all__ = ["AzureTtsEngine"]

#: Voice used when the settings name none. Svetlana is the Russian neural voice
#: Azure documents first.
_DEFAULT_VOICE: Final = "ru-RU-SvetlanaNeural"

#: Percentage bounds for ``prosody rate``. Azure accepts more in principle, but
#: past these the voice stops being intelligible, which is not a setting worth
#: honouring.
_RATE_PERCENT: Final = (-50.0, 100.0)

#: Percentage bounds for ``prosody pitch``. Narrower than rate: a neural voice
#: shifted by more than half an octave sounds synthetic in a way the local
#: engines do not.
_PITCH_PERCENT: Final = (-50.0, 50.0)


class AzureTtsEngine(CloudTtsEngine):
    """Azure Speech synthesis over HTTPS, streaming raw PCM."""

    name: ClassVar[str] = "azure"
    title: ClassVar[str] = "Azure Speech"
    default_ref: ClassVar[str] = "azure"
    supports_streaming: ClassVar[bool] = True

    __slots__ = ()

    def _build_request(
        self,
        text: str,
        speed: float,
        pitch: float,
        *,
        stream: bool,
    ) -> CloudRequest:
        """POST an SSML document.

        ``stream`` changes nothing about the request: the same endpoint streams
        or does not depending on how the caller reads the response, which is the
        one provider here where that is true.
        """
        del stream
        voice = self._require_loaded()
        return CloudRequest(
            url=f"{self._host()}/cognitiveservices/v1",
            headers={
                "Ocp-Apim-Subscription-Key": self._credential,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
            },
            body=self._ssml(text, voice, speed, pitch).encode("utf-8"),
            audio_format=AudioFormat.PCM,
            sample_rate=self.native_sample_rate,
        )

    def _ssml(self, text: str, voice: VoiceSpec, speed: float, pitch: float) -> str:
        """Wrap one phrase in the smallest document Azure accepts.

        The escaping is the load-bearing part. Everything else here is a fixed
        shape; ``escape`` is what stands between a user saying "чай & кофе" and
        a 400 from the service.
        """
        voice_id = voice.voice_id or _DEFAULT_VOICE
        language = self._options.option("language") or self._language_of(voice_id)
        rate = percent_offset(speed, _RATE_PERCENT)
        tone = percent_offset(pitch, _PITCH_PERCENT)
        return (
            f'<speak version="1.0" xml:lang="{language}">'
            f'<voice name="{voice_id}">'
            f'<prosody rate="{rate:+d}%" pitch="{tone:+d}%">'
            f"{escape(text)}"
            f"</prosody></voice></speak>"
        )

    def _language_of(self, voice_id: str) -> str:
        """The locale from the voice name: ``ru-RU-SvetlanaNeural`` is ``ru-RU``."""
        parts = voice_id.split("-")
        if len(parts) >= 2 and len(parts[0]) == 2:
            return f"{parts[0]}-{parts[1]}"
        return "ru-RU"

    def _host(self) -> str:
        """The regional base URL.

        Raises:
            TtsAuthError: No region is configured. Reported as a credential
                problem because that is what it is from the user's side - the
                key and the region are one setting in their head, and both are
                filled in on the same screen.
        """
        configured = self._options.option("endpoint")
        if configured:
            return configured.rstrip("/")
        region = self._options.option("region")
        if not region:
            raise TtsAuthError(
                f"{self.name}: no region configured",
                user_message=(
                    "Для Azure Speech не указан регион. "
                    "Впишите его в настройках голоса, например westeurope."
                ),
            )
        return f"https://{region}.tts.speech.microsoft.com"

    def _voice_request(self) -> CloudRequest:
        """GET the region's voice list.

        Worth the request here: which neural voices a region carries differs
        between regions, and a hard-coded table would offer voices that return
        400 in half of them.
        """
        return CloudRequest(
            url=f"{self._host()}/cognitiveservices/voices/list",
            headers={
                "Ocp-Apim-Subscription-Key": self._credential,
                "Accept": "application/json",
            },
            method="GET",
        )

    def _parse_voices(self, data: bytes) -> tuple[VoiceSpec, ...]:
        """Read the list, keeping the neural voices.

        Azure still lists retired standard voices, which synthesize but sound a
        decade old; offering them next to the neural ones would be a worse combo
        box, not a longer one.
        """
        import json

        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return ()
        if not isinstance(payload, list):
            return ()
        found: list[VoiceSpec] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            spec = self._voice_of(entry)
            if spec is not None:
                found.append(spec)
        return tuple(found)

    def _voice_of(self, entry: JsonObject) -> VoiceSpec | None:
        """One list entry as a spec, or ``None`` when it is not usable."""
        short_name = entry.get("ShortName")
        if not isinstance(short_name, str) or not short_name:
            return None
        kind = entry.get("VoiceType")
        if isinstance(kind, str) and "Neural" not in kind:
            return None
        locale = entry.get("Locale")
        language = locale[:2].lower() if isinstance(locale, str) and locale else "ru"
        label = entry.get("LocalName") or entry.get("DisplayName")
        return VoiceSpec(
            engine=self.name,
            voice_id=short_name,
            language=language,
            display_name=label if isinstance(label, str) else "",
            sample_rate=self.native_sample_rate,
        )
