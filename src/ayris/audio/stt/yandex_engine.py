"""Yandex SpeechKit, short recognition.

The default provider for a Russian assistant, and the one with the shortest
round trip from a machine in Russia. It takes raw LPCM in a query string rather
than a container, answers with one field, and says nothing at all about how sure
it is - so :data:`~ayris.audio.stt.cloud_base.ASSUMED_CONFIDENCE` stands in.

**Two auth schemes, and the difference is not cosmetic.** A service account can
be used through a long-lived API key (``Api-Key <key>``) or through an IAM token
that expires in twelve hours (``Bearer <token>``). Ayris cannot refresh an IAM
token - that needs the account's private key, which is not something to keep in
a credential store next to the key it would replace - so the API key is the
default and IAM is offered for the user who already has a token pipeline. The
scheme is a setting, not a guess: sending the wrong prefix gets a 401 that looks
exactly like a wrong key.

**folderId is required and is not a secret.** It identifies the billing folder,
travels in the query string, and belongs in ``config.toml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final
from urllib.parse import urlencode

from ayris.audio.stt.base import TranscriptResult
from ayris.audio.stt.cloud_base import (
    ASSUMED_CONFIDENCE,
    CloudSttEngine,
    _Request,
)
from ayris.core.errors import SttError

if TYPE_CHECKING:
    from ayris.audio.stt.base import AudioBuffer, SttOptions

__all__ = ["YandexSttEngine"]

#: Short recognition: up to 30 seconds and 1 MB, which is what a spoken command
#: is. The streaming API is a different protocol (gRPC) and a different task.
_ENDPOINT: Final = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"

#: SpeechKit wants a full locale, not a two-letter code.
_LOCALES: Final = {
    "ru": "ru-RU",
    "en": "en-US",
    "tr": "tr-TR",
    "kk": "kk-KK",
    "uz": "uz-UZ",
}


class YandexSttEngine(CloudSttEngine):
    """Speech recognition through Yandex SpeechKit."""

    name: ClassVar[str] = "yandex"
    title: ClassVar[str] = "Яндекс SpeechKit"
    default_ref: ClassVar[str] = "yandex"
    default_endpoint: ClassVar[str] = _ENDPOINT

    __slots__ = ()

    def _build_request(self, audio: AudioBuffer, options: SttOptions) -> _Request:
        """POST raw LPCM with the parameters in the query string.

        Raises:
            SttError: ``folder_id`` is not configured. SpeechKit answers a
                missing folder with a 400 that does not say which field is
                wrong, so catching it here saves the user that hunt.
        """
        folder = options.option("folder_id")
        if not folder:
            raise SttError(
                f"{self.name}: folder_id is not configured",
                user_message=(
                    "Для Яндекс SpeechKit нужен идентификатор каталога. "
                    "Укажите folder_id в настройках голоса."
                ),
            )

        query = {
            "lang": _LOCALES.get(options.language, options.language or "ru-RU"),
            "folderId": folder,
            "format": "lpcm",
            "sampleRateHertz": str(self.sample_rate),
            "profanityFilter": "false",
            # SpeechKit calls this "raw" vs "normalized"; normalisation is what
            # turns "двадцать два" into "22" and adds the punctuation.
            "rawResults": "false" if options.punctuation else "true",
        }
        model = options.option("model")
        if model:
            query["model"] = model

        url = f"{self._endpoint(options)}?{urlencode(query)}"
        return _Request(url=url, headers=self._auth_headers(options), body=audio.pcm)

    def _auth_headers(self, options: SttOptions) -> dict[str, str]:
        """Authorization header for the configured scheme.

        Raises:
            SttError: The scheme is neither ``api-key`` nor ``iam``.
        """
        scheme = (options.option("auth_scheme") or "api-key").lower()
        if scheme in ("api-key", "apikey", "api_key"):
            return {"Authorization": f"Api-Key {self._credential}"}
        if scheme in ("iam", "bearer", "iam-token"):
            return {"Authorization": f"Bearer {self._credential}"}
        raise SttError(
            f"{self.name}: unknown auth scheme {scheme!r}",
            user_message=(
                f"Неизвестный способ авторизации Яндекс SpeechKit: «{scheme}». "
                f"Допустимые значения — api-key и iam."
            ),
        )

    def _parse_response(self, data: bytes, options: SttOptions) -> TranscriptResult:
        """Read the single ``result`` field.

        An empty string is a real answer here and means SpeechKit heard nothing
        recognisable, so it becomes an empty result rather than an error.
        """
        payload = self._decode_json(data)
        raw = payload.get("result")
        if raw is None:
            raise SttError(
                f"{self.name}: response has no 'result' field: {sorted(payload)}",
                user_message=f"{self._provider_name} вернул ответ в неизвестном формате.",
            )
        text = str(raw).strip()
        if not text:
            return TranscriptResult.empty(
                engine=self.name,
                language=options.language,
                device="cloud",
                model=options.option("model", "general"),
            )
        return self._result(
            text=text,
            # SpeechKit's short recognition returns no confidence at all: the
            # field exists in the streaming protocol only, and is documented
            # there as not yet meaningful.
            confidence=ASSUMED_CONFIDENCE,
            language=options.language,
            model=options.option("model", "general"),
        )
