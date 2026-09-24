"""A user-supplied, OpenAI-compatible provider.

The six branded providers each pin a host, so ``openai`` always talks to OpenAI
and ``deepseek`` always to DeepSeek. This one pins nothing: the base URL, the
model name and the key all come from the user. Any service that speaks the
OpenAI ``/chat/completions`` streaming schema — an OpenRouter-style broker,
Together, Groq, Fireworks, a self-hosted vLLM — works the moment its endpoint,
model and key are filled in, without a new provider class per site.

Mechanically it is exactly :class:`~ayris.nlu.llm.cloud.OpenAiCompatibleClient`:
``data: {json}`` frames, ``Authorization: Bearer``, ``{base}/chat/completions``
and ``{base}/models``. The only difference is that there is no default host to
fall back on, so it reports itself unconfigured until a base URL is set as well
as a key — otherwise a request would be POSTed to ``/chat/completions`` with no
host and fail in a way the user cannot read.
"""

from __future__ import annotations

from typing import ClassVar

from ayris.nlu.llm.cloud import OpenAiCompatibleClient

__all__ = ["CustomOpenAiLlmClient"]


class CustomOpenAiLlmClient(OpenAiCompatibleClient):
    """Any OpenAI-compatible endpoint the user points Ayris at.

    The base URL arrives through :attr:`~ayris.nlu.llm.cloud.CloudOptions.base_url`
    (the factory reads it from ``ai.host`` for this provider only) and the model
    through :attr:`~ayris.nlu.llm.cloud.CloudOptions.model`. Both auth and stream
    framing are inherited unchanged.
    """

    name: ClassVar[str] = "custom"
    title: ClassVar[str] = "Свой провайдер"
    default_base_url: ClassVar[str] = ""
    default_model: ClassVar[str] = ""

    @property
    def configured(self) -> bool:
        """Needs both a key and an endpoint — there is no default host here.

        A key without a base URL is not usable: unlike the branded providers this
        client has nowhere to send the request. Reporting ``False`` makes the
        worker answer with the «не настроено» hint instead of failing a doomed
        POST to a hostless URL.
        """
        return bool(self._api_key and self._base_url)
