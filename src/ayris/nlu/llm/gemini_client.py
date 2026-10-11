"""Google Gemini as a cloud LLM provider.

Gemini is reached through Google's OpenAI-compatibility endpoint
(``/v1beta/openai``), which speaks the same ``/chat/completions`` streaming
schema as everyone else, so this is the same thin provider as
:mod:`~ayris.nlu.llm.deepseek_client` with Google's host and a Gemini default
model. The key is the one from Google AI Studio, sent as a bearer token.

The endpoint is region-restricted by Google; where it is blocked, the very same
models are reachable through the ``openrouter`` provider instead
(``google/gemini-...``), over the identical schema and this same base class.
"""

from __future__ import annotations

from typing import ClassVar

from ayris.nlu.llm.cloud import CloudOptions, OpenAiCompatibleClient

__all__ = ["GeminiLlmClient"]


class GeminiLlmClient(OpenAiCompatibleClient):
    """Gemini over Google's OpenAI-compatible endpoint."""

    name: ClassVar[str] = "gemini"
    title: ClassVar[str] = "Gemini (Google)"
    default_base_url: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta/openai"
    # A moving alias, not a pinned version: Google restricts older ids
    # (``gemini-2.5-flash`` and down) to projects that already used them, so a
    # pinned version 404s on a freshly created key. ``gemini-flash-latest`` always
    # resolves to the current Flash model, which new projects can reach.
    default_model: ClassVar[str] = "gemini-flash-latest"

    #: Model-id prefixes Google's endpoint actually serves. A configured id that
    #: matches none of these is stale config carried over from another provider —
    #: the single ``ai.model`` field is kept when the provider switches — so fall
    #: back to the default and keep answering, instead of 404-ing a local model
    #: name (``Vikhr-…``) against Google.
    _FAMILIES: ClassVar[tuple[str, ...]] = ("gemini", "gemma", "learnlm")

    def __init__(self, options: CloudOptions) -> None:
        super().__init__(options)
        # Google's model list (``/models``) names models ``models/gemini-flash-latest``,
        # and the settings model-picker stores that verbatim. But the OpenAI-compatible
        # endpoint wants the bare id in the request body — the prefixed form comes back
        # as HTTP 404 "model not found". Strip a leading ``models/`` so both the picked
        # name and a hand-typed id work.
        prefix = "models/"
        if self._model.startswith(prefix):
            self._model = self._model[len(prefix) :]
        if not self._model.lower().startswith(self._FAMILIES):
            self._model = self.default_model
