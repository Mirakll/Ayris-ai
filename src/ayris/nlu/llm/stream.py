"""Turning a token stream into sentences, so speech can start before it ends.

The whole reason the LLM path streams is latency: TTS should begin speaking the
first sentence while the model is still writing the second. That needs the token
deltas assembled into sentences the moment each one is complete — no sooner, or
the voice stops mid-thought, and no later, or the early start is wasted.

:class:`SentenceAssembler` is the state that makes that incremental. It buffers
the text that has arrived and, on each fragment, hands back the sentences that
are now *confirmed* complete. Confirmation is deliberately conservative: a
sentence is only released once text follows its terminator, because
:func:`~ayris.audio.tts.sentence_split.split_sentences` cannot tell «Готово.» the
end of a thought from «т. е.» an abbreviation until it sees what comes next. The
last piece therefore always stays buffered until the following fragment confirms
it, and :meth:`flush` releases whatever remains when the stream ends.

A run-on with no terminator in sight is not held forever: the same length valve
:func:`split_sentences` uses breaks it at a clause boundary, so a model that
writes one enormous sentence still starts speaking on time.
"""

from __future__ import annotations

from ayris.audio.tts.sentence_split import MAX_CHUNK_CHARS, split_sentences

__all__ = ["SentenceAssembler"]


class SentenceAssembler:
    """Accumulates streamed text and releases whole sentences as they complete.

    Not thread-safe: one assembler belongs to one in-flight request, fed from the
    single thread draining that request's deltas.
    """

    __slots__ = ("_buffer", "_emitted", "_max_chars")

    def __init__(self, *, max_chars: int = MAX_CHUNK_CHARS) -> None:
        self._buffer = ""
        self._emitted = 0
        self._max_chars = max_chars

    def feed(self, text: str) -> list[str]:
        """Add a fragment; return the sentences it just completed, in order.

        The list is usually empty — most fragments land in the middle of a
        sentence — and occasionally holds one or more when a fragment carried a
        boundary or pushed a run-on past the length valve.
        """
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """Release everything still buffered; call once when the stream ends.

        This is where the last sentence of an answer comes from: it was held back
        because nothing followed its terminator to confirm it, and the stream
        ending is that confirmation.
        """
        return self._drain(final=True)

    @property
    def emitted_count(self) -> int:
        """How many sentences have been released so far in this request."""
        return self._emitted

    def reset(self) -> None:
        """Forget all state, ready for a new request."""
        self._buffer = ""
        self._emitted = 0

    def _drain(self, *, final: bool) -> list[str]:
        sentences = split_sentences(self._buffer, max_chars=self._max_chars)
        # Every sentence but the last is confirmed by the text that follows it;
        # the last waits for the next fragment, unless the stream just ended.
        confirmed = len(sentences) if final else max(0, len(sentences) - 1)
        ready = sentences[self._emitted : confirmed]
        self._emitted = max(self._emitted, confirmed)
        if final:
            self.reset()
        return ready
