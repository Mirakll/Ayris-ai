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

    __slots__ = ("_buffer", "_emitted", "_max_chars", "_released")

    def __init__(self, *, max_chars: int = MAX_CHUNK_CHARS) -> None:
        self._buffer = ""
        self._emitted = 0
        self._released = 0
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
        return self._released

    def reset(self) -> None:
        """Forget all state, ready for a new request."""
        self._buffer = ""
        self._emitted = 0
        self._released = 0

    def _drain(self, *, final: bool) -> list[str]:
        sentences = split_sentences(self._buffer, max_chars=self._max_chars)
        # Every sentence but the last is confirmed by the text that follows it;
        # the last waits for the next fragment, unless the stream just ended.
        confirmed = len(sentences) if final else max(0, len(sentences) - 1)
        ready = sentences[self._emitted : confirmed]
        self._emitted = max(self._emitted, confirmed)
        self._released += len(ready)
        if final:
            self.reset()
            return ready
        if confirmed:
            self._trim(sentences[:confirmed])
        return ready

    def _trim(self, confirmed: list[str]) -> None:
        """Drop confirmed sentences from the buffer so the next fragment re-splits
        only the unconfirmed tail.

        Without this the whole, growing answer is re-scanned by
        :func:`split_sentences` on *every* token — O(n²) over a long reply. The
        cut is taken only when the confirmed sentences appear verbatim and in
        order in the buffer: :func:`split_sentences` may merge a short piece into
        its neighbour, drop a non-speakable one, or clause-split a run-on, and
        then the raw position is not recoverable — that fragment keeps the full
        buffer and pays the full scan, which is correct, only not faster.
        Trimming at a confirmed boundary cannot change how the tail splits (the
        abbreviation look-back never reaches past a sentence end); ``_emitted`` is
        the index into the *current* buffer's split and resets with it, while the
        public :attr:`emitted_count` keeps counting across trims. A differential
        test pins the emitted sequence identical to the untrimmed assembler.
        """
        cut = 0
        for sentence in confirmed:
            found = self._buffer.find(sentence, cut)
            if found < 0:
                return  # lossy split this fragment — leave the buffer whole
            cut = found + len(sentence)
        self._buffer = self._buffer[cut:]
        self._emitted = 0
