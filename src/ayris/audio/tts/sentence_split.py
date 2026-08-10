"""Splitting text into speakable pieces, so the first sound arrives early.

The specification puts local synthesis under 500 ms to first audio. Synthesizing a
four-sentence answer takes longer than that on any CPU engine, so the player asks
for the first sentence, starts it, and lets the worker synthesize the second while
the first is still sounding. That only works if the split is right: a break in the
wrong place makes the voice stop mid-thought, and a missed break makes the user
wait for the whole paragraph.

**Why not ``text.split(".")``.** Russian abbreviates with periods constantly - «т.
е.», «и т. д.», «ул. Ленина, д. 5», «рис. 3» - and every one of those would become
a sentence boundary. So would decimals («3.14»), version numbers («Python 3.12»),
initials («А. С. Пушкин») and ellipses. Each of those produces an audible stumble.

**Why not a sentence tokenizer library.** The candidates all bring a model file and
a language pack, which for this job - deciding where a voice may pause - is
disproportionate. The rules below cover Russian and English punctuation with a
lookup table of abbreviations and two character-class checks, and they are testable
without a download.

**Long sentences still get split.** A subordinate-clause chain can run past 300
characters, which is six seconds of speech and well past the latency budget for the
first chunk. Past :data:`MAX_CHUNK_CHARS` the text is broken at the last comma,
semicolon, colon or dash - places a human speaker would breathe anyway - and only
falls back to a word boundary when the clause has no punctuation at all.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "ABBREVIATIONS",
    "MAX_CHUNK_CHARS",
    "MIN_CHUNK_CHARS",
    "SENTENCE_ENDINGS",
    "is_speakable",
    "normalize_whitespace",
    "split_sentences",
]

#: Characters that may end a sentence.
SENTENCE_ENDINGS: Final = ".!?…"

#: Past this many characters a sentence is broken at a clause boundary, so the
#: first chunk stays inside the latency budget.
MAX_CHUNK_CHARS: Final = 300

#: A trailing fragment shorter than this is merged into the previous chunk instead
#: of being spoken on its own: «Да.» as a separate synthesis call costs a model
#: warm-up and sounds clipped.
MIN_CHUNK_CHARS: Final = 12

#: Words that end in a period without ending a sentence. Lower-cased, without the
#: trailing period. Russian first, since that is what Ayris speaks; the English
#: entries are here because model names and quoted documentation leak into answers.
ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        # Russian: enumeration and reference
        "т",
        "е",
        "д",
        "п",
        "др",
        "пр",
        "см",
        "ср",
        "напр",
        "итд",
        "итп",
        "тыс",
        "млн",
        "млрд",
        "руб",
        "коп",
        # Russian: address and document
        "г",
        "гг",
        "обл",
        "ул",
        "пер",
        "просп",
        "пл",
        "кв",
        "корп",
        "стр",
        "рис",
        "табл",
        "гл",
        "разд",
        "ст",
        "им",
        "с",
        "стp",
        # Russian: titles
        "проф",
        "доц",
        "акад",
        "чл",
        "канд",
        "дн",
        "тов",
        # units and time
        "мм",
        "км",
        "кг",
        "мл",
        "мин",
        "сек",
        "ч",
        "мес",
        # English
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "vs",
        "etc",
        "e",
        "i",
        "fig",
        "no",
        "vol",
        "approx",
        "inc",
        "ltd",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)

#: Clause boundaries used when a sentence is too long to speak in one piece.
_CLAUSE_BREAKS: Final = ",;:—–"

#: Collapses runs of whitespace, including the newlines an LLM answer arrives with.
_WHITESPACE: Final = re.compile(r"\s+")

#: The word immediately before a candidate boundary, without its period.
_TRAILING_WORD: Final = re.compile(r"([^\s.!?…]+)[.!?…]*$")

#: A single letter followed by a period is an initial («А. С. Пушкин») or a list
#: marker, never a sentence end.
_SINGLE_LETTER: Final = re.compile(r"^\w$", re.UNICODE)

#: A letter or a digit. Text with none of either has nothing to pronounce.
_SPEAKABLE: Final = re.compile(r"[^\W_]", re.UNICODE)


def is_speakable(text: str) -> bool:
    """Whether there is anything in ``text`` an engine could pronounce.

    «...», «!!!», a lone dash and a bare emoji all arrive from an LLM answer
    often enough to matter, and each one costs a synthesis call that comes back
    as a click. Checked here rather than in the engines so that every engine
    behaves the same way, and so the split never hands out a piece that is
    nothing but punctuation.
    """
    return _SPEAKABLE.search(text) is not None


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs and trim the ends.

    Applied before splitting so that a boundary check never has to reason about a
    newline in the middle of a sentence, and so the pieces handed to an engine
    contain no line breaks - Piper pronounces some of them as pauses and Silero
    turns them into artefacts.
    """
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into pieces that can each be synthesized on their own.

    Args:
        text: Plain text, possibly multi-paragraph.
        max_chars: Soft ceiling per piece. A sentence longer than this is broken
            at a clause boundary; ``0`` disables the second pass and returns whole
            sentences however long they are.

    Returns:
        Pieces in order, each stripped and each with something to pronounce in
        it. Nothing with words in it is dropped - a lost clause would be a lost
        sentence in the spoken answer - but a piece that is only punctuation is,
        because :func:`is_speakable` is what decides that «...» is not a phrase.
    """
    normalized = normalize_whitespace(text)
    if not is_speakable(normalized):
        return []

    sentences = _split_by_terminator(normalized)
    if max_chars <= 0:
        return [piece for piece in sentences if is_speakable(piece)]

    pieces: list[str] = []
    for sentence in sentences:
        pieces.extend(_split_long(sentence, max_chars))
    return _merge_short([piece for piece in pieces if is_speakable(piece)])


def _split_by_terminator(text: str) -> list[str]:
    """Break at ``.!?…`` that genuinely end a sentence."""
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char not in SENTENCE_ENDINGS:
            index += 1
            continue

        # Consume a run of terminators and the closing quotes or brackets that
        # belong to the sentence: «Готово!» must not leave the quote behind.
        end = index + 1
        while end < length and text[end] in SENTENCE_ENDINGS:
            end += 1
        while end < length and text[end] in "\"»”')]":
            end += 1

        if end >= length:
            break
        if not _is_boundary(text, index, end):
            index = end
            continue

        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = end
        index = end

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_boundary(text: str, terminator: int, end: int) -> bool:
    """Whether the terminator at ``terminator`` really ends a sentence.

    ``end`` is the first index after the terminator run and its closing quotes.
    """
    # Nothing but whitespace may follow: "3.14" and "www.example.com" are one word.
    if not text[end].isspace():
        return False

    following = text[end:].lstrip()
    if not following:
        return False

    # A sentence starts with a capital, a digit, a quote or a dash. A lower-case
    # continuation means the period belonged to an abbreviation that is not in the
    # table - the safer reading, because a wrong break is audible and a missed one
    # only delays the pause.
    first = following[0]
    if not (first.isupper() or first.isdigit() or first in "\"«“'(-—–"):
        return False

    if text[terminator] != ".":
        # ! ? and … are unambiguous; only the period is overloaded.
        return True

    match = _TRAILING_WORD.search(text[:terminator])
    if match is None:
        return True
    word = match.group(1)
    if _SINGLE_LETTER.match(word):
        return False
    if word.casefold() in ABBREVIATIONS:
        return False
    # "5." in "д. 5. Далее" is a house number, and "1." in "1. Открыть файл" is a
    # list marker; neither ends a sentence, even though a capital follows. Breaking
    # there strands the number as its own utterance, which is the audible mistake
    # this whole function exists to avoid. A single digit is already handled by the
    # check above - \w matches digits too - so this covers "12." and longer.
    return not word.isdigit()


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence at clause boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]

    pieces: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        cut = _clause_cut(rest, max_chars)
        if cut <= 0:
            break
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return [piece for piece in pieces if piece]


def _clause_cut(text: str, max_chars: int) -> int:
    """Index to cut ``text`` at, preferring punctuation over a word boundary."""
    window = text[: max_chars + 1]
    for index in range(len(window) - 1, 0, -1):
        if window[index] in _CLAUSE_BREAKS:
            return index + 1
    space = window.rfind(" ")
    return space + 1 if space > 0 else 0


def _merge_short(pieces: list[str]) -> list[str]:
    """Fold a too-short tail into its predecessor.

    Only the tail: a short piece in the middle is a real sentence («Готово.») and
    speaking it separately is correct. A short *final* piece is almost always the
    remainder of a clause split, and synthesizing it alone adds a pause the text
    did not ask for.
    """
    if len(pieces) < 2:
        return pieces
    if len(pieces[-1]) >= MIN_CHUNK_CHARS:
        return pieces
    merged = list(pieces[:-1])
    merged[-1] = f"{merged[-1]} {pieces[-1]}"
    return merged
