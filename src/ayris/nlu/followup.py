"""«Повтори», «отмена», and the questions Ayris asks when a slot is missing.

Three kinds of utterance are not commands at all, but moves in a conversation,
and none of them can be handled by the trigger library. «Отмена» has to work
while a macro is halfway through and while the assistant is still talking, so it
cannot wait its turn behind a match. «Повтори» refers to whatever ran last, so
there is nothing to write a trigger against. And an answer to a question Ayris
just asked — «какую программу?» → «хром» — is a bare noun that matches nothing
and means everything. This module reads all three off the phrase and hands the
pipeline of task 18 a decision.

**Cancel is recognised by the whole phrase, and only by the whole phrase.**
«Стоп» is a cancel; «стоп музыка» is a command about music, and treating the
first word as a cancel would make that command impossible to say. So the test is
equality against :data:`CANCEL_PHRASES` after normalisation, not a prefix or a
word-containment check. That is what lets cancel keep the highest priority — it
is checked before the matcher runs — without stealing anything from the library:
a phrase that means something more specific has more words in it, and therefore
is not equal to any cancel phrase.

**Repeat has two meanings and the wording picks one.** «Повтори» asks to hear the
last answer again; «ещё раз» asks to do the last thing again. Both are useful and
they are not interchangeable, so :data:`REPEAT_ANSWER_PHRASES` and
:data:`REPEAT_ACTION_PHRASES` are separate tables rather than one table and a
guess. The two have different deadlines behind them, which is the deeper reason
they are separated: an answer can be repeated long after the dialogue closed
(:attr:`~ayris.nlu.context.ContextTtl.answer`), an action only while it is still
the thing being talked about (:attr:`~ayris.nlu.context.ContextTtl.followup`).

**A dangerous action is never repeated on one word.** «Ещё раз» after «выключи
компьютер» comes back as :attr:`FollowUpKind.CONFIRM` with a question to ask,
not as a repeat. Whether an action is dangerous is not decided here: it is
carried in :attr:`~ayris.nlu.context.LastCommand.dangerous`, set by whoever
declared the action, because deciding it from the intent name would be a security
decision made by string matching.

**A repeat does not become the thing to repeat.** :attr:`FollowUp.records_context`
is ``False`` for every repeat, and the pipeline consults it instead of
remembering unconditionally. Without that, «ещё раз» would overwrite the last
command with itself and the second «ещё раз» would repeat the repeat — which is
either a no-op or an infinite regress, depending on how it was implemented.

**Clarification is the same mechanism pointed the other way.** :func:`ask_slot`
turns a missing slot into a spoken question and a :class:`PendingRequest` in the
context; the next utterance goes to :func:`answer_pending` before anything else
looks at it. That function is where the awkward cases live: the answer may be a
cancel, it may fail to parse into the slot's type — in which case the question is
asked once more and then abandoned rather than looping — and it may be a plain
«да» to a confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from ayris.core.events import CancelRequested
from ayris.nlu.context import PendingKind, PendingRequest
from ayris.nlu.normalize import normalize
from ayris.nlu.slot_types import BuiltinSlotType, SlotContext
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ayris.core.events import EventBus
    from ayris.nlu.context import ContextSnapshot, LastAnswer, LastCommand
    from ayris.nlu.normalize import NormalizedText
    from ayris.nlu.slot_types import SlotTypeRegistry

__all__ = [
    "CANCEL_PHRASES",
    "CANCEL_REASON",
    "MAX_CLARIFY_ATTEMPTS",
    "NO_PHRASES",
    "REPEAT_ACTION_PHRASES",
    "REPEAT_ANSWER_PHRASES",
    "YES_PHRASES",
    "AnswerStatus",
    "FollowUp",
    "FollowUpKind",
    "PendingAnswer",
    "RepeatMode",
    "answer_pending",
    "ask_slot",
    "confirm_command",
    "is_cancel_phrase",
    "publish_cancel",
    "repeat_mode",
    "resolve_followup",
    "slot_question",
]

_log = get_logger(__name__)

#: Reason put on the :class:`~ayris.core.events.CancelRequested` this module
#: publishes, so the log distinguishes a spoken cancel from the hotkey and from
#: the overlay's button.
CANCEL_REASON: Final = "голосовая отмена"

#: How many times a question is asked before Ayris gives up on it. Two: the first
#: answer may have been misheard, a third attempt is an argument with the user.
MAX_CLARIFY_ATTEMPTS: Final = 2

#: Everything that means «stop what you are doing», as the whole utterance.
#:
#: Written in the shape :func:`~ayris.nlu.normalize.normalize` produces — folded
#: case, no ``ё``, no address — because that is what they are compared against.
CANCEL_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "отмена",
        "отмени",
        "отменить",
        "стоп",
        "стой",
        "отставить",
        "хватит",
        "прекрати",
        "прекратить",
        "перестань",
        "отбой",
        "молчи",
        "замолчи",
        "тихо",
        "не надо",
        "забудь",
    }
)

#: «Say that again.» Refers to the last answer.
REPEAT_ANSWER_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "повтори",
        "повтори ответ",
        "повтори последнее",
        "повтори последний ответ",
        "повтори что сказала",
        "повтори что ты сказала",
        "что ты сказала",
        "что ты сказал",
        "что сказала",
        "не расслышал",
        "не расслышала",
        "не понял",
        "не поняла",
    }
)

#: «Do that again.» Refers to the last command.
REPEAT_ACTION_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "еще раз",
        "и еще раз",
        "давай еще раз",
        "сделай еще раз",
        "повтори еще раз",
        "повтори действие",
        "повтори команду",
        "снова",
        "давай снова",
        "сделай снова",
        "как в прошлый раз",
        "то же самое",
        "сделай то же самое",
    }
)

#: «Yes» to a confirmation. Deliberately short: a long sentence answering a
#: yes/no question is more likely a new command than an elaborate agreement.
YES_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "да",
        "ага",
        "давай",
        "давай да",
        "конечно",
        "подтверждаю",
        "подтверждай",
        "верно",
        "точно",
        "именно",
        "ок",
        "окей",
        "хорошо",
        "делай",
        "продолжай",
        "угу",
    }
)

#: «No» to a confirmation. Overlaps :data:`CANCEL_PHRASES` on purpose — «отмена»
#: in answer to «повторить выключение?» is both a refusal and a cancel, and both
#: readings end in the same place.
NO_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "нет",
        "не",
        "не надо",
        "не нужно",
        "не хочу",
        "нет не надо",
        "отмена",
        "отмени",
        "отставить",
        "не стоит",
        "передумал",
        "передумала",
    }
)

#: Words dropped from both ends before a phrase is compared to a table. Politeness
#: and hesitation are not part of what was asked, and «повтори пожалуйста» has to
#: reach the same entry as «повтори». Kept small and closed: every word here is
#: one that can never be a command by itself.
_TRIMMED: Final[frozenset[str]] = frozenset(
    {"пожалуйста", "плиз", "ну", "а", "и", "давай-ка", "ка", "там", "уж"}
)

#: The question to ask for a slot of each built-in type. Type rather than name,
#: because a template's slot names are the user's and may be anything, while the
#: type is what says which answers are acceptable.
_SLOT_QUESTIONS: Final[Mapping[str, str]] = {
    BuiltinSlotType.APP.value: "Какую программу?",
    BuiltinSlotType.TIME.value: "На какое время?",
    BuiltinSlotType.DURATION.value: "На сколько времени?",
    BuiltinSlotType.VOLUME.value: "Какую громкость поставить?",
    BuiltinSlotType.PERCENT.value: "Сколько процентов?",
    BuiltinSlotType.SITE.value: "Какой сайт?",
    BuiltinSlotType.QUERY.value: "Что именно?",
    BuiltinSlotType.INT.value: "Какое число?",
    BuiltinSlotType.FLOAT.value: "Какое число?",
}

#: Asked when the type says nothing useful. Names the slot, because the user
#: wrote that name and will recognise it.
_SLOT_QUESTION_FALLBACK: Final = "Уточни, пожалуйста: {slot}."

#: Said when «повтори» has nothing to reach.
NOTHING_TO_REPEAT: Final = "Мне нечего повторить."
#: Said when «ещё раз» has nothing to reach.
NOTHING_TO_REDO: Final = "Я ещё ничего не делала."
#: Said when a clarification has been asked as often as it is going to be.
GAVE_UP: Final = "Не разобрала. Скажи команду целиком, пожалуйста."


class RepeatMode(StrEnum):
    """Which of the two repeats a phrase asked for."""

    ANSWER = "answer"
    ACTION = "action"


class FollowUpKind(StrEnum):
    """What a phrase turned out to be, once conversation moves are ruled out."""

    #: Not a conversational move. The pipeline should go on and match it.
    NONE = "none"
    #: «Отмена». Abort everything; the caller publishes and resets.
    CANCEL = "cancel"
    #: Say :attr:`FollowUp.answer` again.
    REPEAT_ANSWER = "repeat_answer"
    #: Run :attr:`FollowUp.command` again, slots and all.
    REPEAT_ACTION = "repeat_action"
    #: A repeat of something dangerous. Ask :attr:`FollowUp.pending` first.
    CONFIRM = "confirm"
    #: A repeat was asked for and there is nothing to repeat. Say so.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FollowUp:
    """The decision about one phrase.

    ``message`` is Russian and meant to be spoken as it is. It is filled for
    :attr:`FollowUpKind.UNAVAILABLE` and for :attr:`FollowUpKind.REPEAT_ANSWER`,
    where the message *is* the answer being repeated, so a caller that wants to
    speak and nothing else can ignore the rest of the fields.
    """

    kind: FollowUpKind
    phrase: str = ""
    answer: LastAnswer | None = None
    command: LastCommand | None = None
    pending: PendingRequest | None = None
    message: str = ""

    @property
    def handled(self) -> bool:
        """Whether this phrase was a conversational move, i.e. not for the matcher."""
        return self.kind is not FollowUpKind.NONE

    @property
    def speak(self) -> str:
        """What to say right now, or ``""`` when nothing needs saying."""
        if self.kind is FollowUpKind.CONFIRM and self.pending is not None:
            return self.pending.question
        return self.message

    @property
    def records_context(self) -> bool:
        """Whether this should be remembered as the last command.

        Always ``False``. A repeat that recorded itself would be the thing the
        next repeat repeats; a cancel that recorded itself would be revived by
        «ещё раз». The property exists so that the pipeline can ask rather than
        having to know, and so that this paragraph has somewhere to live.
        """
        return False


def _trim(words: Iterable[str]) -> tuple[str, ...]:
    """Drop politeness and hesitation from both ends of a phrase."""
    trimmed = list(words)
    while trimmed and trimmed[0] in _TRIMMED:
        trimmed.pop(0)
    while trimmed and trimmed[-1] in _TRIMMED:
        trimmed.pop()
    return tuple(trimmed)


def _key(text: str | NormalizedText) -> str:
    """The phrase in the shape the tables are written in."""
    phrase = normalize(text) if isinstance(text, str) else text
    return " ".join(_trim(phrase.words))


def is_cancel_phrase(text: str | NormalizedText) -> bool:
    """Whether the *whole* phrase is a cancel.

    Equality, not containment: see the module docstring on «стоп музыка». A
    caller that wants to know whether a cancel word appears somewhere in a phrase
    wants something else, and should not want it.
    """
    return _key(text) in CANCEL_PHRASES


def repeat_mode(text: str | NormalizedText) -> RepeatMode | None:
    """Which repeat a phrase is asking for, or ``None`` when it is not one.

    The action table is consulted first. «Повтори ещё раз» appears in both if you
    read it by prefix, and the more specific reading — the user said «ещё раз»,
    so they mean the action — is the one to take.
    """
    key = _key(text)
    if key in REPEAT_ACTION_PHRASES:
        return RepeatMode.ACTION
    if key in REPEAT_ANSWER_PHRASES:
        return RepeatMode.ANSWER
    return None


def resolve_followup(text: str | NormalizedText, snapshot: ContextSnapshot) -> FollowUp:
    """Read a phrase as a conversational move, or report that it is not one.

    Called by the pipeline *before* the matcher, because a cancel outranks
    everything in the library and a repeat is not in it. A phrase that is neither
    comes back as :attr:`FollowUpKind.NONE` and costs two set lookups.

    An outstanding clarification is not handled here: that is
    :func:`answer_pending`, and it runs before this, because «да» is an answer
    when a question is open and nothing at all when none is.
    """
    key = _key(text)
    if key in CANCEL_PHRASES:
        return FollowUp(kind=FollowUpKind.CANCEL, phrase=key)
    mode = repeat_mode(key)
    if mode is None:
        return FollowUp(kind=FollowUpKind.NONE, phrase=key)
    if mode is RepeatMode.ANSWER:
        return _repeat_answer(key, snapshot)
    return _repeat_action(key, snapshot)


def _repeat_answer(phrase: str, snapshot: ContextSnapshot) -> FollowUp:
    """«Повтори». The answer if there is one, otherwise fall back or admit it.

    The fallback to repeating the *action* is there for the common case where the
    last thing that happened had no spoken answer at all — «открой браузер» says
    nothing — and «повтори» plainly means «do it again». It is refused for a
    dangerous command, where guessing which of the two the user meant is not an
    acceptable way to arrive at a shutdown.
    """
    if snapshot.answer is not None:
        return FollowUp(
            kind=FollowUpKind.REPEAT_ANSWER,
            phrase=phrase,
            answer=snapshot.answer,
            message=snapshot.answer.text,
        )
    command = snapshot.command
    if command is not None and snapshot.follow_up_active and not command.dangerous:
        return FollowUp(kind=FollowUpKind.REPEAT_ACTION, phrase=phrase, command=command)
    return FollowUp(kind=FollowUpKind.UNAVAILABLE, phrase=phrase, message=NOTHING_TO_REPEAT)


def _repeat_action(phrase: str, snapshot: ContextSnapshot) -> FollowUp:
    """«Ещё раз». The last command, with a confirmation if it is dangerous."""
    command = snapshot.command
    if command is None or not snapshot.follow_up_active:
        return FollowUp(kind=FollowUpKind.UNAVAILABLE, phrase=phrase, message=NOTHING_TO_REDO)
    if command.dangerous:
        return FollowUp(
            kind=FollowUpKind.CONFIRM,
            phrase=phrase,
            command=command,
            pending=confirm_command(command),
        )
    return FollowUp(kind=FollowUpKind.REPEAT_ACTION, phrase=phrase, command=command)


def confirm_command(command: LastCommand, *, question: str = "") -> PendingRequest:
    """The question to ask before repeating ``command``.

    ``known`` carries the slots across, because the point of a repeat is that the
    user does not say them again, and the confirmation must not lose them on the
    way.
    """
    return PendingRequest(
        kind=PendingKind.CONFIRM,
        question=question or f"Повторить «{command.title}»?",
        command_id=command.command_id,
        intent=command.intent,
        known=dict(command.slots),
    )


def slot_question(slot: str, slot_type: str = "") -> str:
    """What to ask out loud for a missing slot.

    By type first, by name as a fallback. A user whose template says
    ``{кому:str}`` gets «Уточни, пожалуйста: кому.», which is clumsy but true,
    and better than a generic question that leaves them guessing which of the
    two holes Ayris is asking about.
    """
    question = _SLOT_QUESTIONS.get(slot_type)
    if question is not None:
        return question
    return _SLOT_QUESTION_FALLBACK.format(slot=slot)


def ask_slot(
    slot: str,
    *,
    slot_type: str = "",
    intent: str = "",
    command_id: int | None = None,
    known: Mapping[str, Any] | None = None,
    question: str = "",
    attempts: int = 0,
) -> PendingRequest:
    """The pending request for a missing slot, with the question already worded.

    ``known`` is the slots that *did* fill. Keeping them on the request is what
    makes the answer usable: «поставь громкость» → «какую громкость?» → «сорок»
    has to end up as one command with every slot in it, and the ones from the
    first utterance are only remembered here.
    """
    return PendingRequest(
        kind=PendingKind.SLOT,
        question=question or slot_question(slot, slot_type),
        command_id=command_id,
        intent=intent,
        slot=slot,
        slot_type=slot_type,
        known=dict(known or {}),
        attempts=attempts,
    )


class AnswerStatus(StrEnum):
    """What the next utterance did to an outstanding question."""

    #: There was no question. The phrase is somebody else's problem.
    NOT_PENDING = "not_pending"
    #: The user cancelled instead of answering.
    CANCELLED = "cancelled"
    #: «Да» to a confirmation.
    CONFIRMED = "confirmed"
    #: «Нет» to a confirmation.
    DECLINED = "declined"
    #: The slot parsed. :attr:`PendingAnswer.slots` is ready to run with.
    FILLED = "filled"
    #: The answer did not parse and the question is worth asking once more.
    RETRY = "retry"
    #: It did not parse and the attempts are used up. Give up out loud.
    EXHAUSTED = "exhausted"
    #: The phrase was not an answer at all, but a new command. Drop the question.
    NEW_COMMAND = "new_command"


@dataclass(frozen=True, slots=True)
class PendingAnswer:
    """What became of an outstanding question when the user spoke again."""

    status: AnswerStatus
    request: PendingRequest | None = None
    slots: Mapping[str, Any] | None = None
    value: object | None = None
    retry: PendingRequest | None = None
    message: str = ""

    @property
    def handled(self) -> bool:
        """Whether the phrase was consumed as an answer.

        ``False`` for :attr:`AnswerStatus.NOT_PENDING` and for
        :attr:`AnswerStatus.NEW_COMMAND`: in both the phrase still has to be
        matched as a command, and the only difference is whether a question was
        dropped in the process.
        """
        return self.status not in (AnswerStatus.NOT_PENDING, AnswerStatus.NEW_COMMAND)

    @property
    def speak(self) -> str:
        """What to say right now, or ``""``."""
        if self.status is AnswerStatus.RETRY and self.retry is not None:
            return self.retry.question
        return self.message


def answer_pending(
    pending: PendingRequest | None,
    text: str | NormalizedText,
    *,
    registry: SlotTypeRegistry | None = None,
    context: SlotContext | None = None,
) -> PendingAnswer:
    """Read the next utterance as the answer to an outstanding question.

    Runs before everything else in the pipeline, and gets out of the way quickly:
    with no question open it returns :attr:`AnswerStatus.NOT_PENDING` without
    normalising anything expensive.

    Args:
        pending: The outstanding question, from
            :meth:`~ayris.nlu.context.DialogContext.pending`. Already ``None``
            when it timed out — the expiry lives in the context, not here.
        text: What the user said.
        registry: Slot types, for parsing a :attr:`~ayris.nlu.context.PendingKind.SLOT`
            answer. Without one the raw text is accepted as the value, which is
            right for a free-text slot and the best available otherwise.
        context: Clock and application dictionary for the parser, exactly as
            :meth:`ayris.nlu.slots.SlotTemplate.extract` takes it.

    Returns:
        A :class:`PendingAnswer`. :attr:`~PendingAnswer.handled` says whether the
        phrase was used up; when it was not, the caller carries on and matches it
        as a command.
    """
    if pending is None:
        return PendingAnswer(status=AnswerStatus.NOT_PENDING)
    key = _key(text)
    if not key:
        return PendingAnswer(status=AnswerStatus.NOT_PENDING, request=pending)
    if key in CANCEL_PHRASES:
        return PendingAnswer(status=AnswerStatus.CANCELLED, request=pending)
    if pending.kind is PendingKind.CONFIRM:
        return _answer_confirm(pending, key)
    return _answer_slot(pending, key, registry=registry, context=context)


def _answer_confirm(pending: PendingRequest, key: str) -> PendingAnswer:
    """Read «да» or «нет». Anything else is a new command, not a maybe.

    The «no» table is checked first because it and the cancel table overlap, and
    an ambiguous phrase should resolve to *not* doing the dangerous thing.
    """
    if key in NO_PHRASES:
        return PendingAnswer(status=AnswerStatus.DECLINED, request=pending)
    if key in YES_PHRASES:
        return PendingAnswer(
            status=AnswerStatus.CONFIRMED,
            request=pending,
            slots=dict(pending.known),
        )
    _log.debug("уточнение «%s» снято: пользователь сказал другое", pending.question)
    return PendingAnswer(status=AnswerStatus.NEW_COMMAND, request=pending)


def _answer_slot(
    pending: PendingRequest,
    key: str,
    *,
    registry: SlotTypeRegistry | None,
    context: SlotContext | None,
) -> PendingAnswer:
    """Parse an answer into the missing slot, or ask once more."""
    value: object | None = key
    slot_type = registry.get(pending.slot_type) if registry is not None else None
    if slot_type is not None:
        value = slot_type.safe_parse(key, context or SlotContext())
    if value is None:
        attempts = pending.attempts + 1
        if attempts >= MAX_CLARIFY_ATTEMPTS:
            _log.info("уточнение «%s» брошено после %d попыток", pending.question, attempts)
            return PendingAnswer(
                status=AnswerStatus.EXHAUSTED,
                request=pending,
                message=GAVE_UP,
            )
        return PendingAnswer(
            status=AnswerStatus.RETRY,
            request=pending,
            retry=replace(pending, attempts=attempts),
        )
    slots = {**pending.known, pending.slot: value}
    return PendingAnswer(status=AnswerStatus.FILLED, request=pending, slots=slots, value=value)


def publish_cancel(bus: EventBus, *, reason: str = CANCEL_REASON) -> None:
    """Announce a cancel and let everything that cares react.

    One line, but it is the seam the whole «Айрис, стоп» requirement hangs on:
    :class:`~ayris.audio.tts.service.TtsBridge` stops the player and clears its
    queue, :meth:`~ayris.nlu.context.DialogContext.cancel` drops the context, and
    the action runner of task 19 aborts what it is doing. None of them know about
    each other, and adding a fourth listener does not touch this function.
    """
    bus.publish(CancelRequested(reason=reason))
