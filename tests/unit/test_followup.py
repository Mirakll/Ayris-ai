"""Task 17: «отмена», «повтори», and the questions Ayris asks back.

This is the conversational half of the context task — the part that decides a
phrase is a move in a dialogue rather than a command, and the part that asks for
a slot the user did not say. The context itself is tested in
:mod:`tests.unit.test_context`; here the context is a prop, built with the same
hand-cranked clock so that «окно follow-up закрылось» is a statement about the
code rather than about how long the runner took.

Three things in this file are worth knowing before reading it.

The tables are matched on the *whole* phrase. «Стоп» is a cancel and «стоп
музыка» is a command about music, so :class:`TestCancelPhrases` spends most of
its length on phrases that must *not* be cancels — that is where the requirement
actually lives, and a test file that only checked the positive cases would pass
with a prefix match in place.

A repeat must not become the thing that gets repeated. :class:`TestRepeatAction`
checks the second «ещё раз» in a row, which is the shape the bug takes: it is
either a no-op or an infinite regress, and both look fine on the first call.

The clarification ladder is finite by design. :class:`TestSlotClarification`
walks it to the end — one question, one retry, then out loud — because the
failure mode of a wrong ladder is Ayris asking the same thing forever.

Groups:

* :class:`TestCancelPhrases` — what is a cancel, and what only contains one.
* :class:`TestRepeatMode` — which of the two repeats a phrase asked for.
* :class:`TestRepeatAnswer` — «повтори», and its fallback to the action.
* :class:`TestRepeatAction` — «ещё раз», the confirmation, and no regress.
* :class:`TestSlotQuestions` — the wording of a question about a slot.
* :class:`TestSlotClarification` — the ladder from question to giving up.
* :class:`TestConfirm` — «да», «нет», and something else entirely.
* :class:`TestScenarios` — the dialogues from the task, end to end.
* :class:`TestCancelSeam` — a cancel reaching the player and the context.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from ayris.audio.tts.service import connect_player
from ayris.core.events import CancelRequested, EventBus
from ayris.nlu.anaphora import resolve_anaphora
from ayris.nlu.context import (
    DEFAULT_FOLLOWUP_TTL,
    DEFAULT_PENDING_TTL,
    DialogContext,
    LastCommand,
    ObjectKind,
    PendingKind,
    PendingRequest,
)
from ayris.nlu.followup import (
    CANCEL_PHRASES,
    CANCEL_REASON,
    GAVE_UP,
    MAX_CLARIFY_ATTEMPTS,
    NOTHING_TO_REDO,
    NOTHING_TO_REPEAT,
    AnswerStatus,
    FollowUpKind,
    RepeatMode,
    answer_pending,
    ask_slot,
    confirm_command,
    is_cancel_phrase,
    publish_cancel,
    repeat_mode,
    resolve_followup,
    slot_question,
)
from ayris.nlu.matcher import Matcher, Trigger, TriggerKind
from ayris.nlu.slot_types import BuiltinSlotType, SlotContext, default_registry

if TYPE_CHECKING:
    from ayris.nlu.slot_types import SlotTypeRegistry

pytestmark = pytest.mark.unit

#: The moment every test starts at, matching :mod:`tests.unit.test_context`.
BASE_TIME: datetime = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


class Clock:
    """A clock that only moves when a test says so."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start if start is not None else BASE_TIME

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        """Move time forward and return the new reading."""
        self.now += timedelta(seconds=seconds)
        return self.now


@pytest.fixture
def clock() -> Clock:
    """A hand-cranked clock, at :data:`BASE_TIME`."""
    return Clock()


@pytest.fixture
def context(clock: Clock) -> DialogContext:
    """A context with no window and no persistence."""
    return DialogContext(clock=clock, window_probe=lambda: None, autosave=False)


@pytest.fixture
def registry() -> SlotTypeRegistry:
    """The built-in slot types, for parsing a clarification's answer."""
    return default_registry()


class TestCancelPhrases:
    """What counts as «stop», and — mostly — what does not."""

    @pytest.mark.parametrize(
        "phrase",
        ["отмена", "отмени", "стоп", "стой", "хватит", "прекрати", "не надо", "забудь"],
    )
    def test_a_cancel_word_on_its_own_is_a_cancel(self, phrase):
        assert is_cancel_phrase(phrase)

    @pytest.mark.parametrize(
        "phrase",
        ["Айрис, стоп", "СТОП", "стоп!", "отмена, пожалуйста", "ну хватит", "а стоп"],
    )
    def test_the_phrase_is_read_the_way_it_is_said(self, phrase):
        # Address, case, punctuation and politeness are not part of what was
        # asked. Normalisation removes the first three, ``_trim`` the last.
        assert is_cancel_phrase(phrase)

    @pytest.mark.parametrize(
        "phrase",
        [
            "стоп музыка",
            "стоп таймер",
            "останови музыку",
            "не надо музыку",
            "отмени будильник",
            "хватит музыки",
        ],
    )
    def test_a_phrase_that_means_something_more_specific_is_not_a_cancel(self, phrase):
        # The whole point of matching on equality. A prefix or containment test
        # would make every one of these impossible to say, and «стоп музыка» is
        # in the specification by name.
        assert not is_cancel_phrase(phrase)

    def test_an_empty_phrase_is_not_a_cancel(self):
        assert not is_cancel_phrase("")
        assert not is_cancel_phrase("   ")
        assert not is_cancel_phrase("пожалуйста")

    def test_the_table_is_written_in_the_shape_it_is_compared_against(self):
        # A phrase with «ё» or an uppercase letter in the table would never match
        # anything, silently.
        for phrase in CANCEL_PHRASES:
            assert phrase == phrase.lower()
            assert "ё" not in phrase
            assert is_cancel_phrase(phrase)


class TestRepeatMode:
    """Which of the two repeats a phrase is asking for."""

    @pytest.mark.parametrize(
        "phrase",
        ["повтори", "повтори ответ", "что ты сказала", "не расслышал", "повтори пожалуйста"],
    )
    def test_asking_to_hear_it_again(self, phrase):
        assert repeat_mode(phrase) is RepeatMode.ANSWER

    @pytest.mark.parametrize(
        "phrase",
        ["еще раз", "ещё раз", "давай еще раз", "снова", "то же самое", "повтори команду"],
    )
    def test_asking_to_do_it_again(self, phrase):
        assert repeat_mode(phrase) is RepeatMode.ACTION

    def test_the_more_specific_reading_wins(self):
        # «Повтори ещё раз» is in both tables if you read it by prefix. The user
        # said «ещё раз», so they mean the action.
        assert repeat_mode("повтори еще раз") is RepeatMode.ACTION

    @pytest.mark.parametrize("phrase", ["открой браузер", "повтори последнюю песню", "", "стоп"])
    def test_anything_else_is_not_a_repeat(self, phrase):
        assert repeat_mode(phrase) is None


class TestRepeatAnswer:
    """«Повтори»: the answer, the fallback, or an admission."""

    def test_the_last_answer_comes_back_ready_to_speak(self, context):
        context.remember_answer("сейчас двенадцать часов")
        result = resolve_followup("повтори", context.snapshot())
        assert result.kind is FollowUpKind.REPEAT_ANSWER
        assert result.speak == "сейчас двенадцать часов"
        assert result.handled
        assert result.answer is not None

    def test_with_no_answer_but_a_fresh_command_it_means_do_it_again(self, context):
        # «Открой браузер» says nothing out loud, so «повтори» right after it
        # plainly means «do it again» rather than «say nothing again».
        context.remember_command(intent="запуск", phrase="открой браузер")
        result = resolve_followup("повтори", context.snapshot())
        assert result.kind is FollowUpKind.REPEAT_ACTION
        assert result.command is not None
        assert result.command.phrase == "открой браузер"

    def test_the_fallback_is_refused_for_something_dangerous(self, context):
        # Guessing which of the two readings the user meant is not an acceptable
        # way to arrive at a shutdown.
        context.remember_command(intent="выключение", phrase="выключи компьютер", dangerous=True)
        result = resolve_followup("повтори", context.snapshot())
        assert result.kind is FollowUpKind.UNAVAILABLE
        assert result.speak == NOTHING_TO_REPEAT

    def test_with_nothing_to_repeat_it_says_so(self, context):
        result = resolve_followup("повтори", context.snapshot())
        assert result.kind is FollowUpKind.UNAVAILABLE
        assert result.speak == NOTHING_TO_REPEAT
        assert result.handled

    def test_an_answer_is_still_there_after_the_dialogue_closed(self, clock, context):
        # The two deadlines are the deeper reason the repeats are separate: an
        # answer outlives the follow-up window.
        context.remember_answer("сейчас двенадцать часов")
        clock.advance(DEFAULT_FOLLOWUP_TTL + 5)
        assert resolve_followup("повтори", context.snapshot()).kind is FollowUpKind.REPEAT_ANSWER


class TestRepeatAction:
    """«Ещё раз»: the command, a confirmation, and no regress."""

    def test_the_last_command_comes_back_with_its_slots(self, context):
        context.remember_command(
            intent="таймер", phrase="поставь таймер на 5 минут", slots={"duration": 300}
        )
        result = resolve_followup("еще раз", context.snapshot())
        assert result.kind is FollowUpKind.REPEAT_ACTION
        assert result.command is not None
        assert result.command.slots == {"duration": 300}

    def test_something_dangerous_is_confirmed_first(self, context):
        context.remember_command(intent="выключение", phrase="выключи компьютер", dangerous=True)
        result = resolve_followup("еще раз", context.snapshot())
        assert result.kind is FollowUpKind.CONFIRM
        assert result.speak == "Повторить «выключи компьютер»?"
        assert result.pending is not None
        assert result.pending.kind is PendingKind.CONFIRM

    def test_a_confirmation_carries_the_slots_across(self):
        # The point of a repeat is that the user does not say the slots again.
        command = LastCommand(intent="таймер", phrase="таймер", slots={"duration": 300})
        request = confirm_command(command)
        assert request.known == {"duration": 300}
        assert request.intent == "таймер"

    def test_a_confirmation_can_be_worded_by_the_caller(self):
        command = LastCommand(intent="выключение", phrase="выключи компьютер", dangerous=True)
        request = confirm_command(command, question="Точно выключить компьютер?")
        assert request.question == "Точно выключить компьютер?"

    def test_a_command_past_the_follow_up_window_is_not_repeated(self, clock, context):
        context.remember_command(intent="таймер", phrase="поставь таймер")
        clock.advance(DEFAULT_FOLLOWUP_TTL + 1)
        result = resolve_followup("еще раз", context.snapshot())
        assert result.kind is FollowUpKind.UNAVAILABLE
        assert result.speak == NOTHING_TO_REDO

    def test_with_nothing_done_yet_it_says_so(self, context):
        result = resolve_followup("еще раз", context.snapshot())
        assert result.kind is FollowUpKind.UNAVAILABLE
        assert result.speak == NOTHING_TO_REDO

    def test_a_repeat_is_never_the_thing_the_next_repeat_repeats(self, context):
        # The failure mode is either a no-op or an infinite regress, and both
        # look fine on the first call. The pipeline consults ``records_context``
        # instead of remembering unconditionally.
        context.remember_command(intent="таймер", phrase="поставь таймер на 5 минут")
        first = resolve_followup("еще раз", context.snapshot())
        assert not first.records_context

        second = resolve_followup("еще раз", context.snapshot())
        assert second.kind is FollowUpKind.REPEAT_ACTION
        assert second.command is not None
        assert second.command.phrase == "поставь таймер на 5 минут"

    def test_a_cancel_is_not_recorded_either(self, context):
        # A cancel that recorded itself would be revived by «ещё раз».
        result = resolve_followup("отмена", context.snapshot())
        assert result.kind is FollowUpKind.CANCEL
        assert not result.records_context
        assert result.handled

    def test_an_ordinary_command_is_left_for_the_matcher(self, context):
        result = resolve_followup("открой браузер", context.snapshot())
        assert result.kind is FollowUpKind.NONE
        assert not result.handled
        assert result.speak == ""


class TestSlotQuestions:
    """The wording of a question about a missing slot."""

    @pytest.mark.parametrize(
        ("slot_type", "question"),
        [
            (BuiltinSlotType.VOLUME, "Какую громкость поставить?"),
            (BuiltinSlotType.APP, "Какую программу?"),
            (BuiltinSlotType.DURATION, "На сколько времени?"),
            (BuiltinSlotType.TIME, "На какое время?"),
        ],
    )
    def test_the_question_follows_the_type(self, slot_type, question):
        # By type and not by name: the name is the user's and may be anything,
        # while the type is what says which answers are acceptable.
        assert slot_question("что-нибудь", slot_type.value) == question

    def test_an_unfamiliar_type_is_asked_about_by_name(self):
        # Clumsy but true, and better than a generic question that leaves the
        # user guessing which of the two holes Ayris means.
        assert slot_question("кому", "str") == "Уточни, пожалуйста: кому."
        assert slot_question("кому") == "Уточни, пожалуйста: кому."

    def test_every_built_in_type_has_something_to_say(self):
        # A type whose question came back as the unformatted template would ask
        # the user «Уточни, пожалуйста: {slot}.» out loud.
        for slot_type in BuiltinSlotType:
            question = slot_question("слот", slot_type.value)
            assert question.strip()
            assert "{" not in question
            assert question.endswith(("?", "."))

    def test_a_fresh_question_has_not_been_asked_yet(self):
        pending = ask_slot("громкость", slot_type=BuiltinSlotType.VOLUME.value)
        assert pending.kind is PendingKind.SLOT
        assert pending.attempts == 0
        assert pending.slot == "громкость"
        assert pending.slot_type == BuiltinSlotType.VOLUME.value

    def test_the_slots_that_did_fill_are_kept_on_the_request(self):
        # This is the only place they are remembered, and the answer is useless
        # without them: «поставь таймер» → «на сколько?» → «пять минут» has to
        # end up as one command.
        known: dict[str, Any] = {"label": "чай"}
        pending = ask_slot("длительность", intent="таймер", known=known)
        known["label"] = "кофе"
        assert pending.known == {"label": "чай"}

    def test_the_wording_can_be_overridden_by_the_caller(self):
        pending = ask_slot(
            "громкость", slot_type=BuiltinSlotType.VOLUME.value, question="Насколько?"
        )
        assert pending.question == "Насколько?"


class TestSlotClarification:
    """The ladder: one question, one retry, then out loud."""

    @staticmethod
    def volume(**kwargs: Any) -> PendingRequest:
        """A question about the volume, the type that parses most answers."""
        return ask_slot(
            "громкость", slot_type=BuiltinSlotType.VOLUME.value, intent="громкость", **kwargs
        )

    def test_the_answer_fills_the_slot(self, registry):
        result = answer_pending(self.volume(), "сорок", registry=registry)
        assert result.status is AnswerStatus.FILLED
        assert result.slots == {"громкость": 40}
        assert result.handled

    def test_what_already_filled_comes_back_with_it(self, registry):
        pending = ask_slot(
            "длительность",
            slot_type=BuiltinSlotType.DURATION.value,
            intent="таймер",
            known={"label": "чай"},
        )
        result = answer_pending(pending, "пять минут", registry=registry)
        assert result.status is AnswerStatus.FILLED
        assert result.slots == {"label": "чай", "длительность": timedelta(minutes=5)}

    def test_the_parser_gets_the_clock_it_was_handed(self, registry):
        # «В семь» is a time on a particular day, and which day depends on now.
        # The context travels from the caller through to the slot type.
        pending = ask_slot("время", slot_type=BuiltinSlotType.TIME.value, intent="будильник")
        result = answer_pending(
            pending, "в семь", registry=registry, context=SlotContext(now=BASE_TIME)
        )
        assert result.status is AnswerStatus.FILLED
        assert result.value == datetime(2026, 8, 12, 19, 0, tzinfo=UTC)

    def test_without_a_registry_the_normalised_words_are_the_value(self):
        # Right for a free-text slot, and the best available otherwise. The value
        # is a string, and it is the phrase after normalisation — which has
        # already turned «сорок» into digits, without knowing it is a volume.
        result = answer_pending(self.volume(), "сорок")
        assert result.status is AnswerStatus.FILLED
        assert result.slots == {"громкость": "40"}

    def test_an_answer_that_is_not_one_is_asked_about_again(self, registry):
        result = answer_pending(self.volume(), "какая-то чепуха", registry=registry)
        assert result.status is AnswerStatus.RETRY
        assert result.retry is not None
        assert result.retry.attempts == 1
        assert result.speak == "Какую громкость поставить?"
        assert result.handled

    def test_the_second_miss_ends_it_out_loud(self, registry):
        asked = answer_pending(self.volume(), "какая-то чепуха", registry=registry)
        assert asked.retry is not None
        result = answer_pending(asked.retry, "снова чепуха", registry=registry)
        assert result.status is AnswerStatus.EXHAUSTED
        assert result.speak == GAVE_UP
        assert result.retry is None
        assert result.handled

    def test_the_ladder_is_exactly_as_long_as_the_constant_says(self, registry):
        # Walked to the end rather than checked at one step: the failure mode of
        # a wrong ladder is Ayris asking the same thing forever.
        pending = self.volume()
        questions = 0
        for _ in range(10):
            result = answer_pending(pending, "не пойми что", registry=registry)
            if result.status is not AnswerStatus.RETRY:
                break
            assert result.retry is not None
            pending = result.retry
            questions += 1
        assert result.status is AnswerStatus.EXHAUSTED
        assert questions == MAX_CLARIFY_ATTEMPTS - 1

    def test_a_cancel_instead_of_an_answer_is_a_cancel(self, registry):
        pending = self.volume()
        result = answer_pending(pending, "отмена", registry=registry)
        assert result.status is AnswerStatus.CANCELLED
        assert result.request is pending
        assert result.handled

    def test_saying_nothing_leaves_the_question_where_it_was(self, registry):
        # An empty transcript is silence, not a wrong answer, and must not spend
        # one of the two attempts.
        pending = self.volume()
        result = answer_pending(pending, "   ", registry=registry)
        assert result.status is AnswerStatus.NOT_PENDING
        assert result.request is pending
        assert not result.handled

    def test_with_no_question_open_a_bare_word_is_nobody_s_answer(self, registry):
        result = answer_pending(None, "сорок", registry=registry)
        assert result.status is AnswerStatus.NOT_PENDING
        assert result.request is None
        assert not result.handled

    def test_a_question_that_timed_out_is_not_answered_later(self, clock, context, registry):
        # The expiry lives in the context, so the seam being checked here is that
        # ``pending()`` is what the pipeline asks — a stale question would eat a
        # phrase the user meant as a command.
        context.set_pending(self.volume())
        clock.advance(DEFAULT_PENDING_TTL + 1)
        assert context.pending() is None
        result = answer_pending(context.pending(), "сорок", registry=registry)
        assert result.status is AnswerStatus.NOT_PENDING


class TestConfirm:
    """«Да», «нет», and something else entirely."""

    @staticmethod
    def pending(**kwargs: Any) -> PendingRequest:
        """A confirmation of a shutdown, slots and all."""
        command = LastCommand(
            intent="выключение",
            phrase="выключи компьютер",
            slots={"delay": 60},
            dangerous=True,
            **kwargs,
        )
        return confirm_command(command)

    def test_yes_runs_it_with_the_slots_it_remembered(self):
        result = answer_pending(self.pending(), "да")
        assert result.status is AnswerStatus.CONFIRMED
        assert result.slots == {"delay": 60}
        assert result.handled

    @pytest.mark.parametrize("phrase", ["нет", "не нужно", "передумал", "не стоит"])
    def test_no_means_no(self, phrase):
        result = answer_pending(self.pending(), phrase)
        assert result.status is AnswerStatus.DECLINED
        assert result.slots is None
        assert result.handled

    def test_a_word_that_is_both_a_refusal_and_a_cancel_reads_as_a_cancel(self):
        # «Не надо» is in both tables. Both readings end in the same place —
        # nothing dangerous happens — and the cancel is checked first.
        result = answer_pending(self.pending(), "не надо")
        assert result.status is AnswerStatus.CANCELLED

    @pytest.mark.parametrize("phrase", ["ну да", "да, пожалуйста", "Ага!", "ОК"])
    def test_politeness_and_punctuation_do_not_change_the_answer(self, phrase):
        assert answer_pending(self.pending(), phrase).status is AnswerStatus.CONFIRMED

    def test_something_else_entirely_drops_the_question(self):
        # Not a maybe: the user moved on, and the phrase still has to be matched
        # as a command, which is why this one is not ``handled``.
        pending = self.pending()
        result = answer_pending(pending, "открой браузер")
        assert result.status is AnswerStatus.NEW_COMMAND
        assert result.request is pending
        assert not result.handled

    def test_a_long_agreement_is_a_new_command(self):
        # The «yes» table is short on purpose: a sentence answering a yes/no
        # question is more likely a command than an elaborate agreement.
        result = answer_pending(self.pending(), "да открой браузер")
        assert result.status is AnswerStatus.NEW_COMMAND


class TestScenarios:
    """The dialogues from the task, one utterance at a time.

    Each of these is what the pipeline of task 18 will do in a loop: ask the
    context what is outstanding, read the phrase as an answer, then as a
    conversational move, then — and only then — match it. The helpers below are
    that loop with everything else left out.
    """

    @staticmethod
    def matcher() -> Matcher:
        return Matcher.from_triggers(
            [
                Trigger(id=1, command_id=10, pattern="открой браузер"),
                Trigger(id=2, command_id=20, pattern="закрой браузер"),
                Trigger(id=3, command_id=30, pattern="который час"),
                Trigger(
                    id=4,
                    command_id=40,
                    pattern="поставь таймер на {длительность:duration}",
                    kind=TriggerKind.TEMPLATE,
                ),
            ]
        )

    @staticmethod
    def say(context: DialogContext, phrase: str, registry: SlotTypeRegistry) -> Any:
        """One turn of the dialogue, in the order the pipeline uses.

        Returns whatever the phrase turned out to be: a :class:`PendingAnswer`, a
        :class:`FollowUp`, or a match. ``touch`` comes *last*, and that ordering
        is the point: it is what keeps the follow-up window open for the next
        utterance, so touching before reading the phrase would reopen a window
        that had already closed and make a stale «ещё раз» work forever.
        """
        answer = answer_pending(context.pending(), phrase, registry=registry)
        if answer.handled:
            context.clear_pending()
            context.touch()
            return answer
        follow_up = resolve_followup(phrase, context.snapshot())
        if follow_up.handled:
            context.touch()
            return follow_up
        rewritten = resolve_anaphora(phrase, context.snapshot())
        match = TestScenarios.matcher().match(rewritten.text, context=context.snapshot())
        context.touch()
        return match

    def test_open_the_browser_then_close_it(self, context, registry):
        # «Закрой его» is not in the library and never will be: it is «закрой
        # браузер» once the context has said what «его» is.
        match = self.say(context, "открой браузер", registry)
        assert match is not None
        assert match.command_id == 10
        context.remember_command(intent="запуск", phrase="открой браузер", command_id=10)
        context.remember_object(ObjectKind.APP, "браузер")

        match = self.say(context, "закрой его", registry)
        assert match is not None
        assert match.command_id == 20

    def test_a_timer_and_then_never_mind(self, context, registry):
        match = self.say(context, "поставь таймер на 5 минут", registry)
        assert match is not None
        assert match.command_id == 40
        context.remember_command(
            intent="таймер",
            phrase="поставь таймер на 5 минут",
            slots={"длительность": timedelta(minutes=5)},
            command_id=40,
        )

        cancelled = self.say(context, "отмена", registry)
        assert cancelled.kind is FollowUpKind.CANCEL
        # The caller does the two things a cancel means, and after them «ещё раз»
        # has nothing to revive.
        context.cancel(reason=CANCEL_REASON)
        assert self.say(context, "еще раз", registry).kind is FollowUpKind.UNAVAILABLE

    def test_what_time_is_it_then_say_that_again(self, context, registry):
        match = self.say(context, "который час", registry)
        assert match is not None
        assert match.command_id == 30
        context.remember_command(intent="время", phrase="который час", command_id=30)
        context.remember_answer("сейчас двенадцать часов")

        repeated = self.say(context, "повтори", registry)
        assert repeated.kind is FollowUpKind.REPEAT_ANSWER
        assert repeated.speak == "сейчас двенадцать часов"

    def test_the_missing_slot_is_asked_about_and_then_given(self, context, registry):
        # «Поставь таймер» without a duration matches nothing, so the pipeline
        # asks; the answer is a bare noun that matches nothing either, and only
        # the outstanding question makes it mean anything.
        assert self.say(context, "поставь таймер", registry) is None
        context.set_pending(
            ask_slot(
                "длительность",
                slot_type=BuiltinSlotType.DURATION.value,
                intent="таймер",
                command_id=40,
            )
        )
        assert context.pending() is not None
        assert context.pending().question == "На сколько времени?"

        answer = self.say(context, "пять минут", registry)
        assert answer.status is AnswerStatus.FILLED
        assert answer.slots == {"длительность": timedelta(minutes=5)}
        assert context.pending() is None

    def test_a_cancel_in_the_middle_of_the_clarification(self, context, registry):
        context.set_pending(
            ask_slot("громкость", slot_type=BuiltinSlotType.VOLUME.value, intent="громкость")
        )
        answer = self.say(context, "отмена", registry)
        assert answer.status is AnswerStatus.CANCELLED
        assert context.pending() is None
        # And the phrase never reached the matcher, so nothing was set to 40 or
        # anything else on the way out.
        assert answer.slots is None

    def test_a_command_said_in_the_middle_of_the_clarification_wins(self, context, registry):
        context.set_pending(confirm_command(LastCommand(intent="выключение", phrase="выключи")))
        match = self.say(context, "который час", registry)
        # NEW_COMMAND is not ``handled``, so ``say`` carried on to the matcher.
        assert match is not None
        assert match.command_id == 30

    def test_the_dialogue_closes_on_its_own(self, clock, context, registry):
        context.remember_command(intent="запуск", phrase="открой браузер", command_id=10)
        context.remember_object(ObjectKind.APP, "браузер")
        clock.advance(DEFAULT_FOLLOWUP_TTL + 1)

        # Anaphora outlives the follow-up window — the object has its own, longer
        # deadline — but the action does not.
        assert self.say(context, "еще раз", registry).kind is FollowUpKind.UNAVAILABLE
        match = self.say(context, "закрой его", registry)
        assert match is not None
        assert match.command_id == 20


class TestCancelSeam:
    """What «отмена» reaches: the player, and the context."""

    class FakePlayer:
        """Enough of a player for the bridge: it counts being stopped."""

        def __init__(self) -> None:
            self.cancels = 0
            self.observers: tuple[Any, Any] = (None, None)

        def cancel(self) -> bool:
            self.cancels += 1
            return True

        def set_observers(self, *, on_started: Any = None, on_finished: Any = None) -> None:
            self.observers = (on_started, on_finished)

    @staticmethod
    def make_bus() -> EventBus:
        return EventBus(thread_id=None)

    def test_the_reason_says_it_was_spoken(self):
        # The hotkey and the overlay's button publish the same event; the log has
        # to tell them apart.
        bus = self.make_bus()
        seen: list[CancelRequested] = []
        bus.subscribe(CancelRequested, seen.append)
        publish_cancel(bus)
        assert len(seen) == 1
        assert seen[0].reason == CANCEL_REASON

    def test_a_caller_can_name_its_own_reason(self):
        bus = self.make_bus()
        seen: list[CancelRequested] = []
        bus.subscribe(CancelRequested, seen.append)
        publish_cancel(bus, reason="горячая клавиша")
        assert seen[0].reason == "горячая клавиша"

    def test_the_assistant_stops_talking(self):
        # Assertions on the fake, not on the speakers: whether anything is
        # audible is a hardware question and is not asked in CI.
        bus = self.make_bus()
        player = self.FakePlayer()
        cancelled_synthesis: list[bool] = []
        bridge = connect_player(bus, player, on_cancel=lambda: cancelled_synthesis.append(True))
        try:
            publish_cancel(bus)
            assert player.cancels == 1
            # The worker is told too: a sentence it finishes after the queue was
            # cleared would be played by the next request.
            assert cancelled_synthesis == [True]
        finally:
            bridge.close()

    def test_the_context_and_the_player_do_not_know_about_each_other(self, context):
        # One event, two independent listeners. Adding a third — the action
        # runner of task 19 — touches neither.
        bus = self.make_bus()
        player = self.FakePlayer()
        bridge = connect_player(bus, player)
        context.attach(bus)
        context.remember_command(intent="таймер", phrase="поставь таймер")
        context.set_pending(ask_slot("громкость", slot_type=BuiltinSlotType.VOLUME.value))
        try:
            publish_cancel(bus)
            assert player.cancels == 1
            snapshot = context.snapshot()
            assert snapshot.command is None
            assert snapshot.pending is None
        finally:
            context.detach()
            bridge.close()

    def test_after_the_bridge_is_closed_the_player_is_left_alone(self):
        bus = self.make_bus()
        player = self.FakePlayer()
        bridge = connect_player(bus, player)
        bridge.close()
        publish_cancel(bus)
        assert player.cancels == 0

    @pytest.mark.hardware
    def test_the_room_goes_quiet(self):
        """«Айрис, стоп» while a phrase is sounding, on real speakers.

        Not run in CI and not runnable in the sandbox: the null ALSA device has
        no timing, so «interrupted mid-word» is not a thing that can happen here.
        Left as an instruction rather than a skip so that the manual check has
        somewhere to live.
        """
        pytest.skip("проверяется вручную на машине с колонками")
