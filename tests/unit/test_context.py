"""Task 17: what the assistant remembers, and what «его» turns out to mean.

Every test here injects the clock and the window probe. That is not tidiness: the
context is built entirely out of deadlines, and a test that waited for a real
thirty seconds would be both slow and flaky, while one that asserted on
`time.sleep(0.01)` would assert on nothing. `Clock` below moves time by hand, so
«TTL истёк» is a statement about the code rather than about the CI runner's load.

The window probe is the other injection, and it is what makes contextual triggers
testable at all off Windows. `get_active_window` is the single seam the WinAPI
lives behind; substituting a `WindowInfo` gives the filters something to match
without a foreground window existing. The one test that calls the real thing is
skipped everywhere but win32 — see :class:`TestWindowProbe`.

Groups:

* :class:`TestTimeOfDay` — the Russian split of the day, and its labels.
* :class:`TestGender` — guessing agreement from a noun's ending.
* :class:`TestWindowInfo` — glob and substring matching over title and process.
* :class:`TestWindowProbe` — the cache, its invalidation, and the real call.
* :class:`TestObjects` — what «его» can reach, and for how long.
* :class:`TestCommand` — the last command and the follow-up window.
* :class:`TestAnswer` — the last thing said, on its own longer deadline.
* :class:`TestPending` — an outstanding question, and its timeout.
* :class:`TestCancel` — «отмена» and a profile switch, and what each keeps.
* :class:`TestVariables` — what a `when_variable` condition reads.
* :class:`TestTtl` — the four deadlines, read out of the «Команды» tab.
* :class:`TestPersistence` — what survives a restart, and what must not.
* :class:`TestBus` — the cancel and the spoken answer, arriving as events.
* :class:`TestPronouns` — which words carry a reference, and where.
* :class:`TestAnaphora` — «закрой его» → «закрой гугл хром», and the question.
* :class:`TestHourRange` — `9-18`, and the night shift through midnight.
* :class:`TestVariableCondition` — one test against one profile variable.
* :class:`TestConditions` — a whole condition set, and its complaints.
* :class:`TestPredicate` — what the matcher filters candidates with.
* :class:`TestConditionalMatching` — one phrase, two windows, two commands.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from ayris.core.config import CommandsConfig
from ayris.core.database import Database, reset_database
from ayris.core.events import CancelRequested, EventBus, TtsStarted
from ayris.core.repositories import Repositories
from ayris.nlu.anaphora import (
    PLACE_KINDS,
    QUESTION_OBJECT,
    QUESTION_PLACE,
    AnaphoraStatus,
    find_mention,
    is_pronoun,
    resolve_anaphora,
    resolve_reference,
)
from ayris.nlu.context import (
    CONTEXT_VARIABLE,
    DEFAULT_ANSWER_TTL,
    DEFAULT_FOLLOWUP_TTL,
    DEFAULT_OBJECT_TTL,
    DEFAULT_PENDING_TTL,
    MAX_OBJECTS,
    ContextSnapshot,
    ContextTtl,
    DialogContext,
    Gender,
    MemoryContextStore,
    ObjectKind,
    PendingKind,
    PendingRequest,
    TimeOfDay,
    VariableContextStore,
    WindowInfo,
    get_active_window,
    guess_gender,
    invalidate_active_window,
)
from ayris.nlu.matcher import Matcher, Trigger
from ayris.nlu.trigger_filters import (
    CONDITION_KEYS,
    UNCONDITIONAL,
    HourRange,
    TriggerConditions,
    VariableCondition,
    VariableTest,
    conditions_from_payload,
    context_predicate,
    describe_conditions,
    validate_conditions,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ayris.core.models import JsonObject
    from ayris.nlu.index import IndexedTrigger

pytestmark = pytest.mark.unit

#: The moment every test starts at. Noon UTC, so that a local-hour assertion is
#: the same on a runner in any timezone this project cares about.
BASE_TIME: datetime = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


class Clock:
    """A clock that only moves when a test says so.

    Handed to :class:`~ayris.nlu.context.DialogContext` as ``clock``. Wall-clock
    rather than monotonic on purpose: the context outlives the process and stores
    absolute timestamps, so the injected clock has to be of the same kind.
    """

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
    """A context with no window and no persistence — the shape most tests want.

    ``window_probe`` returns ``None`` rather than being left at its default: the
    default reads the real foreground window, which on a Windows runner would make
    every snapshot depend on whatever happened to be in front.
    """
    return DialogContext(clock=clock, window_probe=lambda: None, autosave=False)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """An open, migrated database on disk, for the persistence group."""
    handle = Database.open(tmp_path / "ayris.db")
    yield handle
    handle.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    """Repositories over the test database."""
    return Repositories(database)


def local_time(hour: int) -> datetime:
    """An instant that reads as ``hour`` o'clock on this machine.

    The context stores UTC and the conditions are about the user's clock, so a
    test that wrote ``12:00 UTC`` and expected «день» would pass in London and
    fail in Vladivostok. Building the moment out of a local hour keeps the
    assertion about the code.
    """
    return datetime(2026, 8, 12, hour, 0).astimezone()


@dataclass(frozen=True, slots=True)
class FakeEntry:
    """The one field of an index entry a condition predicate reads.

    Standing in for :class:`~ayris.nlu.index.IndexedTrigger` so that this file
    does not have to build an index to check a predicate.
    """

    conditions: TriggerConditions


class TestTimeOfDay:
    """The four parts of the day, on the split Russian greetings assume."""

    @pytest.mark.parametrize(
        ("hour", "expected"),
        [
            (5, TimeOfDay.MORNING),
            (10, TimeOfDay.MORNING),
            (11, TimeOfDay.DAY),
            (16, TimeOfDay.DAY),
            (17, TimeOfDay.EVENING),
            (22, TimeOfDay.EVENING),
            (23, TimeOfDay.NIGHT),
            (3, TimeOfDay.NIGHT),
        ],
    )
    def test_an_hour_lands_in_its_part_of_the_day(self, hour, expected):
        assert TimeOfDay.from_hour(hour) is expected

    def test_every_part_of_the_day_has_a_russian_label(self):
        # The labels go into «доброе утро» and into the trigger editor, so a
        # missing one would be visible to the user rather than merely wrong.
        assert [item.label for item in TimeOfDay] == ["ночь", "утро", "день", "вечер"]


class TestGender:
    """Guessing agreement from an ending, well enough to pick between two names."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("браузер", Gender.MASCULINE),
            ("гугл хром", Gender.MASCULINE),
            ("папка", Gender.FEMININE),
            ("музыка", Gender.FEMININE),
            ("тетрадь", Gender.FEMININE),
            ("фото", Gender.NEUTER),
            ("файлы", Gender.PLURAL),
            ("документы", Gender.PLURAL),
        ],
    )
    def test_the_ending_decides(self, name, expected):
        assert guess_gender(name) is expected

    def test_the_last_word_is_the_one_that_counts(self):
        # «гугл хром» is masculine because «хром» is; the first word says nothing
        # about how the phrase agrees.
        assert guess_gender("гугл таблица") is Gender.FEMININE

    def test_a_short_word_ending_in_i_is_not_plural(self):
        # «три» is not a plural noun, and the length floor is what stops the
        # guess from turning every short word into one.
        assert guess_gender("три") is Gender.MASCULINE


class TestWindowInfo:
    """What a `when_window` condition is matched against."""

    @pytest.fixture
    def window(self) -> WindowInfo:
        return WindowInfo(title="Adobe Photoshop 2024 - фото.psd", process="Photoshop.exe")

    @pytest.mark.parametrize(
        "pattern",
        ["*Photoshop*", "*photoshop*", "photoshop", "Photoshop.exe", "*.psd", "фото"],
    )
    def test_a_pattern_that_describes_the_window(self, window, pattern):
        assert window.matches(pattern)

    @pytest.mark.parametrize("pattern", ["word", "*Illustrator*", "notepad.exe"])
    def test_a_pattern_that_does_not(self, window, pattern):
        assert not window.matches(pattern)

    def test_matching_covers_the_process_as_well_as_the_title(self):
        # An elevated window returns no title, and the process name is then the
        # only thing a condition can be written against.
        info = WindowInfo(title="", process="WINWORD.EXE")
        assert info.matches("winword.exe")
        assert not info.matches("*Photoshop*")

    def test_an_empty_pattern_matches_anything(self):
        # A condition the user left blank is not a condition, and refusing every
        # window instead would silently disable their trigger.
        assert WindowInfo(title="что угодно").matches("")

    def test_a_reading_with_nothing_in_it_is_empty(self):
        assert WindowInfo().is_empty
        assert not WindowInfo(title="Блокнот").is_empty
        assert not WindowInfo(process="notepad.exe").is_empty

    def test_as_json_carries_the_three_fields(self, window):
        assert window.as_json() == {
            "title": "Adobe Photoshop 2024 - фото.psd",
            "process": "Photoshop.exe",
            "pid": 0,
        }


class TestWindowProbe:
    """The cache in front of the WinAPI call, and the call itself."""

    @pytest.fixture(autouse=True)
    def _clean_cache(self) -> Iterator[None]:
        # The cache is module state, so every test in this class starts and ends
        # with it empty. Otherwise the first test's reading leaks into the second.
        invalidate_active_window()
        yield
        invalidate_active_window()

    def test_a_second_reading_inside_the_ttl_costs_no_probe(self, monkeypatch):
        # The matcher asks for the window per utterance and the filters per
        # candidate; without the cache a library of a thousand triggers would
        # make a thousand WinAPI calls for one phrase.
        calls = []

        def probe() -> WindowInfo:
            calls.append(1)
            return WindowInfo(title=f"окно {len(calls)}", process="p.exe")

        monkeypatch.setattr("ayris.nlu.context._probe_active_window", probe)
        first = get_active_window(ttl=100.0)
        second = get_active_window(ttl=100.0)
        assert first == second
        assert len(calls) == 1

    def test_invalidating_forces_the_next_reading(self, monkeypatch):
        calls = []

        def probe() -> WindowInfo:
            calls.append(1)
            return WindowInfo(title=f"окно {len(calls)}", process="p.exe")

        monkeypatch.setattr("ayris.nlu.context._probe_active_window", probe)
        assert get_active_window(ttl=100.0) is not None
        invalidate_active_window()
        second = get_active_window(ttl=100.0)
        assert len(calls) == 2
        assert second is not None
        assert second.title == "окно 2"

    def test_a_zero_ttl_probes_every_time(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ayris.nlu.context._probe_active_window",
            lambda: (calls.append(1), WindowInfo(title="окно"))[1],
        )
        get_active_window(ttl=0.0)
        get_active_window(ttl=0.0)
        assert len(calls) == 2

    def test_a_probe_that_finds_nothing_is_cached_too(self, monkeypatch):
        # A full-screen application with no readable title gives ``None``, and
        # re-asking about it every candidate would cost the same as asking about
        # a real window.
        calls = []
        monkeypatch.setattr(
            "ayris.nlu.context._probe_active_window",
            lambda: (calls.append(1), None)[1],
        )
        assert get_active_window(ttl=100.0) is None
        assert get_active_window(ttl=100.0) is None
        assert len(calls) == 1

    def test_the_probe_never_raises_off_windows(self):
        # ``_probe_active_window`` starts by checking the platform, so on Linux
        # this is the whole of its behaviour — and the filters have to survive it.
        if sys.platform == "win32":  # pragma: no cover - covered by the test below
            pytest.skip("на Windows опрос возвращает настоящее окно")
        assert get_active_window(ttl=0.0) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="WinAPI есть только на Windows")
    def test_a_real_foreground_window_is_read(self):  # pragma: no cover - windows only
        # On a CI runner the foreground window may be nothing at all, so the
        # assertion is about the shape of the answer, not about which window it
        # is. What this pins down is that the ctypes signatures are right.
        info = get_active_window(ttl=0.0)
        if info is None:
            pytest.skip("на раннере нет окна на переднем плане")
        assert not info.is_empty
        assert isinstance(info.title, str)
        assert isinstance(info.process, str)
        assert info.pid >= 0


class TestObjects:
    """What was mentioned, in what order, and until when."""

    def test_a_mention_becomes_the_object_a_pronoun_reaches(self, context):
        item = context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        assert item is not None
        snapshot = context.snapshot()
        assert snapshot.last_object == item
        assert snapshot.object_of((ObjectKind.APP,)) == item

    def test_the_machine_value_and_the_spoken_name_are_both_kept(self, context):
        # The name is what goes back into a rewritten phrase, the value is what
        # an action runs on. Losing either would break one of the two.
        item = context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        assert item is not None
        assert item.name == "гугл хром"
        assert item.target == "chrome"

    def test_an_object_with_no_value_targets_its_own_name(self, context):
        item = context.remember_object(ObjectKind.APP, "браузер")
        assert item is not None
        assert item.target == "браузер"

    def test_a_nameless_object_is_refused_rather_than_stored(self, context):
        assert context.remember_object(ObjectKind.APP, "   ") is None
        assert context.snapshot().objects == ()

    def test_mentioning_the_same_thing_twice_moves_it_to_the_front(self, context):
        # Otherwise a list of eight holds one thing eight times, and «его» stops
        # being able to reach anything else the user talked about.
        context.remember_object(ObjectKind.APP, "браузер", "chrome")
        context.remember_object(ObjectKind.FILE, "отчёт", "C:/отчёт.txt")
        context.remember_object(ObjectKind.APP, "браузер", "chrome")
        objects = context.snapshot().objects
        assert [item.name for item in objects] == ["браузер", "отчёт"]

    def test_the_same_name_under_a_different_kind_is_a_different_thing(self, context):
        context.remember_object(ObjectKind.APP, "хром", "chrome")
        context.remember_object(ObjectKind.WINDOW, "хром", "chrome")
        assert len(context.snapshot().objects) == 2

    def test_only_the_last_few_mentions_are_kept(self, context):
        for number in range(MAX_OBJECTS + 3):
            context.remember_object(ObjectKind.FILE, f"файл {number}", f"C:/{number}.txt")
        objects = context.snapshot().objects
        assert len(objects) == MAX_OBJECTS
        assert objects[0].name == f"файл {MAX_OBJECTS + 2}"

    def test_object_of_narrows_by_kind(self, context):
        context.remember_object(ObjectKind.APP, "браузер")
        context.remember_object(ObjectKind.URL, "ютуб", "https://youtube.com")
        snapshot = context.snapshot()
        found = snapshot.object_of((ObjectKind.APP,))
        assert found is not None
        assert found.name == "браузер"

    def test_object_of_prefers_the_asked_gender_but_does_not_require_it(self, context):
        # «закрой её» about «программа» is a thing people say, and a filter would
        # answer nothing where a preference answers the likely thing.
        context.remember_object(ObjectKind.APP, "браузер")
        context.remember_object(ObjectKind.APP, "музыка")
        snapshot = context.snapshot()
        masculine = snapshot.object_of(gender=Gender.MASCULINE)
        feminine = snapshot.object_of(gender=Gender.FEMININE)
        neuter = snapshot.object_of(gender=Gender.NEUTER)
        assert masculine is not None
        assert feminine is not None
        assert neuter is not None
        assert masculine.name == "браузер"
        assert feminine.name == "музыка"
        # Nothing is neuter, so recency decides and the answer is still usable.
        assert neuter.name == "музыка"

    def test_an_object_of_a_kind_never_mentioned_is_not_invented(self, context):
        context.remember_object(ObjectKind.APP, "браузер")
        assert context.snapshot().object_of((ObjectKind.URL,)) is None

    def test_a_mention_older_than_its_ttl_stops_being_reachable(self, clock, context):
        context.remember_object(ObjectKind.APP, "браузер")
        clock.advance(DEFAULT_OBJECT_TTL + 1)
        assert context.snapshot().objects == ()

    def test_a_ttl_of_zero_keeps_mentions_forever(self, clock):
        # ``0`` switches the deadline off rather than expiring everything at once,
        # which is the reading a user setting the field to zero expects.
        context = DialogContext(
            ttl=ContextTtl(objects=0.0),
            clock=clock,
            window_probe=lambda: None,
            autosave=False,
        )
        context.remember_object(ObjectKind.APP, "браузер")
        clock.advance(10_000)
        assert len(context.snapshot().objects) == 1


class TestCommand:
    """The last command, and the window in which it can still be repeated."""

    def test_a_command_is_remembered_with_its_slots(self, context):
        command = context.remember_command(
            intent="таймер",
            phrase="поставь таймер на 5 минут",
            slots={"duration": 300},
        )
        snapshot = context.snapshot()
        assert snapshot.command == command
        assert snapshot.command is not None
        assert snapshot.command.slots == {"duration": 300}
        assert snapshot.can_repeat_action

    def test_the_title_falls_back_from_phrase_to_intent_to_action(self, context):
        spoken = context.remember_command(
            intent="таймер", action="timer.set", phrase="поставь таймер"
        )
        assert spoken.title == "поставь таймер"
        without_phrase = context.remember_command(intent="таймер", action="timer.set")
        assert without_phrase.title == "таймер"
        bare = context.remember_command(action="timer.set")
        assert bare.title == "timer.set"

    def test_speaking_keeps_the_follow_up_window_open(self, clock, context):
        context.remember_command(intent="таймер", phrase="таймер")
        clock.advance(DEFAULT_FOLLOWUP_TTL - 1)
        context.touch()
        clock.advance(DEFAULT_FOLLOWUP_TTL - 1)
        assert context.snapshot().follow_up_active

    def test_silence_past_the_follow_up_ttl_closes_the_window(self, clock, context):
        context.remember_command(intent="таймер", phrase="таймер")
        clock.advance(DEFAULT_FOLLOWUP_TTL + 1)
        snapshot = context.snapshot()
        assert not snapshot.follow_up_active
        assert snapshot.command is None
        assert not snapshot.can_repeat_action

    def test_a_misheard_phrase_still_counts_as_the_user_talking(self, clock, context):
        # ``touch`` is called for every recognised utterance, matched or not.
        # Letting the window close under a failed command is how «закрой его»
        # stops working right after the assistant misheard something.
        context.remember_command(intent="таймер", phrase="таймер")
        clock.advance(DEFAULT_FOLLOWUP_TTL - 5)
        context.touch()
        clock.advance(10)
        assert context.snapshot().follow_up_active

    def test_a_dangerous_command_says_so(self, context):
        command = context.remember_command(intent="выключение", dangerous=True)
        assert command.dangerous
        snapshot = context.snapshot()
        assert snapshot.command is not None
        assert snapshot.command.dangerous


class TestAnswer:
    """The last thing said out loud, on a deadline of its own."""

    def test_an_answer_is_remembered_for_repeating(self, context):
        answer = context.remember_answer("сейчас двенадцать часов")
        assert answer is not None
        snapshot = context.snapshot()
        assert snapshot.answer == answer
        assert snapshot.can_repeat_answer

    def test_an_empty_answer_is_not_worth_remembering(self, context):
        assert context.remember_answer("   ") is None
        assert context.snapshot().answer is None

    def test_the_assistant_talking_does_not_extend_the_follow_up_window(self, clock, context):
        # A long answer should not keep the dialogue open longer than a short one:
        # the window is about the user's attention, not about Ayris's.
        context.touch()
        clock.advance(DEFAULT_FOLLOWUP_TTL - 1)
        context.remember_answer("очень длинный ответ")
        clock.advance(2)
        assert not context.snapshot().follow_up_active

    def test_an_answer_outlives_the_follow_up_window(self, clock, context):
        # «повтори» half a minute later is a normal thing to say, so the answer's
        # deadline is the long one and it is not tied to the command's.
        context.remember_answer("сейчас двенадцать часов")
        clock.advance(DEFAULT_FOLLOWUP_TTL + 5)
        assert context.snapshot().can_repeat_answer

    def test_an_answer_older_than_its_ttl_is_forgotten(self, clock, context):
        context.remember_answer("сейчас двенадцать часов")
        clock.advance(DEFAULT_ANSWER_TTL + 1)
        assert context.snapshot().answer is None


class TestPending:
    """A question that was asked out loud, and the answer that is due next."""

    @staticmethod
    def request(question: str = "Какую громкость поставить?") -> PendingRequest:
        return PendingRequest(
            kind=PendingKind.SLOT,
            question=question,
            slot="volume",
            slot_type="volume",
        )

    def test_a_question_is_outstanding_until_it_is_answered(self, context):
        context.set_pending(self.request())
        assert context.pending() is not None
        assert context.snapshot().pending is not None

    def test_the_request_is_stamped_with_the_context_clock(self, clock, context):
        # The caller builds a ``PendingRequest`` without a clock of its own, so
        # the timeout would otherwise be measured from the real time while the
        # rest of the context ran on the injected one.
        stamped = context.set_pending(self.request())
        assert stamped.at == clock.now

    def test_a_question_nobody_answered_times_out(self, clock, context):
        context.set_pending(self.request())
        clock.advance(DEFAULT_PENDING_TTL + 1)
        assert context.pending() is None
        assert context.snapshot().pending is None

    def test_clearing_returns_what_was_dropped(self, context):
        context.set_pending(self.request())
        dropped = context.clear_pending()
        assert dropped is not None
        assert dropped.slot == "volume"
        assert context.pending() is None

    def test_clearing_nothing_is_not_an_error(self, context):
        assert context.clear_pending() is None

    def test_asking_a_question_keeps_the_dialogue_alive(self, context):
        # Ayris asking counts as the dialogue continuing: the user's answer must
        # not arrive to find the follow-up window already shut.
        assert not context.snapshot().follow_up_active
        context.set_pending(self.request())
        assert context.snapshot().follow_up_active


class TestCancel:
    """What «отмена» drops, and what a profile switch drops on top of that."""

    def test_a_cancel_drops_everything_that_could_still_act(self, context):
        context.remember_command(intent="таймер", phrase="поставь таймер")
        context.remember_object(ObjectKind.APP, "браузер")
        context.set_pending(TestPending.request())
        context.cancel(reason="тест")
        snapshot = context.snapshot()
        assert snapshot.command is None
        assert snapshot.objects == ()
        assert snapshot.pending is None
        assert not snapshot.follow_up_active

    def test_a_cancel_keeps_the_last_answer(self, context):
        # «Отмена» stops an action; it does not mean the user no longer wants to
        # hear what was said a moment ago.
        context.remember_answer("сейчас двенадцать часов")
        context.cancel()
        snapshot = context.snapshot()
        assert snapshot.answer is not None
        assert snapshot.can_repeat_answer

    def test_a_reset_forgets_the_answer_too(self, context):
        context.remember_answer("сейчас двенадцать часов")
        context.remember_object(ObjectKind.APP, "браузер")
        context.set_variable("режим", "работа")
        context.reset(profile_id=3)
        snapshot = context.snapshot()
        assert snapshot.answer is None
        assert snapshot.objects == ()
        assert snapshot.variables == {}
        assert context.profile_id == 3


class TestVariables:
    """What a `when_variable` condition and a message template read."""

    def test_a_variable_is_visible_in_the_snapshot(self, context):
        context.set_variable("режим", "работа")
        snapshot = context.snapshot()
        assert snapshot.variables == {"режим": "работа"}
        assert snapshot.variable("режим") == "работа"

    def test_a_name_is_found_whatever_its_case(self, context):
        # Conditions are typed by hand in the command editor, and «Режим» there
        # against «режим» here is not a mistake worth failing over.
        context.set_variable("Режим", "работа")
        assert context.snapshot().variable("режим") == "работа"

    def test_an_unset_variable_answers_the_default(self, context):
        assert context.snapshot().variable("нет такой", "по умолчанию") == "по умолчанию"
        assert context.snapshot().variable("нет такой") is None

    def test_loading_a_profile_replaces_the_whole_set(self, context):
        context.set_variable("режим", "работа")
        context.load_variables({"режим": "отдых", "громкость": 40})
        assert context.snapshot().variables == {"режим": "отдых", "громкость": 40}

    def test_the_snapshot_does_not_share_the_live_dictionary(self, context):
        # A snapshot is handed around and read from other threads, so it has to
        # be a copy — otherwise a variable set mid-match changes the past.
        context.set_variable("режим", "работа")
        snapshot = context.snapshot()
        context.set_variable("режим", "отдых")
        assert snapshot.variables == {"режим": "работа"}


class TestTtl:
    """The four deadlines, and where the user sets them."""

    def test_the_defaults_are_the_documented_ones(self):
        ttl = ContextTtl()
        assert ttl.followup == DEFAULT_FOLLOWUP_TTL
        assert ttl.objects == DEFAULT_OBJECT_TTL
        assert ttl.answer == DEFAULT_ANSWER_TTL
        assert ttl.pending == DEFAULT_PENDING_TTL

    def test_the_deadlines_come_out_of_the_commands_tab(self):
        commands = CommandsConfig(
            followup_ttl_s=15.0,
            object_ttl_s=90.0,
            answer_ttl_s=300.0,
            clarify_timeout_s=10.0,
        )
        ttl = ContextTtl.from_config(commands)
        assert ttl == ContextTtl(followup=15.0, objects=90.0, answer=300.0, pending=10.0)

    def test_a_context_reports_the_deadlines_in_force(self, clock):
        ttl = ContextTtl(followup=5.0)
        context = DialogContext(ttl=ttl, clock=clock, window_probe=lambda: None)
        assert context.ttl == ttl
        assert context.snapshot().ttl == ttl


class TestPersistence:
    """What survives a restart, and what deliberately does not."""

    @staticmethod
    def make(store, clock: Clock, *, profile_id: int | None = None) -> DialogContext:
        return DialogContext(
            store=store,
            clock=clock,
            window_probe=lambda: None,
            profile_id=profile_id,
        )

    def test_the_context_comes_back_after_a_restart(self, clock):
        store = MemoryContextStore()
        before = self.make(store, clock, profile_id=1)
        before.remember_command(intent="таймер", phrase="поставь таймер на 5 минут")
        before.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        before.remember_answer("сейчас двенадцать часов")

        after = self.make(store, clock, profile_id=1)
        assert after.restore()
        snapshot = after.snapshot()
        assert snapshot.command is not None
        assert snapshot.command.phrase == "поставь таймер на 5 минут"
        assert [item.target for item in snapshot.objects] == ["chrome"]
        assert snapshot.answer is not None
        assert snapshot.answer.text == "сейчас двенадцать часов"

    def test_an_outstanding_question_is_not_restored(self, clock):
        # The user who restarts the program has no idea a question was open, and
        # their first command would be swallowed as its answer.
        store = MemoryContextStore()
        before = self.make(store, clock, profile_id=1)
        before.remember_answer("что-нибудь")
        before.set_pending(TestPending.request())

        after = self.make(store, clock, profile_id=1)
        after.restore()
        assert after.pending() is None

    def test_what_expired_while_the_program_was_off_does_not_come_back(self, clock):
        store = MemoryContextStore()
        before = self.make(store, clock, profile_id=1)
        before.remember_command(intent="таймер", phrase="таймер")
        before.remember_object(ObjectKind.APP, "браузер")
        before.remember_answer("сейчас двенадцать часов")

        clock.advance(DEFAULT_OBJECT_TTL + 1)
        after = self.make(store, clock, profile_id=1)
        assert after.restore()
        snapshot = after.snapshot()
        assert snapshot.command is None
        assert snapshot.objects == ()
        # The answer's deadline is the long one, so this one is still there.
        assert snapshot.answer is not None

    def test_a_payload_from_another_profile_is_refused(self, clock):
        store = MemoryContextStore()
        before = self.make(store, clock, profile_id=1)
        before.remember_answer("сейчас двенадцать часов")

        after = self.make(store, clock, profile_id=2)
        assert not after.restore()
        assert after.snapshot().answer is None

    def test_a_payload_from_another_version_is_refused(self, clock):
        # Better an empty context than half-read fields from a layout that has
        # since changed shape.
        store = MemoryContextStore()
        store.save({"version": 999, "answer": {"text": "из будущего"}})
        context = self.make(store, clock)
        assert not context.restore()
        assert context.snapshot().answer is None

    def test_an_empty_store_restores_nothing_and_says_so(self, clock):
        assert not self.make(MemoryContextStore(), clock).restore()

    def test_a_restore_that_found_only_stale_fields_says_so(self, clock):
        store = MemoryContextStore()
        before = self.make(store, clock, profile_id=1)
        before.remember_answer("сейчас двенадцать часов")
        clock.advance(DEFAULT_ANSWER_TTL + 1)
        assert not self.make(store, clock, profile_id=1).restore()

    def test_a_profile_switch_wipes_the_stored_copy(self, clock):
        store = MemoryContextStore()
        context = self.make(store, clock, profile_id=1)
        context.remember_answer("сейчас двенадцать часов")
        context.reset(profile_id=2)
        assert store.load() is None

    def test_autosave_off_writes_nothing(self, clock):
        store = MemoryContextStore()
        context = DialogContext(
            store=store,
            clock=clock,
            window_probe=lambda: None,
            autosave=False,
        )
        context.remember_answer("сейчас двенадцать часов")
        assert store.load() is None
        # An explicit save still works — the flag is about writing on every
        # change, not about the store being unusable.
        context.save()
        assert store.load() is not None

    def test_the_database_store_round_trips_through_a_real_file(self, clock, repos):
        # ``MemoryContextStore`` cannot catch a value that JSON refuses, and the
        # real store goes through the ``variables`` table's own encoding.
        store = VariableContextStore(repos.variables)
        before = self.make(store, clock, profile_id=1)
        before.remember_command(intent="таймер", slots={"duration": 300})
        before.remember_object(ObjectKind.FILE, "отчёт", "C:/отчёт.txt")

        after = self.make(store, clock, profile_id=1)
        assert after.restore()
        snapshot = after.snapshot()
        assert snapshot.command is not None
        assert snapshot.command.slots == {"duration": 300}
        assert [item.name for item in snapshot.objects] == ["отчёт"]

    def test_the_database_store_keeps_the_blob_under_one_known_name(self, clock, repos):
        store = VariableContextStore(repos.variables)
        self.make(store, clock).remember_answer("сейчас двенадцать часов")
        assert isinstance(repos.variables.get_value(CONTEXT_VARIABLE), dict)
        store.clear()
        assert repos.variables.get_value(CONTEXT_VARIABLE) is None

    def test_a_store_that_throws_does_not_take_the_dialogue_down(self, clock):
        # A locked or missing database means the context does not survive this
        # restart. It must not mean «отмена» stops working.
        class BrokenStore:
            def load(self) -> JsonObject | None:
                raise RuntimeError("нет доступа")

            def save(self, state: JsonObject) -> None:
                raise RuntimeError("нет доступа")

            def clear(self) -> None:
                raise RuntimeError("нет доступа")

        context = DialogContext(store=BrokenStore(), clock=clock, window_probe=lambda: None)
        context.remember_answer("сейчас двенадцать часов")
        assert context.snapshot().answer is not None
        assert not context.restore()
        context.reset()
        assert context.snapshot().answer is None


class TestBus:
    """The two events the context listens to, and what it does with them."""

    @staticmethod
    def make_bus() -> EventBus:
        # ``thread_id=None`` delivers in the publishing thread: there is no Qt
        # loop here to hand an event over to.
        return EventBus(thread_id=None)

    def test_a_cancel_on_the_bus_clears_the_context(self, context):
        bus = self.make_bus()
        context.attach(bus)
        context.remember_command(intent="таймер", phrase="поставь таймер")
        context.remember_object(ObjectKind.APP, "браузер")
        context.remember_answer("сейчас двенадцать часов")
        try:
            bus.publish(CancelRequested(reason="стоп-слово"))
            snapshot = context.snapshot()
            assert snapshot.command is None
            assert snapshot.objects == ()
            assert snapshot.answer is not None
        finally:
            context.detach()

    def test_speaking_is_recorded_as_the_last_answer(self, context):
        bus = self.make_bus()
        context.attach(bus)
        try:
            bus.publish(TtsStarted(text="сейчас двенадцать часов", engine="silero"))
            snapshot = context.snapshot()
            assert snapshot.answer is not None
            assert snapshot.answer.text == "сейчас двенадцать часов"
        finally:
            context.detach()

    def test_a_clarifying_question_is_not_recorded_as_an_answer(self, context):
        # The question is spoken like anything else. Recording it would make
        # «повтори» repeat the question and overwrite what the user asked for.
        bus = self.make_bus()
        context.attach(bus)
        context.remember_answer("сейчас двенадцать часов")
        context.set_pending(TestPending.request())
        try:
            bus.publish(TtsStarted(text="Какую громкость поставить?"))
            snapshot = context.snapshot()
            assert snapshot.answer is not None
            assert snapshot.answer.text == "сейчас двенадцать часов"
        finally:
            context.detach()

    def test_answers_can_be_left_untracked(self, context):
        bus = self.make_bus()
        context.attach(bus, track_answers=False)
        try:
            bus.publish(TtsStarted(text="сейчас двенадцать часов"))
            assert context.snapshot().answer is None
            # The cancel subscription is still live either way.
            context.remember_object(ObjectKind.APP, "браузер")
            bus.publish(CancelRequested())
            assert context.snapshot().objects == ()
        finally:
            context.detach()

    def test_attaching_twice_does_not_double_the_subscription(self, context):
        bus = self.make_bus()
        context.attach(bus)
        context.attach(bus)
        try:
            bus.publish(TtsStarted(text="сейчас двенадцать часов"))
            assert context.snapshot().answer is not None
        finally:
            context.detach()
        bus.publish(CancelRequested())
        assert context.snapshot().answer is not None

    def test_after_detaching_the_bus_is_ignored(self, context):
        bus = self.make_bus()
        context.attach(bus)
        context.detach()
        context.remember_object(ObjectKind.APP, "браузер")
        bus.publish(CancelRequested())
        assert len(context.snapshot().objects) == 1

    def test_detaching_twice_is_not_an_error(self, context):
        bus = self.make_bus()
        context.attach(bus)
        context.detach()
        context.detach()


class TestPronouns:
    """Which words carry a reference, and where they sit in a phrase."""

    @pytest.mark.parametrize(
        "word",
        ["его", "него", "ее", "нее", "их", "им", "это", "тот", "ту", "там", "туда"],
    )
    def test_a_pronoun_is_recognised_on_its_own(self, word):
        assert is_pronoun(word)

    @pytest.mark.parametrize("word", ["браузер", "закрой", "хром", "самое", ""])
    def test_an_ordinary_word_is_not_a_pronoun(self, word):
        assert not is_pronoun(word)

    def test_the_first_pronoun_in_a_phrase_is_the_one_taken(self):
        mention = find_mention("закрой его и открой ее")
        assert mention is not None
        assert mention.pronoun.text == "его"
        assert mention.start == 1
        assert mention.end == 2

    def test_a_longer_phrase_wins_over_the_shorter_one_it_starts_with(self):
        # «то же» inside «то же самое» would leave «самое» dangling in the
        # rewritten phrase.
        mention = find_mention("сделай то же самое")
        assert mention is not None
        assert mention.pronoun.text == "то же самое"
        assert mention.length == 3

    def test_a_phrase_without_pronouns_has_no_mention(self):
        assert find_mention("закрой гугл хром") is None

    def test_a_place_pronoun_only_reaches_places(self):
        mention = find_mention("сохрани туда")
        assert mention is not None
        assert mention.pronoun.place
        assert mention.pronoun.kinds == PLACE_KINDS
        assert mention.pronoun.question == QUESTION_PLACE

    def test_an_ambiguous_pronoun_suggests_no_gender(self):
        # «им» is dative singular masculine and dative plural at once; a hint
        # here would be actively misleading.
        mention = find_mention("займись им")
        assert mention is not None
        assert mention.pronoun.gender is None


class TestAnaphora:
    """«закрой его» → «закрой гугл хром», and what happens when it cannot."""

    def test_a_phrase_with_no_pronoun_comes_back_untouched(self, context):
        result = resolve_anaphora("закрой гугл хром", context.snapshot())
        assert result.status is AnaphoraStatus.ABSENT
        assert result.text == "закрой гугл хром"
        assert not result.resolved
        assert not result.needs_question
        assert result.question == ""

    def test_a_pronoun_is_replaced_with_the_spoken_name(self, context):
        # The name, not the machine value: the rewritten phrase goes back through
        # the matcher, which was written against what people say.
        context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        result = resolve_anaphora("закрой его", context.snapshot())
        assert result.status is AnaphoraStatus.RESOLVED
        assert result.text == "закрой гугл хром"
        assert result.replacement == "гугл хром"
        assert result.object is not None
        assert result.object.target == "chrome"

    def test_the_rest_of_the_phrase_survives_the_substitution(self, context):
        context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        result = resolve_anaphora("а теперь закрой его пожалуйста", context.snapshot())
        assert result.text == "а теперь закрой гугл хром пожалуйста"

    def test_gender_picks_between_two_things_mentioned(self, context):
        context.remember_object(ObjectKind.APP, "браузер", "chrome")
        context.remember_object(ObjectKind.APP, "музыка", "spotify")
        snapshot = context.snapshot()
        assert resolve_anaphora("закрой его", snapshot).text == "закрой браузер"
        assert resolve_anaphora("закрой ее", snapshot).text == "закрой музыка"

    def test_a_pronoun_with_nothing_to_point_at_asks(self, context):
        result = resolve_anaphora("закрой его", context.snapshot())
        assert result.status is AnaphoraStatus.UNRESOLVED
        assert result.needs_question
        assert result.question == QUESTION_OBJECT
        # The text still comes back usable for a caller that ignores the status.
        assert result.text == "закрой его"

    def test_a_mention_that_expired_stops_being_reachable(self, clock, context):
        # The half-minute that «закрой его» is good for is the whole point of the
        # object deadline: a pronoun resolving to something said ten minutes ago
        # is worse than a question.
        context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        assert resolve_anaphora("закрой его", context.snapshot()).resolved
        clock.advance(DEFAULT_OBJECT_TTL + 1)
        result = resolve_anaphora("закрой его", context.snapshot())
        assert result.status is AnaphoraStatus.UNRESOLVED
        assert result.question == QUESTION_OBJECT

    def test_a_place_pronoun_needs_somewhere_to_have_been_mentioned(self, context):
        context.remember_object(ObjectKind.TEXT, "отчет")
        result = resolve_anaphora("сохрани туда", context.snapshot())
        assert result.status is AnaphoraStatus.UNRESOLVED
        assert result.question == QUESTION_PLACE

        context.remember_object(ObjectKind.FILE, "папка отчеты", "C:/отчеты")
        assert resolve_anaphora("сохрани туда", context.snapshot()).resolved

    def test_the_caller_can_narrow_what_a_pronoun_may_reach(self, context):
        # The only trigger left is «закрой {app}», so «закрой его» must not
        # resolve to the URL that happened to be mentioned last.
        context.remember_object(ObjectKind.APP, "браузер", "chrome")
        context.remember_object(ObjectKind.URL, "ютуб", "https://youtube.com")
        snapshot = context.snapshot()
        assert resolve_anaphora("закрой его", snapshot).text == "закрой ютуб"
        narrowed = resolve_anaphora("закрой его", snapshot, kinds=(ObjectKind.APP,))
        assert narrowed.text == "закрой браузер"

    def test_two_restrictions_with_nothing_in_common_are_not_split(self, context):
        # «сохрани туда» where the caller will only take a volume level is a
        # request nobody can satisfy — better a question than the newest object
        # of the wrong kind.
        context.remember_object(ObjectKind.FILE, "папка отчеты", "C:/отчеты")
        result = resolve_anaphora("сохрани туда", context.snapshot(), kinds=(ObjectKind.TEXT,))
        assert result.status is AnaphoraStatus.UNRESOLVED

    def test_a_slot_value_that_turned_out_to_be_a_pronoun_resolves(self, context):
        # «закрой {app}» matches «закрой его» perfectly well, and it is the
        # captured value that needs the context rather than the phrase.
        context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        found = resolve_reference("его", context.snapshot())
        assert found is not None
        assert found.target == "chrome"

    def test_a_slot_value_that_is_a_real_name_is_not_a_reference(self, context):
        context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        assert resolve_reference("хром", context.snapshot()) is None

    def test_a_reference_with_nothing_remembered_is_not_invented(self, context):
        assert resolve_reference("его", context.snapshot()) is None


class TestHourRange:
    """`when_time: "9-18"`, and the night shift that runs through midnight."""

    @pytest.mark.parametrize(
        ("raw", "start", "end"),
        [
            ("9-18", 9, 18),
            ("9:30-18:00", 9, 18),
            ("21-6", 21, 6),
            (" 0 - 24 ", 0, 0),
            ("8–17", 8, 17),
            ("8—17", 8, 17),
        ],
    )
    def test_a_range_is_read_off_the_written_form(self, raw, start, end):
        # Minutes are accepted and dropped: an hour is the granularity a trigger
        # needs, and refusing «9:30-18:00» would only annoy whoever typed it.
        span = HourRange.parse(raw)
        assert span is not None
        assert (span.start, span.end) == (start, end)

    @pytest.mark.parametrize("raw", ["полшестого", "9", "9-", "-18", "25-30", "с 9 до 18", ""])
    def test_a_range_nobody_can_read_is_not_guessed_at(self, raw):
        assert HourRange.parse(raw) is None

    def test_hours_inside_a_daytime_range_are_the_ones_between_its_ends(self):
        span = HourRange(start=9, end=18)
        assert not span.wraps
        assert span.contains(9)
        assert span.contains(17)
        # The end is exclusive: «9-18» is over when 18:00 strikes.
        assert not span.contains(18)
        assert not span.contains(8)

    def test_a_night_shift_runs_through_midnight(self):
        # Reading «21-6» as an empty range would silently disable the trigger.
        span = HourRange(start=21, end=6)
        assert span.wraps
        assert span.contains(23)
        assert span.contains(0)
        assert span.contains(5)
        assert not span.contains(6)
        assert not span.contains(12)

    def test_a_range_with_equal_ends_means_all_day(self):
        assert HourRange(start=0, end=0).contains(13)
        assert HourRange(start=7, end=7).contains(3)

    def test_an_hour_outside_the_clock_is_folded_rather_than_refused(self):
        assert HourRange(start=9, end=18).contains(24 + 10)

    def test_an_impossible_hour_is_refused_outright(self):
        with pytest.raises(ValueError, match="0..24"):
            HourRange(start=25, end=3)

    def test_a_range_describes_itself_in_russian(self):
        assert HourRange(start=9, end=18).describe() == "с 9:00 до 18:00"


class TestVariableCondition:
    """`when_variable`, in the two shapes it gets written in."""

    @staticmethod
    def snapshot(**variables: object) -> ContextSnapshot:
        return ContextSnapshot(variables=dict(variables))

    @pytest.mark.parametrize(
        ("raw", "name", "test", "expected"),
        [
            ("режим=работа", "режим", VariableTest.EQUALS, "работа"),
            ("режим==работа", "режим", VariableTest.EQUALS, "работа"),
            ("режим!=работа", "режим", VariableTest.NOT_EQUALS, "работа"),
            ("режим<>работа", "режим", VariableTest.NOT_EQUALS, "работа"),
            ("тихий_час", "тихий_час", VariableTest.TRUTHY, None),
            ("!тихий_час", "тихий_час", VariableTest.FALSY, None),
        ],
    )
    def test_the_hand_written_form_is_read(self, raw, name, test, expected):
        condition, problem = VariableCondition.parse(raw)
        assert problem == ""
        assert condition == VariableCondition(name=name, test=test, expected=expected)

    def test_the_generated_form_carries_a_value_of_any_type(self):
        # A mapping is the only shape that can say «громкость = 40» and mean the
        # number rather than the string.
        condition, problem = VariableCondition.parse(
            {"name": "громкость", "test": "eq", "value": 40}
        )
        assert problem == ""
        assert condition == VariableCondition(
            name="громкость", test=VariableTest.EQUALS, expected=40
        )

    def test_a_mapping_with_a_value_and_no_test_means_equality(self):
        condition, _ = VariableCondition.parse({"name": "режим", "value": "работа"})
        assert condition is not None
        assert condition.test is VariableTest.EQUALS

    def test_a_mapping_with_neither_value_nor_test_means_truth(self):
        condition, _ = VariableCondition.parse({"var": "тихий_час"})
        assert condition is not None
        assert condition.test is VariableTest.TRUTHY

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "=работа",
            "!",
            42,
            {"test": "eq", "value": "работа"},
            {"name": "режим", "test": "похоже"},
            {"name": "режим", "test": 5},
        ],
    )
    def test_a_condition_nobody_can_read_complains_instead_of_passing(self, raw):
        condition, problem = VariableCondition.parse(raw)
        assert condition is None
        assert problem

    def test_equality_compares_by_meaning_rather_than_by_type(self):
        # A variable holding 40 and a condition written "40" are the same thing
        # to whoever typed them.
        condition = VariableCondition(name="громкость", test=VariableTest.EQUALS, expected="40")
        assert condition.holds(self.snapshot(громкость=40))
        assert condition.holds(self.snapshot(громкость="40"))
        assert not condition.holds(self.snapshot(громкость=41))

    def test_a_missing_variable_fails_every_test_about_a_value(self):
        # Nothing is true about a value that is not there, and a `None` that
        # merely looks falsy must not satisfy `falsy`.
        empty = self.snapshot()
        for test in (
            VariableTest.EQUALS,
            VariableTest.NOT_EQUALS,
            VariableTest.CONTAINS,
            VariableTest.TRUTHY,
            VariableTest.FALSY,
            VariableTest.EXISTS,
        ):
            assert not VariableCondition(name="режим", test=test, expected="").holds(empty)
        assert VariableCondition(name="режим", test=VariableTest.MISSING).holds(empty)

    @pytest.mark.parametrize("value", ["false", "нет", "выкл", "0", "", "none", 0, False, None, []])
    def test_the_words_people_write_for_false_count_as_false(self, value):
        # ``bool("false")`` is ``True``, and the value came out of JSON or a text
        # field far more often than out of a checkbox.
        condition = VariableCondition(name="режим", test=VariableTest.TRUTHY)
        assert not condition.holds(self.snapshot(режим=value))

    @pytest.mark.parametrize("value", ["работа", "true", "да", 1, True, 0.5, ["что-то"]])
    def test_anything_else_counts_as_true(self, value):
        condition = VariableCondition(name="режим", test=VariableTest.TRUTHY)
        assert condition.holds(self.snapshot(режим=value))

    def test_containment_is_tested_on_folded_text(self):
        condition = VariableCondition(name="режим", test=VariableTest.CONTAINS, expected="РАБОТ")
        assert condition.holds(self.snapshot(режим="Работа из дома"))

    def test_a_condition_describes_itself_for_the_command_editor(self):
        assert (
            VariableCondition(name="режим", test=VariableTest.EQUALS, expected="работа").describe()
            == "переменная режим = «работа»"
        )
        assert VariableCondition(name="тихий_час").describe() == "переменная тихий_час включена"


class TestConditions:
    """A whole condition set: what it is read from, and when it holds."""

    @staticmethod
    def snapshot(
        *,
        window: WindowInfo | None = None,
        at: datetime | None = None,
        profile_id: int | None = None,
        **variables: object,
    ) -> ContextSnapshot:
        return ContextSnapshot(
            at=at if at is not None else BASE_TIME,
            profile_id=profile_id,
            window=window,
            variables=dict(variables),
        )

    def test_a_trigger_with_no_conditions_is_always_a_candidate(self):
        assert UNCONDITIONAL.is_empty
        assert UNCONDITIONAL.holds(self.snapshot())
        assert UNCONDITIONAL.describe() == ""
        assert not UNCONDITIONAL.needs_window

    def test_a_payload_without_condition_keys_gets_the_shared_empty_set(self):
        # The index holds one of these per unconditional trigger, and a library
        # of a thousand of them should not hold a thousand equal objects.
        assert conditions_from_payload({}) is UNCONDITIONAL
        assert conditions_from_payload({"phrase": "свет"}) is UNCONDITIONAL

    def test_the_five_keys_are_read_out_of_a_payload(self):
        conditions = conditions_from_payload(
            {
                "when_window": "*Photoshop*",
                "when_process": ["photoshop.exe", "winword.exe"],
                "when_time": ["вечером", "9-12"],
                "when_variable": "режим=работа",
                "when_profile": [1, "2"],
            }
        )
        assert conditions.windows == ("*Photoshop*",)
        assert conditions.processes == ("photoshop.exe", "winword.exe")
        assert conditions.times == (TimeOfDay.EVENING,)
        assert conditions.hours == (HourRange(start=9, end=12),)
        assert conditions.variables == (
            VariableCondition(name="режим", test=VariableTest.EQUALS, expected="работа"),
        )
        assert conditions.profiles == (1, 2)
        assert set(CONDITION_KEYS) == {
            "when_window",
            "when_process",
            "when_time",
            "when_variable",
            "when_profile",
        }

    def test_conditions_must_all_hold_at_once(self):
        conditions = conditions_from_payload(
            {"when_process": "photoshop.exe", "when_variable": "режим=работа"}
        )
        photoshop = WindowInfo(title="Adobe Photoshop", process="photoshop.exe")
        assert conditions.holds(self.snapshot(window=photoshop, режим="работа"))
        assert not conditions.holds(self.snapshot(window=photoshop, режим="отдых"))
        assert not conditions.holds(self.snapshot(режим="работа"))

    def test_values_inside_one_condition_are_alternatives(self):
        conditions = conditions_from_payload({"when_process": ["photoshop.exe", "winword.exe"]})
        assert conditions.holds(self.snapshot(window=WindowInfo(process="winword.exe")))
        assert conditions.holds(self.snapshot(window=WindowInfo(process="photoshop.exe")))
        assert not conditions.holds(self.snapshot(window=WindowInfo(process="notepad.exe")))

    def test_a_title_and_a_process_name_are_two_ways_of_saying_the_same_thing(self):
        # Whoever wrote both meant «this application», not «both at once».
        conditions = conditions_from_payload(
            {"when_window": "*Photoshop*", "when_process": "photoshop.exe"}
        )
        assert conditions.holds(self.snapshot(window=WindowInfo(title="Adobe Photoshop")))
        assert conditions.holds(self.snapshot(window=WindowInfo(process="Photoshop.exe")))

    def test_an_unknown_window_fails_a_condition_that_needs_one(self):
        # Treating an unreadable window as a match would make «сохрани» fire in
        # Photoshop on a machine where the probe is broken — the one case where
        # being wrong is invisible.
        conditions = conditions_from_payload({"when_window": "*Photoshop*"})
        assert conditions.needs_window
        assert not conditions.holds(self.snapshot(window=None))

    def test_a_part_of_the_day_is_read_off_the_snapshot_clock(self):
        conditions = conditions_from_payload({"when_time": "вечером"})
        assert conditions.holds(self.snapshot(at=local_time(20)))
        assert not conditions.holds(self.snapshot(at=local_time(8)))

    def test_a_part_of_the_day_and_an_hour_range_are_alternatives(self):
        conditions = conditions_from_payload({"when_time": ["ночью", "9-12"]})
        assert conditions.times == (TimeOfDay.NIGHT,)
        assert conditions.hours == (HourRange(start=9, end=12),)
        assert conditions.holds(self.snapshot(at=local_time(3)))
        assert conditions.holds(self.snapshot(at=local_time(10)))
        assert not conditions.holds(self.snapshot(at=local_time(15)))

    def test_a_profile_condition_keeps_a_trigger_to_its_own_profile(self):
        conditions = conditions_from_payload({"when_profile": 2})
        assert conditions.holds(self.snapshot(profile_id=2))
        assert not conditions.holds(self.snapshot(profile_id=1))
        assert not conditions.holds(self.snapshot())

    def test_a_malformed_condition_is_dropped_rather_than_satisfied(self):
        payload = {"when_time": "полшестого", "when_process": "photoshop.exe"}
        conditions, problems = TriggerConditions.parse(payload)
        assert problems
        assert conditions.times == ()
        assert conditions.hours == ()
        # What was readable still applies.
        assert conditions.processes == ("photoshop.exe",)
        assert not conditions.holds(self.snapshot())

    def test_a_trigger_whose_only_condition_was_unreadable_becomes_unconditional(self):
        # There is nothing better to do: dropping the trigger entirely would make
        # a typo in one key silently remove a command.
        conditions = conditions_from_payload({"when_time": "полшестого"})
        assert conditions.is_empty
        assert conditions.holds(self.snapshot())

    @pytest.mark.parametrize(
        "payload",
        [
            {"when_window": 5},
            {"when_time": 5},
            {"when_variable": 5},
            {"when_profile": True},
            {"when_profile": "первый"},
        ],
    )
    def test_the_command_editor_gets_a_russian_complaint(self, payload):
        problems = validate_conditions(payload)
        assert problems
        assert all(isinstance(problem, str) and problem for problem in problems)

    def test_a_valid_payload_has_nothing_to_complain_about(self):
        assert validate_conditions({"when_window": "*Photoshop*", "when_profile": 1}) == ()

    def test_conditions_describe_themselves_for_the_trigger_list(self):
        conditions = conditions_from_payload(
            {
                "when_window": "*Photoshop*",
                "when_time": "вечером",
                "when_variable": "режим=работа",
                "when_profile": 1,
            }
        )
        described = describe_conditions(conditions)
        assert described == (
            "окно: *Photoshop*",
            "время: вечер",
            "переменная режим = «работа»",
            "профиль: 1",
        )
        assert conditions.describe() == "; ".join(described)

    def test_a_repeated_part_of_the_day_is_only_kept_once(self):
        conditions = conditions_from_payload({"when_time": ["вечером", "вечер"]})
        assert conditions.times == (TimeOfDay.EVENING,)


class TestPredicate:
    """The predicate the matcher filters candidates with."""

    @staticmethod
    def entry(payload: dict[str, object] | None = None) -> FakeEntry:
        return FakeEntry(
            conditions=conditions_from_payload(payload or {}),
        )

    def test_nothing_to_filter_by_means_no_predicate_at_all(self):
        # The matcher skips the per-candidate call entirely rather than invoking
        # one that always says yes.
        assert context_predicate(None) is None

    def test_a_snapshot_filters_by_the_conditions_the_index_compiled(self):
        snapshot = ContextSnapshot(window=WindowInfo(process="photoshop.exe"))
        predicate = context_predicate(snapshot)
        assert predicate is not None
        assert predicate(self.entry())
        assert predicate(self.entry({"when_process": "photoshop.exe"}))
        assert not predicate(self.entry({"when_process": "winword.exe"}))

    def test_an_extra_restriction_composes_in(self):
        snapshot = ContextSnapshot(window=WindowInfo(process="photoshop.exe"))
        predicate = context_predicate(snapshot, extra=lambda _entry: False)
        assert predicate is not None
        assert not predicate(self.entry())

    def test_an_extra_restriction_survives_a_missing_snapshot(self):
        # The pipeline passes ``extra`` to limit matching to one command when it
        # re-runs a repeat, and that has to work with no context at all.
        def only_nothing(entry: IndexedTrigger) -> bool:
            return False

        assert context_predicate(None, extra=only_nothing) is only_nothing


class TestConditionalMatching:
    """One phrase, two commands: «сохрани» in Photoshop and in Word.

    The point of the whole conditions feature, and the one group that goes through
    the matcher rather than testing a predicate in isolation — because filtering
    happens while candidates are collected, and a test that called
    :meth:`TriggerConditions.holds` directly would pass even if the matcher never
    consulted it.
    """

    @staticmethod
    def matcher() -> Matcher:
        return Matcher.from_triggers(
            [
                Trigger(
                    id=1,
                    command_id=10,
                    pattern="сохрани",
                    conditions=conditions_from_payload({"when_window": "*Photoshop*"}),
                ),
                Trigger(
                    id=2,
                    command_id=20,
                    pattern="сохрани",
                    conditions=conditions_from_payload({"when_process": "winword.exe"}),
                ),
            ]
        )

    @staticmethod
    def context(clock: Clock, window: WindowInfo | None) -> DialogContext:
        return DialogContext(clock=clock, window_probe=lambda: window, autosave=False)

    def test_the_same_phrase_reaches_different_commands(self, clock):
        matcher = self.matcher()
        photoshop = self.context(clock, WindowInfo(title="отчёт.psd — Adobe Photoshop"))
        word = self.context(clock, WindowInfo(title="Документ1 — Word", process="WINWORD.EXE"))

        in_photoshop = matcher.match("сохрани", context=photoshop.snapshot())
        in_word = matcher.match("сохрани", context=word.snapshot())
        assert in_photoshop is not None
        assert in_word is not None
        assert in_photoshop.command_id == 10
        assert in_word.command_id == 20

    def test_a_phrase_whose_conditions_all_fail_matches_nothing(self, clock):
        # Not «the best of the two»: a command that must not fire here has no
        # business firing because it was the only candidate left.
        elsewhere = self.context(clock, WindowInfo(title="Блокнот", process="notepad.exe"))
        assert self.matcher().match("сохрани", context=elsewhere.snapshot()) is None

    def test_an_unreadable_window_matches_nothing_either(self, clock):
        blind = self.context(clock, None)
        assert self.matcher().match("сохрани", context=blind.snapshot()) is None

    def test_without_a_context_conditions_are_not_applied_at_all(self):
        # The caller that passes no snapshot is asking «what could this mean»,
        # not «what does it mean here» — the trigger editor's preview does this.
        result = self.matcher().match("сохрани")
        assert result is not None
        assert result.command_id in {10, 20}
        assert len(self.matcher().match_all("сохрани")) == 2

    def test_an_inactive_trigger_is_dropped_before_ranking_not_after(self, clock):
        # A condition checked after the winner is picked disambiguates nothing:
        # the Photoshop trigger would win on priority, fail its condition, and
        # leave the Word one unscored.
        matcher = Matcher.from_triggers(
            [
                Trigger(
                    id=1,
                    command_id=10,
                    pattern="сохрани",
                    priority=100,
                    conditions=conditions_from_payload({"when_window": "*Photoshop*"}),
                ),
                Trigger(id=2, command_id=20, pattern="сохрани"),
            ]
        )
        word = self.context(clock, WindowInfo(process="winword.exe"))
        result = matcher.match("сохрани", context=word.snapshot())
        assert result is not None
        assert result.command_id == 20

    def test_a_variable_switches_a_phrase_between_two_commands(self, clock):
        matcher = Matcher.from_triggers(
            [
                Trigger(
                    id=1,
                    command_id=10,
                    pattern="начинаем",
                    conditions=conditions_from_payload({"when_variable": "режим=работа"}),
                ),
                Trigger(
                    id=2,
                    command_id=20,
                    pattern="начинаем",
                    conditions=conditions_from_payload({"when_variable": "режим=отдых"}),
                ),
            ]
        )
        context = self.context(clock, None)
        context.set_variable("режим", "работа")
        first = matcher.match("начинаем", context=context.snapshot())
        context.set_variable("режим", "отдых")
        second = matcher.match("начинаем", context=context.snapshot())
        assert first is not None
        assert second is not None
        assert (first.command_id, second.command_id) == (10, 20)

    def test_the_window_is_read_once_per_phrase_not_once_per_trigger(self, clock):
        # A library of a thousand conditional triggers must cost one WinAPI round
        # trip, which is why the predicate closes over a snapshot.
        probes = []

        def probe() -> WindowInfo:
            probes.append(1)
            return WindowInfo(title="Adobe Photoshop")

        context = DialogContext(clock=clock, window_probe=probe, autosave=False)
        matcher = Matcher.from_triggers(
            [
                Trigger(
                    id=number,
                    command_id=number,
                    pattern=f"команда {number}",
                    conditions=conditions_from_payload({"when_window": "*Photoshop*"}),
                )
                for number in range(1, 21)
            ]
        )
        snapshot = context.snapshot()
        assert matcher.match("команда 7", context=snapshot) is not None
        assert len(probes) == 1

    def test_a_rewritten_phrase_goes_back_through_the_matcher(self, clock):
        # «закрой его» end to end: the pronoun is replaced with the remembered
        # name, and the result matches the ordinary trigger.
        matcher = Matcher.from_triggers([Trigger(id=1, command_id=10, pattern="закрой гугл хром")])
        context = self.context(clock, None)
        assert matcher.match("закрой его") is None

        context.remember_object(ObjectKind.APP, "гугл хром", "chrome")
        rewritten = resolve_anaphora("закрой его", context.snapshot())
        assert rewritten.resolved
        result = matcher.match(rewritten.text, context=context.snapshot())
        assert result is not None
        assert result.command_id == 10
