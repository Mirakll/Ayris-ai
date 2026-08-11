"""Task 16: «открой гугл хром» → a program that can actually be launched.

The module splits into a half that is pure and a half that has to touch Windows,
and the tests split the same way. :class:`~ayris.nlu.apps.AppResolver` is a
function of a dictionary — it reads no registry, no Start menu and no disk — so
everything interesting about it runs here, on Linux, in CI: the Russian cases,
the tolerance for a misheard alias, the priority of the user's own vocabulary.
:func:`~ayris.nlu.apps.scan_installed_apps` is the other half, and it is behind a
``skipif``; what is checked everywhere is that importing and calling it off
Windows is harmless, because every caller runs on both.

Two orderings are the whole design and each is asserted directly.

*Exact beats stem beats fuzzy.* «хроме» is an alias and scores 1.0; «хрома» is
not, and reaches Chrome by sharing four letters of five; «хрм» is a recogniser
that swallowed a vowel and reaches it by similarity alone. The scores have to
stay ordered, because :class:`~ayris.nlu.slot_types.AppType` refuses anything
under 0.75 and a guess promoted to certainty launches the wrong program.

*A user alias beats the shipped dictionary.* «почта» ships pointing at Outlook.
A user who says «открой почту» meaning Thunderbird sets that once, and the next
dictionary update does not get a vote.

Groups:

* :class:`TestShippedDictionary` — ``resources/nlu/apps.json`` really loads.
* :class:`TestExactAliases` — the names, aliases and executables as written.
* :class:`TestRussianCases` — «в хроме», «блокноте», «оперу» without a table.
* :class:`TestFuzzyMatching` — «хрм», and what must stay out of reach.
* :class:`TestUserAliases` — the user's vocabulary, and its priority.
* :class:`TestInstalled` — merging a scan, including a program nobody shipped.
* :class:`TestPrefix` — where an app name ends and the rest of the phrase starts.
* :class:`TestLoading` — a broken dictionary is a packaging bug and says so.
* :class:`TestScanner` — the Windows half, skipped elsewhere.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from ayris.nlu.apps import (
    APPS_SCHEMA_VERSION,
    DEFAULT_ALIAS_THRESHOLD,
    AppEntry,
    AppResolver,
    AppsError,
    AppsFile,
    InstalledApp,
    apps_file,
    load_apps,
    scan_installed_apps,
)

if TYPE_CHECKING:
    from pathlib import Path

#: The dictionary as it ships. Read once: it is immutable, and every resolver
#: below is built from it rather than from a fixture invented for the test — a
#: dictionary that works only against made-up entries proves nothing about the
#: one the application will use.
CATALOG: AppsFile = load_apps()

#: Entries a couple of tests name directly, so a rename in the file shows up as a
#: failed lookup here rather than as a silently skipped assertion.
CHROME = "chrome"


@pytest.fixture(scope="module")
def resolver() -> AppResolver:
    """The shipped dictionary, indexed, with nothing merged into it."""
    return AppResolver.from_apps(CATALOG.apps)


class TestShippedDictionary:
    """The file that ships has to load, and to be reachable once loaded."""

    def test_it_loads(self) -> None:
        assert CATALOG.schema_version == APPS_SCHEMA_VERSION
        assert len(CATALOG.apps) > 20

    def test_ids_are_unique(self) -> None:
        """``id`` is stored in ``app_aliases`` and named in templates."""
        ids = [entry.id for entry in CATALOG.apps]
        assert len(ids) == len(set(ids))

    def test_every_entry_can_be_launched_or_is_a_store_app(self) -> None:
        """An entry with neither an executable nor an AUMID is unreachable."""
        for entry in CATALOG.apps:
            assert entry.executables or entry.uwp, entry.id

    def test_every_alias_reaches_its_own_entry(self, resolver: AppResolver) -> None:
        """The one that catches a collision: two entries claiming one spelling.

        Folding is not the identity — «Notepad++» and «notepad» collapse to the
        same key — so an alias added in good faith can quietly be answered by
        whichever entry the file lists first. That is invisible until a user says
        it, which is why it is checked over the whole dictionary here.
        """
        for entry in CATALOG.apps:
            for alias in entry.aliases:
                match = resolver.resolve(alias)
                assert match is not None, (entry.id, alias)
                assert match.app_id == entry.id, (entry.id, alias, match.app_id)

    def test_every_entry_is_reachable_by_name_or_alias(self, resolver: AppResolver) -> None:
        """A program nobody can say is a line of JSON doing nothing."""
        for entry in CATALOG.apps:
            spellings = (entry.name, *entry.aliases)
            matched = [resolver.resolve(text) for text in spellings]
            assert any(m is not None and m.app_id == entry.id for m in matched), entry.id

    def test_the_file_sits_where_the_loader_looks(self) -> None:
        assert apps_file().exists()


class TestExactAliases:
    """The spellings written down, which have to score 1.0 and nothing less."""

    @pytest.mark.parametrize(
        ("phrase", "app_id"),
        [
            ("хром", "chrome"),
            ("гугл хром", "chrome"),
            ("гуглхром", "chrome"),
            ("chrome", "chrome"),
            ("google chrome", "chrome"),
            ("яндекс браузер", "yandex-browser"),
            ("яндекс", "yandex-browser"),
            ("мозилла", "firefox"),
            ("лиса", "firefox"),
            ("эдж", "edge"),
            ("опера", "opera"),
            ("телеграм", "telegram"),
            ("телеграмм", "telegram"),
            ("тг", "telegram"),
            ("дискорд", "discord"),
            ("ватсап", "whatsapp"),
            ("скайп", "skype"),
            ("зум", "zoom"),
            ("вс код", "vscode"),
            ("блокнот", "notepad"),
            ("нотпад", "notepad-plus-plus"),
            ("пайчарм", "pycharm"),
            ("ворд", "word"),
            ("эксель", "excel"),
            ("таблицы", "excel"),
            ("презентации", "powerpoint"),
            ("почта", "outlook"),
            ("спотифай", "spotify"),
            ("плеер", "vlc"),
            ("обс", "obs"),
            ("стим", "steam"),
            ("эпик", "epic-games"),
            ("фотошоп", "photoshop"),
            ("фигма", "figma"),
            ("пейнт", "paint"),
            ("проводник", "explorer"),
            ("мой компьютер", "explorer"),
            ("калькулятор", "calculator"),
            ("диспетчер задач", "task-manager"),
            ("настройки", "settings"),
            ("параметры", "settings"),
            ("консоль", "cmd"),
            ("терминал", "cmd"),
            ("пауэршелл", "powershell"),
        ],
    )
    def test_alias(self, resolver: AppResolver, phrase: str, app_id: str) -> None:
        match = resolver.resolve(phrase)
        assert match is not None
        assert match.app_id == app_id
        assert match.confidence == 1.0

    @pytest.mark.parametrize(
        ("phrase", "app_id"),
        [
            ("notepad", "notepad"),
            ("firefox", "firefox"),
            ("steam", "steam"),
            ("calc", "calculator"),
        ],
    )
    def test_executables_are_indexed_too(
        self, resolver: AppResolver, phrase: str, app_id: str
    ) -> None:
        """«запусти notepad» is a thing people say, and it is not in the aliases."""
        match = resolver.resolve(phrase)
        assert match is not None
        assert match.app_id == app_id

    def test_capitalisation_and_punctuation_are_folded(self, resolver: AppResolver) -> None:
        """Through the same normaliser as a phrase, so «Гугл Хром!» is one key."""
        match = resolver.resolve("Гугл Хром!")
        assert match is not None
        assert match.app_id == CHROME

    @pytest.mark.parametrize(
        "phrase",
        ["открой хром", "запусти хром", "включи хром", "в хроме", "с хрома", "покажи блокнот"],
    )
    def test_leading_verbs_and_prepositions_are_dropped(
        self, resolver: AppResolver, phrase: str
    ) -> None:
        """A slot at the end of a template captures the verb along with the name."""
        assert resolver.resolve(phrase) is not None

    @pytest.mark.parametrize("phrase", ["", "   ", "в", "открой", "на"])
    def test_nothing_to_resolve(self, resolver: AppResolver, phrase: str) -> None:
        """A preposition alone is not a program — and the last word is never cut."""
        assert resolver.resolve(phrase) is None


class TestRussianCases:
    """Oblique forms, handled by stem length instead of a table of endings."""

    @pytest.mark.parametrize(
        ("phrase", "app_id"),
        [
            ("хрома", "chrome"),
            ("телеграма", "telegram"),
            ("фотошопа", "photoshop"),
            ("стима", "steam"),
            ("фигмы", "figma"),
        ],
    )
    def test_a_case_ending_still_resolves(
        self, resolver: AppResolver, phrase: str, app_id: str
    ) -> None:
        match = resolver.resolve(phrase)
        assert match is not None
        assert match.app_id == app_id
        assert DEFAULT_ALIAS_THRESHOLD <= match.confidence < 1.0

    @pytest.mark.parametrize(
        ("phrase", "app_id"),
        [("хроме", "chrome"), ("блокноте", "notepad"), ("оперу", "opera"), ("экселе", "excel")],
    )
    def test_the_common_cases_are_written_down_anyway(
        self, resolver: AppResolver, phrase: str, app_id: str
    ) -> None:
        """Stemming is the safety net; the forms users actually say are aliases."""
        match = resolver.resolve(phrase)
        assert match is not None
        assert (match.app_id, match.confidence) == (app_id, 1.0)

    @pytest.mark.parametrize("phrase", ["про", "фот", "ка", "теле"])
    def test_a_bare_prefix_is_not_a_licence(self, resolver: AppResolver, phrase: str) -> None:
        """«про» must not launch «проводник»: the ratio is against the longer word."""
        assert resolver.resolve(phrase) is None


class TestFuzzyMatching:
    """A misheard alias, and the line under which nothing is close enough."""

    def test_a_swallowed_vowel_still_reaches_chrome(self, resolver: AppResolver) -> None:
        """The checklist case: «хрм» is what a recogniser does to «хром»."""
        match = resolver.resolve("хрм")
        assert match is not None
        assert match.app_id == CHROME
        assert match.confidence == pytest.approx(0.75)

    @pytest.mark.parametrize("phrase", ["бубубу", "погоду", "музыку", "холодильник"])
    def test_nothing_close_enough(self, resolver: AppResolver, phrase: str) -> None:
        """A word that is not a program has to come back as ``None``, not as a guess."""
        assert resolver.resolve(phrase) is None

    def test_two_letter_aliases_only_match_exactly(self, resolver: AppResolver) -> None:
        """«тг» is one edit from half the alphabet, so it is exempt from fuzzing."""
        assert resolver.resolve("тг") is not None
        assert resolver.resolve("тд") is None

    def test_a_lower_threshold_lets_more_through(self) -> None:
        """The tolerance is a parameter, and a caller that wants it looser says so."""
        strict = AppResolver.from_apps(CATALOG.apps, threshold=0.95)
        loose = AppResolver.from_apps(CATALOG.apps, threshold=0.6)
        assert strict.resolve("хрм") is None
        assert loose.resolve("хрм") is not None

    def test_confidence_is_ordered_across_the_three_passes(self, resolver: AppResolver) -> None:
        """Exact, then stem, then fuzzy — a caller refuses by number, not by pass."""
        exact = resolver.resolve("хром")
        stem = resolver.resolve("хрома")
        fuzzy = resolver.resolve("хрм")
        assert exact is not None
        assert stem is not None
        assert fuzzy is not None
        assert exact.confidence > stem.confidence > fuzzy.confidence


class TestUserAliases:
    """The user's own vocabulary, which the shipped dictionary does not outvote."""

    def test_a_user_alias_wins(self, resolver: AppResolver) -> None:
        """«почта» ships as Outlook; a user pointing it at Thunderbird keeps it."""
        assert resolver.resolve("почта") is not None
        custom = resolver.with_user_aliases({"почта": "thunderbird"})
        match = custom.resolve("почта")
        assert match is not None
        assert match.app_id == "thunderbird"
        assert match.confidence == 1.0

    def test_a_user_alias_may_name_a_program_nobody_shipped(self, resolver: AppResolver) -> None:
        """No entry and no scan, so the id is all there is — and that is enough."""
        custom = resolver.with_user_aliases({"работа": "my-crm"})
        match = custom.resolve("работа")
        assert match is not None
        assert (match.app_id, match.name, match.entry) == ("my-crm", "my-crm", None)
        assert match.target == ""

    def test_user_aliases_are_folded_like_any_other(self, resolver: AppResolver) -> None:
        custom = resolver.with_user_aliases({"Моя Почта!": CHROME})
        assert custom.resolve("моя почта") is not None

    def test_blank_aliases_are_dropped(self, resolver: AppResolver) -> None:
        custom = resolver.with_user_aliases({"  ": CHROME, "нормальный": CHROME})
        assert custom.user_aliases == {"нормальный": CHROME}

    def test_layering_returns_a_new_resolver(self, resolver: AppResolver) -> None:
        """A background scan finishing must not change a match already in flight."""
        custom = resolver.with_user_aliases({"почта": "thunderbird"})
        assert custom is not resolver
        assert resolver.user_aliases == {}


class TestInstalled:
    """What a scan adds, merged without the scan having to run."""

    def test_a_catalogued_program_gets_its_real_path(self, resolver: AppResolver) -> None:
        """The dictionary guesses ``chrome.exe``; the scan knows where it is."""
        merged = resolver.with_installed(
            [
                InstalledApp(
                    name="Google Chrome",
                    target=r"C:\Program Files\Google\Chrome\chrome.exe",
                    executable="chrome.exe",
                    source="app-paths",
                    catalog_id=CHROME,
                )
            ]
        )
        match = merged.resolve("хром")
        assert match is not None
        assert match.installed is not None
        assert match.target == r"C:\Program Files\Google\Chrome\chrome.exe"

    def test_an_unknown_program_becomes_reachable(self, resolver: AppResolver) -> None:
        """A Start-menu shortcut the dictionary never heard of still answers."""
        merged = resolver.with_installed(
            [InstalledApp(name="Хитрый Софт", target=r"C:\X\hitry.exe", executable="hitry.exe")]
        )
        match = merged.resolve("хитрый софт")
        assert match is not None
        assert match.app_id == "хитрый софт"
        assert match.entry is None
        assert match.target == r"C:\X\hitry.exe"

    def test_a_scan_does_not_outrank_the_dictionary(self, resolver: AppResolver) -> None:
        """A stray shortcut called «Хром» must not take «хром» from Chrome."""
        merged = resolver.with_installed(
            [InstalledApp(name="Хром", target=r"C:\Fake\fake.exe", executable="fake.exe")]
        )
        match = merged.resolve("хром")
        assert match is not None
        assert match.app_id == CHROME

    def test_without_a_scan_the_target_is_the_dictionary_guess(self, resolver: AppResolver) -> None:
        match = resolver.resolve("ворд")
        assert match is not None
        assert match.installed is None
        assert match.target == "winword.exe"

    def test_a_store_app_launches_by_aumid(self) -> None:
        """A Store application has no path at all, so the AUMID is the target."""
        entry = AppEntry(id="store-app", name="Store App", uwp="Some.App_8wek!App")
        match = AppResolver.from_apps([entry]).resolve("store app")
        assert match is not None
        assert match.target == "Some.App_8wek!App"

    def test_the_first_record_for_an_id_wins(self, resolver: AppResolver) -> None:
        """Sources are ordered cheapest and most trustworthy first."""
        merged = resolver.with_installed(
            [
                InstalledApp(name="Chrome", target="first.exe", catalog_id=CHROME),
                InstalledApp(name="Chrome", target="second.exe", catalog_id=CHROME),
            ]
        )
        assert merged.installed[CHROME].target == "first.exe"


class TestPrefix:
    """Where the program name ends, for a template whose slot is not last."""

    @pytest.mark.parametrize(
        ("phrase", "app_id", "rest"),
        [
            ("хром и найди погоду", "chrome", "и найди погоду"),
            ("открой хром", "chrome", ""),
            ("блокнот запиши мысль", "notepad", "запиши мысль"),
            # The «и» is gone, and that is the fuzzy pass rather than a mistake:
            # «блокнот и» is one edit from «блокнот» over nine characters, which
            # clears the tolerance. Harmless — the program is still right and a
            # leading conjunction is not something the next slot reads — but it
            # is the reason a remainder cannot be compared blindly.
            ("блокнот и запиши мысль", "notepad", "запиши мысль"),
        ],
    )
    def test_prefix_and_remainder(
        self, resolver: AppResolver, phrase: str, app_id: str, rest: str
    ) -> None:
        found = resolver.resolve_prefix(phrase)
        assert found is not None
        match, remainder = found
        assert (match.app_id, remainder) == (app_id, rest)

    def test_the_longest_name_wins(self, resolver: AppResolver) -> None:
        """«яндекс браузер» must beat «яндекс», or «браузер» is left as a tail."""
        found = resolver.resolve_prefix("яндекс браузер и открой почту")
        assert found is not None
        match, remainder = found
        assert match.app_id == "yandex-browser"
        assert remainder == "открой почту"

    def test_no_program_at_the_front(self, resolver: AppResolver) -> None:
        assert resolver.resolve_prefix("найди погоду в москве") is None


class TestLoading:
    """A dictionary is data that ships, so a problem in it is a packaging bug."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AppsError, match="cannot read"):
            load_apps(tmp_path / "нет.json")

    def test_not_json(self, tmp_path: Path) -> None:
        target = tmp_path / "apps.json"
        target.write_text("{ это не json", encoding="utf-8")
        with pytest.raises(AppsError, match="not valid JSON"):
            load_apps(target)

    def test_a_bad_record(self, tmp_path: Path) -> None:
        """A pydantic failure has to arrive as one Russian sentence, not a trace."""
        target = tmp_path / "apps.json"
        target.write_text(
            json.dumps({"schema_version": 1, "apps": [{"id": "ПЛОХОЙ ID", "name": "x"}]}),
            encoding="utf-8",
        )
        with pytest.raises(AppsError) as failure:
            load_apps(target)
        assert "заполнен неверно" in failure.value.user_message

    def test_duplicate_ids(self, tmp_path: Path) -> None:
        target = tmp_path / "apps.json"
        payload = {
            "schema_version": 1,
            "apps": [{"id": "same", "name": "A"}, {"id": "same", "name": "B"}],
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(AppsError):
            load_apps(target)

    def test_a_newer_schema_is_refused(self, tmp_path: Path) -> None:
        """Reading a v2 file as v1 would drop whatever v2 added, silently."""
        target = tmp_path / "apps.json"
        target.write_text(
            json.dumps({"schema_version": APPS_SCHEMA_VERSION + 1, "apps": []}),
            encoding="utf-8",
        )
        with pytest.raises(AppsError, match="newer than supported"):
            load_apps(target)

    def test_a_utf8_bom_is_tolerated(self, tmp_path: Path) -> None:
        """Notepad writes one, and a user editing the file by hand is expected."""
        target = tmp_path / "apps.json"
        target.write_text(
            json.dumps({"schema_version": 1, "apps": []}),
            encoding="utf-8-sig",
        )
        assert load_apps(target).apps == ()

    def test_entries_are_frozen(self) -> None:
        entry = CATALOG.apps[0]
        with pytest.raises(ValueError, match="frozen"):
            entry.name = "иначе"  # type: ignore[misc]


class TestScanner:
    """The half that needs Windows. Off it, calling this must be a no-op."""

    @pytest.mark.skipif(sys.platform == "win32", reason="проверяет поведение вне Windows")
    def test_off_windows_it_returns_nothing(self) -> None:
        """Every caller runs on both, so the guard lives here, not in each of them."""
        assert scan_installed_apps() == []

    @pytest.mark.skipif(sys.platform != "win32", reason="сканер читает реестр Windows")
    def test_on_windows_it_finds_something(self) -> None:
        """A Windows machine has a Start menu; finding nothing at all is a bug."""
        found = scan_installed_apps(limit=50)
        assert found
        assert all(app.name and app.target for app in found)

    @pytest.mark.skipif(sys.platform != "win32", reason="сканер читает реестр Windows")
    def test_the_limit_is_honoured(self) -> None:
        assert len(scan_installed_apps(limit=5)) <= 5

    @pytest.mark.skipif(sys.platform != "win32", reason="сканер читает реестр Windows")
    def test_what_it_finds_can_be_merged(self) -> None:
        """The point of the scan: the resolver answers with a real path afterwards."""
        merged = AppResolver.from_apps(CATALOG.apps).with_installed(scan_installed_apps(limit=200))
        assert len(merged.aliases) >= len(AppResolver.from_apps(CATALOG.apps).aliases)
