"""Task 20: what this machine can launch, and which of it the user just named.

:mod:`ayris.actions.system.app_index` has two halves and only one of them needs
Windows. The scanners read the registry, the Start menu and the shell namespace;
everything they hand back goes through a parser, and the parsers are ordinary
functions over ordinary dicts. So the interesting part — what counts as
launchable, which record wins when two describe the same program, what «браузер»
means on a machine with two of them — is checked here, on Linux, in CI.

Three rules are the whole design and each is asserted directly.

*A source decides a tie, not the order of the scan.* Chrome is found in the Start
menu, in ``App Paths`` and in ``PATH``. Which of the three the user is offered
must not depend on which scanner ran first, so :func:`~…app_index.dedupe` folds
them by weight and the weights are ordered on purpose.

*Ambiguity is a question, not a guess.* Two programs equally close to what was
said, neither launched more often than the other, and the resolver raises
:class:`~…app_index.AppAmbiguous` carrying both names. The launch counter is what
breaks the tie next time, which is why it survives a restart.

*A user alias ends the argument.* «почта» ships pointing at Outlook and «браузер»
is a category word; someone who bound either of them to a particular program has
already answered the question those two branches would otherwise raise.

Groups:

* :class:`TestIndexSource` — the weights and the Russian titles.
* :class:`TestAppPathsParser` — the registry key that names executables.
* :class:`TestUninstallParser` — display names, and the three filters.
* :class:`TestShortcutParser` — Start-menu ``.lnk``, uninstallers dropped.
* :class:`TestPathParser` — executables in ``PATH``.
* :class:`TestUwpParser` — Store apps, and why a desktop entry is refused.
* :class:`TestLinkCatalog` — attaching a dictionary id, by three keys.
* :class:`TestDedupe` — collapsing records, by target and by program.
* :class:`TestPhraseKey` — «Открой Хром!» and «хром» are one key.
* :class:`TestIndexedAppJson` — the cache record, both ways.
* :class:`TestSnapshotJson` — the cache file, and what it refuses.
* :class:`TestCandidate` — target, rank, installed.
* :class:`TestResolver` — the phrases, the categories, the tie-break.
* :class:`TestUserAliases` — the regression: an alias must not be asked about.
* :class:`TestAppIndexCache` — TTL, the file, the launch counter.
* :class:`TestAppIndexRefresh` — the background scan and its failures.
* :class:`TestGlobalIndex` — the process-wide instance.
* :class:`TestScanner` — the Windows half, skipped elsewhere.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ayris.actions.system.app_index import (
    AMBIGUITY_GAP,
    DEFAULT_TTL_S,
    GENERIC_ALIASES,
    INDEX_FILE_NAME,
    INDEX_SCHEMA_VERSION,
    STORE_PREFIX,
    AppAmbiguous,
    AppCandidate,
    AppIndex,
    AppIndexResolver,
    AppIndexSnapshot,
    AppNotFound,
    IndexedApp,
    IndexSource,
    dedupe,
    get_app_index,
    link_catalog,
    parse_app_paths_entry,
    parse_path_executable,
    parse_shortcut,
    parse_uninstall_entry,
    parse_uwp_entry,
    phrase_key,
    scan_installed,
    set_app_index,
)
from ayris.nlu.apps import AppEntry, AppsFile, load_apps
from ayris.utils.logger import ROOT_LOGGER_NAME

pytestmark = pytest.mark.unit

#: The dictionary as it ships. Read once, and used instead of invented entries:
#: a resolver that only works against a fixture proves nothing about the one the
#: application will build.
CATALOG: AppsFile = load_apps()

#: The AUMID of «Параметры Windows», taken from the dictionary rather than copied:
#: the point of the Store tests is that a real moniker survives the round trip,
#: and a rename in the dictionary should show up here rather than pass silently.
SETTINGS_AUMID = next(entry.uwp for entry in CATALOG.apps if entry.id == "settings")

#: A plausible scan of a real machine: two browsers, an editor, a desktop program
#: from the registry, a Store application and something nobody shipped a
#: dictionary entry for.
SNAPSHOT_APPS = (
    IndexedApp(
        name="Google Chrome",
        target=r"C:\Program Files\Google\Chrome\chrome.exe",
        source=IndexSource.START_MENU,
        executable="chrome.exe",
        catalog_id="chrome",
    ),
    IndexedApp(
        name="Mozilla Firefox",
        target=r"C:\Program Files\Mozilla Firefox\firefox.exe",
        source=IndexSource.APP_PATHS,
        executable="firefox.exe",
        catalog_id="firefox",
    ),
    IndexedApp(
        name="Visual Studio Code",
        target=r"C:\Users\u\AppData\Local\Programs\VS Code\Code.exe",
        source=IndexSource.START_MENU,
        executable="Code.exe",
        catalog_id="vscode",
    ),
    IndexedApp(
        name="Диспетчер задач",
        target=r"C:\Windows\System32\taskmgr.exe",
        source=IndexSource.APP_PATHS,
        executable="taskmgr.exe",
        catalog_id="task-manager",
    ),
    IndexedApp(
        name="Параметры",
        target=f"{STORE_PREFIX}{SETTINGS_AUMID}",
        source=IndexSource.UWP,
        aumid=SETTINGS_AUMID,
        catalog_id="settings",
    ),
    IndexedApp(
        name="Хитрый Софт",
        target=r"C:\Program Files\Hitry\hitry.exe",
        source=IndexSource.START_MENU,
        executable="hitry.exe",
    ),
)

#: Two dictionary entries one letter apart. Invented deliberately: the shipped
#: dictionary has no genuine near-collision between *installed* programs, and the
#: ambiguity threshold is the one rule that needs one.
TWIN_ENTRIES = (
    AppEntry(
        id="viber",
        name="Viber",
        kind="messenger",
        executables=("viber.exe",),
        aliases=("вайбер",),
    ),
    AppEntry(
        id="vaibor",
        name="Вайбор",
        kind="messenger",
        executables=("vaibor.exe",),
        aliases=("вайбор",),
    ),
)

TWIN_APPS = (
    IndexedApp(
        name="Viber",
        target=r"C:\V\viber.exe",
        source=IndexSource.START_MENU,
        executable="viber.exe",
        catalog_id="viber",
    ),
    IndexedApp(
        name="Вайбор",
        target=r"C:\B\vaibor.exe",
        source=IndexSource.START_MENU,
        executable="vaibor.exe",
        catalog_id="vaibor",
    ),
)


@pytest.fixture(scope="module")
def snapshot() -> AppIndexSnapshot:
    """The scan above, stamped as fresh. Immutable, so it is shared."""
    return AppIndexSnapshot(apps=SNAPSHOT_APPS, scanned_at=1_000.0)


@pytest.fixture(scope="module")
def resolver(snapshot: AppIndexSnapshot) -> AppIndexResolver:
    """That scan indexed against the shipped dictionary."""
    return AppIndexResolver.build(snapshot, entries=CATALOG.apps)


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree.

    ``setup_logging`` sets ``propagate = False`` on that logger and ``caplog``
    listens on the interpreter root, so a plain ``caplog.at_level`` sees nothing
    once any earlier test in the run has configured logging.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


class TestIndexSource:
    """The weights are the tie-break, so their order is part of the contract."""

    def test_weights_are_ordered(self) -> None:
        """Start menu over App Paths over Store over uninstall over PATH."""
        order = [
            IndexSource.START_MENU,
            IndexSource.APP_PATHS,
            IndexSource.UWP,
            IndexSource.UNINSTALL,
            IndexSource.PATH,
        ]
        weights = [source.weight for source in order]
        assert weights == sorted(weights, reverse=True)
        assert len(set(weights)) == len(weights)

    def test_every_source_has_a_russian_title(self) -> None:
        """The settings window lists the sources; a missing title is a blank row."""
        for source in IndexSource:
            assert source.title_ru
            assert any("а" <= letter.lower() <= "я" for letter in source.title_ru)


class TestAppPathsParser:
    """``App Paths``: the key names the executable, the default value its path."""

    def test_a_normal_entry(self) -> None:
        app = parse_app_paths_entry(
            "chrome.exe",
            {
                "": r'"C:\Program Files\Google\Chrome\chrome.exe"',
                "Path": r"C:\Program Files\Google\Chrome",
            },
        )
        assert app is not None
        assert app.name == "chrome"
        assert app.target == r"C:\Program Files\Google\Chrome\chrome.exe"
        assert app.executable == "chrome.exe"
        assert app.working_dir == r"C:\Program Files\Google\Chrome"
        assert app.source is IndexSource.APP_PATHS

    def test_the_icon_defaults_to_the_target(self) -> None:
        """An executable is its own icon source, which is what the shell assumes."""
        app = parse_app_paths_entry("code.exe", {"": r"C:\VS\code.exe"})
        assert app is not None
        assert app.icon == app.target

    def test_a_working_directory_is_optional(self) -> None:
        app = parse_app_paths_entry("notepad.exe", {"": r"C:\Windows\notepad.exe"})
        assert app is not None
        assert app.working_dir == ""

    def test_a_key_without_a_default_value_is_dropped(self) -> None:
        """Some installers leave the key behind with nothing in it."""
        assert parse_app_paths_entry("a.exe", {}) is None
        assert parse_app_paths_entry("a.exe", {"": "   "}) is None

    def test_an_empty_key_name_is_dropped(self) -> None:
        assert parse_app_paths_entry("", {"": r"C:\x.exe"}) is None


class TestUninstallParser:
    """The uninstall keys: the display names, and the three filters."""

    def test_a_display_name_and_an_icon(self) -> None:
        """«Яндекс Браузер» is a name no executable carries."""
        app = parse_uninstall_entry(
            {
                "DisplayName": "Яндекс Браузер",
                "DisplayIcon": r"C:\Yandex\browser.exe,0",
                "InstallLocation": r"C:\Yandex",
            }
        )
        assert app is not None
        assert app.name == "Яндекс Браузер"
        assert app.target == r"C:\Yandex\browser.exe"
        assert app.working_dir == r"C:\Yandex"
        assert app.source is IndexSource.UNINSTALL

    def test_the_icon_index_is_cut_from_the_target_but_kept_in_the_icon(self) -> None:
        """``,0`` selects an icon inside the file; ``ShellExecuteW`` chokes on it."""
        app = parse_uninstall_entry({"DisplayName": "P", "DisplayIcon": "/opt/p.exe,3"})
        assert app is not None
        assert app.target == "/opt/p.exe"
        assert app.icon == "/opt/p.exe,3"

    def test_quotes_around_the_icon_are_stripped(self) -> None:
        app = parse_uninstall_entry({"DisplayName": "P", "DisplayIcon": '"/opt/p.exe",1'})
        assert app is not None
        assert app.target == "/opt/p.exe"

    def test_the_executable_is_the_file_name(self) -> None:
        """A posix path in the fixture: the tests run on Linux, where
        ``PureWindowsPath`` semantics are not what :class:`Path` gives."""
        app = parse_uninstall_entry({"DisplayName": "P", "DisplayIcon": "/opt/vendor/p.exe"})
        assert app is not None
        assert app.executable == "p.exe"

    def test_a_system_component_is_dropped(self) -> None:
        """Runtimes and driver packages, which nobody launches by voice."""
        values = {"DisplayName": "VC++ Runtime", "DisplayIcon": "/opt/vc.exe", "SystemComponent": 1}
        assert parse_uninstall_entry(values) is None

    def test_a_system_component_written_as_text_is_still_dropped(self) -> None:
        """Registry values arrive as ``int`` or as ``str`` depending on the writer."""
        values = {"DisplayName": "X", "DisplayIcon": "/opt/x.exe", "SystemComponent": "1"}
        assert parse_uninstall_entry(values) is None

    def test_a_broken_system_component_flag_is_not_a_crash(self) -> None:
        values = {"DisplayName": "X", "DisplayIcon": "/opt/x.exe", "SystemComponent": "да"}
        assert parse_uninstall_entry(values) is not None

    def test_no_display_name_is_dropped(self) -> None:
        assert parse_uninstall_entry({"DisplayIcon": "/opt/p.exe"}) is None
        assert parse_uninstall_entry({"DisplayName": "  ", "DisplayIcon": "/opt/p.exe"}) is None

    def test_an_icon_that_is_not_an_executable_is_dropped(self) -> None:
        """A ``.ico`` or an MSI cache path points at no program to start."""
        assert parse_uninstall_entry({"DisplayName": "P", "DisplayIcon": "/opt/p.ico"}) is None
        assert parse_uninstall_entry({"DisplayName": "P"}) is None


class TestShortcutParser:
    """Start-menu shortcuts, labelled by file name and never resolved."""

    def test_a_shortcut_is_its_own_target(self) -> None:
        """``ShellExecuteW`` reads the ``.lnk`` itself, COM and all."""
        app = parse_shortcut(Path("/menu/Telegram.lnk"))
        assert app is not None
        assert app.name == "Telegram"
        assert app.target == "/menu/Telegram.lnk"
        assert app.icon == "/menu/Telegram.lnk"
        assert app.source is IndexSource.START_MENU
        assert app.is_shortcut

    @pytest.mark.parametrize(
        "name",
        [
            "Uninstall Telegram.lnk",
            "uninstall.lnk",
            "Удалить Оперу.lnk",
            "Деинсталляция.lnk",
            "Remove Steam.lnk",
        ],
    )
    def test_an_uninstaller_is_dropped(self, name: str) -> None:
        """Every vendor ships one, and a wrong hit here deletes a program."""
        assert parse_shortcut(Path("/menu") / name) is None

    def test_something_that_is_not_a_shortcut_is_dropped(self) -> None:
        assert parse_shortcut(Path("/menu/readme.txt")) is None
        assert parse_shortcut(Path("/menu/.lnk")) is None


class TestPathParser:
    """``PATH``: what makes «запусти ffmpeg» work at all."""

    def test_an_executable(self) -> None:
        app = parse_path_executable(Path("/tools/bin/ffmpeg.exe"))
        assert app is not None
        assert (app.name, app.executable, app.source) == ("ffmpeg", "ffmpeg.exe", IndexSource.PATH)

    @pytest.mark.parametrize("name", ["lib.dll", "script.cmd", "readme", "python3.11"])
    def test_only_executables(self, name: str) -> None:
        """A ``.cmd`` in ``PATH`` is a build helper, not something anyone says."""
        assert parse_path_executable(Path("/tools") / name) is None


class TestUwpParser:
    """``shell:AppsFolder``: a Store app has no path, only a moniker."""

    def test_a_store_application(self) -> None:
        app = parse_uwp_entry("Параметры", SETTINGS_AUMID)
        assert app is not None
        assert app.aumid == SETTINGS_AUMID
        assert app.target == f"{STORE_PREFIX}{SETTINGS_AUMID}"
        assert app.is_store
        assert app.source is IndexSource.UWP

    def test_the_prefix_is_not_doubled(self) -> None:
        """The shell hands back the moniker already prefixed, sometimes."""
        app = parse_uwp_entry("Калькулятор", f"{STORE_PREFIX}Microsoft.WindowsCalculator_8wek!App")
        assert app is not None
        assert app.aumid == "Microsoft.WindowsCalculator_8wek!App"
        assert app.target.count(STORE_PREFIX) == 1

    def test_a_desktop_entry_is_refused(self) -> None:
        """No ``!``, so it is a desktop program the Start-menu scan already has —
        with a path that can be launched directly."""
        assert parse_uwp_entry("Проводник", "Microsoft.Windows.Explorer") is None

    def test_a_nameless_item_is_refused(self) -> None:
        assert parse_uwp_entry("  ", "Some.Package_8wek!App") is None

    def test_a_store_app_has_no_icon(self) -> None:
        """Only the shell knows how to draw one, so the field stays empty."""
        app = parse_uwp_entry("Параметры", SETTINGS_AUMID)
        assert app is not None
        assert app.icon == ""


class TestLinkCatalog:
    """The scan finds programs; the dictionary is what gives them Russian names."""

    def test_linked_by_executable(self) -> None:
        """The usual case: ``chrome.exe`` in ``App Paths`` is the Chrome entry."""
        found = [IndexedApp(name="chrome", target=r"C:\c\chrome.exe", source=IndexSource.APP_PATHS)]
        linked = link_catalog(found, CATALOG.apps)
        assert linked[0].catalog_id == "chrome"

    def test_linked_by_aumid(self) -> None:
        """A Store app has no executable to match on, only its moniker."""
        found = [
            IndexedApp(
                name="Что-то из Store",
                target=f"{STORE_PREFIX}{SETTINGS_AUMID}",
                source=IndexSource.UWP,
                aumid=SETTINGS_AUMID.upper(),
            )
        ]
        assert link_catalog(found, CATALOG.apps)[0].catalog_id == "settings"

    def test_linked_by_name(self) -> None:
        """A Start-menu shortcut has neither: «Диспетчер задач.lnk» is a label."""
        found = [
            IndexedApp(name="Диспетчер задач", target="/m/t.lnk", source=IndexSource.START_MENU)
        ]
        assert link_catalog(found, CATALOG.apps)[0].catalog_id == "task-manager"

    def test_linked_by_alias(self) -> None:
        """Aliases count as names, so «Хром.lnk» links as well as «Google Chrome.lnk»."""
        found = [IndexedApp(name="Хром", target="/m/h.lnk", source=IndexSource.START_MENU)]
        assert link_catalog(found, CATALOG.apps)[0].catalog_id == "chrome"

    def test_an_existing_id_is_left_alone(self) -> None:
        """A record that already knows what it is must not be re-guessed."""
        found = [
            IndexedApp(
                name="chrome",
                target=r"C:\c\chrome.exe",
                source=IndexSource.APP_PATHS,
                catalog_id="mine",
            )
        ]
        assert link_catalog(found, CATALOG.apps)[0].catalog_id == "mine"

    def test_an_unknown_program_stays_unlinked(self) -> None:
        """Most of a real machine is not in a 34-entry dictionary."""
        found = [IndexedApp(name="Хитрый Софт", target="/h/hitry.exe", source=IndexSource.PATH)]
        assert link_catalog(found, CATALOG.apps)[0].catalog_id == ""

    def test_nothing_is_matched_fuzzily(self) -> None:
        """The scan runs unattended: a wrong link points «открой хром» at a stranger."""
        found = [IndexedApp(name="Гугл Хрома", target="/g/gh.exe", source=IndexSource.START_MENU)]
        assert link_catalog(found, CATALOG.apps)[0].catalog_id == ""


class TestDedupe:
    """One program, five sources. Which record the user is offered is a rule."""

    def test_the_same_target_from_two_sources_folds_to_the_better_one(self) -> None:
        """Chrome is in ``App Paths`` and in ``PATH``; the label comes from neither
        by accident."""
        low = IndexedApp(name="chrome", target=r"C:\c\chrome.exe", source=IndexSource.PATH)
        high = IndexedApp(name="chrome", target=r"C:\c\chrome.exe", source=IndexSource.APP_PATHS)
        assert dedupe([low, high]) == (high,)
        assert dedupe([high, low]) == (high,), "scan order must not decide"

    def test_two_paths_to_one_catalogued_program_fold_as_well(self) -> None:
        """The shortcut and the executable are different targets and one program;
        offering both would read as «Chrome или Chrome?»."""
        shortcut = IndexedApp(
            name="Google Chrome",
            target=r"C:\menu\Google Chrome.lnk",
            source=IndexSource.START_MENU,
            catalog_id="chrome",
        )
        executable = IndexedApp(
            name="chrome",
            target=r"C:\c\chrome.exe",
            source=IndexSource.APP_PATHS,
            catalog_id="chrome",
        )
        assert dedupe([executable, shortcut]) == (shortcut,)

    def test_uncatalogued_programs_are_kept_apart(self) -> None:
        """Two unknown programs share the empty catalogue id and are still two."""
        first = IndexedApp(name="A", target="/a/a.exe", source=IndexSource.PATH)
        second = IndexedApp(name="B", target="/b/b.exe", source=IndexSource.PATH)
        assert len(dedupe([first, second])) == 2

    def test_a_store_app_is_identified_by_its_moniker(self) -> None:
        """Two records of one package, written with and without the prefix."""
        bare = IndexedApp(
            name="Параметры",
            target=SETTINGS_AUMID,
            source=IndexSource.UWP,
            aumid=SETTINGS_AUMID,
        )
        prefixed = IndexedApp(
            name="Параметры Windows",
            target=f"{STORE_PREFIX}{SETTINGS_AUMID}",
            source=IndexSource.UWP,
            aumid=SETTINGS_AUMID,
        )
        assert len(dedupe([bare, prefixed])) == 1

    def test_the_order_is_by_source_then_by_name(self) -> None:
        """A stable order, so the settings list does not reshuffle between scans."""
        apps = [
            IndexedApp(name="Яблоко", target="/1.exe", source=IndexSource.PATH),
            IndexedApp(name="бета", target="/2.exe", source=IndexSource.START_MENU),
            IndexedApp(name="Альфа", target="/3.exe", source=IndexSource.START_MENU),
        ]
        assert [app.name for app in dedupe(apps)] == ["Альфа", "бета", "Яблоко"]

    def test_a_record_without_a_target_is_dropped(self) -> None:
        assert dedupe([IndexedApp(name="A", target="", source=IndexSource.PATH)]) == ()

    def test_the_limit_is_enforced_and_logged(self, ayris_log: pytest.LogCaptureFixture) -> None:
        """``PATH`` on a developer machine alone can run into the thousands."""
        apps = [
            IndexedApp(name=f"p{number:03d}", target=f"/p/{number}.exe", source=IndexSource.PATH)
            for number in range(10)
        ]
        result = dedupe(apps, limit=4)
        assert [app.name for app in result] == ["p000", "p001", "p002", "p003"]
        assert "урезан" in ayris_log.text


class TestPhraseKey:
    """The fold that makes «Открой Хром!» and «хром» the same lookup."""

    @pytest.mark.parametrize(
        "phrase",
        ["хром", "Хром", "  ХРОМ  ", "Открой хром", "запусти Хром!", "включи хром"],
    )
    def test_one_key(self, phrase: str) -> None:
        assert phrase_key(phrase) == "хром"

    def test_a_preposition_is_dropped(self) -> None:
        """Slots at the end of a template arrive with the preposition attached."""
        assert phrase_key("в хроме") == "хроме"

    def test_several_leading_words_are_dropped(self) -> None:
        assert phrase_key("открой в хроме") == "хроме"

    def test_the_last_word_survives_even_when_it_is_a_verb(self) -> None:
        """«открой» alone is not a program name, but stripping it leaves nothing
        to look up — and an empty key is what the resolver refuses on."""
        assert phrase_key("открой") == "открой"

    def test_nothing_is_nothing(self) -> None:
        assert phrase_key("   ") == ""
        assert phrase_key("!!!") == ""

    def test_the_rest_of_the_phrase_is_kept(self) -> None:
        assert phrase_key("Открой Visual Studio Code") == "visual studio code"


class TestIndexedAppJson:
    """The cache record. Written on every scan, read on every start."""

    def test_a_round_trip_keeps_everything(self) -> None:
        for app in SNAPSHOT_APPS:
            assert IndexedApp.from_json(app.to_json()) == app

    def test_empty_fields_are_not_written(self) -> None:
        """Four thousand records: the omitted keys are most of the file."""
        payload = IndexedApp(name="A", target="/a.exe", source=IndexSource.PATH).to_json()
        assert set(payload) == {"name", "target", "source"}

    def test_the_source_is_written_as_its_value(self) -> None:
        """A :class:`~enum.StrEnum` member, so the file stays readable by hand."""
        payload = SNAPSHOT_APPS[0].to_json()
        assert payload["source"] == "start-menu"

    @pytest.mark.parametrize(
        "payload",
        [
            {"target": "/a.exe", "source": "path"},
            {"name": "  ", "target": "/a.exe", "source": "path"},
            {"name": "A", "source": "path"},
            {"name": "A", "target": "  ", "source": "path"},
            {"name": "A", "target": "/a.exe"},
            {"name": "A", "target": "/a.exe", "source": "registry"},
        ],
    )
    def test_a_damaged_record_is_dropped_not_repaired(self, payload: dict[str, str]) -> None:
        """The cache is rebuilt from the machine in seconds; guessing is not worth it."""
        assert IndexedApp.from_json(payload) is None


class TestSnapshotJson:
    """The cache file: a version, a timestamp, the records and the counters."""

    def test_a_round_trip(self) -> None:
        original = AppIndexSnapshot(
            apps=SNAPSHOT_APPS,
            scanned_at=1_700_000_000.0,
            launches={"chrome": 7},
        )
        restored = AppIndexSnapshot.from_json(original.to_json())
        assert restored is not None
        assert restored.apps == original.apps
        assert restored.scanned_at == original.scanned_at
        assert dict(restored.launches) == {"chrome": 7}

    def test_the_version_is_written(self) -> None:
        assert AppIndexSnapshot().to_json()["schema_version"] == INDEX_SCHEMA_VERSION

    def test_another_version_is_refused_whole(self) -> None:
        """A field could have changed meaning, so nothing in the file is trusted."""
        payload = AppIndexSnapshot(apps=SNAPSHOT_APPS).to_json()
        payload["schema_version"] = INDEX_SCHEMA_VERSION + 1
        assert AppIndexSnapshot.from_json(payload) is None
        assert AppIndexSnapshot.from_json({"scanned_at": 1.0}) is None

    def test_a_broken_record_does_not_cost_the_others(self) -> None:
        """One truncated line is not a reason to rescan the whole machine."""
        payload = AppIndexSnapshot(apps=SNAPSHOT_APPS).to_json()
        payload["apps"] = [payload["apps"][0], {"name": "A"}, "мусор", payload["apps"][1]]
        restored = AppIndexSnapshot.from_json(payload)
        assert restored is not None
        assert len(restored.apps) == 2

    def test_apps_that_are_not_a_list_are_refused(self) -> None:
        assert (
            AppIndexSnapshot.from_json({"schema_version": INDEX_SCHEMA_VERSION, "apps": {}}) is None
        )

    @pytest.mark.parametrize("count", [0, -3, "5", 1.5, None])
    def test_a_nonsense_counter_is_dropped(self, count: object) -> None:
        """A counter breaks a tie between two programs; a wrong one is worse than none."""
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "apps": [],
            "launches": {"chrome": count},
        }
        restored = AppIndexSnapshot.from_json(payload)
        assert restored is not None
        assert dict(restored.launches) == {}

    def test_launches_that_are_not_a_mapping_are_ignored(self) -> None:
        payload = {"schema_version": INDEX_SCHEMA_VERSION, "apps": [], "launches": []}
        restored = AppIndexSnapshot.from_json(payload)
        assert restored is not None
        assert dict(restored.launches) == {}

    def test_age_and_staleness(self) -> None:
        snapshot = AppIndexSnapshot(scanned_at=1_000.0)
        assert snapshot.age_s(1_060.0) == 60.0
        assert snapshot.is_stale(ttl_s=30.0, now=1_060.0)
        assert not snapshot.is_stale(ttl_s=120.0, now=1_060.0)

    def test_a_clock_that_jumped_backwards_is_not_stale(self) -> None:
        """A DST change or an NTP correction must not trigger a rescan storm."""
        snapshot = AppIndexSnapshot(scanned_at=2_000.0)
        assert snapshot.age_s(1_000.0) == 0.0
        assert not snapshot.is_stale(ttl_s=30.0, now=1_000.0)

    def test_a_snapshot_that_never_scanned_is_always_stale(self) -> None:
        assert AppIndexSnapshot().is_stale(ttl_s=DEFAULT_TTL_S, now=0.0)

    def test_with_launches_copies(self) -> None:
        """Immutable on purpose: a match in flight keeps its own picture."""
        original = AppIndexSnapshot(apps=SNAPSHOT_APPS, scanned_at=1.0)
        updated = original.with_launches({"chrome": 1})
        assert dict(original.launches) == {}
        assert updated.apps is original.apps


class TestCandidate:
    """What the resolver hands to the launcher."""

    def test_an_installed_program_launches_by_its_scanned_target(self) -> None:
        candidate = AppCandidate(
            app_id="chrome", name="Chrome", confidence=1.0, app=SNAPSHOT_APPS[0]
        )
        assert candidate.installed
        assert candidate.target == SNAPSHOT_APPS[0].target
        assert candidate.source_weight == IndexSource.START_MENU.weight

    def test_an_uninstalled_program_falls_back_to_the_dictionary(self) -> None:
        """«открой калькулятор» has to work on a machine whose Start menu the scan
        could not read: ``calc.exe`` is resolved by the shell itself."""
        entry = next(item for item in CATALOG.apps if item.id == "calculator")
        candidate = AppCandidate(app_id="calculator", name=entry.name, confidence=1.0, entry=entry)
        assert not candidate.installed
        assert candidate.source_weight == 0
        assert candidate.target == entry.primary_executable

    def test_a_store_program_falls_back_to_its_moniker(self) -> None:
        entry = next(item for item in CATALOG.apps if item.id == "settings")
        candidate = AppCandidate(app_id="settings", name=entry.name, confidence=1.0, entry=entry)
        assert candidate.target == f"{STORE_PREFIX}{SETTINGS_AUMID}"

    def test_nothing_known_launches_nothing(self) -> None:
        assert AppCandidate(app_id="x", name="X", confidence=0.5).target == ""

    def test_the_rank_prefers_confidence_then_use_then_source(self) -> None:
        """Three levels, in that order — a guess that was launched often still
        loses to an exact match."""
        sure = AppCandidate(app_id="a", name="A", confidence=1.0)
        used = AppCandidate(app_id="b", name="B", confidence=0.9, launches=50)
        assert sure.rank > used.rank
        often = AppCandidate(app_id="c", name="C", confidence=0.9, launches=1)
        assert used.rank > often.rank
        found = AppCandidate(app_id="d", name="D", confidence=0.9, launches=1, app=SNAPSHOT_APPS[0])
        assert found.rank > often.rank


class TestResolver:
    """«Какое приложение открыть» — the whole point of the module."""

    @pytest.mark.parametrize(
        ("phrase", "app_id"),
        [
            ("хром", "chrome"),
            ("Открой хром", "chrome"),
            ("chrome", "chrome"),
            ("гугл хром", "chrome"),
            ("вс код", "vscode"),
            ("code", "vscode"),
            ("диспетчер задач", "task-manager"),
            ("настройки", "settings"),
        ],
    )
    def test_a_named_program(self, resolver: AppIndexResolver, phrase: str, app_id: str) -> None:
        """Cyrillic, Latin and the verb in front all reach the same program."""
        assert resolver.resolve(phrase).app_id == app_id

    def test_a_typo_still_resolves(self, resolver: AppIndexResolver) -> None:
        """«хрм» is what a recogniser does to «хром» on a bad microphone; the
        confidence drops but the answer is the same."""
        candidate = resolver.resolve("хрм")
        assert candidate.app_id == "chrome"
        assert candidate.confidence < 1.0

    def test_a_program_the_dictionary_never_heard_of(self, resolver: AppIndexResolver) -> None:
        """Most of a real machine: the id falls back to the folded name."""
        candidate = resolver.resolve("Хитрый Софт")
        assert candidate.name == "Хитрый Софт"
        assert candidate.installed
        assert candidate.entry is None

    def test_a_program_in_the_dictionary_but_not_installed(
        self, resolver: AppIndexResolver
    ) -> None:
        """Answered anyway: the shell may still find it, and «не установлен» is a
        better message when the launch fails than «не знаю такого»."""
        candidate = resolver.resolve("фотошоп")
        assert candidate.app_id == "photoshop"
        assert not candidate.installed

    def test_nothing_recognisable(self, resolver: AppIndexResolver) -> None:
        with pytest.raises(AppNotFound) as caught:
            resolver.resolve("непонятно что")
        assert caught.value.user_message == "Не нашла приложение «непонятно что»."

    @pytest.mark.parametrize("phrase", ["", "   ", "!!!"])
    def test_an_empty_phrase_is_a_different_message(
        self, resolver: AppIndexResolver, phrase: str
    ) -> None:
        """«открой» with the slot unfilled: nothing was named, so nothing was missed."""
        with pytest.raises(AppNotFound) as caught:
            resolver.resolve(phrase)
        assert caught.value.user_message == "Не поняла, какое приложение открыть."

    def test_a_category_with_one_answer(self, resolver: AppIndexResolver) -> None:
        """One editor installed, so «редактор» needs no question."""
        assert resolver.resolve("редактор").app_id == "vscode"

    @pytest.mark.parametrize("phrase", ["браузер", "интернет", "в браузере"])
    def test_a_category_with_two_answers_asks(
        self, resolver: AppIndexResolver, phrase: str
    ) -> None:
        """Chrome and Firefox, neither used more than the other. A question is
        cheap; opening the wrong browser is not."""
        with pytest.raises(AppAmbiguous) as caught:
            resolver.resolve(phrase)
        assert caught.value.options == ("Google Chrome", "Mozilla Firefox")
        assert caught.value.user_message == (
            "Какое приложение открыть: Google Chrome или Mozilla Firefox?"
        )

    def test_the_launch_history_answers_the_category(self, snapshot: AppIndexSnapshot) -> None:
        """The counter is what makes the second «открой браузер» silent."""
        resolver = AppIndexResolver.build(
            snapshot.with_launches({"firefox": 3}),
            entries=CATALOG.apps,
        )
        assert resolver.resolve("браузер").app_id == "firefox"

    def test_a_category_nothing_answers(self, resolver: AppIndexResolver) -> None:
        """No messenger on this machine, and «мессенджер» names no program to guess."""
        with pytest.raises(AppNotFound) as caught:
            resolver.resolve("мессенджер")
        assert caught.value.user_message == "Не нашла ни одного приложения: «мессенджер»."

    def test_every_generic_alias_names_a_category_the_dictionary_uses(self) -> None:
        """A typo here turns a category word into «не нашла приложение»."""
        kinds = {entry.kind for entry in CATALOG.apps if entry.kind}
        assert set(GENERIC_ALIASES.values()) <= kinds

    def test_two_close_programs_are_a_question(self) -> None:
        """The threshold in the flesh: one letter apart, both installed, neither
        used. Invented entries, because the shipped dictionary has no such pair."""
        resolver = AppIndexResolver.build(
            AppIndexSnapshot(apps=TWIN_APPS, scanned_at=1_000.0),
            entries=TWIN_ENTRIES,
        )
        with pytest.raises(AppAmbiguous) as caught:
            resolver.resolve("вайбр")
        assert caught.value.options == ("Viber", "Вайбор")
        assert caught.value.user_message == "Какое приложение открыть: Viber или Вайбор?"

    def test_an_exact_alias_is_never_a_question(self) -> None:
        """«вайбер» is one of them exactly; the near-miss of the other is irrelevant."""
        resolver = AppIndexResolver.build(
            AppIndexSnapshot(apps=TWIN_APPS, scanned_at=1_000.0),
            entries=TWIN_ENTRIES,
        )
        assert resolver.resolve("вайбер").app_id == "viber"
        assert resolver.resolve("вайбор").app_id == "vaibor"

    def test_one_launch_settles_the_pair(self) -> None:
        """A rival used less often than the winner is dropped, not offered: someone
        who opens Chrome every day should not be asked about Chromium."""
        resolver = AppIndexResolver.build(
            AppIndexSnapshot(apps=TWIN_APPS, scanned_at=1_000.0, launches={"viber": 1}),
            entries=TWIN_ENTRIES,
        )
        assert resolver.resolve("вайбр").app_id == "viber"
        assert resolver.resolve("мессенджер").app_id == "viber"

    def test_the_gap_is_what_makes_a_rival(self) -> None:
        """Not a magic number: it has to leave room for a near-miss and exclude a
        program that merely shares a few letters."""
        assert 0.0 < AMBIGUITY_GAP < 0.2

    def test_candidates_for_the_settings_window(self, resolver: AppIndexResolver) -> None:
        """The same ranking, without the exception — this feeds a list to click on."""
        assert [item.app_id for item in resolver.candidates("браузер")] == ["chrome", "firefox"]
        assert [item.app_id for item in resolver.candidates("редактор")] == ["vscode"]
        assert [item.app_id for item in resolver.candidates("хром")] == ["chrome"]
        assert resolver.candidates("   ") == []

    def test_app_looks_up_the_scanned_record(self, resolver: AppIndexResolver) -> None:
        found = resolver.app("chrome")
        assert found is not None
        assert found.target == SNAPSHOT_APPS[0].target
        assert resolver.app("photoshop") is None

    def test_the_better_source_wins_inside_the_resolver(self) -> None:
        """Two records of one program reach :meth:`build` when the cache predates a
        dedupe change; the label the user hears must still be the better one."""
        weak = IndexedApp(
            name="chrome",
            target=r"C:\c\chrome.exe",
            source=IndexSource.PATH,
            catalog_id="chrome",
        )
        resolver = AppIndexResolver.build(
            AppIndexSnapshot(apps=(weak, SNAPSHOT_APPS[0]), scanned_at=1.0),
            entries=CATALOG.apps,
        )
        found = resolver.app("chrome")
        assert found is not None
        assert found.source is IndexSource.START_MENU


class TestUserAliases:
    """The setting that ends the argument. Twice a regression, hence its own class."""

    @pytest.fixture(scope="class")
    def bound(self, snapshot: AppIndexSnapshot) -> AppIndexResolver:
        """«браузер» bound to Firefox, and a word of the user's own invention."""
        return AppIndexResolver.build(
            snapshot,
            entries=CATALOG.apps,
            user_aliases={"браузер": "firefox", "работа": "vscode"},
        )

    def test_a_bound_category_word_is_not_asked_about(self, bound: AppIndexResolver) -> None:
        """Regression: «браузер» reached :meth:`_resolve_kind` before the alias was
        consulted, so a user who had explicitly chosen Firefox was still asked
        «Google Chrome или Mozilla Firefox?»."""
        candidate = bound.resolve("браузер")
        assert candidate.app_id == "firefox"
        assert candidate.confidence == 1.0

    def test_a_bound_word_of_ones_own(self, bound: AppIndexResolver) -> None:
        assert bound.resolve("работа").app_id == "vscode"

    def test_a_bound_word_is_not_asked_about_either(self) -> None:
        """Regression: an alias pointing at one of two near-identical programs
        still raised, because the rival sweep ran after the match."""
        resolver = AppIndexResolver.build(
            AppIndexSnapshot(apps=TWIN_APPS, scanned_at=1_000.0),
            entries=TWIN_ENTRIES,
            user_aliases={"вайбр": "vaibor"},
        )
        assert resolver.resolve("вайбр").app_id == "vaibor"

    def test_candidates_follow_the_same_rule(self, bound: AppIndexResolver) -> None:
        """The settings list must not offer a choice the user already made."""
        assert [item.app_id for item in bound.candidates("браузер")] == ["firefox"]

    def test_unbound_words_are_unaffected(self, bound: AppIndexResolver) -> None:
        """Binding one word does not turn off the rest of the dictionary."""
        assert bound.resolve("хром").app_id == "chrome"
        assert bound.resolve("редактор").app_id == "vscode"


class TestAppIndexCache:
    """When to scan, and what survives a restart."""

    def test_the_first_run_scans_and_writes(self, tmp_path: Path) -> None:
        """No cache at all is the one case that makes the user wait."""
        calls: list[int] = []

        def scanner() -> list[IndexedApp]:
            calls.append(1)
            return list(SNAPSHOT_APPS)

        index = self._index(tmp_path, scanner=scanner)
        assert index.snapshot is None
        assert index.is_stale
        snapshot = index.ensure_ready()
        assert len(calls) == 1
        assert snapshot.apps == SNAPSHOT_APPS
        assert index.cache_path.exists()

    def test_a_fresh_cache_is_not_rescanned(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def scanner() -> list[IndexedApp]:
            calls.append(1)
            return list(SNAPSHOT_APPS)

        index = self._index(tmp_path, scanner=scanner)
        index.ensure_ready()
        index.ensure_ready()
        index.resolve("хром")
        assert len(calls) == 1

    def test_the_file_is_json_of_the_current_version(self, tmp_path: Path) -> None:
        """Readable by hand, and readable by a future version that has to migrate it."""
        index = self._index(tmp_path)
        index.ensure_ready()
        payload = json.loads(index.cache_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == INDEX_SCHEMA_VERSION
        assert len(payload["apps"]) == len(SNAPSHOT_APPS)

    def test_the_default_file_name_is_used(self, tmp_path: Path) -> None:
        assert self._index(tmp_path).cache_path.name == INDEX_FILE_NAME

    def test_a_second_index_reads_the_cache_instead_of_scanning(self, tmp_path: Path) -> None:
        """The point of the file: a cold start does not walk the registry."""
        first = self._index(tmp_path)
        first.ensure_ready()
        first.note_launch("chrome")

        def forbidden() -> list[IndexedApp]:
            raise AssertionError("the cache should have answered")

        second = self._index(tmp_path, scanner=forbidden)
        snapshot = second.ensure_ready()
        assert len(snapshot.apps) == len(SNAPSHOT_APPS)
        assert dict(snapshot.launches) == {"chrome": 1}

    def test_the_launch_counter_is_written_immediately(self, tmp_path: Path) -> None:
        """It breaks ties, so losing it to a crash costs the user the question again."""
        index = self._index(tmp_path)
        index.ensure_ready()
        assert index.note_launch("chrome") == 1
        assert index.note_launch("chrome") == 2
        payload = json.loads(index.cache_path.read_text(encoding="utf-8"))
        assert payload["launches"] == {"chrome": 2}

    def test_a_launch_rebuilds_the_resolver(self, tmp_path: Path) -> None:
        """The counter is an input to resolution, so the cached resolver is stale."""
        index = self._index(tmp_path)
        index.ensure_ready()
        assert index.resolver().launches == {}
        index.note_launch("chrome")
        assert index.resolver().launches == {"chrome": 1}

    def test_a_launch_without_an_id_is_ignored(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        index.ensure_ready()
        assert index.note_launch("") == 0

    def test_a_stale_cache_answers_and_refreshes_behind(self, tmp_path: Path) -> None:
        """The trade at the heart of the class: a twelve-hour-old index still knows
        where Chrome lives, and «Открываю Chrome» must not wait for a registry walk."""
        now = [1_000.0]
        index = self._index(tmp_path, clock=lambda: now[0])
        first = index.ensure_ready()
        now[0] += 500.0
        assert index.is_stale
        again = index.ensure_ready()
        assert again.scanned_at == first.scanned_at
        index.close()
        assert index.snapshot is not None
        assert index.snapshot.scanned_at == 1_500.0

    def test_a_background_refresh_keeps_the_counters(self, tmp_path: Path) -> None:
        """A rescan replaces what is installed, not what the user has been doing."""
        now = [1_000.0]
        index = self._index(tmp_path, clock=lambda: now[0])
        index.ensure_ready()
        index.note_launch("chrome")
        now[0] += 500.0
        index.ensure_ready()
        index.close()
        assert index.snapshot is not None
        assert dict(index.snapshot.launches) == {"chrome": 1}

    def test_wait_scans_on_this_thread(self, tmp_path: Path) -> None:
        """What «Обновить список» in the settings window calls."""
        now = [1_000.0]
        index = self._index(tmp_path, clock=lambda: now[0])
        index.ensure_ready()
        now[0] += 500.0
        assert index.ensure_ready(wait=True).scanned_at == 1_500.0
        assert not index.refreshing

    def test_a_broken_cache_file_is_rebuilt(
        self, tmp_path: Path, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """A truncated write must cost one slow scan, not a crash on start."""
        path = tmp_path / INDEX_FILE_NAME
        path.write_text("{ это не json", encoding="utf-8")
        index = self._index(tmp_path)
        assert len(index.ensure_ready().apps) == len(SNAPSHOT_APPS)
        assert "повреждён" in ayris_log.text

    def test_a_cache_of_another_version_is_rebuilt(
        self, tmp_path: Path, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / INDEX_FILE_NAME
        payload = AppIndexSnapshot(apps=SNAPSHOT_APPS, scanned_at=1_000.0).to_json()
        payload["schema_version"] = INDEX_SCHEMA_VERSION + 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        index = self._index(tmp_path)
        index.ensure_ready()
        assert "другой версии" in ayris_log.text

    def test_a_cache_that_is_not_an_object_is_rebuilt(self, tmp_path: Path) -> None:
        (tmp_path / INDEX_FILE_NAME).write_text("[1, 2, 3]", encoding="utf-8")
        assert len(self._index(tmp_path).ensure_ready().apps) == len(SNAPSHOT_APPS)

    def test_the_cache_directory_is_created(self, tmp_path: Path) -> None:
        """First start on a clean profile: the directory does not exist yet."""
        index = self._index(tmp_path, cache_path=tmp_path / "deep" / "er" / INDEX_FILE_NAME)
        index.ensure_ready()
        assert index.cache_path.exists()

    @staticmethod
    def _index(
        tmp_path: Path,
        *,
        cache_path: Path | None = None,
        scanner: Callable[[], list[IndexedApp]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> AppIndex:
        """An index wired to a temporary file and a fake clock.

        ``threshold`` and ``entries`` are passed explicitly so nothing here reads
        the user's settings or the profile directory.
        """
        return AppIndex(
            cache_path=cache_path or tmp_path / INDEX_FILE_NAME,
            ttl_s=100.0,
            scanner=scanner if scanner is not None else (lambda: list(SNAPSHOT_APPS)),
            entries=CATALOG.apps,
            threshold=0.8,
            clock=clock if clock is not None else (lambda: 1_000.0),
        )


class TestAppIndexRefresh:
    """The background thread, and the ways a scan can fail."""

    def test_only_one_scan_at_a_time(self, tmp_path: Path) -> None:
        """Five sources of I/O: two threads walking them at once is pure waste."""
        gate = threading.Event()

        def slow() -> list[IndexedApp]:
            assert gate.wait(timeout=5.0)
            return list(SNAPSHOT_APPS)

        index = AppIndex(
            cache_path=tmp_path / INDEX_FILE_NAME,
            scanner=slow,
            entries=CATALOG.apps,
            threshold=0.8,
        )
        try:
            assert index.refresh()
            assert index.refreshing
            assert not index.refresh()
        finally:
            gate.set()
            index.close()
        assert not index.refreshing
        assert index.snapshot is not None
        assert index.snapshot.apps == SNAPSHOT_APPS

    def test_a_failed_scan_keeps_the_previous_index(
        self, tmp_path: Path, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """Any of five sources can be broken on a given machine; a stale index is
        better than none, and the traceback belongs in the log."""
        broken = [False]

        def scanner() -> list[IndexedApp]:
            if broken[0]:
                raise RuntimeError("реестр закрылся")
            return list(SNAPSHOT_APPS)

        now = [1_000.0]
        index = AppIndex(
            cache_path=tmp_path / INDEX_FILE_NAME,
            ttl_s=100.0,
            scanner=scanner,
            entries=CATALOG.apps,
            threshold=0.8,
            clock=lambda: now[0],
        )
        index.ensure_ready()
        broken[0] = True
        now[0] += 500.0
        snapshot = index.refresh_now()
        assert snapshot.apps == SNAPSHOT_APPS
        assert "сканирование приложений не удалось" in ayris_log.text

    def test_a_failed_first_scan_leaves_an_empty_index(self, tmp_path: Path) -> None:
        """Nothing to fall back on, and still no exception out of ``ensure_ready``."""

        def scanner() -> list[IndexedApp]:
            raise OSError("нет доступа к реестру")

        index = AppIndex(
            cache_path=tmp_path / INDEX_FILE_NAME,
            scanner=scanner,
            entries=CATALOG.apps,
            threshold=0.8,
            clock=lambda: 1_000.0,
        )
        snapshot = index.ensure_ready()
        assert snapshot.apps == ()
        assert snapshot.scanned_at == 1_000.0

    def test_close_is_safe_without_a_thread(self, tmp_path: Path) -> None:
        """Teardown runs whether or not anything was ever scanned."""
        index = AppIndex(cache_path=tmp_path / INDEX_FILE_NAME, entries=(), threshold=0.8)
        index.close()

    def test_an_unwritable_cache_is_a_warning_not_a_failure(
        self, tmp_path: Path, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """A read-only profile costs a scan per start; it must not cost the feature."""
        blocked = tmp_path / "file"
        blocked.write_text("", encoding="utf-8")
        index = AppIndex(
            cache_path=blocked / "sub" / INDEX_FILE_NAME,
            scanner=lambda: list(SNAPSHOT_APPS),
            entries=CATALOG.apps,
            threshold=0.8,
            clock=lambda: 1_000.0,
        )
        assert index.ensure_ready().apps == SNAPSHOT_APPS
        assert "не сохранён" in ayris_log.text


class TestGlobalIndex:
    """One index per process, and the seam every other test in task 20 uses."""

    def test_the_instance_is_reused(self) -> None:
        set_app_index(None)
        try:
            assert get_app_index() is get_app_index()
        finally:
            set_app_index(None)

    def test_an_injected_index_wins(self, tmp_path: Path) -> None:
        mine = AppIndex(cache_path=tmp_path / INDEX_FILE_NAME, entries=(), threshold=0.8)
        set_app_index(mine)
        try:
            assert get_app_index() is mine
        finally:
            set_app_index(None)

    def test_dropping_it_builds_a_new_one(self, tmp_path: Path) -> None:
        mine = AppIndex(cache_path=tmp_path / INDEX_FILE_NAME, entries=(), threshold=0.8)
        set_app_index(mine)
        set_app_index(None)
        try:
            assert get_app_index() is not mine
        finally:
            set_app_index(None)


class TestScanner:
    """The Windows half. Guarded rather than skipped, so callers need no guard."""

    @pytest.mark.skipif(sys.platform == "win32", reason="проверяет поведение вне Windows")
    def test_nothing_is_found_elsewhere(self) -> None:
        """Every caller runs on both platforms; a raising scanner would push the
        platform check into each of them."""
        assert scan_installed() == []

    @pytest.mark.skipif(sys.platform != "win32", reason="нужен реестр Windows")
    def test_a_real_machine_has_programs(self) -> None:
        """Smoke test on the platform that matters: five sources, and at least one
        of them has to answer on any Windows install."""
        found = scan_installed(limit=50)
        assert found
        assert len(found) <= 50
        assert all(app.name and app.target for app in found)

    @pytest.mark.skipif(sys.platform != "win32", reason="нужен реестр Windows")
    def test_the_scan_is_deduplicated_and_linked(self) -> None:
        """What :func:`scan_installed` returns has already been through both."""
        found = scan_installed(limit=200)
        keys = [app.dedupe_key for app in found]
        assert len(keys) == len(set(keys))
