"""Задача 27: шаблоны автозаполнения и менеджеры паролей — без хранилищ.

Two seams make this testable on a machine with no vault installed.
:class:`~ayris.actions.system.secrets.base.CliRunner` is the only place a
subprocess is started, so ``keepassxc-cli`` and ``bw`` can be replaced by a
recorder that returns canned output and remembers exactly what it was called with;
and :func:`~ayris.actions.system.secrets.set_provider` installs a provider by name,
so a template can point at «keepass» without keepass existing.

What the recorder is really for is one assertion repeated in both provider groups:
**the master password is not in argv**. Both tools read it from standard input
precisely because an argument vector is public — Task Manager shows it, and so does
any script walking the process list. A test that only checked «unlock works» would
pass just as happily with the password on the command line, which is the bug worth
preventing.

The rest divides into three concerns.

*A field name decides whether a value is a secret.* Nothing about a template says
«this is a password»; the name does, and it has to work for names a person invents
(``мир_карта``, ``pass_госуслуги``) rather than only the ones listed here. Get this
wrong and the very first use logs a card number.

*How the value reaches the field is a security decision, not a convenience.* Typing
touches no clipboard at all. The clipboard route wipes rather than restores —
«restore what was there» and «leave nothing behind» are different promises, and a
card number needs the second — and a secret is wiped even when the setting says not
to.

*The names travel, the value does not.* The audit row, the log line and the result
all carry the template and the field; the value appears in none of them, checked
against the value itself rather than against a flag.

Groups:

* :class:`TestFieldSpecs` — known names, and invented ones that mean «secret».
* :class:`TestReferences` — ``secret:``/``keepass:``…, and why ``https:`` is not one.
* :class:`TestTemplates` — lookup, case, the errors that name what exists.
* :class:`TestResolving` — literals, provider references, defaulted keyring refs.
* :class:`TestKeePass` — the CLI contract: stdin, ``--quiet``, locked, missing.
* :class:`TestBitwarden` — the session key, the item cache, cards, locked, missing.
* :class:`TestProviderRegistry` — one instance per name, and what is usable here.
* :class:`TestKeyringProvider` — the write path, and the refs it writes to.
* :class:`TestAutoFillAction` — typing, pasting, wiping, an empty field.
* :class:`TestSecretsStayOut` — the audit row and the log file.
* :class:`TestLiveVaults` — hardware: a real ``keepassxc-cli``/``bw`` if present.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest

from ayris.actions.input.backend import (
    RecordingBackend,
    reset_input_backend,
    set_input_backend,
)
from ayris.actions.registry import ActionRegistry
from ayris.actions.system import autofill as autofill_module
from ayris.actions.system.autofill import (
    AutoFill,
    FillMode,
    UnknownField,
    UnknownTemplate,
    describe_template,
    field_spec,
    parse_reference,
    resolve_field,
    template_fields,
    template_names,
)
from ayris.actions.system.clipboard import (
    ClipboardKind,
    FakeClipboard,
    reset_clipboard,
    set_clipboard,
)
from ayris.actions.system.clipboard import (
    _suppressed as clipboard_suppressed,
)
from ayris.actions.system.secrets import (
    PROVIDERS,
    BitwardenProvider,
    CliResult,
    CliRunner,
    KeePassProvider,
    KeyringProvider,
    SecretEntry,
    SecretProvider,
    SecretProviderError,
    SecretProviderMissing,
    SecretValue,
    SessionCache,
    VaultLocked,
    available_providers,
    find_cli,
    get_provider,
    reset_providers,
    set_provider,
)
from ayris.actions.system.secrets.keepass import _parse_listing
from ayris.actions.system.secrets.keyring_store import autofill_ref, normalise_ref
from ayris.core.config import AutofillActionsConfig, InputActionsConfig
from ayris.core.database import Database, reset_database
from ayris.core.models import ExecutionResult
from ayris.core.repositories import Repositories
from ayris.core.secrets import is_valid_ref
from ayris.utils import logger as logger_module

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

CARD = "4111111111111111"
PASSWORD = "Пароль-от-Госуслуг-77"
MASTER = "мастер-пароль-хранилища"


# --------------------------------------------------------------------------- #
# Подставные хранилища
# --------------------------------------------------------------------------- #


class FakeProvider(SecretProvider):
    """A vault in a dict, with a lock that can be closed."""

    name = "keepass"
    title_ru = "KeePassXC"

    def __init__(self, values: dict[str, str] | None = None, *, locked: bool = False) -> None:
        self.values = values or {}
        self.locked = locked
        self.asked: list[tuple[str, str]] = []

    def available(self) -> bool:
        return True

    def unlocked(self) -> bool:
        return not self.locked

    def unlock(self, master_password: str) -> None:
        self.locked = master_password != MASTER
        if self.locked:
            raise self.locked_error()

    def lock(self) -> None:
        self.locked = True

    def list_entries(self) -> tuple[SecretEntry, ...]:
        return tuple(SecretEntry(path=key) for key in self.values)

    def get_field(self, entry: str, field: str = "password") -> SecretValue:
        if self.locked:
            raise self.locked_error()
        self.asked.append((entry, field))
        try:
            return SecretValue(self.values[f"{entry}#{field}"])
        except KeyError:
            raise SecretProviderError(
                f"no {entry}#{field}", user_message=f"Нет записи «{entry}»."
            ) from None


class FakeRunner(CliRunner):
    """A CLI that answers from a script and remembers how it was invoked.

    ``calls`` keeps ``argv``, ``stdin`` and the environment of every call, which is
    what the «master password not in argv» assertions read.
    """

    def __init__(self, *replies: CliResult) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float = 45.0,
        env: dict[str, str] | None = None,
    ) -> CliResult:
        self.calls.append(
            {"argv": tuple(argv), "stdin": stdin, "env": dict(env or {}), "timeout": timeout}
        )
        if not self.replies:
            return CliResult()
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    @property
    def argv_words(self) -> str:
        """Everything ever passed as an argument, as one string. For «not in»."""
        return " ".join(" ".join(call["argv"]) for call in self.calls)


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autofill_module, "_PASTE_SETTLE_S", 0.0)
    from ayris.actions.system import clipboard as clipboard_module

    monkeypatch.setattr(clipboard_module, "_PASTE_SETTLE_S", 0.0)


@pytest.fixture(autouse=True)
def _clean_seams() -> Iterator[None]:
    yield
    reset_providers()
    reset_clipboard()
    # Both calls: the reset drops the cached instance, but an installed override
    # outlives it and would follow this file into the next one.
    set_input_backend(None)
    reset_input_backend()
    clipboard_suppressed.clear()
    for name in (PASSWORD, CARD, MASTER):
        logger_module.forget_secret(name)


@pytest.fixture
def fake_clipboard() -> FakeClipboard:
    backend = FakeClipboard()
    set_clipboard(backend)
    return backend


@pytest.fixture
def keyboard() -> RecordingBackend:
    backend = RecordingBackend()
    set_input_backend(backend)
    return backend


def config(**overrides: object) -> AutofillActionsConfig:
    """An ``[actions.autofill]`` section with the defaults a test wants changed."""
    return AutofillActionsConfig(**overrides)  # type: ignore[arg-type]


def use_config(monkeypatch: pytest.MonkeyPatch, section: AutofillActionsConfig) -> None:
    """Make ``_config()`` — and therefore the action — see this section."""
    from ayris.core.config import get_settings

    base = get_settings()
    patched = base.model_copy(
        update={"actions": base.actions.model_copy(update={"autofill": section})}
    )
    monkeypatch.setattr(autofill_module, "get_settings", lambda: patched)


def _input_config(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> InputActionsConfig:
    """Override ``[actions.input]`` for whoever reads it, and return the section.

    The patch goes on :mod:`ayris.core.config` itself because ``keys._timings()``
    imports ``get_settings`` from there on every call instead of holding a
    module-level reference.
    """
    from ayris.core import config as config_module

    base = config_module.get_settings()
    section = base.actions.input.model_copy(update=overrides)
    patched = base.model_copy(
        update={"actions": base.actions.model_copy(update={"input": section})}
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: patched)
    return section


TEMPLATES: dict[str, dict[str, str]] = {
    "почта": {"email": "ayris@example.com"},
    "адрес": {"city": "Москва", "street": "Тверская", "house": "1"},
    "госуслуги": {
        "username": "ivanov",
        "password": "keepass:Госуслуги#password",
    },
    "карта": {"cardholder": "IVAN IVANOV", "number": "keyring:"},
}


# --------------------------------------------------------------------------- #
# Что за поле
# --------------------------------------------------------------------------- #


class TestFieldSpecs:
    """Whether a value is a secret is decided by the name of its field."""

    @pytest.mark.parametrize(
        ("name", "secret"),
        [
            ("email", False),
            ("city", False),
            ("cardholder", False),
            ("expiry_year", False),
            ("password", True),
            ("cvv", True),
            ("card", True),
            ("number", True),
            ("totp", True),
        ],
    )
    def test_известные_поля(self, name: str, secret: bool) -> None:
        assert field_spec(name).secret is secret
        assert field_spec(name).title_ru

    @pytest.mark.parametrize(
        "name",
        ["мир_карта", "pass_госуслуги", "мой ПАРОЛЬ", "cvc2", "код_из_смс", "номер счёта", "iban"],
    )
    def test_выдуманные_имена_с_приметами_считаются_секретом(self, name: str) -> None:
        """The default for an unknown name has to be «secret», not «text».

        Anything else means the first use of a field somebody invented puts its
        value in the log, and nobody finds out until they send the log in.
        """
        assert field_spec(name).secret

    @pytest.mark.parametrize("name", ["ник", "город", "любимый цвет", "дата рождения"])
    def test_безобидные_выдуманные_имена_не_секрет(self, name: str) -> None:
        assert not field_spec(name).secret

    def test_регистр_и_пробелы_не_мешают(self) -> None:
        assert field_spec("  Password  ").name == "password"
        assert field_spec("EMAIL").secret is False


class TestReferences:
    """A configured value is either a literal or a pointer at a vault."""

    @pytest.mark.parametrize(
        ("raw", "source", "entry", "field"),
        [
            ("secret:", "secret", "", ""),
            ("keyring:autofill.карта.number", "keyring", "autofill.карта.number", ""),
            ("keepass:Банки/Сбербанк#password", "keepass", "Банки/Сбербанк", "password"),
            ("bitwarden:Госуслуги", "bitwarden", "Госуслуги", ""),
            ("  keepass : Почта # username ", "keepass", "Почта", "username"),
        ],
    )
    def test_ссылки_разбираются(self, raw: str, source: str, entry: str, field: str) -> None:
        ref = parse_reference(raw)
        assert ref is not None
        assert (ref.source, ref.entry, ref.field) == (source, entry, field)

    @pytest.mark.parametrize(
        "raw",
        [
            "ayris@example.com",
            "https://gosuslugi.ru",
            "Москва, Тверская 1",
            "12:30",
            "",
            "C:\\Users\\Иван",
        ],
    )
    def test_остальное_берётся_буквально(self, raw: str) -> None:
        """The check is «one of four names, then a colon», not «has a colon».

        A URL, a time and a Windows path all contain colons and are all values
        somebody puts in a template.
        """
        assert parse_reference(raw) is None

    def test_secret_означает_настроенный_провайдер(self) -> None:
        assert parse_reference("secret:Почта")

        ref = parse_reference("secret:Почта")
        assert ref is not None
        assert ref.provider_name() == ""
        named = parse_reference("keepass:Почта")
        assert named is not None
        assert named.provider_name() == "keepass"


class TestTemplates:
    """Finding a template and its fields, by what a person actually says."""

    @pytest.fixture(autouse=True)
    def _templates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_config(monkeypatch, config(templates=TEMPLATES))

    def test_список_шаблонов_в_порядке_настроек(self) -> None:
        assert template_names() == ("почта", "адрес", "госуслуги", "карта")

    def test_шаблон_находится_без_учёта_регистра(self) -> None:
        assert [spec.name for spec in template_fields("Почта")] == ["email"]
        assert [spec.name for spec in template_fields("АДРЕС")] == ["city", "street", "house"]

    def test_неизвестный_шаблон_перечисляет_известные(self) -> None:
        with pytest.raises(UnknownTemplate) as info:
            template_fields("паспорт")
        assert "почта" in info.value.user_message
        assert "«паспорт»" in info.value.user_message

    def test_описание_не_показывает_секрет(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The settings tab shows what a field is, never what it holds."""
        use_config(
            monkeypatch,
            config(templates={"карта": {"number": CARD, "cardholder": "IVAN IVANOV"}}),
        )
        summaries = {row.name: row for row in describe_template("карта")}
        assert summaries["number"].secret
        assert summaries["number"].preview == "[скрыто]"
        assert CARD not in str(summaries)
        assert summaries["cardholder"].preview == "IVAN IVANOV"

    def test_описание_ссылки_называет_источник(self) -> None:
        summaries = {row.name: row for row in describe_template("госуслуги")}
        assert summaries["password"].preview == "из keepass"
        assert summaries["username"].preview == "ivanov"

    def test_длинное_значение_укорачивается(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_config(monkeypatch, config(templates={"о себе": {"notes": "я" * 300}}))
        preview = describe_template("о себе")[0].preview
        assert len(preview) <= autofill_module._PREVIEW_CHARS + 1
        assert preview.endswith("…")


class TestResolving:
    """Getting to a value: from the file, or from a vault."""

    @pytest.fixture(autouse=True)
    def _templates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_config(monkeypatch, config(templates=TEMPLATES))

    def test_обычное_значение_из_настроек(self) -> None:
        resolved = resolve_field("почта", "email")
        assert resolved.value == "ayris@example.com"
        assert resolved.source == "config"
        assert not resolved.secret

    def test_единственное_поле_можно_не_называть(self) -> None:
        """«Вставь мою почту» is a whole template with one thing in it."""
        assert resolve_field("почта").value == "ayris@example.com"

    def test_у_многополевого_шаблона_поле_обязательно(self) -> None:
        with pytest.raises(UnknownField) as info:
            resolve_field("адрес")
        assert "city" in info.value.user_message

    def test_неизвестное_поле_перечисляет_существующие(self) -> None:
        with pytest.raises(UnknownField) as info:
            resolve_field("адрес", "подъезд")
        assert "street" in info.value.user_message

    def test_поле_находится_без_учёта_регистра(self) -> None:
        assert resolve_field("адрес", "CITY").value == "Москва"

    def test_ссылка_идёт_в_провайдер(self) -> None:
        provider = FakeProvider({"Госуслуги#password": PASSWORD})
        set_provider("keepass", provider)
        resolved = resolve_field("госуслуги", "password")
        assert resolved.value == PASSWORD
        assert resolved.source == "keepass"
        assert resolved.secret
        assert provider.asked == [("Госуслуги", "password")]

    def test_пустая_запись_в_keyring_подставляется_сама(self) -> None:
        """``keyring:`` with nothing after it means «where Ayris put it».

        That is the ref :func:`save_secret` writes to, so the settings tab can
        store a card number without the user inventing a name for it.
        """
        provider = FakeProvider({f"{autofill_ref('карта', 'number')}#number": CARD})
        provider.name = "keyring"
        set_provider("keyring", provider)
        resolved = resolve_field("карта", "number")
        assert resolved.value == CARD
        assert provider.asked == [(autofill_ref("карта", "number"), "number")]

    def test_пустая_запись_в_хранилище_это_ошибка(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A vault has no «where Ayris put it» — the entry has to be named."""
        use_config(monkeypatch, config(templates={"карта": {"number": "keepass:"}}))
        set_provider("keepass", FakeProvider())
        with pytest.raises(UnknownField) as info:
            resolve_field("карта", "number")
        assert "KeePassXC" in info.value.user_message

    def test_закрытое_хранилище_объясняет_себя(self) -> None:
        set_provider("keepass", FakeProvider({"Госуслуги#password": PASSWORD}, locked=True))
        with pytest.raises(VaultLocked) as info:
            resolve_field("госуслуги", "password")
        assert "мастер-пароль" in info.value.user_message.lower()

    def test_значение_прячется_из_логов_пока_живёт(self) -> None:
        set_provider("keepass", FakeProvider({"Госуслуги#password": PASSWORD}))
        resolved = resolve_field("госуслуги", "password")
        with resolved:
            assert logger_module.redact(PASSWORD) == logger_module.SECRET_PLACEHOLDER
        assert logger_module.redact(PASSWORD) == PASSWORD

    def test_значение_не_попадает_в_repr(self) -> None:
        set_provider("keepass", FakeProvider({"Госуслуги#password": PASSWORD}))
        resolved = resolve_field("госуслуги", "password")
        assert PASSWORD not in repr(resolved)
        assert PASSWORD not in f"{resolved}"


# --------------------------------------------------------------------------- #
# KeePassXC
# --------------------------------------------------------------------------- #


class TestKeePass:
    """The ``keepassxc-cli`` contract, checked against a recorded run."""

    @pytest.fixture
    def kdbx(self, tmp_path: Path) -> Path:
        path = tmp_path / "пароли.kdbx"
        path.write_bytes(b"kdbx")
        return path

    @pytest.fixture
    def cli(self, tmp_path: Path) -> Path:
        path = tmp_path / "keepassxc-cli"
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        return path

    def provider(
        self, cli: Path, kdbx: Path, runner: FakeRunner, **overrides: object
    ) -> KeePassProvider:
        return KeePassProvider(
            runner=runner,
            settings=config(
                keepass_cli=str(cli), keepass_database=str(kdbx), **overrides  # type: ignore[arg-type]
            ),
        )

    def test_мастер_пароль_идёт_в_stdin_а_не_в_аргументы(self, cli: Path, kdbx: Path) -> None:
        """The one assertion this whole class exists for.

        An argument vector is readable by every process on the machine; standard
        input is not. Both tools prompt for exactly that reason.
        """
        runner = FakeRunner(CliResult(stdout="Госуслуги\nПочта\n"))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        call = runner.calls[0]
        assert MASTER not in runner.argv_words
        assert call["stdin"] == f"{MASTER}\n"
        assert call["argv"][1:] == ("ls", "--quiet", "--flatten", str(kdbx))
        assert provider.unlocked()

    def test_перевод_строки_обязателен(self, cli: Path, kdbx: Path) -> None:
        """``keepassxc-cli`` reads a line; without the newline it waits out the timeout."""
        runner = FakeRunner(CliResult(stdout="Почта\n"))
        self.provider(cli, kdbx, runner).unlock(MASTER)
        assert runner.calls[0]["stdin"].endswith("\n")

    def test_поле_читается_показанным(self, cli: Path, kdbx: Path) -> None:
        runner = FakeRunner(CliResult(stdout="Госуслуги\n"), CliResult(stdout=f"{PASSWORD}\n"))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        with provider.get_field("Госуслуги", "password") as secret:
            assert secret.value == PASSWORD
            assert str(secret) == "[скрыто]"
        argv = runner.calls[-1]["argv"]
        assert argv[1:] == (
            "show",
            "--quiet",
            "--show-protected",
            "-a",
            "Password",
            str(kdbx),
            "Госуслуги",
        )

    def test_пробелы_в_пароле_сохраняются(self, cli: Path, kdbx: Path) -> None:
        """Only the trailing newline is stripped.

        A password may legitimately begin or end with a space, and ``.strip()``
        would hand over a different password that looks the same in a log.
        """
        spaced = "  пароль с пробелами  "
        runner = FakeRunner(CliResult(stdout="Почта\n"), CliResult(stdout=f"{spaced}\r\n"))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        secret = provider.get_field("Почта", "password")
        assert secret.value == spaced
        secret.forget()

    def test_имя_поля_переводится_в_атрибут(self, cli: Path, kdbx: Path) -> None:
        runner = FakeRunner(CliResult(stdout="Почта\n"), CliResult(stdout="ivanov\n"))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        provider.get_field("Почта", "login").forget()
        assert "UserName" in runner.calls[-1]["argv"]

    def test_закрытое_хранилище_не_читается(self, cli: Path, kdbx: Path) -> None:
        runner = FakeRunner(CliResult(stdout=f"{PASSWORD}\n"))
        provider = self.provider(cli, kdbx, runner)
        with pytest.raises(VaultLocked):
            provider.get_field("Госуслуги", "password")
        assert runner.calls == []  # ничего не запускалось, пароля не было

    def test_неверный_мастер_пароль(self, cli: Path, kdbx: Path) -> None:
        runner = FakeRunner(CliResult(code=1, stderr="Invalid credentials were provided"))
        provider = self.provider(cli, kdbx, runner)
        with pytest.raises(VaultLocked) as info:
            provider.unlock("не тот")
        assert info.value.user_message == "Мастер-пароль KeePassXC не подошёл."
        assert not provider.unlocked()

    def test_база_не_найдена(self, cli: Path, kdbx: Path) -> None:
        runner = FakeRunner(CliResult(code=1, stderr="No such file or directory"))
        provider = self.provider(cli, kdbx, runner)
        with pytest.raises(SecretProviderMissing) as info:
            provider.unlock(MASTER)
        assert "путь в настройках" in info.value.user_message

    def test_нет_cli_или_нет_базы(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two different «not set up» cases, one Russian sentence each time."""
        monkeypatch.setattr("shutil.which", lambda _name: None)
        no_cli = KeePassProvider(
            runner=FakeRunner(), settings=config(keepass_database=str(tmp_path / "п.kdbx"))
        )
        assert not no_cli.available()
        with pytest.raises(SecretProviderMissing):
            no_cli.unlock(MASTER)
        no_db = KeePassProvider(runner=FakeRunner(), settings=config(keepass_cli="keepassxc-cli"))
        assert not no_db.available()

    def test_ошибка_не_цитирует_вывод_программы(self, cli: Path, kdbx: Path) -> None:
        """stdout is where the secret is; only names may reach the message."""
        runner = FakeRunner(CliResult(stdout="Почта\n"), CliResult(code=2, stdout=PASSWORD))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        with pytest.raises(SecretProviderError) as info:
            provider.get_field("Госуслуги", "password")
        assert PASSWORD not in info.value.user_message

    def test_список_читается_один_раз(self, cli: Path, kdbx: Path) -> None:
        """Each call re-derives the key, which is slow by design."""
        runner = FakeRunner(CliResult(stdout="Госуслуги\nПочта\n"))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        first = provider.list_entries()
        second = provider.list_entries()
        assert first == second
        assert len(runner.calls) == 1

    def test_блокировка_забывает_всё(self, cli: Path, kdbx: Path) -> None:
        runner = FakeRunner(CliResult(stdout="Почта\n"))
        provider = self.provider(cli, kdbx, runner)
        provider.unlock(MASTER)
        provider.lock()
        assert not provider.unlocked()
        assert logger_module.redact(MASTER) == MASTER

    @pytest.mark.parametrize(
        ("stdout", "paths"),
        [
            ("Почта\nБанки/Сбербанк\n", ["Почта", "Банки/Сбербанк"]),
            ("Банки/\nБанки/Сбербанк\n", ["Банки/Сбербанк"]),
            ("[empty]\n", []),
            ("", []),
        ],
    )
    def test_разбор_списка(self, stdout: str, paths: list[str]) -> None:
        assert [entry.path for entry in _parse_listing(stdout)] == paths

    def test_подпись_записи_читаема(self) -> None:
        entry = SecretEntry(path="Банки/Сбербанк", title="Сбербанк", username="ivanov")
        assert entry.label == "Сбербанк (ivanov)"
        assert SecretEntry(path="Почта").label == "Почта"


# --------------------------------------------------------------------------- #
# Bitwarden
# --------------------------------------------------------------------------- #


VAULT_ITEMS = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "Госуслуги",
        "type": 1,
        "login": {
            "username": "ivanov",
            "password": PASSWORD,
            "totp": "otpauth://totp/x",
            "uris": [{"uri": "https://gosuslugi.ru"}],
        },
        "fields": [{"name": "снилс", "value": "123-456-789 00"}],
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "name": "Visa",
        "type": 3,
        "card": {
            "number": CARD,
            "cardholderName": "IVAN IVANOV",
            "code": "737",
            "expMonth": "12",
            "expYear": "2030",
        },
    },
]


class TestBitwarden:
    """``bw``: a session key instead of a password, and one listing per unlock."""

    @pytest.fixture
    def cli(self, tmp_path: Path) -> Path:
        path = tmp_path / "bw"
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        return path

    def provider(self, cli: Path, runner: FakeRunner, **overrides: object) -> BitwardenProvider:
        return BitwardenProvider(
            runner=runner,
            settings=config(bitwarden_cli=str(cli), **overrides),  # type: ignore[arg-type]
        )

    def unlocked(self, cli: Path, runner: FakeRunner) -> BitwardenProvider:
        provider = self.provider(cli, runner)
        provider.unlock(MASTER)
        return provider

    def test_мастер_пароль_идёт_в_окружение_а_не_в_аргументы(self, cli: Path) -> None:
        """``bw`` takes it from ``BW_PASSWORD``; a child's environment is private."""
        runner = FakeRunner(CliResult(stdout="СЕССИЯ==\n"))
        provider = self.provider(cli, runner)
        provider.unlock(MASTER)
        call = runner.calls[0]
        assert MASTER not in runner.argv_words
        assert call["env"]["BW_PASSWORD"] == MASTER
        assert call["argv"][1:] == (
            "--nointeraction",
            "unlock",
            "--raw",
            "--passwordenv",
            "BW_PASSWORD",
        )
        assert provider.unlocked()

    def test_ключ_сессии_идёт_в_окружение_и_не_в_аргументы(self, cli: Path) -> None:
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        provider = self.unlocked(cli, runner)
        provider.list_entries()
        listing = runner.calls[-1]
        assert listing["argv"][1:] == ("--nointeraction", "list", "items")
        assert listing["env"]["BW_SESSION"] == "СЕССИЯ=="
        assert "BW_PASSWORD" not in listing["env"]
        assert "СЕССИЯ==" not in runner.argv_words

    def test_без_интерактива_всегда(self, cli: Path) -> None:
        """Otherwise a locked vault makes ``bw`` wait for a terminal that is not there."""
        runner = FakeRunner(CliResult(stdout="СЕССИЯ==\n"))
        self.unlocked(cli, runner)
        assert all("--nointeraction" in call["argv"] for call in runner.calls)

    def test_запись_находится_по_имени(self, cli: Path) -> None:
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        provider = self.unlocked(cli, runner)
        with provider.get_field("госуслуги", "password") as secret:
            assert secret.value == PASSWORD
        with provider.get_field("Госуслуги", "username") as secret:
            assert secret.value == "ivanov"

    def test_поля_карты(self, cli: Path) -> None:
        """A card number is a secret exactly like a password."""
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        provider = self.unlocked(cli, runner)
        for field, expected in (
            ("number", CARD),
            ("cvv", "737"),
            ("cardholder", "IVAN IVANOV"),
            ("expiry_year", "2030"),
        ):
            with provider.get_field("Visa", field) as secret:
                assert secret.value == expected

    def test_своё_поле_записи(self, cli: Path) -> None:
        """Where a bank puts what a form needs and Bitwarden has no name for."""
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        provider = self.unlocked(cli, runner)
        with provider.get_field("Госуслуги", "СНИЛС") as secret:
            assert secret.value == "123-456-789 00"

    def test_хранилище_читается_один_раз(self, cli: Path) -> None:
        """``bw`` is a Node program: a third of a second before it does anything."""
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        provider = self.unlocked(cli, runner)
        for field in ("password", "username", "totp"):
            provider.get_field("Госуслуги", field).forget()
        assert sum("list" in call["argv"] for call in runner.calls) == 1

    def test_список_без_значений(self, cli: Path) -> None:
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        entries = self.unlocked(cli, runner).list_entries()
        assert [entry.title for entry in entries] == ["Госуслуги", "Visa"]
        assert entries[0].username == "ivanov"
        assert entries[0].url == "https://gosuslugi.ru"
        assert "totp" in entries[0].fields
        assert "number" in entries[1].fields
        assert PASSWORD not in str(entries)
        assert CARD not in str(entries)

    def test_нет_записи_или_нет_поля(self, cli: Path) -> None:
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout=json.dumps(VAULT_ITEMS))
        )
        provider = self.unlocked(cli, runner)
        with pytest.raises(SecretProviderError) as no_item:
            provider.get_field("Сбербанк", "password")
        assert "«Сбербанк»" in no_item.value.user_message
        with pytest.raises(SecretProviderError) as no_field:
            provider.get_field("Visa", "totp")
        assert "«totp»" in no_field.value.user_message

    def test_закрытое_хранилище_сбрасывает_сессию(self, cli: Path) -> None:
        runner = FakeRunner(
            CliResult(stdout="СЕССИЯ==\n"), CliResult(code=1, stderr="Vault is locked.")
        )
        provider = self.unlocked(cli, runner)
        with pytest.raises(VaultLocked) as info:
            provider.list_entries()
        assert "Bitwarden" in info.value.user_message
        assert not provider.unlocked()

    def test_пустой_ключ_сессии_это_ошибка(self, cli: Path) -> None:
        runner = FakeRunner(CliResult(stdout="  \n"))
        with pytest.raises(SecretProviderError):
            self.provider(cli, runner).unlock(MASTER)

    def test_непонятный_ответ_это_ошибка(self, cli: Path) -> None:
        runner = FakeRunner(CliResult(stdout="СЕССИЯ==\n"), CliResult(stdout="не json"))
        provider = self.unlocked(cli, runner)
        with pytest.raises(SecretProviderError) as info:
            provider.list_entries()
        assert "непонятный" in info.value.user_message

    def test_нет_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: None)
        provider = BitwardenProvider(runner=FakeRunner(), settings=config())
        assert not provider.available()
        with pytest.raises(SecretProviderMissing):
            provider.unlock(MASTER)

    def test_блокировка_говорит_bw_забыть_ключ(self, cli: Path) -> None:
        runner = FakeRunner(CliResult(stdout="СЕССИЯ==\n"))
        provider = self.unlocked(cli, runner)
        provider.lock()
        assert not provider.unlocked()
        assert runner.calls[-1]["argv"][1:] == ("--nointeraction", "lock")
        assert runner.calls[-1]["env"]["BW_SESSION"] == "СЕССИЯ=="


# --------------------------------------------------------------------------- #
# Общая обвязка провайдеров
# --------------------------------------------------------------------------- #


class TestProviderRegistry:
    """One instance per name, because an unlocked vault is worth keeping."""

    def test_провайдер_переживает_вызов(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rebuilding it per call would ask for the master password every time."""
        use_config(monkeypatch, config())
        assert get_provider("keepass") is get_provider("keepass")

    def test_неизвестный_провайдер_это_ошибка(self) -> None:
        with pytest.raises(SecretProviderError) as info:
            get_provider("1password")
        assert "1password" in info.value.user_message

    def test_сброс_закрывает_хранилища(self) -> None:
        provider = FakeProvider({"а#password": PASSWORD})
        set_provider("keepass", provider)
        reset_providers()
        assert provider.locked

    def test_сброс_не_падает_на_упрямом_провайдере(self) -> None:
        """Locking is cleanup; a failure there must not propagate."""

        class Stubborn(FakeProvider):
            def lock(self) -> None:
                raise RuntimeError("не закроюсь")

        set_provider("keepass", Stubborn())
        reset_providers()

    def test_доступные_провайдеры_не_открывают_хранилищ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: None)
        set_provider("keyring", FakeProvider())
        names = available_providers()
        assert "keepass" not in names
        assert "bitwarden" not in names

    def test_все_провайдеры_зарегистрированы(self) -> None:
        assert set(PROVIDERS) == {"keyring", "keepass", "bitwarden"}

    def test_поиск_cli_предпочитает_настроенный_путь(self, tmp_path: Path) -> None:
        """A portable KeePassXC in a folder is the normal case, and never in PATH."""
        portable = tmp_path / "keepassxc-cli.exe"
        portable.write_text("", encoding="utf-8")
        assert find_cli(str(portable), "keepassxc-cli") == str(portable)
        assert find_cli("", "заведомо-несуществующая-программа") == ""

    def test_сессия_истекает(self) -> None:
        cache = SessionCache(0.0)
        cache.store(MASTER)
        assert not cache
        assert logger_module.redact(MASTER) == MASTER

    def test_сессия_держится_и_сбрасывается(self) -> None:
        cache = SessionCache(600.0)
        cache.store(MASTER)
        assert cache.get() == MASTER
        assert logger_module.redact(MASTER) == logger_module.SECRET_PLACEHOLDER
        cache.drop()
        assert cache.get() == ""
        assert logger_module.redact(MASTER) == MASTER


class TestKeyringProvider:
    """The Credential Manager, and the only write path Ayris has."""

    def test_ref_складывается_из_шаблона_и_поля(self) -> None:
        """Latin, short and readable — the three things the store demands.

        ``is_valid_ref`` accepts ``^[a-z][a-z0-9_.-]{0,31}$`` and nothing else, so a
        Russian template name has to be transliterated rather than escaped: before
        this, saving a card number failed outright.
        """
        assert autofill_ref("карта", "number") == "autofill.karta.number"
        assert autofill_ref("Госуслуги", "пароль") == "autofill.gosuslugi.parol"
        assert autofill_ref("моя. карта", "номер поля") == "autofill.moya_karta.nomer_polya"
        assert all(is_valid_ref(ref) for ref in (autofill_ref("карта", "number"),))

    def test_длинное_имя_укорачивается_и_остаётся_уникальным(self) -> None:
        """Too long for 32 characters: keep the tail, add a digest of the original."""
        long_one = autofill_ref("моя очень длинная карточка для покупок", "number")
        other = autofill_ref("моя очень длинная карточка для отпуска", "number")
        assert is_valid_ref(long_one)
        assert long_one != other
        assert long_one == autofill_ref("моя очень длинная карточка для покупок", "number")
        assert long_one.startswith("autofill.number-")

    def test_имя_без_латиницы_и_кириллицы_тоже_даёт_ref(self) -> None:
        """An alphabet the table does not know still has to end up somewhere."""
        ref = autofill_ref("卡", "号")
        assert is_valid_ref(ref)
        assert ref != autofill_ref("卡", "码")

    def test_ref_написанный_руками_приводится_к_той_же_записи(self) -> None:
        """``keyring:autofill.карта.number`` in a template points where it reads."""
        assert normalise_ref("autofill.карта.number") == autofill_ref("карта", "number")
        assert normalise_ref("stt.api_key") == "stt.api_key"

    def test_запись_чтение_и_удаление(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Against a store in a dict: the real one needs the Windows keyring."""
        stored: dict[str, str] = {}

        class FakeStore:
            def is_available(self) -> bool:
                return True

            def save(self, ref: str, value: str) -> None:
                stored[ref] = value

            def get(self, ref: str) -> str | None:
                return stored.get(ref)

            def delete(self, ref: str) -> bool:
                return stored.pop(ref, None) is not None

            def stored_refs(self) -> tuple[str, ...]:
                return tuple(stored)

        provider = KeyringProvider(store=FakeStore())  # type: ignore[arg-type]
        ref = provider.save_field("карта", "number", CARD)
        assert ref == autofill_ref("карта", "number")
        assert provider.unlocked()  # логин пользователя и есть разблокировка
        with provider.get_field("карта", "number") as secret:
            assert secret.value == CARD
        assert [entry.path for entry in provider.list_entries()] == [ref]
        assert provider.delete_field("карта", "number")
        assert not provider.delete_field("карта", "number")
        with pytest.raises(SecretProviderError) as info:
            provider.get_field("карта", "number")
        assert CARD not in info.value.user_message

    def test_недоступный_keyring_объясняет_себя(self) -> None:
        class Broken:
            def is_available(self) -> bool:
                return False

        provider = KeyringProvider(store=Broken())  # type: ignore[arg-type]
        assert not provider.available()
        with pytest.raises(SecretProviderMissing) as info:
            provider.list_entries()
        assert "Диспетчер учётных данных" in info.value.user_message


# --------------------------------------------------------------------------- #
# Действие
# --------------------------------------------------------------------------- #


class TestAutoFillAction:
    """Filling the field under the cursor, and what is left behind afterwards."""

    @pytest.fixture(autouse=True)
    def _templates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_config(monkeypatch, config(templates=TEMPLATES))

    @pytest.fixture
    def vault(self) -> FakeProvider:
        provider = FakeProvider({"Госуслуги#password": PASSWORD})
        set_provider("keepass", provider)
        return provider

    def test_печатает_и_не_трогает_буфер(
        self, keyboard: RecordingBackend, fake_clipboard: FakeClipboard
    ) -> None:
        """The default route. A clipboard that was never touched cannot leak."""
        result = AutoFill().run(AutoFill.Params(template="почта", field="email"))
        assert result.ok
        assert keyboard.typed == "ayris@example.com"
        assert fake_clipboard.writes == []
        assert fake_clipboard.clears == 0
        assert result.data["mode"] == "type"

    def test_ответ_называет_поле_а_не_значение(
        self, keyboard: RecordingBackend, fake_clipboard: FakeClipboard, vault: FakeProvider
    ) -> None:
        result = AutoFill().run(AutoFill.Params(template="госуслуги", field="password"))
        assert result.ok
        assert result.message_ru == "Подставил «Пароль» из шаблона «госуслуги»."
        assert PASSWORD not in result.message_ru
        assert PASSWORD not in str(result.data)
        assert result.value == len(PASSWORD)
        assert result.data == {
            "template": "госуслуги",
            "field": "password",
            "source": "keepass",
            "secret": True,
            "mode": "type",
            "clipboard_cleared": False,
            "length": len(PASSWORD),
        }

    def test_через_буфер_с_очисткой(
        self, keyboard: RecordingBackend, fake_clipboard: FakeClipboard, vault: FakeProvider
    ) -> None:
        result = AutoFill().run(
            AutoFill.Params(template="госуслуги", field="password", mode=FillMode.CLIPBOARD)
        )
        assert result.ok
        assert fake_clipboard.writes == [PASSWORD]
        assert keyboard.keys == ("+ctrl", "+v", "-v", "-ctrl")
        assert fake_clipboard.clears == 1
        assert fake_clipboard.read().kind is ClipboardKind.EMPTY
        assert result.data["clipboard_cleared"] is True

    def test_вставка_не_попадает_в_историю(
        self, keyboard: RecordingBackend, fake_clipboard: FakeClipboard, vault: FakeProvider
    ) -> None:
        """The monitor would otherwise see the password as an ordinary copy."""
        AutoFill().run(
            AutoFill.Params(template="госуслуги", field="password", mode=FillMode.CLIPBOARD)
        )
        from ayris.actions.system.clipboard import _claim_suppressed

        assert _claim_suppressed(PASSWORD)

    def test_секрет_очищается_даже_против_настройки(
        self, keyboard: RecordingBackend, fake_clipboard: FakeClipboard, vault: FakeProvider
    ) -> None:
        """The setting decides about a street address, not about a password."""
        AutoFill().run(
            AutoFill.Params(
                template="госуслуги",
                field="password",
                mode=FillMode.CLIPBOARD,
                clear_clipboard=False,
            )
        )
        assert fake_clipboard.clears == 1

    def test_обычное_значение_можно_оставить_в_буфере(
        self, keyboard: RecordingBackend, fake_clipboard: FakeClipboard
    ) -> None:
        result = AutoFill().run(
            AutoFill.Params(
                template="адрес", field="city", mode=FillMode.CLIPBOARD, clear_clipboard=False
            )
        )
        assert result.ok
        assert fake_clipboard.clears == 0
        assert fake_clipboard.read().text == "Москва"

    def test_режим_auto_берётся_из_настроек(
        self,
        monkeypatch: pytest.MonkeyPatch,
        keyboard: RecordingBackend,
        fake_clipboard: FakeClipboard,
    ) -> None:
        use_config(monkeypatch, config(templates=TEMPLATES, paste_mode="clipboard"))
        result = AutoFill().run(AutoFill.Params(template="почта"))
        assert result.data["mode"] == "clipboard"
        assert fake_clipboard.writes == ["ayris@example.com"]

    def test_буфер_очищается_даже_если_вставка_упала(
        self, fake_clipboard: FakeClipboard, vault: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed Ctrl+V must not leave the password on the clipboard."""
        from ayris.actions.system import clipboard as clipboard_module

        def refuse() -> None:
            raise RuntimeError("ввод заблокирован")

        monkeypatch.setattr(clipboard_module, "paste_shortcut", refuse)
        with pytest.raises(RuntimeError):
            AutoFill().run(
                AutoFill.Params(template="госуслуги", field="password", mode=FillMode.CLIPBOARD)
            )
        assert fake_clipboard.clears == 1

    def test_пустое_значение_это_отказ(
        self, monkeypatch: pytest.MonkeyPatch, keyboard: RecordingBackend
    ) -> None:
        use_config(monkeypatch, config(templates={"почта": {"email": ""}}))
        result = AutoFill().run(AutoFill.Params(template="почта"))
        assert not result.ok
        assert result.message_ru == "Поле «Почта» пустое, подставлять нечего."
        assert keyboard.typed == ""

    def test_неизвестный_шаблон_доходит_до_пользователя(self, keyboard: RecordingBackend) -> None:
        with pytest.raises(UnknownTemplate):
            AutoFill().run(AutoFill.Params(template="паспорт"))

    def test_действие_в_реестре(self) -> None:
        registry = ActionRegistry()
        registry.discover()
        assert isinstance(registry.get("AutoFill"), AutoFill)
        assert AutoFill.meta.category.value == "input"

    def test_режимы_подписаны_по_русски(self) -> None:
        assert {mode.title_ru for mode in FillMode} == {
            "Как в настройках",
            "Напечатать",
            "Через буфер обмена",
        }


# --------------------------------------------------------------------------- #
# Секреты не остаются нигде
# --------------------------------------------------------------------------- #


class LoggingBackend(RecordingBackend):
    """A backend that logs what it types — the mistake the filter exists for.

    Not a straw man: a debug line with the text being typed is the obvious thing
    to add while chasing a layout problem, and it is invisible until someone
    sends in their log.
    """

    def type_text(self, text: str) -> None:
        logging.getLogger("ayris.test.backend").debug("печатаю: %s", text)
        super().type_text(text)


class TestSecretsStayOut:
    """The two places a value would outlive the moment it was needed."""

    @pytest.fixture(autouse=True)
    def _templates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_config(monkeypatch, config(templates=TEMPLATES))
        set_provider("keepass", FakeProvider({"Госуслуги#password": PASSWORD}))

    def test_в_аудите_только_имена(self, tmp_path: Path, keyboard: RecordingBackend) -> None:
        """The audit row is the log of what Ayris did, and it is enough here.

        A template name and a field name say what happened; the value would only
        add a password to a table that is kept on purpose.
        """
        database = Database.open(tmp_path / "ayris.db")
        try:
            repos = Repositories(database)
            registry = ActionRegistry(audit=repos.audit, audit_enabled=lambda: True)
            registry.add(AutoFill)
            try:
                registry.execute("AutoFill", {"template": "госуслуги", "field": "password"})
            finally:
                registry.shutdown()
            entry = repos.audit.recent(1)[0]
            assert entry.result is ExecutionResult.OK
            assert entry.params == {
                "template": "госуслуги",
                "field": "password",
                "mode": "auto",
                "clear_clipboard": None,
            }
            assert PASSWORD not in str(entry.params)
        finally:
            database.close()
            reset_database()
        raw = (tmp_path / "ayris.db").read_bytes()
        assert PASSWORD.encode("utf-8") not in raw

    def test_в_логе_ничего_нет_даже_на_debug(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The value is guarded for exactly as long as it exists.

        Which is what makes the debug line in :class:`LoggingBackend` harmless:
        the filter sits on the handler, so it catches a line nobody reviewed.

        ``char_delay_ms=0`` because that is the case worth checking: with a delay
        ``TypeText`` hands the backend one character at a time, and a line holding
        a single letter is nothing any filter could recognise. Zero delay is the
        route where the whole value reaches a logging call in one piece.
        """
        input_config = _input_config(monkeypatch, char_delay_ms=0)
        assert input_config.char_delay_ms == 0
        set_input_backend(LoggingBackend())
        log_dir = tmp_path / "logs"
        logger_module.setup_logging("DEBUG", console=False, log_dir=log_dir)
        try:
            result = AutoFill().run(AutoFill.Params(template="госуслуги", field="password"))
            assert result.ok
        finally:
            logger_module.shutdown_logging()
        written = "\n".join(path.read_text("utf-8") for path in log_dir.glob("*.log"))
        assert written
        assert PASSWORD not in written
        assert logger_module.SECRET_PLACEHOLDER in written
        assert "госуслуги" in written  # имена шаблона и поля логировать можно

    def test_после_вставки_значение_перестаёт_прятаться(self, keyboard: RecordingBackend) -> None:
        """The guard is not a leak of its own: the value is dropped when done."""
        AutoFill().run(AutoFill.Params(template="госуслуги", field="password"))
        assert logger_module.redact(PASSWORD) == PASSWORD


# --------------------------------------------------------------------------- #
# Живые хранилища
# --------------------------------------------------------------------------- #


@pytest.mark.hardware
class TestLiveVaults:
    """Against a real ``keepassxc-cli``/``bw``, when the machine has one.

    Never in CI: these need software installed, a database file and a master
    password. Kept because the recorded-output tests above can only prove the
    argument vector is right, not that the tool accepts it.
    """

    def test_keepass_cli_отвечает(self) -> None:
        executable = find_cli("", "keepassxc-cli")
        if not executable:
            pytest.skip("keepassxc-cli не установлен")
        from ayris.actions.system.secrets.base import SubprocessRunner

        result = SubprocessRunner().run((executable, "--version"), timeout=15.0)
        assert result.ok
        assert result.stdout.strip()

    def test_bw_отвечает(self) -> None:
        executable = find_cli("", "bw", "bitwarden-cli")
        if not executable:
            pytest.skip("bw не установлен")
        from ayris.actions.system.secrets.base import SubprocessRunner

        result = SubprocessRunner().run((executable, "--version"), timeout=30.0)
        assert result.ok

    def test_настоящий_keyring_помнит_секрет(self) -> None:
        provider = KeyringProvider()
        if not provider.available():
            pytest.skip("Диспетчер учётных данных недоступен")
        ref = provider.save_field("ayris_тест_27", "password", "проверка-27")
        try:
            with provider.get_field("ayris_тест_27", "password") as secret:
                assert secret.value == "проверка-27"
        finally:
            assert provider.delete_field("ayris_тест_27", "password")
        assert ref.startswith("autofill.")
