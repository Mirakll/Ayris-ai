"""Templates that fill a form field, with the secret half kept out of the file.

A template is a named set of fields — ``почта``, ``телефон``, ``карта`` — and each
field is either a value written down in ``config.toml`` or a *reference* to a place
that keeps values properly. That split is the whole design:

* an email, a city and a street are not secrets, they go in the settings file, and
  a backup of that file is harmless;
* a password, a card number and a CVV never appear there. The settings file holds
  ``secret:`` and the value lives in the Windows Credential Manager, in KeePassXC
  or in Bitwarden — see :mod:`ayris.actions.system.secrets`.

References are spelled ``<source>:<entry>[#<field>]``:

``secret:``
    The configured provider, entry named after the template itself. With the
    keyring — the default — this is ``autofill.<template>.<field>``, which is where
    :func:`save_secret` puts what the user types into the settings tab.
``keepass:Банки/Сбербанк#password``
    A named entry in a vault, and optionally which of its fields to take. Without
    ``#`` the template's own field name is used.
``keyring:autofill.karta.number``
    A ref in the Credential Manager, spelled out. Refs there are latin, so a
    Russian name written by hand is transliterated to the same ref
    :func:`~ayris.actions.system.secrets.keyring_store.autofill_ref` generates.

Insertion has two routes and the default is to type, because typing never puts the
value on the clipboard at all. The clipboard route exists for the fields that
ignore synthesised keystrokes — some banking pages and a few Java forms — and there
the clipboard is wiped the moment the paste lands, rather than left holding a card
number until the next copy. Either way the value is registered with
:func:`ayris.utils.logger.guard_secret` for as long as it is in memory, so a log
line that manages to interpolate it prints ``[скрыто]``.

What reaches the log, the audit trail and the history is the *name* of the template
and of the field. Never the value, at any level, including ``DEBUG``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.input.keys import TypeMode, TypeText
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.clipboard import get_clipboard, suppress_record
from ayris.actions.system.secrets import (
    KeyringProvider,
    SecretProviderError,
    get_provider,
)
from ayris.actions.system.secrets.keyring_store import autofill_ref
from ayris.core.config import get_settings
from ayris.core.errors import ActionError
from ayris.utils.logger import forget_secret, get_logger, guard_secret

if TYPE_CHECKING:
    from ayris.core.config import AutofillActionsConfig

__all__ = [
    "KNOWN_FIELDS",
    "AutoFill",
    "FieldSpec",
    "FieldSummary",
    "FillMode",
    "ResolvedField",
    "SecretRef",
    "UnknownField",
    "UnknownTemplate",
    "describe_template",
    "field_spec",
    "forget_secret_field",
    "parse_reference",
    "resolve_field",
    "save_secret",
    "template_fields",
    "template_names",
]

_log = get_logger(__name__)

#: How long to wait after Ctrl+V before wiping the clipboard. The receiving
#: application reads the clipboard on its own schedule — a few tens of
#: milliseconds after the keystroke — so clearing instantly would clear it out
#: from under the paste. Short enough that the value is not sitting there.
_PASTE_SETTLE_S: Final = 0.15

#: What the preview of a non-secret value is clipped to in the settings list.
_PREVIEW_CHARS: Final = 60


class UnknownTemplate(ActionError):
    """No template by that name is configured."""


class UnknownField(ActionError):
    """The template exists but has no such field."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """What Ayris knows about a field name before seeing its value.

    ``secret`` decides masking, guarding and whether the clipboard is wiped
    afterwards — it is a property of the *name*, so a card number is treated as a
    card number even when a user typed it straight into ``config.toml`` against
    advice.
    """

    name: str
    title_ru: str
    secret: bool = False


def _spec(name: str, title_ru: str, *, secret: bool = False) -> FieldSpec:
    return FieldSpec(name=name, title_ru=title_ru, secret=secret)


#: The fields worth naming. Everything else is an «arbitrary pair» and gets a
#: :class:`FieldSpec` made up on the spot by :func:`field_spec`.
KNOWN_FIELDS: Final[dict[str, FieldSpec]] = {
    "email": _spec("email", "Почта"),
    "phone": _spec("phone", "Телефон"),
    "full_name": _spec("full_name", "ФИО"),
    "first_name": _spec("first_name", "Имя"),
    "last_name": _spec("last_name", "Фамилия"),
    "address": _spec("address", "Адрес"),
    "city": _spec("city", "Город"),
    "street": _spec("street", "Улица"),
    "house": _spec("house", "Дом"),
    "flat": _spec("flat", "Квартира"),
    "postcode": _spec("postcode", "Индекс"),
    "country": _spec("country", "Страна"),
    "company": _spec("company", "Организация"),
    "site": _spec("site", "Сайт"),
    "username": _spec("username", "Логин"),
    "login": _spec("login", "Логин"),
    "cardholder": _spec("cardholder", "Владелец карты"),
    "expiry": _spec("expiry", "Срок действия"),
    "expiry_month": _spec("expiry_month", "Месяц окончания"),
    "expiry_year": _spec("expiry_year", "Год окончания"),
    "password": _spec("password", "Пароль", secret=True),
    "pin": _spec("pin", "PIN", secret=True),
    "cvv": _spec("cvv", "CVV", secret=True),
    "code": _spec("code", "Код", secret=True),
    "card": _spec("card", "Номер карты", secret=True),
    "card_number": _spec("card_number", "Номер карты", secret=True),
    "number": _spec("number", "Номер карты", secret=True),
    "account": _spec("account", "Номер счёта", secret=True),
    "totp": _spec("totp", "Одноразовый код", secret=True),
    "token": _spec("token", "Токен", secret=True),
    "secret": _spec("secret", "Секрет", secret=True),
}

#: Fragments that make an unknown field name a secret one. A person naming a field
#: ``мир_карта`` or ``pass_госуслуги`` means what the words say, and defaulting such
#: a field to «not a secret» would put it in the log the first time it is used.
_SECRET_MARKERS: Final = (
    "pass",
    "пароль",
    "pin",
    "пин",
    "cvv",
    "cvc",
    "card",
    "карт",
    "secret",
    "секрет",
    "token",
    "токен",
    "totp",
    "otp",
    "код",
    "счёт",
    "счет",
    "account",
    "iban",
    "cvn",
)

#: Sources a reference can name. ``secret`` means «whatever the config says».
_SOURCES: Final = ("secret", "keyring", "keepass", "bitwarden")


def field_spec(name: str) -> FieldSpec:
    """What is known about ``name``, invented from the name itself if need be."""
    key = name.strip().casefold()
    known = KNOWN_FIELDS.get(key)
    if known is not None:
        return known
    secret = any(marker in key for marker in _SECRET_MARKERS)
    return FieldSpec(name=key or name, title_ru=name.strip() or name, secret=secret)


@dataclass(frozen=True, slots=True)
class SecretRef:
    """A parsed ``secret:``/``keepass:``/``bitwarden:``/``keyring:`` reference."""

    source: str
    entry: str = ""
    field: str = ""

    def provider_name(self) -> str:
        """Which provider to ask. ``""`` means «the configured one»."""
        return "" if self.source == "secret" else self.source


def parse_reference(raw: str) -> SecretRef | None:
    """A reference out of a configured value, or ``None`` if it is a literal.

    A value is a reference only when it starts with one of the four known sources
    followed by a colon. Anything else is taken literally — including a URL, which
    is why ``https:`` is not in the list and why the check is exact rather than
    «contains a colon».
    """
    text = raw.strip()
    head, sep, tail = text.partition(":")
    if not sep or head.strip().casefold() not in _SOURCES:
        return None
    entry, _, wanted = tail.strip().partition("#")
    return SecretRef(
        source=head.strip().casefold(),
        entry=entry.strip(),
        field=wanted.strip(),
    )


@dataclass(frozen=True, slots=True)
class ResolvedField:
    """A field with its value, ready to insert.

    The value is kept out of ``repr`` so that a stray ``%s`` on the whole object
    cannot print it; :meth:`guard` additionally registers it with the logging
    filter, which catches the case where something prints the value itself.

    Usable as a context manager: leaving the block forgets a secret value, so the
    window in which a log line could carry it ends with the insertion rather than
    with the process.
    """

    spec: FieldSpec
    value: str = dataclass_field(repr=False)
    source: str = "config"
    template: str = ""

    @property
    def secret(self) -> bool:
        return self.spec.secret

    @property
    def title_ru(self) -> str:
        return self.spec.title_ru

    def guard(self) -> ResolvedField:
        """Start redacting this value from the log, if it is a secret."""
        if self.spec.secret and self.value:
            guard_secret(self.value)
        return self

    def forget(self) -> None:
        """Stop redacting it."""
        if self.spec.secret and self.value:
            forget_secret(self.value)

    def __enter__(self) -> ResolvedField:
        return self.guard()

    def __exit__(self, *_: object) -> None:
        self.forget()


@dataclass(frozen=True, slots=True)
class FieldSummary:
    """One row of the settings tab: what a field is, and what it holds.

    ``preview`` is the mask for a secret and the clipped value for everything else.
    Built for display, so it is safe to put in a widget or a log line.
    """

    name: str
    title_ru: str
    secret: bool
    source: str
    preview: str


def _config(settings: AutofillActionsConfig | None = None) -> AutofillActionsConfig:
    return settings if settings is not None else get_settings().actions.autofill


def template_names(*, settings: AutofillActionsConfig | None = None) -> tuple[str, ...]:
    """Configured template names, in the order the settings file lists them."""
    return tuple(_config(settings).templates)


def _template(template: str, *, settings: AutofillActionsConfig | None = None) -> dict[str, str]:
    """The raw field map of a template, matched case-insensitively.

    Case-insensitive because the name is spoken: «заполни Карта» and «заполни
    карта» are the same request, and a template called ``Карта`` in the settings
    file should answer to both.
    """
    templates = _config(settings).templates
    found = templates.get(template)
    if found is not None:
        return found
    needle = template.strip().casefold()
    for name, fields in templates.items():
        if name.strip().casefold() == needle:
            return fields
    known = ", ".join(templates) or "ни одного"
    raise UnknownTemplate(
        f"no autofill template named {template!r}",
        user_message=f"Шаблона «{template}» нет. Есть: {known}.",
    )


def template_fields(
    template: str, *, settings: AutofillActionsConfig | None = None
) -> tuple[FieldSpec, ...]:
    """Specs of a template's fields, in configured order."""
    return tuple(field_spec(name) for name in _template(template, settings=settings))


def describe_template(
    template: str, *, settings: AutofillActionsConfig | None = None
) -> tuple[FieldSummary, ...]:
    """A template as the settings tab shows it — never with a secret value in it."""
    summaries: list[FieldSummary] = []
    for name, raw in _template(template, settings=settings).items():
        spec = field_spec(name)
        ref = parse_reference(raw)
        if ref is not None:
            source = ref.provider_name() or _config(settings).provider
            preview = f"из {source}"
        elif spec.secret:
            source = "config"
            preview = "[скрыто]"
        else:
            source = "config"
            preview = raw if len(raw) <= _PREVIEW_CHARS else f"{raw[:_PREVIEW_CHARS]}…"
        summaries.append(
            FieldSummary(
                name=spec.name,
                title_ru=spec.title_ru,
                secret=spec.secret,
                source=source,
                preview=preview,
            )
        )
    return tuple(summaries)


def resolve_field(
    template: str,
    field: str = "",
    *,
    settings: AutofillActionsConfig | None = None,
) -> ResolvedField:
    """The value of one field of one template, fetched from wherever it lives.

    An empty ``field`` is allowed when the template has exactly one field, because
    «вставь мою почту» is a whole template with one thing in it. Otherwise the
    error names the fields there are, which is the only useful thing to say.
    """
    fields = _template(template, settings=settings)
    name = field.strip()
    if not name:
        if len(fields) != 1:
            listed = ", ".join(fields) or "ни одного"
            raise UnknownField(
                f"template {template!r} needs an explicit field",
                user_message=f"У шаблона «{template}» несколько полей: {listed}.",
            )
        name = next(iter(fields))
    raw = _lookup(fields, name)
    if raw is None:
        listed = ", ".join(fields) or "ни одного"
        raise UnknownField(
            f"template {template!r} has no field {name!r}",
            user_message=f"У шаблона «{template}» нет поля «{name}». Есть: {listed}.",
        )
    spec = field_spec(name)
    ref = parse_reference(raw)
    if ref is None:
        return ResolvedField(spec=spec, value=raw, source="config", template=template)
    return _resolve_reference(ref, spec=spec, template=template)


def _lookup(fields: dict[str, str], name: str) -> str | None:
    """A field of a template by name, case-insensitively."""
    found = fields.get(name)
    if found is not None:
        return found
    needle = name.strip().casefold()
    for key, value in fields.items():
        if key.strip().casefold() == needle:
            return value
    return None


def _resolve_reference(
    ref: SecretRef,
    *,
    spec: FieldSpec,
    template: str,
) -> ResolvedField:
    """Ask a provider for a referenced value.

    The provider's own errors — «not installed», «locked», «no such entry» — are
    already Russian sentences, so they travel up as they are. What is added here is
    the name of the template, because «хранилище закрыто» is more useful when it
    says which field wanted it.
    """
    provider = get_provider(ref.provider_name())
    entry = ref.entry
    wanted = ref.field or spec.name
    if not entry:
        if provider.name != "keyring":
            raise UnknownField(
                f"reference in {template}.{spec.name} names no entry",
                user_message=(
                    f"Для поля «{spec.title_ru}» не указана запись в {provider.title_ru}."
                ),
            )
        entry = autofill_ref(template, spec.name)
    _log.debug("autofill %s.%s resolved from %s", template, spec.name, provider.name)
    try:
        secret = provider.get_field(entry, wanted)
    except SecretProviderError as exc:
        _log.warning("autofill %s.%s: %s", template, spec.name, exc)
        raise
    try:
        return ResolvedField(
            spec=spec,
            value=secret.value,
            source=provider.name,
            template=template,
        )
    finally:
        # The value now lives in the ResolvedField, which guards it in turn; the
        # SecretValue's own registration would otherwise outlive its usefulness.
        secret.forget()


def save_secret(template: str, field: str, value: str) -> str:
    """Put a secret in the Credential Manager. Returns the ref it went to.

    Writing always goes to the keyring, whatever ``provider`` says: KeePassXC and
    Bitwarden are read-only here on purpose — an assistant editing someone's vault
    is a different feature with a different risk.
    """
    return KeyringProvider().save_field(template, field, value)


def forget_secret_field(template: str, field: str) -> bool:
    """Delete a stored secret. ``False`` if there was nothing there."""
    return KeyringProvider().delete_field(template, field)


class FillMode(StrEnum):
    """How a value gets into the field under the cursor."""

    AUTO = "auto"
    TYPE = "type"
    CLIPBOARD = "clipboard"

    @property
    def title_ru(self) -> str:
        return _FILL_TITLES[self]


_FILL_TITLES: Final[dict[FillMode, str]] = {
    FillMode.AUTO: "Как в настройках",
    FillMode.TYPE: "Напечатать",
    FillMode.CLIPBOARD: "Через буфер обмена",
}


@register
class AutoFill(Action):
    """Fill the focused field from a template.

    «Вставь мою почту», «введи номер карты» — the value comes from the template,
    the template says where the value lives, and this action only decides how it
    reaches the field. The result reports the template and the field by name and
    the length of what was inserted; the value itself appears nowhere.
    """

    meta: ClassVar = ActionMeta(
        name="AutoFill",
        category=ActionCategory.INPUT,
        title_ru="Заполнить поле",
        description_ru=(
            "Подставляет значение из шаблона: почту, адрес, телефон или "
            "пароль из менеджера паролей."
        ),
    )

    class Params(ActionParams):
        template: str = Field(
            ...,
            min_length=1,
            max_length=100,
            description="Название шаблона, например «почта» или «карта»",
        )
        field: str = Field(
            default="",
            max_length=100,
            description="Какое поле подставить; пусто — единственное поле шаблона",
        )
        mode: FillMode = Field(
            default=FillMode.AUTO,
            description="Как подставить значение",
            json_schema_extra={"choices_ru": {str(m): m.title_ru for m in FillMode}},
        )
        clear_clipboard: bool | None = Field(
            default=None,
            description="Очистить буфер после вставки; пусто — из настроек",
        )

    def run(self, params: Params) -> ActionResult[int]:
        config = _config()
        resolved = resolve_field(params.template, params.field, settings=config)
        with resolved:
            if not resolved.value:
                return ActionResult.failed(
                    f"Поле «{resolved.title_ru}» пустое, подставлять нечего.",
                    value=0,
                    data={"template": params.template, "field": resolved.spec.name},
                )
            mode = self._resolve_mode(params.mode, config)
            if mode is FillMode.CLIPBOARD:
                cleared = self._paste(resolved, params=params, config=config)
            else:
                self._type(resolved)
                cleared = False
            _log.info(
                "autofill %s.%s: %d символов, %s%s",
                params.template,
                resolved.spec.name,
                len(resolved.value),
                mode.value,
                ", буфер очищен" if cleared else "",
            )
            return ActionResult.done(
                f"Подставил «{resolved.title_ru}» из шаблона «{params.template}».",
                value=len(resolved.value),
                data={
                    "template": params.template,
                    "field": resolved.spec.name,
                    "source": resolved.source,
                    "secret": resolved.secret,
                    "mode": mode.value,
                    "clipboard_cleared": cleared,
                    "length": len(resolved.value),
                },
            )

    @staticmethod
    def _resolve_mode(mode: FillMode, config: AutofillActionsConfig) -> FillMode:
        """``auto`` is whatever the settings say; the setting has only two values."""
        if mode is not FillMode.AUTO:
            return mode
        return FillMode.CLIPBOARD if config.paste_mode == "clipboard" else FillMode.TYPE

    @staticmethod
    def _type(resolved: ResolvedField) -> None:
        """Type the value. The clipboard is not touched at all on this route.

        Goes through :class:`~ayris.actions.input.keys.TypeText` in ``unicode``
        mode rather than ``auto``: ``auto`` switches to the clipboard above a
        length threshold, which for a long passphrase would quietly do the one
        thing this route exists to avoid.
        """
        TypeText().run(TypeText.Params(text=resolved.value, mode=TypeMode.UNICODE))

    @staticmethod
    def _paste(
        resolved: ResolvedField,
        *,
        params: AutoFill.Params,
        config: AutofillActionsConfig,
    ) -> bool:
        """Clipboard, Ctrl+V, and a wipe. Returns whether the wipe happened.

        The old clipboard contents are *not* restored here, unlike ``TypeText``'s
        clipboard route: restoring means writing something back, and «write the
        previous value» and «leave nothing behind» are different promises. This
        one keeps the second, which is what a card number needs.
        """
        from ayris.actions.system.clipboard import paste_shortcut

        wipe = config.clear_clipboard if params.clear_clipboard is None else params.clear_clipboard
        if resolved.secret:
            # A secret is wiped whatever the setting says: the setting decides
            # about a street address, not about a password sitting in the buffer
            # until the next copy.
            wipe = True
        backend = get_clipboard()
        suppress_record(resolved.value)
        backend.write_text(resolved.value)
        try:
            paste_shortcut()
            time.sleep(_PASTE_SETTLE_S)
        finally:
            if wipe:
                backend.clear()
        return wipe
