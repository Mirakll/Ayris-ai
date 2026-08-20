"""Wi-Fi and Bluetooth: switching the radios, listing networks, connecting.

Three layers, and the middle one is the reason the file is worth reading.

*The radios* are switched through WinRT ``Windows.Devices.Radios.Radio``, which is
the only way to turn Wi-Fi or Bluetooth off the way the Action Center does —
without administrator rights, without unbinding the adapter, and reversibly. The
projection (``winrt`` or ``winsdk``) is imported lazily and may be absent, so
everything below survives its absence: Wi-Fi falls back to ``netsh``, and
Bluetooth says so in Russian instead of failing obscurely.

*``netsh wlan``* answers everything the Radio API does not: which networks are in
range, how strong they are, how they are protected, and which of them have a saved
profile. It is a console program with localized output, so **the parsing lives in
pure functions over recorded text** — :func:`parse_wlan_interfaces`,
:func:`parse_wlan_networks`, :func:`parse_wlan_profiles` — and those are what the
tests exercise. None of them matches a Russian field label. What they match is
structure: ``SSID 3 :`` opens a network because ``SSID`` and ``BSSID`` are the two
keys ``netsh`` does not translate; a signal strength is the value that ends in
``%``; a security type is read from the *value* side, where ``WPA2`` and ``WEP``
are spelled the same in every locale. A build that renames «Проверка подлинности»
changes nothing here.

*Elevation* is deliberately not a property of these actions. WinRT switches a
radio unelevated, and ``netsh wlan connect`` / ``add profile`` work on the current
user's profile, so nothing here is marked ``require_admin`` — that flag would put
a UAC prompt in front of «включи вайфай» on every machine, including the ones
where it is not needed. The single path that genuinely needs rights is the
``netsh interface set interface admin=disabled`` fallback used when WinRT is
missing, and it borrows them for one command through
:func:`ayris.utils.admin.run_elevated` rather than elevating the assistant.

**A Wi-Fi password never lands anywhere but the credential store.** The parameter
is marked secret, so the registry writes ``***`` into ``audit`` and the log; the
value goes into ``keyring`` under a reference derived from the SSID, and the
profile XML that carries it to ``netsh`` is written to a file that is deleted in a
``finally``. Nothing in this module logs a password, including at DEBUG.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol
from xml.sax.saxutils import escape

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable, SecretsError
from ayris.core.secrets import get_secrets
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "NETSH_TIMEOUT_S",
    "BluetoothDevice",
    "ConnectWifi",
    "ListBluetooth",
    "ListWifi",
    "NetshResult",
    "NetshRunner",
    "RadioBackend",
    "RadioKind",
    "RadioMode",
    "RadioState",
    "RecordingRadio",
    "ScriptedNetsh",
    "SetBluetooth",
    "SetWifi",
    "SubprocessNetsh",
    "UnavailableRadio",
    "WifiNetwork",
    "WifiSecurity",
    "WinRtRadio",
    "WlanInterface",
    "connect_wifi",
    "console_encoding",
    "get_netsh",
    "get_radio",
    "parse_wlan_interfaces",
    "parse_wlan_networks",
    "parse_wlan_profiles",
    "profile_xml",
    "require_interface",
    "saved_profiles",
    "scan_networks",
    "secret_ref_for_ssid",
    "security_from_values",
    "set_netsh",
    "set_radio",
    "signal_from_values",
    "switch_radio",
    "wlan_interfaces",
]

_log = get_logger(__name__)

#: How long a ``netsh`` call may take. A scan is the slow one — the driver has to
#: sweep the band — and the action's own timeout is set above this.
NETSH_TIMEOUT_S: Final = 20.0

#: ``CREATE_NO_WINDOW``: ``netsh`` must not flash a console window at the user.
#: Spelled out because :mod:`subprocess` only defines it on Windows.
_CREATE_NO_WINDOW: Final = 0x08000000

#: Encoding to fall back on when Windows has no console to report a code page for.
_FALLBACK_ENCODING: Final = "utf-8"

#: A ``key : value`` line, which is what almost every ``netsh`` line is. The key
#: may carry a trailing index — ``SSID 3``, ``BSSID 1`` — and that index is what
#: makes the output parseable without reading the label.
_KEY_VALUE: Final = re.compile(r"^(?P<indent>\s*)(?P<key>[^:]+?)\s*:\s?(?P<value>.*)$")

#: Trailing index of a repeated key: the ``3`` of ``SSID 3``.
_INDEXED_KEY: Final = re.compile(r"^(?P<name>.*?)\s+(?P<index>\d+)$")

#: A percentage value, which is how ``netsh`` reports signal strength in every
#: locale. The key beside it is translated; this is not.
_PERCENT: Final = re.compile(r"^(\d{1,3})\s*%$")

#: Placeholder ``netsh`` prints for an empty list: ``<None>``, «<Отсутствует>».
_EMPTY_MARKER: Final = re.compile(r"^<.*>$")


class RadioKind(StrEnum):
    """Which radio is meant. Mirrors the two members of WinRT ``RadioKind``."""

    WIFI = "wifi"
    BLUETOOTH = "bluetooth"

    @property
    def title_ru(self) -> str:
        return "Wi-Fi" if self is RadioKind.WIFI else "Bluetooth"


class RadioState(StrEnum):
    """State of one radio.

    ``UNKNOWN`` is not «off»: a machine with no adapter and a machine whose radio
    is switched off need different sentences said to them, and collapsing the two
    is how «включи вайфай» comes to answer «уже включён» on a desktop that has
    never had a Wi-Fi card.
    """

    ON = "on"
    OFF = "off"
    DISABLED = "disabled"
    UNKNOWN = "unknown"

    @property
    def title_ru(self) -> str:
        return _RADIO_STATE_RU[self]


_RADIO_STATE_RU: Final[dict[RadioState, str]] = {
    RadioState.ON: "включён",
    RadioState.OFF: "выключен",
    RadioState.DISABLED: "отключён в системе",
    RadioState.UNKNOWN: "неизвестно",
}


class RadioMode(StrEnum):
    """What :class:`SetWifi` and :class:`SetBluetooth` were asked to do."""

    ON = "on"
    OFF = "off"
    TOGGLE = "toggle"

    @property
    def title_ru(self) -> str:
        return _RADIO_MODE_RU[self]

    def target(self, current: RadioState) -> bool:
        """The state to switch to, resolving ``TOGGLE`` against ``current``.

        An unknown current state toggles to «on»: the reason a person says
        «переключи вайфай» to a radio Ayris cannot read is almost always that they
        want it working.
        """
        if self is RadioMode.ON:
            return True
        if self is RadioMode.OFF:
            return False
        return current is not RadioState.ON


_RADIO_MODE_RU: Final[dict[RadioMode, str]] = {
    RadioMode.ON: "Включить",
    RadioMode.OFF: "Выключить",
    RadioMode.TOGGLE: "Переключить",
}


class WifiSecurity(StrEnum):
    """How a network is protected, as far as it can be told from a scan.

    Read from the *value* side of the ``netsh`` output, where the tokens are the
    same in every locale. ``UNKNOWN`` means a value nobody here recognised — a
    newer scheme, most likely — and is deliberately not folded into ``OPEN``: an
    open network is one a person may join without a password, and guessing that
    about an unrecognised one would be a security claim made on no evidence.
    """

    OPEN = "open"
    WEP = "wep"
    WPA = "wpa"
    WPA2 = "wpa2"
    WPA3 = "wpa3"
    UNKNOWN = "unknown"

    @property
    def title_ru(self) -> str:
        return _SECURITY_RU[self]

    @property
    def needs_password(self) -> bool:
        """Whether joining this network requires a key from the user."""
        return self is not WifiSecurity.OPEN


_SECURITY_RU: Final[dict[WifiSecurity, str]] = {
    WifiSecurity.OPEN: "открытая",
    WifiSecurity.WEP: "WEP",
    WifiSecurity.WPA: "WPA",
    WifiSecurity.WPA2: "WPA2",
    WifiSecurity.WPA3: "WPA3",
    WifiSecurity.UNKNOWN: "защита неизвестна",
}

#: Security tokens, strongest first — a network advertising both WPA3 and WPA2 is
#: reported as the better of the two, which is what it will actually be joined by.
_SECURITY_TOKENS: Final[tuple[tuple[str, WifiSecurity], ...]] = (
    ("wpa3", WifiSecurity.WPA3),
    ("wpa2", WifiSecurity.WPA2),
    ("rsna", WifiSecurity.WPA2),
    ("wpa", WifiSecurity.WPA),
    ("wep", WifiSecurity.WEP),
)

#: Values that mean «no authentication» in the locales Windows ships. Matched only
#: after every security token has been ruled out, and only as a whole value.
_OPEN_VALUES: Final[frozenset[str]] = frozenset(
    {"open", "none", "открыть", "открытая", "нет", "отсутствует"}
)


@dataclass(frozen=True, slots=True)
class WlanInterface:
    """One wireless adapter, as ``netsh wlan show interfaces`` describes it.

    ``connected`` is derived from the presence of an SSID rather than from the
    state text: «connected», «подключен» and «Соединено» are three spellings of
    the same thing across locales and builds, while an SSID is either there or it
    is not.
    """

    name: str = ""
    description: str = ""
    guid: str = ""
    ssid: str = ""
    bssid: str = ""
    state: str = ""
    profile: str = ""

    @property
    def connected(self) -> bool:
        """Whether the adapter is on a network right now."""
        return bool(self.ssid)


@dataclass(frozen=True, slots=True)
class WifiNetwork:
    """One network in range.

    ``signal`` is the percentage ``netsh`` reports, ``0`` when it said nothing.
    ``saved`` is filled in by :meth:`ListWifi.run` from the profile list, not by
    the scan: ``netsh wlan show networks`` does not mention profiles at all.
    """

    ssid: str = ""
    security: WifiSecurity = WifiSecurity.UNKNOWN
    enterprise: bool = False
    signal: int = 0
    channel: int = 0
    radio: str = ""
    bssid: str = ""
    saved: bool = False

    @property
    def hidden(self) -> bool:
        """Whether the network broadcasts no name."""
        return not self.ssid

    @property
    def title_ru(self) -> str:
        """The network as Ayris would read it out."""
        name = self.ssid or "сеть без имени"
        parts = [f"«{name}»", f"{self.signal}%", self.security.title_ru]
        if self.saved:
            parts.append("есть профиль")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class BluetoothDevice:
    """One paired Bluetooth device."""

    name: str = ""
    kind: str = ""
    connected: bool = False
    paired: bool = True

    @property
    def title_ru(self) -> str:
        state = "подключено" if self.connected else "сопряжено"
        return f"«{self.name or 'без имени'}» — {state}"


# --------------------------------------------------------------------------- #
# pure parsers over recorded netsh output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Line:
    """One parsed ``key : value`` line, with the bits that survive translation."""

    indent: int
    key: str
    value: str
    index: int = 0

    @property
    def folded(self) -> str:
        """The key, lowercased and stripped of its index, for comparison."""
        return self.key.casefold()


def _blocks(output: str) -> Iterator[tuple[_Line, ...]]:
    """Split ``netsh`` output into the runs of lines a blank line separates.

    Windows separates one adapter from the next with a blank line, and separates
    the adapter block from the trailing global notice («Состояние размещенной
    сети») the same way. That blank line is the only record delimiter in the output
    that no locale touches, so it is the one used.
    """
    block: list[_Line] = []
    for raw in _rows(output):
        if not raw.strip():
            if block:
                yield tuple(block)
                block = []
            continue
        line = _parse_line(raw)
        if line is not None:
            block.append(line)
    if block:
        yield tuple(block)


def _rows(output: str) -> list[str]:
    """``output`` split into lines, with carriage returns thrown away first.

    Dropping ``\\r`` rather than letting :meth:`str.splitlines` handle it is what
    makes a blank line trustworthy as a record delimiter. Text that has been
    through a newline-translating writer carries ``\\r\\r\\n``, which
    ``splitlines`` reads as a line break followed by an empty line — so every
    single line would look like its own block, and :func:`parse_wlan_interfaces`
    would find no adapter in output that plainly describes one.
    """
    return output.replace("\r", "").split("\n")


def _parse_line(raw: str) -> _Line | None:
    """One ``key : value`` line, or ``None`` when the line is not one."""
    match = _KEY_VALUE.match(raw.rstrip())
    if match is None:
        return None
    key = match.group("key").strip()
    if not key:
        return None
    index = 0
    indexed = _INDEXED_KEY.match(key)
    if indexed is not None:
        key = indexed.group("name")
        index = int(indexed.group("index"))
    return _Line(
        indent=len(match.group("indent")),
        key=key,
        value=match.group("value").strip(),
        index=index,
    )


def _lines(output: str) -> Iterator[_Line]:
    """Yield the ``key : value`` lines of ``netsh`` output, skipping the rest.

    The rest is headings, rules of dashes and blank lines. A heading ends in a
    colon with nothing after it, which this yields as an empty value — callers
    that care about the difference check ``value``.
    """
    for raw in _rows(output):
        if not raw.strip():
            continue
        line = _parse_line(raw)
        if line is not None:
            yield line


def parse_wlan_interfaces(output: str) -> tuple[WlanInterface, ...]:
    """Parse ``netsh wlan show interfaces``.

    One adapter is one blank-line-delimited block containing a value shaped like a
    GUID. Both halves of that matter. The block boundary keeps the trailing
    «Состояние размещенной сети» notice out of the record, and the **value** shape
    is what finds the GUID at all: the key is «Идентификатор GUID» in Russian and
    ``GUID`` in English, so looking for the key would work in one locale and not
    the other. A GUID is spelled the same everywhere.

    With the GUID line located, the two keys above it are the adapter's name and
    its description, by position. Position works there and nowhere else in the
    block: Windows 11 inserted «Тип интерфейса» between the physical address and
    the state, so counting *down* from the GUID gives a different field on Windows
    10 and Windows 11. The state is therefore read as the value immediately above
    the SSID — Windows prints them adjacent in both — and as the last value of the
    block when the adapter is not on a network and prints no SSID at all.

    Output describing no adapter — «Нет интерфейса беспроводной сети», or the
    message about ``wlansvc`` not running — yields an empty tuple. The caller turns
    that into :class:`~ayris.core.errors.ActionUnavailable`; deciding it here would
    mean the parser had to know what it is being asked for.
    """
    interfaces = []
    for block in _blocks(output):
        filled = [line for line in block if line.value]
        guid_at = next(
            (position for position, line in enumerate(filled) if _GUID_VALUE.match(line.value)),
            -1,
        )
        if guid_at < 0:
            continue
        by_key = {line.folded: line.value for line in filled}
        ssid_at = next(
            (position for position, line in enumerate(filled) if line.folded == "ssid"),
            -1,
        )
        interfaces.append(
            WlanInterface(
                name=filled[guid_at - 2].value if guid_at >= 2 else "",
                description=filled[guid_at - 1].value if guid_at >= 1 else "",
                guid=filled[guid_at].value,
                ssid=by_key.get("ssid", ""),
                bssid=by_key.get("bssid", ""),
                state=_state_value(filled, ssid_at),
                profile=_profile_value(filled, by_key.get("ssid", "")),
            )
        )
    return tuple(interfaces)


#: A GUID value, braces optional. The one field of an adapter block that is spelled
#: identically in every locale, which is why the block is found by it.
_GUID_VALUE: Final = re.compile(
    r"^\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?$",
    re.IGNORECASE,
)


def _state_value(filled: Sequence[_Line], ssid_at: int) -> str:
    """The adapter's state text: the value above the SSID, or the block's last.

    Localized text that is only ever displayed or logged —
    :attr:`WlanInterface.connected` is derived from the SSID, not from this, so a
    build that renames it breaks nothing that matters.
    """
    if ssid_at > 0:
        return filled[ssid_at - 1].value
    return filled[-1].value if filled else ""


def _profile_value(filled: Sequence[_Line], ssid: str) -> str:
    """The saved profile the adapter connected through, or ``""``.

    Windows names the profile after the SSID, and the key («Профиль») is localized,
    so this looks for a second line carrying the SSID as its value. Its absence is
    meaningful: the adapter joined without a profile, and there is nothing to
    reconnect through later.
    """
    if not ssid:
        return ""
    matches = [line for line in filled if line.value == ssid and line.folded != "ssid"]
    return ssid if matches else ""


@dataclass(slots=True)
class _Bss:
    """One radio of one network, while it is being read line by line.

    Mutable and private, because the fields arrive on separate lines and the
    strongest of several has to be kept whole: taking the best signal from one
    BSSID and the channel from another would report a channel the network is not
    actually reachable on at that strength.
    """

    bssid: str = ""
    signal: int = 0
    channel: int = 0
    radio: str = ""

    def better(self, other: _Bss | None) -> _Bss:
        """Whichever of the two has the stronger signal, this one on a tie."""
        if other is None:
            return self
        return other if other.signal > self.signal else self


def parse_wlan_networks(output: str) -> tuple[WifiNetwork, ...]:
    """Parse ``netsh wlan show networks mode=bssid``.

    ``SSID <n>`` opens a network and ``BSSID <n>`` opens one of its radios; both
    keys keep their Latin spelling in every locale, which is the whole reason this
    can be parsed at all. The lines between the SSID and the first BSSID describe
    the network, and their *values* are what :func:`security_from_values` reads —
    «Проверка подлинности» and «Authentication» differ only on the side this
    ignores.

    Inside a BSSID block, **indentation decides what counts**. Windows nests a
    «Нагрузка BSS» sub-block one level deeper, and that sub-block contains bare
    numbers — connected stations, channel utilisation — which look exactly like a
    channel to anything reading values alone. Lines deeper than the block's own
    fields are therefore skipped, which is why «Подключенные станции: 0» does not
    become channel 0.

    Only the strongest BSSID of a network is kept. A person choosing a network by
    name does not care that the office router has three radios, and reporting
    «MyNet, 40%» because the weaker one came second in the output would be wrong in
    the one field they do care about.
    """
    networks: list[WifiNetwork] = []
    ssid: str | None = None
    described: list[str] = []
    current: _Bss | None = None
    best = _Bss()
    field_indent = 0

    def flush() -> None:
        if ssid is None:
            return
        strongest = best.better(current)
        security, enterprise = security_from_values(described)
        networks.append(
            WifiNetwork(
                ssid=ssid,
                security=security,
                enterprise=enterprise,
                signal=strongest.signal,
                channel=strongest.channel,
                radio=strongest.radio,
                bssid=strongest.bssid,
            )
        )

    for line in _lines(output):
        folded = line.folded
        if folded == "ssid" and line.index:
            flush()
            ssid, described = line.value, []
            current, best, field_indent = None, _Bss(), 0
            continue
        if ssid is None:
            continue
        if folded == "bssid" and line.index:
            best = best.better(current)
            current, field_indent = _Bss(bssid=line.value), 0
            continue
        if current is None:
            described.append(line.value)
            continue
        if field_indent == 0:
            field_indent = line.indent
        if line.indent > field_indent:
            continue
        if _PERCENT.match(line.value):
            current.signal = signal_from_values([line.value])
        elif not current.channel and _looks_like_channel(line.value):
            current.channel = int(line.value)
        elif not current.radio and _looks_like_radio(line.value):
            current.radio = line.value
    flush()
    return tuple(networks)


#: ``802.11ax``, ``802.11n`` and the rest. The radio type is the one free-text
#: value in a BSSID block with a recognisable shape, so it is matched rather than
#: read off a localized key.
_RADIO_TYPE: Final = re.compile(r"^802\.11\S*$")

#: Highest channel number any Wi-Fi band uses — 6 GHz stops at 233. The upper
#: bound is what keeps a bare number from a neighbouring field, like the 31250 of
#: «Средняя доступная емкость», from being read as a channel.
_MAX_CHANNEL: Final = 233


def _looks_like_radio(value: str) -> bool:
    return bool(_RADIO_TYPE.match(value))


def _looks_like_channel(value: str) -> bool:
    """Whether ``value`` could be a channel number: bare digits in range."""
    return value.isdigit() and 1 <= int(value) <= _MAX_CHANNEL


def security_from_values(values: Iterable[str]) -> tuple[WifiSecurity, bool]:
    """Read the security type out of the *values* of a network's block.

    Keys are translated, values are not: «Проверка подлинности : WPA2-Personal»
    and «Authentication : WPA2-Personal» differ only on the side this ignores.
    WEP is the case that makes the approach necessary rather than merely
    convenient — a WEP network authenticates as «Открыть»/«Open» and names WEP
    only under the encryption key, so a parser reading one labelled field would
    report it as open.

    Returns the type and whether it is an enterprise network, which is the one
    thing a voice assistant cannot help with: joining it needs a certificate or a
    domain account, not a password.
    """
    seen = [value.casefold() for value in values]
    enterprise = any("enterprise" in value or "802.1x" in value for value in seen)
    for token, security in _SECURITY_TOKENS:
        if any(token in value for value in seen):
            return security, enterprise
    if any(value in _OPEN_VALUES for value in seen):
        return WifiSecurity.OPEN, enterprise
    return WifiSecurity.UNKNOWN, enterprise


def signal_from_values(values: Iterable[str]) -> int:
    """The largest percentage among ``values``, or ``0`` when there is none.

    A percentage is the only value in a BSSID block shaped like one, so no key has
    to be recognised. Out-of-range numbers are clamped rather than rejected: some
    drivers report 100+ and the number is a hint to a human either way.
    """
    best = 0
    for value in values:
        match = _PERCENT.match(value.strip())
        if match is not None:
            best = max(best, min(100, int(match.group(1))))
    return best


def parse_wlan_profiles(output: str) -> tuple[str, ...]:
    """Parse ``netsh wlan show profiles`` into saved profile names.

    The structure again, not the labels: a profile is an *indented* line with a
    non-empty value, while the heading above it («Профили пользователя», followed
    by a rule of dashes) is not indented and carries no value. ``<Отсутствует>``
    and ``<None>`` mark an empty group and are dropped.

    Duplicates are collapsed keeping the first occurrence, because the same
    profile can appear under both the group-policy and the user heading.
    """
    names: list[str] = []
    for line in _lines(output):
        if line.indent == 0 or not line.value:
            continue
        if _EMPTY_MARKER.match(line.value):
            continue
        if line.value not in names:
            names.append(line.value)
    return tuple(names)


def profile_xml(ssid: str, password: str, *, security: WifiSecurity = WifiSecurity.WPA2) -> str:
    """Build the WLAN profile XML ``netsh wlan add profile`` takes.

    ``keyMaterial`` is the password in clear text — that is what the format is —
    which is why the caller writes this to a file it deletes and never logs the
    result. ``connectionMode`` is ``auto`` so the network is rejoined later
    without asking again, which is what «подключись и запомни» means.

    An open network gets a profile with no key at all: sending an empty
    ``keyMaterial`` makes ``netsh`` reject the profile rather than treat it as
    open.
    """
    name = escape(ssid)
    if security is WifiSecurity.OPEN or not password:
        security_block = (
            "<authEncryption><authentication>open</authentication>"
            "<encryption>none</encryption><useOneX>false</useOneX></authEncryption>"
        )
    else:
        security_block = (
            f"<authEncryption><authentication>{_AUTH_XML[security]}</authentication>"
            f"<encryption>{_CIPHER_XML[security]}</encryption>"
            "<useOneX>false</useOneX></authEncryption>"
            "<sharedKey><keyType>passPhrase</keyType><protected>false</protected>"
            f"<keyMaterial>{escape(password)}</keyMaterial></sharedKey>"
        )
    return (
        '<?xml version="1.0"?>'
        '<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">'
        f"<name>{name}</name>"
        f"<SSIDConfig><SSID><name>{name}</name></SSID></SSIDConfig>"
        "<connectionType>ESS</connectionType>"
        "<connectionMode>auto</connectionMode>"
        f"<MSM><security>{security_block}</security></MSM>"
        "</WLANProfile>"
    )


#: ``authentication`` element per security type. ``WPA3SAE`` is what Windows 11
#: calls WPA3-Personal in a profile; anything unrecognised is written as WPA2,
#: which every current access point still accepts.
_AUTH_XML: Final[dict[WifiSecurity, str]] = {
    WifiSecurity.WEP: "open",
    WifiSecurity.WPA: "WPAPSK",
    WifiSecurity.WPA2: "WPA2PSK",
    WifiSecurity.WPA3: "WPA3SAE",
    WifiSecurity.UNKNOWN: "WPA2PSK",
    WifiSecurity.OPEN: "open",
}

_CIPHER_XML: Final[dict[WifiSecurity, str]] = {
    WifiSecurity.WEP: "WEP",
    WifiSecurity.WPA: "TKIP",
    WifiSecurity.WPA2: "AES",
    WifiSecurity.WPA3: "AES",
    WifiSecurity.UNKNOWN: "AES",
    WifiSecurity.OPEN: "none",
}


def secret_ref_for_ssid(ssid: str) -> str:
    """Credential-store reference for one network's password.

    A reference has to be a short lowercase identifier
    (:func:`ayris.core.secrets.is_valid_ref`), and an SSID is arbitrary text —
    spaces, emoji, Cyrillic — so the name is hashed rather than sanitised.
    Sanitising would collide: «Home Wi-Fi» and «home_wi_fi» would end up sharing
    one entry and one password.
    """
    digest = hashlib.sha256(ssid.encode("utf-8")).hexdigest()[:16]
    return f"wifi.{digest}"


# --------------------------------------------------------------------------- #
# netsh, behind an interface
# --------------------------------------------------------------------------- #


def console_encoding() -> str:
    """Encoding a console program's output has to be decoded with.

    The console keeps the OEM code page — 866 on a Russian Windows — while
    :func:`locale.getpreferredencoding` reports the ANSI one, 1251. Decoding
    ``netsh`` with the wrong one of those turns every SSID with a Cyrillic letter
    into mojibake, and a person cannot connect to a network they cannot read the
    name of.

    A windowed build has no console of its own, so the OEM code page is asked for
    directly in that case: the child ``netsh`` allocates a console at the system
    default, not at whatever this process's locale happens to be.
    """
    for codepage in (winapi.console_output_codepage(), winapi.oem_codepage()):
        if codepage:
            return f"cp{codepage}"
    return _FALLBACK_ENCODING


@dataclass(frozen=True, slots=True)
class NetshResult:
    """What one ``netsh`` call produced.

    A non-zero ``returncode`` is not automatically a failure worth reporting:
    ``netsh wlan show interfaces`` exits non-zero when there is no wireless
    interface, which is an answer rather than an error, so callers look at the
    text too.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def text(self) -> str:
        """Both streams, for the parsers — ``netsh`` writes some notices to stderr."""
        return self.stdout if self.stdout.strip() else self.stderr


class NetshRunner(Protocol):
    """``netsh``, as far as this module needs it.

    Behind an interface for the reason the task spells out: the CI runner has no
    wireless adapter, so the tests hand over recorded output and check which
    arguments were asked for.
    """

    def run(self, arguments: Sequence[str], *, timeout_s: float = NETSH_TIMEOUT_S) -> NetshResult:
        """Run ``netsh`` with ``arguments`` and return what it said."""
        ...


class SubprocessNetsh:
    """The real one: ``netsh`` in a child process with no console window."""

    def run(self, arguments: Sequence[str], *, timeout_s: float = NETSH_TIMEOUT_S) -> NetshResult:
        try:
            completed = subprocess.run(
                ["netsh", *arguments],
                capture_output=True,
                timeout=timeout_s,
                creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _log.warning("netsh %s не ответил за %.0f с", " ".join(arguments), timeout_s)
            raise ActionError(
                f"netsh {' '.join(arguments)} timed out after {timeout_s}s",
                user_message="Windows не ответила на запрос о сети вовремя",
            ) from exc
        except OSError as exc:
            raise ActionUnavailable(
                f"netsh could not be started: {exc}",
                user_message="Не удалось запустить netsh — управление сетью недоступно",
            ) from exc
        encoding = console_encoding()
        return NetshResult(
            returncode=completed.returncode,
            stdout=completed.stdout.decode(encoding, errors="replace"),
            stderr=completed.stderr.decode(encoding, errors="replace"),
        )


class ScriptedNetsh:
    """Answers from a script instead of from the driver. The test seam.

    Keyed by a prefix of the arguments, so a test writes ``{"wlan show
    interfaces": ...}`` and does not have to predict whether the call also passed
    ``mode=bssid``. Every call is recorded in :attr:`calls`, which is how the tests
    check that «выключи вайфай» reached ``netsh`` as the command it should have and
    not as something coarser.

    An unscripted call answers empty with ``returncode`` 1 rather than raising: a
    test about a missing adapter is exactly a test about ``netsh`` having nothing
    to say.
    """

    def __init__(self, answers: Mapping[str, str] | None = None) -> None:
        self.answers: dict[str, str] = dict(answers or {})
        self.calls: list[tuple[str, ...]] = []
        self.codes: dict[str, int] = {}

    def run(self, arguments: Sequence[str], *, timeout_s: float = NETSH_TIMEOUT_S) -> NetshResult:
        del timeout_s
        self.calls.append(tuple(arguments))
        joined = " ".join(arguments)
        for prefix, output in self.answers.items():
            if joined.startswith(prefix):
                return NetshResult(returncode=self.codes.get(prefix, 0), stdout=output)
        return NetshResult(returncode=1, stdout="")

    def called(self, prefix: str) -> bool:
        """Whether any recorded call starts with ``prefix``."""
        return any(" ".join(call).startswith(prefix) for call in self.calls)


_netsh: NetshRunner | None = None


def get_netsh() -> NetshRunner:
    """The ``netsh`` runner in force. Real unless a test replaced it."""
    global _netsh
    if _netsh is None:
        _netsh = SubprocessNetsh()
    return _netsh


def set_netsh(runner: NetshRunner | None) -> None:
    """Install a runner, or go back to the real one with ``None``."""
    global _netsh
    _netsh = runner


# --------------------------------------------------------------------------- #
# the radios, behind an interface
# --------------------------------------------------------------------------- #


class RadioBackend(Protocol):
    """Switching a radio on and off, and reading which way it is set."""

    @property
    def available(self) -> bool:
        """Whether this backend can control anything at all."""
        ...

    def state(self, kind: RadioKind) -> RadioState:
        """Current state of one radio, ``UNKNOWN`` when there is none of that kind."""
        ...

    def switch(self, kind: RadioKind, *, on: bool) -> RadioState:
        """Set one radio and return the state it ended up in."""
        ...

    def devices(self) -> tuple[BluetoothDevice, ...]:
        """Paired Bluetooth devices, empty when they cannot be enumerated."""
        ...


class UnavailableRadio:
    """No WinRT here. Every question answers «неизвестно», every switch refuses.

    The default on a machine without the projection, and on Linux. It refuses
    rather than pretending, because :class:`SetWifi` has a ``netsh`` fallback to
    try and :class:`SetBluetooth` has none — the difference has to be visible.
    """

    @property
    def available(self) -> bool:
        return False

    def state(self, kind: RadioKind) -> RadioState:
        del kind
        return RadioState.UNKNOWN

    def switch(self, kind: RadioKind, *, on: bool) -> RadioState:
        del on
        raise ActionUnavailable(
            f"no radio backend for {kind}",
            user_message=(
                f"{kind.title_ru} нельзя переключить: "
                "нет WinRT — установите пакет winrt или winsdk"
            ),
        )

    def devices(self) -> tuple[BluetoothDevice, ...]:
        return ()


class RecordingRadio:
    """Keeps the states in a dict instead of touching a radio. Dry run and tests.

    ``switch`` honours the request, so an action's own logic — resolving
    ``toggle``, noticing that the radio is already the way it was asked for — is
    tested against something that behaves like a radio rather than against a stub
    that always says yes.
    """

    def __init__(
        self,
        *,
        states: Mapping[RadioKind, RadioState] | None = None,
        available: bool = True,
        paired: Sequence[BluetoothDevice] = (),
    ) -> None:
        self.states: dict[RadioKind, RadioState] = dict(states or {})
        self._available = available
        self.paired = tuple(paired)
        self.switches: list[tuple[RadioKind, bool]] = []

    @property
    def available(self) -> bool:
        return self._available

    def state(self, kind: RadioKind) -> RadioState:
        return self.states.get(kind, RadioState.UNKNOWN)

    def switch(self, kind: RadioKind, *, on: bool) -> RadioState:
        self.switches.append((kind, on))
        if self.state(kind) is RadioState.UNKNOWN:
            raise ActionUnavailable(
                f"no {kind} radio",
                user_message=f"{kind.title_ru}-адаптер не найден",
            )
        self.states[kind] = RadioState.ON if on else RadioState.OFF
        return self.states[kind]

    def devices(self) -> tuple[BluetoothDevice, ...]:
        return self.paired


class WinRtRadio:
    """The real one, over ``Windows.Devices.Radios`` and ``Windows.Devices.Enumeration``.

    The projection is imported on first use and may be either ``winrt`` (the
    current package) or ``winsdk`` (its predecessor, still what many machines
    have). Neither is a dependency of Ayris: Wi-Fi degrades to ``netsh`` without
    them and Bluetooth says so plainly, which is better than making every install
    carry a projection for a feature not everyone uses.

    ``Radio.request_access_async`` is asked once. A user who declines gets a
    sentence about it and no retry loop — the setting lives in Windows privacy
    settings, and hammering the prompt would not change it.
    """

    #: Bluetooth classic and LE, as ``System.Devices.Aep.ProtocolId`` values. The
    #: two have to be asked for separately: a headset is classic, a tracker is LE,
    #: and a person means both by «блютус-устройства».
    _PROTOCOLS: ClassVar[tuple[str, ...]] = (
        "{e0cbf06c-cd8b-4647-bb8a-263b43f0f974}",
        "{bb7bb05e-5972-42b5-bfaf-6c1c123a4f9d}",
    )

    def __init__(self) -> None:
        self._radios: Any = None
        self._enumeration: Any = None
        self._access: str = ""
        self._loaded = False

    # -- the projection ---------------------------------------------------- #

    def _load(self) -> None:
        """Import a projection once, remembering that it failed if it did."""
        if self._loaded:
            return
        self._loaded = True
        for package in ("winrt.windows.devices", "winsdk.windows.devices"):
            try:
                radios = __import__(f"{package}.radios", fromlist=["Radio"])
                enumeration = __import__(f"{package}.enumeration", fromlist=["DeviceInformation"])
            except ImportError:
                continue
            self._radios, self._enumeration = radios, enumeration
            _log.debug("WinRT-проекция: %s", package)
            return
        _log.info("WinRT-проекции нет, Wi-Fi пойдёт через netsh, Bluetooth недоступен")

    @property
    def available(self) -> bool:
        self._load()
        return self._radios is not None

    # -- access ------------------------------------------------------------ #

    def _ensure_access(self) -> None:
        """Ask Windows for permission to touch the radios, once.

        A refusal is a plain Russian sentence naming where the setting lives.
        Ayris cannot change it from here, and a user who said no deserves to be
        told what they said no to rather than «operation failed».
        """
        if self._access == "allowed":
            return
        self._load()
        if self._radios is None:
            raise ActionUnavailable(
                "no WinRT projection",
                user_message="Нет WinRT — переключение радио недоступно",
            )
        status = str(_await(self._radios.Radio.request_access_async())).lower()
        self._access = "allowed" if "allowed" in status else status
        if self._access != "allowed":
            raise ActionUnavailable(
                f"radio access {status}",
                user_message=(
                    "Windows не разрешила управлять радио — "
                    "включите доступ в «Параметры → Конфиденциальность → Радио»"
                ),
            )

    def _find(self, kind: RadioKind) -> Any:
        """The first radio of ``kind``, or ``None`` when the machine has none."""
        self._ensure_access()
        wanted = "wifi" if kind is RadioKind.WIFI else "bluetooth"
        for radio in _await(self._radios.Radio.get_radios_async()) or ():
            if str(radio.kind).rsplit(".", 1)[-1].casefold() == wanted:
                return radio
        return None

    # -- the interface ----------------------------------------------------- #

    def state(self, kind: RadioKind) -> RadioState:
        try:
            radio = self._find(kind)
        except ActionError:
            return RadioState.UNKNOWN
        if radio is None:
            return RadioState.UNKNOWN
        return _radio_state(radio.state)

    def switch(self, kind: RadioKind, *, on: bool) -> RadioState:
        radio = self._find(kind)
        if radio is None:
            raise ActionUnavailable(
                f"no {kind} radio present",
                user_message=f"{kind.title_ru}-адаптер не найден",
            )
        target = self._radios.RadioState.ON if on else self._radios.RadioState.OFF
        status = str(_await(radio.set_state_async(target))).lower()
        if "allowed" not in status:
            raise ActionError(
                f"set_state_async answered {status}",
                user_message=f"Windows не дала переключить {kind.title_ru}",
            )
        return _radio_state(radio.state)

    def devices(self) -> tuple[BluetoothDevice, ...]:
        """Paired Bluetooth devices, classic and LE, with their connection state.

        ``System.Devices.Aep.IsConnected`` has to be asked for explicitly —
        association endpoints do not carry it by default — and a device that does
        not report it is listed as paired-but-not-connected rather than skipped. A
        headset missing from the list would read as «не сопряжено», which is worse
        than an uncertain status line.
        """
        self._load()
        if self._enumeration is None:
            return ()
        wanted = ["System.Devices.Aep.IsConnected", "System.Devices.Aep.IsPaired"]
        found: list[BluetoothDevice] = []
        for protocol in self._PROTOCOLS:
            query = (
                f'System.Devices.Aep.ProtocolId:="{protocol}" '
                "AND System.Devices.Aep.IsPaired:=System.StructuredQueryType.Boolean#True"
            )
            try:
                infos = _await(
                    self._enumeration.DeviceInformation.find_all_async(
                        query,
                        wanted,
                        self._enumeration.DeviceInformationKind.ASSOCIATION_ENDPOINT,
                    )
                )
            except (OSError, RuntimeError, AttributeError) as exc:
                _log.warning("не удалось перечислить Bluetooth-устройства: %s", exc)
                continue
            kind = "Bluetooth" if protocol == self._PROTOCOLS[0] else "Bluetooth LE"
            for info in infos or ():
                found.append(
                    BluetoothDevice(
                        name=str(info.name or ""),
                        kind=kind,
                        connected=bool(_prop(info, "System.Devices.Aep.IsConnected")),
                    )
                )
        return tuple(found)


def _prop(info: Any, name: str) -> Any:
    """One WinRT device property, ``None`` when it was not returned."""
    try:
        return info.properties[name]
    except (KeyError, AttributeError, TypeError):
        return None


def _radio_state(value: Any) -> RadioState:
    """Map a WinRT ``RadioState`` onto ours by name, not by number.

    By name because the numbers are an ABI detail and the projections spell the
    value differently — ``RadioState.ON``, ``RadioState_On``, ``2`` — while the
    last component of the string is stable across both.
    """
    name = str(value).rsplit(".", 1)[-1].strip("_").casefold()
    return _WINRT_STATES.get(name, RadioState.UNKNOWN)


_WINRT_STATES: Final[dict[str, RadioState]] = {
    "on": RadioState.ON,
    "off": RadioState.OFF,
    "disabled": RadioState.DISABLED,
    "unknown": RadioState.UNKNOWN,
}


def _await(operation: Any) -> Any:
    """Drive one WinRT ``IAsyncOperation`` to its result on this thread.

    Actions run on a worker thread with no event loop of its own, and the WinRT
    projections expose awaitables rather than blocking calls, so a loop is created
    for the duration of the one call. ``get_results()`` is tried first: when the
    operation has already completed — which switching a radio usually has by the
    time it returns — that skips the loop entirely.
    """
    import asyncio

    try:
        if int(getattr(operation, "status", -1)) == 1:  # AsyncStatus.COMPLETED
            return operation.get_results()
    except (AttributeError, TypeError, ValueError, OSError):
        pass

    async def _run() -> Any:
        return await operation

    return asyncio.run(_run())


_radio: RadioBackend | None = None


def get_radio() -> RadioBackend:
    """The radio backend in force, detected on first use.

    Real WinRT when a projection is importable, :class:`UnavailableRadio`
    otherwise. Detection is deferred rather than done at import: importing a WinRT
    projection costs real time, and a profile that never says «включи блютус»
    should not pay it.
    """
    global _radio
    if _radio is None:
        candidate = WinRtRadio() if sys.platform == "win32" else UnavailableRadio()
        _radio = candidate if candidate.available else UnavailableRadio()
    return _radio


def set_radio(backend: RadioBackend | None) -> None:
    """Install a backend, or forget the detected one with ``None``."""
    global _radio
    _radio = backend


# --------------------------------------------------------------------------- #
# reading the wireless state through netsh
# --------------------------------------------------------------------------- #


def wlan_interfaces() -> tuple[WlanInterface, ...]:
    """Adapters ``netsh`` knows about. Empty tuple when there are none."""
    return parse_wlan_interfaces(get_netsh().run(("wlan", "show", "interfaces")).text)


def require_interface() -> WlanInterface:
    """The wireless adapter to work with, refusing in Russian when there is none.

    The first adapter, because a machine with two is rare and picking between them
    by name is a setting nobody has asked for yet.
    """
    interfaces = wlan_interfaces()
    if not interfaces:
        raise ActionUnavailable(
            "no wireless interface reported by netsh",
            user_message="Беспроводной адаптер не найден — Wi-Fi на этом компьютере недоступен",
        )
    return interfaces[0]


def saved_profiles() -> tuple[str, ...]:
    """Saved Wi-Fi profiles, newest ordering as ``netsh`` gives them."""
    return parse_wlan_profiles(get_netsh().run(("wlan", "show", "profiles")).text)


def scan_networks(*, limit: int = 0) -> tuple[WifiNetwork, ...]:
    """Networks in range, strongest first, with the saved-profile flag filled in.

    Sorted by signal because that is the order a person picks from, and a scan
    lists whatever the driver returns in whatever order it cached it.
    """
    output = get_netsh().run(("wlan", "show", "networks", "mode=bssid")).text
    profiles = {name.casefold() for name in saved_profiles()}
    networks = [
        _with_saved(network, network.ssid.casefold() in profiles)
        for network in parse_wlan_networks(output)
    ]
    networks.sort(key=lambda network: (-network.signal, network.ssid.casefold()))
    return tuple(networks[:limit] if limit else networks)


def _with_saved(network: WifiNetwork, saved: bool) -> WifiNetwork:
    return WifiNetwork(
        ssid=network.ssid,
        security=network.security,
        enterprise=network.enterprise,
        signal=network.signal,
        channel=network.channel,
        radio=network.radio,
        bssid=network.bssid,
        saved=saved,
    )


# --------------------------------------------------------------------------- #
# switching a radio
# --------------------------------------------------------------------------- #


def switch_radio(kind: RadioKind, mode: RadioMode) -> ActionResult[RadioState]:
    """Switch one radio, through WinRT when it is there and ``netsh`` when it is not.

    Reports «уже включён» without touching anything when the radio is already the
    way it was asked for, which matters for ``toggle`` only in the log — but for
    ``on`` it is the difference between a no-op and a needless reconnect that drops
    every socket on the machine.
    """
    radio = get_radio()
    current = radio.state(kind)
    if current is RadioState.DISABLED:
        return ActionResult.failed(
            f"{kind.title_ru} отключён в системе — включите адаптер в диспетчере устройств",
            detail=f"{kind} radio state is Disabled",
            value=current,
        )
    target = mode.target(current)
    if current is not RadioState.UNKNOWN and (current is RadioState.ON) == target:
        return ActionResult.done(
            f"{kind.title_ru} и так {current.title_ru}",
            value=current,
            data={"radio": kind.value, "state": current.value, "changed": False},
        )
    if radio.available:
        state = radio.switch(kind, on=target)
    elif kind is RadioKind.WIFI:
        state = _switch_wifi_via_netsh(on=target)
    else:
        raise ActionUnavailable(
            "bluetooth needs WinRT",
            user_message=(
                "Bluetooth нельзя переключить без WinRT — "
                "netsh этого не умеет, установите пакет winrt"
            ),
        )
    return ActionResult.done(
        f"{kind.title_ru} {state.title_ru}",
        value=state,
        data={"radio": kind.value, "state": state.value, "changed": True},
    )


def _switch_wifi_via_netsh(*, on: bool) -> RadioState:
    """Fallback: enable or disable the adapter itself, borrowing rights for it.

    Coarser than the Radio API on purpose, because it is all ``netsh`` has —
    there is no ``netsh wlan set radio``. Disabling the adapter unbinds it rather
    than parking the radio, so Windows forgets which network it was on; the
    docstring says so because the difference surfaces later, when «включи вайфай»
    comes back to an adapter that does not reconnect on its own.

    The command needs administrator rights, and they are borrowed for that one
    command through :func:`ayris.utils.admin.run_elevated` instead of the
    assistant running elevated.
    """
    from ayris.utils import admin

    interface = require_interface()
    state = "enabled" if on else "disabled"
    arguments = (
        "interface",
        "set",
        "interface",
        f'name="{interface.name}"',
        f"admin={state}",
    )
    if admin.is_elevated():
        result = get_netsh().run(arguments)
        if not result.ok:
            raise ActionError(
                f"netsh {' '.join(arguments)} exited {result.returncode}",
                user_message=f"Не удалось {'включить' if on else 'выключить'} Wi-Fi адаптер",
            )
    else:
        try:
            run = admin.run_elevated("netsh", arguments)
        except admin.ElevationDeclined as exc:
            raise ActionError(
                "user declined UAC for netsh interface set",
                user_message="Без прав администратора адаптер Wi-Fi не переключить",
            ) from exc
        except admin.ElevationUnavailable as exc:
            raise ActionUnavailable(
                f"cannot elevate: {exc}",
                user_message="Нет способа получить права администратора для переключения адаптера",
            ) from exc
        if run.exit_code not in (0, None):
            raise ActionError(
                f"elevated netsh exited {run.exit_code}",
                user_message=f"Не удалось {'включить' if on else 'выключить'} Wi-Fi адаптер",
            )
    return RadioState.ON if on else RadioState.OFF


# --------------------------------------------------------------------------- #
# connecting
# --------------------------------------------------------------------------- #


def _store_password(ssid: str, password: str) -> str:
    """Put a Wi-Fi password in the credential store, returning its reference.

    The only place a password is written. It never reaches ``config.toml``, the
    log or the ``audit`` table: the parameter is marked secret so the registry
    records ``***``, and what is returned here is the reference, which is a hash of
    the SSID and reveals nothing.
    """
    ref = secret_ref_for_ssid(ssid)
    try:
        get_secrets().save(ref, password)
    except SecretsError as exc:
        _log.warning("пароль сети не сохранён в хранилище: %s", exc)
        return ""
    return ref


def _stored_password(ssid: str) -> str:
    """Password saved earlier for ``ssid``, or ``""``."""
    try:
        return get_secrets().get(secret_ref_for_ssid(ssid)) or ""
    except SecretsError as exc:
        _log.warning("не удалось прочитать пароль сети: %s", exc)
        return ""


def _add_profile(ssid: str, password: str, security: WifiSecurity) -> None:
    """Hand ``netsh`` a profile for ``ssid``, then delete the file it came in.

    The profile format carries the key in clear text, so the file lives in a
    temporary directory for the length of one ``netsh`` call and is removed in a
    ``finally`` — including when ``netsh`` fails, which is the case that would
    otherwise leave a password on disk.
    """
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp(prefix="ayris-wlan-"))
    path = directory / "profile.xml"
    try:
        path.write_text(profile_xml(ssid, password, security=security), encoding="utf-8")
        result = get_netsh().run(("wlan", "add", "profile", f"filename={path}", "user=current"))
        if not result.ok:
            raise ActionError(
                f"netsh wlan add profile exited {result.returncode}",
                user_message=f"Windows не приняла профиль сети «{ssid}»",
            )
    finally:
        path.unlink(missing_ok=True)
        directory.rmdir()


def connect_wifi(
    ssid: str, password: str = "", *, security: WifiSecurity | None = None
) -> ActionResult[str]:
    """Join ``ssid``, by saved profile when there is one and by password otherwise.

    The order is deliberate. A saved profile is tried first even when a password
    was supplied, because rewriting a working profile from a mis-heard password is
    how a voice assistant disconnects someone from their own network. A password is
    only turned into a profile when there is no profile to use — or when connecting
    through the existing one failed.
    """
    interface = require_interface()
    profiles = {name.casefold(): name for name in saved_profiles()}
    profile = profiles.get(ssid.casefold())
    stored = password or _stored_password(ssid)

    if profile is not None:
        result = _netsh_connect(profile, ssid, interface.name)
        if result.ok:
            return ActionResult.done(
                f"Подключаюсь к «{ssid}» по сохранённому профилю",
                value=ssid,
                data={"ssid": ssid, "profile": profile, "used_password": False},
            )
        _log.info("профиль «%s» не подошёл, пробую заново", profile)

    if not stored:
        known = _security_of(ssid, security)
        if known.needs_password:
            raise ActionError(
                f"no profile and no password for {ssid!r}",
                user_message=f"Для «{ssid}» нужен пароль — профиля на этом компьютере нет",
            )

    _add_profile(ssid, stored, _security_of(ssid, security))
    result = _netsh_connect(ssid, ssid, interface.name)
    if not result.ok:
        raise ActionError(
            f"netsh wlan connect exited {result.returncode}",
            user_message=f"Не удалось подключиться к «{ssid}»",
        )
    ref = _store_password(ssid, stored) if stored else ""
    return ActionResult.done(
        f"Подключаюсь к «{ssid}»",
        value=ssid,
        data={"ssid": ssid, "profile": ssid, "used_password": bool(stored), "secret_ref": ref},
    )


def _security_of(ssid: str, given: WifiSecurity | None) -> WifiSecurity:
    """Security of ``ssid``: what the caller said, else what a scan says.

    A scan is consulted rather than assuming WPA2 because an open network gets a
    different profile — one with no key element at all — and building a WPA2
    profile for it fails in a way that reads like a wrong password.
    """
    if given is not None:
        return given
    for network in scan_networks():
        if network.ssid.casefold() == ssid.casefold():
            return network.security
    return WifiSecurity.WPA2


def _netsh_connect(profile: str, ssid: str, interface: str) -> NetshResult:
    arguments = ["wlan", "connect", f"name={profile}", f"ssid={ssid}"]
    if interface:
        arguments.append(f"interface={interface}")
    return get_netsh().run(arguments)


# --------------------------------------------------------------------------- #
# the actions
# --------------------------------------------------------------------------- #


@register
class SetWifi(Action):
    """Turn Wi-Fi on, off, or the other way from however it is now.

    Not ``require_admin``: WinRT switches the radio for an unelevated caller, and
    marking the action otherwise would put a UAC prompt in front of «включи
    вайфай» on every machine — including the ones where it is not needed. The one
    path that does need rights is the ``netsh`` fallback, and it asks for them per
    command.
    """

    meta: ClassVar = ActionMeta(
        name="SetWifi",
        category=ActionCategory.SYSTEM,
        title_ru="Wi-Fi",
        description_ru="Включить, выключить или переключить Wi-Fi",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        mode: RadioMode = Field(
            default=RadioMode.TOGGLE,
            description="Включить, выключить или переключить",
        )

    def run(self, params: Params) -> ActionResult[RadioState]:
        return switch_radio(RadioKind.WIFI, params.mode)


@register
class SetBluetooth(Action):
    """Turn Bluetooth on, off, or the other way from however it is now.

    Unlike Wi-Fi this has no ``netsh`` fallback, because ``netsh`` has nothing to
    do with Bluetooth. Without a WinRT projection the action refuses and says
    which package is missing.
    """

    meta: ClassVar = ActionMeta(
        name="SetBluetooth",
        category=ActionCategory.SYSTEM,
        title_ru="Bluetooth",
        description_ru="Включить, выключить или переключить Bluetooth",
        timeout_ms=15_000,
    )

    class Params(ActionParams):
        mode: RadioMode = Field(
            default=RadioMode.TOGGLE,
            description="Включить, выключить или переключить",
        )

    def run(self, params: Params) -> ActionResult[RadioState]:
        return switch_radio(RadioKind.BLUETOOTH, params.mode)


@register
class ListWifi(Action):
    """Networks in range: name, signal, protection, and whether there is a profile.

    ``timeout_ms`` is generous because a scan is slow — the driver sweeps the band
    — and cutting it short returns the stale cache, which is worse than waiting.
    """

    meta: ClassVar = ActionMeta(
        name="ListWifi",
        category=ActionCategory.SYSTEM,
        title_ru="Список сетей Wi-Fi",
        description_ru="Доступные сети: имя, сигнал, тип защиты, сохранён ли профиль",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        limit: int = Field(
            default=10,
            ge=1,
            le=100,
            description="Сколько сетей вернуть, самые сильные первыми",
        )

    def run(self, params: Params) -> ActionResult[tuple[WifiNetwork, ...]]:
        require_interface()
        networks = scan_networks(limit=params.limit)
        if not networks:
            return ActionResult.done(
                "Сетей Wi-Fi не видно",
                value=(),
                data={"networks": [], "count": 0},
            )
        return ActionResult.done(
            _networks_ru(networks),
            value=networks,
            detail="; ".join(network.title_ru for network in networks),
            data={
                "count": len(networks),
                "networks": [
                    {
                        "ssid": network.ssid,
                        "signal": network.signal,
                        "security": network.security.value,
                        "enterprise": network.enterprise,
                        "saved": network.saved,
                        "channel": network.channel,
                    }
                    for network in networks
                ],
            },
        )


def _networks_ru(networks: Sequence[WifiNetwork]) -> str:
    """The scan as one spoken sentence, naming the three strongest.

    Three because a list read aloud stops being useful past that, and the rest are
    in ``value`` for whatever is showing them on screen.
    """
    head = ", ".join(network.title_ru for network in networks[:3])
    if len(networks) <= 3:
        return f"Нашла {len(networks)}: {head}"
    return f"Нашла {len(networks)}, самые сильные: {head}"


@register
class ConnectWifi(Action):
    """Join a network by saved profile, or with a password given once.

    The password parameter is marked secret, which is what keeps it out of the
    ``audit`` table and the log — the registry masks it before recording anything.
    It is stored in the Windows Credential Manager under a reference derived from
    the SSID, so the next «подключись к домашней сети» needs nothing said aloud.
    """

    meta: ClassVar = ActionMeta(
        name="ConnectWifi",
        category=ActionCategory.SYSTEM,
        title_ru="Подключиться к Wi-Fi",
        description_ru="Подключение по сохранённому профилю или с паролем",
        timeout_ms=45_000,
    )

    class Params(ActionParams):
        ssid: str = Field(
            min_length=1,
            max_length=32,
            description="Имя сети",
        )
        password: str = Field(
            default="",
            max_length=63,
            description="Пароль. Не нужен, если профиль уже сохранён",
            json_schema_extra={"secret": True},
        )

    def run(self, params: Params) -> ActionResult[str]:
        return connect_wifi(params.ssid.strip(), params.password)


@register
class ListBluetooth(Action):
    """Paired Bluetooth devices: name, kind, and whether they are connected now."""

    meta: ClassVar = ActionMeta(
        name="ListBluetooth",
        category=ActionCategory.SYSTEM,
        title_ru="Устройства Bluetooth",
        description_ru="Сопряжённые устройства и какие из них подключены",
        timeout_ms=15_000,
    )

    def run(self, params: ActionParams) -> ActionResult[tuple[BluetoothDevice, ...]]:
        del params
        radio = get_radio()
        if not radio.available:
            raise ActionUnavailable(
                "bluetooth enumeration needs WinRT",
                user_message="Список Bluetooth-устройств недоступен без WinRT",
            )
        state = radio.state(RadioKind.BLUETOOTH)
        if state is RadioState.UNKNOWN:
            raise ActionUnavailable(
                "no bluetooth radio",
                user_message="Bluetooth-адаптер не найден",
            )
        devices = radio.devices()
        if not devices:
            message = (
                "Сопряжённых Bluetooth-устройств нет"
                if state is RadioState.ON
                else f"Bluetooth {state.title_ru}, устройств не видно"
            )
            return ActionResult.done(message, value=(), data={"count": 0, "devices": []})
        connected = [device for device in devices if device.connected]
        return ActionResult.done(
            f"Сопряжено {len(devices)}, подключено {len(connected)}",
            value=devices,
            detail="; ".join(device.title_ru for device in devices),
            data={
                "count": len(devices),
                "connected": len(connected),
                "radio": state.value,
                "devices": [
                    {"name": device.name, "kind": device.kind, "connected": device.connected}
                    for device in devices
                ],
            },
        )
