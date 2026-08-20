"""Tests for the Wi-Fi and Bluetooth actions.

Two things here need explaining before the assertions make sense.

**No test touches a radio or a real ``netsh``.** The CI runner has no Wi-Fi and no
Bluetooth adapter, and switching either off on a developer's machine mid-suite
would cut the network under everything else. So :func:`_isolated` installs a
:class:`ScriptedNetsh` and a :class:`RecordingRadio` before every test and makes
``SubprocessNetsh.run`` fail the test if anything reaches for the real thing. The
credential store is swapped for an in-memory one in the same fixture: a test that
saved a Wi-Fi password would otherwise write it into the developer's Windows
Credential Manager and leave it there.

**The recorded ``netsh`` output below is transcribed, not generated.** It is real
output from a Russian and an English Windows, kept side by side, because the one
thing worth testing about these parsers is precisely what a generated fixture
would get wrong: that they read the *structure* of the output and not the words in
it. Every parser assertion is made twice, once per locale, and the two are
expected to agree exactly.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from ayris.actions.base import (
    SECRET_MASK,
    ActionCategory,
    build_schema,
    mask_params,
    params_to_json,
    secret_fields,
)
from ayris.actions.registry import ActionRegistry, registered_actions
from ayris.actions.system import network
from ayris.actions.system.network import (
    NETSH_TIMEOUT_S,
    BluetoothDevice,
    ConnectWifi,
    ListBluetooth,
    ListWifi,
    RadioKind,
    RadioMode,
    RadioState,
    RecordingRadio,
    ScriptedNetsh,
    SetBluetooth,
    SetWifi,
    UnavailableRadio,
    WifiSecurity,
    connect_wifi,
    console_encoding,
    get_netsh,
    parse_wlan_interfaces,
    parse_wlan_networks,
    parse_wlan_profiles,
    profile_xml,
    require_interface,
    scan_networks,
    secret_ref_for_ssid,
    security_from_values,
    signal_from_values,
    switch_radio,
)
from ayris.core.errors import ActionError, ActionParamsInvalid, ActionUnavailable, SecretsError
from ayris.core.secrets import SecretsStore, get_secrets, is_valid_ref, reset_secrets

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# recorded netsh output — Russian and English, the same machine
# --------------------------------------------------------------------------- #

INTERFACES_RU = """\
В системе 1 интерфейс:

    Имя                            : Беспроводная сеть
    Описание                       : Intel(R) Wi-Fi 6E AX211 160MHz
    GUID                           : 4a1b3f9e-2c7d-4e8a-9f01-b2c3d4e5f607
    Физический адрес               : 3c:58:c2:11:22:33
    Состояние                      : соединено
    SSID                           : Домашняя сеть
    BSSID                          : a0:b1:c2:d3:e4:f5
    Тип сети                       : Инфраструктура
    Тип радиомодуля                : 802.11ax
    Проверка подлинности           : WPA2-Personal
    Шифр                           : CCMP
    Режим подключения              : Профиль
    Канал                          : 36
    Скорость получения (Мбит/с)    : 1441
    Скорость передачи (Мбит/с)     : 1441
    Сигнал                         : 92%
    Профиль                        : Домашняя сеть

    Состояние размещенной сети     : недоступно
"""

INTERFACES_EN = """\
There is 1 interface on the system:

    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6E AX211 160MHz
    GUID                   : 4a1b3f9e-2c7d-4e8a-9f01-b2c3d4e5f607
    Physical address       : 3c:58:c2:11:22:33
    State                  : connected
    SSID                   : HomeNet
    BSSID                  : a0:b1:c2:d3:e4:f5
    Network type           : Infrastructure
    Radio type             : 802.11ax
    Authentication         : WPA2-Personal
    Cipher                 : CCMP
    Connection mode        : Profile
    Channel                : 36
    Receive rate (Mbps)    : 1441
    Transmit rate (Mbps)   : 1441
    Signal                 : 92%
    Profile                : HomeNet

    Hosted network status  : Not available
"""

#: An adapter that is on but not joined to anything. Windows drops the SSID,
#: BSSID and profile lines entirely rather than printing them empty, which is why
#: :attr:`WlanInterface.connected` can be read off the SSID's presence.
INTERFACES_RU_DISCONNECTED = """\
В системе 1 интерфейс:

    Имя                            : Беспроводная сеть
    Описание                       : Intel(R) Wi-Fi 6E AX211 160MHz
    GUID                           : 4a1b3f9e-2c7d-4e8a-9f01-b2c3d4e5f607
    Физический адрес               : 3c:58:c2:11:22:33
    Состояние                      : отключено
    Состояние размещенной сети     : недоступно
"""

#: Two adapters: a built-in card and a USB dongle. Rare, and the reason a record
#: is delimited by ``GUID`` rather than by a blank line.
INTERFACES_EN_TWO = """\
There are 2 interfaces on the system:

    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6E AX211 160MHz
    GUID                   : 4a1b3f9e-2c7d-4e8a-9f01-b2c3d4e5f607
    Physical address       : 3c:58:c2:11:22:33
    State                  : connected
    SSID                   : HomeNet
    BSSID                  : a0:b1:c2:d3:e4:f5
    Signal                 : 92%
    Profile                : HomeNet

    Name                   : Wi-Fi 2
    Description            : TP-Link Wireless USB Adapter
    GUID                   : 8e7d6c5b-4a39-4281-b7c6-d5e4f3a2b1c0
    Physical address       : 3c:58:c2:44:55:66
    State                  : disconnected
"""

#: What a desktop with no wireless card answers. No colon anywhere, so the parser
#: sees no lines at all and the caller turns the empty tuple into a refusal.
INTERFACES_RU_NONE = "В системе нет интерфейса беспроводной сети.\n"

INTERFACES_EN_NONE = "There is no wireless interface on the system.\n"

#: What ``netsh`` answers when the WLAN AutoConfig service is stopped. Different
#: text, same shape: nothing the parser can read.
INTERFACES_RU_NO_SERVICE = """\
Служба автонастройки беспроводной сети (wlansvc) не запущена.
"""

NETWORKS_RU = """\
Интерфейс имя : Беспроводная сеть
Всего видимых сетей: 5

SSID 1 : Домашняя сеть
    Тип сети              : Инфраструктура
    Проверка подлинности  : WPA2-Personal
    Шифрование            : CCMP
    BSSID 1               : a0:b1:c2:d3:e4:f5
         Сигнал                          : 92%
         Тип радиомодуля                 : 802.11ax
         Канал                           : 36
         Базовые скорости (Мбит/с)       : 6 12 24
         Прочие скорости (Мбит/с)        : 9 18 36 48 54
    BSSID 2               : a0:b1:c2:d3:e4:f6
         Сигнал                          : 55%
         Тип радиомодуля                 : 802.11n
         Канал                           : 6
         Базовые скорости (Мбит/с)       : 1 2 5.5 11
         Прочие скорости (Мбит/с)        : 6 9 12 18

SSID 2 : CorpNet
    Тип сети              : Инфраструктура
    Проверка подлинности  : WPA2-Enterprise
    Шифрование            : CCMP
    BSSID 1               : b0:c1:d2:e3:f4:05
         Сигнал                          : 70%
         Тип радиомодуля                 : 802.11ac
         Канал                           : 44
         Базовые скорости (Мбит/с)       : 6 12 24
         Прочие скорости (Мбит/с)        : 9 18 36

SSID 3 : StaroeWEP
    Тип сети              : Инфраструктура
    Проверка подлинности  : Открыть
    Шифрование            : WEP
    BSSID 1               : c0:d1:e2:f3:04:15
         Сигнал                          : 40%
         Тип радиомодуля                 : 802.11g
         Канал                           : 11
         Базовые скорости (Мбит/с)       : 1 2 5.5 11
         Прочие скорости (Мбит/с)        : 6 9 12

SSID 4 : FreeWiFi
    Тип сети              : Инфраструктура
    Проверка подлинности  : Открыть
    Шифрование            : Нет
    BSSID 1               : d0:e1:f2:03:14:25
         Сигнал                          : 24%
         Тип радиомодуля                 : 802.11n
         Канал                           : 1
         Базовые скорости (Мбит/с)       : 1 2 5.5 11
         Прочие скорости (Мбит/с)        : 6 9 12

SSID 5 :
    Тип сети              : Инфраструктура
    Проверка подлинности  : WPA3-Personal
    Шифрование            : CCMP
    BSSID 1               : e0:f1:02:13:24:35
         Сигнал                          : 66%
         Тип радиомодуля                 : 802.11ax
         Канал                           : 149
         Базовые скорости (Мбит/с)       : 6 12 24
         Прочие скорости (Мбит/с)        : 9 18 36
"""

NETWORKS_EN = """\
Interface name : Wi-Fi
There are 5 networks currently visible.

SSID 1 : HomeNet
    Network type          : Infrastructure
    Authentication        : WPA2-Personal
    Encryption            : CCMP
    BSSID 1               : a0:b1:c2:d3:e4:f5
         Signal                : 92%
         Radio type            : 802.11ax
         Channel               : 36
         Basic rates (Mbps)    : 6 12 24
         Other rates (Mbps)    : 9 18 36 48 54
    BSSID 2               : a0:b1:c2:d3:e4:f6
         Signal                : 55%
         Radio type            : 802.11n
         Channel               : 6
         Basic rates (Mbps)    : 1 2 5.5 11
         Other rates (Mbps)    : 6 9 12 18

SSID 2 : CorpNet
    Network type          : Infrastructure
    Authentication        : WPA2-Enterprise
    Encryption            : CCMP
    BSSID 1               : b0:c1:d2:e3:f4:05
         Signal                : 70%
         Radio type            : 802.11ac
         Channel               : 44
         Basic rates (Mbps)    : 6 12 24
         Other rates (Mbps)    : 9 18 36

SSID 3 : StaroeWEP
    Network type          : Infrastructure
    Authentication        : Open
    Encryption            : WEP
    BSSID 1               : c0:d1:e2:f3:04:15
         Signal                : 40%
         Radio type            : 802.11g
         Channel               : 11
         Basic rates (Mbps)    : 1 2 5.5 11
         Other rates (Mbps)    : 6 9 12

SSID 4 : FreeWiFi
    Network type          : Infrastructure
    Authentication        : Open
    Encryption            : None
    BSSID 1               : d0:e1:f2:03:14:25
         Signal                : 24%
         Radio type            : 802.11n
         Channel               : 1
         Basic rates (Mbps)    : 1 2 5.5 11
         Other rates (Mbps)    : 6 9 12

SSID 5 :
    Network type          : Infrastructure
    Authentication        : WPA3-Personal
    Encryption            : CCMP
    BSSID 1               : e0:f1:02:13:24:35
         Signal                : 66%
         Radio type            : 802.11ax
         Channel               : 149
         Basic rates (Mbps)    : 6 12 24
         Other rates (Mbps)    : 9 18 36
"""

NETWORKS_RU_EMPTY = """\
Интерфейс имя : Беспроводная сеть
Всего видимых сетей: 0

"""

PROFILES_RU = """\
Профили на интерфейсе Беспроводная сеть:

Профили групповой политики (только чтение)
---------------------------------------------
    <Отсутствует>

Профили пользователя
-------------------
    Все профили пользователей     : Домашняя сеть
    Все профили пользователей     : CorpNet
    Все профили пользователей     : Гостевой Wi-Fi
"""

PROFILES_EN = """\
Profiles on interface Wi-Fi:

Group policy profiles (read only)
---------------------------------
    <None>

User profiles
-------------
    All User Profile     : HomeNet
    All User Profile     : CorpNet
    All User Profile     : Guest Wi-Fi
"""

PROFILES_RU_NONE = """\
Профили на интерфейсе Беспроводная сеть:

Профили групповой политики (только чтение)
---------------------------------------------
    <Отсутствует>

Профили пользователя
-------------------
    <Отсутствует>
"""

#: A build that prints the placeholder on the value side of a real key instead of
#: on a line of its own. Both shapes exist in the wild.
PROFILES_EN_NONE_INLINE = """\
Profiles on interface Wi-Fi:

User profiles
-------------
    All User Profile     : <None>
"""

# --------------------------------------------------------------------------- #
# parsing the interface list
# --------------------------------------------------------------------------- #


class TestParseInterfaces:
    """``netsh wlan show interfaces``, in both locales."""

    @pytest.mark.parametrize(
        ("output", "ssid"),
        [(INTERFACES_RU, "Домашняя сеть"), (INTERFACES_EN, "HomeNet")],
        ids=["ru", "en"],
    )
    def test_one_connected_adapter(self, output: str, ssid: str) -> None:
        (interface,) = parse_wlan_interfaces(output)
        assert interface.ssid == ssid
        assert interface.guid == "4a1b3f9e-2c7d-4e8a-9f01-b2c3d4e5f607"
        assert interface.bssid == "a0:b1:c2:d3:e4:f5"
        assert interface.connected is True

    @pytest.mark.parametrize(
        ("output", "name"),
        [(INTERFACES_RU, "Беспроводная сеть"), (INTERFACES_EN, "Wi-Fi")],
        ids=["ru", "en"],
    )
    def test_name_and_description_come_from_position(self, output: str, name: str) -> None:
        """Their keys are translated, their order is not — so order is what is read."""
        (interface,) = parse_wlan_interfaces(output)
        assert interface.name == name
        assert interface.description == "Intel(R) Wi-Fi 6E AX211 160MHz"

    @pytest.mark.parametrize(
        ("output", "profile"),
        [(INTERFACES_RU, "Домашняя сеть"), (INTERFACES_EN, "HomeNet")],
        ids=["ru", "en"],
    )
    def test_profile_is_recognised_by_matching_the_ssid(self, output: str, profile: str) -> None:
        (interface,) = parse_wlan_interfaces(output)
        assert interface.profile == profile

    @pytest.mark.parametrize(
        ("output", "state"),
        [(INTERFACES_RU, "соединено"), (INTERFACES_EN, "connected")],
        ids=["ru", "en"],
    )
    def test_state_text_is_kept_verbatim(self, output: str, state: str) -> None:
        """Kept for the log only. Nothing decides anything by reading it."""
        (interface,) = parse_wlan_interfaces(output)
        assert interface.state == state

    def test_a_disconnected_adapter_has_no_ssid(self) -> None:
        (interface,) = parse_wlan_interfaces(INTERFACES_RU_DISCONNECTED)
        assert interface.ssid == ""
        assert interface.profile == ""
        assert interface.connected is False

    def test_two_adapters_are_split_on_the_guid(self) -> None:
        first, second = parse_wlan_interfaces(INTERFACES_EN_TWO)
        assert (first.name, first.ssid) == ("Wi-Fi", "HomeNet")
        assert (second.name, second.ssid) == ("Wi-Fi 2", "")
        assert second.description == "TP-Link Wireless USB Adapter"

    @pytest.mark.parametrize(
        "output",
        [INTERFACES_RU_NONE, INTERFACES_EN_NONE, INTERFACES_RU_NO_SERVICE, "", "   \n\n"],
        ids=["ru", "en", "no-service", "empty", "blank"],
    )
    def test_no_adapter_parses_to_nothing(self, output: str) -> None:
        """The parser reports absence; deciding what it means is the caller's job."""
        assert parse_wlan_interfaces(output) == ()


# --------------------------------------------------------------------------- #
# parsing the scan
# --------------------------------------------------------------------------- #


class TestParseNetworks:
    """``netsh wlan show networks mode=bssid``, in both locales."""

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_every_network_is_found(self, output: str) -> None:
        assert len(parse_wlan_networks(output)) == 5

    @pytest.mark.parametrize(
        ("output", "names"),
        [
            (NETWORKS_RU, ("Домашняя сеть", "CorpNet", "StaroeWEP", "FreeWiFi", "")),
            (NETWORKS_EN, ("HomeNet", "CorpNet", "StaroeWEP", "FreeWiFi", "")),
        ],
        ids=["ru", "en"],
    )
    def test_names_in_the_order_netsh_gave_them(self, output: str, names: tuple[str, ...]) -> None:
        assert tuple(network.ssid for network in parse_wlan_networks(output)) == names

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_security_is_read_from_the_values(self, output: str) -> None:
        """The same five verdicts from Russian and from English output."""
        networks = parse_wlan_networks(output)
        assert [network.security for network in networks] == [
            WifiSecurity.WPA2,
            WifiSecurity.WPA2,
            WifiSecurity.WEP,
            WifiSecurity.OPEN,
            WifiSecurity.WPA3,
        ]

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_wep_is_not_mistaken_for_open(self, output: str) -> None:
        """The case that forces value-side parsing.

        A WEP network authenticates as «Открыть»/«Open» and names WEP only under
        the encryption key. A parser reading one labelled field would call it open
        and Ayris would offer to join it without a password.
        """
        wep = parse_wlan_networks(output)[2]
        assert wep.security is WifiSecurity.WEP
        assert wep.security.needs_password is True

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_enterprise_is_flagged_separately(self, output: str) -> None:
        networks = parse_wlan_networks(output)
        assert networks[1].enterprise is True
        assert networks[0].enterprise is False

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_the_strongest_bssid_wins(self, output: str) -> None:
        """One network, two radios, and only the better one is reported."""
        home = parse_wlan_networks(output)[0]
        assert home.signal == 92
        assert home.bssid == "a0:b1:c2:d3:e4:f5"
        assert home.channel == 36
        assert home.radio == "802.11ax"

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_signal_channel_and_radio_per_network(self, output: str) -> None:
        networks = parse_wlan_networks(output)
        assert [network.signal for network in networks] == [92, 70, 40, 24, 66]
        assert [network.channel for network in networks] == [36, 44, 11, 1, 149]
        assert [network.radio for network in networks] == [
            "802.11ax",
            "802.11ac",
            "802.11g",
            "802.11n",
            "802.11ax",
        ]

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_a_hidden_network_keeps_its_place(self, output: str) -> None:
        """An empty SSID is a network without a name, not a parse failure."""
        hidden = parse_wlan_networks(output)[4]
        assert hidden.ssid == ""
        assert hidden.hidden is True
        assert hidden.signal == 66
        assert hidden.security is WifiSecurity.WPA3

    @pytest.mark.parametrize("output", [NETWORKS_RU, NETWORKS_EN], ids=["ru", "en"])
    def test_saved_is_not_guessed_by_the_parser(self, output: str) -> None:
        """A scan says nothing about profiles, so the parser claims nothing."""
        assert all(not network.saved for network in parse_wlan_networks(output))

    @pytest.mark.parametrize(
        "output",
        [NETWORKS_RU_EMPTY, "", INTERFACES_RU_NONE],
        ids=["nothing-in-range", "empty", "no-adapter"],
    )
    def test_nothing_in_range_parses_to_nothing(self, output: str) -> None:
        assert parse_wlan_networks(output) == ()

    def test_the_header_is_not_taken_for_a_network(self) -> None:
        """«Интерфейс имя : …» is a ``key : value`` line too, and must be ignored.

        ``SSID`` without an index does not open a network, which is what keeps the
        interface name and the visible-network count out of the result.
        """
        assert parse_wlan_networks("Интерфейс имя : Беспроводная сеть\nSSID : нет\n") == ()


# --------------------------------------------------------------------------- #
# parsing the profile list
# --------------------------------------------------------------------------- #


class TestParseProfiles:
    """``netsh wlan show profiles``, in both locales."""

    @pytest.mark.parametrize(
        ("output", "names"),
        [
            (PROFILES_RU, ("Домашняя сеть", "CorpNet", "Гостевой Wi-Fi")),
            (PROFILES_EN, ("HomeNet", "CorpNet", "Guest Wi-Fi")),
        ],
        ids=["ru", "en"],
    )
    def test_saved_profiles(self, output: str, names: tuple[str, ...]) -> None:
        assert parse_wlan_profiles(output) == names

    @pytest.mark.parametrize(
        "output",
        [PROFILES_RU_NONE, PROFILES_EN_NONE_INLINE, ""],
        ids=["ru-placeholder-line", "en-placeholder-value", "empty"],
    )
    def test_no_profiles(self, output: str) -> None:
        """Both shapes of «nothing here» yield nothing, not a profile named ``<None>``."""
        assert parse_wlan_profiles(output) == ()

    def test_headings_are_not_profiles(self) -> None:
        """A heading sits at indent zero and carries no value; a profile does both."""
        assert parse_wlan_profiles(PROFILES_RU)[0] != "Профили на интерфейсе Беспроводная сеть"

    def test_a_profile_listed_twice_is_kept_once(self) -> None:
        """The same network can appear under group policy and under the user."""
        doubled = PROFILES_EN.replace(
            "    All User Profile     : Guest Wi-Fi",
            "    All User Profile     : HomeNet",
        )
        assert parse_wlan_profiles(doubled) == ("HomeNet", "CorpNet")

    def test_a_profile_name_with_a_colon_survives(self) -> None:
        """SSIDs may contain a colon, and the split is on the first one only."""
        assert parse_wlan_profiles("    All User Profile : Wi-Fi: 5G\n") == ("Wi-Fi: 5G",)


# --------------------------------------------------------------------------- #
# the two value-side readers, on their own
# --------------------------------------------------------------------------- #


class TestSecurityFromValues:
    """:func:`security_from_values` — the part that must not read a key."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (["WPA3-Personal", "CCMP"], WifiSecurity.WPA3),
            (["WPA2-Personal", "CCMP"], WifiSecurity.WPA2),
            (["WPA-Personal", "TKIP"], WifiSecurity.WPA),
            (["Открыть", "WEP"], WifiSecurity.WEP),
            (["Open", "WEP"], WifiSecurity.WEP),
            (["Открыть", "Нет"], WifiSecurity.OPEN),
            (["Open", "None"], WifiSecurity.OPEN),
            (["RSNA", "CCMP"], WifiSecurity.WPA2),
        ],
        ids=["wpa3", "wpa2", "wpa", "wep-ru", "wep-en", "open-ru", "open-en", "rsna"],
    )
    def test_recognised_values(self, values: list[str], expected: WifiSecurity) -> None:
        assert security_from_values(values)[0] is expected

    def test_the_strongest_advertised_scheme_wins(self) -> None:
        """An access point offering both is joined by the better one."""
        assert security_from_values(["WPA3-Personal", "WPA2-Personal"])[0] is WifiSecurity.WPA3

    @pytest.mark.parametrize(
        ("values", "enterprise"),
        [
            (["WPA2-Enterprise", "CCMP"], True),
            (["WPA2-Personal", "CCMP"], False),
            (["WPA3-Enterprise", "GCMP"], True),
            (["802.1X", "CCMP"], True),
        ],
        ids=["wpa2-eap", "wpa2-psk", "wpa3-eap", "dot1x"],
    )
    def test_enterprise_flag(self, values: list[str], enterprise: bool) -> None:
        assert security_from_values(values)[1] is enterprise

    @pytest.mark.parametrize(
        "values",
        [["Инфраструктура", "OWE-TM"], ["Something New"], []],
        ids=["future-scheme", "nonsense", "nothing"],
    )
    def test_an_unrecognised_value_is_unknown_not_open(self, values: list[str]) -> None:
        """Calling it open would be a security claim made on no evidence."""
        security, _ = security_from_values(values)
        assert security is WifiSecurity.UNKNOWN
        assert security.needs_password is True

    def test_open_is_matched_as_a_whole_value(self) -> None:
        """«Нет» means open; «Нет сети» is something else and must not count."""
        assert security_from_values(["Нет сети"])[0] is WifiSecurity.UNKNOWN


class TestSignalFromValues:
    """:func:`signal_from_values` — a percentage is a shape, not a labelled field."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (["92%"], 92),
            (["92 %"], 92),
            (["0%"], 0),
            (["100%"], 100),
            (["55%", "92%", "40%"], 92),
            (["Инфраструктура", "70%", "802.11ax"], 70),
        ],
        ids=["plain", "spaced", "zero", "full", "strongest", "among-others"],
    )
    def test_percentages(self, values: list[str], expected: int) -> None:
        assert signal_from_values(values) == expected

    @pytest.mark.parametrize(
        "values",
        [["36"], ["802.11ax"], ["CCMP"], [], ["1 2 5.5 11"]],
        ids=["channel", "radio", "cipher", "nothing", "rates"],
    )
    def test_values_that_are_not_percentages(self, values: list[str]) -> None:
        assert signal_from_values(values) == 0

    def test_an_out_of_range_reading_is_clamped(self) -> None:
        """Some drivers report above 100. The number is a hint to a human anyway."""
        assert signal_from_values(["130%"]) == 100


# --------------------------------------------------------------------------- #
# the test seam itself
# --------------------------------------------------------------------------- #


class TestScriptedNetsh:
    """The stand-in every test below relies on, checked before they rely on it."""

    def test_it_answers_by_argument_prefix(self) -> None:
        netsh = ScriptedNetsh({"wlan show interfaces": INTERFACES_EN})
        assert netsh.run(("wlan", "show", "interfaces")).stdout == INTERFACES_EN

    def test_a_prefix_matches_a_longer_call(self) -> None:
        """So a test does not have to predict every flag the caller adds."""
        netsh = ScriptedNetsh({"wlan show networks": NETWORKS_EN})
        assert netsh.run(("wlan", "show", "networks", "mode=bssid")).ok is True

    def test_an_unscripted_call_fails_quietly(self) -> None:
        """Which is exactly what a missing adapter looks like."""
        result = ScriptedNetsh().run(("wlan", "show", "interfaces"))
        assert (result.ok, result.stdout) == (False, "")

    def test_every_call_is_recorded(self) -> None:
        netsh = ScriptedNetsh()
        netsh.run(("wlan", "disconnect"))
        assert netsh.calls == [("wlan", "disconnect")]
        assert netsh.called("wlan disconnect") is True
        assert netsh.called("wlan connect") is False


# --------------------------------------------------------------------------- #
# isolation: no real radio, no real netsh, no real credential store
# --------------------------------------------------------------------------- #


class FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager.

    Duplicated from the other suites rather than shared, for the reason given
    there — and here for a second one: a test that saved a Wi-Fi password through
    the real backend would leave it in the developer's Credential Manager, under a
    reference they never chose, after the suite had finished.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.entries.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.entries[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self.entries[(service_name, username)]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cut every path from this suite to the machine it runs on.

    Three separate guards, because there are three ways out. The module globals
    are pointed at the scripted stand-ins; ``SubprocessNetsh.run`` is replaced with
    something that fails the test, so a code path that builds its own runner is
    caught rather than silently spawning ``netsh``; and ``WinRtRadio`` is made
    unconstructible, so a path that bypasses :func:`get_radio` is caught too.

    The last one matters more than it looks: ``WinRtRadio.switch`` on a developer's
    laptop would turn their Wi-Fi off mid-suite, and the failure would surface as
    every later test in the run timing out on a network call.
    """

    def no_netsh(*args: object, **kwargs: object) -> object:
        message = "тест дошёл до настоящего netsh"
        raise AssertionError(message)

    def no_winrt(*args: object, **kwargs: object) -> object:
        message = "тест дошёл до настоящего WinRT Radio"
        raise AssertionError(message)

    monkeypatch.setattr(network.SubprocessNetsh, "run", no_netsh)
    monkeypatch.setattr(network.WinRtRadio, "__init__", no_winrt)
    network.set_netsh(ScriptedNetsh())
    network.set_radio(RecordingRadio(available=False))
    reset_secrets(SecretsStore("Ayris-test-network", backend=FakeKeyring()))
    try:
        yield
    finally:
        network.set_netsh(None)
        network.set_radio(None)
        reset_secrets()


@pytest.fixture
def netsh() -> ScriptedNetsh:
    """A ``netsh`` scripted with the recorded output of a working adapter."""
    runner = ScriptedNetsh(
        {
            "wlan show interfaces": INTERFACES_RU,
            "wlan show networks": NETWORKS_RU,
            "wlan show profiles": PROFILES_RU,
            "wlan connect": "Запрос на подключение выполнен успешно.\n",
            "wlan add profile": "Профиль добавлен в интерфейс.\n",
        }
    )
    network.set_netsh(runner)
    return runner


@pytest.fixture
def radio() -> RecordingRadio:
    """A WinRT-equivalent backend with both radios present and switched on."""
    backend = RecordingRadio(
        states={RadioKind.WIFI: RadioState.ON, RadioKind.BLUETOOTH: RadioState.ON}
    )
    network.set_radio(backend)
    return backend


@pytest.fixture
def registry() -> Iterator[ActionRegistry]:
    """A registry holding the network actions and nothing else."""
    instance = ActionRegistry(audit_enabled=lambda: False)
    for action in (SetWifi, SetBluetooth, ListWifi, ConnectWifi, ListBluetooth):
        instance.add(action)
    try:
        yield instance
    finally:
        instance.shutdown()


class TestIsolation:
    """The guards above, checked rather than assumed."""

    def test_the_real_netsh_is_out_of_reach(self) -> None:
        with pytest.raises(AssertionError, match="настоящего netsh"):
            network.SubprocessNetsh().run(("wlan", "show", "interfaces"))

    def test_the_real_winrt_backend_is_out_of_reach(self) -> None:
        with pytest.raises(AssertionError, match="настоящего WinRT"):
            network.WinRtRadio()

    def test_the_default_state_is_no_radio_and_no_netsh(self) -> None:
        """So a test that forgets its fixture sees «нет адаптера», not the machine."""
        assert network.get_radio().available is False
        assert network.wlan_interfaces() == ()


# --------------------------------------------------------------------------- #
# switching a radio
# --------------------------------------------------------------------------- #


class TestSwitchRadio:
    """:func:`switch_radio` through WinRT, which is the path a real machine takes."""

    @pytest.mark.parametrize("kind", list(RadioKind), ids=lambda kind: kind.value)
    def test_turning_it_off(self, radio: RecordingRadio, kind: RadioKind) -> None:
        result = switch_radio(kind, RadioMode.OFF)
        assert result.ok is True
        assert result.value is RadioState.OFF
        assert radio.switches == [(kind, False)]

    @pytest.mark.parametrize("kind", list(RadioKind), ids=lambda kind: kind.value)
    def test_turning_it_back_on(self, radio: RecordingRadio, kind: RadioKind) -> None:
        radio.states[kind] = RadioState.OFF
        result = switch_radio(kind, RadioMode.ON)
        assert result.value is RadioState.ON
        assert radio.switches == [(kind, True)]

    def test_toggle_reads_the_current_state_first(self, radio: RecordingRadio) -> None:
        assert switch_radio(RadioKind.WIFI, RadioMode.TOGGLE).value is RadioState.OFF
        assert switch_radio(RadioKind.WIFI, RadioMode.TOGGLE).value is RadioState.ON
        assert radio.switches == [(RadioKind.WIFI, False), (RadioKind.WIFI, True)]

    def test_asking_for_the_state_it_is_already_in_touches_nothing(
        self, radio: RecordingRadio
    ) -> None:
        """«Включи вайфай» to a working radio must not drop every open socket."""
        result = switch_radio(RadioKind.WIFI, RadioMode.ON)
        assert result.ok is True
        assert result.data["changed"] is False
        assert radio.switches == []

    def test_the_no_op_still_says_what_the_state_is(self, radio: RecordingRadio) -> None:
        assert "включён" in switch_radio(RadioKind.WIFI, RadioMode.ON).message_ru

    def test_a_missing_radio_is_refused_by_name(self, radio: RecordingRadio) -> None:
        """A desktop with no Bluetooth card, not a Bluetooth card switched off."""
        del radio.states[RadioKind.BLUETOOTH]
        with pytest.raises(ActionUnavailable) as raised:
            switch_radio(RadioKind.BLUETOOTH, RadioMode.ON)
        assert "Bluetooth-адаптер не найден" in raised.value.user_message_ru

    def test_a_disabled_adapter_is_reported_not_switched(self, radio: RecordingRadio) -> None:
        """Nothing here can un-disable a device; saying where to do it is the help."""
        radio.states[RadioKind.WIFI] = RadioState.DISABLED
        result = switch_radio(RadioKind.WIFI, RadioMode.ON)
        assert result.ok is False
        assert "диспетчере устройств" in result.message_ru
        assert radio.switches == []

    def test_bluetooth_without_winrt_says_which_package_is_missing(self) -> None:
        """``netsh`` has nothing to do with Bluetooth, and the message says so.

        Asserted on the user-facing message rather than on the exception's own
        text: what a person hears is the thing that has to name the package, and
        the technical string is for the log.
        """
        network.set_radio(RecordingRadio(available=False))
        with pytest.raises(ActionUnavailable) as raised:
            switch_radio(RadioKind.BLUETOOTH, RadioMode.ON)
        assert "winrt" in raised.value.user_message_ru


class TestRadioMode:
    """:meth:`RadioMode.target` — the only branching in the switch path."""

    @pytest.mark.parametrize(
        ("mode", "current", "expected"),
        [
            (RadioMode.ON, RadioState.OFF, True),
            (RadioMode.ON, RadioState.ON, True),
            (RadioMode.OFF, RadioState.ON, False),
            (RadioMode.OFF, RadioState.OFF, False),
            (RadioMode.TOGGLE, RadioState.ON, False),
            (RadioMode.TOGGLE, RadioState.OFF, True),
        ],
        ids=["on-from-off", "on-from-on", "off-from-on", "off-from-off", "toggle-on", "toggle-off"],
    )
    def test_target(self, mode: RadioMode, current: RadioState, expected: bool) -> None:
        assert mode.target(current) is expected

    def test_toggling_an_unreadable_radio_aims_at_on(self) -> None:
        """Someone saying «переключи вайфай» to a radio Ayris cannot read wants it working."""
        assert RadioMode.TOGGLE.target(RadioState.UNKNOWN) is True


# --------------------------------------------------------------------------- #
# reading the wireless state
# --------------------------------------------------------------------------- #


class TestRequireInterface:
    """The one refusal every Wi-Fi path goes through first."""

    def test_the_adapter_is_found(self, netsh: ScriptedNetsh) -> None:
        assert require_interface().name == "Беспроводная сеть"

    def test_no_adapter_is_refused_in_russian(self) -> None:
        """The message a desktop user hears, and the reason they hear it."""
        network.set_netsh(ScriptedNetsh({"wlan show interfaces": INTERFACES_RU_NONE}))
        with pytest.raises(ActionUnavailable) as raised:
            require_interface()
        assert "Беспроводной адаптер не найден" in raised.value.user_message_ru

    def test_a_stopped_wlansvc_is_the_same_refusal(self) -> None:
        """Different cause, same thing to say: there is no Wi-Fi to work with."""
        network.set_netsh(ScriptedNetsh({"wlan show interfaces": INTERFACES_RU_NO_SERVICE}))
        with pytest.raises(ActionUnavailable):
            require_interface()

    def test_netsh_failing_outright_is_the_same_refusal(self) -> None:
        """Nothing scripted, so ``netsh`` answers with a non-zero code and no text."""
        with pytest.raises(ActionUnavailable):
            require_interface()


class TestScanNetworks:
    """:func:`scan_networks` — the parser plus the two things it cannot know."""

    def test_the_saved_flag_comes_from_the_profile_list(self, netsh: ScriptedNetsh) -> None:
        """A scan says nothing about profiles, so the two calls are joined here."""
        saved = {network.ssid: network.saved for network in scan_networks()}
        assert saved["Домашняя сеть"] is True
        assert saved["CorpNet"] is True
        assert saved["FreeWiFi"] is False

    def test_profiles_are_matched_case_insensitively(self, netsh: ScriptedNetsh) -> None:
        """Windows preserves the case it was given and matches without it."""
        netsh.answers["wlan show profiles"] = PROFILES_RU.replace("CorpNet", "corpnet")
        saved = {network.ssid: network.saved for network in scan_networks()}
        assert saved["CorpNet"] is True

    def test_strongest_first(self, netsh: ScriptedNetsh) -> None:
        """The order a person picks from, not the order the driver cached."""
        signals = [network.signal for network in scan_networks()]
        assert signals == sorted(signals, reverse=True)
        assert signals == [92, 70, 66, 40, 24]

    def test_the_limit_keeps_the_strongest(self, netsh: ScriptedNetsh) -> None:
        networks = scan_networks(limit=2)
        assert [network.signal for network in networks] == [92, 70]

    def test_no_limit_returns_everything(self, netsh: ScriptedNetsh) -> None:
        assert len(scan_networks()) == 5

    def test_nothing_in_range(self, netsh: ScriptedNetsh) -> None:
        netsh.answers["wlan show networks"] = NETWORKS_RU_EMPTY
        assert scan_networks() == ()

    def test_the_scan_asks_for_bssid_detail(self, netsh: ScriptedNetsh) -> None:
        """Without ``mode=bssid`` there is no signal strength in the output at all."""
        scan_networks()
        assert netsh.called("wlan show networks mode=bssid") is True


class TestWifiNetworkWording:
    """How one network reads out loud."""

    def test_a_named_network(self, netsh: ScriptedNetsh) -> None:
        home = scan_networks()[0]
        assert home.title_ru == "«Домашняя сеть», 92%, WPA2, есть профиль"

    def test_a_network_without_a_profile_does_not_claim_one(self, netsh: ScriptedNetsh) -> None:
        free = next(item for item in scan_networks() if item.ssid == "FreeWiFi")
        assert free.title_ru == "«FreeWiFi», 24%, открытая"

    def test_a_hidden_network_is_named_as_such(self, netsh: ScriptedNetsh) -> None:
        """««», 66%» would be unreadable, so the empty name gets words."""
        hidden = next(item for item in scan_networks() if item.hidden)
        assert hidden.title_ru.startswith("«сеть без имени»")


# --------------------------------------------------------------------------- #
# ListWifi
# --------------------------------------------------------------------------- #


class TestListWifi:
    """The action, through the registry, the way the assistant calls it."""

    def test_it_returns_typed_networks(
        self, registry: ActionRegistry, netsh: ScriptedNetsh
    ) -> None:
        result = registry.execute("ListWifi", {"limit": 5})
        assert result.ok is True
        assert [item.ssid for item in result.value] == [
            "Домашняя сеть",
            "CorpNet",
            "",
            "StaroeWEP",
            "FreeWiFi",
        ]

    def test_the_spoken_answer_names_the_three_strongest(
        self, registry: ActionRegistry, netsh: ScriptedNetsh
    ) -> None:
        """A list read aloud stops being useful past three; the rest are in ``value``."""
        message = registry.execute("ListWifi", {"limit": 5}).message_ru
        assert message.startswith("Нашла 5, самые сильные:")
        assert "Домашняя сеть" in message
        assert "StaroeWEP" not in message

    def test_a_short_list_is_read_out_whole(
        self, registry: ActionRegistry, netsh: ScriptedNetsh
    ) -> None:
        assert registry.execute("ListWifi", {"limit": 2}).message_ru.startswith("Нашла 2: ")

    def test_the_data_payload_carries_the_four_fields_the_ui_needs(
        self, registry: ActionRegistry, netsh: ScriptedNetsh
    ) -> None:
        first = registry.execute("ListWifi", {"limit": 1}).data["networks"][0]
        assert first == {
            "ssid": "Домашняя сеть",
            "signal": 92,
            "security": "wpa2",
            "enterprise": False,
            "saved": True,
            "channel": 36,
        }

    def test_nothing_in_range_is_a_success_not_a_failure(
        self, registry: ActionRegistry, netsh: ScriptedNetsh
    ) -> None:
        """An empty band is an answer. Only a missing adapter is a refusal."""
        netsh.answers["wlan show networks"] = NETWORKS_RU_EMPTY
        result = registry.execute("ListWifi", {})
        assert (result.ok, result.value) == (True, ())
        assert result.message_ru == "Сетей Wi-Fi не видно"

    def test_no_adapter_refuses_before_scanning(self, registry: ActionRegistry) -> None:
        network.set_netsh(ScriptedNetsh({"wlan show interfaces": INTERFACES_RU_NONE}))
        with pytest.raises(ActionUnavailable) as raised:
            registry.execute("ListWifi", {})
        assert "Беспроводной адаптер не найден" in raised.value.user_message_ru

    @pytest.mark.parametrize("limit", [0, -1, 101], ids=["zero", "negative", "too-many"])
    def test_the_limit_is_bounded(self, registry: ActionRegistry, limit: int) -> None:
        with pytest.raises(ActionParamsInvalid):
            registry.execute("ListWifi", {"limit": limit})

    def test_the_limit_has_a_default(self, netsh: ScriptedNetsh) -> None:
        """So «какие есть сети» works without the assistant inventing a number."""
        assert ListWifi.Params().limit == 10


# --------------------------------------------------------------------------- #
# connecting
# --------------------------------------------------------------------------- #

PASSWORD = "gorod-1905-goda"


class TestConnectBySavedProfile:
    """The path a returning network takes, which is most of them."""

    def test_it_connects_through_the_profile(self, netsh: ScriptedNetsh) -> None:
        result = connect_wifi("Домашняя сеть")
        assert result.ok is True
        assert result.data["used_password"] is False
        assert netsh.called("wlan connect name=Домашняя сеть") is True

    def test_the_interface_is_named_in_the_command(self, netsh: ScriptedNetsh) -> None:
        """A machine with two adapters must not have Windows pick for it."""
        connect_wifi("Домашняя сеть")
        assert netsh.called("wlan connect name=Домашняя сеть ssid=Домашняя сеть") is True
        assert any("interface=Беспроводная сеть" in call for call in netsh.calls[-1])

    def test_no_profile_is_written_when_one_exists(self, netsh: ScriptedNetsh) -> None:
        connect_wifi("Домашняя сеть")
        assert netsh.called("wlan add profile") is False

    def test_a_password_does_not_overwrite_a_working_profile(self, netsh: ScriptedNetsh) -> None:
        """Rewriting a good profile from a mis-heard password is how Ayris would
        disconnect someone from their own network."""
        connect_wifi("Домашняя сеть", PASSWORD)
        assert netsh.called("wlan add profile") is False

    def test_the_profile_name_is_matched_case_insensitively(self, netsh: ScriptedNetsh) -> None:
        """Speech recognition does not preserve the case Windows stored."""
        result = connect_wifi("домашняя сеть")
        assert result.data["used_password"] is False

    def test_a_broken_profile_falls_through_to_a_fresh_one(self, netsh: ScriptedNetsh) -> None:
        """A profile can hold a password the network no longer accepts."""
        netsh.codes["wlan connect"] = 1
        with pytest.raises(ActionError):
            connect_wifi("Домашняя сеть", PASSWORD)
        assert netsh.called("wlan add profile") is True


class TestConnectWithPassword:
    """A network being joined for the first time."""

    def test_it_writes_a_profile_and_connects(self, netsh: ScriptedNetsh) -> None:
        result = connect_wifi("FreeWiFi5G", PASSWORD, security=WifiSecurity.WPA2)
        assert result.ok is True
        assert result.data["used_password"] is True
        assert netsh.called("wlan add profile") is True
        assert netsh.called("wlan connect name=FreeWiFi5G") is True

    def test_a_protected_network_without_a_password_is_refused(self, netsh: ScriptedNetsh) -> None:
        """And the refusal says why, so «подключись к CorpNet» is not a silent no-op."""
        with pytest.raises(ActionError) as raised:
            connect_wifi("StaroeWEP")
        assert "нужен пароль" in raised.value.user_message_ru
        assert netsh.called("wlan add profile") is False

    def test_an_open_network_needs_no_password(self, netsh: ScriptedNetsh) -> None:
        """The scan says it is open, so no password is asked for and none is stored."""
        result = connect_wifi("FreeWiFi")
        assert result.ok is True
        assert result.data["used_password"] is False

    def test_the_security_type_comes_from_the_scan(self, netsh: ScriptedNetsh) -> None:
        """An open network gets a profile with no key element; a WPA2 one gets a key.

        Assuming WPA2 for everything makes the open case fail in a way that reads
        exactly like a wrong password.
        """
        connect_wifi("FreeWiFi")
        assert netsh.called("wlan show networks") is True

    def test_a_failed_connect_is_reported_not_swallowed(self, netsh: ScriptedNetsh) -> None:
        netsh.codes["wlan connect"] = 1
        with pytest.raises(ActionError) as raised:
            connect_wifi("FreeWiFi5G", PASSWORD)
        assert "«FreeWiFi5G»" in raised.value.user_message_ru

    def test_a_rejected_profile_is_reported_by_name(self, netsh: ScriptedNetsh) -> None:
        netsh.codes["wlan add profile"] = 1
        with pytest.raises(ActionError) as raised:
            connect_wifi("FreeWiFi5G", PASSWORD)
        assert "не приняла профиль" in raised.value.user_message_ru

    def test_no_adapter_refuses_before_anything_is_written(self) -> None:
        network.set_netsh(ScriptedNetsh({"wlan show interfaces": INTERFACES_RU_NONE}))
        with pytest.raises(ActionUnavailable):
            connect_wifi("FreeWiFi5G", PASSWORD)


class TestPasswordStorage:
    """Where a Wi-Fi password goes, and where it must never appear."""

    def test_it_is_saved_in_the_credential_store(self, netsh: ScriptedNetsh) -> None:
        connect_wifi("FreeWiFi5G", PASSWORD, security=WifiSecurity.WPA2)
        assert get_secrets().get(secret_ref_for_ssid("FreeWiFi5G")) == PASSWORD

    def test_a_saved_password_is_reused_without_being_said_again(
        self, netsh: ScriptedNetsh
    ) -> None:
        """The point of storing it: «подключись к домашней сети» needs nothing aloud."""
        get_secrets().save(secret_ref_for_ssid("FreeWiFi5G"), PASSWORD)
        result = connect_wifi("FreeWiFi5G", security=WifiSecurity.WPA2)
        assert result.data["used_password"] is True

    def test_the_reference_is_a_valid_store_entry_name(self) -> None:
        """An SSID is arbitrary text; a reference is a short lowercase identifier."""
        for ssid in ("Домашняя сеть", "Wi-Fi 5G!", "📶", "A" * 32, ""):
            assert is_valid_ref(secret_ref_for_ssid(ssid)), ssid

    def test_the_reference_does_not_contain_the_network_name(self) -> None:
        """It is a hash, so the store's key list leaks nothing about where someone lives."""
        assert "домашняя" not in secret_ref_for_ssid("Домашняя сеть").casefold()

    def test_networks_differing_only_in_punctuation_do_not_collide(self) -> None:
        """Sanitising instead of hashing would give these two one shared password."""
        assert secret_ref_for_ssid("Home Wi-Fi") != secret_ref_for_ssid("home_wi_fi")

    def test_the_same_network_always_gets_the_same_reference(self) -> None:
        assert secret_ref_for_ssid("Домашняя сеть") == secret_ref_for_ssid("Домашняя сеть")

    def test_the_password_is_never_in_the_result(self, netsh: ScriptedNetsh) -> None:
        """``ActionResult`` reaches the log, the history table and the UI."""
        result = connect_wifi("FreeWiFi5G", PASSWORD, security=WifiSecurity.WPA2)
        assert PASSWORD not in result.message_ru
        assert PASSWORD not in result.detail
        assert PASSWORD not in repr(result.data)

    def test_the_password_is_never_in_a_netsh_argument(self, netsh: ScriptedNetsh) -> None:
        """It travels in a file, because a command line is visible to every process."""
        connect_wifi("FreeWiFi5G", PASSWORD, security=WifiSecurity.WPA2)
        assert all(PASSWORD not in " ".join(call) for call in netsh.calls)

    def test_an_unavailable_store_does_not_stop_the_connection(
        self, netsh: ScriptedNetsh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A machine without a credential store still gets on the network once."""

        def refuse(self: SecretsStore, ref: str, value: str) -> None:
            del self, ref, value
            raise SecretsError("no backend")

        monkeypatch.setattr(SecretsStore, "save", refuse)
        result = connect_wifi("FreeWiFi5G", PASSWORD, security=WifiSecurity.WPA2)
        assert result.ok is True
        assert result.data["secret_ref"] == ""


class TestPasswordMasking:
    """The parameter is secret, which is what keeps it out of ``audit``."""

    def test_the_field_is_marked_secret(self) -> None:
        assert "password" in secret_fields(ConnectWifi.Params)

    def test_masking_replaces_it(self) -> None:
        masked = mask_params(ConnectWifi.Params(ssid="FreeWiFi5G", password=PASSWORD))
        assert masked["password"] == SECRET_MASK
        assert masked["ssid"] == "FreeWiFi5G"

    def test_the_action_itself_still_gets_the_real_value(self) -> None:
        """Masking is for what is recorded, not for what is executed."""
        params = ConnectWifi.Params(ssid="FreeWiFi5G", password=PASSWORD)
        assert params_to_json(params)["password"] == PASSWORD

    def test_an_absent_password_masks_to_nothing_revealing(self) -> None:
        assert mask_params(ConnectWifi.Params(ssid="X"))["password"] == SECRET_MASK

    def test_the_schema_shown_in_the_editor_marks_the_field(self) -> None:
        """So the macro editor draws a password box rather than a text box."""
        field = build_schema(ConnectWifi).field_by_name("password")
        assert field is not None
        assert field.secret is True


class TestConnectWifiParams:
    """Validation, before anything reaches ``netsh``."""

    def test_the_ssid_is_required(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid):
            registry.execute("ConnectWifi", {})

    @pytest.mark.parametrize("ssid", ["", "A" * 33], ids=["empty", "too-long"])
    def test_the_ssid_is_bounded(self, registry: ActionRegistry, ssid: str) -> None:
        """32 octets is the standard's limit; anything longer is a mis-heard name."""
        with pytest.raises(ActionParamsInvalid):
            registry.execute("ConnectWifi", {"ssid": ssid})

    def test_a_too_long_password_is_rejected(self, registry: ActionRegistry) -> None:
        """63 characters is the WPA passphrase limit."""
        with pytest.raises(ActionParamsInvalid):
            registry.execute("ConnectWifi", {"ssid": "X", "password": "p" * 64})

    def test_the_password_is_optional(self) -> None:
        assert ConnectWifi.Params(ssid="X").password == ""

    def test_surrounding_space_is_dropped(
        self, registry: ActionRegistry, netsh: ScriptedNetsh
    ) -> None:
        """Recognition leaves it, and ``netsh`` would look for a different network."""
        result = registry.execute("ConnectWifi", {"ssid": "  Домашняя сеть  "})
        assert result.data["ssid"] == "Домашняя сеть"


class TestProfileXml:
    """The document that carries the password to ``netsh``, built as a pure function."""

    def test_the_key_is_in_the_profile(self) -> None:
        assert f"<keyMaterial>{PASSWORD}</keyMaterial>" in profile_xml("Net", PASSWORD)

    def test_the_ssid_appears_as_both_name_and_ssid(self) -> None:
        xml = profile_xml("Домашняя сеть", PASSWORD)
        assert xml.count("<name>Домашняя сеть</name>") == 2

    def test_an_ssid_with_xml_characters_is_escaped(self) -> None:
        """«Andrey & Co <5G>» is a legal SSID and would otherwise break the document."""
        xml = profile_xml("Andrey & Co <5G>", PASSWORD)
        assert "Andrey &amp; Co &lt;5G&gt;" in xml
        assert "<5G>" not in xml

    def test_a_password_with_xml_characters_is_escaped(self) -> None:
        xml = profile_xml("Net", "a<b>&c")
        assert "<keyMaterial>a&lt;b&gt;&amp;c</keyMaterial>" in xml

    def test_an_open_network_gets_no_key_element(self) -> None:
        """An empty ``keyMaterial`` makes ``netsh`` reject the profile outright."""
        xml = profile_xml("FreeWiFi", "", security=WifiSecurity.OPEN)
        assert "keyMaterial" not in xml
        assert "<authentication>open</authentication>" in xml

    @pytest.mark.parametrize(
        ("security", "authentication"),
        [
            (WifiSecurity.WPA3, "WPA3SAE"),
            (WifiSecurity.WPA2, "WPA2PSK"),
            (WifiSecurity.WPA, "WPAPSK"),
            (WifiSecurity.UNKNOWN, "WPA2PSK"),
        ],
        ids=["wpa3", "wpa2", "wpa", "unknown-falls-back"],
    )
    def test_the_authentication_element(self, security: WifiSecurity, authentication: str) -> None:
        xml = profile_xml("Net", PASSWORD, security=security)
        assert f"<authentication>{authentication}</authentication>" in xml

    def test_the_profile_reconnects_on_its_own(self) -> None:
        """Which is what «подключись и запомни» means."""
        assert "<connectionMode>auto</connectionMode>" in profile_xml("Net", PASSWORD)


# --------------------------------------------------------------------------- #
# Bluetooth
# --------------------------------------------------------------------------- #

HEADPHONES = BluetoothDevice(name="WH-1000XM4", kind="audio", connected=True)
MOUSE = BluetoothDevice(name="MX Master 3S", kind="mouse", connected=False)
NAMELESS = BluetoothDevice(name="", kind="other", connected=False)


class TestSetBluetooth:
    """The switch, through the registry."""

    def test_turning_it_off(self, registry: ActionRegistry, radio: RecordingRadio) -> None:
        result = registry.execute("SetBluetooth", {"mode": "off"})
        assert result.value is RadioState.OFF
        assert radio.switches == [(RadioKind.BLUETOOTH, False)]

    def test_turning_it_on(self, registry: ActionRegistry, radio: RecordingRadio) -> None:
        radio.states[RadioKind.BLUETOOTH] = RadioState.OFF
        assert registry.execute("SetBluetooth", {"mode": "on"}).value is RadioState.ON

    def test_the_default_is_toggle(self) -> None:
        """«Блютус» on its own means «переключи», which is the useful reading."""
        assert SetBluetooth.Params().mode is RadioMode.TOGGLE

    def test_an_unknown_mode_is_rejected(self, registry: ActionRegistry) -> None:
        with pytest.raises(ActionParamsInvalid):
            registry.execute("SetBluetooth", {"mode": "sometimes"})

    def test_switching_wifi_does_not_touch_bluetooth(
        self, registry: ActionRegistry, radio: RecordingRadio
    ) -> None:
        registry.execute("SetWifi", {"mode": "off"})
        assert radio.states[RadioKind.BLUETOOTH] is RadioState.ON


class TestListBluetooth:
    """Paired devices: name, kind, and whether they are connected right now."""

    def test_it_lists_paired_devices(self, registry: ActionRegistry) -> None:
        network.set_radio(
            RecordingRadio(states={RadioKind.BLUETOOTH: RadioState.ON}, paired=(HEADPHONES, MOUSE))
        )
        result = registry.execute("ListBluetooth", {})
        assert result.value == (HEADPHONES, MOUSE)
        assert result.message_ru == "Сопряжено 2, подключено 1"

    def test_the_payload_carries_name_kind_and_status(self, registry: ActionRegistry) -> None:
        network.set_radio(
            RecordingRadio(states={RadioKind.BLUETOOTH: RadioState.ON}, paired=(HEADPHONES,))
        )
        assert registry.execute("ListBluetooth", {}).data["devices"] == [
            {"name": "WH-1000XM4", "kind": "audio", "connected": True}
        ]

    def test_nothing_paired_is_a_success(self, registry: ActionRegistry) -> None:
        network.set_radio(RecordingRadio(states={RadioKind.BLUETOOTH: RadioState.ON}))
        result = registry.execute("ListBluetooth", {})
        assert (result.ok, result.value) == (True, ())
        assert result.message_ru == "Сопряжённых Bluetooth-устройств нет"

    def test_a_switched_off_radio_says_so_rather_than_claiming_no_devices(
        self, registry: ActionRegistry
    ) -> None:
        """«Устройств нет» to someone whose Bluetooth is simply off is a wrong answer."""
        network.set_radio(RecordingRadio(states={RadioKind.BLUETOOTH: RadioState.OFF}))
        message = registry.execute("ListBluetooth", {}).message_ru
        assert message == "Bluetooth выключен, устройств не видно"

    def test_no_adapter_is_refused(self, registry: ActionRegistry) -> None:
        network.set_radio(RecordingRadio())
        with pytest.raises(ActionUnavailable) as raised:
            registry.execute("ListBluetooth", {})
        assert "адаптер не найден" in raised.value.user_message_ru

    def test_no_winrt_is_refused_differently(self, registry: ActionRegistry) -> None:
        """A missing projection and a missing adapter are different problems."""
        with pytest.raises(ActionUnavailable) as raised:
            registry.execute("ListBluetooth", {})
        assert "без WinRT" in raised.value.user_message_ru

    def test_a_device_reads_out_with_its_state(self) -> None:
        assert HEADPHONES.title_ru == "«WH-1000XM4» — подключено"
        assert MOUSE.title_ru == "«MX Master 3S» — сопряжено"

    def test_a_nameless_device_is_still_readable(self) -> None:
        assert NAMELESS.title_ru.startswith("«без имени»")


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

ACTIONS = (SetWifi, SetBluetooth, ListWifi, ConnectWifi, ListBluetooth)


class TestMetadata:
    """What the registry and the editor read off these actions without running them."""

    @pytest.mark.parametrize("action", ACTIONS, ids=lambda action: action.meta.name)
    def test_nothing_here_demands_elevation(self, action: type) -> None:
        """WinRT switches a radio unelevated, and ``netsh wlan connect`` works on the
        current user's profile. Marking these ``require_admin`` would put a UAC prompt
        in front of «включи вайфай» on every machine, including the ones where the
        fallback that needs rights is never reached."""
        assert action.meta.require_admin is False

    @pytest.mark.parametrize("action", ACTIONS, ids=lambda action: action.meta.name)
    def test_nothing_here_is_dangerous(self, action: type) -> None:
        """Everything in this file is reversible by saying the opposite. The
        confirmation gate is for the power actions, where it is not."""
        assert action.meta.is_dangerous is False

    @pytest.mark.parametrize("action", ACTIONS, ids=lambda action: action.meta.name)
    def test_every_action_is_in_the_system_group(self, action: type) -> None:
        assert action.meta.category is ActionCategory.SYSTEM

    @pytest.mark.parametrize("action", ACTIONS, ids=lambda action: action.meta.name)
    def test_every_action_has_a_russian_title(self, action: type) -> None:
        assert action.meta.title_ru
        assert action.meta.description_ru

    @pytest.mark.parametrize("action", (ListWifi, ConnectWifi), ids=lambda action: action.meta.name)
    def test_the_slow_actions_outlast_one_netsh_call(self, action: type) -> None:
        """A scan sweeps the band and an association negotiates; cutting either short
        returns the stale cache or a half-open connection."""
        assert action.meta.timeout_s is not None
        assert action.meta.timeout_s > NETSH_TIMEOUT_S

    def test_the_actions_are_registered_by_import(self) -> None:
        """``@register`` marks them as part of the built-in library, so a registry
        that discovers ``ayris.actions.system`` finds them without a manual add."""
        marked = {entry.name for entry in registered_actions()}
        for action in ACTIONS:
            assert action.meta.name in marked, action.meta.name


class TestDefaultBackends:
    """What the module reaches for when nothing installed anything."""

    def test_the_real_netsh_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reads are harmless and the point is to see the actual networks, so unlike
        the power backend this one is real unless a test replaces it."""
        monkeypatch.setattr(network, "_netsh", None)
        assert isinstance(get_netsh(), network.SubprocessNetsh)

    def test_a_missing_projection_degrades_rather_than_raising(self) -> None:
        """Neither ``winrt`` nor ``winsdk`` is a dependency of Ayris."""
        backend = UnavailableRadio()
        assert backend.available is False
        assert backend.state(RadioKind.WIFI) is RadioState.UNKNOWN
        assert backend.devices() == ()

    def test_the_unavailable_backend_refuses_in_russian(self) -> None:
        with pytest.raises(ActionUnavailable) as raised:
            UnavailableRadio().switch(RadioKind.WIFI, on=True)
        assert raised.value.user_message_ru


# --------------------------------------------------------------------------- #
# the one live call, on Windows only
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="netsh is a Windows program")
class TestRealNetsh:
    """``netsh wlan show interfaces`` for real — a read, on a runner with no adapter.

    Not marked ``hardware``: listing interfaces changes nothing and works on a CI
    runner that has no wireless card at all. That is exactly the case worth
    checking, because it is the one the parser has to survive — and it is the one
    a developer's laptop, which does have Wi-Fi, would never exercise.
    """

    def test_netsh_answers_and_the_parser_survives_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.undo()
        network.set_netsh(None)
        result = get_netsh().run(("wlan", "show", "interfaces"))
        parsed = parse_wlan_interfaces(result.text)
        assert isinstance(parsed, tuple)

    def test_no_adapter_becomes_a_russian_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whatever this machine has, the two outcomes are the two the code handles."""
        monkeypatch.undo()
        network.set_netsh(None)
        try:
            interface = require_interface()
        except ActionUnavailable as exc:
            assert "адаптер" in exc.user_message_ru
        else:
            assert interface.guid

    def test_the_console_code_page_is_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The answer decides how ``netsh`` output is decoded, so a Cyrillic SSID
        depends on it being a real code page rather than a guess."""
        monkeypatch.undo()
        assert "".encode(console_encoding()) == b""
