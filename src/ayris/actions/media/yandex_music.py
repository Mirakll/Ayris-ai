"""The advanced level: Яндекс Музыка's own interface, clicked from the inside.

A transport control can say «pause» and «next». It cannot say «включи Сплина», «моя
волна», «лайкни этот трек» or «добавь его в Любимку» — those are not transport
commands, they are the player's own features, and the only place they exist is its
interface. The desktop app is Electron, so its interface is a web page, so the way
in is the DevTools protocol and one line of JavaScript per wish.

**The whole technique is ``element.click()`` on a real ``<button>``.** No
coordinates, no synthesised mouse events, no focus: the app does not have to be in
the foreground, the mouse does not move, and a node scrolled out of view is clicked
just as well as a visible one — which matters, because the Моя волна wheel keeps
half its presets off-screen. Nine attempts at this failed earlier by clicking the
wrapper ``<div>`` around a button (``PLAY_BUTTON_WITH_COVER``) instead of the button
inside it; wrappers carry no handler. Hence the rule behind :data:`SELECTORS`: every
entry ends at something clickable.

**Navigation goes through the app's own router**, ``window.next.router.push``.
``location.assign`` and ``Page.navigate`` reload the page, which stops the music and
leaves the buttons dead for a second — never use them here. The one route the router
refuses is ``/``: the home page (which *is* Моя волна) opens by clicking the navbar
item instead.

**Every name the app could rename lives in one table** — :data:`SELECTORS` and
:data:`ROUTES` — and ``[actions.media] selectors`` overrides any entry from the
settings file, so an app update that renames a ``data-test-id`` is a one-line fix
for the user rather than a new release. Where a name is likely to move, the
JavaScript also has a second way in: the wave presets are matched by ``aria-label``
and the playlist menu by its visible text.

What this module needs and cannot arrange by itself is the debug port. Яндекс Музыка
opens one only when started with ``--remote-debugging-port``, and it has no autostart
entry to add the flag to, so Ayris starts the app itself (see :func:`open_music`). An
app already running *without* the flag is not restarted behind the user's back:
these actions fail with :class:`~ayris.actions.media.cdp.CdpUnavailable`, whose
message says what to do, and pause/play/next/previous keep working through
:mod:`ayris.actions.media.smtc` meanwhile.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from enum import StrEnum
from string import Template
from typing import Any, ClassVar, Final

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.media import cdp
from ayris.actions.media.cdp import CdpError, CdpTransport, CdpUnavailable
from ayris.actions.media.smtc import current_session
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.app_index import get_app_index
from ayris.actions.system.apps import LaunchKind, LaunchRequest, get_launcher
from ayris.core.config import get_settings
from ayris.core.errors import ActionError
from ayris.utils.logger import get_logger

__all__ = [
    "ROUTES",
    "SELECTORS",
    "LikeMode",
    "SearchKind",
    "YMAddToPlaylist",
    "YMLike",
    "YMPlaylist",
    "YMSearch",
    "YMWave",
    "debug_flags",
    "open_music",
    "selector",
    "set_transport",
    "start_from_search",
    "toggle_in_playlist",
]

_log = get_logger(__name__)

#: Name the application is looked up under when Ayris has to start it. Matched by
#: :class:`~ayris.actions.system.app_index.AppIndex`, so «Яндекс Музыка» finds the
#: Start-menu entry the installer made.
PLAYER_NAME: Final = "Яндекс Музыка"

#: Flags that make the app open a DevTools port. The second one is not optional:
#: Chromium 111 and later refuse a DevTools socket unless origins are allowed.
DEBUG_FLAG_TEMPLATE: Final = "--remote-debugging-port={port} --remote-allow-origins=*"


class SearchKind(StrEnum):
    """What a search is looking for, which decides which card gets clicked."""

    TRACK = "track"
    ARTIST = "artist"
    ALBUM = "album"
    PLAYLIST = "playlist"

    @property
    def title_ru(self) -> str:
        return _KIND_TITLES[self]


_KIND_TITLES: Final[dict[SearchKind, str]] = {
    SearchKind.TRACK: "трек",
    SearchKind.ARTIST: "исполнитель",
    SearchKind.ALBUM: "альбом",
    SearchKind.PLAYLIST: "плейлист",
}


class LikeMode(StrEnum):
    """What «лайк» should mean when the track is already liked."""

    LIKE = "like"
    UNLIKE = "unlike"
    TOGGLE = "toggle"

    @property
    def title_ru(self) -> str:
        return _LIKE_TITLES[self]


_LIKE_TITLES: Final[dict[LikeMode, str]] = {
    LikeMode.LIKE: "поставить лайк",
    LikeMode.UNLIKE: "снять лайк",
    LikeMode.TOGGLE: "переключить лайк",
}


#: Every selector the JavaScript below uses, in one place because the task asks for
#: it and because ``data-test-id`` is the app's internal business: it can be renamed
#: by an update that changes nothing a user would notice. All of these were clicked
#: on the running app, version 5.115.3.
#:
#: Two conventions hold throughout. A name ending in ``_play`` or ``_pause`` is a
#: real ``<button>``, never the card around it. ``play_inner`` is meant to be
#: queried *inside* a card or a row, which is how one track out of a list is
#: started.
SELECTORS: Final[Mapping[str, str]] = {
    # The bottom bar, on every page except the home page.
    "playerbar": '[data-test-id="PLAYERBAR_DESKTOP"]',
    "playerbar_play": '[data-test-id="PLAYERBAR_DESKTOP"] [data-test-id="PLAY_BUTTON"]',
    "playerbar_pause": '[data-test-id="PLAYERBAR_DESKTOP"] [data-test-id="PAUSE_BUTTON"]',
    "playerbar_like": '[data-test-id="PLAYERBAR_DESKTOP"] [data-test-id="LIKE_BUTTON"]',
    "playerbar_menu": '[data-test-id="PLAYERBAR_DESKTOP_CONTEXT_MENU_BUTTON"]',
    # The home page is Моя волна and hides the bottom bar, keeping its own.
    "vibe_playerbar": '[data-test-id="VIBE_PLAYERBAR"]',
    "vibe_play": '[data-test-id="VIBE_PLAYERBAR"] [data-test-id="PLAY_BUTTON"]',
    "vibe_pause": '[data-test-id="VIBE_PLAYERBAR"] [data-test-id="PAUSE_BUTTON"]',
    "vibe_like": '[data-test-id="VIBE_PLAYERBAR"] [data-test-id="LIKE_BUTTON"]',
    # Своё имя, не ``TRACK_CONTEXT_MENU_BUTTON``: у нижней панели главной страницы
    # вообще всё своё. Меню при этом трековое — в нём «Добавить в плейлист» и
    # «Перейти к треку», проверено на играющей волне.
    "vibe_menu": '[data-test-id="VIBE_PLAYERBAR"] [data-test-id="VIBE_CONTEXT_MENU_BUTTON"]',
    # Present only while the wave is playing «in context» — popular tracks of an
    # artist, a playlist. Its absence is what a clean Моя волна looks like.
    "vibe_reset": '[data-test-id="RESET_VIBE_CONTEXT_BUTTON"]',
    "vibe_wheel_item": '[data-test-id="WHEEL_VIBE_ITEM"]',
    "nav_home": '[data-test-id="NAVBAR_NAVIGATION_ITEM_HOME"]',
    # Start button in the header of a playlist, artist or album page — same name on
    # all three, which is why one routine covers them.
    "page_play": '[data-test-id="BASE_PAGE_HEADER_CONTROLS"] [data-test-id="PLAY_BUTTON"]',
    "page_pause": '[data-test-id="BASE_PAGE_HEADER_CONTROLS"] [data-test-id="PAUSE_BUTTON"]',
    # Search results. Каждый вид результата — своя карточка с собственным
    # data-test-id: проверено на живой выдаче, `a[href*="playlist"]` вместо карточки
    # находил только закреплённые плейлисты в боковом меню, а `a[href*="/album"]` —
    # ссылки на треки вида /album/track?…, то есть вообще не альбомы.
    "search_track_card": '[data-test-id="SEARCH_TRACK_CARD"]',
    "search_artist_card": '[data-test-id="HORIZONTAL_ARTIST_CARD"]',
    "search_album_card": '[data-test-id="HORIZONTAL_ALBUM_CARD"]',
    "search_playlist_card": '[data-test-id="HORIZONTAL_PLAYLIST_CARD"]',
    "track_title": '[data-test-id="TRACK_TITLE"]',
    "album_title": '[data-test-id="ALBUM_TITLE"]',
    "playlist_title": '[data-test-id="PLAYLIST_TITLE"]',
    "artist_title": '[data-test-id="SEPARATED_ARTIST_TITLE"]',
    "play_inner": '[data-test-id="PLAY_BUTTON"]',
    # Track menu → «Добавить в плейлист» → the playlist itself.
    "menu_add_to_playlist": '[data-test-id="TRACK_CONTEXT_MENU_ADD_TO_PLAYLIST_BUTTON"]',
    "submenu_any": (
        '[data-test-id="TRACK_SUBMENU_ITEM"], '
        '[data-test-id="TRACK_SUBMENU_LIKE_PLAYLIST_BUTTON"]'
    ),
    "submenu_item": '[data-test-id="TRACK_SUBMENU_ITEM"]',
    "submenu_like_playlist": '[data-test-id="TRACK_SUBMENU_LIKE_PLAYLIST_BUTTON"]',
    "submenu_in_playlist": '[data-test-id="TRACK_SUBMENU_IN_PLAYLIST_ICON"]',
}

#: Router paths. Query strings rather than path segments, which is what the app's
#: own links use, and ``/`` is deliberately absent: the router ignores it.
ROUTES: Final[Mapping[str, str]] = {
    "search": "/search?text={text}",
    "playlist": "/playlists?playlistUuid={uuid}",
    "artist": "/artist?artistId={artist_id}",
    "album": "/album?albumId={album_id}",
    "track": "/album/track?albumId={album_id}&trackId={track_id}",
    "collection": "/collection",
}

#: Which card a search result of each kind is, where its name is written, and what
#: else on the card belongs to the name. Empty ``label`` means «the card's own text»,
#: which is right for the artist card: it is mostly a title. The other three carry a
#: nested title element, and taking the whole card's text there would drag the album's
#: year and the playlist's like count into the name being matched.
#:
#: ``extra`` exists for one case, and it is the most common wish of all: a track named
#: by its artist. «Сплин Выхода нет» matches nothing on a card whose title is «Выхода
#: нет» — the artist is a separate link next to it.
_SEARCH_PLAN: Final[Mapping[SearchKind, tuple[str, str, str]]] = {
    SearchKind.TRACK: ("search_track_card", "track_title", "artist_title"),
    SearchKind.ARTIST: ("search_artist_card", "", ""),
    SearchKind.ALBUM: ("search_album_card", "album_title", ""),
    SearchKind.PLAYLIST: ("search_playlist_card", "playlist_title", ""),
}

#: What the JavaScript's ``reason`` means in Russian. The page reports a machine
#: word, the user hears a sentence, and the translation is a table rather than
#: seventeen string literals scattered through the actions.
_REASONS: Final[Mapping[str, str]] = {
    "no-router": "Не получилось перейти по странице Яндекс Музыки.",
    "no-results": "Яндекс Музыка ничего не нашла по запросу.",
    "no-match": "Не нашёл этого в выдаче Яндекс Музыки.",
    "no-play-button": "Нашёл, но не нашёл кнопку запуска.",
    "no-header-play": "Страница открылась, но кнопки запуска на ней нет.",
    "no-like-button": "Не вижу кнопки лайка — похоже, ничего не играет.",
    "no-vibe-playerbar": "Не получилось открыть Мою волну.",
    "no-home-button": "Не нашёл кнопку главной страницы.",
    "no-preset": "Такой волны нет в списке.",
    "no-track-menu": "Не вижу меню трека — похоже, ничего не играет.",
    "no-add-item": "В меню трека нет пункта «Добавить в плейлист».",
    "no-submenu": "Список плейлистов не открылся.",
    "no-playlist": "Не нашёл такой плейлист.",
}

_DEFAULT_REASON: Final = "Яндекс Музыка не ответила так, как ожидалось."

#: Helpers every routine below starts with. Written once, sent with each expression:
#: the page is not ours to install anything into, and ``Runtime.evaluate`` shares one
#: global scope between calls, which is why :func:`_script` wraps all of this in a
#: function instead of letting it land next to the app's own names.
#:
#: ``element.click()`` and nothing else — see the module docstring. ``fire`` exists
#: only to close a menu (a real outside click is what dismisses it), and dispatches
#: DOM events, not input: no virtual-key code is synthesised anywhere in this file.
#: ``settle`` is the answer to a router that swaps the page's data without clearing
#: its DOM: it waits until a set of cards stops changing, and knows what was there
#: before, so the previous search's results are never mistaken for this one's.
_PRELUDE: Final = """
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const q = (sel, scope) => (scope || document).querySelector(sel);
const qa = (sel, scope) => Array.from((scope || document).querySelectorAll(sel));
const norm = (value) => (value || "").replace(/\\s+/g, " ").trim().toLowerCase();
const naming = (el) => norm(el && (el.getAttribute("aria-label") || el.textContent));
const waitFor = async (sel, ms) => {
  const started = performance.now();
  for (;;) {
    const found = q(sel);
    if (found) return found;
    if (performance.now() - started > ms) return null;
    await sleep(80);
  }
};
const words = (value) => value.split(/[^0-9\\p{L}]+/u).filter(Boolean);
const score = (have, want) => {
  if (!have || !want) return 0;
  if (have === want) return 100;
  if (have.startsWith(want)) return 80;
  if (have.includes(want)) return 60;
  if (want.includes(have)) return 40;
  // Сказанное вслух почти никогда не совпадает подстрокой: «сплин выхода нет»
  // против «выхода нет · сплин». Поэтому считаем, сколько слов запроса нашлось,
  // и вычитаем за лишние слова: иначе сборник, где есть все слова запроса и ещё
  // двадцать своих, победит короткое точное название. Потолок этой ветки ниже
  // самого слабого совпадения подстрокой — она последняя в очереди, не первая.
  const wanted = words(want);
  const found = words(have);
  if (!wanted.length || !found.length) return 0;
  const hits = wanted.filter(
    (one) => found.some((other) => other === one || (one.length > 3 && other.startsWith(one)))
  ).length;
  if (!hits) return 0;
  const noise = Math.min(found.length / wanted.length, 4);
  return Math.max(1, Math.round(35 * (hits / wanted.length) - 2 * (noise - 1)));
};
const best = (items, want) => {
  let top = null;
  let topScore = 0;
  for (const item of items) {
    const value = score(item.label, want);
    if (value > topScore) {
      top = item;
      topScore = value;
    }
  }
  return top;
};
const route = (path) => {
  const router = window.next && window.next.router;
  if (!router) return false;
  router.push(path);
  return true;
};
const mark = (sel, label) => qa(sel)
  .slice(0, 8)
  .map((el) => naming(label ? (q(label, el) || el) : el))
  .join("|");
const settle = async (sel, label, stale, ms) => {
  const started = performance.now();
  let last = null;
  let moved = false;
  for (;;) {
    const now = mark(sel, label);
    const waited = performance.now() - started;
    if (now && now !== stale) moved = true;
    if (now && now === last && (moved || waited > 900)) return now;
    if (waited > ms) return now;
    last = now;
    await sleep(120);
  }
};
const fire = (kind) => {
  const Kind = kind.startsWith("pointer") && window.PointerEvent ? PointerEvent : MouseEvent;
  document.body.dispatchEvent(new Kind(kind, {bubbles: true}));
};
const closeMenus = () => {
  fire("pointerdown");
  fire("mousedown");
  fire("click");
};
"""


def _script(body: str) -> Template:
    """Помощники и тело — в одной области видимости, ничего не попадает в ``window``.

    ``Runtime.evaluate`` выполняет выражение как скрипт в глобальной области, и она
    у страницы одна на все соединения. Объявленный там ``const sleep`` живёт до
    перезагрузки, поэтому второй вызов подряд падал бы с ``SyntaxError: Identifier
    'sleep' has already been declared`` — а второй вызов подряд это просто вторая
    команда пользователя. Обёртка снимает и это, и риск затереть страницу своим
    ``q`` или ``route``.
    """
    return Template("(() => {" + _PRELUDE + "return " + body.strip() + ";})()")


#: Start a track, artist, album or playlist from the search page.
#:
#: One routine for all four because the app makes them the same shape: route to the
#: results, find the card whose name matches best, click the play button inside it.
#: The fallback matters for playlists and albums, whose result card is a link with no
#: play button of its own — then the link's own ``href`` goes through the router and
#: the page header's play button is used instead, which is the verified way to start
#: an entity from its page.
#:
#: The fingerprint taken *before* the navigation is the load-bearing part. The router
#: swaps the page's data, not its DOM: for about a second after ``push`` the previous
#: search's cards are still there, and a plain ``waitFor`` finds them instantly. That
#: is «включи Сплина» starting Кино because Кино was searched a minute ago — measured,
#: not imagined. :js:func:`settle` waits for the card set to change and stop moving.
_JS_START = _script(
    """
(async () => {
  const stale = mark($card, $label);
  if (!route($route)) return {ok: false, reason: "no-router"};
  const anchor = await waitFor($card, $wait);
  if (!anchor) return {ok: false, reason: "no-results"};
  await settle($card, $label, stale, $wait);
  if (!q($card)) return {ok: false, reason: "no-results"};
  // Сопоставляем по названию вместе с исполнителем, а показываем одно название:
  // «включи Сплин Выхода нет» — это про карточку, где Сплина нет в заголовке.
  const items = qa($card).map((el) => ({
    el: el,
    title: naming($label ? (q($label, el) || el) : el),
    extra: $extra ? naming(q($extra, el)) : "",
  })).filter((item) => item.title).map((item) => ({
    el: item.el,
    title: item.title,
    label: item.extra ? item.title + " " + item.extra : item.title,
  }));
  const found = $want ? best(items, norm($want)) : items[0];
  if (!found) {
    return {ok: false, reason: "no-match", seen: items.slice(0, 12).map((item) => item.title)};
  }
  const card = found.el.closest('[data-test-id]') || found.el;
  const play = q($play_inner, found.el) || q($play_inner, card);
  if (play) {
    play.click();
    return {ok: true, label: found.title, via: "card", count: items.length};
  }
  // Именно `getAttribute`: у свойства `href` абсолютный вид со схемой
  // music-application://, а роутеру нужен путь.
  const link = found.el.getAttribute("href") ? found.el : q("a[href]", found.el);
  const href = link && link.getAttribute("href");
  if (!href) return {ok: false, reason: "no-play-button", label: found.title};
  if (!route(href)) return {ok: false, reason: "no-router"};
  const header = await waitFor($page_play + ", " + $page_pause, $wait);
  if (!header) return {ok: false, reason: "no-header-play"};
  const already = q($page_pause);
  if (!already) header.click();
  return {ok: true, label: found.title, via: "page", href: href, already: !!already};
})()
"""
)

#: Like or unlike whatever is playing.
#:
#: ``aria-pressed`` is the state, and it is read before and after: «лайкни» said
#: about an already-liked track must not quietly remove the like, which is exactly
#: what a blind click would do.
_JS_LIKE = _script(
    """
(async () => {
  const button = q($playerbar_like) || q($vibe_like);
  if (!button) return {ok: false, reason: "no-like-button"};
  const before = button.getAttribute("aria-pressed") === "true";
  if ($want !== null && before === $want) {
    return {ok: true, changed: false, liked: before};
  }
  button.click();
  await sleep(250);
  const after = q($playerbar_like) || q($vibe_like) || button;
  return {ok: true, changed: true, liked: after.getAttribute("aria-pressed") === "true"};
})()
"""
)

#: Моя волна: the home page, optionally with its context reset, optionally a preset.
#:
#: The reset button is the part that was got wrong before: after an artist or a
#: playlist the wave plays «Популярные треки артиста», which is not Моя волна, and
#: only ``RESET_VIBE_CONTEXT_BUTTON`` puts it back. When there is no context the
#: button is not in the markup at all — its absence is the clean state.
_JS_WAVE = _script(
    """
(async () => {
  if (!q($vibe_playerbar)) {
    const home = q($nav_home);
    if (!home) return {ok: false, reason: "no-home-button"};
    home.click();
  }
  const bar = await waitFor($vibe_playerbar, $wait);
  if (!bar) return {ok: false, reason: "no-vibe-playerbar"};
  await sleep(200);
  let reset = "";
  if ($reset) {
    const button = q($vibe_reset);
    if (button) {
      reset = naming(button);
      button.click();
      await sleep(500);
    }
  }
  if ($preset) {
    const items = qa($wheel_item).map((el) => ({el: el, label: naming(el)}));
    const found = best(items, norm($preset));
    if (!found) {
      return {ok: false, reason: "no-preset", seen: items.slice(0, 30).map((i) => i.label)};
    }
    const play = q($play_inner, found.el) || found.el;
    play.click();
    return {ok: true, preset: found.label, reset: reset};
  }
  const playing = q($vibe_pause);
  if (!playing) {
    const play = q($vibe_play);
    if (play) play.click();
  }
  return {ok: true, preset: "", reset: reset, already: !!playing};
})()
"""
)

#: Add the current track to a playlist by its name: menu, «Добавить в плейлист»,
#: then the item whose text matches. A track already in that playlist is marked with
#: an icon, and then nothing is clicked — the same item would remove it.
#:
#: ``$remove`` inverts exactly that condition, which is what «отмени» does: the item
#: is clicked only when the icon *is* there. One template rather than two, because
#: the twenty lines that find the item are the same either way and the whole
#: difference is which state is the one worth clicking in.
#:
#: The state after the click is deliberately not read back here. The app closes the
#: whole menu in response to it, so there is nothing left to read; and the icon in a
#: freshly opened menu catches up with the click only about a second later — measured,
#: and the reason :data:`_SETTLE_S` exists. Five reads in a row of an untouched menu
#: gave the same answer every time, so the icon itself is trustworthy, just not
#: instant.
_JS_ADD_TO_PLAYLIST = _script(
    """
(async () => {
  const menu = q($playerbar_menu) || q($vibe_menu);
  if (!menu) return {ok: false, reason: "no-track-menu"};
  menu.click();
  const add = await waitFor($menu_add, 4000);
  if (!add) return {ok: false, reason: "no-add-item"};
  add.click();
  const opened = await waitFor($submenu_any, 4000);
  if (!opened) return {ok: false, reason: "no-submenu"};
  await sleep(250);
  const items = qa($submenu_any).map((el) => ({el: el, label: naming(el)}));
  const found = best(items, norm($playlist));
  if (!found) {
    closeMenus();
    return {ok: false, reason: "no-playlist", seen: items.slice(0, 30).map((i) => i.label)};
  }
  const inside = !!q($in_playlist, found.el);
  if (inside !== $remove) {
    closeMenus();
    return {ok: true, changed: false, label: found.label, inside: inside};
  }
  found.el.click();
  // Меню приложение закрывает само в ответ на клик по пункту, но если версия
  // перестанет — открытое подменю останется висеть поверх интерфейса, и это
  // увидит пользователь. Закрыть стоит дешевле, чем проверить.
  await sleep(400);
  closeMenus();
  return {ok: true, changed: true, label: found.label, inside: inside};
})()
"""
)


def selector(name: str) -> str:
    """One selector, with the settings file's override applied if there is one.

    Raises:
        KeyError: no such entry. A programming mistake, not a runtime condition.
    """
    override = get_settings().actions.media.selectors.get(name, "").strip()
    return override or SELECTORS[name]


def debug_flags(port: int) -> str:
    """Command line that makes Яндекс Музыка listen on ``port``."""
    return DEBUG_FLAG_TEMPLATE.format(port=port)


def _js(value: Any) -> str:
    """A Python value as a JavaScript literal. ``json.dumps`` is exactly that."""
    return json.dumps(value, ensure_ascii=False)


_transport: CdpTransport | None = None


def set_transport(transport: CdpTransport | None) -> None:
    """Use ``transport`` instead of opening a real connection. Test seam."""
    global _transport
    _transport = transport


def _launch_player(port: int) -> str:
    """Start the app with the debug flags. Returns the command that was run.

    Raises:
        AppNotFound: the application is not installed, or is not in the index and
            ``[actions.media] player_path`` is not set either.
        ActionError: the launch itself failed.
    """
    media = get_settings().actions.media
    arguments = debug_flags(port)
    path = media.player_path.strip()
    if path:
        request = LaunchRequest(
            target=path,
            kind=LaunchKind.of(path),
            arguments=arguments,
            name=PLAYER_NAME,
        )
    else:
        candidate = get_app_index().resolve(media.player_name or PLAYER_NAME)
        request = LaunchRequest.for_candidate(candidate, arguments=arguments)
    pid = get_launcher().launch(request)
    _log.info("запускаю Яндекс Музыку с отладкой: %s (pid %s)", request.command_ru, pid)
    return request.command_ru


#: Вопрос, которым проверяется, что интерфейс приложения уже поднялся.
#:
#: Отдельной строкой, потому что это единственное выражение, которое уезжает в
#: страницу не из ``_JS_*``: короткое, без помощников и без обёртки.
_JS_READY: Final = "!!(window.next && window.next.router)"


def _wait_ready(client: cdp.CdpClient, *, timeout_s: float) -> cdp.CdpClient:
    """Дождаться, пока страница приложения поднимет свой роутер, и вернуть клиента.

    Отладочный порт открывается раньше интерфейса: сразу после холодного старта
    ``window.next.router`` ещё нет, окно показывает заставку, и первая же команда
    упала бы с «не получилось перейти по странице Яндекс Музыки» — притом что
    приложение поднялось нормально, просто на полсекунды раньше нас.

    Raises:
        CdpUnavailable: интерфейс не загрузился за ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        # Во время загрузки контекст страницы могут заменить — это не отказ порта.
        with suppress(CdpError):
            if client.evaluate(_JS_READY, await_promise=False):
                return client
        if time.monotonic() >= deadline:
            client.close()
            raise CdpUnavailable(
                "Яндекс Музыка did not finish loading its interface",
                user_message="Яндекс Музыка ещё загружается. Попробуй ещё раз.",
            )
        time.sleep(0.3)


def _attach() -> cdp.CdpClient:
    """A connected CDP client, starting the app first if that is what it takes.

    Raises:
        CdpUnavailable: the app is running without the debug port (which is not
            fixed by restarting it under the user), starting it is switched off, or
            it did not open the port in time.
    """
    media = get_settings().actions.media
    port = media.debug_port
    if cdp.is_port_open(port):
        return _wait_ready(
            cdp.connect(port, timeout=media.command_timeout_s),
            timeout_s=media.launch_timeout_s,
        )
    if current_session() is not None:
        # The app is up — SMTC sees its session — but it was started without the
        # flag. Restarting it would cut the music off mid-track for a reason the
        # user did not ask about, so this stays their decision.
        raise CdpUnavailable(
            f"Яндекс Музыка runs without --remote-debugging-port={port}",
            user_message=(
                "Яндекс Музыка запущена без отладочного порта: могу только ставить "
                "на паузу и переключать треки. Закрой и снова открой её через Ayris, "
                "чтобы работало остальное."
            ),
        )
    if not media.launch_app:
        raise CdpUnavailable(
            f"debug port {port} is closed and launching is disabled",
            user_message="Яндекс Музыка не запущена.",
        )
    _launch_player(port)
    deadline = time.monotonic() + media.launch_timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.4)
        if cdp.is_port_open(port):
            return _wait_ready(
                cdp.connect(port, timeout=media.command_timeout_s),
                timeout_s=max(deadline - time.monotonic(), 1.0),
            )
    raise CdpUnavailable(
        f"Яндекс Музыка did not open port {port} in {media.launch_timeout_s:g} s",
        user_message="Яндекс Музыка не успела запуститься.",
    )


@contextmanager
def open_music() -> Iterator[CdpTransport]:
    """A page to evaluate JavaScript in, closed again afterwards.

    Short-lived on purpose: one action, one connection. Holding the socket open
    between commands would mean surviving the app's own reloads for no gain — the
    handshake is one round trip on loopback.
    """
    if _transport is not None:
        yield _transport
        return
    client = _attach()
    try:
        yield client
    finally:
        client.close()


def _outcome(raw: Any, *, what: str) -> Mapping[str, Any]:
    """The routine's answer, or the failure it reported, as a typed error.

    Raises:
        CdpError: the page returned something that is not an outcome object.
        ActionError: the routine ran and reported why it could not finish.
    """
    if not isinstance(raw, Mapping):
        raise CdpError(f"{what}: the page returned {type(raw).__name__}, expected an object")
    if raw.get("ok"):
        return raw
    reason = str(raw.get("reason") or "unknown")
    seen = raw.get("seen")
    detail = f"{what}: {reason}"
    if seen:
        detail = f"{detail}; seen {seen}"
    raise ActionError(detail, user_message=_REASONS.get(reason, _DEFAULT_REASON))


def start_from_search(kind: SearchKind, query: str) -> Mapping[str, Any]:
    """Find ``query`` on the search page and start the best match of ``kind``."""
    card_name, label_name, extra_name = _SEARCH_PLAN[kind]
    media = get_settings().actions.media
    script = _JS_START.substitute(
        route=_js(ROUTES["search"].format(text=query)),
        card=_js(selector(card_name)),
        label=_js(selector(label_name) if label_name else ""),
        extra=_js(selector(extra_name) if extra_name else ""),
        want=_js(query),
        play_inner=_js(selector("play_inner")),
        page_play=_js(selector("page_play")),
        page_pause=_js(selector("page_pause")),
        wait=media.render_timeout_ms,
    )
    with open_music() as page:
        return _outcome(page.evaluate(script), what=f"start {kind}")


#: Сколько отметка о плейлисте догоняет клик по пункту подменю.
#:
#: Замерено на живом приложении: сразу после клика заново открытое меню показывает
#: состояние *до* него, а через секунду — верное. Это ровно то, что прочитает
#: «отмени» сказанное сразу после «добавь»: увидит трек не добавленным и не станет
#: ничего снимать. Голосом так быстро не выйдет, но `undo` зовут и из кода.
_SETTLE_S: Final = 1.2

#: Когда в последний раз кликали пункт подменю. Ноль — ни разу за эту сессию.
_last_toggle = 0.0


def toggle_in_playlist(playlist: str, *, remove: bool) -> Mapping[str, Any]:
    """Put the playing track into ``playlist``, or take it out of there.

    One function for both directions because in the app it is one menu item: the
    submenu entry adds a track that is not in the playlist and removes one that is,
    and everything before that click — opening the menu, finding «Добавить в
    плейлист», matching the name — is identical.
    """
    global _last_toggle
    settle = _SETTLE_S - (time.monotonic() - _last_toggle)
    if settle > 0:
        time.sleep(settle)
    script = _JS_ADD_TO_PLAYLIST.substitute(
        playerbar_menu=_js(selector("playerbar_menu")),
        vibe_menu=_js(selector("vibe_menu")),
        menu_add=_js(selector("menu_add_to_playlist")),
        submenu_any=_js(selector("submenu_any")),
        in_playlist=_js(selector("submenu_in_playlist")),
        playlist=_js(playlist),
        remove=_js(remove),
    )
    what = "remove from playlist" if remove else "add to playlist"
    with open_music() as page:
        answer = _outcome(page.evaluate(script), what=what)
    if answer.get("changed"):
        _last_toggle = time.monotonic()
    return answer


def _playing_track() -> str:
    """What is playing right now, as one line, or empty when nothing is."""
    session = current_session()
    return session.label_ru if session is not None else ""


@register
class YMSearch(Action):
    """Play a named track, artist or album — «включи Сплин», «включи Выхода нет».

    The search page is reached by route rather than by typing: the query goes into
    the URL, so no text is entered anywhere and no key is pressed.
    """

    meta: ClassVar = ActionMeta(
        name="YMSearch",
        category=ActionCategory.MEDIA,
        title_ru="Включить в Яндекс Музыке",
        description_ru="Найти трек, исполнителя или альбом и включить его",
        timeout_ms=60_000,
    )

    class Params(ActionParams):
        query: str = Field(
            min_length=1,
            max_length=200,
            description="Что включить: «Сплин», «Выхода нет»",
        )
        kind: SearchKind = Field(
            default=SearchKind.TRACK,
            description="Искать трек, исполнителя, альбом или плейлист",
        )

    def run(self, params: Params) -> ActionResult[str]:
        found = start_from_search(params.kind, params.query)
        label = str(found.get("label") or params.query)
        return ActionResult.done(
            f"Включаю {label}.",
            value=label,
            detail=f"{params.kind} {label!r} via {found.get('via')}",
            data={"kind": str(params.kind), "query": params.query, "label": label},
        )


@register
class YMPlaylist(Action):
    """Play a playlist by name — «включи Любимку», «поставь мой плейлист номер 1»."""

    meta: ClassVar = ActionMeta(
        name="YMPlaylist",
        category=ActionCategory.MEDIA,
        title_ru="Включить плейлист",
        description_ru="Найти плейлист по названию и включить его",
        timeout_ms=60_000,
    )

    class Params(ActionParams):
        name: str = Field(
            min_length=1,
            max_length=200,
            description="Название плейлиста",
        )

    def run(self, params: Params) -> ActionResult[str]:
        found = start_from_search(SearchKind.PLAYLIST, params.name)
        label = str(found.get("label") or params.name)
        return ActionResult.done(
            f"Включаю плейлист {label}.",
            value=label,
            detail=f"playlist {label!r} via {found.get('via')}",
            data={"name": params.name, "label": label},
        )


@register
class YMWave(Action):
    """Моя волна, with or without a preset — «включи мою волну», «волну по Сплину».

    ``reset_context`` is what makes plain «мою волну» actually plain: after an
    artist or a playlist the wave keeps playing in that context, and the app calls
    that Моя волна too.
    """

    meta: ClassVar = ActionMeta(
        name="YMWave",
        category=ActionCategory.MEDIA,
        title_ru="Моя волна",
        description_ru="Включить Мою волну, при желании — по артисту, жанру или настроению",
        timeout_ms=60_000,
    )

    class Params(ActionParams):
        preset: str = Field(
            default="",
            max_length=200,
            description="Пресет волны: «по артисту Сплин», «Радостно на душе»",
        )
        reset_context: bool = Field(
            default=True,
            description="Сбросить контекст, если волна играет по артисту или плейлисту",
        )

    def run(self, params: Params) -> ActionResult[str]:
        media = get_settings().actions.media
        script = _JS_WAVE.substitute(
            vibe_playerbar=_js(selector("vibe_playerbar")),
            vibe_play=_js(selector("vibe_play")),
            vibe_pause=_js(selector("vibe_pause")),
            vibe_reset=_js(selector("vibe_reset")),
            wheel_item=_js(selector("vibe_wheel_item")),
            nav_home=_js(selector("nav_home")),
            play_inner=_js(selector("play_inner")),
            preset=_js(params.preset),
            # A preset sets its own context, so resetting first would be undone
            # immediately.
            reset=_js(params.reset_context and not params.preset),
            wait=media.render_timeout_ms,
        )
        with open_music() as page:
            found = _outcome(page.evaluate(script), what="wave")
        preset = str(found.get("preset") or "")
        reset = str(found.get("reset") or "")
        message = f"Включаю волну: {preset}." if preset else "Включаю Мою волну."
        return ActionResult.done(
            message,
            value=preset or "Моя волна",
            detail=f"wave preset={preset!r} reset={reset!r} already={found.get('already')}",
            data={"preset": preset, "reset": reset},
        )


@register
class YMLike(Action):
    """Like the track that is playing — «лайк», «убери лайк»."""

    meta: ClassVar = ActionMeta(
        name="YMLike",
        category=ActionCategory.MEDIA,
        title_ru="Лайк текущего трека",
        description_ru="Поставить или снять лайк у играющего трека",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        mode: LikeMode = Field(
            default=LikeMode.LIKE,
            description="Поставить лайк, снять или переключить",
        )

    def run(self, params: Params) -> ActionResult[bool]:
        wanted = {LikeMode.LIKE: True, LikeMode.UNLIKE: False, LikeMode.TOGGLE: None}
        script = _JS_LIKE.substitute(
            playerbar_like=_js(selector("playerbar_like")),
            vibe_like=_js(selector("vibe_like")),
            want=_js(wanted[params.mode]),
        )
        with open_music() as page:
            found = _outcome(page.evaluate(script), what="like")
        liked = bool(found.get("liked"))
        changed = bool(found.get("changed"))
        track = _playing_track()
        about = f" — {track}" if track else ""
        if not changed:
            message = f"Лайк уже стоял{about}." if liked else f"Лайка и не было{about}."
        else:
            message = f"Поставил лайк{about}." if liked else f"Убрал лайк{about}."
        return ActionResult.done(
            message,
            value=liked,
            detail=f"like liked={liked} changed={changed} track={track!r}",
            data={"liked": liked, "changed": changed, "track": track},
        )


@register
class YMAddToPlaylist(Action):
    """Add the playing track to a playlist by name — «добавь в Любимку».

    Supports undo: «отмени» removes the track that was just added. The undo token
    records the matched playlist name and the track SMTC reported at add time; if a
    different track is playing when undo is called, the undo is refused — removing the
    wrong track without asking would be worse than the wrong track staying in.

    Both directions were run against the live app: adding, then undoing right away,
    leaves the playlist exactly as it was found. The click is symmetric — the same
    submenu item adds when the icon is absent and removes when it is there — and the
    only thing the undo needs from :data:`_SETTLE_S` is time for that icon to catch up.
    """

    meta: ClassVar = ActionMeta(
        name="YMAddToPlaylist",
        category=ActionCategory.MEDIA,
        title_ru="Добавить трек в плейлист",
        description_ru="Добавить играющий трек в плейлист по названию",
        timeout_ms=30_000,
        supports_undo=True,
    )

    class Params(ActionParams):
        playlist: str = Field(
            min_length=1,
            max_length=200,
            description="Название плейлиста, например «Любимка»",
        )

    def run(self, params: Params) -> ActionResult[str]:
        found = toggle_in_playlist(params.playlist, remove=False)
        label = str(found.get("label") or params.playlist)
        track = _playing_track() or "трек"
        changed = bool(found.get("changed"))
        message = (
            f"Добавил {track} в {label}." if changed else f"{track} уже был в плейлисте {label}."
        )
        # Token: playlist name and the track that was added, separated by |.
        # Both fields are stored in matched/SMTC form so the undo check is exact.
        token = f"{label}|{track}"
        return ActionResult.done(
            message,
            value=label,
            undo_token=token if changed else None,
            detail=f"add {track!r} to {label!r} changed={changed}",
            data={"playlist": label, "track": track, "changed": changed},
        )

    def undo(self, token: str) -> ActionResult[str]:
        """Remove the track that run() added, unless something else is playing now."""
        parts = token.split("|", 1)
        if len(parts) != 2:
            raise ActionError(
                f"malformed YMAddToPlaylist undo token {token!r}",
                user_message="Не помню, что именно добавлял — не могу отменить.",
            )
        playlist, track = parts
        now = _playing_track()
        if now and now != track:
            raise ActionError(
                f"undo refused: added {track!r} but now playing {now!r}",
                user_message=(
                    f"Сейчас играет «{now}», а не «{track}» — не буду убирать из плейлиста, "
                    "чтобы не ошибиться."
                ),
            )
        found = toggle_in_playlist(playlist, remove=True)
        label = str(found.get("label") or playlist)
        track_ru = now or track
        changed = bool(found.get("changed"))
        message = (
            f"Убрал {track_ru} из {label}."
            if changed
            else f"{track_ru} и так не было в плейлисте {label}."
        )
        return ActionResult.done(
            message,
            value=label,
            detail=f"undo add {track_ru!r} from {label!r} changed={changed}",
            data={"playlist": label, "track": track_ru, "changed": changed},
        )
