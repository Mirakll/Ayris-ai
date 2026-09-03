"""Media actions: what is playing, and telling the player what to do next.

Two levels, and the split is not ours — it is what the machine offers.

:mod:`ayris.actions.media.smtc` is the level that always works. Windows keeps a
registry of the players running on it (the thing the volume flyout draws), and it
answers «what is playing» and forwards play, pause, next and previous to the
player that owns the session. No debug port, no browser, nothing to launch: if
Яндекс Музыка is open at all, this works.

:mod:`ayris.actions.media.yandex_music` is the level that needs the desktop app
started with a debug port, and buys everything a transport control cannot express:
start *this* artist, this playlist, Моя волна, like the current track, add it to a
named playlist. It drives the app's own interface over CDP —
:mod:`ayris.actions.media.cdp` — by clicking real buttons in it.

**Nothing here presses a key.** Not the media keys, not letters: the person using
Ayris plays games, and a synthesised keystroke lands in whatever has focus. SMTC
talks to the player's session object and CDP clicks a DOM node; neither sends a
virtual-key code anywhere. The media-key path exists for players that expose no
SMTC session at all, and is off unless ``[actions.media] media_keys_fallback`` is
turned on.
"""

from __future__ import annotations
