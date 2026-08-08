"""Audio subsystem: capture, VAD, wake word, speech-to-text, text-to-speech.

Everything here runs in worker processes (see :mod:`ayris.core`), so a crash in
an engine cannot take the GUI down with it.
"""

from __future__ import annotations
