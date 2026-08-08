"""Speech-to-text engines: offline (Vosk, faster-whisper) and cloud providers.

Each engine implements the common ABC; the Auto router picks between online and
offline depending on connectivity.
"""

from __future__ import annotations
