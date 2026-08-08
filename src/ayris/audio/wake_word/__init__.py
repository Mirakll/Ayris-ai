"""Wake word engines (openWakeWord, Porcupine, Vosk KWS).

Each engine is a separate module implementing the common ABC, so adding an
engine never requires touching calling code.
"""

from __future__ import annotations
