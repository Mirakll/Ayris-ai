"""The four parts of the «Голос» tab, each a section builder over one VoiceTab.

A section takes the tab in its constructor and builds its cards through the
tab's helpers, so every config write still goes through the one
:class:`~ayris.gui.tabs.base.SettingsTab` binding machinery. Splitting the page
this way keeps each part — recognition, synthesis, wake word, microphone — short
enough to read on its own.
"""

from ayris.gui.tabs.voice_sections.audio_input import AudioInputSection
from ayris.gui.tabs.voice_sections.stt import SttSection
from ayris.gui.tabs.voice_sections.tts import TtsSection
from ayris.gui.tabs.voice_sections.wake_word import WakeWordSection

__all__ = [
    "AudioInputSection",
    "SttSection",
    "TtsSection",
    "WakeWordSection",
]
