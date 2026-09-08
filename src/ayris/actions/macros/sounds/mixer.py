"""Policy and shared-output adapter for macro sounds."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from ayris.audio.tts.base import AudioChunk
from ayris.audio.tts.player import SpeechRequest, TtsPlayer
from ayris.core.errors import AudioError


class MixPolicy(StrEnum):
    QUEUE = "queue"
    DUCK = "duck"


class SoundOutput(Protocol):
    @property
    def speaking(self) -> bool: ...

    def submit(self, request: SpeechRequest) -> None: ...

    def cancel(self, request_id: str) -> bool: ...


class PlayerOutput:
    """Send sound requests through the TTS player, the sole device owner."""

    def __init__(self, player: TtsPlayer) -> None:
        self._player = player

    @property
    def speaking(self) -> bool:
        return self._player.speaking

    def submit(self, request: SpeechRequest) -> None:
        self._player.speak(request)

    def cancel(self, request_id: str) -> bool:
        return self._player.cancel(request_id)


@dataclass(frozen=True, slots=True)
class SoundHandle:
    request_id: str
    _done: threading.Event
    _output: SoundOutput

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def cancel(self) -> bool:
        return self._output.cancel(self.request_id)


class SoundMixer:
    """Apply queue/duck policy and per-sound gain above the common audio output."""

    def __init__(
        self,
        output: SoundOutput,
        *,
        policy: MixPolicy = MixPolicy.QUEUE,
        volume: float = 0.8,
        duck_db: float = -12.0,
        max_voices: int = 4,
    ) -> None:
        self._output = output
        self.policy = MixPolicy(policy)
        self.volume = min(1.0, max(0.0, volume))
        self.duck_db = min(0.0, duck_db)
        self.max_voices = max(1, max_voices)
        self._lock = threading.Lock()
        self._owners: dict[str, list[SoundHandle]] = {}

    def play(
        self,
        audio: AudioChunk,
        *,
        volume: float = 1.0,
        owner: str = "",
        wait: bool = False,
    ) -> SoundHandle:
        """Play one sound through the shared output according to the selected policy."""
        gain = self.volume * min(1.0, max(0.0, volume))
        request_id = f"macro-sound:{uuid4().hex}"
        done = threading.Event()
        handle = SoundHandle(request_id, done, self._output)
        with self._lock:
            voices = sum(len(items) for items in self._owners.values())
            if voices >= self.max_voices:
                raise AudioError(
                    "macro sound voice limit reached",
                    user_message="Слишком много звуков команд воспроизводится одновременно.",
                )
            self._owners.setdefault(owner, []).append(handle)

        def finish(_reason: str) -> None:
            done.set()
            with self._lock:
                items = self._owners.get(owner, [])
                if handle in items:
                    items.remove(handle)
                if not items:
                    self._owners.pop(owner, None)

        request = SpeechRequest(
            text="[звук команды]",
            chunks=(audio,),
            request_id=request_id,
            duration_estimate_ms=int(audio.duration_ms),
            gain=gain,
            mix=True,
            defer_while_speaking=self.policy is MixPolicy.QUEUE,
            duck_gain=10.0 ** (self.duck_db / 20.0),
            on_finished=finish,
        )
        self._output.submit(request)
        if wait:
            handle.wait()
        return handle

    def stop(self, owner: str = "") -> int:
        """Stop only macro-sound requests; ordinary speech is untouched."""
        with self._lock:
            selected = (
                list(self._owners.get(owner, ()))
                if owner
                else [handle for handles in self._owners.values() for handle in handles]
            )
        return sum(bool(handle.cancel()) for handle in selected)
