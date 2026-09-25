"""The recommended local models, and whether one fits this machine.

Picking a local model is the one place a beginner needs a hand: a 7-billion
model that is wonderful on a gaming PC swaps itself to death on a laptop, and the
name alone does not say which is which. So this module is a short, curated list —
the models worth suggesting — each carrying what it costs to run and a verdict
computed against the memory actually present: «влезет», «влезет впритык» or
«не влезет».

**It is a recommendation layer, not the download catalogue.** The GGUF files Ayris
can download for :mod:`~ayris.nlu.llm.llamacpp_client` live in
``resources/models/llm.json`` (task 14, with real URLs and checksums); this list
is what the «ИИ» tab offers *before* a download, spanning Ollama tags too, which
are pulled by name and never downloaded through the model manager. A spec that
has a shipped GGUF names it in :attr:`LlmModelSpec.gguf_catalog_id` so the UI can
wire the two together.

**The sizes are for orientation, like the price table.** RAM and VRAM figures are
the ballpark a Q4_K_M quant of each model needs with a small context; they size a
recommendation, not an allocator, and the verdict keeps a headroom band so
«влезет» never means «влезет and nothing else runs».

**Memory detection is best-effort.** :func:`detect_total_ram_mb` asks psutil and
returns ``0`` when it cannot answer, which the verdict reads as «unknown» rather
than «nothing fits» — a missing reading must not hide every model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ayris.core.errors import LlmError
from ayris.utils.logger import get_logger

__all__ = [
    "RECOMMENDED",
    "LlmModelSpec",
    "LlmPurpose",
    "Verdict",
    "detect_total_ram_mb",
    "for_engine",
    "get_spec",
    "guard_ram_limit",
    "verdict_for",
    "verdict_label",
]

_log = get_logger(__name__)

#: Headroom left free above a model's own need before it counts as a comfortable
#: fit: the OS, Ayris itself and the browser the user is talking about all want
#: memory too, so «влезет» means «влезет с запасом», not «занимает всё до байта».
DEFAULT_HEADROOM_MB: Final = 2048


class LlmPurpose(StrEnum):
    """What a model is good for, so the picker can group by need rather than size."""

    CHAT = "chat"
    NLU = "nlu"
    WEAK_HARDWARE = "weak"


class Verdict(StrEnum):
    """Whether a model fits the machine's memory."""

    FITS = "fits"
    TIGHT = "tight"
    NO_FIT = "no_fit"
    UNKNOWN = "unknown"


_VERDICT_LABELS: Final[dict[Verdict, str]] = {
    Verdict.FITS: "влезет",
    Verdict.TIGHT: "влезет впритык",
    Verdict.NO_FIT: "не влезет",
    Verdict.UNKNOWN: "не удалось оценить память",
}


@dataclass(frozen=True, slots=True)
class LlmModelSpec:
    """One recommended model, with what it costs to run and how to reach it.

    The numbers describe a Q4_K_M quant with a small context and are meant for
    orientation, not allocation — enough to say «this laptop can run it» without
    promising an exact byte count.
    """

    id: str
    name: str
    ollama_tag: str
    params_b: float
    quantization: str
    file_size_bytes: int
    requires_ram_mb: int
    requires_vram_mb: int
    purposes: tuple[LlmPurpose, ...]
    engines: tuple[str, ...]
    description: str = ""
    gguf_catalog_id: str = ""
    language: str = "multi"

    @property
    def human_size(self) -> str:
        """The download size as a short «≈4.7 ГБ» string for the picker."""
        gb = self.file_size_bytes / (1024**3)
        if gb >= 1.0:
            return f"≈{gb:.1f} ГБ"
        mb = self.file_size_bytes / (1024**2)
        return f"≈{mb:.0f} МБ"

    def runs_on(self, engine: str) -> bool:
        """Whether ``engine`` (ollama / lmstudio / llamacpp) can run this model."""
        return engine.strip().lower() in self.engines


#: The curated shortlist offered in the «ИИ» tab, small on purpose. Ordered from
#: lightest to heaviest so the picker reads top-to-bottom as «weak hardware first».
#: Ollama tags are real and pulled by name; ``gguf_catalog_id`` cross-references
#: an entry in ``resources/models/llm.json`` when Ayris ships that GGUF for the
#: llama.cpp path (task 14). Sizes/RAM are Q4_K_M ballparks — see the module note.
RECOMMENDED: Final[tuple[LlmModelSpec, ...]] = (
    LlmModelSpec(
        id="llama3.2-1b",
        name="Llama 3.2 1B",
        ollama_tag="llama3.2:1b",
        params_b=1.24,
        quantization="Q4_K_M",
        file_size_bytes=808 * 1024 * 1024,
        requires_ram_mb=2048,
        requires_vram_mb=2048,
        purposes=(LlmPurpose.WEAK_HARDWARE, LlmPurpose.NLU),
        engines=("ollama", "llamacpp"),
        description="Самая лёгкая: понимает команды даже на слабом ноутбуке.",
    ),
    LlmModelSpec(
        id="qwen2.5-1.5b",
        name="Qwen2.5 1.5B Instruct",
        ollama_tag="qwen2.5:1.5b",
        params_b=1.54,
        quantization="Q4_K_M",
        file_size_bytes=1_120 * 1024 * 1024,
        requires_ram_mb=2560,
        requires_vram_mb=2560,
        purposes=(LlmPurpose.WEAK_HARDWARE, LlmPurpose.NLU),
        engines=("ollama", "llamacpp"),
        description="Крепкий разбор команд при минимальных требованиях.",
        gguf_catalog_id="qwen25-1_5b-instruct-q4",
    ),
    LlmModelSpec(
        id="gemma2-2b",
        name="Gemma 2 2B",
        ollama_tag="gemma2:2b",
        params_b=2.61,
        quantization="Q4_K_M",
        file_size_bytes=1_710 * 1024 * 1024,
        requires_ram_mb=3072,
        requires_vram_mb=3072,
        purposes=(LlmPurpose.WEAK_HARDWARE, LlmPurpose.CHAT),
        engines=("ollama", "llamacpp"),
        description="Живой разговор на скромной машине.",
    ),
    LlmModelSpec(
        id="qwen2.5-3b",
        name="Qwen2.5 3B Instruct",
        ollama_tag="qwen2.5:3b",
        params_b=3.09,
        quantization="Q4_K_M",
        file_size_bytes=1_930 * 1024 * 1024,
        requires_ram_mb=4096,
        requires_vram_mb=4096,
        purposes=(LlmPurpose.CHAT, LlmPurpose.NLU),
        engines=("ollama", "llamacpp"),
        description="Хороший баланс: разговор и точные команды.",
        gguf_catalog_id="qwen25-3b-instruct-q4",
    ),
    LlmModelSpec(
        id="llama3.2-3b",
        name="Llama 3.2 3B",
        ollama_tag="llama3.2:3b",
        params_b=3.21,
        quantization="Q4_K_M",
        file_size_bytes=2_020 * 1024 * 1024,
        requires_ram_mb=4096,
        requires_vram_mb=4096,
        purposes=(LlmPurpose.CHAT, LlmPurpose.NLU),
        engines=("ollama", "llamacpp"),
        description="Внятный собеседник среднего размера.",
    ),
    LlmModelSpec(
        id="phi3-mini",
        name="Phi-3 Mini 3.8B",
        ollama_tag="phi3:mini",
        params_b=3.82,
        quantization="Q4_K_M",
        file_size_bytes=2_390 * 1024 * 1024,
        requires_ram_mb=4608,
        requires_vram_mb=4608,
        purposes=(LlmPurpose.CHAT, LlmPurpose.NLU),
        engines=("ollama", "llamacpp"),
        description="Сильная логика при компактном размере.",
    ),
    LlmModelSpec(
        id="qwen2.5-7b",
        name="Qwen2.5 7B Instruct",
        ollama_tag="qwen2.5:7b",
        params_b=7.62,
        quantization="Q4_K_M",
        file_size_bytes=4_680 * 1024 * 1024,
        requires_ram_mb=6656,
        requires_vram_mb=6144,
        purposes=(LlmPurpose.CHAT,),
        engines=("ollama", "llamacpp"),
        description="Разговор ближе к облачному — для крепкого железа.",
        gguf_catalog_id="qwen25-7b-instruct-q4",
    ),
    LlmModelSpec(
        id="gemma2-9b",
        name="Gemma 2 9B",
        ollama_tag="gemma2:9b",
        params_b=9.24,
        quantization="Q4_K_M",
        file_size_bytes=5_760 * 1024 * 1024,
        requires_ram_mb=8704,
        requires_vram_mb=8192,
        purposes=(LlmPurpose.CHAT,),
        engines=("ollama", "llamacpp"),
        description="Самая умная из списка — нужен игровой ПК.",
    ),
)

_BY_ID: Final[dict[str, LlmModelSpec]] = {spec.id: spec for spec in RECOMMENDED}
_BY_TAG: Final[dict[str, LlmModelSpec]] = {spec.ollama_tag: spec for spec in RECOMMENDED}


def get_spec(id_or_tag: str) -> LlmModelSpec | None:
    """Return the recommended spec matching an id or an Ollama tag, else ``None``."""
    key = id_or_tag.strip()
    return _BY_ID.get(key) or _BY_TAG.get(key)


def for_engine(engine: str) -> tuple[LlmModelSpec, ...]:
    """The recommended specs runnable on ``engine`` (ollama / lmstudio / llamacpp)."""
    return tuple(spec for spec in RECOMMENDED if spec.runs_on(engine))


def verdict_for(
    spec: LlmModelSpec,
    total_ram_mb: int,
    *,
    headroom_mb: int = DEFAULT_HEADROOM_MB,
) -> Verdict:
    """Judge whether ``spec`` fits a machine with ``total_ram_mb`` of memory.

    ``total_ram_mb <= 0`` means the reading failed (see :func:`detect_total_ram_mb`)
    and yields :attr:`Verdict.UNKNOWN` rather than «не влезет» — a missing reading
    must not hide every model. Otherwise: it *fits* when the model's need plus a
    headroom band still leaves room, *tight* when it fits but eats the headroom,
    and *no fit* when the model alone is larger than the machine.
    """
    if total_ram_mb <= 0:
        return Verdict.UNKNOWN
    need = spec.requires_ram_mb
    if total_ram_mb < need:
        return Verdict.NO_FIT
    if total_ram_mb < need + headroom_mb:
        return Verdict.TIGHT
    return Verdict.FITS


def verdict_label(verdict: Verdict) -> str:
    """The Russian phrase shown next to a model in the picker."""
    return _VERDICT_LABELS[verdict]


def guard_ram_limit(spec: LlmModelSpec, *, ram_limit_mb: int) -> None:
    """Refuse to load ``spec`` when it needs more than the configured RAM cap.

    ``ram_limit_mb <= 0`` is «no cap» (§12) and never blocks. A positive limit
    smaller than the model's own need raises a typed :class:`LlmError` whose
    ``user_message`` explains the refusal in plain Russian, so the «ИИ» tab can
    show it instead of letting a too-large model swap the machine to a halt.
    """
    if ram_limit_mb <= 0:
        return
    if spec.requires_ram_mb <= ram_limit_mb:
        return
    raise LlmError(
        f"model {spec.id!r} needs {spec.requires_ram_mb} MB > RAM limit {ram_limit_mb} MB",
        user_message=(
            f"Модель «{spec.name}» требует около {spec.requires_ram_mb} МБ, "
            f"а лимит памяти в настройках — {ram_limit_mb} МБ. "
            "Выберите модель полегче или поднимите лимит в разделе «Производительность»."
        ),
    )


def detect_total_ram_mb() -> int:
    """Total physical RAM in mebibytes, or ``0`` when it cannot be measured.

    Best-effort by design: psutil may be missing or refuse to answer, and a
    missing reading is reported as ``0`` (→ :attr:`Verdict.UNKNOWN`) rather than
    guessed, so the picker degrades to «не удалось оценить» instead of «ничего не
    влезет».
    """
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
    except Exception:
        _log.debug("не удалось определить объём RAM", exc_info=True)
        return 0
    return total // (1024 * 1024)
