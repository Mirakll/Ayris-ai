"""Чистая логика аудио-слоя без единого реального устройства.

Три подсистемы здесь проверяются на голом коде, без PortAudio, без нативных
библиотек синтеза и без весов моделей:

* :mod:`ayris.audio.tts.quota` — учёт символов и предупреждение о лимите
  облачного синтеза. Живёт на in-memory SQLite и на несвязанной шине событий,
  так что запись, порог сброса, предупреждение «один раз за период» и уровни
  уведомлений наблюдаемы напрямую.
* :mod:`ayris.audio.devices` — обёртки над ``sounddevice``. Реальный модуль
  подменяется фейком в ``sys.modules``, а классы потоков конструируются поверх
  фейкового сырого потока, поэтому ветки старта/остановки/закрытия/записи и
  перечисления устройств проходятся, ни разу не тронув звуковую карту.
* :mod:`ayris.audio.tts.player` — добор веток микширования, отмены по
  идентификатору и подмены устройства через фейковый :class:`PlaybackBackend`.

Всё, что осталось за кадром, требует настоящего PortAudio (например, что
``abort`` действительно роняет буфер драйвера) и вынесено в тесты с маркером
``hardware``.
"""

from __future__ import annotations

import sys
import types
from array import array
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest

from ayris.audio.devices import (
    AudioDevice,
    DeviceDirection,
    PlaybackRequest,
    RawDevice,
    SoundDeviceBackend,
    StreamRequest,
    _SoundDeviceOutput,
    _SoundDeviceStream,
)
from ayris.audio.tts.base import AudioChunk
from ayris.audio.tts.player import (
    PlaybackReason,
    SpeechRequest,
    TtsPlayer,
    _MixVoice,
    _Stream,
)
from ayris.audio.tts.quota import (
    FLUSH_AFTER_CHARS,
    FLUSH_AFTER_SEC,
    QuotaTracker,
    UsageRow,
    current_period,
)
from ayris.core.database import Database
from ayris.core.errors import AudioError
from ayris.core.events import EventBus, NotificationRequested
from ayris.core.models import utc_now

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

pytestmark = pytest.mark.unit


# ======================================================================
# ayris.audio.tts.quota — учёт символов и предупреждение о лимите
# ======================================================================


@pytest.fixture
def database() -> Iterator[Database]:
    """In-memory SQLite со схемой; закрывается после теста, без файла профиля."""
    db = Database.open(":memory:")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def bus() -> EventBus:
    """Несвязанная шина: обработчики выполняются на месте, синхронно."""
    return EventBus(thread_id=None)


def _collect(bus: EventBus) -> list[NotificationRequested]:
    """Подписать сильный приёмник и вернуть растущий список уведомлений."""
    received: list[NotificationRequested] = []
    bus.subscribe(NotificationRequested, received.append, weak=False)
    return received


def _stored(database: Database, provider: str) -> int | None:
    """Символы провайдера прямо из таблицы, минуя буфер трекера."""
    row = database.query_one("SELECT characters FROM tts_usage WHERE provider = ?", (provider,))
    return None if row is None else int(row["characters"])


class TestCurrentPeriod:
    """Месяц выставления счёта как ``YYYY-MM`` — всегда в UTC."""

    def test_a_naive_datetime_is_read_as_utc(self) -> None:
        assert current_period(datetime(2024, 3, 15, 12, 0)) == "2024-03"

    def test_an_aware_datetime_is_converted_to_utc(self) -> None:
        # +10 в первую минуту апреля — это ещё март по UTC.
        moment = datetime(2024, 4, 1, 5, 0, tzinfo=timezone(timedelta(hours=10)))
        assert current_period(moment) == "2024-03"

    def test_a_utc_datetime_keeps_its_month(self) -> None:
        assert current_period(datetime(2024, 7, 9, tzinfo=UTC)) == "2024-07"

    def test_the_default_is_the_current_utc_month(self) -> None:
        assert current_period() == datetime.now(UTC).strftime("%Y-%m")


class TestUsageRow:
    """Доля израсходованного от лимита."""

    def test_share_of_a_positive_limit(self) -> None:
        assert UsageRow("cloud", "2024-01", characters=800).share_of(1000) == 0.8

    def test_share_of_no_limit_is_zero(self) -> None:
        row = UsageRow("cloud", "2024-01", characters=800)
        assert row.share_of(0) == 0.0
        assert row.share_of(-5) == 0.0


class TestQuotaRecording:
    """Буферизация в памяти и сброс в SQLite: пороги, накопление, ошибки."""

    def test_a_small_record_waits_for_an_explicit_flush(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", 5)
        assert _stored(database, "cloud") is None
        tracker.flush()
        assert tracker.usage("cloud").characters == 5

    def test_the_character_threshold_forces_a_flush(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", FLUSH_AFTER_CHARS)
        assert _stored(database, "cloud") == FLUSH_AFTER_CHARS

    def test_an_old_buffer_flushes_on_time(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker._since = utc_now() - timedelta(seconds=FLUSH_AFTER_SEC + 5)
        tracker.record("cloud", 10)
        assert _stored(database, "cloud") == 10

    def test_blank_provider_and_nonpositive_counts_are_ignored(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("", 100)
        tracker.record("cloud", 0)
        tracker.record("cloud", -5)
        tracker.flush()
        assert tracker.all_usage() == ()

    def test_flushes_accumulate_across_records(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", 30)
        tracker.flush()
        tracker.record("cloud", 12)
        tracker.flush()
        row = tracker.usage("cloud")
        assert row.characters == 42
        assert row.requests == 2

    def test_flush_with_nothing_pending_is_safe(self, database: Database) -> None:
        QuotaTracker(database).flush()
        row = database.query_one("SELECT COUNT(*) AS n FROM tts_usage")
        assert row is not None
        assert row["n"] == 0

    def test_close_flushes_and_is_idempotent(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", 7)
        tracker.close()
        tracker.close()
        assert _stored(database, "cloud") == 7

    def test_a_database_error_drops_the_counts_without_raising(
        self, database: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", 500)

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("диск заполнен")

        monkeypatch.setattr(database, "executemany", boom)
        tracker.flush()  # ошибка проглочена, буфер очищен
        assert tracker.usage("cloud").characters == 0


class TestQuotaReading:
    """Чтение расхода: неизвестный провайдер, явный период, порядок, остаток."""

    def test_usage_is_zero_for_an_unseen_provider(self, database: Database) -> None:
        row = QuotaTracker(database).usage("nobody")
        assert row == UsageRow("nobody", current_period())

    def test_usage_counts_a_pending_record(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", 42)
        row = tracker.usage("cloud")
        assert row.characters == 42
        assert row.requests == 1

    def test_usage_reads_an_explicit_period(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("cloud", 10)
        tracker.flush()
        assert tracker.usage("cloud", current_period()).characters == 10
        assert tracker.usage("cloud", "1970-01").characters == 0

    def test_all_usage_is_sorted_by_spend(self, database: Database) -> None:
        tracker = QuotaTracker(database)
        tracker.record("a", 100)
        tracker.record("b", 300)
        tracker.record("c", 200)
        tracker.flush()
        rows = tracker.all_usage()
        assert [row.provider for row in rows] == ["b", "c", "a"]
        assert [row.characters for row in rows] == [300, 200, 100]

    def test_remaining_is_negative_without_a_limit(self, database: Database) -> None:
        assert QuotaTracker(database).remaining("cloud") == -1

    def test_remaining_counts_down_from_the_limit(self, database: Database) -> None:
        tracker = QuotaTracker(database, limits={"cloud": 1000})
        tracker.record("cloud", 300)
        assert tracker.remaining("cloud") == 700

    def test_remaining_never_reports_below_zero(self, database: Database) -> None:
        tracker = QuotaTracker(database, limits={"cloud": 1000})
        tracker.record("cloud", 1500)
        assert tracker.remaining("cloud") == 0


class TestQuotaWarnings:
    """Предупреждение о лимите: один раз за период, уровень, форматирование."""

    def test_a_warning_fires_when_the_ratio_is_crossed(
        self, database: Database, bus: EventBus
    ) -> None:
        received = _collect(bus)
        tracker = QuotaTracker(database, bus=bus, limits={"cloud": 1000})
        tracker.record("cloud", 800)
        tracker.flush()
        assert len(received) == 1
        note = received[0]
        assert note.level == "warning"
        assert note.title == "Лимит синтеза речи: cloud"
        assert "(80%)" in note.message
        assert "Осталось немного." in note.message

    def test_thousands_are_grouped_with_spaces(self, database: Database, bus: EventBus) -> None:
        received = _collect(bus)
        tracker = QuotaTracker(database, bus=bus, limits={"cloud": 10000})
        tracker.record("cloud", 8000)  # >= FLUSH_AFTER_CHARS: сброс сам собой
        assert len(received) == 1
        assert "8 000" in received[0].message
        assert "10 000" in received[0].message

    def test_an_exhausted_quota_is_reported_as_an_error(
        self, database: Database, bus: EventBus
    ) -> None:
        received = _collect(bus)
        tracker = QuotaTracker(database, bus=bus, limits={"cloud": 1000})
        tracker.record("cloud", 1000)
        tracker.flush()
        assert len(received) == 1
        assert received[0].level == "error"
        assert "Синтез переключится на локальный голос." in received[0].message

    def test_the_warning_fires_once_until_the_limits_change(
        self, database: Database, bus: EventBus
    ) -> None:
        received = _collect(bus)
        tracker = QuotaTracker(database, bus=bus, limits={"cloud": 1000})
        tracker.record("cloud", 800)
        tracker.flush()
        assert len(received) == 1
        tracker.record("cloud", 50)
        tracker.flush()
        assert len(received) == 1  # тот же период — молчок
        tracker.set_limits({"cloud": 1000})  # сбрасывает отметку «уже предупредили»
        tracker.record("cloud", 50)
        tracker.flush()
        assert len(received) == 2

    def test_a_tracker_without_a_bus_still_records(self, database: Database) -> None:
        tracker = QuotaTracker(database, limits={"cloud": 1000})
        tracker.record("cloud", 900)
        assert tracker.usage("cloud").characters == 900

    def test_a_limit_without_a_usage_row_warns_nobody(
        self, database: Database, bus: EventBus
    ) -> None:
        received = _collect(bus)
        tracker = QuotaTracker(database, bus=bus, limits={"ghost": 500})
        tracker._check_limit_locked("ghost", "1999-01")
        assert received == []


# ======================================================================
# ayris.audio.devices — обёртки над sounddevice без единого устройства
# ======================================================================


class _FakeRawStream:
    """Замена ``sounddevice.RawInputStream``/``RawOutputStream``.

    Каждый метод отмечает, что его вызвали, и по требованию бросает
    исключение — так проходятся и успешные, и провальные ветки обёрток без
    единого обращения к PortAudio.
    """

    def __init__(
        self,
        *,
        active: bool = True,
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self._active = active
        self._fail = fail
        self.on_write: Callable[[], None] | None = None
        self.started = 0
        self.stopped = 0
        self.aborted = 0
        self.closed = 0
        self.close_ignore_errors: bool | None = None
        self.abort_ignore_errors: bool | None = None
        self.writes: list[bytes] = []

    @property
    def active(self) -> bool:
        if "active" in self._fail:
            raise RuntimeError("устройство отвалилось")
        return self._active

    def start(self) -> None:
        self.started += 1
        if "start" in self._fail:
            raise RuntimeError("не запускается")

    def stop(self) -> None:
        self.stopped += 1
        if "stop" in self._fail:
            raise RuntimeError("не останавливается")

    def abort(self, *, ignore_errors: bool = False) -> None:
        self.aborted += 1
        self.abort_ignore_errors = ignore_errors
        if "abort" in self._fail:
            raise RuntimeError("не прерывается")

    def write(self, pcm: bytes) -> None:
        if self.on_write is not None:
            self.on_write()
        if "write" in self._fail:
            raise RuntimeError("запись не удалась")
        self.writes.append(bytes(pcm))

    def close(self, *, ignore_errors: bool = False) -> None:
        self.closed += 1
        self.close_ignore_errors = ignore_errors
        if "close" in self._fail:
            raise RuntimeError("не закрывается")


def _ignore_callback(_pcm: bytes, _overflow: bool) -> None:
    """Приёмник захвата, который выбрасывает всё, что ему дали."""


class TestDeviceMetadata:
    """Направление устройства и подпись в комбобоксе."""

    def test_output_names_its_role_and_hint(self) -> None:
        assert DeviceDirection.OUTPUT.role == "воспроизведения"
        assert "наушники" in DeviceDirection.OUTPUT.missing_hint.lower()

    def test_input_names_its_role_and_hint(self) -> None:
        assert DeviceDirection.INPUT.role == "записи"
        assert "микрофон" in DeviceDirection.INPUT.missing_hint.lower()

    def test_a_label_without_a_host_api_is_just_the_name(self) -> None:
        device = AudioDevice(id="x", name="Наушники", direction=DeviceDirection.OUTPUT, index=0)
        assert device.label == "Наушники"

    def test_a_label_with_a_host_api_is_annotated(self) -> None:
        device = AudioDevice(
            id="x",
            name="Наушники",
            direction=DeviceDirection.OUTPUT,
            index=0,
            host_api="WASAPI",
        )
        assert device.label == "Наушники (WASAPI)"


class TestSoundDeviceStream:
    """Обёртка захвата: формат, живость, старт/стоп/закрытие и их ошибки."""

    def test_it_exposes_the_negotiated_format(self) -> None:
        stream = _SoundDeviceStream(_FakeRawStream(), 16000, 1)
        assert stream.sample_rate == 16000
        assert stream.channels == 1

    def test_active_reflects_the_raw_stream(self) -> None:
        assert _SoundDeviceStream(_FakeRawStream(active=True), 16000, 1).active is True

    def test_active_is_false_once_closed(self) -> None:
        stream = _SoundDeviceStream(_FakeRawStream(active=True), 16000, 1)
        stream.close()
        assert stream.active is False

    def test_active_swallows_a_dead_device(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"active"}))
        assert _SoundDeviceStream(raw, 16000, 1).active is False

    def test_start_runs_the_device(self) -> None:
        raw = _FakeRawStream()
        _SoundDeviceStream(raw, 16000, 1).start()
        assert raw.started == 1

    def test_start_wraps_a_failure_as_audio_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"start"}))
        with pytest.raises(AudioError, match="cannot start audio stream"):
            _SoundDeviceStream(raw, 16000, 1).start()

    def test_stop_keeps_the_device_open(self) -> None:
        raw = _FakeRawStream()
        _SoundDeviceStream(raw, 16000, 1).stop()
        assert raw.stopped == 1

    def test_stop_is_a_noop_after_close(self) -> None:
        raw = _FakeRawStream()
        stream = _SoundDeviceStream(raw, 16000, 1)
        stream.close()
        stream.stop()
        assert raw.stopped == 0

    def test_stop_swallows_a_device_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"stop"}))
        _SoundDeviceStream(raw, 16000, 1).stop()
        assert raw.stopped == 1

    def test_close_is_idempotent_and_ignores_errors(self) -> None:
        raw = _FakeRawStream()
        stream = _SoundDeviceStream(raw, 16000, 1)
        stream.close()
        stream.close()
        assert raw.closed == 1
        assert raw.close_ignore_errors is True

    def test_close_swallows_a_device_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"close"}))
        _SoundDeviceStream(raw, 16000, 1).close()
        assert raw.closed == 1


class TestSoundDeviceOutput:
    """Обёртка воспроизведения: запись, пропуск закрытого потока, abort."""

    def test_it_exposes_the_negotiated_format(self) -> None:
        out = _SoundDeviceOutput(_FakeRawStream(), 48000, 2)
        assert out.sample_rate == 48000
        assert out.channels == 2

    def test_active_reflects_the_raw_stream(self) -> None:
        assert _SoundDeviceOutput(_FakeRawStream(active=True), 48000, 2).active is True

    def test_active_is_false_once_closed(self) -> None:
        out = _SoundDeviceOutput(_FakeRawStream(active=True), 48000, 2)
        out.close()
        assert out.active is False

    def test_active_swallows_a_dead_device(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"active"}))
        assert _SoundDeviceOutput(raw, 48000, 2).active is False

    def test_start_runs_the_device(self) -> None:
        raw = _FakeRawStream()
        _SoundDeviceOutput(raw, 48000, 2).start()
        assert raw.started == 1

    def test_start_wraps_a_failure_as_audio_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"start"}))
        with pytest.raises(AudioError, match="cannot start output stream"):
            _SoundDeviceOutput(raw, 48000, 2).start()

    def test_writing_nothing_is_skipped(self) -> None:
        raw = _FakeRawStream()
        _SoundDeviceOutput(raw, 48000, 2).write(b"")
        assert raw.writes == []

    def test_writing_to_a_closed_stream_is_skipped(self) -> None:
        raw = _FakeRawStream()
        out = _SoundDeviceOutput(raw, 48000, 2)
        out.close()
        out.write(b"\x01\x00")
        assert raw.writes == []

    def test_a_write_reaches_the_device(self) -> None:
        raw = _FakeRawStream()
        _SoundDeviceOutput(raw, 48000, 2).write(b"\x01\x00")
        assert raw.writes == [b"\x01\x00"]

    def test_a_lost_device_becomes_an_audio_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"write"}))
        with pytest.raises(AudioError, match="cannot write to output stream"):
            _SoundDeviceOutput(raw, 48000, 2).write(b"\x01\x00")

    def test_a_write_racing_a_close_is_swallowed(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"write"}))
        out = _SoundDeviceOutput(raw, 48000, 2)
        raw.on_write = out.close  # закрытие успевает до броска исключения
        out.write(b"\x01\x00")  # ошибки нет: поток уже помечен закрытым

    def test_stop_aborts_the_buffer(self) -> None:
        raw = _FakeRawStream()
        _SoundDeviceOutput(raw, 48000, 2).stop()
        assert raw.aborted == 1
        assert raw.abort_ignore_errors is True

    def test_stop_is_a_noop_after_close(self) -> None:
        raw = _FakeRawStream()
        out = _SoundDeviceOutput(raw, 48000, 2)
        out.close()
        out.stop()
        assert raw.aborted == 0

    def test_stop_swallows_a_device_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"abort"}))
        _SoundDeviceOutput(raw, 48000, 2).stop()
        assert raw.aborted == 1

    def test_close_is_idempotent_and_ignores_errors(self) -> None:
        raw = _FakeRawStream()
        out = _SoundDeviceOutput(raw, 48000, 2)
        out.close()
        out.close()
        assert raw.closed == 1
        assert raw.close_ignore_errors is True

    def test_close_swallows_a_device_error(self) -> None:
        raw = _FakeRawStream(fail=frozenset({"close"}))
        _SoundDeviceOutput(raw, 48000, 2).close()
        assert raw.closed == 1


class _FakeSd:
    """Замена модуля ``sounddevice`` в ``sys.modules``.

    Отдаёт заранее заданные списки устройств и хост-API, считает пересканы и
    открытия потоков и по требованию роняет любой из вызовов PortAudio.
    """

    def __init__(
        self,
        *,
        devices: list[dict[str, Any]] | None = None,
        host_apis: list[dict[str, Any]] | None = None,
        default_device: tuple[Any, Any] = (0, 1),
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self._devices = devices if devices is not None else []
        self._host_apis = host_apis if host_apis is not None else []
        self.default = types.SimpleNamespace(device=default_device)
        self._fail = fail
        self.rescans = 0
        self.reinits = 0
        self.last_input_callback: Callable[..., None] | None = None
        self.last_output: _FakeRawStream | None = None
        self.checked: list[dict[str, Any]] = []

    def query_devices(self) -> list[dict[str, Any]]:
        if "query" in self._fail:
            raise RuntimeError("PortAudio упал")
        return self._devices

    def query_hostapis(self) -> list[dict[str, Any]]:
        return self._host_apis

    def _terminate(self) -> None:
        if "rescan" in self._fail:
            raise RuntimeError("terminate упал")
        self.rescans += 1

    def _initialize(self) -> None:
        self.reinits += 1

    def RawInputStream(self, **kwargs: Any) -> _FakeRawStream:  # noqa: N802 - зеркалит sounddevice
        if "open_input" in self._fail:
            raise RuntimeError("устройство занято")
        self.last_input_callback = kwargs.get("callback")
        return _FakeRawStream()

    def check_input_settings(self, **kwargs: Any) -> None:
        self.checked.append(kwargs)
        if "unsupported" in self._fail:
            raise ValueError("формат не поддерживается")

    def RawOutputStream(self, **kwargs: Any) -> _FakeRawStream:  # noqa: N802 - зеркалит sounddevice
        if "open_output" in self._fail:
            raise RuntimeError("устройство занято")
        stream = _FakeRawStream()
        self.last_output = stream
        return stream


def _backend_with(monkeypatch: pytest.MonkeyPatch, sd: _FakeSd) -> SoundDeviceBackend:
    """Подсунуть фейковый модуль в ``sys.modules`` и вернуть свежий бэкенд."""
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return SoundDeviceBackend()


class TestSoundDeviceBackend:
    """Бэкенд PortAudio через подменённый ``sounddevice``."""

    def test_a_missing_library_is_an_audio_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        with pytest.raises(AudioError, match="unavailable"):
            SoundDeviceBackend().raw_devices()

    def test_devices_are_mapped_from_the_library(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd(
            devices=[
                {
                    "name": "Микрофон",
                    "hostapi": 0,
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 44100.0,
                },
                {
                    "name": "Динамики",
                    "hostapi": 0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48000.0,
                },
            ],
            host_apis=[{"name": "MME"}],
            default_device=(0, 1),
        )
        devices = _backend_with(monkeypatch, sd).raw_devices()
        assert len(devices) == 2
        assert devices[0].name == "Микрофон"
        assert devices[0].host_api == "MME"
        assert devices[0].max_input_channels == 2
        assert devices[0].default_input is True
        assert devices[1].default_output is True
        assert devices[1].default_sample_rate == 48000.0

    def test_a_missing_name_and_host_api_get_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sd = _FakeSd(
            devices=[{"max_input_channels": 1}, {"name": "X", "hostapi": 9}],
            host_apis=[{"name": "MME"}],
            default_device=(-1, -1),
        )
        devices = _backend_with(monkeypatch, sd).raw_devices()
        assert devices[0].name == "device 0"
        assert devices[0].host_api == ""  # индекс -1: нет хост-API
        assert devices[1].host_api == ""  # индекс 9 вне диапазона
        assert all(not d.default_input and not d.default_output for d in devices)

    def test_bad_defaults_fall_back_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd(
            devices=[{"name": "A", "hostapi": 0, "max_output_channels": 2}],
            host_apis=[{"name": "MME"}],
            default_device=("x", "y"),
        )
        devices = _backend_with(monkeypatch, sd).raw_devices()
        assert devices[0].default_output is False

    def test_an_enumeration_failure_is_an_audio_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sd = _FakeSd(fail=frozenset({"query"}))
        with pytest.raises(AudioError, match="cannot enumerate audio devices"):
            _backend_with(monkeypatch, sd).raw_devices()

    def test_the_module_is_imported_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd(devices=[], host_apis=[])
        backend = _backend_with(monkeypatch, sd)
        backend.raw_devices()
        backend.raw_devices()
        assert backend._module is sd  # второй вызов берёт закэшированный модуль

    def test_refresh_reinitialises_portaudio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd()
        _backend_with(monkeypatch, sd).refresh()
        assert sd.rescans == 1
        assert sd.reinits == 1

    def test_refresh_swallows_a_rescan_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd(fail=frozenset({"rescan"}))
        _backend_with(monkeypatch, sd).refresh()  # без исключения
        assert sd.reinits == 0

    def test_opening_an_input_stream_shims_the_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sd = _FakeSd()
        backend = _backend_with(monkeypatch, sd)
        received: list[tuple[bytes, bool]] = []

        def sink(pcm: bytes, overflow: bool) -> None:
            received.append((pcm, overflow))

        request = StreamRequest(device_index=0, sample_rate=16000, channels=1, block_frames=320)
        stream = backend.open_input_stream(request, sink)
        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert sd.last_input_callback is not None
        sd.last_input_callback(b"abcd", 2, None, 7)
        assert received == [(b"abcd", True)]

    def test_opening_an_input_stream_wraps_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd(fail=frozenset({"open_input"}))
        backend = _backend_with(monkeypatch, sd)
        request = StreamRequest(device_index=3, sample_rate=16000, channels=1, block_frames=320)
        with pytest.raises(AudioError, match="cannot open device 3"):
            backend.open_input_stream(request, _ignore_callback)

    def test_supports_rate_is_true_when_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd()
        backend = _backend_with(monkeypatch, sd)
        request = StreamRequest(device_index=0, sample_rate=16000, channels=1, block_frames=320)
        assert backend.supports_rate(request) is True
        assert sd.checked

    def test_supports_rate_is_false_when_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd(fail=frozenset({"unsupported"}))
        backend = _backend_with(monkeypatch, sd)
        request = StreamRequest(device_index=0, sample_rate=16000, channels=1, block_frames=320)
        assert backend.supports_rate(request) is False

    def test_opening_an_output_stream_starts_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = _FakeSd()
        backend = _backend_with(monkeypatch, sd)
        request = PlaybackRequest(device_index=1, sample_rate=48000, channels=2, block_frames=960)
        stream = backend.open_output_stream(request)
        assert stream.sample_rate == 48000
        assert stream.channels == 2
        assert sd.last_output is not None

    def test_opening_an_output_stream_wraps_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sd = _FakeSd(fail=frozenset({"open_output"}))
        backend = _backend_with(monkeypatch, sd)
        request = PlaybackRequest(device_index=2, sample_rate=48000, channels=2, block_frames=960)
        with pytest.raises(AudioError, match="cannot open output device 2"):
            backend.open_output_stream(request)


# ======================================================================
# ayris.audio.tts.player — детерминированные ветки без потока-писателя
# ======================================================================


def _tone(ms: int, sample_rate: int, *, level: int = 6000) -> bytes:
    """Ненулевой ``int16``-PCM нужной длины (постоянный отсчёт — этого хватает)."""
    frames = max(1, sample_rate * ms // 1000)
    return array("h", [level] * frames).tobytes()


class _FakeOutStream:
    """Минимальный :class:`OutputStream`: пишет в список, по флагу роняет запись."""

    def __init__(self, sample_rate: int, channels: int, *, fail_write: bool = False) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._fail_write = fail_write
        self.writes: list[bytes] = []
        self.started = 0
        self.stopped = 0
        self.closed = 0
        #: Побочный эффект перед записью — тест ставит отмену прямо в момент write.
        self.on_write: Callable[[], None] | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def active(self) -> bool:
        return True

    def start(self) -> None:
        self.started += 1

    def write(self, pcm: bytes) -> None:
        if self.on_write is not None:
            self.on_write()
        if self._fail_write:
            raise AudioError("устройство потеряно")
        self.writes.append(bytes(pcm))

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


class _FakeOutBackend:
    """Бэкенд с одним выходным устройством; поток нужен лишь для конструктора."""

    def __init__(self, *, channels: int = 1) -> None:
        self._channels = channels
        self.opened: list[PlaybackRequest] = []

    def raw_devices(self) -> tuple[RawDevice, ...]:
        return (
            RawDevice(
                index=0,
                name="Динамики",
                host_api="MME",
                max_output_channels=self._channels,
                default_output=True,
            ),
        )

    def refresh(self) -> None:
        pass

    def open_output_stream(self, request: PlaybackRequest) -> _FakeOutStream:
        self.opened.append(request)
        return _FakeOutStream(request.sample_rate, request.channels)


class TestPlayerConstruction:
    """Конструктор, наблюдатели и смена устройства без запуска потока."""

    def test_a_player_without_a_backend_builds_the_default(self) -> None:
        # _default_backend() строит SoundDeviceBackend, но PortAudio при этом
        # не трогается — модуль импортируется лениво, только при открытии потока.
        player = TtsPlayer(volume=0.5)
        assert player.volume == 0.5
        assert player.running is False

    def test_the_volume_is_clamped(self) -> None:
        assert TtsPlayer(_FakeOutBackend(), volume=5.0).volume == 1.0
        assert TtsPlayer(_FakeOutBackend(), volume=-1.0).volume == 0.0

    def test_set_observers_replaces_only_what_is_given(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        calls: list[str] = []

        def on_started(_request: SpeechRequest, _ms: int) -> None:
            calls.append("started")

        def on_finished(_request: SpeechRequest, _reason: str) -> None:
            calls.append("finished")

        player.set_observers(on_started=on_started, on_finished=on_finished)
        player.set_observers()  # оба None — ничего не меняет
        player._on_started(SpeechRequest(), 0)
        player._on_finished(SpeechRequest(), PlaybackReason.COMPLETED)
        assert calls == ["started", "finished"]

    def test_set_device_ignores_an_unchanged_spec(self) -> None:
        player = TtsPlayer(_FakeOutBackend(), device="")
        player.set_device("")  # то же значение — ранний выход
        assert player._device_spec == ""

    def test_set_device_logs_when_a_stream_is_open(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        player._stream = _Stream(
            stream=_FakeOutStream(48000, 2),
            device_id="d",
            sample_rate=48000,
            channels=2,
        )
        player.set_device("наушники")
        assert player._device_spec == "наушники"


class TestPlayerConverterCache:
    """Кэш ресемплера в :class:`_Stream` живёт между кусками одной фразы."""

    def test_the_resampler_is_reused_across_chunks(self) -> None:
        stream = _Stream(
            stream=_FakeOutStream(48000, 1), device_id="d", sample_rate=48000, channels=1
        )
        first = stream.converter_for(AudioChunk(_tone(20, 22050), sample_rate=22050))
        second = stream.converter_for(AudioChunk(_tone(20, 22050), sample_rate=22050))
        assert first is not None
        assert first is second  # тот же ресемплер — без щелчка на границе
        assert stream.converter_for(AudioChunk(_tone(20, 48000), sample_rate=48000)) is None


class TestPlayerFinishAndDrop:
    """Завершение и снятие с очереди: приватный колбэк, микс, публичный наблюдатель."""

    def test_a_missing_private_callback_is_a_noop(self) -> None:
        TtsPlayer._notify_private_finish(SpeechRequest(), PlaybackReason.COMPLETED)

    def test_a_raising_private_callback_is_swallowed(self) -> None:
        def boom(_reason: str) -> None:
            raise RuntimeError("нет")

        request = SpeechRequest(request_id="m", on_finished=boom)
        TtsPlayer._notify_private_finish(request, PlaybackReason.CANCELLED)  # без исключения

    def test_finishing_a_mix_request_skips_the_public_observer(self) -> None:
        public: list[str] = []
        private: list[str] = []

        def on_finished(_request: SpeechRequest, reason: str) -> None:
            public.append(reason)

        def note(reason: str) -> None:
            private.append(reason)

        player = TtsPlayer(_FakeOutBackend(), on_finished=on_finished)
        request = SpeechRequest(request_id="m", mix=True, on_finished=note)
        player._current = request
        player._finish(request, PlaybackReason.COMPLETED)
        assert private == [PlaybackReason.COMPLETED]
        assert public == []  # микс не идёт публичному наблюдателю
        assert player._current is None

    def test_cancel_by_id_aborts_the_current_mix(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        player._current = SpeechRequest(request_id="m1", mix=True)
        assert player.cancel("m1") is True
        assert player._cancel.is_set()

    def test_cancel_by_id_drops_matching_queued_phrases(self) -> None:
        dropped: list[str] = []

        def on_finished(request: SpeechRequest, _reason: str) -> None:
            dropped.append(request.request_id)

        player = TtsPlayer(_FakeOutBackend(), on_finished=on_finished)
        player._normal.append(SpeechRequest(request_id="keep"))
        player._normal.append(SpeechRequest(request_id="drop"))
        player._urgent.append(SpeechRequest(request_id="drop"))
        assert player.cancel("drop") is True
        assert sorted(dropped) == ["drop", "drop"]
        assert [item.request_id for item in player._normal] == ["keep"]

    def test_cancel_by_id_marks_a_running_mix_voice(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        player._mix_voices.append(_MixVoice(SpeechRequest(request_id="v1", mix=True), b""))
        assert player.cancel("v1") is True
        assert "v1" in player._cancelled_mix

    def test_reporting_a_dropped_mix_skips_the_public_observer(self) -> None:
        public: list[str] = []
        private: list[str] = []

        def on_finished(_request: SpeechRequest, reason: str) -> None:
            public.append(reason)

        def note(reason: str) -> None:
            private.append(reason)

        player = TtsPlayer(_FakeOutBackend(), on_finished=on_finished)
        player._report_dropped(SpeechRequest(request_id="m", mix=True, on_finished=note))
        assert private == [PlaybackReason.CANCELLED]
        assert public == []

    def test_reporting_a_dropped_phrase_swallows_a_raising_observer(self) -> None:
        def on_finished(_request: SpeechRequest, _reason: str) -> None:
            raise RuntimeError("нет")

        player = TtsPlayer(_FakeOutBackend(), on_finished=on_finished)
        player._report_dropped(SpeechRequest(request_id="n"))  # без исключения


class TestPlayerMixing:
    """Микширование макро-звуков в выходной поток, без потока-писателя."""

    def _stream(self, out: _FakeOutStream) -> _Stream:
        return _Stream(stream=out, device_id="d", sample_rate=48000, channels=1)

    def test_an_empty_mix_request_completes_immediately(self) -> None:
        reasons: list[str] = []

        def note(reason: str) -> None:
            reasons.append(reason)

        player = TtsPlayer(_FakeOutBackend())
        stream = self._stream(_FakeOutStream(48000, 1))
        request = SpeechRequest(
            request_id="e", mix=True, chunks=(AudioChunk(b""),), on_finished=note
        )
        player._mixed.append(request)
        player._activate_mix_voices(stream, allow_deferred=True)
        assert reasons == [PlaybackReason.COMPLETED]
        assert player._mix_voices == []

    def test_a_channel_mismatch_fails_the_mix_request(self) -> None:
        reasons: list[str] = []

        def note(reason: str) -> None:
            reasons.append(reason)

        player = TtsPlayer(_FakeOutBackend())
        stream = self._stream(_FakeOutStream(48000, 1))
        stereo = AudioChunk(b"\x00\x00\x00\x00", sample_rate=48000, channels=2)
        request = SpeechRequest(request_id="c", mix=True, chunks=(stereo,), on_finished=note)
        player._mixed.append(request)
        player._activate_mix_voices(stream, allow_deferred=True)
        assert reasons == [PlaybackReason.ERROR]
        assert player._mix_voices == []

    def test_a_mix_request_is_resampled_to_the_stream(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        stream = self._stream(_FakeOutStream(48000, 1))
        chunk = AudioChunk(_tone(40, 22050), sample_rate=22050)
        player._mixed.append(SpeechRequest(request_id="r", mix=True, chunks=(chunk,)))
        player._activate_mix_voices(stream, allow_deferred=True)
        assert len(player._mix_voices) == 1
        assert player._mix_voices[0].pcm  # ресемпл дал ненулевой звук

    def test_a_deferred_mix_request_waits_while_speaking(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        stream = self._stream(_FakeOutStream(48000, 1))
        chunk = AudioChunk(_tone(20, 48000), sample_rate=48000)
        player._mixed.append(
            SpeechRequest(request_id="d", mix=True, chunks=(chunk,), defer_while_speaking=True)
        )
        player._activate_mix_voices(stream, allow_deferred=False)
        assert player._mix_voices == []  # отложенный звук пока не звучит
        assert len(player._mixed) == 1

    def test_active_mix_voices_drain_in_silence(self) -> None:
        reasons: list[str] = []

        def note(reason: str) -> None:
            reasons.append(reason)

        player = TtsPlayer(_FakeOutBackend())
        player._running = True
        out = _FakeOutStream(48000, 1)
        player._stream = _Stream(stream=out, device_id="d", sample_rate=48000, channels=1)
        short = _tone(5, 48000)  # короче одного блока в 20 мс
        player._mix_voices.append(
            _MixVoice(SpeechRequest(request_id="v", mix=True, on_finished=note), short)
        )
        player._drain_mix_voices()
        assert reasons == [PlaybackReason.COMPLETED]
        assert player._mix_voices == []
        assert out.writes  # тишина с примешанным хвостом записана

    def test_a_lost_device_fails_the_draining_voices(self) -> None:
        reasons: list[str] = []

        def note(reason: str) -> None:
            reasons.append(reason)

        player = TtsPlayer(_FakeOutBackend())
        player._running = True
        out = _FakeOutStream(48000, 1, fail_write=True)
        player._stream = _Stream(stream=out, device_id="d", sample_rate=48000, channels=1)
        player._mix_voices.append(
            _MixVoice(SpeechRequest(request_id="v", mix=True, on_finished=note), _tone(40, 48000))
        )
        player._drain_mix_voices()
        assert reasons == [PlaybackReason.ERROR]
        assert player._mix_voices == []

    def test_a_cancelled_mix_voice_is_dropped_mid_block(self) -> None:
        reasons: list[str] = []

        def note(reason: str) -> None:
            reasons.append(reason)

        player = TtsPlayer(_FakeOutBackend())
        stream = self._stream(_FakeOutStream(48000, 1))
        voice = _MixVoice(
            SpeechRequest(request_id="v", mix=True, on_finished=note), _tone(40, 48000)
        )
        player._mix_voices.append(voice)
        player._cancelled_mix.add("v")
        silence = bytes(len(_tone(20, 48000)))
        player._mix_piece(silence, stream, speech=False)
        assert reasons == [PlaybackReason.CANCELLED]
        assert player._mix_voices == []

    def test_draining_yields_to_queued_speech(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        player._running = True
        out = _FakeOutStream(48000, 1)
        player._stream = _Stream(stream=out, device_id="d", sample_rate=48000, channels=1)
        player._mix_voices.append(
            _MixVoice(SpeechRequest(request_id="v", mix=True), _tone(40, 48000))
        )
        player._normal.append(SpeechRequest(request_id="speak"))
        player._drain_mix_voices()
        assert out.writes == []  # уступили место речи, ничего не домешивали
        assert len(player._mix_voices) == 1

    def test_draining_stops_when_the_stream_is_gone(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        player._running = True
        player._stream = None
        player._mix_voices.append(
            _MixVoice(SpeechRequest(request_id="v", mix=True), _tone(40, 48000))
        )
        player._drain_mix_voices()  # выходит без потока, не падая
        assert len(player._mix_voices) == 1


class TestPlayerSerialPlayback:
    """Последовательное воспроизведение фразы: отмена до, во время и после цикла."""

    def test_cancel_before_the_first_chunk_returns_cancelled(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        player._cancel.set()
        chunk = AudioChunk(_tone(20, 48000), sample_rate=48000)
        result = player._play(SpeechRequest(request_id="p", chunks=(chunk,)))
        assert result == PlaybackReason.CANCELLED

    def test_cancel_after_the_last_chunk_returns_cancelled(self) -> None:
        player = TtsPlayer(_FakeOutBackend())

        def chunks() -> Iterator[AudioChunk]:
            yield AudioChunk(b"")  # пустой: цикл пропускает, отмена ещё не стоит
            player._cancel.set()  # отмена приходит после последнего звука

        result = player._play(SpeechRequest(request_id="p", chunks=chunks()))
        assert result == PlaybackReason.CANCELLED

    def test_a_cancelled_write_error_is_treated_as_a_stop(self) -> None:
        player = TtsPlayer(_FakeOutBackend())
        out = _FakeOutStream(48000, 1, fail_write=True)
        out.on_write = player._cancel.set  # запись роняет устройство ровно в момент отмены
        player._stream = _Stream(stream=out, device_id="d", sample_rate=48000, channels=1)
        chunk = AudioChunk(_tone(20, 48000), sample_rate=48000)
        assert player._write_chunk(chunk, SpeechRequest(request_id="w")) is False
