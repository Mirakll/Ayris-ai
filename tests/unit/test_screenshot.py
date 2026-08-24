"""Задача 26: снимки экрана — области, имена файлов, вывод.

Что здесь проверяется и почему именно это. Захват пикселей — дело `mss` и
драйвера, его нельзя проверить осмысленно ни в песочнице, ни на раннере, поэтому
бэкенд подменяется через `set_capture_backend` и тесты смотрят на расчёты вокруг
него: какой прямоугольник запрошен при разных DPI и отрицательных координатах, во
что превратился шаблон имени, что вернулось в `ActionResult`.

Единственный тест с настоящим экраном помечен `hardware` и утверждает только то,
что можно утверждать: кадр не пустой и его размер совпадает с запрошенным. Про
содержимое пикселей — ни слова: на раннере там чёрный рабочий стол.
"""

from __future__ import annotations

import struct
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from ayris.actions.system import screenshot
from ayris.actions.system import windows as win
from ayris.actions.system.screenshot import (
    Frame,
    OutputMode,
    Screenshot,
    ScreenshotRegion,
    ScreenshotWindow,
    Shot,
    build_path,
    clamp_rect,
    normalize_rect,
    render_filename,
    safe_component,
)
from ayris.core.errors import ActionError
from ayris.utils import monitors, winapi

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit

#: Пустые границы отключают обрезку в `capture_rect` — так тестируются проверки
#: самого прямоугольника, без виртуального рабочего стола этой машины.
NO_BOUNDS = winapi.Rect()


# --------------------------------------------------------------------------- #
# Инструменты
# --------------------------------------------------------------------------- #


def make_frame(
    width: int = 8,
    height: int = 4,
    *,
    fill: tuple[int, int, int] = (32, 64, 96),
    rect: winapi.Rect | None = None,
    monitor: str = "",
    window: str = "",
) -> Frame:
    """Кадр заданного размера, залитый одним цветом (BGRA, как отдаёт mss)."""
    red, green, blue = fill
    pixel = bytes((blue, green, red, 0))
    return Frame(
        width=width,
        height=height,
        bgra=pixel * (width * height),
        rect=rect or winapi.Rect(0, 0, width, height),
        monitor=monitor,
        window=window,
    )


class FakeCapture:
    """Бэкенд, который запоминает запрошенные прямоугольники и рисует заливку."""

    def __init__(self, *, fill: tuple[int, int, int] = (10, 20, 30)) -> None:
        self.calls: list[winapi.Rect] = []
        self._fill = fill

    def grab(self, rect: winapi.Rect) -> Frame:
        self.calls.append(rect)
        return make_frame(rect.width, rect.height, fill=self._fill, rect=rect)


class BlackCapture:
    """Бэкенд, отдающий чёрный кадр, — так ведёт себя защищённое окно."""

    def grab(self, rect: winapi.Rect) -> Frame:
        return make_frame(rect.width, rect.height, fill=(0, 0, 0), rect=rect)


@pytest.fixture
def capture() -> Iterator[FakeCapture]:
    """Подменённый бэкенд захвата на время теста."""
    fake = FakeCapture()
    screenshot.set_capture_backend(fake)
    try:
        yield fake
    finally:
        screenshot.set_capture_backend(None)


@pytest.fixture
def region() -> Iterator[list[winapi.Rect | None]]:
    """Ответы выделения области по очереди, вместо Qt-overlay."""
    answers: list[winapi.Rect | None] = []

    def provider(_timeout_s: float, _dim: float) -> winapi.Rect | None:
        return answers.pop(0) if answers else None

    screenshot.set_region_provider(provider)
    try:
        yield answers
    finally:
        screenshot.set_region_provider(None)


def display(
    rect: winapi.Rect,
    *,
    dpi: int = 96,
    primary: bool = False,
    external_index: int = -1,
    index: int = 0,
    name: str = "",
) -> monitors.MonitorInfo:
    """Монитор для подстановки в `monitors.list_monitors`.

    `address`, `scale` и `label` у `MonitorInfo` вычисляемые, поэтому задаются через
    то, из чего они считаются: `primary`/`external_index`, `dpi` и `name`.
    """
    return monitors.MonitorInfo(
        handle=index + 1,
        index=index,
        rect=rect,
        work=rect,
        device=rf"\\.\DISPLAY{index + 1}",
        name=name or f"Монитор {index + 1}",
        dpi=dpi,
        primary=primary,
        external_index=external_index,
    )


def install_displays(
    monkeypatch: pytest.MonkeyPatch,
    found: tuple[monitors.MonitorInfo, ...],
) -> tuple[monitors.MonitorInfo, ...]:
    """Подменить и список мониторов, и границы виртуального рабочего стола.

    Второе обязательно: `capture_rect` без явных `bounds` обрезает по
    `virtual_bounds()`, и на машине с одним монитором в нуле все отрицательные
    координаты из тестов схлопнулись бы в ноль.
    """
    monkeypatch.setattr(monitors, "list_monitors", lambda: list(found))
    bounds = monitors.virtual_bounds(found)
    monkeypatch.setattr(monitors, "virtual_bounds", lambda _found=None: bounds)
    return found


# --------------------------------------------------------------------------- #
# Прямоугольники
# --------------------------------------------------------------------------- #


class TestRects:
    """Нормализация, обрезка и отрицательные координаты."""

    def test_normalize_orders_the_corners(self) -> None:
        """Выделение справа налево и снизу вверх — такой же прямоугольник."""
        assert normalize_rect(300, 200, 100, 50) == winapi.Rect(100, 50, 300, 200)

    def test_normalize_keeps_an_already_sane_rect(self) -> None:
        assert normalize_rect(0, 0, 10, 20) == winapi.Rect(0, 0, 10, 20)

    def test_negative_coordinates_survive_normalization(self) -> None:
        """Монитор слева от основного живёт в отрицательных X — это норма.

        Виртуальный рабочий стол считается от левого верхнего угла основного
        монитора, поэтому у всего, что стоит левее или выше него, координаты
        отрицательные. Обнулить их — значит снять не тот кадр.
        """
        rect = normalize_rect(-1920, -200, -920, 300)
        assert rect == winapi.Rect(-1920, -200, -920, 300)
        assert (rect.width, rect.height) == (1000, 500)

    def test_clamp_cuts_to_the_bounds(self) -> None:
        bounds = winapi.Rect(0, 0, 1920, 1080)
        assert clamp_rect(winapi.Rect(-100, -50, 500, 400), bounds) == winapi.Rect(0, 0, 500, 400)
        assert clamp_rect(winapi.Rect(1800, 1000, 2400, 1400), bounds) == winapi.Rect(
            1800, 1000, 1920, 1080
        )

    def test_clamp_keeps_negatives_inside_negative_bounds(self) -> None:
        """Обрезка по рабочему столу, начинающемуся в минусах, ничего не сдвигает."""
        bounds = winapi.Rect(-1920, -1080, 1920, 1080)
        rect = winapi.Rect(-1800, -900, -100, -50)
        assert clamp_rect(rect, bounds) == rect

    def test_clamp_of_a_rect_outside_the_bounds_is_empty(self) -> None:
        """Прямоугольник целиком за пределами — пустой, а не отрицательный."""
        clamped = clamp_rect(winapi.Rect(3000, 3000, 3200, 3200), winapi.Rect(0, 0, 1920, 1080))
        assert clamped.is_empty

    def test_capture_rect_passes_negatives_through(
        self,
        capture: FakeCapture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Область на левом мониторе запрашивается с отрицательным X."""
        install_displays(
            monkeypatch,
            (
                display(winapi.Rect(0, 0, 1920, 1080), primary=True),
                display(winapi.Rect(-1920, 0, 0, 1080), external_index=0, index=1),
            ),
        )
        screenshot.capture_rect(winapi.Rect(-1500, 100, -500, 600))
        assert capture.calls == [winapi.Rect(-1500, 100, -500, 600)]

    def test_capture_rect_clamps_when_bounds_are_given(self, capture: FakeCapture) -> None:
        screenshot.capture_rect(
            winapi.Rect(-100, -100, 200, 200),
            bounds=winapi.Rect(0, 0, 1920, 1080),
        )
        assert capture.calls == [winapi.Rect(0, 0, 200, 200)]

    @pytest.mark.parametrize(
        ("left", "top", "right", "bottom"),
        [(0, 0, 0, 0), (10, 10, 12, 20), (10, 10, 20, 12), (0, 0, 40_000, 100)],
    )
    def test_impossible_rects_are_refused(
        self,
        capture: FakeCapture,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> None:
        """Пустое, слишком узкое и слишком большое — с понятной ошибкой."""
        with pytest.raises(ActionError):
            screenshot.capture_rect(winapi.Rect(left, top, right, bottom), bounds=NO_BOUNDS)
        assert capture.calls == []


class TestMixedDpi:
    """Расчёт областей на конфигурации с разным масштабом у мониторов."""

    @pytest.fixture
    def displays(self, monkeypatch: pytest.MonkeyPatch) -> tuple[monitors.MonitorInfo, ...]:
        """Ноутбук 150 % в нуле, внешний 4K слева, ещё один сверху.

        Это не выдуманная конфигурация, а обычная: физические пиксели у мониторов
        свои, координаты — общие, и левый с верхним уезжают в минус.
        """
        return install_displays(
            monkeypatch,
            (
                display(winapi.Rect(0, 0, 2560, 1440), dpi=144, primary=True),
                display(winapi.Rect(-3840, 0, 0, 2160), external_index=0, index=1),
                display(winapi.Rect(0, -1080, 1920, 0), external_index=1, index=2),
            ),
        )

    def test_the_fixture_addresses_are_what_the_user_says(
        self,
        displays: tuple[monitors.MonitorInfo, ...],
    ) -> None:
        assert [item.address for item in displays] == ["primary", "external_1", "external_2"]
        assert displays[0].scale == pytest.approx(1.5)

    def test_monitor_rect_is_taken_in_physical_pixels(
        self,
        capture: FakeCapture,
        displays: tuple[monitors.MonitorInfo, ...],
    ) -> None:
        """У монитора с масштабом 150 % берутся физические 2560×1440.

        Логический размер того же монитора — 1707×960, и если снимать по нему,
        получается обрезанный кадр. `MonitorInfo.rect` уже физический, тест
        закрепляет, что никто по дороге не делит на масштаб.
        """
        frame = screenshot.capture_monitor("primary")
        assert capture.calls == [displays[0].rect]
        assert (frame.width, frame.height) == (2560, 1440)
        assert frame.monitor == displays[0].label

    def test_a_monitor_left_of_the_primary_keeps_negative_x(
        self,
        capture: FakeCapture,
        displays: tuple[monitors.MonitorInfo, ...],
    ) -> None:
        screenshot.capture_monitor("external_1")
        assert capture.calls == [winapi.Rect(-3840, 0, 0, 2160)]

    def test_a_monitor_above_the_primary_keeps_negative_y(
        self,
        capture: FakeCapture,
        displays: tuple[monitors.MonitorInfo, ...],
    ) -> None:
        screenshot.capture_monitor("external_2")
        assert capture.calls == [winapi.Rect(0, -1080, 1920, 0)]

    def test_all_monitors_is_the_union_of_the_layout(
        self,
        capture: FakeCapture,
        displays: tuple[monitors.MonitorInfo, ...],
    ) -> None:
        """Кадр «все мониторы» — объединяющий прямоугольник, начиная с минусов."""
        frame = screenshot.capture_all()
        assert capture.calls == [winapi.Rect(-3840, -1080, 2560, 2160)]
        assert (frame.width, frame.height) == (6400, 3240)

    def test_an_unknown_monitor_names_the_ones_there_are(
        self,
        capture: FakeCapture,
        displays: tuple[monitors.MonitorInfo, ...],
    ) -> None:
        """«Не нашла монитор» бесполезно без списка тех, что есть."""
        with pytest.raises(ActionError) as caught:
            screenshot.capture_monitor("external_9")
        assert displays[0].label in caught.value.user_message
        assert capture.calls == []


class TestWindowBounds:
    """Границы окна берутся у композитора, а не у GetWindowRect."""

    def test_window_bounds_asks_dwm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`DWMWA_EXTENDED_FRAME_BOUNDS`, иначе в кадр попадёт тень окна.

        `GetWindowRect` у окна со стандартной рамкой возвращает прямоугольник на
        7–8 пикселей шире с каждой стороны: там живёт невидимая область тени.
        Снимок по нему получается с полосками рабочего стола по краям.
        """
        asked: list[int] = []

        def fake_bounds(hwnd: int) -> winapi.Rect:
            asked.append(hwnd)
            return winapi.Rect(100, 100, 900, 700)

        monkeypatch.setattr(winapi, "extended_frame_bounds", fake_bounds)
        assert screenshot.window_bounds(0xABCD) == winapi.Rect(100, 100, 900, 700)
        assert asked == [0xABCD]

    def test_capture_window_uses_those_bounds(
        self,
        capture: FakeCapture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_displays(monkeypatch, (display(winapi.Rect(0, 0, 1920, 1080), primary=True),))
        monkeypatch.setattr(
            winapi,
            "extended_frame_bounds",
            lambda _hwnd: winapi.Rect(8, 50, 808, 450),
        )
        frame = screenshot.capture_window(1, title="Блокнот")
        assert capture.calls == [winapi.Rect(8, 50, 808, 450)]
        assert frame.window == "Блокнот"
        assert (frame.width, frame.height) == (800, 400)


# --------------------------------------------------------------------------- #
# Кадр
# --------------------------------------------------------------------------- #


class TestFrame:
    """Проверки самого кадра: размеры, форматы, признак чёрного."""

    def test_a_frame_checks_its_own_buffer(self) -> None:
        with pytest.raises(ValueError, match="expected 16"):
            Frame(width=2, height=2, bgra=b"\x00" * 12)

    def test_a_frame_without_area_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no area"):
            Frame(width=0, height=4, bgra=b"")

    @pytest.mark.parametrize("quality", [60, 95])
    def test_png_and_jpeg_keep_the_size(self, quality: int) -> None:
        frame = make_frame(16, 9, fill=(200, 100, 50))
        for payload in (frame.to_png(), frame.to_jpeg(quality=quality)):
            with Image.open(BytesIO(payload)) as reopened:
                assert reopened.size == (16, 9)

    def test_to_image_keeps_the_channel_order(self) -> None:
        """BGRA от mss должно стать RGB, а не BGR: иначе снимок синеет."""
        image = make_frame(2, 2, fill=(200, 100, 50)).to_image()
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (200, 100, 50)

    def test_dib_is_bottom_up_with_no_file_header(self) -> None:
        """CF_DIB — это BITMAPINFOHEADER и пиксели снизу вверх, без «BM».

        Положить в буфер обмена PNG-байты с файловым заголовком — самая частая
        ошибка в этом месте: вставка молча даёт пустую картинку. Заголовок здесь
        разбирается обратно, чтобы это нельзя было сломать незаметно.
        """
        dib = make_frame(4, 3).to_dib()
        size, width, height, planes, bits = struct.unpack("<IiiHH", dib[:16])
        assert (size, width, height, planes, bits) == (40, 4, 3, 1, 32)
        assert not dib.startswith(b"BM")
        assert len(dib) == 40 + 4 * 3 * 4

    def test_dib_rows_are_reversed(self) -> None:
        """Нижняя строка кадра лежит в DIB первой."""
        top = bytes((0, 0, 255, 0)) * 2
        bottom = bytes((255, 0, 0, 0)) * 2
        frame = Frame(width=2, height=2, bgra=top + bottom)
        pixels = frame.to_dib()[40:]
        assert pixels[0] == 255  # синий канал нижней строки
        assert pixels[8] == 0  # верхняя строка ушла во вторую половину

    def test_dib_alpha_is_forced_opaque(self) -> None:
        """GDI оставляет альфу неопределённой, и вставка выходит прозрачной."""
        dib = make_frame(2, 2).to_dib()
        assert set(dib[40:][3::4]) == {0xFF}

    def test_a_black_frame_is_recognised_as_blank(self) -> None:
        """Защищённое от захвата окно отдаёт чёрный кадр — это надо заметить."""
        assert make_frame(4, 4, fill=(0, 0, 0)).is_blank
        assert make_frame(4, 4, fill=(2, 1, 3)).is_blank  # шум кодека тоже чёрный

    def test_a_normal_frame_is_not_blank(self) -> None:
        assert not make_frame(4, 4, fill=(0, 0, 40)).is_blank

    def test_as_dict_carries_no_pixels(self) -> None:
        """Кадр уходит в `ActionResult.data`, а это идёт в лог и в аудит."""
        described = make_frame(6, 4, monitor="primary").as_dict()
        assert described["width"] == 6
        assert "bgra" not in described


# --------------------------------------------------------------------------- #
# Имена файлов
# --------------------------------------------------------------------------- #


class TestFilenames:
    """Подстановка шаблона и защита от перезаписи."""

    #: Время снимка — локальное, без часового пояса: имя файла пишется так же.
    WHEN = datetime(2026, 8, 24, 15, 7, 42)

    def test_every_placeholder_is_substituted(self) -> None:
        rendered = render_filename(
            "{date}_{time}_{monitor}_{window}_{n}",
            when=self.WHEN,
            monitor="external_1",
            window="Блокнот",
            index=3,
        )
        assert rendered == "2026-08-24_15-07-42_external_1_Блокнот_3"

    def test_the_default_template_gives_a_sortable_name(self) -> None:
        assert render_filename("ayris_{date}_{time}", when=self.WHEN) == "ayris_2026-08-24_15-07-42"

    def test_missing_values_do_not_leave_double_separators(self) -> None:
        """Шаблон с {window} на снимке монитора не должен давать «a__b»."""
        rendered = render_filename("shot_{monitor}_{window}_{date}", when=self.WHEN)
        assert "__" not in rendered
        assert rendered == "shot_2026-08-24"

    def test_an_unknown_placeholder_is_left_alone(self) -> None:
        """Опечатку в настройках лучше увидеть в имени файла, чем потерять."""
        assert render_filename("{daet}", when=self.WHEN) == "{daet}"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('Проект: "план" <2>', "Проект план 2"),
            ("C:\\Windows\\explorer.exe", "C Windows explorer.exe"),
            ("   ...   ", ""),
            ("con", "_con"),
            ("LPT1.png", "_LPT1.png"),
        ],
    )
    def test_components_are_made_safe_for_ntfs(self, raw: str, expected: str) -> None:
        """Заголовок окна попадает в имя файла и приносит с собой `:` и `?`.

        Плюс зарезервированные имена DOS: файл `con.png` нельзя создать до сих
        пор, и падает это не на открытии, а на записи.
        """
        assert safe_component(raw) == expected

    def test_a_long_title_is_cut(self) -> None:
        assert len(safe_component("я" * 500)) <= 64

    def test_build_path_creates_the_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "снимки" / "август"
        path = build_path(target, "shot_{date}", extension="png", when=self.WHEN)
        assert path.parent.is_dir()
        assert path.name == "shot_2026-08-24.png"

    def test_a_series_placeholder_finds_the_first_free_number(self, tmp_path: Path) -> None:
        for index in (1, 2, 3):
            (tmp_path / f"shot_{index}.png").write_bytes(b"")
        path = build_path(tmp_path, "shot_{n}", extension="png", when=self.WHEN)
        assert path.name == "shot_4.png"

    def test_without_a_series_placeholder_a_suffix_is_added(self, tmp_path: Path) -> None:
        """Два снимка в одну секунду не должны затирать друг друга."""
        (tmp_path / "shot.png").write_bytes(b"")
        (tmp_path / "shot_2.png").write_bytes(b"")
        path = build_path(tmp_path, "shot", extension="png", when=self.WHEN)
        assert path.name == "shot_3.png"

    def test_the_extension_follows_the_format(self, tmp_path: Path) -> None:
        assert build_path(tmp_path, "shot", extension="jpg", when=self.WHEN).suffix == ".jpg"


# --------------------------------------------------------------------------- #
# Вывод
# --------------------------------------------------------------------------- #


@pytest.fixture
def clipboard(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bytes]]:
    """Перехваченный буфер обмена: что в него положили, в порядке форматов."""
    payloads: list[tuple[int, bytes]] = []
    monkeypatch.setattr(winapi, "clipboard_set_binary", payloads.extend)
    monkeypatch.setattr(winapi, "register_clipboard_format", lambda _name: 49_999)
    return payloads


class TestOutput:
    """Файл, буфер обмена или и то и другое."""

    def test_save_writes_a_readable_png(self, tmp_path: Path) -> None:
        frame = make_frame(12, 8, fill=(90, 140, 190))
        path = screenshot.save_frame(frame, directory=tmp_path, template="shot")
        with Image.open(path) as saved:
            assert saved.size == (12, 8)
            assert saved.format == "PNG"

    def test_jpeg_is_written_when_the_format_says_so(self, tmp_path: Path) -> None:
        path = screenshot.save_frame(
            make_frame(12, 8),
            directory=tmp_path,
            template="shot",
            image_format="jpeg",
            quality=80,
        )
        assert path.suffix == ".jpg"
        assert path.stat().st_size > 0

    def test_clipboard_only_writes_no_file(
        self,
        tmp_path: Path,
        clipboard: list[tuple[int, bytes]],
    ) -> None:
        shot = screenshot.deliver(
            make_frame(6, 6),
            OutputMode.CLIPBOARD,
            directory=tmp_path,
            template="shot",
        )
        assert shot.path is None
        assert shot.clipboard
        # Не `iterdir() == []`: conftest уводит сюда же %APPDATA% профиля.
        assert not list(tmp_path.glob("*.png"))
        assert clipboard

    def test_both_writes_a_file_and_copies(
        self,
        tmp_path: Path,
        clipboard: list[tuple[int, bytes]],
    ) -> None:
        shot = screenshot.deliver(
            make_frame(6, 6),
            OutputMode.BOTH,
            directory=tmp_path,
            template="shot",
        )
        assert shot.path is not None
        assert shot.path.is_file()
        assert shot.clipboard

    def test_the_clipboard_gets_the_picture_twice(
        self,
        clipboard: list[tuple[int, bytes]],
    ) -> None:
        """CF_DIB для всего, что старше Paint, и PNG для того, что понимает альфу."""
        assert screenshot.copy_frame(make_frame(4, 4))
        formats = [identifier for identifier, _payload in clipboard]
        assert 8 in formats  # CF_DIB
        assert 49_999 in formats  # зарегистрированный PNG

    def test_a_refusing_clipboard_does_not_lose_the_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Буфер держит чужой процесс — снимок всё равно сохранён."""

        def refuse(_payloads: object) -> None:
            raise winapi.WinApiError("clipboard is busy")

        monkeypatch.setattr(winapi, "clipboard_set_binary", refuse)
        monkeypatch.setattr(winapi, "register_clipboard_format", lambda _name: 49_999)
        shot = screenshot.deliver(
            make_frame(6, 6),
            OutputMode.BOTH,
            directory=tmp_path,
            template="shot",
        )
        assert shot.path is not None
        assert shot.path.is_file()
        assert not shot.clipboard


# --------------------------------------------------------------------------- #
# Действия
# --------------------------------------------------------------------------- #


class TestActions:
    """То, что видит реестр: параметры, результат, отмена."""

    @pytest.fixture(autouse=True)
    def _one_monitor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clipboard: list[tuple[int, bytes]],
    ) -> None:
        install_displays(monkeypatch, (display(winapi.Rect(0, 0, 1920, 1080), primary=True),))

    def test_screenshot_of_everything(self, capture: FakeCapture) -> None:
        result = Screenshot().run(Screenshot.Params(output=OutputMode.CLIPBOARD))
        assert result.ok
        assert isinstance(result.value, Shot)
        assert result.value.frame.width == 1920
        assert capture.calls == [winapi.Rect(0, 0, 1920, 1080)]

    def test_screenshot_of_one_monitor_names_it(self, capture: FakeCapture) -> None:
        """В ответе — то имя монитора, которое видит человек, а не адрес."""
        result = Screenshot().run(Screenshot.Params(monitor="primary", output=OutputMode.CLIPBOARD))
        assert result.ok
        assert "Монитор 1" in result.message_ru

    def test_region_by_numbers_does_not_ask_the_user(self, capture: FakeCapture) -> None:
        """Заданы координаты — overlay не поднимается вообще.

        Провайдер не подменён: если действие всё-таки решит спросить, оно уйдёт в
        настоящий Qt-overlay и тест упадёт без QApplication, что и требуется.
        """
        result = ScreenshotRegion().run(
            ScreenshotRegion.Params(
                left=100,
                top=50,
                width=300,
                height=200,
                output=OutputMode.CLIPBOARD,
            )
        )
        assert result.ok
        assert capture.calls == [winapi.Rect(100, 50, 400, 250)]

    def test_region_reports_a_cancelled_selection(
        self,
        capture: FakeCapture,
        region: list[winapi.Rect | None],
    ) -> None:
        """Esc — это не ошибка: пользователь передумал."""
        region.append(None)
        result = ScreenshotRegion().run(ScreenshotRegion.Params())
        assert result.ok
        assert result.value is None
        assert "Отменила" in result.message_ru
        assert capture.calls == []

    def test_region_captures_what_was_dragged(
        self,
        capture: FakeCapture,
        region: list[winapi.Rect | None],
    ) -> None:
        region.append(winapi.Rect(200, 100, 700, 500))
        result = ScreenshotRegion().run(ScreenshotRegion.Params(output=OutputMode.CLIPBOARD))
        assert result.ok
        assert capture.calls == [winapi.Rect(200, 100, 700, 500)]

    def test_half_a_rectangle_is_refused_at_validation(self) -> None:
        with pytest.raises(ValueError, match="ширину"):
            ScreenshotRegion.Params(width=300)

    def test_window_shot_uses_the_compositor_bounds(
        self,
        capture: FakeCapture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        found = win.WindowRecord(hwnd=42, title="Блокнот — заметки", process="notepad.exe", pid=1)
        monkeypatch.setattr(screenshot, "list_windows", lambda _query: [found])
        monkeypatch.setattr(screenshot, "select_window", lambda _records, _query: found)
        monkeypatch.setattr(
            winapi,
            "extended_frame_bounds",
            lambda _hwnd: winapi.Rect(8, 8, 792, 592),
        )
        result = ScreenshotWindow().run(
            ScreenshotWindow.Params(title="Блокнот", output=OutputMode.CLIPBOARD)
        )
        assert result.ok
        assert capture.calls == [winapi.Rect(8, 8, 792, 592)]
        assert "Блокнот" in result.message_ru

    def test_a_black_frame_is_explained(self) -> None:
        """Кадр чёрный — сказать, почему, а не отдать чёрный PNG молча."""
        screenshot.set_capture_backend(BlackCapture())
        try:
            result = Screenshot().run(Screenshot.Params(output=OutputMode.CLIPBOARD))
        finally:
            screenshot.set_capture_backend(None)
        assert result.ok
        assert result.value is not None
        assert result.value.blank
        assert "чёрн" in result.message_ru.lower()


# --------------------------------------------------------------------------- #
# Настоящий экран
# --------------------------------------------------------------------------- #


@pytest.mark.hardware
class TestRealScreen:
    """Единственное, что проверяется на живом экране, — что кадр вообще есть."""

    def test_a_real_capture_has_the_requested_size(self) -> None:
        """Размер совпадает и буфер полный. Про пиксели — ничего.

        На раннере рабочий стол чёрный, а composited-окна отдают чёрный кадр и на
        живой машине. Утверждать здесь что-то про содержимое — значит написать
        тест, который однажды покраснеет без причины.
        """
        found = monitors.list_monitors()
        if not found:
            pytest.skip("нет ни одного монитора")
        frame = screenshot.capture_monitor(found[0].address)
        assert (frame.width, frame.height) == (found[0].rect.width, found[0].rect.height)
        assert len(frame.bgra) == frame.pixels * 4
