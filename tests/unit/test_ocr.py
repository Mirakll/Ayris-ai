"""Задача 26: распознавание текста — выбор движка, подготовка, разбор, координаты.

Настоящий движок здесь почти не участвует, и это не компромисс. Всё, что может
сломаться в этом коде, живёт *вокруг* распознавания: какой движок выбран и почему,
во сколько раз увеличена картинка, куда попали блоки после обратного пересчёта,
что окажется в озвучке из полутора экранов текста. Каждый из этих вопросов
проверяется на поддельном движке и синтетической картинке — быстро, детерминированно
и одинаково на Windows, в песочнице и на Linux-раннере, где winrt нет вообще.

Отдельно — два теста на живом `Windows.Media.Ocr` с фикстурами
`tests/fixtures/images`. Они пропускаются, если в системе нет нужного языкового
пакета: это нормальное состояние Windows, а не повод красить прогон.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest
from PIL import Image, ImageDraw

from ayris.actions.registry import registered_actions
from ayris.actions.system import ocr
from ayris.actions.system import ocr_engines as engines_module
from ayris.actions.system import screenshot as shot_module
from ayris.actions.system.ocr import (
    MAX_PIXELS,
    MAX_SCALE,
    OcrRegion,
    OcrScreen,
    Reading,
    TextOutput,
    binarize,
    for_speech,
    prepare,
    read_frame,
    scale_for,
)
from ayris.actions.system.ocr_engines import (
    Choice,
    available_engines,
    engine_by_name,
    reset_engine_cache,
    select_engine,
)
from ayris.actions.system.ocr_engines.base import (
    OcrBlock,
    OcrEngine,
    OcrEngineError,
    OcrText,
    primary_subtag,
)
from ayris.actions.system.ocr_engines.paddle import PaddleOcr, parse_paddle_result
from ayris.actions.system.ocr_engines.tesseract import (
    TESSERACT_CODES,
    TesseractOcr,
    find_tesseract,
    join_lines,
    parse_image_data,
)
from ayris.actions.system.ocr_engines.windows_ocr import WindowsOcr, _await
from ayris.actions.system.screenshot import Frame
from ayris.core import config as config_module
from ayris.utils import monitors, winapi

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

#: Картинки с текстом лежат рядом с тестами, а не рисуются на ходу: шрифт на
#: раннере может оказаться другим, и «распознала» превратилось бы в лотерею.
IMAGES = Path(__file__).resolve().parents[1] / "fixtures" / "images"


# --------------------------------------------------------------------------- #
# Поддельные движки
# --------------------------------------------------------------------------- #


class FakeEngine(OcrEngine):
    """Движок, который всегда доступен и отдаёт заранее заданный ответ."""

    name: ClassVar[str] = "fake"
    title_ru: ClassVar[str] = "Поддельный"
    wants_upscale: ClassVar[bool] = True
    languages: ClassVar[tuple[str, ...]] = ("ru", "en")

    def __init__(self, language: str) -> None:
        super().__init__(language)
        self.seen: list[tuple[int, int]] = []
        self.blocks: tuple[OcrBlock, ...] = (
            OcrBlock(text="Привет", rect=winapi.Rect(10, 20, 110, 50), confidence=0.9),
        )

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def available_languages(cls) -> tuple[str, ...]:
        return cls.languages

    def recognise(self, image: Image.Image) -> OcrText:
        self.seen.append(image.size)
        return OcrText(
            text="\n".join(block.text for block in self.blocks),
            blocks=self.blocks,
            engine=self.name,
            language=self.language,
        )


class MissingEngine(FakeEngine):
    """Движок, которого нет на машине, — Tesseract без бинарника."""

    name: ClassVar[str] = "missing"
    title_ru: ClassVar[str] = "Отсутствующий"

    @classmethod
    def is_available(cls) -> bool:
        return False

    @classmethod
    def available_languages(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def describe_missing(cls) -> str:
        return "Поставь его установщиком и отметь русский язык."


class WrongLanguageEngine(FakeEngine):
    """Установлен, но не умеет нужного языка — Tesseract без rus.traineddata."""

    name: ClassVar[str] = "englishonly"
    title_ru: ClassVar[str] = "Только английский"
    languages: ClassVar[tuple[str, ...]] = ("en-US",)


class BrokenEngine(FakeEngine):
    """Движок, у которого падает даже проверка доступности."""

    name: ClassVar[str] = "broken"
    title_ru: ClassVar[str] = "Сломанный"

    @classmethod
    def is_available(cls) -> bool:
        raise RuntimeError("сломался на проверке")

    @classmethod
    def available_languages(cls) -> tuple[str, ...]:
        raise RuntimeError("сломался на языках")


class ScreenEngine(FakeEngine):
    """Как Windows OCR: увеличения не хочет."""

    name: ClassVar[str] = "screen"
    wants_upscale: ClassVar[bool] = False


class SmallEngine(FakeEngine):
    """Движок с жёстким пределом на сторону картинки."""

    name: ClassVar[str] = "small"
    max_dimension: ClassVar[int] = 100


@pytest.fixture(autouse=True)
def _clean_engines() -> Iterator[None]:
    """Кэш собранных движков — глобальный, между тестами его надо забывать."""
    reset_engine_cache()
    yield
    reset_engine_cache()
    TesseractOcr.configure("")


def use(monkeypatch: pytest.MonkeyPatch, *classes: type[OcrEngine]) -> None:
    """Сделать `classes` полным списком движков в этом порядке."""
    monkeypatch.setattr(engines_module, "ENGINES", tuple(classes))
    reset_engine_cache()


def one_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один монитор 1920×1080 в нуле — чтобы захват не зависел от машины."""
    only = monitors.MonitorInfo(
        handle=1,
        index=0,
        rect=winapi.Rect(0, 0, 1920, 1080),
        work=winapi.Rect(0, 0, 1920, 1040),
        device=r"\\.\DISPLAY1",
        name="Монитор 1",
        primary=True,
    )
    monkeypatch.setattr(monitors, "list_monitors", lambda: [only])
    monkeypatch.setattr(monitors, "virtual_bounds", lambda _found=None: only.rect)


def make_frame(
    width: int = 400,
    height: int = 200,
    *,
    rect: winapi.Rect | None = None,
) -> Frame:
    """Кадр-заглушка нужного размера (серая заливка, не чёрная)."""
    return Frame(
        width=width,
        height=height,
        bgra=bytes((120, 120, 120, 0)) * (width * height),
        rect=rect or winapi.Rect(0, 0, width, height),
    )


def picture(width: int = 400, height: int = 120) -> Image.Image:
    """Белая картинка с чёрной полосой — чтобы autocontrast было что растягивать."""
    image = Image.new("RGB", (width, height), (245, 245, 245))
    ImageDraw.Draw(image).rectangle((10, 10, width - 10, height // 2), fill=(40, 40, 40))
    return image


# --------------------------------------------------------------------------- #
# Выбор движка и фоллбек
# --------------------------------------------------------------------------- #


class TestEngineChoice:
    """Что выбрано, что вместо чего и что об этом сказано пользователю."""

    def test_auto_takes_the_first_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use(monkeypatch, FakeEngine, WrongLanguageEngine)
        choice = select_engine("auto", ("ru", "en"))
        assert choice.engine.name == "fake"
        assert not choice.is_fallback
        assert choice.note_ru == ""

    def test_a_named_engine_is_honoured_even_if_it_is_not_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use(monkeypatch, FakeEngine, WrongLanguageEngine)
        choice = select_engine("englishonly", ("en",))
        assert choice.engine.name == "englishonly"
        assert not choice.is_fallback

    def test_a_missing_binary_falls_back_to_the_next(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Названный движок не установлен — работаем на следующем, но говорим об этом.

        Это главный сценарий задачи: в настройках выбран Tesseract, его никто не
        поставил, и распознавание всё равно должно случиться.
        """
        use(monkeypatch, MissingEngine, FakeEngine)
        choice = select_engine("missing", ("ru",))
        assert choice.engine.name == "fake"
        assert choice.is_fallback
        assert choice.fallback_from == "missing"
        assert "Отсутствующий" in choice.note_ru
        assert "не установлен" in choice.reason

    def test_the_fallback_note_names_both_engines(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use(monkeypatch, MissingEngine, FakeEngine)
        note = select_engine("missing", ("ru",)).note_ru
        assert "Отсутствующий" in note
        assert "Поддельный" in note

    def test_an_engine_without_the_language_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Установлен, но без rus — для русского текста бесполезен.

        Это отдельный от «не установлен» случай, и причина фоллбека должна их
        различать: инструкция пользователю разная.
        """
        use(monkeypatch, WrongLanguageEngine, FakeEngine)
        choice = select_engine("englishonly", ("ru",))
        assert choice.engine.name == "fake"
        assert "не знает" in choice.reason

    def test_a_broken_engine_does_not_block_the_rest(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Движок падает на `is_available` — выбор идёт дальше, а не наружу."""
        use(monkeypatch, BrokenEngine, FakeEngine)
        assert select_engine("auto", ("ru",)).engine.name == "fake"

    def test_no_usable_engine_says_what_to_install(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Названный движок недоступен, замены нет — его же инструкция по установке."""
        use(monkeypatch, MissingEngine)
        with pytest.raises(OcrEngineError) as caught:
            select_engine("missing", ("ru",))
        assert "установщиком" in caught.value.user_message

    def test_nothing_at_all_points_at_windows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """При `auto` без единого движка говорим про Windows OCR.

        Он первый по предпочтению и единственный, который пользователь может
        включить сам, ничего не устанавливая. Сравнение идёт с самим
        `describe_missing`, а не с куском фразы: у него две ветки — «нет пакетов
        winrt» и «нет языкового пакета», и на linux-джобе CI верна первая.
        """
        use(monkeypatch, MissingEngine)
        with pytest.raises(OcrEngineError) as caught:
            select_engine("auto", ("ru",))
        assert caught.value.user_message == WindowsOcr.describe_missing()

    def test_an_unknown_preference_is_treated_as_auto(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Опечатка в настройках не должна отключать распознавание совсем."""
        use(monkeypatch, FakeEngine)
        assert select_engine("tesserakt", ("ru",)).engine.name == "fake"

    def test_the_language_comes_from_the_wish_list_in_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Порядок языков в настройках — это порядок, а не набор.

        Windows OCR читает одним языком, и английский движок на русском тексте
        отдаёт уверенную чушь. Поэтому первый язык, который движок знает, и есть
        ответ.
        """
        use(monkeypatch, FakeEngine)
        assert select_engine("auto", ("de", "en", "ru")).language == "en"
        assert select_engine("auto", ("ru", "en")).language == "ru"

    def test_the_engine_is_built_once_per_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Native-объект дорогой: второй вызов должен вернуть тот же движок."""
        use(monkeypatch, FakeEngine)
        first = select_engine("auto", ("ru",)).engine
        assert select_engine("auto", ("ru",)).engine is first
        assert select_engine("auto", ("en",)).engine is not first

    def test_resetting_the_cache_builds_a_new_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use(monkeypatch, FakeEngine)
        first = select_engine("auto", ("ru",)).engine
        reset_engine_cache()
        assert select_engine("auto", ("ru",)).engine is not first

    def test_available_engines_skips_the_broken_and_the_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use(monkeypatch, BrokenEngine, MissingEngine, FakeEngine)
        assert available_engines() == ("fake",)

    def test_engine_by_name_knows_the_real_three(self) -> None:
        """Имена в настройках — это те же имена, что у классов."""
        assert engine_by_name("windows") is WindowsOcr
        assert engine_by_name("tesseract") is TesseractOcr
        assert engine_by_name("paddle") is PaddleOcr
        assert engine_by_name("auto") is None

    def test_windows_ocr_is_the_default_order(self) -> None:
        """Порядок по умолчанию: свой, установленный, тяжёлый.

        Windows первым не по алфавиту: он единственный, который уже есть на любой
        Windows 10/11, работает офлайн и знает русский из коробки.
        """
        assert engines_module.ENGINES[0] is WindowsOcr
        assert engines_module.ENGINES[-1] is PaddleOcr


class TestLanguageMatching:
    """Три движка называют языки тремя способами."""

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [("ru-RU", "ru"), ("en-US", "en"), ("eng", "eng"), (" RU ", "ru"), ("zh_CN", "zh")],
    )
    def test_primary_subtag(self, tag: str, expected: str) -> None:
        assert primary_subtag(tag) == expected

    def test_a_wish_for_ru_finds_the_engines_own_tag(self) -> None:
        """Пользователь говорит «ru», Windows отвечает «ru-RU», Tesseract — «rus»."""

        class Bcp47(FakeEngine):
            languages: ClassVar[tuple[str, ...]] = ("en-US", "ru-RU")

        assert Bcp47.match_language(("ru",)) == "ru-RU"
        assert Bcp47.match_language(("de", "en")) == "en-US"
        assert Bcp47.match_language(("ja",)) is None

    def test_tesseract_codes_cover_the_two_that_matter(self) -> None:
        assert TesseractOcr.code_for("ru") == "rus"
        assert TesseractOcr.code_for("en") == "eng"
        assert TesseractOcr.code_for("rus") == "rus"
        assert set(TESSERACT_CODES) >= {"ru", "en"}

    def test_a_wish_for_ru_finds_tesseracts_rus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`ru` из настроек должно находить установленный `rus`.

        Без этого фоллбек на Tesseract не работал бы никогда: сравнение первых
        подтегов даёт «ru» против «rus» и не совпадает ни с чем.
        """
        monkeypatch.setattr(TesseractOcr, "_languages", ("eng", "osd", "rus"))
        assert TesseractOcr.match_language(("ru", "en")) == "rus"
        assert TesseractOcr.match_language(("de", "en")) == "eng"
        assert TesseractOcr.match_language(("ja",)) is None


class TestTesseractAvailability:
    """Проверка бинарника и языковых данных — на моках, без запуска процесса."""

    def test_a_missing_binary_means_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TesseractOcr, "_languages", None)
        monkeypatch.setattr(TesseractOcr, "executable", classmethod(lambda _cls: ""))
        assert TesseractOcr.available_languages() == ()
        assert not TesseractOcr.is_available()
        assert "не установлен" in TesseractOcr.describe_missing()

    def test_a_binary_without_russian_data_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """«Установлен, но без языков» — другая инструкция, чем «не установлен»."""
        monkeypatch.setattr(TesseractOcr, "_languages", ())
        monkeypatch.setattr(
            TesseractOcr,
            "executable",
            classmethod(lambda _cls: r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        )
        assert not TesseractOcr.is_available()
        message = TesseractOcr.describe_missing()
        assert "без языковых данных" in message
        assert "traineddata" in message

    def test_the_language_list_is_parsed_from_the_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Первая строка `--list-langs` — заголовок, остальные — коды."""

        class Completed:
            stdout = "List of available languages (3):\neng\nosd\nrus\n"
            stderr = ""

        monkeypatch.setattr(TesseractOcr, "_languages", None)
        monkeypatch.setattr(
            TesseractOcr,
            "executable",
            classmethod(lambda _cls: "tesseract"),
        )
        monkeypatch.setattr(
            "ayris.actions.system.ocr_engines.tesseract.subprocess.run",
            lambda *_args, **_kwargs: Completed(),
        )
        assert TesseractOcr.available_languages() == ("eng", "osd", "rus")
        assert TesseractOcr.is_available()

    def test_a_crashing_binary_is_just_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Бинарник есть, но не запускается — это `False`, а не traceback наружу."""

        def explode(*_args: object, **_kwargs: object) -> None:
            raise OSError("не тот формат исполняемого файла")

        monkeypatch.setattr(TesseractOcr, "_languages", None)
        monkeypatch.setattr(TesseractOcr, "executable", classmethod(lambda _cls: "tesseract"))
        monkeypatch.setattr(
            "ayris.actions.system.ocr_engines.tesseract.subprocess.run",
            explode,
        )
        assert TesseractOcr.available_languages() == ()

    def test_the_configured_path_wins_over_path(self, tmp_path: Path) -> None:
        fake = tmp_path / "tesseract.exe"
        fake.write_bytes(b"")
        assert find_tesseract(str(fake)) == str(fake)

    def test_a_configured_path_pointing_at_nothing_falls_back_to_the_search(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Путь в настройках устарел — ищем как обычно, а не сдаёмся."""
        monkeypatch.setattr(
            "ayris.actions.system.ocr_engines.tesseract.shutil.which",
            lambda _name: "C:/tools/tesseract.exe",
        )
        assert find_tesseract(str(tmp_path / "нет.exe")) == "C:/tools/tesseract.exe"

    def test_configure_forgets_the_cached_languages(self) -> None:
        """Иначе смена пути в настройках не подействует до перезапуска."""
        TesseractOcr._languages = ("eng",)
        TesseractOcr.configure("")
        assert TesseractOcr._languages is None


class TestPaddleAvailability:
    """Paddle не должен импортироваться ради ответа «его нет»."""

    def test_availability_is_answered_without_importing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`find_spec`, а не `import`: импорт paddle стоит секунды и сотни мегабайт."""
        monkeypatch.setattr(
            "ayris.actions.system.ocr_engines.paddle.importlib.util.find_spec",
            lambda _name: None,
        )
        assert not PaddleOcr.is_available()
        assert PaddleOcr.available_languages() == ()

    def test_the_install_hint_names_the_package(self) -> None:
        assert "paddlepaddle" in PaddleOcr.describe_missing()


# --------------------------------------------------------------------------- #
# Разбор ответов движков
# --------------------------------------------------------------------------- #


class TestTesseractParsing:
    """`image_to_data` отдаёт параллельные списки по всем уровням разом."""

    DATA: ClassVar[dict[str, list[object]]] = {
        "level": [1, 2, 3, 4, 5, 5, 5],
        "block_num": [0, 1, 1, 1, 1, 1, 1],
        "par_num": [0, 0, 1, 1, 1, 1, 1],
        "line_num": [0, 0, 0, 1, 1, 1, 2],
        "text": ["", "", "", "", "Привет", "Айрис", "Открой"],
        "conf": [-1, -1, -1, -1, 96, 91, 88],
        "left": [0, 0, 0, 0, 10, 120, 10],
        "top": [0, 0, 0, 0, 20, 20, 70],
        "width": [800, 700, 700, 300, 100, 90, 110],
        "height": [600, 400, 400, 40, 30, 30, 30],
    }

    def test_only_word_level_boxes_become_blocks(self) -> None:
        """Уровни 1–4 — это страница, блок, абзац и строка, а не текст."""
        blocks = parse_image_data(self.DATA)
        assert [block.text for block in blocks] == ["Привет", "Айрис", "Открой"]

    def test_coordinates_become_a_rect(self) -> None:
        first = parse_image_data(self.DATA)[0]
        assert first.rect == winapi.Rect(10, 20, 110, 50)

    def test_confidence_is_scaled_to_a_fraction(self) -> None:
        """Tesseract считает в процентах, `OcrBlock.confidence` — в долях."""
        assert parse_image_data(self.DATA)[0].confidence == pytest.approx(0.96)

    def test_unreadable_boxes_are_dropped(self) -> None:
        """conf == -1 значит «нашёл, но не прочитал» — это не пустой текст."""
        data = dict(self.DATA)
        data["conf"] = [-1, -1, -1, -1, 96, -1, 88]
        assert [block.text for block in parse_image_data(data)] == ["Привет", "Открой"]

    def test_empty_data_gives_no_blocks(self) -> None:
        assert parse_image_data({}) == []

    def test_lines_are_rebuilt_from_the_numbering(self) -> None:
        """Перевод строки берётся из block/par/line, а не из второго запуска бинарника."""
        assert join_lines(self.DATA) == "Привет Айрис\nОткрой"

    def test_join_lines_survives_missing_numbering(self) -> None:
        assert join_lines({"text": ["Привет", "Айрис"]}) == "Привет Айрис"


class TestPaddleParsing:
    """Форма ответа PaddleOCR менялась между версиями дважды."""

    def test_the_nested_shape_is_flattened(self) -> None:
        raw = [
            [
                [[[10, 20], [110, 20], [110, 50], [10, 50]], ("Привет", 0.98)],
                [[[10, 70], [120, 70], [120, 100], [10, 100]], ("Айрис", 0.91)],
            ]
        ]
        blocks = parse_paddle_result(raw)
        assert [block.text for block in blocks] == ["Привет", "Айрис"]
        assert blocks[0].rect == winapi.Rect(10, 20, 110, 50)
        assert blocks[0].confidence == pytest.approx(0.98)

    def test_the_flat_shape_is_accepted_too(self) -> None:
        """Старая версия отдаёт список записей без обёртки по страницам.

        Отличить один такой список от страницы можно только по второму элементу:
        у записи там текст, у страницы — ещё одна запись с рамкой.
        """
        raw = [[[[0, 0], [50, 0], [50, 20], [0, 20]], ("Ок", 0.7)]]
        assert [block.text for block in parse_paddle_result(raw)] == ["Ок"]

    def test_nothing_found_is_not_an_error(self) -> None:
        """`[None]` — так Paddle сообщает, что текста нет."""
        assert parse_paddle_result([None]) == []
        assert parse_paddle_result([]) == []
        assert parse_paddle_result(None) == []

    def test_unrecognisable_entries_are_skipped(self) -> None:
        """Форма поменяется в третий раз — прочитаем то, что поняли."""
        raw = [[["мусор"], [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Да", 0.5)], None]]
        assert [block.text for block in parse_paddle_result(raw)] == ["Да"]

    def test_a_rotated_box_becomes_its_bounding_rect(self) -> None:
        """Paddle отдаёт четыре угла, и они не обязаны быть по осям."""
        raw = [[[[12, 30], [110, 20], [112, 52], [14, 62]], ("Косо", 0.6)]]
        assert parse_paddle_result(raw)[0].rect == winapi.Rect(12, 20, 112, 62)


class TestOcrText:
    """Результат: строки, фильтр по уверенности, координаты на экране."""

    TEXT: ClassVar[OcrText] = OcrText(
        text="Привет\nАйрис",
        blocks=(
            OcrBlock(text="Привет", rect=winapi.Rect(10, 20, 110, 50), confidence=0.95),
            OcrBlock(text="Айрис", rect=winapi.Rect(10, 70, 120, 100), confidence=0.30),
        ),
        engine="fake",
        language="ru",
    )

    def test_lines_ignore_the_blank_ones(self) -> None:
        assert OcrText(text="а\n\n  \nб").lines == ("а", "б")

    def test_empty_text_is_empty_even_with_spaces(self) -> None:
        assert OcrText(text="   \n ").is_empty

    def test_a_low_confidence_block_is_dropped(self) -> None:
        filtered = self.TEXT.filtered(0.5)
        assert [block.text for block in filtered.blocks] == ["Привет"]
        assert filtered.text == "Привет"

    def test_a_zero_threshold_changes_nothing(self) -> None:
        assert self.TEXT.filtered(0.0) is self.TEXT

    def test_blocks_without_confidence_survive_the_filter(self) -> None:
        """Windows OCR уверенности не сообщает вовсе.

        Если отбрасывать блоки без неё, настройка `min_confidence` превращается в
        «выключить движок по умолчанию».
        """
        text = OcrText(text="Привет", blocks=(OcrBlock(text="Привет"),))
        assert text.filtered(0.9).blocks == text.blocks

    def test_coordinates_are_mapped_back_onto_the_desktop(self) -> None:
        """Движок мерил в пикселях увеличенного кадра — вернуть на рабочий стол.

        Кадр сняли с (1000, 500), увеличили втрое, движок нашёл текст в (30, 60)
        своей картинки. На экране это (1010, 520) — иначе по такой координате
        нельзя ни кликнуть, ни подсветить.
        """
        moved = OcrText(
            blocks=(OcrBlock(text="Привет", rect=winapi.Rect(30, 60, 330, 150)),),
        ).on_screen(scale=3.0, origin=winapi.Rect(1000, 500, 1600, 900))
        assert moved.blocks[0].rect == winapi.Rect(1010, 520, 1110, 550)

    def test_a_negative_origin_is_kept_negative(self) -> None:
        """Монитор слева от основного — координаты в минусе и должны там остаться."""
        moved = OcrText(
            blocks=(OcrBlock(text="Привет", rect=winapi.Rect(20, 40, 120, 80)),),
        ).on_screen(scale=2.0, origin=winapi.Rect(-1920, -100, 0, 980))
        assert moved.blocks[0].rect == winapi.Rect(-1910, -80, -1860, -60)

    def test_scale_one_only_shifts(self) -> None:
        moved = OcrText(
            blocks=(OcrBlock(text="Привет", rect=winapi.Rect(5, 5, 15, 15)),),
        ).on_screen(scale=1.0, origin=winapi.Rect(100, 200, 400, 500))
        assert moved.blocks[0].rect == winapi.Rect(105, 205, 115, 215)

    def test_an_impossible_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            self.TEXT.on_screen(scale=0.0, origin=winapi.Rect())

    def test_as_dict_carries_the_blocks(self) -> None:
        described = self.TEXT.as_dict()
        assert described["engine"] == "fake"
        assert len(described["blocks"]) == 2


# --------------------------------------------------------------------------- #
# Подготовка картинки
# --------------------------------------------------------------------------- #


class TestPrepare:
    """Масштаб — от движка, а не от картинки."""

    def test_an_engine_that_does_not_want_it_gets_the_original(self) -> None:
        """Windows OCR обучен на экранном тексте: тройной 4K — девять раз память зря."""
        assert scale_for(ScreenEngine("ru"), 1920 * 1080, target_dpi=300) == 1.0

    def test_the_factor_comes_from_the_dpi(self) -> None:
        assert scale_for(FakeEngine("ru"), 400 * 200, target_dpi=288) == pytest.approx(3.0)
        assert scale_for(FakeEngine("ru"), 400 * 200, target_dpi=192) == pytest.approx(2.0)

    def test_a_dpi_of_96_means_no_scaling(self) -> None:
        assert scale_for(FakeEngine("ru"), 400 * 200, target_dpi=96) == 1.0

    def test_the_factor_is_capped(self) -> None:
        assert scale_for(FakeEngine("ru"), 100 * 100, target_dpi=600) == pytest.approx(MAX_SCALE)

    def test_a_huge_frame_is_scaled_less(self) -> None:
        """Предел по площади, а не по стороне: тройной 4K — сто мегапикселей.

        Обрезать картинку нельзя (текст в обрезанном не прочитается вовсе), поэтому
        уменьшается коэффициент.
        """
        pixels = 3840 * 2160
        factor = scale_for(FakeEngine("ru"), pixels, target_dpi=300)
        assert 1.0 < factor < 3.0
        assert pixels * factor**2 <= MAX_PIXELS * 1.01

    def test_an_enormous_frame_is_not_scaled_at_all(self) -> None:
        assert scale_for(FakeEngine("ru"), MAX_PIXELS * 2, target_dpi=300) == 1.0

    def test_an_empty_frame_does_not_divide_by_zero(self) -> None:
        assert scale_for(FakeEngine("ru"), 0, target_dpi=300) == 1.0

    def test_prepare_resizes_and_reports_the_factor(self) -> None:
        image, factor = prepare(make_frame(200, 100), FakeEngine("ru"), target_dpi=192)
        assert factor == pytest.approx(2.0)
        assert image.size == (400, 200)

    def test_prepare_leaves_a_screen_engine_alone(self) -> None:
        image, factor = prepare(make_frame(200, 100), ScreenEngine("ru"), target_dpi=300)
        assert factor == 1.0
        assert image.size == (200, 100)

    def test_an_engine_limit_shrinks_the_picture(self) -> None:
        """Windows отказывается от картинки больше 10000 px по стороне.

        Уменьшить — это разница между результатом и исключением, а коэффициент при
        этом надо пересчитать, иначе блоки уедут.
        """
        image, factor = prepare(make_frame(400, 200), SmallEngine("ru"), target_dpi=300)
        assert max(image.size) <= SmallEngine.max_dimension
        assert factor == pytest.approx(image.width / 400)

    def test_binarize_gives_two_values(self) -> None:
        flat = binarize(picture())
        assert flat.mode == "L"
        assert set(flat.tobytes()) <= {0, 255}

    def test_binarize_is_only_applied_when_asked(self) -> None:
        plain, _ = prepare(make_frame(40, 20), ScreenEngine("ru"), binarise=False)
        flat, _ = prepare(make_frame(40, 20), ScreenEngine("ru"), binarise=True)
        assert plain.mode == "RGB"
        assert flat.mode == "L"


# --------------------------------------------------------------------------- #
# Распознавание кадра
# --------------------------------------------------------------------------- #


class TestReadFrame:
    """Шов, через который проходят оба действия."""

    def test_a_reading_carries_the_engine_and_the_scale(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use(monkeypatch, FakeEngine)
        reading = read_frame(make_frame(200, 100), languages=("ru",), target_dpi=192)
        assert reading.choice.engine.name == "fake"
        assert reading.scale == pytest.approx(2.0)
        assert reading.text.text == "Привет"

    def test_the_engine_sees_the_prepared_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use(monkeypatch, FakeEngine)
        reading = read_frame(make_frame(200, 100), languages=("ru",), target_dpi=192)
        engine = reading.choice.engine
        assert isinstance(engine, FakeEngine)
        assert engine.seen == [(400, 200)]

    def test_blocks_come_back_in_screen_coordinates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Кадр снят не в нуле и увеличен — блок должен указывать на экран."""
        use(monkeypatch, FakeEngine)
        frame = make_frame(200, 100, rect=winapi.Rect(500, 300, 700, 400))
        reading = read_frame(frame, languages=("ru",), target_dpi=192)
        block = reading.text.blocks[0]
        assert block.rect == winapi.Rect(505, 310, 555, 325)

    def test_a_black_frame_is_refused_with_the_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Защищённое окно — чёрный кадр; распознавать в нём нечего, и это не «нет текста»."""
        use(monkeypatch, FakeEngine)
        black = Frame(width=8, height=8, bgra=bytes(8 * 8 * 4))
        with pytest.raises(ocr.OcrFailed) as caught:
            read_frame(black, languages=("ru",))
        assert "чёрный" in caught.value.user_message

    def test_a_low_confidence_block_is_filtered_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        use(monkeypatch, FakeEngine)
        engine = FakeEngine("ru")
        engine.blocks = (
            OcrBlock(text="точно", rect=winapi.Rect(0, 0, 10, 10), confidence=0.9),
            OcrBlock(text="вряд ли", rect=winapi.Rect(0, 20, 10, 30), confidence=0.2),
        )
        monkeypatch.setattr(
            "ayris.actions.system.ocr.select_engine",
            lambda *_args, **_kwargs: Choice(engine=engine, language="ru"),
        )
        reading = read_frame(make_frame(80, 40), min_confidence=0.5)
        assert [block.text for block in reading.text.blocks] == ["точно"]


# --------------------------------------------------------------------------- #
# Что происходит с текстом
# --------------------------------------------------------------------------- #


class TestSpeech:
    """Экран текста нельзя прочитать вслух целиком."""

    def test_a_short_text_is_read_whole(self) -> None:
        assert for_speech("Громкость 30 процентов", 400) == "Громкость 30 процентов"

    def test_line_breaks_become_spaces(self) -> None:
        """Синтезатор споткнётся на переводе строки посреди фразы."""
        assert for_speech("Привет\nАйрис", 400) == "Привет Айрис"

    def test_a_long_text_is_cut_at_a_word(self) -> None:
        text = " ".join(["словоформа"] * 60)
        spoken = for_speech(text, 60)
        assert len(spoken) < len(text)
        assert "дальше не читаю" in spoken
        head = spoken.split("…")[0]
        assert not head.endswith("словофор")

    def test_the_cut_mentions_the_clipboard(self) -> None:
        """Пользователь должен узнать, где остальное, а не догадываться."""
        assert "буфере обмена" in for_speech("я" * 500, 100)

    def test_a_single_long_word_is_still_cut(self) -> None:
        """Пробела в первой половине нет — режем по длине, а не отдаём всё."""
        spoken = for_speech("а" * 300, 50)
        assert len(spoken) < 300


class TestDelivery:
    """Буфер обмена и текст ответа."""

    @pytest.fixture
    def copied(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Перехваченный `copy_text`: что положили в буфер обмена."""
        put: list[str] = []

        def fake_copy(text: str) -> bool:
            put.append(text)
            return True

        monkeypatch.setattr(ocr, "copy_text", fake_copy)
        return put

    def reading(self, text: str = "Привет\nАйрис") -> Reading:
        return Reading(
            text=OcrText(text=text, blocks=(OcrBlock(text=text),), engine="fake", language="ru"),
            frame=make_frame(80, 40),
            choice=Choice(engine=FakeEngine("ru"), language="ru"),
        )

    def test_clipboard_mode_copies(self, copied: list[str]) -> None:
        result = ocr.deliver(self.reading(), TextOutput.CLIPBOARD)
        assert result.clipboard
        assert copied == ["Привет\nАйрис"]

    def test_speak_mode_does_not_copy(self, copied: list[str]) -> None:
        result = ocr.deliver(self.reading(), TextOutput.SPEAK)
        assert not result.clipboard
        assert copied == []

    def test_both_copies_too(self, copied: list[str]) -> None:
        assert ocr.deliver(self.reading(), TextOutput.BOTH).clipboard
        assert copied == ["Привет\nАйрис"]

    def test_nothing_recognised_is_not_copied(self, copied: list[str]) -> None:
        assert not ocr.deliver(self.reading(""), TextOutput.BOTH).clipboard
        assert copied == []

    def test_a_refusing_clipboard_does_not_lose_the_reading(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Буфер держит чужой процесс — текст всё равно распознан."""
        monkeypatch.setattr(ocr, "copy_text", lambda _text: False)
        result = ocr.deliver(self.reading(), TextOutput.BOTH)
        assert not result.clipboard
        assert result.text.text == "Привет\nАйрис"

    def test_copy_text_ignores_empty(self) -> None:
        assert not ocr.copy_text("   ")


# --------------------------------------------------------------------------- #
# Действия
# --------------------------------------------------------------------------- #


class TestActions:
    """То, что видит реестр."""

    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Поддельный движок, поддельный захват, один монитор, буфер обмена молчит."""
        use(monkeypatch, FakeEngine)
        one_monitor(monkeypatch)
        monkeypatch.setattr(ocr, "copy_text", lambda _text: True)

        class Backend:
            @staticmethod
            def grab(rect: winapi.Rect) -> Frame:
                return make_frame(max(rect.width, 1), max(rect.height, 1), rect=rect)

        shot_module.set_capture_backend(Backend())
        try:
            yield
        finally:
            shot_module.set_capture_backend(None)

    def test_the_metadata_is_registered(self) -> None:
        """Реестр находит оба действия, и таймаут учитывает медленный движок."""
        names = {entry.name for entry in registered_actions()}
        assert {"OcrScreen", "OcrRegion"} <= names
        assert OcrScreen.meta.timeout_ms >= 60_000
        assert OcrRegion.meta.timeout_ms > OcrScreen.meta.timeout_ms

    def test_ocr_screen_reads_the_whole_desktop(self) -> None:
        result = OcrScreen().run(OcrScreen.Params(output=TextOutput.CLIPBOARD))
        assert result.ok
        assert isinstance(result.value, Reading)
        assert result.value.text.text == "Привет"
        assert "На экране" in result.message_ru

    def test_ocr_screen_names_the_monitor(self) -> None:
        result = OcrScreen().run(OcrScreen.Params(monitor="primary", output=TextOutput.CLIPBOARD))
        assert result.ok
        assert "Монитор 1" in result.message_ru

    def test_ocr_screen_names_the_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ayris.actions.system import windows as win

        found = win.WindowRecord(hwnd=7, title="Блокнот — заметки", process="notepad.exe", pid=3)
        monkeypatch.setattr(ocr, "list_windows", lambda _query: [found])
        monkeypatch.setattr(ocr, "select_window", lambda _records, _query: found)
        monkeypatch.setattr(
            winapi,
            "extended_frame_bounds",
            lambda _hwnd: winapi.Rect(0, 0, 800, 600),
        )
        result = OcrScreen().run(OcrScreen.Params(window="Блокнот", output=TextOutput.CLIPBOARD))
        assert result.ok
        assert "Блокнот" in result.message_ru

    def test_ocr_region_by_numbers_does_not_ask(self) -> None:
        result = OcrRegion().run(
            OcrRegion.Params(left=10, top=20, width=300, height=200, output=TextOutput.CLIPBOARD)
        )
        assert result.ok
        assert result.value is not None
        assert result.value.frame.rect == winapi.Rect(10, 20, 310, 220)

    def test_ocr_region_reports_a_cancelled_selection(self) -> None:
        """Esc — не ошибка, и распознавать после него нечего."""
        shot_module.set_region_provider(lambda _timeout_s, _dim: None)
        try:
            result = OcrRegion().run(OcrRegion.Params())
        finally:
            shot_module.set_region_provider(None)
        assert result.ok
        assert result.value is None
        assert "Отменила" in result.message_ru

    def test_ocr_region_reads_what_was_dragged(self) -> None:
        shot_module.set_region_provider(lambda _timeout_s, _dim: winapi.Rect(100, 100, 500, 300))
        try:
            result = OcrRegion().run(OcrRegion.Params(output=TextOutput.CLIPBOARD))
        finally:
            shot_module.set_region_provider(None)
        assert result.ok
        assert result.value is not None
        assert result.value.frame.rect == winapi.Rect(100, 100, 500, 300)

    def test_half_a_rectangle_is_refused_at_validation(self) -> None:
        with pytest.raises(ValueError, match="высоту"):
            OcrRegion.Params(width=300)

    def test_the_language_override_is_a_one_item_wish_list(self) -> None:
        assert OcrScreen.Params(language=" ru ").languages == ("ru",)
        assert OcrScreen.Params().languages == ()

    def test_no_text_found_is_said_plainly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """«Текста не нашла» — это успех действия, а не сбой."""
        engine = FakeEngine("ru")
        engine.blocks = ()
        monkeypatch.setattr(
            "ayris.actions.system.ocr.select_engine",
            lambda *_args, **_kwargs: Choice(engine=engine, language="ru"),
        )
        result = OcrScreen().run(OcrScreen.Params(output=TextOutput.CLIPBOARD))
        assert result.ok
        assert "не нашла" in result.message_ru

    def test_the_fallback_is_mentioned_in_the_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Про подмену движка пользователь узнаёт в том же ответе, один раз."""
        monkeypatch.setattr(
            "ayris.actions.system.ocr.select_engine",
            lambda *_args, **_kwargs: Choice(
                engine=FakeEngine("ru"),
                language="ru",
                fallback_from="missing",
                reason="не установлен",
            ),
        )
        result = OcrScreen().run(OcrScreen.Params(output=TextOutput.CLIPBOARD))
        assert "Поддельный" in result.message_ru

    def test_the_settings_decide_when_nothing_is_passed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Пустой `output` в параметрах — значит как в `[actions.ocr]`."""
        base = config_module.get_settings()

        def speak_only() -> config_module.Settings:
            section = base.actions.ocr.model_copy(update={"output": "speak"})
            actions = base.actions.model_copy(update={"ocr": section})
            return base.model_copy(update={"actions": actions})

        monkeypatch.setattr(ocr, "get_settings", speak_only)
        result = OcrScreen().run(OcrScreen.Params())
        assert result.ok
        assert result.value is not None
        assert not result.value.clipboard
        assert result.message_ru.startswith("Привет")


# --------------------------------------------------------------------------- #
# Ожидание WinRT и COM-апартаменты
# --------------------------------------------------------------------------- #


class TestApartments:
    """Блокирующее ожидание там, где COM на потоке уже поднят как STA.

    Не выдумка: `comtypes` (его тянут действия с громкостью) оставляет свой поток
    в STA навсегда, и OCR, попавший на тот же поток следом, падал с `RuntimeError`
    про apartment. Ловится только совместным прогоном файлов, поэтому проверка тут.
    """

    class Sta:
        """Операция, которая отдаёт результат только не с исходного потока."""

        def __init__(self, home: int) -> None:
            self.home = home
            self.calls = 0

        def get(self) -> str:
            self.calls += 1
            if threading.get_ident() == self.home:
                raise RuntimeError("Cannot call blocking method from single-threaded apartment.")
            return "готово"

    def test_the_wait_moves_onto_another_thread(self) -> None:
        operation = self.Sta(threading.get_ident())
        assert _await(operation) == "готово"
        assert operation.calls == 2

    def test_a_plain_result_costs_no_thread(self) -> None:
        class Plain:
            calls = 0

            def get(self) -> str:
                type(self).calls += 1
                return "сразу"

        assert _await(Plain()) == "сразу"
        assert Plain.calls == 1

    def test_other_runtime_errors_are_not_retried(self) -> None:
        class Broken:
            calls = 0

            def get(self) -> str:
                type(self).calls += 1
                raise RuntimeError("бумс")

        with pytest.raises(RuntimeError, match="бумс"):
            _await(Broken())
        assert Broken.calls == 1

    def test_a_failure_on_the_other_thread_comes_back(self) -> None:
        """Ошибка изнутри потока — вызывающему, а не в пустоту."""

        class Failing:
            asked = 0

            def get(self) -> str:
                type(self).asked += 1
                if type(self).asked == 1:
                    raise RuntimeError("blocking method from single-threaded apartment")
                raise OSError("движок отказал")

        with pytest.raises(OSError, match="движок отказал"):
            _await(Failing())


# --------------------------------------------------------------------------- #
# Настоящий движок на настоящих картинках
# --------------------------------------------------------------------------- #


class TestRealRecognition:
    """Windows OCR на подготовленных PNG. Пропускается без языкового пакета."""

    @staticmethod
    def engine_for(language: str) -> WindowsOcr:
        """Движок нужного языка, или `skip` с причиной.

        Отсутствие пакета распознавания — обычное состояние Windows, установленной
        не на русском. Красить прогон за это нельзя: проверять нечего, а не сломано.
        """
        if not WindowsOcr.is_available():
            pytest.skip("в системе нет ни одного пакета распознавания Windows OCR")
        tag = WindowsOcr.match_language((language,))
        if tag is None:
            installed = ", ".join(WindowsOcr.available_languages())
            pytest.skip(f"нет пакета распознавания для «{language}»; есть: {installed}")
        return WindowsOcr(tag)

    def image(self, name: str) -> Image.Image:
        path = IMAGES / name
        if not path.is_file():  # pragma: no cover - фикстуры лежат в репозитории
            pytest.skip(f"нет фикстуры {name}")
        with Image.open(path) as source:
            return source.convert("RGB")

    def test_russian_text_is_read(self) -> None:
        """Русский PNG читается русским движком — построчно и без ошибок."""
        engine = self.engine_for("ru")
        text = engine.recognise(self.image("text_ru.png"))
        assert "Привет" in text.text
        assert len(text.lines) == 3
        assert "браузер" in text.text

    def test_english_text_is_read_by_the_english_engine(self) -> None:
        """Английский — английским движком: русский прочитает «Hello» как «Нено».

        Один движок Windows OCR знает один язык, и это единственная причина, по
        которой порядок языков в настройках вообще что-то значит.
        """
        engine = self.engine_for("en")
        text = engine.recognise(self.image("text_en.png"))
        assert "Hello" in text.text
        assert "browser" in text.text

    def test_blocks_have_boxes_inside_the_picture(self) -> None:
        """Координаты блоков — в пикселях переданной картинки, а не экрана."""
        engine = self.engine_for("ru")
        image = self.image("text_ru.png")
        text = engine.recognise(image)
        assert text.blocks
        for block in text.blocks:
            assert 0 <= block.rect.left < block.rect.right <= image.width
            assert 0 <= block.rect.top < block.rect.bottom <= image.height

    def test_a_blank_picture_gives_no_text(self) -> None:
        engine = self.engine_for("ru")
        assert engine.recognise(self.image("blank.png")).is_empty

    def test_the_max_side_is_reported(self) -> None:
        """Предел приходит из WinRT, поэтому это метод, а не константа."""
        self.engine_for("ru")
        assert WindowsOcr.max_side() > 0
