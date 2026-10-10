"""Проверки распознавания клавиатурной белиберды (:mod:`ayris.nlu.gibberish`)."""

from __future__ import annotations

import pytest

from ayris.nlu.gibberish import looks_like_gibberish


class TestLooksLikeGibberish:
    """Узко: только явный клавиатурный мусор, не короткие слова и реакции."""

    @pytest.mark.parametrize(
        "text",
        [
            "фывп",  # скольжение по домашнему ряду ЙЦУКЕН
            "фыва",
            "йцукен",
            "ячсми",
            "asdf",
            "qwerty",
            "ждлкпр",  # согласные без гласных
            "  Фывп  ",  # регистр и пробелы по краям не мешают
        ],
    )
    def test_keyboard_mash_is_gibberish(self, text: str) -> None:
        assert looks_like_gibberish(text)

    @pytest.mark.parametrize(
        "text",
        [
            "нишево",  # реальная реплика-реакция — её разбирает модель, не гейт
            "привет",
            "вода",  # домашний ряд, но настоящее слово
            "блин",
            "ахах",
            "как",  # короткое слово в стороне не трогаем
            "что",
            "расскажи про черепах",  # несколько слов — почти всегда реальная фраза
            "погода",
            "ааааа",  # один повторённый символ — это эмоция, не мусор
            "ммммм",
        ],
    )
    def test_real_words_and_reactions_are_not_gibberish(self, text: str) -> None:
        assert not looks_like_gibberish(text)

    @pytest.mark.parametrize("text", ["", "   ", "фы", "ab", "12", "!!!", "фывп123"])
    def test_too_short_or_not_pure_letters_is_not_gibberish(self, text: str) -> None:
        assert not looks_like_gibberish(text)
