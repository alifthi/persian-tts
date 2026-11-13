from persian_tts.text import normalize, text_to_ids, unknown_chars, BLANK_ID
from persian_tts.text.numbers import int_to_words, number_to_words


def test_int_to_words():
    assert int_to_words(0) == "صفر"
    assert int_to_words(15) == "پانزده"
    assert int_to_words(121) == "صد و بیست و یک"
    assert int_to_words(1000) == "هزار"
    assert int_to_words(2_000_005) == "دو میلیون و پنج"
    assert number_to_words("3.5") == "سه ممیز پنج دهم"


def test_normalize_arabic_letters_and_digits():
    out = normalize("كتاب علي ۱۲ صفحه دارد")
    assert "ک" in out and "ی" in out
    assert "دوازده" in out
    assert not unknown_chars(out)


def test_zwnj_and_spaces():
    assert normalize("می‌‌روم  ") == "می‌روم"
    assert normalize("سلام ، خوبی ؟") == "سلام، خوبی؟"


def test_text_to_ids_blank():
    ids = text_to_ids("سلام")
    assert len(ids) == 9 and ids[0] == BLANK_ID and ids[-1] == BLANK_ID
