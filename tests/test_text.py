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


def test_phoneme_frontend():
    import pytest
    pytest.importorskip("ctypes")
    from persian_tts.text import get_frontend
    from persian_tts.text.phonemes import BOS_ID, EOS_ID, PAD_ID, phonemize, unknown_phonemes
    p = phonemize("سلام، حال شما چطور است؟")
    assert p.startswith("salˈɑm,") and p.endswith("?") and not unknown_phonemes(p)
    symbols, encode = get_frontend("phonemes")
    ids = encode("کتاب")
    assert len(symbols) == 256 and ids[0] == BOS_ID and ids[-1] == EOS_ID and ids[1::2] == [PAD_ID] * (len(ids) // 2)
