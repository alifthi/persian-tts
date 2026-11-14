"""Persian text normalization for TTS.

Unifies Arabic/Persian code points, digits, whitespace and zero-width non-joiner (ZWNJ),
expands numbers and common symbols to words, and strips characters outside the vocabulary.
"""
import re
import unicodedata

from .numbers import number_to_words

ZWNJ = "‌"

_CHAR_MAP = {
    # Arabic -> Persian letters
    "ي": "ی", "ى": "ی", "ئ": "ئ", "ك": "ک", "ۀ": "هٔ", "ة": "ه", "ٱ": "ا", "إ": "ا", "أ": "أ",
    "ٲ": "ا", "ٳ": "ا", "ڪ": "ک", "ھ": "ه", "ہ": "ه", "ە": "ه", "ۍ": "ی", "ې": "ی", "ے": "ی",
    # punctuation
    "«": '"', "»": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "٫": ".", "٬": ",", "،": "،",
    "؛": "؛", "؟": "؟", "…": "...", "–": "-", "—": "-", "ـ": "",  # tatweel removed
    # whitespace variants
    " ": " ", " ": " ", " ": " ", "​": "", "‍": "", "‎": "", "‏": "",
    "﻿": "",
}
for i in range(10):
    _CHAR_MAP[chr(0x06F0 + i)] = str(i)  # Persian digits
    _CHAR_MAP[chr(0x0660 + i)] = str(i)  # Arabic-Indic digits

_TRANS = str.maketrans(_CHAR_MAP)

_SYMBOLS = {
    "%": " درصد ", "٪": " درصد ", "&": " و ", "+": " به اضافه ", "=": " مساوی ", "@": " ات ",
    "$": " دلار ", "€": " یورو ", "#": " شماره ", "/": " ", "\\": " ", "|": " ",
}

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_ZWNJ = re.compile(ZWNJ + "{2,}")


def _expand_number(m: re.Match) -> str:
    tok = m.group(0)
    # treat "," as thousands separator, "." as decimal (only the last '.')
    if tok.count(".") > 1:
        return " ".join(number_to_words(p) for p in tok.split(".") if p)
    try:
        return " " + number_to_words(tok) + " "
    except ValueError:
        return " ".join(number_to_words(d) for d in tok if d.isdigit())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TRANS)
    text = text.replace("\r", " ").replace("\n", " ")
    for k, v in _SYMBOLS.items():
        text = text.replace(k, v)
    text = _NUM_RE.sub(_expand_number, text)
    text = _MULTI_ZWNJ.sub(ZWNJ, text)
    # ZWNJ next to a space is meaningless
    text = re.sub(f" ?{ZWNJ} ?", lambda m: ZWNJ if m.group(0) == ZWNJ else " ", text)
    text = _MULTI_SPACE.sub(" ", text)
    # no space before punctuation
    text = re.sub(r" ([.,،؛:!؟?])", r"\1", text)
    return text.strip()
