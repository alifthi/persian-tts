"""Map each model input symbol to IPA phonemes of the pretrained (English) Piper base model.

Used only once, to initialize the Persian character embeddings from the pretrained phoneme
embeddings (mean of listed phonemes), so the fine-tune starts from meaningful acoustics.
Phonemes are restricted to ones English espeak actually produces (others were never trained).
"""
from .symbols import BLANK, PAD

CHAR_TO_IPA = {
    PAD: ["_"], BLANK: ["_"],  # Piper intersperses '_' between phonemes, same role as our blank
    " ": [" "], "!": ["!"], '"': ["'"], "'": ["'"], "(": ["("], ")": [")"], ",": [","], "-": ["-"],
    ".": ["."], ":": [":"], ";": [";"], "?": ["?"], "،": [","], "؛": [";"], "؟": ["?"],
    # letters
    "آ": ["ɑ", "ː"], "ا": ["ɑ", "ː"], "ء": ["ʔ"], "أ": ["ʔ", "æ"], "ؤ": ["ʔ", "o"], "ئ": ["ʔ", "e"],
    "ب": ["b"], "پ": ["p"], "ت": ["t"], "ث": ["s"], "ج": ["d", "ʒ"], "چ": ["t", "ʃ"], "ح": ["h"],
    "خ": ["k", "h"], "د": ["d"], "ذ": ["z"], "ر": ["ɾ"], "ز": ["z"], "ژ": ["ʒ"], "س": ["s"],
    "ش": ["ʃ"], "ص": ["s"], "ض": ["z"], "ط": ["t"], "ظ": ["z"], "ع": ["ʔ"], "غ": ["ɡ"], "ف": ["f"],
    "ق": ["ɡ"], "ک": ["k"], "گ": ["ɡ"], "ل": ["l"], "م": ["m"], "ن": ["n"], "و": ["v", "u"],
    "ه": ["h"], "ی": ["j", "i"], "ٔ": ["e"],
    # diacritics
    "ً": ["æ", "n"], "ٌ": ["o", "n"], "ٍ": ["e", "n"], "َ": ["æ"], "ُ": ["o"],
    "ِ": ["e"], "ّ": ["ː"], "ْ": ["_"], "ٰ": ["ɑ"],
    "‌": ["_"],  # ZWNJ: word-internal boundary
}
# latin letters map to themselves where the base vocabulary has them
for _c in "abcdefghijklmnopqrstuvwxyz":
    CHAR_TO_IPA.setdefault(_c, ["ɡ"] if _c == "g" else [_c])
