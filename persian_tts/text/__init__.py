"""Text frontend: normalize Persian text and map it to model token ids."""
from typing import List

from .normalizer import normalize
from .symbols import BLANK_ID, PAD_ID, SYMBOL_TO_ID, SYMBOLS


def text_to_ids(text: str, add_blank: bool = True, already_normalized: bool = False) -> List[int]:
    if not already_normalized:
        text = normalize(text)
    ids = [SYMBOL_TO_ID[c] for c in text.lower() if c in SYMBOL_TO_ID]
    if add_blank:
        out = [BLANK_ID] * (len(ids) * 2 + 1)
        out[1::2] = ids
        ids = out
    return ids


def get_frontend(name: str = "chars"):
    """-> (symbols, encode(text, already_normalized=False) -> ids). 'phonemes' needs espeak-ng."""
    if name == "chars":
        return SYMBOLS, lambda t, already_normalized=False: text_to_ids(t, True, already_normalized)
    if name == "phonemes":
        from .phonemes import PHONEME_SYMBOLS, phonemize, phonemes_to_ids
        return PHONEME_SYMBOLS, lambda t, already_normalized=False: phonemes_to_ids(phonemize(t, already_normalized))
    raise ValueError(f"unknown text frontend {name!r} (chars | phonemes)")


def unknown_chars(text: str) -> set:
    return {c for c in text.lower() if c not in SYMBOL_TO_ID}


__all__ = ["normalize", "text_to_ids", "unknown_chars", "get_frontend", "SYMBOLS", "PAD_ID", "BLANK_ID"]
