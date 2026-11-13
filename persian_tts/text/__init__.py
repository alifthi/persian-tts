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


def unknown_chars(text: str) -> set:
    return {c for c in text.lower() if c not in SYMBOL_TO_ID}


__all__ = ["normalize", "text_to_ids", "unknown_chars", "SYMBOLS", "PAD_ID", "BLANK_ID"]
