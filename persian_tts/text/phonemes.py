"""Phoneme frontend: Persian text -> espeak-ng IPA -> Piper phoneme ids.

Persian script omits short vowels (کرد = kard / kord / kerd), so a character model must guess
pronunciation. espeak-ng's `fa` voice resolves them, and its IPA is exactly what the Piper base
model was trained on, so the pretrained phoneme embeddings transfer 1:1.

espeak-ng (GPL-3.0) is loaded with ctypes and is needed at training *and* inference time.
"""
import ctypes
import ctypes.util
import json
import re
from pathlib import Path
from typing import List

from .normalizer import normalize

# Piper's phoneme table (index = id). Unused slots are placeholders so the embedding shape matches.
PHONEME_SYMBOLS: List[str] = json.loads((Path(__file__).with_name("piper_phonemes.json")).read_text(encoding="utf-8"))
PHONEME_TO_ID = {s: i for i, s in enumerate(PHONEME_SYMBOLS) if not s.startswith("<unused")}
PAD_ID, BOS_ID, EOS_ID = PHONEME_TO_ID["_"], PHONEME_TO_ID["^"], PHONEME_TO_ID["$"]

_PUNCT_MAP = {"،": ",", "؛": ";", "؟": "?", ",": ",", ";": ";", "?": "?", ".": ".", "!": "!", ":": ":"}
_SPLIT = re.compile(r"([،؛؟,;?.!:])")
_DROP = re.compile(r"[‌‍()\"«»\-]")  # ZWNJ/ZWJ and bracket/quote marks are not phonemes

_lib = None


def _espeak():
    global _lib
    if _lib is None:
        path = ctypes.util.find_library("espeak-ng") or "libespeak-ng.so.1"
        lib = ctypes.cdll.LoadLibrary(path)
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        if lib.espeak_Initialize(0x02, 0, None, 0) < 0:  # AUDIO_OUTPUT_SYNCHRONOUS, no audio produced
            raise RuntimeError("espeak-ng initialization failed")
        if lib.espeak_SetVoiceByName(b"fa") != 0:
            raise RuntimeError("espeak-ng has no 'fa' voice")
        _lib = lib
    return _lib


def _ipa(chunk: str) -> str:
    lib = _espeak()
    buf = ctypes.create_string_buffer(chunk.encode("utf-8"))
    ptr = ctypes.c_void_p(ctypes.addressof(buf))
    parts = []
    while ptr.value:  # espeak returns one clause per call and advances the pointer
        r = lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, 0x02)  # espeakCHARS_UTF8, IPA output
        if r:
            parts.append(r.decode("utf-8").strip())
    return " ".join(p for p in parts if p)


def phonemize(text: str, already_normalized: bool = False) -> str:
    """Normalized Persian text -> IPA string with Piper-style punctuation (', ', '. ', ...)."""
    if not already_normalized:
        text = normalize(text)
    text = _DROP.sub(" ", text)
    out = []
    for piece in _SPLIT.split(text):
        if not piece.strip():
            continue
        if piece in _PUNCT_MAP:
            if out:
                out[-1] = out[-1].rstrip() + _PUNCT_MAP[piece] + " "
            continue
        out.append(_ipa(piece.strip()) + " ")
    return "".join(out).strip()


def phonemes_to_ids(phonemes: str) -> List[int]:
    """Piper layout: ^ _ p1 _ p2 _ ... _ $ (pad between phonemes acts as the VITS blank)."""
    ids = [BOS_ID, PAD_ID]
    for ch in phonemes:
        i = PHONEME_TO_ID.get(ch)
        if i is not None:
            ids += [i, PAD_ID]
    ids.append(EOS_ID)
    return ids


def unknown_phonemes(phonemes: str) -> set:
    return {c for c in phonemes if c not in PHONEME_TO_ID}
