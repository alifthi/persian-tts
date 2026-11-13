"""Character vocabulary. Index 0 is padding, index 1 is the inter-character blank (VITS 'add_blank')."""

PAD = "_"
BLANK = "~"  # inserted between characters, never appears in text
PUNCT = " !\"'(),-.:;?،؛؟"
# Persian alphabet + hamza variants + common diacritics (harakat), ZWNJ
LETTERS = "آءأؤئابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیٔ"
DIACRITICS = "ًٌٍَُِّْٰ"  # tanvin, fatha, damma, kasra, shadda, sukun, dagger alef
ZWNJ = "‌"
LATIN = "abcdefghijklmnopqrstuvwxyz"  # lowercased; rare in data, kept so brand names do not vanish

SYMBOLS = [PAD, BLANK] + list(PUNCT) + list(LETTERS) + list(DIACRITICS) + [ZWNJ] + list(LATIN)
assert len(SYMBOLS) == len(set(SYMBOLS)), "duplicate symbols"
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD_ID = 0
BLANK_ID = 1
