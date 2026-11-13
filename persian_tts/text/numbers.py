"""Persian number-to-words conversion (integers and decimals)."""

_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون", "کوادریلیون"]
_AND = " و "


def _three_digits(n: int) -> str:
    parts = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rest < 20:
        parts.append(_TEENS[rest - 10])
    else:
        t, o = divmod(rest, 10)
        if t:
            parts.append(_TENS[t])
        if o:
            parts.append(_ONES[o])
    return _AND.join(parts)


def int_to_words(n: int) -> str:
    if n == 0:
        return "صفر"
    if n < 0:
        return "منفی " + int_to_words(-n)
    groups = []
    i = 0
    while n > 0:
        n, g = divmod(n, 1000)
        if g:
            if i >= len(_SCALES):
                raise ValueError("number too large")
            words = _three_digits(g)
            if i == 1 and g == 1:
                words = "هزار"  # "یک هزار" -> "هزار"
            elif _SCALES[i]:
                words = f"{words} {_SCALES[i]}"
            groups.append(words)
        i += 1
    return _AND.join(reversed(groups))


_FRACTION_NAMES = {1: "دهم", 2: "صدم", 3: "هزارم", 4: "ده‌هزارم", 5: "صدهزارم", 6: "میلیونم"}


def number_to_words(s: str) -> str:
    """Convert a numeric string with ASCII digits (optionally with '.' decimal) to Persian words."""
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        frac = frac.rstrip("0")
        w = int_to_words(int(whole or "0"))
        if not frac:
            return w
        if len(frac) in _FRACTION_NAMES:
            return f"{w} ممیز {int_to_words(int(frac))} {_FRACTION_NAMES[len(frac)]}"
        return f"{w} ممیز " + " ".join(int_to_words(int(d)) for d in frac)
    return int_to_words(int(s))
