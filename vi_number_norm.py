#!/usr/bin/env python3
"""Vietnamese numbers read as words, and equivalent spellings merged (ctctknk-docs/30).

Parakeet always *speaks* numbers ("hai mươi sáu phẩy hai mươi lăm") while
FLEURS / LSVSC transcripts write digits ("26,25"); utterances with digits
carried ~half of the teacher's word errors on those sets, and as CTC targets
they contradict the teacher's logits.  ``verbalize`` rewrites digits the way
the teacher says them (majority forms counted in its hypotheses on 22/09:
"mốt" 353:3, "lăm" 275:65, "mươi tư" 96:51, "linh" 226:62, decimal "phẩy").

``canonical`` merges spellings the teacher itself uses interchangeably
("nghìn"/"ngàn" 1139:860, "tỷ"/"tỉ" 376:853, "mươi tư"/"mươi bốn", "trăm
linh"/"trăm lẻ"); scoring applies it to reference and hypothesis alike.

Run the module to execute its test cases.
"""

from __future__ import annotations

import re
import unicodedata

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALES = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ"]
MAX_READ_AS_NUMBER = 15          # longer digit runs (ids) are read digit by digit

# 1.234.567 (Vietnamese thousands) | 1,234,567 (English thousands) | decimals | integers
NUMBER = re.compile(
    r"(?<![\w])(?:"
    r"(?P<dot_thousands>\d{1,3}(?:\.\d{3})+)(?:,(?P<dot_frac>\d+))?"
    r"|(?P<comma_thousands>\d{1,3}(?:,\d{3}){2,})"
    r"|(?P<int>\d+)(?:(?P<sep>[.,])(?P<frac>\d+))?"
    r")(?P<percent>\s?%)?(?![\d])")


def read_digits(digits: str) -> str:
    return " ".join(DIGITS[int(d)] for d in digits)


def below_thousand(n: int, full: bool) -> list[str]:
    """0 < n < 1000.  ``full``: a higher group precedes, so write 'không trăm' / 'linh'."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words: list[str] = []
    if hundreds or full:
        words += [DIGITS[hundreds], "trăm"]
    if tens == 0:
        if units:
            words += (["linh"] if words else []) + [DIGITS[units]]
        return words
    words += ["mười"] if tens == 1 else [DIGITS[tens], "mươi"]
    if units == 1 and tens > 1:
        words.append("mốt")
    elif units == 4 and tens > 1:
        words.append("tư")
    elif units == 5:
        words.append("lăm")
    elif units:
        words.append(DIGITS[units])
    return words


def read_integer(digits: str) -> str:
    digits = digits.lstrip("0") or "0"
    if len(digits) > MAX_READ_AS_NUMBER:
        return read_digits(digits)
    n = int(digits)
    if n == 0:
        return DIGITS[0]
    groups = []
    while n:
        n, group = divmod(n, 1000)
        groups.append(group)
    top = len(groups) - 1
    words: list[str] = []
    for index in range(top, -1, -1):
        if groups[index] == 0:
            continue
        words += below_thousand(groups[index], full=index < top)
        if SCALES[index]:
            words.append(SCALES[index])
    return " ".join(words)


def read_fraction(digits: str) -> str:
    """'25' -> 'hai mươi lăm' (as the teacher reads it); leading zeros one by one."""
    rest = digits.lstrip("0")
    parts = [DIGITS[0]] * (len(digits) - len(rest))
    if rest:
        parts.append(read_integer(rest) if len(rest) <= 3 else read_digits(rest))
    return " ".join(parts)


def _replace(match: re.Match) -> str:
    groups = match.groupdict()
    if groups["dot_thousands"]:
        words = read_integer(groups["dot_thousands"].replace(".", ""))
        if groups["dot_frac"]:
            words += " phẩy " + read_fraction(groups["dot_frac"])
    elif groups["comma_thousands"]:
        words = read_integer(groups["comma_thousands"].replace(",", ""))
    else:
        integer = groups["int"]
        words = (read_digits(integer) if integer.startswith("0") and len(integer) > 1
                 else read_integer(integer))
        if groups["frac"] is not None:
            words += (" phẩy " if groups["sep"] == "," else " chấm ") + read_fraction(groups["frac"])
    if groups["percent"]:
        words += " phần trăm"
    return f" {words} "


def verbalize(text: str) -> str:
    """Digits -> Vietnamese words; letters glued to digits are split ('11n' -> '11 n')."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", text)
    return NUMBER.sub(_replace, text)


CANONICAL = [
    (re.compile(r"\bngàn\b"), "nghìn"),
    (re.compile(r"\btỉ\b"), "tỷ"),
    (re.compile(r"\bmươi bốn\b"), "mươi tư"),
    (re.compile(r"\btrăm lẻ\b"), "trăm linh"),
]


def base_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    kept = [c if unicodedata.category(c).startswith(("L", "N")) else " " for c in text]
    return " ".join("".join(kept).split())


def canonical(normalized: str) -> str:
    for pattern, replacement in CANONICAL:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def normalize_for_asr(text: str) -> str:
    """Numbers as words, lower case, letters/digits only, equivalent spellings merged."""
    return canonical(base_normalize(verbalize(text)))


CASES = {
    "0": "không", "5": "năm", "10": "mười", "15": "mười lăm", "14": "mười bốn",
    "21": "hai mươi mốt", "24": "hai mươi tư", "25": "hai mươi lăm", "30": "ba mươi",
    "105": "một trăm linh năm", "110": "một trăm mười",
    "1005": "một nghìn không trăm linh năm", "2020": "hai nghìn không trăm hai mươi",
    "146950": "một trăm bốn mươi sáu nghìn chín trăm năm mươi",
    "146.950": "một trăm bốn mươi sáu nghìn chín trăm năm mươi",
    "1.000.000": "một triệu", "3000000000": "ba tỷ",
    "1.500.000": "một triệu năm trăm nghìn",
    "26,25": "hai mươi sáu phẩy hai mươi lăm", "2.4": "hai chấm bốn", "5.0": "năm chấm không",
    "0,05": "không phẩy không năm", "15%": "mười lăm phần trăm",
    "0912345678": "không chín một hai ba bốn năm sáu bảy tám",
    "802.11n": "tám trăm linh hai chấm mười một n",
    "ngàn tỉ": "nghìn tỷ", "hai mươi bốn": "hai mươi tư", "một trăm lẻ năm": "một trăm linh năm",
    "Năm 1945, 70% dân số": "năm một nghìn chín trăm bốn mươi lăm bảy mươi phần trăm dân số",
    "Lịch sử 26,25 điểm": "lịch sử hai mươi sáu phẩy hai mươi lăm điểm",
}


if __name__ == "__main__":
    failures = 0
    for source, expected in CASES.items():
        got = normalize_for_asr(source)
        failures += got != expected
        print(f"{'PASS' if got == expected else 'FAIL'}  {source!r:>24} -> {got!r}"
              + ("" if got == expected else f"  (expected {expected!r})"))
    print("ALL PASS" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
