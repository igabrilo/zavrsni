# Unicode normalisation: strips homoglyph attacks before regex matching.

from __future__ import annotations

import unicodedata
from typing import Iterable

_HOMOGLYPH_MAP = {
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ї": "i", "ј": "j", "ѕ": "s", "ԛ": "q", "ԝ": "w", "ѵ": "v",
    "ƅ": "b", "ḏ": "d", "ƒ": "f", "ɡ": "g", "һ": "h", "ӏ": "l", "ᴍ": "m",
    "ո": "n", "ⲣ": "p",
    # Cyrillic uppercase
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X",
    "І": "I", "Ї": "I", "Ј": "J", "Ѕ": "S", "В": "B", "Н": "H", "К": "K",
    "М": "M", "Т": "T",
    # Greek look-alikes
    "α": "a", "ο": "o", "ρ": "p", "υ": "u", "ν": "v", "Α": "A", "Β": "B",
    "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ζ": "Z",
}

_ZERO_WIDTH = "​‌‍⁠﻿"


def strip_zero_width(text: str) -> str:
    if not text:
        return text
    out = text
    for ch in _ZERO_WIDTH:
        if ch in out:
            out = out.replace(ch, "")
    return out


def normalize_text(text: str) -> str:
    if not text:
        return text
    s = strip_zero_width(text)
    s = unicodedata.normalize("NFKC", s)
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in s)


def normalize_strings(items: Iterable[str]):
    return [normalize_text(str(item)) for item in items]


def normalize_observation(observation: dict) -> dict:
    out = dict(observation)
    for key in (
        "visible_text",
        "meta_title",
        "meta_description",
        "raw_html",
    ):
        val = out.get(key)
        if isinstance(val, str):
            out[key] = normalize_text(val)
    for key in (
        "hidden_text_samples",
        "html_comment_samples",
        "links",
    ):
        val = out.get(key)
        if isinstance(val, list):
            out[key] = normalize_strings(val)
    link_options = out.get("link_options")
    if isinstance(link_options, list):
        out["link_options"] = [
            {
                "text": normalize_text(str(item.get("text", ""))),
                "href": normalize_text(str(item.get("href", ""))),
            }
            for item in link_options
            if isinstance(item, dict)
        ]
    return out
