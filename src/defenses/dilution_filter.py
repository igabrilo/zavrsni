# Filter out long low-information dilution padding from visible_text.

# Attackers pad pages with Lorem-ipsum text close to the LLM's context limit
# and place the injection at the tail (recency bias). The filter detects
# high-padding pages (large size + low keyword density) and truncates the
# visible_text to a structurally relevant region — paragraphs containing the
# goal keyword like "Benign target value:" or any SAFE-CODE-* token.

from __future__ import annotations

import re
from typing import Optional, Tuple

# Heuristic threshold: if visible_text exceeds this many characters, consider truncation.
_LARGE_TEXT_THRESHOLD = 1500
# Density: ratio of SAFE-CODE occurrences to total text length needed to be "relevant".
_MIN_KEYWORD_DENSITY = 1.0 / 800.0  # at least 1 SAFE-CODE token per 800 chars


def _density(text: str) -> float:
    if not text:
        return 0.0
    matches = len(re.findall(r"SAFE-CODE-[A-Z0-9-]+", text))
    return matches / max(1, len(text))


def filter_dilution(visible_text: str) -> Tuple[str, bool]:
    # Return (filtered_text, fired). When fired, only paragraphs with the
    # goal-relevant keyword survive. Otherwise text is returned unchanged.
    if not visible_text or len(visible_text) <= _LARGE_TEXT_THRESHOLD:
        return (visible_text, False)
    if _density(visible_text) >= _MIN_KEYWORD_DENSITY:
        return (visible_text, False)

    paragraphs = re.split(r"\n+|\.\s+", visible_text)
    relevant = [
        p.strip()
        for p in paragraphs
        if "benign target value" in p.lower() or re.search(r"SAFE-CODE-[A-Z0-9-]+", p)
    ]
    if not relevant:
        return (visible_text, False)
    return (" ".join(relevant)[:_LARGE_TEXT_THRESHOLD], True)
