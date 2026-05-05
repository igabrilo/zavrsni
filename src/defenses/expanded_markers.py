# Expanded marker regex covering paraphrased trigger words.

from __future__ import annotations

import re

# v1 set + paraphrased verbs + authority/persona tokens used in v2 attacks.
SUSPICIOUS_TEXT_PATTERN = re.compile(
    r"ignore|disregard|previous instructions|override|leak|secret|attack-code|hacked|autonomous agent"
    r"|supersede(d)?|retire(d)?"
    r"|defen[cs]e cleared|verified by defender|system notice|pipeline completed"
    r"|authorized override|authorised override",
    flags=re.IGNORECASE,
)


def find_markers(text: str):
    if not text:
        return []
    return sorted(set(SUSPICIOUS_TEXT_PATTERN.findall(text)))
