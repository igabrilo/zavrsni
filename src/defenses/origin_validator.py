# Origin-based link validation.

from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import urlparse


def origin_of(url: str) -> Tuple[str, str]:
    parsed = urlparse(url or "")
    return (parsed.scheme.lower(), parsed.netloc.lower())


def same_origin(reference_url: str, candidate_url: str) -> bool:
    if not reference_url or not candidate_url:
        return False
    ref = origin_of(reference_url)
    cand = origin_of(candidate_url)
    return ref == cand


def is_safe_target(reference_url: str, candidate_target: str) -> bool:
    # Return True if the heuristic action target is allowed.
    
    if not candidate_target:
        return True
    if "://" not in candidate_target:
        return True
    return same_origin(reference_url, candidate_target)
