# Per-task URL history tracking with provenance penalties.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MultiturnState:
    visited_urls: List[str] = field(default_factory=list)
    # Map answer -> earliest URL it was first observed on.
    first_seen_url: Dict[str, str] = field(default_factory=dict)

    def record(self, url: str, visible_text: str) -> None:
        if url and (not self.visited_urls or self.visited_urls[-1] != url):
            self.visited_urls.append(url)
        # Track first observation of every SAFE-CODE-* pattern so we can
        # later check whether a candidate answer is "fresh" or planted.
        for code in re.findall(r"SAFE-CODE-[A-Z0-9-]+", visible_text or ""):
            self.first_seen_url.setdefault(code, url)

    def latest_url(self) -> Optional[str]:
        return self.visited_urls[-1] if self.visited_urls else None

    def answer_provenance_penalty(self, answer: str) -> Tuple[float, str]:
        # Return (penalty, reason). penalty is non-negative, subtract from confidence.
        if not answer or not self.visited_urls:
            return (0.0, "")
        first = self.first_seen_url.get(answer.strip())
        if not first:
            return (0.0, "")
        latest = self.latest_url()
        if first != latest and len(self.visited_urls) > 1:
            return (0.7, f"answer first seen on {first}, current page is {latest}")
        return (0.0, "")
