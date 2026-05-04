# Shared types for the defence pipeline.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class SanitizeMeta:
    sanitized: bool = False
    marker_count: int = 0
    filtered_link_count: int = 0
    layers_fired: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sanitized": self.sanitized,
            "marker_count": self.marker_count,
            "filtered_link_count": self.filtered_link_count,
            "layers_fired": list(self.layers_fired),
            **self.details,
        }


def merge_meta(base: Dict[str, Any], extra: SanitizeMeta) -> Dict[str, Any]:
    out = dict(base)
    out["sanitized"] = bool(out.get("sanitized")) or extra.sanitized
    out["marker_count"] = int(out.get("marker_count", 0)) + extra.marker_count
    out["filtered_link_count"] = (
        int(out.get("filtered_link_count", 0)) + extra.filtered_link_count
    )
    fired = list(out.get("layers_fired", []))
    fired.extend(extra.layers_fired)
    out["layers_fired"] = fired
    for k, v in extra.details.items():
        out.setdefault(k, v)
    return out
