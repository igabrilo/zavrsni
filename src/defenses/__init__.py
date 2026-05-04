# Pluggable defence pipeline for the web-navigation agent.

from __future__ import annotations

import os
from typing import Optional, Set


def _parse_layers_off() -> Set[str]:
    raw = os.getenv("DEFENSE_LAYERS_OFF", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def make_defender(version: str = "v1", confidence_threshold: float = 0.65):
    version = (version or "v1").lower()
    if version == "v2":
        from .composite import CompositeDefenderV2

        disabled = _parse_layers_off()
        return CompositeDefenderV2(
            confidence_threshold=confidence_threshold,
            disabled_layers=disabled,
        )

    from .v1_adapter import V1Defender

    return V1Defender(confidence_threshold=confidence_threshold)
