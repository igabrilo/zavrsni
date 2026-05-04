# Re-exposes the existing v1 PromptInjectionDefender via the make_defender factory.

from __future__ import annotations

from src.phase1.colab_agent import PromptInjectionDefender


class V1Defender(PromptInjectionDefender):
    # Identical to v1; exists so DEFENSE_VERSION=v1 works through the factory.

    pass
