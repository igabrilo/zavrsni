# Pulls payload-bearing text out of channels v1 ignores.

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

EXTRA_CHANNELS_KEY = "extra_channels"

# Selectors and their JavaScript extraction snippets are kept here so the
# Playwright caller can build the full extraction script without duplicating logic.
EXTRACTION_JS = r"""
() => {
  const out = [];
  const push = (label, value) => {
    const text = (value || "").trim();
    if (text) out.push(label + ": " + text.slice(0, 500));
  };
  document.querySelectorAll('svg title, svg desc').forEach(el => push('svg', el.textContent || ''));
  document.querySelectorAll('[aria-label]').forEach(el => push('aria-label', el.getAttribute('aria-label') || ''));
  document.querySelectorAll('[aria-describedby]').forEach(el => push('aria-describedby', el.getAttribute('aria-describedby') || ''));
  document.querySelectorAll('img[alt]').forEach(el => push('alt', el.getAttribute('alt') || ''));
  document.querySelectorAll('meta').forEach(el => {
    const name = el.getAttribute('name') || el.getAttribute('property') || '';
    if (!name || name === 'description') return;
    push('meta:' + name, el.getAttribute('content') || '');
  });
  document.querySelectorAll('script[type="application/ld+json"]').forEach(el => push('jsonld', el.textContent || ''));
  push('title', document.title || '');
  document.querySelectorAll('noscript').forEach(el => push('noscript', el.textContent || ''));
  document.querySelectorAll('details').forEach(el => push('details', el.textContent || ''));
  return out.slice(0, 40);
}
"""


def collect_pool(observation: Dict[str, Any]) -> str:
    items = observation.get(EXTRA_CHANNELS_KEY) or []
    return " ".join(str(item) for item in items)


def blank_channels(observation: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(observation)
    out[EXTRA_CHANNELS_KEY] = []
    return out


def scan_for_safe_codes(observation: Dict[str, Any]) -> List[str]:
    # Return all SAFE-CODE-* patterns appearing in the extra channels.
    pool = collect_pool(observation)
    return re.findall(r"SAFE-CODE-[A-Z0-9-]+", pool)
