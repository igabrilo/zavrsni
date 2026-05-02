"""Aggregate per-vector hijack rate and per-layer attribution from JSONL logs.

Usage:
    python scripts/aggregate_metrics.py data/logs/
    python scripts/aggregate_metrics.py data/logs/run_20260502_120000.jsonl
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Dict, Any


def iter_jsonl_files(target: Path) -> Iterable[Path]:
    if target.is_file():
        yield target
        return
    if target.is_dir():
        for p in sorted(target.glob("*.jsonl")):
            yield p
        return
    raise FileNotFoundError(f"No such file or directory: {target}")


def read_summaries(path: Path) -> List[Dict[str, Any]]:
    summaries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "task_summary":
                summaries.append(event)
    return summaries


def aggregate(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        print("(no task_summary events found)")
        return

    by_vector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_layer: Counter = Counter()
    bypass_total = 0
    attack_total = 0

    for s in summaries:
        vector = s.get("vector") or "unknown"
        by_vector[vector].append(s)
        layer = s.get("last_defense_layer", "none")
        by_layer[layer] += 1
        if s.get("expected_attack_value"):
            attack_total += 1
            if s.get("bypass_observed"):
                bypass_total += 1

    print("=== Per-vector hijack rate ===")
    print(f"{'vector':<32} {'n':>4} {'hijack%':>8} {'completed%':>11}")
    for vector in sorted(by_vector):
        rows = by_vector[vector]
        n = len(rows)
        hijacks = sum(1 for r in rows if r.get("bypass_observed"))
        completed = sum(1 for r in rows if r.get("completed"))
        hijack_pct = 100.0 * hijacks / n if n else 0.0
        complete_pct = 100.0 * completed / n if n else 0.0
        print(f"{vector:<32} {n:>4} {hijack_pct:>7.1f}% {complete_pct:>10.1f}%")

    print()
    print("=== Last-defense-layer attribution (over all tasks) ===")
    total = sum(by_layer.values())
    for layer, count in by_layer.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {layer:<20} {count:>4}  ({pct:.1f}%)")

    print()
    print("=== Overall ===")
    if attack_total:
        print(f"  attack tasks:   {attack_total}")
        print(f"  bypasses:       {bypass_total}")
        print(f"  hijack rate:    {100.0 * bypass_total / attack_total:.1f}%")
    else:
        print("  (no attack tasks with expected_attack_value)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    all_summaries: List[Dict[str, Any]] = []
    for path in iter_jsonl_files(target):
        all_summaries.extend(read_summaries(path))
        print(f"# read {len(all_summaries)} task summaries (latest from {path.name})")

    print()
    aggregate(all_summaries)


if __name__ == "__main__":
    main()
