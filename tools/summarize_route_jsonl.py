#!/usr/bin/env python3
"""Aggregate safe JSONL timing fields. Never prints content or secrets."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(statistics.median(vals))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_route_jsonl.py DIR")
        return 2
    directory = Path(sys.argv[1])
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and "duration_ms" in rec:
                records.append(rec)
    if not records:
        print("records=0")
        return 0
    routes = {}
    errors = 0
    for rec in records:
        routes[str(rec.get("route") or "?")] = routes.get(str(rec.get("route") or "?"), 0) + 1
        if rec.get("error_kind") or rec.get("outcome") not in (None, "complete"):
            errors += 1

    def col(name: str) -> list[float]:
        out = []
        for rec in records:
            value = rec.get(name)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    def show(label: str, vals: list[float]) -> None:
        if not vals:
            print(f"{label}=n/a")
            return
        print(
            f"{label}_n={len(vals)} min={min(vals):.0f} median={median(vals):.0f} max={max(vals):.0f}"
        )

    print(f"records={len(records)} errors={errors} routes={routes}")
    show("upstream_headers_ms", col("upstream_headers_ms"))
    show("first_sse_event_ms", col("first_sse_event_ms"))
    show("first_semantic_ms", col("first_semantic_ms"))
    show("ttft_ms", col("ttft_ms"))
    show("duration_ms", col("duration_ms"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
