#!/usr/bin/env python3
"""Structurally set upstream.proxy on a config file without printing secrets.

Usage:
  python tools/set_upstream_proxy.py --config PATH --proxy socks5://127.0.0.1:10888
  python tools/set_upstream_proxy.py --config PATH --proxy direct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--backup", default="")
    args = parser.parse_args()
    path = Path(args.config)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "upstream" not in data:
        print("FAILED: config missing upstream object", file=sys.stderr)
        return 1
    upstream = data["upstream"]
    if not isinstance(upstream, dict):
        print("FAILED: upstream is not an object", file=sys.stderr)
        return 1
    if args.backup:
        backup = Path(args.backup)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
        print(f"backup_bytes={backup.stat().st_size}")
    value = args.proxy.strip()
    if value.lower() in {"", "direct", "none", "null"}:
        upstream["proxy"] = None
        kind = "direct"
    else:
        upstream["proxy"] = value
        kind = value.split("://", 1)[0]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"updated_proxy_kind={kind}")
    print(f"upstream_keys={len(upstream)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
