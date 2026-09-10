#!/usr/bin/env python3
"""Secret-safe audit of cline-proxy runtime evidence.

Reads JSONL request summaries and a redirected console log. Emits only
aggregates, field names, counts, timings, and coarse fingerprints.
Never prints request content, reasoning, tool arguments, Authorization
values, API keys, cookies, or raw session ids.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SAFE_STRING_FIELDS = {
    "protocol",
    "requested_model",
    "upstream_model",
    "model_family",
    "session",
    "upstream_strategy",
    "selected_key_name",
    "reasoning_effort",
    "thinking_exposure",
    "response_shape",
    "outcome",
    "error_kind",
    "count_method",
    "message",
}

CSI_RE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
ANSI_OSC_RE = re.compile(rb"\x1b\].*?(?:\x07|\x1b\\)")
WRITER_CLOSED_RE = re.compile(rb"writer closed", re.I)
SECRETISH_RE = re.compile(
    rb"(?i)(authorization\s*[:=]|bearer\s+[A-Za-z0-9._+\-/=]{8,}|api[_-]?key\s*[:=]|sk-[A-Za-z0-9]{8,})"
)

INFO_NOISE = (
    b"stable prefix telemetry",
    b"request optimization",
    b"exact GLM token accounting",
    b"client request accepted",
    b"OpenAI upstream attempt selected",
    b"OpenAI stream closed",
    b"token count completed",
)


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def bucket_tokens(n: int | None) -> str:
    if n is None:
        return "unknown"
    edges = [
        (8_192, "<8k"),
        (32_768, "8-32k"),
        (65_536, "32-64k"),
        (131_072, "64-128k"),
        (196_608, "128-192k"),
        (229_376, "192-224k"),
        (262_144, "224-256k"),
        (393_216, "256-384k"),
        (524_288, "384-512k"),
        (1_048_576, "512k-1M"),
    ]
    for edge, label in edges:
        if n < edge:
            return label
    return ">1M"


def scan_console(path: Path) -> dict:
    data = path.read_bytes()
    csi = len(CSI_RE.findall(data))
    osc = len(ANSI_OSC_RE.findall(data))
    writer_closed = len(WRITER_CLOSED_RE.findall(data))
    secretish = len(SECRETISH_RE.findall(data))
    noise = {label.decode(): data.count(label) for label in INFO_NOISE}
    info_lines = data.count(b" INFO ") + data.count(b"INFO ")
    warn_lines = data.count(b" WARN ") + data.count(b"WARN ")
    error_lines = data.count(b" ERROR ") + data.count(b"ERROR ")
    # Compact summary lines from obs.rs (checkmark / ballot x).
    summary_ok = data.count("\u2713".encode("utf-8"))
    summary_err = data.count("\u2717".encode("utf-8"))
    return {
        "bytes": len(data),
        "csi_sequences": csi,
        "osc_sequences": osc,
        "has_ansi": bool(csi or osc),
        "writer_closed": writer_closed,
        "secretish_pattern_hits": secretish,
        "info_lines_approx": info_lines,
        "warn_lines_approx": warn_lines,
        "error_lines_approx": error_lines,
        "compact_ok_marks": summary_ok,
        "compact_err_marks": summary_err,
        "lifecycle_noise": noise,
    }


def scan_jsonl(path: Path) -> dict:
    records = []
    parse_errors = 0
    field_names: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(rec, dict):
            parse_errors += 1
            continue
        field_names.update(rec.keys())
        records.append(rec)

    outcomes = Counter()
    errors = Counter()
    keys = Counter()
    strategies = Counter()
    protocols = Counter()
    input_buckets = Counter()
    budget_buckets = Counter()
    durations: list[float] = []
    ttfTs: list[float] = []
    first_events: list[float] = []
    prompt_tokens: list[int] = []
    localish_tokens: list[int] = []
    cache_ratios: list[float] = []
    slow: list[dict] = []
    errors_detail: list[dict] = []

    content_leaks = 0
    leak_fields: Counter[str] = Counter()
    forbidden = (
        "prompt",
        "messages",
        "content",
        "reasoning",
        "tool_result",
        "arguments",
        "api_key",
        "authorization",
        "cookie",
    )

    for rec in records:
        for key in rec:
            lk = key.lower()
            if any(part in lk for part in forbidden) and key not in {
                "prompt_tokens",
                "reasoning_tokens",
                "reasoning_effort",
                "reasoning_ratio",
                "canonicalized_arguments",
                "historical_reasoning_bytes_removed",
                "first_reasoning_ms",
            }:
                content_leaks += 1
                leak_fields[key] += 1
        outcome = rec.get("outcome") or rec.get("message") or "unknown"
        outcomes[str(outcome)] += 1
        if rec.get("error_kind"):
            errors[str(rec["error_kind"])] += 1
        if rec.get("selected_key_name"):
            keys[str(rec["selected_key_name"])] += 1
        if rec.get("upstream_strategy"):
            strategies[str(rec["upstream_strategy"])] += 1
        if rec.get("protocol"):
            protocols[str(rec["protocol"])] += 1
        dur = rec.get("duration_ms")
        if isinstance(dur, (int, float)):
            durations.append(float(dur))
        ttft = rec.get("ttft_ms")
        if isinstance(ttft, (int, float)):
            ttfTs.append(float(ttft))
        fe = rec.get("upstream_first_event_ms")
        if isinstance(fe, (int, float)):
            first_events.append(float(fe))
        pt = rec.get("prompt_tokens")
        if isinstance(pt, int):
            prompt_tokens.append(pt)
            input_buckets[bucket_tokens(pt)] += 1
            max_out = rec.get("effective_max_tokens")
            if isinstance(max_out, int):
                budget_buckets[bucket_tokens(pt + max_out)] += 1
        # Local exact count may currently live only in console INFO, not JSONL.
        for alt in ("local_input_tokens", "input_tokens", "exact_input_tokens"):
            val = rec.get(alt)
            if isinstance(val, int):
                localish_tokens.append(val)
                break
        ratio = rec.get("cache_hit_ratio")
        if isinstance(ratio, (int, float)):
            cache_ratios.append(float(ratio))
        interesting = False
        if rec.get("error_kind") or rec.get("outcome") not in (None, "complete"):
            interesting = True
        if isinstance(dur, (int, float)) and dur >= 60_000:
            interesting = True
        if interesting:
            row = {
                "outcome": rec.get("outcome"),
                "error_kind": rec.get("error_kind"),
                "duration_ms": rec.get("duration_ms"),
                "ttft_ms": rec.get("ttft_ms"),
                "upstream_first_event_ms": rec.get("upstream_first_event_ms"),
                "first_reasoning_ms": rec.get("first_reasoning_ms"),
                "first_text_ms": rec.get("first_text_ms"),
                "first_tool_call_ms": rec.get("first_tool_call_ms"),
                "prompt_tokens": rec.get("prompt_tokens"),
                "cached_tokens": rec.get("cached_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
                "effective_max_tokens": rec.get("effective_max_tokens"),
                "request_bytes": rec.get("request_bytes"),
                "upstream_strategy": rec.get("upstream_strategy"),
                "downstream_stream": rec.get("downstream_stream"),
                "selected_key_name": rec.get("selected_key_name"),
                "failover_count": rec.get("failover_count"),
                "cache_hit_ratio": rec.get("cache_hit_ratio"),
                "has_flight": bool(rec.get("flight")),
            }
            if rec.get("error_kind") or rec.get("outcome") not in (None, "complete"):
                errors_detail.append(row)
            else:
                slow.append(row)

    durations.sort()
    ttfTs.sort()
    first_events.sort()
    prompt_tokens.sort()

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "min": vals[0],
            "p50": percentile(vals, 0.50),
            "p95": percentile(vals, 0.95),
            "max": vals[-1],
            "mean": round(statistics.fmean(vals), 1),
        }

    return {
        "records": len(records),
        "parse_errors": parse_errors,
        "field_names": dict(field_names),
        "outcomes": dict(outcomes),
        "error_kinds": dict(errors),
        "keys": dict(keys),
        "strategies": dict(strategies),
        "protocols": dict(protocols),
        "input_token_buckets": dict(input_buckets),
        "budget_buckets": dict(budget_buckets),
        "duration_ms": stats(durations),
        "ttft_ms": stats(ttfTs),
        "upstream_first_event_ms": stats(first_events),
        "prompt_tokens": stats([float(x) for x in prompt_tokens]),
        "localish_token_fields_present": len(localish_tokens),
        "cache_hit_ratio": stats(cache_ratios),
        "content_like_field_hits": content_leaks,
        "content_like_fields": dict(leak_fields),
        "slow_complete_requests": slow,
        "error_or_noncomplete": errors_detail,
    }


def sanitize_runtime_state(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"type": type(data).__name__}
    out: dict = {"top_level_keys": sorted(data.keys())}
    if "schema_version" in data:
        out["schema_version"] = data["schema_version"]
    if "active_key_name" in data:
        out["active_key_name"] = data["active_key_name"]
    keys = data.get("keys")
    if isinstance(keys, dict):
        out["key_count"] = len(keys)
        states = Counter()
        for meta in keys.values():
            if isinstance(meta, dict):
                states[str(meta.get("state") or meta.get("status") or "unknown")] += 1
        out["key_states"] = dict(states)
        out["key_names"] = sorted(str(name) for name in keys)
    return out


def sanitize_config_structure(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"error": "config is not an object"}

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for k, v in value.items():
                path_k = f"{prefix}.{k}" if prefix else k
                lk = k.lower()
                secret = any(
                    token in lk
                    for token in ("api_key", "authorization", "secret", "token", "cookie", "password")
                )
                if secret and isinstance(v, str):
                    yield path_k, {"kind": "secret_string", "len": len(v), "empty": v == ""}
                else:
                    yield from walk(v, path_k)
        elif isinstance(value, list):
            yield prefix, {"kind": "array", "len": len(value)}
            for i, item in enumerate(value):
                yield from walk(item, f"{prefix}[{i}]")
        elif isinstance(value, bool):
            yield prefix, {"kind": "bool", "value": value}
        elif isinstance(value, (int, float)):
            yield prefix, {"kind": "number", "value": value}
        elif isinstance(value, str):
            yield prefix, {"kind": "string", "len": len(value)}
        elif value is None:
            yield prefix, {"kind": "null"}
        else:
            yield prefix, {"kind": type(value).__name__}

    fields = dict(walk(data))
    keys = data.get("cline_api_keys")
    key_info = []
    if isinstance(keys, list):
        for item in keys:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            enabled = item.get("enabled")
            api_key = item.get("api_key")
            key_info.append(
                {
                    "name": name if isinstance(name, str) else None,
                    "enabled": enabled if isinstance(enabled, bool) else None,
                    "api_key_len": len(api_key) if isinstance(api_key, str) else None,
                }
            )
    return {
        "top_level_keys": sorted(data.keys()),
        "fields": fields,
        "cline_keys": key_info,
        "enabled_key_count": sum(1 for k in key_info if k.get("enabled")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-config-structure", action="store_true")
    args = parser.parse_args()
    runtime = Path(args.runtime_dir)
    report: dict = {"runtime_dir": str(runtime)}

    jsonl_dir = runtime / "logs"
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl")) if jsonl_dir.is_dir() else []
    report["jsonl_files"] = [{"name": p.name, "bytes": p.stat().st_size} for p in jsonl_files]
    merged = {
        "records": 0,
        "parse_errors": 0,
        "field_names": Counter(),
        "outcomes": Counter(),
        "error_kinds": Counter(),
        "keys": Counter(),
        "strategies": Counter(),
        "protocols": Counter(),
        "input_token_buckets": Counter(),
        "budget_buckets": Counter(),
        "slow_complete_requests": [],
        "error_or_noncomplete": [],
        "content_like_field_hits": 0,
        "content_like_fields": Counter(),
        "localish_token_fields_present": 0,
    }
    duration_parts = []
    ttft_parts = []
    first_event_parts = []
    prompt_parts = []
    cache_parts = []
    for path in jsonl_files:
        part = scan_jsonl(path)
        merged["records"] += part["records"]
        merged["parse_errors"] += part["parse_errors"]
        merged["field_names"].update(part["field_names"])
        merged["outcomes"].update(part["outcomes"])
        merged["error_kinds"].update(part["error_kinds"])
        merged["keys"].update(part["keys"])
        merged["strategies"].update(part["strategies"])
        merged["protocols"].update(part["protocols"])
        merged["input_token_buckets"].update(part["input_token_buckets"])
        merged["budget_buckets"].update(part["budget_buckets"])
        merged["slow_complete_requests"].extend(part["slow_complete_requests"])
        merged["error_or_noncomplete"].extend(part["error_or_noncomplete"])
        merged["content_like_field_hits"] += part["content_like_field_hits"]
        merged["content_like_fields"].update(part["content_like_fields"])
        merged["localish_token_fields_present"] += part["localish_token_fields_present"]
        if part["duration_ms"].get("n"):
            duration_parts.append(part["duration_ms"])
        if part["ttft_ms"].get("n"):
            ttft_parts.append(part["ttft_ms"])
        if part["upstream_first_event_ms"].get("n"):
            first_event_parts.append(part["upstream_first_event_ms"])
        if part["prompt_tokens"].get("n"):
            prompt_parts.append(part["prompt_tokens"])
        if part["cache_hit_ratio"].get("n"):
            cache_parts.append(part["cache_hit_ratio"])
        # Keep per-file min/max by reusing last file stats if only one file.
        merged["duration_ms"] = part["duration_ms"]
        merged["ttft_ms"] = part["ttft_ms"]
        merged["upstream_first_event_ms"] = part["upstream_first_event_ms"]
        merged["prompt_tokens"] = part["prompt_tokens"]
        merged["cache_hit_ratio"] = part["cache_hit_ratio"]

    for key in (
        "field_names",
        "outcomes",
        "error_kinds",
        "keys",
        "strategies",
        "protocols",
        "input_token_buckets",
        "budget_buckets",
        "content_like_fields",
    ):
        merged[key] = dict(merged[key])
    report["jsonl"] = merged

    proxy_log = runtime / "proxy.log"
    if proxy_log.exists():
        report["proxy_log"] = scan_console(proxy_log)
    else:
        report["proxy_log"] = None

    state = runtime / "runtime-state.json"
    if state.exists():
        report["runtime_state"] = sanitize_runtime_state(state)
    else:
        report["runtime_state"] = None

    if args.include_config_structure:
        cfg = runtime / "config.json"
        if cfg.exists():
            report["config_structure"] = sanitize_config_structure(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
