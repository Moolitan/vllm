"""Opt-in JSONL timestamps for end-to-end request latency experiments.

The helper is deliberately generic: it does not know about Skill objects or
CSKCache internals.  A caller opts in by setting ``VLLM_REQUEST_TIMELINE_PATH``
and by assigning a request ID containing ``cskcache-latency-``.  Production
requests therefore pay only one environment/substring check.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


_TRACE_PATH = os.getenv("VLLM_REQUEST_TIMELINE_PATH")
_REQUEST_MARKER = "cskcache-latency-"


def _boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return "unavailable"


_BOOT_ID = _boot_id()


def record_request_timeline(event: str, request_id: str, **fields: Any) -> None:
    """Append one timing record for an explicitly tagged benchmark request."""

    if not _TRACE_PATH or _REQUEST_MARKER not in request_id:
        return
    payload = {
        "event": event,
        "request_id": request_id,
        "boot_id": _BOOT_ID,
        "monotonic_ns": time.monotonic_ns(),
        "unix_ns": time.time_ns(),
        "pid": os.getpid(),
        **fields,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        _TRACE_PATH,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)
