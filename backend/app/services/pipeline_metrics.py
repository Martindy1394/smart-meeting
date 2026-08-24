"""Structured per-stage latency / error metrics for ASR, translate, summarize."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("smart_meeting.metrics")

_lock = threading.Lock()
_counts: dict[str, int] = defaultdict(int)
_errors: dict[str, int] = defaultdict(int)
_latency_ms: dict[str, list[float]] = defaultdict(list)
_MAX_SAMPLES = 200


def record(stage: str, *, latency_ms: float, ok: bool = True, detail: str = "") -> None:
    stage = (stage or "unknown").strip().lower()
    with _lock:
        _counts[stage] += 1
        if not ok:
            _errors[stage] += 1
        bucket = _latency_ms[stage]
        bucket.append(float(latency_ms))
        if len(bucket) > _MAX_SAMPLES:
            del bucket[: len(bucket) - _MAX_SAMPLES]
    level = logging.INFO if ok else logging.WARNING
    logger.log(
        level,
        "stage=%s ok=%s latency_ms=%.1f%s",
        stage,
        ok,
        latency_ms,
        f" detail={detail}" if detail else "",
    )


@contextmanager
def track(stage: str, *, detail: str = "") -> Iterator[None]:
    t0 = time.perf_counter()
    ok = True
    err = ""
    try:
        yield
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        record(
            stage,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            ok=ok,
            detail=detail or err,
        )


def snapshot() -> dict:
    with _lock:
        out = {}
        for stage, n in _counts.items():
            samples = list(_latency_ms.get(stage) or [])
            avg = sum(samples) / len(samples) if samples else 0.0
            p95 = sorted(samples)[int(0.95 * (len(samples) - 1))] if samples else 0.0
            out[stage] = {
                "count": n,
                "errors": int(_errors.get(stage) or 0),
                "avg_latency_ms": round(avg, 1),
                "p95_latency_ms": round(p95, 1),
                "queue_depth": 0,
            }
        return out


def set_queue_depth(stage: str, depth: int) -> None:
    # Soft annotation — depth is read via jobs module into health.
    with _lock:
        key = f"{stage}__queue"
        _counts[key] = int(depth)
