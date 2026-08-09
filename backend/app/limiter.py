"""Lightweight in-memory rate limiter implemented as a FastAPI dependency.

Uses a fixed-window counter keyed by client IP + scope.  This avoids external
dependencies and the signature-introspection issues that decorator-based
limiters cause under ``from __future__ import annotations``.  For multi-process
production deployments, back this with Redis instead.
"""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, status

from .config import settings


class _FixedWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: float) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, [])
            cutoff = now - window_seconds
            # Drop timestamps outside the current window.
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= limit:
                retry_after = int(window_seconds - (now - bucket[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please slow down.",
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)


_limiter = _FixedWindowLimiter()

_UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def _parse_rate(rate: str) -> tuple[int, float]:
    """Parse a rate string like ``"20/minute"`` into (limit, window_seconds)."""
    try:
        count, unit = rate.split("/")
        return int(count), float(_UNITS[unit.strip().rstrip("s")])
    except Exception:
        return 60, 60.0


def rate_limit(scope: str, rate: str):
    """Return a dependency enforcing ``rate`` for the given scope."""
    limit, window = _parse_rate(rate)

    def dependency(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        _limiter.check(f"{scope}:{client_ip}", limit, window)

    return dependency


def auth_rate_limit():
    return rate_limit("auth", settings.rate_limit_auth)


def ai_rate_limit():
    return rate_limit("ai", settings.rate_limit_ai)
